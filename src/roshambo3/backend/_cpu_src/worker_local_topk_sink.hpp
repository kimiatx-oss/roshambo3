#pragma once

#include "combo_executor.hpp"
#include "worker_local_topk.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace r3 {
namespace cpu {

/// Refuses a scored row whose self-overlaps cannot describe its pair. Every
/// row a worker-local top-K collects passes here -- the engine's worker route
/// and `score_pairs(worker_local_topk=...)` alike -- including the ones a
/// non-positive Tanimoto denominator reports as 0 and so ranks last, which no
/// check of the returned top-K could see. Gaussian overlaps obey
/// V_AB <= sqrt(V_AA V_BB) <= (V_AA + V_BB) / 2; 2 V_AB > V_AA + V_BB is
/// exactly a shape Tanimoto above 1 and holds wherever that denominator is not
/// positive. `margin` is the relative margin for the route these rows' self-
/// overlaps took; it is chosen, and defined, only in Python
/// (roshambo3.scores.SELF_OVERLAP_MARGIN_POLICY) and passed
/// in per call or per block -- there is no copy of its value here. `row` names
/// the library row or conformer ID.
inline void check_self_overlap_bound(const float* scores,
                                     std::size_t n_score_columns,
                                     std::size_t query_row, const char* what,
                                     long long row, double margin) {
    constexpr std::size_t kOverlapVolume = 3, kSelfOverlapQuery = 5,
                          kSelfOverlapLibrary = 6;
    if (n_score_columns <= kSelfOverlapLibrary) return;
    const double overlap = scores[kOverlapVolume];
    const double selves =
        double(scores[kSelfOverlapQuery]) + double(scores[kSelfOverlapLibrary]);
    if (!(2.0 * overlap > selves * (1.0 + margin))) return;
    char message[512];
    std::snprintf(
        message, sizeof message,
        "%s %lld scored against query row %zu reports pair overlap %.9g above "
        "the mean of its self-overlaps %.9g (query) and %.9g (library), which "
        "consistent inputs cannot produce. The self-overlaps do not describe "
        "the scored coordinates -- usually a stored self-overlap sidecar from "
        "a different preparation; self_overlap_policy='recompute' "
        "scores without it.",
        what, row, query_row, double(scores[kOverlapVolume]),
        double(scores[kSelfOverlapQuery]), double(scores[kSelfOverlapLibrary]));
    throw std::domain_error(message);
}

/// Adapter from the combo pool's native completed-row observer to the collector.
///
/// The caller passes a non-null `id_maps_owner` token that strongly pins the
/// immutable backing allocations for `conformer_ids` and optional
/// `molecule_ids`. The token is retained through finalization/discard; the
/// arrays are borrowed views and are neither copied nor budgeted here. Any
/// array copies made by the caller remain its preparation-memory responsibility.
/// The caller must call `mark_scoring_succeeded()` only after the executor's
/// full scoring call returns successfully. If scoring throws/cancels, call
/// `discard_after_failure()` after worker join; this destroys any partial
/// local state and prevents those rows from being published as a result.
class WorkerLocalTopKRowSink final : public ComboRowSink {
public:
    using Result = std::vector<std::vector<WorkerTopKHit>>;

    WorkerLocalTopKRowSink(
        std::size_t worker_count, std::size_t query_count, std::size_t k,
        std::size_t rank_column, WorkerTopKLevel level,
        std::size_t library_row_count,
        std::shared_ptr<const void> id_maps_owner,
        const std::int64_t* conformer_ids,
        const std::int64_t* molecule_ids = nullptr,
        std::size_t state_budget_bytes = kWorkerTopKDefaultStateBudgetBytes,
        std::size_t additional_peak_bytes = 0,
        std::size_t id_map_preparation_peak_bytes = 0,
        std::size_t final_row_scratch_bytes = 0,
        std::size_t python_result_bytes = 0, bool include_molecule_ids = false,
        bool direct_ids = false,
        double self_overlap_bound_margin =
            std::numeric_limits<double>::quiet_NaN())
        : id_maps_owner_(std::move(id_maps_owner)),
          has_molecule_ids_(include_molecule_ids || molecule_ids != nullptr),
          id_map_preparation_peak_bytes_(id_map_preparation_peak_bytes),
          final_row_scratch_bytes_(final_row_scratch_bytes),
          python_result_bytes_(python_result_bytes),
          collector_(make_collector(
              id_maps_owner_, worker_count, query_count, k, rank_column, level,
              library_row_count, conformer_ids, molecule_ids,
              state_budget_bytes, additional_peak_bytes, direct_ids)),
          direct_ids_(direct_ids), query_count_(query_count),
          self_overlap_bound_margin_(self_overlap_bound_margin) {}

    void on_row(std::size_t worker_id, std::size_t output_row,
                std::size_t query_row, std::size_t library_row,
                const float* scores, std::size_t n_score_columns) override {
        if (state_ != State::Collecting || !collector_) {
            throw std::logic_error(
                "worker-local top-K row arrived after sink completion/discard");
        }
        // The mapped route's margin is the collector's, given at construction;
        // a sink built without one (a direct-ID collector, whose rows arrive
        // through on_row_ids) refuses rather than check against no margin.
        if (!std::isfinite(self_overlap_bound_margin_)) {
            throw std::logic_error(
                "worker-local top-K sink has no self-overlap bound margin");
        }
        check_self_overlap_bound(scores, n_score_columns, query_row,
                                 "library row", (long long)library_row,
                                 self_overlap_bound_margin_);
        collector_->on_row(worker_id, output_row, query_row, library_row,
                           scores, n_score_columns);
    }

    /// `self_overlap_bound_margin` is the margin of the route the row's block
    /// took (WorkerLocalTopKBlockRowSink holds one per block).
    void on_row_ids(std::size_t worker_id, std::size_t query_row,
                    std::int64_t conformer_id, std::int64_t molecule_id,
                    const float* scores, std::size_t n_score_columns,
                    double self_overlap_bound_margin) {
        if (!direct_ids_ || state_ != State::Collecting || !collector_) {
            throw std::logic_error(
                "direct-ID row arrived outside a collecting direct-ID sink");
        }
        check_self_overlap_bound(scores, n_score_columns, query_row,
                                 "conformer", (long long)conformer_id,
                                 self_overlap_bound_margin);
        collector_->on_row_ids(worker_id, query_row, conformer_id, molecule_id,
                               has_molecule_ids_, scores, n_score_columns);
    }

    /// Must be called only after the executor call returned without error.
    void mark_scoring_succeeded() {
        if (state_ != State::Collecting || !collector_) {
            throw std::logic_error(
                "worker-local top-K sink is not collecting rows");
        }
        state_ = State::Scored;
    }

    /// Discard partial rows after an exception/cancellation, after worker join.
    void discard_after_failure() noexcept {
        state_ = State::Discarded;
        collector_.reset();
    }

    /// Refuses to expose partial callbacks unless scoring completed normally.
    Result finalize() const {
        if (state_ != State::Scored || !collector_) {
            throw std::logic_error(
                "worker-local top-K can finalize only after successful scoring");
        }
        return collector_->finalize();
    }

    bool has_molecule_ids() const noexcept {
        return has_molecule_ids_;
    }

    bool direct_ids() const noexcept {
        return direct_ids_;
    }

    std::size_t query_count() const noexcept {
        return query_count_;
    }

    std::size_t estimated_state_bytes() const noexcept {
        return collector_ ? collector_->estimated_state_bytes() : 0;
    }

    std::size_t estimated_worker_state_bytes() const noexcept {
        return collector_ ? collector_->estimated_worker_state_bytes() : 0;
    }

    std::size_t estimated_finalize_scratch_bytes() const noexcept {
        return collector_ ? collector_->estimated_finalize_scratch_bytes() : 0;
    }

    std::size_t estimated_result_bytes() const noexcept {
        return collector_ ? collector_->estimated_result_bytes() : 0;
    }

    std::size_t estimated_additional_peak_bytes() const noexcept {
        return collector_ ? collector_->estimated_additional_peak_bytes() : 0;
    }

    std::size_t estimated_id_map_preparation_peak_bytes() const noexcept {
        return id_map_preparation_peak_bytes_;
    }

    std::size_t estimated_final_row_scratch_bytes() const noexcept {
        return final_row_scratch_bytes_;
    }

    std::size_t estimated_python_result_bytes() const noexcept {
        return python_result_bytes_;
    }

    std::size_t state_budget_bytes() const noexcept {
        return collector_ ? collector_->state_budget_bytes() : 0;
    }

private:
    enum class State : unsigned char { Collecting, Scored, Discarded };

    static std::unique_ptr<WorkerLocalTopK> make_collector(
        const std::shared_ptr<const void>& id_maps_owner,
        std::size_t worker_count, std::size_t query_count, std::size_t k,
        std::size_t rank_column, WorkerTopKLevel level,
        std::size_t library_row_count, const std::int64_t* conformer_ids,
        const std::int64_t* molecule_ids, std::size_t state_budget_bytes,
        std::size_t additional_peak_bytes, bool direct_ids) {
        if (!direct_ids && id_maps_owner.use_count() == 0) {
            throw std::invalid_argument(
                "worker-local top-K row sink requires a strong ID-map owner");
        }
        return std::make_unique<WorkerLocalTopK>(
            worker_count, query_count, k, rank_column, level, library_row_count,
            conformer_ids, molecule_ids, state_budget_bytes,
            additional_peak_bytes, direct_ids);
    }

    // Declare the token before the collector so destruction releases the
    // borrowed-view consumer before dropping its backing-storage owner.
    std::shared_ptr<const void> id_maps_owner_;
    bool has_molecule_ids_ = false;
    std::size_t id_map_preparation_peak_bytes_ = 0;
    std::size_t final_row_scratch_bytes_ = 0;
    std::size_t python_result_bytes_ = 0;
    std::unique_ptr<WorkerLocalTopK> collector_;
    bool direct_ids_ = false;
    std::size_t query_count_ = 0;
    // The mapped route's bound margin; NaN (unset) on a direct-ID collector.
    double self_overlap_bound_margin_ =
        std::numeric_limits<double>::quiet_NaN();
    State state_ = State::Collecting;
};

}
}  // namespace r3::cpu

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace r3 {
namespace cpu {

constexpr std::size_t kWorkerTopKScoreColumns = 20;
/// Default planning budget for opt-in CPU worker-local top-K state (256 MiB).
constexpr std::size_t kWorkerTopKDefaultStateBudgetBytes = 256u << 20;
/// Cache-line-padded per-worker row scratch used by sink-only scoring.
constexpr std::size_t kWorkerTopKFinalRowScratchBytes = 128;

enum class WorkerTopKLevel : std::uint8_t {
    Conformer,
    Molecule,
};

struct WorkerTopKHit {
    std::int64_t conformer_id = 0;
    std::int64_t molecule_id = 0;
    bool has_molecule_id = false;
    std::array<float, kWorkerTopKScoreColumns> scores{};
};

struct WorkerLocalTopKMemoryEstimate {
    std::size_t local_capacity = 0;
    std::size_t candidate_capacity = 0;
    std::size_t worker_state_bytes = 0;
    std::size_t finalize_scratch_bytes = 0;
    std::size_t native_result_bytes = 0;
    std::size_t additional_peak_bytes = 0;
    std::size_t peak_bytes = 0;
};

/// Checked allocation-free preflight for the complete optional top-K path.
/// `additional_peak_bytes` covers owned ID-map storage/copies, sink-only
/// worker row scratch and Python-facing result arrays. The constructor calls
/// this same function so factories can reject before preparing those sidecars.
WorkerLocalTopKMemoryEstimate estimate_worker_local_topk_memory(
    std::size_t worker_count, std::size_t query_count, std::size_t k,
    std::size_t rank_column, WorkerTopKLevel level,
    std::size_t library_row_count, std::size_t additional_peak_bytes,
    std::size_t state_budget_bytes);

/// Exact bounded top-K over rows reported by one persistent combo executor.
///
/// The scorer remains responsible for every score and winning pose. This class
/// only sees completed 20-column rows. `conformer_ids` and (for molecule mode)
/// `molecule_ids` are borrowed immutable maps indexed by `library_row`; their
/// owners must keep them alive until all callbacks and `finalize()` complete.
/// They are neither copied nor included in the collector-state estimate.
///
/// Each worker id owns one padded shard and is its only writer. `finalize()`
/// must run after the executor has joined all workers. No callback may race
/// with finalization or reuse the collector after finalization begins.
///
/// The tie id is the global conformer id, matching HostTopK, DeviceTopK and
/// MoleculeTopK's existing winner rule. In molecule mode a molecule's partial
/// maximum is ordered by (score descending, winning conformer id ascending),
/// and the final molecule list uses that same winning id as its tie id.
class WorkerLocalTopK final {
public:
    WorkerLocalTopK(
        std::size_t worker_count, std::size_t query_count, std::size_t k,
        std::size_t rank_column, WorkerTopKLevel level,
        std::size_t library_row_count, const std::int64_t* conformer_ids,
        const std::int64_t* molecule_ids = nullptr,
        std::size_t state_budget_bytes = kWorkerTopKDefaultStateBudgetBytes,
        std::size_t additional_peak_bytes = 0, bool direct_ids = false);

    WorkerLocalTopK(const WorkerLocalTopK&) = delete;
    WorkerLocalTopK& operator=(const WorkerLocalTopK&) = delete;
    WorkerLocalTopK(WorkerLocalTopK&&) = delete;
    WorkerLocalTopK& operator=(WorkerLocalTopK&&) = delete;
    ~WorkerLocalTopK();

    /// The native row-observer shape used by ComboRowSink.
    ///
    /// `output_row` is the stable input-output row; query/library rows are the
    /// already validated pair maps. Completion order does not participate in
    /// ranking. Repeated rows for the same (query, library row) are idempotent
    /// when their complete score rows match bit for bit.
    void on_row(std::size_t worker_id, std::size_t output_row,
                std::size_t query_row, std::size_t library_row,
                const float* scores, std::size_t n_score_columns);

    /// Offer a finalized row with IDs supplied directly by a validated block
    /// descriptor. This avoids a scan-sized row-to-ID map. Descriptor
    /// validation must establish global GID uniqueness and disjoint row spans;
    /// this method only forwards IDs into the same exact bounded reducer.
    void on_row_ids(std::size_t worker_id, std::size_t query_row,
                    std::int64_t conformer_id, std::int64_t molecule_id,
                    bool has_molecule_id, const float* scores,
                    std::size_t n_score_columns);

    /// One canonically ordered result vector per query, each with at most K
    /// rows. Every retained row carries its original 20 score/pose columns.
    std::vector<std::vector<WorkerTopKHit>> finalize() const;

    /// Maximum number of candidate rows any worker/query heap can hold.
    std::size_t local_capacity() const noexcept {
        return local_capacity_;
    }

    /// Sum of the fixed candidate capacities across worker/query shards.
    std::size_t candidate_capacity() const noexcept {
        return candidate_capacity_;
    }

    /// Peak estimate: persistent worker state + one-query merge scratch + the
    /// returned result allocation. Constructor rejects it before allocating
    /// any shard state if it exceeds `state_budget_bytes`.
    ///
    /// This is a planning estimate, not a hard allocator cap: STL hash-node,
    /// bucket and allocator-rounding costs vary by implementation. The named
    /// per-candidate weight includes both bounded indexes.
    std::size_t estimated_state_bytes() const noexcept {
        return estimated_state_bytes_;
    }

    std::size_t estimated_worker_state_bytes() const noexcept {
        return estimated_worker_state_bytes_;
    }

    std::size_t estimated_finalize_scratch_bytes() const noexcept {
        return estimated_finalize_scratch_bytes_;
    }

    std::size_t estimated_result_bytes() const noexcept {
        return estimated_result_bytes_;
    }

    std::size_t estimated_additional_peak_bytes() const noexcept {
        return estimated_additional_peak_bytes_;
    }

    std::size_t state_budget_bytes() const noexcept {
        return state_budget_bytes_;
    }

    /// Maximum number of candidate rows held by any worker/query heap.
    std::size_t maximum_held_candidates() const noexcept;

private:
    class BoundedTopK;
    struct alignas(64) WorkerShard;

    std::size_t worker_count_;
    std::size_t query_count_;
    std::size_t k_;
    std::size_t rank_column_;
    std::size_t library_row_count_;
    std::size_t local_capacity_;
    std::size_t candidate_capacity_;
    std::size_t estimated_worker_state_bytes_;
    std::size_t estimated_finalize_scratch_bytes_;
    std::size_t estimated_result_bytes_;
    std::size_t estimated_additional_peak_bytes_;
    std::size_t estimated_state_bytes_;
    std::size_t state_budget_bytes_;
    WorkerTopKLevel level_;
    const std::int64_t* conformer_ids_;
    const std::int64_t* molecule_ids_;
    bool direct_ids_;
    std::vector<std::unique_ptr<WorkerShard>> workers_;
};

}
}  // namespace r3::cpu

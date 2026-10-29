// Exact worker-local top-K for the CPU combo executor (portable C++, no SIMD).
//
// Sits after scoring: the scorer has already chosen each pair's winning pose
// and filled all 20 output columns, and hands the finished row to the worker
// that produced it. Each worker keeps, per query, a bounded indexed heap of
// its best min(K, library_rows) rows; `finalize()` streams the union of those
// worker heaps through one more bounded heap per query. This replaces a
// `pairs x 20` dense score matrix plus a central reducer.
//
// Key invariants:
//   * One total order everywhere: rank-column score descending, ties by
//     ascending global conformer id -- the same rule as `numerics.rank_order`,
//     HostTopK, DeviceTopK and MoleculeTopK. Float32 ties are common at scale,
//     so the tie-break is part of correctness, not cosmetics.
//   * Exactness of worker-local pruning: a row absent from its worker's local
//     top-K has K rows ahead of it in that worker, which are also in the
//     global input, so it cannot be in the global top-K. In molecule mode the
//     same argument holds over per-molecule partial maxima.
//   * Completion order never participates in ranking.
//   * Bounded state: nothing grows with the number of scored pairs or of
//     distinct library molecules.
//   * Each worker id owns one shard and is its only writer, so `offer` takes
//     no lock; `finalize()` runs only after all workers have joined.
#include "worker_local_topk.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace r3 {
namespace cpu {

namespace {

// One retained row. The 20 score and pose columns travel together, so a
// winner's pose cannot be separated from its score. `library_row` is kept
// only to detect a global conformer id naming two different library rows.
struct Candidate {
    std::array<float, kWorkerTopKScoreColumns> scores{};
    std::int64_t conformer_id = 0;
    std::int64_t molecule_id = 0;
    std::size_t library_row = 0;
    bool has_molecule_id = false;
};

// The production total order: higher rank-column score first, equal scores
// broken by the lower global conformer id (`numerics.rank_order`). NaN never
// reaches this comparison; the callbacks drop NaN rank scores before `offer`.
bool better(const Candidate& lhs, const Candidate& rhs,
            std::size_t rank_column) noexcept {
    const float left = lhs.scores[rank_column];
    const float right = rhs.scores[rank_column];
    if (left > right) return true;
    if (left < right) return false;
    return lhs.conformer_id < rhs.conformer_id;
}

// Bitwise, not float, equality: a repeated (query, conformer) row is accepted
// as idempotent only when every one of its 20 columns has the same bits.
bool same_score_row(const Candidate& lhs, const Candidate& rhs) noexcept {
    return std::memcmp(lhs.scores.data(), rhs.scores.data(),
                       kWorkerTopKScoreColumns * sizeof(float)) == 0;
}

// Overflow-checked size arithmetic for the budget preflight: P * Q * K can
// exceed size_t for hostile inputs, and the estimate must fail rather than
// wrap to a small number that passes the budget.
std::size_t checked_product(std::size_t lhs, std::size_t rhs,
                            const std::string& what) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::length_error(what);
    }
    return lhs * rhs;
}

std::size_t checked_sum(std::size_t lhs, std::size_t rhs,
                        const std::string& what) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        throw std::length_error(what);
    }
    return lhs + rhs;
}

// Conservative planning weights, not claims about a particular STL allocator.
// The per-candidate estimate includes its payload and both hash indexes,
// including bucket/node slack; allocator-specific rounding can still vary.
constexpr std::size_t kEstimatedBytesPerCandidate = 256;
constexpr std::size_t kEstimatedBytesPerQueryShard = 128;
constexpr std::size_t kEstimatedBytesPerWorkerShard = 128;
constexpr std::size_t kEstimatedBytesPerFinalMerge = 128;
constexpr std::size_t kEstimatedBytesPerResultAllocation = 64;
// Allowance for the native sink wrapper and shared ownership/control shell.
constexpr std::size_t kEstimatedBytesPerSinkShell = 128;
constexpr std::size_t kEstimatedBytesPerFinalCandidate =
    kEstimatedBytesPerCandidate + sizeof(Candidate);

}  // namespace

// Allocation-free preflight for the whole opt-in top-K path. With
// K_eff = min(K, library_rows), it prices persistent worker state
// (P * Q * K_eff candidates plus shard shells), one query's final-merge
// scratch (the merge heap is reused query by query), the returned result
// vectors, and caller-supplied sidecars, and throws before anything is
// allocated if the sum exceeds the budget. These are planning weights,
// not an allocator cap or a measured RSS. K is never silently reduced.
WorkerLocalTopKMemoryEstimate estimate_worker_local_topk_memory(
    std::size_t worker_count, std::size_t query_count, std::size_t k,
    std::size_t rank_column, WorkerTopKLevel level,
    std::size_t library_row_count, std::size_t additional_peak_bytes,
    std::size_t state_budget_bytes) {
    if (worker_count == 0) {
        throw std::invalid_argument(
            "worker-local top-K needs at least one worker");
    }
    if (rank_column >= kWorkerTopKScoreColumns) {
        throw std::invalid_argument("top-K rank column must be in [0, 19]");
    }
    if (level != WorkerTopKLevel::Conformer &&
        level != WorkerTopKLevel::Molecule) {
        throw std::invalid_argument("unknown worker-local top-K level");
    }

    // A shard can never hold more distinct rows than the library has.
    const std::size_t local_capacity = std::min(k, library_row_count);
    // Appended to every error so a refusal names P, Q, K and the budget.
    const std::string shape =
        " (P=" + std::to_string(worker_count) +
        ", Q=" + std::to_string(query_count) + ", K=" + std::to_string(k) +
        ", localK=" + std::to_string(local_capacity) +
        ", budget=" + std::to_string(state_budget_bytes) + ")";
    const bool needs_shards = query_count != 0 && local_capacity != 0;
    const std::size_t shard_count =
        needs_shards ? checked_product(
                           worker_count, query_count,
                           "worker/query shard count overflows size_t" + shape)
                     : 0;
    WorkerLocalTopKMemoryEstimate estimate;
    estimate.local_capacity = local_capacity;
    estimate.candidate_capacity = checked_product(
        shard_count, local_capacity,
        "worker-local candidate capacity overflows size_t" + shape);
    const std::size_t candidate_bytes = checked_product(
        estimate.candidate_capacity, kEstimatedBytesPerCandidate,
        "worker-local estimated candidate bytes overflow size_t" + shape);
    const std::size_t final_rows = checked_product(
        query_count, local_capacity,
        "worker-local result capacity overflows size_t" + shape);
    const std::size_t result_payload_bytes = checked_product(
        final_rows, sizeof(WorkerTopKHit),
        "worker-local estimated result payload overflows size_t" + shape);
    const std::size_t result_header_bytes = checked_product(
        query_count, sizeof(std::vector<WorkerTopKHit>),
        "worker-local estimated result headers overflow size_t" + shape);
    // One allocation per query's result vector plus the outer vector.
    const std::size_t result_allocation_count =
        query_count == 0
            ? 0
            : checked_sum(
                  query_count, 1,
                  "worker-local result allocation count overflows size_t" +
                      shape);
    const std::size_t result_allocator_bytes = checked_product(
        result_allocation_count, kEstimatedBytesPerResultAllocation,
        "worker-local result allocator allowance overflows size_t" + shape);
    estimate.native_result_bytes = checked_sum(
        checked_sum(result_payload_bytes, result_header_bytes,
                    "worker-local estimated result storage overflows size_t" +
                        shape),
        result_allocator_bytes,
        "worker-local estimated result storage overflows size_t" + shape);
    const std::size_t query_shard_bytes = checked_product(
        shard_count, kEstimatedBytesPerQueryShard,
        "worker-local estimated query-shard bytes overflow size_t" + shape);
    const std::size_t worker_shard_bytes =
        needs_shards
            ? checked_product(
                  worker_count, kEstimatedBytesPerWorkerShard,
                  "worker-local estimated worker-shard bytes overflow size_t" +
                      shape)
            : 0;
    estimate.worker_state_bytes = checked_sum(
        checked_sum(sizeof(WorkerLocalTopK), kEstimatedBytesPerSinkShell,
                    "worker-local collector shell estimate overflows size_t" +
                        shape),
        checked_sum(
            candidate_bytes,
            checked_sum(query_shard_bytes, worker_shard_bytes,
                        "worker-local estimated worker state overflows size_t" +
                            shape),
            "worker-local estimated worker state overflows size_t" + shape),
        "worker-local estimated worker state overflows size_t" + shape);
    if (needs_shards) {
        estimate.finalize_scratch_bytes = checked_sum(
            checked_product(
                local_capacity, kEstimatedBytesPerFinalCandidate,
                "worker-local final-merge scratch overflows size_t" + shape),
            kEstimatedBytesPerFinalMerge,
            "worker-local final-merge scratch overflows size_t" + shape);
    }
    estimate.additional_peak_bytes = additional_peak_bytes;
    estimate.peak_bytes = checked_sum(
        checked_sum(
            estimate.worker_state_bytes, estimate.finalize_scratch_bytes,
            "worker-local peak state estimate overflows size_t" + shape),
        checked_sum(
            estimate.native_result_bytes, additional_peak_bytes,
            "worker-local peak result/sidecar estimate overflows size_t" +
                shape),
        "worker-local peak state estimate overflows size_t" + shape);
    if (estimate.peak_bytes > state_budget_bytes) {
        throw std::length_error(
            "worker-local top-K peak estimate " +
            std::to_string(estimate.peak_bytes) + " bytes (worker state " +
            std::to_string(estimate.worker_state_bytes) +
            ", final-merge scratch " +
            std::to_string(estimate.finalize_scratch_bytes) +
            ", native result storage " +
            std::to_string(estimate.native_result_bytes) +
            ", owned maps/row scratch/Python result sidecars " +
            std::to_string(estimate.additional_peak_bytes) +
            ") exceeds the configured budget " +
            std::to_string(state_budget_bytes) + " bytes" + shape +
            "; batch query rows, reduce K, or raise the explicit budget. "
            "This is a conservative estimate, not a strict allocator cap.");
    }
    return estimate;
}

// A fixed-capacity indexed binary min-heap: the root is the WORST retained row
// under `better`, so admission is one comparison against the root. Two hash
// indexes, each bounded by the capacity, sit beside it:
//   * `index_by_identity_`: identity -> heap slot, where identity is the
//     conformer id in conformer mode and the molecule id in molecule mode.
//     It lets a molecule's partial maximum be replaced IN PLACE, so the heap
//     never holds stale entries and stays at most `capacity_` long.
//   * `library_row_by_gid_`: conformer id -> library row, so a duplicated id
//     fails closed in molecule mode too.
// The same class serves a worker's per-query shard and the final merge.
class WorkerLocalTopK::BoundedTopK {
public:
    BoundedTopK(std::size_t capacity, std::size_t rank_column,
                WorkerTopKLevel level)
        : capacity_(capacity), rank_column_(rank_column), level_(level) {
        heap_.reserve(capacity_);
        if (capacity_ != 0) {
            index_by_identity_.reserve(capacity_);
            library_row_by_gid_.reserve(capacity_);
        }
    }

    // Admit one row. Three cases, in order: the identity is already held
    // (validate the duplicate; in molecule mode keep the better partial
    // maximum), the heap has room (push), or the heap is full (replace the
    // root only if the candidate beats it). Duplicate-id checks can only see
    // rows that are still retained here; the binding validates global id
    // uniqueness over the full map before scoring.
    void offer(const Candidate& candidate) {
        if (capacity_ == 0) return;

        const auto same_gid = library_row_by_gid_.find(candidate.conformer_id);
        if (same_gid != library_row_by_gid_.end() &&
            same_gid->second != candidate.library_row) {
            throw std::invalid_argument(
                "duplicate conformer id maps to different library rows; "
                "global conformer ids must be unique");
        }

        const std::int64_t identity = identity_of(candidate);
        const auto found = index_by_identity_.find(identity);
        if (same_gid != library_row_by_gid_.end() &&
            found == index_by_identity_.end()) {
            throw std::invalid_argument(
                "one conformer id appears under multiple top-K identities");
        }
        if (found != index_by_identity_.end()) {
            const std::size_t at = found->second;
            Candidate& current = heap_[at];
            validate_duplicate(candidate, current);
            // Conformer mode: a validated repeat is bit-identical, so there
            // is nothing to update.
            if (level_ == WorkerTopKLevel::Conformer) return;
            // Molecule mode: keep the molecule's best conformer, re-keying the
            // gid index when the winning conformer changes.
            if (better(candidate, current, rank_column_)) {
                const std::int64_t old_gid = current.conformer_id;
                if (old_gid != candidate.conformer_id) {
                    library_row_by_gid_.erase(old_gid);
                    library_row_by_gid_.emplace(candidate.conformer_id,
                                                candidate.library_row);
                }
                current = candidate;
                // The slot only improved, so this sifts down (toward the
                // leaves, away from the worst-at-root position).
                fix(at);
            }
            return;
        }

        if (heap_.size() < capacity_) {
            const std::size_t at = heap_.size();
            heap_.push_back(candidate);
            const auto inserted = index_by_identity_.emplace(identity, at);
            if (!inserted.second) {
                throw std::logic_error("top-K identity index lost uniqueness");
            }
            library_row_by_gid_.emplace(candidate.conformer_id,
                                        candidate.library_row);
            sift_up(at);
            return;
        }

        // The root is the worst retained row under the full total ordering,
        // including the gid tie-break. A same-score candidate with a lower gid
        // can therefore replace it; score-only thresholds would miss that.
        if (!better(candidate, heap_.front(), rank_column_)) return;
        const std::int64_t evicted_identity = identity_of(heap_.front());
        const std::int64_t evicted_gid = heap_.front().conformer_id;
        index_by_identity_.erase(evicted_identity);
        library_row_by_gid_.erase(evicted_gid);
        heap_.front() = candidate;
        const auto inserted = index_by_identity_.emplace(identity, 0);
        if (!inserted.second) {
            throw std::logic_error("top-K identity index lost uniqueness");
        }
        library_row_by_gid_.emplace(candidate.conformer_id,
                                    candidate.library_row);
        sift_down(0);
    }

    // Retained rows in canonical rank order. `better` is a strict total order
    // over rows with distinct conformer ids, so an unstable sort is
    // deterministic here.
    std::vector<Candidate> ordered() const {
        std::vector<Candidate> out = heap_;
        std::sort(out.begin(), out.end(),
                  [this](const Candidate& lhs, const Candidate& rhs) {
                      return better(lhs, rhs, rank_column_);
                  });
        return out;
    }

    const std::vector<Candidate>& heap() const noexcept {
        return heap_;
    }

    std::size_t size() const noexcept {
        return heap_.size();
    }

    void clear() noexcept {
        heap_.clear();
        index_by_identity_.clear();
        library_row_by_gid_.clear();
    }

private:
    std::int64_t identity_of(const Candidate& candidate) const noexcept {
        return level_ == WorkerTopKLevel::Molecule ? candidate.molecule_id
                                                   : candidate.conformer_id;
    }

    // A repeat of a held identity is legal only if it is the same library row
    // (a conformer id is unique to one row) and, for the same conformer, its
    // 20 columns match bit for bit. Anything else is a caller or scorer bug
    // and throws rather than silently picking one.
    void validate_duplicate(const Candidate& candidate,
                            const Candidate& current) const {
        if (level_ == WorkerTopKLevel::Conformer &&
            candidate.library_row != current.library_row) {
            throw std::invalid_argument(
                "duplicate conformer id maps to different library rows; "
                "global conformer ids must be unique");
        }
        if (candidate.conformer_id == current.conformer_id &&
            candidate.library_row != current.library_row) {
            throw std::invalid_argument(
                "duplicate conformer id maps to different library rows; "
                "global conformer ids must be unique");
        }
        if (candidate.conformer_id == current.conformer_id &&
            !same_score_row(candidate, current)) {
            throw std::runtime_error(
                "the same query/conformer row produced different score bits");
        }
    }

    bool worse(const Candidate& lhs, const Candidate& rhs) const noexcept {
        return better(rhs, lhs, rank_column_);
    }

    // Every heap move goes through here so the identity -> slot index stays
    // consistent with the heap array.
    void swap_entries(std::size_t lhs, std::size_t rhs) {
        std::swap(heap_[lhs], heap_[rhs]);
        const auto left = index_by_identity_.find(identity_of(heap_[lhs]));
        const auto right = index_by_identity_.find(identity_of(heap_[rhs]));
        if (left == index_by_identity_.end() ||
            right == index_by_identity_.end()) {
            throw std::logic_error("top-K identity index lost a heap entry");
        }
        left->second = lhs;
        right->second = rhs;
    }

    void sift_up(std::size_t at) {
        while (at != 0) {
            const std::size_t parent = (at - 1) / 2;
            if (!worse(heap_[at], heap_[parent])) break;
            swap_entries(at, parent);
            at = parent;
        }
    }

    void sift_down(std::size_t at) {
        for (;;) {
            const std::size_t left = 2 * at + 1;
            if (left >= heap_.size()) return;
            const std::size_t right = left + 1;
            std::size_t worst_child = left;
            if (right < heap_.size() && worse(heap_[right], heap_[left])) {
                worst_child = right;
            }
            if (!worse(heap_[worst_child], heap_[at])) return;
            swap_entries(at, worst_child);
            at = worst_child;
        }
    }

    // Restore the heap property after the entry at `at` changed in place.
    void fix(std::size_t at) {
        if (at != 0 && worse(heap_[at], heap_[(at - 1) / 2])) {
            sift_up(at);
        } else {
            sift_down(at);
        }
    }

    std::size_t capacity_;
    std::size_t rank_column_;
    WorkerTopKLevel level_;
    std::vector<Candidate> heap_;
    std::unordered_map<std::int64_t, std::size_t> index_by_identity_;
    // This second bounded index makes duplicate GIDs fail closed in molecule
    // mode too, where the primary identity is molecule_id rather than gid.
    std::unordered_map<std::int64_t, std::size_t> library_row_by_gid_;
};

// One worker's state: a BoundedTopK per query. Each worker owns one padded
// (64-byte aligned, separately allocated) shard and is its only writer.
struct alignas(64) WorkerLocalTopK::WorkerShard {
    explicit WorkerShard(std::size_t query_count, std::size_t capacity,
                         std::size_t rank_column, WorkerTopKLevel level) {
        queries.reserve(query_count);
        for (std::size_t q = 0; q < query_count; ++q) {
            queries.emplace_back(capacity, rank_column, level);
        }
    }

    std::vector<BoundedTopK> queries;
};

WorkerLocalTopK::WorkerLocalTopK(
    std::size_t worker_count, std::size_t query_count, std::size_t k,
    std::size_t rank_column, WorkerTopKLevel level,
    std::size_t library_row_count, const std::int64_t* conformer_ids,
    const std::int64_t* molecule_ids, std::size_t state_budget_bytes,
    std::size_t additional_peak_bytes, bool direct_ids)
    : worker_count_(worker_count), query_count_(query_count), k_(k),
      rank_column_(rank_column), library_row_count_(library_row_count),
      local_capacity_(std::min(k, library_row_count)), candidate_capacity_(0),
      estimated_worker_state_bytes_(0), estimated_finalize_scratch_bytes_(0),
      estimated_result_bytes_(0),
      estimated_additional_peak_bytes_(additional_peak_bytes),
      estimated_state_bytes_(0), state_budget_bytes_(state_budget_bytes),
      level_(level), conformer_ids_(conformer_ids), molecule_ids_(molecule_ids),
      direct_ids_(direct_ids) {
    if (worker_count_ == 0) {
        throw std::invalid_argument(
            "worker-local top-K needs at least one worker");
    }
    if (rank_column_ >= kWorkerTopKScoreColumns) {
        throw std::invalid_argument("top-K rank column must be in [0, 19]");
    }
    if (!direct_ids_ && library_row_count_ != 0 && conformer_ids_ == nullptr) {
        throw std::invalid_argument(
            "worker-local top-K needs a global conformer id for each library row");
    }
    if (!direct_ids_ && level_ == WorkerTopKLevel::Molecule &&
        library_row_count_ != 0 && molecule_ids_ == nullptr) {
        throw std::invalid_argument(
            "molecule top-K needs a molecule id for each library row");
    }
    if (level_ != WorkerTopKLevel::Conformer &&
        level_ != WorkerTopKLevel::Molecule) {
        throw std::invalid_argument("unknown worker-local top-K level");
    }

    // The same preflight the factories call: it throws before any shard is
    // allocated if the planned peak exceeds the budget.
    const WorkerLocalTopKMemoryEstimate estimate =
        estimate_worker_local_topk_memory(
            worker_count_, query_count_, k_, rank_column_, level_,
            library_row_count_, estimated_additional_peak_bytes_,
            state_budget_bytes_);
    local_capacity_ = estimate.local_capacity;
    candidate_capacity_ = estimate.candidate_capacity;
    estimated_worker_state_bytes_ = estimate.worker_state_bytes;
    estimated_finalize_scratch_bytes_ = estimate.finalize_scratch_bytes;
    estimated_result_bytes_ = estimate.native_result_bytes;
    estimated_state_bytes_ = estimate.peak_bytes;
    const bool needs_shards = query_count_ != 0 && local_capacity_ != 0;
    if (needs_shards) {
        workers_.reserve(worker_count_);
        for (std::size_t worker = 0; worker < worker_count_; ++worker) {
            workers_.emplace_back(std::make_unique<WorkerShard>(
                query_count_, local_capacity_, rank_column_, level_));
        }
    }
}

WorkerLocalTopK::~WorkerLocalTopK() = default;

// Row-index callback: the conformer and molecule ids come from the borrowed
// immutable maps indexed by `library_row`. Called concurrently, one thread per
// `worker_id`; it touches only that worker's shard. `output_row` is part of the
// ComboRowSink shape but unused, because completion order and output position
// do not participate in ranking.
void WorkerLocalTopK::on_row(std::size_t worker_id, std::size_t output_row,
                             std::size_t query_row, std::size_t library_row,
                             const float* scores, std::size_t n_score_columns) {
    (void)output_row;
    if (direct_ids_) {
        throw std::logic_error(
            "row-index top-K callback is unavailable in direct-ID mode");
    }
    if (worker_id >= worker_count_) {
        throw std::out_of_range("combo worker id is outside this top-K state");
    }
    if (query_row >= query_count_) {
        throw std::out_of_range("query row is outside this top-K state");
    }
    if (library_row >= library_row_count_) {
        throw std::out_of_range("library row is outside the immutable id map");
    }
    if (scores == nullptr) {
        throw std::invalid_argument("completed score row pointer is null");
    }
    if (n_score_columns != kWorkerTopKScoreColumns) {
        throw std::invalid_argument(
            "worker-local top-K requires all 20 score columns");
    }
    if (k_ == 0 || local_capacity_ == 0) return;

    Candidate candidate;
    std::copy_n(scores, kWorkerTopKScoreColumns, candidate.scores.begin());
    candidate.conformer_id = conformer_ids_[library_row];
    if (candidate.conformer_id < 0) {
        throw std::invalid_argument(
            "global conformer ids must be non-negative");
    }
    candidate.library_row = library_row;
    candidate.has_molecule_id = molecule_ids_ != nullptr;
    candidate.molecule_id =
        candidate.has_molecule_id ? molecule_ids_[library_row] : 0;
    if (level_ == WorkerTopKLevel::Molecule && !candidate.has_molecule_id) {
        throw std::invalid_argument(
            "molecule top-K has no molecule id for this row");
    }

    // The existing reducers admit no NaN rank score (`NaN >= threshold` is
    // false). Keep that exact behavior; infinities remain ordered normally.
    if (std::isnan(candidate.scores[rank_column_])) return;
    workers_[worker_id]->queries[query_row].offer(candidate);
}

// Direct-ID callback for worker-owned block decoding, where ids arrive with a
// validated block descriptor instead of through a scan-sized row -> id map
// Descriptor validation, not this method, guarantees global id
// uniqueness and disjoint row spans.
void WorkerLocalTopK::on_row_ids(std::size_t worker_id, std::size_t query_row,
                                 std::int64_t conformer_id,
                                 std::int64_t molecule_id, bool has_molecule_id,
                                 const float* scores,
                                 std::size_t n_score_columns) {
    if (!direct_ids_) {
        throw std::logic_error(
            "direct-ID top-K callback requires direct-ID mode");
    }
    if (worker_id >= worker_count_) {
        throw std::out_of_range("combo worker id is outside this top-K state");
    }
    if (query_row >= query_count_) {
        throw std::out_of_range("query row is outside this top-K state");
    }
    if (scores == nullptr) {
        throw std::invalid_argument("completed score row pointer is null");
    }
    if (n_score_columns != kWorkerTopKScoreColumns) {
        throw std::invalid_argument(
            "worker-local top-K requires all 20 score columns");
    }
    if (conformer_id < 0) {
        throw std::invalid_argument(
            "global conformer ids must be non-negative");
    }
    if (level_ == WorkerTopKLevel::Molecule && molecule_id < 0) {
        throw std::invalid_argument("global molecule ids must be non-negative");
    }
    if (level_ == WorkerTopKLevel::Molecule && !has_molecule_id) {
        throw std::invalid_argument(
            "molecule top-K requires a molecule id per row");
    }
    if (k_ == 0 || local_capacity_ == 0) return;

    Candidate candidate;
    std::copy_n(scores, kWorkerTopKScoreColumns, candidate.scores.begin());
    candidate.conformer_id = conformer_id;
    // A validated corpus GID is unique and stable, so it is a compact
    // identity token for this row without allocating an O(N) pair map.
    candidate.library_row = static_cast<std::size_t>(conformer_id);
    candidate.has_molecule_id = has_molecule_id;
    candidate.molecule_id = candidate.has_molecule_id ? molecule_id : 0;
    // Same NaN exclusion as `on_row`.
    if (std::isnan(candidate.scores[rank_column_])) return;
    workers_[worker_id]->queries[query_row].offer(candidate);
}

// Merge the worker shards into one canonically ordered list per query, in
// input query order. Must run after the executor has joined every worker, and
// only after the score call succeeded: partial shard state from a failed or
// cancelled call is not a result. In molecule mode the
// final heap's molecule-id index merges partial maxima of a molecule split
// across workers, keeping the best score and then the lower winning gid.
std::vector<std::vector<WorkerTopKHit>> WorkerLocalTopK::finalize() const {
    std::vector<std::vector<WorkerTopKHit>> result(query_count_);
    if (k_ == 0 || local_capacity_ == 0 || query_count_ == 0) return result;

    for (auto& query_result : result)
        query_result.reserve(local_capacity_);

    BoundedTopK merged(local_capacity_, rank_column_, level_);
    for (std::size_t query = 0; query < query_count_; ++query) {
        // Merge at most W*K retained rows, streaming them through another
        // bounded indexed heap instead of materializing a W*K score matrix.
        for (std::size_t worker = 0; worker < worker_count_; ++worker) {
            for (const Candidate& candidate :
                 workers_[worker]->queries[query].heap()) {
                merged.offer(candidate);
            }
        }
        for (const Candidate& candidate : merged.ordered()) {
            WorkerTopKHit hit;
            hit.conformer_id = candidate.conformer_id;
            hit.molecule_id = candidate.molecule_id;
            hit.has_molecule_id = candidate.has_molecule_id;
            hit.scores = candidate.scores;
            result[query].push_back(std::move(hit));
        }
        // Reuse the single final-merge heap and its bounded index allocations
        // for the next query. Clear retains capacity but no stale entries.
        merged.clear();
    }
    return result;
}

std::size_t WorkerLocalTopK::maximum_held_candidates() const noexcept {
    if (workers_.empty()) return 0;
    std::size_t maximum = 0;
    for (std::size_t worker = 0; worker < worker_count_; ++worker) {
        for (std::size_t query = 0; query < query_count_; ++query) {
            maximum =
                std::max(maximum, workers_[worker]->queries[query].size());
        }
    }
    return maximum;
}

}
}  // namespace r3::cpu

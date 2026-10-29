// Python bindings for the exact worker-local CPU top-K collector.
//
// Instead of streaming every scored row into a dense
// library-sized array, each persistent combo worker keeps an exact
// per-query top-K -- same rank column, same canonical ID tie order -- and the
// at most workers * queries * K survivors are merged once at the end. Ranking
// semantics live in the collector (worker_local_topk.hpp); this file only
// constructs it, maps row indices to global IDs, and estimates memory.
//
// Two ID modes:
//   mapped: the caller hands whole-library conformer/molecule ID arrays, which
//           are copied into an immutable bundle and checked for uniqueness
//           (make_worker_local_topk_row_sink);
//   direct: the worker-owned scan has no library-sized map; each block
//           sink turns a block row into gid_base + row and reads the block's
//           own MID slice (make_worker_local_topk_{direct,block}_row_sink).
//
// Memory figures here are planning estimates checked against a budget before
// anything is allocated, not measured RSS.
#include "worker_local_topk_bindings.hpp"

#include "worker_local_topk_sink.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace r3 {
namespace cpu {
namespace {

struct WorkerTopKIdMapBundle {
    std::vector<std::int64_t> conformer_ids;
    std::vector<std::int64_t> molecule_ids;
};

// Python-side and preparation memory not owned by the collector's heaps; it
// is added to the collector's own estimate before the budget check.
struct WorkerTopKSidecarEstimate {
    std::size_t id_map_preparation_peak_bytes = 0;
    std::size_t final_row_scratch_bytes = 0;
    std::size_t python_result_bytes = 0;
    std::size_t total_bytes = 0;
};

// Fixed per-object allowances for the Python result containers. Estimates
// for the budget arithmetic, not measured object sizes.
constexpr std::size_t kEstimatedPythonArrayShellBytes = 256;
constexpr std::size_t kEstimatedPythonTupleAndListSlotBytes = 136;
constexpr std::size_t kEstimatedPythonResultListShellBytes = 256;

// The self-overlap bound margin a sink checks rows against. It is defined and chosen
// only in Python (roshambo3.scores.SELF_OVERLAP_MARGIN_POLICY, by self-overlap
// route): a caller passes it, and a caller that passes None gets the
// policy's margin for `route` read from roshambo3.scores at this call. Called
// with the GIL held (sink construction).
double resolve_self_overlap_bound_margin(const py::object& margin,
                                         const char* route) {
    const double value =
        margin.is_none()
            ? py::module_::import("roshambo3.scores")
                  .attr("self_overlap_bound_margin")(route)
                  .cast<double>()
            : margin.cast<double>();
    // Finite and above -1, so (1 + margin) is a positive factor, as the CUDA
    // bound scan requires.
    if (!(std::isfinite(value) && value > -1.0)) {
        throw std::invalid_argument(
            "self_overlap_bound_margin must be finite and > -1");
    }
    return value;
}

// Direct-ID adapter for one worker's physical block: translates the scorer's
// block-local library row into the global conformer ID (gid_base + row) and
// the block's molecule ID, then forwards to the shared direct-ID collector.
// One instance per (worker, block); rows outside the claimed block refuse.
// Each block carries the bound margin of the self-overlap route it took.
class WorkerLocalTopKBlockRowSink final : public ComboRowSink {
public:
    WorkerLocalTopKBlockRowSink(
        std::shared_ptr<WorkerLocalTopKRowSink> collector,
        std::size_t worker_id, std::int64_t gid_base, std::size_t block_rows,
        py::array molecule_ids_owner, const std::int64_t* molecule_ids,
        double self_overlap_bound_margin)
        : collector_(std::move(collector)), worker_id_(worker_id),
          gid_base_(gid_base), block_rows_(block_rows),
          molecule_ids_owner_(std::move(molecule_ids_owner)),
          molecule_ids_(molecule_ids),
          self_overlap_bound_margin_(self_overlap_bound_margin) {}

    void on_row(std::size_t, std::size_t, std::size_t query_row,
                std::size_t library_row, const float* scores,
                std::size_t n_score_columns) override {
        if (library_row >= block_rows_) {
            throw std::out_of_range(
                "scorer emitted a row outside its claimed physical block");
        }
        const std::int64_t gid =
            gid_base_ + static_cast<std::int64_t>(library_row);
        const std::int64_t mid = molecule_ids_ ? molecule_ids_[library_row] : 0;
        collector_->on_row_ids(worker_id_, query_row, gid, mid, scores,
                               n_score_columns, self_overlap_bound_margin_);
    }

    double self_overlap_bound_margin() const noexcept {
        return self_overlap_bound_margin_;
    }

    // The shape-only and rescore modes score a block in Python
    // (cpu_modes.score_block) rather than through the native combo executor.
    // Legacy shape/rescore returns one bounded physical block. Feed those
    // exact rows into the same native collector; never construct scan-sized
    // score arrays or duplicate its conformer/molecule ranking semantics.
    void
    ingest_scores(py::array_t<float, py::array::c_style> scores,
                  py::array_t<std::int32_t, py::array::c_style> query_rows,
                  py::array_t<std::int32_t, py::array::c_style> library_rows) {
        if (scores.ndim() != 2 || scores.shape(1) != kWorkerTopKScoreColumns ||
            query_rows.ndim() != 1 || library_rows.ndim() != 1 ||
            scores.shape(0) != query_rows.size() ||
            scores.shape(0) != library_rows.size())
            throw std::invalid_argument(
                "block top-K scores/pair maps disagree");
        const std::size_t rows = std::size_t(scores.shape(0));
        const float* values = scores.data();
        const std::int32_t* qi = query_rows.data();
        const std::int32_t* li = library_rows.data();
        // Validate all maps before offering any row. Array owners and MID
        // owner stay live until GIL reacquisition; one owner per worker.
        for (std::size_t row = 0; row < rows; ++row)
            if (qi[row] < 0 ||
                std::size_t(qi[row]) >= collector_->query_count() ||
                li[row] < 0 || std::size_t(li[row]) >= block_rows_)
                throw std::out_of_range("block top-K pair map outside block");
        py::gil_scoped_release release;
        for (std::size_t row = 0; row < rows; ++row)
            on_row(0, row, std::size_t(qi[row]), std::size_t(li[row]),
                   values + row * kWorkerTopKScoreColumns,
                   kWorkerTopKScoreColumns);
    }

private:
    std::shared_ptr<WorkerLocalTopKRowSink> collector_;
    std::size_t worker_id_;
    std::int64_t gid_base_;
    std::size_t block_rows_;
    // Keeps the mapped MID slice alive until the scorer has joined and emitted
    // every native row. This object is only destroyed with the caller's GIL.
    py::array molecule_ids_owner_;
    const std::int64_t* molecule_ids_;
    double self_overlap_bound_margin_;
};

// Overflow-checked size arithmetic for the estimates: an estimate that wraps
// would pass any budget, so it throws instead.
std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                        const char* description) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        throw std::length_error(description);
    }
    return lhs + rhs;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs,
                        const char* description) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::length_error(description);
    }
    return lhs * rhs;
}

WorkerTopKLevel parse_level(const std::string& level) {
    if (level == "conformer") return WorkerTopKLevel::Conformer;
    if (level == "molecule") return WorkerTopKLevel::Molecule;
    throw std::invalid_argument(
        "worker_local_topk level must be 'conformer' or 'molecule'");
}

// Any integer dtype is accepted here; copy_id_array normalizes it to int64.
std::size_t validate_id_array(const py::array& ids, std::size_t expected_rows,
                              const char* name) {
    if (!py::isinstance<py::array>(ids)) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a NumPy array");
    }
    if (ids.ndim() != 1) {
        throw std::invalid_argument(std::string(name) +
                                    " must be one-dimensional");
    }
    const std::string kind =
        py::str(ids.dtype().attr("kind")).cast<std::string>();
    if (kind != "i" && kind != "u") {
        throw std::invalid_argument(
            std::string(name) +
            " must have a signed or unsigned integer dtype");
    }
    const py::ssize_t n = ids.shape(0);
    if (n < 0 || static_cast<std::size_t>(n) != expected_rows) {
        throw std::invalid_argument(std::string(name) +
                                    " length must equal library row count " +
                                    std::to_string(expected_rows));
    }
    return static_cast<std::size_t>(n);
}

// Worst-case bytes outside the collector: ID-map preparation peak (the
// normalized copies, the sorted uniqueness copy, the bundle), per-worker final
// row scratch, and the Python result -- queries * min(K, rows) score rows plus
// array/tuple shells. `include_id_maps = false` is the direct-ID mode, which
// builds no maps.
WorkerTopKSidecarEstimate
estimate_python_sidecars(std::size_t workers, std::size_t queries,
                         std::size_t k, std::size_t library_rows,
                         bool has_molecule_ids, bool include_id_maps = true) {
    const std::size_t map_count = has_molecule_ids ? 2 : 1;
    const std::size_t map_payload =
        include_id_maps
            ? checked_mul(
                  checked_mul(
                      library_rows, sizeof(std::int64_t),
                      "worker-local ID-map payload estimate overflows size_t"),
                  map_count,
                  "worker-local ID-map payload estimate overflows size_t")
            : 0;
    // A temporary sorted conformer-ID copy validates global-ID uniqueness
    // before any score row can be published. Count its peak alongside the
    // immutable bundle copies made below.
    const std::size_t uniqueness_copy =
        include_id_maps
            ? checked_mul(
                  library_rows, sizeof(std::int64_t),
                  "worker-local ID uniqueness scratch estimate overflows size_t")
            : 0;
    const std::size_t normalization_copy = include_id_maps ? map_payload : 0;
    const std::size_t map_storage = checked_add(
        checked_add(
            checked_add(
                map_payload, uniqueness_copy,
                "worker-local ID-map preparation estimate overflows size_t"),
            normalization_copy,
            "worker-local ID-map preparation estimate overflows size_t"),
        include_id_maps ? sizeof(WorkerTopKIdMapBundle) + 256 : 0,
        "worker-local ID-map preparation estimate overflows size_t");
    const std::size_t row_scratch =
        checked_mul(workers, kWorkerTopKFinalRowScratchBytes,
                    "worker-local final-row scratch estimate overflows size_t");

    const std::size_t local_k = std::min(k, library_rows);
    const std::size_t result_rows =
        checked_mul(queries, local_k,
                    "worker-local Python result row estimate overflows size_t");
    const std::size_t bytes_per_result_row =
        kWorkerTopKScoreColumns * sizeof(float) +
        sizeof(std::int64_t) * (has_molecule_ids ? 2 : 1);
    const std::size_t result_payload = checked_mul(
        result_rows, bytes_per_result_row,
        "worker-local Python result payload estimate overflows size_t");
    const std::size_t arrays_per_query = has_molecule_ids ? 3 : 2;
    const std::size_t result_query_shell = checked_mul(
        queries,
        arrays_per_query * kEstimatedPythonArrayShellBytes +
            kEstimatedPythonTupleAndListSlotBytes,
        "worker-local Python result shell estimate overflows size_t");
    const std::size_t result_shell = checked_add(
        result_query_shell, kEstimatedPythonResultListShellBytes,
        "worker-local Python result shell estimate overflows size_t");

    WorkerTopKSidecarEstimate estimate;
    estimate.id_map_preparation_peak_bytes = map_storage;
    estimate.final_row_scratch_bytes = row_scratch;
    estimate.python_result_bytes =
        checked_add(result_payload, result_shell,
                    "worker-local Python result estimate overflows size_t");
    estimate.total_bytes = checked_add(
        checked_add(estimate.id_map_preparation_peak_bytes,
                    estimate.final_row_scratch_bytes,
                    "worker-local owned sidecar estimate overflows size_t"),
        estimate.python_result_bytes,
        "worker-local owned sidecar estimate overflows size_t");
    return estimate;
}

// Direct-ID collector for the worker-owned scan: no ID maps, so IDs arrive
// with each row through WorkerLocalTopKBlockRowSink. The budget preflight
// throws before any heap is built.
std::shared_ptr<WorkerLocalTopKRowSink> make_worker_local_topk_direct_row_sink(
    std::size_t worker_count, std::size_t query_count, std::size_t k,
    std::size_t rank_column, const std::string& level_name,
    std::size_t library_row_count, bool has_molecule_ids,
    std::size_t state_budget_bytes) {
    const WorkerTopKLevel level = parse_level(level_name);
    if (level == WorkerTopKLevel::Molecule && !has_molecule_ids) {
        throw std::invalid_argument(
            "molecule worker-local top-K requires molecule IDs");
    }
    const WorkerTopKSidecarEstimate sidecars =
        estimate_python_sidecars(worker_count, query_count, k,
                                 library_row_count, has_molecule_ids, false);
    (void)estimate_worker_local_topk_memory(
        worker_count, query_count, k, rank_column, level, library_row_count,
        sidecars.total_bytes, state_budget_bytes);
    return std::make_shared<WorkerLocalTopKRowSink>(
        worker_count, query_count, k, rank_column, level, library_row_count,
        std::shared_ptr<const void>{}, nullptr, nullptr, state_budget_bytes,
        sidecars.total_bytes, 0, sidecars.final_row_scratch_bytes,
        sidecars.python_result_bytes, has_molecule_ids, true);
}

// Per-block adapter over a direct-ID collector. The MID slice is borrowed, not
// copied: it must already be contiguous int64 of exactly `block_rows`, and the
// adapter holds a reference to the array so the pointer stays valid.
std::shared_ptr<WorkerLocalTopKBlockRowSink>
make_worker_local_topk_block_row_sink(
    std::shared_ptr<WorkerLocalTopKRowSink> collector, std::size_t worker_id,
    std::int64_t gid_base, std::size_t block_rows,
    const py::object& molecule_ids,
    const py::object& self_overlap_bound_margin) {
    // A caller that supplies no explicit route uses the supported recompute
    // margin policy. Engine callers pass their resolved route margin directly.
    const double margin = resolve_self_overlap_bound_margin(
        self_overlap_bound_margin, "recompute");
    if (!collector || !collector->direct_ids()) {
        throw std::invalid_argument(
            "block sink requires a direct-ID worker top-K collector");
    }
    if (gid_base < 0 || block_rows == 0 ||
        block_rows - 1 >
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() -
                                     gid_base)) {
        throw std::invalid_argument(
            "block GID range must be non-empty and fit signed int64");
    }
    const bool has_mids = !molecule_ids.is_none();
    if (has_mids != collector->has_molecule_ids()) {
        throw std::invalid_argument(
            "block MID sidecar presence differs from direct top-K config");
    }
    py::array mids_owner;
    if (has_mids) {
        if (!py::isinstance<py::array>(molecule_ids)) {
            throw std::invalid_argument(
                "block molecule IDs must be a NumPy array");
        }
        mids_owner = py::reinterpret_borrow<py::array>(molecule_ids);
        if (mids_owner.ndim() != 1 ||
            static_cast<std::size_t>(mids_owner.shape(0)) != block_rows ||
            !mids_owner.dtype().is(py::dtype::of<std::int64_t>()) ||
            !(mids_owner.flags() & py::array::c_style)) {
            throw std::invalid_argument(
                "block molecule IDs must be contiguous int64 with one row ID each");
        }
    }
    const std::int64_t* mids =
        has_mids ? static_cast<const std::int64_t*>(mids_owner.data())
                 : nullptr;
    return std::make_shared<WorkerLocalTopKBlockRowSink>(
        std::move(collector), worker_id, gid_base, block_rows,
        std::move(mids_owner), mids, margin);
}

// Normalize any integer ID array to int64; unsigned IDs above INT64_MAX
// refuse rather than wrap.
void copy_id_array(const py::array& source,
                   std::vector<std::int64_t>& destination) {
    destination.resize(static_cast<std::size_t>(source.shape(0)));
    const std::string kind =
        py::str(source.dtype().attr("kind")).cast<std::string>();
    if (kind == "i") {
        using SignedIds = py::array_t<std::int64_t, py::array::c_style |
                                                        py::array::forcecast>;
        SignedIds typed = SignedIds::ensure(source);
        if (!typed)
            throw std::invalid_argument("could not normalize signed ID array");
        const auto values = typed.unchecked<1>();
        for (py::ssize_t i = 0; i < source.shape(0); ++i)
            destination[static_cast<std::size_t>(i)] = values(i);
    } else {
        using UnsignedIds =
            py::array_t<std::uint64_t,
                        py::array::c_style | py::array::forcecast>;
        UnsignedIds typed = UnsignedIds::ensure(source);
        if (!typed)
            throw std::invalid_argument(
                "could not normalize unsigned ID array");
        const auto values = typed.unchecked<1>();
        for (py::ssize_t i = 0; i < source.shape(0); ++i) {
            const std::uint64_t value = values(i);
            if (value > static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max())) {
                throw std::invalid_argument(
                    "unsigned global IDs must fit in signed int64");
            }
            destination[static_cast<std::size_t>(i)] =
                static_cast<std::int64_t>(value);
        }
    }
}

// Mapped-ID collector: copies the library ID maps into one immutable bundle
// whose shared_ptr is the collector's lifetime token, so worker threads read
// stable arrays independent of the Python objects. Global conformer IDs must
// be unique: they identify hits and break score ties (numerics.rank_order).
std::shared_ptr<WorkerLocalTopKRowSink> make_worker_local_topk_row_sink(
    std::size_t worker_count, std::size_t query_count, std::size_t k,
    std::size_t rank_column, const std::string& level_name,
    std::size_t library_row_count, const py::array& conformer_ids,
    const py::object& molecule_ids, std::size_t state_budget_bytes,
    const py::object& self_overlap_bound_margin) {
    const double margin = resolve_self_overlap_bound_margin(
        self_overlap_bound_margin, "recompute");
    const WorkerTopKLevel level = parse_level(level_name);
    validate_id_array(conformer_ids, library_row_count,
                      "library_conformer_ids");
    const bool has_molecule_ids = !molecule_ids.is_none();
    if (level == WorkerTopKLevel::Molecule && !has_molecule_ids) {
        throw std::invalid_argument(
            "molecule worker-local top-K requires library_molecule_ids");
    }
    py::array molecule_array;
    if (has_molecule_ids) {
        if (!py::isinstance<py::array>(molecule_ids)) {
            throw std::invalid_argument(
                "library_molecule_ids must be a NumPy array");
        }
        molecule_array = py::reinterpret_borrow<py::array>(molecule_ids);
        validate_id_array(molecule_array, library_row_count,
                          "library_molecule_ids");
    }

    const WorkerTopKSidecarEstimate sidecars = estimate_python_sidecars(
        worker_count, query_count, k, library_row_count, has_molecule_ids);
    // This pure arithmetic preflight rejects before copying IDs, allocating
    // uniqueness scratch, or constructing any worker/query heap.
    (void)estimate_worker_local_topk_memory(
        worker_count, query_count, k, rank_column, level, library_row_count,
        sidecars.total_bytes, state_budget_bytes);

    auto mutable_bundle = std::make_shared<WorkerTopKIdMapBundle>();
    copy_id_array(conformer_ids, mutable_bundle->conformer_ids);
    if (has_molecule_ids)
        copy_id_array(molecule_array, mutable_bundle->molecule_ids);

    std::vector<std::int64_t> sorted_ids = mutable_bundle->conformer_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) !=
        sorted_ids.end()) {
        throw std::invalid_argument(
            "duplicate global conformer IDs are invalid; IDs must be unique "
            "across distinct library rows");
    }

    std::shared_ptr<const WorkerTopKIdMapBundle> immutable_bundle =
        std::move(mutable_bundle);
    std::shared_ptr<const void> owner =
        std::static_pointer_cast<const void>(immutable_bundle);
    const std::int64_t* gids = immutable_bundle->conformer_ids.empty()
                                   ? nullptr
                                   : immutable_bundle->conformer_ids.data();
    const std::int64_t* molids = immutable_bundle->molecule_ids.empty()
                                     ? nullptr
                                     : immutable_bundle->molecule_ids.data();
    return std::make_shared<WorkerLocalTopKRowSink>(
        worker_count, query_count, k, rank_column, level, library_row_count,
        std::move(owner), gids, molids, state_budget_bytes,
        sidecars.total_bytes, sidecars.id_map_preparation_peak_bytes,
        sidecars.final_row_scratch_bytes, sidecars.python_result_bytes,
        has_molecule_ids, false, margin);
}

// One (scores[n, columns], gids[n] [, molecule_ids[n]]) tuple per query, in the
// collector's final rank order; n <= K.
py::list finalize_arrays(const WorkerLocalTopKRowSink& sink) {
    const auto rows = sink.finalize();
    py::list results;
    for (const auto& query_rows : rows) {
        const py::ssize_t n = static_cast<py::ssize_t>(query_rows.size());
        py::array_t<float> scores(
            {n, static_cast<py::ssize_t>(kWorkerTopKScoreColumns)});
        py::array_t<std::int64_t> gids(n);
        py::array_t<std::int64_t> molecule_ids;
        if (sink.has_molecule_ids())
            molecule_ids = py::array_t<std::int64_t>(n);
        float* score_data = scores.mutable_data();
        std::int64_t* gid_data = gids.mutable_data();
        std::int64_t* molecule_data =
            sink.has_molecule_ids() ? molecule_ids.mutable_data() : nullptr;
        for (py::ssize_t i = 0; i < n; ++i) {
            const WorkerTopKHit& hit = query_rows[static_cast<std::size_t>(i)];
            std::copy(hit.scores.begin(), hit.scores.end(),
                      score_data + i * kWorkerTopKScoreColumns);
            gid_data[i] = hit.conformer_id;
            if (sink.has_molecule_ids()) molecule_data[i] = hit.molecule_id;
        }
        if (sink.has_molecule_ids())
            results.append(py::make_tuple(std::move(scores), std::move(gids),
                                          std::move(molecule_ids)));
        else
            results.append(py::make_tuple(std::move(scores), std::move(gids)));
    }
    return results;
}

}  // namespace

void bind_worker_local_topk_api(py::module_& module) {
    py::class_<WorkerLocalTopKRowSink, ComboRowSink,
               std::shared_ptr<WorkerLocalTopKRowSink>>(
        module, "_WorkerLocalTopKRowSink")
        .def("mark_scoring_succeeded",
             &WorkerLocalTopKRowSink::mark_scoring_succeeded)
        .def("discard_after_failure",
             &WorkerLocalTopKRowSink::discard_after_failure)
        .def("finalize", &finalize_arrays)
        .def_property_readonly(
            "memory_estimate", [](const WorkerLocalTopKRowSink& sink) {
                py::dict result;
                result["peak_bytes"] = sink.estimated_state_bytes();
                result["worker_state_bytes"] =
                    sink.estimated_worker_state_bytes();
                result["finalize_scratch_bytes"] =
                    sink.estimated_finalize_scratch_bytes();
                result["native_result_bytes"] = sink.estimated_result_bytes();
                result["id_map_preparation_peak_bytes"] =
                    sink.estimated_id_map_preparation_peak_bytes();
                result["worker_final_row_scratch_bytes"] =
                    sink.estimated_final_row_scratch_bytes();
                result["python_result_bytes"] =
                    sink.estimated_python_result_bytes();
                result["sidecar_bytes"] =
                    sink.estimated_additional_peak_bytes();
                result["budget_bytes"] = sink.state_budget_bytes();
                return result;
            });

    py::class_<WorkerLocalTopKBlockRowSink, ComboRowSink,
               std::shared_ptr<WorkerLocalTopKBlockRowSink>>(
        module, "_CPUBlockTopKRowSink")
        .def("ingest_scores", &WorkerLocalTopKBlockRowSink::ingest_scores,
             py::arg("scores"), py::arg("query_rows"), py::arg("library_rows"))
        .def_property_readonly(
            "self_overlap_bound_margin",
            &WorkerLocalTopKBlockRowSink::self_overlap_bound_margin);

    module.def(
        "make_worker_local_topk_row_sink", &make_worker_local_topk_row_sink,
        py::arg("worker_count"), py::arg("query_count"), py::arg("k"),
        py::arg("rank_column"), py::arg("level"), py::arg("library_row_count"),
        py::arg("library_conformer_ids"),
        py::arg("library_molecule_ids") = py::none(),
        py::arg("state_budget_bytes") = kWorkerTopKDefaultStateBudgetBytes,
        py::arg("self_overlap_bound_margin") = py::none());

    module.def(
        "estimate_worker_local_topk_direct_peak_bytes",
        [](std::size_t workers, std::size_t queries, std::size_t k,
           std::size_t rank_column, const std::string& level_name,
           std::size_t rows, bool has_molecule_ids, std::size_t budget) {
            const auto level = parse_level(level_name);
            if (level == WorkerTopKLevel::Molecule && !has_molecule_ids)
                throw std::invalid_argument(
                    "molecule worker-local top-K requires molecule IDs");
            const auto sidecars = estimate_python_sidecars(
                workers, queries, k, rows, has_molecule_ids, false);
            return estimate_worker_local_topk_memory(
                       workers, queries, k, rank_column, level, rows,
                       sidecars.total_bytes, budget)
                .peak_bytes;
        },
        py::arg("workers"), py::arg("n_queries"), py::arg("k"),
        py::arg("rank_column"), py::arg("level"), py::arg("library_row_count"),
        py::arg("has_molecule_ids"), py::arg("state_budget_bytes"));
    module.def("make_worker_local_topk_direct_row_sink",
               &make_worker_local_topk_direct_row_sink, py::arg("worker_count"),
               py::arg("query_count"), py::arg("k"), py::arg("rank_column"),
               py::arg("level"), py::arg("library_row_count"),
               py::arg("has_molecule_ids") = false,
               py::arg("state_budget_bytes") =
                   kWorkerTopKDefaultStateBudgetBytes);

    module.def("make_worker_local_topk_block_row_sink",
               &make_worker_local_topk_block_row_sink, py::arg("collector"),
               py::arg("worker_id"), py::arg("gid_base"), py::arg("block_rows"),
               py::arg("molecule_ids") = py::none(),
               py::arg("self_overlap_bound_margin") = py::none());
}

}
}  // namespace r3::cpu

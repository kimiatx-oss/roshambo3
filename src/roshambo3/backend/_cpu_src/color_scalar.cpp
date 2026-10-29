// Exact scalar color at the pose selected by the unchanged shape search.
// This translation unit is intentionally separate from interface.cpp: adding
// branches to its large shape dispatch can change GCC's inlining and FMAs.
//
// WHAT THIS FILE IS
//
// The CPU color and combo (shape + color) scoring translation unit. It holds
// the exact scalar color overlap and its gradient, the adapters that
// connect the SIMD color kernels in sibling files (conformer grain, start
// lanes, prepared canonical four-start,
// compatible YMM8 pairs, interleaved NEON streams) and the query
// color fields to the combo optimizer, the optimizer itself
// (optimize_scalar_combo, mode 3, doc/COLOR_CONTRACT.md section 7), the
// fixed-pose color rescore (rescore_scalar_color, mode 2), and their pybind11
// bindings.
//
// FRAMES. Production calls use the shared FP32 reduced-frame pose,
// pose_contract "shared_fp32_reduced_v1"
// (doc/CPU_SHARED_POSE_CONTRACT.md): each optimizer step prepares one M = s*R
// and t' = s*t, with s = COORD_SCALE, and shape, color and the exact final
// scorers all consume those same FP32 bytes. Query color is prescaled once to
// A' = s*A and color decays become d' = d/(s*s). The older physical-frame FP64
// color pose survives as the labeled comparator "legacy_physical_fp64_pose".
//
// NUMERICS. A vector arm reassociates the color sum, so its float differs from
// the scalar arm's; the comparison is on ranked output, never bitwise
// (doc/COLOR_CONTRACT.md section 8). Every reduced-precision arm keeps an
// exact FP64 scalar fallback for matrices or coordinates outside its declared
// range, and the optimizer's approximate value never becomes a reported score:
// reported scores come from a separate accurate final evaluation (unless the
// experimental R3_FINAL_RESCORE_MODE build option is set).
//
// COLOR GRAINS (the color_grain argument). This build supports 0, 11 and 16
// and refuses every other grain id.
//
//     0       scalar exact FP64
//     11      FP32 eight conformers, in input order
//     16      two pairs x four start lanes, scheduled; the production route
//
// MAP, in file order (by name):
//
//   build options      R3_BOUND_EXECUTION, R3_FINAL_SHAPE_VALUE_ONLY,
//                      R3_FINAL_RESCORE_MODE
//   matrix and poses   Matrix, ComboWorkerScratch, Padded*, ColorPose,
//                      shared_color_pose, legacy_color_pose,
//                      reduced_color_matrix
//   exact scalar color overlap, overlap_reduced_color,
//                      prepare_combo_self_overlaps, pose_color,
//                      color_value_gradient_pose
//   evaluator seam     feature_color_evaluation_pose,
//                      evaluate_shared_analytic_color,
//                      evaluate_analytic_color, color_arm_stamp
//   combo weighting    tanimoto, channel_weight, mixed_descent
//   worker runtime     combo_rows_per_work_unit, run_combo_workers
//   census             ComboStats
//   field adapters     evaluate_channel_color_fields,
//                      evaluate_shared_channel_color_fields
//   grain predicates   is_color_conformer_grain, is_color_start_grain
//   prepared handles   PreparedColorFieldHandle, prepare_color_field_handle,
//                      prepare_combo_color_field_state, prepare_*_batches
//   the optimizer      optimize_scalar_combo: three dispatch branches (start
//                      lanes; conformer blocks; one pair per work unit),
//                      then the details dictionary
//   bindings           bind_scalar_color: replays, prepared handles,
//                      diagnostic probes, optimize_scalar_combo, rescore
//
// The noinline helpers (evaluate_prepared_shape,
// prepare_combo_color_field_state, build_combo_color_field_state) exist to keep
// code out of this already large optimizer TU: adding it inline has moved
// unrelated shape/score code across GCC's inlining cliff, which the
// codegen canary detects.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "final_selection_policy.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include "cpu_platform.hpp"
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__aarch64__)
#include "optimizer_exp_neon.hpp"
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "color_evaluator.hpp"
#include "color_conformer_kernel.hpp"
#include "color_exact_final.hpp"
#include "final_rescore.hpp"
#include "color_field_channels.hpp"
#include "combo_executor.hpp"
#include "shape_combo_api.hpp"
#include "shared_pose_api.hpp"
#include "score.hpp"
#include "worker_local_topk.hpp"
#include "worker_local_topk_bindings.hpp"

// Compiled-in execution policies (the shipped "AS2" policy is bound outer-core
// 2, final shape values 1, color final values 0):
//
//   R3_BOUND_EXECUTION 2: the canonical four-start color kernel and the shape
//     evaluators are bound once per work unit and called through
//     evaluate_bound_policy (1 = bound, generic evaluate; 0 = unbound).
//   R3_FINAL_SHAPE_VALUE_ONLY 1: the converged-objective shape pass of the
//     winner-only policy binds a value-only shape execution.
//   R3_SEARCH_VALUE_ONLY (defined further down, default 0): see the
//     winner-only branch of optimize_scalar_combo.
#ifndef R3_BOUND_EXECUTION
#define R3_BOUND_EXECUTION 2
#endif
#ifndef R3_FINAL_SHAPE_VALUE_ONLY
#define R3_FINAL_SHAPE_VALUE_ONLY 1
#endif

// Experimental final arithmetic, independent of optimizer exp and start
// selection. Production remains accurate unless explicitly built otherwise.
#ifndef R3_FINAL_RESCORE_MODE
#define R3_FINAL_RESCORE_MODE 0
#endif
static_assert(R3_FINAL_RESCORE_MODE >= 0 && R3_FINAL_RESCORE_MODE <= 3,
              "final mode must be accurate/scalar-poly/within/across");

namespace py = pybind11;

namespace r3 {
namespace cpu {
namespace {
// (pi/2)^1.5, the color overlap's constant prefactor (doc/COLOR_CONTRACT.md
// section 1). Matrix folds it into every cell's coefficient.
constexpr double COLOR_PREFACTOR = 1.9687012432153024;  // (pi/2)^1.5

// The color interaction matrix, built from the per-type-pair Gaussian width
// array r and height array p (n_features x n_features, [query type][library
// type]). Each nonzero-width cell stores
//
//    coeff = (pi/2)^1.5 * h^2 / w^1.5    and    decay = -w/2
//
// so one feature pair contributes coeff * exp(decay * |x - y|^2) in physical
// Angstrom (doc/COLOR_CONTRACT.md section 1). Type 0 is not a color type (a
// library row's counts[0] is its shape atom count), so every color loop starts
// at 1. A zero-width cell keeps both entries zero and the evaluators skip it; a
// zero height still visits it. `canonical` holds when every color cell has
// width 1 and height delta(a, b): only same-type pairs can then contribute, and
// the loops visit only the diagonal -- the exact predicate for dropping
// off-type work.
struct Matrix {
    int n;
    bool canonical = true;
    double coeff[MAX_COLOR_FEATURES][MAX_COLOR_FEATURES]{};
    double decay[MAX_COLOR_FEATURES][MAX_COLOR_FEATURES]{};

    ColorInteractionMatrixView view() const {
        return {n, MAX_COLOR_FEATURES, canonical, &coeff[0][0], &decay[0][0]};
    }

    Matrix(const py::array_t<float>& r, const py::array_t<float>& p) {
        auto rr = r.unchecked<2>();
        auto pp = p.unchecked<2>();
        n = int(rr.shape(0));
        if (n < 1 || n > MAX_COLOR_FEATURES || rr.shape(1) != n ||
            pp.shape(0) != n || pp.shape(1) != n)
            throw std::invalid_argument(
                "color interaction matrix shape is invalid");
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b) {
                if (!std::isfinite(rr(a, b)) || !std::isfinite(pp(a, b)) ||
                    rr(a, b) < 0.0f)
                    throw std::invalid_argument(
                        "color interaction widths must be finite and nonnegative; "
                        "heights must be finite");
            }
        for (int a = 1; a < n; ++a)
            for (int b = 1; b < n; ++b) {
                const double w = rr(a, b), h = pp(a, b);
                canonical &= (w == 1.0 && h == (a == b ? 1.0 : 0.0));
                if (w != 0.0) {
                    coeff[a][b] = COLOR_PREFACTOR * h * h / std::pow(w, 1.5);
                    decay[a][b] = -0.5 * w;
                }
            }
    }
};

// Scratch is retained by an explicit executor and first touched by the worker
// that owns it. The slot itself is cache-line aligned in ComboExecutor; these
// buffers are never shared between native workers.
struct alignas(64) ComboWorkerScratch {
    ColorConformerWorkspace conformer_workspace;
    std::array<ExactFinalScratch, COLOR_CONFORMER_LANES> exact_final;
    ColorConformerBatch8 color_batch;
};

// One value per 64-byte slot, so rows written concurrently by different workers
// during library decoding never share a cache line (the ownership rule). The
// Padded* wrappers below index like a plain vector of values.
struct alignas(64) ColorConformerSlot {
    ColorConformer value{};
};

struct alignas(64) ShapeLibraryWorkspaceSlot {
    ShapeLibraryWorkspace value{};
};

struct alignas(64) ShapeConformerSlot {
    Conformer value{};
};

struct PaddedColorConformers {
    std::vector<ColorConformerSlot> rows;

    std::size_t size() const noexcept {
        return rows.size();
    }

    ColorConformer& operator[](std::size_t row) noexcept {
        return rows[row].value;
    }

    const ColorConformer& operator[](std::size_t row) const noexcept {
        return rows[row].value;
    }
};

struct PaddedShapeLibraryWorkspaces {
    std::vector<ShapeLibraryWorkspaceSlot> rows;

    std::size_t size() const noexcept {
        return rows.size();
    }

    bool empty() const noexcept {
        return rows.empty();
    }

    ShapeLibraryWorkspace& operator[](std::size_t row) noexcept {
        return rows[row].value;
    }

    const ShapeLibraryWorkspace& operator[](std::size_t row) const noexcept {
        return rows[row].value;
    }
};

struct PaddedShapeConformers {
    std::vector<ShapeConformerSlot> rows;

    std::size_t size() const noexcept {
        return rows.size();
    }

    Conformer& operator[](std::size_t row) noexcept {
        return rows[row].value;
    }

    const Conformer& operator[](std::size_t row) const noexcept {
        return rows[row].value;
    }
};

// The calling worker's retained scratch: owned by the persistent executor when
// there is one, otherwise a thread_local fallback for the one-shot pool.
ComboWorkerScratch& combo_worker_scratch(std::size_t worker,
                                         ComboExecutor* executor) {
    if (executor) return executor->worker_context<ComboWorkerScratch>(worker);
    thread_local ComboWorkerScratch fallback_scratch;
    return fallback_scratch;
}

// A color consumer receives one already-composed pose.  The legacy reference
// constructor below is deliberately kept separate; production shared-mode
// callers populate these fields only from PreparedReducedPose's FP32 bytes.
// Fields: a library point B maps to y = matrix * B + translation, the frame of
// the query coordinates it is compared with. translation_gradient_scale turns
// the translation descent into the optimizer's reduced-frame units: 1 for the
// shared pose (its translation already is t'), 1/COORD_SCALE for the legacy
// physical pose. physical_coordinate_scale is the number of frame units per
// physical Angstrom (COORD_SCALE for the shared pose, 1 for the legacy one);
// the FP32 admission screens use it to keep their limits physical.
struct ColorPose {
    double matrix[9]{};
    double translation[3]{};
    double translation_gradient_scale = 1.0;
    double physical_coordinate_scale = 1.0;
};

// Widen the prepared FP32 M = s*R and t' to double without recomputing them:
// the shared-pose contract forbids any consumer, fallback included, from
// rebuilding a rotation from the quaternion.
ColorPose shared_color_pose(const PreparedReducedPose& prepared) {
    ColorPose pose;
    for (int k = 0; k < 9; ++k)
        pose.matrix[k] = double(prepared.matrix[k]);
    for (int k = 0; k < 3; ++k)
        pose.translation[k] = double(prepared.translation[k]);
    pose.translation_gradient_scale = 1.0;
    pose.physical_coordinate_scale = double(COORD_SCALE);
    return pose;
}

// The older physical pose: normalize the quaternion and build R in FP64,
// translation in physical Angstrom. Kept for pose_contract
// "legacy_physical_fp64_pose", the legacy evaluator seam and the probes, as a
// labeled comparator.
ColorPose legacy_color_pose(const float* quaternion,
                            const float* translation_physical) {
    const double norm = std::sqrt(double(quaternion[0]) * quaternion[0] +
                                  double(quaternion[1]) * quaternion[1] +
                                  double(quaternion[2]) * quaternion[2] +
                                  double(quaternion[3]) * quaternion[3]);
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::invalid_argument(
            "color pose quaternion must be finite and nonzero");
    const double w = double(quaternion[0]) / norm;
    const double x = double(quaternion[1]) / norm;
    const double y = double(quaternion[2]) / norm;
    const double z = double(quaternion[3]) / norm;
    ColorPose pose;
    const double rotation[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                                2 * (x * z + y * w),     2 * (x * y + z * w),
                                1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                                2 * (x * z - y * w),     2 * (y * z + x * w),
                                1 - 2 * (x * x + y * y)};
    for (int k = 0; k < 9; ++k)
        pose.matrix[k] = rotation[k];
    for (int k = 0; k < 3; ++k)
        pose.translation[k] = double(translation_physical[k]);
    pose.translation_gradient_scale = 1.0 / double(COORD_SCALE);
    return pose;
}

// A copy of `physical` carrying the reduced-frame coefficients and decays of a
// ColorInvariantPlan (c' = c, d' = d/(s*s); doc/CPU_SHARED_POSE_CONTRACT.md).
// Paired with prescaled query coordinates and the shared pose.
Matrix reduced_color_matrix(const Matrix& physical,
                            const ColorInvariantPlan& plan) {
    Matrix reduced(physical);
    const ColorInvariantMatrixView view = plan.reduced_matrix_view();
    if (view.frame != ColorDecayFrame::Reduced || !plan.matches_view(view) ||
        view.n_features != physical.n)
        throw std::invalid_argument("invalid reduced color matrix plan");
    for (int ta = 0; ta < physical.n; ++ta)
        for (int tb = 0; tb < physical.n; ++tb) {
            reduced.coeff[ta][tb] = plan.coefficient(ta, tb);
            reduced.decay[ta][tb] = plan.reduced_decay(ta, tb);
        }
    reduced.canonical = view.canonical;
    return reduced;
}

// The exact FP64 color overlap
//
//    V_AB = sum over interacting cells (ta, tb), real features i of A in ta
//           and j of B in tb, of coeff[ta][tb] * exp(decay * |x_i - y_j|^2)
//
// with posed library coordinates from `posed_b` (three doubles per feature in
// feature order), or B's raw coordinates when it is null -- which is how self
// overlaps are computed. `terms` counts evaluated pair terms for the census.
// This is the exact rescore and the exact fallback of every reduced-
// precision arm; its source-order double accumulation is the scalar reference.
// Ordered loops, including (i,i) and both (i,j)/(j,i). Segment bounds come
// from the decoder, never padded L. The general path keeps matrix axes in
// [query type, library type] order, including asymmetric custom matrices.
double overlap(const ColorConformer& a, const ColorConformer& b,
               const Matrix& m, int astride, int bstride, const double* posed_b,
               uint64_t* terms) {
    double sum = 0.0;
    for (int ta = 1; ta < m.n; ++ta) {
        const int tb0 = m.canonical ? ta : 1;
        const int tb1 = m.canonical ? ta + 1 : m.n;
        for (int tb = tb0; tb < tb1; ++tb) {
            const double c = m.coeff[ta][tb];
            // A zero WIDTH skips the cell; a zero HEIGHT still visits it and
            // contributes zero, matching the operation census.
            if (m.decay[ta][tb] == 0.0) continue;
            for (int i = 0; i < a.seg_count[ta]; ++i) {
                const float* x =
                    a.color_xyz + size_t(a.seg_offset[ta] + i) * astride;
                for (int j = 0; j < b.seg_count[tb]; ++j) {
                    const size_t bj = size_t(b.seg_offset[tb] + j);
                    const float* raw = b.color_xyz + bj * bstride;
                    const double* y = posed_b ? posed_b + 3 * bj : nullptr;
                    const double dx =
                        double(x[0]) - (y ? y[0] : double(raw[0]));
                    const double dy =
                        double(x[1]) - (y ? y[1] : double(raw[1]));
                    const double dz =
                        double(x[2]) - (y ? y[2] : double(raw[2]));
                    // One running double sum in loop order. Changing the order
                    // (for example a triangular self sum with doubled
                    // off-diagonal) is a numerics change.
                    sum += c * std::exp(m.decay[ta][tb] *
                                        (dx * dx + dy * dy + dz * dz));
                    if (terms) ++*terms;
                }
            }
        }
    }
    return sum;
}

// Reduced-frame counterpart of overlap(): coefficients and decays come from the
// ColorInvariantPlan (d' = d/(s*s)) and the query rows are expected to hold
// s*A. Used for the shared-pose query color self-overlap, which then normalizes
// color Tanimoto in the same frame as the cross term. Same ordered loops
// and zero-width skip as overlap().
double overlap_reduced_color(const ColorConformer& query_reduced,
                             const ColorConformer& library,
                             const ColorInvariantPlan& plan, int query_stride,
                             int library_stride, const double* posed_library,
                             uint64_t* terms) {
    double sum = 0.0;
    for (int ta = 1; ta < plan.n_features(); ++ta) {
        const int tb_begin = plan.canonical() ? ta : 1;
        const int tb_end = plan.canonical() ? ta + 1 : plan.n_features();
        for (int tb = tb_begin; tb < tb_end; ++tb) {
            const double decay = plan.reduced_decay(ta, tb);
            if (decay == 0.0) continue;
            const double coefficient = plan.coefficient(ta, tb);
            for (int i = 0; i < query_reduced.seg_count[ta]; ++i) {
                const float* x = query_reduced.color_xyz +
                                 size_t(query_reduced.seg_offset[ta] + i) *
                                     size_t(query_stride);
                for (int j = 0; j < library.seg_count[tb]; ++j) {
                    const size_t bj = size_t(library.seg_offset[tb] + j);
                    const float* raw =
                        library.color_xyz + bj * size_t(library_stride);
                    const double* y =
                        posed_library ? posed_library + 3 * bj : nullptr;
                    const double dx =
                        double(x[0]) - (y ? y[0] : double(raw[0]));
                    const double dy =
                        double(x[1]) - (y ? y[1] : double(raw[1]));
                    const double dz =
                        double(x[2]) - (y ? y[2] : double(raw[2]));
                    sum += coefficient *
                           std::exp(decay * (dx * dx + dy * dy + dz * dz));
                    if (terms) ++*terms;
                }
            }
        }
    }
    return sum;
}

// Explicit, caller-owned one-time preparation for the four invariant values
// shared by every optimizer grain and final normalization. The Python owner
// binds this object to a SHA-256 of the normalized inputs; no process-global
// cache or borrowed-array lifetime is involved.
struct PreparedComboSelfOverlaps {
    std::string input_digest;
    std::vector<float> shape_query;
    std::vector<float> shape_library;
    std::vector<double> color_query;
    std::vector<double> color_library;
    int n_features = 0;
    bool shape_prepared = true;
    bool color_prepared = true;
    bool library_shape_from_stored_volume = false;
    bool library_color_from_sidecar = false;
    double shape_preparation_seconds = 0.0;
    double color_preparation_seconds = 0.0;
    uint64_t shape_query_terms = 0;
    uint64_t shape_library_terms = 0;
    uint64_t color_query_terms = 0;
    uint64_t color_library_terms = 0;
};

using PreparedComboSelfOverlapsHandle =
    std::shared_ptr<PreparedComboSelfOverlaps>;

// Compute, or validate and adopt, the four self-overlap vectors once and
// outside alignment, so every grain and the final normalization read the same
// values. Library values are indexed by LIBRARY row, never by pair row.
//
//   shape query     self_overlap of s*A, reduced units, no prefactor
//   shape library   supplied reduced values, or stored physical VAA divided
//                   by SHAPE_PREFACTOR, or recomputed from s*B
//   color query     reduced-frame ordered self overlap (shared-pose frame)
//   color library   physical-frame ordered overlap(B, B), or a sidecar
//
// A color self overlap is mathematically frame-invariant (d' * |s x|^2 = d *
// |x|^2), which is why the library side can stay physical; the two frames are
// not bitwise equal. The Python owner binds the handle to a SHA-256 of the
// normalized inputs (input_digest); optimize_scalar_combo refuses a handle
// whose digest differs.
void validate_self_coordinates(const float* data, size_t n_rows,
                               size_t capacity, int stride,
                               const char* label) {
    const size_t count = n_rows * capacity * size_t(stride);
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(data[i]))
            throw std::invalid_argument(
                std::string(label) + " coordinates/weights must be finite");
    // With stride 4 the fourth column is a per-point weight; it must be
    // nonnegative, padding slots included.
    if (stride == 4) {
        for (size_t row = 0; row < n_rows; ++row)
            for (size_t slot = 0; slot < capacity; ++slot)
                if (data[(row * capacity + slot) * 4 + 3] < 0.0f)
                    throw std::invalid_argument(
                        std::string(label) +
                        " coordinate weights must be nonnegative");
    }
}


void validate_self_library_counts(const uint8_t* data, int n_library,
                                  int library_capacity, int n_features) {
    for (int row = 0; row < n_library; ++row) {
        const uint8_t* counts = data + size_t(row) * size_t(n_features);
        int n_color = 0;
        for (int type = 1; type < n_features; ++type)
            n_color += int(counts[type]);
        if (counts[0] > library_capacity ||
            n_color > library_capacity - int(counts[0]))
            throw std::invalid_argument(
                "library atom and feature counts exceed the coordinate row");
    }
}

PreparedComboSelfOverlapsHandle prepare_combo_self_overlaps(
    py::array_t<float, py::array::c_style> qc,
    py::array_t<int32_t, py::array::c_style> qt,
    py::array_t<int32_t, py::array::c_style> qn,
    py::array_t<float, py::array::c_style> lc,
    py::array_t<uint8_t, py::array::c_style> lcnt,
    py::array_t<float, py::array::c_style> r,
    py::array_t<float, py::array::c_style> p, const std::string& input_digest,
    py::object library_shape_self_reduced, py::object library_shape_volume_a3,
    py::object library_color_self, bool prepare_shape, bool prepare_color) {
    if (qc.ndim() != 3 || (qc.shape(2) != 3 && qc.shape(2) != 4) ||
        qt.ndim() != 2 || qt.shape(0) != qc.shape(0) ||
        qt.shape(1) != qc.shape(1) || qn.ndim() != 1 ||
        qn.shape(0) != qc.shape(0))
        throw std::invalid_argument(
            "self-overlap preparation needs matching query coordinates, "
            "types and atom counts");
    if (lc.ndim() != 3 || (lc.shape(2) != 3 && lc.shape(2) != 4) ||
        lcnt.ndim() != 2 || lcnt.shape(0) != lc.shape(0))
        throw std::invalid_argument(
            "self-overlap preparation needs [n_library,L,3|4] coordinates "
            "and matching count rows");
    if (r.ndim() != 2 || p.ndim() != 2 || r.shape(0) != r.shape(1) ||
        p.shape(0) != r.shape(0) || p.shape(1) != r.shape(1))
        throw std::invalid_argument(
            "self-overlap preparation needs matching square interaction matrices");
    if (input_digest.empty())
        throw std::invalid_argument("self-overlap input digest is required");
    if (!prepare_shape && (!library_shape_self_reduced.is_none() ||
                           !library_shape_volume_a3.is_none()))
        throw std::invalid_argument(
            "shape sidecar values require prepare_shape=true");
    if (!prepare_color && !library_color_self.is_none())
        throw std::invalid_argument(
            "color sidecar values require prepare_color=true");

    const int n_queries = int(qc.shape(0));
    const int query_capacity = int(qc.shape(1));
    const int query_stride = int(qc.shape(2));
    const int n_library = int(lc.shape(0));
    const int library_capacity = int(lc.shape(1));
    const int library_stride = int(lc.shape(2));
    const int n_features = int(r.shape(0));
    if (lcnt.shape(1) != n_features)
        throw std::invalid_argument(
            "library counts and interaction matrix feature widths differ");

    // A prepared denominator is meaningful only for finite, nonnegative
    // self terms. Validate even unused padding here so a malformed snapshot
    // cannot become a reusable handle that hides the original bad data.
    validate_self_coordinates(qc.data(), size_t(n_queries), size_t(query_capacity),
                         query_stride, "query");
    validate_self_coordinates(lc.data(), size_t(n_library), size_t(library_capacity),
                         library_stride, "library");
    for (int row = 0; row < n_queries; ++row)
        if (qn.data()[row] < 0 || qn.data()[row] > query_capacity)
            throw std::invalid_argument(
                "query atom count is outside the query coordinate row");
    validate_self_library_counts(lcnt.data(), n_library, library_capacity, n_features);

    Matrix mat(r, p);
    auto result = std::make_shared<PreparedComboSelfOverlaps>();
    result->input_digest = input_digest;
    result->n_features = n_features;
    result->shape_prepared = prepare_shape;
    result->color_prepared = prepare_color;
    if (prepare_shape) {
        result->shape_query.resize(size_t(n_queries));
        result->shape_library.resize(size_t(n_library));
    }
    if (prepare_color) {
        result->color_query.resize(size_t(n_queries));
        result->color_library.resize(size_t(n_library));
    }

    // Shape: the query side is always computed here, in the reduced frame the
    // shape evaluators use (A' = COORD_SCALE * A).
    const auto shape_started = std::chrono::steady_clock::now();
    std::vector<float> qshape(size_t(n_queries) * size_t(query_capacity) * 3);
    for (int row = 0; prepare_shape && row < n_queries; ++row) {
        const float* src = qc.data() + size_t(row) * size_t(query_capacity) *
                                           size_t(query_stride);
        float* dst = qshape.data() + size_t(row) * size_t(query_capacity) * 3;
        for (int slot = 0; slot < query_capacity; ++slot)
            for (int axis = 0; axis < 3; ++axis)
                dst[size_t(slot) * 3 + size_t(axis)] =
                    COORD_SCALE *
                    src[size_t(slot) * size_t(query_stride) + size_t(axis)];
        const Conformer query{query_capacity == 0 ? nullptr : dst,
                              qn.data()[row]};
        result->shape_query[size_t(row)] = self_overlap(query, nullptr);
        result->shape_query_terms +=
            uint64_t(query.n_atoms) * uint64_t(query.n_atoms);
    }
    if (!prepare_shape) {
        // The channel remains on the documented recompute path at score time.
    } else if (!library_shape_self_reduced.is_none()) {
        const auto values = library_shape_self_reduced
                                .cast<py::array_t<float, py::array::c_style>>();
        if (values.ndim() != 1 || values.shape(0) != n_library)
            throw std::invalid_argument(
                "prepared reduced library shape self-overlaps must have one value per library row");
        for (int row = 0; row < n_library; ++row) {
            const float value = values.data()[row];
            if (!std::isfinite(value) || value < 0.0f)
                throw std::invalid_argument(
                    "prepared reduced library shape self-overlaps must be finite and nonnegative");
            result->shape_library[size_t(row)] = value;
        }
    } else if (library_shape_volume_a3.is_none()) {
        for (int row = 0; row < n_library; ++row) {
            const uint8_t* counts =
                lcnt.data() + size_t(row) * size_t(n_features);
            const int n_atoms = int(counts[0]);
            std::vector<float> scaled(size_t(n_atoms) * 3);
            const float* src = lc.data() + size_t(row) *
                                               size_t(library_capacity) *
                                               size_t(library_stride);
            for (int atom = 0; atom < n_atoms; ++atom)
                for (int axis = 0; axis < 3; ++axis)
                    scaled[size_t(atom) * 3 + size_t(axis)] =
                        COORD_SCALE *
                        src[size_t(atom) * size_t(library_stride) +
                            size_t(axis)];
            const Conformer library{scaled.empty() ? nullptr : scaled.data(),
                                    n_atoms};
            result->shape_library[size_t(row)] = self_overlap(library, nullptr);
            result->shape_library_terms +=
                uint64_t(n_atoms) * uint64_t(n_atoms);
        }
    } else {
        const auto volumes =
            library_shape_volume_a3
                .cast<py::array_t<double, py::array::c_style>>();
        if (volumes.ndim() != 1 || volumes.shape(0) != n_library)
            throw std::invalid_argument(
                "stored library vaa must have one physical-volume value per library row");
        for (int row = 0; row < n_library; ++row) {
            const double volume = volumes.data()[row];
            if (!std::isfinite(volume) || volume < 0.0)
                throw std::invalid_argument(
                    "stored library vaa values must be finite and nonnegative");
            result->shape_library[size_t(row)] =
                float(volume / double(SHAPE_PREFACTOR));
        }
        result->library_shape_from_stored_volume = true;
    }
    if (prepare_shape)
        result->shape_preparation_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          shape_started)
                .count();

    // Color: the query side is prescaled and summed with the reduced plan, as
    // the shared-pose optimizer would; the library side stays physical.
    const auto color_started = std::chrono::steady_clock::now();
    std::unique_ptr<ColorInvariantPlan> color_plan;
    if (prepare_color)
        color_plan = std::make_unique<ColorInvariantPlan>(mat.view());
    std::vector<ColorConformer> query_views(
        prepare_color ? static_cast<size_t>(n_queries) : 0);
    std::vector<std::vector<float>> reduced_query_xyz(
        prepare_color ? static_cast<size_t>(n_queries) : 0);
    std::vector<ColorConformer> reduced_query_views(
        prepare_color ? static_cast<size_t>(n_queries) : 0);
    for (int row = 0; prepare_color && row < n_queries; ++row) {
        const float* src = qc.data() + size_t(row) * size_t(query_capacity) *
                                           size_t(query_stride);
        const int32_t* types = qt.data() + size_t(row) * size_t(query_capacity);
        query_views[size_t(row)] = build_color_view_from_types(
            src, types, qn.data()[row], query_capacity, n_features,
            query_stride);
        const ColorConformer& physical = query_views[size_t(row)];
        auto& reduced_xyz = reduced_query_xyz[size_t(row)];
        reduced_xyz.resize(size_t(physical.n_color) * 3);
        for (int feature = 0; feature < physical.n_color; ++feature) {
            const float* source =
                physical.color_xyz + size_t(feature) * size_t(query_stride);
            for (int axis = 0; axis < 3; ++axis)
                reduced_xyz[size_t(feature) * 3 + size_t(axis)] =
                    COORD_SCALE * source[axis];
        }
        reduced_query_views[size_t(row)] = physical;
        reduced_query_views[size_t(row)].shape_xyz = nullptr;
        reduced_query_views[size_t(row)].color_xyz =
            reduced_xyz.empty() ? nullptr : reduced_xyz.data();
        uint64_t terms = 0;
        result->color_query[size_t(row)] = overlap_reduced_color(
            reduced_query_views[size_t(row)], reduced_query_views[size_t(row)],
            *color_plan, 3, 3, nullptr, &terms);
        result->color_query_terms += terms;
    }

    if (!prepare_color) {
        // Leave library color values empty; optimize_scalar_combo recomputes.
    } else if (library_color_self.is_none()) {
        for (int row = 0; row < n_library; ++row) {
            const float* src = lc.data() + size_t(row) *
                                               size_t(library_capacity) *
                                               size_t(library_stride);
            const uint8_t* counts =
                lcnt.data() + size_t(row) * size_t(n_features);
            const ColorConformer library = build_color_view_from_counts(
                src, counts, n_features, library_stride);
            uint64_t terms = 0;
            result->color_library[size_t(row)] =
                overlap(library, library, mat, library_stride, library_stride,
                        nullptr, &terms);
            result->color_library_terms += terms;
        }
    } else {
        const auto values =
            library_color_self.cast<py::array_t<double, py::array::c_style>>();
        if (values.ndim() != 1 || values.shape(0) != n_library)
            throw std::invalid_argument(
                "color library self-overlaps must have one value per library row");
        for (int row = 0; row < n_library; ++row) {
            const double value = values.data()[row];
            if (!std::isfinite(value) || value < 0.0)
                throw std::invalid_argument(
                    "color library self-overlaps must be finite and nonnegative");
            result->color_library[size_t(row)] = value;
        }
        result->library_color_from_sidecar = true;
    }
    if (prepare_color)
        result->color_preparation_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          color_started)
                .count();
    return result;
}

// Private format-3 seam: both inputs are validated, caller-owned native
// snapshots. Copy denominator vectors without borrowing coordinate lifetimes
// or reevaluating query self expressions for each physical library block.
PreparedComboSelfOverlapsHandle combine_r3g_self_overlaps(
    const PreparedComboSelfOverlapsHandle& query,
    const PreparedComboSelfOverlapsHandle& library,
    const std::string& input_digest) {
    if (!query || !library || input_digest.empty() ||
        query->n_features != library->n_features ||
        !query->shape_prepared || !query->color_prepared ||
        !library->shape_prepared || !library->color_prepared ||
        !query->shape_library.empty() || !query->color_library.empty() ||
        !library->shape_query.empty() || !library->color_query.empty() ||
        query->shape_query.size() != query->color_query.size() ||
        library->shape_library.size() != library->color_library.size())
        throw std::invalid_argument("incompatible format-3 query/library self snapshots");
    auto result = std::make_shared<PreparedComboSelfOverlaps>();
    result->input_digest = input_digest;
    result->n_features = query->n_features;
    result->shape_query = query->shape_query;
    result->color_query = query->color_query;
    result->shape_library = library->shape_library;
    result->color_library = library->color_library;
    result->library_shape_from_stored_volume = library->library_shape_from_stored_volume;
    result->library_color_from_sidecar = library->library_color_from_sidecar;
    // Query evaluations belong to the cached query snapshot, not this bind.
    result->shape_library_terms = library->shape_library_terms;
    result->color_library_terms = library->color_library_terms;
    result->shape_preparation_seconds = library->shape_preparation_seconds;
    result->color_preparation_seconds = library->color_preparation_seconds;
    return result;
}

// Format-3's authenticated decoded block supplies physical VAA and optional
// matrix-qualified color self. Capture Python metadata/pointers while held;
// keep every owning array alive until the released-GIL native work completes.
PreparedComboSelfOverlapsHandle prepare_r3g_library_self_overlaps(
    py::array_t<float, py::array::c_style> lc,
    py::array_t<uint8_t, py::array::c_style> lcnt,
    py::array_t<float, py::array::c_style> r,
    py::array_t<float, py::array::c_style> p,
    const std::string& input_digest,
    py::array_t<double, py::array::c_style> vaa,
    py::object library_color_self, bool prepare_shape, bool prepare_color,
    bool shape_float32_reciprocal) {
    if (lc.ndim() != 3 || (lc.shape(2) != 3 && lc.shape(2) != 4) ||
        lcnt.ndim() != 2 || lcnt.shape(0) != lc.shape(0))
        throw std::invalid_argument("self-overlap preparation needs [n_library,L,3|4] coordinates and matching count rows");
    if (r.ndim() != 2 || p.ndim() != 2 || r.shape(0) != r.shape(1) ||
        p.shape(0) != r.shape(0) || p.shape(1) != r.shape(1))
        throw std::invalid_argument("self-overlap preparation needs matching square interaction matrices");
    if (input_digest.empty())
        throw std::invalid_argument("self-overlap input digest is required");
    if (!prepare_shape && !prepare_color)
        throw std::invalid_argument("format-3 self preparation needs a channel");
    if (!prepare_color && !library_color_self.is_none())
        throw std::invalid_argument(
            "format-3 color self values require prepare_color=true");
    if (lc.shape(0) > std::numeric_limits<int>::max() ||
        lc.shape(1) > std::numeric_limits<int>::max() ||
        r.shape(0) > std::numeric_limits<int>::max())
        throw std::invalid_argument("format-3 self snapshot dimensions exceed native integer range");
    const int n_library = int(lc.shape(0)), capacity = int(lc.shape(1));
    const int stride = int(lc.shape(2)), n_features = int(r.shape(0));
    if (lcnt.shape(1) != n_features)
        throw std::invalid_argument("library counts and interaction matrix feature widths differ");
    if (vaa.ndim() != 1 || vaa.shape(0) != n_library)
        throw std::invalid_argument("stored library vaa must have one physical-volume value per library row");
    py::array_t<double, py::array::c_style> colors;
    const double* color_data = nullptr;
    const bool has_color_self = prepare_color && !library_color_self.is_none();
    if (has_color_self) {
        colors = library_color_self.cast<py::array_t<double, py::array::c_style>>();
        if (colors.ndim() != 1 || colors.shape(0) != n_library)
            throw std::invalid_argument("color library self-overlaps must have one value per library row");
        color_data = colors.data();
    }
    const float* coords = lc.data();
    const uint8_t* counts = lcnt.data();
    const double* volumes = vaa.data();
    Matrix mat(r, p);
    auto result = std::make_shared<PreparedComboSelfOverlaps>();
    result->input_digest = input_digest;
    result->n_features = n_features;
    result->shape_prepared = prepare_shape;
    result->color_prepared = prepare_color;
    {
        py::gil_scoped_release release;
        validate_self_coordinates(coords, size_t(n_library), size_t(capacity), stride, "library");
        validate_self_library_counts(counts, n_library, capacity, n_features);
        if (prepare_shape) {
            result->shape_library.resize(size_t(n_library));
            const auto shape_started = std::chrono::steady_clock::now();
            for (int row = 0; row < n_library; ++row) {
                const double volume = volumes[row];
                if (!std::isfinite(volume) || volume < 0.0)
                    throw std::invalid_argument("stored library vaa values must be finite and nonnegative");
                // Shape/rescore retain the public shape route's float32
                // reciprocal multiplication. Shared-pose combo's declared
                // route divides the physical value in float64 and rounds once.
                result->shape_library[size_t(row)] = shape_float32_reciprocal
                    ? float(float(volume) * (1.0f / SHAPE_PREFACTOR))
                    : float(volume / double(SHAPE_PREFACTOR));
            }
            result->library_shape_from_stored_volume = true;
            result->shape_preparation_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - shape_started).count();
        }
        if (prepare_color) {
            result->color_library.resize(size_t(n_library));
            const auto color_started = std::chrono::steady_clock::now();
            for (int row = 0; row < n_library; ++row) {
                if (has_color_self) {
                    const double value = color_data[row];
                    if (!std::isfinite(value) || value < 0.0)
                        throw std::invalid_argument("color library self-overlaps must be finite and nonnegative");
                    result->color_library[size_t(row)] = value;
                } else {
                    const ColorConformer library = build_color_view_from_counts(
                        coords + size_t(row) * size_t(capacity) * size_t(stride),
                        counts + size_t(row) * size_t(n_features), n_features, stride);
                    uint64_t terms = 0;
                    result->color_library[size_t(row)] = overlap(library, library, mat,
                        stride, stride, nullptr, &terms);
                    result->color_library_terms += terms;
                }
            }
            result->library_color_from_sidecar = has_color_self;
            result->color_preparation_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - color_started).count();
        }
    }
    return result;
}

// Pose library color features into the pose's frame in FP64: y_j = matrix * B_j
// + translation, three doubles per feature in feature order. The quaternion
// overload builds the legacy physical pose first.
std::vector<double> pose_color(const ColorConformer& b, int stride,
                               const ColorPose& pose) {
    std::vector<double> out(size_t(b.n_color) * 3);
    for (int j = 0; j < b.n_color; ++j) {
        const float* p = b.color_xyz + size_t(j) * stride;
        for (int k = 0; k < 3; ++k)
            out[size_t(j) * 3 + k] =
                pose.matrix[3 * k] * p[0] + pose.matrix[3 * k + 1] * p[1] +
                pose.matrix[3 * k + 2] * p[2] + pose.translation[k];
    }
    return out;
}

std::vector<double> pose_color(const ColorConformer& b, int stride,
                               const float* quaternion,
                               const float* translation_physical) {
    return pose_color(b, stride,
                      legacy_color_pose(quaternion, translation_physical));
}

// Exact FP64 color value and descent direction at one pose. The
// seven-component layout matches the shape gradient: descent[0] is unused
// (always 0), descent[1..3] are the half-angle rotation components and
// descent[4..6] the translation components, all as the negative derivative of V
// that the optimizer subtracts (it maximizes overlap). For each term v = coeff
// * exp(decay * |delta|^2), with delta = x - y, w = -2 * decay and the moment
// arm r = y - translation (the rotated library point):
//
//    descent[1..3] += -2 * w * v * (r x delta)
//    descent[4..6] += -w * v * translation_gradient_scale * delta
ColorEvaluation color_value_gradient_pose(const ColorConformer& a,
                                          const ColorConformer& b,
                                          const Matrix& mat, int astride,
                                          int bstride, const ColorPose& pose,
                                          uint64_t* terms) {
    const auto posed = pose_color(b, bstride, pose);
    ColorEvaluation out;
    for (int ta = 1; ta < mat.n; ++ta) {
        const int tb0 = mat.canonical ? ta : 1;
        const int tb1 = mat.canonical ? ta + 1 : mat.n;
        for (int tb = tb0; tb < tb1; ++tb) {
            if (mat.decay[ta][tb] == 0.0) continue;
            const double width = -2.0 * mat.decay[ta][tb];
            for (int i = 0; i < a.seg_count[ta]; ++i) {
                const float* x =
                    a.color_xyz + size_t(a.seg_offset[ta] + i) * astride;
                for (int j = 0; j < b.seg_count[tb]; ++j) {
                    const double* y =
                        posed.data() + 3 * size_t(b.seg_offset[tb] + j);
                    const double dx = double(x[0]) - y[0];
                    const double dy = double(x[1]) - y[1];
                    const double dz = double(x[2]) - y[2];
                    const double value =
                        mat.coeff[ta][tb] *
                        std::exp(mat.decay[ta][tb] *
                                 (dx * dx + dy * dy + dz * dz));
                    out.value += value;
                    // dV/d(half-angle) = 2*r*term*(Rb x delta); Adagrad
                    // subtracts this descent vector. Translation is scaled
                    // by d(t_physical)/d(t_reduced) = 1/COORD_SCALE.
                    const double rx = y[0] - pose.translation[0];
                    const double ry = y[1] - pose.translation[1];
                    const double rz = y[2] - pose.translation[2];
                    const double torque = -2.0 * width * value;
                    out.descent[1] += torque * (ry * dz - rz * dy);
                    out.descent[2] += torque * (rz * dx - rx * dz);
                    out.descent[3] += torque * (rx * dy - ry * dx);
                    const double translate =
                        -width * value * pose.translation_gradient_scale;
                    out.descent[4] += translate * dx;
                    out.descent[5] += translate * dy;
                    out.descent[6] += translate * dz;
                    if (terms) ++*terms;
                }
            }
        }
    }
    return out;
}

ColorEvaluation color_value_gradient(const ColorConformer& a,
                                     const ColorConformer& b, const Matrix& mat,
                                     int astride, int bstride, const float* q,
                                     const float* t_physical, uint64_t* terms) {
    return color_value_gradient_pose(a, b, mat, astride, bstride,
                                     legacy_color_pose(q, t_physical), terms);
}

// =========================================================================
// Architecture-common evaluator seam, and the color grains this build
// supports.
// =========================================================================
//
// The supported grains are 0, 11 and 16 (see the file header); every entry
// point refuses any other grain id.
void require_supported_color_grain(int grain) {
    if (grain != 0 && grain != 11 && grain != 16)
        throw std::invalid_argument(
            "unsupported CPU color grain (this build supports grains 0, 11 "
            "and 16)");
}

// One color value (and optionally descent) evaluation at an already-composed
// pose. Grains 0, 11 and 16 all take the exact FP64 path here; grains 11 and
// 16 have batched kernels of their own in optimize_scalar_combo.
ColorEvaluation feature_color_evaluation_pose(
    const ColorConformer& a, const ColorConformer& b, const Matrix& mat,
    int astride, int bstride, const ColorPose& pose, int grain, uint64_t* terms,
    bool gradient) {
    require_supported_color_grain(grain);
    if (gradient)
        return color_value_gradient_pose(a, b, mat, astride, bstride, pose,
                                         terms);
    const auto posed = pose_color(b, bstride, pose);
    ColorEvaluation exact;
    exact.value = overlap(a, b, mat, astride, bstride, posed.data(), terms);
    return exact;
}

// Legacy-pose overload (unit quaternion + physical translation), used by the
// legacy evaluator seam, the rescore path and the probes.
ColorEvaluation feature_color_evaluation(
    const ColorConformer& a, const ColorConformer& b, const Matrix& mat,
    int astride, int bstride, const float* quaternion,
    const float* translation_physical, int grain, uint64_t* terms,
    bool gradient) {
    return feature_color_evaluation_pose(
        a, b, mat, astride, bstride,
        legacy_color_pose(quaternion, translation_physical), grain, terms,
        gradient);
}

// Shared-pose analytic color: the query is the prescaled reduced conformer, the
// matrix the reduced one, and the pose the prepared FP32 bytes widened by
// shared_color_pose. Grain 0 is evaluated inline (exact FP64); other grains go
// through feature_color_evaluation_pose with the same pose.
ColorEvaluation evaluate_shared_analytic_color(
    const ReducedColorConformer& query_reduced,
    const ColorConformer& library_physical, const Matrix& reduced_matrix,
    int query_stride, int library_stride, const PreparedReducedPose& prepared,
    int grain, uint64_t* terms, bool gradient) {
    const ColorPose pose = shared_color_pose(prepared);
    const ColorConformer& query = query_reduced.features;
    if (grain == 0) {
        if (gradient)
            return color_value_gradient_pose(query, library_physical,
                                             reduced_matrix, query_stride,
                                             library_stride, pose, terms);
        const auto posed = pose_color(library_physical, library_stride, pose);
        ColorEvaluation exact;
        exact.value =
            overlap(query, library_physical, reduced_matrix, query_stride,
                    library_stride, posed.data(), terms);
        return exact;
    }
    return feature_color_evaluation_pose(
        query, library_physical, reduced_matrix, query_stride, library_stride,
        pose, grain, terms, gradient);
}

// The legacy-pose ColorEvaluator seam: an opaque context plus a function
// pointer, so the one-pair optimizer branch can swap analytic color for channel
// fields (evaluate_channel_color_fields) without a per-call branch. The
// request's matrix view must be the context's own matrix; a mismatch is a
// wiring error and throws.
struct AnalyticColorEvaluatorContext {
    const Matrix* matrix;
    int grain;
};

ColorEvaluation evaluate_analytic_color(void* opaque,
                                        const ColorEvaluationRequest& request,
                                        uint64_t* analytic_pair_terms) {
    const auto& context = *static_cast<AnalyticColorEvaluatorContext*>(opaque);
    const Matrix& mat = *context.matrix;
    if (request.interaction.n_features != mat.n ||
        request.interaction.canonical != mat.canonical ||
        request.interaction.coefficients != &mat.coeff[0][0] ||
        request.interaction.decays != &mat.decay[0][0])
        throw std::invalid_argument(
            "color evaluator matrix view does not match context");
    if (context.grain == 0) {
        if (request.gradient)
            return color_value_gradient(
                *request.query, *request.library, mat, request.query_stride,
                request.library_stride, request.quaternion,
                request.translation_physical, analytic_pair_terms);
        const auto posed =
            pose_color(*request.library, request.library_stride,
                       request.quaternion, request.translation_physical);
        ColorEvaluation out;
        out.value =
            overlap(*request.query, *request.library, mat, request.query_stride,
                    request.library_stride, posed.data(), analytic_pair_terms);
        return out;
    }
    return feature_color_evaluation(
        *request.query, *request.library, mat, request.query_stride,
        request.library_stride, request.quaternion,
        request.translation_physical, context.grain, analytic_pair_terms,
        request.gradient);
}

// Self-description of a color grain for result provenance (every arm records
// its runtime ISA and reduction width): arm name, ISA,
// vector width and encoding, value/gradient precision, exp and its -80 tail
// policy, FP32 admission limits, reduction order, and whether the grain is
// compiled into this binary and runnable on this CPU. For grain 16 it also
// describes the prepared canonical route, the compatible-pair and NEON stream
// policies and the final rescore mode, and keeps the older grain-16 fields
// under color_general_fallback_* (they describe the retained general evaluator,
// not the fast route).
py::dict color_arm_stamp(int grain) {
    require_supported_color_grain(grain);
    py::dict out;
    out["color_grain"] = grain;
    out["color_scalar_default"] = true;
    out["color_target_compiled"] = true;
    out["color_runtime_available"] = true;
    out["color_reduction_width"] = 1;
    out["color_reduction"] =
        "ordered scalar accumulation; no horizontal reduction";
    if (grain == 0) {
        out["color_arm"] = "scalar";
        out["color_isa"] = "SCALAR";
        out["color_vector_width"] = 1;
        out["color_compile_method"] = "baseline translation unit";
    } else if (grain == 11) {
        // Grain 11: conformer lanes.
        out["color_arm"] = "avx2_fp32_poly_conformer8";
        out["color_isa"] = "AVX2+FMA";
        out["color_vector_width"] = 8;
        out["color_vector_widths"] = py::make_tuple(8);
        out["color_vector_encoding"] = "VEX.256 only on AVX2+FMA target";
        out["color_value_precision"] =
            "fp32 terms and per-conformer fp32 accumulators; fp64 plane widening";
        out["color_gradient_precision"] =
            "fp32 terms and per-conformer fp32 accumulators; fp64 plane widening";
        out["color_exp_tail_cutoff"] = -80.0;
        out["color_exp_tail_policy"] = "terms below exp(-80) set to zero";
        out["color_fp32_coefficient_max"] = 1.0e8;
        out["color_fp32_width_max"] = 1.0e6;
        out["color_fp32_coordinate_abs_max"] = 1.0e9;
        out["color_out_of_range_policy"] = "exact fp64 scalar fallback";
        out["color_lane_owner"] = "independent query/library conformer pair";
        out["color_reduction_width"] = 1;
        out["color_reduction"] =
            "no cross-conformer reduction; fp32 lane subtotals widened once per target plane";
        out["color_count_order"] = "input order";
        out["color_compile_method"] = "translation-unit target x86-64-v3";
        out["color_runtime_features"] = py::make_tuple("avx2", "fma");
        out["color_exp"] = "degree-7 range-reduced AVX2 polynomial";
        out["color_scalar_default"] = true;
#if defined(__AVX2__)
        __builtin_cpu_init();
        out["color_target_compiled"] = true;
        out["color_runtime_available"] =
            __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        out["color_target_compiled"] = false;
        out["color_runtime_available"] = false;
#endif
    } else {
        // Grain 16: start lanes (overwritten below for its NEON and
        // prepared-route fields).
        out["color_arm"] = "avx2_fp32_poly_scheduled_start8";
        out["color_isa"] = "AVX2+FMA";
        out["color_vector_width"] = 8;
        out["color_vector_widths"] = py::make_tuple(8);
        out["color_vector_encoding"] = "VEX.256 existing pair evaluator";
        out["color_pose_transform"] =
            "unchanged per-start fp64 transform and fp32 pose pack";
        out["color_declared_numerics_change"] =
            "no transform, exponential, cutoff, fallback, or within-start reduction change";
        out["color_lane_owner"] =
            "four independent starts for pair A in lanes 0..3 and pair B in lanes 4..7; no cross-start reduction";
        out["color_start_count"] = 4;
        out["color_start_mode"] = 1;
        out["color_start_fallback"] =
            "other start modes dispatch through the existing scalar optimizer path";
        out["color_value_precision"] =
            "fp32 polynomial terms and per-start fp32 accumulators; fp64 plane widening; exact fp64 per-start and winner rescoring";
        out["color_gradient_precision"] =
            "fp32 polynomial terms and per-start fp32 accumulators; fp64 plane widening; no cross-start reduction";
        out["color_exp_tail_cutoff"] = -80.0;
        out["color_exp_tail_policy"] = "terms below exp(-80) set to zero";
        out["color_fp32_coefficient_max"] = 1.0e8;
        out["color_fp32_width_max"] = 1.0e6;
        out["color_fp32_coordinate_abs_max"] = 1.0e9;
        out["color_out_of_range_policy"] = "exact fp64 scalar fallback";
        out["color_reduction_width"] = 1;
        out["color_reduction"] =
            "no cross-start reduction; fp32 subtotals widened once per target plane";
        out["color_count_order"] = "input pair order; no count bucketing";
        out["color_compile_method"] = "translation-unit target x86-64-v3";
        out["color_runtime_features"] = py::make_tuple("avx2", "fma");
        out["color_exp"] = "degree-7 range-reduced AVX2 polynomial";
        out["color_scalar_default"] = true;
#if defined(__AVX2__)
        __builtin_cpu_init();
        out["color_target_compiled"] = true;
        out["color_runtime_available"] =
            __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        out["color_target_compiled"] = false;
        out["color_runtime_available"] = false;
#endif
    }
#if defined(__aarch64__)
    if (grain == 16) {
        out["color_arm"] = "neon_fp32_scheduled_start8";
        out["color_isa"] = "NEON";
        out["color_vector_width"] = 4;
        out["color_vector_widths"] = py::make_tuple(4);
        out["color_vector_encoding"] =
            "two independent 128-bit FP32 NEON vectors";
        out["color_compile_method"] = "native AArch64 translation unit";
        out["color_runtime_features"] = py::make_tuple("asimd");
        out["color_target_compiled"] = true;
        out["color_runtime_available"] = true;
        out["color_exp"] =
            "NEON degree-7 polynomial or explicit FCVTZS integer-bias rawbits";
        out["color_pose_transform"] =
            "shared FP32 reduced pose and prepared coordinate pack";
        out["color_declared_numerics_change"] =
            "native FMA and plane accumulation order; exact finals retained";
    }
#endif
    if (grain == 16) {
        // Grain16's prepared canonical route is the FP32 production
        // path. This stamp describes that route while keeping its retained
        // general/noncanonical and unsafe-input FP64 fallback explicit.
        out["color_prepared_canonical_route"] = true;
#if defined(__aarch64__)
        out["color_prepared_canonical_helper"] =
            "neon_fp32_canonical_analytic_neon4";
        out["color_prepared_canonical_isa"] = "NEON";
        out["color_prepared_canonical_vector_encoding"] =
            "128-bit NEON FP32; four independent start lanes";
#else
        out["color_prepared_canonical_helper"] =
            "avx2_fp32_canonical_analytic_xmm4";
        out["color_prepared_canonical_isa"] = "AVX2+FMA";
        out["color_prepared_canonical_vector_encoding"] =
            "VEX.128 XMM packed FP32 FMA; YMM integer term-count bookkeeping; four independent start lanes";
#endif
        out["color_prepared_canonical_vector_width"] = 4;
        // The four-start helper above remains the singleton route. Compatible
        // pairs select a separate eight-lane helper on native x86 builds.
#if defined(__x86_64__)
        out["color_compatible_pair_policy_enabled"] =
            bool(R3_COLOR_COMPATIBLE_PAIRS);
        out["color_compatible_pair_helper"] =
            R3_COLOR_COMPATIBLE_PAIRS ? "avx2_fp32_canonical_analytic_ymm8"
                                     : "disabled";
        out["color_compatible_pair_vector_width"] =
            R3_COLOR_COMPATIBLE_PAIRS ? 8 : 0;
#else
        out["color_compatible_pair_policy_enabled"] = false;
        out["color_compatible_pair_helper"] = "disabled";
        out["color_compatible_pair_vector_width"] = 0;
#endif
        out["color_neon_streams"] = R3_COLOR_START_STREAMS;
#if defined(__aarch64__)
        out["color_neon_compatible_pair_policy_enabled"] = R3_COLOR_START_STREAMS == 2;
        out["color_neon_compatible_pair_helper"] =
            R3_COLOR_START_STREAMS == 2 ? "two_independent_neon4_streams" : "disabled";
#else
        out["color_neon_compatible_pair_policy_enabled"] = false;
        out["color_neon_compatible_pair_helper"] = "disabled";
#endif
        // The older grain-16 fields above describe the retained general
        // evaluator/fallback, not this prepared canonical fast route.
        out["color_general_fallback_arm"] = out["color_arm"];
        out["color_general_fallback_isa"] = out["color_isa"];
        out["color_general_fallback_vector_width"] = out["color_vector_width"];
        out["color_general_fallback_vector_encoding"] =
            out["color_vector_encoding"];
        out["color_pose_transform"] =
            "prepared canonical route: shared FP32 reduced pose transforms FP32 coordinates into FP32 posed storage; general/noncanonical or unsafe fallback retains FP64";
        out["color_declared_numerics_change"] =
            "prepared canonical route uses FP32 transforms, plane accumulation, and accurate analytic finals; general/noncanonical and unsafe inputs retain the FP64 fallback; evaluate with ranked-output gates, not FP64 bit equality";
        out["color_value_precision"] =
            "prepared canonical route: FP32 transform, plane accumulation, and accurate analytic final; scalar widens once at the API boundary; general/noncanonical and unsafe fallback retains FP64";
        out["color_gradient_precision"] =
            "prepared canonical route: FP32 terms and plane accumulators per start; components widen once at the API boundary; no cross-start reduction; general/noncanonical and unsafe fallback retains FP64";
        out["color_reduction"] =
            "prepared canonical route accumulates all planes in FP32 per start and widens once after the complete plane sum; no cross-start reduction";
        out["color_analytic_final_exp"] =
            "std::exp(float) accurate analytic final; optimizer approximation is not used for final scoring";
        out["color_out_of_range_policy"] =
            "unsafe canonical inputs and general/noncanonical calls use the retained exact FP64 scalar fallback";
        out["final_rescore_mode"] = 0;
        out["final_rescore_compiled_mode"] = R3_FINAL_RESCORE_MODE;
        out["final_rescore_scope"] =
            "winner route always accurate mode0; compiled experimental final mode requires all_start control; actual mode in census";
#if R3_FINAL_RESCORE_MODE != 0
        out["experimental_final_rescore_exp"] =
            "degree-7 range-reduced polynomial exp; shape argument -ln(2)*d2; canonical color decay; accurate unsafe/general fallback";
        out["experimental_final_rescore_precision"] =
            "prepared canonical route: FP32 transform, plane accumulation, and experimental polynomial final; scalar widens once at the API boundary; general/noncanonical and unsafe fallback retains FP64";
        out["experimental_final_rescore_numerics_change"] =
            "experimental FP32 polynomial final arithmetic and SIMD reduction; optimizer/start-selection policy unchanged; general/noncanonical and unsafe inputs retain FP64 accurate fallback";
#endif
    }
    out["color_compiled"] = true;
    return out;
}

#if defined(__AVX2__)
#endif

// =========================================================================
// Combo weighting: the reported score and the direction that ascends it.
// =========================================================================
//
// Tanimoto V_AB / (V_AA + V_BB - V_AB), defined as 0 when the denominator is
// not positive (two empty color sides give 0/0).
double tanimoto(double cross, double self_a, double self_b) {
    const double den = self_a + self_b - cross;
    return den > 0.0 ? cross / den : 0.0;
}

// channel_weight is dT/dV_AB for T = V_AB / (V_AA + V_BB - V_AB), guarded like
// tanimoto:
//
//    dT/dV_AB = (V_AA + V_BB) / (V_AA + V_BB - V_AB)^2
//
// The two independent dT/dV weights. get_gradient(..., gnorm=1) and
// color_value_gradient both return RAW negative overlap derivatives in the
// same six pose coordinates, so there is no hidden legacy 1/(AA+BB) factor.
double channel_weight(double cross, double self_a, double self_b) {
    const double sum = self_a + self_b;
    const double den = sum - cross;
    return den > 0.0 ? sum / (den * den) : 0.0;
}

// Descent direction of the reported combo score T_combo = (1 - mixing) *
// T_shape + mixing * T_color, by the chain rule: each channel's raw overlap
// derivative is weighted by its own dT/dV before mixing (adding raw
// gradients would optimize neither reported Tanimoto). Computed in double,
// stored as float; component 0 stays 0.
void mixed_descent(const float shape_g[7], float shape_cross, float shape_aa,
                   float shape_bb, const ColorEvaluation& color,
                   double color_aa, double color_bb, float mixing,
                   float out[7]) {
    const double ws = channel_weight(shape_cross, shape_aa, shape_bb);
    const double wc = channel_weight(color.value, color_aa, color_bb);
    out[0] = 0.0f;
    for (int k = 1; k < 7; ++k)
        out[k] = float((1.0 - mixing) * ws * shape_g[k] +
                       mixing * wc * color.descent[k]);
}

// =========================================================================
// Worker runtime: how pair rows become work units and run on native threads.
// =========================================================================
using Clock = std::chrono::steady_clock;

std::vector<int> combo_allowed_cpus() {
    return r3::cpu::platform::allowed_cpus();
}

// Pair rows per work unit, mirroring the three optimizer branches: start
// grain 16 at the standard start mode takes two pairs (eight start lanes),
// conformer grain 11 COLOR_CONFORMER_LANES pairs, everything else one pair.
// The executor claims in whole units; combo_thread_config_stamp reports the
// same decomposition.
size_t combo_rows_per_work_unit(int color_grain, int start_mode) {
    size_t block_size = 1;
    if (color_grain == 16 && start_mode == 1)
        block_size = 2;
    else if (color_grain == 11)
        block_size = COLOR_CONFORMER_LANES;
    return block_size;
}

size_t combo_work_units(int color_grain, int start_mode, size_t n_pairs) {
    const size_t block_size = combo_rows_per_work_unit(color_grain, start_mode);
    return (n_pairs + block_size - 1) / block_size;
}

const char* combo_work_unit_name(int color_grain, int start_mode) {
    if (color_grain == 16 && start_mode == 1)
        return "two_pair_start_block";
    if (color_grain == 11) return "eight_pair_color_block";
    return "one_pair";
}

// Runtime contract: release the Python GIL while native workers run and
// pin each worker to a CPU from the caller's allowed affinity mask. Workers use
// coarse dynamic claims; one atomic fetch_add claims a whole row chunk, never
// an individual alignment. The caller supplies topology policy; this helper
// does not infer siblings from CPU numbers or change machine-wide settings.
// Without an executor, this spawns min(threads, work_units) std::threads.
// Each claims claim_chunk_rows / rows_per_work_unit units per
// relaxed fetch_add: a unit is claimed exactly once and join() publishes the
// results, so no stronger ordering is needed. The first exception cancels the
// remaining claims and is rethrown after every thread joins. threads == 1 runs
// inline (still without the GIL). With an executor, its persistent workers run
// the same lambda and report elapsed time, placement and claim counts
// from their own bookkeeping.
template <typename Work>
void run_combo_workers(
    size_t work_units, int requested_threads, Work&& work,
    std::vector<double>* worker_elapsed_seconds = nullptr,
    std::vector<ComboWorkerPlacement>* worker_placements = nullptr,
    bool require_worker_affinity = false, ComboExecutor* executor = nullptr,
    size_t rows_per_work_unit = 1, size_t claim_chunk_rows = 256,
    std::vector<uint64_t>* worker_claim_counts = nullptr) {
    if (requested_threads < 1)
        throw std::invalid_argument("threads must be >= 1");
    if (rows_per_work_unit == 0)
        throw std::invalid_argument("rows_per_work_unit must be >= 1");
    // Executor path: its persistent workers run every unit while the GIL is
    // released; timing, placement and claim counts come from its bookkeeping.
    if (executor) {
        if (size_t(requested_threads) != executor->worker_count())
            throw std::invalid_argument(
                "combo executor worker count does not match threads");
        if (require_worker_affinity && !executor->require_worker_affinity())
            throw std::invalid_argument(
                "combo executor was not created with required affinity");
        {
            py::gil_scoped_release nogil;
            executor->run(
                work_units, rows_per_work_unit,
                [&](size_t worker, size_t unit) { work(int(worker), unit); });
        }
        if (worker_elapsed_seconds) {
            if (worker_elapsed_seconds->size() < executor->worker_count())
                worker_elapsed_seconds->resize(executor->worker_count(), 0.0);
            for (size_t worker = 0; worker < executor->worker_count(); ++worker)
                (*worker_elapsed_seconds)[worker] =
                    executor->worker_elapsed_seconds(worker);
        }
        if (worker_placements) {
            if (worker_placements->size() < executor->worker_count())
                worker_placements->resize(executor->worker_count());
            for (size_t worker = 0; worker < executor->worker_count(); ++worker)
                (*worker_placements)[worker] =
                    executor->worker_placement(worker);
        }
        if (worker_claim_counts)
            *worker_claim_counts = executor->stats().last_claims_by_worker;
        return;
    }
    if (work_units == 0) return;
    if (requested_threads == 1) {
        py::gil_scoped_release nogil;
        for (size_t unit = 0; unit < work_units; ++unit)
            work(0, unit);
        return;
    }
    const int requested_workers = std::min<int>(
        requested_threads,
        int(std::min<size_t>(work_units,
                             size_t(std::numeric_limits<int>::max()))));

    const std::vector<int> cpus = combo_allowed_cpus();
    const int workers = requested_workers;
    if (claim_chunk_rows < 128 || claim_chunk_rows > 512 ||
        claim_chunk_rows % 16 != 0 || rows_per_work_unit == 0)
        throw std::invalid_argument(
            "invalid combo claim chunk or work-unit row width");
    if (require_worker_affinity && cpus.empty())
        throw std::runtime_error(
            "required combo worker affinity has no allowed CPU mask");
    if (require_worker_affinity && size_t(workers) > cpus.size())
        throw std::runtime_error(
            "required combo workers exceed the process allowed CPU mask");
    if (worker_elapsed_seconds &&
        worker_elapsed_seconds->size() < size_t(workers))
        worker_elapsed_seconds->resize(size_t(workers), 0.0);
    if (worker_placements && worker_placements->size() < size_t(workers))
        worker_placements->resize(size_t(workers));

    std::vector<std::exception_ptr> worker_failures(
        static_cast<size_t>(workers));
    std::exception_ptr spawn_failure;

    // Cursor, cancel flag and per-worker claim counters each own a cache line:
    // the cursor is the one intentionally shared word; the rest must not
    // false-share with it.
    struct alignas(64) WorkCursor {
        std::atomic<size_t> next{0};
    };

    struct alignas(64) WorkCancellation {
        std::atomic<bool> requested{false};
    };

    struct alignas(64) WorkClaimCount {
        uint64_t value = 0;
    };

    std::vector<WorkClaimCount> claims_by_worker;
    if (worker_claim_counts) claims_by_worker.resize(size_t(workers));
    WorkCursor next_work_unit;
    WorkCancellation cancelled;
    const size_t claim_items =
        std::max<size_t>(1, claim_chunk_rows / rows_per_work_unit);
    const auto worker_qos = r3::cpu::platform::current_qos();
    auto run_range = [&](int worker) {
        std::chrono::steady_clock::time_point worker_started{};
        try {
            // Copy the caller's macOS thread QoS class to the worker (a
            // scheduling hint; "unsupported" elsewhere). Then pin to one CPU of
            // the allowed mask, round robin; with require_worker_affinity a
            // failed or different binding is an error rather than a silent
            // unpinned run.
            if (worker_qos != "unsupported" && worker_qos != "unspecified")
                r3::cpu::platform::set_qos(worker_qos);
            if (!cpus.empty()) {
                const int requested_cpu = cpus[size_t(worker) % cpus.size()];
                const bool affinity_succeeded =
                    r3::cpu::platform::bind_current_thread(requested_cpu);
                if (worker_placements) {
                    auto& placement = (*worker_placements)[size_t(worker)];
                    placement.requested_cpu = requested_cpu;
                    placement.affinity_attempted = true;
                    placement.affinity_succeeded = affinity_succeeded;
                }
                if (require_worker_affinity && !affinity_succeeded)
                    throw std::runtime_error(
                        "required combo worker CPU affinity could not be set");
                if (worker_placements || require_worker_affinity) {
                    const auto actual = r3::cpu::platform::allowed_cpus();
                    if (require_worker_affinity &&
                        (actual.size() != 1 || actual[0] != requested_cpu))
                        throw std::runtime_error(
                            "required combo worker OS affinity differs from its requested CPU");
                    if (worker_placements)
                        (*worker_placements)[size_t(worker)].actual_cpus =
                            actual;
                }
            }
            if (worker_elapsed_seconds)
                worker_started = std::chrono::steady_clock::now();
            while (!cancelled.requested.load(std::memory_order_relaxed)) {
                const size_t first = next_work_unit.next.fetch_add(
                    claim_items, std::memory_order_relaxed);
                if (first >= work_units) break;
                if (worker_claim_counts)
                    ++claims_by_worker[size_t(worker)].value;
                const size_t stop = work_units - first < claim_items
                                        ? work_units
                                        : first + claim_items;
                for (size_t unit = first; unit < stop; ++unit) {
                    if (cancelled.requested.load(std::memory_order_relaxed))
                        break;
                    work(worker, unit);
                }
            }
        } catch (...) {
            // This catch owns the entire native-thread entry, including
            // placement observation allocations, so nothing escapes through
            // std::thread and turns into std::terminate.
            worker_failures[size_t(worker)] = std::current_exception();
            cancelled.requested.store(true, std::memory_order_relaxed);
        }
        if (worker_elapsed_seconds)
            (*worker_elapsed_seconds)[size_t(worker)] =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              worker_started)
                    .count();
    };

    std::vector<std::thread> pool;
    pool.reserve(size_t(workers));
    {
        py::gil_scoped_release nogil;
        try {
            for (int worker = 0; worker < workers; ++worker)
                pool.emplace_back(run_range, worker);
        } catch (...) {
            spawn_failure = std::current_exception();
        }
        for (auto& thread : pool)
            thread.join();
    }
    if (spawn_failure) std::rethrow_exception(spawn_failure);
    for (const auto& failure : worker_failures)
        if (failure) std::rethrow_exception(failure);
    if (worker_claim_counts) {
        worker_claim_counts->resize(size_t(workers));
        for (int worker = 0; worker < workers; ++worker)
            (*worker_claim_counts)[size_t(worker)] =
                claims_by_worker[size_t(worker)].value;
    }
}

// Per-call counters and phase timers of optimize_scalar_combo. Inside worker
// lambdas they are written only under with_census or with_timers, and both are
// refused for threads > 1, so the plain members never race. "terms" are useful
// pair terms, "issued" counts include padded vector lanes, and the final_rescore_*
// members count the experimental final-rescore batches (R3_FINAL_RESCORE_MODE !=
// 0).
struct ComboStats {
    uint64_t converged_objective_evaluations = 0;
    uint64_t converged_objective_color_terms = 0;
    uint64_t analytic_final_shape_evaluations = 0;
    uint64_t winner_only_final_pairs = 0;
    double converged_objective_s = 0.0;
    FinalRescoreCounts final_rescore_counts;
    uint64_t final_rescore_batches = 0, final_rescore_pairs = 0;
    uint64_t shared_pose_preparations = 0;
    uint64_t color_gradient_evaluations = 0;
    uint64_t color_per_start_final_evaluations = 0;
    uint64_t color_winner_rescore_evaluations = 0;
    uint64_t color_gradient_terms = 0;
    uint64_t color_per_start_final_terms = 0;
    uint64_t color_winner_rescore_terms = 0;
    uint64_t color_query_self_terms = 0;
    uint64_t color_library_self_terms = 0;
    uint64_t shape_query_self_terms = 0;
    uint64_t shape_library_self_terms = 0;
    uint64_t shape_query_self_evaluations = 0;
    uint64_t shape_library_self_evaluations = 0;
    uint64_t color_query_self_evaluations = 0;
    uint64_t color_library_self_evaluations = 0;
    bool prepared_self_overlaps_used = false;
    bool prepared_shape_query_self_used = false;
    bool prepared_shape_library_self_used = false;
    bool prepared_color_query_self_used = false;
    bool prepared_color_library_self_used = false;
    bool shared_shape_query_self_used = false;
    bool shared_color_query_self_used = false;
    ColorConformerCensus color_conformer_census;
    ColorConformerCensus color_start_census;
    ColorFieldBatchStats color_start_field_stats;
    ColorFieldBatchStats color_conformer_field_stats;
    uint64_t color_field_samples = 0;
    uint64_t color_field_interpolated_samples = 0;
    uint64_t color_field_analytic_selected_samples = 0;
    uint64_t color_field_analytic_selected_terms = 0;
    uint64_t color_field_analytic_fallback_samples = 0;
    uint64_t color_field_inactive_fallback_samples = 0;
    uint64_t color_field_out_of_grid_fallback_samples = 0;
    uint64_t color_field_analytic_fallback_terms = 0;
    uint64_t shape_gradient_terms = 0;
    uint64_t shape_issued_terms = 0;
    uint64_t shape_fallback_evaluations = 0;
    uint64_t shape_fallback_terms = 0;
    uint64_t shape_field_samples = 0;
    uint64_t shape_gradient_evaluations = 0;
    uint64_t shape_workspace_bytes = 0;
    uint64_t color_field_bytes = 0;
    std::vector<uint64_t> color_field_bytes_by_target_type;
    std::vector<double> color_field_build_s_by_target_type;
    std::vector<double> color_field_pack_s_by_target_type;
    std::vector<uint64_t> color_field_active_queries_by_target_type;
    double color_field_build_s = 0.0;
    double color_field_pack_s = 0.0;
    std::array<uint64_t, 4> color_field_bytes_by_backend{};
    std::array<double, 4> color_field_build_s_by_backend{};
    std::array<double, 4> color_field_pack_s_by_backend{};
    double shape_gradient_s = 0.0;
    double shape_target_workspace_prepare_s = 0.0;
    double color_gradient_s = 0.0;
    double color_conformer_stable_sort_s = 0.0;
    double color_conformer_pack_s = 0.0;
    double color_conformer_pose_transform_s = 0.0;
    double color_conformer_matrix_screen_s = 0.0;
    double color_conformer_scalar_fallback_s = 0.0;
    double color_conformer_pose_soa_pack_s = 0.0;
    double color_conformer_vector_kernel_s = 0.0;
    double color_conformer_field_sample_s = 0.0;
    double color_start_pack_s = 0.0;
    double color_start_pose_transform_s = 0.0;
    double color_start_matrix_screen_s = 0.0;
    double color_start_scalar_fallback_s = 0.0;
    double color_start_pose_soa_pack_s = 0.0;
    double color_start_vector_kernel_s = 0.0;
    double color_start_field_sample_s = 0.0;
    uint64_t color_start_source_coordinate_components_loaded = 0;
    uint64_t color_start_shared_query_broadcast8_calls = 0;
    uint64_t color_start_paired_query_broadcast4_calls = 0;
    uint64_t color_start_scheduled_batch_numeric_bytes = 0;
    double combo_weighting_s = 0.0;
    double pose_update_s = 0.0;
    double shared_pose_prepare_s = 0.0;
    double per_start_final_s = 0.0;
    double winner_rescore_s = 0.0;
    bool census = false;
    bool timers = false;
};

// The fast shape adapters are separate TUs. This noinline bridge keeps their
// virtual dispatch and adapter bodies out of the already large optimizer TU.
__attribute__((noinline)) void
evaluate_prepared_shape(const PreparedShapeEvaluator& evaluator,
                        const ShapePoseRequest& request,
                        ShapeEvaluation& output) {
    evaluator.evaluate_one(request, output);
}

// Fold one prepared shape evaluation's own operation counts into the census.
void count_prepared_shape(const ShapeEvaluation& evaluation,
                          ComboStats& stats) {
    stats.shape_gradient_terms += evaluation.counts.useful_terms;
    stats.shape_issued_terms += evaluation.counts.issued_terms;
    stats.shape_fallback_evaluations += evaluation.counts.fallback_evaluations;
    stats.shape_fallback_terms += evaluation.counts.fallback_terms;
    stats.shape_field_samples += evaluation.counts.field_samples;
}

// =========================================================================
// Query color fields (per-plane hybrid). Each query owns a
// ColorFieldBundle with one plane per target type, built in physical Angstrom
// and storing value plus three spatial-gradient channels. Gradients are
// interpolated from those stored channels, never differentiated from the
// interpolated value. Inactive, analytic-selected and out-of-grid
// samples are evaluated analytically inside the bundle. The counters below
// classify every sample.
// =========================================================================
struct ChannelColorEvaluatorContext {
    const ColorFieldBundle* bundle = nullptr;
    const Matrix* matrix = nullptr;
    uint64_t* samples = nullptr;
    uint64_t* interpolated_samples = nullptr;
    uint64_t* analytic_selected_samples = nullptr;
    uint64_t* analytic_selected_terms = nullptr;
    uint64_t* analytic_fallback_samples = nullptr;
    uint64_t* inactive_fallback_samples = nullptr;
    uint64_t* out_of_grid_fallback_samples = nullptr;
    uint64_t* analytic_fallback_terms = nullptr;
};

// Legacy-pose field color: for each target plane that some query type
// contributes to, each posed library feature samples its query's plane. The
// spatial gradient g = dV/dy (physical) becomes descent the same way as the
// analytic path: rotation -2 * (r x g), translation -g / COORD_SCALE.
ColorEvaluation
evaluate_channel_color_fields(void* opaque,
                              const ColorEvaluationRequest& request,
                              uint64_t* analytic_pair_terms) {
    (void)
        analytic_pair_terms;  // Field samples/fallbacks have separate counters.
    const auto& context = *static_cast<ChannelColorEvaluatorContext*>(opaque);
    if (context.bundle == nullptr || context.matrix == nullptr ||
        request.query == nullptr || request.library == nullptr)
        throw std::invalid_argument("incomplete channel color field evaluator");
    const Matrix& mat = *context.matrix;
    if (request.interaction.n_features != mat.n ||
        request.interaction.canonical != mat.canonical ||
        request.interaction.coefficients != &mat.coeff[0][0] ||
        request.interaction.decays != &mat.decay[0][0])
        throw std::invalid_argument(
            "color evaluator matrix view does not match field context");

    const ColorConformer& a = *request.query;
    const ColorConformer& b = *request.library;
    const int astride = request.query_stride;
    const int bstride = request.library_stride;
    const auto posed = pose_color(b, bstride, request.quaternion,
                                  request.translation_physical);
    for (double value : posed) {
        if (!std::isfinite(value) ||
            std::abs(value) > double(std::numeric_limits<float>::max())) {
            // The field interface is fp32-coordinate by design. Preserve the
            // exact scalar path for values it cannot represent rather than
            // narrowing an extreme coordinate into an invalid sample.
            if (request.gradient)
                return color_value_gradient(
                    a, b, mat, astride, bstride, request.quaternion,
                    request.translation_physical, analytic_pair_terms);
            ColorEvaluation exact;
            exact.value = overlap(a, b, mat, astride, bstride, posed.data(),
                                  analytic_pair_terms);
            return exact;
        }
    }

    // A target plane is skipped when no query type with features has a nonzero
    // cell into it.
    ColorEvaluation out;
    for (int tb = 1; tb < mat.n; ++tb) {
        bool contributes = false;
        for (int ta = 1; ta < mat.n; ++ta)
            contributes |= a.seg_count[ta] > 0 && mat.coeff[ta][tb] != 0.0 &&
                           mat.decay[ta][tb] != 0.0;
        if (!contributes) continue;
        for (int j = 0; j < b.seg_count[tb]; ++j) {
            const size_t feature = size_t(b.seg_offset[tb] + j);
            const double* y = posed.data() + feature * 3;
            const float point[3] = {float(y[0]), float(y[1]), float(y[2])};
            const ColorFieldSample sample =
                context.bundle->sample_target_type(tb, point);
            if (context.samples) ++*context.samples;
            // Classify the sample: interpolated; analytic because this plane
            // selected backend 0; or analytic fallback (inactive plane, or
            // point off the grid).
            if (sample.used_field) {
                if (context.interpolated_samples)
                    ++*context.interpolated_samples;
            } else if (context.bundle->backend_code(tb) ==
                       COLOR_FIELD_ANALYTIC) {
                if (context.analytic_selected_samples)
                    ++*context.analytic_selected_samples;
                if (context.analytic_selected_terms) {
                    for (int ta = 1; ta < mat.n; ++ta)
                        if (mat.coeff[ta][tb] != 0.0 &&
                            mat.decay[ta][tb] != 0.0)
                            *context.analytic_selected_terms +=
                                uint64_t(a.seg_count[ta]);
                }
            } else {
                if (context.analytic_fallback_samples)
                    ++*context.analytic_fallback_samples;
                if (context.bundle->field_active(tb)) {
                    if (context.out_of_grid_fallback_samples)
                        ++*context.out_of_grid_fallback_samples;
                } else if (context.inactive_fallback_samples) {
                    ++*context.inactive_fallback_samples;
                }
                if (context.analytic_fallback_terms) {
                    for (int ta = 1; ta < mat.n; ++ta)
                        if (mat.coeff[ta][tb] != 0.0 &&
                            mat.decay[ta][tb] != 0.0)
                            *context.analytic_fallback_terms +=
                                uint64_t(a.seg_count[ta]);
                }
            }
            out.value += sample.value;
            if (!request.gradient) continue;
            const double gx = sample.spatial_gradient[0];
            const double gy = sample.spatial_gradient[1];
            const double gz = sample.spatial_gradient[2];
            const double rx = y[0] - request.translation_physical[0];
            const double ry = y[1] - request.translation_physical[1];
            const double rz = y[2] - request.translation_physical[2];
            // A channel contains dV/dy in physical Angstrom units. The
            // optimizer consumes negative derivatives in half-angle and
            // reduced-translation coordinates, matching the analytic
            // gradient contract.
            out.descent[1] += -2.0 * (ry * gz - rz * gy);
            out.descent[2] += -2.0 * (rz * gx - rx * gz);
            out.descent[3] += -2.0 * (rx * gy - ry * gx);
            out.descent[4] += -gx / double(COORD_SCALE);
            out.descent[5] += -gy / double(COORD_SCALE);
            out.descent[6] += -gz / double(COORD_SCALE);
        }
    }
    return out;
}

// Shared-pose scalar field adapter for the nonstandard-start fallback. The
// field grid stays in physical Angstroms, but every library coordinate is
// first transformed by the exact prepared FP32 M/t' bytes. Any analytic
// fallback is evaluated in the reduced query/decay frame with those same
// pose bytes; it never rebuilds a quaternion transform.
ColorEvaluation evaluate_shared_channel_color_fields(
    const ChannelColorEvaluatorContext& context,
    const ReducedColorConformer& query_reduced,
    const ColorConformer& library_physical, int library_stride,
    const Matrix& reduced_matrix, const ColorInvariantPlan& plan,
    const PreparedReducedPose& prepared, bool gradient,
    uint64_t* analytic_pair_terms) {
    if (context.bundle == nullptr || context.matrix == nullptr)
        throw std::invalid_argument(
            "incomplete shared-pose channel field evaluation");
    const Matrix& physical_matrix = *context.matrix;
    plan.require_matching_input(physical_matrix.view());
    if (plan.reduced_matrix_view().frame != ColorDecayFrame::Reduced ||
        !plan.matches_view(plan.reduced_matrix_view()))
        throw std::invalid_argument(
            "shared-pose field evaluation needs the prepared reduced matrix");

    const ColorPose pose = shared_color_pose(prepared);
    const std::vector<double> posed_reduced =
        pose_color(library_physical, library_stride, pose);
    const double scale = double(COORD_SCALE);
    // The grid is physical: divide the reduced posed point by s. A point FP32
    // cannot hold sends the whole evaluation to the shared analytic path.
    for (double value : posed_reduced) {
        const double physical = value / scale;
        if (!std::isfinite(physical) ||
            std::abs(physical) > double(std::numeric_limits<float>::max())) {
            return evaluate_shared_analytic_color(
                query_reduced, library_physical, reduced_matrix, 3,
                library_stride, prepared, 0, analytic_pair_terms, gradient);
        }
    }

    ColorEvaluation out;
    const ColorConformer& query = query_reduced.features;
    for (int tb = 1; tb < physical_matrix.n; ++tb) {
        bool contributes = false;
        for (int ta = 1; ta < physical_matrix.n; ++ta)
            contributes |= query.seg_count[ta] > 0 &&
                           physical_matrix.coeff[ta][tb] != 0.0 &&
                           physical_matrix.decay[ta][tb] != 0.0;
        if (!contributes) continue;
        for (int j = 0; j < library_physical.seg_count[tb]; ++j) {
            const size_t at = size_t(library_physical.seg_offset[tb] + j) * 3;
            const float point[3] = {float(posed_reduced[at] / scale),
                                    float(posed_reduced[at + 1] / scale),
                                    float(posed_reduced[at + 2] / scale)};
            const ColorFieldSample sample =
                context.bundle->sample_target_type(tb, point);
            if (context.samples) ++*context.samples;
            // Same sample classification as evaluate_channel_color_fields.
            if (sample.used_field) {
                if (context.interpolated_samples)
                    ++*context.interpolated_samples;
            } else if (context.bundle->backend_code(tb) ==
                       COLOR_FIELD_ANALYTIC) {
                if (context.analytic_selected_samples)
                    ++*context.analytic_selected_samples;
                if (context.analytic_selected_terms) {
                    for (int ta = 1; ta < physical_matrix.n; ++ta)
                        if (physical_matrix.coeff[ta][tb] != 0.0 &&
                            physical_matrix.decay[ta][tb] != 0.0)
                            *context.analytic_selected_terms +=
                                uint64_t(query.seg_count[ta]);
                }
            } else {
                if (context.analytic_fallback_samples)
                    ++*context.analytic_fallback_samples;
                if (context.bundle->field_active(tb)) {
                    if (context.out_of_grid_fallback_samples)
                        ++*context.out_of_grid_fallback_samples;
                } else if (context.inactive_fallback_samples) {
                    ++*context.inactive_fallback_samples;
                }
                if (context.analytic_fallback_terms) {
                    for (int ta = 1; ta < physical_matrix.n; ++ta)
                        if (physical_matrix.coeff[ta][tb] != 0.0 &&
                            physical_matrix.decay[ta][tb] != 0.0)
                            *context.analytic_fallback_terms +=
                                uint64_t(query.seg_count[ta]);
                }
            }
            out.value += sample.value;
            if (!gradient) continue;
            const double gx = sample.spatial_gradient[0];
            const double gy = sample.spatial_gradient[1];
            const double gz = sample.spatial_gradient[2];
            // Torque arm is (M B)/s, and physical-lattice translation
            // derivatives are converted once to reduced optimizer units.
            const double rx =
                (posed_reduced[at] - double(prepared.translation[0])) / scale;
            const double ry =
                (posed_reduced[at + 1] - double(prepared.translation[1])) /
                scale;
            const double rz =
                (posed_reduced[at + 2] - double(prepared.translation[2])) /
                scale;
            out.descent[1] += -2.0 * (ry * gz - rz * gy);
            out.descent[2] += -2.0 * (rz * gx - rx * gz);
            out.descent[3] += -2.0 * (rx * gy - ry * gx);
            out.descent[4] -= gx / scale;
            out.descent[5] -= gy / scale;
            out.descent[6] -= gz / scale;
        }
    }
    return out;
}

// =========================================================================
// Grain families (see the file header).
// =========================================================================
bool is_color_conformer_grain(int grain) {
    return grain == 11;
}

bool is_color_start_grain(int grain) {
    return grain == 16;
}

// =========================================================================
// Prepared handles and pre-packed batches.
// =========================================================================
//
// Grain 11 in the mode-2 rescore only; the optimizer packs per work
// unit instead.
struct PreparedColorConformerBatches {
    // Each entry is the original pair/output row represented by the matching
    // lane in blocks[row / 8].
    std::vector<size_t> output_order;
    std::vector<ColorConformerBatch8> blocks;
    double stable_sort_seconds = 0.0;
    double pack_seconds = 0.0;
};

// Query fields are built at the query-preparation boundary and then passed
// back into repeated ordinary combo calls. The handle owns each bundle and
// snapshots the exact inputs that define those fields, so the optimizer can
// reject accidental reuse with a different query or interaction matrix.
struct PreparedColorFieldHandle {
    std::vector<std::shared_ptr<ColorFieldBundle>> bundles;
    // Legacy one-row view is retained for uniform broadcast callers. The
    // flattened row-major matrix is the binding used by per-query dispatch.
    std::vector<int> backend_codes;
    std::vector<int> backend_codes_by_query;
    std::vector<float> query_coords, interaction_r, interaction_p;
    std::vector<int32_t> query_types, query_n_atoms;
    std::array<py::ssize_t, 3> query_shape{};
    py::ssize_t query_types_rows = 0, query_types_width = 0;
    py::ssize_t query_n_atoms_size = 0;
    int n_features = 0;
    double spacing_angstrom = 0.5, padding_angstrom = 4.0;
    size_t bytes = 0;
    // Payload accounting includes native input snapshots and copied bundle
    // source/matrix arrays; allocator/object overhead remains outside this
    // estimate, like the Engine's other conservative query-context totals.
    size_t retained_bytes = 0;
    std::vector<size_t> bytes_by_target_type;
    std::vector<double> build_seconds_by_target_type;
    std::vector<double> pack_seconds_by_target_type;
    std::array<uint64_t, 4> bytes_by_backend{};
    std::array<double, 4> build_seconds_by_backend{};
    std::array<double, 4> pack_seconds_by_backend{};
    double build_seconds = 0.0, pack_seconds = 0.0;
    std::string half_decode_path = "not_used";
    const int coefficient_layout =
        ColorFieldBundle::coefficient_layout_version();
    const std::string node_quantization =
        ColorFieldBundle::node32_source_quantization();
    const float coordinate_scale = COORD_SCALE;
    // Native handles cannot cross a loaded DSO/class generation; these explicit
    // immutable identities additionally bind its layout and source precision.
    const int prefilter_version = 1;  // h*h/12 independently for every channel

    // Reuse requires the same layout, quantization, coordinate scale and
    // prefilter identities, the same shapes, backend codes and geometry, and
    // byte-identical query and matrix inputs (memcmp against the snapshots).
    bool matches(const py::array_t<float, py::array::c_style>& qc,
                 const py::array_t<int32_t, py::array::c_style>& qt,
                 const py::array_t<int32_t, py::array::c_style>& qn,
                 const py::array_t<float, py::array::c_style>& r,
                 const py::array_t<float, py::array::c_style>& p,
                 const std::vector<int>& codes_by_query, double spacing,
                 double padding) const {
        if (coefficient_layout !=
                ColorFieldBundle::coefficient_layout_version() ||
            node_quantization !=
                ColorFieldBundle::node32_source_quantization() ||
            coordinate_scale != COORD_SCALE || prefilter_version != 1 ||
            qc.ndim() != 3 || qt.ndim() != 2 || qn.ndim() != 1 ||
            r.ndim() != 2 || p.ndim() != 2 || qc.shape(0) != query_shape[0] ||
            qc.shape(1) != query_shape[1] || qc.shape(2) != query_shape[2] ||
            qt.shape(0) != query_types_rows ||
            qt.shape(1) != query_types_width ||
            qn.shape(0) != query_n_atoms_size || r.shape(0) != n_features ||
            r.shape(1) != n_features || p.shape(0) != n_features ||
            p.shape(1) != n_features ||
            codes_by_query != backend_codes_by_query ||
            spacing != spacing_angstrom || padding != padding_angstrom)
            return false;
        const auto same_bytes = [](const void* observed, const void* expected,
                                   size_t nbytes) {
            return nbytes == 0 || std::memcmp(observed, expected, nbytes) == 0;
        };
        return same_bytes(qc.data(), query_coords.data(),
                          size_t(qc.size()) * sizeof(float)) &&
               same_bytes(qt.data(), query_types.data(),
                          size_t(qt.size()) * sizeof(int32_t)) &&
               same_bytes(qn.data(), query_n_atoms.data(),
                          size_t(qn.size()) * sizeof(int32_t)) &&
               same_bytes(r.data(), interaction_r.data(),
                          size_t(r.size()) * sizeof(float)) &&
               same_bytes(p.data(), interaction_p.data(),
                          size_t(p.size()) * sizeof(float));
    }
};

// One ColorFieldBundle per query: its color features become typed field
// sources, with the flattened physical matrix and that query's per-target-type
// backend codes (0 analytic, 1 FP32 nodes, 2 FP16 nodes, 3 FP32 bricks).
// Spacing and padding are in physical Angstrom.
std::vector<std::shared_ptr<ColorFieldBundle>>
make_color_field_bundles(const std::vector<ColorConformer>& queries,
                         int query_stride, const Matrix& mat,
                         const std::vector<std::vector<int>>& backend_codes,
                         double spacing, double padding) {
    const int nf = mat.n;
    std::vector<double> coefficients(size_t(nf) * size_t(nf));
    std::vector<double> decays(size_t(nf) * size_t(nf));
    for (int ta = 0; ta < nf; ++ta) {
        for (int tb = 0; tb < nf; ++tb) {
            const size_t dst = size_t(ta) * size_t(nf) + size_t(tb);
            coefficients[dst] = mat.coeff[ta][tb];
            decays[dst] = mat.decay[ta][tb];
        }
    }
    std::vector<std::shared_ptr<ColorFieldBundle>> bundles(queries.size());
    for (size_t row = 0; row < queries.size(); ++row) {
        std::vector<ColorFieldSource> sources;
        const ColorConformer& query = queries[row];
        sources.reserve(size_t(query.n_color));
        for (int ta = 1; ta < nf; ++ta) {
            for (int i = 0; i < query.seg_count[ta]; ++i) {
                const float* x =
                    query.color_xyz +
                    size_t(query.seg_offset[ta] + i) * size_t(query_stride);
                ColorFieldSource source;
                source.xyz[0] = x[0];
                source.xyz[1] = x[1];
                source.xyz[2] = x[2];
                source.type = ta;
                sources.push_back(source);
            }
        }
        bundles[row] = std::make_shared<ColorFieldBundle>(
            sources, nf, coefficients.data(), decays.data(),
            backend_codes[row].data(), spacing, padding);
    }
    return bundles;
}

// Python-facing builder of PreparedColorFieldHandle. Backend codes are
// [n_features] (one policy for every query) or [n_query, n_features] (static
// per-query dispatch); type 0 must be analytic and at least one plane
// must select a field. Builds the bundles, snapshots every defining input for
// matches(), and totals bytes and build/pack seconds by target type and by
// backend.
std::shared_ptr<PreparedColorFieldHandle>
prepare_color_field_handle(py::array_t<float, py::array::c_style> qc,
                           py::array_t<int32_t, py::array::c_style> qt,
                           py::array_t<int32_t, py::array::c_style> qn,
                           py::array_t<float, py::array::c_style> r,
                           py::array_t<float, py::array::c_style> p,
                           py::object backend_codes_object, double spacing,
                           double padding) {
    if (qc.ndim() != 3 || qt.ndim() != 2 || qn.ndim() != 1 ||
        qt.shape(0) != qc.shape(0) || qt.shape(1) != qc.shape(1) ||
        qn.shape(0) != qc.shape(0) || (qc.shape(2) != 3 && qc.shape(2) != 4))
        throw std::invalid_argument(
            "prepared color fields need matching query arrays");
    Matrix mat(r, p);
    if (qt.shape(1) != qc.shape(1) || !std::isfinite(spacing) ||
        spacing <= 0.0 || !std::isfinite(padding) || padding < 0.0)
        throw std::invalid_argument("invalid prepared color field geometry");
    const py::array codes_array = py::array::ensure(backend_codes_object);
    const int nq = int(qc.shape(0));
    if (!codes_array ||
        !((codes_array.ndim() == 1 && codes_array.shape(0) == mat.n) ||
          (codes_array.ndim() == 2 && codes_array.shape(0) == nq &&
           codes_array.shape(1) == mat.n)))
        throw std::invalid_argument(
            "color field backend codes must be [n_features] or "
            "[n_query,n_features]");
    const py::module_ numpy = py::module_::import("numpy");
    if (!py::cast<bool>(numpy.attr("issubdtype")(codes_array.dtype(),
                                                 numpy.attr("integer"))))
        throw std::invalid_argument(
            "color field backend codes must be integers");
    std::vector<std::vector<int>> codes_by_query(
        size_t(nq), std::vector<int>(size_t(mat.n), 0));
    std::vector<int> codes_flat(size_t(nq) * size_t(mat.n), 0);
    std::vector<int> legacy_codes(size_t(mat.n), 0);
    bool any_selected = false;
    bool uniform_codes = true;
    for (int row = 0; row < nq; ++row) {
        for (int tb = 0; tb < mat.n; ++tb) {
            const py::object item =
                codes_array.ndim() == 1
                    ? codes_array[py::int_(tb)]
                    : codes_array[py::make_tuple(py::int_(row), py::int_(tb))];
            if (py::cast<bool>(item.attr("__lt__")(py::int_(0))) ||
                py::cast<bool>(item.attr("__gt__")(
                    py::int_(static_cast<int>(COLOR_FIELD_BRICK_FP32)))))
                throw std::invalid_argument(
                    "color field backend codes must be 0, 1, 2 or 3");
            const int code = py::cast<int>(item);
            if (tb == 0 && code != COLOR_FIELD_ANALYTIC)
                throw std::invalid_argument(
                    "color field target type zero must use backend 0");
            codes_by_query[size_t(row)][size_t(tb)] = code;
            codes_flat[size_t(row) * size_t(mat.n) + size_t(tb)] = code;
            if (row == 0)
                legacy_codes[size_t(tb)] = code;
            else
                uniform_codes &= legacy_codes[size_t(tb)] == code;
            any_selected |= tb > 0 && code != COLOR_FIELD_ANALYTIC;
        }
    }
    if (!any_selected)
        throw std::invalid_argument(
            "prepared color field handle needs a selected target plane");

    const int qs = int(qc.shape(2));
    const float* qbase = qc.data();
    const int32_t* tbase = qt.data();
    const int32_t* nbase = qn.data();
    std::vector<ColorConformer> queries(static_cast<size_t>(nq));
    for (int row = 0; row < nq; ++row) {
        if (nbase[row] < 0 || nbase[row] > qc.shape(1))
            throw std::invalid_argument(
                "prepared query atom count is outside its coordinate row");
        for (py::ssize_t atom = 0; atom < qt.shape(1); ++atom) {
            const int32_t type =
                tbase[size_t(row) * size_t(qt.shape(1)) + size_t(atom)];
            if (type < 0 || type >= mat.n)
                throw std::invalid_argument(
                    "prepared query type is outside the interaction matrix");
        }
        queries[size_t(row)] = build_color_view_from_types(
            qbase + size_t(row) * size_t(qc.shape(1)) * size_t(qs),
            tbase + size_t(row) * size_t(qt.shape(1)), nbase[row],
            int(qc.shape(1)), mat.n, qs);
    }
    auto handle = std::make_shared<PreparedColorFieldHandle>();
    handle->bundles = make_color_field_bundles(queries, qs, mat, codes_by_query,
                                               spacing, padding);
    if (uniform_codes) handle->backend_codes = legacy_codes;
    handle->backend_codes_by_query = codes_flat;
    handle->query_shape = {qc.shape(0), qc.shape(1), qc.shape(2)};
    handle->query_types_rows = qt.shape(0);
    handle->query_types_width = qt.shape(1);
    handle->query_n_atoms_size = qn.shape(0);
    handle->n_features = mat.n;
    handle->spacing_angstrom = spacing;
    handle->padding_angstrom = padding;
    handle->query_coords.assign(qc.data(), qc.data() + qc.size());
    handle->query_types.assign(qt.data(), qt.data() + qt.size());
    handle->query_n_atoms.assign(qn.data(), qn.data() + qn.size());
    handle->interaction_r.assign(r.data(), r.data() + r.size());
    handle->interaction_p.assign(p.data(), p.data() + p.size());
    handle->bytes_by_target_type.assign(size_t(mat.n), 0);
    handle->build_seconds_by_target_type.assign(size_t(mat.n), 0.0);
    handle->pack_seconds_by_target_type.assign(size_t(mat.n), 0.0);
    for (const auto& bundle : handle->bundles) {
        handle->bytes += bundle->field_bytes();
        handle->build_seconds += bundle->build_seconds();
        for (int tb = 1; tb < mat.n; ++tb) {
            const int backend = bundle->backend_code(tb);
            const size_t plane_bytes = bundle->field_bytes(tb);
            const double build_s = bundle->build_seconds(tb);
            const double pack_s = bundle->pack_seconds(tb);
            handle->bytes_by_target_type[size_t(tb)] += plane_bytes;
            handle->build_seconds_by_target_type[size_t(tb)] += build_s;
            handle->pack_seconds_by_target_type[size_t(tb)] += pack_s;
            handle->pack_seconds += pack_s;
            if (backend > COLOR_FIELD_ANALYTIC &&
                backend <= COLOR_FIELD_BRICK_FP32) {
                handle->bytes_by_backend[size_t(backend)] += plane_bytes;
                handle->build_seconds_by_backend[size_t(backend)] += build_s;
                handle->pack_seconds_by_backend[size_t(backend)] += pack_s;
            }
        }
        if (std::string(bundle->half_decode_path()) != "not_used")
            handle->half_decode_path = bundle->half_decode_path();
    }
    // Retained-payload estimate: field bytes, the input snapshots and codes,
    // each bundle's two flattened matrices, the bundle pointers and copied
    // sources.
    handle->retained_bytes =
        handle->bytes + size_t(qc.size()) * sizeof(float) +
        size_t(qt.size()) * sizeof(int32_t) +
        size_t(qn.size()) * sizeof(int32_t) + size_t(r.size()) * sizeof(float) +
        size_t(p.size()) * sizeof(float) + codes_flat.size() * sizeof(int) +
        size_t(nq) * size_t(mat.n) * size_t(mat.n) * 2 * sizeof(double) +
        size_t(nq) * size_t(mat.n) * sizeof(std::shared_ptr<ColorFieldBundle>);
    for (const auto& query : queries)
        handle->retained_bytes +=
            size_t(query.n_color) * sizeof(ColorFieldSource);
    return handle;
}

struct ComboColorFieldState {
    std::vector<int> backend_codes;
    std::vector<std::vector<int>> backend_codes_by_query;
    bool uniform_backend_codes = true;
    bool use_color_fields = false;
    std::shared_ptr<PreparedColorFieldHandle> prepared_handle;
    std::vector<std::shared_ptr<ColorFieldBundle>> bundles;
};

// Keep the experimental query-field policy and handle setup outside the large
// combo optimizer. Adding this setup inline moved unrelated shape/score code
// across GCC's inlining cliff under the codegen canary; this boundary changes
// no evaluator or pose arithmetic.
__attribute__((noinline)) ComboColorFieldState prepare_combo_color_field_state(
    int nf, int color_grain, const py::object& color_field_backends,
    double color_field_spacing, double color_field_padding,
    const py::object& prepared_color_fields,
    const py::array_t<float, py::array::c_style>& qc,
    const py::array_t<int32_t, py::array::c_style>& qt,
    const py::array_t<int32_t, py::array::c_style>& qn,
    const py::array_t<float, py::array::c_style>& r,
    const py::array_t<float, py::array::c_style>& p) {
    ComboColorFieldState state;
    state.backend_codes.assign(size_t(nf), 0);
    const int nq = int(qc.shape(0));
    state.backend_codes_by_query.assign(size_t(nq),
                                        std::vector<int>(size_t(nf), 0));
    if (!color_field_backends.is_none()) {
        const py::array codes = py::array::ensure(color_field_backends);
        if (!codes || !((codes.ndim() == 1 && codes.shape(0) == nf) ||
                        (codes.ndim() == 2 && codes.shape(0) == nq &&
                         codes.shape(1) == nf)))
            throw std::invalid_argument(
                "color_field_backends must be [n_features] or "
                "[n_query,n_features]");
        const py::module_ numpy = py::module_::import("numpy");
        const bool integer_codes = py::cast<bool>(
            numpy.attr("issubdtype")(codes.dtype(), numpy.attr("integer")));
        if (!integer_codes)
            throw std::invalid_argument(
                "color_field_backends must be an integer vector indexed by target type");
        for (int row = 0; row < nq; ++row) {
            for (int tb = 0; tb < nf; ++tb) {
                const py::object item =
                    codes.ndim() == 1
                        ? codes[py::int_(tb)]
                        : codes[py::make_tuple(py::int_(row), py::int_(tb))];
                if (py::cast<bool>(item.attr("__lt__")(py::int_(0))) ||
                    py::cast<bool>(item.attr("__gt__")(
                        py::int_(static_cast<int>(COLOR_FIELD_BRICK_FP32)))))
                    throw std::invalid_argument(
                        "color field backend codes must be 0, 1, 2 or 3");
                const int code = py::cast<int>(item);
                if (tb == 0 && code != COLOR_FIELD_ANALYTIC)
                    throw std::invalid_argument(
                        "color field target type zero is reserved and must use backend 0");
                state.backend_codes_by_query[size_t(row)][size_t(tb)] = code;
                if (row == 0)
                    state.backend_codes[size_t(tb)] = code;
                else
                    state.uniform_backend_codes &=
                        state.backend_codes[size_t(tb)] == code;
                state.use_color_fields |=
                    tb > 0 && code != COLOR_FIELD_ANALYTIC;
            }
        }
    }
    if (state.use_color_fields && color_grain != 11 && color_grain != 16)
        throw std::invalid_argument(
            "color_field_backends currently requires color_grain=11 or "
            "start-lane grain 16 so unselected and out-of-grid "
            "planes use the prepared analytic evaluator");
    if (state.use_color_fields &&
        (!std::isfinite(color_field_spacing) || color_field_spacing <= 0.0 ||
         !std::isfinite(color_field_padding) || color_field_padding < 0.0))
        throw std::invalid_argument("invalid color field spacing or padding");
    if (!prepared_color_fields.is_none()) {
        if (!state.use_color_fields)
            throw std::invalid_argument(
                "prepared_color_fields requires selected color_field_backends");
        state.prepared_handle =
            py::cast<std::shared_ptr<PreparedColorFieldHandle>>(
                prepared_color_fields);
        if (!state.prepared_handle->matches(
                qc, qt, qn, r, p,
                [&]() {
                    std::vector<int> flattened;
                    flattened.reserve(size_t(nq) * size_t(nf));
                    for (const auto& row : state.backend_codes_by_query)
                        flattened.insert(flattened.end(), row.begin(),
                                         row.end());
                    return flattened;
                }(),
                color_field_spacing, color_field_padding))
            throw std::invalid_argument(
                "prepared_color_fields do not match this query, interaction matrix, backends, or geometry");
    }
    return state;
}

// Adopt a prepared handle's bundles or build them in-call, then account bytes
// and build/pack seconds per target type and backend. Prepared bundles report
// zero in-call build/pack time: that cost was paid at query preparation.
__attribute__((noinline)) void build_combo_color_field_state(
    ComboColorFieldState& state, const std::vector<ColorConformer>& queries,
    int query_stride, const Matrix& mat, double spacing, double padding,
    ComboStats& stats) {
    state.bundles.resize(queries.size());
    if (!state.use_color_fields) return;
    if (state.prepared_handle) {
        if (state.prepared_handle->bundles.size() != queries.size())
            throw std::invalid_argument(
                "prepared_color_fields query count does not match optimizer input");
        state.bundles = state.prepared_handle->bundles;
    } else {
        state.bundles = make_color_field_bundles(queries, query_stride, mat,
                                                 state.backend_codes_by_query,
                                                 spacing, padding);
    }
    for (size_t row = 0; row < queries.size(); ++row) {
        const ColorFieldBundle& fields = *state.bundles[row];
        stats.color_field_bytes += uint64_t(fields.field_bytes());
        for (int tb = 1; tb < mat.n; ++tb) {
            const int backend = fields.backend_code(tb);
            stats.color_field_bytes_by_target_type[size_t(tb)] +=
                uint64_t(fields.field_bytes(tb));
            const double build_s =
                state.prepared_handle ? 0.0 : fields.build_seconds(tb);
            const double pack_s =
                state.prepared_handle ? 0.0 : fields.pack_seconds(tb);
            stats.color_field_build_s += build_s;
            stats.color_field_build_s_by_target_type[size_t(tb)] += build_s;
            stats.color_field_pack_s += pack_s;
            stats.color_field_pack_s_by_target_type[size_t(tb)] += pack_s;
            if (backend > COLOR_FIELD_ANALYTIC &&
                backend <= COLOR_FIELD_BRICK_FP32) {
                stats.color_field_bytes_by_backend[size_t(backend)] +=
                    uint64_t(fields.field_bytes(tb));
                stats.color_field_build_s_by_backend[size_t(backend)] +=
                    build_s;
                stats.color_field_pack_s_by_backend[size_t(backend)] += pack_s;
            }
            if (fields.field_active(tb))
                ++stats.color_field_active_queries_by_target_type[size_t(tb)];
        }
    }
}

// Rescore packing for grain 11: pairs grouped into
// COLOR_CONFORMER_LANES-wide batches in input order; output_order maps each
// lane back to its row.
PreparedColorConformerBatches prepare_color_conformer_batches(
    const std::vector<ColorConformer>& queries,
    const std::vector<ColorConformer>& libraries, const int32_t* pair_queries,
    const int32_t* pair_libraries, size_t n_pairs, int query_stride,
    int library_stride, int n_features) {
    PreparedColorConformerBatches prepared;
    if (n_pairs == 0) return prepared;
    if (n_pairs > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("conformer pair batch exceeds integer limits");

    std::vector<ColorConformerPairInput> pairs(n_pairs);
    for (size_t row = 0; row < n_pairs; ++row) {
        const size_t query = size_t(pair_queries[row]);
        const size_t library = size_t(pair_libraries[row]);
        pairs[row] = {&queries[query], &libraries[library], query_stride,
                      library_stride};
    }
    prepared.output_order.resize(n_pairs);
    for (size_t row = 0; row < n_pairs; ++row)
        prepared.output_order[row] = row;

    prepared.blocks.reserve((n_pairs + COLOR_CONFORMER_LANES - 1) /
                            COLOR_CONFORMER_LANES);
    for (size_t first = 0; first < n_pairs; first += COLOR_CONFORMER_LANES) {
        const int valid =
            int(std::min(size_t(COLOR_CONFORMER_LANES), n_pairs - first));
        std::vector<ColorConformerPairInput> block(static_cast<size_t>(valid));
        for (int lane = 0; lane < valid; ++lane)
            block[size_t(lane)] =
                pairs[prepared.output_order[first + size_t(lane)]];
        const auto pack_started = Clock::now();
        prepared.blocks.push_back(
            pack_color_conformer_batch8(block.data(), valid, n_features));
        prepared.pack_seconds +=
            std::chrono::duration<double>(Clock::now() - pack_started).count();
    }
    return prepared;
}

// One start-lane work unit: up to two pairs, each expanded to four
// independent start lanes [A0 A1 A2 A3 | B0 B1 B2 B3].
struct PreparedColorStartBlock {
    int n_pairs = 0;
    std::array<size_t, 2> rows{};
    std::array<ColorConformerPairInput, 2> source_pairs{};
};

// =========================================================================
// The combo optimizer.
// =========================================================================
//
// Mode 3 (doc/COLOR_CONTRACT.md section 7): optimize every pair on the mixed
// objective and report exact scores at its winning pose. For each pair and each
// start:
//
//   1. compose the current state with the start, q_eff = q * conj(q_start),
//      and under the shared contract prepare one FP32 pose;
//   2. shape value and descent from the selected shape evaluator;
//   3. color value and descent from the selected color grain or field;
//   4. mixed_descent, then one adagrad_exp_step (t and lr_t in reduced units);
//
// repeated `steps` times from the identity; then the start's converged pose is
// scored and the best start (strict >, so ties keep the earliest start) is
// reported. final_start_policy "all_start" scores every start's converged pose
// exactly; "winner" ("default" resolves to it when R3_FINAL_WINNER_ONLY is
// 1) evaluates the SEARCH objective at each converged pose and accurately
// scores only the winner. Winner-only applies to the prepared canonical
// four-start route with field shape and analytic color; every other route keeps
// all-start scoring. The measured cost of that trade: held-out conformer
// K100 0.9925 (doc/COLOR_CONTRACT.md, final-start selection policy).
//
// Phases below, in order: option validation; pose contract and prepared
// handles; library decoding by workers; shape evaluators and library
// workspaces; self overlaps (prepared, else the shared query cache, else
// in-call); output and sink setup; the four dispatch branches; the details
// dictionary. Returns (dense [n_pairs, N_SCORE_COLUMNS] float32 or None,
// details). With a row_sink each finished row is handed to the sink on the
// worker that produced it.
py::tuple optimize_scalar_combo(
    py::array_t<float, py::array::c_style> qc,
    py::array_t<int32_t, py::array::c_style> qt,
    py::array_t<int32_t, py::array::c_style> qn,
    py::array_t<float, py::array::c_style> lc,
    py::array_t<uint8_t, py::array::c_style> lcnt,
    py::array_t<int32_t, py::array::c_style> pq,
    py::array_t<int32_t, py::array::c_style> pl,
    py::array_t<float, py::array::c_style> r,
    py::array_t<float, py::array::c_style> p, int steps, float lr_q, float lr_t,
    int start_mode, float mixing, bool with_census, bool with_timers,
    int color_grain, py::object color_field_backends,
    double color_field_spacing, double color_field_padding,
    py::object prepared_color_fields, int shape_backend, int shape_grain,
    int shape_width, float shape_field_spacing, float shape_field_padding,
    py::object prepared_shape_evaluators, int threads, bool with_worker_timers,
    bool require_worker_affinity, py::object prepared_shared_pose_queries,
    const std::string& pose_contract, bool use_color_invariant_plan,
    bool reuse_exact_final_scratch, py::object prepared_combo_self_overlaps,
    const std::string& prepared_combo_self_overlap_digest,
    std::shared_ptr<ComboExecutor> executor,
    std::shared_ptr<ComboRowSink> row_sink, bool return_dense,
    bool with_library_self_timers, bool factor_canonical_torque,
    const std::string& optimizer_exp_mode, bool canonical_color_kernel,
    const std::string& final_start_policy) {
    const bool winner_policy = resolve_final_start_policy(final_start_policy);
    const auto requested_exponential =
        parse_optimizer_exp_mode(optimizer_exp_mode);
    bool unclamped_is_safe = false;
    // Unclamped raw-bit exp (the public default) lets
    // out-of-range conversions raise INVALID, so it is allowed only when this
    // one caller thread runs every work unit and its MXCSR (x86) or FPCR (ARM)
    // masks that exception. Otherwise resolve_optimizer_exp_mode falls back to
    // the clamped "rawbits". Any non-baseline exp needs shared-pose grain 16.
    if (requested_exponential == OptimizerExpMode::RawBitsUnclamped) {
        // Public CPU workers invoke one native thread per block. Check on that
        // executing thread, once per call, not in an optimizer/pair loop.
        // Internal executors/threads have independently owned FP state; use
        // guarded exp there rather than assuming the caller's mask applies.
#if defined(__AVX2__)
        unclamped_is_safe = threads == 1 && !executor &&
                            ((_mm_getcsr() & _MM_MASK_INVALID) != 0);
#elif defined(__aarch64__)
        unclamped_is_safe =
            threads == 1 && !executor && neon_invalid_exceptions_masked();
#endif
    }
    const auto optimizer_exponential =
        resolve_optimizer_exp_mode(requested_exponential, unclamped_is_safe);
    if (optimizer_exponential != OptimizerExpMode::Baseline &&
        (color_grain != 16 || pose_contract != "shared_fp32_reduced_v1"))
        throw std::invalid_argument(
            "rawbits requires shared-pose combo color_grain=16");
    // Serialize the complete native scoring call, not merely each worker
    // submission: library and target preparation must not interleave between
    // concurrent callers that share this explicitly owned executor. Release
    // the GIL while waiting so the active caller can finish its Python wrapper.
    std::unique_lock<std::mutex> executor_call_lock;
    if (executor) {
        {
            py::gil_scoped_release nogil;
            executor_call_lock = executor->acquire_call();
        }
        if (executor->is_closed())
            throw std::runtime_error("combo executor is closed");
    }
    // Option validation. Census and phase timers need threads == 1 and dense
    // output; worker timers and required affinity need threads >= 2; sink-only
    // output supports the retained grains 0, 11 and 16 only.
    if (qc.ndim() != 3 || qt.ndim() != 2 || qt.shape(0) != qc.shape(0) ||
        qt.shape(1) != qc.shape(1))
        throw std::invalid_argument(
            "query types must match query coordinate rows and width");
    if (steps < 0 || start_count(start_mode) < 0 || !std::isfinite(mixing) ||
        mixing < 0.0f || mixing > 1.0f)
        throw std::invalid_argument("invalid scalar combo schedule or mixing");
    if (!return_dense && color_grain != 0 && color_grain != 11 &&
        color_grain != 16)
        throw std::invalid_argument(
            "sink-only combo output currently supports retained color grains "
            "0, 11, and 16");
    require_supported_color_grain(color_grain);
    if (threads < 1) throw std::invalid_argument("threads must be >= 1");
    if (!return_dense && !row_sink)
        throw std::invalid_argument(
            "sink-only combo scoring requires a native row_sink");
    if (!return_dense && (with_census || with_timers || with_worker_timers))
        throw std::invalid_argument(
            "sink-only combo scoring cannot return census, phase timers, or "
            "worker timers; use dense output for diagnostics");
    if (executor && size_t(threads) != executor->worker_count())
        throw std::invalid_argument(
            "combo executor worker count does not match threads");
    if (executor && executor->is_closed())
        throw std::runtime_error("combo executor is closed");
    if (threads > 1 && (with_census || with_timers))
        throw std::invalid_argument(
            "parallel combo calls do not support census or phase timers; "
            "count or time with threads=1, then time with native workers");
    if (with_worker_timers && threads < 2)
        throw std::invalid_argument("worker timers require threads >= 2");
    if (require_worker_affinity && threads < 2)
        throw std::invalid_argument(
            "required worker affinity applies only to parallel combo calls");
    if (with_worker_timers && (with_census || with_timers))
        throw std::invalid_argument(
            "worker timers are a separate diagnostic from census and phase timers");
    // Shape evaluator configuration (the fast shape evaluators): the legal
    // (backend, grain, width) combinations. Trajectory shape batches the lanes
    // of one block, so it also needs a color grain whose blocks share one
    // query.
    if (shape_backend < int(ShapeBackend::ScalarAnalytic) ||
        shape_backend > int(ShapeBackend::PipelinedField) ||
        shape_grain < int(ShapeGrain::Scalar) ||
        shape_grain > int(ShapeGrain::Trajectory) ||
        (shape_width != 1 && shape_width != 8) ||
        !std::isfinite(shape_field_spacing) ||
        !std::isfinite(shape_field_padding) || shape_field_spacing < 0.0f ||
        shape_field_padding < 0.0f)
        throw std::invalid_argument("invalid combo shape evaluator config");
    ShapeEvaluatorConfig shape_config;
    shape_config.backend = ShapeBackend(shape_backend);
    shape_config.grain = ShapeGrain(shape_grain);
    shape_config.width = shape_width;
    shape_config.field_spacing_angstrom = shape_field_spacing;
    shape_config.field_padding_angstrom = shape_field_padding;
    if (shape_config.backend == ShapeBackend::ScalarAnalytic &&
        (shape_config.grain != ShapeGrain::Scalar || shape_config.width != 1))
        throw std::invalid_argument(
            "scalar analytic shape requires grain=scalar and width=1");
    if (shape_config.backend == ShapeBackend::AnalyticSimd &&
        (shape_config.width != 8 ||
         (shape_config.grain != ShapeGrain::QueryAtom &&
          shape_config.grain != ShapeGrain::LibraryAtom &&
          shape_config.grain != ShapeGrain::Trajectory)))
        throw std::invalid_argument(
            "analytic SIMD shape requires width=8 and an atom/trajectory grain");
    if (shape_config.backend == ShapeBackend::ScalarField &&
        (shape_config.grain != ShapeGrain::Scalar || shape_config.width != 1))
        throw std::invalid_argument(
            "scalar field shape requires grain=scalar and width=1");
    if ((shape_config.backend == ShapeBackend::PairedChannelField ||
         shape_config.backend == ShapeBackend::PipelinedField) &&
        (shape_config.grain != ShapeGrain::LibraryAtom ||
         shape_config.width != 8))
        throw std::invalid_argument(
            "vector field shape requires grain=library_atom and width=8");
    if (shape_config.backend != ShapeBackend::ScalarAnalytic &&
        shape_config.backend != ShapeBackend::AnalyticSimd &&
        (!(shape_field_spacing > 0.0f) || !(shape_field_padding > 0.0f)))
        throw std::invalid_argument(
            "field shape requires positive spacing and padding");
    if (shape_config.grain == ShapeGrain::Trajectory &&
        (shape_config.backend != ShapeBackend::AnalyticSimd ||
         (color_grain != 11 && color_grain != 16) ||
         (color_grain == 16 && start_mode != 1)))
        throw std::invalid_argument(
            "trajectory shape currently requires analytic SIMD with color "
            "grain 11 or standard-start grain 16");
    const int nf = int(r.shape(0));
    const int qs = int(qc.shape(2)), ls = int(lc.shape(2));
    const size_t query_rows = size_t(qc.shape(0));
    const size_t query_capacity = size_t(qc.shape(1));
    const size_t library_rows = size_t(lc.shape(0));
    const size_t library_capacity = size_t(lc.shape(1));
    const size_t pair_rows = size_t(pq.size());
    const float *qbase = qc.data(), *lbase = lc.data();
    const int32_t *tbase = qt.data(), *nbase = qn.data();
    const uint8_t* cbase = lcnt.data();
    const int32_t *qi = pq.data(), *li = pl.data();
    const int starts = start_count(start_mode);
    // The optimizer's translation state is reduced (t' = s * t), so the
    // physical translation learning rate is scaled once here.
    const float lr_t_scaled = lr_t * COORD_SCALE;
    Matrix mat(r, p);
    const bool shared_pose_mode = pose_contract == "shared_fp32_reduced_v1";
    if (pose_contract != "legacy_physical_fp64_pose" && !shared_pose_mode)
        throw std::invalid_argument("unknown combo pose contract");
    // Prepared self overlaps: shared contract only, and only with the
    // digest the handle was built for. Either half (shape, color) may be
    // absent.
    PreparedComboSelfOverlapsHandle prepared_self_overlaps;
    if (!prepared_combo_self_overlaps.is_none()) {
        if (!shared_pose_mode)
            throw std::invalid_argument(
                "prepared combo self-overlaps require shared_fp32_reduced_v1");
        prepared_self_overlaps = prepared_combo_self_overlaps
                                     .cast<PreparedComboSelfOverlapsHandle>();
        if (!prepared_self_overlaps ||
            prepared_combo_self_overlap_digest.empty() ||
            prepared_self_overlaps->input_digest !=
                prepared_combo_self_overlap_digest)
            throw std::invalid_argument(
                "prepared combo self-overlaps do not match normalized inputs");
        if (prepared_self_overlaps->n_features != nf ||
            (prepared_self_overlaps->shape_prepared &&
             (prepared_self_overlaps->shape_query.size() != query_rows ||
              prepared_self_overlaps->shape_library.size() != library_rows)) ||
            (prepared_self_overlaps->color_prepared &&
             (prepared_self_overlaps->color_query.size() != query_rows ||
              prepared_self_overlaps->color_library.size() != library_rows)) ||
            (!prepared_self_overlaps->shape_prepared &&
             !prepared_self_overlaps->color_prepared))
            throw std::invalid_argument(
                "prepared combo self-overlap rows or feature width do not match inputs");
    } else if (!prepared_combo_self_overlap_digest.empty()) {
        throw std::invalid_argument(
            "prepared self-overlap digest was supplied without a handle");
    }
    PreparedShapeEvaluatorSetHandle explicit_shape_evaluators;
    if (!prepared_shape_evaluators.is_none())
        explicit_shape_evaluators =
            prepared_shape_evaluators.cast<PreparedShapeEvaluatorSetHandle>();
    if (explicit_shape_evaluators) {
        shape_config.storage = explicit_shape_evaluators->config.storage;
        shape_config.interpolation =
            explicit_shape_evaluators->config.interpolation;
    }
    // Shared-pose query preparation: a caller-owned
    // PreparedSharedPoseQuerySet is reused when it matches these queries, shape
    // config and matrix; otherwise one is built below, once the shape
    // evaluators exist. Its shape storage/interpolation settings override the
    // call's.
    PreparedSharedPoseQuerySetHandle shared_pose_queries;
    bool shared_shape_preparation_in_call = false;
    double shared_shape_preparation_seconds = 0.0;
    if (shared_pose_mode) {
        if (color_grain == 16 && start_mode != 1)
            throw std::invalid_argument(
                "shared-pose color grain 16 requires the standard four-start mode");
        if (!prepared_shared_pose_queries.is_none()) {
            shared_pose_queries = prepared_shared_pose_queries
                                      .cast<PreparedSharedPoseQuerySetHandle>();
            if (!explicit_shape_evaluators && shared_pose_queries) {
                shape_config.storage =
                    shared_pose_queries->shape_config.storage;
                shape_config.interpolation =
                    shared_pose_queries->shape_config.interpolation;
            }
            bool shared_queries_match = false;
            {
                py::gil_scoped_release nogil;
                shared_queries_match =
                    shared_pose_queries &&
                    shared_pose_queries->matches(
                        qbase, int(query_rows), int(query_capacity), qs, tbase,
                        nbase, shape_config, mat.view());
            }
            if (!shared_queries_match)
                throw std::invalid_argument(
                    "prepared shared-pose queries do not match inputs/config");
            if (!shared_pose_queries->color_plan ||
                shared_pose_queries->color_plan->n_features() != nf)
                throw std::invalid_argument(
                    "prepared shared-pose color plan has wrong feature count");
            if (explicit_shape_evaluators) {
                if (explicit_shape_evaluators !=
                    shared_pose_queries->shape_evaluators)
                    throw std::invalid_argument(
                        "shared-pose and explicit shape evaluator handles differ");
            }
        }
    } else if (!prepared_shared_pose_queries.is_none()) {
        throw std::invalid_argument(
            "prepared_shared_pose_queries requires shared_fp32_reduced_v1");
    }
    // Start-lane colors retain the conformer grain's analytic fallback
    // contract.
    // Pass its established validator grain to the unchanged noinline setup
    // boundary so the codegen canary remains byte-for-byte intact.
    const int field_validation_grain =
        is_color_start_grain(color_grain) ? 11 : color_grain;
    ComboColorFieldState color_field_state = prepare_combo_color_field_state(
        nf, field_validation_grain, color_field_backends, color_field_spacing,
        color_field_padding, prepared_color_fields, qc, qt, qn, r, p);
    const bool use_color_fields = color_field_state.use_color_fields;
    const std::vector<int>& field_backend_codes =
        color_field_state.backend_codes;
    const std::vector<std::vector<int>>& field_backend_codes_by_query =
        color_field_state.backend_codes_by_query;
    const bool uniform_field_backend_codes =
        color_field_state.uniform_backend_codes;
    const std::shared_ptr<PreparedColorFieldHandle>& prepared_field_handle =
        color_field_state.prepared_handle;
    ComboStats stats;
    std::vector<double> worker_elapsed_seconds;
    std::vector<ComboWorkerPlacement> worker_placements;
    std::vector<uint64_t> scoring_claims_by_worker;
    if (with_worker_timers) {
        const size_t workers = std::min<size_t>(
            size_t(threads),
            combo_work_units(color_grain, start_mode, pair_rows));
        worker_elapsed_seconds.assign(workers, 0.0);
        worker_placements.resize(workers);
        scoring_claims_by_worker.assign(workers, 0);
    }
    stats.census = with_census;
    stats.timers = with_timers;
    stats.color_field_bytes_by_target_type.assign(size_t(nf), 0);
    stats.color_field_build_s_by_target_type.assign(size_t(nf), 0.0);
    stats.color_field_pack_s_by_target_type.assign(size_t(nf), 0.0);
    stats.color_field_active_queries_by_target_type.assign(size_t(nf), 0);
    std::vector<ColorConformer> qv;
    PaddedColorConformers lv;
    PaddedShapeConformers shape_library_views;
    std::vector<float> qshape;
    // Library shape rows: three floats per atom slot, rounded up to 16 floats
    // (one 64-byte line) per row, with overflow checked before allocating.
    const size_t library_row_floats = library_capacity * 3;
    if (library_row_floats > std::numeric_limits<size_t>::max() - 15)
        throw std::overflow_error("combo library workspace row is too large");
    const size_t library_row_stride = (library_row_floats + 15) & ~size_t(15);
    if (library_row_stride != 0 &&
        library_rows >
            (std::numeric_limits<size_t>::max() - 16) / library_row_stride)
        throw std::overflow_error("combo library workspace is too large");
    std::unique_ptr<float[]> lshape_storage;
    float* lshape = nullptr;
    // Each prepared coordinate row begins on its own cache line. The spare
    // 64-byte tail lets us align the backing array without a custom allocator.
    // Default-initialize the fundamental array without zeroing it on the
    // caller thread. Its owning library-preparation workers first-touch the
    // actual coordinate rows as they copy them below.
    {
        py::gil_scoped_release nogil;
        qv.resize(query_rows);
        lv.rows.resize(library_rows);
        shape_library_views.rows.resize(library_rows);
        qshape.resize(query_rows * query_capacity * 3);
        lshape_storage.reset(new float[library_rows * library_row_stride + 16]);
        const uintptr_t lshape_address =
            (reinterpret_cast<uintptr_t>(lshape_storage.get()) + 63) &
            ~uintptr_t(63);
        lshape = reinterpret_cast<float*>(lshape_address);
        for (size_t row = 0; row < qv.size(); ++row) {
            const float* src = qbase + row * query_capacity * size_t(qs);
            qv[row] = build_color_view_from_types(
                src, tbase + row * query_capacity, nbase[row],
                int(query_capacity), nf, qs);
            for (size_t j = 0; j < query_capacity; ++j)
                for (int k = 0; k < 3; ++k)
                    qshape[(row * query_capacity + j) * 3 + size_t(k)] =
                        COORD_SCALE * src[j * size_t(qs) + size_t(k)];
        }
    }
    // Decode only library rows that some pair references; workers decode them
    // in parallel and first-touch their own rows.
    std::vector<uint8_t> library_row_used;
    std::vector<size_t> used_library_rows;
    {
        py::gil_scoped_release nogil;
        library_row_used.assign(lv.size(), 0);
        for (size_t row = 0; row < pair_rows; ++row) {
            if (qi[row] < 0 || size_t(qi[row]) >= qv.size() || li[row] < 0 ||
                size_t(li[row]) >= lv.size())
                throw std::invalid_argument("combo pair row is out of range");
            library_row_used[size_t(li[row])] = 1;
        }
        used_library_rows.reserve(library_row_used.size());
        for (size_t row = 0; row < library_row_used.size(); ++row)
            if (library_row_used[row]) used_library_rows.push_back(row);
    }
    run_combo_workers(
        used_library_rows.size(), threads,
        [&](int, size_t used_index) {
            const size_t row = used_library_rows[used_index];
            const float* src = lbase + row * library_capacity * size_t(ls);
            lv[row] = build_color_view_from_counts(
                src, cbase + row * size_t(nf), nf, ls);
            const float* xyz = library_capacity == 0
                                   ? nullptr
                                   : lshape + row * library_row_stride;
            shape_library_views[row] = {xyz, lv[row].n_real};
            for (size_t j = 0; j < library_capacity; ++j)
                for (int k = 0; k < 3; ++k)
                    lshape[row * library_row_stride + j * 3 + size_t(k)] =
                        src[j * size_t(ls) + size_t(k)];
        },
        nullptr, nullptr, require_worker_affinity, executor.get(), 1,
        executor ? executor->claim_chunk_rows() : 256);
    // Automatic shared-pose preparation: reuse explicit
    // shape evaluators if supplied, else build them, then prescale the query
    // color and build the reduced color plan.
    if (shared_pose_mode && !shared_pose_queries) {
        PreparedShapeEvaluatorSetHandle shape_inputs;
        if (explicit_shape_evaluators) {
            shape_inputs = explicit_shape_evaluators;
            bool shape_inputs_match = false;
            {
                py::gil_scoped_release nogil;
                shape_inputs_match = shape_inputs->matches(
                    shape_config, qshape.data(), int(query_capacity), nbase,
                    qv.size());
            }
            if (!shape_inputs_match)
                throw std::invalid_argument(
                    "prepared shape evaluators do not match this query/config set");
        } else {
            {
                py::gil_scoped_release nogil;
                shape_inputs = prepare_shape_evaluator_set(
                    qshape.data(), int(query_capacity), nbase, qv.size(),
                    shape_config);
            }
            shared_shape_preparation_in_call = true;
            if (shape_inputs)
                shared_shape_preparation_seconds =
                    shape_inputs->preparation_seconds;
        }
        if (!shape_inputs)
            throw std::invalid_argument(
                "shared-pose auto preparation needs shape evaluators");
        {
            py::gil_scoped_release nogil;
            const ColorInvariantPlan color_plan(mat.view());
            shared_pose_queries = prepare_shared_pose_query_set(
                qbase, int(query_rows), int(query_capacity), qs, tbase, nbase,
                shape_inputs, color_plan);
        }
        shared_pose_queries->preparation_seconds +=
            shared_shape_preparation_seconds;
    }
    // Reduced query color views and the reduced matrix (d' = d/(s*s)) used by
    // every shared-pose color consumer below.
    std::vector<ColorConformer> shared_qv;
    Matrix shared_reduced_matrix(mat);
    if (shared_pose_mode) {
        py::gil_scoped_release nogil;
        shared_qv.reserve(shared_pose_queries->reduced_query_colors.size());
        for (const ReducedColorConformer& query :
             shared_pose_queries->reduced_query_colors)
            shared_qv.push_back(query.features);
        shared_reduced_matrix =
            reduced_color_matrix(mat, *shared_pose_queries->color_plan);
    }
    std::vector<Conformer> shape_query_views;
    {
        py::gil_scoped_release nogil;
        shape_query_views.resize(qv.size());
        for (size_t row = 0; row < qv.size(); ++row) {
            const float* xyz = query_capacity == 0
                                   ? nullptr
                                   : qshape.data() + row * query_capacity * 3;
            shape_query_views[row] = {xyz, qv[row].n_real};
        }
    }
    // Shape evaluators: the shared-pose set, an explicit caller set, or an
    // in-call build for non-scalar backends. Scalar analytic shape needs none;
    // it calls get_gradient directly.
    PreparedShapeEvaluatorSetHandle shape_evaluator_set;
    double shape_query_preparation_seconds = 0.0;
    bool shape_query_preparation_in_call = false;
    if (shared_pose_mode) {
        shape_evaluator_set = shared_pose_queries->shape_evaluators;
        bool shape_evaluator_match = false;
        {
            py::gil_scoped_release nogil;
            shape_evaluator_match = shape_evaluator_set &&
                                    shape_evaluator_set->matches(
                                        shape_config, qshape.data(),
                                        int(query_capacity), nbase, qv.size());
        }
        if (!shape_evaluator_match)
            throw std::invalid_argument(
                "prepared shared-pose shape evaluators do not match queries");
        shape_query_preparation_seconds =
            shape_evaluator_set->preparation_seconds;
        shape_query_preparation_in_call = shared_shape_preparation_in_call;
    } else if (explicit_shape_evaluators) {
        shape_evaluator_set = explicit_shape_evaluators;
        bool shape_evaluator_match = false;
        {
            py::gil_scoped_release nogil;
            shape_evaluator_match = shape_evaluator_set->matches(
                shape_config, qshape.data(), int(query_capacity), nbase,
                qv.size());
        }
        if (!shape_evaluator_match)
            throw std::invalid_argument(
                "prepared shape evaluators do not match this query/config set");
        shape_query_preparation_seconds =
            shape_evaluator_set->preparation_seconds;
    } else if (shape_config.backend != ShapeBackend::ScalarAnalytic) {
        const auto prep_started = Clock::now();
        {
            py::gil_scoped_release nogil;
            shape_evaluator_set =
                prepare_shape_evaluator_set(qshape.data(), int(query_capacity),
                                            nbase, qv.size(), shape_config);
        }
        shape_query_preparation_seconds =
            std::chrono::duration<double>(Clock::now() - prep_started).count();
        shape_query_preparation_in_call = true;
    }
    // Library-side shape workspaces, for the backends that pack the target
    // side, built in parallel once per referenced library row.
    const bool shape_needs_library_workspace =
        shape_evaluator_set &&
        ((shape_config.backend == ShapeBackend::AnalyticSimd &&
          shape_config.grain == ShapeGrain::LibraryAtom) ||
         shape_config.backend == ShapeBackend::PipelinedField);
    PaddedShapeLibraryWorkspaces shape_library_workspaces;
    if (shape_needs_library_workspace) {
        {
            py::gil_scoped_release nogil;
            shape_library_workspaces.rows.resize(lv.size());
        }
        const auto prep_started = Clock::now();
        run_combo_workers(
            used_library_rows.size(), threads,
            [&](int worker, size_t used_index) {
                const size_t row = used_library_rows[used_index];
                const Conformer& library = shape_library_views[row];
                ShapeLibraryWorkspace workspace;
                if (shape_config.backend == ShapeBackend::AnalyticSimd) {
                    workspace = prepare_analytic_shape_library_workspace(
                        library, shape_config.grain, shape_config.width);
                } else {
                    workspace = prepare_field_shape_library_workspace(
                        library, shape_config.backend, shape_config.width);
                }
                if (!workspace.matches(library))
                    throw std::invalid_argument(
                        "shape target workspace does not match its source conformer");
                shape_library_workspaces[row] = std::move(workspace);
            },
            nullptr, nullptr, require_worker_affinity, executor.get(), 1,
            executor ? executor->claim_chunk_rows() : 256);
        {
            py::gil_scoped_release nogil;
            for (size_t row : used_library_rows)
                stats.shape_workspace_bytes +=
                    shape_library_workspaces[row].prepared_bytes;
        }
        stats.shape_target_workspace_prepare_s =
            std::chrono::duration<double>(Clock::now() - prep_started).count();
    }
    // Phase-timer sinks per grain family; null when timers are off.
    ColorConformerPhaseTimes conformer_phase_times{
        with_timers ? &stats.color_conformer_pose_transform_s : nullptr,
        with_timers ? &stats.color_conformer_matrix_screen_s : nullptr,
        with_timers ? &stats.color_conformer_scalar_fallback_s : nullptr,
        with_timers ? &stats.color_conformer_pose_soa_pack_s : nullptr,
        with_timers ? &stats.color_conformer_vector_kernel_s : nullptr,
        with_timers ? &stats.color_conformer_field_sample_s : nullptr};
    ColorConformerPhaseTimes start_schedule_phase_times{
        with_timers ? &stats.color_start_pose_transform_s : nullptr,
        with_timers ? &stats.color_start_matrix_screen_s : nullptr,
        with_timers ? &stats.color_start_scalar_fallback_s : nullptr,
        with_timers ? &stats.color_start_pose_soa_pack_s : nullptr,
        with_timers ? &stats.color_start_vector_kernel_s : nullptr,
        with_timers ? &stats.color_start_field_sample_s : nullptr};
    {
        py::gil_scoped_release nogil;
        build_combo_color_field_state(color_field_state, qv, qs, mat,
                                      color_field_spacing, color_field_padding,
                                      stats);
    }
    const std::vector<std::shared_ptr<ColorFieldBundle>>& field_bundles =
        color_field_state.bundles;
    const bool need_shared_query_self =
        shared_pose_mode && shared_pose_queries && !prepared_self_overlaps;
    if (need_shared_query_self) {
        // The query set and reduced color plan are immutable and shared by all
        // owning block workers. Compute the exact existing A/A normalization
        // expressions once per prepared query context, not once per library
        // block. A separate once_flag keeps concurrent first claims safe.
        {
            py::gil_scoped_release nogil;
            std::call_once(shared_pose_queries->query_self_once, [&]() {
                const auto query_self_started = Clock::now();
                std::vector<float> shape_values(shape_query_views.size());
                std::vector<double> color_values(shared_qv.size());
                for (size_t row = 0; row < shape_query_views.size(); ++row) {
                    const Conformer& query = shape_query_views[row];
                    shape_values[row] = self_overlap(query, nullptr);
                    ++stats.shape_query_self_evaluations;
                    if (with_census) {
                        stats.shape_query_self_terms +=
                            uint64_t(query.n_atoms) * uint64_t(query.n_atoms);
                    }
                }
                for (size_t row = 0; row < shared_qv.size(); ++row) {
                    uint64_t terms = 0;
                    color_values[row] = overlap_reduced_color(
                        shared_qv[row], shared_qv[row],
                        *shared_pose_queries->color_plan, 3, 3, nullptr,
                        with_census ? &terms : nullptr);
                    ++stats.color_query_self_evaluations;
                    if (with_census) {
                        stats.color_query_self_terms += terms;
                    }
                }
                shared_pose_queries->shape_query_self = std::move(shape_values);
                shared_pose_queries->color_query_self = std::move(color_values);
                shared_pose_queries->query_self_preparation_seconds =
                    std::chrono::duration<double>(Clock::now() -
                                                  query_self_started)
                        .count();
                shared_pose_queries->query_self_ready = true;
            });
        }
        if (shared_pose_queries->shape_query_self.size() != qv.size() ||
            shared_pose_queries->color_query_self.size() != qv.size())
            throw std::invalid_argument(
                "cached shared-pose query self-overlaps have wrong dimensions");
    }
    // Self-overlap sources, in priority order: prepared values, the shared
    // query cache (query side only), then in-call computation over the
    // referenced library rows and all queries. Library color self stays
    // physical (overlap with `mat`); query color self follows the frame of the
    // query coordinates.
    std::vector<float> saa_fallback, sbb_fallback;
    std::vector<double> caa_fallback, cbb_fallback;
    const bool use_prepared_shape =
        prepared_self_overlaps && prepared_self_overlaps->shape_prepared;
    const bool use_prepared_color =
        prepared_self_overlaps && prepared_self_overlaps->color_prepared;
    const bool use_shared_shape_query =
        need_shared_query_self && !use_prepared_shape;
    const bool use_shared_color_query =
        need_shared_query_self && !use_prepared_color;
    const float* saa = use_prepared_shape
                           ? prepared_self_overlaps->shape_query.data()
                           : (use_shared_shape_query
                                  ? shared_pose_queries->shape_query_self.data()
                                  : nullptr);
    const float* sbb = use_prepared_shape
                           ? prepared_self_overlaps->shape_library.data()
                           : nullptr;
    const double* caa =
        use_prepared_color ? prepared_self_overlaps->color_query.data()
                           : (use_shared_color_query
                                  ? shared_pose_queries->color_query_self.data()
                                  : nullptr);
    const double* cbb = use_prepared_color
                            ? prepared_self_overlaps->color_library.data()
                            : nullptr;
    float* saa_fallback_data = nullptr;
    float* sbb_fallback_data = nullptr;
    double* caa_fallback_data = nullptr;
    double* cbb_fallback_data = nullptr;
    stats.prepared_self_overlaps_used =
        use_prepared_shape || use_prepared_color;
    stats.prepared_shape_query_self_used = use_prepared_shape;
    stats.prepared_shape_library_self_used = use_prepared_shape;
    stats.prepared_color_query_self_used = use_prepared_color;
    stats.prepared_color_library_self_used = use_prepared_color;
    stats.shared_shape_query_self_used = use_shared_shape_query;
    stats.shared_color_query_self_used = use_shared_color_query;
    // Coarse, separate diagnostics: two clocks per channel per call, never
    // inside a pair/trajectory. These elapsed spans are caller work
    // for a parallel dense call, and worker-local work in a block-owned scan.
    double library_shape_self_seconds = 0.0;
    double library_color_self_seconds = 0.0;
    {
        py::gil_scoped_release nogil;
        if (!use_prepared_shape) {
            const auto started =
                with_library_self_timers ? Clock::now() : Clock::time_point{};
            sbb_fallback.resize(lv.size());
            sbb_fallback_data = sbb_fallback.data();
            sbb = sbb_fallback_data;
            for (size_t i : used_library_rows) {
                const int n = lv[i].n_real;
                std::vector<float> scaled(size_t(n) * 3);
                const float* src = lshape + i * library_row_stride;
                for (int j = 0; j < n * 3; ++j)
                    scaled[size_t(j)] = COORD_SCALE * src[j];
                Conformer b{scaled.data(), n};
                sbb_fallback_data[i] = self_overlap(b, nullptr);
                ++stats.shape_library_self_evaluations;
                if (with_census)
                    stats.shape_library_self_terms +=
                        uint64_t(b.n_atoms) * uint64_t(b.n_atoms);
            }
            if (with_library_self_timers)
                library_shape_self_seconds =
                    std::chrono::duration<double>(Clock::now() - started)
                        .count();
        }
        if (!use_prepared_shape && !use_shared_shape_query) {
            saa_fallback.resize(qv.size());
            saa_fallback_data = saa_fallback.data();
            saa = saa_fallback_data;
            for (size_t i = 0; i < qv.size(); ++i) {
                Conformer a{qshape.data() + i * query_capacity * 3,
                            qv[i].n_real};
                saa_fallback_data[i] = self_overlap(a, nullptr);
                ++stats.shape_query_self_evaluations;
                if (with_census)
                    stats.shape_query_self_terms +=
                        uint64_t(a.n_atoms) * uint64_t(a.n_atoms);
            }
        }
        if (!use_prepared_color) {
            const auto started =
                with_library_self_timers ? Clock::now() : Clock::time_point{};
            cbb_fallback.resize(lv.size());
            cbb_fallback_data = cbb_fallback.data();
            cbb = cbb_fallback_data;
            for (size_t i : used_library_rows) {
                cbb_fallback_data[i] = overlap(
                    lv[i], lv[i], mat, ls, ls, nullptr,
                    with_census ? &stats.color_library_self_terms : nullptr);
                ++stats.color_library_self_evaluations;
            }
            if (with_library_self_timers)
                library_color_self_seconds =
                    std::chrono::duration<double>(Clock::now() - started)
                        .count();
        }
        if (!use_prepared_color && !use_shared_color_query) {
            caa_fallback.resize(qv.size());
            caa_fallback_data = caa_fallback.data();
            caa = caa_fallback_data;
            for (size_t i = 0; i < qv.size(); ++i) {
                caa_fallback_data[i] =
                    shared_pose_mode
                        ? overlap_reduced_color(
                              shared_qv[i], shared_qv[i],
                              *shared_pose_queries->color_plan, 3, 3, nullptr,
                              with_census ? &stats.color_query_self_terms
                                          : nullptr)
                        : overlap(qv[i], qv[i], mat, qs, qs, nullptr,
                                  with_census ? &stats.color_query_self_terms
                                              : nullptr);
                ++stats.color_query_self_evaluations;
            }
        }
    }

    // Output: a dense [n_pairs, N_SCORE_COLUMNS] array, or (sink-only) one
    // worker-private row that each finished pair is written into and handed to
    // row_sink. winning_starts is returned in details only with census.
    py::array_t<float> output;
    if (return_dense) {
        output = py::array_t<float>(
            {py::ssize_t(pair_rows), py::ssize_t(N_SCORE_COLUMNS)});
    }
    const bool collect_winning_starts =
        with_census || with_timers || with_worker_timers;
    py::array_t<int32_t> winning_starts;
    if (collect_winning_starts)
        winning_starts = py::array_t<int32_t>{py::ssize_t(pair_rows)};
    float* out = return_dense ? output.mutable_data() : nullptr;
    int32_t* winners =
        collect_winning_starts ? winning_starts.mutable_data() : nullptr;
    // Optional census trace only; no allocation/copy in uninstrumented scans.
    // ROSHAMBO3_FINAL_SELECTION_TRACE records, per row and start, the converged SEARCH
    // objective, the 16 pose floats (M, t', unit quaternion) and the four self
    // values, the inputs of the _replay_fixed_pose_* replays.
    py::object converged_objectives = py::none(),
               converged_pose_bytes = py::none();
    py::object converged_objective_self = py::none();
    float* objective_trace = nullptr;
    float* pose_trace = nullptr;
    double* objective_self_trace = nullptr;
    if (with_census && winner_policy && std::getenv("ROSHAMBO3_FINAL_SELECTION_TRACE")) {
        auto objectives =
            py::array_t<float>({py::ssize_t(pair_rows), py::ssize_t(4)});
        auto bytes = py::array_t<float>(
            {py::ssize_t(pair_rows), py::ssize_t(4), py::ssize_t(16)});
        auto self =
            py::array_t<double>({py::ssize_t(pair_rows), py::ssize_t(4)});
        objective_trace = objectives.mutable_data();
        pose_trace = bytes.mutable_data();
        objective_self_trace = self.mutable_data();
        converged_objectives = std::move(objectives);
        converged_pose_bytes = std::move(bytes);
        converged_objective_self = std::move(self);
        std::fill(objective_trace, objective_trace + pair_rows * 4,
                  std::numeric_limits<float>::quiet_NaN());
        std::fill(pose_trace, pose_trace + pair_rows * 64,
                  std::numeric_limits<float>::quiet_NaN());
        std::fill(objective_self_trace, objective_self_trace + pair_rows * 4,
                  std::numeric_limits<double>::quiet_NaN());
    }

    // One cache-line-aligned score row per worker, the size budgeted by
    // kWorkerTopKFinalRowScratchBytes.
    struct alignas(64) FinalRowScratch {
        std::array<float, N_SCORE_COLUMNS> scores{};
    };

    static_assert(sizeof(FinalRowScratch) == kWorkerTopKFinalRowScratchBytes,
                  "update the sink-only row-scratch budget estimate");
    std::vector<FinalRowScratch> final_row_scratch;
    if (!return_dense) final_row_scratch.resize(static_cast<size_t>(threads));
    auto mutable_row = [&](int worker, size_t row) -> float* {
        if (return_dense) return out + row * size_t(N_SCORE_COLUMNS);
        if (worker < 0 || size_t(worker) >= final_row_scratch.size())
            throw std::out_of_range(
                "sink-only combo worker id is outside row scratch");
        return final_row_scratch[size_t(worker)].scores.data();
    };
    // Publish a finished row. Columns 0-15 are the scores and the reporting
    // pose; 16-19 are the start quaternion, which is the identity because the
    // reported pose already composes the start (and is checked finite).
    auto emit_final_row = [&](int worker, size_t row) {
        const float* scores = mutable_row(worker, row);
        // Work-unit boundary: never publish invalid scores/reporting poses or
        // rely on NaN comparisons in a downstream top-K sink.
        for (int column = 0; column < 16; ++column)
            if (!std::isfinite(scores[column]))
                throw std::runtime_error(
                    "nonfinite final score or reporting pose");
        if (row_sink)
            row_sink->on_row(size_t(worker), row, size_t(qi[row]),
                             size_t(li[row]), mutable_row(worker, row),
                             size_t(N_SCORE_COLUMNS));
    };
    // Legacy-pose winner record. Score columns (roshambo3/scores.py): 0 combo
    // Tanimoto, 1 shape Tanimoto, 2 color Tanimoto, 3 shape overlap (with
    // SHAPE_PREFACTOR), 4 color overlap, 5-8 self overlaps (query shape,
    // library shape, query color, library color), 9-12 pose quaternion, 13-15
    // physical translation, 16-19 start quaternion (identity, column 16 = 1).
    // The legacy pose re-scores the winner from its reported float32 pose.
    auto write_winner = [&](int worker, size_t row, size_t ai, size_t bi,
                            const Conformer& a, const Conformer& b,
                            const float best_q[4], const float best_t[3],
                            int best_start) {
        // A distinct exact winner rescore, even though the per-start final
        // values above are analytic: this pins the reporting rule if search
        // evaluation later becomes approximate or is reordered.
        auto tick = with_timers ? Clock::now() : Clock::time_point{};
        float m[9];
        quat_to_mat(best_q, m);
        for (int k = 0; k < 9; ++k)
            m[k] *= COORD_SCALE;
        float tscaled[3] = {best_t[0] * COORD_SCALE, best_t[1] * COORD_SCALE,
                            best_t[2] * COORD_SCALE};
        const float sab = pair_overlap_at_pose(a, b, m, tscaled, nullptr);
        const auto posed = pose_color(lv[bi], ls, best_q, best_t);
        const double cab =
            overlap(qv[ai], lv[bi], mat, qs, ls, posed.data(),
                    with_census ? &stats.color_winner_rescore_terms : nullptr);
        if (with_timers)
            stats.winner_rescore_s +=
                std::chrono::duration<double>(Clock::now() - tick).count();
        if (with_census) ++stats.color_winner_rescore_evaluations;
        float* o = mutable_row(worker, row);
        for (int k = 0; k < N_SCORE_COLUMNS; ++k)
            o[k] = 0.0f;
        o[1] = float(tanimoto(sab, saa[ai], sbb[bi]));
        o[2] = float(tanimoto(cab, caa[ai], cbb[bi]));
        o[0] = (1.0f - mixing) * o[1] + mixing * o[2];
        o[3] = SHAPE_PREFACTOR * sab;
        o[4] = float(cab);
        o[5] = SHAPE_PREFACTOR * saa[ai];
        o[6] = SHAPE_PREFACTOR * sbb[bi];
        o[7] = float(caa[ai]);
        o[8] = float(cbb[bi]);
        for (int k = 0; k < 4; ++k)
            o[9 + k] = best_q[k];
        for (int k = 0; k < 3; ++k)
            o[13 + k] = best_t[k];
        o[16] = 1.0f;
        if (with_census) winners[row] = best_start;
        emit_final_row(worker, row);
    };
    ExactFinalScratchStats exact_final_stats;
    // Shared-pose final scoring already produced the exact overlaps used for
    // start selection. Serialize those retained values with the selected
    // PreparedReducedPose; do not evaluate the winning pose a second time.
    // Same columns as write_winner. Translation is reported in physical
    // Angstrom, t' / COORD_SCALE; the quaternion is the prepared unit
    // quaternion.
    auto write_winner_shared = [&](int worker, size_t row, size_t ai, size_t bi,
                                   const PreparedReducedPose& pose,
                                   int best_start, float sab, double cab) {
        if (!shared_pose_mode)
            throw std::invalid_argument("shared winner lacks prepared pose");
        float* o = mutable_row(worker, row);
        for (int k = 0; k < N_SCORE_COLUMNS; ++k)
            o[k] = 0.0f;
        o[1] = float(tanimoto(sab, saa[ai], sbb[bi]));
        o[2] = float(tanimoto(cab, caa[ai], cbb[bi]));
        o[0] = (1.0f - mixing) * o[1] + mixing * o[2];
        o[3] = SHAPE_PREFACTOR * sab;
        o[4] = float(cab);
        o[5] = SHAPE_PREFACTOR * saa[ai];
        o[6] = SHAPE_PREFACTOR * sbb[bi];
        o[7] = float(caa[ai]);
        o[8] = float(cbb[bi]);
        for (int k = 0; k < 4; ++k)
            o[9 + k] = pose.unit_quaternion[k];
        for (int k = 0; k < 3; ++k)
            o[13 + k] = pose.translation[k] / COORD_SCALE;
        o[16] = 1.0f;
        if (with_census) winners[row] = best_start;
        emit_final_row(worker, row);
    };
    // -------------------------------------------------------------------------
    // Branch 1: start lanes, grain 16 at the standard four-start mode
    // A work unit is up to two pairs; each pair's four starts become
    // independent lanes [A0 A1 A2 A3 | B0 B1 B2 B3], each with its own pose,
    // gradient and Adagrad cache, and nothing is reduced across lanes. Under
    // the shared contract with a canonical matrix, grain 16 uses the prepared
    // four-start kernels (XMM4 / NEON4; compatible YMM8 pairs on
    // x86; two interleaved NEON4 streams on ARM), bound once per work
    // unit. Other inputs pack an eight-lane scheduled batch for the
    // conformer kernel.
    // -------------------------------------------------------------------------
    if (is_color_start_grain(color_grain) && start_mode == 1) {
#if defined(__AVX2__) || defined(__aarch64__)
        // Census of scheduled-batch bytes, one cache-line-padded counter per
        // worker.
        struct alignas(64) StartBatchByteCounter {
            uint64_t value = 0;
        };

        std::vector<StartBatchByteCounter> start_batch_bytes(
            static_cast<size_t>(threads));
        const size_t n_start_blocks = (pair_rows + 1) / 2;
        run_combo_workers(
            n_start_blocks, threads,
            [&](int worker, size_t block_index) {
                ComboWorkerScratch& worker_scratch =
                    combo_worker_scratch(size_t(worker), executor.get());
                PreparedColorStartBlock block;
                const size_t first_pair_row = block_index * 2;
                block.n_pairs =
                    int(std::min(size_t(2), pair_rows - first_pair_row));
                for (int pair = 0; pair < block.n_pairs; ++pair) {
                    const size_t pair_row = first_pair_row + size_t(pair);
                    block.rows[size_t(pair)] = pair_row;
                    block.source_pairs[size_t(pair)] = {
                        &qv[size_t(qi[pair_row])], &lv[size_t(li[pair_row])],
                        qs, ls};
                }
                ColorConformerWorkspace& block_conformer_workspace =
                    worker_scratch.conformer_workspace;
                const int n_start_lanes = block.n_pairs * 4;
                size_t row[COLOR_CONFORMER_LANES]{},
                    ai[COLOR_CONFORMER_LANES]{}, bi[COLOR_CONFORMER_LANES]{};
                Conformer shape_a[COLOR_CONFORMER_LANES]{},
                    shape_b[COLOR_CONFORMER_LANES]{};
                float start_conjugate[COLOR_CONFORMER_LANES][4]{};
                float q[COLOR_CONFORMER_LANES][4]{},
                    t[COLOR_CONFORMER_LANES][3]{},
                    cache[COLOR_CONFORMER_LANES][7]{};
                // Lane = pair * 4 + start. Every lane starts at the identity;
                // its start enters only through start_conjugate when q_eff is
                // composed.
                for (int lane = 0; lane < n_start_lanes; ++lane) {
                    const int pair = lane / 4;
                    const int start = lane % 4;
                    row[lane] = block.rows[size_t(pair)];
                    ai[lane] = size_t(qi[row[lane]]);
                    bi[lane] = size_t(li[row[lane]]);
                    shape_a[lane] = shape_query_views[ai[lane]];
                    shape_b[lane] = shape_library_views[bi[lane]];
                    float q_start[4], start_matrix[9];
                    start_quaternion(1, start, q_start);
                    quat_to_mat(q_start, start_matrix);
                    if (!matrix_is_signed_permutation(start_matrix,
                                                      SIGNED_PERM_TOL))
                        throw std::invalid_argument(
                            "combo start is not a signed permutation");
                    start_conjugate[lane][0] = q_start[0];
                    for (int k = 1; k < 4; ++k)
                        start_conjugate[lane][k] = -q_start[k];
                    q[lane][0] = 1.0f;
                }
                const bool trajectory_shape =
                    shape_evaluator_set &&
                    shape_config.grain == ShapeGrain::Trajectory;
                PreparedShapeEvaluatorHandle trajectory_shape_evaluator;
                PreparedShapeBatchHandle trajectory_shape_batch;
                if (trajectory_shape) {
                    for (int lane = 1; lane < n_start_lanes; ++lane)
                        if (ai[lane] != ai[0])
                            throw std::invalid_argument(
                                "trajectory shape grain requires one shared query per "
                                "color start block");
                    trajectory_shape_evaluator =
                        shape_evaluator_set->evaluators[ai[0]];
                    const Conformer* targets[COLOR_CONFORMER_LANES]{};
                    for (int lane = 0; lane < n_start_lanes; ++lane)
                        targets[lane] = &shape_b[lane];
                    trajectory_shape_batch =
                        trajectory_shape_evaluator->prepare_batch(
                            targets, size_t(n_start_lanes));
                }
                // Shared contract: one prepared input per pair (not per lane),
                // and the opt-in exact-final scratch bound to that pair for the
                // whole trajectory.
                ReducedColorConformerPairInput prepared_start_pairs[2]{};
                ColorCanonicalFourStartDescriptor
                    canonical_four_start_descriptors[2]{};
                int canonical_four_start_count = 0;
                ColorConformerBatch8& prepared_scheduled_batch =
                    worker_scratch.color_batch;
                const ColorInvariantPlan* shared_plan =
                    shared_pose_mode ? shared_pose_queries->color_plan.get()
                                     : nullptr;
                auto& exact_scratch = worker_scratch.exact_final;
                ColorConformerPairInput expanded_views[COLOR_CONFORMER_LANES]{};
                for (int lane = 0; lane < n_start_lanes; ++lane)
                    expanded_views[lane] = block.source_pairs[lane / 4];
                if (shared_pose_mode) {
                    for (int pair = 0; pair < block.n_pairs; ++pair) {
                        const int lane = pair * 4;
                        prepared_start_pairs[pair] = {
                            &shared_pose_queries
                                 ->reduced_query_colors[ai[lane]],
                            &lv[bi[lane]], 3, ls};
                        if (reuse_exact_final_scratch)
                            exact_scratch[pair].begin_trajectory(
                                *prepared_start_pairs[pair].query,
                                *prepared_start_pairs[pair].library,
                                *shared_plan, 3, ls);
                    }
                }
                // Prepared canonical four-start route: shared pose, the
                // canonical kernel requested, analytic color (fields only when
                // the field hybrid is compiled in). If any
                // pair's descriptor cannot be prepared (noncanonical matrix or
                // unsafe input) the whole block takes the scheduled batch path.
                bool use_prepared_four_start_kernel =
                    shared_pose_mode && canonical_color_kernel &&
                    (!use_color_fields || R3_COLOR_START_HYBRID) &&
                    shared_plan != nullptr;
                if (use_prepared_four_start_kernel) {
                    for (int pair = 0; pair < block.n_pairs; ++pair) {
                        if (!prepare_color_canonical_four_start_descriptor(
                                prepared_start_pairs[pair], *shared_plan,
                                canonical_four_start_descriptors[pair])) {
                            use_prepared_four_start_kernel = false;
                            break;
                        }
                        if (use_color_fields) {
                            auto& d = canonical_four_start_descriptors[pair];
                            d.field_bundle = field_bundles[ai[pair * 4]].get();
                            d.field_stats = with_census
                                                ? &stats.color_start_field_stats
                                                : nullptr;
                            for (int type = 1; type < d.n_features; ++type) {
                                d.field_backend[size_t(type)] =
                                    d.field_bundle->backend_code(type);
                                d.field_active[size_t(type)] =
                                    d.field_bundle->field_active(type);
                            }
                        }
                    }
                    if (use_prepared_four_start_kernel) {
                        canonical_four_start_count = block.n_pairs;
                        prepare_color_canonical_four_start_workspace(
                            canonical_four_start_descriptors,
                            canonical_four_start_count,
                            block_conformer_workspace
                                .canonical_four_start_workspace);
                    }
                }
                // Scheduled route: expand the two source pairs to eight lanes
                // and pack them once per work unit, not once per step.
                if (!use_prepared_four_start_kernel) {
                    const auto pack_started =
                        with_timers ? Clock::now() : Clock::time_point{};
                    if (shared_pose_mode) {
                        ReducedColorConformerPairInput
                            expanded[COLOR_CONFORMER_LANES]{};
                        for (int lane = 0; lane < n_start_lanes; ++lane)
                            expanded[lane] = prepared_start_pairs[lane / 4];
                        pack_color_conformer_batch8_prepared_into(
                            expanded, n_start_lanes, nf,
                            prepared_scheduled_batch);
                    } else {
                        pack_color_conformer_batch8_into(
                            expanded_views, n_start_lanes, nf,
                            prepared_scheduled_batch);
                    }
                    if (with_timers)
                        stats.color_start_pack_s +=
                            std::chrono::duration<double>(Clock::now() -
                                                          pack_started)
                                .count();
                    start_batch_bytes[size_t(worker)].value +=
                        uint64_t(prepared_scheduled_batch.query_xyz.size() *
                                     sizeof(float) +
                                 prepared_scheduled_batch.library_xyz.size() *
                                     sizeof(float) +
                                 prepared_scheduled_batch.query_mask.size() *
                                     sizeof(int32_t) +
                                 prepared_scheduled_batch.library_mask.size() *
                                     sizeof(int32_t));
                }

                // Winner-only final policy: only when the block is on the
                // prepared canonical route, color is analytic (no fields),
                // shape is a field backend other than the trajectory grain, and
                // every descriptor's immutable inputs were proven safe.
                // Otherwise all four starts are scored exactly below.
                const bool winner_only_final =
                    winner_policy && use_prepared_four_start_kernel &&
                    !use_color_fields && !trajectory_shape &&
                    shape_config.backend >= ShapeBackend::ScalarField &&
                    std::all_of(
                        canonical_four_start_descriptors,
                        canonical_four_start_descriptors +
                            canonical_four_start_count,
                        [](const auto& d) { return d.immutable_inputs_safe; });
                // Bound execution: fix the color SEARCH policy and each
                // pair's shape execution once for this work unit rather than
                // re-dispatching every step. The value-only shape binding
                // serves only the converged-objective pass.
                std::optional<BoundCanonicalColorExecution> bound_color;
                std::optional<BoundShapeExecution> bound_shape[2];
                std::optional<BoundShapeExecution> bound_shape_values[2];
                if constexpr (R3_FINAL_SHAPE_VALUE_ONLY) {
                    if (winner_only_final)
                        for (int pair = 0; pair < block.n_pairs; ++pair) {
                            const int lane = pair * 4;
                            const auto* workspace =
                                shape_library_workspaces.empty()
                                    ? nullptr
                                    : &shape_library_workspaces[bi[lane]];
                            bound_shape_values[pair].emplace(
                                shape_evaluator_set->evaluators[ai[lane]]
                                    ->bind_execution(shape_b[lane], workspace,
                                                     true));
                        }
                }
                if constexpr (R3_BOUND_EXECUTION) {
                    if (use_prepared_four_start_kernel)
                        bound_color.emplace(
                            canonical_four_start_descriptors,
                            canonical_four_start_count, *shared_plan,
                            block_conformer_workspace
                                .canonical_four_start_workspace,
                            optimizer_exponential, factor_canonical_torque);
                    if (shared_pose_mode && shape_evaluator_set &&
                        !trajectory_shape)
                        for (int pair = 0; pair < block.n_pairs; ++pair) {
                            const int lane = pair * 4;
                            const auto* workspace =
                                shape_library_workspaces.empty()
                                    ? nullptr
                                    : &shape_library_workspaces[bi[lane]];
                            bound_shape[pair].emplace(
                                shape_evaluator_set->evaluators[ai[lane]]
                                    ->bind_execution(shape_b[lane], workspace));
                        }
                }
                // The canonical four-start color call at the compiled binding
                // level (R3_BOUND_EXECUTION 2, 1 or 0; see the top of the
                // file).
                auto evaluate_canonical =
                    [&](const PreparedReducedPose* const* poses,
                        ColorEvaluation* outputs, uint64_t* terms,
                        ColorConformerCensus* census,
                        ColorConformerPhaseTimes* phases,
                        ColorSearchOutput search_output) {
                        if constexpr (R3_BOUND_EXECUTION == 2)
                            bound_color->evaluate_bound_policy(
                                poses, outputs, terms, census, phases,
                                search_output);
                        else if constexpr (R3_BOUND_EXECUTION)
                            bound_color->evaluate(poses, outputs, terms, census,
                                                  phases, search_output);
                        else
                            evaluate_color_canonical_four_start_batch(
                                canonical_four_start_descriptors,
                                canonical_four_start_count, *shared_plan, poses,
                                true, outputs, terms, census, phases,
                                block_conformer_workspace
                                    .canonical_four_start_workspace,
                                optimizer_exponential, factor_canonical_torque,
                                search_output);
                    };
                // Optimizer steps for all start lanes (steps 1-4 of the
                // function comment).
                for (int step = 0; step < steps; ++step) {
                    float q_eff[COLOR_CONFORMER_LANES][4]{};
                    float tphys[COLOR_CONFORMER_LANES][3]{};
                    float gs[COLOR_CONFORMER_LANES][7]{};
                    float sab[COLOR_CONFORMER_LANES]{};
                    PreparedReducedPose poses[COLOR_CONFORMER_LANES]{};
                    const PreparedReducedPose*
                        pose_ptrs[COLOR_CONFORMER_LANES]{};
                    // Trajectory shape: one batched shape call for all lanes,
                    // which share one query (checked when the block was set
                    // up).
                    if (trajectory_shape) {
                        float matrices[COLOR_CONFORMER_LANES][9]{};
                        ShapePoseRequest requests[COLOR_CONFORMER_LANES]{};
                        ShapeEvaluation evaluations[COLOR_CONFORMER_LANES]{};
                        for (int lane = 0; lane < n_start_lanes; ++lane) {
                            quat_mul(q[lane], start_conjugate[lane],
                                     q_eff[lane]);
                            const float* matrix = nullptr;
                            const float* translation = nullptr;
                            if (shared_pose_mode) {
                                poses[lane] = prepare_reduced_pose_fp32(
                                    q_eff[lane], t[lane]);
                                pose_ptrs[lane] = &poses[lane];
                                matrix = poses[lane].matrix;
                                translation = poses[lane].translation;
                                if (with_census || with_timers)
                                    ++stats.shared_pose_preparations;
                            } else {
                                quat_to_mat(q_eff[lane], matrices[lane]);
                                for (int k = 0; k < 9; ++k)
                                    matrices[lane][k] *= COORD_SCALE;
                                matrix = matrices[lane];
                                translation = t[lane];
                            }
                            for (int k = 0; k < 3; ++k)
                                tphys[lane][k] = t[lane][k] / COORD_SCALE;
                            requests[lane] = {
                                &shape_b[lane],
                                nullptr,
                                matrix,
                                translation,
                                1.0f,
                                shared_pose_mode ? &poses[lane] : nullptr};
                        }
                        const auto tick =
                            with_timers ? Clock::now() : Clock::time_point{};
                        trajectory_shape_evaluator->evaluate_batch(
                            requests, size_t(n_start_lanes),
                            trajectory_shape_batch.get(), evaluations);
                        if (with_timers)
                            stats.shape_gradient_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                        for (int lane = 0; lane < n_start_lanes; ++lane) {
                            sab[lane] = evaluations[lane].s_ab;
                            std::copy(evaluations[lane].descent,
                                      evaluations[lane].descent + 7, gs[lane]);
                            if (with_census) {
                                ++stats.shape_gradient_evaluations;
                                count_prepared_shape(evaluations[lane], stats);
                            }
                        }
                    } else {
                        for (int lane = 0; lane < n_start_lanes; ++lane) {
                            quat_mul(q[lane], start_conjugate[lane],
                                     q_eff[lane]);
                            float legacy_matrix[9]{};
                            const float* matrix = nullptr;
                            const float* translation = nullptr;
                            if (shared_pose_mode) {
                                poses[lane] = prepare_reduced_pose_fp32(
                                    q_eff[lane], t[lane]);
                                pose_ptrs[lane] = &poses[lane];
                                matrix = poses[lane].matrix;
                                translation = poses[lane].translation;
                                if (with_census || with_timers)
                                    ++stats.shared_pose_preparations;
                            } else {
                                quat_to_mat(q_eff[lane], legacy_matrix);
                                for (int k = 0; k < 9; ++k)
                                    legacy_matrix[k] *= COORD_SCALE;
                                matrix = legacy_matrix;
                                translation = t[lane];
                            }
                            auto tick = with_timers ? Clock::now()
                                                    : Clock::time_point{};
                            if (shape_evaluator_set) {
                                ShapeEvaluation evaluation;
                                const ShapeLibraryWorkspace* workspace =
                                    shape_library_workspaces.empty()
                                        ? nullptr
                                        : &shape_library_workspaces[bi[lane]];
                                const ShapePoseRequest request{
                                    &shape_b[lane],
                                    workspace,
                                    matrix,
                                    translation,
                                    1.0f,
                                    shared_pose_mode ? &poses[lane] : nullptr};
                                if (R3_BOUND_EXECUTION && shared_pose_mode)
                                    bound_shape[lane / 4]->evaluate(poses[lane],
                                                                    evaluation);
                                else
                                    evaluate_prepared_shape(
                                        *shape_evaluator_set
                                             ->evaluators[ai[lane]],
                                        request, evaluation);
                                sab[lane] = evaluation.s_ab;
                                std::copy(evaluation.descent,
                                          evaluation.descent + 7, gs[lane]);
                                if (with_census) {
                                    ++stats.shape_gradient_evaluations;
                                    count_prepared_shape(evaluation, stats);
                                }
                            } else {
                                get_gradient(shape_a[lane], shape_b[lane],
                                             matrix, translation, 1.0f,
                                             gs[lane], &sab[lane], nullptr);
                                if (with_census) {
                                    ++stats.shape_gradient_evaluations;
                                    stats.shape_gradient_terms +=
                                        uint64_t(shape_a[lane].n_atoms) *
                                        uint64_t(shape_b[lane].n_atoms);
                                    stats.shape_issued_terms +=
                                        uint64_t(shape_a[lane].n_atoms) *
                                        uint64_t(shape_b[lane].n_atoms);
                                }
                            }
                            if (with_timers)
                                stats.shape_gradient_s +=
                                    std::chrono::duration<double>(Clock::now() -
                                                                  tick)
                                        .count();
                            for (int k = 0; k < 3; ++k)
                                tphys[lane][k] = t[lane][k] / COORD_SCALE;
                        }
                    }

                    // Color for all lanes in one call: the prepared canonical
                    // kernel, else the scheduled conformer batch; the legacy
                    // pose uses the unprepared kernel.
                    ColorEvaluation colors[COLOR_CONFORMER_LANES];
                    uint64_t batch_terms[COLOR_CONFORMER_LANES]{};
                    const ColorFieldBundle*
                        field_contexts[COLOR_CONFORMER_LANES]{};
                    if (use_color_fields)
                        for (int lane = 0; lane < n_start_lanes; ++lane)
                            field_contexts[lane] =
                                field_bundles[ai[lane]].get();
                    auto tick =
                        with_timers ? Clock::now() : Clock::time_point{};
                    if (shared_pose_mode) {
                        if (use_prepared_four_start_kernel) {
                            evaluate_canonical(
                                pose_ptrs, colors, batch_terms,
                                with_census ? &stats.color_start_census
                                            : nullptr,
                                with_timers ? &start_schedule_phase_times
                                            : nullptr,
                                ColorSearchOutput::ValueAndGradient);
                        } else {
                            evaluate_color_conformer_batch8_prepared(
                                prepared_scheduled_batch, *shared_plan,
                                pose_ptrs, true, colors, batch_terms,
                                with_census ? &stats.color_start_census
                                            : nullptr,
                                with_timers ? &start_schedule_phase_times
                                            : nullptr,
                                &block_conformer_workspace,
                                use_color_invariant_plan,
                                use_color_fields ? field_contexts : nullptr,
                                with_census && use_color_fields
                                    ? &stats.color_start_field_stats
                                    : nullptr,
                                factor_canonical_torque, optimizer_exponential,
                                canonical_color_kernel);
                        }
                    } else {
                        evaluate_color_conformer_batch8(
                            prepared_scheduled_batch, mat.view(), q_eff, tphys,
                            true, colors, batch_terms,
                            with_census ? &stats.color_start_census : nullptr,
                            with_timers ? &start_schedule_phase_times : nullptr,
                            &block_conformer_workspace,
                            use_color_fields ? field_contexts : nullptr,
                            with_census && use_color_fields
                                ? &stats.color_start_field_stats
                                : nullptr);
                    }
                    if (with_timers)
                        stats.color_gradient_s +=
                            std::chrono::duration<double>(Clock::now() - tick)
                                .count();
                    for (int lane = 0; lane < n_start_lanes; ++lane) {
                        if (with_census) {
                            ++stats.color_gradient_evaluations;
                            stats.color_gradient_terms += batch_terms[lane];
                        }
                        tick = with_timers ? Clock::now() : Clock::time_point{};
                        // Mix and step each lane independently.
                        float mixed_g[7]{};
                        mixed_descent(gs[lane], sab[lane], saa[ai[lane]],
                                      sbb[bi[lane]], colors[lane],
                                      caa[ai[lane]], cbb[bi[lane]], mixing,
                                      mixed_g);
                        if (with_timers)
                            stats.combo_weighting_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                        tick = with_timers ? Clock::now() : Clock::time_point{};
                        adagrad_exp_step(q[lane], t[lane], mixed_g, cache[lane],
                                         lr_q, lr_t_scaled);
                        if (with_timers)
                            stats.pose_update_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                    }
                }

                if (winner_only_final) {
                    // Fresh evaluations after the LAST update. None of the overlaps
                    // or poses from the optimizer loop above are reused here.
                    const auto objective_tick =
                        with_timers ? Clock::now() : Clock::time_point{};
                    PreparedReducedPose final_poses[COLOR_CONFORMER_LANES]{};
                    const PreparedReducedPose*
                        final_pose_ptrs[COLOR_CONFORMER_LANES]{};
                    float field_shape[COLOR_CONFORMER_LANES]{};
                    ColorEvaluation final_search_color[COLOR_CONFORMER_LANES]{};
                    uint64_t objective_terms[COLOR_CONFORMER_LANES]{};
                    for (int lane = 0; lane < n_start_lanes; ++lane) {
                        float q_eff[4];
                        quat_mul(q[lane], start_conjugate[lane], q_eff);
                        final_poses[lane] =
                            prepare_reduced_pose_fp32(q_eff, t[lane]);
                        final_pose_ptrs[lane] = &final_poses[lane];
                        const auto* workspace =
                            shape_library_workspaces.empty()
                                ? nullptr
                                : &shape_library_workspaces[bi[lane]];
                        ShapeEvaluation evaluation;
                        const ShapePoseRequest request{
                            &shape_b[lane],
                            workspace,
                            final_poses[lane].matrix,
                            final_poses[lane].translation,
                            1.0f,
                            &final_poses[lane]};
                        if constexpr (R3_FINAL_SHAPE_VALUE_ONLY)
                            bound_shape_values[lane / 4]->evaluate(
                                final_poses[lane], evaluation);
                        else if constexpr (R3_BOUND_EXECUTION)
                            bound_shape[lane / 4]->evaluate(final_poses[lane],
                                                            evaluation);
                        else
                            evaluate_prepared_shape(
                                *shape_evaluator_set->evaluators[ai[lane]],
                                request, evaluation);
                        field_shape[lane] = evaluation.s_ab;
                        if (with_census) {
                            ++stats.shared_pose_preparations;
                            ++stats.converged_objective_evaluations;
                        }
                    }
                // Analytic SEARCH color: unchanged optimizer exp and source-order
                // four-start helper, not the accurate analytic reporting kernel.
                // Keep gradient=true's SEARCH arithmetic policy while explicitly
                // selecting a specialization that omits derivative instructions.
                // R3_SEARCH_VALUE_ONLY defaults to 0: production evaluates
                // value AND descent here and discards the descent. The exact
                // ValueOnly specialization exists but is not adopted
                // ("color final values 0").
                // Passing gradient=false instead is not equivalent: it once
                // silently selected a different exponential and reduction.
#ifndef R3_SEARCH_VALUE_ONLY
#define R3_SEARCH_VALUE_ONLY 0
#endif
                    evaluate_canonical(
                        final_pose_ptrs, final_search_color, objective_terms,
                        nullptr, nullptr,
                        R3_SEARCH_VALUE_ONLY
                            ? ColorSearchOutput::ValueOnly
                            : ColorSearchOutput::ValueAndGradient);
                    if (with_timers)
                        stats.converged_objective_s +=
                            std::chrono::duration<double>(Clock::now() -
                                                          objective_tick)
                                .count();
                    // Normalize each start's SEARCH objective with the
                    // unrounded self values and keep the best (strict >,
                    // earliest start on ties).
                    for (int pair = 0; pair < block.n_pairs; ++pair) {
                        int winner = pair * 4;
                        float best = -std::numeric_limits<float>::infinity();
                        for (int start = 0; start < 4; ++start) {
                            const int lane = pair * 4 + start;
                            const float ts =
                                float(tanimoto(field_shape[lane], saa[ai[lane]],
                                               sbb[bi[lane]]));
                            const float tc =
                                float(tanimoto(final_search_color[lane].value,
                                               caa[ai[lane]], cbb[bi[lane]]));
                            const float objective =
                                (1.0f - mixing) * ts + mixing * tc;
                            if (objective_trace) {
                                objective_trace[row[lane] * 4 + size_t(start)] =
                                    objective;
                                float* dst =
                                    pose_trace +
                                    (row[lane] * 4 + size_t(start)) * 16;
                                std::copy(final_poses[lane].matrix,
                                          final_poses[lane].matrix + 9, dst);
                                std::copy(final_poses[lane].translation,
                                          final_poses[lane].translation + 3,
                                          dst + 9);
                                std::copy(final_poses[lane].unit_quaternion,
                                          final_poses[lane].unit_quaternion + 4,
                                          dst + 12);
                                double* self =
                                    objective_self_trace + row[lane] * 4;
                                self[0] = saa[ai[lane]];
                                self[1] = sbb[bi[lane]];
                                self[2] = caa[ai[lane]];
                                self[3] = cbb[bi[lane]];
                            }
                            if (converged_start_is_better(objective, best)) {
                                best = objective;
                                winner = lane;
                            }
                            if (with_census)
                                stats.converged_objective_color_terms +=
                                    objective_terms[lane];
                        }
                        // The reported scores: accurate analytic shape and
                        // score_exact_final_color at the winner's pose, once
                        // per pair. SEARCH values are never reported.
                        const auto final_tick =
                            with_timers ? Clock::now() : Clock::time_point{};
                        const auto& pose = final_poses[winner];
                        const float sab = pair_overlap_at_pose(
                            shape_a[winner], shape_b[winner], pose.matrix,
                            pose.translation, nullptr);
                        const double cab =
                            score_exact_final_color(
                                shared_pose_queries
                                    ->reduced_query_colors[ai[winner]],
                                lv[bi[winner]], *shared_plan, pose, 3, ls,
                                reuse_exact_final_scratch ? &exact_scratch[pair]
                                                          : nullptr,
                                with_census ? &stats.color_winner_rescore_terms
                                            : nullptr,
                                with_census || with_timers ? &exact_final_stats
                                                           : nullptr,
                                with_timers)
                                .value;
                        if (with_census) {
                            ++stats.winner_only_final_pairs;
                            ++stats.analytic_final_shape_evaluations;
                            ++stats.color_winner_rescore_evaluations;
                        }
                        if (with_timers)
                            stats.winner_rescore_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              final_tick)
                                    .count();
                        // winner % 4 is the winning lane's start index.
                        write_winner_shared(worker, row[winner], ai[winner],
                                            bi[winner], pose, winner % 4, sab,
                                            cab);
                    }
                    return;
                }
                // Exact scalar final scores drive strict-gt start selection. Lane
                // order is pair A starts 0..3 then pair B starts 0..3, so ties retain
                // the same earliest standard start as the sequential path.
                float best_mixed[2];
                float best_q[2][4]{};
                float best_t[2][3]{};
                PreparedReducedPose best_pose[2]{};
                float best_shape_cross[2]{};
                double best_color_cross[2]{};
                int best_start[2]{};
                for (int pair = 0; pair < block.n_pairs; ++pair) {
                    best_mixed[pair] = -std::numeric_limits<float>::infinity();
                    best_q[pair][0] = 1.0f;
                }
#if R3_FINAL_RESCORE_MODE != 0
                // Experimental final arithmetic, compiled out by default:
                // rescore all starts in batches of the block's pairs at one
                // start index through final_rescore_batch, with
                // R3_FINAL_RESCORE_MODE 1 (scalar polynomial), 2 (SIMD within a
                // pair) or 3 (SIMD across pairs). Mode 0 keeps the accurate
                // per-lane scoring below.
                PreparedReducedPose final_poses[8]{};
                FinalRescoreCross final_cross[8]{};
                if (shared_pose_mode) {
                    const auto tick =
                        with_timers ? Clock::now() : Clock::time_point{};
                    for (int lane = 0; lane < n_start_lanes; ++lane) {
                        float q_eff[4];
                        quat_mul(q[lane], start_conjugate[lane], q_eff);
                        final_poses[lane] =
                            prepare_reduced_pose_fp32(q_eff, t[lane]);
                        if (with_census || with_timers)
                            ++stats.shared_pose_preparations;
                    }
                    // Each lane owns a different independent input pair at the SAME
                    // start index. Four starts of one pair are never SIMD pair lanes.
                    for (int start = 0; start < 4; ++start) {
                        FinalRescorePair inputs[2]{};
                        FinalRescoreCross outputs[2]{};
                        for (int pair = 0; pair < block.n_pairs; ++pair) {
                            const int lane = pair * 4 + start;
                            inputs[pair] = {
                                shape_a[lane],
                                shape_b[lane],
                                &shared_pose_queries
                                     ->reduced_query_colors[ai[lane]],
                                &lv[bi[lane]],
                                shared_plan,
                                &final_poses[lane],
                                3,
                                ls,
                                reuse_exact_final_scratch ? &exact_scratch[pair]
                                                          : nullptr};
                        }
                        FinalRescoreCounts counts;
                        final_rescore_batch(
                            inputs, size_t(block.n_pairs), outputs,
                            static_cast<FinalRescoreMode>(R3_FINAL_RESCORE_MODE),
                            with_census ? &counts : nullptr);
                        if (with_census) {
                            stats.color_per_start_final_terms +=
                                counts.color_terms;
                            auto& total = stats.final_rescore_counts;
                            total.shape_terms += counts.shape_terms;
                            total.color_terms += counts.color_terms;
                            total.shape_issued_lanes +=
                                counts.shape_issued_lanes;
                            total.color_issued_lanes +=
                                counts.color_issued_lanes;
                            total.accurate_fallback_pairs +=
                                counts.accurate_fallback_pairs;
                            total.packed_coordinate_bytes +=
                                counts.packed_coordinate_bytes;
                            total.horizontal_reductions +=
                                counts.horizontal_reductions;
                            total.temporary_allocations +=
                                counts.temporary_allocations;
                            total.allocated_payload_bytes +=
                                counts.allocated_payload_bytes;
                            ++stats.final_rescore_batches;
                            stats.final_rescore_pairs +=
                                uint64_t(block.n_pairs);
                        }
                        for (int pair = 0; pair < block.n_pairs; ++pair)
                            final_cross[pair * 4 + start] = outputs[pair];
                    }
                    if (with_timers)
                        stats.per_start_final_s +=
                            std::chrono::duration<double>(Clock::now() - tick)
                                .count();
                }
#endif
                // All-start scoring: each start's converged pose gets an exact
                // shape and color score (shared: score_exact_final_color on the
                // prepared pose; legacy: FP64 overlap at the physical pose),
                // then Tanimoto, mixing and selection.
                for (int lane = 0; lane < n_start_lanes; ++lane) {
                    const int pair = lane / 4;
                    const int start = lane % 4;
                    auto tick =
                        with_timers ? Clock::now() : Clock::time_point{};
                    float q_eff[4], m[9];
                    quat_mul(q[lane], start_conjugate[lane], q_eff);
                    PreparedReducedPose final_pose;
                    if (shared_pose_mode) {
#if R3_FINAL_RESCORE_MODE != 0
                        final_pose = final_poses[lane];
#else
                        final_pose = prepare_reduced_pose_fp32(q_eff, t[lane]);
                        if (with_census || with_timers)
                            ++stats.shared_pose_preparations;
#endif
                        std::copy(final_pose.matrix, final_pose.matrix + 9, m);
                    } else {
                        quat_to_mat(q_eff, m);
                        for (int k = 0; k < 9; ++k)
                            m[k] *= COORD_SCALE;
                    }
                    float tphys[3] = {t[lane][0] / COORD_SCALE,
                                      t[lane][1] / COORD_SCALE,
                                      t[lane][2] / COORD_SCALE};
                    // Legacy shape scoring uses the translation after its
                    // physical round trip, the value reproducible from the
                    // reported float32 pose.
                    float treport[3] = {tphys[0] * COORD_SCALE,
                                        tphys[1] * COORD_SCALE,
                                        tphys[2] * COORD_SCALE};
                    const float* final_matrix =
                        shared_pose_mode ? final_pose.matrix : m;
                    const float* final_translation =
                        shared_pose_mode ? final_pose.translation : treport;
                    const float shape_cross =
#if R3_FINAL_RESCORE_MODE != 0
                        shared_pose_mode
                            ? final_cross[lane].shape
                            :
#endif
                            pair_overlap_at_pose(shape_a[lane], shape_b[lane],
                                                 final_matrix,
                                                 final_translation, nullptr);
                    if (with_census) ++stats.analytic_final_shape_evaluations;
                    double color_cross = 0.0;
                    if (shared_pose_mode) {
#if R3_FINAL_RESCORE_MODE != 0
                        color_cross = final_cross[lane].color;
#else
                        color_cross =
                            score_exact_final_color(
                                shared_pose_queries
                                    ->reduced_query_colors[ai[lane]],
                                lv[bi[lane]], *shared_plan, final_pose, 3, ls,
                                reuse_exact_final_scratch ? &exact_scratch[pair]
                                                          : nullptr,
                                with_census ? &stats.color_per_start_final_terms
                                            : nullptr,
                                with_census || with_timers ? &exact_final_stats
                                                           : nullptr,
                                with_timers)
                                .value;
#endif
                    } else {
                        const auto posed =
                            pose_color(lv[bi[lane]], ls, q_eff, tphys);
                        color_cross = overlap(
                            qv[ai[lane]], lv[bi[lane]], mat, qs, ls,
                            posed.data(),
                            with_census ? &stats.color_per_start_final_terms
                                        : nullptr);
                    }
                    if (with_timers)
                        stats.per_start_final_s +=
                            std::chrono::duration<double>(Clock::now() - tick)
                                .count();
                    if (with_census) ++stats.color_per_start_final_evaluations;
                    const float ts = float(
                        tanimoto(shape_cross, saa[ai[lane]], sbb[bi[lane]]));
                    const float tc = float(
                        tanimoto(color_cross, caa[ai[lane]], cbb[bi[lane]]));
                    const float mixed = (1.0f - mixing) * ts + mixing * tc;
                    if (converged_start_is_better(mixed, best_mixed[pair])) {
                        best_mixed[pair] = mixed;
                        best_start[pair] = start;
                        if (shared_pose_mode) {
                            best_pose[pair] = final_pose;
                            best_shape_cross[pair] = shape_cross;
                            best_color_cross[pair] = color_cross;
                        } else {
                            for (int k = 0; k < 4; ++k)
                                best_q[pair][k] = q_eff[k];
                            for (int k = 0; k < 3; ++k)
                                best_t[pair][k] = tphys[k];
                        }
                    }
                }
                for (int pair = 0; pair < block.n_pairs; ++pair) {
                    const int lane = pair * 4;
                    if (shared_pose_mode)
                        write_winner_shared(
                            worker, row[lane], ai[lane], bi[lane],
                            best_pose[pair], best_start[pair],
                            best_shape_cross[pair], best_color_cross[pair]);
                    else
                        write_winner(worker, row[lane], ai[lane], bi[lane],
                                     shape_a[lane], shape_b[lane], best_q[pair],
                                     best_t[pair], best_start[pair]);
                }
            },
            with_worker_timers ? &worker_elapsed_seconds : nullptr,
            with_worker_timers ? &worker_placements : nullptr,
            require_worker_affinity, executor.get(), 2,
            executor ? executor->claim_chunk_rows() : 256,
            with_worker_timers ? &scoring_claims_by_worker : nullptr);
        for (const StartBatchByteCounter& counter : start_batch_bytes)
            stats.color_start_scheduled_batch_numeric_bytes += counter.value;
#else
        throw std::invalid_argument("AVX2 start-lane grain unavailable");
#endif
    } else if (is_color_conformer_grain(color_grain)) {
#if defined(__AVX2__)
        // ---------------------------------------------------------------------
        // Branch 2: conformer grain 11.
        // A work unit is one block of pairs, one pair per state lane;
        // starts run sequentially and each restarts q, t and the Adagrad cache
        // of every lane.
        // ---------------------------------------------------------------------
        // Each state lane owns an independent pair, pose and Adagrad cache.
        const size_t rows_per_block = size_t(COLOR_CONFORMER_LANES);
        const size_t n_blocks =
            (pair_rows + rows_per_block - 1) / rows_per_block;
        run_combo_workers(
            n_blocks, threads,
            [&](int worker, size_t block_index) {
                ComboWorkerScratch& worker_scratch =
                    combo_worker_scratch(size_t(worker), executor.get());
                ColorConformerWorkspace& block_conformer_workspace =
                    worker_scratch.conformer_workspace;
                const ColorConformerBatch8* color_batch_ptr = nullptr;
                ColorConformerPairInput source_pairs[COLOR_CONFORMER_LANES]{};
                const int n_conformers = int(std::min(
                    rows_per_block, pair_rows - block_index * rows_per_block));
                const size_t first_row = block_index * rows_per_block;
                size_t row[COLOR_CONFORMER_LANES]{},
                    ai[COLOR_CONFORMER_LANES]{}, bi[COLOR_CONFORMER_LANES]{};
                Conformer shape_a[COLOR_CONFORMER_LANES]{},
                    shape_b[COLOR_CONFORMER_LANES]{};
                ReducedColorConformerPairInput
                    prepared_pairs[COLOR_CONFORMER_LANES]{};
                for (int lane = 0; lane < n_conformers; ++lane) {
                    row[lane] = first_row + size_t(lane);
                    ai[lane] = size_t(qi[row[lane]]);
                    bi[lane] = size_t(li[row[lane]]);
                    shape_a[lane] = shape_query_views[ai[lane]];
                    shape_b[lane] = shape_library_views[bi[lane]];
                    source_pairs[lane] = {&qv[ai[lane]], &lv[bi[lane]], qs, ls};
                    if (shared_pose_mode)
                        prepared_pairs[lane] = {
                            &shared_pose_queries
                                 ->reduced_query_colors[ai[lane]],
                            &lv[bi[lane]], 3, ls};
                }
                const ColorInvariantPlan* shared_plan =
                    shared_pose_mode ? shared_pose_queries->color_plan.get()
                                     : nullptr;
                ColorConformerBatch8& prepared_color_batch =
                    worker_scratch.color_batch;
                const auto shared_pack_started =
                    with_timers ? Clock::now() : Clock::time_point{};
                if (shared_pose_mode)
                    pack_color_conformer_batch8_prepared_into(
                        prepared_pairs, n_conformers, nf,
                        prepared_color_batch);
                else
                    pack_color_conformer_batch8_into(source_pairs,
                                                     n_conformers, nf,
                                                     prepared_color_batch);
                color_batch_ptr = &prepared_color_batch;
                const ColorConformerBatch8& color_batch = *color_batch_ptr;
                if (with_timers) {
                    const double elapsed =
                        std::chrono::duration<double>(Clock::now() -
                                                      shared_pack_started)
                            .count();
                    stats.color_conformer_pack_s += elapsed;
                }

                float best_mixed[COLOR_CONFORMER_LANES];
                float best_q[COLOR_CONFORMER_LANES][4]{};
                float best_t[COLOR_CONFORMER_LANES][3]{};
                PreparedReducedPose best_pose[COLOR_CONFORMER_LANES]{};
                float best_shape_cross[COLOR_CONFORMER_LANES]{};
                double best_color_cross[COLOR_CONFORMER_LANES]{};
                int best_start[COLOR_CONFORMER_LANES]{};
                auto& exact_scratch = worker_scratch.exact_final;
                for (int lane = 0; lane < n_conformers; ++lane) {
                    best_mixed[lane] = -std::numeric_limits<float>::infinity();
                    best_q[lane][0] = 1.0f;
                    if (shared_pose_mode && reuse_exact_final_scratch)
                        exact_scratch[lane].begin_trajectory(
                            *prepared_pairs[lane].query,
                            *prepared_pairs[lane].library, *shared_plan, 3, ls);
                }

                const bool trajectory_shape =
                    shape_evaluator_set &&
                    shape_config.grain == ShapeGrain::Trajectory;
                PreparedShapeEvaluatorHandle trajectory_shape_evaluator;
                PreparedShapeBatchHandle trajectory_shape_batch;
                if (trajectory_shape) {
                    for (int lane = 1; lane < n_conformers; ++lane)
                        if (ai[lane] != ai[0])
                            throw std::invalid_argument(
                                "trajectory shape grain requires one shared query per "
                                "color conformer block");
                    trajectory_shape_evaluator =
                        shape_evaluator_set->evaluators[ai[0]];
                    const Conformer* targets[COLOR_CONFORMER_LANES]{};
                    for (int lane = 0; lane < n_conformers; ++lane)
                        targets[lane] = &shape_b[lane];
                    trajectory_shape_batch =
                        trajectory_shape_evaluator->prepare_batch(
                            targets, size_t(n_conformers));
                }

                for (int start = 0; start < starts; ++start) {
                    float q_start[4];
                    start_quaternion(start_mode, start, q_start);
                    float start_matrix[9];
                    quat_to_mat(q_start, start_matrix);
                    if (!matrix_is_signed_permutation(start_matrix,
                                                      SIGNED_PERM_TOL))
                        throw std::invalid_argument(
                            "combo start is not a signed permutation");
                    float q_conj[4] = {q_start[0], -q_start[1], -q_start[2],
                                       -q_start[3]};
                    float q[COLOR_CONFORMER_LANES][4]{};
                    float t[COLOR_CONFORMER_LANES][3]{};
                    float cache[COLOR_CONFORMER_LANES][7]{};
                    for (int lane = 0; lane < n_conformers; ++lane)
                        q[lane][0] = 1.0f;

                    for (int step = 0; step < steps; ++step) {
                        float q_eff[COLOR_CONFORMER_LANES][4]{};
                        float tphys[COLOR_CONFORMER_LANES][3]{};
                        float gs[COLOR_CONFORMER_LANES][7]{};
                        float sab[COLOR_CONFORMER_LANES]{};
                        PreparedReducedPose poses[COLOR_CONFORMER_LANES]{};
                        const PreparedReducedPose*
                            pose_ptrs[COLOR_CONFORMER_LANES]{};
                        if (trajectory_shape) {
                            float matrices[COLOR_CONFORMER_LANES][9]{};
                            ShapePoseRequest requests[COLOR_CONFORMER_LANES]{};
                            ShapeEvaluation
                                evaluations[COLOR_CONFORMER_LANES]{};
                            for (int lane = 0; lane < n_conformers; ++lane) {
                                quat_mul(q[lane], q_conj, q_eff[lane]);
                                const float* matrix = nullptr;
                                const float* translation = nullptr;
                                if (shared_pose_mode) {
                                    poses[lane] = prepare_reduced_pose_fp32(
                                        q_eff[lane], t[lane]);
                                    pose_ptrs[lane] = &poses[lane];
                                    matrix = poses[lane].matrix;
                                    translation = poses[lane].translation;
                                    if (with_census || with_timers)
                                        ++stats.shared_pose_preparations;
                                } else {
                                    quat_to_mat(q_eff[lane], matrices[lane]);
                                    for (int k = 0; k < 9; ++k)
                                        matrices[lane][k] *= COORD_SCALE;
                                    matrix = matrices[lane];
                                    translation = t[lane];
                                }
                                for (int k = 0; k < 3; ++k)
                                    tphys[lane][k] = t[lane][k] / COORD_SCALE;
                                requests[lane] = {
                                    &shape_b[lane],
                                    nullptr,
                                    matrix,
                                    translation,
                                    1.0f,
                                    shared_pose_mode ? &poses[lane] : nullptr};
                            }
                            const auto tick = with_timers ? Clock::now()
                                                          : Clock::time_point{};
                            trajectory_shape_evaluator->evaluate_batch(
                                requests, size_t(n_conformers),
                                trajectory_shape_batch.get(), evaluations);
                            if (with_timers)
                                stats.shape_gradient_s +=
                                    std::chrono::duration<double>(Clock::now() -
                                                                  tick)
                                        .count();
                            for (int lane = 0; lane < n_conformers; ++lane) {
                                sab[lane] = evaluations[lane].s_ab;
                                std::copy(evaluations[lane].descent,
                                          evaluations[lane].descent + 7,
                                          gs[lane]);
                                if (with_census) {
                                    ++stats.shape_gradient_evaluations;
                                    count_prepared_shape(evaluations[lane],
                                                         stats);
                                }
                            }
                        } else {
                            for (int lane = 0; lane < n_conformers; ++lane) {
                                quat_mul(q[lane], q_conj, q_eff[lane]);
                                float legacy_matrix[9]{};
                                const float* matrix = nullptr;
                                const float* translation = nullptr;
                                if (shared_pose_mode) {
                                    poses[lane] = prepare_reduced_pose_fp32(
                                        q_eff[lane], t[lane]);
                                    pose_ptrs[lane] = &poses[lane];
                                    matrix = poses[lane].matrix;
                                    translation = poses[lane].translation;
                                    if (with_census || with_timers)
                                        ++stats.shared_pose_preparations;
                                } else {
                                    quat_to_mat(q_eff[lane], legacy_matrix);
                                    for (int k = 0; k < 9; ++k)
                                        legacy_matrix[k] *= COORD_SCALE;
                                    matrix = legacy_matrix;
                                    translation = t[lane];
                                }
                                auto tick = with_timers ? Clock::now()
                                                        : Clock::time_point{};
                                if (shape_evaluator_set) {
                                    ShapeEvaluation evaluation;
                                    const ShapeLibraryWorkspace* workspace =
                                        shape_library_workspaces.empty()
                                            ? nullptr
                                            : &shape_library_workspaces
                                                  [bi[lane]];
                                    const ShapePoseRequest request{
                                        &shape_b[lane],
                                        workspace,
                                        matrix,
                                        translation,
                                        1.0f,
                                        shared_pose_mode ? &poses[lane] : nullptr};
                                    evaluate_prepared_shape(
                                        *shape_evaluator_set
                                             ->evaluators[ai[lane]],
                                        request, evaluation);
                                    sab[lane] = evaluation.s_ab;
                                    std::copy(evaluation.descent,
                                              evaluation.descent + 7, gs[lane]);
                                    if (with_census) {
                                        ++stats.shape_gradient_evaluations;
                                        count_prepared_shape(evaluation, stats);
                                    }
                                } else {
                                    get_gradient(shape_a[lane], shape_b[lane],
                                                 matrix, translation, 1.0f,
                                                 gs[lane], &sab[lane], nullptr);
                                    if (with_census) {
                                        ++stats.shape_gradient_evaluations;
                                        stats.shape_gradient_terms +=
                                            uint64_t(shape_a[lane].n_atoms) *
                                            uint64_t(shape_b[lane].n_atoms);
                                        stats.shape_issued_terms +=
                                            uint64_t(shape_a[lane].n_atoms) *
                                            uint64_t(shape_b[lane].n_atoms);
                                    }
                                }
                                if (with_timers)
                                    stats.shape_gradient_s +=
                                        std::chrono::duration<double>(
                                            Clock::now() - tick)
                                            .count();
                                for (int k = 0; k < 3; ++k)
                                    tphys[lane][k] = t[lane][k] / COORD_SCALE;
                            }
                        }
                        ColorEvaluation colors[COLOR_CONFORMER_LANES];
                        uint64_t batch_terms[COLOR_CONFORMER_LANES]{};
                        const ColorFieldBundle*
                            field_contexts[COLOR_CONFORMER_LANES]{};
                        if (use_color_fields)
                            for (int lane = 0; lane < n_conformers; ++lane)
                                field_contexts[lane] =
                                    field_bundles[ai[lane]].get();
                        auto tick =
                            with_timers ? Clock::now() : Clock::time_point{};
                        if (shared_pose_mode) {
                            evaluate_color_conformer_batch8_prepared(
                                prepared_color_batch, *shared_plan, pose_ptrs,
                                true, colors, batch_terms,
                                with_census ? &stats.color_conformer_census
                                            : nullptr,
                                with_timers ? &conformer_phase_times : nullptr,
                                &block_conformer_workspace,
                                use_color_invariant_plan,
                                use_color_fields ? field_contexts : nullptr,
                                with_census && use_color_fields
                                    ? &stats.color_conformer_field_stats
                                    : nullptr,
                                factor_canonical_torque, optimizer_exponential,
                                canonical_color_kernel);
                        } else {
                            evaluate_color_conformer_batch8(
                                color_batch, mat.view(), q_eff, tphys, true,
                                colors, batch_terms,
                                with_census ? &stats.color_conformer_census
                                            : nullptr,
                                with_timers ? &conformer_phase_times : nullptr,
                                &block_conformer_workspace,
                                use_color_fields ? field_contexts : nullptr,
                                with_census && use_color_fields
                                    ? &stats.color_conformer_field_stats
                                    : nullptr);
                        }
                        if (with_timers)
                            stats.color_gradient_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                        for (int lane = 0; lane < n_conformers; ++lane) {
                            if (with_census) {
                                ++stats.color_gradient_evaluations;
                                stats.color_gradient_terms += batch_terms[lane];
                            }
                            tick = with_timers ? Clock::now()
                                               : Clock::time_point{};
                            float mixed_g[7]{};
                            mixed_descent(gs[lane], sab[lane], saa[ai[lane]],
                                          sbb[bi[lane]], colors[lane],
                                          caa[ai[lane]], cbb[bi[lane]], mixing,
                                          mixed_g);
                            if (with_timers)
                                stats.combo_weighting_s +=
                                    std::chrono::duration<double>(Clock::now() -
                                                                  tick)
                                        .count();
                            tick = with_timers ? Clock::now()
                                               : Clock::time_point{};
                            adagrad_exp_step(q[lane], t[lane], mixed_g,
                                             cache[lane], lr_q, lr_t_scaled);
                            if (with_timers)
                                stats.pose_update_s +=
                                    std::chrono::duration<double>(Clock::now() -
                                                                  tick)
                                        .count();
                        }
                    }

                    for (int lane = 0; lane < n_conformers; ++lane) {
                        auto tick =
                            with_timers ? Clock::now() : Clock::time_point{};
                        // This start's converged pose: exact shape and color,
                        // Tanimoto, mixing; the best start so far (strict >) is
                        // retained.
                        float q_eff[4], m[9];
                        quat_mul(q[lane], q_conj, q_eff);
                        PreparedReducedPose final_pose;
                        if (shared_pose_mode) {
                            final_pose =
                                prepare_reduced_pose_fp32(q_eff, t[lane]);
                            if (with_census || with_timers)
                                ++stats.shared_pose_preparations;
                            std::copy(final_pose.matrix, final_pose.matrix + 9,
                                      m);
                        } else {
                            quat_to_mat(q_eff, m);
                            for (int k = 0; k < 9; ++k)
                                m[k] *= COORD_SCALE;
                        }
                        float tphys[3] = {t[lane][0] / COORD_SCALE,
                                          t[lane][1] / COORD_SCALE,
                                          t[lane][2] / COORD_SCALE};
                        float treport[3] = {tphys[0] * COORD_SCALE,
                                            tphys[1] * COORD_SCALE,
                                            tphys[2] * COORD_SCALE};
                        const float* final_matrix =
                            shared_pose_mode ? final_pose.matrix : m;
                        const float* final_translation =
                            shared_pose_mode ? final_pose.translation : treport;
                        const float shape_cross = pair_overlap_at_pose(
                            shape_a[lane], shape_b[lane], final_matrix,
                            final_translation, nullptr);
                        double color_cross = 0.0;
                        if (shared_pose_mode) {
                            color_cross =
                                score_exact_final_color(
                                    shared_pose_queries
                                        ->reduced_query_colors[ai[lane]],
                                    lv[bi[lane]], *shared_plan, final_pose, 3,
                                    ls,
                                    reuse_exact_final_scratch
                                        ? &exact_scratch[lane]
                                        : nullptr,
                                    with_census
                                        ? &stats.color_per_start_final_terms
                                        : nullptr,
                                    with_census || with_timers
                                        ? &exact_final_stats
                                        : nullptr,
                                    with_timers)
                                    .value;
                        } else {
                            const auto posed =
                                pose_color(lv[bi[lane]], ls, q_eff, tphys);
                            color_cross = overlap(
                                qv[ai[lane]], lv[bi[lane]], mat, qs, ls,
                                posed.data(),
                                with_census ? &stats.color_per_start_final_terms
                                            : nullptr);
                        }
                        if (with_timers)
                            stats.per_start_final_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                        if (with_census)
                            ++stats.color_per_start_final_evaluations;
                        const float ts = float(tanimoto(
                            shape_cross, saa[ai[lane]], sbb[bi[lane]]));
                        const float tc = float(tanimoto(
                            color_cross, caa[ai[lane]], cbb[bi[lane]]));
                        const float mixed = (1.0f - mixing) * ts + mixing * tc;
                        if (converged_start_is_better(mixed,
                                                      best_mixed[lane])) {
                            best_mixed[lane] = mixed;
                            best_start[lane] = start;
                            if (shared_pose_mode) {
                                best_pose[lane] = final_pose;
                                best_shape_cross[lane] = shape_cross;
                                best_color_cross[lane] = color_cross;
                            } else {
                                for (int k = 0; k < 4; ++k)
                                    best_q[lane][k] = q_eff[k];
                                for (int k = 0; k < 3; ++k)
                                    best_t[lane][k] = tphys[k];
                            }
                        }
                    }
                }
                for (int lane = 0; lane < n_conformers; ++lane) {
                    if (shared_pose_mode)
                        write_winner_shared(
                            worker, row[lane], ai[lane], bi[lane],
                            best_pose[lane], best_start[lane],
                            best_shape_cross[lane], best_color_cross[lane]);
                    else
                        write_winner(worker, row[lane], ai[lane], bi[lane],
                                     shape_a[lane], shape_b[lane], best_q[lane],
                                     best_t[lane], best_start[lane]);
                }
            },
            with_worker_timers ? &worker_elapsed_seconds : nullptr,
            with_worker_timers ? &worker_placements : nullptr,
            require_worker_affinity, executor.get(), rows_per_block,
            executor ? executor->claim_chunk_rows() : 256,
            with_worker_timers ? &scoring_claims_by_worker : nullptr);
#else
        throw std::invalid_argument(
            "AVX2 conformer color grain unavailable");
#endif
    } else {
        // ---------------------------------------------------------------------
        // Branch 3: one pair per work unit, the reference structure that the
        // other branches batch. Grain 0, and grain 16 at nonstandard start
        // modes (run as scalar grain 0). Legacy-pose color goes through
        // the ColorEvaluator seam (analytic or channel fields); shared-pose
        // color through the shared analytic or field adapters.
        // ---------------------------------------------------------------------
        run_combo_workers(
            pair_rows, threads,
            [&](int worker, size_t row) {
                ComboWorkerScratch& worker_scratch =
                    combo_worker_scratch(size_t(worker), executor.get());
                const size_t ai = size_t(qi[row]), bi = size_t(li[row]);
                const int analytic_grain =
                    is_color_start_grain(color_grain) ? 0 : color_grain;
                AnalyticColorEvaluatorContext evaluator_context{
                    &mat, analytic_grain};
                ChannelColorEvaluatorContext field_context{
                    field_bundles[ai].get(),
                    &mat,
                    with_census ? &stats.color_field_samples : nullptr,
                    with_census ? &stats.color_field_interpolated_samples
                                : nullptr,
                    with_census ? &stats.color_field_analytic_selected_samples
                                : nullptr,
                    with_census ? &stats.color_field_analytic_selected_terms
                                : nullptr,
                    with_census ? &stats.color_field_analytic_fallback_samples
                                : nullptr,
                    with_census ? &stats.color_field_inactive_fallback_samples
                                : nullptr,
                    with_census
                        ? &stats.color_field_out_of_grid_fallback_samples
                        : nullptr,
                    with_census ? &stats.color_field_analytic_fallback_terms
                                : nullptr};
                const ColorEvaluator evaluator =
                    use_color_fields
                        ? ColorEvaluator{&field_context,
                                         &evaluate_channel_color_fields}
                        : ColorEvaluator{&evaluator_context,
                                         &evaluate_analytic_color};
                const ColorInteractionMatrixView interaction = mat.view();
                const Conformer& a = shape_query_views[ai];
                const Conformer& b = shape_library_views[bi];
                float best_mixed = -std::numeric_limits<float>::infinity();
                float best_q[4] = {1, 0, 0, 0}, best_t[3] = {0, 0, 0};
                PreparedReducedPose best_prepared_pose;
                float best_shape_cross = 0.0f;
                double best_color_cross = 0.0;
                int best_start = 0;
                ExactFinalScratch& exact_scratch =
                    worker_scratch.exact_final[0];
                if (shared_pose_mode && reuse_exact_final_scratch)
                    exact_scratch.begin_trajectory(
                        shared_pose_queries->reduced_query_colors[ai], lv[bi],
                        *shared_pose_queries->color_plan, 3, ls);
                // Each start: run `steps` Adagrad steps from the identity,
                // composing with the start rotation (a signed permutation,
                // whose inverse is its conjugate), then score the converged
                // pose exactly and keep the best start.
                for (int start = 0; start < starts; ++start) {
                    float q_start[4];
                    start_quaternion(start_mode, start, q_start);
                    float start_matrix[9];
                    quat_to_mat(q_start, start_matrix);
                    if (!matrix_is_signed_permutation(start_matrix,
                                                      SIGNED_PERM_TOL))
                        throw std::invalid_argument(
                            "combo start is not a signed permutation");
                    float q_conj[4] = {q_start[0], -q_start[1], -q_start[2],
                                       -q_start[3]};
                    float q[4] = {1, 0, 0, 0}, t[3] = {0, 0, 0};
                    float cache[7] = {0, 0, 0, 0, 0, 0, 0};
                    for (int step = 0; step < steps; ++step) {
                        float q_eff[4], m[9];
                        quat_mul(q, q_conj, q_eff);
                        // Shared contract: one FP32 pose per step for shape and
                        // color alike. Legacy: shape gets s * R with the
                        // reduced translation, color the quaternion with t / s.
                        PreparedReducedPose step_pose;
                        if (shared_pose_mode) {
                            step_pose = prepare_reduced_pose_fp32(q_eff, t);
                            if (with_census || with_timers)
                                ++stats.shared_pose_preparations;
                            std::copy(step_pose.matrix, step_pose.matrix + 9,
                                      m);
                        } else {
                            quat_to_mat(q_eff, m);
                            for (int k = 0; k < 9; ++k)
                                m[k] *= COORD_SCALE;
                        }
                        const float* shape_matrix =
                            shared_pose_mode ? step_pose.matrix : m;
                        const float* shape_translation =
                            shared_pose_mode ? step_pose.translation : t;
                        float gs[7], sab = 0.0f;
                        auto tick =
                            with_timers ? Clock::now() : Clock::time_point{};
                        if (shape_evaluator_set) {
                            ShapeEvaluation evaluation;
                            const ShapeLibraryWorkspace* workspace =
                                shape_library_workspaces.empty()
                                    ? nullptr
                                    : &shape_library_workspaces[bi];
                            const ShapePoseRequest shape_request{
                                &b,
                                workspace,
                                shape_matrix,
                                shape_translation,
                                1.0f,
                                shared_pose_mode ? &step_pose : nullptr};
                            evaluate_prepared_shape(
                                *shape_evaluator_set->evaluators[ai],
                                shape_request, evaluation);
                            sab = evaluation.s_ab;
                            std::copy(evaluation.descent,
                                      evaluation.descent + 7, gs);
                            if (with_census) {
                                ++stats.shape_gradient_evaluations;
                                count_prepared_shape(evaluation, stats);
                            }
                        } else {
                            get_gradient(a, b, shape_matrix, shape_translation,
                                         1.0f, gs, &sab, nullptr);
                        }
                        if (with_timers)
                            stats.shape_gradient_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                        if (with_census && !shape_evaluator_set) {
                            ++stats.shape_gradient_evaluations;
                            const uint64_t terms =
                                uint64_t(a.n_atoms) * uint64_t(b.n_atoms);
                            stats.shape_gradient_terms += terms;
                            stats.shape_issued_terms += terms;
                        }
                        float tphys[3] = {t[0] / COORD_SCALE,
                                          t[1] / COORD_SCALE,
                                          t[2] / COORD_SCALE};
                        tick = with_timers ? Clock::now() : Clock::time_point{};
                        // Color at this step's pose: shared adapters or the
                        // legacy evaluator seam.
                        ColorEvaluation color;
                        if (shared_pose_mode) {
                            if (use_color_fields) {
                                color = evaluate_shared_channel_color_fields(
                                    field_context,
                                    shared_pose_queries
                                        ->reduced_query_colors[ai],
                                    lv[bi], ls, shared_reduced_matrix,
                                    *shared_pose_queries->color_plan, step_pose,
                                    true,
                                    with_census ? &stats.color_gradient_terms
                                                : nullptr);
                            } else {
                                color = evaluate_shared_analytic_color(
                                    shared_pose_queries
                                        ->reduced_query_colors[ai],
                                    lv[bi], shared_reduced_matrix, 3, ls,
                                    step_pose, analytic_grain,
                                    with_census ? &stats.color_gradient_terms
                                                : nullptr,
                                    true);
                            }
                        } else {
                            ColorEvaluationRequest request;
                            request.query = &qv[ai];
                            request.library = &lv[bi];
                            request.interaction = interaction;
                            request.query_stride = qs;
                            request.library_stride = ls;
                            request.quaternion = q_eff;
                            request.translation_physical = tphys;
                            request.gradient = true;
                            color = evaluate_color(
                                evaluator, request,
                                with_census ? &stats.color_gradient_terms
                                            : nullptr);
                        }
                        if (with_timers)
                            stats.color_gradient_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                        if (with_census) ++stats.color_gradient_evaluations;
                        tick = with_timers ? Clock::now() : Clock::time_point{};
                        float mixed_g[7]{};
                        mixed_descent(gs, sab, saa[ai], sbb[bi], color, caa[ai],
                                      cbb[bi], mixing, mixed_g);
                        if (with_timers)
                            stats.combo_weighting_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                        tick = with_timers ? Clock::now() : Clock::time_point{};
                        adagrad_exp_step(q, t, mixed_g, cache, lr_q,
                                         lr_t_scaled);
                        if (with_timers)
                            stats.pose_update_s +=
                                std::chrono::duration<double>(Clock::now() -
                                                              tick)
                                    .count();
                    }
                    auto tick =
                        with_timers ? Clock::now() : Clock::time_point{};
                    float q_eff[4], m[9];
                    quat_mul(q, q_conj, q_eff);
                    PreparedReducedPose final_pose;
                    if (shared_pose_mode) {
                        final_pose = prepare_reduced_pose_fp32(q_eff, t);
                        if (with_census || with_timers)
                            ++stats.shared_pose_preparations;
                        std::copy(final_pose.matrix, final_pose.matrix + 9, m);
                    } else {
                        quat_to_mat(q_eff, m);
                        for (int k = 0; k < 9; ++k)
                            m[k] *= COORD_SCALE;
                    }
                    float tphys[3] = {t[0] / COORD_SCALE, t[1] / COORD_SCALE,
                                      t[2] / COORD_SCALE};
                    // Select on exactly the score we can reproduce from the reported
                    // float32 physical pose, including its translation round trip.
                    float treport[3] = {tphys[0] * COORD_SCALE,
                                        tphys[1] * COORD_SCALE,
                                        tphys[2] * COORD_SCALE};
                    const float* final_matrix =
                        shared_pose_mode ? final_pose.matrix : m;
                    const float* final_translation =
                        shared_pose_mode ? final_pose.translation : treport;
                    const float sab = pair_overlap_at_pose(
                        a, b, final_matrix, final_translation, nullptr);
                    double cab = 0.0;
                    if (shared_pose_mode) {
                        uint64_t exact_terms = 0;
                        cab =
                            score_exact_final_color(
                                shared_pose_queries->reduced_query_colors[ai],
                                lv[bi], *shared_pose_queries->color_plan,
                                final_pose, 3, ls,
                                reuse_exact_final_scratch ? &exact_scratch
                                                          : nullptr,
                                &exact_terms,
                                with_census || with_timers ? &exact_final_stats
                                                           : nullptr,
                                with_timers)
                                .value;
                        if (with_census)
                            stats.color_per_start_final_terms += exact_terms;
                    } else {
                        auto posed = pose_color(lv[bi], ls, q_eff, tphys);
                        cab = overlap(qv[ai], lv[bi], mat, qs, ls, posed.data(),
                                      with_census
                                          ? &stats.color_per_start_final_terms
                                          : nullptr);
                    }
                    if (with_timers)
                        stats.per_start_final_s +=
                            std::chrono::duration<double>(Clock::now() - tick)
                                .count();
                    if (with_census) ++stats.color_per_start_final_evaluations;
                    const float ts = float(tanimoto(sab, saa[ai], sbb[bi]));
                    const float tc = float(tanimoto(cab, caa[ai], cbb[bi]));
                    const float mixed = (1.0f - mixing) * ts + mixing * tc;
                    if (converged_start_is_better(mixed, best_mixed)) {
                        best_mixed = mixed;
                        best_start = start;
                        if (shared_pose_mode) {
                            best_prepared_pose = final_pose;
                            best_shape_cross = sab;
                            best_color_cross = cab;
                        } else {
                            for (int k = 0; k < 4; ++k)
                                best_q[k] = q_eff[k];
                            for (int k = 0; k < 3; ++k)
                                best_t[k] = tphys[k];
                        }
                    }
                }
                if (shared_pose_mode)
                    write_winner_shared(worker, row, ai, bi, best_prepared_pose,
                                        best_start, best_shape_cross,
                                        best_color_cross);
                else
                    write_winner(worker, row, ai, bi, a, b, best_q, best_t,
                                 best_start);
            },
            with_worker_timers ? &worker_elapsed_seconds : nullptr,
            with_worker_timers ? &worker_placements : nullptr,
            require_worker_affinity, executor.get(), 1,
            executor ? executor->claim_chunk_rows() : 256,
            with_worker_timers ? &scoring_claims_by_worker : nullptr);
    }
    // =========================================================================
    // Result details. Always: final-policy flags, converged-objective counts,
    // the effective optimizer exp and library self-evaluation counts. With
    // census, timers or worker timers: configuration and shape stamps. With
    // census: evaluation and term counts per phase and each family's lane
    // census. With timers: phase seconds, which nest and overlap and are never
    // rates. With fields (and census or timers): field bytes, timings and
    // per-query geometry. With worker timers: per-worker elapsed time, claims
    // and observed affinity.
    // =========================================================================
    py::dict details;
    details["winner_only_final_requested"] = winner_policy;
    details["final_start_policy_requested"] =
        winner_policy ? "winner" : "all_start";
    details["final_rescore_compiled_mode"] = R3_FINAL_RESCORE_MODE;
    details["winner_only_final_pairs"] = stats.winner_only_final_pairs;
    details["converged_objective_evaluations"] =
        stats.converged_objective_evaluations;
    details["converged_objective_color_terms"] =
        stats.converged_objective_color_terms;
    details["analytic_final_shape_evaluations"] =
        stats.analytic_final_shape_evaluations;
    details["converged_objective_seconds"] = stats.converged_objective_s;
    if (objective_trace) {
        details["converged_objectives"] = converged_objectives;
        details["converged_pose_bytes"] = converged_pose_bytes;
        details["converged_objective_self"] = converged_objective_self;
    }
    details["optimizer_exp_mode_effective"] =
        optimizer_exponential == OptimizerExpMode::RawBitsUnclamped
            ? "rawbits_unclamped"
        : optimizer_exponential == OptimizerExpMode::RawBits ? "rawbits"
                                                             : "baseline";
    // Existing counters are collected regardless of clocks. Expose them for
    // fail-closed benchmark attribution without enabling timing instrumentation.
    details["library_shape_self_evaluations"] =
        stats.shape_library_self_evaluations;
    details["library_color_self_evaluations"] =
        stats.color_library_self_evaluations;
    if (with_library_self_timers) {
        details["library_shape_self_seconds"] = library_shape_self_seconds;
        details["library_color_self_seconds"] = library_color_self_seconds;
    }
    if (with_census || with_timers || with_worker_timers) {
        details["pose_contract"] = pose_contract;
        details["color_grain"] = color_grain;
        details["start_mode"] = start_mode;
        details["color_invariant_plan_enabled"] =
            shared_pose_mode && use_color_invariant_plan;
        details["exact_final_scratch_enabled"] =
            shared_pose_mode && reuse_exact_final_scratch;
        details["shared_pose_preparations"] = stats.shared_pose_preparations;
        if (shared_pose_mode) {
            details["shared_query_preparation_seconds"] =
                shared_pose_queries->preparation_seconds;
            details["shared_query_reduced_color_bytes"] =
                shared_pose_queries->reduced_color_bytes;
            details["shared_shape_field_build_seconds"] =
                shared_pose_queries->shape_evaluators->field_build_seconds;
            details["shared_shape_field_bytes"] =
                shared_pose_queries->shape_evaluators->field_bytes;
        }
        ShapeEvaluatorStamp shape_stamp{
            "scalar_analytic", "scalar", "scalar", "source_order",
            "libm_exp2",       1,        false,    false};
        if (shape_evaluator_set && !shape_evaluator_set->evaluators.empty())
            shape_stamp = shape_evaluator_set->evaluators.front()->stamp();
        py::dict selected_shape;
        selected_shape["backend"] = shape_stamp.backend;
        selected_shape["grain"] = shape_stamp.grain;
        selected_shape["isa"] = shape_stamp.isa;
        selected_shape["reduction"] = shape_stamp.reduction;
        selected_shape["exponential"] = shape_stamp.exponential;
        selected_shape["width"] = shape_stamp.width;
        selected_shape["vector_path"] = shape_stamp.vector_path;
        selected_shape["requires_library_soa"] =
            shape_stamp.requires_library_soa;
        selected_shape["field_spacing_angstrom"] = shape_field_spacing;
        selected_shape["field_padding_angstrom"] = shape_field_padding;
        details["shape_evaluator_stamp"] = selected_shape;
        details["shape_query_preparation_seconds"] =
            shape_query_preparation_seconds;
        details["shape_query_preparation_in_call"] =
            shape_query_preparation_in_call;
        details["shape_query_snapshot_bytes"] =
            shape_evaluator_set ? shape_evaluator_set->query_snapshot_bytes : 0;
        details["shape_query_workspace_bytes"] =
            shape_evaluator_set ? shape_evaluator_set->query_workspace_bytes
                                : 0;
        details["shape_query_field_build_seconds"] =
            shape_evaluator_set ? shape_evaluator_set->field_build_seconds
                                : 0.0;
        details["shape_query_field_bytes"] =
            shape_evaluator_set ? shape_evaluator_set->field_bytes : 0;
        details["shape_query_field_nodes"] =
            shape_evaluator_set ? shape_evaluator_set->field_nodes : 0;
        details["shape_target_workspace_preparation_seconds"] =
            stats.shape_target_workspace_prepare_s;
        details["shape_target_workspace_prepared_bytes"] =
            stats.shape_workspace_bytes;
    }
    if (with_census) {
        // Reports the compiled final mode only if such a batch actually ran.
        details["final_rescore_mode"] =
            stats.final_rescore_batches ? R3_FINAL_RESCORE_MODE : 0;
        details["final_rescore_batches"] = stats.final_rescore_batches;
        details["final_rescore_independent_pairs"] = stats.final_rescore_pairs;
        const auto& final_counts = stats.final_rescore_counts;
        details["final_rescore_shape_terms"] = final_counts.shape_terms;
        details["final_rescore_color_terms"] = final_counts.color_terms;
        details["final_rescore_shape_issued_lanes"] =
            final_counts.shape_issued_lanes;
        details["final_rescore_color_issued_lanes"] =
            final_counts.color_issued_lanes;
        details["final_rescore_accurate_fallback_pairs"] =
            final_counts.accurate_fallback_pairs;
        details["final_rescore_packed_coordinate_bytes"] =
            final_counts.packed_coordinate_bytes;
        details["final_rescore_horizontal_reductions"] =
            final_counts.horizontal_reductions;
        details["final_rescore_temporary_allocations"] =
            final_counts.temporary_allocations;
        details["final_rescore_allocated_payload_bytes"] =
            final_counts.allocated_payload_bytes;
        details["color_gradient_evaluations"] =
            stats.color_gradient_evaluations;
        details["color_per_start_final_evaluations"] =
            stats.color_per_start_final_evaluations;
        details["color_winner_rescore_evaluations"] =
            stats.color_winner_rescore_evaluations;
        details["color_evaluations"] = stats.color_gradient_evaluations +
                                       stats.color_per_start_final_evaluations +
                                       stats.color_winner_rescore_evaluations;
        details["color_gradient_terms"] = stats.color_gradient_terms;
        details["color_per_start_final_terms"] =
            stats.color_per_start_final_terms;
        details["color_winner_rescore_terms"] =
            stats.color_winner_rescore_terms;
        details["color_query_self_terms"] = stats.color_query_self_terms;
        details["color_library_self_terms"] = stats.color_library_self_terms;
        details["shape_query_self_terms"] = stats.shape_query_self_terms;
        details["shape_library_self_terms"] = stats.shape_library_self_terms;
        details["shape_query_self_evaluations"] =
            stats.shape_query_self_evaluations;
        details["shape_library_self_evaluations"] =
            stats.shape_library_self_evaluations;
        details["color_query_self_evaluations"] =
            stats.color_query_self_evaluations;
        details["color_library_self_evaluations"] =
            stats.color_library_self_evaluations;
        details["prepared_self_overlaps_used"] =
            stats.prepared_self_overlaps_used;
        details["prepared_shape_query_self_used"] =
            stats.prepared_shape_query_self_used;
        details["prepared_shape_library_self_used"] =
            stats.prepared_shape_library_self_used;
        details["prepared_color_query_self_used"] =
            stats.prepared_color_query_self_used;
        details["prepared_color_library_self_used"] =
            stats.prepared_color_library_self_used;
        details["shared_shape_query_self_used"] =
            stats.shared_shape_query_self_used;
        details["shared_color_query_self_used"] =
            stats.shared_color_query_self_used;
        details["shape_gradient_terms"] = stats.shape_gradient_terms;
        details["shape_gradient_evaluations"] =
            stats.shape_gradient_evaluations;
        details["shape_issued_terms"] = stats.shape_issued_terms;
        details["shape_field_samples"] = stats.shape_field_samples;
        details["shape_fallback_evaluations"] =
            stats.shape_fallback_evaluations;
        details["shape_fallback_terms"] = stats.shape_fallback_terms;
        details["color_canonical"] = mat.canonical;
        if (collect_winning_starts) details["winning_starts"] = winning_starts;
        details["color_field_current_four_start_field_groups"] =
            stats.color_start_field_stats.current_four_start_field_groups;
        details["color_field_current_four_start_remainder_groups"] =
            stats.color_start_field_stats.current_four_start_remainder_groups;
        details["color_field_samples"] =
            stats.color_field_samples +
            stats.color_conformer_field_stats.samples +
            stats.color_start_field_stats.samples;
        details["color_field_interpolated_samples"] =
            stats.color_field_interpolated_samples +
            stats.color_conformer_field_stats.interpolated_samples +
            stats.color_start_field_stats.interpolated_samples;
        details["color_field_interpolated_terms"] =
            stats.color_conformer_field_stats.interpolated_terms +
            stats.color_start_field_stats.interpolated_terms;
        details["color_field_analytic_selected_samples"] =
            stats.color_field_analytic_selected_samples +
            stats.color_conformer_field_stats.analytic_selected_samples +
            stats.color_start_field_stats.analytic_selected_samples;
        details["color_field_analytic_selected_terms"] =
            stats.color_field_analytic_selected_terms +
            stats.color_conformer_field_stats.analytic_selected_terms +
            stats.color_start_field_stats.analytic_selected_terms;
        details["color_field_analytic_fallback_samples"] =
            stats.color_field_analytic_fallback_samples +
            stats.color_conformer_field_stats.analytic_fallback_samples +
            stats.color_start_field_stats.analytic_fallback_samples;
        details["color_field_inactive_fallback_samples"] =
            stats.color_field_inactive_fallback_samples +
            stats.color_conformer_field_stats.inactive_fallback_samples +
            stats.color_start_field_stats.inactive_fallback_samples;
        details["color_field_out_of_grid_fallback_samples"] =
            stats.color_field_out_of_grid_fallback_samples +
            stats.color_conformer_field_stats.out_of_grid_fallback_samples +
            stats.color_start_field_stats.out_of_grid_fallback_samples;
        details["color_field_analytic_fallback_terms"] =
            stats.color_field_analytic_fallback_terms +
            stats.color_conformer_field_stats.analytic_fallback_terms +
            stats.color_start_field_stats.analytic_fallback_terms;
        if (shared_pose_mode) {
            details["exact_final_scratch_buffer_growths"] =
                exact_final_stats.buffer_growths;
            details["exact_final_scratch_payload_bytes_allocated"] =
                exact_final_stats.payload_bytes_allocated;
            details["exact_final_library_points_prepared"] =
                exact_final_stats.library_points_prepared;
            details["exact_final_winner_candidates_retained"] =
                exact_final_stats.winner_candidates_retained;
            details["exact_final_winner_point_bytes_copied"] =
                exact_final_stats.winner_point_bytes_copied;
        }
        if (is_color_conformer_grain(color_grain)) {
            std::vector<uint64_t> useful(size_t(nf) * size_t(nf));
            std::vector<uint64_t> issued(size_t(nf) * size_t(nf));
            std::vector<uint64_t> count_bins(size_t(nf) * 19);
            for (int ta = 0; ta < nf; ++ta) {
                for (int tb = 0; tb < nf; ++tb) {
                    const size_t dst = size_t(ta) * size_t(nf) + size_t(tb);
                    const size_t src =
                        size_t(ta) * size_t(MAX_COLOR_FEATURES) + size_t(tb);
                    useful[dst] = stats.color_conformer_census
                                      .useful_terms_by_interaction[src];
                    issued[dst] = stats.color_conformer_census
                                      .issued_lanes_by_interaction[src];
                }
                for (int bin = 0; bin < 19; ++bin)
                    count_bins[size_t(ta) * 19 + size_t(bin)] =
                        stats.color_conformer_census
                            .target_count_bins[size_t(ta) * 19 + size_t(bin)];
            }
            details["color_conformer_vector_evaluations"] =
                stats.color_conformer_census.vector_evaluations;
            details["color_conformer_scalar_fallback_pairs"] =
                stats.color_conformer_census.scalar_fallback_pairs;
            details["color_conformer_interaction_matrix_shape"] =
                py::make_tuple(nf, nf);
            details["color_conformer_useful_terms_by_interaction"] = useful;
            details["color_conformer_issued_lanes_by_interaction"] = issued;
            details["color_conformer_target_count_bins_by_type"] = count_bins;
            details["color_conformer_census_weighting"] =
                "one observation per real pair lane and target type; useful "
                "terms include exact scalar fallback and field-interpolated "
                "work; issued lanes count AVX2 analytic terms only";
        }
        if (is_color_start_grain(color_grain) && start_mode == 1) {
            std::vector<uint64_t> useful(size_t(nf) * size_t(nf));
            std::vector<uint64_t> issued(size_t(nf) * size_t(nf));
            std::vector<uint64_t> count_bins(size_t(nf) * 19);
            for (int ta = 0; ta < nf; ++ta) {
                for (int tb = 0; tb < nf; ++tb) {
                    const size_t dst = size_t(ta) * size_t(nf) + size_t(tb);
                    const size_t src =
                        size_t(ta) * size_t(MAX_COLOR_FEATURES) + size_t(tb);
                    useful[dst] = stats.color_start_census
                                      .useful_terms_by_interaction[src];
                    issued[dst] = stats.color_start_census
                                      .issued_lanes_by_interaction[src];
                }
                for (int bin = 0; bin < 19; ++bin)
                    count_bins[size_t(ta) * 19 + size_t(bin)] =
                        stats.color_start_census
                            .target_count_bins[size_t(ta) * 19 + size_t(bin)];
            }
            details["color_start_lane_vector_evaluations"] =
                stats.color_start_census.vector_evaluations;
            details["color_compatible_ymm8_evaluations"] =
                stats.color_start_census.compatible_ymm8_evaluations;
            details["color_compatible_neon4_two_stream_evaluations"] =
                stats.color_start_census
                    .compatible_neon4_two_stream_evaluations;
            details["color_singleton_xmm4_evaluations"] =
#if defined(__x86_64__)
                stats.color_start_census.singleton_four_start_evaluations;
#else
                uint64_t(0);
#endif
            details["color_singleton_neon4_evaluations"] =
#if defined(__aarch64__)
                stats.color_start_census.singleton_four_start_evaluations;
#else
                uint64_t(0);
#endif
            details["color_singleton_four_start_evaluations"] =
                stats.color_start_census.singleton_four_start_evaluations;
            details["color_start_lane_scalar_fallback_pairs"] =
                stats.color_start_census.scalar_fallback_pairs;
            details["color_start_lane_interaction_matrix_shape"] =
                py::make_tuple(nf, nf);
            details["color_start_lane_useful_terms_by_interaction"] = useful;
            details["color_start_lane_issued_lanes_by_interaction"] = issued;
            details["color_start_lane_target_count_bins_by_type"] = count_bins;
            details["color_start_lane_census_weighting"] =
                "per actual mode-3 gradient call; useful work includes exact "
                "fallback and field-interpolated terms; issued lanes count "
                "AVX2 analytic term lanes only, with no reduction across starts";
            details["color_start_lane_source_coordinate_components_loaded"] =
                stats.color_start_source_coordinate_components_loaded;
            details["color_start_lane_shared_query_broadcast8_calls"] =
                stats.color_start_shared_query_broadcast8_calls;
            details["color_start_lane_paired_query_broadcast4_calls"] =
                stats.color_start_paired_query_broadcast4_calls;
            details["color_start_lane_scheduled_batch_numeric_bytes"] =
                stats.color_start_scheduled_batch_numeric_bytes;
        }
    }
    if (with_timers) {
        details["shape_gradient"] = stats.shape_gradient_s;
        details["color_gradient"] = stats.color_gradient_s;
        details["combo_weighting"] = stats.combo_weighting_s;
        details["pose_update"] = stats.pose_update_s;
        details["per_start_final_value"] = stats.per_start_final_s;
        details["exact_final_rescore"] = stats.winner_rescore_s;
        if (shared_pose_mode) {
            details["exact_final_coordinate_prepare_seconds"] =
                exact_final_stats.coordinate_prepare_seconds;
            details["exact_final_winner_retention_seconds"] =
                exact_final_stats.winner_retention_seconds;
        }
        if (is_color_conformer_grain(color_grain)) {
            details["color_conformer_stable_sort"] =
                stats.color_conformer_stable_sort_s;
            details["color_conformer_pack"] = stats.color_conformer_pack_s;
            details["color_conformer_pose_transform"] =
                stats.color_conformer_pose_transform_s;
            details["color_conformer_matrix_screen"] =
                stats.color_conformer_matrix_screen_s;
            details["color_conformer_scalar_fallback"] =
                stats.color_conformer_scalar_fallback_s;
            details["color_conformer_pose_soa_pack"] =
                stats.color_conformer_pose_soa_pack_s;
            details["color_conformer_vector_kernel"] =
                stats.color_conformer_vector_kernel_s;
            details["color_conformer_field_sample"] =
                stats.color_conformer_field_sample_s;
            details["color_conformer_phase_timing_scope"] =
                "attribution only; components overlap color_gradient and "
                "must not be used as rates";
        }
        if (is_color_start_grain(color_grain) && start_mode == 1) {
            details["color_start_lane_pack"] = stats.color_start_pack_s;
            details["color_start_lane_pose_transform"] =
                stats.color_start_pose_transform_s;
            details["color_start_lane_matrix_screen"] =
                stats.color_start_matrix_screen_s;
            details["color_start_lane_scalar_fallback"] =
                stats.color_start_scalar_fallback_s;
            details["color_start_lane_pose_soa_pack"] =
                stats.color_start_pose_soa_pack_s;
            details["color_start_lane_vector_kernel"] =
                stats.color_start_vector_kernel_s;
            details["color_start_lane_field_sample"] =
                stats.color_start_field_sample_s;
            details["color_start_lane_phase_timing_scope"] =
                "attribution only; components overlap color_gradient and "
                "must not be used as rates";
        }
    }
    if (use_color_fields && (with_census || with_timers)) {
        details["color_field_prepared_handle_used"] =
            bool(prepared_field_handle);
        if (prepared_field_handle) {
            details["color_field_prepared_build_seconds"] =
                prepared_field_handle->build_seconds;
            details["color_field_prepared_pack_seconds"] =
                prepared_field_handle->pack_seconds;
            details["color_field_prepared_bytes"] =
                prepared_field_handle->bytes;
        }
        if (uniform_field_backend_codes)
            details["color_field_backend_codes"] = field_backend_codes;
        else
            details["color_field_backend_codes"] = field_backend_codes_by_query;
        details["color_field_backend_codes_by_query"] =
            field_backend_codes_by_query;
        details["color_field_bytes"] = stats.color_field_bytes;
        details["color_field_bytes_by_target_type"] =
            stats.color_field_bytes_by_target_type;
        details["color_field_build_seconds"] = stats.color_field_build_s;
        details["color_field_build_seconds_by_target_type"] =
            stats.color_field_build_s_by_target_type;
        details["color_field_pack_seconds"] = stats.color_field_pack_s;
        details["color_field_pack_seconds_by_target_type"] =
            stats.color_field_pack_s_by_target_type;
        details["color_field_bytes_by_backend"] =
            stats.color_field_bytes_by_backend;
        details["color_field_build_seconds_by_backend"] =
            stats.color_field_build_s_by_backend;
        details["color_field_pack_seconds_by_backend"] =
            stats.color_field_pack_s_by_backend;
        details["color_field_active_queries_by_target_type"] =
            stats.color_field_active_queries_by_target_type;
        py::list query_geometry;
        for (size_t row = 0; row < field_bundles.size(); ++row) {
            py::dict query_record;
            query_record["query_row"] = row;
            py::list target_planes;
            for (int tb = 0; tb < nf; ++tb) {
                const ColorFieldBundle& fields = *field_bundles[row];
                py::dict plane_record;
                plane_record["target_type"] = tb;
                plane_record["backend_code"] = fields.backend_code(tb);
                plane_record["sampling_path"] = fields.sampling_path(tb);
                plane_record["payload_address_mod64"] =
                    fields.payload_alignment(tb);
                plane_record["coefficient_layout_version"] =
                    fields.backend_code(tb) == COLOR_FIELD_BRICK_FP32
                        ? fields.coefficient_layout_version()
                        : 0;
                plane_record["source_node_quantization"] =
                    fields.backend_code(tb) == COLOR_FIELD_NODE_FP32
                        ? fields.node32_source_quantization()
                    : fields.backend_code(tb) == COLOR_FIELD_ANALYTIC
                        ? "none"
                        : "half_rounded";
                const bool active = fields.field_active(tb);
                plane_record["active"] = active;
                if (active) {
                    const auto cells = fields.grid_cells(tb);
                    const auto origin = fields.grid_origin_angstrom(tb);
                    plane_record["cells"] =
                        py::make_tuple(cells[0], cells[1], cells[2]);
                    plane_record["origin_angstrom"] =
                        py::make_tuple(origin[0], origin[1], origin[2]);
                    plane_record["spacing_angstrom"] =
                        fields.grid_spacing_angstrom(tb);
                } else {
                    plane_record["cells"] = py::none();
                    plane_record["origin_angstrom"] = py::none();
                    plane_record["spacing_angstrom"] = py::none();
                }
                target_planes.append(std::move(plane_record));
            }
            query_record["target_type_planes"] = std::move(target_planes);
            query_geometry.append(std::move(query_record));
        }
        details["color_field_geometry_by_query"] = std::move(query_geometry);
    }
    if (with_worker_timers) {
        details["worker_elapsed_seconds"] = worker_elapsed_seconds;
        details["scoring_claims_by_worker"] = scoring_claims_by_worker;
        details["workers_with_claims"] = std::count_if(
            scoring_claims_by_worker.begin(), scoring_claims_by_worker.end(),
            [](uint64_t claims) { return claims != 0; });
        py::list placements;
        for (const auto& placement : worker_placements) {
            py::dict item;
            if (placement.requested_cpu < 0)
                item["requested_cpu"] = py::none();
            else
                item["requested_cpu"] = placement.requested_cpu;
            item["affinity_attempted"] = placement.affinity_attempted;
            item["affinity_succeeded"] = placement.affinity_succeeded;
            item["actual_cpus"] = placement.actual_cpus;
            placements.append(std::move(item));
        }
        details["worker_affinity_observed"] = std::move(placements);
    }
    if (return_dense) return py::make_tuple(output, details);
    return py::make_tuple(py::none(), details);
}
}  // namespace

// =========================================================================
// Python bindings. Ordinary screening uses optimize_scalar_combo,
// rescore_scalar_color and the prepare_* handles; the _replay_fixed_pose_* replays and the
// *_probe functions are diagnostics that run one kernel at fixed inputs so a
// Python driver can check or time it independently.
// =========================================================================
void bind_scalar_color(py::module_& module) {
    module.attr("__final_winner_only__") = kWinnerOnlyFinalSelection;
    // Fixed winner replay consumes the exact recorded pose bytes, never
    // re-normalizes the reporting quaternion or rebuilds another transform.
    // Returns [SHAPE_PREFACTOR * shape V_AB, color V_AB] at one recorded
    // 16-float pose (M, t', unit quaternion), with the prepared query's reduced
    // shape and color: the accurate reporting scorers, as used for a winner.
    module.def(
        "_replay_fixed_pose_cross",
        [](PreparedSharedPoseQuerySetHandle query, int query_row,
           py::array_t<float, py::array::c_style> library,
           py::array_t<uint8_t, py::array::c_style> counts,
           py::array_t<float, py::array::c_style> bytes) {
            if (!query || query_row < 0 ||
                size_t(query_row) >= query->reduced_query_colors.size() ||
                library.ndim() != 2 || library.shape(1) < 3 ||
                counts.ndim() != 1 || counts.shape(0) != query->n_features ||
                bytes.ndim() != 1 || bytes.size() != 16)
                throw std::invalid_argument(
                    "invalid fixed-pose replay input");
            for (int k = 0; k < 16; ++k)
                if (!std::isfinite(bytes.data()[k]))
                    throw std::invalid_argument("nonfinite fixed-pose replay pose");
            PreparedReducedPose pose;
            std::copy(bytes.data(), bytes.data() + 9, pose.matrix);
            std::copy(bytes.data() + 9, bytes.data() + 12, pose.translation);
            std::copy(bytes.data() + 12, bytes.data() + 16,
                      pose.unit_quaternion);
            const int n = int(counts.data()[0]), stride = int(library.shape(1));
            size_t occupied = 0;
            for (int feature = 0; feature < query->n_features; ++feature)
                occupied += size_t(counts.data()[feature]);
            if (occupied > size_t(library.shape(0)))
                throw std::invalid_argument(
                    "invalid replay library shape/color counts");
            std::vector<float> xyz(size_t(n) * 3);
            for (int atom = 0; atom < n; ++atom)
                for (int axis = 0; axis < 3; ++axis)
                    xyz[size_t(atom) * 3 + axis] =
                        library.data()[size_t(atom) * stride + axis];
            Conformer b{xyz.data(), n};
            const auto color = build_color_view_from_counts(
                library.data(), counts.data(), query->n_features, stride);
            const float sab = pair_overlap_at_pose(
                query->shape_evaluators->evaluators[size_t(query_row)]
                    ->query_reduced(),
                b, pose.matrix, pose.translation, nullptr);
            const double cab =
                score_exact_final_color(
                    query->reduced_query_colors[size_t(query_row)], color,
                    *query->color_plan, pose, 3, stride, nullptr, nullptr,
                    nullptr, false)
                    .value;
            py::array_t<float> out(2);
            out.mutable_data()[0] = SHAPE_PREFACTOR * sab;
            out.mutable_data()[1] = float(cab);
            return out;
        });
    // Independent diagnostic replay of the SEARCH evaluator at four saved
    // pose matrices, with the exact (unrounded) objective denominators.
    // Inputs: four 16-float poses, the four self values (shape AA, BB, color
    // AA, BB) and mixing; returns the four mixed objectives. exp_mode,
    // factor_torque, gradient and search_values_only select the evaluation as
    // the optimizer would.
    module.def(
        "_replay_fixed_pose_search",
        [](PreparedSharedPoseQuerySetHandle query, int query_row,
           py::array_t<float, py::array::c_style> library,
           py::array_t<uint8_t, py::array::c_style> counts,
           py::array_t<float, py::array::c_style> bytes,
           py::array_t<double, py::array::c_style> self, float mixing,
           const std::string& exp_mode, bool factor_torque, bool gradient,
           bool search_values_only) {
            if (!query || !query->shape_evaluators || query_row < 0 ||
                size_t(query_row) >= query->reduced_query_colors.size() ||
                library.ndim() != 2 || library.shape(1) < 3 ||
                counts.ndim() != 1 || counts.shape(0) != query->n_features ||
                bytes.ndim() != 2 || bytes.shape(0) != 4 ||
                bytes.shape(1) != 16 || self.ndim() != 1 || self.size() != 4 ||
                !std::isfinite(mixing))
                throw std::invalid_argument("invalid SEARCH replay input");
            size_t occupied = 0;
            for (int feature = 0; feature < query->n_features; ++feature)
                occupied += size_t(counts.data()[feature]);
            if (occupied > size_t(library.shape(0)))
                throw std::invalid_argument(
                    "invalid replay library shape/color counts");
            for (int k = 0; k < 64; ++k)
                if (!std::isfinite(bytes.data()[k]))
                    throw std::invalid_argument(
                        "nonfinite SEARCH replay pose");
            for (int k = 0; k < 4; ++k)
                if (!std::isfinite(self.data()[k]))
                    throw std::invalid_argument(
                        "nonfinite SEARCH replay self");
            const int n = int(counts.data()[0]), stride = int(library.shape(1));
            std::vector<float> xyz(size_t(n) * 3);
            for (int atom = 0; atom < n; ++atom)
                for (int axis = 0; axis < 3; ++axis)
                    xyz[size_t(atom) * 3 + axis] =
                        library.data()[size_t(atom) * stride + axis];
            Conformer shape{xyz.data(), n};
            const auto& config = query->shape_evaluators->config;
            const auto workspace = prepare_field_shape_library_workspace(
                shape, config.backend, config.width);
            const auto color = build_color_view_from_counts(
                library.data(), counts.data(), query->n_features, stride);
            const ReducedColorConformerPairInput pair{
                &query->reduced_query_colors[size_t(query_row)], &color, 3,
                stride};
            ColorCanonicalFourStartDescriptor descriptor;
            if (!prepare_color_canonical_four_start_descriptor(
                    pair, *query->color_plan, descriptor))
                throw std::invalid_argument(
                    "SEARCH replay needs canonical descriptor");
            ColorCanonicalFourStartWorkspace color_workspace;
            prepare_color_canonical_four_start_workspace(&descriptor, 1,
                                                         color_workspace);
            PreparedReducedPose poses[COLOR_CONFORMER_LANES]{};
            const PreparedReducedPose* ptrs[COLOR_CONFORMER_LANES]{};
            ColorEvaluation color_values[COLOR_CONFORMER_LANES]{};
            uint64_t terms[COLOR_CONFORMER_LANES]{};
            float shape_values[4]{};
            for (int start = 0; start < 4; ++start) {
                const float* src = bytes.data() + size_t(start) * 16;
                std::copy(src, src + 9, poses[start].matrix);
                std::copy(src + 9, src + 12, poses[start].translation);
                std::copy(src + 12, src + 16, poses[start].unit_quaternion);
                ptrs[start] = &poses[start];
                const ShapePoseRequest request{&shape,
                                               &workspace,
                                               poses[start].matrix,
                                               poses[start].translation,
                                               1.0f,
                                               &poses[start]};
                ShapeEvaluation value;
                evaluate_prepared_shape(
                    *query->shape_evaluators->evaluators[size_t(query_row)],
                    request, value);
                shape_values[start] = value.s_ab;
            }
            evaluate_color_canonical_four_start_batch(
                &descriptor, 1, *query->color_plan, ptrs, gradient,
                color_values, terms, nullptr, nullptr, color_workspace,
                parse_optimizer_exp_mode(exp_mode), factor_torque,
                search_values_only ? ColorSearchOutput::ValueOnly
                                   : ColorSearchOutput::ValueAndGradient);
            py::array_t<float> objectives(4);
            for (int start = 0; start < 4; ++start) {
                const float ts = float(tanimoto(
                    shape_values[start], self.data()[0], self.data()[1]));
                const float tc = float(tanimoto(
                    color_values[start].value, self.data()[2], self.data()[3]));
                objectives.mutable_data()[start] =
                    (1.0f - mixing) * ts + mixing * tc;
            }
            return objectives;
        },
        py::arg("query"), py::arg("query_row"), py::arg("library"),
        py::arg("counts"), py::arg("bytes"), py::arg("self"), py::arg("mixing"),
        py::arg("exp_mode"), py::arg("factor_torque"), py::arg("gradient"),
        py::arg("search_values_only") = false);
    // Field-hybrid compile flags (R3_COLOR_FIELD_*, R3_COLOR_START_*), exported for tests.
    module.attr("__color_field_hybrid__") = bool(R3_COLOR_START_HYBRID);
    module.attr("__color_start_reduced_coords__") = bool(R3_COLOR_START_REDUCED_COORDS);
    bind_combo_executor_api(module);
    bind_worker_local_topk_api(module);
    // The prepared self-overlap handle; read-only views for tests and
    // attribution.
    py::class_<PreparedComboSelfOverlaps, PreparedComboSelfOverlapsHandle>(
        module, "_PreparedComboSelfOverlaps")
        .def_property_readonly("shape_preparation_seconds",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.shape_preparation_seconds;
                               })
        .def_property_readonly("color_preparation_seconds",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.color_preparation_seconds;
                               })
        .def_property_readonly("shape_query_terms",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.shape_query_terms;
                               })
        .def_property_readonly("shape_library_terms",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.shape_library_terms;
                               })
        .def_property_readonly("color_query_terms",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.color_query_terms;
                               })
        .def_property_readonly("color_library_terms",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.color_library_terms;
                               })
        .def_property_readonly("library_shape_from_stored_volume",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.library_shape_from_stored_volume;
                               })
        .def_property_readonly("library_color_from_sidecar",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.library_color_from_sidecar;
                               })
        .def_property_readonly("shape_prepared",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.shape_prepared;
                               })
        .def_property_readonly("color_prepared",
                               [](const PreparedComboSelfOverlaps& self) {
                                   return self.color_prepared;
                               })
        .def_property_readonly(
            "shape_library_values",
            [](const PreparedComboSelfOverlaps& self) {
                py::array_t<float> out(self.shape_library.size());
                std::copy(self.shape_library.begin(), self.shape_library.end(),
                          out.mutable_data());
                return out;
            })
        .def_property_readonly(
            "shape_query_values",
            [](const PreparedComboSelfOverlaps& self) {
                py::array_t<float> out(self.shape_query.size());
                std::copy(self.shape_query.begin(), self.shape_query.end(),
                          out.mutable_data());
                return out;
            })
        .def_property_readonly(
            "color_library_values", [](const PreparedComboSelfOverlaps& self) {
                py::array_t<double> out(self.color_library.size());
                std::copy(self.color_library.begin(), self.color_library.end(),
                          out.mutable_data());
                return out;
            });
    module.def("prepare_combo_self_overlaps", &prepare_combo_self_overlaps,
               py::arg("query_coords"), py::arg("query_types"),
               py::arg("query_n_atoms"), py::arg("library_coords"),
               py::arg("library_counts"), py::arg("interaction_r"),
               py::arg("interaction_p"), py::arg("input_digest"),
               py::arg("library_shape_self_reduced") = py::none(),
               py::arg("library_shape_volume_a3") = py::none(),
               py::arg("library_color_self") = py::none(),
               py::arg("prepare_shape") = true,
               py::arg("prepare_color") = true);
    module.def("_prepare_r3g_library_self_overlaps", &prepare_r3g_library_self_overlaps,
               py::arg("library_coords"), py::arg("library_counts"),
               py::arg("interaction_r"), py::arg("interaction_p"),
               py::arg("input_digest"), py::arg("vaa"),
               py::arg("color_self") = py::none(),
               py::arg("prepare_shape") = true,
               py::arg("prepare_color") = true,
               py::arg("shape_float32_reciprocal") = false);
    module.def("_combine_r3g_self_overlaps", &combine_r3g_self_overlaps,
               py::arg("query"), py::arg("library"), py::arg("input_digest"),
               py::call_guard<py::gil_scoped_release>());
    // Predict, without running anything, the work decomposition
    // optimize_scalar_combo would use (units, claims, CPU attempts).
    // workers_used is the legacy pool size, not the number of workers
    // that received claims; that is details["scoring_claims_by_worker"].
    module.def(
        "combo_thread_config_stamp",
        [](int threads, int color_grain, int start_mode, py::ssize_t n_pairs,
           bool require_worker_affinity, int claim_chunk_rows) {
            if (threads < 1)
                throw std::invalid_argument("threads must be >= 1");
            require_supported_color_grain(color_grain);
            if (n_pairs < 0)
                throw std::invalid_argument("n_pairs must be >= 0");
            if (claim_chunk_rows < 128 || claim_chunk_rows > 512 ||
                claim_chunk_rows % 16 != 0)
                throw std::invalid_argument(
                    "claim_chunk_rows must be a multiple of 16 in [128, 512]");
            if (require_worker_affinity && threads < 2)
                throw std::invalid_argument(
                    "required worker affinity applies only to parallel combo calls");
            const size_t units =
                combo_work_units(color_grain, start_mode, size_t(n_pairs));
            const size_t rows_per_work_unit =
                combo_rows_per_work_unit(color_grain, start_mode);
            const size_t work_units_per_claim = std::max<size_t>(
                1, size_t(claim_chunk_rows) / rows_per_work_unit);
            const size_t claim_count =
                units / work_units_per_claim +
                (units % work_units_per_claim != 0 ? 1 : 0);
            const size_t worker_parallelism_limit =
                std::min(size_t(threads), claim_count);
            const int workers =
                threads == 1 || units == 0
                    ? (units == 0 ? 0 : 1)
                    : std::min<int>(
                          threads,
                          int(std::min<size_t>(
                              units, size_t(std::numeric_limits<int>::max()))));
            const std::vector<int> cpus = combo_allowed_cpus();
            if (require_worker_affinity && workers > 0 && cpus.empty())
                throw std::invalid_argument(
                    "required combo worker affinity has no allowed CPU mask");
            if (require_worker_affinity && size_t(workers) > cpus.size())
                throw std::invalid_argument(
                    "required combo workers exceed the process allowed CPU mask");
            py::dict stamp;
            stamp["threads_requested"] = threads;
            stamp["workers_used"] = workers;
            stamp["workers_used_semantics"] =
                "legacy one-shot pool capacity; not observed workers with claims";
            stamp["work_units"] = units;
            stamp["work_unit"] = combo_work_unit_name(color_grain, start_mode);
            stamp["schedule"] = "dynamic_coarse_claims";
            stamp["claim_chunk_rows"] = claim_chunk_rows;
            stamp["work_units_per_claim"] = work_units_per_claim;
            stamp["claim_count"] = claim_count;
            stamp["worker_parallelism_limit"] = worker_parallelism_limit;
            stamp["process_allowed_cpus"] = cpus;
            std::vector<int> worker_cpus;
            if (threads > 1 && workers > 0 && !cpus.empty()) {
                worker_cpus.reserve(size_t(workers));
                for (int worker = 0; worker < workers; ++worker)
                    worker_cpus.push_back(cpus[size_t(worker) % cpus.size()]);
            }
            stamp["worker_cpu_affinity_attempts"] = worker_cpus;
            stamp["worker_affinity_required"] = require_worker_affinity;
            stamp["affinity_policy"] =
                "one logical CPU per native worker from the process allowed mask; "
                "setaffinity is advisory; topology is supplied by the caller";
            return stamp;
        },
        py::arg("threads"), py::arg("color_grain"), py::arg("start_mode"),
        py::arg("n_pairs"), py::arg("require_worker_affinity") = false,
        py::arg("claim_chunk_rows") = 256);
    py::class_<PreparedColorFieldHandle,
               std::shared_ptr<PreparedColorFieldHandle>>(
        module, "PreparedColorFieldHandle",
        "Query-scoped color field storage. Build once with prepare_color_fields and pass it to repeated ordinary mode-3 score_pairs calls.")
        .def_property_readonly(
            "backend_codes",
            [](const PreparedColorFieldHandle& x) -> py::object {
                if (!x.backend_codes.empty()) return py::cast(x.backend_codes);
                const size_t nq = x.bundles.size();
                const size_t nf = size_t(x.n_features);
                std::vector<std::vector<int>> matrix(nq,
                                                     std::vector<int>(nf, 0));
                for (size_t row = 0; row < nq; ++row)
                    for (size_t tb = 0; tb < nf; ++tb)
                        matrix[row][tb] =
                            x.backend_codes_by_query[row * nf + tb];
                return py::cast(matrix);
            })
        .def_property_readonly(
            "backend_codes_by_query",
            [](const PreparedColorFieldHandle& x) {
                const size_t nq = x.bundles.size();
                const size_t nf = size_t(x.n_features);
                std::vector<std::vector<int>> matrix(nq,
                                                     std::vector<int>(nf, 0));
                for (size_t row = 0; row < nq; ++row)
                    for (size_t tb = 0; tb < nf; ++tb)
                        matrix[row][tb] =
                            x.backend_codes_by_query[row * nf + tb];
                return matrix;
            })
        .def_property_readonly(
            "field_bytes",
            [](const PreparedColorFieldHandle& x) { return x.bytes; })
        .def_property_readonly(
            "retained_bytes",
            [](const PreparedColorFieldHandle& x) { return x.retained_bytes; })
        .def_property_readonly("bundle_generations",
                               [](const PreparedColorFieldHandle& x) {
                                   std::vector<uint64_t> out;
                                   for (const auto& b : x.bundles)
                                       out.push_back(b->generation());
                                   return out;
                               })
        .def_property_readonly("bundle_pack_counts",
                               [](const PreparedColorFieldHandle& x) {
                                   std::vector<std::vector<int>> out;
                                   for (const auto& b : x.bundles)
                                       out.push_back(b->pack_counts());
                                   return out;
                               })
        .def_property_readonly("bundle_peak_payload_bytes",
                               [](const PreparedColorFieldHandle& x) {
                                   std::vector<size_t> out;
                                   for (const auto& b : x.bundles)
                                       out.push_back(b->peak_payload_bytes());
                                   return out;
                               })
        .def_property_readonly("field_bytes_by_target_type",
                               [](const PreparedColorFieldHandle& x) {
                                   return x.bytes_by_target_type;
                               })
        .def_property_readonly(
            "build_seconds",
            [](const PreparedColorFieldHandle& x) { return x.build_seconds; })
        .def_property_readonly("build_seconds_by_target_type",
                               [](const PreparedColorFieldHandle& x) {
                                   return x.build_seconds_by_target_type;
                               })
        .def_property_readonly(
            "pack_seconds",
            [](const PreparedColorFieldHandle& x) { return x.pack_seconds; })
        .def_property_readonly("pack_seconds_by_target_type",
                               [](const PreparedColorFieldHandle& x) {
                                   return x.pack_seconds_by_target_type;
                               })
        .def_property_readonly("field_bytes_by_backend",
                               [](const PreparedColorFieldHandle& x) {
                                   return x.bytes_by_backend;
                               })
        .def_property_readonly("build_seconds_by_backend",
                               [](const PreparedColorFieldHandle& x) {
                                   return x.build_seconds_by_backend;
                               })
        .def_property_readonly("pack_seconds_by_backend",
                               [](const PreparedColorFieldHandle& x) {
                                   return x.pack_seconds_by_backend;
                               })
        .def_property_readonly("half_decode_path",
                               [](const PreparedColorFieldHandle& x) {
                                   return x.half_decode_path;
                               })
        .def_property_readonly(
            "n_queries",
            [](const PreparedColorFieldHandle& x) { return x.bundles.size(); })
        .def_property_readonly("field_active_by_target_type",
                               [](const PreparedColorFieldHandle& x) {
                                   std::vector<uint64_t> active(
                                       size_t(x.n_features), 0);
                                   for (const auto& bundle : x.bundles)
                                       for (int tb = 1; tb < x.n_features; ++tb)
                                           if (bundle->field_active(tb))
                                               ++active[size_t(tb)];
                                   return active;
                               });
    module.def("prepare_color_fields", &prepare_color_field_handle,
               py::arg("query_coords"), py::arg("query_types"),
               py::arg("query_n_atoms"), py::arg("interaction_r"),
               py::arg("interaction_p"), py::arg("backend_codes"),
               py::arg("spacing_angstrom") = 0.5,
               py::arg("padding_angstrom") = 4.0);
    module.def(
        "color_arm_stamp", &color_arm_stamp,
        "Report the analytic color arm's ISA, lane width, reduction policy, "
        "compile guard and runtime feature availability.");
    // One exact scalar (grain 0) color evaluation at a given legacy pose.
    module.def("scalar_color_simd_probe", [](py::array_t<float,
                                                         py::array::c_style>
                                                 qc,
                                             py::array_t<int32_t,
                                                         py::array::c_style>
                                                 qt,
                                             int qn,
                                             py::array_t<float,
                                                         py::array::c_style>
                                                 lc,
                                             py::array_t<uint8_t,
                                                         py::array::c_style>
                                                 lcnt,
                                             py::array_t<float,
                                                         py::array::c_style>
                                                 r,
                                             py::array_t<float,
                                                         py::array::c_style>
                                                 p,
                                             py::array_t<float,
                                                         py::array::c_style>
                                                 q,
                                             py::array_t<float,
                                                         py::array::c_style>
                                                 t,
                                             int grain, bool gradient) {
        if (grain != 0)
            throw std::invalid_argument(
                "scalar_color_simd_probe supports color grain 0 only");
        if (qc.ndim() != 2 || qt.ndim() != 1 || qt.shape(0) != qc.shape(0))
            throw std::invalid_argument(
                "query types must match query coordinate width");
        Matrix mat(r, p);
        ColorConformer a = build_color_view_from_types(qc.data(), qt.data(), qn,
                                                       int(qc.shape(0)), mat.n,
                                                       int(qc.shape(1)));
        ColorConformer b = build_color_view_from_counts(
            lc.data(), lcnt.data(), mat.n, int(lc.shape(1)));
        // Grain 0 always returns the value and the descent direction; the
        // gradient flag stays in the signature for existing callers.
        (void)gradient;
        uint64_t terms = 0;
        ColorEvaluation v = color_value_gradient(a, b, mat, int(qc.shape(1)),
                                                 int(lc.shape(1)), q.data(),
                                                 t.data(), &terms);
        py::array_t<double> g(7);
        for (int k = 0; k < 7; ++k)
            g.mutable_data()[k] = v.descent[k];
        return py::make_tuple(v.value, g, terms);
    });
    // Times the grain-11 conformer batch kernel on preposed data, optionally
    // with query fields and coexisting shape gradients;
    // native_kernel_scope in the result states what the timer covers.
    module.def(
        "color_conformer_native_batch_probe",
        [](py::array_t<float, py::array::c_style> qc,
           py::array_t<int32_t, py::array::c_style> qt, int qn,
           py::array_t<float, py::array::c_style> lc,
           py::array_t<uint8_t, py::array::c_style> lcnt,
           py::array_t<float, py::array::c_style> r,
           py::array_t<float, py::array::c_style> p,
           py::array_t<float, py::array::c_style> q,
           py::array_t<float, py::array::c_style> t, py::object backend_codes,
           int repeats, bool include_shape) {
            if (repeats < 1 || repeats > 10000000)
                throw std::invalid_argument(
                    "native conformer batch repeats must be in [1, 10000000]");
            Matrix mat(r, p);
            if (qc.ndim() != 2 || qc.shape(1) != 3 || qt.ndim() != 1 ||
                qt.shape(0) != qc.shape(0) || qn < 0 || qn > qc.shape(0) ||
                lc.ndim() != 3 || lc.shape(0) < 1 ||
                lc.shape(0) > COLOR_CONFORMER_LANES || lc.shape(2) != 3 ||
                lcnt.ndim() != 2 || lcnt.shape(0) != lc.shape(0) ||
                lcnt.shape(1) != mat.n || q.ndim() != 2 ||
                q.shape(0) != lc.shape(0) || q.shape(1) != 4 || t.ndim() != 2 ||
                t.shape(0) != lc.shape(0) || t.shape(1) != 3)
                throw std::invalid_argument(
                    "native conformer batch needs matching query, library, count, matrix and pose arrays");

            ColorConformer query = build_color_view_from_types(
                qc.data(), qt.data(), qn, int(qc.shape(0)), mat.n, 3);
            const int n_pairs = int(lc.shape(0));
            const int library_stride = int(lc.shape(1));
            ColorConformerPairInput pair_inputs[COLOR_CONFORMER_LANES]{};
            std::vector<ColorConformer> libraries(static_cast<size_t>(n_pairs),
                                                  ColorConformer{});
            for (int lane = 0; lane < n_pairs; ++lane) {
                libraries[size_t(lane)] = build_color_view_from_counts(
                    lc.data() + size_t(lane) * size_t(library_stride) * 3,
                    lcnt.data() + size_t(lane) * size_t(mat.n), mat.n, 3);
                pair_inputs[lane] = {&query, &libraries[size_t(lane)], 3, 3};
            }
            ColorConformerBatch8 batch =
                pack_color_conformer_batch8(pair_inputs, n_pairs, mat.n);

            const py::array codes_array = py::array::ensure(backend_codes);
            if (!codes_array || codes_array.ndim() != 1 ||
                codes_array.shape(0) != mat.n)
                throw std::invalid_argument(
                    "native conformer batch backend codes must match the interaction feature count");
            const py::module_ numpy = py::module_::import("numpy");
            if (!py::cast<bool>(numpy.attr("issubdtype")(
                    codes_array.dtype(), numpy.attr("integer"))))
                throw std::invalid_argument(
                    "native conformer batch backend codes must be integers");
            std::vector<int> codes(size_t(mat.n), 0);
            bool use_fields = false;
            for (int tb = 0; tb < mat.n; ++tb) {
                const py::object item = codes_array[py::int_(tb)];
                const int code = py::cast<int>(item);
                if (code < COLOR_FIELD_ANALYTIC ||
                    code > COLOR_FIELD_BRICK_FP32 ||
                    (tb == 0 && code != COLOR_FIELD_ANALYTIC))
                    throw std::invalid_argument(
                        "native conformer batch backend codes must be 0, 1, 2 or 3, with type zero analytic");
                codes[size_t(tb)] = code;
                use_fields |= code != COLOR_FIELD_ANALYTIC;
            }

            std::unique_ptr<ColorFieldBundle> fields;
            if (use_fields) {
                std::vector<ColorFieldSource> sources;
                sources.reserve(size_t(query.n_color));
                for (int ta = 1; ta < mat.n; ++ta)
                    for (int i = 0; i < query.seg_count[ta]; ++i) {
                        const float* xyz = query.color_xyz +
                                           size_t(query.seg_offset[ta] + i) * 3;
                        ColorFieldSource source;
                        source.xyz[0] = xyz[0];
                        source.xyz[1] = xyz[1];
                        source.xyz[2] = xyz[2];
                        source.type = ta;
                        sources.push_back(source);
                    }
                std::vector<double> coefficients(size_t(mat.n) * size_t(mat.n));
                std::vector<double> decays(size_t(mat.n) * size_t(mat.n));
                for (int ta = 0; ta < mat.n; ++ta)
                    for (int tb = 0; tb < mat.n; ++tb) {
                        const size_t at =
                            size_t(ta) * size_t(mat.n) + size_t(tb);
                        coefficients[at] = mat.coeff[ta][tb];
                        decays[at] = mat.decay[ta][tb];
                    }
                fields = std::make_unique<ColorFieldBundle>(
                    sources, mat.n, coefficients.data(), decays.data(),
                    codes.data(), 0.5, 4.0);
            }
            const ColorFieldBundle* field_contexts[COLOR_CONFORMER_LANES]{};
            if (fields)
                for (int lane = 0; lane < n_pairs; ++lane)
                    field_contexts[lane] = fields.get();

            float quaternions[COLOR_CONFORMER_LANES][4]{};
            float translations[COLOR_CONFORMER_LANES][3]{};
            for (int lane = 0; lane < n_pairs; ++lane) {
                for (int k = 0; k < 4; ++k)
                    quaternions[lane][k] =
                        q.data()[size_t(lane) * 4 + size_t(k)];
                for (int k = 0; k < 3; ++k)
                    translations[lane][k] =
                        t.data()[size_t(lane) * 3 + size_t(k)];
            }
            std::vector<float> query_shape_xyz(size_t(qn) * 3);
            for (int atom = 0; atom < qn; ++atom)
                for (int k = 0; k < 3; ++k)
                    query_shape_xyz[size_t(atom) * 3 + size_t(k)] =
                        COORD_SCALE * qc.data()[size_t(atom) * 3 + size_t(k)];
            const Conformer shape_query{query_shape_xyz.data(), qn};
            std::vector<Conformer> shape_libraries(static_cast<size_t>(n_pairs),
                                                   Conformer{});
            float shape_matrices[COLOR_CONFORMER_LANES][9]{};
            float shape_translations[COLOR_CONFORMER_LANES][3]{};
            for (int lane = 0; lane < n_pairs; ++lane) {
                const int n_shape =
                    int(lcnt.data()[size_t(lane) * size_t(mat.n)]);
                shape_libraries[size_t(lane)] = {
                    lc.data() + size_t(lane) * size_t(library_stride) * 3,
                    n_shape};
                quat_to_mat(quaternions[lane], shape_matrices[lane]);
                for (int k = 0; k < 9; ++k)
                    shape_matrices[lane][k] *= COORD_SCALE;
                for (int k = 0; k < 3; ++k)
                    shape_translations[lane][k] =
                        translations[lane][k] * COORD_SCALE;
            }
            ColorConformerWorkspace workspace;
            prepare_color_conformer_batch8_preposed(batch, quaternions,
                                                    translations, &workspace);

            // One untimed census pass exposes selected-field and fast-analytic
            // fallback work. It is not included in the native timing below.
            ColorEvaluation last[COLOR_CONFORMER_LANES]{};
            uint64_t terms[COLOR_CONFORMER_LANES]{};
            float shape_values[COLOR_CONFORMER_LANES]{};
            float shape_gradients[COLOR_CONFORMER_LANES][7]{};
            ColorFieldBatchStats field_stats;
            evaluate_color_conformer_batch8(
                batch, mat.view(), quaternions, translations, true, last, terms,
                nullptr, nullptr, &workspace, fields ? field_contexts : nullptr,
                fields ? &field_stats : nullptr, true);

            volatile double sink = 0.0;
            const auto kernel_started = Clock::now();
            for (int iteration = 0; iteration < repeats; ++iteration) {
                std::atomic_signal_fence(std::memory_order_seq_cst);
                if (include_shape)
                    for (int lane = 0; lane < n_pairs; ++lane)
                        get_gradient(shape_query, shape_libraries[size_t(lane)],
                                     shape_matrices[lane],
                                     shape_translations[lane], 1.0f,
                                     shape_gradients[lane], &shape_values[lane],
                                     nullptr);
                evaluate_color_conformer_batch8(
                    batch, mat.view(), quaternions, translations, true, last,
                    terms, nullptr, nullptr, &workspace,
                    fields ? field_contexts : nullptr, nullptr, true);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                double reduction = 0.0;
                for (int lane = 0; lane < n_pairs; ++lane) {
                    reduction += last[lane].value;
                    for (int k = 0; k < 7; ++k)
                        reduction += last[lane].descent[k] * double(k + 1);
                    if (include_shape) {
                        reduction += double(shape_values[lane]);
                        for (int k = 0; k < 7; ++k)
                            reduction += double(shape_gradients[lane][k]) *
                                         double(k + 1);
                    }
                }
                sink = reduction;
            }
            const double kernel_seconds =
                std::chrono::duration<double>(Clock::now() - kernel_started)
                    .count();

            py::array_t<double> values(n_pairs), gradients({n_pairs, 7});
            py::array_t<float> shape_value_array(n_pairs),
                shape_gradient_array({n_pairs, 7});
            py::array_t<uint64_t> term_array(n_pairs);
            for (int lane = 0; lane < n_pairs; ++lane) {
                values.mutable_data()[lane] = last[lane].value;
                term_array.mutable_data()[lane] = terms[lane];
                for (int k = 0; k < 7; ++k)
                    gradients.mutable_data()[size_t(lane) * 7 + size_t(k)] =
                        last[lane].descent[k];
                shape_value_array.mutable_data()[lane] = shape_values[lane];
                for (int k = 0; k < 7; ++k)
                    shape_gradient_array
                        .mutable_data()[size_t(lane) * 7 + size_t(k)] =
                        shape_gradients[lane][k];
            }
            py::dict result;
            result["kernel_seconds"] = kernel_seconds;
            result["kernel_seconds_per_evaluation"] = kernel_seconds / repeats;
            result["kernel_repeats"] = repeats;
            result["values"] = values;
            result["gradients"] = gradients;
            result["shape_coexistence"] = include_shape;
            result["shape_values"] = shape_value_array;
            result["shape_gradients"] = shape_gradient_array;
            result["analytic_term_counts"] = term_array;
            result["observable_sink"] = double(sink);
            result["selected_samples"] = field_stats.interpolated_samples;
            result["interpolated_terms"] = field_stats.interpolated_terms;
            result["analytic_selected_samples"] =
                field_stats.analytic_selected_samples;
            result["analytic_selected_terms"] =
                field_stats.analytic_selected_terms;
            result["analytic_fallback_samples"] =
                field_stats.analytic_fallback_samples;
            result["inactive_fallback_samples"] =
                field_stats.inactive_fallback_samples;
            result["out_of_grid_fallback_samples"] =
                field_stats.out_of_grid_fallback_samples;
            result["analytic_fallback_terms"] =
                field_stats.analytic_fallback_terms;
            result["field_bytes"] = fields ? fields->field_bytes() : 0;
            result["field_build_seconds"] =
                fields ? fields->build_seconds() : 0.0;
            result["field_bytes_by_target_type"] =
                fields ? fields->field_bytes_by_target_type()
                       : std::vector<size_t>();
            result["field_build_seconds_by_target_type"] =
                fields ? fields->build_seconds_by_target_type()
                       : std::vector<double>();
            result["field_pack_seconds_by_target_type"] =
                fields ? fields->pack_seconds_by_target_type()
                       : std::vector<double>();
            result["field_backend_codes"] =
                fields ? fields->backend_codes() : std::vector<int>();
            result["half_decode_path"] =
                fields ? std::string(fields->half_decode_path())
                       : std::string("not_used");
            result["native_kernel_scope"] =
                "preposed double/fp32 target SoA; includes native support/field-eligibility guards and analytic/field evaluation plus observable reduction; excludes query-scoped field build, pose transform and fp32 pose packing; shape gradient is included only when shape_coexistence=true";
            return result;
        },
        py::arg("query_xyz"), py::arg("query_types"), py::arg("query_n_atoms"),
        py::arg("library_xyz"), py::arg("library_counts"),
        py::arg("interaction_r"), py::arg("interaction_p"),
        py::arg("quaternions"), py::arg("translations_physical"),
        py::arg("backend_codes"), py::arg("repeats"),
        py::arg("include_shape") = false);
    // Legacy mixed descent direction at one pose, for the finite-difference
    // checks of mixed_descent.
    module.def(
        "scalar_combo_direction",
        [](py::array_t<float, py::array::c_style> qc,
           py::array_t<int32_t, py::array::c_style> qt, int qn,
           py::array_t<float, py::array::c_style> lc,
           py::array_t<uint8_t, py::array::c_style> lcnt,
           py::array_t<float, py::array::c_style> r,
           py::array_t<float, py::array::c_style> p,
           py::array_t<float, py::array::c_style> q,
           py::array_t<float, py::array::c_style> t, float mixing) {
            if (qc.ndim() != 2 || qt.ndim() != 1 || qt.shape(0) != qc.shape(0))
                throw std::invalid_argument(
                    "query types must match query coordinate width");
            Matrix mat(r, p);
            const int qs = int(qc.shape(1)), ls = int(lc.shape(1));
            ColorConformer ac = build_color_view_from_types(
                qc.data(), qt.data(), qn, int(qc.shape(0)), mat.n, qs);
            ColorConformer bc =
                build_color_view_from_counts(lc.data(), lcnt.data(), mat.n, ls);
            std::vector<float> as(size_t(qn) * 3), bs(size_t(bc.n_real) * 3);
            for (int i = 0; i < qn; ++i)
                for (int k = 0; k < 3; ++k)
                    as[size_t(i) * 3 + k] =
                        COORD_SCALE * qc.data()[size_t(i) * qs + k];
            for (int i = 0; i < bc.n_real; ++i)
                for (int k = 0; k < 3; ++k)
                    bs[size_t(i) * 3 + k] = lc.data()[size_t(i) * ls + k];
            Conformer a{as.data(), qn}, b{bs.data(), bc.n_real};
            const float saa = self_overlap(a, nullptr);
            std::vector<float> bself = bs;
            for (float& x : bself)
                x *= COORD_SCALE;
            Conformer bscaled{bself.data(), bc.n_real};
            const float sbb = self_overlap(bscaled, nullptr);
            float m[9],
                tr[3] = {t.data()[0] * COORD_SCALE, t.data()[1] * COORD_SCALE,
                         t.data()[2] * COORD_SCALE};
            quat_to_mat(q.data(), m);
            for (int k = 0; k < 9; ++k)
                m[k] *= COORD_SCALE;
            float gs[7], sab;
            get_gradient(a, b, m, tr, 1.0f, gs, &sab, nullptr);
            const double caa = overlap(ac, ac, mat, qs, qs, nullptr, nullptr);
            const double cbb = overlap(bc, bc, mat, ls, ls, nullptr, nullptr);
            const ColorEvaluation color = color_value_gradient(
                ac, bc, mat, qs, ls, q.data(), t.data(), nullptr);
            float g[7];
            mixed_descent(gs, sab, saa, sbb, color, caa, cbb, mixing, g);
            py::array_t<float> out(7);
            for (int k = 0; k < 7; ++k)
                out.mutable_data()[k] = g[k];
            return out;
        });
    // Exact FP64 color value and descent at one legacy pose.
    module.def(
        "scalar_color_value_gradient",
        [](py::array_t<float, py::array::c_style> qc,
           py::array_t<int32_t, py::array::c_style> qt, int qn,
           py::array_t<float, py::array::c_style> lc,
           py::array_t<uint8_t, py::array::c_style> lcnt,
           py::array_t<float, py::array::c_style> r,
           py::array_t<float, py::array::c_style> p,
           py::array_t<float, py::array::c_style> q,
           py::array_t<float, py::array::c_style> t) {
            if (qc.ndim() != 2 || qt.ndim() != 1 || qt.shape(0) != qc.shape(0))
                throw std::invalid_argument(
                    "query types must match query coordinate width");
            Matrix mat(r, p);
            const int qs = int(qc.shape(1)), ls = int(lc.shape(1));
            ColorConformer a = build_color_view_from_types(
                qc.data(), qt.data(), qn, int(qc.shape(0)), mat.n, qs);
            ColorConformer b =
                build_color_view_from_counts(lc.data(), lcnt.data(), mat.n, ls);
            ColorEvaluation v = color_value_gradient(
                a, b, mat, qs, ls, q.data(), t.data(), nullptr);
            py::array_t<double> g(7);
            for (int k = 0; k < 7; ++k)
                g.mutable_data()[k] = v.descent[k];
            return py::make_tuple(v.value, g);
        });
    // Native defaults: shared FP32 pose, final_start_policy "default",
    // baseline optimizer exp, scalar color grain 0. The public Python
    // facade passes its own choices.
    module.def("optimize_scalar_combo", &optimize_scalar_combo,
               py::arg("query_coords"), py::arg("query_types"),
               py::arg("query_n_atoms"), py::arg("library_coords"),
               py::arg("library_counts"), py::arg("pair_query_row"),
               py::arg("pair_library_row"), py::arg("interaction_r"),
               py::arg("interaction_p"), py::arg("steps"), py::arg("lr_q"),
               py::arg("lr_t"), py::arg("start_mode"), py::arg("mixing"),
               py::arg("with_census"), py::arg("with_timers"),
               py::arg("color_grain") = 0,
               py::arg("color_field_backends") = py::none(),
               py::arg("color_field_spacing") = 0.5,
               py::arg("color_field_padding") = 4.0,
               py::arg("prepared_color_fields") = py::none(),
               py::arg("shape_backend") = 0, py::arg("shape_grain") = 0,
               py::arg("shape_width") = 1,
               py::arg("shape_field_spacing") = 0.0f,
               py::arg("shape_field_padding") = 8.0f,
               py::arg("prepared_shape_evaluators") = py::none(),
               py::arg("threads") = 1, py::arg("with_worker_timers") = false,
               py::arg("require_worker_affinity") = false,
               py::arg("prepared_shared_pose_queries") = py::none(),
               py::arg("pose_contract") = "shared_fp32_reduced_v1",
               py::arg("use_color_invariant_plan") = false,
               py::arg("reuse_exact_final_scratch") = false,
               py::arg("prepared_combo_self_overlaps") = py::none(),
               py::arg("prepared_combo_self_overlap_digest") = "",
               py::arg("executor") = py::none(),
               py::arg("row_sink") = py::none(), py::arg("return_dense") = true,
               py::arg("with_library_self_timers") = false,
               py::arg("factor_canonical_torque") = false,
               py::arg("optimizer_exp_mode") = "baseline",
               py::arg("canonical_color_kernel") = false,
               py::arg("final_start_policy") = "default");
    // Physical-frame mode-2 self terms, deliberately NOT combo's reduced
    // self values. Empty query/library rows allow their independent lifetimes.
    module.def(
        "prepare_rescore_color_self",
        [](py::array_t<float, py::array::c_style> qc,
           py::array_t<int32_t, py::array::c_style> qt,
           py::array_t<int32_t, py::array::c_style> qn,
           py::array_t<float, py::array::c_style> lc,
           py::array_t<uint8_t, py::array::c_style> counts,
           py::array_t<float, py::array::c_style> r,
           py::array_t<float, py::array::c_style> p) {
            Matrix mat(r, p);
            if (qc.ndim() != 3 || lc.ndim() != 3 || qc.shape(2) != 3 ||
                lc.shape(2) != 3 || qt.ndim() != 2 ||
                qt.shape(0) != qc.shape(0) || qt.shape(1) != qc.shape(1) ||
                qn.ndim() != 1 || qn.size() != qc.shape(0) ||
                counts.ndim() != 2 || counts.shape(0) != lc.shape(0) ||
                counts.shape(1) != mat.n)
                throw std::invalid_argument(
                    "rescore self input dimensions disagree");
            for (py::ssize_t row = 0; row < qc.shape(0); ++row)
                if (qn.data()[row] < 0 || qn.data()[row] > qc.shape(1))
                    throw std::invalid_argument(
                        "query atom count outside rescore coordinate row");
            for (py::ssize_t row = 0; row < lc.shape(0); ++row) {
                int total = 0;
                for (int type = 0; type < mat.n; ++type)
                    total += counts.data()[row * mat.n + type];
                if (total > lc.shape(1))
                    throw std::invalid_argument(
                        "library counts exceed rescore coordinate row");
            }
            for (py::ssize_t i = 0; i < qc.size(); ++i)
                if (!std::isfinite(qc.data()[i]))
                    throw std::invalid_argument(
                        "rescore query coordinates must be finite");
            for (py::ssize_t i = 0; i < lc.size(); ++i)
                if (!std::isfinite(lc.data()[i]))
                    throw std::invalid_argument(
                        "rescore library coordinates must be finite");
            py::array_t<double> qself(qc.shape(0)), lself(lc.shape(0));
            double* qa = qself.mutable_data();
            double* lb = lself.mutable_data();
            {
                py::gil_scoped_release release;
                for (py::ssize_t row = 0; row < qc.shape(0); ++row) {
                    const ColorConformer view = build_color_view_from_types(
                        qc.data() + row * qc.shape(1) * 3,
                        qt.data() + row * qt.shape(1), qn.data()[row],
                        int(qc.shape(1)), mat.n, 3);
                    qa[row] = overlap(view, view, mat, 3, 3, nullptr, nullptr);
                }
                for (py::ssize_t row = 0; row < lc.shape(0); ++row) {
                    const ColorConformer view = build_color_view_from_counts(
                        lc.data() + row * lc.shape(1) * 3,
                        counts.data() + row * mat.n, mat.n, 3);
                    lb[row] = overlap(view, view, mat, 3, 3, nullptr, nullptr);
                }
            }
            return py::make_tuple(qself, lself);
        });
    // Mode 2: shape has already chosen each row's pose; evaluate
    // exact color once at it and fill the color columns. shape_scores rows are
    // the 20-column shape results, whose legacy physical pose sits in columns
    // 9-12 (quaternion) and 13-15 (translation). Self values come prepared
    // recomputed per row (recompute_self) or once per query and library
    // row. Shape columns are copied unchanged; native work runs without the
    // GIL.
    module.def(
        "rescore_scalar_color",
        [](py::array_t<float, py::array::c_style> qc,
           py::array_t<int32_t, py::array::c_style> qt,
           py::array_t<int32_t, py::array::c_style> qn,
           py::array_t<float, py::array::c_style> lc,
           py::array_t<uint8_t, py::array::c_style> lcnt,
           py::array_t<int32_t, py::array::c_style> pq,
           py::array_t<int32_t, py::array::c_style> pl,
           py::array_t<float, py::array::c_style> r,
           py::array_t<float, py::array::c_style> p,
           py::array_t<float, py::array::c_style> shape, float mixing,
           bool with_census, bool with_timers, bool recompute_self,
           int color_grain, py::object prepared_query_self,
           py::object prepared_library_self) {
            if (qc.ndim() != 3 || qt.ndim() != 2 ||
                qt.shape(0) != qc.shape(0) || qt.shape(1) != qc.shape(1))
                throw std::invalid_argument(
                    "query types must match query coordinate rows and width");
            require_supported_color_grain(color_grain);
            // The start grain has no fixed-pose kernel; rescore it as scalar
            // grain 0.
            const int rescore_grain =
                is_color_start_grain(color_grain) ? 0 : color_grain;
            const auto clock = std::chrono::steady_clock::now;
            auto t0 = clock();
            const int nf = int(r.shape(0));
            const int qs = int(qc.shape(2)), ls = int(lc.shape(2));
            Matrix mat(r, p);
            std::vector<ColorConformer> qv(size_t(qc.shape(0)));
            std::vector<ColorConformer> lv(size_t(lc.shape(0)));
            const float *qbase = qc.data(), *lbase = lc.data();
            const int32_t* tbase = qt.data();
            const int32_t* nbase = qn.data();
            const uint8_t* cbase = lcnt.data();
            const size_t qcapacity = size_t(qc.shape(1));
            const size_t lcapacity = size_t(lc.shape(1));
            const size_t type_width = size_t(qt.shape(1));
            const size_t n_pairs = size_t(pq.size());
            std::vector<double> qself(qv.size()), lself(lv.size());
            auto copy_self = [](const py::object& supplied,
                                std::vector<double>& target) {
                if (supplied.is_none()) return false;
                auto values =
                    supplied.cast<py::array_t<double, py::array::c_style>>();
                if (values.ndim() != 1 ||
                    size_t(values.size()) != target.size())
                    throw std::invalid_argument(
                        "prepared rescore self row count differs");
                for (size_t i = 0; i < target.size(); ++i) {
                    const double value = values.data()[i];
                    if (!std::isfinite(value) || value < 0.0)
                        throw std::invalid_argument(
                            "prepared rescore self must be finite nonnegative");
                    target[i] = value;
                }
                return true;
            };
            if (recompute_self && (!prepared_query_self.is_none() ||
                                   !prepared_library_self.is_none()))
                throw std::invalid_argument(
                    "recompute_self cannot use prepared rescore self");
            const bool reuse_query_self = copy_self(prepared_query_self, qself);
            const bool reuse_library_self =
                copy_self(prepared_library_self, lself);
            py::array_t<float> out({shape.shape(0), shape.shape(1)});
            std::memcpy(out.mutable_data(), shape.data(),
                        size_t(shape.size()) * sizeof(float));
            const float* src = shape.data();
            float* dst = out.mutable_data();
            const int32_t *qi = pq.data(), *li = pl.data();
            // Array owners remain alive. Below this point only cached native data
            // is accessed until the GIL is reacquired before the Python stats.
            auto nogil = std::make_unique<py::gil_scoped_release>();
            for (size_t i = 0; i < qv.size(); ++i)
                qv[i] = build_color_view_from_types(
                    qbase + i * qcapacity * qs, tbase + i * type_width,
                    nbase[i], int(qcapacity), nf, qs);
            for (size_t i = 0; i < lv.size(); ++i)
                lv[i] = build_color_view_from_counts(
                    lbase + i * lcapacity * ls, cbase + i * size_t(nf), nf, ls);
            PreparedColorConformerBatches conformer_batches;
            if (is_color_conformer_grain(rescore_grain))
                conformer_batches = prepare_color_conformer_batches(
                    qv, lv, qi, li, n_pairs, qs, ls, nf);
            const auto prep_s =
                std::chrono::duration<double>(clock() - t0).count();

            uint64_t qterms = 0, lterms = 0, cross_terms = 0;
            t0 = clock();
            if (!recompute_self && !reuse_query_self)
                for (size_t i = 0; i < qv.size(); ++i)
                    qself[i] = overlap(qv[i], qv[i], mat, qs, qs, nullptr,
                                       with_census ? &qterms : nullptr);
            const auto qs_s =
                std::chrono::duration<double>(clock() - t0).count();
            t0 = clock();
            if (!recompute_self && !reuse_library_self)
                for (size_t i = 0; i < lv.size(); ++i)
                    lself[i] = overlap(lv[i], lv[i], mat, ls, ls, nullptr,
                                       with_census ? &lterms : nullptr);
            const auto ls_s =
                std::chrono::duration<double>(clock() - t0).count();

            t0 = clock();
            double conformer_stable_sort_s =
                conformer_batches.stable_sort_seconds;
            double conformer_pack_s = conformer_batches.pack_seconds;
            double conformer_pose_transform_s = 0.0;
            double conformer_matrix_screen_s = 0.0;
            double conformer_scalar_fallback_s = 0.0;
            double conformer_pose_soa_pack_s = 0.0;
            double conformer_vector_kernel_s = 0.0;
            ColorConformerCensus conformer_census;
            ColorConformerWorkspace conformer_workspace;
            ColorConformerPhaseTimes conformer_phase_times{
                with_timers ? &conformer_pose_transform_s : nullptr,
                with_timers ? &conformer_matrix_screen_s : nullptr,
                with_timers ? &conformer_scalar_fallback_s : nullptr,
                with_timers ? &conformer_pose_soa_pack_s : nullptr,
                with_timers ? &conformer_vector_kernel_s : nullptr};
            if (is_color_conformer_grain(rescore_grain)) {
#if defined(__AVX2__)
                const size_t n_blocks = conformer_batches.blocks.size();
                for (size_t block_index = 0; block_index < n_blocks;
                     ++block_index) {
                    const ColorConformerBatch8& prepared =
                        conformer_batches.blocks[block_index];
                    const ColorConformerBatch8* color_batch_ptr = &prepared;
                    const int n_conformers = prepared.n_valid;
                    const size_t first_row = block_index * COLOR_CONFORMER_LANES;
                    const ColorConformerBatch8& color_batch = *color_batch_ptr;
                    size_t rows[COLOR_CONFORMER_LANES]{};
                    float quaternions[COLOR_CONFORMER_LANES][4]{};
                    float translations[COLOR_CONFORMER_LANES][3]{};
                    for (int lane = 0; lane < n_conformers; ++lane) {
                        const size_t row =
                            conformer_batches
                                .output_order[first_row + size_t(lane)];
                        rows[lane] = row;
                        const float* pose = src + row * 20;
                        for (int k = 0; k < 4; ++k)
                            quaternions[lane][k] = pose[9 + k];
                        for (int k = 0; k < 3; ++k)
                            translations[lane][k] = pose[13 + k];
                    }
                    ColorEvaluation colors[COLOR_CONFORMER_LANES];
                    uint64_t batch_terms[COLOR_CONFORMER_LANES]{};
                    evaluate_color_conformer_batch8(
                        color_batch, mat.view(), quaternions, translations,
                        false, colors, batch_terms,
                        with_census ? &conformer_census : nullptr,
                        with_timers ? &conformer_phase_times : nullptr,
                        &conformer_workspace);
                    for (int lane = 0; lane < n_conformers; ++lane) {
                        const size_t row = rows[lane];
                        // Rows come from the batch's output_order.
                        const float* in = src + row * 20;
                        float* o = dst + row * 20;
                        const double aa =
                            recompute_self ? overlap(qv[size_t(qi[row])],
                                                     qv[size_t(qi[row])], mat,
                                                     qs, qs, nullptr, nullptr)
                                           : qself[size_t(qi[row])];
                        const double bb =
                            recompute_self ? overlap(lv[size_t(li[row])],
                                                     lv[size_t(li[row])], mat,
                                                     ls, ls, nullptr, nullptr)
                                           : lself[size_t(li[row])];
                        cross_terms += batch_terms[lane];
                        const double denominator = aa + bb - colors[lane].value;
                        const float tc =
                            denominator > 0.0
                                ? float(colors[lane].value / denominator)
                                : 0.0f;
                        o[2] = tc;
                        o[4] = float(colors[lane].value);
                        o[7] = float(aa);
                        o[8] = float(bb);
                        o[0] = (1.0f - mixing) * in[1] + mixing * tc;
                    }
                }
#else
                throw std::invalid_argument(
                    "AVX2 conformer color grain unavailable");
#endif
            } else {
                for (size_t row = 0; row < n_pairs; ++row) {
                    const float* in = src + row * 20;
                    float* o = dst + row * 20;
                    double ab;
                    if (rescore_grain == 0) {
                        const auto posed = pose_color(lv[size_t(li[row])], ls,
                                                      in + 9, in + 13);
                        ab = overlap(qv[size_t(qi[row])], lv[size_t(li[row])],
                                     mat, qs, ls, posed.data(),
                                     with_census ? &cross_terms : nullptr);
                    } else {
                        ab = feature_color_evaluation(
                                 qv[size_t(qi[row])], lv[size_t(li[row])], mat,
                                 qs, ls, in + 9, in + 13, rescore_grain,
                                 with_census ? &cross_terms : nullptr, false)
                                 .value;
                    }
                    const double aa =
                        recompute_self
                            ? overlap(qv[size_t(qi[row])], qv[size_t(qi[row])],
                                      mat, qs, qs, nullptr, nullptr)
                            : qself[size_t(qi[row])];
                    const double bb =
                        recompute_self
                            ? overlap(lv[size_t(li[row])], lv[size_t(li[row])],
                                      mat, ls, ls, nullptr, nullptr)
                            : lself[size_t(li[row])];
                    // Write only the color columns (2, 4, 7, 8) and remix
                    // column 0 from the unchanged shape Tanimoto in column 1.
                    const double den = aa + bb - ab;
                    const float tc = den > 0.0 ? float(ab / den) : 0.0f;
                    o[2] = tc;
                    o[4] = float(ab);
                    o[7] = float(aa);
                    o[8] = float(bb);
                    o[0] = (1.0f - mixing) * in[1] + mixing * tc;
                }
            }
            const auto cross_s =
                std::chrono::duration<double>(clock() - t0).count();
            nogil.reset();
            py::dict stats;
            stats["query_prep"] = with_timers ? prep_s : 0.0;
            stats["query_self_overlap"] = with_timers ? qs_s : 0.0;
            stats["library_self_overlap"] = with_timers ? ls_s : 0.0;
            // Mode 2 evaluates no color during pose search. The winner-only
            // cross sum belongs solely to exact_final_rescore, not both phases.
            stats["analytic_color"] = 0.0;
            stats["exact_final_rescore"] = with_timers ? cross_s : 0.0;
            stats["color_query_self_terms"] = qterms;
            stats["color_library_self_terms"] = lterms;
            stats["color_cross_terms"] = cross_terms;
            stats["color_canonical"] = mat.canonical;
            if (is_color_conformer_grain(rescore_grain)) {
                stats["color_conformer_vector_evaluations"] =
                    conformer_census.vector_evaluations;
                stats["color_conformer_scalar_fallback_pairs"] =
                    conformer_census.scalar_fallback_pairs;
                stats["color_conformer_interaction_matrix_shape"] =
                    py::make_tuple(nf, nf);
                stats["color_conformer_useful_terms_by_interaction"] =
                    conformer_census.useful_terms_by_interaction;
                stats["color_conformer_issued_lanes_by_interaction"] =
                    conformer_census.issued_lanes_by_interaction;
                stats["color_conformer_target_count_bins"] =
                    conformer_census.target_count_bins;
                stats["color_conformer_stable_sort_seconds"] =
                    with_timers ? conformer_stable_sort_s : 0.0;
                stats["color_conformer_pack_seconds"] =
                    with_timers ? conformer_pack_s : 0.0;
                stats["color_conformer_pose_transform_seconds"] =
                    with_timers ? conformer_pose_transform_s : 0.0;
                stats["color_conformer_matrix_screen_seconds"] =
                    with_timers ? conformer_matrix_screen_s : 0.0;
                stats["color_conformer_scalar_fallback_seconds"] =
                    with_timers ? conformer_scalar_fallback_s : 0.0;
                stats["color_conformer_pose_soa_pack_seconds"] =
                    with_timers ? conformer_pose_soa_pack_s : 0.0;
                stats["color_conformer_vector_kernel_seconds"] =
                    with_timers ? conformer_vector_kernel_s : 0.0;
            }
            return py::make_tuple(out, stats);
        },
        py::arg("query_coords"), py::arg("query_types"),
        py::arg("query_n_atoms"), py::arg("library_coords"),
        py::arg("library_counts"), py::arg("pair_query_row"),
        py::arg("pair_library_row"), py::arg("interaction_r"),
        py::arg("interaction_p"), py::arg("shape_scores"), py::arg("mixing"),
        py::arg("with_census") = false, py::arg("with_timers") = false,
        py::arg("recompute_self") = false, py::arg("color_grain") = 0,
        py::arg("prepared_query_self") = py::none(),
        py::arg("prepared_library_self") = py::none());
}
}
}  // namespace r3::cpu

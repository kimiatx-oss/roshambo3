// Exact shared-pose color final scoring, isolated from the optimizer TU.
//
// Scores one color cross-overlap at a fixed final pose: the query in the
// reduced frame (A' = sA), the library in physical coordinates posed by the
// shared PreparedReducedPose (b' = M raw + t', M = sR). This is the
// reporting score, not the optimizer's SEARCH objective: it uses accurate
// std::exp and never the raw-bit or polynomial SIMD exponentials, and it
// computes no gradient.
//
// Two arithmetic routes, chosen per call:
//   - FP32: canonical-analytic plans whose inputs and transformed
//     coordinates lie in the FP32 envelope. Transform, distance, exp and
//     the running sum are all FP32. A nonfinite FP32 result falls through.
//   - FP64: general matrices and every input the FP32 route refuses. The
//     FP32 pose bytes are promoted to double; the quaternion is never
//     renormalized or rebuilt (doc/CPU_SHARED_POSE_CONTRACT.md).
//
// ExactFinalScratch (header) lets a worker reuse posed-library buffers
// across a trajectory and keep the winning start's posed points for one
// more exact evaluation. Pointer identities are stored only to detect a
// mismatched trajectory, never dereferenced later.
#include "color_exact_final.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "kernel.hpp"

namespace r3 {
namespace cpu {
namespace {

using Clock = std::chrono::steady_clock;

const ColorConformer&
reduced_features(const ReducedColorConformer& query) noexcept {
    return query.features;
}

uintptr_t pointer_identity(const void* pointer) noexcept {
    return reinterpret_cast<uintptr_t>(pointer);
}

// The exact scorer always consumes this plan's own reduced-decay view: the
// query is reduced-frame, so a physical-decay matrix would be wrong.
const ColorInvariantMatrixView&
validate_reduced_matrix(const ColorInvariantPlan& plan,
                        ColorInvariantMatrixView& storage) {
    storage = plan.reduced_matrix_view();
    if (storage.frame != ColorDecayFrame::Reduced ||
        !plan.matches_view(storage))
        throw std::invalid_argument(
            "exact-final color scoring requires this plan's reduced matrix view");
    if (storage.n_features < 1 || storage.n_features > MAX_COLOR_FEATURES ||
        storage.stride < storage.n_features ||
        storage.coefficients == nullptr || storage.decays == nullptr)
        throw std::invalid_argument(
            "exact-final color matrix view is incomplete");
    return storage;
}

// Color segments must be contiguous in type order from offset 0, sum to
// n_color without int overflow, and have a buffer when nonempty.
void validate_segments(const ColorConformer& conformer, const char* label) {
    if (conformer.n_features < 1 || conformer.n_features > MAX_COLOR_FEATURES ||
        conformer.n_color < 0)
        throw std::invalid_argument(std::string(label) +
                                    " has invalid color dimensions");
    int offset = 0;
    for (int type = 1; type < conformer.n_features; ++type) {
        if (conformer.seg_count[type] < 0 ||
            conformer.seg_offset[type] != offset)
            throw std::invalid_argument(std::string(label) +
                                        " has invalid color segments");
        if (conformer.seg_count[type] >
            std::numeric_limits<int>::max() - offset)
            throw std::invalid_argument(std::string(label) +
                                        " color segments overflow");
        offset += conformer.seg_count[type];
    }
    if (offset != conformer.n_color ||
        (conformer.n_color > 0 && conformer.color_xyz == nullptr))
        throw std::invalid_argument(std::string(label) +
                                    " color segments do not match its buffer");
}

void validate_views(const ColorConformer& query, const ColorConformer& library,
                    const ColorInvariantMatrixView& matrix, int query_stride,
                    int library_stride) {
    if (matrix.frame != ColorDecayFrame::Reduced ||
        matrix.n_features != query.n_features ||
        matrix.n_features != library.n_features)
        throw std::invalid_argument(
            "exact-final color views do not match the reduced matrix plan");
    if (query_stride < 3 || library_stride < 3)
        throw std::invalid_argument(
            "exact-final color coordinate strides must be at least three");
    validate_segments(query, "reduced query");
    validate_segments(library, "physical library");
}

// Pose identity is the exact matrix and translation bytes the scorer
// consumed (unit_quaternion is reporting-only and ignored). The contract tag
// is not compared, and there is no per-call finite pose scan:
// a finite pose is a construction precondition
// (doc/CPU_SHARED_POSE_CONTRACT.md).
bool same_scoring_pose(const PreparedReducedPose& a,
                       const PreparedReducedPose& b) noexcept {
    return std::memcmp(a.matrix, b.matrix, sizeof(a.matrix)) == 0 &&
           std::memcmp(a.translation, b.translation, sizeof(a.translation)) ==
               0;
}

// FP64 reference loop: query-type outer, then target type, query atom,
// target atom. A canonical matrix visits only the diagonal (target type ==
// query type); a general matrix visits every target type.
ColorEvaluation evaluate_overlap(const ColorConformer& query,
                                 const ColorConformer& library,
                                 const ColorInvariantMatrixView& matrix,
                                 int query_stride,
                                 const double* posed_library_reduced_xyz,
                                 uint64_t* analytic_pair_terms) {
    ColorEvaluation result;
    for (int query_type = 1; query_type < matrix.n_features; ++query_type) {
        const int target_type_begin = matrix.canonical ? query_type : 1;
        const int target_type_end =
            matrix.canonical ? query_type + 1 : matrix.n_features;
        for (int target_type = target_type_begin; target_type < target_type_end;
             ++target_type) {
            const size_t interaction =
                size_t(query_type) * size_t(matrix.stride) +
                size_t(target_type);
            const double decay = matrix.decays[interaction];
            // Preserve the established rule: zero width skips; a zero
            // coefficient with nonzero width remains an evaluated term.
            if (decay == 0.0) continue;
            const double coefficient = matrix.coefficients[interaction];
            for (int i = 0; i < query.seg_count[query_type]; ++i) {
                const float* x =
                    query.color_xyz + size_t(query.seg_offset[query_type] + i) *
                                          size_t(query_stride);
                for (int j = 0; j < library.seg_count[target_type]; ++j) {
                    const size_t target_feature =
                        size_t(library.seg_offset[target_type] + j);
                    const double* y =
                        posed_library_reduced_xyz + target_feature * 3;
                    const double dx = double(x[0]) - y[0];
                    const double dy = double(x[1]) - y[1];
                    const double dz = double(x[2]) - y[2];
                    result.value +=
                        coefficient *
                        std::exp(decay * (dx * dx + dy * dy + dz * dz));
                    if (analytic_pair_terms != nullptr) ++*analytic_pair_terms;
                }
            }
        }
    }
    return result;
}

// FP32 twin of evaluate_overlap: same loop and term order, with decay and
// coefficient narrowed once per interaction and one FP32 running sum across
// all planes. This accurate-FP32 final is the canonical default.
ColorEvaluation evaluate_overlap_fp32(const ColorConformer& query,
                                      const ColorConformer& library,
                                      const ColorInvariantMatrixView& matrix,
                                      int query_stride,
                                      const float* posed_library_reduced_xyz,
                                      uint64_t* analytic_pair_terms) {
    float total = 0.0f;
    for (int query_type = 1; query_type < matrix.n_features; ++query_type) {
        const int target_type_begin = matrix.canonical ? query_type : 1;
        const int target_type_end =
            matrix.canonical ? query_type + 1 : matrix.n_features;
        for (int target_type = target_type_begin; target_type < target_type_end;
             ++target_type) {
            const size_t interaction =
                size_t(query_type) * size_t(matrix.stride) +
                size_t(target_type);
            const float decay = float(matrix.decays[interaction]);
            // Preserve the established rule: zero width skips; a zero
            // coefficient with nonzero width remains an evaluated term.
            if (decay == 0.0f) continue;
            const float coefficient = float(matrix.coefficients[interaction]);
            for (int i = 0; i < query.seg_count[query_type]; ++i) {
                const float* x =
                    query.color_xyz + size_t(query.seg_offset[query_type] + i) *
                                          size_t(query_stride);
                for (int j = 0; j < library.seg_count[target_type]; ++j) {
                    const size_t target_feature =
                        size_t(library.seg_offset[target_type] + j);
                    const float* y =
                        posed_library_reduced_xyz + target_feature * 3;
                    const float dx = x[0] - y[0];
                    const float dy = x[1] - y[1];
                    const float dz = x[2] - y[2];
                    const float distance_squared = dx * dx + dy * dy + dz * dz;
                    total += coefficient * std::exp(decay * distance_squared);
                    if (analytic_pair_terms != nullptr) ++*analytic_pair_terms;
                }
            }
        }
    }
    ColorEvaluation result;
    result.value = double(total);
    return result;
}

// Pre-transform eligibility for the FP32 route: the reduced translation and
// every active query coordinate within the reduced bound, and every active
// library coordinate within the physical bound (1e9, the same envelope as
// the four-start helpers). Inactive types are not inspected.
bool canonical_fp32_inputs_safe(const ColorConformer& query,
                                const ColorConformer& library,
                                const ColorInvariantPlan& plan,
                                int query_stride, int library_stride,
                                const PreparedReducedPose& pose) {
    const float maximum_reduced =
        std::nextafter(float(1.0e9 * double(COORD_SCALE)), 0.0f);
    constexpr float maximum_raw = 1.0e9f;
    for (int component = 0; component < 3; ++component)
        if (!std::isfinite(pose.translation[component]) ||
            std::abs(pose.translation[component]) > maximum_reduced)
            return false;
    bool active_type[MAX_COLOR_FEATURES]{};
    for (const ColorCanonicalPlane& plane : plan.canonical_planes()) {
        const int type = plane.type;
        active_type[type] = true;
        for (int atom = 0; atom < query.seg_count[type]; ++atom) {
            const float* xyz =
                query.color_xyz +
                size_t(query.seg_offset[type] + atom) * size_t(query_stride);
            for (int component = 0; component < 3; ++component)
                if (!std::isfinite(xyz[component]) ||
                    std::abs(xyz[component]) > maximum_reduced)
                    return false;
        }
    }
    for (int type = 1; type < library.n_features; ++type)
        if (active_type[type])
            for (int atom = 0; atom < library.seg_count[type]; ++atom) {
                const float* xyz = library.color_xyz +
                                   size_t(library.seg_offset[type] + atom) *
                                       size_t(library_stride);
                for (int component = 0; component < 3; ++component)
                    if (!std::isfinite(xyz[component]) ||
                        std::abs(xyz[component]) > maximum_raw)
                        return false;
            }
    return true;
}

// Poses the active library atoms in FP32 into [feature][xyz] and rejects any
// nonfinite or out-of-bound result, which sends the call to FP64.
// Inactive-type slots are zero-filled; the canonical loop never reads them.
bool pose_library_fp32(const ColorConformer& library, int library_stride,
                       const ColorInvariantPlan& plan,
                       const PreparedReducedPose& pose, float* posed) {
    const float maximum_reduced =
        std::nextafter(float(1.0e9 * double(COORD_SCALE)), 0.0f);
    std::fill_n(posed, size_t(library.n_color) * 3, 0.0f);
    const float* m = pose.matrix;
    const float* t = pose.translation;
    for (const ColorCanonicalPlane& plane : plan.canonical_planes()) {
        const int type = plane.type;
        for (int atom = 0; atom < library.seg_count[type]; ++atom) {
            const int feature = library.seg_offset[type] + atom;
            const float* raw =
                library.color_xyz + size_t(feature) * size_t(library_stride);
            const float x =
                m[0] * raw[0] + m[1] * raw[1] + m[2] * raw[2] + t[0];
            const float y =
                m[3] * raw[0] + m[4] * raw[1] + m[5] * raw[2] + t[1];
            const float z =
                m[6] * raw[0] + m[7] * raw[1] + m[8] * raw[2] + t[2];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
                std::abs(x) > maximum_reduced ||
                std::abs(y) > maximum_reduced || std::abs(z) > maximum_reduced)
                return false;
            const size_t output = size_t(feature) * 3;
            posed[output] = x;
            posed[output + 1] = y;
            posed[output + 2] = z;
        }
    }
    return true;
}

}  // namespace

// Grow-only buffers. Switching precision discards the other precision's
// buffer. Returns true when an allocation happened (for the stats).
bool ExactFinalScratch::ensure_current_capacity(size_t points, bool fp32) {
    if (current_fp32_ != fp32) {
        current_library_reduced_xyz_.reset();
        current_library_reduced_xyz_fp32_.reset();
        current_capacity_points_ = 0;
        current_fp32_ = fp32;
    }
    if (points <= current_capacity_points_) return false;
    if (points > std::numeric_limits<size_t>::max() /
                     (3 * (fp32 ? sizeof(float) : sizeof(double))))
        throw std::length_error("exact-final current scratch is too large");
    if (fp32) {
        std::unique_ptr<float[]> replacement(new float[points * 3]);
        current_library_reduced_xyz_fp32_.swap(replacement);
    } else {
        std::unique_ptr<double[]> replacement(new double[points * 3]);
        current_library_reduced_xyz_.swap(replacement);
    }
    current_capacity_points_ = points;
    return true;
}

bool ExactFinalScratch::ensure_winner_capacity(size_t points, bool fp32) {
    if (winner_fp32_ != fp32) {
        winner_library_reduced_xyz_.reset();
        winner_library_reduced_xyz_fp32_.reset();
        winner_capacity_points_ = 0;
        winner_fp32_ = fp32;
    }
    if (points <= winner_capacity_points_) return false;
    if (points > std::numeric_limits<size_t>::max() /
                     (3 * (fp32 ? sizeof(float) : sizeof(double))))
        throw std::length_error("exact-final winner scratch is too large");
    if (fp32) {
        std::unique_ptr<float[]> replacement(new float[points * 3]);
        winner_library_reduced_xyz_fp32_.swap(replacement);
    } else {
        std::unique_ptr<double[]> replacement(new double[points * 3]);
        winner_library_reduced_xyz_.swap(replacement);
    }
    winner_capacity_points_ = points;
    return true;
}

ExactFinalScratch::ExactFinalScratch(ExactFinalScratch&& other) noexcept
    : trajectory_(other.trajectory_),
      trajectory_active_(other.trajectory_active_),
      current_pose_valid_(other.current_pose_valid_),
      winner_pose_valid_(other.winner_pose_valid_),
      current_pose_(other.current_pose_), winner_pose_(other.winner_pose_),
      current_library_reduced_xyz_(
          std::move(other.current_library_reduced_xyz_)),
      winner_library_reduced_xyz_(std::move(other.winner_library_reduced_xyz_)),
      current_library_reduced_xyz_fp32_(
          std::move(other.current_library_reduced_xyz_fp32_)),
      winner_library_reduced_xyz_fp32_(
          std::move(other.winner_library_reduced_xyz_fp32_)),
      current_capacity_points_(other.current_capacity_points_),
      winner_capacity_points_(other.winner_capacity_points_),
      current_fp32_(other.current_fp32_), winner_fp32_(other.winner_fp32_) {
    other.trajectory_ = {};
    other.trajectory_active_ = false;
    other.current_pose_valid_ = false;
    other.winner_pose_valid_ = false;
    other.current_capacity_points_ = 0;
    other.winner_capacity_points_ = 0;
    other.current_fp32_ = false;
    other.winner_fp32_ = false;
}

ExactFinalScratch&
ExactFinalScratch::operator=(ExactFinalScratch&& other) noexcept {
    if (this == &other) return *this;
    trajectory_ = other.trajectory_;
    trajectory_active_ = other.trajectory_active_;
    current_pose_valid_ = other.current_pose_valid_;
    winner_pose_valid_ = other.winner_pose_valid_;
    current_pose_ = other.current_pose_;
    winner_pose_ = other.winner_pose_;
    current_library_reduced_xyz_ =
        std::move(other.current_library_reduced_xyz_);
    winner_library_reduced_xyz_ = std::move(other.winner_library_reduced_xyz_);
    current_library_reduced_xyz_fp32_ =
        std::move(other.current_library_reduced_xyz_fp32_);
    winner_library_reduced_xyz_fp32_ =
        std::move(other.winner_library_reduced_xyz_fp32_);
    current_capacity_points_ = other.current_capacity_points_;
    winner_capacity_points_ = other.winner_capacity_points_;
    current_fp32_ = other.current_fp32_;
    winner_fp32_ = other.winner_fp32_;
    other.trajectory_ = {};
    other.trajectory_active_ = false;
    other.current_pose_valid_ = false;
    other.winner_pose_valid_ = false;
    other.current_capacity_points_ = 0;
    other.winner_capacity_points_ = 0;
    other.current_fp32_ = false;
    other.winner_fp32_ = false;
    return *this;
}

void ExactFinalScratch::begin_trajectory(
    const ReducedColorConformer& query_reduced,
    const ColorConformer& library_physical,
    const ColorInvariantPlan& invariant_plan, int query_stride,
    int library_stride) {
    // A failed attempt to start a new trajectory must not leave the previous
    // winner available to a later exact-final call.
    trajectory_active_ = false;
    current_pose_valid_ = false;
    winner_pose_valid_ = false;
    trajectory_ = {};
    const ColorConformer& query = reduced_features(query_reduced);
    ColorInvariantMatrixView matrix{};
    validate_reduced_matrix(invariant_plan, matrix);
    validate_views(query, library_physical, matrix, query_stride,
                   library_stride);

    trajectory_ = {};
    trajectory_.query_color = pointer_identity(query.color_xyz);
    trajectory_.library_color = pointer_identity(library_physical.color_xyz);
    trajectory_.coefficients = pointer_identity(matrix.coefficients);
    trajectory_.decays = pointer_identity(matrix.decays);
    trajectory_.query_n_color = query.n_color;
    trajectory_.library_n_color = library_physical.n_color;
    trajectory_.n_features = matrix.n_features;
    trajectory_.matrix_stride = matrix.stride;
    trajectory_.query_stride = query_stride;
    trajectory_.library_stride = library_stride;
    trajectory_.canonical = matrix.canonical;
    for (int type = 1; type < matrix.n_features; ++type) {
        trajectory_.query_offsets[type] = query.seg_offset[type];
        trajectory_.query_counts[type] = query.seg_count[type];
        trajectory_.library_offsets[type] = library_physical.seg_offset[type];
        trajectory_.library_counts[type] = library_physical.seg_count[type];
    }
    trajectory_active_ = true;
}

bool ExactFinalScratch::matches_trajectory(
    const ReducedColorConformer& query_reduced,
    const ColorConformer& library_physical,
    const ColorInvariantPlan& invariant_plan, int query_stride,
    int library_stride) const noexcept {
    if (!trajectory_active_) return false;
    const ColorConformer& query = reduced_features(query_reduced);
    const ColorInvariantMatrixView matrix =
        invariant_plan.reduced_matrix_view();
    if (matrix.frame != ColorDecayFrame::Reduced ||
        !invariant_plan.matches_view(matrix))
        return false;
    if (trajectory_.query_color != pointer_identity(query.color_xyz) ||
        trajectory_.library_color !=
            pointer_identity(library_physical.color_xyz) ||
        trajectory_.coefficients != pointer_identity(matrix.coefficients) ||
        trajectory_.decays != pointer_identity(matrix.decays) ||
        trajectory_.query_n_color != query.n_color ||
        trajectory_.library_n_color != library_physical.n_color ||
        trajectory_.n_features != matrix.n_features ||
        trajectory_.matrix_stride != matrix.stride ||
        trajectory_.query_stride != query_stride ||
        trajectory_.library_stride != library_stride ||
        trajectory_.canonical != matrix.canonical ||
        query.n_features != matrix.n_features ||
        library_physical.n_features != matrix.n_features)
        return false;
    for (int type = 1; type < matrix.n_features; ++type)
        if (trajectory_.query_offsets[type] != query.seg_offset[type] ||
            trajectory_.query_counts[type] != query.seg_count[type] ||
            trajectory_.library_offsets[type] !=
                library_physical.seg_offset[type] ||
            trajectory_.library_counts[type] !=
                library_physical.seg_count[type])
            return false;
    return true;
}

// Copies the posed points of the most recent exact call so the winner can be
// rescored without re-transforming. The pose must be the one that call just
// scored. Callers retain only on the existing strict-greater winner update,
// so the earliest of tied starts stays the winner.
void ExactFinalScratch::retain_current_pose_for_winner(
    const PreparedReducedPose& pose,
    ExactFinalScratchStats* preparation_stats) {
    const auto retention_started =
        preparation_stats != nullptr ? Clock::now() : Clock::time_point{};
    if (!trajectory_active_ || !current_pose_valid_ ||
        !same_scoring_pose(pose, current_pose_))
        throw std::invalid_argument(
            "no current exact-final pose matches the requested winner");
    const size_t points = size_t(trajectory_.library_n_color);
    const bool grew = ensure_winner_capacity(points, current_fp32_);
    if (points > 0) {
        if (current_fp32_)
            std::memcpy(winner_library_reduced_xyz_fp32_.get(),
                        current_library_reduced_xyz_fp32_.get(),
                        points * 3 * sizeof(float));
        else
            std::memcpy(winner_library_reduced_xyz_.get(),
                        current_library_reduced_xyz_.get(),
                        points * 3 * sizeof(double));
    }
    winner_pose_ = pose;
    winner_pose_valid_ = true;
    if (grew && preparation_stats != nullptr) {
        ++preparation_stats->buffer_growths;
        preparation_stats->payload_bytes_allocated +=
            uint64_t(winner_capacity_points_ * 3 *
                     (winner_fp32_ ? sizeof(float) : sizeof(double)));
    }
    if (preparation_stats != nullptr) {
        ++preparation_stats->winner_candidates_retained;
        preparation_stats->winner_point_bytes_copied += uint64_t(
            points * 3 * (current_fp32_ ? sizeof(float) : sizeof(double)));
        preparation_stats->winner_retention_seconds +=
            std::chrono::duration<double>(Clock::now() - retention_started)
                .count();
    }
}

ColorEvaluation score_exact_final_color(
    const ReducedColorConformer& query_reduced,
    const ColorConformer& library_physical,
    const ColorInvariantPlan& invariant_plan, const PreparedReducedPose& pose,
    int query_stride, int library_stride, ExactFinalScratch* scratch,
    uint64_t* analytic_pair_terms, ExactFinalScratchStats* preparation_stats,
    bool time_preparation) {
    const ColorConformer& query = reduced_features(query_reduced);
    ColorInvariantMatrixView matrix{};
    const ColorInvariantMatrixView& reduced_matrix =
        validate_reduced_matrix(invariant_plan, matrix);
    validate_views(query, library_physical, reduced_matrix, query_stride,
                   library_stride);

    // Without caller scratch, an empty per-call workspace runs the same
    // transform and overlap loops (the uncached arm in the header).
    ExactFinalScratch per_call_scratch;
    ExactFinalScratch& workspace = scratch ? *scratch : per_call_scratch;
    if (scratch != nullptr && !workspace.matches_trajectory(
                                  query_reduced, library_physical,
                                  invariant_plan, query_stride, library_stride))
        throw std::invalid_argument(
            "exact-final scratch belongs to a different trajectory");
    workspace.current_pose_valid_ = false;

    const auto prepare_started =
        time_preparation && preparation_stats != nullptr ? Clock::now()
                                                         : Clock::time_point{};
    // FP32 route first; any refusal or nonfinite sum falls through to
    // the FP64 route below with the same pose.
    const size_t points = size_t(library_physical.n_color);
    bool prepared_fp32 = false;
    if (invariant_plan.canonical_analytic_supported() &&
        invariant_plan.reduced_matrix_supported_for_avx2() &&
        canonical_fp32_inputs_safe(query, library_physical, invariant_plan,
                                   query_stride, library_stride, pose)) {
        const bool grew_fp32 = workspace.ensure_current_capacity(points, true);
        float* posed_fp32 = workspace.current_library_reduced_xyz_fp32_.get();
        prepared_fp32 = pose_library_fp32(library_physical, library_stride,
                                          invariant_plan, pose, posed_fp32);
        if (grew_fp32 && preparation_stats != nullptr) {
            ++preparation_stats->buffer_growths;
            preparation_stats->payload_bytes_allocated += uint64_t(
                workspace.current_capacity_points_ * 3 * sizeof(float));
        }
        if (prepared_fp32) {
            uint64_t fp32_terms = 0;
            const ColorEvaluation result = evaluate_overlap_fp32(
                query, library_physical, reduced_matrix, query_stride,
                posed_fp32,
                analytic_pair_terms != nullptr ? &fp32_terms : nullptr);
            if (std::isfinite(result.value)) {
                if (analytic_pair_terms != nullptr)
                    *analytic_pair_terms += fp32_terms;
                workspace.current_pose_ = pose;
                workspace.current_pose_valid_ = true;
                if (preparation_stats != nullptr) {
                    preparation_stats->library_points_prepared +=
                        uint64_t(library_physical.n_color);
                    if (time_preparation)
                        preparation_stats->coordinate_prepare_seconds +=
                            std::chrono::duration<double>(Clock::now() -
                                                          prepare_started)
                                .count();
                }
                return result;
            }
        }
    }

    const bool grew = workspace.ensure_current_capacity(points, false);
    double* posed = workspace.current_library_reduced_xyz_.get();

    // The one shared-pose transform is shared by every consumer: this
    // exact scorer promotes those FP32 M/t' bytes for its reference-precision
    // coordinate arithmetic. It never normalizes or reconstructs a quaternion.
    const float* m = pose.matrix;
    const float* t = pose.translation;
    for (int feature = 0; feature < library_physical.n_color; ++feature) {
        const float* raw = library_physical.color_xyz +
                           size_t(feature) * size_t(library_stride);
        const double x = double(raw[0]);
        const double y = double(raw[1]);
        const double z = double(raw[2]);
        const size_t output = size_t(feature) * 3;
        posed[output] = double(m[0]) * x + double(m[1]) * y + double(m[2]) * z +
                        double(t[0]);
        posed[output + 1] = double(m[3]) * x + double(m[4]) * y +
                            double(m[5]) * z + double(t[1]);
        posed[output + 2] = double(m[6]) * x + double(m[7]) * y +
                            double(m[8]) * z + double(t[2]);
    }
    workspace.current_pose_ = pose;
    workspace.current_pose_valid_ = true;

    if (preparation_stats != nullptr) {
        if (grew) {
            ++preparation_stats->buffer_growths;
            preparation_stats->payload_bytes_allocated += uint64_t(
                workspace.current_capacity_points_ * 3 * sizeof(double));
        }
        preparation_stats->library_points_prepared +=
            uint64_t(library_physical.n_color);
        if (time_preparation)
            preparation_stats->coordinate_prepare_seconds +=
                std::chrono::duration<double>(Clock::now() - prepare_started)
                    .count();
    }

    return evaluate_overlap(query, library_physical, reduced_matrix,
                            query_stride, posed, analytic_pair_terms);
}

// Winner rescoring from retained points: a full exact overlap evaluation
// (all pair terms counted again) in whichever precision the points were
// retained, skipping only the coordinate transform.
ColorEvaluation score_exact_final_color_retained(
    const ReducedColorConformer& query_reduced,
    const ColorConformer& library_physical,
    const ColorInvariantPlan& invariant_plan, const PreparedReducedPose& pose,
    int query_stride, int library_stride, ExactFinalScratch& scratch,
    uint64_t* analytic_pair_terms) {
    const ColorConformer& query = reduced_features(query_reduced);
    ColorInvariantMatrixView matrix{};
    const ColorInvariantMatrixView& reduced_matrix =
        validate_reduced_matrix(invariant_plan, matrix);
    validate_views(query, library_physical, reduced_matrix, query_stride,
                   library_stride);
    if (!scratch.matches_trajectory(query_reduced, library_physical,
                                    invariant_plan, query_stride,
                                    library_stride) ||
        !scratch.winner_pose_valid_ ||
        !same_scoring_pose(pose, scratch.winner_pose_))
        throw std::invalid_argument(
            "retained exact-final points do not match the winner pose or inputs");
    if (scratch.winner_fp32_)
        return evaluate_overlap_fp32(
            query, library_physical, reduced_matrix, query_stride,
            scratch.winner_library_reduced_xyz_fp32_.get(),
            analytic_pair_terms);
    return evaluate_overlap(
        query, library_physical, reduced_matrix, query_stride,
        scratch.winner_library_reduced_xyz_.get(), analytic_pair_terms);
}

}  // namespace cpu
}  // namespace r3

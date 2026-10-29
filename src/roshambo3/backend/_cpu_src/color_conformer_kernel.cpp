// Independent fp32x8 color evaluation, one pair per AVX2 lane.
//
// The conformer grain (production color grain 11). Each of the eight
// fp32 lanes owns one independent (query, library) pair with its own pose and
// its own accumulators, so no reduction ever crosses lanes; the lane axis is
// the conformer, not the query feature (cf. grain.hpp for the atom grains).
// This grain beat the earlier feature-grain baseline by 17-48% in six paired
// combo cells.
//
// One call scores a block of up to eight pairs at one pose each, as color
// overlap plus the seven-slot descent vector (color_evaluator.hpp):
//   1. pose the library features (fp64 legacy route, or the exact FP32
//      PreparedReducedPose bytes on the shared-pose route);
//   2. screen each lane: matrix support, ISA, finite/range coordinates;
//      a lane that fails is scored by the exact fp64 scalar `exact_lane`;
//   3. lanes that pass go to one of: the canonical analytic helper, the
//      per-plane ColorFieldBundle samples (per-plane hybrid), or the AVX2
//      analytic loop below.
// Library coordinates are a block-local SoA, [feature][component][lane]
// (`coordinate_index`), with per-type segments and -1/0 lane masks for the
// uneven per-pair counts.
//
// The SIMD path accumulates each target plane in fp32 and adds the plane
// subtotal to the lane's fp64 result, so its rounding depends on the width
// and is compared on ranked output, not bitwise.
#include "color_canonical_kernel.hpp"
#include "color_conformer_kernel.hpp"
#include "color_torque_factoring.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "kernel.hpp"

namespace r3 {
namespace cpu {
namespace {

// Support envelope for the fp32 SIMD path. Inputs outside it are not
// errors; the lane takes the exact scalar route instead.
constexpr double kMaximumCoefficient = 1.0e8;
constexpr double kMaximumWidth = 1.0e6;
constexpr double kMaximumCoordinate = 1.0e9;
constexpr float kMaximumRawCoordinateFp32 = 1.0e9f;

float maximum_reduced_coordinate_fp32() {
    // Round the inclusive FP64 contract down. A float just above the exact
    // bound must fall back instead of being admitted by a rounded-up limit.
    static const float value =
        std::nextafter(float(kMaximumCoordinate * double(COORD_SCALE)), 0.0f);
    return value;
}

// Finite and within the SIMD envelope.
bool safe_fp32_coordinate(float value, float maximum) {
    return std::isfinite(value) && std::abs(value) <= maximum;
}

// Batch8 SoA addressing: `base` is the type's segment offset, then
// [feature][component][lane] with eight lanes per row, so one 32-byte load
// reads one component of one feature slot across all eight pairs.
size_t coordinate_index(size_t base, int feature, int component, int lane) {
    return base +
           (size_t(feature) * 3 + size_t(component)) *
               size_t(COLOR_CONFORMER_LANES) +
           size_t(lane);
}

size_t mask_index(size_t base, int feature, int lane) {
    return base + size_t(feature) * size_t(COLOR_CONFORMER_LANES) +
           size_t(lane);
}

// Whether every active interaction cell fits the fp32 SIMD envelope:
// coefficient finite, nonnegative and <= 1e8, width -2*decay positive and
// <= 1e6, and both still finite after narrowing to float. Used unless the
// invariant plan supplies its precomputed answer
// (`reduced_matrix_supported_for_avx2`).
bool matrix_supported(const ColorInteractionMatrixView& matrix) {
    if (matrix.n_features < 1 || matrix.n_features > MAX_COLOR_FEATURES ||
        matrix.stride < matrix.n_features || matrix.coefficients == nullptr ||
        matrix.decays == nullptr)
        return false;
    for (int tb = 1; tb < matrix.n_features; ++tb) {
        const int ta_end = matrix.canonical ? tb + 1 : matrix.n_features;
        for (int ta = 1; ta < ta_end; ++ta) {
            const double decay = matrix.decay(ta, tb);
            if (decay == 0.0) continue;
            const double coefficient = matrix.coefficient(ta, tb);
            const float f_coefficient = float(coefficient);
            const float f_width = float(-2.0 * decay);
            if (!std::isfinite(coefficient) || coefficient < 0.0 ||
                coefficient > kMaximumCoefficient ||
                -2.0 * decay > kMaximumWidth || !std::isfinite(f_coefficient) ||
                !std::isfinite(f_width) || f_width <= 0.0f)
                return false;
        }
    }
    return true;
}

// A (query type, target type) cell contributes iff its decay is nonzero; the
// invariant plan, when used, answers the same question from a precomputed
// mask.
bool interaction_active(const ColorInteractionMatrixView& matrix, int ta,
                        int tb, const ColorInvariantPlan* plan, bool use_plan) {
    return use_plan ? plan->active_interaction(ta, tb)
                    : matrix.decay(ta, tb) != 0.0;
}

// Legacy route: unit quaternion (w, x, y, z), normalized in fp64, to a
// row-major rotation matrix. The translation is copied unchanged.
void make_pose(const float* quaternion, const float* translation,
               double matrix[9], double translated[3]) {
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
    const double values[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                              2 * (x * z + y * w),     2 * (x * y + z * w),
                              1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                              2 * (x * z - y * w),     2 * (y * z + x * w),
                              1 - 2 * (x * x + y * y)};
    for (int k = 0; k < 9; ++k)
        matrix[k] = values[k];
    for (int k = 0; k < 3; ++k)
        translated[k] = double(translation[k]);
}

// Legacy route: rotate and translate every live library feature of every
// valid lane in fp64 into `posed` (same SoA layout as `library_xyz`), and,
// when the SIMD path needs it, a sanitized fp32 copy.
void pose_target_block(const ColorConformerBatch8& batch,
                       const double matrix[COLOR_CONFORMER_LANES][9],
                       const double translation[COLOR_CONFORMER_LANES][3],
                       std::vector<double>& posed,
                       std::vector<float>* posed_fp32 = nullptr,
                       double coordinate_scale = 1.0) {
    posed.resize(batch.library_xyz.size());
    if (posed_fp32 != nullptr) posed_fp32->resize(posed.size());
    for (int tb = 1; tb < batch.n_features; ++tb) {
        for (int j = 0; j < batch.library_max[tb]; ++j) {
            for (int lane = 0; lane < batch.n_valid; ++lane) {
                if (j >= batch.library_count[tb][lane]) continue;
                const double bx = batch.library_xyz[coordinate_index(
                    batch.library_coordinate_offset[tb], j, 0, lane)];
                const double by = batch.library_xyz[coordinate_index(
                    batch.library_coordinate_offset[tb], j, 1, lane)];
                const double bz = batch.library_xyz[coordinate_index(
                    batch.library_coordinate_offset[tb], j, 2, lane)];
                const size_t at = batch.library_coordinate_offset[tb] +
                                  size_t(j) * 3 * size_t(COLOR_CONFORMER_LANES);
                const double x = matrix[lane][0] * bx + matrix[lane][1] * by +
                                 matrix[lane][2] * bz + translation[lane][0];
                const double y = matrix[lane][3] * bx + matrix[lane][4] * by +
                                 matrix[lane][5] * bz + translation[lane][1];
                const double z = matrix[lane][6] * bx + matrix[lane][7] * by +
                                 matrix[lane][8] * bz + translation[lane][2];
                const size_t x_at = at + size_t(lane);
                const size_t y_at =
                    at + size_t(COLOR_CONFORMER_LANES) + size_t(lane);
                const size_t z_at =
                    at + 2 * size_t(COLOR_CONFORMER_LANES) + size_t(lane);
                posed[x_at] = x;
                posed[y_at] = y;
                posed[z_at] = z;
                if (posed_fp32 != nullptr) {
                    const double limit = kMaximumCoordinate * coordinate_scale;
                    // Every live slot is initialized during this call. A
                    // harmless zero for unsafe values prevents masked SIMD
                    // intermediates from observing stale scratch or 0*NaN;
                    // the unsafe double is never narrowed.
                    (*posed_fp32)[x_at] =
                        std::isfinite(x) && std::abs(x) <= limit ? float(x)
                                                                 : 0.0f;
                    (*posed_fp32)[y_at] =
                        std::isfinite(y) && std::abs(y) <= limit ? float(y)
                                                                 : 0.0f;
                    (*posed_fp32)[z_at] =
                        std::isfinite(z) && std::abs(z) <= limit ? float(z)
                                                                 : 0.0f;
                }
            }
        }
    }
}

// Pose-dependent SIMD eligibility for one lane on the general route: the
// translation and every query and posed library coordinate that takes part
// in an active interaction must be finite and inside the envelope. Each type
// is scanned at most once (the bitmasks), however many partners it has.
bool pair_supported(const ColorConformerBatch8& batch, int lane,
                    const ColorInteractionMatrixView& matrix,
                    const double* posed,
                    const double translation[COLOR_CONFORMER_LANES][3],
                    const ColorInvariantPlan* plan = nullptr,
                    bool use_plan = false, bool reduced_frame = false) {
    const double coordinate_scale = reduced_frame ? double(COORD_SCALE) : 1.0;
    for (int k = 0; k < 3; ++k)
        if (!std::isfinite(translation[lane][k]) ||
            std::abs(translation[lane][k]) >
                kMaximumCoordinate * coordinate_scale)
            return false;
    // Eligibility is still defined by the active matrix pairs below, but the
    // query coordinates depend only on `ta` and the library checks only on
    // `tb`. Avoid repeating those same finite/range scans for every active
    // partner type in the matrix.
    uint32_t checked_query_types = 0;
    uint32_t checked_library_types = 0;
    for (int tb = 1; tb < matrix.n_features; ++tb) {
        const int ta_end = matrix.canonical ? tb + 1 : matrix.n_features;
        for (int ta = 1; ta < ta_end; ++ta) {
            if (!interaction_active(matrix, ta, tb, plan, use_plan) ||
                batch.query_count[ta][lane] == 0 ||
                batch.library_count[tb][lane] == 0)
                continue;
            const uint32_t query_bit = uint32_t(1) << unsigned(ta);
            if ((checked_query_types & query_bit) == 0) {
                checked_query_types |= query_bit;
                for (int i = 0; i < batch.query_count[ta][lane]; ++i) {
                    for (int k = 0; k < 3; ++k) {
                        const float x = batch.query_xyz[coordinate_index(
                            batch.query_coordinate_offset[ta], i, k, lane)];
                        if (!std::isfinite(x) ||
                            std::abs(double(x)) >
                                kMaximumCoordinate * coordinate_scale)
                            return false;
                    }
                }
            }
            const uint32_t library_bit = uint32_t(1) << unsigned(tb);
            if ((checked_library_types & library_bit) == 0) {
                checked_library_types |= library_bit;
                for (int j = 0; j < batch.library_count[tb][lane]; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        const size_t at = coordinate_index(
                            batch.library_coordinate_offset[tb], j, k, lane);
                        const float raw = batch.library_xyz[at];
                        if (!std::isfinite(raw) ||
                            std::abs(double(raw)) > kMaximumCoordinate ||
                            !std::isfinite(posed[at]) ||
                            std::abs(posed[at]) >
                                kMaximumCoordinate * coordinate_scale)
                            return false;
                    }
                }
            }
        }
    }
    return true;
}

// Shared-pose route: apply the lane's exact FP32 PreparedReducedPose (M = sR,
// t' = s t) to the raw library features, in FP32, writing the posed SoA and a
// per-(type, lane) safety flag. Fuses the transform with the finite/range
// screen, using the pack-time raw-coordinate flags. Padding slots and unsafe
// points are written as 0 so masked SIMD lanes never read stale scratch.
void pose_prepared_target_block_fp32(
    const ColorConformerBatch8& batch,
    const PreparedReducedPose* const poses[COLOR_CONFORMER_LANES],
    std::vector<float>& posed,
    uint8_t transformed_safe[MAX_COLOR_FEATURES][COLOR_CONFORMER_LANES]) {
    posed.resize(batch.library_xyz.size());
    const float maximum_reduced = maximum_reduced_coordinate_fp32();
    for (int type = 0; type < MAX_COLOR_FEATURES; ++type)
        for (int lane = 0; lane < batch.n_valid; ++lane) {
            const float* translation = poses[lane]->translation;
            transformed_safe[type][lane] =
                batch.canonical_library_coordinates_safe[size_t(type)]
                                                        [size_t(lane)] &&
                safe_fp32_coordinate(translation[0], maximum_reduced) &&
                safe_fp32_coordinate(translation[1], maximum_reduced) &&
                safe_fp32_coordinate(translation[2], maximum_reduced);
        }

    for (int type = 1; type < batch.n_features; ++type)
        for (int atom = 0; atom < batch.library_max[type]; ++atom)
            for (int lane = 0; lane < batch.n_valid; ++lane) {
                const size_t at =
                    batch.library_coordinate_offset[type] +
                    size_t(atom) * 3 * size_t(COLOR_CONFORMER_LANES);
                const size_t x_at = at + size_t(lane);
                const size_t y_at =
                    at + size_t(COLOR_CONFORMER_LANES) + size_t(lane);
                const size_t z_at =
                    at + 2 * size_t(COLOR_CONFORMER_LANES) + size_t(lane);
                if (atom >= batch.library_count[type][lane]) {
                    posed[x_at] = posed[y_at] = posed[z_at] = 0.0f;
                    continue;
                }

                if (!batch.canonical_library_coordinates_safe[size_t(type)]
                                                             [size_t(lane)]) {
                    transformed_safe[type][lane] = 0;
                    posed[x_at] = posed[y_at] = posed[z_at] = 0.0f;
                    continue;
                }

                const float bx = batch.library_xyz[x_at];
                const float by = batch.library_xyz[y_at];
                const float bz = batch.library_xyz[z_at];
                const float* matrix = poses[lane]->matrix;
                const float* translation = poses[lane]->translation;
                const float x = matrix[0] * bx + matrix[1] * by +
                                matrix[2] * bz + translation[0];
                const float y = matrix[3] * bx + matrix[4] * by +
                                matrix[5] * bz + translation[1];
                const float z = matrix[6] * bx + matrix[7] * by +
                                matrix[8] * bz + translation[2];
                const bool safe = safe_fp32_coordinate(x, maximum_reduced) &&
                                  safe_fp32_coordinate(y, maximum_reduced) &&
                                  safe_fp32_coordinate(z, maximum_reduced);
                transformed_safe[type][lane] &= uint8_t(safe);
                // Inactive SIMD lanes never observe a NaN or infinity, even
                // when this transformed point will take the scalar fallback.
                posed[x_at] = safe ? x : 0.0f;
                posed[y_at] = safe ? y : 0.0f;
                posed[z_at] = safe ? z : 0.0f;
            }
}

// Re-pose one lane in fp64 from the same FP32 pose bytes, for the scalar
// fallback after the FP32 canonical transform rejected that lane.
void pose_prepared_target_lane_fp64(const ColorConformerBatch8& batch, int lane,
                                    const PreparedReducedPose& pose,
                                    std::vector<double>& posed) {
    posed.resize(batch.library_xyz.size());
    const float* matrix = pose.matrix;
    const float* translation = pose.translation;
    for (int type = 1; type < batch.n_features; ++type)
        for (int atom = 0; atom < batch.library_count[type][lane]; ++atom) {
            const size_t at = batch.library_coordinate_offset[type] +
                              size_t(atom) * 3 * size_t(COLOR_CONFORMER_LANES);
            const double bx = double(batch.library_xyz[at + size_t(lane)]);
            const double by =
                double(batch.library_xyz[at + size_t(COLOR_CONFORMER_LANES) +
                                         size_t(lane)]);
            const double bz = double(
                batch.library_xyz[at + 2 * size_t(COLOR_CONFORMER_LANES) +
                                  size_t(lane)]);
            posed[at + size_t(lane)] =
                double(matrix[0]) * bx + double(matrix[1]) * by +
                double(matrix[2]) * bz + double(translation[0]);
            posed[at + size_t(COLOR_CONFORMER_LANES) + size_t(lane)] =
                double(matrix[3]) * bx + double(matrix[4]) * by +
                double(matrix[5]) * bz + double(translation[1]);
            posed[at + 2 * size_t(COLOR_CONFORMER_LANES) + size_t(lane)] =
                double(matrix[6]) * bx + double(matrix[7]) * by +
                double(matrix[8]) * bz + double(translation[2]);
        }
}

// Eligibility for the canonical analytic helper. Query eligibility is
// immutable and was cached at pack time (per-architecture flags); only the
// translation and the transform-time flags are pose dependent.
bool canonical_pair_supported(
    const ColorConformerBatch8& batch, int lane,
    const std::vector<ColorCanonicalPlane>& planes,
    const uint8_t transformed_safe[MAX_COLOR_FEATURES][COLOR_CONFORMER_LANES],
    const float translation[COLOR_CONFORMER_LANES][3]) {
    const float maximum_reduced = maximum_reduced_coordinate_fp32();
    for (int k = 0; k < 3; ++k)
        if (!safe_fp32_coordinate(translation[lane][k], maximum_reduced))
            return false;

    for (const ColorCanonicalPlane& plane : planes) {
        const int type = plane.type;
        if (batch.query_count[type][lane] == 0 ||
            batch.library_count[type][lane] == 0)
            continue;
#if defined(__aarch64__)
        if (!batch.query_plane_supported[type][lane]) return false;
#elif defined(__x86_64__)
        if (!batch.canonical_query_coordinates_safe[size_t(type)][size_t(lane)])
            return false;
#else
        for (int i = 0; i < batch.query_count[type][lane]; ++i)
            for (int component = 0; component < 3; ++component) {
                const float value = batch.query_xyz[coordinate_index(
                    batch.query_coordinate_offset[type], i, component, lane)];
                if (!safe_fp32_coordinate(value, maximum_reduced)) return false;
            }
#endif
        if (!batch.canonical_library_coordinates_safe[size_t(type)]
                                                     [size_t(lane)] ||
            !transformed_safe[type][lane])
            return false;
    }
    return true;
}

// The exact scalar reference for one lane, in fp64 and source order (target
// type, query type, query feature, library feature):
//
//   value      += c exp(decay |d|^2),  d = q - b, width = -2 decay
//   descent[1..3] += -2 width value (r x d),  r = b - t   (rotation)
//   descent[4..6] += -width value d  [/ COORD_SCALE on the physical route]
//
// It is the fallback for every lane the SIMD screen rejects, and the whole
// path on hosts without AVX2.
ColorEvaluation
exact_lane(const ColorConformerBatch8& batch, int lane,
           const ColorInteractionMatrixView& matrix, const double* posed,
           const double translation[COLOR_CONFORMER_LANES][3], bool gradient,
           uint64_t* term_count, bool reduced_frame = false,
           const ColorInvariantPlan* plan = nullptr, bool use_plan = false) {
    ColorEvaluation out;
    for (int tb = 1; tb < matrix.n_features; ++tb) {
        const int ta_end = matrix.canonical ? tb + 1 : matrix.n_features;
        for (int ta = 1; ta < ta_end; ++ta) {
            if (!interaction_active(matrix, ta, tb, plan, use_plan)) continue;
            const double decay = matrix.decay(ta, tb);
            const double coefficient = matrix.coefficient(ta, tb);
            const double width = -2.0 * decay;
            for (int i = 0; i < batch.query_count[ta][lane]; ++i) {
                const double x = batch.query_xyz[coordinate_index(
                    batch.query_coordinate_offset[ta], i, 0, lane)];
                const double y = batch.query_xyz[coordinate_index(
                    batch.query_coordinate_offset[ta], i, 1, lane)];
                const double z = batch.query_xyz[coordinate_index(
                    batch.query_coordinate_offset[ta], i, 2, lane)];
                for (int j = 0; j < batch.library_count[tb][lane]; ++j) {
                    const double bx = posed[coordinate_index(
                        batch.library_coordinate_offset[tb], j, 0, lane)];
                    const double by = posed[coordinate_index(
                        batch.library_coordinate_offset[tb], j, 1, lane)];
                    const double bz = posed[coordinate_index(
                        batch.library_coordinate_offset[tb], j, 2, lane)];
                    const double dx = x - bx;
                    const double dy = y - by;
                    const double dz = z - bz;
                    const double value =
                        coefficient *
                        std::exp(decay * (dx * dx + dy * dy + dz * dz));
                    out.value += value;
                    if (gradient) {
                        const double rx = bx - translation[lane][0];
                        const double ry = by - translation[lane][1];
                        const double rz = bz - translation[lane][2];
                        const double torque = -2.0 * width * value;
                        out.descent[1] += torque * (ry * dz - rz * dy);
                        out.descent[2] += torque * (rz * dx - rx * dz);
                        out.descent[3] += torque * (rx * dy - ry * dx);
                        const double scale =
                            -width * value /
                            (reduced_frame ? 1.0 : double(COORD_SCALE));
                        out.descent[4] += scale * dx;
                        out.descent[5] += scale * dy;
                        out.descent[6] += scale * dz;
                    }
                    if (term_count) ++*term_count;
                }
            }
        }
    }
    return out;
}

// Census only: credit a scalar-fallback lane's useful terms to the
// per-interaction counters, so useful/issued ratios include those lanes.
void record_fallback_terms(const ColorConformerBatch8& batch, int lane,
                           const ColorInteractionMatrixView& matrix,
                           ColorConformerCensus* census,
                           const ColorInvariantPlan* plan = nullptr,
                           bool use_plan = false) {
    if (census == nullptr) return;
    for (int ta = 1; ta < matrix.n_features; ++ta) {
        const int tb_begin = matrix.canonical ? ta : 1;
        const int tb_end = matrix.canonical ? ta + 1 : matrix.n_features;
        for (int tb = tb_begin; tb < tb_end; ++tb) {
            if (!interaction_active(matrix, ta, tb, plan, use_plan)) continue;
            const size_t cell =
                size_t(ta) * size_t(MAX_COLOR_FEATURES) + size_t(tb);
            census->useful_terms_by_interaction[cell] +=
                uint64_t(batch.query_count[ta][lane]) *
                uint64_t(batch.library_count[tb][lane]);
        }
    }
}

#if defined(__AVX2__)
// Same degree-seven polynomial and -80 tail as the selected FP32 width
// candidate. This changes only the axis assigned to the eight fp32 lanes.
//
// exp(x) = 2^n e^z with n = floor(x log2 e) and z = (x log2 e - n) ln 2 in
// [0, ln 2): the degree-seven Taylor polynomial in z times 2^n built directly
// in the exponent bits. Lanes with x < -80 return exactly 0 (and x is
// clamped first, so the exponent bits stay in range).
__m256 exp8_polynomial_bounded(__m256 x) {
    const __m256 cutoff = _mm256_set1_ps(-80.0f);
    const __m256 kept = _mm256_cmp_ps(x, cutoff, _CMP_GE_OQ);
    const __m256 bounded = _mm256_max_ps(x, cutoff);
    const __m256 y =
        _mm256_mul_ps(bounded, _mm256_set1_ps(1.4426950408889634074f));
    const __m256 n = _mm256_floor_ps(y);
    const __m256 z = _mm256_mul_ps(_mm256_sub_ps(y, n),
                                   _mm256_set1_ps(0.69314718055994530942f));
    const __m256i exponent = _mm256_cvttps_epi32(n);
    const __m256i bits = _mm256_slli_epi32(
        _mm256_add_epi32(exponent, _mm256_set1_epi32(127)), 23);
    constexpr float coefficient[] = {
        1.0f,         1.0f,          0.5f,          1.0f / 6.0f,
        1.0f / 24.0f, 1.0f / 120.0f, 1.0f / 720.0f, 1.0f / 5040.0f};
    __m256 polynomial = _mm256_set1_ps(coefficient[7]);
    for (int k = 6; k >= 0; --k)
        polynomial =
            _mm256_fmadd_ps(polynomial, z, _mm256_set1_ps(coefficient[k]));
    return _mm256_and_ps(_mm256_mul_ps(polynomial, _mm256_castsi256_ps(bits)),
                         kept);
}

#endif

}  // namespace

#if defined(__x86_64__)
// Recompute the cached per-(type, lane) reduced-query safety flags (hoisted
// out of the per-pose screen). Call after any change to the
// packed query payload.
void refresh_canonical_query_eligibility(ColorConformerBatch8& batch) {
    for (int type = 0; type < MAX_COLOR_FEATURES; ++type)
        batch.canonical_query_coordinates_safe[size_t(type)].fill(1);
    const double query_limit = kMaximumCoordinate * double(COORD_SCALE);
    for (int type = 1; type < batch.n_features; ++type)
        for (int lane = 0; lane < batch.n_valid; ++lane) {
            bool safe = true;
            for (int i = 0; i < batch.query_count[type][lane] && safe; ++i)
                for (int component = 0; component < 3; ++component) {
                    const float value = batch.query_xyz[coordinate_index(
                        batch.query_coordinate_offset[type], i, component,
                        lane)];
                    if (!std::isfinite(value) ||
                        std::abs(double(value)) > query_limit) {
                        safe = false;
                        break;
                    }
                }
            batch.canonical_query_coordinates_safe[size_t(type)][size_t(lane)] =
                uint8_t(safe);
        }
}
#endif

// Diagnostics only: pose a fixed block once so a timed evaluator call with
// `library_is_preposed` measures the kernel without the transform and pack.
void prepare_color_conformer_batch8_preposed(
    const ColorConformerBatch8& batch,
    const float quaternion[COLOR_CONFORMER_LANES][4],
    const float translation_physical[COLOR_CONFORMER_LANES][3],
    ColorConformerWorkspace* workspace) {
    if (batch.n_features < 1 || batch.n_features > MAX_COLOR_FEATURES ||
        batch.n_valid < 1 || batch.n_valid > COLOR_CONFORMER_LANES ||
        quaternion == nullptr || translation_physical == nullptr ||
        workspace == nullptr)
        throw std::invalid_argument("incomplete preposed color batch");
    double pose_matrix[COLOR_CONFORMER_LANES][9];
    double pose_translation[COLOR_CONFORMER_LANES][3];
    for (int lane = 0; lane < batch.n_valid; ++lane)
        make_pose(quaternion[lane], translation_physical[lane],
                  pose_matrix[lane], pose_translation[lane]);
    pose_target_block(batch, pose_matrix, pose_translation,
                      workspace->posed_library_xyz);
    workspace->posed_library_xyz_fp32.resize(
        workspace->posed_library_xyz.size());
    for (int tb = 1; tb < batch.n_features; ++tb)
        for (int j = 0; j < batch.library_max[tb]; ++j)
            for (int component = 0; component < 3; ++component)
                for (int lane = 0; lane < batch.n_valid; ++lane) {
                    if (j >= batch.library_count[tb][lane]) continue;
                    const size_t at =
                        coordinate_index(batch.library_coordinate_offset[tb], j,
                                         component, lane);
                    workspace->posed_library_xyz_fp32[at] =
                        float(workspace->posed_library_xyz[at]);
                }
}

// Pack up to eight pairs into the block SoA. Each type's segment is sized by
// the largest count among the lanes for that type; shorter lanes are zero
// padded with mask 0, live slots get mask -1 (all bits set, for AND masking).
// Raw library and reduced query coordinate eligibility is cached here, since
// it does not change with the pose.
void pack_color_conformer_batch8_into(const ColorConformerPairInput* pairs,
                                      int n_pairs, int n_features,
                                      ColorConformerBatch8& batch) {
    if (pairs == nullptr || n_pairs < 1 || n_pairs > COLOR_CONFORMER_LANES ||
        n_features < 1 || n_features > MAX_COLOR_FEATURES)
        throw std::invalid_argument("invalid conformer color block dimensions");
    std::fill_n(&batch.query_count[0][0],
                MAX_COLOR_FEATURES * COLOR_CONFORMER_LANES, 0);
    std::fill_n(&batch.library_count[0][0],
                MAX_COLOR_FEATURES * COLOR_CONFORMER_LANES, 0);
    std::fill_n(batch.query_max, MAX_COLOR_FEATURES, 0);
    std::fill_n(batch.library_max, MAX_COLOR_FEATURES, 0);
    std::fill_n(batch.query_coordinate_offset, MAX_COLOR_FEATURES, size_t(0));
    std::fill_n(batch.library_coordinate_offset, MAX_COLOR_FEATURES, size_t(0));
    std::fill_n(batch.query_mask_offset, MAX_COLOR_FEATURES, size_t(0));
    std::fill_n(batch.library_mask_offset, MAX_COLOR_FEATURES, size_t(0));
    batch.n_features = n_features;
    batch.n_valid = n_pairs;
    size_t query_coordinate_size = 0, library_coordinate_size = 0;
    size_t query_mask_size = 0, library_mask_size = 0;
    for (int lane = 0; lane < n_pairs; ++lane) {
        if (pairs[lane].query == nullptr || pairs[lane].library == nullptr ||
            pairs[lane].query->n_features != n_features ||
            pairs[lane].library->n_features != n_features ||
            pairs[lane].query_stride < 3 || pairs[lane].library_stride < 3)
            throw std::invalid_argument("incomplete conformer color pair input");
        for (int t = 1; t < n_features; ++t) {
            const int qn = pairs[lane].query->seg_count[t];
            const int bn = pairs[lane].library->seg_count[t];
            if (qn < 0 || bn < 0)
                throw std::invalid_argument(
                    "negative conformer color segment count");
            batch.query_count[t][lane] = qn;
            batch.library_count[t][lane] = bn;
            batch.query_max[t] = std::max(batch.query_max[t], qn);
            batch.library_max[t] = std::max(batch.library_max[t], bn);
        }
    }
    for (int t = 1; t < n_features; ++t) {
        batch.query_coordinate_offset[t] = query_coordinate_size;
        batch.library_coordinate_offset[t] = library_coordinate_size;
        batch.query_mask_offset[t] = query_mask_size;
        batch.library_mask_offset[t] = library_mask_size;
        query_coordinate_size +=
            size_t(batch.query_max[t]) * 3 * size_t(COLOR_CONFORMER_LANES);
        library_coordinate_size +=
            size_t(batch.library_max[t]) * 3 * size_t(COLOR_CONFORMER_LANES);
        query_mask_size +=
            size_t(batch.query_max[t]) * size_t(COLOR_CONFORMER_LANES);
        library_mask_size +=
            size_t(batch.library_max[t]) * size_t(COLOR_CONFORMER_LANES);
    }
    batch.query_xyz.assign(query_coordinate_size, 0.0f);
    batch.library_xyz.assign(library_coordinate_size, 0.0f);
    batch.query_mask.assign(query_mask_size, 0);
    batch.library_mask.assign(library_mask_size, 0);
    for (auto& safe : batch.canonical_library_coordinates_safe)
        safe.fill(1);
    for (int lane = 0; lane < n_pairs; ++lane) {
        for (int t = 1; t < n_features; ++t) {
            const ColorConformer& query = *pairs[lane].query;
            const ColorConformer& library = *pairs[lane].library;
#if defined(__aarch64__)
            batch.query_plane_supported[t][lane] = true;
#endif
            for (int i = 0; i < query.seg_count[t]; ++i) {
                const float* xyz =
                    query.color_xyz + size_t(query.seg_offset[t] + i) *
                                          size_t(pairs[lane].query_stride);
                batch.query_mask[mask_index(batch.query_mask_offset[t], i,
                                            lane)] = -1;
                for (int k = 0; k < 3; ++k) {
                    batch.query_xyz[coordinate_index(
                        batch.query_coordinate_offset[t], i, k, lane)] = xyz[k];
#if defined(__aarch64__)
                    if (!std::isfinite(xyz[k]) ||
                        std::abs(double(xyz[k])) >
                            kMaximumCoordinate * double(COORD_SCALE))
                        batch.query_plane_supported[t][lane] = false;
#endif
                }
            }
            for (int j = 0; j < library.seg_count[t]; ++j) {
                const float* xyz =
                    library.color_xyz + size_t(library.seg_offset[t] + j) *
                                            size_t(pairs[lane].library_stride);
                batch.library_mask[mask_index(batch.library_mask_offset[t], j,
                                              lane)] = -1;
                for (int k = 0; k < 3; ++k) {
                    batch.library_xyz[coordinate_index(
                        batch.library_coordinate_offset[t], j, k, lane)] =
                        xyz[k];
                    if (!safe_fp32_coordinate(xyz[k],
                                              kMaximumRawCoordinateFp32))
                        batch.canonical_library_coordinates_safe[size_t(t)]
                                                                [size_t(lane)] =
                            0;
                }
            }
        }
    }
#if defined(__x86_64__)
    refresh_canonical_query_eligibility(batch);
#endif
}

ColorConformerBatch8
pack_color_conformer_batch8(const ColorConformerPairInput* pairs, int n_pairs,
                            int n_features) {
    ColorConformerBatch8 batch;
    pack_color_conformer_batch8_into(pairs, n_pairs, n_features, batch);
    return batch;
}

// Prepared-route packer: the same layout, fed the reduced query features.
void pack_color_conformer_batch8_prepared_into(
    const ReducedColorConformerPairInput* pairs, int n_pairs, int n_features,
    ColorConformerBatch8& batch) {
    if (pairs == nullptr || n_pairs < 1 || n_pairs > COLOR_CONFORMER_LANES)
        throw std::invalid_argument(
            "invalid prepared color block dimensions");
    std::array<ColorConformerPairInput, COLOR_CONFORMER_LANES> views{};
    for (int lane = 0; lane < n_pairs; ++lane) {
        if (pairs[lane].query == nullptr)
            throw std::invalid_argument(
                "prepared color pair lacks reduced query features");
        views[size_t(lane)] = {&pairs[lane].query->features,
                               pairs[lane].library, pairs[lane].query_stride,
                               pairs[lane].library_stride};
    }
    pack_color_conformer_batch8_into(views.data(), n_pairs, n_features, batch);
}

ColorConformerBatch8 pack_color_conformer_batch8_prepared(
    const ReducedColorConformerPairInput* pairs, int n_pairs, int n_features) {
    ColorConformerBatch8 batch;
    pack_color_conformer_batch8_prepared_into(pairs, n_pairs, n_features,
                                              batch);
    return batch;
}

// One implementation behind both public entry points. `prepared_poses`
// non-null selects the reduced-frame shared-pose route; null
// selects the legacy physical route from quaternion and translation. Every
// output lane is overwritten; lanes >= n_valid stay zero.
static void evaluate_color_conformer_batch8_impl(
    const ColorConformerBatch8& batch,
    const ColorInteractionMatrixView& interaction,
    const float quaternion[COLOR_CONFORMER_LANES][4],
    const float translation_physical[COLOR_CONFORMER_LANES][3], bool gradient,
    ColorEvaluation output[COLOR_CONFORMER_LANES],
    uint64_t term_counts[COLOR_CONFORMER_LANES], ColorConformerCensus* census,
    ColorConformerPhaseTimes* phase_times, ColorConformerWorkspace* workspace,
    const ColorFieldBundle* const* field_bundles,
    ColorFieldBatchStats* field_stats, bool library_is_preposed,
    const PreparedReducedPose* const* prepared_poses,
    const ColorInvariantPlan* plan, bool use_invariant_plan,
    bool factor_canonical_torque, OptimizerExpMode optimizer_exp_mode,
    bool canonical_color_kernel) {
    if (batch.n_features < 1 || batch.n_features > MAX_COLOR_FEATURES ||
        batch.n_valid < 1 || batch.n_valid > COLOR_CONFORMER_LANES ||
        interaction.n_features != batch.n_features ||
        interaction.stride < interaction.n_features ||
        interaction.coefficients == nullptr || interaction.decays == nullptr ||
        ((prepared_poses == nullptr) &&
         (quaternion == nullptr || translation_physical == nullptr)) ||
        output == nullptr || term_counts == nullptr)
        throw std::invalid_argument("incomplete conformer color evaluation batch");
    const bool reduced_frame = prepared_poses != nullptr;
    if (reduced_frame) {
        if (plan == nullptr || library_is_preposed)
            throw std::invalid_argument(
                "invalid reduced-frame color evaluator inputs");
        const auto reduced_view = plan->reduced_matrix_view();
        if (!plan->matches_view(reduced_view) ||
            !plan->matches_view({interaction.n_features, interaction.stride,
                                 interaction.canonical,
                                 interaction.coefficients, interaction.decays,
                                 ColorDecayFrame::Reduced}))
            throw std::invalid_argument(
                "shared-pose color evaluator requires the plan's reduced matrix view");
        for (int lane = 0; lane < batch.n_valid; ++lane)
            if (prepared_poses[lane] == nullptr)
                throw std::invalid_argument(
                    "shared-pose color lane lacks a prepared shared pose");
    }
    // The canonical helper is eligible when it is requested, the plan's
    // matrix is the canonical one, and no field bundles are given; it then
    // takes the whole call (below) if the ISA/matrix gate also passes.
    const bool canonical_analytic_path =
        canonical_color_kernel && reduced_frame && plan != nullptr &&
        plan->canonical_analytic_supported() && field_bundles == nullptr;

    for (int lane = 0; lane < COLOR_CONFORMER_LANES; ++lane) {
        output[lane] = ColorEvaluation{};
        term_counts[lane] = 0;
    }
    if (census) census->last_vector_support_mask = 0;

    double pose_matrix[COLOR_CONFORMER_LANES][9]{};
    double pose_translation[COLOR_CONFORMER_LANES][3]{};
    float pose_translation_fp32[COLOR_CONFORMER_LANES][3]{};
    auto started = std::chrono::steady_clock::time_point{};
    if (phase_times && phase_times->pose_transform_seconds)
        started = std::chrono::steady_clock::now();
    for (int lane = 0; lane < batch.n_valid; ++lane) {
        if (reduced_frame) {
            const PreparedReducedPose& pose = *prepared_poses[lane];
            for (int k = 0; k < 3; ++k)
                pose_translation_fp32[lane][k] = pose.translation[k];
        } else {
            make_pose(quaternion[lane], translation_physical[lane],
                      pose_matrix[lane], pose_translation[lane]);
            for (int k = 0; k < 3; ++k)
                pose_translation_fp32[lane][k] =
                    float(pose_translation[lane][k]);
        }
    }

    // This gate is independent of transformed coordinates. Resolve it after
    // pose validation/setup but before target storage, so scalar-only calls do
    // not resize or touch the FP32 workspace cache.
    if (phase_times && phase_times->pose_transform_seconds) {
        *phase_times->pose_transform_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started)
                .count();
        started = std::chrono::steady_clock::time_point{};
    }
    auto matrix_gate_started = std::chrono::steady_clock::time_point{};
    if (phase_times && phase_times->matrix_screen_seconds)
        matrix_gate_started = std::chrono::steady_clock::now();
    bool use_vector = canonical_analytic_path
                          ? plan->reduced_matrix_supported_for_avx2()
                      : reduced_frame && use_invariant_plan
                          ? plan->reduced_matrix_supported_for_avx2()
                          : matrix_supported(interaction);
#if defined(__AVX2__)
    if (reduced_frame && use_invariant_plan) {
        use_vector = use_vector && plan->avx2_fma_available();
    } else {
        __builtin_cpu_init();
        use_vector = use_vector && __builtin_cpu_supports("avx2") &&
                     __builtin_cpu_supports("fma");
    }
#elif defined(__aarch64__)
    // Only canonical reduced-frame analytic color is native NEON. General
    // matrices and legacy physical paths retain exact scalar fallback.
    use_vector = use_vector && canonical_analytic_path;
#else
    use_vector = false;
#endif
    const double matrix_gate_seconds =
        phase_times && phase_times->matrix_screen_seconds
            ? std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            matrix_gate_started)
                  .count()
            : 0.0;

    const bool fp32_canonical_transform = canonical_analytic_path && use_vector;
    // Other reduced-frame routes pose in fp64 from the same FP32 pose bytes.
    if (reduced_frame && !fp32_canonical_transform) {
        for (int lane = 0; lane < batch.n_valid; ++lane) {
            const PreparedReducedPose& pose = *prepared_poses[lane];
            for (int k = 0; k < 9; ++k)
                pose_matrix[lane][k] = double(pose.matrix[k]);
            for (int k = 0; k < 3; ++k)
                pose_translation[lane][k] = double(pose.translation[k]);
        }
    }

    if (phase_times && phase_times->pose_transform_seconds)
        started = std::chrono::steady_clock::now();
    ColorConformerWorkspace local_workspace;
    ColorConformerWorkspace& scratch = workspace ? *workspace : local_workspace;
    auto& posed = scratch.posed_library_xyz;
    auto& posed_fp32 = scratch.posed_library_xyz_fp32;
    uint8_t transformed_safe[MAX_COLOR_FEATURES][COLOR_CONFORMER_LANES]{};
    if (library_is_preposed) {
        if (workspace == nullptr || posed.size() != batch.library_xyz.size() ||
            posed_fp32.size() != posed.size())
            throw std::invalid_argument(
                "preposed color batch workspace was not prepared");
    } else if (fp32_canonical_transform) {
        pose_prepared_target_block_fp32(batch, prepared_poses, posed_fp32,
                                        transformed_safe);
    } else {
        auto* posed_f32 = use_vector ? &posed_fp32 : nullptr;
        const double coordinate_scale =
            reduced_frame ? double(COORD_SCALE) : 1.0;
        pose_target_block(batch, pose_matrix, pose_translation, posed,
                          posed_f32, coordinate_scale);
    }
    if (phase_times && phase_times->pose_transform_seconds)
        *phase_times->pose_transform_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started)
                .count();

    // ColorFieldBundle's sampling interface takes physical-Angstrom points and
    // returns physical gradients. (Internally its grids and stored gradient
    // channels are in the reduced frame -- hs = spacing * COORD_SCALE -- and
    // are converted at that interface; see color_field.cpp.)
    // The analytic path above/below remains entirely in A'=sA using the exact
    // prepared FP32 matrix and translation bytes. Only selected field samples
    // convert the already transformed point back to physical coordinates.
    const std::vector<double>* field_posed = &posed;
    if (reduced_frame && field_bundles != nullptr) {
        const auto field_transform_started =
            phase_times && phase_times->pose_transform_seconds
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
        auto& physical = scratch.posed_library_xyz_physical;
        physical.assign(posed.size(), 0.0);
        const double scale = double(COORD_SCALE);
        for (int tb = 1; tb < batch.n_features; ++tb)
            for (int j = 0; j < batch.library_max[tb]; ++j)
                for (int component = 0; component < 3; ++component)
                    for (int lane = 0; lane < batch.n_valid; ++lane) {
                        if (j >= batch.library_count[tb][lane]) continue;
                        const size_t at = coordinate_index(
                            batch.library_coordinate_offset[tb], j, component,
                            lane);
                        physical[at] = posed[at] / scale;
                    }
        field_posed = &physical;
        if (phase_times && phase_times->pose_transform_seconds)
            *phase_times->pose_transform_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              field_transform_started)
                    .count();
    }

    if (phase_times && phase_times->matrix_screen_seconds)
        started = std::chrono::steady_clock::now();
    // Per-lane screen. Rejected lanes are scored right here by `exact_lane`;
    // `simd_mask` marks the lanes left for the vector paths below.
    int32_t simd_mask[COLOR_CONFORMER_LANES]{};
    double fallback_seconds = 0.0;
    for (int lane = 0; lane < batch.n_valid; ++lane) {
        const bool supported =
            use_vector &&
            (canonical_analytic_path
                 ? canonical_pair_supported(
                       batch, lane, plan->canonical_planes(), transformed_safe,
                       pose_translation_fp32)
                 : pair_supported(batch, lane, interaction, posed.data(),
                                  pose_translation, plan,
                                  reduced_frame && use_invariant_plan,
                                  reduced_frame));
        simd_mask[lane] = supported ? -1 : 0;
        if (supported && census)
            census->last_vector_support_mask |= uint8_t(1u << unsigned(lane));
        if (census) {
            if (supported)
                ++census->vector_eligible_lane_calls[size_t(lane)];
            else
                ++census->scalar_fallback_lane_calls[size_t(lane)];
        }
        if (!supported) {
            const bool measure_fallback =
                phase_times && (phase_times->scalar_fallback_seconds ||
                                phase_times->matrix_screen_seconds);
            const auto fallback_started =
                measure_fallback ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
            if (fp32_canonical_transform) {
                pose_prepared_target_lane_fp64(batch, lane,
                                               *prepared_poses[lane], posed);
                for (int k = 0; k < 3; ++k)
                    pose_translation[lane][k] =
                        double(prepared_poses[lane]->translation[k]);
            }
            output[lane] = exact_lane(batch, lane, interaction, posed.data(),
                                      pose_translation, gradient,
                                      &term_counts[lane], reduced_frame, plan,
                                      reduced_frame && use_invariant_plan);
            // Fallback time is moved from the screen phase to its own.
            if (measure_fallback)
                fallback_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - fallback_started)
                        .count();
            record_fallback_terms(batch, lane, interaction, census, plan,
                                  reduced_frame && use_invariant_plan);
            if (census) ++census->scalar_fallback_pairs;
        }
    }
    if (phase_times && phase_times->matrix_screen_seconds)
        *phase_times->matrix_screen_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started)
                .count() -
            fallback_seconds + matrix_gate_seconds;
    if (phase_times && phase_times->scalar_fallback_seconds)
        *phase_times->scalar_fallback_seconds += fallback_seconds;

    // Dedicated canonical dispatch is independent of the older invariant-plan
    // optimization flag. Field bundles, general matrices and unsafe lanes keep
    // their established paths. Call-local modes reach the dedicated helper.
    if (canonical_analytic_path && use_vector) {
#if defined(__aarch64__)
        evaluate_canonical_color_analytic_batch8(
            batch, plan->canonical_planes(),
#else
        evaluate_canonical_color_analytic_batch8(
            batch, *plan,
#endif
            posed_fp32.data(), pose_translation_fp32, simd_mask, gradient,
            output, term_counts, census, field_stats, factor_canonical_torque,
            optimizer_exp_mode);
        return;
    }

    // Decide field use independently for each (target type, pair lane). A
    // selected plane is interpolated only when every target sample for that
    // lane is inside its query-built grid. Otherwise that lane's whole plane
    // remains in the AVX2 analytic loop, so an out-of-grid value never drops
    // to the old scalar analytic implementation on a supported input.
    bool use_field[MAX_COLOR_FEATURES][COLOR_CONFORMER_LANES]{};
    uint64_t query_terms[MAX_COLOR_FEATURES][COLOR_CONFORMER_LANES]{};
    for (int tb = 1; tb < batch.n_features; ++tb) {
        const int ta_begin = interaction.canonical ? tb : 1;
        const int ta_end = interaction.canonical ? tb + 1 : batch.n_features;
        for (int lane = 0; lane < batch.n_valid; ++lane) {
            if (!simd_mask[lane]) continue;
            // Query features this target plane sums over: the analytic terms
            // one field sample replaces, for the term and census counts.
            for (int ta = ta_begin; ta < ta_end; ++ta)
                if (interaction.coefficient(ta, tb) != 0.0 &&
                    interaction_active(interaction, ta, tb, plan,
                                       reduced_frame && use_invariant_plan))
                    query_terms[tb][lane] +=
                        uint64_t(batch.query_count[ta][lane]);
            const uint64_t target_count =
                uint64_t(batch.library_count[tb][lane]);
            if (query_terms[tb][lane] == 0 || target_count == 0) continue;

            const ColorFieldBundle* bundle =
                field_bundles ? field_bundles[lane] : nullptr;
            const int backend =
                bundle ? bundle->backend_code(tb) : COLOR_FIELD_ANALYTIC;
            if (field_stats) {
                field_stats->samples += target_count;
                if (backend == COLOR_FIELD_ANALYTIC) {
                    field_stats->analytic_selected_samples += target_count;
                    field_stats->analytic_selected_terms +=
                        query_terms[tb][lane] * target_count;
                }
            }
            if (backend == COLOR_FIELD_ANALYTIC) continue;

            if (!bundle->field_active(tb)) {
                if (field_stats) {
                    field_stats->analytic_fallback_samples += target_count;
                    field_stats->inactive_fallback_samples += target_count;
                    field_stats->analytic_fallback_terms +=
                        query_terms[tb][lane] * target_count;
                }
                continue;
            }

            // All of this lane's targets of type tb must be inside the grid
            // before any is sampled (all-or-nothing per plane and lane).
            // R3_COLOR_FIELD_BATCH selects the batched `sample_group` instead.
            bool in_domain = true;
#if R3_COLOR_FIELD_BATCH
            auto& fw = scratch.field_samples;
            fw.points.resize(size_t(target_count) * 3);
            for (size_t j = 0; j < size_t(target_count); ++j) {
                const size_t at = batch.library_coordinate_offset[tb] +
                                  j * 3 * size_t(COLOR_CONFORMER_LANES) +
                                  size_t(lane);
                for (int k = 0; k < 3; ++k)
                    fw.points[j * 3 + size_t(k)] = float(
                        (*field_posed)[at + size_t(k) * COLOR_CONFORMER_LANES]);
            }
            in_domain = bundle->sample_group(tb, fw.points.data(),
                                             size_t(target_count), 3, 1, fw);
#else
            for (int j = 0; j < batch.library_count[tb][lane]; ++j) {
                const size_t at =
                    batch.library_coordinate_offset[tb] +
                    size_t(j) * 3 * size_t(COLOR_CONFORMER_LANES) +
                    size_t(lane);
                const float point[3] = {
                    float((*field_posed)[at]),
                    float((*field_posed)[at + size_t(COLOR_CONFORMER_LANES)]),
                    float((
                        *field_posed)[at + 2 * size_t(COLOR_CONFORMER_LANES)])};
                if (!bundle->can_sample_target_type(tb, point)) {
                    in_domain = false;
                    break;
                }
            }
#endif
            if (!in_domain) {
                if (field_stats) {
                    field_stats->analytic_fallback_samples += target_count;
                    field_stats->out_of_grid_fallback_samples += target_count;
                    field_stats->analytic_fallback_terms +=
                        query_terms[tb][lane] * target_count;
                }
                continue;
            }

            const auto field_started =
                phase_times && phase_times->field_sample_seconds
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
            for (int j = 0; j < batch.library_count[tb][lane]; ++j) {
                const size_t at =
                    batch.library_coordinate_offset[tb] +
                    size_t(j) * 3 * size_t(COLOR_CONFORMER_LANES) +
                    size_t(lane);
                const float point[3] = {
                    float((*field_posed)[at]),
                    float((*field_posed)[at + size_t(COLOR_CONFORMER_LANES)]),
                    float((
                        *field_posed)[at + 2 * size_t(COLOR_CONFORMER_LANES)])};
                const ColorFieldSample sample =
#if R3_COLOR_FIELD_BATCH
                    [&]() {
                        ColorFieldSample s;
                        s.value = double(fw.channels[size_t(j)][0]);
                        for (int k = 0; k < 3; ++k)
                            s.spatial_gradient[k] =
                                double(fw.channels[size_t(j)][size_t(k) + 1] *
                                       COORD_SCALE);
                        s.used_field = true;
                        return s;
                    }();
#else
                    bundle->sample_target_type(tb, point);
#endif
                if (!sample.used_field)
                    throw std::logic_error(
                        "eligible channel field sample left its fixed grid");
                output[lane].value += sample.value;
                // The field returns the physical spatial gradient g = dF/db;
                // map it to the same descent slots as `exact_lane`: rotation
                // -2 (r x g) with r in physical units, translation -g/s.
                if (gradient) {
                    const double gx = sample.spatial_gradient[0];
                    const double gy = sample.spatial_gradient[1];
                    const double gz = sample.spatial_gradient[2];
                    const double t_to_physical =
                        reduced_frame ? 1.0 / double(COORD_SCALE) : 1.0;
                    const double rx = (*field_posed)[at] -
                                      pose_translation[lane][0] * t_to_physical;
                    const double ry =
                        (*field_posed)[at + size_t(COLOR_CONFORMER_LANES)] -
                        pose_translation[lane][1] * t_to_physical;
                    const double rz =
                        (*field_posed)[at + 2 * size_t(COLOR_CONFORMER_LANES)] -
                        pose_translation[lane][2] * t_to_physical;
                    output[lane].descent[1] += -2.0 * (ry * gz - rz * gy);
                    output[lane].descent[2] += -2.0 * (rz * gx - rx * gz);
                    output[lane].descent[3] += -2.0 * (rx * gy - ry * gx);
                    output[lane].descent[4] -= gx / double(COORD_SCALE);
                    output[lane].descent[5] -= gy / double(COORD_SCALE);
                    output[lane].descent[6] -= gz / double(COORD_SCALE);
                }
            }
            use_field[tb][lane] = true;
            const uint64_t plane_terms = query_terms[tb][lane] * target_count;
            term_counts[lane] += plane_terms;
            if (census)
                for (int ta = ta_begin; ta < ta_end; ++ta) {
                    if (interaction.coefficient(ta, tb) == 0.0 ||
                        !interaction_active(interaction, ta, tb, plan,
                                            reduced_frame &&
                                                use_invariant_plan))
                        continue;
                    const size_t cell =
                        size_t(ta) * size_t(MAX_COLOR_FEATURES) + size_t(tb);
                    census->useful_terms_by_interaction[cell] +=
                        uint64_t(batch.query_count[ta][lane]) * target_count;
                }
            if (field_stats) {
                field_stats->interpolated_samples += target_count;
                field_stats->interpolated_terms += plane_terms;
            }
            if (phase_times && phase_times->field_sample_seconds)
                *phase_times->field_sample_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - field_started)
                        .count();
        }
    }
    if (census) {
        for (int tb = 1; tb < batch.n_features; ++tb) {
            for (int lane = 0; lane < batch.n_valid; ++lane) {
                const int bin = std::min(batch.library_count[tb][lane], 18);
                ++census->target_count_bins[size_t(tb) * 19 + size_t(bin)];
            }
        }
    }
    // AVX2 analytic loop for every (plane, lane) not taken by a field.
    bool any_simd_lane = false;
    for (int lane = 0; lane < batch.n_valid; ++lane)
        any_simd_lane |= simd_mask[lane] != 0;
    if (!use_vector || !any_simd_lane) return;

    auto& posed_f32 = scratch.posed_library_xyz_fp32;

#if defined(__AVX2__)
    if (phase_times && phase_times->vector_kernel_seconds)
        started = std::chrono::steady_clock::now();
    alignas(32) float translation_x[8]{}, translation_y[8]{},
        translation_z[8]{};
    for (int lane = 0; lane < batch.n_valid; ++lane) {
        if (!simd_mask[lane]) continue;
        translation_x[lane] = float(pose_translation[lane][0]);
        translation_y[lane] = float(pose_translation[lane][1]);
        translation_z[lane] = float(pose_translation[lane][2]);
    }
    const __m256 tx = _mm256_load_ps(translation_x);
    const __m256 ty = _mm256_load_ps(translation_y);
    const __m256 tz = _mm256_load_ps(translation_z);
    // Match the selected width kernel's target-plane order and precision:
    // each conformer lane accumulates fp32 terms within one target plane,
    // then its plane subtotal is widened and added to its fp64 result.
    bool any_analytic_work = false;
    for (int tb = 1; tb < batch.n_features; ++tb) {
        int32_t analytic_mask[COLOR_CONFORMER_LANES]{};
        bool any_analytic_lane = false;
        for (int lane = 0; lane < batch.n_valid; ++lane) {
            analytic_mask[lane] =
                simd_mask[lane] && !use_field[tb][lane] ? -1 : 0;
            any_analytic_lane |= analytic_mask[lane] != 0;
        }
        if (!any_analytic_lane) continue;
        any_analytic_work = true;
        __m256 plane_value = _mm256_setzero_ps();
        __m256 plane_gradient[6];
        for (auto& component : plane_gradient)
            component = _mm256_setzero_ps();
        const int ta_begin = interaction.canonical ? tb : 1;
        const int ta_end = interaction.canonical ? tb + 1 : batch.n_features;
        for (int ta = ta_begin; ta < ta_end; ++ta) {
            if (!interaction_active(interaction, ta, tb, plan,
                                    reduced_frame && use_invariant_plan))
                continue;
            const double decay = interaction.decay(ta, tb);
            const float coefficient = float(interaction.coefficient(ta, tb));
            const float width = float(-2.0 * decay);
            // Per-cell constants: exp argument = decay |d|^2; the translation
            // descent drops the 1/COORD_SCALE on the reduced route, as in
            // `exact_lane`.
            const float exponent_scale = -0.5f * width;
            const float translation_scale =
                reduced_frame ? -width : -width / COORD_SCALE;

            // Loop bounds are the maxima over analytic lanes; the census counts
            // issued lanes = max_q * max_b * 8 against the useful sum.
            int max_query = 0, max_library = 0;
            uint64_t useful = 0;
            for (int lane = 0; lane < batch.n_valid; ++lane) {
                if (!analytic_mask[lane]) continue;
                const int nq = batch.query_count[ta][lane];
                const int nb = batch.library_count[tb][lane];
                max_query = std::max(max_query, nq);
                max_library = std::max(max_library, nb);
                useful += uint64_t(nq) * uint64_t(nb);
                term_counts[lane] += uint64_t(nq) * uint64_t(nb);
            }
            if (census) {
                const size_t cell =
                    size_t(ta) * size_t(MAX_COLOR_FEATURES) + size_t(tb);
                census->useful_terms_by_interaction[cell] += useful;
                census->issued_lanes_by_interaction[cell] +=
                    uint64_t(max_query) * uint64_t(max_library) *
                    uint64_t(COLOR_CONFORMER_LANES);
            }

            // Dispatch once per plane, above both pair loops. A concrete
            // callable lets the canonical helper share the same seam
            // without an indirect call or mode branch for each Gaussian.
            const auto accumulate_plane = [&](auto exponential) {
                if (factor_canonical_torque && reduced_frame &&
                    interaction.canonical && gradient) {
                    accumulate_canonical_factored_torque(
                        batch, tb, posed_f32.data(), analytic_mask, max_query,
                        max_library, coefficient, exponent_scale,
                        translation_scale, tx, ty, tz, plane_value,
                        plane_gradient, exponential);
                    return;
                }
                // Query feature i outer, library feature j inner. A term is
                // live only where both masks and the analytic lane mask are
                // set; masked coordinates are zeroed before use and the value
                // is ANDed with `active`, so padding adds +0.
                for (int i = 0; i < max_query; ++i) {
                    const size_t qx_at = coordinate_index(
                        batch.query_coordinate_offset[ta], i, 0, 0);
                    const size_t qy_at = coordinate_index(
                        batch.query_coordinate_offset[ta], i, 1, 0);
                    const size_t qz_at = coordinate_index(
                        batch.query_coordinate_offset[ta], i, 2, 0);
                    const __m256i query_mask_i =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                            batch.query_mask.data() +
                            mask_index(batch.query_mask_offset[ta], i, 0)));
                    const __m256 query_enabled =
                        _mm256_castsi256_ps(_mm256_and_si256(
                            query_mask_i,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                analytic_mask))));
                    const __m256 qx = _mm256_and_ps(
                        query_enabled,
                        _mm256_loadu_ps(batch.query_xyz.data() + qx_at));
                    const __m256 qy = _mm256_and_ps(
                        query_enabled,
                        _mm256_loadu_ps(batch.query_xyz.data() + qy_at));
                    const __m256 qz = _mm256_and_ps(
                        query_enabled,
                        _mm256_loadu_ps(batch.query_xyz.data() + qz_at));
                    for (int j = 0; j < max_library; ++j) {
                        const size_t bx_at = coordinate_index(
                            batch.library_coordinate_offset[tb], j, 0, 0);
                        const size_t by_at = coordinate_index(
                            batch.library_coordinate_offset[tb], j, 1, 0);
                        const size_t bz_at = coordinate_index(
                            batch.library_coordinate_offset[tb], j, 2, 0);
                        const __m256i library_mask_j =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                batch.library_mask.data() +
                                mask_index(batch.library_mask_offset[tb], j,
                                           0)));
                        const __m256 library_enabled =
                            _mm256_castsi256_ps(_mm256_and_si256(
                                library_mask_j,
                                _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i*>(
                                        analytic_mask))));
                        const __m256 active =
                            _mm256_and_ps(query_enabled, library_enabled);
                        const __m256 bx = _mm256_and_ps(
                            library_enabled,
                            _mm256_loadu_ps(posed_f32.data() + bx_at));
                        const __m256 by = _mm256_and_ps(
                            library_enabled,
                            _mm256_loadu_ps(posed_f32.data() + by_at));
                        const __m256 bz = _mm256_and_ps(
                            library_enabled,
                            _mm256_loadu_ps(posed_f32.data() + bz_at));
                        const __m256 dx = _mm256_sub_ps(qx, bx);
                        const __m256 dy = _mm256_sub_ps(qy, by);
                        const __m256 dz = _mm256_sub_ps(qz, bz);
                        // |d|^2 as dz*dz, then FMA y, then FMA x.
                        const __m256 d2 = _mm256_fmadd_ps(
                            dx, dx,
                            _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)));
                        const __m256 exponent =
                            _mm256_mul_ps(d2, _mm256_set1_ps(exponent_scale));
                        const __m256 value = _mm256_and_ps(
                            _mm256_mul_ps(exponential(exponent),
                                          _mm256_set1_ps(coefficient)),
                            active);
                        plane_value = _mm256_add_ps(plane_value, value);
                        if (gradient) {
                            const __m256 rx = _mm256_sub_ps(bx, tx);
                            const __m256 ry = _mm256_sub_ps(by, ty);
                            const __m256 rz = _mm256_sub_ps(bz, tz);
                            const __m256 torque = _mm256_mul_ps(
                                value, _mm256_set1_ps(-2.0f * width));
                            plane_gradient[0] = _mm256_add_ps(
                                plane_gradient[0],
                                _mm256_mul_ps(
                                    torque,
                                    _mm256_sub_ps(_mm256_mul_ps(ry, dz),
                                                  _mm256_mul_ps(rz, dy))));
                            plane_gradient[1] = _mm256_add_ps(
                                plane_gradient[1],
                                _mm256_mul_ps(
                                    torque,
                                    _mm256_sub_ps(_mm256_mul_ps(rz, dx),
                                                  _mm256_mul_ps(rx, dz))));
                            plane_gradient[2] = _mm256_add_ps(
                                plane_gradient[2],
                                _mm256_mul_ps(
                                    torque,
                                    _mm256_sub_ps(_mm256_mul_ps(rx, dy),
                                                  _mm256_mul_ps(ry, dx))));
                            const __m256 translate = _mm256_mul_ps(
                                value, _mm256_set1_ps(translation_scale));
                            plane_gradient[3] =
                                _mm256_add_ps(plane_gradient[3],
                                              _mm256_mul_ps(translate, dx));
                            plane_gradient[4] =
                                _mm256_add_ps(plane_gradient[4],
                                              _mm256_mul_ps(translate, dy));
                            plane_gradient[5] =
                                _mm256_add_ps(plane_gradient[5],
                                              _mm256_mul_ps(translate, dz));
                        }
                    }
                }
            };
            // The approximate raw-bit exponentials apply only to
            // canonical reduced-frame gradient calls; value-only scoring and
            // general matrices keep the bounded polynomial.
            if (optimizer_exp_mode == OptimizerExpMode::RawBitsUnclamped &&
                reduced_frame && interaction.canonical && gradient)
                accumulate_plane(RawBitsUnclampedOptimizerExp{});
            else if (optimizer_exp_mode == OptimizerExpMode::RawBits &&
                     reduced_frame && interaction.canonical && gradient)
                accumulate_plane(RawBitsOptimizerExp{});
            else
                accumulate_plane(
                    [](__m256 x) { return exp8_polynomial_bounded(x); });
        }
        // Widen each lane's fp32 plane subtotal into its fp64 result.
        alignas(32) float values[8];
        _mm256_store_ps(values, plane_value);
        for (int lane = 0; lane < batch.n_valid; ++lane)
            if (simd_mask[lane]) output[lane].value += double(values[lane]);
        if (gradient) {
            for (int component = 0; component < 6; ++component) {
                _mm256_store_ps(values, plane_gradient[component]);
                for (int lane = 0; lane < batch.n_valid; ++lane)
                    if (simd_mask[lane])
                        output[lane].descent[component + 1] +=
                            double(values[lane]);
            }
        }
    }
    if (phase_times && phase_times->vector_kernel_seconds)
        *phase_times->vector_kernel_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started)
                .count();
#else
    for (int lane = 0; lane < batch.n_valid; ++lane) {
        if (!simd_mask[lane]) continue;
        output[lane] =
            exact_lane(batch, lane, interaction, posed.data(), pose_translation,
                       gradient, &term_counts[lane], reduced_frame, plan,
                       reduced_frame && use_invariant_plan);
        if (census) ++census->scalar_fallback_pairs;
    }
#endif
#if defined(__AVX2__)
    if (census && any_analytic_work) ++census->vector_evaluations;
#endif
}

// Legacy physical-frame entry point (quaternion + physical translation).
void evaluate_color_conformer_batch8(
    const ColorConformerBatch8& batch,
    const ColorInteractionMatrixView& interaction,
    const float quaternion[COLOR_CONFORMER_LANES][4],
    const float translation_physical[COLOR_CONFORMER_LANES][3], bool gradient,
    ColorEvaluation output[COLOR_CONFORMER_LANES],
    uint64_t term_counts[COLOR_CONFORMER_LANES], ColorConformerCensus* census,
    ColorConformerPhaseTimes* phase_times, ColorConformerWorkspace* workspace,
    const ColorFieldBundle* const* field_bundles,
    ColorFieldBatchStats* field_stats, bool library_is_preposed) {
    evaluate_color_conformer_batch8_impl(
        batch, interaction, quaternion, translation_physical, gradient, output,
        term_counts, census, phase_times, workspace, field_bundles, field_stats,
        library_is_preposed, nullptr, nullptr, false, false,
        OptimizerExpMode::Baseline, false);
}

// Shared-pose entry point: the plan's reduced-frame matrix and each lane's
// exact FP32 PreparedReducedPose.
void evaluate_color_conformer_batch8_prepared(
    const ColorConformerBatch8& batch, const ColorInvariantPlan& plan,
    const PreparedReducedPose* const poses[COLOR_CONFORMER_LANES],
    bool gradient, ColorEvaluation output[COLOR_CONFORMER_LANES],
    uint64_t term_counts[COLOR_CONFORMER_LANES], ColorConformerCensus* census,
    ColorConformerPhaseTimes* phase_times, ColorConformerWorkspace* workspace,
    bool use_invariant_plan, const ColorFieldBundle* const* field_bundles,
    ColorFieldBatchStats* field_stats, bool factor_canonical_torque,
    OptimizerExpMode optimizer_exp_mode, bool canonical_color_kernel) {
    const auto reduced = plan.reduced_matrix_view();
    evaluate_color_conformer_batch8_impl(
        batch, reduced.evaluator_view(), nullptr, nullptr, gradient, output,
        term_counts, census, phase_times, workspace, field_bundles, field_stats,
        false, poses, &plan, use_invariant_plan, factor_canonical_torque,
        optimizer_exp_mode, canonical_color_kernel);
}

}  // namespace cpu
}  // namespace r3

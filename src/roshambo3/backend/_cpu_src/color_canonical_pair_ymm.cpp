// Shared-count YMM8 analytic color for two compatible conformers.
//
// Lane layout: one 256-bit vector = two library conformers x four starts.
//   lanes 0..3  descriptors[0] ("low"),  starts 0..3   -> output[0..3]
//   lanes 4..7  descriptors[1] ("high"), starts 0..3   -> output[4..7]
// The two conformers have identical count/offset signatures
// (color_canonical_pair_compatible), so one set of loop bounds, one query
// broadcast and one set of plane constants serve all eight lanes with no
// count masks. Besides the eight poses, only the raw library coordinates
// differ: each atom's low and high coordinates are broadcast into their own
// 128-bit half. Posed scratch is [x|y|z][atom][8 lanes] per plane, at twice
// the singleton offsets.
// No reduction ever crosses lanes, so each start is independent.
//
// Per lane the arithmetic is the XMM4 helper's (same exp policy, term order,
// factoring and normalization); frozen all20 outputs/starts are
// exact with YMM8 pairs plus XMM4 singletons. Measured complete-screen gains
// include +20.59% on the three-query 1P primary and +14.52% on all eight
// 1P; frozen natural pair coverage was 98.677%.
//
// Return value: 0xff when all eight lanes were written; 0 when nothing was
// written, after which the caller retries each conformer through XMM4.
#include "color_canonical_four_start.hpp"
#include "color_conformer_kernel.hpp"

#include <cmath>

#include "kernel.hpp"
#include "optimizer_exp.hpp"
#include "color_default_parameters.hpp"
#include "reduced_pose.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace r3::cpu {
namespace {

constexpr double kMaximumCoordinate = 1.0e9;

float maximum_reduced_coordinate_fp32() {
    static const float value =
        std::nextafter(float(kMaximumCoordinate * double(COORD_SCALE)), 0.0f);
    return value;
}

bool safe_coordinate(float value, float maximum) {
    return std::isfinite(value) && std::abs(value) <= maximum;
}

#if defined(__AVX2__)
// 1 (default): validate transformed coordinates with vector compares.
// 0: the earlier store-and-scalar-check control, kept for ablation.
#ifndef R3_COLOR_VECTOR_CHECKS
#define R3_COLOR_VECTOR_CHECKS 1
#endif

// True only if every lane of x, y and z satisfies |v| <= maximum. The
// ordered (_OQ) compare is false for NaN, and +-inf exceeds any finite
// maximum. Both conformers must pass before anything is published.
bool safe_transformed_coordinates(__m256 x, __m256 y, __m256 z, float maximum) {
    const __m256 bound = _mm256_set1_ps(maximum);
    const __m256 sign_clear =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 okx =
        _mm256_cmp_ps(_mm256_and_ps(x, sign_clear), bound, _CMP_LE_OQ);
    const __m256 oky =
        _mm256_cmp_ps(_mm256_and_ps(y, sign_clear), bound, _CMP_LE_OQ);
    const __m256 okz =
        _mm256_cmp_ps(_mm256_and_ps(z, sign_clear), bound, _CMP_LE_OQ);
    return _mm256_movemask_ps(_mm256_and_ps(_mm256_and_ps(okx, oky), okz)) ==
           0xff;
}

// Baseline exp(x), identical to exp4_polynomial in the XMM4 helper: degree-7
// Taylor series after range reduction, 2^n in the exponent field, x < -80
// returns 0. Used for Baseline mode and every gradient=false call.
__m256 exp8_polynomial(__m256 x) {
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

// Raw-bit exp, identical to exp4_rawbits: x * 2^23/ln2, truncate,
// integer bias, reinterpret; tail lanes zeroed bitwise.
__m256 exp8_rawbits(__m256 x, bool unclamped) {
    const __m256 cutoff = _mm256_set1_ps(-80.0f);
    __m256 kept = _mm256_cmp_ps(x, cutoff, _CMP_GE_OQ);
    if (!unclamped) {
        const __m256 zero = _mm256_setzero_ps();
        kept = _mm256_and_ps(kept, _mm256_cmp_ps(x, zero, _CMP_LE_OQ));
        x = _mm256_min_ps(_mm256_max_ps(x, cutoff), zero);
    }
    const __m256 scaled = _mm256_mul_ps(x, _mm256_set1_ps(12102203.0f));
    const __m256i bits = _mm256_add_epi32(
        _mm256_cvttps_epi32(scaled), _mm256_set1_epi32((127 << 23) - 366393));
    return _mm256_and_ps(_mm256_castsi256_ps(bits), kept);
}


#if R3_COLOR_PARAM_FUSED
// One fused exponent_scale * 12102203 multiply on d^2 (default
// parameters only). See exp4_fused_rawbits.
__m256 exp8_fused_rawbits(__m256 d2, bool unclamped) {
    // Original rounded exponent still decides endpoint/cutoff eligibility.
    const __m256 x = _mm256_mul_ps(
        d2, _mm256_set1_ps(DefaultColorParameters::exponent_scale));
    __m256 kept = _mm256_cmp_ps(x, _mm256_set1_ps(-80.0f), _CMP_GE_OQ);
    __m256 scaled = _mm256_mul_ps(
        d2, _mm256_set1_ps(DefaultColorParameters::fused_rawbits_scale));
    if (!unclamped) {
        kept = _mm256_and_ps(kept,
                             _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ));
        scaled = _mm256_min_ps(
            _mm256_max_ps(
                scaled,
                _mm256_set1_ps(-80.0f * DefaultColorParameters::rawbits_scale)),
            _mm256_setzero_ps());
    }
    const __m256i bits = _mm256_add_epi32(
        _mm256_cvttps_epi32(scaled), _mm256_set1_epi32((127 << 23) - 366393));
    return _mm256_and_ps(_mm256_castsi256_ps(bits), kept);
}
#endif

// Instruction-free register barrier that keeps value = exp * coefficient
// rounded to FP32 before it is summed when ValuesOnly removes its other
// (force) uses; without it the compiler fused across that rounding boundary
// and value-only SEARCH stopped matching full SEARCH. See the XMM4 copy.
template <bool ValuesOnly>
inline __m256 search_value_rounding_boundary(__m256 value) {
    if constexpr (ValuesOnly) asm("" : "+x"(value));
    return value;
}

// Fixed / Normalize / ValuesOnly have the meanings documented on
// evaluate_xmm4_impl. Loop bounds, counts, plane
// constants and query pointers come from descriptors[0]; from
// descriptors[1] the kernel reads only its safety flag, envelope maxima and
// library coordinates/stride.
template <bool Fixed, bool Normalize, bool ValuesOnly, typename Exponential>
uint8_t evaluate_ymm8_impl(
    const ColorCanonicalFourStartDescriptor descriptors[2], bool factor_torque,
    const PreparedReducedPose* const poses[8], bool gradient,
    ColorEvaluation output[8], uint64_t term_counts[8],
    ColorCanonicalFourStartWorkspace& workspace, Exponential exponential) {
    const auto& descriptor = descriptors[0];
    if (!descriptor.immutable_inputs_safe ||
        !descriptors[1].immutable_inputs_safe)
        return 0;
    const float maximum_reduced = maximum_reduced_coordinate_fp32();
    // [element][lane]: poses[0..3] are the low conformer's starts and
    // poses[4..7] the high conformer's, matching the lane layout above.
    float matrix[9][8]{};
    float translation[3][8]{};
    for (int lane = 0; lane < 8; ++lane) {
        if (poses[lane] == nullptr) return 0;
        for (int element = 0; element < 9; ++element) {
            matrix[element][lane] = poses[lane]->matrix[element];
        }
        for (int component = 0; component < 3; ++component) {
            translation[component][lane] = poses[lane]->translation[component];
            if (!safe_coordinate(translation[component][lane], maximum_reduced))
                return 0;
        }
    }

    const __m256 m00 = _mm256_loadu_ps(matrix[0]);
    const __m256 m01 = _mm256_loadu_ps(matrix[1]);
    const __m256 m02 = _mm256_loadu_ps(matrix[2]);
    const __m256 m10 = _mm256_loadu_ps(matrix[3]);
    const __m256 m11 = _mm256_loadu_ps(matrix[4]);
    const __m256 m12 = _mm256_loadu_ps(matrix[5]);
    const __m256 m20 = _mm256_loadu_ps(matrix[6]);
    const __m256 m21 = _mm256_loadu_ps(matrix[7]);
    const __m256 m22 = _mm256_loadu_ps(matrix[8]);
    const __m256 tx = _mm256_loadu_ps(translation[0]);
    const __m256 ty = _mm256_loadu_ps(translation[1]);
    const __m256 tz = _mm256_loadu_ps(translation[2]);
#if R3_COLOR_ANALYTIC_ENVELOPE
    // Envelope proof as in XMM4, with each half using its own
    // conformer's library maxima. Proof in doc/CPU_SHARED_POSE_CONTRACT.md.
    const __m256 ax = _mm256_setr_ps(
        descriptor.library_max_abs[0], descriptor.library_max_abs[0],
        descriptor.library_max_abs[0], descriptor.library_max_abs[0],
        descriptors[1].library_max_abs[0], descriptors[1].library_max_abs[0],
        descriptors[1].library_max_abs[0], descriptors[1].library_max_abs[0]);
    const __m256 ay = _mm256_setr_ps(
        descriptor.library_max_abs[1], descriptor.library_max_abs[1],
        descriptor.library_max_abs[1], descriptor.library_max_abs[1],
        descriptors[1].library_max_abs[1], descriptors[1].library_max_abs[1],
        descriptors[1].library_max_abs[1], descriptors[1].library_max_abs[1]);
    const __m256 az = _mm256_setr_ps(
        descriptor.library_max_abs[2], descriptor.library_max_abs[2],
        descriptor.library_max_abs[2], descriptor.library_max_abs[2],
        descriptors[1].library_max_abs[2], descriptors[1].library_max_abs[2],
        descriptors[1].library_max_abs[2], descriptors[1].library_max_abs[2]);
    const __m256 sign_clear =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    auto envelope = [&](__m256 a, __m256 b, __m256 c, __m256 t) {
        __m256 bound = _mm256_mul_ps(_mm256_and_ps(a, sign_clear), ax);
        bound = _mm256_add_ps(bound,
                              _mm256_mul_ps(_mm256_and_ps(b, sign_clear), ay));
        bound = _mm256_add_ps(bound,
                              _mm256_mul_ps(_mm256_and_ps(c, sign_clear), az));
        bound = _mm256_add_ps(bound, _mm256_and_ps(t, sign_clear));
        return _mm256_add_ps(
            _mm256_mul_ps(
                bound,
                _mm256_set1_ps(1.0f +
                               16.0f * std::numeric_limits<float>::epsilon())),
            _mm256_set1_ps(1.0f));
    };
    const bool envelope_safe = safe_transformed_coordinates(
        envelope(m00, m01, m02, tx), envelope(m10, m11, m12, ty),
        envelope(m20, m21, m22, tz), maximum_reduced);
#endif
    for (int plane_index = 0; plane_index < descriptor.plane_count;
         ++plane_index) {
        const ColorCanonicalFourStartPlane& plane =
            descriptor.planes[size_t(plane_index)];
        const int nb = plane.library_count;
        const float* raw = plane.library_xyz;
        // Pass 1: pose and validate every atom for all eight lanes before
        // any output is written. Plane index i is the same type in both
        // descriptors (guaranteed by the pair signature).
        const auto& high_plane = descriptors[1].planes[size_t(plane_index)];
        const float* high_raw = high_plane.library_xyz;
        float* posed_x =
            workspace.posed_library_xyz.data() + plane.posed_offset * 2;
        float* posed_y = posed_x + size_t(nb) * 8;
        float* posed_z = posed_y + size_t(nb) * 8;
        for (int atom = 0; atom < plane.library_count; ++atom) {
            // Low conformer's atom in lanes 0..3, high conformer's in 4..7.
            const __m256 x =
                _mm256_setr_ps(raw[0], raw[0], raw[0], raw[0], high_raw[0],
                               high_raw[0], high_raw[0], high_raw[0]);
            const __m256 y =
                _mm256_setr_ps(raw[1], raw[1], raw[1], raw[1], high_raw[1],
                               high_raw[1], high_raw[1], high_raw[1]);
            const __m256 z =
                _mm256_setr_ps(raw[2], raw[2], raw[2], raw[2], high_raw[2],
                               high_raw[2], high_raw[2], high_raw[2]);
            __m256 bx = _mm256_mul_ps(m00, x);
            bx = _mm256_fmadd_ps(m01, y, bx);
            bx = _mm256_fmadd_ps(m02, z, bx);
            bx = _mm256_add_ps(bx, tx);
            __m256 by = _mm256_mul_ps(m10, x);
            by = _mm256_fmadd_ps(m11, y, by);
            by = _mm256_fmadd_ps(m12, z, by);
            by = _mm256_add_ps(by, ty);
            __m256 bz = _mm256_mul_ps(m20, x);
            bz = _mm256_fmadd_ps(m21, y, bz);
            bz = _mm256_fmadd_ps(m22, z, bz);
            bz = _mm256_add_ps(bz, tz);
            _mm256_storeu_ps(posed_x, bx);
            _mm256_storeu_ps(posed_y, by);
            _mm256_storeu_ps(posed_z, bz);
#if R3_COLOR_VECTOR_CHECKS
#if R3_COLOR_ANALYTIC_ENVELOPE
            if (!envelope_safe &&
                !safe_transformed_coordinates(bx, by, bz, maximum_reduced))
                return 0;
#else
            if (!safe_transformed_coordinates(bx, by, bz, maximum_reduced))
                return 0;
#endif
#else
            alignas(32) float check[8];
            _mm256_store_ps(check, bx);
            for (float value : check)
                if (!safe_coordinate(value, maximum_reduced)) return 0;
            _mm256_store_ps(check, by);
            for (float value : check)
                if (!safe_coordinate(value, maximum_reduced)) return 0;
            _mm256_store_ps(check, bz);
            for (float value : check)
                if (!safe_coordinate(value, maximum_reduced)) return 0;
#endif
            raw += plane.library_stride;
            high_raw += high_plane.library_stride;
            posed_x += 8;
            posed_y += 8;
            posed_z += 8;
        }
    }

    // Pass 2: Gaussian overlap, plane by plane. total_gradient[0..2] are
    // rotational and [3..5] translational descent.
    __m256 total_value = _mm256_setzero_ps();
    __m256 total_gradient[6];
    for (auto& component : total_gradient)
        component = _mm256_setzero_ps();
    const __m256 pose_tx = _mm256_loadu_ps(translation[0]);
    const __m256 pose_ty = _mm256_loadu_ps(translation[1]);
    const __m256 pose_tz = _mm256_loadu_ps(translation[2]);
    for (int plane_index = 0; plane_index < descriptor.plane_count;
         ++plane_index) {
        const ColorCanonicalFourStartPlane& plane =
            descriptor.planes[size_t(plane_index)];
        const int nb = plane.library_count;
        __m256 plane_value = _mm256_setzero_ps();
        __m256 plane_gradient[6];
        for (auto& component : plane_gradient)
            component = _mm256_setzero_ps();
        const __m256 coefficient =
            _mm256_set1_ps(Normalize ? 1.0f
                           : Fixed   ? DefaultColorParameters::coefficient
                                     : plane.coefficient);
        const __m256 exponent_scale =
            _mm256_set1_ps(Fixed ? DefaultColorParameters::exponent_scale
                                 : plane.exponent_scale);
        const __m256 torque_scale = _mm256_set1_ps(
            Fixed ? DefaultColorParameters::torque_scale : plane.torque_scale);
        const __m256 translation_scale =
            _mm256_set1_ps(Fixed ? DefaultColorParameters::translation_scale
                                 : plane.translation_scale);
        const float* posed_x =
            workspace.posed_library_xyz.data() + plane.posed_offset * 2;
        const float* posed_y = posed_x + size_t(nb) * 8;
        const float* posed_z = posed_y + size_t(nb) * 8;

        // Field-backed descriptors never pair, so this path is analytic only.
        bool sampled_field = false;

        if (!sampled_field && factor_torque && gradient) {
            // Match the existing factored path: accumulate force over each
            // target plane, then form the pose torque once per target atom.
            // Library-outer, query-inner; torque += 2 r_j x f_j.
            const float* library_x = posed_x;
            const float* library_y = posed_y;
            const float* library_z = posed_z;
            for (int j = 0; j < nb; ++j) {
                const __m256 bx = _mm256_loadu_ps(library_x);
                const __m256 by = _mm256_loadu_ps(library_y);
                const __m256 bz = _mm256_loadu_ps(library_z);
                __m256 fx = _mm256_setzero_ps();
                __m256 fy = _mm256_setzero_ps();
                __m256 fz = _mm256_setzero_ps();
                const float* query = plane.query_xyz;
                for (int i = 0; i < plane.query_count; ++i) {
                    const __m256 dx =
                        _mm256_sub_ps(_mm256_set1_ps(query[0]), bx);
                    const __m256 dy =
                        _mm256_sub_ps(_mm256_set1_ps(query[1]), by);
                    const __m256 dz =
                        _mm256_sub_ps(_mm256_set1_ps(query[2]), bz);
                    const __m256 d2 = _mm256_fmadd_ps(
                        dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)));
                    const __m256 value = _mm256_mul_ps(
                        exponential(d2, exponent_scale), coefficient);
                    plane_value = _mm256_add_ps(
                        plane_value,
                        search_value_rounding_boundary<ValuesOnly>(value));
                    const __m256 force =
                        _mm256_mul_ps(value, translation_scale);
                    fx = _mm256_add_ps(fx, _mm256_mul_ps(force, dx));
                    fy = _mm256_add_ps(fy, _mm256_mul_ps(force, dy));
                    fz = _mm256_add_ps(fz, _mm256_mul_ps(force, dz));
                    query += plane.query_stride;
                }
                const __m256 rx = _mm256_sub_ps(bx, pose_tx);
                const __m256 ry = _mm256_sub_ps(by, pose_ty);
                const __m256 rz = _mm256_sub_ps(bz, pose_tz);
                const __m256 two = _mm256_set1_ps(2.0f);
                plane_gradient[0] = _mm256_add_ps(
                    plane_gradient[0],
                    _mm256_mul_ps(two, _mm256_sub_ps(_mm256_mul_ps(ry, fz),
                                                     _mm256_mul_ps(rz, fy))));
                plane_gradient[1] = _mm256_add_ps(
                    plane_gradient[1],
                    _mm256_mul_ps(two, _mm256_sub_ps(_mm256_mul_ps(rz, fx),
                                                     _mm256_mul_ps(rx, fz))));
                plane_gradient[2] = _mm256_add_ps(
                    plane_gradient[2],
                    _mm256_mul_ps(two, _mm256_sub_ps(_mm256_mul_ps(rx, fy),
                                                     _mm256_mul_ps(ry, fx))));
                plane_gradient[3] = _mm256_add_ps(plane_gradient[3], fx);
                plane_gradient[4] = _mm256_add_ps(plane_gradient[4], fy);
                plane_gradient[5] = _mm256_add_ps(plane_gradient[5], fz);
                library_x += 8;
                library_y += 8;
                library_z += 8;
            }
        } else if (!sampled_field) {
            // Keep the x86 un-factored canonical term order: query i, library j.
            // Also serves gradient=false. Value-only SEARCH comes here only
            // when factoring is off; with factoring on it takes the branch
            // above, whose dead force terms are discarded.
            const float* query = plane.query_xyz;
            for (int i = 0; i < plane.query_count; ++i) {
                const __m256 qx = _mm256_set1_ps(query[0]);
                const __m256 qy = _mm256_set1_ps(query[1]);
                const __m256 qz = _mm256_set1_ps(query[2]);
                const float* library_x = posed_x;
                const float* library_y = posed_y;
                const float* library_z = posed_z;
                for (int j = 0; j < nb; ++j) {
                    const __m256 bx = _mm256_loadu_ps(library_x);
                    const __m256 by = _mm256_loadu_ps(library_y);
                    const __m256 bz = _mm256_loadu_ps(library_z);
                    const __m256 dx = _mm256_sub_ps(qx, bx);
                    const __m256 dy = _mm256_sub_ps(qy, by);
                    const __m256 dz = _mm256_sub_ps(qz, bz);
                    const __m256 d2 = _mm256_fmadd_ps(
                        dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)));
                    const __m256 value = _mm256_mul_ps(
                        exponential(d2, exponent_scale), coefficient);
                    plane_value = _mm256_add_ps(
                        plane_value,
                        search_value_rounding_boundary<ValuesOnly>(value));
                    library_x += 8;
                    library_y += 8;
                    library_z += 8;
                    if (!gradient || ValuesOnly) continue;
                    const __m256 rx = _mm256_sub_ps(bx, pose_tx);
                    const __m256 ry = _mm256_sub_ps(by, pose_ty);
                    const __m256 rz = _mm256_sub_ps(bz, pose_tz);
                    const __m256 torque = _mm256_mul_ps(value, torque_scale);
                    plane_gradient[0] = _mm256_add_ps(
                        plane_gradient[0],
                        _mm256_mul_ps(torque,
                                      _mm256_sub_ps(_mm256_mul_ps(ry, dz),
                                                    _mm256_mul_ps(rz, dy))));
                    plane_gradient[1] = _mm256_add_ps(
                        plane_gradient[1],
                        _mm256_mul_ps(torque,
                                      _mm256_sub_ps(_mm256_mul_ps(rz, dx),
                                                    _mm256_mul_ps(rx, dz))));
                    plane_gradient[2] = _mm256_add_ps(
                        plane_gradient[2],
                        _mm256_mul_ps(torque,
                                      _mm256_sub_ps(_mm256_mul_ps(rx, dy),
                                                    _mm256_mul_ps(ry, dx))));
                    const __m256 translate =
                        _mm256_mul_ps(value, translation_scale);
                    plane_gradient[3] = _mm256_add_ps(
                        plane_gradient[3], _mm256_mul_ps(translate, dx));
                    plane_gradient[4] = _mm256_add_ps(
                        plane_gradient[4], _mm256_mul_ps(translate, dy));
                    plane_gradient[5] = _mm256_add_ps(
                        plane_gradient[5], _mm256_mul_ps(translate, dz));
                }
                query += plane.query_stride;
            }
        }
        total_value = _mm256_add_ps(total_value, plane_value);
        if (gradient && !ValuesOnly)
            for (int component = 0; component < 6; ++component)
                total_gradient[component] = _mm256_add_ps(
                    total_gradient[component], plane_gradient[component]);
        // Shared counts: both conformers issue exactly this many pair terms.
        const uint64_t count =
            uint64_t(plane.query_count) * uint64_t(plane.library_count);
        for (int lane = 0; lane < 8; ++lane)
            term_counts[lane] += count;
    }

    // One multiply by the common amplitude restores physical units.
    if constexpr (Normalize) {
        const __m256 amplitude =
            _mm256_set1_ps(DefaultColorParameters::coefficient);
        total_value = _mm256_mul_ps(total_value, amplitude);
        if (gradient && !ValuesOnly)
            for (auto& component : total_gradient)
                component = _mm256_mul_ps(component, amplitude);
    }

    alignas(32) float values[8];
    _mm256_store_ps(values, total_value);
    for (int lane = 0; lane < 8; ++lane)
        output[lane].value = double(values[lane]);
    if (gradient && !ValuesOnly)
        for (int component = 0; component < 6; ++component) {
            _mm256_store_ps(values, total_gradient[component]);
            for (int lane = 0; lane < 8; ++lane)
                output[lane].descent[component + 1] = double(values[lane]);
        }
    return 0xff;
}
#endif

#if defined(__AVX2__)
// Lifts search_output into the ValuesOnly template argument once per call.
template <bool Fixed, bool Normalize, typename Exponential>
uint8_t
evaluate_ymm8(const ColorCanonicalFourStartDescriptor descriptors[2],
              bool factor_torque, const PreparedReducedPose* const poses[8],
              bool gradient, ColorEvaluation output[8], uint64_t term_counts[8],
              ColorCanonicalFourStartWorkspace& workspace,
              Exponential exponential, ColorSearchOutput search_output) {
    if (search_output == ColorSearchOutput::ValueOnly)
        return evaluate_ymm8_impl<Fixed, Normalize, true>(
            descriptors, factor_torque, poses, gradient, output, term_counts,
            workspace, exponential);
    return evaluate_ymm8_impl<Fixed, Normalize, false>(
        descriptors, factor_torque, poses, gradient, output, term_counts,
        workspace, exponential);
}
#endif
}  // namespace

uint8_t evaluate_color_canonical_four_start_ymm8(
    const ColorCanonicalFourStartDescriptor descriptors[2], bool factor_torque,
    const PreparedReducedPose* const poses[8], bool gradient,
    ColorEvaluation output[8], uint64_t term_counts[8],
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode, ColorSearchOutput search_output) {
#if defined(__AVX2__)
    // Both halves take one policy. The pair signature already requires equal
    // default_parameters and amplitude_amortized; the conjunction is kept.
    const bool default_parameters =
        descriptors[0].default_parameters &&
        descriptors[1].default_parameters && !descriptors[0].field_bundle &&
        !descriptors[1].field_bundle &&
        (!R3_COLOR_PARAM_NORMALIZE || (descriptors[0].amplitude_amortized &&
                                descriptors[1].amplitude_amortized));
    const auto evaluate = [&](auto exponential) {
        if ((R3_COLOR_PARAM_FIXED || R3_COLOR_PARAM_NORMALIZE) && default_parameters)
            return evaluate_ymm8<bool(R3_COLOR_PARAM_FIXED), bool(R3_COLOR_PARAM_NORMALIZE)>(
                descriptors, factor_torque, poses, gradient, output,
                term_counts, workspace, exponential, search_output);
        return evaluate_ymm8<false, false>(
            descriptors, factor_torque, poses, gradient, output, term_counts,
            workspace, exponential, search_output);
    };
    // Exponential chosen once per call; gradient=false always takes the
    // polynomial, so value-only SEARCH uses ValuesOnly instead.
#if R3_COLOR_PARAM_FUSED
    if (gradient && default_parameters &&
        optimizer_exp_mode == OptimizerExpMode::RawBitsUnclamped)
        return evaluate(
            [](__m256 d2, __m256) { return exp8_fused_rawbits(d2, true); });
    if (gradient && default_parameters &&
        optimizer_exp_mode == OptimizerExpMode::RawBits)
        return evaluate(
            [](__m256 d2, __m256) { return exp8_fused_rawbits(d2, false); });
#endif
    if (gradient && optimizer_exp_mode == OptimizerExpMode::RawBitsUnclamped)
        return evaluate([](__m256 d2, __m256 beta) {
            return exp8_rawbits(_mm256_mul_ps(d2, beta), true);
        });
    if (gradient && optimizer_exp_mode == OptimizerExpMode::RawBits)
        return evaluate([](__m256 d2, __m256 beta) {
            return exp8_rawbits(_mm256_mul_ps(d2, beta), false);
        });
    return evaluate([](__m256 d2, __m256 beta) {
        return exp8_polynomial(_mm256_mul_ps(d2, beta));
    });
#else
    (void)descriptors;
    (void)factor_torque;
    (void)poses;
    (void)gradient;
    (void)output;
    (void)term_counts;
    (void)workspace;
    (void)optimizer_exp_mode;
    return 0;
#endif
}

}  // namespace r3::cpu

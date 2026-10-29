// AVX/FMA encoded XMM4 canonical analytic color for one conformer.
//
// Lane layout: one 128-bit vector = the four starts of ONE library conformer.
// Lane k is start k; there is no cross-lane reduction, so each start keeps
// an independent value and six-component gradient. Query coordinates are
// stored once and broadcast; each raw library atom is broadcast and posed
// by four different matrices, giving posed scratch laid out as
// [x|y|z][atom][4 starts] per plane.
//
// This is the x86 singleton/tail helper. Compatible adjacent conformers use
// the eight-lane YMM8 helper instead (color_canonical_pair_ymm.cpp); a pair
// that fails there is retried here one conformer at a time. Per lane, the
// operation sequence matches YMM8's under the same policy (the pair
// signature includes default_parameters and amplitude_amortized); frozen
// all20 outputs are exact for the pair/singleton composition.
// ARM's NEON4 helper is a separate implementation with its own established
// term order; no cross-ISA bit identity is claimed.
//
// Return value: 0x0f when all four starts were written, 0 when nothing was
// written and the caller must take the FP64 fallback for the whole group.
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
// maximum, so one sign-clear and compare rejects all three hazards.
// See doc/CPU_SHARED_POSE_CONTRACT.md.
bool safe_transformed_coordinates(__m128 x, __m128 y, __m128 z, float maximum) {
    const __m128 bound = _mm_set1_ps(maximum);
    const __m128 sign_clear = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 okx = _mm_cmp_ps(_mm_and_ps(x, sign_clear), bound, _CMP_LE_OQ);
    const __m128 oky = _mm_cmp_ps(_mm_and_ps(y, sign_clear), bound, _CMP_LE_OQ);
    const __m128 okz = _mm_cmp_ps(_mm_and_ps(z, sign_clear), bound, _CMP_LE_OQ);
    return _mm_movemask_ps(_mm_and_ps(_mm_and_ps(okx, oky), okz)) == 0x0f;
}

// Baseline exp(x): range-reduce n = floor(x log2 e), evaluate the degree-7
// Taylor series in z = (x log2 e - n) ln 2 by Horner FMAs, and scale by 2^n
// built directly in the exponent field. x < -80 returns exactly 0. This is
// the selected FP32 polynomial; it serves Baseline
// mode and every gradient=false call.
__m128 exp4_polynomial(__m128 x) {
    const __m128 cutoff = _mm_set1_ps(-80.0f);
    const __m128 kept = _mm_cmpge_ps(x, cutoff);
    const __m128 bounded = _mm_max_ps(x, cutoff);
    const __m128 y = _mm_mul_ps(bounded, _mm_set1_ps(1.4426950408889634074f));
    const __m128 n = _mm_floor_ps(y);
    const __m128 z =
        _mm_mul_ps(_mm_sub_ps(y, n), _mm_set1_ps(0.69314718055994530942f));
    const __m128i exponent = _mm_cvttps_epi32(n);
    const __m128i bits =
        _mm_slli_epi32(_mm_add_epi32(exponent, _mm_set1_epi32(127)), 23);
    constexpr float coefficient[] = {
        1.0f,         1.0f,          0.5f,          1.0f / 6.0f,
        1.0f / 24.0f, 1.0f / 120.0f, 1.0f / 720.0f, 1.0f / 5040.0f};
    __m128 polynomial = _mm_set1_ps(coefficient[7]);
    for (int k = 6; k >= 0; --k)
        polynomial = _mm_fmadd_ps(polynomial, z, _mm_set1_ps(coefficient[k]));
    return _mm_and_ps(_mm_mul_ps(polynomial, _mm_castsi128_ps(bits)), kept);
}

// Raw-bit exp for nonpositive Gaussian exponents: x * 2^23/ln2
// (12102203), truncated toward zero, plus an INTEGER bias, reinterpreted as
// FP32. Lanes with x < -80 are zeroed bitwise before they reach FP arithmetic.
// Clamped (RawBits) also clamps x to [-80, 0]. Unclamped skips the clamp;
// out-of-range conversions may raise INVALID, which the caller verified is
// masked once at call entry (optimizer_exp.hpp). Forces use this approximate
// weight, not the derivative of the piecewise approximation.
__m128 exp4_rawbits(__m128 x, bool unclamped) {
    const __m128 cutoff = _mm_set1_ps(-80.0f);
    __m128 kept = _mm_cmpge_ps(x, cutoff);
    if (!unclamped) {
        const __m128 zero = _mm_setzero_ps();
        kept = _mm_and_ps(kept, _mm_cmple_ps(x, zero));
        x = _mm_min_ps(_mm_max_ps(x, cutoff), zero);
    }
    const __m128 scaled = _mm_mul_ps(x, _mm_set1_ps(12102203.0f));
    const __m128i bits = _mm_add_epi32(_mm_cvttps_epi32(scaled),
                                       _mm_set1_epi32((127 << 23) - 366393));
    return _mm_and_ps(_mm_castsi128_ps(bits), kept);
}


#if R3_COLOR_PARAM_FUSED
// Fold exponent_scale * 12102203 into one FP32 constant so the
// distance needs one multiply instead of two. The fused product rounds
// differently from the two-step chain; that was accepted under bounded
// ranking gates. Default parameters only.
__m128 exp4_fused_rawbits(__m128 d2, bool unclamped) {
    // Original rounded exponent still decides endpoint/cutoff eligibility.
    const __m128 x =
        _mm_mul_ps(d2, _mm_set1_ps(DefaultColorParameters::exponent_scale));
    __m128 kept = _mm_cmp_ps(x, _mm_set1_ps(-80.0f), _CMP_GE_OQ);
    __m128 scaled = _mm_mul_ps(
        d2, _mm_set1_ps(DefaultColorParameters::fused_rawbits_scale));
    if (!unclamped) {
        kept = _mm_and_ps(kept, _mm_cmp_ps(x, _mm_setzero_ps(), _CMP_LE_OQ));
        scaled = _mm_min_ps(
            _mm_max_ps(
                scaled,
                _mm_set1_ps(-80.0f * DefaultColorParameters::rawbits_scale)),
            _mm_setzero_ps());
    }
    const __m128i bits = _mm_add_epi32(_mm_cvttps_epi32(scaled),
                                       _mm_set1_epi32((127 << 23) - 366393));
    return _mm_and_ps(_mm_castsi128_ps(bits), kept);
}
#endif

// Full SEARCH reuses the rounded product value = exp * coefficient in the
// force terms. When ValuesOnly compiles the derivatives out, that product
// feeds only the sum, and the compiler fused across the rounding boundary:
// the first value-only attempt failed five exact SEARCH replays. The empty
// asm pins `value` in a register, preserving the FP32 rounding of the
// product, and emits no instruction. A no-op when !ValuesOnly.
template <bool ValuesOnly>
inline __m128 search_value_rounding_boundary(__m128 value) {
    if constexpr (ValuesOnly) asm("" : "+x"(value));
    return value;
}

// Template parameters:
//   Fixed      - compile the default coefficient/scales in. Off by
//                default (R3_COLOR_PARAM_FIXED=0); kept for ablation.
//   Normalize  - sum unit-amplitude terms and multiply the seven
//                totals by the common coefficient once at the end. On by
//                default for default-parameter descriptors with >= 8 issued
//                pairs (amplitude_amortized).
//   ValuesOnly - same exp, plane order and value sum as full SEARCH;
//                force/torque results are dead and gradient stores are
//                omitted. Selected by ColorSearchOutput::ValueOnly; the
//                optimizer requests it only when built with
//                R3_SEARCH_VALUE_ONLY=1 (default 0).
// `exponential(d2, exponent_scale)` returns exp(exponent_scale * d2) under
// the call's exp policy, already selected outside the loops.
template <bool Fixed, bool Normalize, bool ValuesOnly, typename Exponential>
uint8_t evaluate_xmm4_impl(const ColorCanonicalFourStartDescriptor& descriptor,
                           bool factor_torque,
                           const PreparedReducedPose* const poses[4],
                           bool gradient, ColorEvaluation output[4],
                           uint64_t term_counts[4],
                           ColorCanonicalFourStartWorkspace& workspace,
                           Exponential exponential) {
    // Fallback triggers (return 0 before any output is written): unsafe
    // immutable inputs, a missing pose, a translation or transformed
    // coordinate outside the reduced envelope.
    if (!descriptor.immutable_inputs_safe) return 0;
    const float maximum_reduced = maximum_reduced_coordinate_fp32();
    // Transpose the four poses to [element][start] so each matrix entry
    // loads as one vector with start k in lane k.
    float matrix[9][4]{};
    float translation[3][4]{};
    for (int lane = 0; lane < 4; ++lane) {
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

    const __m128 m00 = _mm_loadu_ps(matrix[0]);
    const __m128 m01 = _mm_loadu_ps(matrix[1]);
    const __m128 m02 = _mm_loadu_ps(matrix[2]);
    const __m128 m10 = _mm_loadu_ps(matrix[3]);
    const __m128 m11 = _mm_loadu_ps(matrix[4]);
    const __m128 m12 = _mm_loadu_ps(matrix[5]);
    const __m128 m20 = _mm_loadu_ps(matrix[6]);
    const __m128 m21 = _mm_loadu_ps(matrix[7]);
    const __m128 m22 = _mm_loadu_ps(matrix[8]);
    const __m128 tx = _mm_loadu_ps(translation[0]);
    const __m128 ty = _mm_loadu_ps(translation[1]);
    const __m128 tz = _mm_loadu_ps(translation[2]);
#if R3_COLOR_ANALYTIC_ENVELOPE
    // Per pose, bound every transformed coordinate by
    // |row| . library_max_abs + |t'|, inflated by (1 + 16 eps) for rounding
    // and +1 for subnormal error. If all four starts pass, the per-atom
    // checks below are skipped; otherwise they run unchanged. Proof in
    // doc/CPU_SHARED_POSE_CONTRACT.md.
    const __m128 ax = _mm_set1_ps(descriptor.library_max_abs[0]);
    const __m128 ay = _mm_set1_ps(descriptor.library_max_abs[1]);
    const __m128 az = _mm_set1_ps(descriptor.library_max_abs[2]);
    const __m128 sign_clear = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    auto envelope = [&](__m128 a, __m128 b, __m128 c, __m128 t) {
        __m128 bound = _mm_mul_ps(_mm_and_ps(a, sign_clear), ax);
        bound = _mm_add_ps(bound, _mm_mul_ps(_mm_and_ps(b, sign_clear), ay));
        bound = _mm_add_ps(bound, _mm_mul_ps(_mm_and_ps(c, sign_clear), az));
        bound = _mm_add_ps(bound, _mm_and_ps(t, sign_clear));
        return _mm_add_ps(
            _mm_mul_ps(
                bound,
                _mm_set1_ps(1.0f +
                            16.0f * std::numeric_limits<float>::epsilon())),
            _mm_set1_ps(1.0f));
    };
    const bool envelope_safe = safe_transformed_coordinates(
        envelope(m00, m01, m02, tx), envelope(m10, m11, m12, ty),
        envelope(m20, m21, m22, tz), maximum_reduced);
#endif
    // Pass 1: pose every library atom for all four starts and validate it
    // before any output or term count is written, so a late failure leaves
    // the caller's buffers untouched. b' = M raw + t' (M = sR, reduced frame).
    for (int plane_index = 0; plane_index < descriptor.plane_count;
         ++plane_index) {
        const ColorCanonicalFourStartPlane& plane =
            descriptor.planes[size_t(plane_index)];
        const int nb = plane.library_count;
        const float* raw = plane.library_xyz;
        float* posed_x =
            workspace.posed_library_xyz.data() + plane.posed_offset;
        float* posed_y = posed_x + size_t(nb) * 4;
        float* posed_z = posed_y + size_t(nb) * 4;
        for (int atom = 0; atom < plane.library_count; ++atom) {
            const __m128 x = _mm_set1_ps(raw[0]);
            const __m128 y = _mm_set1_ps(raw[1]);
            const __m128 z = _mm_set1_ps(raw[2]);
            __m128 bx = _mm_mul_ps(m00, x);
            bx = _mm_fmadd_ps(m01, y, bx);
            bx = _mm_fmadd_ps(m02, z, bx);
            bx = _mm_add_ps(bx, tx);
            __m128 by = _mm_mul_ps(m10, x);
            by = _mm_fmadd_ps(m11, y, by);
            by = _mm_fmadd_ps(m12, z, by);
            by = _mm_add_ps(by, ty);
            __m128 bz = _mm_mul_ps(m20, x);
            bz = _mm_fmadd_ps(m21, y, bz);
            bz = _mm_fmadd_ps(m22, z, bz);
            bz = _mm_add_ps(bz, tz);
            _mm_storeu_ps(posed_x, bx);
            _mm_storeu_ps(posed_y, by);
            _mm_storeu_ps(posed_z, bz);
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
            alignas(16) float check[4];
            _mm_store_ps(check, bx);
            for (float value : check)
                if (!safe_coordinate(value, maximum_reduced)) return 0;
            _mm_store_ps(check, by);
            for (float value : check)
                if (!safe_coordinate(value, maximum_reduced)) return 0;
            _mm_store_ps(check, bz);
            for (float value : check)
                if (!safe_coordinate(value, maximum_reduced)) return 0;
#endif
            raw += plane.library_stride;
            posed_x += 4;
            posed_y += 4;
            posed_z += 4;
        }
    }

    // Pass 2: Gaussian overlap. Per-plane accumulators are folded into the
    // totals in plane order. total_gradient[0..2] are rotational and [3..5]
    // translational descent (ColorEvaluation slots 1..6).
    __m128 total_value = _mm_setzero_ps();
    __m128 total_gradient[6];
    for (auto& component : total_gradient)
        component = _mm_setzero_ps();
    const __m128 pose_tx = _mm_loadu_ps(translation[0]);
    const __m128 pose_ty = _mm_loadu_ps(translation[1]);
    const __m128 pose_tz = _mm_loadu_ps(translation[2]);
    for (int plane_index = 0; plane_index < descriptor.plane_count;
         ++plane_index) {
        const ColorCanonicalFourStartPlane& plane =
            descriptor.planes[size_t(plane_index)];
        const int nb = plane.library_count;
        __m128 plane_value = _mm_setzero_ps();
        __m128 plane_gradient[6];
        for (auto& component : plane_gradient)
            component = _mm_setzero_ps();
        const __m128 coefficient =
            _mm_set1_ps(Normalize ? 1.0f
                        : Fixed   ? DefaultColorParameters::coefficient
                                  : plane.coefficient);
        const __m128 exponent_scale =
            _mm_set1_ps(Fixed ? DefaultColorParameters::exponent_scale
                              : plane.exponent_scale);
        const __m128 torque_scale = _mm_set1_ps(
            Fixed ? DefaultColorParameters::torque_scale : plane.torque_scale);
        const __m128 translation_scale =
            _mm_set1_ps(Fixed ? DefaultColorParameters::translation_scale
                              : plane.translation_scale);
        const float* posed_x =
            workspace.posed_library_xyz.data() + plane.posed_offset;
        const float* posed_y = posed_x + size_t(nb) * 4;
        const float* posed_z = posed_y + size_t(nb) * 4;

        bool sampled_field = false;
        // Experimental color-field hybrid; compiled out by default
        // (R3_COLOR_START_HYBRID=0).
#if R3_COLOR_START_HYBRID
        const ColorFieldBundle* fields =
            gradient ? descriptor.field_bundle : nullptr;
        const int backend = fields
                                ? descriptor.field_backend[size_t(plane.type)]
                                : COLOR_FIELD_ANALYTIC;
        auto* fs = descriptor.field_stats;
        const uint64_t group_samples = uint64_t(nb) * 4;
        const uint64_t group_terms =
            group_samples * uint64_t(plane.query_count);
        if (fs) {
            fs->samples += group_samples;
            if (backend == COLOR_FIELD_ANALYTIC)
                ++fs->current_four_start_remainder_groups;
            if (backend == COLOR_FIELD_ANALYTIC) {
                fs->analytic_selected_samples += group_samples;
                fs->analytic_selected_terms += group_terms;
            }
        }
        if (backend != COLOR_FIELD_ANALYTIC &&
            descriptor.field_active[size_t(plane.type)]) {
            auto& fw = workspace.field_samples;
#if !R3_COLOR_START_REDUCED_COORDS
            fw.points.resize(size_t(nb) * 12);
            for (int j = 0; j < nb; ++j)
                for (int lane = 0; lane < 4; ++lane) {
                    const size_t at = size_t(j) * 4 + size_t(lane);
                    fw.points[at * 3] =
                        float(double(posed_x[at]) / double(COORD_SCALE));
                    fw.points[at * 3 + 1] =
                        float(double(posed_y[at]) / double(COORD_SCALE));
                    fw.points[at * 3 + 2] =
                        float(double(posed_z[at]) / double(COORD_SCALE));
                }
#endif
            // Color-specific proof: every point in this plane's four starts
            // is checked against this plane's own inclusive geometry. Any
            // miss preserves its current canonical analytic evaluation.
#if R3_COLOR_START_REDUCED_COORDS
            // Consume the exact shared pose bytes directly; no conversion to
            // physical doubles and back to reduced floats on the field hit path.
            sampled_field =
                fields->sample_group(plane.type, posed_x, size_t(nb) * 4, 1,
                                     size_t(nb) * 4, fw, true);
#else
            sampled_field = fields->sample_group(plane.type, fw.points.data(),
                                                 size_t(nb) * 4, 3, 1, fw);
#endif
            if (sampled_field) {
                if (fs) ++fs->current_four_start_field_groups;
                for (int j = 0; j < nb; ++j) {
                    __m128 a = _mm_loadu_ps(fw.channels[size_t(j) * 4].data());
                    __m128 b =
                        _mm_loadu_ps(fw.channels[size_t(j) * 4 + 1].data());
                    __m128 c =
                        _mm_loadu_ps(fw.channels[size_t(j) * 4 + 2].data());
                    __m128 d =
                        _mm_loadu_ps(fw.channels[size_t(j) * 4 + 3].data());
                    // Samples are [f, gx, gy, gz] per start; transpose to
                    // one vector per channel with start k in lane k.
                    _MM_TRANSPOSE4_PS(a, b, c, d);
                    plane_value = _mm_add_ps(plane_value, a);
                    if (gradient && !ValuesOnly) {
                        const __m128 zero = _mm_setzero_ps();
                        const __m128 fx = _mm_sub_ps(zero, b),
                                     fy = _mm_sub_ps(zero, c),
                                     fz = _mm_sub_ps(zero, d);
                        const __m128 rx = _mm_sub_ps(
                            _mm_loadu_ps(posed_x + size_t(j) * 4), pose_tx);
                        const __m128 ry = _mm_sub_ps(
                            _mm_loadu_ps(posed_y + size_t(j) * 4), pose_ty);
                        const __m128 rz = _mm_sub_ps(
                            _mm_loadu_ps(posed_z + size_t(j) * 4), pose_tz);
                        const __m128 two = _mm_set1_ps(2.0f);
                        plane_gradient[0] = _mm_add_ps(
                            plane_gradient[0],
                            _mm_mul_ps(two, _mm_sub_ps(_mm_mul_ps(ry, fz),
                                                       _mm_mul_ps(rz, fy))));
                        plane_gradient[1] = _mm_add_ps(
                            plane_gradient[1],
                            _mm_mul_ps(two, _mm_sub_ps(_mm_mul_ps(rz, fx),
                                                       _mm_mul_ps(rx, fz))));
                        plane_gradient[2] = _mm_add_ps(
                            plane_gradient[2],
                            _mm_mul_ps(two, _mm_sub_ps(_mm_mul_ps(rx, fy),
                                                       _mm_mul_ps(ry, fx))));
                        plane_gradient[3] = _mm_add_ps(plane_gradient[3], fx);
                        plane_gradient[4] = _mm_add_ps(plane_gradient[4], fy);
                        plane_gradient[5] = _mm_add_ps(plane_gradient[5], fz);
                    }
                }
            }
        }
        if (fs && backend != COLOR_FIELD_ANALYTIC) {
            if (sampled_field) {
                fs->interpolated_samples += group_samples;
                fs->interpolated_terms += group_terms;
            } else {
                fs->analytic_fallback_samples += group_samples;
                fs->analytic_fallback_terms += group_terms;
                if (descriptor.field_active[size_t(plane.type)])
                    fs->out_of_grid_fallback_samples += group_samples;
                else
                    fs->inactive_fallback_samples += group_samples;
            }
        }
#endif
        if (!sampled_field && factor_torque && gradient) {
            // Match the existing factored path: accumulate force over each
            // target plane, then form the pose torque once per target atom.
            // Library-outer, query-inner. f_j = sum_i translation_scale *
            // v_ij * (a_i - b'_j); torque += 2 r_j x f_j.
            const float* library_x = posed_x;
            const float* library_y = posed_y;
            const float* library_z = posed_z;
            for (int j = 0; j < nb; ++j) {
                const __m128 bx = _mm_loadu_ps(library_x);
                const __m128 by = _mm_loadu_ps(library_y);
                const __m128 bz = _mm_loadu_ps(library_z);
                __m128 fx = _mm_setzero_ps();
                __m128 fy = _mm_setzero_ps();
                __m128 fz = _mm_setzero_ps();
                const float* query = plane.query_xyz;
                for (int i = 0; i < plane.query_count; ++i) {
                    const __m128 dx = _mm_sub_ps(_mm_set1_ps(query[0]), bx);
                    const __m128 dy = _mm_sub_ps(_mm_set1_ps(query[1]), by);
                    const __m128 dz = _mm_sub_ps(_mm_set1_ps(query[2]), bz);
                    const __m128 d2 = _mm_fmadd_ps(
                        dx, dx, _mm_fmadd_ps(dy, dy, _mm_mul_ps(dz, dz)));
                    const __m128 value = _mm_mul_ps(
                        exponential(d2, exponent_scale), coefficient);
                    plane_value = _mm_add_ps(
                        plane_value,
                        search_value_rounding_boundary<ValuesOnly>(value));
                    const __m128 force = _mm_mul_ps(value, translation_scale);
                    fx = _mm_add_ps(fx, _mm_mul_ps(force, dx));
                    fy = _mm_add_ps(fy, _mm_mul_ps(force, dy));
                    fz = _mm_add_ps(fz, _mm_mul_ps(force, dz));
                    query += plane.query_stride;
                }
                // Moment arm r_j = b'_j - t'.
                const __m128 rx = _mm_sub_ps(bx, pose_tx);
                const __m128 ry = _mm_sub_ps(by, pose_ty);
                const __m128 rz = _mm_sub_ps(bz, pose_tz);
                const __m128 two = _mm_set1_ps(2.0f);
                plane_gradient[0] =
                    _mm_add_ps(plane_gradient[0],
                               _mm_mul_ps(two, _mm_sub_ps(_mm_mul_ps(ry, fz),
                                                          _mm_mul_ps(rz, fy))));
                plane_gradient[1] =
                    _mm_add_ps(plane_gradient[1],
                               _mm_mul_ps(two, _mm_sub_ps(_mm_mul_ps(rz, fx),
                                                          _mm_mul_ps(rx, fz))));
                plane_gradient[2] =
                    _mm_add_ps(plane_gradient[2],
                               _mm_mul_ps(two, _mm_sub_ps(_mm_mul_ps(rx, fy),
                                                          _mm_mul_ps(ry, fx))));
                plane_gradient[3] = _mm_add_ps(plane_gradient[3], fx);
                plane_gradient[4] = _mm_add_ps(plane_gradient[4], fy);
                plane_gradient[5] = _mm_add_ps(plane_gradient[5], fz);
                library_x += 4;
                library_y += 4;
                library_z += 4;
            }
        } else if (!sampled_field) {
            // Keep the x86 un-factored canonical term order: query i, library j.
            // Also serves gradient=false. Value-only SEARCH comes here only
            // when factoring is off; with factoring on it takes the branch
            // above, whose dead force terms are discarded. Per pair:
            // torque_scale * v * (r x d), translation_scale * v * d.
            const float* query = plane.query_xyz;
            for (int i = 0; i < plane.query_count; ++i) {
                const __m128 qx = _mm_set1_ps(query[0]);
                const __m128 qy = _mm_set1_ps(query[1]);
                const __m128 qz = _mm_set1_ps(query[2]);
                const float* library_x = posed_x;
                const float* library_y = posed_y;
                const float* library_z = posed_z;
                for (int j = 0; j < nb; ++j) {
                    const __m128 bx = _mm_loadu_ps(library_x);
                    const __m128 by = _mm_loadu_ps(library_y);
                    const __m128 bz = _mm_loadu_ps(library_z);
                    const __m128 dx = _mm_sub_ps(qx, bx);
                    const __m128 dy = _mm_sub_ps(qy, by);
                    const __m128 dz = _mm_sub_ps(qz, bz);
                    const __m128 d2 = _mm_fmadd_ps(
                        dx, dx, _mm_fmadd_ps(dy, dy, _mm_mul_ps(dz, dz)));
                    const __m128 value = _mm_mul_ps(
                        exponential(d2, exponent_scale), coefficient);
                    plane_value = _mm_add_ps(
                        plane_value,
                        search_value_rounding_boundary<ValuesOnly>(value));
                    library_x += 4;
                    library_y += 4;
                    library_z += 4;
                    if (!gradient || ValuesOnly) continue;
                    const __m128 rx = _mm_sub_ps(bx, pose_tx);
                    const __m128 ry = _mm_sub_ps(by, pose_ty);
                    const __m128 rz = _mm_sub_ps(bz, pose_tz);
                    const __m128 torque = _mm_mul_ps(value, torque_scale);
                    plane_gradient[0] = _mm_add_ps(
                        plane_gradient[0],
                        _mm_mul_ps(torque, _mm_sub_ps(_mm_mul_ps(ry, dz),
                                                      _mm_mul_ps(rz, dy))));
                    plane_gradient[1] = _mm_add_ps(
                        plane_gradient[1],
                        _mm_mul_ps(torque, _mm_sub_ps(_mm_mul_ps(rz, dx),
                                                      _mm_mul_ps(rx, dz))));
                    plane_gradient[2] = _mm_add_ps(
                        plane_gradient[2],
                        _mm_mul_ps(torque, _mm_sub_ps(_mm_mul_ps(rx, dy),
                                                      _mm_mul_ps(ry, dx))));
                    const __m128 translate =
                        _mm_mul_ps(value, translation_scale);
                    plane_gradient[3] = _mm_add_ps(plane_gradient[3],
                                                   _mm_mul_ps(translate, dx));
                    plane_gradient[4] = _mm_add_ps(plane_gradient[4],
                                                   _mm_mul_ps(translate, dy));
                    plane_gradient[5] = _mm_add_ps(plane_gradient[5],
                                                   _mm_mul_ps(translate, dz));
                }
                query += plane.query_stride;
            }
        }
        total_value = _mm_add_ps(total_value, plane_value);
        if (gradient && !ValuesOnly)
            for (int component = 0; component < 6; ++component)
                total_gradient[component] = _mm_add_ps(
                    total_gradient[component], plane_gradient[component]);
        // Exact counts: every issued term is a real pair term.
        const uint64_t count =
            uint64_t(plane.query_count) * uint64_t(plane.library_count);
        for (int lane = 0; lane < 4; ++lane)
            term_counts[lane] += count;
    }

    // Restore physical units at the evaluator boundary. The default
    // amplitude is common to every canonical term, so it multiplies the
    // finished totals instead of each pair term.
    if constexpr (Normalize) {
        const __m128 amplitude =
            _mm_set1_ps(DefaultColorParameters::coefficient);
        total_value = _mm_mul_ps(total_value, amplitude);
        if (gradient && !ValuesOnly)
            for (auto& component : total_gradient)
                component = _mm_mul_ps(component, amplitude);
    }

    alignas(16) float values[4];
    _mm_store_ps(values, total_value);
    for (int lane = 0; lane < 4; ++lane)
        output[lane].value = double(values[lane]);
    if (gradient && !ValuesOnly)
        for (int component = 0; component < 6; ++component) {
            _mm_store_ps(values, total_gradient[component]);
            for (int lane = 0; lane < 4; ++lane)
                output[lane].descent[component + 1] = double(values[lane]);
        }
    return 0x0f;
}
#endif

#if defined(__AVX2__)
// Lifts the runtime search_output choice into the ValuesOnly template
// argument, once per call.
template <bool Fixed, bool Normalize, typename Exponential>
uint8_t
evaluate_xmm4(const ColorCanonicalFourStartDescriptor& descriptor,
              bool factor_torque, const PreparedReducedPose* const poses[4],
              bool gradient, ColorEvaluation output[4], uint64_t term_counts[4],
              ColorCanonicalFourStartWorkspace& workspace,
              Exponential exponential, ColorSearchOutput search_output) {
    if (search_output == ColorSearchOutput::ValueOnly)
        return evaluate_xmm4_impl<Fixed, Normalize, true>(
            descriptor, factor_torque, poses, gradient, output, term_counts,
            workspace, exponential);
    return evaluate_xmm4_impl<Fixed, Normalize, false>(
        descriptor, factor_torque, poses, gradient, output, term_counts,
        workspace, exponential);
}
#endif
}  // namespace

uint8_t evaluate_color_canonical_four_start_xmm4(
    const ColorCanonicalFourStartDescriptor& descriptor, bool factor_torque,
    const PreparedReducedPose* const poses[4], bool gradient,
    ColorEvaluation output[4], uint64_t term_counts[4],
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode, ColorSearchOutput search_output) {
#if defined(__AVX2__)
    // Default-parameter specialization needs the exact default
    // width/height, no field bundle, and, when Normalize is compiled in, at
    // least eight issued pairs to amortize the boundary multiplies.
    const bool default_parameters =
        descriptor.default_parameters && !descriptor.field_bundle &&
        (!R3_COLOR_PARAM_NORMALIZE || descriptor.amplitude_amortized);
    const auto evaluate = [&](auto exponential) {
        if ((R3_COLOR_PARAM_FIXED || R3_COLOR_PARAM_NORMALIZE) && default_parameters)
            return evaluate_xmm4<bool(R3_COLOR_PARAM_FIXED), bool(R3_COLOR_PARAM_NORMALIZE)>(
                descriptor, factor_torque, poses, gradient, output, term_counts,
                workspace, exponential, search_output);
        return evaluate_xmm4<false, false>(
            descriptor, factor_torque, poses, gradient, output, term_counts,
            workspace, exponential, search_output);
    };
    // Select the exponential once per call, never per term. The raw-bit
    // modes apply only when gradient=true; gradient=false always takes the
    // polynomial, which is why value-only SEARCH keeps gradient=true and
    // uses ValuesOnly instead.
#if R3_COLOR_PARAM_FUSED
    if (gradient && default_parameters &&
        optimizer_exp_mode == OptimizerExpMode::RawBitsUnclamped)
        return evaluate(
            [](__m128 d2, __m128) { return exp4_fused_rawbits(d2, true); });
    if (gradient && default_parameters &&
        optimizer_exp_mode == OptimizerExpMode::RawBits)
        return evaluate(
            [](__m128 d2, __m128) { return exp4_fused_rawbits(d2, false); });
#endif
    if (gradient && optimizer_exp_mode == OptimizerExpMode::RawBitsUnclamped)
        return evaluate([](__m128 d2, __m128 beta) {
            return exp4_rawbits(_mm_mul_ps(d2, beta), true);
        });
    if (gradient && optimizer_exp_mode == OptimizerExpMode::RawBits)
        return evaluate([](__m128 d2, __m128 beta) {
            return exp4_rawbits(_mm_mul_ps(d2, beta), false);
        });
    return evaluate([](__m128 d2, __m128 beta) {
        return exp4_polynomial(_mm_mul_ps(d2, beta));
    });
#else
    // Built without AVX2: always fall back.
    (void)descriptor;
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

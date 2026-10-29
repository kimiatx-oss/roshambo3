// Native NEON4 canonical analytic color for one conformer.
//
// Lane layout: one float32x4_t = the four starts of ONE library conformer,
// start k in lane k, no cross-lane reduction. Query coordinates are stored
// once and broadcast (vdupq); each raw library atom is broadcast and posed
// by the four start matrices into [x|y|z][atom][4 starts] scratch.
//
// This is ARM's singleton/tail helper. Compatible adjacent conformers use
// the two-stream helper (color_canonical_two_stream_neon.cpp), which keeps
// this file's term order per stream. ARM's established order is library j
// outer, query i inner, for both the factored and unfactored gradients;
// x86's unfactored order is the reverse, so the two ISAs need not agree bit
// for bit.
//
// Return value: 0x0f when all four starts were written, 0 when nothing was
// written and the caller must take the FP64 fallback for the whole group.
// Measured gain on ARM C4A: 12,841 -> 14,442 conformers/s (+12.46%),
// frozen eight-query one-core complete screen.
#include "color_canonical_four_start.hpp"

#include <cmath>

#include "kernel.hpp"
#include "optimizer_exp.hpp"
#include "optimizer_exp_neon.hpp"
#include "color_default_parameters.hpp"
#include "reduced_pose.hpp"
#include "color_pose_envelope_neon.hpp"

namespace r3::cpu {
namespace {

// Physical coordinate envelope; maximum_reduced_coordinate_fp32 is its
// reduced-frame image rounded toward zero (same as the x86 helpers).
constexpr double kMaximumCoordinate = 1.0e9;

float maximum_reduced_coordinate_fp32() {
    static const float value =
        std::nextafter(float(kMaximumCoordinate * double(COORD_SCALE)), 0.0f);
    return value;
}

bool safe_coordinate(float value, float maximum) {
    return std::isfinite(value) && std::abs(value) <= maximum;
}

// Full SEARCH reuses the rounded product value = exp * coefficient in the
// force terms. With ValuesOnly the product feeds only the sum, and the
// compiler fused across that rounding boundary (the first value-only attempt
// failed five exact SEARCH replays). The empty asm ("+w": a NEON register)
// keeps the FP32 product rounded and emits no instruction.
template <bool ValuesOnly>
inline float32x4_t search_value_rounding_boundary(float32x4_t value) {
    if constexpr (ValuesOnly) asm("" : "+w"(value));
    return value;
}

// Template parameters:
//   FactorTorque - accumulate f_j = sum_i translation_scale * v_ij *
//                  (a_i - b'_j) per target atom, then add 2 r_j x f_j once,
//                  instead of a torque cross product per pair.
//   Fixed        - compile default constants in (off by default).
//   Normalize    - unit-amplitude sums, one coefficient multiply on
//                  the seven totals at the end (on by default when eligible).
//   ValuesOnly   - the full-SEARCH value chain with every
//                  force/torque instruction and gradient store skipped.
// `exponential(d2, exponent_scale)` is exp(exponent_scale * d2) under the
// call's policy (optimizer_exp_neon.hpp), selected once outside the loops.
template <bool FactorTorque, bool Fixed, bool Normalize, bool ValuesOnly,
          typename Exponential>
uint8_t evaluate_neon4_impl(const ColorCanonicalFourStartDescriptor& descriptor,
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
    // [element][start]: each matrix entry loads as one vector.
    alignas(16) float matrix[9][4]{};
    alignas(16) float translation[3][4]{};
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

    // One per-pose bound proves every transformed coordinate is in
    // range, letting the per-atom checks be skipped; a failed proof keeps
    // them. See color_pose_envelope_neon.hpp.
    const bool envelope_safe =
        R3_COLOR_ANALYTIC_ENVELOPE &&
        canonical_pose_envelope_neon4(descriptor, matrix, translation,
                                      maximum_reduced);
    const float32x4_t m00 = vld1q_f32(matrix[0]);
    const float32x4_t m01 = vld1q_f32(matrix[1]);
    const float32x4_t m02 = vld1q_f32(matrix[2]);
    const float32x4_t m10 = vld1q_f32(matrix[3]);
    const float32x4_t m11 = vld1q_f32(matrix[4]);
    const float32x4_t m12 = vld1q_f32(matrix[5]);
    const float32x4_t m20 = vld1q_f32(matrix[6]);
    const float32x4_t m21 = vld1q_f32(matrix[7]);
    const float32x4_t m22 = vld1q_f32(matrix[8]);
    const float32x4_t tx = vld1q_f32(translation[0]);
    const float32x4_t ty = vld1q_f32(translation[1]);
    const float32x4_t tz = vld1q_f32(translation[2]);
    // Pass 1: b' = M raw + t' for every atom and start, validated before any
    // output or term count is written.
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
        for (int atom = 0; atom < nb; ++atom) {
            const float32x4_t x = vdupq_n_f32(raw[0]);
            const float32x4_t y = vdupq_n_f32(raw[1]);
            const float32x4_t z = vdupq_n_f32(raw[2]);
            float32x4_t bx = vmulq_f32(m00, x);
            bx = vfmaq_f32(bx, m01, y);
            bx = vfmaq_f32(bx, m02, z);
            bx = vaddq_f32(bx, tx);
            float32x4_t by = vmulq_f32(m10, x);
            by = vfmaq_f32(by, m11, y);
            by = vfmaq_f32(by, m12, z);
            by = vaddq_f32(by, ty);
            float32x4_t bz = vmulq_f32(m20, x);
            bz = vfmaq_f32(bz, m21, y);
            bz = vfmaq_f32(bz, m22, z);
            bz = vaddq_f32(bz, tz);
            vst1q_f32(posed_x, bx);
            vst1q_f32(posed_y, by);
            vst1q_f32(posed_z, bz);
            if (!envelope_safe) {
                alignas(16) float check[4];
                vst1q_f32(check, bx);
                for (float value : check)
                    if (!safe_coordinate(value, maximum_reduced)) return 0;
                vst1q_f32(check, by);
                for (float value : check)
                    if (!safe_coordinate(value, maximum_reduced)) return 0;
                vst1q_f32(check, bz);
                for (float value : check)
                    if (!safe_coordinate(value, maximum_reduced)) return 0;
            }
            raw += plane.library_stride;
            posed_x += 4;
            posed_y += 4;
            posed_z += 4;
        }
    }

    // Pass 2: Gaussian overlap. total_gradient[0..2] rotational, [3..5]
    // translational descent (ColorEvaluation slots 1..6).
    float32x4_t total_value = vdupq_n_f32(0.0f);
    float32x4_t total_gradient[6];
    for (auto& component : total_gradient)
        component = vdupq_n_f32(0.0f);
    const float32x4_t pose_tx = vld1q_f32(translation[0]);
    const float32x4_t pose_ty = vld1q_f32(translation[1]);
    const float32x4_t pose_tz = vld1q_f32(translation[2]);
    for (int plane_index = 0; plane_index < descriptor.plane_count;
         ++plane_index) {
        const ColorCanonicalFourStartPlane& plane =
            descriptor.planes[size_t(plane_index)];
        const int nb = plane.library_count;
        float32x4_t plane_value = vdupq_n_f32(0.0f);
        float32x4_t plane_gradient[6];
        for (auto& component : plane_gradient)
            component = vdupq_n_f32(0.0f);
        const float32x4_t coefficient =
            vdupq_n_f32(Normalize ? 1.0f
                        : Fixed   ? DefaultColorParameters::coefficient
                                  : plane.coefficient);
        const float32x4_t exponent_scale =
            vdupq_n_f32(Fixed ? DefaultColorParameters::exponent_scale
                              : plane.exponent_scale);
        const float32x4_t torque_scale = vdupq_n_f32(
            Fixed ? DefaultColorParameters::torque_scale : plane.torque_scale);
        const float32x4_t translation_scale =
            vdupq_n_f32(Fixed ? DefaultColorParameters::translation_scale
                              : plane.translation_scale);
        const float* posed_x =
            workspace.posed_library_xyz.data() + plane.posed_offset;
        const float* posed_y = posed_x + size_t(nb) * 4;
        const float* posed_z = posed_y + size_t(nb) * 4;

        // Match the established NEON order: library j, then query i.
        const float* library_x = posed_x;
        const float* library_y = posed_y;
        const float* library_z = posed_z;
        for (int j = 0; j < nb; ++j) {
            const float32x4_t bx = vld1q_f32(library_x);
            const float32x4_t by = vld1q_f32(library_y);
            const float32x4_t bz = vld1q_f32(library_z);
            // Moment arm r_j = b'_j - t', hoisted out of the query loop.
            const float32x4_t rx = vsubq_f32(bx, pose_tx);
            const float32x4_t ry = vsubq_f32(by, pose_ty);
            const float32x4_t rz = vsubq_f32(bz, pose_tz);
            float32x4_t fx = vdupq_n_f32(0.0f);
            float32x4_t fy = vdupq_n_f32(0.0f);
            float32x4_t fz = vdupq_n_f32(0.0f);
            const float* query = plane.query_xyz;
            for (int i = 0; i < plane.query_count; ++i) {
                const float32x4_t dx = vsubq_f32(vdupq_n_f32(query[0]), bx);
                const float32x4_t dy = vsubq_f32(vdupq_n_f32(query[1]), by);
                const float32x4_t dz = vsubq_f32(vdupq_n_f32(query[2]), bz);
                const float32x4_t d2 =
                    vfmaq_f32(vfmaq_f32(vmulq_f32(dz, dz), dy, dy), dx, dx);
                const float32x4_t value =
                    vmulq_f32(exponential(d2, exponent_scale), coefficient);
                plane_value = vaddq_f32(
                    plane_value,
                    search_value_rounding_boundary<ValuesOnly>(value));
                query += plane.query_stride;
                if (!gradient || ValuesOnly) continue;
                if constexpr (FactorTorque) {
                    // Factored: accumulate the spatial force only.
                    const float32x4_t force =
                        vmulq_f32(value, translation_scale);
                    fx = vaddq_f32(fx, vmulq_f32(force, dx));
                    fy = vaddq_f32(fy, vmulq_f32(force, dy));
                    fz = vaddq_f32(fz, vmulq_f32(force, dz));
                } else {
                    // Unfactored: torque_scale * v * (r x d) per pair.
                    const float32x4_t torque = vmulq_f32(value, torque_scale);
                    plane_gradient[0] = vaddq_f32(
                        plane_gradient[0],
                        vmulq_f32(torque, vsubq_f32(vmulq_f32(ry, dz),
                                                    vmulq_f32(rz, dy))));
                    plane_gradient[1] = vaddq_f32(
                        plane_gradient[1],
                        vmulq_f32(torque, vsubq_f32(vmulq_f32(rz, dx),
                                                    vmulq_f32(rx, dz))));
                    plane_gradient[2] = vaddq_f32(
                        plane_gradient[2],
                        vmulq_f32(torque, vsubq_f32(vmulq_f32(rx, dy),
                                                    vmulq_f32(ry, dx))));
                    const float32x4_t translate =
                        vmulq_f32(value, translation_scale);
                    plane_gradient[3] =
                        vaddq_f32(plane_gradient[3], vmulq_f32(translate, dx));
                    plane_gradient[4] =
                        vaddq_f32(plane_gradient[4], vmulq_f32(translate, dy));
                    plane_gradient[5] =
                        vaddq_f32(plane_gradient[5], vmulq_f32(translate, dz));
                }
            }
            if constexpr (FactorTorque) {
                // One torque per target atom: 2 r_j x f_j.
                if (gradient && !ValuesOnly) {
                    const float32x4_t two = vdupq_n_f32(2.0f);
                    plane_gradient[0] =
                        vaddq_f32(plane_gradient[0],
                                  vmulq_f32(two, vsubq_f32(vmulq_f32(ry, fz),
                                                           vmulq_f32(rz, fy))));
                    plane_gradient[1] =
                        vaddq_f32(plane_gradient[1],
                                  vmulq_f32(two, vsubq_f32(vmulq_f32(rz, fx),
                                                           vmulq_f32(rx, fz))));
                    plane_gradient[2] =
                        vaddq_f32(plane_gradient[2],
                                  vmulq_f32(two, vsubq_f32(vmulq_f32(rx, fy),
                                                           vmulq_f32(ry, fx))));
                    plane_gradient[3] = vaddq_f32(plane_gradient[3], fx);
                    plane_gradient[4] = vaddq_f32(plane_gradient[4], fy);
                    plane_gradient[5] = vaddq_f32(plane_gradient[5], fz);
                }
            }
            library_x += 4;
            library_y += 4;
            library_z += 4;
        }
        total_value = vaddq_f32(total_value, plane_value);
        if (gradient && !ValuesOnly)
            for (int component = 0; component < 6; ++component)
                total_gradient[component] = vaddq_f32(
                    total_gradient[component], plane_gradient[component]);
        const uint64_t count =
            uint64_t(plane.query_count) * uint64_t(plane.library_count);
        for (int lane = 0; lane < 4; ++lane)
            term_counts[lane] += count;
    }

    // Restore physical units once at the evaluator boundary.
    if constexpr (Normalize) {
        total_value =
            vmulq_n_f32(total_value, DefaultColorParameters::coefficient);
        if (gradient && !ValuesOnly)
            for (auto& component : total_gradient)
                component =
                    vmulq_n_f32(component, DefaultColorParameters::coefficient);
    }
    alignas(16) float values[4];
    vst1q_f32(values, total_value);
    for (int lane = 0; lane < 4; ++lane)
        output[lane].value = double(values[lane]);
    if (gradient && !ValuesOnly)
        for (int component = 0; component < 6; ++component) {
            vst1q_f32(values, total_gradient[component]);
            for (int lane = 0; lane < 4; ++lane)
                output[lane].descent[component + 1] = double(values[lane]);
        }
    return 0x0f;
}

// Lifts search_output into the ValuesOnly template argument once per call.
template <bool FactorTorque, bool Fixed, bool Normalize, typename Exponential>
uint8_t evaluate_neon4(const ColorCanonicalFourStartDescriptor& descriptor,
                       const PreparedReducedPose* const poses[4], bool gradient,
                       ColorEvaluation output[4], uint64_t term_counts[4],
                       ColorCanonicalFourStartWorkspace& workspace,
                       Exponential exponential,
                       ColorSearchOutput search_output) {
    if (search_output == ColorSearchOutput::ValueOnly)
        return evaluate_neon4_impl<FactorTorque, Fixed, Normalize, true>(
            descriptor, poses, gradient, output, term_counts, workspace,
            exponential);
    return evaluate_neon4_impl<FactorTorque, Fixed, Normalize, false>(
        descriptor, poses, gradient, output, term_counts, workspace,
        exponential);
}
}  // namespace

uint8_t evaluate_color_canonical_four_start_neon4(
    const ColorCanonicalFourStartDescriptor& descriptor, bool factor_torque,
    const PreparedReducedPose* const poses[4], bool gradient,
    ColorEvaluation output[4], uint64_t term_counts[4],
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode, ColorSearchOutput search_output) {
    // Default-parameter specialization: exact default width/height,
    // no field bundle, and >= 8 issued pairs when Normalize is compiled in.
    // Unlike x86, factoring is a template argument here, so the unfactored
    // instantiation also serves gradient=false and carries no factored code.
    const bool default_parameters =
        descriptor.default_parameters && !descriptor.field_bundle &&
        (!R3_COLOR_PARAM_NORMALIZE || descriptor.amplitude_amortized);
    const auto evaluate = [&](auto exponential) {
        if ((R3_COLOR_PARAM_FIXED || R3_COLOR_PARAM_NORMALIZE) && default_parameters) {
            if (gradient && factor_torque)
                return evaluate_neon4<true, bool(R3_COLOR_PARAM_FIXED),
                                      bool(R3_COLOR_PARAM_NORMALIZE)>(
                    descriptor, poses, gradient, output, term_counts, workspace,
                    exponential, search_output);
            return evaluate_neon4<false, bool(R3_COLOR_PARAM_FIXED),
                                  bool(R3_COLOR_PARAM_NORMALIZE)>(
                descriptor, poses, gradient, output, term_counts, workspace,
                exponential, search_output);
        }
        if (gradient && factor_torque)
            return evaluate_neon4<true, false, false>(
                descriptor, poses, gradient, output, term_counts, workspace,
                exponential, search_output);
        return evaluate_neon4<false, false, false>(
            descriptor, poses, gradient, output, term_counts, workspace,
            exponential, search_output);
    };
    // Exponential chosen once per call. Raw-bit modes apply only with
    // gradient=true; gradient=false always takes the polynomial,
    // which is why value-only SEARCH keeps gradient=true. The fused form
    // needs default parameters.
#if R3_COLOR_PARAM_FUSED
    if (gradient && default_parameters &&
        optimizer_exp_mode == OptimizerExpMode::RawBitsUnclamped)
        return evaluate([](float32x4_t d2, float32x4_t) {
            return neon_exp_default_fused_rawbits(d2, true);
        });
    if (gradient && default_parameters &&
        optimizer_exp_mode == OptimizerExpMode::RawBits)
        return evaluate([](float32x4_t d2, float32x4_t) {
            return neon_exp_default_fused_rawbits(d2, false);
        });
#endif
    if (gradient && optimizer_exp_mode == OptimizerExpMode::RawBitsUnclamped)
        return evaluate([](float32x4_t d2, float32x4_t beta) {
            return neon_exp_rawbits(vmulq_f32(d2, beta), true);
        });
    if (gradient && optimizer_exp_mode == OptimizerExpMode::RawBits)
        return evaluate([](float32x4_t d2, float32x4_t beta) {
            return neon_exp_rawbits(vmulq_f32(d2, beta), false);
        });
    return evaluate([](float32x4_t d2, float32x4_t beta) {
        return neon_exp_polynomial(vmulq_f32(d2, beta));
    });
}

}  // namespace r3::cpu

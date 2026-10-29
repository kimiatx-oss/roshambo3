// Two independent NEON4 streams, immutable shared plane metadata.
//
// ARM counterpart of x86's YMM8 pair helper. Two compatible conformers
// (identical count/offset signature, color_canonical_pair_compatible) are
// evaluated together, but as two separate 128-bit streams rather than one
// packed register:
//   stream 0: descriptors[0], starts 0..3 in lanes 0..3 -> output[0..3]
//   stream 1: descriptors[1], starts 0..3 in lanes 0..3 -> output[4..7]
// Both streams share loop bounds, query broadcasts and plane constants, and
// their distance/exp/gradient operations are interleaved for instruction-
// level parallelism; no value ever crosses streams or starts. Per stream the
// operation sequence and term order match the singleton NEON4 helper
// (library j outer, query i inner); every frozen all20/start output was
// exact against the accepted baseline.
//
// Separate streams avoid x86-style cross-half packing and
// shuffles. Isolated all-eight gain +3.100%; stopped at two streams because
// outer-loop register pressure did not justify four.
//
// Return value: 0xff when all eight lanes were written; 0 when nothing was
// written, after which the caller retries each conformer through NEON4.
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
constexpr double kMaximumCoordinate = 1.0e9;

float maximum_reduced_coordinate_fp32() {
    static const float value =
        std::nextafter(float(kMaximumCoordinate * double(COORD_SCALE)), 0.f);
    return value;
}

bool safe_coordinate(float value, float maximum) {
    return std::isfinite(value) && std::abs(value) <= maximum;
}

// Poses one conformer's library atoms for its four starts into `scratch`
// ([x|y|z][atom][4 starts] per plane) and copies the four translations to
// [component][start] for the moment arms. Returns false (0) on the same
// fallback triggers as NEON4: unsafe immutable inputs, a missing pose, or a
// translation or transformed coordinate outside the reduced envelope.
bool prepare_stream(const ColorCanonicalFourStartDescriptor& descriptor,
                    const PreparedReducedPose* const poses[4], float* scratch,
                    float translation[3][4]) {
    if (!descriptor.immutable_inputs_safe) return 0;
    const float maximum_reduced = maximum_reduced_coordinate_fp32();
    alignas(16) float matrix[9][4]{};
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

    // Per-pose envelope proof; failure keeps the per-atom checks.
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
    const float32x4_t maximum = vdupq_n_f32(maximum_reduced);
    for (int plane_index = 0; plane_index < descriptor.plane_count;
         ++plane_index) {
        const ColorCanonicalFourStartPlane& plane =
            descriptor.planes[size_t(plane_index)];
        const int nb = plane.library_count;
        const float* raw = plane.library_xyz;
        float* posed_x = scratch + plane.posed_offset;
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
            // Ordered comparisons against the same finite positive FP32
            // maximum reject NaN and either infinity as well as overflow.
            // Both signed zeros and values exactly at the bound remain safe.
            if (!envelope_safe) {
                const uint32x4_t valid =
                    vandq_u32(vandq_u32(vcleq_f32(vabsq_f32(bx), maximum),
                                        vcleq_f32(vabsq_f32(by), maximum)),
                              vcleq_f32(vabsq_f32(bz), maximum));
                if (vminvq_u32(valid) == 0) return 0;
            }
            raw += plane.library_stride;
            posed_x += 4;
            posed_y += 4;
            posed_z += 4;
        }
    }

    return true;
}

// Instruction-free register barrier that keeps value = exp * coefficient
// rounded to FP32 before it is summed when ValuesOnly removes its force
// uses; without it the compiler fused across that rounding boundary. See
// the NEON4 copy.
template <bool ValuesOnly>
inline float32x4_t search_value_rounding_boundary(float32x4_t value) {
    if constexpr (ValuesOnly) asm("" : "+w"(value));
    return value;
}

// FactorTorque / Fixed / Normalize / ValuesOnly as on evaluate_neon4_impl.
// Accumulators are [stream][7]: slot 0 is
// the value, 1..3 rotational and 4..6 translational descent, matching
// ColorEvaluation::descent indices.
template <bool FactorTorque, bool Fixed, bool Normalize, bool ValuesOnly,
          class Exponential>
uint8_t
evaluate_two_impl(const ColorCanonicalFourStartDescriptor descriptors[2],
                  const PreparedReducedPose* const poses[8], bool gradient,
                  ColorEvaluation output[8], uint64_t term_counts[8],
                  ColorCanonicalFourStartWorkspace& workspace,
                  Exponential exponential) {
    alignas(16) float translation[2][3][4]{};
    // Stream 1's posed scratch follows stream 0's; the workspace was sized
    // to 2 x posed_library_size for a compatible pair. Both streams are
    // posed and validated before any output is written.
    const size_t stride = descriptors[0].posed_library_size;
    if (!prepare_stream(descriptors[0], poses,
                        workspace.posed_library_xyz.data(), translation[0]) ||
        !prepare_stream(descriptors[1], poses + 4,
                        workspace.posed_library_xyz.data() + stride,
                        translation[1]))
        return 0;
    float32x4_t total[2][7];
    for (auto& stream : total)
        for (auto& value : stream)
            value = vdupq_n_f32(0);
    for (int pi = 0; pi < descriptors[0].plane_count; ++pi) {
        const auto& p = descriptors[0].planes[size_t(pi)];
        float32x4_t accum[2][7];
        for (auto& stream : accum)
            for (auto& value : stream)
                value = vdupq_n_f32(0);
        // One set of plane constants for both streams (equal by signature).
        const auto coefficient =
                       vdupq_n_f32(Normalize ? 1.0f
                                   : Fixed ? DefaultColorParameters::coefficient
                                           : p.coefficient),
                   decay = vdupq_n_f32(
                       Fixed ? DefaultColorParameters::exponent_scale
                             : p.exponent_scale);
        const auto translate = vdupq_n_f32(
                       Fixed ? DefaultColorParameters::translation_scale
                             : p.translation_scale),
                   torque =
                       vdupq_n_f32(Fixed ? DefaultColorParameters::torque_scale
                                         : p.torque_scale);
        const float* x0 = workspace.posed_library_xyz.data() + p.posed_offset;
        const float* y0 = x0 + size_t(p.library_count) * 4;
        const float* z0 = y0 + size_t(p.library_count) * 4;
        const float* x1 = x0 + stride;
        const float* y1 = y0 + stride;
        const float* z1 = z0 + stride;
        for (int j = 0; j < p.library_count; ++j) {
            const auto bx0 = vld1q_f32(x0 + 4 * j), by0 = vld1q_f32(y0 + 4 * j),
                       bz0 = vld1q_f32(z0 + 4 * j);
            const auto bx1 = vld1q_f32(x1 + 4 * j), by1 = vld1q_f32(y1 + 4 * j),
                       bz1 = vld1q_f32(z1 + 4 * j);
            const auto rx0 = vsubq_f32(bx0, vld1q_f32(translation[0][0])),
                       ry0 = vsubq_f32(by0, vld1q_f32(translation[0][1])),
                       rz0 = vsubq_f32(bz0, vld1q_f32(translation[0][2]));
            const auto rx1 = vsubq_f32(bx1, vld1q_f32(translation[1][0])),
                       ry1 = vsubq_f32(by1, vld1q_f32(translation[1][1])),
                       rz1 = vsubq_f32(bz1, vld1q_f32(translation[1][2]));
            auto fx0 = vdupq_n_f32(0), fy0 = vdupq_n_f32(0),
                 fz0 = vdupq_n_f32(0);
            auto fx1 = vdupq_n_f32(0), fy1 = vdupq_n_f32(0),
                 fz1 = vdupq_n_f32(0);
            const float* q = p.query_xyz;
            // One query broadcast feeds both streams.
            for (int i = 0; i < p.query_count; ++i, q += p.query_stride) {
                const auto qx = vdupq_n_f32(q[0]), qy = vdupq_n_f32(q[1]),
                           qz = vdupq_n_f32(q[2]);
                const auto dx0 = vsubq_f32(qx, bx0), dx1 = vsubq_f32(qx, bx1);
                const auto dy0 = vsubq_f32(qy, by0), dy1 = vsubq_f32(qy, by1);
                const auto dz0 = vsubq_f32(qz, bz0), dz1 = vsubq_f32(qz, bz1);
                const auto d20 = vfmaq_f32(
                    vfmaq_f32(vmulq_f32(dz0, dz0), dy0, dy0), dx0, dx0);
                const auto d21 = vfmaq_f32(
                    vfmaq_f32(vmulq_f32(dz1, dz1), dy1, dy1), dx1, dx1);
                const auto e0 = exponential(d20, decay),
                           e1 = exponential(d21, decay);
                const auto v0 = vmulq_f32(e0, coefficient),
                           v1 = vmulq_f32(e1, coefficient);
                accum[0][0] =
                    vaddq_f32(accum[0][0],
                              search_value_rounding_boundary<ValuesOnly>(v0));
                accum[1][0] =
                    vaddq_f32(accum[1][0],
                              search_value_rounding_boundary<ValuesOnly>(v1));
                if (!gradient || ValuesOnly) continue;
                const auto force0 = vmulq_f32(v0, translate),
                           force1 = vmulq_f32(v1, translate);
                if constexpr (FactorTorque) {
                    fx0 = vaddq_f32(fx0, vmulq_f32(force0, dx0));
                    fx1 = vaddq_f32(fx1, vmulq_f32(force1, dx1));
                    fy0 = vaddq_f32(fy0, vmulq_f32(force0, dy0));
                    fy1 = vaddq_f32(fy1, vmulq_f32(force1, dy1));
                    fz0 = vaddq_f32(fz0, vmulq_f32(force0, dz0));
                    fz1 = vaddq_f32(fz1, vmulq_f32(force1, dz1));
                } else {
                    const auto t0 = vmulq_f32(v0, torque),
                               t1 = vmulq_f32(v1, torque);
                    accum[0][1] = vaddq_f32(
                        accum[0][1],
                        vmulq_f32(t0, vsubq_f32(vmulq_f32(ry0, dz0),
                                                vmulq_f32(rz0, dy0))));
                    accum[1][1] = vaddq_f32(
                        accum[1][1],
                        vmulq_f32(t1, vsubq_f32(vmulq_f32(ry1, dz1),
                                                vmulq_f32(rz1, dy1))));
                    accum[0][2] = vaddq_f32(
                        accum[0][2],
                        vmulq_f32(t0, vsubq_f32(vmulq_f32(rz0, dx0),
                                                vmulq_f32(rx0, dz0))));
                    accum[1][2] = vaddq_f32(
                        accum[1][2],
                        vmulq_f32(t1, vsubq_f32(vmulq_f32(rz1, dx1),
                                                vmulq_f32(rx1, dz1))));
                    accum[0][3] = vaddq_f32(
                        accum[0][3],
                        vmulq_f32(t0, vsubq_f32(vmulq_f32(rx0, dy0),
                                                vmulq_f32(ry0, dx0))));
                    accum[1][3] = vaddq_f32(
                        accum[1][3],
                        vmulq_f32(t1, vsubq_f32(vmulq_f32(rx1, dy1),
                                                vmulq_f32(ry1, dx1))));
                    accum[0][4] =
                        vaddq_f32(accum[0][4], vmulq_f32(force0, dx0));
                    accum[1][4] =
                        vaddq_f32(accum[1][4], vmulq_f32(force1, dx1));
                    accum[0][5] =
                        vaddq_f32(accum[0][5], vmulq_f32(force0, dy0));
                    accum[1][5] =
                        vaddq_f32(accum[1][5], vmulq_f32(force1, dy1));
                    accum[0][6] =
                        vaddq_f32(accum[0][6], vmulq_f32(force0, dz0));
                    accum[1][6] =
                        vaddq_f32(accum[1][6], vmulq_f32(force1, dz1));
                }
            }
            if constexpr (FactorTorque) {
                if (gradient && !ValuesOnly) {
                    const auto two = vdupq_n_f32(2);
                    accum[0][1] = vaddq_f32(
                        accum[0][1],
                        vmulq_f32(two, vsubq_f32(vmulq_f32(ry0, fz0),
                                                 vmulq_f32(rz0, fy0))));
                    accum[1][1] = vaddq_f32(
                        accum[1][1],
                        vmulq_f32(two, vsubq_f32(vmulq_f32(ry1, fz1),
                                                 vmulq_f32(rz1, fy1))));
                    accum[0][2] = vaddq_f32(
                        accum[0][2],
                        vmulq_f32(two, vsubq_f32(vmulq_f32(rz0, fx0),
                                                 vmulq_f32(rx0, fz0))));
                    accum[1][2] = vaddq_f32(
                        accum[1][2],
                        vmulq_f32(two, vsubq_f32(vmulq_f32(rz1, fx1),
                                                 vmulq_f32(rx1, fz1))));
                    accum[0][3] = vaddq_f32(
                        accum[0][3],
                        vmulq_f32(two, vsubq_f32(vmulq_f32(rx0, fy0),
                                                 vmulq_f32(ry0, fx0))));
                    accum[1][3] = vaddq_f32(
                        accum[1][3],
                        vmulq_f32(two, vsubq_f32(vmulq_f32(rx1, fy1),
                                                 vmulq_f32(ry1, fx1))));
                    accum[0][4] = vaddq_f32(accum[0][4], fx0);
                    accum[1][4] = vaddq_f32(accum[1][4], fx1);
                    accum[0][5] = vaddq_f32(accum[0][5], fy0);
                    accum[1][5] = vaddq_f32(accum[1][5], fy1);
                    accum[0][6] = vaddq_f32(accum[0][6], fz0);
                    accum[1][6] = vaddq_f32(accum[1][6], fz1);
                }
            }
        }
        // Fold this plane into each stream's totals in plane order.
        for (int s = 0; s < 2; ++s) {
            total[s][0] = vaddq_f32(total[s][0], accum[s][0]);
            if (gradient && !ValuesOnly)
                for (int k = 1; k < 7; ++k)
                    total[s][k] = vaddq_f32(total[s][k], accum[s][k]);
        }
        const uint64_t count =
            uint64_t(p.query_count) * uint64_t(p.library_count);
        for (int lane = 0; lane < 8; ++lane)
            term_counts[lane] += count;
    }
    // Restore physical units once at the evaluator boundary.
    if constexpr (Normalize)
        for (auto& stream : total)
            for (int k = 0; k < (gradient && !ValuesOnly ? 7 : 1); ++k)
                stream[k] =
                    vmulq_n_f32(stream[k], DefaultColorParameters::coefficient);
    alignas(16) float values[4];
    for (int s = 0; s < 2; ++s) {
        vst1q_f32(values, total[s][0]);
        for (int lane = 0; lane < 4; ++lane)
            output[s * 4 + lane].value = double(values[lane]);
        if (gradient && !ValuesOnly)
            for (int k = 1; k < 7; ++k) {
                vst1q_f32(values, total[s][k]);
                for (int lane = 0; lane < 4; ++lane)
                    output[s * 4 + lane].descent[k] = double(values[lane]);
            }
    }
    return 0xff;
}

// Lifts search_output into the ValuesOnly template argument once per call.
template <bool FactorTorque, bool Fixed, bool Normalize, class Exponential>
uint8_t evaluate_two(const ColorCanonicalFourStartDescriptor descriptors[2],
                     const PreparedReducedPose* const poses[8], bool gradient,
                     ColorEvaluation output[8], uint64_t term_counts[8],
                     ColorCanonicalFourStartWorkspace& workspace,
                     Exponential exponential, ColorSearchOutput search_output) {
    if (search_output == ColorSearchOutput::ValueOnly)
        return evaluate_two_impl<FactorTorque, Fixed, Normalize, true>(
            descriptors, poses, gradient, output, term_counts, workspace,
            exponential);
    return evaluate_two_impl<FactorTorque, Fixed, Normalize, false>(
        descriptors, poses, gradient, output, term_counts, workspace,
        exponential);
}
}

uint8_t evaluate_color_canonical_two_stream_neon4(
    const ColorCanonicalFourStartDescriptor descriptors[2], bool factor_torque,
    const PreparedReducedPose* const poses[8], bool gradient,
    ColorEvaluation output[8], uint64_t term_counts[8],
    ColorCanonicalFourStartWorkspace& workspace, OptimizerExpMode mode,
    ColorSearchOutput search_output) {
    // Same policy selection as NEON4; both streams take one policy (the
    // pair signature already requires equal default/amortization flags).
    const bool default_parameters =
        descriptors[0].default_parameters &&
        descriptors[1].default_parameters && !descriptors[0].field_bundle &&
        !descriptors[1].field_bundle &&
        (!R3_COLOR_PARAM_NORMALIZE || (descriptors[0].amplitude_amortized &&
                                descriptors[1].amplitude_amortized));
    const auto evaluate = [&](auto exponential) {
        if ((R3_COLOR_PARAM_FIXED || R3_COLOR_PARAM_NORMALIZE) && default_parameters) {
            if (gradient && factor_torque)
                return evaluate_two<true, bool(R3_COLOR_PARAM_FIXED),
                                    bool(R3_COLOR_PARAM_NORMALIZE)>(
                    descriptors, poses, gradient, output, term_counts,
                    workspace, exponential, search_output);
            return evaluate_two<false, bool(R3_COLOR_PARAM_FIXED),
                                bool(R3_COLOR_PARAM_NORMALIZE)>(
                descriptors, poses, gradient, output, term_counts, workspace,
                exponential, search_output);
        }
        if (gradient && factor_torque)
            return evaluate_two<true, false, false>(
                descriptors, poses, gradient, output, term_counts, workspace,
                exponential, search_output);
        return evaluate_two<false, false, false>(descriptors, poses, gradient,
                                                 output, term_counts, workspace,
                                                 exponential, search_output);
    };
#if R3_COLOR_PARAM_FUSED
    if (gradient && default_parameters &&
        mode == OptimizerExpMode::RawBitsUnclamped)
        return evaluate([](float32x4_t d2, float32x4_t) {
            return neon_exp_default_fused_rawbits(d2, true);
        });
    if (gradient && default_parameters && mode == OptimizerExpMode::RawBits)
        return evaluate([](float32x4_t d2, float32x4_t) {
            return neon_exp_default_fused_rawbits(d2, false);
        });
#endif
    if (gradient && mode == OptimizerExpMode::RawBitsUnclamped)
        return evaluate([](float32x4_t d2, float32x4_t beta) {
            return neon_exp_rawbits(vmulq_f32(d2, beta), true);
        });
    if (gradient && mode == OptimizerExpMode::RawBits)
        return evaluate([](float32x4_t d2, float32x4_t beta) {
            return neon_exp_rawbits(vmulq_f32(d2, beta), false);
        });
    return evaluate([](float32x4_t d2, float32x4_t beta) {
        return neon_exp_polynomial(vmulq_f32(d2, beta));
    });
}
}

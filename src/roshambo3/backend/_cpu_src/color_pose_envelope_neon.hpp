// The common positive-coordinate bound, four actual poses at once.
//
// NEON form of `canonical_pose_envelope_safe` (color_canonical_four_start.hpp),
// evaluated for the four start poses that share one canonical color pass.
// For each output row, |M_r| . max|b| + |t_r| bounds every transformed
// library coordinate; if the slackened bound is within `maximum` for all
// three rows of all four poses, the per-atom coordinate checks are provably
// redundant and the caller takes the unchecked transform loop. A failure is
// ordinary ineligibility, not an error: the checked loop runs instead.
// On ARM this bound was +1.25-1.36% over per-atom checks alone at every stage.
//
// The 0 default below is a fallback only: color_canonical_four_start.hpp,
// included first, already sets the macro to 1 on aarch64 and x86-64.
#pragma once
#include <arm_neon.h>
#include "color_canonical_four_start.hpp"
#ifndef R3_COLOR_ANALYTIC_ENVELOPE
#define R3_COLOR_ANALYTIC_ENVELOPE 0
#endif
namespace r3::cpu {
inline bool canonical_pose_envelope_neon4(
    const ColorCanonicalFourStartDescriptor& descriptor,
    const float matrix[9][4], const float translation[3][4], float maximum) {
    // Same constants as the scalar proof, whose comment gives the rounding
    // argument: relative slack for the bound's own rounding chain, plus one
    // additive unit dominating any subnormal rounding error.
    constexpr float slack = 1.f + 16.f * std::numeric_limits<float>::epsilon();
    // matrix[k] holds element k of the four poses' matrices (lane = pose).
    auto valid = vdupq_n_u32(UINT32_MAX);
    for (int row = 0; row < 3; ++row) {
        auto bound = vmulq_n_f32(vabsq_f32(vld1q_f32(matrix[3 * row])),
                                 descriptor.library_max_abs[0]);
        bound = vaddq_f32(bound,
                          vmulq_n_f32(vabsq_f32(vld1q_f32(matrix[3 * row + 1])),
                                      descriptor.library_max_abs[1]));
        bound = vaddq_f32(bound,
                          vmulq_n_f32(vabsq_f32(vld1q_f32(matrix[3 * row + 2])),
                                      descriptor.library_max_abs[2]));
        bound = vaddq_f32(bound, vabsq_f32(vld1q_f32(translation[row])));
        bound = vaddq_f32(vmulq_n_f32(bound, slack), vdupq_n_f32(1.f));
        valid = vandq_u32(valid, vcleq_f32(bound, vdupq_n_f32(maximum)));
    }
    // All four lanes must pass: one pass serves all four poses together.
    return vminvq_u32(valid) != 0;
}
}

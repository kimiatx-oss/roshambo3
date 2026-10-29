// AArch64 Gaussian optimizer weights. No FPCR/FPSR writes in the hot path.
//
// NEON counterparts of optimizer_exp.hpp's x86 exponentials.
// They weight the color Gaussians inside the OPTIMIZER only. Reported scores
// come from the accurate final evaluators, which do not use the raw-bit forms
// (NUMERICS_POLICY section 2: no approximate tier is the source of a
// reported number). The policy is resolved once per call
// (resolve_optimizer_exp_mode), never per iteration.
//
// ARM-specific semantics the port had to re-establish:
// FCVTZS saturates on overflow where x86 CVTT returns the 0x80000000
// sentinel, and invalid-operation exceptions are governed by FPCR, not MXCSR.
#pragma once
#include <arm_neon.h>
#include "color_default_parameters.hpp"

namespace r3::cpu {
// The "baseline" accurate-enough exp(x): range-reduce x = (n + f) ln2 with
// n = floor(x log2 e), evaluate a degree-seven Taylor polynomial of e^z on
// z = f ln2 in [0, ln2), and scale by 2^n built from exponent bits. Inputs
// below -80 return exactly zero. Measured maximum relative error
// 3.73536705e-6 against exp over [-80, 0]. Also the polynomial used by the
// vectorized final rescore, which does not label it bit-exact.
inline float32x4_t neon_exp_polynomial(float32x4_t x) {
    const auto kept = vcgeq_f32(x, vdupq_n_f32(-80.f));
    const auto y =
        vmulq_n_f32(vmaxq_f32(x, vdupq_n_f32(-80.f)), 1.4426950408889634074f);
    const auto n = vrndmq_f32(y);
    const auto z = vmulq_n_f32(vsubq_f32(y, n), .69314718055994530942f);
    const auto bits =
        vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    constexpr float c[] = {1.f,        1.f,         .5f,         1.f / 6.f,
                           1.f / 24.f, 1.f / 120.f, 1.f / 720.f, 1.f / 5040.f};
    auto p = vdupq_n_f32(c[7]);
    for (int k = 6; k >= 0; --k)
        p = vfmaq_f32(vdupq_n_f32(c[k]), p, z);
    return vreinterpretq_f32_u32(vandq_u32(
        kept,
        vreinterpretq_u32_f32(vmulq_f32(p, vreinterpretq_f32_s32(bits)))));
}

// Raw-bit exp: build the FP32 bit pattern directly. 12102203 is
// round(2^23 log2 e), so trunc(x * 12102203) is x log2 e in units of the
// exponent field's lowest bit; adding the integer bias (127 << 23) - 366393
// and reinterpreting gives an approximation of e^x. The same constants as
// x86. At x = 0 it returns 0.978161275, not 1.
//
// Clamped mode masks x outside [-80, 0] (and NaN, via ordered compares) to
// zero. Unclamped mode keeps only x >= -80; the caller may select it only
// after neon_invalid_exceptions_masked() holds on the executing thread: an
// out-of-range conversion signals invalid-operation (sets FPSR.IOC),
// which must not trap. Its saturated bits are masked off either way.
inline float32x4_t neon_exp_rawbits(float32x4_t x, bool unclamped) {
    auto kept = vcgeq_f32(x, vdupq_n_f32(-80.f));
    if (!unclamped) {
        kept = vandq_u32(kept, vcleq_f32(x, vdupq_n_f32(0.f)));
        x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-80.f)), vdupq_n_f32(0.f));
    }
    // FCVTZS truncates toward zero and saturates on overflow (unlike CVTTPS).
    // Add the bias as unsigned bits, so even saturated tails wrap explicitly.
    // Mask BEFORE reinterpretation can expose an invalid bit pattern to FP.
    const auto converted =
        vreinterpretq_u32_s32(vcvtq_s32_f32(vmulq_n_f32(x, 12102203.f)));
    const auto biased =
        vaddq_u32(converted, vdupq_n_u32((127u << 23) - 366393u));
    return vreinterpretq_f32_u32(vandq_u32(biased, kept));
}

// FPCR.IOE (bit 8) clear means invalid-operation exceptions do not trap;
// the analogue of x86's `_MM_MASK_INVALID` test. Read once at scoring entry
// (color_scalar.cpp), never written: no control-word toggles.
inline bool neon_invalid_exceptions_masked() {
    unsigned long fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    return (fpcr & (1ul << 8)) == 0;
}
#if R3_COLOR_PARAM_FUSED
// Raw-bit exp of the default color Gaussian, taking the squared distance d2
// directly: the exponent scale and the 2^23 log2 e scale are pre-multiplied
// into one constant, removing a multiply per term. Fusing changes
// FP32 rounding and cutoff classification, so it was quality-gated;
// the -80 cutoff is still tested on the unfused exponent x so the kept set
// keeps its original rounded definition. On by default (R3_COLOR_PARAM_FUSED in
// color_default_parameters.hpp).
inline float32x4_t neon_exp_default_fused_rawbits(float32x4_t d2,
                                                  bool unclamped) {
    const auto x = vmulq_n_f32(d2, DefaultColorParameters::exponent_scale);
    auto kept = vcgeq_f32(x, vdupq_n_f32(-80.f));
    auto scaled = vmulq_n_f32(d2, DefaultColorParameters::fused_rawbits_scale);
    if (!unclamped) {
        kept = vandq_u32(kept, vcleq_f32(x, vdupq_n_f32(0.f)));
        scaled = vminq_f32(
            vmaxq_f32(
                scaled,
                vdupq_n_f32(-80.f * DefaultColorParameters::rawbits_scale)),
            vdupq_n_f32(0.f));
    }
    const auto converted = vreinterpretq_u32_s32(vcvtq_s32_f32(scaled));
    const auto biased =
        vaddq_u32(converted, vdupq_n_u32((127u << 23) - 366393u));
    return vreinterpretq_f32_u32(vandq_u32(biased, kept));
}
#endif
}

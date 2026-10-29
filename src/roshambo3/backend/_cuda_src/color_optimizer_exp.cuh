#pragma once

// 0=exp2f build-time opt-out; 1=default explicit hardware EX2;
// 2=CPU-style unpolished integer bits; 3=__expf with base conversion.
// This policy applies to optimizer and fresh SEARCH, never final scoring.
#ifndef ROSHAMBO3_COLOR_OPT_EXP_POLICY
#define ROSHAMBO3_COLOR_OPT_EXP_POLICY 1
#endif
static_assert(ROSHAMBO3_COLOR_OPT_EXP_POLICY >= 0 &&
              ROSHAMBO3_COLOR_OPT_EXP_POLICY <= 3, "invalid color exp policy");

__device__ __forceinline__ float r3_color_optimizer_exp(float u) {
#if ROSHAMBO3_COLOR_OPT_EXP_POLICY == 0
    return exp2f(-u);
#elif ROSHAMBO3_COLOR_OPT_EXP_POLICY == 1
    float out;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(-u));
    return out;
#elif ROSHAMBO3_COLOR_OPT_EXP_POLICY == 3
    return __expf(-0.6931471805599453f*u);
#else
    // The existing base-two Gaussian has x=-ln(2)*u. Fold CPU's
    // 12102203*x constant to -2^23*u, with the same integer bias and
    // truncation toward zero. The -80 natural-exponent tail is zeroed.
    // Ordered comparisons reject NaN, negative distance, and infinity;
    // conversion is bounded and invalid bit patterns never enter FP math.
    constexpr float cutoff = 80.0f/0.6931471805599453f;
    const bool kept = u >= 0.0f && u <= cutoff;
    const float bounded = fminf(fmaxf(u, 0.0f), cutoff);
    const int bits = __float2int_rz(-8388608.0f*bounded) +
                     ((127 << 23) - 366393);
    return __int_as_float(kept ? bits : 0);
#endif
}

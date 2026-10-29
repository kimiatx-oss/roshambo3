// AArch64 NEON instantiation of the fixed-pose final-rescore grains: four
// FP32 lanes. exp is the shared polynomial (optimizer_exp_neon.hpp), the
// same expansion as final_detail::polynomial. This TU is compiled only on AArch64; final_rescore_avx2.cpp is the x86 twin.
#include "final_rescore_simd_impl.hpp"
#include "optimizer_exp_neon.hpp"

namespace r3::cpu {
namespace {
// The minimal vector interface final_rescore_simd_impl.hpp is written against.
struct Vector4 {
    using Type = float32x4_t;
    static constexpr int width = 4;

    static Type zero() {
        return vdupq_n_f32(0);
    }

    static Type broadcast(float x) {
        return vdupq_n_f32(x);
    }

    static Type load(const float* x) {
        return vld1q_f32(x);
    }

    static void store(float* x, Type v) {
        vst1q_f32(x, v);
    }

    static Type add(Type a, Type b) {
        return vaddq_f32(a, b);
    }

    static Type mul(Type a, Type b) {
        return vmulq_f32(a, b);
    }

    // a * b + c, single rounding. NEON takes the addend first.
    static Type fma(Type a, Type b, Type c) {
        return vfmaq_f32(c, a, b);
    }

    static Type exp(Type x) {
        return neon_exp_polynomial(x);
    }

    // Horizontal sum of the four lanes; the within-pair grain's reassociation.
    static float reduce(Type x) {
        return vaddvq_f32(x);
    }
};
}

void final_rescore_simd(const FinalRescorePair* p, size_t n,
                        FinalRescoreCross* out, FinalRescoreMode mode,
                        FinalRescoreCounts* c) {
    final_detail::run<Vector4>(p, n, out, mode, c);
}
}

// x86 AVX2 instantiation of the fixed-pose final-rescore grains: eight FP32
// lanes. Built on every non-AArch64 target in place of final_rescore_neon.cpp.
// No x86 mode beat the accurate control.
#include "final_rescore_simd_impl.hpp"
#include <immintrin.h>

namespace r3::cpu {
namespace {
// The minimal vector interface final_rescore_simd_impl.hpp is written against.
struct Vector8 {
    using Type = __m256;
    static constexpr int width = 8;

    static Type zero() {
        return _mm256_setzero_ps();
    }

    static Type broadcast(float x) {
        return _mm256_set1_ps(x);
    }

    static Type load(const float* x) {
        return _mm256_loadu_ps(x);
    }

    static void store(float* x, Type v) {
        _mm256_storeu_ps(x, v);
    }

    static Type add(Type a, Type b) {
        return _mm256_add_ps(a, b);
    }

    static Type mul(Type a, Type b) {
        return _mm256_mul_ps(a, b);
    }

    static Type fma(Type a, Type b, Type c) {
        return _mm256_fmadd_ps(a, b, c);
    }

    // The AVX2 spelling of final_detail::polynomial and neon_exp_polynomial:
    // same constants, same Horner order, same fma per step. Inputs below -80
    // are clamped before the int conversion (so it cannot overflow) and then
    // masked to exactly 0; `kept` is false for NaN as well (ordered compare).
    // `n` is already integral, so truncating cvttps is exact here.
    static Type exp(Type x) {
        const auto kept = _mm256_cmp_ps(x, broadcast(-80), _CMP_GE_OQ);
        const auto y = mul(_mm256_max_ps(x, broadcast(-80)),
                           broadcast(1.4426950408889634074f));
        const auto n = _mm256_floor_ps(y),
                   z = mul(_mm256_sub_ps(y, n),
                           broadcast(.69314718055994530942f));
        const auto bits = _mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)),
            23);
        constexpr float c[] = {1,        1,         .5f,       1.f / 6,
                               1.f / 24, 1.f / 120, 1.f / 720, 1.f / 5040};
        auto p = broadcast(c[7]);
        for (int k = 6; k >= 0; --k)
            p = fma(p, z, broadcast(c[k]));
        return _mm256_and_ps(kept, mul(p, _mm256_castsi256_ps(bits)));
    }

    // Horizontal sum: fold 256 -> 128 bits, then two pairwise hadds.
    static float reduce(Type x) {
        auto a =
            _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
        a = _mm_hadd_ps(a, a);
        a = _mm_hadd_ps(a, a);
        return _mm_cvtss_f32(a);
    }
};
}

void final_rescore_simd(const FinalRescorePair* p, size_t n,
                        FinalRescoreCross* out, FinalRescoreMode mode,
                        FinalRescoreCounts* c) {
    final_detail::run<Vector8>(p, n, out, mode, c);
}
}

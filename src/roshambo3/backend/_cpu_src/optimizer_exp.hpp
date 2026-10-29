#pragma once

#include <stdexcept>
#include <string>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace r3::cpu {

// Immutable call state. This mode changes only eligible canonical optimizer
// Gaussian weights; exact per-start/winner scoring and general matrices retain
// their existing evaluators. It is never stored in process-global state.
enum class OptimizerExpMode { Baseline, RawBits, RawBitsUnclamped };

inline OptimizerExpMode parse_optimizer_exp_mode(const std::string& mode) {
    if (mode == "baseline") return OptimizerExpMode::Baseline;
    if (mode == "rawbits") return OptimizerExpMode::RawBits;
    if (mode == "rawbits_unclamped") return OptimizerExpMode::RawBitsUnclamped;
    throw std::invalid_argument(
        "optimizer_exp_mode must be baseline or rawbits (or rawbits_unclamped)");
}

// Resolve once at native call entry, never inside an optimizer iteration.
inline OptimizerExpMode resolve_optimizer_exp_mode(OptimizerExpMode requested,
                                                   bool unclamped_is_safe) {
    return requested == OptimizerExpMode::RawBitsUnclamped && !unclamped_is_safe
               ? OptimizerExpMode::RawBits
               : requested;
}

#if defined(__AVX2__)
// Reviewed raw-bit candidate: finite canonical x in [-80,0], multiply,
// truncate toward zero, then INTEGER bias. Spatial forces use this approximate
// Gaussian weight, not the derivative of this piecewise approximation.
struct RawBitsOptimizerExp {
    inline __m256 operator()(__m256 x) const {
        const __m256 cutoff = _mm256_set1_ps(-80.0f);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 kept = _mm256_and_ps(_mm256_cmp_ps(x, cutoff, _CMP_GE_OQ),
                                          _mm256_cmp_ps(x, zero, _CMP_LE_OQ));
        const __m256 bounded = _mm256_min_ps(_mm256_max_ps(x, cutoff), zero);
        const __m256 scaled =
            _mm256_mul_ps(bounded, _mm256_set1_ps(12102203.f));
        const __m256i bits =
            _mm256_add_epi32(_mm256_cvttps_epi32(scaled),
                             _mm256_set1_epi32((127 << 23) - 366393));
        return _mm256_and_ps(_mm256_castsi256_ps(bits), kept);
    }
};

// Nonpositive Gaussian exponents only. The caller checks masked invalid
// exceptions once before scoring; this helper never reads/writes MXCSR.
// Out-of-range conversions may set INVALID, but all tail lanes are bitwise
// zeroed before their intermediate (possibly NaN) bits reach FP arithmetic.
struct RawBitsUnclampedOptimizerExp {
    inline __m256 operator()(__m256 x) const {
        const __m256 kept =
            _mm256_cmp_ps(x, _mm256_set1_ps(-80.0f), _CMP_GE_OQ);
        const __m256 scaled = _mm256_mul_ps(x, _mm256_set1_ps(12102203.f));
        const __m256i bits =
            _mm256_add_epi32(_mm256_cvttps_epi32(scaled),
                             _mm256_set1_epi32((127 << 23) - 366393));
        return _mm256_castsi256_ps(
            _mm256_and_si256(bits, _mm256_castps_si256(kept)));
    }
};
#endif
}  // namespace r3::cpu

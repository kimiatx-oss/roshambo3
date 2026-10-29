// Y-Z-X evaluation of the unchanged node-major trilinear field.
#pragma once

#if defined(__AVX2__)

#include <immintrin.h>
#include <cstdint>

namespace r3 {
namespace cpu {

// Each row load is [x0.float4 | x1.float4] for one sample. During Y/Z
// interpolation both halves belong to the same sample, so replicate that
// sample's fraction across them. The final X lerp uses the original pair
// fraction, whose halves belong to samples A and B respectively.
static inline __m256 xlast_repeat_low(__m256 value) {
    return _mm256_permute2f128_ps(value, value, 0x00);
}

static inline __m256 xlast_repeat_high(__m256 value) {
    return _mm256_permute2f128_ps(value, value, 0x11);
}

// `a00` etc. point at the x0 node of each row; x1 is the following four
// channels. Float rows load as one YMM. Half rows load eight adjacent halves
// and widen to the identical FP32 arithmetic path through `load_pair`.
template <class T>
static inline __m256 interp_pair_xlast(const T* a00, const T* a10, const T* a01,
                                       const T* a11, const T* b00, const T* b10,
                                       const T* b01, const T* b11, __m256 wx,
                                       __m256 wy, __m256 wz) {
    const __m256 wya = xlast_repeat_low(wy);
    const __m256 wyb = xlast_repeat_high(wy);
    const __m256 wza = xlast_repeat_low(wz);
    const __m256 wzb = xlast_repeat_high(wz);

    const __m256 ay0 = lerp256(load_pair(a00), load_pair(a10), wya);
    const __m256 ay1 = lerp256(load_pair(a01), load_pair(a11), wya);
    const __m256 by0 = lerp256(load_pair(b00), load_pair(b10), wyb);
    const __m256 by1 = lerp256(load_pair(b01), load_pair(b11), wyb);
    const __m256 a_yz = lerp256(ay0, ay1, wza);  // [A.x0 | A.x1]
    const __m256 b_yz = lerp256(by0, by1, wzb);  // [B.x0 | B.x1]

    const __m256 x0 = _mm256_permute2f128_ps(a_yz, b_yz, 0x20);
    const __m256 x1 = _mm256_permute2f128_ps(a_yz, b_yz, 0x31);
    return lerp256(x0, x1, wx);  // [A.rgba | B.rgba]
}

}  // namespace cpu
}  // namespace r3

#endif  // __AVX2__

// Format-3 codec: the AVX2 BP256-16 unpack (x86-64 only; the extension's
// baseline is x86-64-v3, so AVX2 needs no per-file flag).
//
// One 256-bit word holds one u16 of each of the 16 lanes, so output vector k
// (values 16k .. 16k+15 of a group) is word (k*W)/16 shifted right by
// (k*W)%16, OR the next word shifted left when the value straddles two words,
// masked to W bits, plus the frame minimum -- a 16-bit add, which is the
// modulo-2^16 wrap of FORMAT.md 5.6. Every shift is a compile-time constant
// (K and W are template parameters); the width
// selects a body through a checked switch, never a function table indexed by
// a file byte. Loads are unaligned-safe and touch only the group's W words.
#include "r3g_unpack.hpp"

#if !defined(__AVX2__)
#error "r3g_unpack_avx2.cpp must be compiled with AVX2 enabled (-march=x86-64-v3)"
#endif

#include <immintrin.h>

#include <utility>

namespace r3 {
namespace r3g {
namespace detail {
namespace {

template <unsigned W, unsigned K>
inline void avx2_step(const __m256i* words, __m256i vmn, __m256i msk,
                      std::int16_t* dst) noexcept {
    constexpr unsigned b = K * W, i = b >> 4, sh = b & 15;
    __m256i v;
    if constexpr (sh == 0) {
        v = words[i];
    } else {
        v = _mm256_srli_epi16(words[i], static_cast<int>(sh));
    }
    if constexpr (sh + W > 16) {
        v = _mm256_or_si256(v, _mm256_slli_epi16(words[i + 1], static_cast<int>(16 - sh)));
    }
    if constexpr (W < 16) v = _mm256_and_si256(v, msk);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 16 * K), _mm256_add_epi16(v, vmn));
}

template <unsigned W, unsigned... K>
inline void avx2_group(const std::uint8_t* s, __m256i vmn, __m256i msk,
                       std::int16_t* dst, std::integer_sequence<unsigned, K...>) noexcept {
    __m256i words[W];
    for (unsigned i = 0; i < W; ++i)
        words[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 32 * i));
    (avx2_step<W, K>(words, vmn, msk, dst), ...);
}

template <unsigned W>
void unpack_avx2_w(const std::uint8_t* s, std::size_t groups, std::int16_t mn,
                   std::int16_t* dst) noexcept {
    const __m256i vmn = _mm256_set1_epi16(mn);
    if constexpr (W == 0) {
        for (std::size_t g = 0; g < groups; ++g, dst += 256)
            for (unsigned k = 0; k < 16; ++k)
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 16 * k), vmn);
    } else {
        const __m256i msk = _mm256_set1_epi16(
            static_cast<short>(W >= 16 ? 0xFFFF : ((1u << W) - 1u)));
        for (std::size_t g = 0; g < groups; ++g, s += W * 32, dst += 256)
            avx2_group<W>(s, vmn, msk, dst, std::make_integer_sequence<unsigned, 16>{});
    }
}

}  // namespace

void heavy_rows_avx2(const std::int16_t* axes, std::size_t stride, const RowRun* runs,
                     std::size_t n_runs, std::size_t L, std::int16_t* out) noexcept {
    if ((L & 7) != 0) {
        heavy_rows_scalar(axes, stride, runs, n_runs, L, out);
        return;
    }
    const __m256i iota16 = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m128i iota8 = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
    const std::size_t L16 = L & ~std::size_t(15);
    std::size_t pos = 0;
    for (std::size_t m = 0; m < n_runs; ++m) {
        const std::size_t n = runs[m].n_real;
        const __m256i vn16 = _mm256_set1_epi16(static_cast<short>(n));
        const __m128i vn8 = _mm_set1_epi16(static_cast<short>(n));
        for (std::size_t j = 0; j < runs[m].n_conf; ++j, pos += n) {
            for (int a = 0; a < 3; ++a, out += L) {
                const std::int16_t* src = axes + a * stride + pos;
                std::size_t c = 0;
                for (; c < L16; c += 16) {
                    const __m256i idx = _mm256_add_epi16(iota16, _mm256_set1_epi16(static_cast<short>(c)));
                    const __m256i keep = _mm256_cmpgt_epi16(vn16, idx);
                    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + c), _mm256_and_si256(v, keep));
                }
                if (c < L) {  // one 8-value tail (L % 16 == 8)
                    const __m128i idx = _mm_add_epi16(iota8, _mm_set1_epi16(static_cast<short>(c)));
                    const __m128i keep = _mm_cmpgt_epi16(vn8, idx);
                    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + c));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + c), _mm_and_si128(v, keep));
                }
            }
        }
    }
}

bool unpack_axis_avx2(const std::uint8_t* s, std::size_t groups, unsigned w,
                      std::int16_t mn, std::int16_t* dst) noexcept {
    switch (w) {
        case 0: unpack_avx2_w<0>(s, groups, mn, dst); return true;
        case 1: unpack_avx2_w<1>(s, groups, mn, dst); return true;
        case 2: unpack_avx2_w<2>(s, groups, mn, dst); return true;
        case 3: unpack_avx2_w<3>(s, groups, mn, dst); return true;
        case 4: unpack_avx2_w<4>(s, groups, mn, dst); return true;
        case 5: unpack_avx2_w<5>(s, groups, mn, dst); return true;
        case 6: unpack_avx2_w<6>(s, groups, mn, dst); return true;
        case 7: unpack_avx2_w<7>(s, groups, mn, dst); return true;
        case 8: unpack_avx2_w<8>(s, groups, mn, dst); return true;
        case 9: unpack_avx2_w<9>(s, groups, mn, dst); return true;
        case 10: unpack_avx2_w<10>(s, groups, mn, dst); return true;
        case 11: unpack_avx2_w<11>(s, groups, mn, dst); return true;
        case 12: unpack_avx2_w<12>(s, groups, mn, dst); return true;
        case 13: unpack_avx2_w<13>(s, groups, mn, dst); return true;
        case 14: unpack_avx2_w<14>(s, groups, mn, dst); return true;
        case 15: unpack_avx2_w<15>(s, groups, mn, dst); return true;
        case 16: unpack_avx2_w<16>(s, groups, mn, dst); return true;
        default: return false;
    }
}

}  // namespace detail
}  // namespace r3g
}  // namespace r3

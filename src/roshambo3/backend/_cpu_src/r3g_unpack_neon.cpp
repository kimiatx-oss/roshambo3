// Format-3 codec: the NEON BP256-16 unpack (aarch64 only; armv8-a+simd is the
// extension's baseline there).
//
// The same arithmetic as r3g_unpack_avx2.cpp on two 128-bit halves per
// 256-bit word: lanes 0-7 are bytes [0, 16) of the word, lanes 8-15 bytes
// [16, 32). Shift amounts are compile-time template constants; vshrq_n_u16
// takes 1..16 and vshlq_n_u16 0..15, so a zero right shift is skipped and the
// left shift (16 - sh) is only formed for sh in 1..15. Loads go through
// vld1q_u8 (byte addressed, no alignment) and a reinterpretation, which is the
// little-endian u16 of FORMAT.md 5.3 on aarch64. The 16-bit unsigned add is
// the modulo-2^16 wrap of 5.6.
#include "r3g_unpack.hpp"

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error "r3g_unpack_neon.cpp is the aarch64 unpack path"
#endif

#include <arm_neon.h>

#include <utility>

namespace r3 {
namespace r3g {
namespace detail {
namespace {

template <unsigned SH>
inline uint16x8_t neon_shr(uint16x8_t v) noexcept {
    if constexpr (SH == 0) return v;
    else return vshrq_n_u16(v, SH);
}

template <unsigned W, unsigned K>
inline void neon_step(const uint16x8_t* lo, const uint16x8_t* hi, uint16x8_t vmn,
                      uint16x8_t msk, std::int16_t* dst) noexcept {
    constexpr unsigned b = K * W, i = b >> 4, sh = b & 15;
    uint16x8_t a = neon_shr<sh>(lo[i]);
    uint16x8_t c = neon_shr<sh>(hi[i]);
    if constexpr (sh + W > 16) {
        a = vorrq_u16(a, vshlq_n_u16(lo[i + 1], 16 - sh));
        c = vorrq_u16(c, vshlq_n_u16(hi[i + 1], 16 - sh));
    }
    if constexpr (W < 16) {
        a = vandq_u16(a, msk);
        c = vandq_u16(c, msk);
    }
    vst1q_s16(dst + 16 * K, vreinterpretq_s16_u16(vaddq_u16(a, vmn)));
    vst1q_s16(dst + 16 * K + 8, vreinterpretq_s16_u16(vaddq_u16(c, vmn)));
}

template <unsigned W, unsigned... K>
inline void neon_group(const std::uint8_t* s, uint16x8_t vmn, uint16x8_t msk,
                       std::int16_t* dst, std::integer_sequence<unsigned, K...>) noexcept {
    uint16x8_t lo[W], hi[W];
    for (unsigned i = 0; i < W; ++i) {
        lo[i] = vreinterpretq_u16_u8(vld1q_u8(s + 32 * i));
        hi[i] = vreinterpretq_u16_u8(vld1q_u8(s + 32 * i + 16));
    }
    (neon_step<W, K>(lo, hi, vmn, msk, dst), ...);
}

template <unsigned W>
void unpack_neon_w(const std::uint8_t* s, std::size_t groups, std::int16_t mn,
                   std::int16_t* dst) noexcept {
    // The add is done on unsigned lanes: the same bits as a 16-bit two's
    // complement add (the 5.6 wrap), without a signed-overflow operation
    // (GCC's vaddq_s16 is a plain signed vector `+`, which UBSan flags).
    const uint16x8_t vmn = vdupq_n_u16(static_cast<std::uint16_t>(mn));
    if constexpr (W == 0) {
        for (std::size_t g = 0; g < groups; ++g, dst += 256)
            for (unsigned k = 0; k < 32; ++k) vst1q_s16(dst + 8 * k, vreinterpretq_s16_u16(vmn));
    } else {
        const uint16x8_t msk = vdupq_n_u16(
            static_cast<std::uint16_t>(W >= 16 ? 0xFFFFu : ((1u << W) - 1u)));
        for (std::size_t g = 0; g < groups; ++g, s += W * 32, dst += 256)
            neon_group<W>(s, vmn, msk, dst, std::make_integer_sequence<unsigned, 16>{});
    }
}

}  // namespace

void heavy_rows_neon(const std::int16_t* axes, std::size_t stride, const RowRun* runs,
                     std::size_t n_runs, std::size_t L, std::int16_t* out) noexcept {
    if ((L & 7) != 0) {
        heavy_rows_scalar(axes, stride, runs, n_runs, L, out);
        return;
    }
    const std::uint16_t iota_v[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint16x8_t iota = vld1q_u16(iota_v);
    std::size_t pos = 0;
    for (std::size_t m = 0; m < n_runs; ++m) {
        const std::size_t n = runs[m].n_real;
        const uint16x8_t vn = vdupq_n_u16(static_cast<std::uint16_t>(n));
        for (std::size_t j = 0; j < runs[m].n_conf; ++j, pos += n) {
            for (int a = 0; a < 3; ++a, out += L) {
                const std::int16_t* src = axes + a * stride + pos;
                for (std::size_t c = 0; c < L; c += 8) {
                    const uint16x8_t keep = vcltq_u16(vaddq_u16(iota, vdupq_n_u16(static_cast<std::uint16_t>(c))), vn);
                    const uint16x8_t v = vreinterpretq_u16_s16(vld1q_s16(src + c));
                    vst1q_s16(out + c, vreinterpretq_s16_u16(vandq_u16(v, keep)));
                }
            }
        }
    }
}

bool unpack_axis_neon(const std::uint8_t* s, std::size_t groups, unsigned w,
                      std::int16_t mn, std::int16_t* dst) noexcept {
    switch (w) {
        case 0: unpack_neon_w<0>(s, groups, mn, dst); return true;
        case 1: unpack_neon_w<1>(s, groups, mn, dst); return true;
        case 2: unpack_neon_w<2>(s, groups, mn, dst); return true;
        case 3: unpack_neon_w<3>(s, groups, mn, dst); return true;
        case 4: unpack_neon_w<4>(s, groups, mn, dst); return true;
        case 5: unpack_neon_w<5>(s, groups, mn, dst); return true;
        case 6: unpack_neon_w<6>(s, groups, mn, dst); return true;
        case 7: unpack_neon_w<7>(s, groups, mn, dst); return true;
        case 8: unpack_neon_w<8>(s, groups, mn, dst); return true;
        case 9: unpack_neon_w<9>(s, groups, mn, dst); return true;
        case 10: unpack_neon_w<10>(s, groups, mn, dst); return true;
        case 11: unpack_neon_w<11>(s, groups, mn, dst); return true;
        case 12: unpack_neon_w<12>(s, groups, mn, dst); return true;
        case 13: unpack_neon_w<13>(s, groups, mn, dst); return true;
        case 14: unpack_neon_w<14>(s, groups, mn, dst); return true;
        case 15: unpack_neon_w<15>(s, groups, mn, dst); return true;
        case 16: unpack_neon_w<16>(s, groups, mn, dst); return true;
        default: return false;
    }
}

}  // namespace detail
}  // namespace r3g
}  // namespace r3

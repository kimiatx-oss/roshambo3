// Format-3 codec: the scalar BP256-16 unpack, the reference path.
//
// A direct transcription of FORMAT.md 5.3: value (g, k, l) is bits
// [k*w, (k+1)*w) of lane l's bitstream, which is the lane's u16 of words
// 0..w-1 of group g, least significant bit first. A value spans at most two
// words (w <= 16), the second only when the shift plus the width passes 16;
// the second word's index is then (k*w + w - 1) / 16 <= w - 1, inside the group.
// Bit-identical with reference.unpack_axis and the SIMD paths, including the
// modulo-2^16 wrap of 5.6.
#include "r3g_unpack.hpp"

#include <cstring>

namespace r3 {
namespace r3g {
namespace detail {
namespace {

inline std::uint32_t u16_at(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8);
}

template <unsigned W>
void unpack_scalar_w(const std::uint8_t* s, std::size_t groups, std::int16_t mn,
                     std::int16_t* dst) noexcept {
    constexpr std::uint32_t mask = W >= 16 ? 0xFFFFu : ((1u << W) - 1u);
    for (std::size_t g = 0; g < groups; ++g, s += W * 32, dst += 256) {
        for (unsigned k = 0; k < 16; ++k) {
            const unsigned b = k * W, i = b >> 4, sh = b & 15;
            for (unsigned l = 0; l < 16; ++l) {
                std::uint32_t u = 0;
                if (W != 0) {
                    u = u16_at(s + 32 * i + 2 * l) >> sh;
                    if (sh + W > 16) u |= u16_at(s + 32 * (i + 1) + 2 * l) << (16 - sh);
                    u &= mask;
                }
                dst[16 * k + l] = wrap16(mn, u);
            }
        }
    }
}

}  // namespace

void heavy_rows_scalar(const std::int16_t* axes, std::size_t stride, const RowRun* runs,
                       std::size_t n_runs, std::size_t L, std::int16_t* out) noexcept {
    std::size_t pos = 0;
    for (std::size_t m = 0; m < n_runs; ++m) {
        const std::size_t n = runs[m].n_real;
        for (std::size_t j = 0; j < runs[m].n_conf; ++j, pos += n) {
            for (int a = 0; a < 3; ++a, out += L) {
                std::memcpy(out, axes + a * stride + pos, n * sizeof(std::int16_t));
                std::memset(out + n, 0, (L - n) * sizeof(std::int16_t));
            }
        }
    }
}

bool unpack_axis_scalar(const std::uint8_t* s, std::size_t groups, unsigned w,
                        std::int16_t mn, std::int16_t* dst) noexcept {
    switch (w) {
        case 0: unpack_scalar_w<0>(s, groups, mn, dst); return true;
        case 1: unpack_scalar_w<1>(s, groups, mn, dst); return true;
        case 2: unpack_scalar_w<2>(s, groups, mn, dst); return true;
        case 3: unpack_scalar_w<3>(s, groups, mn, dst); return true;
        case 4: unpack_scalar_w<4>(s, groups, mn, dst); return true;
        case 5: unpack_scalar_w<5>(s, groups, mn, dst); return true;
        case 6: unpack_scalar_w<6>(s, groups, mn, dst); return true;
        case 7: unpack_scalar_w<7>(s, groups, mn, dst); return true;
        case 8: unpack_scalar_w<8>(s, groups, mn, dst); return true;
        case 9: unpack_scalar_w<9>(s, groups, mn, dst); return true;
        case 10: unpack_scalar_w<10>(s, groups, mn, dst); return true;
        case 11: unpack_scalar_w<11>(s, groups, mn, dst); return true;
        case 12: unpack_scalar_w<12>(s, groups, mn, dst); return true;
        case 13: unpack_scalar_w<13>(s, groups, mn, dst); return true;
        case 14: unpack_scalar_w<14>(s, groups, mn, dst); return true;
        case 15: unpack_scalar_w<15>(s, groups, mn, dst); return true;
        case 16: unpack_scalar_w<16>(s, groups, mn, dst); return true;
        default: return false;
    }
}

}  // namespace detail
}  // namespace r3g
}  // namespace r3

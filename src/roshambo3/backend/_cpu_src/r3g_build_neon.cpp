// Exact integer coordinate construction and bounded layout conversion for ARM64.
#include "r3g_build.hpp"
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#include <cstring>

namespace r3 { namespace r3g { namespace detail {
namespace {
// Eight rows of eight int16 values become eight columns. All loads/stores at
// the call sites are unaligned and stay within a validated segment or tile.
inline void transpose8(int16x8_t (&v)[8]) noexcept {
    const auto a0 = vtrnq_s16(v[0], v[1]);
    const auto a1 = vtrnq_s16(v[2], v[3]);
    const auto a2 = vtrnq_s16(v[4], v[5]);
    const auto a3 = vtrnq_s16(v[6], v[7]);
    const auto b0 = vtrnq_s32(vreinterpretq_s32_s16(a0.val[0]), vreinterpretq_s32_s16(a1.val[0]));
    const auto b1 = vtrnq_s32(vreinterpretq_s32_s16(a0.val[1]), vreinterpretq_s32_s16(a1.val[1]));
    const auto b2 = vtrnq_s32(vreinterpretq_s32_s16(a2.val[0]), vreinterpretq_s32_s16(a3.val[0]));
    const auto b3 = vtrnq_s32(vreinterpretq_s32_s16(a2.val[1]), vreinterpretq_s32_s16(a3.val[1]));
    const int16x8_t lo[4] = {vreinterpretq_s16_s32(b0.val[0]), vreinterpretq_s16_s32(b1.val[0]),
                              vreinterpretq_s16_s32(b0.val[1]), vreinterpretq_s16_s32(b1.val[1])};
    const int16x8_t hi[4] = {vreinterpretq_s16_s32(b2.val[0]), vreinterpretq_s16_s32(b3.val[0]),
                              vreinterpretq_s16_s32(b2.val[1]), vreinterpretq_s16_s32(b3.val[1])};
    for (unsigned i = 0; i < 4; ++i) {
        v[i] = vcombine_s16(vget_low_s16(lo[i]), vget_low_s16(hi[i]));
        v[i + 4] = vcombine_s16(vget_high_s16(lo[i]), vget_high_s16(hi[i]));
    }
}

template<unsigned K>
inline int16x4_t mean4(int32x4_t sum) noexcept {
    // |sum| <= 16*32768, so N <= 1048592 and every intermediate fits
    // uint32/int32. The reciprocal estimate is at most one below floor(N/D):
    // N < 2^32, and floor(2^32/D) has an error strictly less than one.
    constexpr unsigned D = 2 * K;
    const uint32x4_t n = vaddq_u32(vshlq_n_u32(vreinterpretq_u32_s32(vabsq_s32(sum)), 1), vdupq_n_u32(K));
    uint32x4_t q;
    if constexpr ((D & (D - 1)) == 0) {
        constexpr unsigned shift = K == 1 ? 1 : K == 2 ? 2 : K == 4 ? 3 : K == 8 ? 4 : 5;
        q = vshrq_n_u32(n, shift);
    } else {
        constexpr std::uint32_t reciprocal = static_cast<std::uint32_t>((std::uint64_t{1} << 32) / D);
        const uint32x2_t qlo = vshrn_n_u64(vmull_u32(vget_low_u32(n), vdup_n_u32(reciprocal)), 32);
        const uint32x2_t qhi = vshrn_n_u64(vmull_u32(vget_high_u32(n), vdup_n_u32(reciprocal)), 32);
        q = vcombine_u32(qlo, qhi);
        const uint32x4_t remainder = vsubq_u32(n, vmulq_n_u32(q, D));
        q = vaddq_u32(q, vandq_u32(vcgeq_u32(remainder, vdupq_n_u32(D)), vdupq_n_u32(1)));
    }
    const int32x4_t signed_q = vreinterpretq_s32_u32(q);
    return vmovn_s32(vbslq_s32(vcltq_s32(sum, vdupq_n_s32(0)), vnegq_s32(signed_q), signed_q));
}

template<unsigned K>
void feature(std::int16_t* tile, std::size_t slot, const std::uint8_t* atoms,
             std::size_t begin, std::size_t lanes) noexcept {
    for (unsigned axis = 0; axis < 3; ++axis) {
        auto* dst = tile + (slot * 3 + axis) * kBuildLanes + begin;
        std::size_t lane = 0;
        if constexpr (K == 1) {
            const auto* src = tile + (std::size_t(atoms[0]) * 3 + axis) * kBuildLanes + begin;
            for (; lane + 8 <= lanes; lane += 8) vst1q_s16(dst + lane, vld1q_s16(src + lane));
            for (; lane < lanes; ++lane) dst[lane] = src[lane];
        } else {
            for (; lane + 4 <= lanes; lane += 4) {
                int32x4_t sum = vdupq_n_s32(0);
                for (unsigned atom = 0; atom < K; ++atom) {
                    const auto* src = tile + (std::size_t(atoms[atom]) * 3 + axis) * kBuildLanes + begin + lane;
                    sum = vaddq_s32(sum, vmovl_s16(vld1_s16(src)));
                }
                vst1_s16(dst + lane, mean4<K>(sum));
            }
            for (; lane < lanes; ++lane) {
                std::int32_t sum = 0;
                for (unsigned atom = 0; atom < K; ++atom)
                    sum += tile[(std::size_t(atoms[atom]) * 3 + axis) * kBuildLanes + begin + lane];
                dst[lane] = builder_mean(sum, K);
            }
        }
    }
}

template<unsigned FractionBits>
inline float32x4_t convert4(int16x4_t v, float step) noexcept {
    const int32x4_t wide = vmovl_s16(v);
    if constexpr (FractionBits != 0) return vcvtq_n_f32_s32(wide, FractionBits);
    else return vmulq_n_f32(vcvtq_f32_s32(wide), step);
}

template<unsigned FractionBits>
void emit_cpu(const std::int16_t* tile, std::size_t L, std::size_t lanes,
              float step, float* out) noexcept {
    // Transpose eight slots at a time into each row's slot-major run before
    // widening. Only actual lanes and slots are stored, including both tails.
    for (std::size_t slot = 0; slot < L; slot += 8) {
        const std::size_t slots = L - slot < 8 ? L - slot : 8;
        for (std::size_t lane = 0; lane < lanes; lane += 8) {
            const std::size_t rows = lanes - lane < 8 ? lanes - lane : 8;
            int16x8_t xyz[3][8];
            for (unsigned axis = 0; axis < 3; ++axis) {
                for (unsigned s = 0; s < 8; ++s) {
                    if (s >= slots) xyz[axis][s] = vdupq_n_s16(0);
                    else {
                        const auto* src = tile + ((slot + s) * 3 + axis) * kBuildLanes + lane;
                        if (rows == 8) xyz[axis][s] = vld1q_s16(src);
                        else {
                            // CPU callers need not initialize inactive rows.
                            std::int16_t tail[8]{};
                            std::memcpy(tail, src, rows * sizeof(*src));
                            xyz[axis][s] = vld1q_s16(tail);
                        }
                    }
                }
                transpose8(xyz[axis]);
            }
            for (std::size_t row = 0; row < rows; ++row) {
                float* dst = out + ((lane + row) * L + slot) * 3;
                float32x4x3_t value;
                for (unsigned axis = 0; axis < 3; ++axis)
                    value.val[axis] = convert4<FractionBits>(vget_low_s16(xyz[axis][row]), step);
                if (slots >= 4) vst3q_f32(dst, value);
                else {
                    float tmp[12];
                    vst3q_f32(tmp, value);
                    std::memcpy(dst, tmp, slots * 3 * sizeof(float));
                }
                if (slots > 4) {
                    for (unsigned axis = 0; axis < 3; ++axis)
                        value.val[axis] = convert4<FractionBits>(vget_high_s16(xyz[axis][row]), step);
                    if (slots == 8) vst3q_f32(dst + 12, value);
                    else {
                        float tmp[12];
                        vst3q_f32(tmp, value);
                        std::memcpy(dst + 12, tmp, (slots - 4) * 3 * sizeof(float));
                    }
                }
            }
        }
    }
}
} // namespace

void build_heavy_tile_neon(const std::int16_t* axes, std::size_t axis_stride,
    std::size_t pos, std::size_t n_real, std::size_t lane_begin, std::size_t lanes,
    std::int16_t* tile) noexcept {
    for (unsigned axis = 0; axis < 3; ++axis) {
        const auto* src = axes + axis * axis_stride + pos;
        std::size_t slot = 0;
        for (; slot + 8 <= n_real; slot += 8) {
            std::size_t lane = 0;
            for (; lane + 8 <= lanes; lane += 8) {
                int16x8_t block[8];
                for (unsigned row = 0; row < 8; ++row)
                    block[row] = vld1q_s16(src + (lane + row) * n_real + slot);
                transpose8(block);
                for (unsigned s = 0; s < 8; ++s)
                    vst1q_s16(tile + ((slot + s) * 3 + axis) * kBuildLanes + lane_begin + lane, block[s]);
            }
            for (; lane < lanes; ++lane)
                for (unsigned s = 0; s < 8; ++s)
                    tile[((slot + s) * 3 + axis) * kBuildLanes + lane_begin + lane] = src[lane * n_real + slot + s];
        }
        for (; slot < n_real; ++slot)
            for (std::size_t lane = 0; lane < lanes; ++lane)
                tile[(slot * 3 + axis) * kBuildLanes + lane_begin + lane] = src[lane * n_real + slot];
    }
}

void build_feature_tile_neon(std::int16_t* tile, std::size_t feature_slot,
    const std::uint8_t* atoms, unsigned arity, std::size_t lane_begin, std::size_t lanes) noexcept {
    switch (arity) {
#define R3_FEATURE_CASE(K) case K: feature<K>(tile, feature_slot, atoms, lane_begin, lanes); break
        R3_FEATURE_CASE(1); R3_FEATURE_CASE(2); R3_FEATURE_CASE(3); R3_FEATURE_CASE(4);
        R3_FEATURE_CASE(5); R3_FEATURE_CASE(6); R3_FEATURE_CASE(7); R3_FEATURE_CASE(8);
        R3_FEATURE_CASE(9); R3_FEATURE_CASE(10); R3_FEATURE_CASE(11); R3_FEATURE_CASE(12);
        R3_FEATURE_CASE(13); R3_FEATURE_CASE(14); R3_FEATURE_CASE(15); R3_FEATURE_CASE(16);
#undef R3_FEATURE_CASE
    }
}

void emit_gpu_tile_neon(const std::int16_t* tile, std::size_t L, int tile_mode,
                       std::int16_t* out) noexcept {
    if (tile_mode == -1) {
        if (out != tile) std::memcpy(out, tile, L * 3 * kBuildLanes * sizeof(std::int16_t));
        return;
    }
    for (std::size_t slot = 0; slot < L; ++slot)
        for (std::size_t lane = 0; lane < kBuildLanes; lane += 8) {
            int16x8x3_t value;
            for (unsigned axis = 0; axis < 3; ++axis)
                value.val[axis] = vld1q_s16(tile + (slot * 3 + axis) * kBuildLanes + lane);
            vst3q_s16(out + (slot * kBuildLanes + lane) * 3, value);
        }
}

void emit_cpu_tile_neon(const std::int16_t* tile, std::size_t L, std::size_t lanes,
                       float step, float* out) noexcept {
    emit_cpu<0>(tile, L, lanes, step, out);
}
CpuTileEmitter select_cpu_tile_emitter_neon(float step) noexcept {
    if (step == 0x1p-8f) return emit_cpu<8>;
    if (step == 0x1p-10f) return emit_cpu<10>;
    return emit_cpu_tile_neon;
}
}}} // namespace r3::r3g::detail
#endif

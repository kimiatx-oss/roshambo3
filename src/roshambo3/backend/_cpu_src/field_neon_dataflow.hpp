// The shipped ARM FP32 field sampler: one uint32 base offset per atom, SoA
// prepared block, vector torque and paired stencil trees.
//
// x86's one-base-offset and vector-torque dataflow, ported to NEON:
// address +11.99% and torque +1.50% all-eight, composed (with the two-stream
// color helper) +17.19%. Interleaving two independent stencil trees added
// +0.75% all-eight.
// Each matched the accepted baseline's frozen all-eight output bytes.
//
// Valid only when FieldNaive32::prepare_neon_offsets32 (field.hpp) proved,
// once per query, that every valid corner AND the eight-float zero sentinel
// fit uint32 element offsets and that the cell counts compare exactly in
// FP32. Otherwise callers use the size_t path in field_neon_prepared.hpp.
//
// Invariant: an invalid atom never adds a stride to the sentinel offset. The
// proof covers only an eight-float sentinel, so `sentinel + row` is not known
// to be in bounds; samplers return zero before any corner address is formed.
#pragma once
#include "field_neon_prepared.hpp"

namespace r3::cpu::detail {
// One base element offset, SoA fractions and world levers. Invalid atoms are
// excluded before loads/torque; strides are never added to a short sentinel.
struct BaseBlock32 {
    alignas(16) uint32_t base[8], valid[8];
    alignas(16) float fx[8], fy[8], fz[8], mx[8], my[8], mz[8];
    int count;
};

// Prepare atoms [first, first + 8): the same transform, bounds and fraction
// work as field_neon_prepared.hpp's prepare(), but with packed uint32 base
// arithmetic (legal under the proof) and lever arms zeroed on invalid lanes,
// written as SoA so torque can reload them as vectors.
__attribute__((always_inline)) inline void
prepare_base32(const FieldNaive32& f, const ConformerSoA& b, int first,
               const float m[9], const float t[3], BaseBlock32& block) {
    block.count = std::min(8, b.n - first);
    const uint32_t row = uint32_t(f.g.nodes_x()) * 4,
                   plane = row * uint32_t(f.g.nodes_y());
    const auto zero = vdupq_n_f32(0);
    int j = 0;
    for (; j + 4 <= block.count; j += 4) {
        const auto x = vld1q_f32(b.x + first + j),
                   y = vld1q_f32(b.y + first + j),
                   z = vld1q_f32(b.z + first + j);
        const auto mx =
            vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(y, m[1]), x, m[0]), z, m[2]);
        const auto my =
            vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(y, m[4]), x, m[3]), z, m[5]);
        const auto mz =
            vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(y, m[7]), x, m[6]), z, m[8]);
        const auto tx = vmulq_n_f32(
            vsubq_f32(vaddq_f32(mx, vdupq_n_f32(t[0])), vdupq_n_f32(f.g.ox)),
            f.g.inv_hs);
        const auto ty = vmulq_n_f32(
            vsubq_f32(vaddq_f32(my, vdupq_n_f32(t[1])), vdupq_n_f32(f.g.oy)),
            f.g.inv_hs);
        const auto tz = vmulq_n_f32(
            vsubq_f32(vaddq_f32(mz, vdupq_n_f32(t[2])), vdupq_n_f32(f.g.oz)),
            f.g.inv_hs);
        auto valid =
            vandq_u32(vandq_u32(vcgeq_f32(tx, zero), vcgeq_f32(ty, zero)),
                      vcgeq_f32(tz, zero));
        valid = vandq_u32(
            valid,
            vandq_u32(vandq_u32(vcltq_f32(tx, vdupq_n_f32(float(f.g.nx))),
                                vcltq_f32(ty, vdupq_n_f32(float(f.g.ny)))),
                      vcltq_f32(tz, vdupq_n_f32(float(f.g.nz)))));
        const auto sx = vbslq_f32(valid, tx, zero),
                   sy = vbslq_f32(valid, ty, zero),
                   sz = vbslq_f32(valid, tz, zero);
        const auto ix = vcvtq_u32_f32(sx), iy = vcvtq_u32_f32(sy),
                   iz = vcvtq_u32_f32(sz);
        // base = 4 ix + iy row + iz plane, in elements (four channels/node).
        const auto base = vaddq_u32(
            vshlq_n_u32(ix, 2), vmlaq_n_u32(vmulq_n_u32(iy, row), iz, plane));
        vst1q_u32(
            block.base + j,
            vbslq_u32(valid, base, vdupq_n_u32(uint32_t(f.zero_offset()))));
        vst1q_u32(block.valid + j, valid);
        vst1q_f32(block.fx + j, vsubq_f32(sx, vcvtq_f32_u32(ix)));
        vst1q_f32(block.fy + j, vsubq_f32(sy, vcvtq_f32_u32(iy)));
        vst1q_f32(block.fz + j, vsubq_f32(sz, vcvtq_f32_u32(iz)));
        vst1q_f32(block.mx + j, vbslq_f32(valid, mx, zero));
        vst1q_f32(block.my + j, vbslq_f32(valid, my, zero));
        vst1q_f32(block.mz + j, vbslq_f32(valid, mz, zero));
    }
    // Scalar tail. Explicit std::fma spells the vector lanes' association
    // ((y m1 + x m0) + z m2), so tail atoms round like vector atoms.
    for (; j < block.count; ++j) {
        const int i = first + j;
        const float x = b.x[i], y = b.y[i], z = b.z[i];
        const float mx = std::fma(m[2], z, std::fma(m[0], x, m[1] * y));
        const float my = std::fma(m[5], z, std::fma(m[3], x, m[4] * y));
        const float mz = std::fma(m[8], z, std::fma(m[6], x, m[7] * y));
        const float tx = (mx + t[0] - f.g.ox) * f.g.inv_hs,
                    ty = (my + t[1] - f.g.oy) * f.g.inv_hs,
                    tz = (mz + t[2] - f.g.oz) * f.g.inv_hs;
        const bool valid = tx >= 0 && ty >= 0 && tz >= 0 &&
                           tx < float(f.g.nx) && ty < float(f.g.ny) &&
                           tz < float(f.g.nz);
        block.valid[j] = valid ? UINT32_MAX : 0;
        block.mx[j] = valid ? mx : 0;
        block.my[j] = valid ? my : 0;
        block.mz[j] = valid ? mz : 0;
        if (valid) {
            const uint32_t ix = uint32_t(tx), iy = uint32_t(ty),
                           iz = uint32_t(tz);
            block.base[j] = iz * plane + iy * row + ix * 4;
            block.fx[j] = tx - ix;
            block.fy[j] = ty - iy;
            block.fz[j] = tz - iz;
        } else {
            block.base[j] = uint32_t(f.zero_offset());
            block.fx[j] = block.fy[j] = block.fz[j] = 0;
        }
    }
}

// One atom's sample from its base offset: four x-pair rows at base,
// +row, +plane, +plane+row, each two nodes (eight floats) wide.
inline float32x4_t sample_base32(const FieldNaive32& f, const BaseBlock32& s,
                                 int j, size_t row, size_t plane) {
    if (!s.valid[j]) return vdupq_n_f32(0);
    const float* p = f.v.data() + s.base[j];
    // Explicit loads and the original XYZ tree. No row/endpoints array.
    const auto a = lerp_prepared(vld1q_f32(p), vld1q_f32(p + 4), s.fx[j]);
    const auto b =
        lerp_prepared(vld1q_f32(p + row), vld1q_f32(p + row + 4), s.fx[j]);
    const auto c =
        lerp_prepared(vld1q_f32(p + plane), vld1q_f32(p + plane + 4), s.fx[j]);
    const auto d = lerp_prepared(vld1q_f32(p + plane + row),
                                 vld1q_f32(p + plane + row + 4), s.fx[j]);
    return lerp_prepared(lerp_prepared(a, b, s.fy[j]),
                         lerp_prepared(c, d, s.fy[j]), s.fz[j]);
}

// Keep the two independent XYZ trees in registers. Finish each X row as its
// endpoints arrive, and each Y face before loading the upper Z face. Invalid
// atoms take the single-stencil path before any paired pointer/stride work.
__attribute__((always_inline)) inline void
sample_base32_pair(const FieldNaive32& f, const BaseBlock32& s, int j,
                   size_t row, size_t plane, float32x4_t& first,
                   float32x4_t& second) {
    if (!s.valid[j] || !s.valid[j + 1]) {
        first = sample_base32(f, s, j, row, plane);
        second = sample_base32(f, s, j + 1, row, plane);
        return;
    }
    // Two constant bases sharing the same row/plane offsets: the one
    // linked-code adjustment, after an advancing-pointer form grew the frame.
    const float* const p = f.v.data() + s.base[j];
    const float* const q = f.v.data() + s.base[j + 1];
    const float fx = s.fx[j], qx = s.fx[j + 1];
    const auto a = lerp_prepared(vld1q_f32(p), vld1q_f32(p + 4), fx);
    const auto aa = lerp_prepared(vld1q_f32(q), vld1q_f32(q + 4), qx);
    const auto b =
        lerp_prepared(vld1q_f32(p + row), vld1q_f32(p + row + 4), fx);
    const auto bb =
        lerp_prepared(vld1q_f32(q + row), vld1q_f32(q + row + 4), qx);
    const auto lower = lerp_prepared(a, b, s.fy[j]);
    const auto lower_q = lerp_prepared(aa, bb, s.fy[j + 1]);
    const auto c =
        lerp_prepared(vld1q_f32(p + plane), vld1q_f32(p + plane + 4), fx);
    const auto cc =
        lerp_prepared(vld1q_f32(q + plane), vld1q_f32(q + plane + 4), qx);
    const auto d = lerp_prepared(vld1q_f32(p + plane + row),
                                 vld1q_f32(p + plane + row + 4), fx);
    const auto dd = lerp_prepared(vld1q_f32(q + plane + row),
                                  vld1q_f32(q + plane + row + 4), qx);
    const auto upper = lerp_prepared(c, d, s.fy[j]);
    const auto upper_q = lerp_prepared(cc, dd, s.fy[j + 1]);
    first = lerp_prepared(lower, upper, s.fz[j]);
    second = lerp_prepared(lower_q, upper_q, s.fz[j + 1]);
}

// Keep the accepted full-gradient body literal and inlined. A separate values
// routine must not cause LTO to outline preparation from the shipping loop.
__attribute__((always_inline)) inline void
gradient_base32(const FieldNaive32& f, const ConformerSoA& b, const float m[9],
                const float t[3], float norm, float g[7], float* score) {
    auto sum = vdupq_n_f32(0);
    float torque[3]{};
    BaseBlock32 block;
    const size_t row = size_t(f.g.nodes_x()) * 4,
                 plane = row * size_t(f.g.nodes_y());
    for (int first = 0; first < b.n; first += 8) {
        prepare_base32(f, b, first, m, t, block);
        int j = 0;
#if R3_FIELD_NEON_TORQUE
        for (; j + 4 <= block.count; j += 4) {
#if R3_NEON_PAIRED_STENCILS
            float32x4_t a, bb, c, d;
            sample_base32_pair(f, block, j, row, plane, a, bb);
            sample_base32_pair(f, block, j + 2, row, plane, c, d);
#else
            const auto a = sample_base32(f, block, j, row, plane),
                       bb = sample_base32(f, block, j + 1, row, plane);
            const auto c = sample_base32(f, block, j + 2, row, plane),
                       d = sample_base32(f, block, j + 3, row, plane);
#endif
            sum = vaddq_f32(sum, a);
            sum = vaddq_f32(sum, bb);
            sum = vaddq_f32(sum, c);
            sum = vaddq_f32(sum, d);
            // Transpose four [f, gx, gy, gz] samples into per-channel vectors
            // over atoms j..j+3: TRN1(a, b) = [a.f, b.f, a.gy, b.gy] and
            // TRN2(a, b) = [a.gx, b.gx, a.gz, b.gz]; low/high halves of the
            // two pairs then combine into gx, gy, gz.
            const auto ab0 = vtrn1q_f32(a, bb), ab1 = vtrn2q_f32(a, bb);
            const auto cd0 = vtrn1q_f32(c, d), cd1 = vtrn2q_f32(c, d);
            const auto gx = vmulq_n_f32(
                vcombine_f32(vget_low_f32(ab1), vget_low_f32(cd1)), -norm);
            const auto gy = vmulq_n_f32(
                vcombine_f32(vget_high_f32(ab0), vget_high_f32(cd0)), -norm);
            const auto gz = vmulq_n_f32(
                vcombine_f32(vget_high_f32(ab1), vget_high_f32(cd1)), -norm);
            const auto mx = vmulq_n_f32(vld1q_f32(block.mx + j), 2.f);
            const auto my = vmulq_n_f32(vld1q_f32(block.my + j), 2.f);
            const auto mz = vmulq_n_f32(vld1q_f32(block.mz + j), 2.f);
            // Match scalar fnmsub: the second product is rounded, then
            // the first product minus it is fused. Norm stays per atom.
            alignas(16) float x[4], y[4], z[4];
            vst1q_f32(x, vfmaq_f32(vnegq_f32(vmulq_f32(mz, gy)), my, gz));
            vst1q_f32(y, vfmaq_f32(vnegq_f32(vmulq_f32(mx, gz)), mz, gx));
            vst1q_f32(z, vfmaq_f32(vnegq_f32(vmulq_f32(my, gx)), mx, gy));
            // Fold valid atoms one at a time in source order -- the vector
            // computes products, not the reduction (normalization is not
            // factored outside the sum).
            for (int lane = 0; lane < 4; ++lane) {
                if (!block.valid[j + lane]) continue;
                torque[0] += x[lane];
                torque[1] += y[lane];
                torque[2] += z[lane];
            }
        }
#endif
        for (; j < block.count; ++j) {
            const auto v = sample_base32(f, block, j, row, plane);
            sum = vaddq_f32(sum, v);
            if (!block.valid[j]) continue;
            const float gx = -norm * vgetq_lane_f32(v, 1),
                        gy = -norm * vgetq_lane_f32(v, 2),
                        gz = -norm * vgetq_lane_f32(v, 3);
            torque[0] += 2.f * block.my[j] * gz - 2.f * block.mz[j] * gy;
            torque[1] += 2.f * block.mz[j] * gx - 2.f * block.mx[j] * gz;
            torque[2] += 2.f * block.mx[j] * gy - 2.f * block.my[j] * gx;
        }
    }
    g[0] = 0;
    g[1] = torque[0];
    g[2] = torque[1];
    g[3] = torque[2];
    g[4] = -norm * vgetq_lane_f32(sum, 1);
    g[5] = -norm * vgetq_lane_f32(sum, 2);
    g[6] = -norm * vgetq_lane_f32(sum, 3);
    if (score) *score = vgetq_lane_f32(sum, 0);
}

// Value-only twin of gradient_base32 for the fresh final SEARCH evaluation:
// identical preparation, sample grouping and summation order, so the score
// equals the gradient path's; torque work is omitted.
inline void value_base32(const FieldNaive32& f, const ConformerSoA& b,
                         const float m[9], const float t[3], float* score) {
    auto sum = vdupq_n_f32(0);
    BaseBlock32 block;
    const size_t row = size_t(f.g.nodes_x()) * 4,
                 plane = row * size_t(f.g.nodes_y());
    for (int first = 0; first < b.n; first += 8) {
        prepare_base32(f, b, first, m, t, block);
        int j = 0;
#if R3_FIELD_NEON_TORQUE
        for (; j + 4 <= block.count; j += 4) {
#if R3_NEON_PAIRED_STENCILS
            float32x4_t a, bb, c, d;
            sample_base32_pair(f, block, j, row, plane, a, bb);
            sample_base32_pair(f, block, j + 2, row, plane, c, d);
#else
            const auto a = sample_base32(f, block, j, row, plane),
                       bb = sample_base32(f, block, j + 1, row, plane);
            const auto c = sample_base32(f, block, j + 2, row, plane),
                       d = sample_base32(f, block, j + 3, row, plane);
#endif
            sum = vaddq_f32(sum, a);
            sum = vaddq_f32(sum, bb);
            sum = vaddq_f32(sum, c);
            sum = vaddq_f32(sum, d);
        }
#endif
        for (; j < block.count; ++j)
            sum = vaddq_f32(sum, sample_base32(f, block, j, row, plane));
    }
    if (score) *score = vgetq_lane_f32(sum, 0);
}
}

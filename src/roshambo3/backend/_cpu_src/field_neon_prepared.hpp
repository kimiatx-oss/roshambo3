// The prepared-state NEON field loop: split each eight-atom block into a
// PREPARE phase (pose transform, bounds, cell, fractions, corner offsets,
// lever arms) and a CONSUME phase (corner loads, interpolation, reduction).
//
// This began as the "common prepared-state control" for scheduling
// experiments, and the control itself won: +5.29% eight-query geomean over
// the original FP32 loop and +3.82% over the then-best half policy. The
// overlap variants it was built to enable -- two interleaved trees,
// prepare-one-block-ahead -- both lost to it, so the gain is attributed to
// the prepared loop, not to overlap. PREPARE is vectorized across four atoms
// (+5.89% all-eight).
//
// This is the size_t-addressed path. When the field's uint32 offsets are
// proven at setup, FP32 uses field_neon_dataflow.hpp instead; this file
// remains the fallback for oversized fields and the path for FP16 storage
// (Row policies in field_neon_half_rows.hpp).
#pragma once
#include "field_neon.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <utility>

namespace r3::cpu::detail {
// One atom's prepared sample: four row offsets in ELEMENTS (row = one x-pair
// of nodes, eight floats), the cell fractions, the world-space lever arm M b
// and validity. 64 bytes. A 40-byte one-base record lost 6-9% on the
// complete screen; its linked row loop stored and reloaded endpoint rows
// through the stack.
struct Stencil {
    // prepare() writes every member of every real atom before consume().
    // Unused tails are never read or copied; n=0 never consumes a buffer.
    size_t offset[4];
    float fx, fy, fz, mx, my, mz;
    bool valid;
};

// Eight atoms, matching the public width-eight grain (field_neon.hpp).
struct PreparedBlock {
    Stencil atom[8];
    int count;
};

// Field constants hoisted out of the atom loop, built once per evaluator
// call. `row`/`plane` are element strides; `zero` is the zeroed tail.
struct PreparationGeometry {
    size_t row, plane, zero;
    unsigned nodes_y;
    float ox, oy, oz, inv_h, nx, ny, nz;

    template <class Field>
    explicit PreparationGeometry(const Field& f)
        : row(size_t(f.g.nodes_x()) * 4), plane(row * size_t(f.g.nodes_y())),
          zero(f.zero_offset()), nodes_y(f.g.nodes_y()), ox(f.g.ox), oy(f.g.oy),
          oz(f.g.oz), inv_h(f.g.inv_hs), nx(float(f.g.nx)), ny(float(f.g.ny)),
          nz(float(f.g.nz)) {}
};

// Scatter lane `Lane` of the vector preparation into one AoS Stencil record,
// so the unchanged sampler consumes it (the record interface is kept).
// `base` holds two lanes' 64-bit offsets, hence `Lane % 2`. Invalid atoms
// point every corner at the zeroed tail.
template <int Lane>
inline void materialize(Stencil& s, float32x4_t mx, float32x4_t my,
                        float32x4_t mz, float32x4_t fx, float32x4_t fy,
                        float32x4_t fz, uint32x4_t valid, uint64x2_t base,
                        const PreparationGeometry& g) {
    s.mx = vgetq_lane_f32(mx, Lane);
    s.my = vgetq_lane_f32(my, Lane);
    s.mz = vgetq_lane_f32(mz, Lane);
    s.fx = vgetq_lane_f32(fx, Lane);
    s.fy = vgetq_lane_f32(fy, Lane);
    s.fz = vgetq_lane_f32(fz, Lane);
    s.valid = vgetq_lane_u32(valid, Lane) != 0;
    const size_t offset = vgetq_lane_u64(base, Lane % 2);
    s.offset[0] = s.valid ? offset : g.zero;
    s.offset[1] = s.valid ? offset + g.row : g.zero;
    s.offset[2] = s.valid ? offset + g.plane : g.zero;
    s.offset[3] = s.valid ? offset + g.plane + g.row : g.zero;
}

// Prepare atoms [first, first + 8) of `b` under the shared pose (m, t).
// Four whole atoms per NEON pass, then a scalar tail; both apply the same
// half-open bounds. The vector transform's FMA association may differ from
// the compiler's scalar tail; the frozen all-eight panel
// output still matched bytes.
template <class Field>
void prepare(const Field& f, const ConformerSoA& b, int first, const float m[9],
             const float t[3], PreparedBlock& block,
             const PreparationGeometry& geometry) {
    block.count = std::min(8, b.n - first);
    const auto& g = f.g;
    const size_t row = size_t(g.nodes_x()) * 4,
                 plane = row * size_t(g.nodes_y());
    int j = 0;
    if constexpr (R3_FIELD_NEON_PREP == 1) {
        const auto zero = vdupq_n_f32(0);
        for (; j + 4 <= block.count; j += 4) {
            const int i = first + j;
            const auto x = vld1q_f32(b.x + i), y = vld1q_f32(b.y + i),
                       z = vld1q_f32(b.z + i);
            // Same shared rigid pose; only independent atoms occupy SIMD lanes.
            const auto mx = vfmaq_n_f32(
                vfmaq_n_f32(vmulq_n_f32(y, m[1]), x, m[0]), z, m[2]);
            const auto my = vfmaq_n_f32(
                vfmaq_n_f32(vmulq_n_f32(y, m[4]), x, m[3]), z, m[5]);
            const auto mz = vfmaq_n_f32(
                vfmaq_n_f32(vmulq_n_f32(y, m[7]), x, m[6]), z, m[8]);
            const auto tx =
                vmulq_n_f32(vsubq_f32(vaddq_f32(mx, vdupq_n_f32(t[0])),
                                      vdupq_n_f32(geometry.ox)),
                            geometry.inv_h);
            const auto ty =
                vmulq_n_f32(vsubq_f32(vaddq_f32(my, vdupq_n_f32(t[1])),
                                      vdupq_n_f32(geometry.oy)),
                            geometry.inv_h);
            const auto tz =
                vmulq_n_f32(vsubq_f32(vaddq_f32(mz, vdupq_n_f32(t[2])),
                                      vdupq_n_f32(geometry.oz)),
                            geometry.inv_h);
            // Ordered comparisons reject NaN/Inf and the excluded upper face.
            auto valid =
                vandq_u32(vandq_u32(vcgeq_f32(tx, zero), vcgeq_f32(ty, zero)),
                          vcgeq_f32(tz, zero));
            valid = vandq_u32(
                valid,
                vandq_u32(vandq_u32(vcltq_f32(tx, vdupq_n_f32(geometry.nx)),
                                    vcltq_f32(ty, vdupq_n_f32(geometry.ny))),
                          vcltq_f32(tz, vdupq_n_f32(geometry.nz))));
            // Select zero BEFORE conversion, so overflow/nonfinite lanes never
            // reach integer conversion or address arithmetic with unsafe values.
            const auto sx = vbslq_f32(valid, tx, zero),
                       sy = vbslq_f32(valid, ty, zero),
                       sz = vbslq_f32(valid, tz, zero);
            // sx >= 0, so FCVTZU truncation is floor.
            const auto ix = vcvtq_u32_f32(sx), iy = vcvtq_u32_f32(sy),
                       iz = vcvtq_u32_f32(sz);
            const auto fx = vsubq_f32(sx, vcvtq_f32_u32(ix)),
                       fy = vsubq_f32(sy, vcvtq_f32_u32(iy)),
                       fz = vsubq_f32(sz, vcvtq_f32_u32(iz));
            // 64-bit offsets preserve the field's size_t addressing contract.
            // lo_yz/hi_yz = iz * nodes_y + iy for lanes 0-1 / 2-3 (UMULL
            // widens), the node-row index before scaling by `row`.
            const auto lo_yz =
                vaddq_u64(vmull_n_u32(vget_low_u32(iz), geometry.nodes_y),
                          vmovl_u32(vget_low_u32(iy)));
            const auto hi_yz =
                vaddq_u64(vmull_n_u32(vget_high_u32(iz), geometry.nodes_y),
                          vmovl_u32(vget_high_u32(iy)));
            // AArch64 NEON has no vector 64-bit integer multiply. Materialize
            // row addresses after the vector cell work, without scalar FP math.
            const uint64x2_t lo = {vgetq_lane_u64(lo_yz, 0) * row +
                                       uint64_t(vgetq_lane_u32(ix, 0)) * 4,
                                   vgetq_lane_u64(lo_yz, 1) * row +
                                       uint64_t(vgetq_lane_u32(ix, 1)) * 4};
            const uint64x2_t hi = {vgetq_lane_u64(hi_yz, 0) * row +
                                       uint64_t(vgetq_lane_u32(ix, 2)) * 4,
                                   vgetq_lane_u64(hi_yz, 1) * row +
                                       uint64_t(vgetq_lane_u32(ix, 3)) * 4};
            materialize<0>(block.atom[j], mx, my, mz, fx, fy, fz, valid, lo,
                           geometry);
            materialize<1>(block.atom[j + 1], mx, my, mz, fx, fy, fz, valid, lo,
                           geometry);
            materialize<2>(block.atom[j + 2], mx, my, mz, fx, fy, fz, valid, hi,
                           geometry);
            materialize<3>(block.atom[j + 3], mx, my, mz, fx, fy, fz, valid, hi,
                           geometry);
        }
    }
    for (; j < block.count; ++j) {
        auto& s = block.atom[j];
        const int i = first + j;
        const float x = b.x[i], y = b.y[i], z = b.z[i];
        s.mx = m[0] * x + m[1] * y + m[2] * z;
        s.my = m[3] * x + m[4] * y + m[5] * z;
        s.mz = m[6] * x + m[7] * y + m[8] * z;
        const float tx = (s.mx + t[0] - g.ox) * g.inv_hs;
        const float ty = (s.my + t[1] - g.oy) * g.inv_hs;
        const float tz = (s.mz + t[2] - g.oz) * g.inv_hs;
        s.valid = tx >= 0 && ty >= 0 && tz >= 0 && tx < float(g.nx) &&
                  ty < float(g.ny) && tz < float(g.nz);
        if (s.valid) {
            const int ix = int(tx), iy = int(ty), iz = int(tz);
            s.fx = tx - ix;
            s.fy = ty - iy;
            s.fz = tz - iz;
            const size_t base =
                (size_t(iz) * size_t(g.nodes_y()) + size_t(iy)) * row +
                size_t(ix) * 4;
            s.offset[0] = base;
            s.offset[1] = base + row;
            s.offset[2] = base + plane;
            s.offset[3] = base + plane + row;
        } else {
            s.fx = s.fy = s.fz = 0;
            for (auto& offset : s.offset)
                offset = f.zero_offset();
        }
    }
}

template <class Field>
void prepare(const Field& f, const ConformerSoA& b, int first, const float m[9],
             const float t[3], PreparedBlock& block) {
    const PreparationGeometry geometry(f);
    prepare(f, b, first, m, t, block, geometry);
}

// The tree's lerp, `a + t (b - a)`, as in field_neon.cpp and field_avx2.hpp.
inline float32x4_t lerp_prepared(float32x4_t a, float32x4_t b, float t) {
    return vfmaq_n_f32(a, vsubq_f32(b, a), t);
}

// Row policy: how one x-pair row (two nodes) is loaded and how four rows
// become a sample. FP32Rows is the shipped arithmetic; field_neon_half_rows.hpp
// holds the FP16 policies. Policies change arithmetic only -- preparation,
// scheduling and reduction are shared, as a matched test requires.
struct FP32Rows {
    struct Row {
        float32x4_t lo, hi;
    };

    static Row load(const FieldNaive32& f, size_t offset) {
        const float* p = f.v.data() + offset;
        return {vld1q_f32(p), vld1q_f32(p + 4)};
    }

    static float32x4_t finish(const Row r[4], const Stencil& s) {
        const auto a = lerp_prepared(r[0].lo, r[0].hi, s.fx),
                   b = lerp_prepared(r[1].lo, r[1].hi, s.fx);
        const auto c = lerp_prepared(r[2].lo, r[2].hi, s.fx),
                   d = lerp_prepared(r[3].lo, r[3].hi, s.fx);
        return lerp_prepared(lerp_prepared(a, b, s.fy),
                             lerp_prepared(c, d, s.fy), s.fz);
    }
};

// Interpolate N prepared stencils. N=1 is the shipped schedule; N=2 is the
// losing two-tree variant (schedule 1), kept as a reproducible control.
// Invalid stencils load the zeroed tail (safe) and return zero.
template <int N, class Policy, class Field>
void sample_trees(const Field& f, const Stencil* s, float32x4_t out[N]) {
    typename Policy::Row rows[N][4];
    // Each phase issues both independent actual stencil trees, not just
    // coordinate preparation. N=2 is instantiated only for two real atoms.
    for (int row = 0; row < 4; ++row)
        for (int lane = 0; lane < N; ++lane)
            rows[lane][row] = Policy::load(f, s[lane].offset[row]);
    for (int lane = 0; lane < N; ++lane)
        out[lane] = s[lane].valid ? Policy::finish(rows[lane], s[lane])
                                  : vdupq_n_f32(0);
}

// The prepared gradient loop for any Row policy. Same contract and same
// atom-order reduction as field_neon.cpp's original loop.
template <class Policy, class Field>
void gradient_prepared(const Field& f, const ConformerSoA& b, const float m[9],
                       const float t[3], float norm, float g[7], float* score) {
    std::fill(g, g + 7, 0.f);
    auto sum = vdupq_n_f32(0.f);
    float torque[3]{};
    auto accumulate = [&](const Stencil& s, float32x4_t v) {
        sum = vaddq_f32(sum, v);
        // Invalid or nonfinite lever arms add no torque (0 * Inf = NaN).
        if (!s.valid) return;
        if (!(std::isfinite(s.mx) && std::isfinite(s.my) &&
              std::isfinite(s.mz)))
            return;
        const float gx = -norm * vgetq_lane_f32(v, 1),
                    gy = -norm * vgetq_lane_f32(v, 2),
                    gz = -norm * vgetq_lane_f32(v, 3);
        torque[0] += 2.f * s.my * gz - 2.f * s.mz * gy;
        torque[1] += 2.f * s.mz * gx - 2.f * s.mx * gz;
        torque[2] += 2.f * s.mx * gy - 2.f * s.my * gx;
    };
    auto consume = [&](const PreparedBlock& block) {
        int j = 0;
        if constexpr (R3_FIELD_NEON_SCHEDULE == 1) {
            for (; j + 2 <= block.count; j += 2) {
                float32x4_t v[2];
                sample_trees<2, Policy>(f, block.atom + j, v);
                accumulate(block.atom[j], v[0]);
                accumulate(block.atom[j + 1], v[1]);
            }
        }
        for (; j < block.count; ++j) {
            float32x4_t v[1];
            sample_trees<1, Policy>(f, block.atom + j, v);
            accumulate(block.atom[j], v[0]);
        }
    };
    // Two buffers only matter for schedule 2, which swaps POINTERS (no
    // record copies); schedule 3 consumes then re-prepares `current`.
    // No initial memset: prepare() writes every member that is consumed
    // (a 1040-byte clear would be redundant).
    PreparedBlock storage[2];
    auto* current = &storage[0];
    auto* next = &storage[1];
    const PreparationGeometry geometry(f);
    if (b.n > 0) prepare(f, b, 0, m, t, *current, geometry);
    for (int first = 0; first < b.n; first += 8) {
        if constexpr (R3_FIELD_NEON_SCHEDULE == 2) {
            if (first + 8 < b.n)
                prepare(f, b, first + 8, m, t, *next, geometry);
            consume(*current);
            std::swap(current, next);
        } else {
            consume(*current);
            if (first + 8 < b.n)
                prepare(f, b, first + 8, m, t, *current, geometry);
        }
    }
    g[1] = torque[0];
    g[2] = torque[1];
    g[3] = torque[2];
    g[4] = -norm * vgetq_lane_f32(sum, 1);
    g[5] = -norm * vgetq_lane_f32(sum, 2);
    g[6] = -norm * vgetq_lane_f32(sum, 3);
    if (score) *score = vgetq_lane_f32(sum, 0);
}
}

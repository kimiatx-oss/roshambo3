// Shared scalar pieces of the fixed-pose final rescore (final_rescore.hpp):
// the polynomial exp, the pose rotation, and mode 1, the scalar-polynomial
// control.
//
// A measurement campaign compared two SIMD grains for the
// final shape+color cross terms -- lanes over atom pairs within one molecule
// pair (mode 2) and lanes owning independent molecule pairs (mode 3) -- and
// needed a scalar control using the SAME exp so that the exp strategy, the
// grain and the reduction order could be told apart. `scalar_poly` is that
// control: the polynomial exp in the accurate shape path's term order. Its
// polynomial output is not bit-exact against the accurate final and is not
// labeled so.
#pragma once
#include "final_rescore.hpp"
#include <cmath>
#include <cstring>
#include <vector>

namespace r3::cpu::final_detail {
// Exactly the existing degree-seven range-reduced exp polynomial, with safe
// negative-input handling before integer conversion; below -80 contributes 0.
//
// The same expansion as `neon_exp_polynomial` (optimizer_exp_neon.hpp)
// and Vector8::exp in final_rescore_avx2.cpp, so modes 1-3 share one exp.
// Range reduction: e^x = 2^n * e^((y - n) ln 2) with y = x log2(e),
// n = floor(y), so the Taylor argument z lies in [0, ln 2); 2^n is built
// directly in the exponent field. `!(x >= -80)` also sends NaN to 0, and the
// early return keeps n + 127 positive before the shift.
inline float polynomial(float x) {
    if (!(x >= -80.f)) return 0.f;
    const float y = x * 1.4426950408889634074f, n = std::floor(y);
    const float z = (y - n) * .69314718055994530942f;
    const uint32_t bits = uint32_t(int(n) + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    constexpr float c[] = {1,        1,         .5f,       1.f / 6,
                           1.f / 24, 1.f / 120, 1.f / 720, 1.f / 5040};
    float p = c[7];
    for (int k = 6; k >= 0; --k)
        p = std::fma(p, z, c[k]);
    return p * scale;
}

// R * b for the row-major 3x3 pose matrix (which carries COORD_SCALE, so a
// physical-frame library point lands in the reduced frame). Translation is
// applied by the caller.
inline void rotate(const PreparedReducedPose& p, const float* b, float out[3]) {
    for (int k = 0; k < 3; ++k)
        out[k] = p.matrix[3 * k] * b[0] + p.matrix[3 * k + 1] * b[1] +
                 p.matrix[3 * k + 2] * b[2];
}

// Mode 1: one pair, scalar, polynomial exp. Shape: the query is in the
// reduced frame, so each term is exp2(-|d|^2), rewritten as
// exp(-ln2 * |d|^2) for the natural-base polynomial (the equivalence is
// stated explicitly). The pose moves the LIBRARY atom: d = (a - t) - R b,
// in that association. Color: only canonical (diagonal) type planes are
// visited; posed library features are packed once per pair, then each
// query/target term of matching type is fma'd with the plane coefficient.
inline FinalRescoreCross scalar_poly(const FinalRescorePair& p,
                                     FinalRescoreCounts* c) {
    FinalRescoreCross out;
    const auto& a = p.shape_query_reduced;
    const auto& b = p.shape_library_physical;
    for (int j = 0; j < b.n_atoms; ++j) {
        float mb[3];
        rotate(*p.pose, b.xyz + 3 * j, mb);
        for (int i = 0; i < a.n_atoms; ++i) {
            const float dx = (a.xyz[3 * i] - p.pose->translation[0]) - mb[0];
            const float dy =
                (a.xyz[3 * i + 1] - p.pose->translation[1]) - mb[1];
            const float dz =
                (a.xyz[3 * i + 2] - p.pose->translation[2]) - mb[2];
            out.shape += polynomial(-.69314718055994530942f *
                                    (dx * dx + dy * dy + dz * dz));
            if (c) {
                ++c->shape_terms;
                ++c->shape_issued_lanes;
            }
        }
    }
    float color = 0;
    if (p.color_plan) {
        const auto& q = p.color_query_reduced->features;
        const auto& l = *p.color_library_physical;
        // Per-call allocation is real work in the measured kernel and is
        // counted (temporary_allocations), not hidden in setup.
        std::vector<float> posed(size_t(l.n_color) * 3);
        for (int j = 0; j < l.n_color; ++j) {
            rotate(*p.pose, l.color_xyz + size_t(j) * p.library_color_stride,
                   posed.data() + 3 * j);
            for (int k = 0; k < 3; ++k)
                posed[3 * j + k] += p.pose->translation[k];
        }
        if (c) {
            c->packed_coordinate_bytes += posed.size() * sizeof(float);
            c->temporary_allocations += !posed.empty();
            c->allocated_payload_bytes += posed.capacity() * sizeof(float);
        }
        for (const auto& plane : p.color_plan->canonical_planes()) {
            const int t = plane.type;
            const float decay = float(p.color_plan->reduced_decay(t, t));
            if (decay == 0) continue;
            const float coefficient = float(p.color_plan->coefficient(t, t));
            for (int i = 0; i < q.seg_count[t]; ++i) {
                const float* x = q.color_xyz + size_t(q.seg_offset[t] + i) *
                                                   p.query_color_stride;
                for (int j = 0; j < l.seg_count[t]; ++j) {
                    const float* y = posed.data() + 3 * (l.seg_offset[t] + j);
                    const float dx = x[0] - y[0], dy = x[1] - y[1],
                                dz = x[2] - y[2];
                    color = std::fma(
                        coefficient,
                        polynomial(decay * (dx * dx + dy * dy + dz * dz)),
                        color);
                    if (c) {
                        ++c->color_terms;
                        ++c->color_issued_lanes;
                    }
                }
            }
        }
    }
    // FP32 accumulation, widened once at the API boundary (FinalRescoreCross).
    out.color = double(color);
    return out;
}
}

// Baseline-NEON field entry points (see field_neon.hpp for the contract).
//
// This translation unit is compiled for the baseline AArch64 target only, so
// it can run on any ARMv8 core. It holds the original per-atom sampler
// (kept as the R3_FIELD_NEON_SCHEDULE=0 control and the AoS `Conformer` entry),
// the dispatch onto the prepared and base32 paths in the headers, and the
// FP16 capability forwards; FP16 arithmetic lives in its own TUs.
#include "field_neon.hpp"
#include "field_neon_prepared.hpp"
#include "field_neon_dataflow.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include "cpu_platform.hpp"

namespace r3::cpu {
namespace {
// `a + t (b - a)` on all four channels at once -- the same lerp as
// field_avx2.hpp's `lerp256`, one node per register instead of two.
float32x4_t lerp(float32x4_t a, float32x4_t b, float t) {
    return vfmaq_n_f32(a, vsubq_f32(b, a), t);
}

// One trilinear sample [f, gx, gy, gz] at reduced-frame point (x, y, z).
// Stored gradients are interpolated as channels, never differentiated from
// the interpolated value. X first, then Y, then Z: the established tree order.
float32x4_t sample(const FieldNaive32& f, float x, float y, float z) {
    const auto& g = f.g;
    const float tx = (x - g.ox) * g.inv_hs, ty = (y - g.oy) * g.inv_hs,
                tz = (z - g.oz) * g.inv_hs;
    // Compare before integer conversion: negative, NaN, infinity and overflow
    // are border zero, including odd-tail samples.
    if (!(tx >= 0.f && ty >= 0.f && tz >= 0.f && tx < float(g.nx) &&
          ty < float(g.ny) && tz < float(g.nz)))
        return vdupq_n_f32(0.f);
    // tx >= 0 here, so truncation is floor; tx < nx keeps ix + 1 a real node.
    const int ix = int(tx), iy = int(ty), iz = int(tz);
    const float fx = tx - ix, fy = ty - iy, fz = tz - iz;
    const auto x00 = lerp(vld1q_f32(f.node(ix, iy, iz)),
                          vld1q_f32(f.node(ix + 1, iy, iz)), fx);
    const auto x10 = lerp(vld1q_f32(f.node(ix, iy + 1, iz)),
                          vld1q_f32(f.node(ix + 1, iy + 1, iz)), fx);
    const auto x01 = lerp(vld1q_f32(f.node(ix, iy, iz + 1)),
                          vld1q_f32(f.node(ix + 1, iy, iz + 1)), fx);
    const auto x11 = lerp(vld1q_f32(f.node(ix, iy + 1, iz + 1)),
                          vld1q_f32(f.node(ix + 1, iy + 1, iz + 1)), fx);
    return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz);
}

// The original loop: transform up to eight atoms, then sample them one
// at a time. `coord(j, c)` abstracts AoS versus SoA library storage.
template <class Coordinate>
void gradient(const FieldNaive32& f, int n, Coordinate coord, const float m[9],
              const float t[3], float gnorm, float g[7], float* score) {
    std::fill(g, g + 7, 0.f);
    auto sum = vdupq_n_f32(0.f);
    float torque[3]{};
    for (int first = 0; first < n; first += 8) {
        const int count = std::min(8, n - first);
        float mx[8], my[8], mz[8];
        for (int j = 0; j < count; ++j) {
            const float bx = coord(first + j, 0), by = coord(first + j, 1),
                        bz = coord(first + j, 2);
            mx[j] = m[0] * bx + m[1] * by + m[2] * bz;
            my[j] = m[3] * bx + m[4] * by + m[5] * bz;
            mz[j] = m[6] * bx + m[7] * by + m[8] * bz;
        }
        // Each NEON load/interpolation evaluates [value,gx,gy,gz]. The next
        // atom has an independent stencil; no AVX cross-half shuffle exists.
        for (int j = 0; j < count; ++j) {
            const auto v = sample(f, mx[j] + t[0], my[j] + t[1], mz[j] + t[2]);
            sum = vaddq_f32(sum, v);
            // A zero sample (outside the lattice) or a nonfinite lever arm
            // contributes no torque: a zero stencil alone would still give
            // 0 * Inf = NaN.
            if (vgetq_lane_f32(v, 1) == 0.f && vgetq_lane_f32(v, 2) == 0.f &&
                vgetq_lane_f32(v, 3) == 0.f)
                continue;
            if (!(std::isfinite(mx[j]) && std::isfinite(my[j]) &&
                  std::isfinite(mz[j])))
                continue;
            const float gx = -gnorm * vgetq_lane_f32(v, 1),
                        gy = -gnorm * vgetq_lane_f32(v, 2),
                        gz = -gnorm * vgetq_lane_f32(v, 3);
            // (2 M b) x grad: field.hpp's lever arm, before translation.
            torque[0] += 2.f * my[j] * gz - 2.f * mz[j] * gy;
            torque[1] += 2.f * mz[j] * gx - 2.f * mx[j] * gz;
            torque[2] += 2.f * mx[j] * gy - 2.f * my[j] * gx;
        }
    }
    g[1] = torque[0];
    g[2] = torque[1];
    g[3] = torque[2];
    // Translation gradient: the channel sum is scaled once at the end, not
    // per atom as in field.hpp -- equal in exact arithmetic, but rounded
    // differently.
    g[4] = -gnorm * vgetq_lane_f32(sum, 1);
    g[5] = -gnorm * vgetq_lane_f32(sum, 2);
    g[6] = -gnorm * vgetq_lane_f32(sum, 3);
    if (score) *score = vgetq_lane_f32(sum, 0);
}
}

void field_gradient_neon(const FieldNaive32& f, const Conformer& b,
                         const float m[9], const float t[3], float norm,
                         float g[7], float* score) {
    gradient(
        f, b.n_atoms, [&](int j, int c) { return b.xyz[3 * j + c]; }, m, t,
        norm, g, score);
}

// Production FP32 shape gradient. With the default schedule and a field whose
// uint32 offsets were proven at query setup, this is the base32 path; an
// unproved or oversized field takes the checked size_t prepared path.
void field_gradient_neon_pipelined(const FieldNaive32& f, const ConformerSoA& b,
                                   const float m[9], const float t[3],
                                   float norm, float g[7], float* score) {
#if R3_FIELD_NEON_SCHEDULE != 0
#if R3_FIELD_NEON_ADDRESS && R3_FIELD_NEON_SCHEDULE == 3
    if (f.neon_offsets32) {
        detail::gradient_base32(f, b, m, t, norm, g, score);
        return;
    }
#endif
    detail::gradient_prepared<detail::FP32Rows>(f, b, m, t, norm, g, score);
#else
    gradient(
        f, b.n,
        [&](int j, int c) {
            return c == 0 ? b.x[j] : c == 1 ? b.y[j] : b.z[j];
        },
        m, t, norm, g, score);
#endif
}

void field_value_neon_pipelined(const FieldNaive32& f, const ConformerSoA& b,
                                const float m[9], const float t[3],
                                float* score) {
#if R3_FIELD_NEON_ADDRESS && R3_FIELD_NEON_SCHEDULE == 3
    if (f.neon_offsets32) {
        detail::value_base32(f, b, m, t, score);
        return;
    }
#endif
    // The general size_t addressing and historical schedule optouts retain
    // their checked numerical implementation and exact source-order score.
    float gradient[7];
    field_gradient_neon_pipelined(f, b, m, t, 1.0f, gradient, score);
}

// Both FP16 queries forward to the same FEAT_FP16 probe: storage conversion
// and native half arithmetic are gated by one capability on this path. FHM
// (FMLAL/FMLSL) is a separate feature and is probed independently.
//
// This is stricter than necessary, deliberately. H0 only converts half nodes
// to FP32 and interpolates in FP32, which by itself needs no FEAT_FP16; only
// H2's native half lerps do. Gating both on FEAT_FP16 refuses H0 on a core
// that has the conversion but not the arithmetic -- it never admits a path
// the core cannot run, and H0 is compiled in the FP16 translation unit
// anyway. Such a core simply keeps the FP32 field default. Splitting the gate
// would buy nothing on the hosts measured: the M4 Pro and the Neoverse-V2
// host that ran the half-field arms both have FEAT_FP16.
bool field_neon_fp16_supported() noexcept {
    return platform::fp16_supported();
}

bool field_neon_fp16_arithmetic_supported() noexcept {
    return platform::fp16_supported();
}

bool field_neon_fp16_fml_supported() noexcept {
    return platform::fp16_fml_supported();
}
}

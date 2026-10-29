// H1: only this TU targets fp16fml. Baseline dispatch tests ASIMDFHM.
// (Through platform::fp16_fml_supported(): HWCAP_ASIMDFHM on
// Linux, the FEAT_FHM sysctl on macOS.)
//
// The H1 policy: FP16 nodes, X interpolated by widening multiply-
// accumulate. FHM is a feature independent of FP16 and is probed on its own;
// a fixture masking only that bit verified the FP32 fallback. H1 passed
// recall but lost on speed and stopped at the primary measurement;
// it remains an explicit private policy.
#include "field_neon_gradient.hpp"
#include "field_neon_half_rows.hpp"

namespace r3::cpu {
namespace {
// One x-pair row: low + (-low * tx) + high * tx, accumulated in FP32 from
// half products (FMLSL on the low four halves, FMLAL2 on the high four).
inline float32x4_t row(const uint16_t* p, float16x8_t tx) {
    const auto endpoints = vreinterpretq_f16_u16(vld1q_u16(p));
    const auto low = vcvt_f32_f16(vget_low_f16(endpoints));
    // tx was narrowed once per actual sample and duplicated into both halves.
    return vfmlalq_high_f16(vfmlslq_low_f16(low, endpoints, tx), endpoints, tx);
}

inline float32x4_t lerp(float32x4_t a, float32x4_t b, float t) {
    return vfmaq_n_f32(a, vsubq_f32(b, a), t);
}

// Per-atom H1 sample for the original schedule; same bounds as FP32.
float32x4_t sample(const FieldNaive16& f, float x, float y, float z) {
    const auto& g = f.g;
    const float tx = (x - g.ox) * g.inv_hs, ty = (y - g.oy) * g.inv_hs,
                tz = (z - g.oz) * g.inv_hs;
    if (!(tx >= 0.f && ty >= 0.f && tz >= 0.f && tx < float(g.nx) &&
          ty < float(g.ny) && tz < float(g.nz)))
        return vdupq_n_f32(0.f);
    const int ix = int(tx), iy = int(ty), iz = int(tz);
    const float fx = tx - ix, fy = ty - iy, fz = tz - iz;
    const auto weight = vdupq_n_f16(static_cast<float16_t>(fx));
    const size_t stride = size_t(g.nodes_x()) * 4,
                 plane = stride * size_t(g.nodes_y());
    const uint16_t* p = f.node(ix, iy, iz);
    const auto a = row(p, weight), b = row(p + stride, weight);
    const auto c = row(p + plane, weight), d = row(p + plane + stride, weight);
    return lerp(lerp(a, b, fy), lerp(c, d, fy), fz);
}
}

void field_gradient_neon_fp16_fml_pipelined(const FieldNaive16& f,
                                            const ConformerSoA& b,
                                            const float m[9], const float t[3],
                                            float norm, float g[7],
                                            float* score) {
#if R3_FIELD_NEON_SCHEDULE != 0
    detail::gradient_prepared<detail::FMLRows>(f, b, m, t, norm, g, score);
#else
    detail::gradient_fp16(f, b, m, t, norm, g, score,
                          [](const FieldNaive16& field, float x, float y,
                             float z) { return sample(field, x, y, z); });
#endif
}
}

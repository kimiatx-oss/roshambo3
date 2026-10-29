// FP16 Row policies for gradient_prepared (field_neon_prepared.hpp). A row is
// one 16-byte load of an x-pair, [x0.half4 | x1.half4]; the policy decides
// where widening to FP32 happens. Preparation, scheduling and reduction are
// shared with FP32Rows, so the comparison between them is matched.
#pragma once
#include "field_neon_prepared.hpp"

namespace r3::cpu::detail {
// H0: widen both endpoints of every row (eight FCVTLs per sample), then the
// FP32 XYZ tree. Only conversions touch half data; all arithmetic is FP32.
struct HalfRows {
    using Row = float16x8_t;

    static Row load(const FieldNaive16& f, size_t offset) {
        return vreinterpretq_f16_u16(vld1q_u16(f.v.data() + offset));
    }

    static float32x4_t edge(Row r, float fraction) {
        return lerp_prepared(vcvt_f32_f16(vget_low_f16(r)),
                             vcvt_f32_f16(vget_high_f16(r)), fraction);
    }

    static float32x4_t finish(const Row r[4], const Stencil& s) {
        const auto a = edge(r[0], s.fx), b = edge(r[1], s.fx),
                   c = edge(r[2], s.fx), d = edge(r[3], s.fx);
        return lerp_prepared(lerp_prepared(a, b, s.fy),
                             lerp_prepared(c, d, s.fy), s.fz);
    }
};

// H2: Y and Z lerps in native half on packed x-endpoint rows (FEAT_FP16),
// with fy/fz narrowed once per sample; widen only after Z, X last in FP32.
struct MixedRows : HalfRows {
    static float32x4_t finish(const Row r[4], const Stencil& s) {
        const auto fy = static_cast<float16_t>(s.fy),
                   fz = static_cast<float16_t>(s.fz);
        const auto a = vfmaq_n_f16(r[0], vsubq_f16(r[1], r[0]), fy);
        const auto b = vfmaq_n_f16(r[2], vsubq_f16(r[3], r[2]), fy);
        return edge(vfmaq_n_f16(a, vsubq_f16(b, a), fz), s.fx);
    }
};

#if defined(__ARM_FEATURE_FP16_FML)

// H1: per row, a - a tx + b tx with widening FMLSL (low half) and FMLAL2
// (high half) onto the widened low endpoint; tx is narrowed once per sample
// and shared by all four rows. Y and Z in FP32. Compiled only in the
// +fp16fml translation unit. Passes recall, 1.6-1.8% slower than F0
// and 3.0-3.6% slower than H2; closed.
struct FMLRows : HalfRows {
    static float32x4_t finish(const Row r[4], const Stencil& s) {
        const auto tx = vdupq_n_f16(static_cast<float16_t>(s.fx));
        const auto edge = [&](Row endpoints) {
            const auto low = vcvt_f32_f16(vget_low_f16(endpoints));
            return vfmlalq_high_f16(vfmlslq_low_f16(low, endpoints, tx),
                                    endpoints, tx);
        };
        const auto a = edge(r[0]), b = edge(r[1]), c = edge(r[2]),
                   d = edge(r[3]);
        return lerp_prepared(lerp_prepared(a, b, s.fy),
                             lerp_prepared(c, d, s.fy), s.fz);
    }
};
#endif
}

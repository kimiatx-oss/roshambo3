// Runtime-dispatched AArch64 FP16-storage/FP32-interpolation sampler.
// This translation unit alone is compiled for ARMv8.2-A FP16 instructions;
// callers first test HWCAP_ASIMDHP in the baseline NEON translation unit.
// (That test is platform::fp16_supported(): HWCAP_ASIMDHP on
// Linux, the FEAT_FP16 sysctl on macOS; see cpu_platform.cpp.)
//
// Two FP16-storage policies over the same half nodes: H0 widens all
// eight corners and interpolates in FP32; H2 ("mixed YZ") does the Y and Z
// lerps in native half and X last in FP32. Both are explicit low-level
// policies, not the ARM default: prepared FP32 measured faster, and
// H0's loss came from runtime widening, not from the half-rounded values
// (F0_Q and H0 produced equal screening hashes).
// Coordinates, fractions (until H2 narrows them), pose and every reduction
// stay FP32.
#include "field_neon.hpp"
#include "field_neon_gradient.hpp"
#include "field_neon_prepared.hpp"
#include "field_neon_half_rows.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace r3::cpu {
namespace {
using detail::HalfRows;
using detail::MixedRows;

// `a + t (b - a)`, FP32, as in field_neon.cpp.
float32x4_t lerp_fp32(float32x4_t a, float32x4_t b, float t) {
    return vfmaq_n_f32(a, vsubq_f32(b, a), t);
}

float32x4_t widen_lerp_pair(const uint16_t* pair, float fraction) {
    // One 16-byte load is [x0.half4 | x1.half4]. Widen each half separately,
    // then keep all arithmetic and fractions in FP32.
    const float16x8_t halves = vreinterpretq_f16_u16(vld1q_u16(pair));
    const float32x4_t x0 = vcvt_f32_f16(vget_low_f16(halves));
    const float32x4_t x1 = vcvt_f32_f16(vget_high_f16(halves));
    return lerp_fp32(x0, x1, fraction);
}

// H0 sample: half storage, FP32 XYZ tree. Same bounds and tree order as the
// FP32 sampler; only the node loads differ (16 bytes per x-pair, not 32).
float32x4_t sample_fp16_storage(const FieldNaive16& field, float x, float y,
                                float z) {
    const FieldGeometry& g = field.g;
    const float tx = (x - g.ox) * g.inv_hs;
    const float ty = (y - g.oy) * g.inv_hs;
    const float tz = (z - g.oz) * g.inv_hs;
    // Bounds precede integer conversion and any address formation. In
    // particular NaN/infinity and coordinates beyond the lattice are border
    // zero, just as in the FP32 sampler.
    if (!(tx >= 0.f && ty >= 0.f && tz >= 0.f && tx < float(g.nx) &&
          ty < float(g.ny) && tz < float(g.nz)))
        return vdupq_n_f32(0.f);

    const int ix = int(tx), iy = int(ty), iz = int(tz);
    const float fx = tx - float(ix), fy = ty - float(iy), fz = tz - float(iz);
    const size_t row = size_t(g.nodes_x()) * 4;
    const size_t plane = row * size_t(g.nodes_y());
    const uint16_t* p00 = field.node(ix, iy, iz);
    const uint16_t* p10 = p00 + row;
    const uint16_t* p01 = p00 + plane;
    const uint16_t* p11 = p01 + row;

    // Four adjacent x-row pair loads, eight widening conversions, and the
    // same seven FP32 lerps as the established trilinear interpolation tree.
    const float32x4_t x00 = widen_lerp_pair(p00, fx);
    const float32x4_t x10 = widen_lerp_pair(p10, fx);
    const float32x4_t x01 = widen_lerp_pair(p01, fx);
    const float32x4_t x11 = widen_lerp_pair(p11, fx);
    return lerp_fp32(lerp_fp32(x00, x10, fy), lerp_fp32(x01, x11, fy), fz);
}

// H2 sample. A distinct numerical contract, not just half storage: half
// fractions, half intermediate rounding and X-last axis order.
// Finite half endpoints can still overflow in `r10 - r00`; eligibility is
// the one-time +-32720 node bound in shape_field_evaluator.cpp, so there is
// no per-corner check here.
float32x4_t sample_fp16_mixed_yz(const FieldNaive16& field, float x, float y,
                                 float z) {
    const FieldGeometry& g = field.g;
    const float tx = (x - g.ox) * g.inv_hs;
    const float ty = (y - g.oy) * g.inv_hs;
    const float tz = (z - g.oz) * g.inv_hs;
    // Bounds and fractions stay FP32; no half conversion or pointer formation
    // happens until the coordinate is known to be a finite in-grid cell.
    if (!(tx >= 0.f && ty >= 0.f && tz >= 0.f && tx < float(g.nx) &&
          ty < float(g.ny) && tz < float(g.nz)))
        return vdupq_n_f32(0.f);

    const int ix = int(tx), iy = int(ty), iz = int(tz);
    const float fx = tx - float(ix), fy = ty - float(iy), fz = tz - float(iz);
    const size_t row = size_t(g.nodes_x()) * 4;
    const size_t plane = row * size_t(g.nodes_y());
    const uint16_t* p00 = field.node(ix, iy, iz);
    const uint16_t* p10 = p00 + row;
    const uint16_t* p01 = p00 + plane;
    const uint16_t* p11 = p01 + row;

    // Each row pair is [x0.half4 | x1.half4]. Keep both X endpoints packed
    // while Y and Z are evaluated in native half; widen only after Z, then do
    // the final X lerp in FP32. Half fractions are rounded once here.
    const float16x8_t r00 = vreinterpretq_f16_u16(vld1q_u16(p00));
    const float16x8_t r10 = vreinterpretq_f16_u16(vld1q_u16(p10));
    const float16x8_t r01 = vreinterpretq_f16_u16(vld1q_u16(p01));
    const float16x8_t r11 = vreinterpretq_f16_u16(vld1q_u16(p11));
    const float16_t fy_half = static_cast<float16_t>(fy);
    const float16_t fz_half = static_cast<float16_t>(fz);
    const float16x8_t y0 = vfmaq_n_f16(r00, vsubq_f16(r10, r00), fy_half);
    const float16x8_t y1 = vfmaq_n_f16(r01, vsubq_f16(r11, r01), fy_half);
    const float16x8_t yz = vfmaq_n_f16(y0, vsubq_f16(y1, y0), fz_half);
    const float32x4_t x0 = vcvt_f32_f16(vget_low_f16(yz));
    const float32x4_t x1 = vcvt_f32_f16(vget_high_f16(yz));
    return lerp_fp32(x0, x1, fx);
}

using detail::gradient_fp16;

// Original per-atom loops (R3_FIELD_NEON_SCHEDULE=0) for the two policies.

void gradient_fp16_storage(const FieldNaive16& field,
                           const ConformerSoA& library, const float matrix[9],
                           const float translation[3], float gnorm,
                           float gradient[7], float* score) {
    gradient_fp16(field, library, matrix, translation, gnorm, gradient, score,
                  [](const FieldNaive16& f, float x, float y, float z) {
                      return sample_fp16_storage(f, x, y, z);
                  });
}

void gradient_fp16_mixed_yz(const FieldNaive16& field,
                            const ConformerSoA& library, const float matrix[9],
                            const float translation[3], float gnorm,
                            float gradient[7], float* score) {
    gradient_fp16(field, library, matrix, translation, gnorm, gradient, score,
                  [](const FieldNaive16& f, float x, float y, float z) {
                      return sample_fp16_mixed_yz(f, x, y, z);
                  });
}

}  // namespace

// FP32 -> FP16 nodes, once per query at setup, eight per pass plus a tail.
// FCVTN rounds per FPCR (round-to-nearest-even by default), the NEON
// counterpart of field.hpp's F16C `_MM_FROUND_TO_NEAREST_INT`. The caller
// has already refused fields outside the finite half range.
void quantize_field_neon_fp16(const FieldNaive32& src, FieldNaive16& dst) {
    dst.g = src.g;
    dst.v.resize(src.v.size());
    size_t i = 0;
    for (; i + 8 <= src.v.size(); i += 8) {
        const float32x4_t lo = vld1q_f32(&src.v[i]);
        const float32x4_t hi = vld1q_f32(&src.v[i + 4]);
        const float16x4_t half_lo = vcvt_f16_f32(lo);
        const float16x4_t half_hi = vcvt_f16_f32(hi);
        vst1q_u16(&dst.v[i],
                  vreinterpretq_u16_f16(vcombine_f16(half_lo, half_hi)));
    }
    for (; i < src.v.size(); ++i) {
        const float32x4_t one = vdupq_n_f32(src.v[i]);
        const float16x4_t half = vcvt_f16_f32(one);
        dst.v[i] = vget_lane_u16(vreinterpret_u16_f16(half), 0);
    }
}

// Half nodes back to FP32 storage (the F0_Q control). Widening is exact.
// The widened field is new storage, so the uint32-offset proof is cleared
// and recomputed for it rather than inherited.
void widen_field_neon_fp16(const FieldNaive16& src, FieldNaive32& dst) {
    dst.neon_offsets32 = false;
    dst.g = src.g;
    dst.v.resize(src.v.size());
    size_t i = 0;
    for (; i + 4 <= src.v.size(); i += 4)
        vst1q_f32(dst.v.data() + i, vcvt_f32_f16(vreinterpret_f16_u16(
                                        vld1_u16(src.v.data() + i))));
    for (; i < src.v.size(); ++i) {
        const auto h = vreinterpret_f16_u16(vdup_n_u16(src.v[i]));
        dst.v[i] = vgetq_lane_f32(vcvt_f32_f16(h), 0);
    }
    dst.prepare_neon_offsets32();
}

// Prepared-loop entry points: the same prepare/consume loop as FP32 with the
// half Row policies from field_neon_half_rows.hpp.
void field_gradient_neon_fp16_pipelined(const FieldNaive16& field,
                                        const ConformerSoA& library,
                                        const float matrix[9],
                                        const float translation[3], float gnorm,
                                        float gradient[7], float* score) {
#if R3_FIELD_NEON_SCHEDULE != 0
    detail::gradient_prepared<HalfRows>(field, library, matrix, translation,
                                        gnorm, gradient, score);
#else
    gradient_fp16_storage(field, library, matrix, translation, gnorm, gradient,
                          score);
#endif
}

void field_gradient_neon_fp16_mixed_yz_pipelined(const FieldNaive16& field,
                                                 const ConformerSoA& library,
                                                 const float matrix[9],
                                                 const float translation[3],
                                                 float gnorm, float gradient[7],
                                                 float* score) {
#if R3_FIELD_NEON_SCHEDULE != 0
    detail::gradient_prepared<MixedRows>(field, library, matrix, translation,
                                         gnorm, gradient, score);
#else
    gradient_fp16_mixed_yz(field, library, matrix, translation, gnorm, gradient,
                           score);
#endif
}

}  // namespace r3::cpu

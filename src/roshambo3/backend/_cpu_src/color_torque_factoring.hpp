// Force-then-torque factoring for the generic eight-lane canonical color
// evaluator (AVX2, ColorConformerBatch8 grain).
//
// For a fixed transformed target feature b'_j, the moment arm
// r_j = b'_j - t' is the same for every query source i, so
//
//     sum_i 2 r_j x f_ij  ==  2 r_j x (sum_i f_ij),
//     f_ij = -w * v_ij * (a_i - b'_j)          (pair translation descent)
//
// and the three per-pair cross products collapse to one per target feature.
// The two sides are equal in real arithmetic only: this changes FP32
// association, accepted under bounded ranking gates, not bit identity with
// the unfactored order.
//
// The four-start XMM4/NEON4/YMM8 helpers carry their own inline copies of
// the same factoring rather than calling this batch8 helper.
#pragma once

#include "color_conformer_kernel.hpp"
#include "kernel.hpp"

#if defined(__AVX2__)
#include <immintrin.h>

namespace r3 {
namespace cpu {

// Canonical analytic reduced-frame plane, one pair per SIMD lane.
// Accumulate target spatial descent first; rotate that force only once per
// target feature. Inputs are borrowed, immutable, and already pose-converted.
template <class Exponential>
inline void accumulate_canonical_factored_torque(
    const ColorConformerBatch8& batch, int type, const float* posed,
    const int32_t* analytic_mask, int nq, int nb, float coefficient,
    float exponent_scale, float translation_scale, __m256 tx, __m256 ty,
    __m256 tz, __m256& total_value, __m256 (&descent)[6],
    Exponential exponential) {
    const __m256 enabled = _mm256_castsi256_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(analytic_mask)));
    // Batch8 SoA index: [feature][component][lane], eight lanes per row.
    const auto at = [](size_t base, int i, int k) {
        return base + (size_t(i) * 3 + size_t(k)) * 8;
    };
    // Per-feature padding mask for each lane, ANDed with the caller's
    // safe-lane mask. Masked coordinates load as +0, and masked values are
    // zeroed below, so padded or unsafe lanes contribute nothing.
    const auto mask = [&](const std::vector<int32_t>& values, size_t base,
                          int i) {
        return _mm256_and_ps(
            enabled, _mm256_castsi256_ps(
                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                             values.data() + base + size_t(i) * 8))));
    };
    // Target-outer traversal: the spatial force f_j accumulates over every
    // query source before the single torque below.
    for (int j = 0; j < nb; ++j) {
        const __m256 bm =
            mask(batch.library_mask, batch.library_mask_offset[type], j);
        const size_t bbase = batch.library_coordinate_offset[type];
        const __m256 bx =
            _mm256_and_ps(bm, _mm256_loadu_ps(posed + at(bbase, j, 0)));
        const __m256 by =
            _mm256_and_ps(bm, _mm256_loadu_ps(posed + at(bbase, j, 1)));
        const __m256 bz =
            _mm256_and_ps(bm, _mm256_loadu_ps(posed + at(bbase, j, 2)));
        __m256 fx = _mm256_setzero_ps(), fy = fx, fz = fx;
        for (int i = 0; i < nq; ++i) {
            const __m256 qm =
                mask(batch.query_mask, batch.query_mask_offset[type], i);
            const size_t qbase = batch.query_coordinate_offset[type];
            const __m256 dx = _mm256_sub_ps(
                _mm256_and_ps(qm, _mm256_loadu_ps(batch.query_xyz.data() +
                                                  at(qbase, i, 0))),
                bx);
            const __m256 dy = _mm256_sub_ps(
                _mm256_and_ps(qm, _mm256_loadu_ps(batch.query_xyz.data() +
                                                  at(qbase, i, 1))),
                by);
            const __m256 dz = _mm256_sub_ps(
                _mm256_and_ps(qm, _mm256_loadu_ps(batch.query_xyz.data() +
                                                  at(qbase, i, 2))),
                bz);
            const __m256 d2 = _mm256_fmadd_ps(
                dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)));
            const __m256 value = _mm256_and_ps(
                _mm256_mul_ps(exponential(_mm256_mul_ps(
                                  d2, _mm256_set1_ps(exponent_scale))),
                              _mm256_set1_ps(coefficient)),
                _mm256_and_ps(qm, bm));
            total_value = _mm256_add_ps(total_value, value);
            const __m256 force =
                _mm256_mul_ps(value, _mm256_set1_ps(translation_scale));
            fx = _mm256_add_ps(fx, _mm256_mul_ps(force, dx));
            fy = _mm256_add_ps(fy, _mm256_mul_ps(force, dy));
            fz = _mm256_add_ps(fz, _mm256_mul_ps(force, dz));
        }
        // Moment arm r_j = b'_j - t', then descent[0..2] += 2 r_j x f_j and
        // descent[3..5] += f_j (ColorEvaluation::descent slots 1..6).
        const __m256 rx = _mm256_sub_ps(bx, tx);
        const __m256 ry = _mm256_sub_ps(by, ty);
        const __m256 rz = _mm256_sub_ps(bz, tz);
        const __m256 two = _mm256_set1_ps(2.f);
        descent[0] = _mm256_add_ps(
            descent[0],
            _mm256_mul_ps(two, _mm256_sub_ps(_mm256_mul_ps(ry, fz),
                                             _mm256_mul_ps(rz, fy))));
        descent[1] = _mm256_add_ps(
            descent[1],
            _mm256_mul_ps(two, _mm256_sub_ps(_mm256_mul_ps(rz, fx),
                                             _mm256_mul_ps(rx, fz))));
        descent[2] = _mm256_add_ps(
            descent[2],
            _mm256_mul_ps(two, _mm256_sub_ps(_mm256_mul_ps(rx, fy),
                                             _mm256_mul_ps(ry, fx))));
        descent[3] = _mm256_add_ps(descent[3], fx);
        descent[4] = _mm256_add_ps(descent[4], fy);
        descent[5] = _mm256_add_ps(descent[5], fz);
    }
}

}
}  // namespace r3::cpu
#endif

// Native NEON canonical analytic color: two independent four-start vectors.
//
// ARM implementation of evaluate_canonical_color_analytic_batch8, the
// generic eight-lane canonical evaluator called from
// color_conformer_kernel.cpp. A lane is one (query, library conformer, pose)
// pair with its own per-type counts; the batch is packed as
// [feature][component][8 lanes] SoA with padding masks for uneven counts.
// NEON is four lanes wide, so each plane runs as two independent halves
// (lanes 0..3, then 4..7). Unsafe lanes were already evaluated by the
// caller's exact scalar path and are masked off here.
//
// The prepared four-start helpers replaced this route for canonical,
// shared-pose, four-start, field-free color in normal public screening;
// this evaluator serves the configurations they do not cover.
#include "color_canonical_kernel.hpp"
#include "optimizer_exp_neon.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace r3::cpu {
namespace {
// Batch8 SoA index: [atom][component][lane] within a plane's segment.
size_t at(size_t base, int atom, int component, int lane) {
    return base + (size_t(atom) * 3 + component) * COLOR_CONFORMER_LANES + lane;
}

// Loads four lanes and zeroes (+0) the masked ones.
float32x4_t masked_load(const float* p, uint32x4_t mask) {
    return vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(vld1q_f32(p)), mask));
}
}

void evaluate_canonical_color_analytic_batch8(
    const ColorConformerBatch8& batch,
    const std::vector<ColorCanonicalPlane>& planes, const float* posed,
    const float translation[COLOR_CONFORMER_LANES][3],
    const int32_t safe[COLOR_CONFORMER_LANES], bool gradient,
    ColorEvaluation output[COLOR_CONFORMER_LANES],
    uint64_t terms[COLOR_CONFORMER_LANES], ColorConformerCensus* census,
    ColorFieldBatchStats* stats, bool factor, OptimizerExpMode mode) {
    if (batch.n_features < 1 || batch.n_features > MAX_COLOR_FEATURES ||
        batch.n_valid < 1 || batch.n_valid > COLOR_CONFORMER_LANES ||
        !translation || !safe || !output || !terms)
        throw std::invalid_argument("incomplete canonical NEON color batch");
    for (const auto& p : planes)
        if (p.type < 1 || p.type >= batch.n_features ||
            !std::isfinite(p.coefficient) || !std::isfinite(p.width) ||
            !std::isfinite(p.exponent_scale) ||
            !std::isfinite(p.torque_scale) ||
            !std::isfinite(p.translation_scale))
            throw std::invalid_argument("invalid canonical NEON color plane");
    // Census: per-type target-count histogram, 19 bins (0..17, 18+).
    if (census)
        for (int type = 1; type < batch.n_features; ++type)
            for (int lane = 0; lane < batch.n_valid; ++lane)
                ++census->target_count_bins[size_t(type) * 19 +
                                            std::min(
                                                batch.library_count[type][lane],
                                                18)];
    // Exponential mode is immutable call state. Exact/fixed-pose calls use
    // the polynomial, regardless of the optimizer's requested weight mode.
    // Select the inline callable once, not a mode branch per Gaussian term.
    const auto evaluate = [&](auto exponential) {
        bool work = false;
        float total_values[COLOR_CONFORMER_LANES]{};
        float total_gradients[6][COLOR_CONFORMER_LANES]{};
        for (const auto& p : planes) {
            const int type = p.type;
            // Loop bounds are the maxima over safe lanes; shorter lanes are
            // padded and masked. Term counts record each lane's real pairs.
            int nq = 0, nb = 0;
            uint64_t useful = 0;
            for (int lane = 0; lane < batch.n_valid; ++lane)
                if (safe[lane]) {
                    const int q = batch.query_count[type][lane],
                              b = batch.library_count[type][lane];
                    nq = std::max(nq, q);
                    nb = std::max(nb, b);
                    const uint64_t count = uint64_t(q) * uint64_t(b);
                    useful += count;
                    terms[lane] += count;
                    if (stats && q && b) {
                        stats->samples += b;
                        stats->analytic_selected_samples += b;
                        stats->analytic_selected_terms += count;
                    }
                }
            if (!nq || !nb) continue;
            work = true;
            if (census) {
                const size_t cell = size_t(type) * MAX_COLOR_FEATURES + type;
                census->useful_terms_by_interaction[cell] += useful;
                census->issued_lanes_by_interaction[cell] +=
                    uint64_t(nq) * nb * 8;
            }
            // Two independent four-lane halves of the eight-lane batch.
            for (int half = 0; half < 8; half += 4) {
                const auto enabled =
                    vreinterpretq_u32_s32(vld1q_s32(safe + half));
                float tv[3][4]{};
                for (int lane = 0; lane < 4; ++lane)
                    if (half + lane < batch.n_valid && safe[half + lane])
                        for (int k = 0; k < 3; ++k)
                            tv[k][lane] = translation[half + lane][k];
                const auto tx = vld1q_f32(tv[0]), ty = vld1q_f32(tv[1]),
                           tz = vld1q_f32(tv[2]);
                auto total = vld1q_f32(total_values + half);
                float32x4_t d[6];
                for (int c = 0; c < 6; ++c)
                    d[c] = vld1q_f32(total_gradients[c] + half);
                // Per-feature padding mask ANDed with the safe-lane mask.
                const auto mask = [&](const std::vector<int32_t>& masks,
                                      size_t base, int atom) {
                    return vandq_u32(enabled, vreinterpretq_u32_s32(vld1q_s32(
                                                  masks.data() + base +
                                                  size_t(atom) * 8 + half)));
                };
                for (int j = 0; j < nb; ++j) {
                    const auto bm = mask(batch.library_mask,
                                         batch.library_mask_offset[type], j);
                    const auto bx = masked_load(
                        posed + at(batch.library_coordinate_offset[type], j, 0,
                                   half),
                        bm);
                    const auto by = masked_load(
                        posed + at(batch.library_coordinate_offset[type], j, 1,
                                   half),
                        bm);
                    const auto bz = masked_load(
                        posed + at(batch.library_coordinate_offset[type], j, 2,
                                   half),
                        bm);
                    // Library-outer, query-inner; moment arm r = b' - t'.
                    const auto rx = vsubq_f32(bx, tx), ry = vsubq_f32(by, ty),
                               rz = vsubq_f32(bz, tz);
                    auto fx = vdupq_n_f32(0.f), fy = fx, fz = fx;
                    for (int i = 0; i < nq; ++i) {
                        const auto qm = mask(batch.query_mask,
                                             batch.query_mask_offset[type], i);
                        const auto dx = vsubq_f32(
                            masked_load(
                                batch.query_xyz.data() +
                                    at(batch.query_coordinate_offset[type], i,
                                       0, half),
                                qm),
                            bx);
                        const auto dy = vsubq_f32(
                            masked_load(
                                batch.query_xyz.data() +
                                    at(batch.query_coordinate_offset[type], i,
                                       1, half),
                                qm),
                            by);
                        const auto dz = vsubq_f32(
                            masked_load(
                                batch.query_xyz.data() +
                                    at(batch.query_coordinate_offset[type], i,
                                       2, half),
                                qm),
                            bz);
                        const auto d2 = vfmaq_f32(
                            vfmaq_f32(vmulq_f32(dz, dz), dy, dy), dx, dx);
                        // Gaussian term, zeroed bitwise on padded pairs.
                        const auto v = vreinterpretq_f32_u32(vandq_u32(
                            vandq_u32(qm, bm),
                            vreinterpretq_u32_f32(vmulq_n_f32(
                                exponential(vmulq_n_f32(d2, p.exponent_scale)),
                                p.coefficient))));
                        total = vaddq_f32(total, v);
                        if (!gradient) continue;
                        const auto f = vmulq_n_f32(v, p.translation_scale);
                        const auto gx = vmulq_f32(f, dx), gy = vmulq_f32(f, dy),
                                   gz = vmulq_f32(f, dz);
                        // Factored: sum the force, torque once per
                        // target below. Unfactored: torque per pair.
                        if (factor) {
                            fx = vaddq_f32(fx, gx);
                            fy = vaddq_f32(fy, gy);
                            fz = vaddq_f32(fz, gz);
                        } else {
                            const auto torque = vmulq_n_f32(v, p.torque_scale);
                            d[0] = vaddq_f32(
                                d[0], vmulq_f32(torque,
                                                vsubq_f32(vmulq_f32(ry, dz),
                                                          vmulq_f32(rz, dy))));
                            d[1] = vaddq_f32(
                                d[1], vmulq_f32(torque,
                                                vsubq_f32(vmulq_f32(rz, dx),
                                                          vmulq_f32(rx, dz))));
                            d[2] = vaddq_f32(
                                d[2], vmulq_f32(torque,
                                                vsubq_f32(vmulq_f32(rx, dy),
                                                          vmulq_f32(ry, dx))));
                            d[3] = vaddq_f32(d[3], gx);
                            d[4] = vaddq_f32(d[4], gy);
                            d[5] = vaddq_f32(d[5], gz);
                        }
                    }
                    if (gradient && factor) {
                        d[0] = vaddq_f32(
                            d[0], vmulq_n_f32(vsubq_f32(vmulq_f32(ry, fz),
                                                        vmulq_f32(rz, fy)),
                                              2.f));
                        d[1] = vaddq_f32(
                            d[1], vmulq_n_f32(vsubq_f32(vmulq_f32(rz, fx),
                                                        vmulq_f32(rx, fz)),
                                              2.f));
                        d[2] = vaddq_f32(
                            d[2], vmulq_n_f32(vsubq_f32(vmulq_f32(rx, fy),
                                                        vmulq_f32(ry, fx)),
                                              2.f));
                        d[3] = vaddq_f32(d[3], fx);
                        d[4] = vaddq_f32(d[4], fy);
                        d[5] = vaddq_f32(d[5], fz);
                    }
                }
                vst1q_f32(total_values + half, total);
                if (gradient)
                    for (int c = 0; c < 6; ++c)
                        vst1q_f32(total_gradients[c] + half, d[c]);
            }
        }
        for (int lane = 0; lane < batch.n_valid; ++lane)
            if (safe[lane]) {
                output[lane].value = double(total_values[lane]);
                if (gradient)
                    for (int c = 0; c < 6; ++c)
                        output[lane].descent[c + 1] =
                            double(total_gradients[c][lane]);
            }
        if (census && work) ++census->vector_evaluations;
    };
    // Raw-bit exp only for gradient=true optimizer calls; every
    // gradient=false call uses the polynomial.
    if (gradient && mode == OptimizerExpMode::RawBitsUnclamped)
        evaluate([](float32x4_t x) { return neon_exp_rawbits(x, true); });
    else if (gradient && mode == OptimizerExpMode::RawBits)
        evaluate([](float32x4_t x) { return neon_exp_rawbits(x, false); });
    else
        evaluate([](float32x4_t x) { return neon_exp_polynomial(x); });
}
}

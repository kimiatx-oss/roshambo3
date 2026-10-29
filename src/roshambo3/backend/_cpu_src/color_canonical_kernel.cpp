// Direct analytic color evaluator for the canonical identity matrix.
//
// AVX2 implementation of evaluate_canonical_color_analytic_batch8, called
// from color_conformer_kernel.cpp's generic eight-lane conformer evaluator
// when the plan is canonical-analytic (ColorInvariantPlan: every width 1,
// one positive diagonal coefficient, zero off-diagonal coefficients).
// A lane is one
// (query, library conformer, pose) pair with its own per-type counts; data
// is the batch8 [feature][component][8 lanes] SoA, and padding/unsafe lanes
// are masked to zero. Only diagonal planes are visited, using the plan's
// prepared FP32 constants, so no general interaction or field-policy
// machinery runs here.
//
// The prepared four-start helpers replaced this route for canonical,
// shared-pose, four-start, field-free color in normal public screening;
// this evaluator serves the configurations they do not cover. The ARM
// counterpart is color_canonical_neon.cpp.
#include "color_canonical_kernel.hpp"
#include "color_torque_factoring.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace r3 {
namespace cpu {
namespace {

// Batch8 SoA index: [feature][component][lane].
size_t coordinate_index(size_t base, int feature, int component, int lane) {
    return base +
           (size_t(feature) * 3 + size_t(component)) *
               size_t(COLOR_CONFORMER_LANES) +
           size_t(lane);
}

// Batch8 padding-mask index: [feature][lane].
size_t mask_index(size_t base, int feature, int lane) {
    return base + size_t(feature) * size_t(COLOR_CONFORMER_LANES) +
           size_t(lane);
}

#if defined(__AVX2__)
// Keep this path on the selected degree-seven FP32 polynomial. Another exp
// can replace this single canonical call boundary without changing matrix,
// pose, or fallback handling.
__m256 exp8_canonical_polynomial_bounded(__m256 x) {
    const __m256 cutoff = _mm256_set1_ps(-80.0f);
    const __m256 kept = _mm256_cmp_ps(x, cutoff, _CMP_GE_OQ);
    const __m256 bounded = _mm256_max_ps(x, cutoff);
    const __m256 y =
        _mm256_mul_ps(bounded, _mm256_set1_ps(1.4426950408889634074f));
    const __m256 n = _mm256_floor_ps(y);
    const __m256 z = _mm256_mul_ps(_mm256_sub_ps(y, n),
                                   _mm256_set1_ps(0.69314718055994530942f));
    const __m256i exponent = _mm256_cvttps_epi32(n);
    const __m256i bits = _mm256_slli_epi32(
        _mm256_add_epi32(exponent, _mm256_set1_epi32(127)), 23);
    constexpr float coefficient[] = {
        1.0f,         1.0f,          0.5f,          1.0f / 6.0f,
        1.0f / 24.0f, 1.0f / 120.0f, 1.0f / 720.0f, 1.0f / 5040.0f};
    __m256 polynomial = _mm256_set1_ps(coefficient[7]);
    for (int k = 6; k >= 0; --k)
        polynomial =
            _mm256_fmadd_ps(polynomial, z, _mm256_set1_ps(coefficient[k]));
    return _mm256_and_ps(_mm256_mul_ps(polynomial, _mm256_castsi256_ps(bits)),
                         kept);
}
#endif

}  // namespace

void evaluate_canonical_color_analytic_batch8(
    const ColorConformerBatch8& batch, const ColorInvariantPlan& plan,
    const float* posed_library_xyz_fp32,
    const float pose_translation[COLOR_CONFORMER_LANES][3],
    const int32_t safe_lane_mask[COLOR_CONFORMER_LANES], bool gradient,
    ColorEvaluation output[COLOR_CONFORMER_LANES],
    uint64_t term_counts[COLOR_CONFORMER_LANES], ColorConformerCensus* census,
    ColorFieldBatchStats* analytic_stats, bool factor_canonical_torque,
    OptimizerExpMode optimizer_exp_mode) {
    if (batch.n_features < 1 || batch.n_features > MAX_COLOR_FEATURES ||
        batch.n_valid < 1 || batch.n_valid > COLOR_CONFORMER_LANES ||
        pose_translation == nullptr || safe_lane_mask == nullptr ||
        output == nullptr || term_counts == nullptr ||
        plan.n_features() != batch.n_features ||
        !plan.canonical_analytic_supported() ||
        !plan.reduced_matrix_supported_for_avx2())
        throw std::invalid_argument(
            "incomplete canonical analytic color batch");
    const std::vector<ColorCanonicalPlane>& planes = plan.canonical_planes();

    // Census: per-type target-count histogram, 19 bins (0..17, 18+).
    if (census) {
        for (int type = 1; type < batch.n_features; ++type)
            for (int lane = 0; lane < batch.n_valid; ++lane) {
                const int bin = std::min(batch.library_count[type][lane], 18);
                ++census->target_count_bins[size_t(type) * 19 + size_t(bin)];
            }
    }

    // Preserve optional analytic-field counters for callers that request
    // them, while avoiding construction or inspection of field policy state.
    if (analytic_stats) {
        for (const ColorCanonicalPlane& plane : planes) {
            const int type = plane.type;
            for (int lane = 0; lane < batch.n_valid; ++lane) {
                if (!safe_lane_mask[lane]) continue;
                const uint64_t nq = uint64_t(batch.query_count[type][lane]);
                const uint64_t nb = uint64_t(batch.library_count[type][lane]);
                if (nq == 0 || nb == 0) continue;
                analytic_stats->samples += nb;
                analytic_stats->analytic_selected_samples += nb;
                analytic_stats->analytic_selected_terms += nq * nb;
            }
        }
    }

#if defined(__AVX2__)
    // Select an inline callable once per evaluation, never a mutable global
    // or an indirect function-pointer call inside the feature-pair loop.
    const auto evaluate = [&](auto exponential) {
        // Moment-arm origin t' per lane; unsafe lanes stay zero.
        alignas(32) float translation_x[COLOR_CONFORMER_LANES]{};
        alignas(32) float translation_y[COLOR_CONFORMER_LANES]{};
        alignas(32) float translation_z[COLOR_CONFORMER_LANES]{};
        for (int lane = 0; lane < batch.n_valid; ++lane) {
            if (!safe_lane_mask[lane]) continue;
            translation_x[lane] = pose_translation[lane][0];
            translation_y[lane] = pose_translation[lane][1];
            translation_z[lane] = pose_translation[lane][2];
        }
        const __m256i safe_mask_i = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(safe_lane_mask));
        const __m256 tx = _mm256_load_ps(translation_x);
        const __m256 ty = _mm256_load_ps(translation_y);
        const __m256 tz = _mm256_load_ps(translation_z);
        bool any_analytic_work = false;
        __m256 total_value = _mm256_setzero_ps();
        __m256 total_gradient[6];
        for (auto& component : total_gradient)
            component = _mm256_setzero_ps();

        for (const ColorCanonicalPlane& plane : planes) {
            const int type = plane.type;
            if (batch.query_max[type] == 0 || batch.library_max[type] == 0)
                continue;

            // Loop bounds are the maxima over safe lanes; shorter lanes are
            // padded and masked. Term counts record each lane's real pairs,
            // the census records useful versus issued (padded) lanes.
            int max_query = 0;
            int max_library = 0;
            uint64_t useful = 0;
            for (int lane = 0; lane < batch.n_valid; ++lane) {
                if (!safe_lane_mask[lane]) continue;
                const int nq = batch.query_count[type][lane];
                const int nb = batch.library_count[type][lane];
                max_query = std::max(max_query, nq);
                max_library = std::max(max_library, nb);
                useful += uint64_t(nq) * uint64_t(nb);
                term_counts[lane] += uint64_t(nq) * uint64_t(nb);
            }
            if (max_query == 0 || max_library == 0) continue;
            any_analytic_work = true;
            if (census) {
                const size_t cell =
                    size_t(type) * size_t(MAX_COLOR_FEATURES) + size_t(type);
                census->useful_terms_by_interaction[cell] += useful;
                census->issued_lanes_by_interaction[cell] +=
                    uint64_t(max_query) * uint64_t(max_library) *
                    uint64_t(COLOR_CONFORMER_LANES);
            }

            __m256 plane_value = _mm256_setzero_ps();
            __m256 plane_gradient[6];
            for (auto& component : plane_gradient)
                component = _mm256_setzero_ps();

            const __m256 coefficient = _mm256_set1_ps(plane.coefficient);
            const __m256 exponent_scale = _mm256_set1_ps(plane.exponent_scale);
            const __m256 torque_scale = _mm256_set1_ps(plane.torque_scale);
            const __m256 translation_scale =
                _mm256_set1_ps(plane.translation_scale);

            // Factored (color_torque_factoring.hpp): library-outer,
            // one torque per target. Otherwise query-outer, library-inner,
            // with a torque cross product per pair.
            if (factor_canonical_torque && gradient) {
                accumulate_canonical_factored_torque(
                    batch, type, posed_library_xyz_fp32, safe_lane_mask,
                    max_query, max_library, plane.coefficient,
                    plane.exponent_scale, plane.translation_scale, tx, ty, tz,
                    plane_value, plane_gradient, exponential);
            } else
                for (int i = 0; i < max_query; ++i) {
                    const size_t qx_at = coordinate_index(
                        batch.query_coordinate_offset[type], i, 0, 0);
                    const size_t qy_at = coordinate_index(
                        batch.query_coordinate_offset[type], i, 1, 0);
                    const size_t qz_at = coordinate_index(
                        batch.query_coordinate_offset[type], i, 2, 0);
                    const __m256i query_mask_i =
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                            batch.query_mask.data() +
                            mask_index(batch.query_mask_offset[type], i, 0)));
                    const __m256 query_enabled = _mm256_castsi256_ps(
                        _mm256_and_si256(query_mask_i, safe_mask_i));
                    const __m256 qx = _mm256_and_ps(
                        query_enabled,
                        _mm256_loadu_ps(batch.query_xyz.data() + qx_at));
                    const __m256 qy = _mm256_and_ps(
                        query_enabled,
                        _mm256_loadu_ps(batch.query_xyz.data() + qy_at));
                    const __m256 qz = _mm256_and_ps(
                        query_enabled,
                        _mm256_loadu_ps(batch.query_xyz.data() + qz_at));

                    for (int j = 0; j < max_library; ++j) {
                        const size_t bx_at = coordinate_index(
                            batch.library_coordinate_offset[type], j, 0, 0);
                        const size_t by_at = coordinate_index(
                            batch.library_coordinate_offset[type], j, 1, 0);
                        const size_t bz_at = coordinate_index(
                            batch.library_coordinate_offset[type], j, 2, 0);
                        const __m256i library_mask_j =
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                batch.library_mask.data() +
                                mask_index(batch.library_mask_offset[type], j,
                                           0)));
                        const __m256 library_enabled = _mm256_castsi256_ps(
                            _mm256_and_si256(library_mask_j, safe_mask_i));
                        const __m256 active =
                            _mm256_and_ps(query_enabled, library_enabled);
                        const __m256 bx = _mm256_and_ps(
                            library_enabled,
                            _mm256_loadu_ps(posed_library_xyz_fp32 + bx_at));
                        const __m256 by = _mm256_and_ps(
                            library_enabled,
                            _mm256_loadu_ps(posed_library_xyz_fp32 + by_at));
                        const __m256 bz = _mm256_and_ps(
                            library_enabled,
                            _mm256_loadu_ps(posed_library_xyz_fp32 + bz_at));
                        const __m256 dx = _mm256_sub_ps(qx, bx);
                        const __m256 dy = _mm256_sub_ps(qy, by);
                        const __m256 dz = _mm256_sub_ps(qz, bz);
                        const __m256 d2 = _mm256_fmadd_ps(
                            dx, dx,
                            _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)));
                        const __m256 exponent =
                            _mm256_mul_ps(d2, exponent_scale);
                        // Zeroed bitwise on padded or unsafe pairs.
                        const __m256 value = _mm256_and_ps(
                            _mm256_mul_ps(exponential(exponent), coefficient),
                            active);
                        plane_value = _mm256_add_ps(plane_value, value);
                        // descent[1..3] += torque_scale * v * (r x d),
                        // descent[4..6] += translation_scale * v * d.
                        if (gradient) {
                            const __m256 rx = _mm256_sub_ps(bx, tx);
                            const __m256 ry = _mm256_sub_ps(by, ty);
                            const __m256 rz = _mm256_sub_ps(bz, tz);
                            const __m256 torque =
                                _mm256_mul_ps(value, torque_scale);
                            plane_gradient[0] = _mm256_add_ps(
                                plane_gradient[0],
                                _mm256_mul_ps(
                                    torque,
                                    _mm256_sub_ps(_mm256_mul_ps(ry, dz),
                                                  _mm256_mul_ps(rz, dy))));
                            plane_gradient[1] = _mm256_add_ps(
                                plane_gradient[1],
                                _mm256_mul_ps(
                                    torque,
                                    _mm256_sub_ps(_mm256_mul_ps(rz, dx),
                                                  _mm256_mul_ps(rx, dz))));
                            plane_gradient[2] = _mm256_add_ps(
                                plane_gradient[2],
                                _mm256_mul_ps(
                                    torque,
                                    _mm256_sub_ps(_mm256_mul_ps(rx, dy),
                                                  _mm256_mul_ps(ry, dx))));
                            const __m256 translate =
                                _mm256_mul_ps(value, translation_scale);
                            plane_gradient[3] =
                                _mm256_add_ps(plane_gradient[3],
                                              _mm256_mul_ps(translate, dx));
                            plane_gradient[4] =
                                _mm256_add_ps(plane_gradient[4],
                                              _mm256_mul_ps(translate, dy));
                            plane_gradient[5] =
                                _mm256_add_ps(plane_gradient[5],
                                              _mm256_mul_ps(translate, dz));
                        }
                    }
                }

            total_value = _mm256_add_ps(total_value, plane_value);
            if (gradient) {
                for (int component = 0; component < 6; ++component)
                    total_gradient[component] = _mm256_add_ps(
                        total_gradient[component], plane_gradient[component]);
            }
        }
        alignas(32) float values[COLOR_CONFORMER_LANES];
        _mm256_store_ps(values, total_value);
        for (int lane = 0; lane < batch.n_valid; ++lane)
            if (safe_lane_mask[lane]) output[lane].value = double(values[lane]);
        if (gradient)
            for (int component = 0; component < 6; ++component) {
                _mm256_store_ps(values, total_gradient[component]);
                for (int lane = 0; lane < batch.n_valid; ++lane)
                    if (safe_lane_mask[lane])
                        output[lane].descent[component + 1] =
                            double(values[lane]);
            }
        if (census && any_analytic_work) ++census->vector_evaluations;
    };
    // Exact/fixed-pose evaluations retain the degree-seven baseline even if
    // the enclosing optimizer chose raw weights for its gradient trajectory.
    if (gradient && optimizer_exp_mode == OptimizerExpMode::RawBitsUnclamped)
        evaluate(RawBitsUnclampedOptimizerExp{});
    else if (gradient && optimizer_exp_mode == OptimizerExpMode::RawBits)
        evaluate(RawBitsOptimizerExp{});
    else
        evaluate([](__m256 x) { return exp8_canonical_polynomial_bounded(x); });
#else
    (void)posed_library_xyz_fp32;
    (void)gradient;
    if (census) (void)census;
    if (analytic_stats) (void)analytic_stats;
    throw std::logic_error(
        "canonical AVX2 evaluator called without an AVX2 build");
#endif
}

}  // namespace cpu
}  // namespace r3

// Fixed-pose final rescoring: the shape and color cross overlaps of
// independent (query, library, pose) pairs, with no optimizer, no start
// selection and no self volumes.
//
// This file is the architecture-neutral half: input validation, the
// eligibility test that decides which pairs may use the polynomial-exp
// SIMD paths, and the accurate fallback. The SIMD grains themselves are in
// final_rescore_simd_impl.hpp, instantiated by final_rescore_neon.cpp
// (4-wide) or final_rescore_avx2.cpp (8-wide).
//
// Status: mode 0, Accurate,
// ships; modes 1-3 are experiments reachable only through the compile-time
// R3_FINAL_RESCORE_MODE in color_scalar.cpp and the private diagnostic binding.
// All x86 modes lost; across-pair lost on both architectures; ARM within-pair
// was +2.43% on the dense q7 query only. None was promoted.
#include "final_rescore_detail.hpp"
#include <stdexcept>
#include <limits>

namespace r3::cpu {
namespace {
// Nonfinite coordinates are a caller error and refuse (throw). Finite but
// huge ones (> 1e9) are legal and only make the pair ineligible for the
// polynomial paths, which then use the accurate fallback.
bool points_safe(const float* xyz, int n, int stride) {
    if (n < 0 || (n && !xyz) || stride < 3)
        throw std::invalid_argument("invalid final-rescore points");
    bool safe = true;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k)
            if (!std::isfinite(xyz[size_t(i) * stride + k]))
                throw std::invalid_argument(
                    "nonfinite final-rescore coordinate");
            else
                safe &= std::abs(xyz[size_t(i) * stride + k]) <= 1.e9f;
    return safe;
}

// True when a pair may take the polynomial-exp paths (modes 1-3); false routes
// it to `accurate`. Throws on malformed input in every mode, so validation does
// not depend on the mode selected. The pose matrix carries COORD_SCALE (as in
// pair_overlap_at_pose), so its entries are bounded well inside 3. Color
// additionally requires canonical analytic planes and a reduced matrix the
// vector path supports; general matrices keep the accurate fallback.
// The checks run per pair, outside any sample loop.
bool eligible(const FinalRescorePair& p) {
    if (!p.pose)
        throw std::invalid_argument("invalid shared final-rescore pose");
    bool safe = true;
    for (float v : p.pose->matrix)
        safe &= std::abs(v) <= 3.f;
    for (float v : p.pose->translation)
        safe &= std::abs(v) <= 1.e9f;
    safe &= points_safe(p.shape_query_reduced.xyz,
                        p.shape_query_reduced.n_atoms, 3);
    safe &= points_safe(p.shape_library_physical.xyz,
                        p.shape_library_physical.n_atoms, 3);
    if (p.color_plan) {
        if (!p.color_query_reduced || !p.color_library_physical)
            throw std::invalid_argument("incomplete final-rescore color pair");
        const auto& q = p.color_query_reduced->features;
        const auto& b = *p.color_library_physical;
        if (q.n_features != p.color_plan->n_features() ||
            b.n_features != q.n_features)
            throw std::invalid_argument("final-rescore color feature mismatch");
        safe &= points_safe(q.color_xyz, q.n_color, p.query_color_stride);
        safe &= points_safe(b.color_xyz, b.n_color, p.library_color_stride);
        // Segments must tile [0, n_color) in type order starting at type 1,
        // since the kernels index features as seg_offset[t] + i.
        for (const auto* view : {&q, &b}) {
            int offset = 0;
            for (int t = 1; t < view->n_features; ++t) {
                if (view->seg_count[t] < 0 || view->seg_offset[t] != offset ||
                    view->seg_count[t] >
                        std::numeric_limits<int>::max() - offset)
                    throw std::invalid_argument(
                        "invalid final-rescore color segments");
                offset += view->seg_count[t];
            }
            if (offset != view->n_color)
                throw std::invalid_argument(
                    "invalid final-rescore color count");
        }
        safe &= p.color_plan->canonical_analytic_supported() &&
                p.color_plan->reduced_matrix_supported_for_avx2();
        // The exact-final color path's coordinate range, carried into reduced
        // units (x COORD_SCALE) and stepped down one ulp so the comparison is
        // conservative after float rounding.
        const float bound =
            std::nextafter(float(1.e9 * double(COORD_SCALE)), 0.f);
        for (float value : p.pose->translation)
            safe &= std::abs(value) <= bound;
        for (int i = 0; i < q.n_color; ++i)
            for (int k = 0; k < 3; ++k)
                safe &=
                    std::abs(
                        q.color_xyz[size_t(i) * p.query_color_stride + k]) <=
                    bound;
        // Conservative interval bound proves all packed FP32 target points
        // respect the retained exact-final eligibility range, before sampling.
        float maximum[3]{};
        for (int i = 0; i < b.n_color; ++i)
            for (int k = 0; k < 3; ++k)
                maximum[k] = std::max(
                    maximum[k],
                    std::abs(
                        b.color_xyz[size_t(i) * p.library_color_stride + k]));
        for (int k = 0; k < 3; ++k) {
            const float upper =
                std::abs(p.pose->translation[k]) +
                std::abs(p.pose->matrix[3 * k]) * maximum[0] +
                std::abs(p.pose->matrix[3 * k + 1]) * maximum[1] +
                std::abs(p.pose->matrix[3 * k + 2]) * maximum[2];
            // Positive interval bound includes conservative FP32 rounding margin.
            safe &=
                upper * (1.f + 16.f * std::numeric_limits<float>::epsilon()) <=
                bound;
        }
    }
    return safe;
}

// Mode 0 and the fallback for ineligible pairs: the existing accurate shape
// (score.hpp pair_overlap_at_pose, source-order fp32 with Exp2Negative) and
// exact-final color, unchanged. Color keeps its original double result here
// (see FinalRescoreCross). Shape terms are computed from atom counts; color
// terms are the ones score_exact_final_color reports issuing. They are not
// recomputed from the plan's active_interactions(): a canonical plan lists
// every ta <= tb pair with nonzero width there, but both exact-final routes
// visit only the diagonal planes, so that product overcounted any query with
// more than one feature type.
void accurate(const FinalRescorePair& p, FinalRescoreCross& out,
              FinalRescoreCounts* c) {
    out.shape =
        pair_overlap_at_pose(p.shape_query_reduced, p.shape_library_physical,
                             p.pose->matrix, p.pose->translation, nullptr);
    ExactFinalScratchStats scratch_stats;
    uint64_t color_terms = 0;
    out.color =
        p.color_plan
            ? score_exact_final_color(
                  *p.color_query_reduced, *p.color_library_physical,
                  *p.color_plan, *p.pose, p.query_color_stride,
                  p.library_color_stride, p.accurate_color_scratch,
                  c ? &color_terms : nullptr, c ? &scratch_stats : nullptr)
                  .value
            : 0;
    if (c) {
        c->temporary_allocations += scratch_stats.buffer_growths;
        c->allocated_payload_bytes += scratch_stats.payload_bytes_allocated;
        const uint64_t shape = uint64_t(p.shape_query_reduced.n_atoms) *
                               p.shape_library_physical.n_atoms;
        c->shape_terms += shape;
        c->shape_issued_lanes += shape;
        c->color_terms += color_terms;
        c->color_issued_lanes += color_terms;
    }
}
}

// Splits the batch into maximal runs of eligible pairs, each handed whole to
// the SIMD entry; ineligible pairs go one at a time to `accurate`. Lanes are
// therefore only ever filled with genuinely independent pairs, never
// duplicated poses or starts -- across-pair width is never faked.
void final_rescore_batch(const FinalRescorePair* pairs, size_t n,
                         FinalRescoreCross* out, FinalRescoreMode mode,
                         FinalRescoreCounts* c) {
    if (n && (!pairs || !out))
        throw std::invalid_argument("null final-rescore batch");
    if (int(mode) < 0 || int(mode) > 3)
        throw std::invalid_argument("unknown final-rescore mode");
    size_t first = 0;
    while (first < n) {
        const bool safe = eligible(pairs[first]);
        if (mode == FinalRescoreMode::Accurate || !safe) {
            accurate(pairs[first], out[first], c);
            if (c && !safe && mode != FinalRescoreMode::Accurate)
                ++c->accurate_fallback_pairs;
            ++first;
            continue;
        }
        if (mode == FinalRescoreMode::ScalarPolynomial) {
            out[first] = final_detail::scalar_poly(pairs[first], c);
            ++first;
            continue;
        }
        // Dispatch/eligibility is outside sample loops. Contiguous eligible
        // runs retain actual independent-pair ordering, no duplicated lanes.
        size_t end = first + 1;
        while (end < n && eligible(pairs[end]))
            ++end;
        final_rescore_simd(pairs + first, end - first, out + first, mode, c);
        first = end;
    }
}
}

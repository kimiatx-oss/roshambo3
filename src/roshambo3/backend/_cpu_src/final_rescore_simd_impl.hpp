// The two SIMD grains of the fixed-pose final rescore, written once against
// a minimal vector interface V (Vector4 = NEON, Vector8 = AVX2) supplied by
// final_rescore_{neon,avx2}.cpp.
//
//   within (mode 2): one pair at a time; lanes take W consecutive query atoms
//                    (shape) or W same-type target features (color), and two
//                    horizontal reductions finish the pair. A declared FP32
//                    reassociation versus the scalar order (NUMERICS_POLICY sec. 2).
//   across (mode 3): lane k owns pair k and walks that pair's terms in the
//                    scalar order; unequal counts are masked to the longest
//                    pair. No reduction, but the masked lanes are paid for.
//
// Both use the same polynomial exp, so the grain and the exp strategy are
// separable in the measurements. Measured outcome: across lost on both
// architectures even at full width; within lost on x86 and was mixed on ARM.
#pragma once
#include "final_rescore_detail.hpp"
#include <algorithm>

namespace r3::cpu::final_detail {
// Architecture-specific vector operations supplied by separate TUs. All
// gathers/transpose packing here are real work included in kernel timings.
//
// W Gaussian terms exp(scale * |d|^2), multiplied by a 0/1 mask. Padding lanes
// carry d = 0, so V::exp sees exactly 0 -- a safe argument for its floor/int
// conversion -- and returns 1, which the mask zeroes.
template <class V>
typename V::Type terms(const float* dx, const float* dy, const float* dz,
                       const float* scale, const float* mask) {
    auto x = V::load(dx), y = V::load(dy), z = V::load(dz);
    auto d2 = V::add(V::add(V::mul(x, x), V::mul(y, y)), V::mul(z, z));
    return V::mul(V::exp(V::mul(V::load(scale), d2)), V::load(mask));
}

// The pair's library color features, posed (R b + t) and packed xyz-
// contiguous. Same arithmetic as the scalar control; allocation is counted.
inline std::vector<float> pack_color(const FinalRescorePair& p,
                                     FinalRescoreCounts* c) {
    std::vector<float> posed;
    if (!p.color_plan) return posed;
    const auto& b = *p.color_library_physical;
    posed.resize(size_t(b.n_color) * 3);
    for (int j = 0; j < b.n_color; ++j) {
        rotate(*p.pose, b.color_xyz + size_t(j) * p.library_color_stride,
               posed.data() + 3 * j);
        for (int k = 0; k < 3; ++k)
            posed[3 * j + k] += p.pose->translation[k];
    }
    if (c) {
        c->packed_coordinate_bytes += posed.size() * sizeof(float);
        c->temporary_allocations += !posed.empty();
        c->allocated_payload_bytes += posed.capacity() * sizeof(float);
    }
    return posed;
}

// Mode 2. Lanes are filled by scalar stores to aligned staging arrays and
// one vector load -- the gather/transpose is real work and inside the timed
// kernel. Shape lanes hold partial sums across query atoms; color lanes across
// target features of one plane.
template <class V>
FinalRescoreCross within(const FinalRescorePair& p, FinalRescoreCounts* c) {
    constexpr int W = V::width;
    auto shape = V::zero(), color = V::zero();
    alignas(32) float dx[W]{}, dy[W]{}, dz[W]{}, scale[W]{}, mask[W]{};
    const auto& a = p.shape_query_reduced;
    const auto& b = p.shape_library_physical;
    for (int j = 0; j < b.n_atoms; ++j) {
        float mb[3];
        rotate(*p.pose, b.xyz + 3 * j, mb);
        for (int i = 0; i < a.n_atoms; i += W) {
            const int active = std::min(W, a.n_atoms - i);
            for (int k = 0; k < W; ++k) {
                mask[k] = k < active ? 1.f : 0.f;
                // -ln 2: exp2(-|d|^2) as exp(-ln2 |d|^2), as in scalar_poly.
                scale[k] = -.69314718055994530942f;
                dx[k] = dy[k] = dz[k] = 0;
                if (k < active) {
                    dx[k] =
                        (a.xyz[3 * (i + k)] - p.pose->translation[0]) - mb[0];
                    dy[k] = (a.xyz[3 * (i + k) + 1] - p.pose->translation[1]) -
                            mb[1];
                    dz[k] = (a.xyz[3 * (i + k) + 2] - p.pose->translation[2]) -
                            mb[2];
                }
            }
            shape = V::add(shape, terms<V>(dx, dy, dz, scale, mask));
            if (c) {
                c->shape_terms += active;
                c->shape_issued_lanes += W;
            }
        }
    }
    const auto posed = pack_color(p, c);
    if (p.color_plan) {
        const auto& q = p.color_query_reduced->features;
        const auto& l = *p.color_library_physical;
        for (const auto& plane : p.color_plan->canonical_planes()) {
            const int t = plane.type;
            const float decay = float(p.color_plan->reduced_decay(t, t));
            if (decay == 0) continue;
            const auto coefficient =
                V::broadcast(float(p.color_plan->coefficient(t, t)));
            for (int i = 0; i < q.seg_count[t]; ++i) {
                const float* x = q.color_xyz + size_t(q.seg_offset[t] + i) *
                                                   p.query_color_stride;
                for (int j = 0; j < l.seg_count[t]; j += W) {
                    const int active = std::min(W, l.seg_count[t] - j);
                    for (int k = 0; k < W; ++k) {
                        mask[k] = k < active ? 1.f : 0.f;
                        scale[k] = decay;
                        dx[k] = dy[k] = dz[k] = 0;
                        if (k < active) {
                            const float* y =
                                posed.data() + 3 * (l.seg_offset[t] + j + k);
                            dx[k] = x[0] - y[0];
                            dy[k] = x[1] - y[1];
                            dz[k] = x[2] - y[2];
                        }
                    }
                    color = V::fma(coefficient,
                                   terms<V>(dx, dy, dz, scale, mask), color);
                    if (c) {
                        c->color_terms += active;
                        c->color_issued_lanes += W;
                    }
                }
            }
        }
    }
    // One horizontal reduction each for shape and color, per pair.
    if (c) c->horizontal_reductions += 2;
    return {V::reduce(shape), double(V::reduce(color))};
}

// Mode 3: n <= W pairs, lane k owning pair k. Every lane walks i/j over the
// group's maximum counts and masks out-of-range terms, so a short pair rides
// along idle; lane utilization is the measured cost (25% measured on
// AVX2 and 50% on NEON for its two-pair groups). Within a lane, terms are
// added in the scalar order, so no horizontal reduction is needed.
template <class V>
void across(const FinalRescorePair* p, int n, FinalRescoreCross* out,
            FinalRescoreCounts* c) {
    constexpr int W = V::width;
    auto shape = V::zero(), color = V::zero();
    alignas(32) float dx[W]{}, dy[W]{}, dz[W]{}, scale[W]{}, mask[W]{},
        coeff[W]{};
    float rotated[W][3]{};
    int maxq = 0, maxb = 0;
    for (int lane = 0; lane < n; ++lane) {
        maxq = std::max(maxq, p[lane].shape_query_reduced.n_atoms);
        maxb = std::max(maxb, p[lane].shape_library_physical.n_atoms);
    }
    for (int j = 0; j < maxb; ++j) {
        for (int lane = 0; lane < n; ++lane)
            if (j < p[lane].shape_library_physical.n_atoms)
                rotate(*p[lane].pose,
                       p[lane].shape_library_physical.xyz + 3 * j,
                       rotated[lane]);
        for (int i = 0; i < maxq; ++i) {
            int active = 0;
            for (int lane = 0; lane < W; ++lane) {
                mask[lane] = 0;
                scale[lane] = -.69314718055994530942f;
                dx[lane] = dy[lane] = dz[lane] = 0;
                if (lane < n && j < p[lane].shape_library_physical.n_atoms &&
                    i < p[lane].shape_query_reduced.n_atoms) {
                    const auto& a = p[lane].shape_query_reduced;
                    const auto& pose = *p[lane].pose;
                    dx[lane] =
                        (a.xyz[3 * i] - pose.translation[0]) - rotated[lane][0];
                    dy[lane] = (a.xyz[3 * i + 1] - pose.translation[1]) -
                               rotated[lane][1];
                    dz[lane] = (a.xyz[3 * i + 2] - pose.translation[2]) -
                               rotated[lane][2];
                    mask[lane] = 1;
                    ++active;
                }
            }
            shape = V::add(shape, terms<V>(dx, dy, dz, scale, mask));
            if (c) {
                c->shape_terms += active;
                c->shape_issued_lanes += W;
            }
        }
    }
    // Color iterates type t = 1.. in order; a lane whose plan has no such
    // type, or a zero decay (non-canonical/inactive), contributes zeros.
    std::vector<float> posed[W];
    for (int lane = 0; lane < n; ++lane)
        posed[lane] = pack_color(p[lane], c);
    for (int t = 1; t < MAX_COLOR_FEATURES; ++t) {
        maxq = maxb = 0;
        int nq[W]{}, nb[W]{};
        const float* query_xyz[W]{};
        const float* library_xyz[W]{};
        // Matrix/count/segment lookups are immutable for the whole type group,
        // not repeated per Gaussian sample. Both grains retain original type,
        // query and target order within every independent output lane.
        for (int lane = 0; lane < W; ++lane) {
            scale[lane] = coeff[lane] = 0;
            if (lane >= n || !p[lane].color_plan ||
                t >= p[lane].color_plan->n_features())
                continue;
            const auto& pair = p[lane];
            const auto& q = pair.color_query_reduced->features;
            const auto& b = *pair.color_library_physical;
            const float decay = float(pair.color_plan->reduced_decay(t, t));
            if (decay == 0) continue;
            nq[lane] = q.seg_count[t];
            nb[lane] = b.seg_count[t];
            if (nq[lane])
                query_xyz[lane] = q.color_xyz + size_t(q.seg_offset[t]) *
                                                    pair.query_color_stride;
            if (nb[lane])
                library_xyz[lane] = posed[lane].data() + 3 * b.seg_offset[t];
            scale[lane] = decay;
            coeff[lane] = float(pair.color_plan->coefficient(t, t));
            maxq = std::max(maxq, nq[lane]);
            maxb = std::max(maxb, nb[lane]);
        }
        const auto coefficient = V::load(coeff);
        for (int i = 0; i < maxq; ++i)
            for (int j = 0; j < maxb; ++j) {
                int active = 0;
                for (int lane = 0; lane < W; ++lane) {
                    mask[lane] = dx[lane] = dy[lane] = dz[lane] = 0;
                    if (i >= nq[lane] || j >= nb[lane]) continue;
                    const float* x = query_xyz[lane] +
                                     size_t(i) * p[lane].query_color_stride;
                    const float* y = library_xyz[lane] + 3 * j;
                    dx[lane] = x[0] - y[0];
                    dy[lane] = x[1] - y[1];
                    dz[lane] = x[2] - y[2];
                    mask[lane] = 1;
                    ++active;
                }
                color = V::fma(coefficient, terms<V>(dx, dy, dz, scale, mask),
                               color);
                if (c) {
                    c->color_terms += active;
                    c->color_issued_lanes += W;
                }
            }
    }
    alignas(32) float s[W], v[W];
    V::store(s, shape);
    V::store(v, color);
    for (int lane = 0; lane < n; ++lane)
        out[lane] = {s[lane], double(v[lane])};
}

// Architecture entry body: mode 2 pair by pair; mode 3 in groups of W pairs,
// the last group partial. The caller guarantees every pair is eligible.
template <class V>
void run(const FinalRescorePair* p, size_t n, FinalRescoreCross* out,
         FinalRescoreMode mode, FinalRescoreCounts* c) {
    if (mode == FinalRescoreMode::WithinPair) {
        for (size_t i = 0; i < n; ++i)
            out[i] = within<V>(p[i], c);
        return;
    }
    for (size_t i = 0; i < n; i += V::width)
        across<V>(p + i, int(std::min(n - i, size_t(V::width))), out + i, c);
}
}

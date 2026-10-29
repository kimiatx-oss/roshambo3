// The scoring core: self-overlap, the pair sum, the gradient, and the search.
#pragma once

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "kernel.hpp"

namespace r3 {
namespace cpu {

using simd::Exp2Negative;
using simd::VecF32;

//: A query or library conformer, already in the reduced frame.
//: SoA is deliberate but not load-bearing at width 1 -- the grain is chosen
//: elsewhere, and this layout is the one the atom-grain arms want (a
//: conformer's atoms contiguous), which is the OPPOSITE of what the corpus's
//: 32-row tile provides.
struct Conformer {
    const float* xyz;  // [n_atoms * 3], interleaved x,y,z per atom
    int n_atoms;
};

//: Atom capacity of the fixed-size SoA copy of a library conformer
//: (`ConformerSoA`, field_avx2.hpp and field_neon.hpp). It covers every count
//: the corpus format can store -- `n_real` is one byte -- so no stored
//: conformer is refused or truncated. It was 64, and the AVX2 field path read
//: past it for larger conformers: wrong scores, and at times a crash.
inline constexpr int kConformerSoACapacity = 256;

// EVERY FUNCTION IN THIS FILE IS `static`, for the reason kernel.hpp's
// quaternion-helpers comment gives in full: this header is compiled into
// both of _roshambo3_cpu's translation units, at two different ISA levels,
// and `static` is what stops the linker from merging their two COMDAT
// bodies into one under LTO. `Conformer` and `PairResult` stay
// ordinary namespace-scope types with external linkage -- unlike the
// functions, their identity (not their codegen) has to be the SAME across
// both TUs, because `run_avx512_grain`/`run_avx512_block`
// (grain_avx512.cpp) are declared taking them by reference and defined in
// the other TU's object file; giving them internal linkage too would make
// each TU see a different type and fail to LINK (loudly, not silently --
// the safer failure mode, but not the one wanted here).

// --- the reduced self-overlap ---------------------------------------------
//
// S_AA = sum_i sum_j exp2(-|a_i - a_j|^2), over all ORDERED pairs -- so two
// coincident atoms give four times one atom, not twice. `reference/overlap.py`
// says so explicitly and it is easy to get wrong by summing i<j.
static inline float self_overlap(const Conformer& a, OpCensus* census) {
    float s = 0.0f;
    for (int i = 0; i < a.n_atoms; ++i) {
        const float ax = a.xyz[3 * i], ay = a.xyz[3 * i + 1],
                    az = a.xyz[3 * i + 2];
        for (int j = 0; j < a.n_atoms; ++j) {
            const float dx = ax - a.xyz[3 * j];
            const float dy = ay - a.xyz[3 * j + 1];
            const float dz = az - a.xyz[3 * j + 2];
            const float d2 = dx * dx + dy * dy + dz * dz;
            s += Exp2Negative(VecF32(-d2)).v;
        }
    }
    if (census) {
        census->exp2_calls += uint64_t(a.n_atoms) * a.n_atoms;
        census->self_overlap_terms += uint64_t(a.n_atoms) * a.n_atoms;
    }
    return s;
}

// --- the exact analytic pair sum at a fixed pose ---------------------------
//
// S_AB = sum_i sum_j exp2(-|(a_i - t) - M b_j|^2), accumulated in SOURCE ORDER
// and in fp32. Both of those are the specification rather than a choice:
// `doc/NUMERICS_POLICY.md` section 2 pins one reduction order and fp32
// accumulation throughout, and lists a reordered reduction among the six axes
// that reach the same saturated output distribution as --use_fast_math.
//
// `m` is the pose matrix ALREADY carrying COORD_SCALE, as the device's does.
static inline float pair_overlap_at_pose(const Conformer& a, const Conformer& b,
                                         const float m[9], const float t[3],
                                         OpCensus* census) {
    float s = 0.0f;
    for (int j = 0; j < b.n_atoms; ++j) {
        const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1],
                    bz = b.xyz[3 * j + 2];
        const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
        const float mby = m[3] * bx + m[4] * by + m[5] * bz;
        const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
        for (int i = 0; i < a.n_atoms; ++i) {
            const float dx = (a.xyz[3 * i] - t[0]) - mbx;
            const float dy = (a.xyz[3 * i + 1] - t[1]) - mby;
            const float dz = (a.xyz[3 * i + 2] - t[2]) - mbz;
            const float d2 = dx * dx + dy * dy + dz * dz;
            s += Exp2Negative(VecF32(-d2)).v;
        }
    }
    if (census) {
        const uint64_t terms = uint64_t(a.n_atoms) * b.n_atoms;
        census->exp2_calls += terms;
        census->pair_terms_rescore += terms;
    }
    return s;
}

// --- the gradient ---------------------------------------------------------
//
// Returns the DESCENT direction, which is what the Adagrad step consumes --
// the same sign convention the device uses, and the reason
// `reference.overlap_gradient`'s translation components are the exact negative
// of these (cos = -1.000000000 over five random pairs, magnitude ratio 1).
//
// `gnorm` is 1/(S_AA + S_BB), folded into the per-term coefficient rather than
// applied to the finished sum: free there, and it keeps the accumulation O(1)
// whatever the molecule size.
static inline void get_gradient(const Conformer& a, const Conformer& b,
                                const float m[9], const float t[3], float gnorm,
                                float g[7], float* out_s, OpCensus* census) {
    for (int k = 0; k < 7; ++k)
        g[k] = 0.0f;
    float vs = 0.0f;
    const float tfac = MINUS_2LN2 * gnorm;

    for (int j = 0; j < b.n_atoms; ++j) {
        const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1],
                    bz = b.xyz[3 * j + 2];
        const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
        const float mby = m[3] * bx + m[4] * by + m[5] * bz;
        const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
        // C = 2 M b, the space-frame lever arm.
        const float cx = 2.0f * mbx, cy = 2.0f * mby, cz = 2.0f * mbz;
        for (int i = 0; i < a.n_atoms; ++i) {
            const float dx = (a.xyz[3 * i] - t[0]) - mbx;
            const float dy = (a.xyz[3 * i + 1] - t[1]) - mby;
            const float dz = (a.xyz[3 * i + 2] - t[2]) - mbz;
            const float d2 = dx * dx + dy * dy + dz * dz;
            const float kij = Exp2Negative(VecF32(-d2)).v;
            const float temp = tfac * kij;
            // g[1..3]: the torque C x delta. NOT dV/dq_i -- there is no fourth
            // rotational degree of freedom and g[0] stays zero, which is
            // exactly why the additive update rule cannot move q[0] and the
            // exponential map can.
            g[1] += temp * (cy * dz - cz * dy);
            g[2] += temp * (cz * dx - cx * dz);
            g[3] += temp * (cx * dy - cy * dx);
            g[4] += temp * dx;
            g[5] += temp * dy;
            g[6] += temp * dz;
            vs += kij;
        }
    }
    if (census) {
        const uint64_t terms = uint64_t(a.n_atoms) * b.n_atoms;
        census->exp2_calls += terms;
        census->pair_terms_optimizer += terms;
    }
    if (out_s) *out_s = vs;
}

// --- one Adagrad step, on the manifold ------------------------------------
//
// Stage timing put this whole function at 20.5% of the field path at 24
// threads, O(1) in atom count -- which is why the analytic path never surfaced
// it. Its cost splits three ways: the six sqrt/divide pairs, the libm sin/cos
// inside `quat_left_exp`, and the 16-multiply quaternion product.
//
// Note the loop does SIX sqrt/divide pairs, not seven: index 0 is skipped
// because `g[0]` is identically zero (the gradient loop never writes it).
static inline void adagrad_exp_step(float q[4], float t[3], const float g[7],
                                    float cache[7], float lr_q, float lr_t) {
    float d[3], dt[3];
// MEASURED AND REJECTED -- opt-in only, kept as the evidence for arm 2.
// Vectorizing the six sqrt/divide pairs IN PLACE buys nothing: 0.9826x on the
// min and 1.0133x on the median (Zen 3, 4 pinned cores, gaps 0.1-1.7%), against
// an ablation upper bound of 1.057x for removing them outright. The pack and
// unpack of six scalars -- masked load, vector op, store, extract -- costs what
// the vector operation saves. It is also not exact in situ (see below), so it
// is a numerics change that buys nothing, which is the one combination never
// worth shipping. The batched arm 2 is where the 5.7% actually lives.
#if defined(__AVX2__) && defined(ROSHAMBO3_STEP_VECTOR)
    // The six sqrt/divide pairs, done as one vector op each instead of six.
    //
    // **This is worth 5.7% of the search window on Zen 3 and NOTHING on Cascade
    // Lake** (ablation: Zen 3 +5.7%, Cascade Lake -0.9%, both with min-gaps under
    // 0.5%). Same instruction count removed on both; the difference is whether
    // the out-of-order engine had the capacity to hide it. Measuring only on
    // the quiet machine would have killed this arm.
    //
    // Exact, not approximate: `vsqrtps` and `vdivps` are IEEE operations on the
    // same inputs, and the six lanes are independent, so this changes WHERE the
    // values live and not WHEN anything is added. It is therefore not a
    // reassociation and owes no regrade (NUMERICS_POLICY) -- and
    // tests/test_optimizer_step_is_bit_identical.py holds that claim to
    // bit-identity rather than asserting it.
    //
    // Indices 1..6 only: g[0] is identically zero because the gradient loop
    // never writes it. A masked load keeps the read inside the 7-float arrays.
    {
        const __m256i keep = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);
        const __m256 gv = _mm256_maskload_ps(&g[1], keep);
        const __m256 cv = _mm256_maskload_ps(&cache[1], keep);
        // Scalar writes `cache[i] += g[i]*g[i]`, which GCC contracts to an FMA
        // under this project's -ffp-contract setting; matching it here is what
        // makes the result bit-identical rather than merely close.
        const __m256 nc = _mm256_fmadd_ps(gv, gv, cv);
        _mm256_maskstore_ps(&cache[1], keep, nc);
        const __m256 lr = _mm256_setr_ps(-lr_q, -lr_q, -lr_q, -lr_t, -lr_t,
                                         -lr_t, 0.0f, 0.0f);
        const __m256 den =
            _mm256_add_ps(_mm256_sqrt_ps(nc), _mm256_set1_ps(1e-8f));
        const __m256 r = _mm256_div_ps(_mm256_mul_ps(lr, gv), den);
        alignas(32) float o[8];
        _mm256_store_ps(o, r);
        d[0] = o[0];
        d[1] = o[1];
        d[2] = o[2];
        dt[0] = o[3];
        dt[1] = o[4];
        dt[2] = o[5];
    }
#else
    for (int i = 0; i < 3; ++i) {
        cache[i + 1] += g[i + 1] * g[i + 1];
        cache[i + 4] += g[i + 4] * g[i + 4];
        d[i] = -lr_q * g[i + 1] / (std::sqrt(cache[i + 1]) + 1e-8f);
        dt[i] = -lr_t * g[i + 4] / (std::sqrt(cache[i + 4]) + 1e-8f);
    }
#endif
    quat_left_exp(q, d);
    for (int i = 0; i < 3; ++i)
        t[i] += dt[i];
}

#if defined(__AVX2__)
//: `LANES` pairs' Adagrad state, held SoA so a component is a vector.
//:
//: Arm 2. Arm 1 vectorized the six sqrt/divide pairs of ONE pair in place
//: and bought nothing -- packing six scalars into a register and extracting
//: them again cost what the vector operation saved. Held this way the packing
//: does not exist: `cache[i]` is already eight pairs' i-th component, contiguous
//: and aligned. Measured ceiling on the update alone: **2.99x** exact.
template <int LANES>
struct BatchedPoseState {
    static_assert(LANES == 8, "one AVX2 register per component");
    alignas(32) float q[4][LANES];
    alignas(32) float t[3][LANES];
    alignas(32) float cache[7][LANES];
    alignas(32) float g[7][LANES];

    void reset(int n) {
        for (int p = 0; p < LANES; ++p) {
            q[0][p] = 1.0f;
            q[1][p] = q[2][p] = q[3][p] = 0.0f;
            for (int i = 0; i < 3; ++i)
                t[i][p] = 0.0f;
            for (int i = 0; i < 7; ++i) {
                cache[i][p] = 0.0f;
                g[i][p] = 0.0f;
            }
        }
        (void)n;
    }
};

//: The Adagrad step for `LANES` pairs at once -- the scaling batched, the
//: transcendental left alone.
//:
//: `sin`/`cos` stay scalar and stay libm on purpose. Vectorizing them needs a
//: bounded-domain polynomial, which is an ACCURACY change judged on recall
//: (as `exp2` is judged) -- and the split measured 2.99x for the
//: exact half against 7.78x with the polynomial, so three quarters of the prize
//: is available without owing a regrade. The polynomial is a separate arm.
template <int LANES>
static inline void batched_adagrad_exp_step(BatchedPoseState<LANES>& S,
                                            int n_lanes, float lr_q,
                                            float lr_t) {
    __m256 d[3], dt[3];
    for (int i = 0; i < 3; ++i) {
        const __m256 gq = _mm256_load_ps(S.g[i + 1]);
        const __m256 gt = _mm256_load_ps(S.g[i + 4]);
        const __m256 cq =
            _mm256_fmadd_ps(gq, gq, _mm256_load_ps(S.cache[i + 1]));
        const __m256 ct =
            _mm256_fmadd_ps(gt, gt, _mm256_load_ps(S.cache[i + 4]));
        _mm256_store_ps(S.cache[i + 1], cq);
        _mm256_store_ps(S.cache[i + 4], ct);
        const __m256 eps = _mm256_set1_ps(1e-8f);
        d[i] = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(-lr_q), gq),
                             _mm256_add_ps(_mm256_sqrt_ps(cq), eps));
        dt[i] = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(-lr_t), gt),
                              _mm256_add_ps(_mm256_sqrt_ps(ct), eps));
    }
    alignas(32) float dx[3][LANES], sv[LANES], cv[LANES];
    for (int i = 0; i < 3; ++i)
        _mm256_store_ps(dx[i], d[i]);
    for (int p = 0; p < n_lanes; ++p) {
        const float a2 =
            dx[0][p] * dx[0][p] + dx[1][p] * dx[1][p] + dx[2][p] * dx[2][p];
        if (a2 > 1e-12f) {
            const float aa = std::sqrt(a2);
            cv[p] = std::cos(aa);
            sv[p] = std::sin(aa) / aa;
        } else {
            cv[p] = 1.0f - 0.5f * a2;
            sv[p] = 1.0f - a2 * (1.0f / 6.0f);
        }
    }
    for (int p = n_lanes; p < LANES; ++p) {
        cv[p] = 1.0f;
        sv[p] = 0.0f;
    }
    const __m256 sm = _mm256_load_ps(sv), cm = _mm256_load_ps(cv);
    const __m256 e0 = cm, e1 = _mm256_mul_ps(sm, d[0]),
                 e2 = _mm256_mul_ps(sm, d[1]), e3 = _mm256_mul_ps(sm, d[2]);
    const __m256 q0 = _mm256_load_ps(S.q[0]), q1 = _mm256_load_ps(S.q[1]),
                 q2 = _mm256_load_ps(S.q[2]), q3 = _mm256_load_ps(S.q[3]);
    _mm256_store_ps(
        S.q[0],
        _mm256_sub_ps(
            _mm256_mul_ps(e0, q0),
            _mm256_fmadd_ps(e1, q1,
                            _mm256_fmadd_ps(e2, q2, _mm256_mul_ps(e3, q3)))));
    _mm256_store_ps(
        S.q[1],
        _mm256_fmadd_ps(e0, q1,
                        _mm256_fmadd_ps(e1, q0,
                                        _mm256_sub_ps(_mm256_mul_ps(e2, q3),
                                                      _mm256_mul_ps(e3, q2)))));
    _mm256_store_ps(
        S.q[2],
        _mm256_fmadd_ps(e0, q2,
                        _mm256_fmadd_ps(e2, q0,
                                        _mm256_sub_ps(_mm256_mul_ps(e3, q1),
                                                      _mm256_mul_ps(e1, q3)))));
    _mm256_store_ps(
        S.q[3],
        _mm256_fmadd_ps(e0, q3,
                        _mm256_fmadd_ps(e1, q2,
                                        _mm256_sub_ps(_mm256_mul_ps(e3, q0),
                                                      _mm256_mul_ps(e2, q1)))));
    for (int i = 0; i < 3; ++i)
        _mm256_store_ps(S.t[i], _mm256_add_ps(_mm256_load_ps(S.t[i]), dt[i]));
}
#endif  // __AVX2__

//: One (query, library) pair's result, in the reduced frame.
struct PairResult {
    float tanimoto_shape = 0.0f;
    float s_ab = 0.0f;  // reduced overlap at the winning pose
    //: The FIELD's own estimate of `s_ab` at the same pose, in the same units:
    //: `sum_j F(M b_j + t)` is exactly `sum_i sum_j exp2(-|d_ij|^2)`, which is
    //: what `pair_overlap_at_pose` computes exactly. Left at -1 when nothing
    //: asked for it, because computing it costs one extra field evaluation
    //: per start and the shipping path has no use for it.
    float s_ab_field = -1.0f;
    float s_aa = 0.0f;
    float s_bb = 0.0f;
    float pose_q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float pose_t[3] = {0.0f, 0.0f, 0.0f};
    int best_start = 0;
};

// --- the pose search -----------------------------------------------------
//
// `n_steps` Adagrad steps per start, no convergence test -- which is the
// SHIPPED schedule and a warp-width decision rather than an algorithmic one.
// Kept here so this kernel is the control a convergence test is measured
// against.
//
// `lr_t` is in scaled Angstrom, i.e. the caller has already multiplied by
// COORD_SCALE, exactly as the CUDA launchers do -- and the reported
// translation is divided by it on the way out. Getting that wrong moves the
// optimizer 1/0.7768 = 1.29x further per step, which changes the search rather
// than the arithmetic (NUMERICS_POLICY section 4).
static inline PairResult optimize_pair(const Conformer& a, const Conformer& b,
                                       float s_aa, float s_bb, int start_mode,
                                       int n_steps, float lr_q,
                                       float lr_t_scaled, OpCensus* census) {
    PairResult best;
    best.s_aa = s_aa;
    best.s_bb = s_bb;
    best.tanimoto_shape = -1.0f;

    const float gnorm = 1.0f / (s_aa + s_bb);
    const int n_starts = start_count(start_mode);

    for (int n = 0; n < n_starts; ++n) {
        float q_start[4];
        start_quaternion(start_mode, n, q_start);
        float r_start[9];
        quat_to_mat(q_start, r_start);
        // The conjugation's precondition, checked rather than assumed.
        if (!matrix_is_signed_permutation(r_start, SIGNED_PERM_TOL)) {
            continue;  // refused by the caller; see interface.cpp
        }

        // Every start begins at the IDENTITY, with R^T folded into the pose
        // matrix. `M_eff = Q~ R^T`, formed once per step, so the atom loop
        // reads the same coordinates for every start and pays nothing extra.
        float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float t[3] = {0.0f, 0.0f, 0.0f};
        float cache[7] = {0, 0, 0, 0, 0, 0, 0};

        float g[7], s_ab = 0.0f;
        for (int step = 0; step <= n_steps; ++step) {
            // The composite pose: conjugated rotation, then the start's
            // transpose. q_start is a signed permutation so its inverse is its
            // conjugate.
            float q_conj[4] = {q_start[0], -q_start[1], -q_start[2],
                               -q_start[3]};
            float q_eff[4];
            quat_mul(q, q_conj, q_eff);
            float m[9];
            quat_to_mat(q_eff, m);
            for (int k = 0; k < 9; ++k)
                m[k] *= COORD_SCALE;

            if (step == n_steps) {
                // The final evaluation, with no update -- this is the exact
                // analytic score at the converged pose, which is the number
                // that gets reported. Tier 3 is the specification and no
                // approximate tier may be the source of a reported number.
                s_ab = pair_overlap_at_pose(a, b, m, t, census);
                break;
            }
            get_gradient(a, b, m, t, gnorm, g, &s_ab, census);
            adagrad_exp_step(q, t, g, cache, lr_q, lr_t_scaled);
            if (census) ++census->optimizer_steps;
        }

        const float den = s_aa + s_bb - s_ab;
        // Consistent self-overlaps keep `den` at least half their sum, so a
        // non-positive one means inconsistent inputs: it scores 0 here and on
        // the device alike, and the engine refuses the row. Every CPU
        // shape site uses this same guard.
        const float ts = (den > 0.0f) ? (s_ab / den) : 0.0f;
        // Descending score, ties by ASCENDING start index -- the same
        // tie-break the device's cross-start argmax uses, so the answer is
        // independent of the order the starts happen to be visited in.
        if (ts > best.tanimoto_shape) {
            best.tanimoto_shape = ts;
            best.s_ab = s_ab;
            best.best_start = n;
            float q_conj[4] = {q_start[0], -q_start[1], -q_start[2],
                               -q_start[3]};
            quat_mul(q, q_conj, best.pose_q);
            for (int k = 0; k < 3; ++k)
                best.pose_t[k] = t[k] / COORD_SCALE;
        }
        if (census) ++census->starts;
    }
    if (census) ++census->pairs;
    return best;
}

}  // namespace cpu
}  // namespace r3

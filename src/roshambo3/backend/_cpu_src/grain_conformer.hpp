// Grain 3: lane = library conformer. The GPU's grain, at CPU width.
//
// The GPU arrangement: one thread is one (query, library)
// pair, so a warp is 32 INDEPENDENT PAIRS and costs the union of its 32 lanes'
// choices rather than the choice. This is that arrangement at width 8 or 16.
//
// WHAT IT BUYS, and it is structural rather than incidental: **there is no
// horizontal reduction anywhere**. Each lane accumulates its own pair's
// gradient and its own overlap, so the seven accumulators are simply seven
// vectors that are never summed across lanes. The atom grains in `grain.hpp`
// pay one `ReduceSum` per accumulator per pair; this pays none. It is also the
// only grain whose inner loop has no ragged tail on the atom count -- the tail
// moves to the PAIR count, which is padded once per block.
//
// WHAT IT COSTS, and it is the register question: the
// pose and Adagrad state is per lane, so `q` (4), `t` (3) and `cache` (7) are
// **14 live vectors** where the atom grains hold them in scalars. Against 16
// YMM on AVX2 that is the whole file; against 32 ZMM on AVX-512 it is half.
// That count is suggestive rather than decisive -- there
// are far more physical registers than architectural ones, and a well-placed
// spill can be forwarded by the renamer rather than reaching L1 -- so what is
// reported is spill traffic and an ablation, not the count.
//
// AND IT WANTS A DIFFERENT LAYOUT. A vector load must gather W conformers'
// atom k, which is the transpose of what `Conformer` holds and exactly what the
// corpus's 32-row tile provides. A cooperating-lane
// grain and the tile are SUBSTITUTES, not independent wins, and must not be
// banked separately. The transpose is built here per block rather than read
// from a tiled corpus, so its cost is visible instead of hidden in the producer.
//
// COLOR: this grain does NOT foreclose combo modes. A pair-dependent
// Gaussian width or prefactor is a per-lane quantity like everything else here,
// so it becomes one more vector; nothing in the structure assumes the exponent
// is shared across lanes. That is a stronger position than the atom grains,
// where a per-pair width would have to be broadcast into a reduction.
#pragma once

#include <vector>

#include "grain.hpp"

namespace r3 {
namespace cpu {

namespace {
//: `LaneBlock::build`'s body, extracted for the same reason
//: `soa_conformer_fill` exists in grain.hpp: internal linkage per TU rather
//: than one merged COMDAT copy across ISA levels. NOT currently ODR-used
//: from grain_avx512.cpp's TU -- only `interface.cpp` calls `.build()`; the
//: AVX-512 grain only ever receives an already-built `const LaneBlock&`
//: (`run_avx512_block`, grain_avx512.cpp) -- so this is prophylactic, not a
//: response to a live leak. It stays paired with `SoaConformer` for the same
//: reason: both are shared-struct array-filling loops the auto-vectorizer
//: could treat differently at `-mavx512f` the moment either TU calls them.
inline void lane_block_fill(std::vector<float>& x, std::vector<float>& y,
                            std::vector<float>& z, std::vector<float>& w,
                            std::vector<float>& s_bb, int& max_atoms,
                            int& lanes, int& n_valid, int& sum_atoms,
                            const Conformer* confs, const float* self_bb, int n,
                            int W) {
    lanes = W;
    n_valid = n;
    max_atoms = 0;
    sum_atoms = 0;
    for (int c = 0; c < n; ++c) {
        if (confs[c].n_atoms > max_atoms) max_atoms = confs[c].n_atoms;
        sum_atoms += confs[c].n_atoms;
    }
    const size_t sz = size_t(max_atoms) * size_t(W);
    x.assign(sz, 0.0f);
    y.assign(sz, 0.0f);
    z.assign(sz, 0.0f);
    w.assign(sz, 0.0f);
    s_bb.assign(size_t(W), 1.0f);
    for (int c = 0; c < n; ++c) {
        s_bb[size_t(c)] = self_bb[c];
        for (int k = 0; k < confs[c].n_atoms; ++k) {
            const size_t o = size_t(k) * size_t(W) + size_t(c);
            x[o] = confs[c].xyz[3 * k];
            y[o] = confs[c].xyz[3 * k + 1];
            z[o] = confs[c].xyz[3 * k + 2];
            // Padding again: 1.0f for a real atom, 0.0f for padding, and for
            // a lane with no pair at all. One multiply, never a branch,
            // and it covers the ragged ATOM count and the ragged LANE
            // count with the same mechanism.
            w[o] = 1.0f;
        }
    }
}
}  // namespace

//: W library conformers, transposed so that atom k of all W is contiguous.
//:
//: Laid out [atom][lane] rather than [lane][atom]: the inner loop walks atoms
//: and wants one aligned load per component per atom.
struct LaneBlock {
    std::vector<float> x, y, z, w;  // [max_atoms * lanes]
    std::vector<float> s_bb;        // [lanes]
    int max_atoms = 0;
    int lanes = 0;
    int n_valid = 0;  // how many lanes carry a real pair
    //: Total real atoms across the block's lanes. The census needs it: a term
    //: count of `n_A * max_atoms * n_valid` is neither what the kernel executes
    //: (that is padded to `lanes`) nor what is real, and the atom grains report
    //: real terms -- so a consumer comparing grains would be comparing units.
    int sum_atoms = 0;

    //: `always_inline`: see `SoaConformer::build` in grain.hpp.
    __attribute__((always_inline)) void
    build(const Conformer* confs, const float* self_bb, int n, int W) {
        lane_block_fill(x, y, z, w, s_bb, max_atoms, lanes, n_valid, sum_atoms,
                        confs, self_bb, n, W);
    }
};

// --- the conformer-grain search --------------------------------------------
//
// One call scores W pairs that share a query. Everything below is per lane.
template <class Tag, bool Poly>
inline void
optimize_block_conformer(Tag d, const Conformer& a, const LaneBlock& blk,
                         float s_aa, int start_mode, int n_steps, float lr_q,
                         float lr_t_scaled, float cutoff_d2, float converge_tol,
                         int only_start, PairResult* out, OpCensus* census) {
    using V = decltype(Zero(d));
    const int W = Lanes(d);
    const int n_starts = start_count(start_mode);

    // gnorm = 1 / (s_aa + s_bb) is per lane, because s_bb is.
    V vs_aa = Set(d, s_aa);
    V vs_bb = Load(d, blk.s_bb.data());
    V gnorm = Zero(d);
    {
        alignas(64) float tmp[64];
        Store(Add(vs_aa, vs_bb), d, tmp);
        for (int l = 0; l < W; ++l)
            tmp[l] = 1.0f / tmp[l];
        gnorm = Load(d, tmp);
    }

    // THREAD-LOCAL AND REUSED, not allocated per block.
    //
    // **This is tidiness, not a fix, and the comment that used to stand here
    // said otherwise.** It claimed allocator contention was why this grain
    // stopped scaling past four threads. That was a hypothesis, the measurement
    // never supported it, and review falsified it directly: per-block
    // allocation under the corrected partition scales identically (1.14M
    // against 1.13M start-pairs/s at 24 threads). The four-thread ceiling was
    // the partition snapping thread boundaries to query boundaries, and with
    // four queries in the fixture that left four threads with work.
    //
    // Reusing them is still right -- it removes an allocation per block for
    // nothing -- so it stays, with the reason corrected rather than the claim
    // retained. `thread_local` because the lifetime is exactly a thread's and
    // threading is internal to the extension.
    thread_local std::vector<float> best_t, best_sab, best_q, best_tr;
    thread_local std::vector<int> best_start;
    best_t.assign(size_t(W), -1.0f);
    best_sab.assign(size_t(W), 0.0f);
    best_q.assign(size_t(W) * 4, 0.0f);
    best_tr.assign(size_t(W) * 3, 0.0f);
    best_start.assign(size_t(W), 0);
    for (int l = 0; l < W; ++l)
        best_q[size_t(l) * 4] = 1.0f;

    for (int n = 0; n < n_starts; ++n) {
        // Convergence arm, second pass: run exactly one start, so the per-start
        // score trajectory can be assembled from outside without the kernel
        // having to carry a per-start output array. A diagnostic knob, not a
        // schedule -- `only_start < 0` is every start, which is the shipped
        // behavior and the default.
        if (only_start >= 0 && n != only_start) continue;
        float q_start[4];
        start_quaternion(start_mode, n, q_start);
        float r_start[9];
        quat_to_mat(q_start, r_start);
        if (!matrix_is_signed_permutation(r_start, SIGNED_PERM_TOL)) continue;

        // Per-lane state. THE 14 LIVE VECTORS the register question is about.
        V q0 = Set(d, 1.0f), q1 = Zero(d), q2 = Zero(d), q3 = Zero(d);
        V t0 = Zero(d), t1 = Zero(d), t2 = Zero(d);
        V c1 = Zero(d), c2 = Zero(d), c3 = Zero(d);
        V c4 = Zero(d), c5 = Zero(d), c6 = Zero(d);

        V s_ab = Zero(d);
        for (int step = 0; step <= n_steps; ++step) {
            // The composite pose, per lane. `quat_to_mat` and the conjugation
            // are done lanewise through a scratch buffer rather than
            // vectorized: they are O(steps) where the atom loop is
            // O(steps * n_a * n_b), so ~1/700th of the work, and a lanewise
            // sqrt/branch here costs less than it would cost to get a
            // vectorized normalization subtly wrong.
            alignas(64) float mq[16][4], mm[16][9];
            alignas(64) float bq0[64], bq1[64], bq2[64], bq3[64];
            Store(q0, d, bq0);
            Store(q1, d, bq1);
            Store(q2, d, bq2);
            Store(q3, d, bq3);
            for (int l = 0; l < W; ++l) {
                const float qq[4] = {bq0[l], bq1[l], bq2[l], bq3[l]};
                const float qc[4] = {q_start[0], -q_start[1], -q_start[2],
                                     -q_start[3]};
                quat_mul(qq, qc, mq[l]);
                quat_to_mat(mq[l], mm[l]);
                for (int k = 0; k < 9; ++k)
                    mm[l][k] *= COORD_SCALE;
            }
            V m[9];
            for (int k = 0; k < 9; ++k) {
                alignas(64) float col[64];
                for (int l = 0; l < W; ++l)
                    col[l] = mm[l][k];
                m[k] = Load(d, col);
            }

            V g1 = Zero(d), g2 = Zero(d), g3 = Zero(d);
            V g4 = Zero(d), g5 = Zero(d), g6 = Zero(d), vs = Zero(d);
            const bool last = (step == n_steps);

            for (int k = 0; k < blk.max_atoms; ++k) {
                const float* bx = &blk.x[size_t(k) * size_t(W)];
                const V vbx = Load(d, bx);
                const V vby = Load(d, &blk.y[size_t(k) * size_t(W)]);
                const V vbz = Load(d, &blk.z[size_t(k) * size_t(W)]);
                const V vbw = Load(d, &blk.w[size_t(k) * size_t(W)]);
                const V mbx =
                    Add(Add(Mul(m[0], vbx), Mul(m[1], vby)), Mul(m[2], vbz));
                const V mby =
                    Add(Add(Mul(m[3], vbx), Mul(m[4], vby)), Mul(m[5], vbz));
                const V mbz =
                    Add(Add(Mul(m[6], vbx), Mul(m[7], vby)), Mul(m[8], vbz));
                const V cx = Add(mbx, mbx), cy = Add(mby, mby),
                        cz = Add(mbz, mbz);
                for (int i = 0; i < a.n_atoms; ++i) {
                    const V dx = Sub(Sub(Set(d, a.xyz[3 * i]), t0), mbx);
                    const V dy = Sub(Sub(Set(d, a.xyz[3 * i + 1]), t1), mby);
                    const V dz = Sub(Sub(Set(d, a.xyz[3 * i + 2]), t2), mbz);
                    const V d2 =
                        Add(Add(Mul(dx, dx), Mul(dy, dy)), Mul(dz, dz));
                    // The cutoff is a UNION test across the lanes, which is
                    // the whole question on the GPU: a warp of 32
                    // pays for the union of its lanes' choices rather than the
                    // choice, and the cutoff lost on the GPU partly for that
                    // reason. At width 8 the union is over 8, and this measures
                    // whether that is enough for it to pay.
                    //
                    // Masked, never branched per lane -- a lane whose term is
                    // beyond the cutoff still rides along. The saving comes
                    // only when EVERY lane is beyond it, which is exactly the
                    // union, and skipping then avoids the transcendental and
                    // the whole gradient accumulation for that (i, k).
                    if (census) ++census->cutoff_visits;
                    if (cutoff_d2 > 0.0f && AllGreater(d, d2, cutoff_d2)) {
                        if (census) ++census->cutoff_skips;
                        continue;
                    }
                    const V kij =
                        Mul(Exp2Grade<Poly>::apply(Sub(Zero(d), d2)), vbw);
                    vs = Add(vs, kij);
                    if (!last) {
                        const V temp = Mul(Mul(Set(d, MINUS_2LN2), gnorm), kij);
                        g1 = Add(g1, Mul(temp, Sub(Mul(cy, dz), Mul(cz, dy))));
                        g2 = Add(g2, Mul(temp, Sub(Mul(cz, dx), Mul(cx, dz))));
                        g3 = Add(g3, Mul(temp, Sub(Mul(cx, dy), Mul(cy, dx))));
                        g4 = Add(g4, Mul(temp, dx));
                        g5 = Add(g5, Mul(temp, dy));
                        g6 = Add(g6, Mul(temp, dz));
                    }
                }
            }
            s_ab = vs;
            // The census, per lane and per term. Counted here rather than
            // derived, for the census's reason: a derivation of what the loop SHOULD
            // do is exactly the thing that goes wrong quietly. `exp2_calls`
            // counts the PADDED terms because that is what the kernel executes
            // -- a lane with no atom still evaluates the polynomial and is
            // zeroed by the mask, and charging only the real terms would make
            // this grain look cheaper than it is.
            if (census) {
                const uint64_t padded =
                    uint64_t(a.n_atoms) * uint64_t(blk.max_atoms) * uint64_t(W);
                // REAL terms: sum over lanes of n_A * that lane's atom count,
                // which is what the atom grains report.
                const uint64_t real =
                    uint64_t(a.n_atoms) * uint64_t(blk.sum_atoms);
                census->exp2_calls += padded;
                if (last)
                    census->pair_terms_rescore += real;
                else
                    census->pair_terms_optimizer += real;
            }
            if (last) break;

            // Adagrad, per lane, no reduction. The exponential map's sin/cos
            // are lanewise for the same reason quat_to_mat is.
            alignas(64) float bg[7][64], bc[7][64];
            Store(g1, d, bg[1]);
            Store(g2, d, bg[2]);
            Store(g3, d, bg[3]);
            Store(g4, d, bg[4]);
            Store(g5, d, bg[5]);
            Store(g6, d, bg[6]);
            Store(c1, d, bc[1]);
            Store(c2, d, bc[2]);
            Store(c3, d, bc[3]);
            Store(c4, d, bc[4]);
            Store(c5, d, bc[5]);
            Store(c6, d, bc[6]);
            alignas(64) float bt0[64], bt1[64], bt2[64];
            Store(t0, d, bt0);
            Store(t1, d, bt1);
            Store(t2, d, bt2);
            // The UNION again, and deliberately so. A lane that has converged
            // saves nothing on its own -- the vector is stepped either way --
            // so the loop can only exit when every lane in the block agrees.
            // That is the warp-union argument applied to convergence rather than to a
            // cutoff, and the cutoff arm measured the same union firing on 5.6% of
            // evaluations, so the prior here is not encouraging.
            bool all_converged = (converge_tol > 0.0f);
            for (int l = 0; l < W; ++l) {
                float qq[4] = {bq0[l], bq1[l], bq2[l], bq3[l]};
                float tt[3] = {bt0[l], bt1[l], bt2[l]};
                float gg[7] = {0,        bg[1][l], bg[2][l], bg[3][l],
                               bg[4][l], bg[5][l], bg[6][l]};
                float cc[7] = {0,        bc[1][l], bc[2][l], bc[3][l],
                               bc[4][l], bc[5][l], bc[6][l]};
                const float t_before[3] = {tt[0], tt[1], tt[2]};
                const float q_before[4] = {qq[0], qq[1], qq[2], qq[3]};
                adagrad_exp_step(qq, tt, gg, cc, lr_q, lr_t_scaled);
                // Convergence arm. How far this lane moved on this step, in the
                // reduced frame: the translation delta and the quaternion
                // delta, which is what "converged" has to mean for a pose.
                if (converge_tol > 0.0f) {
                    float dt2 = 0.0f, dq2 = 0.0f;
                    for (int c = 0; c < 3; ++c) {
                        const float e = tt[c] - t_before[c];
                        dt2 += e * e;
                    }
                    for (int c = 0; c < 4; ++c) {
                        const float e = qq[c] - q_before[c];
                        dq2 += e * e;
                    }
                    if (dt2 > converge_tol * converge_tol ||
                        dq2 > converge_tol * converge_tol) {
                        all_converged = false;
                    }
                }
                bq0[l] = qq[0];
                bq1[l] = qq[1];
                bq2[l] = qq[2];
                bq3[l] = qq[3];
                bt0[l] = tt[0];
                bt1[l] = tt[1];
                bt2[l] = tt[2];
                for (int j = 1; j < 7; ++j)
                    bc[j][l] = cc[j];
            }
            q0 = Load(d, bq0);
            q1 = Load(d, bq1);
            q2 = Load(d, bq2);
            q3 = Load(d, bq3);
            t0 = Load(d, bt0);
            t1 = Load(d, bt1);
            t2 = Load(d, bt2);
            c1 = Load(d, bc[1]);
            c2 = Load(d, bc[2]);
            c3 = Load(d, bc[3]);
            c4 = Load(d, bc[4]);
            c5 = Load(d, bc[5]);
            c6 = Load(d, bc[6]);
            // Scaled by the lane count so this is PER PAIR, as it is in every
            // other grain. The kernel really executes one traversal per block,
            // but a census whose units change with the grain is a trap for the
            // consumer that divides by it -- ceiling estimates do exactly that.
            if (census) census->optimizer_steps += uint64_t(blk.n_valid);
            if (all_converged) {
                if (census) census->converged_blocks += 1;
                // One more pass for the exact rescore at the converged pose:
                // Tier 3 is the specification and an early exit may not skip
                // it (NUMERICS_POLICY §2).
                q0 = Load(d, bq0);
                q1 = Load(d, bq1);
                q2 = Load(d, bq2);
                q3 = Load(d, bq3);
                t0 = Load(d, bt0);
                t1 = Load(d, bt1);
                t2 = Load(d, bt2);
                step = n_steps - 1;  // the ++ makes the next pass the last
            }
        }

        // Per-lane best, with the same tie-break: descending score, ties by
        // ascending start index.
        alignas(64) float bsab[64], bden[64], bq0f[64], bq1f[64], bq2f[64],
            bq3f[64], bt0f[64], bt1f[64], bt2f[64];
        Store(s_ab, d, bsab);
        Store(Sub(Add(vs_aa, vs_bb), s_ab), d, bden);
        Store(q0, d, bq0f);
        Store(q1, d, bq1f);
        Store(q2, d, bq2f);
        Store(q3, d, bq3f);
        Store(t0, d, bt0f);
        Store(t1, d, bt1f);
        Store(t2, d, bt2f);
        for (int l = 0; l < blk.n_valid; ++l) {
            const float ts = (bden[l] > 0.0f) ? (bsab[l] / bden[l]) : 0.0f;
            if (ts > best_t[size_t(l)]) {
                best_t[size_t(l)] = ts;
                best_sab[size_t(l)] = bsab[l];
                best_start[size_t(l)] = n;
                const float qq[4] = {bq0f[l], bq1f[l], bq2f[l], bq3f[l]};
                const float qc[4] = {q_start[0], -q_start[1], -q_start[2],
                                     -q_start[3]};
                quat_mul(qq, qc, &best_q[size_t(l) * 4]);
                best_tr[size_t(l) * 3 + 0] = bt0f[l] / COORD_SCALE;
                best_tr[size_t(l) * 3 + 1] = bt1f[l] / COORD_SCALE;
                best_tr[size_t(l) * 3 + 2] = bt2f[l] / COORD_SCALE;
            }
        }
        if (census) census->starts += uint64_t(blk.n_valid);
    }

    for (int l = 0; l < blk.n_valid; ++l) {
        out[l].tanimoto_shape = best_t[size_t(l)];
        out[l].s_ab = best_sab[size_t(l)];
        out[l].s_aa = s_aa;
        out[l].s_bb = blk.s_bb[size_t(l)];
        out[l].best_start = best_start[size_t(l)];
        for (int k = 0; k < 4; ++k)
            out[l].pose_q[k] = best_q[size_t(l) * 4 + k];
        for (int k = 0; k < 3; ++k)
            out[l].pose_t[k] = best_tr[size_t(l) * 3 + k];
        if (census) ++census->pairs;
    }
}

}  // namespace cpu
}  // namespace r3

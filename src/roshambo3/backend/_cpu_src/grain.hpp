// The vectorization grain: which axis the lanes are spread across.
//
// On a GPU one thread is one (query, library) pair, so a warp
// is 32 independent pairs and costs the UNION of its 32 lanes' choices rather
// than the choice. That is why the distance cutoff, start-mode pruning and
// per-pair adaptivity all lost there. At width 8 the union is over 8 lanes and
// the calculus changes -- but only once it is known which axis the lanes are
// spread across, because the candidates differ in what they can skip, what
// state they must hold, and whether they want the corpus's tile layout at all.
//
// TWO ATOM GRAINS LIVE HERE. The third candidate, lane = library conformer,
// needs eight conformers' atom k contiguous -- which is the corpus's 32-row
// tile, a different data path -- and is not in this header.
//
//   QUERY_ATOM   lanes over the query's atoms. ANALYTIC ONLY, and that is not
//                an oversight: the field path's inner loop runs over library
//                atoms alone, which is the entire point of precomputing a
//                query field, so there is no query-atom loop there to
//                vectorize. n_A is constant for the whole screen, so the
//                padding is decided once per query rather than once per pair.
//
//   LIBRARY_ATOM lanes over the library conformer's atoms. Applies to both
//                paths. n_B varies pair to pair (12-50 in the panel), so the
//                tail is paid per pair -- and it is the only grain of the two
//                that could later COMPRESS on a cutoff rather than merely mask,
//                because the surviving atoms are the lanes.
//
// BOTH ARE TEMPLATED ON THE TAG, and that is load-bearing rather than tidy.
// Running the SAME grain source at width 1 and width 8 separates two questions
// that would otherwise arrive together: does restructuring the loop change the
// answer, and does the width change it? At width 1 a grain must reproduce the
// scalar kernel's ranked output exactly; whatever moves at width 8 is then the
// reduction order alone, which is the accepted change the reference anchor
// is calibrated for.
//
// WIDTH 1 IS NOT BIT-IDENTICAL TO THE ANCHOR, AND IT CANNOT CHEAPLY BE MADE SO.
//
// The template exists so the same grain source runs at width 1 and width 8,
// the intent being that width 1 pins the arithmetic while the loop structure
// changes, leaving the width as the only variable at width 8. That intent is
// only partly achievable and the reason is worth recording, because the next
// person will try the same thing.
//
// Measured, and each step ruled something out:
//
//   * the terms are provably IDENTICAL. A term-by-term replication of both
//     inner expressions over a real pair finds no differing `kij`.
//   * the summation ORDER is identical -- j outer, i inner, in both.
//   * summing those identical terms in that order by hand reproduces the
//     ANCHOR exactly and the grain not at all.
//   * so the difference is the compiler's, and it is FMA CONTRACTION: GCC's
//     default `-ffp-contract=fast` contracts the grain's `d2` and leaves the
//     anchor's identical-looking `dx*dx + dy*dy + dz*dz` alone. Over 2e6 random
//     triples the anchor's spelling agrees with plain mul+add on 100.0000% and
//     with an FMA chain on 86.65%; the rescore's measured agreement was 85.55%.
//   * it is NOT the association (rewriting it moved nothing), NOT
//     auto-vectorization, and NOT loop unrolling -- all tested by ablation.
//
// **It is not fixable by spelling**, because the contraction is applied to
// plain `Mul`/`Add` regardless of how they are written, and it is
// context-dependent: the same expression in two loop bodies gets two answers.
// `-ffp-contract=on` would make it ISO-specified and does make an isolated
// replica agree -- but on the full kernel it is itself a declared numerics
// change that MOVES THE ANCHOR: 0.0061% of rows identical to the committed
// fixture, max |dT| 1.6e-4. Adopting it means regenerating the fixture every
// optimized path is measured against, to buy an isolation that is a
// convenience rather than a requirement.
//
// So the arithmetic-identity ambition is dropped and the DECOMPOSITION is kept,
// which is what was actually wanted:
//
//     anchor -> width 1 : contraction (the compiler's choice)
//     width 1 -> width 8 : the reduction order, and nothing else
//
// Both are measured, and both are judged where they must be -- on ranked
// output against the reference anchor, never bitwise. That is not a retreat
// to a weaker test; it is the test the policy specifies, and this
// exercise is a good demonstration of why: a bitwise gate here would have been
// failed by a compiler's discretionary FMA placement.
//
// COLOR IS NOT FORECLOSED BY EITHER ATOM GRAIN, but they are not equal
// on it.
//
//   LIBRARY_ATOM carries it cleanly. A pair-dependent Gaussian width or
//   prefactor is constant across the lanes of one pair -- the lanes are library
//   atoms of a single pair -- so it broadcasts once per pair and costs one more
//   uniform value, not one more vector.
//
//   QUERY_ATOM carries it at the same cost for the same reason: its lanes are
//   also within one pair. What neither can do cheaply is a per-ATOM-TYPE width,
//   which would vary WITHIN the lanes and become a gather; the color combo modes
//   need per-pair parameters, not per-atom ones, so that is not the case in
//   question.
//
// Both are therefore weaker than the conformer grain, where a per-pair
// parameter is a per-lane vector like everything else, but neither forecloses.
//
// PADDING IS A MULTIPLY BY A 0/1 MASK, NEVER A BRANCH. The padding branch is
// this codebase's highest-yield recurring defect: the 4th coordinate
// component is a 0/1 mask and not a radius, and a loop that iterates to L with
// a `w == 0` skip pays twice, in traffic and in branch. It was worth 1.41-1.73x
// to one GPU arm. Multiplying `kij` by `w` costs one FLOP per term, is exact
// for real atoms (w is exactly 1.0f, and multiplication by 1.0f moves no bits)
// and exactly zero for padding, so it changes the reduction's LENGTH but not
// any real lane's value.
#pragma once

#include <vector>

// score.hpp, not kernel.hpp: `Conformer`, `PairResult`, `self_overlap` and
// `adagrad_exp_step` all live there, and this header uses all four. Including
// kernel.hpp alone compiled only when a translation unit happened to include
// score.hpp first, which interface.cpp stopped doing.
#include "score.hpp"

namespace r3 {
namespace cpu {

enum class Grain { QUERY_ATOM = 0, LIBRARY_ATOM = 1 };

//: Which `exp2` a grain uses, selected at COMPILE time.
//:
//: A runtime flag would put a branch in the innermost loop, which is the one
//: place this kernel cannot afford one -- and it would also make the two grades
//: share a code path, so a measurement of one would carry the other's register
//: pressure. Templated instead, at the cost of two instantiations.
//:
//: The grade is a separate axis from the width ON PURPOSE. A change of
//: transcendental implementation is LARGER than a reassociation, and falls
//: outside the reference anchor's scope, so bundling it
//: into the width gate would make that gate unreadable.
template <bool Poly>
struct Exp2Grade;

template <>
struct Exp2Grade<false> {
    static const char* name() {
        return "libm";
    }

    template <class V>
    static V apply(V x) {
        return Exp2Negative(x);
    }
};

template <>
struct Exp2Grade<true> {
    static const char* name() {
        return "minimax_d5";
    }

    template <class V>
    static V apply(V x) {
        return Exp2NegativeMinimax(x);
    }
};

namespace {
//: The actual fill/transpose loop behind `SoaConformer::build`, pulled out
//: of the class and into an anonymous namespace so each translation unit
//: keeps its OWN copy (see kernel.hpp's quaternion-helpers comment for
//: the full mechanism). A class's member functions cannot individually be
//: made `static` the way a free function can -- linkage is a property of
//: the ENCLOSING TYPE -- and `SoaConformer` itself must stay an ordinary,
//: externally-linked type: `run_avx512_grain` (grain_avx512.cpp) takes a
//: `const SoaConformer*` across the TU boundary, so both TUs need to agree
//: it is the SAME type. This is why the fix is "extract the body, force the
//: member to dissolve into its caller" rather than "wrap the struct."
//:
//: This is not a theoretical concern for THIS loop specifically: GCC's
//: auto-vectorizer turns it into a `vpermt2ps`/zmm-register transpose under
//: `-mavx512f` (confirmed by disassembly) while `-march=x86-64-v3` alone
//: leaves it scalar -- so with a shared inline body, whichever
//: TU's compile happened to win the COMDAT merge decided whether calling
//: this from the v3 field path SIGILLs on a machine without AVX-512.
inline void soa_conformer_fill(std::vector<float>& x, std::vector<float>& y,
                               std::vector<float>& z, std::vector<float>& w,
                               int& n_atoms, int& n_padded, const Conformer& c,
                               int lanes) {
    n_atoms = c.n_atoms;
    n_padded = ((c.n_atoms + lanes - 1) / lanes) * lanes;
    x.assign(static_cast<size_t>(n_padded), 0.0f);
    y.assign(static_cast<size_t>(n_padded), 0.0f);
    z.assign(static_cast<size_t>(n_padded), 0.0f);
    w.assign(static_cast<size_t>(n_padded), 0.0f);
    for (int i = 0; i < c.n_atoms; ++i) {
        x[i] = c.xyz[3 * i];
        y[i] = c.xyz[3 * i + 1];
        z[i] = c.xyz[3 * i + 2];
        w[i] = 1.0f;  // exactly 1.0f: multiplying by it moves no bits
    }
}
}  // namespace

//: A conformer in structure-of-arrays form, padded to a whole number of lanes.
//:
//: `Conformer` in score.hpp is interleaved x,y,z per atom, which is what the
//: scalar kernel wants and what the corpus hands over. A vector load needs
//: eight atoms' x contiguous, so the coordinates are transposed once here.
//:
//: The repack cost is not hidden: for the QUERY it is paid once per screen and
//: is free at any panel size; for the LIBRARY it is paid once per conformer per
//: pair, 3 * n_B stores against n_A * n_B * 34 FLOPs of inner loop -- about
//: 0.1% at the panel's mean sizes. It is counted in the census either way.
struct SoaConformer {
    std::vector<float> x, y, z, w;
    int n_atoms = 0;
    int n_padded = 0;

    //: `always_inline`, not just `inline`: this compiler's inlining
    //: decisions move under innocuous-looking changes, so "probably
    //: inlined at -O3" is not the guarantee needed here. Forced
    //: inlining means NO out-of-line body of `build` itself is ever
    //: emitted -- it dissolves entirely into its caller, leaving only the
    //: anonymous-namespace `soa_conformer_fill` call, which is TU-private.
    __attribute__((always_inline)) void build(const Conformer& c, int lanes) {
        soa_conformer_fill(x, y, z, w, n_atoms, n_padded, c, lanes);
    }
};

// --- the gradient, lanes over QUERY atoms ----------------------------------
//
// The j-loop stays scalar and the i-loop vectorizes. The seven accumulators are
// VECTORS held across BOTH loops and reduced exactly once per pair, so the
// horizontal reduction is paid once rather than once per j -- which is the
// structural reason to prefer accumulating in the inner loop's lane space over
// reducing per outer iteration.
template <class Tag, bool Poly>
inline void get_gradient_query_grain(Tag d, const SoaConformer& a,
                                     const Conformer& b, const float m[9],
                                     const float t[3], float gnorm, float g[7],
                                     float* out_s, OpCensus* census) {
    using V = decltype(Zero(d));
    const int W = Lanes(d);
    const float tfac = MINUS_2LN2 * gnorm;

    V g1 = Zero(d), g2 = Zero(d), g3 = Zero(d);
    V g4 = Zero(d), g5 = Zero(d), g6 = Zero(d), vs = Zero(d);

    for (int j = 0; j < b.n_atoms; ++j) {
        const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1],
                    bz = b.xyz[3 * j + 2];
        const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
        const float mby = m[3] * bx + m[4] * by + m[5] * bz;
        const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
        const V vmbx = Set(d, mbx), vmby = Set(d, mby), vmbz = Set(d, mbz);
        const V cx = Set(d, 2.0f * mbx), cy = Set(d, 2.0f * mby),
                cz = Set(d, 2.0f * mbz);
        const V t0 = Set(d, t[0]), t1 = Set(d, t[1]), t2 = Set(d, t[2]);
        const V vtfac = Set(d, tfac);

        for (int i = 0; i < a.n_padded; i += W) {
            const V ax = Load(d, &a.x[i]), ay = Load(d, &a.y[i]),
                    az = Load(d, &a.z[i]), aw = Load(d, &a.w[i]);
            const V dx = Sub(Sub(ax, t0), vmbx);
            const V dy = Sub(Sub(ay, t1), vmby);
            const V dz = Sub(Sub(az, t2), vmbz);
            const V d2 = Add(Add(Mul(dx, dx), Mul(dy, dy)), Mul(dz, dz));
            // The padding contributes exactly zero, by multiply.
            const V kij = Mul(Exp2Grade<Poly>::apply(Sub(Zero(d), d2)), aw);
            const V temp = Mul(vtfac, kij);
            g1 = MulAdd(temp, Sub(Mul(cy, dz), Mul(cz, dy)), g1);
            g2 = MulAdd(temp, Sub(Mul(cz, dx), Mul(cx, dz)), g2);
            g3 = MulAdd(temp, Sub(Mul(cx, dy), Mul(cy, dx)), g3);
            g4 = MulAdd(temp, dx, g4);
            g5 = MulAdd(temp, dy, g5);
            g6 = MulAdd(temp, dz, g6);
            vs = Add(vs, kij);
        }
    }
    g[0] = 0.0f;
    g[1] = ReduceSum(d, g1);
    g[2] = ReduceSum(d, g2);
    g[3] = ReduceSum(d, g3);
    g[4] = ReduceSum(d, g4);
    g[5] = ReduceSum(d, g5);
    g[6] = ReduceSum(d, g6);
    if (out_s) *out_s = ReduceSum(d, vs);
    if (census) {
        const uint64_t terms = uint64_t(a.n_atoms) * b.n_atoms;
        census->exp2_calls += uint64_t(a.n_padded) * b.n_atoms;  // PADDED
        census->pair_terms_optimizer += terms;
    }
}

// --- the gradient, lanes over LIBRARY atoms --------------------------------
//
// The loops swap: j vectorizes and i is scalar and inner. `M*b` and `C = 2*M*b`
// become vector quantities computed once per group of W library atoms, which is
// the one place this grain does strictly less work than the query grain -- it
// amortizes the pose transform across lanes rather than recomputing it per
// scalar j.
template <class Tag, bool Poly>
inline void
get_gradient_library_grain(Tag d, const Conformer& a, const SoaConformer& b,
                           const float m[9], const float t[3], float gnorm,
                           float g[7], float* out_s, OpCensus* census) {
    using V = decltype(Zero(d));
    const int W = Lanes(d);
    const float tfac = MINUS_2LN2 * gnorm;

    V g1 = Zero(d), g2 = Zero(d), g3 = Zero(d);
    V g4 = Zero(d), g5 = Zero(d), g6 = Zero(d), vs = Zero(d);
    const V vtfac = Set(d, tfac);

    for (int j = 0; j < b.n_padded; j += W) {
        const V bx = Load(d, &b.x[j]), by = Load(d, &b.y[j]),
                bz = Load(d, &b.z[j]), bw = Load(d, &b.w[j]);
        const V mbx = Add(Add(Mul(Set(d, m[0]), bx), Mul(Set(d, m[1]), by)),
                          Mul(Set(d, m[2]), bz));
        const V mby = Add(Add(Mul(Set(d, m[3]), bx), Mul(Set(d, m[4]), by)),
                          Mul(Set(d, m[5]), bz));
        const V mbz = Add(Add(Mul(Set(d, m[6]), bx), Mul(Set(d, m[7]), by)),
                          Mul(Set(d, m[8]), bz));
        const V cx = Add(mbx, mbx), cy = Add(mby, mby), cz = Add(mbz, mbz);

        for (int i = 0; i < a.n_atoms; ++i) {
            const V ax = Set(d, a.xyz[3 * i] - t[0]);
            const V ay = Set(d, a.xyz[3 * i + 1] - t[1]);
            const V az = Set(d, a.xyz[3 * i + 2] - t[2]);
            const V dx = Sub(ax, mbx), dy = Sub(ay, mby), dz = Sub(az, mbz);
            const V d2 = Add(Add(Mul(dx, dx), Mul(dy, dy)), Mul(dz, dz));
            const V kij = Mul(Exp2Grade<Poly>::apply(Sub(Zero(d), d2)), bw);
            const V temp = Mul(vtfac, kij);
            g1 = MulAdd(temp, Sub(Mul(cy, dz), Mul(cz, dy)), g1);
            g2 = MulAdd(temp, Sub(Mul(cz, dx), Mul(cx, dz)), g2);
            g3 = MulAdd(temp, Sub(Mul(cx, dy), Mul(cy, dx)), g3);
            g4 = MulAdd(temp, dx, g4);
            g5 = MulAdd(temp, dy, g5);
            g6 = MulAdd(temp, dz, g6);
            vs = Add(vs, kij);
        }
    }
    g[0] = 0.0f;
    g[1] = ReduceSum(d, g1);
    g[2] = ReduceSum(d, g2);
    g[3] = ReduceSum(d, g3);
    g[4] = ReduceSum(d, g4);
    g[5] = ReduceSum(d, g5);
    g[6] = ReduceSum(d, g6);
    if (out_s) *out_s = ReduceSum(d, vs);
    if (census) {
        const uint64_t terms = uint64_t(a.n_atoms) * b.n_atoms;
        census->exp2_calls += uint64_t(a.n_atoms) * b.n_padded;  // PADDED
        census->pair_terms_optimizer += terms;
    }
}

// --- the exact rescore, both grains ----------------------------------------
//
// Tier 3, and it stays exact: `NUMERICS_POLICY.md` §2 makes every approximate
// tier's job candidate selection and never final scoring. The only thing the
// grain changes here is the summation order.
template <class Tag, bool Poly>
inline float pair_overlap_query_grain(Tag d, const SoaConformer& a,
                                      const Conformer& b, const float m[9],
                                      const float t[3], OpCensus* census) {
    using V = decltype(Zero(d));
    const int W = Lanes(d);
    V s = Zero(d);
    for (int j = 0; j < b.n_atoms; ++j) {
        const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1],
                    bz = b.xyz[3 * j + 2];
        const V vmbx = Set(d, m[0] * bx + m[1] * by + m[2] * bz);
        const V vmby = Set(d, m[3] * bx + m[4] * by + m[5] * bz);
        const V vmbz = Set(d, m[6] * bx + m[7] * by + m[8] * bz);
        const V t0 = Set(d, t[0]), t1 = Set(d, t[1]), t2 = Set(d, t[2]);
        for (int i = 0; i < a.n_padded; i += W) {
            const V dx = Sub(Sub(Load(d, &a.x[i]), t0), vmbx);
            const V dy = Sub(Sub(Load(d, &a.y[i]), t1), vmby);
            const V dz = Sub(Sub(Load(d, &a.z[i]), t2), vmbz);
            const V d2 = Add(Add(Mul(dx, dx), Mul(dy, dy)), Mul(dz, dz));
            s = Add(s, Mul(Exp2Grade<Poly>::apply(Sub(Zero(d), d2)),
                           Load(d, &a.w[i])));
        }
    }
    if (census) {
        const uint64_t terms = uint64_t(a.n_atoms) * b.n_atoms;
        census->exp2_calls += uint64_t(a.n_padded) * b.n_atoms;
        census->pair_terms_rescore += terms;
    }
    return ReduceSum(d, s);
}

template <class Tag, bool Poly>
inline float pair_overlap_library_grain(Tag d, const Conformer& a,
                                        const SoaConformer& b, const float m[9],
                                        const float t[3], OpCensus* census) {
    using V = decltype(Zero(d));
    const int W = Lanes(d);
    V s = Zero(d);
    for (int j = 0; j < b.n_padded; j += W) {
        const V bx = Load(d, &b.x[j]), by = Load(d, &b.y[j]),
                bz = Load(d, &b.z[j]), bw = Load(d, &b.w[j]);
        const V mbx = Add(Add(Mul(Set(d, m[0]), bx), Mul(Set(d, m[1]), by)),
                          Mul(Set(d, m[2]), bz));
        const V mby = Add(Add(Mul(Set(d, m[3]), bx), Mul(Set(d, m[4]), by)),
                          Mul(Set(d, m[5]), bz));
        const V mbz = Add(Add(Mul(Set(d, m[6]), bx), Mul(Set(d, m[7]), by)),
                          Mul(Set(d, m[8]), bz));
        for (int i = 0; i < a.n_atoms; ++i) {
            const V dx = Sub(Set(d, a.xyz[3 * i] - t[0]), mbx);
            const V dy = Sub(Set(d, a.xyz[3 * i + 1] - t[1]), mby);
            const V dz = Sub(Set(d, a.xyz[3 * i + 2] - t[2]), mbz);
            const V d2 = Add(Add(Mul(dx, dx), Mul(dy, dy)), Mul(dz, dz));
            s = Add(s, Mul(Exp2Grade<Poly>::apply(Sub(Zero(d), d2)), bw));
        }
    }
    if (census) {
        const uint64_t terms = uint64_t(a.n_atoms) * b.n_atoms;
        census->exp2_calls += uint64_t(a.n_atoms) * b.n_padded;
        census->pair_terms_rescore += terms;
    }
    return ReduceSum(d, s);
}

// --- the search, parameterized by grain and width --------------------------
//
// A near-copy of `score.hpp`'s `optimize_pair`, and deliberately not a
// refactor of it. The scalar path is the REFERENCE ANCHOR: its source hash
// is what every committed fixture names, and changing that file to accommodate
// a new arm would invalidate the fixture and cost a re-derivation of the thing
// every later comparison is measured against. The anchor's value is that it
// does not move. So the duplication here is bought deliberately, and the two
// are held together by a test rather than by a shared function -- the width-1
// instantiation of this template must reproduce the anchor's ranked output
// exactly, which is a stronger check than sharing code would have been, because
// it also covers the loop restructuring.
template <class Tag, bool Poly>
inline PairResult
optimize_pair_grain(Tag d, Grain grain, const Conformer& a, const Conformer& b,
                    float s_aa, float s_bb, int start_mode, int n_steps,
                    float lr_q, float lr_t_scaled, const SoaConformer* a_soa,
                    OpCensus* census) {
    PairResult best;
    best.s_aa = s_aa;
    best.s_bb = s_bb;
    best.tanimoto_shape = -1.0f;

    const float gnorm = 1.0f / (s_aa + s_bb);
    const int n_starts = start_count(start_mode);

    // The library conformer is transposed once per pair; the query's transpose
    // is hoisted to the caller because n_A is constant for the whole screen.
    SoaConformer b_soa;
    if (grain == Grain::LIBRARY_ATOM) b_soa.build(b, Lanes(d));

    for (int n = 0; n < n_starts; ++n) {
        float q_start[4];
        start_quaternion(start_mode, n, q_start);
        float r_start[9];
        quat_to_mat(q_start, r_start);
        if (!matrix_is_signed_permutation(r_start, SIGNED_PERM_TOL)) continue;

        float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float t[3] = {0.0f, 0.0f, 0.0f};
        float cache[7] = {0, 0, 0, 0, 0, 0, 0};

        float g[7], s_ab = 0.0f;
        for (int step = 0; step <= n_steps; ++step) {
            float q_conj[4] = {q_start[0], -q_start[1], -q_start[2],
                               -q_start[3]};
            float q_eff[4];
            quat_mul(q, q_conj, q_eff);
            float m[9];
            quat_to_mat(q_eff, m);
            for (int k = 0; k < 9; ++k)
                m[k] *= COORD_SCALE;

            if (step == n_steps) {
                s_ab = (grain == Grain::QUERY_ATOM)
                           ? pair_overlap_query_grain<Tag, Poly>(d, *a_soa, b,
                                                                 m, t, census)
                           : pair_overlap_library_grain<Tag, Poly>(
                                 d, a, b_soa, m, t, census);
                break;
            }
            if (grain == Grain::QUERY_ATOM) {
                get_gradient_query_grain<Tag, Poly>(d, *a_soa, b, m, t, gnorm,
                                                    g, &s_ab, census);
            } else {
                get_gradient_library_grain<Tag, Poly>(d, a, b_soa, m, t, gnorm,
                                                      g, &s_ab, census);
            }
            adagrad_exp_step(q, t, g, cache, lr_q, lr_t_scaled);
            if (census) ++census->optimizer_steps;
        }

        const float den = s_aa + s_bb - s_ab;
        const float ts = (den > 0.0f) ? (s_ab / den) : 0.0f;
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

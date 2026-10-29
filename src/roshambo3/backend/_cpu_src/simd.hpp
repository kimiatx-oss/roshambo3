// A thin, deliberately Highway-shaped SIMD abstraction, with a scalar target.
//
// WHY THIS EXISTS RATHER THAN A DEPENDENCY
// ----------------------------------------
// The scalar kernel came first; the vector width was chosen later. The plan
// named Google Highway as the portable layer, and the honest state is:
// neither Highway nor xsimd is installed on the build machines, and while PyPI
// is reachable from them, GitHub is not. Vendoring a library to write scalar
// code would be committing to it before the data to choose existed -- and it
// would make the SIMD library the eighth runtime dependency of a package that
// has seven.
//
// So the kernel is written against the interface below, whose names and shapes
// are Highway's on purpose (Vec / Mask / Load / Store / MulAdd / IfThenElse /
// ReduceSum). A vector target then has three options and this file does not prejudge
// which: implement the same interface with per-ISA intrinsics, vendor Highway
// and typedef straight through, or keep the scalar target for a grain where it
// wins. What it must NOT have to do is rewrite the kernel, which is the whole
// reason for the indirection.
//
// WHAT THE SCALAR TARGET IS FOR
// -----------------------------
// It is the reference, and it is also the control every later arm is measured
// against. So it must be *honest* scalar code: no reliance on the compiler
// happening to vectorize it, because `doc/NUMERICS_POLICY.md` makes
// reassociation a declared numerics change and an auto-vectorized reduction
// reassociates silently. `Lanes() == 1` here, and the accumulation order below
// is source order, which is what makes this target comparable to the CUDA
// kernel's Tier-3 sum term for term.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

// At file scope, not inside the namespace: an include inside `namespace r3`
// puts every declaration in <immintrin.h> and everything it pulls in -- which
// on GCC 13 includes <cstdlib> -- into r3::simd, and the errors it produces
// name `r3::simd::std::exp2` rather than the include.
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace r3 {
namespace simd {

// ---------------------------------------------------------------------------
// The scalar target.
// ---------------------------------------------------------------------------
struct ScalarTag {};

inline constexpr const char* target_name(ScalarTag) {
    return "SCALAR";
}

inline constexpr int Lanes(ScalarTag) {
    return 1;
}

struct VecF32 {
    float v;

    VecF32() : v(0.0f) {}

    explicit VecF32(float x) : v(x) {}
};

struct MaskF32 {
    bool m;

    MaskF32() : m(false) {}

    explicit MaskF32(bool b) : m(b) {}
};

inline VecF32 Zero(ScalarTag) {
    return VecF32(0.0f);
}

inline VecF32 Set(ScalarTag, float x) {
    return VecF32(x);
}

inline VecF32 Load(ScalarTag, const float* p) {
    return VecF32(*p);
}

inline void Store(VecF32 a, ScalarTag, float* p) {
    *p = a.v;
}

inline VecF32 Add(VecF32 a, VecF32 b) {
    return VecF32(a.v + b.v);
}

inline VecF32 Sub(VecF32 a, VecF32 b) {
    return VecF32(a.v - b.v);
}

inline VecF32 Mul(VecF32 a, VecF32 b) {
    return VecF32(a.v * b.v);
}

// MulAdd is `a*b + c`, spelled as one call so there is one place that decides
// whether it contracts.
//
// It is worth being accurate about why, because the obvious reading is too
// strong. `doc/NUMERICS_POLICY.md` does list `-fmad=false` among the six
// forbidden nvcc flags -- but that is a WITHIN-implementation rule, there so
// that two runs of the CUDA kernel are comparable with each other. Across
// implementations, contraction is measured at median 6.0e-8 / p99 2.7e-5 /
// 1.24% basin flips (NUMERICS_POLICY §1), which is the SAME distribution as a
// reordered reduction, i.e. the same size as the vector-width reassociation the
// CPU backend explicitly accepts. So it is recorded -- the compiler, its version
// and its flags determine it and every CPU record carries all three -- and
// judged on recall, not pinned.
//
// The call still exists because one spelling is better than two, and because
// swapping its body is how a later arm changes contraction deliberately rather
// than by accident of optimization level.
inline VecF32 MulAdd(VecF32 a, VecF32 b, VecF32 c) {
    return VecF32(std::fma(a.v, b.v, c.v));
}

inline MaskF32 Le(VecF32 a, VecF32 b) {
    return MaskF32(a.v <= b.v);
}

inline VecF32 IfThenElseZero(MaskF32 m, VecF32 a) {
    return m.m ? a : VecF32(0.0f);
}

inline float ReduceSum(ScalarTag, VecF32 a) {
    return a.v;
}

//: Are ALL lanes strictly greater than `x`? The cutoff arm's union test.
//:
//: The predicate a cutoff needs at vector width, and the one that decides
//: whether a cutoff can pay: a lane whose term is beyond the cutoff saves
//: nothing on its own, because the vector is evaluated either way. Work is
//: skipped only when every lane agrees.
inline bool AllGreater(ScalarTag, VecF32 a, float x) {
    return a.v > x;
}

// ---------------------------------------------------------------------------
// exp2 on the negative half-line, which is the only place this kernel needs it.
// ---------------------------------------------------------------------------
//
// The reduced frame makes the inner term exactly `exp2f(-|delta|^2)` with no
// multiply (numerics.py's COORD_SCALE), so the argument is always <= 0 and the
// result always in [0, 1].
//
// THE ACCURACY BAR IS NOT libm. The CUDA kernel's `exp2f` compiles to
// `ex2.approx.ftz.f32`, a ~2-ULP hardware approximation with denormals flushed
// -- so matching a correctly-rounded `exp2f` would be paying for accuracy the
// specification never had. Choosing the grade is a separate axis; this uses
// the accurate one so that a disagreement with `reference/overlap.py` is a bug
// in the kernel rather than a budget item.
//
// Kept behind a named function for exactly that reason: a grade swaps the
// body, not the call sites.
inline VecF32 Exp2Negative(VecF32 x) {
    return VecF32(std::exp2(x.v));
}

//: Below this argument `exp2` is negligible against the sum it joins, and
//: the exponent-assembly grades must flush to zero rather than wrap.
//: exp2(-24) = 6.0e-8 against a sum of order 830.
inline constexpr float kNegligibleExp2Arg = 24.0f;

// The same minimax grade at width 1, so that "which exp2" and "which width"
// stay independent axes rather than arriving together. Coefficients are
// `exp2_grades.hpp`'s `exp2_minimax` verbatim; duplicated rather than included
// so this header stays free-standing, and pinned by a test that the two agree
// bit for bit.
inline VecF32 Exp2NegativeMinimax(VecF32 x) {
    if (x.v < -kNegligibleExp2Arg) return VecF32(0.0f);
    const float k = std::nearbyint(x.v);
    const float f = x.v - k;
    float p = 1.33978436e-3f;
    p = std::fma(p, f, 9.61799621e-3f);
    p = std::fma(p, f, 5.55051871e-2f);
    p = std::fma(p, f, 2.40226507e-1f);
    p = std::fma(p, f, 6.93147182e-1f);
    p = std::fma(p, f, 1.0f);
    int32_t bits = (static_cast<int32_t>(k) + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return VecF32(p * scale);
}

// ---------------------------------------------------------------------------
// The AVX2 target. Width 8.
// ---------------------------------------------------------------------------
//
// Written as intrinsics behind the interface above rather than by vendoring
// Highway, which the scalar target left this file able to do: neither Highway
// nor xsimd is installed on the build machines and GitHub is unreachable from
// them, so vendoring would have been a dependency decision taken for the
// convenience of one target. The interface is Highway's shape, so swapping the implementation
// later is a typedef.
//
// **`ReduceSum` fixes a reduction order and that is a declared numerics
// change.** `doc/NUMERICS_POLICY.md` §2's rule is that a change to *where* a
// value lives is free and a change to *when* it is added is not, and an 8-lane
// tree reduction adds in a different order than the scalar target's source
// order. The CPU backend accepts exactly this -- in exchange for the
// ranked-output comparison against the scalar reference anchor being
// the FIRST thing a second width produces, before any throughput number from it
// is quoted. That is not a formality here; it is the reason this target exists
// before any timing code does.
#if defined(__AVX2__)
struct Avx2Tag {};

inline constexpr const char* target_name(Avx2Tag) {
    return "AVX2";
}

inline constexpr int Lanes(Avx2Tag) {
    return 8;
}

struct VecF32x8 {
    __m256 v;

    VecF32x8() : v(_mm256_setzero_ps()) {}

    explicit VecF32x8(__m256 x) : v(x) {}

    explicit VecF32x8(float x) : v(_mm256_set1_ps(x)) {}
};

struct MaskF32x8 {
    __m256 m;

    MaskF32x8() : m(_mm256_setzero_ps()) {}

    explicit MaskF32x8(__m256 x) : m(x) {}
};

inline VecF32x8 Zero(Avx2Tag) {
    return VecF32x8(_mm256_setzero_ps());
}

inline VecF32x8 Set(Avx2Tag, float x) {
    return VecF32x8(_mm256_set1_ps(x));
}

inline VecF32x8 Load(Avx2Tag, const float* p) {
    return VecF32x8(_mm256_loadu_ps(p));
}

inline void Store(VecF32x8 a, Avx2Tag, float* p) {
    _mm256_storeu_ps(p, a.v);
}

inline VecF32x8 Add(VecF32x8 a, VecF32x8 b) {
    return VecF32x8(_mm256_add_ps(a.v, b.v));
}

inline VecF32x8 Sub(VecF32x8 a, VecF32x8 b) {
    return VecF32x8(_mm256_sub_ps(a.v, b.v));
}

inline VecF32x8 Mul(VecF32x8 a, VecF32x8 b) {
    return VecF32x8(_mm256_mul_ps(a.v, b.v));
}

inline VecF32x8 MulAdd(VecF32x8 a, VecF32x8 b, VecF32x8 c) {
    return VecF32x8(_mm256_fmadd_ps(a.v, b.v, c.v));
}

inline MaskF32x8 Le(VecF32x8 a, VecF32x8 b) {
    return MaskF32x8(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ));
}

inline VecF32x8 IfThenElseZero(MaskF32x8 m, VecF32x8 a) {
    return VecF32x8(_mm256_and_ps(m.m, a.v));
}

// A fixed tree reduction: (0+4)(1+5)(2+6)(3+7), then pairwise. The ORDER is
// pinned here rather than left to the compiler, so that two runs of this target
// agree with each other bitwise even though neither agrees with the scalar
// target -- which is the within-implementation half of NUMERICS_POLICY that
// still applies.
inline float ReduceSum(Avx2Tag, VecF32x8 a) {
    __m128 lo = _mm256_castps256_ps128(a.v);
    __m128 hi = _mm256_extractf128_ps(a.v, 1);
    __m128 s = _mm_add_ps(lo, hi);                  // 0+4 1+5 2+6 3+7
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));         // +2 +3
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));  // +1
    return _mm_cvtss_f32(s);
}

// exp2 on the negative half-line, LANEWISE THROUGH libm, deliberately.
//
// This target's first job is the reduction-order gate: show that a second vector width
// does not move the ranked output. That question is about the REDUCTION ORDER,
// and answering it requires changing nothing else -- so this calls the same
// `std::exp2` the anchor calls, once per lane, and is slow on purpose.
//
// A vector polynomial is the obvious next arm, `minimax_d5` the recommended
// grade (3.79e-6 max relative error, inside the ~2e-3 budget, no gather). It is
// deliberately NOT folded in here: a transcendental change is a LARGER numerics
// change than a reassociation, and it lands outside the reference anchor's
// scope entirely. Bundling the two would make the width gate unreadable --
// exactly the "contaminate every arm measured after it" problem of fast-math.
inline bool AllGreater(Avx2Tag, VecF32x8 a, float x) {
    return _mm256_movemask_ps(
               _mm256_cmp_ps(a.v, _mm256_set1_ps(x), _CMP_GT_OQ)) == 0xFF;
}

inline VecF32x8 Exp2Negative(VecF32x8 x) {
    alignas(32) float b[8];
    _mm256_store_ps(b, x.v);
    for (int i = 0; i < 8; ++i)
        b[i] = std::exp2(b[i]);
    return VecF32x8(_mm256_load_ps(b));
}

//: Below this argument `exp2` is negligible against the sum it joins, and the
//: exponent-assembly grades must flush to zero rather than wrap.

// The vector polynomial, the `minimax_d5` grade, lifted to 8 lanes.//
// FLUSH TO ZERO BELOW THE NEGLIGIBLE THRESHOLD, AND THIS IS NOT OPTIONAL.
//
// The exponent-assembly trick is only valid while `k + 127` stays a legal
// biased exponent. `exp2_grades.hpp`'s scalar grades carry the comment "k is in
// [-24, 0] here, so no overflow path" -- true of a caller that applies the
// `NEGLIGIBLE_D2 = 24` cutoff, and NOT true of these grain kernels, which do
// not cut off at all. A distant atom pair gives d2 far below -24, `k + 127`
// goes negative, the shift wraps, and the result is garbage rather than the
// underflow-to-zero libm would give.
//
// It was garbage: the first version of this shipped without the mask and
// the reduction-order gate caught it immediately -- recall 0.8137, RBO 0.8188, top-1 6/8
// and max |dT| **1.863**, a Tanimoto difference larger than the range of a
// Tanimoto. That is what a gate run before the throughput number is for; the
// arm was 6.9x the anchor and would have looked like the headline result.
//
// The threshold is restated rather than re-derived, because the cutoff and
// the grade are the SAME knob and choosing them independently double-counts
// the accuracy cost: exp2(-24) = 6.0e-8 against a
// sum of order 830.
//
// This grade's measured cost: 3.79e-6 max relative error -- three orders
// inside the ~2e-3 accuracy budget and in
// `ex2.approx`'s own accuracy class, which is the bar the GPU has always been
// held to rather than libm's correctly-rounded one -- with no memory and no
// gather, which the table grades cannot say.
//
// It is a SEPARATE FUNCTION and a separate arm, not a replacement for the one
// above, because the two answer different questions. A change of
// transcendental implementation is LARGER than a reassociation, and it falls
// outside the reference anchor's scope entirely. So the width gate runs on libm
// and this runs after it, for the reason fast-math is excluded: an unbounded
// numerics change folded in early contaminates every arm measured after it.
inline VecF32x8 Exp2NegativeMinimax(VecF32x8 x) {
    // exp2(x) = 2^k * exp2(f), k = round(x), f = x - k in [-0.5, 0.5].
    const __m256 k =
        _mm256_round_ps(x.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(x.v, k);
    __m256 p = _mm256_set1_ps(1.33978436e-3f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.61799621e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.55051871e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.40226507e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.93147182e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    // 2^k by exponent assembly. On AVX-512 this is `vscalefps`, one
    // instruction; on AVX2 it is the integer shift below, which is the
    // concrete form of the biggest AVX-512-ism for this
    // kernel. k is in [-24, 0] here (the argument is bounded by the reduced
    // frame and the cutoff), so there is no overflow path to handle.
    const __m256i ki = _mm256_cvtps_epi32(k);
    const __m256i biased =
        _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    const __m256 r = _mm256_mul_ps(p, _mm256_castsi256_ps(biased));
    // Masked, not branched, and a branch here would be per-lane
    // anyway. `x >= -24` keeps the lane, everything below it goes to zero.
    const __m256 keep =
        _mm256_cmp_ps(x.v, _mm256_set1_ps(-kNegligibleExp2Arg), _CMP_GE_OQ);
    return VecF32x8(_mm256_and_ps(keep, r));
}
#endif  // __AVX2__

// ---------------------------------------------------------------------------
// The AVX-512 target. Width 16.
// ---------------------------------------------------------------------------
//
// Behind `#pragma GCC target`, NOT behind a global `-march`, and that is the
// load-bearing choice. The translation unit stays at `-march=x86-64-v3`, so
// `score.hpp`'s scalar path -- the REFERENCE ANCHOR, whose source hash every
// committed fixture names -- is compiled exactly as it was and produces exactly
// the numbers the fixture holds. Raising the whole TU to AVX-512 would change
// the anchor's own codegen, which is how an anchor stops being one.
//
// It also means ONE binary carries both widths and picks at run time via
// `__builtin_cpu_supports`, which is the arrangement `probe.cpp` already proved
// works on both machines, and which lets `selected_target()` be a measurement
// of the process rather than a restatement of the build.
// Gated on the REAL macro: this block compiles only in a translation unit that
// actually carries `-mavx512f`, which is `grain_avx512.cpp` and nothing else.
// The pragma that used to stand here compiled everywhere and worked nowhere.
#if defined(__AVX512F__)
#define R3_HAS_AVX512_TARGET 1

struct Avx512Tag {};

inline constexpr const char* target_name(Avx512Tag) {
    return "AVX512F";
}

inline constexpr int Lanes(Avx512Tag) {
    return 16;
}

struct VecF32x16 {
    __m512 v;

    VecF32x16() : v(_mm512_setzero_ps()) {}

    explicit VecF32x16(__m512 x) : v(x) {}

    explicit VecF32x16(float x) : v(_mm512_set1_ps(x)) {}
};

inline VecF32x16 Zero(Avx512Tag) {
    return VecF32x16(_mm512_setzero_ps());
}

inline VecF32x16 Set(Avx512Tag, float x) {
    return VecF32x16(_mm512_set1_ps(x));
}

inline VecF32x16 Load(Avx512Tag, const float* p) {
    return VecF32x16(_mm512_loadu_ps(p));
}

inline void Store(VecF32x16 a, Avx512Tag, float* p) {
    _mm512_storeu_ps(p, a.v);
}

inline VecF32x16 Add(VecF32x16 a, VecF32x16 b) {
    return VecF32x16(_mm512_add_ps(a.v, b.v));
}

inline VecF32x16 Sub(VecF32x16 a, VecF32x16 b) {
    return VecF32x16(_mm512_sub_ps(a.v, b.v));
}

inline VecF32x16 Mul(VecF32x16 a, VecF32x16 b) {
    return VecF32x16(_mm512_mul_ps(a.v, b.v));
}

inline VecF32x16 MulAdd(VecF32x16 a, VecF32x16 b, VecF32x16 c) {
    return VecF32x16(_mm512_fmadd_ps(a.v, b.v, c.v));
}

// `_mm512_reduce_add_ps` is a GCC sequence rather than one instruction, and
// its order is the compiler's. Spelled out here for the same reason the AVX2
// reduction is: two runs of THIS target must agree with each other bitwise
// even though neither agrees with the scalar target.
inline float ReduceSum(Avx512Tag, VecF32x16 a) {
    __m256 lo = _mm512_castps512_ps256(a.v);
    __m256 hi =
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a.v), 1));
    __m256 s = _mm256_add_ps(lo, hi);
    __m128 s4 =
        _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 0x55));
    return _mm_cvtss_f32(s4);
}

inline bool AllGreater(Avx512Tag, VecF32x16 a, float x) {
    return _mm512_cmp_ps_mask(a.v, _mm512_set1_ps(x), _CMP_GT_OQ) == 0xFFFF;
}

inline VecF32x16 Exp2Negative(VecF32x16 x) {
    alignas(64) float b[16];
    _mm512_store_ps(b, x.v);
    for (int i = 0; i < 16; ++i)
        b[i] = std::exp2(b[i]);
    return VecF32x16(_mm512_load_ps(b));
}

// The minimax grade at width 16, and the one place this target is not merely
// "AVX2 but wider".
//
// `_mm512_scalef_ps` assembles 2^k in ONE instruction, the single biggest
// AVX-512-ism for this kernel -- the AVX2 path emulates it
// with a convert, an integer add and a shift, three operations plus a domain
// hazard, and this is the arm that measures whether that matters. The
// flush-to-zero mask is still required and is still a mask, not a branch:
// `scalef` handles the exponent range but the polynomial is only fitted on
// [-0.5, 0.5] and the terms below -24 are meant to vanish.
inline VecF32x16 Exp2NegativeMinimax(VecF32x16 x) {
    const __m512 k = _mm512_roundscale_ps(x.v, _MM_FROUND_TO_NEAREST_INT);
    const __m512 f = _mm512_sub_ps(x.v, k);
    __m512 p = _mm512_set1_ps(1.33978436e-3f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.61799621e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.55051871e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.40226507e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.93147182e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    // The isolation knob. `vscalefps` assembles 2^k in ONE instruction and is
    // the single biggest AVX-512-ism for this kernel; the AVX2
    // path emulates it with a convert, an integer add and a shift. Defining
    // R3_NO_VSCALEF builds the AVX-512 kernel with that same emulation, so the
    // difference between the two builds is the instruction and nothing else.
    // Without an arm like this, `vscalefps`'s contribution is bundled into the
    // width's and neither can be quoted.
#if defined(R3_NO_VSCALEF)
    const __m512i ki = _mm512_cvtps_epi32(k);
    const __m512i biased =
        _mm512_slli_epi32(_mm512_add_epi32(ki, _mm512_set1_epi32(127)), 23);
    const __m512 r = _mm512_mul_ps(p, _mm512_castsi512_ps(biased));
#else
    const __m512 r = _mm512_scalef_ps(p, k);
#endif
    const __mmask16 keep = _mm512_cmp_ps_mask(
        x.v, _mm512_set1_ps(-kNegligibleExp2Arg), _CMP_GE_OQ);
    return VecF32x16(_mm512_maskz_mov_ps(keep, r));
}

#endif  // __AVX512F__

}  // namespace simd
}  // namespace r3

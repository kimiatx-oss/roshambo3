// The paired AVX2 field sampler: two samples per 256-bit vector.
//
// The geometry here is NOT the analytic path's grain and the difference is
// structural rather than a tuning choice.
//
//   analytic       : lane = one PAIR. Eight independent pairs ride the vector,
//                    the seven accumulators never mix across lanes, and the
//                    score is bit-identical at every width.
//   field    (here): lane = one CHANNEL. One `half4` is four floats = 128 bits,
//                    so a vector holds TWO SAMPLES of the same pair, and the
//                    interpolation tree narrows with the reduction rather than
//                    staying eight-wide.
//
// That is the shape the uploaded references argue for, and it falls out of the
// data: a stencil is four contiguous x-PAIR loads, not eight scattered reads,
// because x is contiguous in the node array. Each x-pair is exactly one YMM in
// fp32 (8 floats) and exactly one XMM widened to a YMM in fp16 (8 halves).
//
// **The accumulators fall out of the channel layout for free.** A sample is
// `[f, gx, gy, gz]`, and the search needs
//
//     vs   += f                      -- lane 0
//     g[4..6] += gfac * (gx, gy, gz)  -- lanes 1..3
//
// so value and translation gradient are ONE vector add. The torque
// `g[1..3] = C x grad` is a cross product, and with the same `[_, x, y, z]`
// lane order it is two lane-local shuffles and an FMA -- and `vshufps` operates
// WITHIN each 128-bit half, so one instruction does both samples.
#pragma once

#if defined(__AVX2__)

#include <immintrin.h>
#include <cfloat>
#include <type_traits>

#include "field.hpp"

namespace r3 {
namespace cpu {

//: `a + t (b - a)`, the lerp the whole tree is built from.
static inline __m256 lerp256(__m256 a, __m256 b, __m256 t) {
    return _mm256_fmadd_ps(t, _mm256_sub_ps(b, a), a);
}

//: `[low128(a) | low128(b)]` and `[high128(a) | high128(b)]`. VEX-encoded, so
//: this kernel is AVX2-clean; the EVEX-256 variant that would reach ymm16-31
//: needs AVX-512 and is deferred with the rest of the AVX-512 work.
static inline __m256 join_lo(__m256 a, __m256 b) {
    return _mm256_permute2f128_ps(a, b, 0x20);
}

static inline __m256 join_hi(__m256 a, __m256 b) {
    return _mm256_permute2f128_ps(a, b, 0x31);
}

//: Load one x-pair -- both x corners, all four channels -- as one YMM.
//: fp32 is a straight 32-byte load; fp16 is a 16-byte load plus one
//: `vcvtph2ps`. Same instruction count in the tree either way, half the bytes
//: off the cache for fp16.
static inline __m256 load_pair(const float* p) {
    return _mm256_loadu_ps(p);
}

static inline __m256 load_pair(const uint16_t* p) {
    return _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

}  // namespace cpu
}  // namespace r3

#include "field_xlast_avx2.hpp"

namespace r3 {
namespace cpu {

// Private ablations. Interpolation and safety policy are independent.
#ifndef R3_FIELD_AVX2_XLAST
#define R3_FIELD_AVX2_XLAST 0
#endif
#ifndef R3_FIELD_AVX2_STRICT_BOUNDS
#define R3_FIELD_AVX2_STRICT_BOUNDS 1
#endif
#ifndef R3_FIELD_AVX2_ALTERNATING
#define R3_FIELD_AVX2_ALTERNATING 1
#endif
#ifndef R3_FIELD_AVX2_PACKED_HALF
#define R3_FIELD_AVX2_PACKED_HALF 1
#endif
#ifndef R3_FIELD_AVX2_SCHEDULE
#define R3_FIELD_AVX2_SCHEDULE 0
#endif

//: One sample's four x-pair pointers. Four pointers, not eight: x is
//: contiguous, so each addresses both of its x corners.
template <class T>
struct PairStencilT {
    const T* p00;  // [T000 | T100]
    const T* p10;  // [T010 | T110]
    const T* p01;  // [T001 | T101]
    const T* p11;  // [T011 | T111]
};

using PairStencil = PairStencilT<float>;

//: Whether the sample is inside the lattice, and its stencil if so. Outside,
//: the field is exactly zero -- its true asymptote, which is what the GPU gets
//: free from `cudaAddressModeBorder` and this path pays a test for.
template <class Field, class T>
static inline bool stencil_of(const Field& f, float x, float y, float z,
                              PairStencilT<T>& s, float& fx, float& fy,
                              float& fz) {
    const FieldGeometry& g = f.g;
    const float tx = (x - g.ox) * g.inv_hs;
    const float ty = (y - g.oy) * g.inv_hs;
    const float tz = (z - g.oz) * g.inv_hs;
    const int ix = int(tx), iy = int(ty), iz = int(tz);
    if (!(tx >= 0.f && ty >= 0.f && tz >= 0.f) || ix >= g.nx || iy >= g.ny ||
        iz >= g.nz)
        return false;
    fx = tx - float(ix);
    fy = ty - float(iy);
    fz = tz - float(iz);
    // ONE address, three strides. The four corners of a stencil are at fixed
    // offsets from the first -- +one node row in y, +one node plane in z --
    // so computing four independent `node(ix, iy+dy, iz+dz)` addresses repeats
    // the same two multiplies four times. With the vector part measured at
    // roughly a third of the per-sample cost, the scalar address arithmetic is
    // not a rounding error here.
    const size_t row = size_t(f.g.nodes_x()) * 4;
    const size_t plane = row * size_t(f.g.nodes_y());
    s.p00 = f.node(ix, iy, iz);
    s.p10 = s.p00 + row;
    s.p01 = s.p00 + plane;
    s.p11 = s.p01 + row;
    return true;
}

//: The x stage for one sample: four pair loads, two full-width lerps.
//: `q00`/`q10` are consumed into `xz0` before `q01`/`q11` are loaded, so the
//: live set stays small -- the references are explicit that minimizing live
//: ranges beats relying on the renamer to forward spills.
template <class T>
static inline void x_stage(const PairStencilT<T>& s, __m256 vfx, __m256& xz0,
                           __m256& xz1) {
    const __m256 q00 = load_pair(s.p00);
    const __m256 q10 = load_pair(s.p10);
    xz0 = lerp256(join_lo(q00, q10), join_hi(q00, q10), vfx);
    const __m256 q01 = load_pair(s.p01);
    const __m256 q11 = load_pair(s.p11);
    xz1 = lerp256(join_lo(q01, q11), join_hi(q01, q11), vfx);
}

//: Broadcast `a` into the low half and `b` into the high half.
static inline __m256 pack2_broadcast(float a, float b) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(a)),
                                _mm_set1_ps(b), 1);
}

//: Finish two samples together. Six full-width lerps and one more, against the
//: eight (six full plus two half-width) two independent samples would need --
//: and the final z lerp is full width instead of two XMM tails, which is the
//: point of pairing.
static inline __m256 finish_pair(__m256 a_z0, __m256 a_z1, __m256 b_z0,
                                 __m256 b_z1, __m256 vfy, __m256 vfz) {
    const __m256 y0 = lerp256(join_lo(a_z0, b_z0), join_hi(a_z0, b_z0), vfy);
    const __m256 y1 = lerp256(join_lo(a_z1, b_z1), join_hi(a_z1, b_z1), vfy);
    return lerp256(y0, y1, vfz);  // [A.f A.gx A.gy A.gz | B....]
}

//: `C x grad`, in lanes 1..3, for BOTH samples at once. `vshufps` is lane-local
//: so one instruction shuffles each 128-bit half independently, which is
//: exactly the two-sample layout.
static inline __m256 cross3(__m256 c, __m256 v) {
    const __m256 c_yzx = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
    const __m256 c_zxy = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(2, 1, 3, 0));
    const __m256 v_yzx = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 3, 2, 0));
    const __m256 v_zxy = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 1, 3, 0));
    return _mm256_fmsub_ps(c_yzx, v_zxy, _mm256_mul_ps(c_zxy, v_yzx));
}

//: The field gradient over one library conformer, two atoms at a time.
//: Numerically this is NOT the scalar path term for term -- pairing changes
//: nothing about the per-sample arithmetic, but the accumulation order over
//: atoms is the same, so the two agree to fp32 rounding rather than bitwise.
//: A width-dependent CPU score is accepted provided the width is in the
//: record; the comparison is on ranked output.
//: One prepared sample: where its stencil is and where inside the cell it
//: falls. Separating PREPARATION from the interpolation is what lets several
//: samples' loads be in flight at once -- the address arithmetic for sample
//: k+1 issues while sample k's four cache lines are still arriving, which is
//: the only thing that helps a memory-bound kernel.
template <class T>
struct PreparedSampleT {
    PairStencilT<T> s;
    float fx, fy, fz;
    float c1, c2, c3;  // the lever arm C = 2 M b
};

static const float kZeroNode[8] = {0, 0, 0, 0, 0, 0, 0, 0};
static const uint16_t kZeroNode16[8] = {0, 0, 0, 0, 0, 0, 0, 0};

static inline const float* zero_node(const float*) {
    return kZeroNode;
}

static inline const uint16_t* zero_node(const uint16_t*) {
    return kZeroNode16;
}

template <class Field, class T>
static inline void prepare_sample(const Field& f, const Conformer& b,
                                  const float m[9], const float t[3], int j,
                                  PreparedSampleT<T>& out) {
    const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1], bz = b.xyz[3 * j + 2];
    const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
    const float mby = m[3] * bx + m[4] * by + m[5] * bz;
    const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
    out.c1 = 2.f * mbx;
    out.c2 = 2.f * mby;
    out.c3 = 2.f * mbz;
    if (!stencil_of(f, mbx + t[0], mby + t[1], mbz + t[2], out.s, out.fx,
                    out.fy, out.fz)) {
        // Outside the lattice the field is exactly zero. Zeroing the STENCIL
        // rather than branching keeps every sample on one path.
        out.s.p00 = out.s.p10 = out.s.p01 = out.s.p11 = zero_node(out.s.p00);
        out.fx = out.fy = out.fz = 0.f;
    }
}

//: Transform, map and bounds-test EIGHT atoms at once.
//:
//: The ablation says why this exists: of a sample's cost, the scalar prepare is
//: **63%** and the interpolation **1%**. Every
//: step of the prepare -- the 3x3 pose transform, the lattice mapping, the
//: bounds test, the fractional part -- is elementwise over atoms and was being
//: done one atom at a time, while the ANALYTIC kernel has always done its own
//: `M*b` eight pairs at a time in lanes. This closes that asymmetry.
//:
//: The library record is AoS (`xyz[3*j]`), so the three coordinate streams come
//: in by gather. A gather is not cheap, but it replaces eight scalar loads plus
//: the arithmetic that followed them.
//: The library conformer in SoA, built ONCE per pair and reused by every step
//: and start. The AoS record forces a gather per coordinate stream, and on
//: Zen 3 a gather is microcoded: the first blocked prepare used three
//: `vpgatherdd` per eight atoms and came out 0.86-0.89x, SLOWER than the
//: scalar prepare it replaced. This is the analytic `LaneBlock` argument in a
//: different grain -- transpose once, then every pass reads contiguous.
//:
//: Cost is one pass over n_B floats per pair, amortized over 4 starts x 25
//: steps = 100 passes.
struct ConformerSoA {
    alignas(32) float x[kConformerSoACapacity], y[kConformerSoACapacity],
        z[kConformerSoACapacity];
    int n = 0, padded = 0;
    //: max |b_i| over the conformer's own frame. A rotation preserves it, so
    //: one number bounds every atom's displacement at every pose and every
    //: step -- which is what makes the all-in-bounds test a per-pose scalar
    //: check rather than per-atom vector work.
    float radius = 0.f;

    void build(const Conformer& b) {
        // Callers refuse larger conformers (interface.cpp,
        // shape_field_evaluator.cpp); clamping `n` as well keeps every reader,
        // which loops to `n`, inside the arrays even if one did not.
        n = b.n_atoms < kConformerSoACapacity ? b.n_atoms
                                              : kConformerSoACapacity;
        padded = (n + 7) & ~7;
        for (int j = 0; j < n; ++j) {
            x[j] = b.xyz[3 * j];
            y[j] = b.xyz[3 * j + 1];
            z[j] = b.xyz[3 * j + 2];
        }
        // Padding lanes are zeroed rather than left stale: they are masked out
        // downstream, but a NaN or a huge stale value would still be
        // transformed and bounds-tested, and the rule is that padding is a
        // value that costs nothing, never a branch.
        for (int j = n; j < padded; ++j) {
            x[j] = y[j] = z[j] = 0.0f;
        }
        float r2 = 0.f;
        for (int j = 0; j < n; ++j) {
            const float d = x[j] * x[j] + y[j] * y[j] + z[j] * z[j];
            if (!std::isfinite(d)) {
                r2 = INFINITY;
                break;
            }
            if (d > r2) r2 = d;
        }
        // Upward allowance covers three rounded products, two additions and
        // sqrt. Nonfinite atoms reject the proof instead of disappearing from
        // the maximum through unordered comparisons.
        radius = std::nextafter(std::sqrt(r2) * (1.f + 16.f * FLT_EPSILON),
                                INFINITY);
    }
};

//: Does every atom of this conformer, at this pose, land strictly inside the
//: lattice -- so that the per-atom bounds machinery cannot fire?
//:
//: `m` is a rotation times COORD_SCALE, so its rows all have the same norm;
//: that norm is computed rather than assumed, because assuming it would make
//: the test silently wrong for any future `m` that is not a scaled rotation.
//: Componentwise |(m b)_k| <= ||m_row_k|| * |b| <= ||m_row_k|| * radius.
//:
//: CONSERVATIVE by construction: a false negative costs the general path, a
//: false positive would read outside the lattice. The margin below is there so
//: that float rounding in the kernel's own map cannot turn a boundary case
//: into the second kind.
template <bool GRIDXF = false, class Field>
static inline bool conformer_inside(const Field& f, const ConformerSoA& b,
                                    const float m[9], const float t[3]) {
    const FieldGeometry& g = f.g;
    const float o[3] = {g.ox, g.oy, g.oz};
    const int nc[3] = {g.nx, g.ny, g.nz};
    if (!std::isfinite(b.radius) || !std::isfinite(g.inv_hs) ||
        !(g.inv_hs > 0.f))
        return false;
    // Squared norm terms can underflow even though their unsquared dot
    // products are significant after multiplication by a large coefficient.
    // This absolute allowance also covers flush-to-zero of subnormal squares;
    // it rounds away for ordinary coordinates/poses and rejects extreme proofs.
    const float tiny_norm = 8.f * std::sqrt(FLT_MIN);
    for (int k = 0; k < 3; ++k) {
        const float scale = GRIDXF ? g.inv_hs : 1.f;
        const float r0 = m[3 * k] * scale, r1 = m[3 * k + 1] * scale,
                    r2 = m[3 * k + 2] * scale;
        const float rad = std::nextafter(
            (std::sqrt(r0 * r0 + r1 * r1 + r2 * r2) + tiny_norm) *
                (b.radius + tiny_norm) * (1.f + 16.f * FLT_EPSILON),
            INFINITY);
        // FP32 row-norm/radius upper bounds plus an allowance for the actual
        // nested-FMA dot, add/subtract and grid multiply. 64 epsilon exceeds
        // their gamma bounds and covers subnormal underflow via the +1 floor.
        // GRIDXF uses the rounded folded coefficients and translation that its
        // prepare path actually consumes, rather than the unfused pose proof.
        const float center = (t[k] - o[k]) * g.inv_hs;
        const float rg = GRIDXF ? rad : rad * g.inv_hs;
        const float magnitude =
            GRIDXF ? std::fabs(center) + rg + 1.f
                   : (std::fabs(t[k]) + std::fabs(o[k]) + rad + 1.f) * g.inv_hs;
        const float margin =
            64.f * FLT_EPSILON * (magnitude + float(nc[k]) + 1.f);
        if (!std::isfinite(rad) || !std::isfinite(center) ||
            !std::isfinite(margin) || nc[k] <= 0 || !(center - rg > margin) ||
            !(center + rg < float(nc[k]) - margin))
            return false;
    }
    return true;
}

struct AtomBlock {
    alignas(32) int ix[8], iy[8], iz[8];
    alignas(32) float fx[8], fy[8], fz[8];
    alignas(32) float c1[8], c2[8], c3[8];
    alignas(32) int ok[8];
};

template <class Field>
static inline void prepare_eight(const Field& f, const ConformerSoA& b,
                                 const float m[9], const float t[3], int j0,
                                 int n, AtomBlock& out) {
    const FieldGeometry& g = f.g;
    // Contiguous, not gathered. This is the whole point of the SoA record.
    const __m256 bx = _mm256_load_ps(b.x + j0);
    const __m256 by = _mm256_load_ps(b.y + j0);
    const __m256 bz = _mm256_load_ps(b.z + j0);

    // M * b, eight atoms at a time.
    const __m256 mbx = _mm256_fmadd_ps(
        _mm256_set1_ps(m[0]), bx,
        _mm256_fmadd_ps(_mm256_set1_ps(m[1]), by,
                        _mm256_mul_ps(_mm256_set1_ps(m[2]), bz)));
    const __m256 mby = _mm256_fmadd_ps(
        _mm256_set1_ps(m[3]), bx,
        _mm256_fmadd_ps(_mm256_set1_ps(m[4]), by,
                        _mm256_mul_ps(_mm256_set1_ps(m[5]), bz)));
    const __m256 mbz = _mm256_fmadd_ps(
        _mm256_set1_ps(m[6]), bx,
        _mm256_fmadd_ps(_mm256_set1_ps(m[7]), by,
                        _mm256_mul_ps(_mm256_set1_ps(m[8]), bz)));

    const __m256 two = _mm256_set1_ps(2.0f);
    _mm256_store_ps(out.c1, _mm256_mul_ps(two, mbx));
    _mm256_store_ps(out.c2, _mm256_mul_ps(two, mby));
    _mm256_store_ps(out.c3, _mm256_mul_ps(two, mbz));

    const __m256 inv = _mm256_set1_ps(g.inv_hs);
    const __m256 tx =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(mbx, _mm256_set1_ps(t[0])),
                                    _mm256_set1_ps(g.ox)),
                      inv);
    const __m256 ty =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(mby, _mm256_set1_ps(t[1])),
                                    _mm256_set1_ps(g.oy)),
                      inv);
    const __m256 tz =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(mbz, _mm256_set1_ps(t[2])),
                                    _mm256_set1_ps(g.oz)),
                      inv);

    // Truncation toward zero matches the scalar `int(tx)`; the negative side is
    // caught by the bounds test below, so the two agree everywhere it matters.
    const __m256i vix = _mm256_cvttps_epi32(tx);
    const __m256i viy = _mm256_cvttps_epi32(ty);
    const __m256i viz = _mm256_cvttps_epi32(tz);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out.ix), vix);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out.iy), viy);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out.iz), viz);

    _mm256_store_ps(out.fx, _mm256_sub_ps(tx, _mm256_cvtepi32_ps(vix)));
    _mm256_store_ps(out.fy, _mm256_sub_ps(ty, _mm256_cvtepi32_ps(viy)));
    _mm256_store_ps(out.fz, _mm256_sub_ps(tz, _mm256_cvtepi32_ps(viz)));

    // Inside the lattice on all three axes, as one mask.
    const __m256 zero = _mm256_setzero_ps();
    __m256 in =
        _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(tx, zero, _CMP_GE_OQ),
                                    _mm256_cmp_ps(ty, zero, _CMP_GE_OQ)),
                      _mm256_cmp_ps(tz, zero, _CMP_GE_OQ));
    in = _mm256_and_ps(in, _mm256_castsi256_ps(_mm256_cmpgt_epi32(
                               _mm256_set1_epi32(g.nx), vix)));
    in = _mm256_and_ps(in, _mm256_castsi256_ps(_mm256_cmpgt_epi32(
                               _mm256_set1_epi32(g.ny), viy)));
    in = _mm256_and_ps(in, _mm256_castsi256_ps(_mm256_cmpgt_epi32(
                               _mm256_set1_epi32(g.nz), viz)));
    // Lanes past the real atom count are not atoms and must not contribute.
    alignas(32) int lanemask[8];
    for (int k = 0; k < 8; ++k)
        lanemask[k] = (k < n) ? -1 : 0;
    in = _mm256_and_ps(in, _mm256_castsi256_ps(_mm256_load_si256(
                               reinterpret_cast<__m256i*>(lanemask))));
    _mm256_store_si256(reinterpret_cast<__m256i*>(out.ok),
                       _mm256_castps_si256(in));
}

//: `UNROLL` PAIRS of samples in flight, so 2*UNROLL atoms per iteration.
//:
//: Why this matters more here than it did for the analytic kernel: that one is
//: arithmetic-bound, so its vector units were the scarce thing. This one is
//: MEMORY-bound -- full vectorization bought 1.03-1.21x -- and the way to help
//: a memory-bound loop is to have several independent stencils in flight, not
//: to make each one wider. Each pair in flight holds 4 live YMM after its x
//: stage, against 16 architectural on AVX2, so UNROLL = 2 is the most that
//: fits comfortably and 3 is where the compiler starts spilling. Measured, not
//: assumed: the references warn against treating the renamer as extra
//: architectural registers.
template <int UNROLL, class Field>
inline void field_gradient_avx2_unrolled(const Field& f, const Conformer& b,
                                         const float m[9], const float t[3],
                                         float gnorm, float g[7], float* out_s,
                                         OpCensus* census) {
    using Node = std::remove_pointer_t<decltype(f.node(0, 0, 0))>;
    const float gfac = -gnorm;
    __m256 acc = _mm256_setzero_ps();
    __m256 tor = _mm256_setzero_ps();
    alignas(32) float cbuf[8];

    PreparedSampleT<Node> ps[2 * UNROLL];
    int j = 0;
    const int step = 2 * UNROLL;
    for (; j + step <= b.n_atoms; j += step) {
        // All the address arithmetic first, so every load below is
        // independent of every other and the machine can keep them in flight.
        for (int u = 0; u < step; ++u)
            prepare_sample(f, b, m, t, j + u, ps[u]);

        __m256 z0[2 * UNROLL], z1[2 * UNROLL];
        for (int u = 0; u < step; ++u)
            x_stage(ps[u].s, pack2_broadcast(ps[u].fx, ps[u].fx), z0[u], z1[u]);

        for (int u = 0; u < UNROLL; ++u) {
            const PreparedSampleT<Node>& A = ps[2 * u];
            const PreparedSampleT<Node>& B = ps[2 * u + 1];
            const __m256 v = finish_pair(
                z0[2 * u], z1[2 * u], z0[2 * u + 1], z1[2 * u + 1],
                pack2_broadcast(A.fy, B.fy), pack2_broadcast(A.fz, B.fz));
            cbuf[0] = 0.f;
            cbuf[1] = A.c1;
            cbuf[2] = A.c2;
            cbuf[3] = A.c3;
            cbuf[4] = 0.f;
            cbuf[5] = B.c1;
            cbuf[6] = B.c2;
            cbuf[7] = B.c3;
            const __m256 c = _mm256_load_ps(cbuf);
            acc = _mm256_add_ps(acc, v);
            tor = _mm256_add_ps(tor, cross3(c, v));
        }
        if (census) census->field_samples += uint64_t(step);
    }

    alignas(32) float ab[8], tb[8];
    _mm256_store_ps(ab, acc);
    _mm256_store_ps(tb, tor);
    float vs = ab[0] + ab[4];
    float G[3] = {ab[1] + ab[5], ab[2] + ab[6], ab[3] + ab[7]};
    float T[3] = {tb[1] + tb[5], tb[2] + tb[6], tb[3] + tb[7]};

    for (; j < b.n_atoms; ++j) {
        const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1],
                    bz = b.xyz[3 * j + 2];
        const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
        const float mby = m[3] * bx + m[4] * by + m[5] * bz;
        const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
        // The odd atom, through the same paired path with its partner masked
        // to zero -- rather than a second scalar sampler that would have to be
        // kept in step with this one.
        PreparedSampleT<Node> one[2];
        prepare_sample(f, b, m, t, j, one[0]);
        one[1] = one[0];
        one[1].s.p00 = one[1].s.p10 = one[1].s.p01 = one[1].s.p11 =
            zero_node(one[1].s.p00);
        one[1].fx = one[1].fy = one[1].fz = 0.f;
        one[1].c1 = one[1].c2 = one[1].c3 = 0.f;
        __m256 az0, az1, bz0, bz1;
        x_stage(one[0].s, pack2_broadcast(one[0].fx, one[0].fx), az0, az1);
        x_stage(one[1].s, pack2_broadcast(one[1].fx, one[1].fx), bz0, bz1);
        const __m256 v = finish_pair(az0, az1, bz0, bz1,
                                     pack2_broadcast(one[0].fy, one[1].fy),
                                     pack2_broadcast(one[0].fz, one[1].fz));
        alignas(32) float cb2[8] = {0.f, one[0].c1, one[0].c2, one[0].c3,
                                    0.f, 0.f,       0.f,       0.f};
        const __m256 c2v = _mm256_load_ps(cb2);
        alignas(32) float vb[8], tb2[8];
        _mm256_store_ps(vb, v);
        _mm256_store_ps(tb2, cross3(c2v, v));
        vs += vb[0];
        G[0] += vb[1];
        G[1] += vb[2];
        G[2] += vb[3];
        T[0] += tb2[1];
        T[1] += tb2[2];
        T[2] += tb2[3];
        if (census) ++census->field_samples;
    }

    g[0] = 0.0f;
    g[1] = gfac * T[0];
    g[2] = gfac * T[1];
    g[3] = gfac * T[2];
    g[4] = gfac * G[0];
    g[5] = gfac * G[1];
    g[6] = gfac * G[2];
    if (out_s) *out_s = vs;
}

//: The search kernel with the prepare vectorized eight atoms at a time.
//:
//: Same arithmetic as the scalar-prepare version and the same accumulation
//: order over atoms, so it agrees with it to fp32 rounding rather than
//: bitwise -- an accepted difference, and the gate is on ranked output.
// --- the register-resident search kernel -----------------------------------
//
// Three defects in the blocked kernel below, all the same one wearing different
// clothes: vector state spilled to the stack and read back scalar. Measured at
// **33 loads per sample when only four are the field stencil**.
//
//   1. `AtomBlock` stores ten 8-wide arrays that the next loop reloads one
//      element at a time.
//   2. `cbuf` does eight SCALAR stores followed by one 32-byte aligned load per
//      pair -- a store-forwarding hazard by construction.
//   3. The per-atom `if (blk.ok[k])` branch, to handle a case the
//      access-pattern arm measured as never occurring: ZERO clamped samples at
//      every spacing with the shipped 8 A pad.
//
// This version keeps the prepare's output in registers. Per-pair weights come
// out of the 8-wide vectors by `vpermps` rather than by scalar reload; the
// torque accumulates in SoA, where `c1/c2/c3` already live, so no [_,x,y,z]
// vector ever has to be built; and out-of-bounds samples are given an offset
// into the field's zeroed tail instead of a branch.
//
// **The bounds test is kept, not removed.** It never fires on this panel, but
// the guarantee does not hold in general: a library molecule of radius r_B
// reaches r_B + travel from the origin while the field covers r_A + pad, so a
// large library conformer against a small query can leave the lattice. What is
// removed is the BRANCH, not the check.
struct PreparedBlock {
    __m256 fx, fy, fz;            // fractional position in the cell
    __m256 c1, c2, c3;            // lever arm: 2 M b, or M b / hs under GRIDXF
    alignas(32) uint32_t off[8];  // linear node offset; zero-tail out of bounds
};

// The grid-space pose and the torque compensation are one operation. A caller
// can no longer fold M/hs while silently keeping the world-space torque scale.
// This is per pose, not per atom/block; both specializations inline to the
// original arithmetic and the GRIDXF=false arm has no added work.
template <bool GRIDXF>
struct FieldPose {
    float agrid[9], dgrid[3];
    const float* m;
    const float* t;
    float torque_scale;

    FieldPose(const FieldGeometry& g, const float world_m[9],
              const float world_t[3])
        : m(world_m), t(world_t), torque_scale(1.0f) {
        if constexpr (GRIDXF) {
            const float origin[3] = {g.ox, g.oy, g.oz};
            for (int k = 0; k < 9; ++k)
                agrid[k] = world_m[k] * g.inv_hs;
            for (int k = 0; k < 3; ++k)
                dgrid[k] = (world_t[k] - origin[k]) * g.inv_hs;
            m = agrid;
            t = dgrid;
            torque_scale = 2.0f * g.hs;
        }
    }

    void write_torque(float gfac, float x, float y, float z, float g[7]) const {
        g[1] = gfac * torque_scale * x;
        g[2] = gfac * torque_scale * y;
        g[3] = gfac * torque_scale * z;
    }
};

alignas(32) inline constexpr int32_t kLaneMask[9][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},         {-1, 0, 0, 0, 0, 0, 0, 0},
    {-1, -1, 0, 0, 0, 0, 0, 0},       {-1, -1, -1, 0, 0, 0, 0, 0},
    {-1, -1, -1, -1, 0, 0, 0, 0},     {-1, -1, -1, -1, -1, 0, 0, 0},
    {-1, -1, -1, -1, -1, -1, 0, 0},   {-1, -1, -1, -1, -1, -1, -1, 0},
    {-1, -1, -1, -1, -1, -1, -1, -1},
};

//: `FULL` says every one of the eight lanes is a real atom, which is true of
//: every block but the last. The tail mask below is a scalar
//: branch-and-store chain -- eight compares, eight stores, then a vector load
//: -- and it was running on EVERY block to serve the one that needs it. A3.
//: `STRIDE` is the scalars per lattice cell in the target layout: 4 for a
//: vertex field, 8 for B2's duplicated-edge field.
//: `ALLIN` says the caller has PROVED every atom of this conformer lands
//: inside the lattice at this pose (`conformer_inside`), so the per-atom bounds
//: machinery cannot fire and is removed: six compares, three mask ANDs, the
//: sentinel blend and the three lever-arm ANDs.
//:
//: It is removed only for FULL blocks. A tail block still has padding lanes,
//: and a padding lane's coordinates are (0,0,0) -- which under an in-bounds
//: pose maps to a REAL cell, not an out-of-bounds one, so it would contribute a
//: spurious sample rather than a zero. The tail therefore keeps the mask, the
//: blend and the lever-arm ANDs, and drops only the six bounds compares.
//: Getting this wrong is silent: the scores stay plausible and the padding
//: contributes to every tail block.
//: `GRIDXF` folds the grid spacing and the lattice origin INTO the pose, so an
//: atom maps straight to grid coordinates. The caller passes `A = m/hs` and
//: `d = (t - o)/hs` in place of `m` and `t`, and each axis becomes
//:
//:     q = A0*bx + A1*by + A2*bz + d          -- three FMAs
//:
//: against the seven operations the split form needs (three for `m b`, one for
//: the doubled lever arm, then add `t`, subtract `o`, multiply by `1/hs`).
//:
//: The lever arm comes out of the same three FMAs. The block stores `q` rather
//: than `C = 2 m b`, and the kernel recovers the torque afterwards from
//:
//:     m b = hs*q + (o - t)
//:     tau = 2 hs SUM q_i x g_i  +  2 (o - t) x SUM g_i
//:
//: which is exact in real arithmetic and a REASSOCIATION in float -- so this
//: is judged on recall and ranking, not bit-identity, unlike the rest of the
//: prepare-path work.
//:
//: It also removes the lever-arm masking. An out-of-bounds or padding lane is
//: already sent to the zeroed tail, so its sampled gradient is exactly zero and
//: `q x 0` and its contribution to `SUM g` are both zero whatever `q` holds.
//: The split form had to mask because `C = 2 m b` multiplied a nonzero lever
//: arm by that zero gradient -- also zero, but it could not skip the AND
//: without reasoning about the sentinel, which is what this comment is.
template <bool FULL, int STRIDE, bool ALLIN, bool GRIDXF,
          bool STRICT_BOUNDS = false, class Field>
static inline void prepare_block(const Field& f, const ConformerSoA& b,
                                 const float m[9], const float t[3], int j0,
                                 int n, PreparedBlock& out) {
    const FieldGeometry& g = f.g;
    const __m256 bx = _mm256_load_ps(b.x + j0);
    const __m256 by = _mm256_load_ps(b.y + j0);
    const __m256 bz = _mm256_load_ps(b.z + j0);

    __m256 tx, ty, tz;
    if constexpr (GRIDXF) {
        // `m` is A = pose/hs and `t` is d = (translation - origin)/hs.
        //
        // The lever arm is kept as `m b / hs` -- the SAME three operations that
        // produce the grid coordinate, one add short of it -- rather than
        // recovered afterwards from q. Both are algebraically exact; only this
        // one is numerically safe.
        //
        // Recovering it costs a subtraction of two large near-equal terms. At
        // the shipped 8 A pad the lattice origin sits ~9.5 A from the pose
        // translation while the torque itself is small, so
        // `2 hs SUM q x g + 2 (o - t) x SUM g` cancels about 66 against -69 to
        // leave -3.7: measured, that lost a decimal digit on the torque (9e-6
        // relative, against 5e-7 on the translation, which needs no such
        // correction). Splitting the chain one operation earlier avoids the
        // cancellation entirely and is CHEAPER than the corrected form, because
        // the epilogue then needs one scale instead of a cross product.
        auto crow = [&](int r) {
            return _mm256_fmadd_ps(
                _mm256_set1_ps(m[3 * r]), bx,
                _mm256_fmadd_ps(
                    _mm256_set1_ps(m[3 * r + 1]), by,
                    _mm256_mul_ps(_mm256_set1_ps(m[3 * r + 2]), bz)));
        };
        out.c1 = crow(0);
        out.c2 = crow(1);
        out.c3 = crow(2);  // m b / hs
        tx = _mm256_add_ps(out.c1, _mm256_set1_ps(t[0]));
        ty = _mm256_add_ps(out.c2, _mm256_set1_ps(t[1]));
        tz = _mm256_add_ps(out.c3, _mm256_set1_ps(t[2]));
    } else {
        const __m256 mbx = _mm256_fmadd_ps(
            _mm256_set1_ps(m[0]), bx,
            _mm256_fmadd_ps(_mm256_set1_ps(m[1]), by,
                            _mm256_mul_ps(_mm256_set1_ps(m[2]), bz)));
        const __m256 mby = _mm256_fmadd_ps(
            _mm256_set1_ps(m[3]), bx,
            _mm256_fmadd_ps(_mm256_set1_ps(m[4]), by,
                            _mm256_mul_ps(_mm256_set1_ps(m[5]), bz)));
        const __m256 mbz = _mm256_fmadd_ps(
            _mm256_set1_ps(m[6]), bx,
            _mm256_fmadd_ps(_mm256_set1_ps(m[7]), by,
                            _mm256_mul_ps(_mm256_set1_ps(m[8]), bz)));
        const __m256 two = _mm256_set1_ps(2.0f);
        out.c1 = _mm256_mul_ps(two, mbx);
        out.c2 = _mm256_mul_ps(two, mby);
        out.c3 = _mm256_mul_ps(two, mbz);
        const __m256 inv = _mm256_set1_ps(g.inv_hs);
        tx = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_add_ps(mbx, _mm256_set1_ps(t[0])),
                          _mm256_set1_ps(g.ox)),
            inv);
        ty = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_add_ps(mby, _mm256_set1_ps(t[1])),
                          _mm256_set1_ps(g.oy)),
            inv);
        tz = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_add_ps(mbz, _mm256_set1_ps(t[2])),
                          _mm256_set1_ps(g.oz)),
            inv);
    }

    const __m256i vix = _mm256_cvttps_epi32(tx);
    const __m256i viy = _mm256_cvttps_epi32(ty);
    const __m256i viz = _mm256_cvttps_epi32(tz);
    out.fx = _mm256_sub_ps(tx, _mm256_cvtepi32_ps(vix));
    out.fy = _mm256_sub_ps(ty, _mm256_cvtepi32_ps(viy));
    out.fz = _mm256_sub_ps(tz, _mm256_cvtepi32_ps(viz));

    __m256 in;
    if constexpr (ALLIN) {
        in = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    } else if constexpr (STRICT_BOUNDS) {
        // Ordered bounds reject NaN, infinity and conversion overflow.
        const __m256 zero = _mm256_setzero_ps();
        in = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(tx, zero, _CMP_GE_OQ),
                                         _mm256_cmp_ps(ty, zero, _CMP_GE_OQ)),
                           _mm256_cmp_ps(tz, zero, _CMP_GE_OQ));
        in = _mm256_and_ps(
            in, _mm256_and_ps(
                    _mm256_cmp_ps(tx, _mm256_set1_ps(float(g.nx)), _CMP_LT_OQ),
                    _mm256_and_ps(_mm256_cmp_ps(ty, _mm256_set1_ps(float(g.ny)),
                                                _CMP_LT_OQ),
                                  _mm256_cmp_ps(tz, _mm256_set1_ps(float(g.nz)),
                                                _CMP_LT_OQ))));
    } else {
        const __m256 zero = _mm256_setzero_ps();
        in = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(tx, zero, _CMP_GE_OQ),
                                         _mm256_cmp_ps(ty, zero, _CMP_GE_OQ)),
                           _mm256_cmp_ps(tz, zero, _CMP_GE_OQ));
        in = _mm256_and_ps(in, _mm256_castsi256_ps(_mm256_cmpgt_epi32(
                                   _mm256_set1_epi32(g.nx), vix)));
        in = _mm256_and_ps(in, _mm256_castsi256_ps(_mm256_cmpgt_epi32(
                                   _mm256_set1_epi32(g.ny), viy)));
        in = _mm256_and_ps(in, _mm256_castsi256_ps(_mm256_cmpgt_epi32(
                                   _mm256_set1_epi32(g.nz), viz)));
    }
    if constexpr (!FULL) {
        in = _mm256_and_ps(
            in, _mm256_castsi256_ps(_mm256_load_si256(
                    reinterpret_cast<const __m256i*>(kLaneMask[n]))));
    } else {
        (void)n;
    }

    if constexpr (STRICT_BOUNDS && !(ALLIN && FULL)) {
        // Invalid fractions must be finite even when sampling zero. Lever
        // masking also remains necessary for GRIDXF: infinity times zero is NaN.
        out.fx = _mm256_and_ps(out.fx, in);
        out.fy = _mm256_and_ps(out.fy, in);
        out.fz = _mm256_and_ps(out.fz, in);
        out.c1 = _mm256_and_ps(out.c1, in);
        out.c2 = _mm256_and_ps(out.c2, in);
        out.c3 = _mm256_and_ps(out.c3, in);
    }

    // The linear offset, computed 8-wide: ((iz*ny + iy)*nx + ix)*4. One int
    // array leaves the vector unit instead of three, and the scalar loop below
    // does a single add rather than two multiplies and two adds per atom.
    constexpr int kShift = STRIDE == 32 ? 5 : (STRIDE == 8 ? 3 : 2);
    const __m256i row_e = _mm256_set1_epi32(g.nodes_x() << kShift);
    const __m256i plane_e =
        _mm256_set1_epi32((g.nodes_x() * g.nodes_y()) << kShift);
    __m256i lin =
        _mm256_add_epi32(_mm256_slli_epi32(vix, kShift),
                         _mm256_add_epi32(_mm256_mullo_epi32(viy, row_e),
                                          _mm256_mullo_epi32(viz, plane_e)));
    if constexpr (ALLIN && FULL) {
        // Nothing can be out of bounds and there are no padding lanes, so the
        // sentinel blend and the lever-arm masks have nothing to do.
        _mm256_store_si256(reinterpret_cast<__m256i*>(out.off), lin);
    } else {
        // Out of bounds -> the zeroed tail. Selected, never branched.
        const __m256i tail = _mm256_set1_epi32(int(f.zero_offset()));
        lin = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(tail), _mm256_castsi256_ps(lin), in));
        _mm256_store_si256(reinterpret_cast<__m256i*>(out.off), lin);
        if constexpr (!GRIDXF && !STRICT_BOUNDS) {
            // Padding and out-of-bounds lanes contribute no lever arm. Not
            // needed under GRIDXF: see the note on the template above.
            out.c1 = _mm256_and_ps(out.c1, in);
            out.c2 = _mm256_and_ps(out.c2, in);
            out.c3 = _mm256_and_ps(out.c3, in);
        }
    }
}

//: One CORNER of two samples: `[A.node | B.node]`, four channels each.
//:
//: This is what removes the lane-crossing permutes from the tree. The old
//: shape loaded one sample's x-PAIR as a single 32-byte vector, which puts the
//: left and right corners in the two halves and then needs `vperm2f128` to
//: separate them -- and needs two more to interleave A with B before the y
//: stage. Building `[A | B]` at load time instead means every lerp is
//: elementwise and NOTHING crosses a lane after the load: the pair layout the
//: y and z stages already wanted is the layout the x stage produces.
//:
//: Costs one extra load instruction per corner (two 16-byte loads instead of
//: one 32-byte), but `vinsertf128` folds its memory operand, so the high half
//: is free. Measured in the loop body: 48 `vperm2f128` become 32
//: `vinsertf128`, and 8 `vpermps` of `lane_broadcast` disappear because the x
//: stage now wants the same packed weight the y and z stages do.
static inline __m256 corner_pair(const float* a, const float* b) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(a)),
                                _mm_load_ps(b), 1);
}

static inline __m256 corner_pair(const uint16_t* a, const uint16_t* b) {
    if constexpr (R3_FIELD_AVX2_PACKED_HALF != 0) {
        const __m128i lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a));
        const __m128i hi = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b));
        return _mm256_cvtph_ps(_mm_unpacklo_epi64(lo, hi));
    }
    const __m128 lo =
        _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a)));
    const __m128 hi =
        _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b)));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

//: B1: load each sample's whole 32-byte x-EDGE, then split left from right.
//: **MEASURED AND NOT ADOPTED** -- kept, with the reason, because the
//: structural prediction was right and the time was not.
//:
//: The field already stores the two x-adjacent nodes contiguously, so one
//: 32-byte load fetches both -- where `corner_pair` issues two 16-byte loads
//: for the same bytes. The cross-half work is unchanged (two `vperm2f128`
//: here against two `vinsertf128` there); what halves is the LOAD instruction
//: count, which is the resource B3 showed we are nearer to being bound by.
//:
//: The caveat is alignment: an x-edge starts at a 16-byte-aligned address, so
//: a 32-byte load crosses a 64-byte line boundary about half the time. Whether
//: the halved instruction count beats the split-load penalty is exactly what
//: this measures.
static inline void edge_pair(const float* a, const float* b, __m256& left,
                             __m256& right) {
    const __m256 ea = _mm256_loadu_ps(a);  // [A.x0 | A.x1]
    const __m256 eb = _mm256_loadu_ps(b);  // [B.x0 | B.x1]
    left = _mm256_permute2f128_ps(ea, eb, 0x20);
    right = _mm256_permute2f128_ps(ea, eb, 0x31);
}

static inline void edge_pair(const uint16_t* a, const uint16_t* b, __m256& left,
                             __m256& right) {
    // fp16: the whole edge is 16 bytes, so one load and one widening each.
    const __m256 ea =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
    const __m256 eb =
        _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
    left = _mm256_permute2f128_ps(ea, eb, 0x20);
    right = _mm256_permute2f128_ps(ea, eb, 0x31);
}

//: The trilinear tree on x-edge loads. Same arithmetic as `interp_pair`, same
//: results bit for bit; only how the corners reach the registers differs.
template <class T>
static inline __m256 interp_pair_edge(const T* a00, const T* a10, const T* a01,
                                      const T* a11, const T* b00, const T* b10,
                                      const T* b01, const T* b11, __m256 wx,
                                      __m256 wy, __m256 wz) {
    __m256 l, r;
    edge_pair(a00, b00, l, r);
    const __m256 x00 = lerp256(l, r, wx);
    edge_pair(a10, b10, l, r);
    const __m256 x10 = lerp256(l, r, wx);
    edge_pair(a01, b01, l, r);
    const __m256 x01 = lerp256(l, r, wx);
    edge_pair(a11, b11, l, r);
    const __m256 x11 = lerp256(l, r, wx);
    return lerp256(lerp256(x00, x10, wy), lerp256(x01, x11, wy), wz);
}

//: The whole trilinear tree for two samples, with no lane crossing. Seven
//: lerps, exactly as the separable form prescribes, and the result is already
//: `[A.rgba | B.rgba]`.
template <class T>
static inline __m256 interp_pair(const T* a00, const T* a10, const T* a01,
                                 const T* a11, const T* b00, const T* b10,
                                 const T* b01, const T* b11, __m256 wx,
                                 __m256 wy, __m256 wz) {
    const __m256 x00 =
        lerp256(corner_pair(a00, b00), corner_pair(a00 + 4, b00 + 4), wx);
    const __m256 x10 =
        lerp256(corner_pair(a10, b10), corner_pair(a10 + 4, b10 + 4), wx);
    const __m256 x01 =
        lerp256(corner_pair(a01, b01), corner_pair(a01 + 4, b01 + 4), wx);
    const __m256 x11 =
        lerp256(corner_pair(a11, b11), corner_pair(a11 + 4, b11 + 4), wx);
    return lerp256(lerp256(x00, x10, wy), lerp256(x01, x11, wy), wz);
}

//: Broadcast lane K of the LOW half into the low half, and lane K+4 of the
//: high half into the high half -- an IMMEDIATE, lane-local shuffle.
//:
//: This is what pairing point k with point k+4 buys, and it is the whole
//: reason for that schedule. Under the old adjacent pairing (0,1),(2,3),...
//: a pair's two points sat in the same 128-bit half of the natural SoA
//: fraction vector, so broadcasting each into its own half needed a VARIABLE
//: `vpermps` with an index vector. Pairing across the halves makes every
//: weight a `vpermilps` with an immediate: no index register, cheaper port,
//: and no live vector holding the index.
//:
//: `K` must be a template parameter -- a runtime K would force the variable
//: form back.
template <int K>
static inline __m256 pair_splat(__m256 x) {
    static_assert(K >= 0 && K < 4, "lane within a 128-bit half");
    return _mm256_permute_ps(x, _MM_SHUFFLE(K, K, K, K));
}

// B1: two independent XYZ trees, consuming each row/face early.
// Prepared offsets already select the existing zero sentinel for invalid or
// padded lanes. Keep all corner loads, weights and per-tree lerps unchanged.
template <int K0, int K1>
#if R3_FIELD_AVX2_SCHEDULE == 2
__attribute__((always_inline))
#endif
static inline void
interp_two_pairs_xyz(const float* base, const PreparedBlock& pb, size_t row,
                     size_t plane, __m256& out0, __m256& out1) {
    const float* a0 = base + pb.off[K0];
    const float* b0 = base + pb.off[K0 + 4];
    const float* a1 = base + pb.off[K1];
    const float* b1 = base + pb.off[K1 + 4];
    const __m256 wx0 = pair_splat<K0>(pb.fx);
    const __m256 wx1 = pair_splat<K1>(pb.fx);
    auto x_rows = [&](size_t offset, __m256& x0, __m256& x1) {
#if R3_FIELD_AVX2_SCHEDULE == 2
        // B2's sole adjustment: stage-group the two rows' endpoint loads.
        // The B1 linked DSO emitted two calls and caller state saves per block;
        // keep this instruction schedule within the existing block boundary.
        const __m256 left0 = corner_pair(a0 + offset, b0 + offset);
        const __m256 right0 = corner_pair(a0 + offset + 4, b0 + offset + 4);
        const __m256 left1 = corner_pair(a1 + offset, b1 + offset);
        const __m256 right1 = corner_pair(a1 + offset + 4, b1 + offset + 4);
        x0 = lerp256(left0, right0, wx0);
        x1 = lerp256(left1, right1, wx1);
#else
        const __m256 left0 = corner_pair(a0 + offset, b0 + offset);
        const __m256 left1 = corner_pair(a1 + offset, b1 + offset);
        x0 = lerp256(left0, corner_pair(a0 + offset + 4, b0 + offset + 4), wx0);
        x1 = lerp256(left1, corner_pair(a1 + offset + 4, b1 + offset + 4), wx1);
#endif
    };
    __m256 x00, x01, x10, x11;
    x_rows(0, x00, x01);
    x_rows(row, x10, x11);
    const __m256 y00 = lerp256(x00, x10, pair_splat<K0>(pb.fy));
    const __m256 y01 = lerp256(x01, x11, pair_splat<K1>(pb.fy));
    x_rows(plane, x00, x01);
    x_rows(plane + row, x10, x11);
    const __m256 y10 = lerp256(x00, x10, pair_splat<K0>(pb.fy));
    const __m256 y11 = lerp256(x01, x11, pair_splat<K1>(pb.fy));
    out0 = lerp256(y00, y10, pair_splat<K0>(pb.fz));
    out1 = lerp256(y01, y11, pair_splat<K1>(pb.fz));
}

//: The trilinear tree on an x-affine field. Identical structure to
//: `interp_pair`, with each x stage a single FMA instead of a subtract and an
//: FMA -- 4 fewer operations per sample pair, 16 per block.
static inline __m256 interp_pair_affine(const float* ab, const float* ad,
                                        const float* bb_, const float* bd,
                                        size_t row, size_t plane, __m256 wx,
                                        __m256 wy, __m256 wz) {
    const __m256 x00 =
        _mm256_fmadd_ps(corner_pair(ad, bd), wx, corner_pair(ab, bb_));
    const __m256 x10 = _mm256_fmadd_ps(corner_pair(ad + row, bd + row), wx,
                                       corner_pair(ab + row, bb_ + row));
    const __m256 x01 = _mm256_fmadd_ps(corner_pair(ad + plane, bd + plane), wx,
                                       corner_pair(ab + plane, bb_ + plane));
    const __m256 x11 =
        _mm256_fmadd_ps(corner_pair(ad + plane + row, bd + plane + row), wx,
                        corner_pair(ab + plane + row, bb_ + plane + row));
    return lerp256(lerp256(x00, x10, wy), lerp256(x01, x11, wy), wz);
}

//: Family C's evaluator: the trilinear polynomial for two samples in SEVEN
//: FMAs, against the lerp tree's seven subtracts and seven FMAs.
//:
//:   f = (c000 + u c100) + v (c010 + u c110)
//:     + w [ (c001 + u c101) + v (c011 + u c111) ]
static inline __m256 interp_pair_coeff(const float* a, const float* b,
                                       __m256 wu, __m256 wv, __m256 ww) {
    auto cp = [&](int k) {
        return corner_pair(a + 4 * k, b + 4 * k);
    };
    const __m256 hi = _mm256_fmadd_ps(wv, _mm256_fmadd_ps(wu, cp(7), cp(6)),
                                      _mm256_fmadd_ps(wu, cp(5), cp(3)));
    const __m256 lo = _mm256_fmadd_ps(wv, _mm256_fmadd_ps(wu, cp(4), cp(2)),
                                      _mm256_fmadd_ps(wu, cp(1), cp(0)));
    return _mm256_fmadd_ps(ww, hi, lo);
}

//: The same polynomial evaluated FLAT rather than by Horner nesting.
//:
//: Horner is three FMAs deep. That is already shorter than the lerp tree's six
//: dependent operations, but it serializes: each level waits on the one below.
//: This form pays three extra multiplies to build the uv/uw/vw/uvw products and
//: then has FOUR independent FMAs at its widest point, trading depth for width.
//:
//: Which is better depends on whether the kernel is short of independent work.
//: With four sample pairs already in flight per block it may not be, and then
//: the extra multiplies are pure cost -- against a machine whose binding
//: resource is FP register file capacity, more in-flight FP ops is the wrong
//: direction. Measured rather than argued.
static inline __m256 interp_pair_coeff_flat(const float* a, const float* b,
                                            __m256 wu, __m256 wv, __m256 ww) {
    auto cp = [&](int k) {
        return corner_pair(a + 4 * k, b + 4 * k);
    };
    const __m256 uv = _mm256_mul_ps(wu, wv);
    const __m256 uw = _mm256_mul_ps(wu, ww);
    const __m256 vw = _mm256_mul_ps(wv, ww);
    const __m256 uvw = _mm256_mul_ps(uv, ww);
    const __m256 t0 = _mm256_fmadd_ps(wu, cp(1), cp(0));
    const __m256 t1 = _mm256_fmadd_ps(uv, cp(4), _mm256_mul_ps(wv, cp(2)));
    const __m256 t2 = _mm256_fmadd_ps(uw, cp(5), _mm256_mul_ps(ww, cp(3)));
    const __m256 t3 = _mm256_fmadd_ps(uvw, cp(7), _mm256_mul_ps(vw, cp(6)));
    return _mm256_add_ps(_mm256_add_ps(t0, t1), _mm256_add_ps(t2, t3));
}

//: Lane K of an 8-wide vector, broadcast to all eight lanes. The x stage
//: needs this: its vector holds ONE sample's two x corners across all four
//: channels, so every lane wants the SAME fraction.
template <int K>
static inline __m256 lane_broadcast(__m256 v) {
    return _mm256_permutevar8x32_ps(v, _mm256_set1_epi32(K));
}

//: Lanes 2u and 2u+1 in the two halves. The y and z stages need this: their
//: vector holds sample A low and B high, so the halves want DIFFERENT
//: fractions. One `vpermps` where the previous kernel did two scalar loads,
//: two broadcasts and an insert.
template <int U>
static inline __m256 pair_weight(__m256 v) {
    const __m256i idx = _mm256_setr_epi32(2 * U, 2 * U, 2 * U, 2 * U, 2 * U + 1,
                                          2 * U + 1, 2 * U + 1, 2 * U + 1);
    return _mm256_permutevar8x32_ps(v, idx);
}

//: Transpose four pair-results, each `[P_k | P_{k+4}]`, into SoA: eight
//: samples' f, gx, gy, gz.
//:
//: With the (k, k+4) pair schedule this lands in NATURAL point order --
//: [P0 P1 P2 P3 | P4 P5 P6 P7] -- because each 128-bit half of the transposed
//: result gathers one component from four consecutive points. Under the old
//: adjacent pairing it came out as (0,2,4,6,1,3,5,7) and the lever arms had to
//: be permuted to match, which cost three more variable `vpermps`. Those are
//: now unnecessary: the prepared coordinates are already in natural order.
static inline void to_soa(__m256 v0, __m256 v1, __m256 v2, __m256 v3, __m256& f,
                          __m256& gx, __m256& gy, __m256& gz) {
    const __m256 t0 = _mm256_unpacklo_ps(v0, v1);
    const __m256 t1 = _mm256_unpackhi_ps(v0, v1);
    const __m256 t2 = _mm256_unpacklo_ps(v2, v3);
    const __m256 t3 = _mm256_unpackhi_ps(v2, v3);
    f = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    gx = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    gy = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    gz = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
}

//: The permutation `to_soa` leaves in the sample order, applied to the SoA
//: lever arms so they line up with it.
static inline __m256 match_soa_order(__m256 v) {
    const __m256i idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    return _mm256_permutevar8x32_ps(v, idx);
}

//: The inverse of `to_soa`: four natural-order 8-lane vectors into four
//: pair-AoS vectors, `[P_k.a P_k.b P_k.c P_k.d | P_{k+4}....]`. Lane-local
//: throughout, so nothing crosses a half.
static inline void to_pair_aos(__m256 a, __m256 b, __m256 c, __m256 d,
                               __m256& p0, __m256& p1, __m256& p2, __m256& p3) {
    const __m256 t0 = _mm256_unpacklo_ps(a, b);  // a0 b0 a1 b1 | a4 b4 a5 b5
    const __m256 t1 = _mm256_unpackhi_ps(a, b);  // a2 b2 a3 b3 | ...
    const __m256 t2 = _mm256_unpacklo_ps(c, d);
    const __m256 t3 = _mm256_unpackhi_ps(c, d);
    p0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    p1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    p2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    p3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
}

//: The search kernel with PAIR-AoS accumulation: two live accumulators, not
//: seven. **MEASURED AND NOT SHIPPED** -- kept because the reason it loses is
//: worth having.
//:
//: It is 1.60-1.73x against the scalar-prepare baseline where the SoA form is
//: 1.67-1.85x, on the same fixture. The premise was right when it was formed:
//: the pre-A1 kernel spilled 92 vector moves to the stack and seven long-lived
//: 8-lane accumulators were the obvious cause. But adopting the (k, k+4) pair
//: schedule freed enough live state on its own that BOTH shapes now compile
//: with zero spills -- so the problem this was aimed at no longer exists, and
//: what remains is its cost: the torque becomes four pairwise `cross3` calls,
//: 24 shuffle-and-arithmetic operations per block, where the SoA form does the
//: whole cross product in six.
//:
//: Keep it in mind if a future change re-introduces register pressure; the
//: trade would flip back.
//:
//: The SoA form below keeps seven 8-lane accumulators -- value, three gradient
//: components, three torque components -- and AVX2 has sixteen YMM registers
//: to hold those plus the interpolation tree, three coordinates, three
//: weights, offsets and four pair results. It does not fit: the assembly shows
//: **83 stack-touching vector moves**, and each accumulator becomes a
//: store-to-load recurrence on every block, which costs more than its
//: instruction count suggests.
//:
//: Here the sampled records are never transposed. `[f gx gy gz | ...]` is
//: accumulated as it comes out of the tree, and the torque is formed pairwise
//: against lever arms transposed ONCE per block into the same layout. Two
//: accumulators, two final reductions, and the cross product's operand is
//: exactly the `[_ x y z]` shape `cross3` already wants.
template <bool PIPE, bool ALLIN = false, bool GRIDXF = false, class Field>
inline void field_gradient_avx2_pairaos(const Field& f, const ConformerSoA& b,
                                        const float m[9], const float t[3],
                                        float gnorm, float g[7], float* out_s,
                                        OpCensus* census) {
    const FieldPose<GRIDXF> pose(f.g, m, t);
    const float* pm = pose.m;
    const float* pt = pose.t;
    using Node = std::remove_pointer_t<decltype(f.node(0, 0, 0))>;
    const Node* base = f.v.data();
    const size_t row = size_t(f.g.nodes_x()) * 4;
    const size_t plane = row * size_t(f.g.nodes_y());

    __m256 acc = _mm256_setzero_ps();  // [f gx gy gz | f gx gy gz]
    __m256 tor = _mm256_setzero_ps();  // [_ t1 t2 t3 | _ t1 t2 t3]
    PreparedBlock pb;

    // A3: the full blocks and the tail are separate loops, so the tail's mask
    // never runs on a full block. `n_full` is a multiple of eight.
    const int n_full = b.n & ~7;
    PreparedBlock pb_next;
    if constexpr (PIPE) {
        if (b.n > 0) {
            const int n0 = (b.n < 8) ? b.n : 8;
            if (n_full > 0)
                prepare_block<true, 4, ALLIN, GRIDXF>(f, b, pm, pt, 0, 8, pb);
            else
                prepare_block<false, 4, ALLIN, GRIDXF>(f, b, pm, pt, 0, n0, pb);
        }
    }
    for (int j0 = 0; j0 <= b.n; j0 += 8) {
        if (j0 >= b.n) break;
        const int n = (b.n - j0 < 8) ? (b.n - j0) : 8;
        if constexpr (PIPE) {
            const int jn = j0 + 8;
            if (jn < b.n) {
                const int nn = (b.n - jn < 8) ? (b.n - jn) : 8;
                if (jn < n_full)
                    prepare_block<true, 4, ALLIN, GRIDXF>(f, b, pm, pt, jn, 8,
                                                          pb_next);
                else
                    prepare_block<false, 4, ALLIN, GRIDXF>(f, b, pm, pt, jn, nn,
                                                           pb_next);
            }
        } else {
            if (j0 < n_full)
                prepare_block<true, 4, ALLIN, GRIDXF>(f, b, pm, pt, j0, 8, pb);
            else
                prepare_block<false, 4, ALLIN, GRIDXF>(f, b, pm, pt, j0, n, pb);
        }

        // The lever arms, transposed once per block into pair-AoS. The leading
        // zero channel lines up with the value channel, which the cross
        // product ignores.
        __m256 C0, C1, C2, C3;
        to_pair_aos(_mm256_setzero_ps(), pb.c1, pb.c2, pb.c3, C0, C1, C2, C3);

        auto do_pair = [&](int lo, int hi, __m256 wx, __m256 wy, __m256 wz) {
            const Node* a0 = base + pb.off[lo];
            const Node* b0 = base + pb.off[hi];
            return interp_pair(a0, a0 + row, a0 + plane, a0 + plane + row, b0,
                               b0 + row, b0 + plane, b0 + plane + row, wx, wy,
                               wz);
        };
#define R3_ACC(K, LO, HI, CV)                                                  \
    {                                                                          \
        const __m256 vv = do_pair(LO, HI, pair_splat<K>(pb.fx),                \
                                  pair_splat<K>(pb.fy), pair_splat<K>(pb.fz)); \
        acc = _mm256_add_ps(acc, vv);                                          \
        tor = _mm256_add_ps(tor, cross3(CV, vv));                              \
    }
        R3_ACC(0, 0, 4, C0)
        R3_ACC(1, 1, 5, C1)
        R3_ACC(2, 2, 6, C2)
        R3_ACC(3, 3, 7, C3)
#undef R3_ACC
        if (census) census->field_samples += uint64_t(n);
        if constexpr (PIPE) pb = pb_next;
    }

    // Fold the halves, then read the four channels. Two reductions, not seven.
    alignas(32) float a8[8], t8[8];
    _mm256_store_ps(a8, acc);
    _mm256_store_ps(t8, tor);
    const float gfac = -gnorm;
    g[0] = 0.0f;
    pose.write_torque(gfac, t8[1] + t8[5], t8[2] + t8[6], t8[3] + t8[7], g);
    g[4] = gfac * (a8[1] + a8[5]);
    g[5] = gfac * (a8[2] + a8[6]);
    g[6] = gfac * (a8[3] + a8[7]);
    if (out_s) *out_s = a8[0] + a8[4];
}

//: The search kernel on an x-affine field. Everything except the tree is
//: shared with `field_gradient_avx2_reg`.
template <bool PIPE, bool ALLIN = false, bool GRIDXF = false>
inline void
field_gradient_avx2_affine(const FieldAffine32& f, const ConformerSoA& b,
                           const float m[9], const float t[3], float gnorm,
                           float g[7], float* out_s, OpCensus* census) {
    const FieldPose<GRIDXF> pose(f.g, m, t);
    const float* pm = pose.m;
    const float* pt = pose.t;
    const float* bs = f.base.data();
    const float* dv = f.dxv.data();
    const size_t row = size_t(f.g.nodes_x()) * 4;
    const size_t plane = row * size_t(f.g.nodes_y());

    __m256 vs = _mm256_setzero_ps();
    __m256 Gx = _mm256_setzero_ps(), Gy = _mm256_setzero_ps(),
           Gz = _mm256_setzero_ps();
    __m256 T1 = _mm256_setzero_ps(), T2 = _mm256_setzero_ps(),
           T3 = _mm256_setzero_ps();
    PreparedBlock pb;

    const int n_full = b.n & ~7;
    PreparedBlock pb_next;
    if constexpr (PIPE) {
        if (b.n > 0) {
            const int n0 = (b.n < 8) ? b.n : 8;
            if (n_full > 0)
                prepare_block<true, 4, ALLIN, GRIDXF>(f, b, pm, pt, 0, 8, pb);
            else
                prepare_block<false, 4, ALLIN, GRIDXF>(f, b, pm, pt, 0, n0, pb);
        }
    }
    for (int j0 = 0; j0 < b.n; j0 += 8) {
        const int n = (b.n - j0 < 8) ? (b.n - j0) : 8;
        if constexpr (PIPE) {
            const int jn = j0 + 8;
            if (jn < b.n) {
                const int nn = (b.n - jn < 8) ? (b.n - jn) : 8;
                if (jn < n_full)
                    prepare_block<true, 4, ALLIN, GRIDXF>(f, b, pm, pt, jn, 8,
                                                          pb_next);
                else
                    prepare_block<false, 4, ALLIN, GRIDXF>(f, b, pm, pt, jn, nn,
                                                           pb_next);
            }
        } else {
            if (j0 < n_full)
                prepare_block<true, 4, ALLIN, GRIDXF>(f, b, pm, pt, j0, 8, pb);
            else
                prepare_block<false, 4, ALLIN, GRIDXF>(f, b, pm, pt, j0, n, pb);
        }

        __m256 v[4];
        auto do_pair = [&](int lo, int hi, __m256 wx, __m256 wy, __m256 wz) {
            return interp_pair_affine(bs + pb.off[lo], dv + pb.off[lo],
                                      bs + pb.off[hi], dv + pb.off[hi], row,
                                      plane, wx, wy, wz);
        };
        v[0] = do_pair(0, 4, pair_splat<0>(pb.fx), pair_splat<0>(pb.fy),
                       pair_splat<0>(pb.fz));
        v[1] = do_pair(1, 5, pair_splat<1>(pb.fx), pair_splat<1>(pb.fy),
                       pair_splat<1>(pb.fz));
        v[2] = do_pair(2, 6, pair_splat<2>(pb.fx), pair_splat<2>(pb.fy),
                       pair_splat<2>(pb.fz));
        v[3] = do_pair(3, 7, pair_splat<3>(pb.fx), pair_splat<3>(pb.fy),
                       pair_splat<3>(pb.fz));

        __m256 sf, sgx, sgy, sgz;
        to_soa(v[0], v[1], v[2], v[3], sf, sgx, sgy, sgz);
        vs = _mm256_add_ps(vs, sf);
        Gx = _mm256_add_ps(Gx, sgx);
        Gy = _mm256_add_ps(Gy, sgy);
        Gz = _mm256_add_ps(Gz, sgz);
        T1 = _mm256_add_ps(
            T1, _mm256_fmsub_ps(pb.c2, sgz, _mm256_mul_ps(pb.c3, sgy)));
        T2 = _mm256_add_ps(
            T2, _mm256_fmsub_ps(pb.c3, sgx, _mm256_mul_ps(pb.c1, sgz)));
        T3 = _mm256_add_ps(
            T3, _mm256_fmsub_ps(pb.c1, sgy, _mm256_mul_ps(pb.c2, sgx)));
        if (census) census->field_samples += uint64_t(n);
        if constexpr (PIPE) pb = pb_next;
    }

    auto hsum = [](__m256 x) {
        __m128 lo = _mm256_castps256_ps128(x);
        lo = _mm_add_ps(lo, _mm256_extractf128_ps(x, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        return _mm_cvtss_f32(lo);
    };
    const float gfac = -gnorm;
    g[0] = 0.0f;
    pose.write_torque(gfac, hsum(T1), hsum(T2), hsum(T3), g);
    g[4] = gfac * hsum(Gx);
    g[5] = gfac * hsum(Gy);
    g[6] = gfac * hsum(Gz);
    if (out_s) *out_s = hsum(vs);
}

template <bool PIPE, bool ALLIN_DISPATCH = false, bool GRIDXF = false>
inline PairResult optimize_pair_field_affine(
    const Conformer& a, const Conformer& b, const FieldAffine32& f, float s_aa,
    float s_bb, int start_mode, int n_steps, float lr_q, float lr_t_scaled,
    OpCensus* census, bool winner_only = false, bool want_field_score = false) {
    ConformerSoA soa;
    soa.build(b);
    return optimize_pair_field_with(
        a, b, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled, census,
        [&soa](const FieldAffine32& ff, const Conformer&, const float mm[9],
               const float tt[3], float gn, float gg[7], float* os,
               OpCensus* cc) {
            if constexpr (ALLIN_DISPATCH) {
                // Per POSE, not per atom, exactly as the shipped arm
                // does it: six scalar comparisons over ~30 atoms, and
                // a false negative costs only the general path.
                if (cc) ++cc->allin_pose_visits;
                if (conformer_inside<GRIDXF>(ff, soa, mm, tt)) {
                    if (cc) ++cc->allin_poses;
                    field_gradient_avx2_affine<PIPE, true, GRIDXF>(
                        ff, soa, mm, tt, gn, gg, os, cc);
                    return;
                }
            }
            field_gradient_avx2_affine<PIPE, false, GRIDXF>(ff, soa, mm, tt, gn,
                                                            gg, os, cc);
        },
        winner_only, want_field_score);
}

//: Family C: the search kernel on the coefficient field.
template <bool PIPE, bool FLAT = false, bool ALLIN = false, bool GRIDXF = false>
inline void field_gradient_avx2_coeff(const FieldCoeff32& f,
                                      const ConformerSoA& b, const float m[9],
                                      const float t[3], float gnorm, float g[7],
                                      float* out_s, OpCensus* census) {
    const FieldPose<GRIDXF> pose(f.g, m, t);
    const float* pm = pose.m;
    const float* pt = pose.t;
    const float* base = f.v.data();

    __m256 vs = _mm256_setzero_ps();
    __m256 Gx = _mm256_setzero_ps(), Gy = _mm256_setzero_ps(),
           Gz = _mm256_setzero_ps();
    __m256 T1 = _mm256_setzero_ps(), T2 = _mm256_setzero_ps(),
           T3 = _mm256_setzero_ps();
    PreparedBlock pb, pb_next;

    const int n_full = b.n & ~7;
    if constexpr (PIPE) {
        if (b.n > 0) {
            const int n0 = (b.n < 8) ? b.n : 8;
            if (n_full > 0)
                prepare_block<true, 32, ALLIN, GRIDXF>(f, b, pm, pt, 0, 8, pb);
            else
                prepare_block<false, 32, ALLIN, GRIDXF>(f, b, pm, pt, 0, n0,
                                                        pb);
        }
    }
    for (int j0 = 0; j0 < b.n; j0 += 8) {
        const int n = (b.n - j0 < 8) ? (b.n - j0) : 8;
        if constexpr (PIPE) {
            const int jn = j0 + 8;
            if (jn < b.n) {
                const int nn = (b.n - jn < 8) ? (b.n - jn) : 8;
                if (jn < n_full)
                    prepare_block<true, 32, ALLIN, GRIDXF>(f, b, pm, pt, jn, 8,
                                                           pb_next);
                else
                    prepare_block<false, 32, ALLIN, GRIDXF>(f, b, pm, pt, jn,
                                                            nn, pb_next);
            }
        } else {
            if (j0 < n_full)
                prepare_block<true, 32, ALLIN, GRIDXF>(f, b, pm, pt, j0, 8, pb);
            else
                prepare_block<false, 32, ALLIN, GRIDXF>(f, b, pm, pt, j0, n,
                                                        pb);
        }

        __m256 v[4];
        auto do_pair = [&](int lo, int hi, __m256 wu, __m256 wv, __m256 ww) {
            if constexpr (FLAT)
                return interp_pair_coeff_flat(base + pb.off[lo],
                                              base + pb.off[hi], wu, wv, ww);
            else
                return interp_pair_coeff(base + pb.off[lo], base + pb.off[hi],
                                         wu, wv, ww);
        };
        v[0] = do_pair(0, 4, pair_splat<0>(pb.fx), pair_splat<0>(pb.fy),
                       pair_splat<0>(pb.fz));
        v[1] = do_pair(1, 5, pair_splat<1>(pb.fx), pair_splat<1>(pb.fy),
                       pair_splat<1>(pb.fz));
        v[2] = do_pair(2, 6, pair_splat<2>(pb.fx), pair_splat<2>(pb.fy),
                       pair_splat<2>(pb.fz));
        v[3] = do_pair(3, 7, pair_splat<3>(pb.fx), pair_splat<3>(pb.fy),
                       pair_splat<3>(pb.fz));

        __m256 sf, sgx, sgy, sgz;
        to_soa(v[0], v[1], v[2], v[3], sf, sgx, sgy, sgz);
        vs = _mm256_add_ps(vs, sf);
        Gx = _mm256_add_ps(Gx, sgx);
        Gy = _mm256_add_ps(Gy, sgy);
        Gz = _mm256_add_ps(Gz, sgz);
        T1 = _mm256_add_ps(
            T1, _mm256_fmsub_ps(pb.c2, sgz, _mm256_mul_ps(pb.c3, sgy)));
        T2 = _mm256_add_ps(
            T2, _mm256_fmsub_ps(pb.c3, sgx, _mm256_mul_ps(pb.c1, sgz)));
        T3 = _mm256_add_ps(
            T3, _mm256_fmsub_ps(pb.c1, sgy, _mm256_mul_ps(pb.c2, sgx)));
        if (census) census->field_samples += uint64_t(n);
        if constexpr (PIPE) pb = pb_next;
    }

    auto hsum = [](__m256 x) {
        __m128 lo = _mm256_castps256_ps128(x);
        lo = _mm_add_ps(lo, _mm256_extractf128_ps(x, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        return _mm_cvtss_f32(lo);
    };
    const float gfac = -gnorm;
    g[0] = 0.0f;
    pose.write_torque(gfac, hsum(T1), hsum(T2), hsum(T3), g);
    g[4] = gfac * hsum(Gx);
    g[5] = gfac * hsum(Gy);
    g[6] = gfac * hsum(Gz);
    if (out_s) *out_s = hsum(vs);
}

template <bool PIPE, bool FLAT = false, bool ALLIN_DISPATCH = false,
          bool GRIDXF = false>
inline PairResult optimize_pair_field_coeff(
    const Conformer& a, const Conformer& b, const FieldCoeff32& f, float s_aa,
    float s_bb, int start_mode, int n_steps, float lr_q, float lr_t_scaled,
    OpCensus* census, bool winner_only = false, bool want_field_score = false) {
    ConformerSoA soa;
    soa.build(b);
    return optimize_pair_field_with(
        a, b, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled, census,
        [&soa](const FieldCoeff32& ff, const Conformer&, const float mm[9],
               const float tt[3], float gn, float gg[7], float* os,
               OpCensus* cc) {
            if constexpr (ALLIN_DISPATCH) {
                // Per POSE, not per atom, exactly as the shipped arm
                // does it: six scalar comparisons over ~30 atoms, and
                // a false negative costs only the general path.
                if (cc) ++cc->allin_pose_visits;
                if (conformer_inside<GRIDXF>(ff, soa, mm, tt)) {
                    if (cc) ++cc->allin_poses;
                    field_gradient_avx2_coeff<PIPE, FLAT, true, GRIDXF>(
                        ff, soa, mm, tt, gn, gg, os, cc);
                    return;
                }
            }
            field_gradient_avx2_coeff<PIPE, FLAT, false, GRIDXF>(
                ff, soa, mm, tt, gn, gg, os, cc);
        },
        winner_only, want_field_score);
}

//: B2: the search kernel on the duplicated-edge field. One ALIGNED 32-byte
//: load per corner, then the same two `vperm2f128` B1 uses to split left from
//: right. Identical arithmetic to every other arm.
template <bool PIPE, bool ALLIN = false, bool GRIDXF = false>
inline void
field_gradient_avx2_edgefield(const FieldEdge32& f, const ConformerSoA& b,
                              const float m[9], const float t[3], float gnorm,
                              float g[7], float* out_s, OpCensus* census) {
    const FieldPose<GRIDXF> pose(f.g, m, t);
    const float* pm = pose.m;
    const float* pt = pose.t;
    const float* base = f.v.data();
    const size_t row = size_t(f.g.nodes_x()) * 8;
    const size_t plane = row * size_t(f.g.nodes_y());

    __m256 vs = _mm256_setzero_ps();
    __m256 Gx = _mm256_setzero_ps(), Gy = _mm256_setzero_ps(),
           Gz = _mm256_setzero_ps();
    __m256 T1 = _mm256_setzero_ps(), T2 = _mm256_setzero_ps(),
           T3 = _mm256_setzero_ps();
    PreparedBlock pb;

    const int n_full = b.n & ~7;
    PreparedBlock pb_next;
    if constexpr (PIPE) {
        if (b.n > 0) {
            const int n0 = (b.n < 8) ? b.n : 8;
            if (n_full > 0)
                prepare_block<true, 8, ALLIN, GRIDXF>(f, b, pm, pt, 0, 8, pb);
            else
                prepare_block<false, 8, ALLIN, GRIDXF>(f, b, pm, pt, 0, n0, pb);
        }
    }
    for (int j0 = 0; j0 < b.n; j0 += 8) {
        const int n = (b.n - j0 < 8) ? (b.n - j0) : 8;
        if constexpr (PIPE) {
            const int jn = j0 + 8;
            if (jn < b.n) {
                const int nn = (b.n - jn < 8) ? (b.n - jn) : 8;
                if (jn < n_full)
                    prepare_block<true, 8, ALLIN, GRIDXF>(f, b, pm, pt, jn, 8,
                                                          pb_next);
                else
                    prepare_block<false, 8, ALLIN, GRIDXF>(f, b, pm, pt, jn, nn,
                                                           pb_next);
            }
        } else {
            if (j0 < n_full)
                prepare_block<true, 8, ALLIN, GRIDXF>(f, b, pm, pt, j0, 8, pb);
            else
                prepare_block<false, 8, ALLIN, GRIDXF>(f, b, pm, pt, j0, n, pb);
        }

        auto edge2 = [](const float* a, const float* bb, __m256& l, __m256& r) {
            // ALIGNED: every record is 32-byte aligned by construction.
            const __m256 ea = _mm256_load_ps(a);
            const __m256 eb = _mm256_load_ps(bb);
            l = _mm256_permute2f128_ps(ea, eb, 0x20);
            r = _mm256_permute2f128_ps(ea, eb, 0x31);
        };
        __m256 v[4];
        auto do_pair = [&](int lo, int hi, __m256 wx, __m256 wy, __m256 wz) {
            const float* a0 = base + pb.off[lo];
            const float* b0 = base + pb.off[hi];
            __m256 l, r;
            edge2(a0, b0, l, r);
            const __m256 x00 = lerp256(l, r, wx);
            edge2(a0 + row, b0 + row, l, r);
            const __m256 x10 = lerp256(l, r, wx);
            edge2(a0 + plane, b0 + plane, l, r);
            const __m256 x01 = lerp256(l, r, wx);
            edge2(a0 + plane + row, b0 + plane + row, l, r);
            const __m256 x11 = lerp256(l, r, wx);
            return lerp256(lerp256(x00, x10, wy), lerp256(x01, x11, wy), wz);
        };
        v[0] = do_pair(0, 4, pair_splat<0>(pb.fx), pair_splat<0>(pb.fy),
                       pair_splat<0>(pb.fz));
        v[1] = do_pair(1, 5, pair_splat<1>(pb.fx), pair_splat<1>(pb.fy),
                       pair_splat<1>(pb.fz));
        v[2] = do_pair(2, 6, pair_splat<2>(pb.fx), pair_splat<2>(pb.fy),
                       pair_splat<2>(pb.fz));
        v[3] = do_pair(3, 7, pair_splat<3>(pb.fx), pair_splat<3>(pb.fy),
                       pair_splat<3>(pb.fz));

        __m256 sf, sgx, sgy, sgz;
        to_soa(v[0], v[1], v[2], v[3], sf, sgx, sgy, sgz);
        vs = _mm256_add_ps(vs, sf);
        Gx = _mm256_add_ps(Gx, sgx);
        Gy = _mm256_add_ps(Gy, sgy);
        Gz = _mm256_add_ps(Gz, sgz);
        T1 = _mm256_add_ps(
            T1, _mm256_fmsub_ps(pb.c2, sgz, _mm256_mul_ps(pb.c3, sgy)));
        T2 = _mm256_add_ps(
            T2, _mm256_fmsub_ps(pb.c3, sgx, _mm256_mul_ps(pb.c1, sgz)));
        T3 = _mm256_add_ps(
            T3, _mm256_fmsub_ps(pb.c1, sgy, _mm256_mul_ps(pb.c2, sgx)));
        if (census) census->field_samples += uint64_t(n);
        if constexpr (PIPE) pb = pb_next;
    }

    auto hsum = [](__m256 x) {
        __m128 lo = _mm256_castps256_ps128(x);
        lo = _mm_add_ps(lo, _mm256_extractf128_ps(x, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        return _mm_cvtss_f32(lo);
    };
    const float gfac = -gnorm;
    g[0] = 0.0f;
    pose.write_torque(gfac, hsum(T1), hsum(T2), hsum(T3), g);
    g[4] = gfac * hsum(Gx);
    g[5] = gfac * hsum(Gy);
    g[6] = gfac * hsum(Gz);
    if (out_s) *out_s = hsum(vs);
}

template <bool PIPE, bool ALLIN_DISPATCH = false, bool GRIDXF = false>
inline PairResult optimize_pair_field_edgefield(
    const Conformer& a, const Conformer& b, const FieldEdge32& f, float s_aa,
    float s_bb, int start_mode, int n_steps, float lr_q, float lr_t_scaled,
    OpCensus* census, bool winner_only = false, bool want_field_score = false) {
    ConformerSoA soa;
    soa.build(b);
    return optimize_pair_field_with(
        a, b, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled, census,
        [&soa](const FieldEdge32& ff, const Conformer&, const float mm[9],
               const float tt[3], float gn, float gg[7], float* os,
               OpCensus* cc) {
            if constexpr (ALLIN_DISPATCH) {
                // Per POSE, not per atom, exactly as the shipped arm
                // does it: six scalar comparisons over ~30 atoms, and
                // a false negative costs only the general path.
                if (cc) ++cc->allin_pose_visits;
                if (conformer_inside<GRIDXF>(ff, soa, mm, tt)) {
                    if (cc) ++cc->allin_poses;
                    field_gradient_avx2_edgefield<PIPE, true, GRIDXF>(
                        ff, soa, mm, tt, gn, gg, os, cc);
                    return;
                }
            }
            field_gradient_avx2_edgefield<PIPE, false, GRIDXF>(ff, soa, mm, tt,
                                                               gn, gg, os, cc);
        },
        winner_only, want_field_score);
}

//: `PIPE` software-pipelines the prepare one block ahead.
//:
//: The serial chain from a coordinate to its field load is long: transform,
//: lattice map, truncate, multiply out the linear index, shift, then the
//: offset leaves the vector unit and comes back as a scalar address before the
//: load can even issue. Within a block the four pairs are already independent,
//: which is why unrolling THEM bought nothing; what is not overlapped is the
//: prepare of the NEXT block against the tree of the current one.
//:
//: The cost is a second live `PreparedBlock` -- six more YMM values and eight
//: more ints -- in a kernel that already uses all sixteen architectural
//: registers and spills. Whether the overlap beats the extra pressure is
//: exactly what this measures.
template <bool EDGE, bool PIPE, int STAGE = 0, bool ALLIN = false,
          bool GRIDXF = false, bool XLAST = (R3_FIELD_AVX2_XLAST != 0),
          bool STRICT_BOUNDS = (R3_FIELD_AVX2_STRICT_BOUNDS != 0),
          bool ALTERNATING = (R3_FIELD_AVX2_ALTERNATING != 0),
          bool TWO_PAIRS = (R3_FIELD_AVX2_SCHEDULE != 0),
          bool NEED_GRADIENT = true, class Field>
inline void
field_gradient_avx2_reg(const Field& f, const ConformerSoA& b, const float m[9],
                        const float t[3], float gnorm, float g[7], float* out_s,
                        OpCensus* census, const PreparedBlock* pre = nullptr) {
    using Node = std::remove_pointer_t<decltype(f.node(0, 0, 0))>;
    const Node* base = f.v.data();
    const size_t row = size_t(f.g.nodes_x()) * 4;
    const size_t plane = row * size_t(f.g.nodes_y());

    __m256 vs = _mm256_setzero_ps();
    __m256 Gx = _mm256_setzero_ps(), Gy = _mm256_setzero_ps(),
           Gz = _mm256_setzero_ps();
    __m256 T1 = _mm256_setzero_ps(), T2 = _mm256_setzero_ps(),
           T3 = _mm256_setzero_ps();
    PreparedBlock storage[(PIPE && STAGE != 2) ? 2 : 1];
    PreparedBlock* current = &storage[0];
    PreparedBlock* following = &storage[(PIPE && STAGE != 2) ? 1 : 0];

    // A3: the full blocks and the tail are separate loops, so the tail's mask
    // never runs on a full block. `n_full` is a multiple of eight.
    const FieldPose<GRIDXF> pose(f.g, m, t);
    const float* pm = pose.m;
    const float* pt = pose.t;

    if constexpr (STAGE == 3) {
        for (int k = 0; k < 7; ++k)
            g[k] = 0.0f;
        if (out_s) *out_s = 0.0f;
        return;
    }
    const int n_full = b.n & ~7;
    int block_index = 0;
    if constexpr (PIPE && STAGE != 2) {
        if (b.n > 0) {
            const int n0 = (b.n < 8) ? b.n : 8;
            if (n_full > 0)
                prepare_block<true, 4, ALLIN, GRIDXF, STRICT_BOUNDS>(
                    f, b, pm, pt, 0, 8, *current);
            else
                prepare_block<false, 4, ALLIN, GRIDXF, STRICT_BOUNDS>(
                    f, b, pm, pt, 0, n0, *current);
        }
    }
    for (int j0 = 0; j0 <= b.n; j0 += 8) {
        if (j0 >= b.n) break;
        const int n = (b.n - j0 < 8) ? (b.n - j0) : 8;
        if constexpr (STAGE == 2) {
            // Reference selection is compile-time; no precomputed block copy.
        } else if constexpr (PIPE) {
            // Prepare the NEXT block before this one's field data is touched,
            // so its address chain -- transform, map, truncate, linear index,
            // shift, vector-to-scalar -- runs under this block's loads and
            // lerps instead of ahead of them.
            const int jn = j0 + 8;
            if (jn < b.n) {
                const int nn = (b.n - jn < 8) ? (b.n - jn) : 8;
                if (jn < n_full)
                    prepare_block<true, 4, ALLIN, GRIDXF, STRICT_BOUNDS>(
                        f, b, pm, pt, jn, 8, *following);
                else
                    prepare_block<false, 4, ALLIN, GRIDXF, STRICT_BOUNDS>(
                        f, b, pm, pt, jn, nn, *following);
            }
        } else {
            if (j0 < n_full)
                prepare_block<true, 4, ALLIN, GRIDXF, STRICT_BOUNDS>(
                    f, b, pm, pt, j0, 8, *current);
            else
                prepare_block<false, 4, ALLIN, GRIDXF, STRICT_BOUNDS>(
                    f, b, pm, pt, j0, n, *current);
        }

        const PreparedBlock& pb = [&]() -> const PreparedBlock& {
            if constexpr (STAGE == 2)
                return pre[block_index++];
            else
                return *current;
        }();
        if constexpr (STAGE == 1) {
            vs = _mm256_add_ps(
                vs, _mm256_add_ps(_mm256_add_ps(pb.fx, pb.fy), pb.fz));
            Gx = _mm256_add_ps(Gx, pb.c1);
            Gy = _mm256_add_ps(Gy, pb.c2);
            Gz = _mm256_add_ps(Gz, pb.c3);
            T1 = _mm256_add_ps(T1,
                               _mm256_cvtepi32_ps(_mm256_load_si256(
                                   reinterpret_cast<const __m256i*>(pb.off))));
        } else {
            // One pair of samples, with no lane crossing after the load: the
            // corners are assembled as [A | B] and every lerp is elementwise.
            //
            // Pair k with k+4, NOT k with k+1. The two schedules cost the same in
            // loads and lerps; this one makes every weight an immediate lane-local
            // shuffle instead of a variable permute, and makes the transpose below
            // land in natural point order.
            __m256 v[4];
            auto do_pair = [&](int lo, int hi, __m256 wx, __m256 wy,
                               __m256 wz) {
                const Node* a0 = base + pb.off[lo];
                const Node* b0 = base + pb.off[hi];
                if constexpr (XLAST)
                    return interp_pair_xlast(
                        a0, a0 + row, a0 + plane, a0 + plane + row, b0,
                        b0 + row, b0 + plane, b0 + plane + row, wx, wy, wz);
                else if constexpr (EDGE)
                    return interp_pair_edge(
                        a0, a0 + row, a0 + plane, a0 + plane + row, b0,
                        b0 + row, b0 + plane, b0 + plane + row, wx, wy, wz);
                else
                    return interp_pair(
                        a0, a0 + row, a0 + plane, a0 + plane + row, b0,
                        b0 + row, b0 + plane, b0 + plane + row, wx, wy, wz);
            };
            if constexpr (TWO_PAIRS && !XLAST && !EDGE &&
                          std::is_same_v<std::remove_cv_t<Node>, float>) {
                interp_two_pairs_xyz<0, 1>(base, pb, row, plane, v[0], v[1]);
                interp_two_pairs_xyz<2, 3>(base, pb, row, plane, v[2], v[3]);
            } else {
                v[0] = do_pair(0, 4, pair_splat<0>(pb.fx), pair_splat<0>(pb.fy),
                               pair_splat<0>(pb.fz));
                v[1] = do_pair(1, 5, pair_splat<1>(pb.fx), pair_splat<1>(pb.fy),
                               pair_splat<1>(pb.fz));
                v[2] = do_pair(2, 6, pair_splat<2>(pb.fx), pair_splat<2>(pb.fy),
                               pair_splat<2>(pb.fz));
                v[3] = do_pair(3, 7, pair_splat<3>(pb.fx), pair_splat<3>(pb.fy),
                               pair_splat<3>(pb.fz));
            }

            __m256 sf, sgx, sgy, sgz;
            to_soa(v[0], v[1], v[2], v[3], sf, sgx, sgy, sgz);
            // Natural order now, so the lever arms are used as prepared.
            const __m256 c1 = pb.c1;
            const __m256 c2 = pb.c2;
            const __m256 c3 = pb.c3;

            vs = _mm256_add_ps(vs, sf);
            Gx = _mm256_add_ps(Gx, sgx);
            Gy = _mm256_add_ps(Gy, sgy);
            Gz = _mm256_add_ps(Gz, sgz);
            // C x grad, in SoA: no [_,x,y,z] vector, no cbuf, no shuffles.
            T1 = _mm256_add_ps(
                T1, _mm256_fmsub_ps(c2, sgz, _mm256_mul_ps(c3, sgy)));
            T2 = _mm256_add_ps(
                T2, _mm256_fmsub_ps(c3, sgx, _mm256_mul_ps(c1, sgz)));
            T3 = _mm256_add_ps(
                T3, _mm256_fmsub_ps(c1, sgy, _mm256_mul_ps(c2, sgx)));
        }
        if (census) census->field_samples += uint64_t(n);
        if constexpr (PIPE && STAGE != 2) {
            // Both diagnostic and complete paths advance once. The final
            // iteration never reads uninitialized following storage.
            if (j0 + 8 < b.n) {
                if constexpr (ALTERNATING) {
                    PreparedBlock* previous = current;
                    current = following;
                    following = previous;
                } else {
                    *current =
                        *following;  // Retained original-copy comparator.
                }
            }
        }
    }

    auto hsum = [](__m256 x) {
        __m128 lo = _mm256_castps256_ps128(x);
        __m128 hi = _mm256_extractf128_ps(x, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        return _mm_cvtss_f32(lo);
    };
    if constexpr (NEED_GRADIENT) {
        const float gfac = -gnorm;
        g[0] = 0.0f;
        const float sgx = hsum(Gx), sgy = hsum(Gy), sgz = hsum(Gz);
        pose.write_torque(gfac, hsum(T1), hsum(T2), hsum(T3), g);
        g[4] = gfac * sgx;
        g[5] = gfac * sgy;
        g[6] = gfac * sgz;
    }
    if (out_s) *out_s = hsum(vs);
}

template <class Field>
inline void field_gradient_avx2_blocked(const Field& f, const ConformerSoA& b,
                                        const float m[9], const float t[3],
                                        float gnorm, float g[7], float* out_s,
                                        OpCensus* census) {
    using Node = std::remove_pointer_t<decltype(f.node(0, 0, 0))>;
    const float gfac = -gnorm;
    const size_t row = size_t(f.g.nodes_x()) * 4;
    const size_t plane = row * size_t(f.g.nodes_y());
    __m256 acc = _mm256_setzero_ps();
    __m256 tor = _mm256_setzero_ps();
    alignas(32) float cbuf[8];
    AtomBlock blk;

    for (int j0 = 0; j0 < b.n; j0 += 8) {
        const int n = (b.n - j0 < 8) ? (b.n - j0) : 8;
        prepare_eight(f, b, m, t, j0, n, blk);

        PairStencilT<Node> st[8];
        for (int k = 0; k < 8; ++k) {
            if (blk.ok[k]) {
                st[k].p00 = f.node(blk.ix[k], blk.iy[k], blk.iz[k]);
                st[k].p10 = st[k].p00 + row;
                st[k].p01 = st[k].p00 + plane;
                st[k].p11 = st[k].p01 + row;
            } else {
                st[k].p00 = st[k].p10 = st[k].p01 = st[k].p11 =
                    zero_node(st[k].p00);
                blk.fx[k] = blk.fy[k] = blk.fz[k] = 0.f;
                blk.c1[k] = blk.c2[k] = blk.c3[k] = 0.f;
            }
        }
        for (int u = 0; u < 8; u += 2) {
            __m256 az0, az1, bz0, bz1;
            x_stage(st[u], pack2_broadcast(blk.fx[u], blk.fx[u]), az0, az1);
            x_stage(st[u + 1], pack2_broadcast(blk.fx[u + 1], blk.fx[u + 1]),
                    bz0, bz1);
            const __m256 v = finish_pair(
                az0, az1, bz0, bz1, pack2_broadcast(blk.fy[u], blk.fy[u + 1]),
                pack2_broadcast(blk.fz[u], blk.fz[u + 1]));
            cbuf[0] = 0.f;
            cbuf[1] = blk.c1[u];
            cbuf[2] = blk.c2[u];
            cbuf[3] = blk.c3[u];
            cbuf[4] = 0.f;
            cbuf[5] = blk.c1[u + 1];
            cbuf[6] = blk.c2[u + 1];
            cbuf[7] = blk.c3[u + 1];
            const __m256 c = _mm256_load_ps(cbuf);
            acc = _mm256_add_ps(acc, v);
            tor = _mm256_add_ps(tor, cross3(c, v));
        }
        if (census) census->field_samples += uint64_t(n);
    }

    alignas(32) float ab[8], tb[8];
    _mm256_store_ps(ab, acc);
    _mm256_store_ps(tb, tor);
    g[0] = 0.0f;
    g[1] = gfac * (tb[1] + tb[5]);
    g[2] = gfac * (tb[2] + tb[6]);
    g[3] = gfac * (tb[3] + tb[7]);
    g[4] = gfac * (ab[1] + ab[5]);
    g[5] = gfac * (ab[2] + ab[6]);
    g[6] = gfac * (ab[3] + ab[7]);
    if (out_s) *out_s = ab[0] + ab[4];
}

inline void field_gradient_avx2(const FieldNaive32& f, const Conformer& b,
                                const float m[9], const float t[3], float gnorm,
                                float g[7], float* out_s, OpCensus* census) {
    const float gfac = -gnorm;
    __m256 acc = _mm256_setzero_ps();  // [f, gx, gy, gz | ...]
    __m256 tor = _mm256_setzero_ps();  // [_, tx, ty, tz | ...]

    alignas(32) float cbuf[8];
    int j = 0;
    for (; j + 1 < b.n_atoms; j += 2) {
        PairStencil sa, sb;
        float ax, ay, az, bx_, by_, bz_;
        bool oka, okb;
        float ca[4], cb[4];
        {
            const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1],
                        bz = b.xyz[3 * j + 2];
            const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
            const float mby = m[3] * bx + m[4] * by + m[5] * bz;
            const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
            ca[0] = 0.f;
            ca[1] = 2.f * mbx;
            ca[2] = 2.f * mby;
            ca[3] = 2.f * mbz;
            oka = stencil_of(f, mbx + t[0], mby + t[1], mbz + t[2], sa, ax, ay,
                             az);
        }
        {
            const int k = j + 1;
            const float bx = b.xyz[3 * k], by = b.xyz[3 * k + 1],
                        bz = b.xyz[3 * k + 2];
            const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
            const float mby = m[3] * bx + m[4] * by + m[5] * bz;
            const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
            cb[0] = 0.f;
            cb[1] = 2.f * mbx;
            cb[2] = 2.f * mby;
            cb[3] = 2.f * mbz;
            okb = stencil_of(f, mbx + t[0], mby + t[1], mbz + t[2], sb, bx_,
                             by_, bz_);
        }
        // A sample outside the lattice contributes exactly zero. Zeroing the
        // STENCIL rather than branching keeps both lanes on one path -- the
        // rule that padding is a mask and never a branch, applied to the
        // boundary instead of to the atom count.
        static const float kZero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (!oka) {
            sa.p00 = sa.p10 = sa.p01 = sa.p11 = kZero;
            ax = ay = az = 0.f;
        }
        if (!okb) {
            sb.p00 = sb.p10 = sb.p01 = sb.p11 = kZero;
            bx_ = by_ = bz_ = 0.f;
        }

        __m256 a_z0, a_z1, b_z0, b_z1;
        x_stage(sa, pack2_broadcast(ax, ax), a_z0, a_z1);
        x_stage(sb, pack2_broadcast(bx_, bx_), b_z0, b_z1);
        const __m256 v =
            finish_pair(a_z0, a_z1, b_z0, b_z1, pack2_broadcast(ay, by_),
                        pack2_broadcast(az, bz_));
        for (int k = 0; k < 4; ++k) {
            cbuf[k] = ca[k];
            cbuf[4 + k] = cb[k];
        }
        const __m256 c = _mm256_load_ps(cbuf);
        acc = _mm256_add_ps(acc, v);
        tor = _mm256_add_ps(tor, cross3(c, v));
        if (census) census->field_samples += 2;
    }

    // Fold the two halves together, then the odd atom if there is one.
    alignas(32) float ab[8], tb[8];
    _mm256_store_ps(ab, acc);
    _mm256_store_ps(tb, tor);
    float vs = ab[0] + ab[4];
    float G[3] = {ab[1] + ab[5], ab[2] + ab[6], ab[3] + ab[7]};
    float T[3] = {tb[1] + tb[5], tb[2] + tb[6], tb[3] + tb[7]};

    for (; j < b.n_atoms; ++j) {
        const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1],
                    bz = b.xyz[3 * j + 2];
        const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
        const float mby = m[3] * bx + m[4] * by + m[5] * bz;
        const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
        float vs_, fx_, fy_, fz_;
        sample_field(f, mbx + t[0], mby + t[1], mbz + t[2], vs_, fx_, fy_, fz_);
        const float C[3] = {2.f * mbx, 2.f * mby, 2.f * mbz};
        vs += vs_;
        G[0] += fx_;
        G[1] += fy_;
        G[2] += fz_;
        T[0] += C[1] * fz_ - C[2] * fy_;
        T[1] += C[2] * fx_ - C[0] * fz_;
        T[2] += C[0] * fy_ - C[1] * fx_;
        if (census) ++census->field_samples;
    }

    // `gfac` is applied ONCE, at the end, to the summed gradient rather than
    // per sample. Legitimate because it is a per-pair constant and the sum is
    // linear in it -- and it saves n_B multiplies per step.
    g[0] = 0.0f;
    g[1] = gfac * T[0];
    g[2] = gfac * T[1];
    g[3] = gfac * T[2];
    g[4] = gfac * G[0];
    g[5] = gfac * G[1];
    g[6] = gfac * G[2];
    if (out_s) *out_s = vs;
}

}  // namespace cpu
}  // namespace r3

#endif  // __AVX2__

#if defined(__AVX2__)
namespace r3 {
namespace cpu {

//: The paired-AVX2 field search. Same optimizer as the scalar arm -- only the
//: gradient differs, which is what makes the two comparable.
//: Pair-AoS accumulation, re-offered as an arm now that pipelining changed the
//: register picture it was originally judged against.
template <bool PIPE, bool ALLIN_DISPATCH = false, bool GRIDXF = false,
          class Field>
inline PairResult optimize_pair_field_pairaos(
    const Conformer& a, const Conformer& b, const Field& f, float s_aa,
    float s_bb, int start_mode, int n_steps, float lr_q, float lr_t_scaled,
    OpCensus* census, bool winner_only = false, bool want_field_score = false) {
    ConformerSoA soa;
    soa.build(b);
    return optimize_pair_field_with(
        a, b, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled, census,
        [&soa](const Field& ff, const Conformer&, const float mm[9],
               const float tt[3], float gn, float gg[7], float* os,
               OpCensus* cc) {
            if constexpr (ALLIN_DISPATCH) {
                // Per POSE, not per atom, exactly as the shipped arm
                // does it: six scalar comparisons over ~30 atoms, and
                // a false negative costs only the general path.
                if (cc) ++cc->allin_pose_visits;
                if (conformer_inside<GRIDXF>(ff, soa, mm, tt)) {
                    if (cc) ++cc->allin_poses;
                    field_gradient_avx2_pairaos<PIPE, true, GRIDXF>(
                        ff, soa, mm, tt, gn, gg, os, cc);
                    return;
                }
            }
            field_gradient_avx2_pairaos<PIPE, false, GRIDXF>(ff, soa, mm, tt,
                                                             gn, gg, os, cc);
        },
        winner_only, want_field_score);
}

template <bool EDGE, bool PIPE, int STAGE = 0, bool ALLIN_DISPATCH = false,
          bool GRIDXF = false, class Field>
inline PairResult optimize_pair_field_blocked(
    const Conformer& a, const Conformer& b, const Field& f, float s_aa,
    float s_bb, int start_mode, int n_steps, float lr_q, float lr_t_scaled,
    OpCensus* census, bool winner_only = false, bool want_field_score = false) {
    // Transposed once here, outside the step loop, and captured by value so
    // the gradient sees the SoA record rather than the AoS one.
    ConformerSoA soa;
    soa.build(b);
    if constexpr (STAGE == 2) {
        // Blocks prepared once at step 0 and reused for all of them: the poses
        // this reports are wrong, which is why it is a probe and never a
        // scoring path. Declared inside the branch so no other stage carries it
        // in the lambda's capture.
        alignas(32) PreparedBlock cache[9];
        bool filled = false;
        return optimize_pair_field_with(
            a, b, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled, census,
            [&soa, &cache, &filled](const Field& ff, const Conformer&,
                                    const float mm[9], const float tt[3],
                                    float gn, float gg[7], float* os,
                                    OpCensus* cc) {
                if (!filled) {
                    const int n_full = soa.n & ~7;
                    int bi = 0;
                    for (int j0 = 0; j0 < soa.n; j0 += 8, ++bi) {
                        const int n = (soa.n - j0 < 8) ? (soa.n - j0) : 8;
                        if (j0 < n_full)
                            prepare_block<true, 4, false, false>(
                                ff, soa, mm, tt, j0, 8, cache[bi]);
                        else
                            prepare_block<false, 4, false, false>(
                                ff, soa, mm, tt, j0, n, cache[bi]);
                    }
                    filled = true;
                }
                field_gradient_avx2_reg<EDGE, PIPE, 2>(ff, soa, mm, tt, gn, gg,
                                                       os, cc, cache);
            });
    } else {
        return optimize_pair_field_with(
            a, b, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled, census,
            [&soa](const Field& ff, const Conformer&, const float mm[9],
                   const float tt[3], float gn, float gg[7], float* os,
                   OpCensus* cc) {
                if constexpr (ALLIN_DISPATCH) {
                    // Per POSE, not per atom: six scalar comparisons amortized
                    // over ~30 atoms. A false negative costs the general path,
                    // which is what the else branch is.
                    if (cc) ++cc->allin_pose_visits;
                    if (conformer_inside<GRIDXF>(ff, soa, mm, tt)) {
                        if (cc) ++cc->allin_poses;
                        field_gradient_avx2_reg<EDGE, PIPE, STAGE, true,
                                                GRIDXF>(ff, soa, mm, tt, gn, gg,
                                                        os, cc);
                        return;
                    }
                }
                field_gradient_avx2_reg<EDGE, PIPE, STAGE, false, GRIDXF>(
                    ff, soa, mm, tt, gn, gg, os, cc);
            },
            winner_only, want_field_score);
    }
}

//: The shipping configuration, eight pairs at a time.
//:
//: Only this configuration. The blocked kernel has several template arms and
//: batching all of them would multiply instantiations for no measured reason --
//: the shipped path is `gridxf + allin + pipeline`, which is the
//: default, so that is the one that pays. Anything else keeps the scalar path.
template <bool EDGE, class Field>
inline void optimize_pairs_field_blocked(
    const Conformer& a, const Conformer* const* b, int n_lanes, const Field& f,
    float s_aa, const float* s_bb, int start_mode, int n_steps, float lr_q,
    float lr_t_scaled, OpCensus* census, PairResult* out,
    bool winner_only = false, bool want_field_score = false) {
    constexpr int LANES = 8;
    // One SoA per lane, transposed once outside the step loop exactly as the
    // scalar path does it -- the batching changes the pose update, not how a
    // conformer is laid out for the gradient.
    ConformerSoA soa[LANES];
    for (int p = 0; p < n_lanes; ++p)
        soa[p].build(*b[p]);

    optimize_pairs_field_with<LANES>(
        a, b, n_lanes, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled,
        census, out, winner_only, want_field_score,
        [&](int lane, const float* mm, const float* tt, float gn, float* gg,
            float* os, OpCensus* cc) {
            if (cc) ++cc->allin_pose_visits;
            if (conformer_inside<true>(f, soa[lane], mm, tt)) {
                if (cc) ++cc->allin_poses;
                field_gradient_avx2_reg<EDGE, true, 0, true, true>(
                    f, soa[lane], mm, tt, gn, gg, os, cc);
                return;
            }
            field_gradient_avx2_reg<EDGE, true, 0, false, true>(
                f, soa[lane], mm, tt, gn, gg, os, cc);
        });
}

template <int UNROLL, class Field>
inline PairResult optimize_pair_field_avx2(
    const Conformer& a, const Conformer& b, const Field& f, float s_aa,
    float s_bb, int start_mode, int n_steps, float lr_q, float lr_t_scaled,
    OpCensus* census, bool winner_only = false, bool want_field_score = false) {
    // No GRIDXF/ALLIN here, and that is a property of the arm rather than an
    // omission: this path prepares one SAMPLE at a time (`PreparedSampleT`)
    // rather than a block, so it shares no code with `prepare_block`, which is
    // where both of those live. Giving it either means writing them again.
    return optimize_pair_field_with(
        a, b, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled, census,
        [](const Field& ff, const Conformer& bb, const float mm[9],
           const float tt[3], float gn, float gg[7], float* os, OpCensus* cc) {
            field_gradient_avx2_unrolled<UNROLL>(ff, bb, mm, tt, gn, gg, os,
                                                 cc);
        },
        winner_only, want_field_score);
}

}  // namespace cpu
}  // namespace r3
#endif

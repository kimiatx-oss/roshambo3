// The query field on a CPU: build, layouts, and the sampler.
//
// The GPU answer is a `half4` hardware texture and **every part of that
// reasoning is a property of the texture unit** -- there is no TMU here, so a
// sample is an addressing problem rather than a free interpolation.
//
// WHAT IS FIXED AND WHAT VARIES. The representation is the shipped one and does
// not vary here: `half4 = (F - c.lap F, grad F - c.grad lap F)` with
// `c = hs^2/12`, mirroring `field_fill_kernel` (`kernels.cu:2929-2972`). What
// varies is layout, storage precision, spacing and kernel shape. The
// interpolant question -- Hermite, quadratic B-spline -- is separate.
//
// THE SIGN, WHICH IS THE DOCUMENTED FIRST-BUILD TRAP. The stored gradient
// already carries `-2 ln2` from the build (`cg = -2*b*e`), so the search must
// NOT apply it again: doing so scales *and flips the sign*, and produced grid
// Tanimoto 0.013 against the analytic path's 0.389 on the GPU. The consumer
// side is `gfac = -gnorm` (`kernels.cu:4234`), NOT `+gnorm`, and that minus is
// what makes the field torque agree with the analytic one term for term:
//
//     stored gx(r) = sum_i (-2 ln2)(r - a_i)_x k_ij = +2 ln2 sum_i delta_x k_ij
//     gxv = gfac * stored gx = -2 ln2 * gnorm * sum_i delta_x k_ij
//     analytic g4 = sum_i MINUS_2LN2 * gnorm * k_ij * delta_x        (identical)
//
// with `delta = (a_i - t) - Mb` and the sample taken at `r = Mb + t`.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#if defined(__F16C__)
#include <immintrin.h>
#endif

#include "kernel.hpp"

namespace r3 {
namespace cpu {

//: ln2, in the spelling the CUDA field build uses.
constexpr float FIELD_LN2 = 0.6931471805599453f;

//: The lattice. Reduced frame throughout, as `plan_field_lattice_range` builds
//: it -- with ONE difference from the CUDA convention that has to be stated
//: because it is invisible and it bit on the first build.
//:
//: **The CUDA build takes the query RAW and scales it here; this one takes it
//: ALREADY REDUCED and must not.** `interface.cpp` pre-scales `qbuf` by
//: COORD_SCALE at load (and says why: the library is not pre-scaled because the
//: pose matrix carries the scale instead). Scaling again inside the field build
//: shrinks the query by COORD_SCALE^2 = 0.604 while the library still arrives
//: singly scaled.
//:
//: That defect is nearly invisible, which is why it is written down: the
//: lattice is placed around the same doubly-scaled query, so the geometry stays
//: SELF-CONSISTENT and the field looks fine -- only the query's position
//: relative to the library is wrong. It cost ~6% of mean Tanimoto and, being a
//: geometry error rather than an interpolation one, **did not improve as h
//: shrank**: identical at h = 0.8 and h = 0.1. The h-independence is what
//: identified it.
struct FieldGeometry {
    float ox = 0.f, oy = 0.f, oz = 0.f;
    float hs = 1.f, inv_hs = 1.f;
    int nx = 1, ny = 1, nz = 1;  // CELLS

    //: Extra zeroed nodes past the lattice, so an out-of-bounds sample can be
    //: given a real offset instead of a branch. A stencil reads p00, p00+row,
    //: p00+plane and p00+plane+row, so the tail must cover plane+row+8 floats
    //: for all four to land inside it. The field is exactly zero outside the
    //: lattice -- its true asymptote, which is what the GPU gets free from
    //: cudaAddressModeBorder -- so reading zeros there is the CORRECT answer
    //: and not merely a safe one.
    size_t zero_tail_floats() const {
        const size_t row = size_t(nodes_x()) * 4;
        return row * size_t(nodes_y()) + row + 8;
    }

    int nodes_x() const {
        return nx + 1;
    }

    int nodes_y() const {
        return ny + 1;
    }

    int nodes_z() const {
        return nz + 1;
    }

    size_t nodes() const {
        return size_t(nodes_x()) * size_t(nodes_y()) * size_t(nodes_z());
    }
};

//: `query_xyz` is ALREADY in the reduced frame -- see the note above.
inline FieldGeometry plan_field(const float* query_xyz, int n_atoms,
                                float h_angstrom, float pad_a) {
    const double s = double(COORD_SCALE);
    // `h` and the pad are quoted in Angstrom, so THEY are scaled; the query is
    // not, because it arrives scaled.
    const double hs = double(h_angstrom) * s;
    const double pad = double(pad_a) * s;
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (int i = 0; i < n_atoms; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double p = double(query_xyz[3 * i + k]);
            if (p < lo[k]) lo[k] = p;
            if (p > hi[k]) hi[k] = p;
        }
    }
    if (n_atoms <= 0) {
        lo[0] = lo[1] = lo[2] = hi[0] = hi[1] = hi[2] = 0.0;
    }
    FieldGeometry g;
    g.hs = float(hs);
    g.inv_hs = float(1.0 / hs);
    g.ox = float(lo[0] - pad);
    g.oy = float(lo[1] - pad);
    g.oz = float(lo[2] - pad);
    g.nx = std::max(1, int(std::ceil((hi[0] + pad - (lo[0] - pad)) / hs)));
    g.ny = std::max(1, int(std::ceil((hi[1] + pad - (lo[1] - pad)) / hs)));
    g.nz = std::max(1, int(std::ceil((hi[2] + pad - (lo[2] - pad)) / hs)));
    return g;
}

//: Allocator that over-aligns to a cache line.
//:
//: The field's loads are 16 bytes wide (one node, four channels) and a node is
//: 16 bytes, so every corner load is naturally aligned **provided the base is**.
//: `std::vector<float>` gets that from malloc's 16-byte guarantee on x86-64,
//: which is true and is not something this code says anywhere -- an incidental
//: property the kernel would silently depend on, and which a custom allocator
//: or a different container would break. Since the kernel uses ALIGNED load
//: intrinsics, breaking it is a fault at run time rather than a slowdown.
//:
//: 64 rather than 16 so the base is also cache-line aligned, which additionally
//: makes the 32-byte x-edge loads of the B1 arm aligned exactly half the time
//: rather than at the mercy of where malloc happened to land.
template <class T, size_t A = 64>
struct AlignedAllocator {
    using value_type = T;

    //: `allocator_traits`' default rebind cannot see through the non-type
    //: alignment parameter, so it is spelled out.
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, A>;
    };

    AlignedAllocator() = default;

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, A>&) {}

    T* allocate(size_t n) {
        void* p = nullptr;
        const size_t bytes = ((n * sizeof(T)) + A - 1) / A * A;
        if (posix_memalign(&p, A, bytes ? bytes : A) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(p);
    }

    void deallocate(T* p, size_t) noexcept {
        free(p);
    }

    template <class U>
    bool operator==(const AlignedAllocator<U, A>&) const {
        return true;
    }

    template <class U>
    bool operator!=(const AlignedAllocator<U, A>&) const {
        return false;
    }
};

template <class T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

//: Node-major `[z][y][x][4]`, fp32. The control layout: x is contiguous, so a
//: stencil is four x-PAIR loads rather than eight scattered reads, which is the
//: baseline the brick has to beat.
struct FieldNaive32 {
    FieldGeometry g;
    AlignedVector<float> v;
#if defined(__aarch64__)
    // Immutable query preparation proves ELEMENT offsets, including all
    // valid node channels and the minimal eight-float sentinel. Unproved or
    // oversized fields retain the size_t evaluator; no hot-loop discovery.
    bool neon_offsets32 = false;

    void prepare_neon_offsets32() {
        constexpr uint64_t limit = UINT32_MAX;
        uint64_t nodes = 1;
        neon_offsets32 = false;
        for (int cells : {g.nx, g.ny, g.nz}) {
            // Exact FP32 cell bounds are part of the conversion proof.
            if (cells < 1 || cells > (1 << 24)) return;
            const uint64_t dimension = uint64_t(cells) + 1;
            if (nodes > ((limit - 7) / 4) / dimension) return;
            nodes *= dimension;
        }
        const uint64_t zero = nodes * 4;
        neon_offsets32 = zero + 7 <= limit && v.size() >= zero + 8;
    }
#endif

    //: Offset of the zeroed tail, in scalars. An out-of-bounds sample is
    //: pointed here rather than branched around.
    size_t zero_offset() const {
        return g.nodes() * 4;
    }  // 4 per node

    const float* node(int ix, int iy, int iz) const {
        return &v[(((size_t(iz) * size_t(g.nodes_y()) + size_t(iy)) *
                    size_t(g.nodes_x())) +
                   size_t(ix)) *
                  4];
    }

    size_t bytes() const {
        return v.size() * sizeof(float);
    }
};

//: Build, mirroring `field_fill_kernel` term for term including the Laplacian
//: prefilter. `query_xyz` is ALREADY REDUCED (see `plan_field`); unlike the
//: CUDA build, this one must not scale it again.
inline void build_field(const float* query_xyz, const float* query_w,
                        int n_atoms, FieldGeometry g, FieldNaive32& out) {
    out.g = g;
    out.v.assign(g.nodes() * 4 + g.zero_tail_floats(), 0.0f);
#if defined(__aarch64__)
    out.prepare_neon_offsets32();
#endif
    // UNIT_FAC: the reduced frame makes the pair exponent exactly
    // exp2(-|delta'|^2), so `fac` is 1 and `b` is ln2 (NUMERICS_POLICY section 4).
    const float b = FIELD_LN2;
    const float b2 = b * b, b3 = b2 * b;
    const float c = g.hs * g.hs / 12.0f;

    for (int iz = 0; iz < g.nodes_z(); ++iz) {
        for (int iy = 0; iy < g.nodes_y(); ++iy) {
            for (int ix = 0; ix < g.nodes_x(); ++ix) {
                const float x = g.ox + float(ix) * g.hs;
                const float y = g.oy + float(iy) * g.hs;
                const float z = g.oz + float(iz) * g.hs;
                float F = 0.f, Lp = 0.f;
                float g0 = 0.f, g1 = 0.f, g2 = 0.f;
                float l0 = 0.f, l1 = 0.f, l2 = 0.f;
                for (int i = 0; i < n_atoms; ++i) {
                    const float ax = query_xyz[3 * i + 0];
                    const float ay = query_xyz[3 * i + 1];
                    const float az = query_xyz[3 * i + 2];
                    const float w = query_w ? query_w[i] : 1.0f;
                    const float dx = x - ax, dy = y - ay, dz = z - az;
                    const float d2 = dx * dx + dy * dy + dz * dz;
                    const float e = std::exp2(-d2) * w;
                    F += e;
                    Lp += e * (4.0f * b2 * d2 - 6.0f * b);
                    const float cg = -2.0f * b * e;
                    const float cl = e * (20.0f * b2 - 8.0f * b3 * d2);
                    g0 += cg * dx;
                    g1 += cg * dy;
                    g2 += cg * dz;
                    l0 += cl * dx;
                    l1 += cl * dy;
                    l2 += cl * dz;
                }
                float* n =
                    &out.v[(((size_t(iz) * size_t(g.nodes_y()) + size_t(iy)) *
                             size_t(g.nodes_x())) +
                            size_t(ix)) *
                           4];
                n[0] = F - c * Lp;
                n[1] = g0 - c * l0;
                n[2] = g1 - c * l1;
                n[3] = g2 - c * l2;
            }
        }
    }
}

//: The same node array in fp16. Half the bytes, and the reason to try it is
//: not arithmetic: a stencil is four x-PAIR loads, and in fp32 each pair is 32 B
//: while in fp16 it is 16 B. The LINE count is unchanged -- four lines either
//: way -- so fp16 alone tests the BANDWIDTH mechanism, and the brick layout
//: tests the LINE-COUNT one. Measuring them separately is the only way to know
//: which is binding.
struct FieldNaive16 {
    FieldGeometry g;
    AlignedVector<uint16_t> v;

    //: Offset of the zeroed tail, in scalars. An out-of-bounds sample is
    //: pointed here rather than branched around.
    size_t zero_offset() const {
        return g.nodes() * 4;
    }

    const uint16_t* node(int ix, int iy, int iz) const {
        return &v[(((size_t(iz) * size_t(g.nodes_y()) + size_t(iy)) *
                    size_t(g.nodes_x())) +
                   size_t(ix)) *
                  4];
    }

    size_t bytes() const {
        return v.size() * sizeof(uint16_t);
    }
};

//: Storage-only FP16 uses FP32 interpolation: quantization happens once at
//: build and every lerp downstream stays FP32. The separate AArch64-only
//: mixed Y/Z experiment lives in field_neon_fp16.cpp and is selected explicitly
//: by the query-prepared evaluator; it does not change this AVX2 F16C path.
inline void quantize_field(const FieldNaive32& src, FieldNaive16& out) {
    out.g = src.g;
    out.v.assign(src.v.size(), 0);
#if defined(__F16C__)
    size_t i = 0;
    for (; i + 8 <= src.v.size(); i += 8) {
        const __m256 f = _mm256_loadu_ps(&src.v[i]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out.v[i]),
                         _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT));
    }
    for (; i < src.v.size(); ++i)
        out.v[i] = _cvtss_sh(src.v[i], _MM_FROUND_TO_NEAREST_INT);
#else
    for (size_t i = 0; i < src.v.size(); ++i)
        out.v[i] = 0;
#endif
}

//: Family C: per-cell trilinear COEFFICIENTS.
//:
//: The interpolant is unchanged -- this is the same trilinear polynomial in a
//: different basis, expanded from the corner values by finite differences:
//:
//:   f(u,v,w) = c000 + c100 u + c010 v + c001 w
//:                   + c110 uv + c101 uw + c011 vw + c111 uvw
//:
//: Evaluated by a Horner-like nest it is SEVEN FMAs, where the lerp tree is
//: seven lerps = seven subtracts plus seven FMAs. That halves the tree's FP
//: operation count, which the dispatch-stall profile identifies as the binding
//: resource: 47% of cycles are spent waiting on the FP physical register file,
//: and the PRF is exhausted by FP ops in flight rather than by architectural
//: pressure. Fewer FP ops is the only lever that reaches it.
//:
//: It also improves locality, which is a second effect worth separating in the
//: measurement. A corner stencil reads four x-edges at four different (y,z),
//: so up to four cache lines; a coefficient cell is 32 contiguous floats = 128
//: bytes = two lines, always.
//:
//: Costs about 8x the vertex field, and destroys sharing between neighboring
//: cells -- every cell carries its own copy of what the corner form shares.
struct FieldCoeff32 {
    FieldGeometry g;
    AlignedVector<float> v;  // 32 floats per CELL, coefficient-major

    size_t index(int ix, int iy, int iz) const {
        return (((size_t(iz) * size_t(g.nodes_y()) + size_t(iy)) *
                 size_t(g.nodes_x())) +
                size_t(ix)) *
               32;
    }

    const float* node(int ix, int iy, int iz) const {
        return &v[index(ix, iy, iz)];
    }

    size_t zero_offset() const {
        return g.nodes() * 32;
    }

    size_t bytes() const {
        return v.size() * sizeof(float);
    }
};

inline void make_coeff_field(const FieldNaive32& src, FieldCoeff32& out) {
    out.g = src.g;
    const size_t row = size_t(src.g.nodes_x()) * 32;
    const size_t tail = row * size_t(src.g.nodes_y()) + row + 64;
    out.v.assign(src.g.nodes() * 32 + tail, 0.0f);
    const int nx = src.g.nodes_x(), ny = src.g.nodes_y(), nz = src.g.nodes_z();
    for (int iz = 0; iz + 1 < nz; ++iz)
        for (int iy = 0; iy + 1 < ny; ++iy)
            for (int ix = 0; ix + 1 < nx; ++ix) {
                const float* corner_000 = src.node(ix, iy, iz);
                const float* corner_100 = src.node(ix + 1, iy, iz);
                const float* corner_010 = src.node(ix, iy + 1, iz);
                const float* corner_110 = src.node(ix + 1, iy + 1, iz);
                const float* corner_001 = src.node(ix, iy, iz + 1);
                const float* corner_101 = src.node(ix + 1, iy, iz + 1);
                const float* corner_011 = src.node(ix, iy + 1, iz + 1);
                const float* corner_111 = src.node(ix + 1, iy + 1, iz + 1);
                // Named `corner_ijk` rather than `tijk`: the lint rule that
                // forbids ticket ids in identifiers cannot tell a trilinear
                // corner index from a ticket number, and it is right to be
                // strict -- `T36GridGeom` is what the rule exists to stop.
                float* c = &out.v[out.index(ix, iy, iz)];
                for (int k = 0; k < 4; ++k) {
                    c[0 + k] = corner_000[k];
                    c[4 + k] = corner_100[k] - corner_000[k];
                    c[8 + k] = corner_010[k] - corner_000[k];
                    c[12 + k] = corner_001[k] - corner_000[k];
                    c[16 + k] = corner_110[k] - corner_100[k] - corner_010[k] +
                                corner_000[k];
                    c[20 + k] = corner_101[k] - corner_100[k] - corner_001[k] +
                                corner_000[k];
                    c[24 + k] = corner_011[k] - corner_010[k] - corner_001[k] +
                                corner_000[k];
                    c[28 + k] = corner_111[k] - corner_110[k] - corner_101[k] -
                                corner_011[k] + corner_100[k] + corner_010[k] +
                                corner_001[k] - corner_000[k];
                }
            }
}

//: B2: the x-edge field. Every edge stored as its own ALIGNED 32-byte record.
//:
//: This exists to answer one question: is B1 losing because its 32-byte edge
//: loads are misaligned? On the shared vertex field they cannot be made
//: aligned by padding -- a node is the LEFT corner of one edge and the RIGHT
//: corner of the next, so edges start at every node and half of them land on a
//: 16-byte-but-not-32 boundary, with a quarter straddling a cache line.
//: Duplicating the shared node into two records removes the constraint: every
//: record starts at a multiple of 32 from a 64-byte-aligned base.
//:
//: Costs 2x along x (each interior node is stored twice). If B1 was being
//: punished by alignment, this recovers it; if B1 was a wash for another
//: reason, this will be too, and the duplication will cost cache on top.
struct FieldEdge32 {
    FieldGeometry g;
    AlignedVector<float> v;  // 8 floats per edge record: [node(ix) node(ix+1)]

    size_t index(int ix, int iy, int iz) const {
        return (((size_t(iz) * size_t(g.nodes_y()) + size_t(iy)) *
                 size_t(g.nodes_x())) +
                size_t(ix)) *
               8;
    }

    const float* node(int ix, int iy, int iz) const {
        return &v[index(ix, iy, iz)];
    }

    size_t zero_offset() const {
        return g.nodes() * 8;
    }

    size_t bytes() const {
        return v.size() * sizeof(float);
    }
};

inline void make_edge_field(const FieldNaive32& src, FieldEdge32& out) {
    out.g = src.g;
    // The tail must cover a whole stencil reach in the 8-scalar stride.
    const size_t row = size_t(src.g.nodes_x()) * 8;
    const size_t tail = row * size_t(src.g.nodes_y()) + row + 16;
    out.v.assign(src.g.nodes() * 8 + tail, 0.0f);
    const int nx = src.g.nodes_x();
    for (int iz = 0; iz < src.g.nodes_z(); ++iz)
        for (int iy = 0; iy < src.g.nodes_y(); ++iy)
            for (int ix = 0; ix < nx; ++ix) {
                const size_t o = out.index(ix, iy, iz);
                const float* l = src.node(ix, iy, iz);
                for (int c = 0; c < 4; ++c)
                    out.v[o + c] = l[c];
                if (ix + 1 < nx) {
                    const float* r = src.node(ix + 1, iy, iz);
                    for (int c = 0; c < 4; ++c)
                        out.v[o + 4 + c] = r[c];
                }
            }
}

//: The x-AFFINE field: `[base, dx]` split into two planes rather than
//: `[f(x0), f(x1)]` adjacent.
//:
//: Trilinear's x stage is `a + wx*(b - a)` -- a subtract and an FMA. Storing
//: the difference instead of the second endpoint makes it `base + wx*dx`, one
//: FMA, and the interpolation is EXACTLY the same function: this is a change
//: of basis within the cell, not an approximation. Four x lerps per sample
//: pair, so 16 subtractions leave an eight-atom block.
//:
//: Costs 2x the field bytes, which was measured as not binding: per-sample
//: cost is flat across a 13.7x change in footprint, the L1 miss rate is 0.61%,
//: and fp16 storage -- which halves the bytes -- bought nothing. Spending
//: footprint to buy arithmetic is the right direction here, and it is the
//: direction the GPU could not take because its 64 KB L1 made footprint the
//: binding constraint.
//:
//: SPLIT PLANES rather than packed `[base|dx]` records: a packed 32-byte
//: record puts base and dx in the two halves of one vector, and bringing dx
//: down to meet base is a lane crossing -- reintroducing exactly what the
//: (k,k+4) pair schedule removed. Two arrays keep every load in the same lane
//: geometry `corner_pair` already uses.
struct FieldAffine32 {
    FieldGeometry g;
    AlignedVector<float> base;  // f at the node
    AlignedVector<float> dxv;   // f(x+1) - f(x), zero on the last x column

    size_t index(int ix, int iy, int iz) const {
        return (((size_t(iz) * size_t(g.nodes_y()) + size_t(iy)) *
                 size_t(g.nodes_x())) +
                size_t(ix)) *
               4;
    }

    const float* node(int ix, int iy, int iz) const {
        return &base[index(ix, iy, iz)];
    }

    const float* node_dx(int ix, int iy, int iz) const {
        return &dxv[index(ix, iy, iz)];
    }

    size_t zero_offset() const {
        return g.nodes() * 4;
    }

    size_t bytes() const {
        return (base.size() + dxv.size()) * sizeof(float);
    }
};

inline void make_affine(const FieldNaive32& src, FieldAffine32& out) {
    out.g = src.g;
    const size_t n = src.g.nodes() * 4 + src.g.zero_tail_floats();
    out.base.assign(n, 0.0f);
    out.dxv.assign(n, 0.0f);
    const int nx = src.g.nodes_x();
    for (int iz = 0; iz < src.g.nodes_z(); ++iz)
        for (int iy = 0; iy < src.g.nodes_y(); ++iy)
            for (int ix = 0; ix < nx; ++ix) {
                const float* a = src.node(ix, iy, iz);
                const size_t o = out.index(ix, iy, iz);
                for (int c = 0; c < 4; ++c)
                    out.base[o + c] = a[c];
                if (ix + 1 < nx) {
                    const float* bb = src.node(ix + 1, iy, iz);
                    // Formed in the SAME order the lerp would have subtracted
                    // them, so the affine and vertex forms differ by fp32
                    // rounding on the product only, not on the difference.
                    for (int c = 0; c < 4; ++c)
                        out.dxv[o + c] = bb[c] - a[c];
                }
            }
}

//: One trilinear sample, scalar, on the naive layout.
//:
//: Outside the lattice the field is exactly zero -- the CUDA path gets this
//: from `cudaAddressModeBorder` with no software bounds test, and zero is the
//: field's true asymptote rather than a clamp. Here it is an explicit test,
//: which is a real cost the GPU does not pay and is part of what this ticket
//: is measuring.
inline void sample_field(const FieldNaive32& f, float x, float y, float z,
                         float& val, float& gx, float& gy, float& gz) {
    const FieldGeometry& g = f.g;
    const float tx = (x - g.ox) * g.inv_hs;
    const float ty = (y - g.oy) * g.inv_hs;
    const float tz = (z - g.oz) * g.inv_hs;
    const int ix = int(std::floor(tx)), iy = int(std::floor(ty)),
              iz = int(std::floor(tz));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= g.nx || iy >= g.ny || iz >= g.nz) {
        val = gx = gy = gz = 0.0f;
        return;
    }
    const float fx = tx - float(ix), fy = ty - float(iy), fz = tz - float(iz);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int dz = 0; dz < 2; ++dz) {
        const float wz = dz ? fz : (1.0f - fz);
        for (int dy = 0; dy < 2; ++dy) {
            const float wy = dy ? fy : (1.0f - fy);
            for (int dx = 0; dx < 2; ++dx) {
                const float w = wz * wy * (dx ? fx : (1.0f - fx));
                const float* n = f.node(ix + dx, iy + dy, iz + dz);
                acc[0] += w * n[0];
                acc[1] += w * n[1];
                acc[2] += w * n[2];
                acc[3] += w * n[3];
            }
        }
    }
    val = acc[0];
    gx = acc[1];
    gy = acc[2];
    gz = acc[3];
}

}  // namespace cpu
}  // namespace r3

namespace r3 {
namespace cpu {

// --- the field-guided pose search -----------------------------------------
//
// Structurally `optimize_pair` from score.hpp with ONE substitution: the
// O(n_A * n_B) query-atom loop inside `get_gradient` becomes O(n_B) field
// samples. That is the whole of the field idea and the only arithmetic that
// moves (the CUDA doc block says the same of its own kernel).
//
// **The reported score is still the EXACT analytic rescore at the converged
// pose.** Tier 3 is the specification and no approximate tier may be the source
// of a reported number (NUMERICS_POLICY section 2). So the field affects WHICH
// POSE is found and nothing else, which is also what makes recall the right
// acceptance criterion: the error path runs through pose quality, not through
// score error.
inline void field_gradient(const FieldNaive32& f, const Conformer& b,
                           const float m[9], const float t[3], float gnorm,
                           float g[7], float* out_s, OpCensus* census) {
    for (int k = 0; k < 7; ++k)
        g[k] = 0.0f;
    float vs = 0.0f;
    // `gfac = -gnorm`, NOT `+gnorm` (kernels.cu:4234). The stored gradient
    // already carries -2 ln2 from the build; this minus is what makes the
    // field torque agree with the analytic one rather than pointing the other
    // way. See the header comment for the term-by-term identity.
    const float gfac = -gnorm;

    for (int j = 0; j < b.n_atoms; ++j) {
        const float bx = b.xyz[3 * j], by = b.xyz[3 * j + 1],
                    bz = b.xyz[3 * j + 2];
        const float mbx = m[0] * bx + m[1] * by + m[2] * bz;
        const float mby = m[3] * bx + m[4] * by + m[5] * bz;
        const float mbz = m[6] * bx + m[7] * by + m[8] * bz;
        float vs_, fx_, fy_, fz_;
        sample_field(f, mbx + t[0], mby + t[1], mbz + t[2], vs_, fx_, fy_, fz_);
        const float gxv = gfac * fx_;
        const float gyv = gfac * fy_;
        const float gzv = gfac * fz_;
        // C = 2 M b, the space-frame lever arm -- the same one the analytic
        // path forms, and formed from the pre-translation position.
        const float C[3] = {2.f * mbx, 2.f * mby, 2.f * mbz};
        g[1] += C[1] * gzv - C[2] * gyv;
        g[2] += C[2] * gxv - C[0] * gzv;
        g[3] += C[0] * gyv - C[1] * gxv;
        g[4] += gxv;
        g[5] += gyv;
        g[6] += gzv;
        vs += vs_;
        if (census) ++census->field_samples;
    }
    if (out_s) *out_s = vs;
}

//: Templated on the gradient so the scalar and AVX2 samplers share ONE
//: optimizer. Two copies of an Adagrad loop is two chances for them to drift,
//: and the loop is not what either arm is measuring.
template <class Field, class Gradient>
inline PairResult
optimize_pair_field_with(const Conformer& a, const Conformer& b, const Field& f,
                         float s_aa, float s_bb, int start_mode, int n_steps,
                         float lr_q, float lr_t_scaled, OpCensus* census,
                         Gradient grad, bool winner_only = false,
                         bool want_field_score = false) {
    PairResult best;
    best.s_aa = s_aa;
    best.s_bb = s_bb;
    best.tanimoto_shape =
        winner_only ? -std::numeric_limits<float>::infinity() : -1.0f;
    float best_m[9] = {0}, best_t[3] = {0};
    // Explicit, because `best_start` defaults to 0 and every start can be
    // skipped by the signed-permutation check -- in which case best_m is
    // zeros and rescoring it would score a garbage pose.
    bool have_start = false;
    const float gnorm = 1.0f / (s_aa + s_bb);
    const int n_starts = start_count(start_mode);

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

        float m_final[9];
        float s_ab_field = -1.0f;
        for (int step = 0; step <= n_steps; ++step) {
            const float q_conj[4] = {q_start[0], -q_start[1], -q_start[2],
                                     -q_start[3]};
            float q_eff[4];
            quat_mul(q, q_conj, q_eff);
            float m[9];
            quat_to_mat(q_eff, m);
            for (int k = 0; k < 9; ++k)
                m[k] *= COORD_SCALE;

            if (step == n_steps) {
                for (int k = 0; k < 9; ++k)
                    m_final[k] = m[k];
                if (winner_only || want_field_score) {
                    // The FIELD's own value at the converged pose. The loop
                    // otherwise breaks before evaluating anything here, so this
                    // is a real extra field evaluation -- cheap against the
                    // exact O(n_A n_B) rescore it may replace, and NOT free.
                    float gtmp[7];
                    grad(f, b, m, t, gnorm, gtmp, &s_ab_field, census);
                }
                if (!winner_only) {
                    // EXACT analytic rescore at the converged pose, identical
                    // to the analytic path's final step.
                    s_ab = pair_overlap_at_pose(a, b, m, t, census);
                }
                break;
            }
            grad(f, b, m, t, gnorm, g, &s_ab, census);
            adagrad_exp_step(q, t, g, cache, lr_q, lr_t_scaled);
            if (census) ++census->optimizer_steps;
        }

        // Under `winner_only` the exact rescore has not run, so the start is
        // SELECTED on the field's own estimate and only the survivor is
        // rescored, below. That is the whole arm: the rescore is 4x per pair
        // when the reported number needs 1x.
        const float sel_ab = winner_only ? s_ab_field : s_ab;
        const float den = s_aa + s_bb - sel_ab;
        const float ts = (den > 0.0f) ? (sel_ab / den) : 0.0f;
        if (ts > best.tanimoto_shape) {
            best.tanimoto_shape = ts;
            best.s_ab = sel_ab;
            best.s_ab_field = s_ab_field;
            best.best_start = n;
            for (int k = 0; k < 9; ++k)
                best_m[k] = m_final[k];
            for (int k = 0; k < 3; ++k)
                best_t[k] = t[k];
            have_start = true;
            const float q_conj[4] = {q_start[0], -q_start[1], -q_start[2],
                                     -q_start[3]};
            quat_mul(q, q_conj, best.pose_q);
            for (int k = 0; k < 3; ++k)
                best.pose_t[k] = t[k] / COORD_SCALE;
        }
        if (census) ++census->starts;
    }
    if (winner_only && have_start) {
        // ONE exact rescore, on the survivor only. The reported number is still
        // the exact one; what changed is which start it is computed for.
        best.s_ab = pair_overlap_at_pose(a, b, best_m, best_t, census);
        const float den = s_aa + s_bb - best.s_ab;
        best.tanimoto_shape = (den > 0.0f) ? (best.s_ab / den) : 0.0f;
    }
    if (census) ++census->pairs;
    return best;
}

#if defined(__AVX2__)
//: `LANES` pairs advanced in lockstep, so their pose updates batch.
//:
//: Structurally identical to `optimize_pair_field_with` above --
//: same starts, same steps, same rescore placement -- with one difference: the
//: pairs step together, so `adagrad_exp_step` becomes one
//: `batched_adagrad_exp_step` per step instead of `LANES` of them.
//:
//: What does NOT change is the gradient: it is called once per lane per step,
//: exactly as before, because it is already atom-vectorised internally. Only
//: the O(1)-in-atom-count tail batches, which is the part the ablation
//: measured at 14.1% of the search window on Zen 3.
//:
//: `grad` takes a lane index; the caller owns one `ConformerSoA` per lane.
template <int LANES, class Field, class GradN>
inline void optimize_pairs_field_with(
    const Conformer& a, const Conformer* const* b, int n_lanes, const Field& f,
    float s_aa, const float* s_bb, int start_mode, int n_steps, float lr_q,
    float lr_t_scaled, OpCensus* census, PairResult* out, bool winner_only,
    bool want_field_score, GradN grad) {
    float gnorm[LANES], best_m[LANES][9], best_t[LANES][3];
    bool have_start[LANES];
    for (int p = 0; p < n_lanes; ++p) {
        out[p] = PairResult();
        out[p].s_aa = s_aa;
        out[p].s_bb = s_bb[p];
        out[p].tanimoto_shape =
            winner_only ? -std::numeric_limits<float>::infinity() : -1.0f;
        gnorm[p] = 1.0f / (s_aa + s_bb[p]);
        have_start[p] = false;
    }
    const int n_starts = start_count(start_mode);
    BatchedPoseState<LANES> S;

    for (int n = 0; n < n_starts; ++n) {
        float q_start[4];
        start_quaternion(start_mode, n, q_start);
        float r_start[9];
        quat_to_mat(q_start, r_start);
        // Start selection depends only on `start_mode` and `n`, so every lane
        // skips or takes the same starts -- no per-lane divergence to mask.
        if (!matrix_is_signed_permutation(r_start, SIGNED_PERM_TOL)) continue;
        const float q_conj[4] = {q_start[0], -q_start[1], -q_start[2],
                                 -q_start[3]};

        S.reset(n_lanes);
        float s_ab[LANES] = {0}, s_ab_field[LANES], m_final[LANES][9];
        for (int p = 0; p < LANES; ++p)
            s_ab_field[p] = -1.0f;

        for (int step = 0; step <= n_steps; ++step) {
            for (int p = 0; p < n_lanes; ++p) {
                const float qq[4] = {S.q[0][p], S.q[1][p], S.q[2][p],
                                     S.q[3][p]};
                const float tt[3] = {S.t[0][p], S.t[1][p], S.t[2][p]};
                float q_eff[4], m[9];
                quat_mul(qq, q_conj, q_eff);
                quat_to_mat(q_eff, m);
                for (int k = 0; k < 9; ++k)
                    m[k] *= COORD_SCALE;

                if (step == n_steps) {
                    for (int k = 0; k < 9; ++k)
                        m_final[p][k] = m[k];
                    if (winner_only || want_field_score) {
                        float gtmp[7];
                        grad(p, m, tt, gnorm[p], gtmp, &s_ab_field[p], census);
                    }
                    if (!winner_only)
                        s_ab[p] = pair_overlap_at_pose(a, *b[p], m, tt, census);
                    continue;
                }
                float gg[7];
                grad(p, m, tt, gnorm[p], gg, &s_ab[p], census);
                for (int k = 0; k < 7; ++k)
                    S.g[k][p] = gg[k];
            }
            if (step == n_steps) break;
            batched_adagrad_exp_step<LANES>(S, n_lanes, lr_q, lr_t_scaled);
            if (census) census->optimizer_steps += uint64_t(n_lanes);
        }

        for (int p = 0; p < n_lanes; ++p) {
            const float sel = winner_only ? s_ab_field[p] : s_ab[p];
            const float den = s_aa + s_bb[p] - sel;
            const float ts = (den > 0.0f) ? (sel / den) : 0.0f;
            if (ts > out[p].tanimoto_shape) {
                out[p].tanimoto_shape = ts;
                out[p].s_ab = sel;
                out[p].s_ab_field = s_ab_field[p];
                out[p].best_start = n;
                for (int k = 0; k < 9; ++k)
                    best_m[p][k] = m_final[p][k];
                for (int k = 0; k < 3; ++k)
                    best_t[p][k] = S.t[k][p];
                have_start[p] = true;
                const float qq[4] = {S.q[0][p], S.q[1][p], S.q[2][p],
                                     S.q[3][p]};
                quat_mul(qq, q_conj, out[p].pose_q);
                for (int k = 0; k < 3; ++k)
                    out[p].pose_t[k] = S.t[k][p] / COORD_SCALE;
            }
            if (census) ++census->starts;
        }
    }
    for (int p = 0; p < n_lanes; ++p) {
        if (winner_only && have_start[p]) {
            out[p].s_ab =
                pair_overlap_at_pose(a, *b[p], best_m[p], best_t[p], census);
            const float den = s_aa + s_bb[p] - out[p].s_ab;
            out[p].tanimoto_shape = (den > 0.0f) ? (out[p].s_ab / den) : 0.0f;
        }
        if (census) ++census->pairs;
    }
}
#endif  // __AVX2__

//: The scalar field search, which is the correctness reference the vector arm
//: is checked against.
inline PairResult optimize_pair_field(const Conformer& a, const Conformer& b,
                                      const FieldNaive32& f, float s_aa,
                                      float s_bb, int start_mode, int n_steps,
                                      float lr_q, float lr_t_scaled,
                                      OpCensus* census) {
    return optimize_pair_field_with(
        a, b, f, s_aa, s_bb, start_mode, n_steps, lr_q, lr_t_scaled, census,
        [](const FieldNaive32& ff, const Conformer& bb, const float mm[9],
           const float tt[3], float gn, float gg[7], float* os,
           OpCensus* cc) { field_gradient(ff, bb, mm, tt, gn, gg, os, cc); });
}

}  // namespace cpu
}  // namespace r3

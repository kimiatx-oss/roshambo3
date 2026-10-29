// Per-target-type CPU color fields. This is an experimental evaluator;
// it does not populate production score columns or change the analytic path.
//
// ColorFieldBundle (x86 AVX2/FMA/F16C helpers with scalar fallbacks) is the
// four-channel [f, gx, gy, gz] stored-gradient field the CPU combo path can
// route through. It mirrors the CUDA color field
// build (`field_fill_kernel` in _cuda_src/kernels.cu): the same reduced frame,
// the same `fac * INV_TWO_ALPHA` exponent, and the same Laplacian prefilter.
// The original value-only ColorFieldSet experiment is in the paper
// repository.
//
// A field is indexed by the LIBRARY (target) feature type tb and sums every
// query feature that interacts with tb, so sampling F_tb at each library
// feature of type tb replaces the query-feature loop of the analytic pair sum.
// The backend is chosen per target-type plane, once, at query setup;
// a plane that cannot be built faithfully stays on the exact analytic route,
// and a sampled group falls back whole, never partially.
//
// Status: experimental. The shipped default is analytic color; fields
// measured losing to composed analytic color on ordinary queries, with a
// small conditional win on one 24-hydrophobe query. Exact final scoring is
// analytic regardless; the field only steers the pose optimizer. The private
// `R3_COLOR_FIELD_*` compile controls select node/brick arithmetic ablations and
// all default to zero.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "color_view.hpp"
#include "color_field_channels.hpp"
#include "kernel.hpp"

// Per-function ISA targets: the translation unit itself builds for the
// baseline ISA, and only the functions tagged below may use AVX2/FMA (and
// F16C). They are entered only after the runtime CPU probe in the
// ColorFieldBundle constructor. `noinline` keeps the target-specific code out
// of baseline-ISA callers.
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#define R3_CHANNEL_FIELD_X86_TARGETS 1
#define R3_CHANNEL_FIELD_AVX2 __attribute__((target("avx2,fma"), noinline))
#define R3_CHANNEL_FIELD_F16C __attribute__((target("avx2,fma,f16c"), noinline))
#else
#define R3_CHANNEL_FIELD_X86_TARGETS 0
#define R3_CHANNEL_FIELD_AVX2
#define R3_CHANNEL_FIELD_F16C
#endif

namespace py = pybind11;
#ifndef R3_COLOR_FIELD_BRICK
#define R3_COLOR_FIELD_BRICK 0
#endif
#ifndef R3_COLOR_FIELD_NODE
#define R3_COLOR_FIELD_NODE 0
#endif
#ifndef R3_COLOR_FIELD_NODE32_Q
#define R3_COLOR_FIELD_NODE32_Q 0
#endif
#ifndef R3_COLOR_FIELD_BATCH
#define R3_COLOR_FIELD_BATCH 0
#endif
namespace r3 {
namespace cpu {
namespace {
constexpr size_t kMaxGridValues = 50000000;
// Four channels are built in fp32 scratch even when their final storage is
// half nodes or polynomial bricks. Bound the aggregate scratch, not just one
// axis, so a general matrix with many active target types cannot surprise the
// host with an unbounded allocation.
constexpr size_t kMaxChannelValues = 32000000;
constexpr float kLn2 = 0.6931471805599453f;
// 1/(2 alpha), the same constant as ROSHAMBO3_INV_TWO_ALPHA in
// _cuda_src/kernels.cuh: it turns a color exponent into the exp2 coefficient
// of the reduced frame (see the `term.fac` computation below).
constexpr float kInvTwoAlpha = 1.1952087f;
// Process-wide count of bundle constructions, read by the ownership
// diagnostics (`construction_count`).
std::atomic<uint64_t> channel_bundle_constructions{0};

size_t checked_product(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::invalid_argument(
            "color field lattice size overflows size_t");
    return a * b;
}

uint16_t float_to_half(float value) {
    // IEEE-754 binary32 to binary16, round-to-nearest-even. Color fields are
    // nonnegative and finite, but the full cases keep this conversion safe if
    // a future interaction matrix reaches a subnormal or infinity.
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp = (bits >> 23) & 0xffu;
    const uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) {
        if (mant == 0) return uint16_t(sign | 0x7c00u);
        return uint16_t(sign | 0x7e00u | (mant >> 13));
    }
    int e = int(exp) - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);
        const uint32_t full = mant | 0x800000u;
        const int shift = 14 - e;
        uint32_t rounded = full >> shift;
        const uint32_t remainder = full & ((uint32_t(1) << shift) - 1u);
        const uint32_t halfway = uint32_t(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
            ++rounded;
        return uint16_t(sign | rounded);
    }
    uint32_t rounded = mant >> 13;
    const uint32_t remainder = mant & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
        if (rounded == 0x400u) {
            rounded = 0;
            ++e;
            if (e >= 31) return uint16_t(sign | 0x7c00u);
        }
    }
    return uint16_t(sign | (uint32_t(e) << 10) | rounded);
}

// binary16 -> binary32 widening, including subnormals, infinities and NaN.
// The software decode for hosts without F16C. Every finite half value is
// exactly representable in float, so the widening itself never rounds.
float half_to_float(uint16_t half) {
    const uint32_t sign = uint32_t(half & 0x8000u) << 16;
    uint32_t exp = (half >> 10) & 0x1fu;
    uint32_t mant = half & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3ffu;
            bits = sign | (uint32_t(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// One query feature's contribution to one target-type plane, in the reduced
// frame: value = coefficient * exp2(-fac * |u - xyz|^2), with
// beta = ln2 * fac and its powers precomputed for the gradient and the
// Laplacian prefilter terms.
struct ChannelBuildTerm {
    float xyz[3];
    float coefficient;
    float fac;
    float beta;
    float beta2;
    float beta3;
};

// The four stored channels at one lattice node (x, y, z), in fp32:
//
//   out[0]    = F - c lap F                     prefiltered value
//   out[1..3] = dF/du - c d(lap F)/du           prefiltered gradient
//
// with F = sum_terms coefficient * exp2(-fac |u - a|^2) and c = h^2/12. This
// is the CUDA `field_fill_kernel` formula term for term, with beta = ln2*fac
// in place of ln2. Per kernels.cu, the prefilter makes the signed value
// bias fourth order; gradients are stored, never differenced.
void build_node_scalar(const std::vector<ChannelBuildTerm>& terms, float x,
                       float y, float z, float c, float out[4]) {
    float v[4] = {}, lap[4] = {};
    for (const ChannelBuildTerm& term : terms) {
        const float dx = x - term.xyz[0];
        const float dy = y - term.xyz[1];
        const float dz = z - term.xyz[2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        const float e = term.coefficient * ::exp2f(-term.fac * d2);
        const float cg = -2.0f * term.beta * e;
        const float cl = e * (20.0f * term.beta2 - 8.0f * term.beta3 * d2);
        v[0] += e;
        lap[0] += e * (4.0f * term.beta2 * d2 - 6.0f * term.beta);
        v[1] += cg * dx;
        v[2] += cg * dy;
        v[3] += cg * dz;
        lap[1] += cl * dx;
        lap[2] += cl * dy;
        lap[3] += cl * dz;
    }
    for (int k = 0; k < 4; ++k)
        out[k] = v[k] - c * lap[k];
}

#if R3_CHANNEL_FIELD_X86_TARGETS
// Channel-SIMD helpers: one XMM holds one node's four channels [f,gx,gy,gz],
// so a whole node is interpolated per instruction (SIMD across four
// channels, not eight-target AoS gathers).
//
// `a + t (b - a)` across all four channels.
__attribute__((target("avx2,fma"), always_inline)) inline __m128
channel_lerp(__m128 a, __m128 b, __m128 t) {
    return _mm_fmadd_ps(t, _mm_sub_ps(b, a), a);
}

// `[node a | node b]`: two fp32 nodes, one per 128-bit half of a YMM.
__attribute__((target("avx2,fma"), always_inline)) inline __m256
channel_pair(const float* a, const float* b) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(a)),
                                _mm_loadu_ps(b), 1);
}

// The half-node version: two 8-byte half4 loads packed into one XMM, then a
// single `vcvtph2ps` widens both nodes into a YMM.
__attribute__((target("avx2,fma,f16c"), always_inline)) inline __m256
channel_half_pair(const uint16_t* a, const uint16_t* b) {
    const __m128i lo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a));
    const __m128i hi = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b));
    return _mm256_cvtph_ps(_mm_unpacklo_epi64(lo, hi));
}

// build_node_scalar with the four channel accumulators held in one XMM each
// (value and Laplacian). The per-term scalar expressions are unchanged; only
// the four channel sums are vectorized.
R3_CHANNEL_FIELD_AVX2 void
build_node_avx2(const std::vector<ChannelBuildTerm>& terms, float x, float y,
                float z, float c, float out[4]) {
    __m128 value = _mm_setzero_ps();
    __m128 lap = _mm_setzero_ps();
    for (const ChannelBuildTerm& term : terms) {
        const float dx = x - term.xyz[0];
        const float dy = y - term.xyz[1];
        const float dz = z - term.xyz[2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        const float e = term.coefficient * ::exp2f(-term.fac * d2);
        const float cg = -2.0f * term.beta * e;
        const float cl = e * (20.0f * term.beta2 - 8.0f * term.beta3 * d2);
        // _mm_set_ps lists lanes high to low: lane 0 = value, 1..3 = x/y/z.
        const __m128 v = _mm_set_ps(cg * dz, cg * dy, cg * dx, e);
        const __m128 l =
            _mm_set_ps(cl * dz, cl * dy, cl * dx,
                       e * (4.0f * term.beta2 * d2 - 6.0f * term.beta));
        value = _mm_add_ps(value, v);
        lap = _mm_add_ps(lap, l);
    }
    _mm_storeu_ps(out, _mm_sub_ps(value, _mm_mul_ps(_mm_set1_ps(c), lap)));
}

// Trilinear interpolation of one node-stored sample: all four channels of the
// eight cell corners, given the cell's base node (ix, iy, iz) and the
// per-axis weights w[0] = 1 - t, w[1] = t. Node layout is AoS4,
// x fastest: node (ix, iy, iz) starts at ((iz * ny + iy) * nx + ix) * 4.
// `half` selects the software-decoded fp16 payload (no F16C).
//
// The default (R3_COLOR_FIELD_NODE == 0) is a weighted sum: eight FMAs of each
// corner by the product weight wx*wy*wz. The private controls replace it with a
// lerp tree: NODE == 1 is seven XMM lerps (x, then y, then z); NODE == 2
// (fp32 nodes here; the F16C function below has its own) pairs the z = 0 and
// z = 1 corners in the two halves of a YMM so the x and y lerps cover both z
// planes at once, with the z lerp last. These are different roundings of the
// same trilinear interpolant.
__attribute__((target("avx2,fma"), always_inline)) inline void
interpolate_node_avx2_inline(const float* fp32, const uint16_t* fp16, bool half,
                             size_t nx, size_t ny, size_t ix, size_t iy,
                             size_t iz, const float wx[2], const float wy[2],
                             const float wz[2], float out[4]) {
#if R3_COLOR_FIELD_NODE == 1
    const size_t base = ((iz * ny + iy) * nx + ix) * 4;
    const size_t row = nx * 4, plane = ny * row;
    __m128 corner[8];
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const size_t offset = base + size_t(dz) * plane +
                                      size_t(dy) * row + size_t(dx) * 4;
                if (half) {
                    float v[4];
                    for (int ch = 0; ch < 4; ++ch)
                        v[ch] = half_to_float(fp16[offset + ch]);
                    corner[dz * 4 + dy * 2 + dx] = _mm_loadu_ps(v);
                } else
                    corner[dz * 4 + dy * 2 + dx] = _mm_loadu_ps(fp32 + offset);
            }
    const __m128 x = _mm_set1_ps(wx[1]), y = _mm_set1_ps(wy[1]),
                 z = _mm_set1_ps(wz[1]);
    const __m128 a = channel_lerp(corner[0], corner[1], x),
                 b = channel_lerp(corner[2], corner[3], x);
    const __m128 c = channel_lerp(corner[4], corner[5], x),
                 d = channel_lerp(corner[6], corner[7], x);
    _mm_storeu_ps(
        out, channel_lerp(channel_lerp(a, b, y), channel_lerp(c, d, y), z));
    return;
#elif R3_COLOR_FIELD_NODE == 2
    if (!half) {
        const float* p = fp32 + ((iz * ny + iy) * nx + ix) * 4;
        const size_t row = nx * 4, plane = ny * row;
        const __m256 a = channel_pair(p, p + plane),
                     b = channel_pair(p + 4, p + plane + 4);
        const __m256 c = channel_pair(p + row, p + plane + row),
                     d = channel_pair(p + row + 4, p + plane + row + 4);
        const __m256 x = _mm256_set1_ps(wx[1]), y = _mm256_set1_ps(wy[1]);
        const __m256 low = _mm256_fmadd_ps(x, _mm256_sub_ps(b, a), a);
        const __m256 high = _mm256_fmadd_ps(x, _mm256_sub_ps(d, c), c);
        const __m256 yz = _mm256_fmadd_ps(y, _mm256_sub_ps(high, low), low);
        const __m128 l = _mm256_castps256_ps128(yz),
                     h = _mm256_extractf128_ps(yz, 1);
        _mm_storeu_ps(out,
                      _mm_fmadd_ps(_mm_set1_ps(wz[1]), _mm_sub_ps(h, l), l));
        return;
    }
#endif
    // Default: corners accumulated in z, y, x order with product weights.
    __m128 sum = _mm_setzero_ps();
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const size_t index =
                    ((iz + size_t(dz)) * ny + iy + size_t(dy)) * nx + ix +
                    size_t(dx);
                const float weight = wx[dx] * wy[dy] * wz[dz];
                __m128 node;
                if (half) {
                    const uint16_t* p = fp16 + index * 4;
                    float channels[4] = {
                        half_to_float(p[0]), half_to_float(p[1]),
                        half_to_float(p[2]), half_to_float(p[3])};
                    node = _mm_loadu_ps(channels);
                } else {
                    node = _mm_loadu_ps(fp32 + index * 4);
                }
                sum = _mm_fmadd_ps(node, _mm_set1_ps(weight), sum);
            }
        }
    }
    _mm_storeu_ps(out, sum);
}

// Keep the F16C intrinsic in a function whose target explicitly requires it.
// The ordinary AVX2 interpolation function above remains compilable for
// x86-64-v2 builds, and runtime dispatch enters this arm only after F16C is
// detected.
__attribute__((target("avx2,fma,f16c"), always_inline)) inline void
interpolate_node_avx2_f16c_inline(const uint16_t* fp16, size_t nx, size_t ny,
                                  size_t ix, size_t iy, size_t iz,
                                  const float wx[2], const float wy[2],
                                  const float wz[2], float out[4]) {
#if R3_COLOR_FIELD_NODE == 2
    const uint16_t* p = fp16 + ((iz * ny + iy) * nx + ix) * 4;
    const size_t row = nx * 4, plane = ny * row;
    const __m256 a = channel_half_pair(p, p + plane),
                 b = channel_half_pair(p + 4, p + plane + 4);
    const __m256 c = channel_half_pair(p + row, p + plane + row),
                 d = channel_half_pair(p + row + 4, p + plane + row + 4);
    const __m256 x = _mm256_set1_ps(wx[1]), y = _mm256_set1_ps(wy[1]);
    const __m256 low = _mm256_fmadd_ps(x, _mm256_sub_ps(b, a), a);
    const __m256 high = _mm256_fmadd_ps(x, _mm256_sub_ps(d, c), c);
    const __m256 yz = _mm256_fmadd_ps(y, _mm256_sub_ps(high, low), low);
    const __m128 l = _mm256_castps256_ps128(yz),
                 h = _mm256_extractf128_ps(yz, 1);
    _mm_storeu_ps(out, _mm_fmadd_ps(_mm_set1_ps(wz[1]), _mm_sub_ps(h, l), l));
    return;
#elif R3_COLOR_FIELD_NODE == 1
    // Same seven-XMM-lerp control, with one independent half4 conversion per corner.
    __m128 corner[8];
    const uint16_t* p = fp16 + ((iz * ny + iy) * nx + ix) * 4;
    const size_t row = nx * 4, plane = ny * row;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
                corner[dz * 4 + dy * 2 + dx] = _mm_cvtph_ps(_mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(p + size_t(dz) * plane +
                                                     size_t(dy) * row +
                                                     size_t(dx) * 4)));
    const __m128 x = _mm_set1_ps(wx[1]), y = _mm_set1_ps(wy[1]),
                 z = _mm_set1_ps(wz[1]);
    const __m128 a = channel_lerp(corner[0], corner[1], x),
                 b = channel_lerp(corner[2], corner[3], x);
    const __m128 c = channel_lerp(corner[4], corner[5], x),
                 d = channel_lerp(corner[6], corner[7], x);
    _mm_storeu_ps(
        out, channel_lerp(channel_lerp(a, b, y), channel_lerp(c, d, y), z));
    return;
#endif
    __m128 sum = _mm_setzero_ps();
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const size_t index =
                    ((iz + size_t(dz)) * ny + iy + size_t(dy)) * nx + ix +
                    size_t(dx);
                const uint16_t* p = fp16 + index * 4;
                const __m128i h =
                    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
                const __m128 node = _mm_cvtph_ps(h);
                const float weight = wx[dx] * wy[dy] * wz[dz];
                sum = _mm_fmadd_ps(node, _mm_set1_ps(weight), sum);
            }
        }
    }
    _mm_storeu_ps(out, sum);
}

// Polynomial-brick sample. Each cell stores 8 basis coefficients x 4
// channels = 32 floats, so the trilinear interpolant is evaluated as
//
//   c1 + x cx + y cy + z cz + xy cxy + xz cxz + yz cyz + xyz cxyz
//
// with (x, y, z) the in-cell fractions (see `brick_coeff`). The default
// multiplies out the eight monomials; BRICK == 1 is a seven-FMA Horner
// form, and BRICK == 2 is a packed layout (slots {1,z | x,xz | y,yz |
// xy,xyz}) that needs three YMM FMAs plus one XMM FMA.
__attribute__((target("avx2,fma"), always_inline)) inline void
interpolate_brick_avx2_inline(const float* brick, size_t cell, float x, float y,
                              float z, float out[4]) {
#if R3_COLOR_FIELD_BRICK == 1
    const float* c = brick + cell * 32;
    const __m128 vx = _mm_set1_ps(x), vy = _mm_set1_ps(y), vz = _mm_set1_ps(z);
    const __m128 a = _mm_fmadd_ps(vx, _mm_loadu_ps(c + 4), _mm_loadu_ps(c));
    const __m128 b =
        _mm_fmadd_ps(vx, _mm_loadu_ps(c + 16), _mm_loadu_ps(c + 8));
    const __m128 d =
        _mm_fmadd_ps(vx, _mm_loadu_ps(c + 20), _mm_loadu_ps(c + 12));
    const __m128 e =
        _mm_fmadd_ps(vx, _mm_loadu_ps(c + 28), _mm_loadu_ps(c + 24));
    const __m128 l = _mm_fmadd_ps(vy, b, a), h = _mm_fmadd_ps(vy, e, d);
    _mm_storeu_ps(out, _mm_fmadd_ps(vz, h, l));
#elif R3_COLOR_FIELD_BRICK == 2
    const float* c = brick + cell * 32;
    const __m256 vx = _mm256_set1_ps(x), vy = _mm256_set1_ps(y);
    const __m256 a =
        _mm256_fmadd_ps(vx, _mm256_loadu_ps(c + 8), _mm256_loadu_ps(c));
    const __m256 b =
        _mm256_fmadd_ps(vx, _mm256_loadu_ps(c + 24), _mm256_loadu_ps(c + 16));
    const __m256 yz = _mm256_fmadd_ps(vy, b, a);
    _mm_storeu_ps(out,
                  _mm_fmadd_ps(_mm_set1_ps(z), _mm256_extractf128_ps(yz, 1),
                               _mm256_castps256_ps128(yz)));
#else
    const float basis[8] = {1.0f, x, y, z, x * y, x * z, y * z, x * y * z};
    __m128 sum = _mm_setzero_ps();
    const float* coeff = brick + cell * 32;
    for (int k = 0; k < 8; ++k)
        sum = _mm_fmadd_ps(_mm_loadu_ps(coeff + size_t(k) * 4),
                           _mm_set1_ps(basis[k]), sum);
    _mm_storeu_ps(out, sum);
#endif
}

// Out-of-line, target-attributed entry points for the inline kernels above,
// callable from baseline-ISA code after the runtime feature check.
R3_CHANNEL_FIELD_AVX2 void
interpolate_node_avx2(const float* fp32, const uint16_t* fp16, bool half,
                      size_t nx, size_t ny, size_t ix, size_t iy, size_t iz,
                      const float wx[2], const float wy[2], const float wz[2],
                      float out[4]) {
    interpolate_node_avx2_inline(fp32, fp16, half, nx, ny, ix, iy, iz, wx, wy,
                                 wz, out);
}

R3_CHANNEL_FIELD_F16C void
interpolate_node_avx2_f16c(const uint16_t* fp16, size_t nx, size_t ny,
                           size_t ix, size_t iy, size_t iz, const float wx[2],
                           const float wy[2], const float wz[2], float out[4]) {
    interpolate_node_avx2_f16c_inline(fp16, nx, ny, ix, iy, iz, wx, wy, wz,
                                      out);
}

R3_CHANNEL_FIELD_AVX2 void interpolate_brick_avx2(const float* brick,
                                                  size_t cell, float x, float y,
                                                  float z, float out[4]) {
    interpolate_brick_avx2_inline(brick, cell, x, y, z, out);
}
#endif

void build_node(const std::vector<ChannelBuildTerm>& terms, float x, float y,
                float z, float c, bool use_avx2, float out[4]) {
#if R3_CHANNEL_FIELD_X86_TARGETS
    if (use_avx2) {
        build_node_avx2(terms, x, y, z, c, out);
        return;
    }
#else
    (void)use_avx2;
#endif
    build_node_scalar(terms, x, y, z, c, out);
}

// Linear node (or brick-cell) index, x fastest. Multiply by 4 for the AoS4
// float offset of a node, or by 32 for a brick cell.
size_t node_index(size_t nx, size_t ny, size_t ix, size_t iy, size_t iz) {
    return (iz * ny + iy) * nx + ix;
}

// Map a group of library points to (base cell, in-cell fraction) per axis,
// ahead of sampling. `points` is strided: point j's axis k is at
// points[j * point_stride + k * coordinate_stride], which serves both AoS and
// SoA callers. `Reduced` says the points are already in the reduced frame;
// otherwise they are angstrom and are scaled by COORD_SCALE here.
//
// Returns false if ANY point is outside its plane's lattice; the caller must
// then evaluate the whole group analytically and consume none of `w`.
// Nonfinite coordinates throw rather than fall back. A point exactly on the
// upper face is clamped into the last cell with fraction 1.
template <bool Reduced>
bool prepare_channel_group(const float* points, size_t count,
                           size_t point_stride, size_t coordinate_stride,
                           const float origin[3], const int cells[3],
                           float inv_h, ColorFieldSampleWorkspace& w) {
    w.prepared.resize(count);
    bool all_inside = true;
    for (size_t j = 0; j < count; ++j) {
        auto& p = w.prepared[j];
        for (int k = 0; k < 3; ++k) {
            const float x =
                points[j * point_stride + size_t(k) * coordinate_stride];
            if (!std::isfinite(x))
                throw std::invalid_argument(
                    "library coordinates must be finite");
            const float u = Reduced ? x : COORD_SCALE * x;
            const float f = (u - origin[k]) * inv_h;
            if (!std::isfinite(f) || f < 0 || f > float(cells[k])) {
                all_inside = false;
                // Keep scanning: later points must still pass the finite
                // check above.
                continue;
            }
            int i = int(std::floor(f));
            p.fraction[k] = f - float(i);
            if (i >= cells[k]) {
                i = cells[k] - 1;
                p.fraction[k] = 1;
            }
            p.base[k] = size_t(i);
        }
    }
    return all_inside;
}

// Coefficient `basis` of the trilinear interpolant over one cell, written in
// the monomial basis {1, x, y, z, xy, xz, yz, xyz} of the in-cell fractions,
// from the eight corner nodes v<dx><dy><dz> (all four channels at once). It
// is the same interpolant as the node path, rearranged so a sample is a
// polynomial evaluation instead of eight corner loads.
std::array<float, 4>
brick_coeff(const std::array<float, 4>& v000, const std::array<float, 4>& v100,
            const std::array<float, 4>& v010, const std::array<float, 4>& v110,
            const std::array<float, 4>& v001, const std::array<float, 4>& v101,
            const std::array<float, 4>& v011, const std::array<float, 4>& v111,
            int basis) {
    std::array<float, 4> out{};
    for (int ch = 0; ch < 4; ++ch) {
        switch (basis) {
        case 0:
            out[ch] = v000[ch];
            break;
        case 1:
            out[ch] = v100[ch] - v000[ch];
            break;
        case 2:
            out[ch] = v010[ch] - v000[ch];
            break;
        case 3:
            out[ch] = v001[ch] - v000[ch];
            break;
        case 4:
            out[ch] = v110[ch] - v100[ch] - v010[ch] + v000[ch];
            break;
        case 5:
            out[ch] = v101[ch] - v100[ch] - v001[ch] + v000[ch];
            break;
        case 6:
            out[ch] = v011[ch] - v010[ch] - v001[ch] + v000[ch];
            break;
        default:
            out[ch] = v111[ch] - v110[ch] - v101[ch] - v011[ch] + v100[ch] +
                      v010[ch] + v001[ch] - v000[ch];
            break;
        }
    }
    return out;
}

}  // namespace

// ---- ColorFieldBundle: four-channel stored-gradient fields ----------------

size_t ColorFieldBundle::Plane::bytes() const {
    return node_fp32.size() * sizeof(float) +
           node_fp16.size() * sizeof(uint16_t) +
           brick_fp32.size() * sizeof(float);
}

size_t ColorFieldBundle::cell_index(int query_type, int target_type) const {
    return size_t(query_type) * size_t(n_features_) + size_t(target_type);
}

// Query-scoped construction. For each target type tb with a non-analytic
// backend code, the constructor:
//   1. collects the query features that interact with tb as reduced-frame
//      exp2 terms (`ChannelBuildTerm`);
//   2. plans a lattice over just those features' bounding box plus padding
//      (so each plane has its own geometry);
//   3. builds the fp32 four-channel nodes, then optionally rounds them to
//      half and either keeps half nodes or packs fp32 brick coefficients.
// Any failure at any step marks only that plane inactive, records why in
// `inactive_reason`, and leaves it on the exact analytic route:
//   1 analytic requested, 2 no interacting query feature, 3 a term not
//   finite in fp32, 4 lattice geometry not representable, 5 over the size
//   or aggregate channel budget, 6 a nonfinite channel, 7 a half overflow.
// Build and pack times are recorded separately per plane, so construction
// cost is reported apart from alignment.
ColorFieldBundle::ColorFieldBundle(const std::vector<ColorFieldSource>& sources,
                                   int n_features, const double* coefficient,
                                   const double* decay,
                                   const int* backend_codes,
                                   double spacing_angstrom,
                                   double padding_angstrom)
    : n_features_(n_features), sources_(sources) {
    const auto preparation_started = std::chrono::steady_clock::now();
    generation_ =
        channel_bundle_constructions.fetch_add(1, std::memory_order_relaxed) +
        1;
    if (n_features_ < 1 || n_features_ > MAX_COLOR_FEATURES ||
        coefficient == nullptr || decay == nullptr || backend_codes == nullptr)
        throw std::invalid_argument(
            "invalid color field matrix or feature count");
    if (backend_codes[0] != COLOR_FIELD_ANALYTIC)
        throw std::invalid_argument(
            "color field target type zero is reserved and must use backend 0");
    if (!std::isfinite(spacing_angstrom) || spacing_angstrom <= 0.0 ||
        !std::isfinite(padding_angstrom) || padding_angstrom < 0.0)
        throw std::invalid_argument(
            "field spacing must be positive and padding nonnegative");

    // Probe once; every sample dispatches on these cached flags. The private
    // FORCE controls exercise the scalar and no-F16C fallbacks on capable
    // hosts.
#if R3_CHANNEL_FIELD_X86_TARGETS && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    avx2_fma_supported_ =
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    f16c_supported_ = avx2_fma_supported_ && __builtin_cpu_supports("f16c");
#if R3_COLOR_FIELD_FORCE_SCALAR
    avx2_fma_supported_ = false;
    f16c_supported_ = false;
#elif R3_COLOR_FIELD_FORCE_NO_F16C
    f16c_supported_ = false;
#endif
#endif

    coefficient_.resize(size_t(n_features_) * size_t(n_features_));
    decay_.resize(size_t(n_features_) * size_t(n_features_));
    for (int ta = 0; ta < n_features_; ++ta) {
        for (int tb = 0; tb < n_features_; ++tb) {
            const size_t index = cell_index(ta, tb);
            const double c = coefficient[index], d = decay[index];
            if (!std::isfinite(c) || c < 0.0 || !std::isfinite(d) || d > 0.0)
                throw std::invalid_argument(
                    "color coefficients must be finite/nonnegative and decays finite/nonpositive");
            coefficient_[index] = c;
            decay_[index] = d;
        }
    }
    for (const ColorFieldSource& source : sources_) {
        if (source.type <= 0 || source.type >= n_features_)
            throw std::invalid_argument(
                "query feature types must be in [1,n_features)");
        for (float x : source.xyz)
            if (!std::isfinite(x))
                throw std::invalid_argument("query coordinates must be finite");
    }

    planes_.resize(size_t(n_features_));
    size_t aggregate_channel_values = 0;
    const double scale = double(COORD_SCALE);
    const double hs = spacing_angstrom * scale;
    const double pad = padding_angstrom * scale;
    if (!std::isfinite(hs) || hs <= 0.0 || !std::isfinite(pad))
        throw std::invalid_argument("scaled color field geometry is invalid");

    for (int tb = 1; tb < n_features_; ++tb) {
        Plane& plane = planes_[size_t(tb)];
        plane.backend = backend_codes[tb];
        if (plane.backend < COLOR_FIELD_ANALYTIC ||
            plane.backend > COLOR_FIELD_BRICK_FP32)
            throw std::invalid_argument("unknown color field backend code");
        if (plane.backend == COLOR_FIELD_ANALYTIC) {
            plane.inactive_reason = 1;
            continue;
        }

        const auto plane_start = std::chrono::steady_clock::now();
        std::vector<ChannelBuildTerm> terms;
        terms.reserve(sources_.size());
        double lo[3] = {std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
        double hi[3] = {-std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity()};
        for (const ColorFieldSource& source : sources_) {
            const size_t index = cell_index(source.type, tb);
            const double amplitude = coefficient_[index];
            const double source_decay = decay_[index];
            if (amplitude == 0.0 || source_decay == 0.0) continue;
            ChannelBuildTerm term{};
            for (int k = 0; k < 3; ++k) {
                term.xyz[k] = float(double(source.xyz[k]) * scale);
                lo[k] = std::min(lo[k], double(source.xyz[k]) * scale);
                hi[k] = std::max(hi[k], double(source.xyz[k]) * scale);
            }
            term.coefficient = float(amplitude);
            // In the reduced frame, exp(decay*|x-y|^2) is
            // exp2(-fac*|u-v|^2), where u=COORD_SCALE*x.
            term.fac = float(-2.0 * source_decay * double(kInvTwoAlpha));
            term.beta = kLn2 * term.fac;
            term.beta2 = term.beta * term.beta;
            term.beta3 = term.beta2 * term.beta;
            if (!std::isfinite(term.coefficient) || !std::isfinite(term.fac) ||
                !std::isfinite(term.beta3)) {
                terms.clear();
                plane.inactive_reason = 3;
                break;  // The selected representation cannot encode this plane.
            }
            terms.push_back(term);
        }
        if (terms.empty()) {
            if (!plane.inactive_reason) plane.inactive_reason = 2;
            continue;
        }

        bool geometry_ok = true;
        for (int k = 0; k < 3; ++k) {
            const double origin = lo[k] - pad;
            const double dimension = std::ceil((hi[k] + pad - origin) / hs);
            if (!std::isfinite(dimension) || dimension < 1.0 ||
                dimension > double(std::numeric_limits<int>::max() - 1)) {
                geometry_ok = false;
                break;
            }
            plane.geometry.origin[k] = float(origin);
            plane.geometry.cells[k] = std::max(1, int(dimension));
        }
        plane.geometry.h = float(hs);
        plane.geometry.inv_h = 1.0f / plane.geometry.h;
        if (!geometry_ok || !std::isfinite(plane.geometry.h) ||
            !std::isfinite(plane.geometry.inv_h) || plane.geometry.h <= 0.0f) {
            plane.inactive_reason = 4;
            continue;
        }

        size_t nodes;
        try {
            nodes = checked_product(
                checked_product(size_t(plane.geometry.cells[0]) + 1,
                                size_t(plane.geometry.cells[1]) + 1),
                size_t(plane.geometry.cells[2]) + 1);
        } catch (const std::invalid_argument&) {
            plane.inactive_reason = 5;
            continue;  // Too large for a field: retain the analytic fallback.
        }
        const size_t channel_values = checked_product(nodes, size_t(4));
        // Planes are admitted in target-type order against one aggregate
        // budget; a plane that would exceed it stays analytic.
        if (channel_values > kMaxChannelValues - aggregate_channel_values) {
            plane.inactive_reason = 5;
            continue;  // Per-plane analytic fallback for oversized selections.
        }
        aggregate_channel_values += channel_values;

        const size_t nx = size_t(plane.geometry.cells[0]) + 1;
        const size_t ny = size_t(plane.geometry.cells[1]) + 1;
        const size_t nz = size_t(plane.geometry.cells[2]) + 1;
        std::vector<float> raw(channel_values);
        peak_payload_bytes_ =
            std::max(peak_payload_bytes_,
                     field_bytes() + raw.capacity() * sizeof(float));
        // c = h^2/12, as in the CUDA build and the shape field (field.hpp).
        const float prefilter = plane.geometry.h * plane.geometry.h / 12.0f;
        bool finite_channels = true;
        for (size_t iz = 0; iz < nz && finite_channels; ++iz) {
            const float z =
                plane.geometry.origin[2] + float(iz) * plane.geometry.h;
            for (size_t iy = 0; iy < ny && finite_channels; ++iy) {
                const float y =
                    plane.geometry.origin[1] + float(iy) * plane.geometry.h;
                for (size_t ix = 0; ix < nx; ++ix) {
                    const float x =
                        plane.geometry.origin[0] + float(ix) * plane.geometry.h;
                    float channels[4];
                    build_node(terms, x, y, z, prefilter, avx2_fma_supported_,
                               channels);
                    for (float channel : channels)
                        finite_channels &= std::isfinite(channel);
                    if (!finite_channels) break;
                    const size_t index = node_index(nx, ny, ix, iy, iz) * 4;
                    std::copy(channels, channels + 4, raw.begin() + index);
                }
            }
        }
        if (!finite_channels) {
            plane.inactive_reason = 6;
            // Unsupported magnitudes retain the exact analytic route. Keep
            // the attempted construction cost, but do not count its scratch
            // against later selected planes or expose a partial field.
            plane.build_s = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - plane_start)
                                .count();
            build_seconds_ += plane.build_s;
            aggregate_channel_values -= channel_values;
            continue;
        }
        plane.active = true;
        plane.build_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - plane_start)
                            .count();
        build_seconds_ += plane.build_s;

        if (plane.backend == COLOR_FIELD_NODE_FP32 && !R3_COLOR_FIELD_NODE32_Q) {
            plane.node_fp32.swap(raw);
            continue;
        }

        // Every other backend starts from half-rounded nodes: fp16 nodes keep
        // them, bricks pack them (as the reviewed brick primitive did), and
        // the private NODE32_Q control widens them back to fp32 so storage and
        // source quantization can be compared separately.

        const auto pack_start = std::chrono::steady_clock::now();
        ++plane.pack_count;
        std::vector<uint16_t> half_nodes(channel_values);
        peak_payload_bytes_ =
            std::max(peak_payload_bytes_,
                     field_bytes() + raw.capacity() * sizeof(float) +
                         half_nodes.capacity() * sizeof(uint16_t));
        bool half_overflow = false;
        for (size_t i = 0; i < channel_values; ++i) {
            half_nodes[i] = float_to_half(raw[i]);
            if (!std::isfinite(half_to_float(half_nodes[i]))) {
                half_overflow = true;
                break;
            }
        }
        if (half_overflow) {
            plane.inactive_reason = 7;
            // A half plane whose channels exceed binary16 range is not a
            // partially valid field. Retain the exact analytic route instead.
            plane.active = false;
            plane.pack_s = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - pack_start)
                               .count();
            aggregate_channel_values -= channel_values;
            continue;
        }
        if (plane.backend == COLOR_FIELD_NODE_FP32) {
            plane.node_fp32.resize(channel_values);
            peak_payload_bytes_ =
                std::max(peak_payload_bytes_,
                         field_bytes() + raw.capacity() * sizeof(float) +
                             half_nodes.capacity() * sizeof(uint16_t));
            for (size_t i = 0; i < channel_values; ++i)
                plane.node_fp32[i] = half_to_float(half_nodes[i]);
        } else if (plane.backend == COLOR_FIELD_NODE_FP16) {
            plane.node_fp16.swap(half_nodes);
        } else {
            // Match the reviewed brick primitive: decode its half4
            // stored-gradient samples, then pack fp32 cell coefficients.
            // The brick payload is fp32, but inherits half-node quantization.
            const size_t cells = checked_product(
                checked_product(size_t(plane.geometry.cells[0]),
                                size_t(plane.geometry.cells[1])),
                size_t(plane.geometry.cells[2]));
            const size_t brick_values = checked_product(cells, size_t(32));
            plane.brick_fp32.resize(brick_values);
            peak_payload_bytes_ =
                std::max(peak_payload_bytes_,
                         field_bytes() + raw.capacity() * sizeof(float) +
                             half_nodes.capacity() * sizeof(uint16_t));
            for (size_t iz = 0; iz < size_t(plane.geometry.cells[2]); ++iz) {
                for (size_t iy = 0; iy < size_t(plane.geometry.cells[1]);
                     ++iy) {
                    for (size_t ix = 0; ix < size_t(plane.geometry.cells[0]);
                         ++ix) {
                        const auto read = [&](size_t dx, size_t dy, size_t dz) {
                            std::array<float, 4> value{};
                            const size_t index =
                                node_index(nx, ny, ix + dx, iy + dy, iz + dz) *
                                4;
                            for (int ch = 0; ch < 4; ++ch)
                                value[ch] = half_to_float(
                                    half_nodes[index + size_t(ch)]);
                            return value;
                        };
                        const auto v000 = read(0, 0, 0), v100 = read(1, 0, 0);
                        const auto v010 = read(0, 1, 0), v110 = read(1, 1, 0);
                        const auto v001 = read(0, 0, 1), v101 = read(1, 0, 1);
                        const auto v011 = read(0, 1, 1), v111 = read(1, 1, 1);
                        const size_t cell = node_index(
                            size_t(plane.geometry.cells[0]),
                            size_t(plane.geometry.cells[1]), ix, iy, iz);
                        // Slot permutation for BRICK == 2: pairs the
                        // monomials {1,z}, {x,xz}, {y,yz}, {xy,xyz} so each
                        // pair fills one YMM (interpolate_brick_avx2_inline).
                        for (int basis = 0; basis < 8; ++basis) {
                            const auto coeff =
                                brick_coeff(v000, v100, v010, v110, v001, v101,
                                            v011, v111, basis);
                            std::copy(
                                coeff.begin(), coeff.end(),
                                plane.brick_fp32.begin() + cell * 32 +
                                    size_t(R3_COLOR_FIELD_BRICK == 2
                                               ? std::array<int, 8>{0, 2, 4, 1,
                                                                    6, 3, 5,
                                                                    7}[basis]
                                               : basis) *
                                        4);
                        }
                    }
                }
            }
        }
        plane.pack_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - pack_start)
                           .count();
    }
    preparation_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      preparation_started)
            .count();
}

uint64_t ColorFieldBundle::construction_count() {
    return channel_bundle_constructions.load(std::memory_order_relaxed);
}

int ColorFieldBundle::coefficient_layout_version() {
    return R3_COLOR_FIELD_BRICK == 2 ? 2 : 1;
}

std::vector<int> ColorFieldBundle::pack_counts() const {
    std::vector<int> out;
    for (const auto& plane : planes_)
        out.push_back(plane.pack_count);
    return out;
}

std::vector<int> ColorFieldBundle::inactive_reasons() const {
    std::vector<int> out;
    for (const auto& plane : planes_)
        out.push_back(plane.inactive_reason);
    return out;
}

std::array<size_t, 3>
ColorFieldBundle::payload_alignment(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    const auto& p = planes_[size_t(target_type)];
    return {reinterpret_cast<uintptr_t>(p.node_fp32.data()) % 64,
            reinterpret_cast<uintptr_t>(p.node_fp16.data()) % 64,
            reinterpret_cast<uintptr_t>(p.brick_fp32.data()) % 64};
}

const char* ColorFieldBundle::node32_source_quantization() {
    return R3_COLOR_FIELD_NODE32_Q ? "half_rounded_then_widened" : "raw_fp32";
}

// Names the sampling arithmetic this plane ACTUALLY runs on this host (not
// merely what the ISA could do), so recorded results identify their path.
const char* ColorFieldBundle::sampling_path(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    const auto& p = planes_[size_t(target_type)];
    if (!p.active) return "analytic_exact";
    if (!avx2_fma_supported_) return "scalar_weighted_stored_channels";
    if (p.backend == COLOR_FIELD_BRICK_FP32) {
        return R3_COLOR_FIELD_BRICK == 2   ? "ymm_packed_horner_3plus1_fma"
               : R3_COLOR_FIELD_BRICK == 1 ? "xmm_horner_7_fma"
                                    : "xmm_weighted_brick";
    }
    if (p.backend == COLOR_FIELD_NODE_FP16 && !f16c_supported_)
        return R3_COLOR_FIELD_NODE == 1 ? "software_half_decode_xmm_xyz"
                                 : "software_half_decode_xmm_weighted";
    if (p.backend == COLOR_FIELD_NODE_FP16)
        return R3_COLOR_FIELD_NODE == 2   ? "packed_two_half4_then_ymm_widen_xyz"
               : R3_COLOR_FIELD_NODE == 1 ? "independent_half4_xmm_widen_xyz"
                                   : "independent_half4_xmm_weighted";
    return R3_COLOR_FIELD_NODE == 2   ? "paired_ymm_xyz"
           : R3_COLOR_FIELD_NODE == 1 ? "xmm_xyz"
                               : "xmm_weighted_nodes";
}

// `xyz` is in angstrom; the plane geometry is in the reduced frame.
bool ColorFieldBundle::in_domain(const Geometry& geometry,
                                 const float* xyz) const {
    for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(xyz[k]))
            throw std::invalid_argument("library coordinates must be finite");
        const float u = COORD_SCALE * xyz[k];
        const float f = (u - geometry.origin[k]) * geometry.inv_h;
        if (f < 0.0f || f > float(geometry.cells[k])) return false;
    }
    return true;
}

// Exact analytic plane value and d/dx gradient in fp64 and angstrom:
// sum coefficient * exp(decay |x - a|^2) over the query sources, with
// decay = -r/2 <= 0. The fallback for inactive planes and outside points.
ColorFieldBundle::ValueGradient
ColorFieldBundle::exact_at(const float* xyz, int target_type) const {
    ValueGradient out;
    for (const ColorFieldSource& source : sources_) {
        const size_t index = cell_index(source.type, target_type);
        const double amplitude = coefficient_[index];
        const double exponent = decay_[index];
        if (amplitude == 0.0 || exponent == 0.0) continue;
        const double dx = double(xyz[0]) - double(source.xyz[0]);
        const double dy = double(xyz[1]) - double(source.xyz[1]);
        const double dz = double(xyz[2]) - double(source.xyz[2]);
        const double value =
            amplitude * std::exp(exponent * (dx * dx + dy * dy + dz * dz));
        out.value += value;
        const double scale = 2.0 * exponent * value;
        out.grad[0] += scale * dx;
        out.grad[1] += scale * dy;
        out.grad[2] += scale * dz;
    }
    return out;
}

// Single-point field sample for an active plane; the caller has already
// checked `in_domain`. Dispatches on backend and cached ISA flags to the same
// kernels the group path uses.
ColorFieldBundle::ValueGradient
ColorFieldBundle::interpolate(const Plane& plane, const float* xyz) const {
    ValueGradient out;
    const Geometry& geometry = plane.geometry;
    size_t base[3];
    float t[3];
    for (int k = 0; k < 3; ++k) {
        const float f =
            (COORD_SCALE * xyz[k] - geometry.origin[k]) * geometry.inv_h;
        int i = int(std::floor(f));
        float frac = f - float(i);
        if (i >= geometry.cells[k]) {
            i = geometry.cells[k] - 1;
            frac = 1.0f;
        }
        base[k] = size_t(i);
        t[k] = frac;
    }
    float channels[4] = {};
    const size_t nx = size_t(geometry.cells[0]) + 1;
    const size_t ny = size_t(geometry.cells[1]) + 1;
    if (plane.backend == COLOR_FIELD_BRICK_FP32) {
        const size_t cell =
            node_index(size_t(geometry.cells[0]), size_t(geometry.cells[1]),
                       base[0], base[1], base[2]);
#if R3_CHANNEL_FIELD_X86_TARGETS
        if (avx2_fma_supported_) {
            interpolate_brick_avx2(plane.brick_fp32.data(), cell, t[0], t[1],
                                   t[2], channels);
        } else
#endif
        {
            const float basis[8] = {
                1.0f,        t[0],        t[1],        t[2],
                t[0] * t[1], t[0] * t[2], t[1] * t[2], t[0] * t[1] * t[2]};
            for (int b = 0; b < 8; ++b)
                for (int ch = 0; ch < 4; ++ch)
                    channels[ch] +=
                        plane
                            .brick_fp32[cell * 32 +
                                        size_t(R3_COLOR_FIELD_BRICK == 2
                                                   ? std::array<int, 8>{0, 2, 4,
                                                                        1, 6, 3,
                                                                        5, 7}[b]
                                                   : b) *
                                            4 +
                                        size_t(ch)] *
                        basis[b];
        }
    } else {
        const float wx[2] = {1.0f - t[0], t[0]};
        const float wy[2] = {1.0f - t[1], t[1]};
        const float wz[2] = {1.0f - t[2], t[2]};
#if R3_CHANNEL_FIELD_X86_TARGETS
        if (avx2_fma_supported_) {
            if (plane.backend == COLOR_FIELD_NODE_FP16 && f16c_supported_) {
                interpolate_node_avx2_f16c(plane.node_fp16.data(), nx, ny,
                                           base[0], base[1], base[2], wx, wy,
                                           wz, channels);
            } else {
                interpolate_node_avx2(
                    plane.node_fp32.data(), plane.node_fp16.data(),
                    plane.backend == COLOR_FIELD_NODE_FP16, nx, ny, base[0],
                    base[1], base[2], wx, wy, wz, channels);
            }
        } else
#endif
        {
            for (int dz = 0; dz < 2; ++dz) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const size_t index =
                            node_index(nx, ny, base[0] + size_t(dx),
                                       base[1] + size_t(dy),
                                       base[2] + size_t(dz)) *
                            4;
                        const float weight = wx[dx] * wy[dy] * wz[dz];
                        for (int ch = 0; ch < 4; ++ch) {
                            const float node =
                                plane.backend == COLOR_FIELD_NODE_FP16
                                    ? half_to_float(
                                          plane.node_fp16[index + size_t(ch)])
                                    : plane.node_fp32[index + size_t(ch)];
                            channels[ch] += node * weight;
                        }
                    }
                }
            }
        }
    }
    out.value = double(channels[0]);
    // Stored gradients are dF/du in the reduced frame u=scale*x. Convert once
    // to dF/dx for the scalar color evaluator's physical-coordinate contract.
    for (int k = 0; k < 3; ++k)
        out.grad[k] = double(channels[k + 1] * COORD_SCALE);
    return out;
}

ColorFieldSample
ColorFieldBundle::sample_target_type(int target_type,
                                     const float* xyz_angstrom) const {
    if (target_type <= 0 || target_type >= n_features_ ||
        xyz_angstrom == nullptr)
        throw std::invalid_argument("invalid color field sample request");
    for (int k = 0; k < 3; ++k)
        if (!std::isfinite(xyz_angstrom[k]))
            throw std::invalid_argument("library coordinates must be finite");
    const Plane& plane = planes_[size_t(target_type)];
    const bool use_field =
        plane.active && in_domain(plane.geometry, xyz_angstrom);
    const ValueGradient value = use_field ? interpolate(plane, xyz_angstrom)
                                          : exact_at(xyz_angstrom, target_type);
    ColorFieldSample out;
    out.value = value.value;
    for (int k = 0; k < 3; ++k)
        out.spatial_gradient[k] = value.grad[k];
    out.used_field = use_field;
    out.backend_code = use_field ? plane.backend : COLOR_FIELD_ANALYTIC;
    return out;
}

bool ColorFieldBundle::can_sample_target_type(int target_type,
                                              const float* xyz_angstrom) const {
    if (target_type <= 0 || target_type >= n_features_ ||
        xyz_angstrom == nullptr)
        throw std::invalid_argument("invalid color field eligibility request");
    const Plane& plane = planes_[size_t(target_type)];
    return plane.active && in_domain(plane.geometry, xyz_angstrom);
}

// The hot path: sample one target-type plane at a group of library features.
// All-or-nothing (see color_field_channels.hpp): on false, nothing in `w` may
// be consumed and the caller evaluates the whole group analytically. On true,
// `w.channels[j]` holds point j's four channels in REDUCED-frame units
// (dF/du); conversion to the caller's frame is the caller's job. The whole
// group is prepared first, so the all-or-nothing domain decision is made
// before any sample is computed.
bool ColorFieldBundle::sample_group(int target_type, const float* points,
                                    size_t count, size_t point_stride,
                                    size_t coordinate_stride,
                                    ColorFieldSampleWorkspace& w,
                                    bool coordinates_reduced) const {
    if (target_type <= 0 || target_type >= n_features_ || (count && !points))
        throw std::invalid_argument("invalid color field group request");
    const Plane& plane = planes_[size_t(target_type)];
    if (!plane.active) {
        // Inactive planes still reject nonfinite input, as the active path
        // does (for inactive and outside groups alike).
        for (size_t j = 0; j < count; ++j)
            for (int k = 0; k < 3; ++k)
                if (!std::isfinite(points[j * point_stride +
                                          size_t(k) * coordinate_stride]))
                    throw std::invalid_argument(
                        "library coordinates must be finite");
        return false;
    }
    const auto& g = plane.geometry;
    const bool all_inside =
        coordinates_reduced
            ? prepare_channel_group<true>(points, count, point_stride,
                                          coordinate_stride, g.origin, g.cells,
                                          g.inv_h, w)
            : prepare_channel_group<false>(points, count, point_stride,
                                           coordinate_stride, g.origin, g.cells,
                                           g.inv_h, w);
    if (!all_inside) return false;
    w.channels.resize(count);
#if R3_CHANNEL_FIELD_X86_TARGETS
    if (avx2_fma_supported_) {
        if (plane.backend == COLOR_FIELD_NODE_FP16 && f16c_supported_)
            sample_prepared_f16c(plane, w);
        else
            sample_prepared_avx2(plane, w);
        return true;
    }
#endif
    // Generic fallback consumes the same preparation and logical layout.
    const size_t nx = size_t(plane.geometry.cells[0]) + 1,
                 ny = size_t(plane.geometry.cells[1]) + 1;
    for (size_t j = 0; j < count; ++j) {
        const auto& p = w.prepared[j];
        auto& out = w.channels[j];
        out.fill(0);
        const float x = p.fraction[0], y = p.fraction[1], z = p.fraction[2];
        if (plane.backend == COLOR_FIELD_BRICK_FP32) {
            const size_t cell =
                node_index(nx - 1, ny - 1, p.base[0], p.base[1], p.base[2]);
            const float basis[8] = {1, x, y, z, x * y, x * z, y * z, x * y * z};
            for (int b = 0; b < 8; ++b)
                for (int ch = 0; ch < 4; ++ch) {
                    const size_t slot = size_t(
                        R3_COLOR_FIELD_BRICK == 2
                            ? std::array<int, 8>{0, 2, 4, 1, 6, 3, 5, 7}[b]
                            : b);
                    out[size_t(ch)] +=
                        plane.brick_fp32[cell * 32 + slot * 4 + size_t(ch)] *
                        basis[b];
                }
        } else {
            const float wx[2] = {1 - x, x}, wy[2] = {1 - y, y},
                        wz[2] = {1 - z, z};
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const size_t index =
                            node_index(nx, ny, p.base[0] + size_t(dx),
                                       p.base[1] + size_t(dy),
                                       p.base[2] + size_t(dz)) *
                            4;
                        const float weight = wx[dx] * wy[dy] * wz[dz];
                        for (int ch = 0; ch < 4; ++ch) {
                            const float v =
                                plane.backend == COLOR_FIELD_NODE_FP16
                                    ? half_to_float(
                                          plane.node_fp16[index + size_t(ch)])
                                    : plane.node_fp32[index + size_t(ch)];
                            out[size_t(ch)] += v * weight;
                        }
                    }
        }
    }
    return true;
}

#if R3_CHANNEL_FIELD_X86_TARGETS
// Batch samplers over already-prepared records. Each loop lives inside one
// target-attributed function, so the inline kernels are inlined into it and
// the batch makes no per-sample out-of-line call.
R3_CHANNEL_FIELD_AVX2 void
ColorFieldBundle::sample_prepared_avx2(const Plane& plane,
                                       ColorFieldSampleWorkspace& w) const {
    const size_t nx = size_t(plane.geometry.cells[0]) + 1,
                 ny = size_t(plane.geometry.cells[1]) + 1;
    if (plane.backend == COLOR_FIELD_BRICK_FP32) {
        for (size_t j = 0; j < w.prepared.size(); ++j) {
            const auto& p = w.prepared[j];
            const size_t cell =
                node_index(nx - 1, ny - 1, p.base[0], p.base[1], p.base[2]);
            interpolate_brick_avx2_inline(plane.brick_fp32.data(), cell,
                                          p.fraction[0], p.fraction[1],
                                          p.fraction[2], w.channels[j].data());
        }
    } else {
        const bool half = plane.backend == COLOR_FIELD_NODE_FP16;
        for (size_t j = 0; j < w.prepared.size(); ++j) {
            const auto& p = w.prepared[j];
            const float wx[2] = {1 - p.fraction[0], p.fraction[0]},
                        wy[2] = {1 - p.fraction[1], p.fraction[1]},
                        wz[2] = {1 - p.fraction[2], p.fraction[2]};
            interpolate_node_avx2_inline(plane.node_fp32.data(),
                                         plane.node_fp16.data(), half, nx, ny,
                                         p.base[0], p.base[1], p.base[2], wx,
                                         wy, wz, w.channels[j].data());
        }
    }
}

R3_CHANNEL_FIELD_F16C void
ColorFieldBundle::sample_prepared_f16c(const Plane& plane,
                                       ColorFieldSampleWorkspace& w) const {
    const size_t nx = size_t(plane.geometry.cells[0]) + 1,
                 ny = size_t(plane.geometry.cells[1]) + 1;
    for (size_t j = 0; j < w.prepared.size(); ++j) {
        const auto& p = w.prepared[j];
        const float wx[2] = {1 - p.fraction[0], p.fraction[0]},
                    wy[2] = {1 - p.fraction[1], p.fraction[1]},
                    wz[2] = {1 - p.fraction[2], p.fraction[2]};
        interpolate_node_avx2_f16c_inline(plane.node_fp16.data(), nx, ny,
                                          p.base[0], p.base[1], p.base[2], wx,
                                          wy, wz, w.channels[j].data());
    }
}
#else
void ColorFieldBundle::sample_prepared_avx2(const Plane&,
                                            ColorFieldSampleWorkspace&) const {}

void ColorFieldBundle::sample_prepared_f16c(const Plane&,
                                            ColorFieldSampleWorkspace&) const {}
#endif

size_t ColorFieldBundle::field_bytes() const {
    size_t total = 0;
    for (const Plane& plane : planes_)
        total += plane.bytes();
    return total;
}

size_t ColorFieldBundle::field_bytes(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    return planes_[size_t(target_type)].bytes();
}

double ColorFieldBundle::build_seconds(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    return planes_[size_t(target_type)].build_s;
}

double ColorFieldBundle::pack_seconds(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    return planes_[size_t(target_type)].pack_s;
}

std::vector<size_t> ColorFieldBundle::field_bytes_by_target_type() const {
    std::vector<size_t> out(static_cast<size_t>(n_features_), size_t(0));
    for (int t = 0; t < n_features_; ++t)
        out[size_t(t)] = planes_[size_t(t)].bytes();
    return out;
}

std::vector<double> ColorFieldBundle::build_seconds_by_target_type() const {
    std::vector<double> out(static_cast<size_t>(n_features_), 0.0);
    for (int t = 0; t < n_features_; ++t)
        out[size_t(t)] = planes_[size_t(t)].build_s;
    return out;
}

std::vector<double> ColorFieldBundle::pack_seconds_by_target_type() const {
    std::vector<double> out(static_cast<size_t>(n_features_), 0.0);
    for (int t = 0; t < n_features_; ++t)
        out[size_t(t)] = planes_[size_t(t)].pack_s;
    return out;
}

int ColorFieldBundle::backend_code(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    return planes_[size_t(target_type)].backend;
}

std::vector<int> ColorFieldBundle::backend_codes() const {
    std::vector<int> out(static_cast<size_t>(n_features_), 0);
    for (int t = 0; t < n_features_; ++t)
        out[size_t(t)] = planes_[size_t(t)].backend;
    return out;
}

bool ColorFieldBundle::field_active(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    return planes_[size_t(target_type)].active;
}

std::vector<int> ColorFieldBundle::field_active_by_target_type() const {
    std::vector<int> out(static_cast<size_t>(n_features_), 0);
    for (int t = 0; t < n_features_; ++t)
        out[size_t(t)] = planes_[size_t(t)].active ? 1 : 0;
    return out;
}

std::array<int, 3> ColorFieldBundle::grid_cells(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    const Geometry& g = planes_[size_t(target_type)].geometry;
    return {g.cells[0], g.cells[1], g.cells[2]};
}

// Geometry is stored in the reduced frame; these report it in angstrom.
std::array<double, 3>
ColorFieldBundle::grid_origin_angstrom(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    const Geometry& g = planes_[size_t(target_type)].geometry;
    const double inverse_scale = 1.0 / double(COORD_SCALE);
    return {double(g.origin[0]) * inverse_scale,
            double(g.origin[1]) * inverse_scale,
            double(g.origin[2]) * inverse_scale};
}

double ColorFieldBundle::grid_spacing_angstrom(int target_type) const {
    if (target_type < 0 || target_type >= n_features_)
        throw std::invalid_argument("color target type is out of range");
    return double(planes_[size_t(target_type)].geometry.h) /
           double(COORD_SCALE);
}

std::vector<int> ColorFieldBundle::target_types() const {
    std::vector<int> out;
    for (int t = 1; t < n_features_; ++t)
        if (planes_[size_t(t)].active) out.push_back(t);
    return out;
}

const char* ColorFieldBundle::half_decode_path() const {
    if (f16c_supported_) return "avx2_fma_f16c";
    if (avx2_fma_supported_) return "software_decode_avx2_fma";
    return "software_scalar";
}

void bind_color_fields(py::module_& module) {
    using DoubleArray =
        py::array_t<double, py::array::c_style | py::array::forcecast>;
    py::class_<ColorFieldBundle>(
        module, "ColorFieldBundle",
        "Experimental four-channel CPU color field bundle. It stores and interpolates value and spatial-gradient channels; backend 3 uses fp32 polynomial coefficients packed from half-rounded AoS4 nodes. Exact final scoring remains analytic.")
        .def(
            py::init([](py::array_t<float, py::array::c_style> query_xyz,
                        py::array_t<int32_t, py::array::c_style> query_types,
                        DoubleArray coefficients, DoubleArray decays,
                        py::object backend_codes, double spacing_angstrom,
                        double padding_angstrom) {
                const auto q = query_xyz.unchecked<2>();
                const auto qt = query_types.unchecked<1>();
                const auto c = coefficients.unchecked<2>();
                const auto d = decays.unchecked<2>();
                const py::array bc = py::array::ensure(backend_codes);
                if (q.shape(1) != 3 || qt.shape(0) != q.shape(0))
                    throw std::invalid_argument(
                        "query coordinates must be [n,3] and match query types");
                const int n = int(c.shape(0));
                if (n < 1 || n > MAX_COLOR_FEATURES || c.shape(1) != n ||
                    d.shape(0) != n || d.shape(1) != n || !bc ||
                    bc.ndim() != 1 || bc.shape(0) != n)
                    throw std::invalid_argument(
                        "color interaction matrix shape is invalid");
                const py::module_ numpy = py::module_::import("numpy");
                if (!py::cast<bool>(numpy.attr("issubdtype")(
                        bc.dtype(), numpy.attr("integer"))))
                    throw std::invalid_argument(
                        "color field backend codes must be an integer vector");
                std::vector<ColorFieldSource> sources(size_t(q.shape(0)));
                for (py::ssize_t i = 0; i < q.shape(0); ++i) {
                    sources[size_t(i)].xyz[0] = q(i, 0);
                    sources[size_t(i)].xyz[1] = q(i, 1);
                    sources[size_t(i)].xyz[2] = q(i, 2);
                    sources[size_t(i)].type = qt(i);
                }
                std::vector<int> codes(static_cast<size_t>(n), 0);
                for (int i = 0; i < n; ++i) {
                    const py::object item = bc[py::int_(i)];
                    if (py::cast<bool>(item.attr("__lt__")(py::int_(0))) ||
                        py::cast<bool>(item.attr("__gt__")(py::int_(
                            static_cast<int>(COLOR_FIELD_BRICK_FP32)))))
                        throw std::invalid_argument(
                            "color field backend codes must be 0, 1, 2 or 3");
                    codes[size_t(i)] = py::cast<int>(item);
                }
                return ColorFieldBundle(sources, n, coefficients.data(),
                                        decays.data(), codes.data(),
                                        spacing_angstrom, padding_angstrom);
            }),
            py::arg("query_xyz"), py::arg("query_types"),
            py::arg("coefficients"), py::arg("decays"),
            py::arg("backend_codes"), py::arg("spacing_angstrom") = 0.5,
            py::arg("padding_angstrom") = 4.0)
        .def(
            "sample",
            [](const ColorFieldBundle& bundle, int target_type,
               py::array_t<float, py::array::c_style> positions) {
                const auto xyz = positions.unchecked<2>();
                if (xyz.shape(1) != 3)
                    throw std::invalid_argument(
                        "sample positions must have shape [n,3]");
                py::array_t<double> values(xyz.shape(0));
                py::array_t<double> gradients({xyz.shape(0), py::ssize_t(3)});
                py::array_t<bool> used(xyz.shape(0));
                py::array_t<int32_t> backends(xyz.shape(0));
                auto v = values.mutable_unchecked<1>();
                auto g = gradients.mutable_unchecked<2>();
                auto u = used.mutable_unchecked<1>();
                auto b = backends.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < xyz.shape(0); ++i) {
                    const float point[3] = {xyz(i, 0), xyz(i, 1), xyz(i, 2)};
                    const ColorFieldSample sample =
                        bundle.sample_target_type(target_type, point);
                    v(i) = sample.value;
                    for (int k = 0; k < 3; ++k)
                        g(i, k) = sample.spatial_gradient[k];
                    u(i) = sample.used_field;
                    b(i) = sample.backend_code;
                }
                py::dict result;
                result["value"] = values;
                result["spatial_gradient"] = gradients;
                result["used_field"] = used;
                result["backend_code"] = backends;
                return result;
            },
            py::arg("target_type"), py::arg("positions"))
        .def_property_readonly(
            "field_bytes",
            [](const ColorFieldBundle& x) { return x.field_bytes(); })
        .def_property_readonly("field_bytes_by_target_type",
                               &ColorFieldBundle::field_bytes_by_target_type)
        .def_property_readonly(
            "build_seconds",
            [](const ColorFieldBundle& x) { return x.build_seconds(); })
        .def_property_readonly("build_seconds_by_target_type",
                               &ColorFieldBundle::build_seconds_by_target_type)
        .def_property_readonly("pack_seconds_by_target_type",
                               &ColorFieldBundle::pack_seconds_by_target_type)
        .def_property_readonly("backend_codes",
                               &ColorFieldBundle::backend_codes)
        .def_property_readonly("field_active_by_target_type",
                               &ColorFieldBundle::field_active_by_target_type)
        .def_property_readonly("target_types", &ColorFieldBundle::target_types)
        .def_property_readonly("avx2_fma_supported",
                               &ColorFieldBundle::avx2_fma_supported)
        .def_property_readonly("f16c_supported",
                               &ColorFieldBundle::f16c_supported)
        .def_property_readonly("half_decode_path",
                               [](const ColorFieldBundle& x) {
                                   return std::string(x.half_decode_path());
                               })
        .def_property_readonly("generation", &ColorFieldBundle::generation)
        .def("payload_alignment", &ColorFieldBundle::payload_alignment)
        .def("sampling_path", &ColorFieldBundle::sampling_path)
        .def_property_readonly_static(
            "node32_source_quantization",
            [](py::object) {
                return ColorFieldBundle::node32_source_quantization();
            })
        .def_property_readonly_static(
            "construction_count",
            [](py::object) { return ColorFieldBundle::construction_count(); })
        .def_property_readonly("pack_counts", &ColorFieldBundle::pack_counts)
        .def_property_readonly("inactive_reasons",
                               &ColorFieldBundle::inactive_reasons)
        .def_property_readonly("preparation_seconds",
                               &ColorFieldBundle::preparation_seconds)
        .def_property_readonly("peak_payload_bytes",
                               &ColorFieldBundle::peak_payload_bytes)
        .def_property_readonly_static(
            "coefficient_layout_version",
            [](py::object) {
                return ColorFieldBundle::coefficient_layout_version();
            })
        .def(
            "field_bytes_for",
            [](const ColorFieldBundle& x, int target_type) {
                return x.field_bytes(target_type);
            },
            py::arg("target_type"))
        .def(
            "build_seconds_for",
            [](const ColorFieldBundle& x, int target_type) {
                return x.build_seconds(target_type);
            },
            py::arg("target_type"))
        .def("pack_seconds_for", &ColorFieldBundle::pack_seconds,
             py::arg("target_type"))
        .def("backend_code_for", &ColorFieldBundle::backend_code,
             py::arg("target_type"))
        .def("field_active_for", &ColorFieldBundle::field_active,
             py::arg("target_type"))
        .def("grid_cells", &ColorFieldBundle::grid_cells,
             py::arg("target_type"))
        .def("grid_origin_angstrom", &ColorFieldBundle::grid_origin_angstrom,
             py::arg("target_type"))
        .def("grid_spacing_angstrom", &ColorFieldBundle::grid_spacing_angstrom,
             py::arg("target_type"));
}

}
}  // namespace r3::cpu

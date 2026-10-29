// Four grades of exp2 on the negative half-line, and the accuracy bar is not libm.
//
// The reduced frame makes the kernel's inner term exactly `exp2(-|delta|^2)`
// (numerics.py's COORD_SCALE), so **the argument is always <= 0 and the result
// always in [0, 1]**. That bounded domain is what makes cheap grades possible
// at all, and it is why a general-purpose exp2 is the wrong instrument.
//
// THE BAR. The CUDA kernel's `exp2f` compiles to `ex2.approx.ftz.f32` -- a
// hardware approximation with a ~2 ULP bound and denormals flushed. So matching
// libm's correctly-rounded `exp2f` would be paying for accuracy the
// specification never had, and `doc/NUMERICS_POLICY.md` never asked for.
//
// AND THE CHOICE IS A RECALL QUESTION, NOT A BIT QUESTION. A change of exp2
// implementation is a LARGER numerics change than a reassociation -- it is in
// the class `doc/NUMERICS_POLICY.md` section 1 enumerates, where every perturbing
// axis reaches the same saturated output distribution -- so no grade here is
// ever validated bitwise and none needs to be. Each is judged on ranked output
// against the scalar kernel, and on throughput, and recommended on the pair.
//
// The useful domain, derived rather than assumed: `exp2(-24) = 6.0e-8`, against
// an accumulated sum of order n_A * n_B ~ 830. A term below that is invisible
// in fp32, so `d2 > 24` may be flushed to zero -- which is the same knob as
// the distance cutoff, stated once here so the two cannot be chosen
// independently and double-count.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace r3 {
namespace exp2g {

//: Beyond this the term is below fp32's contribution to the sum. exp2(-24) =
//: 6.0e-8; the sum is order 830. Shared with the distance cutoff by construction.
constexpr float NEGLIGIBLE_D2 = 24.0f;

enum Grade : int {
    GRADE_LIBM = 0,     // std::exp2f, correctly rounded. The control.
    GRADE_MINIMAX = 1,  // ~2 ULP, ex2.approx's class. Degree 5 on [-0.5,0.5].
    GRADE_CHEAP = 2,    // degree 2, deliberately coarse.
    GRADE_TABLE = 3,    // 4096-entry fp32 table on d2, linearly interpolated.
    GRADE_TABLE_NOLERP =
        4,  // the same table, nearest entry -- the control that
            // says what the interpolation is worth.
    N_GRADES = 5,
};

inline const char* grade_name(int g) {
    switch (g) {
    case GRADE_LIBM:
        return "libm";
    case GRADE_MINIMAX:
        return "minimax_d5";
    case GRADE_CHEAP:
        return "cheap_d2";
    case GRADE_TABLE:
        return "table_4096_lerp";
    case GRADE_TABLE_NOLERP:
        return "table_4096_nearest";
    default:
        return "unknown";
    }
}

// --- grade 1: the minimax polynomial ---------------------------------------
//
// exp2(x) = 2^k * exp2(f) with k = round(x), f = x - k in [-0.5, 0.5]. The
// power of two is assembled by placing k in the exponent field directly, which
// is the operation `vscalefps` does in one instruction on AVX-512 -- the
// single biggest AVX-512-ism for this kernel, and the reason the width
// question is not only about lane count.
//
// Coefficients are the degree-5 minimax fit of 2^f on [-0.5, 0.5]. Written as
// literals rather than derived at run time so the arithmetic is fixed and the
// stamp means something.
inline float exp2_minimax(float x) {
    const float k = std::nearbyint(x);
    const float f = x - k;
    // Horner, 5 FMAs.
    float p = 1.33978436e-3f;
    p = std::fma(p, f, 9.61799621e-3f);
    p = std::fma(p, f, 5.55051871e-2f);
    p = std::fma(p, f, 2.40226507e-1f);
    p = std::fma(p, f, 6.93147182e-1f);
    p = std::fma(p, f, 1.0f);
    // 2^k by exponent assembly. k is in [-24, 0] here, so no overflow path.
    int32_t ki = static_cast<int32_t>(k);
    int32_t bits = (ki + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// --- grade 2: deliberately coarse ------------------------------------------
inline float exp2_cheap(float x) {
    const float k = std::nearbyint(x);
    const float f = x - k;
    // Degree 2. Max relative error ~1e-2 on [-0.5, 0.5]; here to establish
    // where the accuracy/throughput curve actually bends, not to be shipped.
    float p = std::fma(2.16755911e-1f, f, 6.58448458e-1f);
    p = std::fma(p, f, 1.00172475f);
    int32_t ki = static_cast<int32_t>(k);
    int32_t bits = (ki + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// --- grades 3 and 4: the table ---------------------------------------------
//
// Indexed on d2 rather than on the exp2 argument, because d2 is what the kernel
// has and the negation is then free. 4096 entries over [0, 24] is 16 kB in
// fp32 -- a third of Zen 3's 32 kB L1d, which is the sizing constraint that
// matters and the reason 16,384 entries was rejected before it was written:
// that is 64 kB and IS the whole L1d twice over.
//
// **The table is a gather.** Its index depends on the lane's own d2, so in a
// vectorized kernel it costs exactly the resource measured as the field
// path's bottleneck. Reported in gathers as well as in cycles so the two
// decisions can be added rather than taken separately.
constexpr int TABLE_N = 4096;

struct Table {
    float v[TABLE_N + 1];
    float inv_step;

    Table() {
        inv_step = float(TABLE_N) / NEGLIGIBLE_D2;
        for (int i = 0; i <= TABLE_N; ++i) {
            const double d2 =
                double(i) * double(NEGLIGIBLE_D2) / double(TABLE_N);
            v[i] = float(std::exp2(-d2));
        }
    }
};

inline const Table& table() {
    static const Table t;
    return t;
}

inline float exp2_table_lerp(float d2) {
    if (d2 >= NEGLIGIBLE_D2) return 0.0f;
    const Table& t = table();
    const float u = d2 * t.inv_step;
    const int i = int(u);
    const float frac = u - float(i);
    return std::fma(frac, t.v[i + 1] - t.v[i], t.v[i]);
}

inline float exp2_table_nearest(float d2) {
    if (d2 >= NEGLIGIBLE_D2) return 0.0f;
    const Table& t = table();
    return t.v[int(d2 * t.inv_step + 0.5f)];
}

//: Evaluate grade `g` at `-d2`. Takes d2 (non-negative) because that is what
//: the kernel holds and what the table indexes on.
inline float eval(int g, float d2) {
    switch (g) {
    case GRADE_LIBM:
        return std::exp2(-d2);
    case GRADE_MINIMAX:
        return exp2_minimax(-d2);
    case GRADE_CHEAP:
        return exp2_cheap(-d2);
    case GRADE_TABLE:
        return exp2_table_lerp(d2);
    case GRADE_TABLE_NOLERP:
        return exp2_table_nearest(d2);
    default:
        return std::exp2(-d2);
    }
}

}  // namespace exp2g
}  // namespace r3

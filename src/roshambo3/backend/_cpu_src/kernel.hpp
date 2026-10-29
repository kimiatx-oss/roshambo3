// The CPU shape kernel: reduced frame, exact analytic pair sum, Adagrad on the
// exponential map, start modes as a conjugation.
//
// This is deliberately the SPECIFICATION's shape rather than the production
// path's: no field, no cutoff, no SIMD, no threads. Each of those is measured
// against this.
//
// WHAT IT IMPLEMENTS, and where each constant comes from
// -----------------------------------------------------
// Grant-Pickup first-order Gaussian overlap, uniform carbon radius:
//
//   V_AB = 8 (pi/2A)^{3/2} sum_i sum_j exp(-(A/2) |a_i - b_j|^2)
//   T    = V_AB / (V_AA + V_BB - V_AB)
//
// evaluated in the REDUCED-OVERLAP FRAME, which `doc/NUMERICS_POLICY.md` section 2
// makes a precondition rather than an option: the old C++ backend was permitted
// to stay in volume units *because it was an oracle*, and "if it is ever
// promoted to a scoring path, it must adopt the canonical frame first, as a
// declared change". So coordinates arrive pre-multiplied by
// COORD_SCALE = sqrt(alpha log2 e) = 0.77687365, the inner term is exactly
// exp2(-|delta|^2) with no multiply, and the prefactor is applied once at the
// point of reporting.
//
// THE GRADIENT is the kernel's, not dV/dq: g[0] == 0, and with Mb = R(q) s b_j,
// delta = (a_i - t) - Mb, C = 2 Mb,
//
//   temp     = (-2 ln2 / (S_AA + S_BB)) * exp2(-|delta|^2)
//   g[1..3] += temp * (C x delta)      // a space-frame torque
//   g[4..6] += temp * delta
//
// The 1/(S_AA + S_BB) is folded into the per-term coefficient rather than
// applied afterwards, which is what keeps the accumulated gradient O(1)
// whatever the molecule size and removes seven multiplies per step.
//
// THE START MODES ARE A CONJUGATION, not initial poses. See the corrected block
// comment in ../_cuda_src/kernels.cu: every start begins at the
// identity quaternion and R^T is folded into the pose matrix, which is exact
// only because every start rotation is a signed permutation -- checked here at
// run time, as it is there, rather than commented.
//
// DERIVED FROM `reference/overlap.py` AND THE PUBLISHED EXPRESSION, NOT FROM
// `kernels.cu`. Pitfall 5 is the reason and it is not hypothetical: "C++ and
// CUDA agree, so the math is right -- they agree because they share the typo",
// and a quaternion normalization defect survived in ROSHAMBO2's two backends
// for as long as they were each other's only check. The CUDA source was read
// here for CONVENTIONS -- which way the pose applies, what the start set is,
// where the scale lives -- and not for the algebra.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "simd.hpp"

namespace r3 {
namespace cpu {

// --- the pinned constants, mirroring roshambo3/numerics.py -----------------
// A test asserts these against the Python derivation, so they cannot drift.
constexpr double KAPPA = 2.41798793102;
constexpr double CARBON_RADIUS = 1.7;

constexpr float SHAPE_PREFACTOR = 20.5795288f;  // 8 (pi/2A)^1.5
constexpr float COORD_SCALE = 0.77687365f;      // sqrt(alpha log2 e)
constexpr float MINUS_2LN2 = -1.38629436f;      // d/dx exp2(-x) carries -2 ln2

//: The number of score columns a row carries. Must equal
//: roshambo3.scores.N_SCORE_COLUMNS.
constexpr int N_SCORE_COLUMNS = 20;

//: Signed-permutation tolerance, matching ROSHAMBO3_SIGNED_PERM_TOL in
//: ../_cuda_src/kernels.cu. A 1-degree turn misses by 0.0175, so this admits
//: float32 noise and nothing else.
constexpr float SIGNED_PERM_TOL = 1e-4f;

//: Instrumented operation counts, for the census. Counted rather than
//: derived, because a derivation of what the loop "should" do is exactly the
//: kind of thing that goes wrong silently -- and every later CPU arm
//: is reported as a fraction of a ceiling computed from these.
struct OpCensus {
    uint64_t exp2_calls = 0;
    uint64_t pair_terms_optimizer = 0;  // atom pairs visited inside the search
    uint64_t pair_terms_rescore = 0;    // atom pairs in the final exact sum
    uint64_t optimizer_steps = 0;
    uint64_t starts = 0;
    uint64_t pairs = 0;
    uint64_t self_overlap_terms = 0;
    //: (i, k) evaluations skipped because EVERY lane was beyond the cutoff
    //: (cutoff arm). Counted rather than derived: the union rate is the whole
    //: question, and a derivation of how often 8 lanes agree is exactly the
    //: kind of estimate that turns out wrong.
    uint64_t cutoff_skips = 0;
    //: (i, k) evaluations the block loop VISITED, cutoff or not. The
    //: denominator for `cutoff_skips`, counted rather than derived -- deriving
    //: it from the corpus-padded width instead of the block's own max_atoms
    //: understated the skip rate by 4%, which is the reason the census
    //: counts rather than derives.
    uint64_t cutoff_visits = 0;
    //: Pose evaluations the conservative bounding-sphere test proved lie
    //: wholly inside the lattice, and the total it was applied to. Counted
    //: rather than derived for the same reason `cutoff_skips` is: how often a
    //: whole conformer clears the halo is exactly the quantity an estimate
    //: gets wrong, and it is what decides whether the fast path is worth its
    //: second instantiation.
    uint64_t allin_poses = 0;
    //: Eight-pair blocks the batched optimizer step actually ran, and pairs
    //: that fell back. The batched path needs EIGHT CONSECUTIVE pairs sharing
    //: a query and silently takes the per-pair path otherwise, so a schedule
    //: that reorders or prunes the pair list can turn the arm off without any
    //: error -- and the composed number then reads as "these do not compose"
    //: when the truth is "one of them stopped running". Counted, not inferred
    //: from the pair list.
    uint64_t batched_blocks = 0;
    uint64_t allin_pose_visits = 0;
    //: Blocks whose every lane converged before the step budget ran out
    //: (convergence arm). The union again: a block exits only when all lanes agree.
    uint64_t converged_blocks = 0;
    //: Field samples taken during the pose search. The field path's
    //: unit of work: O(n_B) per step where the analytic path is O(n_A n_B),
    //: which is the entire reason the field exists. Counted, not derived --
    //: a model predicted 2,585.66 per pair from a corpus-wide atom mean and this
    //: kernel's own screen gives a different number for the same reason
    //: (a different atom count), so a derivation cannot check the loop.
    uint64_t field_samples = 0;
};

// --- quaternion helpers ----------------------------------------------------
//
// EVERY FUNCTION BELOW IS `static`, AND THAT IS LOAD-BEARING. This
// header is included, transitively, by BOTH of _roshambo3_cpu's translation
// units -- interface.cpp at `-march=x86-64-v3` and grain_avx512.cpp at
// `-mavx512f` (CMakeLists.txt:500-509, grain_avx512.cpp:1-31). As plain
// `inline` functions they would have external linkage, one COMDAT group
// per name, so the SAME function is compiled twice, once per ISA, and
// implicit LTO (pybind11::lto) keeps exactly one merged copy of each --
// whichever TU's object file the linker resolves the symbol from first.
// That resolution is LINK ORDER
// (the order `interface.cpp` and `grain_avx512.cpp` are listed in
// `pybind11_add_module`), not inlining and not luck: reversing that order in
// a relink flips `SoaConformer::build` (grain.hpp) to a body containing
// genuine `vpermt2ps`/zmm AVX-512 instructions, which would SIGILL on a
// machine without AVX-512 the instant the (correctly, by source) v3 field
// path called it. `static` gives each translation unit its OWN copy under
// its OWN name -- no COMDAT, no cross-TU merge, so which ISA a caller gets
// is determined by which TU it is compiled in, not by an accident of link
// order. `inline` is kept alongside `static` so a TU that includes but does
// not call a given function (e.g. `grain_avx512.cpp` never calls
// `self_overlap`) does not warn `-Wunused-function`.

// Rotation matrix of (w,x,y,z), NORMALIZED FIRST. Normalizing is the point,
// not hygiene: the defect the float64 oracle exists to catch was a missing
// normalization that made the applied transform non-orthogonal, with |q|
// reaching 1.216 and det(M) 1.313, so 1.2% of poses reported a Tanimoto no
// rigid motion can achieve.
static inline void quat_to_mat(const float q[4], float m[9]) {
    float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(n > 0.0f) || !std::isfinite(n)) {
        m[0] = m[4] = m[8] = 1.0f;
        m[1] = m[2] = m[3] = m[5] = m[6] = m[7] = 0.0f;
        return;
    }
    const float w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
    m[0] = 1 - 2 * (y * y + z * z);
    m[1] = 2 * (x * y - w * z);
    m[2] = 2 * (x * z + w * y);
    m[3] = 2 * (x * y + w * z);
    m[4] = 1 - 2 * (x * x + z * z);
    m[5] = 2 * (y * z - w * x);
    m[6] = 2 * (x * z - w * y);
    m[7] = 2 * (y * z + w * x);
    m[8] = 1 - 2 * (x * x + y * y);
}

// q <- exp(d) (x) q, with |d| the half-angle. Unit by construction: a product
// of two unit quaternions, so there is no renormalization to get subtly wrong.
// This is `opt_mode = 1`, the default on both CUDA kernels, and the
// reason the singularity the CUDA comment used to describe does not exist:
// q[0]' = e0*q0 - e1*q1 - e2*q2 - e3*q3 is nonzero at q = (0,1,0,0) for any
// d[0] != 0, and left multiplication by a unit quaternion is transitive on the
// unit sphere.
static inline void quat_left_exp(float q[4], const float d[3]) {
    const float a2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    float c, s;
    if (a2 > 1e-12f) {
        const float a = std::sqrt(a2);
        c = std::cos(a);
        s = std::sin(a) / a;
    } else {
        // The series, so a zero gradient is a no-op rather than a 0/0.
        c = 1.0f - 0.5f * a2;
        s = 1.0f - a2 * (1.0f / 6.0f);
    }
    const float e0 = c, e1 = s * d[0], e2 = s * d[1], e3 = s * d[2];
    const float q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    q[0] = e0 * q0 - e1 * q1 - e2 * q2 - e3 * q3;
    q[1] = e0 * q1 + e1 * q0 + e2 * q3 - e3 * q2;
    q[2] = e0 * q2 - e1 * q3 + e2 * q0 + e3 * q1;
    q[3] = e0 * q3 + e1 * q2 - e2 * q1 + e3 * q0;
}

static inline void quat_mul(const float a[4], const float b[4], float out[4]) {
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

// --- the start set ---------------------------------------------------------

//: start_mode -> number of starts. Mirrors `start_mode_n` in kernels.cuh.
static inline int start_count(int mode) {
    switch (mode) {
    case 0:
        return 1;
    case 1:
        return 4;
    case 2:
        return 10;
    default:
        return -1;
    }
}

// The start quaternion, from the same axis/angle table the CUDA kernel uses:
// identity, pi about each axis, then +-pi/2 about each.
static inline void start_quaternion(int mode, int index, float q[4]) {
    int axis = 0;
    float ang = 0.0f;
    if (mode == 1) {
        axis = (index == 0) ? 0 : (index - 1);
        ang = (index == 0) ? 0.0f : float(M_PI);
    } else if (mode == 2) {
        axis = (index == 0) ? 0 : ((index - 1) % 3);
        ang = (index == 0)   ? 0.0f
              : (index <= 3) ? float(M_PI)
              : (index <= 6) ? 0.5f * float(M_PI)
                             : -0.5f * float(M_PI);
    }
    const float h = 0.5f * ang;
    q[0] = std::cos(h);
    q[1] = q[2] = q[3] = 0.0f;
    q[1 + axis] = std::sin(h);
}

// Is R a signed permutation? The correspondence between "rotate the query" and
// "conjugate the problem" is EXACT only if it is: Adagrad keeps a
// per-component cache, and the componentwise update commutes with the
// conjugation only when R^T permutes components rather than mixing them. Every
// start in modes 0/1/2 is an axis-aligned multiple of 90 degrees and so
// qualifies; a new start mode off that group must re-check rather than inherit.
// Checked at run time here as it is on the device, not asserted in a comment.
static inline bool matrix_is_signed_permutation(const float m[9], float tol) {
    for (int r = 0; r < 3; ++r) {
        int nonzero = 0;
        for (int c = 0; c < 3; ++c) {
            const float v = std::fabs(m[3 * r + c]);
            if (v > tol && std::fabs(v - 1.0f) > tol) return false;
            if (v > tol) ++nonzero;
        }
        if (nonzero != 1) return false;
    }
    for (int c = 0; c < 3; ++c) {
        int nonzero = 0;
        for (int r = 0; r < 3; ++r)
            if (std::fabs(m[3 * r + c]) > tol) ++nonzero;
        if (nonzero != 1) return false;
    }
    return true;
}

}  // namespace cpu
}  // namespace r3

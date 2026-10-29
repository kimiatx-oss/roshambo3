// MIT License
// 
// Copyright (c) 2026 Kimia Therapeutics
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK_ERROR(err) \
    do { \
        cudaError_t cudaErr = (err); \
        if (cudaErr != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cudaErr) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

// The repository-wide block size remains fixed. The combo build factor is scoped
// to the combo/field optimizer so its causal comparison does not also perturb
// analytic, field-build, overlap, and test kernels.
#define NTHREADS 256
#ifndef ROSHAMBO3_NTHREADS
#define ROSHAMBO3_NTHREADS 256
#endif
#define COMBO_NTHREADS ROSHAMBO3_NTHREADS

#ifndef ROSHAMBO3_FINAL_EXACT_SHAPE
#define ROSHAMBO3_FINAL_EXACT_SHAPE 1
#endif
#ifndef ROSHAMBO3_FINAL_EXACT_COLOR
#define ROSHAMBO3_FINAL_EXACT_COLOR 1
#endif



__device__ __constant__ float EPSILON = 1E-8f;

__device__ __constant__ float PI = 3.14159265358f;

////////////////////////////////////////////////////////////////////////////////
/// The reduced-overlap frame (doc/NUMERICS_POLICY.md section 4).
///
/// The kernels do NOT work in volume units. They accumulate a *reduced* overlap
///
///     S = sum_ij w_i w_j exp2(-|s*(r_i - r_j)|^2)
///
/// and the physical volume is recovered once, at the point of reporting, as
/// V = SHAPE_PREFACTOR * S (shape) or V = COLOR_PREFACTOR * S (color). Three
/// separate simplifications, all of them exact rearrangements:
///
///  1. The Tanimoto T = V_AB/(V_AA + V_BB - V_AB) is homogeneous of degree
///     zero, so a prefactor shared by all three volumes cancels identically.
///     For shape that prefactor is 8*(pi/2A)^1.5 = 20.5795; for color, with
///     the default diagonal interaction matrix (r = 1 on the diagonal), every
///     feature pair carries the same (pi/2)^1.5 = 1.9687. A non-diagonal
///     matrix does not break this: the per-pair residual p^2 * r^-1.5 is
///     folded into the accumulated term itself (see volume_single /
///     get_gradient), so the extracted constant is always exactly (pi/2)^1.5.
///     roshambo3.numerics.check_interaction_matrix() reports at query prep
///     whether the matrix is the canonical one, so a caller who has changed it
///     is told rather than silently handed a differently-conditioned sum.
///
///  2. Coordinates are carried pre-multiplied by COORD_SCALE, so the pair term
///     is exactly exp2f(-|delta|^2) with no multiply at all in the innermost
///     loop. __expf(x) is ex2.approx(x * log2(e)) on this hardware anyway, so
///     this makes the operation the machine actually performs explicit instead
///     of hiding it behind a general-purpose exp() call -- and it puts the
///     exponent and a future int16 quantization in one shared frame.
///
///  3. The gradient is divided by S_AA + S_BB *inside* the accumulation (it is
///     folded into the per-term coefficient, so it costs nothing), which keeps
///     the accumulated gradient O(1) instead of O(V) whatever the molecule size.
///
/// The scale lives in the coordinates, so the kernel's translation `t` is in
/// scaled Angstrom: the host launchers multiply lr_t by COORD_SCALE on the way
/// in and optimize_one divides t by COORD_SCALE on the way out, which makes the
/// optimizer trajectory identical to the pre-refactor one rather than merely
/// similar.
///
/// These constants are derived from KAPPA = 2.41798793102 and a carbon radius
/// of 1.7 A. roshambo3/numerics.py holds the authoritative definitions and
/// test/test_numerics_policy.py asserts that these literals still match it.
////////////////////////////////////////////////////////////////////////////////

// 8 * (pi / (2*A))^1.5 with A = KAPPA/1.7^2 = 0.836674
#define ROSHAMBO3_SHAPE_PREFACTOR 20.579529f
// (pi/2)^1.5 -- the color prefactor at the canonical interaction width r = 1
#define ROSHAMBO3_COLOR_PREFACTOR 1.9687012f
// sqrt(alpha * log2(e)) with alpha = A/2 = 0.418337
#define ROSHAMBO3_COORD_SCALE     0.77687365f
#define ROSHAMBO3_INV_COORD_SCALE (1.0f/ROSHAMBO3_COORD_SCALE)
// 1/(2*alpha): turns a color interaction width r into its exponent
// coefficient in the shape frame, i.e. exp(-(r/2)d^2) == exp2(-r*INV_TWO_ALPHA*d'^2)
#define ROSHAMBO3_INV_TWO_ALPHA   1.1952087f
// d/dx exp2(-x) = -ln2*exp2(-x); every gradient term carries -2*ln2
#define ROSHAMBO3_MINUS_2LN2      (-1.3862944f)

// int16 library staging. Coordinates quantize at the wire step (NUMERICS_POLICY 4e) -- [-128.0, +127.99609375] A of range against a corpus maximum of 13.4 A --
// and the weight column at 1/4096, which represents the corpus's {0,1} weights
// exactly. Both are applied to the RAW library coordinates, before the reduced
// frame's COORD_SCALE, which lives in the pose rotation matrix.
#define ROSHAMBO3_STAGE_Q_XYZ       256.0f
#define ROSHAMBO3_STAGE_INV_Q_XYZ   0.00390625f
#define ROSHAMBO3_STAGE_Q_W         4096.0f
#define ROSHAMBO3_STAGE_INV_Q_W     (1.0f/4096.0f)

// How many start orientations each start_mode has. Indexed by start_mode, so
// the count below is also the bound a start_mode has to be range-checked
// against before it reaches this array -- reading past it reads __constant__
// memory, which does not fault.
#define ROSHAMBO3_START_MODE_COUNT 3
__device__ __constant__ int start_mode_n[ROSHAMBO3_START_MODE_COUNT] = {1, 4, 10};

// The signed-permutation precondition the grid kernel's conjugation
// reparameterization rests on, checked rather than commented. See the block
// comment over `start_quaternion` in kernels.cu for what breaks without it and
// where the tolerance comes from. `require_signed_permutation_starts` throws on
// a violation or an out-of-range mode; `signed_permutation_start_count` does the
// same and returns the number of starts checked. Both are host-only and are
// called once per launch, not per pair.
void require_signed_permutation_starts(int start_mode);
int  signed_permutation_start_count(int start_mode);

// The same predicate on one arbitrary unit quaternion (w, x, y, z), which is
// what a new start set would be expressed as.
bool rotation_is_signed_permutation(const float * quat);


// coordinate arrays are [N,D] (x,y,z, weight/could be used as radius)
// The same 4 as a COMPILE-TIME constant.  `D` above is a __constant__, so
// `D == 4` is a runtime test and cannot gate a 128-bit vector load; the
// streaming tiers need to know at compile time that a library atom's three
// components are contiguous and 16-byte aligned.
#define ROSHAMBO3_D 4
__device__ __constant__ int D = ROSHAMBO3_D;

// host-side copy of D, for sizing allocations
#define D_HOST 4


////////////////////////////////////////////////////////////////////////////////
/// int16 on the wire, and why `D` above is no longer the stride of BOTH
/// operands.
///
/// The corpus stores coordinates as int16 quantized at the wire step in Angstrom
/// (format 3, doc/FORMAT.md). The host once dequantized them to float32 and
/// appended a fourth component, so the BUS carried 16 bytes per atom for a
/// record the disk holds in 6 -- and every storage tier on the device then
/// re-quantized straight back to int16 (`COORD_FETCH`, the staging prologues,
/// `STAGE16`). Sending the shorts stops the round trip. The precondition
/// is exhaustive, not an argument: for all 65,536 int16 values `v`,
/// `(short)__float2int_rn(1000.0f * (0.001f * (float)v)) == v`
/// At the current wire step: `(short)__float2int_rn(256.0f * (0.00390625f * (float)v)) == v`.
///
/// **The two constants called COORD_SCALE.** The FORMAT's is the wire step, and it is
/// `ROSHAMBO3_STAGE_INV_Q_XYZ` above. The reduced frame's is
/// `ROSHAMBO3_COORD_SCALE = 0.77687365f`. A dequantization must be written
///
///     cscale * (ROSHAMBO3_STAGE_INV_Q_XYZ * (float)v)
///
/// with those parentheses -- two roundings, exactly as the host's
/// `float32(v) * float32(WIRE_QUANT_STEP)` followed by the kernel's `cscale *` was.
/// Folding the two into one constant is ONE rounding where there were two, and
/// `doc/NUMERICS_POLICY.md` 4c records that costing 17.7% of pairs their bits.
/// `r3_fetch` below is the only place the first multiply is written.
///
/// **There is no int16 float4 analogue.** `short3` has `alignof 2` and `short4`
/// needs the 8-byte record this deletes, so the three components are three
/// 2-byte loads. Whether a PLANAR int16 tile reads better than this interleaved
/// one was measured separately; this is the straightforward interleaved form,
/// [tile][atom][lane][3], so that the tile transform, the row arithmetic and
/// the count vector are all unchanged.
///
/// **The fourth component is not lost, it was never a record.** It held
/// `slot < n_real` on the shape plane and `slot - color_offset < n_color` on
/// the color plane (`format/dataset.py::combine_shape_color`) -- a per-SLOT
/// expansion of the count vector, exactly as the deleted type plane was.
/// Every loop that reads it is bounded by those same counts, so the value it
/// yields is always exactly 1.0f; `r3_shape_weight` below reconstructs it from
/// the count word rather than carrying it, and the one loop that is NOT
/// count-bounded (`volume_test_kernel`'s, which passes the PADDED width for the
/// real count) keeps the float32 record and says so at its call site.
#define ROSHAMBO3_D16 3

/// The QUERY record's width in the analytic kernel's SHARED copy: three int16
/// planes, x, y, z, interleaved per atom -- the same record `ROSHAMBO3_D16`
/// names for the library, for the same reason and at the same wire step.
///
/// **The weight is not lost, it was never a record** (see above, and the NOW
/// note over `get_gradient`): every loop that could read it is bounded by the
/// real atom counts, so the value it yields is always exactly 1.0f. An int16
/// query record therefore REQUIRES `NOW`, exactly as the three-plane staged
/// library record does, and `optimize_pairs_analytic` refuses the combination
/// by name rather than reading a plane that is not there.
///
/// The shared copy holds the query AFTER the start-mode rotation. Shipped
/// starts are signed permutations, so requantization is the identity on snapped
/// query coordinates at either wire step. Other
/// rotation sets would require a representation check. General off-grid inputs
/// can move by half the wire step per component. See NUMERICS_POLICY 4e.
#define ROSHAMBO3_QUERY_D16 3


/// What a pair-kernel launch actually got from the driver, recorded so a
/// sweep cell can carry it rather than infer it.
///
/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is the quantity that makes a
/// carveout sweep readable: the carveout is not a floor the driver overrides,
/// it decides the shared PARTITION, and `blocks/SM = floor(partition / demand)`
/// follows from that. One sweep ran CSTAGE at one carveout for its arms and
/// another for its tie point -- 32 KB against 64 KB, i.e. one resident block
/// against two or three -- and read the difference as a CSTAGE effect. With
/// this column beside every timing that misreading is not available.
///
/// Defined in `interface.cpp` (it writes the runtime stats block) and called
/// from `kernels.cu`'s launch macros, which is why it is declared here: the two
/// are separate translation units.
///
/// `carveout` is -1 where the launcher does not set one.
void r3_record_launch(int blocks_per_sm, unsigned long long shared_bytes,
                      int carveout, int nthreads);

/// The width of one operand record, in ELEMENTS, for each element type the
/// kernels accept. `float` is the four-component record (x, y, z, weight);
/// `short` is the int16 three-component one. `D`/`D_HOST` above mean the FLOAT
/// record's width and nothing else now -- every site that used them for a
/// LIBRARY stride goes through this.
template<typename Elt> struct r3_record_width;
template<> struct r3_record_width<float> { enum { value = ROSHAMBO3_D };   };
template<> struct r3_record_width<short> { enum { value = ROSHAMBO3_D16 }; };

/// ONE coordinate, in the units the caller's `cscale` expects, from either
/// record. The int16 arm applies the FORMAT's inverse quantization and nothing
/// else; the reduced frame's scale is applied by the caller, to this result,
/// which is what keeps the two roundings separate (see above).
///
/// The load-hint overloads exist because the field kernel's streaming tiers
/// pick one per tier, and a hint is a property of the load, not
/// of the element type.
///
/// `__CUDACC__`, and it is not decoration: `interface.cpp` is a .cpp compiled by
/// the HOST compiler and it includes this header. CUDA's host headers define
/// `__device__` and `__forceinline__` away to nothing there, so a `__device__`
/// function whose body names `__ldcg`, `__ldcs` or `__float2int_rn` becomes an
/// ordinary host function calling an intrinsic g++ has never heard of. Every
/// other declaration in this file is either `__host__ __device__` with a plain
/// C++ body or a prototype, which is why the trap has not been sprung before.
#ifdef __CUDACC__
__device__ __forceinline__ float r3_fetch(const float * p){ return *p; }
__device__ __forceinline__ float r3_fetch(const short * p){
    return ROSHAMBO3_STAGE_INV_Q_XYZ*(float)*p;
}
__device__ __forceinline__ float r3_fetch_cg(const float * p){ return __ldcg(p); }
__device__ __forceinline__ float r3_fetch_cg(const short * p){
    return ROSHAMBO3_STAGE_INV_Q_XYZ*(float)__ldcg(p);
}
__device__ __forceinline__ float r3_fetch_cs(const float * p){ return __ldcs(p); }
__device__ __forceinline__ float r3_fetch_cs(const short * p){
    return ROSHAMBO3_STAGE_INV_Q_XYZ*(float)__ldcs(p);
}

/// The int16 short for slot component `*p`, ALREADY QUANTIZED -- what the
/// staging prologues want, because staging is a re-quantization at the same
/// step and the exhaustive precondition above makes that the identity. The
/// float arm performs the quantization the shorts arrive pre-quantized in, so
/// the two arms stage the same bits.
__device__ __forceinline__ short r3_quantize(const float * p){
    return (short)__float2int_rn(ROSHAMBO3_STAGE_Q_XYZ*(*p));
}
__device__ __forceinline__ short r3_quantize(const short * p){ return *p; }
#endif  // __CUDACC__


////////////////////////////////////////////////////////////////////////////////
/// The transposed library tile.
///
/// The shipped library array is packed per molecule -- [mol][atom][4] -- so the
/// 32 threads of a warp, which hold 32 *different* library molecules, read 32 addresses
/// NmolB*16 bytes apart. Every warp-wide float4 load therefore touches 32
/// distinct sectors instead of the 16 a fully coalesced load would (measured:
/// 31.92 sectors/request, and L1TEX at 68.9% -- the highest-utilized
/// pipe in the shipped binary, above FMA's 51.6%).
///
/// The fix is a pure relabelling of the same bytes: store library molecules in groups of
/// ROSHAMBO3_BTILE, transposed, as
///
///     [tile][atom][lane][4],   tile = row / BTILE, lane = row % BTILE
///
/// so the 32 lanes of a warp read 32 *consecutive* float4s -- 512 contiguous
/// bytes, exactly 16 sectors. A second variant goes one step further and stores
/// a tile *planar*,
///
///     [tile][atom][component][lane]
///
/// so a warp-wide load of one component covers 128 contiguous bytes with every
/// byte used, instead of 512 bytes of which one quarter is used. Same total
/// distinct sectors; four times fewer L1TEX wavefronts, which is the quantity
/// the grain optimum was found to track.
/// Nothing about the arithmetic moves in either: the same
/// values are read in the same order and summed in the same order, so the
/// scores are bit-identical (verified). Only the stride
/// between two consecutive atoms of one molecule changes, from D to BTILE*D.
///
/// This is a data-layout change, not a grain change:
/// it does not need cooperating lanes and therefore does not reassociate the
/// gradient sum. See doc/NUMERICS_POLICY.md section 2 for why that distinction
/// is the whole point.
///
/// The tile width must be the warp size for the coalescing to be exact.
#define ROSHAMBO3_BTILE 32


////////////////////////////////////////////////////////////////////////////////
/// The per-conformer COUNT VECTOR, and why there is no library type plane.
///
/// The corpus has always stored a per-conformer record -- `n_real` plus one
/// uint8 feature count per type. The host once EXPANDED that into a
/// `[n_rows, L] int32` per-slot type plane, shipped it, and the kernel then
/// linearly SCANNED it back into the same run boundaries it was built from
/// (`j = NB; while (type[j] == 0) ++j; ... while (type[j] != 0) ++j;`). A
/// shape-only screen shipped a plane of zeros for the privilege: 10.5 MB per
/// 65,536-conformer launch at L = 40, uploaded unconditionally.
///
/// The plane never carried information the counts do not. It carried it per
/// SLOT instead of per RUN. So the counts travel instead, one 64-bit word per
/// conformer:
///
///     byte 0     n_real            -- the real (heavy) atom count. This IS the
///                                     old `molB_num_atoms`; there is one
///                                     number now, not two that agree by
///                                     convention.
///     byte t     count of type t   -- for t in 1 .. N_features-1
///     byte 7     n_color           -- REDUNDANT: sum(bytes 1..6). It makes the
///                                     color bound one mask read instead of a
///                                     six-way sum, and it gives `roshambo3
///                                     verify` something to check the vector
///                                     against.
///
/// `N_features` is the corpus's own type count -- 1 for a shape-only corpus, 7
/// for a color one -- which is already what the kernel is passed
/// (`interaction_r.shape[0]`, type 0 being the ordinary-atom row). Byte 7 is
/// spent on the redundant sum, so seven type bytes is the ceiling and the
/// launcher refuses more rather than silently aliasing the sum.
///
/// The array is `[n_t_rows]` and is **NOT tiled**: a warp holds 32 consecutive
/// library rows, so 32 lanes read 32 consecutive 8-byte words -- 256 contiguous
/// bytes -- which is already the coalesced shape the tile transform exists to
/// produce for the [atom][lane][4] coordinate record. Tiling a per-conformer
/// scalar would scatter it.
///
/// A conformer's color slots begin at `color_offset` and run for `n_color`.
/// That offset is a LAUNCH-WIDE scalar and not part of the word, because every
/// launch is single-bucket: `format/dataset.py::combine_shape_color` puts a
/// library conformer's features at the shape plane's PADDED width, which is the
/// group's `L_shape`. The QUERY plane is built by `engine/query.py` and puts
/// them immediately after that query's real atoms, so the query's offset is its
/// own `n_real` -- the two planes do not agree and a single plumbed offset
/// would be wrong about one of them. `compute_self_overlaps_gpu` takes
/// `color_offset < 0` to mean "at the real atom count", which is the query's
/// case.
#define ROSHAMBO3_MAX_FEATURES 7

typedef unsigned long long r3_counts_t;

/// The count of type `t`, 0 <= t < 8.
__host__ __device__ inline int r3_count(r3_counts_t w, int t){
    return (int)((w >> (8*t)) & 0xFFull);
}

/// Byte 0: the real atom count. What `molB_num_atoms` used to be.
__host__ __device__ inline int r3_n_real(r3_counts_t w){
    return (int)(w & 0xFFull);
}

/// Byte 7: the redundant color total.
__host__ __device__ inline int r3_n_color(r3_counts_t w){
    return (int)((w >> 56) & 0xFFull);
}

/// The feature type of the `u`-th slot of this conformer's color run, or 0 for
/// anything outside it -- which is exactly what the expanded type plane held at
/// that slot, including the shape plane's padding (u < 0) and the color plane's
/// (u >= n_color). Positional, because the runs are contiguous and ascending in
/// type: `pharmacophore.py::generate_color_atoms` sorts into FEATURES_ENUM
/// order and asserts that as a postcondition, and the corpus builder asserts it
/// per molecule.
///
/// At most six compares, unrolled, on a value already in a register.
__host__ __device__ inline int r3_slot_type(r3_counts_t w, int u, int N_features){
    if (u < 0) return 0;
    int acc = 0;
    #pragma unroll
    for (int t = 1; t < ROSHAMBO3_MAX_FEATURES; ++t){
        if (t >= N_features) break;
        acc += r3_count(w, t);
        if (u < acc) return t;
    }
    return 0;
}

/// What the FOURTH coordinate component held, from the count word.
///
/// `format/dataset.py::combine_shape_color` writes `slot < n_real` over the
/// shape plane and `slot - color_offset < n_color` over the color plane, the
/// two masks INDEPENDENT so a stored pseudo-atom is never read into another
/// molecule's slot. That is a per-SLOT expansion of exactly the
/// bytes this word carries -- the same relationship the deleted type plane
/// had -- so the int16 record drops the column and this reproduces it, bit for
/// bit, from the record that remains.
///
/// Every loop that reads a shape weight is bounded by `n_real` and every loop
/// that reads a color one is bounded by a real run, so on a count-bounded loop
/// this is always exactly 1.0f. It is written as the general mask anyway,
/// rather than as a `return 1.0f` with the argument in a comment, because a
/// future loop that is NOT count-bounded would then be wrong rather than
/// merely unproven.
__host__ __device__ inline float r3_slot_weight(r3_counts_t w, int cblo, int j){
    const int u = j - cblo;
    return ((j >= 0 && j < r3_n_real(w)) || (u >= 0 && u < r3_n_color(w)))
           ? 1.0f : 0.0f;
}

/// One operand slot's weight, from whichever record the operand is in: the
/// float32 record's fourth component, or the int16 record's reconstruction
/// above. There is no primary template, so a third element type is a compile
/// error rather than a silent default.
template<typename Elt> struct r3_weight;
template<> struct r3_weight<float> {
    __device__ static __forceinline__ float get(const float * p, int off,
                                                r3_counts_t, int, int){
        return p[off];
    }
};
template<> struct r3_weight<short> {
    __device__ static __forceinline__ float get(const short *, int,
                                                r3_counts_t w, int cblo, int j){
        return r3_slot_weight(w, cblo, j);
    }
};



////////////////////////////////////////////////////////////////////////////////
/// Device-side output reduction.
///
/// The dense output is (n_q, n_t, 20) float32. "At 1e11 x 80 B the current
/// design writes 8 TB; even 4-byte score-only output is 400 GB." So the
/// survivors have to be selected on the device, not on the host. ``OutputSink``
/// is the running-threshold + stream-compaction form: a pair whose score
/// reaches ``*threshold`` claims a slot with one ``atomicAdd`` and writes its
/// 20 columns there; everything else is never written at all. "The host then
/// only ever sees ~N' rows, which deletes the ``_prepare_results`` deep-copy
/// cost entirely rather than shrinking it."
///
/// **Determinism.** ``count`` is an atomic, so which slot a survivor lands in,
/// and (on overflow) which survivors are kept at all, are *not* reproducible.
/// The selected top-K is, and that is the property that matters:
///
///  * on a launch that does not overflow, the sink's *set* is exactly
///    ``{rows : score >= *threshold}`` -- a function of the threshold alone,
///    independent of block size, grid size, and of how the library range was
///    split across GPUs or batches;
///  * the final ranking applies ``roshambo3.numerics.rank_order`` (descending
///    score, ties broken by ascending conformer id), which is a *total* order
///    on the survivors, so the top-K set and its order are unique;
///  * the threshold only has to satisfy ``threshold <= (true K-th largest
///    score)`` for the sink to contain every member of the true top-K. Any
///    valid threshold gives the same answer, so the threshold itself may be
///    derived non-deterministically (as the overflow path below does) without
///    making the output non-deterministic.
///
/// A NaN score fails ``>=`` and is therefore dropped rather than ranked.
////////////////////////////////////////////////////////////////////////////////
struct OutputSink {
    float *        scores;      ///< [capacity, 20] survivor rows (device)
    int *          ids;         ///< [capacity] global library row index (device)
    unsigned int * count;       ///< single counter (device); may exceed capacity
    const float *  threshold;   ///< single float (device); pairs with score >= it survive
    int            capacity;
    long           id_base;     ///< added to the kernel's local index to give a global id
};



// The wire values `opt_mode` carries from Python through every host entry
// point below. The device-side template parameter is a SCOPED enum (UpdateRule, in
// kernels.cu) with these same values; the split is deliberate, because a
// scoped enum is what makes OPT and PROBE non-transposable in the
// template lists they share (see the note over optimize_one).
//   0 ADAGRAD_QADD  the shipped Adagrad step on q[1..3] + renormalization
//   1 ADAGRAD_EXP   the same step through the exponential map. THE DEFAULT
//                  on BOTH kernels: the analytic path's shipped cell
//                  and every field-path cell. The analytic path's other cells
//                  still carry mode 0 only and say so.
//   2..4            Adam, QN2 and BFGS: the ANALYTIC path only. Each carries
//                  per-pose auxiliary state (8, 12 and 33 floats) that the field
//                  kernel's 64-register pin has no room for, and each is a
//                  measurement arm rather than a shipping rule
//                  (`experimental.SHIPPING_UPDATE_RULE`).
#define UPDATE_ADAGRAD_QADD 0
#define UPDATE_ADAGRAD_EXP  1
#define UPDATE_ADAM_EXP     2
#define UPDATE_QN2_EXP      3
#define UPDATE_BFGS_EXP     4
#define UPDATE_RULE_COUNT         5

// The structure of the color pair loop, and where the color term is
// evaluated.
//
//   0 MASKED_PADDED  iterate [n_real, L) x [n_real, L) and let the padding
//                    slots' zero weight and zero interaction height null their
//                    own terms. The original structure. **RETIRED --
//                    refused, not instantiated.**
//   1 BOUNDED        the same all-pairs form, bounded to the real color slots.
//                    The bound and nothing else.
//   2 SAME_TYPE      walk the library's color slots; for each, run only the
//                    query's segment carrying that same feature type.
//   3 SAME_TYPE_NOW  the same, with the weight multiply dropped -- which only
//                    this structure made provable at first, because it was
//                    the only one that could not reach a padding slot. The
//                    color analogue of `now`. **Now every surviving
//                    structure is run-bounded and none can reach a padding
//                    slot, so 1, 2 and 3 all drop the multiply and 2 and 3 are
//                    now the same code.** Collapsing them is a follow-up.
//
// 2 and 3 require the CANONICAL interaction matrix (p zero off the diagonal),
// because an off-diagonal interaction is a cross-type term and they do not
// evaluate one. The launcher refuses them otherwise rather than silently
// scoring a different quantity.
//: **0 is retired and is refused, not instantiated.** `MASKED_PADDED` was
//: the only structure that needed a loop reaching into the padding, and it was
//: therefore the last consumer of the 0/1 mask in the 4th coordinate component
//: on the library side. `BOUNDED` was measured reproducing it BITWISE
//: on the canonical matrix over 245,760 score words and over every block of
//: a 200,000-conformer color corpus, so deleting it loses no reachable
//: number. The value is left in place rather than renumbered here; the renumber
//: is a separate cleanup so that a stale integer from an archived script fails
//: loudly instead of silently selecting a different structure.
#define COLOR_LOOP_MASKED_PADDED 0
#define COLOR_LOOP_BOUNDED       1
#define COLOR_LOOP_SAME_TYPE     2
#define COLOR_LOOP_SAME_TYPE_NOW 3
#define COLOR_LOOP_COUNT         4

// The structure measured as the winner, and the one every cell other than
// the unstaged qvec+now cell is instantiated with. The arms live on that one
// cell because it is the only cell a color launch can reach: library staging is
// shape-only, so `stage_width` resolves to 0 for color.
#define COLOR_LOOP_DEFAULT       COLOR_LOOP_SAME_TYPE_NOW

// Where the color term is evaluated.
//
//   0 IN_GRADIENT  in every gradient call, so the pose is optimized against the
//                  mixed objective. Mode 3, combo-optimal. The earlier
//                  default, at N_starts*(steps+1) = 104 evaluations per pair.
//   1 AT_POSE      the pose is optimized on shape alone and the color term is
//                  evaluated once per start, at the converged pose. Mode
//                  2, and what doc/ARCHITECTURE.md describes. 4 evaluations.
//
// This is a RUNTIME argument, not a template parameter and not a separate
// kernel, so mode 3 stays reachable by an argument rather than by a rebuild.
#define COLOR_PLACEMENT_IN_GRADIENT 0
#define COLOR_PLACEMENT_AT_POSE     1
#define COLOR_PLACEMENT_COUNT       2

// `optimize_overlap_sparse_blockwise_gpu` with the prefix-V diagnostic tap
// switched on by a template parameter. Writes, per (output row, start mode),
// ROSHAMBO3_PROBE_NSLOT floats: the shape Tanimoto after {0,1,2,3,5,10} Adagrad
// updates, then at convergence (nsteps). The 20 score columns are unaffected --
// the tap only reads a value the optimizer already holds in a register at the
// top of each iteration -- and they were checked to
// come back bit-identical to the shipped kernel's over the whole panel.
#define ROSHAMBO3_PROBE_NPREFIX 6
#define ROSHAMBO3_PROBE_NSLOT   7



// Measurement arms: three switches over the same kernel, each one a
// single-factor change to L1TEX *request* volume.
//   qvec    -- read the shared query atom as one 16-byte float4 instead of the
//              three 32-bit LDS the compiler emits. BIT-IDENTICAL (same values,
//              same order, wider load). This is the arm that matters: the shape
//              gradient loop's SASS has zero LDG and three LDS per
//              atom-pair-gradient, so ~94% of the kernel's L1TEX requests are
//              query reads, not library reads.
//   stage_w -- 0 = off; otherwise the block width, and the library molecule is
//              copied into a per-thread planar shared buffer once per pair
//              instead of being re-read from global on every optimizer step.
//              BIT-IDENTICAL. The caller must chunk work items to stage_w.
//   stage_i16 -- quantize the staged library to int16 (the wire-step). A DECLARED
//              numerics change; halves the staged footprint so the buffer fits
//              a full 256-thread block.
//   xhoist  -- the cross-product factorization, the shipped
//              default. A declared, characterized numerics change.
//   cut2    -- > 0 drops atom pairs beyond that reduced squared distance. A
//              DECLARED numerics change: it removes terms from the sum.
//   now     -- drop `*wa*wb` from the shape term. Value-preserving: the shape
//              loops are bounded by the REAL atom counts, and every weight a
//              shape loop can reach is exactly 1.0f. Verified bitwise, because
//              it changes what ptxas may contract into an FFMA.
//   stage_p3-- store only x,y,z in the int16 staged record: 6 bytes per atom
//              instead of 8, which is what keeps two blocks per SM resident at
//              L = 32. Requires `now` and `stage_i16`.
// The library's per-slot type plane and per-conformer atom count are one
// `r3_counts_t` per conformer (see the count-vector note above);
// `library_color_offset` is where a library conformer's color run begins, i.e.
// the group's padded shape width.
// `EltB` is the LIBRARY record's element type -- `short` for the int16
// wire record (the corpus path), `float` for the four-component one (the
// oracle wrappers, the hand-built fixtures, and any caller whose coordinates
// are not on the corpus's wire-step grid, for whom quantizing would be a
// numerics change rather than an encoding). Both are instantiated at the
// bottom of kernels.cu; a third element type is a link error.
template<typename EltB>
void optimize_pairs_analytic(
        const float * molAs, const int * molA_types, const int * molA_num_atoms, int NmolA,
        const EltB * molBs, const r3_counts_t * molB_counts, int NmolB,
        int library_color_offset,
        const int * out_library_row,
        const int * wi_qrow, const int * wi_tstart, const int * wi_tcount, int n_work_items,
        const float * rmat, const float * pmat, int N_features,
        float * scores, bool optim_color, float lr_q, float lr_t, int nsteps, float mixing_param,
        int start_mode, int device_id, bool legacy_combo_gradient,
        const float * va_all, const float * vb_all,
        int tile_mode, bool qvec, int stage_w, bool stage_i16, bool xhoist, float cut2,
        bool now, bool stage_p3, int opt_mode = 0,
        int color_loop = COLOR_LOOP_DEFAULT,
        int color_placement = COLOR_PLACEMENT_IN_GRADIENT,
        int * winning_starts = nullptr, float * start_objectives = nullptr);




// Self-overlap precompute. See kernels.cu for why this replaces
// the per-(pair, start_mode) volume_single() calls optimize_one() used to make,
// and why cscale must be ROSHAMBO3_COORD_SCALE-on-raw-coordinates for a library
// but 1.0f-on-prescale_coords_gpu()-output for a query. out is [n_mols, 2]:
// reduced [S_shape, S_color], in the same frame volume_single returns (multiply
// by ROSHAMBO3_SHAPE_PREFACTOR / ROSHAMBO3_COLOR_PREFACTOR for a physical
// volume). mols/mol_types/mol_num_atoms/out must already be device pointers.
// `tiled` selects the transposed 32-lane library layout (see ROSHAMBO3_BTILE
// above); `mols`/`mol_types` are then [tile][atom][lane][...] and `n_mols` is
// still the real molecule count (the trailing lanes of the last tile are
// padding and are never visited).
// `mol_counts` is one `r3_counts_t` per molecule -- byte 0 the real atom
// count, bytes 1.. the per-type feature counts -- replacing both the per-slot
// type plane and the separate atom-count array. It is NOT tiled even when
// `mols` is: a per-conformer scalar is already coalesced, and tiling it would
// scatter it. `color_offset` is where a molecule's color run begins;
// **negative means "at its own real atom count"**, which is the query plane's
// layout (`engine/query.py`), while a library's is the group's padded shape
// width (`format/dataset.py::combine_shape_color`). The two differ, which is
// why this is an argument rather than an assumption.
// `Elt` is the record's element type. The QUERY self-overlap passes
// `float` -- it is `prescale_coords_gpu`'s output, in the reduced frame
// already, and it is one record per query rather than the place the bytes are.
// The LIBRARY self-overlap passes `short`, the int16 wire record, and is
// bit-identical to the float32 arm on any coordinate the corpus can store
// (see the int16 note over `r3_fetch`).
template<typename Elt>
void compute_self_overlaps_gpu(const Elt * mols, const r3_counts_t * mol_counts,
                               int Nmol, long n_mols, int color_offset,
                               const float * rmat, const float * pmat, int N_features, bool optim_color,
                               float cscale, float * out, int device_id, int tile_mode=0,
                               int color_loop = COLOR_LOOP_DEFAULT);

// Prescale n_mols molecules' worth of raw coordinates (Nmol atoms each) by
// ROSHAMBO3_COORD_SCALE, leaving the weight column untouched -- the exact
// per-atom operation start_mode_transform's scaling loop performs. Used before
// compute_self_overlap(s)_gpu with cscale=1.0f for a query's self-overlap, so
// the result is bit-identical to what start_mode n=0 already put in shared
// memory. mols/out must already be device pointers, out sized n_mols*Nmol*4
// floats.
void prescale_coords_gpu(const float * mols, float * out, int Nmol, long n_mols, int device_id, bool query_wire_grid = false);


////////////////////////////////////////////////////////////////////////////////
/// The GRID path in production.
///
/// The query's shape field is precomputed once per query onto a lattice and the
/// Tier-2 pose search reads (F, dF/du, dF/dv, dF/dw) from a hardware-filtered
/// half4 3D texture instead of summing over the query's atoms. The reported
/// score is NOT the grid's: every converged pose is rescored with the exact
/// analytic pair sum (a cascade), so the interpolation error decides which
/// pose is chosen and nothing else.
///
/// Ship settings, as measured:
///   * `tex4hnb`  -- half4 node texel, one TEX, no software bounds test (the
///                   texture's cudaAddressModeBorder returns 0 outside the box,
///                   which is the field's true asymptote).
///   * h = 0.8 A  -- a pre-registered gate passes at 0.8 and fails at 1.0;
///                   h = 0.5 measured costing 4.68x at 45-atom library molecules.
///   * (WA = 4) + UPD = 2 -- four cooperating lanes per conformer over the atom
///                   loop, and the seven-component Adagrad update split one
///                   component per lane (1.43-1.48x in the probe).
///   * int16 3-plane staging of the library, carveout set EXPLICITLY.
///
/// Everything below is in the reduced frame (see above): the lattice spacing
/// stored in `hs` is ROSHAMBO3_COORD_SCALE * h_angstrom.
////////////////////////////////////////////////////////////////////////////////

// Lattice geometry, host- and device-visible. `nx/ny/nz` are CELL counts; the
// node array is (nx+1)(ny+1)(nz+1) texels.
struct FieldGeometry {
    float ox, oy, oz;   // reduced-frame origin (node 0,0,0)
    float hs;           // reduced-frame lattice spacing
    int   nx, ny, nz;   // cells per axis
};

// What the kernel needs: the geometry plus the texture handle. One per query.
struct QueryField {
    unsigned long long tex;   // cudaTextureObject_t
    float ox, oy, oz;
    float inv_hs;
    int   nx, ny, nz;
};

// Choose the lattice for one query from its raw (Angstrom) coordinates.
// `pad_a` is the margin in Angstrom on every side (8.0 in the measurements).
FieldGeometry plan_field_lattice(const float * molA_host, int n_real, int NmolA,
                          float h_angstrom, float pad_a);

// The same lattice choice over an arbitrary RUN of query slots, which is
// what a per-plane color field needs -- the query's features of one type are a
// contiguous run at [qcs[t], qce[t]) rather than the prefix [0, n_real).
// `plan_field_lattice` is this function over [0, n_real).
FieldGeometry plan_field_lattice_range(const float * molA_host, int i0, int i1,
                          int NmolA, float h_angstrom, float pad_a);

// Fill `out_nodes_dev` -- (nx+1)(ny+1)(nz+1) half4 texels, 8 B each -- with the
// prefiltered field and its analytic gradient at every node, on the device.
// `molA_dev` is one query's RAW coordinates [NmolA, 4] on the device.
void build_query_field(const float * molA_dev, int n_real, int NmolA,
                         FieldGeometry G, void * out_nodes_dev, int device_id);

// The same fill over a run of slots and at a non-unit Gaussian coefficient.
// `fac` is the same `a * ROSHAMBO3_INV_TWO_ALPHA` the analytic color loop uses,
// so the field and the pair sum it replaces are the same function of the pose.
// The shape build is this function at fac = 1 over [0, n_real), and is compiled
// as its own instantiation so its arithmetic is character-identical to what
// shipped.
void build_query_color_field(const float * molA_dev, int i0, int i1, int NmolA,
                         FieldGeometry G, float fac, void * out_nodes_dev,
                         int device_id);

// The production grid kernel. Same work-item contract as
// optimize_pairs_analytic, except that a work item owns
// `COMBO_NTHREADS / (4 * start_groups_per_row)` output rows:
// four lanes cooperate on one (query, library) pair, and
// `start_groups_per_row` such groups may share that pair and split its start
// orientations.
//   upd       0 = every lane runs all seven Adagrad components (the control),
//             2 = component c on lane c % 4 (the shipped arm),
//             3 = 2 + the butterfly all-reduce replaced by a reduce-scatter.
//   carveout  the shared/L1 split, 0..100. ALWAYS set explicitly.
//   tier      WHERE the library coordinates live between optimizer steps.
//             0 = staged in SMEM (shipped), 1 = streamed from global with an
//             ordinary load, 2 = streamed through __ldcg (L2-only, never
//             allocates L1). 1 and 2 request zero staged shared memory. All
//             3 = register-resident (APL = ceil(L/WA) atoms/lane, templated).
//             Tiers 0-2 are bit-identical; tier 3 is too.
//   minb      the __launch_bounds__ min-blocks, i.e. the ptxas register
//             budget (4 -> 64 regs, 3 -> 85, 2 -> 128, 1 -> 255). Only tier 3
//             uses anything but 4.
//   start_groups_per_row
//             `S`. How many lane groups SHARE one library row and split
//             the start orientations between them. 1 (the default, and what
//             shipped) gives every group its own row and one group loops all
//             N_starts. S > 1 makes a block cover COMBO_NTHREADS/(4*S) rows -- so the
//             work-item `tcount` must be built to the same width -- and shrinks
//             the TIER 0 staged record per unit of work by S. Constrained to a
//             power of two with 4*S <= 32 so the per-row best stays an in-warp
//             reduction, and to S <= the mode's start count. Not instantiated
//             for tier 3; see the launcher for why.
//   opt_mode  the update RULE (not update_lanes). 0 = the additive Adagrad
//             step plus a renormalization, 1 = the same step through the
//             exponential map, which is the DEFAULT. 2..4 are refused here and
//             instantiated on the analytic path only.
// `EltB` is the LIBRARY record's element type -- see the note over
// `optimize_pairs_analytic`. Both are instantiated at the bottom of kernels.cu.
// Shared by the native API boundary and launcher so unsupported diagnostic
// requests are refused before any device allocation or texture construction.
inline bool field_start_metadata_supported(
        int tile_mode, int lanes_per_group, int coord_residency, int update_lanes,
        int update_rule, int query_color_residency, int color_ballot,
        int interleave, int color_residency, int color_stage,
        int start_groups_per_row, int stream_min_blocks_per_sm, int color_mode) {
    const int minb = stream_min_blocks_per_sm ? stream_min_blocks_per_sm : 4;
    const bool common = tile_mode > 0 && lanes_per_group == 4 && coord_residency == 0
        && update_lanes == 2 && update_rule == 1 && query_color_residency == 0
        && color_ballot == 0 && interleave == 0;
    const bool global = color_residency == 1 && color_stage == 0;
    const bool staged = color_residency == 0 && color_stage == 48;
    const bool shape_or_rescore = color_mode >= 0 && color_mode <= 1
        && global && minb == 4 && start_groups_per_row == 1;
    const bool combo = color_mode == 2 && (global || staged) && minb == 2
        && (start_groups_per_row == 1 || start_groups_per_row == 4);
    return common && (shape_or_rescore || combo);
}

template<typename EltB>
void optimize_pairs_field(
        const float * molAs, const int * molA_num_atoms, int NmolA,
        const EltB * molBs, const r3_counts_t * molB_counts, int NmolB,
        int library_color_offset,
        const int * out_library_row,
        const int * wi_qrow, const int * wi_tstart, const int * wi_tcount, int n_work_items,
        const QueryField * grids_dev,
        float * scores, float lr_q, float lr_t, int nsteps, int start_mode,
        const float * va_all, const float * vb_all,
        int upd, int carveout, int device_id, int tier, int minb, int opt_mode,
        int start_groups_per_row,
        // Everything the color planes need, and nothing a shape-only
        // launch has to name. `color_mode` is 0 (no color), 1 (mode 2,
        // color at the converged pose) or 2 (mode 3, color in the gradient).
        const int * molA_types,
        const float * rmat, const float * pmat, int N_features,
        float mixing_param, bool legacy_combo_gradient, int color_mode,
        const QueryField * color_grids,
        // The lane layout and the register budget, both runtime so they can
        // be swept.
        //   lanes_per_group  WA -- lanes cooperating on one row's atom loop
        //   grid_minb        MINB -- __launch_bounds__'s min blocks per SM,
        //                    which is what SETS the register count
        //   color_load_hint  CLOAD -- the color channel's own load hint
        //   interleave       ILV -- 0/1 dual gradient accumulators, 2/3 exact
        //                    overlap-prepass plus one gradient accumulator,
        //                    4/5 one accumulator with prior-pose prefactors;
        //                    odd values stagger the channel order by warp.
        //   color_residency  CTIER -- where the library COLOR coordinates
        //                    live between steps. Its own axis because
        //                    `coord_residency` governs library SHAPE atoms and
        //                    nothing else, and color is the set with the
        //                    strongest reuse argument.
        //   color_stage      CSTAGE -- the fixed slot count the SMEM and
        //                    register color arms are sized against, with a
        //                    global fallback for the tail.
        //   stream_min_blocks_per_sm  MINB on the STREAMING and staged tiers.
        //                    0 means the pinned 4 that shipped. Its own
        //                    argument because `min_blocks_per_sm` reaches the
        //                    kernel only under REGISTER, and making it reach
        //                    the others would silently change the shipped shape
        //                    path at two of nine buckets.
        int lanes_per_group, int color_residency, int color_stage,
        int interleave, int stream_min_blocks_per_sm,
        int query_color_residency, int query_shape_width,
        int color_ballot, int library_shape_width,
        // The LIBRARY tile's layout, with the same sign convention the
        // analytic launcher and `compute_self_overlaps_gpu` already use --
        // > 0 the interleaved tile [tile][atom][lane][component], < 0 the
        // planar one [tile][atom][component][lane]. It is `TILEB`'s sign, and
        // it reaches the kernel as a template argument, so a request whose
        // cell is not instantiated is refused by the dispatch ladder's throw
        // rather than silently run at the other layout. See ROSHAMBO3_BTILE.
        int tile_mode = 1, int * winning_starts = nullptr,
        float * start_objectives = nullptr);

// wrapper functions for the testing framework
void test_overlap_gpu(const float * molAs, int NmolA, long nA,
                      const float * molBs, int NmolB, long nB, float * V);

void test_gradient_gpu(const float * molA, int NmolA,
                       const float * molB, int NmolB,
                       const float * q, const float * t, float * gradient);

// Private sampler diagnostic: production fetch over supplied half4 nodes.
void numerics_sample_field_gpu(QueryField field, const float* points,
                               int count, float* samples);

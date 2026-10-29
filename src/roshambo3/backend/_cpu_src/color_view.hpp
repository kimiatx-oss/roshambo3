// The one read-only CPU color view, and the one segment-offset
// derivation every later scalar, SIMD and field color path is meant to share.
//
// WHY THIS EXISTS, RATHER THAN LETTING EACH KERNEL DECODE ITS OWN OFFSETS
// -------------------------------------------------------------------------
// A conformer's color record (the merged record, `doc/COLOR_CONTRACT.md`
// section 2) is real atoms `[0, n_real)`, then color pseudo-atoms grouped in
// one contiguous ascending run per feature type, then padding. Any kernel
// that wants to evaluate "every query atom of type t against every library
// atom of type t" has to know where type t's run starts and how long it is.
// If every kernel re-derives that from the count vector, this is exactly
// what happens eventually: two derivations agree on every
// case anyone tested and disagree on one that was not, and the disagreement
// looks like a wrong score rather than a wrong offset.
//
// THE ASYMMETRY THIS HEADER DOES NOT PAPER OVER
// -------------------------------------------------------------------------
// The QUERY carries an explicit per-atom type array (built by
// `engine/query.py`) and has to be DECODED into counts, with the contiguity
// invariant checked as it is decoded -- `build_color_view_from_types` below.
// The LIBRARY carries the count vector directly: there is no
// per-atom type array to decode, the counts already are the input --
// `build_color_view_from_counts` below. These are two different functions on
// purpose. Calling one twice, or assuming the two are interchangeable, is
// the trap this ticket's text names explicitly.
//
// Both decoders end at the SAME offset arithmetic
// (`segment_offsets_from_counts`), which is the one place "how many of each
// type" becomes "where each type starts" -- mirrored, not duplicated, from
// `roshambo3.layout.segment_offsets_from_counts` (Python) and from the CUDA
// convention `r3_slot_type` (`kernels.cuh:432`) already uses: a running sum,
// compared against a slot index. Read there for CONVENTION -- which axis is
// which, where padding sits -- never for the algebra (shared algebra shares
// its typos; carried over from `reference/color.py`'s own header comment).
//
// Every function here is `static inline`, on the same reasoning
// `kernel.hpp` and `score.hpp` give in full -- this header is included,
// transitively, from `interface.cpp` (`-march=x86-64-v3`) and is written so
// that including it from `grain_avx512.cpp` (`-mavx512f`) later, once a SIMD
// color kernel needs it, costs nothing: `static` gives each translation unit
// its own copy under its own name, so there is nothing for the linker to
// choose between regardless of which TUs end up calling it.
// `ColorConformer` itself stays an ordinary (external-linkage) struct, like
// `Conformer` and `PairResult` -- its IDENTITY, not its codegen, has to be
// the same type across any TU that names it.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace r3 {
namespace cpu {

//: Feature types a count vector can carry. Must equal
//: roshambo3.layout.MAX_FEATURES and ROSHAMBO3_MAX_FEATURES (kernels.cuh).
constexpr int MAX_COLOR_FEATURES = 7;

//: Bumped whenever the merged-record wire convention this header reads
//: changes (segment order, padding rule, or the count-vector column
//: meaning) -- stamped into the loaded extension
//: (`__color_abi_version__`) so a caller can tell which convention a binary
//: was built to read, the way `selected_target()` says which ISA it chose.
constexpr int COLOR_ABI_VERSION = 1;

//: The one read-only view of a single color conformer: where its shape run
//: and color run are, and where each feature type's segment is within the
//: color run. Never owns memory -- both coordinate pointers alias the
//: caller's own buffer (the query's or the library's padded plane), exactly
//: as `Conformer` does for shape alone.
struct ColorConformer {
    const float* shape_xyz;  // [n_real * 3], interleaved x,y,z
    int n_real;
    const float* color_xyz;  // [n_color * 3], = shape_xyz + n_real*3
    int n_color;             // total REAL feature count (not a
                             // padded width)
    int n_features;          // width of the interaction matrix this view
                             // was built against
    //: seg_offset[t] is feature type t's start, relative to color_xyz (so
    //: color_xyz + 3*seg_offset[t] is the first coordinate of type t's run).
    //: Index 0 is unused -- type 0 is never a feature.
    int seg_offset[MAX_COLOR_FEATURES];
    int seg_count[MAX_COLOR_FEATURES];
};

// The ONE segment-offset derivation. `counts[t]` for t in [1, n_features) is
// how many slots type t occupies; `offsets[t]` is where that run starts,
// relative to the color run's own beginning. Mirrors
// `roshambo3.layout.segment_offsets_from_counts` (a running sum) and the
// convention `r3_slot_type` (kernels.cuh:432) reads by comparing a slot
// index against the same running sum from the other direction.
static inline int segment_offsets_from_counts(const int* counts, int n_features,
                                              int* offsets /* [n_features] */) {
    int running = 0;
    if (n_features > 0) offsets[0] = 0;  // type 0: never a feature, unused
    for (int t = 1; t < n_features; ++t) {
        offsets[t] = running;
        running += counts[t];
    }
    return running;  // total color count
}

// LIBRARY decoder: `counts_row` is the count vector's row directly
// (column 0 = n_real, column t = type t's count) -- already counts, nothing
// to decode from an explicit per-atom type array, because there is none on
// this side. `xyz` is the conformer's own padded coordinate row (interleaved
// x,y,z, stride `stride_floats` per slot -- 3 for a bare xyz plane, 4 if a
// weight/occupancy component rides along; only the leading 3 are ever read
// here).
static inline ColorConformer
build_color_view_from_counts(const float* xyz, const uint8_t* counts_row,
                             int n_features, int stride_floats = 3) {
    if (n_features < 1 || n_features > MAX_COLOR_FEATURES)
        throw std::invalid_argument("n_features must be in [1, " +
                                    std::to_string(MAX_COLOR_FEATURES) +
                                    "]; got " + std::to_string(n_features));
    ColorConformer v{};
    v.n_features = n_features;
    v.n_real = int(counts_row[0]);
    int counts[MAX_COLOR_FEATURES] = {0};
    for (int t = 1; t < n_features; ++t)
        counts[t] = int(counts_row[t]);
    v.n_color = segment_offsets_from_counts(counts, n_features, v.seg_offset);
    for (int t = 1; t < n_features; ++t)
        v.seg_count[t] = counts[t];
    v.shape_xyz = xyz;
    v.color_xyz = xyz + size_t(v.n_real) * size_t(stride_floats);
    return v;
}

// QUERY decoder: `types` is the explicit per-atom type plane over the WHOLE
// padded row (length >= n_real + n_color; slots at or past the color run's
// end carry type 0, i.e. padding, and are ignored). Validates the same
// "one contiguous ascending run per type" invariant
// `roshambo3.layout.counts_from_types` checks in Python -- checked here
// because this is the side handed an explicit type array to decode, not the
// side handed counts directly (the asymmetry this header's own header
// comment names). Throws `std::invalid_argument` on a violation, exactly as
// the Python decoder raises `ValueError`, so a caller seeing either message
// is looking at the same check.
static inline ColorConformer
build_color_view_from_types(const float* xyz, const int32_t* types, int n_real,
                            int L, int n_features, int stride_floats = 3) {
    if (n_features < 1 || n_features > MAX_COLOR_FEATURES)
        throw std::invalid_argument("n_features must be in [1, " +
                                    std::to_string(MAX_COLOR_FEATURES) +
                                    "]; got " + std::to_string(n_features));
    ColorConformer v{};
    v.n_features = n_features;
    v.n_real = n_real;
    int counts[MAX_COLOR_FEATURES] = {0};
    int last_type = 0;
    // Reviewer correction: a padding (type 0) slot followed later by a
    // nonzero one is a GAP in the color run -- [1, 0, 1] used to `continue`
    // straight past the zero and accept it, deriving offsets that assume
    // contiguity from n_real, so the padding slot silently became the
    // second feature and the real one past it was never read. Once any
    // padding slot is seen, every later slot must ALSO be padding: nonzero
    // slots may occupy only [n_real, n_real+n_color), contiguously.
    bool seen_padding = false;
    for (int k = n_real; k < L; ++k) {
        const int t = int(types[k]);
        if (t == 0) {
            seen_padding = true;
            continue;
        }
        if (seen_padding)
            throw std::invalid_argument(
                "query color run has a padding gap at slot " +
                std::to_string(k) +
                ": an earlier slot in [n_real, L) was "
                "padding (type 0) while this one carries a real feature "
                "type. Real features must occupy exactly "
                "[n_real, n_real+n_color) with no gap.");
        if (t < 0 || t >= n_features)
            throw std::invalid_argument("query color type " +
                                        std::to_string(t) + " at slot " +
                                        std::to_string(k) + " is outside [1, " +
                                        std::to_string(n_features - 1) + "]");
        if (t < last_type)
            throw std::invalid_argument(
                "query color types are not one contiguous ascending run per "
                "type: type " +
                std::to_string(t) + " follows type " +
                std::to_string(last_type) + " at slot " + std::to_string(k) +
                ". See roshambo3.pharmacophore.generate_color_atoms.");
        last_type = t;
        counts[t] += 1;
    }
    v.n_color = segment_offsets_from_counts(counts, n_features, v.seg_offset);
    for (int t = 1; t < n_features; ++t)
        v.seg_count[t] = counts[t];
    v.shape_xyz = xyz;
    v.color_xyz = xyz + size_t(n_real) * size_t(stride_floats);
    return v;
}

}  // namespace cpu
}  // namespace r3

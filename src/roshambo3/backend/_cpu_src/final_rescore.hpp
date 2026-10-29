// Independent fixed-pose finals. No optimizer/start-selection policy.
#pragma once
#include <cstddef>
#include <cstdint>
#include "score.hpp"
#include "color_exact_final.hpp"

namespace r3::cpu {
enum class FinalRescoreMode {
    Accurate = 0,
    ScalarPolynomial = 1,
    WithinPair = 2,
    AcrossPairs = 3
};

struct FinalRescorePair {
    Conformer shape_query_reduced{};
    Conformer shape_library_physical{};
    const ReducedColorConformer* color_query_reduced = nullptr;
    const ColorConformer* color_library_physical = nullptr;
    const ColorInvariantPlan* color_plan = nullptr;
    const PreparedReducedPose* pose = nullptr;
    int query_color_stride = 3, library_color_stride = 3;
    ExactFinalScratch* accurate_color_scratch = nullptr;
};

struct FinalRescoreCross {
    float shape = 0;
    // FP32 canonical accumulation widens once at the existing API boundary.
    // General/unsafe accurate color fallback retains its original double value.
    double color = 0;
};

struct FinalRescoreCounts {
    uint64_t shape_terms = 0, color_terms = 0;
    uint64_t shape_issued_lanes = 0, color_issued_lanes = 0;
    uint64_t accurate_fallback_pairs = 0, packed_coordinate_bytes = 0;
    uint64_t horizontal_reductions = 0;
    uint64_t temporary_allocations = 0, allocated_payload_bytes = 0;
};

// Zero-sized batches accept null pointers. Inputs/poses remain caller-owned;
// outputs preserve input order. Nonfinite real inputs/poses refuse; finite unsafe
// ranges/general matrices use the retained accurate fallback. Padding is unread.
// Self volumes and normalization stay caller-owned
// and unchanged: this API never recomputes self, chooses starts or reports poses.
void final_rescore_batch(const FinalRescorePair*, size_t, FinalRescoreCross*,
                         FinalRescoreMode, FinalRescoreCounts* = nullptr);
// Architecture entry: only finite bounded, canonical-or-shape-only inputs.
void final_rescore_simd(const FinalRescorePair*, size_t, FinalRescoreCross*,
                        FinalRescoreMode, FinalRescoreCounts*);
}

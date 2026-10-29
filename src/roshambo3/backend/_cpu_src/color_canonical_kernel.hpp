// Analytic-only evaluator for the exact canonical identity matrix.
#pragma once

#include <cstdint>
#if defined(__aarch64__)
#include <vector>
#endif

#include "color_conformer_kernel.hpp"
#include "color_invariant_plan.hpp"

namespace r3 {
namespace cpu {

// Accumulate only the lanes already cleared by the caller's pose-dependent
// finite/range screen. Unsafe lanes have already been evaluated by the exact
// scalar path and are left untouched here.
void evaluate_canonical_color_analytic_batch8(
    const ColorConformerBatch8& batch,
#if defined(__aarch64__)
    const std::vector<ColorCanonicalPlane>& planes,
#else
    const ColorInvariantPlan& plan,
#endif
    const float* posed_library_xyz_fp32,
    const float pose_translation[COLOR_CONFORMER_LANES][3],
    const int32_t safe_lane_mask[COLOR_CONFORMER_LANES], bool gradient,
    ColorEvaluation output[COLOR_CONFORMER_LANES],
    uint64_t term_counts[COLOR_CONFORMER_LANES],
    ColorConformerCensus* census = nullptr,
    ColorFieldBatchStats* analytic_stats = nullptr,
    bool factor_canonical_torque = false,
    OptimizerExpMode optimizer_exp_mode = OptimizerExpMode::Baseline);

}  // namespace cpu
}  // namespace r3

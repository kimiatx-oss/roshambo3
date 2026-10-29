// One AVX2 lane owns one independent query/library color alignment.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "color_evaluator.hpp"
#include "color_field_channels.hpp"
#include "color_canonical_four_start.hpp"
#include "reduced_pose.hpp"
#include "optimizer_exp.hpp"

namespace r3 {
namespace cpu {

constexpr int COLOR_CONFORMER_LANES = 8;

struct ColorConformerPairInput {
    const ColorConformer* query = nullptr;
    const ColorConformer* library = nullptr;
    int query_stride = 3;
    int library_stride = 3;
};

// Strongly typed input for shared-pose prepared evaluators. Query coordinates
// must already be in the reduced frame; library coordinates remain physical.
// Pairing this type with a PreparedReducedPose prevents a caller from
// accidentally packing a physical query into a reduced-frame kernel.
struct ReducedColorConformerPairInput {
    const ReducedColorConformer* query = nullptr;
    const ColorConformer* library = nullptr;
    int query_stride = 3;
    int library_stride = 3;
};

// Block-local SoA in [feature][component][lane] layout. Each
// segment is bounded by the real count for that pair; masks cover the padded
// lane and uneven per-type tails.
struct ColorConformerBatch8 {
    int n_features = 0;
    int n_valid = 0;
    int query_count[MAX_COLOR_FEATURES][COLOR_CONFORMER_LANES]{};
    int library_count[MAX_COLOR_FEATURES][COLOR_CONFORMER_LANES]{};
#if defined(__x86_64__)
    // Reduced-frame query-coordinate eligibility is immutable after packing
    // and reused across the optimizer's pose evaluations. Target raw/posed
    // coordinate checks remain pose-dependent in the evaluator.
    std::array<std::array<uint8_t, COLOR_CONFORMER_LANES>, MAX_COLOR_FEATURES>
        canonical_query_coordinates_safe{};
#endif
    int query_max[MAX_COLOR_FEATURES]{};
    int library_max[MAX_COLOR_FEATURES]{};
    size_t query_coordinate_offset[MAX_COLOR_FEATURES]{};
    size_t library_coordinate_offset[MAX_COLOR_FEATURES]{};
    size_t query_mask_offset[MAX_COLOR_FEATURES]{};
    size_t library_mask_offset[MAX_COLOR_FEATURES]{};
    std::vector<float> query_xyz;
    std::vector<float> library_xyz;
    std::vector<int32_t> query_mask;
    std::vector<int32_t> library_mask;
#if defined(__aarch64__)
    // Immutable reduced query coordinates are packed before optimizer steps.
    // Keep a type-local result: an absent target plane must not reject a lane.
    bool query_plane_supported[MAX_COLOR_FEATURES][COLOR_CONFORMER_LANES]{};
#endif
    // Raw target coordinates are immutable after packing. Cache their
    // canonical-path finite/range eligibility here so the prepared FP32
    // transform can fuse only the pose-dependent output checks.
    std::array<std::array<uint8_t, COLOR_CONFORMER_LANES>, MAX_COLOR_FEATURES>
        canonical_library_coordinates_safe{};
};

struct ColorConformerCensus {
    // Flat [query_type, target_type] arrays, with MAX_COLOR_FEATURES stride.
    std::array<uint64_t, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES>
        useful_terms_by_interaction{};
    std::array<uint64_t, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES>
        issued_lanes_by_interaction{};
    // Target-plane counts 0..17, then >17, weighted by each gradient call.
    std::array<uint64_t, MAX_COLOR_FEATURES * 19> target_count_bins{};
    uint64_t vector_evaluations = 0;
    uint64_t compatible_ymm8_evaluations = 0;
    uint64_t compatible_neon4_two_stream_evaluations = 0;
    uint64_t singleton_four_start_evaluations = 0;
    uint64_t scalar_fallback_pairs = 0;
    // Per packed lane, weighted by evaluator call. This retains eligibility
    // identity for paired-plan mask-parity diagnostics.
    std::array<uint64_t, COLOR_CONFORMER_LANES> vector_eligible_lane_calls{};
    std::array<uint64_t, COLOR_CONFORMER_LANES> scalar_fallback_lane_calls{};
    uint8_t last_vector_support_mask = 0;
};

struct ColorFieldBatchStats {
    uint64_t current_four_start_field_groups = 0;
    uint64_t current_four_start_remainder_groups = 0;
    uint64_t samples = 0;
    uint64_t interpolated_samples = 0;
    uint64_t interpolated_terms = 0;
    uint64_t analytic_selected_samples = 0;
    uint64_t analytic_selected_terms = 0;
    uint64_t analytic_fallback_samples = 0;
    uint64_t inactive_fallback_samples = 0;
    uint64_t out_of_grid_fallback_samples = 0;
    uint64_t analytic_fallback_terms = 0;
};

struct ColorConformerPhaseTimes {
    double* pose_transform_seconds = nullptr;
    double* matrix_screen_seconds = nullptr;
    double* scalar_fallback_seconds = nullptr;
    double* pose_soa_pack_seconds = nullptr;
    double* vector_kernel_seconds = nullptr;
    double* field_sample_seconds = nullptr;
};

// Reuse one workspace per active optimizer block/thread. Allocating the pose
// buffers once keeps dynamic pose transforms in the kernel without repeating
// heap allocation on every alignment step.
struct ColorConformerWorkspace {
    ColorFieldSampleWorkspace field_samples;
    // The canonical prepared route owns only FP32 transformed coordinates.
    // Double scratch below is retained for legacy/general and unsafe fallback.
    std::vector<double> posed_library_xyz;
    std::vector<float> posed_library_xyz_fp32;
    // Prepared shared-pose color fields are sampled in physical Angstroms;
    // analytic scoring continues to use the reduced-frame buffers above.
    std::vector<double> posed_library_xyz_physical;
    std::vector<float> posed_library_xyz_physical_fp32;
    ColorCanonicalFourStartWorkspace canonical_four_start_workspace;
};

ColorConformerBatch8
pack_color_conformer_batch8(const ColorConformerPairInput* pairs, int n_pairs,
                            int n_features);
void pack_color_conformer_batch8_into(const ColorConformerPairInput* pairs,
                                      int n_pairs, int n_features,
                                      ColorConformerBatch8& batch);

ColorConformerBatch8 pack_color_conformer_batch8_prepared(
    const ReducedColorConformerPairInput* pairs, int n_pairs, int n_features);
void pack_color_conformer_batch8_prepared_into(
    const ReducedColorConformerPairInput* pairs, int n_pairs, int n_features,
    ColorConformerBatch8& batch);

#if defined(__x86_64__)
// Refresh the reduced-frame query guard after a packed query payload changes.
void refresh_canonical_query_eligibility(ColorConformerBatch8& batch);
#endif

void evaluate_color_conformer_batch8(
    const ColorConformerBatch8& batch,
    const ColorInteractionMatrixView& interaction,
    const float quaternion[COLOR_CONFORMER_LANES][4],
    const float translation_physical[COLOR_CONFORMER_LANES][3], bool gradient,
    ColorEvaluation output[COLOR_CONFORMER_LANES],
    uint64_t term_counts[COLOR_CONFORMER_LANES],
    ColorConformerCensus* census = nullptr,
    ColorConformerPhaseTimes* phase_times = nullptr,
    ColorConformerWorkspace* workspace = nullptr,
    const ColorFieldBundle* const* field_bundles = nullptr,
    ColorFieldBatchStats* field_stats = nullptr,
    bool library_is_preposed = false);

// Shared-pose path. Query coordinates in `batch` must be reduced (sA),
// library coordinates remain physical, and every lane consumes the exact
// FP32 M=sR,t'=s*t bytes from its prepared pose. The plan supplies d'=d/s^2.
void evaluate_color_conformer_batch8_prepared(
    const ColorConformerBatch8& batch, const ColorInvariantPlan& plan,
    const PreparedReducedPose* const poses[COLOR_CONFORMER_LANES],
    bool gradient, ColorEvaluation output[COLOR_CONFORMER_LANES],
    uint64_t term_counts[COLOR_CONFORMER_LANES],
    ColorConformerCensus* census = nullptr,
    ColorConformerPhaseTimes* phase_times = nullptr,
    ColorConformerWorkspace* workspace = nullptr,
    bool use_invariant_plan = true,
    const ColorFieldBundle* const* field_bundles = nullptr,
    ColorFieldBatchStats* field_stats = nullptr,
    bool factor_canonical_torque = false,
    OptimizerExpMode optimizer_exp_mode = OptimizerExpMode::Baseline,
    bool canonical_color_kernel = false);

// Build the transformed library SoA once for fixed-pose kernel diagnostics.
// The matching evaluator call can then use preposed_library=true to skip the
// pose transform and fp32 SoA pack inside each timed native evaluation.
void prepare_color_conformer_batch8_preposed(
    const ColorConformerBatch8& batch,
    const float quaternion[COLOR_CONFORMER_LANES][4],
    const float translation_physical[COLOR_CONFORMER_LANES][3],
    ColorConformerWorkspace* workspace);

}  // namespace cpu
}  // namespace r3

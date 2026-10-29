// Immutable shared-input descriptors for canonical four-start color.
#pragma once
#ifndef R3_COLOR_ANALYTIC_ENVELOPE
#if defined(__x86_64__) || defined(__aarch64__)
#define R3_COLOR_ANALYTIC_ENVELOPE 1
#else
#define R3_COLOR_ANALYTIC_ENVELOPE 0
#endif
#endif
#ifndef R3_COLOR_START_STREAMS
#if defined(__aarch64__)
#define R3_COLOR_START_STREAMS 2
#else
#define R3_COLOR_START_STREAMS 1
#endif
#endif
#ifndef R3_COLOR_START_HYBRID
#define R3_COLOR_START_HYBRID 0
#endif
#ifndef R3_COLOR_START_REDUCED_COORDS
#define R3_COLOR_START_REDUCED_COORDS 0
#endif
#ifndef R3_COLOR_COMPATIBLE_PAIRS
#if defined(__x86_64__)
#define R3_COLOR_COMPATIBLE_PAIRS 1
#else
#define R3_COLOR_COMPATIBLE_PAIRS 0
#endif
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>

#include "color_evaluator.hpp"
#include "color_view.hpp"
#include "color_field_channels.hpp"
#include "reduced_pose.hpp"

namespace r3::cpu {

struct ReducedColorConformerPairInput;
struct PreparedReducedPose;
struct ColorConformerCensus;
struct ColorConformerPhaseTimes;
struct ColorFieldBatchStats;
class ColorInvariantPlan;
enum class OptimizerExpMode;

// SEARCH selects the existing optimizer exponential and value fold. The
// derivative output choice is independent: final selection needs that exact
// value chain while discarding every derivative.
enum class ColorSearchOutput { ValueAndGradient, ValueOnly };

// One active canonical plane for a single conformer. The pointers already
// include the source segment offsets. Coordinates stay in their original
// compact interleaved records and are shared by all four starts.
struct ColorCanonicalFourStartPlane {
    int type = 0;
    const float* query_xyz = nullptr;
    const float* library_xyz = nullptr;
    int query_count = 0;
    int library_count = 0;
    int query_stride = 3;
    int library_stride = 3;
    size_t posed_offset = 0;
    float coefficient = 0.0f;
    float exponent_scale = 0.0f;
    float torque_scale = 0.0f;
    float translation_scale = 0.0f;
};

// Prepared once before an optimizer's step loop. Empty-count planes are
// omitted, so the evaluator's plane loop contains work only. A pair helper may
// pair two descriptors as separate low/high halves, but must retain each descriptor's
// own counts and offsets.
struct ColorCanonicalFourStartDescriptor {
    const ColorInvariantPlan* query_plan = nullptr;
    const ColorFieldBundle* field_bundle = nullptr;
    ColorFieldBatchStats* field_stats = nullptr;
    // Immutable query-plane selection bound before the optimizer loop.
    std::array<int, MAX_COLOR_FEATURES> field_backend{};
    std::array<bool, MAX_COLOR_FEATURES> field_active{};
    int n_features = 0;
    int shape_count = 0;
    int plane_count = 0;
    size_t posed_library_size = 0;
    std::array<int, MAX_COLOR_FEATURES> library_count{};
    std::array<int, MAX_COLOR_FEATURES> library_offset{};
    std::array<ColorCanonicalFourStartPlane, MAX_COLOR_FEATURES> planes{};
    bool immutable_inputs_safe = false;
    // Exact default width/height, diagonal interaction and analytic-only units.
    bool default_parameters = false;
    // N removes one amplitude multiply per issued pair, then restores seven
    // value/gradient components. Bind this arithmetic break-even once.
    bool amplitude_amortized = false;
    // Computed once from active physical library points, for a per-pose
    // conservative transformed-coordinate envelope.
    float library_max_abs[3]{};
};

// Finite FP32 magnitudes have the same order as sign-cleared integer bits.
// Descriptor preparation may run under a different denormal mode from workers:
// floating comparisons must not discard subnormal coordinate maxima here.
inline float canonical_max_abs_finite(float a, float b) {
    uint32_t x, y;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    x &= 0x7fffffffu;
    y &= 0x7fffffffu;
    x = x < y ? y : x;
    std::memcpy(&a, &x, sizeof a);
    return a;
}

// Proves the existing transformed-coordinate bound using the actual supplied
// matrix, including arbitrary finite raw replay matrices. A failed proof is
// ordinary ineligibility: retain the checked per-atom transform path.
inline bool canonical_pose_envelope_safe(
    const ColorCanonicalFourStartDescriptor& descriptor,
    const PreparedReducedPose& pose, float maximum) {
    constexpr float slack =
        1.0f + 16.0f * std::numeric_limits<float>::epsilon();
    for (int row = 0; row < 3; ++row) {
        float bound =
            std::abs(pose.matrix[3 * row]) * descriptor.library_max_abs[0];
        bound +=
            std::abs(pose.matrix[3 * row + 1]) * descriptor.library_max_abs[1];
        bound +=
            std::abs(pose.matrix[3 * row + 2]) * descriptor.library_max_abs[2];
        bound += std::abs(pose.translation[row]);
        // The additive unit dominates all possible subnormal rounding error;
        // relative slack covers positive bound and transform rounding chains.
        if (!(bound * slack + 1.0f <= maximum)) return false;
    }
    return true;
}

// Reused transformed-library scratch, laid out as [component][atom][4 starts].
struct ColorCanonicalFourStartWorkspace {
    bool compatible_pair = false;
    std::vector<float> posed_library_xyz;
    ColorFieldSampleWorkspace field_samples;
};

// Scoped to the decoded work unit that owns descriptors and target buffers.
// No query cache retains this view. Workspace capacity and full pair signature
// are bound once; mutable buffer reuse requires a new checked construction.
class BoundCanonicalColorExecution {
public:
    BoundCanonicalColorExecution(
        const ColorCanonicalFourStartDescriptor* descriptors, int n_conformers,
        const ColorInvariantPlan& plan,
        ColorCanonicalFourStartWorkspace& workspace,
        OptimizerExpMode exponential, bool factor_torque);
    void evaluate(const PreparedReducedPose* const poses[8],
                  ColorEvaluation output[8], uint64_t term_counts[8],
                  ColorConformerCensus* census,
                  ColorConformerPhaseTimes* phase_times,
                  ColorSearchOutput search_output =
                      ColorSearchOutput::ValueAndGradient) const;
    void evaluate_bound_policy(const PreparedReducedPose* const poses[8],
                               ColorEvaluation output[8],
                               uint64_t term_counts[8],
                               ColorConformerCensus* census,
                               ColorConformerPhaseTimes* phase_times,
                               ColorSearchOutput search_output =
                                   ColorSearchOutput::ValueAndGradient) const;

private:
    template <OptimizerExpMode Exponential, bool FactorTorque>
    static void execute_bound(const BoundCanonicalColorExecution& execution,
                              const PreparedReducedPose* const poses[8],
                              ColorEvaluation output[8],
                              uint64_t term_counts[8],
                              ColorConformerCensus* census,
                              ColorConformerPhaseTimes* phase_times,
                              ColorSearchOutput search_output);
    using Evaluate = void (*)(const BoundCanonicalColorExecution&,
                              const PreparedReducedPose* const*,
                              ColorEvaluation*, uint64_t*,
                              ColorConformerCensus*, ColorConformerPhaseTimes*,
                              ColorSearchOutput);
    Evaluate evaluate_;
    const ColorCanonicalFourStartDescriptor* descriptors_;
    int n_conformers_;
    const ColorInvariantPlan* plan_;
    ColorCanonicalFourStartWorkspace* workspace_;
    OptimizerExpMode exponential_;
    bool factor_torque_;
    bool vector_supported_;
};

bool color_canonical_pair_compatible(
    const ColorCanonicalFourStartDescriptor& low,
    const ColorCanonicalFourStartDescriptor& high);

uint8_t evaluate_color_canonical_four_start_ymm8(
    const ColorCanonicalFourStartDescriptor descriptors[2], bool factor_torque,
    const PreparedReducedPose* const poses[8], bool gradient,
    ColorEvaluation output[8], uint64_t term_counts[8],
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode,
    ColorSearchOutput search_output = ColorSearchOutput::ValueAndGradient);

// False means this input/plan is not eligible for the specialized helper; the
// caller keeps the established generic prepared evaluator. Non-finite or
// out-of-range coordinates instead produce a prepared descriptor with
// immutable_inputs_safe=false, which takes the exact fallback at evaluation.
bool prepare_color_canonical_four_start_descriptor(
    const ReducedColorConformerPairInput& pair, const ColorInvariantPlan& plan,
    ColorCanonicalFourStartDescriptor& descriptor);

// Size transformed-target scratch once before the optimizer loop. Helpers
// consume the immutable descriptors and this prepared workspace without
// resizing or repacking inside an iteration.
void prepare_color_canonical_four_start_workspace(
    const ColorCanonicalFourStartDescriptor* descriptors, int n_conformers,
    ColorCanonicalFourStartWorkspace& workspace);

// The architecture helper processes exactly one conformer's four starts.
// A zero return mask means the entire group must use the exact fallback. It
// writes results/term counts only for a returned all-four-safe group.
// `poses`, `output`, and `term_counts` each address four consecutive starts.
// The workspace must have been sized for this descriptor before entry.
uint8_t evaluate_color_canonical_four_start_xmm4(
    const ColorCanonicalFourStartDescriptor& descriptor, bool factor_torque,
    const PreparedReducedPose* const poses[4], bool gradient,
    ColorEvaluation output[4], uint64_t term_counts[4],
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode,
    ColorSearchOutput search_output = ColorSearchOutput::ValueAndGradient);

uint8_t evaluate_color_canonical_four_start_neon4(
    const ColorCanonicalFourStartDescriptor& descriptor, bool factor_torque,
    const PreparedReducedPose* const poses[4], bool gradient,
    ColorEvaluation output[4], uint64_t term_counts[4],
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode,
    ColorSearchOutput search_output = ColorSearchOutput::ValueAndGradient);

uint8_t evaluate_color_canonical_two_stream_neon4(
    const ColorCanonicalFourStartDescriptor descriptors[2], bool factor_torque,
    const PreparedReducedPose* const poses[8], bool gradient,
    ColorEvaluation output[8], uint64_t term_counts[8],
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode,
    ColorSearchOutput search_output = ColorSearchOutput::ValueAndGradient);

// Evaluates one or two independent conformer descriptors in established
// [pair A starts 0..3, pair B starts 0..3] order. Unsafe four-start groups
// take the existing analytic FP64 fallback using the same prepared pose.
void evaluate_color_canonical_four_start_batch(
    const ColorCanonicalFourStartDescriptor* descriptors, int n_conformers,
    const ColorInvariantPlan& plan, const PreparedReducedPose* const poses[8],
    bool gradient, ColorEvaluation output[8], uint64_t term_counts[8],
    ColorConformerCensus* census, ColorConformerPhaseTimes* phase_times,
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode, bool factor_torque,
    ColorSearchOutput search_output = ColorSearchOutput::ValueAndGradient);

}  // namespace r3::cpu

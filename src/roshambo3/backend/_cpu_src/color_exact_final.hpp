// Exact shared-pose color rescoring with worker-local point scratch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "color_evaluator.hpp"
#include "color_invariant_plan.hpp"
#include "reduced_pose.hpp"

namespace r3 {
namespace cpu {

// Counters for the coordinate-preparation portion of one or more exact-final
// calls. `payload_bytes_allocated` counts requested coordinate-buffer payloads,
// not allocator metadata or resident memory. `coordinate_prepare_seconds` is
// populated only when the caller opts into timing.
struct ExactFinalScratchStats {
    uint64_t buffer_growths = 0;
    uint64_t payload_bytes_allocated = 0;
    uint64_t library_points_prepared = 0;
    uint64_t winner_candidates_retained = 0;
    uint64_t winner_point_bytes_copied = 0;
    double coordinate_prepare_seconds = 0.0;
    double winner_retention_seconds = 0.0;
};

// Private to one worker. It retains transformed points only within one
// trajectory and stores input pointer identities as integers for fail-closed
// checks; it never dereferences a caller pointer after a scoring call returns.
// `begin_trajectory` invalidates the prior winner before the next pair. It is
// intentionally not internally synchronized.
class ExactFinalScratch {
public:
    ExactFinalScratch() = default;
    ExactFinalScratch(const ExactFinalScratch&) = delete;
    ExactFinalScratch& operator=(const ExactFinalScratch&) = delete;
    ExactFinalScratch(ExactFinalScratch&& other) noexcept;
    ExactFinalScratch& operator=(ExactFinalScratch&& other) noexcept;

    size_t capacity_points() const noexcept {
        return current_capacity_points_;
    }

    size_t capacity_bytes() const noexcept {
        return current_capacity_points_ * 3 *
               (current_fp32_ ? sizeof(float) : sizeof(double));
    }

    size_t winner_capacity_bytes() const noexcept {
        return winner_capacity_points_ * 3 *
               (winner_fp32_ ? sizeof(float) : sizeof(double));
    }

    void begin_trajectory(const ReducedColorConformer& query_reduced,
                          const ColorConformer& library_physical,
                          const ColorInvariantPlan& invariant_plan,
                          int query_stride, int library_stride);
    void retain_current_pose_for_winner(
        const PreparedReducedPose& pose,
        ExactFinalScratchStats* preparation_stats = nullptr);

private:
    friend ColorEvaluation score_exact_final_color(
        const ReducedColorConformer&, const ColorConformer&,
        const ColorInvariantPlan&, const PreparedReducedPose&, int, int,
        ExactFinalScratch*, uint64_t*, ExactFinalScratchStats*, bool);
    friend ColorEvaluation score_exact_final_color_retained(
        const ReducedColorConformer&, const ColorConformer&,
        const ColorInvariantPlan&, const PreparedReducedPose&, int, int,
        ExactFinalScratch&, uint64_t*);

    struct TrajectoryIdentity {
        uintptr_t query_color = 0;
        uintptr_t library_color = 0;
        uintptr_t coefficients = 0;
        uintptr_t decays = 0;
        int query_n_color = 0;
        int library_n_color = 0;
        int n_features = 0;
        int matrix_stride = 0;
        int query_stride = 0;
        int library_stride = 0;
        bool canonical = false;
        int query_offsets[MAX_COLOR_FEATURES]{};
        int query_counts[MAX_COLOR_FEATURES]{};
        int library_offsets[MAX_COLOR_FEATURES]{};
        int library_counts[MAX_COLOR_FEATURES]{};
    };

    bool matches_trajectory(const ReducedColorConformer& query_reduced,
                            const ColorConformer& library_physical,
                            const ColorInvariantPlan& invariant_plan,
                            int query_stride,
                            int library_stride) const noexcept;
    bool ensure_current_capacity(size_t points, bool fp32);
    bool ensure_winner_capacity(size_t points, bool fp32);

    TrajectoryIdentity trajectory_{};
    bool trajectory_active_ = false;
    bool current_pose_valid_ = false;
    bool winner_pose_valid_ = false;
    PreparedReducedPose current_pose_{};
    PreparedReducedPose winner_pose_{};
    std::unique_ptr<double[]> current_library_reduced_xyz_;
    std::unique_ptr<double[]> winner_library_reduced_xyz_;
    std::unique_ptr<float[]> current_library_reduced_xyz_fp32_;
    std::unique_ptr<float[]> winner_library_reduced_xyz_fp32_;
    size_t current_capacity_points_ = 0;
    size_t winner_capacity_points_ = 0;
    bool current_fp32_ = false;
    bool winner_fp32_ = false;
};

// Run another exact overlap loop against the selected start's retained posed
// points. This is a distinct evaluation (and counts every real pair term); it
// skips only the already-prepared coordinate transform. The caller must begin
// the trajectory, score each start at its final pose, retain only on the
// existing strict-`>` winner update, and call this once for the winner.
ColorEvaluation score_exact_final_color_retained(
    const ReducedColorConformer& query_reduced,
    const ColorConformer& library_physical,
    const ColorInvariantPlan& invariant_plan, const PreparedReducedPose& pose,
    int query_stride, int library_stride, ExactFinalScratch& scratch,
    uint64_t* analytic_pair_terms = nullptr);

// One exact color cross-overlap at one final pose. The query view is in the
// reduced frame; the library view remains in physical coordinates. The helper
// consumes only PreparedReducedPose's FP32 matrix and reduced translation,
// then accumulates overlap terms in FP32 for the canonical prepared route and
// in FP64 for general matrices and unsafe fallback.
// Passing `scratch == nullptr` gives the uncached per-call workspace arm; both
// paths execute the same transform and overlap loops. It never caches a score.
ColorEvaluation
score_exact_final_color(const ReducedColorConformer& query_reduced,
                        const ColorConformer& library_physical,
                        const ColorInvariantPlan& invariant_plan,
                        const PreparedReducedPose& pose, int query_stride,
                        int library_stride, ExactFinalScratch* scratch,
                        uint64_t* analytic_pair_terms = nullptr,
                        ExactFinalScratchStats* preparation_stats = nullptr,
                        bool time_preparation = false);

}  // namespace cpu
}  // namespace r3

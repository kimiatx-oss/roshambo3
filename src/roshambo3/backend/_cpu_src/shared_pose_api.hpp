// Owned query preparation for the explicit shared-pose contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "color_invariant_plan.hpp"
#include "reduced_pose.hpp"
#include "shape_combo_api.hpp"

namespace pybind11 {
class module_;
}

namespace r3 {
namespace cpu {

struct PreparedSharedPoseQuerySet {
    ShapeEvaluatorConfig shape_config{};
    PreparedShapeEvaluatorSetHandle shape_evaluators;
    std::shared_ptr<const ColorInvariantPlan> color_plan;
    std::vector<std::vector<float>> reduced_query_color_xyz;
    std::vector<ReducedColorConformer> reduced_query_colors;
    std::vector<float> source_query_coordinates;
    std::vector<int32_t> source_query_types;
    std::vector<int32_t> source_query_atom_counts;
    // Exact shared-pose A/A normalization terms. The block scorer initializes
    // these once, then reuses them across every worker-owned library block.
    std::once_flag query_self_once;
    std::vector<float> shape_query_self;
    std::vector<double> color_query_self;
    bool query_self_ready = false;
    double query_self_preparation_seconds = 0.0;
    int query_capacity = 0;
    int query_stride = 0;
    int n_features = 0;
    double preparation_seconds = 0.0;
    uint64_t reduced_color_bytes = 0;

    bool
    matches(const float* query_coordinates, int n_queries, int capacity,
            int stride, const int32_t* query_types,
            const int32_t* query_atom_counts,
            const ShapeEvaluatorConfig& wanted_shape,
            const ColorInteractionMatrixView& wanted_matrix) const noexcept;
};

using PreparedSharedPoseQuerySetHandle =
    std::shared_ptr<PreparedSharedPoseQuerySet>;

PreparedSharedPoseQuerySetHandle prepare_shared_pose_query_set(
    const float* query_coordinates, int n_queries, int query_capacity,
    int query_stride, const int32_t* query_types,
    const int32_t* query_atom_counts,
    const PreparedShapeEvaluatorSetHandle& shape_evaluators,
    const ColorInvariantPlan& color_plan);

void bind_shared_pose_api(pybind11::module_& module);

}  // namespace cpu
}  // namespace r3

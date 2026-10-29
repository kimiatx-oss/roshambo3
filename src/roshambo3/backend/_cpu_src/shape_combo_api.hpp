// Query-scoped combo shape evaluator preparation and Python API.
//
// One immutable `PreparedShapeEvaluator` per query row, built once before
// alignment timing and reused across scoring calls (query-scoped preparation
// with an explicit lifetime). Field backends build their
// query grid here, so field construction cost is reported separately from
// alignment time. See shape_evaluator.hpp for the per-evaluator contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "shape_evaluator.hpp"

namespace pybind11 {
class module_;
}

namespace r3 {
namespace cpu {

// The evaluators for every query row of one call, plus preparation telemetry.
// Byte and time fields are diagnostics for stamps and benchmark reports; they
// do not take part in matching.
struct PreparedShapeEvaluatorSet {
    ShapeEvaluatorConfig config{};
    std::vector<PreparedShapeEvaluatorHandle> evaluators;
    double preparation_seconds = 0.0;
    uint64_t query_snapshot_bytes = 0;   // owned reduced xyz copies
    uint64_t query_workspace_bytes = 0;  // query-atom SoA, analytic SIMD only
    double field_build_seconds = 0.0;
    double field_quantize_seconds = 0.0;
    uint64_t field_bytes = 0;
    uint64_t field_nodes = 0;

    bool matches_config(const ShapeEvaluatorConfig& wanted) const noexcept {
        return shape_evaluator_configs_match(wanted, config);
    }

    // True only if the config and every row's reduced coordinates are
    // byte-identical to what was prepared; a reused handle is never "close".
    bool matches(const ShapeEvaluatorConfig& wanted, const float* query_reduced,
                 int query_capacity, const int32_t* query_atoms,
                 size_t query_count) const noexcept;
};

using PreparedShapeEvaluatorSetHandle =
    std::shared_ptr<PreparedShapeEvaluatorSet>;

// `query_reduced` is `[query_count][query_capacity][3]`, already scaled by
// COORD_SCALE; `query_atoms[row]` is that row's real atom count (<= capacity).
PreparedShapeEvaluatorSetHandle
prepare_shape_evaluator_set(const float* query_reduced, int query_capacity,
                            const int32_t* query_atoms, size_t query_count,
                            const ShapeEvaluatorConfig& config);

void bind_shape_combo_api(pybind11::module_& module);

}  // namespace cpu
}  // namespace r3

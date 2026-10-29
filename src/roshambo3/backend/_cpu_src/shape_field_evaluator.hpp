// Query-prepared scalar and vector shape-field adapters.
#pragma once

#include <cstddef>
#include <cstdint>

#include "shape_evaluator.hpp"

namespace r3 {
namespace cpu {

struct FieldNaive32;

struct ShapeFieldPreparationMetadata {
    int query_atoms = 0;
    float spacing_angstrom = 0.0f;
    float padding_angstrom = 0.0f;
    int nodes_x = 0;
    int nodes_y = 0;
    int nodes_z = 0;
    size_t field_bytes = 0;
    size_t query_snapshot_bytes = 0;
    double field_build_seconds = 0.0;
    double field_quantize_seconds = 0.0;
    ShapeFieldStorage actual_storage = ShapeFieldStorage::FP32;
    ShapeFieldInterpolation actual_interpolation =
        ShapeFieldInterpolation::FP32;
    const char* storage_fallback = "none";
};

// Return preparation-only metadata for a field evaluator. This is kept
// out of the timed fixed-pose interface; callers may record it once when the
// query-scoped handle is constructed.
ShapeFieldPreparationMetadata
shape_field_preparation_metadata(const PreparedShapeEvaluator& evaluator);

// Immutable grid view. The evaluator owns its storage for the handle lifetime.
const FieldNaive32& shape_field_grid(const PreparedShapeEvaluator& evaluator);

// One compiled builder for both legacy per-call and reusable fields. Keeping
// the arithmetic in one translation unit prevents inlining/FMA differences.
void build_shape_field_grid(const float* query_reduced, int query_atoms,
                            float spacing, float padding, FieldNaive32& field);

}  // namespace cpu
}  // namespace r3

// Public query-scoped combo shape evaluator preparation. Kept separate from
// the large combo optimizer binding so new adapters do not widen its codegen.
//
// Python passes physical query coordinates; this file converts them once to
// the reduced frame (`A' = COORD_SCALE * A`, NUMERICS_POLICY section 4) and hands each
// row to the backend factory selected by `ShapeEvaluatorConfig`. Every
// config is validated explicitly: an unsupported combination is refused,
// never mapped to a nearby path (shape_evaluator.hpp).
#include "shape_combo_api.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "score.hpp"
#include "shape_field_evaluator.hpp"
#include "shared_pose_api.hpp"

namespace py = pybind11;

namespace r3 {
namespace cpu {
namespace {

using Clock = std::chrono::steady_clock;

// Range-check the integer knobs coming from Python and build the config.
// Width is 1 or 8 here; each backend factory then decides which widths it
// implements (analytic SIMD is width 8 only). A zero spacing passes here but is
// rejected by the field factories: zero means "unspecified" and the combo
// layer does not substitute a spacing policy.
ShapeEvaluatorConfig make_config(int backend, int grain, int width,
                                 float spacing, float padding, int storage,
                                 int interpolation) {
    if (backend < int(ShapeBackend::ScalarAnalytic) ||
        backend > int(ShapeBackend::PipelinedField))
        throw std::invalid_argument("shape_backend must be in [0, 4]");
    if (grain < int(ShapeGrain::Scalar) || grain > int(ShapeGrain::Trajectory))
        throw std::invalid_argument("shape_grain must be in [0, 3]");
    if (width != 1 && width != 8)
        throw std::invalid_argument("shape_width must be 1 or 8");
    if (storage < int(ShapeFieldStorage::FP32) ||
        storage > int(ShapeFieldStorage::FP16))
        throw std::invalid_argument(
            "field_storage must be 0 (FP32) or 1 (FP16)");
    if (interpolation < int(ShapeFieldInterpolation::FP32) ||
        interpolation > int(ShapeFieldInterpolation::MixedFP16YZ))
        throw std::invalid_argument(
            "field_interpolation must be 0 (FP32) or 1 (mixed FP16 Y/Z)");
    if (interpolation == int(ShapeFieldInterpolation::MixedFP16YZ) &&
        storage != int(ShapeFieldStorage::FP16))
        throw std::invalid_argument(
            "mixed FP16 Y/Z interpolation requires FP16 field storage");
    if (!std::isfinite(spacing) || !std::isfinite(padding) || spacing < 0.0f ||
        padding < 0.0f)
        throw std::invalid_argument(
            "shape field spacing and padding must be finite and nonnegative");
    ShapeEvaluatorConfig config;
    config.backend = ShapeBackend(backend);
    config.grain = ShapeGrain(grain);
    config.width = width;
    config.field_spacing_angstrom = spacing;
    config.field_padding_angstrom = padding;
    config.storage = ShapeFieldStorage(storage);
    config.interpolation = ShapeFieldInterpolation(interpolation);
    return config;
}

// Backend dispatch. Each factory owns its own grain/width validation.
PreparedShapeEvaluatorHandle
make_shape_evaluator(const ShapeEvaluatorConfig& config, const Conformer& query,
                     ShapeQueryMetadata metadata) {
    switch (config.backend) {
    case ShapeBackend::ScalarAnalytic:
    case ShapeBackend::AnalyticSimd:
        return make_analytic_shape_evaluator(config, query, metadata);
    case ShapeBackend::ScalarField:
        return make_scalar_field_shape_evaluator(config, query, metadata);
    case ShapeBackend::PairedChannelField:
        return make_paired_channel_field_shape_evaluator(config, query,
                                                         metadata);
    case ShapeBackend::PipelinedField:
        return make_pipelined_field_shape_evaluator(config, query, metadata);
    }
    throw std::invalid_argument("unsupported combo shape backend");
}

py::dict stamp_dict(const ShapeEvaluatorStamp& stamp) {
    py::dict result;
    result["backend"] = stamp.backend;
    result["grain"] = stamp.grain;
    result["isa"] = stamp.isa;
    result["storage"] = stamp.storage;
    result["reduction"] = stamp.reduction;
    result["exponential"] = stamp.exponential;
    result["width"] = stamp.width;
    result["vector_path"] = stamp.vector_path;
    result["requires_library_soa"] = stamp.requires_library_soa;
    return result;
}

}  // namespace

PreparedShapeEvaluatorSetHandle
prepare_shape_evaluator_set(const float* query_reduced, int query_capacity,
                            const int32_t* query_atoms, size_t query_count,
                            const ShapeEvaluatorConfig& config) {
    if (query_capacity < 0 || (query_count != 0 && query_atoms == nullptr) ||
        (query_count != 0 && query_capacity != 0 && query_reduced == nullptr))
        throw std::invalid_argument("invalid reduced combo shape queries");
    // `preparation_seconds` covers every row, field builds included; the
    // field_* totals break the field share out of it.
    auto result = std::make_shared<PreparedShapeEvaluatorSet>();
    result->config = config;
    result->evaluators.reserve(query_count);
    const auto begin = Clock::now();
    for (size_t row = 0; row < query_count; ++row) {
        const int n = query_atoms[row];
        if (n < 0 || n > query_capacity)
            throw std::invalid_argument(
                "query atom count exceeds the combo shape row capacity");
        // Rows are padded to `query_capacity`; only the first `n` atoms are
        // real, and the evaluator copies exactly those.
        const float* row_xyz =
            n == 0 ? nullptr : query_reduced + row * size_t(query_capacity) * 3;
        const Conformer query{row_xyz, n};
        PreparedShapeEvaluatorHandle evaluator =
            make_shape_evaluator(config, query, ShapeQueryMetadata{int(row)});
        if (config.backend == ShapeBackend::ScalarField ||
            config.backend == ShapeBackend::PairedChannelField ||
            config.backend == ShapeBackend::PipelinedField) {
            const ShapeFieldPreparationMetadata metadata =
                shape_field_preparation_metadata(*evaluator);
            result->field_build_seconds += metadata.field_build_seconds;
            result->field_quantize_seconds += metadata.field_quantize_seconds;
            result->field_bytes += metadata.field_bytes;
            result->field_nodes += uint64_t(metadata.nodes_x) *
                                   uint64_t(metadata.nodes_y) *
                                   uint64_t(metadata.nodes_z);
        }
        result->evaluators.push_back(std::move(evaluator));
        result->query_snapshot_bytes += uint64_t(n) * 3 * sizeof(float);
        // Query-atom SIMD keeps an SoA copy of the query padded to a whole
        // number of 8-lane vectors, four floats (x, y, z, weight) per slot;
        // see `SoaConformer` in grain.hpp. Other grains keep none.
        if (config.backend == ShapeBackend::AnalyticSimd &&
            config.grain == ShapeGrain::QueryAtom) {
            const uint64_t padded = ((uint64_t(n) + 7) / 8) * 8;
            result->query_workspace_bytes += padded * 4 * sizeof(float);
        }
    }
    result->preparation_seconds =
        std::chrono::duration<double>(Clock::now() - begin).count();
    return result;
}

// Reuse check for a cached set: the same config and, row by row, the same
// atom count and bit-identical reduced coordinates. Any mismatch returns
// false so the caller rebuilds rather than scoring against another query.
bool PreparedShapeEvaluatorSet::matches(const ShapeEvaluatorConfig& wanted,
                                        const float* query_reduced,
                                        int query_capacity,
                                        const int32_t* query_atoms,
                                        size_t query_count) const noexcept {
    if (!matches_config(wanted) || query_capacity < 0 ||
        query_count != evaluators.size() ||
        (query_count != 0 && query_atoms == nullptr) ||
        (query_count != 0 && query_capacity != 0 && query_reduced == nullptr))
        return false;
    for (size_t row = 0; row < query_count; ++row) {
        const int n = query_atoms[row];
        if (n < 0 || n > query_capacity) return false;
        const Conformer query{
            n == 0 ? nullptr : query_reduced + row * size_t(query_capacity) * 3,
            n};
        if (!evaluators[row]->matches(wanted, query)) return false;
    }
    return true;
}

void bind_shape_combo_api(py::module_& module) {
    py::class_<PreparedShapeEvaluatorSet, PreparedShapeEvaluatorSetHandle>(
        module, "PreparedShapeEvaluatorSet",
        "Query-scoped combo shape evaluators. Prepare once and reuse across "
        "alignment calls; query coordinates are copied into the set.")
        .def_property_readonly("n_queries",
                               [](const PreparedShapeEvaluatorSet& x) {
                                   return x.evaluators.size();
                               })
        .def_property_readonly("preparation_seconds",
                               [](const PreparedShapeEvaluatorSet& x) {
                                   return x.preparation_seconds;
                               })
        .def_property_readonly("query_snapshot_bytes",
                               [](const PreparedShapeEvaluatorSet& x) {
                                   return x.query_snapshot_bytes;
                               })
        .def_property_readonly("query_workspace_bytes",
                               [](const PreparedShapeEvaluatorSet& x) {
                                   return x.query_workspace_bytes;
                               })
        .def_property_readonly("field_build_seconds",
                               [](const PreparedShapeEvaluatorSet& x) {
                                   return x.field_build_seconds;
                               })
        .def_property_readonly("field_quantize_seconds",
                               [](const PreparedShapeEvaluatorSet& x) {
                                   return x.field_quantize_seconds;
                               })
        .def_property_readonly(
            "field_bytes",
            [](const PreparedShapeEvaluatorSet& x) { return x.field_bytes; })
        .def_property_readonly(
            "field_nodes",
            [](const PreparedShapeEvaluatorSet& x) { return x.field_nodes; })
        .def_property_readonly(
            "stamps", [](const PreparedShapeEvaluatorSet& x) {
                py::list out;
                for (const auto& evaluator : x.evaluators)
                    out.append(stamp_dict(evaluator->stamp()));
                return out;
            });
    module.def(
        "prepare_combo_shape_evaluators",
        [](py::array_t<float, py::array::c_style> coords,
           py::array_t<int32_t, py::array::c_style> query_atoms, int backend,
           int grain, int width, float spacing, float padding,
           int field_storage, int field_interpolation) {
            if (coords.ndim() != 3 ||
                (coords.shape(2) != 3 && coords.shape(2) != 4) ||
                query_atoms.ndim() != 1 ||
                query_atoms.shape(0) != coords.shape(0))
                throw std::invalid_argument(
                    "shape query coordinates must be [n_query, atoms, 3|4] and "
                    "query_n_atoms must match the query rows");
            const ShapeEvaluatorConfig config =
                make_config(backend, grain, width, spacing, padding,
                            field_storage, field_interpolation);
            // Accept xyz or xyzw rows (stride 3 or 4); the fourth column is
            // ignored. Scale to the reduced frame here, once per query, so
            // the evaluators never see physical query coordinates.
            const size_t nq = size_t(coords.shape(0));
            const int capacity = int(coords.shape(1));
            const int stride = int(coords.shape(2));
            std::vector<float> reduced(nq * size_t(capacity) * 3);
            for (size_t row = 0; row < nq; ++row) {
                const float* src =
                    coords.data() + row * size_t(capacity) * size_t(stride);
                float* dst = reduced.data() + row * size_t(capacity) * 3;
                for (int atom = 0; atom < capacity; ++atom)
                    for (int axis = 0; axis < 3; ++axis)
                        dst[size_t(atom) * 3 + size_t(axis)] =
                            COORD_SCALE *
                            src[size_t(atom) * size_t(stride) + size_t(axis)];
            }
            return prepare_shape_evaluator_set(reduced.data(), capacity,
                                               query_atoms.data(), nq, config);
        },
        py::arg("query_coords"), py::arg("query_n_atoms"),
        py::arg("shape_backend"), py::arg("shape_grain"),
        py::arg("shape_width") = 1, py::arg("field_spacing_angstrom") = 0.0f,
        // The 8 A padding default matches the CPU field path's halo.
        py::arg("field_padding_angstrom") = 8.0f, py::arg("field_storage") = 0,
        py::arg("field_interpolation") = 0,
        "Build immutable query-scoped evaluators for mode-3 combo scoring.");
    module.def(
        // The stamp for a config without preparing a query: builds a
        // throwaway evaluator on an empty query and reports what it would
        // run (backend, grain, ISA, reduction, exponential).
        "combo_shape_config_stamp",
        [](int backend, int grain, int width, float spacing, float padding,
           int field_storage, int field_interpolation) {
            const auto config =
                make_config(backend, grain, width, spacing, padding,
                            field_storage, field_interpolation);
            Conformer empty{nullptr, 0};
            auto evaluator = make_shape_evaluator(config, empty, {});
            py::dict result = stamp_dict(evaluator->stamp());
            result["field_spacing_angstrom"] = spacing;
            result["field_padding_angstrom"] = padding;
            return result;
        },
        py::arg("shape_backend"), py::arg("shape_grain"),
        py::arg("shape_width") = 1, py::arg("field_spacing_angstrom") = 0.0f,
        py::arg("field_padding_angstrom") = 8.0f, py::arg("field_storage") = 0,
        py::arg("field_interpolation") = 0);
    bind_shared_pose_api(module);
}

}  // namespace cpu
}  // namespace r3

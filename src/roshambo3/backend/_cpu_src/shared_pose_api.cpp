// Owned query color and interaction preparation. The optimizer's large
// dispatch TU consumes this immutable handle without rebuilding either.
//
// Query-side preparation for the shared FP32 reduced-frame pose contract
// (doc/CPU_SHARED_POSE_CONTRACT.md). Done once per query set, before
// alignment timing:
//   - query color features are converted to the reduced frame, `A' = s*A`
//     with s = COORD_SCALE, preserving feature order, type segments and real
//     counts; library features stay physical and are moved by `M*B + t'`;
//   - the interaction matrix becomes a `ColorInvariantPlan` holding both the
//     physical decay `d` and the reduced decay `d' = d / s^2`
//     (color_invariant_plan.cpp); the coefficient is unchanged;
//   - the already-prepared shape evaluators are checked against the same
//     coordinates, so shape and color cannot be prepared from different
//     queries.
// Shape atoms and color features are distinct point sets that share only the
// rigid transform.
#include "shared_pose_api.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "color_view.hpp"
#include "kernel.hpp"
#include "score.hpp"
#include "field.hpp"

namespace py = pybind11;

namespace r3 {
namespace cpu {
namespace {

using Clock = std::chrono::steady_clock;
// (pi/2)^1.5, the physical color prefactor; the same value as
// COLOR_PREFACTOR in color_scalar.cpp and numerics.COLOR_PREFACTOR.
constexpr double kColorPrefactor = 1.9687012432153024;

bool same_shape_config(const ShapeEvaluatorConfig& a,
                       const ShapeEvaluatorConfig& b) noexcept {
    return shape_evaluator_configs_match(a, b);
}

// Turn the Python interaction matrices (`interaction_r` = widths r_ij,
// `interaction_p` = heights p_ij, indexed [query_type][library_type]) into
// the physical Gaussian `c * exp(d |a - b|^2)` per type pair:
//
//     c = (pi/2)^1.5 * p^2 / r^1.5,    d = -r / 2
//
// (reference/color.py). The arithmetic mirrors color_scalar.cpp's `Matrix`
// so both color paths see the same coefficients. Rows and columns 0 are
// skipped: type 0 is an ordinary atom, never a color feature. A zero width
// means the two types do not interact and leaves the entry at zero (it
// would otherwise divide by zero); a zero height is a real, zero-valued
// interaction. `canonical` records whether the matrix is the identity
// (unit widths, unit diagonal heights, zero off-diagonal heights), which
// enables the specialized canonical consumers.
ColorInvariantPlan
make_color_plan(const py::array_t<float, py::array::c_style>& widths,
                const py::array_t<float, py::array::c_style>& heights) {
    if (widths.ndim() != 2 || heights.ndim() != 2 || widths.shape(0) < 1 ||
        widths.shape(0) > MAX_COLOR_FEATURES ||
        widths.shape(1) != widths.shape(0) ||
        heights.shape(0) != widths.shape(0) ||
        heights.shape(1) != widths.shape(1))
        throw std::invalid_argument(
            "shared-pose interaction matrices must be matching square arrays");
    constexpr int stride = MAX_COLOR_FEATURES;
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> coefficients{};
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> decays{};
    const int n = int(widths.shape(0));
    bool canonical = true;
    auto r = widths.unchecked<2>();
    auto p = heights.unchecked<2>();
    for (int ta = 0; ta < n; ++ta) {
        for (int tb = 0; tb < n; ++tb) {
            const float width32 = r(ta, tb);
            const float height32 = p(ta, tb);
            if (!std::isfinite(width32) || !std::isfinite(height32) ||
                width32 < 0.0f)
                throw std::invalid_argument(
                    "shared-pose interaction widths/heights are invalid");
        }
    }
    for (int ta = 1; ta < n; ++ta) {
        for (int tb = 1; tb < n; ++tb) {
            const double width = double(r(ta, tb));
            const double height = double(p(ta, tb));
            canonical &= (width == 1.0 && height == (ta == tb ? 1.0 : 0.0));
            if (width != 0.0) {
                const size_t at = size_t(ta) * size_t(stride) + size_t(tb);
                coefficients[at] =
                    kColorPrefactor * height * height / std::pow(width, 1.5);
                decays[at] = -0.5 * width;
            }
        }
    }
    const ColorInteractionMatrixView input{n, stride, canonical,
                                           coefficients.data(), decays.data()};
    return ColorInvariantPlan(input);
}

}  // namespace

// Reuse check for a cached prepared set. Compares the SOURCE arrays the set
// was built from (physical coordinates, types, counts) byte for byte, plus
// shape config and matrix identity, rather than the derived reduced data:
// any change at all in the caller's input forces a rebuild.
bool PreparedSharedPoseQuerySet::matches(
    const float* query_coordinates, int n_queries, int capacity, int stride,
    const int32_t* query_types, const int32_t* query_atom_counts,
    const ShapeEvaluatorConfig& wanted_shape,
    const ColorInteractionMatrixView& wanted_matrix) const noexcept {
    if (!shape_evaluators || !color_plan ||
        !same_shape_config(wanted_shape, shape_config) || n_queries < 0 ||
        size_t(n_queries) != reduced_query_colors.size() ||
        capacity != query_capacity || stride != query_stride || capacity < 0 ||
        stride < 3 ||
        (n_queries > 0 &&
         (query_types == nullptr || query_atom_counts == nullptr)) ||
        (n_queries > 0 && capacity > 0 && query_coordinates == nullptr) ||
        !color_plan->matches_input(wanted_matrix))
        return false;

    const size_t coord_count =
        size_t(n_queries) * size_t(capacity) * size_t(stride);
    const size_t type_count = size_t(n_queries) * size_t(capacity);
    if (source_query_coordinates.size() != coord_count ||
        source_query_types.size() != type_count ||
        source_query_atom_counts.size() != size_t(n_queries))
        return false;
    if (coord_count != 0 &&
        std::memcmp(source_query_coordinates.data(), query_coordinates,
                    coord_count * sizeof(float)) != 0)
        return false;
    if (type_count != 0 && std::memcmp(source_query_types.data(), query_types,
                                       type_count * sizeof(int32_t)) != 0)
        return false;
    if (n_queries != 0 &&
        std::memcmp(source_query_atom_counts.data(), query_atom_counts,
                    size_t(n_queries) * sizeof(int32_t)) != 0)
        return false;
    return true;
}

PreparedSharedPoseQuerySetHandle prepare_shared_pose_query_set(
    const float* query_coordinates, int n_queries, int query_capacity,
    int query_stride, const int32_t* query_types,
    const int32_t* query_atom_counts,
    const PreparedShapeEvaluatorSetHandle& shape_evaluators,
    const ColorInvariantPlan& color_plan) {
    if (n_queries < 0 || query_capacity < 0 || query_stride < 3 ||
        (n_queries > 0 &&
         (query_types == nullptr || query_atom_counts == nullptr)) ||
        (n_queries > 0 && query_capacity > 0 && query_coordinates == nullptr) ||
        !shape_evaluators)
        throw std::invalid_argument("invalid shared-pose query preparation");

    // Keep verbatim copies of the inputs for `matches`, and own everything
    // the workers will read, so the handle outlives the caller's arrays.
    auto result = std::make_shared<PreparedSharedPoseQuerySet>();
    result->shape_evaluators = shape_evaluators;
    result->shape_config = shape_evaluators->config;
    result->color_plan = std::make_shared<ColorInvariantPlan>(color_plan);
    result->query_capacity = query_capacity;
    result->query_stride = query_stride;
    result->n_features = color_plan.n_features();
    result->reduced_query_color_xyz.reserve(size_t(n_queries));
    result->reduced_query_colors.resize(size_t(n_queries));

    const size_t coord_count =
        size_t(n_queries) * size_t(query_capacity) * size_t(query_stride);
    const size_t type_count = size_t(n_queries) * size_t(query_capacity);
    if (coord_count != 0)
        result->source_query_coordinates.assign(
            query_coordinates, query_coordinates + coord_count);
    if (type_count != 0)
        result->source_query_types.assign(query_types,
                                          query_types + type_count);
    if (n_queries != 0)
        result->source_query_atom_counts.assign(query_atom_counts,
                                                query_atom_counts + n_queries);

    const auto begin = Clock::now();
    // `shape_query_reduced` is only for the consistency check below; the
    // shape evaluators already own their reduced copies.
    std::vector<float> shape_query_reduced(size_t(n_queries) *
                                           size_t(query_capacity) * 3);
    for (int row = 0; row < n_queries; ++row) {
        const float* src = query_coordinates + size_t(row) *
                                                   size_t(query_capacity) *
                                                   size_t(query_stride);
        float* shape_dst = shape_query_reduced.data() +
                           size_t(row) * size_t(query_capacity) * 3;
        for (int slot = 0; slot < query_capacity; ++slot)
            for (int axis = 0; axis < 3; ++axis)
                shape_dst[size_t(slot) * 3 + size_t(axis)] =
                    COORD_SCALE *
                    src[size_t(slot) * size_t(query_stride) + size_t(axis)];

        // Select this row's color features (type > 0) in physical
        // coordinates, then scale each one into the reduced frame. The
        // physical view's counts and type segments are kept; only the
        // coordinate pointer is swapped for the owned reduced copy.
        const ColorConformer physical = build_color_view_from_types(
            src, query_types + size_t(row) * size_t(query_capacity),
            query_atom_counts[row], query_capacity, color_plan.n_features(),
            query_stride);
        auto& reduced_xyz = result->reduced_query_color_xyz.emplace_back(
            size_t(physical.n_color) * 3);
        for (int feature = 0; feature < physical.n_color; ++feature) {
            const float* raw =
                physical.color_xyz + size_t(feature) * size_t(query_stride);
            for (int axis = 0; axis < 3; ++axis)
                reduced_xyz[size_t(feature) * 3 + size_t(axis)] =
                    COORD_SCALE * raw[axis];
        }
        ColorConformer reduced_view = physical;
        reduced_view.shape_xyz = nullptr;  // shape uses its own evaluators
        reduced_view.color_xyz =
            reduced_xyz.empty() ? nullptr : reduced_xyz.data();
        result->reduced_query_colors[size_t(row)].features = reduced_view;
    }

    // The shape set must have been prepared from exactly these coordinates
    // (scaled the same way) and this config; otherwise shape and color would
    // score different queries under one handle.
    if (!shape_evaluators->matches(result->shape_config,
                                   shape_query_reduced.data(), query_capacity,
                                   query_atom_counts, size_t(n_queries)))
        throw std::invalid_argument(
            "shared-pose shape evaluators do not match the query coordinates/config");
    result->preparation_seconds =
        std::chrono::duration<double>(Clock::now() - begin).count();
    result->reduced_color_bytes = 0;
    for (const auto& xyz : result->reduced_query_color_xyz)
        result->reduced_color_bytes += uint64_t(xyz.size()) * sizeof(float);
    return result;
}

void bind_shared_pose_api(py::module_& module) {
    py::class_<PreparedSharedPoseQuerySet, PreparedSharedPoseQuerySetHandle>(
        module, "PreparedSharedPoseQuerySet",
        "Owned query colors, reduced interaction matrix, and shape evaluators "
        "for the explicit shared FP32 reduced-pose contract.")
        .def_property_readonly("n_queries",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.reduced_query_colors.size();
                               })
        .def_property_readonly("preparation_seconds",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.preparation_seconds;
                               })
        .def_property_readonly("reduced_color_bytes",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.reduced_color_bytes;
                               })
        // The query self-overlaps (S_AA) are filled lazily by the block
        // scorer under `query_self_once` and then reused across every
        // worker-owned library block.
        .def_property_readonly("query_self_prepared",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.query_self_ready;
                               })
        .def_property_readonly("shape_query_self_evaluations",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.shape_query_self.size();
                               })
        .def_property_readonly("color_query_self_evaluations",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.color_query_self.size();
                               })
        .def_property_readonly("query_self_preparation_seconds",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.query_self_preparation_seconds;
                               })
        .def_property_readonly(
            "query_self_bytes",
            [](const PreparedSharedPoseQuerySet& x) {
                return x.shape_query_self.size() * sizeof(float) +
                       x.color_query_self.size() * sizeof(double);
            })
        .def_property_readonly(
            "shape_field_build_seconds",
            [](const PreparedSharedPoseQuerySet& x) {
                return x.shape_evaluators->field_build_seconds;
            })
        .def_property_readonly("shape_field_bytes",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.shape_evaluators->field_bytes;
                               })
        .def_property_readonly("matrix_features",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.color_plan->n_features();
                               })
        .def_property_readonly("matrix_canonical",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.color_plan->canonical();
                               })
        .def_property_readonly("matrix_source_stride",
                               [](const PreparedSharedPoseQuerySet& x) {
                                   return x.color_plan->source_stride();
                               });

    // Invoke the same prepared evaluator consumed by production combo
    // optimization. Return its sampled overlap before any exact final rescore.
    module.def("numerics_shape_evaluate", [](py::array_t<float,py::array::c_style> query,
                                            py::array_t<float,py::array::c_style> library,
                                            int backend, int grain, int width,
                                            float spacing, py::object physical_matrix,
                                            py::object physical_translation) {
        if (query.ndim()!=2 || library.ndim()!=2 || query.shape(1)!=3 || library.shape(1)!=3 ||
            backend<0 || backend>4 || grain<0 || grain>3)
            throw std::invalid_argument("shape probe needs [n,3] points and declared backend/grain");
        if (query.shape(0)>std::numeric_limits<int>::max() || library.shape(0)>std::numeric_limits<int>::max())
            throw std::invalid_argument("shape probe count exceeds int capacity");
        for (const auto* array : {&query,&library})
            for (py::ssize_t i=0;i<array->size();++i)
                if (!std::isfinite(array->data()[i]))
                    throw std::invalid_argument("shape probe points must be finite");
        std::vector<float> reduced(query.size());
        for (py::ssize_t i=0;i<query.size();++i) reduced[i]=COORD_SCALE*query.data()[i];
        const int count=query.shape(0);
        ShapeEvaluatorConfig config{};
        config.backend=ShapeBackend(backend); config.grain=ShapeGrain(grain); config.width=width;
        config.field_spacing_angstrom=spacing; config.field_padding_angstrom=4.f;
        auto prepared=prepare_shape_evaluator_set(reduced.data(),count,&count,1,config);
        const Conformer lib{library.data(),int(library.shape(0))};
        auto workspace=backend<2 ? prepare_analytic_shape_library_workspace(lib,config.grain,width)
                                 : prepare_field_shape_library_workspace(lib,config.backend,width);
        float matrix[9]={COORD_SCALE,0,0,0,COORD_SCALE,0,0,0,COORD_SCALE};
        float translation[3]={0,0,0};
        // Diagnostic affine input goes through the same production evaluator
        // and its ordinary matrix/translation arithmetic. Existing callers
        // retain the identity defaults; no optimizer or scoring path changes.
        if (!physical_matrix.is_none()) {
            const auto supplied=physical_matrix.cast<py::array_t<float,py::array::c_style>>();
            if (supplied.ndim()!=2 || supplied.shape(0)!=3 || supplied.shape(1)!=3)
                throw std::invalid_argument("physical_matrix must be float32 [3,3]");
            for (int k=0;k<9;++k) {
                if (!std::isfinite(supplied.data()[k]))
                    throw std::invalid_argument("physical_matrix must be finite");
                matrix[k]=COORD_SCALE*supplied.data()[k];
                if (!std::isfinite(matrix[k]))
                    throw std::invalid_argument("scaled physical_matrix must be finite");
            }
        }
        if (!physical_translation.is_none()) {
            const auto supplied=physical_translation.cast<py::array_t<float,py::array::c_style>>();
            if (supplied.ndim()!=1 || supplied.shape(0)!=3)
                throw std::invalid_argument("physical_translation must be float32 [3]");
            for (int k=0;k<3;++k) {
                if (!std::isfinite(supplied.data()[k]))
                    throw std::invalid_argument("physical_translation must be finite");
                translation[k]=COORD_SCALE*supplied.data()[k];
                if (!std::isfinite(translation[k]))
                    throw std::invalid_argument("scaled physical_translation must be finite");
            }
        }
        ShapePoseRequest request{};
        request.library=&lib; request.library_workspace=&workspace;
        request.matrix=matrix; request.translation=translation;
        ShapeEvaluation result{};
        prepared->evaluators[0]->evaluate_one(request,result);
        const auto stamp=prepared->evaluators[0]->stamp();
        py::dict out;
        out["overlap_volume"]=SHAPE_PREFACTOR*result.s_ab;
        out["field_samples"]=result.counts.field_samples;
        out["fallback_evaluations"]=result.counts.fallback_evaluations;
        out["backend"]=stamp.backend; out["grain"]=stamp.grain;
        out["exponential"]=stamp.exponential; out["reduction"]=stamp.reduction;
        out["storage"]=stamp.storage; out["width"]=stamp.width;
        if (backend>=2) {
            const auto geometry=plan_field(reduced.data(),count,spacing,4.f);
            out["grid_origin_reduced"]=py::make_tuple(geometry.ox,geometry.oy,geometry.oz);
            out["grid_spacing_reduced"]=geometry.hs;
        }
        return out;
    },py::arg("query"),py::arg("library"),py::arg("backend"),py::arg("grain"),
      py::arg("width"),py::arg("spacing_angstrom")=.25f,
      py::arg("physical_matrix")=py::none(),py::arg("physical_translation")=py::none());

    module.def(
        "prepare_combo_shared_pose_queries",
        [](py::array_t<float, py::array::c_style> coords,
           py::array_t<int32_t, py::array::c_style> types,
           py::array_t<int32_t, py::array::c_style> query_atoms,
           py::array_t<float, py::array::c_style> widths,
           py::array_t<float, py::array::c_style> heights,
           PreparedShapeEvaluatorSetHandle shape_evaluators) {
            if (coords.ndim() != 3 ||
                (coords.shape(2) != 3 && coords.shape(2) != 4) ||
                types.ndim() != 2 || types.shape(0) != coords.shape(0) ||
                types.shape(1) != coords.shape(1) || query_atoms.ndim() != 1 ||
                query_atoms.shape(0) != coords.shape(0))
                throw std::invalid_argument(
                    "shared-pose queries need matching [n_query,L,3|4], "
                    "[n_query,L], and [n_query] arrays");
            const ColorInvariantPlan plan = make_color_plan(widths, heights);
            return prepare_shared_pose_query_set(
                coords.data(), int(coords.shape(0)), int(coords.shape(1)),
                int(coords.shape(2)), types.data(), query_atoms.data(),
                shape_evaluators, plan);
        },
        py::arg("query_coords"), py::arg("query_types"),
        py::arg("query_n_atoms"), py::arg("interaction_r"),
        py::arg("interaction_p"), py::arg("shape_evaluators"),
        "Own query color coordinates in the reduced frame, the exact physical/"
        "reduced matrix views, and the matching immutable shape evaluators.");
}

}  // namespace cpu
}  // namespace r3

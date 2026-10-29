// Private fixed-pose diagnostic boundary. No production optimizer dispatch.
//
// `final_rescore_fixed` is the measurement hook: it builds FinalRescorePair records from dense numpy arrays,
// times `repeats` calls of final_rescore_batch WITHOUT counters, then makes one
// separate counted call and reports any drift between the two. The initial ARM
// fixture's timing included counters and is recorded as advisory; the later
// replay used this split.
#include "final_rescore.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <chrono>
#include <vector>
namespace py = pybind11;

namespace r3::cpu {
void bind_final_rescore(py::module_& m) {
    m.def(
        "final_rescore_fixed",
        [](py::array_t<float, py::array::c_style> query,
           py::array_t<int32_t, py::array::c_style> query_counts,
           py::array_t<float, py::array::c_style> library,
           py::array_t<int32_t, py::array::c_style> library_counts,
           py::array_t<int32_t, py::array::c_style> pair_query,
           py::array_t<int32_t, py::array::c_style> pair_library,
           py::array_t<float, py::array::c_style> matrices,
           py::array_t<float, py::array::c_style> translations,
           py::array_t<double, py::array::c_style> coefficients,
           py::array_t<double, py::array::c_style> physical_decays,
           bool canonical, int mode, int repeats) {
            if (query.ndim() != 3 || library.ndim() != 3 ||
                query.shape(2) != 3 || library.shape(2) != 3 ||
                query_counts.ndim() != 2 || library_counts.ndim() != 2 ||
                query_counts.shape(0) != query.shape(0) ||
                library_counts.shape(0) != library.shape(0) ||
                query_counts.shape(1) != library_counts.shape(1) ||
                query_counts.shape(1) < 1 ||
                query_counts.shape(1) > MAX_COLOR_FEATURES ||
                pair_query.ndim() != 1 || pair_library.ndim() != 1 ||
                pair_query.size() != pair_library.size() ||
                matrices.ndim() != 3 ||
                matrices.shape(0) != pair_query.size() ||
                matrices.shape(1) != 3 || matrices.shape(2) != 3 ||
                translations.ndim() != 2 ||
                translations.shape(0) != pair_query.size() ||
                translations.shape(1) != 3 || coefficients.ndim() != 2 ||
                coefficients.shape(0) != query_counts.shape(1) ||
                coefficients.shape(1) != coefficients.shape(0) ||
                physical_decays.ndim() != 2 ||
                physical_decays.shape(0) != coefficients.shape(0) ||
                physical_decays.shape(1) != coefficients.shape(1) ||
                repeats < 1)
                throw std::invalid_argument(
                    "invalid fixed final-rescore diagnostic dimensions");
            const int nf = int(query_counts.shape(1));
            const size_t n = size_t(pair_query.size());
            ColorInvariantPlan plan({nf, nf, canonical, coefficients.data(),
                                     physical_decays.data()});
            std::vector<PreparedReducedPose> poses(n);
            std::vector<ReducedColorConformer> q(n);
            std::vector<ColorConformer> b(n);
            std::vector<FinalRescorePair> pairs(n);
            // Each row is [n_real shape atoms | type-1 features | type-2 ...],
            // with counts[0] = n_real and counts[t] the type-t feature count;
            // trailing slots are padding and are never read.
            auto view = [&](const float* xyz, const int32_t* counts,
                            int slots) {
                ColorConformer v{};
                v.shape_xyz = xyz;
                v.n_real = counts[0];
                v.n_features = nf;
                if (v.n_real < 0 || v.n_real > slots)
                    throw std::invalid_argument(
                        "invalid diagnostic real count");
                int offset = 0;
                for (int t = 1; t < nf; ++t) {
                    if (counts[t] < 0 || counts[t] > slots - v.n_real - offset)
                        throw std::invalid_argument(
                            "invalid diagnostic feature count");
                    v.seg_offset[t] = offset;
                    v.seg_count[t] = counts[t];
                    offset += counts[t];
                }
                v.n_color = offset;
                v.color_xyz = xyz + 3 * v.n_real;
                return v;
            };
            for (size_t i = 0; i < n; ++i) {
                const int qi = pair_query.data()[i],
                          bi = pair_library.data()[i];
                if (qi < 0 || qi >= query.shape(0) || bi < 0 ||
                    bi >= library.shape(0))
                    throw std::invalid_argument("invalid diagnostic pair row");
                q[i].features = view(
                    query.data() + size_t(qi) * query.shape(1) * 3,
                    query_counts.data() + size_t(qi) * nf, int(query.shape(1)));
                b[i] = view(library.data() + size_t(bi) * library.shape(1) * 3,
                            library_counts.data() + size_t(bi) * nf,
                            int(library.shape(1)));
                std::copy_n(matrices.data() + 9 * i, 9, poses[i].matrix);
                std::copy_n(translations.data() + 3 * i, 3,
                            poses[i].translation);
                pairs[i] = {{q[i].features.shape_xyz, q[i].features.n_real},
                            {b[i].shape_xyz, b[i].n_real},
                            &q[i],
                            &b[i],
                            &plan,
                            &poses[i],
                            3,
                            3,
                            nullptr};
            }
            std::vector<FinalRescoreCross> out(n);
            FinalRescoreCounts counts;
            const auto started = std::chrono::steady_clock::now();
            {
                py::gil_scoped_release unlocked;
                for (int r = 0; r < repeats; ++r)
                    final_rescore_batch(pairs.data(), n, out.data(),
                                        FinalRescoreMode(mode), nullptr);
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              started)
                    .count();
            // Diagnostics are a separate single call outside the timing window.
            // Return timed outputs and record any instrumentation-induced drift.
            std::vector<FinalRescoreCross> counted(n);
            final_rescore_batch(pairs.data(), n, counted.data(),
                                FinalRescoreMode(mode), &counts);
            float shape_drift = 0;
            double color_drift = 0;
            for (size_t i = 0; i < n; ++i) {
                shape_drift = std::max(
                    shape_drift, std::abs(out[i].shape - counted[i].shape));
                color_drift = std::max(
                    color_drift, std::abs(out[i].color - counted[i].color));
            }
            py::array_t<float> shape(n);
            py::array_t<double> color(n);
            for (size_t i = 0; i < n; ++i) {
                shape.mutable_data()[i] = out[i].shape;
                color.mutable_data()[i] = out[i].color;
            }
            py::dict d;
            d["shape"] = shape;
            d["color"] = color;
            d["kernel_seconds"] = seconds;
            d["repeats"] = repeats;
            d["pairs"] = n;
            d["shape_terms"] = counts.shape_terms;
            d["color_terms"] = counts.color_terms;
            d["shape_issued_lanes"] = counts.shape_issued_lanes;
            d["color_issued_lanes"] = counts.color_issued_lanes;
            d["accurate_fallback_pairs"] = counts.accurate_fallback_pairs;
            d["packed_coordinate_bytes"] = counts.packed_coordinate_bytes;
            d["horizontal_reductions"] = counts.horizontal_reductions;
            d["temporary_allocations"] = counts.temporary_allocations;
            d["allocated_payload_bytes"] = counts.allocated_payload_bytes;
            d["timing_has_counters"] = false;
            d["diagnostic_calls"] = 1;
            d["counted_shape_max_abs_drift"] = shape_drift;
            d["counted_color_max_abs_drift"] = color_drift;
            d["setup_record_bytes"] =
                n * (sizeof(PreparedReducedPose) +
                     sizeof(ReducedColorConformer) + sizeof(ColorConformer) +
                     sizeof(FinalRescorePair) + sizeof(FinalRescoreCross));
            return d;
        },
        py::arg("query_reduced"), py::arg("query_counts"),
        py::arg("library_physical"), py::arg("library_counts"),
        py::arg("pair_query"), py::arg("pair_library"), py::arg("matrices"),
        py::arg("translations"), py::arg("coefficients"),
        py::arg("physical_decays"), py::arg("canonical"), py::arg("mode"),
        py::arg("repeats") = 1);
}
}

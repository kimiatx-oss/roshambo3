// Reduced-frame pose/value/gradient gate for both plan modes and ISAs.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "color_conformer_kernel.hpp"
#include "color_exact_final.hpp"
#include "kernel.hpp"
#include "reduced_pose.hpp"

using namespace r3::cpu;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ColorConformer make_view(const std::vector<float>& xyz,
                         int count_type1, int count_type2) {
    ColorConformer view{};
    view.color_xyz = xyz.empty() ? nullptr : xyz.data();
    view.n_color = count_type1 + count_type2;
    view.n_features = 3;
    view.seg_offset[1] = 0;
    view.seg_count[1] = count_type1;
    view.seg_offset[2] = count_type1;
    view.seg_count[2] = count_type2;
    return view;
}

ColorInvariantPlan make_plan() {
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> coefficient{};
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> decay{};
    const int s = MAX_COLOR_FEATURES;
    coefficient[1*s+1] = 1.3;
    coefficient[1*s+2] = 0.7;
    coefficient[2*s+1] = 0.5;
    coefficient[2*s+2] = 1.1;
    decay[1*s+1] = -0.8;
    decay[1*s+2] = -0.5;
    decay[2*s+1] = -0.6;
    decay[2*s+2] = -0.9;
    return ColorInvariantPlan({3, s, false, coefficient.data(), decay.data()});
}

double independent_reduced_value(const ReducedColorConformer& query,
                   const ColorConformer& library,
                   const ColorInvariantPlan& plan,
                   const PreparedReducedPose& pose) {
    const ColorInvariantMatrixView matrix = plan.reduced_matrix_view();
    double sum = 0.0;
    for (int ta = 1; ta < matrix.n_features; ++ta) {
        const int tb_begin = matrix.canonical ? ta : 1;
        const int tb_end = matrix.canonical ? ta + 1 : matrix.n_features;
        for (int tb = tb_begin; tb < tb_end; ++tb) {
            const size_t at = size_t(ta) * size_t(matrix.stride) +
                              size_t(tb);
            const double decay = matrix.decays[at];
            if (decay == 0.0) continue;
            for (int i = 0; i < query.features.seg_count[ta]; ++i) {
                const float* a = query.features.color_xyz +
                    size_t(query.features.seg_offset[ta] + i) * 3;
                for (int j = 0; j < library.seg_count[tb]; ++j) {
                    const float* b = library.color_xyz +
                        size_t(library.seg_offset[tb] + j) * 3;
                    double posed[3]{};
                    for (int row = 0; row < 3; ++row)
                        posed[row] = double(pose.matrix[3*row+0]) * b[0] +
                                     double(pose.matrix[3*row+1]) * b[1] +
                                     double(pose.matrix[3*row+2]) * b[2] +
                                     double(pose.translation[row]);
                    const double dx = double(a[0]) - posed[0];
                    const double dy = double(a[1]) - posed[1];
                    const double dz = double(a[2]) - posed[2];
                    sum += matrix.coefficients[at] * std::exp(
                        decay * (dx*dx + dy*dy + dz*dz));
                }
            }
        }
    }
    return sum;
}

double independent_physical_value(const ColorConformer& query,
        const ColorConformer& library, const ColorInvariantPlan& plan,
        const float quaternion[4], const float reduced_translation[3]) {
    double norm2 = 0.0;
    for (int k = 0; k < 4; ++k)
        norm2 += double(quaternion[k]) * double(quaternion[k]);
    const double norm = std::sqrt(norm2);
    const double w = double(quaternion[0]) / norm;
    const double x = double(quaternion[1]) / norm;
    const double y = double(quaternion[2]) / norm;
    const double z = double(quaternion[3]) / norm;
    const double rotation[9] = {
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)};
    const double scale = double(COORD_SCALE);
    const ColorInvariantMatrixView matrix = plan.physical_matrix_view();
    double sum = 0.0;
    for (int ta = 1; ta < matrix.n_features; ++ta) {
        const int tb_begin = matrix.canonical ? ta : 1;
        const int tb_end = matrix.canonical ? ta + 1 : matrix.n_features;
        for (int tb = tb_begin; tb < tb_end; ++tb) {
            const size_t at = size_t(ta) * size_t(matrix.stride) +
                              size_t(tb);
            const double decay = matrix.decays[at];
            if (decay == 0.0) continue;
            for (int i = 0; i < query.seg_count[ta]; ++i) {
                const float* a = query.color_xyz +
                    size_t(query.seg_offset[ta] + i) * 3;
                for (int j = 0; j < library.seg_count[tb]; ++j) {
                    const float* b = library.color_xyz +
                        size_t(library.seg_offset[tb] + j) * 3;
                    double posed[3]{};
                    for (int row = 0; row < 3; ++row)
                        posed[row] = rotation[3*row+0] * b[0] +
                                     rotation[3*row+1] * b[1] +
                                     rotation[3*row+2] * b[2] +
                                     double(reduced_translation[row]) / scale;
                    const double dx = double(a[0]) - posed[0];
                    const double dy = double(a[1]) - posed[1];
                    const double dz = double(a[2]) - posed[2];
                    sum += matrix.coefficients[at] * std::exp(
                        decay * (dx*dx + dy*dy + dz*dz));
                }
            }
        }
    }
    return sum;
}

void test_pose_bytes_and_exact_gradient() {
    const ColorInvariantPlan plan = make_plan();
    const float physical_query_xyz[] = {
        0.13f, -0.24f, 0.32f,
        -0.31f, 0.18f, 0.44f,
        0.29f, 0.37f, -0.12f};
    const std::vector<float> physical_query_coords(
        physical_query_xyz, physical_query_xyz + 9);
    ColorConformer physical_query = make_view(physical_query_coords, 2, 1);
    std::vector<float> query_xyz(9);
    for (size_t i = 0; i < query_xyz.size(); ++i)
        query_xyz[i] = COORD_SCALE * physical_query_xyz[i];
    ReducedColorConformer query;
    query.features = make_view(query_xyz, 2, 1);
    const std::vector<float> library_xyz = {
        0.38f, -0.08f, 0.16f,
        -0.17f, 0.45f, 0.21f,
        0.26f, 0.31f, -0.33f};
    ColorConformer library = make_view(library_xyz, 1, 2);

    const float q_raw[4] = {1.3f, -0.2f, 0.4f, 0.1f};
    const float t_reduced[3] = {0.07f, -0.04f, 0.03f};
    const PreparedReducedPose pose =
        prepare_reduced_pose_fp32(q_raw, t_reduced);
    require(pose.translation[0] == t_reduced[0] &&
            pose.translation[1] == t_reduced[1] &&
            pose.translation[2] == t_reduced[2],
            "reduced translation bytes changed during pose preparation");
    float norm = std::sqrt(q_raw[0]*q_raw[0] + q_raw[1]*q_raw[1] +
                           q_raw[2]*q_raw[2] + q_raw[3]*q_raw[3]);
    require(pose.unit_quaternion[0] == q_raw[0]/norm,
            "pose quaternion was not normalized in FP32");

    ColorConformerPairInput pairs[2] = {
        {&query.features, &library, 3, 3},
        {&query.features, &library, 3, 3}};
    ColorConformerBatch8 batch = pack_color_conformer_batch8(pairs, 2, 3);
    PreparedReducedPose pose_storage[2] = {pose, pose};
    const PreparedReducedPose* poses[COLOR_CONFORMER_LANES]{};
    poses[0] = &pose_storage[0];
    poses[1] = &pose_storage[1];
    ColorEvaluation on[COLOR_CONFORMER_LANES]{}, off[COLOR_CONFORMER_LANES]{};
    uint64_t on_terms[COLOR_CONFORMER_LANES]{}, off_terms[COLOR_CONFORMER_LANES]{};
    ColorConformerCensus on_census{}, off_census{};
    evaluate_color_conformer_batch8_prepared(batch, plan, poses, true, on,
        on_terms, &on_census, nullptr, nullptr, true);
    evaluate_color_conformer_batch8_prepared(batch, plan, poses, true, off,
        off_terms, &off_census, nullptr, nullptr, false);
    require(on_census.last_vector_support_mask ==
            off_census.last_vector_support_mask,
            "plan ON/OFF lane-eligibility masks differ");
    require(on_terms[0] == off_terms[0] && on_terms[1] == off_terms[1],
            "plan ON/OFF term census differs");
    for (int lane = 0; lane < 2; ++lane) {
        require(on[lane].value == off[lane].value,
                "plan ON/OFF analytic values differ");
        for (int k = 1; k < 7; ++k)
            require(on[lane].descent[k] == off[lane].descent[k],
                    "plan ON/OFF analytic gradients differ");
    }

    const double expected = independent_reduced_value(query, library, plan,
                                                      pose);
    const double physical_expected = independent_physical_value(
        physical_query, library, plan, q_raw, t_reduced);
    require(std::abs(expected - physical_expected) <
                3.0e-5 * std::max(1.0, std::abs(physical_expected)),
            "reduced Gaussian value disagrees with independent physical formula");
    require(std::abs(on[0].value - expected) <
                3.0e-3 * std::max(1.0, std::abs(expected)),
            "shared evaluator disagrees with FP64 reduced-frame oracle");

    constexpr float translation_step = 0.001f;
    PreparedReducedPose plus = pose, minus = pose;
    plus.translation[0] += translation_step;
    minus.translation[0] -= translation_step;
    const double d_t = (independent_reduced_value(query, library, plan, plus) -
                        independent_reduced_value(query, library, plan, minus)) /
                       (2.0 * translation_step);
    require(std::abs(on[0].descent[4] + d_t) <
                1.0e-2 * std::max(1.0, std::abs(d_t)),
            "reduced translation descent has an extra/missing scale");

    constexpr float angle_half = 0.21f;
    constexpr float angle_step = 0.001f;
    auto z_pose = [&](float h) {
        const float rotation[4] = {
            std::cos(h), 0.0f, 0.0f, std::sin(h)};
        return prepare_reduced_pose_fp32(rotation, t_reduced);
    };
    const PreparedReducedPose rotation_pose = z_pose(angle_half);
    const ColorConformer* q_pair[1] = {&query.features};
    const ColorConformer* l_pair[1] = {&library};
    ColorConformerPairInput rotation_pair[1] = {{q_pair[0], l_pair[0], 3, 3}};
    const ColorConformerBatch8 rotation_batch =
        pack_color_conformer_batch8(rotation_pair, 1, 3);
    PreparedReducedPose rotation_storage[1] = {rotation_pose};
    const PreparedReducedPose* rotation_ptrs[COLOR_CONFORMER_LANES]{};
    rotation_ptrs[0] = &rotation_storage[0];
    ColorEvaluation rotation_eval[COLOR_CONFORMER_LANES]{};
    uint64_t rotation_terms[COLOR_CONFORMER_LANES]{};
    evaluate_color_conformer_batch8_prepared(rotation_batch, plan,
        rotation_ptrs, true, rotation_eval, rotation_terms);
    const double d_h = (independent_reduced_value(query, library, plan,
                                    z_pose(angle_half + angle_step)) -
                        independent_reduced_value(query, library, plan,
                                    z_pose(angle_half - angle_step))) /
                       (2.0 * angle_step);
    require(std::abs(rotation_eval[0].descent[3] + d_h) <
                2.0e-2 * std::max(1.0, std::abs(d_h)),
            "reduced-pose torque sign/factor differs from finite differences");
}

void test_range_fallback_and_exact_scratch() {
    const ColorInvariantPlan plan = make_plan();
    constexpr int n_pairs = 7;
    const int query_counts[n_pairs][2] = {
        {2, 1}, {1, 0}, {0, 0}, {1, 2}, {0, 1}, {2, 0}, {1, 0}};
    const int library_counts[n_pairs][2] = {
        {1, 2}, {1, 0}, {0, 0}, {0, 2}, {2, 0}, {1, 1}, {1, 0}};
    std::vector<float> query_xyz[n_pairs];
    std::vector<float> library_xyz[n_pairs];
    ReducedColorConformer queries[n_pairs];
    ColorConformer libraries[n_pairs];
    ColorConformerPairInput pairs[n_pairs];
    for (int lane = 0; lane < n_pairs; ++lane) {
        const int q_count = query_counts[lane][0] + query_counts[lane][1];
        const int l_count = library_counts[lane][0] + library_counts[lane][1];
        query_xyz[lane].resize(3 * q_count);
        library_xyz[lane].resize(3 * l_count);
        for (int i = 0; i < q_count; ++i) {
            query_xyz[lane][3*i+0] = COORD_SCALE * (0.13f + 0.04f*i);
            query_xyz[lane][3*i+1] = COORD_SCALE * (-0.11f + 0.03f*i);
            query_xyz[lane][3*i+2] = COORD_SCALE * (0.07f - 0.02f*i);
        }
        for (int i = 0; i < l_count; ++i) {
            library_xyz[lane][3*i+0] = 0.21f + 0.03f*i;
            library_xyz[lane][3*i+1] = -0.17f + 0.02f*i;
            library_xyz[lane][3*i+2] = 0.09f - 0.01f*i;
        }
        queries[lane].features = make_view(query_xyz[lane],
            query_counts[lane][0], query_counts[lane][1]);
        libraries[lane] = make_view(library_xyz[lane],
            library_counts[lane][0], library_counts[lane][1]);
        pairs[lane] = {&queries[lane].features, &libraries[lane], 3, 3};
    }
    // The final lane is deliberately outside the physical-coordinate fast
    // path range. Counts vary by lane and include a true zero-feature plane.
    library_xyz[n_pairs - 1][0] = 2.0e9f;
    const ColorConformerBatch8 batch = pack_color_conformer_batch8(
        pairs, n_pairs, 3);
    const float q[4] = {1, 0, 0, 0};
    const float t[3] = {0, 0, 0};
    const PreparedReducedPose pose = prepare_reduced_pose_fp32(q, t);
    PreparedReducedPose pose_storage[COLOR_CONFORMER_LANES]{};
    const PreparedReducedPose* poses[COLOR_CONFORMER_LANES]{};
    for (int lane = 0; lane < n_pairs; ++lane) {
        pose_storage[lane] = pose;
        poses[lane] = &pose_storage[lane];
    }
    ColorEvaluation on[COLOR_CONFORMER_LANES]{}, off[COLOR_CONFORMER_LANES]{};
    uint64_t on_terms[COLOR_CONFORMER_LANES]{}, off_terms[COLOR_CONFORMER_LANES]{};
    ColorConformerCensus on_census{}, off_census{};
    evaluate_color_conformer_batch8_prepared(batch, plan, poses, true, on,
        on_terms, &on_census, nullptr, nullptr, true);
    evaluate_color_conformer_batch8_prepared(batch, plan, poses, true, off,
        off_terms, &off_census, nullptr, nullptr, false);
    require(on_census.last_vector_support_mask ==
            off_census.last_vector_support_mask,
            "plan ON/OFF fallback fixture lane masks differ");
    require((on_census.last_vector_support_mask & (1u << (n_pairs - 1))) == 0,
            "out-of-range target lane was marked AVX2 eligible");
    require(on_census.scalar_fallback_lane_calls[n_pairs - 1] == 1 &&
            off_census.scalar_fallback_lane_calls[n_pairs - 1] == 1,
            "out-of-range target lane did not use scalar fallback in both modes");
    for (int lane = 0; lane < n_pairs; ++lane) {
        require(on_terms[lane] == off_terms[lane],
                "plan ON/OFF fallback fixture term census differs");
        require(on[lane].value == off[lane].value,
                "plan ON/OFF fallback fixture values differ");
        for (int k = 1; k < 7; ++k)
            require(on[lane].descent[k] == off[lane].descent[k],
                    "plan ON/OFF fallback fixture gradients differ");
    }

    ExactFinalScratch scratch;
    scratch.begin_trajectory(queries[0], libraries[0], plan, 3, 3);
    uint64_t start_terms = 0, winner_terms = 0;
    const ColorEvaluation start = score_exact_final_color(
        queries[0], libraries[0], plan, pose, 3, 3, &scratch, &start_terms);
    scratch.retain_current_pose_for_winner(pose);
    const ColorEvaluation winner = score_exact_final_color_retained(
        queries[0], libraries[0], plan, pose, 3, 3, scratch, &winner_terms);
    require(start.value == winner.value,
            "retained winner did not run the same exact overlap");
    require(start_terms == winner_terms && start_terms != 0,
            "retained winner did not count a distinct exact pass");
}

}  // namespace

int main() {
    try {
        test_pose_bytes_and_exact_gradient();
        test_range_fallback_and_exact_scratch();
    } catch (const std::exception& error) {
        std::cerr << "FAIL T191 reduced pose/color: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS T191 reduced pose/color\n";
    return 0;
}

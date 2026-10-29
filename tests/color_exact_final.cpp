// Fixed-pose exact scorer/scratch parity and lifetime oracle.
#include "color_exact_final.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kernel.hpp"

namespace {

using namespace r3::cpu;
constexpr int kNFeatures = 7;
constexpr int kMatrixStride = MAX_COLOR_FEATURES;
constexpr int kCoordinateStride = 4;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

bool same_pose_bytes(const PreparedReducedPose& a,
                     const PreparedReducedPose& b) {
    return std::memcmp(a.matrix, b.matrix, sizeof(a.matrix)) == 0 &&
        std::memcmp(a.translation, b.translation,
                    sizeof(a.translation)) == 0;
}

struct Inputs {
    std::array<uint8_t, MAX_COLOR_FEATURES> query_counts{};
    std::array<uint8_t, MAX_COLOR_FEATURES> library_counts{};
    std::vector<float> reduced_query_xyz;
    std::vector<float> physical_library_xyz;
    ReducedColorConformer query{};
    ColorConformer library{};
};

Inputs make_inputs(bool empty_query = false, bool empty_library = false) {
    Inputs result;
    if (!empty_query)
        result.query_counts = {0, 2, 0, 2, 1, 3, 3};  // 11 real colors.
    if (!empty_library)
        result.library_counts = {0, 3, 2, 0, 4, 3, 4};  // 16 real colors.

    int query_points = 0;
    int library_points = 0;
    for (int type = 1; type < kNFeatures; ++type) {
        query_points += result.query_counts[type];
        library_points += result.library_counts[type];
    }
    result.reduced_query_xyz.resize(std::max<size_t>(
        size_t(query_points) * 3, 3));
    for (int j = 0; j < query_points; ++j) {
        const float x = float((j % 5) - 2) * 0.21f + 0.03f;
        const float y = float((j % 4) - 1) * -0.17f + 0.11f;
        const float z = float((j % 7) - 3) * 0.09f - 0.07f;
        result.reduced_query_xyz[size_t(j) * 3] = COORD_SCALE * x;
        result.reduced_query_xyz[size_t(j) * 3 + 1] = COORD_SCALE * y;
        result.reduced_query_xyz[size_t(j) * 3 + 2] = COORD_SCALE * z;
    }
    result.physical_library_xyz.resize(std::max<size_t>(
        size_t(library_points) * kCoordinateStride, kCoordinateStride));
    for (int j = 0; j < library_points; ++j) {
        result.physical_library_xyz[size_t(j) * kCoordinateStride] =
            float((j % 6) - 2) * 0.14f - 0.08f;
        result.physical_library_xyz[size_t(j) * kCoordinateStride + 1] =
            float((j % 5) - 1) * 0.19f + 0.04f;
        result.physical_library_xyz[size_t(j) * kCoordinateStride + 2] =
            float((j % 8) - 3) * -0.12f + 0.02f;
        result.physical_library_xyz[size_t(j) * kCoordinateStride + 3] =
            1000.0f + float(j);  // Padding lane must never be read.
    }

    result.query.features = build_color_view_from_counts(
        result.reduced_query_xyz.empty() ? nullptr :
            result.reduced_query_xyz.data(),
        result.query_counts.data(), kNFeatures, 3);
    result.library = build_color_view_from_counts(
        result.physical_library_xyz.empty() ? nullptr :
            result.physical_library_xyz.data(),
        result.library_counts.data(), kNFeatures, kCoordinateStride);
    return result;
}

struct MatrixStorage {
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> coefficients{};
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> decays{};

    explicit MatrixStorage(bool canonical) {
        for (int query_type = 1; query_type < kNFeatures; ++query_type) {
            for (int target_type = 1; target_type < kNFeatures; ++target_type) {
                if (canonical && target_type < query_type) continue;
                const size_t at = size_t(query_type) * kMatrixStride +
                                  size_t(target_type);
                coefficients[at] = query_type == target_type
                    ? 0.65 + 0.08 * query_type
                    : 0.12 + 0.01 * (query_type + 2 * target_type);
                decays[at] = -0.035 * (1 + query_type + target_type);
            }
        }
        // Nonzero width with zero coefficient still counts as a term.
        coefficients[size_t(1) * kMatrixStride + 4] = 0.0;
        decays[size_t(1) * kMatrixStride + 4] = -0.27;
        // Zero width with nonzero coefficient is skipped entirely.
        coefficients[size_t(2) * kMatrixStride + 5] = 9.0;
        decays[size_t(2) * kMatrixStride + 5] = 0.0;
    }

    ColorInvariantPlan plan(bool canonical) const {
        const ColorInteractionMatrixView input{
            kNFeatures, kMatrixStride, canonical, coefficients.data(),
            decays.data()};
        return ColorInvariantPlan(input);
    }
};

struct CanonicalIdentityMatrixStorage {
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> coefficients{};
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> decays{};

    CanonicalIdentityMatrixStorage() {
        for (int query_type = 1; query_type < kNFeatures; ++query_type)
            for (int target_type = 1; target_type < kNFeatures; ++target_type) {
                const size_t at = size_t(query_type) * kMatrixStride +
                                  size_t(target_type);
                decays[at] = -0.5;
                if (query_type == target_type) coefficients[at] = 1.25;
            }
    }

    ColorInvariantPlan plan() const {
        return ColorInvariantPlan({kNFeatures, kMatrixStride, true,
            coefficients.data(), decays.data()});
    }
};

PreparedReducedPose pose(float angle_x, float angle_z,
                         float tx, float ty, float tz) {
    // Compose two deterministic axis rotations in a float quaternion.
    const float q[4] = {
        std::cos(angle_x * 0.5f) * std::cos(angle_z * 0.5f),
        std::sin(angle_x * 0.5f) * std::cos(angle_z * 0.5f),
        std::sin(angle_x * 0.5f) * std::sin(angle_z * 0.5f),
        std::cos(angle_x * 0.5f) * std::sin(angle_z * 0.5f)};
    const float t[3] = {tx, ty, tz};
    return prepare_reduced_pose_fp32(q, t);
}

uint64_t expected_terms(const ColorConformer& query,
                        const ColorConformer& library,
                        const ColorInvariantPlan& plan) {
    uint64_t result = 0;
    for (int query_type = 1; query_type < plan.n_features(); ++query_type) {
        const int target_begin = plan.canonical() ? query_type : 1;
        const int target_end = plan.canonical()
            ? query_type + 1 : plan.n_features();
        for (int target_type = target_begin; target_type < target_end;
             ++target_type) {
            if (plan.reduced_decay(query_type, target_type) == 0.0) continue;
            result += uint64_t(query.seg_count[query_type]) *
                      uint64_t(library.seg_count[target_type]);
        }
    }
    return result;
}

double canonical_fp32_oracle(const ReducedColorConformer& query,
        const ColorConformer& library, const ColorInvariantPlan& plan,
        const PreparedReducedPose& pose, int query_stride, int library_stride) {
    float total = 0.0f;
    for (const ColorCanonicalPlane& plane : plan.canonical_planes()) {
        const int type = plane.type;
        for (int i = 0; i < query.features.seg_count[type]; ++i) {
            const float* x = query.features.color_xyz +
                size_t(query.features.seg_offset[type] + i) *
                    size_t(query_stride);
            for (int j = 0; j < library.seg_count[type]; ++j) {
                const float* raw = library.color_xyz +
                    size_t(library.seg_offset[type] + j) *
                        size_t(library_stride);
                const float* m = pose.matrix;
                const float* t = pose.translation;
                const float bx = m[0] * raw[0] + m[1] * raw[1] +
                                 m[2] * raw[2] + t[0];
                const float by = m[3] * raw[0] + m[4] * raw[1] +
                                 m[5] * raw[2] + t[1];
                const float bz = m[6] * raw[0] + m[7] * raw[1] +
                                 m[8] * raw[2] + t[2];
                const float dx = x[0] - bx;
                const float dy = x[1] - by;
                const float dz = x[2] - bz;
                const float distance_squared =
                    dx * dx + dy * dy + dz * dz;
                const float decay = float(plan.reduced_decay(type, type));
                const float coefficient = float(plan.coefficient(type, type));
                total += coefficient * std::exp(decay * distance_squared);
            }
        }
    }
    return double(total);
}

void test_four_plus_one(const bool canonical) {
    Inputs input = make_inputs();
    MatrixStorage matrix_storage(canonical);
    const ColorInvariantPlan matrix = matrix_storage.plan(canonical);
    const uint64_t terms_per_call = expected_terms(input.query.features,
                                                   input.library, matrix);
    require(terms_per_call > 0, "fixture should issue overlap terms");

    ExactFinalScratch cached;
    ExactFinalScratchStats cached_stats;
    cached.begin_trajectory(input.query, input.library, matrix, 3,
                            kCoordinateStride);
    ExactFinalScratchStats uncached_stats;
    uint64_t cached_terms = 0;
    uint64_t uncached_terms = 0;
    double best_cached = -1.0;
    double best_uncached = -1.0;
    int cached_start = -1;
    int uncached_start = -1;
    PreparedReducedPose winning_pose{};
    const PreparedReducedPose winning_post_update_pose =
        pose(0.46f, -0.38f, 0.31f, -0.14f, 0.22f);
    const PreparedReducedPose prior_gradient_pose =
        pose(-0.21f, 0.17f, -0.24f, 0.33f, -0.19f);
    require(!same_pose_bytes(prior_gradient_pose,
                             winning_post_update_pose),
            "post-update pose fixture must differ from gradient pose");

    for (int start = 0; start < 4; ++start) {
        // The final scored pose is explicitly the value after the last update.
        const PreparedReducedPose final_pose = start == 3
            ? winning_post_update_pose
            : pose(0.12f * float(start), -0.09f * float(start),
                   0.07f * float(start), -0.05f, 0.03f);
        const ColorEvaluation cached_result = score_exact_final_color(
            input.query, input.library, matrix, final_pose, 3,
            kCoordinateStride, &cached, &cached_terms, &cached_stats);
        const ColorEvaluation uncached_result = score_exact_final_color(
            input.query, input.library, matrix, final_pose, 3,
            kCoordinateStride, nullptr, &uncached_terms, &uncached_stats);
        require(same_bits(cached_result.value, uncached_result.value),
                "cached and uncached per-start score bits differ");
        // Strict > preserves the earliest start on exact ties.
        if (cached_result.value > best_cached) {
            best_cached = cached_result.value;
            cached_start = start;
            winning_pose = final_pose;
            cached.retain_current_pose_for_winner(winning_pose,
                                                   &cached_stats);
        }
        if (uncached_result.value > best_uncached) {
            best_uncached = uncached_result.value;
            uncached_start = start;
        }
    }
    require(cached_start == uncached_start,
            "cached and uncached winner starts differ");
    require(cached_terms == 4 * terms_per_call &&
            uncached_terms == 4 * terms_per_call,
            "four separate exact per-start term censuses were not preserved");

    uint64_t cached_winner_terms = 0;
    uint64_t uncached_winner_terms = 0;
    const ColorEvaluation cached_winner = score_exact_final_color_retained(
        input.query, input.library, matrix, winning_pose, 3,
        kCoordinateStride, cached, &cached_winner_terms);
    const ColorEvaluation uncached_winner = score_exact_final_color(
        input.query, input.library, matrix, winning_pose, 3,
        kCoordinateStride, nullptr, &uncached_winner_terms, &uncached_stats);
    require(same_bits(cached_winner.value, uncached_winner.value) &&
            same_bits(cached_winner.value, best_cached),
            "retained winner is not a distinct exact score at the winner pose");
    require(cached_winner_terms == terms_per_call &&
            uncached_winner_terms == terms_per_call,
            "distinct fifth exact winner term census was lost");
    require(cached_stats.winner_candidates_retained >= 1 &&
            cached_stats.winner_point_bytes_copied >=
                input.library.n_color * 3 *
                    (matrix.canonical_analytic_supported()
                        ? sizeof(float) : sizeof(double)),
            "winner point retention was not counted");
    require(cached_stats.library_points_prepared ==
                4 * uint64_t(input.library.n_color) &&
            uncached_stats.library_points_prepared ==
                5 * uint64_t(input.library.n_color),
            "transformed point preparation count does not show reuse");
    require(cached_stats.buffer_growths == 2 &&
            uncached_stats.buffer_growths == 5,
            "scratch allocation growth counts do not match the arms");
}

void test_retained_lifetime_and_invalidations() {
    Inputs input = make_inputs();
    MatrixStorage storage(false);
    const ColorInvariantPlan matrix = storage.plan(false);
    ExactFinalScratch scratch;
    scratch.begin_trajectory(input.query, input.library, matrix, 3,
                             kCoordinateStride);
    const PreparedReducedPose final_pose = pose(0.34f, 0.15f,
                                                -0.18f, 0.22f, 0.07f);
    const ColorEvaluation before_mutation = score_exact_final_color(
        input.query, input.library, matrix, final_pose, 3,
        kCoordinateStride, &scratch);
    scratch.retain_current_pose_for_winner(final_pose);

    // The retained coordinates own their bytes; rescoring does not dereference
    // the caller's library coordinate buffer after preparation returned.
    for (size_t i = 0; i < input.physical_library_xyz.size(); ++i)
        input.physical_library_xyz[i] += float(i + 1) * 0.91f;
    const ColorEvaluation after_mutation = score_exact_final_color_retained(
        input.query, input.library, matrix, final_pose, 3,
        kCoordinateStride, scratch);
    require(same_bits(before_mutation.value, after_mutation.value),
            "retained winner reread caller library coordinates");

    ColorConformer other_library = input.library;
    std::vector<float> other_xyz(input.physical_library_xyz.size(), 0.0f);
    other_library.color_xyz = other_xyz.data();
    bool rejected = false;
    try {
        (void)score_exact_final_color_retained(input.query, other_library,
            matrix, final_pose, 3, kCoordinateStride, scratch);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "retained winner accepted a different library buffer");

    // Invalid start of another trajectory must invalidate the prior winner
    // even though validation fails before a new trajectory can be installed.
    rejected = false;
    try {
        scratch.begin_trajectory(input.query, input.library, matrix, 2,
                                 kCoordinateStride);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid trajectory fixture was accepted");
    rejected = false;
    try {
        (void)score_exact_final_color_retained(input.query, input.library,
            matrix, final_pose, 3, kCoordinateStride, scratch);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "failed trajectory begin left the previous winner valid");

    scratch.begin_trajectory(input.query, input.library, matrix, 3,
                             kCoordinateStride);
    rejected = false;
    try {
        (void)score_exact_final_color_retained(input.query, input.library,
            matrix, final_pose, 3, kCoordinateStride, scratch);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "new trajectory reused a prior winner before scoring");

    // The same worker scratch can be reused for a subsequent valid pair, but
    // its winner is not available until that pair has produced a new strict-gt
    // selection. The current/winner capacities remain reusable.
    const size_t retained_capacity = scratch.capacity_points();
    const PreparedReducedPose next_pose = pose(-0.16f, 0.29f,
                                               0.05f, -0.12f, 0.08f);
    const auto next_score = score_exact_final_color(input.query,
        input.library, matrix, next_pose, 3, kCoordinateStride, &scratch);
    scratch.retain_current_pose_for_winner(next_pose);
    const auto next_winner = score_exact_final_color_retained(input.query,
        input.library, matrix, next_pose, 3, kCoordinateStride, scratch);
    require(same_bits(next_score.value, next_winner.value) &&
            scratch.capacity_points() == retained_capacity,
            "repeated trajectory call did not reuse scratch safely");
}

void test_stable_ties_and_post_update_pose() {
    Inputs input = make_inputs();
    CanonicalIdentityMatrixStorage storage;
    const ColorInvariantPlan matrix = storage.plan();
    require(matrix.canonical_analytic_supported(),
            "identity fixture did not select the canonical analytic route");
    ExactFinalScratch scratch;
    scratch.begin_trajectory(input.query, input.library, matrix, 3,
                             kCoordinateStride);
    const PreparedReducedPose tied_pose =
        pose(0.18f, -0.11f, 0.09f, 0.04f, -0.06f);
    double best = -1.0;
    int best_start = -1;
    uint64_t per_start_terms = 0;
    for (int start = 0; start < 4; ++start) {
        const auto value = score_exact_final_color(input.query,
            input.library, matrix, tied_pose, 3, kCoordinateStride,
            &scratch, &per_start_terms);
        if (value.value > best) {
            best = value.value;
            best_start = start;
            scratch.retain_current_pose_for_winner(tied_pose);
        }
    }
    require(best_start == 0,
            "strict-greater winner selection did not preserve earliest tie");
    require(per_start_terms == 4 * expected_terms(input.query.features,
                                                   input.library, matrix),
            "tie fixture lost one of its four start evaluations");
    const auto retained = score_exact_final_color_retained(input.query,
        input.library, matrix, tied_pose, 3, kCoordinateStride, scratch);
    require(same_bits(best, retained.value),
            "tie winner's separate fifth score changed");
    const size_t expected_bytes = size_t(input.library.n_color) * 3 *
                                  sizeof(float);
    require(scratch.capacity_bytes() == expected_bytes &&
            scratch.winner_capacity_bytes() == expected_bytes,
            "canonical worker scratch did not retain FP32 coordinates");
    const double fp32_expected = canonical_fp32_oracle(input.query,
        input.library, matrix, tied_pose, 3, kCoordinateStride);
    require(std::abs(best - fp32_expected) <=
                2.0e-6 * std::max(1.0, std::abs(fp32_expected)),
            "canonical exact final differs from the accurate FP32 analytic oracle");

    // A raw target coordinate beyond the FP32 fast-route contract must use
    // the existing double fallback, including when the worker scratch changes
    // precision after a prior canonical FP32 score.
    input.physical_library_xyz[0] = 1.2e9f;
    const auto fallback = score_exact_final_color(input.query, input.library,
        matrix, tied_pose, 3, kCoordinateStride, &scratch);
    const size_t fallback_bytes = size_t(input.library.n_color) * 3 *
                                  sizeof(double);
    require(std::isfinite(fallback.value) &&
            scratch.capacity_bytes() == fallback_bytes,
            "unsafe canonical coordinates did not use double fallback scratch");
    scratch.retain_current_pose_for_winner(tied_pose);
    const auto retained_fallback = score_exact_final_color_retained(
        input.query, input.library, matrix, tied_pose, 3,
        kCoordinateStride, scratch);
    require(same_bits(fallback.value, retained_fallback.value) &&
            scratch.winner_capacity_bytes() == fallback_bytes,
            "unsafe canonical fallback did not retain its double coordinates");

    // A deliberately stale pre-update pose has a distant translation and
    // therefore a different exact score from the post-update final pose.
    const PreparedReducedPose stale_gradient =
        pose(-0.12f, 0.23f, 30.0f, -15.0f, 9.0f);
    const PreparedReducedPose after_last_update =
        pose(0.27f, -0.19f, 0.0f, 0.0f, 0.0f);
    const auto stale_score = score_exact_final_color(input.query,
        input.library, matrix, stale_gradient, 3, kCoordinateStride, nullptr);
    const auto final_score = score_exact_final_color(input.query,
        input.library, matrix, after_last_update, 3, kCoordinateStride,
        nullptr);
    require(!same_bits(stale_score.value, final_score.value),
            "post-update fixture cannot distinguish stale and final poses");
}

void test_empty_and_concurrent() {
    MatrixStorage storage(false);
    const ColorInvariantPlan matrix = storage.plan(false);
    Inputs empty_query = make_inputs(true, false);
    ExactFinalScratch scratch;
    scratch.begin_trajectory(empty_query.query, empty_query.library, matrix,
                             3, kCoordinateStride);
    const PreparedReducedPose identity = pose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    uint64_t terms = 0;
    const auto zero = score_exact_final_color(empty_query.query,
        empty_query.library, matrix, identity, 3, kCoordinateStride,
        &scratch, &terms);
    require(zero.value == 0.0 && terms == 0,
            "empty query plane should produce no overlap terms");

    Inputs empty_library = make_inputs(false, true);
    ExactFinalScratch empty_scratch;
    empty_scratch.begin_trajectory(empty_library.query,
        empty_library.library, matrix, 3, kCoordinateStride);
    terms = 0;
    const auto empty = score_exact_final_color(empty_library.query,
        empty_library.library, matrix, identity, 3, kCoordinateStride,
        &empty_scratch, &terms);
    require(empty.value == 0.0 && terms == 0,
            "empty library plane should produce no overlap terms");

    Inputs shared = make_inputs();
    const PreparedReducedPose shared_pose =
        pose(-0.29f, 0.37f, 0.13f, -0.08f, 0.19f);
    const auto reference = score_exact_final_color(shared.query,
        shared.library, matrix, shared_pose, 3, kCoordinateStride, nullptr);
    std::array<double, 4> values{};
    std::array<std::thread, 4> workers;
    for (size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::thread([&, worker] {
            ExactFinalScratch local;
            local.begin_trajectory(shared.query, shared.library, matrix, 3,
                                   kCoordinateStride);
            const auto actual = score_exact_final_color(shared.query,
                shared.library, matrix, shared_pose, 3, kCoordinateStride,
                &local);
            values[worker] = actual.value;
        });
    }
    for (auto& worker : workers) worker.join();
    for (double value : values)
        require(same_bits(value, reference.value),
                "independent worker scratch changed exact score bits");
}

}  // namespace

int main() {
    try {
        test_four_plus_one(true);
        test_four_plus_one(false);
        test_retained_lifetime_and_invalidations();
        test_stable_ties_and_post_update_pose();
        test_empty_and_concurrent();
    } catch (const std::exception& error) {
        std::cerr << "FAIL T192 exact-final scratch: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS T192 exact-final scratch parity/lifetime\n";
    return 0;
}

// Fixed-pose correctness gate for independent-conformer AVX2 lanes.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "color_conformer_kernel.hpp"
#include "kernel.hpp"

using namespace r3::cpu;

namespace {
constexpr int kN = 4;
constexpr int kStride = 4;

size_t coordinate_index(size_t base, int feature, int component, int lane) {
    return base + (size_t(feature) * 3 + size_t(component)) *
                      size_t(COLOR_CONFORMER_LANES) + size_t(lane);
}

struct OwnedConformer {
    std::vector<float> xyz;
    ColorConformer view{};
};

OwnedConformer make_conformer(const int counts[kN], int lane, bool library) {
    OwnedConformer owned;
    int total = 0;
    owned.view.n_features = kN;
    for (int t = 1; t < kN; ++t) {
        owned.view.seg_offset[t] = total;
        owned.view.seg_count[t] = counts[t];
        total += counts[t];
    }
    owned.view.n_color = total;
    owned.xyz.assign(size_t(total) * kStride, -777.0f);
    for (int t = 1; t < kN; ++t) {
        for (int j = 0; j < counts[t]; ++j) {
            const size_t at = size_t(owned.view.seg_offset[t] + j) * kStride;
            const float direction = library ? -1.0f : 1.0f;
            owned.xyz[at] = direction * (0.13f * (lane + 1) + 0.17f * t + 0.09f * j);
            owned.xyz[at + 1] = direction * (-0.21f * lane + 0.11f * t - 0.07f * j);
            owned.xyz[at + 2] = direction * (0.08f * lane - 0.19f * t + 0.14f * j);
        }
    }
    owned.view.color_xyz = owned.xyz.data();
    return owned;
}

void pose(const float q[4], const float tr[3], double m[9]) {
    const double norm = std::sqrt(double(q[0])*q[0] + double(q[1])*q[1] +
                                  double(q[2])*q[2] + double(q[3])*q[3]);
    const double w = q[0]/norm, x = q[1]/norm, y = q[2]/norm, z = q[3]/norm;
    const double values[9] = {
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)};
    std::copy(values, values+9, m);
    (void)tr;
}

ColorEvaluation exact(const OwnedConformer& a, const OwnedConformer& b,
        const ColorInteractionMatrixView& matrix, int lane,
        const float q[4], const float tr[3], uint64_t* terms) {
    double rotation[9];
    pose(q, tr, rotation);
    ColorEvaluation out;
    *terms = 0;
    for (int ta = 1; ta < matrix.n_features; ++ta) {
        for (int tb = 1; tb < matrix.n_features; ++tb) {
            const double decay = matrix.decay(ta, tb);
            if (decay == 0.0) continue;
            const double width = -2.0 * decay;
            for (int i = 0; i < a.view.seg_count[ta]; ++i) {
                const size_t ai = size_t(a.view.seg_offset[ta] + i) * kStride;
                for (int j = 0; j < b.view.seg_count[tb]; ++j) {
                    const size_t bi = size_t(b.view.seg_offset[tb] + j) * kStride;
                    double y[3];
                    for (int k = 0; k < 3; ++k)
                        y[k] = rotation[3*k] * b.xyz[bi] +
                               rotation[3*k+1] * b.xyz[bi+1] +
                               rotation[3*k+2] * b.xyz[bi+2] + tr[k];
                    const double dx = double(a.xyz[ai]) - y[0];
                    const double dy = double(a.xyz[ai+1]) - y[1];
                    const double dz = double(a.xyz[ai+2]) - y[2];
                    const double value = matrix.coefficient(ta, tb) *
                        std::exp(decay * (dx*dx + dy*dy + dz*dz));
                    out.value += value;
                    const double rx = y[0] - tr[0];
                    const double ry = y[1] - tr[1];
                    const double rz = y[2] - tr[2];
                    const double torque = -2.0 * width * value;
                    out.descent[1] += torque * (ry*dz-rz*dy);
                    out.descent[2] += torque * (rz*dx-rx*dz);
                    out.descent[3] += torque * (rx*dy-ry*dx);
                    const double translate = -width * value / double(COORD_SCALE);
                    out.descent[4] += translate * dx;
                    out.descent[5] += translate * dy;
                    out.descent[6] += translate * dz;
                    ++*terms;
                }
            }
        }
    }
    (void)lane;
    return out;
}

ColorEvaluation exact_reduced_canonical(const OwnedConformer& query,
        const OwnedConformer& library, const ColorInvariantPlan& plan,
        const PreparedReducedPose& pose, uint64_t* terms) {
    const ColorInvariantMatrixView matrix = plan.reduced_matrix_view();
    ColorEvaluation out;
    *terms = 0;
    for (int target_type = 1; target_type < matrix.n_features;
         ++target_type) {
        for (int query_type = 1; query_type <= target_type; ++query_type) {
            const size_t cell = size_t(query_type) *
                size_t(matrix.stride) + size_t(target_type);
            const double decay = matrix.decays[cell];
            if (decay == 0.0) continue;
            const double coefficient = matrix.coefficients[cell];
            const double width = -2.0 * decay;
            for (int i = 0; i < query.view.seg_count[query_type]; ++i) {
                const size_t qi = size_t(
                    query.view.seg_offset[query_type] + i) *
                    size_t(kStride);
                for (int j = 0;
                     j < library.view.seg_count[target_type]; ++j) {
                    const size_t bi = size_t(
                        library.view.seg_offset[target_type] + j) *
                        size_t(kStride);
                    double posed[3]{};
                    for (int row = 0; row < 3; ++row)
                        posed[row] = double(pose.matrix[3*row]) *
                            double(library.xyz[bi]) +
                        double(pose.matrix[3*row+1]) *
                            double(library.xyz[bi+1]) +
                        double(pose.matrix[3*row+2]) *
                            double(library.xyz[bi+2]) +
                        double(pose.translation[row]);
                    const double dx = double(query.xyz[qi]) - posed[0];
                    const double dy = double(query.xyz[qi+1]) - posed[1];
                    const double dz = double(query.xyz[qi+2]) - posed[2];
                    const double value = coefficient * std::exp(
                        decay * (dx*dx + dy*dy + dz*dz));
                    out.value += value;
                    const double rx = posed[0] - double(pose.translation[0]);
                    const double ry = posed[1] - double(pose.translation[1]);
                    const double rz = posed[2] - double(pose.translation[2]);
                    const double torque = -2.0 * width * value;
                    out.descent[1] += torque * (ry*dz-rz*dy);
                    out.descent[2] += torque * (rz*dx-rx*dz);
                    out.descent[3] += torque * (rx*dy-ry*dx);
                    const double translate = -width * value;
                    out.descent[4] += translate * dx;
                    out.descent[5] += translate * dy;
                    out.descent[6] += translate * dz;
                    ++*terms;
                }
            }
        }
    }
    return out;
}

bool close(double actual, double expected) {
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= 2.5e-3 * std::max(1.0, std::abs(expected));
}
}  // namespace

int main(int argc, char** argv) {
#if defined(__AVX2__)
    __builtin_cpu_init();
    const bool vector_available = __builtin_cpu_supports("avx2") &&
                                  __builtin_cpu_supports("fma");
#else
    const bool vector_available = false;
#endif
    constexpr int kPairs = 7;  // one real tail lane, not a full AVX2 block
    std::array<OwnedConformer, kPairs> query, library;
    std::array<ColorConformerPairInput, kPairs> inputs{};
    std::array<std::array<int, kN>, kPairs> query_counts{}, library_counts{};
    for (int lane = 0; lane < kPairs; ++lane) {
        query_counts[lane] = {0, lane % 3 == 0 ? 0 : lane % 4 + 1,
                              lane == 0 ? 0 : lane % 3 + 1, lane % 2};
        library_counts[lane] = {0, lane % 5, lane % 3 == 0 ? 0 : lane % 4,
                                lane % 3 + 1};
        query[lane] = make_conformer(query_counts[lane].data(), lane, false);
        library[lane] = make_conformer(library_counts[lane].data(), lane, true);
        inputs[lane] = {&query[lane].view, &library[lane].view,
                        kStride, kStride};
    }
    // A valid, nonzero, extreme-coordinate pair must take the exact fallback.
    const int lane = kPairs - 1;
    const size_t q_extreme = size_t(query[lane].view.seg_offset[2]) * kStride;
    const size_t b_extreme = size_t(library[lane].view.seg_offset[3]) * kStride;
    query[lane].xyz[q_extreme] = 1.2e9f;
    query[lane].xyz[q_extreme+1] = 0.25f;
    query[lane].xyz[q_extreme+2] = -0.5f;
    library[lane].xyz[b_extreme] = 1.2e9f;
    library[lane].xyz[b_extreme+1] = 0.25f;
    library[lane].xyz[b_extreme+2] = -0.5f;

    double coefficients[MAX_COLOR_FEATURES * MAX_COLOR_FEATURES]{};
    double decays[MAX_COLOR_FEATURES * MAX_COLOR_FEATURES]{};
    auto set = [&](int ta, int tb, double c, double d) {
        coefficients[size_t(ta) * MAX_COLOR_FEATURES + tb] = c;
        decays[size_t(ta) * MAX_COLOR_FEATURES + tb] = d;
    };
    set(1, 1, 0.7, -0.4);
    set(1, 2, 0.4, -0.2);
    set(2, 1, 1.1, -0.8);  // deliberate asymmetry
    set(2, 3, 0.8, -0.3);
    set(3, 2, 0.5, -0.6);
    set(3, 3, 0.0, -0.1);  // terms are counted even with zero coefficient
    const ColorInteractionMatrixView matrix{
        kN, MAX_COLOR_FEATURES, false, coefficients, decays};
    ColorConformerBatch8 batch = pack_color_conformer_batch8(
        inputs.data(), kPairs, kN);
    const size_t b_extreme_packed = coordinate_index(
        batch.library_coordinate_offset[3], 0, 0, lane);

    // Poison every padded feature slot: per-segment masks must prevent NaNs
    // in inactive query/library tails from reaching values or gradients.
    const float poison = std::numeric_limits<float>::quiet_NaN();
    auto coord = [](size_t base, int item, int component, int vector_lane) {
        return base + (size_t(item) * 3 + size_t(component)) *
                      COLOR_CONFORMER_LANES + size_t(vector_lane);
    };
    for (int t = 1; t < kN; ++t) {
        for (int item = 0; item < batch.query_max[t]; ++item)
            for (int l = 0; l < kPairs; ++l)
                if (item >= batch.query_count[t][l])
                    for (int k = 0; k < 3; ++k)
                        batch.query_xyz[coord(batch.query_coordinate_offset[t],
                                             item, k, l)] = poison;
        for (int item = 0; item < batch.library_max[t]; ++item)
            for (int l = 0; l < kPairs; ++l)
                if (item >= batch.library_count[t][l])
                    for (int k = 0; k < 3; ++k)
                        batch.library_xyz[coord(batch.library_coordinate_offset[t],
                                               item, k, l)] = poison;
    }

    alignas(32) float quaternions[COLOR_CONFORMER_LANES][4]{};
    alignas(32) float translations[COLOR_CONFORMER_LANES][3]{};
    for (int l = 0; l < kPairs; ++l) {
        quaternions[l][0] = l == lane ? 1.0f : 0.85f;
        quaternions[l][1] = l == lane ? 0.0f : 0.07f * float(l + 1);
        quaternions[l][2] = l == lane ? 0.0f : -0.025f * float(l + 1);
        quaternions[l][3] = l == lane ? 0.0f : 0.04f * float(l + 1);
        translations[l][0] = l == lane ? 0.0f : 0.012f * float(l + 1);
        translations[l][1] = l == lane ? 0.0f : -0.009f * float(l + 1);
        translations[l][2] = l == lane ? 0.0f : 0.005f * float(l + 1);
    }

    ColorEvaluation output[COLOR_CONFORMER_LANES];
    uint64_t term_counts[COLOR_CONFORMER_LANES]{};
    ColorConformerCensus census;
    double pose_seconds = 0.0, screen_seconds = 0.0;
    double fallback_seconds = 0.0, pack_seconds = 0.0, vector_seconds = 0.0;
    ColorConformerPhaseTimes phase_times{
        &pose_seconds, &screen_seconds, &fallback_seconds, &pack_seconds,
        &vector_seconds};
    ColorConformerWorkspace workspace;
    const float cache_poison = std::numeric_limits<float>::quiet_NaN();
    workspace.posed_library_xyz_fp32.assign(batch.library_xyz.size(),
                                             cache_poison);
    evaluate_color_conformer_batch8(batch, matrix, quaternions, translations,
        true, output, term_counts, &census, &phase_times, &workspace);
    bool ok = true;
    for (int l = 0; l < kPairs; ++l) {
        uint64_t expected_terms = 0;
        const ColorEvaluation expected = exact(query[l], library[l], matrix,
            l, quaternions[l], translations[l],
            &expected_terms);
        if (!close(output[l].value, expected.value)) {
            std::printf("FAIL lane=%d value=%.9g expected=%.9g\n", l,
                        output[l].value, expected.value);
            ok = false;
        }
        for (int k = 1; k < 7; ++k)
            if (!close(output[l].descent[k], expected.descent[k])) {
                std::printf("FAIL lane=%d descent[%d]=%.9g expected=%.9g\n",
                            l, k, output[l].descent[k], expected.descent[k]);
                ok = false;
            }
        if (term_counts[l] != expected_terms) {
            std::printf("FAIL lane=%d terms=%llu expected=%llu\n", l,
                (unsigned long long)term_counts[l],
                (unsigned long long)expected_terms);
            ok = false;
        }
    }
    if (output[7].value != 0.0 || term_counts[7] != 0) {
        std::puts("FAIL partial batch wrote an invalid output lane");
        ok = false;
    }
    const uint64_t expected_fallbacks = vector_available ? 1 : kPairs;
    if (census.scalar_fallback_pairs != expected_fallbacks) {
        std::printf("FAIL exact fallback count=%llu\n",
                    (unsigned long long)census.scalar_fallback_pairs);
        ok = false;
    }
    const uint8_t expected_support_mask = vector_available ? 0x3f : 0x00;
    if (census.last_vector_support_mask != expected_support_mask) {
        std::printf("FAIL physical support mask=0x%02x expected=0x%02x\n",
                    unsigned(census.last_vector_support_mask),
                    unsigned(expected_support_mask));
        ok = false;
    }
    if (vector_available) {
        for (int l = 0; l < kPairs - 1; ++l)
            for (int tb = 1; tb < kN; ++tb)
                for (int j = 0; j < batch.library_count[tb][l]; ++j)
                    for (int component = 0; component < 3; ++component) {
                        const size_t at = coordinate_index(
                            batch.library_coordinate_offset[tb], j,
                            component, l);
                        const float expected = float(
                            workspace.posed_library_xyz[at]);
                        if (std::memcmp(&workspace.posed_library_xyz_fp32[at],
                                        &expected, sizeof(float)) != 0) {
                            std::printf("FAIL fused FP32 cache lane=%d type=%d item=%d component=%d\n",
                                l, tb, j, component);
                            ok = false;
                        }
                    }
        if (workspace.posed_library_xyz_fp32[b_extreme_packed] != 0.0f) {
            std::puts("FAIL unsafe out-of-range double was converted instead of sanitized");
            ok = false;
        }
        for (int tb = 1; tb < kN; ++tb)
            for (int j = 0; j < batch.library_max[tb]; ++j)
                for (int l = 0; l < COLOR_CONFORMER_LANES; ++l)
                    if (j >= batch.library_count[tb][l] || l >= kPairs) {
                        for (int component = 0; component < 3; ++component) {
                            const size_t at = coordinate_index(
                                batch.library_coordinate_offset[tb], j,
                                component, l);
                            if (!std::isnan(
                                    workspace.posed_library_xyz_fp32[at])) {
                                std::puts("FAIL fused FP32 cache overwrote an inactive/tail slot");
                                ok = false;
                            }
                        }
                    }
    }

    // Unsupported broad matrix/ISA cases keep the FP32 workspace untouched;
    // no resized cache or second conversion traversal is needed by scalar.
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES>
        unsupported_coefficients{};
    std::copy(coefficients, coefficients + unsupported_coefficients.size(),
              unsupported_coefficients.begin());
    unsupported_coefficients[size_t(1) * MAX_COLOR_FEATURES + 1] = 1.0e9;
    const ColorInteractionMatrixView unsupported_matrix{
        kN, MAX_COLOR_FEATURES, false, unsupported_coefficients.data(),
        decays};
    ColorConformerWorkspace scalar_workspace;
    scalar_workspace.posed_library_xyz_fp32.assign(1, cache_poison);
    ColorEvaluation scalar_output[COLOR_CONFORMER_LANES]{};
    uint64_t scalar_terms[COLOR_CONFORMER_LANES]{};
    evaluate_color_conformer_batch8(batch, unsupported_matrix, quaternions,
        translations, true, scalar_output, scalar_terms, nullptr, nullptr,
        &scalar_workspace);
    if (scalar_workspace.posed_library_xyz_fp32.size() != 1 ||
        !std::isnan(scalar_workspace.posed_library_xyz_fp32[0])) {
        std::puts("FAIL scalar-only matrix resized or wrote the FP32 workspace");
        ok = false;
    }

    // Reuse the same populated cache with a shorter active block. Old values
    // in now-inactive lanes must be masked from the vector kernel.
    ColorConformerBatch8 short_batch = batch;
    short_batch.n_valid = 3;
    ColorEvaluation short_output[COLOR_CONFORMER_LANES]{};
    uint64_t short_terms[COLOR_CONFORMER_LANES]{};
    ColorConformerCensus short_census;
    evaluate_color_conformer_batch8(short_batch, matrix, quaternions,
        translations, true, short_output, short_terms, &short_census, nullptr,
        &workspace);
    if (short_census.last_vector_support_mask != (vector_available ? 0x07 : 0x00) ||
        short_output[3].value != 0.0 || short_terms[3] != 0) {
        std::puts("FAIL reused FP32 cache leaked into inactive lanes");
        ok = false;
    }

    // Non-finite raw inputs take exact fallback and are not narrowed into the
    // FP32 cache. Lane 6 has an identity pose so infinity sign is preserved.
    auto check_nonfinite_target = [&](float raw, int invalid_lane,
                                      bool expect_nan) {
        ColorConformerBatch8 edge_batch = batch;
        const size_t at = coordinate_index(
            edge_batch.library_coordinate_offset[3], 0, 0, invalid_lane);
        edge_batch.library_xyz[at] = raw;
        ColorConformerWorkspace edge_workspace;
        edge_workspace.posed_library_xyz_fp32.assign(
            edge_batch.library_xyz.size(), cache_poison);
        ColorEvaluation edge_output[COLOR_CONFORMER_LANES]{};
        uint64_t edge_terms[COLOR_CONFORMER_LANES]{};
        ColorConformerCensus edge_census;
        evaluate_color_conformer_batch8(edge_batch, matrix, quaternions,
            translations, true, edge_output, edge_terms, &edge_census,
            nullptr, &edge_workspace);
        const double posed_value = edge_workspace.posed_library_xyz[at];
        const bool classified = expect_nan ? std::isnan(posed_value)
            : std::isinf(posed_value) &&
                (std::signbit(posed_value) == std::signbit(double(raw)));
        if (!classified ||
            (edge_census.last_vector_support_mask &
                uint8_t(1u << unsigned(invalid_lane))) != 0 ||
            (vector_available &&
             edge_workspace.posed_library_xyz_fp32[at] != 0.0f)) {
            std::printf("FAIL non-finite pose/cache guard lane=%d raw=%g\n",
                        invalid_lane, double(raw));
            ok = false;
        }
    };
    check_nonfinite_target(std::numeric_limits<float>::quiet_NaN(), 1, true);
    check_nonfinite_target(std::numeric_limits<float>::infinity(), lane, false);
    check_nonfinite_target(-std::numeric_limits<float>::infinity(), lane, false);

    // The independent preposed contract remains unchanged: preparation makes
    // its own FP32 cache, and evaluation consumes the same double/FP32 pair.
    ColorConformerWorkspace preposed_workspace;
    prepare_color_conformer_batch8_preposed(batch, quaternions, translations,
                                             &preposed_workspace);
    ColorEvaluation preposed_output[COLOR_CONFORMER_LANES]{};
    uint64_t preposed_terms[COLOR_CONFORMER_LANES]{};
    evaluate_color_conformer_batch8(batch, matrix, quaternions, translations,
        true, preposed_output, preposed_terms, nullptr, nullptr,
        &preposed_workspace, nullptr, nullptr, true);
    for (int l = 0; l < kPairs; ++l) {
        if (preposed_terms[l] != term_counts[l] ||
            preposed_output[l].value != output[l].value) {
            std::printf("FAIL preposed evaluation changed lane=%d\n", l);
            ok = false;
        }
        for (int k = 1; k < 7; ++k)
            if (preposed_output[l].descent[k] != output[l].descent[k]) {
                std::printf("FAIL preposed gradient changed lane=%d component=%d\n",
                            l, k);
                ok = false;
            }
    }

    // Exercise the same eligibility screen under the mandatory prepared
    // reduced pose. Keep the physical query/library inputs and matrix as the
    // only owned source; the query adapter stores A'=sA and the plan supplies
    // d'=d/s^2. The range-invalid lane must remain the only exact fallback
    // whether its query, raw library coordinate, or both trigger the check.
    uint64_t prepared_fallbacks = 0;
    auto prepared_support_mask = [&]() {
        std::array<OwnedConformer, kPairs> reduced_query;
        std::array<ReducedColorConformer, kPairs> reduced_views;
        std::array<ReducedColorConformerPairInput, kPairs> reduced_inputs{};
        for (int l = 0; l < kPairs; ++l) {
            reduced_query[l].view = query[l].view;
            reduced_query[l].xyz = query[l].xyz;
            for (int t = 1; t < kN; ++t)
                for (int i = 0; i < query[l].view.seg_count[t]; ++i) {
                    const size_t at = size_t(query[l].view.seg_offset[t] + i) *
                                      size_t(kStride);
                    for (int k = 0; k < 3; ++k)
                        reduced_query[l].xyz[at + size_t(k)] =
                            float(double(query[l].xyz[at + size_t(k)]) *
                                  double(COORD_SCALE));
                }
            reduced_query[l].view.color_xyz = reduced_query[l].xyz.data();
            reduced_views[l].features = reduced_query[l].view;
            reduced_inputs[l] = {&reduced_views[l], &library[l].view,
                                 kStride, kStride};
        }
        const ColorConformerBatch8 reduced_batch =
            pack_color_conformer_batch8_prepared(
                reduced_inputs.data(), kPairs, kN);
        ColorInvariantPlan plan(matrix);
        PreparedReducedPose prepared_poses[COLOR_CONFORMER_LANES]{};
        const PreparedReducedPose* pose_ptrs[COLOR_CONFORMER_LANES]{};
        for (int l = 0; l < kPairs; ++l) {
            float reduced_translation[3];
            for (int k = 0; k < 3; ++k)
                reduced_translation[k] = translations[l][k] * COORD_SCALE;
            prepared_poses[l] = prepare_reduced_pose_fp32(
                quaternions[l], reduced_translation);
            pose_ptrs[l] = &prepared_poses[l];
        }
        ColorEvaluation reduced_output[COLOR_CONFORMER_LANES];
        uint64_t reduced_terms[COLOR_CONFORMER_LANES]{};
        ColorConformerCensus reduced_census;
        evaluate_color_conformer_batch8_prepared(
            reduced_batch, plan, pose_ptrs, true, reduced_output,
            reduced_terms, &reduced_census, nullptr, nullptr);
        prepared_fallbacks = reduced_census.scalar_fallback_pairs;
        return reduced_census.last_vector_support_mask;
    };
    auto expect_prepared_support = [&](uint8_t expected_mask,
                                       const char* scenario) {
        const uint8_t actual_mask = prepared_support_mask();
        int eligible = 0;
        for (int l = 0; l < kPairs; ++l)
            eligible += (expected_mask & uint8_t(1u << unsigned(l))) != 0;
        const uint64_t expected_exact = vector_available
            ? uint64_t(kPairs - eligible) : uint64_t(kPairs);
        if (actual_mask != expected_mask ||
            prepared_fallbacks != expected_exact) {
            std::printf("FAIL reduced support mask %s mask=0x%02x expected=0x%02x fallback=%llu expected_fallback=%llu\n",
                scenario, unsigned(actual_mask), unsigned(expected_mask),
                (unsigned long long)prepared_fallbacks,
                (unsigned long long)expected_exact);
            ok = false;
        }
    };
    expect_prepared_support(expected_support_mask, "both invalid");
    library[lane].xyz[b_extreme] = 0.5f;
    expect_prepared_support(expected_support_mask, "query-only invalid");
    query[lane].xyz[q_extreme] = 0.5f;
    expect_prepared_support(vector_available ? 0x7f : 0x00, "valid");
    query[lane].xyz[q_extreme] = 1.2e9f;
    library[lane].xyz[b_extreme] = 1.2e9f;
    query[lane].xyz[q_extreme] = 0.5f;
    expect_prepared_support(expected_support_mask, "library-only invalid");
    query[lane].xyz[q_extreme] = 1.2e9f;
    library[lane].xyz[b_extreme] = 1.2e9f;

    for (int ta = 1; ta < kN; ++ta) {
        for (int tb = 1; tb < kN; ++tb) {
            if (decays[size_t(ta) * MAX_COLOR_FEATURES + tb] == 0.0)
                continue;
            uint64_t useful = 0;
            int max_query = 0, max_library = 0;
            for (int l = 0; l < kPairs; ++l) {
                useful += uint64_t(query_counts[l][ta]) *
                          uint64_t(library_counts[l][tb]);
                if (!vector_available || l == lane) continue;
                max_query = std::max(max_query, query_counts[l][ta]);
                max_library = std::max(max_library, library_counts[l][tb]);
            }
            const size_t cell = size_t(ta) * MAX_COLOR_FEATURES + size_t(tb);
            if (census.useful_terms_by_interaction[cell] != useful ||
                census.issued_lanes_by_interaction[cell] !=
                    uint64_t(max_query) * uint64_t(max_library) * 8) {
                std::printf("FAIL census cell=(%d,%d) useful=%llu/%llu issued=%llu/%llu\n",
                    ta, tb,
                    (unsigned long long)census.useful_terms_by_interaction[cell],
                    (unsigned long long)useful,
                    (unsigned long long)census.issued_lanes_by_interaction[cell],
                    (unsigned long long)(uint64_t(max_query) *
                                         uint64_t(max_library) * 8));
                ok = false;
            }
        }
    }
    if (census.vector_evaluations != uint64_t(vector_available)) {
        std::printf("FAIL vector evaluation count=%llu\n",
                    (unsigned long long)census.vector_evaluations);
        ok = false;
    }
    const bool phases_ok = pose_seconds > 0.0 && screen_seconds > 0.0 &&
        fallback_seconds > 0.0 &&
        pack_seconds == 0.0 &&
        (vector_available ? vector_seconds > 0.0 : vector_seconds == 0.0);
    if (!phases_ok) {
        std::printf("FAIL phase timing pose=%g screen=%g fallback=%g pack=%g vector=%g\n",
            pose_seconds, screen_seconds, fallback_seconds, pack_seconds,
            vector_seconds);
        ok = false;
    }
    ColorEvaluation value_only[COLOR_CONFORMER_LANES];
    uint64_t value_terms[COLOR_CONFORMER_LANES]{};
    alignas(32) float translations_next[COLOR_CONFORMER_LANES][3]{};
    for (int l = 0; l < COLOR_CONFORMER_LANES; ++l)
        for (int k = 0; k < 3; ++k)
            translations_next[l][k] = translations[l][k];
    translations_next[1][0] += 0.2f;
    evaluate_color_conformer_batch8(batch, matrix, quaternions,
        translations_next, false, value_only, value_terms, nullptr, nullptr,
        &workspace);
    for (int l = 0; l < kPairs; ++l) {
        uint64_t expected_terms = 0;
        const ColorEvaluation expected = exact(query[l], library[l], matrix,
            l, quaternions[l], translations_next[l], &expected_terms);
        if (!close(value_only[l].value, expected.value) ||
            value_terms[l] != expected_terms) {
            std::printf("FAIL value-only lane=%d value=%.9g expected=%.9g terms=%llu expected_terms=%llu\n",
                l, value_only[l].value, expected.value,
                (unsigned long long)value_terms[l],
                (unsigned long long)expected_terms);
            ok = false;
        }
        for (int k = 1; k < 7; ++k)
            if (value_only[l].descent[k] != 0.0) {
                std::printf("FAIL value-only lane=%d has nonzero gradient[%d]\n", l, k);
                ok = false;
            }
    }
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES>
        canonical_coefficients{}, canonical_decays{};
    for (int t = 1; t < kN; ++t) {
        canonical_coefficients[size_t(t) * MAX_COLOR_FEATURES + t] =
            0.35 + 0.2 * t;
        canonical_decays[size_t(t) * MAX_COLOR_FEATURES + t] =
            -0.25 * t;
    }
    const ColorInteractionMatrixView canonical{
        kN, MAX_COLOR_FEATURES, true, canonical_coefficients.data(),
        canonical_decays.data()};
    ColorEvaluation canonical_output[COLOR_CONFORMER_LANES];
    uint64_t canonical_terms[COLOR_CONFORMER_LANES]{};
    evaluate_color_conformer_batch8(batch, canonical, quaternions,
        translations, true, canonical_output, canonical_terms, nullptr,
        nullptr, &workspace);
    for (int l = 0; l < kPairs; ++l) {
        uint64_t expected_terms = 0;
        const ColorEvaluation expected = exact(query[l], library[l],
            canonical, l, quaternions[l], translations[l], &expected_terms);
        if (!close(canonical_output[l].value, expected.value) ||
            canonical_terms[l] != expected_terms) {
            std::printf("FAIL canonical lane=%d value=%.9g expected=%.9g terms=%llu expected_terms=%llu\n",
                l, canonical_output[l].value, expected.value,
                (unsigned long long)canonical_terms[l],
                (unsigned long long)expected_terms);
            ok = false;
        }
        for (int k = 1; k < 7; ++k)
            if (!close(canonical_output[l].descent[k], expected.descent[k])) {
                std::printf("FAIL canonical lane=%d descent[%d]=%.9g expected=%.9g\n",
                    l, k, canonical_output[l].descent[k], expected.descent[k]);
                ok = false;
            }
    }

    // A raw canonical matrix takes the analytic-only specialization even
    // when the optional invariant-plan consumer is disabled. Active
    // diagonal range failures still take exact fallback. A separate
    // finite, componentwise-in-range cross-only pose outlier is ignored by
    // the specialization because its coefficient is zero; the old general
    // support scan conservatively fell back for it. Preserve established
    // term counts: SIMD counts diagonal cells, while exact fallback counts
    // the width-active canonical upper triangle.
    std::array<double, MAX_COLOR_FEATURES * MAX_COLOR_FEATURES>
        identity_coefficients{}, identity_decays{};
    constexpr double kCanonicalCoefficient = 1.9687012432153024;
    for (int ta = 1; ta < kN; ++ta)
        for (int tb = 1; tb < kN; ++tb) {
            identity_decays[size_t(ta) * MAX_COLOR_FEATURES +
                            size_t(tb)] = -0.5;
            if (ta == tb)
                identity_coefficients[size_t(ta) * MAX_COLOR_FEATURES +
                                      size_t(tb)] = kCanonicalCoefficient;
        }
    const ColorInvariantPlan canonical_plan({
        kN, MAX_COLOR_FEATURES, true, identity_coefficients.data(),
        identity_decays.data()});
    if (!canonical_plan.canonical_analytic_supported() ||
        canonical_plan.canonical_planes().size() != size_t(kN - 1)) {
        std::puts("FAIL identity matrix did not prepare canonical planes");
        ok = false;
    }

    constexpr int canonical_pose_fallback_lane = 1;
    constexpr int canonical_raw_fallback_lane = 2;
    constexpr int canonical_cross_only_lane = 4;
    const size_t transformed_outlier = size_t(
        library[canonical_pose_fallback_lane].view.seg_offset[1]) *
        size_t(kStride);
    for (int component = 0; component < 3; ++component)
        library[canonical_pose_fallback_lane].xyz[
            transformed_outlier + size_t(component)] = 9.9e8f;
    const size_t raw_library_outlier = size_t(
        library[canonical_raw_fallback_lane].view.seg_offset[1]) *
        size_t(kStride);
    library[canonical_raw_fallback_lane].xyz[raw_library_outlier] = 1.2e9f;
    const size_t raw_outlier = size_t(
        query[canonical_raw_fallback_lane].view.seg_offset[1]) *
        size_t(kStride);
    query[canonical_raw_fallback_lane].xyz[raw_outlier] = 1.2e9f;
    const size_t cross_query_outlier = size_t(
        query[canonical_cross_only_lane].view.seg_offset[2]) *
        size_t(kStride);
    const size_t cross_library_outlier = size_t(
        library[canonical_cross_only_lane].view.seg_offset[3]) *
        size_t(kStride);
    for (int component = 0; component < 3; ++component) {
        query[canonical_cross_only_lane].xyz[
            cross_query_outlier + size_t(component)] = 9.9e8f;
        library[canonical_cross_only_lane].xyz[
            cross_library_outlier + size_t(component)] = 9.9e8f;
    }
    std::array<OwnedConformer, kPairs> reduced_query;
    std::array<ReducedColorConformer, kPairs> reduced_query_views{};
    std::array<ReducedColorConformerPairInput, kPairs> reduced_inputs{};
    for (int l = 0; l < kPairs; ++l) {
        reduced_query[l].view = query[l].view;
        reduced_query[l].xyz = query[l].xyz;
        for (int type = 1; type < kN; ++type)
            for (int i = 0; i < query[l].view.seg_count[type]; ++i) {
                const size_t at = size_t(query[l].view.seg_offset[type] + i) *
                                  size_t(kStride);
                for (int component = 0; component < 3; ++component)
                    reduced_query[l].xyz[at + size_t(component)] =
                        float(double(query[l].xyz[at + size_t(component)]) *
                              double(COORD_SCALE));
            }
        reduced_query[l].view.color_xyz = reduced_query[l].xyz.data();
        reduced_query_views[l].features = reduced_query[l].view;
        reduced_inputs[l] = {&reduced_query_views[l], &library[l].view,
                             kStride, kStride};
    }
    const ColorConformerBatch8 canonical_batch =
        pack_color_conformer_batch8_prepared(
            reduced_inputs.data(), kPairs, kN);
    if (!canonical_batch.canonical_library_coordinates_safe[1]
            [canonical_pose_fallback_lane] ||
        canonical_batch.canonical_library_coordinates_safe[1]
            [canonical_raw_fallback_lane]) {
        std::puts("FAIL packed canonical library raw-coordinate eligibility is incorrect");
        ok = false;
    }
#if defined(__x86_64__)
    if (!canonical_batch.canonical_query_coordinates_safe[1][0] ||
        canonical_batch.canonical_query_coordinates_safe[1]
            [canonical_raw_fallback_lane]) {
        std::puts("FAIL packed canonical query eligibility was not cached");
        ok = false;
    }
#elif defined(__aarch64__)
    if (!canonical_batch.query_plane_supported[1][0] ||
        canonical_batch.query_plane_supported[1][canonical_raw_fallback_lane]) {
        std::puts("FAIL packed NEON canonical query-plane cache is incorrect");
        ok = false;
    }
    const size_t reduced_outlier = size_t(
        reduced_query[canonical_raw_fallback_lane].view.seg_offset[1]) *
        size_t(kStride);
    const float saved_outlier = reduced_query[canonical_raw_fallback_lane].xyz[
        reduced_outlier];
    reduced_query[canonical_raw_fallback_lane].xyz[reduced_outlier] = 0.0f;
    const ColorConformerBatch8 repacked_safe =
        pack_color_conformer_batch8_prepared(
            reduced_inputs.data(), kPairs, kN);
    if (!repacked_safe.query_plane_supported[1][canonical_raw_fallback_lane]) {
        std::puts("FAIL repacked NEON canonical query-plane cache stayed unsafe");
        ok = false;
    }
    reduced_query[canonical_raw_fallback_lane].xyz[reduced_outlier] =
        saved_outlier;
    const ColorConformerBatch8 repacked_unsafe =
        pack_color_conformer_batch8_prepared(
            reduced_inputs.data(), kPairs, kN);
    if (repacked_unsafe.query_plane_supported[1][canonical_raw_fallback_lane]) {
        std::puts("FAIL repacked NEON canonical query-plane cache stayed safe");
        ok = false;
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--query-plane-cache-only") == 0) {
        if (!ok) return 1;
        std::puts("PASS NEON query-plane cache active/inactive/repack");
        return 0;
    }
#else
    (void)argc;
    (void)argv;
#endif
    PreparedReducedPose canonical_poses[COLOR_CONFORMER_LANES]{};
    const PreparedReducedPose* canonical_pose_ptrs[
        COLOR_CONFORMER_LANES]{};
    for (int l = 0; l < kPairs; ++l) {
        float reduced_translation[3]{};
        for (int component = 0; component < 3; ++component)
            reduced_translation[component] =
                translations[l][component] * COORD_SCALE;
        canonical_poses[l] = prepare_reduced_pose_fp32(
            quaternions[l], reduced_translation);
        canonical_pose_ptrs[l] = &canonical_poses[l];
    }
    ColorEvaluation dedicated_canonical_output[COLOR_CONFORMER_LANES]{};
    uint64_t dedicated_canonical_terms[COLOR_CONFORMER_LANES]{};
    ColorConformerCensus canonical_census;
    ColorFieldBatchStats canonical_stats;
    ColorConformerWorkspace canonical_workspace;
    const uint8_t canonical_vector_mask = uint8_t(0x7f) &
        uint8_t(~uint8_t((1u << canonical_pose_fallback_lane) |
                         (1u << canonical_raw_fallback_lane)));
    evaluate_color_conformer_batch8_prepared(
        canonical_batch, canonical_plan, canonical_pose_ptrs, true,
        dedicated_canonical_output, dedicated_canonical_terms, &canonical_census, nullptr,
        &canonical_workspace, false, nullptr, &canonical_stats, false,
        OptimizerExpMode::Baseline, true);
    if (canonical_census.last_vector_support_mask !=
            (vector_available ? canonical_vector_mask : 0x00) ||
        canonical_census.vector_evaluations != uint64_t(vector_available) ||
        canonical_census.scalar_fallback_pairs !=
            uint64_t(vector_available ? 2 : kPairs)) {
        std::printf("FAIL canonical lane dispatch mask=0x%02x vector=%llu fallback=%llu\n",
            unsigned(canonical_census.last_vector_support_mask),
            (unsigned long long)canonical_census.vector_evaluations,
            (unsigned long long)canonical_census.scalar_fallback_pairs);
        ok = false;
    }
    uint64_t expected_analytic_samples = 0;
    uint64_t expected_analytic_terms = 0;
    if (vector_available) {
        for (int lane = 0; lane < kPairs; ++lane) {
            if ((canonical_vector_mask & uint8_t(1u << unsigned(lane))) == 0)
                continue;
            for (int type = 1; type < kN; ++type) {
                const uint64_t nq = uint64_t(query_counts[lane][type]);
                const uint64_t nb = uint64_t(library_counts[lane][type]);
                if (nq == 0 || nb == 0) continue;
                expected_analytic_samples += nb;
                expected_analytic_terms += nq * nb;
            }
        }
    }
    if (canonical_stats.samples != expected_analytic_samples ||
        canonical_stats.analytic_selected_samples != expected_analytic_samples ||
        canonical_stats.analytic_selected_terms != expected_analytic_terms) {
        std::printf("FAIL canonical analytic stats samples=%llu/%llu selected=%llu/%llu terms=%llu/%llu\n",
            (unsigned long long)canonical_stats.samples,
            (unsigned long long)expected_analytic_samples,
            (unsigned long long)canonical_stats.analytic_selected_samples,
            (unsigned long long)expected_analytic_samples,
            (unsigned long long)canonical_stats.analytic_selected_terms,
            (unsigned long long)expected_analytic_terms);
        ok = false;
    }
    for (int l = 0; l < kPairs; ++l) {
        uint64_t exact_terms = 0;
        const ColorEvaluation expected = exact_reduced_canonical(
            reduced_query[l], library[l], canonical_plan,
            canonical_poses[l], &exact_terms);
        if (!close(dedicated_canonical_output[l].value, expected.value)) {
            std::printf("FAIL T214 lane=%d value=%.9g expected=%.9g\n",
                l, dedicated_canonical_output[l].value, expected.value);
            ok = false;
        }
        for (int component = 1; component < 7; ++component)
            if (!close(dedicated_canonical_output[l].descent[component],
                       expected.descent[component])) {
                std::printf("FAIL T214 lane=%d descent[%d]=%.9g expected=%.9g\n",
                    l, component, dedicated_canonical_output[l].descent[component],
                    expected.descent[component]);
                ok = false;
            }
        uint64_t expected_terms = exact_terms;
        if (vector_available && l != canonical_pose_fallback_lane &&
            l != canonical_raw_fallback_lane) {
            expected_terms = 0;
            for (int type = 1; type < kN; ++type)
                expected_terms += uint64_t(query_counts[l][type]) *
                                  uint64_t(library_counts[l][type]);
        }
        if (dedicated_canonical_terms[l] != expected_terms) {
            std::printf("FAIL T214 lane=%d terms=%llu expected=%llu\n",
                l, (unsigned long long)dedicated_canonical_terms[l],
                (unsigned long long)expected_terms);
            ok = false;
        }
    }

    ColorEvaluation canonical_plan_output[COLOR_CONFORMER_LANES]{};
    uint64_t canonical_plan_terms[COLOR_CONFORMER_LANES]{};
    ColorConformerWorkspace canonical_plan_workspace;
    evaluate_color_conformer_batch8_prepared(
        canonical_batch, canonical_plan, canonical_pose_ptrs, true,
        canonical_plan_output, canonical_plan_terms, nullptr, nullptr,
        &canonical_plan_workspace, true, nullptr, nullptr, false,
        OptimizerExpMode::Baseline, true);
    for (int l = 0; l < kPairs; ++l)
        if (canonical_plan_output[l].value != dedicated_canonical_output[l].value ||
            canonical_plan_terms[l] != dedicated_canonical_terms[l]) {
            std::printf("FAIL T214 invariant-plan toggle changed lane=%d\n", l);
            ok = false;
        }

    // A non-null empty field array disables only the new canonical helper;
    // the existing general analytic AVX2 evaluator remains available as an
    // independent regression for output and width-based term counts.
    const ColorFieldBundle* analytic_only_fields[
        COLOR_CONFORMER_LANES]{};
    ColorEvaluation general_output[COLOR_CONFORMER_LANES]{};
    uint64_t general_terms[COLOR_CONFORMER_LANES]{};
    ColorConformerWorkspace general_workspace;
    evaluate_color_conformer_batch8_prepared(
        canonical_batch, canonical_plan, canonical_pose_ptrs, true,
        general_output, general_terms, nullptr, nullptr,
        &general_workspace, false, analytic_only_fields, nullptr, false,
        OptimizerExpMode::Baseline, true);
    for (int l = 0; l < kPairs; ++l) {
        const bool old_cross_only_fallback = vector_available &&
            (l == canonical_cross_only_lane || l == lane);
        if ((!old_cross_only_fallback &&
             general_terms[l] != dedicated_canonical_terms[l]) ||
            !close(general_output[l].value, dedicated_canonical_output[l].value)) {
            std::printf("FAIL T214 vs existing analytic lane=%d value=%.9g/%.9g terms=%llu/%llu\n",
                l, dedicated_canonical_output[l].value, general_output[l].value,
                (unsigned long long)dedicated_canonical_terms[l],
                (unsigned long long)general_terms[l]);
            ok = false;
        }
        for (int component = 1; component < 7; ++component)
            if (!close(general_output[l].descent[component],
                       dedicated_canonical_output[l].descent[component])) {
                std::printf("FAIL T214 vs existing analytic lane=%d descent[%d]=%.9g/%.9g\n",
                    l, component, dedicated_canonical_output[l].descent[component],
                    general_output[l].descent[component]);
                ok = false;
            }
    }
    if (!ok) return 1;
    std::puts("PASS conformer kernel fixed-pose oracle, tails, masks, fallback");
    return 0;
}

#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "color_invariant_plan.hpp"
#include "color_canonical_four_start.hpp"
#include "color_conformer_kernel.hpp"
#include "kernel.hpp"
#include "reduced_pose.hpp"

namespace {

using r3::cpu::ColorActiveInteraction;
using r3::cpu::ColorConformer;
using r3::cpu::ColorDecayFrame;
using r3::cpu::ColorInteractionMatrixView;
using r3::cpu::ColorInvariantMatrixView;
using r3::cpu::ColorInvariantPlan;
using r3::cpu::ReducedColorConformer;
using r3::cpu::ReducedColorConformerPairInput;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(double a, double b) {
    return std::abs(a - b) <= 2e-15 * std::max(1.0, std::abs(b));
}

ColorInteractionMatrixView make_view(int n, int stride,
        const std::vector<double>& coefficients,
        const std::vector<double>& decays, bool canonical = false) {
    return {n, stride, canonical, coefficients.data(), decays.data()};
}

void test_owned_snapshot_and_frames() {
    constexpr int n = 4;
    constexpr int stride = 6;
    std::vector<double> coefficients(size_t(n) * stride, 0.0);
    std::vector<double> decays(size_t(n) * stride, 0.0);
    auto set = [&](int ta, int tb, double c, double d) {
        const size_t at = size_t(ta) * stride + size_t(tb);
        coefficients[at] = c;
        decays[at] = d;
    };
    set(1, 1, 2.0, -0.5);
    set(3, 1, 4.0, -0.25);
    set(1, 2, 6.0, -0.75);
    set(2, 2, 0.0, -0.125);  // zero coefficient still counts as active
    set(3, 2, 8.0, 0.0);     // zero width stays inactive
    set(1, 3, 10.0, -0.375);
    const ColorInteractionMatrixView input = make_view(
        n, stride, coefficients, decays);
    ColorInvariantPlan plan(input);

    require(plan.n_features() == n, "feature count not snapshotted");
    require(plan.source_stride() == stride, "source stride not snapshotted");
    require(!plan.canonical(), "general matrix became canonical");
    require(plan.matches_input(input), "original input did not match plan");
    require(plan.coefficient(2, 2) == 0.0,
            "coefficient snapshot changed zero coefficient");
    require(plan.physical_decay(2, 2) == -0.125,
            "physical decay was not snapshotted");
    const double scale = double(r3::cpu::COORD_SCALE);
    require(close(plan.reduced_decay(2, 2), -0.125 / (scale * scale)),
            "reduced decay did not use the pinned scale");

    const ColorInvariantMatrixView physical = plan.physical_matrix_view();
    const ColorInvariantMatrixView reduced = plan.reduced_matrix_view();
    require(physical.frame == ColorDecayFrame::Physical &&
            reduced.frame == ColorDecayFrame::Reduced,
            "matrix frame tag is missing");
    require(physical.n_features == n && reduced.n_features == n &&
            physical.stride == n && reduced.stride == n &&
            physical.canonical == reduced.canonical,
            "framed views lost matrix metadata");
    require(plan.matches_view(physical) && plan.matches_view(reduced),
            "plan did not identify its owned framed views");
    ColorInvariantMatrixView wrong_frame = physical;
    wrong_frame.frame = ColorDecayFrame::Reduced;
    require(!plan.matches_view(wrong_frame),
            "frame mismatch passed the view identity check");
    ColorInvariantMatrixView wrong_stride = physical;
    wrong_stride.stride += 1;
    require(!plan.matches_view(wrong_stride),
            "stride mismatch passed the view identity check");
    ColorInvariantMatrixView wrong_n = physical;
    wrong_n.n_features -= 1;
    require(!plan.matches_view(wrong_n),
            "feature-count mismatch passed the view identity check");

    const auto& active = plan.active_interactions();
    require(active.size() == 5, "wrong active interaction count");
    const std::array<std::array<int, 2>, 5> expected{{
        {{1, 1}}, {{3, 1}}, {{1, 2}}, {{2, 2}}, {{1, 3}}
    }};
    for (size_t i = 0; i < active.size(); ++i) {
        require(active[i].query_type == expected[i][0] &&
                active[i].target_type == expected[i][1],
                "active interactions changed established target-major order");
    }
    require(active[3].coefficient == 0.0,
            "zero-coefficient/nonzero-width interaction was dropped");
    require(plan.active_interaction(3, 1) &&
            plan.active_interaction(2, 2) &&
            !plan.active_interaction(3, 2) &&
            !plan.active_interaction(0, 2),
            "prepared active-interaction mask disagrees with widths");
    const auto t1 = plan.target_interaction_range(1);
    const auto t2 = plan.target_interaction_range(2);
    const auto t3 = plan.target_interaction_range(3);
    require(t1[0] == 0 && t1[1] == 2 && t2[0] == 2 && t2[1] == 4 &&
            t3[0] == 4 && t3[1] == 5,
            "per-target interaction ranges are incorrect");

    // The plan owns the caller's matrix values and remains stable after the
    // input storage changes; the matcher then fails closed.
    coefficients[size_t(1) * stride + 1] = 99.0;
    decays[size_t(1) * stride + 1] = 0.0;
    require(!plan.matches_input(make_view(n, stride, coefficients, decays)),
            "mutated caller matrix still matched the plan");
    require(plan.coefficient(1, 1) == 2.0 &&
            plan.physical_decay(1, 1) == -0.5,
            "caller mutation changed the owned plan");
    bool threw = false;
    try {
        plan.require_matching_input(make_view(n, stride, coefficients,
                                             decays));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "checked input matcher did not reject stale input");
}

void test_matrix_support_is_matrix_only() {
    constexpr int n = 3;
    constexpr int stride = 3;
    std::vector<double> coefficients(n * stride, 0.0);
    std::vector<double> decays(n * stride, 0.0);
    coefficients[1 * stride + 2] = 5.0;
    decays[1 * stride + 2] = -0.5;
    ColorInvariantPlan supported(make_view(n, stride, coefficients, decays));
    require(supported.matrix_supported_for_avx2(),
            "valid general matrix was rejected");

    coefficients[1 * stride + 2] = -1.0;
    ColorInvariantPlan unsupported(make_view(n, stride, coefficients, decays));
    require(!unsupported.matrix_supported_for_avx2(),
            "negative coefficient passed matrix support check");
    require(unsupported.active_interactions().size() == 1,
            "unsupported matrix did not retain active interaction metadata");

    coefficients[1 * stride + 2] = 5.0;
    decays[1 * stride + 2] = 0.0;
    ColorInvariantPlan empty(make_view(n, stride, coefficients, decays));
    require(empty.matrix_supported_for_avx2() &&
            empty.active_interactions().empty(),
            "zero-width matrix did not remain supported and inactive");

    coefficients[1 * stride + 1] = 1.0;
    coefficients[2 * stride + 2] = 2.0;
    decays[1 * stride + 1] = -0.5;
    decays[2 * stride + 2] = -0.25;
    ColorInvariantPlan canonical(make_view(
        n, stride, coefficients, decays, true));
    require(canonical.canonical() &&
            canonical.active_interactions().size() == 2 &&
            canonical.active_interactions()[0].query_type == 1 &&
            canonical.active_interactions()[1].query_type == 2 &&
            !canonical.active_interaction(2, 1),
            "canonical matrix activity/order changed");
    require(!canonical.canonical_analytic_supported() &&
            canonical.canonical_planes().empty(),
            "a caller canonical flag promoted a nonidentity matrix");

    constexpr double canonical_coefficient = 1.9687012432153024;
    for (int ta = 1; ta < n; ++ta) {
        for (int tb = 1; tb < n; ++tb) {
            coefficients[size_t(ta) * stride + size_t(tb)] =
                ta == tb ? canonical_coefficient : 0.0;
            decays[size_t(ta) * stride + size_t(tb)] = -0.5;
        }
    }
    ColorInvariantPlan canonical_identity(make_view(
        n, stride, coefficients, decays, true));
    const auto& canonical_planes = canonical_identity.canonical_planes();
    const float reduced_width = float(-2.0 *
        (-0.5 / (double(r3::cpu::COORD_SCALE) *
                 double(r3::cpu::COORD_SCALE))));
    require(canonical_identity.canonical_analytic_supported() &&
            canonical_planes.size() == size_t(n - 1) &&
            canonical_planes[0].type == 1 &&
            canonical_planes[1].type == 2 &&
            canonical_planes[0].coefficient ==
                float(canonical_coefficient) &&
            canonical_planes[0].width == reduced_width &&
            canonical_planes[0].exponent_scale == -0.5f * reduced_width &&
            canonical_planes[0].torque_scale == -2.0f * reduced_width &&
            canonical_planes[0].translation_scale == -reduced_width,
            "canonical diagonal constants were not snapshotted in FP32");

    coefficients[1 * stride + 1] = 0.0;
    coefficients[2 * stride + 2] = 0.0;
    decays[1 * stride + 1] = 0.0;
    decays[2 * stride + 2] = 0.0;
    decays[1 * stride + 2] = -0.5;
    coefficients[1 * stride + 2] =
        std::numeric_limits<double>::quiet_NaN();
    ColorInvariantPlan nonfinite(make_view(n, stride, coefficients, decays));
    require(!nonfinite.matrix_supported_for_avx2(),
            "nonfinite coefficient passed matrix support check");

    coefficients[1 * stride + 2] = 1.0e8;
    decays[1 * stride + 2] = -5.0e5;
    ColorInvariantPlan upper_boundary(make_view(
        n, stride, coefficients, decays));
    require(upper_boundary.matrix_supported_for_avx2(),
            "inclusive SIMD coefficient/width boundary was rejected");
    require(!upper_boundary.reduced_matrix_supported_for_avx2(),
            "reduced-frame width expansion passed the SIMD bound");
    decays[1 * stride + 2] = -3.0e5;
    ColorInvariantPlan reduced_supported(make_view(
        n, stride, coefficients, decays));
    require(reduced_supported.matrix_supported_for_avx2() &&
            reduced_supported.reduced_matrix_supported_for_avx2(),
            "reduced-frame in-bound SIMD width was rejected");
    coefficients[1 * stride + 2] = 1.0e8 + 1.0;
    ColorInvariantPlan coefficient_overflow(make_view(
        n, stride, coefficients, decays));
    require(!coefficient_overflow.matrix_supported_for_avx2(),
            "coefficient above the SIMD bound was accepted");
    require(!coefficient_overflow.reduced_matrix_supported_for_avx2(),
            "coefficient above the reduced-frame SIMD bound was accepted");
    coefficients[1 * stride + 2] = 1.0;
    decays[1 * stride + 2] = -5.0e5 - 1.0;
    ColorInvariantPlan width_overflow(make_view(
        n, stride, coefficients, decays));
    require(!width_overflow.matrix_supported_for_avx2(),
            "width above the SIMD bound was accepted");
    decays[1 * stride + 2] = -std::numeric_limits<double>::denorm_min();
    ColorInvariantPlan fp32_underflow(make_view(
        n, stride, coefficients, decays));
    require(!fp32_underflow.matrix_supported_for_avx2(),
            "width that rounds to zero in fp32 was accepted");
}

void test_prepared_four_start_rejects_unsupported_canonical_matrix() {
    constexpr int n = 3;
    constexpr double excessive_coefficient = 1.0e40;
    std::vector<double> coefficients(n * n, 0.0);
    std::vector<double> decays(n * n, 0.0);
    for (int query_type = 1; query_type < n; ++query_type) {
        for (int target_type = 1; target_type < n; ++target_type) {
            coefficients[size_t(query_type) * n + size_t(target_type)] =
                query_type == target_type ? excessive_coefficient : 0.0;
            decays[size_t(query_type) * n + size_t(target_type)] = -0.5;
        }
    }
    const ColorInvariantPlan plan(make_view(
        n, n, coefficients, decays, true));
    require(plan.canonical_analytic_supported(),
            "fixture must be algebraically canonical");
    require(!plan.reduced_matrix_supported_for_avx2(),
            "fixture coefficient unexpectedly passed the FP32 bound");

    float query_xyz[6] = {0.1f, 0.2f, 0.3f, -0.2f, 0.1f, 0.05f};
    float library_xyz[6] = {-0.1f, 0.15f, 0.2f, 0.25f, -0.05f, 0.1f};
    ReducedColorConformer query{};
    query.features.n_features = n;
    query.features.n_color = 2;
    query.features.color_xyz = query_xyz;
    query.features.seg_offset[1] = 0;
    query.features.seg_count[1] = 1;
    query.features.seg_offset[2] = 1;
    query.features.seg_count[2] = 1;
    ColorConformer library{};
    library.n_features = n;
    library.n_color = 2;
    library.color_xyz = library_xyz;
    library.seg_offset[1] = 0;
    library.seg_count[1] = 1;
    library.seg_offset[2] = 1;
    library.seg_count[2] = 1;
    const ReducedColorConformerPairInput pair{
        &query, &library, 3, 3};
    r3::cpu::ColorCanonicalFourStartDescriptor descriptor{};
    require(!r3::cpu::prepare_color_canonical_four_start_descriptor(
                pair, plan, descriptor),
            "unsupported FP32 matrix entered the prepared four-start route");
    require(descriptor.plane_count == 0,
            "rejected descriptor retained narrowed FP32 plane constants");
}

ColorInvariantPlan make_plan_from_temporary_matrix() {
    constexpr int n = 3;
    std::vector<double> coefficients(n * n, 0.0);
    std::vector<double> decays(n * n, 0.0);
    coefficients[1 * n + 2] = 7.0;
    decays[1 * n + 2] = -0.75;
    return ColorInvariantPlan({n, n, false,
                               coefficients.data(), decays.data()});
}

void test_lifetime_and_concurrent_reuse() {
    ColorInvariantPlan plan = make_plan_from_temporary_matrix();
    require(plan.coefficient(1, 2) == 7.0 &&
            plan.physical_decay(1, 2) == -0.75 &&
            plan.active_interactions().size() == 1,
            "plan depended on the temporary caller matrix lifetime");

    constexpr int n = 4;
    constexpr int stride = 4;
    std::vector<double> coefficients(n * stride, 0.0);
    std::vector<double> decays(n * stride, 0.0);
    coefficients[1 * stride + 3] = 3.0;
    decays[1 * stride + 3] = -0.25;
    const auto input = make_view(n, stride, coefficients, decays);
    auto shared = std::make_shared<const ColorInvariantPlan>(input);
    std::atomic<bool> all_ok{true};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&, shared]() {
            for (int repeat = 0; repeat < 1000; ++repeat) {
                const auto range = shared->target_interaction_range(3);
                if (!shared->matches_input(input) || range[0] != 0 ||
                    range[1] != 1 ||
                    shared->active_interactions()[range[0]].query_type != 1)
                    all_ok.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (auto& worker : workers) worker.join();
    require(all_ok.load(std::memory_order_relaxed),
            "concurrent immutable-plan reads disagreed");
}

void test_shape_and_isa_guards() {
    std::vector<double> coefficients(4, 0.0), decays(4, 0.0);
    bool threw = false;
    try {
        ColorInvariantPlan bad({2, 1, false, coefficients.data(),
                                decays.data()});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "malformed dimensions were accepted");

    ColorInvariantPlan plan({2, 2, true, coefficients.data(), decays.data()});
    const auto& isa = plan.isa_capabilities();
    require(plan.avx2_fma_available() ==
                (isa.avx2_fma_compiled && isa.avx2_fma_runtime_supported),
            "ISA availability does not combine compile/runtime checks");
    ColorInvariantMatrixView wrong_canonical = plan.physical_matrix_view();
    wrong_canonical.canonical = false;
    require(!plan.matches_view(wrong_canonical),
            "canonical mismatch passed the view identity check");
}

}  // namespace

void test_complete_compatible_signature() {
    using r3::cpu::ColorCanonicalFourStartDescriptor;
    using r3::cpu::color_canonical_pair_compatible;
    ColorCanonicalFourStartDescriptor a{};
    a.immutable_inputs_safe = true;
    a.n_features = 4;
    a.shape_count = 7;
    a.plane_count = 1;
    a.library_count[1] = 3;
    a.library_count[2] = 2; // Query-empty plane still belongs to the signature.
    a.library_offset[2] = 3;
    a.posed_library_size = 36;
    float query[3]{}, low[3]{}, high[3]{};
    a.planes[0].query_xyz = query;
    a.planes[0].library_xyz = low;
    auto b = a;
    b.planes[0].library_xyz = high;
    require(color_canonical_pair_compatible(a,b), "different library pointers must pair");
    const auto matching = b;
    b.shape_count++;
    require(!color_canonical_pair_compatible(a,b), "shape mismatch paired");
    b = matching; b.library_count[2]++;
    require(!color_canonical_pair_compatible(a,b), "inactive color count mismatch paired");
    b = matching; b.library_offset[2]++;
    require(!color_canonical_pair_compatible(a,b), "color offset mismatch paired");
    b = matching; b.planes[0].query_xyz = high;
    require(!color_canonical_pair_compatible(a,b), "different query coordinates paired");
    b = matching; b.planes[0].query_count++;
    require(!color_canonical_pair_compatible(a,b), "query count mismatch paired");
    b = matching; b.planes[0].posed_offset++;
    require(!color_canonical_pair_compatible(a,b), "posed offset mismatch paired");
    b = matching; b.immutable_inputs_safe = false;
    require(!color_canonical_pair_compatible(a,b), "unsafe immutable input paired");
}

int main() {
    try {
        test_owned_snapshot_and_frames();
        test_matrix_support_is_matrix_only();
        test_prepared_four_start_rejects_unsupported_canonical_matrix();
        test_lifetime_and_concurrent_reuse();
        test_shape_and_isa_guards();
        test_complete_compatible_signature();
        std::cout << "PASS T188 color invariant plan\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL T188 color invariant plan: " << error.what() << "\n";
        return 1;
    }
}

// One owned snapshot of matrix- and ISA-level color invariants.
#pragma once

#include <array>
#include <bitset>
#include <cstddef>
#include <vector>

#include "color_evaluator.hpp"

namespace r3 {
namespace cpu {

enum class ColorDecayFrame {
    Physical,
    Reduced,
};

// This frame tag is carried beside the evaluator's lightweight pointer view.
// A physical and reduced-decay matrix have identical coefficients and shape,
// but are not interchangeable when the pose/query coordinates are scaled.
struct ColorInvariantMatrixView {
    int n_features = 0;
    int stride = 0;
    bool canonical = false;
    const double* coefficients = nullptr;
    const double* decays = nullptr;
    ColorDecayFrame frame = ColorDecayFrame::Physical;

    ColorInteractionMatrixView evaluator_view() const noexcept {
        return {n_features, stride, canonical, coefficients, decays};
    }
};

struct ColorActiveInteraction {
    int query_type = 0;
    int target_type = 0;
    double coefficient = 0.0;
    double physical_decay = 0.0;
    double reduced_decay = 0.0;
};

// Matrix-only constants for the exact canonical identity-interaction
// analytic kernel. The raw R/P predicate is evaluated while the immutable
// plan is prepared; these FP32 values preserve the prepared reduced-frame
// evaluator's conversion order and are reused by every pose evaluation.
struct ColorCanonicalPlane {
    int type = 0;
    float coefficient = 0.0f;
    float width = 0.0f;
    float exponent_scale = 0.0f;
    float torque_scale = 0.0f;
    float translation_scale = 0.0f;
};

struct ColorIsaCapabilities {
    bool avx2_fma_compiled = false;
    bool avx2_fma_runtime_supported = false;

    bool avx2_fma_available() const noexcept {
        return avx2_fma_compiled && avx2_fma_runtime_supported;
    }
};

// Immutable, owned preparation for the matrix-level color evaluator state.
// It snapshots the caller's matrix (so a later caller mutation cannot make a
// cached plan stale), the ordered nonzero-width interactions, reduced-frame
// decays, matrix-only SIMD eligibility and process ISA capability. Pair counts,
// coordinate validity/ranges, masks and fallback decisions deliberately stay
// with each pose evaluation.
class ColorInvariantPlan {
public:
    explicit ColorInvariantPlan(const ColorInteractionMatrixView& input);

    int n_features() const noexcept {
        return n_features_;
    }

    int source_stride() const noexcept {
        return source_stride_;
    }

    bool canonical() const noexcept {
        return canonical_;
    }

    bool canonical_analytic_supported() const noexcept {
        return canonical_analytic_supported_;
    }

    bool default_parameters() const noexcept {
        return default_parameters_;
    }

    const std::vector<ColorCanonicalPlane>& canonical_planes() const noexcept {
        return canonical_planes_;
    }

    const std::vector<double>& coefficients() const noexcept {
        return coefficients_;
    }

    const std::vector<double>& physical_decays() const noexcept {
        return physical_decays_;
    }

    const std::vector<double>& reduced_decays() const noexcept {
        return reduced_decays_;
    }

    double coefficient(int query_type, int target_type) const;
    double physical_decay(int query_type, int target_type) const;
    double reduced_decay(int query_type, int target_type) const;

    ColorInvariantMatrixView physical_matrix_view() const noexcept;
    ColorInvariantMatrixView reduced_matrix_view() const noexcept;

    // Returns true only when the source view still describes the exact matrix
    // snapshot used to construct this plan. `require_matching_input` is the
    // fail-closed form for cached-handle call boundaries.
    bool matches_input(const ColorInteractionMatrixView& input) const noexcept;
    void require_matching_input(const ColorInteractionMatrixView& input) const;

    // Framed views are identity-checked, including the frame tag and the
    // n/canonical/stride metadata. This catches accidentally using reduced
    // decays with physical coordinates (or the reverse).
    bool matches_view(const ColorInvariantMatrixView& view) const noexcept;

    bool matrix_supported_for_avx2() const noexcept {
        return matrix_supported_for_avx2_;
    }

    // The shared-pose evaluator consumes reduced-decay matrix values, whose FP32
    // width support boundary is frame-dependent. Keep the physical-frame
    // legacy screen above and expose its exact reduced-frame counterpart.
    bool reduced_matrix_supported_for_avx2() const noexcept {
        return reduced_matrix_supported_for_avx2_;
    }

    bool active_interaction(int query_type, int target_type) const noexcept;

    const ColorIsaCapabilities& isa_capabilities() const noexcept {
        return isa_capabilities_;
    }

    bool avx2_fma_available() const noexcept {
        return isa_capabilities_.avx2_fma_available();
    }

    const std::vector<ColorActiveInteraction>&
    active_interactions() const noexcept {
        return active_interactions_;
    }

    // Index range in active_interactions() for one target type. The order is
    // the evaluator's established target-major then query-type order.
    std::array<size_t, 2> target_interaction_range(int target_type) const;

private:
    int n_features_ = 0;
    int source_stride_ = 0;
    bool canonical_ = false;
    bool canonical_analytic_supported_ = false;
    bool default_parameters_ = false;
    std::vector<double> coefficients_;
    std::vector<double> physical_decays_;
    std::vector<double> reduced_decays_;
    std::vector<ColorCanonicalPlane> canonical_planes_;
    std::vector<ColorActiveInteraction> active_interactions_;
    std::bitset<MAX_COLOR_FEATURES * MAX_COLOR_FEATURES> active_mask_{};
    std::array<size_t, MAX_COLOR_FEATURES + 1> target_offsets_{};
    ColorIsaCapabilities isa_capabilities_{};
    bool matrix_supported_for_avx2_ = false;
    bool reduced_matrix_supported_for_avx2_ = false;
};

}  // namespace cpu
}  // namespace r3

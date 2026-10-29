// Immutable, owned snapshot of a color interaction matrix and the
// matrix- and ISA-level facts every color evaluator needs, computed once at
// query preparation instead of per pose evaluation.
//
// Matrix conventions (color_scalar.cpp Matrix): for widths w and heights h,
// coefficient = (pi/2)^1.5 h^2 / w^1.5 and physical decay = -w/2, so a term
// is coefficient * exp(decay * d^2) with d in Angstrom. Type 0 is reserved
// for ordinary atoms and never interacts. Evaluators work in the reduced
// frame (coordinates scaled by s = COORD_SCALE), so the plan also stores
// reduced decays = physical decay / s^2.
//
// Pose-dependent facts (pair counts, coordinate ranges, fallback) are
// deliberately not here; they stay with each evaluation.
#include "color_invariant_plan.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernel.hpp"
#include "color_default_parameters.hpp"

namespace r3 {
namespace cpu {
namespace {

// Matrix support envelope of the FP32 SIMD evaluators. An entry outside it
// is not an error: the matrix is marked unsupported and evaluators keep
// their scalar route.
constexpr double kMaximumCoefficient = 1.0e8;
constexpr double kMaximumWidth = 1.0e6;

size_t matrix_index(int n_features, int query_type, int target_type) {
    if (query_type < 0 || query_type >= n_features || target_type < 0 ||
        target_type >= n_features)
        throw std::out_of_range("color matrix type index is out of range");
    return size_t(query_type) * size_t(n_features) + size_t(target_type);
}

bool same_double_bits(double a, double b) noexcept {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// One active entry is SIMD-supported if its coefficient is finite,
// nonnegative and bounded, its width -2*decay is bounded, and both survive
// narrowing to FP32 with a strictly positive FP32 width. Called separately
// with physical and reduced decays: the width bound is frame-dependent.
bool supported_avx2_interaction(double coefficient, double decay) noexcept {
    const float f_coefficient = float(coefficient);
    const float f_width = float(-2.0 * decay);
    return std::isfinite(coefficient) && coefficient >= 0.0 &&
           coefficient <= kMaximumCoefficient &&
           -2.0 * decay <= kMaximumWidth && std::isfinite(f_coefficient) &&
           std::isfinite(f_width) && f_width > 0.0f;
}

// Compiled-in AVX2+FMA AND runtime CPU support; both are needed before an
// x86 evaluator may dispatch to AVX2 code. Always false on ARM.
ColorIsaCapabilities detect_isa_capabilities() noexcept {
    ColorIsaCapabilities result;
#if defined(__AVX2__) && defined(__FMA__)
    result.avx2_fma_compiled = true;
#endif
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    result.avx2_fma_runtime_supported =
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    return result;
}

}  // namespace

ColorInvariantPlan::ColorInvariantPlan(const ColorInteractionMatrixView& input)
    : n_features_(input.n_features), source_stride_(input.stride),
      canonical_(input.canonical),
      isa_capabilities_(detect_isa_capabilities()) {
    if (n_features_ < 1 || n_features_ > MAX_COLOR_FEATURES ||
        source_stride_ < n_features_ || input.coefficients == nullptr ||
        input.decays == nullptr)
        throw std::invalid_argument("invalid color invariant matrix view");

    const size_t matrix_size = size_t(n_features_) * size_t(n_features_);
    coefficients_.resize(matrix_size);
    physical_decays_.resize(matrix_size);
    reduced_decays_.resize(matrix_size);
    const double coordinate_scale = double(COORD_SCALE);
    const double scale_squared = coordinate_scale * coordinate_scale;
    for (int ta = 0; ta < n_features_; ++ta) {
        for (int tb = 0; tb < n_features_; ++tb) {
            const size_t dst = matrix_index(n_features_, ta, tb);
            const size_t src = size_t(ta) * size_t(source_stride_) + size_t(tb);
            coefficients_[dst] = input.coefficients[src];
            physical_decays_[dst] = input.decays[src];
            reduced_decays_[dst] = physical_decays_[dst] / scale_squared;
        }
    }

    // Active interactions in target-major, then query-type order, indexed
    // by target type through target_offsets_. Nonzero width defines
    // activity. A canonical matrix keeps only query types ta <= tb.
    for (int tb = 0; tb <= n_features_; ++tb)
        target_offsets_[size_t(tb)] = active_interactions_.size();
    for (int tb = 1; tb < n_features_; ++tb) {
        target_offsets_[size_t(tb)] = active_interactions_.size();
        const int ta_end = canonical_ ? tb + 1 : n_features_;
        for (int ta = 1; ta < ta_end; ++ta) {
            const size_t at = matrix_index(n_features_, ta, tb);
            const double decay = physical_decays_[at];
            // Preserve the kernel's established term-count semantics: a
            // zero coefficient with nonzero width still counts as an issued
            // interaction, so activity is defined by width alone.
            if (decay == 0.0) continue;
            active_mask_.set(size_t(ta) * size_t(MAX_COLOR_FEATURES) +
                             size_t(tb));
            active_interactions_.push_back(
                {ta, tb, coefficients_[at], decay, reduced_decays_[at]});
        }
    }
    target_offsets_[size_t(n_features_)] = active_interactions_.size();

    // This is the matrix-only portion of the old per-evaluation support
    // screen. Pose-dependent input/range checks stay in the evaluator.
    matrix_supported_for_avx2_ = true;
    reduced_matrix_supported_for_avx2_ = true;
    for (const auto& entry : active_interactions_) {
        if (!supported_avx2_interaction(entry.coefficient,
                                        entry.physical_decay))
            matrix_supported_for_avx2_ = false;
        if (!supported_avx2_interaction(entry.coefficient, entry.reduced_decay))
            reduced_matrix_supported_for_avx2_ = false;
        if (!matrix_supported_for_avx2_ && !reduced_matrix_supported_for_avx2_)
            break;
    }

    // The bool carried by this view is derived from the raw R/P arrays at
    // immutable query preparation. Fail closed if a directly constructed
    // view claims canonicality but its converted snapshot is not the exact
    // identity-interaction matrix. Reserved type 0 is deliberately skipped.
    canonical_analytic_supported_ = canonical_;
    double diagonal_coefficient = 0.0;
    bool have_diagonal = false;
    for (int ta = 1; ta < n_features_ && canonical_analytic_supported_; ++ta) {
        for (int tb = 1; tb < n_features_; ++tb) {
            const size_t at = matrix_index(n_features_, ta, tb);
            // Every width is 1 (decay -w/2 = -0.5); one positive diagonal
            // coefficient shared bitwise by all types; zero off-diagonal.
            if (physical_decays_[at] != -0.5) {
                canonical_analytic_supported_ = false;
                break;
            }
            if (ta == tb) {
                const double coefficient = coefficients_[at];
                if (!std::isfinite(coefficient) || !(coefficient > 0.0)) {
                    canonical_analytic_supported_ = false;
                    break;
                }
                if (!have_diagonal) {
                    diagonal_coefficient = coefficient;
                    have_diagonal = true;
                } else if (!same_double_bits(coefficient,
                                             diagonal_coefficient)) {
                    canonical_analytic_supported_ = false;
                    break;
                }
            } else if (coefficients_[at] != 0.0) {
                canonical_analytic_supported_ = false;
                break;
            }
        }
    }
    // Canonical planes: one per non-reserved type, the diagonal entry's
    // constants narrowed to FP32 in the order the prepared reduced-frame
    // evaluators use: width = w/s^2, exponent_scale = -width/2,
    // torque_scale = -2*width, translation_scale = -width.
    // default_parameters additionally requires the diagonal coefficient to be
    // exactly the default (pi/2)^1.5 (w = h = 1) and the narrowed FP32
    // constants to equal DefaultColorParameters', which gates the
    // default-parameter kernels.
    if (canonical_analytic_supported_) {
        default_parameters_ =
            have_diagonal &&
            same_double_bits(diagonal_coefficient,
                             DefaultColorParameters::physical_coefficient);
        canonical_planes_.reserve(size_t(n_features_ - 1));
        for (int type = 1; type < n_features_; ++type) {
            const size_t at = matrix_index(n_features_, type, type);
            const float width = float(-2.0 * reduced_decays_[at]);
            // A caller can prepare under a different stable rounding mode.
            // Specialize only if the actual prepared FP32 constants match.
            default_parameters_ = default_parameters_ &&
                                  float(coefficients_[at]) ==
                                      DefaultColorParameters::coefficient &&
                                  width == DefaultColorParameters::width;
            canonical_planes_.push_back({type, float(coefficients_[at]), width,
                                         -0.5f * width, -2.0f * width, -width});
        }
    }
}

double ColorInvariantPlan::coefficient(int query_type, int target_type) const {
    return coefficients_[matrix_index(n_features_, query_type, target_type)];
}

double ColorInvariantPlan::physical_decay(int query_type,
                                          int target_type) const {
    return physical_decays_[matrix_index(n_features_, query_type, target_type)];
}

double ColorInvariantPlan::reduced_decay(int query_type,
                                         int target_type) const {
    return reduced_decays_[matrix_index(n_features_, query_type, target_type)];
}

ColorInvariantMatrixView
ColorInvariantPlan::physical_matrix_view() const noexcept {
    return {n_features_,
            n_features_,
            canonical_,
            coefficients_.data(),
            physical_decays_.data(),
            ColorDecayFrame::Physical};
}

ColorInvariantMatrixView
ColorInvariantPlan::reduced_matrix_view() const noexcept {
    return {n_features_,
            n_features_,
            canonical_,
            coefficients_.data(),
            reduced_decays_.data(),
            ColorDecayFrame::Reduced};
}

// Bitwise (not ==) comparison, so -0.0 vs 0.0 and NaN payload changes are
// detected as a different matrix.
bool ColorInvariantPlan::matches_input(
    const ColorInteractionMatrixView& input) const noexcept {
    if (input.n_features != n_features_ || input.stride != source_stride_ ||
        input.canonical != canonical_ || input.coefficients == nullptr ||
        input.decays == nullptr)
        return false;
    for (int ta = 0; ta < n_features_; ++ta) {
        for (int tb = 0; tb < n_features_; ++tb) {
            const size_t at = size_t(ta) * size_t(n_features_) + size_t(tb);
            const size_t src = size_t(ta) * size_t(source_stride_) + size_t(tb);
            if (!same_double_bits(coefficients_[at], input.coefficients[src]) ||
                !same_double_bits(physical_decays_[at], input.decays[src]))
                return false;
        }
    }
    return true;
}

void ColorInvariantPlan::require_matching_input(
    const ColorInteractionMatrixView& input) const {
    if (!matches_input(input))
        throw std::invalid_argument(
            "color invariant plan does not match its input matrix");
}

// Pointer identity: a view matches only if it points into this plan's own
// storage, with the decay array that matches its frame tag.
bool ColorInvariantPlan::matches_view(
    const ColorInvariantMatrixView& view) const noexcept {
    if (view.n_features != n_features_ || view.stride != n_features_ ||
        view.canonical != canonical_ ||
        view.coefficients != coefficients_.data())
        return false;
    if (view.frame == ColorDecayFrame::Physical)
        return view.decays == physical_decays_.data();
    if (view.frame == ColorDecayFrame::Reduced)
        return view.decays == reduced_decays_.data();
    return false;
}

bool ColorInvariantPlan::active_interaction(int query_type,
                                            int target_type) const noexcept {
    if (query_type < 1 || query_type >= n_features_ || target_type < 1 ||
        target_type >= n_features_ || (canonical_ && query_type > target_type))
        return false;
    const size_t bit =
        size_t(query_type) * size_t(MAX_COLOR_FEATURES) + size_t(target_type);
    return active_mask_.test(bit);
}

std::array<size_t, 2>
ColorInvariantPlan::target_interaction_range(int target_type) const {
    if (target_type < 1 || target_type >= n_features_)
        throw std::out_of_range("color target type is out of range");
    return {target_offsets_[size_t(target_type)],
            target_offsets_[size_t(target_type + 1)]};
}

}  // namespace cpu
}  // namespace r3

// Shared CPU color-evaluation contract for analytic and query-built backends.
// Representation choice is made once per query; every evaluator
// returns the same physical overlap and optimizer-coordinate descent.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "color_view.hpp"

namespace r3 {
namespace cpu {

struct ColorEvaluation {
    // Physical color overlap including the interaction-matrix prefactor.
    double value = 0.0;
    // Negative derivative consumed by Adagrad. Entries 1..3 are quaternion
    // half-angle coordinates; 4..6 are reduced-frame translation. Entry 0 is
    // always zero, matching the shape optimizer's seven-slot convention.
    double descent[7]{};
};

struct ColorInteractionMatrixView {
    int n_features = 0;
    int stride = 0;
    bool canonical = false;
    const double* coefficients = nullptr;
    const double* decays = nullptr;

    double coefficient(int query_type, int target_type) const {
        return coefficients[size_t(query_type) * size_t(stride) +
                            size_t(target_type)];
    }

    double decay(int query_type, int target_type) const {
        return decays[size_t(query_type) * size_t(stride) +
                      size_t(target_type)];
    }
};

struct ColorEvaluationRequest {
    const ColorConformer* query = nullptr;
    const ColorConformer* library = nullptr;
    ColorInteractionMatrixView interaction{};
    // Number of floats between consecutive coordinate records. color_xyz
    // already points at the first color record in each conformer.
    int query_stride = 3;
    int library_stride = 3;
    // Effective pose: normalized quaternion and physical translation.
    const float* quaternion = nullptr;
    const float* translation_physical = nullptr;
    bool gradient = true;
};

using ColorEvaluatorFunction =
    ColorEvaluation (*)(void* context, const ColorEvaluationRequest& request,
                        uint64_t* analytic_pair_terms);

struct ColorEvaluator {
    void* context = nullptr;
    ColorEvaluatorFunction evaluate = nullptr;
};

// `analytic_pair_terms` counts real query/library feature pairs for analytic
// evaluators. Interpolated backends leave it unchanged and account for their
// own samples/build work in backend-specific counters.
static inline ColorEvaluation
evaluate_color(const ColorEvaluator& evaluator,
               const ColorEvaluationRequest& request,
               uint64_t* analytic_pair_terms = nullptr) {
    if (evaluator.evaluate == nullptr || request.query == nullptr ||
        request.library == nullptr || request.quaternion == nullptr ||
        request.translation_physical == nullptr ||
        request.interaction.n_features < 1 || request.interaction.stride < 1 ||
        request.interaction.coefficients == nullptr ||
        request.interaction.decays == nullptr)
        throw std::invalid_argument("incomplete color evaluation request");
    return evaluator.evaluate(evaluator.context, request, analytic_pair_terms);
}

}  // namespace cpu
}  // namespace r3

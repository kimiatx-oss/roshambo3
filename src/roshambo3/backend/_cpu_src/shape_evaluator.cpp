// Shared query ownership and validation for fixed-pose evaluators.
//
// The base class every combo shape backend derives from (scalar and SIMD
// analytic in shape_analytic.cpp, the field arms in shape_field_evaluator.cpp).
// It owns an immutable copy of the reduced query, the request validation all
// backends share, and the worker-local `BoundShapeExecution` view. The
// numerical work lives entirely in the subclasses' `evaluate_one`; nothing
// here touches arithmetic. Interface contract: shape_evaluator.hpp.
#include "shape_evaluator.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "score.hpp"

namespace r3 {
namespace cpu {

// Owned reduced-frame query snapshot. `view` aliases `xyz`, so the handle
// stays valid after the caller's arrays are gone and can be shared across
// calls and threads (it is never mutated after construction).
struct PreparedShapeEvaluator::QueryStorage {
    std::vector<float> xyz;
    int n_atoms = 0;
    ShapeQueryMetadata metadata{};
    Conformer view{};
};

PreparedShapeEvaluator::PreparedShapeEvaluator(
    const ShapeEvaluatorConfig& config, const Conformer& query_reduced,
    ShapeQueryMetadata metadata)
    : query_(new QueryStorage), config_(config) {
    if (query_reduced.n_atoms < 0 ||
        (query_reduced.n_atoms > 0 && query_reduced.xyz == nullptr))
        throw std::invalid_argument("invalid reduced shape query");
    query_->n_atoms = query_reduced.n_atoms;
    query_->metadata = metadata;
    const size_t n = size_t(query_reduced.n_atoms);
    if (n > query_->xyz.max_size() / 3)
        throw std::invalid_argument("reduced shape query is too large");
    if (n != 0)
        query_->xyz.assign(query_reduced.xyz, query_reduced.xyz + 3 * n);
    query_->view = {query_->xyz.empty() ? nullptr : query_->xyz.data(),
                    query_->n_atoms};
}

PreparedShapeEvaluator::~PreparedShapeEvaluator() = default;

// Every evaluation enters through here. When a shared-contract pose is
// attached, the request's matrix/translation pointers must BE that pose's
// arrays -- pointer identity, not value equality -- so no adapter can score
// against a separately rebuilt or copied pose
// (doc/CPU_SHARED_POSE_CONTRACT.md).
void validate_shape_pose_request(const ShapePoseRequest& request) {
    if (request.library == nullptr || request.matrix == nullptr ||
        request.translation == nullptr)
        throw std::invalid_argument("shape pose request is incomplete");
    if (request.prepared_pose != nullptr &&
        (request.matrix != request.prepared_pose->matrix ||
         request.translation != request.prepared_pose->translation))
        throw std::invalid_argument(
            "shape request does not consume the shared prepared pose bytes");
}

const Conformer& PreparedShapeEvaluator::query_reduced() const noexcept {
    return query_->view;
}

ShapeQueryMetadata PreparedShapeEvaluator::query_metadata() const noexcept {
    return query_->metadata;
}

// Exact reuse check: same config and bit-identical reduced coordinates
// (memcmp, so -0.0f vs 0.0f or differing NaN payloads also mismatch).
bool PreparedShapeEvaluator::matches(
    const ShapeEvaluatorConfig& config,
    const Conformer& query_reduced) const noexcept {
    if (!shape_evaluator_configs_match(config, config_) ||
        query_reduced.n_atoms != query_->n_atoms ||
        (query_->n_atoms > 0 && query_reduced.xyz == nullptr))
        return false;
    const size_t bytes = size_t(query_->n_atoms) * 3 * sizeof(float);
    return bytes == 0 ||
           std::memcmp(query_->xyz.data(), query_reduced.xyz, bytes) == 0;
}

void PreparedShapeEvaluator::evaluate_batch(const ShapePoseRequest* requests,
                                            size_t count,
                                            ShapeEvaluation* outputs) const {
    if (count == 0) return;
    if (requests == nullptr || outputs == nullptr)
        throw std::invalid_argument("shape evaluation batch is incomplete");
    for (size_t i = 0; i < count; ++i) {
        const ShapePoseRequest& request = requests[i];
        validate_shape_pose_request(request);
        evaluate_one(request, outputs[i]);
    }
}

// Default: no multi-pose preparation. Only the trajectory grain (the
// LaneBlock, shape_analytic.cpp) overrides this; other backends validate the
// targets and return an empty handle, and the one-pose path is used.
PreparedShapeBatchHandle
PreparedShapeEvaluator::prepare_batch(const Conformer* const* libraries,
                                      size_t count) const {
    if (count != 0 && libraries == nullptr)
        throw std::invalid_argument("shape library batch is null");
    for (size_t i = 0; i < count; ++i)
        if (libraries[i] == nullptr)
            throw std::invalid_argument(
                "shape library batch has a null target");
    return {};
}

void PreparedShapeEvaluator::evaluate_batch(const ShapePoseRequest* requests,
                                            size_t count,
                                            const PreparedShapeBatch* batch,
                                            ShapeEvaluation* outputs) const {
    if (batch != nullptr)
        throw std::invalid_argument(
            "shape evaluator does not accept a prepared trajectory batch");
    evaluate_batch(requests, count, outputs);
}

// A workspace belongs to one target: same source pointer and atom count. A
// cheap identity check, not a content comparison -- the target buffer must
// stay immutable while the workspace is in use.
bool ShapeLibraryWorkspace::matches(const Conformer& library) const noexcept {
    return source_atoms == library.n_atoms && source_xyz == library.xyz &&
           (source_xyz != nullptr || source_atoms == 0);
}

// Install the prepared pose's own arrays into the bound request, so the
// aliasing rule in `validate_shape_pose_request` holds by construction.
void BoundShapeExecution::evaluate(const PreparedReducedPose& pose,
                                   ShapeEvaluation& output) const {
    ShapePoseRequest request = request_;
    request.matrix = pose.matrix;
    request.translation = pose.translation;
    request.prepared_pose = &pose;
    evaluate_(evaluator_, request, output);
}

// Check the target/workspace association once per decoded work unit, then
// evaluate many poses through a plain function pointer (bind the outer
// core policy once rather than re-dispatching per call; see
// doc/CPU_SHARED_POSE_CONTRACT.md, "Worker-scoped bound execution").
BoundShapeExecution PreparedShapeEvaluator::make_bound_execution(
    const Conformer& library, const ShapeLibraryWorkspace* workspace,
    BoundShapeExecution::Evaluate evaluate) const {
    if (library.n_atoms < 0 ||
        (library.n_atoms != 0 && library.xyz == nullptr) ||
        (workspace && !workspace->matches(library)))
        throw std::invalid_argument(
            "shape execution requires a matching immutable library workspace");
    BoundShapeExecution bound;
    bound.evaluator_ = this;
    bound.request_.library = &library;
    bound.request_.library_workspace = workspace;
    bound.evaluate_ = evaluate;
    return bound;
}

// Generic binding: route to the virtual `evaluate_one`. Field adapters
// override this to select their concrete numerical arm once and, for the
// pipelined FP32 arm only, honor `values_only` (score-only final winner
// selection); the generic routes ignore it and always produce the
// full gradient.
BoundShapeExecution
PreparedShapeEvaluator::bind_execution(const Conformer& library,
                                       const ShapeLibraryWorkspace* workspace,
                                       bool values_only) const {
    (void)
        values_only;  // Generic numerical routes retain their exact evaluator.
    return make_bound_execution(
        library, workspace,
        [](const void* evaluator, const ShapePoseRequest& request,
           ShapeEvaluation& output) {
            static_cast<const PreparedShapeEvaluator*>(evaluator)->evaluate_one(
                request, output);
        });
}

}  // namespace cpu
}  // namespace r3

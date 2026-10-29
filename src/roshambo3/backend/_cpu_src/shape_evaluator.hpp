// Fixed-pose shape evaluator seam shared by the analytic and field
// adapters. The optimizer owns pose updates and objective weighting; an
// evaluator owns only query-scoped preparation and one or more shape
// overlap/gradient evaluations at supplied poses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "reduced_pose.hpp"

namespace r3 {
namespace cpu {

struct Conformer;
struct SoaConformer;
struct ConformerSoA;
struct LaneBlock;

enum class ShapeBackend : int {
    ScalarAnalytic = 0,
    AnalyticSimd = 1,
    ScalarField = 2,
    PairedChannelField = 3,
    PipelinedField = 4,
};

enum class ShapeGrain : int {
    Scalar = 0,
    QueryAtom = 1,
    LibraryAtom = 2,
    Trajectory = 3,
};

// Storage choice belongs to the immutable query-field preparation. FP32 stays
// the default and all pose, interpolation (unless a separate architecture
// experiment says otherwise), and reduction arithmetic remains FP32.
enum class ShapeFieldStorage : int {
    FP32 = 0,
    FP16 = 1,
};

// Private bounded experiment: keep FP16 endpoints through Y/Z interpolation,
// then widen the two X endpoints for the final FP32 lerp. FP32 remains the
// default and is used whenever this mode is not explicitly requested.
enum class ShapeFieldInterpolation : int {
    FP32 = 0,
    MixedFP16YZ = 1,
};

// `ScalarAnalytic` + `Scalar` + width 1 is the legacy, exact path. Every
// other combination is explicit; adapters must reject combinations they do
// not implement instead of silently selecting a nearby path.
struct ShapeEvaluatorConfig {
    ShapeBackend backend = ShapeBackend::ScalarAnalytic;
    ShapeGrain grain = ShapeGrain::Scalar;
    int width = 1;
    // Field adapters consume spacing and padding in physical Angstrom. Zero
    // means unspecified and is invalid for field backends; the combo layer
    // does not substitute color-field geometry or choose a new policy.
    float field_spacing_angstrom = 0.0f;
    float field_padding_angstrom = 0.0f;
    ShapeFieldStorage storage = ShapeFieldStorage::FP32;
    ShapeFieldInterpolation interpolation = ShapeFieldInterpolation::FP32;
};

inline bool
shape_evaluator_configs_match(const ShapeEvaluatorConfig& a,
                              const ShapeEvaluatorConfig& b) noexcept {
    return a.backend == b.backend && a.grain == b.grain && a.width == b.width &&
           a.field_spacing_angstrom == b.field_spacing_angstrom &&
           a.field_padding_angstrom == b.field_padding_angstrom &&
           a.storage == b.storage && a.interpolation == b.interpolation;
}

struct ShapeQueryMetadata {
    // Stable caller-side row identifier for diagnostics. It is not required
    // for mathematical matching: the reduced coordinate snapshot and config
    // determine whether a prepared handle is reusable.
    int query_index = -1;
};

struct ShapePoseRequest {
    const Conformer* library = nullptr;  // physical Angstrom coordinates
    const struct ShapeLibraryWorkspace* library_workspace = nullptr;
    const float* matrix = nullptr;       // COORD_SCALE * R_eff, 3x3 row-major
    const float* translation = nullptr;  // reduced-frame translation
    float gnorm = 1.0f;
    // Set only by shared-contract calls. The legacy scalar arguments are
    // required to alias this object's matrix/translation arrays exactly.
    const PreparedReducedPose* prepared_pose = nullptr;
};

void validate_shape_pose_request(const ShapePoseRequest& request);

// Own the optional, backend-specific target layouts for an optimizer call.
// The two layouts are intentionally distinct: analytic atom grains use
// `SoaConformer`; field samplers use `ConformerSoA`. Carry source identity
// beside them so adapters can cheaply reject a workspace for another target.
struct ShapeLibraryWorkspace {
    std::shared_ptr<const SoaConformer> analytic_atom;
    std::shared_ptr<const ConformerSoA> field_atom;
    const float* source_xyz = nullptr;
    int source_atoms = 0;
    // Retained numeric payload bytes only (allocator/control metadata excluded).
    size_t prepared_bytes = 0;

    bool matches(const Conformer& library) const noexcept;
};

// Optional immutable data for a stable set of target trajectories. The
// trajectory grain uses one LaneBlock per color batch; the optimizer prepares
// it once, then reuses it for every iteration and start in that batch.
class PreparedShapeBatch {
public:
    virtual ~PreparedShapeBatch() = default;
};

using PreparedShapeBatchHandle = std::shared_ptr<const PreparedShapeBatch>;

struct ShapeEvaluationCounts {
    uint64_t useful_terms = 0;
    uint64_t issued_terms = 0;
    uint64_t field_samples = 0;
    uint64_t fallback_evaluations = 0;
    uint64_t fallback_terms = 0;
};

struct ShapeEvaluation {
    float s_ab = 0.0f;
    // Raw negative overlap derivative, in get_gradient(gnorm=...) coordinates:
    // [0] is zero, [1:4] are space-frame half-angle torque, and [4:7] are
    // reduced-frame translation. Combo Tanimoto weighting is applied once by
    // the shared optimizer after this result returns.
    float descent[7]{};
    ShapeEvaluationCounts counts{};
};

// A worker-local execution view. Construction checks target/workspace
// association once; the query handle, library and workspace must outlive the
// view and remain immutable. Rebind for every decoded work unit. It cannot be
// constructed from a raw callback by a caller.
class BoundShapeExecution {
public:
    void evaluate(const PreparedReducedPose& pose,
                  ShapeEvaluation& output) const;

private:
    friend class PreparedShapeEvaluator;
    BoundShapeExecution() = default;
    using Evaluate = void (*)(const void*, const ShapePoseRequest&,
                              ShapeEvaluation&);
    const void* evaluator_ = nullptr;
    ShapePoseRequest request_{};
    Evaluate evaluate_ = nullptr;
};

struct ShapeEvaluatorStamp {
    std::string backend;
    std::string grain;
    std::string isa;
    std::string reduction;
    std::string exponential;
    int width = 1;
    bool vector_path = false;
    bool requires_library_soa = false;
    std::string storage;
};

// Immutable query snapshot plus an evaluator implementation. The query passed
// at construction is copied in reduced coordinates, so the handle owns all
// data needed by query-prepared fields and can safely be reused across calls.
// `matches` validates that a supplied query and option set are the exact ones
// used to prepare the handle. Implementations may hold immutable grids/packed
// data; evaluate methods must keep scratch local to the call or thread.
class PreparedShapeEvaluator {
public:
    PreparedShapeEvaluator(const ShapeEvaluatorConfig& config,
                           const Conformer& query_reduced,
                           ShapeQueryMetadata metadata = {});
    virtual ~PreparedShapeEvaluator();

    PreparedShapeEvaluator(const PreparedShapeEvaluator&) = delete;
    PreparedShapeEvaluator& operator=(const PreparedShapeEvaluator&) = delete;

    const ShapeEvaluatorConfig& config() const noexcept {
        return config_;
    }

    ShapeQueryMetadata query_metadata() const noexcept;
    const Conformer& query_reduced() const noexcept;
    bool matches(const ShapeEvaluatorConfig& config,
                 const Conformer& query_reduced) const noexcept;

    virtual ShapeEvaluatorStamp stamp() const = 0;
    virtual void evaluate_one(const ShapePoseRequest& request,
                              ShapeEvaluation& output) const = 0;
    virtual BoundShapeExecution
    bind_execution(const Conformer& library,
                   const ShapeLibraryWorkspace* workspace,
                   bool values_only = false) const;

    // A batch always shares this prepared query, but may contain arbitrary
    // library rows and poses. The default is a correct scalar wrapper; a
    // trajectory-grain implementation overrides it. `count` may be any tail
    // from 1 through the selected width.
    virtual void evaluate_batch(const ShapePoseRequest* requests, size_t count,
                                ShapeEvaluation* outputs) const;
    virtual PreparedShapeBatchHandle
    prepare_batch(const Conformer* const* libraries, size_t count) const;
    virtual void evaluate_batch(const ShapePoseRequest* requests, size_t count,
                                const PreparedShapeBatch* batch,
                                ShapeEvaluation* outputs) const;

protected:
    BoundShapeExecution
    make_bound_execution(const Conformer& library,
                         const ShapeLibraryWorkspace* workspace,
                         BoundShapeExecution::Evaluate evaluate) const;

private:
    struct QueryStorage;
    std::unique_ptr<QueryStorage> query_;
    ShapeEvaluatorConfig config_;
};

using PreparedShapeEvaluatorHandle =
    std::shared_ptr<const PreparedShapeEvaluator>;

// Implemented in separate translation units to keep each evaluator out of the
// shared optimizer's large dispatch body: one analytic factory, and each
// independently stamped field factory against this seam.
PreparedShapeEvaluatorHandle
make_analytic_shape_evaluator(const ShapeEvaluatorConfig& config,
                              const Conformer& query_reduced,
                              ShapeQueryMetadata metadata = {});
PreparedShapeEvaluatorHandle
make_scalar_field_shape_evaluator(const ShapeEvaluatorConfig& config,
                                  const Conformer& query_reduced,
                                  ShapeQueryMetadata metadata = {});
PreparedShapeEvaluatorHandle
make_paired_channel_field_shape_evaluator(const ShapeEvaluatorConfig& config,
                                          const Conformer& query_reduced,
                                          ShapeQueryMetadata metadata = {});
PreparedShapeEvaluatorHandle
make_pipelined_field_shape_evaluator(const ShapeEvaluatorConfig& config,
                                     const Conformer& query_reduced,
                                     ShapeQueryMetadata metadata = {});
ShapeLibraryWorkspace
prepare_analytic_shape_library_workspace(const Conformer& library,
                                         ShapeGrain grain, int width);
ShapeLibraryWorkspace
prepare_field_shape_library_workspace(const Conformer& library,
                                      ShapeBackend backend, int width);

}  // namespace cpu
}  // namespace r3

// Query-prepared shape-field adapters over the established field sampler.
//
// The field path of the CPU shape evaluator. Instead of summing
// analytic Gaussian overlaps over every query/library atom pair, the query's
// shape density is built ONCE per query into a grid of four stored,
// prefiltered channels [f, gx, gy, gz]; each pose then costs one trilinear
// sample per library atom. Gradients are stored and interpolated, never
// differenced (color fields follow the same rule). The field steers
// the pose search; reported scores come from exact analytic rescoring of the
// converged pose (field.hpp, NUMERICS_POLICY section 2).
//
// Three arms implement the PreparedShapeEvaluator interface over one grid:
//   Scalar         one library atom per sample, portable scalar code.
//   PairedChannel  AVX2: two samples x four channels per YMM (field_avx2.hpp);
//                  on AArch64 the NEON field_gradient_neon.
//   Pipelined      eight prepared library atoms per block, one block ahead
//                  (`PIPE=true`); needs the library ConformerSoA
//                  workspace, which caps a library conformer at
//                  kConformerSoACapacity atoms.
// Storage (FP32 or FP16 nodes) and interpolation precision are independent
// selectors, chosen once at query preparation; the metadata and `stamp()`
// report the ACTUAL mode and any fallback, not just what was requested
// Several private `R3_FIELD_*` compile-time controls select the ablation
// arms; the defaults live in field_avx2.hpp and
// field_neon.hpp.
#include "shape_field_evaluator.hpp"

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "score.hpp"
#include "field.hpp"
#include "field_avx2.hpp"
#if defined(__aarch64__)
#include "field_neon.hpp"
#include "field_neon_dataflow.hpp"
#endif

namespace r3 {
namespace cpu {
namespace {
#ifndef R3_FIELD_NEON_POLICY
#define R3_FIELD_NEON_POLICY 0
#endif
// Private compile controls: 0 existing F0/H0/H2, 1 F0_Q, 3 H1.
// The explicit policy names:
//   F0    FP32 nodes, FP32 XYZ interpolation and reduction.
//   F0_Q  the H0 half-quantized nodes widened ONCE into FP32 storage, which
//         isolates the value change from the runtime conversion cost.
//   H0    FP16 nodes widened per sample; FP32 XYZ and reduction.
//   H1    FP16 nodes; X lerp by widening half FMAs (needs ASIMDFHM).
//   H2    FP16 nodes; native-half Y/Z lerps, FP32 X last (MixedFP16YZ).

enum class FieldArm { Scalar, PairedChannel, Pipelined };

// Each arm accepts exactly one backend/grain/width triple, so a config cannot
// silently land on a different kernel than the one it names. FP16 storage is
// implemented only by the pipelined arm, and the mixed-half Y/Z interpolation
// only on top of FP16 storage.
void validate_field_config(const ShapeEvaluatorConfig& config, FieldArm arm) {
    const ShapeBackend expected_backend =
        arm == FieldArm::Scalar          ? ShapeBackend::ScalarField
        : arm == FieldArm::PairedChannel ? ShapeBackend::PairedChannelField
                                         : ShapeBackend::PipelinedField;
    const ShapeGrain expected_grain =
        arm == FieldArm::Pipelined || arm == FieldArm::PairedChannel
            ? ShapeGrain::LibraryAtom
            : ShapeGrain::Scalar;
    const int expected_width = arm == FieldArm::Scalar ? 1 : 8;
    if (config.backend != expected_backend || config.grain != expected_grain ||
        config.width != expected_width)
        throw std::invalid_argument(
            "unsupported shape-field backend/grain/width");
    if (!std::isfinite(config.field_spacing_angstrom) ||
        config.field_spacing_angstrom <= 0.0f ||
        !std::isfinite(config.field_padding_angstrom) ||
        config.field_padding_angstrom <= 0.0f)
        throw std::invalid_argument(
            "shape-field spacing and padding must be positive");
    if (config.storage != ShapeFieldStorage::FP32 &&
        config.storage != ShapeFieldStorage::FP16)
        throw std::invalid_argument("unknown shape-field storage format");
    if (config.interpolation != ShapeFieldInterpolation::FP32 &&
        config.interpolation != ShapeFieldInterpolation::MixedFP16YZ)
        throw std::invalid_argument("unknown shape-field interpolation format");
    if (config.interpolation == ShapeFieldInterpolation::MixedFP16YZ &&
        config.storage != ShapeFieldStorage::FP16)
        throw std::invalid_argument(
            "mixed FP16 Y/Z interpolation requires FP16 field storage");
    if (config.storage == ShapeFieldStorage::FP16 && arm != FieldArm::Pipelined)
        throw std::invalid_argument(
            "FP16 field storage is implemented only by the pipelined field arm");
}

void validate_query(const Conformer& query) {
    if (query.n_atoms < 0 || (query.n_atoms > 0 && query.xyz == nullptr))
        throw std::invalid_argument("invalid reduced shape-field query");
    const size_t count = size_t(query.n_atoms) * 3;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(query.xyz[i]))
            throw std::invalid_argument(
                "shape-field query coordinates must be finite");
}

// `a * b` into `result`; false instead of wrapping on size_t overflow.
bool checked_mul(size_t a, size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    result = a * b;
    return true;
}

// Validate the grid arithmetic before plan_field converts its ceil'd extents
// to int or build_field allocates nodes/tail. All units here are explicit:
// cell extents are dimensionless counts; row/plane/tail values are float
// elements, not bytes or half elements.
bool field_geometry_representable(const Conformer& query,
                                  float spacing_angstrom,
                                  float padding_angstrom) {
    // Spacing and padding in the reduced-coordinate units the query uses
    // (angstrom times COORD_SCALE), mirroring what plan_field computes.
    const double hs = double(spacing_angstrom) * double(COORD_SCALE);
    const double pad = double(padding_angstrom) * double(COORD_SCALE);
    if (!(std::isfinite(hs) && hs > 0.0 && std::isfinite(pad))) return false;
    const float hs_f = float(hs), inv_hs_f = float(1.0 / hs);
    if (!(std::isfinite(hs_f) && hs_f > 0.f && std::isfinite(inv_hs_f) &&
          inv_hs_f > 0.f))
        return false;

    // Axis-aligned bounding box of the query atoms; an empty query uses a
    // zero box at the origin, as plan_field does.
    double lo[3] = {0.0, 0.0, 0.0};
    double hi[3] = {0.0, 0.0, 0.0};
    if (query.n_atoms > 0) {
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = double(query.xyz[axis]);
            hi[axis] = lo[axis];
        }
        for (int atom = 1; atom < query.n_atoms; ++atom)
            for (int axis = 0; axis < 3; ++axis) {
                const double v = double(query.xyz[3 * atom + axis]);
                lo[axis] = std::min(lo[axis], v);
                hi[axis] = std::max(hi[axis], v);
            }
    }
    size_t cells[3]{};
    for (int axis = 0; axis < 3; ++axis) {
        const double origin = lo[axis] - pad;
        const double upper = hi[axis] + pad;
        const double extent = std::ceil((upper - origin) / hs);
        if (!(std::isfinite(origin) && std::isfinite(upper) &&
              std::isfinite(extent) && extent >= 1.0 &&
              extent <= double(std::numeric_limits<int>::max() - 1)) ||
            !std::isfinite(float(origin)))
            return false;
        cells[axis] = size_t(extent) + 1;  // nodes = cells + one endpoint
    }

    // Node storage is four floats per node; `row` is one x-row of nodes and
    // `tail` the padding after the node array, both in float elements.
    size_t nodes_xy = 0, nodes = 0, row = 0, plane_elements = 0;
    if (!checked_mul(cells[0], cells[1], nodes_xy) ||
        !checked_mul(nodes_xy, cells[2], nodes) ||
        !checked_mul(cells[0], size_t(4), row) ||
        !checked_mul(row, cells[1], plane_elements))
        return false;
    size_t tail = 0, tail_with_row = 0, values = 0;
    if (!checked_mul(row, cells[1], tail) ||
        tail > std::numeric_limits<size_t>::max() - row - 8)
        return false;
    tail_with_row = tail + row + 8;
    if (!checked_mul(nodes, size_t(4), values) ||
        values > std::numeric_limits<size_t>::max() - tail_with_row)
        return false;
    // PreparedBlock offsets (including the zero sentinel) are uint32_t,
    // while prepare_block constructs row/plane constants with signed-int
    // multiplication and left shifts. size_t/vector capacity alone is not
    // sufficient: reject these geometries before any field allocation.
    if (values > size_t(std::numeric_limits<uint32_t>::max()) ||
        row > size_t(std::numeric_limits<int>::max()) ||
        plane_elements > size_t(std::numeric_limits<int>::max()))
        return false;
    values += tail_with_row;
    const AlignedVector<float> empty;
    return values <= empty.max_size() && plane_elements <= empty.max_size();
}

// FP16 storage is used only if every stored channel value is finite in half
// precision (|v| <= 65504, the largest finite half). Otherwise the evaluator
// keeps FP32 and records the fallback; it never stores an infinity.
bool field_fits_fp16(const FieldNaive32& field) {
    constexpr float max_half = 65504.0f;
    for (float value : field.v)
        if (!std::isfinite(value) || std::fabs(value) > max_half) return false;
    return true;
}

bool field_fits_mixed_fp16_yz(const FieldNaive32& field) {
    // Conservative global bound checked once at query preparation. With node
    // magnitudes <=32720, an endpoint difference is <=65440; rounding its
    // half subtraction can add at most 16, and the final half FMA rounding at
    // most 8, so a Y result is <=32744. The Z endpoint difference is then
    // <=65488 and rounds to at most 65504, still finite; the final Z result is
    // <=32768. Fractions are in [0,1] before half conversion. This leaves
    // margin for operation rounding and avoids per-corner hot-loop checks.
    constexpr float max_abs = 32720.0f;
    for (float value : field.v)
        if (!std::isfinite(value) || std::fabs(value) > max_abs) return false;
    return true;
}

// Runtime ISA probes. The build may target a feature that the running CPU
// lacks, so FP16 storage is gated at query preparation, never per sample.
bool fp16_storage_supported() noexcept {
#if defined(__aarch64__)
    return field_neon_fp16_supported();
#elif defined(__AVX2__) && defined(__F16C__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c");
#else
    return false;
#endif
}

bool mixed_fp16_yz_supported() noexcept {
#if defined(__aarch64__)
    return field_neon_fp16_arithmetic_supported();
#else
    // This candidate requires native AArch64 half-vector arithmetic. F16C
    // remains sufficient for the storage-only FP16 arm.
    return false;
#endif
}

void validate_request(const ShapePoseRequest& request) {
    validate_shape_pose_request(request);
    if (request.library->n_atoms < 0 ||
        (request.library->n_atoms > 0 && request.library->xyz == nullptr) ||
        !std::isfinite(request.gnorm))
        throw std::invalid_argument("invalid shape-field pose request");
}

// One query's prepared field evaluator. Construction validates the config and
// the reduced query, proves the grid geometry representable, builds the FP32
// field, and then (if FP16 was requested and every guard passes) quantizes it.
// Exactly one of `field32_` / `field16_` holds the live grid afterward. The
// object is immutable after construction, so one handle serves every worker
// thread scoring against this query (query-field reuse).
class PreparedFieldShapeEvaluator final : public PreparedShapeEvaluator {
public:
    PreparedFieldShapeEvaluator(const ShapeEvaluatorConfig& config,
                                const Conformer& query,
                                ShapeQueryMetadata metadata, FieldArm arm)
        : PreparedShapeEvaluator(config, query, metadata), arm_(arm) {
        validate_field_config(config, arm);
        validate_query(query_reduced());

        const Conformer& owned_query = query_reduced();
        if (!field_geometry_representable(owned_query,
                                          config.field_spacing_angstrom,
                                          config.field_padding_angstrom))
            throw std::invalid_argument(
                "shape-field dimensions, strides, or allocation size are not representable");
        const auto started = std::chrono::steady_clock::now();
        FieldNaive32 built;
        build_shape_field_grid(owned_query.xyz, owned_query.n_atoms,
                               config.field_spacing_angstrom,
                               config.field_padding_angstrom, built);
        metadata_.query_atoms = owned_query.n_atoms;
        metadata_.spacing_angstrom = config.field_spacing_angstrom;
        metadata_.padding_angstrom = config.field_padding_angstrom;
        metadata_.nodes_x = built.g.nodes_x();
        metadata_.nodes_y = built.g.nodes_y();
        metadata_.nodes_z = built.g.nodes_z();
        metadata_.query_snapshot_bytes =
            size_t(owned_query.n_atoms) * 3 * sizeof(float);
        metadata_.field_build_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started)
                .count();
        // FP16 request: the guards run in order and the first failure names
        // the fallback; the evaluator then stays on FP32. Under the private
        // F0_Q control (POLICY 1) the pipelined arm quantizes even without an
        // FP16 request and widens the result straight back to FP32.
        if (config.storage == ShapeFieldStorage::FP16
#if defined(__aarch64__) && R3_FIELD_NEON_POLICY == 1
            || arm == FieldArm::Pipelined
#endif
        ) {
#if defined(__aarch64__) && R3_FIELD_NEON_POLICY == 3
            if (arm == FieldArm::Pipelined &&
                !field_neon_fp16_fml_supported()) {
                metadata_.storage_fallback = "widening_fma_unavailable";
            } else
#endif
                if (config.interpolation ==
                        ShapeFieldInterpolation::MixedFP16YZ &&
                    !mixed_fp16_yz_supported()) {
                metadata_.storage_fallback =
                    "native_half_arithmetic_unavailable";
            } else if (config.interpolation ==
                           ShapeFieldInterpolation::MixedFP16YZ &&
                       !field_fits_mixed_fp16_yz(built)) {
                metadata_.storage_fallback = "mixed_half_abs_bound_exceeded";
            } else if (!fp16_storage_supported()) {
                metadata_.storage_fallback = "runtime_feature_unavailable";
            } else if (!field_fits_fp16(built)) {
                metadata_.storage_fallback =
                    "field_value_outside_finite_fp16_range";
            } else {
                const auto quantize_started = std::chrono::steady_clock::now();
#if defined(__aarch64__)
                quantize_field_neon_fp16(built, field16_);
#elif defined(__AVX2__) && defined(__F16C__)
                quantize_field(built, field16_);
#endif
                metadata_.field_quantize_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - quantize_started)
                        .count();
                metadata_.actual_storage = ShapeFieldStorage::FP16;
                metadata_.actual_interpolation = config.interpolation;
#if defined(__aarch64__) && R3_FIELD_NEON_POLICY == 3
                if (arm == FieldArm::Pipelined)
                    metadata_.actual_interpolation =
                        ShapeFieldInterpolation::FP32;
#endif
#if defined(__aarch64__) && R3_FIELD_NEON_POLICY == 1
                // Same quantized bits as H0, widened once outside hot work.
                widen_field_neon_fp16(field16_, built);
                metadata_.actual_storage = ShapeFieldStorage::FP32;
                metadata_.actual_interpolation = ShapeFieldInterpolation::FP32;
                field16_.v.clear();
#endif
            }
        }
        if (metadata_.actual_storage == ShapeFieldStorage::FP16)
            metadata_.field_bytes = field16_.bytes();
        else {
            field32_ = std::move(built);
            metadata_.field_bytes = field32_.bytes();
        }
    }

    // A human-readable description of the arithmetic actually compiled and
    // selected (backend, grain, ISA, storage, reduction order), recorded with
    // results so a number can be tied to the kernel that produced it. The
    // strings describe the ACTUAL storage/fallback, and the #if blocks mirror
    // the private compile controls that change the kernel.
    ShapeEvaluatorStamp stamp() const override {
        ShapeEvaluatorStamp out;
        out.width = config().width;
        out.exponential =
            "FP32 std::exp2 during query-field build; none per pose";
        out.storage =
            metadata_.actual_storage == ShapeFieldStorage::FP16
                ? (metadata_.actual_interpolation ==
                           ShapeFieldInterpolation::MixedFP16YZ
                       ? "FP16 values; native-half Y/Z lerps; FP32 coordinates, X lerp and reductions"
                       : "FP16 values; FP32 coordinates/interpolation/reductions")
                : "FP32 values";
        switch (arm_) {
        case FieldArm::Scalar:
            out.backend = "scalar_field_fp32_prefiltered_half4";
            out.grain = "one library atom sample; four stored channels";
            out.isa = "scalar";
            out.reduction = "scalar source-order FP32 sample accumulation";
            out.vector_path = false;
            out.requires_library_soa = false;
            break;
        case FieldArm::PairedChannel:
            out.backend = "paired_channel_field_fp32_prefiltered_half4";
            out.grain = "two library samples by four stored channels";
            out.isa = "AVX2+FMA";
            out.reduction = "paired-channel FP32 fold with scalar odd tail";
            out.vector_path = true;
            out.requires_library_soa = false;
            break;
        case FieldArm::Pipelined:
#if defined(__AVX2__)
            if constexpr (R3_FIELD_AVX2_XLAST != 0) {
                out.backend =
                    metadata_.actual_storage == ShapeFieldStorage::FP16
                        ? "pipelined_field_fp16storage_fp32yzx_prefiltered_half4"
                        : "pipelined_field_fp32yzx_prefiltered_half4";
            } else if (metadata_.actual_storage == ShapeFieldStorage::FP16 &&
                       R3_FIELD_AVX2_PACKED_HALF != 0) {
                out.backend =
                    "pipelined_field_fp16storage_fp32xyz_packed_prefiltered_half4";
            } else
#endif
                out.backend =
                    metadata_.actual_storage == ShapeFieldStorage::FP16
                        ? (metadata_.actual_interpolation ==
                                   ShapeFieldInterpolation::MixedFP16YZ
                               ? "pipelined_field_fp16mixedyz_fp32x_prefiltered_half4"
                               : "pipelined_field_fp16storage_fp32xyz_prefiltered_half4")
                        : "pipelined_field_fp32_prefiltered_half4";
            out.grain = "eight library atoms; pipelined general-bounds path";
            out.isa = metadata_.actual_storage == ShapeFieldStorage::FP16
                          ? "AVX2+FMA+F16C"
                          : "AVX2+FMA";
            out.reduction = "eight-atom FP32 lanes then horizontal reduction";
            out.vector_path = true;
            out.requires_library_soa = true;
            break;
        }
#if defined(__aarch64__)
        if (arm_ != FieldArm::Scalar) {
            out.isa = metadata_.actual_storage == ShapeFieldStorage::FP16
                          ? "NEON+ASIMDHP"
                          : "NEON";
            out.grain =
                "eight prepared atoms; four stored channels per NEON vector";
            out.reduction =
                "source-order FP32 channel-vector accumulation; scalar torque";
        }
#endif
        if (std::string(metadata_.storage_fallback) != "none")
            out.storage += std::string("; requested FP16 fallback=") +
                           metadata_.storage_fallback;
#if defined(__aarch64__) && R3_FIELD_NEON_POLICY == 1
        if (arm_ == FieldArm::Pipelined &&
            metadata_.storage_fallback == std::string("none")) {
            out.backend =
                "pipelined_field_fp32_quantized_once_prefiltered_half4";
            out.storage =
                "FP32 values; half-quantized nodes widened once; FP32 XYZ and reductions";
        }
#elif defined(__aarch64__) && R3_FIELD_NEON_POLICY == 3
        if (arm_ == FieldArm::Pipelined &&
            metadata_.actual_storage == ShapeFieldStorage::FP16) {
            out.backend =
                "pipelined_field_fp16_widening_fma_x_fp32yz_prefiltered_half4";
            out.storage =
                "FP16 values; half-rounded X fraction; widening FP32 X FMA; FP32 Y/Z and reductions";
            out.isa = "NEON+ASIMDHP+ASIMDFHM";
        }
#endif
        // Scheduling is an independent private compile-time selector.
#if defined(__aarch64__) && R3_FIELD_NEON_SCHEDULE != 0
        if (arm_ == FieldArm::Pipelined) {
#if R3_FIELD_NEON_SCHEDULE == 1
            out.backend += "_two_stencil_trees";
            out.grain =
                "block8 full prepared state; two interleaved actual stencils; source-order reduction";
#elif R3_FIELD_NEON_SCHEDULE == 2
            out.backend += "_pointer_pipeline";
            out.grain =
                "block8 one-ahead full prepared state; two persistent buffers; pointer swap";
#else
            out.backend += "_prepared_control";
            out.grain = "block8 full prepared state; scalar stencil control";
#if R3_FIELD_NEON_PREP
            out.grain =
                "block8 full prepared state; NEON4 atom preparation; scalar stencil control";
#endif
#if R3_FIELD_NEON_ADDRESS
            if (metadata_.actual_storage == ShapeFieldStorage::FP32 &&
                field32_.neon_offsets32) {
                out.grain =
                    "block8 full prepared state; NEON4 atom preparation; proven uint32 element BASE; explicit XYZ stencil";
#if R3_NEON_PAIRED_STENCILS && R3_FIELD_NEON_TORQUE
                out.grain += "; two interleaved independent XYZ stencils";
#endif
#if R3_FIELD_NEON_TORQUE
                out.reduction =
                    "source-order FP32 channel accumulation; NEON4 across-atom world-lever cross products and source-order scalar torque fold";
#endif
            } else if (metadata_.actual_storage == ShapeFieldStorage::FP32) {
                out.grain =
                    "block8 full prepared state; NEON4 atom preparation; checked size_t offset fallback";
            }
#endif
#endif
        }
#endif
        return out;
    }

    // Unbound route: validates the request on every call, then dispatches to
    // the arm. The bound route below skips the checks that preparation fixed.
    void evaluate_one(const ShapePoseRequest& request,
                      ShapeEvaluation& output) const override {
        validate_request(request);
        switch (arm_) {
        case FieldArm::Scalar:
            evaluate_selected<FieldArm::Scalar, false>(request, output);
            break;
        case FieldArm::PairedChannel:
            evaluate_selected<FieldArm::PairedChannel, false>(request, output);
            break;
        case FieldArm::Pipelined:
            evaluate_selected<FieldArm::Pipelined, false>(request, output);
            break;
        }
    }

    // Bind one library conformer (and its workspace) to this query once per
    // trajectory, so the per-pose call does not re-check properties fixed by
    // preparation: the trusted execution contract. `values_only` selects
    // the score-only specialization for final winner selection, which would
    // otherwise compute SEARCH gradients and torque only to discard them;
    // it exists only for the pipelined arm on FP32 storage.
    BoundShapeExecution
    bind_execution(const Conformer& library,
                   const ShapeLibraryWorkspace* workspace,
                   bool values_only = false) const override {
        // No pose is supplied here: all per-pose field address/domain checks
        // remain in the selected numerical kernel.
        if (arm_ == FieldArm::Pipelined &&
            (workspace == nullptr || !workspace->matches(library) ||
             workspace->field_atom == nullptr ||
             workspace->field_atom->n != library.n_atoms ||
             library.n_atoms > kConformerSoACapacity))
            throw std::invalid_argument(
                "bound pipelined field requires the matching library SoA");
        if (values_only && arm_ == FieldArm::Pipelined &&
            metadata_.actual_storage == ShapeFieldStorage::FP32)
            return bind_selected<FieldArm::Pipelined, true>(library, workspace);
        switch (arm_) {
        case FieldArm::Scalar:
            return bind_selected<FieldArm::Scalar>(library, workspace);
        case FieldArm::PairedChannel:
            return bind_selected<FieldArm::PairedChannel>(library, workspace);
        case FieldArm::Pipelined:
            return bind_selected<FieldArm::Pipelined>(library, workspace);
        }
        throw std::logic_error("invalid field execution route");
    }

    // The bound callback is a captureless lambda (a plain function pointer
    // plus the evaluator pointer), with the arm and value-only choice fixed
    // as template arguments, so the per-pose call makes no arm switch.
    template <FieldArm Arm, bool ValuesOnly = false>
    BoundShapeExecution
    bind_selected(const Conformer& library,
                  const ShapeLibraryWorkspace* workspace) const {
        return make_bound_execution(
            library, workspace,
            [](const void* evaluator, const ShapePoseRequest& request,
               ShapeEvaluation& output) {
                static_cast<const PreparedFieldShapeEvaluator*>(
                    static_cast<const PreparedShapeEvaluator*>(evaluator))
                    ->evaluate_selected<Arm, true, ValuesOnly>(request, output);
            });
    }

    // Score one pose: `s_ab` is the field-sampled shape overlap of the posed
    // library conformer with the query, and `descent` the seven-component
    // pose gradient in the layout the field samplers fill (field.hpp).
    // `Bound` means bind_execution already checked the workspace.
    template <FieldArm Arm, bool Bound, bool ValuesOnly = false>
    void evaluate_selected(const ShapePoseRequest& request,
                           ShapeEvaluation& output) const {
        output = {};
        float score = 0.0f;
        float descent[7]{};
        switch (Arm) {
        case FieldArm::Scalar:
            field_gradient(field32_, *request.library, request.matrix,
                           request.translation, request.gnorm, descent, &score,
                           nullptr);
            break;
        case FieldArm::PairedChannel:
#if defined(__AVX2__)
            field_gradient_avx2(field32_, *request.library, request.matrix,
                                request.translation, request.gnorm, descent,
                                &score, nullptr);
#elif defined(__aarch64__)
            field_gradient_neon(field32_, *request.library, request.matrix,
                                request.translation, request.gnorm, descent,
                                &score);
#else
            throw std::runtime_error("paired-channel field requires AVX2");
#endif
            break;
        case FieldArm::Pipelined:
#if defined(__AVX2__) || defined(__aarch64__)
            evaluate_pipelined<Bound, ValuesOnly>(request, descent, score);
#else
            throw std::runtime_error("pipelined field requires AVX2");
#endif
            break;
        }
        output.s_ab = score;
        for (int k = 0; k < 7; ++k)
            output.descent[k] = descent[k];
        const uint64_t n = uint64_t(request.library->n_atoms);
        output.counts.field_samples = n;
        // These counters use sample units, not analytic query/library pair
        // terms. The pipeline issues a full eight-lane tail block; its padded
        // lanes are excluded from field_samples and included here.
        output.counts.useful_terms = n;
        output.counts.issued_terms =
#if defined(__aarch64__)
            n;  // ARM samples real atoms; vector lanes are the four channels.
#else
            Arm == FieldArm::Pipelined && n != 0 ? ((n + 7) & ~uint64_t(7)) : n;
#endif
    }

    const ShapeFieldPreparationMetadata& preparation_metadata() const noexcept {
        return metadata_;
    }

    // FP32 grid view for callers that sample the same query field elsewhere.
    // An FP16-stored evaluator has discarded its FP32 grid, so it refuses.
    const FieldNaive32& grid() const {
        if (metadata_.actual_storage != ShapeFieldStorage::FP32)
            throw std::logic_error(
                "FP16-stored shape field has no FP32 grid view");
        return field32_;
    }

private:
#if defined(__AVX2__) || defined(__aarch64__)
    // The pipelined arm. AArch64 picks among the NEON kernels by actual
    // storage and interpolation (and the private schedule/address controls); x86
    // instantiates field_gradient_avx2_reg once per storage type. ConformerSoA
    // holds at most kConformerSoACapacity atoms, hence the n_atoms guard.
    template <bool Bound, bool ValuesOnly>
    void evaluate_pipelined(const ShapePoseRequest& request, float descent[7],
                            float& score) const {
        const ShapeLibraryWorkspace* workspace = request.library_workspace;
        if (!Bound &&
            (workspace == nullptr || !workspace->matches(*request.library) ||
             workspace->field_atom == nullptr ||
             workspace->field_atom->n != request.library->n_atoms ||
             request.library->n_atoms > kConformerSoACapacity))
            throw std::invalid_argument(
                "pipelined field requires the matching prepared library SoA");
        // The `PIPE=true`, `STAGE=0`, general-bounds specialization. Do not
        // silently enable ALLIN or GRIDXF: those are different reported arms.
#if defined(__aarch64__)
        if constexpr (ValuesOnly) {
            field_value_neon_pipelined(field32_, *workspace->field_atom,
                                       request.matrix, request.translation,
                                       &score);
            return;
        }
#if R3_FIELD_NEON_ADDRESS && R3_FIELD_NEON_SCHEDULE == 3
        if constexpr (!ValuesOnly) {
            // The accepted FP32 full-gradient route remains literal/inlined
            // independently of the final-values specialization. All prepared
            // geometry, per-pose domain, sentinel and tail checks are retained.
            if (metadata_.actual_storage == ShapeFieldStorage::FP32 &&
                field32_.neon_offsets32) {
                detail::gradient_base32(field32_, *workspace->field_atom,
                                        request.matrix, request.translation,
                                        request.gnorm, descent, &score);
                return;
            }
        }
#endif
#if R3_FIELD_NEON_POLICY == 3
        if (metadata_.actual_storage == ShapeFieldStorage::FP16)
            field_gradient_neon_fp16_fml_pipelined(
                field16_, *workspace->field_atom, request.matrix,
                request.translation, request.gnorm, descent, &score);
        else
#endif
            if (metadata_.actual_storage == ShapeFieldStorage::FP16 &&
                metadata_.actual_interpolation ==
                    ShapeFieldInterpolation::MixedFP16YZ)
            field_gradient_neon_fp16_mixed_yz_pipelined(
                field16_, *workspace->field_atom, request.matrix,
                request.translation, request.gnorm, descent, &score);
        else if (metadata_.actual_storage == ShapeFieldStorage::FP16)
            field_gradient_neon_fp16_pipelined(
                field16_, *workspace->field_atom, request.matrix,
                request.translation, request.gnorm, descent, &score);
        else
            field_gradient_neon_pipelined(field32_, *workspace->field_atom,
                                          request.matrix, request.translation,
                                          request.gnorm, descent, &score);
#else
#ifndef R3_FIELD_AVX2_PIPE
#define R3_FIELD_AVX2_PIPE 1
#endif
#ifndef R3_FIELD_AVX2_PROOF_DISPATCH
#define R3_FIELD_AVX2_PROOF_DISPATCH 0
#endif
#ifndef R3_FIELD_AVX2_COVERAGE
#define R3_FIELD_AVX2_COVERAGE 0
#endif
        // Generic over FieldNaive32 / FieldNaive16. By default
        // (PROOF_DISPATCH and COVERAGE both 0) this is a single call with
        // ALLIN=false and GRIDXF=false: the per-pose `conformer_inside` proof
        // and its all-inside fast path are private diagnostics here.
        auto evaluate = [&](const auto& field) {
            bool inside = false;
            if constexpr (R3_FIELD_AVX2_PROOF_DISPATCH || R3_FIELD_AVX2_COVERAGE)
                inside = conformer_inside(field, *workspace->field_atom,
                                          request.matrix, request.translation);
            if constexpr (R3_FIELD_AVX2_COVERAGE) {
                struct Coverage {
                    uint64_t hit = 0, fallback = 0, full_blocks = 0, tails = 0;
                    ~Coverage() {
                        std::fprintf(
                            stderr,
                            "field AVX2 coverage hit=%llu fallback=%llu full_blocks=%llu tails=%llu\n",
                            (unsigned long long)hit,
                            (unsigned long long)fallback,
                            (unsigned long long)full_blocks,
                            (unsigned long long)tails);
                    }
                };
                static thread_local Coverage coverage;
                if (inside) {
                    ++coverage.hit;
                    coverage.full_blocks += workspace->field_atom->n / 8;
                    coverage.tails += (workspace->field_atom->n % 8) != 0;
                } else
                    ++coverage.fallback;
            }
            if constexpr (R3_FIELD_AVX2_PROOF_DISPATCH) {
                if (inside) {
                    field_gradient_avx2_reg<false, (R3_FIELD_AVX2_PIPE != 0), 0, true,
                                            false>(
                        field, *workspace->field_atom, request.matrix,
                        request.translation, request.gnorm, descent, &score,
                        nullptr);
                    return;
                }
            }
            // Template order: EDGE, PIPE, STAGE, ALLIN, GRIDXF, XLAST,
            // STRICT_BOUNDS, ALTERNATING, TWO_PAIRS, NEED_GRADIENT.
            field_gradient_avx2_reg<false, (R3_FIELD_AVX2_PIPE != 0), 0, false, false,
                                    (R3_FIELD_AVX2_XLAST != 0),
                                    (R3_FIELD_AVX2_STRICT_BOUNDS != 0),
                                    (R3_FIELD_AVX2_ALTERNATING != 0),
                                    (R3_FIELD_AVX2_SCHEDULE != 0), !ValuesOnly>(
                field, *workspace->field_atom, request.matrix,
                request.translation, request.gnorm, descent, &score, nullptr);
        };
        if (metadata_.actual_storage == ShapeFieldStorage::FP16)
            evaluate(field16_);
        else
            evaluate(field32_);
#endif
    }
#endif

    FieldArm arm_;
    FieldNaive32 field32_;
    FieldNaive16 field16_;
    ShapeFieldPreparationMetadata metadata_{};
};

// Validate before construction so the factories reject an unsupported arm
// for this build (no AVX2 and no AArch64) before any field is built.
ShapeEvaluatorConfig checked_config(const ShapeEvaluatorConfig& config,
                                    FieldArm arm) {
    validate_field_config(config, arm);
#if !defined(__AVX2__) && !defined(__aarch64__)
    if (arm != FieldArm::Scalar)
        throw std::invalid_argument("vector shape-field backend requires AVX2");
#endif
    return config;
}

}  // namespace

PreparedShapeEvaluatorHandle
make_scalar_field_shape_evaluator(const ShapeEvaluatorConfig& config,
                                  const Conformer& query_reduced,
                                  ShapeQueryMetadata metadata) {
    return std::make_shared<PreparedFieldShapeEvaluator>(
        checked_config(config, FieldArm::Scalar), query_reduced, metadata,
        FieldArm::Scalar);
}

PreparedShapeEvaluatorHandle
make_paired_channel_field_shape_evaluator(const ShapeEvaluatorConfig& config,
                                          const Conformer& query_reduced,
                                          ShapeQueryMetadata metadata) {
    return std::make_shared<PreparedFieldShapeEvaluator>(
        checked_config(config, FieldArm::PairedChannel), query_reduced,
        metadata, FieldArm::PairedChannel);
}

PreparedShapeEvaluatorHandle
make_pipelined_field_shape_evaluator(const ShapeEvaluatorConfig& config,
                                     const Conformer& query_reduced,
                                     ShapeQueryMetadata metadata) {
    return std::make_shared<PreparedFieldShapeEvaluator>(
        checked_config(config, FieldArm::Pipelined), query_reduced, metadata,
        FieldArm::Pipelined);
}

// Per-library-conformer preparation, built once and reused across poses and
// starts. Only the pipelined arm needs anything: an SoA copy of the library
// coordinates (x[], y[], z[] padded to a multiple of eight) that its
// eight-atom blocks load directly. The other arms read `library.xyz`.
ShapeLibraryWorkspace
prepare_field_shape_library_workspace(const Conformer& library,
                                      ShapeBackend backend, int width) {
    if (library.n_atoms < 0 || (library.n_atoms > 0 && library.xyz == nullptr))
        throw std::invalid_argument(
            "invalid library conformer for field workspace");
    ShapeLibraryWorkspace workspace;
    workspace.source_xyz = library.xyz;
    workspace.source_atoms = library.n_atoms;
    if (backend == ShapeBackend::PipelinedField ||
        backend == ShapeBackend::PairedChannelField) {
        if (width != 8)
            throw std::invalid_argument(
                "vector field workspace requires width 8");
    } else if (backend == ShapeBackend::ScalarField) {
        if (width != 1)
            throw std::invalid_argument(
                "scalar field workspace requires width 1");
    } else {
        throw std::invalid_argument("not a field backend");
    }
    if (backend == ShapeBackend::PipelinedField) {
#if defined(__AVX2__) || defined(__aarch64__)
        if (library.n_atoms > kConformerSoACapacity)
            throw std::invalid_argument("field SoA supports at most " +
                                        std::to_string(kConformerSoACapacity) +
                                        " atoms");
        auto soa = std::make_shared<ConformerSoA>();
        soa->build(library);
        workspace.prepared_bytes = sizeof(ConformerSoA);
        workspace.field_atom = std::move(soa);
#else
        throw std::invalid_argument("pipelined field workspace requires AVX2");
#endif
    }
    return workspace;
}

ShapeFieldPreparationMetadata
shape_field_preparation_metadata(const PreparedShapeEvaluator& evaluator) {
    const auto* field =
        dynamic_cast<const PreparedFieldShapeEvaluator*>(&evaluator);
    if (field == nullptr)
        throw std::invalid_argument(
            "shape evaluator is not a field adapter");
    return field->preparation_metadata();
}

const FieldNaive32& shape_field_grid(const PreparedShapeEvaluator& evaluator) {
    const auto* field =
        dynamic_cast<const PreparedFieldShapeEvaluator*>(&evaluator);
    if (field == nullptr)
        throw std::invalid_argument(
            "shape evaluator is not a field adapter");
    return field->grid();
}

// The single compiled field builder for both the prepared evaluator and the
// legacy per-call route: one translation unit means one inlining/FMA
// contraction decision, so both routes produce the same grid bits.
void build_shape_field_grid(const float* query_reduced, int query_atoms,
                            float spacing, float padding, FieldNaive32& field) {
    field.g = plan_field(query_reduced, query_atoms, spacing, padding);
    build_field(query_reduced, nullptr, query_atoms, field.g, field);
}

}  // namespace cpu
}  // namespace r3

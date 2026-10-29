// Immutable per-conformer preparation for four shared-count starts.
//
// Architecture-neutral half of the prepared canonical color route used by the
// CPU combo optimizer (shape + color, analytic color, field-free canonical
// matrix). One library conformer is optimized from four independent starts;
// all four share the query and library coordinates, plane counts and offsets,
// and differ only in pose. This file:
//
//   - builds the immutable ColorCanonicalFourStartDescriptor once per
//     (query, library conformer) before the optimizer's step loop,
//   - decides once whether two adjacent conformers share an exact count
//     signature (YMM8 on x86, two-stream NEON on ARM),
//   - sizes the posed-library scratch once, and
//   - dispatches each step to the ISA helper, falling back per four-start
//     group to the FP64 formula below when a helper returns a zero mask.
//
// Lane numbering throughout is [conformer A starts 0..3, conformer B starts
// 0..3]; the helpers never reduce across starts. Measured gain over the
// retained grain-16 route, frozen eight-query one-core complete screen:
// x86 23,329 -> 25,259 conformers/s (+8.28%), ARM 12,841 -> 14,442
// (+12.46%).
#include "color_canonical_four_start.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "color_conformer_kernel.hpp"
#include "color_invariant_plan.hpp"
#include "kernel.hpp"
#include "reduced_pose.hpp"

namespace r3::cpu {
namespace {

// Coordinate envelope of the analytic FP32 route, in physical units. Raw
// library coordinates are checked against it directly; reduced-frame values
// (query, translation, transformed library) against its COORD_SCALE image,
// rounded toward zero so the FP32 bound never exceeds the double one.
constexpr double kMaximumCoordinate = 1.0e9;

float maximum_reduced_coordinate_fp32() {
    static const float value =
        std::nextafter(float(kMaximumCoordinate * double(COORD_SCALE)), 0.0f);
    return value;
}

bool safe_coordinate(float value, float maximum) {
    return std::isfinite(value) && std::abs(value) <= maximum;
}

// FP64 fallback for one start: the same canonical Gaussian sum and gradient
// as the SIMD helpers, but with the pose promoted to double, std::exp, and
// the plan's double reduced decay/coefficient. It ignores the optimizer exp
// mode, so a fallback lane does not reproduce the FP32 SEARCH arithmetic
// bit for bit; it serves only inputs outside the FP32 envelope.
// `width` is the reduced-frame Gaussian width w/s^2, so `-2*width*value` and
// `-width*value` are the scalar twins of the FP32 planes' torque_scale and
// translation_scale. descent[1..3] are rotational, descent[4..6] translation
// (ColorEvaluation's seven-slot convention; slot 0 stays zero).
ColorEvaluation exact_canonical_four_start_lane(
    const ColorCanonicalFourStartDescriptor& descriptor,
    const PreparedReducedPose& pose, const ColorInvariantPlan& plan,
    bool gradient, uint64_t* term_count) {
    ColorEvaluation out;
    double pose_matrix[9];
    double translation[3];
    for (int element = 0; element < 9; ++element)
        pose_matrix[element] = double(pose.matrix[element]);
    for (int component = 0; component < 3; ++component)
        translation[component] = double(pose.translation[component]);

    for (int plane_index = 0; plane_index < descriptor.plane_count;
         ++plane_index) {
        const ColorCanonicalFourStartPlane& plane =
            descriptor.planes[size_t(plane_index)];
        const double decay = plan.reduced_decay(plane.type, plane.type);
        const double coefficient = plan.coefficient(plane.type, plane.type);
        const double width = -2.0 * decay;
        for (int i = 0; i < plane.query_count; ++i) {
            const float* query =
                plane.query_xyz + size_t(i) * size_t(plane.query_stride);
            const double x = double(query[0]);
            const double y = double(query[1]);
            const double z = double(query[2]);
            for (int j = 0; j < plane.library_count; ++j) {
                const float* raw = plane.library_xyz +
                                   size_t(j) * size_t(plane.library_stride);
                // Library feature in physical units -> reduced frame:
                // b' = M raw + t', with M = sR (PreparedReducedPose).
                const double bx = pose_matrix[0] * double(raw[0]) +
                                  pose_matrix[1] * double(raw[1]) +
                                  pose_matrix[2] * double(raw[2]) +
                                  translation[0];
                const double by = pose_matrix[3] * double(raw[0]) +
                                  pose_matrix[4] * double(raw[1]) +
                                  pose_matrix[5] * double(raw[2]) +
                                  translation[1];
                const double bz = pose_matrix[6] * double(raw[0]) +
                                  pose_matrix[7] * double(raw[1]) +
                                  pose_matrix[8] * double(raw[2]) +
                                  translation[2];
                const double dx = x - bx;
                const double dy = y - by;
                const double dz = z - bz;
                const double value =
                    coefficient *
                    std::exp(decay * (dx * dx + dy * dy + dz * dz));
                out.value += value;
                if (gradient) {
                    // Moment arm r = b' - t'; unfactored per-pair torque.
                    const double rx = bx - translation[0];
                    const double ry = by - translation[1];
                    const double rz = bz - translation[2];
                    const double torque = -2.0 * width * value;
                    out.descent[1] += torque * (ry * dz - rz * dy);
                    out.descent[2] += torque * (rz * dx - rx * dz);
                    out.descent[3] += torque * (rx * dy - ry * dx);
                    const double scale = -width * value;
                    out.descent[4] += scale * dx;
                    out.descent[5] += scale * dy;
                    out.descent[6] += scale * dz;
                }
                if (term_count) ++*term_count;
            }
        }
    }
    return out;
}

}  // namespace

// Contract in the header: false = ineligible (generic evaluator keeps the
// pair); true with immutable_inputs_safe=false = eligible shape but some
// immutable coordinate is outside the envelope, so every evaluation takes the
// FP64 fallback. Runs once per pair before the optimizer loop, so the
// hot helpers never revalidate plane bounds or immutable coordinates.
bool prepare_color_canonical_four_start_descriptor(
    const ReducedColorConformerPairInput& pair, const ColorInvariantPlan& plan,
    ColorCanonicalFourStartDescriptor& descriptor) {
    descriptor = ColorCanonicalFourStartDescriptor{};
    if (pair.query == nullptr || pair.library == nullptr ||
        pair.query_stride < 3 || pair.library_stride < 3 ||
        plan.n_features() < 1 || plan.n_features() > MAX_COLOR_FEATURES ||
        !plan.canonical_analytic_supported() ||
        !plan.reduced_matrix_supported_for_avx2())
        return false;

    const ColorConformer& query = pair.query->features;
    const ColorConformer& library = *pair.library;
    const int n_features = plan.n_features();
    if (query.n_features != n_features || library.n_features != n_features ||
        query.n_real < 0 || library.n_real < 0 || query.n_color < 0 ||
        library.n_color < 0)
        return false;

    descriptor.n_features = n_features;
    descriptor.query_plan = &plan;
    descriptor.default_parameters = plan.default_parameters();
    descriptor.shape_count = library.n_real;
    descriptor.immutable_inputs_safe = true;
    // Color segments (types 1..n-1; type 0 is reserved for ordinary atoms
    // and not part of the color extent) must be contiguous, in type order,
    // and exactly cover n_color. Anything else is ineligible, not unsafe.
    int query_extent = 0, library_extent = 0;
    for (int type = 0; type < n_features; ++type) {
        const int query_count = query.seg_count[type];
        const int library_count = library.seg_count[type];
        if (query_count < 0 || library_count < 0 ||
            query.seg_offset[type] < 0 || library.seg_offset[type] < 0)
            return false;
        if (type != 0) {
            if (query.seg_offset[type] != query_extent ||
                library.seg_offset[type] != library_extent ||
                query_count > query.n_color - query_extent ||
                library_count > library.n_color - library_extent)
                return false;
            query_extent += query_count;
            library_extent += library_count;
        }
        descriptor.library_count[size_t(type)] = library_count;
        descriptor.library_offset[size_t(type)] = library.seg_offset[type];
    }
    if (query_extent != query.n_color || library_extent != library.n_color)
        return false;

    const float maximum_reduced = maximum_reduced_coordinate_fp32();
    const auto& plan_planes = plan.canonical_planes();
    // Issued pair terms, accumulated until at least 8. Normalization (unit-amplitude sums with one
    // coefficient multiply at the evaluator boundary) saves a multiply per
    // issued pair but adds seven at the boundary, so it is enabled only for
    // conformers with at least eight issued pairs (the "sparse >= 8 gate").
    uint64_t amplitude_terms = 0;
    for (const ColorCanonicalPlane& source : plan_planes) {
        if (source.type < 1 || source.type >= n_features) return false;
        const int type = source.type;
        const int nq = query.seg_count[type];
        const int nb = library.seg_count[type];
        // Empty planes are omitted: the helpers' plane loop is work only.
        if (nq == 0 || nb == 0) continue;
        if (amplitude_terms < 8)
            amplitude_terms +=
                std::min(uint64_t(8), uint64_t(nq) * uint64_t(nb));
        if (descriptor.plane_count >= MAX_COLOR_FEATURES ||
            query.color_xyz == nullptr || library.color_xyz == nullptr)
            return false;

        ColorCanonicalFourStartPlane& target =
            descriptor.planes[size_t(descriptor.plane_count++)];
        target.type = type;
        target.query_xyz = query.color_xyz + size_t(query.seg_offset[type]) *
                                                 size_t(pair.query_stride);
        target.library_xyz =
            library.color_xyz +
            size_t(library.seg_offset[type]) * size_t(pair.library_stride);
        target.query_count = nq;
        target.library_count = nb;
        target.query_stride = pair.query_stride;
        target.library_stride = pair.library_stride;
        // Posed scratch per plane is [x|y|z][atom][4 starts] = 12 floats per
        // library atom. The YMM8/two-stream paths double this (see the
        // workspace sizing below).
        target.posed_offset = descriptor.posed_library_size;
        descriptor.posed_library_size += size_t(nb) * 12;
        target.coefficient = source.coefficient;
        target.exponent_scale = source.exponent_scale;
        target.torque_scale = source.torque_scale;
        target.translation_scale = source.translation_scale;

        // Query coordinates are already reduced (A' = sA): reduced bound.
        // Library coordinates are physical: physical bound. A miss clears
        // immutable_inputs_safe; it does not make the pair ineligible.
        for (int atom = 0; atom < nq; ++atom) {
            const float* xyz =
                target.query_xyz + size_t(atom) * size_t(target.query_stride);
            for (int component = 0; component < 3; ++component)
                descriptor.immutable_inputs_safe &=
                    safe_coordinate(xyz[component], maximum_reduced);
        }
        for (int atom = 0; atom < nb; ++atom) {
            const float* xyz = target.library_xyz +
                               size_t(atom) * size_t(target.library_stride);
            for (int component = 0; component < 3; ++component) {
                descriptor.immutable_inputs_safe &=
                    safe_coordinate(xyz[component], 1.0e9f);
#if R3_COLOR_ANALYTIC_ENVELOPE
                // Integer-bit maxima: exact under any denormal
                // mode, so a worker in another FP mode can trust them. See
                // doc/CPU_SHARED_POSE_CONTRACT.md.
                descriptor.library_max_abs[component] =
                    canonical_max_abs_finite(
                        descriptor.library_max_abs[component], xyz[component]);
#endif
            }
        }
    }
    descriptor.amplitude_amortized = amplitude_terms >= 8;
    return true;
}

// Two conformers may share one eight-lane evaluation only if every count,
// offset, stride, constant and the query pointer itself agree, so a single
// set of loop bounds and query broadcasts serves both. What may differ is
// the library coordinate data (pointers and envelope maxima). The signature
// covers all
// active planes, not shape atom count alone. Unsafe descriptors and
// field-backed descriptors never pair.
bool color_canonical_pair_compatible(
    const ColorCanonicalFourStartDescriptor& a,
    const ColorCanonicalFourStartDescriptor& b) {
    if (!a.immutable_inputs_safe || !b.immutable_inputs_safe ||
        a.field_bundle || b.field_bundle || a.n_features != b.n_features ||
        a.default_parameters != b.default_parameters ||
        a.amplitude_amortized != b.amplitude_amortized ||
        a.shape_count != b.shape_count || a.plane_count != b.plane_count ||
        a.posed_library_size != b.posed_library_size ||
        a.library_count != b.library_count ||
        a.library_offset != b.library_offset)
        return false;
    for (int i = 0; i < a.plane_count; ++i) {
        const auto& x = a.planes[size_t(i)];
        const auto& y = b.planes[size_t(i)];
        if (x.type != y.type || x.query_xyz != y.query_xyz ||
            x.query_count != y.query_count ||
            x.library_count != y.library_count ||
            x.query_stride != y.query_stride ||
            x.library_stride != y.library_stride ||
            x.posed_offset != y.posed_offset ||
            x.coefficient != y.coefficient ||
            x.exponent_scale != y.exponent_scale ||
            x.torque_scale != y.torque_scale ||
            x.translation_scale != y.translation_scale)
            return false;
    }
    return true;
}

// Sizes the scratch once per work unit so no helper resizes inside an
// iteration. A compatible pair stores both conformers' posed atoms
// (2 x posed_library_size); otherwise one buffer is reused by each singleton
// in turn, sized for the larger descriptor.
void prepare_color_canonical_four_start_workspace(
    const ColorCanonicalFourStartDescriptor* descriptors, int n_conformers,
    ColorCanonicalFourStartWorkspace& workspace) {
    if (descriptors == nullptr || n_conformers < 1 || n_conformers > 2)
        throw std::invalid_argument(
            "incomplete canonical four-start workspace preparation");
    size_t posed_library_size = 0;
    for (int pair = 0; pair < n_conformers; ++pair)
        posed_library_size = std::max(
            posed_library_size, descriptors[size_t(pair)].posed_library_size);
    workspace.compatible_pair = false;
#if (defined(__x86_64__) && R3_COLOR_COMPATIBLE_PAIRS) || \
    (defined(__aarch64__) && R3_COLOR_START_STREAMS == 2)
    workspace.compatible_pair =
        n_conformers == 2 &&
        color_canonical_pair_compatible(descriptors[0], descriptors[1]);
#endif
    workspace.posed_library_xyz.resize(posed_library_size *
                                       (workspace.compatible_pair ? 2 : 1));
#if R3_COLOR_START_HYBRID
    size_t maximum_samples = 0;
    for (int pair = 0; pair < n_conformers; ++pair) {
        const auto& descriptor = descriptors[size_t(pair)];
        if (!descriptor.field_bundle) continue;
        for (int pi = 0; pi < descriptor.plane_count; ++pi) {
            const auto& plane = descriptor.planes[size_t(pi)];
            if (descriptor.field_active[size_t(plane.type)])
                maximum_samples =
                    std::max(maximum_samples, size_t(plane.library_count) * 4);
        }
    }
#if !R3_COLOR_START_REDUCED_COORDS
    workspace.field_samples.points.reserve(maximum_samples * 3);
#endif
    workspace.field_samples.prepared.reserve(maximum_samples);
    workspace.field_samples.channels.reserve(maximum_samples);
#endif
}

// Shared body of the public batch entry and the bound execution view.
//   Bound      - the caller is a BoundCanonicalColorExecution whose
//                constructor already checked descriptors, plan identity and
//                workspace signature/capacity, so argument checks are elided.
//   BindPolicy - the exp mode and torque factoring come from template
//                arguments rather than runtime values, giving one concrete
//                instantiation per policy ("A2"; "A1" keeps the runtime
//                policy).
// Dispatch order: pair helper (YMM8 / two-stream NEON) when the workspace
// was prepared as a compatible pair; then, for any conformer the pair helper
// did not complete, the singleton XMM4/NEON4 helper; then the FP64 fallback.
template <bool Bound, bool BindPolicy = false,
          OptimizerExpMode BoundExponential = OptimizerExpMode::Baseline,
          bool BoundFactorTorque = false>
void evaluate_canonical_batch_impl(
    const ColorCanonicalFourStartDescriptor* descriptors, int n_conformers,
    const ColorInvariantPlan& plan, const PreparedReducedPose* const poses[8],
    bool gradient, ColorEvaluation output[8], uint64_t term_counts[8],
    ColorConformerCensus* census, ColorConformerPhaseTimes* phase_times,
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode, bool factor_torque,
    ColorSearchOutput search_output, bool vector_supported) {
    if constexpr (BindPolicy) {
        // A2 binds the immutable numerical policy to a concrete evaluator
        // instantiation. A1 retains its original runtime policy comparator.
        optimizer_exp_mode = BoundExponential;
        factor_torque = BoundFactorTorque;
    }
    if (!Bound &&
        (descriptors == nullptr || n_conformers < 1 || n_conformers > 2 ||
         poses == nullptr || output == nullptr || term_counts == nullptr))
        throw std::invalid_argument(
            "incomplete prepared canonical four-start batch");

    const int n_lanes = n_conformers * 4;
    for (int lane = 0; lane < n_lanes; ++lane) {
        output[lane] = ColorEvaluation{};
        term_counts[lane] = 0;
    }
    if (census) census->last_vector_support_mask = 0;

    for (int pair = 0; pair < n_conformers; ++pair) {
        for (int lane = 0; lane < 4; ++lane)
            if (!Bound && poses[pair * 4 + lane] == nullptr)
                throw std::invalid_argument(
                    "canonical four-start lane lacks a prepared pose");
    }

    // A pair mask of 0xff means both conformers' eight lanes were written.
    // Zero means nothing was published (the pair helpers validate both
    // conformers before writing), and each conformer is retried alone.
    const auto started = phase_times && phase_times->vector_kernel_seconds
                             ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
    bool any_vector_work = false;
    uint8_t pair_safe_mask = 0;
#if defined(__x86_64__)
    if (workspace.compatible_pair && vector_supported)
        pair_safe_mask = evaluate_color_canonical_four_start_ymm8(
            descriptors, factor_torque, poses, gradient, output, term_counts,
            workspace, optimizer_exp_mode, search_output);
#elif defined(__aarch64__) && R3_COLOR_START_STREAMS == 2
    if (workspace.compatible_pair)
        pair_safe_mask = evaluate_color_canonical_two_stream_neon4(
            descriptors, factor_torque, poses, gradient, output, term_counts,
            workspace, optimizer_exp_mode, search_output);
#endif
    for (int pair = 0; pair < n_conformers; ++pair) {
        const int lane_base = pair * 4;
        const auto* group_poses = poses + lane_base;
        uint8_t safe_mask = 0;
#if defined(__aarch64__)
        if (pair_safe_mask == 0xff)
            safe_mask = 0x0f;
        else
            safe_mask = evaluate_color_canonical_four_start_neon4(
                descriptors[size_t(pair)], factor_torque, group_poses, gradient,
                output + lane_base, term_counts + lane_base, workspace,
                optimizer_exp_mode, search_output);
#elif defined(__x86_64__)
        if (pair_safe_mask == 0xff)
            safe_mask = 0x0f;
        else if (vector_supported)
            safe_mask = evaluate_color_canonical_four_start_xmm4(
                descriptors[size_t(pair)], factor_torque, group_poses, gradient,
                output + lane_base, term_counts + lane_base, workspace,
                optimizer_exp_mode, search_output);
#endif
        // 0x0f: all four starts published by a vector helper. Any other
        // value (helpers return all or nothing) falls through to FP64.
        if (safe_mask == 0x0f) {
            if (census) {
                if (pair_safe_mask == 0xff) {
                    if (pair == 0) {
#if defined(__aarch64__)
                        ++census->compatible_neon4_two_stream_evaluations;
#else
                        ++census->compatible_ymm8_evaluations;
#endif
                    }
                } else {
                    ++census->singleton_four_start_evaluations;
                }
            }
            any_vector_work |= descriptors[size_t(pair)].plane_count != 0;
            if (census) {
                for (int lane = 0; lane < 4; ++lane) {
                    census->last_vector_support_mask |=
                        uint8_t(1u << unsigned(lane_base + lane));
                    ++census->vector_eligible_lane_calls[size_t(lane_base +
                                                                lane)];
                    for (int type = 1;
                         type < descriptors[size_t(pair)].n_features; ++type) {
                        // 19 bins per type: counts 0..17, then 18+.
                        const int count = descriptors[size_t(pair)]
                                              .library_count[size_t(type)];
                        ++census
                              ->target_count_bins[size_t(type) * 19 +
                                                  size_t(std::min(count, 18))];
                    }
                }
                for (int plane = 0;
                     plane < descriptors[size_t(pair)].plane_count; ++plane) {
                    const auto& p =
                        descriptors[size_t(pair)].planes[size_t(plane)];
                    // Exact counts, no padding: useful == issued.
                    const uint64_t useful =
                        uint64_t(p.query_count) * uint64_t(p.library_count) * 4;
                    const size_t cell =
                        size_t(p.type) * size_t(MAX_COLOR_FEATURES) +
                        size_t(p.type);
                    census->useful_terms_by_interaction[cell] += useful;
                    census->issued_lanes_by_interaction[cell] += useful;
                }
            }
            continue;
        }

        // A pose or immutable input outside the common FP32 envelope sends
        // this complete four-start group through the established FP64 formula.
        for (int lane = 0; lane < 4; ++lane) {
            const int output_lane = lane_base + lane;
            output[output_lane] = exact_canonical_four_start_lane(
                descriptors[size_t(pair)], *poses[output_lane], plan,
                gradient && search_output != ColorSearchOutput::ValueOnly,
                &term_counts[output_lane]);
            if (census) {
                ++census->scalar_fallback_pairs;
                ++census->scalar_fallback_lane_calls[size_t(output_lane)];
                for (int plane = 0;
                     plane < descriptors[size_t(pair)].plane_count; ++plane) {
                    const auto& p =
                        descriptors[size_t(pair)].planes[size_t(plane)];
                    const size_t cell =
                        size_t(p.type) * size_t(MAX_COLOR_FEATURES) +
                        size_t(p.type);
                    census->useful_terms_by_interaction[cell] +=
                        uint64_t(p.query_count) * uint64_t(p.library_count);
                }
                for (int type = 1; type < descriptors[size_t(pair)].n_features;
                     ++type) {
                    const int count =
                        descriptors[size_t(pair)].library_count[size_t(type)];
                    ++census->target_count_bins[size_t(type) * 19 +
                                                size_t(std::min(count, 18))];
                }
            }
        }
    }
    if (census && any_vector_work) ++census->vector_evaluations;
    if (phase_times && phase_times->vector_kernel_seconds)
        *phase_times->vector_kernel_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started)
                .count();
}

// Checked entry: validates its arguments on every call and reads the ISA
// capability from the plan. The optimizer uses it only when built with
// R3_BOUND_EXECUTION=0; the default (2) uses evaluate_bound_policy.
void evaluate_color_canonical_four_start_batch(
    const ColorCanonicalFourStartDescriptor* descriptors, int n_conformers,
    const ColorInvariantPlan& plan, const PreparedReducedPose* const poses[8],
    bool gradient, ColorEvaluation output[8], uint64_t term_counts[8],
    ColorConformerCensus* census, ColorConformerPhaseTimes* phase_times,
    ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode optimizer_exp_mode, bool factor_torque,
    ColorSearchOutput search_output) {
    evaluate_canonical_batch_impl<false>(
        descriptors, n_conformers, plan, poses, gradient, output, term_counts,
        census, phase_times, workspace, optimizer_exp_mode, factor_torque,
        search_output, plan.avx2_fma_available());
}

// Binds the immutable execution contract once per decoded work unit:
// plan identity, compatible-pair signature and workspace capacity are checked
// here, and the exp-mode/factoring policy is resolved to one of six concrete
// instantiations. A workspace prepared for another signature is rejected
// rather than silently reused.
BoundCanonicalColorExecution::BoundCanonicalColorExecution(
    const ColorCanonicalFourStartDescriptor* descriptors, int n_conformers,
    const ColorInvariantPlan& plan, ColorCanonicalFourStartWorkspace& workspace,
    OptimizerExpMode exponential, bool factor_torque)
    : descriptors_(descriptors), n_conformers_(n_conformers), plan_(&plan),
      workspace_(&workspace), exponential_(exponential),
      factor_torque_(factor_torque),
      vector_supported_(plan.avx2_fma_available()) {
    if (!descriptors || n_conformers < 1 || n_conformers > 2)
        throw std::invalid_argument("invalid bound canonical color work unit");
    size_t required = 0;
    for (int pair = 0; pair < n_conformers; ++pair) {
        if (descriptors[pair].query_plan != &plan)
            throw std::invalid_argument(
                "bound canonical color query plan identity mismatch");
        required = std::max(required, descriptors[pair].posed_library_size);
    }
    bool compatible = false;
#if (defined(__x86_64__) && R3_COLOR_COMPATIBLE_PAIRS) || \
    (defined(__aarch64__) && R3_COLOR_START_STREAMS == 2)
    compatible = n_conformers == 2 && color_canonical_pair_compatible(
                                          descriptors[0], descriptors[1]);
#endif
    if (workspace.compatible_pair != compatible ||
        required > workspace.posed_library_xyz.size() / (compatible ? 2 : 1))
        throw std::invalid_argument(
            "bound canonical color workspace signature or capacity mismatch");
    switch (exponential) {
    case OptimizerExpMode::Baseline:
        evaluate_ = factor_torque
                        ? &execute_bound<OptimizerExpMode::Baseline, true>
                        : &execute_bound<OptimizerExpMode::Baseline, false>;
        break;
    case OptimizerExpMode::RawBits:
        evaluate_ = factor_torque
                        ? &execute_bound<OptimizerExpMode::RawBits, true>
                        : &execute_bound<OptimizerExpMode::RawBits, false>;
        break;
    case OptimizerExpMode::RawBitsUnclamped:
        evaluate_ =
            factor_torque
                ? &execute_bound<OptimizerExpMode::RawBitsUnclamped, true>
                : &execute_bound<OptimizerExpMode::RawBitsUnclamped, false>;
        break;
    default:
        throw std::invalid_argument(
            "unknown bound canonical SEARCH exponential policy");
    }
}

// Bound route with the runtime policy ("A1"). Always requests gradients:
// value-only SEARCH is selected through `search_output`, never by passing
// gradient=false, which would select a different exponential.
void BoundCanonicalColorExecution::evaluate(
    const PreparedReducedPose* const poses[8], ColorEvaluation output[8],
    uint64_t term_counts[8], ColorConformerCensus* census,
    ColorConformerPhaseTimes* phase_times,
    ColorSearchOutput search_output) const {
    evaluate_canonical_batch_impl<true>(
        descriptors_, n_conformers_, *plan_, poses, true, output, term_counts,
        census, phase_times, *workspace_, exponential_, factor_torque_,
        search_output, vector_supported_);
}

template <OptimizerExpMode Exponential, bool FactorTorque>
void BoundCanonicalColorExecution::execute_bound(
    const BoundCanonicalColorExecution& execution,
    const PreparedReducedPose* const poses[8], ColorEvaluation output[8],
    uint64_t term_counts[8], ColorConformerCensus* census,
    ColorConformerPhaseTimes* phase_times, ColorSearchOutput search_output) {
    evaluate_canonical_batch_impl<true, true, Exponential, FactorTorque>(
        execution.descriptors_, execution.n_conformers_, *execution.plan_,
        poses, true, output, term_counts, census, phase_times,
        *execution.workspace_, Exponential, FactorTorque, search_output,
        execution.vector_supported_);
}

// Bound route through the policy-specialized instantiation chosen in the
// constructor ("A2"); the optimizer default (R3_BOUND_EXECUTION=2).
void BoundCanonicalColorExecution::evaluate_bound_policy(
    const PreparedReducedPose* const poses[8], ColorEvaluation output[8],
    uint64_t term_counts[8], ColorConformerCensus* census,
    ColorConformerPhaseTimes* phase_times,
    ColorSearchOutput search_output) const {
    evaluate_(*this, poses, output, term_counts, census, phase_times,
              search_output);
}

}  // namespace r3::cpu

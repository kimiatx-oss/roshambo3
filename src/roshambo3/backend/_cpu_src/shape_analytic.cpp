// Fixed-pose exact and SIMD analytic shape evaluators. This TU keeps the
// grain instantiations out of the shared combo optimizer's dispatch body.
//
// Analytic shape overlap at a supplied pose: for reduced query atoms `a_i`
// and posed library atoms `M b_j + t'`, S_AB = sum_ij exp2(-|a_i - M b_j -
// t'|^2) in the reduced frame (NUMERICS_POLICY section 4), plus the descent gradient in
// `get_gradient(gnorm=...)` coordinates -- [1:4] space-frame half-angle
// torque, [4:7] reduced translation. The optimizer, not the evaluator, owns
// the pose update and the Tanimoto weighting (shape_evaluator.hpp).
//
// Three implementations, all from the atom and conformer grains:
//   scalar      `get_gradient`, libm exp2, source-order sum: the exact path;
//   atom grain  AVX2 width 8 over query atoms or library atoms, minimax_d5
//               exp2, one horizontal reduction per pose;
//   trajectory  AVX2 width 8, one independent POSE per lane, no cross-lane
//               reduction, so each lane's sum is in source atom order.
// The SIMD arms change the exponential and (atom grain) the summation order:
// a CPU score is a function of the vector width and is
// compared on ranked output, so each arm names its reduction and exponential
// in its stamp.
// AVX2 only: on other targets the factory refuses the SIMD backends.
#include "shape_evaluator.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "grain.hpp"
#include "grain_conformer.hpp"
#include "score.hpp"
#include "simd.hpp"

namespace r3 {
namespace cpu {
namespace {

const char* grain_name(ShapeGrain grain) {
    switch (grain) {
    case ShapeGrain::Scalar:
        return "scalar";
    case ShapeGrain::QueryAtom:
        return "query_atom";
    case ShapeGrain::LibraryAtom:
        return "library_atom";
    case ShapeGrain::Trajectory:
        return "trajectory";
    }
    return "invalid";
}

// Census mapping: `useful_terms` = real atom pairs, `issued_terms` = exp2
// evaluations actually executed (padding lanes included).
ShapeEvaluationCounts counts_from(const OpCensus& census) {
    ShapeEvaluationCounts counts;
    counts.useful_terms = census.pair_terms_optimizer;
    counts.issued_terms = census.exp2_calls;
    return counts;
}

// The legacy exact combo shape path (`ScalarAnalytic`, grain scalar, width 1).
class ScalarAnalyticShapeEvaluator final : public PreparedShapeEvaluator {
public:
    ScalarAnalyticShapeEvaluator(const ShapeEvaluatorConfig& config,
                                 const Conformer& query,
                                 ShapeQueryMetadata metadata)
        : PreparedShapeEvaluator(config, query, metadata) {}

    ShapeEvaluatorStamp stamp() const override {
        return {"scalar_analytic", "scalar", "scalar", "source_order",
                "libm_exp2",       1,        false,    false};
    }

    void evaluate_one(const ShapePoseRequest& request,
                      ShapeEvaluation& output) const override {
        validate_shape_pose_request(request);
        output = ShapeEvaluation{};
        OpCensus census;
        get_gradient(query_reduced(), *request.library, request.matrix,
                     request.translation, request.gnorm, output.descent,
                     &output.s_ab, &census);
        output.counts = counts_from(census);
    }
};

#if defined(__AVX2__)

// The atom grains behind the fixed-pose seam. QueryAtom vectorizes over
// query atoms and pre-builds the query SoA once at preparation; LibraryAtom
// vectorizes over library atoms and needs the target's SoA, which the caller
// prepares per target (`prepare_analytic_shape_library_workspace`) and
// passes in the request workspace.
class AtomGrainShapeEvaluator final : public PreparedShapeEvaluator {
public:
    AtomGrainShapeEvaluator(const ShapeEvaluatorConfig& config,
                            const Conformer& query, ShapeQueryMetadata metadata)
        : PreparedShapeEvaluator(config, query, metadata),
          grain_(config.grain) {
        if (grain_ == ShapeGrain::QueryAtom)
            query_soa_.build(query_reduced(), 8);
    }

    ShapeEvaluatorStamp stamp() const override {
        const bool query = grain_ == ShapeGrain::QueryAtom;
        return {"analytic_simd",
                grain_name(grain_),
                "AVX2",
                "avx2_lane_tree_width8",
                "minimax_d5",
                8,
                true,
                !query};
    }

    void evaluate_one(const ShapePoseRequest& request,
                      ShapeEvaluation& output) const override {
        validate_shape_pose_request(request);
        output = ShapeEvaluation{};
        OpCensus census;
        if (grain_ == ShapeGrain::QueryAtom) {
            get_gradient_query_grain<simd::Avx2Tag, true>(
                simd::Avx2Tag{}, query_soa_, *request.library, request.matrix,
                request.translation, request.gnorm, output.descent,
                &output.s_ab, &census);
        } else {
            // The workspace must belong to THIS target; a stale SoA from
            // another library conformer would score the wrong atoms.
            if (request.library_workspace == nullptr ||
                !request.library_workspace->matches(*request.library) ||
                !request.library_workspace->analytic_atom)
                throw std::invalid_argument(
                    "library-atom shape evaluation needs a matching prepared workspace");
            const SoaConformer& b = *request.library_workspace->analytic_atom;
            if (b.n_atoms != request.library->n_atoms)
                throw std::invalid_argument(
                    "prepared workspace atom count does not match target");
            get_gradient_library_grain<simd::Avx2Tag, true>(
                simd::Avx2Tag{}, query_reduced(), b, request.matrix,
                request.translation, request.gnorm, output.descent,
                &output.s_ab, &census);
        }
        output.counts = counts_from(census);
    }

private:
    ShapeGrain grain_;
    SoaConformer query_soa_;
};

// Up to eight library targets transposed into one `LaneBlock` ([atom][lane],
// zero-weight padding for ragged atom counts and empty lanes; see
// grain_conformer.hpp), prepared once per color batch and reused for every
// iteration and start. `source_*` record each lane's target so a request
// that arrives in a different order is refused rather than mis-scored.
struct TrajectoryBatch final : PreparedShapeBatch {
    LaneBlock block;
    std::array<const float*, 8> source_xyz{};
    std::array<int, 8> source_atoms{};
    size_t count = 0;
};

// The trajectory grain: eight poses evaluated together, lane = one
// trajectory (its own target, matrix, translation and gnorm). Lanes never
// mix, so there is no horizontal reduction; this is the fixed-pose form of
// the conformer grain (`optimize_block_conformer`, grain_conformer.hpp).
// It has no single-pose path.
class TrajectoryShapeEvaluator final : public PreparedShapeEvaluator {
public:
    TrajectoryShapeEvaluator(const ShapeEvaluatorConfig& config,
                             const Conformer& query,
                             ShapeQueryMetadata metadata)
        : PreparedShapeEvaluator(config, query, metadata) {}

    ShapeEvaluatorStamp stamp() const override {
        return {"analytic_simd",
                "trajectory",
                "AVX2",
                "independent_trajectory_lanes_no_cross_lane_reduction",
                "minimax_d5",
                8,
                true,
                false};
    }

    void evaluate_one(const ShapePoseRequest&,
                      ShapeEvaluation&) const override {
        throw std::invalid_argument(
            "trajectory shape grain requires a prepared multi-pose batch");
    }

    PreparedShapeBatchHandle prepare_batch(const Conformer* const* libraries,
                                           size_t count) const override {
        if (count == 0 || count > 8 || libraries == nullptr)
            throw std::invalid_argument(
                "AVX2 trajectory shape batch needs 1 through 8 targets");
        auto batch = std::make_shared<TrajectoryBatch>();
        batch->count = count;
        std::array<Conformer, 8> conformers{};
        std::array<float, 8> self_bb{};
        for (size_t lane = 0; lane < count; ++lane) {
            if (libraries[lane] == nullptr || (libraries[lane]->n_atoms > 0 &&
                                               libraries[lane]->xyz == nullptr))
                throw std::invalid_argument(
                    "AVX2 trajectory shape batch has an invalid target");
            conformers[lane] = *libraries[lane];
            // LaneBlock carries an S_BB slot, but this evaluator never reads
            // it: normalization arrives per request as `gnorm`.
            self_bb[lane] = 1.0f;
            batch->source_xyz[lane] = libraries[lane]->xyz;
            batch->source_atoms[lane] = libraries[lane]->n_atoms;
        }
        batch->block.build(conformers.data(), self_bb.data(), int(count), 8);
        return batch;
    }

    void evaluate_batch(const ShapePoseRequest* requests, size_t count,
                        const PreparedShapeBatch* prepared,
                        ShapeEvaluation* outputs) const override {
        if (count == 0 || count > 8 || requests == nullptr ||
            outputs == nullptr || prepared == nullptr)
            throw std::invalid_argument(
                "AVX2 trajectory shape evaluation needs a prepared batch");
        const auto* batch = dynamic_cast<const TrajectoryBatch*>(prepared);
        if (batch == nullptr || batch->count != count)
            throw std::invalid_argument(
                "trajectory shape batch does not match its preparation");
        for (size_t lane = 0; lane < count; ++lane) {
            const ShapePoseRequest& request = requests[lane];
            validate_shape_pose_request(request);
            if (request.library->xyz != batch->source_xyz[lane] ||
                request.library->n_atoms != batch->source_atoms[lane])
                throw std::invalid_argument(
                    "trajectory shape request does not match prepared target order");
        }
        evaluate_trajectory_batch(*batch, requests, count, outputs);
    }

private:
    void evaluate_trajectory_batch(const TrajectoryBatch& batch,
                                   const ShapePoseRequest* requests,
                                   size_t count,
                                   ShapeEvaluation* outputs) const {
        const simd::Avx2Tag d{};
        const int width = simd::Lanes(d);
        const Conformer& a = query_reduced();
        // Transpose the per-lane poses into broadcast-by-lane vectors: m[k]
        // holds matrix element k of every lane's pose. Unused lanes get a
        // zero matrix and zero gnorm; their LaneBlock weights are zero too,
        // and their outputs are never written.
        alignas(32) float lane_values[8]{};
        simd::VecF32x8 m[9];
        for (int k = 0; k < 9; ++k) {
            for (size_t lane = 0; lane < count; ++lane)
                lane_values[lane] = requests[lane].matrix[k];
            for (size_t lane = count; lane < 8; ++lane)
                lane_values[lane] = 0.0f;
            m[k] = simd::Load(d, lane_values);
        }
        simd::VecF32x8 t[3];
        for (int k = 0; k < 3; ++k) {
            for (size_t lane = 0; lane < count; ++lane)
                lane_values[lane] = requests[lane].translation[k];
            for (size_t lane = count; lane < 8; ++lane)
                lane_values[lane] = 0.0f;
            t[k] = simd::Load(d, lane_values);
        }
        for (size_t lane = 0; lane < count; ++lane)
            lane_values[lane] = requests[lane].gnorm;
        for (size_t lane = count; lane < 8; ++lane)
            lane_values[lane] = 0.0f;
        // d/dx exp2(-x) = -ln2 exp2(-x), and d|delta|^2 brings a 2: the
        // per-lane gradient factor is -2 ln2 * gnorm (MINUS_2LN2, kernel.hpp).
        const simd::VecF32x8 vtfac =
            simd::Mul(simd::Set(d, MINUS_2LN2), simd::Load(d, lane_values));

        simd::VecF32x8 g1 = simd::Zero(d), g2 = simd::Zero(d),
                       g3 = simd::Zero(d), g4 = simd::Zero(d),
                       g5 = simd::Zero(d), g6 = simd::Zero(d),
                       vs = simd::Zero(d);
        const LaneBlock& block = batch.block;
        // Library atom j of all lanes is one aligned load per component. The
        // loop body is the same expression tree, in the same order, as the
        // conformer grain's inner loop, without its cutoff test.
        for (int j = 0; j < block.max_atoms; ++j) {
            const size_t offset = size_t(j) * size_t(width);
            const simd::VecF32x8 bx = simd::Load(d, &block.x[offset]);
            const simd::VecF32x8 by = simd::Load(d, &block.y[offset]);
            const simd::VecF32x8 bz = simd::Load(d, &block.z[offset]);
            const simd::VecF32x8 bw = simd::Load(d, &block.w[offset]);
            const simd::VecF32x8 mbx =
                simd::Add(simd::Add(simd::Mul(m[0], bx), simd::Mul(m[1], by)),
                          simd::Mul(m[2], bz));
            const simd::VecF32x8 mby =
                simd::Add(simd::Add(simd::Mul(m[3], bx), simd::Mul(m[4], by)),
                          simd::Mul(m[5], bz));
            const simd::VecF32x8 mbz =
                simd::Add(simd::Add(simd::Mul(m[6], bx), simd::Mul(m[7], by)),
                          simd::Mul(m[8], bz));
            // C = 2 M b, the space-frame lever arm of kernel.hpp's gradient
            // definition (`cx = 2 mbx` in score.hpp and grain.hpp).
            const simd::VecF32x8 cx = simd::Add(mbx, mbx);
            const simd::VecF32x8 cy = simd::Add(mby, mby);
            const simd::VecF32x8 cz = simd::Add(mbz, mbz);
            for (int i = 0; i < a.n_atoms; ++i) {
                const simd::VecF32x8 dx =
                    simd::Sub(simd::Sub(simd::Set(d, a.xyz[3 * i]), t[0]), mbx);
                const simd::VecF32x8 dy = simd::Sub(
                    simd::Sub(simd::Set(d, a.xyz[3 * i + 1]), t[1]), mby);
                const simd::VecF32x8 dz = simd::Sub(
                    simd::Sub(simd::Set(d, a.xyz[3 * i + 2]), t[2]), mbz);
                const simd::VecF32x8 d2 =
                    simd::Add(simd::Add(simd::Mul(dx, dx), simd::Mul(dy, dy)),
                              simd::Mul(dz, dz));
                // kij = exp2(-|delta|^2) * weight; weight 0 zeroes padding.
                const simd::VecF32x8 kij = simd::Mul(
                    Exp2Grade<true>::apply(simd::Sub(simd::Zero(d), d2)), bw);
                const simd::VecF32x8 temp = simd::Mul(vtfac, kij);
                // g1..g3: torque (lever x delta); g4..g6: translation.
                g1 = simd::Add(g1,
                               simd::Mul(temp, simd::Sub(simd::Mul(cy, dz),
                                                         simd::Mul(cz, dy))));
                g2 = simd::Add(g2,
                               simd::Mul(temp, simd::Sub(simd::Mul(cz, dx),
                                                         simd::Mul(cx, dz))));
                g3 = simd::Add(g3,
                               simd::Mul(temp, simd::Sub(simd::Mul(cx, dy),
                                                         simd::Mul(cy, dx))));
                g4 = simd::Add(g4, simd::Mul(temp, dx));
                g5 = simd::Add(g5, simd::Mul(temp, dy));
                g6 = simd::Add(g6, simd::Mul(temp, dz));
                vs = simd::Add(vs, kij);
            }
        }

        // Unpack lanes to per-pose outputs; descent[0] stays zero.
        alignas(32) float reduced[8]{};
        simd::Store(vs, d, reduced);
        for (size_t lane = 0; lane < count; ++lane) {
            outputs[lane] = ShapeEvaluation{};
            outputs[lane].s_ab = reduced[lane];
        }
        const simd::VecF32x8 gradients[6] = {g1, g2, g3, g4, g5, g6};
        for (int component = 0; component < 6; ++component) {
            simd::Store(gradients[component], d, reduced);
            for (size_t lane = 0; lane < count; ++lane)
                outputs[lane].descent[component + 1] = reduced[lane];
        }
        for (size_t lane = 0; lane < count; ++lane)
            outputs[lane].counts.useful_terms =
                uint64_t(a.n_atoms) * uint64_t(requests[lane].library->n_atoms);
        // Credit the complete issued SIMD batch once, so summing the per-pose
        // census includes padded lanes and ragged library atom tails exactly.
        outputs[0].counts.issued_terms =
            uint64_t(a.n_atoms) * uint64_t(block.max_atoms) * uint64_t(width);
    }
};

#endif  // __AVX2__

}  // namespace

// Factory for the analytic backends. Refuses, rather than substitutes, any
// grain/width it does not implement: scalar must be grain scalar, width 1;
// SIMD must be width 8 on an AVX2 build.
PreparedShapeEvaluatorHandle
make_analytic_shape_evaluator(const ShapeEvaluatorConfig& config,
                              const Conformer& query_reduced,
                              ShapeQueryMetadata metadata) {
    if (config.backend == ShapeBackend::ScalarAnalytic) {
        if (config.grain != ShapeGrain::Scalar || config.width != 1)
            throw std::invalid_argument(
                "scalar analytic shape requires grain=scalar and width=1");
        return std::make_shared<ScalarAnalyticShapeEvaluator>(
            config, query_reduced, metadata);
    }
    if (config.backend != ShapeBackend::AnalyticSimd)
        throw std::invalid_argument(
            "analytic evaluator factory requires an analytic backend");
#if defined(__AVX2__)
    if (config.width != 8)
        throw std::invalid_argument(
            "analytic SIMD shape currently supports AVX2 width 8 only");
    if (config.grain == ShapeGrain::QueryAtom ||
        config.grain == ShapeGrain::LibraryAtom)
        return std::make_shared<AtomGrainShapeEvaluator>(config, query_reduced,
                                                         metadata);
    if (config.grain == ShapeGrain::Trajectory)
        return std::make_shared<TrajectoryShapeEvaluator>(config, query_reduced,
                                                          metadata);
#else
    (void)query_reduced;
    (void)metadata;
#endif
    throw std::invalid_argument(
        "unsupported analytic shape grain or unavailable AVX2 target");
}

// Per-target preparation. Only the library-atom grain needs anything: an SoA
// copy padded to whole 8-lane vectors (x, y, z, weight). Every grain still
// records the source identity so evaluators can verify the workspace.
ShapeLibraryWorkspace
prepare_analytic_shape_library_workspace(const Conformer& library,
                                         ShapeGrain grain, int width) {
    if (library.n_atoms < 0 || (library.n_atoms > 0 && library.xyz == nullptr))
        throw std::invalid_argument("invalid shape library conformer");
    ShapeLibraryWorkspace workspace;
    workspace.source_xyz = library.xyz;
    workspace.source_atoms = library.n_atoms;
    if (grain == ShapeGrain::LibraryAtom) {
#if defined(__AVX2__)
        if (width != 8)
            throw std::invalid_argument(
                "library-atom analytic workspace requires width 8");
        auto soa = std::make_shared<SoaConformer>();
        soa->build(library, width);
        workspace.prepared_bytes = uint64_t(soa->n_padded) * 4 * sizeof(float);
        workspace.analytic_atom = std::move(soa);
#else
        throw std::invalid_argument(
            "library-atom analytic workspace requires AVX2");
#endif
    }
    return workspace;
}

}  // namespace cpu
}  // namespace r3

// Native correctness probe for the query-prepared field adapters.
// Build: g++ -O2 -march=x86-64-v3 -std=c++17 \
//   -I src/roshambo3/backend/_cpu_src \
//   src/roshambo3/backend/_cpu_src/shape_evaluator.cpp \
//   src/roshambo3/backend/_cpu_src/shape_field_evaluator.cpp \
//   tests/shape_field_evaluator.cpp -o "$TMPDIR/shape-field" && "$TMPDIR/shape-field"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "score.hpp"
#include "field.hpp"
#include "field_avx2.hpp"
#if defined(__aarch64__)
#include "field_neon.hpp"
#endif
#include "shape_evaluator.hpp"
#include "shape_combo_api.hpp"
#include "shape_field_evaluator.hpp"

using namespace r3::cpu;

namespace {

bool close(float x, float y, float abs_tol = 4e-5f,
           float rel_tol = 4e-5f) {
    return std::isfinite(x) && std::isfinite(y) &&
           std::fabs(x - y) <= abs_tol + rel_tol * std::fabs(y);
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

uint64_t field_control() {
#if defined(__aarch64__)
    uint64_t value; asm volatile("mrs %0, fpcr" : "=r"(value)); return value;
#else
    return _mm_getcsr();
#endif
}
void field_control(uint64_t value) {
#if defined(__aarch64__)
    asm volatile("msr fpcr, %0\n\tisb" :: "r"(value) : "memory");
#else
    _mm_setcsr(value);
#endif
}

void compare_bound_field_modes(const PreparedShapeEvaluator& evaluator,
                               const ShapePoseRequest& request) {
    const auto original = field_control();
    struct Restore { uint64_t value; ~Restore() { field_control(value); } } restore{original};
    PreparedReducedPose pose;
    std::copy(request.matrix, request.matrix + 9, pose.matrix);
    std::copy(request.translation, request.translation + 3, pose.translation);
    const auto values = evaluator.bind_execution(*request.library, request.library_workspace, true);
#if defined(__aarch64__)
    constexpr int flush_modes = 2;
#else
    constexpr int flush_modes = 4;
#endif
    for (int rounding = 0; rounding < 4; ++rounding)
        for (int flush = 0; flush < flush_modes; ++flush) {
#if defined(__aarch64__)
            field_control((original & ~((uint64_t(3)<<22)|(uint64_t(1)<<24))) |
                (uint64_t(rounding)<<22) | (uint64_t(flush)<<24));
#else
            field_control((original & ~((3u<<13)|(1u<<6)|(1u<<15))) | (rounding<<13) |
                ((flush&1)<<6) | ((flush>>1)<<15));
#endif
            ShapeEvaluation checked{}, value_output{};
            evaluator.evaluate_one(request, checked);
            values.evaluate(pose, value_output);
            require(std::memcmp(&checked.s_ab, &value_output.s_ab, sizeof(float)) == 0,
                    "bound field values must preserve full evaluator bytes in every FP mode");
        }
}

ShapeEvaluatorConfig config(ShapeBackend backend, ShapeGrain grain, int width) {
    ShapeEvaluatorConfig out;
    out.backend = backend;
    out.grain = grain;
    out.width = width;
    out.field_spacing_angstrom = 0.5f;
    out.field_padding_angstrom = 8.0f;
    return out;
}

// Independent double-precision trilinear oracle over the stored FP32 [f,gx,gy,gz]
// nodes. It does not call sample_field or any production gradient routine.
void direct_stored_sample(const FieldNaive32& field, float x, float y, float z,
                          float out[4]) {
    const FieldGeometry& g = field.g;
    const double tx = (double(x) - g.ox) * g.inv_hs;
    const double ty = (double(y) - g.oy) * g.inv_hs;
    const double tz = (double(z) - g.oz) * g.inv_hs;
    const int ix = int(std::floor(tx)), iy = int(std::floor(ty)),
              iz = int(std::floor(tz));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= g.nx || iy >= g.ny || iz >= g.nz) {
        for (int c = 0; c < 4; ++c) out[c] = 0.0f;
        return;
    }
    const double fx = tx - ix, fy = ty - iy, fz = tz - iz;
    double accum[4] = {};
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const double wx = dx ? fx : 1.0 - fx;
                const double wy = dy ? fy : 1.0 - fy;
                const double wz = dz ? fz : 1.0 - fz;
                const double weight = wx * wy * wz;
                const float* node = field.node(ix + dx, iy + dy, iz + dz);
                for (int c = 0; c < 4; ++c)
                    accum[c] += weight * double(node[c]);
            }
    for (int c = 0; c < 4; ++c) out[c] = float(accum[c]);
}

void direct_stored_gradient(const FieldNaive32& field, const Conformer& library,
                            const float matrix[9], const float translation[3],
                            float gnorm, ShapeEvaluation& out) {
    out = {};
    const float gfac = -gnorm;
    for (int j = 0; j < library.n_atoms; ++j) {
        const float bx = library.xyz[3*j];
        const float by = library.xyz[3*j+1];
        const float bz = library.xyz[3*j+2];
        const float mbx = matrix[0]*bx + matrix[1]*by + matrix[2]*bz;
        const float mby = matrix[3]*bx + matrix[4]*by + matrix[5]*bz;
        const float mbz = matrix[6]*bx + matrix[7]*by + matrix[8]*bz;
        float channels[4];
        direct_stored_sample(field, mbx + translation[0], mby + translation[1],
                             mbz + translation[2], channels);
        const float value = channels[0], fx = channels[1];
        const float fy = channels[2], fz = channels[3];
        const float gx = gfac * fx, gy = gfac * fy, gz = gfac * fz;
        const float cx = 2.0f*mbx, cy = 2.0f*mby, cz = 2.0f*mbz;
        out.s_ab += value;
        out.descent[1] += cy*gz - cz*gy;
        out.descent[2] += cz*gx - cx*gz;
        out.descent[3] += cx*gy - cy*gx;
        out.descent[4] += gx;
        out.descent[5] += gy;
        out.descent[6] += gz;
    }
    out.counts.field_samples = uint64_t(library.n_atoms);
}

void compare(const ShapeEvaluation& got, const ShapeEvaluation& expected,
             float tolerance, const char* arm) {
    if (!close(got.s_ab, expected.s_ab, tolerance, tolerance)) {
        std::fprintf(stderr, "FAIL %s overlap got=%.9g expected=%.9g\n",
                     arm, got.s_ab, expected.s_ab);
        throw std::runtime_error("field overlap mismatch");
    }
    require(got.descent[0] == 0.0f, "shape descent slot 0 must remain zero");
    for (int k = 1; k < 7; ++k) {
        if (!close(got.descent[k], expected.descent[k], tolerance, tolerance)) {
            std::fprintf(stderr,
                         "FAIL %s descent[%d] got=%.9g expected=%.9g\n",
                         arm, k, got.descent[k], expected.descent[k]);
            throw std::runtime_error("field stored-gradient mismatch");
        }
    }
    require(got.counts.fallback_evaluations == 0 &&
            got.counts.fallback_terms == 0,
            "field adapter must not hide an exact fallback");
}

void compare_batch_to_one(const PreparedShapeEvaluator& evaluator,
                          const std::vector<ShapePoseRequest>& requests) {
    std::vector<ShapeEvaluation> batched(requests.size());
    evaluator.evaluate_batch(requests.data(), requests.size(), batched.data());
    for (size_t i = 0; i < requests.size(); ++i) {
        ShapeEvaluation scalar{};
        evaluator.evaluate_one(requests[i], scalar);
        compare(batched[i], scalar, 2e-6f, "batch-vs-one");
        compare_bound_field_modes(evaluator, requests[i]);
        PreparedReducedPose pose;
        std::copy(requests[i].matrix, requests[i].matrix + 9, pose.matrix);
        std::copy(requests[i].translation, requests[i].translation + 3, pose.translation);
        const auto values = evaluator.bind_execution(
            *requests[i].library, requests[i].library_workspace, true);
        ShapeEvaluation value_output;
        values.evaluate(pose, value_output);
        require(std::memcmp(&scalar.s_ab, &value_output.s_ab, sizeof(float)) == 0,
                "every prepared request tail retains exact field value bytes");
    }
}

#if defined(__aarch64__)
uint64_t read_fpcr() {
    uint64_t value = 0;
    asm volatile("mrs %0, fpcr" : "=r"(value));
    return value;
}

void write_fpcr(uint64_t value) {
    asm volatile("msr fpcr, %0\n\tisb" :: "r"(value) : "memory");
}

struct RestoreFpcr {
    uint64_t original = read_fpcr();
    ~RestoreFpcr() { write_fpcr(original); }
};

void run_mixed_field(const FieldNaive16& field,
                     const std::vector<float>& physical_xyz,
                     const float matrix[9], const float translation[3],
                     float gradient[7], float& score) {
    const Conformer library{physical_xyz.data(),
                            int(physical_xyz.size() / size_t(3))};
    ConformerSoA soa;
    soa.build(library);
    field_gradient_neon_fp16_mixed_yz_pipelined(
        field, soa, matrix, translation, 1.0f, gradient, &score);
}

void test_mixed_half_edge_cases() {
    if (!field_neon_fp16_arithmetic_supported()) return;

    FieldGeometry geometry;
    geometry.ox = geometry.oy = geometry.oz = 0.0f;
    geometry.hs = geometry.inv_hs = 1.0f;
    geometry.nx = geometry.ny = geometry.nz = 2;

    FieldNaive32 boundary32;
    boundary32.g = geometry;
    boundary32.v.assign(geometry.nodes() * 4 + geometry.zero_tail_floats(), 0.0f);
    for (int iz = 0; iz < geometry.nodes_z(); ++iz)
        for (int iy = 0; iy < geometry.nodes_y(); ++iy)
            for (int ix = 0; ix < geometry.nodes_x(); ++ix) {
                float* node = &boundary32.v[
                    (((size_t(iz) * size_t(geometry.nodes_y()) + size_t(iy)) *
                      size_t(geometry.nodes_x()) + size_t(ix)) * 4)];
                node[0] = iy == 0 ? -32752.0f : 32752.0f;
                node[1] = -4.0f;
                node[2] = 5.0f;
                node[3] = -6.0f;
            }
    FieldNaive16 boundary16;
    quantize_field_neon_fp16(boundary32, boundary16);

    // Exercise all eight cell corners, all six faces, and the highest legal
    // cells with a coordinate immediately below each exclusive upper bound.
    const float upper = std::nextafter(2.0f, 0.0f);
    std::vector<float> reduced = {
        0,0,0,  1,0,0,  0,1,0,  1,1,0,
        0,0,1,  1,0,1,  0,1,1,  1,1,1,
        0,0.25f,0.75f, upper,0.25f,0.75f,
        0.25f,0,0.75f, 0.25f,upper,0.75f,
        0.25f,0.75f,0, 0.25f,0.75f,upper,
        0.25f,0.5f,0.5f};
    std::vector<float> physical(reduced.size());
    for (size_t i = 0; i < reduced.size(); ++i)
        physical[i] = reduced[i] / COORD_SCALE;
    const float matrix[9] = {COORD_SCALE,0,0,0,COORD_SCALE,0,0,0,COORD_SCALE};
    const float translation[3] = {0,0,0};
    float edge_gradient[7]{};
    float edge_score = 0.0f;
    run_mixed_field(boundary16, physical, matrix, translation,
                    edge_gradient, edge_score);
    for (float value : edge_gradient)
        require(std::isfinite(value), "mixed half face/corner output is nonfinite");

    const std::vector<float> boundary_pose = {
        0.25f / COORD_SCALE, 0.5f / COORD_SCALE, 0.5f / COORD_SCALE};
    float gradient[7]{};
    float score = 0.0f;
    run_mixed_field(boundary16, boundary_pose, matrix, translation,
                    gradient, score);
    require(score == 0.0f && gradient[4] == 4.0f &&
            gradient[5] == -5.0f && gradient[6] == 6.0f &&
            close(gradient[1], 11.0f, 2e-5f, 2e-5f) &&
            close(gradient[2], 1.0f, 2e-5f, 2e-5f) &&
            close(gradient[3], -6.5f, 2e-5f, 2e-5f),
            "mixed half boundary subtraction, signed gradients, or torque failed");
    for (float value : gradient)
        require(std::isfinite(value), "mixed half boundary output is nonfinite");

    // Under FZ16, the smallest positive half subnormal is flushed during the
    // native-half Y/Z lerps; without FZ16 it survives and widens as 2^-24.
    FieldNaive16 underflow;
    underflow.g = geometry;
    underflow.v.assign(geometry.nodes() * 4 + geometry.zero_tail_floats(), 0);
    for (size_t node = 0; node < geometry.nodes(); ++node)
        underflow.v[node * 4] = 0x0001u;
    const std::vector<float> one_atom = {
        0.5f / COORD_SCALE, 0.5f / COORD_SCALE, 0.5f / COORD_SCALE};
    RestoreFpcr restore;
    const uint64_t fz16_mask = uint64_t(1) << 19;
    write_fpcr(restore.original & ~fz16_mask);
    float no_fz_gradient[7]{};
    float no_fz_score = 0.0f;
    run_mixed_field(underflow, one_atom, matrix, translation,
                    no_fz_gradient, no_fz_score);
    write_fpcr(restore.original | fz16_mask);
    float fz_gradient[7]{};
    float fz_score = 0.0f;
    run_mixed_field(underflow, one_atom, matrix, translation,
                    fz_gradient, fz_score);
    require(no_fz_score == std::ldexp(1.0f, -24) && fz_score == 0.0f &&
            std::isfinite(no_fz_score) && std::isfinite(fz_score),
            "FZ16 half-subnormal behavior changed or produced a nonfinite result");
}
#endif

#if defined(__aarch64__)
// score_pairs' AArch64 field path uses the pipelined gradient, not the AoS
// one, on the premise that the two are the same arithmetic. Pin it:
// bytes of the score and all seven gradient terms, for both addressing paths
// (proved uint32 offsets and the checked size_t fallback), across atom counts
// on and off the four-wide body, and poses inside, straddling and outside the
// grid.
void test_aos_and_pipelined_gradients_agree() {
    const std::vector<float> query_xyz = {
        0.20f*COORD_SCALE, -0.10f*COORD_SCALE, 0.70f*COORD_SCALE,
       -0.50f*COORD_SCALE,  0.40f*COORD_SCALE, 0.30f*COORD_SCALE,
        0.65f*COORD_SCALE,  0.25f*COORD_SCALE,-0.35f*COORD_SCALE,
       -0.15f*COORD_SCALE, -0.55f*COORD_SCALE,-0.20f*COORD_SCALE};
    FieldNaive32 proved;
    build_field(query_xyz.data(), nullptr, 4, plan_field(query_xyz.data(), 4, 0.5f, 8.0f),
                proved);
    require(proved.neon_offsets32, "a small field must prove uint32 offsets");
    FieldNaive32 unproved = proved;
    unproved.neon_offsets32 = false;

    uint32_t state = 280u;
    auto next = [&state] {
        state = state * 1664525u + 1013904223u;
        return float(state >> 8) / float(1u << 24) * 2.0f - 1.0f;  // [-1, 1)
    };
    int compared = 0, nonzero = 0, zero = 0;
    for (const int atoms : {1, 3, 4, 5, 8, 9, 13, 32, 255}) {
        std::vector<float> xyz(size_t(atoms) * 3);
        for (float& value : xyz) value = 1.5f * next();
        const Conformer conformer{xyz.data(), atoms};
        ConformerSoA soa;
        soa.build(conformer);
        for (const float angle : {0.0f, 0.27f, 1.9f})
            for (const float shift : {0.0f, 3.0f, 7.0f, 9.5f, 1000.0f}) {
                const float c = std::cos(angle), s = std::sin(angle);
                const float m[9] = {COORD_SCALE*c, -COORD_SCALE*s, 0.0f,
                                    COORD_SCALE*s,  COORD_SCALE*c, 0.0f,
                                    0.0f, 0.0f, COORD_SCALE};
                const float t[3] = {shift*COORD_SCALE, 0.13f, -0.5f*shift*COORD_SCALE};
                for (const float norm : {1.0f, 0.25f})
                    for (const FieldNaive32* field : {&proved, &unproved}) {
                        float g_aos[7]{}, g_pipe[7]{};
                        float s_aos = -1.0f, s_pipe = -2.0f;
                        field_gradient_neon(*field, conformer, m, t, norm, g_aos, &s_aos);
                        field_gradient_neon_pipelined(*field, soa, m, t, norm, g_pipe,
                                                      &s_pipe);
                        require(std::memcmp(&s_aos, &s_pipe, sizeof s_aos) == 0 &&
                                    std::memcmp(g_aos, g_pipe, sizeof g_aos) == 0,
                                "AoS and pipelined NEON field gradients must agree bytewise");
                        ++compared;
                        (s_aos != 0.0f ? nonzero : zero)++;
                    }
            }
    }
    require(compared > 0 && nonzero > 0 && zero > 0,
            "the sweep must reach poses on and off the field");
}
#endif

}  // namespace

int main() {
    try {
#if defined(__aarch64__)
        test_mixed_half_edge_cases();
        test_aos_and_pipelined_gradients_agree();
#endif
        // Query coordinates are reduced; target coordinates are physical Å.
        std::vector<float> query_xyz = {
            0.20f*COORD_SCALE, -0.10f*COORD_SCALE, 0.70f*COORD_SCALE,
           -0.50f*COORD_SCALE,  0.40f*COORD_SCALE, 0.30f*COORD_SCALE,
            0.65f*COORD_SCALE,  0.25f*COORD_SCALE,-0.35f*COORD_SCALE};
        const Conformer query{query_xyz.data(), 3};
        std::vector<float> library_xyz = {
             0.60f,  0.20f, -0.30f,
            -0.35f,  0.50f,  0.15f,
             0.75f, -0.20f,  0.45f,
             0.12f,  0.90f, -0.25f,
            -0.60f, -0.40f,  0.55f};
        const Conformer library{library_xyz.data(), 5};
        const float angle = 0.27f;
        const float c = std::cos(angle), s = std::sin(angle);
        const float matrix[9] = {
            COORD_SCALE*c, -COORD_SCALE*s, 0.0f,
            COORD_SCALE*s,  COORD_SCALE*c, 0.0f,
            0.0f, 0.0f, COORD_SCALE};
        const float translation[3] = {0.13f, -0.21f, 0.08f};

        FieldNaive32 oracle_field;
        const FieldGeometry geometry = plan_field(query.xyz, query.n_atoms,
                                                   0.5f, 8.0f);
        build_field(query.xyz, nullptr, query.n_atoms, geometry, oracle_field);
        ShapeEvaluation oracle{};
        direct_stored_gradient(oracle_field, library, matrix, translation, 1.0f,
                               oracle);

        const ShapeEvaluatorConfig scalar_cfg =
            config(ShapeBackend::ScalarField, ShapeGrain::Scalar, 1);
        const ShapeEvaluatorConfig paired_cfg =
            config(ShapeBackend::PairedChannelField, ShapeGrain::LibraryAtom, 8);
        const ShapeEvaluatorConfig pipe_cfg =
            config(ShapeBackend::PipelinedField, ShapeGrain::LibraryAtom, 8);
        ShapeEvaluatorConfig half_cfg = pipe_cfg;
        half_cfg.storage = ShapeFieldStorage::FP16;
        ShapeEvaluatorConfig mixed_half_cfg = half_cfg;
        mixed_half_cfg.interpolation = ShapeFieldInterpolation::MixedFP16YZ;
        PreparedShapeEvaluatorSet empty_shape_set;
        empty_shape_set.config = half_cfg;
        require(empty_shape_set.evaluators.empty() &&
                empty_shape_set.matches_config(half_cfg) &&
                !empty_shape_set.matches_config(mixed_half_cfg),
                "empty prepared shape sets must reject interpolation mismatch");
        const ShapeQueryMetadata qm{42};
        // Tiny queries whose grids cannot be addressed by PreparedBlock.
        // These must throw invalid_argument before attempting multi-GB field
        // allocations (bad_alloc is deliberately not accepted).
        for (const auto dimensions : {std::array<float, 3>{0.f, 0.f, 0.f},
                                      std::array<float, 3>{600000000.f, 0.f, 0.f},
                                      std::array<float, 3>{24000.f, 24000.f, 0.f}}) {
            const float oversized_xyz[6] = {0.f, 0.f, 0.f,
                dimensions[0]*COORD_SCALE, dimensions[1]*COORD_SCALE,
                dimensions[2]*COORD_SCALE};
            const Conformer oversized{oversized_xyz, 2};
            auto oversized_cfg = pipe_cfg;
            oversized_cfg.field_spacing_angstrom = 1.f;
            oversized_cfg.field_padding_angstrom =
                dimensions[0] == 0.f ? 512.f : 0.5f;
            bool rejected = false;
            try {
                (void)make_pipelined_field_shape_evaluator(
                    oversized_cfg, oversized, qm);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "unaddressable field must fail before allocation");
        }
        const auto scalar = make_scalar_field_shape_evaluator(
            scalar_cfg, query, qm);
        const auto paired = make_paired_channel_field_shape_evaluator(
            paired_cfg, query, qm);
        const auto pipelined = make_pipelined_field_shape_evaluator(
            pipe_cfg, query, qm);
        const auto half_stored = make_pipelined_field_shape_evaluator(
            half_cfg, query, qm);
        const auto mixed_half = make_pipelined_field_shape_evaluator(
            mixed_half_cfg, query, qm);
        const auto scalar_stamp = scalar->stamp();
        const auto paired_stamp = paired->stamp();
        const auto pipe_stamp = pipelined->stamp();
        const auto half_stamp = half_stored->stamp();
        const auto mixed_half_stamp = mixed_half->stamp();
        require(scalar_stamp.backend.find("fp32") != std::string::npos &&
                scalar_stamp.width == 1 && !scalar_stamp.vector_path &&
                !scalar_stamp.requires_library_soa,
                "scalar field stamp must name its actual path");
        require(paired_stamp.grain.find(
#if defined(__aarch64__)
                    "four stored channels per NEON"
#else
                    "two library samples by four"
#endif
                    ) !=
                    std::string::npos && paired_stamp.width == 8 &&
                paired_stamp.vector_path && !paired_stamp.requires_library_soa,
                "paired-channel stamp must not claim trajectory lanes");
        require(pipe_stamp.grain.find(
#if defined(__aarch64__)
#if defined(R3_FIELD_NEON_SCHEDULE) && R3_FIELD_NEON_SCHEDULE != 0
                    "full prepared state"
#else
                    "eight prepared atoms"
#endif
#else
                    "eight library atoms"
#endif
                    ) !=
                    std::string::npos && pipe_stamp.requires_library_soa,
                "pipeline stamp must name its T124 atom grain/workspace");
        const auto prep = shape_field_preparation_metadata(*pipelined);
        const auto half_prep = shape_field_preparation_metadata(*half_stored);
        const auto mixed_half_prep = shape_field_preparation_metadata(*mixed_half);
        require(half_stored->matches(half_cfg, query) &&
                !half_stored->matches(pipe_cfg, query),
                "storage precision must participate in prepared-handle matching");
        if (half_prep.actual_storage == ShapeFieldStorage::FP16) {
            require(half_prep.field_bytes * 2 == prep.field_bytes &&
                    half_prep.field_quantize_seconds >= 0.0 &&
                    half_stamp.storage.find("FP16") != std::string::npos,
                    "FP16 storage must halve node bytes and stamp actual precision");
        } else {
            require(half_prep.storage_fallback != "none" &&
                    half_stamp.storage.find("fallback=") != std::string::npos,
                    "unsupported or unsafe FP16 must record an FP32 fallback");
        }
        require(mixed_half->matches(mixed_half_cfg, query) &&
                !mixed_half->matches(half_cfg, query),
                "mixed interpolation must participate in prepared-handle matching");
        if (mixed_half_prep.actual_storage == ShapeFieldStorage::FP16) {
            require(mixed_half_prep.actual_interpolation ==
                        ShapeFieldInterpolation::MixedFP16YZ &&
                    std::string(mixed_half_stamp.storage).find(
                        "native-half Y/Z") != std::string::npos &&
                    std::string(mixed_half_stamp.backend).find(
                        "fp16mixedyz_fp32x") != std::string::npos,
                    "mixed candidate stamp must name its actual Y/Z-half, X-float math");
        } else {
            require(mixed_half_prep.actual_storage == ShapeFieldStorage::FP32 &&
                    mixed_half_prep.actual_interpolation ==
                        ShapeFieldInterpolation::FP32 &&
                    mixed_half_prep.storage_fallback != "none",
                    "unsupported or unsafe mixed half must fail closed to FP32");
        }
        // A tiny field with repeated query atoms makes stored values exceed
        // half's finite range without making field construction expensive.
        std::vector<float> oversized_values(size_t(70000) * 3, 0.0f);
        const Conformer oversized_query{oversized_values.data(), 70000};
        ShapeEvaluatorConfig unsafe_half_cfg = half_cfg;
        unsafe_half_cfg.field_spacing_angstrom = 1000.0f;
        unsafe_half_cfg.field_padding_angstrom = 0.1f;
        const auto unsafe_half = make_pipelined_field_shape_evaluator(
            unsafe_half_cfg, oversized_query, ShapeQueryMetadata{44});
        const auto unsafe_prep = shape_field_preparation_metadata(*unsafe_half);
        require(unsafe_prep.actual_storage == ShapeFieldStorage::FP32 &&
                std::string(unsafe_prep.storage_fallback) ==
                    "field_value_outside_finite_fp16_range" &&
                unsafe_prep.field_bytes > 0,
                "out-of-range FP32 nodes must fail closed to FP32 storage");

        // At this small spacing the repeated atoms produce finite FP16 field
        // nodes above the mixed lerp's conservative +/-32720 bound but below
        // FP16's storage maximum. Storage-only may use them; mixed arithmetic
        // must select the complete FP32 fallback at query preparation.
        std::vector<float> mixed_bound_values(size_t(40000) * 3, 0.0f);
        const Conformer mixed_bound_query{mixed_bound_values.data(), 40000};
        ShapeEvaluatorConfig mixed_bound_storage_cfg = half_cfg;
        mixed_bound_storage_cfg.field_spacing_angstrom = 0.01f;
        mixed_bound_storage_cfg.field_padding_angstrom = 0.001f;
        ShapeEvaluatorConfig mixed_bound_cfg = mixed_half_cfg;
        mixed_bound_cfg.field_spacing_angstrom = 0.01f;
        mixed_bound_cfg.field_padding_angstrom = 0.001f;
        const auto storage_bound_field = make_pipelined_field_shape_evaluator(
            mixed_bound_storage_cfg, mixed_bound_query, ShapeQueryMetadata{45});
        const auto mixed_bound_field = make_pipelined_field_shape_evaluator(
            mixed_bound_cfg, mixed_bound_query, ShapeQueryMetadata{46});
        const auto storage_bound_prep =
            shape_field_preparation_metadata(*storage_bound_field);
        const auto mixed_bound_prep =
            shape_field_preparation_metadata(*mixed_bound_field);
#if defined(__aarch64__)
        if (field_neon_fp16_supported()) {
            require(storage_bound_prep.actual_storage == ShapeFieldStorage::FP16 &&
                    mixed_bound_prep.actual_storage == ShapeFieldStorage::FP32 &&
                    std::string(mixed_bound_prep.storage_fallback) ==
                        "mixed_half_abs_bound_exceeded",
                    "mixed half subtraction guard must reject >32720 values while storage-only remains safe");
        }
#endif

        require(scalar->matches(scalar_cfg, query),
                "same query snapshot/config must match prepared field");
        std::vector<float> changed_query = query_xyz;
        changed_query[0] += 0.125f;
        require(!scalar->matches(scalar_cfg,
                                 Conformer{changed_query.data(), 3}),
                "different query snapshot must not reuse a field");
        require(prep.query_atoms == 3 && prep.field_bytes > 0 &&
                prep.query_snapshot_bytes == query_xyz.size()*sizeof(float) &&
                prep.nodes_x == geometry.nodes_x() &&
                prep.nodes_y == geometry.nodes_y() &&
                prep.nodes_z == geometry.nodes_z() && prep.field_build_seconds >= 0.0,
                "field preparation metadata is incomplete");

        ShapeLibraryWorkspace workspace = prepare_field_shape_library_workspace(
            library, ShapeBackend::PipelinedField, 8);
        require(workspace.matches(library) && workspace.field_atom != nullptr &&
                workspace.field_atom->n == library.n_atoms &&
                workspace.prepared_bytes == sizeof(ConformerSoA),
                "pipelined field workspace must be prepared for this target");
        const ShapeLibraryWorkspace paired_workspace =
            prepare_field_shape_library_workspace(
                library, ShapeBackend::PairedChannelField, 8);
        require(paired_workspace.matches(library) &&
                paired_workspace.field_atom == nullptr &&
                paired_workspace.prepared_bytes == 0,
                "paired-channel path must not build an unused field SoA");

        ShapePoseRequest scalar_request{&library, nullptr, matrix, translation, 1.0f};
        ShapePoseRequest paired_request{&library, nullptr, matrix, translation, 1.0f};
        ShapePoseRequest pipe_request{&library, &workspace, matrix, translation, 1.0f};
        ShapeEvaluation scalar_out{}, paired_out{}, pipe_out{};
        scalar->evaluate_one(scalar_request, scalar_out);
        paired->evaluate_one(paired_request, paired_out);
        pipelined->evaluate_one(pipe_request, pipe_out);
        ShapeEvaluation half_out{};
        half_stored->evaluate_one(pipe_request, half_out);
        ShapeEvaluation mixed_half_out{};
        mixed_half->evaluate_one(pipe_request, mixed_half_out);
        PreparedReducedPose shared_pose;
        std::copy(matrix, matrix + 9, shared_pose.matrix);
        std::copy(translation, translation + 3, shared_pose.translation);
        for (const auto& evaluator : {scalar, paired, pipelined, half_stored, mixed_half}) {
            const auto bound = evaluator->bind_execution(library,
                evaluator->config().backend == ShapeBackend::PipelinedField ? &workspace : nullptr);
            ShapeEvaluation checked{}, trusted{};
            evaluator->evaluate_one(pipe_request, checked);
            bound.evaluate(shared_pose, trusted);
            require(std::memcmp(&checked.s_ab, &trusted.s_ab, sizeof(float)) == 0 &&
                    std::memcmp(checked.descent, trusted.descent, sizeof checked.descent) == 0 &&
                    checked.counts.field_samples == trusted.counts.field_samples &&
                    checked.counts.issued_terms == trusted.counts.issued_terms,
                    "bound field numerical route must preserve exact values/gradients/counts");
            const auto values = evaluator->bind_execution(library,
                evaluator->config().backend == ShapeBackend::PipelinedField ? &workspace : nullptr, true);
            ShapeEvaluation value_output;
            values.evaluate(shared_pose, value_output);
            require(std::memcmp(&checked.s_ab, &value_output.s_ab, sizeof(float)) == 0,
                    "field values specialization must preserve exact channel/reduction bytes");
        }
        compare(scalar_out, oracle, 2e-6f, "scalar");
        compare(paired_out, oracle, 7e-5f, "paired-channel");
        compare(pipe_out, oracle, 7e-5f, "pipelined");
        compare(half_out, pipe_out, 2e-3f, "fp16-storage");
        compare(mixed_half_out, pipe_out, 5e-3f, "mixed-half-yz");
        require(scalar_out.counts.field_samples == 5 &&
                paired_out.counts.field_samples == 5 &&
                pipe_out.counts.field_samples == 5 &&
                pipe_out.counts.issued_terms ==
#if defined(__aarch64__)
                    5,
#else
                    8,
#endif
                "field sample/tail census must use real and issued sample units");

        // gnorm is passed through once: score is unchanged and the stored
        // descent scales linearly. No shape Tanimoto normalization belongs here.
        ShapePoseRequest scaled_request{&library, nullptr, matrix, translation, 0.25f};
        ShapeEvaluation scaled_out{};
        scalar->evaluate_one(scaled_request, scaled_out);
        require(close(scaled_out.s_ab, scalar_out.s_ab, 1e-7f, 1e-7f),
                "gnorm must not change the raw field overlap");
        for (int k = 1; k < 7; ++k)
            require(close(scaled_out.descent[k], 0.25f*scalar_out.descent[k],
                          2e-6f, 2e-6f),
                    "gnorm must scale the stored descent exactly once");

        // A far-outside pose exercises the zero halo and general bounds path.
        const float outside_t[3] = {1000.0f, -1000.0f, 800.0f};
        ShapePoseRequest outside_scalar{&library, nullptr, matrix, outside_t, 1.0f};
        ShapePoseRequest outside_pair{&library, nullptr, matrix, outside_t, 1.0f};
        ShapePoseRequest outside_pipe{&library, &workspace, matrix, outside_t, 1.0f};
        ShapeEvaluation outside{};
        scalar->evaluate_one(outside_scalar, outside);
        require(outside.s_ab == 0.0f, "scalar sampler must return zero outside field");
        paired->evaluate_one(outside_pair, outside);
        require(outside.s_ab == 0.0f, "paired sampler must return zero outside field");
        pipelined->evaluate_one(outside_pipe, outside);
        require(outside.s_ab == 0.0f, "pipelined sampler must return zero outside field");
        half_stored->evaluate_one(outside_pipe, outside);
        require(outside.s_ab == 0.0f && std::isfinite(outside.descent[4]),
                "FP16 sampler must preserve safe zero-border behavior");
        mixed_half->evaluate_one(outside_pipe, outside);
        require(outside.s_ab == 0.0f &&
                std::all_of(outside.descent, outside.descent + 7,
                            [](float v) { return std::isfinite(v); }),
                "mixed-half sampler must preserve safe zero-border behavior");

        // The SoA's defined partial-block padding is zero; values beyond
        // the padded block are deliberately poisoned and must never be read.
        auto poison = std::make_shared<ConformerSoA>();
        poison->build(library);
        for (int j = poison->padded; j < 64; ++j)
            poison->x[j] = poison->y[j] = poison->z[j] =
                std::numeric_limits<float>::quiet_NaN();
        ShapeLibraryWorkspace poisoned_workspace;
        poisoned_workspace.field_atom = poison;
        poisoned_workspace.source_xyz = library.xyz;
        poisoned_workspace.source_atoms = library.n_atoms;
        ShapePoseRequest poison_request{&library, &poisoned_workspace,
                                        matrix, translation, 1.0f};
        ShapeEvaluation poison_out{};
        pipelined->evaluate_one(poison_request, poison_out);
        compare(poison_out, oracle, 7e-5f, "poisoned tail");
        mixed_half->evaluate_one(poison_request, poison_out);
        compare(poison_out, pipe_out, 5e-3f, "mixed-half poisoned tail");

        // Missing and mismatched SOA are errors, not a hidden scalar fallback.
        bool missing_rejected = false, mismatch_rejected = false;
        try {
            pipelined->evaluate_one(paired_request, pipe_out);
        } catch (const std::invalid_argument&) { missing_rejected = true; }
        std::vector<float> other_xyz = library_xyz;
        const Conformer other_library{other_xyz.data(), library.n_atoms};
        ShapePoseRequest mismatch_request{&other_library, &workspace,
                                          matrix, translation, 1.0f};
        try {
            pipelined->evaluate_one(mismatch_request, pipe_out);
        } catch (const std::invalid_argument&) { mismatch_rejected = true; }
        require(missing_rejected && mismatch_rejected,
                "pipelined sampler must reject absent/foreign workspace");
        bool bound_missing = false, bound_mismatch = false, bound_count = false;
        try { (void)pipelined->bind_execution(library, nullptr); }
        catch (const std::invalid_argument&) { bound_missing = true; }
        try { (void)pipelined->bind_execution(other_library, &workspace); }
        catch (const std::invalid_argument&) { bound_mismatch = true; }
        auto stale = workspace;
        stale.source_atoms += 1;
        try { (void)pipelined->bind_execution(library, &stale); }
        catch (const std::invalid_argument&) { bound_count = true; }
        require(bound_missing && bound_mismatch && bound_count,
                "checked binding must reject missing/stale/foreign workspaces before iteration");

        // color11's eight-pair mapping and color16's two-pair x four-start
        // mapping both reduce to arbitrary request tails over one query.
        std::array<std::vector<float>, 8> batch_xyz;
        std::array<Conformer, 8> batch_library;
        std::array<ShapeLibraryWorkspace, 8> batch_workspace;
        std::array<ShapePoseRequest, 8> batch_requests;
        std::array<std::array<float, 3>, 8> batch_translation{};
        for (int i = 0; i < 8; ++i) {
            batch_xyz[size_t(i)] = library_xyz;
            batch_xyz[size_t(i)][0] += 0.03f*float(i);
            batch_library[size_t(i)] = {batch_xyz[size_t(i)].data(), 5};
            batch_workspace[size_t(i)] = prepare_field_shape_library_workspace(
                batch_library[size_t(i)], ShapeBackend::PipelinedField, 8);
            batch_translation[size_t(i)] = {
                translation[0] + 0.04f*float(i),
                translation[1] - 0.025f*float(i),
                translation[2] + 0.015f*float(i)};
            batch_requests[size_t(i)] = {
                &batch_library[size_t(i)], &batch_workspace[size_t(i)], matrix,
                batch_translation[size_t(i)].data(), 1.0f};
        }
        std::vector<ShapePoseRequest> request_vec(batch_requests.begin(),
                                                   batch_requests.end());
        compare_batch_to_one(*pipelined, request_vec);
        compare_batch_to_one(*half_stored, request_vec);
        compare_batch_to_one(*mixed_half, request_vec);
        // color16 maps two conformers to four independent poses each. Each
        // four-request group owns its query evaluator but may share arbitrary
        // target/workspace records with other groups.
        std::vector<float> query2_xyz = query_xyz;
        query2_xyz[1] += 0.18f*COORD_SCALE;
        const Conformer query2{query2_xyz.data(), 3};
        const auto pipelined_q2 = make_pipelined_field_shape_evaluator(
            pipe_cfg, query2, ShapeQueryMetadata{43});
        std::array<ShapePoseRequest, 8> color16_requests;
        for (int i = 0; i < 8; ++i) {
            const int pair = i < 4 ? 0 : 4;
            color16_requests[size_t(i)] = {
                &batch_library[size_t(pair)], &batch_workspace[size_t(pair)],
                matrix, batch_translation[size_t(i)].data(), 1.0f};
        }
        std::vector<ShapePoseRequest> four_starts(color16_requests.begin(),
                                                   color16_requests.begin()+4);
        compare_batch_to_one(*pipelined, four_starts);
        std::vector<ShapePoseRequest> four_starts_q2(color16_requests.begin()+4,
                                                     color16_requests.end());
        compare_batch_to_one(*pipelined_q2, four_starts_q2);
        const auto mixed_half_q2 = make_pipelined_field_shape_evaluator(
            mixed_half_cfg, query2, ShapeQueryMetadata{43});
        compare_batch_to_one(*mixed_half_q2, four_starts_q2);
        std::vector<ShapePoseRequest> three_pose_tail(request_vec.begin(),
                                                       request_vec.begin()+3);
        compare_batch_to_one(*pipelined, three_pose_tail);
        compare_batch_to_one(*mixed_half, three_pose_tail);
        std::vector<ShapePoseRequest> seven_pose_tail(request_vec.begin(),
                                                       request_vec.begin()+7);
        compare_batch_to_one(*pipelined, seven_pose_tail);
        compare_batch_to_one(*mixed_half, seven_pose_tail);

        // Empty query/target and mixed query handles are valid. A zero-atom
        // workspace preserves null-pointer identity and samples no grid data.
        const Conformer empty{nullptr, 0};
        const auto empty_field = make_pipelined_field_shape_evaluator(
            pipe_cfg, empty, ShapeQueryMetadata{7});
        const ShapeLibraryWorkspace empty_workspace =
            prepare_field_shape_library_workspace(
                empty, ShapeBackend::PipelinedField, 8);
        require(empty_workspace.matches(empty) && empty_workspace.field_atom &&
                empty_workspace.field_atom->n == 0 &&
                empty_workspace.prepared_bytes == sizeof(ConformerSoA),
                "empty target workspace must retain null/zero identity");
        const float identity[9] = {COORD_SCALE,0,0,0,COORD_SCALE,0,
                                   0,0,COORD_SCALE};
        const float zero_t[3] = {0,0,0};
        ShapePoseRequest empty_req{&empty, &empty_workspace, identity, zero_t, 1.0f};
        ShapeEvaluation empty_out{};
        empty_field->evaluate_one(empty_req, empty_out);
        require(empty_out.s_ab == 0.0f && empty_out.counts.field_samples == 0,
                "empty query/target must contribute zero");
        ShapePoseRequest mixed_req{&library, &workspace, matrix, translation, 1.0f};
        pipelined->evaluate_one(mixed_req, pipe_out);
        empty_field->evaluate_one(empty_req, empty_out);
        require(pipe_out.s_ab != 0.0f && empty_out.s_ab == 0.0f,
                "separate query handles must not leak prepared fields");

        // Controls do not silently select a nearby implementation.
        bool wrong_width_rejected = false, wrong_workspace_width_rejected = false;
        try {
            auto wrong = paired_cfg;
            wrong.width = 4;
            (void)make_paired_channel_field_shape_evaluator(wrong, query, qm);
        } catch (const std::invalid_argument&) { wrong_width_rejected = true; }
        try {
            (void)prepare_field_shape_library_workspace(
                library, ShapeBackend::PairedChannelField, 4);
        } catch (const std::invalid_argument&) {
            wrong_workspace_width_rejected = true;
        }
        require(wrong_width_rejected && wrong_workspace_width_rejected,
                "unsupported field width must be rejected");

        std::puts("PASS T185 shape field evaluator");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "FAIL T185 shape field evaluator: %s\n", ex.what());
        return 1;
    }
}

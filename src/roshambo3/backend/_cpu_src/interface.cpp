// The pybind entry point for the CPU shape kernel.
//
// One function, `score_pairs`, whose signature is a subset of the CUDA
// wrapper's -- and which REFUSES every knob it cannot honor rather than
// ignoring one. That rule is older than this file (and restated
// throughout `backend/cuda.py`) and it is the whole reason a CPU backend can be
// trusted at all: an ignored knob is a silent wrong answer, and half of what
// `ScreenConfig.for_width()` emits is about shared memory, warps and carveouts.
//
// It also reports which SIMD dispatch target it was compiled for and which it
// SELECTED at run time. A decision-grade CPU number is refused without the
// second, on the grounds that a build which silently ran a baseline scalar path
// looks like a slow kernel rather than a broken build.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <thread>
#include "cpu_platform.hpp"
#include <string>

#include "color_view.hpp"
#include "final_selection_policy.hpp"
#include "grain.hpp"
#include "field.hpp"
#include "field_avx2.hpp"
#if defined(__aarch64__)
#include "field_neon.hpp"
#endif
#include "grain_conformer.hpp"
#include "score.hpp"
#include "shape_combo_api.hpp"
#include "shape_field_evaluator.hpp"

// Phase timers that PRICE a call without changing its
// result. Deliberately coarse -- every bracket below sits OUTSIDE `run_range`
// / `run_conformer_range` (the per-pair dispatch lambdas further down this
// file) rather than inside them. That is not a stylistic choice: this file's
// own history is the evidence against a finer one. A single counter
// increment added INSIDE the per-pair loop, gated on a null check that is
// always taken in a normal run, moved the SHIPPING path's scores on 76% of
// pairs with the arithmetic itself untouched (see the `batched_fallback_pairs`
// postmortem below, near the pair loop) -- the inlining cliff, where this
// function is large enough that ANY edit to it can change what the compiler
// inlines and therefore which FMAs get contracted. Every timer here brackets
// a block that already exists as its own scope (a `std::vector` fill loop, or
// the `if (threads==1) {...} else {...}` dispatch as a whole), so the
// per-pair bodies are not touched at all. `tests/test_cpu_color_seam.py`
// checks the consequence directly: scores are bit-identical whether these
// timers are on or off.
//
// `analytic_color_s`, `field_color_s`, `exact_final_rescore_s` and
// `reduction_s` are named here and always 0.0: there is no color kernel yet
// (this seam is plumbing and a census, not a fast kernel -- color=True is
// refused before it reaches this function), and the CPU seam does no
// device-side reduction (that is `engine.HostTopK`, host Python, outside this
// boundary). The names exist now so color kernels do not have to invent
// what to call their own phases.
struct PhaseTimers {
    double query_prep_s = 0.0;
    double query_self_overlap_s = 0.0;
    double library_self_overlap_s = 0.0;
    double field_construction_s = 0.0;
    double shape_search_s = 0.0;
    double analytic_color_s = 0.0;
    double field_color_s = 0.0;
    double exact_final_rescore_s = 0.0;
    double reduction_s = 0.0;
};

//: RAII bracket: adds elapsed wall time to `*acc` on destruction, and does
//: nothing at all (not even reading the clock) when `acc` is null -- the
//: `with_phase_timers=false` default path pays a null check, not a
//: `steady_clock::now()`.
struct ScopedPhaseTimer {
    double* acc;
    std::chrono::steady_clock::time_point t0;

    explicit ScopedPhaseTimer(double* a)
        : acc(a), t0(a ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{}) {}

    ~ScopedPhaseTimer() {
        if (!acc) return;
        *acc +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
    }
};

// Which SOURCE was compiled, as opposed to which commit was checked out.
//
// This is the CPU subtree's stamp and not the CUDA one. It previously included
// `roshambo3_source_stamp.h`, which carries ROSHAMBO3_CUDA_SRC_* -- macros that
// describe a different subtree and would have made this extension report the
// CUDA kernel's identity as its own. Harmless while nothing read them; not
// harmless now that this kernel is the reference anchor and a fixture
// names its generator by exactly this hash.
//
// Picked up with __has_include so a compile driven by something other than this
// project's CMake still builds -- and reports "unknown" rather than silently
// reporting clean.
#if defined(__has_include)
#if __has_include("roshambo3_cpu_source_stamp.h")
#include "roshambo3_cpu_source_stamp.h"
#endif
#endif

#ifndef ROSHAMBO3_CPU_SRC_SHA256
#define ROSHAMBO3_CPU_SRC_SHA256 "unknown"
#endif
#ifndef ROSHAMBO3_CPU_SRC_TREE
#define ROSHAMBO3_CPU_SRC_TREE "unknown"
#endif
// Tri-state, and it has to be: 1 dirty, 0 clean, -1 nobody looked. A boolean
// would make an unstamped build indistinguishable from a clean one -- the same
// defect once fixed for `checked_a_gpu_build`.
#ifndef ROSHAMBO3_CPU_SRC_DIRTY
#define ROSHAMBO3_CPU_SRC_DIRTY (-1)
#endif
#ifndef ROSHAMBO3_CPU_SRC_DIRTY_FILES
#define ROSHAMBO3_CPU_SRC_DIRTY_FILES ""
#endif
#ifndef ROSHAMBO3_SOURCE_STAMP_PHASE
#define ROSHAMBO3_SOURCE_STAMP_PHASE "none"
#endif

namespace py = pybind11;
using namespace r3::cpu;

// Defined in `grain_avx512.cpp`, the only file compiled with `-mavx512f`.
// Declared rather than defined here on purpose: an in-file target attribute
// does not reach an implicit template instantiation, and a `#pragma GCC target`
// region did not either -- both produced silently wrong code that returned
// constants independent of the input. That file records what it cost to find
// out. The anchor stays at -march=x86-64-v3 because only that ONE file carries
// the flag.
namespace r3 {
namespace cpu {
void bind_scalar_color(py::module_& module);
void bind_color_fields(py::module_& module);
void bind_shape_combo_api(py::module_& module);
void bind_final_rescore(py::module_& module);
PairResult run_avx512_grain(bool poly, Grain gr, const Conformer& a,
                            const Conformer& b, float s_aa, float s_bb,
                            int start_mode, int steps, float lr_q,
                            float lr_t_scaled, const SoaConformer* a_soa,
                            OpCensus* census);
void run_avx512_block(bool poly, const Conformer& a, const LaneBlock& blk,
                      float s_aa, int start_mode, int steps, float lr_q,
                      float lr_t_scaled, float cutoff_d2, float converge_tol,
                      int only_start, PairResult* out, OpCensus* census);
}
}

namespace {

// The target this translation unit was compiled for. With several targets
// and a runtime dispatch, `selected_target()` stops being a constant and starts being a
// measurement. The distinction between "compiled for" and "selected" exists
// now, before it can be got wrong, rather than being retrofitted after the
// first number is quoted.
//: Does THIS PROCESS have AVX-512? Asked of the CPU, not of the build.
//:
//: A decision-grade CPU number is refused without the selected dispatch
//: target, on the grounds that a build which silently ran a narrower path looks
//: like a slow kernel rather than a broken measurement. Width 16 is therefore
//: REFUSED on a machine without AVX-512 rather than quietly served at width 8.
static bool have_avx512() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f") != 0;
#else
    return false;
#endif
}

// The kernel is compiled at ONE target -- `-march=x86-64-v3`, set in
// CMakeLists -- so it has no runtime dispatch and `selected_target()` is
// legitimately a constant here. What it must not be is a WRONG constant: this
// returned "SCALAR" for a binary carrying 120 `vfmadd`, and a decision-grade
// CPU number is refused on exactly this field. A stamp says what the binary
// contains.
//
// Real multi-target dispatch is not yet implemented; `_cpu_src/probe.cpp`
// already demonstrates it working (AVX2 and AVX512F hosts) via
// `__builtin_cpu_supports`, so the mechanism is proven and only the kernel-side
// plumbing remains.
const char* compiled_targets() {
#if defined(__AVX512F__)
    return "AVX512F";
#elif defined(__AVX2__) && defined(__FMA__)
    // ASSERTED by a v3 translation unit, and safe because of a LINKER fact
    // rather than a compiler one: `interface.cpp` references
    // `run_avx512_grain`, so no binary can link without `grain_avx512.cpp`.
    // If that reference ever becomes conditional this string starts lying, and
    // the honest fix would be to export the name from that file instead.
    //
    // A COMMA-SEPARATED LIST, because the binary genuinely holds two targets:
    // this translation unit at -march=x86-64-v3 and `grain_avx512.cpp` at
    // -mavx512f. `backend/cpu.py` splits on the comma, so the stamp reports
    // both and `selected_target()` names the one this process will use.
    // Returning them joined by a `+` made the list a single unsplittable
    // string, and the invariant that the selected target is a MEMBER of the
    // compiled set silently stopped holding.
    return "X86_64_V3,AVX512F";
#elif defined(__AVX__)
    return "AVX";
#elif defined(__aarch64__)
    return "NEON";
#else
    return "SCALAR";
#endif
}

// The target this PROCESS can dispatch to, which is no longer the
// same question as what the build macros say.
//
// `compiled_targets()` is derived from `__AVX512F__`/`__AVX2__` in THIS
// translation unit, which is compiled at -march=x86-64-v3 and therefore reports
// X86_64_V3 even on a machine where the separately-compiled width-16 kernel is
// available and will be used. A decision-grade CPU number is refused without
// the SELECTED target precisely so that a narrower path cannot be reported as a
// wider one; reporting a wider capability as narrower is the same defect with
// the sign flipped, and it would make every AVX-512 record claim AVX2.
const char* selected_target() {
    // NOT `compiled_targets()` on the fallback. That string names what the
    // BINARY holds, which now includes AVX512F from the separate translation
    // unit -- so falling through to it made a machine with no AVX-512 at all
    // report having selected it. Caught immediately on such a machine; on a machine with it the wrong answer would have looked
    // right. `selected_target()` answers what this PROCESS will run, and
    // nothing else.
    if (have_avx512()) return "AVX512F";
#if defined(__AVX2__) && defined(__FMA__)
    return "X86_64_V3";
#elif defined(__AVX__)
    return "AVX";
#elif defined(__aarch64__)
    return "NEON";
#else
    return "SCALAR";
#endif
}

// Every knob the CUDA path takes that describes hardware this backend does
// not have. Refused BY NAME, with what it means, so a caller who passed one
// learns why it cannot apply rather than getting a number that ignored it.
//
// Published to Python as `inert_knob_names()` so the Engine can tell a knob a
// CALLER chose from one `ScreenConfig.for_width` emits unconditionally because
// it describes a CUDA launch. The first must be refused; the second was never
// a request, and refusing it made this backend unreachable through `Engine`.
// One list, in the one place that knows what a CUDA launch means.
static const std::pair<const char*, const char*> kInertKnobs[] = {
    {"smem_carveout", "the shared/L1 carveout is a CUDA SM partition"},
    {"min_blocks_per_sm", "__launch_bounds__ min-blocks is a register budget"},
    {"stage_width", "the staged-library block width is a shared-memory rung"},
    {"coord_residency", "where coordinates live between steps is SM storage"},
    {"color_residency", "likewise, for the color plane"},
    {"color_stage", "likewise"},
    {"lanes_per_group",
     "cooperating LANES are a warp concept; the CPU has its own grain"},
    {"start_groups_per_row", "start groups split a warp's lanes over one row"},
    {"stream_min_blocks_per_sm", "a per-SM residency target"},
    {"interleave", "an SM-level schedule"},
    {"color_ballot", "a warp-vote"},
    {"query_color_residency", "SM storage for the query's color plane"},
};

void refuse_inert(const py::kwargs& kw) {
    const auto& inert = kInertKnobs;

    // `library_tile_mode` is NOT in that list, because it is not a tuning
    // knob -- it is a statement of how the caller packed `library_coords`,
    // and this backend can read one of the three layouts it names. 0 is the
    // untiled per-conformer array this kernel expects; 1 and -1 are the
    // interleaved and planar 32-row warp tiles, which it would silently
    // misread. So it is honored when it names our layout and refused when it
    // names one we cannot read.
    //
    // Refusing it unconditionally, as this boundary used to, made the CPU
    // backend unreachable through `Engine` at all: engine.py passes
    // `library_tile_mode` on every launch because the party that packed the
    // buffer is the only one that knows its layout.
    if (kw.contains("library_tile_mode")) {
        const int tm = kw["library_tile_mode"].cast<int>();
        if (tm != 0)
            throw std::invalid_argument(
                "the cpu backend reads the untiled per-conformer layout "
                "(library_tile_mode=0); this buffer says " +
                std::to_string(tm) +
                ", which is a 32-row warp tile. The tile exists to coalesce a "
                "warp and is a pure tax on a CPU core, which wants one "
                "conformer's atoms contiguous. Build the source with "
                "tile=False. Refused rather than ignored -- both tiles are the "
                "same shape and size as the untiled array, so reading one at "
                "the other's stride returns another conformer's coordinates "
                "and raises nothing.");
    }
    for (const auto& kv : inert) {
        if (kw.contains(kv.first)) {
            throw std::invalid_argument(
                std::string("the cpu backend cannot honor '") + kv.first +
                "': " + kv.second +
                ". Refused rather than ignored -- an "
                "ignored knob is a silent wrong answer, which is why this "
                "boundary refuses by name.");
        }
    }
}

py::array_t<float> score_pairs(
    py::array_t<float, py::array::c_style | py::array::forcecast> query_coords,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        query_n_atoms,
    py::array_t<float, py::array::c_style | py::array::forcecast>
        library_coords,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
        library_counts,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        pair_query_row,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        pair_library_row,
    int steps, float lr_q, float lr_t, int start_mode,
    // `grain < 0` is the ANCHOR path in score.hpp, untouched and
    // still the default, so a caller that does not ask for a grain gets
    // exactly the kernel the anchor fixture names. `width` is the SOURCE
    // width -- what fixes the reduction order -- and is refused rather
    // than silently downgraded when the binary cannot honor it, on the
    // rule that a knob which is ignored is a silently wrong answer.
    int grain, int width, bool poly, int threads, float cutoff_d2,
    float converge_tol, int only_start,
    // `field_h > 0` selects the FIELD pose search at that lattice
    // spacing in Angstrom; 0 keeps the analytic path, which is the default
    // and the shipped one. `field_pad` is the halo around the query, and
    // it is a knob rather than a constant because the field's footprint --
    // the thing this path is about -- scales with it.
    float field_h, float field_pad, int field_width, int field_unroll,
    bool field_fp16, bool field_affine, bool field_edge, bool field_edgefield,
    bool field_pipeline, bool field_pairaos, bool field_coeff, int field_allin,
    int field_gridxf, const py::array_t<float>* library_vaa,
    bool field_winner_only, bool field_batched, bool field_report_field_score,
    bool with_census, py::dict* out_census,
    // Appended after the census pair rather than
    // inserted earlier in the list, so every existing positional call
    // site needed only ONE new trailing `false, nullptr` rather than a
    // renumbering of arguments it already passes correctly.
    bool with_phase_timers, py::dict* out_phase_timers, const py::kwargs& kw) {
    refuse_inert(kw);
    PhaseTimers timers;
    PhaseTimers* tmptr = with_phase_timers ? &timers : nullptr;

    // `field_allin` dispatches per POSE to a kernel with the per-atom bounds
    // machinery compiled out, after a conservative bounding-sphere test proves
    // the whole conformer is inside the lattice. Exactly equivalent when the
    // test passes and the general path when it does not -- so it is an
    // optimization, not an approximation, and it is gated on bit-identity.
    // `field_gridxf` folds the grid spacing and lattice origin into the pose
    // matrix and recovers the torque by an identity afterwards. Algebraically
    // exact, a REASSOCIATION in float -- judged on recall, not bit-identity.
    // The prepare-path pair is ON by default where it applies.
    //
    // A plain `= true` default would break every analytic call: the guard below
    // refuses these knobs when `field_h == 0`, and `field_h == 0` IS the
    // analytic path, which is the shipped CPU default. So the knobs are
    // tri-state -- **-1 means auto**, 0 and 1 are explicit user requests. Auto
    // resolves to on exactly where the register-resident vertex-field kernel is
    // the one running, and to off everywhere else.
    //
    // Resolved BEFORE any guard, and that ordering is load-bearing: -1 is
    // truthy in C++, so a guard reading `if (field_gridxf && ...)` sees auto
    // as a request and refuses it. That is exactly what happened first --
    // every analytic test threw on a knob nobody had asked for.
    //
    // This keeps the refuse-rather-than-ignore contract intact: an EXPLICIT 1
    // on an unsupported configuration still throws below, and only the
    // unspecified case is resolved silently -- which is the difference between
    // choosing a default and ignoring an instruction.
    //
    // Worth 1.044x and 1.048x at one thread across two independent sessions.
    // Each flag alone is a slight regression, 0.983-0.988x; only the
    // pair pays, because each alone leaves the other as the prepare chain's
    // critical path. So auto turns on both or neither.
    // Every condition either guard below tests, in one place. Missing one
    // makes auto select a configuration the next guard then refuses -- which
    // is what happened first time round: `field_pipeline` is required by the
    // gridxf guard at the top of this function and was not in this list, so
    // auto turned gridxf on for the unpipelined anchor and the anchor threw.
    // Every AVX2 block-prepare arm, not just the shipped one. The
    // variants forward ALLIN/GRIDXF to `prepare_block` and now fold the pose
    // as well, so they carry the default rather than silently running
    // without it -- which is what made "do the null arms compose?"
    // unanswerable: the arms were measured against a baseline the default had
    // already moved. `field_unroll != 0` is excluded because that path
    // prepares per sample and shares none of this.
    const bool field_arm_supported = (field_h > 0.0f) && field_width == 8 &&
                                     field_pipeline && field_unroll == 0;
    if (field_allin < 0) field_allin = field_arm_supported ? 1 : 0;
    if (field_gridxf < 0) field_gridxf = field_arm_supported ? 1 : 0;

    if (field_gridxf && !field_pipeline)
        throw std::invalid_argument(
            "field_gridxf is implemented on the pipelined register-resident "
            "kernel only: pass field_pipeline=True");
    // Refused rather than ignored, like every other knob here. The first
    // attempt at this guard was written against an anchor that no longer
    // matched and silently did nothing, so the knobs were accepted everywhere
    // and honored only on two of the three blocked dispatches -- a run could
    // ask for winner-only on the fp16 path and get full-rescore numbers under
    // a winner-only label. tests/test_cpu_field_knobs_are_honored.py pins both
    // halves: honored where supported, raised where not.
    // Now every AVX2 field arm, because they all route through the one
    // search harness (`optimize_pair_field_with`) and it is the harness that
    // implements winner-only. The scalar sampler does not: it calls
    // `optimize_pair_field`, which has no such parameter.
    // Prepared-screen default, with public all_start optout.
    // The established shared field harness already
    // evaluates the ACTUAL converged pose then analytically scores one winner.
    // Shape and shape-pose-color-rescore share this shape-only criterion.
    const bool winner_policy = resolve_final_start_policy(
        kw.contains("final_start_policy")
            ? kw["final_start_policy"].cast<std::string>()
            : "default");
    if (kw.contains("final_start_policy") &&
        kw["final_start_policy"].cast<std::string>() == "all_start")
        field_winner_only = false;
    if (winner_policy && kw.contains("prepared_query_fields") &&
        !kw["prepared_query_fields"].is_none() && field_h > 0.0f &&
        field_width == 8)
        field_winner_only = true;
    if ((field_winner_only || field_report_field_score) &&
        (!(field_h > 0.0f) || field_width != 8))
        throw std::invalid_argument(
            "field_winner_only / field_report_field_score need the AVX2 field "
            "sampler: field_h > 0 with field_width=8. The scalar sampler does "
            "not route through the shared search harness that implements "
            "them.");

    const bool field_variant_arm = field_fp16 || field_affine ||
                                   field_edgefield || field_pairaos ||
                                   field_coeff;
    if ((field_allin || field_gridxf) &&
        (!(field_h > 0.0f) || field_width != 8 || field_unroll != 0))
        throw std::invalid_argument(
            "field_allin/field_gridxf live in the blocked prepare: they need "
            "field_h > 0 with field_width=8 and field_unroll=0. The unrolled "
            "arm prepares one sample at a time and shares none of it.");
    // The variants get the PAIR, not the cross product. It was measured that
    // adding template instantiations to this dispatch moves the shipping
    // path's scores, so every extra arm has a real cost, and gridxf+allin
    // together is the default -- the only combination these
    // arms need to be measured against.
    if (field_variant_arm && (bool(field_allin) != bool(field_gridxf)))
        throw std::invalid_argument(
            "on the fp16/affine/edgefield/pairaos/coeff arms field_allin and "
            "field_gridxf must be set together: only the pair is "
            "instantiated, because each split costs the shipping path");
    if (field_variant_arm && field_allin && !field_pipeline)
        throw std::invalid_argument(
            "the variant arms carry field_allin/field_gridxf only with "
            "field_pipeline=true, which is the shipped configuration and the "
            "one instantiated");

    // `field_batched` was once ACCEPTED and silently ignored outside its one supported
    // configuration: the eight-pair block is guarded on gridxf && allin (and
    // the rest below), and anything else fell through to the per-pair path
    // with no error. A benchmark that set it without the pair measured
    // unbatched code under a batched label -- the same defect
    // tests/test_cpu_field_knobs_are_honored.py was written for.
    //
    // Partial same-query runs use the same optimizer as complete runs. The
    // census counts complete eight-pair blocks only; pairs - 8 * batched_blocks
    // counts pairs in partial runs, not scalar fallbacks.
    //
    // There is no `batched_fallback_pairs` counter: a per-pair increment
    // tripped the inlining cliff and moved the shipping path's scores on 76%
    // of pairs.
    if (field_batched &&
        (!(field_h > 0.0f) || field_width != 8 || field_unroll != 0 ||
         !field_pipeline || !field_gridxf || !field_allin || field_variant_arm))
        throw std::invalid_argument(
            "field_batched is implemented only for the shipped blocked "
            "configuration: field_h > 0 with field_width=8, field_unroll=0, "
            "field_pipeline=true, field_gridxf and field_allin both set, and "
            "none of field_fp16, field_affine, field_edgefield, field_pairaos "
            "or field_coeff. Refused rather than ignored -- it silently ran "
            "the per-pair path before, so a run could report unbatched numbers "
            "under a batched label.");

    // This dispatch sits on GCC's inlining cliff. Adding one parameter and a
    // few never-taken branches here once moved the SHIPPING path's scores by
    // up to 2.2e-3 in Tanimoto on 74% of pairs, with no change to the shipping
    // arithmetic whatsoever: a small change to this function changes what gets
    // inlined, and `-ffp-contract=fast` then fuses a different set of
    // multiply-adds. Over 25 optimizer steps that last-bit difference grows
    // into the third decimal of a Tanimoto.
    //
    // So any edit to this dispatch can move the numbers without touching the
    // arithmetic, and an "exactly equivalent" kernel change must be gated by
    // rebuilding with the kernel changed and the dispatch untouched.

    if (grain < -1 || grain > 2)
        throw std::invalid_argument(
            "grain must be -1 (the scalar reference kernel), 0 (lanes over "
            "query atoms), 1 (lanes over library atoms) or 2 (lanes over "
            "library conformers -- the GPU's grain)");
    if (width != 1 && width != 8 && width != 16)
        throw std::invalid_argument("width must be 1, 8 or 16");
    if (width == 16 && !have_avx512())
        throw std::invalid_argument(
            "width 16 needs AVX-512 and this CPU does not have it. Refused "
            "rather than served at a narrower width: a silently downgraded "
            "width makes a narrow kernel look like a slow wide one, which is "
            "the measurement this check exists to refuse.");
    if (cutoff_d2 < 0.0f)
        throw std::invalid_argument("cutoff_d2 must be >= 0 (0 disables it)");
    if (converge_tol > 0.0f && grain != 2)
        throw std::invalid_argument(
            "converge_tol is implemented in the conformer grain only "
            "(grain=2). Refused rather than ignored, for the reason below.");
    if (only_start >= 0 && grain != 2)
        throw std::invalid_argument(
            "only_start is implemented in the conformer grain only (grain=2). "
            "Refused rather than ignored: a start selector silently ignored "
            "would return the all-starts answer while a caller believed it had "
            "one start, which is the shape of defect this package refuses "
            "knobs for.");
    if (cutoff_d2 > 0.0f && grain != 2)
        throw std::invalid_argument(
            "the distance cutoff is implemented in the conformer grain only "
            "(grain=2). Refused rather than "
            "ignored: a cutoff silently not applied would read as a cutoff "
            "that cost nothing.");
    if (field_h < 0.0f)
        throw std::invalid_argument("field_h must be >= 0 (0 = analytic path)");
    if (field_pad <= 0.0f) throw std::invalid_argument("field_pad must be > 0");
    if (field_width != 1 && field_width != 8)
        throw std::invalid_argument(
            "field_width must be 1 (scalar) or 8 (paired AVX2: two samples per "
            "256-bit vector, channels in lanes)");
    // PAIRS in flight, so 2*field_unroll samples per iteration. The
    // field kernel is memory-bound -- full vectorization bought 1.03-1.21x --
    // and independent stencils in flight are what helps that, so this is a
    // first-class axis rather than a tuning afterthought.
    if (field_unroll < 0 || field_unroll > 4)
        throw std::invalid_argument(
            "field_unroll must be 0 (blocked prepare: 8 atoms transformed at "
            "once) or 1..4 (pairs in flight with a scalar prepare)");
    if (field_affine && (field_width != 8 || field_fp16))
        throw std::invalid_argument(
            "field_affine needs the vector sampler (field_width=8) and fp32 "
            "storage. Refused rather than ignored.");
    if (field_fp16 && field_width != 8)
        throw std::invalid_argument(
            "field_fp16 needs the vector sampler (field_width=8): the scalar "
            "one reads nodes directly and has no widening step. Refused rather "
            "than ignored -- an fp16 flag silently dropped would report fp32 "
            "bytes under an fp16 label, which is the comparison this arm is "
            "for.");
#if !defined(__AVX2__) && !defined(__aarch64__)
    if (field_width == 8)
        throw std::invalid_argument(
            "field_width 8 needs AVX2 and this binary was not built with it. "
            "Refused rather than served scalar: a silently narrowed width makes "
            "a scalar kernel look like a slow vector one.");
#endif
#if defined(__aarch64__)
    if (field_width == 8 &&
        (field_fp16 || field_affine || field_edgefield || field_pairaos ||
         field_coeff || field_report_field_score))
        throw std::invalid_argument(
            "AArch64 field supports retained FP32 stored-gradient layout only");
#endif
    if (field_h > 0.0f && grain >= 0)
        throw std::invalid_argument(
            "the field pose search is implemented on the scalar path only so "
            "far (grain=-1), its correctness pass. Refused rather "
            "than ignored: a field knob silently dropped would report the "
            "ANALYTIC path's rate and recall under a field label, which is the "
            "one comparison the field arms exist to make.");
    if (threads < 1) throw std::invalid_argument("threads must be >= 1");
    if (threads > 1 && with_census)
        throw std::invalid_argument(
            "the census counters are not thread-safe and are deliberately not "
            "made so: a rate taken with them on is a rate of the instrumented "
            "kernel, not the kernel, so the two were never meant to be "
            "combined. Count on one thread, time on many.");
    if (grain < 0 && poly)
        throw std::invalid_argument(
            "the reference kernel (grain=-1) is the anchor and uses libm's "
            "exp2; a polynomial grade would change the arithmetic the fixture "
            "names. Use a grain arm to measure a grade.");
    if (grain < 0 && width != 1)
        throw std::invalid_argument(
            "the reference kernel (grain=-1) exists only at width 1: it is "
            "the anchor, whose value is that it does not move");
#if !defined(__AVX2__)
    if (width == 8)
        throw std::invalid_argument(
            "this build has no AVX2, so width 8 cannot run. Refused rather "
            "than run at width 1 and reported as width 8 -- a stamp says what "
            "the binary CONTAINS, and a silently downgraded width would make a "
            "narrow kernel look like a slow wide one.");
#endif

    if (start_count(start_mode) < 0)
        throw std::invalid_argument(
            "start_mode must be 0, 1 or 2 (1, 4 or 10 starts)");
    if (steps < 0) throw std::invalid_argument("steps must be >= 0");

    auto qc = query_coords.unchecked<3>();    // [n_q, Lq, 3 or 4]
    auto lc = library_coords.unchecked<3>();  // [n_rows, Ll, 3 or 4]
    auto qn = query_n_atoms.unchecked<1>();
    auto lcnt = library_counts.unchecked<2>();
    // Library counts arrive as uint8, so every library conformer fits the
    // field path's ConformerSoA; that is the field path's whole atom-count
    // guard.
    static_assert(
        kConformerSoACapacity >
        std::numeric_limits<decltype(library_counts)::value_type>::max());
    auto pq = pair_query_row.unchecked<1>();
    auto pt = pair_library_row.unchecked<1>();

    const py::ssize_t n_pairs = pq.shape(0);
    const py::ssize_t q_stride = qc.shape(2);
    const py::ssize_t l_stride = lc.shape(2);

    // Coordinates arrive in Angstrom and are used in the reduced frame, so the
    // scale is applied once here per conformer rather than per atom pair. The
    // device folds it into the pose matrix instead; both put exactly one
    // multiply per coordinate between the corpus and the exponent, which is
    // what NUMERICS_POLICY section 4 asks for.
    std::vector<float> qbuf(size_t(qc.shape(0)) * qc.shape(1) * 3);
    std::vector<float> lbuf(size_t(lc.shape(0)) * lc.shape(1) * 3);
    {
        // "query_prep": the one-time per-call cost of staging both planes
        // into the reduced frame's scratch buffers, before any pair is
        // touched. Named for the query side because the query's scale
        // multiply lives here (a color decode on entry extends this same
        // bracket rather than opening a new one).
        ScopedPhaseTimer _t(tmptr ? &tmptr->query_prep_s : nullptr);
        for (py::ssize_t i = 0; i < qc.shape(0); ++i)
            for (py::ssize_t j = 0; j < qc.shape(1); ++j)
                for (int c = 0; c < 3; ++c)
                    qbuf[(size_t(i) * qc.shape(1) + j) * 3 + c] =
                        COORD_SCALE * qc(i, j, c);

        // The library is NOT pre-scaled: the pose matrix carries COORD_SCALE,
        // as it does on the device, so the library coordinate reaches the
        // exponent through exactly one multiply too.
        for (py::ssize_t i = 0; i < lc.shape(0); ++i)
            for (py::ssize_t j = 0; j < lc.shape(1); ++j)
                for (int c = 0; c < 3; ++c)
                    lbuf[(size_t(i) * lc.shape(1) + j) * 3 + c] = lc(i, j, c);
    }

    // Diagnostic only: preserve the internal winning-start index instead of
    // attempting to recover it from START_QUAT (the public rows fold it in).
    std::vector<int> winning_starts(with_census ? size_t(n_pairs) : 0, -1);
    OpCensus census;
    OpCensus* cptr = with_census ? &census : nullptr;

    // Self-overlaps once per conformer, not once per pair. The device hoists
    // this for the same reason; here it also keeps the census honest,
    // since a per-pair recompute would triple the exp2 count for no work.
    std::vector<float> s_aa(qc.shape(0)), s_bb(lcnt.shape(0));
    {
        ScopedPhaseTimer _t(tmptr ? &tmptr->query_self_overlap_s : nullptr);
        if (kw.contains("prepared_shape_query_self")) {
            const auto supplied =
                kw["prepared_shape_query_self"]
                    .cast<py::array_t<float, py::array::c_style>>();
            if (supplied.ndim() != 1 || supplied.size() != qc.shape(0))
                throw std::invalid_argument(
                    "prepared shape query self row count differs");
            for (py::ssize_t i = 0; i < supplied.size(); ++i) {
                const float value = supplied.data()[i];
                if (!std::isfinite(value) || value < 0.0f)
                    throw std::invalid_argument(
                        "prepared shape query self must be finite nonnegative");
                s_aa[i] = value;
            }
        } else {
            for (py::ssize_t i = 0; i < qc.shape(0); ++i) {
                Conformer a{&qbuf[size_t(i) * qc.shape(1) * 3], int(qn(i))};
                s_aa[i] = self_overlap(a, cptr);
            }
        }
    }
    // The library's self-overlap. This loop is SERIAL and O(n_B^2) per library
    // row, in a function that threads everything else -- measured at 31.4%
    // of the field path's total time, larger than the Tier-3 rescore and
    // invisible to every windowed measurement, because it runs
    // once per call and so cancels in t(steps=25) - t(steps=0).
    //
    // `library_vaa` skips it. The corpus already stores this quantity per
    // conformer, in cubic Angstrom with the volume prefactor, so
    //
    //     s_bb = vaa / SHAPE_PREFACTOR
    //
    // exactly in real arithmetic -- the exponentials are identical, which is
    // what COORD_SCALE exists for, and only the prefactor differs.
    //
    // **It is not bit-identical to recomputing, and the reason is not
    // rounding.** Coordinates are stored int16-quantized at the wire-step while
    // `vaa` was computed from the full-precision PREPARED coordinates before
    // quantization, so the two describe slightly different molecules:
    // 2.6e-5 relative median, 1.5e-4 max over 4,096 evaluation conformers. Passing
    // it therefore trades SELF-CONSISTENCY -- numerator and denominator from
    // the same coordinates -- for fidelity to the unquantized molecule, and it
    // is a declared numerics change that owes a recall regrade.
    // Opt-in for that reason, not for compatibility.
    {
        // "library_self_overlap": today the shape self-overlap S_BB only.
        // Named for what census item 4 calls the one-time LIBRARY SELF-COLOR
        // work, because this is that phase's shape-only precursor; S_color_BB
        // extends this exact bracket rather than opening a second one.
        ScopedPhaseTimer _t(tmptr ? &tmptr->library_self_overlap_s : nullptr);
        if (kw.contains("prepared_shape_library_self_reduced")) {
            if (library_vaa)
                throw std::invalid_argument(
                    "prepared shape library self and library_vaa cannot be combined");
            const auto supplied = kw["prepared_shape_library_self_reduced"]
                                      .cast<py::array_t<float, py::array::c_style>>();
            if (supplied.ndim() != 1 || supplied.size() != lcnt.shape(0))
                throw std::invalid_argument(
                    "prepared shape library self row count differs");
            for (py::ssize_t i = 0; i < supplied.size(); ++i) {
                const float value = supplied.data()[i];
                if (!std::isfinite(value) || value < 0.0f)
                    throw std::invalid_argument(
                        "prepared shape library self must be finite nonnegative");
                s_bb[i] = value;
            }
        } else if (library_vaa) {
            auto v = library_vaa->unchecked<1>();
            if (v.shape(0) != lcnt.shape(0))
                throw std::invalid_argument(
                    "library_vaa must have one entry per library row");
            const float inv = 1.0f / SHAPE_PREFACTOR;
            for (py::ssize_t i = 0; i < lcnt.shape(0); ++i)
                s_bb[i] = float(v(i)) * inv;
        } else {
            // Computed on a scaled copy rather than by scaling lbuf, which the
            // pose path needs unscaled.
            std::vector<float> tmp;
            for (py::ssize_t i = 0; i < lcnt.shape(0); ++i) {
                const int n = int(lcnt(i, 0));
                tmp.assign(size_t(n) * 3, 0.0f);
                for (int j = 0; j < n; ++j)
                    for (int c = 0; c < 3; ++c)
                        tmp[size_t(j) * 3 + c] =
                            COORD_SCALE *
                            lbuf[(size_t(i) * lc.shape(1) + j) * 3 + c];
                Conformer bb{tmp.data(), n};
                s_bb[i] = self_overlap(bb, cptr);
            }
        }
    }

    auto out = py::array_t<float>(
        {py::ssize_t(n_pairs), py::ssize_t(N_SCORE_COLUMNS)});
    auto o = out.mutable_unchecked<2>();

    // lr_t in scaled Angstrom, exactly as the CUDA launchers do on entry.
    const float lr_t_scaled = lr_t * COORD_SCALE;

    // The query's transpose is hoisted OUT of the pair loop, which is the whole
    // structural advantage the query-atom grain has: n_A is constant for the
    // entire screen, so the repack and the padding decision are paid once per
    // query rather than once per pair. Built for every grain and width,
    // including width 1, so the timing comparison between grains is not also a
    // comparison of who pays for a transpose.
    std::vector<SoaConformer> q_soa(size_t(qc.shape(0)));
    const int lanes = (width == 16) ? 16 : ((width == 8) ? 8 : 1);
    for (py::ssize_t i = 0; i < qc.shape(0); ++i) {
        Conformer a{&qbuf[size_t(i) * qc.shape(1) * 3], int(qn(i))};
        q_soa[size_t(i)].build(a, lanes);
    }

    // --- grain 2: lanes over library conformers -----------------------------
    //
    // A different loop shape from the atom grains, because a lane is a PAIR:
    // pairs are processed in blocks of W that share a query row, so the query
    // can be broadcast and the W poses evolve independently in the lanes.
    //
    // Blocks are cut at query boundaries rather than every W pairs. A block
    // spanning two queries would need two broadcast queries in one register
    // set, which is the one thing this grain cannot express -- and silently
    // scoring against the wrong query is the class of defect that once cost a
    // measurement (layout.py's docstring: nothing checks at run time that the
    // producer and the kernel agree). A short final block is masked, never
    // branched.
    // A RANGE, not an early return. The first version returned from here, which
    // silently bypassed two things the rest of the function does: the threading
    // path below, so `threads` was ignored for the RECOMMENDED grain and every
    // request for 24 threads ran on one; and the census population at the end,
    // so `score_pairs_with_census(grain=2)` handed back an empty dict. An
    // ignored knob is a silently wrong answer -- the rule `refuse_inert` exists
    // to enforce for GPU knobs -- and it was ignored here by control flow.
    // Block boundaries, computed ONCE, so that threading partitions blocks
    // rather than pairs.
    //
    // The first version snapped each thread's pair range forward to a QUERY
    // boundary, to stop a block straddling two threads. That is correct and
    // useless: this fixture has four queries, so `snap` could produce at most
    // four non-empty ranges and every thread past the fourth got nothing. The
    // grain appeared to ceiling at four threads and ~275k start-pairs/s no
    // matter how many cores were added, which read as a scaling property of the
    // kernel and was an artifact of the partition.
    //
    // Blocks are the natural unit here, so they are the unit that gets split.
    std::vector<py::ssize_t> block_start;
    if (grain == 2) {
        const int W = (width == 16) ? 16 : ((width == 8) ? 8 : 1);
        py::ssize_t r = 0;
        while (r < n_pairs) {
            block_start.push_back(r);
            const int qi0 = pq(r);
            py::ssize_t e = r;
            while (e < n_pairs && pq(e) == qi0 && (e - r) < W)
                ++e;
            r = e;
        }
        block_start.push_back(n_pairs);
    }

    // `lo`/`hi` are BLOCK indices here, not pair indices.
    auto run_conformer_range = [&](py::ssize_t lo, py::ssize_t hi) {
        const int W = (width == 16) ? 16 : ((width == 8) ? 8 : 1);
        for (py::ssize_t bi = lo; bi < hi; ++bi) {
            py::ssize_t r = block_start[size_t(bi)];
            const py::ssize_t e_lim = block_start[size_t(bi) + 1];
            (void)e_lim;
            const int qi = pq(r);
            const py::ssize_t e = e_lim;
            const int n = int(e - r);
            // Thread-local and reused: see grain_conformer.hpp. Allocating
            // these per block cost this grain its threading scalability.
            thread_local std::vector<Conformer> confs;
            thread_local std::vector<float> sbb;
            thread_local LaneBlock blk;
            confs.clear();
            sbb.clear();
            confs.reserve(size_t(n));
            sbb.reserve(size_t(n));
            for (py::ssize_t k = r; k < e; ++k) {
                const int li = pt(k);
                confs.push_back(Conformer{&lbuf[size_t(li) * lc.shape(1) * 3],
                                          int(lcnt(li, 0))});
                sbb.push_back(s_bb[li]);
            }
            blk.build(confs.data(), sbb.data(), n, W);
            Conformer a{&qbuf[size_t(qi) * qc.shape(1) * 3], int(qn(qi))};
            // static_cast, not `size_t(W)`: the latter is a most-vexing
            // parse and declares a FUNCTION.
            thread_local std::vector<PairResult> block_res;
            block_res.assign(static_cast<size_t>(W), PairResult{});
#if defined(__AVX2__)
            if (W == 8) {
                if (poly)
                    optimize_block_conformer<r3::simd::Avx2Tag, true>(
                        r3::simd::Avx2Tag{}, a, blk, s_aa[qi], start_mode,
                        steps, lr_q, lr_t_scaled, cutoff_d2, converge_tol,
                        only_start, block_res.data(), cptr);
                else
                    optimize_block_conformer<r3::simd::Avx2Tag, false>(
                        r3::simd::Avx2Tag{}, a, blk, s_aa[qi], start_mode,
                        steps, lr_q, lr_t_scaled, cutoff_d2, converge_tol,
                        only_start, block_res.data(), cptr);
            } else
#endif
                if (W == 16) {
                run_avx512_block(poly, a, blk, s_aa[qi], start_mode, steps,
                                 lr_q, lr_t_scaled, cutoff_d2, converge_tol,
                                 only_start, block_res.data(), cptr);
            } else {
                if (poly)
                    optimize_block_conformer<r3::simd::ScalarTag, true>(
                        r3::simd::ScalarTag{}, a, blk, s_aa[qi], start_mode,
                        steps, lr_q, lr_t_scaled, cutoff_d2, converge_tol,
                        only_start, block_res.data(), cptr);
                else
                    optimize_block_conformer<r3::simd::ScalarTag, false>(
                        r3::simd::ScalarTag{}, a, blk, s_aa[qi], start_mode,
                        steps, lr_q, lr_t_scaled, cutoff_d2, converge_tol,
                        only_start, block_res.data(), cptr);
            }
            for (int k = 0; k < n; ++k) {
                const py::ssize_t row = r + k;
                for (int c = 0; c < N_SCORE_COLUMNS; ++c)
                    o(row, c) = 0.0f;
                {
                    const PairResult& R = block_res[size_t(k)];
                    float t_rep = R.tanimoto_shape;
                    if (field_report_field_score && R.s_ab_field >= 0.0f) {
                        const float dn = R.s_aa + R.s_bb - R.s_ab_field;
                        t_rep = (dn > 0.0f) ? (R.s_ab_field / dn) : 0.0f;
                    }
                    o(row, 0) = t_rep;
                    o(row, 1) = t_rep;
                }
                if (with_census) winning_starts[size_t(row)] = block_res[size_t(k)].best_start;
                o(row, 3) = SHAPE_PREFACTOR * block_res[size_t(k)].s_ab;
                o(row, 5) = SHAPE_PREFACTOR * block_res[size_t(k)].s_aa;
                o(row, 6) = SHAPE_PREFACTOR * block_res[size_t(k)].s_bb;
                for (int c = 0; c < 4; ++c)
                    o(row, 9 + c) = block_res[size_t(k)].pose_q[c];
                for (int c = 0; c < 3; ++c)
                    o(row, 13 + c) = block_res[size_t(k)].pose_t[c];
                // START_QUAT (columns 16..19) is the IDENTITY, and writing only
                // the zeroed row leaves it (0,0,0,0) -- not a unit quaternion,
                // which `numerics.check_result_invariants` checks and which the
                // ranked-output gate cannot see because it reads column 1.
                o(row, 16) = 1.0f;
            }
        }
    };

    // One field per QUERY, built once and shared by every pair that
    // names it -- which is the field path's whole economy, and the reason the
    // GPU builds it per query too. Read-only afterwards, so the threads below
    // share it without arbitration.
    //
    // "field_construction" spans this whole section (every variant array
    // below), not just the primary fp32 build, so a call that turns on
    // fp16/edge/affine/coeff sees their cost too. Manual start/stop rather
    // than `ScopedPhaseTimer`: the vectors declared in here must outlive this
    // point (the per-pair loop below reads them), so an RAII bracket cannot
    // be a nested `{ }` scope the way the two brackets above are.
    const auto _field_t0 = tmptr ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    const auto prepared_fields =
        kw.contains("prepared_query_fields")
            ? kw["prepared_query_fields"]
                  .cast<PreparedShapeEvaluatorSetHandle>()
            : PreparedShapeEvaluatorSetHandle{};
    if (prepared_fields) {
        if (field_h <= 0.0f || field_width != 8 || field_unroll != 0 ||
            !field_pipeline || field_fp16 || field_affine || field_edge ||
            field_edgefield || field_pairaos || field_coeff)
            throw std::invalid_argument(
                "prepared_query_fields requires FP32 pipelined width8/unroll0 fields");
        auto wanted = prepared_fields->config;
        wanted.backend = ShapeBackend::PipelinedField;
        wanted.grain = ShapeGrain::LibraryAtom;
        wanted.width = 8;
        wanted.field_spacing_angstrom = field_h;
        wanted.field_padding_angstrom = field_pad;
        if (!prepared_fields->matches(wanted, qbuf.data(), int(qc.shape(1)),
                                      query_n_atoms.data(),
                                      size_t(qc.shape(0))))
            throw std::invalid_argument(
                "prepared_query_fields binding mismatch");
    }
    std::vector<FieldNaive32> owned_fields;

    struct FieldViews {
        std::vector<const FieldNaive32*> grids;

        size_t size() const noexcept {
            return grids.size();
        }

        const FieldNaive32& operator[](size_t i) const {
            return *grids[i];
        }
    } fields;

    if (prepared_fields) {
        for (const auto& evaluator : prepared_fields->evaluators)
            fields.grids.push_back(&shape_field_grid(*evaluator));
    }
    if (field_h > 0.0f && !prepared_fields) {
        // SERIAL, deliberately, and it is a measured Amdahl cap rather than an
        // oversight. At library 64 and h = 0.6 this build
        // is 84% of the screen and holds 24-thread scaling to 1.15x; the GPU
        // builds its field in parallel (one thread per node) and this should
        // too.
        //
        // **A parallel version was written, measured at 12.99-13.81x scaling
        // against 2.20-4.76x, and then REVERTED**, because it made the result
        // depend on the thread count: bit-identical across 1..24 threads with
        // this serial build, differing by 2.6e-5 in Tanimoto with the parallel
        // one, on BOTH samplers. The cause is not yet identified -- an isolated
        // C++ test of the same plane-chunking on one thread reproduces the
        // serial field bit for bit, so it is not the chunking as such. A field
        // build is a pure map over lattice nodes and has no license to depend
        // on how it was scheduled; until that is explained rather than
        // patched, the slow build is the correct one to ship.
        owned_fields.resize(size_t(qc.shape(0)));
        for (py::ssize_t i = 0; i < qc.shape(0); ++i) {
            const float* qxyz = &qbuf[size_t(i) * qc.shape(1) * 3];
            build_shape_field_grid(qxyz, int(qn(i)), field_h, field_pad,
                                   owned_fields[size_t(i)]);
            fields.grids.push_back(&owned_fields[size_t(i)]);
        }
    }
    // fp16 STORAGE, fp32 arithmetic. Quantized from the fp32 build rather than
    // built in fp16, so the only difference between the two arms is the
    // storage -- the build arithmetic is identical and cannot confound the
    // comparison.
    // B3: the x-affine form, built from the same fp32 field so the only
    // difference between the arms is the representation.
    // B2: duplicated 32-byte edge records, aligned by construction.
    // Family C: per-cell trilinear coefficients, ~8x the vertex field.
    std::vector<FieldCoeff32> fields_coeff;
    if (field_h > 0.0f && field_coeff) {
        fields_coeff.resize(fields.size());
        for (size_t i = 0; i < fields.size(); ++i)
            make_coeff_field(fields[i], fields_coeff[i]);
    }
    std::vector<FieldEdge32> fields_edge;
    if (field_h > 0.0f && field_edgefield) {
        fields_edge.resize(fields.size());
        for (size_t i = 0; i < fields.size(); ++i)
            make_edge_field(fields[i], fields_edge[i]);
    }
    std::vector<FieldAffine32> fields_aff;
    if (field_h > 0.0f && field_affine) {
        fields_aff.resize(fields.size());
        for (size_t i = 0; i < fields.size(); ++i)
            make_affine(fields[i], fields_aff[i]);
    }
    std::vector<FieldNaive16> fields16;
    if (field_h > 0.0f && field_fp16) {
        fields16.resize(fields.size());
        for (size_t i = 0; i < fields.size(); ++i)
            quantize_field(fields[i], fields16[i]);
    }
    if (tmptr)
        tmptr->field_construction_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          _field_t0)
                .count();

    // The pair loop, as a range so it can be run on one thread or many.
    //
    // Each pair is independent and writes only its own output row, so there is
    // no sharing to arbitrate and no reduction at the end -- which is why this
    // needs `std::thread` and not OpenMP, and why adding it costs the package
    // no new dependency. `roshambo3` has seven runtime dependencies and
    // declined to make a SIMD library the eighth; the same argument applies
    // here with less at stake.
    //
    // **The census is refused above rather than made thread-safe.** Counting
    // and timing were never meant to happen in the same run.
    auto run_range = [&](py::ssize_t lo, py::ssize_t hi) {
        // One writer for both the per-pair and the batched path, so a result
        // cannot be emitted two slightly different ways.
        auto emit = [&](py::ssize_t row, const PairResult& res) {
            for (int c = 0; c < N_SCORE_COLUMNS; ++c)
                o(row, c) = 0.0f;
            // TANIMOTO_SHAPE carries the FIELD's own estimate when asked, so a
            // caller can rank on it. Default is the exact
            // rescored value, which is what every shipping path reads.
            float t_rep = res.tanimoto_shape;
            if (field_report_field_score && res.s_ab_field >= 0.0f) {
                const float dn = res.s_aa + res.s_bb - res.s_ab_field;
                t_rep = (dn > 0.0f) ? (res.s_ab_field / dn) : 0.0f;
            }
            o(row, 0) = t_rep;  // combination == shape, no color
            o(row, 1) = t_rep;
            if (with_census) winning_starts[size_t(row)] = res.best_start;
            o(row, 3) = SHAPE_PREFACTOR * res.s_ab;
            o(row, 5) = SHAPE_PREFACTOR * res.s_aa;
            o(row, 6) = SHAPE_PREFACTOR * res.s_bb;
            for (int c = 0; c < 4; ++c)
                o(row, 9 + c) = res.pose_q[c];
            for (int c = 0; c < 3; ++c)
                o(row, 13 + c) = res.pose_t[c];
            // cols 16-19: the start-mode quaternion. Identity, as on the field
            // path, because the start rotation is folded into the reported pose.
            o(row, 16) = 1.0f;
        };

        for (py::ssize_t r = lo; r < hi; ++r) {
            const int qi = pq(r), li = pt(r);
#if defined(__AVX2__)
            // Complete and partial same-query runs use the same update
            // arithmetic. The per-pair fallback at block/worker/query boundaries
            // changed score and pose bits. Inactive SIMD lanes are
            // initialized; no target or result rows are padded.
            if (field_batched && field_width == 8 && field_h > 0.0f &&
                field_gridxf && field_allin && field_pipeline &&
                field_unroll == 0 && !field_fp16 && !field_affine &&
                !field_edgefield && !field_pairaos && !field_coeff) {
                int lanes = 1;
                while (lanes < 8 && r + lanes < hi && pq(r + lanes) == qi)
                    ++lanes;
                Conformer a8{&qbuf[size_t(qi) * qc.shape(1) * 3],
                             int(qn(qi))};
                Conformer bs[8];
                const Conformer* bp[8];
                float sbb8[8];
                for (int j = 0; j < lanes; ++j) {
                    const int lj = pt(r + j);
                    bs[j] = Conformer{&lbuf[size_t(lj) * lc.shape(1) * 3],
                                      int(lcnt(lj, 0))};
                    bp[j] = &bs[j];
                    sbb8[j] = s_bb[lj];
                }
                PairResult rr[8];
                if (field_edge)
                    optimize_pairs_field_blocked<true>(
                        a8, bp, lanes, fields[size_t(qi)], s_aa[qi], sbb8,
                        start_mode, steps, lr_q, lr_t_scaled, cptr, rr,
                        field_winner_only, field_report_field_score);
                else
                    optimize_pairs_field_blocked<false>(
                        a8, bp, lanes, fields[size_t(qi)], s_aa[qi], sbb8,
                        start_mode, steps, lr_q, lr_t_scaled, cptr, rr,
                        field_winner_only, field_report_field_score);
                for (int j = 0; j < lanes; ++j)
                    emit(r + j, rr[j]);
                if (cptr && lanes == 8) ++cptr->batched_blocks;
                r += lanes - 1;  // the loop increments the final row
                continue;
            }
#endif
            // NOTE: there was a `++cptr->batched_fallback_pairs` here, in the
            // per-pair loop. It moved the SHIPPING path's scores -- 76% of pairs,
            // max |dT| 7.4e-05 -- with the arithmetic untouched and the counter
            // never incremented in a normal run, because `cptr` is null. That is
            // the inlining cliff, hit again by a census counter. The quantity
            // used to be derivable as pairs - 8 * batched_blocks. Eligible
            // partial runs now use batched arithmetic too.
            Conformer a{&qbuf[size_t(qi) * qc.shape(1) * 3], int(qn(qi))};
            Conformer b{&lbuf[size_t(li) * lc.shape(1) * 3], int(lcnt(li, 0))};
            PairResult res;
            if (field_h > 0.0f) {
#if defined(__aarch64__)
                if (field_width == 8) {
                    // The pipelined NEON gradient the combo route's shape
                    // term uses, not the AoS one: the optimizer hands
                    // every step this pair's library conformer, so its SoA
                    // record is built once per pair and bound to the lambda
                    // by reference. Thread-local, as the record is 3 KB and
                    // build() rewrites every slot a reader touches.
                    thread_local ConformerSoA b_soa;
                    b_soa.build(b);
                    res = optimize_pair_field_with(
                        a, b, fields[size_t(qi)], s_aa[qi], s_bb[li],
                        start_mode, steps, lr_q, lr_t_scaled, cptr,
                        [&soa = b_soa](const FieldNaive32& field,
                                       const Conformer& target,
                                       const float m[9], const float t[3],
                                       float norm, float g[7], float* score,
                                       OpCensus* counters) {
                            field_gradient_neon_pipelined(field, soa, m, t,
                                                          norm, g, score);
                            if (counters)
                                counters->field_samples += target.n_atoms;
                        },
                        field_winner_only, field_report_field_score);
                } else
#endif
#if defined(__AVX2__)
                    if (field_width == 8) {
#define R3_FIELD(U, FLD)                                                     \
    optimize_pair_field_avx2<U>(a, b, (FLD)[size_t(qi)], s_aa[qi], s_bb[li], \
                                start_mode, steps, lr_q, lr_t_scaled, cptr,  \
                                field_winner_only, field_report_field_score)
                    // field_unroll == 0 selects the BLOCKED prepare: eight atoms
                    // transformed, mapped and bounds-tested at once. The ablation
                    // put the scalar prepare at 63% of a sample and the
                    // interpolation at 1%, so this is the arm aimed at the part
                    // that costs.
                    // The variant samplers. Each now forwards winner-only and,
                    // where it has the blocked prepare, the default gridxf+allin
                    // pair -- so "does this null arm compose with the stack?" is
                    // a question that can be asked. Before this it could not be:
                    // the knobs were refused, and the auto-default silently
                    // turned the pair OFF for exactly these arms.
#define R3_VARGS(FLD)                                                     \
    a, b, (FLD)[size_t(qi)], s_aa[qi], s_bb[li], start_mode, steps, lr_q, \
        lr_t_scaled, cptr, field_winner_only, field_report_field_score
                    if (field_coeff) {
                        if (field_gridxf)
                            res = field_edge
                                      ? optimize_pair_field_coeff<true, true,
                                                                  true, true>(
                                            R3_VARGS(fields_coeff))
                                      : optimize_pair_field_coeff<true, false,
                                                                  true, true>(
                                            R3_VARGS(fields_coeff));
                        else
                            res = field_pipeline
                                      ? (field_edge
                                             ? optimize_pair_field_coeff<true,
                                                                         true>(
                                                   R3_VARGS(fields_coeff))
                                             : optimize_pair_field_coeff<true,
                                                                         false>(
                                                   R3_VARGS(fields_coeff)))
                                      : optimize_pair_field_coeff<false>(
                                            R3_VARGS(fields_coeff));
                    } else if (field_pairaos) {
                        if (field_gridxf)
                            res = optimize_pair_field_pairaos<true, true, true>(
                                R3_VARGS(fields));
                        else
                            res = field_pipeline
                                      ? optimize_pair_field_pairaos<true>(
                                            R3_VARGS(fields))
                                      : optimize_pair_field_pairaos<false>(
                                            R3_VARGS(fields));
                    } else if (field_edgefield) {
                        if (field_gridxf)
                            res =
                                optimize_pair_field_edgefield<true, true, true>(
                                    R3_VARGS(fields_edge));
                        else
                            res = field_pipeline
                                      ? optimize_pair_field_edgefield<true>(
                                            R3_VARGS(fields_edge))
                                      : optimize_pair_field_edgefield<false>(
                                            R3_VARGS(fields_edge));
                    } else if (field_affine) {
                        if (field_gridxf)
                            res = optimize_pair_field_affine<true, true, true>(
                                R3_VARGS(fields_aff));
                        else
                            res = field_pipeline
                                      ? optimize_pair_field_affine<true>(
                                            R3_VARGS(fields_aff))
                                      : optimize_pair_field_affine<false>(
                                            R3_VARGS(fields_aff));
                    } else if (field_unroll == 0) {
#define R3_BLK_S(EDGE, PIPE, STAGE, FLD)                                      \
    optimize_pair_field_blocked<EDGE, PIPE, STAGE, false, false>(             \
        a, b, (FLD)[size_t(qi)], s_aa[qi], s_bb[li], start_mode, steps, lr_q, \
        lr_t_scaled, cptr, field_winner_only, field_report_field_score)
#define R3_BLK_A(EDGE, PIPE, FLD)                                             \
    optimize_pair_field_blocked<EDGE, PIPE, 0, true, false>(                  \
        a, b, (FLD)[size_t(qi)], s_aa[qi], s_bb[li], start_mode, steps, lr_q, \
        lr_t_scaled, cptr, field_winner_only, field_report_field_score)
#define R3_BLK(EDGE, PIPE, FLD) R3_BLK_S(EDGE, PIPE, 0, FLD)
#define R3_BLK_G(EDGE, PIPE, ALLIN, FLD)                                      \
    optimize_pair_field_blocked<EDGE, PIPE, 0, ALLIN, true>(                  \
        a, b, (FLD)[size_t(qi)], s_aa[qi], s_bb[li], start_mode, steps, lr_q, \
        lr_t_scaled, cptr, field_winner_only, field_report_field_score)
                                if (field_gridxf && field_fp16)
                                // fp16 storage under the gridxf default. The branch
                                // below reads `fields`, the fp32 array -- reaching it
                                // with field_fp16 set would score the wrong field and
                                // report it under an fp16 label. ALLIN only: the pair
                                // is the arm, and the guard above enforces it.
                                res =
                                    field_edge
                                        ? R3_BLK_G(true, true, true, fields16)
                                        : R3_BLK_G(false, true, true, fields16);
                            else if (field_gridxf)
                                res =
                                    field_allin
                                        ? (field_edge ? R3_BLK_G(true, true,
                                                                 true, fields)
                                                      : R3_BLK_G(false, true,
                                                                 true, fields))
                                        : (field_edge
                                               ? R3_BLK_G(true, true, false,
                                                          fields)
                                               : R3_BLK_G(false, true, false,
                                                          fields));
                            else if (field_allin)
                                res =
                                    field_edge
                                        ? (field_pipeline
                                               ? R3_BLK_A(true, true, fields)
                                               : R3_BLK_A(true, false, fields))
                                        : (field_pipeline
                                               ? R3_BLK_A(false, true, fields)
                                               : R3_BLK_A(false, false,
                                                          fields));
                            else if (field_fp16)
                                res =
                                    field_edge
                                        ? (field_pipeline
                                               ? R3_BLK(true, true, fields16)
                                               : R3_BLK(true, false, fields16))
                                        : (field_pipeline
                                               ? R3_BLK(false, true, fields16)
                                               : R3_BLK(false, false,
                                                        fields16));
                            else
                                res =
                                    field_edge
                                        ? (field_pipeline
                                               ? R3_BLK(true, true, fields)
                                               : R3_BLK(true, false, fields))
                                        : (field_pipeline
                                               ? R3_BLK(false, true, fields)
                                               : R3_BLK(false, false, fields));
#undef R3_BLK
#undef R3_BLK_A
#undef R3_BLK_G
#undef R3_BLK_S
                    } else if (field_fp16) {
                        switch (field_unroll) {
                        case 1:
                            res = R3_FIELD(1, fields16);
                            break;
                        case 2:
                            res = R3_FIELD(2, fields16);
                            break;
                        case 3:
                            res = R3_FIELD(3, fields16);
                            break;
                        default:
                            res = R3_FIELD(4, fields16);
                            break;
                        }
                    } else {
                        switch (field_unroll) {
                        case 1:
                            res = R3_FIELD(1, fields);
                            break;
                        case 2:
                            res = R3_FIELD(2, fields);
                            break;
                        case 3:
                            res = R3_FIELD(3, fields);
                            break;
                        default:
                            res = R3_FIELD(4, fields);
                            break;
                        }
                    }
#undef R3_FIELD
#undef R3_VARGS
                } else
#endif
                    res = optimize_pair_field(a, b, fields[size_t(qi)],
                                              s_aa[qi], s_bb[li], start_mode,
                                              steps, lr_q, lr_t_scaled, cptr);
            } else if (grain < 0) {
                res = optimize_pair(a, b, s_aa[qi], s_bb[li], start_mode, steps,
                                    lr_q, lr_t_scaled, cptr);
            } else {
                const Grain gr =
                    (grain == 0) ? Grain::QUERY_ATOM : Grain::LIBRARY_ATOM;
#define R3_RUN(TAG, POLY)                                                \
    optimize_pair_grain<TAG, POLY>(TAG{}, gr, a, b, s_aa[qi], s_bb[li],  \
                                   start_mode, steps, lr_q, lr_t_scaled, \
                                   &q_soa[size_t(qi)], cptr)
                if (width == 16) {
                    res = run_avx512_grain(poly, gr, a, b, s_aa[qi], s_bb[li],
                                           start_mode, steps, lr_q, lr_t_scaled,
                                           &q_soa[size_t(qi)], cptr);
                } else
#if defined(__AVX2__)
                    if (width == 8) {
                    res = poly ? R3_RUN(r3::simd::Avx2Tag, true)
                               : R3_RUN(r3::simd::Avx2Tag, false);
                } else
#endif
                {
                    res = poly ? R3_RUN(r3::simd::ScalarTag, true)
                               : R3_RUN(r3::simd::ScalarTag, false);
                }
#undef R3_RUN
            }
            emit(r, res);
        }
    };  // run_range

    // One threading path for every grain. `run_any` picks the range function;
    // the split, the GIL release and the join are shared, so a grain cannot
    // silently opt out of threading by having its own loop.
    // The index space differs by grain -- pairs for the atom grains, blocks for
    // the conformer grain -- so the partition is computed over `n_units` and
    // each grain interprets its own range.
    const py::ssize_t n_units =
        (grain == 2) ? py::ssize_t(block_start.size()) - 1 : n_pairs;
    auto run_any = [&](py::ssize_t lo, py::ssize_t hi) {
        if (grain == 2)
            run_conformer_range(lo, hi);
        else
            run_range(lo, hi);
    };

    // "shape_search": the whole per-pair (or per-block, grain 2) dispatch,
    // whichever grain and however many threads. This bracket is OUTSIDE
    // `run_range`/`run_conformer_range`/`run_any` -- it times the CALL, never
    // a line inside their bodies -- for the reason this file's header comment
    // on `PhaseTimers` gives in full: a single edit inside those lambdas has
    // previously moved the shipping path's scores with the arithmetic
    // untouched, and no timer measurement is worth risking that.
    const auto _search_t0 = tmptr ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    {
        // Both the serial inner call and the threaded call perform native work.
        py::gil_scoped_release nogil;
        if (threads == 1) {
            run_any(0, n_units);
        } else {
            // The GIL is released for the duration: without it the extra threads
            // would serialize on it and the scaling curve would measure the
            // interpreter rather than the kernel.
            const int n = std::min<int>(threads, int(n_units));
            std::vector<std::thread> pool;
            pool.reserve(size_t(n));
            // Contiguous blocks rather than a strided or dynamic split. The pairs
            // in one block share a query row in this corpus's ordering, so a
            // contiguous block reuses the query that is already in L1 -- and the
            // per-pair cost varies little enough (one group, one padded width)
            // that static balance is not a problem here. It would be across
            // buckets.
            // PIN EACH THREAD TO ONE CPU, taken in order from the process's own
            // affinity mask.
            //
            // Leaving placement to the scheduler under a process-wide mask is what
            // produced a non-monotonic scaling curve: with 8 threads on a
            // 6+2 mask spanning a CCD boundary, threads observably stacked 4-6 deep
            // on two CPUs while others idled, and efficiency fell to 0.43 against
            // 0.91 for the same 8 threads spread across the four CCDs. The first
            // diagnosis blamed the static partition and proposed a dynamic one;
            // that is the wrong direction -- the partition is even, and it is the
            // PLACEMENT that is not. 1:1 pinning is what removes it.
            const auto cpus = r3::cpu::platform::allowed_cpus();
            const auto qos = r3::cpu::platform::current_qos();
            std::vector<std::exception_ptr> worker_errors(
                static_cast<size_t>(n));
            for (int k = 0; k < n; ++k) {
                const py::ssize_t lo = n_units * k / n;
                const py::ssize_t hi = n_units * (k + 1) / n;
                if (lo >= hi) continue;
                const int cpu =
                    cpus.empty() ? -1 : cpus[size_t(k) % cpus.size()];
                pool.emplace_back([&, lo, hi, cpu, qos, k] {
                    try {
                        if (cpu >= 0)
                            r3::cpu::platform::bind_current_thread(cpu);
                        if (qos != "unsupported" && qos != "unspecified")
                            r3::cpu::platform::set_qos(qos);
                        run_any(lo, hi);
                    } catch (...) {
                        worker_errors[size_t(k)] = std::current_exception();
                    }
                });
            }
            for (auto& th : pool)
                th.join();
            for (const auto& error : worker_errors)
                if (error) std::rethrow_exception(error);
        }
    }
    if (tmptr)
        tmptr->shape_search_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          _search_t0)
                .count();

    if (out_phase_timers) {
        (*out_phase_timers)["query_prep"] = timers.query_prep_s;
        (*out_phase_timers)["query_self_overlap"] = timers.query_self_overlap_s;
        (*out_phase_timers)["library_self_overlap"] =
            timers.library_self_overlap_s;
        (*out_phase_timers)["field_construction"] = timers.field_construction_s;
        (*out_phase_timers)["shape_search"] = timers.shape_search_s;
        // Reserved: always 0.0 until a color kernel is added here to price,
        // or (for "reduction") this seam grows a device-side one. See the
        // `PhaseTimers` comment at this file's top for why.
        (*out_phase_timers)["analytic_color"] = timers.analytic_color_s;
        (*out_phase_timers)["field_color"] = timers.field_color_s;
        (*out_phase_timers)["exact_final_rescore"] =
            timers.exact_final_rescore_s;
        (*out_phase_timers)["reduction"] = timers.reduction_s;
    }

    if (out_census) {
        (*out_census)["winning_starts"] = winning_starts;
        (*out_census)["exp2_calls"] = census.exp2_calls;
        (*out_census)["pair_terms_optimizer"] = census.pair_terms_optimizer;
        (*out_census)["pair_terms_rescore"] = census.pair_terms_rescore;
        (*out_census)["optimizer_steps"] = census.optimizer_steps;
        // The field path's unit of work, O(n_B) per step against the
        // analytic path's O(n_A n_B). Counted rather than derived.
        (*out_census)["field_samples"] = census.field_samples;
        (*out_census)["starts"] = census.starts;
        (*out_census)["pairs"] = census.pairs;
        (*out_census)["self_overlap_terms"] = census.self_overlap_terms;
        (*out_census)["cutoff_skips"] = census.cutoff_skips;
        (*out_census)["cutoff_visits"] = census.cutoff_visits;
        (*out_census)["allin_poses"] = census.allin_poses;
        (*out_census)["batched_blocks"] = census.batched_blocks;
        (*out_census)["allin_pose_visits"] = census.allin_pose_visits;
        (*out_census)["converged_blocks"] = census.converged_blocks;
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_roshambo3_cpu, m) {
#if defined(__clang__)
    m.attr("__compiler__") = "clang " __clang_version__;
#else
    m.attr("__compiler__") = "GCC " __VERSION__;
#endif
    r3::cpu::platform::bind_api(m);
    r3::cpu::bind_scalar_color(m);
    r3::cpu::bind_color_fields(m);
    r3::cpu::bind_shape_combo_api(m);
    r3::cpu::bind_final_rescore(m);
    m.def(
        "inert_knob_names",
        []() {
            py::list out;
            for (const auto& kv : kInertKnobs)
                out.append(py::str(kv.first));
            return out;
        },
        "Launch knobs this backend refuses by name: they describe CUDA "
        "hardware. Published so a caller can distinguish a knob it chose from "
        "one a config emits by default.");

    // The one read-only color view, exposed so the
    // round-trip fixture (tests/test_color_view.py) can check the C++
    // decoders against roshambo3.layout's Python ones on the SAME inputs.
    // Deliberately its own entry point, never touched by `score_pairs`: this
    // seam has no color kernel (color=True is still refused at the
    // Python/backend boundary, backend/cpu.py), so there is nothing here for
    // the shipping dispatch to call yet, and nothing in this function can
    // move the shipping path's scores because the shipping path never calls
    // it.
    m.def(
        "build_color_view",
        [](py::array_t<float, py::array::c_style | py::array::forcecast>
               query_coords,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>
               query_types,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>
               query_n_atoms,
           py::array_t<float, py::array::c_style | py::array::forcecast>
               library_coords,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
               library_counts,
           int n_features) {
            auto qc = query_coords.unchecked<3>();  // [n_q, Lq, 3|4]
            auto qt = query_types.unchecked<2>();   // [n_q, Lq]
            auto qn = query_n_atoms.unchecked<1>();
            if (qt.shape(0) != qn.shape(0) || qt.shape(0) != qc.shape(0) ||
                qt.shape(1) != qc.shape(1))
                throw std::invalid_argument(
                    "query_coords, query_types and query_n_atoms must agree "
                    "on shape [n_q, Lq(, 3|4)]");
            const py::ssize_t n_q = qt.shape(0);
            const py::ssize_t Lq = qt.shape(1);

            auto q_n_real = py::array_t<int32_t>(n_q);
            auto q_n_color = py::array_t<int32_t>(n_q);
            auto q_offset =
                py::array_t<int32_t>({n_q, py::ssize_t(n_features)});
            auto q_count = py::array_t<int32_t>({n_q, py::ssize_t(n_features)});
            auto qnr = q_n_real.mutable_unchecked<1>();
            auto qnc = q_n_color.mutable_unchecked<1>();
            auto qof = q_offset.mutable_unchecked<2>();
            auto qco = q_count.mutable_unchecked<2>();
            // Real feature COORDINATE content, one contiguous read, copied
            // out for the round-trip test to compare byte-for-byte against
            // `query_coords[i, n_real:n_real+n_color, :3]` -- the "every
            // real and color coordinate matches" half of the deciding
            // measurement, not just the offsets that locate them.
            int q_max_color = 0;

            std::vector<int32_t> types_row(static_cast<size_t>(Lq));
            std::vector<float> xyz_row(size_t(Lq) * 3);
            for (py::ssize_t i = 0; i < n_q; ++i) {
                for (py::ssize_t j = 0; j < Lq; ++j) {
                    types_row[size_t(j)] = qt(i, j);
                    for (int c = 0; c < 3; ++c)
                        xyz_row[size_t(j) * 3 + size_t(c)] = qc(i, j, c);
                }
                const auto v = r3::cpu::build_color_view_from_types(
                    xyz_row.data(), types_row.data(), int(qn(i)), int(Lq),
                    n_features, 3);
                qnr(i) = v.n_real;
                qnc(i) = v.n_color;
                q_max_color = std::max(q_max_color, v.n_color);
                for (int t = 0; t < n_features; ++t) {
                    qof(i, t) = v.seg_offset[t];
                    qco(i, t) = v.seg_count[t];
                }
            }
            // A second pass rebuilds each view (cheap; this is a debug/test
            // entry point, not a hot path) rather than caching the first
            // pass's `ColorConformer`s -- those alias `xyz_row`, which is
            // reused every iteration, so holding onto them past the loop
            // that filled it would be exactly the dangling-pointer bug this
            // comment is here to head off.
            auto q_color_xyz = py::array_t<float>(
                {n_q, py::ssize_t(std::max(q_max_color, 1)), py::ssize_t(3)});
            std::fill(q_color_xyz.mutable_data(),
                      q_color_xyz.mutable_data() + q_color_xyz.size(), 0.0f);
            {
                auto out3 = q_color_xyz.mutable_unchecked<3>();
                for (py::ssize_t i = 0; i < n_q; ++i) {
                    for (py::ssize_t j = 0; j < Lq; ++j) {
                        types_row[size_t(j)] = qt(i, j);
                        for (int c = 0; c < 3; ++c)
                            xyz_row[size_t(j) * 3 + size_t(c)] = qc(i, j, c);
                    }
                    const auto v = r3::cpu::build_color_view_from_types(
                        xyz_row.data(), types_row.data(), int(qn(i)), int(Lq),
                        n_features, 3);
                    for (int k = 0; k < v.n_color; ++k)
                        for (int c = 0; c < 3; ++c)
                            out3(i, k, c) =
                                v.color_xyz[size_t(k) * 3 + size_t(c)];
                }
            }

            auto lc = library_coords.unchecked<3>();
            auto lcnt = library_counts.unchecked<2>();
            if (lcnt.shape(1) != n_features || lcnt.shape(0) != lc.shape(0))
                throw std::invalid_argument(
                    "library_counts must be [n_library, n_features] and "
                    "agree with library_coords on n_library");
            const py::ssize_t n_l = lcnt.shape(0);
            const py::ssize_t Ll = lc.shape(1);
            auto l_n_real = py::array_t<int32_t>(n_l);
            auto l_n_color = py::array_t<int32_t>(n_l);
            auto l_offset =
                py::array_t<int32_t>({n_l, py::ssize_t(n_features)});
            auto l_count = py::array_t<int32_t>({n_l, py::ssize_t(n_features)});
            auto lnr = l_n_real.mutable_unchecked<1>();
            auto lnc = l_n_color.mutable_unchecked<1>();
            auto lof = l_offset.mutable_unchecked<2>();
            auto lco = l_count.mutable_unchecked<2>();
            int l_max_color = 0;

            std::vector<uint8_t> counts_row(static_cast<size_t>(n_features));
            std::vector<float> lxyz_row(size_t(Ll) * 3);
            for (py::ssize_t i = 0; i < n_l; ++i) {
                for (int t = 0; t < n_features; ++t)
                    counts_row[size_t(t)] = lcnt(i, t);
                for (py::ssize_t j = 0; j < Ll; ++j)
                    for (int c = 0; c < 3; ++c)
                        lxyz_row[size_t(j) * 3 + size_t(c)] = lc(i, j, c);
                const auto v = r3::cpu::build_color_view_from_counts(
                    lxyz_row.data(), counts_row.data(), n_features, 3);
                lnr(i) = v.n_real;
                lnc(i) = v.n_color;
                l_max_color = std::max(l_max_color, v.n_color);
                for (int t = 0; t < n_features; ++t) {
                    lof(i, t) = v.seg_offset[t];
                    lco(i, t) = v.seg_count[t];
                }
            }
            auto l_color_xyz = py::array_t<float>(
                {n_l, py::ssize_t(std::max(l_max_color, 1)), py::ssize_t(3)});
            std::fill(l_color_xyz.mutable_data(),
                      l_color_xyz.mutable_data() + l_color_xyz.size(), 0.0f);
            {
                auto out3 = l_color_xyz.mutable_unchecked<3>();
                for (py::ssize_t i = 0; i < n_l; ++i) {
                    for (int t = 0; t < n_features; ++t)
                        counts_row[size_t(t)] = lcnt(i, t);
                    for (py::ssize_t j = 0; j < Ll; ++j)
                        for (int c = 0; c < 3; ++c)
                            lxyz_row[size_t(j) * 3 + size_t(c)] = lc(i, j, c);
                    const auto v = r3::cpu::build_color_view_from_counts(
                        lxyz_row.data(), counts_row.data(), n_features, 3);
                    for (int k = 0; k < v.n_color; ++k)
                        for (int c = 0; c < 3; ++c)
                            out3(i, k, c) =
                                v.color_xyz[size_t(k) * 3 + size_t(c)];
                }
            }

            py::dict qd, ld, out;
            qd["n_real"] = q_n_real;
            qd["n_color"] = q_n_color;
            qd["seg_offset"] = q_offset;
            qd["seg_count"] = q_count;
            qd["color_xyz"] = q_color_xyz;
            ld["n_real"] = l_n_real;
            ld["n_color"] = l_n_color;
            ld["seg_offset"] = l_offset;
            ld["seg_count"] = l_count;
            ld["color_xyz"] = l_color_xyz;
            out["query"] = qd;
            out["library"] = ld;
            return out;
        },
        py::arg("query_coords"), py::arg("query_types"),
        py::arg("query_n_atoms"), py::arg("library_coords"),
        py::arg("library_counts"), py::arg("n_features"),
        "Build the color view via the C++ decoders "
        "(color_view.hpp's build_color_view_from_types / "
        "_from_counts) and report n_real, n_color, segment offsets/counts "
        "AND the color coordinates each decoder read (zero-padded to the "
        "batch's own max n_color), for both sides -- so a test can check "
        "every field against roshambo3.layout's Python decoders and "
        "against a direct slice of the input arrays, on the same inputs.");

    m.doc() =
        "ROSHAMBO3 CPU shape kernel: scalar, exact, reduced frame.\n"
        "\n"
        "THIS IS THE REFERENCE ANCHOR.\n"
        "\n"
        "It is the third implementation that the CUDA path and every optimized\n"
        "CPU arm are compared AGAINST, and it exists because neither of those\n"
        "can be the reference for the other: a CPU score legitimately depends\n"
        "on vector width and a GPU score legitimately depends on\n"
        "architecture (NUMERICS_POLICY section 8), so two implementations\n"
        "checked only against each other agree whenever they share a defect,\n"
        "at the scale of the project rather than of a function. It is not\n"
        "hypothetical -- a quaternion normalization defect survived in\n"
        "ROSHAMBO2's two backends for exactly as long as they were each\n"
        "other's only check.\n"
        "\n"
        "What makes it an anchor is not that it is accurate and not that it is\n"
        "fast. It is that its algorithm is FROZEN and its source is STAMPED:\n"
        "source-order fp32 accumulation, which is the accumulation every\n"
        "vectorized arm reassociates FROM.\n"
        "\n"
        "SO: CHANGING THIS SOURCE INVALIDATES EVERY FIXTURE GENERATED FROM IT.\n"
        "`__cpu_src_sha256__` is what makes that detectable rather than a\n"
        "matter of remembering -- a committed fixture names the hash that\n"
        "produced it, and a comparison against a fixture whose hash does not\n"
        "match this module's is not a comparison. Regenerate deliberately,\n"
        "record why, and treat it as a declared numerics change; do not\n"
        "regenerate to make something green.\n"
        "\n"
        "It is NOT a second specification (NUMERICS_POLICY section 2): Tier 3\n"
        "remains the specification and it remains the CUDA kernel's. This is\n"
        "what two implementations are COMPARED against; no reported number\n"
        "comes from here.";

    // The seam's entry point returns a plain [n_pairs, 20] array and nothing
    // else. `py::array_t` cannot carry an arbitrary Python attribute, and more
    // to the point a conditional return type would break the seam -- the
    // engine calls this at one site and expects an array.
    m.def(
        "score_pairs",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> qc,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> qn,
           py::array_t<float, py::array::c_style | py::array::forcecast> lc,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> lcnt,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pq,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pt,
           int steps, float lr_q, float lr_t, int start_mode, int grain,
           int width, bool poly, int threads, float cutoff_d2,
           float converge_tol, int only_start, float field_h, float field_pad,
           int field_width, int field_unroll, bool field_fp16,
           bool field_affine, bool field_edge, bool field_edgefield,
           bool field_pipeline,
           bool field_pairaos, bool field_coeff, int field_allin,
           int field_gridxf, std::optional<py::array_t<float>> library_vaa,
           bool field_winner_only, bool field_batched,
           bool field_report_field_score, const py::kwargs& kw) {
            return score_pairs(
                qc, qn, lc, lcnt, pq, pt, steps, lr_q, lr_t, start_mode, grain,
                width, poly, threads, cutoff_d2, converge_tol, only_start,
                field_h, field_pad, field_width, field_unroll, field_fp16,
                field_affine, field_edge, field_edgefield, field_pipeline,
                field_pairaos, field_coeff, field_allin, field_gridxf,
                library_vaa ? &*library_vaa : nullptr, field_winner_only,
                field_batched, field_report_field_score,
                false, nullptr, false, nullptr, kw);
        },
        py::arg("query_coords"), py::arg("query_n_atoms"),
        py::arg("library_coords"), py::arg("library_counts"),
        py::arg("pair_query_row"), py::arg("pair_library_row"),
        py::arg("steps") = 25, py::arg("lr_q") = 0.1f, py::arg("lr_t") = 0.4f,
        py::arg("start_mode") = 1, py::arg("grain") = -1, py::arg("width") = 1,
        // Default false: the shipped grade is libm, and a grade change is a
        // declared numerics change judged on recall.
        py::arg("poly") = false, py::arg("threads") = 1,
        py::arg("cutoff_d2") = 0.0f, py::arg("converge_tol") = 0.0f,
        py::arg("only_start") = -1, py::arg("field_h") = 0.0f,
        py::arg("field_pad") = 8.0f, py::arg("field_width") = 1,
        py::arg("field_unroll") = 1, py::arg("field_fp16") = false,
        py::arg("field_affine") = false, py::arg("field_edge") = false,
        py::arg("field_edgefield") = false, py::arg("field_pipeline") = false,
        py::arg("field_pairaos") = false, py::arg("field_coeff") = false,
        py::arg("field_allin") = -1, py::arg("field_gridxf") = -1,
        py::arg("library_vaa") = std::nullopt,
        py::arg("field_winner_only") = false, py::arg("field_batched") = false,
        py::arg("field_report_field_score") = false);

    // The census entry point is separate, returns (scores, dict), and is for
    // the op count and nothing else. Separate because the counters cost
    // something: a RATE taken with them on is a rate of the instrumented
    // kernel, which is not the kernel. Keeping them off the shipping path is
    // also what this package's docstring asks for -- "the shipping surface
    // here is one function and a handful of enums; arms live behind
    // roshambo3.experimental".
    m.def(
        "score_pairs_with_census",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> qc,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> qn,
           py::array_t<float, py::array::c_style | py::array::forcecast> lc,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> lcnt,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pq,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pt,
           int steps, float lr_q, float lr_t, int start_mode, int grain,
           int width, bool poly, int threads, float cutoff_d2,
           float converge_tol, int only_start, float field_h, float field_pad,
           int field_width, int field_unroll, bool field_fp16,
           bool field_affine, bool field_edge, bool field_edgefield,
           bool field_pipeline,
           bool field_pairaos, bool field_coeff, int field_allin,
           int field_gridxf, std::optional<py::array_t<float>> library_vaa,
           bool field_winner_only, bool field_batched,
           bool field_report_field_score, const py::kwargs& kw) {
            py::dict census;
            auto scores = score_pairs(
                qc, qn, lc, lcnt, pq, pt, steps, lr_q, lr_t, start_mode, grain,
                width, poly, threads, cutoff_d2, converge_tol, only_start,
                field_h, field_pad, field_width, field_unroll, field_fp16,
                field_affine, field_edge, field_edgefield, field_pipeline,
                field_pairaos, field_coeff, field_allin, field_gridxf,
                library_vaa ? &*library_vaa : nullptr, field_winner_only,
                field_batched, field_report_field_score,
                true, &census, false, nullptr, kw);
            return py::make_tuple(scores, census);
        },
        py::arg("query_coords"), py::arg("query_n_atoms"),
        py::arg("library_coords"), py::arg("library_counts"),
        py::arg("pair_query_row"), py::arg("pair_library_row"),
        py::arg("steps") = 25, py::arg("lr_q") = 0.1f, py::arg("lr_t") = 0.4f,
        py::arg("start_mode") = 1, py::arg("grain") = -1, py::arg("width") = 1,
        // Default false: the shipped grade is libm, and a grade change is a
        // declared numerics change judged on recall.
        py::arg("poly") = false, py::arg("threads") = 1,
        py::arg("cutoff_d2") = 0.0f, py::arg("converge_tol") = 0.0f,
        py::arg("only_start") = -1, py::arg("field_h") = 0.0f,
        py::arg("field_pad") = 8.0f, py::arg("field_width") = 1,
        py::arg("field_unroll") = 1, py::arg("field_fp16") = false,
        py::arg("field_affine") = false, py::arg("field_edge") = false,
        py::arg("field_edgefield") = false, py::arg("field_pipeline") = false,
        py::arg("field_pairaos") = false, py::arg("field_coeff") = false,
        py::arg("field_allin") = -1, py::arg("field_gridxf") = -1,
        py::arg("library_vaa") = std::nullopt,
        py::arg("field_winner_only") = false, py::arg("field_batched") = false,
        py::arg("field_report_field_score") = false);

    // A THIRD entry point, mirroring
    // `score_pairs_with_census`'s shape exactly (separate `m.def`, same
    // (scores, dict) return), and for the same reason: pricing structure is
    // not the shipping path, and the shipping `score_pairs` above passes
    // `false, nullptr` for both census and phase timers unconditionally, so
    // neither instrument it. "An instrumented run is never a rate run" --
    // this exists so the STRUCTURE (which phase costs what) can be reported
    // without that structure ever being read as a throughput number.
    m.def(
        "score_pairs_with_phase_timers",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> qc,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> qn,
           py::array_t<float, py::array::c_style | py::array::forcecast> lc,
           py::array_t<uint8_t, py::array::c_style | py::array::forcecast> lcnt,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pq,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pt,
           int steps, float lr_q, float lr_t, int start_mode, int grain,
           int width, bool poly, int threads, float cutoff_d2,
           float converge_tol, int only_start, float field_h, float field_pad,
           int field_width, int field_unroll, bool field_fp16,
           bool field_affine, bool field_edge, bool field_edgefield,
           bool field_pipeline,
           bool field_pairaos, bool field_coeff, int field_allin,
           int field_gridxf, std::optional<py::array_t<float>> library_vaa,
           bool field_winner_only, bool field_batched,
           bool field_report_field_score, const py::kwargs& kw) {
            py::dict phase_timers;
            auto scores = score_pairs(
                qc, qn, lc, lcnt, pq, pt, steps, lr_q, lr_t, start_mode, grain,
                width, poly, threads, cutoff_d2, converge_tol, only_start,
                field_h, field_pad, field_width, field_unroll, field_fp16,
                field_affine, field_edge, field_edgefield, field_pipeline,
                field_pairaos, field_coeff, field_allin, field_gridxf,
                library_vaa ? &*library_vaa : nullptr, field_winner_only,
                field_batched, field_report_field_score,
                false, nullptr, true, &phase_timers, kw);
            return py::make_tuple(scores, phase_timers);
        },
        py::arg("query_coords"), py::arg("query_n_atoms"),
        py::arg("library_coords"), py::arg("library_counts"),
        py::arg("pair_query_row"), py::arg("pair_library_row"),
        py::arg("steps") = 25, py::arg("lr_q") = 0.1f, py::arg("lr_t") = 0.4f,
        py::arg("start_mode") = 1, py::arg("grain") = -1, py::arg("width") = 1,
        py::arg("poly") = false, py::arg("threads") = 1,
        py::arg("cutoff_d2") = 0.0f, py::arg("converge_tol") = 0.0f,
        py::arg("only_start") = -1, py::arg("field_h") = 0.0f,
        py::arg("field_pad") = 8.0f, py::arg("field_width") = 1,
        py::arg("field_unroll") = 1, py::arg("field_fp16") = false,
        py::arg("field_affine") = false, py::arg("field_edge") = false,
        py::arg("field_edgefield") = false, py::arg("field_pipeline") = false,
        py::arg("field_pairaos") = false, py::arg("field_coeff") = false,
        py::arg("field_allin") = -1, py::arg("field_gridxf") = -1,
        py::arg("library_vaa") = std::nullopt,
        py::arg("field_winner_only") = false, py::arg("field_batched") = false,
        py::arg("field_report_field_score") = false);

    m.def("have_avx512", &have_avx512,
          "Does THIS PROCESS have AVX-512? Asked of the CPU, not the build.");
    m.def("compiled_targets", &compiled_targets,
          "Comma-separated SIMD targets this extension was built for.");
    m.def("selected_target", &selected_target,
          "The target this process actually dispatched to. CPU grading refuses a "
          "decision-grade CPU number without it: a build that silently ran a "
          "baseline scalar path looks like a slow kernel, not a broken build.");

    // Every CPU record must carry the ISA *and the vector
    // width*. This is the width of the SIMD layer the SOURCE uses -- 1 while
    // `simd.hpp` is scalar -- and deliberately not "how wide is the ISA": the
    // two differ, and the one that decides whether two scores are comparable is
    // the source's, because that is what sets the reduction order. Note the
    // compiler auto-vectorizes some 3-vector algebra here regardless (12 packed
    // ops at -march=x86-64-v3), which does NOT change this number and does not
    // change the accumulation order -- `-fno-associative-math` forbids that.
    m.attr("__vector_width__") =
        static_cast<int>(r3::simd::Lanes(r3::simd::ScalarTag{}));

    // The loaded extension stamp exposes the color ABI/feature-count limit
    // and selected ISA. Selected ISA is
    // `selected_target()`, already exposed above; these two are the color
    // half. `__color_max_features__` is the count-vector width limit
    // (one 64-bit word, byte 7 reserved) and must equal
    // `roshambo3.layout.MAX_FEATURES` and `ROSHAMBO3_MAX_FEATURES`
    // (kernels.cuh) -- pinned together in
    // `tests/test_color_view.py::TestTheStamp`, the same discipline
    // `test_cpu_kernel.py::TestTheConstantsCannotDrift` already applies to
    // the shape constants below. `__color_abi_version__` names the
    // merged-record CONVENTION
    // this build's decoders read (segment order, padding rule), so a caller
    // holding a fixture generated against one version can tell whether this
    // binary reads the same one.
    m.attr("__color_max_features__") = r3::cpu::MAX_COLOR_FEATURES;
    m.attr("__color_abi_version__") = r3::cpu::COLOR_ABI_VERSION;

    m.attr("__coord_scale__") = COORD_SCALE;
    m.attr("__shape_prefactor__") = SHAPE_PREFACTOR;
    m.attr("__minus_2ln2__") = MINUS_2LN2;
    m.attr("__n_score_columns__") = N_SCORE_COLUMNS;
#ifdef ROSHAMBO3_BUILD_TYPE
    m.attr("__build_type__") = ROSHAMBO3_BUILD_TYPE;
#else
    m.attr("__build_type__") = "unknown";
#endif
#ifdef ROSHAMBO3_CXX_FLAGS
    m.attr("__cxx_flags__") = ROSHAMBO3_CXX_FLAGS;
#else
    m.attr("__cxx_flags__") = "none";
#endif

    // The anchor's identity. The content hash is the identity;
    // the tree hash and the dirty marker are provenance beside it, never in
    // place of it -- git's tree hash describes the INDEX, so an uncommitted
    // edit leaves it unchanged, which is exactly why the tri-state marker is
    // not optional.
    m.attr("__cpu_src_sha256__") = ROSHAMBO3_CPU_SRC_SHA256;
    m.attr("__cpu_src_tree__") = ROSHAMBO3_CPU_SRC_TREE;
    m.attr("__cpu_src_dirty__") = (int)(ROSHAMBO3_CPU_SRC_DIRTY);
    m.attr("__cpu_src_dirty_files__") = ROSHAMBO3_CPU_SRC_DIRTY_FILES;
    m.attr("__source_stamp_phase__") = ROSHAMBO3_SOURCE_STAMP_PHASE;
}

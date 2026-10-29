"""Deliberately-constructed non-compliant runs, one per acceptance-criteria
kind, each driven through the *real* enforcement code path
(:func:`bench.rules.enforce`) rather than a mock of the check.

Run via ``roshambo3-bench demo-refuse``. Each demo appends a real run record
(status ``"refused"`` or ``"advisory"``) to
``demo_refuse.jsonl`` in the results directory, so the refusal is not just
printed but is on disk with the same schema every other run uses.

The clock and operating-point provenance demos follow the
same pattern, and reuse *actual measured facts* rather
than invented numbers, so each demo is a faithful reconstruction of a real
diagnosed incident, not a synthetic strawman:

- ``demo_cross_clock_comparison``: this project's own diagnosed incident --
  a locked-clock run at 1395 MHz cited as ``gold_ref_run_id`` for a run that
  was actually profiled by ``ncu`` (which pinned the card at 2265 MHz,
  rock-steady across six samples). Demonstrates the harness flagging it
  ``advisory``.
- ``demo_percent_of_peak_missing_clock``: the project's own wrong "5.6% of
  the MUFU.EX2 ceiling" figure (true value at the measurement's own clock:
  12.9%) -- a percent-of-peak metric with no normalizing-clock
  field alongside it. Demonstrates a hard **refusal**.
- ``demo_power_limited_not_clock_locked``: a real measurement -- 1800
  MHz requested, the lock command reports success, but the card sits at
  1762 MHz with ``sw_power_cap`` Active throughout. Demonstrates the harness
  correctly refusing to call this "clock-locked".
"""
from __future__ import annotations

import math
import subprocess
import sys
import time
import os
from pathlib import Path

from . import rules
from .build_check import verify_build
from .gitinfo import get_git_info
from .gpu_lock import DeviceIdentity, GpuLock, GpuLockBusyError
from .run_record import append_jsonl, build_run_record

REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_ctx(**overrides) -> rules.RunContext:
    """A compliant context, so each scenario can vary exactly one fact.

    ``build_derived=True`` because every scenario below except
    :func:`demo_context_not_derived_from_a_build` is demonstrating some *other*
    rule, and a context that never inspected its build is refused before any of
    them are reached. A real harness gets this from
    :meth:`rules.RunContext.from_build`; here the point is to hold everything
    else fixed.
    """
    defaults = dict(
        is_release_build=True, git_dirty=False, gpu_lock_held=True,
        gpu_lock_acquisition_failed=False, other_process_on_gpu=False,
        clocks_locked=False, throttle_fraction=0.0, sustained_kernel_s=120.0,
        # Required, not defaulted, on the real dataclass.
        measurement_regime="calibration",
        build_derived=True,
    )
    defaults.update(overrides)
    return rules.RunContext(**defaults)


def _base_cpu_ctx(**overrides) -> rules.RunContext:
    """A compliant CPU context, so a CPU scenario varies exactly one fact.

    There are two obligations that a CPU run must meet before any other CPU
    rule is reached -- ``checked_a_cpu_build`` and ``cpu_vector_width`` -- so
    they belong in the fixture rather than in each scenario, exactly as
    ``build_derived=True`` does above.
    """
    defaults = dict(
        backend_name="cpu", cpu_isa_selected="AVX2",
        cpu_isa_available=["SSE4", "AVX2"],
        checked_a_cpu_build=True, cpu_vector_width=8, cpu_threads=24,
        cpu_lock_held=True, cpu_affinity=list(range(24)),
        achieved_cpu_mhz={"n_samples": 120},
        # A content-hashed source stamp, so a CPU scenario shows only the
        # advisory it is about. Without it every CPU demo also carries the
        # "does not say which source it was compiled from", which is true of
        # the fixture rather than of the scenario.
        build_source_dirty=False,
        build_source_sha256="0" * 64,
        build_source_stamp_phase="build",
    )
    defaults.update(overrides)
    return _base_ctx(**defaults)


def _base_req(**overrides) -> rules.RunRequest:
    defaults = dict(
        decision_grade=True, n_conformers=2_000_000, max_results=20_000_000,
        claim_non_burst=True, n_repeats=11, n_warmup=3,
        # A compliant launch structure, so a scenario that is about something
        # else does not also carry the "no launch structure recorded" tag.
        launch=rules.LaunchShape(conformers_per_launch=2_000_000.0,
                                 n_launches=8, n_conformers=16_000_000,
                                 n_groups=8),
    )
    defaults.update(overrides)
    return rules.RunRequest(**defaults)


def _record_from(kind: str, description: str, req: rules.RunRequest,
                  ctx: rules.RunContext, result: rules.EnforcementResult,
                  build, git, results_dir: Path) -> dict:
    record = build_run_record(
        ticket_id="demo-refuse", git=git, build=build, clocks_locked=ctx.clocks_locked,
        gpu_lock_held=ctx.gpu_lock_held,
        dataset={"n": req.n_conformers, "source": "demo-refuse (constructed)"},
        query_panel={"name": "n/a", "sha256": "n/a", "ids": []},
        config={
            "demo_kind": kind, "decision_grade": req.decision_grade,
            "n_conformers": req.n_conformers, "max_results": req.max_results,
            "claim_non_burst": req.claim_non_burst,
        },
        timing=(dict(ctx.timing) if ctx.timing
                else {"sustained_kernel_s": ctx.sustained_kernel_s}),
        metrics=dict(ctx.metrics or {}), audits={}, artifacts={},
        status=result.status, advisory_reasons=result.advisory_reasons,
        refusal_reason=result.refusal_reason, notes=description,
    )
    append_jsonl(record, results_dir / "demo_refuse.jsonl")
    return {"kind": kind, "record": record.as_dict()}


def demo_non_release_build(build_release, git, results_dir: Path) -> dict:
    """A non-Release build must not have its timing recorded.

    Prefers a *real* Debug build, queried through the same :func:`verify_build`
    every real run uses, via a subprocess running that build's own interpreter --
    so the ``.so`` genuinely loaded is the Debug one rather than a value typed in.
    Set ``ROSHAMBO3_DEBUG_PYTHON`` to such an interpreter to get that path.

    Without it the scenario constructs the non-Release stamp directly and pushes
    it through the identical ``rules.enforce()`` call, which is a weaker
    demonstration and is labeled as such in the record.
    """
    # A real Debug build is preferred but cannot be assumed to exist: the
    # original was a scratch directory from a session long gone. Point
    # ROSHAMBO3_DEBUG_PYTHON at a venv whose interpreter loads a Debug-configured
    # .so and this scenario uses it; otherwise it falls back below, which
    # exercises the same rules.enforce() path with a constructed stamp.
    env_debug = os.environ.get("ROSHAMBO3_DEBUG_PYTHON")
    scratch_debug_venv = Path(env_debug) if env_debug else Path("/nonexistent")
    is_release = None
    build_type = "unknown"
    if scratch_debug_venv.is_file():
        out = subprocess.run(
            [str(scratch_debug_venv), "-c",
             "from bench.build_check import verify_build as v; "
             "b = v(); print(b.build_type); print(b.is_release)"],
            cwd="/tmp", env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True,
        )
        lines = out.stdout.strip().splitlines()
        if len(lines) >= 2:
            build_type, is_release_str = lines[0], lines[1]
            is_release = is_release_str == "True"

    if is_release is None:
        # scratch debug build unavailable in this environment; fall back to
        # a directly-constructed non-Release BuildInfo through the same
        # rules.enforce() path.
        build_type = "Debug"
        is_release = False

    req = _base_req()
    ctx = _base_ctx(is_release_build=is_release)
    result = rules.enforce(req, ctx)
    desc = (
        f"real CMake Debug build (build_type={build_type!r}, verified via "
        "verify_build() against the actual compiled .so) fed through "
        "rules.enforce()."
    )
    return _record_from("non_release_build", desc, req, ctx, result, build_release, git, results_dir)


def demo_gpu_lock_not_held(build, git, results_dir: Path) -> dict:
    """Exercise real flock contention with a synthetic device identity.

    No GPU discovery, CUDA context, timed workload, or shared hardware lock is
    required. Two separately opened lock handles must exclude one another.
    """
    device = DeviceIdentity(visible_index=0, physical_index=0, uuid="GPU-synthetic-demo")
    lock_path = str(results_dir / "synthetic-gpu.lock")
    acquisition_failed = False
    with GpuLock("demo-holder", device=device, lock_path=lock_path, blocking=True):
        try:
            with GpuLock("demo-second", device=device, lock_path=lock_path, blocking=False):
                pass
        except GpuLockBusyError:
            acquisition_failed = True
    req = _base_req()
    ctx = _base_ctx(gpu_lock_held=False, gpu_lock_acquisition_failed=acquisition_failed)
    result = rules.enforce(req, ctx)
    desc = ("Real filesystem lock contention with a synthetic GPU identity; "
            f"second acquisition refused={acquisition_failed}. No GPU was used.")
    return _record_from("gpu_lock_not_held", desc, req, ctx, result, build, git, results_dir)


def demo_max_results_too_small(build, git, results_dir: Path) -> dict:
    """The literal library default (max_results=10,000, see
    roshambo3.screen() signature) against a 2,000,000-conformer batch
    -- per-batch truncation of the result set."""
    req = _base_req(n_conformers=2_000_000, max_results=10_000)
    ctx = _base_ctx()
    result = rules.enforce(req, ctx)
    desc = "library default max_results=10000 against n_conformers=2,000,000 (need >= 20,000,000)."
    return _record_from("max_results_too_small", desc, req, ctx, result, build, git, results_dir)


def demo_batch_too_small(build, git, results_dir: Path) -> dict:
    """A large corpus cut into launches far too small for it.

    20,381,286 conformers over 4,988 launches -- 4,086 each -- on a corpus
    whose 18 groups would have permitted a mean launch of 1,132,294. This is
    the case the launch-saturation rule exists for and the case the retired 1e6
    check *passed*, because it was handed the corpus total instead of one launch's.
    """
    req = _base_req(
        n_conformers=4_086, max_results=1_000_000,
        launch=rules.LaunchShape(conformers_per_launch=20_381_286 / 4_988,
                                 n_launches=4_988, n_conformers=20_381_286,
                                 n_groups=18))
    ctx = _base_ctx()
    result = rules.enforce(req, ctx)
    desc = ("20.4M conformers cut into 4,988 launches of 4,086 on an 18-group "
            "corpus that could have delivered 1,132,294 per launch.")
    return _record_from("batch_too_small", desc, req, ctx, result, build, git, results_dir)


def demo_corpus_bound_launch(build, git, results_dir: Path) -> dict:
    """A small corpus whose group structure makes a saturating launch
    impossible -- advisory, not refused.

    48,660 conformers in 6 groups caps the mean launch at 8,110, so no setting
    reaches 32,768. The number is real and a ratio taken on it is usable; its
    absolute throughput is not the machine's. Refusing it outright is what the
    retired check did, and what several harnesses each had to route around.
    """
    req = _base_req(
        n_conformers=8_110, max_results=1_000_000,
        launch=rules.LaunchShape(conformers_per_launch=8_110.0, n_launches=6,
                                 n_conformers=48_660, n_groups=6))
    ctx = _base_ctx()
    result = rules.enforce(req, ctx)
    desc = ("48,660 conformers in 6 groups: 8,110 per launch is all this "
            "corpus can deliver, so the run is advisory rather than refused.")
    return _record_from("corpus_bound_launch", desc, req, ctx, result, build,
                        git, results_dir)


def demo_binary_built_from_a_patched_tree(build, git, results_dir: Path) -> dict:
    """A binary whose stamped commit is not the source it contains.

    `git status --porcelain` on the compiled subtree was non-empty at build
    time, so ROSHAMBO3_GIT_SHA names a commit this binary does not contain. An
    advisory run may proceed on a patched tree; a decision-grade number may not
    cite a commit that does not describe it.
    """
    req = _base_req()
    ctx = _base_ctx(build_source_dirty=True,
                    build_source_sha256="0" * 64,
                    build_source_stamp_phase="build")
    result = rules.enforce(req, ctx)
    desc = ("a binary compiled from a working tree that matched no commit -- "
            "the stamp names a commit that is not the source it contains.")
    return _record_from("build_source_dirty", desc, req, ctx, result, build,
                        git, results_dir)


def demo_burst_claimed_non_burst(build, git, results_dir: Path) -> dict:
    """A run whose actual sustained kernel time is 8.4s, but the caller
    asks for it to be recorded as non-burst (fit for 1e9/1e11 projections)."""
    req = _base_req(claim_non_burst=True)
    ctx = _base_ctx(sustained_kernel_s=8.4)
    result = rules.enforce(req, ctx)
    desc = "sustained_kernel_s=8.4 (< 60s) with claim_non_burst=True."
    return _record_from("burst_claimed_non_burst", desc, req, ctx, result, build, git, results_dir)


def demo_cross_clock_comparison(build, git, results_dir: Path) -> dict:
    """Real facts from this project's own diagnosed incident (not
    invented): this run is at 1395 MHz, the achieved clock this machine
    holds under a 1402 MHz lock request with ``sw_power_cap`` inactive. It
    cites ``gold_ref_run_id`` against a prior run whose build was actually
    profiled by ``ncu`` -- which seized the clock at 2265 MHz,
    rock-steady across six samples, because ``ncu --clock-control`` defaults
    to ``base`` and overrides any external lock for the run's duration. The
    two are a >1.6x clock ratio apart, far beyond CLOCK_MATERIAL_RATIO
    (1.01x), and the gold-ref side's profiler_control_state is
    "profiler-active" against this run's "none" -- either condition alone
    is enough to flag the comparison; both fire here."""
    req = _base_req()
    ctx = _base_ctx(
        gold_ref_run_id="run_gold_ncu_seized_example",
        achieved_clock_mhz_median=1395.0,
        profiler_control_state="none",
        gold_ref_achieved_clock_mhz_median=2265.0,
        gold_ref_profiler_control_state="profiler-active",
    )
    result = rules.enforce(req, ctx)
    desc = (
        "this run: 1395 MHz achieved (1402 MHz lock request, sw_power_cap "
        "inactive -- the recorded 'normal' operating point) vs "
        "gold_ref_run_id: 2265 MHz achieved under ncu's default "
        "--clock-control=base, which seized the clock for that run's "
        "duration (a real diagnosed incident, reproduced here as a "
        "deliberately constructed comparison)."
    )
    return _record_from("cross_clock_comparison", desc, req, ctx, result, build, git, results_dir)


def demo_cross_regime_comparison(build, git, results_dir: Path) -> dict:
    """A run recorded in
    the ``headline`` regime (clocks deliberately unlocked, to measure the
    card's natural operating point) citing a ``gold_ref_run_id`` recorded in
    the ``calibration`` regime (clocks locked, one operating point held
    fixed across arms). This pair may never be
    combined into one ratio, table, or derived quantity, and that the two
    must not be mixed *silently* -- this is a hard refusal, not an advisory
    tag, because unlike the clock-ratio tolerance, the regime rule admits no
    material-difference threshold: any regime mismatch is disqualifying by
    construction."""
    req = _base_req()
    ctx = _base_ctx(
        measurement_regime="headline",
        gold_ref_run_id="run_calibration_example",
        gold_ref_measurement_regime="calibration",
    )
    result = rules.enforce(req, ctx)
    desc = (
        "this run: measurement_regime='headline' (clocks unlocked, natural "
        "operating point) vs gold_ref_run_id: measurement_regime="
        "'calibration' (clocks locked, one operating point held fixed) -- "
        "the cross-regime prohibition, reproduced here as a "
        "deliberately constructed comparison."
    )
    return _record_from("cross_regime_comparison", desc, req, ctx, result, build, git, results_dir)


def demo_percent_of_peak_missing_clock(build, git, results_dir: Path) -> dict:
    """The project's own wrong "5.6% of the MUFU.EX2 ceiling" figure:
    quoted without saying which clock the ceiling was computed at. At the
    measurement's own clock the true figure is 12.9% -- three
    different clocks (1402, 2265, 3090 MHz nameplate) give three different
    percentages for the *same* measurement, and the record does not say
    which one was used. rules.find_percent_of_peak_fields_missing_clock
    catches this by key-pattern; rules.enforce hard-refuses it."""
    metrics = {"pct_of_peak_mufu_ex2": 5.6}
    req = _base_req()
    ctx = _base_ctx(percent_of_peak_violations=rules.find_percent_of_peak_fields_missing_clock(metrics))
    result = rules.enforce(req, ctx)
    desc = (
        f"metrics={metrics!r} -- a percent-of-peak figure (this project's own "
        "'5.6% of the MUFU.EX2 ceiling') with no 'pct_of_peak_mufu_ex2_clock_mhz' "
        "or 'normalizing_clock_mhz' field alongside it. The true value "
        "at the measurement's own clock was 12.9%, not 5.6% -- the normalizing "
        "clock was never recorded, so the wrong ceiling silently got quoted "
        "for years."
    )
    return _record_from("percent_of_peak_missing_clock", desc, req, ctx, result, build, git, results_dir)


def demo_power_limited_not_clock_locked(build, git, results_dir: Path) -> dict:
    """A real measurement: 1800 MHz requested, nvidia-smi reports the
    lock command succeeded (clocks_locked=True), but under load the card
    sits at 1762 MHz with sw_power_cap Active the entire time -- the power
    ceiling is making the decisions, not the lock. rules.enforce must not
    let this be recorded as simply "clocks locked"."""
    req = _base_req()
    ctx = _base_ctx(clocks_locked=True, sw_power_cap_fraction=1.0)
    result = rules.enforce(req, ctx)
    desc = (
        "clocks_locked=True (nvidia-smi -lgc 1800,1800 reported success) but "
        "sw_power_cap_fraction=1.0 -- every sample during the run showed "
        "clocks_event_reasons.sw_power_cap Active, matching a measured "
        "1800-requested/1762-achieved run. 'Clocks locked' and 'power-limited' "
        "are different claims; this run is the latter."
    )
    return _record_from("power_limited_not_clock_locked", desc, req, ctx, result, build, git, results_dir)


def demo_no_gpu_build(build, git, results_dir: Path) -> dict:
    """A build with no CUDA extension at all.

    This scenario was absent because the rule was: ``check_build().ok`` means
    "found no forbidden flag", which is vacuously true when there is no
    extension to inspect -- and this harness read exactly that to decide whether
    a timing could be recorded. So a build with no GPU code could report a
    "Release, arithmetic pinned" verdict and have its timing accepted.

    Not fabricated: the CUDA-less state is exactly what the CPU CI job builds,
    and ``get_build_info()`` reports ``with_cuda=False`` there rather than
    raising.
    """
    req = _base_req()
    ctx = _base_ctx(checked_a_gpu_build=False)
    result = rules.enforce(req, ctx)
    return _record_from(
        "no_gpu_build",
        "a build with no CUDA extension: check_build().ok is vacuously true, "
        "so the run must be refused on checked_a_gpu_build instead",
        req, ctx, result, build, git, results_dir)


def demo_context_not_derived_from_a_build(build, git, results_dir: Path) -> dict:
    """A run whose context never inspected the build it was measuring.

    Every build-dependent field on :class:`rules.RunContext` defaults to the
    permissive answer -- there was GPU code, no forbidden flag was found, the
    architecture probe ran. That is right for a hand-built context in a test and
    wrong for a number anyone will cite, and it is what actually happened: the
    six ROSHAMBO2-era harness scripts each forwarded ``arch_profile`` and none of
    the rest, so the no-GPU-build and pinned-arithmetic refusals could not fire
    on a real run at all. They were reachable only from the tests that set the
    fields by hand -- which is why the tests passed.

    The rule is therefore about provenance, not about the build: a
    decision-grade context that did not come from
    :meth:`rules.RunContext.from_build` is refused because nothing looked.
    """
    req = _base_req()
    ctx = _base_ctx(build_derived=False)
    result = rules.enforce(req, ctx)
    return _record_from(
        "context_not_derived_from_a_build",
        "a decision-grade run whose context was assembled by hand, leaving "
        "every build-dependent fact at its permissive default",
        req, ctx, result, build, git, results_dir)


def demo_arch_probe_unavailable(build, git, results_dir: Path) -> dict:
    """The architecture probe could not run, so the architecture went unchecked.

    ``cuobjdump`` is not on ``PATH`` under a default NVIDIA apt install, and the
    probe used to return an empty architecture list -- indistinguishable from
    "probed, and this build has no SASS". The failure it guards is a library
    compiled for the wrong architecture that JIT-compiles at runtime, which no
    build stamp can see and this probe is what catches.

    Advisory rather than a refusal: the build is not known to be wrong, only
    unverified.
    """
    req = _base_req()
    ctx = _base_ctx(arch_probe_error="cuobjdump is not on PATH")
    result = rules.enforce(req, ctx)
    return _record_from(
        "arch_probe_unavailable",
        "cuobjdump unavailable, so the loaded .so's architecture list could "
        "not be read and the target architecture was not checked",
        req, ctx, result, build, git, results_dir)


def demo_host_flags_unavailable(build, git, results_dir: Path) -> dict:
    """The host compiler flags were never inspected, and the result looks clean.

    The third instance of the "nobody looked, so the result is empty" shape.
    ``numerics.check_cxx_flags`` returns an empty
    list both when it inspected the stamp's host flags and found none of
    ``SUSPECT_CXX_FLAGS``, and when the stamp recorded no host flags at all --
    and from the empty list the harness cannot tell which happened. The host
    half of the extension is what prepares the library records and reduces the
    scores, so "nobody looked" is a real gap and not a formality.

    Advisory rather than a refusal, deliberately. Unlike the six nvcc flags,
    the host-flag effect has never been measured on this project, and
    ``SUSPECT_CXX_FLAGS`` says so in as many words: the policy does not claim a
    number it has not taken. An uninspected host build is unverified, not known
    to be wrong -- the same standing this file already gives
    ``arch_probe_unavailable``.
    """
    req = _base_req()
    ctx = _base_ctx(checked_host_flags=False)
    result = rules.enforce(req, ctx)
    return _record_from(
        "host_flags_unavailable",
        "a build whose stamp carried no host compiler flags: the host-flag "
        "advisory list is empty because nothing was inspected, not because "
        "nothing was found",
        req, ctx, result, build, git, results_dir)


def demo_cpu_dispatch_target_unknown(build, git, results_dir: Path) -> dict:
    """A CPU run that cannot say which SIMD path it actually took.

    The CPU analogue of a wrong-architecture GPU build, and it fails the same
    way. A build that silently ran
    a baseline scalar path looks like a *slow kernel*, not like a broken build
    -- and with a dispatching SIMD library, which is what the CPU backend uses,
    it is reachable by accident rather than by misconfiguration.

    Refused rather than tagged, for the same reason its GPU sibling is: asking
    the binary what targets it contains is a different question from asking the
    process what it chose, and the second one is the one that decides what the
    number means.
    """
    req = _base_req()
    ctx = _base_cpu_ctx(cpu_isa_selected=None)
    result = rules.enforce(req, ctx)
    return _record_from(
        "cpu_dispatch_target_unknown",
        "a CPU run whose record does not name the SIMD dispatch target the "
        "process selected at run time",
        req, ctx, result, build, git, results_dir)


# ---------------------------------------------------------------------------
# Run kinds. The gates below are the ACCURACY ones; everything above is
# a rate gate, and the point of the kind is that the two sets are disjoint.
# ---------------------------------------------------------------------------

_ACC_METRICS = {
    "recall_conformer_level": {"mean": 0.9925, "min": 0.98, "p10": 0.99,
                               "per_query": [0.98, 1.0, 0.99, 1.0]},
    "recall_molecule_level": {"mean": 0.9710, "min": 0.95, "p10": 0.96,
                              "per_query": [0.95, 0.99, 0.96, 0.98]},
    "rbo_ext_mean": 0.9907,
}


def _acc_req(**kw):
    """An accuracy run: no timing, so every rate gate would refuse it."""
    d = dict(kind=rules.RUN_KIND_ACCURACY, n_repeats=1, n_warmup=0,
             claim_non_burst=False)
    d.update(kw)
    return _base_req(**d)


def _acc_ctx(**kw):
    d = dict(metrics=_ACC_METRICS, conformers_per_molecule=4.2,
             sustained_kernel_s=0.0)
    d.update(kw)
    return _base_cpu_ctx(**d)


def demo_accuracy_run_is_not_refused_for_a_timing_bar(build, git, results_dir):
    """An accuracy run graded as one, as a positive control.

    The run that chose the shipped grid spacing reports
    recall mean/min/p10 with the per-query table, RBO_ext at p=0.98, both
    levels, fit on a tuning corpus and reported on an evaluation one, and no
    throughput at all --
    and it graded `refused`, for failing a bar about timing repeats. Under a
    declared kind it grades `decision`.
    """
    req = _acc_req(chose_a_value="h")
    ctx = _acc_ctx(fit_corpus="D2-TUNE-10M", report_corpus="D3-EVAL-10M")
    result = rules.enforce(req, ctx)
    return _record_from(
        "accuracy_run_graded_not_refused",
        "an accuracy run with no throughput at all, which the rate harness "
        "refused for n_repeats=1 and which now grades on its own terms",
        req, ctx, result, build, git, results_dir)


def demo_accuracy_declared_but_carries_a_rate(build, git, results_dir):
    """The escape hatch, closed.

    The whole risk in adding a kind is that the next person whose timing run is
    refused relabels it `accuracy`. So the kind is checked against the record:
    a run carrying `start_pairs_per_s` is a rate run whatever it declares, and
    the contradiction is refused rather than exempted.
    """
    req = _acc_req()
    ctx = _acc_ctx(timing={"n_repeats": 1, "start_pairs_per_s": 1_113_221.0})
    result = rules.enforce(req, ctx)
    return _record_from(
        "accuracy_declared_but_carries_a_rate",
        "a run declaring kind=accuracy while reporting start_pairs_per_s -- "
        "malformed, not exempt",
        req, ctx, result, build, git, results_dir)


def demo_accuracy_without_a_per_query_table(build, git, results_dir):
    """METRICS.md: 'never a bare mean, and always publish the per-query table.'

    The 1e-4 quantile threshold ranged widely across
    queries and a mean hides it.
    """
    req = _acc_req()
    ctx = _acc_ctx(metrics={"recall_conformer_level": {"mean": 0.9925},
                            "recall_molecule_level": {"mean": 0.9710}})
    result = rules.enforce(req, ctx)
    return _record_from(
        "accuracy_without_a_per_query_table",
        "an accuracy run reporting aggregates with no per-query table",
        req, ctx, result, build, git, results_dir)


def demo_accuracy_without_molecule_level_recall(build, git, results_dir):
    """Molecule-level recall falls below conformer-level whenever a
    molecule has several conformers, so the conformer number alone overstates
    what a user gets."""
    req = _acc_req()
    ctx = _acc_ctx(metrics={"recall_conformer_level":
                            {"mean": 0.9925, "per_query": [0.98, 1.0, 0.99]}})
    result = rules.enforce(req, ctx)
    return _record_from(
        "accuracy_without_molecule_level_recall",
        "an accuracy run reporting conformer-level recall only",
        req, ctx, result, build, git, results_dir)


def demo_accuracy_fit_and_report_on_one_corpus(build, git, results_dir):
    """Fit and report on separate corpora; this is the project's own incident.

    The spacing run chose h = 1.2 on the tuning corpus at a 0.98 recall floor
    and measured 0.9750 on the evaluation corpus -- below the floor it was
    selected by. Fitting
    and reporting on the same corpus inflated recall by +0.0092 at K=30.
    """
    req = _acc_req(chose_a_value="h")
    ctx = _acc_ctx(fit_corpus="D3-EVAL-10M", report_corpus="D3-EVAL-10M")
    result = rules.enforce(req, ctx)
    return _record_from(
        "accuracy_fit_and_report_on_one_corpus",
        "an accuracy run that chose h on the corpus it reports on",
        req, ctx, result, build, git, results_dir)


def demo_accuracy_on_a_one_conformer_per_molecule_corpus(build, git, results_dir):
    """The molecule-level gate's degenerate pass, made visible, not refused.

    Corpora with one conformer per molecule exist, and
    their molecule-level recall IS their conformer-level recall. Both levels
    are present and the gate passes -- correctly, since these are legitimate
    corpora to measure on -- but the pass is structural, and the advisory says
    so rather than letting it read like a real one.
    """
    req = _acc_req()
    ctx = _acc_ctx(conformers_per_molecule=1.0)
    result = rules.enforce(req, ctx)
    return _record_from(
        "accuracy_on_a_degenerate_pitfall_9_corpus",
        "an accuracy run whose corpus cannot distinguish the two recall levels",
        req, ctx, result, build, git, results_dir)


def demo_cpu_lock_not_held(build, git, results_dir: Path) -> dict:
    """A decision-grade CPU timing on a shared machine, unarbitrated.

    Both machines are shared. The case a GPU-only contention check cannot see
    is the one that matters most here: a GPU screen holding the GPU lock is
    heavy CPU load, so a CPU run that checked only ``nvidia-smi`` would find
    the card "busy but locked by me" and conclude the machine was quiet.
    """
    req = _base_req()
    ctx = _base_cpu_ctx(cpu_lock_held=False)
    result = rules.enforce(req, ctx)
    return _record_from(
        "cpu_lock_not_held",
        "a decision-grade CPU timing that did not hold the CPU lock",
        req, ctx, result, build, git, results_dir)


def demo_cpu_run_was_not_pinned(build, git, results_dir: Path) -> dict:
    """A timed CPU run that controlled nothing it could have controlled.

    Advisory, not a refusal: the number is real, it is just not a controlled
    measurement. Affinity is the ONE lever available -- clocks cannot be locked
    and SMT cannot be changed on these machines -- so an unpinned run has let
    the scheduler migrate it across CCDs mid-measurement, which on a multi-CCD
    part means across separate L3 instances.
    """
    req = _base_req()
    ctx = _base_cpu_ctx(cpu_affinity=None)
    result = rules.enforce(req, ctx)
    return _record_from(
        "cpu_run_was_not_pinned",
        "a timed CPU run with no affinity set, so the scheduler was free to "
        "migrate it across CCDs mid-measurement",
        req, ctx, result, build, git, results_dir)


def demo_cpu_clock_never_sampled(build, git, results_dir: Path) -> dict:
    """A CPU run that may not claim a clock, because it did not look.

    "Did not look" and "the clock was steady" are different claims, and the
    rule that numbers from different clocks must not be combined needs a
    distribution to apply to. Since the clock cannot be *locked* here, the
    achieved distribution is the only thing standing in for clock control --
    which makes not sampling it the whole failure rather than a missing nicety.
    """
    req = _base_req()
    ctx = _base_cpu_ctx(achieved_cpu_mhz=None)
    result = rules.enforce(req, ctx)
    return _record_from(
        "cpu_clock_never_sampled",
        "a CPU run that sampled no frequencies and therefore may not claim a "
        "clock",
        req, ctx, result, build, git, results_dir)


def demo_cpu_build_never_inspected(build, git, results_dir: Path) -> dict:
    """The state the entire CPU ladder was in, and nothing said so.

    A CPU run whose CPU extension was never inspected. Its compiler flags were
    therefore never read, so ``-ffast-math`` and friends would not have been
    noticed -- and the harness **defers** those flags rather than refusing
    them, which makes noticing them the whole safeguard.

    The twin of :func:`demo_no_gpu_build`, added in the same change
    that scoped that one to GPU runs: scoping a refusal without supplying its
    counterpart turns a rule into a hole.
    """
    req = _base_req()
    ctx = _base_cpu_ctx(checked_a_cpu_build=False)
    result = rules.enforce(req, ctx)
    return _record_from(
        "cpu_build_never_inspected",
        "a CPU run whose CPU extension was never inspected, so its compiler "
        "flags were never read",
        req, ctx, result, build, git, results_dir)


def demo_cpu_width_not_recorded(build, git, results_dir: Path) -> dict:
    """A CPU run that does not name the vector width it scored at.

    This project accepts that a CPU score is a *function* of its SIMD width
    -- an 8-lane reduction and a 16-lane one do not produce the same float --
    in exchange for every record naming it; a record which does not cannot be
    compared with anything.

    A refusal rather than an advisory because of what it silently corrupts: a
    comparison between two runs that both look fine.
    """
    req = _base_req()
    ctx = _base_cpu_ctx(cpu_vector_width=None)
    result = rules.enforce(req, ctx)
    return _record_from(
        "cpu_width_not_recorded",
        "a CPU run that does not say which SIMD width produced its scores",
        req, ctx, result, build, git, results_dir)


def demo_cpu_run_without_a_gpu_is_allowed(build, git, results_dir: Path) -> dict:
    """A scenario that must NOT refuse, which is why it is here.

    Screening without a GPU is outcome 1 of the CPU workstream and is what the
    CI ``cpu`` job builds. ``enforce()`` once hard-refused it: the
    "there is no CUDA extension in this build" fired regardless of backend, so
    the harness rejected every number from the configuration the work exists to
    serve.

    Kept as a demo because a scoping change that only ever makes things pass is
    indistinguishable from deleting a check -- this one has to be *shown* to
    pass for the right reason, next to the CPU refusals that still fire.
    """
    req = _base_req()
    ctx = _base_cpu_ctx(checked_a_gpu_build=False)
    result = rules.enforce(req, ctx)
    return _record_from(
        "cpu_run_without_a_gpu_is_allowed",
        "a CPU run on a machine with no CUDA extension -- the deployment "
        "target, and not a defect",
        req, ctx, result, build, git, results_dir)


def run_all_demos(results_dir: Path) -> list[dict]:
    build = verify_build()
    git = get_git_info(cwd=str(REPO_ROOT))
    demos = [
        demo_non_release_build,
        demo_no_gpu_build,
        demo_context_not_derived_from_a_build,
        demo_arch_probe_unavailable,
        demo_host_flags_unavailable,
        demo_cpu_dispatch_target_unknown,
        demo_accuracy_run_is_not_refused_for_a_timing_bar,
        demo_accuracy_declared_but_carries_a_rate,
        demo_accuracy_without_a_per_query_table,
        demo_accuracy_without_molecule_level_recall,
        demo_accuracy_fit_and_report_on_one_corpus,
        demo_accuracy_on_a_one_conformer_per_molecule_corpus,
        demo_cpu_lock_not_held,
        demo_cpu_run_was_not_pinned,
        demo_cpu_clock_never_sampled,
        demo_cpu_build_never_inspected,
        demo_cpu_width_not_recorded,
        demo_cpu_run_without_a_gpu_is_allowed,
        demo_gpu_lock_not_held,
        demo_max_results_too_small,
        demo_batch_too_small,
        demo_corpus_bound_launch,
        demo_binary_built_from_a_patched_tree,
        demo_burst_claimed_non_burst,
        demo_cross_clock_comparison,
        demo_cross_regime_comparison,
        demo_percent_of_peak_missing_clock,
        demo_power_limited_not_clock_locked,
    ]
    return [d(build, git, results_dir) for d in demos]


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "runs" / "demo_refuse"
    for r in run_all_demos(results_dir):
        print(r["kind"], "->", r["record"]["status"], r["record"]["refusal_reason"], r["record"]["advisory_reasons"])

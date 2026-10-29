"""The standing regression benchmark's committed baseline, and what makes a
comparison against it valid.

One JSON file, `BASELINE_PATH`, holds one entry per tier: the
number(s) last accepted, the band a future run must stay inside, and the
provenance (machine, build, clock distribution, date) that number was taken
under. Re-baselining requires a stated reason, recorded alongside the old
value rather than over it, so a baseline's history says who moved it and why.

Comparability is checked here, separately from `bench.rules.enforce`'s own
per-run advisories. That is deliberate rather than redundant: NUMERICS_POLICY
keeps a toolkit mismatch merely advisory for a run's *own* citability, because
it is a much weaker perturbation than a forbidden flag. A regression
*comparison* is a different question -- an architecture, toolkit, lock or
clock confound is exactly the kind of thing that can make a real regression
invisible or a spurious one appear, so this module refuses outright on any of
them rather than folding the decision into the run's own advisory list.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from bench import rules

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "runs" / "regression_baseline"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
HISTORY_PATH = RESULTS_DIR / "baseline_history.jsonl"

TIERS = ("bit_exact", "recall", "throughput", "enrichment")


@dataclass
class Provenance:
    """What a number needs to say about itself before it can anchor a band."""

    utc: str
    git_sha: str
    hostname: str
    arch_profile: str
    arch_list: list
    nvcc_version: str
    cuda_src_sha256: str
    cpu_src_sha256: "str | None" = None
    cpu_isa_selected: "str | None" = None
    cpu_vector_width: "int | None" = None
    gpu_uuid: "str | None" = None
    gpu_name: "str | None" = None
    achieved_clock_mhz_median: "float | None" = None
    measurement_regime: "str | None" = None
    run_id: "str | None" = None
    padding_table: "list | None" = None
    """This run's `(bucket, L, L_color, n_conf)` histogram,
    canonicalized by `bench.rules.canonical_padding_table`. A throughput
    comparison whose library screened a different padded-width histogram
    from the baseline's is a comparison between data layouts, not kernels
    -- see `check_comparable`. `None` means this run did not record one,
    which is not the same claim as "the padding matched"."""

    # ---- which BINARY produced this number ---------------------------------
    # Everything above describes the source and the machine, not the
    # build. Without these fields a Debug build would be compared against a
    # Release baseline and graded a code regression, and a run against a
    # different `.so` from the baseline's would go unnoticed.
    #
    # Split deliberately into two classes, because they are not the same
    # question. `build_type` and the three flag strings are a build
    # CONFIGURATION: a difference there means the two numbers came from
    # differently-compiled code and `check_comparable` refuses. The `.so`
    # hash and the requested-architecture string are a build IDENTITY: they
    # move on an ordinary rebuild of identical source and identical
    # configuration, so a difference is reported by
    # `binary_identity_notes()` and is not a refusal -- see that function
    # for why refusing on them would refuse every rebuild.
    build_type: "str | None" = None
    cpu_build_type: "str | None" = None
    cxx_flags: "str | None" = None
    cuda_flags: "str | None" = None
    cpu_cxx_flags: "str | None" = None
    so_sha256: "dict | None" = None
    cuda_architectures_requested: "str | None" = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TierBaseline:
    tier: str
    summary: str
    """One line a human reads: what the number is and what it means."""
    value: Any
    """The accepted number(s) -- a float, or a small dict for a multi-arm tier
    (e.g. throughput's {"cuda": ..., "cpu": ...})."""
    band: dict
    """How far a future run may move before it is a regression. Always
    carries `kind` plus whatever `kind` needs -- see `evaluate_band`."""
    provenance: dict
    measured_from: dict = field(default_factory=dict)
    """The raw measurement(s) the value/band were derived from, so a reader
    does not have to trust the summary -- e.g. throughput's own repeat array."""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Baseline:
    created_utc: str
    reason: str
    git_sha: str
    tiers: dict            # tier name -> TierBaseline dict

    def as_dict(self) -> dict:
        return asdict(self)


def load_baseline(path: Path = BASELINE_PATH) -> "Baseline | None":
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    return Baseline(created_utc=doc["created_utc"], reason=doc["reason"],
                    git_sha=doc["git_sha"], tiers=doc["tiers"])


def write_baseline(tiers: dict, reason: str, *, path: Path = BASELINE_PATH,
                   history_path: Path = HISTORY_PATH) -> Baseline:
    """Replace the committed baseline. Requires a reason -- not a formality:
    the acceptance criterion is explicit that re-baselining to make a failure
    go away is the one thing this procedure must not silently permit, so the
    only guard available at this layer is that skipping the reason is not an
    option and the old baseline is preserved in the history log rather than
    overwritten in place."""
    if not reason or not reason.strip():
        raise ValueError(
            "re-baselining requires --rebaseline REASON: a baseline move with "
            "no recorded reason is indistinguishable from silencing a failure.")
    from bench.gitinfo import get_git_info

    old = load_baseline(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = Baseline(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  reason=reason, git_sha=get_git_info().sha, tiers=tiers)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps({
            "event": "rebaseline", "at": new.created_utc, "reason": reason,
            "new_git_sha": new.git_sha,
            "previous": old.as_dict() if old else None,
        }, default=str) + "\n")
    path.write_text(json.dumps(new.as_dict(), indent=1, default=str) + "\n")
    return new


# ---------------------------------------------------------------------------
# Comparability: refuse rather than warn
# ---------------------------------------------------------------------------

def check_comparable(current: dict, baseline: dict, *, require_lock: bool,
                     compare_arch: bool = True) -> list[str]:
    """Hard-refusal reasons a `current` provenance dict may not be compared
    against a `baseline` provenance dict. Empty means the comparison stands.

    `current` and `baseline` are `Provenance.as_dict()`-shaped. Fields that
    only apply to one side of a tier (e.g. `gpu_uuid` on a CPU-only tier) are
    `None` on both and never trigger a refusal -- only a *known* mismatch
    does, matching `bench.rules.check_regime_comparable`'s model.

    `compare_arch` selects whether the GPU architecture profile and SASS
    architecture list refuse the comparison. **It is a rate question, not an
    accuracy one**. A fat
    multi-arch binary allocates registers differently and can change
    occupancy relative to a single-arch build -- an effect on how FAST the
    kernel runs. Register allocation cannot move a recall number, and the
    other half was measured directly: Volta sm_70 against Blackwell
    sm_120, identical source and bit-identical inputs, only 0.12% of pairs
    bit-identical after 100 Adagrad steps and yet top-1 and top-5 sets *and
    order* identical for 480/480 queries (top-10 for 479/480). Refusing an
    accuracy comparison across architectures therefore refuses on a
    confound that has been measured not to reach the number being compared,
    and would leave the `cheap` merge gate -- `bit_exact` and `recall`,
    neither of which reports a rate -- unable to pass on a fat wheel or any
    other architecture's build.
    """
    reasons = []

    def differs(key, label):
        a, b = current.get(key), baseline.get(key)
        if a is not None and b is not None and a != b:
            reasons.append(f"{label} differs: this run={a!r}, baseline={b!r}")

    if compare_arch:
        differs("arch_profile", "GPU architecture profile")
        if current.get("arch_list") and baseline.get("arch_list"):
            if sorted(current["arch_list"]) != sorted(baseline["arch_list"]):
                reasons.append(
                    "GPU SASS architecture list differs: this run="
                    f"{sorted(current['arch_list'])}, baseline="
                    f"{sorted(baseline['arch_list'])} (wrong-architecture binary family)")
    differs("nvcc_version", "nvcc version")
    # The build CONFIGURATION half of "which binary produced this number".
    # A Debug build against a Release baseline is not a code regression, and
    # before this it graded as one; a changed flag string is NUMERICS_POLICY's
    # own axis and is refused on the same principle the toolkit version is.
    # Deliberately NOT gated on `compare_arch`: -O0 host code and
    # a reassociating flag reach a rate and an accuracy number alike.
    differs("build_type", "CUDA extension build type")
    differs("cpu_build_type", "CPU extension build type")
    differs("cxx_flags", "host compiler flags")
    differs("cuda_flags", "nvcc flags")
    differs("cpu_cxx_flags", "CPU extension compiler flags")
    differs("gpu_uuid", "physical GPU (uuid)")
    differs("cpu_isa_selected", "CPU SIMD dispatch target")
    differs("cpu_vector_width", "CPU vector width")
    differs("measurement_regime", "measurement regime")

    a_pad, b_pad = current.get("padding_table"), baseline.get("padding_table")
    if a_pad is not None and b_pad is not None:
        if rules.canonical_padding_table(a_pad) != rules.canonical_padding_table(b_pad):
            reasons.append(
                "padded-width histogram differs from the baseline's"
                f": this run={rules.canonical_padding_table(a_pad)}, "
                f"baseline={rules.canonical_padding_table(b_pad)} -- two "
                "configurations compared for throughput must screen "
                "identical L histograms, or the comparison is between data "
                "layouts and not between kernels")

    if require_lock:
        a_clock = current.get("achieved_clock_mhz_median")
        b_clock = baseline.get("achieved_clock_mhz_median")
        if a_clock is not None and b_clock is not None:
            if not rules.check_clock_comparable(a_clock, b_clock):
                reasons.append(
                    f"achieved SM clock differs by more than "
                    f"{rules.CLOCK_MATERIAL_RATIO}x: this run={a_clock:.0f} "
                    f"MHz, baseline={b_clock:.0f} MHz (numbers taken at "
                    "different clocks may never be combined)")
    return reasons


def binary_identity_notes(current: dict, baseline: dict) -> list[str]:
    """Non-refusing facts about *which binary* produced each side.

    Separate from :func:`check_comparable` because the honest answer here is
    "different, and that is usually fine": an ordinary rebuild of identical
    source with an identical configuration produces a different `.so` hash,
    so refusing on it would refuse every rebuild, even one whose SASS is
    byte-identical to the baseline's. What is needed is not a refusal but a
    STATEMENT that the two sides are different binaries; without it "benign
    rebuild" and "silently different build" serialize identically. These notes are attached to the grade and
    change no verdict.
    """
    notes = []
    a_so = (current.get("so_sha256") or {}).get("_roshambo3_cuda")
    b_so = (baseline.get("so_sha256") or {}).get("_roshambo3_cuda")
    if a_so and b_so and a_so != b_so:
        notes.append(
            f"the CUDA extension is not the baseline's binary (this run "
            f"{a_so[:12]}, baseline {b_so[:12]}). Build type, compiler flags "
            "and the embedded SASS arch list all match, so this is a rebuild "
            "of the same configuration rather than a reconfiguration -- but "
            "it is a different binary and the comparison should be read "
            "knowing that.")
    a_cpu = (current.get("so_sha256") or {}).get("_roshambo3_cpu")
    b_cpu = (baseline.get("so_sha256") or {}).get("_roshambo3_cpu")
    if a_cpu and b_cpu and a_cpu != b_cpu:
        notes.append(
            f"the CPU extension is not the baseline's binary (this run "
            f"{a_cpu[:12]}, baseline {b_cpu[:12]}).")
    a_req = current.get("cuda_architectures_requested")
    b_req = baseline.get("cuda_architectures_requested")
    if a_req and b_req and a_req != b_req:
        notes.append(
            f"CMAKE_CUDA_ARCHITECTURES was {a_req!r} for this run and "
            f"{b_req!r} for the baseline. Not a refusal: what a comparison "
            "needs is the arch list actually EMBEDDED in the loaded .so "
            "(`arch_list`, recovered by cuobjdump rather than trusted from "
            "CMake), and two different requests can resolve to the same one "
            "-- 'native' and '120-real' both resolve to ['sm_120'] on this "
            "box. Reported because it is the field that distinguishes two "
            "builds a reader might otherwise assume were the same one.")
    return notes


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def evaluate_band(value: float, baseline_value: float, band: dict
                  ) -> tuple[bool, str]:
    """(ok, explanation) -- whether `value` is still inside `band` around
    `baseline_value`. `band["kind"]` selects the arithmetic; see the four
    tiers for which kind each one uses and why."""
    kind = band["kind"]
    if kind == "exact":
        ok = value == baseline_value
        return ok, (f"{value} == {baseline_value}" if ok else
                    f"{value} != {baseline_value} (exact match required)")
    if kind == "abs_tolerance":
        tol = float(band["tolerance"])
        signed = value - baseline_value
        diff = abs(signed)
        # `direction`. The default stays
        # two-sided so nothing that predates this key changes meaning, but a
        # tier whose metric has a preferred direction says so: an accuracy
        # tier's band exists to catch accuracy going DOWN, and as a symmetric
        # band it graded a large IMPROVEMENT a "regression". That is not
        # hypothetical on either accuracy tier here -- FA10 gaining 3 actives
        # in its top 13 moves the old enrichment mean by +5.96 against a 4.0 band, i.e. "the most likely
        # way this tier ever fires is on an improvement", and recall's own
        # 0.995 baseline sits 0.005 below the metric's 1.0 ceiling against a
        # 0.004 band, so a genuine rise to a perfect 1.000 would have graded
        # `regression` too. An upward move past the band is still REPORTED,
        # loudly, in the explanation: a jump nobody intended is worth
        # investigating, it is just not a regression.
        if band.get("direction") == "lower":
            ok = signed >= -tol
            if ok and signed > tol:
                return True, (
                    f"{value:g} vs baseline {baseline_value:g}: +{signed:g} "
                    f"> {tol:g}, an IMPROVEMENT past the band -- not a "
                    "regression (this band is one-sided), but a movement "
                    "this large was not expected either; confirm it is "
                    "intended rather than a changed reference")
            return ok, (f"{value:g} - {baseline_value:g} = {signed:+g} "
                        f"{'>=' if ok else '<'} -{tol:g} (one-sided: only a "
                        "drop is a regression)")
        ok = diff <= tol
        return ok, (f"|{value:g} - {baseline_value:g}| = {diff:g} "
                    f"{'<=' if ok else '>'} {tol:g}")
    if kind == "lower_bound_fraction":
        frac = float(band["fraction"])
        floor = baseline_value * (1.0 - frac)
        ok = value >= floor
        return ok, (f"{value:,.1f} {'>=' if ok else '<'} baseline*"
                    f"(1-{frac:.3f}) = {floor:,.1f} (baseline {baseline_value:,.1f})")
    raise ValueError(f"unknown band kind {kind!r}")


def build_provenance(build, *, git_sha: str, hostname: str,
                     gpu_uuid: "str | None" = None,
                     gpu_name: "str | None" = None,
                     achieved_clock_mhz_median: "float | None" = None,
                     measurement_regime: "str | None" = None,
                     run_id: "str | None" = None,
                     padding_table: "list | None" = None) -> Provenance:
    return Provenance(
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_sha=git_sha, hostname=hostname,
        arch_profile=build.arch_profile, arch_list=list(build.arch_list),
        nvcc_version=build.nvcc_version,
        cuda_src_sha256=build.cuda_src_sha256,
        cpu_src_sha256=(build.cpu_src_sha256 if build.with_cpu else None),
        cpu_isa_selected=build.cpu_selected_target,
        cpu_vector_width=build.cpu_vector_width,
        gpu_uuid=gpu_uuid, gpu_name=gpu_name,
        achieved_clock_mhz_median=achieved_clock_mhz_median,
        measurement_regime=measurement_regime, run_id=run_id,
        padding_table=padding_table,
        # Read off the same VerifiedBuild every other field here comes from,
        # rather than accepted from the caller: a field a caller can supply is
        # a field a caller can leave unset.
        build_type=build.build_type,
        cpu_build_type=(build.cpu_build_type if build.with_cpu else None),
        cxx_flags=build.cxx_flags,
        cuda_flags=build.cuda_flags,
        cpu_cxx_flags=(build.cpu_cxx_flags if build.with_cpu else None),
        so_sha256=dict(build.so_sha256 or {}),
        cuda_architectures_requested=build.cuda_architectures_requested)

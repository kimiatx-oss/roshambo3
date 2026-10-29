"""One call that takes the lock, measures the machine, grades the number, and
writes the record -- so a CPU script gets a graded run record without
remembering to.

**Why this exists rather than a note in a README.** The CPU rules all worked,
yet the CPU numbers produced under them went ungraded, because the obedient
path was harder than the disobedient one at every step. There was no ``CpuLock`` to call, so
taking the lock meant hand-rolling ``flock`` in shell (4 of 31 scripts did).
:class:`bench.rules.RunContext` needs seven GPU-shaped fields a CPU run has no
answer for. And ``enforce()`` refused a CPU run outright for GPU reasons until
those rules were scoped. Against all that, ``json.dump`` was one line.

So the rule this module follows is: **the graded path must be shorter than the
ungraded one.** Not documented better -- shorter.

Usage, and it is meant to replace the ``json.dump`` a timing script ends with::

    from bench.cpu_run import CpuRun

    with CpuRun("my-benchmark", decision_grade=True, cpus=physical_core_cpus()) as run:
        rate = time_the_thing()                 # the lock is held here
        run.timing = {"start_pairs_per_s": rate}
        run.config = {"grain": "conformer", "width": 8, "exp2": "minimax_d5"}

    print(run.status)                           # decision / advisory / refused
    run.write(Path("runs/my-benchmark.jsonl"))

The context manager holds the lock for the body and nothing else, which is the
window that has to be exclusive. Everything the rules need is collected around
it rather than asked of the caller.
"""
from __future__ import annotations

import json
from pathlib import Path

from bench import rules
from bench.build_check import verify_build
from bench.cpu import (CpuFrequencySampler, CpuLock, CpuLockBusyError,
                       cpu_identity, pin_to, query_cpu_state)
from bench.gitinfo import get_git_info
from bench.run_record import append_jsonl, build_run_record


class CpuRun:
    """A timed CPU run that produces a graded record.

    Args:
        ticket_id: the label of the work this number belongs to.
        decision_grade: whether the number will be quoted. **The CPU lock is
            only required for a decision-grade run**, matching the rule --
            correctness work runs unlocked by design.
        cpus: the CPU set to pin to. ``None`` leaves affinity alone, which is
            an advisory on a timed run: affinity is the one thing controllable
            on these machines, so declining it declines the only control there
            is.
        sample_frequency: sample achieved MHz for the duration. Costs a poller
            process; it was measured perturbing its own measurement by 2.7x when
            it landed on a measured core, which ``CpuFrequencySampler`` now
            handles by displacing itself.
        threads: how many threads the run used. Recorded, never thresholded --
            the CPU analogue of the GPU launch-size floor has not been measured.
        blocking: wait for the lock rather than failing fast.
    """

    def __init__(self, ticket_id: str, *, decision_grade: bool = True,
                 cpus: "list | None" = None, sample_frequency: bool = True,
                 threads: "int | None" = None, blocking: bool = False,
                 measurement_regime: str = "calibration",
                 backend_name: str = "cpu",
                 kind: str = rules.RUN_KIND_RATE,
                 chose_a_value: "str | None" = None):
        self.ticket_id = ticket_id
        self.decision_grade = decision_grade
        self.cpus = cpus
        self.sample_frequency = sample_frequency
        self.threads = threads
        self.blocking = blocking
        self.measurement_regime = measurement_regime
        self.backend_name = backend_name
        # Defaults to "rate" so every existing caller keeps its gates.
        # An accuracy run declares kind="accuracy"; `enforce` then skips the
        # three timing gates and applies the accuracy ones instead, and refuses
        # the run outright if the record turns out to carry a throughput after
        # all. See rules.RunRequest.kind.
        self.kind = kind
        self.chose_a_value = chose_a_value

        # Filled by the body. Deliberately plain attributes: a script that has
        # to learn an API to record a number will go back to json.dump.
        self.timing: dict = {}
        self.metrics: dict = {}
        self.config: dict = {}
        self.dataset: dict = {}
        self.query_panel: dict = {"name": "n/a", "sha256": "n/a", "ids": []}
        # Molecule-vs-conformer and tune-vs-report evidence, read by the
        # accuracy gates.
        self.conformers_per_molecule: "float | None" = None
        self.fit_corpus: "str | None" = None
        self.report_corpus: "str | None" = None
        self.audits: dict = {}
        self.artifacts: dict = {}
        self.notes: str = ""

        self._lock = None
        self._sampler = None
        self._before = None
        self._after = None
        self._affinity = None
        self.lock_busy_reason: "str | None" = None
        self.result: "rules.EnforcementResult | None" = None
        self.record = None

    # -- the timed region -------------------------------------------------
    def __enter__(self) -> "CpuRun":
        # Order matters: pin first so the sampler can displace itself off the
        # measured cores, then take the lock, then start sampling. The lock is
        # taken before any measurement begins, as on the GPU side -- there
        # is no CUDA context here to deadlock against, but "arbitrate before
        # you allocate" is the same discipline.
        if self.cpus:
            self._affinity = pin_to(self.cpus)
        else:
            self._affinity = None

        if self.decision_grade:
            try:
                self._lock = CpuLock(self.ticket_id, blocking=self.blocking)
                self._lock.acquire()
            except CpuLockBusyError as exc:
                # Not fatal here. The run may still be worth taking; it is
                # simply not decision-grade, and enforce() will say so rather
                # than this constructor deciding on the caller's behalf.
                self.lock_busy_reason = str(exc)
                self._lock = None

        self._before = query_cpu_state()
        if self.sample_frequency:
            # A context manager, entered by hand rather than through
            # ExitStack: the after-state must be queried between stopping the
            # sampler and releasing the lock, and that ordering is not
            # expressible as plain nesting. `cpus` must be a set -- the
            # sampler subtracts it from the online set to displace its poller
            # off the measured cores (a measured 2.7x self-perturbation).
            self._sampler = CpuFrequencySampler(
                cpus=set(self.cpus) if self.cpus else None)
            self._sampler.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._sampler is not None:
            self._sampler.__exit__(None, None, None)
        self._after = query_cpu_state()
        if self._lock is not None:
            self._lock.release()
        if exc_type is None:
            self._grade()
        return False

    # -- grading ----------------------------------------------------------
    def _grade(self) -> None:
        build = verify_build()
        arith = _check_build()

        # Contention is the union of before and after: a foreign process that
        # arrived mid-run and left is still contention this number paid for,
        # and looking only at one end is how it goes unrecorded.
        foreign = list({str(p) for p in
                        (list(self._before.foreign_processes)
                         + list(self._after.foreign_processes))})
        busy = bool(self._before.busy or self._after.busy)

        ctx = rules.RunContext.from_build(
            build, arith,
            backend_name=self.backend_name,
            measurement_regime=self.measurement_regime,
            cpu_identity=cpu_identity().as_dict(),
            cpu_lock_held=bool(self._lock is not None
                               and self._lock.was_held),
            cpu_lock_acquisition_failed=self.lock_busy_reason is not None,
            cpu_busy=busy,
            cpu_foreign_processes=foreign,
            # NOT `or current_affinity()`. An unpinned process reports the
            # full machine mask, so falling back to it hands enforce() a
            # pinned-looking value and the "was not pinned" advisory becomes
            # unreachable -- while the record below still writes None, so the
            # record and the grade disagreed about the same fact: a green record
            # that checked nothing.
            cpu_affinity=self._affinity,
            cpu_threads=self.threads,
            achieved_cpu_mhz=(self._sampler.summary()
                              if self._sampler is not None else None),
            # GPU-shaped and required. A CPU run has no answer, and the honest
            # answer is the one that claims nothing: no GPU lock was held, no
            # clocks were locked, nothing throttled, no kernel ran. The rules
            # that read these are scoped away from CPU runs, so they
            # record rather than decide.
            # SOURCE dirtiness, not whole-tree. `run.write()` may append to a
            # tracked file, so grading a run would dirty the tree and
            # downgrade the NEXT one to advisory on nothing but its
            # predecessor's ledger entry. The
            # whole-tree flag is still written into the record; only the GRADE
            # stops depending on the run's own output.
            git_dirty=get_git_info().source_dirty,
            gpu_lock_held=False,
            gpu_lock_acquisition_failed=False,
            other_process_on_gpu=False,
            clocks_locked=False,
            throttle_fraction=0.0,
            sustained_kernel_s=float(self.timing.get("sustained_s", 0.0) or 0.0),
            timing=dict(self.timing),
            metrics=dict(self.metrics),
            # The rest of the record, so the run-kind contract sees all of it.
            # Parking a rate in `config` or `audits` used to walk past the
            # check entirely.
            # EVERY caller-writable block that reaches the record, including
            # `query_panel` (per-query rates beside the panel is a natural
            # shape); "the rest of the record" has to mean all of it or the
            # contract is a statement about which blocks someone remembered.
            record_blocks={"config": dict(self.config),
                           "dataset": dict(self.dataset),
                           "audits": dict(self.audits),
                           "artifacts": dict(self.artifacts),
                           "query_panel": dict(self.query_panel),
                           "notes": self.notes},
            conformers_per_molecule=self.conformers_per_molecule,
            fit_corpus=self.fit_corpus,
            report_corpus=self.report_corpus,
        )
        req = rules.RunRequest(
            decision_grade=self.decision_grade,
            n_conformers=int(self.dataset.get("n_conformers", 0) or 0),
            max_results=int(self.config.get("max_results", 0) or 0),
            claim_non_burst=False,
            n_repeats=int(self.timing.get("n_repeats", 1) or 1),
            n_warmup=int(self.timing.get("n_warmup", 0) or 0),
            max_results_applied_per_batch=False,
            kind=self.kind,
            chose_a_value=self.chose_a_value,
        )
        self.result = rules.enforce(req, ctx)
        self.record = build_run_record(
            ticket_id=self.ticket_id, git=get_git_info(), build=build,
            clocks_locked=False, gpu_lock_held=False,
            dataset=self.dataset, query_panel=self.query_panel,
            config=dict(self.config,
                        # Without this the ledger cannot say which gate list
                        # issued a grade, so re-grading from the ledger alone is
                        # impossible.
                        run_kind=self.kind,
                        backend=self.backend_name,
                        cpu_isa_selected=ctx.cpu_isa_selected,
                        cpu_vector_width=ctx.cpu_vector_width,
                        cpu_threads=self.threads),
            timing=self.timing, metrics=self.metrics, audits=self.audits,
            artifacts=self.artifacts,
            status=self.result.status,
            advisory_reasons=self.result.advisory_reasons,
            refusal_reason=self.result.refusal_reason,
            notes=self.notes,
            operating_point={
                "achieved_clock_mhz": (
                    self._sampler.summary().get("achieved_mhz")
                    if self._sampler is not None else None),
                "cpu_lock_held": bool(self._lock is not None
                                      and self._lock.was_held),
                "cpu_affinity": self._affinity,
                "cpu_state_before": self._before.as_dict(),
                "cpu_state_after": self._after.as_dict(),
            },
        )

    @property
    def status(self) -> str:
        if self.result is None:
            return "not-graded"
        return self.result.status

    def write(self, path: "str | Path") -> str:
        """Append the graded record, and return its status."""
        if self.record is None:
            raise RuntimeError(
                "no record to write: the timed block did not complete, so "
                "nothing was graded. A partial run must not be recorded as if "
                "it were a whole one.")
        append_jsonl(self.record, path)
        return self.status


def _check_build():
    from roshambo3 import numerics
    return numerics.check_build()


def summarize(path: "str | Path") -> dict:
    """Grade counts over a runs.jsonl, for a ticket's write-up."""
    path = Path(path)
    if not path.exists():
        return {"n": 0, "by_status": {}}
    by = {}
    n = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by[rec["status"]] = by.get(rec["status"], 0) + 1
        n += 1
    return {"n": n, "by_status": by}


# ---------------------------------------------------------------------------
# The backstop, for the script that forgets anyway
# ---------------------------------------------------------------------------

_RATE_KEY = __import__("re").compile(
    r"^(.*_)?(pairs_per_s|start_pairs_per_s|calls_per_s|conf(ormers)?_per_s"
    r"|throughput|rate|median_s|wall_s|duration_s|elapsed_s|seconds)(_.*)?$",
    __import__("re").I)


def _keys(o, out=None, d=0):
    out = set() if out is None else out
    if d > 12:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(k)
            _keys(v, out, d + 1)
    elif isinstance(o, list):
        for v in o[:200]:
            _keys(v, out, d + 1)
    return out


#: Keys that mark an artifact as a CPU run. Chosen by MEASUREMENT, not by
#: intuition: each appears in at least one CPU-ladder timed artifact and in
#: **zero** non-ladder ones, checked across every committed result JSON in the
#: development repository. Keys that seemed obviously CPU-ish and are not, with their
#: non-ladder counts: ``cpu_state_before`` (52), ``cpu_state_after`` (47),
#: ``physical_cores`` (1), ``isa_flags`` (1), ``grade_note`` (2). Including any
#: of them fails historical GPU directories for a rule written about the CPU.
#:
#: **This gate has a known hole and it is stated here rather than implied.** It
#: catches 11 of the ladder's 15 timed artifacts. The four it misses record
#: *nothing at all* about the machine they ran on, so there is nothing to key on. That is
#: itself the finding: an artifact anonymous enough to evade this check is an
#: artifact whose number could not have been graded anyway. Closing the hole
#: would require inverting the gate to "not GPU-marked", which was measured and
#: fails 23 historical directories. Use :class:`CpuRun` and the question does
#: not arise; this is the backstop for when it was not used.
_CPU_MARKER_KEYS = frozenset({
    "achieved_cpu_mhz", "achieved_mhz_spread", "compiled_targets",
    "cpu_src_sha256", "cpus_sampled", "have_avx512", "logical_cpus",
    "median_is_over_sampled_cpus_only", "pinned_cpus",
    "poller_displaced_off_measured_cpus", "selected_target", "smt_note",
    "vector_width",
})


def looks_like_a_cpu_timed_artifact(doc) -> bool:
    """A committed JSON that reports a CPU rate.

    Two conditions, both necessary. A **rate or elapsed time**, because the
    lock and clock rules are about timed runs and the lock rule permits
    correctness work to run unlocked. And a **CPU build stamp**, so the check
    does not sweep in the GPU ledger, which predates all of this and is graded
    through its own path.
    """
    if not isinstance(doc, dict):
        return False
    keys = _keys(doc)
    if not any(_RATE_KEY.match(k) for k in keys):
        return False
    return bool(_CPU_MARKER_KEYS & keys)


def ungraded_timed_artifacts(results_dir: "str | Path") -> list:
    """Timed CPU artifacts in ``results_dir`` with no ``runs.jsonl`` beside them.

    The backstop for :class:`CpuRun`. The helper makes the graded path the
    short one; this catches the script that took the other path anyway --
    *a mechanism that would pass a forgetful script has not been tested.*
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    timed = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if looks_like_a_cpu_timed_artifact(doc):
            timed.append(f.name)
    if not timed:
        return []
    if (results_dir / "runs.jsonl").exists():
        return []
    return timed

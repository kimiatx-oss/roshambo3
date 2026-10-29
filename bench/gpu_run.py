"""One call that takes the GPU lock, measures the machine, grades the number,
and writes the record -- the GPU analogue of :mod:`bench.cpu_run`.

**Why this exists rather than another hand-assembled sequence.** Every GPU
benchmark script wires ``GpuLock`` + ``query_gpu_state`` + ``ClockPowerSampler``
+ ``detect_profiler_clock_control`` + ``verify_build`` + ``numerics.check_build``
+ ``RunContext.from_build`` + ``RunRequest`` + ``enforce`` by hand, in
whatever order and with whatever subset of fields that script's author
remembered. Even a committed, working benchmark can skip the last step:
one assembled the first eight pieces
correctly and then hand-writes its own bespoke JSON dict instead of calling
``build_run_record()``/``append_jsonl()`` -- so it never appends to a
``runs.jsonl`` at all. That is ``bench.cpu_run``'s own defect class
("a check that runs and does not record") with the devices swapped: not
"graded but not written", but a real GPU throughput number with no standing
ledger entry anywhere.

So this module follows ``bench.cpu_run``'s rule unchanged: **the graded,
recorded path must be shorter than the ad hoc one.**

Usage, replacing a hand-assembled sequence::

    from bench.gpu_run import GpuRun

    with GpuRun("my-benchmark", decision_grade=True,
                measurement_regime="headline") as run:
        # the GPU lock is held from here; the CUDA extension may be
        # imported and a context created only inside this block
        from roshambo3.engine import Engine, R3gSource, ScreenConfig, TopK
        ...run the timed screen...
        run.timing = {"conformers_per_s": rate, "sustained_s": kernel_s}
        run.config = {...}
        run.dataset = {...}

    print(run.status)                       # decision / advisory / refused
    run.write(Path("runs/my-benchmark.jsonl"))

The device is resolved and the lock acquired in ``__enter__`` using
only ``nvidia-smi`` subprocess calls -- never a CUDA runtime/driver call --
so the lock is always taken before the body creates a CUDA context, never
after.
"""
from __future__ import annotations

import os
from pathlib import Path

from bench import rules
from bench.build_check import verify_build
from bench.clocks import ClockPowerSampler, detect_profiler_clock_control
from bench.gitinfo import get_git_info
from bench.gpu_lock import (DeviceIdentity, GpuLock, GpuLockBusyError,
                            query_gpu_state, resolve_default_device)
from bench.run_record import append_jsonl, build_run_record


class GpuRun:
    """A timed GPU run that produces a graded, recorded run record.

    Args:
        ticket_id: the label of the work this number belongs to.
        decision_grade: whether the number will be quoted. **The GPU flock is
            only attempted for a decision-grade run**, matching
            :class:`bench.cpu_run.CpuRun` -- correctness work runs unlocked by
            design.
        device: the physical GPU (:class:`bench.gpu_lock.DeviceIdentity`).
            ``None`` resolves via :func:`resolve_default_device`, which
            succeeds only when exactly one GPU is enumerated -- correct on
            a single-GPU host, and a deliberate refusal to guess on a multi-GPU host.
        measurement_regime: ``"calibration"`` or ``"headline"``.
            No default: a caller that has not decided which regime this run
            is in must say so explicitly, the same discipline
            ``RunContext.measurement_regime`` itself enforces.
        sample_interval_s: how often :class:`bench.clocks.ClockPowerSampler`
            polls ``nvidia-smi`` for the run's duration.
        blocking: wait for the GPU lock rather than failing fast.
        kind: :data:`bench.rules.RUN_KIND_RATE` (default) or
            :data:`bench.rules.RUN_KIND_ACCURACY` -- the run-kind contract.
        gold_ref_run_id / gold_ref_measurement_regime / gold_ref_achieved_clock_mhz_median
            / gold_ref_gpu_uuid / gold_ref_arch_profile / gold_ref_padding_table:
            the prior run this one is being compared against, if any -- passed
            straight through to ``RunContext`` so ``enforce()`` can refuse an
            invalid comparison (clocks, regime, device,
            padding) instead of silently producing a number.
    """

    def __init__(self, ticket_id: str, *, decision_grade: bool = True,
                 device: "DeviceIdentity | None" = None,
                 measurement_regime: str,
                 sample_interval_s: float = 0.5,
                 blocking: bool = False,
                 kind: str = rules.RUN_KIND_RATE,
                 chose_a_value: "str | None" = None,
                 padding_table: "list | None" = None,
                 gold_ref_run_id: "str | None" = None,
                 gold_ref_measurement_regime: "str | None" = None,
                 gold_ref_achieved_clock_mhz_median: "float | None" = None,
                 gold_ref_gpu_uuid: "str | None" = None,
                 gold_ref_arch_profile: "str | None" = None,
                 gold_ref_padding_table: "list | None" = None):
        self.ticket_id = ticket_id
        self.decision_grade = decision_grade
        self._device_arg = device
        self.measurement_regime = measurement_regime
        self.sample_interval_s = sample_interval_s
        self.blocking = blocking
        self.kind = kind
        self.chose_a_value = chose_a_value
        self.padding_table = padding_table
        self.gold_ref_run_id = gold_ref_run_id
        self.gold_ref_measurement_regime = gold_ref_measurement_regime
        self.gold_ref_achieved_clock_mhz_median = gold_ref_achieved_clock_mhz_median
        self.gold_ref_gpu_uuid = gold_ref_gpu_uuid
        self.gold_ref_arch_profile = gold_ref_arch_profile
        self.gold_ref_padding_table = gold_ref_padding_table

        # Filled by the body. Plain attributes, same rationale as CpuRun: a
        # script that has to learn an API to record a number goes back to
        # json.dump.
        self.timing: dict = {}
        self.metrics: dict = {}
        self.config: dict = {}
        self.dataset: dict = {}
        self.audits: dict = {}
        self.artifacts: dict = {}
        self.notes: str = ""
        self.launch: "rules.LaunchShape | None" = None
        self.device_stage_seconds: "dict | None" = None
        """This run's `ScreenStats.device_stages` -- pass it through so
        `rules.launch_overhead_fraction` can actually fire. Without it the
        launch-overhead advisory never fires from a GpuRun; measured
        fractions are 1.6-4.3%, against a 5% threshold."""
        self.measured_total_s: "float | None" = None
        """The denominator `launch_overhead_fraction` divides by -- see
        `rules.RunContext.measured_total_s`'s own docstring on why this is
        NOT `launch_s`."""
        self.conformers_per_molecule: "float | None" = None
        self.fit_corpus: "str | None" = None
        self.report_corpus: "str | None" = None
        self.padding_change_is_the_subject: bool = False
        # True for the shipped device-resident top-K
        # -- see bench.rules.RunRequest.max_results_applied_per_batch.
        # Defaults to the CONSERVATIVE legacy assumption (per-batch
        # truncation is a live hazard) so a caller must affirmatively declare
        # a device-resident top-K rather than the harness assuming one.
        self.max_results_applied_per_batch: bool = True

        self.device: "DeviceIdentity | None" = None
        self.profiler = None
        self._lock: "GpuLock | None" = None
        self._sampler: "ClockPowerSampler | None" = None
        self._gpu_before = None
        self._gpu_after = None
        self.lock_busy_reason: "str | None" = None
        self.result: "rules.EnforcementResult | None" = None
        self.record = None

    # -- the timed region -------------------------------------------------
    def __enter__(self) -> "GpuRun":
        # Every call in this method is an `nvidia-smi` subprocess -- no CUDA
        # runtime/driver call, no context. The lock must be acquired
        # before the body (which is free to `import roshambo3`) ever runs.
        self.device = (self._device_arg if self._device_arg is not None
                       else resolve_default_device())
        self.profiler = detect_profiler_clock_control()

        if self.decision_grade:
            try:
                self._lock = GpuLock(self.ticket_id, self.device,
                                     blocking=self.blocking)
                self._lock.acquire()
            except GpuLockBusyError as exc:
                # Not fatal here, mirroring CpuRun: the run may still be
                # worth taking, it is simply not decision-grade, and
                # enforce() says so rather than this constructor deciding.
                self.lock_busy_reason = str(exc)
                self._lock = None

        mine = {os.getpid(), os.getppid()}
        self._gpu_before = query_gpu_state(self.device, exclude_pids=mine)
        self._sampler = ClockPowerSampler(self.device,
                                          interval_s=self.sample_interval_s)
        self._sampler.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._sampler is not None:
            self._sampler.__exit__(None, None, None)
        mine = {os.getpid(), os.getppid()}
        self._gpu_after = query_gpu_state(self.device, exclude_pids=mine)
        if self._lock is not None:
            self._lock.release()
        if exc_type is None:
            self._grade()
        return False

    # -- grading ------------------------------------------------------------
    def _grade(self) -> None:
        from roshambo3 import numerics

        build = verify_build()
        arith = numerics.check_build()
        clocks = self._sampler.summary() if self._sampler is not None else {}
        achieved = (clocks.get("achieved_clock_mhz") or {})

        foreign_before = bool(self._gpu_before.other_process_present)
        foreign_after = bool(self._gpu_after.other_process_present)

        ctx = rules.RunContext.from_build(
            build, arith,
            backend_name="cuda",
            measurement_regime=self.measurement_regime,
            git_dirty=get_git_info().source_dirty,
            gpu_lock_held=bool(self._lock is not None and self._lock.was_held),
            gpu_lock_acquisition_failed=(self.decision_grade
                                        and self.lock_busy_reason is not None),
            other_process_on_gpu=(foreign_before or foreign_after),
            # `nvidia-smi -lgc` may be refused to an unprivileged user, and
            # this module never attempts it: a number meant to be quoted
            # runs unlocked at max power (headline regime), and
            # asking for a lock only matters for an internal-parity
            # calibration arm, which is a decision each caller makes by
            # setting measurement_regime="calibration" and is still reported
            # as "not locked" here, honestly, rather than as a failed attempt.
            clocks_locked=False,
            throttle_fraction=float(clocks.get("throttle_fraction", 0.0) or 0.0),
            sustained_kernel_s=float(self.timing.get("sustained_s", 0.0) or 0.0),
            timing=dict(self.timing),
            metrics=dict(self.metrics),
            record_blocks={"config": dict(self.config),
                           "dataset": dict(self.dataset),
                           "audits": dict(self.audits),
                           "artifacts": dict(self.artifacts),
                           "query_panel": self.dataset.get("query_panel", {}),
                           "notes": self.notes},
            conformers_per_molecule=self.conformers_per_molecule,
            fit_corpus=self.fit_corpus,
            report_corpus=self.report_corpus,
            gpu_uuid=self.device.uuid,
            achieved_clock_mhz_median=achieved.get("median"),
            sw_power_cap_fraction=float(clocks.get("sw_power_cap_fraction", 0.0) or 0.0),
            profiler_control_state=self.profiler.state,
            profiler_control_tool=self.profiler.tool,
            padding_table=self.padding_table,
            device_stage_seconds=self.device_stage_seconds,
            measured_total_s=self.measured_total_s,
            gold_ref_run_id=self.gold_ref_run_id,
            gold_ref_measurement_regime=self.gold_ref_measurement_regime,
            gold_ref_achieved_clock_mhz_median=self.gold_ref_achieved_clock_mhz_median,
            gold_ref_gpu_uuid=self.gold_ref_gpu_uuid,
            gold_ref_arch_profile=self.gold_ref_arch_profile,
            gold_ref_padding_table=self.gold_ref_padding_table,
        )
        req = rules.RunRequest(
            decision_grade=self.decision_grade,
            n_conformers=int(self.dataset.get("n_conformers", 0) or 0),
            max_results=int(self.config.get("max_results", 0) or 0),
            claim_non_burst=False,
            n_repeats=int(self.timing.get("n_repeats", 1) or 1),
            n_warmup=int(self.timing.get("n_warmup", 0) or 0),
            padding_change_is_the_subject=self.padding_change_is_the_subject,
            launch=self.launch,
            kind=self.kind,
            chose_a_value=self.chose_a_value,
            max_results_applied_per_batch=self.max_results_applied_per_batch,
        )
        self.result = rules.enforce(req, ctx)
        self.record = build_run_record(
            ticket_id=self.ticket_id, git=get_git_info(), build=build,
            clocks_locked=False, gpu_lock_held=ctx.gpu_lock_held,
            dataset=self.dataset, query_panel=self.dataset.get("query_panel", {}),
            config=dict(self.config, run_kind=self.kind, backend="cuda"),
            timing=self.timing, metrics=self.metrics, audits=dict(
                self.audits,
                gpu_before=[a.__dict__ for a in self._gpu_before.compute_apps],
                gpu_after=[a.__dict__ for a in self._gpu_after.compute_apps],
                profiler_control=self.profiler.as_dict(),
            ),
            artifacts=self.artifacts,
            status=self.result.status,
            advisory_reasons=self.result.advisory_reasons,
            refusal_reason=self.result.refusal_reason,
            notes=self.notes,
            gold_ref_run_id=self.gold_ref_run_id,
            operating_point={
                "achieved_clock_mhz": achieved,
                # A clock distribution with no sample count is
                # indistinguishable from one resting on a single sample
                # (bench/clocks.py's docstring), so these reach the record.
                "n_samples": clocks.get("n_samples"),
                "energy_j": clocks.get("energy_j"),
                "duration_s": clocks.get("duration_s"),
                "sw_power_cap_fraction": clocks.get("sw_power_cap_fraction"),
                "power_limited": self.result.power_limited,
                "clocks_locked_requested_succeeded": False,
                "profiler_control": self.profiler.as_dict(),
                "device": self.device.as_dict(),
                # Computed by enforce() and previously discarded entirely --
                # every CUDA record here has sustained_s far under the 60s
                # floor and nothing said so. False does not mean "fine", it
                # means "not long enough to license a 1e9/1e11 projection".
                "burst": self.result.burst,
            },
            padding=({"table": self.padding_table} if self.padding_table
                    else {}),
        )

    @property
    def status(self) -> str:
        if self.result is None:
            return "not-graded"
        return self.result.status

    def write(self, path: "str | Path") -> str:
        """Append the graded record, and return its status.

        Mirrors ``CpuRun.write()``: a run that never completed the timed
        block has no record, and raising here rather than writing a partial
        one is what keeps "a check that runs and does not record" from
        happening the other way -- recording a run that did not finish.
        """
        if self.record is None:
            raise RuntimeError(
                "no record to write: the timed block did not complete (or "
                "raised), so nothing was graded. A partial run must not be "
                "recorded as if it were a whole one.")
        append_jsonl(self.record, path)
        return self.status

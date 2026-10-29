"""`bench/gpu_run.py`'s `GpuRun` -- the GPU analogue of `bench.cpu_run.CpuRun`.

No GPU or build required: every hardware-touching dependency (`GpuLock`,
`query_gpu_state`, `ClockPowerSampler`, `resolve_default_device`,
`detect_profiler_clock_control`, `verify_build`, `numerics.check_build`) is
replaced with a fake, so this runs in CI the same way `tests/test_bench_cpu.py`
and `tests/test_bench_rules.py` do -- by constructing the objects `rules.py`
consumes directly rather than asking real hardware for them.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from bench import gpu_run as gr
from bench import rules
from bench.gpu_lock import ComputeApp, DeviceIdentity, GpuState


def _fake_device() -> DeviceIdentity:
    return DeviceIdentity(visible_index=0, physical_index=0,
                          uuid="GPU-fake-0000-0000-0000-000000000000")


def _fake_gpu_state(device, foreign=False) -> GpuState:
    apps = ([ComputeApp(pid="1", process_name="someone_else", used_gpu_memory="100 MiB")]
           if foreign else [])
    return GpuState(device=device, compute_apps=apps, utilization_gpu_pct=0.0,
                    memory_used_mib=0.0, power_draw_w=15.0)


class _FakeLock:
    """Records acquire()/release() order without touching a real lock file."""

    events: list = []

    def __init__(self, ticket_id, device, blocking=False):
        self.ticket_id = ticket_id
        self.device = device
        self.was_held = False

    def acquire(self):
        _FakeLock.events.append("acquire")
        self.was_held = True

    def release(self):
        _FakeLock.events.append("release")


class _FakeSampler:
    def __init__(self, device, interval_s=0.5):
        pass

    def __enter__(self):
        _FakeLock.events.append("sampler_start")
        return self

    def __exit__(self, *exc):
        _FakeLock.events.append("sampler_stop")
        return False

    def summary(self):
        return {"n_samples": 7, "achieved_clock_mhz": {"median": 1800.0},
                "sw_power_cap_fraction": 0.0, "throttle_fraction": 0.0,
                "energy_j": 123.4, "duration_s": 5.0}


class _FakeProfiler:
    state = "none"
    tool = None

    def as_dict(self):
        return {"state": self.state, "tool": self.tool}


class _FakeVerifiedBuild:
    build_type = "Release"
    cxx_flags = "-O3"
    cuda_flags = ""
    with_cuda = True
    cuda_architectures_requested = "120-real"
    nvcc_version = "13.2.78"
    git_sha = "deadbeef"
    cuda_src_sha256 = "abc123"
    cuda_src_tree = "abc123"
    cuda_src_dirty = False
    cuda_src_dirty_files = []
    cuda_src_stamp_phase = "build"
    arch_list = ["sm_120"]
    ptx_present = False
    so_sha256 = {}
    arch_probe_error = None
    with_cpu = False
    cpu_build_type = "unknown"

    @property
    def is_release(self):
        return self.build_type == "Release"

    @property
    def cpu_is_release(self):
        return self.cpu_build_type == "Release"

    @property
    def arch_profile(self):
        return "native"

    def as_dict(self):
        return {"type": self.build_type}


class _FakeArithmeticCheck:
    ok = True
    violations = []
    advisories = []
    checked_a_gpu_build = True
    overridden = False


@pytest.fixture(autouse=True)
def _reset_events():
    _FakeLock.events = []
    yield


@pytest.fixture
def patched(monkeypatch):
    device = _fake_device()
    monkeypatch.setattr(gr, "resolve_default_device", lambda: device)
    monkeypatch.setattr(gr, "GpuLock", _FakeLock)
    monkeypatch.setattr(gr, "query_gpu_state",
                        lambda dev, exclude_pids=None: _fake_gpu_state(dev))
    monkeypatch.setattr(gr, "ClockPowerSampler", _FakeSampler)
    monkeypatch.setattr(gr, "detect_profiler_clock_control",
                        lambda: _FakeProfiler())
    monkeypatch.setattr(gr, "verify_build", lambda: _FakeVerifiedBuild())
    monkeypatch.setattr("roshambo3.numerics.check_build",
                        lambda: _FakeArithmeticCheck())
    monkeypatch.setattr("bench.gitinfo.get_git_info",
                        lambda: mock.Mock(sha="deadbeef", dirty=False,
                                         branch="main", source_dirty=False,
                                         as_dict=lambda: {"sha": "deadbeef"}))
    return device


class TestLockOrdering:
    def test_lock_is_acquired_before_the_body_and_released_after(self, patched):
        """The lock must precede whatever the body does that could
        create a CUDA context. This does not create a real context (no GPU
        needed to run this test), but it does assert the *order* GpuRun
        itself guarantees, which avoids the lock/context deadlock."""
        with gr.GpuRun("T99", decision_grade=True,
                      measurement_regime="headline") as run:
            _FakeLock.events.append("body")
            run.timing = {"conformers_per_s": 1.0}
            run.dataset = {"n_conformers": 10}
            run.config = {"max_results": 100}
        assert _FakeLock.events == [
            "acquire", "sampler_start", "body", "sampler_stop", "release"]


class TestGrading:
    def test_a_clean_rate_run_grades_and_records(self, patched, tmp_path):
        with gr.GpuRun("T99", decision_grade=True,
                      measurement_regime="headline") as run:
            run.timing = {"conformers_per_s": 5_000_000.0, "sustained_s": 61.0,
                         "n_repeats": 11, "n_warmup": 3}
            run.dataset = {"n_conformers": 10_000_000, "query_panel": {}}
            run.config = {"max_results": 100_000_000}
            run.max_results_applied_per_batch = False
        assert run.status in ("decision", "advisory")
        assert run.result is not None and not run.result.hard_refuse
        out = tmp_path / "runs.jsonl"
        status = run.write(str(out))
        assert status == run.status
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["config"]["backend"] == "cuda"
        assert rec["status"] == run.status

    def test_max_results_applied_per_batch_reaches_the_pitfall_12_gate(self, patched):
        """The amendment this whole ticket is about: the flag GpuRun exposes
        must actually be the one bench.rules.enforce() reads, not a
        same-named field that silently does not connect."""
        def build(flag):
            with gr.GpuRun("T99", decision_grade=False,
                          measurement_regime="headline") as run:
                # n_repeats/n_warmup clear check_repeats' own hard-refusal
                # gate (>=3 warmup, >=11 timed) so execution reaches the
                # advisory this test is about; n_conformers clears the
                # launch-saturation gate (SATURATION_MARGIN *
                # SATURATING_CONFORMERS_PER_LAUNCH = 32,768) for the same
                # reason.
                run.timing = {"conformers_per_s": 1.0, "n_repeats": 11,
                             "n_warmup": 3, "sustained_s": 61.0}
                run.dataset = {"n_conformers": 100_000}
                run.config = {"max_results": 100}  # << 10 * n_conformers
                run.max_results_applied_per_batch = flag
            return run.result

        unfixed = build(True)
        fixed = build(False)
        assert any("per-batch truncation" in r for r in unfixed.advisory_reasons)
        assert not any("per-batch truncation" in r for r in fixed.advisory_reasons)

    def test_padding_table_reaches_the_record(self, patched, tmp_path):
        """Review finding: `padding_table_from_corpus` existed and
        `run.padding_table` was never set by any caller, so the pitfall-8
        refusal in `check_padding_comparable` was inert on every record."""
        table = [[6, 40, 0, 25247]]
        with gr.GpuRun("T99", decision_grade=False,
                      measurement_regime="headline") as run:
            run.timing = {"conformers_per_s": 1.0}
            run.dataset = {"n_conformers": 25247}
            run.config = {"max_results": 100}
            run.padding_table = table
        assert run.record.padding.get("table") == table

    def test_device_stage_seconds_reaches_run_context(self, patched):
        """Review finding: collected via stage_timing=True and written to
        the ledger, but never forwarded to RunContext, so
        launch_overhead_fraction could never fire from a GpuRun."""
        with gr.GpuRun("T99", decision_grade=False,
                      measurement_regime="headline") as run:
            run.timing = {"conformers_per_s": 1.0}
            run.dataset = {"n_conformers": 1000}
            run.config = {"max_results": 100}
            run.device_stage_seconds = {"enabled": True, "alloc_s": 10.0,
                                        "precompute_s": 0.0, "free_s": 0.0}
            run.measured_total_s = 100.0
        frac = rules.launch_overhead_fraction(run.device_stage_seconds,
                                              run.measured_total_s)
        assert frac == pytest.approx(0.10)

    def test_operating_point_carries_sampler_and_burst_fields(self, patched):
        """Review finding: n_samples/energy_j/duration_s were sampled and
        dropped before reaching the record, and `burst` was computed by
        enforce() and discarded entirely."""
        with gr.GpuRun("T99", decision_grade=False,
                      measurement_regime="headline") as run:
            # n_repeats/n_warmup clear check_repeats' hard-refusal gate;
            # sustained_s=1.0 (far under the 60s floor) is what makes burst
            # True once that gate is cleared and burst is actually computed.
            run.timing = {"conformers_per_s": 1.0, "sustained_s": 1.0,
                         "n_repeats": 11, "n_warmup": 3}
            run.dataset = {"n_conformers": 100_000}  # clears the saturation gate
            run.config = {"max_results": 1_000_000}
            run.max_results_applied_per_batch = False
        op = run.record.operating_point
        assert op["n_samples"] == 7
        assert op["energy_j"] == 123.4
        assert op["duration_s"] == 5.0
        assert op["burst"] is True  # sustained_s=1.0 << 60s floor

    def test_a_partial_run_refuses_to_write(self, patched, tmp_path):
        run = gr.GpuRun("T99", decision_grade=False,
                        measurement_regime="headline")
        with pytest.raises(ValueError, match="boom"):
            with run:
                raise ValueError("boom")
        # __exit__ returned False (the body's own exception propagates,
        # unmasked) and, because exc_type was not None, _grade() never ran --
        # so there is no record, and write() must refuse rather than persist
        # a run that did not complete (CpuRun's own rule, mirrored here).
        assert run.record is None
        with pytest.raises(RuntimeError):
            run.write(str(tmp_path / "runs.jsonl"))

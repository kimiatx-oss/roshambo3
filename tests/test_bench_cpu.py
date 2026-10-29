"""The CPU half of the measurement provenance.

`bench/rules.py` is why any number here can be trusted and every rule in it was
about a GPU. These test the analogues, and in particular that adding them
cannot change how an existing GPU record grades.
"""
from __future__ import annotations

import os

import pytest

from bench import cpu as bcpu
from bench import rules
from tests.test_bench_rules import _ctx, _req  # noqa: F401  (shared builders)


class TestIdentityIsAskedNotAssumed:
    def test_it_reports_the_machine_it_is_on(self):
        i = bcpu.cpu_identity()
        assert i.host and i.logical_cpus > 0
        assert i.physical_cores > 0
        assert i.logical_cpus >= i.physical_cores

    def test_smt_is_derived_from_the_topology(self):
        i = bcpu.cpu_identity()
        assert i.smt_enabled == (i.threads_per_core > 1)

    def test_the_lscpu_maximum_is_recorded_but_not_used_as_a_clock(self):
        """One test host's lscpu says 7022.94 MHz, which no Zen 3 part reaches. It is
        carried so a reader can see it was ignored rather than overlooked --
        a clock nobody configured is not a measurement."""
        i = bcpu.cpu_identity()
        d = i.as_dict()
        assert "lscpu_max_mhz" in d
        assert "achieved" not in str(d.get("lscpu_max_mhz", ""))

    def test_physical_core_cpus_picks_one_sibling_per_core(self):
        i = bcpu.cpu_identity()
        got = bcpu.physical_core_cpus()
        if not got:
            pytest.skip("no topology exposed")
        assert len(got) == i.physical_cores
        assert len(set(got)) == len(got)

    def test_cache_parsing_handles_the_instances_suffix(self):
        assert bcpu._cache_bytes("768 KiB (24 instances)") == (786432, 24)
        assert bcpu._cache_bytes("128 MiB (4 instances)") == (134217728, 4)
        assert bcpu._cache_bytes("") == (0, 0)


class TestContention:
    def test_it_does_not_report_its_own_probe_as_foreign_load(self):
        """`ps` lands in its own output at a spuriously high pcpu -- a fresh
        process whose CPU time is divided by an almost-zero lifetime -- and a
        check that flagged its own probe would fail always."""
        c = bcpu.query_cpu_state()
        assert not any(comm.strip() == "ps"
                       for _, _, comm in c.foreign_processes)

    def test_busy_is_not_merely_nonzero_load(self):
        """A threshold that fires on a desktop and a shell is one people learn
        to pass with a flag."""
        c = bcpu.CpuContention(loadavg_1m=0.5, loadavg_5m=0.5,
                               logical_cpus=48, foreign_processes=[])
        assert not c.busy
        c = bcpu.CpuContention(loadavg_1m=40.0, loadavg_5m=40.0,
                               logical_cpus=48, foreign_processes=[])
        assert c.busy


class TestAffinityIsTheOneThingWeControl:
    def test_pin_returns_what_was_actually_set(self):
        """Not what was asked for: the kernel may narrow a request (a cpuset,
        an offline CPU), and recording the request would record a claim."""
        original = bcpu.current_affinity()
        if len(original) < 2:
            pytest.skip("need at least two CPUs")
        try:
            got = bcpu.pin_to(original[:2])
            assert got == sorted(original[:2])
            assert bcpu.current_affinity() == got
        finally:
            os.sched_setaffinity(0, set(original))


class TestTheCpuLockIsHostScoped:
    def test_the_path_carries_the_host(self, monkeypatch):
        from roshambo3.paths import cpu_lock_path
        monkeypatch.setenv("ROSHAMBO3_CPU_LOCK", "/tmp/x.lock")
        import socket
        assert str(cpu_lock_path()) == f"/tmp/x.lock.{socket.gethostname()}"

    def test_it_is_host_scoped_and_nothing_finer(self, monkeypatch):
        """Two runs pinned to disjoint cores still contend for last-level
        cache, memory bandwidth and the boost budget -- one host's four L3
        instances are six cores apiece. A per-core-set lock would arbitrate the
        wrong thing while reading as safety."""
        from roshambo3.paths import cpu_lock_path
        monkeypatch.setenv("ROSHAMBO3_CPU_LOCK", "/tmp/x.lock")
        assert cpu_lock_path() == cpu_lock_path()


class TestTheCpuRulesDoNotTouchAGpuRun:
    """Every CPU field defaults to the answer that keeps a GPU verdict what it
    was, and the rules are gated on `backend_name`."""

    #: Substrings only a CPU rule can produce. Asserted against rather than
    #: `status == "decision"`, because the shared `_ctx()` fixture is already
    #: advisory for two reasons that predate the CPU rules (no source content
    #: hash, no recorded launch structure). What matters is that the CPU rules add
    #: nothing to that list, not what the list happens to be.
    CPU_ONLY = ("dispatch target", "ROSHAMBO3_CPU_LOCK", "not pinned",
                "may not claim a clock", "another process was using this machine")

    def test_a_gpu_run_grades_exactly_as_it_did_before(self):
        r = rules.enforce(_req(), _ctx())
        assert not r.hard_refuse
        assert not any(s in a for a in r.advisory_reasons for s in self.CPU_ONLY)

    def test_the_cpu_rules_are_gated_on_the_backend_name(self):
        """The same context with the CPU fields left at their defaults must
        refuse under a CPU backend and not under CUDA -- otherwise the gate is
        not doing anything."""
        assert not rules.enforce(_req(), _ctx()).hard_refuse
        assert rules.enforce(_req(), _ctx(backend_name="cpu")).hard_refuse


class TestTheCpuRules:
    def _cpu(self, **kw):
        # Two obligations of "compliant" are refusals:
        # `checked_a_cpu_build` (the twin of checked_a_gpu_build -- without it,
        # scoping the build-inspection refusal to GPU runs would have left a hole where a CPU
        # run inspected nothing) and `cpu_vector_width` (a record that does not name its width cannot be compared with
        # anything). Both are derived by RunContext.from_build for real
        # callers; this fixture builds a context directly, so it states them.
        base = dict(backend_name="cpu", cpu_isa_selected="AVX2",
                    checked_a_cpu_build=True, cpu_vector_width=8,
                    cpu_lock_held=True, cpu_affinity=[0, 1],
                    achieved_cpu_mhz={"n_samples": 30})
        base.update(kw)
        return _ctx(**base)

    def test_an_unknown_dispatch_target_is_refused(self):
        """Pitfall 1's analogue: a build that silently ran a baseline scalar
        path looks like a slow kernel, not a broken build."""
        r = rules.enforce(_req(), self._cpu(cpu_isa_selected=None))
        assert r.hard_refuse and "dispatch target" in r.refusal_reason

    def test_a_decision_grade_cpu_timing_needs_the_cpu_lock(self):
        r = rules.enforce(_req(), self._cpu(cpu_lock_held=False))
        assert r.hard_refuse and "ROSHAMBO3_CPU_LOCK" in r.refusal_reason

    def test_an_advisory_cpu_run_may_go_unlocked(self):
        r = rules.enforce(_req(decision_grade=False),
                          self._cpu(cpu_lock_held=False))
        assert not r.hard_refuse

    def test_an_unpinned_run_is_advisory_not_refused(self):
        r = rules.enforce(_req(), self._cpu(cpu_affinity=None))
        assert not r.hard_refuse
        assert any("not pinned" in a for a in r.advisory_reasons)

    def test_a_run_that_never_sampled_may_not_claim_a_clock(self):
        r = rules.enforce(_req(), self._cpu(achieved_cpu_mhz=None))
        assert any("may not claim a clock" in a for a in r.advisory_reasons)
        r = rules.enforce(_req(), self._cpu(achieved_cpu_mhz={"n_samples": 0}))
        assert any("may not claim a clock" in a for a in r.advisory_reasons)

    def test_foreign_load_is_advisory(self):
        r = rules.enforce(_req(), self._cpu(
            cpu_busy=True, cpu_foreign_processes=[[1, 99.0, "python3"]]))
        assert not r.hard_refuse
        assert any("another process" in a for a in r.advisory_reasons)

    def test_a_fully_compliant_cpu_run_draws_no_cpu_advisory(self):
        """Not `status == decision`: the shared fixture carries two advisories
        that predate the CPU rules. An advisory that fires on a compliant run is one
        people learn to ignore, which is what this actually guards."""
        r = rules.enforce(_req(), self._cpu())
        assert not r.hard_refuse
        assert not any(s in a for a in r.advisory_reasons for s in
                       TestTheCpuRulesDoNotTouchAGpuRun.CPU_ONLY)


class TestTheFrequencySamplerReportsRatherThanRequests:
    def test_an_unsampled_summary_says_so_rather_than_returning_empty(self):
        s = bcpu.CpuFrequencySampler()
        got = s.summary()
        assert got["n_samples"] == 0
        assert "may not claim a clock" in got["note"]

    def test_it_samples_a_real_distribution(self):
        with bcpu.CpuFrequencySampler(interval_s=0.2) as s:
            x = 0
            for i in range(2_000_000):
                x += i
        got = s.summary()
        if got["n_samples"] == 0:
            assert "may not claim a clock" in got["note"]
            pytest.skip("host exposes no readable CPU frequency samples")
        assert got["n_samples"] > 0
        assert got["achieved_mhz"]["min"] > 0
        assert got["achieved_mhz"]["max"] >= got["achieved_mhz"]["median"]

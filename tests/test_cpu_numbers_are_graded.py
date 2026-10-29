"""A CPU rate in a results directory must carry a graded run record.

A backstop. :class:`bench.cpu_run.CpuRun` makes the graded path the short
one; this catches the script that takes the other path anyway. A mechanism
which would pass a forgetful script has not been tested, so this file also asserts the check *fires*.

Why a test rather than a note: once the CPU rules existed, 75 numbers were
still quoted without them, because "call enforce()" was something to
remember. Remembering is what failed.

Whole-history certification is a developer audit, not a release correctness
prerequisite: run scripts/audit_cpu_result_ledgers.py. These tests exercise the
checker on synthetic inputs, including missing-record and marker-key failures.
"""
from __future__ import annotations

import json

from bench.cpu_run import (looks_like_a_cpu_timed_artifact,
                           ungraded_timed_artifacts)

def test_the_check_fires_on_a_forgetful_script(tmp_path):
    """The half that matters: a directory that forgot must be caught."""
    (tmp_path / "throughput.json").write_text(json.dumps({
        "stamp": {"selected_target": "X86_64_V3", "cxx_flags": " -O3"},
        "rows": [{"start_pairs_per_s": 70171.0, "vector_width": 8}],
    }))
    assert ungraded_timed_artifacts(tmp_path) == ["throughput.json"]

    (tmp_path / "runs.jsonl").write_text("{}\n")
    assert ungraded_timed_artifacts(tmp_path) == []


def test_explicit_developer_audit_keeps_missing_ledger_detection(tmp_path):
    from scripts.audit_cpu_result_ledgers import audit
    ticket = tmp_path / "T999"
    ticket.mkdir()
    (ticket / "rate.json").write_text(json.dumps({
        "stamp": {"selected_target": "X86_64_V3"},
        "rows": [{"start_pairs_per_s": 1.}]}))
    assert audit(tmp_path) == {"T999": ["rate.json"]}
    (ticket / "runs.jsonl").write_text("{}\n")
    assert audit(tmp_path) == {}


def test_correctness_artifacts_are_not_swept_in(tmp_path):
    """An untimed artifact needs no lock and no grade.

    "Correctness work (unit tests, untimed kernels) may run
    unlocked." Demanding a graded record for an oracle-agreement file would
    invent an obligation, which is the mirror of the defect this check repairs.
    """
    (tmp_path / "oracle_agreement.json").write_text(json.dumps({
        "stamp": {"selected_target": "X86_64_V3"},
        "volumes_vs_float64_oracle": {"worst_rel_err": 7.6e-7},
    }))
    assert ungraded_timed_artifacts(tmp_path) == []


def test_the_gpu_ledger_is_not_swept_in(tmp_path):
    """A GPU timed artifact has no CPU stamp and is graded through its own
    path; sweeping it in would fail the whole historical ledger for a rule
    written about the CPU."""
    (tmp_path / "gpu_rate.json").write_text(json.dumps({
        "timing": {"pairs_per_s": 69.8e6}, "gpu": {"name": "Blackwell"},
    }))
    assert not looks_like_a_cpu_timed_artifact(
        json.loads((tmp_path / "gpu_rate.json").read_text()))
    assert ungraded_timed_artifacts(tmp_path) == []


def test_the_stampless_hole_is_real_and_stated(tmp_path):
    """The backstop's known limitation, asserted so it cannot be forgotten.

    An artifact that records a rate and *nothing about the machine* evades the
    check, because there is no key to recognize it by. Four of the ladder's 15
    timed artifacts are like this. Inverting the gate to "not GPU-marked" was
    measured and fails 23 historical GPU directories, so the hole stays and is
    documented in `_CPU_MARKER_KEYS`.

    This test asserts the LIMITATION rather than the capability. If someone
    later closes the hole, this test fails and they update it deliberately --
    which is the point: a limitation nobody wrote down is one that gets
    rediscovered as a bug.
    """
    (tmp_path / "anonymous_rate.json").write_text(json.dumps({
        "argv": ["tickets/T999/arm.py"],
        "rows": [{"start_pairs_per_s": 70171.0}],
    }))
    assert ungraded_timed_artifacts(tmp_path) == [], (
        "the stampless hole has been closed -- good. Update this test and "
        "_CPU_MARKER_KEYS' docstring, which both say it is open.")


def test_a_stamped_forgetful_artifact_is_caught_by_each_marker_key(tmp_path):
    """Every marker key must actually work, not just the one in the fire-test.

    A set of keys where only the first was ever exercised is a set where the
    rest are decoration.
    """
    from bench.cpu_run import _CPU_MARKER_KEYS
    for key in sorted(_CPU_MARKER_KEYS):
        d = tmp_path / key
        d.mkdir()
        (d / "rate.json").write_text(json.dumps({
            key: "whatever", "rows": [{"start_pairs_per_s": 1.0}]}))
        assert ungraded_timed_artifacts(d) == ["rate.json"], (
            f"marker key {key!r} does not catch a forgetful artifact")

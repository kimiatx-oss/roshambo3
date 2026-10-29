"""Untimed ETA correctness with simulated clocks; no CUDA or rate benchmark."""
import math
import pytest
from roshambo3.web.progress import SubmissionProgress


def test_screen_clock_excludes_preparation_and_queue_wait():
    # Job creation might have been hours ago; only the screening origin counts.
    tracker = SubmissionProgress(1000, 10_000)
    assert tracker.update(100, 10_001)["eta_seconds"] is None
    update = tracker.update(200, 10_002)
    assert update["screen_elapsed_s"] == 2
    assert update["submission_rate"] == 100
    assert update["eta_seconds"] == 8


def test_small_fast_screens_never_report_a_warmup_eta():
    tracker = SubmissionProgress(1000, 0)
    assert tracker.update(10, 0.1) is None
    assert tracker.update(500, 0.5)["eta_seconds"] is None
    final = tracker.update(1000, 1)
    assert final["eta_seconds"] is None and final["progress"] == 1


def test_rate_smoothing_uses_actual_sample_duration_and_recent_work():
    tracker = SubmissionProgress(10_000, 0)
    tracker.update(100, 1)
    before = tracker.update(200, 2)
    faster = tracker.update(500, 3)
    rate = 100 + (1 - math.exp(-1 / 5)) * (300 - 100)
    assert faster["submission_rate"] == pytest.approx(rate)
    assert faster["eta_seconds"] == pytest.approx(9500 / rate)
    assert 100 < faster["submission_rate"] < 300
    # A long loaded-machine pause is part of the next observed interval,
    # reducing rate without synchronizing the device or taking another sample.
    slower = tracker.update(600, 13)
    assert slower["submission_rate"] < before["submission_rate"]
    assert slower["eta_seconds"] > faster["eta_seconds"]


def test_throttle_does_not_lose_work_between_publications():
    tracker = SubmissionProgress(1000, 0)
    for i in range(1, 50):
        assert tracker.update(i, i / 100) is None
    update = tracker.update(50, 0.5)
    assert update["submitted_conformers"] == 50
    assert update["submission_rate"] == 100
    assert tracker.update(90, 0.9) is None
    assert tracker.update(100, 1)["submission_rate"] == 100


@pytest.mark.parametrize("total", [0, 1000])
def test_empty_and_completed_work_have_no_eta(total):
    tracker = SubmissionProgress(total, 0)
    tracker.update(total // 2, 1)
    update = tracker.update(total, 2)
    assert update["eta_seconds"] is None
    assert math.isfinite(update["submission_rate"])


@pytest.mark.parametrize("state", ["completed", "failed", "canceled"])
def test_terminal_job_clears_remaining_work_estimate(tmp_path, state):
    from roshambo3.web.store import Store
    store = Store(tmp_path)
    ident = store.create({})["id"]
    store.update(ident, "screening", eta_seconds=100)
    store.update(ident, state)
    assert store.get(ident)["eta_seconds"] is None

"""Targeted conversion-clamp ablation; no approximation change."""
import numpy as np
import pytest
from test_cpu_shared_pose_contract import _inputs, _prepare, _score


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("offset", [0., 100.])
@pytest.mark.parametrize("factor", [False, True])
def test_unclamped_screen_rows_are_identical(canonical, offset, factor):
    data = _inputs(canonical=canonical)
    # Distant pairs exercise the cutoff and int32 overflow without introducing
    # invalid input geometry. FP exception masks stay at the caller's setting.
    data[3][:, :, 0] += offset
    prepared = _prepare(data)
    common = dict(grain=16, factor_torque=factor, canonical_color_kernel=True)
    guarded, _ = _score(data, prepared, optimizer_exp_mode="rawbits", **common)
    unclamped, _ = _score(data, prepared, optimizer_exp_mode="rawbits_unclamped", **common)
    np.testing.assert_array_equal(unclamped, guarded)
    assert np.isfinite(unclamped).all()


def test_internal_native_threads_fall_back_to_guarded_exp(monkeypatch):
    from roshambo3.backend import cpu
    data = _inputs(canonical=True)
    data[3][:, :, 0] += 100.
    prepared = _prepare(data)
    original = cpu.score_pairs

    def threaded(**kwargs):
        kwargs["threads"] = 2
        # Parallel calls use worker timers, not the serial-only census path.
        kwargs["with_census"] = False
        kwargs["with_worker_timers"] = True
        return original(**kwargs)

    monkeypatch.setattr(cpu, "score_pairs", threaded)
    common = dict(grain=16, factor_torque=True, canonical_color_kernel=True)
    guarded, _ = _score(data, prepared, optimizer_exp_mode="rawbits", **common)
    fallback, stats = _score(data, prepared,
        optimizer_exp_mode="rawbits_unclamped", **common)
    assert stats["optimizer_exp_mode_effective"] == "rawbits"
    np.testing.assert_array_equal(fallback, guarded)

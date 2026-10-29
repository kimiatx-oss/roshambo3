"""Real raw-pose boundaries and complete-call exceptional result policy."""
import numpy as np
import pytest
from roshambo3.backend import cpu
from test_cpu_color_start_lanes import _population
from test_cpu_winner_only_final import run

@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_raw_pose_boundary_rejects_exceptional_bytes(value):
    kw = _population(1, 'canonical')
    _, _, handle = run(kw)
    pose = np.zeros(16, dtype=np.float32)
    pose[[0, 4, 8]] = 1
    pose[12] = 1
    for index in [0, 9, 12]:
        invalid = pose.copy()
        invalid[index] = value
        with pytest.raises(ValueError, match='nonfinite fixed-pose replay pose'):
            cpu.extension()._replay_fixed_pose_cross(handle, 0,
                kw['library_coords'][0], kw['library_counts'][0], invalid)

def test_raw_finite_nonunit_matrix_is_consumed_as_supplied():
    kw = _population(1, 'canonical')
    _, _, handle = run(kw)
    pose = np.zeros(16, dtype=np.float32)
    pose[[0, 4, 8]] = 1
    pose[12] = 3  # Reporting quaternion cannot rebuild the supplied matrix.
    evaluate = lambda p: cpu.extension()._replay_fixed_pose_cross(
        handle, 0, kw['library_coords'][0], kw['library_counts'][0], p)
    first = evaluate(pose)
    pose[12] = 7
    np.testing.assert_array_equal(first, evaluate(pose))
    pose[[0, 4, 8]] = 2
    second = evaluate(pose)
    assert np.isfinite(second).all() and not np.array_equal(first, second)

@pytest.mark.parametrize('policy', ['winner', 'all_start'])
@pytest.mark.parametrize('value', [np.nan, np.inf, np.finfo(np.float32).max])
def test_extreme_input_never_returns_nonfinite_scores(policy, value):
    kw = _population(1, 'canonical')
    kw['library_coords'][0, 0, :3] = value
    try:
        scores, _, _ = run(kw, final_start_policy=policy)
    except (ValueError, RuntimeError):
        return  # Checked construction/final work-unit error reaches caller.
    assert np.isfinite(scores).all()

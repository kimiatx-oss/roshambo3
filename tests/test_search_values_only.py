"""Exact SEARCH arithmetic survives omission of discarded derivatives."""
import numpy as np
import pytest
from roshambo3.backend import cpu
from test_cpu_color_start_lanes import _population
from test_cpu_winner_only_final import run

@pytest.mark.parametrize('mode', ['baseline', 'rawbits', 'rawbits_unclamped'])
@pytest.mark.parametrize('factor', [False, True])
def test_search_values_preserve_saved_and_arbitrary_pose_objectives(mode, factor, monkeypatch):
    monkeypatch.setenv('ROSHAMBO3_FINAL_SELECTION_TRACE', '1')
    kw = _population(3, 'canonical')
    _, stats, handle = run(kw, exp_mode=mode, factor=factor)
    rng = np.random.default_rng(9137)
    for row in range(3):
        poses = np.ascontiguousarray(stats['converged_pose_bytes'][row])
        for arbitrary in [False, True]:
            if arbitrary:
                poses = poses.copy()
                poses[:, :12] = rng.normal(size=(4, 12)).astype(np.float32)
            args = (handle, int(kw['pair_query_row'][row]),
                kw['library_coords'][kw['pair_library_row'][row]],
                kw['library_counts'][kw['pair_library_row'][row]], poses,
                np.ascontiguousarray(stats['converged_objective_self'][row]), .5, mode, factor, True)
            full = cpu.extension()._replay_fixed_pose_search(*args)
            values = cpu.extension()._replay_fixed_pose_search(*args, True)
            assert full.tobytes() == values.tobytes()

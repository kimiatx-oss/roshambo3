"""Optimizer evidence integrity tests; native certification is explicitly opt-in."""
import os
from pathlib import Path

import numpy as np
from roshambo3.numerics import WIRE_QUANT_STEP
import pytest

from bench.numerics_optimizer import (METRICS, exceeding, geometry, load_inputs,
                                      permuted, rescore, run, statistics, thresholds)


def arrays():
    return dict(query_coords=np.array([[[0, 0, 0, 0], [1, 0, 0, 0]]], np.float32),
        query_n_atoms=np.array([2], np.int32),
        library_coords=np.array([[[0, 0, 0], [int(1 / WIRE_QUANT_STEP), 0, 0]]], np.int16),
        library_counts=np.array([[2]], np.uint8),
        pair_query_row=np.zeros(4, np.int32), pair_library_row=np.zeros(4, np.int32))


def rows():
    out = np.zeros((4, 20), np.float32)
    out[:, 9] = 1
    out[:, 16] = 1
    return out


def test_rescore_uses_own_pose_and_ignores_reported_score():
    a = arrays()
    good, translated = rows(), rows()
    good[:, 1] = -999
    translated[:, 1] = 999
    translated[:, 13] = 3
    assert np.allclose(rescore(a, good)[0], 1)
    result = statistics(a, good, translated, np.ones(4, bool))
    assert result['overall']['mean_loss'] > .8
    assert result['overall']['basin_change'] == 1


def test_source_order_control_preserves_physical_cloud():
    a = arrays()
    b = permuted(a, 12)
    for before, after in zip(geometry(a, 0), geometry(b, 0)):
        assert sorted(map(tuple, before)) == sorted(map(tuple, after))
    assert np.array_equal(a['library_coords'][0, :, 0], [0, int(1 / WIRE_QUANT_STEP)])


def test_start_disagreement_is_distinct_from_pose_basin():
    a, b = rows(), rows()
    result = statistics(arrays(), a, b, np.ones(4, bool),
                        anchor_starts=[0]*4, candidate_starts=[2]*4)
    assert result['overall']['winning_start_disagreement'] == 1
    assert result['overall']['basin_change'] == 0
    assert result['by_stratum']['T<0.4']['n'] == 0
    assert result['by_stratum']['T<0.4']['mean_loss'] is None


def test_limits_are_frozen_from_calibration_only():
    calibration = [dict(overall=dict(n=4, **dict(zip(METRICS, [.01, .25, .5]))))]
    limits = thresholds(calibration)
    candidate = dict(overall=dict(zip(METRICS, [.011, .25, .5])))
    assert exceeding(candidate, limits) == ['mean_loss']
    candidate['overall']['mean_loss'] = -.4
    assert exceeding(candidate, limits) == []
    assert limits['mean_loss'] == .01
    with pytest.raises(ValueError):
        thresholds([])


def test_input_loader_does_not_need_expected_scores(tmp_path):
    a = arrays()
    path = tmp_path / 'inputs.npz'
    np.savez(path, **a)
    assert np.array_equal(load_inputs(path)['library_coords'], a['library_coords'])


# Historical opt-in pilot is not a default coverage observation. The current
# live optimizer tests own that cell; skipping this experiment must not erase it.
def test_live_optimizer_qualification(numerics_coverage):
    if os.environ.get('ROSHAMBO3_RUN_OPTIMIZER_QUALIFICATION') != '1':
        pytest.skip('live optimizer qualification requires explicit panel execution; remains not run')
    evidence = run(load_inputs(Path(__file__).parent / 'data/anchor_fixture.npz'))
    arm = evidence['arms']['cpu_analytic']
    numerics_coverage.stamp(build=evidence['scalar_stamp'],
        arithmetic={'controls': 'atom permutations; transcendental controls missing'}, evidence=evidence)
    assert evidence['heldout_valid'], 'independent held-out arithmetic controls exceeded frozen thresholds'
    assert arm['plant_exceeding'], 'required lr_q*1.1 plant survived'
    assert evidence['qualified'], evidence['gaps']


def test_skipped_historical_pilot_does_not_overwrite_live_optimizer_coverage(tmp_path):
    """Exercise actual collection/skip/report behavior, including the old pilot."""
    import json
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    probe = tmp_path / 'test_coverage_probe.py'
    probe.write_text('''import pytest
from test_numerics_optimizer import test_live_optimizer_qualification

@pytest.mark.numerics_coverage(backend="cpu", path="vector", check="optimizer", channel="shape")
def test_executing_check(numerics_coverage):
    numerics_coverage.stamp(build={"probe": True}, arithmetic={"probe": True})
    assert 2 + 2 == 4
''')
    environment = dict(os.environ)
    environment.pop('ROSHAMBO3_RUN_OPTIMIZER_QUALIFICATION', None)
    environment['PYTHONPATH'] = os.pathsep.join([str(root / 'tests'), str(root),
        str(root / 'src'), *sys.path, environment.get('PYTHONPATH', '')])
    report = tmp_path / 'report'
    subprocess.run([sys.executable, '-S', '-m', 'pytest', '-q', '-p',
        'bench.numerics_pytest', '--numerics-report-dir', str(report), str(probe)],
        env=environment, cwd=root, check=True, capture_output=True, text=True)
    rows = json.loads((report / 'coverage.json').read_text())['rows']
    from bench.numerics_report import hardware_for
    cell = next(row for row in rows if all(row[key] == value for key, value in
        dict(backend='cpu', hardware=hardware_for('cpu'), path='vector',
             check='optimizer', channel='shape').items()))
    assert cell['status'] == 'passed', cell
    assert any('test_executing_check' in reason for reason in cell['reasons'])
    assert not any('test_live_optimizer_qualification' in reason for reason in cell['reasons'])

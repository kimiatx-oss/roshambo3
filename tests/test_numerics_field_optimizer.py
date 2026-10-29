"""The FIELD order control must reject geometry/typing changes."""
import numpy as np
import pytest

from bench.numerics_field_optimizer import assert_same_multisets, CALIBRATION, HELDOUT


def inputs():
    coords = np.array([[[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1], [0, 0, 0, 0]]], np.float32)
    return dict(query_coords=coords.copy(), query_n_atoms=np.array([1], np.int32),
                query_types=np.array([[0, 1, 2, 0]], np.int32),
                library_coords=coords.copy(), library_counts=np.array([[1, 1, 1]], np.uint8))


def test_order_control_accepts_only_unchanged_typed_multisets():
    original = inputs()
    reordered = {k: v.copy() for k, v in original.items()}
    # Explicit query feature typing permits a joint coordinate/type reorder.
    for key in ('query_coords', 'query_types'):
        reordered[key][:, 1:3] = reordered[key][:, 1:3][:, ::-1]
    assert_same_multisets(original, reordered)
    reordered['query_types'][0, 1] = 1
    with pytest.raises(AssertionError):
        assert_same_multisets(original, reordered)


def test_order_control_detects_coordinate_or_count_change():
    original = inputs()
    for key in ('query_coords', 'library_counts'):
        changed = {k: v.copy() for k, v in original.items()}
        changed[key].flat[0] += 1
        with pytest.raises(AssertionError):
            assert_same_multisets(original, changed)


def test_calibration_and_heldout_seeds_are_disjoint():
    assert len(CALIBRATION) == 32
    assert len(HELDOUT) == 16
    assert not set(CALIBRATION) & set(HELDOUT)


@pytest.mark.parametrize('occupied', ['scratch', 'output', 'shape', 'combo'])
def test_orphan_or_complete_artifacts_refused_without_modification(tmp_path, occupied):
    from bench.numerics_field_optimizer import require_fresh_paths
    scratch, output = tmp_path / 'poses.npz', tmp_path / 'report.json'
    paths = dict(scratch=scratch, output=output,
                 shape=tmp_path / 'report-shape-frozen.json',
                 combo=tmp_path / 'report-combo-frozen.json')
    require_fresh_paths(scratch, output)
    paths[occupied].write_bytes(b'irreplaceable frozen evidence')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        require_fresh_paths(scratch, output)
    assert paths[occupied].read_bytes() == b'irreplaceable frozen evidence'
    assert len(list(tmp_path.iterdir())) == 1


def test_non_npz_scratch_is_refused_before_observation(tmp_path):
    from bench.numerics_field_optimizer import require_fresh_paths
    with pytest.raises(ValueError, match='scratch must end'):
        require_fresh_paths(tmp_path / 'poses', tmp_path / 'report.json')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('worker', [False, True])
def test_cli_collision_refuses_before_native_snapshot(tmp_path, monkeypatch, worker):
    import sys
    from bench import numerics_field_optimizer as runner
    orphan = tmp_path / 'report-combo-frozen.json'
    orphan.write_bytes(b'frozen')
    argv = ['field_optimizer', '--build', str(tmp_path / 'build'),
            '--scratch', str(tmp_path / 'poses.npz'), '--out', str(tmp_path / 'report.json')]
    if worker:
        argv.append('--worker')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(runner, 'snapshot', lambda _: pytest.fail('native snapshot reached'))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert orphan.read_bytes() == b'frozen'


@pytest.mark.parametrize('prefix,indices', [
    ('library', (1, 2)),  # distinct implicit feature types
    ('library', (0, 1)),  # shape versus color
    ('library', (0, 3)),  # shape versus padding
    ('query', (0, 1)),  # shape versus color
    ('query', (0, 3)),  # both type zero, different prefix occupancy
])
def test_segment_crossing_rejected_even_when_global_multiset_is_identical(prefix, indices):
    original = inputs()
    changed = {k: v.copy() for k, v in original.items()}
    key = prefix + '_coords'
    changed[key][0, list(indices)] = changed[key][0, list(indices[::-1])]
    if prefix == 'query':
        changed['query_types'][0, list(indices)] = changed['query_types'][0, list(indices[::-1])]
    with pytest.raises(AssertionError):
        assert_same_multisets(original, changed)


def test_all_preregistered_seed_sets_are_disjoint():
    from bench.numerics_field_optimizer import SAME_GEOMETRY_CALIBRATION, SAME_GEOMETRY_HELDOUT
    groups = [CALIBRATION, HELDOUT, SAME_GEOMETRY_CALIBRATION, SAME_GEOMETRY_HELDOUT]
    assert [len(group) for group in groups] == [32, 16, 32, 16]
    assert len(set().union(*groups)) == sum(map(len, groups))


def test_executed_control_seeds_preserve_all_native_segments():
    # Revalidate the exact deterministic inputs used by both retained native
    # experiments with the stronger independent guard, without a GPU rerun.
    from bench.numerics_field_optimizer import SAME_GEOMETRY_CALIBRATION, SAME_GEOMETRY_HELDOUT
    from bench.numerics_optimizer_real import panel, pack
    cases, _ = panel()
    for objective in ('shape', 'combo'):
        original = pack(cases, objective)
        for seeds in (CALIBRATION, HELDOUT, SAME_GEOMETRY_CALIBRATION, SAME_GEOMETRY_HELDOUT):
            for seed in seeds:
                assert_same_multisets(original, pack(cases, objective, seed))


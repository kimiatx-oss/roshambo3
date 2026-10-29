"""Distinct-worker gates, stored f64 consumption and abortable failure paths."""
from types import SimpleNamespace

import numpy as np
import pytest

from bench.format3_cpu_binding_control import (check_consumed_values,
    reused_geometry_inputs, run_binding_pass)


def test_synthetic_prototype_geometry_is_reused_under_explicit_unique_names():
    molecules = reused_geometry_inputs(n_molecules=8, conformers=2)
    assert len({mol.GetProp('_Name') for mol in molecules}) == 8
    for i in range(4):
        assert molecules[i].GetNumConformers() == 2
        for j in range(2):
            np.testing.assert_array_equal(molecules[i].GetConformer(j).GetPositions(),
                                          molecules[i+4].GetConformer(j).GetPositions())
    assert len({mol.GetNumHeavyAtoms() for mol in molecules}) == 2


def test_stored_color_self_check_catches_f64_change_hidden_by_f32():
    from roshambo3.numerics import SHAPE_PREFACTOR
    shape = np.array([10.], np.float32)
    color = np.array([1.], np.float64)
    changed = np.nextafter(color, np.inf)
    np.testing.assert_array_equal(changed.astype(np.float32), color.astype(np.float32))
    record = dict(_expected_shape=shape,
        _consumed_shape=(shape.astype(np.float64)/float(SHAPE_PREFACTOR)).astype(np.float32),
        _expected_color=color, _consumed_color=changed)
    with pytest.raises(AssertionError):
        check_consumed_values(record, mode='combo')


def _inputs(tmp_path, mode):
    from bench.format3_cpu_matrix import _query
    from roshambo3.engine import R3gSource
    from tests.test_self_overlap_routes import build_inputs
    corpus, _, generator = build_inputs(tmp_path / 'corpus')
    source = R3gSource(corpus, color=mode == 'combo', color_generator=generator, tile=False)
    try:
        query = _query(source)
    finally:
        source.close()
    return corpus, query


@pytest.mark.parametrize('mode', ['shape', 'combo'])
def test_component_owns_distinct_blocks_on_distinct_native_workers(tmp_path, mode):
    corpus, query = _inputs(tmp_path, mode)
    record = run_binding_pass(corpus, query, mode, 2, timeout_s=5.)
    participants = record['first_claim_participants']
    assert len({p['worker_id'] for p in participants}) == 2
    assert len({p['thread_id'] for p in participants}) == 2
    assert len({tuple(p['descriptor']) for p in participants}) == 2
    assert 1 <= record['post_release_binding_invocation_peak'] <= 2
    assert record['stored_vaa_verified_blocks'] == record['blocks']
    assert record['consumed_shape_bits_equal']
    assert record['consumed_color_f64_bits_equal'] == (mode == 'combo')
    assert record['retained_self_handle_bytes'] == 0
    assert all(not any(k.endswith('_s') for k in block) for block in record['block_records'])
    assert 'binding_submission_wall_s' not in record
    assert 'last_worker_elapsed_seconds' not in record['executor_stats']


def test_component_aborts_other_waiting_workers_and_preserves_root_failure(tmp_path, monkeypatch):
    from roshambo3.engine import R3gSource
    corpus, query = _inputs(tmp_path, 'combo')
    original = R3gSource.decode_cpu_block
    def broken(source, descriptor):
        if descriptor.block_index == 0:
            raise RuntimeError('injected component decoder failure')
        return original(source, descriptor)
    monkeypatch.setattr(R3gSource, 'decode_cpu_block', broken)
    with pytest.raises(RuntimeError, match='injected component decoder failure'):
        run_binding_pass(corpus, query, 'combo', 2, timeout_s=.5)


def test_component_refuses_more_participants_than_distinct_blocks(tmp_path):
    corpus, query = _inputs(tmp_path, 'shape')
    blocks = len(corpus.block_directory('file'))
    with pytest.raises(ValueError, match='not enough distinct physical blocks'):
        run_binding_pass(corpus, query, 'shape', blocks+1, timeout_s=.5)


def test_qualification_cli_finalizes_summary_and_footer(tmp_path, monkeypatch):
    import json
    import os
    import sys
    if not hasattr(os, 'sched_getaffinity'):
        pytest.skip('campaign harness uses Linux CPU affinity')
    from rdkit import Chem
    import bench.gitinfo
    from bench.gitinfo import GitInfo
    from bench.format3_cpu_binding_control import main
    from tests.test_self_overlap_routes import build_inputs
    corpus, base, _ = build_inputs(tmp_path / 'corpus')
    query_path = tmp_path / 'query.sdf'
    with Chem.SDWriter(str(query_path)) as writer:
        writer.write(base)
    prefix = tmp_path / 'qualification'
    monkeypatch.setattr(bench.gitinfo, 'get_git_info',
        lambda: GitInfo('1'*40, False, 'synthetic-test', source_dirty=False))
    monkeypatch.setattr(sys, 'argv', ['binding-control', 'qualify',
        '--corpus', '4='+corpus.root, '--queries', str(query_path),
        '--workers', '2', '--modes', 'shape', '--output-prefix', str(prefix)])
    main()
    summary = json.loads((tmp_path / 'qualification-summary.json').read_text())
    records = [json.loads(line) for line in (tmp_path / 'qualification-passes.jsonl').read_text().splitlines()]
    assert summary['status'] == 'complete' and summary['passes'] == 1
    assert records[-1]['kind'] == 'footer'
    assert records[-1]['campaign_kind'] == 'qualify' and records[-1]['status'] == 'complete'

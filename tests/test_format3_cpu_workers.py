"""Format-3 CPU physical blocks, independent of pending builder emission."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.backend.spec import ColorPlacement
from roshambo3.engine import Engine, ScreenConfig
from roshambo3.engine.query import from_arrays
from roshambo3.engine.r3g_cpu import R3gWorkerSelfCache
from roshambo3.format.r3g.errors import CorpusValidationError
from roshambo3.format.r3g.reader import open_corpus, verify
from tests.format3_cpu_fixtures import PhysicalCPUFixtureSource, make_cpu_corpus


def _queries(source):
    descriptor = min(source.cpu_block_descriptors(), key=lambda d: d.gid_base)
    decoded = source.decode_cpu_block(descriptor)
    counts = decoded.result.counts[0]
    nr = int(counts[0])
    coords = decoded.result.coords[0]
    kwargs = {}
    if source.color:
        types = np.repeat(np.arange(1, counts.size), counts[1:])
        kwargs.update(color_generator=source.color_generator,
                      color=[(coords[nr:nr + len(types)], types)])
    return from_arrays([coords[:nr]], ['cpu-query'], **kwargs)


def _config(color):
    return ScreenConfig(color=color, color_placement=ColorPlacement.IN_GRADIENT,
                        mixing=.5 if color else 0., steps=25, start_mode=1)


@pytest.mark.parametrize('color', [False, True])
def test_native_blocks_match_conformance_arrays(color, tmp_path):
    path = Path(__file__).parent / 'data' / 'format3' / ('color' if color else 'shape')
    cache = str(tmp_path / 'cache')
    verify(str(path), cache_dir=cache)
    source = PhysicalCPUFixtureSource(open_corpus(str(path), cache_dir=cache), color=color)
    try:
        descriptors = source.cpu_block_descriptors()
        costs = [d.n_conf * d.library_width for d in descriptors]
        assert costs == sorted(costs, reverse=True)
        for d in descriptors:
            block = source.decode_cpu_block(d)
            group = source.corpus.group(d.group_index)
            heavy = group.decode_block(d.block_index)
            for row, nr in enumerate(block.result.counts[:, 0]):
                np.testing.assert_array_equal(block.result.coords[row, :nr],
                    heavy[row, :, :nr].T.astype(np.float32) * np.float32(2 ** -8))
            np.testing.assert_array_equal(block.result.vaa,
                group.vaa[d.group_row0:d.group_row0 + d.n_conf])
            np.testing.assert_array_equal(block.molecule_ids,
                np.repeat(group.block_runs(d.block_index)['mol_id'],
                          group.block_runs(d.block_index)['n_conf']))
            assert block.result.coords.shape == (d.n_conf, d.library_width, 3)
            if color:
                assert block.result.color_self is not None
            assert all(t >= 0 for t in block.result.timings)
            assert not hasattr(d, "counts") and not hasattr(d, "molecule_ids")
            group.close()
    finally:
        source.close()


def test_directory_expensive_first_and_compact_gid(tmp_path):
    corpus = make_cpu_corpus(tmp_path / 'corpus', sizes=(2, 3, 1, 2),
                             n_real=(2, 14, 2, 14), block_rows=3)
    source = PhysicalCPUFixtureSource(corpus, group_indices=[1])
    descriptors = source.cpu_block_descriptors()
    assert [d.n_conf for d in descriptors] == [3, 2]
    assert [d.compact_row0 for d in descriptors] == [0, 3]
    assert [d.gid_base for d in descriptors] == [3, 6]
    source.close()


@pytest.mark.parametrize('section', ['vaa', 'color_self'])
def test_corrupt_stored_self_refused_with_group_and_block(section, tmp_path):
    corpus = make_cpu_corpus(tmp_path / 'corpus', color=True)
    # Preserve the scan identity to exercise the worker's own section check,
    # rather than the reader's changed-file preflight.
    entry = corpus.group_table[0]
    with corpus.group(0) as group:
        offset = group._og.sections[section]['offset']
        path = Path(group.path)
    stat = path.stat()
    with path.open('r+b') as file:
        file.seek(offset)
        value = file.read(1)
        file.seek(offset)
        file.write(bytes([value[0] ^ 1]))
    import os
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    source = PhysicalCPUFixtureSource(corpus, color=True)
    with pytest.raises(CorpusValidationError, match=r'group 0 block.*' + section):
        source.cpu_block_descriptors()


@pytest.mark.parametrize('with_color_self', [False, True])
def test_native_verified_self_binding_and_optional_recompute(with_color_self, tmp_path):
    corpus = make_cpu_corpus(tmp_path / 'corpus', color=True,
                             with_color_self=with_color_self)
    source = PhysicalCPUFixtureSource(corpus, color=True)
    q = _queries(source)
    cache = R3gWorkerSelfCache(source, 'q' * 64)
    for d in source.cpu_block_descriptors():
        decoded = source.decode_cpu_block(d)
        handle, reason, hit, replaced = cache.block(cpu, d, decoded,
            decoded.result.coords, q.coords, q.types, q.n_real,
            np.asarray(q.interaction_r, np.float32), np.asarray(q.interaction_p, np.float32))
        assert reason is None and not hit and not replaced
        assert handle.library.library_shape_from_stored_volume
        assert handle.library.library_color_from_sidecar == with_color_self
        assert handle.library.color_terms == (0 if with_color_self else d.n_conf)
        assert cache.routes[d.group_index, d.block_index]['vaa'] == 'verified_stored'
    source.close()


@pytest.mark.parametrize('workers', [1, 8])
def test_shape_worker_topk_identity_across_writer_block_sizes(workers, tmp_path):
    expected = None
    for rows in (2, 4, 8):
        source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / str(rows), block_rows=rows))
        q = _queries(source)
        engine = Engine(backend='cpu')
        stats, result = engine.run_worker_owned_cpu(source, q, _config(False), workers=workers,
            k=5, level='molecule', self_overlap_policy='prepared',
            shape_backend='pipelined_field', shape_grain='library_atom', shape_width=8)
        actual = result[0]
        if expected is None:
            expected = actual
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)
        assert stats.cpu_scan['stored_vaa_verified_blocks'] == len(source.cpu_block_descriptors())
        assert stats.cpu_scan['library_shape_self_evaluations'] == 0
        source.close()


@pytest.mark.parametrize('workers', [1, 8])
@pytest.mark.parametrize('with_color_self', [False, True])
@pytest.mark.parametrize('level', ['conformer', 'molecule'])
def test_combo_worker_matches_dense_and_stored_route(workers, with_color_self, level, tmp_path):
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', color=True,
        with_color_self=with_color_self), color=True)
    q = _queries(source)
    config = _config(True)
    blocks = [source.decode_cpu_block(d) for d in source.cpu_block_descriptors()]
    coords = np.concatenate([b.result.coords for b in blocks])
    counts = np.concatenate([b.result.counts for b in blocks])
    gids = np.concatenate([b.descriptor.gid_base + np.arange(b.descriptor.n_conf) for b in blocks])
    mids = np.concatenate([b.molecule_ids for b in blocks])
    vaa = np.concatenate([b.result.vaa for b in blocks])
    color_self = (np.concatenate([b.result.color_self for b in blocks])
                  if with_color_self else None)
    # Build one authenticated denominator handle for the exact same rows. The
    # scorer comparison exercises block partitioning, not self-algorithm drift.
    from types import SimpleNamespace
    whole = SimpleNamespace(group_index=blocks[0].descriptor.group_index,
        block_index=blocks[0].descriptor.block_index, n_conf=len(coords),
        gid_base=blocks[0].descriptor.gid_base, library_width=coords.shape[1])
    decoded = SimpleNamespace(descriptor=whole,
        result=SimpleNamespace(vaa=vaa, color_self=color_self, counts=counts))
    handle, *_ = R3gWorkerSelfCache(source, 'd' * 64).block(cpu, whole, decoded,
        coords, q.coords, q.types, q.n_real, np.asarray(q.interaction_r, np.float32),
        np.asarray(q.interaction_p, np.float32))
    kwargs = dict(query_coords=q.coords, query_types=q.types, query_n_atoms=q.n_real,
        library_coords=coords, library_counts=counts,
        pair_query_row=np.zeros(len(coords), np.int32),
        pair_library_row=np.arange(len(coords), dtype=np.int32),
        interaction_r=q.interaction_r, interaction_p=q.interaction_p,
        steps=25, start_mode=1, color=True, color_mode='combo_optimal', mixing=.5,
        shape_backend='scalar_analytic', shape_grain='scalar', shape_width=1,
        pose_contract='shared_fp32_reduced_v1', prepared_self_overlaps=handle)
    dense = cpu.score_pairs(**kwargs)
    from tests.test_cpu_worker_owned_scan import _reference_topk
    expected = _reference_topk(dense, gids, mids, k=5, level=level, rank_column=0)
    engine = Engine(backend='cpu')
    for repeat in range(2):
        stats, actual = engine.run_worker_owned_cpu(source, q, config, workers=workers,
            k=5, level=level, self_overlap_policy='prepared',
            shape_backend='scalar_analytic', shape_grain='scalar', shape_width=1)
        for a, e in zip(actual[0], expected):
            np.testing.assert_array_equal(a, e)
        counters = stats.cpu_scan
        assert counters['stored_vaa_verified_blocks'] == len(blocks)
        assert counters['color_self_sidecar_blocks'] == (len(blocks) if with_color_self else 0)
        assert counters['library_shape_self_evaluations'] == 0
        assert counters['library_color_self_evaluations'] == (
            0 if with_color_self else len(coords))
        assert counters['library_color_self_binding_evaluations'] == (
            0 if with_color_self else len(coords))
    source.close()


@pytest.mark.parametrize('workers', [1, 8])
def test_exact_rank_ties_use_source_gid_across_expensive_first_shards(workers, tmp_path):
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', block_rows=3,
        tied=True, sizes=(1, 3, 2, 1)), color=False)
    assert source.cpu_block_descriptors()[0].gid_base != 0
    q = _queries(source)
    _, result = Engine(backend='cpu').run_worker_owned_cpu(source, q, _config(False),
        workers=workers, k=7, self_overlap_policy='prepared',
        shape_backend='pipelined_field', shape_grain='library_atom', shape_width=8)
    np.testing.assert_array_equal(result[0][1], np.arange(7))
    source.close()


@pytest.mark.parametrize('section', ['vaa', 'color_self', 'coords'])
def test_worker_refuses_authenticated_slice_change_after_setup(section, tmp_path):
    import os
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', color=True), color=True)
    d = source.cpu_block_descriptors()[0]
    with source.corpus.group(d.group_index) as group:
        path = Path(group.path)
        offset = group._og.sections[section]['offset']
        if section == 'coords':
            offset += int(group.blocks[d.block_index]['offset'])
        else:
            offset += d.group_row0 * (4 if section == 'vaa' else 8)
    stat = path.stat()
    with path.open('r+b') as file:
        file.seek(offset)
        value = file.read(1)
        file.seek(offset)
        file.write(bytes([value[0] ^ 1]))
    # A preserved stat fingerprint must still fail the authenticated slice.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(CorpusValidationError, match='group 0 block'):
        source.decode_cpu_block(d)
    source.close()


def test_d033_native_sink_refuses_inconsistent_stored_shape(tmp_path):
    # Correct hashes certify content, not its scientific truth. The runtime
    # self-overlap bound still refuses self values inconsistent with a scored row.
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus'))
    q = _queries(source)
    d = source.cpu_block_descriptors()[0]
    # Model a self value accepted by the writer/hash chain but under-claimed
    # relative to the same coordinates, without bypassing the native sink.
    original_decode = source.decode_cpu_block
    def lowered(descriptor):
        decoded = original_decode(descriptor)
        decoded.result.vaa[:] = .01
        return decoded
    source.decode_cpu_block = lowered
    with pytest.raises(ValueError, match='self-overlap|self overlap|bound'):
        Engine(backend='cpu').run_worker_owned_cpu(source, q, _config(False), workers=1,
            k=3, self_overlap_policy='prepared', shape_backend='pipelined_field',
            shape_grain='library_atom', shape_width=8)
    source.close()


@pytest.mark.parametrize('color', [False, True])
def test_writer_block_size_worker_count_topk_identity_matrix(color, tmp_path):
    import psutil
    expected = None
    physical = psutil.cpu_count(logical=False) or 1
    for block_rows in (1024, 2048, 4096):
        sizes = (128,) * 33
        source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / str(block_rows),
            color=color, block_rows=block_rows, sizes=sizes, n_real=(2,) * len(sizes)), color=color)
        q = _queries(source)
        for workers in sorted({1, 8, physical}):
            kwargs = (dict(shape_backend='scalar_analytic', shape_grain='scalar', shape_width=1)
                      if color else dict(shape_backend='pipelined_field',
                      shape_grain='library_atom', shape_width=8))
            stats, result = Engine(backend='cpu').run_worker_owned_cpu(source, q,
                _config(color), workers=workers, k=15, self_overlap_policy='prepared', **kwargs)
            scores, gids, mids = result[0]
            # Each molecule owns exactly 128 rows in this fixture. Assert the
            # public identity pair as well as all retained score columns.
            identity = np.stack((mids, gids % 128), axis=1)
            if expected is None:
                expected = (scores, identity)
            np.testing.assert_array_equal(identity, expected[1])
            np.testing.assert_array_equal(scores, expected[0])
            assert stats.cpu_scan['stored_vaa_verified_blocks'] == len(source.cpu_block_descriptors())
        source.close()


@pytest.mark.parametrize('defect', ['split', 'high_id'])
def test_worker_directory_refuses_cross_block_molecule_and_signed_id_overflow(defect, tmp_path, monkeypatch):
    import roshambo3.engine.r3g_cpu as module
    corpus = make_cpu_corpus(tmp_path / 'corpus', block_rows=8)
    original = module._read_section_rows
    def malformed(group, name, first, count):
        values = original(group, name, first, count)
        if name == 'runs':
            values = values.copy()
            if defect == 'split':
                values[0]['n_conf'] -= 1
            else:
                values[0]['mol_id'] = 1 << 63
            group._og.sections['runs']['sha256'] = hashlib.sha256(values).hexdigest()
        return values
    monkeypatch.setattr(module, '_read_section_rows', malformed)
    with pytest.raises(CorpusValidationError, match='molecule spans|mol_id'):
        PhysicalCPUFixtureSource(corpus).cpu_block_descriptors()


def test_format3_worker_color_gate_uses_actual_generator_before_decode(tmp_path, monkeypatch):
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', color=True), color=True)
    q = _queries(source)
    source.color_generator.interaction_matrix_r[1, 1] = 2
    monkeypatch.setattr(source, 'decode_cpu_block', lambda _: pytest.fail('decoded before color gate'))
    from roshambo3.format.r3g.descriptor import DefinitionMismatch
    with pytest.raises(DefinitionMismatch, match='r'):
        Engine(backend='cpu').run_worker_owned_cpu(source, q, _config(True), workers=1, k=1)
    source.close()


@pytest.mark.parametrize('with_color_self', [False, True])
def test_rescore_stored_self_routes_and_counters(with_color_self, tmp_path):
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', color=True,
        with_color_self=with_color_self), color=True)
    q = _queries(source)
    config = ScreenConfig(color=True, color_placement=ColorPlacement.AT_POSE,
                          mixing=.5, steps=25, start_mode=1)
    engine = Engine(backend='cpu')
    for repeat in range(2):
        stats, _ = engine.run_worker_owned_cpu(source, q, config, workers=8,
            k=3, self_overlap_policy='prepared', shape_backend='pipelined_field',
            shape_grain='library_atom', shape_width=8)
        assert stats.cpu_scan['stored_vaa_verified_blocks'] == stats.n_blocks
        assert stats.cpu_scan['color_self_sidecar_blocks'] == (stats.n_blocks if with_color_self else 0)
        assert stats.cpu_scan['library_color_self_binding_evaluations'] == (
            0 if with_color_self else stats.n_conformers)
        assert stats.cpu_scan['library_color_self_evaluations'] == (
            0 if with_color_self else stats.n_conformers)
        assert stats.cpu_scan['library_shape_self_evaluations'] == 0
    source.close()


def test_selected_descriptors_preserve_directory_claim_order(tmp_path):
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', block_rows=3))
    q = _queries(source)
    expected = source.cpu_block_descriptors()
    seen = []
    original = source.decode_cpu_block
    def track(d):
        seen.append(d)
        return original(d)
    source.decode_cpu_block = track
    Engine(backend='cpu').run_worker_owned_cpu(source, q, _config(False), workers=1,
        block_descriptors=tuple(reversed(expected)), k=3, self_overlap_policy='prepared',
        shape_backend='pipelined_field', shape_grain='library_atom', shape_width=8)
    assert [id(d) for d in seen] == [id(d) for d in expected]
    source.close()


def test_directory_retains_no_expanded_conformer_metadata(tmp_path, monkeypatch):
    from roshambo3.format.r3g.reader import Group
    small = make_cpu_corpus(tmp_path / 'small', block_rows=1, sizes=(1, 1), n_real=(2, 2))
    large = make_cpu_corpus(tmp_path / 'large', block_rows=1, sizes=(4096, 4096), n_real=(2, 2))
    monkeypatch.setattr(Group, 'counts', property(
        lambda _: pytest.fail('directory expanded per-conformer counts')))
    sources = [PhysicalCPUFixtureSource(corpus) for corpus in (small, large)]
    for source in sources:
        descriptors = source.cpu_block_descriptors()
        assert len(descriptors) == 2
        assert all(not hasattr(d, "counts") and not hasattr(d, "molecule_ids")
                   for d in descriptors)
        assert all(not hasattr(group, '_cache') for group in source._cpu_groups.values())
    assert sources[0]._cpu_metadata_owned_bytes == sources[1]._cpu_metadata_owned_bytes
    for source in sources:
        source.close()


@pytest.mark.parametrize('workers', [1, 3])
def test_self_snapshot_residency_is_bounded_by_active_workers(workers, tmp_path):
    sizes = (2,) * 48
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', color=True,
        with_color_self=False, sizes=sizes, n_real=(2,) * len(sizes), block_rows=2), color=True)
    q = _queries(source)
    engine = Engine(backend='cpu')
    stats, _ = engine.run_worker_owned_cpu(source, q, _config(True), workers=workers,
        k=5, self_overlap_policy='prepared', shape_backend='scalar_analytic',
        shape_grain='scalar', shape_width=1)
    assert stats.n_blocks == 48
    assert stats.cpu_scan['self_cache_peak_blocks'] <= workers
    assert stats.cpu_scan['prepared_self_retained_bytes'] == 0
    assert engine._worker_self_cache.entries == {}
    assert engine._worker_self_cache.routes == {}
    assert stats.cpu_scan['library_color_self_binding_evaluations'] == sum(sizes)
    source.close()


@pytest.mark.parametrize('color', [False, True])
def test_public_cpu_worker_facade_consumes_format3_native_block(color, tmp_path):
    from roshambo3.engine import CPUWorkerOptions, TopK
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', color=color), color=color)
    q = _queries(source)
    config = _config(color)
    config.cpu_worker = CPUWorkerOptions(workers=2)
    stats, result = Engine(backend='cpu').run(source, q, config, TopK(3))
    assert len(result[0][1]) == 3
    assert stats.cpu_scan['stored_vaa_verified_blocks'] == stats.n_blocks
    assert stats.cpu_scan['self_cache_peak_blocks'] <= 2
    source.close()


@pytest.mark.parametrize('policy', ['prepared', 'recompute'])
def test_query_matrix_mismatch_refused_before_block_claim(policy, tmp_path, monkeypatch):
    source = PhysicalCPUFixtureSource(make_cpu_corpus(tmp_path / 'corpus', color=True), color=True)
    q = _queries(source)
    q.interaction_r = q.interaction_r.copy()
    q.interaction_r[1, 1] = 2
    monkeypatch.setattr(source, 'decode_cpu_block', lambda _: pytest.fail('decoded before matrix gate'))
    from roshambo3.format.r3g.descriptor import DefinitionMismatch
    with pytest.raises(DefinitionMismatch, match='r'):
        Engine(backend='cpu').run_worker_owned_cpu(source, q, _config(True),
            workers=1, k=1, self_overlap_policy=policy)
    source.close()


def test_shape_scan_of_color_corpus_reads_no_color_sections(tmp_path, monkeypatch):
    import roshambo3.engine.r3g_cpu as module
    corpus = make_cpu_corpus(tmp_path / 'corpus', color=True)
    original = module._read_section_rows
    def shape_only(group, name, first, count):
        assert name == 'runs' or name == 'vaa', f'shape path read {name}'
        return original(group, name, first, count)
    monkeypatch.setattr(module, '_read_section_rows', shape_only)
    source = PhysicalCPUFixtureSource(corpus, color=False)
    q = _queries(source)
    stats, _ = Engine(backend='cpu').run_worker_owned_cpu(source, q, _config(False),
        workers=2, k=3, self_overlap_policy='prepared', shape_backend='pipelined_field',
        shape_grain='library_atom', shape_width=8)
    assert stats.cpu_scan['color_self_sidecar_blocks'] == 0
    assert stats.cpu_scan['stored_vaa_verified_blocks'] == stats.n_blocks
    assert all(set(binding) == {'runs', 'vaa', '_coords'} for binding in source._cpu_bindings.values())
    source.close()

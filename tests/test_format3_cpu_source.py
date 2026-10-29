"""Ordinary CPU jobs share the physical worker decoder and source identities."""
import gc
import weakref

import numpy as np
import pytest

from roshambo3.engine import R3gSource
from tests.codec_fixtures import make_codec_fixture


@pytest.mark.parametrize('groups', [None, [1, 0], [1]])
@pytest.mark.parametrize('launch', [1, 100])
@pytest.mark.parametrize('depth', [0, 3])
def test_native_cpu_jobs_match_independent_rows(groups, launch, depth, tmp_path):
    fx = make_codec_fixture(tmp_path / 'corpus', color_layout='merged')
    source = R3gSource(fx.corpus, groups=groups, color=True, tile=False,
                      workers=2, prefetch_depth=depth, conformers_per_launch=launch)
    if launch == 100 and groups != [1]:
        assert any(job.n_blocks > 1 for job in source.jobs)
    offsets = np.concatenate(([0], np.cumsum([g['n_conf'] for g in fx.corpus.group_table])))
    for job in source.jobs:
        # Targets are bounded by the target or one indivisible physical block.
        assert job.n_conf <= max(launch, max(int(b['rows']) for b in fx.corpus.block_directory()))
    for job, block in zip(source.jobs, source.blocks(), strict=True):
        sl = slice(block.gid_base, block.gid_base + block.n_conformers)
        np.testing.assert_array_equal(block.coords,
            fx.shape_xyz_i16[sl].transpose(0, 2, 1).astype(np.float32) * np.float32(2**-8))
        np.testing.assert_array_equal(block.counts, fx.counts[sl])
        np.testing.assert_array_equal(block.mol_id, fx.molecule_ids[sl])
        assert block.gid_base == offsets[job.group_index] + job.row0
        assert block.tile_mode == 0 and block.buffer is None
        assert not block.tail_molecule_open and block.color_offset == -1
    source.close()
    assert source._cpu_groups == {}


def test_early_close_joins_prefetch_and_releases_job_arrays(tmp_path, monkeypatch):
    fx = make_codec_fixture(tmp_path / 'corpus', color_layout='merged')
    source = R3gSource(fx.corpus, tile=False, color=True, workers=2,
                      prefetch_depth=2, conformers_per_launch=1)
    decode = source._decode_cpu
    references = []
    def tracked(job):
        block = decode(job)
        references.extend((weakref.ref(block.coords), weakref.ref(block.counts), weakref.ref(block.mol_id)))
        return block
    monkeypatch.setattr(source, '_decode_cpu', tracked)
    stream = source.blocks()
    block = next(stream)
    del block
    stream.close()  # Drains every submitted job, including unconsumed ones.
    source.close()
    gc.collect()
    assert references and all(ref() is None for ref in references)
    assert source._cpu_groups == {}


def test_shape_on_color_reads_no_feature_or_color_self_sections(tmp_path, monkeypatch):
    from roshambo3.engine import r3g_cpu
    fx = make_codec_fixture(tmp_path / 'corpus', color_layout='merged')
    seen = []
    read = r3g_cpu._read_section_rows
    def tracked(group, name, first, count):
        seen.append(name)
        return read(group, name, first, count)
    monkeypatch.setattr(r3g_cpu, '_read_section_rows', tracked)
    source = R3gSource(fx.corpus, tile=False, color=False, workers=2)
    for block in source.blocks():
        assert block.counts.shape[1] == 1
    source.close()
    assert set(seen) == {'runs', 'vaa'}


def test_ordinary_cpu_preserves_explicit_fixed_pose_budget_override(tmp_path):
    from roshambo3.engine import Engine, ScreenConfig, DenseScores, from_arrays
    from roshambo3.backend.spec import PoseSearch
    fx = make_codec_fixture(tmp_path / 'corpus', color_layout='none')
    source = R3gSource(fx.corpus, tile=False, color=False, prefetch_depth=0)
    queries = from_arrays([fx.shape_xyz_i16[0, :, :fx.n_real[0]].T.astype(np.float32) * np.float32(2**-8)], ['q'])
    reason = 'fixed-pose format-3 CPU production regression'
    config = ScreenConfig(pose_search=PoseSearch.ANALYTIC, steps=1,
                          lr_q=0., lr_t=0., budget_override=reason)
    engine = Engine(backend='cpu')
    try:
        stats, result = engine.run(source, queries, config, DenseScores())
        assert stats.n_conformers == len(fx.global_ids)
        assert np.isfinite(result.scores).all()
    finally:
        engine.close()
        source.close()

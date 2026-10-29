"""Untimed Phase 1 coverage, exclusively conformance and writer corpora."""
from pathlib import Path

import numpy as np
import pytest

from roshambo3.engine import R3gSource, PinnedPool, make_source
from roshambo3.engine.buffers import pack_into
from roshambo3.format.r3g import reader, reference as R, native as N
from roshambo3.format.r3g.writer import Molecule, write_corpus
from roshambo3.numerics import WIRE_QUANT_STEP
from roshambo3.pharmacophore import RDKitPharmacophoreGenerator

DATA = Path(__file__).parent / "data" / "format3"


def opened(root, cache):
    reader.verify(str(root), cache_dir=str(cache), use_cache=False)
    return reader.open_corpus(str(root), cache_dir=str(cache))


@pytest.fixture(scope="module", params=[False, True], ids=["shape", "color"])
def written(request, tmp_path_factory):
    color = request.param
    root = tmp_path_factory.mktemp("gpu-source") / "corpus"
    gen = RDKitPharmacophoreGenerator()
    rng = np.random.default_rng(318)
    molecules = [Molecule(
        mol_id=100 + i, coords=rng.integers(-1000, 1001, (nc, 3, 8), dtype=np.int16),
        vaa=np.full(nc, 100, np.float32), features=[(1, [0]), (2, [1, 2])] if color else [],
        color_self=np.full(nc, 2, np.float64) if color else None,
        ext_id=f"m{i}", smiles="CCCCCCCC")
        for i, nc in enumerate([1, 3, 7, 40, 2, 17, 5, 33, 1])]
    write_corpus(str(root), molecules, color=gen.descriptor_color() if color else None,
                 block_rows=11, prep_version="0.1.0.dev0", prep_commit=None)
    corpus = opened(root, root.parent / "scan")
    return corpus, color, gen


def source_with_pool(corpus, **kwargs):
    source = R3gSource(corpus, **kwargs)
    source._pool = PinnedPool(max(2, source.prefetch_depth + 1),
                              source.max_block_elements, pinned=False)
    return source


def reference_group(corpus, gi):
    entry = corpus.group_table[gi]
    name = R.group_file_name(entry["bucket"], entry["group"])
    group = R.open_group(str(Path(corpus.root) / name))
    try:
        result = R.decode_group(group)
        runs = group.section_array("runs")
        result["mol_id"] = np.repeat(runs["mol_id"].astype(np.int64), runs["n_conf"])
        return result
    finally:
        group.close()


@pytest.mark.parametrize("launch", [1, 50, 100_000])
@pytest.mark.parametrize("depth", [3, 8])
@pytest.mark.parametrize("planar", [False, True])
def test_whole_block_launches_match_independent_layout(written, launch, depth, planar):
    corpus, color, gen = written
    source = source_with_pool(corpus, color=color, color_generator=gen,
                              conformers_per_launch=launch, workers=8,
                              prefetch_depth=depth, planar=planar)
    assert any(int(r["rows"]) % 32 for r in corpus.block_directory())
    if launch == 100_000:
        assert source.n_launches == len(corpus.group_table)
        assert any(j.n_blocks > 1 for j in source.jobs)
    references = {gi: reference_group(corpus, gi) for gi in source.group_indices}
    offsets = np.concatenate(([0], np.cumsum([g["n_conf"] for g in corpus.group_table])))
    seen = 0
    for job, block in zip(source.jobs, source.blocks(), strict=True):
        try:
            e = corpus.group_table[job.group_index]
            r = references[job.group_index]
            sl = slice(job.row0, job.row0 + job.n_conf)
            merged = np.ascontiguousarray(r["merged"][sl])
            expected = np.full(block.coords.size, 27183, dtype=np.int16)
            coords, _ = pack_into(merged, e["L"], expected, WIRE_QUANT_STEP,
                                   tile=True, planar=planar)
            np.testing.assert_array_equal(block.coords, coords)
            np.testing.assert_array_equal(block.counts, r["counts"][sl])
            np.testing.assert_array_equal(block.mol_id, r["mol_id"][sl])
            assert block.gid_base == offsets[job.group_index] + job.row0
            assert block.L == e["L"] and block.L_shape == e["L_field_width"]
            assert block.color_offset == -1 and not block.tail_molecule_open
            assert block.buffer is not None and np.shares_memory(block.coords, block.buffer)
            assert block.tile_mode == (-1 if planar else 1)
            seen += block.n_conformers
        finally:
            source._release(block)
    assert seen == source.n_conformers == corpus.n_conf
    assert source._pool._q.qsize() == len(source._pool.buffers)


@pytest.mark.parametrize("variant", ["shape", "color"])
@pytest.mark.parametrize("sparse", [False, True])
def test_conformance_all_widths_and_sparse_global_ids(variant, sparse, tmp_path):
    corpus = opened(DATA / variant, tmp_path / "scan")
    selected = ([len(corpus.group_table) - 1, 0, 2] if sparse
                else list(reversed(range(len(corpus.group_table)))))
    gen = RDKitPharmacophoreGenerator()
    if corpus.has_color:
        # The conformance fixture intentionally uses identity matrices, so
        # supply the actual generator with those exact recorded semantics.
        gen.interaction_matrix_r = np.asarray(corpus.color["r"]["values"], np.float32)
        gen.interaction_matrix_p = np.asarray(corpus.color["p"]["values"], np.float32)
    source = source_with_pool(corpus, groups=selected, color=variant == "color",
                              color_generator=gen, conformers_per_launch=7)
    for job in source.jobs:
        block = source._decode(job)
        try:
            e = corpus.group_table[job.group_index]
            ref = reference_group(corpus, job.group_index)
            sl = slice(job.row0, job.row0 + job.n_conf)
            out = np.empty(block.coords.size, dtype=np.int16)
            expected, _ = pack_into(ref["merged"][sl], e["L"], out, WIRE_QUANT_STEP)
            np.testing.assert_array_equal(block.coords, expected)
            np.testing.assert_array_equal(block.mol_id, ref["mol_id"][sl])
            np.testing.assert_array_equal(block.counts, ref["counts"][sl])
            assert block.gid_base == sum(g["n_conf"] for g in corpus.group_table[:job.group_index]) + job.row0
            assert block.L_shape == e["L_field_width"]
        finally:
            source._release(block)
    assert source.n_conformers == sum(corpus.group_table[i]["n_conf"] for i in selected)
    assert source.bucket_widths() == sorted({corpus.group_table[i]["L"] for i in selected})


def test_shape_screen_color_corpus_retains_width_but_only_shape_counts(written):
    corpus, color, _ = written
    if not color:
        return
    source = source_with_pool(corpus, color=False)
    block = source._decode(source.jobs[0])
    try:
        assert block.counts.shape == (block.n_conformers, 1)
        assert block.L == corpus.group_table[0]["L"]
        assert block.color_offset == -1
        assert set(source._sidecars[0]) == {"runs"}
        expected = reference_group(corpus, 0)["merged"].copy()
        expected[:, :, 8:] = 0
        out = np.empty(block.coords.size, np.int16)
        coords, _ = pack_into(expected, block.L, out, WIRE_QUANT_STEP)
        np.testing.assert_array_equal(block.coords, coords)
    finally:
        source._release(block)


def test_pool_returned_after_failure_and_prefetch_close(written, monkeypatch):
    corpus, _, _ = written
    source = source_with_pool(corpus, conformers_per_launch=1, prefetch_depth=8)
    original = N.write_gpu_launch
    def fail(*args, **kwargs):
        raise ValueError("injected native failure")
    monkeypatch.setattr(N, "write_gpu_launch", fail)
    with pytest.raises(ValueError, match="injected"):
        source._decode(source.jobs[0])
    assert source._pool._q.qsize() == len(source._pool.buffers)
    monkeypatch.setattr(N, "write_gpu_launch", original)
    stream = source.blocks()
    first = next(stream)
    source._release(first)
    stream.close()
    assert source._pool._q.qsize() == len(source._pool.buffers)


def test_bounded_source_metadata_and_factory(written):
    from roshambo3.api import open_corpus
    from roshambo3.web.storage import BoundedSource
    corpus, color, gen = written
    assert open_corpus(corpus) is corpus
    assert isinstance(make_source(corpus), R3gSource)
    source = BoundedSource(corpus, color=color, color_generator=gen)
    source.cache_groups = 1
    source._pool = PinnedPool(2, source.max_block_elements, pinned=False)
    for job in source.jobs:
        block = source._decode(job)
        source._release(block)
        assert len(source._sidecars) <= 1
    source.close()
    assert not source._sidecars


@pytest.mark.parametrize("selection", [[-1], [100_000], [0, 0]])
def test_invalid_group_selection(written, selection):
    with pytest.raises(ValueError):
        R3gSource(written[0], groups=selection)


def test_bounded_metadata_eviction_keeps_live_arrays(tmp_path):
    from roshambo3.web.storage import BoundedSource
    corpus = opened(DATA / "color", tmp_path / "scan")
    source = BoundedSource(corpus)
    source.cache_groups = 2
    live = source._group_sidecars(0)["runs"]
    saved = live.copy()
    for gi in range(len(corpus.group_table)):
        source._group_sidecars(gi)
        assert len(source._sidecars) <= 2
    np.testing.assert_array_equal(live, saved)
    assert 0 not in source._sidecars


def test_api_open_path_dispatch_uses_scan_gate(written, tmp_path, monkeypatch):
    from roshambo3.api import open_corpus
    from roshambo3.format.r3g.errors import CorpusValidationError
    corpus, _, _ = written
    monkeypatch.setenv("ROSHAMBO3_SCAN_CACHE", str(tmp_path / "scan"))
    with pytest.raises(CorpusValidationError, match="scan"):
        open_corpus(corpus.root)
    reader.verify(corpus.root)
    assert isinstance(open_corpus(corpus.root), reader.Corpus)


def test_shape_on_untrusted_color_never_gates_or_reads_features(tmp_path, monkeypatch):
    from roshambo3 import definitions
    from roshambo3.format.r3g.reader import Group
    gen = RDKitPharmacophoreGenerator()
    color = gen.descriptor_color()
    color["definition"] += "\n# A custom untrusted definition for shape-only coverage.\n"
    root = tmp_path / "untrusted-color"
    molecule = Molecule(101, np.zeros((3, 3, 8), np.int16),
                        np.full(3, 10, np.float32), features=[(1, [0])],
                        color_self=np.ones(3), ext_id="custom", smiles="CCCCCCCC")
    write_corpus(str(root), [molecule], color=color, block_rows=2,
                 prep_version="0.1.0.dev0", prep_commit=None)
    corpus = opened(root, tmp_path / "scan")
    with pytest.raises(definitions.UntrustedDefinitionError):
        definitions.check_definition(corpus.color, trust_file=str(tmp_path / "empty-trust"))
    def forbidden(*args, **kwargs):
        raise AssertionError("shape-only source must not invoke a color gate")
    monkeypatch.setattr(definitions, "check_corpus_color", forbidden)
    original_section = Group.section
    def section(group, name):
        assert not name.startswith("feature"), "shape-only source read feature metadata"
        return original_section(group, name)
    monkeypatch.setattr(Group, "section", section)
    source = source_with_pool(corpus, color=False)
    block = source._decode(source.jobs[0])
    try:
        assert block.counts.shape == (3, 1)
        np.testing.assert_array_equal(block.counts[:, 0], [8, 8, 8])
        assert block.L == corpus.group_table[0]["L"]
        assert block.L_shape == corpus.group_table[0]["L_field_width"]
        assert set(source._sidecars[0]) == {"runs"}
    finally:
        source._release(block)

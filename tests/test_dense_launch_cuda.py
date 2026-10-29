"""Dense launches carry no pair index; sparse launches still read one.

A dense launch scores every query against every row of the block, query-major,
and the engine passes ``pair_query_row=None, pair_library_row=None``
for it: the device derives each row's (query, library row) from the launch
shape, and the top-K maps rows back the same way. The explicit pair list is the
sparse form.

The arithmetic of every score is unchanged -- only how a thread finds its
query and conformer -- so the two forms must agree **bit for bit** on every
output: the dense score block, the conformer top-K and the molecule top-K, in
shape, rescore and combo, on the FIELD and ANALYTIC pose searches. The check
runs the same engine screen twice in one build, once as shipped (dense) and
once through a backend that writes the equivalent query-major pair list out
explicitly, which is exactly what the engine once passed.

The sparse path has no production caller, so it is exercised directly: a
shuffled subset of pairs through the kernel and the conformer top-K, against
the dense block's rows and a host ranking of the same rows.
"""
from contextlib import closing
import types

import numpy as np
import pytest

from roshambo3.backend import cuda
from roshambo3.engine import (DenseScores, Engine, MoleculeTopK, R3gSource,
                              ScreenConfig, TopK)
from roshambo3.backend.spec import ColorPlacement, PoseSearch
from roshambo3.numerics import rank_order
from roshambo3.scores import N_SCORE_COLUMNS
from tests.format3_gpu_fixtures import prepared_gpu_corpus  # noqa: F401
from gpu_device import device_present

pytestmark = pytest.mark.skipif(not cuda.available() or not device_present(),
                                reason="CUDA extension/device unavailable")


def _explicit_backend(calls=None):
    """The CUDA backend, but every dense launch is replaced by its explicit
    query-major pair list -- the arrays `Engine._pair_index` used to build.
    Writeable, so the wrapper range-checks them too."""
    module = types.ModuleType("roshambo3.backend.cuda")
    for name in dir(cuda):
        if not name.startswith("__"):
            setattr(module, name, getattr(cuda, name))

    def score_pairs(**kw):
        assert kw["pair_query_row"] is None and kw["pair_library_row"] is None, (
            "the engine must launch dense")
        n_q = int(kw["query_coords"].shape[0])
        n = int(kw["library_counts"].shape[0])
        kw["pair_query_row"] = np.repeat(np.arange(n_q, dtype=np.int32), n)
        kw["pair_library_row"] = np.tile(np.arange(n, dtype=np.int32), n_q)
        if calls is not None:
            calls.append(dict(kw))
        return cuda.score_pairs(**kw)

    module.score_pairs = score_pairs
    return module


MODES = {
    "shape": dict(),
    "rescore": dict(color=True, mixing=0.5,
                    color_placement=ColorPlacement.AT_POSE),
    "combo": dict(color=True, mixing=0.5,
                  color_placement=ColorPlacement.IN_GRADIENT),
}


def _config(fx, mode, pose_search):
    if mode != "shape" and fx.generator is None:
        pytest.skip("color modes need the color fixture")
    if mode == "shape" and fx.generator is not None:
        pytest.skip("the color fixture's queries carry color; shape runs on the shape fixture")
    if pose_search is PoseSearch.ANALYTIC and mode == "combo":
        pytest.skip("combo is a FIELD mode")
    return ScreenConfig(pose_search=pose_search, **MODES[mode])


def _screen(fx, config, sink, backend=None):
    src = R3gSource(fx.corpus, color=config.color,
                    color_generator=fx.generator if config.color else None,
                    prefetch_depth=0)
    with closing(Engine(backend=backend) if backend is not None
                 else Engine()) as engine:
        return engine.run(src, fx.queries, config, sink)[1]


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint32 if a.dtype == np.float32 else a.dtype)


@pytest.mark.parametrize("pose_search", [PoseSearch.FIELD, PoseSearch.ANALYTIC],
                         ids=["field", "analytic"])
@pytest.mark.parametrize("mode", list(MODES))
class TestDenseIsBitIdenticalToTheExplicitPairList:
    def test_the_dense_score_block(self, prepared_gpu_corpus, mode, pose_search):
        fx = prepared_gpu_corpus
        config = _config(fx, mode, pose_search)
        calls = []
        dense = _screen(fx, config, DenseScores(max_bytes=1 << 26))
        explicit = _screen(fx, config, DenseScores(max_bytes=1 << 26),
                           _explicit_backend(calls))
        assert calls, "the explicit arm never launched"
        np.testing.assert_array_equal(dense.gids, explicit.gids)
        np.testing.assert_array_equal(_bits(dense.scores), _bits(explicit.scores))

    @pytest.mark.parametrize("sink", [TopK, MoleculeTopK], ids=["conformer", "molecule"])
    def test_the_device_topk(self, prepared_gpu_corpus, mode, pose_search, sink):
        fx = prepared_gpu_corpus
        config = _config(fx, mode, pose_search)
        dense = _screen(fx, config, sink(k=50))
        explicit = _screen(fx, config, sink(k=50), _explicit_backend())
        assert len(dense) == len(explicit) == fx.queries.n_conformers
        for (ds, dg), (es, eg) in zip(dense, explicit):
            assert len(dg) > 0
            np.testing.assert_array_equal(dg, eg)
            np.testing.assert_array_equal(_bits(ds), _bits(es))


class RepeatSource:
    """Launches far wider than one work item: each fixture group's conformers,
    every row repeated ``copies`` times in place (so molecules stay contiguous
    runs), as one block per group. Several work items per query, several
    queries per launch -- what the dense index arithmetic has to get right."""
    prefetch_depth = 0
    producer_wait_s = 0.

    def __init__(self, fx, copies):
        from roshambo3.layout import tile32
        self.blocks_ = []
        base = 0
        for group in fx.rows.groups:
            merged = np.repeat(group.merged, copies, axis=0)
            n = merged.shape[0]
            self.blocks_.append(dict(
                coords=tile32(merged.transpose(0, 2, 1)),
                counts=np.ascontiguousarray(np.repeat(group.counts, copies, axis=0)),
                n_conformers=n, L=group.L, bucket=group.bucket, gid_base=base,
                L_shape=group.L_field_width,
                mol_id=np.repeat(group.mol_id, copies).astype(np.int64),
                color_offset=-1, tail_molecule_open=False))
            base += n
        self.n_conformers = base
        self.max_block_conformers = max(b["n_conformers"] for b in self.blocks_)
        self.max_block_elements = max(b["coords"].size for b in self.blocks_)

    def blocks(self):
        from roshambo3.engine.source import ConformerBlock
        for b in self.blocks_:
            yield ConformerBlock(**b)


@pytest.mark.parametrize("copies", [37, 211])
@pytest.mark.parametrize("pose_search", [PoseSearch.FIELD, PoseSearch.ANALYTIC],
                         ids=["field", "analytic"])
@pytest.mark.parametrize("mode", list(MODES))
def test_wide_launches_are_bit_identical(prepared_gpu_corpus, mode, pose_search, copies):
    fx = prepared_gpu_corpus
    config = _config(fx, mode, pose_search)
    src = RepeatSource(fx, copies)
    assert src.max_block_conformers > 256    # > one work item per query
    for sink in (lambda: DenseScores(max_bytes=1 << 28), lambda: TopK(k=64),
                 lambda: MoleculeTopK(k=16)):
        with closing(Engine()) as engine:
            dense = engine.run(src, fx.queries, config, sink())[1]
        with closing(Engine(backend=_explicit_backend())) as engine:
            explicit = engine.run(src, fx.queries, config, sink())[1]
        if isinstance(dense, list):
            for (ds, dg), (es, eg) in zip(dense, explicit, strict=True):
                np.testing.assert_array_equal(dg, eg)
                np.testing.assert_array_equal(_bits(ds), _bits(es))
        else:
            np.testing.assert_array_equal(dense.gids, explicit.gids)
            np.testing.assert_array_equal(_bits(dense.scores), _bits(explicit.scores))


def _one_launch(fx, mode="shape"):
    """The arguments of one real engine launch, captured with its explicit
    pair list, and that launch's dense score block."""
    config = ScreenConfig(**MODES[mode])
    calls = []
    _screen(fx, config, DenseScores(max_bytes=1 << 26), _explicit_backend(calls))
    kw = max(calls, key=lambda c: len(c["pair_query_row"]))
    kw = dict(kw, out=None, topk=None)
    kw.pop("max_rows_per_query", None)
    dense = cuda.score_pairs(**dict(kw, pair_query_row=None, pair_library_row=None))
    return kw, dense


class TestTheSparsePathIsIntact:
    """No production caller sends a sparse list; these keep it honest."""

    @pytest.fixture
    def launch(self, prepared_gpu_corpus):
        return _one_launch(prepared_gpu_corpus,
                           "rescore" if prepared_gpu_corpus.generator else "shape")

    def _subset(self, kw, seed):
        n_q = int(kw["query_coords"].shape[0])
        n = int(kw["library_counts"].shape[0])
        rng = np.random.default_rng(seed)
        rows = rng.choice(n_q * n, size=max(1, (n_q * n) // 3), replace=False)
        rng.shuffle(rows)                    # not query-major, not sorted
        return rows, (rows // n).astype(np.int32), (rows % n).astype(np.int32)

    def test_a_shuffled_subset_scores_like_the_dense_rows(self, launch):
        kw, dense = launch
        rows, pq, pt = self._subset(kw, 313)
        sparse = cuda.score_pairs(**dict(kw, pair_query_row=pq, pair_library_row=pt))
        assert sparse.shape == (len(rows), N_SCORE_COLUMNS)
        np.testing.assert_array_equal(_bits(sparse), _bits(dense[rows]))

    def test_a_shuffled_subset_through_the_device_topk(self, launch):
        kw, dense = launch
        rows, pq, pt = self._subset(kw, 314)
        n_q = int(kw["query_coords"].shape[0])
        gid_base = 1000
        k = 7
        tk = cuda.extension().DeviceTopK(n_q, k, 0, N_SCORE_COLUMNS, True, 0)
        cuda.score_pairs(**dict(kw, pair_query_row=pq, pair_library_row=pt,
                                topk=tk, gid_base=gid_base))
        got = tk.result(0)
        assert len(got) == n_q
        for q, (scores, gids) in enumerate(got):
            mine = np.flatnonzero(pq == q)
            sub = dense[rows[mine]]
            order = rank_order(sub[:, 0], gid_base + pt[mine].astype(np.int64))[:k]
            np.testing.assert_array_equal(gids, gid_base + pt[mine][order])
            np.testing.assert_array_equal(_bits(scores), _bits(sub[order]))

    def test_the_molecule_topk_still_refuses_a_non_dense_list(self, launch):
        kw, _ = launch
        n = int(kw["library_counts"].shape[0])
        n_q = int(kw["query_coords"].shape[0])
        # Every pair, but not in query-major order: only the verification
        # that a dense launch skips can refuse it.
        rows = np.random.default_rng(315).permutation(n_q * n)
        pq, pt = (rows // n).astype(np.int32), (rows % n).astype(np.int32)
        sink = MoleculeTopK(k=5)
        sink.prepare(n_q, 0, n)
        sink.before_block(types.SimpleNamespace(
            mol_id=np.arange(n, dtype=np.int64), n_conformers=n,
            gid_base=0, tail_molecule_open=False))
        with pytest.raises(RuntimeError, match="dense query-major pair block"):
            cuda.score_pairs(**dict(kw, pair_query_row=pq, pair_library_row=pt,
                                    topk=sink))

    def test_half_a_pair_list_is_refused(self, launch):
        kw, _ = launch
        with pytest.raises(ValueError, match="or neither"):
            cuda.score_pairs(**dict(kw, pair_library_row=None))

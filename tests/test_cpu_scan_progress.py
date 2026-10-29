"""Completed CPU block publication preserves results and failure cleanup."""
from dataclasses import replace
import threading
import time
import numpy as np
import pytest
from roshambo3.engine import CPUWorkerOptions, Engine, MoleculeTopK
from tests.codec_fixtures import FEATURES_ENUM, make_codec_fixture
from tests.test_cpu_worker_public_optin import _config
from tests.test_cpu_worker_owned_scan import _query_set, _source


def inputs(tmp_path):
    fx = make_codec_fixture(tmp_path / 'corpus', color_layout='merged')
    return fx, _query_set(fx, (0, 4)), replace(_config(), cpu_worker=CPUWorkerOptions(workers=2))


def test_completed_progress_serialized_and_bitwise_neutral(tmp_path):
    fx, queries, config = inputs(tmp_path)
    engine = Engine(backend='cpu')
    _, expected = engine.run(_source(fx.corpus), queries, config, MoleculeTopK(3))
    snapshots = []
    active = threading.Lock()
    def progress(stats):
        assert active.acquire(blocking=False), 'concurrent progress callback'
        try:
            snapshots.append(stats)
            time.sleep(.001)  # Release the GIL so another worker can finish.
        finally:
            active.release()
    stats, actual = engine.run(_source(fx.corpus), queries, config, MoleculeTopK(3), progress=progress)
    assert snapshots and snapshots[-1].n_conformers == stats.n_conformers
    assert [s.n_blocks for s in snapshots] == list(range(1, stats.n_blocks + 1))
    assert all(a.n_conformers < b.n_conformers for a, b in zip(snapshots, snapshots[1:]))
    assert all(s.n_rows == queries.n_conformers * s.n_conformers for s in snapshots)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
    engine.close()


def test_progress_failure_and_cancellation_allow_reuse(tmp_path):
    fx, queries, config = inputs(tmp_path)
    engine = Engine(backend='cpu')
    def failed(stats):
        raise RuntimeError('progress failed')
    with pytest.raises(RuntimeError, match='progress failed'):
        engine.run(_source(fx.corpus), queries, config, MoleculeTopK(3), progress=failed)
    assert engine._worker_query_cache is None and engine._worker_self_cache is None
    canceled = threading.Event()
    with pytest.raises(InterruptedError):
        engine.run(_source(fx.corpus), queries, config, MoleculeTopK(3),
                   progress=lambda stats: canceled.set(), canceled=canceled.is_set)
    stats, _ = engine.run(_source(fx.corpus), queries, config, MoleculeTopK(3))
    assert stats.n_conformers == fx.corpus.n_conformers
    engine.close()

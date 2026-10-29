"""CPU worker pools: one pinned CPU per worker, and no thread outlives its scan.

Product checks that need no dedicated many-CPU harness or fresh
interpreter: they run on as few as two CPUs, in an ordinary pytest
process: each check looks only at the threads the pool itself reported.
"""
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from roshambo3.backend import cpu

cpu.extension()  # Missing native code is a test failure, never a green skip.

linux_threads = pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or not Path("/proc/self/task").is_dir(),
    reason="reads Linux CPU affinity and /proc/self/task")


def _wait_gone(tids, timeout=0.5):
    """The TIDs in `tids` still present after `timeout` seconds.

    A joined thread is gone at once; the grace covers only the kernel's
    reaping, not a pool that detached its workers instead of joining them.
    """
    deadline = time.monotonic() + timeout
    while True:
        alive = {int(p.name) for p in Path("/proc/self/task").iterdir()} & set(tids)
        if not alive or time.monotonic() > deadline:
            return alive
        time.sleep(0.01)


def _allowed_cpus():
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 2, "worker-pool coverage requires two allowed CPUs"
    return cpus


@linux_threads
@pytest.mark.parametrize("strict", [False, True],
                         ids=["engine-default", "strict-affinity"])
def test_each_worker_is_pinned_to_its_own_allowed_cpu_and_gone_after_close(strict):
    """Strict mode makes the constructor verify the binding itself; the
    engine uses the default, so the binding must hold there too."""
    allowed = _allowed_cpus()
    n = min(4, len(allowed))
    seen = {}
    barrier = threading.Barrier(n)

    def callback(worker, item):
        tid = threading.get_native_id()
        seen[tid] = (worker, tuple(sorted(os.sched_getaffinity(tid))))
        barrier.wait(timeout=10)   # every worker takes exactly one item

    with cpu.CPUComboExecutor(n, claim_chunk_rows=128,
                              require_worker_affinity=strict) as executor:
        executor.run_indexed(n, callback, rows_per_item=128)
    assert len(seen) == n
    masks = sorted(mask for _, mask in seen.values())
    assert all(len(mask) == 1 for mask in masks), masks
    assert masks == [(c,) for c in allowed[:n]], (masks, allowed[:n])
    assert not _wait_gone(seen), "an executor worker outlived close()"


@linux_threads
def test_a_worker_scan_leaves_no_scoring_thread_behind(tmp_path, monkeypatch):
    """The engine's worker route closes its pool before `run` returns, so the
    next scan starts from fresh threads rather than inheriting stale ones."""
    from roshambo3.engine import CPUWorkerOptions, Engine, TopK
    from tests.codec_fixtures import FEATURES_ENUM, make_codec_fixture
    from tests.test_cpu_worker_public_optin import _config
    from tests.test_cpu_worker_owned_scan import _query_set, _source
    _allowed_cpus()
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    queries = _query_set(fx)
    config = _config(field_spacing_a=.5)
    config.cpu_worker = CPUWorkerOptions(workers=2)
    scoring = {}
    original = cpu.score_pairs

    def recording(*args, **kwargs):
        tid = threading.get_native_id()
        scoring[tid] = tuple(sorted(os.sched_getaffinity(tid)))
        return original(*args, **kwargs)

    monkeypatch.setattr(cpu, "score_pairs", recording)
    previous = set()
    for _ in range(2):
        scoring.clear()
        engine = Engine(backend=cpu)
        _, result = engine.run(_source(fx.corpus), queries, config, TopK(2))
        engine.close()
        assert scoring and np.isfinite(result[0][0]).all()
        assert threading.get_native_id() not in scoring, "scored on the caller"
        # Each scoring thread is pinned to one CPU of the caller's mask, and
        # no two share one.
        masks = list(scoring.values())
        assert all(len(m) == 1 for m in masks), masks
        assert len(set(masks)) == len(masks), masks
        assert set(m[0] for m in masks) <= set(os.sched_getaffinity(0))
        assert not _wait_gone(scoring), "a scoring thread outlived its scan"
        assert set(scoring).isdisjoint(previous)
        previous = set(scoring)

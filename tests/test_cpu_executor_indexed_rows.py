"""Focused tests for the persistent indexed CPU worker seam."""

import threading

import numpy as np
import pytest

from roshambo3.backend.cpu import CPUComboExecutor, extension


@pytest.mark.parametrize("workers", [1, 2])
def test_indexed_executor_claims_each_item_once(workers):
    caller_thread = threading.get_ident()
    claimed = []
    with CPUComboExecutor(workers, claim_chunk_rows=128) as executor:
        executor.run_indexed(
            17,
            lambda worker, item: claimed.append(
                (worker, item, threading.get_ident())),
            rows_per_item=128)
        assert sorted(item for _, item, _ in claimed) == list(range(17))
        assert all(worker < workers for worker, _, _ in claimed)
        assert all(thread_id != caller_thread for _, _, thread_id in claimed)
        assert executor.stats["work_items_completed_by_worker"]
        assert sum(executor.stats["work_items_completed_by_worker"]) == 17


def test_indexed_executor_propagates_failure_and_can_be_reused():
    with CPUComboExecutor(2, claim_chunk_rows=128) as executor:
        def fail_on_first(worker, item):
            if item == 0:
                raise ValueError("fixture callback failure")

        with pytest.raises(RuntimeError, match="Python combo work callback failed"):
            executor.run_indexed(100, fail_on_first, rows_per_item=128)

        completed = []
        executor.run_indexed(
            3, lambda worker, item: completed.append(item),
            rows_per_item=128)
        assert sorted(completed) == [0, 1, 2]


def test_direct_id_topk_sink_has_no_scan_sized_id_map():
    sink = extension().make_worker_local_topk_direct_row_sink(
        worker_count=2, query_count=2, k=5, rank_column=0,
        level="conformer", library_row_count=1_000_000,
        has_molecule_ids=True)
    assert sink.memory_estimate["id_map_preparation_peak_bytes"] == 0
    sink.mark_scoring_succeeded()
    result = sink.finalize()
    assert len(result) == 2
    for scores, gids, mids in result:
        assert scores.shape == (0, 20)
        assert gids.shape == (0,)
        assert mids.shape == (0,)
        assert scores.dtype == np.float32
        assert gids.dtype == np.int64
        assert mids.dtype == np.int64

"""Explicitly owned persistent workers and session lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from roshambo3.backend import cpu
from cpu_test_capabilities import IS_ARM, RETAINED_COLOR_GRAIN
from roshambo3.reference.color import (
    MODE_COMBO_OPTIMAL,
    MODE_SHAPE_POSE_COLOR_RESCORE,
)

from test_cpu_color_start_lanes import _population
from test_cpu_combo_threads import _combo_options


pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")


@pytest.mark.parametrize("color_grain", [11, 16])
def test_explicit_executor_reuses_workers_and_close_is_idempotent(color_grain):
    # More than one minimum-size claim exercises dynamic coarse scheduling and
    # its tail while also keeping enough heterogeneous feature counts.
    kw = _population(129, "general")
    options = _combo_options()
    options["color_grain"] = color_grain
    expected = cpu.score_pairs(threads=1, **options, **kw)
    executor = cpu.CPUComboExecutor(2, claim_chunk_rows=128)
    stamp = cpu.combo_thread_config_stamp(
        threads=2, color_grain=color_grain, start_mode=1,
        n_pairs=len(kw["pair_query_row"]),
        claim_chunk_rows=executor.claim_chunk_rows)
    assert stamp["schedule"] == "dynamic_coarse_claims"
    assert stamp["claim_chunk_rows"] == executor.claim_chunk_rows
    assert stamp["claim_count"] == 2
    assert stamp["worker_parallelism_limit"] == 2

    first = cpu.score_pairs(threads=2, executor=executor, **options, **kw)
    before_second = executor.stats
    second = cpu.score_pairs(threads=2, executor=executor, **options, **kw)
    after_second = executor.stats

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert before_second["submissions"] > 0
    assert after_second["submissions"] > before_second["submissions"]
    assert sum(after_second["work_items_completed_by_worker"]) == (
        after_second["work_items_completed"])
    assert sum(after_second["claims_by_worker"]) >= 4
    assert len(after_second["claims_by_worker"]) == 2

    executor.close()
    executor.close()
    assert executor.closed
    with pytest.raises(RuntimeError, match="closed"):
        cpu.score_pairs(threads=2, executor=executor, **options, **kw)


def test_persistent_executor_reports_actual_scoring_claim_workers():
    kw = _population(1, "general")
    options = _combo_options()
    expected = cpu.score_pairs(threads=1, **options, **kw)

    with cpu.CPUComboExecutor(2) as executor:
        scores, diagnostic = cpu.score_pairs(
            threads=2, executor=executor, with_worker_timers=True,
            **options, **kw)
        claims = diagnostic["scoring_claims_by_worker"]
        assert len(claims) == executor.threads
        assert sum(claims) == 1
        assert diagnostic["workers_with_claims"] == sum(
            count > 0 for count in claims)
        assert diagnostic["workers_with_claims"] == 1
        assert executor.stats["last_claims_by_worker"] == claims

    np.testing.assert_array_equal(scores, expected)


@pytest.mark.parametrize("color_grain", [11, 16])
def test_four_workers_preserve_sparse_repeated_out_of_order_rows_and_tails(
        color_grain):
    # Five 128-row claims leave a non-full final SIMD grain. The library map
    # touches only 97/128 source rows, visits them out of order and repeatedly,
    # and the last target is intentionally much denser than the others.
    n_pairs, n_library_rows = 515, 128
    kw = _population(n_library_rows, "general")
    kw["pair_query_row"] = ((np.arange(n_pairs, dtype=np.int32) * 2 + 1) % 3)
    kw["pair_library_row"] = (
        (np.arange(n_pairs, dtype=np.int32) * 37 + 11) % 97)
    kw["library_counts"][:, 1:] = 0
    kw["library_counts"][96, 3] = 54
    kw["library_coords"][:, 2:, :3] = 0.0125
    kw["library_coords"][:, 2:, 3] = 0.0
    kw["library_coords"][96, 2:, :3] = np.arange(
        54 * 3, dtype=np.float32).reshape(54, 3) * 0.001
    kw["library_coords"][96, 2:, 3] = 1.0

    options = _combo_options()
    options.update(color_grain=color_grain, steps=1)
    expected = cpu.score_pairs(threads=1, **options, **kw)
    with cpu.CPUComboExecutor(4, claim_chunk_rows=128) as executor:
        actual, diagnostic = cpu.score_pairs(
            threads=4, executor=executor, with_worker_timers=True,
            **options, **kw)
        claims = diagnostic["scoring_claims_by_worker"]
        stamp = cpu.combo_thread_config_stamp(
            threads=4, color_grain=color_grain, start_mode=1,
            n_pairs=n_pairs, claim_chunk_rows=128)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (n_pairs, 20)
    assert stamp["claim_count"] == 5
    assert stamp["worker_parallelism_limit"] == 4
    assert len(claims) == 4
    assert sum(claims) == stamp["claim_count"]
    assert diagnostic["workers_with_claims"] == sum(
        claim_count > 0 for claim_count in claims)


@pytest.mark.parametrize("color_grain", [11, 16])
def test_empty_batch_is_safe_with_persistent_workers(color_grain):
    kw = _population(0, "general")
    options = _combo_options()
    options["color_grain"] = color_grain
    expected = cpu.score_pairs(threads=1, **options, **kw)
    with cpu.CPUComboExecutor(4, claim_chunk_rows=128) as executor:
        actual, diagnostic = cpu.score_pairs(
            threads=4, executor=executor, with_worker_timers=True,
            **options, **kw)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (0, 20)
    assert diagnostic["scoring_claims_by_worker"] == [0, 0, 0, 0]
    assert diagnostic["workers_with_claims"] == 0


def test_calls_sharing_one_executor_serialize_without_state_leakage():
    kw = _population(17, "general")
    options = _combo_options()
    expected = cpu.score_pairs(threads=1, **options, **kw)

    with cpu.CPUComboExecutor(2) as executor:
        with ThreadPoolExecutor(max_workers=3) as callers:
            results = list(callers.map(
                lambda _: cpu.score_pairs(
                    threads=2, executor=executor, **options, **kw), range(3)))
        assert executor.stats["submissions"] > 3

    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_close_waits_for_an_active_score_call_without_deadlock(monkeypatch):
    kw = _population(1, "general")
    options = _combo_options()
    expected = cpu.score_pairs(threads=1, **options, **kw)
    native = cpu.extension()
    original = native.optimize_scalar_combo
    entered = Event()
    proceed = Event()

    def pause_before_native_call(*args):
        entered.set()
        if not proceed.wait(timeout=10):
            raise RuntimeError("test did not release the scoring call")
        return original(*args)

    monkeypatch.setattr(native, "optimize_scalar_combo", pause_before_native_call)
    executor = cpu.CPUComboExecutor(2)
    with ThreadPoolExecutor(max_workers=2) as callers:
        score = callers.submit(
            cpu.score_pairs, threads=2, executor=executor,
            **options, **kw)
        close = None
        try:
            assert entered.wait(timeout=5)
            close = callers.submit(executor.close)
            assert not close.done()
        finally:
            proceed.set()
        np.testing.assert_array_equal(score.result(timeout=30), expected)
        assert close is not None
        close.result(timeout=30)
    assert executor.closed


@pytest.mark.skipif(IS_ARM, reason="trajectory shape failure injection requires AVX2")
def test_worker_failure_cancels_submission_and_executor_can_be_reused():
    kw = _population(9, "general", shared_queries=False)
    prepared = cpu.prepare_combo_shape_evaluators(
        kw["query_coords"], kw["query_n_atoms"],
        shape_backend="analytic_simd", shape_grain="trajectory",
        shape_width=8)
    failing_options = _combo_options(
        shape_backend="analytic_simd", shape_grain="trajectory",
        shape_width=8, prepared=prepared)
    good_options = _combo_options()
    expected = cpu.score_pairs(threads=1, **good_options, **kw)

    with cpu.CPUComboExecutor(2) as executor:
        with pytest.raises(ValueError, match="one shared query"):
            cpu.score_pairs(threads=2, executor=executor,
                            **failing_options, **kw)
        recovered = cpu.score_pairs(
            threads=2, executor=executor, **good_options, **kw)

    np.testing.assert_array_equal(recovered, expected)


@pytest.mark.parametrize("claim_chunk_rows", [127, 528])
def test_executor_rejects_unbounded_or_unaligned_claim_chunks(claim_chunk_rows):
    with pytest.raises(ValueError, match="claim_chunk_rows"):
        cpu.CPUComboExecutor(1, claim_chunk_rows=claim_chunk_rows)


def test_executor_is_mode3_only_and_worker_count_must_match():
    kw = _population(1, "general")
    options = dict(
        color=True, color_mode=MODE_COMBO_OPTIMAL, color_grain=RETAINED_COLOR_GRAIN,
        mixing=0.45, steps=1, start_mode=1)
    with cpu.CPUComboExecutor(2) as executor:
        with pytest.raises(ValueError, match="match the CPUComboExecutor"):
            cpu.score_pairs(threads=1, executor=executor, **options, **kw)
        mode2_options = dict(options)
        mode2_options["color_mode"] = MODE_SHAPE_POSE_COLOR_RESCORE
        with pytest.raises(ValueError, match="only for mode-3"):
            cpu.score_pairs(threads=2, executor=executor,
                            **mode2_options, **kw)

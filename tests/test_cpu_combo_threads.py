"""Independent native workers for complete CPU combo alignments."""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pytest

from roshambo3.backend import cpu
from cpu_test_capabilities import IS_ARM, RETAINED_COLOR_GRAIN
from roshambo3.reference.color import MODE_COMBO_OPTIMAL

from test_cpu_color_start_lanes import _population


pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")


def _combo_options(*, shape_backend="scalar_analytic", shape_grain="scalar",
                   shape_width=1, prepared=None):
    return dict(
        color=True,
        color_mode=MODE_COMBO_OPTIMAL,
        color_grain=RETAINED_COLOR_GRAIN,
        mixing=0.45,
        steps=2,
        start_mode=1,
        shape_backend=shape_backend,
        shape_grain=shape_grain,
        shape_width=shape_width,
        shape_field_spacing_angstrom=(
            0.5 if shape_backend == "pipelined_field" else None),
        prepared_shape_evaluators=prepared,
    )


def _allowed_cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@pytest.mark.skipif(IS_ARM, reason="trajectory shape grain requires AVX2")
# The stamp below requires worker affinity, which refuses more workers than the
# process mask holds; small CI runners have two.
@pytest.mark.skipif(_allowed_cpus() < 4, reason="needs four CPUs in the affinity mask")
def test_parallel_trajectory_shape_keeps_odd_batch_rows_exact_and_ordered():
    kw = _population(33, "general", shared_queries=True)
    kw["pair_library_row"] = np.arange(32, -1, -1, dtype=np.int32)
    prepared = cpu.prepare_combo_shape_evaluators(
        kw["query_coords"], kw["query_n_atoms"], shape_backend="analytic_simd",
        shape_grain="trajectory", shape_width=8)
    options = _combo_options(shape_backend="analytic_simd",
                             shape_grain="trajectory", shape_width=8,
                             prepared=prepared)

    serial = cpu.score_pairs(threads=1, **options, **kw)
    parallel = cpu.score_pairs(threads=4, **options, **kw)

    np.testing.assert_array_equal(parallel, serial)
    stamp = cpu.combo_thread_config_stamp(
        threads=4, color_grain=RETAINED_COLOR_GRAIN, start_mode=1,
        n_pairs=len(kw["pair_query_row"]), require_worker_affinity=True)
    assert stamp["threads_requested"] == 4
    assert stamp["workers_used"] == 4
    assert stamp["work_units"] == 5
    assert stamp["work_unit"] == "eight_pair_color_block"
    assert stamp["schedule"] == "dynamic_coarse_claims"
    assert stamp["claim_chunk_rows"] == 256
    assert stamp["claim_count"] == 1
    assert stamp["worker_parallelism_limit"] == 1
    assert stamp["workers_used"] == 4
    assert "not observed workers" in stamp["workers_used_semantics"]
    assert stamp["worker_affinity_required"]


@pytest.mark.parametrize("color_grain", [11, 16])
def test_parallel_batch_families_preserve_exact_rows(color_grain):
    stamp = cpu.color_build_stamp(color_grain)
    if (not stamp.get("color_target_compiled") or
            not stamp.get("color_runtime_available")):
        pytest.skip(f"color grain {color_grain} is not available on this CPU build")
    kw = _population(9, "general")
    kw["pair_library_row"] = np.arange(8, -1, -1, dtype=np.int32)
    options = _combo_options()
    options["color_grain"] = color_grain

    serial = cpu.score_pairs(threads=1, **options, **kw)
    parallel = cpu.score_pairs(threads=3, **options, **kw)
    np.testing.assert_array_equal(parallel, serial)


def test_parallel_pipelined_field_reuses_one_prepared_query_set():
    kw = _population(17, "general")
    kw["pair_library_row"] = np.arange(16, -1, -1, dtype=np.int32)
    prepared = cpu.prepare_combo_shape_evaluators(
        kw["query_coords"], kw["query_n_atoms"],
        shape_backend="pipelined_field", shape_grain="library_atom",
        shape_width=8, field_spacing_angstrom=0.5,
        field_padding_angstrom=8.0)
    options = _combo_options(shape_backend="pipelined_field",
                             shape_grain="library_atom", shape_width=8,
                             prepared=prepared)

    serial = cpu.score_pairs(threads=1, **options, **kw)
    parallel = cpu.score_pairs(threads=3, **options, **kw)

    np.testing.assert_array_equal(parallel, serial)
    stamp = cpu.combo_thread_config_stamp(
        threads=3, color_grain=RETAINED_COLOR_GRAIN, start_mode=1, n_pairs=len(kw["pair_query_row"]))
    assert stamp["workers_used"] == 3
    assert stamp["work_units"] == (9 if IS_ARM else 3)


def test_parallel_start_ties_keep_start_zero_and_diagnostic_calls_refuse():
    kw = _population(9, "general")
    kw["query_n_atoms"][:] = 0
    kw["query_types"][:] = 0
    kw["library_counts"][:] = 0
    kw["library_coords"][:] = 0.0
    options = dict(
        color=True,
        color_mode=MODE_COMBO_OPTIMAL,
        color_grain=RETAINED_COLOR_GRAIN,
        mixing=0.45,
        steps=1,
        lr_q=0.0,
        lr_t=0.0,
        budget_override="T186 fixed-pose tie fixture",
        start_mode=1,
    )

    serial, census = cpu.score_pairs(
        threads=1, with_census=True, **options, **kw)
    parallel = cpu.score_pairs(threads=2, **options, **kw)
    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(census["winning_starts"], np.zeros(9, np.int32))
    with pytest.raises(ValueError, match="do not support census"):
        cpu.score_pairs(threads=2, with_census=True, **options, **kw)


def test_parallel_worker_timer_is_separate_and_does_not_change_scores():
    kw = _population(17, "general")
    options = _combo_options()
    expected = cpu.score_pairs(threads=1, **options, **kw)
    affinity_supported = cpu.extension().cpu_platform_info()["affinity_supported"]
    scores, diagnostic = cpu.score_pairs(
        threads=2, with_worker_timers=True, require_worker_affinity=affinity_supported,
        **options, **kw)

    np.testing.assert_array_equal(scores, expected)
    elapsed = np.asarray(diagnostic["worker_elapsed_seconds"], dtype=np.float64)
    assert elapsed.shape == (2,)
    assert np.all(np.isfinite(elapsed))
    assert np.all(elapsed >= 0.0)
    claims = diagnostic["scoring_claims_by_worker"]
    assert len(claims) == 2
    assert sum(claims) == 1
    assert diagnostic["workers_with_claims"] == 1
    # An idle worker can finish within one clock tick on macOS. A worker that
    # actually scored its claim must still report positive elapsed time.
    assert np.all(elapsed[np.asarray(claims) > 0] > 0.0)
    placements = diagnostic["worker_affinity_observed"]
    assert len(placements) == 2
    for worker in placements:
        assert worker["affinity_attempted"] == affinity_supported
        assert isinstance(worker["affinity_succeeded"], bool)
        assert bool(worker["actual_cpus"]) == affinity_supported
    with pytest.raises(ValueError, match="separate diagnostic"):
        cpu.score_pairs(threads=2, with_worker_timers=True,
                        with_census=True, **options, **kw)
    with pytest.raises(ValueError, match="requires threads >= 2"):
        cpu.score_pairs(threads=1, require_worker_affinity=True,
                        **options, **kw)


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_threads_requires_a_positive_integral_worker_count(threads):
    kw = _population(1, "general")
    with pytest.raises(ValueError, match="threads must"):
        cpu.score_pairs(threads=threads, **_combo_options(), **kw)


@pytest.mark.skipif(IS_ARM, reason="trajectory shape grain requires AVX2")
def test_parallel_trajectory_guard_propagates_worker_exceptions():
    kw = _population(9, "general", shared_queries=False)
    prepared = cpu.prepare_combo_shape_evaluators(
        kw["query_coords"], kw["query_n_atoms"], shape_backend="analytic_simd",
        shape_grain="trajectory", shape_width=8)
    with pytest.raises(ValueError, match="one shared query"):
        cpu.score_pairs(
            threads=3,
            **_combo_options(shape_backend="analytic_simd",
                             shape_grain="trajectory", shape_width=8,
                             prepared=prepared),
            **kw)


def test_repeated_concurrent_combo_calls_do_not_share_worker_state():
    kw = _population(17, "general")
    options = _combo_options()
    expected = cpu.score_pairs(threads=1, **options, **kw)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(
            lambda _: cpu.score_pairs(threads=2, **options, **kw), range(3)))
    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_concurrent_calls_share_prepared_field_handle_read_only():
    kw = _population(17, "general")
    prepared = cpu.prepare_combo_shape_evaluators(
        kw["query_coords"], kw["query_n_atoms"],
        shape_backend="pipelined_field", shape_grain="library_atom",
        shape_width=8, field_spacing_angstrom=0.5,
        field_padding_angstrom=8.0)
    options = _combo_options(shape_backend="pipelined_field",
                             shape_grain="library_atom", shape_width=8,
                             prepared=prepared)
    options["shape_field_spacing_angstrom"] = 0.5
    expected = cpu.score_pairs(threads=1, **options, **kw)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: cpu.score_pairs(threads=2, **options, **kw), range(2)))
    for result in results:
        np.testing.assert_array_equal(result, expected)

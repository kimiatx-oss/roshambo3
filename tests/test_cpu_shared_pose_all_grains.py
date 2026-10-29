"""End-to-end coverage for default shared-pose adoption by color grain."""

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.reference.color import MODE_COMBO_OPTIMAL
from cpu_test_capabilities import RETAINED_COLOR_GRAIN


pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")


def _inputs(*, canonical=False):
    q = np.zeros((2, 4, 4), dtype=np.float32)
    q[0, 0] = (0.0, 0.0, 0.0, 1.0)
    q[0, 1] = (0.5, 0.1, 0.0, 1.0)
    q[0, 2] = (-0.2, 0.4, 0.1, 1.0)
    q[1, 0] = (0.1, -0.1, 0.0, 1.0)
    q[1, 1] = (0.3, 0.2, -0.1, 1.0)
    q[1, 2] = (9.0, 9.0, 9.0, 1.0)  # padding, not a second color feature
    qt = np.array([[0, 1, 2, 0], [0, 1, 0, 0]], dtype=np.int32)
    qn = np.array([1, 1], dtype=np.int32)

    library = np.zeros((3, 3, 4), dtype=np.float32)
    library[0, 0, :3] = (0.0, 0.0, 0.0)
    library[0, 1, :3] = (0.4, -0.1, 0.0)
    library[0, 2, :3] = (-0.1, 0.5, 0.2)
    library[1, 0, :3] = (0.2, 0.1, -0.1)
    library[1, 1, :3] = (0.1, 0.2, 0.3)
    library[1, 2, :3] = (-0.3, 0.6, -0.2)
    library[2, 0, :3] = (-0.2, 0.1, 0.1)
    library[2, 1, :3] = (0.7, -0.2, 0.2)
    library[2, 2, :3] = (4.0, 4.0, 4.0)  # masked absent feature
    counts = np.array([[1, 1, 1], [1, 2, 0], [1, 1, 0]], dtype=np.uint8)

    widths = np.zeros((3, 3), dtype=np.float32)
    heights = np.zeros((3, 3), dtype=np.float32)
    if canonical:
        widths[1:, 1:] = 1.0
        heights[1, 1] = heights[2, 2] = 1.0
    else:
        widths[1:, 1:] = [[0.8, 1.2], [1.5, 0.7]]
        heights[1:, 1:] = [[1.0, 0.7], [0.4, 0.9]]
    return {
        "query_coords": q,
        "query_types": qt,
        "query_n_atoms": qn,
        "library_coords": library,
        "library_counts": counts,
        "pair_query_row": np.array([0, 1, 0], dtype=np.int32),
        "pair_library_row": np.array([0, 1, 2], dtype=np.int32),
        "interaction_r": widths,
        "interaction_p": heights,
    }


def _prepare(data, *, shape_evaluators=None):
    return cpu.prepare_combo_shared_pose_queries(
        data["query_coords"], data["query_types"], data["query_n_atoms"],
        data["interaction_r"], data["interaction_p"],
        shape_backend="scalar_analytic", shape_grain="scalar", shape_width=1,
        shape_evaluators=shape_evaluators)


def _score(data, *, grain, start_mode=1, prepared=None, threads=1,
           with_census=True, with_worker_timers=False,
           color_field_backends=None, prepared_color_fields=None,
           prepared_shape_evaluators=None):
    options = dict(
        **data,
        color=True,
        color_mode=MODE_COMBO_OPTIMAL,
        color_grain=grain,
        steps=2,
        start_mode=start_mode,
        mixing=0.5,
        threads=threads,
        with_census=with_census,
        with_worker_timers=with_worker_timers,
    )
    if color_field_backends is not None:
        options["color_field_backends"] = color_field_backends
    if prepared_color_fields is not None:
        options["prepared_color_fields"] = prepared_color_fields
    if prepared_shape_evaluators is not None:
        options["prepared_shape_evaluators"] = prepared_shape_evaluators
    if prepared is not None:
        options["pose_contract"] = "shared_fp32_reduced_v1"
        options["prepared_shared_pose_queries"] = prepared
    return cpu.score_pairs(**options)


def _assert_bytes_equal(left, right, label):
    a = np.ascontiguousarray(np.asarray(left))
    b = np.ascontiguousarray(np.asarray(right))
    assert a.shape == b.shape, f"{label} shapes differ: {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"{label} dtypes differ: {a.dtype} vs {b.dtype}"
    assert a.tobytes() == b.tobytes(), f"{label} bytes differ"


def _assert_common_census(data, stats, grain, start_mode):
    starts = {0: 1, 1: 4, 2: 10}[start_mode]
    n_pairs = int(data["pair_query_row"].size)
    n_steps = 2
    assert stats["pose_contract"] == "shared_fp32_reduced_v1"
    assert stats["color_grain"] == grain
    assert stats["shared_pose_preparations"] == n_pairs * starts * (n_steps + 1)
    assert stats["color_gradient_evaluations"] == n_pairs * starts * n_steps
    assert stats["color_per_start_final_evaluations"] == n_pairs * starts
    assert stats["color_winner_rescore_evaluations"] == 0
    from cpu_test_capabilities import IS_ARM
    if stats.get("color_field_prepared_handle_used", False) and not IS_ARM:
        assert (stats["color_field_interpolated_terms"] +
                stats["color_field_analytic_selected_terms"] +
                stats["color_field_analytic_fallback_terms"]) > 0
    else:
        assert stats["color_gradient_terms"] > 0
    assert stats["color_per_start_final_terms"] > 0
    assert stats["color_winner_rescore_terms"] == 0
    assert np.asarray(stats["winning_starts"]).shape == (n_pairs,)

    # The three-pair input intentionally leaves partial native blocks in the
    # conformer8 and paired-start families. Row 1 has an absent
    # type-2 target segment and query row 1 lacks type 2; padded sentinels must
    # remain masked under every requested grain.
    assert data["library_counts"][2, 2] == 0
    assert data["query_types"][1, 2] == 0

    if grain == 11:
        assert (stats["color_conformer_vector_evaluations"] > 0 or
                stats["color_conformer_scalar_fallback_pairs"] > 0)
    elif grain == 16 and start_mode == 1:
        assert (stats["color_start_lane_vector_evaluations"] > 0 or
                stats["color_start_lane_scalar_fallback_pairs"] > 0)


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("grain", [0, 11, 16])
def test_default_and_explicit_shared_pose_are_byte_identical_for_every_grain(
        canonical, grain):
    data = _inputs(canonical=canonical)
    prepared = _prepare(data)

    # Omitting both pose-contract and shared-query arguments is the ordinary
    # mode-3 API. It must select the same shared FP32 pose implementation as
    # the explicit prepared-handle form.
    default_scores, default_stats = _score(data, grain=grain)
    prepared_scores, prepared_stats = _score(
        data, grain=grain, prepared=prepared)

    assert default_scores.shape == (3, 20)
    assert np.isfinite(default_scores).all()
    _assert_bytes_equal(default_scores, prepared_scores, "score rows")
    _assert_bytes_equal(default_stats["winning_starts"],
                        prepared_stats["winning_starts"], "winner ids")
    _assert_common_census(data, default_stats, grain, 1)
    _assert_common_census(data, prepared_stats, grain, 1)
    for key in ("color_gradient_terms", "color_per_start_final_terms",
                "color_winner_rescore_terms"):
        assert default_stats[key] == prepared_stats[key]


@pytest.mark.parametrize("grain", [0, 11])
def test_single_start_contract_matches_explicit_prepared_for_generic_families(
        grain):
    data = _inputs(canonical=False)
    prepared = _prepare(data)
    default_scores, default_stats = _score(
        data, grain=grain, start_mode=0)
    prepared_scores, prepared_stats = _score(
        data, grain=grain, start_mode=0, prepared=prepared)
    _assert_bytes_equal(default_scores, prepared_scores, "one-start scores")
    _assert_bytes_equal(default_stats["winning_starts"],
                        prepared_stats["winning_starts"], "one-start winners")
    _assert_common_census(data, default_stats, grain, 0)
    _assert_common_census(data, prepared_stats, grain, 0)


@pytest.mark.parametrize("grain", [0, 11])
def test_ten_start_contract_matches_explicit_prepared_for_representative_families(
        grain):
    data = _inputs(canonical=True)
    prepared = _prepare(data)
    default_scores, default_stats = _score(
        data, grain=grain, start_mode=2)
    prepared_scores, prepared_stats = _score(
        data, grain=grain, start_mode=2, prepared=prepared)
    _assert_bytes_equal(default_scores, prepared_scores, "ten-start scores")
    _assert_bytes_equal(default_stats["winning_starts"],
                        prepared_stats["winning_starts"], "ten-start winners")
    _assert_common_census(data, default_stats, grain, 2)
    _assert_common_census(data, prepared_stats, grain, 2)


@pytest.mark.parametrize("start_mode", [0, 2])
def test_scheduled_start_grain_rejects_nonstandard_start_modes(start_mode):
    data = _inputs()
    with pytest.raises((ValueError, NotImplementedError)):
        _score(data, grain=16, start_mode=start_mode)


@pytest.mark.parametrize("grain,start_mode", [(11, 1), (16, 1)])
def test_prepared_color_fields_preserve_in_grid_and_analytic_fallback(
        grain, start_mode):
    data = _inputs()
    data["library_coords"] = data["library_coords"].copy()
    # One target plane is deliberately outside the query-built field domain;
    # other pairs remain in-domain and exercise interpolation in the same call.
    data["library_coords"][0, 1, :3] = (100.0, 100.0, 100.0)
    field_codes = np.array([0, 1, 0], dtype=np.int32)
    color_fields = cpu.prepare_color_fields(
        query_coords=data["query_coords"], query_types=data["query_types"],
        query_n_atoms=data["query_n_atoms"],
        interaction_r=data["interaction_r"],
        interaction_p=data["interaction_p"], backend_codes=field_codes)
    prepared = _prepare(data)

    default_scores, default_stats = _score(
        data, grain=grain, start_mode=start_mode,
        color_field_backends=field_codes,
        prepared_color_fields=color_fields)
    prepared_scores, prepared_stats = _score(
        data, grain=grain, start_mode=start_mode, prepared=prepared,
        color_field_backends=field_codes,
        prepared_color_fields=color_fields)

    _assert_bytes_equal(default_scores, prepared_scores, "field score rows")
    _assert_bytes_equal(default_stats["winning_starts"],
                        prepared_stats["winning_starts"], "field winners")
    for stats in (default_stats, prepared_stats):
        _assert_common_census(data, stats, grain, start_mode)
        assert stats["color_field_prepared_handle_used"] is True
        from cpu_test_capabilities import IS_ARM
        if IS_ARM:
            assert stats["color_field_interpolated_samples"] == 0
            assert stats["color_field_out_of_grid_fallback_samples"] == 0
            continue
        assert stats["color_field_interpolated_samples"] > 0
        assert stats["color_field_interpolated_terms"] > 0
        assert stats["color_field_out_of_grid_fallback_samples"] > 0
        assert stats["color_per_start_final_evaluations"] == (
            int(data["pair_query_row"].size) * {0: 1, 1: 4, 2: 10}[start_mode])
        assert stats["color_winner_rescore_evaluations"] == 0

    # The approximate field only drives optimization. The reported winner's
    # exact analytic score is retained from its per-start final evaluation.
    exact, _ = cpu.extension().rescore_scalar_color(
        data["query_coords"], data["query_types"], data["query_n_atoms"],
        data["library_coords"], data["library_counts"],
        data["pair_query_row"], data["pair_library_row"],
        data["interaction_r"], data["interaction_p"], default_scores,
        np.float32(0.5), False, False, False, grain)
    columns = [0, 2, 4, 7, 8]
    # rescore_scalar_color is the explicitly legacy physical-pose oracle. The
    # selected shared FP32 pose and retained exact score may differ by small
    # rounding deltas; this is an adjacent oracle check, not a bit-identity
    # contract between the two pose frames.
    np.testing.assert_allclose(default_scores[:, columns], exact[:, columns],
                               rtol=1e-4, atol=1e-5)


def test_prepared_shape_evaluator_is_reused_by_default_shared_pose_setup():
    # The retained color grain this ISA builds: 11 (AVX2) on x86, 16 (NEON) on ARM.
    data = _inputs()
    shape = cpu.prepare_combo_shape_evaluators(
        data["query_coords"], data["query_n_atoms"],
        shape_backend="scalar_analytic", shape_grain="scalar", shape_width=1)
    default_scores, default_stats = _score(
        data, grain=RETAINED_COLOR_GRAIN, prepared_shape_evaluators=shape)
    prepared = _prepare(data, shape_evaluators=shape)
    explicit_scores, explicit_stats = _score(
        data, grain=RETAINED_COLOR_GRAIN, prepared=prepared,
        prepared_shape_evaluators=shape)
    _assert_bytes_equal(default_scores, explicit_scores, "shape-reuse scores")
    _assert_bytes_equal(default_stats["winning_starts"],
                        explicit_stats["winning_starts"], "shape-reuse winners")
    assert default_stats["shape_query_preparation_in_call"] is False
    assert explicit_stats["shape_query_preparation_in_call"] is False
    assert default_stats["shape_query_preparation_seconds"] >= 0.0
    assert explicit_stats["shape_query_preparation_seconds"] >= 0.0


@pytest.mark.parametrize("grain", [11, 16])
def test_explicit_shared_pose_handle_is_thread_safe_for_pair_workers(grain):
    data = _inputs()
    # Ensure at least two native work units so this exercises independent
    # workers rather than only the serial partial-block path.
    data["pair_query_row"] = np.resize(
        data["pair_query_row"], 16).astype(np.int32, copy=False)
    data["pair_library_row"] = np.resize(
        data["pair_library_row"], 16).astype(np.int32, copy=False)
    prepared = _prepare(data)
    serial_scores, _ = _score(data, grain=grain, prepared=prepared)
    parallel_scores, worker_stats = _score(
        data, grain=grain, prepared=prepared, threads=2, with_census=False,
        with_worker_timers=True)
    _assert_bytes_equal(serial_scores, parallel_scores, "threaded score rows")
    assert worker_stats["pose_contract"] == "shared_fp32_reduced_v1"
    assert worker_stats["color_grain"] == grain
    elapsed = np.asarray(worker_stats["worker_elapsed_seconds"], dtype=float)
    assert elapsed.shape == (2,)
    assert np.isfinite(elapsed).all() and np.all(elapsed >= 0.0)

"""Four-start scheduling: the start-lane color grain (16) against scalar."""

import platform

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.reference.color import (
    MODE_COMBO_OPTIMAL,
    apply_pose_elementwise,
    fixed_pose_tolerance,
    score_columns,
)
from roshambo3.layout import color_view_from_counts, color_view_from_types


pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")


def _population(n_pairs, matrix_kind="general", shared_queries=False):
    n_query, n_features, width = 3, 4, 56
    query_coords = np.full((n_query, width, 4), 9876.0, np.float32)
    query_types = np.zeros((n_query, width), np.int32)
    query_n_atoms = np.full(n_query, 2, np.int32)
    library_coords = np.full((n_pairs, width, 4), -8765.0, np.float32)
    library_counts = np.zeros((n_pairs, n_features), np.uint8)

    for row in range(n_query):
        query_coords[row, 0] = (0.18 * row, -0.07 * row, 0.025, 1.0)
        query_coords[row, 1] = (-0.11 * row, 0.13, -0.04, 1.0)
        cursor = 2
        for typ in range(1, n_features):
            count = 17 if row == 2 and typ == 3 else (row + typ) % 3
            for item in range(count):
                query_coords[row, cursor] = (
                    0.09 * row + 0.12 * typ + 0.03 * item,
                    -0.06 * typ + 0.02 * item,
                    0.04 * row - 0.025 * item,
                    1.0,
                )
                query_types[row, cursor] = typ
                cursor += 1

    for row in range(n_pairs):
        library_coords[row, 0] = (0.015 * row, 0.025, -0.02, 1.0)
        library_coords[row, 1] = (0.09, -0.04 * row, 0.03, 1.0)
        library_counts[row, 0] = 2
        cursor = 2
        for typ in range(1, n_features):
            count = (2 * row + typ) % 18
            library_counts[row, typ] = count
            for item in range(count):
                library_coords[row, cursor] = (
                    0.035 * row + 0.08 * typ - 0.02 * item,
                    0.025 * typ + 0.015 * item,
                    -0.018 * row + 0.03 * item,
                    1.0,
                )
                cursor += 1

    interaction_r = np.ones((n_features, n_features), np.float32)
    if matrix_kind == "canonical":
        interaction_p = np.eye(n_features, dtype=np.float32)
    else:
        interaction_p = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.25, 0.55],
            [0.0, 0.4, 1.0, 0.15],
            [0.0, 0.7, 0.3, 1.0],
        ], dtype=np.float32)
        interaction_r[1:, 1:] = np.asarray([
            [0.9, 1.1, 0.8], [1.2, 1.0, 0.95], [0.85, 1.15, 1.05]
        ], dtype=np.float32)

    pair_query_row = (np.zeros(n_pairs, np.int32) if shared_queries else
                      np.arange(n_pairs, dtype=np.int32) % n_query)
    return dict(
        query_coords=query_coords,
        query_types=query_types,
        query_n_atoms=query_n_atoms,
        library_coords=library_coords,
        library_counts=library_counts,
        pair_query_row=pair_query_row,
        pair_library_row=np.arange(n_pairs, dtype=np.int32),
        interaction_r=interaction_r,
        interaction_p=interaction_p,
    )


def _oracle_row(kw, row, scores, mixing):
    query_row = int(kw["pair_query_row"][row])
    library_row = int(kw["pair_library_row"][row])
    query_view = color_view_from_types(
        kw["query_types"][query_row:query_row + 1],
        kw["query_n_atoms"][query_row:query_row + 1],
        kw["interaction_r"].shape[0])
    library_view = color_view_from_counts(
        kw["library_counts"][library_row:library_row + 1])
    qn = int(kw["query_n_atoms"][query_row])
    ln = int(kw["library_counts"][library_row, 0])
    nq = int(query_view.n_color[0])
    nl = int(library_view.n_color[0])
    qtypes = np.repeat(np.arange(1, query_view.n_features),
                       query_view.counts[0, 1:])
    ltypes = np.repeat(np.arange(1, library_view.n_features),
                       library_view.counts[0, 1:])
    query_shape = kw["query_coords"][query_row, :qn, :3]
    library_shape = kw["library_coords"][library_row, :ln, :3]
    query_color = kw["query_coords"][query_row, qn:qn + nq, :3]
    library_color = kw["library_coords"][library_row, ln:ln + nl, :3]
    quaternion, translation = scores[row, 9:13], scores[row, 13:16]
    posed_shape = apply_pose_elementwise(library_shape, quaternion, translation)
    posed_color = apply_pose_elementwise(library_color, quaternion, translation)
    expected = score_columns(
        query_shape, library_shape, query_color, qtypes, library_color, ltypes,
        kw["interaction_r"], kw["interaction_p"], mixing,
        quaternion, translation)
    tolerance = fixed_pose_tolerance(
        query_shape, posed_shape, query_color, qtypes, posed_color, ltypes,
        kw["interaction_r"], kw["interaction_p"], mixing,
        include_input_rounding=True)
    return expected, tolerance


@pytest.mark.parametrize("expected_start", [0, 3])
def test_shared_pose_winner_score_reuse_covers_first_and_last_start(
        expected_start):
    kw = _population(1, "canonical")
    library_shape = np.asarray([[1.0, 0.0, 0.0], [0.3, 0.2, 0.0]],
                               dtype=np.float32)
    kw["library_coords"][0, :2, :3] = library_shape
    if expected_start == 0:
        kw["query_coords"][0, :2, :3] = library_shape
    else:
        # The fourth start is a pi rotation around z.  Make only that fixed
        # pose align the intentionally asymmetric two-atom shape.
        kw["query_coords"][0, :2, :3] = -library_shape
    # Keep all color centers at the rotation origin so shape alone selects the
    # winner while color's exact cross-overlap is still retained/serialized.
    kw["query_coords"][0, 2:, :3] = 0.0
    kw["library_coords"][0, 2:, :3] = 0.0

    scores, census = cpu.score_pairs(
        color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
        steps=1, lr_q=0.0, lr_t=0.0,
        budget_override="T232 first/last shared winner fixture",
        start_mode=1, color_grain=0,
        pose_contract="shared_fp32_reduced_v1",
        with_census=True, **kw)

    assert census["pose_contract"] == "shared_fp32_reduced_v1"
    np.testing.assert_array_equal(census["winning_starts"], [expected_start])
    assert census["color_per_start_final_evaluations"] == 4
    assert census["color_winner_rescore_evaluations"] == 0
    assert census["color_winner_rescore_terms"] == 0
    assert np.isfinite(scores).all()

    # The legacy rescorer copies the input shape row, so these shape-only
    # fields must remain unchanged. It recomputes canonical color and self
    # overlaps in FP64, whereas production uses FP32 for the canonical route;
    # compare only final score columns [0, 2, 4] at FP32 output tolerance and
    # do not use its independently recomputed self-overlap fields as an oracle.
    exact, _ = cpu.extension().rescore_scalar_color(
        kw["query_coords"], kw["query_types"], kw["query_n_atoms"],
        kw["library_coords"], kw["library_counts"],
        kw["pair_query_row"], kw["pair_library_row"],
        kw["interaction_r"], kw["interaction_p"], scores,
        np.float32(0.45), False, False, False, 0)
    np.testing.assert_array_equal(scores[:, [1, 3, 5, 6]],
                                  exact[:, [1, 3, 5, 6]])
    np.testing.assert_allclose(scores[:, [0, 2, 4]], exact[:, [0, 2, 4]],
                               rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize("factor_torque", [False, True])
@pytest.mark.parametrize("n_pairs", [1, 3])
def test_prepared_canonical_four_start_uses_shared_count_descriptors(
        factor_torque, n_pairs):
    kw = _population(n_pairs, "canonical")
    options = dict(color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
                   steps=2, lr_q=0.0, lr_t=0.0,
                   budget_override="T230 prepared four-start fixture",
                   start_mode=1, pose_contract="shared_fp32_reduced_v1",
                   canonical_color_kernel=True,
                   factor_canonical_torque=factor_torque,
                   optimizer_exp_mode="baseline", with_census=True)
    candidate, census = cpu.score_pairs(color_grain=16, **options, **kw)
    reference, reference_census = cpu.score_pairs(
        color_grain=16, **{**options, "canonical_color_kernel": False}, **kw)

    assert np.isfinite(candidate).all()
    assert census["color_start_lane_scheduled_batch_numeric_bytes"] == 0
    assert census["color_start_lane_vector_evaluations"] > 0
    assert census["color_gradient_evaluations"] == 8 * n_pairs
    assert census["color_per_start_final_evaluations"] == 4 * n_pairs
    np.testing.assert_array_equal(census["winning_starts"],
                                  reference_census["winning_starts"])
    np.testing.assert_allclose(candidate[:, [0, 2, 4]],
                               reference[:, [0, 2, 4]],
                               rtol=3e-5, atol=2e-6)


def test_prepared_canonical_four_start_keeps_unsafe_input_on_exact_fallback():
    kw = _population(1, "canonical")
    kw["library_coords"][0, 2, :3] = (1.0e10, -1.0e10, 1.0e10)
    options = dict(color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
                   steps=1, lr_q=0.0, lr_t=0.0,
                   budget_override="T230 immutable-input fallback fixture",
                   start_mode=1, pose_contract="shared_fp32_reduced_v1",
                   canonical_color_kernel=True, with_census=True)
    candidate, census = cpu.score_pairs(color_grain=16, **options, **kw)
    reference, _ = cpu.score_pairs(
        color_grain=16, **{**options, "canonical_color_kernel": False}, **kw)

    assert census["color_start_lane_vector_evaluations"] == 0
    assert census["color_start_lane_scalar_fallback_pairs"] == 4
    np.testing.assert_allclose(candidate, reference, rtol=1e-6, atol=1e-7)


def test_start_lane_stamps_describe_dispatch_and_compiled_availability():
    is_aarch64 = platform.machine().lower() in ("aarch64", "arm64")
    stamp = cpu.color_build_stamp(16)
    assert stamp["color_compiled"] is True
    if is_aarch64:
        assert stamp["color_arm"] == "neon_fp32_scheduled_start8"
        assert stamp["color_isa"] == "NEON"
        assert stamp["color_vector_width"] == 4
    else:
        assert stamp["color_arm"] == "avx2_fp32_poly_scheduled_start8"
        assert stamp["color_isa"] == "AVX2+FMA"
        assert stamp["color_vector_width"] == 8
    assert stamp["color_start_count"] == 4
    assert stamp["color_start_mode"] == 1
    assert "no cross-start reduction" in stamp["color_reduction"]
    assert stamp["color_scalar_default"] is True
    prepared_helper = (
        "neon_fp32_canonical_analytic_neon4" if is_aarch64 else
        "avx2_fp32_canonical_analytic_xmm4")
    prepared_isa = "NEON" if is_aarch64 else "AVX2+FMA"
    assert stamp["color_prepared_canonical_route"] is True
    assert stamp["color_prepared_canonical_helper"] == prepared_helper
    assert stamp["color_prepared_canonical_isa"] == prepared_isa
    assert stamp["color_prepared_canonical_vector_width"] == 4
    assert "four independent start lanes" in stamp[
        "color_prepared_canonical_vector_encoding"]
    assert stamp["color_general_fallback_arm"] == stamp["color_arm"]
    assert stamp["color_general_fallback_vector_width"] == \
        stamp["color_vector_width"]


@pytest.mark.parametrize("n_pairs", [1, 2, 3, 7, 8, 9])
@pytest.mark.parametrize("matrix_kind", ["canonical", "general"])
def test_no_update_start_ranking_is_exact_and_pair_order_survives_tails(
        n_pairs, matrix_kind):
    kw = _population(n_pairs, matrix_kind)
    options = dict(color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
                   steps=1, lr_q=0.0, lr_t=0.0,
                   budget_override="T182 fixed-pose selector fixture",
                   start_mode=1, with_census=True)
    scalar, scalar_census = cpu.score_pairs(color_grain=0, **options, **kw)
    candidate, census = cpu.score_pairs(color_grain=16, **options, **kw)

    # Explicit zero learning rates hold every trajectory at its four starts;
    # exact per-start scoring chooses the winner in every two-pair block/tail.
    np.testing.assert_array_equal(candidate, scalar)
    np.testing.assert_array_equal(census["winning_starts"],
                                  scalar_census["winning_starts"])
    assert census["color_gradient_evaluations"] == 4 * n_pairs
    assert census["color_per_start_final_evaluations"] == 4 * n_pairs
    assert census["color_winner_rescore_evaluations"] == 0


def test_duplicate_pair_ids_keep_each_output_row_and_start_ties_are_stable():
    kw = _population(3, "general")
    kw["pair_query_row"][:] = (0, 0, 1)
    kw["pair_library_row"][:] = (0, 0, 2)
    baseline = cpu.score_pairs(
        color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
        steps=1, lr_q=0.0, lr_t=0.0,
        budget_override="T182 fixed-pose selector fixture",
        start_mode=1, color_grain=0, **kw)
    candidate = cpu.score_pairs(
        color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
        steps=1, lr_q=0.0, lr_t=0.0,
        budget_override="T182 fixed-pose selector fixture",
        start_mode=1, color_grain=16, **kw)
    np.testing.assert_array_equal(candidate, baseline)

    # No shape or color terms: all four exact final scores tie, so the
    # established strict-greater selector must retain start zero per row.
    empty = _population(3, "general")
    empty["query_n_atoms"][:] = 0
    empty["query_types"][:] = 0
    empty["library_counts"][:] = 0
    empty["library_coords"][:] = 0.0
    for arm in (0, 16):
        _, census = cpu.score_pairs(
            color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
            steps=1, lr_q=0.0, lr_t=0.0,
            budget_override="T182 fixed-pose selector fixture",
            start_mode=1, with_census=True,
            color_grain=arm, **empty)
        np.testing.assert_array_equal(census["winning_starts"], [0, 0, 0])


@pytest.mark.parametrize("n_pairs", [1, 2, 3, 7, 8, 9])
def test_gradient_lanes_keep_own_pose_exactly_rescored_and_match_oracle(n_pairs):
    kw = _population(n_pairs, "general")
    mixing = 0.45
    scores, census = cpu.score_pairs(
        color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=mixing,
        steps=2, start_mode=1, with_census=True,
        pose_contract="legacy_physical_fp64_pose",
        color_grain=16, **kw)
    assert np.isfinite(scores).all()
    assert census["color_gradient_evaluations"] == 4 * n_pairs * 2
    assert census["color_per_start_final_evaluations"] == 4 * n_pairs
    assert census["color_winner_rescore_evaluations"] == n_pairs
    if platform.machine().lower() in ("aarch64", "arm64"):
        # The retained general-matrix/legacy-FP64 route on ARM is analytic
        # scalar fallback. Native NEON4 specializes canonical shared poses.
        assert census["color_start_lane_vector_evaluations"] == 0
        assert census["color_start_lane_scalar_fallback_pairs"] > 0
    else:
        assert census["color_start_lane_vector_evaluations"] > 0
    assert sum(census["color_start_lane_useful_terms_by_interaction"]) == \
        census["color_gradient_terms"]
    assert sum(census["color_start_lane_target_count_bins_by_type"][19:]) == \
        census["color_gradient_evaluations"] * 3

    exact, _ = cpu.extension().rescore_scalar_color(
        kw["query_coords"], kw["query_types"], kw["query_n_atoms"],
        kw["library_coords"], kw["library_counts"],
        kw["pair_query_row"], kw["pair_library_row"],
        kw["interaction_r"], kw["interaction_p"], scores,
        np.float32(mixing), False, False, False, 16)
    np.testing.assert_array_equal(
        exact[:, [0, 2, 4, 7, 8]], scores[:, [0, 2, 4, 7, 8]])
    expected_rows = [_oracle_row(kw, row, scores, mixing)
                     for row in range(n_pairs)]
    columns = (
        ("tanimoto_combination", 0), ("tanimoto_shape", 1),
        ("tanimoto_color", 2), ("overlap_volume", 3),
        ("overlap_color", 4), ("self_overlap_query", 5),
        ("self_overlap_library", 6), ("self_color_query", 7),
        ("self_color_library", 8),
    )
    for row, (expected_row, tolerance) in enumerate(expected_rows):
        for name, column in columns:
            assert abs(float(scores[row, column]) - expected_row[name]) <= \
                max(tolerance[name], 1e-7), (row, name)
    expected = np.asarray([
        expected_row["tanimoto_combination"]
        for expected_row, _ in expected_rows])
    np.testing.assert_array_equal(
        np.argsort(-scores[:, 0], kind="stable"),
        np.argsort(-expected, kind="stable"))

    assert census["color_start_lane_scheduled_batch_numeric_bytes"] > 0


@pytest.mark.parametrize("start_mode", [0, 2])
def test_nonstandard_start_modes_keep_the_existing_sequential_path(start_mode):
    kw = _population(3, "general")
    options = dict(color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
                   steps=1, start_mode=start_mode,
                   pose_contract="legacy_physical_fp64_pose")
    scalar = cpu.score_pairs(color_grain=0, **options, **kw)
    fallback = cpu.score_pairs(color_grain=16, **options, **kw)
    np.testing.assert_array_equal(fallback, scalar)


def test_unsupported_fp32_matrix_uses_exact_gradient_fallback():
    kw = _population(3, "general")
    kw["interaction_r"][1, 1] = np.float32(1.0e7)
    options = dict(color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.45,
                   steps=1, start_mode=1, with_census=True)
    scalar, _ = cpu.score_pairs(color_grain=0, **options, **kw)
    fallback, census = cpu.score_pairs(color_grain=16, **options, **kw)
    np.testing.assert_array_equal(fallback, scalar)
    assert census["color_start_lane_scalar_fallback_pairs"] == 4 * 3
    assert census["color_start_lane_vector_evaluations"] == 0


@pytest.mark.parametrize("backend,selected_types", [
    (1, [1]),       # fp32 trilinear, one selected target type
    (1, [1, 2]),    # fp32 trilinear, two selected target types
    (3, [1]),       # fp32 brick, one selected target type
])
def test_start_lane_field_companion_reuses_prepared_fields_and_rescores_exactly(
        backend, selected_types):
    kw = _population(3, "general", shared_queries=True)
    # One selected source pair leaves its prepared target plane outside the
    # query field's fixed domain; that lane must take analytic fallback.
    kw["library_coords"][0, 2, :3] = (100.0, 100.0, 100.0)
    codes = np.zeros(kw["interaction_r"].shape[0], np.int32)
    codes[selected_types] = backend
    prepared = cpu.prepare_color_fields(
        query_coords=kw["query_coords"], query_types=kw["query_types"],
        query_n_atoms=kw["query_n_atoms"],
        interaction_r=kw["interaction_r"], interaction_p=kw["interaction_p"],
        backend_codes=codes)
    mixing = 0.45
    scores, census = cpu.score_pairs(
        color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=mixing,
        steps=2, start_mode=1, with_census=True,
        color_grain=16, color_field_backends=codes,
        prepared_color_fields=prepared, **kw)
    assert np.isfinite(scores).all()
    assert census["color_field_prepared_handle_used"] is True
    if platform.machine().lower() in ("aarch64", "arm64"):
        # General-matrix ARM fallback does not enter the vector field sampler.
        # Keep the numerical rescore assertions below on this actual route.
        assert census["color_start_lane_vector_evaluations"] == 0
        assert census["color_start_lane_scalar_fallback_pairs"] > 0
        assert census["color_field_interpolated_samples"] == 0
        assert census["color_field_interpolated_terms"] == 0
        assert census["color_field_out_of_grid_fallback_samples"] == 0
    else:
        assert census["color_field_interpolated_samples"] > 0
        assert census["color_field_interpolated_terms"] > 0
        assert census["color_field_out_of_grid_fallback_samples"] > 0
    exact, _ = cpu.extension().rescore_scalar_color(
        kw["query_coords"], kw["query_types"], kw["query_n_atoms"],
        kw["library_coords"], kw["library_counts"],
        kw["pair_query_row"], kw["pair_library_row"],
        kw["interaction_r"], kw["interaction_p"], scores,
        np.float32(mixing), False, False, False, 16)
    np.testing.assert_array_equal(
        exact[:, [0, 2, 4, 7, 8]], scores[:, [0, 2, 4, 7, 8]])

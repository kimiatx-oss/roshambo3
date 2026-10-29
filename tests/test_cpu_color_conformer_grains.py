"""Optimizer and binding integration for conformer-lane grain 11."""

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.reference.color import MODE_COMBO_OPTIMAL


pytestmark = pytest.mark.skipif(
    not cpu.available() or not cpu.color_build_stamp(11).get("color_runtime_available"),
    reason="conformer-lane integration requires AVX2+FMA")


def _population(matrix_kind):
    n_query, n_library, n_features, width = 3, 11, 4, 10
    query_coords = np.full((n_query, width, 4), 9876.0, np.float32)
    query_types = np.zeros((n_query, width), np.int32)
    query_counts = np.zeros(n_query, np.int32)
    library_coords = np.full((n_library, width, 4), -8765.0, np.float32)
    library_counts = np.zeros((n_library, n_features), np.uint8)

    for row in range(n_query):
        query_coords[row, 0] = (0.03 * row, -0.02 * row, 0.01, 1.0)
        query_types[row, 0] = 0
        cursor = 1
        for typ in range(1, n_features):
            count = 1 + ((row + typ) % 2)
            for item in range(count):
                query_coords[row, cursor] = (
                    0.17 * row + 0.11 * typ + 0.04 * item,
                    -0.09 * typ + 0.02 * item,
                    0.07 * row - 0.03 * item,
                    1.0,
                )
                query_types[row, cursor] = typ
                cursor += 1
        query_counts[row] = cursor

    for row in range(n_library):
        library_coords[row, 0] = (0.025 * row, 0.015, -0.01, 1.0)
        library_counts[row, 0] = 1
        cursor = 1
        for typ in range(1, n_features):
            count = (row + 2 * typ) % 4
            library_counts[row, typ] = count
            for item in range(count):
                library_coords[row, cursor] = (
                    0.13 * row + 0.08 * typ - 0.025 * item,
                    0.05 * typ + 0.035 * item,
                    -0.04 * row + 0.02 * item,
                    1.0,
                )
                cursor += 1

    interaction_r = np.ones((n_features, n_features), np.float32)
    if matrix_kind == "canonical":
        interaction_p = np.eye(n_features, dtype=np.float32)
    else:
        interaction_p = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.25, 0.55],
            [0.0, 0.4, 1.0, 0.15],
            [0.0, 0.7, 0.3, 1.0],
        ], dtype=np.float32)
        interaction_r[1:, 1:] = np.array([
            [0.9, 1.1, 0.8], [1.2, 1.0, 0.95], [0.85, 1.15, 1.05]
        ], dtype=np.float32)

    return dict(
        query_coords=query_coords,
        query_types=query_types,
        query_n_atoms=query_counts,
        library_coords=library_coords,
        library_counts=library_counts,
        pair_query_row=(np.arange(n_library) % n_query).astype(np.int32),
        pair_library_row=np.arange(n_library, dtype=np.int32),
        interaction_r=interaction_r,
        interaction_p=interaction_p,
    )


def test_conformer_grain_stamp_and_compiled_availability():
    stamp = cpu.color_build_stamp(11)
    assert stamp["color_compiled"] is True
    assert stamp["color_arm"] == "avx2_fp32_poly_conformer8"
    assert stamp["color_vector_width"] == 8
    assert stamp["color_lane_owner"] == "independent query/library conformer pair"
    assert stamp["color_exp"] == "degree-7 range-reduced AVX2 polynomial"
    assert stamp["color_scalar_default"] is True
    assert stamp["color_count_order"] == "input order"


@pytest.mark.parametrize("matrix_kind", ["canonical", "general"])
def test_mode3_conformer_batches_keep_pair_identity_and_exact_final(matrix_kind):
    kw = _population(matrix_kind)
    common = dict(color=True, color_mode=MODE_COMBO_OPTIMAL, mixing=0.5,
                  steps=2, start_mode=0, with_census=True)
    scalar, scalar_census = cpu.score_pairs(color_grain=0, **common, **kw)
    ordinary, census = cpu.score_pairs(color_grain=11, **common, **kw)

    assert np.isfinite(ordinary).all()
    assert np.max(np.abs(ordinary[:, :9] - scalar[:, :9])) <= 5e-3
    np.testing.assert_array_equal(
        ordinary[:, [0, 2, 4, 7, 8]],
        cpu.extension().rescore_scalar_color(
            kw["query_coords"], kw["query_types"], kw["query_n_atoms"],
            kw["library_coords"], kw["library_counts"],
            kw["pair_query_row"], kw["pair_library_row"],
            kw["interaction_r"], kw["interaction_p"], ordinary,
            np.float32(0.5), False, False, False, 0)[0][:, [0, 2, 4, 7, 8]])

    for key in ("shape_gradient_terms", "color_gradient_terms",
                "color_per_start_final_terms", "color_winner_rescore_terms",
                "color_gradient_evaluations", "color_winner_rescore_evaluations"):
        assert census[key] == scalar_census[key]
    assert census["color_winner_rescore_evaluations"] == 0
    assert census["color_conformer_vector_evaluations"] > 0
    assert census["color_conformer_scalar_fallback_pairs"] == 0
    assert tuple(census["color_conformer_interaction_matrix_shape"]) == (4, 4)
    assert sum(census["color_conformer_useful_terms_by_interaction"]) == \
        census["color_gradient_terms"]


def test_mode2_fixed_pose_conformer_batch_preserves_shape_columns():
    kw = _population("general")
    from roshambo3.reference.color import MODE_SHAPE_POSE_COLOR_RESCORE

    common = dict(color=True, color_mode=MODE_SHAPE_POSE_COLOR_RESCORE,
                  mixing=0.5, steps=3, start_mode=0, with_census=True)
    scalar, scalar_census = cpu.score_pairs(color_grain=0, **common, **kw)
    candidate, census = cpu.score_pairs(color_grain=11, **common, **kw)
    shape_columns = [1, 3, 5, 6] + list(range(9, 20))
    np.testing.assert_array_equal(candidate[:, shape_columns],
                                  scalar[:, shape_columns])
    assert census["color_cross_terms"] == scalar_census["color_cross_terms"]
    assert census["color_conformer_vector_evaluations"] > 0

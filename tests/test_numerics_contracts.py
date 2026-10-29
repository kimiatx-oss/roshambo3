"""Live fp64 contracts; no committed output arrays or native backend required."""
import math
from dataclasses import replace

import numpy as np
import pytest

from roshambo3.reference import numerics_contracts as nc
from roshambo3.reference.color import score_columns
from roshambo3.reference.overlap import ALPHA, PAIR_PREFACTOR


@pytest.fixture
def panel():
    rng = np.random.default_rng(270)
    return [(rng.normal(size=(5, 3)), rng.normal(size=(7, 3))) for _ in range(12)]


def test_grids_round_nearest_even_and_dequantize_in_fp32():
    c = np.array([[0.5, 1.5, -2.5], [316, -600, 0]]) * float(nc.numerics.WIRE_QUANT_STEP)

    grid = nc.CoordinateGrid()
    actual = nc.reconstruct_coordinates(c, grid)
    wire = np.rint(c.astype(np.float32) * np.float32(1.0 / grid.step))
    expected = (wire.astype(np.float32) * np.float32(grid.step)).astype(np.float64)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(wire[0], [0, 2, -2])
    with pytest.raises(ValueError, match="overflow"):
        nc.reconstruct_coordinates([[129, 0, 0]], grid)
    with pytest.raises(ValueError, match="finite"):
        nc.reconstruct_coordinates([[math.nan, 0, 0]], grid)


def test_library_query_and_rotated_query_grids_are_separate():
    c = [[0.0016, -0.0024, 0.0036]]
    rotation = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
    path = nc.PathArithmetic("declared", library_grid=nc.CoordinateGrid(0.002),
                             query_grid=nc.CoordinateGrid(0.001),
                             rotated_query_grid=nc.CoordinateGrid(0.003))
    a, b = nc.scored_coordinates(c, c, contract=path, query_rotation=rotation)
    np.testing.assert_array_equal(b, nc.reconstruct_coordinates(c, path.library_grid))
    expected = nc.reconstruct_coordinates(
        nc.reconstruct_coordinates(c, path.query_grid) @ rotation.T,
        path.rotated_query_grid)
    np.testing.assert_array_equal(a, expected)
    assert not np.array_equal(a, b)


def test_atom_grids_do_not_silently_quantize_derived_centroids():
    c = [[0.00012345, 0.0016789, -0.0004321]]
    path = nc.PathArithmetic("wire", query_grid=nc.CoordinateGrid(),
                             library_grid=nc.CoordinateGrid())
    a, b = nc.scored_coordinates(c, c, contract=path, color=True)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, c)


def test_full_row_reuses_physical_color_contract_at_scored_coordinates():
    a = np.array([[0.12345, 0, 0], [1, 0.23456, 0]])
    b = np.array([[0, 0.34567, 0], [0.45678, 1, 0]])
    ac, bc = a[:1] / 3, b / 7
    r = np.array([[0, 0, 0], [0, 0.7, 0.2], [0, 0.2, 1.3]])
    p = np.array([[0, 0, 0], [0, 1.1, 0.4], [0, 0.4, 0.8]])
    path = nc.PathArithmetic("physical", library_grid=nc.CoordinateGrid())
    pose = dict(pose_quat=[0.9, 0.1, 0.2, -0.3], pose_translation=[0.3, -0.2, 0.1])
    got = nc.fixed_pose_row(a, b, contract=path, query_color=ac, query_types=[1],
                            library_color=bc, library_types=[1, 2],
                            interaction_r=r, interaction_p=p, mixing=0.4, **pose)
    aa, bb = nc.scored_coordinates(a, b, contract=path, **pose)
    ca, cb = nc.scored_coordinates(ac, bc, contract=path, color=True, **pose)
    expected = score_columns(aa, bb, ca, [1], cb, [1, 2], r, p, 0.4)
    assert got == expected
    assert len(got) == 9


def test_shared_pose_requires_and_consumes_prepared_bytes():
    a, b = np.array([[0.2, 0.3, 0.4]]), np.array([[0.5, 0.6, 0.7]])
    path = nc.PathArithmetic("shared", pose_contract="shared_fp32_reduced_v1")
    with pytest.raises(ValueError, match="prepared"):
        nc.scored_coordinates(a, b, contract=path)
    scale = math.sqrt(ALPHA / (2 * math.log(2)))
    matrix = (np.eye(3) * scale).astype(np.float32)
    translation = np.array([0.01, -0.02, 0.03], dtype=np.float32)
    aa, bb = nc.scored_coordinates(a, b, contract=path, prepared_matrix=matrix,
                                   prepared_translation=translation)
    np.testing.assert_array_equal(aa, a)
    np.testing.assert_array_equal(bb, (b @ matrix.astype(np.float64).T + translation) / scale)


def test_independent_constants_and_closed_form_bounds():
    # One coincident atom: argument error vanishes, leaving transcendental,
    # coefficient, one serial add, and a possible FTZ loss.
    path = nc.PathArithmetic("single", argument_roundings=0, coefficient_roundings=0,
                             exp2_relative_error=0, sum_roundings=0)
    c = [[0, 0, 0]]
    row = nc.fixed_pose_row(c, c, contract=path)
    assert row["overlap_volume"] == PAIR_PREFACTOR
    assert row["tanimoto_shape"] == 1
    bounds = nc.residual_bounds(c, c, contract=path)
    assert bounds["overlap_volume"] == PAIR_PREFACTOR * np.finfo(np.float32).tiny
    assert bounds["overlap_color"] == 0
    assert bounds["tanimoto_color"] == 0


def test_declared_uncertainty_monotonically_widens_bounds(panel):
    a, b = panel[0]
    path = nc.PathArithmetic("scalar")
    base = nc.residual_bounds(a, b, contract=path)
    wider = nc.residual_bounds(a, b, contract=replace(path, coordinate_error=1e-5))
    assert wider["overlap_volume"] > base["overlap_volume"]
    assert wider["tanimoto_shape"] > base["tanimoto_shape"]
    more = nc.residual_bounds(a, b, contract=replace(path, exp2_relative_error=1e-4))
    assert more["overlap_volume"] > base["overlap_volume"]


def test_live_fp32_shape_sum_inside_derived_bounds(panel):
    path = nc.PathArithmetic("serial")
    for a, b in panel:
        row = nc.fixed_pose_row(a, b, contract=path)
        bound = nc.residual_bounds(a, b, contract=path)
        # A simple independent fp32 evaluator, not production output bits.
        d = a.astype(np.float32)[:, None] - b.astype(np.float32)[None]
        terms = np.exp(-np.float32(ALPHA / 2) * np.sum(d*d, axis=-1)).ravel()
        total = np.float32(0)
        for term in terms:
            total = np.float32(total + term)
        volume = float(np.float32(PAIR_PREFACTOR) * total)
        assert abs(volume - row["overlap_volume"]) <= bound["overlap_volume"]


def test_perturbed_derived_width_is_refused(panel):
    path = nc.PathArithmetic("serial")
    observed, expected, bounds = [], [], []
    for a, b in panel:
        row = nc.fixed_pose_row(a, b, contract=path)
        wrong = dict(row)
        d2 = np.sum((a[:, None] - b[None])**2, axis=-1)
        wrong["overlap_volume"] = float(PAIR_PREFACTOR * np.exp(-0.5*ALPHA*1.0001*d2).sum())
        observed.append(wrong)
        expected.append(row)
        bounds.append(nc.residual_bounds(a, b, contract=path))
    with pytest.raises(AssertionError, match="overlap_volume"):
        nc.assert_residuals(observed, expected, bounds)


def test_mean_uses_signed_declared_effect_and_correlated_uncertainty():
    expected = [{"score": 0.5}] * 4
    observed = [{"score": 0.50003}] * 4
    bounds = [{"score": 1e-5}] * 4
    offsets = [{"score": 3e-5}] * 4
    report = nc.assert_residuals(observed, expected, bounds, signed_offsets=offsets)
    np.testing.assert_allclose(report["score"]["mean_interval"], [2e-5, 4e-5])
    # The repeated systematic error must not benefit from sqrt(N) cancellation.
    report = nc.residual_report(observed, expected, bounds)
    assert not report["score"]["mean_pass"]
    assert not report["score"]["rows_pass"]
    with pytest.raises(ValueError, match="nonempty"):
        nc.residual_report([], [], [])
    with pytest.raises(ValueError, match="finite"):
        nc.residual_report(observed, expected, [{"score": math.inf}] * 4)


def test_certification_requires_stated_operation_model():
    path = nc.PathArithmetic("diagnostic")
    with pytest.raises(ValueError, match="operation-model"):
        path.validate_for_certification()
    replace(path, operation_model_derivation="synthetic serial test model").validate_for_certification()
    shared = replace(path, pose_contract="shared_fp32_reduced_v1",
                     operation_model_derivation="synthetic prepared pose")
    with pytest.raises(ValueError, match="diagnostic"):
        shared.validate_for_certification()
    with pytest.raises(ValueError, match="arithmetic contract"):
        nc.assert_residuals([{"x": 1}], [{"x": 1}], [{"x": 0}], certification=True)


def test_cancellation_sensitive_input_cast_is_bounded():
    # Relative exponent-only bounds miss a lost displacement at large origins.
    a, b = np.array([[1e7, 0, 0]]), np.array([[1e7+0.1, 0, 0]])
    path = nc.PathArithmetic("large_origin")
    row = nc.fixed_pose_row(a, b, contract=path)
    bound = nc.residual_bounds(a, b, contract=path)
    assert abs(PAIR_PREFACTOR-row["overlap_volume"]) <= bound["overlap_volume"]


def test_uncertainty_includes_exact_grid_displacement(panel):
    a, b = panel[0]
    path = nc.PathArithmetic("wire", query_grid=nc.CoordinateGrid(),
                             library_grid=nc.CoordinateGrid())
    quantized = nc.fixed_pose_row(a, b, contract=path)
    physical = nc.fixed_pose_row(a, b, contract=nc.PathArithmetic("physical"))
    bound = nc.residual_bounds(a, b, contract=path)
    uncertainty = nc.fixed_pose_uncertainty(a, b, contract=path)
    for key in quantized:
        assert uncertainty[key] == abs(quantized[key]-physical[key]) + bound[key]


@pytest.mark.parametrize("rows", [[{}], [{"x": float("nan")}], [{"x": 1.}, {"y": 1.}]])
def test_empty_nonfinite_or_inconsistent_rows_are_refused(rows):
    from roshambo3.reference.numerics_contracts import residual_report
    with pytest.raises(ValueError):
        residual_report(rows, rows, rows)

"""Scalar analytic color, checked against frozen independent data."""
import json
from pathlib import Path

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.layout import color_view_from_counts, color_view_from_types
from roshambo3.reference.color import (apply_pose_elementwise,
                                       fixed_pose_tolerance, score_columns)
from roshambo3.reference.color_census import color_term_census

pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")
FIXTURE = Path(__file__).parent / "data/color_contract_fixture.npz"


def packed(case, posed=True):
    qn = len(case["query_shape"])
    ln = len(case["library_shape"])
    qc = len(case["query_color"])
    lc = len(case["library_color"])
    width = max(qn + qc, ln + lc, 1) + 4  # explicit poisoned padding below
    a = np.zeros((1, width, 4), np.float32)
    b = np.zeros((1, width, 4), np.float32)
    types = np.zeros((1, width), np.int32)
    a[0, :qn, :3] = case["query_shape"]
    a[0, :qn, 3] = 1
    a[0, qn:qn+qc, :3] = case["query_color"]
    a[0, qn:qn+qc, 3] = 1
    types[0, qn:qn+qc] = case["query_types"]
    suffix = "_posed" if posed else ""
    b[0, :ln, :3] = case["library_shape" + suffix]
    b[0, :ln, 3] = 1
    b[0, ln:ln+lc, :3] = case["library_color" + suffix]
    b[0, ln:ln+lc, 3] = 1
    counts = np.zeros((1, case["interaction_r"].shape[0]), np.uint8)
    counts[0, 0] = ln
    for t in case["library_types"]:
        counts[0, int(t)] += 1
    return dict(query_coords=a, query_types=types,
                query_n_atoms=np.array([qn], np.int32),
                library_coords=b, library_counts=counts,
                pair_query_row=np.array([0], np.int32),
                pair_library_row=np.array([0], np.int32),
                interaction_r=np.asarray(case["interaction_r"], np.float32),
                interaction_p=np.asarray(case["interaction_p"], np.float32))


def cases():
    with np.load(FIXTURE, allow_pickle=False) as z:
        record = json.loads(str(z["record"]))
        for name, meta in record["cases"].items():
            yield name, meta, {key[len(name)+2:]: z[key].copy()
                               for key in z.files if key.startswith(name+"__")}, record


@pytest.mark.parametrize("name,meta,case,record", list(cases()),
                         ids=lambda x: x if isinstance(x, str) else None)
def test_fixed_pose_color_matches_independent_fixture(name, meta, case, record):
    kw = packed(case)
    shape = np.zeros((1, 20), np.float32)
    expected = dict(zip(record["scalar_columns"], case["columns"]))
    shape[0, 1] = expected["tanimoto_shape"]
    shape[0, 9] = 1.0
    args = [kw[k] for k in ("query_coords", "query_types", "query_n_atoms",
                            "library_coords", "library_counts",
                            "pair_query_row", "pair_library_row",
                            "interaction_r", "interaction_p")]
    result, census = cpu.extension().rescore_scalar_color(
        *args, shape, np.float32(meta["mixing"]), True, False)
    recomputed, _ = cpu.extension().rescore_scalar_color(
        *args, shape, np.float32(meta["mixing"]), False, False, True)
    assert np.array_equal(result, recomputed), name
    # The only legal loop bounds are the segment counts, never padded L.
    poisoned = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    poisoned[0][0, len(case["query_shape"])+len(case["query_color"]):, :3] = 9999
    poisoned[3][0, len(case["library_shape"])+len(case["library_color"]):, :3] = -9999
    poisoned_result, _ = cpu.extension().rescore_scalar_color(
        *poisoned, shape, np.float32(meta["mixing"]), False, False)
    assert np.array_equal(result, poisoned_result), name
    tol = fixed_pose_tolerance(
        case["query_shape"], case["library_shape_posed"],
        case["query_color"], case["query_types"],
        case["library_color_posed"], case["library_types"],
        case["interaction_r"], case["interaction_p"], meta["mixing"])
    for key, col in (("tanimoto_color", 2), ("overlap_color", 4),
                     ("self_color_query", 7), ("self_color_library", 8)):
        assert abs(float(result[0, col]) - expected[key]) <= max(tol[key], 1e-7), (name, key)
    assert abs(float(result[0, 0]) -
               ((1-meta["mixing"])*float(shape[0, 1]) +
                meta["mixing"]*float(result[0, 2]))) < 1e-6
    assert np.isfinite(result).all()
    qview = color_view_from_types(kw["query_types"], kw["query_n_atoms"],
                                  kw["interaction_r"].shape[0])
    lview = color_view_from_counts(kw["library_counts"])
    expected_terms = color_term_census(
        qview, kw["pair_query_row"], lview, kw["pair_library_row"],
        kw["interaction_r"], kw["interaction_p"],
        same_type_fast_path=census["color_canonical"])
    assert census["color_cross_terms"] == expected_terms["color_final_rescore_terms"]
    assert census["color_query_self_terms"] == expected_terms["query_self_color_terms"]
    assert census["color_library_self_terms"] == expected_terms["library_self_color_terms"]


def test_mode2_preserves_shape_result_bits_and_mixing_zero_still_scores_color():
    name, meta, case, _ = next(x for x in cases()
                               if x[0] == "mode_discrimination_distinct_shapes")
    kw = packed(case, posed=False)
    shape = cpu.score_pairs(steps=25, start_mode=1, **kw)
    color = cpu.score_pairs(steps=25, start_mode=1, color=True,
                            mixing=0.0, **kw)
    assert np.array_equal(shape[:, 1], color[:, 1])
    assert np.array_equal(shape[:, 3:4], color[:, 3:4])
    assert np.array_equal(shape[:, 5:7], color[:, 5:7])
    assert np.array_equal(shape[:, 9:], color[:, 9:])
    assert np.array_equal(shape[:, 0], color[:, 0])
    assert color[0, 2] > 0 and color[0, 4] > 0
    mixed = cpu.score_pairs(steps=25, start_mode=1, color=True,
                            mixing=0.5, **kw)
    assert np.array_equal(color[:, 1:], mixed[:, 1:])
    assert np.isclose(mixed[0, 0], 0.5*(mixed[0, 1]+mixed[0, 2]))
    q = mixed[0, 9:13]
    t = mixed[0, 13:16]
    oracle = score_columns(
        case["query_shape"], case["library_shape"],
        case["query_color"], case["query_types"],
        case["library_color"], case["library_types"],
        case["interaction_r"], case["interaction_p"], 0.5, q, t)
    posed_shape = apply_pose_elementwise(case["library_shape"], q, t)
    posed_color = apply_pose_elementwise(case["library_color"], q, t)
    tol = fixed_pose_tolerance(
        case["query_shape"], posed_shape,
        case["query_color"], case["query_types"], posed_color,
        case["library_types"], case["interaction_r"],
        case["interaction_p"], 0.5, include_input_rounding=True)
    for key, col in (("tanimoto_color", 2), ("overlap_color", 4),
                     ("self_color_query", 7), ("self_color_library", 8)):
        assert abs(float(mixed[0, col]) - oracle[key]) <= max(tol[key], 1e-7), key

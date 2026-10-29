"""Portable fault checks for coordinate-aware color arithmetic comparisons."""
from __future__ import annotations

import numpy as np
import pytest

from tests.color_coordinate_comparison import (
    assert_color_row_matches,
    assert_cross_path_residual,
    color_oracle_row,
    color_row_bounds,
    snap_query_like_cuda,
)


def _case():
    query = np.array([
        [0.0007, 0.1, -0.2],
        [1.0032, 0.1, -0.2],
        [-0.3011, 0.6, 0.2],
    ], dtype=np.float32)
    library = np.array([
        [0.0002, 0.1, -0.2],
        [1.0027, 0.1, -0.2],
        [-0.3006, 0.6, 0.2],
    ], dtype=np.float32)
    types = np.array([1, 1, 2], dtype=np.int32)
    widths = np.ones((3, 3), dtype=np.float32)
    heights = np.diag([0.0, 1.0, 1.0]).astype(np.float32)
    step = np.float32(2**-8)
    inverse = np.float32(2**8)
    return query, library, types, widths, heights, step, inverse


def test_off_grid_oracles_use_each_paths_coordinates_and_matching_self_terms():
    query, library, types, widths, heights, step, inverse = _case()
    snapped = snap_query_like_cuda(query, step=step, inverse=inverse)
    assert not np.array_equal(query, snapped)

    raw_expected = color_oracle_row(
        query, types, library, types, widths, heights)
    snapped_expected = color_oracle_row(
        snapped, types, library, types, widths, heights)
    # The self term is a function of the scored coordinates. This synthetic
    # pair makes a query-frame split visible independently of the cross term.
    assert raw_expected["self_color_query"] != snapped_expected["self_color_query"]

    raw_bounds = color_row_bounds(
        query, types, library, types, widths, heights,
        name="portable-field-control")
    snapped_bounds = color_row_bounds(
        snapped, types, library, types, widths, heights,
        name="portable-analytic-control")
    assert_color_row_matches(raw_expected, raw_expected, raw_bounds)
    assert_color_row_matches(snapped_expected, snapped_expected, snapped_bounds)
    assert_cross_path_residual(
        raw_expected["overlap_color"], snapped_expected["overlap_color"],
        raw_expected["overlap_color"], snapped_expected["overlap_color"],
        raw_bounds["overlap_color"], snapped_bounds["overlap_color"])

    wrong_self_frame = dict(snapped_expected)
    wrong_self_frame["self_color_query"] = raw_expected["self_color_query"]
    with pytest.raises(AssertionError, match="self_color_query"):
        assert_color_row_matches(wrong_self_frame, snapped_expected, snapped_bounds)


def test_same_coordinate_comparator_rejects_an_arithmetic_fault():
    query, library, types, widths, heights, step, inverse = _case()
    snapped = snap_query_like_cuda(query, step=step, inverse=inverse)
    expected = color_oracle_row(snapped, types, library, types, widths, heights)
    bounds = color_row_bounds(
        snapped, types, library, types, widths, heights,
        name="portable-fault-plant")

    faulty_overlap = dict(expected)
    faulty_overlap["overlap_color"] += 1e-2
    with pytest.raises(AssertionError, match="overlap_color"):
        assert_color_row_matches(faulty_overlap, expected, bounds)

    with pytest.raises(AssertionError, match="cross-path arithmetic residual"):
        assert_cross_path_residual(
            expected["overlap_color"] + 1e-2,
            expected["overlap_color"],
            expected["overlap_color"],
            expected["overlap_color"],
            bounds["overlap_color"],
            bounds["overlap_color"])

"""Zero-denominator shape scores and the device top-K's NaN boundary."""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import scores as S
from roshambo3.backend import PoseSearch, cuda as backend
from roshambo3.engine import TopK
from roshambo3.layout import tile32

needs_gpu = pytest.mark.skipif(not backend.available(),
                               reason="no CUDA extension in this build")


def _empty_shape_pairs():
    """One empty query, two empty libraries, and one ordinary library row."""
    L = 16
    query_coords = np.zeros((1, L, 4), np.float32)
    query_types = np.zeros((1, L), np.int32)
    library = np.zeros((32, L, 4), np.float32)
    counts = np.zeros((32, 1), np.uint8)
    for row in (2, 3, 4):
        library[row, 0, :3] = (0.25 * row, 0.0, 0.0)
        library[row, 0, 3] = 1.0
        counts[row, 0] = 1
    return dict(
        query_coords=query_coords,
        query_types=query_types,
        query_n_atoms=np.array([0], np.int32),
        library_coords=tile32(library),
        library_counts=counts,
        pair_query_row=np.zeros(3, np.int32),
        pair_library_row=np.array([0, 1, 2], np.int32),
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.zeros((1, 1), np.float32),
    )


@needs_gpu
@pytest.mark.parametrize("pose_search", [PoseSearch.FIELD, PoseSearch.ANALYTIC])
def test_empty_empty_shape_tanimoto_is_zero_on_both_kernels(pose_search):
    scores = backend.score_pairs(
        pose_search=pose_search, color=False, mixing=0.0, steps=0,
        start_mode=0, **_empty_shape_pairs())
    assert np.array_equal(scores[:, S.TANIMOTO_SHAPE], np.zeros(3, np.float32))
    assert np.array_equal(scores[:, S.TANIMOTO_COMBINATION],
                          scores[:, S.TANIMOTO_SHAPE])
    assert np.isfinite(scores).all()


@needs_gpu
def test_device_topk_keeps_finite_rows_when_empty_empty_scores_were_nan():
    """Without the zero convention, rows 0 and 1 produce NaN while rows 2.. are finite zero.

    This sends that real score block through the device sink, whose compaction
    gate must drop NaNs before selection. With the zero convention all three
    tie at zero and canonical id order supplies the answer.
    """
    sink = TopK(k=3)
    sink.prepare(1, 0)
    assert backend.score_pairs(
        pose_search=PoseSearch.ANALYTIC, color=False, mixing=0.0,
        steps=0, start_mode=0, topk=sink, gid_base=0,
        max_rows_per_query=32, **_empty_shape_pairs()) is None
    (scores, ids), = sink.finalize()
    assert np.array_equal(ids, np.array([0, 1, 2], np.int64))
    assert np.array_equal(scores[:, S.TANIMOTO_COMBINATION],
                          np.zeros(3, np.float32))
    assert np.isfinite(scores).all()

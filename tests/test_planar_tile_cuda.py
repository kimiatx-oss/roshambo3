"""A planar library tile request is refused, not silently served.

``layout.tile32`` is ``[tile][atom][lane][component]``; its ``planar`` variant
is ``[tile][atom][component][lane]``, which the kernels would reach as
``TILEB < 0``. The build instantiates no planar pair-kernel cell, so a
``library_tile_mode < 0`` request must be refused by name on both pose-search
paths rather than run at the interleaved layout -- an inert layout and a slow
one look identical from outside.
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import backend
from roshambo3.backend import PoseSearch
from roshambo3.backend.cuda import score_pairs
from roshambo3.layout import tile32

needs_gpu = pytest.mark.skipif(not backend.available(),
                               reason="no CUDA extension")

N_ROWS, L, N_REAL, N_QUERY = 64, 24, 14, 3


def _fixture(seed: int = 0x89):
    """A small int16 shape library and a float32 query panel.

    int16 deliberately: the planar tile's whole argument is about the int16 wire
    record, whose component fetch is 2 bytes at a 6-byte stride.
    """
    rng = np.random.default_rng(seed)
    xyz = rng.integers(-9000, 9000, size=(N_ROWS, L, 3)).astype(np.int16)
    xyz[:, N_REAL:, :] = 0
    counts = np.zeros((N_ROWS, 1), np.uint8)
    counts[:, 0] = N_REAL
    q = np.zeros((N_QUERY, L, 4), np.float32)
    q[:, :N_REAL, :3] = rng.normal(0.0, 1.6, size=(N_QUERY, N_REAL, 3))
    q[:, :N_REAL, 3] = 1.0
    qt = np.zeros((N_QUERY, L), np.int32)
    qn = np.full(N_QUERY, N_REAL, np.int32)
    pq = np.repeat(np.arange(N_QUERY, dtype=np.int32), N_ROWS)
    pt = np.tile(np.arange(N_ROWS, dtype=np.int32), N_QUERY)
    return dict(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_counts=counts, pair_query_row=pq, pair_library_row=pt,
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.ones((1, 1), np.float32)), xyz


def _score(xyz, *, planar: bool, **kw):
    a, _ = _fixture()
    a["library_coords"] = tile32(xyz, planar=planar)
    a.setdefault("library_tile_mode", -1 if planar else 1)
    a.update(kw)
    return score_pairs(**a)


@needs_gpu
@pytest.mark.parametrize("pose_search", [PoseSearch.FIELD, PoseSearch.ANALYTIC])
def test_a_planar_request_is_refused_by_name(pose_search):
    _, xyz = _fixture()
    with pytest.raises(Exception) as exc:
        _score(xyz, planar=True, pose_search=pose_search)
    assert "library_tile_mode" in str(exc.value)
    # And the same request at the interleaved tile is fine, so the refusal is
    # about the layout.
    _score(xyz, planar=False, pose_search=pose_search)


@needs_gpu
def test_the_untiled_layout_is_refused():
    """``tile_mode == 0`` is not a layout either pair kernel reads."""
    _, xyz = _fixture()
    with pytest.raises(ValueError, match="library_tile_mode"):
        _score(xyz, planar=False, library_tile_mode=0)

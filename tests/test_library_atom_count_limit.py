"""The largest library conformer scores; one atom more is refused, loudly.

Library counts are uint8 everywhere -- the corpus's n_real sidecar and
every backend's count vector -- so 255 atoms is the limit. Past it, a plain
``astype(np.uint8)`` wraps: 256 atoms became 0 and 257 became 1, scored and
reported with no error, while the CPU field kernels' fixed-size library record
held only 64 and read past it for anything larger. These tests pin both edges
of the limit at every public entry: the largest conformer must score
correctly, and the next size up must raise rather than wrap.
"""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3.layout import MAX_LIBRARY_COUNT, library_counts_u8

TOO_MANY = (MAX_LIBRARY_COUNT + 1, MAX_LIBRARY_COUNT + 2)


def _self_pair(n_atoms, width=None, seed=3):
    """One conformer of `n_atoms`, paired with itself; counts as int32."""
    rng = np.random.default_rng(seed)
    width = width or n_atoms
    coords = np.zeros((1, width, 4), np.float32)
    real = min(n_atoms, width)
    side = 1.5 * real ** (1 / 3)
    coords[0, :real, :3] = rng.uniform(-side / 2, side / 2, size=(real, 3))
    coords[0, :real, 3] = 1.0
    return dict(query_coords=coords,
                query_n_atoms=np.array([real], np.int32),
                library_coords=coords,
                library_counts=np.array([[n_atoms]], np.int32),
                pair_query_row=np.zeros(1, np.int32),
                pair_library_row=np.zeros(1, np.int32))


def test_the_limit_is_the_format_limit():
    """One number: what a corpus can store is what a backend accepts."""
    from roshambo3.prep.build import MAX_N_REAL
    assert MAX_LIBRARY_COUNT == MAX_N_REAL == np.iinfo(np.uint8).max


@pytest.mark.parametrize("n", TOO_MANY + (-1,))
def test_counts_outside_uint8_are_refused_not_wrapped(n):
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        library_counts_u8(np.array([[n]], np.int64))


def test_counts_inside_uint8_pass_unchanged():
    counts = np.array([[0], [64], [65], [MAX_LIBRARY_COUNT]], np.int64)
    np.testing.assert_array_equal(library_counts_u8(counts), counts)


class TestCPU:
    @pytest.fixture(autouse=True)
    def _cpu(self):
        pytest.importorskip("roshambo3._roshambo3_cpu")
        from roshambo3.backend import cpu
        self.cpu = cpu

    @pytest.mark.parametrize("field", [False, True], ids=["analytic", "field"])
    def test_the_largest_conformer_scores_itself(self, field):
        kw = dict(steps=25, lr_q=0.1, lr_t=0.4, start_mode=1)
        if field:
            kw.update(field_h=0.5, field_width=8, field_unroll=0,
                      field_pipeline=True, field_gridxf=True, field_allin=True)
        row = self.cpu.score_pairs(**_self_pair(MAX_LIBRARY_COUNT), **kw)
        assert row[0, 1] > 0.999, row[0, 1]

    @pytest.mark.parametrize("n", TOO_MANY)
    def test_one_atom_more_is_refused(self, n):
        with pytest.raises(ValueError, match="library_counts"):
            self.cpu.score_pairs(**_self_pair(n), steps=25, lr_q=0.1,
                                 lr_t=0.4, start_mode=0)


    def test_a_count_past_the_row_width_is_refused(self):
        """In range for uint8, but more atoms than the row holds: the scorer
        read the next row's atoms, or past the buffer for the last row."""
        with pytest.raises(ValueError, match="library_coords rows hold 40"):
            self.cpu.score_pairs(**_self_pair(80, width=40), steps=25,
                                 lr_q=0.1, lr_t=0.4, start_mode=0)

    def test_a_count_row_without_coordinates_is_refused(self):
        """Pair rows are bounded by the count rows, so extra count rows sent
        the scorer past the end of the coordinates."""
        pair = _self_pair(20)
        pair.update(library_counts=np.full((3, 1), 20, np.int32),
                    pair_query_row=np.zeros(3, np.int32),
                    pair_library_row=np.arange(3, dtype=np.int32))
        with pytest.raises(ValueError, match="library_counts has 3 rows"):
            self.cpu.score_pairs(**pair, steps=25, lr_q=0.1, lr_t=0.4,
                                 start_mode=0)

@pytest.mark.parametrize("n", TOO_MANY)
def test_cuda_refuses_one_atom_more_before_any_device_work(n):
    from roshambo3.backend import cuda
    try:
        cuda.score_pairs(**_self_pair(n), steps=25, lr_q=0.1, lr_t=0.4,
                         query_types=np.zeros((1, n), np.int32),
                         interaction_r=np.ones((1, 1), np.float32),
                         interaction_p=np.ones((1, 1), np.float32))
    except ValueError as error:
        assert "library_counts" in str(error)
    except ImportError:
        pytest.skip("CUDA wrapper validates after the extension check here")
    else:
        pytest.fail("a count past uint8 was accepted")

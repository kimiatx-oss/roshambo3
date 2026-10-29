"""The tile layout is a contract between the format, the producer and the kernel."""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.layout import BTILE, tile32


def test_btile_matches_the_kernel_header():
    """BTILE must equal ROSHAMBO3_BTILE in kernels.cuh.

    Nothing checks this at runtime -- a mismatch is silent data corruption, not
    a build error -- so it is checked here. Skips until the header exists.
    """
    import pathlib
    hdr = (pathlib.Path(__file__).resolve().parent.parent
           / "src/roshambo3/backend/_cuda_src/kernels.cuh")
    if not hdr.exists():
        pytest.skip("kernels.cuh not ported yet")
    import re
    m = re.search(r"#define\s+ROSHAMBO3_BTILE\s+(\d+)", hdr.read_text())
    assert m, "ROSHAMBO3_BTILE not found in kernels.cuh"
    assert int(m.group(1)) == BTILE


def test_max_features_matches_the_kernel_header():
    """MAX_FEATURES must equal ROSHAMBO3_MAX_FEATURES in kernels.cuh.

    It is the width of the per-conformer count vector the device reads, packed
    into one 64-bit word whose byte 7 carries the redundant color total. A
    mismatch would not be a build error either: the host would pack a byte the
    kernel never reads, or read one the host never wrote.
    """
    import pathlib
    import re

    from roshambo3.layout import MAX_FEATURES

    hdr = (pathlib.Path(__file__).resolve().parent.parent
           / "src/roshambo3/backend/_cuda_src/kernels.cuh")
    if not hdr.exists():
        pytest.skip("kernels.cuh not ported yet")
    m = re.search(r"#define\s+ROSHAMBO3_MAX_FEATURES\s+(\d+)", hdr.read_text())
    assert m, "ROSHAMBO3_MAX_FEATURES not found in kernels.cuh"
    assert int(m.group(1)) == MAX_FEATURES


@pytest.mark.parametrize("n", [1, 31, 32, 33, 64, 70])
@pytest.mark.parametrize("L", [8, 16, 40])
def test_tile32_round_trips(n, L):
    """Tiling reorders bytes; it must not change or lose any."""
    a = np.random.default_rng(n * 100 + L).random((n, L, 4)).astype(np.float32)
    t = tile32(a)
    n_tiles = t.shape[0] // BTILE
    assert t.shape == (n_tiles * BTILE, L, 4)
    back = (t.reshape(n_tiles, L, BTILE, 4)
             .transpose(0, 2, 1, 3)
             .reshape(-1, L, 4)[:n])
    assert np.array_equal(back, a)


def test_tile32_zero_pads_the_short_final_tile():
    a = np.ones((33, 8, 4), np.float32)
    t = tile32(a)
    assert t.shape[0] == 64
    n_tiles = 2
    back = t.reshape(n_tiles, 8, BTILE, 4).transpose(0, 2, 1, 3).reshape(-1, 8, 4)
    assert np.array_equal(back[:33], a)
    assert not back[33:].any(), "padding lanes must be zero"


def test_tile32_planar_moves_the_component_axis_inside_the_lane_axis():
    a = np.arange(32 * 2 * 4, dtype=np.float32).reshape(32, 2, 4)
    plain, planar = tile32(a), tile32(a, planar=True)
    assert plain.shape == planar.shape
    # Same values, different order.
    assert np.array_equal(np.sort(plain.ravel()), np.sort(planar.ravel()))
    assert not np.array_equal(plain, planar)

"""``pack_into`` writes the pinned buffer once; prove it writes the same bytes.

``pack_into`` replaced a four-traversal, two-allocation form with a single pass
that addresses the tiled destination directly. `doc/NUMERICS_POLICY.md` calls a
change to *where* a value lives free and a change to *when* it is added a
declared numerics change -- this is the former, and the way that claim is kept
honest is to compare bytes against an independent definition of the layout.

**The int16 wire record changed what the record IS**, so the oracle changed with it. The packer no
longer dequantizes anything: the wire record is the corpus's own int16
coordinates, three planes, laid into tile order. The independent definition of
that order is :func:`roshambo3.layout.tile32`, which is where the tile layout is
specified and which the packer deliberately does not call (calling it would
allocate the full-block temporary the single pass removed). Comparing against it is a
stronger check than the previous file's -- which compared the single pass
against the multi-pass form it replaced, both written by the same hand.

The float32 record the packer used to produce is not gone; it is still what
``combine_shape_color(..., elt=np.float32)`` and
``Group.iter_coords("f32x4")`` build, and
``test_the_wire_record_dequantizes_to_the_float32_record_it_replaced`` below
pins the two together at the one place it matters: what the kernel ends up
multiplying.

No GPU is needed. The comparison is over the host buffer, before anything is
uploaded, which is exactly where a layout change can be settled.
"""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3.engine.buffers import pack_into, require_wire_quantization
from roshambo3.layout import BTILE, tile32
from roshambo3.numerics import WIRE_QUANT_STEP


#: The poison a reused pinned buffer is pre-filled with, so a slot the packer
#: fails to write is visible rather than plausibly zero.
#:
#: It has to be a value the data cannot take, and that is a real constraint now
#: that the record IS the data: with float32 coordinates any sentinel was
#: potentially available outside the valid coordinate range. With int16
#: every representable coordinate is reachable, so the sentinel is the extreme
#: value and the generators below stop one integer short of it. A corpus
#: coordinate is +-13.4 A against the format's [-128, +127.99609375] A range, so nothing real
#: is being excluded.
POISON = np.int16(32767)
COORD_MAX = 30000


def _reference(xyz, L, tile=True):
    """The wire record, built the obvious way: transpose, then tile.

    ``tile32`` allocates and permutes; the packer addresses its destination
    directly and must reach the same bytes.
    """
    per_conformer = np.ascontiguousarray(xyz.transpose(0, 2, 1))
    if not tile:
        return per_conformer
    return tile32(per_conformer)


def _both(xyz, L, tile):
    nb = xyz.shape[0]
    n_rows = ((nb + BTILE - 1) // BTILE) * BTILE if tile else nb
    need = max(1, n_rows * L * 3)
    # Poisoned, not zeroed: a real pinned buffer is reused across blocks and
    # holds the previous block's data. A test on a zeroed buffer cannot see a
    # slot the new code fails to write, because zero is a plausible value.
    a = np.full(need, POISON, np.int16)
    got, nb_a = pack_into(xyz, L, a, WIRE_QUANT_STEP, tile=tile)
    want = _reference(xyz, L, tile=tile)
    assert nb_a == nb
    return got, want


# BTILE is 32; the interesting sizes are the ones that straddle it.
@pytest.mark.parametrize("nb", [0, 1, 2, 31, 32, 33, 63, 64, 65, 96, 100, 4096])
@pytest.mark.parametrize("L", [8, 16, 24, 32, 40])
@pytest.mark.parametrize("tile", [True, False])
def test_the_single_pass_writes_the_same_bytes(nb, L, tile):
    rng = np.random.default_rng(0xB10C ^ (nb * 1000 + L))
    xyz = rng.integers(-COORD_MAX, COORD_MAX, size=(nb, 3, L), dtype=np.int16)
    got, want = _both(xyz, L, tile)
    assert got.shape == want.shape
    assert np.array_equal(got, want), (
        f"nb={nb} L={L} tile={tile}: "
        f"{int((got != want).sum())} of {got.size} shorts differ")


def test_the_record_is_the_stored_shorts_unchanged():
    """Not merely equal to the reference -- equal to what is on disk.

    The byte comparison above would also pass if the packer and ``tile32``
    were wrong in the same way. This one goes back to the source array.
    """
    rng = np.random.default_rng(11)
    nb, L = 100, 24
    xyz = rng.integers(-COORD_MAX, COORD_MAX, size=(nb, 3, L), dtype=np.int16)
    need = ((nb + BTILE - 1) // BTILE) * BTILE * L * 3
    buf = np.full(need, POISON, np.int16)
    out, _ = pack_into(xyz, L, buf, WIRE_QUANT_STEP, tile=True)
    n_tiles = out.shape[0] // BTILE
    dst = out.reshape(n_tiles, L, BTILE, 3)
    for c in range(nb):
        t, lane = divmod(c, BTILE)
        assert np.array_equal(dst[t, :, lane, :], xyz[c].T)


def test_padding_lanes_are_zero_not_stale():
    """A partial last tile's unused lanes must be zeroed, not left poisoned.

    The multi-pass form zeroed rows `nb:` in untiled coordinates and relied
    on the permutation to carry them into place. Addressing the destination
    directly moves that responsibility into the packer, so it gets its own
    test rather than only being covered incidentally by the byte comparison.
    """
    nb, L = 33, 16                     # 33 conformers -> 2 tiles, 31 pad lanes
    rng = np.random.default_rng(5)
    xyz = rng.integers(-500, 500, size=(nb, 3, L), dtype=np.int16)
    need = 64 * L * 3
    buf = np.full(need, POISON, np.int16)
    out, _ = pack_into(xyz, L, buf, WIRE_QUANT_STEP, tile=True)
    dst = out.reshape(2, L, BTILE, 3)
    assert np.array_equal(dst[1, :, 1:, :], np.zeros((L, BTILE - 1, 3),
                                                     np.int16))


def test_the_wire_record_dequantizes_to_the_float32_record_it_replaced():
    """The whole basis of the int16 wire record, at the host end.

    The float32 record carried ``float32(v) * float32(WIRE_QUANT_STEP)``. The int16
    record carries ``v``, and the kernel forms
    ``ROSHAMBO3_STAGE_INV_Q_XYZ * (float)v`` -- the same product, written the
    other way round. Multiplication is commutative in IEEE-754, so the two are
    the same bits, and this asserts it over every representable value rather
    than over a sample.
    """
    v = np.arange(-32768, 32768, dtype=np.int16)
    old = v.astype(np.float32) * WIRE_QUANT_STEP
    new = WIRE_QUANT_STEP * v.astype(np.float32)
    assert np.array_equal(old.view(np.uint32), new.view(np.uint32))


def test_a_corpus_off_the_wire_step_is_refused():
    """The equality that is load-bearing now that nothing dequantizes on the
    host: the corpus's own quantization step must BE the wire record's.

    A corpus stored at any other step would have every coordinate scaled by the
    wrong constant on the device, with no other symptom -- every score
    plausible, every score wrong.
    """
    require_wire_quantization(WIRE_QUANT_STEP)          # the corpus format's step
    with pytest.raises(ValueError, match="wrong constant"):
        require_wire_quantization(0.0005)
    xyz = np.zeros((2, 3, 8), np.int16)
    with pytest.raises(ValueError, match="wrong constant"):
        pack_into(xyz, 8, np.zeros(2 * 8 * 3 * 32, np.int16), 0.002)


def test_a_float32_buffer_is_refused():
    """The pool stages int16. A float32 buffer would silently
    reinterpret every short as a quarter of a float."""
    xyz = np.zeros((2, 3, 8), np.int16)
    with pytest.raises(ValueError, match="int16"):
        pack_into(xyz, 8, np.zeros(4096, np.float32), WIRE_QUANT_STEP)


# ---------------------------------------------------------------------------
# Packing several blocks into one buffer, at an offset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [(4096, 4096), (32, 32, 32), (64, 33),
                                   (4096, 4096, 1808)])
@pytest.mark.parametrize("L", [16, 24, 40])
def test_packing_at_an_offset_writes_the_same_bytes_as_packing_alone(sizes, L):
    """A coalesced launch is one buffer holding several on-disk blocks.

    ``pack_into`` writes the *prefix* of whatever buffer it is handed, so the
    whole of the coalesced decode is ``pack_into(..., dst_flat=buf[off:])`` with
    ``off = rows_so_far * L * 3``. That is only the same bytes if the
    destination's tile addressing depends on nothing but the slice it was
    given -- which is what this checks, block for block, against each block
    packed alone into its own buffer.

    The last block may be short (a group's tail); every earlier one is a whole
    number of tiles, which is what makes ``off`` tile-aligned and, on the
    device, what makes ``gid = gid_base + row`` contiguous across the run.
    """
    rng = np.random.default_rng(0x0FF5E7 ^ (L * 31 + len(sizes)))
    blocks = [rng.integers(-COORD_MAX, COORD_MAX, size=(nb, 3, L), dtype=np.int16)
              for nb in sizes]

    total = sum(sizes)
    rows_total = ((total + BTILE - 1) // BTILE) * BTILE
    # Poisoned for the same reason as `_both` above.
    big = np.full(rows_total * L * 3, POISON, np.int16)

    off_rows = 0
    for xyz, nb in zip(blocks, sizes):
        alone = np.full((((nb + BTILE - 1) // BTILE) * BTILE) * L * 3,
                        POISON, np.int16)
        want, _ = pack_into(xyz, L, alone, WIRE_QUANT_STEP, tile=True)
        got, _ = pack_into(xyz, L, big[off_rows * L * 3:], WIRE_QUANT_STEP, tile=True)
        assert np.array_equal(got, want), f"block of {nb} at row {off_rows}"
        off_rows += ((nb + BTILE - 1) // BTILE) * BTILE

    assert off_rows == rows_total, (
        "the padded total and the sum of the per-block padded rows must agree, "
        "or the run has interior padding and every later gid is shifted")
    assert not np.any(big == POISON), "a short was left unwritten"


# ---------------------------------------------------------------------------
# The planar tile.
#
# `[tile][atom][component][lane]` instead of `[tile][atom][lane][component]`.
# The buffer is the same size and holds the same values; only the stride
# between a row's three components changes, from 1 to BTILE. Everything above
# is repeated against `tile32(planar=True)` for the same reason it is written
# against `tile32` at all: the packer deliberately does not call it, so an
# independent definition of the order is what keeps the claim honest.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nb", [0, 1, 2, 31, 32, 33, 63, 64, 65, 96, 100, 4096])
@pytest.mark.parametrize("L", [8, 16, 24, 32, 40])
def test_the_planar_pass_writes_the_same_bytes(nb, L):
    rng = np.random.default_rng(0x89AA ^ (nb * 1000 + L))
    xyz = rng.integers(-COORD_MAX, COORD_MAX, size=(nb, 3, L), dtype=np.int16)
    n_rows = ((nb + BTILE - 1) // BTILE) * BTILE
    buf = np.full(max(1, n_rows * L * 3), POISON, np.int16)
    got, nb_a = pack_into(xyz, L, buf, WIRE_QUANT_STEP, tile=True, planar=True)
    want = tile32(np.ascontiguousarray(xyz.transpose(0, 2, 1)), planar=True)
    assert nb_a == nb
    assert got.shape == want.shape
    assert np.array_equal(got, want), (
        f"nb={nb} L={L}: {int((got != want).sum())} of {got.size} shorts differ")


def test_the_planar_record_is_the_stored_shorts_unchanged():
    """Back to the source array, so packer and ``tile32`` cannot agree wrongly."""
    rng = np.random.default_rng(0x89BB)
    nb, L = 100, 24
    xyz = rng.integers(-COORD_MAX, COORD_MAX, size=(nb, 3, L), dtype=np.int16)
    need = ((nb + BTILE - 1) // BTILE) * BTILE * L * 3
    buf = np.full(need, POISON, np.int16)
    out, _ = pack_into(xyz, L, buf, WIRE_QUANT_STEP, tile=True, planar=True)
    n_tiles = out.shape[0] // BTILE
    dst = out.reshape(n_tiles, L, 3, BTILE)
    for c in range(nb):
        t, lane = divmod(c, BTILE)
        assert np.array_equal(dst[t, :, :, lane], xyz[c].T)


def test_the_planar_tile_is_a_permutation_of_the_interleaved_one():
    """Same multiset of shorts, different order -- which is the whole claim.

    If the two layouts ever stopped being a relabelling of one byte for one
    byte, "the same values are read in the same order" would stop being true
    and bit-identity would stop being the right bar.
    """
    rng = np.random.default_rng(0x89CC)
    nb, L = 70, 16
    xyz = rng.integers(-COORD_MAX, COORD_MAX, size=(nb, 3, L), dtype=np.int16)
    need = 96 * L * 3
    a = np.full(need, POISON, np.int16)
    b = np.full(need, POISON, np.int16)
    i, _ = pack_into(xyz, L, a, WIRE_QUANT_STEP, tile=True, planar=False)
    p, _ = pack_into(xyz, L, b, WIRE_QUANT_STEP, tile=True, planar=True)
    assert i.shape == p.shape
    assert not np.array_equal(i, p)          # it really is a different order
    n_tiles = i.shape[0] // BTILE
    assert np.array_equal(i.reshape(n_tiles, L, BTILE, 3).transpose(0, 1, 3, 2),
                          p.reshape(n_tiles, L, 3, BTILE))


def test_planar_padding_lanes_are_zero_not_stale():
    nb, L = 33, 16                     # 33 conformers -> 2 tiles, 31 pad lanes
    rng = np.random.default_rng(0x89DD)
    xyz = rng.integers(-500, 500, size=(nb, 3, L), dtype=np.int16)
    buf = np.full(64 * L * 3, POISON, np.int16)
    out, _ = pack_into(xyz, L, buf, WIRE_QUANT_STEP, tile=True, planar=True)
    dst = out.reshape(2, L, 3, BTILE)
    assert np.array_equal(dst[1, :, :, 1:], np.zeros((L, 3, BTILE - 1),
                                                     np.int16))

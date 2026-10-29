"""The int16 library record on the wire, and the two things that make it a
change rather than a rename.

**1. It is the same arithmetic.** `volume_single`, `get_gradient` and both pair
kernels take the library record's element type as a template parameter, with
`r3_fetch` / `r3_quantize` / `r3_weight` the only places that type is visible.
So the int16 arm and the float32 arm are one code path with two loads, and on a
library whose coordinates are ON the corpus's wire-step grid they must return the
same bits -- not close, the same. That is asserted here on both kernels, at
several start modes, with no corpus: the fixture is built from int16 and the
float32 record is derived from it, which is exactly the relationship a corpus
block has.

Why it is exact, stated so the test is not mistaken for a tolerance: the float32
record carried `float32(v) * float32(WIRE_QUANT_STEP)` and the int16 record carries `v`,
from which the kernel forms `ROSHAMBO3_STAGE_INV_Q_XYZ * (float)v` -- the same
product written the other way round, and multiplication is commutative in
IEEE-754. Every device-side re-quantization the storage tiers perform is then
the identity, exhaustively over all 65,536 values.

**2. It is load-bearing.** A kernel that had quietly stopped reading the int16
plane -- or that read it at the wrong stride, which is the failure the record
width change makes possible -- would pass every test above. So a deliberately
corrupted plane must CHANGE the score, in the spirit of a shifted count
vector control. Corrupting one component of one atom of one conformer is enough, and
that conformer's score is where it has to show.
"""

from __future__ import annotations

from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np
import pytest

from roshambo3 import scores as S
from roshambo3.backend import cuda as backend
from roshambo3.backend.cuda import PoseSearch

pytestmark = pytest.mark.skipif(not backend.available(),
                                reason="no CUDA extension in this build")

L = 24
N_QUERIES = 3
N_LIBRARY = 64
N_REAL_LO, N_REAL_HI = 12, 24


def _fixture(seed: int = 20260826):
    """A library ON the corpus grid, in both records.

    Geometry molecule-shaped rather than uniform noise, for the reason
    `bench/make_golden.py` gives: a cloud of unrelated points scores near zero
    everywhere and the optimizer has no basin to find, so a difference in the
    scoring path would have nothing to show up in.
    """
    from roshambo3.layout import tile32

    rng = np.random.default_rng(seed)
    n_rows = N_QUERIES + N_LIBRARY
    n_real = rng.integers(N_REAL_LO, N_REAL_HI + 1, size=n_rows).astype(np.int32)

    # int16 planes, [n, 3, L], exactly as the corpus stores them.
    planes = np.zeros((n_rows, 3, L), np.int16)
    for i in range(n_rows):
        k = int(n_real[i])
        step = rng.normal(0.0, 1.0, size=(k, 3))
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        xyz = np.cumsum(step * 1.5, axis=0)
        xyz -= xyz.mean(axis=0)
        planes[i, :, :k] = np.rint(xyz.T  / WIRE_QUANT_STEP).astype(np.int16)

    # The QUERY record is float32 either way -- only the library moves.
    qc = np.zeros((N_QUERIES, L, 4), np.float32)
    qc[:, :, :3] = (planes[:N_QUERIES].transpose(0, 2, 1).astype(np.float32)
                    * WIRE_QUANT_STEP)
    cols = np.arange(L)[None, :]
    qc[:, :, 3] = (cols < n_real[:N_QUERIES, None]).astype(np.float32)

    lib16 = tile32(np.ascontiguousarray(
        planes[N_QUERIES:].transpose(0, 2, 1)))            # [n, L, 3] int16
    lib32 = np.zeros((N_LIBRARY, L, 4), np.float32)
    lib32[:, :, :3] = (planes[N_QUERIES:].transpose(0, 2, 1).astype(np.float32)
                       * WIRE_QUANT_STEP)
    lib32[:, :, 3] = (cols < n_real[N_QUERIES:, None]).astype(np.float32)
    lib32 = tile32(lib32)

    common = dict(
        query_coords=qc,
        query_types=np.zeros((N_QUERIES, L), np.int32),
        query_n_atoms=np.ascontiguousarray(n_real[:N_QUERIES]),
        library_counts=n_real[N_QUERIES:].astype(np.uint8).reshape(-1, 1),
        pair_query_row=np.repeat(np.arange(N_QUERIES, dtype=np.int32),
                                 N_LIBRARY),
        pair_library_row=np.tile(np.arange(N_LIBRARY, dtype=np.int32),
                                 N_QUERIES),
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.zeros((1, 1), np.float32),
    )
    return common, lib16, lib32


def _run(common, lib, **kw):
    from roshambo3.backend import score_pairs

    args = dict(steps=25, lr_t=0.4, lr_q=0.1, start_mode=1,
                pose_search=PoseSearch.FIELD)
    args.update(kw)
    return score_pairs(library_coords=lib, **common, **args)


@pytest.fixture(scope="module")
def fixture():
    return _fixture()


@pytest.mark.parametrize("pose_search", [PoseSearch.FIELD, PoseSearch.ANALYTIC])
@pytest.mark.parametrize("start_mode", [0, 1, 2])
def test_the_int16_record_reproduces_the_float32_one_bit_for_bit(
        fixture, pose_search, start_mode):
    common, lib16, lib32 = fixture
    a = _run(common, lib16, pose_search=pose_search, start_mode=start_mode)
    b = _run(common, lib32, pose_search=pose_search, start_mode=start_mode)
    assert a.shape == b.shape
    # Bit for bit, on all 20 columns. A tolerance here would defeat the point:
    # the claim is that the element type is a load and not an expression.
    diff = int((a.view(np.uint32) != b.view(np.uint32)).sum())
    assert diff == 0, (
        f"{diff} of {a.size} score words differ between the int16 and the "
        f"float32 library record at pose_search={pose_search}, "
        f"start_mode={start_mode}")


def test_the_two_records_carry_the_same_coordinates():
    """The premise of the test above, checked separately so that a fixture bug
    cannot make the bit-identity claim vacuous."""
    _, lib16, lib32 = _fixture()
    assert lib16.dtype == np.int16 and lib16.shape[2] == 3
    assert lib32.dtype == np.float32 and lib32.shape[2] == 4
    assert np.array_equal(
        (lib16.astype(np.float32) * WIRE_QUANT_STEP).view(np.uint32),
        lib32[:, :, :3].view(np.uint32))


class TestTheInt16PlaneIsLoadBearing:
    """A shifted-count-vector control, for the coordinate record.

    A kernel that had stopped reading the int16 plane, or that read it at the
    float32 record's stride, would pass every agreement test above -- the
    float32 arm would still be right. These are the negative controls that
    would not.
    """

    @pytest.mark.parametrize("pose_search",
                             [PoseSearch.FIELD, PoseSearch.ANALYTIC])
    def test_corrupting_one_coordinate_changes_that_conformer_score(
            self, fixture, pose_search):
        common, lib16, _ = fixture
        clean = _run(common, lib16, pose_search=pose_search)

        # One component of one real atom of library row 0, moved 5 A. Row 0 is
        # tile 0 lane 0, so [atom][lane][component] puts it at [0, 0, 0].
        bad = lib16.copy()
        bad[0, 0, 0] = np.int16(int(bad[0, 0, 0]) + 5000)
        dirty = _run(common, bad, pose_search=pose_search)

        # Every pair involving library row 0 must move, and no other pair may.
        moved = np.any(clean.view(np.uint32) != dirty.view(np.uint32), axis=1)
        rows_of_zero = common["pair_library_row"] == 0
        assert moved[rows_of_zero].all(), (
            "moving a library atom 5 A left its own conformer's score "
            "unchanged -- the coordinate plane is not being read")
        assert not moved[~rows_of_zero].any(), (
            "moving one library atom changed another conformer's score -- the "
            "record is being read at the wrong stride")

    def test_transposing_the_last_two_components_changes_the_score(self, fixture):
        """A three-component record read at the four-component stride reads a
        neighboring lane's x as this lane's w and shifts everything after it.
        A swap inside the record is the cheap end of the same family and must
        also be visible."""
        common, lib16, _ = fixture
        clean = _run(common, lib16)
        bad = lib16.copy()
        bad[:, :, [1, 2]] = bad[:, :, [2, 1]]
        dirty = _run(common, bad)
        assert np.any(clean.view(np.uint32) != dirty.view(np.uint32))


class TestTheRecordShapeIsChecked:
    def test_an_int16_record_of_the_wrong_width_is_refused(self, fixture):
        common, lib16, _ = fixture
        wide = np.zeros(lib16.shape[:2] + (4,), np.int16)
        with pytest.raises(ValueError, match=r"\[n_rows, L, 3\]"):
            _run(common, wide)

    def test_a_float32_record_of_the_wrong_width_is_refused(self, fixture):
        common, _, lib32 = fixture
        narrow = np.zeros(lib32.shape[:2] + (3,), np.float32)
        with pytest.raises(ValueError, match=r"\[n_rows, L, 4\]"):
            _run(common, narrow)

"""Start-per-lane (`S`) on the field kernel: row coverage, and its refusals.

`S` lane groups would share one library row and split its start orientations
between them, so a block covers ``NTHREADS / (WA * S)`` distinct rows instead
of ``NTHREADS / WA``. Three numbers have to agree for that to be right -- the
kernel's ``rgrp = (threadIdx.x / WA) / S``, the work-item ``chunk`` built in
``interface.cpp``, and the staged buffer's group stride ``SPAD`` -- and **when
they disagree the symptom is a missing or duplicated library row, not a crash**.
Nothing faults, nothing raises, and the scores that do come back are
individually correct. So coverage is asserted directly, at the shape path's
shipped ``S = 1``.

The shape-only field path compiles ``S = 1`` only; ``S = 4`` is compiled for
the combination schedule alone. ``ScreenConfig.start_groups_per_row`` stays
unset and resolves per mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.backend import cuda as backend
from roshambo3.backend.cuda import CoordResidency, PoseSearch, UpdateRule

pytestmark = pytest.mark.skipif(not backend.available(),
                                reason="no CUDA extension in this build")

#: The `S` the shape-only field path compiles.
SHAPE_S = (1,)


def _fixture(n_library: int):
    """The `bench/make_golden.py` seeded input, cut to `n_library` rows.

    Seeded and molecule-shaped rather than uniform noise, and it needs no
    corpus, so this file runs anywhere a GPU does. The row count is a caller
    argument because the coverage claim is about the WORK-ITEM TAIL: with
    NTHREADS = 256 and WA = 4 a work item covers 64/S rows, so a library count
    that is a multiple of 64 exercises only full items and would pass a wrong
    `chunk` at S = 1, 2 and 4 alike.
    """
    import bench.make_golden as mk
    from roshambo3.layout import tile32

    coords, n_real = mk.make_input()
    nq, L = mk.N_QUERIES, mk.L
    assert n_library <= mk.N_LIBRARY
    qc = np.ascontiguousarray(coords[:nq])
    qn = np.ascontiguousarray(n_real[:nq])
    lib = np.ascontiguousarray(coords[nq:nq + n_library])
    tc = tile32(lib)
    tn = np.ascontiguousarray(n_real[nq:nq + n_library])
    return dict(
        query_coords=qc,
        query_types=np.zeros((nq, L), np.int32),
        query_n_atoms=qn,
        library_coords=tc,
        # A shape-only library's count vector is one column -- the real
        # atom count -- and there is no type plane at all.
        library_counts=tn.astype(np.uint8).reshape(-1, 1),
        pair_query_row=np.repeat(np.arange(nq, dtype=np.int32), n_library),
        pair_library_row=np.tile(np.arange(n_library, dtype=np.int32), nq),
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.zeros((1, 1), np.float32),
    )


def _run(fx, *, out=None, **kw):
    from roshambo3.backend import score_pairs

    args = dict(steps=25, lr_t=0.2, lr_q=0.1, pose_search=PoseSearch.FIELD,
                start_mode=2, update_rule=UpdateRule.ADAGRAD_EXP)
    args.update(kw)
    return score_pairs(**fx, out=out, **args)


# ---------------------------------------------------------------------------
# 1. Row coverage. This runs before every other claim in this file.
# ---------------------------------------------------------------------------
class TestEveryLibraryRowIsScoredExactlyOnce:
    """The failure `S` invites is silent: a `chunk` that disagrees with the
    kernel's row count drops rows off the end of each work item or writes the
    next item's rows twice. Neither raises, and the rows that *are* written are
    right, so a score comparison alone would report a clean pass over a subset.

    The detector is a sentinel: `score_pairs` writes into a caller-supplied
    `out`, so the buffer is pre-filled with NaN and any row the kernel never
    touched is still NaN afterwards. That catches "dropped". "Duplicated" is
    caught by the same assertion in a different guise -- a block that writes a
    row twice has NOT written some other row, because the total number of
    (block, row-group) slots is fixed -- and by the identity check below, which
    pins each output row to its OWN library row rather than to some other row's
    perfectly valid score.
    """

    # 200 is not a multiple of 64, so the last work item is a partial tail,
    # which is the case a wrong `chunk` survives.
    N_LIBRARY = 200

    @pytest.mark.parametrize("S", SHAPE_S)
    def test_no_row_is_left_unwritten(self, S):
        fx = _fixture(self.N_LIBRARY)
        n_pairs = fx["pair_query_row"].size
        out = np.full((n_pairs, 20), np.nan, np.float32)
        _run(fx, out=out, start_groups_per_row=S)
        missing = np.flatnonzero(np.isnan(out[:, 0]))
        assert missing.size == 0, (
            f"S={S}: {missing.size} of {n_pairs} output rows were never "
            f"written (first: {missing[:8].tolist()}). The kernel's row count "
            f"NTHREADS/(WA*S) and interface.cpp's work-item `chunk` disagree.")

    @pytest.mark.parametrize("S", SHAPE_S)
    def test_each_output_row_carries_its_own_library_rows_self_overlap(self, S):
        """Coverage without identity is not coverage.

        Column 6 is the library conformer's own shape self-overlap, a per-row
        constant the kernel copies from `vb_all[trow]`. If a work item's rows
        are shifted -- the arithmetic error a wrong `chunk` actually produces --
        every score in the block is a valid score of the WRONG pair, and only a
        per-row identity like this one sees it.
        """
        fx = _fixture(self.N_LIBRARY)
        ref = _run(fx, start_groups_per_row=1)
        got = _run(fx, start_groups_per_row=S)
        assert np.array_equal(got[:, 6], ref[:, 6]), (
            f"S={S}: the per-library-row self-overlap column does not line up "
            f"with S=1, so output rows are carrying another row's pair.")

    @pytest.mark.parametrize("n_library", [1, 7, 63, 64, 65, 200, 512])
    def test_coverage_holds_at_every_work_item_boundary(self, n_library):
        """Sweeping the row count rather than `S` is what makes the
        work-item tail the variable."""
        S = 1
        fx = _fixture(n_library)
        n_pairs = fx["pair_query_row"].size
        out = np.full((n_pairs, 20), np.nan, np.float32)
        _run(fx, out=out, start_groups_per_row=S)
        assert not np.isnan(out[:, 0]).any(), (
            f"S={S}, n_library={n_library}: unwritten rows at the tail.")


# ---------------------------------------------------------------------------
# 2. What the machinery refuses, and why each refusal exists.
# ---------------------------------------------------------------------------
class TestTheRefusalsAreByNameNotByAccident:

    def _fx(self):
        return _fixture(64)

    @pytest.mark.parametrize("S", [0, 3, 5, 6, 16, 32])
    def test_a_non_power_of_two_or_out_of_range_S_is_refused(self, S):
        with pytest.raises(Exception, match="start_groups_per_row"):
            _run(self._fx(), start_groups_per_row=S)

    def test_S_beyond_the_start_count_is_refused_rather_than_idled(self):
        """At start_mode 1 there are four starts, so S = 8 would give four
        groups a row and no start to run on it. That is a caller mistake with
        no legitimate reading, so it is named rather than silently wasted."""
        with pytest.raises(Exception, match="exceeds the 4 starts"):
            _run(self._fx(), start_mode=1, start_groups_per_row=8)

    def test_the_register_tier_refuses_S_with_its_reason(self):
        """TIER 3 holds coordinates per LANE, so the S groups sharing a row
        still hold their own copies: the tier has no automatic saving to
        offer, and there is no distribute-and-shuffle design that
        would give it one. Refused by name rather than instantiated as 36 more
        cells of a table whose entries would all read 'no change'."""
        with pytest.raises(Exception, match="REGISTER"):
            _run(self._fx(), start_groups_per_row=2,
                 coord_residency=CoordResidency.REGISTER,
                 smem_carveout=0)

    def test_the_analytic_path_refuses_S(self):
        """The analytic kernel gives one thread a whole pair; there is no lane
        group to share a row and no start split to make. Silently ignoring the
        argument there is a known failure mode -- a resolved default becoming
        an ignored argument."""
        with pytest.raises(Exception, match="analytic kernel"):
            _run(self._fx(), pose_search=PoseSearch.ANALYTIC,
                 start_groups_per_row=2)


# ---------------------------------------------------------------------------
# 3. Color mode 3 selects its own operating point while retaining the
#    shape-only default.
# ---------------------------------------------------------------------------
def test_start_mode_default_resolves_by_mode():
    from roshambo3.engine.config import ScreenConfig

    shape = ScreenConfig()
    combo = ScreenConfig(color=True)
    assert shape.start_groups_per_row is None
    assert combo.start_groups_per_row is None
    assert shape.for_width(32)["start_groups_per_row"] is None
    assert combo.for_width(32)["start_groups_per_row"] is None

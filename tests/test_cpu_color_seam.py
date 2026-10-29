"""CPU color boundary validation and phase timers.

Three things pinned here:

* **Invalid color input fails before dispatch** (`layout.check_color_layout`,
  called from `backend.cpu.score_pairs`) -- never silently scored shape-only.
* Valid scalar color input now reaches the scalar color path; malformed input still fails
  before dispatch.
* **Phase timers price a call without changing it.** Every score is
  bit-identical whether `with_phase_timers` is on or off.
"""
from __future__ import annotations

import numpy as np
import pytest

cpu = pytest.importorskip("roshambo3.backend.cpu")
pytestmark = pytest.mark.skipif(not cpu.available(),
                                reason="the CPU kernel is not built")


def _shape_only_fixture(n_q=3, n_l=5, L=10, seed=7):
    rng = np.random.default_rng(seed)
    qc = np.zeros((n_q, L, 4), np.float32)
    qn = np.zeros(n_q, np.int32)
    for i in range(n_q):
        n = int(rng.integers(4, L))
        qc[i, :n, :3] = rng.standard_normal((n, 3))
        qc[i, :n, 3] = 1.0
        qn[i] = n
    lc = np.zeros((n_l, L, 4), np.float32)
    lcnt = np.zeros((n_l, 1), np.uint8)
    for i in range(n_l):
        n = int(rng.integers(4, L))
        lc[i, :n, :3] = rng.standard_normal((n, 3))
        lc[i, :n, 3] = 1.0
        lcnt[i, 0] = n
    pq = np.repeat(np.arange(n_q, dtype=np.int32), n_l)
    pl = np.tile(np.arange(n_l, dtype=np.int32), n_q)
    return dict(query_coords=qc, query_n_atoms=qn, library_coords=lc,
               library_counts=lcnt, pair_query_row=pq, pair_library_row=pl)


def _color_fixture(n_features=3):
    """A small, valid color-shaped input: shared enumeration, contiguous
    query types, room in L for both the shape and color runs."""
    qc = np.zeros((1, 8, 4), np.float32)
    qc[0, :4, :3] = 1.0
    qc[0, :4, 3] = 1.0
    qn = np.array([4], np.int32)
    qt = np.array([[0, 0, 0, 0, 1, 2, 0, 0]], np.int32)
    lc = np.zeros((1, 8, 4), np.float32)
    lc[0, :5, :3] = 1.0
    lc[0, :5, 3] = 1.0
    lcnt = np.array([[4, 1, 0]], np.uint8)
    r = np.eye(n_features)
    p = np.eye(n_features)
    return dict(query_coords=qc, query_types=qt, query_n_atoms=qn,
               library_coords=lc, library_counts=lcnt,
               interaction_r=r, interaction_p=p,
               pair_query_row=np.array([0], np.int32),
               pair_library_row=np.array([0], np.int32))


class TestTheRefusalOrder:
    """Validate before dispatch, never silently falling back to shape-only."""

    def test_valid_color_input_is_scored(self):
        out = cpu.score_pairs(color=True, mixing=0.5, steps=5,
                              **_color_fixture())
        assert np.isfinite(out).all()
        assert out[0, 2] > 0 and out[0, 7] > 0

    def test_missing_query_types_fails_before_the_refusal(self):
        kw = _color_fixture()
        kw["query_types"] = None
        with pytest.raises(ValueError, match="requires query_types"):
            cpu.score_pairs(color=True, steps=5, **kw)

    def test_missing_interaction_matrices_fails_before_the_refusal(self):
        kw = _color_fixture()
        kw["interaction_r"] = None
        kw["interaction_p"] = None
        with pytest.raises(ValueError, match="requires interaction_r"):
            cpu.score_pairs(color=True, steps=5, **kw)

    def test_malformed_interaction_matrix_fails_with_its_own_message(self):
        kw = _color_fixture()
        kw["interaction_r"] = np.array([[np.nan]])
        kw["interaction_p"] = np.array([[0.0]])
        with pytest.raises(ValueError, match="finite"):
            cpu.score_pairs(color=True, steps=5, **kw)

    def test_enumeration_mismatch_fails_with_its_own_message(self):
        kw = _color_fixture()
        # library_counts carries one more column than the matrix admits.
        kw["library_counts"] = np.array([[4, 1, 0, 0]], np.uint8)
        with pytest.raises(ValueError, match="SAME feature enumeration"):
            cpu.score_pairs(color=True, steps=5, **kw)

    def test_a_count_vector_claiming_more_than_l_fails(self):
        kw = _color_fixture()
        kw["library_counts"] = np.array([[4, 5, 0]], np.uint8)  # 9 > L=8
        with pytest.raises(ValueError, match="L="):
            cpu.score_pairs(color=True, steps=5, **kw)

    def test_a_noncontiguous_library_feature_run_is_refused(self):
        kw = _color_fixture()
        # Occupy a slot past n_real(4) + n_color(1) = 5.
        kw["library_coords"][0, 7, :3] = 1.0
        kw["library_coords"][0, 7, 3] = 1.0
        with pytest.raises(ValueError, match="contiguous heavy-atom prefix followed by the color features"):
            cpu.score_pairs(color=True, steps=5, **kw)

    def test_a_padding_gap_in_the_query_color_run_is_refused(self):
        """The query-side counterpart of the split-record check above: a
        padding (type 0) slot BEFORE the last real feature slot, which
        `layout.counts_from_types` used to accept silently (filtering zeros
        before checking contiguity) and which made both this seam's decoder
        and Python's agree on a coordinate-wrong view. Reaches this refusal
        through `score_pairs`'s own boundary validation, not just the
        decoder directly."""
        kw = _color_fixture()
        # _color_fixture's query type plane is [0,0,0,0,1,2,0,0], n_real=4,
        # real features at slots 4 and 5. Move the second one out past a
        # padding gap: [0,0,0,0,1,0,2,0].
        kw["query_types"] = np.array([[0, 0, 0, 0, 1, 0, 2, 0]], np.int32)
        with pytest.raises(ValueError, match="padding gap"):
            cpu.score_pairs(color=True, steps=5, **kw)

    def test_mixing_without_color_is_still_refused(self):
        """Shape-only cannot silently discount its score by mixing."""
        kw = _shape_only_fixture()
        with pytest.raises(ValueError, match="shape_only mode requires mixing=0"):
            cpu.score_pairs(mixing=0.5, steps=5, **kw)

    def test_shape_only_default_is_unaffected(self):
        """The refusal wiring must not perturb the ordinary shape-only path
        at all -- no query_types, no interaction matrices, mixing=0."""
        scores = cpu.score_pairs(steps=5, **_shape_only_fixture())
        assert scores.shape[1] == 20
        assert np.isfinite(scores).all()


class TestPhaseTimersPriceWithoutChangingTheResult:
    def test_single_threaded_scores_are_bit_identical(self):
        kw = _shape_only_fixture()
        base = cpu.score_pairs(steps=25, start_mode=1, threads=1, **kw)
        timed, timers = cpu.score_pairs(steps=25, start_mode=1, threads=1,
                                        with_phase_timers=True, **kw)
        assert np.array_equal(base, timed)
        for key in ("query_prep", "query_self_overlap",
                   "library_self_overlap", "field_construction",
                   "shape_search", "analytic_color", "field_color",
                   "exact_final_rescore", "reduction"):
            assert key in timers
            assert timers[key] >= 0.0
        # Shape-only still leaves the reserved color phases at zero.
        assert timers["analytic_color"] == 0.0
        assert timers["field_color"] == 0.0
        assert timers["exact_final_rescore"] == 0.0
        assert timers["reduction"] == 0.0
        # The phases that DO run cost something real -- not asserted as
        # "positive" (a sub-microsecond phase can round to 0.0 on a coarse
        # clock), but the dominant one (the search) must be.
        assert timers["shape_search"] > 0.0

    def test_multi_threaded_scores_are_bit_identical_too(self):
        kw = _shape_only_fixture(n_q=4, n_l=16, L=12)
        base = cpu.score_pairs(steps=25, start_mode=1, threads=4, **kw)
        timed, _ = cpu.score_pairs(steps=25, start_mode=1, threads=4,
                                   with_phase_timers=True, **kw)
        assert np.array_equal(base, timed)

    def test_field_construction_is_priced_when_the_field_path_runs(self):
        kw = _shape_only_fixture(n_q=1, n_l=2, L=12)
        _, timers = cpu.score_pairs(steps=5, start_mode=0, field_h=0.6,
                                    with_phase_timers=True, **kw)
        assert timers["field_construction"] > 0.0

    def test_census_and_phase_timers_are_mutually_exclusive(self):
        kw = _shape_only_fixture()
        with pytest.raises(ValueError, match="cannot be combined"):
            cpu.score_pairs(steps=5, with_census=True,
                            with_phase_timers=True, **kw)

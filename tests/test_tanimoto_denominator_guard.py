"""Inconsistent self-overlaps are clamped and refused on every CPU shape path."""
from __future__ import annotations

import platform

import numpy as np
import pytest

from functools import partial

from roshambo3 import scores
from roshambo3.scores import (N_SCORE_COLUMNS, OVERLAP_VOLUME,
                              SELF_OVERLAP_BOUND_MARGIN, SELF_OVERLAP_LIBRARY,
                              SELF_OVERLAP_QUERY, InconsistentSelfOverlapError)

# `route` is required; these rows exercise the shipped stored-frame
# contract.
check_self_overlap_bound = partial(scores.check_self_overlap_bound,
                                   route=scores.ROUTE_VERIFIED_STORED)


def _row(overlap, self_query, self_library):
    r = np.zeros((1, N_SCORE_COLUMNS), np.float32)
    r[0, OVERLAP_VOLUME] = overlap
    r[0, SELF_OVERLAP_QUERY] = self_query
    r[0, SELF_OVERLAP_LIBRARY] = self_library
    return r


class TestTheBound:
    def test_consistent_and_empty_rows_pass(self):
        check_self_overlap_bound(_row(90.0, 100.0, 100.0))
        check_self_overlap_bound(_row(100.0, 100.0, 100.0))   # self-match
        check_self_overlap_bound(_row(0.0, 0.0, 0.0))         # empty pair
        check_self_overlap_bound(np.zeros((0, N_SCORE_COLUMNS), np.float32))

    def test_rounding_inside_the_shipped_margin_passes(self):
        check_self_overlap_bound(
            _row(100.0 * (1 + SELF_OVERLAP_BOUND_MARGIN * .5), 100.0, 100.0))

    def test_a_tanimoto_above_one_is_refused(self):
        # V_AB = 60 against selves of 100 and 1: Tanimoto 60/41.
        with pytest.raises(InconsistentSelfOverlapError, match="conformer 7"):
            check_self_overlap_bound(_row(60.0, 100.0, 1.0), [7])

    def test_a_non_positive_denominator_is_refused(self):
        # V_AB >= V_AA + V_BB: reported as Tanimoto 0, so it ranks last and no
        # check of a returned top-K would see it.
        with pytest.raises(InconsistentSelfOverlapError):
            check_self_overlap_bound(_row(120.0, 100.0, 1.0))

    def test_the_margin_is_where_it_says(self):
        edge = 100.0 * (1 + SELF_OVERLAP_BOUND_MARGIN)
        check_self_overlap_bound(_row(edge * (1 - 1e-6), 100.0, 100.0))
        with pytest.raises(InconsistentSelfOverlapError):
            check_self_overlap_bound(_row(edge * (1 + 1e-5), 100.0, 100.0))


def test_the_engine_checks_the_rows_it_returns():
    """The device backends' whole check: they compute self-overlaps in-kernel."""
    from roshambo3.engine.engine import _check_returned_rows
    good = (_row(90.0, 100.0, 100.0), np.array([3]))
    bad = (_row(60.0, 100.0, 1.0), np.array([11]))
    _check_returned_rows([good, good], None)
    with pytest.raises(InconsistentSelfOverlapError, match="conformer 11"):
        _check_returned_rows([good, bad], None)
    from roshambo3.engine.sink import DenseResult
    dense = np.stack([np.vstack([good[0], bad[0]])])        # [1 query, 2, 20]
    with pytest.raises(InconsistentSelfOverlapError, match="conformer 11"):
        _check_returned_rows(DenseResult(dense, np.array([3, 11])), None)
    _check_returned_rows(DenseResult(dense[:, :1], np.array([3])), None)
    _check_returned_rows(7, None)                 # a Callback sink's count


def _containing_pair(seed=2):
    """Query of 10 atoms; library of the same 10 plus 20 more around them.

    Their overlap exceeds the query's self-overlap, so a library self-overlap
    stored far too small makes the denominator negative.
    """
    rng = np.random.default_rng(seed)
    core = rng.uniform(-2.0, 2.0, size=(10, 3))
    shell = rng.uniform(-3.5, 3.5, size=(20, 3))
    width = 32
    q = np.zeros((1, width, 4), np.float32)
    q[0, :10, :3], q[0, :10, 3] = core, 1.0
    lib = np.zeros((1, width, 4), np.float32)
    lib[0, :30, :3], lib[0, :30, 3] = np.vstack([core, shell]), 1.0
    return dict(query_coords=q, query_n_atoms=np.array([10], np.int32),
                library_coords=lib,
                library_counts=np.array([[30]], np.uint8),
                pair_query_row=np.zeros(1, np.int32),
                pair_library_row=np.zeros(1, np.int32),
                library_vaa=np.array([1e-3], np.float32))


SHAPE_PATHS = [
    pytest.param(dict(grain=-1), id="anchor"),
    pytest.param(dict(grain=0), id="query-grain"),
    pytest.param(dict(grain=1), id="library-grain"),
    pytest.param(dict(grain=2, width=1), id="conformer-grain-scalar"),
    pytest.param(dict(field_h=0.5, field_width=8, field_unroll=0,
                      field_pipeline=True), id="field-all-start"),
    pytest.param(dict(field_h=0.5, field_width=8, field_unroll=0,
                      field_pipeline=True, field_winner_only=True),
                 id="field-winner"),
]

# These rows exercise x86-only kernels/layouts: the wide conformer grain
# requires AVX2 width 8, and reporting the field score requires the x86 AVX2
# field layout. All architectures also exercise the conformer grain's scalar
# width-1 path above, as well as both retained field paths.
if platform.machine().lower() in {"x86_64", "amd64"}:
    SHAPE_PATHS.extend([
        pytest.param(dict(grain=2, width=8), id="conformer-grain-avx2"),
        pytest.param(dict(field_h=0.5, field_width=8, field_unroll=0,
                          field_pipeline=True, field_report_field_score=True),
                     id="field-score-all-start"),
        pytest.param(dict(field_h=0.5, field_width=8, field_unroll=0,
                          field_pipeline=True, field_winner_only=True,
                          field_report_field_score=True),
                     id="field-score-winner"),
    ])


@pytest.mark.parametrize("path", SHAPE_PATHS)
def test_a_negative_denominator_reports_zero_on_every_cpu_shape_path(path):
    from roshambo3.backend import cpu
    ext = cpu.extension()
    pair = _containing_pair()
    # Exercise the native denominator clamp with a deliberately inconsistent
    # low-level stored value. The public Python scorer rejects this array
    # without its explicit metadata contract (covered below); this direct
    # extension call isolates the zero-denominator behavior itself.
    row = ext.score_pairs(steps=25, lr_q=0.1, lr_t=0.4, start_mode=1,
                          threads=1, **path, **pair)
    overlap, sq, sl = (row[0, OVERLAP_VOLUME], row[0, SELF_OVERLAP_QUERY],
                       row[0, SELF_OVERLAP_LIBRARY])
    assert overlap >= sq + sl, "fixture no longer drives the denominator negative"
    assert row[0, 1] == 0.0, f"shape Tanimoto {row[0, 1]} for {path}"
    assert row[0, 0] == 0.0


def test_the_public_cpu_scorer_refuses_a_corrupt_library_vaa():
    """Raw arrays need stored-frame metadata before their VAA is consumed."""
    from roshambo3.backend import cpu
    pair = _containing_pair()
    with pytest.raises(ValueError, match="stored-frame metadata"):
        cpu.score_pairs(steps=25, lr_q=0.1, lr_t=0.4, start_mode=1, **pair)
    pair.pop("library_vaa")
    cpu.score_pairs(steps=25, lr_q=0.1, lr_t=0.4, start_mode=1, **pair)

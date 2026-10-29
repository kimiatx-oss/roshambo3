"""The GPU self-overlap bound scan and self-overlap counters.

A consumer of STORED self-overlaps on the GPU needs to know about every row
whose overlap exceeds the mean of its self-overlaps -- including the rows the
Tanimoto denominator clamp reports as 0, which rank last and which no check of
the returned top-K can see. `check_self_overlap_bound` counts them on the
device over the whole score block. Nothing reads stored self-overlaps on the
GPU yet, so on today's recompute path the count must be 0; the predicate itself
is tested on planted rows through the extension's scan hook.
"""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import scores
from roshambo3.backend import cuda

from gpu_device import device_present

pytestmark = pytest.mark.skipif(
    not cuda.available() or not device_present(),
    reason="no CUDA extension in this build, or no device")


def _host_count(v, margin):
    """`scores.check_self_overlap_bound`'s predicate, counted."""
    v = np.asarray(v)
    overlap = v[:, scores.OVERLAP_VOLUME].astype(np.float64)
    selves = (v[:, scores.SELF_OVERLAP_QUERY].astype(np.float64)
              + v[:, scores.SELF_OVERLAP_LIBRARY].astype(np.float64))
    bad = np.flatnonzero(2.0 * overlap > selves * (1.0 + margin))
    return int(bad.size), (int(bad[0]) if bad.size else -1)


def _row(vab, vaa, vbb):
    r = np.zeros(scores.N_SCORE_COLUMNS, np.float32)
    r[scores.OVERLAP_VOLUME] = vab
    r[scores.SELF_OVERLAP_QUERY] = vaa
    r[scores.SELF_OVERLAP_LIBRARY] = vbb
    return r


def test_the_scan_flags_both_failure_shapes_and_nothing_else():
    m = scores.SELF_OVERLAP_BOUND_MARGIN
    ext = cuda.extension()
    rows = np.stack([
        _row(500.0, 800.0, 900.0),            # consistent
        _row(0.0, 0.0, 0.0),                   # empty row passes
        _row(860.0, 800.0, 900.0),             # Tanimoto > 1: ranks first
        _row(1800.0, 800.0, 900.0),            # non-positive denominator: ranks last
        _row(np.nan, 800.0, 900.0),            # NaN fails the comparison, as on CPU
        _row(850.0 * (1 + m) * (1 - 1e-5), 800.0, 900.0),   # inside the margin
        _row(850.0 * (1 + m) * (1 + 1e-5), 800.0, 900.0),   # just past it
    ])
    n, first = ext.self_overlap_bound_scan(rows, float(m))
    assert (n, first) == (3, 2)
    assert (n, first) == _host_count(rows, m)


def test_the_scan_agrees_with_the_host_predicate_row_for_row():
    rng = np.random.default_rng(208)
    v = np.zeros((200_003, scores.N_SCORE_COLUMNS), np.float32)
    v[:, scores.SELF_OVERLAP_QUERY] = rng.uniform(100, 1000, len(v))
    v[:, scores.SELF_OVERLAP_LIBRARY] = rng.uniform(100, 1000, len(v))
    half = 0.5 * (v[:, 5].astype(np.float64) + v[:, 6])
    # Concentrated around the bound so both sides and the margin are populated.
    v[:, scores.OVERLAP_VOLUME] = half * rng.uniform(0.995, 1.005, len(v))
    ext = cuda.extension()
    for margin in (scores.SELF_OVERLAP_BOUND_MARGIN, 0.0, 2e-3):
        got = ext.self_overlap_bound_scan(v, float(margin))
        assert tuple(got) == _host_count(v, margin)
        assert got[0] > 0


def test_an_empty_block_counts_nothing():
    ext = cuda.extension()
    assert tuple(ext.self_overlap_bound_scan(
        np.zeros((0, scores.N_SCORE_COLUMNS), np.float32), 1e-3)) == (0, -1)


# --------------------------------------------------------------------------
# through score_pairs, on the recompute path
# --------------------------------------------------------------------------

def _panel():
    import bench.make_golden as mk
    from roshambo3.layout import tile32

    coords, n_real = mk.make_input()
    nq, nl = mk.N_QUERIES, mk.N_LIBRARY
    kw = dict(
        query_coords=np.ascontiguousarray(coords[:nq]),
        query_types=np.zeros((nq, mk.L), np.int32),
        query_n_atoms=np.ascontiguousarray(n_real[:nq]),
        library_coords=tile32(np.ascontiguousarray(coords[nq:])),
        library_counts=n_real[nq:].astype(np.uint8).reshape(-1, 1),
        pair_query_row=np.repeat(np.arange(nq, dtype=np.int32), nl),
        pair_library_row=np.tile(np.arange(nl, dtype=np.int32), nq),
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.zeros((1, 1), np.float32), **mk.OPT)
    return mk, kw


@pytest.mark.parametrize("pose_search", [cuda.PoseSearch.FIELD,
                                         cuda.PoseSearch.ANALYTIC])
def test_flag_off_and_on_score_identically_and_count_every_row(pose_search):
    mk, kw = _panel()
    off_stats, on_stats = {}, {}
    off = cuda.score_pairs(pose_search=pose_search, call_stats=off_stats, **kw)
    on = cuda.score_pairs(pose_search=pose_search, check_self_overlap_bound=True,
                          call_stats=on_stats, **kw)
    plain = cuda.score_pairs(pose_search=pose_search, **kw)
    assert np.array_equal(off.view(np.uint32), on.view(np.uint32))
    assert np.array_equal(off.view(np.uint32), plain.view(np.uint32))
    n_rows = len(kw["pair_query_row"])
    assert off_stats == {
        "query_self_evaluations": mk.N_QUERIES,
        "library_self_evaluations": mk.N_LIBRARY,
        "library_self_supplied": False,
        "self_bound_rows_checked": -1,
        "self_bound_rows_flagged": -1,
        "self_bound_first_row": -1}
    assert on_stats["self_bound_rows_checked"] == n_rows
    # The recompute path: self-overlaps describe the scored coordinates.
    assert (on_stats["self_bound_rows_flagged"],
            on_stats["self_bound_first_row"]) == _host_count(on, scores.SELF_OVERLAP_BOUND_MARGIN)
    assert on_stats["self_bound_rows_flagged"] == 0
    assert on_stats["library_self_evaluations"] == mk.N_LIBRARY
    assert on_stats["library_self_supplied"] is False


def test_the_margin_comes_from_scores_and_the_scan_sees_rows_the_topk_drops(
        monkeypatch):
    """A negative margin flags ordinary rows, which makes two things visible:
    the margin is read from the policy table (the recompute route's entry of
    `scores.SELF_OVERLAP_MARGIN_POLICY`) at call time, and with a
    device top-K of k=5 the scan still counts all n rows -- the property a
    returned-row check lacks."""
    mk, kw = _panel()
    monkeypatch.setitem(scores.SELF_OVERLAP_MARGIN_POLICY,
                        scores.ROUTE_RECOMPUTE, (-0.5, -0.5))
    dense_stats = {}
    dense = cuda.score_pairs(check_self_overlap_bound=True,
                             call_stats=dense_stats, **kw)
    want = _host_count(dense, -0.5)
    assert want[0] > 5 * mk.N_QUERIES
    assert (dense_stats["self_bound_rows_flagged"],
            dense_stats["self_bound_first_row"]) == want
    tk = cuda.extension().DeviceTopK(mk.N_QUERIES, 5, 0, scores.N_SCORE_COLUMNS,
                                     True, 0)
    topk_stats = {}
    cuda.score_pairs(check_self_overlap_bound=True, call_stats=topk_stats,
                     topk=tk, **kw)
    returned = sum(len(i) for _, i in tk.result(0))
    assert returned == 5 * mk.N_QUERIES
    assert topk_stats["self_bound_rows_flagged"] == want[0]
    assert topk_stats["self_bound_rows_checked"] == len(kw["pair_query_row"])


def test_stage_timings_accumulate_the_counters():
    mk, kw = _panel()
    ext = cuda.extension()
    was = bool(ext.stage_timings().get("enabled", False))
    ext.stage_timing_enable(True)
    try:
        ext.stage_timing_reset()
        cuda.score_pairs(**kw)
        cuda.score_pairs(check_self_overlap_bound=True, **kw)
        t = ext.stage_timings()
    finally:
        ext.stage_timing_enable(was)
    assert t["n_calls"] == 2
    assert t["query_self_evaluations"] == 2 * mk.N_QUERIES
    assert t["library_self_evaluations"] == 2 * mk.N_LIBRARY
    assert t["library_self_supplied_calls"] == 0
    assert t["self_bound_rows_checked"] == len(kw["pair_query_row"])
    assert t["self_bound_rows_flagged"] == 0
    assert t["self_bound_s"] >= 0.0


def test_call_stats_must_be_the_callers_own_int64_array():
    """The binding refuses a buffer it would have to convert, and so would
    write the counters into a copy nobody reads."""
    from roshambo3.layout import library_counts_u8

    _, kw = _panel()
    v = np.empty((len(kw["pair_query_row"]), scores.N_SCORE_COLUMNS), np.float32)
    with pytest.raises(RuntimeError, match="call_stats"):
        cuda.extension().optimize_pairs(
            kw["query_coords"], kw["query_types"], kw["query_n_atoms"],
            kw["library_coords"], library_counts_u8(kw["library_counts"]),
            kw["interaction_r"], kw["interaction_p"],
            kw["pair_query_row"], kw["pair_library_row"], v,
            optim_color=False, mixing_param=0.0, lr_q=0.1, lr_t=0.4, nsteps=1,
            start_mode=0, call_stats=np.zeros(6, np.int32))

"""Portable checks for the Metal campaign's strict ranking gate."""

import numpy as np

from bench.metal import ranking


def test_molecule_ranking_preserves_large_ids_and_score_ties():
    mol_ids = np.array([2**60 + 7, 2**60 + 8, 2**60 + 7, 2**60 + 9], np.int64)
    conf_indices = np.array([9, 2, 4, 1], np.int64)
    scores = np.zeros((4, 20), np.float32)
    scores[:, 0] = [0.9, 0.9, 0.95, 0.9]
    molecules, winners = ranking(scores, mol_ids, conf_indices, k=3)
    np.testing.assert_array_equal(molecules, [2**60 + 7, 2**60 + 8, 2**60 + 9])
    np.testing.assert_array_equal(winners, [4, 2, 1])


def _timings():
    return [
        dict(
            mode=mode,
            backend=backend,
            repeat=repeat,
            seconds=seconds if repeat < 5 else 10000,
        )
        for mode in ("shape", "rescore", "combo")
        for backend, seconds in (("cpu", 10), ("metal_fast", 2), ("metal_precise", 3))
        for repeat in range(11)
    ]


def test_summary_selects_first_five_matched_repeats():
    from bench.metal import summarize_timings

    result = summarize_timings(_timings(), 5)
    for mode in result.values():
        assert mode["cpu"] == 10
        assert mode["cpu_speedup"] == 5
        assert mode["fast_exp2_speedup"] == 1.5


def test_summary_rejects_missing_and_duplicate_measurements():
    import pytest

    from bench.metal import summarize_timings

    records = _timings()
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        summarize_timings(records[1:], 5)
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        summarize_timings(records + [records[0]], 5)


def test_resummarizing_preserves_original_provenance(tmp_path):
    import json

    from bench.metal import summarize_existing

    provenance = dict(warmups=3, repeats=11, provisional=False)
    original = json.dumps(provenance)
    (tmp_path / "provenance.json").write_text(original)
    (tmp_path / "cpu-preliminary.json").write_text(
        json.dumps(dict(seconds={"10": 60, "14": 50}))
    )
    (tmp_path / "timings.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _timings())
    )
    report = summarize_existing(tmp_path, 5)
    assert report["cpu_workers"] == 14
    assert report["selected_repeat_indices"] == list(range(5))
    assert report["originally_requested_repeats"] == 11
    assert report["minimum_repeats_met"]
    assert report["raw_sample_count"] == 99
    assert (tmp_path / "provenance.json").read_text() == original


def test_fixed_pose_audit_includes_self_terms_and_tanimotos():
    from bench.metal import audit_score_columns

    oracle = np.ones((2, 9), np.float64)
    replay = oracle.copy()
    reported = oracle.copy()
    assert audit_score_columns(reported, replay, oracle)["passed"]
    reported[0, 6] += 0.001  # correct cross overlap, stale library denominator
    audit = audit_score_columns(reported, replay, oracle)
    assert not audit["passed"]
    assert audit["reported_oracle_max_relative"][6] > 1e-4
    reported = oracle.copy()
    replay[1, 0] += 0.001
    assert not audit_score_columns(reported, replay, oracle)["passed"]

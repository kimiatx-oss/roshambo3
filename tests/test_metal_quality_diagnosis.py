"""Portable guards for the causal quality analysis, without changing ranking rules."""

import numpy as np

from bench.metal import compare, replace_self_terms


def test_common_self_terms_remove_denominator_only_rank_reversal():
    a = np.zeros((2, 20), np.float32)
    a[:, 3] = [9, 8]
    a[:, 5] = 10
    a[:, 6] = [10, 8]
    a = replace_self_terms(a, a, "shape")
    b = a.copy()
    b[0, 6] = 12
    b = replace_self_terms(b, b, "shape")
    mids = np.array([11, 22])
    conf_indices = np.array([0, 0])
    assert not compare(a, b, mids, conf_indices)["top1"]
    restored = replace_self_terms(b, a, "shape")
    assert compare(a, restored, mids, conf_indices)["top1"]
    np.testing.assert_array_equal(restored, a)
    assert b[0, 6] == 12


def test_common_self_terms_recompute_both_channels_and_guard_empty_denominator():
    rows = np.zeros((1, 20), np.float32)
    rows[0, 3:5] = [3, 2]
    rows[0, 9:16] = np.arange(7)
    source = np.zeros_like(rows)
    source[0, 5:9] = [5, 4, 1, 1]
    actual = replace_self_terms(rows, source, "combo")
    assert actual[0, 1] == 0.5
    assert actual[0, 2] == 0  # zero denominator
    assert actual[0, 0] == 0.25
    np.testing.assert_array_equal(actual[:, 9:], rows[:, 9:])

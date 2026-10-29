"""Small exhaustive proof checks for worker-local top-K pruning.

These tests check, independently of the native data structure,
that retaining local conformer winners, or local distinct-molecule maxima,
preserves the dense answer for arbitrary worker ownership and completion order.
"""

from itertools import product

import numpy as np

from roshambo3.numerics import rank_order


def _panel():
    # Deliberately non-monotonic global ids, tied scores, a molecule with three
    # conformers, and a second query with a different rank order.
    gids = np.asarray([30, 10, 50, 20, 60, 40], dtype=np.int64)
    molecules = np.asarray([7, 7, 7, 8, 9, 9], dtype=np.int64)
    scores = np.asarray([
        [0.90, 0.90, 0.90, 0.80, 0.70, 0.70],
        [0.40, 0.80, 0.80, 0.80, 0.20, 0.20],
    ], dtype=np.float32)
    rows = np.zeros((scores.shape[0], scores.shape[1], 20), dtype=np.float32)
    rows[:, :, 0] = scores
    rows[:, :, 1] = scores * np.float32(0.5)
    # No-color rows are valid full score records; retain distinctive winner
    # poses so the test checks score/pose association, not just the ids.
    for q in range(rows.shape[0]):
        for lib in range(rows.shape[1]):
            rows[q, lib, 9:13] = (1.0, float(q), float(lib), 0.0)
            rows[q, lib, 13:16] = (float(lib), float(q), float(q + lib))
            rows[q, lib, 16:20] = (1.0, 0.0, 0.0, 0.0)
    return gids, molecules, rows


def _better(left, right):
    """Production molecule winner order: score desc, winning gid asc."""
    return (left[1][0] > right[1][0]
            or (left[1][0] == right[1][0] and left[0] < right[0]))


def _ordered(items, k):
    if k <= 0 or not items:
        return []
    ids = np.asarray([item[0] for item in items], dtype=np.int64)
    values = np.asarray([item[1][0] for item in items], dtype=np.float32)
    return [items[i] for i in rank_order(values, ids)[:k]]


def _dense_conformer(rows, gids, k):
    out = []
    for query_rows in rows:
        by_gid = {}
        for i, row in enumerate(query_rows):
            candidate = (int(gids[i]), row)
            current = by_gid.get(candidate[0])
            if current is None or _better(candidate, current):
                by_gid[candidate[0]] = candidate
        out.append(_ordered(list(by_gid.values()), k))
    return out


def _dense_molecule(rows, gids, molecule_ids, k):
    out = []
    for query_rows in rows:
        best = {}
        for i, row in enumerate(query_rows):
            molecule_id = int(molecule_ids[i])
            candidate = (int(gids[i]), row)
            current = best.get(molecule_id)
            if current is None or _better(candidate, current):
                best[molecule_id] = candidate
        out.append(_ordered(list(best.values()), k))
    return out


def _worker_result(rows, gids, molecule_ids, assignment, completion_order,
                   k, *, molecule_level):
    n_workers = max(assignment, default=-1) + 1
    out = []
    for q, query_rows in enumerate(rows):
        local_lists = []
        for worker in range(n_workers):
            seen = {}
            for lib in completion_order:
                if assignment[lib] != worker:
                    continue
                candidate = (int(gids[lib]), query_rows[lib])
                if molecule_level:
                    key = int(molecule_ids[lib])
                    current = seen.get(key)
                    if current is None or _better(candidate, current):
                        seen[key] = candidate
                else:
                    seen[int(gids[lib])] = candidate
            local_lists.append(_ordered(list(seen.values()), k))

        # Bounded final merge: only the local lists are visible here. Molecule
        # partial maxima with the same molecule id are reduced again by score,
        # then winning gid, before the distinct-molecule top-K.
        merged = {}
        for local in local_lists:
            for candidate in local:
                key = (int(molecule_ids[np.flatnonzero(gids == candidate[0])[0]])
                       if molecule_level else candidate[0])
                current = merged.get(key)
                if current is None or _better(candidate, current):
                    merged[key] = candidate
        out.append(_ordered(list(merged.values()), k))
    return out


def _assert_same_hits(actual, expected):
    assert len(actual) == len(expected)
    for got_query, want_query in zip(actual, expected):
        assert [gid for gid, _ in got_query] == [gid for gid, _ in want_query]
        for (_, got_row), (_, want_row) in zip(got_query, want_query):
            np.testing.assert_array_equal(got_row, want_row)


def test_local_conformer_topk_matches_dense_for_all_small_worker_splits():
    gids, molecules, rows = _panel()
    n = len(gids)
    orders = [tuple(range(n)), tuple(reversed(range(n))), (2, 0, 5, 1, 3, 4)]

    for assignment in product(range(3), repeat=n):
        for order in orders:
            for k in (0, 1, 2, n + 3):
                actual = _worker_result(rows, gids, molecules, assignment,
                                        order, k, molecule_level=False)
                _assert_same_hits(actual, _dense_conformer(rows, gids, k))


def test_local_distinct_molecule_maxima_match_dense_when_molecules_split():
    gids, molecules, rows = _panel()
    n = len(gids)
    orders = [tuple(range(n)), tuple(reversed(range(n))), (2, 0, 5, 1, 3, 4)]

    for assignment in product(range(3), repeat=n):
        # Every assignment includes cases where a molecule's conformers land
        # on several workers, and where one worker sees many tied conformers of
        # one molecule ahead of several other molecules.
        for order in orders:
            for k in (0, 1, 2, 8):
                actual = _worker_result(rows, gids, molecules, assignment,
                                        order, k, molecule_level=True)
                _assert_same_hits(actual,
                                  _dense_molecule(rows, gids, molecules, k))


def test_empty_inputs_and_zero_queries_have_stable_shapes():
    empty_rows = np.empty((2, 0, 20), dtype=np.float32)
    empty_ids = np.empty(0, dtype=np.int64)
    assert _dense_conformer(empty_rows, empty_ids, 5) == [[], []]
    assert _dense_molecule(empty_rows, empty_ids, empty_ids, 5) == [[], []]

    no_queries = np.empty((0, 3, 20), dtype=np.float32)
    assert _dense_conformer(no_queries, np.arange(3), 2) == []
    assert _dense_molecule(no_queries, np.arange(3), np.arange(3), 2) == []


def test_repeated_same_pair_event_is_idempotent():
    gids, molecules, rows = _panel()
    assignment = (0, 1, 0, 1, 0, 1)
    repeated = (0, 1, 2, 3, 4, 5, 1, 4)
    duplicated_rows = np.concatenate([rows, rows[:, [1, 4], :]], axis=1)
    duplicated_gids = np.concatenate([gids, gids[[1, 4]]])
    duplicated_molecules = np.concatenate([molecules, molecules[[1, 4]]])
    duplicated_assignment = assignment + (1, 0)

    actual = _worker_result(duplicated_rows, duplicated_gids,
                            duplicated_molecules, duplicated_assignment,
                            repeated, 3, molecule_level=False)
    _assert_same_hits(actual, _dense_conformer(rows, gids, 3))

    actual_molecules = _worker_result(
        duplicated_rows, duplicated_gids, duplicated_molecules,
        duplicated_assignment, repeated, 3, molecule_level=True)
    _assert_same_hits(actual_molecules,
                      _dense_molecule(rows, gids, molecules, 3))

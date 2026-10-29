"""Ranking, tie-breaking and conformer/molecule identity.

This module owns the project's *only* ranking rule.  Every downstream caller
must call :func:`rank_g` or :func:`top_n` rather than ``argsort``.

**Why this exists.**  ``numpy.argsort``'s default quicksort is not stable, and
float32 score ties are common at M = 1e7 (the score is a ratio of two ~1e3-term
fp32 reductions, so the achievable resolution near the top of a 1e7-row list is
comparable to the spacing between adjacent ranks).  An unstable tie-break makes
two runs that agree on *every score bit* disagree on rank -- manufacturing
"rank changes" out of nothing and burying real ones in that noise.

The rule, from ``doc/NUMERICS_POLICY.md`` 4b:

    rank_g(x) = 1 + |{y : g(y) > g(x)}|,  ties broken by ascending canonical id,
    applied *identically* to both rankings being compared.

``roshambo3.numerics.rank_order`` is the implementation of the ordering; this
module is the metrics-side surface for it, adds the 1-based competition rank,
and adds the conformer -> molecule reduction.
"""

from __future__ import annotations

import numpy as np

from roshambo3.numerics import rank_order as _rank_order

__all__ = [
    "rank_order",
    "rank_g",
    "top_n",
    "top_k_order",
    "per_query_rank",
    "reduce_over_conformers",
    "competition_rank",
    "canonical_ids",
    "assert_unique_ids",
    "TieBreakError",
]


class TieBreakError(ValueError):
    pass


def canonical_ids(mol_id, conf_idx=None):
    """A **unique** int64 tie-break key ordering by (mol_id asc, conf_idx asc).

    The policy says "ties broken by ascending molecule_id".  That is a total
    order only when there is one conformer per molecule -- not true of a
    library with 9.5 or 40.4 conformers per molecule.  Passing a repeated
    ``mol_id`` as the tie-break id silently reintroduces exactly the
    nondeterminism the rule exists to remove, *and* breaks any metric that
    assumes a ranked list names each item once (RBO reads > 1 when it happens,
    which is how this was found).

    So: at conformer level the canonical id is the rank of ``(mol_id,
    conf_idx)`` in lexicographic order.  It is unique by construction, it is a
    strictly increasing function of molecule_id, and it therefore breaks ties
    by ascending molecule_id exactly as the policy says -- with conf_idx as the
    sub-order the policy leaves unspecified.  Rank rather than a bit-packed
    key because ``mol_id`` here is a 64-bit hash and any packing would overflow.
    """
    m = np.asarray(mol_id).ravel()
    c = (np.zeros(m.shape[0], np.int64) if conf_idx is None
         else np.asarray(conf_idx).ravel())
    order = np.lexsort((c, m))
    key = np.empty(m.shape[0], np.int64)
    key[order] = np.arange(m.shape[0], dtype=np.int64)
    return key


def assert_unique_ids(ids, what="the tie-break id"):
    """Refuse a non-unique canonical id.

    Called once per report rather than per query, because at M = 1e7 the check
    itself is not free.  Use :func:`canonical_ids` to build a compliant key
    from ``(mol_id, conf_idx)``.
    """
    if ids is None:
        return
    a = np.asarray(ids).ravel()
    if np.unique(a).size != a.size:
        raise TieBreakError(
            f"{what} is not unique ({a.size - np.unique(a).size} duplicates of "
            f"{a.size} rows). A repeated id is not a tie-break: it leaves the "
            "order of the tied rows to an unstable sort, which is the exact "
            "failure doc/NUMERICS_POLICY.md 4b exists to prevent, and it makes "
            "a ranked list name the same item twice. Build the key with "
            "roshambo3.metrics.canonical_ids(mol_id, conf_idx).")


def rank_order(scores, ids=None):
    """Indices of ``scores`` in policy order: descending score, ties by
    ascending ``ids`` (row index when ``ids`` is None).

    Thin re-export so that a downstream module needs exactly one import to get
    the canonical order.
    """
    return _rank_order(np.asarray(scores), ids=None if ids is None else np.asarray(ids))


def rank_g(scores, ids=None):
    """1-based *position* rank under the policy tie-break.

    Returns ``r`` with ``r[i]`` = the position of element ``i`` in the ordered
    list, 1 = best.  Because the tie-break is a total order on ``(-score, id)``
    this is a permutation of ``1..n`` -- there are no shared ranks, which is
    exactly the point: two runs with identical scores get identical ranks.

    Note this is a *position* rank, not the competition rank
    ``1 + |{y : g(y) > g(x)}|`` which assigns tied elements the same number.
    The two agree whenever there are no ties, and where there are ties the
    position rank is the one a top-k cut actually uses.  :func:`competition_rank`
    computes the other if a caller specifically wants it.
    """
    order = rank_order(scores, ids)
    r = np.empty(order.shape[0], np.int64)
    r[order] = np.arange(1, order.shape[0] + 1)
    return r


def competition_rank(scores):
    """``1 + |{y : g(y) > g(x)}|`` -- ties share a rank.

    Provided because the definition is stated in this form.  It is
    *not* what a top-k cut uses (a cut needs a total order), so :func:`rank_g`
    is the default everywhere else.
    """
    s = np.asarray(scores, np.float64)
    neg_sorted = np.sort(-s)                      # ascending in -score
    n_strictly_greater = np.searchsorted(neg_sorted, -s, side="left")
    return n_strictly_greater.astype(np.int64) + 1


def top_k_order(scores, k, ids=None):
    """Indices of the top ``k`` in policy order, without sorting all of ``scores``.

    Exact, not approximate: it partitions to find the k-th largest value, takes
    **every** index at or above it (so a tied boundary cannot silently drop a
    row), applies the full policy order to that small set, and truncates.  At
    M = 1e7 and k = 1e4 this is ~30x faster than a full sort, which is what
    makes the simulation path in :mod:`roshambo3.metrics.simulate` cheap enough to
    sweep.  ``test_t08_metrics.py`` asserts it equals ``rank_order(...)[:k]``.
    """
    s = np.asarray(scores)
    n = s.shape[0]
    k = int(min(k, n))
    if k <= 0:
        return np.empty(0, np.int64)
    if k >= n // 4:
        return rank_order(s, ids)[:k]
    part = np.argpartition(-s, k - 1)[:k]
    thresh = s[part].min()
    cand = np.flatnonzero(s >= thresh)
    sub = rank_order(s[cand], None if ids is None else np.asarray(ids)[cand])
    return cand[sub][:k]


def top_n(scores, n, ids=None):
    """Canonical ids (or indices) of the top ``n`` under the policy tie-break."""
    order = top_k_order(scores, int(n), ids)
    return order if ids is None else np.asarray(ids)[order]


def per_query_rank(scores, n_target, ids=None):
    """``rank_g`` applied within each query of a flat ``[n_query * n_target]``
    score vector (or a ``[n_query, n_target]`` matrix).

    ``ids`` is the per-library canonical id vector, shared by every query, so
    the tie-break is the same rule in every row.
    """
    s = np.asarray(scores)
    if s.ndim == 1:
        s = s.reshape(-1, int(n_target))
    out = np.empty(s.shape, np.int64)
    for q in range(s.shape[0]):
        out[q] = rank_g(s[q], ids)
    return out


def reduce_over_conformers(scores, mol_ids, ids=None):
    """Conformer-level scores -> molecule-level scores, de-duplicated globally.

    Molecule-level reduction has two halves, both handled here.

    *Identity.*  ``_reduce_over_conformers`` in the shipped code runs per batch
    and ``_merge_results`` did no cross-batch de-duplication, so
    a molecule whose conformers straddle two groups could appear twice.  Here
    the reduction is over the whole vector at once, keyed on ``mol_ids``, so a
    molecule appears exactly once by construction whatever the batching was.

    *Order statistics.*  The molecule score is a max over a **variable** number
    of conformers.  A 25-conformer molecule has a higher expected max than a
    3-conformer one purely by order statistics, and per-score noise perturbs the
    25-conformer max more.  ``n_conf`` is returned so molecule-level recall can
    be stratified by it -- which :mod:`roshambo3.metrics.recall` does automatically.

    Returns ``(uniq_mol_ids, best_score, n_conf, argmax_index)``, with
    ``uniq_mol_ids`` ascending so the molecule-level tie-break is the same
    ascending-id rule as the conformer-level one.
    """
    s = np.asarray(scores, np.float64).ravel()
    m = np.asarray(mol_ids).ravel()
    if s.shape != m.shape:
        raise ValueError(f"scores {s.shape} and mol_ids {m.shape} must match")
    uniq, inv = np.unique(m, return_inverse=True)
    n_conf = np.bincount(inv, minlength=uniq.shape[0])
    best = np.full(uniq.shape[0], -np.inf)
    np.maximum.at(best, inv, s)
    # argmax with the same tie-break: among conformers attaining the max, the
    # one with the smallest index (= smallest canonical id when ids ascend).
    idx = np.arange(s.shape[0]) if ids is None else np.asarray(ids)
    hit = s >= best[inv]
    arg = np.full(uniq.shape[0], np.iinfo(np.int64).max, np.int64)
    np.minimum.at(arg, inv[hit], np.asarray(idx)[hit].astype(np.int64))
    return uniq, best, n_conf, arg

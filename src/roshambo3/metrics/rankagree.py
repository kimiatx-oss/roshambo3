"""Rank agreement: RBO_ext, Kendall tau-b, and one metric that is refused.

**RBO_ext (Webber, Moffat & Zobel 2010, eq. 32)** is the primary rank-agreement
statistic.  It is top-weighted, handles non-conjoint lists (the candidate's
top-N and gold's top-N need not contain the same items -- which is the whole
point), and its single parameter ``p`` has an interpretation: the expected
depth a user examines is ``1/(1-p)``.

* ``p = 0.98`` -> expected viewed depth **50**.  This is the reported headline
  because 50 is a number a chemist can be told and reason about.
* ``p = 0.998`` -> expected depth 500, reported alongside so a change that only
  moves the deep tail is visible.

**Kendall tau-b restricted to G_N** is reported as a secondary statistic and
never alone: it weights the swap ``(rank 1, rank 2)`` exactly as heavily as
``(rank 99999, rank 100000)``, which is not the loss function of a screen.

**Spearman on the intersection is disqualified.**  Calling
:func:`spearman_on_intersection` raises.  It is present precisely so that the
disqualification is executable rather than a sentence in a document someone
skims: it conditions on ``G_N intersect A_N``, deleting exactly the items the
approximation got wrong, so a method that drops half the true top-N can score
*higher* than one that keeps everything with small perturbations.  It rewards
the failure mode we are trying to detect.
"""

from __future__ import annotations

import numpy as np

from .ranking import TieBreakError, assert_unique_ids, rank_order, top_k_order

__all__ = [
    "rbo_ext",
    "kendall_tau_b_on_gold",
    "spearman_on_intersection",
    "DisqualifiedMetric",
    "TieBreakError",
    "rank_agreement_report",
    "gold_top_landing_table",
    "RBO_P_HEADLINE",
    "RBO_P_DEEP",
]

RBO_P_HEADLINE = 0.98
RBO_P_DEEP = 0.998


class DisqualifiedMetric(RuntimeError):
    """Raised by a metric this project has ruled out, with the reason."""


def rbo_ext(list_s, list_l, p=RBO_P_HEADLINE):
    """RBO_EXT of two ranked id lists at persistence ``p``.

    ``list_s`` and ``list_l`` are sequences of ids, best first.  They may have
    different lengths and need not be conjoint.  Webber et al. eq. (32):

        RBO_EXT = (1-p)/p * [ sum_{d=1..l} X_d/d p^d
                              + sum_{d=s+1..l} X_s (d-s)/(s d) p^d ]
                  + [ (X_l - X_s)/l + X_s/s ] p^l

    with ``X_d`` the overlap at depth d, ``s <= l`` the two list lengths.

    Sanity anchors, all unit-tested: identical lists -> 1.0 exactly; disjoint
    lists -> 0.0; prefix agreement dominates, so swapping ranks 1 and 2 costs
    far more than swapping ranks 99 and 100.
    """
    a = list(list_s)
    b = list(list_l)
    if len(a) > len(b):
        a, b = b, a
    s, l = len(a), len(b)
    if s == 0:
        return 0.0
    if not (0.0 < p < 1.0):
        raise ValueError("RBO persistence p must be in (0,1)")
    if len(set(a)) != s or len(set(b)) != l:
        raise TieBreakError(
            "RBO_ext requires each list to name every item at most once. A "
            "repeated id (e.g. molecule_id used as the tie-break key on a "
            "multi-conformer tier) makes the overlap count exceed the depth "
            "and RBO exceed 1. Build the key with "
            "roshambo3.metrics.canonical_ids(mol_id, conf_idx).")

    seen_a, seen_b = set(), set()
    X = np.empty(l + 1, np.float64)
    X[0] = 0.0
    overlap = 0
    for d in range(1, l + 1):
        if d <= s:
            x, y = a[d - 1], b[d - 1]
            if x == y:
                # x enters both prefixes at this depth, so it is a new member
                # of the intersection (neither list repeats an id).
                overlap += 1
            else:
                if x in seen_b:
                    overlap += 1
                if y in seen_a:
                    overlap += 1
            seen_a.add(x)
            seen_b.add(y)
        else:
            y = b[d - 1]
            if y in seen_a:
                overlap += 1
            seen_b.add(y)
        X[d] = overlap

    d1 = np.arange(1, l + 1, dtype=np.float64)
    pw = p ** d1
    term1 = float(np.sum(X[1:] / d1 * pw))
    if l > s:
        d2 = np.arange(s + 1, l + 1, dtype=np.float64)
        term2 = float(np.sum(X[s] * (d2 - s) / (s * d2) * (p ** d2)))
    else:
        term2 = 0.0
    tail = ((X[l] - X[s]) / l + X[s] / s) * (p ** l)
    return float((1.0 - p) / p * (term1 + term2) + tail)


def kendall_tau_b_on_gold(gold_scores, cand_scores, n_gold, ids=None):
    """Kendall tau-b between gold and candidate score, restricted to G_N.

    Secondary only.  It is uniform in rank -- a swap at the top and a swap in
    the tail cost the same -- so it must never be reported without RBO_ext
    beside it, and :func:`rank_agreement_report` enforces that by always
    emitting both.
    """
    from scipy.stats import kendalltau

    g = np.asarray(gold_scores, np.float64)
    a = np.asarray(cand_scores, np.float64)
    G = top_k_order(g, int(n_gold), ids)
    if G.size < 2:
        return float("nan")
    t = kendalltau(g[G], a[G], variant="b")
    return float(t.statistic)


def spearman_on_intersection(*_args, **_kwargs):
    """Disqualified.  Always raises :class:`DisqualifiedMetric`."""
    raise DisqualifiedMetric(
        "Spearman on the intersection G_N n A_N is disqualified in ROSHAMBO3 "
        "(doc/METRICS.md section 4). It conditions on the set the approximation got right, "
        "deleting exactly the items it got wrong, so a method that drops half "
        "the true top-N can score HIGHER than one that keeps everything with "
        "small perturbations -- the metric rewards the failure mode it is "
        "supposed to detect. Use rbo_ext(p=0.98) as the primary rank-agreement "
        "statistic and kendall_tau_b_on_gold as a secondary. It is not "
        "reportable even as a secondary or 'for reference'."
    )


def gold_top_landing_table(gold_scores, cand_scores, top=10, ids=None):
    """Where each of the gold top-``top`` landed in the candidate ranking.

    This table must be published raw and read before any
    summary statistic is trusted, because outliers hide in aggregates.  Columns:
    gold rank, library id, gold score, candidate score, candidate rank, signed
    score delta.
    """
    g = np.asarray(gold_scores, np.float64)
    a = np.asarray(cand_scores, np.float64)
    idv = np.arange(g.shape[0]) if ids is None else np.asarray(ids)
    G = top_k_order(g, int(top), ids)
    # Rank of a specific item without sorting all M: count strictly-better
    # scores plus equal scores with a smaller canonical id -- the policy
    # tie-break, evaluated pointwise.
    def _cand_rank(j):
        eq = a == a[j]
        return int(np.count_nonzero(a > a[j])
                   + np.count_nonzero(eq & (idv < idv[j])) + 1)
    cand_rank = {int(j): _cand_rank(int(j)) for j in G}
    return [
        {
            "gold_rank": int(i + 1),
            "target_id": int(idv[j]),
            "gold_score": float(g[j]),
            "cand_score": float(a[j]),
            "cand_rank": int(cand_rank[int(j)]),
            "delta_signed": float(a[j] - g[j]),
        }
        for i, j in enumerate(G)
    ]


def rank_agreement_report(gold, cand, n_gold, ids=None, query_ids=None, top_table=10):
    """RBO_ext at both p, tau-b, and the gold-top landing table, per query.

    Spearman-on-intersection is absent by construction; the report carries a
    ``disqualified`` field naming it and why, so a reader of the JSON sees the
    exclusion rather than wondering whether it was forgotten.
    """
    g = np.atleast_2d(np.asarray(gold))
    a = np.atleast_2d(np.asarray(cand))
    assert_unique_ids(ids)
    nq = g.shape[0]
    query_ids = list(query_ids) if query_ids is not None else [f"q{i}" for i in range(nq)]

    rows = []
    for q in range(nq):
        Gord = top_k_order(g[q], int(n_gold), ids)
        Aord = top_k_order(a[q], int(n_gold), ids)
        idv = np.arange(g.shape[1]) if ids is None else np.asarray(ids)
        rows.append({
            "query_index": q,
            "query_id": query_ids[q],
            "rbo_ext_p0.98": rbo_ext(idv[Gord], idv[Aord], RBO_P_HEADLINE),
            "rbo_ext_p0.998": rbo_ext(idv[Gord], idv[Aord], RBO_P_DEEP),
            "kendall_tau_b_on_gold": kendall_tau_b_on_gold(g[q], a[q], n_gold, ids),
            "gold_top_landing": gold_top_landing_table(g[q], a[q], top_table, ids),
        })

    def agg(key):
        v = np.array([r[key] for r in rows], np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "mean": float(v.mean()), "min": float(v.min()),
                "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
                "max": float(v.max())}

    return {
        "n_gold": int(n_gold),
        "p_headline": RBO_P_HEADLINE,
        "p_deep": RBO_P_DEEP,
        "expected_viewed_depth": {"p0.98": 50.0, "p0.998": 500.0},
        "aggregate": {k: agg(k) for k in
                      ("rbo_ext_p0.98", "rbo_ext_p0.998", "kendall_tau_b_on_gold")},
        "per_query": [{k: v for k, v in r.items() if k != "gold_top_landing"}
                      for r in rows],
        "gold_top_landing": {r["query_id"]: r["gold_top_landing"] for r in rows},
        "disqualified": {
            "spearman_on_intersection":
                "conditions on G_N n A_N, deleting the items the approximation "
                "got wrong; not reported, not even as a secondary",
        },
        "secondary_only": {
            "kendall_tau_b_on_gold":
                "uniform in rank -- weights (1,2) as heavily as (99999,100000); "
                "never quote without RBO_ext",
        },
    }

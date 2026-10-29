"""Recall, the epsilon->gamma curve, and survivor fractions.

``R(N, k) = |G_N intersect A_k| / N`` -- of the gold top-N, how many are inside
the candidate's top-k.  Everything a cascade cares about is a special case:

* ``R(N', N')`` is the headline: rescore nothing, just take the candidate's own
  top-N'.
* ``R(N', k*N')`` for ``k in {1, 2, 5, 10}`` **is the epsilon->gamma curve**:
  retrieve ``k*N'`` candidates cheaply, rescore all of them with gold, keep the
  top N'.  Recall of that cascade is exactly ``R(N', k*N')``, because a rescored
  survivor is ranked by gold and therefore lands in the gold top-N' iff it was
  in the gold top-N' to begin with.
* For a filter that returns an arbitrary survivor set ``S`` (not a top-k of any
  score), recall is ``|G_N' intersect S| / N'`` and the cost axis is the
  **survivor fraction** ``|S| / M``.  :func:`survivor_recall` is that form, and
  it is the one an operating-point choice consumes.

**Aggregation.**  Never a bare mean.  This was *measured*: the 1e-4
quantile threshold ranged 0.70 -> 0.95 over four random queries, so a good mean
routinely hides a query where the method fails outright.  Every aggregate here
carries mean, min and the 10th percentile, and every report carries the full
per-query table.

**Conformer vs molecule level.**  Both are always computed --
:func:`recall_report_both` is the entry point that cannot report one without
the other, and is what a caller should use.  :func:`recall_report` computes the
single level it is asked for and is the building block.  See
``ranking.reduce_over_conformers`` for why molecule level is not a cosmetic
variant of conformer level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ranking import (assert_unique_ids, rank_order,
                      reduce_over_conformers, top_k_order)

__all__ = [
    "recall_at",
    "recall_curve",
    "survivor_recall",
    "PerQueryRecall",
    "RecallReport",
    "recall_report",
    "recall_report_both",
    "EPS_GAMMA_K",
]

#: The k/N' multipliers the epsilon->gamma curve is reported at.  Fixed; a
#: caller may pass more but these are always present.
EPS_GAMMA_K = (1, 2, 5, 10)


def _order(scores, ids):
    return rank_order(np.asarray(scores), None if ids is None else np.asarray(ids))


def recall_at(gold_scores, cand_scores, n_gold, k, ids=None):
    """``R(N, k)`` for one query.

    ``ids`` is the per-library canonical id, used for the tie-break in **both**
    rankings.  Passing it to one and not the other is the asymmetric-tie-break
    bug this library exists to prevent, so there is no way to do that here.
    """
    g = np.asarray(gold_scores)
    a = np.asarray(cand_scores)
    if g.shape != a.shape:
        raise ValueError(f"gold {g.shape} and candidate {a.shape} must match")
    n_gold = int(n_gold)
    k = int(min(k, g.shape[0]))
    G = top_k_order(g, n_gold, ids)
    A = top_k_order(a, k, ids)
    return float(np.intersect1d(G, A, assume_unique=True).size) / float(n_gold)


def survivor_recall(gold_scores, survivor_mask, n_gold, ids=None):
    """``(recall, survivor_fraction)`` for a filter that returns a set.

    ``survivor_mask`` is a boolean over library molecules (or an index array).  This is
    the form a Tier 0/1 filter reports in: it does not rank, it admits.
    """
    g = np.asarray(gold_scores)
    m = np.asarray(survivor_mask)
    if m.dtype != bool:
        b = np.zeros(g.shape[0], bool)
        b[m] = True
        m = b
    G = top_k_order(g, int(n_gold), ids)
    kept = int(m.sum())
    rec = float(m[G].sum()) / float(int(n_gold))
    return rec, kept / float(g.shape[0])


def recall_curve(gold_scores, cand_scores, n_prime, ks=EPS_GAMMA_K, ids=None):
    """The epsilon->gamma curve for one query: ``{k: R(N', k*N')}``.

    Also returns the survivor fraction ``k*N'/M`` for each k, because a recall
    number without its cost is not an operating point.
    """
    m = int(np.asarray(gold_scores).shape[0])
    out = {}
    for k in ks:
        out[int(k)] = {
            "k_over_nprime": int(k),
            "n_retrieved": int(min(k * n_prime, m)),
            "survivor_fraction": float(min(k * n_prime, m)) / float(m),
            "recall": recall_at(gold_scores, cand_scores, n_prime, k * n_prime, ids),
        }
    return out


@dataclass
class PerQueryRecall:
    """One row of the mandatory per-query table."""
    query_index: int
    query_id: str
    level: str                     # "conformer" or "molecule"
    n_prime: int
    m_targets: int
    recall_by_k: dict              # {k: recall}
    n_conf_of_gold_topn: dict = field(default_factory=dict)


def _agg(values):
    v = np.asarray(values, np.float64)
    return {
        "n_queries": int(v.size),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "p10": float(np.percentile(v, 10)),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
    }


@dataclass
class RecallReport:
    """Everything that must be published together.

    ``per_query`` is not optional and is not a debugging aid: the aggregate is
    only interpretable next to it.
    """
    n_prime: int
    ks: tuple
    m_targets: int
    level: str
    per_query: list                 # list[PerQueryRecall]
    aggregate: dict                 # {k: {mean,min,p10,...}}
    survivor_fraction: dict         # {k: float}
    headline: dict                  # aggregate at k = 1, i.e. R(N',N')
    by_conf_count: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "n_prime": self.n_prime,
            "ks": list(self.ks),
            "m_targets": self.m_targets,
            "level": self.level,
            "aggregate": self.aggregate,
            "survivor_fraction": self.survivor_fraction,
            "headline_R_Nprime_Nprime": self.headline,
            "by_conf_count": self.by_conf_count,
            "per_query": [
                {
                    "query_index": r.query_index,
                    "query_id": r.query_id,
                    "recall_by_k": r.recall_by_k,
                }
                for r in self.per_query
            ],
        }

    def per_query_matrix(self, k=1):
        """``[n_query]`` vector of ``R(N', k*N')`` -- the input to a paired
        bootstrap over queries."""
        return np.array([r.recall_by_k[int(k)] for r in self.per_query], np.float64)


def recall_report(gold, cand, n_prime, ks=EPS_GAMMA_K, ids=None, query_ids=None,
                  mol_ids=None, level="conformer"):
    """Build a :class:`RecallReport` over a ``[n_query, n_target]`` score pair.

    ``mol_ids`` (per library) switches on the molecule-level reduction; when it
    is given, ``level='molecule'`` reduces both gold and candidate over
    conformers first, and the report is additionally stratified by the gold
    top-N' molecules' conformer counts (the order-statistics half).
    """
    g = np.atleast_2d(np.asarray(gold))
    a = np.atleast_2d(np.asarray(cand))
    if g.shape != a.shape:
        raise ValueError(f"gold {g.shape} and candidate {a.shape} must match")
    assert_unique_ids(ids)
    nq = g.shape[0]
    query_ids = list(query_ids) if query_ids is not None else [f"q{i}" for i in range(nq)]

    rows, conf_counts_by_query = [], []
    for q in range(nq):
        gq, aq, idq = g[q], a[q], ids
        ncq = None
        if level == "molecule":
            if mol_ids is None:
                raise ValueError("level='molecule' requires mol_ids")
            umol, gbest, ncnt, _ = reduce_over_conformers(gq, mol_ids, ids)
            _, abest, _, _ = reduce_over_conformers(aq, mol_ids, ids)
            gq, aq, idq = gbest, abest, umol
            G = top_k_order(gq, int(n_prime), idq)
            ncq = ncnt[G]
        curve = recall_curve(gq, aq, n_prime, ks, idq)
        rows.append(PerQueryRecall(
            query_index=q, query_id=query_ids[q], level=level,
            n_prime=int(n_prime), m_targets=int(gq.shape[0]),
            recall_by_k={int(k): curve[int(k)]["recall"] for k in ks}))
        conf_counts_by_query.append(ncq)

    aggregate = {int(k): _agg([r.recall_by_k[int(k)] for r in rows]) for k in ks}
    m = rows[0].m_targets
    surv = {int(k): float(min(int(k) * n_prime, m)) / float(m) for k in ks}

    by_conf = {}
    if level == "molecule" and conf_counts_by_query[0] is not None:
        # stratify the *headline* recall by the median conformer count of the
        # query's own gold top-N'.  A per-query statistic needs a per-query
        # stratum key.
        med = np.array([np.median(c) for c in conf_counts_by_query])
        edges = np.unique(np.percentile(med, [0, 33.3, 66.7, 100]))
        lab = np.clip(np.searchsorted(edges, med, side="right") - 1, 0, len(edges) - 2)
        r1 = np.array([r.recall_by_k[int(ks[0])] for r in rows])
        for b in np.unique(lab):
            sel = lab == b
            by_conf[f"median_nconf_in_[{edges[b]:.1f},{edges[b + 1]:.1f}]"] = {
                **_agg(r1[sel]), "n_queries": int(sel.sum())}

    return RecallReport(
        n_prime=int(n_prime), ks=tuple(int(k) for k in ks), m_targets=int(m),
        level=level, per_query=rows, aggregate=aggregate,
        survivor_fraction=surv, headline=aggregate[int(ks[0])], by_conf_count=by_conf)


def recall_report_both(gold, cand, n_prime, ks=EPS_GAMMA_K, ids=None,
                       query_ids=None, mol_ids=None):
    """Both levels, in one call: ``{"conformer": ..., "molecule": ...}``.

    The rule is "never conflate conformer-level and molecule-level recall -- report
    both, always", and the way it is actually broken is not a wrong
    calculation but an omitted one: a caller asks for the level it is thinking
    about and the other is simply absent from the artifact.  This entry point
    cannot do that.

    ``mol_ids`` is per library conformer.  Without it the molecule-level
    reduction cannot be computed, and the returned ``"molecule"`` entry says so
    explicitly rather than being missing -- on a one-conformer-per-molecule
    corpus the two levels coincide and that is a formality; on a library with
    9.48 or 40.41 conformers per molecule it is not.
    """
    out = {"conformer": recall_report(gold, cand, n_prime, ks=ks, ids=ids,
                                     query_ids=query_ids, level="conformer")}
    if mol_ids is None:
        out["molecule"] = {
            "reported": False,
            "why": "mol_ids was not given, so the conformer -> molecule "
                   "reduction could not be computed. Both levels are "
                   "required; pass mol_ids (Catalog.mol_ids(gids)).",
        }
    else:
        out["molecule"] = recall_report(gold, cand, n_prime, ks=ks, ids=ids,
                                        query_ids=query_ids, mol_ids=mol_ids,
                                        level="molecule")
    return out

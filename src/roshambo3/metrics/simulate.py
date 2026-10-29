"""Cost a proposed approximation *before* implementing it.

The inherited linearization -- "decade width w ~ 0.056, so a score
error eps moves an item's rank by 10^(eps/w)" -- is a rule of thumb about the
*shape of the score distribution*, not about recall, and it says nothing about
the joint behavior of many perturbed items competing for the same N' slots.
Do not trust it.  Simulate instead:

    take the gold score vector, add the *empirical* Delta distribution
    (resampled preserving its measured dependence on molecule size and on the
    gold score, not an independent-noise assumption), re-rank, read off recall.

That is exact given the Delta distribution, it costs milliseconds, and it is
the correct way to set an accuracy budget for a proposed approximation.

Two models are provided:

``EmpiricalDelta`` -- fit to a *measured* (gold, candidate) pair.  Resamples
Delta from within cells of (gold-score decile x HAC decile), so both measured
dependences are preserved.  Scaling the sampled Delta by ``scale`` sweeps the
error magnitude while keeping its shape and its conditioning.

``GaussianDelta`` -- an explicit sd, for a filter that does not exist yet and
therefore has no empirical Delta.  It is the *optimistic* model: real score
errors are heavy-tailed and correlated with size, and both make recall worse
at the same sd.  Reported as such.

Both feed :func:`simulate_recall`, which returns the epsilon->gamma curve:
recall of the gold top-N' as a function of survivor fraction.
"""

from __future__ import annotations

import numpy as np

from .ranking import top_k_order
from .recall import EPS_GAMMA_K
from .stats import bca_ci

__all__ = ["EmpiricalDelta", "GaussianDelta", "simulate_recall",
           "epsilon_gamma_curve", "decade_width"]


def decade_width(scores, q_lo=1e-4, q_hi=1e-3):
    """Measured ``dT`` per 10x change in rank near the retrieval quantile.

    Reported for continuity with the inherited linearisation, and so a reader
    can see whether the 0.056 figure holds on their own data.  It is a
    *diagnostic*, never an input to a recall claim.
    """
    s = np.sort(np.asarray(scores, np.float64))[::-1]
    m = s.shape[0]
    i_lo = max(int(q_lo * m) - 1, 0)
    i_hi = max(int(q_hi * m) - 1, 0)
    if i_hi <= i_lo:
        return float("nan")
    return float((s[i_lo] - s[i_hi]) / np.log10(float(i_hi + 1) / float(i_lo + 1)))


class GaussianDelta:
    """Independent N(bias, sd) score error.  The optimistic model."""

    kind = "gaussian"

    def __init__(self, sd, bias=0.0):
        self.sd = float(sd)
        self.bias = float(bias)

    def cell_index(self, gold, hac=None):
        """No conditioning; every item is in the same cell."""
        return np.zeros(np.asarray(gold).shape[0], np.int64)

    def sample_cells(self, cells, rng, scale=1.0):
        return self.bias * scale + rng.normal(0.0, self.sd * scale, size=cells.shape)

    def sample(self, gold, hac=None, rng=None, scale=1.0):
        rng = np.random.default_rng(0) if rng is None else rng
        return self.sample_cells(self.cell_index(gold, hac), rng, scale)

    def describe(self):
        return {"kind": self.kind, "sd": self.sd, "bias": self.bias,
                "caveat": ("independent Gaussian noise is optimistic: real "
                           "score errors are heavy-tailed and size-correlated, "
                           "and both cost more recall at the same sd")}


class EmpiricalDelta:
    """Delta resampled within (gold-score decile x HAC decile) cells.

    ``fit`` needs a measured candidate.  ``sample`` then produces a Delta for
    an arbitrary gold vector by drawing, for each item, from the cell its own
    (gold score, HAC) lands in -- so the measured dependence on both is carried
    over instead of being averaged into one sd.
    """

    kind = "empirical"

    def __init__(self, score_edges, hac_edges, pool, offset, size, n_hac):
        self.score_edges = score_edges
        self.hac_edges = hac_edges
        self.pool = pool            # all cell samples, concatenated
        self.offset = offset        # [n_cells] start of each cell in pool
        self.size = size            # [n_cells] length of each cell
        self.n_hac = int(n_hac)

    @classmethod
    def fit(cls, gold, cand, hac, n_score_bins=10, n_hac_bins=10,
            max_per_cell=200_000, seed=20260816):
        g = np.asarray(gold, np.float64).ravel()
        d = (np.asarray(cand, np.float64).ravel() - g)
        h = np.asarray(hac, np.float64).ravel()
        if h.shape != g.shape:
            h = np.tile(h, g.size // h.size)
        se = np.unique(np.quantile(g, np.linspace(0, 1, n_score_bins + 1)))
        he = np.unique(np.quantile(h, np.linspace(0, 1, n_hac_bins + 1)))
        si = np.clip(np.searchsorted(se, g, "right") - 1, 0, len(se) - 2)
        hi = np.clip(np.searchsorted(he, h, "right") - 1, 0, len(he) - 2)
        rng = np.random.default_rng(seed)
        n_s, n_h = len(se) - 1, len(he) - 1
        cell = si * n_h + hi
        order = np.argsort(cell, kind="stable")
        cs, ds = cell[order], d[order]
        bounds = np.searchsorted(cs, np.arange(n_s * n_h + 1))
        pools, offset, size = [], np.zeros(n_s * n_h, np.int64), np.zeros(n_s * n_h, np.int64)
        at = 0
        fb = d if d.size <= max_per_cell else d[rng.choice(d.size, max_per_cell,
                                                           replace=False)]
        for c in range(n_s * n_h):
            v = ds[bounds[c]:bounds[c + 1]]
            if v.size > max_per_cell:
                v = v[rng.choice(v.size, max_per_cell, replace=False)]
            if v.size == 0:
                v = fb           # empty cell falls back to the pooled Delta
            pools.append(v)
            offset[c] = at
            size[c] = v.size
            at += v.size
        return cls(se, he, np.concatenate(pools), offset, size, n_h)

    def cell_index(self, gold, hac=None):
        g = np.asarray(gold, np.float64).ravel()
        si = np.clip(np.searchsorted(self.score_edges, g, "right") - 1,
                     0, len(self.score_edges) - 2)
        if hac is None:
            hi = np.zeros(g.size, np.int64)
        else:
            h = np.asarray(hac, np.float64).ravel()
            hi = np.clip(np.searchsorted(self.hac_edges, h, "right") - 1,
                         0, len(self.hac_edges) - 2)
        return (si * self.n_hac + hi).astype(np.int64)

    def sample_cells(self, cells, rng, scale=1.0):
        """Vectorised draw: one uniform per item, indexed into its own cell."""
        c = np.asarray(cells, np.int64)
        u = rng.random(c.size)
        j = self.offset[c] + (u * self.size[c]).astype(np.int64)
        return self.pool[np.minimum(j, self.offset[c] + self.size[c] - 1)] * float(scale)

    def sample(self, gold, hac=None, rng=None, scale=1.0):
        rng = np.random.default_rng(0) if rng is None else rng
        return self.sample_cells(self.cell_index(gold, hac), rng, scale)

    def describe(self):
        return {"kind": self.kind,
                "n_cells": int((self.size > 0).sum()),
                "n_score_bins": len(self.score_edges) - 1,
                "n_hac_bins": len(self.hac_edges) - 1,
                "conditioning": "gold-score decile x HAC decile",
                "note": ("resampled within cells, so the measured dependence "
                         "on n and g is preserved -- not an independent-noise "
                         "assumption")}


def simulate_recall(gold, model, n_prime, ks=EPS_GAMMA_K, hac=None, ids=None,
                    scale=1.0, n_rep=20, seed=20260816):
    """Recall of the gold top-N' after perturbing gold by ``model``, per query.

    ``gold`` is ``[n_query, n_target]``.  Returns per-query, per-k recall
    averaged over ``n_rep`` noise draws, plus the raw per-(query, rep) table so
    a caller can bootstrap over queries.
    """
    G = np.atleast_2d(np.asarray(gold, np.float64))
    nq, m = G.shape
    rng = np.random.default_rng(seed)
    ks = tuple(int(k) for k in ks)
    out = np.zeros((nq, len(ks), n_rep))
    kmax = int(min(max(ks) * n_prime, m))
    for q in range(nq):
        gold_top = top_k_order(G[q], int(n_prime), ids)
        gold_set = np.zeros(m, bool)
        gold_set[gold_top] = True
        cells = model.cell_index(G[q], hac)          # fixed across reps
        for r in range(n_rep):
            pert = G[q] + model.sample_cells(cells, rng, scale=scale)
            order = top_k_order(pert, kmax, ids)
            for ki, k in enumerate(ks):
                kk = int(min(k * n_prime, m))
                out[q, ki, r] = gold_set[order[:kk]].sum() / float(n_prime)
    per_query = out.mean(2)
    return {
        "scale": float(scale),
        "n_prime": int(n_prime),
        "ks": list(ks),
        "n_rep": int(n_rep),
        "model": model.describe(),
        "per_query_recall": per_query,            # [n_query, len(ks)]
        "survivor_fraction": [float(min(k * n_prime, m)) / float(m) for k in ks],
        "mean": [float(per_query[:, i].mean()) for i in range(len(ks))],
        "min": [float(per_query[:, i].min()) for i in range(len(ks))],
        "p10": [float(np.percentile(per_query[:, i], 10)) for i in range(len(ks))],
    }


def epsilon_gamma_curve(gold, model, n_prime, scales, ks=EPS_GAMMA_K, hac=None,
                        ids=None, n_rep=20, seed=20260816, B=2000,
                        floor_note=None):
    """The full epsilon -> gamma surface with a BCa CI over queries.

    ``scales`` sweeps the error magnitude.  For each scale the realized
    per-pair error is measured (so the x-axis is a real epsilon, not a nominal
    one) and recall is reported at every retrieval depth with a bootstrap CI
    resampled over queries.  An operating point is picked from it.
    """
    G = np.atleast_2d(np.asarray(gold, np.float64))
    rng = np.random.default_rng(seed)
    rows = []
    for s in scales:
        sim = simulate_recall(G, model, n_prime, ks, hac, ids, scale=s,
                              n_rep=n_rep, seed=seed)
        probe = model.sample(G[0], hac, rng=rng, scale=s)
        eps = {"sd": float(np.std(probe)),
               "p99_abs": float(np.percentile(np.abs(probe), 99)),
               "median_abs": float(np.median(np.abs(probe)))}
        row = {"scale": float(s), "epsilon": eps, "ks": list(sim["ks"]),
               "survivor_fraction": sim["survivor_fraction"], "recall": {}}
        for i, k in enumerate(sim["ks"]):
            v = sim["per_query_recall"][:, i]
            ci = bca_ci(v, np.mean, B=B, rng=rng)
            row["recall"][int(k)] = {
                "mean": float(v.mean()), "min": float(v.min()),
                "p10": float(np.percentile(v, 10)),
                "ci95": [ci["lo"], ci["hi"]],
                "survivor_fraction": sim["survivor_fraction"][i],
            }
        rows.append(row)
    return {"model": model.describe(), "n_prime": int(n_prime),
            "m_targets": int(G.shape[1]), "n_queries": int(G.shape[0]),
            "curve": rows,
            "floor": floor_note or
                     ("the epsilon axis has a stratum-dependent floor; see "
                      "roshambo3.metrics.floors.EPSILON_FLOOR. Below it a proposed "
                      "approximation is indistinguishable from a recompile.")}

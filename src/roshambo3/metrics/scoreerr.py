"""Score error: signed, stratified, and regressed on size.

``Delta = a - g`` (approximate minus gold).  **Signed, always.**  Collapsing to
``|Delta|`` destroys the only thing that makes the number actionable:

* ``mean Delta > 0`` inflates scores and drags non-hits into the retrieved set
  -- a *precision* cost, paid in wasted rescoring.
* ``mean Delta < 0`` deflates scores and pushes true hits out -- a *recall*
  cost, paid in molecules never seen again.

These are not symmetric in consequence and must never be averaged together.
A smoother objective can also find a genuinely *better* optimum, and an
absolute value makes that improvement look identical to an error.

Three populations, reported separately and labeled:

* **global** -- every pair.  Dominated by the tail nobody retrieves.
* **G_N'** -- the gold top-N'.  The stratum that actually matters.
* **A_N'** -- the candidate top-N'.  Reported and **explicitly labeled
  selection-biased upward**: A_N' is chosen partly *because* its scores are
  high, so its mean Delta is biased positive by construction.  It is here
  because a cascade's precision cost lives in it, not because it is an unbiased
  estimate of anything.

Size bias is fit by OLS with a **query fixed effect** (each query gets its own
intercept, so between-query differences in mean error cannot masquerade as a
size effect), and its CI comes from a **BCa bootstrap resampled over queries**.
Items within a query are not independent; an item bootstrap would give an
interval a factor of sqrt(items-per-query) too narrow.
"""

from __future__ import annotations

import numpy as np

from .floors import (
    SCORE_STRATA, assert_gold_supports_per_pair_claim, improvement_verdict,
    stratum_of,
)
from .ranking import assert_unique_ids, rank_order, top_k_order
from .stats import DEFAULT_B, bca_ci

__all__ = [
    "signed_summary",
    "delta_by_stratum",
    "delta_by_decile",
    "size_bias",
    "score_error_report",
]


def signed_summary(delta):
    """The summary every Delta table reports.  Signed mean and median first;
    magnitudes are present but never alone."""
    d = np.asarray(delta, np.float64).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0}
    return {
        "n": int(d.size),
        "mean_signed": float(d.mean()),
        "median_signed": float(np.median(d)),
        "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "p01_signed": float(np.percentile(d, 1)),
        "p99_signed": float(np.percentile(d, 99)),
        "frac_positive": float(np.mean(d > 0)),
        "frac_negative": float(np.mean(d < 0)),
        "frac_zero": float(np.mean(d == 0)),
        "mean_abs": float(np.abs(d).mean()),
        "p99_abs": float(np.percentile(np.abs(d), 99)),
        "max_abs": float(np.abs(d).max()),
        "interpretation": ("positive mean inflates scores -> precision cost; "
                           "negative mean deflates -> recall cost"),
    }


def delta_by_stratum(delta, t_ref):
    """Signed Delta conditioned on the **reference** score stratum.

    Stratifying on the candidate's own score would sort pairs by their own
    error, which is the classic way to make an approximation look good.
    """
    d = np.asarray(delta, np.float64).ravel()
    t = np.asarray(t_ref, np.float64).ravel()
    out = {}
    for name, lo, hi in SCORE_STRATA:
        m = (t >= lo) & (t < hi)
        out[name] = signed_summary(d[m])
        out[name]["stratum_bounds"] = [lo, hi]
    return out


def delta_by_decile(delta, key, n_bins=10, label="decile"):
    """Signed mean Delta per decile of ``key``.

    Required as an explicit table because a single linear fit hides a U-shape,
    and a size-bucketed kernel plausibly introduces a per-bucket *step* that a
    linear beta averages to nothing.
    """
    d = np.asarray(delta, np.float64).ravel()
    k = np.asarray(key, np.float64).ravel()
    ok = np.isfinite(d) & np.isfinite(k)
    d, k = d[ok], k[ok]
    if d.size == 0:
        return {}
    edges = np.unique(np.quantile(k, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, k, side="right") - 1, 0, len(edges) - 2)
    out = {}
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        out[f"{label}_{b}"] = {
            "range": [float(edges[b]), float(edges[b + 1])],
            "key_mean": float(k[m].mean()),
            **signed_summary(d[m]),
        }
    return out


def _ols_with_query_fe(delta, x, query_index):
    """Slope of ``delta ~ x`` after removing a per-query intercept.

    Within-transformation: subtract each query's own mean from both sides.
    Algebraically identical to dummy-variable OLS on the slope, and O(n).
    """
    d = np.asarray(delta, np.float64)
    xx = np.asarray(x, np.float64)
    q = np.asarray(query_index, np.int64)
    nq = int(q.max()) + 1
    cnt = np.bincount(q, minlength=nq).astype(np.float64)
    cnt[cnt == 0] = 1.0
    dm = d - (np.bincount(q, weights=d, minlength=nq) / cnt)[q]
    xm = xx - (np.bincount(q, weights=xx, minlength=nq) / cnt)[q]
    sxx = float(np.dot(xm, xm))
    if sxx <= 0:
        return float("nan")
    return float(np.dot(xm, dm) / sxx)


def size_bias(delta, hac, query_index, B=DEFAULT_B, alpha=0.05, rng=None):
    """OLS slope of signed Delta on heavy-atom count, with a query fixed effect.

    Units: **Tanimoto per heavy atom**.  The CI is a BCa bootstrap that
    resamples *queries* (all of a query's pairs move together), never items.

    Returns beta, its CI, the per-query beta spread, and the HAC decile table
    that a linear fit cannot substitute for.
    """
    d = np.asarray(delta, np.float64).ravel()
    h = np.asarray(hac, np.float64).ravel()
    q = np.asarray(query_index, np.int64).ravel()
    ok = np.isfinite(d) & np.isfinite(h)
    d, h, q = d[ok], h[ok], q[ok]
    uq, qidx = np.unique(q, return_inverse=True)
    beta = _ols_with_query_fe(d, h, qidx)

    # Bootstrap unit = a query.  The within-query fixed-effect estimator is
    # beta = sum_q Sxy_q / sum_q Sxx_q with per-query centered cross-products,
    # so a cluster resample only needs those two numbers per query -- exact,
    # and O(n_queries) per replicate instead of O(n_pairs).
    nq_ = uq.size
    cnt = np.bincount(qidx, minlength=nq_).astype(np.float64)
    cnt[cnt == 0] = 1.0
    dbar = np.bincount(qidx, weights=d, minlength=nq_) / cnt
    hbar = np.bincount(qidx, weights=h, minlength=nq_) / cnt
    dc, hc = d - dbar[qidx], h - hbar[qidx]
    Sxy = np.bincount(qidx, weights=hc * dc, minlength=nq_)
    Sxx = np.bincount(qidx, weights=hc * hc, minlength=nq_)
    suff = np.stack([Sxy, Sxx], 1)

    def stat(rows):
        s = np.asarray(rows, np.float64)
        den = s[:, 1].sum()
        return float(s[:, 0].sum() / den) if den > 0 else float("nan")

    ci = bca_ci(suff, stat=stat, alpha=alpha, B=B, rng=rng)
    ci["estimate"] = beta  # the point estimate is the full-sample fit

    per_q = []
    for i in range(nq_):
        if cnt[i] >= 3 and Sxx[i] > 0:
            per_q.append(float(Sxy[i] / Sxx[i]))
    return {
        "beta_tanimoto_per_heavy_atom": beta,
        "ci95": [ci["lo"], ci["hi"]],
        "bootstrap": {k: ci[k] for k in ("B", "z0", "acceleration", "n_queries")},
        "bootstrap_unit": "query (items within a query are not independent)",
        "n_pairs": int(d.size),
        "per_query_beta": {"n": len(per_q),
                           "mean": float(np.mean(per_q)) if per_q else float("nan"),
                           "sd": float(np.std(per_q, ddof=1)) if len(per_q) > 1 else 0.0,
                           "min": float(np.min(per_q)) if per_q else float("nan"),
                           "max": float(np.max(per_q)) if per_q else float("nan")},
        "hac_decile_table": delta_by_decile(d, h, label="hac_decile"),
        "warning": ("a single linear beta averages away a per-bucket step; "
                    "read hac_decile_table before quoting beta"),
    }


def score_error_report(gold, cand, n_prime, hac=None, ids=None, bucket=None,
                       query_ids=None, B=DEFAULT_B, rng=None,
                       per_pair_claims=True, global_subsample=None,
                       global_subsample_seed=20260816):
    """The full sign-aware score-error bundle over a ``[n_query, n_target]`` pair.

    ``per_pair_claims`` gates the gold-scope check: with it on (the default) the
    report *refuses* to emit a per-pair claim on the G_N' population if any of
    those pairs sits below T = 0.55, where gold is not a per-pair reference.
    Aggregate statistics are computed regardless, because gold is sound for
    aggregates at every stratum.
    """
    g = np.atleast_2d(np.asarray(gold, np.float64))
    a = np.atleast_2d(np.asarray(cand, np.float64))
    assert_unique_ids(ids)
    nq, m = g.shape
    query_ids = list(query_ids) if query_ids is not None else [f"q{i}" for i in range(nq)]
    n_prime = int(n_prime)

    gN_idx, aN_idx, qofg, qofa = [], [], [], []
    for q in range(nq):
        G = top_k_order(g[q], n_prime, ids)
        A = top_k_order(a[q], n_prime, ids)
        gN_idx.append(G)
        aN_idx.append(A)
        qofg.append(np.full(G.size, q, np.int64))
        qofa.append(np.full(A.size, q, np.int64))
    gN_idx = np.concatenate(gN_idx)
    aN_idx = np.concatenate(aN_idx)
    qofg = np.concatenate(qofg)
    qofa = np.concatenate(qofa)

    # The "global" population is every pair, which at M = 1e7 is 3.2e8 float64s
    # per array. A deterministic per-query subsample keeps the global panel
    # statistics honest (they are aggregates, and the subsample is uniform
    # within a query) without materialising 5 GB. The subsample size and seed
    # are recorded so the number is reproducible.
    if global_subsample is not None and int(global_subsample) < m:
        gsub = int(global_subsample)
        sel = np.random.default_rng(global_subsample_seed).choice(
            m, size=gsub, replace=False)
        sel.sort()
        d_all = (a[:, sel] - g[:, sel]).ravel()
        t_all = g[:, sel].ravel()
        q_all = np.repeat(np.arange(nq), gsub)
        hac_all = None if hac is None else np.asarray(hac, np.float64)[sel]
    else:
        gsub, sel = None, None
        d_all = (a - g).ravel()
        t_all = g.ravel()
        q_all = np.repeat(np.arange(nq), m)
        hac_all = None if hac is None else np.asarray(hac, np.float64)

    d_gN = a[qofg, gN_idx] - g[qofg, gN_idx]
    t_gN = g[qofg, gN_idx]
    d_aN = a[qofa, aN_idx] - g[qofa, aN_idx]
    t_aN = g[qofa, aN_idx]

    rep = {
        "n_prime": n_prime,
        "n_queries": int(nq),
        "m_targets": int(m),
        "global_population": ("all pairs" if gsub is None else
                              f"deterministic uniform subsample of {gsub} of "
                              f"{m} library molecules per query, seed "
                              f"{global_subsample_seed}"),
        "sign_convention": "Delta = candidate - gold (signed; never |Delta|)",
        "global": signed_summary(d_all),
        "on_gold_top_nprime": signed_summary(d_gN),
        "on_cand_top_nprime": signed_summary(d_aN),
        "on_cand_top_nprime_label":
            "SELECTION-BIASED UPWARD: A_N' is chosen partly because its scores "
            "are high, so mean Delta on it is positive by construction. Use it "
            "for the precision cost of a cascade, never as an unbiased "
            "estimate of score error.",
        "by_stratum_global": delta_by_stratum(d_all, t_all),
        "by_gold_score_decile": delta_by_decile(d_all, t_all, label="gold_decile"),
        "physical_gate_global": improvement_verdict(d_all, t_all),
        "physical_gate_on_gold_top_nprime": improvement_verdict(d_gN, t_gN),
        "per_query_mean_delta_on_gold_top_nprime": {
            query_ids[q]: float(d_gN[qofg == q].mean()) for q in range(nq)},
    }

    mean_gN = bca_ci(np.array([d_gN[qofg == q].mean() for q in range(nq)]),
                     np.mean, B=B, rng=rng)
    rep["mean_delta_on_gold_top_nprime_ci95"] = mean_gN

    if per_pair_claims:
        try:
            assert_gold_supports_per_pair_claim(
                t_gN, what="a per-pair score-error claim on the gold top-N'")
            rep["gold_scope"] = {"per_pair_ok": True,
                                 "min_reference_score": float(t_gN.min())}
        except Exception as e:
            rep["gold_scope"] = {
                "per_pair_ok": False,
                "min_reference_score": float(t_gN.min()),
                "refusal": str(e),
                "frac_gold_topn_below_0.55": float(np.mean(t_gN < 0.55)),
                "note": ("aggregate statistics above are still valid -- gold is "
                         "sound for means, ranks and recall at every stratum. "
                         "Only per-pair adjudication is refused."),
            }
            rep["on_gold_top_nprime_restricted_T>=0.55"] = signed_summary(
                d_gN[t_gN >= 0.55])

    if hac is not None:
        h_gN = np.asarray(hac, np.float64)[gN_idx]
        rep["size_bias_on_gold_top_nprime"] = size_bias(d_gN, h_gN, qofg, B=B, rng=rng)
        rep["size_bias_global"] = size_bias(
            d_all, np.tile(hac_all, nq), q_all, B=B, rng=rng)
    if bucket is not None:
        b_gN = np.asarray(bucket)[gN_idx]
        rep["by_padding_bucket_on_gold_top_nprime"] = {
            f"bucket_{int(b)}": signed_summary(d_gN[b_gN == b])
            for b in np.unique(b_gN)}
    rep["strata_labels_present"] = sorted(set(stratum_of(t_all[:1000]).tolist()))
    return rep

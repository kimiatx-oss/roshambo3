"""Bootstrap, paired comparison, and multiple-comparisons machinery.

Three rules are implemented here rather than described:

1. **Resample over queries, never over items.**  Items within a query are not
   independent -- they share the query molecule, its size, its shape and its
   whole score distribution.  A bootstrap over pairs would report a confidence
   interval a factor of sqrt(items-per-query) too narrow.  Every function in
   this module takes a per-query vector (or a per-query index for a per-item
   quantity) and resamples the *query* index.

2. **BCa, not percentile.**  Recall at N' is bounded above by 1 and its
   sampling distribution is skewed hard against that bound whenever the method
   is good, which is exactly the regime we operate in.  The percentile interval
   is visibly wrong there; BCa's bias correction and acceleration fix it.

3. **The winner's curse is quantified, not warned about.**  With K
   configurations and per-query paired sd ``sigma_d`` over ``n`` queries, the
   selected winner's expected inflation is ``(sigma_d/sqrt(n)) * sqrt(2 ln K)``.
   ``winner_inflation`` returns that number so a sweep can print it next to its
   own result and see whether the "win" is smaller than the selection bias.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bca_ci",
    "paired_bootstrap",
    "winner_inflation",
    "required_n_for_delta",
    "resolvable_delta",
    "holm_bonferroni",
    "MAX_EVAL_FINALISTS",
    "DEFAULT_B",
]

DEFAULT_B = 10_000
MAX_EVAL_FINALISTS = 3
"""EVAL finalists are capped at 3 with Holm-Bonferroni correction.  The cap is
the point: correction controls the error rate, it does not restore the power
spent testing 30 things on the held-out set."""


def _jackknife(values, stat):
    n = values.shape[0]
    out = np.empty(n, np.float64)
    idx = np.arange(n)
    for i in range(n):
        out[i] = stat(values[idx != i])
    return out


def bca_ci(values, stat=np.mean, alpha=0.05, B=DEFAULT_B, rng=None):
    """BCa bootstrap CI of ``stat`` over the **rows** of ``values``.

    ``values`` is a per-query vector (or per-query matrix whose rows are
    resampled as units).  Returns ``dict`` with the point estimate, the
    interval, ``z0``, the acceleration, and B.
    """
    from scipy.stats import norm

    v = np.asarray(values, np.float64)
    n = v.shape[0]
    rng = np.random.default_rng(20260816) if rng is None else rng
    theta_hat = float(stat(v))

    idx = rng.integers(0, n, size=(int(B), n))
    boot = np.array([stat(v[i]) for i in idx], np.float64)

    frac_less = float(np.mean(boot < theta_hat))
    if frac_less <= 0.0:
        frac_less = 1.0 / (2 * B)
    elif frac_less >= 1.0:
        frac_less = 1.0 - 1.0 / (2 * B)
    z0 = float(norm.ppf(frac_less))

    jk = _jackknife(v, stat)
    jk_mean = jk.mean()
    d = jk_mean - jk
    denom = 6.0 * (float(np.sum(d ** 2)) ** 1.5)
    a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    def adj(q):
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z))))

    lo_q, hi_q = adj(alpha / 2.0), adj(1.0 - alpha / 2.0)
    lo = float(np.quantile(boot, np.clip(lo_q, 0.0, 1.0)))
    hi = float(np.quantile(boot, np.clip(hi_q, 0.0, 1.0)))
    return {"estimate": theta_hat, "lo": lo, "hi": hi, "alpha": alpha,
            "B": int(B), "z0": z0, "acceleration": a, "n_queries": int(n),
            "method": "BCa, resampled over queries"}


def paired_bootstrap(base_per_query, cand_per_query, alpha=0.05, B=DEFAULT_B,
                     delta_min=0.01, rng=None):
    """Paired BCa bootstrap of ``mean(cand - base)`` over queries.

    ``real`` is True iff the CI excludes 0 **and** the point estimate exceeds
    ``delta_min`` in magnitude.  Both conditions, always: a statistically
    significant 0.002 recall difference is a measurement of the panel, not of
    the method.
    """
    b = np.asarray(base_per_query, np.float64)
    c = np.asarray(cand_per_query, np.float64)
    if b.shape != c.shape:
        raise ValueError("paired bootstrap needs the same queries on both sides")
    d = c - b
    ci = bca_ci(d, np.mean, alpha=alpha, B=B, rng=rng)
    excludes_zero = (ci["lo"] > 0.0) or (ci["hi"] < 0.0)
    ci.update({
        "delta_min": float(delta_min),
        "ci_excludes_zero": bool(excludes_zero),
        "exceeds_delta_min": bool(abs(ci["estimate"]) >= delta_min),
        "real": bool(excludes_zero and abs(ci["estimate"]) >= delta_min),
        "sd_d": float(d.std(ddof=1)) if d.size > 1 else 0.0,
    })
    return ci


def winner_inflation(sd_d, n, K):
    """Expected inflation of the selected winner's measured score.

    ``(sd_d / sqrt(n)) * sqrt(2 ln K)`` -- the expected maximum of K
    approximately-normal, mean-zero selection errors.  At sd_d = 0.02, n = 32:
    +0.0075 (K=10), +0.0092 (K=30), +0.0114 (K=200).
    """
    K = int(K)
    if K < 2:
        return 0.0
    return float(sd_d / np.sqrt(float(n)) * np.sqrt(2.0 * np.log(K)))


def required_n_for_delta(sd_d, delta, z=1.96):
    """``n = (z * sd_d / delta)^2`` -- queries needed to resolve ``delta``."""
    return int(np.ceil((z * float(sd_d) / float(delta)) ** 2))


def resolvable_delta(sd_d, n, z=1.96):
    """The smallest paired difference ``n`` queries can resolve."""
    return float(z * float(sd_d) / np.sqrt(float(n)))


def holm_bonferroni(pvalues, alpha=0.05, labels=None,
                    max_finalists=MAX_EVAL_FINALISTS):
    """Holm-Bonferroni step-down, with the EVAL finalist cap enforced.

    Raises if more than ``max_finalists`` hypotheses are submitted: the cap is
    a protocol rule, not a suggestion, and silently correcting 30 EVAL tests
    would be exactly the behavior it exists to prevent.
    """
    p = np.asarray(pvalues, np.float64)
    if p.size > int(max_finalists):
        raise ValueError(
            f"{p.size} hypotheses submitted but EVAL finalists are capped at "
            f"{max_finalists} (doc/METRICS.md section 8). Correction controls the error "
            "rate; it does not restore the power spent testing many things on "
            "held-out data. Select on the TUNE panel, then bring at most "
            f"{max_finalists} to EVAL.")
    labels = list(labels) if labels is not None else [f"H{i}" for i in range(p.size)]
    order = np.argsort(p, kind="stable")
    m = p.size
    out = [None] * m
    rejected_so_far = True
    for rank, i in enumerate(order):
        thresh = alpha / (m - rank)
        rej = bool(rejected_so_far and p[i] <= thresh)
        rejected_so_far = rej
        out[i] = {"label": labels[i], "p": float(p[i]), "threshold": float(thresh),
                  "reject": rej, "step": int(rank + 1)}
    return {"alpha": alpha, "m": int(m), "results": out}

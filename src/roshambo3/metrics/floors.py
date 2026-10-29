"""The measured noise floors, as data, stratified by score.

**A single scalar floor is wrong here by three orders of magnitude.**  Two
independent measurements (a seven-build arithmetic ensemble, a
rotate-and-re-prepare gauge audit) both find the same cliff at T ~ 0.55: below
it the method has no resolution at all, above it the ensemble agrees to ~1e-6
or better.  Quoting the overall p99.9 (2.9e-3) as "the floor" would be
pessimistic by three orders exactly where a screen ranks, and would license
approximations that are three orders too loose at the top of the list.

Two different floors, which must not be confused:

``EPSILON_FLOOR`` -- **arithmetic irreproducibility**.  The spread of the
per-pair score across seven real builds of the same source.  This is the floor
on epsilon in the epsilon->gamma curve: an approximation whose score error is
below it is indistinguishable from a recompile.

``GAUGE_FLOOR`` -- **gauge (input-orientation) non-invariance**.
The score's failure to be invariant under a rigid rotation of the *input*
molecules, which survives any amount of arithmetic pinning because it is a
property of the principal-axis frame, not of the build.

The gauge floor was misquoted project-wide as "p99 = 5.8e-4".  Measured, its
p99 is **4.5e-6 on the retrieved list** and **5.8e-5 over all pairs**; 5.8e-4
is near the *max* on the retrieved list.  Any threshold set against 5.8e-4 is
roughly two orders of magnitude too loose.  ``GAUGE_FLOOR`` carries the
distribution, and :func:`gauge_floor` refuses to return a scalar without being
told which statistic and which stratum.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SCORE_STRATA",
    "EPSILON_FLOOR",
    "GAUGE_FLOOR",
    "GOLD_SCOPE",
    "stratum_of",
    "epsilon_floor",
    "gauge_floor",
    "below_floor",
    "improvement_verdict",
    "assert_gold_supports_per_pair_claim",
]

#: The 0.55 edge is the cliff four independent measurements found.
SCORE_STRATA = (
    ("T<0.4", 0.0, 0.4),
    ("0.4<=T<0.55", 0.4, 0.55),
    ("0.55<=T<0.7", 0.55, 0.7),
    ("T>=0.7", 0.7, 1.01),
)

#: Ensemble spread of the per-pair shape Tanimoto across **seven real
#: arithmetic builds**.  n is small in the middle strata -- read ``n``
#: before quoting a percentile from them.
EPSILON_FLOOR = {
    "T<0.4":        {"n": 174, "median": 2.078e-3, "mean": 2.961e-3,
                     "p90": 7.156e-3, "p99": 1.195e-2, "max": 1.325e-2},
    "0.4<=T<0.55":  {"n": 283, "median": 4.265e-3, "mean": 3.502e-3,
                     "p90": 7.630e-3, "p99": 1.667e-2, "max": 3.191e-2},
    "0.55<=T<0.7":  {"n": 43,  "median": 4.172e-7, "mean": 2.161e-3,
                     "p90": 7.716e-3, "p99": 1.114e-2, "max": 1.181e-2},
    "T>=0.7":       {"n": 250, "median": 2.980e-7, "mean": 3.266e-7,
                     "p90": 5.364e-7, "p99": 7.761e-7, "max": 8.941e-7},
}

#: Whole-panel figures from the same ensemble (463,200 pairs, all strata
#: pooled).  Present so that a *panel-wide* statistic has a panel-wide floor;
#: never use these for a top-of-list claim.
EPSILON_FLOOR_ALL_PAIRS = {"n": 463_200, "median": 1.788e-7, "p90": 4.172e-7,
                           "p99": 1.101e-4, "p99_9": 2.864e-3, "max": 3.191e-2}

#: The top-scoring 250 pairs -- the population a screen actually ranks.
EPSILON_FLOOR_TOP250 = {"n": 250, "median": 2.980e-7, "max": 8.941e-7}

#: Per-pair |dT| between two ARCHITECTURES running identical source, flags and
#: inputs: Volta sm_70 against Blackwell sm_120, CUDA 12.9 both sides,
#: n = 463,200, stratified on :data:`SCORE_STRATA`.
#:
#: **This is the repo's measurement of a CROSS-IMPLEMENTATION difference**, and
#: it is a different quantity from :data:`EPSILON_FLOOR` even though both are
#: per-pair |dT| by stratum. The epsilon floor varies compiler FLAGS on one
#: machine and one architecture; this varies the ARCHITECTURE with the flags
#: held fixed. That distinction matters: the numerics
#: policy pins arithmetic *within* one implementation, while an anchor compares
#: *across* them.
#:
#: **VALUES ARE FULL PRECISION AND MUST STAY THAT WAY.** float32 score deltas
#: land on a lattice, so a rounded threshold moves pairs across it wholesale. An
#: earlier version carried these to four significant figures, and because the
#: T>=0.7 p99 is exactly ``6 * 2**-24`` the constant landed 2.8e-11 BELOW that
#: lattice point and flipped the 42 pairs sitting exactly on it to "exceeding":
#: 10.93 percent reported where the true figure is 6.05.
#:
#: **STEPS MISMATCH, AND IT IS NOT SMALL.** This table is the **100-step**
#: distribution -- its ``all_pairs`` block is bit-identical to the ``steps=100``
#: row of the steps sweep. The shipped schedule and the anchor
#: fixture run **25 steps**, where the same measurement's all-pairs p99 is
#: :data:`ARCHITECTURE_DELTA_ALL_PAIRS_25_STEPS`, **87x tighter**, and the
#: optimizer basin-flip rate is 0.003 percent against 1.22. So using this table
#: as a 25-step tolerance is loose by roughly two orders, in the direction that
#: makes a comparison pass. The repo has **no per-stratum cross-architecture
#: distribution at 25 steps**; producing one is owed. Until then a passing per-pair verdict is weak
#: evidence and reads as "not grossly different" rather than "agrees".
ARCHITECTURE_DELTA = {
    "T<0.4":           {"n": 209231, "median": 2.9802322387695312e-08,
                      "p90": 1.4901161193847656e-07, "p99": 4.947930574417201e-05,
                      "p99_9": 0.0011349514126778454, "max": 0.007512837648391724},
    "0.4<=T<0.55":     {"n": 201308, "median": 5.960464477539063e-08,
                      "p90": 1.7881393432617188e-07, "p99": 1.9995272159576208e-05,
                      "p99_9": 0.001286495536565983, "max": 0.008883506059646606},
    "0.55<=T<0.7":     {"n": 47389, "median": 5.960464477539063e-08,
                      "p90": 1.7881393432617188e-07, "p99": 9.012222290040624e-07,
                      "p99_9": 0.0003464393615723021, "max": 0.005238711833953857},
    "T>=0.7":          {"n": 5272, "median": 0.0,
                      "p90": 1.7881393432617188e-07, "p99": 3.5762786865234375e-07,
                      "p99_9": 4.313826560974217e-06, "max": 3.165006637573242e-05},
}

#: The same architecture comparison at the SHIPPED 25 steps, all pairs pooled
#: Present so the size of the step-count
#: gap in :data:`ARCHITECTURE_DELTA` is visible rather than inferred. Not usable
#: as a stratum tolerance -- ``doc/METRICS.md`` section 6 is explicit that a
#: single scalar is wrong here by three orders -- but it bounds how much of that
#: table is amplification rather than architecture.
ARCHITECTURE_DELTA_ALL_PAIRS_25_STEPS = 2.980232238769531e-07

#: Reversed reduction order, all pairs.
#: **This, not** :data:`ARCHITECTURE_DELTA`, **is the measurement that varies
#: reduction order**: the architecture run used identical source on both GPUs and therefore
#: reduces in the same order on both. It is the honest citation for a CPU
#: vector-width comparison, where the width change IS a reassociation.
#: At the same panel and step count it is 2.83e-5 against the architecture
#: axis's 2.60e-5 -- the same distribution.
REDUCTION_ORDER_DELTA_ALL_PAIRS = {'n': 463200, 'frac_zero': 0.11442357512953368, 'median': 8.940696716308594e-08, 'p90': 2.980232238769531e-07, 'p99': 2.8312355279922624e-05, 'p99_9': 0.0012503643035889356, 'max': 0.01656678318977356}

#: The gauge audit, keyed by gold step count.  ``retrieved_list`` is the top-10-per-query
#: population; ``all_pairs`` is the whole 4.8M-pair panel.  Both sides
#: (query-rotated and library-rotated) are recorded; the library side is the
#: larger and is what a screen is exposed to, so it is the default.
GAUGE_FLOOR = {
    "GOLD-400": {
        "target_side": {"p99_retrieved_list": 6.56e-7, "p99_all_pairs": 1.18e-5,
                        "p99_9_retrieved_list": 1.10e-4, "p99_9_all_pairs": 5.51e-4},
        "query_side": {"p99_retrieved_list": 4.77e-7, "p99_all_pairs": 6.11e-6,
                       "p99_9_retrieved_list": 7.94e-5, "p99_9_all_pairs": 3.06e-4},
    },
    "GOLD-100": {
        "target_side": {"p99_retrieved_list": 4.47e-6, "p99_all_pairs": 5.78e-5,
                        "p99_9_retrieved_list": 1.12e-4, "p99_9_all_pairs": 1.72e-3},
        "query_side": {"p99_retrieved_list": 1.31e-6, "p99_all_pairs": 2.27e-5,
                       "p99_9_retrieved_list": 9.61e-5, "p99_9_all_pairs": 1.13e-3},
    },
}

#: The retrieved-list p99 this project quotes for threshold-setting:
#: **4.5e-6**, not 5.8e-4.
GAUGE_FLOOR_RETRIEVED_P99 = 4.47e-6
GAUGE_FLOOR_ALL_PAIRS_P99 = 5.78e-5

#: What gold licenses, per stratum (doc/METRICS.md section 7).
GOLD_SCOPE = {
    "T>=0.7":      {"status": "SPECIFICATION",
                    "per_pair_ok": True,  "resolution_p99": 1.17e-6},
    "0.55<=T<0.7": {"status": "AGGREGATE REFERENCE",
                    "per_pair_ok": True,  "resolution_p99": 2.80e-2},
    "0.4<=T<0.55": {"status": "ADVISORY",
                    "per_pair_ok": False, "resolution_p99": 6.98e-2},
    "T<0.4":       {"status": "NOT A REFERENCE",
                    "per_pair_ok": False, "resolution_p99": 6.45e-2},
}

#: Below this reference score, gold cannot adjudicate a per-pair claim at all.
PER_PAIR_FLOOR_T = 0.55


class GoldScopeError(RuntimeError):
    """Raised when a per-pair claim is attempted where gold cannot support it."""


def stratum_of(t_ref):
    """Label each reference score with its stratum name."""
    t = np.asarray(t_ref, np.float64)
    out = np.empty(t.shape, dtype=object)
    for name, lo, hi in SCORE_STRATA:
        out[(t >= lo) & (t < hi)] = name
    out[t >= SCORE_STRATA[-1][2]] = SCORE_STRATA[-1][0]
    return out


def epsilon_floor(t_ref, statistic="median"):
    """The arithmetic floor at each pair's stratum, elementwise.

    ``statistic`` selects which order statistic of the ensemble spread to use.
    ``median`` is the default because it is what "the builds agree to X" means
    for a typical pair; use ``p99`` when bounding a per-pair claim.
    """
    lab = stratum_of(t_ref)
    out = np.empty(np.asarray(t_ref).shape, np.float64)
    for name in EPSILON_FLOOR:
        out[lab == name] = EPSILON_FLOOR[name][statistic]
    return out


def gauge_floor(stratum="retrieved_list", statistic="p99", gold="GOLD-400",
                side="target_side"):
    """The gauge floor, with the stratum and statistic named.

    There is deliberately no zero-argument form.  Every historical misuse of
    this number came from quoting one scalar for two different populations.
    """
    key = f"{statistic}_{stratum}"
    try:
        return float(GAUGE_FLOOR[gold][side][key])
    except KeyError:
        raise KeyError(
            f"no gauge floor for gold={gold!r} side={side!r} "
            f"statistic={statistic!r} stratum={stratum!r}; available: "
            f"{sorted(GAUGE_FLOOR[gold][side])}") from None


def below_floor(delta, t_ref, statistic="median", kind="arithmetic"):
    """Elementwise: is this score difference below the noise floor of its own
    stratum?  ``kind`` in {'arithmetic', 'gauge'}."""
    d = np.abs(np.asarray(delta, np.float64))
    if kind == "arithmetic":
        f = epsilon_floor(t_ref, statistic)
    elif kind == "gauge":
        f = np.full(d.shape, gauge_floor("all_pairs", statistic))
    else:
        raise ValueError("kind must be 'arithmetic' or 'gauge'")
    return d < f


def improvement_verdict(delta, t_ref, statistic="median"):
    """The physical gate required **before** any statistical test.

    "Any change whose entire effect is per-score shifts below the gauge floor
    cannot be claimed to improve accuracy" -- because statistical significance
    below the floor is measuring noise, not signal.  Applied per stratum,
    against both floors, and returning a verdict rather than a number so a
    caller cannot accidentally use it as a p-value.
    """
    d = np.abs(np.asarray(delta, np.float64))
    t = np.asarray(t_ref, np.float64)
    if d.size == 0:
        return {"verdict": "NO_DATA"}
    arith = epsilon_floor(t, statistic)
    gauge_all = gauge_floor("all_pairs", "p99")
    gauge_ret = gauge_floor("retrieved_list", "p99")
    frac_above_arith = float(np.mean(d >= arith))
    frac_above_gauge = float(np.mean(d >= gauge_ret))
    if frac_above_arith < 0.01 and frac_above_gauge < 0.01:
        verdict = "BELOW_FLOOR_NOT_AN_IMPROVEMENT"
    elif frac_above_arith < 0.01:
        verdict = "BELOW_ARITHMETIC_FLOOR"
    else:
        verdict = "ABOVE_FLOOR_TESTABLE"
    return {
        "verdict": verdict,
        "frac_pairs_above_stratum_arithmetic_floor": frac_above_arith,
        "frac_pairs_above_gauge_retrieved_p99": frac_above_gauge,
        "gauge_floor_retrieved_p99": gauge_ret,
        "gauge_floor_all_pairs_p99": gauge_all,
        "statistic": statistic,
        "note": ("a change entirely below its stratum's floor is a recompile, "
                 "not an improvement; run no statistical test on it"),
    }


def assert_gold_supports_per_pair_claim(t_ref, what="this per-pair claim"):
    """Refuse a per-pair claim below T = 0.55.

    Gold is SPECIFICATION at T >= 0.7, AGGREGATE
    REFERENCE at 0.55-0.7, ADVISORY at 0.4-0.55 and **NOT A REFERENCE** below
    0.4.  It is sound for aggregates (means, ranks, recall) at every stratum
    and unsound per-pair below 0.55, where it is wrong by ~1e-2 for a large
    minority of pairs.  This function is called by
    :mod:`roshambo3.metrics.scoreerr` on every per-pair report so the refusal is
    executable, not advisory.
    """
    t = np.asarray(t_ref, np.float64)
    if t.size and float(np.min(t)) < PER_PAIR_FLOOR_T:
        n_bad = int(np.sum(t < PER_PAIR_FLOOR_T))
        raise GoldScopeError(
            f"refusing {what}: {n_bad} of {t.size} pairs have a reference score "
            f"below T = {PER_PAIR_FLOOR_T}, where GOLD is ADVISORY or NOT A "
            "REFERENCE (doc/METRICS.md section 7). Gold is sound for aggregates "
            "(means, ranks, recall) at every stratum but unsound per-pair "
            "below 0.55 -- it is exact for the median retrieved pair and wrong "
            "by ~1e-2 for 12.9%% of them. Restrict to T >= 0.55, or make an "
            "aggregate claim instead.")

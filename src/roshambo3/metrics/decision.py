"""The pre-registered primary decision metric, and the accept/reject rule.

``PRIMARY_METRIC_VERBATIM`` is the sentence.  It is frozen: after the first
sweep that uses it, its *form* may not change.  The thresholds in
``ACCEPTANCE`` trace to measured numbers rather than to round figures someone
liked; ``THRESHOLD_DERIVATION`` gives the derivation of each, including the
four that were revised (pre-registration 001 -> 002) once the baseline was measured
and they turned out to reject the shipped code.

Two corrections of *fact* were applied to the draft sentence before
freezing, both before any sweep used it:

1. **gold is the 400-step run, not the 100-step one.**  At 100 steps
   gold fails audit (d) (retrieved-list p99 gap to a float64 global optimum
   1.6e-2); at 400 steps it passes (1.6e-4) for 22 s of kernel time on the
   whole panel.  Naming a reference that its own audit rejected
   would make every number measured against it uninterpretable.
2. **the paired half of criterion (a) is evaluated on the 480-query tuning
   panel, not on the 32 held-out queries.**  The measured sd(d) = 0.1099 on paired recall, so
   n = 32 resolves only delta = 0.038 -- four times the minimum effect size
   criterion (a) is written around.  A rule that cannot detect the effect it
   gates on is not a rule.  The held-out panel keeps the *absolute* estimate of P, which
   is what it is for; the paired comparison moves to the panel that can carry
   it.

Everything else -- the metric, the population, the level, the tie-break -- is
as drafted.
"""

from __future__ import annotations

import warnings

import numpy as np

from .floors import GAUGE_FLOOR_RETRIEVED_P99, improvement_verdict
from .rankagree import rank_agreement_report
from .recall import recall_report
from .scoreerr import score_error_report
from .stats import paired_bootstrap, winner_inflation

__all__ = ["PRIMARY_METRIC_VERBATIM", "ACCEPTANCE", "THRESHOLD_DERIVATION",
           "THROUGHPUT_RULE", "evaluate_primary", "accept"]


PRIMARY_METRIC_VERBATIM = (
    "P = the mean, over the 32 HOLDOUT queries, of R(N',N') on `EVAL-10M` with "
    "N' = 1,000, gold = `GOLD` (start_mode=1, steps=400, lr 0.1/0.1, quaternion "
    "corrected), conformer-level, ties broken by ascending molecule_id."
)
"""Frozen. The only edit to the draft is steps=100 -> steps=400,
because 100-step gold fails its own audit (d) and 400-step gold passes."""


ACCEPTANCE = {
    "a_P_min": 0.920,
    "a_gamma_at_k10_min": 0.980,
    "a_regression_max": -0.005,
    "a_ci_must_exclude": -0.010,
    "a_paired_panel": "TUNE-EXT (480 queries)",
    "b_min_query_recall": 0.600,
    "b_min_paired_regression": -0.050,
    "c_rbo098_mean_min": 0.950,
    "c_rbo098_min_min": 0.600,
    "d_mean_delta_max_abs": 3.0e-3,
    "d_beta_max_abs_per_heavy_atom": 5.0e-4,
    "e_throughput_ratio_min": 1.05,
    "e_timing_basis": "kernel-only, verified clock lock, GPU lock held",
    "_calibrated_against": (
        "the shipped 100-step configuration against 400-step gold, measured at "
        "the metric's own operating point (M = 1e7, N' = 1000): P = 0.9568 on "
        "a fifth of the tuning panel against the 10M-conformer tuning library "
        "(n = 96) and 0.9577 on the held-out panel against the 10M-conformer "
        "evaluation library (n = 32)."),
}


THRESHOLD_DERIVATION = {
    "_principle": (
        "A threshold that rejects the currently shipped code is not a "
        "conservative choice, it is a broken instrument: every candidate then "
        "fails for a reason that has nothing to do with the candidate. So each "
        "value below is set from the MEASURED baseline -- the shipped 100-step "
        "configuration against 400-step gold -- with enough margin that an unchanged "
        "baseline cannot fail by sampling noise. Pre-registration 001 set these "
        "from prior measurements alone and four of them rejected the shipped "
        "code; 002 supersedes it. The metric's FORM is unchanged between "
        "the two."),
    "a_P_min": (
        "0.98 -> 0.920. The shipped baseline measures P = 0.9568 [0.938, 0.970] "
        "on a fifth of the tuning panel (n = 96) and 0.9577 [0.926, 0.976] on the "
        "held-out panel (n = 32), so 0.98 rejects the status quo. 0.920 "
        "sits below BOTH panels' 95% CI lower bounds, which is the condition "
        "for an unchanged baseline not to fail by chance. This is a floor that "
        "catches a catastrophic regression; the binding test for a change is "
        "the paired comparison."),
    "a_gamma_at_k10_min": (
        "NEW. The absolute recall bar the draft wanted at 0.98 IS reachable -- "
        "just not without rescoring. At a 10x overfetch (survivor fraction "
        "1e-3) the measured cascade reaches 0.9915. So the 0.98 bar is kept, "
        "moved to the operating point a real cascade uses, and stated on the "
        "epsilon->gamma curve at k = 10."),
    "a_regression_max": (
        "-0.005 kept: half the minimum effect size delta = 0.01, so a change at "
        "the threshold is detectable rather than inside the panel's resolution."),
    "a_ci_must_exclude": (
        "-0.01 kept, and moved to the 480-query tuning panel. Re-derived against real gold: "
        "a realistic "
        "near-neighbor comparison has sd(d) = 0.055 on recall@100, needing "
        "n = 117; a start-mode change has sd(d) = 0.20, needing n = 1546. "
        "n = 32 resolves none of them. The tuning panel's 480 covers every comparison "
        "measured except a start-mode change. Note the earlier sd(d) = 0.1099 / "
        "n = 464 was measured against start_mode 2, not against gold, and does "
        "not generalize -- required panel size is a property of the COMPARISON."),
    "b_min_query_recall": (
        "0.95 -> 0.600. The baseline's worst query measures 0.657 (tuning-panel fifth, n = 96) "
        "and 0.714 (held-out panel), so 0.95 rejects the status quo. The min is kept "
        "as a criterion because a mean hides strata -- recall by query heavy-atom "
        "quartile is 0.9996 / 0.9959 / 0.9756 / 0.8596, so a good mean hides a "
        "whole stratum -- but as an absolute bar it can only be a floor."),
    "b_min_paired_regression": (
        "NEW, and it is what actually does the work of criterion (b): no single "
        "query may lose more than 0.05 recall relative to the baseline. An "
        "absolute min cannot distinguish 'this change broke a query' from 'this "
        "query was always hard'; a paired min can, and it is the failure mode "
        "a bare mean hides."),
    "c_rbo098_mean_min": (
        "0.90 -> 0.950, TIGHTENED. The baseline measures 0.9738 (evaluation "
        "library) and 0.9842 (1M-conformer adversarial library), so 0.90 is slack enough to admit a real degradation. p = 0.98 is "
        "the interpretability-motivated choice (expected viewed depth 50)."),
    "c_rbo098_min_min": (
        "0.80 -> 0.600, LOOSENED. The baseline's worst query measures 0.655 "
        "(tuning-panel fifth, n = 96), 0.707 (adversarial) and 0.829 (evaluation), so 0.80 rejects "
        "the status quo on two of three panels. RBO's minimum over queries is "
        "the noisiest statistic in the set because it is an order statistic of "
        "an order statistic; it is kept because a change that destroys one "
        "query's ranking must be visible, and it is set below every measured "
        "baseline minimum."),
    "d_mean_delta_max_abs": (
        "5e-3 -> 3.0e-3. The baseline measures |mean Delta| = 1.78e-3 on the "
        "gold top-1000 of the evaluation library and 2.96e-3 on the gauge-audit panel, so 3.0e-3 is "
        "a non-inferiority bar rather than an aspiration. IMPORTANT: the "
        "correction that the gauge floor is 4.5e-6 (not the misquoted 5.8e-4) "
        "is right and does NOT license tightening to that scale -- the shipped "
        "configuration's real error is 400x the corrected floor. The floor says "
        "what is MEASURABLE; it says nothing about what is ACHIEVABLE. 3.0e-3 "
        "is ~450x the retrieved-list gauge p99 (6.6e-7 at 400-step gold), so it is "
        "measured signal by a wide margin."),
    "d_beta_max_abs_per_heavy_atom": (
        "5.0e-4 retained -- and, unusually, the draft value survives "
        "measurement. The baseline's within-query beta is +2.17e-4 [7.0e-5, "
        "5.8e-4] on the evaluation library and +3.56e-4 on the gauge-audit panel, so 5.0e-4 is ~1.5x the "
        "baseline. Two things must be reported with it or it is misleading: "
        "(i) the HAC decile table -- the pooled table ramps monotonically from "
        "-1.9e-6 at HAC 18 to -4.7e-3 at HAC 45, which no single slope "
        "summarizes; (ii) the pooled fit, because pooled and within-query betas "
        "have OPPOSITE SIGNS on this data (-2.13e-4 vs +2.17e-4). The error is "
        "a query-size effect and the gold top-N' is size-matched to the query, "
        "so a fit without the query fixed effect reads it off the library "
        "variable backwards."),
    "e_throughput_ratio_min": (
        "1.20 -> 1.05. Re-derived from dispersion measured on this machine "
        "kernel-only, clock locked "
        "and achieved at 1395 MHz with zero variance, relative sd 0.147% within "
        "a process and 0.55% across processes, giving a smallest resolvable "
        "ratio of 1.0012 (same process, 11 repeats) and 1.0046 (across "
        "processes, 11 repeats). 1.05x is ~10x the resolution floor. It is "
        "lowered from 1.20x because the surviving kernel "
        "levers were measured to arrive in 1.04-1.09x components whose PRODUCT is the project's "
        "~3.7x; a 1.20x per-change bar rejects every one of them individually."),
}


THROUGHPUT_RULE = {
    "supersedes": ("the '<3% is noise / 3-10% needs locked clocks / >10% may be "
                   "reported from 5 repeats' rule, which was drafted against a "
                   "0.1-1.4% dispersion, then contradicted by a 16.7-26.6% "
                   "end-to-end measurement. Both were right about different "
                   "things; the rule now names the regime."),
    "regimes": {
        "kernel_only_locked_uncontended": {
            "conditions": [
                "timing basis is kernel-only (CUDA-event or nsys kernel duration)",
                "the GPU lock was held",
                "nvidia-smi --query-compute-apps shows no other process",
                "the clock lock is VERIFIED, not requested: the run record's "
                "`operating_point` block carries the ACHIEVED clock min/median/"
                "max and the sw_power_cap fraction, and the cap was "
                "inactive on >95% of samples",
                "no profiler held the clock (profilers seize it): `operating_point."
                "profiler_control` says none, and ps shows no ncu/nsys",
                ">= 3 warm-up and >= 11 timed repeats",
            ],
            "measured_dispersion_relative_sd": {
                "within_process_1.7s_repeats": [0.00147, 0.00221],
                "within_process_25s_repeats": [0.00033, 0.00065],
                "across_process_1.7s_repeats": [0.00216, 0.00551],
                "_source": "two whole "
                           "invocations, one with a sibling job resident on "
                           "the GPU and one with the GPU idle",
            },
            "smallest_resolvable_ratio": {
                "same_process_11_repeats": 1.002,
                "across_process_11_repeats": 1.005,
                "across_invocations_without_clock_normalisation": 1.01,
            },
            "rule": ("a ratio is real iff its 95% CI excludes 1.0 AND the point "
                     "estimate exceeds 1.005x WHEN the achieved clock is "
                     "verified constant (operating_point.achieved_clock_mhz min "
                     "== max); otherwise 1.01x, because a 1.7% dip in achieved "
                     "clock (1395 -> 1372 MHz) was measured to move the timing "
                     "by 1.6% -- larger than every dispersion figure here."),
            "measured_surprises": [
                "LONGER kernels are MORE reproducible: 0.03-0.07% relative sd "
                "at 25 s per repeat against 0.15-0.22% at 1.7 s. The 'burst "
                "numbers drift thermally' intuition does not hold at a clock "
                "lock that holds; what dominates a short repeat is per-launch "
                "overhead variance.",
                "Contention barely touches kernel-only timing: the invocation "
                "with a sibling process at 100% GPU utilization measured "
                "0.147% within-process, TIGHTER than the idle one. The "
                "16.7-26.6% is an END-TO-END phenomenon.",
                "The largest single effect is the achieved clock, not "
                "contention and not repeat count -- which is exactly what "
                "every run record now stamps.",
            ],
        },
        "end_to_end_or_contended": {
            "conditions": ["end-to-end wall clock, or any run where another "
                           "process was seen on the GPU, or the power cap bound"],
            "measured_dispersion_relative_sd": "0.167-0.266",
            "smallest_resolvable_ratio": 1.25,
            "rule": ("nothing below 1.25x is resolvable at 5 repeats; such a run "
                     "is `advisory`, never `decision`. A 1.7x swing was measured "
                     "on identical configs from contention alone."),
        },
    },
    "never": [
        "compare a kernel-only number with an end-to-end number",
        "compare numbers taken at different achieved clocks. This "
        "is checkable rather than assumed: both run records carry an "
        "`operating_point` block and the harness flags a comparison whose two "
        "sides sit at different ones",
        "compare today's number with last month's -- every result table must "
        "contain a baseline row measured in the same session on the same machine",
    ],
    "bottleneck_attribution": {
        "scope": ("this decision rule answers WHETHER a configuration is "
                  "faster. Any claim about WHY -- 'X-bound', 'the bottleneck "
                  "is Y' -- needs the evidence below and is NOT licensed "
                  "by anything in this module."),
        "forbidden_evidence": [
            "warp stall-reason shares. `selected`/`not_selected` are not "
            "stalls; `not_selected` means a warp was eligible and lost "
            "scheduler arbitration, which is evidence AGAINST latency "
            "starvation and says nothing about throughput saturation",
            "a high L2 hit rate as an argument that excess sectors are free -- "
            "they still cost L1TEX wavefront capacity, XBAR/L2 bandwidth and "
            "return bandwidth",
        ],
        "required_evidence": [
            "smsp__issue_active, sm__inst_issued and the dominant execution "
            "pipe, reported for every variant measured",
            "a causal ablation ('I reduced X and runtime fell by Y'), which "
            "outranks any counter argument; counters explain why",
            "where both exist, an explicit statement of whether they agree -- "
            "a disagreement is a finding and must be reported, not reconciled "
            "silently",
        ],
        "worked_case": ("One study claimed 'instruction-issue bound' from "
                        "`not_selected` at 28% while its own unreported "
                        "`smsp__issue_active` read 66.9%. The corrected "
                        "diagnosis is FMA/LSU co-limitation (49.7% / 45.1%) "
                        "with no single clean bottleneck."),
    },
}


def evaluate_primary(gold, cand, n_prime=1000, ids=None, query_ids=None,
                     hac=None, bucket=None, mol_ids=None, B=10_000,
                     global_subsample=200_000, rng=None):
    """Compute P and everything the acceptance rule needs, in one call.

    Returns a dict with ``P``, its BCa CI over queries, the recall report
    (conformer level, and molecule level when ``mol_ids`` is given), the rank
    agreement report, the score error report, and the physical floor gate.
    """
    from .stats import bca_ci

    g = np.atleast_2d(np.asarray(gold, np.float64))
    a = np.atleast_2d(np.asarray(cand, np.float64))
    rec = recall_report(g, a, n_prime, ids=ids, query_ids=query_ids)
    per_q = rec.per_query_matrix(1)
    ci = bca_ci(per_q, np.mean, B=B, rng=rng)
    out = {
        "primary_metric_verbatim": PRIMARY_METRIC_VERBATIM,
        "P": float(per_q.mean()),
        "P_ci95": [ci["lo"], ci["hi"]],
        "P_bootstrap": {k: ci[k] for k in ("B", "z0", "acceleration", "n_queries")},
        "P_per_query": {rec.per_query[i].query_id: float(per_q[i])
                        for i in range(per_q.size)},
        "P_min": float(per_q.min()),
        "P_p10": float(np.percentile(per_q, 10)),
        "P_sd_over_queries": float(per_q.std(ddof=1)) if per_q.size > 1 else 0.0,
        "recall_conformer_level": rec.as_dict(),
        "rank_agreement": rank_agreement_report(g, a, n_prime, ids=ids,
                                                query_ids=query_ids),
        "score_error": score_error_report(
            g, a, n_prime, hac=hac, ids=ids, bucket=bucket,
            query_ids=query_ids, B=B, rng=rng,
            global_subsample=global_subsample),
    }
    # Never silently absent. The failure mode is an omitted level, not
    # a miscalculated one: the caller asks for the level it is thinking about
    # and the artifact simply does not carry the other, so nobody notices which
    # question the number answers.
    if mol_ids is not None:
        out["recall_molecule_level"] = recall_report(
            g, a, n_prime, ids=ids, query_ids=query_ids, mol_ids=mol_ids,
            level="molecule").as_dict()
    else:
        out["recall_molecule_level"] = {
            "reported": False,
            "why": "mol_ids was not given, so the conformer -> molecule "
                   "reduction could not be computed. On a "
                   "one-conformer-per-molecule corpus the two levels coincide; "
                   "on the multi-conformer libraries (9.48 or 40.41 conformers per "
                   "molecule) they do "
                   "not, and no compound-level statistic can be read off the "
                   "conformer-level report.",
        }
        warnings.warn(
            "evaluate_primary() was called without mol_ids, so only "
            "conformer-level recall could be computed. doc/METRICS.md section 3 "
            "requires both levels; pass mol_ids from Catalog.mol_ids(gids).",
            RuntimeWarning, stacklevel=2)
    return out


def accept(primary, base_primary=None, throughput_ratio=None,
           throughput_ci=None, timing_basis=None, K_configs_searched=None,
           B=10_000, rng=None):
    """The five-criterion accept/reject rule, evaluated and reported in full.

    Every criterion reports its measured value, its threshold and its verdict,
    including the ones that pass -- a rule that only prints failures cannot be
    audited.
    """
    A = ACCEPTANCE
    res = {}

    P = primary["P"]
    res["a_P"] = {"value": P, "threshold": A["a_P_min"], "pass": P >= A["a_P_min"],
                  "ci95": primary["P_ci95"]}

    agg = primary["recall_conformer_level"]["aggregate"]
    k10 = agg.get(10) or agg.get("10")
    if k10 is not None:
        res["a_gamma_k10"] = {
            "value": k10["mean"], "min": k10["min"],
            "threshold": A["a_gamma_at_k10_min"],
            "survivor_fraction": primary["recall_conformer_level"][
                "survivor_fraction"].get(10) or primary[
                "recall_conformer_level"]["survivor_fraction"].get("10"),
            "pass": bool(k10["mean"] >= A["a_gamma_at_k10_min"]),
            "what": ("recall of the cascade that retrieves 10*N' with the "
                     "candidate and rescores with gold -- the operating point "
                     "a real screen uses")}

    if base_primary is not None:
        b = np.array(list(base_primary["P_per_query"].values()), np.float64)
        c = np.array(list(primary["P_per_query"].values()), np.float64)
        pb = paired_bootstrap(b, c, B=B, delta_min=0.01, rng=rng)
        res["a_paired"] = {
            "delta": pb["estimate"], "ci95": [pb["lo"], pb["hi"]],
            "sd_d": pb["sd_d"], "n_queries": pb["n_queries"],
            "threshold_point": A["a_regression_max"],
            "threshold_ci_excludes": A["a_ci_must_exclude"],
            "pass": bool(pb["estimate"] >= A["a_regression_max"]
                         and pb["lo"] > A["a_ci_must_exclude"]),
            "panel_note": ("the paired comparison must run on "
                           + A["a_paired_panel"]),
        }
        if K_configs_searched:
            res["a_paired"]["winner_inflation"] = winner_inflation(
                pb["sd_d"], pb["n_queries"], K_configs_searched)
            res["a_paired"]["winner_inflation_note"] = (
                "subtract this from any TUNE-selected winner's TUNE score; "
                "never quote a TUNE number as a result")

    res["b_min_recall"] = {"value": primary["P_min"],
                           "threshold": A["b_min_query_recall"],
                           "pass": primary["P_min"] >= A["b_min_query_recall"]}

    if base_primary is not None:
        b = np.array(list(base_primary["P_per_query"].values()), np.float64)
        c = np.array(list(primary["P_per_query"].values()), np.float64)
        worst = float((c - b).min())
        res["b_min_paired_regression"] = {
            "value": worst, "threshold": A["b_min_paired_regression"],
            "pass": bool(worst >= A["b_min_paired_regression"]),
            "worst_query": list(primary["P_per_query"])[int(np.argmin(c - b))],
            "what": ("the largest single-query recall loss against the "
                     "baseline. An absolute min cannot tell 'this change broke "
                     "a query' from 'this query was always hard'.")}

    ra = primary["rank_agreement"]["aggregate"]["rbo_ext_p0.98"]
    res["c_rbo"] = {"mean": ra["mean"], "min": ra["min"],
                    "threshold_mean": A["c_rbo098_mean_min"],
                    "threshold_min": A["c_rbo098_min_min"],
                    "pass": bool(ra["mean"] >= A["c_rbo098_mean_min"]
                                 and ra["min"] >= A["c_rbo098_min_min"])}

    se = primary["score_error"]
    md = se["on_gold_top_nprime"]["mean_signed"]
    d_ok = abs(md) <= A["d_mean_delta_max_abs"]
    beta_entry = se.get("size_bias_on_gold_top_nprime")
    beta = beta_entry["beta_tanimoto_per_heavy_atom"] if beta_entry else None
    b_ok = True if beta is None else abs(beta) <= A["d_beta_max_abs_per_heavy_atom"]
    res["d_score_error"] = {
        "mean_delta_signed_on_gold_topN": md,
        "threshold_mean_abs": A["d_mean_delta_max_abs"],
        "beta_per_heavy_atom": beta,
        "beta_ci95": beta_entry["ci95"] if beta_entry else None,
        "threshold_beta_abs": A["d_beta_max_abs_per_heavy_atom"],
        "pass": bool(d_ok and b_ok),
        "sign": ("inflates scores -> precision cost" if md > 0 else
                 "deflates scores -> recall cost" if md < 0 else "no shift"),
        "physical_gate": se["physical_gate_on_gold_top_nprime"],
        "gauge_floor_retrieved_p99": GAUGE_FLOOR_RETRIEVED_P99,
    }

    if throughput_ratio is not None:
        ok = throughput_ratio >= A["e_throughput_ratio_min"]
        if throughput_ci is not None:
            ok = ok and (throughput_ci[0] > 1.0)
        res["e_throughput"] = {
            "ratio": float(throughput_ratio), "ci95": throughput_ci,
            "threshold": A["e_throughput_ratio_min"],
            "timing_basis": timing_basis,
            "pass": bool(ok),
            "rule": THROUGHPUT_RULE["regimes"],
            "attribution_is_out_of_scope": THROUGHPUT_RULE["bottleneck_attribution"],
        }

    decided = [v for v in res.values() if isinstance(v, dict) and "pass" in v]
    res["ACCEPT"] = bool(decided) and all(v["pass"] for v in decided)
    res["criteria_evaluated"] = [k for k in res if k != "ACCEPT"]
    return res

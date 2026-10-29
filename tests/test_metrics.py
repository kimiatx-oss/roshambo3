"""Unit tests for ``roshambo3.metrics``.

Two levels, as the ticket's deciding measurement asks:

* a **toy ranking** small enough that ``R(N,k)``, ``RBO_ext`` and the tie-break
  can be verified by inspection -- written out longhand in the test so a reader
  can check the expected number without running anything;
* **Tier 0 analytic cases**, so the library's ranking and score-error path
  is exercised against numbers that come from the physics rather than from
  another part of this codebase.
"""

import numpy as np
import pytest

import roshambo3.metrics as M


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------

def test_tie_break_is_descending_score_then_ascending_id():
    # by inspection: 0.5 (id 1), 0.5 (id 3), 0.5 (id 0) -> ids 0,1,3; then 0.2
    s = np.array([0.5, 0.5, 0.2, 0.5], np.float32)
    assert M.rank_order(s).tolist() == [0, 1, 3, 2]
    assert M.rank_g(s).tolist() == [1, 2, 4, 3]


def test_tie_break_uses_supplied_ids_not_row_order():
    s = np.array([0.5, 0.5, 0.5], np.float32)
    ids = np.array([30, 10, 20])
    assert M.rank_order(s, ids=ids).tolist() == [1, 2, 0]


def test_tie_break_is_identical_in_gold_and_candidate():
    """The bug this rule exists to prevent: identical scores, different order.

    Both rankings get the same rule, so an all-ties gold and an all-ties
    candidate agree perfectly -- recall 1.0, RBO 1.0 -- instead of disagreeing
    by whatever an unstable sort happened to do.
    """
    rng = np.random.default_rng(0)
    s = np.full(1000, 0.5, np.float32)
    perm = rng.permutation(1000)
    assert M.recall_at(s, s[perm][np.argsort(perm)], 10, 10) == 1.0
    a = M.rank_order(s)
    b = M.rank_order(s.copy())
    assert np.array_equal(a, b)


def test_rank_g_is_a_permutation_even_with_massive_float32_ties():
    s = np.zeros(10_000, np.float32)
    r = M.rank_g(s)
    assert sorted(r.tolist()) == list(range(1, 10_001))


def test_competition_rank_matches_the_ticket_definition():
    s = np.array([0.9, 0.5, 0.5, 0.1])
    # 1 + |{y : g(y) > g(x)}|
    assert M.competition_rank(s).tolist() == [1, 2, 2, 4]


def test_top_k_order_is_exact_against_a_full_sort():
    rng = np.random.default_rng(7)
    for n, k in ((1000, 7), (10_000, 500), (5_000, 4_999)):
        s = rng.random(n).astype(np.float32)
        s[rng.integers(0, n, n // 5)] = 0.25          # force ties
        ids = rng.permutation(n)
        assert np.array_equal(M.top_k_order(s, k, ids),
                              M.rank_order(s, ids)[:k])


# ---------------------------------------------------------------------------
# Recall, by inspection
# ---------------------------------------------------------------------------

def test_recall_on_a_hand_checkable_toy():
    # gold order:      a b c d e f g h   (scores 8..1)
    # candidate order: c a b e d h f g
    gold = np.array([8, 7, 6, 5, 4, 3, 2, 1], np.float64)
    cand = np.array([7, 6, 8, 4, 5, 2, 1, 3], np.float64)
    # G_3 = {a,b,c} = {0,1,2}; A_3 = {c,a,b} = {2,0,1}  -> 3/3
    assert M.recall_at(gold, cand, 3, 3) == 1.0
    # G_2 = {a,b} = {0,1}; A_2 = {c,a} = {2,0} -> 1/2
    assert M.recall_at(gold, cand, 2, 2) == 0.5
    # G_2 with k=4: A_4 = {c,a,b,e} -> both found -> 1.0
    assert M.recall_at(gold, cand, 2, 4) == 1.0
    # G_8 with k=8 is trivially 1
    assert M.recall_at(gold, cand, 8, 8) == 1.0


def test_epsilon_gamma_curve_is_monotone_in_k():
    rng = np.random.default_rng(3)
    gold = rng.random(2000)
    cand = gold + rng.normal(0, 0.02, 2000)
    c = M.recall_curve(gold, cand, 50)
    rec = [c[k]["recall"] for k in (1, 2, 5, 10)]
    assert rec == sorted(rec)
    assert c[10]["survivor_fraction"] == pytest.approx(500 / 2000)


def test_survivor_recall_reports_both_axes():
    gold = np.arange(100, 0, -1).astype(np.float64)   # best = index 0
    surv = np.zeros(100, bool)
    surv[:5] = True                                    # keeps gold ranks 1..5
    r, f = M.survivor_recall(gold, surv, 10)
    assert r == 0.5 and f == 0.05


def test_recall_report_publishes_the_per_query_table_and_min():
    rng = np.random.default_rng(11)
    gold = rng.random((6, 500))
    cand = gold.copy()
    cand[3] = rng.random(500)          # one query is destroyed
    rep = M.recall_report(gold, cand, 20)
    d = rep.as_dict()
    assert len(d["per_query"]) == 6
    assert d["aggregate"][1]["min"] < 0.2 < d["aggregate"][1]["mean"]
    assert "p10" in d["aggregate"][1]


# ---------------------------------------------------------------------------
# Conformer vs molecule level
# ---------------------------------------------------------------------------

def test_molecule_level_dedupes_across_batches_and_keeps_conf_counts():
    # molecule 7 has three conformers, molecule 9 has one
    mol = np.array([7, 9, 7, 7, 9])
    sc = np.array([0.1, 0.9, 0.5, 0.2, 0.3])
    uniq, best, ncnf, arg = M.reduce_over_conformers(sc, mol)
    assert uniq.tolist() == [7, 9]
    assert best.tolist() == [0.5, 0.9]
    assert ncnf.tolist() == [3, 2]
    assert arg.tolist() == [2, 1]


def test_repeated_molecule_id_as_tie_break_is_refused():
    """The bug this check was written from: on a multi-conformer tier the
    molecule_id repeats, RBO's overlap can exceed its depth, and RBO reads
    1.019 -- an impossible value that is the only visible symptom."""
    mol = np.array([7, 7, 9, 9])
    with pytest.raises(M.TieBreakError):
        M.rbo_ext([7, 7, 9], [7, 9, 9])
    rng = np.random.default_rng(31)
    g = rng.random((2, 4))
    with pytest.raises(M.TieBreakError):
        M.recall_report(g, g, 2, ids=mol)
    with pytest.raises(M.TieBreakError):
        M.rank_agreement_report(g, g, 2, ids=mol)


def test_canonical_ids_are_unique_and_order_by_molecule_then_conformer():
    mol = np.array([90, 10, 90, 10])
    conf = np.array([1, 0, 0, 3])
    key = M.canonical_ids(mol, conf)
    assert sorted(key.tolist()) == [0, 1, 2, 3]
    # ascending key must mean ascending (mol_id, conf_idx)
    order = np.argsort(key)
    assert mol[order].tolist() == [10, 10, 90, 90]
    assert conf[order].tolist() == [0, 3, 0, 1]
    M.assert_unique_ids(key)


def test_rbo_cannot_exceed_one_on_any_valid_input():
    rng = np.random.default_rng(41)
    for _ in range(50):
        n = int(rng.integers(2, 40))
        a = rng.permutation(n).tolist()
        b = rng.permutation(n).tolist()
        for p in (0.5, 0.98, 0.998):
            v = M.rbo_ext(a, b, p)
            assert 0.0 <= v <= 1.0 + 1e-12


def test_molecule_level_recall_differs_from_conformer_level():
    mol = np.repeat(np.arange(50), 4)                # 50 molecules x 4 confs
    rng = np.random.default_rng(5)
    gold = rng.random((3, 200))
    cand = gold + rng.normal(0, 0.05, (3, 200))
    conf = M.recall_report(gold, cand, 10)
    molr = M.recall_report(gold, cand, 10, mol_ids=mol, level="molecule")
    assert molr.m_targets == 50 and conf.m_targets == 200
    assert set(molr.as_dict()["by_conf_count"])      # stratification present


# ---------------------------------------------------------------------------
# Rank agreement
# ---------------------------------------------------------------------------

def test_rbo_identical_lists_is_one():
    a = list(range(20))
    assert M.rbo_ext(a, a, 0.98) == pytest.approx(1.0, abs=1e-12)
    assert M.rbo_ext(a, a, 0.9) == pytest.approx(1.0, abs=1e-12)


def test_rbo_disjoint_lists_is_zero():
    assert M.rbo_ext(list(range(10)), list(range(100, 110)), 0.98) == 0.0


def test_rbo_is_top_weighted():
    base = list(range(200))
    swap_top = base.copy(); swap_top[0], swap_top[1] = swap_top[1], swap_top[0]
    swap_deep = base.copy(); swap_deep[98], swap_deep[99] = swap_deep[99], swap_deep[98]
    assert M.rbo_ext(base, swap_top, 0.98) < M.rbo_ext(base, swap_deep, 0.98)
    # and the deep swap is nearly invisible at p = 0.98 (depth ~50)
    assert M.rbo_ext(base, swap_deep, 0.98) > 0.999


def test_rbo_hand_checkable_two_element_case():
    """s = l = 2, lists [a,b] and [b,a]. X_1 = 0, X_2 = 2.

    RBO_EXT = (1-p)/p * (0/1 p + 2/2 p^2) + [(X_2-X_2)/2 + X_2/2] p^2
            = (1-p) p + p^2.
    """
    for p in (0.5, 0.9, 0.98):
        expect = (1 - p) * p + p ** 2
        assert M.rbo_ext(["a", "b"], ["b", "a"], p) == pytest.approx(expect)


def test_rbo_deeper_persistence_sees_deeper_disagreement():
    base = list(range(1000))
    perturbed = base[:300] + list(reversed(base[300:]))
    assert M.rbo_ext(base, perturbed, 0.998) < M.rbo_ext(base, perturbed, 0.98)


def test_spearman_on_intersection_is_refused():
    with pytest.raises(M.DisqualifiedMetric) as e:
        M.spearman_on_intersection([1, 2, 3], [1, 2, 3])
    assert "disqualified" in str(e.value).lower()


def test_rank_agreement_report_always_emits_the_disqualification_and_landing_table():
    rng = np.random.default_rng(2)
    gold = rng.random((2, 300))
    cand = gold + rng.normal(0, 0.01, (2, 300))
    r = M.rank_agreement_report(gold, cand, 50)
    assert "spearman_on_intersection" in r["disqualified"]
    assert "kendall_tau_b_on_gold" in r["secondary_only"]
    assert len(r["gold_top_landing"]["q0"]) == 10
    assert r["gold_top_landing"]["q0"][0]["gold_rank"] == 1


def test_landing_table_cand_rank_matches_a_full_ranking():
    rng = np.random.default_rng(19)
    gold = rng.random(5000)
    cand = gold + rng.normal(0, 0.05, 5000)
    tab = M.gold_top_landing_table(gold, cand, 10)
    full = M.rank_g(cand)
    for row in tab:
        assert row["cand_rank"] == int(full[row["target_id"]])


# ---------------------------------------------------------------------------
# Score error: signs, strata, size bias
# ---------------------------------------------------------------------------

def test_signed_summary_never_collapses_the_sign():
    d = np.array([-0.02, -0.02, 0.01])
    s = M.signed_summary(d)
    assert s["mean_signed"] < 0 < s["mean_abs"]
    assert s["frac_negative"] == pytest.approx(2 / 3)


def test_size_bias_recovers_a_planted_slope_with_a_query_fixed_effect():
    rng = np.random.default_rng(4)
    nq, ni, beta = 40, 200, 2.0e-4
    hac = rng.integers(10, 60, (nq, ni)).astype(float)
    # a large per-query intercept that a fit without fixed effects would
    # partially absorb into the slope, because query mean HAC varies
    intercept = rng.normal(0, 0.05, nq)[:, None]
    delta = intercept + beta * hac + rng.normal(0, 1e-3, (nq, ni))
    q = np.repeat(np.arange(nq), ni)
    out = M.size_bias(delta.ravel(), hac.ravel(), q, B=500)
    assert out["beta_tanimoto_per_heavy_atom"] == pytest.approx(beta, rel=0.05)
    assert out["ci95"][0] < beta < out["ci95"][1]
    assert out["bootstrap_unit"].startswith("query")
    assert out["hac_decile_table"]


def test_size_bias_bootstrap_is_over_queries_not_items():
    """A query-level bootstrap must give a *wider* interval than an item-level
    one when the error is correlated within a query -- which is the whole
    reason the ticket mandates it."""
    rng = np.random.default_rng(8)
    nq, ni = 20, 500
    hac = rng.integers(10, 60, (nq, ni)).astype(float)
    per_query_slope = rng.normal(2e-4, 3e-4, nq)[:, None]
    delta = per_query_slope * hac + rng.normal(0, 1e-4, (nq, ni))
    q = np.repeat(np.arange(nq), ni)
    cluster = M.size_bias(delta.ravel(), hac.ravel(), q, B=500)
    width_cluster = cluster["ci95"][1] - cluster["ci95"][0]
    # a naive item bootstrap: pretend every item is its own query
    naive = M.size_bias(delta.ravel(), hac.ravel(),
                        np.repeat(np.arange(nq * ni // 50), 50), B=500)
    assert width_cluster > 3 * (naive["ci95"][1] - naive["ci95"][0])


def test_score_error_report_labels_the_selection_bias_and_refuses_low_strata():
    rng = np.random.default_rng(6)
    gold = rng.random((4, 400)) * 0.9          # spans well below T = 0.55
    cand = gold + rng.normal(0, 0.01, (4, 400))
    rep = M.score_error_report(gold, cand, 20, hac=rng.integers(10, 60, 400),
                               B=300)
    assert "SELECTION-BIASED" in rep["on_cand_top_nprime_label"]
    assert set(rep["by_stratum_global"]) == {n for n, _, _ in M.SCORE_STRATA}
    # the top-20 of a uniform[0,0.9] draw is above 0.55, so this one passes
    assert rep["gold_scope"]["per_pair_ok"]


def test_per_pair_claim_below_055_is_refused():
    with pytest.raises(M.GoldScopeError) as e:
        M.assert_gold_supports_per_pair_claim(np.array([0.9, 0.6, 0.3]))
    assert "0.55" in str(e.value)


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------

def test_epsilon_floor_is_stratum_dependent_by_three_orders():
    lo = M.epsilon_floor(np.array([0.30]))[0]
    hi = M.epsilon_floor(np.array([0.85]))[0]
    assert lo / hi > 1000


def test_gauge_floor_has_no_scalar_form():
    with pytest.raises(KeyError):
        M.gauge_floor(1)            # the stratum must be named, not guessed
    assert M.gauge_floor("retrieved_list", "p99") == pytest.approx(6.56e-7)
    assert M.GAUGE_FLOOR_RETRIEVED_P99 == pytest.approx(4.47e-6)
    # the misquoted 5.8e-4 is nowhere near a p99
    assert M.GAUGE_FLOOR_RETRIEVED_P99 < 5.8e-4 / 100


def test_improvement_below_the_floor_is_not_an_improvement():
    t = np.full(1000, 0.8)                       # top stratum, floor ~3e-7
    tiny = np.full(1000, 1e-9)
    v = M.improvement_verdict(tiny, t)
    assert v["verdict"] == "BELOW_FLOOR_NOT_AN_IMPROVEMENT"
    big = np.full(1000, 1e-2)
    assert M.improvement_verdict(big, t)["verdict"] == "ABOVE_FLOOR_TESTABLE"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_bca_ci_covers_a_known_mean():
    rng = np.random.default_rng(1)
    v = rng.normal(0.5, 0.1, 200)
    ci = M.bca_ci(v, np.mean, B=2000, rng=rng)
    assert ci["lo"] < 0.5 < ci["hi"]
    assert ci["method"].startswith("BCa")


def test_paired_bootstrap_requires_both_significance_and_effect_size():
    rng = np.random.default_rng(9)
    base = rng.normal(0.9, 0.01, 400)
    tiny = base + 0.002                   # significant but below delta = 0.01
    out = M.paired_bootstrap(base, tiny, B=1000, rng=rng)
    assert out["ci_excludes_zero"] and not out["exceeds_delta_min"]
    assert out["real"] is False


def test_winner_inflation_reproduces_the_measured_figures():
    assert M.winner_inflation(0.02, 32, 10) == pytest.approx(0.0075, abs=2e-4)
    assert M.winner_inflation(0.02, 32, 30) == pytest.approx(0.0092, abs=2e-4)
    assert M.winner_inflation(0.02, 32, 200) == pytest.approx(0.0114, abs=2e-4)


def test_panel_sizing_reproduces_the_published_panel():
    assert M.required_n_for_delta(0.1099, 0.01) == 464
    assert M.resolvable_delta(0.1099, 32) == pytest.approx(0.038, abs=5e-4)


def test_holm_bonferroni_caps_eval_finalists_at_three():
    out = M.holm_bonferroni([0.001, 0.02, 0.9])
    assert out["results"][0]["reject"] and not out["results"][2]["reject"]
    with pytest.raises(ValueError):
        M.holm_bonferroni([0.01] * 4)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_simulated_recall_falls_as_the_error_grows():
    rng = np.random.default_rng(12)
    gold = np.sort(rng.random((4, 20_000)))[:, ::-1]
    rs = [M.simulate_recall(gold, M.GaussianDelta(sd), 100, n_rep=3)["mean"][0]
          for sd in (1e-6, 1e-3, 1e-2, 1e-1)]
    assert rs == sorted(rs, reverse=True)
    assert rs[0] > 0.99 and rs[-1] < 0.5


def test_empirical_delta_preserves_its_conditioning():
    rng = np.random.default_rng(13)
    gold = rng.random(20_000)
    hac = rng.integers(10, 60, 20_000).astype(float)
    # error that only exists for large molecules
    cand = gold + np.where(hac > 40, rng.normal(0, 0.05, 20_000), 0.0)
    mdl = M.EmpiricalDelta.fit(gold, cand, hac)
    d = mdl.sample(gold, hac, rng=rng)
    assert np.std(d[hac > 40]) > 10 * np.std(d[hac < 30])


def test_gaussian_model_is_labelled_optimistic():
    assert "optimistic" in M.GaussianDelta(0.01).describe()["caveat"]


def test_epsilon_gamma_curve_shape():
    rng = np.random.default_rng(14)
    gold = rng.random((6, 5000))
    out = M.epsilon_gamma_curve(gold, M.GaussianDelta(1.0), 50,
                                scales=[1e-6, 1e-2], n_rep=3, B=200)
    assert len(out["curve"]) == 2
    r_small = out["curve"][0]["recall"][1]["mean"]
    r_big = out["curve"][1]["recall"][1]["mean"]
    assert r_small > r_big
    for row in out["curve"]:
        assert row["recall"][10]["mean"] >= row["recall"][1]["mean"] - 1e-12
        assert "ci95" in row["recall"][1]


# ---------------------------------------------------------------------------
# Pre-registration and the HOLDOUT ledger
# ---------------------------------------------------------------------------

def test_holdout_ledger_refuses_a_second_touch_of_the_same_config(tmp_path):
    led = M.HoldoutLedger(str(tmp_path / "ledger.jsonl"))
    led.touch(preregistration_id="PR-1", config_hash="abc", run_id="r1")
    with pytest.raises(M.HoldoutViolation):
        led.touch(preregistration_id="PR-1", config_hash="abc", run_id="r2")
    # a new pre-registration id is a declared new experiment and is allowed
    led.touch(preregistration_id="PR-2", config_hash="abc", run_id="r3")
    assert led.n_touches() == 2


def test_prereg_refuses_a_metric_change_as_an_amendment(tmp_path):
    p = str(tmp_path / "pr.json")
    kw = dict(preregistration_id="PR-1", hypothesis="h", knobs={},
              thresholds={}, panel_sha256={}, dataset_manifests={},
              planned_n_runs=1, planned_analysis="a")
    M.PreRegistration(primary_metric_verbatim="P = X", **kw).write(p)
    with pytest.raises(M.HoldoutViolation):
        M.PreRegistration(primary_metric_verbatim="P = Y", **kw).write(p)


def test_the_primary_metric_sentence_is_frozen_and_names_gold_400():
    s = M.PRIMARY_METRIC_VERBATIM
    assert "R(N',N')" in s and "N' = 1,000" in s
    assert "32 HOLDOUT queries" in s and "EVAL-10M" in s
    assert "steps=400" in s          # the gold reference runs 400 steps, not 100
    assert "ascending molecule_id" in s


def test_throughput_rule_names_its_regime_and_defers_attribution_to_D009():
    r = M.THROUGHPUT_RULE
    assert set(r["regimes"]) == {"kernel_only_locked_uncontended",
                                 "end_to_end_or_contended"}
    sr = r["regimes"]["kernel_only_locked_uncontended"]["smallest_resolvable_ratio"]
    assert sr["same_process_11_repeats"] == 1.002
    assert sr["across_process_11_repeats"] == 1.005
    # a 1.7% clock dip moved timing by 1.6%, so an unverified clock costs a
    # bigger margin than any dispersion figure
    assert sr["across_invocations_without_clock_normalisation"] == 1.01
    assert r["regimes"]["end_to_end_or_contended"][
        "smallest_resolvable_ratio"] == 1.25
    # A WHY claim is out of scope here, and stall shares are forbidden
    a = r["bottleneck_attribution"]
    assert any("not_selected" in s for s in a["forbidden_evidence"])
    assert any("smsp__issue_active" in s for s in a["required_evidence"])
    assert any("causal ablation" in s for s in a["required_evidence"])


def test_acceptance_thresholds_sit_between_the_floor_and_the_measured_baseline():
    """Two conditions, both necessary, and the ticket only names the first.

    A threshold must be well ABOVE the noise floor (or it measures noise) and
    at or above the MEASURED baseline (or it rejects the shipped code for
    reasons unrelated to any candidate). The shipped configuration's own mean
    Delta on the retrieved list is 1.78e-3 -- 400x the corrected gauge p99 --
    so tightening toward 4.5e-6 would have failed the second condition.
    """
    d = M.ACCEPTANCE["d_mean_delta_max_abs"]
    assert d > M.GAUGE_FLOOR_RETRIEVED_P99 * 100      # measurable
    assert 1.78e-3 <= d <= 5e-3                       # admits the baseline
    b = M.ACCEPTANCE["d_beta_max_abs_per_heavy_atom"]
    assert 3.6e-4 <= b <= 1e-3
    # and the absolute recall floor must sit below both panels' CI lower bounds
    assert M.ACCEPTANCE["a_P_min"] < 0.9256


# ---------------------------------------------------------------------------
# Tier 0: the library on numbers that come from the physics
# ---------------------------------------------------------------------------

def test_tier0_single_carbon_self_volume_is_the_corrected_constant():
    """The corrected constant: 20.5795263 A^3, not 20.5836.

    The library does not compute this -- it is here because every threshold in
    :mod:`roshambo3.metrics.floors` is quoted in Tanimoto units derived from this
    frame, and a 2e-4 relative error in the constant would silently move them.
    """
    from roshambo3 import numerics as N
    import math
    exact = 8.0 * (math.pi / (2.0 * float(N.A))) ** 1.5
    assert exact == pytest.approx(20.5795263, rel=1e-6)
    # the device header carries the float32 rounding of it
    assert float(N.SHAPE_PREFACTOR) == pytest.approx(exact, rel=1e-6)
    # and the previously-quoted 20.5836 is 2e-4 relative off -- 200x the
    # tolerance the sentence quoting it demanded
    assert abs(20.5836 - exact) / exact == pytest.approx(2.0e-4, rel=0.2)


def test_tier0_analytic_two_atom_overlap_ranks_as_the_physics_says():
    """Two carbons at separation d: overlap falls monotonically in d, so the
    Tanimoto ranking of a sweep over d must be exactly the ascending-d order.

    Computed in float64 from the reduced frame's own constants, so this is a
    check of the library's ranking against an analytic curve, not against
    another part of this codebase.
    """
    from roshambo3 import numerics as N
    s2 = float(N.COORD_SCALE) ** 2
    d = np.linspace(0.0, 6.0, 25)
    saa = 1.0 + np.exp2(-s2 * 0.0)                      # per-atom self overlap
    sab = 2.0 * (1.0 + np.exp2(-s2 * d ** 2))
    T = sab / (2 * saa + 2 * saa - sab)
    assert M.rank_order(T).tolist() == list(range(25))
    assert M.recall_at(T, T, 5, 5) == 1.0
    # the reduced frame is the physics, not a re-parameterisation of it: the
    # kernel's pair term exp2(-|s dx|^2) IS the Gaussian overlap exp(-alpha dx^2)
    import math
    assert s2 * math.log(2.0) == pytest.approx(float(N.ALPHA), rel=1e-6)
    assert np.exp2(-s2 * d ** 2) == pytest.approx(
        np.exp(-float(N.ALPHA) * d ** 2), rel=1e-6)


def test_tier0_gradient_free_invariance_gives_recall_one():
    """A rigid motion applied to both molecules leaves every score unchanged,
    so recall against the rotated run must be exactly 1.0 at every k -- the
    degenerate case the whole library must not get wrong."""
    rng = np.random.default_rng(21)
    s = rng.random(5000).astype(np.float32)
    rep = M.recall_report(s[None, :], s[None, :], 100)
    assert all(v["mean"] == 1.0 for v in rep.aggregate.values())

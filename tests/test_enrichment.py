"""Unit tests for ``roshambo3.metrics.enrichment``.

Three tiers:

* **hand-derived closed-form cases** -- all actives first, all actives last,
  actives uniformly interleaved, and a tie spanning the cut boundary -- with
  the expected number written out in the test so a reader can check it
  without running anything;
* **refusals** -- conformer-level input, no actives, a zero-size cut -- each
  asserted to raise the specific exception the module promises, not just
  "an error";
* **cross-checks against RDKit's independent implementation**
  (``rdkit.ML.Scoring.Scoring``) of BEDROC, AUC and EF, since both are
  computing the same published formulas and agreement is the strongest
  available evidence beyond hand-derivation for the less trivial cases
  (BEDROC's normalization, AUC's midrank tie correction).
"""

import numpy as np
import pytest

import roshambo3.metrics as M
from roshambo3.metrics.enrichment import (
    DEGENERACY_MIN_DISTINCT_VALUES,
    boundary_tie_info,
    cut_size,
)

rdkit_scoring = pytest.importorskip(
    "rdkit.ML.Scoring.Scoring",
    reason="rdkit is a project dependency; if it is ever missing this whole "
           "cross-check tier skips cleanly rather than failing the suite.")


# ---------------------------------------------------------------------------
# The cut and its rounding
# ---------------------------------------------------------------------------

def test_cut_size_ceil_is_the_default_and_never_degenerates_to_zero():
    # floor(0.01 * 50) = 0 -- exactly the degenerate cut the docstring warns
    # about; ceil is never zero for a positive fraction and N >= 1.
    assert cut_size(50, 0.01) == 1          # default is ceil
    assert cut_size(50, 0.01, rounding="ceil") == 1
    assert cut_size(50, 0.01, rounding="floor") == 0
    assert cut_size(20, 0.05, rounding="ceil") == 1
    assert cut_size(20, 0.05, rounding="floor") == 1   # exact fraction: agree


def test_cut_size_rejects_bad_fraction_and_rounding():
    with pytest.raises(ValueError):
        cut_size(100, 0.0)
    with pytest.raises(ValueError):
        cut_size(100, 1.5)
    with pytest.raises(ValueError):
        cut_size(100, 0.1, rounding="round")


# ---------------------------------------------------------------------------
# Hand-derived closed-form cases (deciding measurement 1)
# ---------------------------------------------------------------------------

def test_ef_all_actives_first_hits_the_analytic_maximum():
    # N=100, 10 actives, all in front. Top 10% cut (=10 rows) is all-active.
    # EF = (10/10) / (10/100) = 10.0 exactly -- the largest EF this N/n_actives
    # combination can produce, by inspection.
    n, n_act = 100, 10
    scores = np.arange(n, 0, -1, dtype=np.float64)
    labels = np.zeros(n, bool)
    labels[:n_act] = True
    mol_ids = np.arange(n)
    r = M.enrichment_factor(scores, labels, 0.10, mol_ids)
    assert r.cut == 10
    assert r.n_actives_in_cut == 10
    assert r.ef == pytest.approx(10.0)
    assert not r.degenerate


def test_ef_all_actives_last_hits_the_analytic_minimum():
    n, n_act = 100, 10
    scores = np.arange(n, 0, -1, dtype=np.float64)
    labels = np.zeros(n, bool)
    labels[-n_act:] = True
    r = M.enrichment_factor(scores, labels, 0.10, np.arange(n))
    assert r.n_actives_in_cut == 0
    assert r.ef == 0.0


def test_ef_actives_uniformly_interleaved_is_exactly_one():
    # An active every 10th rank, cut at 10%: by construction the cut contains
    # exactly 10% of the actives -- EF = 1.0 exactly, the "random" value.
    n = 100
    scores = np.arange(n, 0, -1, dtype=np.float64)
    labels = np.zeros(n, bool)
    labels[9::10] = True                 # ranks 10, 20, ..., 100
    r = M.enrichment_factor(scores, labels, 0.10, np.arange(n))
    assert r.n_actives == 10
    assert r.n_actives_in_cut == 1
    assert r.ef == pytest.approx(1.0)


def test_ef_boundary_tie_is_decided_by_the_canonical_tie_break():
    """The case the ticket says will actually be got wrong.

    N=20, cut=2 (ceil(0.1*20)). Row 0 is the unique best score; rows 1-5 are
    a 5-way tie for 2nd place, of which row 4 is the only active. Whichever
    of the tied rows lands in the cut is decided entirely by the canonical
    ascending-id tie-break, not by the score -- by inspection, ascending id
    picks row 1 first, so the (inactive) row 1 fills the second cut slot and
    the active row 4 is excluded. n_tied_at_boundary must report all 5 tied
    rows, and n_active_tied_at_boundary the 1 active among them, so a reader
    can see the metric is resting on the tie-break.
    """
    scores = np.array([10.0, 5.0, 5.0, 5.0, 5.0, 5.0] + [1.0] * 14)
    labels = np.zeros(20, bool)
    labels[4] = True                     # the only active is mid-tie
    ids = np.arange(20)                  # ascending id = ascending row index
    r = M.enrichment_factor(scores, labels, 0.10, np.arange(20), ids=ids)
    assert r.cut == 2
    assert r.n_tied_at_boundary == 5
    assert r.n_active_tied_at_boundary == 1
    assert r.n_actives_in_cut == 0       # the active loses the tie-break
    assert r.ef == 0.0

    # Flip the tie-break by relabelling ids so the active-bearing row sorts
    # first: same scores, same labels, opposite outcome -- proof EF is
    # reading the tie-break, not the score.
    ids_flipped = ids.copy()
    ids_flipped[1], ids_flipped[4] = ids_flipped[4], ids_flipped[1]
    r2 = M.enrichment_factor(scores, labels, 0.10, np.arange(20), ids=ids_flipped)
    assert r2.n_actives_in_cut == 1
    assert r2.ef > 0.0


def test_boundary_tie_info_is_one_when_the_boundary_is_not_tied():
    # n_tied_at_boundary counts every row sharing the boundary score,
    # including the boundary row itself -- 1 means "just itself", i.e. no
    # tie. Only cut<=0 or cut>=n (no boundary at all) reports 0.
    scores = np.array([10.0, 9.0, 5.0, 5.0, 5.0] + [1.0] * 15)
    n_tied, n_active_tied = boundary_tie_info(scores, cut=2, labels=np.zeros(20, bool))
    assert (n_tied, n_active_tied) == (1, 0)


def test_boundary_tie_info_is_zero_at_the_edges():
    scores = np.arange(10, dtype=np.float64)
    assert boundary_tie_info(scores, cut=0) == (0, 0)
    assert boundary_tie_info(scores, cut=10) == (0, 0)     # cut == n


# ---------------------------------------------------------------------------
# Degeneracy: the ticket's own worked example, and the true refusals
# ---------------------------------------------------------------------------

def test_five_actives_can_only_take_six_values_and_is_flagged():
    rng = np.random.default_rng(11)
    n = 2000
    scores = rng.random(n)
    labels = np.zeros(n, bool)
    labels[:5] = True
    r = M.enrichment_factor(scores, labels, 0.05, np.arange(n))   # cut >> 5
    assert r.n_possible_distinct_values == 6      # 0..5 actives possible
    assert r.degenerate is True


def test_generous_sample_is_not_flagged_degenerate():
    rng = np.random.default_rng(12)
    n = 5000
    scores = rng.random(n)
    labels = rng.random(n) < 0.2                  # ~1000 actives
    r = M.enrichment_factor(scores, labels, 0.05, np.arange(n))
    assert r.n_possible_distinct_values > DEGENERACY_MIN_DISTINCT_VALUES
    assert r.degenerate is False


def test_no_actives_is_refused_not_a_bare_nan():
    scores = np.random.default_rng(1).random(50)
    with pytest.raises(M.DegenerateEnrichmentError):
        M.enrichment_factor(scores, np.zeros(50, bool), 0.1, np.arange(50))
    with pytest.raises(M.DegenerateEnrichmentError):
        M.roc_auc(scores, np.zeros(50, bool), np.arange(50))
    with pytest.raises(M.DegenerateEnrichmentError):
        M.bedroc(scores, np.zeros(50, bool), np.arange(50))


def test_no_inactives_is_refused_for_auc_but_defined_at_one_for_bedroc():
    # AUC needs both classes to compare against; BEDROC's normalization has a
    # well-defined limit (RIE_max == RIE_min) at "everyone is active", which
    # RDKit's reference implementation also returns as 1.0 rather than 0/0.
    scores = np.random.default_rng(2).random(30)
    labels = np.ones(30, bool)
    with pytest.raises(M.DegenerateEnrichmentError):
        M.roc_auc(scores, labels, np.arange(30))
    b = M.bedroc(scores, labels, np.arange(30))
    assert b["bedroc"] == pytest.approx(1.0)
    assert b["degenerate"] is True                # only 1 possible value


def test_zero_size_cut_is_refused():
    scores = np.random.default_rng(3).random(20)
    labels = np.zeros(20, bool)
    labels[0] = True
    with pytest.raises(M.DegenerateEnrichmentError):
        M.enrichment_factor(scores, labels, 0.01, np.arange(20), rounding="floor")
    # the default (ceil) rounding does not hit this on the same input
    M.enrichment_factor(scores, labels, 0.01, np.arange(20))


# ---------------------------------------------------------------------------
# Conformer-level input is refused, not silently answered
# ---------------------------------------------------------------------------

def test_conformer_level_mol_ids_is_refused_everywhere():
    n = 20
    scores = np.random.default_rng(4).random(n)
    labels = np.zeros(n, bool)
    labels[:2] = True
    mol_ids_with_dupes = np.array([0, 0, 1, 2, 3] * 4)
    for fn, args in (
        (M.enrichment_factor, (scores, labels, 0.1, mol_ids_with_dupes)),
        (M.enrichment_factors, (scores, labels, (0.1, 0.2), mol_ids_with_dupes)),
        (M.roc_auc, (scores, labels, mol_ids_with_dupes)),
        (M.bedroc, (scores, labels, mol_ids_with_dupes)),
        (M.enrichment_report, (scores, labels, mol_ids_with_dupes)),
    ):
        with pytest.raises(M.ConformerLevelInputError):
            fn(*args)


def test_assert_compound_level_message_points_at_the_reduction():
    with pytest.raises(M.ConformerLevelInputError, match="reduce_over_conformers"):
        M.assert_compound_level(np.array([1, 1, 2]))


# ---------------------------------------------------------------------------
# Max-pooled EF: a different estimator, built from the existing reduction
# ---------------------------------------------------------------------------

def test_max_pool_scores_matches_reduce_over_conformers_reused_on_the_query_axis():
    scores = np.array([
        [1.0, 5.0, 2.0, 9.0, 0.5],
        [3.0, 1.0, 8.0, 2.0, 0.5],
        [0.0, 0.0, 1.0, 1.0, 9.0],
    ])
    mol_ids = np.array([10, 20, 30, 40, 50])
    uniq, best, n_obs = M.max_pool_scores(scores, mol_ids)
    assert uniq.tolist() == [10, 20, 30, 40, 50]
    assert best.tolist() == [3.0, 5.0, 8.0, 9.0, 9.0]
    assert n_obs.tolist() == [3, 3, 3, 3, 3]


def test_max_pooled_ef_differs_from_mean_of_per_query_ef():
    """Max-pooling inflates every score, so it is not interchangeable with
    the mean of per-query EFs -- demonstrated, not just asserted."""
    rng = np.random.default_rng(21)
    n_query, n_lib = 4, 300
    mol_ids = np.arange(n_lib)
    labels = rng.random(n_lib) < 0.1
    scores = rng.random((n_query, n_lib))

    per_query_ef = [
        M.enrichment_factor(scores[q], labels, 0.1, mol_ids).ef
        for q in range(n_query)
    ]
    _, pooled, _ = M.max_pool_scores(scores, mol_ids)
    pooled_ef = M.enrichment_factor(pooled, labels, 0.1, mol_ids).ef

    assert pooled_ef != pytest.approx(np.mean(per_query_ef))


# ---------------------------------------------------------------------------
# ROC-AUC: midrank tie correction, verified against a brute-force count and
# against RDKit's trapezoidal ROC-curve implementation.
# ---------------------------------------------------------------------------

def test_auc_matches_the_brute_force_mann_whitney_definition_with_ties():
    # AUC = P(random active > random inactive) + 0.5 P(tie), by definition.
    # Build in an explicit tie between an active and an inactive to make sure
    # midranks -- not the canonical id tie-break -- decide it.
    scores = np.array([5.0, 4.0, 4.0, 3.0, 1.0])
    labels = np.array([True, True, False, False, False])
    auc = M.roc_auc(scores, labels, np.arange(5))["auc"]
    pos, neg = scores[labels], scores[~labels]
    brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert auc == pytest.approx(brute)


def test_auc_perfect_and_inverted_separation():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert M.roc_auc(scores, np.array([1, 1, 0, 0, 0], bool), np.arange(5))["auc"] == 1.0
    assert M.roc_auc(scores, np.array([0, 0, 0, 1, 1], bool), np.arange(5))["auc"] == 0.0


@pytest.mark.parametrize("seed", range(8))
def test_auc_matches_rdkit_on_tie_free_random_data(seed):
    # RDKit's CalcAUC needs a pre-sorted, tie-free input to be unambiguous;
    # continuous random floats are tie-free with probability 1, so this
    # isolates "does the formula agree" from "does the tie convention agree"
    # (the latter is covered by the hand-built tie case above, where RDKit's
    # order-dependent trapezoidal integration is not a valid oracle).
    rng = np.random.default_rng(seed)
    n = rng.integers(15, 120)
    scores = rng.random(n)
    labels = rng.random(n) < 0.3
    if labels.sum() in (0, n):
        pytest.skip("degenerate draw")
    mine = M.roc_auc(scores, labels, np.arange(n))["auc"]
    order = np.argsort(-scores, kind="stable")
    rows = [[float(scores[i]), bool(labels[i])] for i in order]
    theirs = rdkit_scoring.CalcAUC(rows, 1)
    assert mine == pytest.approx(theirs, abs=1e-9)


# ---------------------------------------------------------------------------
# BEDROC: cross-checked against RDKit's CalcBEDROC (same published formula)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("alpha", [8.0, M.BEDROC_ALPHA_DEFAULT,
                                   M.BEDROC_ALPHA_EARLY_1PCT])
def test_bedroc_matches_rdkit(seed, alpha):
    rng = np.random.default_rng(seed)
    n = rng.integers(20, 250)
    scores = rng.random(n)
    labels = rng.random(n) < 0.15
    if labels.sum() in (0, n):
        pytest.skip("degenerate draw")
    mine = M.bedroc(scores, labels, np.arange(n), alpha=alpha)["bedroc"]
    order = np.argsort(-scores, kind="stable")
    rows = [[float(scores[i]), bool(labels[i])] for i in order]
    theirs = rdkit_scoring.CalcBEDROC(rows, 1, alpha)
    assert mine == pytest.approx(theirs, abs=1e-9)


def test_bedroc_all_actives_first_is_near_one_all_actives_last_is_near_zero():
    n, n_act = 500, 20
    scores = np.arange(n, 0, -1, dtype=np.float64)
    labels_first = np.zeros(n, bool)
    labels_first[:n_act] = True
    labels_last = np.zeros(n, bool)
    labels_last[-n_act:] = True
    b_first = M.bedroc(scores, labels_first, np.arange(n))["bedroc"]
    b_last = M.bedroc(scores, labels_last, np.arange(n))["bedroc"]
    assert b_first > 0.99
    assert b_last < 0.01


# ---------------------------------------------------------------------------
# EF at several fractions, cross-checked against RDKit's CalcEnrichment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_ef_matches_rdkit_calc_enrichment(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(100, 500)
    scores = rng.random(n)
    labels = rng.random(n) < 0.1
    if labels.sum() == 0:
        pytest.skip("degenerate draw")
    fractions = (0.005, 0.01, 0.05)
    mine = M.enrichment_factors(scores, labels, fractions, np.arange(n))
    order = np.argsort(-scores, kind="stable")
    rows = [[float(scores[i]), bool(labels[i])] for i in order]
    theirs = rdkit_scoring.CalcEnrichment(rows, 1, list(fractions))
    for f, t in zip(fractions, theirs):
        assert mine[f].ef == pytest.approx(t, abs=1e-9)


# ---------------------------------------------------------------------------
# The report and the replicate summary
# ---------------------------------------------------------------------------

def test_enrichment_report_bundles_ef_auc_bedroc_for_one_target():
    rng = np.random.default_rng(30)
    n = 400
    scores = rng.random(n)
    labels = rng.random(n) < 0.1
    rep = M.enrichment_report(scores, labels, np.arange(n), target_id="T1")
    d = rep.as_dict()
    assert d["target_id"] == "T1"
    assert set(d["ef"]) == {"0.005", "0.01", "0.05"}
    assert "auc" in d and "bedroc" in d
    assert set(d["bedroc"]) == {"20.0", "321.9"}


def test_replicate_summary_reports_min_and_p10_never_a_bare_mean():
    # A good mean must not be reportable without min/p10 beside it.
    values = np.array([0.95, 0.96, 0.97, 0.98, 0.10])   # one destroyed target
    s = M.replicate_summary(values, B=500)
    assert s["min"] == pytest.approx(0.10)
    assert s["p10"] < s["mean"]
    assert "mean_ci95_bca" in s

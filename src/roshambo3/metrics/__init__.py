"""``roshambo3.metrics`` -- the single implementation of recall, rank agreement and
score error for ROSHAMBO3.

Nothing downstream may claim an accuracy number without going through this
module.  The point is not convenience: two analyses that each write their own
recall will differ in tie-breaking, in aggregation, or in which population they
condition on, and their numbers will then be incomparable without anyone
noticing.

Import path -- it is an installed package, so there is no path manipulation::

    import roshambo3.metrics as M

What is here, and the rule each piece enforces:

===========================  =====================================================
:mod:`~roshambo3.metrics.ranking`  the one tie-break (descending score, ascending id),
                              applied identically to gold and candidate; the
                              conformer -> molecule reduction
:mod:`~roshambo3.metrics.recall`   ``R(N,k)``, the **epsilon->gamma curve**, survivor
                              fractions; mean/min/p10 and the mandatory
                              per-query table
:mod:`~roshambo3.metrics.enrichment` EF at any cut, max-pooled EF, ROC-AUC, BEDROC;
                              refuses conformer-level input;
                              reports the boundary-tie count alongside every EF
:mod:`~roshambo3.metrics.rankagree` RBO_ext at p = 0.98 and 0.998; tau-b as a
                              secondary; Spearman-on-intersection **raises**
:mod:`~roshambo3.metrics.scoreerr` signed Delta by stratum / on G_N' / on A_N'
                              (labeled selection-biased); size bias by OLS with
                              a query fixed effect and a BCa bootstrap over
                              queries
:mod:`~roshambo3.metrics.simulate` cost an approximation before implementing it: add
                              the empirical Delta to gold, re-rank, read off
                              recall
:mod:`~roshambo3.metrics.stats`    BCa bootstrap over queries, paired bootstrap,
                              winner's-curse inflation, Holm-Bonferroni with the
                              3-finalist cap
:mod:`~roshambo3.metrics.floors`   the measured, **stratum-dependent** noise floors,
                              and the refusal to make a per-pair claim below
                              T = 0.55
:mod:`~roshambo3.metrics.prereg`   pre-registration schema and the HOLDOUT ledger
:mod:`~roshambo3.metrics.decision` the frozen primary metric P and the accept rule
===========================  =====================================================
"""

from .decision import (
    ACCEPTANCE, PRIMARY_METRIC_VERBATIM, THRESHOLD_DERIVATION, THROUGHPUT_RULE,
    accept, evaluate_primary,
)
from .enrichment import (
    BEDROC_ALPHA_DEFAULT, BEDROC_ALPHA_EARLY_1PCT, DEGENERACY_MIN_DISTINCT_VALUES,
    EF_ROUNDING_DEFAULT, EF_ROUNDINGS, ConformerLevelInputError,
    DegenerateEnrichmentError, EnrichmentResult, assert_compound_level, bedroc,
    boundary_tie_info, cut_size, enrichment_factor, enrichment_factors,
    enrichment_report, max_pool_scores, replicate_summary, roc_auc,
)
from .floors import (
    EPSILON_FLOOR, EPSILON_FLOOR_ALL_PAIRS, EPSILON_FLOOR_TOP250, GAUGE_FLOOR,
    GAUGE_FLOOR_ALL_PAIRS_P99, GAUGE_FLOOR_RETRIEVED_P99, GOLD_SCOPE,
    GoldScopeError, SCORE_STRATA, assert_gold_supports_per_pair_claim,
    below_floor, epsilon_floor, gauge_floor, improvement_verdict, stratum_of,
)
from .prereg import HoldoutLedger, HoldoutViolation, PreRegistration
from .rankagree import (
    DisqualifiedMetric, RBO_P_DEEP, RBO_P_HEADLINE, gold_top_landing_table,
    kendall_tau_b_on_gold, rank_agreement_report, rbo_ext,
    spearman_on_intersection,
)
from .ranking import (
    TieBreakError, assert_unique_ids, canonical_ids, competition_rank,
    per_query_rank, rank_g, rank_order, reduce_over_conformers, top_k_order,
    top_n,
)
from .recall import (
    EPS_GAMMA_K, RecallReport, recall_at, recall_curve, recall_report,
    recall_report_both, survivor_recall,
)
from .scoreerr import (
    delta_by_decile, delta_by_stratum, score_error_report, signed_summary,
    size_bias,
)
from .simulate import (
    EmpiricalDelta, GaussianDelta, decade_width, epsilon_gamma_curve,
    simulate_recall,
)
from .stats import (
    MAX_EVAL_FINALISTS, bca_ci, holm_bonferroni, paired_bootstrap,
    required_n_for_delta, resolvable_delta, winner_inflation,
)

__all__ = [n for n in dir() if not n.startswith("_")]

__version__ = "1.0.0"

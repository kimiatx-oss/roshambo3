"""Enrichment: does the ranking put actives at the top?

Every other module in this package measures *fidelity of the port* -- recall
against an internal gold, RBO against a reference ranking, score error against
an exact rescore.  None of it says whether a labeled active lands near rank 1,
which is the question the comparison paper answers and the one a chemist asks
first.  This module is that metric.

**EF1% (comparison paper eq 6).**

    EF1% = (#actives in top 1% / size of top 1%) / (total #actives / total N)

Everything below generalises it: EF at an arbitrary fraction, max-pooled EF
for a multi-query target, ROC-AUC, and BEDROC.  All four are compound-level by
construction (see "Molecule-level, refused" below) and all four reuse the
project's ranking primitives -- :func:`roshambo3.metrics.top_k_order` and
:func:`roshambo3.metrics.rank_g` -- rather than re-sorting.

**The cut rounding, decided.**  ``cut = ceil(fraction * N)``, not floor.
Three reasons, in order of how much they'd cost to get wrong:

1. ``floor`` can round a positive fraction to a **zero-size cut** on a small
   target (``floor(0.01 * 50) = 0``): EF is then a division by zero rather
   than a small number. ``ceil`` is never zero for ``fraction > 0, N >= 1``.
2. "The top f%" reads as a *lower bound* on the fraction of the library
   retrieved -- ceil is the rounding that never retrieves less than the
   nominal fraction asked for.
3. It matches existing practice: RDKit's ``CalcEnrichment``
   (``rdkit.ML.Scoring.Scoring``) uses ``math.ceil(numMol * f)`` for exactly
   this cut, so this choice does not introduce a new convention the field
   does not already use.

``floor`` is reachable via ``rounding="floor"`` -- by argument, not by
editing this file.  On a small target the two differ
by one row, which on a 5-active target is a whole enrichment point; see
:func:`enrichment_factor`'s ``n_tied_at_boundary`` field for when the choice
that matters more -- the tie-break -- is the one actually deciding the cut.

**Ties at the cut boundary.**  The cut is taken under
``roshambo3.numerics.rank_order`` -- descending score, ties by ascending
canonical id -- via :func:`top_k_order`, exactly as every other ranked-list
statistic in this package.  When one or more rows share the score value that
sits at the cut boundary, which of them land inside the cut is decided by the
tie-break, not by the score.  ``enrichment_factor`` always reports
``n_tied_at_boundary`` (how many rows in the whole ranking share that exact
score -- 1 means no tie, just the boundary row itself) and
``n_active_tied_at_boundary`` (how many of those are labeled active), so a
reader can see when EF is resting on the tie-break rather than on the score.  ``tests/test_enrichment.py`` constructs the case directly: a
single active tied with several inactives exactly at the cut.

**Molecule-level, refused.**  EF's label attaches to a *compound*: "this
molecule is a known ligand of the target" is not a property of one of its
conformers.  A compound is ranked by its best-scoring conformer (see ``roshambo3.metrics.reduce_over_conformers``), and every
function here **requires** a ``mol_ids`` array and refuses if it is not
unique per row -- i.e. refuses conformer-level input rather than silently
answering a different question. This module does not do the reduction itself: pass
already-reduced, one-row-per-compound scores, with ``mol_ids`` naming which
compound each row is (any array that is unique per row works; the compound's
own canonical id is the natural choice and doubles as the tie-break ``ids``
when none of the scores collide across compounds).

**Max-pooled EF is a different estimator from mean-per-query EF.**  For a
target with several query ligands, :func:`max_pool_scores` takes each
library compound's best score across the queries -- literally the same
reduction as :func:`roshambo3.metrics.reduce_over_conformers`, just over the
query axis instead of the conformer axis, so it is implemented by calling it
rather than re-deriving it -- and the pooled scores are ranked *once*. Pooling
inflates every compound's score (it is a max over several draws), actives and
inactives alike, so a pooled EF and the mean of per-query EFs are never
interchangeable and must never be averaged together.

**ROC-AUC uses midranks, not the canonical tie-break -- on purpose.**  AUC is
the Mann-Whitney U-statistic: the probability a random active outscores a
random inactive, with a tie credited 0.5 either way.  That definition is
symmetric in the tie by construction, so resolving a tie by an arbitrary
canonical id (as EF and BEDROC must, because a *cut* needs a total order)
would inject a dependency on molecule id that the statistic is not supposed to
have. :func:`roc_auc` therefore uses ``scipy.stats.rankdata(method="average")``
-- the standard tie-corrected estimator -- and this is the one place in
``roshambo3.metrics`` where the canonical id tie-break is deliberately *not*
used, with the reason recorded here rather than left for someone to discover
by diffing against the rest of the package.

**BEDROC** (Truchon, J.-F.; Bayly, C. I. *J. Chem. Inf. Model.* **2007**, *47*,
488-508) sits between EF (one cut) and AUC (the whole ranking): an
exponentially-decaying weight over rank, so early ranks count far more than
late ones without discarding the tail entirely.  It uses the canonical
tie-break (same reasoning as EF: it is built from a definite rank per active,
same as a cut). Reported at ``alpha = 20`` (the paper's general-purpose
choice) and ``alpha = 321.9`` (the convention for 1% early recognition) --
both; the paper reports only EF1%, so BEDROC and AUC
are what a claim about ranking quality (as opposed to one cut) rests on.

**Degeneracy.**  With ``n_actives`` actives and a cut of size ``cut``, the
number of actives landing in the cut is hypergeometric and can take at most
``min(cut, n_actives, N - n_actives) + 1`` distinct values -- the
example, "with 5 actives the metric can only take 6 values", is this formula
at ``cut >= 5``. :func:`enrichment_factor` always reports this count and a
``degenerate`` flag (threshold :data:`DEGENERACY_MIN_DISTINCT_VALUES`) rather
than silently returning a number with no sampling resolution behind it. The
truly undefined cases -- no actives at all, or a cut of zero -- raise
:class:`DegenerateEnrichmentError` instead, because there is no ratio to
report, degenerate or not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ranking import rank_g, reduce_over_conformers, top_k_order

__all__ = [
    "EF_ROUNDINGS",
    "EF_ROUNDING_DEFAULT",
    "BEDROC_ALPHA_DEFAULT",
    "BEDROC_ALPHA_EARLY_1PCT",
    "DEGENERACY_MIN_DISTINCT_VALUES",
    "ConformerLevelInputError",
    "DegenerateEnrichmentError",
    "assert_compound_level",
    "cut_size",
    "boundary_tie_info",
    "EnrichmentResult",
    "enrichment_factor",
    "enrichment_factors",
    "max_pool_scores",
    "roc_auc",
    "bedroc",
    "enrichment_report",
    "replicate_summary",
]

EF_ROUNDINGS = ("ceil", "floor")
EF_ROUNDING_DEFAULT = "ceil"

#: The paper's general-purpose BEDROC choice.
BEDROC_ALPHA_DEFAULT = 20.0
#: The convention for 1% early recognition: alpha such that ~80% of the
#: BEDROC weight sits in the top 1% of the ranking (Truchon & Bayly 2007).
BEDROC_ALPHA_EARLY_1PCT = 321.9

#: Threshold on the number of distinct values EF's hypergeometric support can
#: take before it is flagged degenerate. 10 is chosen so that the
#: worked example (5 actives -> 6 possible values) is flagged, with a little
#: room to spare, while a target with a few dozen actives is not flagged
#: merely for being modestly sized.
DEGENERACY_MIN_DISTINCT_VALUES = 10


class ConformerLevelInputError(ValueError):
    """Raised when ``mol_ids`` is not unique per row.

    EF's label is a property of a compound, not of one of its conformers,
    so an EF computed over unaggregated conformer-level rows is not a wrong
    number -- it is the answer to a different question. The fix
    is the reduction ``roshambo3.metrics.reduce_over_conformers``.
    """


class DegenerateEnrichmentError(ValueError):
    """Raised when the requested ratio has no defined value.

    Covers "no actives at all" (every EF/AUC/BEDROC denominator involving the
    active rate is then zero) and "a fraction that rounds to a zero-size
    cut". Both are refusals, not a number with a degeneracy flag attached,
    because there is no ratio to attach the flag to.
    """


def assert_compound_level(mol_ids, what="mol_ids"):
    """Refuse conformer-level input: ``mol_ids`` must be unique per row.

    Mirrors :func:`roshambo3.metrics.assert_unique_ids` in spirit (that one
    guards the ranking tie-break id; this one guards the statistical unit EF
    is defined over) but is a separate check and a separate exception type,
    because the two failures are diagnosed differently: a repeated *tie-break*
    id is a ranking bug even at molecule level, whereas a repeated *mol_id*
    means the rows are conformers, not compounds, and no tie-break fixes that.
    """
    m = np.asarray(mol_ids).ravel()
    if m.shape[0] == 0:
        raise ValueError(f"{what} is empty")
    n_uniq = np.unique(m).size
    if n_uniq != m.shape[0]:
        raise ConformerLevelInputError(
            f"{what} has {m.shape[0] - n_uniq} duplicate value(s) out of "
            f"{m.shape[0]} rows: this looks like conformer-level input, not "
            "compound-level. Enrichment's label is a property of a compound "
            "-- 'this molecule is a known ligand' -- not of one of its "
            "conformers, so an enrichment computed over unaggregated "
            "conformers is not a wrong number, it is an answer to a "
            "different question. Reduce to one best-scoring row "
            "per compound first, with "
            "roshambo3.metrics.reduce_over_conformers(scores, mol_ids), "
            "which this module deliberately does not do for you.")


def cut_size(n, fraction, rounding=EF_ROUNDING_DEFAULT):
    """The number of rows in the top ``fraction`` of a library of size ``n``.

    ``rounding='ceil'`` (default, see the module docstring for why) or
    ``'floor'``, reachable by argument. Neither is clamped to be nonzero --
    a ``floor`` cut of zero is exactly the degenerate case
    :func:`enrichment_factor` is written to catch and refuse, not to paper
    over by silently bumping it to 1.
    """
    if rounding not in EF_ROUNDINGS:
        raise ValueError(f"rounding must be one of {EF_ROUNDINGS}, got {rounding!r}")
    n = int(n)
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    raw = fraction * n
    cut = math.ceil(raw) if rounding == "ceil" else math.floor(raw)
    return int(min(cut, n))


def boundary_tie_info(scores, cut, ids=None, labels=None):
    """How many rows share the score value sitting at the cut boundary.

    Returns ``(n_tied_at_boundary, n_active_tied_at_boundary)``. The first is
    the count of rows anywhere in ``scores`` (inside the cut or not) whose
    value equals the score of the last row the cut admits -- i.e. how many
    rows a different, non-compliant tie-break could have swapped across the
    boundary. **1 means "no tie": just the boundary row itself.** Only when
    ``cut <= 0`` or ``cut >= n`` (no boundary exists at all) is it 0.
    ``labels`` is optional; when given, the second return value is how many
    of those tied rows are actives, which is what determines whether the
    tie-break could actually have changed EF's numerator (a tie entirely
    among inactives changes who is *in* the cut but not the active count).
    ``cut <= 0`` or ``cut >= n`` has no boundary to speak of and both counts
    are 0.
    """
    s = np.asarray(scores)
    n = s.shape[0]
    if cut <= 0 or cut >= n:
        return 0, 0
    order = top_k_order(s, cut, ids)
    boundary_score = s[order[-1]]
    tied = s == boundary_score
    n_tied = int(tied.sum())
    n_active_tied = int(np.asarray(labels)[tied].sum()) if labels is not None else 0
    return n_tied, n_active_tied


@dataclass
class EnrichmentResult:
    """One EF measurement, with every number a reader needs to trust it."""

    fraction: float
    rounding: str
    n: int
    n_actives: int
    cut: int
    n_actives_in_cut: int
    ef: float
    baseline_active_rate: float
    n_tied_at_boundary: int
    n_active_tied_at_boundary: int
    n_possible_distinct_values: int
    degenerate: bool

    def as_dict(self):
        return {
            "fraction": self.fraction,
            "rounding": self.rounding,
            "n": self.n,
            "n_actives": self.n_actives,
            "cut": self.cut,
            "n_actives_in_cut": self.n_actives_in_cut,
            "ef": self.ef,
            "baseline_active_rate": self.baseline_active_rate,
            "n_tied_at_boundary": self.n_tied_at_boundary,
            "n_active_tied_at_boundary": self.n_active_tied_at_boundary,
            "n_possible_distinct_values": self.n_possible_distinct_values,
            "degenerate": self.degenerate,
            "degenerate_threshold": DEGENERACY_MIN_DISTINCT_VALUES,
        }


def enrichment_factor(scores, labels, fraction, mol_ids, ids=None,
                      rounding=EF_ROUNDING_DEFAULT):
    """EF at ``fraction`` (the comparison paper's eq 6, generalised to any f).

        EF = (n_actives_in_cut / cut) / (n_actives / n)

    ``scores`` and ``labels`` (boolean or 0/1) are one row per **compound** --
    enforced via ``mol_ids``, see :func:`assert_compound_level`, which raises
    :class:`ConformerLevelInputError` rather than silently reducing or
    silently answering the wrong question. ``ids`` is the canonical tie-break
    id (defaults to row position, which is a valid but arbitrary tie-break --
    pass the compound's real canonical id for a reproducible one).

    Raises :class:`DegenerateEnrichmentError` if there are no actives at all
    (the denominator is undefined) or if ``fraction`` rounds to a zero-size
    cut. A small-but-nonzero active count or cut is not refused -- it comes
    back with ``degenerate=True`` and the count of values the statistic could
    actually take: report it, do not drop it.
    """
    assert_compound_level(mol_ids)
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must match")
    n = s.shape[0]
    n_actives = int(y.sum())
    if n_actives == 0:
        raise DegenerateEnrichmentError(
            "no actives in this pool: EF's denominator (active rate) is "
            "zero, so EF is undefined, not merely small.")

    cut = cut_size(n, fraction, rounding)
    if cut <= 0:
        raise DegenerateEnrichmentError(
            f"fraction={fraction} rounds to a zero-size cut at N={n} under "
            f"rounding={rounding!r}; EF cannot divide by a zero-size cut. "
            "Use a larger fraction, more compounds, or rounding='ceil' "
            "(the default) instead of 'floor'.")

    order = top_k_order(s, cut, ids)
    n_in_cut = int(y[order].sum())
    baseline = n_actives / n
    ef = (n_in_cut / cut) / baseline

    n_tied, n_active_tied = boundary_tie_info(s, cut, ids, y)
    n_possible = min(cut, n_actives, n - n_actives) + 1
    degenerate = n_possible <= DEGENERACY_MIN_DISTINCT_VALUES

    return EnrichmentResult(
        fraction=float(fraction), rounding=rounding, n=n, n_actives=n_actives,
        cut=cut, n_actives_in_cut=n_in_cut, ef=float(ef),
        baseline_active_rate=float(baseline),
        n_tied_at_boundary=n_tied, n_active_tied_at_boundary=n_active_tied,
        n_possible_distinct_values=int(n_possible), degenerate=bool(degenerate))


def enrichment_factors(scores, labels, fractions, mol_ids, ids=None,
                       rounding=EF_ROUNDING_DEFAULT):
    """:func:`enrichment_factor` at several fractions in one call.

    Computed once per fraction (the cut differs), sharing the validated
    ``scores``/``labels``/``mol_ids``. Returns ``{fraction: EnrichmentResult}``.
    """
    assert_compound_level(mol_ids)
    return {float(f): enrichment_factor(scores, labels, f, mol_ids, ids, rounding)
            for f in fractions}


def max_pool_scores(scores, mol_ids):
    """Max-pool a ``[n_query, n_compound]`` score matrix over queries.

    Each library compound takes its best score across the target's queries;
    the pooled scores are then ranked once (by the caller, e.g. via
    :func:`enrichment_factor` on the result). This is exactly
    :func:`roshambo3.metrics.reduce_over_conformers`'s reduction applied to
    the query axis instead of the conformer axis -- "many noisy observations
    of the same labeled unit; keep the best" is the same operation either
    way, so it is reused rather than re-derived. **This is a different
    estimator from the mean of per-query EFs**, with a different null: max
    is not linear, so pooling inflates every compound's score, actives and
    inactives alike, and a pooled EF must never be averaged with per-query
    EFs.

    Returns ``(compound_ids, pooled_scores, n_queries)`` with
    ``compound_ids`` ascending -- i.e. usable directly as the canonical tie-
    break id for the pooled ranking, same convention as
    ``reduce_over_conformers``. ``labels`` are per-compound and do not need
    pooling; reindex them to the returned ``compound_ids`` order (e.g. via a
    dict or a sorted-array lookup) before calling :func:`enrichment_factor`
    on the pooled result -- :func:`enrichment_report` does this for you when
    given the multi-query form directly.
    """
    s = np.atleast_2d(np.asarray(scores, np.float64))
    nq, m = s.shape
    cid = np.asarray(mol_ids).ravel()
    if cid.shape[0] != m:
        raise ValueError(f"mol_ids has {cid.shape[0]} entries but scores has "
                         f"{m} compounds per query")
    tiled = np.tile(cid, nq)          # matches s.ravel()'s row-major order
    uniq, best, n_obs, _ = reduce_over_conformers(s.ravel(), tiled)
    return uniq, best, n_obs


def roc_auc(scores, labels, mol_ids):
    """ROC-AUC via the Mann-Whitney U statistic with midrank tie correction.

    See the module docstring for why this is the one function in the package
    that does **not** use the canonical id tie-break: AUC's definition
    credits a tie 0.5 either way, which midranks compute exactly and an
    arbitrary id-based resolution would not.

    Raises :class:`DegenerateEnrichmentError` if either class is absent (AUC
    is undefined with no actives or no inactives to compare against).
    """
    from scipy.stats import rankdata

    assert_compound_level(mol_ids)
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must match")
    n = s.shape[0]
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise DegenerateEnrichmentError(
            f"AUC undefined: n_actives={n_pos}, n_inactives={n_neg}; both "
            "classes must be present.")

    ranks = rankdata(s, method="average")   # ascending; midranks on ties
    U = float(ranks[y].sum() - n_pos * (n_pos + 1) / 2.0)
    auc = U / (n_pos * n_neg)
    n_tied_groups = int(s.shape[0] - np.unique(s).shape[0])
    return {"auc": float(auc), "n": n, "n_actives": n_pos, "n_inactives": n_neg,
            "n_rows_in_a_tied_group": n_tied_groups,
            "tie_method": "midrank (scipy.stats.rankdata, method='average'); "
                          "NOT the canonical id tie-break, see module docstring"}


def bedroc(scores, labels, mol_ids, ids=None, alpha=BEDROC_ALPHA_DEFAULT):
    """BEDROC at decay parameter ``alpha`` (Truchon & Bayly 2007).

    Uses the canonical id tie-break (via :func:`roshambo3.metrics.rank_g`),
    same reasoning as EF: it is built from one definite rank per active row,
    same as a cut needs.

        RIE   = (1/n_actives) * sum_i exp(-alpha * r_i / n)  /  denom
        denom = (1/n) * (1 - exp(-alpha)) / (exp(alpha/n) - 1)
        BEDROC = (RIE - RIE_min) / (RIE_max - RIE_min)

    with ``RIE_max``/``RIE_min`` the value RIE takes when every active is
    ranked first/last. This is the closed-form, tie-free-assuming formula
    used by RDKit's ``rdkit.ML.Scoring.Scoring.CalcBEDROC`` (cross-checked
    against it directly in ``tests/test_enrichment.py``); ``r_i`` here comes
    from this project's canonical tie-break rather than an unspecified input
    order.

    Raises :class:`DegenerateEnrichmentError` if there are no actives. If
    every row is active, ``RIE_max == RIE_min`` and BEDROC is defined as 1.0
    (matching RDKit) rather than a 0/0 -- also reported as
    ``n_possible_distinct_values = 1`` via the degenerate flag, since with no
    inactives the statistic cannot take any other value.
    """
    assert_compound_level(mol_ids)
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must match")
    n = s.shape[0]
    n_actives = int(y.sum())
    if n_actives == 0:
        raise DegenerateEnrichmentError("BEDROC undefined: no actives.")
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    r = rank_g(s, ids).astype(np.float64)     # 1-based, total order
    ratio = n_actives / n
    denom = (1.0 / n) * (1.0 - math.exp(-alpha)) / (math.exp(alpha / n) - 1.0)
    sum_exp = float(np.sum(np.exp(-alpha * r[y] / n)))
    rie = sum_exp / (n_actives * denom)
    rie_max = (1.0 - math.exp(-alpha * ratio)) / (ratio * (1.0 - math.exp(-alpha)))
    rie_min = (1.0 - math.exp(alpha * ratio)) / (ratio * (1.0 - math.exp(alpha)))
    if rie_max != rie_min:
        score = (rie - rie_min) / (rie_max - rie_min)
    else:
        score = 1.0

    n_possible = min(n_actives, n - n_actives) + 1
    return {"bedroc": float(score), "alpha": alpha, "n": n,
            "n_actives": n_actives, "active_ratio": float(ratio),
            "rie": float(rie), "rie_min": float(rie_min), "rie_max": float(rie_max),
            "n_possible_distinct_values": int(n_possible),
            "degenerate": bool(n_possible <= DEGENERACY_MIN_DISTINCT_VALUES),
            "degenerate_threshold": DEGENERACY_MIN_DISTINCT_VALUES,
            "reference": "Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488-508"}


@dataclass
class EnrichmentReport:
    """EF at several fractions, AUC, and BEDROC at several alphas, for one
    target's ranking -- everything reported per target, in one call."""

    target_id: str
    n: int
    n_actives: int
    ef: dict = field(default_factory=dict)          # {fraction: EnrichmentResult}
    auc: dict = field(default_factory=dict)
    bedroc: dict = field(default_factory=dict)      # {alpha: dict}

    def as_dict(self):
        return {
            "target_id": self.target_id,
            "n": self.n,
            "n_actives": self.n_actives,
            "ef": {str(f): r.as_dict() for f, r in self.ef.items()},
            "auc": self.auc,
            "bedroc": {str(a): r for a, r in self.bedroc.items()},
        }


def enrichment_report(scores, labels, mol_ids, ids=None, target_id="target",
                      fractions=(0.005, 0.01, 0.05), rounding=EF_ROUNDING_DEFAULT,
                      alphas=(BEDROC_ALPHA_DEFAULT, BEDROC_ALPHA_EARLY_1PCT)):
    """EF at ``fractions``, AUC, and BEDROC at ``alphas``, for one ranking.

    ``scores``/``labels``/``mol_ids`` are one row per compound, exactly as
    :func:`enrichment_factor` requires (refused otherwise). This is the
    per-target call an enrichment study makes once per target (shape-only arm) or once
    per pooled ranking (max-pooled arm, after :func:`max_pool_scores`); the
    between-target table they build from many of these is theirs, since a
    43- or 15-row table with its own between-target uncertainty is what
    acceptance criteria ask for -- this function is the one-target
    unit it is built from. :func:`replicate_summary` is the aggregator once
    you have one array per target.
    """
    assert_compound_level(mol_ids)
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    ef = enrichment_factors(s, y, fractions, mol_ids, ids, rounding)
    auc = roc_auc(s, y, mol_ids)
    bed = {a: bedroc(s, y, mol_ids, ids, alpha=a) for a in alphas}
    return EnrichmentReport(target_id=target_id, n=int(s.shape[0]),
                            n_actives=int(y.sum()), ef=ef, auc=auc, bedroc=bed)


def replicate_summary(values, B=10_000, rng=None):
    """Mean/min/p10/median/sd over replicates, plus a BCa CI over them.

    "Replicate" is whatever the caller resampled over -- the rule ("the
    honest replication unit is the query, not the item") generalises here to
    "the target, not the compound" for a between-target summary, or "the
    query, not the item" again for a within-target, across-queries summary.
    Which one this is is the caller's business (pass the right array); this
    function is the same aggregation either way, reusing
    ``roshambo3.metrics.stats.bca_ci`` rather than a second bootstrap
    implementation. Never report a bare mean: this always carries min and p10
    beside it.
    """
    from .stats import bca_ci

    v = np.asarray(values, np.float64).ravel()
    ci = bca_ci(v, np.mean, B=B, rng=rng)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "p10": float(np.percentile(v, 10)),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "mean_ci95_bca": [ci["lo"], ci["hi"]],
        "bootstrap": {k: ci[k] for k in ("B", "z0", "acceleration", "n_queries")},
    }

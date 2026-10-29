# Metrics policy: recall, rank agreement, score error, enrichment

Implemented by `src/roshambo3/metrics/`, tested by `tests/test_metrics.py` and
`tests/test_enrichment.py`. Enrichment (EF, ROC-AUC, BEDROC) is section 11.
The paper repository (https://github.com/kimiatx-oss/roshambo3-paper) publishes an excerpt of the development record
(`tickets/`, `decisions/`); the measurements summarized here are not otherwise
distributed.

Several measured numbers below were taken on the ROSHAMBO2 code base; where
this document names code, it names ROSHAMBO3's.

This is the counterpart to `doc/NUMERICS_POLICY.md`. That document pins the
arithmetic; this one pins the **measurement**. Same principle: wherever a rule
can be checked it *is* checked, in code, and the check is named here. A rule
that appears only in this file is a rule a refactor will break without anyone
noticing.

**The hard gate:** nothing downstream may claim a recall, rank-agreement,
score-error, or enrichment number that did not come from `roshambo3.metrics`.
Two harnesses that each write their own recall (or their own EF) will differ in
tie-breaking, in aggregation, or in which population they condition on, and
their numbers will then be incomparable without anyone being able to see why.
DUDE-Z and LIT-PCBA enrichment therefore call into section 11 rather than
computing EF inline.

A `decision` status on a `kind=rate` run certifies rate evidence only. Recall,
RBO, and top-K values in that run's metrics are supplemental; cite them as
decision-grade accuracy evidence only from a separately graded `kind=accuracy`
record. Rate and accuracy grades remain separate.

---

## Bounded regression detectors and full-library qualification

The [screening workflow](SCREENING_REGRESSION.md) specifies which obligation a
change triggers. The frozen screening detector reports R(8,16) on 128 selected
conformers per query (80 selected molecules), both levels, all per-query rows,
mean/p10/min, query/target size and selected conformer-count strata, top-end
movement and signed bias. Its 128-row challenger-enriched subset cannot estimate
global full-library recall. Exact near-tie order is diagnostic under this rule.
The budgets belong to that local operating point; do not copy the default
`metrics.accept` thresholds or the full-library acceptance values into a new
library/depth.

Less frequent qualification evaluates the complete production cascade at its
actual survivor budget against full-population references, using both immutable
fixed and current baselines to prevent cumulative drift. Reuse a prior
qualification only where source, geometry, configuration and population match.
Queries/targets are the independent bootstrap units, never individual conformers.
Missing hardware/corpus or unqualified fixture transfer remains visible unknown
coverage, not a pass. Candidate-only runtime receipts separate an observed
budget fit from environment qualification; contention remains advisory. The
explicit finite optimizer-model lane retains its frozen failing assertions,
while default coverage records that model unqualified/not-run. Deliberate
score/pose movement still requires qualification; physical measurement
uncertainty never excuses systematic selection bias.

## 1. How to use it

```python
import roshambo3.metrics as M

ids = M.canonical_ids(mol_id, conf_idx)      # unique tie-break key -- see 2
rep = M.evaluate_primary(gold, cand, n_prime=1000, ids=ids,
                         query_ids=qids, hac=n_real, bucket=bucket,
                         mol_ids=mol_id)     # mol_ids -> molecule level too
verdict = M.accept(rep, base_primary=baseline_rep,
                   throughput_ratio=r, throughput_ci=[lo, hi],
                   timing_basis="kernel-only, locked, uncontended")
```

`gold` and `cand` are `[n_query, n_target]` score matrices (column 1,
`T_shape`, of the 20-column score row -- see `roshambo3.scores`). Everything else is derived.

### The surface, by what you are asking

| question | call | notes |
|---|---|---|
| what rank is this item? | `rank_g(scores, ids)` | 1-based, total order, no ties |
| what are the top k? | `top_k_order(scores, k, ids)` | exact; ~30x faster than a full sort at M = 1e7 |
| what is recall? | `recall_at(gold, cand, N, k, ids)` | `R(N,k) = \|G_N ∩ A_k\| / N` |
| what does a cascade get? | `recall_curve(gold, cand, N', ks, ids)` | **the ε→γ curve** |
| what does *my filter* get? | `survivor_recall(gold, mask, N', ids)` | returns `(recall, survivor_fraction)` |
| the whole recall bundle | `recall_report_both(...)` | both levels in one object; `recall_report(..., level=)` is the single-level building block |
| do the rankings agree? | `rank_agreement_report(...)` | RBO_ext(0.98) and (0.998), τ-b, gold-top-10 landing table |
| how wrong are the scores? | `score_error_report(...)` | signed Δ by stratum, on G_N′, on A_N′, size bias |
| what would X cost, before I build it? | `EmpiricalDelta.fit(...)` → `epsilon_gamma_curve(...)` | the simulation path |
| is this difference real? | `paired_bootstrap(base, cand)` | BCa, over queries, with δ = 0.01 |
| how big must the panel be? | `required_n_for_delta(sd_d, δ)` | and `resolvable_delta(sd_d, n)` |
| how much did searching K configs inflate my winner? | `winner_inflation(sd_d, n, K)` | |
| may I run on HOLDOUT? | `HoldoutLedger(...).touch(...)` | refuses a repeat of the same `config_hash` |
| does a hit land at the top? | `enrichment_factor(scores, labels, f, mol_ids)` | EF at any cut fraction; section 11 |
| EF, AUC and BEDROC together, one target | `enrichment_report(scores, labels, mol_ids, ...)` | section 11 |
| pool a multi-query target first | `max_pool_scores(scores, mol_ids)` | different estimator from mean-of-per-query EF; section 11 |
| between-target (or between-query) spread | `replicate_summary(values)` | mean/min/p10 + BCa CI; section 11 |

Five calls **raise** rather than return, on purpose:

* `spearman_on_intersection(...)` → `DisqualifiedMetric`, always.
* `assert_unique_ids(ids)` / `rbo_ext` → `TieBreakError` on a repeated id.
* `assert_gold_supports_per_pair_claim(t_ref)` → `GoldScopeError` below T = 0.55.
* `assert_compound_level(mol_ids)`, and every enrichment call that takes
  `mol_ids` → `ConformerLevelInputError` on a repeated id -- EF's label is a
  compound property, so a conformer-level input is a different question, not
  a wrong number (section 11).
* `enrichment_factor` / `roc_auc` / `bedroc` → `DegenerateEnrichmentError`
  when the ratio is truly undefined (no actives, or a zero-size cut) -- a
  small-but-nonzero active count returns a value with a `degenerate` flag
  instead, per section 11.

---

## 2. The tie-break, and why the canonical id is not `molecule_id`

The rule is: **descending score, ties broken by ascending canonical id, applied
identically to both rankings being compared.** `roshambo3.numerics.rank_order`
is the implementation; `roshambo3.metrics.rank_g` is the metrics-side surface.

The canonical id **must be unique per row**. "Ascending molecule_id" is a total
order only on a one-conformer-per-molecule library, and false on a
multi-conformer one (9.5 and 40.4 conformers per molecule on two measured
libraries). A repeated id is not a tie-break: it leaves the order of the tied
rows to an unstable sort, which is the exact failure `NUMERICS_POLICY` §4b
exists to prevent, and it makes a ranked list name the same item twice.

Use `M.canonical_ids(mol_id, conf_idx)`. It is the rank of `(mol_id, conf_idx)`
in lexicographic order: unique by construction, and a strictly increasing
function of `molecule_id`, so it *is* the policy's ascending-molecule_id rule
with the sub-order the policy left unspecified.

Passing `mol_id` as the id on a multi-conformer library makes `rbo_ext` return
an impossible value (1.019 was measured), so every report entry point calls
`assert_unique_ids` once.

---

## 3. Recall, and the ε→γ curve

`R(N, k) = |G_N ∩ A_k| / N`.

`R(N', k·N')` **is** the ε→γ curve: retrieve `k·N'` candidates with the cheap
scorer, rescore all of them with gold, keep the top `N'`. Recall of that
cascade is exactly `R(N', k·N')`, because a rescored survivor is ranked by gold
and lands in the gold top-N′ iff it was there to begin with. The cost axis is
the **survivor fraction** `k·N'/M`.

For a filter that admits a set rather than ranking, use `survivor_recall`.

**Aggregate as mean, min and p10 — never a bare mean, and always publish the
per-query table.** The reason is measured: the 1e-4 quantile threshold ranged
0.70 → 0.95 over four random queries. On one measured 10M-conformer panel, mean
recall is 0.958 and the worst query is 0.714 — and recall by query heavy-atom
quartile is 0.9996 / 0.9959 / 0.9756 / 0.8596, so the mean hides an entire
stratum, not just an outlier.

**The measured curve, both panels, at the metric's own operating point**
(M ≈ 1e7, N′ = 1000, shipped 100-step scorer against the 400-step gold):

| k | survivor fraction | tuning panel (n=96) | held-out panel (n=32) |
|---|---|---|---|
| 1 | 1e-4 | 0.9568 [0.938, 0.970] | 0.9577 [0.926, 0.976] |
| 2 | 2e-4 | 0.9767 | 0.9778 |
| 5 | 5e-4 | 0.9872 | 0.9873 |
| 10 | 1e-3 | 0.9918 | 0.9915 |

The tuning and held-out panels agree to 0.001 at every depth. **P is not
portable across operating points, though:** the same measurement gives 0.9822
at (M = 1e6, N′ = 100) and 0.9308 at (M = 1e4, N′ = 10). An absolute threshold
on P is a statement about one (library, M, N′) and must be re-derived for another.

**Report conformer level and molecule level, and stratify molecule level by
conformer count.** The molecule score is a max over a *variable* number of
conformers, so a 25-conformer molecule has a higher expected max purely by
order statistics. Measured on a 1M-conformer multi-conformer library:
molecule-level recall falls 1.000 → 0.994 → 0.953 across terciles of the gold
top-N′'s median conformer count.

The way this goes wrong is an **omitted** level, not a miscalculated one: a
caller asks for the level it is thinking about and the artifact simply does not
carry the other, so nobody can tell which question the number answered.
Therefore:

* `recall_report_both()` is the entry point and cannot report one level without
  the other. Without `mol_ids` the molecule-level entry says *why* it is absent
  rather than being missing.
* `evaluate_primary()` always carries a `recall_molecule_level` key on the same
  terms, and warns when it could not be computed.

The result API distinguishes the two levels as well, because "the top 100
molecules" is the question a user asks and a conformer-level top 100 does not
answer it — see `roshambo3.engine.MoleculeTopK`.

---

## 4. Rank agreement

* **RBO_ext at p = 0.98** is the headline (expected viewed depth 50 — the
  interpretability-motivated choice). **p = 0.998** (depth 500) alongside it.
* **Kendall τ-b restricted to G_N** is a secondary statistic and never appears
  alone: it weights the swap (rank 1, rank 2) exactly as heavily as
  (rank 99999, rank 100000).
* **The gold top-10 landing table is published raw** and must be read before
  any summary statistic is trusted.
* **Spearman on the intersection is disqualified**, and calling it raises. It
  conditions on `G_N ∩ A_N`, deleting exactly the items the approximation got
  wrong, so a method that drops half the true top-N can score *higher* than one
  that keeps everything with small perturbations. It is not reportable even as
  a secondary.
* **NDCG** is usable only with its gain function fixed in the pre-registration,
  before results are seen. `roshambo3.metrics` does not implement it, so that a
  study which wants it has to write the gain function down first.

---

## 5. Score error

**Δ = candidate − gold. Signed, always.** `mean Δ > 0` inflates scores and
drags non-hits in — a *precision* cost; `mean Δ < 0` deflates and pushes true
hits out — a *recall* cost. These are not symmetric and must never be collapsed
into |Δ|. A smoother objective can also find a genuinely *better* optimum, and
an absolute value makes that improvement look identical to an error.

Three populations, always separated and labeled:

* **global** — dominated by the tail nobody retrieves;
* **G_N′** — the gold top-N′, the stratum that matters;
* **A_N′** — the candidate top-N′, labeled **selection-biased upward** in the
  output itself. Measured on a 10M-conformer panel: mean Δ is −1.78e-3 on G_N′
  and −0.40e-3 on A_N′, a 4.5× understatement.

Size bias: OLS with a **query fixed effect**, CI from a **BCa bootstrap
resampled over queries, never items**. `β` is in Tanimoto per heavy atom.

**Read the HAC decile table before quoting β.** On the same panel the pooled
decile table runs monotonically from −1.9e-6 at HAC 18 to −4.7e-3 at HAC 45 — a
2500× ramp — while the within-query β is *positive*. A single β is not a safe
summary, which is why `size_bias` returns the decile table in the same dict and
carries a warning field.

Also stratify by gold-score decile and by padding bucket; `score_error_report`
does both automatically.

---

## 6. The floors — stratum-dependent, and two of them

A single scalar floor is wrong here by three orders of magnitude.

**ε floor (arithmetic irreproducibility)** — the spread of the per-pair score
across seven real builds of the same source:

| stratum | median | p90 | n |
|---|---|---|---|
| T < 0.4 | 2.1e-3 | 7.2e-3 | 174 |
| 0.4–0.55 | 4.3e-3 | 7.6e-3 | 283 |
| 0.55–0.7 | 4.2e-7 | **7.7e-3** | 43 |
| T ≥ 0.7 | 3.0e-7 | 5.4e-7 | 250 |

Note the 0.55–0.7 row: its *median* is 4.2e-7 but its p90 is four orders
higher on n = 43. Quote the median for "the builds typically agree to X" and
the p90/p99 for bounding a per-pair claim.

**Gauge floor (input-orientation non-invariance)** — a property of the method,
not the build, and it does not shrink by pinning anything. Its p99 is
**4.5e-6 on the retrieved list** and 5.8e-5 over all pairs; 5.8e-4 is near the
*max* and is not a p99. These values describe the GOLD-100 library-side audit.
`gauge_floor()` has defaults for the GOLD-400 library-side retrieved-list p99;
name `gold`, `side`, `stratum`, and `statistic` explicitly when quoting a floor.
For the retrieved-list value above, use
`gauge_floor(gold="GOLD-100", side="target_side", stratum="retrieved_list", statistic="p99")`.

**The floor tells you what is *measurable*, not what is *achievable*.** The
gauge p99 of 4.5e-6 does **not** license tightening accuracy thresholds toward
it: the shipped configuration's own score error on the retrieved list is
**1.78e-3**, 400× that floor. A threshold set against either gauge number is
set against the wrong quantity.

**The physical gate runs before any statistical test.** A change whose entire
effect is below its stratum's floor cannot be claimed to improve accuracy;
`improvement_verdict()` returns `BELOW_FLOOR_NOT_AN_IMPROVEMENT` and a caller
should not run a test on it.

---

## 7. Gold's scope, enforced

Executable in `floors.assert_gold_supports_per_pair_claim`:

| stratum | status | per-pair claims |
|---|---|---|
| T ≥ 0.7 | SPECIFICATION | yes, down to the reported p99 |
| 0.55–0.7 | AGGREGATE REFERENCE | only above the reported p99 |
| 0.4–0.55 | ADVISORY | **no** |
| < 0.4 | NOT A REFERENCE | **no** |

Gold is sound for **aggregates — means, ranks, recall — at every stratum** and
unsound per-pair below 0.55. `score_error_report` refuses the per-pair claim
and still emits the aggregates, with the refusal recorded in the output.

**This bites in practice, and every study must plan for it:** 15.0% of the gold
top-1000 on a 10M-conformer panel, and 22.6% of the gold top-10 on a 480-query
tuning panel, sit below T = 0.55. A filter's recall on the bottom of the top-N′
cannot be adjudicated per-pair against gold.

---

## 8. When is an improvement real?

**Accuracy.** Paired bootstrap over queries, B = 10,000, BCa 95% CI. Real iff
the CI excludes 0 **and** the point estimate exceeds δ = 0.01. Before either,
the physical gate of §6.

**Panel size is a property of the comparison, not of the panel.** Measured
against real gold:

| comparison | sd(d) | n for δ = 0.01 | δ resolvable at n = 32 |
|---|---|---|---|
| prod-100 vs 200 steps, recall@100 | 0.0550 | 117 | 0.019 |
| prod-100 vs 200 steps, N′ = 10 | 0.0801 | 247 | 0.028 |
| prod-100 vs start_mode 2 | 0.1178 | 533 | 0.041 |
| prod-100 vs start_mode 0 | 0.2006 | 1546 | 0.070 |

A 480-query panel carries a realistic near-neighbor comparison at δ = 0.01 with
room to spare; 32 queries do not, for any comparison measured.

**Multiple comparisons.** The winner's inflation is
`(sd_d/√n)·√(2 ln K)`. Measured for the realistic pair at K = 30: **+0.037 at
n = 32** (3.7× the minimum effect size) and **+0.0095 at n = 480**. Therefore:
**never quote a TUNE number as a result** — TUNE selects a configuration, it
does not estimate its performance. Quote the single EVAL run of the selected
configuration, cap EVAL finalists at 3, and correct with Holm–Bonferroni
(`holm_bonferroni` raises above 3).

**Throughput** — the rule names its regime. Measured over two whole
invocations, one with another tenant resident on the GPU and one with the GPU
idle, kernel-only, GPU lock held, clock locked at 1402 MHz requested:

| what | relative sd |
|---|---|
| within one CUDA context, 1.7 s repeats × 11 | 0.147% / 0.221% |
| within one CUDA context, 24.9 s repeats × 5 | 0.033% / 0.065% |
| across three fresh processes | 0.55% / 0.216% |
| **the same in-process measurement, two invocations apart** | **1.6%** |

| regime | conditions | smallest resolvable ratio |
|---|---|---|
| kernel-only, **achieved** clock verified constant (min == max), GPU lock held, ≥3 warm-up + ≥11 repeats, no profiler attached | | **1.005×** (1.002× same process) |
| kernel-only, achieved clock varied over the run | | **1.01×** |
| end-to-end wall clock, or any contention | | **1.25×** (end-to-end dispersion of 16.7–26.6% was measured) |

Three measured facts behind these regimes:

* **Longer kernels are *more* reproducible, not less.** 25 s repeats are 3–7×
  tighter than 1.7 s ones. What dominates a short repeat is per-launch overhead
  variance, not thermal drift.
* **Contention barely touches kernel-only timing.** The invocation with another
  tenant at 100% GPU utilization measured *tighter* than the idle one. The
  16.7–26.6% dispersion is an end-to-end phenomenon.
* **The largest single effect is the achieved clock.** A 1.7% dip (1395 →
  1372 MHz) moved timing by 1.6% — larger than every dispersion figure above.
  This is why every run record stamps the achieved clock, and why the threshold
  doubles when the clock is not verified constant.

Never compare a kernel-only number with an end-to-end one, never compare
numbers taken at different achieved clocks, and every result table must contain
a baseline row measured in the same session on the same machine.

**Attribution is out of scope here.** This policy answers *whether* a
configuration is faster. Any claim about *why* — "X-bound", "the bottleneck is
Y" — must be made from `smsp__issue_active`, `sm__inst_issued` and the dominant
execution pipe, never from warp stall-reason shares, and causal ablation
outranks any counter argument.

---

## 9. Pre-registration and the held-out set

A pre-registration is written **before** a sweep: `{preregistration_id,
hypothesis, knobs and levels, primary metric verbatim, thresholds, panel
sha256, dataset manifests, planned n_runs, planned analysis}`. Changing the
metric is a **new id**, never an amendment — `PreRegistration.write` refuses to
overwrite a file whose primary metric differs.

`HoldoutLedger.touch` refuses to run the HOLDOUT panel twice on the same
`config_hash` under the same `preregistration_id`, and counts touches. **The
touch count must be reported in every write-up that quotes a HOLDOUT number.**

The frozen primary metric and the derivation of every threshold are in
`roshambo3.metrics.PRIMARY_METRIC_VERBATIM` and
`roshambo3.metrics.THRESHOLD_DERIVATION`.

---

## 10. Query exclusion

Exclude query compounds from retrieval evaluation when the library contains
them. Record the exclusion policy and verify that self-hits cannot inflate
reported retrieval quality. Public experiment bundles must include their own
exclusion manifest and validation.

---

## 11. Enrichment: does the ranking put actives at the top?

Everything above measures *fidelity of the port*: recall against an internal
gold, RBO against a reference ranking, score error against an exact rescore.
None of it says whether a labeled active lands near rank 1, which is the
question the comparison paper answers with EF1% and the one a reviewer asks
first. `roshambo3.metrics.enrichment` is that metric: EF at any cut,
max-pooled EF for a multi-query target, ROC-AUC, and BEDROC.

**The cut rounding is `ceil`, not `floor`.** `cut = ceil(fraction * N)`.
`floor` can round a positive fraction to a **zero-size cut** on a small
target (`floor(0.01 * 50) = 0`), turning EF into a division by zero instead
of a small number; `ceil` never does, and it matches existing practice
(RDKit's `CalcEnrichment` uses `math.ceil(numMol * f)` for the same cut).
`floor` is reachable via `rounding="floor"` -- by argument, not by editing
the module -- for exactly the case where the difference matters: a small
target, where the two rounding rules differ by one row and one row is a
whole enrichment point.

**Ties at the cut boundary are resolved by the canonical tie-break**, same
rule as everywhere else in this package (`rank_order`: descending score,
ascending canonical id). `enrichment_factor` always reports
`n_tied_at_boundary` (rows sharing the boundary's exact score -- **1 means no
tie**, only `cut <= 0` or `cut >= N` reports 0) and
`n_active_tied_at_boundary`, so a reader can see when EF is resting on the
tie-break rather than on the score. `tests/test_enrichment.py` builds this
case directly and then flips the tie-break to show the outcome flips with it.

**Conformer-level input is refused, not silently reduced.** EF's label is a
property of a *compound* ("this molecule is a known ligand"), not of one of
its conformers, so every enrichment call requires `mol_ids` and raises
`ConformerLevelInputError` if it is not unique per row: a conformer-level EF
is not a wrong number, it is an answer to a different question. The module
points at the existing `reduce_over_conformers` for the reduction rather than
doing it itself.

**Max-pooled EF is a different estimator from the mean of per-query EFs, and
they must never be averaged together.** For a target with several query
ligands, `max_pool_scores` takes each library compound's best score across
the queries -- literally `reduce_over_conformers`'s reduction applied to the
query axis instead of the conformer axis, reused rather than re-derived --
and the pooled scores are ranked once. Max-pooling inflates every compound's
score, actives and inactives alike, which is why the comparison paper reports
it as a distinct number from the per-query distribution, and why the two must
be reported separately.

**ROC-AUC uses midranks, deliberately not the canonical tie-break.** AUC is
the Mann-Whitney statistic -- the probability a random active outscores a
random inactive, crediting a tie 0.5 either way -- which is symmetric in the
tie by construction. Resolving that tie by an arbitrary canonical id (as EF
and BEDROC must, because a *cut* needs a total order) would inject a
dependency on molecule id that AUC's own definition does not have. This is
the one place in `roshambo3.metrics` where the id tie-break is not used, and
the reason is recorded in the module rather than left to be found by diffing.

**BEDROC** (Truchon & Bayly, *J. Chem. Inf. Model.* 2007, *47*, 488-508) is
reported at **alpha = 20** (the paper's general-purpose choice) and
**alpha = 321.9** (the 1%-early-recognition convention) -- both, always; the
comparison paper reports only EF1%, so AUC and BEDROC are what a claim about
ranking quality (rather than one cut) rests on. Cross-checked directly
against `rdkit.ML.Scoring.Scoring.CalcBEDROC`/`CalcAUC`/`CalcEnrichment`
across randomized trials: exact agreement to float precision
(`tests/test_enrichment.py`).

**Degeneracy is reported, not hidden, and only truly undefined ratios
refuse.** With `n_actives` actives and a cut of size `cut`, the number of
actives landing in the cut is hypergeometric and can take at most
`min(cut, n_actives, N - n_actives) + 1` distinct values -- "5 actives -> 6
possible values" is this formula at `cut >= 5`. Every EF result carries this
count and a `degenerate` flag (`DEGENERACY_MIN_DISTINCT_VALUES = 10`) rather
than a bare ratio; LIT-PCBA's small-active-count targets are expected to trip
it, and a study must report which targets did rather than drop them from a
mean. Only "no actives at all" or "a fraction that rounds to a zero-size cut"
raise `DegenerateEnrichmentError` -- there is no ratio to attach a flag to.

**Uncertainty: the replication unit is the target, not the compound** (the
over-queries rule of section 8, one level up). `replicate_summary` -- mean, sd,
min, p10, median, and a BCa CI over whatever array is handed to it -- is the
same aggregation for a between-target summary (43 DUDE-Z or 15 LIT-PCBA rows)
and for a within-target, across-query summary (LIT-PCBA's per-query EF1%
distribution), built on `roshambo3.metrics.stats.bca_ci` rather than a second
bootstrap. Never report a bare mean: `replicate_summary`'s output always
carries min and p10 beside it.

**The published ROSHAMBO2 baselines, in the right units.** The ROSHAMBO2
Supporting Information's DUDE-Z tab holds **counts of actives recovered in the
top 1%, not enrichment factors**: its panel means 6.3 and 9.1 and the +2.8
between them are counts. Converted over the population DUDE-Z ships, the
published EF1% means are **9.682** and **13.876** and the shift is **+4.194
[+2.336, +6.458]**. Any comparison against the DUDE-Z baseline uses the EF1%
values, never the counts. LIT-PCBA's tables in the same archive are genuine
enrichment factors at full float precision: 2.88 / 3.8 / 0.92 reproduce them
to three decimals (2.8803, 3.8027, 0.9225).

**How big is a reproduction's target effect, relative to its own spread?**

* The required mean shift is a **linearity identity**, not an estimate:
  mean(combination) − mean(shape) = mean(per-target delta) for any
  per-target heterogeneity, given equal *n* on both sides. **+0.92** for
  LIT-PCBA max-pooled (3.8 − 2.88); +4.194 EF1% units for DUDE-Z.
* The between-target sd decides whether the panel resolves that shift. For
  LIT-PCBA the crossover sd -- above which 15 targets stop resolving a 0.92
  shift at 95% confidence -- is **1.82**. LIT-PCBA's small crossover against
  a small shift, with its small per-target active counts inflating
  between-target spread if anything, makes it the *tighter* case, not a
  fallback.
* **A reproduction must compute its own sd_d** from its measured per-target
  rows via `replicate_summary` and `roshambo3.metrics.stats
  .required_n_for_delta`, and read where it lands, before treating a
  non-reproduction of a published shift as a finding rather than panel noise.

**EF is not invariant to the decoy set, and this module cannot make it so.**
DUDE-Z's decoys are property-*unmatched* by construction (easier); LIT-PCBA's
are experimentally-determined inactives (harder). An EF measured on one is
not comparable with an EF measured on the other, and neither is comparable
across a change in library composition -- `enrichment_factor`'s `n` and
`baseline_active_rate` fields are recorded precisely so a reader can check
whether two EFs being compared share a population before comparing them, but
no function here can detect a decoy-set mismatch from the numbers alone.

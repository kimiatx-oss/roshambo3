# Numerical and screening regression workflow

The development history of this workflow is not distributed; the paper
repository (https://github.com/kimiatx-oss/roshambo3-paper) publishes only an excerpt of the development record.

The product contract is recall and useful top-end ranking at the actual library
size and candidate budget. Reproducibility, execution correctness and fixture
qualification answer different questions. An intentional numerical change
requires recorded qualification; it never authorizes skipping safety checks or
regenerating a reference until the result becomes green.

## Obligation inventory

| Obligation | Trigger | Routine check | Qualification / interpretation |
|---|---|---|---|
| Data, units, counts, feature identity, bounds and malformed-input safety | Every change, including intentional numerics | Format/reader/query/packing tests; `assert_arithmetic_pinned`; the self-overlap bound `roshambo3.scores.check_self_overlap_bound`, which refuses any row whose pair overlap exceeds the mean of its two self-overlaps (`2 V_AB > (V_AA + V_BB)(1 + margin)`, margin `SELF_OVERLAP_BOUND_MARGIN = 1e-3`), i.e. self-overlaps that do not describe the scored coordinates (FORMAT.md section 18) | Exact invariants; recall cannot excuse a wrong scale, missing rows, identity mismatch or unsafe reader |
| Result and export row association | Every scorer/collector/export change | `test_numerics_end_to_end.py` plus canonical identity checks | Actual export-interface plants must fail; don't identify a row by its score or corpus-local gid across corpora |
| Same-build reproducibility | Every configuration that promises it | Same-worker/chunk/start/dispatch controls, stable ties | Exact under the promised axes; cross-architecture or deliberate arithmetic changes require a separately named comparison |
| Unchanged computation | Layout, scheduling, representation that promises identical values | Existing bitwise/layout/kernel equivalence tests and FIELD fingerprint | Preserve exact checks; no blanket numerical-change switch |
| Same-geometry arithmetic and gradients | Scoring, preparation, coordinate units or feature implementation | Live fp64 fixed-pose/frame/gradient and systematic-bias controls | Compare actual represented coordinates. The matched-coordinate off-grid color comparator (below) covers the integrated color path; raw different-coordinate scores cannot isolate arithmetic |
| Optimizer execution | Search changes and all screen configurations | Known self-match recovery, independent own-pose rescore, bounded screening detector | Preserve signed improvement separately from loss; displacement/start disagreement is a diagnostic, not universal screening loss |
| Optimizer control-model qualification | New backend/arithmetic/search model | Explicit `test_numerics_optimizer_real.py` campaign | Its finite perturbation ensemble has a held-out failure at the accepted step. Invoke `--optimizer-model-qualification`; limits stay unchanged and default coverage stays unqualified/not-run |
| Fixture uniqueness / basin separation | Prepared geometry, chemistry, quantization or oracle model changes | Explicit `numerics_prepared_qualification` command | A source bound that cannot transfer gives **unqualified prepared geometry**. Finite independent competing-pose search on actual geometry can requalify it; failed transfer is not observed optimizer failure |
| Bounded screening regression | Every scoring/selection/ranking change | Frozen real-query candidate-only panel, fixed **and** current cached references | N=8, retained k=16 on M=128 conformers / 80 selected molecules; challenger-enriched local detector only |
| Representation/search approximation accuracy | Intentional numerical changes | Detector plus strict independent invariants | Less frequent complete library qualification with task-derived budgets, both levels, bad strata and enrichment where labeled |
| Combined production cascade | Filter, survivor budget, candidate scorer or final scorer changes | Detector catches local faults | Evaluate complete cascade at actual retained budget against full-population references. Separate pass-level acceptances or candidate unions do not establish combined recall |
| CPU scoring/code generation | Native CPU source/dispatch/compiler/toolchain changes | Existing portable checks | Mandatory before merge: a fresh clean build of the exact candidate source, with every CPU output arm the change should leave unchanged checked as identical. The inlining-canary procedure used during development is not part of this repository. Test/docs-only work does not trigger it |

### The bounded screening detector

The detector's rules, selection recipe and loss budgets are frozen; the loss
budgets are written before any held-out evaluation. Its operating point:

- **Panel:** eight real held-out queries (the DUD-Z targets ADA, AMPC, MAPK2,
  ANDR, PARP1, RENI, EGFR, HS90A); per query, M=128 challenger-enriched
  conformers drawn from 80 selected molecules.
- **Retained budget:** k=16 per query; eight desired hit slots per query, 64
  overall.
- **Loss budgets:** at most three lost desired hit slots out of 64 overall, and
  at most two of eight on any one query. Exceeding either fails the detector.
- **References:** the candidate is compared against both an immutable fixed
  reference and the current baseline; both comparisons decide.

These are engineering alarm budgets for this operating point, not estimates of
large-library loss, physical uncertainty, or a transplant of the N=k=1000
acceptance used for complete-library qualification. A challenger-enriched
subset cannot certify whole-library recall.

## What you must provide

The routine detector reads no bulk corpus and builds no reference on its own.
Before running it, supply:

- **`$PANEL_DIR`** — a frozen query panel: one 128-row panel per evaluation
  query, with its recorded panel and geometry hashes. Freeze it before the
  change under test and never edit it afterwards. Build it once with:

  ```bash
  $PYTHON bench/screening_regression.py panel-build \
    --corpora "$CORPORA_DIR" --selection-cache "$SELECTION_CACHE" \
    --queries "$QUERY_DIR" --output "$PANEL_DIR" \
    [--target ADA --target AMPC ...]
  ```

  `$CORPORA_DIR/<target>/` is a prepared corpus with merged color records at
  the library wire step; `$SELECTION_CACHE` holds `<target>.shape.conf.npz`
  (a complete shape screen covering every corpus gid exactly once) and
  `screens.json`; `$QUERY_DIR/<target>.sdf` holds exactly one query molecule.
  Without `--target`, every target the script knows is built. Every path
  argument is required (the module also reads `ROSHAMBO3_CORPORA` at import).
- **`$REFERENCE_CACHE`** — a cache root holding `fixed/` and `current/`
  reference scores for every panel query, at the same lane, backend and scoring
  configuration as the routine run. `--action qualify` of
  `bench/screening_regression_native.py`, run once from the reference build,
  imports its fresh scores into both roles. Keep one cache root per lane
  (CPU shape, CUDA shape, CUDA combo).
- **Pre-registered loss budgets** — the budgets above for this panel, or, for a
  new library/N/k, budgets derived from the task and recorded before any
  held-out evaluation.
- **`$OUTPUT_DIR`** — a fresh, empty output directory per invocation (for
  example `runs/screening-cpu-<date>`). To assemble a receipt afterwards, use
  the layout under "Assembling a receipt" below.

## Command matrix

Run from the committed candidate checkout. `PYTHON` means a Python environment
with repository dependencies, never an editable install pointing elsewhere.
`NATIVE` means the candidate's source-matching native build directory. The
numerics runner selects this checkout's Python and validates extension source
hashes. CPU-only runs explicitly leave CUDA/Metal coverage unavailable. `$RUNS`
is any output directory you choose.

| Situation | Command / evidence | Reading rule |
|---|---|---|
| Strict public execution and export/feature plants | `$PYTHON bench/run_numerics_checks.py --native-build "$NATIVE" --report-dir "$RUNS/execution" -- -q tests/test_numerics_end_to_end.py tests/test_numerics_qualification_separation.py` | All executed assertions pass; inspect passed/failed/not-run inventory and separate fixture qualification fields |
| Prepared-fixture uniqueness prerequisite | `$PYTHON bench/numerics_prepared_qualification.py --native-build "$NATIVE" --output "$RUNS/prepared-qualification.json"` | Nonzero when any transfer is unqualified; **no global uniqueness claim** from passing execution |
| Full available arithmetic/reproducibility family | `$PYTHON bench/run_numerics_checks.py --native-build "$NATIVE" --report-dir "$RUNS/numerics"` | Read all failures and missing cells; finite optimizer-control-model failures remain explicit |
| Local screening harness mechanics | `$PYTHON bench/run_numerics_checks.py --native-build "$NATIVE" --report-dir "$RUNS/screening-mechanics" -- -q tests/test_screening_regression.py` | Synthetic tests check the detector plumbing; they do not substitute for real held-out detection receipts |
| Candidate-only CPU detector / explicit CUDA detector | `bench/screening_regression_native.py --action routine`; concrete CPU/GPU commands appear below | Cache-bound candidate scored once; both fixed/current comparisons decide; unavailable is not pass |
| Reusing an existing complete-library qualification | The qualification's own recorded evidence | Reuse only for an exact match of population, represented geometry, implementation/config and survivor budget; source matching alone does not establish Python cascade equivalence |
| New deliberate numerical/cascade change | Freeze a new protocol, then run a complete-library screen and enrichment analysis with candidate-specific provenance against both references | New task-derived loss budgets per library/N/k; configure the candidate build and data explicitly rather than reusing another run's paths |
| CPU native/codegen change | Fresh clean build of the exact candidate source; check every CPU output arm the change should leave unchanged as identical (no packaged command; the development inlining canary is not shipped) | A receipt from an earlier source revision cannot satisfy a new one |

GPU timing commands acquire canonical `GpuLock` before a CUDA context, with the
shared base `ROSHAMBO3_GPU_LOCK`. CPU deciding timings use `CpuRun`, pinning and
frequency records. Actual routine setup, screening, reporting and total wall
must fit the frozen 300-second budget. Correctness can execute unlocked; those
wall times are not timing evidence.

## Reading evidence and updating a baseline

Read identity/coverage first, then per-query and size/conformer-count strata,
top-hit landing tables, recall mean/p10/min and signed bias. Queries or targets
are bootstrap units. Selected conformers are dependent alternatives, not extra
query replicates. Positive own-pose score movement is an improvement, whereas
positive reported-score bias can inflate decoys; retain the sign and population.
A near-tie permutation is diagnostic unless a task-specific rule preregistered
exact order as deciding. Physical-noise measurements give context, not a
universal epsilon.

An unchanged-computation change keeps its existing reference and strict equality
claims. An intentional change first records its mechanism and affected paths,
then passes exact safety/identity/unit checks, independent harmful plants, clean
controls and the applicable complete-library qualification. Record any missing
backend or unqualified fixture/control model. Reference promotion needs a review
of source/config/geometry hashes, all per-query/stratum losses and both fixed and
current comparisons. Keep the immutable fixed reference, the prior baseline and
failed receipts. Promote only the current-baseline pointer in a new, explicit
manifest; routine commands must not regenerate either reference.

The accepted 2^-8 coordinate step has FIELD and ANALYTIC recall/enrichment
evidence. It has no general CPU/Metal full-library qualification and no proof of
every combined cascade. Passing strict execution checks do not erase the
existing unqualified fixture transfer or optimizer control-model failures. A
source-geometry-independent search may close fixture coverage; it is a separate
bounded qualification, not an open-ended search until a favorable basin appears.

## Routine check and explicit qualification commands

The default CPU screening check uses a small frozen panel and its caches; it
needs a source-matching CPU extension, but does not scan or require the bulk
corpus. From the candidate checkout, set `PYTHON` to the dependency environment
and `NATIVE` to that build's native directory, then run:

```bash
$PYTHON bench/screening_regression_native.py \
  --native-build "$NATIVE" --panel-dir "$PANEL_DIR" \
  --cache-root "$REFERENCE_CACHE" \
  --output "$OUTPUT_DIR" --action routine
```

Use a fresh output directory per invocation. The command never overwrites a
reference or computes a live reference. A missing panel/cache/backend returns
nonzero; `failure.json` identifies unavailable coverage. Missing or unexpected
row identities fail before recall/bias reporting, rather than fabricating
scores for absent rows. Inspect `summary.json` and `process-envelope.json`:
passing detection and fitting the observed budget are separate from a quiet,
qualified timing environment. A contended run is advisory. The standard CPU
rate ledger retains its refusal of a one-invocation rate claim (0 warmups / 1
repeat); this absolute budget measurement is not a throughput headline.

The explicit GPU lane uses the same command with `--backend cuda --lane shape`
and a CUDA-shape cache root, or `--lane combo` and a CUDA-combo cache root.
Add the source-matching CUDA directory using a second `--native-build` if
CPU/CUDA binaries live separately. Set `CUDA_MPS_PIPE_DIRECTORY` and the
canonical shared `ROSHAMBO3_GPU_LOCK` base before invoking it. GPU preflight and
the per-device lock precede context creation. `--correctness-only` permits
recorded contention and always leaves runtime advisory. `--action sensitivity`
executes the frozen faults and clean controls; its output is not a substitute
for the candidate-only routine time.

### Assembling a receipt

`bench/screening_regression_receipts.py --results "$RESULTS" --output <file>`
assembles the sensitivity and routine outputs of all three lanes into one
receipt (exit 0 only when every lane meets its frozen expectations). It reads
fixed directory names, so give each `screening_regression_native.py` run its
`--output` under `$RESULTS/evaluation/`:

| `--output` | Run |
|---|---|
| `$RESULTS/evaluation/cpu-shape-sensitivity` | CPU shape, `--action sensitivity` |
| `$RESULTS/evaluation/cuda-shape-sensitivity` | `--backend cuda --lane shape --action sensitivity` |
| `$RESULTS/evaluation/cuda-combo-sensitivity` | `--backend cuda --lane combo --action sensitivity` |
| `$RESULTS/evaluation/cuda-combo-feature-native` | `--backend cuda --lane combo --action sensitivity --scenario omit_features` |
| `$RESULTS/evaluation/cpu-shape-routine` | CPU shape, `--action routine` |
| `$RESULTS/evaluation/cuda-shape-routine-advisory` | `--backend cuda --lane shape --action routine` |
| `$RESULTS/evaluation/cuda-combo-routine-advisory` | `--backend cuda --lane combo --action routine` |

Each sensitivity directory holds `<scenario>/evaluation.json`; each routine
directory holds `summary.json`, `process-envelope.json`,
`clean/evaluation.json` and `cpu-runtime.jsonl` (CPU) or `gpu-runtime.json`
(CUDA). `--quiet-cpu-routine <dir>` adds a CPU routine run from a separate quiet
machine as an extra runtime receipt. See the script's `--help`.

Explicitly exercise the original finite optimizer control model with:

```bash
$PYTHON bench/run_numerics_checks.py --native-build "$NATIVE" \
  --report-dir "$RUNS/optimizer-model" -- \
  tests/test_numerics_optimizer_real.py --optimizer-model-qualification
```

Only the two finite-model quality tests use that explicit lane. Their original
assertions, seeds, controls and thresholds remain unchanged and still return
nonzero for the failed held-out control. Default mathematical/integrity tests,
including the live exact zero-amplitude native instrument, remain active.
Default coverage names the model unqualified/not-run; a small screening panel
never certifies optimizer equivalence.

The matched-coordinate color comparator is
`tests/test_color_field_cuda.py::TestTheFieldPathCarriesColor::test_the_field_and_analytic_color_agree_at_the_identity_pose`
(two cells: matched-grid and original off-grid); the portable helper tests are
`tests/test_color_coordinate_comparison.py`. The coarse raw off-grid difference
between differently placed coordinates measured 0.005969616. Raw scores of
different coordinates cannot isolate arithmetic; path-local fp64 checks and the
matched-coordinate arithmetic test supply the appropriate interpretation.

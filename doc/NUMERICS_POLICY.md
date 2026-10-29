# Numerics, precision and determinism policy

The paper repository (https://github.com/kimiatx-oss/roshambo3-paper) publishes an excerpt of the development record
(`tickets/`, `decisions/`); the measurements summarized here are not otherwise
distributed.

**Enforced by** `src/roshambo3/numerics.py`, `tests/test_numerics.py`,
`tests/test_engine.py` and `bench/rules.py`, and by the port gate
`bench/xcheck_roshambo2.py`. Section 3 states which checks exist.

The kernel was ported from ROSHAMBO2, and many measured numbers below were
taken on the ROSHAMBO2 kernel. Where a check is named that was a ROSHAMBO2 test
and has no counterpart in this tree, the text says so: such a rule is stated
here but is not checked automatically.

**Scope:** every measured number below was taken on **Blackwell sm_120** unless
stated otherwise. The historical campaigns retain their recorded settings;
their 100-step configurations are not the current `ScreenConfig` default of
25 steps. Scores are **not bitwise portable across architectures**. In §8's
CUDA comparison, fixed-pose scoring agreed bitwise and optimizer trajectories
diverged. Top-1 and top-5 output agreed for every measured query, while some
top-10 and top-100 memberships and orders changed. These are scoped measurements,
not a guarantee for other backends, hardware, or numerical policies. Read §8
before comparing a number taken on one GPU with a number taken on another.

This document states the policy. Wherever a rule can be checked it *is*
checked, in code, and the check is named here. A rule that appears only in this
file is a rule that a refactor will break without anyone noticing.

---

## 1. Why the arithmetic is pinned rather than given a tolerance

Measured over 463,200 real (query, target) pairs at the then-production
configuration (`start_mode=1`, 100 Adagrad steps, `lr=0.1`, shape-only):

**Bit-identical on all 20 output columns:** kernel choice (dense / sparse
blockwise / sparse threadwise), threadwise block size (32 / 64 / 96), driver PTX
JIT versus offline SASS, `ptxas -O1` versus `-O2`.

**Perturbing**, as median / p99 / max |ΔT_shape|:

| axis | median | p99 | max | basin flips |
|---|---|---|---|---|
| `--use_fast_math` | 6.0e-8 | 2.9e-5 | 1.5e-2 | 1.30% |
| `-fmad=false` | 6.0e-8 | 2.7e-5 | 2.1e-2 | 1.24% |
| `-ftz=true` | 3.0e-8 | 2.5e-5 | 1.6e-2 | 1.20% |
| `-prec-div=false` | 6.0e-8 | 2.9e-5 | 1.2e-2 | 1.27% |
| `-prec-sqrt=false` | 3.0e-8 | 2.6e-5 | 1.6e-2 | 1.22% |
| reversed reduction order | 8.9e-8 | 2.8e-5 | 1.7e-2 | 1.24% |
| int16 coordinate round trip at 0.001 Å | 1.1e-5 | 1.0e-4 | 5.6e-2 | — |
| **CUDA toolkit 13.2 → 12.9** | **0** | **8.9e-8** | **3.0e-7** | **0.002%** |

Two things follow, and they are the whole basis of this policy.

**The amplifier is saturated for anything that touches the inner loop.** All six
compiler axes land on the *same* output distribution, even though the input
perturbations they inject span orders of magnitude. There is therefore no
"small perturbation" regime to hide in: a tolerance of 1e-6 and a tolerance of
1e-4 would admit exactly the same set of builds. So the policy is **pin the
arithmetic, do not tolerate an epsilon.**

**The toolkit version is the exception, and it is a real but much weaker axis.**
It was measured by rebuilding identical source with a CUDA 12.9 toolkit unpacked
into a user prefix (`dpkg -x`, no root). The sm_120 SASS genuinely differs
between the two toolkits, but 92.95% of pairs come out bit-identical on all 20
columns, the converged pose is bit-identical for 99.998% of pairs, and the
difference that does appear is confined to the final `V_AB` reduction at ~1 ULP.
That is two to five orders of magnitude weaker than any of the six flags — 300x
on p99 |ΔT|, 5e4x on the max, 550x on the basin-flip rate — which is why a
forbidden flag is a **hard error** and a toolkit mismatch is an **advisory**. It
is also the one axis that does *not* saturate the amplifier: it perturbs only
the final `V_AB` reduction, never the gradient the optimizer follows, so the
trajectory survives it intact.

---

## 2. What is pinned

**Tier 3 is the specification.** The canonical implementation is the CUDA
kernel, `optimize_pairs_analytic` in
`src/roshambo3/backend/_cuda_src/kernels.cu`: one reduction order (source
order), one transcendental (`exp2f`, used identically in `volume_single` and
`get_gradient`), fp32 accumulation throughout. The float64 oracle in
`src/roshambo3/reference/` is useful at a declared tolerance. It is not the
specification, and reference-to-kernel agreement is not evidence of
correctness, because both can implement the same mistake.

**The reduced-overlap frame of §4 is a precondition for any scoring path.** A
host implementation may stay in volume units only while it is an oracle; if it
is ever promoted to a scoring path, it must adopt the canonical frame first, as
a declared change. The CPU backend (`src/roshambo3/backend/_cpu_src/`) is a
scoring path and evaluates in the reduced frame. A host implementation and the
CUDA kernel cannot agree bitwise — different reduction order, different pose
application, host SIMD versus GPU FMA — so mirroring a kernel rearrangement in
an oracle buys no new guarantee, while a second hand-written copy of the same
rearrangement is exactly the drift this policy exists to prevent.

**Every approximate tier's job is candidate selection, never final scoring.**

The analytic-color field-search default uses explicit PTX
`ex2.approx.ftz.f32` for optimizer Gaussian weights and the fresh SEARCH
objective after the last update. Width, amplitude and analytic gradient
coefficients are unchanged. This selective approximation is confined to pose
search: accurate winner-only shape/color scoring still uses `exp2f`, including
an explicit final-color policy override. The build-time opt-out is CUDA compiler
definition `ROSHAMBO3_COLOR_OPT_EXP_POLICY=0`; no global fast-math flag is
enabled.

Tier 0 (invariant bounds) and Tier 1 (descriptor filters) are conceptual
admission stages, not active filters in the current engine. Tier 2 grid search
selects poses and winning starts. Reported scores and corpus rankings use
analytic final scoring at the selected pose, even when approximate SEARCH
values were used to choose that pose or start. See [ARCHITECTURE.md](ARCHITECTURE.md).

**Precision is part of each backend/evaluator contract.** CUDA and Metal
accumulate overlaps and optimizer state in fp32. CPU evaluators declare their
own arithmetic; an accurate CPU final consumer may use fp64 accumulation while
consuming the same prepared FP32 pose. Summing ~900 terms in fp16 gave ~3%
error in the historical comparison, against ~2e-6 for fp32. BF16 has no
qualified scoring path here.

Field storage and interpolation precision are separate choices. Public CPU
worker screening defaults to FP32 shape interpolation; low-level CPU field
evaluators also support optional native-half Y/Z interpolation with FP32 X
interpolation and reductions, subject to ISA and range checks. CUDA and Metal
texture samplers have their own numerical policies. FP16 field storage and
these sampling approximations belong to pose search; reported scores use
accurate analytic evaluation. They do not imply FP16 final accumulation.

**Pinned CUDA toolkit: 13.2** (`roshambo3.numerics.PINNED_CUDA_TOOLKIT`). It
must be recorded with every number (§5).

**Forbidden nvcc flags** — the enumerated no-go list, with the measured effect
of each, is `roshambo3.numerics.FORBIDDEN_CUDA_FLAGS`:
`--use_fast_math`, `-fmad=false`, `-ftz=true`, `-prec-div=false`,
`-prec-sqrt=false`. `ptxas -O1`/`-O2` are explicitly *not* forbidden, because
they were measured as bit-identical; over-blocking would make the invariant
something people route around.

**Reduction order is part of the arithmetic.** Reversing the inner loop's
accumulation order moves the score as much as `--use_fast_math` does. A kernel
rewrite that reassociates the sum is a declared numerics change, not a
refactor.

**"Kernel choice is bit-identical" describes the measured CUDA analytic
launch forms, not every current kernel or backend.** The FIELD paths use
cooperating lanes and separate search/final-scoring policies. CPU and Metal
have their own arithmetic contracts. The historical analytic equivalence
carries the following qualifications.

*It is conditional on the parallelization grain.* Dense, sparse-blockwise and
sparse-threadwise agree bitwise because all three evaluate one (query, target)
pair on **one thread**, accumulating serially in source order; they differ only
in which thread gets which pair. A kernel that splits one pair's atom loop
across cooperating lanes and recombines it no longer inherits that serial
reduction guarantee. Measured on such a kernel: with W lanes per pair the
shape Tanimoto is bit-identical for only ~10% of pairs, median 4–5 ULP, p99
|ΔT| 1.1e-5, max 1.4e-2 — the same saturated-amplifier distribution the six
compiler axes reach, about 3x *smaller* than `--use_fast_math` on p99, with no
ranking change measurable on a 486,080-pair panel (top-1/5/10 sets and order
unchanged for 100% of queries; top-100 sets unchanged, order 99.875%). Small,
but a *declared* change: no grain change may be treated as a refactor, at any W.
W = 1 must reproduce the kernel it generalizes bit for bit; W > 1 needs its
own declared and checked arithmetic contract.
(These were pinned by ROSHAMBO2's `test_bitwise_reproducibility.py`, which has
no counterpart in this tree.) A reassociation inside Tier 2 still needs
search-quality and ranking qualification: accurate final scoring does not undo
a changed trajectory or winning start.

*Storage changes need their own equivalence check.* The historical transposed
32-lane float-record tile (`ROSHAMBO3_BTILE`, `roshambo3.layout.tile32`) stored
targets as `[tile][atom][lane][4]` instead of `[mol][atom][4]`. That experiment
changed strides while preserving values and per-pair accumulation order and
was **bit-identical on all 20 columns**.

Current GPU corpus inputs instead use three-component int16 wire records:
`[tile][atom][lane][3]` for interleaved tiles or
`[tile][atom][component][lane]` for planar tiles. CUDA reconstructs the physical
coordinate as `float(short) * WIRE_QUANT_STEP` before reduced-frame scaling;
weights and types come from counts. Raw float-record diagnostics remain
available. Quantization, dequantization order, and compiler contraction are
part of the equivalence obligation; the old float4 measurement does not by
itself certify every current storage or residency choice.

A change to where a value lives can preserve arithmetic; changing when it is
added changes the reduction contract. For example, a cross-product
factorization of the shape gradient loop applying the exact identity
`sum_i temp_i*(C_j x D_ij) = C_j x (sum_i temp_i*D_ij)` holds in real
arithmetic and reassociates in float32, so it is a declared numerics change by
the rule above. That hoist (`xhoist`, `_cuda_src/kernels.cuh`) is the shipped
default, together with the transposed target tile, and it is a declared
numerics change: its effect is part of every shipped CUDA analytic score, and
any further change of this kind follows §4's declaration procedure in full,
including regenerating the Tier 3 reference.

*It is conditional on the toolkit.* The equivalence **fails on Blackwell under
CUDA 12.9** (dense vs threadwise, 11 of 4800 values, max 1.2e-4) while passing
under the pinned 13.2 and on sm_70 with 12.9. It is deterministic, so it is
codegen, not noise. Everything asserted about kernel equivalence in this
document was verified under the pinned toolkit and under nothing else (§8b).

**No undeclared bitwise change to Tier 3.** A change to the canonical kernel's
numerics must be a conscious, documented decision, recorded here, never an
incidental side effect of a refactor. The declared changes to the canonical
CUDA scoring path are §4, §4c, §4d and §4e, plus the cross-product hoist above;
§4b changes reported order only, never a score. §4f and §4g record the format-3
data changes, and §4h is a declared change to the CPU path, not to the CUDA
Tier 3 kernel.

---

## 3. What is checked, and where

### Screening acceptance and obligation selection

The [obligation inventory and command matrix](SCREENING_REGRESSION.md) separates
exact safety/identity/unit checks, unchanged-computation promises, fixed-geometry
arithmetic, optimizer execution, fixture qualification and screening accuracy.
All exact invariants and compiler pinning stay in force for deliberate changes.
Screening acceptance uses task-derived recall/top-end budgets at the actual
library and retained-candidate depth, against both immutable fixed and current
baselines. The bounded challenger panel is a cheap detector; it cannot certify
full-library recall or a combined production cascade. Its protocol is in
`doc/SCREENING_REGRESSION.md`; baseline regeneration is an explicit
qualification and review action, never a routine escape hatch.

Public end-to-end assertions check every score row, physical export, identity,
feature semantics and strict self-match RMSD. Fixture basin-bound transfer is
reported separately as qualified/unqualified. A conservative source bound
failing to transfer to represented geometry is not observed optimizer loss.
`bench.numerics_prepared_qualification` fails closed when that stronger
prerequisite is requested; coverage certification retains the missing
`prepared_geometry_uniqueness` obligation. No bound or recovery threshold is
widened. Failed optimizer perturbation controls remain recorded and executable,
and do not establish a universal accuracy tolerance. The integrated
matched-coordinate off-grid color comparator is described in
`doc/SCREENING_REGRESSION.md`. The two finite-perturbation optimizer quality
tests require explicit `--optimizer-model-qualification`; their frozen
assertions still fail when invoked. Default exact mathematical, native
zero-amplitude, row/export and self-match checks remain active. Default coverage
records the model unqualified/not-run, rather than certifying optimizer
equivalence from the bounded screening panel.

Arithmetic pinning and numerical accuracy are separate obligations. Backend
entry points call `numerics.assert_arithmetic_pinned()`; header/constant
parity, canonical interaction checks, result invariants, and ranking tie-break
checks are in force. No oracle tolerance authorizes a forbidden compiler flag.

The portable oracle family computes references live rather than comparing
output-bit fixtures:

| Check | Purpose | Implementation |
|---|---|---|
| Fixed-pose row and mean residual | Detect arithmetic errors without optimizer amplification | `reference/numerics_contracts.py`, `tests/test_numerics_backends.py` |
| Directed singleton shape bias | Detect coherent prefactor bias that passes unchanged generic row bounds; no independence assumption | `bench/numerics_bias.py` and native plant receipts |
| Typed-color and canonical molecular bias | Detect coherent color coefficient bias under separately declared operation models | `bench/numerics_color_bias.py`, `bench/numerics_molecular_color_bias.py` and native plant receipts |
| Gaussian FIELD components | Bound production node construction and general 3D samples; separately detect signed bias and sampler precision loss | `reference/field_intervals.py`, `tests/test_numerics_field_intervals.py` |
| Same-frame self/cross terms | Detect query/library representation splits | Independent scored-coordinate reconstruction and fp64 self terms |
| Same-build determinism | Detect changes across promised scheduling axes | Fresh runs with varying workers/chunks and backend-specific dispatch checks |
| Optimizer controls | Distinguish acceptable arithmetic movement from degraded chosen poses | `bench/numerics_optimizer.py` and preregistered control/held-out experiments |
| Real end-to-end alignment | Recover known, empirically nondegenerate shape/color self matches through public preparation, screening and pose export | `bench/numerics_fixtures.py`, `tests/test_numerics_end_to_end.py` |
| Coverage and fault qualification | Prevent missing or unqualified checks from reading as passed | `bench/numerics_report.py`, pytest terminal/JSON/Markdown receipts |

**Practical regression delivery.** The normal gate runs the available numerics
family, including asymmetric real shape/color alignment, backend comparisons
and planted-regression controls. Read its pytest result together with the
visible passed/failed/not-run inventory. Universal certification is a separate,
stronger target, not a prerequisite to delivering these regression checks.

**Execution is not certification.** Qualification is per path and hardware. A
passing assertion does not certify an unmeasured path, a different arithmetic
model, or a detector that has not caught its required planted defect.
`CoverageReport.certified` requires complete execution and
source/panel/protocol-bound clean-control and fault evidence. Missing backends
and unqualified controls remain visible; the certification CLI exits nonzero.
Read the coverage report for actual coverage and limitations; this inventory
is not grounds for a blanket pass claim.

Fixed-pose bounds follow the declared representation and operation model. A
shared prepared-pose diagnostic cannot certify that production prepared the pose
correctly. Compare that preparation independently. A deterministic mean bound
must not assume independent rounding or divide systematic error by sqrt(N).
Sampler precision is measured independently, never inferred from a device name.

### Scoped aggregate controls and remaining qualification

The residual reporter's mean interval is implied by its row bounds. It is a
diagnostic, not an independently sensitive mean-bias gate. Arbitrarily
correlated rounding errors do not justify shrinking this interval by sqrt(N).
Fixed-pose certification therefore also requires a systematic-bias plant that
passes the row gate and fails a separately derived aggregate gate, with the
aggregate model recorded. Directed origin-singleton shape controls meet this
requirement for scalar CPU and direct CUDA analytic overlap on the executed
hardware: real native plants pass unchanged row gates and fail the directed
mean gate. Their exact-zero operation model does not qualify generic molecular
means, color, vector polynomials or fields. Separately preregistered typed-color
exact-origin controls detect both actual coefficient signs on x86, aarch64
and sm_120. Complete-molecule canonical CPU color controls additionally do so
at identity on the four attributed fixtures, conditional on the qualified glibc
2.39 expf implementation. The generic affine shape component adds explicit FP32
operation intervals for scalar and both width-eight polynomial grains, plus
preposed direct CUDA. Its actual signed smaller plants pass broad row gates and
fail the aggregate on x86, ARM scalar, sm_120 and sm_70. The exact model and
libm dependencies are recorded in `doc/numerics/GENERIC_SHAPE_MEAN_PROTOCOL.md`
and the source-bound native receipts. These controls do not qualify arbitrary
color matrices/poses, unpinned libm versions, sampled FIELD SEARCH means or
complete optimizer trajectories. The seeded CUDA Gaussian FIELD component family
adds directed node-mean intervals, conservative convex-range physical row bounds
and an independent sampler precision check. Its native constant plant passes the
broad row bounds at the same vertices and fails the focused mean bound; the
additional tighter node-row gate also rejects it. This qualifies the stated
Gaussian component panel, not arbitrary molecular FIELD objectives or SEARCH
trajectories. A constant plant caught only by a row assertion cannot discharge
an aggregate-bias obligation.

Optimizer thresholds require controls covering the comparison's arithmetic.
`reference.anchor` explicitly excludes CPU/GPU transcendental changes from its
architecture-only tolerance. A failed held-out control or surviving schedule
plant invalidates qualification; it is not permission to widen a bound.
FIELD's same-geometry order-control canary uses independent calibration and
held-out arithmetic seeds on the fixed public panel. Its geometry-transfer
protocol failed and is retained. On sm_120 it catches a learning-rate plant,
but an sm_70 held-out control fails. This experiment remains opt-in with
unchanged limits; independent known-pose FIELD shape/color tests remain default
gates. It does not derive a FIELD-versus-scalar-anchor allowance. Raw pose
displacement is a schedule proxy; independently polished analytic endpoints are
reported separately and are not a proof of sampled FIELD basin identity.
These checks do not replace task-derived recall and top-end ranking acceptance
(`doc/METRICS.md`, `doc/SCREENING_REGRESSION.md`).

The LIT-PCBA inputs carry provenance and feature semantics. Their independent
multistart qualification rejects symmetric/near-symmetric controls and measures
competing-basin separation. Finite search is evidence, not proof of global
uniqueness. A new representation must fit within the qualified score-gap budget
or requalify the case before making a uniqueness claim. Execution correctness
is reported separately and retains strict physical/row/score assertions when
that transfer cannot be established. Expected scores and poses are computed
from the physical problem, not committed kernel outputs.

**Golden bit comparisons.** The ANALYTIC golden-bit arm is retired. The FIELD
bit comparison remains until its replacement checks have demonstrated fault
sensitivity. Do not regenerate or retire it merely because the oracle family
exists. ROSHAMBO2 cross-check and engine-baseline checks retain their own
prerequisites; the coverage inventory reports when they were not executed.
Same-build bit comparisons remain appropriate for axes promised bit-identical.

Run the portable family against an explicitly identified local native build:

```bash
python -m bench.run_numerics_checks --native-build build/release --report-dir numerics-report
```

This is the ordinary available-backend regression gate. For CUDA, run it
through `bench.run_numerics_locked`, which takes the physical GPU lock for every
requested device and preflights them (refusing a busy GPU) before CUDA is
imported, then runs `bench.run_numerics_checks` with the remaining arguments:

```bash
python -m bench.run_numerics_locked --lock-receipt receipts/lock.json [--devices 0,1] \
    --native-build build/release --report-dir numerics-report
```

`--lock-receipt` must name a new file; it records the lock state, per-device
preflight, exit code and source stamp. `--devices` must include ordinal 0.
`--allow-existing-pid` (with a required `--existing-allocation-reason`)
tolerates a named existing allocation, and the receipt marks such a run
correctness-only. Keep the lock and JUnit receipts. Separately, the stronger command
`python -m bench.numerics_report numerics-report/coverage.json --output numerics-certification --certify`
refuses incomplete qualification. Compatible receipts from different hardware
can be combined; source/panel/protocol mismatches and conflicting same-cell
arithmetic declarations are refused. Default CPU CI runs its available checks
and uploads visible coverage, without treating absent CUDA or Metal runtime
coverage as passed.

`ROSHAMBO3_NUMERICS_OVERRIDE=1` is a recorded diagnostic escape hatch for
intentional arithmetic plants, not a way to certify a production build.

---

## 4. The reduced-overlap frame (declared numerics change #1)

The kernels do not work in volume units. They accumulate

```
S = sum_ij  w_i w_j exp2( -|s (r_i - r_j)|^2 ),    s = COORD_SCALE = 0.77687365
```

and convert to a physical volume once, at the point of reporting:
`V = 20.579529 * S` (shape), `V = 1.9687012 * S` (color). Three exact
rearrangements, all of which are `roshambo3/numerics.py` constants and are
pinned to the device header by
`tests/test_numerics.py::TestTheDeviceHeaderMatchesThePythonDerivation`:

1. **The prefactor cancels.** `T = V_AB/(V_AA + V_BB - V_AB)` is homogeneous of
   degree zero, so `8(π/2A)^1.5 = 20.5795` has no effect on it at all. With the
   default diagonal interaction matrix every color pair likewise shares
   `(π/2)^1.5 = 1.9687`. A non-diagonal matrix does **not** break this: the
   per-pair residual `p² r^-1.5` is folded into the accumulated term, so the
   extracted constant is always exactly `(π/2)^1.5` and the kernel remains
   exact. It is still detected at query prep and warned about, because the
   color channel is then no longer the well-conditioned single sum every
   measured number here was taken on.
2. **The exponent lives in the coordinate scale.** With coordinates
   pre-multiplied by `sqrt(α log₂e) = 0.7768`, the pair term is exactly
   `exp2f(-|Δ|²)` — no multiply at all in the innermost loop. `__expf(x)` is
   `ex2.approx(x·log₂e)` on this hardware anyway, so this makes the operation
   the machine performs explicit instead of hiding it behind a general-purpose
   `exp()`; and it gives the exponent and int16 quantization one shared frame
   instead of two that have to be reconciled. Because the scale lives in the
   coordinates, the kernel's translation is in scaled Ångström: the host
   launchers multiply `lr_t` by `COORD_SCALE` on the way in and the kernel
   divides `t` by it on the way out, so the optimizer trajectory is preserved
   rather than merely similar.
3. **Normalization happens early.** The gradient is divided by `S_AA + S_BB`
   inside the accumulation — folded into the per-term coefficient, so it is
   free — which keeps the accumulated gradient O(1) instead of O(V) whatever
   the molecule size, and removes seven multiplies per optimizer step.

Dynamic range improves in the same move and in the same direction: each
accumulated term is in `[0, 1]` rather than `[0, 20.6]`, and the gradient is
O(1) rather than O(V). Underflow behavior is unchanged — a pair at 40 Å gives
`exp2(-960)`, which is zero in both frames — and the denormal band the
`-ftz=true` row measures sits in exactly the same place. Reduced-precision work
(fp16 or int16) in this loop requires a frame with the constants already out
of it.

The translation conversion is the one place this frame could silently change
the search rather than the arithmetic. Adagrad's step size is set by `lr`, not
by the gradient magnitude, so putting `t` in scaled Ångström without
compensation would move the optimizer `1/0.7768 = 1.29x` further per step. The
host launchers multiply `lr_t` by `COORD_SCALE` to compensate. Checked in
float64: 100 steps land in the same pose to 5e-8 with the conversion, and 1e-2
away without it. (That check was a ROSHAMBO2 test with no counterpart in this
tree.)

The frame also requires `V_AA`/`V_BB` and `V_AB` — the three numbers that go
into one ratio — to be computed with one transcendental implementation
(`volume_single` and `get_gradient` both use `exp2f`), never two.

**This change is not bit-preserving; §7 measures and attributes it.** It
delivers a median of 7 ULP on the shape Tanimoto at a fixed pose, which after
100 Adagrad steps amplifies into the same p99 ~3e-5 distribution every other
perturbing axis reaches. Read §7 before comparing any score taken before this
change with any score taken after it.

## 4b. Declared numerics change #2: the ranking tie-break

Every ranking uses `roshambo3.numerics.rank_order`, which sorts descending with
ties broken on an ascending id (row index when no id is given, which makes the
sort stable). `argsort` is not stable, float32 score ties are common at
M = 1e7, and an unstable tie-break makes two runs that agree perfectly on every
score disagree on rank — manufacturing "rank changes" out of nothing and hiding
real ones in the noise. This changes reported *order* (in the tied group only),
never a score. The rule must be applied identically to whatever is called gold
and whatever is called candidate; `roshambo3.metrics.ranking.rank_g` is defined
the same way.

**Screening and cross-corpus comparison use different identity keys.**
Screening reducers break score ties by ascending global conformer ID (`gid`),
the row's position in that corpus. Molecule-level screening selects each
molecule's best conformer under that same rule, then uses the winning conformer
ID to break ties between molecules. Corpus order and IDs must remain fixed
when testing scheduling or partition invariance.

For comparisons across corpora, join rows by `(mol_id, conf_idx)` and re-rank
both sides using `metrics.canonical_ids(mol_id, conf_idx)`. A corpus-local GID
need not identify the same conformer in another build. `mol_id` alone is also
insufficient at conformer level because it repeats across a molecule's
conformers; metrics report entry points require unique row IDs (`rbo_ext`
returned 1.019 when this rule was violated). `numerics.rank_order` sorts by
the supplied ID but does not itself validate its uniqueness or choose its
identity semantics. See [FORMAT.md](FORMAT.md), section 9.2, and
[METRICS.md](METRICS.md), section 2.

## 4c. Declared numerics change #3: precomputed query self-overlap

`V_AA` (query) and `V_BB` (target) are pose-invariant and query-invariant
respectively. Both are computed once per query and per target conformer, not on
every `(pair, start_mode)` evaluation.

**`V_BB` is unaffected — measured, not assumed.** The target is read raw from
global memory and scaled by `ROSHAMBO3_COORD_SCALE` *inside* `volume_single()`
on every evaluation, whether or not it is precomputed; hoisting it out of the
loop changes *when* the multiply happens, never *how*. Over 463,200 real pairs
`V_BB` is 100.000% bit-identical at every start_mode tested.

**`V_AA` must reproduce the exact two-step computation.**
`start_mode_transform`'s first start (`n=0`) is always the identity rotation,
so it pre-scales the query into shared memory as `COORD_SCALE*mol[i]` (one
rounded multiply, stored), and `volume_single(..., cscale=1.0f)` on that buffer
is trivially exact regardless of FMA, since multiplying by 1.0 introduces no
rounding. Calling `volume_single(..., cscale=COORD_SCALE)` directly on the raw
query is a *different* floating-point expression: under this project's pinned
`-fmad=true`, `cscale*a - cscale*b` is a legal FMA fusion (single rounding of
the whole expression) rather than two independently rounded multiplies
subtracted afterwards. The two are not bit-identical: that raw-coordinate form
disagreed on `V_AA` for 17.7% of pairs (max |Δ| 2.4e-4, ~1e-7 relative).

**The shipped precompute therefore reproduces the two-step computation** —
`prescale_coords_gpu()` (the identical per-atom multiply `start_mode_transform`
performs) followed by `volume_single(..., cscale=1.0f)`. This makes
`start_mode=0` (single start, no rotation) **100.000% bit-identical on all 20
columns** over the 463,200-pair panel.

**What remains, and cannot be removed by construction:** for `start_mode∈{1,2}`
(multiple starts), a per-pair recomputation of `V_AA` would use whichever
start's *rotated* coordinates won that pair, and a nontrivial rotation (e.g.
the π rotations in `transforms_1`/`transforms_2`) is only represented to float32
precision (`PI = 3.14159265358f`), so `cos(π/2)` is not exactly 0 and the
rotated coordinates are not bit-identical to a direct scale even though the
mathematical quantity — a rigid-motion invariant — is the same. The precompute
always reports the `start_mode=0` canonical value; relative to the per-start
value, `V_AA` is 82.3% bit-identical, 94.9% ≤ 1 ULP, max 4 ULP (`start_mode=1`,
steps=0, 463,200 real pairs). This is not a defect to chase: the per-start value
was never more correct than the canonical one, and a ULP bound on an aggregate
is not a usable acceptance criterion (§7). Bound a kernel change by its effect
on the pairs a screen actually ranks.

**By that standard:** at production settings (start_mode=1, 100 Adagrad steps,
shape-only, same 463,200-pair panel), |ΔT_shape| is 0 for 89.4% of pairs, with
p99 by reference-score stratum 1.8e-7 ([0,0.4)), 2.1e-7 ([0.4,0.55)), 2.4e-7
([0.55,0.7) and [0.7,1.01)). These are below every per-stratum p99 §7 reports
for the reduced-overlap frame. Gauge comparisons require the matching
population and statistic, as described in §5. Ranking impact over the panel's
480 queries: **top-1,
top-5 and top-10 sets and order are unchanged for 100% of queries**; top-100 set
overlap averages 99.998% (99.79% of queries keep the exact set), and top-100
order survives intact for 97.3% of queries (mean exact-position agreement
99.85%) — tighter than the reduced-overlap frame's own top-100 numbers (§7:
99.987% mean overlap, 83% exact permutation).

**Verdict: a declared numerics change, smaller than either of the two that
precede it.** It is declared because §2's rule ("no undeclared bitwise change
to Tier 3") admits no exception for a small one, and the Tier 3 reference was
regenerated against it.

## 4d. Declared numerics change #4: queries on the wire grid

**What changed.** `engine.query.from_arrays` — the one funnel that `from_rdkit`
and `load_queries` reach — snaps every query coordinate, shape atoms and color
pseudo-atoms, to the library record's wire grid (step in §4e) with the format's
own operations (`engine/query.py::snap_to_wire_grid`, using
`format/constants.py::quantize`): round to the nearest int16 from the caller's
own precision (`from_rdkit` passes float64, not a float32 cast), then
`q.astype(np.float32) * step`, exactly what `prep/build.py` does to a library
conformer. No folded constant: the device's reduced-frame `COORD_SCALE` multiply
stays separate. `QuerySet.vaa` is computed from the snapped shape atoms with the
direct-difference form (`stored_self_volume`), not the norm/dot identity; no
scoring path reads it. A coordinate outside the int16 range raises instead of
clipping. Corpus queries are already on the grid and are bit-unchanged (tested
exhaustively over all 65,536 int16 values).

**Why.** The CUDA analytic kernel scores the query as an int16 record on the
grid (`r3_quantize` in `start_mode_transform`). Computing `V_AA` from the
unrounded float32 query would make the Tanimoto numerator and denominator
describe two coordinate sets for an off-grid query, and the other backends
(CUDA FIELD, CPU, Metal) would score a different coordinate set from the one the
analytic kernel scored. Snapping gives every backend one query coordinate set.
Measured at the 0.001 Å grid that preceded §4e's step: the fixed-pose mean bias
of the analytic kernel against fp64 (4 off-grid queries x 512, float32 library)
is **+3.16e-5 without the snap and +2.52e-6 with it**; it is +3.0e-8 (max row
2.3e-6) when the library is on the grid as well (int16 library). The remaining
+2.5e-6 on the float32 library is that library's int16 staging (§9). FIELD's
fixed-pose error against its own query is unchanged (+6.4e-8 / +4.0e-8).

**The movement it causes.** 16 real off-grid SDF queries against a 2,000-conformer
smoke library, shipped `ScreenConfig`, `HostTopK` k = 100, 0.001 Å grid. The
query shift is 1,319 coordinate words, max 0.0005 Å. The unsnapped arm was run
twice and is bit-identical, so every difference is the snap's.

- top-1, top-5 and top-10 sets and orders unchanged on 16/16 queries, FIELD and
  ANALYTIC, conformer and molecule level;
- every top-100 **set** unchanged (16/16, both paths, both levels);
- top-100 **order**: **14 of 16 FIELD top-100 orders moved; only 2 of 16 were
  unchanged** (conformer level 2/16, mean exact-position agreement 94.4%;
  molecule level 2/16, 95.1%). ANALYTIC: 15/16 unchanged at conformer level
  (99.8%), 16/16 at molecule level;
- FIELD's moved positions are mostly near-tie swaps, **median score gap
  2.4e-5** (p99 2.5e-3, max 6.0e-3, 89 positions), with **a few optimizer
  basin changes, max |dT| 6.5e-3**;
- |dT_shape| over shared rows: **p99 1.7e-4 (FIELD) and 8.7e-5 (ANALYTIC)**;
  median 1.7e-5 / 1.8e-5; max 6.5e-3 / 9.5e-5. Stratum T >= 0.7: p99 1.7e-4 /
  9.0e-5.

Every p99 is an order of magnitude inside the task-derived score budget
(~2e-3). The FIELD order movement comes from the field path's lattice being
planned from the snapped coordinates (`plan_field_lattice_range` sets the
lattice origin from the query's min/max, so a <= 5e-4 Å snap shifts every
texel), amplified by the optimizer (§8). It is not a change in fixed-pose
arithmetic. It is acceptable under the task-derived budget, but it is **far
more top-100 order movement than §4c** (top-100 order intact on 97.3% of
queries), which is why the numbers are kept here in full.

**Limits.**

- **Rotating starts.** `start_mode = 1` ships (`engine/config.py`), and
  `start_mode_transform` re-quantizes the **rotated** query on the device
  (`r3_quantize` of the rotated coordinates in `_cuda_src/kernels.cu`), while
  `V_AA` is the **unrotated** snapped query's. The shipped starts are signed
  permutations, which preserve the snapped integers, so cross and self terms use
  the same coordinate set (§4e). A start set containing other rotations would
  need a separate representation check.
- **Raw `score_pairs` callers bypass the funnel.** `tests/test_golden.py` and
  `bench/make_golden.py` (the golden fixture) call `backend.score_pairs`
  directly on their own arrays, never `from_arrays`, so their queries are not
  snapped by it. On the CUDA analytic path the query self term is still
  rebuilt on the wire grid (`_cuda_src/interface.cpp` passes
  `prescale_coords_gpu(..., query_wire_grid = !field)`), matching the kernel's
  `r3_quantize`d cross term. CUDA FIELD and CPU score the unsnapped
  coordinates. Raw off-grid results are therefore not comparable across
  backends; snap the query (`snap_to_wire_grid`) before comparing.
- **Metal is unverified.** Metal receives the snapped array like every other
  backend. The claim that its `V_AA` and `V_AB` are then computed from that one
  coordinate set rests on a read-only audit of `backend/metal.py` and
  `kernels.metal`, not on a compiled run.

**Verdict: a declared numerics change for every off-grid (SMILES/SDF) query.**
No committed fixture moves: the golden fixture bypasses the funnel, and the
engine baseline builds its queries from corpus coordinates, on which the snap
is the identity. The same-frame self/cross check (§3) turns the frame
consistency this entry relies on into a standing check.

---

## 4e. The coordinate wire step is 2^-8 Angstrom

The builder, query snapping and all backends use one grid,
**2^-8 A = 0.00390625 A**. The CUDA reciprocal pair is **256.0f /
0.00390625f**, shared by library dequantization and `r3_quantize`; there is no
query/library split. Int16 covers **[-128.0, +127.99609375] A**. Dequantization
at this power-of-two step is exact, and every inline CUDA site retains the
separate reduced-frame multiplication `cscale * (INV_Q * (float)v)` (§4c).
`assert_arithmetic_pinned()` refuses disagreement among Python's wire step
(`numerics.WIRE_QUANT_STEP`), `format/constants.py::COORD_SCALE`, both format-3
`STEP_LOG2` constants (`format/r3g/limits.py`, `format/r3g/reference.py`) and
the compiled CUDA step/reciprocal. Metal substitutes the Python step into its
shader template. Format-2 records declaring 0.001 Å are refused rather than
reinterpreted.

The shipped signed-permutation starts preserve the snapped query integers, so
cross and self terms use the same coordinate set. A start set with other
rotations would require a separate representation check. The single-GPU
analytic fixed-pose bounds of the oracle family use start_mode=0; the rotating
analytic screens below exercise start_mode=1.

The step was accepted on FIELD/ANALYTIC screening recall and enrichment. No
arithmetic flag, test tolerance or optimizer threshold was changed for it.
Checks that were failing at acceptance are noted below; the step is not
universally certified.

**Acceptance rule.** Comparisons are 0.001 Å (format 2) against 2^-8 (format 2
at 2^-8 with the device at 2^-8), on three readings. R1 is top-end agreement
with the 0.001 Å arm: recall@1000 at conformer and molecule level, paired
against a grid-phase control, and RBO_ext; it must pass its existing primary
acceptance criteria. R2 is enrichment non-inferiority: the paired BCa lower bound must be
greater than -1.0 EF1% / -0.01 BEDROC20 / -0.01 AUC, for shape and combo. R3 is
paired screening recall: point >= -0.005, lower bound > -0.010 and mean RBO
>= 0.950.
A failed control criterion is reported as non-discriminating. No threshold was
changed for this step.

**FIELD results:**

- R1 recall@1000: **0.9979; 0.9979** (conformer; molecule), paired versus
  grid-phase control **-0.0009 [-0.0012, -0.0006]**, RBO_ext .98 **0.9968 /
  0.978** (mean/min). R1 passed all criteria.
- R2 shape EF1%, BEDROC20, AUC: **16.892, 0.3692, 0.7684**; combo:
  **22.168, 0.4527, 0.7994**. All six non-inferiority cells passed.
- R3 shape paired recall **-0.0028 [-0.0062, -0.0016]**, RBO mean **0.969**;
  combo **-0.0013 [-0.0022, -0.0007]**, RBO mean **0.974**. Both passed;
  no control criterion was non-discriminating.

**ANALYTIC results** (rotating `start_mode=1`, `pose_search=ANALYTIC`, every arm
under the per-device GPU lock taken before context creation, both binary
identities recorded, repeated arms byte-identical): R2 and R3 passed. Shape R3
paired recall **-0.0030**, CI **[-0.00642, -0.00175]**, RBO mean **0.9694**;
combo **-0.00075**, CI **[-0.00158, -0.00017]**, RBO mean **0.9749**. All six
R2 cells discriminated and passed. The FIELD-derived config is refused by
analytic dispatch preflight; the supported analytic config uses
`color_stage=0`, `color_residency=GLOBAL`, otherwise the same shape/combo
settings. No timing claim is made. Analytic R1 was not run; no R1 data enter
these comparisons.

**Fixed-pose diagnostics.** The CUDA analytic fixed-pose row/frame check passes
at both steps. At 0.001, shape mean residual **1.950e-08**, max **1.840e-07**;
color mean **1.140e-08**, max **4.807e-08**. At 2^-8, shape mean
**2.640e-09**, max **1.758e-07**; color mean **4.740e-09**, max **5.014e-08**.
These are row-implied mean diagnostics, not independently qualified
aggregate-bias bounds. When the step was accepted, the practical family's
empirical basin-qualification assertions, two held-out optimizer checks and the
off-grid FIELD/ANALYTIC color comparison were failing; the step was accepted on
screening recall and enrichment without changing those gates, and without
claiming universal certification. Basin transfer is now reported as
qualified/unqualified (§3), and the off-grid color cell has its own contract
(`tests/test_color_field_cuda.py`).

**Golden fingerprint movement.** Both stored arms are generated at this step;
FIELD remains the bit comparison and ANALYTIC remains retired from it. Against
the 0.001 Å fixture, FIELD changed **20451/40960** words; shape mean signed
movement **-2.923e-06**, p99 absolute movement **1.759e-03**, max
**1.602e-02**. The exhaustive numpy/C/CUDA round trip has zero mismatches for
all 65,536 int16 values. The color fixture and derived FIELD fixture are
generated at this step; their supported CUDA cross-check's worst tolerance
fraction is **0.0252**, with tolerance values unchanged.

**Sites carrying the constant.** `numerics.WIRE_QUANT_STEP` and
`DEVICE_CONSTANTS`; `_cuda_src/kernels.cuh` stage macros, `r3_fetch*` and
`r3_quantize`; every inline macro use in `_cuda_src/kernels.cu`
(`start_mode_transform`, library staging and inline dequantization);
`_cuda_src/interface.cpp` compiled bindings; `backend/cpu.py` dequantization and
stored-self metadata; `_metal_src/kernels.metal` via
`backend/metal.py::_wire_kernel_source`; `format/constants.py` (`COORD_SCALE`,
`quantize`) and the format-3 reader (`format/r3g/`); the color self contract
(`COLOR_SELF_CONTRACT`); `prep/build.py` quantization and self contracts;
`prep/conformer.py` stored coordinates; `engine/query.py::snap_to_wire_grid`;
`engine/buffers.py::require_wire_quantization` and its reader/source callers;
`cli.py` reader step checks; `reference/numerics_contracts.py::CoordinateGrid`;
`reference/field_quality.py::staged`; oracle-family staging in
`bench/numerics_{backends,optimizer,optimizer_real,cuda_winners,cuda_field_winners,end_to_end}.py`;
and Metal benchmark adapters. Both format-3 `STEP_LOG2` copies are -8 and
participate in arithmetic pinning.

---

## 4f. Content-defined atom order and rank ties

Format 3 was compared with the 2^-8 format-2 rung using identical queries and
canonical `(mol_id, conf_idx)` joins. A matched-current-kernel control
reproduced all prior panel shape scores and all twelve-target shape/combo score
columns exactly. Shape scores are not identical across the format switch:
9,294,448 of a 9,600,000-entry panel matrix's entries differ (the matrix
includes 64 excluded self entries); maximum absolute difference is
0.061725914478302. Panel recall@1000 against the 2^-8 rung is mean 0.99990625
and minimum 0.999. On the twelve DUD-Z targets, 19,472 of 2,382,468 shape
scores differ, maximum 0.013398826122283936, with recall@1000 equal to 1.0 on
every target. Those shape changes are already present in a canonical-heavy
preparation control; centroid derivation and canonical within-type feature
order each change zero shape-score words on this panel. Holding the 2^-8 scores
fixed and changing only the id tie keys moves 4,773 DUD-Z shape rank positions
and 7,898 panel rank positions; these are isolated tie effects, not the total
measured score/rank movement.

## 4g. Derived centroids and canonical feature order

Checked on 2,382,468 DUD-Z conformers from 39,710 molecules. Canonical-stored,
enumeration-derived and format-3 arms have identical heavy int16 coordinates;
enumeration-derived and format-3 pseudo-atoms are identical after exact
feature-tuple alignment. Original-source versus hydrogen-completed feature
membership agrees on all 39,710 molecules. Every one of 34,028,088 derived
single-atom features equals its own stored heavy atom exactly. Against
independently rounded stored pseudo-atoms, 34,026,967 single-atom features are
exact and 1,121 differ by one quantum; 3,806,376 multi-atom centroids are exact
and 1,499,899 differ by one quantum. No feature differs by more than one quantum
under minimum-cost within-type Chebyshev matching. The centroid contrast has
combo recall@1000 mean 0.99925 and minimum 0.994; the within-type-order contrast
has recall@1000 1.0 on every target. Full 2^-8-rung-to-format-3 combo
recall@1000 is mean 0.99925 and minimum 0.994. The §4e R1, R2 and R3 rules all
pass for the full format-3 candidate; no threshold was relaxed.

## 4h. CPU FIELD partial batches use batched update arithmetic

The AVX2 `field_batched` configuration uses `optimize_pairs_field_blocked` for
every same-query run of one through eight pairs, including block tails, worker
boundaries and query transitions; partial runs do not switch to
`optimize_pair_field_with` and its scalar pose update. The batched update has
explicit SIMD FMA expressions; the scalar update has a different
contraction/expression schedule, so moving an identical pair across that
dispatch boundary would change its optimizer trajectory.

This is a **declared CPU numerical change**, not a refactor justified by a small
score difference. It chooses the complete-batch arithmetic for partial batches.
Starts, steps, learning rates, Gaussian coefficients, reduction order, final
analytic rescore and compiler flags are unchanged. Inactive SIMD state is
initialized by the optimizer; absent target/result rows are never read or
emitted. Explicitly unbatched configurations retain their own arithmetic. NEON
has no corresponding eight-pair update dispatch and retains its per-pair
production evaluator. Bit identity across CPU architectures is not promised.

It was verified with a baseline-sensitive portable regression, exact same-build
score/pose/collector/ranking comparisons, independent fp64 final-pose checks
and the available numerical-oracle family with its missing-coverage inventory.
Scheduling invariance is separate from before/after score movement. On a real
shape panel, two tail-batch scores change, and the resulting complete top-100
equals the result those pairs produce inside a full batch.

---

## 5. What a run must record

Every run record carries `roshambo3.numerics.numerics_stamp()` — the
`numerics` block of `bench.run_record.RunRecord`. It contains:

* `pinned_cuda_toolkit` and the toolkit this build actually used;
* `cuda_flags`, `cxx_flags`, `cuda_architectures`, `build_type`, `git_sha`,
  straight from the build stamp;
* `cuda_src_sha256`, `cuda_src_tree`, `cuda_src_dirty` and `source_stamp_phase`
  — **which source was compiled**, as opposed to which commit was checked out.
  `git_sha` names an index, and A/B arms are routinely built from a tree that
  departs from its index; `cuda_src_sha256` is a hash of the bytes and moves for
  an uncommitted edit where the other two do not. `cuda_src_dirty` is `null`
  when the check did not run, which is not `false`. A `decision`-grade number
  from a dirty build is refused by `bench/rules.py`;
* the reduced-frame constants the kernel computed with;
* `arithmetic_ok`, plus any violations and advisories;
* `numerics_override`, so a deliberately-perturbed build cannot pass itself off
  as a normal one;
* `cross_architecture_reproducibility` — a measurement, not a disclaimer:
  `roshambo3.numerics.CROSS_ARCH_REPRODUCIBILITY` (§8).

A number quoted without that block is not reproducible. Two numbers may only be
compared bitwise if their stamps agree on toolkit, flags and architecture — and
§8 is why "and architecture" is in that sentence.

**The stamp only exists if the run goes through the harness.** A deciding
measurement written straight to a bare results JSON — no `status`, no
`gpu_lock_held`, no `advisory_reasons`, no `numerics` — cannot demonstrate that
it was clean, and cannot demonstrate that it was not; no enforcement inside the
harness reaches a run that never calls it. So: **route a deciding measurement
through `build_run_record()` and append it to a run-record JSONL.** Exploratory
scripts may do as they like; a number a decision rests on may not. Enforce,
don't advise: otherwise the failure is silent, and only visible much later, when
someone asks whether the GPU was contended and nothing on disk can answer.

Two distinct quantities must **not** be conflated:

* **arithmetic reproducibility at fixed input** — the subject of this document;
* **the gauge floor**, the score's lack of invariance under a rigid rotation
  of the *input* molecules. In the GOLD-100 library-side audit, p99 |ΔT| is
  approximately **4.5e-6 on the retrieved list** (top 10 per query) and
  **5.8e-5 over all pairs**. The previously quoted 5.8e-4 is near the retrieved
  list's maximum, not its p99. This is a property of the method and does not
  shrink by pinning the build. `metrics.floors.GAUGE_FLOOR` records the side,
  step count, population and statistic. Select all four explicitly when
  quoting a value: `gauge_floor` defaults to GOLD-400, not this GOLD-100 audit.
  Compare matching populations and statistics, rather than treating any one
  of these numbers as a universal epsilon.

---

## 6. Measured results behind this policy

| measurement | stated in |
|---|---|
| what perturbs the arithmetic (8 axes) | §1 |
| CUDA toolkit 13.2 vs 12.9 | §1, §7 |
| reduced-overlap frame, fixed pose, in ULPs, and at 100 steps | §7 |
| bitwise reproducibility across kernels/blocks/JIT | §1, §2 |
| precomputed query self-overlap (V_AA), fixed pose and 100 steps, ranking | §4c |
| architecture: Volta sm_70 vs Blackwell sm_120 | §8 |
| GPU count: 1 vs 2 cards, 463,200 pairs | §8 |

The underlying measurement records are not distributed with this repository.

## 7. The reduced-overlap frame, measured — and why ≤ 1 ULP is unattainable

A ≤ 1 ULP criterion for a pure rearrangement of the overlap sum is
unattainable, and is not an acceptance criterion here. The reason is
structural, not a defect in the rearrangement: an overlap is a sum of up
to ~2000 float32 terms, and scaling every term by a constant *before*
summing rather than after necessarily lands the sum a few ULP away,
because the two computations round independently at each of those ~2000
additions. The same holds for computing an identical exponent by a
different sequence of multiplies. **Only the identity transformation can
be ≤ 1 ULP on a reduction this long** — so a ULP bound on an aggregate
is not a usable acceptance criterion for any kernel change here. Bound a
kernel change the way the second half of this section does instead: by
its effect on the pairs a screen actually ranks.

**At a fixed pose** (`steps=0`, so the score is a pure function of the
scoring arithmetic with no optimizer amplification in the loop), over
463,200 pairs, on the shape Tanimoto, volume units versus the reduced frame:

| comparison | bit-identical | ≤ 1 ULP | median ULP | p99 ULP | max ULP |
|---|---|---|---|---|---|
| **full change**, start_mode 0 (identity pose) | 4.12% | 12.18% | 7 | 33 | 81 |
| **full change**, start_mode 1 (best of 4 starts) | 3.99% | 11.85% | 7 | 33 | 81 |
| ↳ ingredient 1: unify the transcendental (`exp()`→`exp2f` in volume_single) | 72.31% | 85.19% | 0 | 3 | 8 |
| ↳ ingredient 2: prefactor cancellation + early normalization | 4.14% | 12.36% | 7 | 32 | 79 |
| ↳ ingredient 3: fold the exponent into the coordinate scale | 36.44% | 64.56% | 1 | 5 | 14 |

The attribution is the useful part. The coordinate fold — the ingredient
that looks most alarming, because it re-quantizes every input coordinate
— costs a median of 1 ULP. Unifying the transcendental costs a median of
0. Essentially all of the movement comes from **changing what the inner
loop accumulates**, which is exactly the operation this codebase cannot
do for free, and the same reason a reversed reduction order is as damaging
as `--use_fast_math` (§1).

### What it does to the pairs a screen ranks

The aggregate above is the wrong number to decide on. This method's intrinsic
uncertainty lives almost entirely in dissimilar pairs, so any aggregate over
all 463,200 pairs is dominated by pairs no hit list ever contains. Stratified
by the *reference* build's own score, at the production 100 steps,
|ΔT_shape|:

| stratum | n | median | p99 | max |
|---|---|---|---|---|
| T_ref in [0.0,0.4) | 209,231 | 2.1e-07 | 5.6e-05 | 1.2e-02 |
| T_ref in [0.4,0.55) | 201,307 | 3.0e-07 | 2.5e-05 | 1.7e-02 |
| T_ref in [0.55,0.7) | 47,390 | 3.6e-07 | 1.7e-06 | 1.2e-02 |
| T_ref in [0.7,1.01) | 5,272 | 4.2e-07 | 1.6e-06 | 5.7e-05 |

| the list a screen returns | n | median | p99 | max |
|---|---|---|---|---|
| top-1 targets per query | 480 | 4.2e-07 | 2.1e-06 | 2.2e-04 |
| top-5 targets per query | 2,400 | 4.2e-07 | 1.9e-06 | 1.2e-03 |
| top-10 targets per query | 4,800 | 4.2e-07 | 2.0e-06 | 3.3e-03 |
| top-100 targets per query | 48,000 | 3.6e-07 | 3.1e-06 | 6.7e-03 |

And the ranking itself, over 480 queries: the **top-1 target is unchanged for
100% of queries**, the top-5 set *and order* are identical for 100%, top-10 set
overlap averages 0.99958 (identical for 99.6% of queries, worst case 90%), and
top-100 set overlap averages 0.99987. Within the top 100 the *order* moves more
than the *membership* does (83% of queries keep the exact permutation), which is
what a last-bit change to a near-tied list looks like.

The tabulated retrieved-list p99 effects are a few times 1e-6, while the
seven-build ensemble's maximum on the top-scoring stratum was 8.9e-7. The
aggregate p99 is 3.2e-5. Retrieved lists, score strata and all-pair populations
are different comparisons; §5 records the gauge statistics for their stated
populations, and `METRICS.md` defines how floors enter an accuracy claim.

For comparison, the CUDA toolkit change leaves the top-1, top-10 *and* top-100
sets identical for every one of 480 queries (mean overlap 1.0000); it never
reorders a hit list at all.

**Verdict: this is a declared numerics change, not a hygiene refactor.**
It is recorded as such (§4), the Tier 3 reference was regenerated
against it, and any accuracy number taken against the volume-unit
arithmetic must be re-taken. What the frame buys is real — one
transcendental instead of two, no prefactor and no exponent coefficient
in the innermost loop, an O(1) gradient, and a frame that reduced
precision and int16 storage can share — but it is not free.

**At the production 100 steps**, the same comparison gives |ΔT_shape| median
2.7e-07, p99 3.2e-05, max 1.7e-02, with a 1.39% basin-flip rate. That is the
optimizer amplifier acting on a last-bit difference, not a property of this
change: it is the same distribution every other perturbing axis reaches. Do not
read the ULP table above as a promise that scores did not move.

**CUDA toolkit 13.2 vs 12.9**, identical source and flags, over the same panel:
92.95% of pairs bit-identical on all 20 columns; |ΔT_shape| median 0.0e+00, p99
8.9e-08, max 3.0e-07; 10 of 463200 pairs land in a different basin.
Conditional on a pair differing at all, the converged quaternion is still
bit-identical 99.97% of the time — the toolkit change moves the final V_AB
reduction, not the gradient the optimizer follows.

---

## 8. Architecture, measured — scores are not portable, hit lists are

The question this answers: can a distributed screen shard across heterogeneous
hardware and merge top-N lists? It was measured on **TITAN V (sm_70, Volta)**
against Blackwell sm_120 as a clean factorial with the toolkit axis of §1:

| | CUDA 12.9 | CUDA 13.2 |
|---|---|---|
| Blackwell sm_120 | toolkit row (§1) | baseline |
| **Volta sm_70** | **this section** | impossible: CUDA 13 dropped sm_70 |

Everything except the GPU was pinned, and each pin was *verified* rather than
assumed:

* same source, same flags, same nvcc **12.9.86** on both machines;
* the same architecture list on both — `70;80;86;89;90;120` + `120-virtual`, one
  fat binary, so architecture is chosen by the driver at module load and the
  compile command is character-identical on the two machines;
* `cuobjdump -sass` is **byte-identical between the two machines** for sm_80 and
  sm_120 — the architectures a CUDA 13.x `cuobjdump` can still decode. The cubin
  *files* differ only in the source path `-lineinfo` embeds, and both machines
  run the same gcc 13.3.0;
* the float32 coordinate arrays handed to the kernel are **bit-identical** on
  both machines (sha256) — which matters, because the host prep path is float64
  LAPACK and the two machines run different NumPy builds on different Python
  versions;
* the same panel as §1 and §7: 480 queries × 965 targets = **463,200 pairs**,
  `start_mode=1`, 100 Adagrad steps, `lr=0.1`, shape-only, so the numbers are
  directly comparable to §7's.

### The result, in the order that matters

**At a fixed pose the scoring arithmetic is exactly portable.** With `steps=0`,
all 20 columns are bit-identical between sm_70 and sm_120 for **100.000%** of
463,200 pairs, at `start_mode=0` and at `start_mode=1` alike. Even in the
100-step runs, `V_AA` and `V_BB` — the two self-overlaps, which are fixed-pose
reductions — never differ by a bit.

**The optimizer is not portable.** At the production 100 steps:

| | Volta↔Blackwell | CUDA 13.2↔12.9 | `--use_fast_math` |
|---|---|---|---|
| pairs bit-identical, all 20 columns | **0.12%** | 92.95% | — |
| \|ΔT_shape\| median | 3.0e-08 | 0 | 6.0e-08 |
| \|ΔT_shape\| p99 | **2.6e-05** | 8.9e-08 | 2.9e-05 |
| \|ΔT_shape\| max | 8.9e-03 | 3.0e-07 | 1.5e-02 |
| basin flips | **1.22%** | 0.002% | 1.30% |

Architecture lands on the *saturated* output distribution — statistically
indistinguishable from `--use_fast_math`, and ~300× the toolkit axis on p99.
**A change of GPU feeds the optimizer amplifier.**

The growth curve shows how:

| Adagrad steps | 0 | 1 | 2 | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|
| pairs bit-identical | 100% | 87.8% | 40.4% | 12.9% | 2.2% | 0.12% | 0.09% | 0.12% |
| \|ΔT\| p99 | 0 | 8.9e-08 | 1.2e-07 | 1.8e-07 | 2.1e-07 | 3.0e-07 | 1.1e-06 | 2.6e-05 |
| basin flips | 0 | 0 | 0 | 0 | 0 | 0 | 0.21% | 1.22% |

One step injects one ULP — the same 8.9e-08 measured for the entire toolkit
axis — and a hundred steps multiply it by ~300.

**And it is codegen, not hardware.** Compiling the primitives this kernel is
built from with the same nvcc for sm_70 and sm_120 and comparing raw bit
patterns: `exp2f` (the kernel's only transcendental, i.e.
`ex2.approx.ftz.f32`), `expf`, an FMA-contractable `dx*dx + dy*dy`, and a
4096-term in-source-order fp32 accumulation of `exp2f(-|Δ|²)` are **all
bit-identical** on the two architectures. The hardware computes the same bits
for the same instruction; what differs is which instructions ptxas emits for the
gradient expression under each target's register budget. That is a fixable class
of difference in principle (explicit `__fmaf_rn`-style intrinsics in
`get_gradient` would remove ptxas's freedom), not an unfixable SFU difference.

### What a screen actually sees

Stratified on the reference (Blackwell) build's own score, the cliff at
T ≈ 0.55 seen in §7 reproduces exactly:

| stratum | n | median | p99 | max |
|---|---|---|---|---|
| T_ref in [0.0,0.4) | 209,231 | 3.0e-08 | 4.9e-05 | 7.5e-03 |
| T_ref in [0.4,0.55) | 201,308 | 6.0e-08 | 2.0e-05 | 8.9e-03 |
| T_ref in [0.55,0.7) | 47,389 | 6.0e-08 | 9.0e-07 | 5.2e-03 |
| **T_ref in [0.7,1.01)** | **5,272** | **0** | **3.6e-07** | **3.2e-05** |

| the list a screen returns | n | median | p99 | max |
|---|---|---|---|---|
| top-1 targets per query | 480 | 6.0e-08 | 4.3e-07 | 8.6e-05 |
| top-5 targets per query | 2,400 | 6.0e-08 | 4.8e-07 | 5.1e-04 |
| top-10 targets per query | 4,800 | 6.0e-08 | 6.6e-07 | 2.5e-03 |
| top-100 targets per query | 48,000 | 6.0e-08 | 2.2e-06 | 4.8e-03 |

**The ranking, over 480 queries:** the top-1 target is unchanged for **100%** of
queries; the top-5 set *and order* are identical for **100%**; top-10 set
overlap averages 0.99979 and is identical for 479 of 480 queries; top-100 set
overlap averages 0.99985 (exact set for 98.5% of queries, exact permutation for
83%). Seven top-100 slots move across the whole panel, out of 48,000.

That is the same order as the reduced-overlap frame's own effect (§7) — a
declared numerics change this project accepted and ships — and better than it
on top-10. **Architecture can change close ranks and some list membership.**
The measured top-1/top-5 agreement does not establish exact top-K invariance
at every depth, or for other backends and configurations.

### The heterogeneous-shard merge

Scoring half the corpus on Blackwell and half on Volta, then merging and
re-ranking: the merged **top-1, top-5 and top-10 reproduce the homogeneous
Blackwell list exactly, in set and in order, for all 480 queries**; the merged
top-100 set is exact for 99.2% of queries (mean overlap 0.99992), and the mean
Tanimoto lost per slot by using the merged list instead of the homogeneous one
is **0 at top-10 and at most 1.5e-05 at top-100**. This mean loss per slot is
a different statistic from the per-pair p99 gauge measurements in §5; it must
not be compared to a single population-independent "gauge floor."

### GPU count is bitwise irrelevant

On two TITAN Vs, `n_gpus=1` and `n_gpus=2` agree on **all 20 columns for
100.000%** of the same 463,200 pairs, on the sparse/blockwise path, and on the
dense path as well. Sharding across cards is free.

### The claim, and its limits

> Per-pair shape Tanimoto agrees between sm_70 and sm_120 to within **2.6e-05 at
> p99 over all pairs** and **3.6e-07 at p99 on the top-scoring (T ≥ 0.7)
> stratum**, with the top-1 and top-5 ranked output changing for **0 of 480**
> queries, top-10 for **1 of 480**, and top-100 for **7 of 480**. At a fixed
> pose the agreement is exact. Across GPU *counts* the agreement is exact.

Three limits, stated so nobody over-reads it:

1. **sm_70 is outside the sm_80+ deploy target**, and Volta is architecturally
   further from Blackwell than Ampere is. So this **bounds** the effect rather
   than predicting Ampere↔Blackwell — and the direction of that bound is the
   useful one: *no reordering across a gap this wide implies portability across
   any supported deployment target.*
2. Shape-only, `start_mode=1`, 100 Adagrad steps. The color channel and other
   optimizer settings are unmeasured, and §4's warning about non-diagonal
   interaction matrices applies there independently.
3. Bitwise comparison across architectures is **forbidden** as an acceptance
   criterion. A stored reference is architecture-gated by construction, and
   anyone who diffs a Blackwell score against an sm_70 one must
   compare *rankings*, not bits.

The measured kernel contained no `__CUDA_ARCH__` guard, no warp intrinsic and
no architecture-gated path, so nothing about the build had to change for Volta.
A fat binary whose only embedded PTX is `compute_120` cannot be JIT-compiled on
an sm_70 device, so the PTX-JIT reproducibility axis is unreachable there.

### 8b. "Kernel choice is bit-identical" is conditional on the pinned toolkit

The kernel-choice equivalence is not an architecture effect and not
nondeterminism (threadwise reproduces itself bitwise), but it depends on the
toolkit:

| build | dense vs blockwise | dense vs threadwise |
|---|---|---|
| **CUDA 13.2, sm_120 (pinned)** | bit-identical | bit-identical |
| CUDA 12.9, sm_120 | bit-identical | **11 of 4800 values differ, cols 0/1/3, max 1.2e-4** |
| CUDA 12.9, sm_70 | bit-identical | bit-identical |

§1 lists kernel choice among the axes measured bit-identical, and a change to
the launch configuration may be made freely on that basis **only under the
pinned CUDA 13.2 toolkit**. Tuning the threadwise grain depends on it most.

Every score in §8 was taken with `method="blockwise"` on both machines, and
dense-vs-blockwise is bit-identical in all three cells above, so §8 is
unaffected.


## 9. Two more changes measured bit-identical

Both belong to §2's "a change to *where* a value lives is free" family, and both
were **verified rather than assumed**, because each could in principle have
perturbed the arithmetic and one of them changes what `ptxas` may contract.

**The 16-byte query load (`qvec`).** Reading the shared-memory query atom as one
`float4` instead of three 32-bit `LDS` loads the same four values in the same
order and combines them with the same arithmetic; only the load width changes.
**Bit-identical on all 20 columns** over 486,080 real pairs. It is a measurement
arm only — it is 1.4% *slower*.

**Dropping `*wa*wb` from the shape term (`nw`).** The shape loops are bounded by
`molA_num_atoms` / `molB_num_atoms` — the **real** atom counts, never the padded
`L` — and `prepare` writes exactly `1.0f` for every real atom and `0.0f` only
for padding, while `start_mode_transform` copies slot 3 through **unscaled**
because it is not a coordinate. So every weight the shape loop can reach is
exactly `1.0f`, and `x*1.0f` is exact in IEEE-754. **Bit-identical on all 20
columns**, on and off the corpus's own quantization grid. The verification is
not ceremonial: removing a multiply changes which expressions `ptxas` may fuse
into an `FFMA`, and that *would* have moved the last bit.

**A dataset fact that looks like a bug and is not.** When a corpus stores
coordinates as int16 at a step, any kernel-side re-quantization at the same step
is an **exact round trip on that corpus**, and int16 shared-memory staging
comes out bit-identical for that reason alone — not because int16 is lossless.
Measured at the 0.001 Å step: off that grid, staging costs p99 |ΔT| **5.74e-5**
(2.9% of the task-derived budget, close to an independently measured 7.4e-5),
with top-1, top-5, top-10 and top-100 *sets* unchanged for 100% of queries.
**A test that asserts int16 staging is bit-identical is asserting a property of
the dataset, not of the kernel**, and would break on a corpus stored at any
other step. Perturbing the corpus off its own grid is what shows the int16
path is live.

**One declared change was measured and rejected.** A distance cutoff on the
shape term (drop pairs with `exp2f(-d2)` below 1e-3, i.e. d > 4.06 Å) is a
declared numerics change by §2's rule — it removes terms from the sum.
Measured: p99 |ΔT| **9.67e-4**, 48% of the task budget, with top-100 *order*
preserved for only **78.4%** of slots — and it is **1.47× slower**, because the
branch costs more than the transcendental it skips. It is not shipped and
should not be revisited.

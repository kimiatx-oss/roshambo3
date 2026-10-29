# The CPU color contract

Executable half: `src/roshambo3/reference/color.py`,
`tests/test_color_contract.py`, and the frozen, stamped fixture
`tests/data/color_contract_fixture.npz` (section 9). It constrains every CPU
color implementation: the color seam, scalar rescore, scalar combo, and every
optimized CPU color arm. 
The paper repository (https://github.com/kimiatx-oss/roshambo3-paper) publishes an excerpt of the development record
(`tickets/`, `decisions/`); the measurements summarized here are not otherwise
distributed.

This document says what CPU color *means*, independently of any
implementation. The CUDA analytic path is a **cross-check, never the oracle**
(C++ and CUDA can agree because they share the same typo); everything below
derives from the published fixed-pose Grant-Pickup color expression and from
central differences of the float64 value, and CUDA supplies only conventions —
which molecule the pose applies to, how types are numbered, which matrix axis
is the query's.

Wherever a rule here can be checked it *is* checked, in
`tests/test_color_contract.py`, and the check is named. Four deliberately
wrong conventions each fail a test that names them:

| wrong convention | failing test |
|---|---|
| swapped type numbering | `test_swapped_type_numbering_fails_the_fixture` |
| wrong pose direction | `test_wrong_pose_direction_fails_the_fixture` |
| omitted self cross-terms | `test_omitted_self_cross_terms_fail_the_fixture` |
| the legacy combination formula | `test_the_legacy_combination_formula_fails_the_fixture` |

---

## 1. The physical formula

The color overlap of query features `a_i` (types `ta_i`) against library
features `b_j` (types `tb_j`), in physical Angstrom:

```
V_color = (pi/2)^1.5 * sum_{i,j : r[ta_i, tb_j] != 0}
          p[ta_i, tb_j]^2 * r[ta_i, tb_j]^-1.5
          * exp( -(r[ta_i, tb_j] / 2) * |a_i - b_j|^2 )
```

* `r = interaction_r` is the pair **width**, `p = interaction_p` the pair
  **height**, both indexed **`[query_type, library_type]`** — the same
  `rmat[ta*N_features + tb]` orientation the device uses, pinned by the
  fixture's deliberately asymmetric-matrix case.
* **Zero width means the two types do not interact**: the pair is *skipped*,
  because `r^-1.5` at `r = 0` divides by zero. It does not mean a
  zero-width Gaussian. A zero **height** with nonzero width is evaluated and
  contributes exactly zero. These are different statements; both are
  contract, and `numerics.check_interaction_matrix` flags the incoherent
  combination (`r = 0, p != 0`).
* In the reduced frame (`doc/NUMERICS_POLICY.md` §4) the identical quantity is
  `exp2(-r * INV_TWO_ALPHA * d'^2)` on `COORD_SCALE`-premultiplied
  coordinates, with `(pi/2)^1.5 = COLOR_PREFACTOR` applied once at reporting
  and the per-pair residual `p^2 r^-1.5` kept inside the accumulated term.
  The frame is an implementation choice; this section is the meaning.

**Canonical versus general matrix.** The canonical matrix — width 1 and
height 1 on the feature diagonal, height 0 off it — is the one every
`PharmacophoreGenerator` default emits and the only configuration in which
every term shares the single prefactor. A fast path may specialize it (the
same-type segmented loop), but the general matrix — off-diagonal
interactions, unequal widths — is handled **exactly or not at all**: an
implementation that cannot handle a general matrix must refuse it, never
silently substitute canonical behavior. The fixture's non-canonical case is
mandatory precisely because an all-ones width matrix hides a wrong width
index and a diagonal height matrix hides an off-diagonal indexing error.

## 2. Type numbering and layout

* **Type 0 is a normal atom and is never a color feature.** Feature types run
  `1 .. N_features-1` in the shipped enumeration —
  `pharmacophore.WORK_DESCENDING_FEATURE_ORDER` (Hydrophobe = 1,
  Acceptor = 2, Aromatic = 3, Donor = 4, PosIonizable = 5, NegIonizable = 6).
  Numeric equality without an enumeration stamp is not semantic equality;
  fixtures record the enumeration by name.
* The interaction matrices are `[N_features, N_features]` **including** row
  and column 0; row/column 0 is never scored.
* A conformer's record is real atoms, then color features **grouped in one
  contiguous ascending run per type**, then padding; the count vector's
  column 0 is `n_real` and column `t` the count of type `t`.
  Query inputs carry explicit types: `layout.color_view_from_types` derives
  their counts through `counts_from_types`. Library inputs already carry the
  count vector: `layout.color_view_from_counts` derives feature segments from
  it, without constructing or decoding a per-atom library type plane.

## 3. The self-overlap is ordered

`S_color_AA` is the color overlap of a molecule with itself: the **full
ordered double sum**, `i == j` terms included, both `(i, j)` and `(j, i)`.
Two coincident identical features therefore contribute **four** times one
feature's self term, not twice. A triangular sum with a doubled off-diagonal
computes the same real number and is *not* automatically bit-identical
(reduction order differs); a sum that omits the `i != j`
cross-terms entirely is simply wrong, and fails
`test_omitted_self_cross_terms_fail_the_fixture` by a factor of ~2 on the
coincident-features fixture case.

## 4. The pose

The pose applies to the **library** molecule, shape and color coordinates
alike: `x -> R(q) x + t`, quaternion `(w, x, y, z)` normalized before use,
translation reported in Angstrom (scores columns 9–15). The query never
moves. Applying the pose to the query, or applying `R(q)^T`, produces a
different score at any nontrivial pose and fails
`test_wrong_pose_direction_fails_the_fixture`.

Rotation derivatives are compared in **axis-angle** coordinates: the
derivative along a space-frame axis-angle increment applied on top of the
current rotation (`reference.color.pose_gradient_numerical`). A central
difference in raw quaternion components is a different quantity — off by the
pose-dependent Jacobian of the chart — and must not be compared against a
rotation generator.

## 5. The nine score columns

`roshambo3.scores` columns 0–8, on any color-scoring run:

| # | name | contract |
|---|---|---|
| 0 | `tanimoto_combination` | **`(1 - mixing) * T_shape + mixing * T_color`.** Never the legacy `T_shape + T_color` (that sum-out-of-two survives only in `scores.tanimoto_combo_legacy`, for published comparisons). |
| 1 | `tanimoto_shape` | `V_AB / (V_AA + V_BB - V_AB)`, guarded (§6) |
| 2 | `tanimoto_color` | same form on the color volumes, guarded (§6) |
| 3 | `overlap_volume` | `V_AB`, physical (shape prefactor applied) |
| 4 | `overlap_color` | `V_color_AB`, physical (color prefactor applied) |
| 5 | `self_overlap_query` | `V_AA` |
| 6 | `self_overlap_library` | `V_BB` |
| 7 | `self_color_query` | `S_color_AA` |
| 8 | `self_color_library` | `S_color_BB` |

`mixing` is in `[0, 1]`. `mixing = 0` makes column 0 the shape Tanimoto
exactly — which is also why **no test at `mixing = 0` can distinguish a color
implementation from a shape-only one**; every discriminating test runs at
nonzero mixing.

## 6. Empty color, and the guard

Every Tanimoto denominator is guarded: `den <= 0` reports `0.0`. In exact
arithmetic `den <= 0` on a channel means both sides are empty on it, so the
guard's semantic content is one rule: **empty color on either or both sides
gives finite zero color scores (columns 2, 4, and the empty side's self
column) and leaves the shape columns untouched — no denominator may produce
NaN or Inf.** `0.0` and not `1.0`, deliberately: two empty planes reported as
a perfect match would rank every featureless pair above every real one.
Column 0 for an empty-color pair at nonzero mixing is therefore
`(1 - mixing) * T_shape` — the mixed objective genuinely discounts a
featureless molecule, and that is the formula's answer, not a defect.

Every implementation, CPU and CUDA, guards the color divide; an unguarded
divide reports NaN color and NaN combination for an empty–empty pair and
violates this contract. Both CUDA pose-search paths (field and analytic) are
checked by `test_empty_empty_color_tanimoto_is_zero_on_both_kernels`.

## 7. The three modes, named for the CPU

CUDA spells modes 2 and 3 `ColorPlacement.AT_POSE`/`IN_GRADIENT` — kernel
placement vocabulary. The public CPU names, `reference.color.COLOR_MODES`:

* **`shape_only`** (mode 1). Pose optimized on shape; color not scored.
  Color columns are `0.0`; column 0 **equals** `tanimoto_shape`; `mixing`
  must be 0 and a nonzero value is refused (otherwise column 0 would
  silently become `(1-mixing) * T_shape`).
* **`shape_pose_color_rescore`** (mode 2). The pose search is *exactly* the
  shape-only one — same trajectory, same start set, per-start winner selected
  on the **shape** Tanimoto, ties to the earliest start in the canonical
  start order. Color is evaluated **exactly once per reported row**, at the
  converged pose, by the exact analytic sum (never a field approximation or
  a differently ordered value left over from a search); every color column is
  populated and column 0 is the mixed score. Defining property, and the
  acceptance bar for every mode-2 implementation: **pose, winning start, and shape score are bit-identical to
  a shape-only run on the same inputs.** An implementation may evaluate
  color per start or winner-only; the reported row must be identical either
  way.
* **`combo_optimal`** (mode 3). The optimizer's objective *is* column 0: the
  gradient is the derivative of `(1-mixing)*T_shape + mixing*T_color`, each
  channel weighted by its own `dT/dV = (V_AA+V_BB)/(V_AA+V_BB-V_AB)^2`
  before mixing (adding raw overlap gradients optimizes neither reported
  Tanimoto). Start selection
  and the final winner are on the **mixed** score, ties to the earliest
  start. The reported row is an exact evaluation of both channels at the
  converged pose. An empty-color pair in this mode has the same *optimal*
  pose as shape-only mathematically, but not bit-identically — the trajectory
  differs; bit-identity with shape-only is a mode-2 property only.

Discriminating the three requires `mixing > 0`; the fixture carries two
mode cases and they exercise different halves of the contract:

* `mode_discrimination` has **byte-identical shapes** on both sides: the
  shape optimum is the identity at exactly 1.0, so it exercises the combo
  objective trading against a *perfect* shape match, and it does **not**
  exercise the shape search — an implementation that returns the identity
  pose without searching satisfies its mode-2 rows, and the convergence gate
  is inert on its two shape-posed modes.
* `mode_discrimination_distinct_shapes` has genuinely distinct shapes: the
  shape-only optimum is **not** the identity and **not** 1.0, mode 3's pose,
  shape score and mixed score all differ from mode 2's by margins far above
  every tolerance in §8, and the convergence gate carries weight on every
  mode. **This is the case that carries mode 2's acceptance bar** — the
  bit-identity property a mode-2 implementation must reproduce is only
  meaningful where the shape search has something to find.

## 8. Comparison rules, and the derived tolerances

**The numerics policy pins arithmetic within one implementation; comparison
across implementations is on ranked output at a declared tolerance.**

1. **Within one CPU implementation, on one platform**: a change that claims
   not to move arithmetic is verified **bit-identically** on all 20 columns
   (the `xcheck`/golden discipline, one level down). A cached self-color
   against a recomputed one, a segmented loop against a masked one —
   same rule.
2. **Fixed-pose values against this contract's fixture**: compared at the
   **derived fixed-pose tolerance**, `reference.color.fixed_pose_tolerance`.
   It is a first-principles forward-error bound on a float32 evaluation of
   the case's own sums (per-term transcendental + argument rounding +
   coefficient rounding + worst-case serial accumulation, Tanimoto
   propagated via `|dT| <= 2 T (1+T) rho`, declared 4x headroom;
   `FIXED_POSE_TOLERANCE_DERIVATION` spells it out). **It is not the shape
   anchor's tolerance**: that is the measured end-to-end cross-architecture
   |dT| after 100 optimizer steps, which bounds a different comparison. And
   it is fixed before any candidate is compared — a tolerance chosen after
   seeing a candidate is not a tolerance.
3. **Reproducing the float64 fixture itself** (another platform, another
   libm): `FLOAT64_REPRODUCTION_RTOL = 1e-11`, derived in
   `reference/color.py`. Bit-exact on the generating platform, reported as
   such but not required.
4. **Across ISA widths, field interpolation, CPU/GPU, or libm**: optimized
   *searches* are judged on **ranked output** under `doc/NUMERICS_POLICY.md`
   §8 and `doc/METRICS.md` §8 (paired bootstrap, BCa CI, delta = 0.01), with
   `numerics.rank_order`'s canonical tie-break on both sides. Bitwise
   cross-implementation comparison is forbidden: scores are not portable
   across architectures, hit lists are.

## 9. What the fixture is

`tests/data/color_contract_fixture.npz`: real color conformers (from the
RDKit pharmacophore generator on the test SDF, shipped enumeration) plus the
adversarial cases — empty planes both sides, one feature, repeated coincident
features, a missing query type, a zero-width interaction cell, the canonical
same-type matrix, a non-canonical matrix with off-diagonal interactions and
unequal widths, an asymmetric matrix that pins the index orientation, and the
mode-discrimination pair. Every case carries its inputs (float32, exactly
what an implementation is handed), the pre-posed library coordinates
(float64 pose application, rounded once), the nine float64 columns at that
pose, and the numerical color and mixed-objective gradients at two step
sizes. The record stamps the generating source hashes, git state, platform,
libm, NumPy/SciPy versions, feature enumeration, and SDF hash — the same
identity discipline as the shape anchor (`reference/anchor.py`), because a fixture that cannot
name what produced it cannot be read when a re-run disagrees.

It is generated by `reference/color.py` and `reference/overlap.py` **only**
— neither optimized backend is its oracle. The CUDA analytic path is then
required to agree with it at the §8.2 tolerance for every case it supports
(`TestTheCudaAnalyticPathIsCrossChecked`), which is a cross-check of the port,
not evidence about the mathematics.

**Every input coordinate in the fixture is a wire-grid value**,
`float32(k) * WIRE_QUANT_STEP`, at the 2^-8 A wire step.
The CUDA analytic path stages shape and color query coordinates as int16
planes (`start_mode_transform`, `r3_quantize`). This is the identity for
on-grid inputs with the shipped signed-permutation starts. General off-grid
inputs may move by up to half the wire step (0.001953125 A per coordinate),
which is a representation change rather than arithmetic error. The fixture
therefore puts input coordinates on the grid so cross-checks compare arithmetic.
Posed library coordinates stay off-grid deliberately: color launches consume
the float32 library record without int16 staging.

Public query preparation snaps queries to this same wire grid before either
FIELD or ANALYTIC screening (`engine/query.py`). Raw backend calls bypass
that preparation, so CPU-versus-CUDA analytic comparisons on off-grid queries
must account for CUDA query staging. CPU float32 input arithmetic itself does
not quantize. The analytic staging mechanism is pinned by
`test_the_analytic_path_consumes_the_query_at_the_wire_grid`.

## 10. Shared FP32 reduced-pose contract

The `shared_fp32_reduced_v1` CPU combo contract retains the physical
Grant-Pickup overlap in §1 but changes the arithmetic frame and pose
construction. After start composition, the optimizer quaternion is normalized
once in FP32 and one FP32 rigid transform is prepared:

```
s = COORD_SCALE
M = s * R_fp32(unit_quaternion_fp32)
t' = s * t_physical
A' = s * A
B' = M * B + t'
d' = d / (s*s)
c' = c
```

The query color features `A'` are owned once during query preparation. Library
features `B` remain in physical Angstrom. The vector kernel, scalar analytic
fallback, every per-start exact score, and the distinct winner rescore consume
the same prepared `M` and `t'` bytes; an exact reference may use FP64
accumulation, but it must not independently renormalize the quaternion or
reconstruct another pose. The coefficient/amplitude is unchanged; signed decay
alone is converted. Translation descent is directly with respect to reduced
`t'`, without a second `1/s`; torque preserves the existing sign/factor using
the same pose and reduced displacement.

This is a declared numerical change: FP32 pose and reduced-frame
rounding may alter fixed-pose values and optimizer trajectories compared with
the separately labeled legacy physical-space FP64 pose. The latter remains
available as an explicit comparator and the independent physical-space
high-precision formula remains an oracle; neither is a production fallback
for the shared contract. The shared pose is the ordinary CPU combo contract
for supported color grains and field policies, not a speed-based toggle.
Within the same build and shared contract, plan and exact-scratch ON/OFF
ablations hold pose, frame, evaluator, and arithmetic fixed and require
bit-identical outputs and diagnostics. Cross-contract or cross-implementation
comparisons use this document's declared fixed-pose tolerance and the
ranked-output rules of §8.4, not bitwise trajectory identity.

### Production CPU color grain build set

The CPU build supports scalar grain 0 and the reviewed shared-pose analytic
conformer/start grains 11 and 16. A request for any other grain id fails at
the Python and native boundaries with an unsupported-grain error. Build
metadata (`cpu.color_build_stamp`) reports each supported grain's target
compilation and runtime ISA availability. This set does not involve shape
backends or GPU code. It is a bounded production choice based on measured arm
results, not a claim that other color arms cannot be useful on other hardware
or workloads.

### Final-start selection policy

Prepared canonical four-start CPU screening selects the actual converged
SEARCH objective: field shape plus analytic SEARCH color under the requested
optimizer exponential, normalized with unchanged self values. Stable ties keep
the earliest start. Accurate current FP32 analytic sums evaluate only that
winner for reporting/top-K; there is no extra optimizer update. Shape-only and
shape-pose-color-rescore share the same shape-only criterion.
``CPUWorkerOptions(final_start_policy="all_start")``, the corresponding Engine
worker keyword, or ``cpu.score_pairs(final_start_policy="all_start")`` retain
accurate all-start selection. Unsupported/unsafe combo routes retain that
comparator. Policy changes do not invalidate query geometry or a persistent
Source's self handles; fresh facade Sources still invalidate library handles.

This is a declared selection change, concordant with GPU policy; it measured
held-out conformer K100 aggregate 0.9925 (three lost hits) and meaningful
molecule K41 = 1.0 on all eight queries. Accurate reporting at the fixed
selected pose is unchanged by the selection policy.
The independent fixed-pose final-rescore arithmetic variants
(`final_rescore.hpp`; `R3_FINAL_RESCORE_MODE`, default 0) are disabled by
default; when compiled explicitly they
operate only on the all-start control path.

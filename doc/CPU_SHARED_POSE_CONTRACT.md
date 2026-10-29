# Shared FP32 reduced-frame pose for CPU shape and color

**Status:** Production `shared_fp32_reduced_v1`, with the trusted construction
and consumer preconditions below. The legacy physical-space color path remains
available under its own `legacy_physical_fp64_pose` label. 
The paper repository (https://github.com/kimiatx-oss/roshambo3-paper) publishes an excerpt of the development record
(`tickets/`, `decisions/`); the measurements summarized here are not otherwise
distributed.

## Pose and coordinate frames

For each trajectory lane at each optimizer iteration, compose the current
state with its selected start first. Normalize that composed quaternion once
in FP32, construct the rotation once in FP32, and prepare one rigid transform:

```
s  = COORD_SCALE
M  = s * R_fp32(unit_quaternion_fp32)
t' = s * t_physical
```

The prepared object carries `M` and `t'` as FP32 reduced-frame values. Shape,
analytic color, scalar exact fallback, per-start exact scoring, and final-winner
exact scoring consume those same prepared matrix and translation bytes. An
individual consumer may use FP64 accumulation, but it may not reconstruct or
renormalize a quaternion to make another pose. The optimizer retains its
existing update coordinates and learning-rate units; `t'` is the value already
held by its reduced-frame state. Physical translation is recovered only for
the public output record.

Shape queries are prepared as `A' = s*A`; physical library atoms remain
physical, so the shape evaluator uses `M*B + t'`. Field grid mapping and stored
shape gradients remain evaluator-specific. They do not create a second rigid
pose.

Color query features are converted once during owned query preparation to
`A' = s*A`, preserving feature order, type segments, masks, and real counts.
Library color features remain physical and each color consumer uses
`M*B + t'`. Shape atoms and color features may be distinct point sets; they
share only the prepared rigid transform.

## Gaussian parameters and derivatives

For the physical interaction `c * exp(d * |A-B|^2)` with signed decay `d`,
the reduced-frame color interaction uses

```
d' = d / (s*s)
c' = c
```

The coefficient/amplitude remains unchanged. Canonical and general matrices
retain their existing type order, weights, zero-width skips, inactive entries,
and ordered self-overlap. The owned query preparation retains both the input
matrix identity and its reduced decay data; a prepared handle with a different
query, shape config, matrix, or frame is rejected.

Color translation descent is evaluated directly in reduced coordinates. With
`delta' = A' - (M*B+t')`, each pair contributes
`2*d'*value*delta'` to the optimizer's negative-gradient convention. There is
no physical translation round-trip or extra `1/s`. Half-angle torque uses the
same reduced displacement and the rotated library arm `M*B`; `d'` cancels the
two coordinate scale factors in the cross product. Existing sign, factor, and
space-frame conventions are preserved.

## Scope and numerical declaration

Ordinary mode-3 combo calls default to `shared_fp32_reduced_v1`, including
scalar color grain 0, vector analytic color grain 11, and standard four-start
color grain 16. These are the color grains the CPU build supports; any other
grain id is refused. Supported shape evaluators include scalar
analytic, analytic SIMD, scalar field, paired-channel field, and pipelined
field. Trajectory shape remains limited to its documented matching analytic
SIMD configurations. Exact scalar fallbacks consume the same prepared pose and
reduced query/matrix data.

Experimental color-field policies are also supported with the shared contract
on retained grains 11 and 16. `color_field_backends` selects analytic or field
evaluation per library feature type, optionally per query;
`prepare_color_fields` supplies a reusable query-scoped handle. Served planes
approximate SEARCH color only; reported scores remain accurate analytic
evaluations at the selected shared pose. Color fields are not the CPU default,
and a nonanalytic color-field request with scalar grain 0 is refused. The
Python and native boundaries check the supported grain, shape evaluator, start,
and field-policy combination. See
[the CPU scoring interface](../src/roshambo3/backend/cpu.py) and
[prepared-field checks](../tests/test_cpu_static_color_field_policy.py).

This is a declared numerical change to pose construction and color
coordinate/exponent rounding (`doc/COLOR_CONTRACT.md` section 10). It does not
claim bitwise trajectory identity with the legacy
physical-space FP64 color pose. The independent physical-space high-precision
oracle remains separately labeled and validates the declared numerical change;
it is not a production fallback. Within the new contract, construction count,
shared matrix/translation identity, reduced Gaussian values, finite-difference
gradients, exact per-start/final scoring, masks/tails, stable tie behavior, and
concurrent immutable query reuse are permanent checks.

## Benchmark protocol

A performance comparison of this contract against the legacy pose is a
bounded, matched, one-core advisory campaign: identical library/query
populations, four starts, 25 optimizer steps, mixing 0.5, both color mappings,
pipelined-field and trajectory-analytic shape, and representative
small/general-color cells. Query color preparation and shape field
construction occur before alignment timing and are reported separately.
Paired repeats, exact worktree/build/source/ISA/population stamps, the CPU
lock and affinity rules (`bench/cpu_run.py`), and a fresh clean build of the
exact source being timed apply. A number taken under host contention is advisory and
records the visible host/context limitations. No speedup is required for the
correctness contract to pass.

## Trusted internal construction and consumers

`PreparedReducedPose` has no contract tag. Its internal caller precondition is
a complete finite matrix and translation whose bytes remain immutable during
evaluation. Active physical library and reduced query coordinates must be
finite at their input boundary; finiteness alone does not imply a supported
analytic distance envelope or safe field address. `prepare_reduced_pose_fp32`
is the checked construction boundary:
it rejects null inputs, nonfinite quaternion/translation components, and an
FP32 squared norm whose square root is zero or nonfinite. These checks do not
alter the construction arithmetic or normalization order. Raw replay APIs check all incoming pose
bytes once; direct C++ users must honor the documented precondition. A public
struct member or a constant marker does not prove validated construction.

The constructor's positive finite sum of nonnegative squared components bounds
each normalized component even when a squared component is subnormal: rounding
can perturb normalization but cannot produce an unbounded quotient. Its small
finite products therefore produce finite matrix entries. This is a finiteness
guarantee, not a claim of exact unit length for underflow-scale quaternions.
Zero/underflowed and overflowed norms fail at construction. Optimizer Adagrad
updates and quaternion composition retain their existing order; every new
composed pose is checked before any evaluator receives it. An exceptional
update is rejected there, including after the last optimizer step, through the
existing worker exception propagation. At final start selection, a nonfinite
objective fails through that same error channel before comparison; the final
work-unit publication boundary rejects nonfinite score/reporting-pose columns
before a row reaches its sink. The complete call fails rather than silently
discarding a NaN hit or selecting a default start through unordered comparisons.
Valid scores keep strict greater-than/earliest-start tie behavior. There is no
deferred error protocol.

Because construction is checked, specialized canonical consumers do not
rescan matrix finiteness and exact-final consumers do not rescan the matrix or
translation. Pose equality
compares actual matrix and translation bytes. Query identity, pointer ownership,
array dimensions, feature eligibility, and field conversion/domain/address
checks retain their independent purpose. Translation range and transformed
coordinate envelope checks remain: finite construction alone does not
establish their supported arithmetic range. Analytic envelope elision is
justified only by the proof below and cannot be inferred from the absence of
a contract tag.

### Analytic envelope proof

The analytic envelope path (`R3_COLOR_ANALYTIC_ENVELOPE=1`, the default on
x86-64 and AArch64, each selected by its own bounded native campaign) records three
maximum absolute physical library coordinates during immutable descriptor
preparation, using sign-cleared integer-bit maxima so descriptor preparation
cannot lose subnormal inputs under DAZ/FZ before reuse in a different worker
mode. Once per supplied pose it computes each absolute matrix-row dot
that bound, plus absolute translation. An inflated bound
`upper * (1 + 16*FLT_EPSILON) + 1 <= maximum_reduced` proves the existing
per-atom envelope. The relative margin dominates rounding in the positive
bound and the three-term FP32/FMA transform; the additive unit dominates the
at most a few subnormal rounding errors. More explicitly, with
`v=FLT_EPSILON`, `eta=2^-126`, define effective inputs as the operands after
the evaluating thread's input-denormal zeroing (if enabled). Absolute effective
atom coordinates are bounded by effective descriptor maxima, regardless of
the mode used during descriptor preparation. Define `B` from these effective
matrix/maxima/translation inputs, and `b` as the computed positive sum.
An operation can additionally lose subnormal intermediate inputs and outputs;
allowing up to three `eta` per operation, six rounded bound operations give
`B <= (b+32*eta)/(1-v)^6`. Four transform operations give
`abs(y) <= (1+v)^4*B+32*eta`; their combined relative factor is
less than `1+11*v`. The rounded inflation has slope at least
`(1-v)^2*(1+16*v) > 1+13*v`, and its additive term exceeds 0.99, dominating
the less than `65*eta` absolute error, including zeroing of subnormal `b`.
These conservative constants cover standard directed rounding and input/output
denormal zeroing (DAZ/FTZ or FZ). The floating-point mode is stable during each
evaluation; descriptor preparation may use a different mode. No hot-path FP
control reads or changes are required.
Overflow or unordered comparisons
fail the proof. SIMD implementations apply the same proof to four/eight
poses using the already loaded matrix vectors. Only a proven group bypasses
the transformed-coordinate comparisons; a failed proof takes the unchanged
per-atom checked transform. Translation envelope checks remain. No query,
exp, force, torque, reduction, or field-address arithmetic changes. Actual
matrix bytes are used, so finite nonunit/raw replay matrices remain supported.
Manually assembled descriptors must describe the actual immutable inputs;
fixtures mutating raw coordinates must refresh their bounds.

Field safety is independent of this analytic proof. In x86 strict prepared
sampling, hardware-defined CVTT conversion can precede the ordered domain mask;
invalid offsets are replaced with the zero sentinel before loads, and invalid
fractions/levers are cleared. This is not an unchecked C++ float-to-int cast.
Scalar casts retain their prerequisite range checks. The ALLIN proof, tail
masks, sentinel allocation and safe offset constraints are independent of the
analytic envelope proof, which adds no ALLIN dispatch and removes no field
address check.

## Worker-scoped bound execution and final SEARCH values

The shipped policy is the same on x86-64 and AArch64: bound outer-core
execution (`R3_BOUND_EXECUTION=2`) and value-only final shape values
(`R3_FINAL_SHAPE_VALUE_ONLY=1`); color final values stay off
(`R3_SEARCH_VALUE_ONLY=0`, so the final color SEARCH pass evaluates value and
descent and discards the descent). One common algorithmic policy is used
because ARM measured aggregate parity under it. Normal-ISA builds pass exact
quality/mode gates and active field/color form checks. Numerical SEARCH
values, pose construction and optimizer/winner policies are unchanged by
binding or value-only evaluation.

`BoundCanonicalColorExecution` belongs to one decoded work unit. Its checked
construction binds the exact immutable query-plan identity, complete singleton/
pair compatibility signature, sized scratch, ISA availability, exponential
policy and torque factor. Descriptors, query snapshot, decoded coordinate
buffers and scratch outlive the view; none may be mutated during its use.
The view is discarded before another decoded work unit reuses worker storage.
It is never retained by a query cache. Raw batch/pose APIs retain their checked
entry boundaries; internal pose arrays are constructed by the shared checked
constructor for every step and final selection.
Bound outer-core execution (`R3_BOUND_EXECUTION=2`) also resolves one
concrete SEARCH exponential/torque policy instantiation at binding, using a
single callable per complete evaluator call. There is no callback in an
atom/texel loop. `R3_BOUND_EXECUTION=1` (bound, generic evaluate) retains
runtime numerical-policy dispatch as an isolated comparator. This binds the outer core policy; actual
linked ISA-helper wrappers can still perform their existing per-call dispatch,
so it is not a claim that every numerical-policy branch was eliminated.

`BoundShapeExecution` similarly binds one immutable evaluator/target/workspace
association. The field adapter selects its concrete scalar, paired-channel or
pipelined numerical arm once. Binding checks source pointer/count association,
required SoA object, matching atom count and supported capacity. Each evaluation
installs the actual prepared matrix/translation aliases internally. The worker
owns the query handle, target and workspace for the whole view lifetime; a new
library buffer, mutated count/configuration, or workspace rebuild requires new
binding. No producer-to-worker coordinate copy, per-iteration allocation,
per-atom callback, field containment assumption or FP-control toggle is added.
General analytic adapters retain their existing checked numerical evaluator.

Descriptor preparation additionally checks nonnegative real/color counts and
contiguous bounded segment offsets against the actual decoded color extent.
It reuses the existing integer-magnitude physical-coordinate maxima and full
plane signatures. These refer to decoded FP32 coordinates (the corpus reader's
int16 times FP32 physical-unit scale), rather than prequantized preparation
coordinates. There is no radius/AABB pass, persisted metadata or corpus
migration for this purpose.

There is no trajectory-wide analytic proof for unrestricted configured
trajectories. Existing Adagrad admits finite learning rates large
enough for a first finite translation update far beyond the analytic envelope:
for a finite ordinary gradient, the ratio to its square-root cache is near one,
so a translation rate of order `1e30` produces a finite translation of that
order. Checked quaternion/translation construction does not reject such a
finite translation. Cache-square or rate-gradient overflow also prevents a
simple update bound from certifying every configured trajectory without the
existing failure checks. A proof restricted to ordinary rates/steps would be a
separate conditional policy; no restriction, clipping or optimizer change is
made. The per-pose analytic proof and checked fallback remain,
including directed rounding, DAZ/FTZ/FZ semantics. Field domain/address checks
remain independent.

`ColorSearchOutput::ValueOnly` is independent of the existing `gradient`
SEARCH/accurate arithmetic selector. With SEARCH selected, it retains the
same transform, requested exponential, factored/unfactored iteration order,
coefficient product and plane fold as full SEARCH. Compile-time specialization
omits force/torque output. A register-only compiler barrier preserves the FP32
coefficient-product rounding boundary that the full SEARCH path gets from
reusing the product for force; without it GCC fuses that product into the
value sum and native exact tests fail. The explicit choice is used only after
the last optimizer update. The final field-values path retains all existing stencil/channel loads,
tail/sentinel handling and value reduction while omitting unused derivative
accumulation. It does not select scalar-channel-only sampling. Accurate winner
reporting, strict earliest-start ties, all-start optout, shape-only/rescore
objectives and invalid-final error propagation keep their existing policies.

On ARM the full-gradient FP32 field dataflow is a literal function body with
a direct inlined FP32 adapter call, not a `NeedGradient` template: native LTO
outlined the templated preparation/evaluation, including in the all-knobs-off
comparator. A separate final-values function retains the exact stencil tree
and source-order value fold. Preparation and full-gradient helper inlining is
explicit, and the linked full evaluator must inline preparation/full gradient.
Geometry eligibility, masked per-pose domain checks, sentinel/tail handling and
the general size_t/schedule/storage fallbacks are unchanged; no containment
proof follows from this structure. It passes the public, directed/flush-mode
and frozen all-eight all-20-column score/pose/start byte gates. Measured
performance supports parity, not a speedup: the matched three-query geometric
ratio against the matched accepted control is 1.001668, within raw spread, and
the shipped bound/final-shape-values composition measures 0.999154 (-0.0846%).
The field memory-safety checks remain independent of descriptor or
analytic-envelope eligibility.

## Shipping transformed-coordinate validation

The reviewed x86 VEX YMM8 pair and XMM4 singleton paths validate transformed
FP32 coordinates with ordered sign-cleared absolute-value comparisons against
the same existing finite reduced-coordinate bound, combining all XYZ lane
masks before publication. Translation envelope checks, transform operation
order, private posed stores and exact canonical fallback remain unchanged.
Both conformers must validate before publishing
values or counts; unsafe late failure does not partially overwrite outputs.
This is a validation schedule change, not a new pose or precision contract.

Experimental x86 field scheduling variants are disabled and there is no
additional accumulator boundary. The ARM field defaults and contracts are
independent of this x86 validation schedule.

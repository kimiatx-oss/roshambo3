# FIELD same-geometry arithmetic controls v2 (before fresh-seed execution)

The v1 geometry-transfer protocol failed both heldout sets. Its raw receipts,
frozen limits, plant detection and independent endpoint diagnostics remain.
Even-transform arithmetic sensitivity did not cover the odd-transform cases.
No v1 bound is widened and no failed receipt is relabeled as a pass.

Use the v1 runner, native settings, multiset invariants, own-pose fp64 metrics
and freeze recipe, now with all 128 fixed panel rows in *both* calibration and
heldout reductions. The independent unit held out is the arithmetic order seed,
not geometry. Calibration seeds 271600..271631 and heldout seeds
271700..271715 have not been executed before this protocol. Freeze max+oracle
before heldout/clean/plant execution. No repeated attempt after a failing result.
The instrument must move poses. All heldout controls must pass and the native
lr_q=.11 plant must fail a gate; otherwise the finite canary is not qualified.

This corrects the unsupported geometry-transfer assumption rather than fitting
the observed residuals. It makes no unseen-geometry claim. Even with IID seed
statistics, the maximum of 32 calibration samples exceeds all 16 heldout samples
with probability only 32/48 = 2/3 for one continuous metric. Multiple metrics
and correlated seeds do not improve that guarantee. The frozen empirical
envelope is conditional evidence, not an architecture-derived theorem.

Scalar analytic anchor comparisons remain separate and unqualified: a software
FIELD reference would need to reproduce independently the declared grid domain,
Gaussian node values and gradients, half texel storage, multidimensional sampler
weight/rounding arithmetic, out-of-bounds handling, all optimizer steps and
sampled cross-start selection. Then independently bounded arithmetic variants
could calibrate their own-pose quality and start/empirical endpoint distributions
against the analytic scalar anchor. Neither analytic final rescoring nor the
current order ensemble supplies that missing sampler/trajectory bound. The
sibling arithmetic lane's convex-hull full3D containment is insufficient for a
tight signed mean or optimizer derivative bound. Keep full qualification false.

Run with --same-geometry, fresh output and scratch paths. Existing odd-transform
fp64 BFGS diagnostics remain a separate empirical analytic endpoint diagnostic;
they do not classify attraction basins of the sampled FIELD objective. All v1
lock, source freshness, build and no-bit-fixture requirements remain in force.

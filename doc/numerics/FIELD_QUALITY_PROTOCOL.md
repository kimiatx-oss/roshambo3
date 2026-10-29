# FIELD scalar-anchor shape quality experiment, v1

This protocol is frozen before generating reference endpoints, sampling native
control endpoints, or inspecting optimized native results. It is a new fixed
geometry experiment, separate from the finite atom-order canary. Shape first;
no combo qualification follows from it.

Use four public asymmetric fixtures from `numerics_fixtures.load_panel`, eight
independent rigid transforms from NumPy/SciPy RNG seed275200, all16 ordered
fixture pairs per transform, physical coordinates wired to .001 Angstrom.
Keep128 pairs for every control. Scalar anchor A is canonical CPU shape,
grain=-1/poly=false, four starts,25steps, lr_q=.1, lr_t=.4. Retain its original
row, winning start, physical pose and independently evaluated fp64 score.

Generate32 calibration control endpoints using seeds275300..275331, then16
heldout endpoints275400..275415. For each seed and each unique query, select
independently Bernoulli lower/upper endpoints of every independently derived
Gaussian-prefilter half4 node interval. Those arrays are arithmetic enclosure
realizations; joint reachability by hardware is not claimed. Use independent
fp64 AdagradExp, normalized quaternion products, four pi-axis starts,25steps,
rotation .1 and translation .4 Angstrom, stored-gradient interpolation, ideal
round-to-even tensor8 fractions. Select software winners using software values.
Tensor8 is solely a reference endpoint generator, never an exact CUDA model.
Save all poses and original inputs before any native optimized call.

Native fixed-pose controls use FIELD steps=0/start_mode=0 on moved original
library clouds, with original query arrays unchanged, field spacing .8/pad8,
canonical shape library counts and width. Record actual sampled objective F
from native start metadata, never final analytic score columns. Explicitly
retain the extra float32->short->float library quantization. Normalize all raw
objectives with Phi(F)=F/(S-F), where S is the original physical fp64 sum of
shape selfs in raw Gaussian units; reject reaching the Tanimoto pole.

Independently derive staging bounds d_A and d_C between original rigid poses
and their re-input clouds, from Gaussian-node operation intervals, fp32 pose
and address arithmetic, full3D range-preserving convex hulls, border zero, and
actual four-lane serial/XOR accumulation. No observed residual sets these
bounds. This deliberately conservative model cannot assume hidden combined
texture weights. Bound native winning pose distortion e_N with the same model,
including native non-normalized returned quaternion and the full rounding
preimage of its returned physical translation.

Freeze R=max_calibration_seed mean[Phi(F_A_reinput)-Phi(F_C_reinput)+d_A+d_C]
before heldout or native optimized results. Do not clip negative R. Test all16
heldout scalar discrepancies against this same R without refitting. Record the
conditional algebraic own-pose loss bound

    mean(T_A-T_N) <= mean(T_A-Phi(F_A_reinput)) + R + mean(e_N).

The candidate must separately satisfy the sampled-objective regret comparison
mean[Phi(F_A_reinput)-Phi(F_N)]<=R; distortion must fit its independent interval.
Also require the allowance below mean(T_A), so the quality gate excludes a
zero-score result. Otherwise mark it uninformative and fail qualification.

Only after the frozen file exists, run clean native25step/four-start FIELD and
a zero-learning-rate native plant (lr_q=lr_t=0, same25steps/starts), plus
the ticket-required lr_q times1.1 native plant (.11 with lr_t=.4). Require both
plants to fail the informative scalar quality or sampled-regret gate. Preserve
all failures, including heldout failures or a broad uninformative bound; no
alternate seeds, endpoints, or thresholds after observing outputs.

The maximum of32 independent continuous calibration values covers all16 IID
heldout values with nominal probability32/48=2/3, not a guarantee. Node controls
and this fixed128-pair geometry panel are explicit finite models, not universal
hardware trajectory or unseen-geometry coverage. Native winning-start and
physical basin acceptance remain separate obligations; this experiment records
starts but does not turn raw pose displacement into a basin certificate.

Use an own Release build, ARCH_NATIVE=OFF, architecture120, source freshness
checks and the physical GPU lock before CUDA context. Preserve the preregistered
protocol hash, input/control NPZ hashes, exact source and loaded binary identities.

The scalar quality inequality is a conditional consequence of independently
controlled sampled regret and directed distortion, not a second independent
scalar error detector. Informativeness failure alone is not plant detection.

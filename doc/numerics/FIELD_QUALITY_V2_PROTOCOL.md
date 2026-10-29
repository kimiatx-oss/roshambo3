# FIELD scalar-anchor shape qualification v2: direct endpoint point controls

Preregister before any v2 endpoint generation or native point observations.
Preserve rejected v1. Shape only; combo follows separately. Fresh geometry seed
276200,32 calibration seeds276300..276331,16 heldout276400..276415. Four public
asymmetric fixtures, all16 ordered pairs, eight rigid transforms:128 pairs.
Coordinates are wired once to short-dequantized .001 Angstrom. Settings remain
four signed-axis starts,25 AdagradExp steps,lr_q=.1,lr_t=.4,spacing.8,pad8.

Retain original canonical scalar CPU anchor A rows, starts, independent fp64
scores, and physical poses. Independent software searches use fp64 normalized
quaternion exponential updates, ideal tensor8 stored-gradient sampling, and
seeded Bernoulli endpoint realizations of independently derived half4 Gaussian
node intervals. Save ALL FOUR terminal poses for every seed and pair, not only
the software winner. Joint hardware reachability of all interval endpoint bits
is not asserted; this is a declared finite arithmetic enclosure control model.

For each physical endpoint, transform the ORIGINAL library cloud, multiply by
the pinned reduced-frame scale in fp64, then round each resulting point once
to fp32. Those rounded point clouds are the declared controls themselves. They
are not re-input library records and are not restored to hypothetical native
poses through two broad hull bounds. Bound physical displacement due to point
rounding relative to the COMPUTED fp64 reduced endpoint by componentwise
u*abs(x)+half-smallest-subnormal divided by scale,u=2^-24, with outward fp64
evaluation margin for positive operations. This is solely a point-rounding
guard, not an exact-real quaternion/rotation error envelope; no cancellation-
sensitive physical transform claim is made. The computed points themselves
define the independent control. Check conversion against that bound and record
its maximum. Keep translations within16
Angstrom. No measured error sets this guard or any score threshold.

Evaluate the original scalar anchor point cloud and all four control endpoints
with the actual production build_query_field and sample_field diagnostic seams,
using the original query, host-derived lattice origin/spacing/cells, half4 nodes,
border mode and fp32 four-lane serial accumulation then XOR offsets1,2.
The actual production lane_group_reduce<WA> loops off=1;off<WA;off<<=1
(kernels.cu); use that source-derived order, not a mirrored synthetic tree. Native point objectives,
not ideal tensor8 values or analytic final columns, choose each control's actual
best start with earliest strict argmax. Check every actual builder node against
independent directed Gaussian intervals. Before calibration, require exact raw
float32 identity-pose equality against actual FIELD steps0/start0 metadata for
all128 original short-wire pairs. Repeat builder bits unchanged in later phases.

Normalize raw sampled overlap with Phi(F)=F/(S-F), S the ORIGINAL physical fp64
sum of shape selfs in raw Gaussian units. Reject a pole. The actual F_A is a
fixed preregistered point control; it need not be an unobserved exact sampled
objective at a different rigid implementation of A. The identity

    T_A-T_N = (T_A-Phi(F_A)) + (Phi(F_A)-Phi(F_N)) + (Phi(F_N)-T_N)

therefore incurs no hypothetical endpoint-restoration charge. The first term is
fixed before N. Freeze the maximum32-control signed sampled regret for the
second term. Independently bound the last term with Gaussian-node, full3D
range-preserving convex hull, native pose/address and actual WA4 accumulation
intervals, including native nonunit quaternion and returned-translation rounding
preimage. The resulting scalar bound is a conditional consequence of sampled
regret, not a second independent detector. Require it below mean(T_A), excluding
zero quality, both overall and in every nonempty fixed score stratum.

Strata are the existing metrics.floors SCORE_STRATA, assigned using ORIGINAL
scalar own-pose fp64 scores before control observations. Freeze separate
maximum32 limits overall and per nonempty stratum for signed sampled regret,
winning-start disagreement against the independent scalar anchor, and empirical
basin disagreement. Basin cases are the64 odd-transform pairs, fixed now. Polish
physical placements with the existing independent fp64 BFGS classifier:
central-gradient step1e-5,gtol1e-8,maxiter500; require terminal gradient<=1e-6 and
translation<=16. Compare stationary endpoint atom RMS against the existing
fixture RECOVERY_NEIGHBORHOOD. Unresolved cases fail; do not omit them. This is
empirical attraction-basin identity, not global/topological certification.

Execution phases: CPU prepare -> native32 calibration probes -> CPU basin
classification and frozen limits -> native16 heldout probes -> CPU heldout
basin validation -> native optimized clean and plants -> CPU final basin/report.
No heldout native point objectives or optimized N precede the frozen file.
Independent replay uses saved original inputs, all control poses, native point
values and independent physical endpoints; no native optimized output defines
R, stratum labels or baseline basin endpoints.

Candidate N is the actual25step/four-start FIELD winner with actual sampled
metadata. Require argmax IDs, directed containment, informative conditional
quality, sampled regret, winner and basin limits, all fixed nonempty strata.
Native .11 rotation-rate plant (times1.1,lr_t=.4) must fail at least one applicable
sampled-regret/winner/basin gate. It may improve mean quality; do not force
scalar loss to reject improvement. Additional zero-learning-rate native plant
must fail sampled regret or conditional scalar quality. Informativeness failure
alone is never plant detection. Preserve all failed controls/plants unchanged.

Finite controls qualify only the declared fixed geometry/model. A maximum32
continuous IID calibration values covers all16 heldout values with nominal2/3
probability for ONE metric; joint coverage across strata/metrics is weaker and
not guaranteed. Do not widen limits, switch seeds, delete failures or silently
claim unseen-geometry or universal optimizer convergence coverage. Native and
software winners/basins are compared with the independently defined anchors,
never by fitting endpoints to native outputs.

Own Release build,ARCH_NATIVE=OFF,arch120; source/model/driver/fixture/binary
hashes and physical GPU lock before context. Capture identity at entry and check
it before every artifact/freeze. Refuse overwriting any prior state artifact.

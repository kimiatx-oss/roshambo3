# FIELD optimizer order-control protocol (before execution)

Use the four public asymmetric shape/typed-color fixtures and the eight frozen
transforms from optimizer-real-v2. Even transforms (64 pairs) calibrate; odd
transforms (64 different pairs) are held out. Use 32 calibration permutations
(seed 271300..271331), then 16 independent held-out permutations (271400..271415).
Run shape and combo, four starts, 25 steps, lr_q=.1, lr_t from SETTINGS;
combo uses the shipped staged color cell, S=4, CSTAGE=48. Wire float32 coordinates
are already on the declared .001A grid. Do not change settings after outputs.

The independent control is reordering the *same* atom multiset and each typed
feature segment. Geometry, typing, exact mathematical objective and schedule
remain invariant. This changes the addition order both in query field node
construction and in library sample/gradient accumulation. NUMERICS_POLICY §2
explicitly allows reassociation inside Tier2 grid search. These controls model
that declared arithmetic variation only, not every texture sampler or device.
They are not comparisons of a production residual distribution to fitted limits.

Against the unpermuted native FIELD anchor, measure own-pose fp64 Tanimoto
mean loss, actual native winning-start disagreement and >.1A raw shape RMS
movement. For each metric freeze the maximum calibration control value (absolute
for signed mean), adding only the pre-existing fixture oracle uncertainty to
mean loss. Persist frozen limits *before* heldout/clean/plant execution. Never
widen limits. Controls must preserve exact coordinate/type multisets and native
IDs must equal earliest argmax of actual sampled search objectives, every call.
Record whether the instrument moves any poses; inert controls cannot certify
arithmetic coverage. Every heldout control and a fresh clean arm must pass.
The native lr_q=.11 arm must exceed at least one frozen metric to qualify the
finite schedule canary. Failure of either requirement remains a recorded failure.

Separately compare FIELD own-pose fp64 quality and native integer starts with
scalar analytic anchor. There is currently no derived FIELD-vs-analytic amount:
full3D sampler arithmetic remains unresolved. These values are diagnostics;
they do not inherit order-control limits. Final analytic rescoring does not
qualify sampled search. Keep top-level qualification false while this gap exists.

Polish all odd-transform clean and planted physical poses and scalar analytic
anchor poses using the existing independent fp64 BFGS protocol. Stationarity
<=1e-6, translation<=16A, endpoint neighborhood .01A+2sqrt(3)*.0005A.
Classify unresolved endpoints explicitly; raw RMS movement is never called a
basin flip. This empirical analytic endpoint diagnostic cannot establish the
identity of FIELD sampled-objective basins or topological/global uniqueness.

Acquire canonical physical-device lock before CUDA context, refuse busy
preflight, verify current native content against the loaded isolated Release
ARCH_NATIVE=OFF explicit120 build. Raw pose/objective arrays are temporary
observations, never expected-output fixtures. Commit runner before execution.
No D013 retrieval or general FIELD SEARCH qualification is claimed.

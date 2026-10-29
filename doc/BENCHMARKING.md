# Benchmarking

Keep corpora outside the checkout and configure `ROSHAMBO3_CORPORA`. Record
source/build identity, CPU or GPU model, compiler flags, toolkit, corpus and
query fingerprints, warmup/repeat phase, cache state, and achieved operating
conditions. Generic run recording helpers are in `bench/`.

Timed runs on a shared machine use `ROSHAMBO3_CPU_LOCK` or
`ROSHAMBO3_GPU_LOCK`. Use the same base path in every competing process; GPU
locks are scoped by host and device UUID. Acquire the GPU lock before making a
CUDA context. Correctness checks may be untimed, but their outputs are not
performance measurements.

Separate end-to-end wall time from kernel time and conformers per second from
conformer-query evaluations per second. Do not combine operating conditions,
warmup and measured repeats, or tuning and held-out evaluation. Profiler clock
control can override externally locked clocks; record the achieved clocks.
Use issue and instruction counters plus causal ablations for bottleneck claims.

Read `NUMERICS_POLICY.md` and `METRICS.md` before interpreting score, recall,
ranking, or throughput differences. Historical campaigns and their required
public evidence will be assembled in the paper candidate's `research/` tree.

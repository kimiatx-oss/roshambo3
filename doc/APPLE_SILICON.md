# Apple Silicon CPU support and measurement

ROSHAMBO3's existing ARM CPU backend builds natively on Apple Silicon with
Python 3.11 or newer and Apple clang. CUDA is disabled for this platform. The
web screening worker supports native CPU and experimental Metal jobs on macOS;
see [the webapp guide](WEB_APP.md) and [Metal GPU guide](METAL.md). The CPU
port retains the Linux ARM kernels, FP32 field defaults, optimizer settings,
and accurate final scoring; it introduces no Apple-specific kernel tuning.

## Build

Use an arm64 terminal and a native Python environment:

```sh
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . \
  --config-settings=cmake.define.ROSHAMBO3_CUDA=OFF
.venv/bin/python -c 'import platform; assert platform.machine() == "arm64"'
.venv/bin/python -c 'from roshambo3.backend.cpu import build_stamp; print(build_stamp())'
```

CMake uses Apple clang's portable `armv8-a+simd` target. FP16 and FP16FML
instructions remain in separately targeted translation units. Do not reuse a
Linux build directory or install an x86 Python under Rosetta for this build.
The loaded native extension reports its compiler, effective arithmetic/ISA
flags, source identity, and dispatch target.

OS services are isolated in `cpu_platform.cpp`; its pure native header does
not depend on Python headers. Linux affinity and HWCAP probes stay there.
macOS uses `sysctlbyname` for FP16/FHM capabilities and pthread QoS for worker
scheduling. Failed feature queries disable optional instructions. No OS probe
or scheduling call is added to an atom, texel, or optimizer iteration loop.
The existing NEON FPCR read remains at scoring entry, and existing rounding
barriers remain unchanged.

## Scheduling policies

`CPUWorkerOptions.cpu_qos` accepts:

| Policy | Behavior |
| --- | --- |
| `inherit` | Preserve ordinary scheduling; the default. |
| `background` | Request background QoS for the submitting thread and compute workers. |
| `user_initiated` | Request user-initiated QoS for both. |

For example, add `cpu_qos="background"` or `cpu_qos="user_initiated"` to an
existing `CPUWorkerOptions` configuration. The same option is supported by
`CPUComboExecutor`, `score_pairs`, and `Engine.run_worker_owned_cpu`.
Executor policy is fixed at construction. Scoped caller policy is restored
on success and failure; threads whose original QoS is unspecified are refused
before mutation because pthread QoS cannot restore that state.

Explicit QoS is refused outside macOS. Strict `require_worker_affinity=True`
remains supported on Linux and is explicitly unavailable on macOS.
`stats.cpu_scan["worker_placement"]` reports requested/read-back worker QoS
and actual affinity observations. These observations establish the requested
scheduling class, not the core on which a worker ran.

Background QoS favors energy-efficient scheduling; user-initiated QoS favors
responsive execution. Neither is a P-core/E-core pinning guarantee. Apple
affinity tags express cache-sharing hints, not core-class masks. See
[Apple's scheduling guidance](https://developer.apple.com/news/?id=vk3m204o)
and [thread policy definition](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/thread_policy.h).

The tested M4 Pro has 10 P-cores and 4 E-cores. Useful experiments are
background QoS with 1/2/4 workers, user-initiated QoS with 1/2/4/8/10 workers,
and user-initiated QoS with 12/14 workers to investigate mixed execution.
Label these by requested QoS until Instruments scheduling traces establish
worker residency. Full Xcode/Instruments is needed for those traces; Command
Line Tools alone do not supply `xctrace`.

## Reproduce the benchmark

Install the development tools (`pytest`, `psutil`) in the same environment.
Run from the repository root. Freeze queries once, then copy the identical
NPZ to the other host; do not regenerate query geometry there:

```sh
.venv/bin/python -m bench.apple_silicon \
  --corpus /path/to/corpus --queries /path/to/queries.npz \
  --out /path/to/freeze-receipt --freeze --freeze-only
.venv/bin/python -m bench.apple_silicon \
  --corpus /path/to/corpus --queries /path/to/queries.npz \
  --out /path/to/provisional-inherit --workers 4,10,14 \
  --warmup 1 --repeats 1 --provisional
.venv/bin/python -m bench.apple_silicon \
  --corpus /path/to/corpus --queries /path/to/queries.npz \
  --out /path/to/provisional-background --workers 4 --qos background \
  --warmup 1 --repeats 1 --provisional
.venv/bin/python -m bench.apple_silicon \
  --corpus /path/to/corpus --queries /path/to/queries.npz \
  --out /path/to/provisional-user-initiated --workers 10,12,14 \
  --qos user_initiated --warmup 1 --repeats 1 --provisional
```

All commands default to combo, shape, and shape-plus-final-color-rescore.
The frozen query panel uses durable `(mol_id, conf_idx)` identities, in this
order:

```
(58059676, 0), (12673788, 0), (57076096, 0), (13625945, 0),
(59925789, 0), (10877595, 0), (84669660, 0), (14288517, 0)
```

The benchmark resolves these keys through the format-3 catalog and refuses
missing identities. `--panel-keys` can select a unique subset of this frozen
panel as comma-separated `mol_id:conf_idx` pairs. The old GIDs 25511, 205370,
236070, 684138, 154726, 200425, 425209, and 685793 name the pre-switch captures
only; they must not be used to select the live panel from a rebuilt corpus.

Every data file is fingerprinted. Records include native source/binary
identity, compiler flags, dependency versions, operating conditions,
per-query wall time, worker/stage counters, and process peak RSS.
Stage counters sum worker time; they are not an additive decomposition of
whole-screen wall time. RSS is a process high-water mark.

The reused Engine retains one query-context generation. Rotating different
queries therefore includes query-context rebuilding; file caches may be warm.
`--fresh` recreates the Engine for each screen. To launch a new Python process
for each query instead, use `python -m bench.fresh_cpu_port` with the same
corpus/query/output arguments. It separately records screen API time and
process time; process time also includes benchmark imports/provenance/output
overhead. Neither mode flushes filesystem caches or claims cold disk I/O.

For complete numerical outputs, use `--level conformer` and set `--k` to the
actual corpus conformer count (`corpus.json`'s `n_conf`). The historical
687,440-row count is not a requirement of the current format or benchmark.
Captures are named
`{mode}-{qos}-{workers}-r{repeat}-m{mol_id}-c{conf_idx}.npz` and contain
`scores`, `mol_ids`, and `conf_indices`. Repeat 0 is saved by default;
`--save-all` saves every repeat, including warmups. Compare matching query
keys and join captured rows by `(mol_ids, conf_indices)`, checking coverage,
uniqueness, and input/provenance hashes before comparing values or rankings.

`bench.compare_cpu_port` is a legacy-capture comparator: it expects the old
`...-g{gid}.npz` filenames and a `gids` array. It does not consume the current
durable-key captures and is not a comparison command for new benchmark runs.
The trajectory-outlier audit helper used in the original port campaign is not
shipped. Current captures retain score/pose columns for an independently
implemented fixed-pose audit, but no replacement audit command is provided.

The existing grading rules require at least 3 warmups and 11 timed repeats.
Short provisional runs keep their refusal receipts; `--provisional` does not
weaken those rules. Quiet-session measurements should use `--warmup 3
--repeats 11` from a clean source snapshot. macOS still receives advisories
for unavailable strict binding and achieved-frequency telemetry. Missing
telemetry is never recorded as a steady clock or verified placement.

## Validation and next measurements

Native macOS arm64 joins Linux x86/ARM in CPU CI, including source and wheel
installation checks. Tests preserve Linux affinity requirements, recognize
both `aarch64` and `arm64`, and use portable descriptor-count checks.
Multiprocessing preparation explicitly preserves RDKit molecule properties
when spawning workers, including the names used to group conformers.

The supplied port was exercised on native macOS arm64 and Linux ARM. Its raw
benchmark receipts and molecular inputs are retained in the originating research
checkout and are not distributed with the tool. The scripts here let users
produce equivalent receipts using their own approved corpus and frozen queries.
No Apple-specific kernel tuning or new performance claim is promoted by this
integration. Platform status is in `../RELEASE_STATUS.md`.

## Per-worker comparison

Measure one worker on each host with the same frozen query file, corpus, and
screen settings. Use `--workers 1 --qos inherit` on Linux and
`--workers 1 --qos user_initiated` on macOS. Retain the worker-placement records:
Linux can verify the scoring worker's CPU binding; macOS QoS read-back alone
cannot establish P-core residency. The process may also perform preparation
and reporting outside that scoring worker.

Report conformer-query evaluations divided by API wall time, and separately
by `cpu_scan.scoring_s` for the one-worker scoring stage. At multiple workers,
API throughput divided by worker count describes scaling efficiency; it is
not a measurement of an individual P-core when E-cores may participate.
Stage counters at multiple workers sum worker time rather than elapsed API
time. Compare complete query panels or matched query IDs, never partial
panels. Preserve repeat counts, compiler identity, grading status, and source
hashes alongside ratios.

A peak-clock ratio is a hypothesis, not measured frequency normalization.
Equal NEON vector width does not establish equal throughput per cycle;
compiler code generation, instruction issue, cache behavior, and sustained
clocks also affect the result. Keep the one-pass laptop measurements
provisional until quiet repeats and residency/frequency evidence are available.

# Running the test suite

Install the development tools in the checkout's own venv (`uv sync --group dev`
or install pytest, pytest-xdist, filelock and psutil alongside the editable
package). The parallel-runner acceptance used Python 3.11.15 and RDKit 2026.09.1.
Check `roshambo3.__file__` before running another checkout's tests.

```bash
export ROSHAMBO3_GPU_LOCK=/path/to/shared/gpu.lock      # base of the per-device lock family
export ROSHAMBO3_CORPORA=/path/to/corpora               # optional prepared corpora
export ROSHAMBO3_TEST_SDF=/path/to/confs.sdf            # optional; tests needing it skip
.venv/bin/python -m bench.test_suite --output /tmp/tests-parallel
```

On a host where another tenant runs an NVIDIA MPS daemon, point
`CUDA_MPS_PIPE_DIRECTORY` at a directory of your own first; otherwise every
process that makes a CUDA context can block indefinitely.

The output directory must be new. CPU tests run on `-n 4 --dist loadfile`;
GPU tests then run in a separate serial process. Resource use is declared in
`tests/suite_resources.json`, never inferred from GPU words in comments or
mock objects. `[]` declares a CPU-only file; `['*']` declares a GPU-only file.
A list of class/function paths selects the actual GPU users in a mixed file;
`"@backend"` selects only its `backend="cuda"` or `backend="metal"` parameter
variants. Other tests in that file run in the CPU lane, including CPU variants
of numerical and web pipelines. The inventory must cover every test file;
new files require a declaration, and tests check for stale files/selectors.

Mixed files are collected in both lanes, with complementary per-item selection.
Lane routing is distinct from scientific qualification deselection. The merged
outcome receipt refuses duplicate test IDs; serial comparison detects omitted
or visibility-sensitive cases. CPU workers have CUDA visibility disabled before
collection, preventing accidental CUDA contexts. GPU imports/probes must not
create a Metal context during collection; put device allocation in fixtures or
test bodies. Direct multiworker pytest selecting GPU tests is refused; the
entry point supplies `--suite-lane cpu` with devices hidden before worker startup.

GPU collection probes and each individual GPU-lane test hold the canonical
host/device `bench.gpu_lock.GpuLock` on every visible NVIDIA device in UUID
order, acquired before fixtures. Resolution/query failures stop the run. The lock is
released between tests. A busy GPU causes a visible failure, never skipped
coverage or an unbounded wait with a live context. Retry the run after the
other GPU owner finishes. `events.jsonl` records test PID/worker and held-lock
intervals. No performance clock control is requested for suite correctness.

Compiler-heavy tests use one host-wide file lock. CMake parallelism is two;
BLAS/OpenMP pools default to one thread in each test process. Tests explicitly
exercising worker/thread counts retain those inputs. The three ordinary
standalone color-oracle setups share content-keyed C++ object files; build,
stamp, sanitizer and ISA-inspection subjects keep their fresh builds. Keys
include sources/headers, compiler identity, flags, relevant compiler environment,
architecture and package versions. Repeated codec corpora share a locked,
content-keyed template; every caller gets private copies (no hard links).
Incomplete cache builds are discarded. Caches live in the run's output by
default; a shared pytest `tmp_path_factory` root serves direct invocations.

For a serial baseline and a checked comparison at the same clean commit:

```bash
.venv/bin/python -m bench.test_suite --serial --output /tmp/tests-serial
.venv/bin/python -m bench.test_suite --output /tmp/tests-parallel --compare /tmp/tests-serial
```

Both commands use fresh caches and identical thread limits and fixture inputs.
The entry point writes one merged numerical report, combining raw observations
by test ID and canonical order. Skips/deferred cells retain reasons; worker loss
invalidates certification. Worker reports never overwrite the controller's
report. `run.json` records before/after wall time, commands, environment and the
20 slowest files (summed pytest setup/call/teardown durations, not elapsed file
wall time in a parallel run). `outcomes.json` retains every report phase,
including subtests, expected failures and reasons; comparison ignores only
execution timing. Numerical JSON is compared exactly, without dropping stamps
or evidence. A matching report can still say NOT CERTIFIED; parallel execution
does not establish missing scientific qualification.

CI should switch to this entry point only after serial/parallel parity has been
reviewed. Hosted CI execution remains serial; its install lists include the new test
tools. Local parity on one host does not claim hosted or cross-platform CI execution.

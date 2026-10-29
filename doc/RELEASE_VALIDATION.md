# Release validation

Build from a fresh source archive in an isolated environment. Record loaded
package/native-extension paths and `roshambo3.get_build_info()`. The first release
publishes source archives and an sdist; locally built wheels are validation
artifacts. Linux x86-64 requires AVX2/FMA (x86-64-v3). Linux AArch64 builds the
baseline NEON path and keeps optional FP16/FP16FML units isolated.

| Gate | Configuration |
|---|---|
| x86 CPU | GNU C++ 13, Python 3.11+, `ROSHAMBO3_CUDA=OFF` |
| ARM CPU | Native AArch64, `ROSHAMBO3_CUDA=OFF` |
| Apple Silicon Metal GPU | Native macOS arm64 15+, `ROSHAMBO3_METAL=ON`, `ROSHAMBO3_REQUIRE_METAL=1` |
| Apple Silicon CPU | Native macOS arm64, Apple clang, `ROSHAMBO3_CUDA=OFF` |
| Older CUDA | CUDA 12.9, TITAN V / sm_70 |
| Current CUDA | CUDA 13.2, Blackwell / sm_120 |
| Web | Node 22, `npm ci`, `npm run build`, Playwright Chromium |
| Packaging | Fresh sdist install, demo prepare/screen/poses, browser assets |

CPU tests run with `python -m pytest tests`. Generic lifecycle, caching and top-K
tests select the retained native color arm; AVX2-only fixtures and trajectory
shape experiments require their ISA. ARM checks cover NEON dispatch, concurrent
call isolation, exact analytic fallback, native worker reductions, and refusal
of unsupported field layouts. Missing host frequency telemetry is reported.
On multi-GPU hosts, telemetry checks resolve the device explicitly and verify
that an implicit default is refused.

Missing GPU/corpus tests skip; count
and explain those skips. A CPU-only pass never certifies CUDA or ARM. The manual
GPU workflow requires appropriately labeled, trusted self-hosted runners.
Correctness tests are untimed. Measured performance needs the device lock,
matched clocks and telemetry described in `doc/BENCHMARKING.md`.

Production arithmetic, ranking tie-breaks, and ISA boundaries are fixed in the
release build. Historical optimizer policies have distinct trajectories:
require repeatability within a policy and independently validate score meaning
at its returned pose. See `doc/CPU_SHARED_POSE_CONTRACT.md`.

Apple platform checks cover QoS restoration after failures, thread-local
policies, explicit refusal of strict affinity, optional ARM feature probes,
and property-preserving RDKit preparation under spawn. `macos-15` CI asserts
arm64 and tests source installation plus a rebuilt wheel. QoS read-back does
not establish P/E-core residency. See `APPLE_SILICON.md`.

Metal integration retains explicit backend selection, precise analytic final
scoring and backend-specific search policies. Linux checks skip native Metal
execution; supplied M4 Pro receipts are distinguished from independently run
integration tests. See `METAL.md` and `../RELEASE_STATUS.md`. Ordinary CPU CI
sets `ROSHAMBO3_METAL=OFF`; the manual Metal workflow requires a native GPU.

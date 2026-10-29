# Release status (1.0.0)

Version 1.0.0 (2026-10-24). The test suite (`python -m bench.test_suite
--workers 4`) passed at the 1.0.0 head, in fresh clones, with no failures:

| platform | CPU lane passed | GPU lane passed |
|---|---|---|
| Linux x86-64, CUDA (RTX PRO 6000 Blackwell, sm_120) | 3,516 | 338 |
| Linux x86-64, CPU only | 3,471 | — |
| Linux aarch64 (Neoverse V2), CPU only | 3,361 | — |

* **CUDA** is the primary backend. CUDA builds have also been tested on Volta
  (TITAN V, sm_70) and Ada (L4, sm_89).
* **CPU** backends (x86 AVX2+FMA, ARM NEON) are supported on Linux and macOS.
  CI builds and tests the CPU package on Linux x86-64, Linux aarch64 and macOS 15.
* **Metal** (Apple Silicon, macOS 15+) is experimental and must be selected
  explicitly; see `doc/METAL.md`. The macOS suite, CPU and Metal, has passed
  on an M4 Pro at earlier development commits but has not been rerun at the
  1.0.0 head.
* The web interface is a lab demo, intended for trusted networks only.

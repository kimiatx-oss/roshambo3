# Development

Use an isolated environment installed from this directory. Do not reuse an
editable install or native binary from another checkout. Inspect the loaded
package and extension paths before trusting a test result.

```bash
python -m pip install -e '.[web]'
python -m pip install pytest psutil httpx "pytest-xdist>=3.8" "filelock>=3.13"
python -m pytest tests
```

The parallel entry point runs the CPU lane on pytest-xdist and GPU tests in a
separate serial process, holding the GPU lock per test. See `doc/TESTING.md`:

```bash
python -m bench.test_suite --output NEW_DIRECTORY
```

Tests requiring CUDA or external corpora can skip when those resources are absent.
A passing CPU suite does not certify GPU or ARM execution. Corpus locations use
`ROSHAMBO3_CORPORA`; bulk data remains outside the checkout. Quoted benchmarks
must use the shared host/device lock convention and record achieved operating
conditions. Read `doc/NUMERICS_POLICY.md`, `doc/METRICS.md`, and the backend
contracts before changing scoring, reductions, or ranking.

Rank descending by score, breaking ties by ascending conformer ID. Preserve
reduction order and declared arithmetic. Never enable fast math to make a build
faster without a declared numerical change and validation.

Historical campaign tests have been separated from ordinary correctness tests.
Their disposition is recorded by the assembly inventory. Promote meaningful
portable checks before permanently removing any research-only test.

This repository starts from a fresh history; the internal development history
is not published.

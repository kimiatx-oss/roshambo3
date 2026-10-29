# User guide

ROSHAMBO3 screens a library of molecular conformers against one or more 3D
queries. It optimizes a rigid alignment for each pair and returns ranked hits
based on shape overlap or a combination of shape and pharmacophore overlap
(called **color**).

The workflow has two stages: prepare the library once as a **corpus**, then
screen queries against it. Screening supports CUDA, native CPU implementations
for x86 and ARM, and an experimental Metal backend on Apple Silicon.

## Installation

Use Python 3.11 or later, CMake 3.24 or later, and a C++ compiler. From this
directory, run:

```bash
python -m pip install .
```

CUDA is optional. The package builds the CUDA extension when a toolkit is
available; CPU screening, preparation, inspection, and metrics also work
without it. To require or disable CUDA explicitly, use one of these commands:

```bash
python -m pip install . --config-settings=cmake.define.ROSHAMBO3_CUDA=ON
python -m pip install . --config-settings=cmake.define.ROSHAMBO3_CUDA=OFF
```

For editable development installs, use `python -m pip install -e .`.
`ROSHAMBO3_ARCH_NATIVE=ON` reduces CUDA compilation time by targeting the local
GPU architecture. Such builds are for local development; use a portable build
with recorded architecture targets for reported measurements.

See [INSTALL.md](INSTALL.md) for build options and
[Apple Silicon support](doc/APPLE_SILICON.md) for native Mac builds. The
experimental [Metal backend](doc/METAL.md) requires Apple Silicon macOS 15+.
The [release status](RELEASE_STATUS.md) describes validation coverage.

## Prepare and screen

Start with a library SDF and a query SDF containing 3D coordinates:

```bash
roshambo3 prepare library.sdf -o corpus
roshambo3 screen query.sdf corpus --backend cpu -k 1000 -o hits.csv --names
```

Preparation places conformers in a canonical frame, groups them by heavy-atom
count, and stores quantized coordinates in a compact format. The resulting
corpus can be reused for subsequent screens. The screen above writes up to
1,000 conformer hits per query to CSV; `--names` includes source identifiers
and SMILES.

The same corpus can be screened from Python:

```python
import roshambo3 as r3

corpus = r3.open_corpus("corpus")
hits = r3.screen("query.sdf", corpus, backend="cpu", k=1000)

for hit in hits:
    print(hit.query_name, hit.gids[:5], hit.tanimoto_shape[:5])
```

To prepare a corpus from Python, call
`r3.prep.prepare("library.sdf", "new_corpus")`. Choose a new output path for
each build.

### Select a backend

| Backend | CLI selection | Requirements |
|---|---|---|
| CUDA | `--backend cuda` | CUDA extension and a compatible NVIDIA GPU |
| CPU | `--backend cpu` | Native CPU extension; x86 requires AVX2 and FMA |
| Metal (experimental) | `--backend metal` | Metal extension and Apple Silicon macOS 15+ |

In Python, pass `backend="cuda"`, `backend="cpu"`, or `backend="metal"` to
`screen()` or `Engine()`. An explicit selection overrides the
`ROSHAMBO3_BACKEND` environment variable. If neither is set, the default is
CUDA. An unavailable backend raises an error; it does not fall back to another
backend.

The basic screening examples select CPU explicitly. To run them on a GPU,
change the backend selection. Use `--device N` or `device_id=N` to select a
CUDA device; CPU screening requires device 0.

## Library preparation

### Input requirements

- **3D coordinates:** `prepare` and `screen` require existing conformers and
  do not generate them from SMILES. The optional web application has a
  separate SMILES conformer-generation workflow; see [WEB_APP.md](doc/WEB_APP.md).
- **Molecule identity:** SDF records for conformers of the same molecule must
  share a name. Records need not be adjacent; preparation groups them by name.
  Molecules with different connectivity, charge states, protonation states,
  or tautomers must have distinct identifiers. Conflicting records under one
  identifier stop the build.
- **Hydrogens:** shape scoring uses heavy atoms. Color preparation perceives
  features on the molecular graph with hydrogens added, then stores the
  feature table in canonical heavy-atom order.
- **Conformer consistency:** conformers grouped under one identifier must
  agree on heavy-atom count and color feature table.

Preparation accepts SDF paths, an RDKit binary-zstd stream, or a supported
SQLite database. The Python API also accepts a sequence of RDKit molecules
with 3D conformers.

### Corpus contents

Corpora use [format 3](doc/FORMAT.md):

| Path | Contents |
|---|---|
| `corpus.json` | Format, coordinate frame, color definition, and group table |
| `molecules.parquet` | One row per molecule: IDs, SMILES, heavy-atom count, molecular weight, and rotatable-bond count |
| `groups/*.r3g` | Conformer coordinates, molecule runs, feature tables, and self-overlaps |
| `build.json` | Build inputs, options, statistics, and timings; not required for screening |

Coordinates use a step of 2^-8 Å (0.00390625 Å) and are stored bit-packed in
blocks that never split a molecule. `--block-rows N` sets the target block
limit, from 1 to 4,096 rows (default 4,096). A molecule larger than that target
gets its own block, up to the 4,096-conformer limit. Block size does not change
scores. Stored SMILES encode the molecular graph, so no separate structure
store is required.

Each conformer has a shape self-overlap (its overlap with itself) computed
from its stored coordinates. A color corpus also stores each molecule's
feature table and derives color pseudo-atoms from the stored heavy-atom
coordinates. Color self-overlaps are written by default for canonical
interaction matrices. `--no-color-self` omits them; noncanonical matrices
also omit them and record the reason in `build.json`.

Every completed corpus passes scan-tier validation before being moved to its
final path. Earlier formats, including format-2 `.rsb3` corpora with sidecars,
cannot be screened or converted. Rebuild them from the source molecules.

### Binary and SQLite input

An RDKit binary stream consists of records containing a little-endian uint32
payload length followed by `Mol.ToBinary()` bytes, all compressed in one zstd
stream:

```bash
roshambo3 prepare library.bin.zst \
  --input-format rdkit-binary-zstd --color --workers 48 \
  --temp-dir /fast/scratch -o corpus-color
```

The reader decompresses the stream once and passes bounded batches of binary
records to workers. Workers deserialize the molecules and write temporary
shards. For this input format, `--workers 0` uses all logical CPUs; for SDF
input, the automatic worker count is capped at eight.

SQLite input supports the PubChem and text-ID layouts documented in
[sqlite_source.py](src/roshambo3/prep/sqlite_source.py). A single
`.sqlite`, `.sqlite3`, or `.db` path is detected automatically. To select a
layout explicitly, use `--input-format sqlite:pubchem` or `sqlite:text-id`.
`--sample-seed`, `--sample-p-lo`, `--sample-p-hi`, and `--hac-window LO HI`
select a deterministic subset. The hash interval includes its lower bound
and excludes its upper bound; the heavy-atom window includes both bounds.

### Disk space and build estimates

Temporary shards and the final corpus coexist during preparation. Allow space
for both, with additional filesystem headroom. `--temp-dir` selects the
scratch location; the build's temporary files are removed on success.

Use `--dry-run` to print a JSON estimate without writing files. It reports
record counts where available, estimated CPU time from a small sample, bytes
to read, an upper estimate of bytes to write, and free space at the output
and scratch locations. Counts may be extrapolated or unknown for a solid
binary stream without `--max-records`.

For binary input, `--max-records N` builds a deterministic prefix of the
stream. `build.json` marks the corpus as an incomplete source subset.

### Progress and recovery

A build targeting `OUT` runs in `OUT.partial` under an exclusive `OUT.lock`.
It writes progress to `OUT.progress.json` and checkpoints completed work
units and groups. If an interruption leaves committed work, resume with the
same inputs and options plus `--resume`:

```bash
roshambo3 prepare library.sdf -o corpus --resume
```

Resuming produces the corpus that an uninterrupted build would have produced.
A new build refuses to reuse an existing partial directory. The completed
build is renamed into place and does not overwrite an output that appeared
while preparation was running.

### Identity checks and rejected records

For a build that must reproduce a known molecule set, use either of these
checks:

- `--expect-ids-sha256 HEX` requires the final `ids_sha256` to match. This is
  the SHA-256 of sorted molecule IDs encoded as little-endian int64, recorded
  in `corpus.json`. Expected and actual hashes are recorded in the build or
  progress record. A mismatch prevents publication.
- `--expect-ids-from DIR` checks each work unit against an existing corpus's
  molecule IDs before committing it. The final ID set must also match,
  except for a `--max-records` prefix. This option can read a format-3
  molecule table or, solely as an ID reference, a format-2 directory's
  `ids.i64` and `manifest.json`. The format-2 `sha256_ids` hash uses the same
  encoding as format 3's `ids_sha256`.

Rejected molecules and dropped conformers are listed beside the output in
`OUT.rejected.jsonl` and `OUT.dropped_conformers.jsonl`. Ordinary progress
messages report counts by reason code. `--verbose-rejects` also prints sample
identifiers and reasons.

For confidential libraries, note that build failures may name the molecule
and include its SMILES in both the console and `OUT.progress.json`. Build
records and rejection lists also contain molecule information.

## Screening modes

The CLI provides three modes. All use field-based pose search by default,
followed by accurate analytic scoring at the selected pose.

| Mode | Pose search objective | Reported scores | Default mixing weight |
|---|---|---|---|
| `--mode shape` (default) | Shape overlap | Shape | 0 |
| `--mode rescore` | Shape overlap | Shape and color at the shape-selected pose | 0.5 |
| `--mode combo` | Combined shape and color Tanimoto | Shape and color at the combination-selected pose | 0.5 |

Rescore and combo require a corpus prepared with `--color`. For example:

```bash
roshambo3 prepare library.sdf --color -o corpus-color
roshambo3 screen query.sdf corpus-color --backend cpu --mode rescore -o rescore.csv
roshambo3 screen query.sdf corpus-color --backend cpu --mode combo -o combo.csv
```

Hits are ranked by the combination score:

```text
T_combo = (1 - mixing) * T_shape + mixing * T_color
```

`--mixing` accepts a weight from 0 to 1. At 0, ranking uses shape alone; at 1,
it uses color alone. In rescore mode, changing the weight changes ranking
without changing the shape-selected pose. In combo mode, it also changes the
search objective. Shape mode requires a weight of 0.

In Python, configure color scoring with `ScreenConfig`. This example performs
a combination screen:

```python
import roshambo3 as r3
from roshambo3 import ScreenConfig
from roshambo3.backend import ColorPlacement

config = ScreenConfig(
    color=True,
    mixing=0.5,
    color_placement=ColorPlacement.IN_GRADIENT,
)
color_hits = r3.screen(
    "query.sdf", "corpus-color", backend="cpu", config=config, k=1000,
)
```

For a shape-pose color rescore, use `ColorPlacement.AT_POSE`. Python defaults
are `color=False`, `mixing=0.0`, and `AT_POSE`; setting `color=True` alone
does not change the mixing weight. The corpus supplies the color definition
used to prepare queries.

Legacy CLI options remain available: `--color` selects shape-pose color
rescore with analytic search on CPU/CUDA and field search on Metal. Its
default mixing weight is 0. `--analytic` selects analytic pose search on
CPU/CUDA. Neither option can be combined with `--mode`.

### Field and analytic search

Field search samples a precomputed query field to optimize the pose, then
reports analytic scores at that pose. Analytic search evaluates the pair sums
throughout optimization. It is available through `--analytic` or Python's
`ScreenConfig(pose_search=PoseSearch.ANALYTIC)`, where `PoseSearch` is imported
from `roshambo3.backend`. Metal supports field search only.

The two search methods can select different poses, scores, and rankings.
Analytic final scoring removes field approximation from the reported value
at the selected pose; it does not make the search methods equivalent.

### CPU resources and engine reuse

For supported field configurations, `screen()` automatically uses a native
worker scan for shape, rescore, and combo, with either conformer or molecule
top-K. The Python API uses the available logical CPUs when `workers` is
omitted. The CLI defaults to eight workers; set `--workers N` explicitly to
adjust it.

`CPUWorkerOptions` provides worker counts, optional memory-budget checks, and
numerical controls through `ScreenConfig(cpu_worker=...)`. Memory budgets
are preflight checks, not process memory caps. Nondefault configurations may
use the ordinary CPU scoring route, which supports conformer top-K only.

For repeated compatible screens, pass a caller-owned
`engine=r3.Engine(backend="cpu")` to reuse prepared query fields. Release
retained state with `engine.close()` when finished. See
[the CPU worker options](src/roshambo3/engine/config.py) for supported settings
and [Apple Silicon support](doc/APPLE_SILICON.md) for macOS QoS controls.

### Custom color definitions

Shipped RDKit definitions are accepted automatically. A custom definition
stored in a corpus must be trusted before it is used to prepare queries.
Inspect it with:

```bash
roshambo3 definitions show corpus-color --diff
```

After reviewing the definition, pass its hash with `--trust-definition SHA256`
for one screen, or use `roshambo3 definitions trust SHA256` to save that
decision. Python screens accept `trust_definitions=["SHA256"]`. A hash prefix
of at least 16 hexadecimal characters can identify the definition for a
single run. Use `definitions list` and `definitions untrust SHA256` to manage
saved entries.

## Results

`screen()` returns a `ScreenResult` containing one `QueryHits` object per
query. Each hit includes a global conformer ID (`gid`), molecule ID, conformer
index, scores, and pose parameters. Coordinates and molecule metadata are
loaded when requested:

```python
hits.hit_molecules(query=0, n=20)     # IDs, SMILES, heavy-atom count, weight, ...
hits.poses(query=0, n=20)             # Coordinates in the stored corpus frame
hits.aligned_poses(query=0, n=20)     # Coordinates aligned to the prepared query
hits.to_table(query=0)               # Named columns
hits.to_pandas(query=0)              # Requires pandas
```

Aligned poses are in the query's prepared frame. Returning to the query's
original input frame also requires its preparation transform. A `gid` is a
position in a particular corpus, so it is not a stable identifier across
corpus rebuilds.

Top-K screening retains the best `k` hits per query. CUDA and Metal reduce results
on the device, avoiding transfer of the full score matrix; CPU worker scans
also reduce results without materializing the full matrix.

### Conformer and molecule ranking

The default `level="conformer"` returns up to `k` conformers per query.
Several conformers of one molecule can occupy separate slots. To request
distinct molecules, use `level="molecule"` or CLI `--level molecule`:

```python
mols = r3.screen(
    "query.sdf", corpus, backend="cpu", k=100, level="molecule",
)
mols.molecules(query=0)              # Up to 100 distinct molecules
mols.aligned_poses(query=0, n=10)    # The winning conformer for each molecule
```

A molecule's score is its best conformer's score, with ties resolved by the
lowest conformer ID. Deduplicating a conformer top-K cannot recover the true
molecule top-K: conformers of one molecule may have displaced other molecules
from the retained list. Use molecule-level results for compound enrichment
statistics.

For a conformer-level result, `hit_molecules()` returns one metadata row per
hit, so molecules can repeat. `distinct_molecules_in_hits()` reports which
molecules are present and how many there are. `molecules()` requires a
molecule-level result.

Choose `k` with the corpus size in mind. Retaining 100,000 of one billion
conformers selects the top 0.01%; retaining the same number from 100 billion
selects the top 0.0001%. These searches can have different score thresholds.

### Full score matrix

For small corpora, `score_matrix()` returns all pair scores:

```python
dense = r3.score_matrix("query.sdf", corpus, backend="cpu")
dense.scores[0, 0]     # First query against conformer dense.gids[0]
```

The array has shape `[n_queries, n_conformers, 20]` with float32 elements.
Each pair requires 80 bytes; eight queries against one million conformers
require 640 MB for scores alone. The default size guard is 4 GiB, adjustable
with `max_bytes`.

### Score columns and poses

Score columns are named in `roshambo3.scores`:

| Columns | Contents |
|---|---|
| 0–2 | Tanimoto: combination, shape, color |
| 3–4 | Shape overlap `V_AB`, color overlap |
| 5–8 | Shape self-overlaps `V_AA`, `V_BB`, then query and library color self-overlaps |
| 9–12 | Pose quaternion |
| 13–15 | Translation |
| 16–19 | Start-mode quaternion |

The pose transforms the library conformer. `aligned_poses()` applies that
transform, then the reported start-mode rotation. Field paths and CPU
analytic search fold the start rotation into the pose, leaving an identity
start quaternion. CUDA analytic search reports it separately. See
[ARCHITECTURE.md](doc/ARCHITECTURE.md) for coordinate and pose conventions.

## Corpus inspection and validation

```bash
roshambo3 info corpus --buckets
roshambo3 verify corpus
roshambo3 verify corpus --tier deep
```

`verify` supports three tiers:

| Tier | Checks |
|---|---|
| `open` | Descriptor, group headers, and block directories |
| `scan` (default) | Structural consistency, including molecule IDs and run tables |
| `deep` | Decode every block; recompute self-overlaps and re-perceive features for sampled molecules |

Add `--all` to deep validation to check every molecule's self-overlaps and
re-perceive its features. Use `--workers N` to control parallel deep
validation.

## Input limits

Preparation rejects molecules that exceed format limits and records them in
the rejection list. Coordinates are never clipped and conformer lists are
never silently truncated. Conflicting molecule identities or inconsistent
records can stop the build, as described under input requirements.

| Quantity | Limit |
|---|---|
| Heavy atoms per conformer | 1–255 |
| Conformers per molecule | 1–4,096 |
| Color features per type and per molecule | At most 255 |
| Color feature types | 1–6 |
| Atoms contributing to one color feature | 1–16 |
| Stored coordinates | Finite; int16 grid from −128 to +127.99609375 Å in the prepared frame |
| External identifier | At most 256 UTF-8 bytes; no control or bidirectional formatting characters |
| SMILES | At most 4,096 bytes of printable ASCII |

See [FORMAT.md](doc/FORMAT.md), section 14, for the complete limits.

Queries are not subject to the corpus's 255-heavy-atom limit. They must have
finite coordinates representable on the same int16 wire grid. Query
preparation snaps coordinates to that grid so all backends score the same
quantized geometry.

CUDA also bounds the padded query width using the device's shared-memory
budget. For shape-only queries, the calculated limits are 14,621 atoms at
102,400 bytes, 14,036 at 98,304 bytes, and 7,014 with the 49,152-byte fallback
budget. These are representation bounds; throughput can fall well before
they are reached. The panel is padded to its widest query, including color
features. CPU and Metal skip this CUDA-specific shared-memory check and
apply their own backend resource checks.

If a launch setting is unsupported, the error identifies the setting and its
supported value. Use the supported configuration for the selected backend.

## Multiple CUDA devices

To partition a screen, assign disjoint corpus groups to separate engines and
merge their top-K results. Group boundaries preserve whole molecules. This
example performs a shape screen on two devices:

```python
import roshambo3 as r3
from roshambo3 import Engine, R3gSource, ScreenConfig, TopK

corpus = r3.open_corpus("corpus")
queries = r3.load_queries("query.sdf", backend="cuda")
n_groups = len(corpus.group_table)
parts = []
for device_id in range(2):
    groups = list(range(device_id, n_groups, 2))
    if not groups:
        continue
    engine = Engine(backend="cuda", device_id=device_id)
    try:
        _, result = engine.run(
            R3gSource(corpus, groups=groups), queries, ScreenConfig(),
            TopK(k=1000),
        )
        parts.append(result)
    finally:
        engine.close()

merged = TopK.merge(parts, k=1000)
```

`merged` contains one `(scores, ids)` pair per query. Each `Engine.run()`
blocks until its result is ready, so this loop runs devices sequentially and
does not provide a multi-GPU speedup. Concurrent orchestration is not supplied
by this example.

Merging preserves the global ranking of the scores computed by the individual
runs, using the same ranking column and tie-break. Partitioning does not alter
that reduction, but different GPU architectures can produce different
per-pair scores and poses.

## Reproducibility

Results are ranked by descending score, with ties resolved by ascending
conformer ID. Float32 ties use this rule without a tolerance.

Scores, optimizer trajectories, and ranked output are not guaranteed to be
identical across GPU architectures, backends, toolchains, or numerical
policies. Accurate analytic final scoring does not remove differences in the
poses selected during search. See [NUMERICS_POLICY.md](doc/NUMERICS_POLICY.md)
for measured comparisons and numerical contracts.

The optimizer defaults to 25 steps. Step count and translation learning rate
are coupled: setting `steps` derives `lr_t` from the measured budget curve.
An explicitly supplied pair outside that curve is rejected.

For reported measurements, record `r3.get_build_info()`, the scoring
configuration, corpus and query identity, and operating conditions. Follow
[BENCHMARKING.md](doc/BENCHMARKING.md) for shared-machine locks and timing
requirements.

## Data and the web application

No prepared corpus is distributed. To reproduce the paper's LIT-PCBA corpus
and PANEL-8 queries, see `scripts/LITPCBA.md` and its driver
`scripts/build_litpcba.py` in the
[paper repository](https://github.com/kimiatx-oss/roshambo3-paper). The conformer
generation step requires nvMolKit and a CUDA GPU; the final corpus is built
with `roshambo3 prepare --color`.

The optional [lab web application](doc/WEB_APP.md) provides query upload,
SMILES conformer generation, screening, pose overlays, and exports. It is
intended for trusted lab networks.

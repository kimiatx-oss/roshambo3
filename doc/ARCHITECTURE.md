# Architecture

> **Status.** This describes the current CUDA, CPU, and Metal implementations.
> Performance numbers below are historical CUDA pipeline measurements, not
> throughput claims for every backend or configuration.

Data movement and reduction are part of the screening architecture. In the
historical D5 measurements, host decode ran at ~7,100 conformers/s/core and host
preparation at ~8,300, against a kernel roughly 12x faster than both; the CPU
bound that pipeline while GPU DRAM sat at 1.2-2.5% of peak. Those measurements
explain the CUDA staging and device top-K design. CPU worker-owned screening
and Metal have their own execution and reduction paths.

## Vocabulary

Pinned here because two of these words have a common alternative that means
something else in this domain, and because one of them was used inconsistently
during the port.

| term | is | is not |
|---|---|---|
| **query** (also *reference*) | the small set you search **with** -- one, or a few dozen, conformers. `molA` in the kernel, `n_q`, `QuerySet`. **This is the molecule rasterized into the field.** | the thing being searched |
| **library** | the large corpus being searched -- 1e8 to 1e11 conformers. `molB` in the kernel, `n_t`, what a `Source` yields as `ConformerBlock`s. | |
| **target** | **not used.** In this field a target is the protein. It also collides with CMake targets and with dataset-tier build targets. | a library molecule |
| **conformer** | one 3D geometry. The unit of everything: ids, buckets, blocks, top-K rows. | a molecule |
| **molecule** | a set of conformers sharing a `mol_id`. All of them share a heavy-atom count, which is what lets bucketing keep them together. | a conformer |
| **gid** | global conformer id: a **position** in library order. | a stable key -- it changes if the corpus is rebuilt differently |
| **tier** | a level of the conceptual Tier-0/1/2/3 filter-and-score ladder below. | where coordinates live between optimizer steps -- that is `CoordResidency` |

### Which molecule moves

Worth stating outright, because it is asymmetric and easy to get backwards:

- the **library molecule** is what the optimized transform is applied to;
- **start modes** select 1, 4, or 10 canonical relative orientations, but the
  backend determines how those orientations are represented.

| path | representation of each start | reported start quaternion |
|---|---|---|
| CUDA analytic | explicitly rotate the query; optimize a library pose against it | separate from the pose |
| CUDA field | keep the query fixed; fold the inverse start rotation into the effective library pose | identity |
| CPU analytic and field | keep the query fixed; fold the inverse start rotation into the effective library pose | identity |
| Metal field | keep the query fixed; fold the inverse start rotation into the effective library pose | identity |

For the fixed-query paths, if `R` is the start rotation and `Q_tilde` is the
optimizer's rotation, the effective rotation is `P = Q_tilde R^T`.
`Q_tilde` starts at identity; the translation is optimized in the fixed query
frame. This lets all starts share the same query field. A field is prepared per
query and field geometry/storage policy, including lattice spacing and padding,
and can be reused across pairs and calls while its binding remains valid.
See [CUDA's start conjugation](../src/roshambo3/backend/_cuda_src/kernels.cu),
[CPU pose search](../src/roshambo3/backend/_cpu_src/score.hpp), and
[Metal pose search](../src/roshambo3/backend/_metal_src/kernels.metal).

`ScreenResult.aligned_poses` applies the reported pose to library coordinates,
then applies the reported start quaternion. That second rotation is identity
when the start is already folded into the pose. The result is in the query's
**prepared frame**; returning to its original input frame additionally requires
the query's preparation transform. `tests/test_reference.py` checks the CUDA
analytic convention against independently recomputed scores; the CPU and field
paths have their own pose and final-score contract tests.

The overlap gradient accumulated by the scoring kernels is **negated**, so
subtracting it ascends overlap. Production gradients use reduced coordinates
and are normalized by the channel's self-overlap sum `S_AA + S_BB`.
Combination optimization additionally weights the channels by the Tanimoto
derivatives and mixing parameter. The unit-magnitude negative-gradient check in
`tests/test_reference.py` uses an unnormalized diagnostic call, not the production
normalization. Rotation components are space-frame half-angle generators from
cross products, not derivatives with respect to four independent quaternion
components. `ADAGRAD_EXP`, the default update rule, composes an exponential-map
rotation with the current quaternion.

## The tier ladder

The tier names describe a conceptual cascade, not four active screening stages:

| tier | role | current screening implementation |
|---|---|---|
| 0 | invariant bounds | no active admission filter in the engine |
| 1 | descriptor filter | no active admission filter in the engine |
| 2 | grid pose search | precomputed query field; shape work is `O(N_B)` per step |
| 3 | analytic scoring | `O(N_A*N_B)` shape pair sum at a selected pose; also available throughout analytic search |

Ordinary engine launches score every query against every library conformer in
the source block. A field search chooses a pose and the reported scores are
analytic evaluations at that pose. How many starts receive an analytic final
evaluation depends on the path and final-start policy described below.

Every shape atom shares one exponent. With `ALPHA` as defined in
`roshambo3.numerics` (the pair coefficient, equal to half the per-atom width),
`F(r) = sum_i exp(-ALPHA * |r - a_i|^2)` gives the reduced overlap as
`S_AB = sum_j F(P b_j + t)`. The physical overlap is
`V_AB = SHAPE_PREFACTOR * S_AB`. This identity makes a query field possible;
lattice sampling, prefiltering, finite storage precision, and interpolation
approximate its evaluation during search. **Analytic final scoring removes the
field approximation from the reported value at the selected pose.** It does not
remove changes to the pose or winning start caused by approximate search.

**`CoordResidency` is not a tier.** It names where a library's coordinates live
between optimizer steps in CUDA -- shared memory, global, global through
`__ldcg` or `__ldcs`, or registers. It has nothing to do with the ladder above.
ROSHAMBO2 called it `grid_tier`, a name that invited exactly that confusion.

## Pose search and analytic final scoring

CUDA's `optimize_pairs_field` and Metal's field optimizer perform pose search
and analytic final scoring in the same kernel launch. CPU evaluators likewise
perform final scoring as part of the native scoring call. The analytic paths
evaluate pair sums throughout the search instead of sampling a shape field.

CUDA and Metal field searches select a start using its converged search
objective, then evaluate the winner analytically for reporting. Prepared CPU
worker screening defaults to the same winner-only policy:
`CPUWorkerOptions(final_start_policy="winner")`. CPU `"all_start"` controls
and unsupported/unsafe combination routes retain accurate all-start selection.
The CUDA analytic path also evaluates converged starts analytically. These
selection policies can choose different poses; analytic reporting does not
imply that every start was ranked by an analytic score.

Three objective modes are defined in [the color contract](COLOR_CONTRACT.md):

| mode | pose search | reported color |
|---|---|---|
| shape-only | shape objective | zero |
| shape-pose color rescore (`AT_POSE`) | same shape trajectory and winning start as shape-only | analytic at the selected pose |
| combination (`IN_GRADIENT`) | mixed shape/color Tanimoto objective | analytic at the selected pose |

The shape-pose color-rescore contract requires the reported pose and shape
score to be bit-identical to shape-only on the same inputs and numerical policy.
Color is evaluated at the winning pose; some implementations also evaluate it
at other converged starts. Combination search includes color in each optimizer
step and can change both the pose and winning start.

**Color fields are supported during combination search.** They use separate
feature planes with their own widths and prefactors, rather than putting all
color into the scalar shape field. CUDA supports eligible color planes through
`color_field_min_sources`; CPU has experimental per-query color-field policies
and prepared handles. CPU and CUDA default to analytic search color. Metal's
default policy serves eligible diagonal planes 1/2 at a query count cutoff of
12, with analytic fallback for unsupported cases, bounds, and memory budgets.
Its default color spacing is 0.25 A. Explicitly selecting all-analytic color is
also supported. See [configuration](../src/roshambo3/engine/config.py),
[CPU color fields](../src/roshambo3/backend/cpu.py), and
[Metal policy](../src/roshambo3/backend/metal.py).

Color fields and approximate optimizer exponentials affect search only. Normal
reported shape and color scores use accurate analytic pair sums at the selected
pose. "Exact" here distinguishes analytic final scoring from search
approximations; it does not mean infinite precision or identical arithmetic
across backends. [NUMERICS_POLICY.md](NUMERICS_POLICY.md) defines those limits.

## The two protocols

`Source` and `Sink` are the ordinary engine's input and output extension points.
Backend selection, evaluator choice, optimizer settings, and numerical policies
are separate configuration concerns. The protocol outline is:

```python
class Source(Protocol):
    """Where conformers come from."""
    def blocks(self) -> Iterator[ConformerBlock]: ...

    @property
    def n_conformers(self) -> int: ...

class Sink(Protocol):
    """What happens to scores."""
    def prepare(self, n_queries: int, device_id: int,
                max_block_conformers: int = 0) -> None: ...
    def add_block(self, block, scores: Optional[np.ndarray]) -> None: ...
    def finalize(self) -> Any: ...

    @property
    def wants_dense(self) -> bool: ...
```

A `ConformerBlock` carries **coordinates and a count vector, and no type
plane**. The count vector is `[n_conformers, n_features]` uint8 indexed by
feature type id: column 0 is the real atom count -- there is no separate
`n_real` array, because that is the same number -- and columns 1.. are the
per-feature counts. `n_real` is a view of column 0. Counts are not tiled;
shape-only sources carry one count column. CUDA packs the counts into one
64-bit word per conformer, including a redundant color total in byte 7, and
recovers per-slot types from contiguous feature runs. A library type plane is
not sent to the scoring backend.

GPU source coordinates are int16 wire records at a step of `2^-8` Angstrom,
dequantized and scaled into the reduced frame for scoring. `tile_mode` identifies
the interleaved or planar GPU tile layout; the CPU source uses untiled FP32
coordinates (`tile_mode=0`). `color_offset` identifies where color runs begin,
and `L_shape` keeps field-spacing policy independent of padded color width.
Molecule-level reduction additionally uses `mol_id` and molecule-boundary
metadata. See [the block definition](../src/roshambo3/engine/source.py).

`R3gSource` (`engine/r3g_source.py`) reads prepared format-3 corpora. Corpus
order defines global conformer IDs, and each block carries its `gid_base`.
Reassigning IDs by changing corpus order can change tie-breaks. Processing
blocks in a different completion order while preserving their IDs does not;
the CPU worker route explicitly schedules physical blocks independently of
corpus order.

A sink may also implement `before_block(block)` when it needs metadata staged
before scoring, as molecule-level device reduction does. With a device sink,
the backend can consume its device score block directly and pass `None` to
`add_block`; sinks with `wants_dense=True` receive host scores.

`Sink` is:

| sink | what it does |
|---|---|
| `TopK` | device-resident running top-K; the dense block never crosses PCIe |
| `HostTopK` | exact host reduction for ordinary CPU scoring |
| `MoleculeTopK` | rank each molecule by its best conformer, preserving that conformer's score row and pose |
| `DenseScores` | returns the raw score matrix with no reduction; **size-guarded** |
| `Callback` | calls `fn(block, scores)` with each complete host score block |

`Engine.make_topk` chooses the appropriate backend implementation, including
Metal's device reducer. `DenseScores` assembles a
`[n_queries, n_conformers, 20]` result in corpus order. It does not simply
concatenate the query-major rows from successive blocks.

### Score collection by backend

On CUDA, scoring writes a device `[n_rows, 20]` block. A separate grid-stride
top-K kernel reads it, admits rows above a running threshold with one
`atomicAdd`, and periodically reselects with two stable radix sorts. Dense
launches derive query/library row mappings from the launch dimensions; sparse
launches supply explicit maps. `gid_base` maps block-local library rows to
global conformer IDs. Molecule ranking first reduces conformers to molecule
winners. Metal also reduces device score blocks, using its own kernels.

In the historical CUDA D5 measurements, fusing scoring and top-K would have
saved writing and re-reading ~328 GB each way, about 0.4 s against a 135 s wall,
on a device whose DRAM was 1.2-2.5% utilized. That saving was under 1% of wall
time, at the cost of coupling the reducer to each scoring kernel. The measured
1.6x gain from device top-K on the 1e8-conformer case came from eliminating
64 GB of device-to-host transfer and host reduction.

`DenseScores` requires host materialization of every score row and carries a
size guard. On CUDA that reintroduces the device-to-host transfer that device
top-K avoids.

The **CPU worker-owned route** has a different collection boundary. Persistent
workers decode physical format-3 blocks, prepare contiguous FP32 coordinates,
score, and reduce into bounded worker-local top-K state. Combination scoring
can emit completed rows directly to a native row sink (`row_sink_only=True`),
without materializing a dense result block. Shape and shape-pose color-rescore
routes currently produce block scores and ingest them into the same collector.
Finalization merges worker states under the canonical ranking order. This route
supports conformer and molecule top-K requests; dense and callback sinks remain
on the ordinary engine path. See
[worker-owned screening](../src/roshambo3/engine/engine.py) and
[the native collector](../src/roshambo3/backend/_cpu_src/worker_local_topk.hpp).

Prepared CPU worker queries retain field evaluators and verified self-overlap
contexts across compatible calls. Library self-overlaps can reuse verified
stored shape values and prepared current-matrix color terms; explicit
`self_overlap_policy="recompute"` retains recomputation. Cache bindings and
reported fallback statistics distinguish those routes.

## The score row

A result row is fixed at 20 float32 columns (80 bytes): 9 scalars, 7 pose
parameters, and the start-mode quaternion. The pose represents six rigid-body
degrees of freedom using a unit quaternion and three translation components.

| cols | contents |
|---|---|
| 0-2 | tanimoto: combination, shape, color |
| 3-4 | overlap volume `V_AB`, overlap color |
| 5-8 | self-overlaps: `V_AA`, `V_BB`, color query, color fit |
| 9-12 | pose quaternion |
| 13-15 | translation in Angstrom |
| 16-19 | start-mode quaternion |

`roshambo3.scores` defines `N_SCORE_COLUMNS = 20`, named indices/slices, and
`SCORE_DTYPE`, a structured view of the same row. Public scoring wrappers enforce
that width even though some reducer internals accept a runtime column count.
Columns 16-19 are identity when the start is folded into the pose; columns
2/4/7/8 are zero for shape-only scoring. These columns remain in the result
layout; a compact score row is not currently the public format.

## Determinism, and what multi-GPU means

For a **fixed set of score rows and conformer IDs**, exact top-K selection is
independent of arrival order and partitioning. CUDA's `topk.cuh` explains the
argument: at a fixed threshold the admitted set is order-independent even though
`atomicAdd` slot assignment is not; stable sorting by ID and then score gives a
canonical selection; and any threshold at or below the true k-th score retains
every possible winner. CPU worker-local reduction likewise merges exact partial
top-K states without using completion order as a tie-break.

That reduction property does **not** establish bit-identical scoring across
arbitrary kernel, launch, numerical-policy, compiler, or hardware changes.
For example, CUDA update-lane choices can change contraction under
`ADAGRAD_EXP`, register residency is not promised bit-identical to every other
residency, and CPU evaluator/optimizer choices have their own arithmetic.
Scheduling and batch invariance must be checked within the selected scoring
contract. Cross-architecture scores need not match bitwise. See
[the backend configuration contracts](../src/roshambo3/backend/spec.py) and
[NUMERICS_POLICY.md](NUMERICS_POLICY.md), especially sections 4h, 8, and 8b.

Multi-GPU screening uses one engine and top-K state per device, then merges
results on the host under the canonical ranking order. Shards must preserve
global conformer IDs. Changing GPU count preserves the answer when each pair's
scoring arithmetic is unchanged; assigning pairs to heterogeneous devices can
change scores and rankings. The numerical policy records measured homogeneous
and heterogeneous comparisons and their scope.

## Ranking has exactly one tie-break

Descending score, ties broken by ascending conformer id -- `numerics.rank_order`,
and the identical order computed on the device by `topk_select`. Float32 ties are
common at scale; ROSHAMBO2's host reducer cut with `argpartition` on score alone
and resolved ties arbitrarily, which flipped one row in 800,000 on D5. An
unstable sort here is a correctness bug, not a performance detail.

Molecule-level ranking first chooses each molecule's best conformer, using the
same ascending global conformer ID to resolve score ties, then ranks molecules
by those winning rows and IDs. Separately, equal objective values across a
pair's starts choose the earliest canonical start index.

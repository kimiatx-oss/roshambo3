# Apple Silicon GPU backend (experimental)

The native Metal backend supports macOS 15+ on Apple Silicon. It implements
shape screening, shape-pose color rescore, and combo optimization with conformer
or molecule top-K. Select it explicitly; the package default remains CUDA and
an unavailable backend never silently falls back.

```sh
python -m pip install . --config-settings=cmake.define.ROSHAMBO3_CUDA=OFF \
  --config-settings=cmake.define.ROSHAMBO3_METAL=ON
roshambo3 screen query.sdf corpus --backend metal --mode combo --mixing 0.5 --level molecule
```

`ROSHAMBO3_METAL=AUTO` builds on native arm64 macOS with a compatible SDK;
`OFF` omits the extension. Command Line Tools suffice: bundled shader sources
compile through Metal at runtime. Source distributions and wheels include those
sources. `metal.available()` checks the actual device, not just extension import.

```python
import roshambo3 as r3
from roshambo3.engine import Engine, ScreenConfig
from roshambo3.backend import ColorPlacement
from roshambo3.pharmacophore import PharmacophoreGenerator

engine = Engine(backend="metal")
try:
    hits = r3.screen("query.sdf", "corpus", engine=engine, level="molecule",
        color_generator=PharmacophoreGenerator(),
        config=ScreenConfig(color=True, mixing=.5,
            color_placement=ColorPlacement.IN_GRADIENT))
finally:
    engine.close()
```

`screen`, `score_matrix`, and query constructors accept `backend="metal"`.
`Engine.make_topk(k, level=..., rank_col=...)` constructs the appropriate sink.
Caller-owned Engines retain query and pipeline state; owned API Engines close
after copying results. Reuse is serialized within each Metal context. Independent
Engines own independent contexts. `close()` releases retained state and permits
subsequent Engine reuse with a new context.

## Numerical contract

Production algorithms are FIELD and ADAGRAD_EXP, with start modes 0/1/2,
existing optimizer travel-budget validation, and FP32 accumulation. Unsupported
algorithms and explicit CUDA tuning settings fail before corpus decoding.

The default shape field builds FP32 value/gradient nodes with the existing
Laplacian prefilter, then stores RGBA16Float textures for native trilinear
filtering. `field_sampler="manual"` retains the FP32 eight-corner reference. The shared rigid
pose scales the FP32 rotation matrix before applying it to physical library
coordinates. Four cooperating lanes use a fixed XOR-2/XOR-1 reduction. Different
threadgroup sizes do not select different reduction arithmetic.

Optimizer color Gaussians use `metal::fast::exp2`; field construction,
self-overlaps, and reported analytic final shape/color scores use precise
functions. General interaction matrices retain their directed entries. Widths,
heights, and reduced-frame decay/amplitude coefficients are prepared once per
query context, outside the optimizer loop. Canonical diagonal matrices use
stable per-type query ranges. Rescore preserves the shape-selected pose.

`metal.create_context(fast_search_exp2=False)` supplies a precise-search control
for experiments; the shipping Engine policy is fast search with precise final
scoring. Context timing records identify the effective policy. This does not
promise bitwise optimizer trajectories across Metal, CPU, CUDA, OS releases,
or GPU families.

Results retain all 20 columns, with physical translations and a folded start
rotation (identity in the final start-quaternion columns). Rank by descending
score and ascending 64-bit GID without soft ties. Device selection uses stable
parallel merge sorting of row indices rather than CUDA's radix implementation;
its comparator and bounded running molecule reduction are checked against host
sorting, including molecules split across blocks. Only retained hits cross the
host boundary in ordinary screening.

## Default color and texture policy

Combo search specializes diagonal interactions and factors color torque in FP32.
Dense hydrophobe and acceptor planes use RGBA16Float fields at 0.25 Angstrom,
with a minimum of 12 sources per plane. Other planes, general interaction
matrices, out-of-bounds samples and fields over the memory budget stay analytic.
Self-overlaps and all reported final scores remain precise analytic sums.

`ScreenConfig.color_field_min_sources=None` inherits this backend policy;
`0` explicitly disables color fields. CPU/CUDA keep their effective analytic
color defaults. `color_field_spacing_a` overrides color spacing independently.

`Engine(backend="metal", backend_options={...})` accepts explicit controls:
`field_sampler="manual"` for FP32 software shape fields,
`color_field_min_sources=0` for analytic color, and
`fast_search_exp2=False` for precise optimizer exponentials.
`texture32` requires M3 or newer; the default `texture16` does not.
Texture and reduction changes can affect optimizer trajectories and close ranks.

Contexts cache query fields and pipelines. Color fields have a 128 MiB budget;
compact bindings and split dispatches respect the 128-texture limit. Inputs use
owned shared buffers and complete command groups before staging reuse.
Cancellation is checked between blocks. No CPU/GPU overlap is claimed.

## Validation and benchmarking

Native GPU checks require a visible Apple GPU and:

```sh
ROSHAMBO3_REQUIRE_METAL=1 python -m pytest tests/test_metal.py tests/test_metal_cli.py tests/test_web.py
```

On other platforms, native execution tests skip; backend discovery, validation,
and portable regression tests remain runnable. `metal.build_stamp()` records
compiler, SDK, source and shader hashes, loaded binary hash and numerical policy.
The manual Apple GPU workflow requires a trusted self-hosted Metal runner.

The supplied M4 Pro installed-wheel validation reports 1,777 passes and 735
skips, with strict optimized-objective ranking gates and 2,328 independent
fixed-pose audits. These receipts were supplied with the port and were not
independently rerun during Linux release integration. M1/M2 hardware was not
measured. The backend retains its experimental status.

`bench.metal` and related harnesses support matched scientific controls,
release-candidate replay and diagnosis. Frozen public inputs, complete studies,
failed gates, timing limitations, supplied receipts, full score captures and
timing outputs are not distributed with this repository.
No new paper claims are established by importing this backend.

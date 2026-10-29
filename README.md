# ROSHAMBO3

3D molecular shape and pharmacophore screening with CUDA and native CPU
implementations for x86 and ARM, including native Apple Silicon. An experimental
[Metal GPU backend](doc/METAL.md) is available on Apple Silicon macOS 15+.
The package prepares size-bucketed, quantized
corpora, optimizes rigid poses, and returns ranked hits. It supports shape-only,
shape optimization with color rescore, and combination optimization when the
corpus and query carry compatible color features.

Version 1.0.0. See `RELEASE_STATUS.md` for what has and has not been validated.

The method and its measurements are described in *Linear-time Shape Alignment
with ROSHAMBO3* (ML4Molecules workshop, NeurIPS 2026); the paper's source and
data are at <https://github.com/kimiatx-oss/roshambo3-paper>. To cite this
software, see `CITATION.cff`.

```bash
python -m pip install .
python examples/make_demo.py --out demo
roshambo3 prepare demo/library.sdf -o demo/corpus --workers 1
```

```python
import roshambo3 as r3
hits = r3.screen("demo/query.sdf", "demo/corpus", k=3, backend="cpu")
hits.hit_molecules(query=0, n=3)
hits.aligned_poses(query=0, n=3)
```

The CLI and optional lab web interface are included. See `INSTALL.md`,
`USER_GUIDE.md`, `DEV_GUIDE.md`, and `doc/WEB_APP.md`. Apple build and
benchmark instructions are in `doc/APPLE_SILICON.md`. Numerical and scoring
contracts live in `doc/`. MIT license; retained attribution is in `NOTICE`.

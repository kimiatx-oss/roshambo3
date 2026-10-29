# Lab web application

Install the optional dependencies with `python -m pip install '.[web]'`.
Copy `web/example.toml`, register a prepared corpus, and start:

```bash
roshambo3 web --config web/example.toml
```

The application offers query upload or SMILES conformer generation, screening,
ranked results, pose overlays, CSV, aligned SDF, and manifest exports. Compatible
color corpora enable color rescore and combo modes. Overlays and aligned SDF
exports read each hit's graph from the corpus itself: a format-3 corpus stores
every molecule's SMILES with atom *i* at coordinate slot *i* (`doc/FORMAT.md`),
so no separate structure index is needed.

Run one API process per job directory. This is a shared lab workspace with shared
job visibility and no account system. Bind to localhost or use a trusted lab
reverse proxy. Keep GPU lock configuration consistent with other workloads.
Set `[server] backend = "cuda"` (the default), `backend = "cpu"`, or experimental `backend = "metal"`.
Metal requires Apple Silicon macOS 15+; see [the Metal contract](METAL.md).
CPU jobs use the native worker-owned scan for shape, rescore, and combo, including
molecule top-K. CPU progress reports completed work; GPU progress reports submitted
work. Both retain cancellation and supervisor timeouts. CPU jobs acquire the
host CPU lock (`ROSHAMBO3_CPU_LOCK`); CUDA jobs acquire the physical GPU lock; Metal jobs lock the device registry ID.

Metal inherits the backend's half-texture policy when `metal_field_sampler` is
omitted: texture16 shape and eligible color planes at cutoff 12 / 0.25 Angstrom.
An explicit `metal_field_sampler = "manual"` selects FP32 software shape sampling;
`"texture32"` needs M3+. Job manifests record the resolved shape and color policy.
Texture search arithmetic can change close rankings; see [METAL.md](METAL.md).

CPU resource settings are `cpu_workers` (omitted means automatic), `cpu_qos`
(default `"inherit"`; explicit QoS requires macOS), `cpu_memory_budget_bytes`,
and `cpu_topk_state_budget_bytes` (both optional positive limits). CPU requires
`device = 0`. These settings are administrator-controlled and recorded with the
job; browser requests cannot override them.

Browser assets are packaged under `src/roshambo3/web/static`. Rebuild with
`npm ci` and `npm run build` from `web/` using Node 22 or later. Browser tests use
`npm test`; Python web tests additionally need `httpx`. A release needs a real
CUDA screen, overlay, and export check in all three modes on the deployment host.

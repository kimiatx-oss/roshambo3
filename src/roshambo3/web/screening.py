"""Adapter: three objectives, molecule reduction, query ensemble max pooling."""
from dataclasses import asdict
import json
import gc
import resource
import time
import numpy as np
from rdkit import Chem

from roshambo3.scores import TANIMOTO_COMBINATION, TANIMOTO_SHAPE


def merge_hits(parts, mol_ids, k, rank_col, conf_indices=None):
    """Exact top-K of the per-molecule maximum over query conformers."""
    if conf_indices is None:
        raise ValueError("format-3 result merging requires conformer identities")
    candidates = {}
    for qi, ((scores, gids), mids, cis) in enumerate(zip(parts, mol_ids, conf_indices)):
        for row, gid, mid, ci in zip(scores, gids, mids, cis):
            # Durable identity joins output across builds; ranking still follows
            # the established score/gid tie policy.
            key = (-float(row[rank_col]), int(gid), qi)
            if int(mid) not in candidates or key < candidates[int(mid)][0]:
                candidates[int(mid)] = (key, row, int(gid), qi, int(ci))
    best = sorted(candidates.items(), key=lambda entry: entry[1][0])[:k]
    result = (np.asarray([v[1] for _, v in best], dtype=np.float32).reshape(-1, 20),
              np.asarray([v[2] for _, v in best], dtype=np.int64),
              np.asarray([v[3] for _, v in best], dtype=np.int32),
              np.asarray([mid for mid, _ in best], dtype=np.int64))
    return result + (np.asarray([v[4] for _, v in best], dtype=np.int64),)


def run_screen(settings, ident):
    from .store import Store
    from .gpu import resource_lock
    store = Store(settings.jobs)
    directory = settings.jobs / ident
    job = store.get(ident)
    request = job["request"]
    db = settings.databases[request["database"]]
    canceled = lambda: (directory / "cancel").exists()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = max(soft, 4096) if hard == resource.RLIM_INFINITY else min(hard, max(soft, 4096))
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    store.update(ident, "waiting_cpu" if settings.backend == "cpu" else "waiting_gpu")
    with resource_lock(settings.backend, settings.device, canceled) as uuid:
        from roshambo3 import load_queries, get_build_info
        from roshambo3.backend import ColorPlacement, PoseSearch
        from roshambo3.engine import CPUWorkerOptions, Engine, MoleculeTopK, ScreenConfig, conformers_per_launch
        from roshambo3 import open_corpus
        from roshambo3.format import Catalog
        from .storage import BoundedCatalog, BoundedSource
        from roshambo3.results import QueryHits, ScreenResult
        from roshambo3.numerics import numerics_stamp
        corpus = open_corpus(str(db.path))
        color = request["mode"] != "shape"
        generator = None
        if color:
            from roshambo3.format.r3g.descriptor import isolated_generator_from_descriptor
            generator = isolated_generator_from_descriptor(corpus.descriptor["color"])
            from roshambo3.definitions import check_corpus_color
            check_corpus_color(corpus, generator)
        mols = list(Chem.SDMolSupplier(str(directory / "query.sdf"), removeHs=False))
        queries = load_queries(mols, color_generator=generator, backend=settings.backend)
        cfg = ScreenConfig(pose_search=PoseSearch.FIELD, color=color,
                           mixing=request["mixing"] if color else 0.0,
                           color_placement=ColorPlacement.IN_GRADIENT if request["mode"] == "combo" else ColorPlacement.AT_POSE)
        rank_col = TANIMOTO_COMBINATION if color else TANIMOTO_SHAPE
        if settings.backend == "cpu":
            from dataclasses import replace
            cfg = replace(cfg, cpu_worker=CPUWorkerOptions(
                workers=settings.cpu_workers, cpu_qos=settings.cpu_qos,
                memory_budget_bytes=settings.cpu_memory_budget_bytes,
                topk_state_budget_bytes=settings.cpu_topk_state_budget_bytes))
        per_launch = conformers_per_launch(queries.n_conformers, request["k"])
        if settings.backend == "metal":
            from roshambo3.backend.metal import launch_conformers
            widths = [int(g["L_field_width"]) for g in corpus.group_table]
            per_launch = launch_conformers(queries.n_conformers, request["k"], max(widths), device_id=settings.device)
        source_options = {
            "color": color, "tile": settings.backend != "cpu", "workers": 2,
            "prefetch_depth": 1, "conformers_per_launch": per_launch,
        }
        if color:
            source_options["color_generator"] = generator
        source = BoundedSource(corpus, **source_options)
        engine_options = ({"backend_options": {"field_sampler": settings.metal_field_sampler}}
                          if settings.backend == "metal" and settings.metal_field_sampler is not None else {})
        engine = Engine(device_id=0 if settings.backend == "cuda" else settings.device,
                        backend=settings.backend, **engine_options)
        from .progress import SubmissionProgress
        started = time.monotonic()
        tracker = SubmissionProgress(source.n_conformers, started)
        store.update(ident, "screening", total_conformers=source.n_conformers,
                     screen_started_at=time.time(), screen_elapsed_s=0.0,
                     submitted_conformers=0, completed_conformers=0,
                     progress_basis="completed" if settings.backend == "cpu" else "submitted", progress=0.0,
                     submission_rate=None, eta_seconds=None)
        def progress(stats):
            update = tracker.update(stats.n_conformers, time.monotonic())
            if update is not None:
                if settings.backend == "cpu":
                    update["completed_conformers"] = update.pop("submitted_conformers")
                    update["completion_rate"] = update.pop("submission_rate")
                store.update(ident, **update)
        try:
            stats, parts = engine.run(source, queries, cfg, engine.make_topk(k=request["k"], level="molecule", rank_col=rank_col),
                                      progress=progress, canceled=canceled)
            store.update(ident, "finalizing", progress=1,
                         submitted_conformers=stats.n_conformers if settings.backend != "cpu" else 0,
                         completed_conformers=stats.n_conformers,
                         screen_elapsed_s=time.monotonic() - started, eta_seconds=None)
            catalog = BoundedCatalog(corpus)
            mids = [catalog.mol_ids(gids) for _, gids in parts]
            cis = [catalog.conf_indices(gids) for _, gids in parts]
            scores, gids, qis, mids, cis = merge_hits(
                parts, mids, request["k"], rank_col, conf_indices=cis)
            meta = catalog.molecules(gids, columns=["ext_id", "smiles"])
            rows = [{"rank": i + 1, "gid": str(int(gid)), "mol_id": str(int(mid)),
                     "conf_idx": int(ci),
                     "query_index": int(qi), "name": meta["ext_id"][i], "smiles": meta["smiles"][i],
                     "shape": float(scores[i, 1]), "color": float(scores[i, 2]),
                     "combination": float(scores[i, 0])}
                    for i, (gid, mid, ci, qi) in enumerate(zip(gids, mids, cis, qis))]
            np.savez(directory / "scores.npz", scores=scores, gids=gids,
                     query_indices=qis, mol_ids=mids, conf_indices=cis)
            # Persist just K coordinates: later browsing never needs a CUDA context
            # or scans through the database. Use the shared pose implementation.
            result = ScreenResult([QueryHits("merged", 0, gids, scores, rank_col, "molecule",
                                             mol_id=mids, conf_idx=cis)], corpus=corpus)
            result._catalog = catalog
            aligned = result.aligned_poses()
            offsets = np.cumsum([0] + [len(x) for x in aligned])
            np.savez(directory / "poses.npz",
                     xyz=np.concatenate(aligned) if aligned else np.empty((0, 3)),
                     offsets=offsets, gids=gids, mol_ids=mids, conf_indices=cis)
            (directory / "hits.json").write_text(json.dumps(rows))
            resolved_options = engine_options.get("backend_options", {})
            if settings.backend == "metal":
                context = engine._context()
                resolved_options = {
                    key: getattr(context, key) for key in (
                        "field_sampler", "color_field_sampler", "color_field_min_sources",
                        "color_field_spacing_a", "color_field_types", "color_field_max_bytes",
                        "specialize_color", "factor_color_torque", "color_range_policy",
                        "fast_search_exp2")
                }
            semantic_fingerprint = corpus.fingerprint
            manifest = {"request": request, "preparation": job["preparation"],
                        "corpus_sha256": semantic_fingerprint, "database": db.id,
                        "build": (asdict(get_build_info()) if settings.backend == "cuda"
                                  else engine.backend.build_stamp()),
                        "resource_identity": uuid, "gpu_uuid": uuid if settings.backend != "cpu" else None,
                        "stats": asdict(stats),
                        "backend": settings.backend, "screen_config": asdict(cfg), "numerics": numerics_stamp(),
                        "backend_options": resolved_options,
                        "ranking": "combination" if color else "shape", "aggregation": "maximum over query conformers",
                        "fd_soft_limit": target, "workers": 2, "prefetch_depth": 1}
            (directory / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2))
            if canceled():
                raise InterruptedError("Canceled")
            store.update(ident, "completed", hits=len(rows), progress=1)
        finally:
            engine.close()
            del source, engine
            gc.collect()

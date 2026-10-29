#!/usr/bin/env python3
"""Frozen bounded screening regression panel core.

Panels are immutable, deterministic 128-conformer extracts. Reference caches are
imported explicitly and remain immutable during routine candidate scoring.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PROTOCOL = "screening-panel-v1"
DEV_TARGETS = ("PUR2", "HMDH", "CXCR4", "FABP4")
EVAL_TARGETS = ("ADA", "AMPC", "MAPK2", "ANDR", "PARP1", "RENI", "EGFR", "HS90A")
ALL_TARGETS = DEV_TARGETS + EVAL_TARGETS
PANEL_SIZE = 128

class Unavailable(RuntimeError):
    pass

def _json_bytes(x):
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def _sha(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def array_sha(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(a.dtype.str.encode()); h.update(b"\0")
        h.update(json.dumps(a.shape, separators=(",", ":")).encode()); h.update(b"\0")
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()

def _immutable_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def config_for(lane, backend):
    if lane not in ("shape", "combo") or backend not in ("cpu", "cuda"):
        raise ValueError("lane must be shape|combo and backend cpu|cuda")
    if backend == "cpu" and lane != "shape":
        raise ValueError("frozen CPU routine is shape-only")
    from roshambo3.optimizer import lr_t_for_steps
    cfg = {"lane": lane, "backend": backend, "pose_search": "ANALYTIC",
           "steps": 100, "lr_q": 0.1, "lr_t": float(lr_t_for_steps(100)),
           "start_mode": 1, "score_column": "t_shape" if lane == "shape" else "t_combination",
           "optimizer_update_rule": "ADAGRAD_EXP"}
    if backend == "cpu":
        cfg.update(threads=1, shape_backend="scalar_analytic",
                   shape_grain="scalar", shape_width=1)
    else:
        cfg.update(device_id=0, coord_residency="SHARED")
    if lane == "combo":
        cfg.update(mixing=0.5, color_placement="IN_GRADIENT",
                   color_stage=0, color_residency="GLOBAL",
                   library_color_layout="merged", library_color_offset=-1)
    return cfg

def panel_paths(directory, target):
    d = Path(directory)
    return d / f"{target}.panel.npz", d / f"{target}.panel.json"

def load_panel(directory, target):
    npz, mp = panel_paths(directory, target)
    if not npz.is_file() or not mp.is_file():
        raise Unavailable(f"panel missing for {target}: {npz}")
    meta = json.loads(mp.read_text())
    if meta.get("protocol") != PROTOCOL or file_sha(npz) != meta.get("npz_sha256"):
        raise ValueError(f"panel file hash/protocol mismatch for {target}")
    with np.load(npz, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    if data["gid"].shape != (PANEL_SIZE,):
        raise ValueError(f"{target} panel must contain 128 rows")
    digest = array_sha(data["coords_i16"], data["counts"], data["gid"],
                       data["mol_id"], data["conf_idx"], data["query_coords"],
                       data["query_types"], data["query_n_atoms"],
                       data["interaction_r"], data["interaction_p"])
    if digest != meta.get("geometry_sha256"):
        raise ValueError(f"panel geometry/identity digest mismatch for {target}")
    return meta, data

def canonical_panel_ids(panel):
    pairs = list(zip(map(int, panel["mol_id"]), map(int, panel["conf_idx"])))
    if len(set(pairs)) != len(pairs):
        raise ValueError("panel contains duplicate (mol_id, conf_idx) identities")
    from roshambo3.metrics import canonical_ids
    return canonical_ids(panel["mol_id"], panel["conf_idx"])

def _hash_key(text):
    return hashlib.sha256(text.encode("utf-8")).digest()

def select_panel_rows(target, gid, mol_id, conf_idx, shape_score):
    """Apply the exact frozen 16+32+32+48 selection rule."""
    gid, mol, conf, score = (np.asarray(x) for x in (gid, mol_id, conf_idx, shape_score))
    if any(x.ndim != 1 for x in (gid, mol, conf, score)) or not (gid.shape == mol.shape == conf.shape == score.shape):
        raise ValueError("selection columns must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(score)) or np.unique(gid).size != gid.size:
        raise ValueError("selection scores must be finite and gids unique")
    if len(set(zip(map(int, mol), map(int, conf)))) != len(mol):
        raise ValueError("selection source has duplicate (mol_id, conf_idx) identities")
    by_score = sorted(range(len(gid)), key=lambda i: (-float(score[i]), int(mol[i]), int(conf[i])))
    best = {}
    for i in by_score:
        best.setdefault(int(mol[i]), i)
    molecules = sorted(best, key=lambda m: (-float(score[best[m]]), m, int(conf[best[m]])))
    if len(molecules) < 80:
        raise ValueError(f"{target}: need 80 distinct molecules; found {len(molecules)}")
    chosen = [(best[molecules[i]], "top16") for i in range(16)]
    chosen += [(best[molecules[i]], "cutoff17_48") for i in range(16, 48)]
    kept_mols = set(molecules[:48])
    hashed = sorted(molecules[48:], key=lambda m: (_hash_key(f"screening-panel-v1:{target}:{m}"), m))
    if len(hashed) < 32:
        raise ValueError(f"{target}: fewer than 32 challenger molecules")
    extra_mols = hashed[:32]; kept_mols.update(extra_mols)
    chosen += [(best[m], "hash32") for m in extra_mols]
    occupied = {i for i, _ in chosen}
    extras = [i for i in range(len(gid)) if int(mol[i]) in kept_mols and i not in occupied]
    extras.sort(key=lambda i: (_hash_key(
        f"screening-panel-extra-v1:{target}:{int(mol[i])}:{int(conf[i])}"),
        int(mol[i]), int(conf[i])))
    if len(extras) < 48:
        raise ValueError(f"{target}: fewer than 48 extra conformers among selected molecules")
    chosen.extend((i, "extra48") for i in extras[:48])
    return [{"gid": int(gid[i]), "mol_id": int(mol[i]), "conf_idx": int(conf[i]),
             "selection_score": float(score[i]), "category": cat} for i, cat in chosen]

def build_panel(target, *, corpora, selection_cache, queries, output):
    from roshambo3.api import load_queries
    from roshambo3.format import Corpus
    from roshambo3.format.catalog import Catalog
    from roshambo3.numerics import WIRE_QUANT_STEP
    from roshambo3.pharmacophore import RDKitPharmacophoreGenerator
    from rdkit import Chem
    corpus_path = Path(corpora) / target
    cache_path = Path(selection_cache) / f"{target}.shape.conf.npz"
    screens_path = Path(selection_cache) / "screens.json"
    query_path = Path(queries) / f"{target}.sdf"
    for p in (corpus_path / "manifest.json", cache_path, screens_path, query_path):
        if not p.exists(): raise Unavailable(f"required input missing: {p}")
    corpus = Corpus(str(corpus_path)); cat = Catalog(corpus)
    if not corpus.merged_color:
        raise ValueError("q2m8 regression panel requires merged color records with per-row offset -1")
    with np.load(cache_path, allow_pickle=False) as z:
        gid_cache, mol_cache, conf_cache, scores_cache = (z[k].copy() for k in ("gid", "mol_id", "conf_idx", "t_shape"))
    cache_rows = {int(g): i for i, g in enumerate(gid_cache)}
    if len(cache_rows) != len(gid_cache) or len(cache_rows) != cat.n_conformers:
        raise ValueError("selection cache must cover every corpus gid exactly once")
    gids_all = np.arange(cat.n_conformers, dtype=np.int64)
    refs_all = cat.locate(gids_all)
    mol_all = np.fromiter((r.mol_id for r in refs_all), np.int64, count=len(refs_all))
    conf_all = np.fromiter((r.conf_idx for r in refs_all), np.int64, count=len(refs_all))
    order = np.asarray([cache_rows[int(g)] for g in gids_all], np.int64)
    if not (np.array_equal(mol_cache[order], mol_all) and np.array_equal(conf_cache[order], conf_all)):
        raise ValueError("cached IDs differ from immutable corpus sidecars")
    selected = select_panel_rows(target, gids_all, mol_all, conf_all, scores_cache[order])
    gids = np.asarray([x["gid"] for x in selected], np.int64)
    refs = cat.locate(gids); loc = [(r.group_index, r.row) for r in refs]
    decoded = cat.coordinates(gids, trim=False)
    width = ((max(int(corpus.groups[g].L) for g, _ in loc) + 7) // 8) * 8
    count_width = max(int(corpus.groups[g].n_features) for g, _ in loc)
    coords = np.zeros((PANEL_SIZE, width, 3), np.int16)
    counts = np.zeros((PANEL_SIZE, count_width), np.uint8)
    for i, ((gi, ri), xyz) in enumerate(zip(loc, decoded)):
        scale = float(corpus.groups[gi].reader.coord_scale)
        if not np.isclose(scale, float(WIRE_QUANT_STEP), rtol=0, atol=1e-12):
            raise ValueError(f"corpus coordinate scale {scale} differs from library wire step")
        q = np.rint(np.asarray(xyz, np.float64) / float(WIRE_QUANT_STEP))
        if np.any(q < -32768) or np.any(q > 32767):
            raise ValueError("selected coordinate exceeds int16 wire range")
        if not np.array_equal(q.astype(np.float32) * np.float32(WIRE_QUANT_STEP),
                              np.asarray(xyz, np.float32)):
            raise ValueError("decoded coordinate is not exactly on the library wire grid")
        coords[i, :xyz.shape[0]] = q.astype(np.int16)
        row_counts = np.asarray(corpus.groups[gi].counts(mmap=True)[ri], np.uint8)
        counts[i, :row_counts.size] = row_counts
    n_real = counts[:, 0].astype(np.int64)
    n_color = counts[:, 1:].sum(axis=1, dtype=np.int64)
    if np.any(n_real + n_color > width):
        raise ValueError("panel row color counts extend beyond merged coordinates")
    mol_meta = cat.molecules(gids, columns=["hac"])
    if not all(mol_meta["found"]): raise ValueError("panel HAC metadata join failed")
    hac = np.asarray(mol_meta["hac"], np.int16)
    query_mols = [m for m in Chem.SDMolSupplier(str(query_path), removeHs=False) if m is not None]
    if len(query_mols) != 1: raise ValueError(f"expected one query molecule, got {len(query_mols)}")
    generator = RDKitPharmacophoreGenerator()
    queries_qs = load_queries(query_path, color_generator=generator, backend="cpu")
    selected_ids = np.asarray([[r.mol_id, r.conf_idx] for r in refs], np.int64)
    categories = np.asarray([x["category"] for x in selected], dtype="U16")
    query_arrays = (queries_qs.coords, queries_qs.types, queries_qs.n_real,
                    queries_qs.interaction_r, queries_qs.interaction_p)
    geometry_sha = array_sha(coords, counts, gids, selected_ids[:,0], selected_ids[:,1], *query_arrays)
    npz, mp = panel_paths(output, target)
    if npz.exists() or mp.exists(): raise FileExistsError(f"refusing to replace {target} panel")
    _immutable_npz(npz, coords_i16=coords, counts=counts, gid=gids,
        mol_id=selected_ids[:,0], conf_idx=selected_ids[:,1], hac=hac,
        category=categories, selection_score=np.asarray([x["selection_score"] for x in selected], np.float32),
        query_coords=queries_qs.coords, query_types=queries_qs.types,
        query_n_atoms=queries_qs.n_real, interaction_r=queries_qs.interaction_r,
        interaction_p=queries_qs.interaction_p)
    screens = json.loads(screens_path.read_text())
    meta = {"protocol": PROTOCOL, "target": target, "n_panel": PANEL_SIZE,
        "query_hac": int(query_mols[0].GetNumHeavyAtoms()), "query_sdf": str(query_path),
        "query_sdf_sha256": file_sha(query_path), "corpus": str(corpus_path),
        "corpus_manifest_sha256": file_sha(corpus_path / "manifest.json"),
        "selection_cache": str(cache_path), "selection_cache_sha256": file_sha(cache_path),
        "selection_cache_screens_sha256": file_sha(screens_path),
        "selection_source_sha256_provenance": screens.get("build", {}).get("cuda_src_sha256", "unknown"),
        "selection_config_provenance": screens.get("configs", {}).get("shape"),
        "selected": selected, "width": width, "count_width": count_width,
        "library_color_layout": "merged", "library_color_offset": -1,
        "selected_category_counts": {c: int(np.sum(categories == c)) for c in ("top16", "cutoff17_48", "hash32", "extra48")},
        "geometry_sha256": geometry_sha, "npz_sha256": file_sha(npz)}
    mp.parent.mkdir(parents=True, exist_ok=True); mp.write_bytes(_json_bytes(meta))
    return meta

def import_cache(target, *, lane, backend, role, source_npz, source_json,
                 panel_dir, output_dir):
    if role not in ("fixed", "current"): raise ValueError("role must be fixed or current")
    panel_meta, panel = load_panel(panel_dir, target)
    manifest = json.loads(Path(source_json).read_text())
    expected_cfg = config_for(lane, backend)
    if any(manifest.get(k) != v for k, v in {
        "target": target, "lane": lane, "backend": backend,
        "panel_sha256": panel_meta["npz_sha256"],
        "geometry_sha256": panel_meta["geometry_sha256"],
        "config": expected_cfg}.items()):
        raise ValueError("reference source artifact config/panel binding mismatch")
    with np.load(source_npz, allow_pickle=False) as z:
        ids, scores = z["ids"].copy(), z["scores"].copy()
    expected_ids = canonical_panel_ids(panel)
    if ids.shape != (PANEL_SIZE,) or scores.shape != (PANEL_SIZE,) or not np.all(np.isfinite(scores)):
        raise ValueError("reference cache must contain 128 finite scores and IDs")
    pairs = list(zip(map(int, panel["mol_id"]), map(int, panel["conf_idx"])))
    if len(set(pairs)) != PANEL_SIZE:
        raise ValueError("panel contains duplicate (mol_id, conf_idx) identities")
    if np.unique(ids).size != PANEL_SIZE or not np.array_equal(np.sort(ids), np.sort(expected_ids)):
        raise ValueError("reference identity set must equal all 128 panel identities")
    score_map = {int(i): float(s) for i, s in zip(ids, scores)}
    scores = np.asarray([score_map[int(i)] for i in expected_ids], np.float32)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"{target}.{lane}.{backend}"
    npz, mp = out / f"{stem}.npz", out / f"{stem}.json"
    if npz.exists() or mp.exists(): raise FileExistsError(f"refusing to overwrite {role} cache")
    _immutable_npz(npz, ids=expected_ids, scores=scores)
    meta = {"protocol": PROTOCOL, "role": role, "target": target, "lane": lane,
        "backend": backend, "config": expected_cfg, "config_sha256": _sha(_json_bytes(expected_cfg)),
        "geometry_sha256": panel_meta["geometry_sha256"], "panel_sha256": panel_meta["npz_sha256"],
        "source_sha256_provenance": manifest.get("candidate_source_sha256", "unknown"),
        "source_artifact": str(source_npz), "source_artifact_sha256": file_sha(source_npz),
        "score_column": expected_cfg["score_column"], "ids_sha256": array_sha(expected_ids),
        "npz_sha256": file_sha(npz)}
    mp.write_bytes(_json_bytes(meta)); return meta

def _load_scores(directory, target, lane, backend):
    stem = f"{target}.{lane}.{backend}"
    npz, mp = Path(directory) / f"{stem}.npz", Path(directory) / f"{stem}.json"
    if not npz.is_file() or not mp.is_file(): raise Unavailable(f"score cache missing: {npz}")
    meta = json.loads(mp.read_text())
    if file_sha(npz) != meta.get("npz_sha256"): raise ValueError(f"score cache hash mismatch: {npz}")
    with np.load(npz, allow_pickle=False) as z: ids, scores = z["ids"].copy(), z["scores"].copy()
    if ids.shape != scores.shape or ids.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError(f"score cache has non-finite or misaligned rows: {npz}")
    return meta, ids, scores

def _align(ids, scores, expected):
    ids, scores = np.asarray(ids, np.int64), np.asarray(scores, np.float64)
    if ids.shape != scores.shape or ids.ndim != 1 or np.unique(ids).size != len(ids):
        raise ValueError("score artifact has duplicate or misaligned identity rows")
    lookup = {int(i): float(s) for i, s in zip(ids, scores)}
    out = np.full(len(expected), -1.0, np.float64); missing = []
    for j, ident in enumerate(expected):
        if int(ident) in lookup: out[j] = lookup[int(ident)]
        else: missing.append(int(ident))
    return out, missing, sorted(set(map(int, ids)) - set(map(int, expected)))

def _summary(values):
    from roshambo3.metrics import replicate_summary
    a = np.asarray(values, np.float64)
    if len(a) == 0: return {"n": 0, "status": "unavailable"}
    if len(a) == 1:
        v = float(a[0]); return {"n": 1, "mean": v, "p10": v, "min": v, "max": v,
            "mean_ci95_bca": None, "ci_status": "singleton_no_inferential_ci"}
    return replicate_summary(a, B=10000)

def evaluate(*, lane, backend, panel_dir, fixed_dir, current_dir, candidate_dir, targets):
    from roshambo3.metrics import recall_report, rank_order, reduce_over_conformers, rank_agreement_report
    result = {"protocol": PROTOCOL, "lane": lane, "backend": backend,
        "routine_config": config_for(lane, backend), "status": "complete",
        "per_query": [], "unavailable": []}
    for target in targets:
        try:
            pm, p = load_panel(panel_dir, target)
            fm, fi, fs = _load_scores(fixed_dir, target, lane, backend)
            cm, ci, cs = _load_scores(current_dir, target, lane, backend)
            am, ai, ass = _load_scores(candidate_dir, target, lane, backend)
        except Unavailable as exc:
            result["unavailable"].append({"target": target, "reason": str(exc)})
            continue
        expected = canonical_panel_ids(p)
        if (am.get("target") != target or am.get("lane") != lane
            or am.get("backend") != backend
            or am.get("panel_sha256") != pm["npz_sha256"]
            or am.get("geometry_sha256") != pm["geometry_sha256"]):
            raise ValueError(f"candidate artifact panel/config binding failed for {target}")
        if am.get("config") != config_for(lane, backend):
            raise ValueError(f"candidate routine config mismatch for {target}")
        pairs = list(zip(map(int, p["mol_id"]), map(int, p["conf_idx"])))
        if len(set(pairs)) != PANEL_SIZE: raise ValueError("panel has duplicate pair identities")
        for role, meta in (("fixed", fm), ("current", cm)):
            if (meta.get("role") != role or meta.get("config") != config_for(lane, backend)
                or meta.get("geometry_sha256") != pm["geometry_sha256"]
                or meta.get("panel_sha256") != pm["npz_sha256"]):
                raise ValueError(f"{role} immutable cache config/geometry/ID binding failed for {target}")
        fixed, mf, uf = _align(fi, fs, expected); current, mc, uc = _align(ci, cs, expected)
        candidate, missing, unexpected = _align(ai, ass, expected)
        if mf or mc or uf or uc: raise ValueError(f"reference cache identity coverage invalid for {target}")
        row = {"target": target, "query_hac": int(pm["query_hac"]), "M": PANEL_SIZE,
            "candidate_n": len(ai), "missing_candidate_ids": missing,
            "unexpected_candidate_ids": unexpected,
            "candidate_identity_coverage": not missing and not unexpected,
            "fixed_source_sha256_provenance": fm.get("source_sha256_provenance"),
            "current_source_sha256_provenance": cm.get("source_sha256_provenance"),
            "fixed_equals_current": bool(np.array_equal(fixed, current))}
        if missing or unexpected:
            row["selection_metrics_status"] = "unavailable_identity_coverage"
            for role in ("fixed", "current"):
                row[role] = {"metrics_status": "unavailable_identity_coverage",
                    "levels": {"conformer": {"status": "unavailable"},
                               "molecule": {"status": "unavailable"}},
                    "signed_score_movement": {}, "signed_bias_gates": {}}
            result["per_query"].append(row)
            continue
        row["selection_metrics_status"] = "available"
        mol_ids = p["mol_id"]
        for role, gold in (("fixed", fixed), ("current", current)):
            levels = {}
            for level in ("conformer", "molecule"):
                rep = recall_report(gold[None, :], candidate[None, :], 8, ks=(2,),
                    ids=expected, query_ids=[target],
                    mol_ids=mol_ids if level == "molecule" else None, level=level)
                levels[level] = {"M": rep.m_targets, "survivor_fraction": rep.survivor_fraction[2],
                    "recall": float(rep.per_query[0].recall_by_k[2]),
                    "by_selected_conformer_count": rep.by_conf_count}
                if level == "conformer":
                    go, ao = rank_order(gold, expected), rank_order(candidate, expected)
                    levels[level]["top1_in_candidate_top8"] = int(go[0]) in set(ao[:8].tolist())
                    levels[level]["exact_top_order"] = {str(k): bool(np.array_equal(go[:k], ao[:k])) for k in (1,5,10)}
                    levels[level]["rank_agreement"] = rank_agreement_report(
                        gold[None,:], candidate[None,:], 8, ids=expected, query_ids=[target])
                    h = np.asarray(p["hac"]); horder = np.lexsort((np.arange(len(h)), h))
                    bins = np.empty(len(h), np.int8); bins[horder] = np.minimum(2, np.arange(len(h))*3//len(h))
                    top = go[:8]; candidate_top = set(ao[:16].tolist())
                    levels[level]["target_hac_strata"] = {
                        f"tercile_{b+1}": {"desired_n": int(np.sum(bins[top] == b)),
                            "recall_at_candidate_16": float(np.mean([int(x) in candidate_top for x in top[bins[top] == b]]))}
                        for b in range(3) if np.any(bins[top] == b)}
                else:
                    um, gb, _, _ = reduce_over_conformers(gold, mol_ids, expected)
                    _, ab, _, _ = reduce_over_conformers(candidate, mol_ids, expected)
                    go, ao = rank_order(gb, um), rank_order(ab, um)
                    levels[level]["top1_in_candidate_top8"] = int(go[0]) in set(ao[:8].tolist())
                    levels[level]["exact_top_order"] = {str(k): bool(np.array_equal(go[:k], ao[:k])) for k in (1,5,10)}
                    levels[level]["rank_agreement"] = rank_agreement_report(gb[None,:], ab[None,:], 8, ids=um, query_ids=[target])
            delta = candidate - gold; top = rank_order(gold, expected)[:8]
            h = np.asarray(p["hac"]); horder = np.lexsort((np.arange(len(h)), h))
            bins = np.empty(len(h), np.int8); bins[horder] = np.minimum(2, np.arange(len(h))*3//len(h))
            bias = {"global": float(np.mean(delta)), "reference_top8": float(np.mean(delta[top]))}
            for b in range(3):
                mask = bins == b
                if np.any(mask): bias[f"target_hac_tercile_{b+1}"] = float(np.mean(delta[mask]))
            row[role] = {"metrics_status": "available", "levels": levels,
                         "signed_score_movement": bias,
                         "signed_bias_gates": {k: abs(v) <= 0.02 for k, v in bias.items()}}
        result["per_query"].append(row)
    if not result["per_query"]:
        result["status"] = "unavailable"; result["detector_pass"] = False; return result
    result["status"] = "partial" if result["unavailable"] else "complete"
    comps = {}
    for role in ("fixed", "current"):
        levels = {}
        valid_queries = [r for r in result["per_query"] if role in r
                         and r[role].get("metrics_status") == "available"]
        for level in ("conformer", "molecule"):
            vals = [r[role]["levels"][level]["recall"] for r in valid_queries]
            s = _summary(vals)
            s["gate"] = s.get("mean", -np.inf) >= .95 and s.get("p10", -np.inf) >= .75 and s.get("min", -np.inf) >= .75
            s["gate_definition"] = {"mean_min": .95, "p10_min": .75, "min_min": .75}
            levels[level] = s
        bias = {}
        for key in ("global", "reference_top8", "target_hac_tercile_1", "target_hac_tercile_2", "target_hac_tercile_3"):
            vals = [r[role]["signed_score_movement"][key] for r in valid_queries
                    if key in r[role]["signed_score_movement"]]
            s = _summary(vals); s["abs_mean_bias_gate"] = abs(s.get("mean", np.inf)) <= .02
            bias[key] = s
        landing = bool(valid_queries) and all(
            r[role]["levels"][lv]["top1_in_candidate_top8"]
            for r in valid_queries for lv in ("conformer", "molecule"))
        comps[role] = {"levels": levels, "signed_score_movement": bias, "top1_landing_all_queries": landing}
    result["comparisons"] = comps
    result["identity_coverage_all_queries"] = all(
        r["candidate_identity_coverage"] for r in result["per_query"])
    result["detector_pass"] = not result["unavailable"] and result["identity_coverage_all_queries"] and all(
        comps[r]["levels"][lv]["gate"] for r in ("fixed","current") for lv in ("conformer","molecule")) and all(
        comps[r]["signed_score_movement"][key]["abs_mean_bias_gate"]
        for r in ("fixed","current") for key in ("global","reference_top8","target_hac_tercile_1","target_hac_tercile_2","target_hac_tercile_3")) and all(
        comps[r]["top1_landing_all_queries"] for r in ("fixed","current"))
    result["per_query_bias_pass"] = all(
        all(r[role]["signed_bias_gates"].values())
        for r in result["per_query"] for role in ("fixed", "current")
        if r[role].get("metrics_status") == "available")
    result["detector_pass"] = result["detector_pass"] and result["per_query_bias_pass"]
    ordered = sorted(result["per_query"], key=lambda r: (r["query_hac"], r["target"]))
    result["query_hac_terciles"] = {}
    for b in range(3):
        grp = ordered[b * len(ordered) // 3:(b + 1) * len(ordered) // 3]
        result["query_hac_terciles"][f"tercile_{b+1}"] = {
            "targets": [r["target"] for r in grp],
            "by_role": {role: {level: _summary([
                r[role]["levels"][level]["recall"] for r in grp if role in r
                and r[role].get("metrics_status") == "available"])
                for level in ("conformer", "molecule")} for role in ("fixed", "current")}}
    result["interpretation"] = "M=128 challenger-enriched panel; N=8 reference hits; candidate keeps 16 (k=2); survivor fraction accompanies recall. This does not estimate whole-library recall."
    return result

def _parser():
    p = argparse.ArgumentParser(description=__doc__); sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("panel-build", help="build immutable deterministic panels")
    b.add_argument("--target", choices=ALL_TARGETS, action="append")
    b.add_argument("--corpora", type=Path, required=True)
    b.add_argument("--selection-cache", type=Path, required=True)
    b.add_argument("--queries", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    c = sp.add_parser("cache-import", help="qualify a separately produced 100-step cache")
    c.add_argument("--target", choices=ALL_TARGETS, required=True)
    c.add_argument("--lane", choices=("shape","combo"), required=True)
    c.add_argument("--backend", choices=("cpu","cuda"), required=True)
    c.add_argument("--role", choices=("fixed","current"), required=True)
    c.add_argument("--source-npz", type=Path, required=True)
    c.add_argument("--source-json", type=Path, required=True)
    c.add_argument("--panel-dir", type=Path, required=True)
    c.add_argument("--output-dir", type=Path, required=True)
    e = sp.add_parser("evaluate", help="compare candidate with independent fixed/current caches")
    e.add_argument("--lane", choices=("shape","combo"), required=True)
    e.add_argument("--backend", choices=("cpu","cuda"), required=True)
    e.add_argument("--panel-dir", type=Path, required=True)
    e.add_argument("--fixed-dir", type=Path, required=True)
    e.add_argument("--current-dir", type=Path, required=True)
    e.add_argument("--candidate-dir", type=Path, required=True)
    e.add_argument("--target", choices=ALL_TARGETS, action="append")
    e.add_argument("--output", type=Path)
    return p

def main(argv=None):
    a = _parser().parse_args(argv)
    try:
        if a.cmd == "panel-build":
            targets = a.target or ALL_TARGETS
            out = [build_panel(t, corpora=a.corpora, selection_cache=a.selection_cache,
                queries=a.queries, output=a.output) for t in targets]
            print(json.dumps({"status":"complete","panels":out}, indent=2))
        elif a.cmd == "cache-import":
            print(json.dumps(import_cache(a.target, lane=a.lane, backend=a.backend, role=a.role,
                source_npz=a.source_npz, source_json=a.source_json,
                panel_dir=a.panel_dir, output_dir=a.output_dir), indent=2))
        else:
            out = evaluate(lane=a.lane, backend=a.backend, panel_dir=a.panel_dir,
                fixed_dir=a.fixed_dir, current_dir=a.current_dir, candidate_dir=a.candidate_dir,
                targets=a.target or EVAL_TARGETS)
            if a.output:
                a.output.parent.mkdir(parents=True, exist_ok=True)
                if a.output.exists(): raise FileExistsError(a.output)
                a.output.write_bytes(_json_bytes(out))
            print(json.dumps(out, indent=2))
            if out["status"] == "unavailable": return 2
            return 0 if out.get("detector_pass") else 1
    except Unavailable as exc:
        print(json.dumps({"status":"unavailable","reason":str(exc)}, indent=2), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


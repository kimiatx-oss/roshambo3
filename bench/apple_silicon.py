"""Frozen-input CPU port benchmark; run with python -m bench.apple_silicon.

QoS labels describe requests, never inferred P/E residency. Timing records use
the existing CPU grading harness, including refusal/advisory receipts.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

from bench.cpu import cpu_identity, physical_core_cpus
from bench.cpu_run import CpuRun
from bench.run_record import sha256_file
from roshambo3.api import screen
from roshambo3.backend import cpu
from roshambo3.backend.spec import ColorPlacement
from roshambo3.engine import CPUWorkerOptions, Engine, ScreenConfig
from roshambo3.engine.query import QuerySet, from_arrays
from roshambo3.format import Catalog, Corpus
from roshambo3.pharmacophore import RDKitPharmacophoreGenerator

# Durable keys in the original eight-query order, resolved read-only from the
# pre-switch tuning-library GIDs. Live runs locate these pairs in the format-3 catalog.
PANEL_KEYS = [(58059676, 0), (12673788, 0), (57076096, 0), (13625945, 0),
              (59925789, 0), (10877595, 0), (84669660, 0), (14288517, 0)]
# Only bench.compare_cpu_port reads the pre-switch captures, whose filenames
# and GID arrays describe a different corpus ordering. Live runs never use
# these corpus-local values.
LEGACY_CAPTURE_GIDS = [25511, 205370, 236070, 684138, 154726, 200425, 425209, 685793]


def panel_gids(catalog, panel_keys=PANEL_KEYS):
    """Resolve the frozen panel's durable identities into this corpus's GIDs."""
    return [catalog.gid_for(mol_id, conf_idx) for mol_id, conf_idx in panel_keys]


def key_label(key):
    return f"m{int(key[0])}-c{int(key[1])}"


def feature_payload(corpus, ref, coordinates):
    """Counts and format-3-derived feature coordinates for one conformer."""
    from roshambo3.format.r3g.reference import derive_positions

    with corpus.group(ref.group_index) as group:
        molecule = int(np.searchsorted(group.run_start, ref.row, side="right") - 1)
        counts = group.counts[ref.row].copy()
        if not corpus.has_color:
            return counts, np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
        first, last = map(int, group.feature_offsets[molecule:molecule + 2])
        features = []
        for feature in range(first, last):
            a0, a1 = map(int, group.feature_atom_offsets[feature:feature + 2])
            atoms = tuple(int(i) for i in group.feature_atoms[a0:a1])
            features.append((int(group.feature_types[feature]), atoms))
    raw = np.rint(coordinates[:ref.n_real] / np.float32(corpus.step)).astype(np.int16)
    pseudo = derive_positions(raw.T, features).astype(np.float32) * np.float32(corpus.step)
    types = np.asarray([t for t, _ in features], dtype=np.int32)
    return counts, pseudo, types


def corpus_fingerprint(root):
    # Hash every data file, not just the manifest's claims about the data.
    hashes = {str(p.relative_to(root)): sha256_file(p)
              for p in sorted(root.rglob("*")) if p.is_file()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def freeze(corpus, destination, panel_keys=PANEL_KEYS):
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(corpus)
    generator = RDKitPharmacophoreGenerator()
    if not corpus.has_color:
        raise ValueError("the frozen Apple panel requires a color format-3 corpus")
    shapes, colors = [], []
    gids = panel_gids(catalog, panel_keys)
    for (mol_id, conf_idx), ref, coordinates in zip(
            panel_keys, catalog.locate(gids), catalog.coordinates(gids)):
        _, pseudo, types = feature_payload(corpus, ref, coordinates)
        shapes.append(coordinates)
        colors.append((pseudo, types))
    names = [f"{mol_id}:{conf_idx}" for mol_id, conf_idx in panel_keys]
    queries = from_arrays(shapes, names,
                          color_generator=generator, color=colors)
    np.savez(destination, coords=queries.coords, types=queries.types,
             n_real=queries.n_real, names=np.array(queries.names),
             interaction_r=queries.interaction_r, interaction_p=queries.interaction_p,
             vaa=queries.vaa)


def load_query(path, index):
    with np.load(path, allow_pickle=False) as data:
        return QuerySet(**{key: (data[key][index:index+1].copy() if key in
                         ("coords", "types", "n_real", "vaa") else
                         [str(data[key][index])] if key == "names" else data[key].copy())
                         for key in data.files})


def configuration(mode, workers, qos, baseline=False):
    return ScreenConfig(color=mode != "shape", mixing=0.0 if mode == "shape" else 0.5,
        color_placement=ColorPlacement.IN_GRADIENT if mode == "combo" else ColorPlacement.AT_POSE,
        cpu_worker=CPUWorkerOptions(workers=workers,
                                   **({} if baseline else {"cpu_qos": qos})))


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.corpus).resolve()
    corpus = Corpus(str(root))
    catalog = Catalog(corpus)
    panel_gids(catalog, args.panel_keys)  # refuse missing identities up front
    query_path = Path(args.queries)
    if args.freeze:
        freeze(corpus, query_path, args.panel_keys)
    if args.freeze_only:
        return
    fingerprint = corpus_fingerprint(root)
    build = cpu.build_stamp()
    platform_info = cpu.extension().cpu_platform_info() if hasattr(cpu.extension(), "cpu_platform_info") else None
    provenance = {"python": sys.version, "executable": sys.executable,
                  "platform": platform.platform(), "cpu": cpu_identity().as_dict(),
                  "build": build, "native_platform": platform_info,
                  "queries_sha256": sha256_file(query_path),
                  "corpus_sha256": fingerprint["sha256"],
                  "core_residency": None,
                  "core_residency_reason": "untraced: QoS is not proof of core placement"}
    provenance["extension_path"] = cpu.extension().__file__
    provenance["extension_sha256"] = sha256_file(cpu.extension().__file__)
    provenance["dependencies"] = {name: importlib.metadata.version(name)
                                  for name in ("numpy", "scipy", "rdkit", "pyarrow", "zstandard")}
    provenance["provisional"] = args.provisional
    manifest = {"queries_sha256": provenance["queries_sha256"],
                "corpus_sha256": fingerprint["sha256"], "k": args.k,
                "level": args.level, "fresh_session": args.fresh,
                "cpu_src_sha256": build["cpu_src_sha256"],
                "cxx_flags": build["cxx_flags"], "compiler": build.get("compiler"),
                "provisional": args.provisional}
    manifest_path = out / "benchmark-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("output directory contains an incompatible benchmark manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out / "corpus-fingerprint.json").write_text(json.dumps(fingerprint, indent=2))
    if sys.platform == "darwin":
        from bench.cpu_darwin import power_state
        provenance["power"] = power_state()
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    for mode in args.modes.split(","):
        for qos in args.qos.split(","):
            for worker_text in args.workers.split(","):
                workers = int(worker_text) if worker_text != "auto" else cpu_identity().logical_cpus
                config = configuration(mode, workers, qos, args.baseline)
                label = f"{mode}-{qos}-{worker_text}"
                completed = set()
                ledger = out / "timings.jsonl"
                if ledger.exists():
                    completed = {(r["label"], r["repeat"],
                                  int(r["mol_id"]), int(r["conf_idx"]))
                                 for r in map(json.loads, ledger.read_text().splitlines())}
                engine = Engine(backend="cpu")
                try:
                    for repeat in range(-args.warmup, args.repeats):
                        indices = np.roll(np.arange(len(args.panel_keys)), repeat)
                        for index in indices:
                            mol_id, conf_idx = args.panel_keys[int(index)]
                            identity = (int(mol_id), int(conf_idx))
                            query_index = args.panel_keys.index(identity)
                            if (label, repeat, *identity) in completed:
                                continue
                            if args.fresh and (repeat >= 0 or repeat == -args.warmup):
                                engine.close()
                                engine = Engine(backend="cpu")
                            query = load_query(query_path, query_index)
                            cpus = physical_core_cpus() if sys.platform == "linux" else None
                            with CpuRun("apple-silicon-port", threads=workers,
                                        cpus=cpus, sample_frequency=sys.platform == "linux",
                                        blocking=True) as receipt:
                                receipt.dataset = {"n_conformers": corpus.n_conf,
                                                   "sha256": fingerprint["sha256"]}
                                receipt.query_panel = {"ids": [dict(mol_id=mol_id,
                                                                     conf_idx=conf_idx)],
                                                       "sha256": sha256_file(query_path),
                                                       "name": "apple-port-eight"}
                                receipt.config = {"mode": mode, "cpu_qos": qos, "workers": workers,
                                                  "max_results": args.k, "steps": config.steps,
                                                  "optimizer": config.resolved_optimizer(),
                                                  "level": args.level, "fresh_session": args.fresh}
                                started = time.perf_counter()
                                result = screen(query, corpus, k=args.k, level=args.level,
                                                config=config, engine=engine, prefetch_depth=0,
                                                color_generator=RDKitPharmacophoreGenerator())
                                seconds = time.perf_counter() - started
                                receipt.timing = {"wall_s": seconds, "n_repeats": args.repeats,
                                                  "n_warmup": args.warmup}
                                receipt.notes = ("PROVISIONAL measurement; concurrent laptop work permitted. "
                                                 if args.provisional else "") + "QoS policy experiment; P/E placement unverified. " + (
                                    "Warmup." if repeat < 0 else "Measured repetition.")
                            hit = result.hits[0]
                            artifact = out / f"{label}-r{repeat}-{key_label(identity)}.npz"
                            if repeat == 0 or args.save_all:
                                np.savez_compressed(artifact, scores=hit.scores,
                                    mol_ids=hit.mol_id, conf_indices=hit.conf_idx)
                            record = {"label": label, "repeat": repeat,
                                      "mol_id": int(mol_id), "conf_idx": int(conf_idx),
                                      "wall_s": seconds, "evaluations_per_s": corpus.n_conf / seconds,
                                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                                        (1 if sys.platform == "darwin" else 1024),
                                      "cpu_scan": result.stats.cpu_scan, "grade": receipt.status}
                            record["provisional"] = args.provisional
                            with ledger.open("a") as stream:
                                stream.write(json.dumps(record) + "\n")
                            receipt.write(out / "run-records.jsonl")
                            print(f"{label} repeat={repeat} identity={identity} "
                                  f"{seconds:.3f}s {receipt.status}", flush=True)
                finally:
                    engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--workers", default="1,2,4,8,10,12,14,auto")
    parser.add_argument("--qos", default="inherit")
    parser.add_argument("--modes", default="combo,shape,rescore")
    def parse_panel_keys(text):
        try:
            return [tuple(map(int, value.split(":"))) for value in text.split(",")]
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError("panel keys must be mol_id:conf_idx pairs") from None
    parser.add_argument("--panel-keys", type=parse_panel_keys, default=PANEL_KEYS,
                        help="subset of the frozen panel as mol_id:conf_idx pairs")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--level", choices=("conformer", "molecule"), default="molecule")
    parser.add_argument("--fresh", action="store_true", help="recreate Engine before each screen")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--provisional", action="store_true",
                        help="label exploratory timings without weakening grading rules")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.k < 1:
        parser.error("repeats/k must be positive; warmup must be nonnegative")
    if set(args.modes.split(",")) - {"combo", "shape", "rescore"}:
        parser.error("unknown mode")
    if (not args.panel_keys or len(set(args.panel_keys)) != len(args.panel_keys)
            or set(args.panel_keys) - set(PANEL_KEYS)):
        parser.error("--panel-keys must be a unique subset of the frozen identities")
    if args.baseline and args.qos != "inherit":
        parser.error("the original release only supports inherited QoS")
    if args.freeze_only and not args.freeze:
        parser.error("--freeze-only requires --freeze")
    run(args)


if __name__ == "__main__":
    main()

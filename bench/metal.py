"""Locked, frozen-input Apple CPU/Metal quality and performance campaign.

Results remain advisory without achieved-frequency/thermal residency evidence.
Use --provisional for short development runs; it cannot certify release gates.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import resource
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import psutil

from bench.apple_silicon import (PANEL_KEYS, corpus_fingerprint, feature_payload,
                                 key_label, load_query)
from bench.cpu_darwin import power_state
from roshambo3 import screen
from roshambo3.backend import cpu, metal
from roshambo3.backend.spec import ColorPlacement
from roshambo3.engine import CPUWorkerOptions, Engine, ScreenConfig
from roshambo3.format import Catalog, Corpus
from roshambo3.metrics.ranking import canonical_ids
from roshambo3.pharmacophore import RDKitPharmacophoreGenerator
from roshambo3.scores import COLUMN_INDEX
from roshambo3.web.gpu import resource_lock


def configuration(mode, workers=None):
    cfg = ScreenConfig(
        color=mode != "shape",
        mixing=0.5 if mode != "shape" else 0,
        color_placement=ColorPlacement.IN_GRADIENT
        if mode == "combo"
        else ColorPlacement.AT_POSE,
    )
    if workers is not None:
        cfg = replace(
            cfg,
            cpu_worker=CPUWorkerOptions(
                workers=workers,
                cpu_qos="user_initiated",
            ),
        )
    return cfg


def cpu_driver_sha256():
    """Bind baseline selection to the Python policy as well as native source."""
    import roshambo3.api
    from roshambo3 import optimizer
    from roshambo3.backend import self_overlap_bindings
    # engine.cpu_self went with the legacy routes; format-3 self
    # bindings live in engine.r3g_cpu and backend.self_overlap_bindings.
    from roshambo3.engine import config, cpu_modes, engine, query, r3g_cpu, source

    digest = hashlib.sha256(inspect.getsource(configuration).encode())
    for module in (
        roshambo3.api,
        engine,
        config,
        cpu_modes,
        r3g_cpu,
        self_overlap_bindings,
        source,
        query,
        optimizer,
        cpu,
    ):
        digest.update(module.__name__.encode())
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


VARIANTS = {
    "baseline": dict(specialize_color=False, factor_color_torque=False),
    "specialized": dict(specialize_color=True, factor_color_torque=False),
    "torque": dict(specialize_color=False, factor_color_torque=True),
    "both": dict(specialize_color=True, factor_color_torque=True),
}


def ranking(scores, mol_ids, conf_indices, k=100):
    ids = canonical_ids(mol_ids, conf_indices)
    order = np.lexsort((ids, -scores[:, 0]))
    _, first = np.unique(np.asarray(mol_ids)[order], return_index=True)
    order = order[np.sort(first)][:k]
    return np.asarray(mol_ids)[order], np.asarray(conf_indices)[order]


def ranking_passes(result):
    return result["top1"] and result["top10_order"] and result["top100_overlap"] == 100


def aligned(reference, candidate):
    """Align two score captures by canonical ``(mol_id, conf_idx)`` identity."""
    with np.load(reference) as a, np.load(candidate) as b:
        am, ac = np.asarray(a["mol_ids"]), np.asarray(a["conf_indices"])
        bm, bc = np.asarray(b["mol_ids"]), np.asarray(b["conf_indices"])
        aid, bid = canonical_ids(am, ac), canonical_ids(bm, bc)
        ai, bi = np.argsort(aid), np.argsort(bid)
        if not np.array_equal(am[ai], bm[bi]) or not np.array_equal(ac[ai], bc[bi]):
            raise ValueError("capture populations differ by molecule/conformer identity")
        return am[ai], ac[ai], a["scores"][ai], b["scores"][bi]


def compare(a, b, mol_ids, conf_indices):
    am, aw = ranking(a, mol_ids, conf_indices)
    bm, bw = ranking(b, mol_ids, conf_indices)
    changes = []
    for k in (10, 100):
        changed = sorted(set(am[:k]) ^ set(bm[:k]))
        for mid in changed:
            i = np.flatnonzero(np.asarray(mol_ids) == mid)
            # Resolve score ties by the canonical conformer index so the
            # pose named in an audit row is independent of source row order.
            ia = i[np.lexsort((np.asarray(conf_indices)[i], -a[i, 0]))[0]]
            ib = i[np.lexsort((np.asarray(conf_indices)[i], -b[i, 0]))[0]]
            changes.append(dict(
                cutoff=k, molecule_id=int(mid),
                a_rank=int(np.flatnonzero(am == mid)[0] + 1) if mid in am else None,
                b_rank=int(np.flatnonzero(bm == mid)[0] + 1) if mid in bm else None,
                a_conf_idx=int(conf_indices[ia]), b_conf_idx=int(conf_indices[ib]),
                a_score=float(a[ia, 0]), b_score=float(b[ib, 0]),
                a_row=a[ia].tolist(), b_row=b[ib].tolist()))
    return dict(top1=bool(am[0] == bm[0]),
                top10_order=bool(np.array_equal(am[:10], bm[:10])),
                top100_overlap=len(set(am) & set(bm)), changes=changes)


def dispatch(ctx, options, query, mode):
    planes = []
    if mode == "combo" and options.get("color_field_min_sources", 0):
        for plane in (1, 2):
            sources = int(np.count_nonzero(query.types[0, int(query.n_real[0]):] == plane))
            if (sources >= 12 and query.interaction_r[plane, plane] > 0
                    and query.interaction_p[plane, plane] != 0):
                planes.append(plane)
    expected = "optimize_diagonal_factored" if mode == "combo" else "optimize"
    if planes:
        expected = f"optimize_color_fields_m{sum(1 << p for p in planes)}"
    actual = [p["plane"] for p in ctx.last_color_fields]
    if ctx.last_optimizer_kernel != expected or actual != planes:
        raise RuntimeError(f"Unexpected dispatch {ctx.last_optimizer_kernel}/{actual}, "
                           f"expected {expected}/{planes}")
    return dict(optimizer_kernel=ctx.last_optimizer_kernel,
                color_fields=ctx.last_color_fields,
                shape_sampler=ctx.field_sampler,
                color_sampler=ctx.color_field_sampler)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))


def replace_self_terms(rows, source, mode):
    """Counterfactual score rows for the retained self-overlap analysis."""
    result = rows.astype(np.float64, copy=True)
    result[:, 5:9] = source[:, 5:9]
    for score, overlap, query, library in ((1, 3, 5, 6), (2, 4, 7, 8)):
        den = result[:, query] + result[:, library] - result[:, overlap]
        result[:, score] = np.divide(
            result[:, overlap], den, out=np.zeros(len(rows)), where=den > 0)
    result[:, 0] = result[:, 1] if mode == "shape" else .5 * (result[:, 1] + result[:, 2])
    return result


def audit_score_columns(reported, replay, oracle):
    """Audit all nine scientific columns, including normalization self terms."""
    oracle = np.asarray(oracle, np.float64)
    scale = np.maximum(1, np.abs(oracle))
    reported_error = np.abs(np.asarray(reported, np.float64) - oracle) / scale
    replay_error = np.abs(np.asarray(replay, np.float64) - oracle) / scale
    return dict(
        columns=list(COLUMN_INDEX)[:9],
        reported_oracle_max_relative=reported_error.max(axis=0).tolist(),
        replay_oracle_max_relative=replay_error.max(axis=0).tolist(),
        passed=bool(np.max(reported_error) <= 1e-4 and np.max(replay_error) <= 1e-4),
    )


def audit_poses(query, catalog, gids, reference_rows, metal_rows, mode):
    """Replay both trajectories through native CPU, Metal, and an independent oracle."""
    from roshambo3.layout import tile32
    from roshambo3.numerics import COLOR_PREFACTOR, COORD_SCALE, SHAPE_PREFACTOR
    from roshambo3.reference.color import quat_multiply, score_columns
    from roshambo3.reference.overlap import quaternion_matrix

    refs = catalog.locate(gids)
    heavy = catalog.coordinates(gids)
    nf = len(query.interaction_r)
    count_rows = []
    libraries, library_types = [], []
    for ref, xyz in zip(refs, heavy):
        counts, pseudo, types = feature_payload(catalog.corpus, ref, xyz)
        count_rows.append(counts.astype(np.int32))
        libraries.append(np.concatenate((xyz[:ref.n_real], pseudo), axis=0))
        library_types.append(types)
    counts = np.asarray(count_rows, np.int32)
    width = max(len(x) for x in libraries)
    physical = np.zeros((len(gids), width, 3), np.float32)
    for i, coords in enumerate(libraries):
        physical[i, : len(coords)] = coords
    packed = np.zeros(((len(gids) + 31) // 32 * 32, width, 4), np.float32)
    packed[: len(gids), :, :3] = physical
    qcounts = np.array(
        [
            [int(query.n_real[0])]
            + [int(np.sum(query.types[0] == t)) for t in range(1, nf)]
        ],
        np.int32,
    )
    frozen = np.concatenate([reference_rows, metal_rows])
    poses = np.empty((len(frozen), 7), np.float32)
    for i, row in enumerate(frozen):
        poses[i, :4] = quat_multiply(row[16:20], row[9:13])
        poses[i, 4:] = quaternion_matrix(row[16:20]) @ row[13:16]
    pairlib = np.tile(np.arange(len(gids), dtype=np.int32), 2)
    replay = metal.score_pairs(
        query_coords=query.coords,
        query_types=query.types,
        query_n_atoms=query.n_real,
        library_coords=tile32(packed),
        library_counts=counts.astype(np.uint8),
        pair_query_row=np.zeros(len(poses), np.int32),
        pair_library_row=pairlib,
        interaction_r=query.interaction_r,
        interaction_p=query.interaction_p,
        color=mode != "shape",
        mixing=0.5 if mode != "shape" else 0.0,
        library_color_offset=-1,
        poses=poses,
    )
    r = query.interaction_r.astype(np.float64)
    h = query.interaction_p.astype(np.float64)
    coefficients = np.zeros_like(r)
    active = r > 0
    coefficients[active] = float(COLOR_PREFACTOR) * h[active] ** 2 * r[active] ** -1.5
    matrices = np.array(
        [quaternion_matrix(p[:4]) * COORD_SCALE for p in poses], np.float32
    )
    native = cpu.extension().final_rescore_fixed(
        query_reduced=np.ascontiguousarray(query.coords[:, :, :3] * COORD_SCALE),
        query_counts=qcounts,
        library_physical=physical,
        library_counts=counts,
        pair_query=np.zeros(len(poses), np.int32),
        pair_library=pairlib,
        matrices=matrices,
        translations=np.ascontiguousarray(poses[:, 4:] * COORD_SCALE),
        coefficients=coefficients,
        physical_decays=-r / 2,
        canonical=True,
        mode=0,
    )
    oracle = []
    oracle_rows = []
    for i, pose in enumerate(poses):
        bi = int(pairlib[i])
        nr = int(counts[bi, 0])
        nq = int(query.n_real[0])
        nc = int(counts[bi, 1:].sum())
        qc = query.coords[0, nq:, :3]
        qt = query.types[0, nq:]
        mask = qt > 0
        row = score_columns(
            query.coords[0, :nq, :3],
            physical[bi, :nr],
            qc[mask],
            qt[mask],
            physical[bi, nr : nr + nc],
            library_types[bi],
            r,
            h,
            0.5 if mode != "shape" else 0.0,
            pose[:4],
            pose[4:],
        )
        values = [row[name] for name in list(COLUMN_INDEX)[:9]]
        if mode == "shape":
            for column in (2, 4, 7, 8):
                values[column] = 0.0
        oracle_rows.append(values)
        oracle.append(
            [row["overlap_volume"], row["overlap_color"] if mode != "shape" else 0.0]
        )
    oracle = np.array(oracle)
    gpu = replay[:, 3:5].astype(np.float64)
    host = np.column_stack(
        [
            np.asarray(native["shape"]) * float(SHAPE_PREFACTOR),
            np.asarray(native["color"]) if mode != "shape" else np.zeros(len(poses)),
        ]
    )
    scale = np.maximum(1, np.abs(oracle))
    gpu_drift = np.abs(gpu - oracle) / scale
    host_drift = np.abs(host - oracle) / scale
    reported = np.abs(gpu - frozen[:, 3:5]) / np.maximum(1, np.abs(gpu))
    score_audit = audit_score_columns(frozen[:, :9], replay[:, :9], oracle_rows)
    return dict(
        score_columns=score_audit,
        poses=len(poses),
        identities=[dict(mol_id=int(m), conf_idx=int(c))
                    for m, c in zip(catalog.mol_ids(gids), catalog.conf_indices(gids))],
        metal_oracle_max_relative=gpu_drift.max(axis=0).tolist(),
        cpu_oracle_max_relative=host_drift.max(axis=0).tolist(),
        reported_replay_max_relative=reported.max(axis=0).tolist(),
        passed=bool(
            np.max(gpu_drift) <= 1e-4
            and np.max(host_drift) <= 1e-4
            and np.max(reported) <= 1e-4
            and score_audit["passed"]
        ),
    )


def quality(reference, candidate, catalog, queries):
    rows = []
    for mode in ("shape", "rescore", "combo"):
        for identity in PANEL_KEYS:
            mol_id, conf_idx = identity
            label = key_label(identity)
            with (
                np.load(reference / f"{mode}-inherit-14-r0-{label}.npz") as a,
                np.load(candidate / f"{mode}-{label}.npz") as b,
            ):
                mids_a, cis_a = a["mol_ids"], a["conf_indices"]
                mids_b, cis_b = b["mol_ids"], b["conf_indices"]
                ai = np.argsort(canonical_ids(mids_a, cis_a))
                bi = np.argsort(canonical_ids(mids_b, cis_b))
                if not (np.array_equal(mids_a[ai], mids_b[bi])
                        and np.array_equal(cis_a[ai], cis_b[bi])):
                    raise ValueError("quality populations differ by pair identity")
                aa, bb = a["scores"][ai], b["scores"][bi]
                mids, cis = mids_a[ai], cis_a[ai]
                am, aw = ranking(aa, mids, cis)
                bm, bw = ranking(bb, mids, cis)
                delta = np.abs(aa[:, :9].astype(np.float64) - bb[:, :9])
                row = dict(
                    mode=mode,
                    query_identity=dict(mol_id=int(mol_id), conf_idx=int(conf_idx)),
                    top1=bool(np.array_equal(am[:1], bm[:1])),
                    top10_order=bool(np.array_equal(am[:10], bm[:10])),
                    top100_membership=set(am.tolist()) == set(bm.tolist()),
                    top100_overlap=len(set(am.tolist()) & set(bm.tolist())),
                    score_delta_p99=np.quantile(delta, 0.99, axis=0).tolist(),
                    score_delta_max=delta.max(axis=0).tolist(),
                )
                worst = np.argsort(delta[:, 0])[-8:]
                audit_pairs = set(zip(mids[worst].tolist(), cis[worst].tolist()))
                if not row["top10_order"]:
                    audit_pairs.update(zip(am[:10].tolist(), aw[:10].tolist()))
                    audit_pairs.update(zip(bm[:10].tolist(), bw[:10].tolist()))
                if not row["top100_membership"]:
                    changed = set(am.tolist()) ^ set(bm.tolist())
                    audit_pairs.update((int(m), int(c)) for m, c in zip(am, aw)
                                       if int(m) in changed)
                    audit_pairs.update((int(m), int(c)) for m, c in zip(bm, bw)
                                       if int(m) in changed)
                ordered_pairs = sorted(audit_pairs)
                row_index = {(int(m), int(c)): i for i, (m, c) in enumerate(zip(mids, cis))}
                worst = np.asarray([row_index[pair] for pair in ordered_pairs], np.int64)
                audit_gids = np.asarray([catalog.gid_for(*pair) for pair in ordered_pairs],
                                         np.int64)
                row["fixed_pose_audit"] = audit_poses(
                    queries[PANEL_KEYS.index(identity)],
                    catalog,
                    audit_gids,
                    aa[worst],
                    bb[worst],
                    mode,
                )
                np.savez(
                    candidate / f"audit-{mode}-{label}.npz",
                    mol_ids=np.asarray([pair[0] for pair in ordered_pairs], np.int64),
                    conf_indices=np.asarray([pair[1] for pair in ordered_pairs], np.int64),
                    reference=aa[worst],
                    candidate=bb[worst],
                )
                rows.append(row)
    return dict(
        passed=all(
            r["top1"]
            and r["top10_order"]
            and r["top100_membership"]
            and r["fixed_pose_audit"]["passed"]
            for r in rows
        ),
        queries=rows,
    )


def summarize_timings(records, repeats):
    """Use the first requested matched repeats, rejecting incomplete receipts."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    summary = {}
    for mode in ("shape", "rescore", "combo"):
        cells = {}
        for backend in ("cpu", "metal_fast", "metal_precise"):
            selected = [
                r
                for r in records
                if r["mode"] == mode
                and r["backend"] == backend
                and 0 <= r["repeat"] < repeats
            ]
            if sorted(r["repeat"] for r in selected) != list(range(repeats)):
                raise ValueError(f"incomplete or duplicate {mode}/{backend} repeats")
            values = [r["seconds"] for r in selected]
            if any(not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError("invalid timing measurement")
            cells[backend] = float(np.median(values))
        cells["cpu_speedup"] = cells["cpu"] / cells["metal_fast"]
        cells["fast_exp2_speedup"] = cells["metal_precise"] / cells["metal_fast"]
        summary[mode] = cells
    return summary


def summarize_existing(out, repeats):
    provenance = json.loads((out / "provenance.json").read_text())
    preliminary = json.loads((out / "cpu-preliminary.json").read_text())
    candidates = {int(k): float(v) for k, v in preliminary["seconds"].items()}
    records = [
        json.loads(line) for line in (out / "timings.jsonl").read_text().splitlines()
    ]
    modes = summarize_timings(records, repeats)
    report = dict(
        cpu_workers=min(candidates, key=candidates.get),
        cpu_preliminary=candidates,
        modes=modes,
        speed_gate=all(x["cpu_speedup"] >= 1.5 for x in modes.values()),
        minimum_repeats_met=provenance["warmups"] >= 3 and repeats >= 5,
        selected_repeat_indices=list(range(repeats)),
        selection="first matched repeats; extra raw samples retained",
        originally_requested_repeats=provenance["repeats"],
        raw_sample_count=len(records),
        reporting_benchmark_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        provisional=provenance["provisional"],
        grade="advisory",
    )
    (out / "performance.json").write_text(json.dumps(report, indent=2))
    return report


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(str(args.corpus))
    catalog = Catalog(corpus)
    queries = [load_query(args.queries, i) for i in range(len(PANEL_KEYS))]
    generator = RDKitPharmacophoreGenerator()
    provenance = dict(
        queries_sha256=hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        corpus=corpus_fingerprint(args.corpus),
        metal=metal.build_stamp(),
        cpu=cpu.build_stamp(),
        platform=platform.platform(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_driver_sha256=cpu_driver_sha256(),
        device=metal.extension().device_info(0),
        power=power_state(),
        warmups=args.warmup,
        repeats=args.repeats,
        provisional=args.provisional,
        panel_identities=[dict(mol_id=int(m), conf_idx=int(c))
                          for m, c in PANEL_KEYS],
        frequency_telemetry=None,
        core_residency=None,
        grade="advisory",
    )
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str)
    )
    records = []

    def panel(engine, mode, workers=None, save=False):
        times = []
        for identity, q in zip(PANEL_KEYS, queries):
            t = time.perf_counter()
            result = screen(
                q,
                corpus,
                engine=engine,
                config=configuration(mode, workers),
                color_generator=generator,
                level="conformer" if save else "molecule",
                k=corpus.n_conf if save else 100,
            )
            times.append(time.perf_counter() - t)
            if save:
                hit = result.hits[0]
                np.savez(
                    args.out / f"{mode}-{key_label(identity)}.npz",
                    scores=hit.scores, mol_ids=hit.mol_id,
                    conf_indices=hit.conf_idx,
                )
        return sum(times)

    with ExitStack() as locks:
        locks.enter_context(resource_lock("cpu", 0, lambda: False))
        locks.enter_context(resource_lock("metal", 0, lambda: False))
        if args.quality_reference:
            expected = json.loads(
                (args.quality_reference / "provenance.json").read_text()
            )
            if (
                expected["queries_sha256"] != provenance["queries_sha256"]
                or expected["corpus_sha256"] != provenance["corpus"]["sha256"]
            ):
                raise ValueError("quality reference corpus/query fingerprints differ")
            with ExitStack() as stack:
                e = Engine(backend="metal")
                stack.callback(e.close)
                for mode in ("shape", "rescore", "combo"):
                    elapsed = panel(e, mode, save=True)
                    print(
                        json.dumps(dict(quality_capture=mode, seconds=elapsed)),
                        flush=True,
                    )
            report = quality(args.quality_reference, args.out, catalog, queries)
            (args.out / "quality.json").write_text(json.dumps(report, indent=2))
        if args.quality_only:
            return
        # Select the CPU baseline on an independent full-panel preliminary pass.
        preliminary_binding = dict(
            queries_sha256=provenance["queries_sha256"],
            corpus_sha256=provenance["corpus"]["sha256"],
            cpu_src_sha256=provenance["cpu"]["cpu_src_sha256"],
            cpu_driver_sha256=cpu_driver_sha256(),
            workers=sorted(args.cpu_workers),
        )
        if args.preliminary:
            saved = json.loads(args.preliminary.read_text())
            if saved["binding"] != preliminary_binding:
                raise ValueError("CPU preliminary inputs/build/worker set differ")
            candidates = {int(k): float(v) for k, v in saved["seconds"].items()}
            if sorted(candidates) != sorted(args.cpu_workers) or any(
                not np.isfinite(v) or v <= 0 for v in candidates.values()
            ):
                raise ValueError("invalid CPU preliminary measurements")
        else:
            candidates = {}
            for workers in args.cpu_workers:
                e = Engine(backend="cpu")
                try:
                    candidates[workers] = sum(
                        panel(e, mode, workers)
                        for mode in ("shape", "rescore", "combo")
                    )
                finally:
                    e.close()
                print(
                    json.dumps(
                        dict(
                            cpu_preliminary_workers=workers, seconds=candidates[workers]
                        )
                    ),
                    flush=True,
                )
        (args.out / "cpu-preliminary.json").write_text(
            json.dumps(
                dict(
                    binding=preliminary_binding,
                    seconds=candidates,
                    reused_from=str(args.preliminary) if args.preliminary else None,
                ),
                indent=2,
            )
        )
        workers = min(candidates, key=candidates.get)
        for mode in ("shape", "rescore", "combo"):
            ce, me = Engine(backend="cpu"), Engine(backend="metal")
            precise = Engine(backend="metal")
            precise._backend_context = metal.create_context(fast_search_exp2=False)
            try:
                for _ in range(args.warmup):
                    panel(ce, mode, workers)
                    panel(me, mode)
                    panel(precise, mode)
                for repeat in range(args.repeats):
                    order = ["cpu", "metal_fast", "metal_precise"]
                    order = order[repeat % 3 :] + order[: repeat % 3]
                    for name in order:
                        before = dict(
                            load=psutil.getloadavg(), swap=psutil.swap_memory().used
                        )
                        elapsed = panel(
                            {"cpu": ce, "metal_fast": me, "metal_precise": precise}[
                                name
                            ],
                            mode,
                            workers if name == "cpu" else None,
                        )
                        record = dict(
                            mode=mode,
                            backend=name,
                            repeat=repeat,
                            seconds=elapsed,
                            operating_before=before,
                            operating_after=dict(
                                load=psutil.getloadavg(), swap=psutil.swap_memory().used
                            ),
                            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        )
                        records.append(record)
                        with (args.out / "timings.jsonl").open("a") as f:
                            f.write(json.dumps(record) + "\n")
                        print(json.dumps(record), flush=True)
            finally:
                ce.close()
                me.close()
                precise.close()
        summarize_existing(args.out, args.repeats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--queries", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quality-reference", type=Path)
    p.add_argument("--quality-only", action="store_true")
    p.add_argument("--provisional", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--summarize-existing",
        action="store_true",
        help="summarize the first matched repeats from existing receipts",
    )
    p.add_argument(
        "--preliminary",
        type=Path,
        help="reuse an identical-input CPU baseline selection receipt",
    )
    p.add_argument("--cpu-workers", type=int, nargs="+", default=[10, 12, 14])
    args = p.parse_args()
    if args.warmup < 0 or args.repeats < 1 or any(w < 1 for w in args.cpu_workers):
        p.error("invalid repeat/worker count")
    if not args.provisional and (args.warmup < 3 or args.repeats < 5):
        p.error("short runs require --provisional")
    if args.summarize_existing:
        summarize_existing(args.out, args.repeats)
        return
    if args.corpus is None or args.queries is None:
        p.error("--corpus and --queries are required for a new campaign")
    if args.quality_only and not args.quality_reference:
        p.error("--quality-only requires --quality-reference")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("output directory must be empty")
    run(args)


if __name__ == "__main__":
    main()

"""Release-candidate gates: independent queries and full-API cold/warm timings."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from bench.apple_silicon import (PANEL_KEYS, corpus_fingerprint, feature_payload,
                                 key_label, load_query)
from bench.cpu_darwin import power_state
from bench.metal import (aligned, audit_poses, compare, configuration, dispatch,
                         dump, ranking, ranking_passes)
from roshambo3 import screen
from roshambo3.backend import cpu, metal
from roshambo3.engine import Engine
from roshambo3.engine.query import from_arrays
from roshambo3.format import Catalog, Corpus
from roshambo3.metrics.ranking import canonical_ids
from roshambo3.pharmacophore import RDKitPharmacophoreGenerator
from roshambo3.web.gpu import resource_lock

CONTROL = dict(field_sampler="manual", color_field_min_sources=0,
               color_field_sampler="manual", specialize_color=True,
               factor_color_torque=True)
CANDIDATE = dict(field_sampler="texture16", color_field_sampler="texture16",
                 color_field_min_sources=12, color_field_spacing_a=0.25,
                 color_field_types=(1, 2), specialize_color=True,
                 factor_color_torque=True)


def freeze(args, corpus, catalog):
    """Select by counts/size only; exclude every frozen-panel molecule."""
    all_gids = np.arange(corpus.n_conf, dtype=np.int64)
    mids = catalog.mol_ids(all_gids)
    conf_indices = catalog.conf_indices(all_gids)
    panel_gids = np.asarray([catalog.gid_for(*key) for key in PANEL_KEYS], np.int64)
    panel_mids = catalog.mol_ids(panel_gids)
    eligible = ~np.isin(mids, panel_mids)
    count_groups = []
    for gi in range(len(corpus.group_table)):
        with corpus.group(gi) as group:
            count_groups.append(group.counts)
    counts = np.concatenate(count_groups)
    rows, selected = [], []
    targets = [(1, n) for n in (11, 12, 13)] + [(2, n) for n in (11, 12, 13)]
    for plane, n in targets + [(1, "dense"), (0, "large")]:
        mask = eligible.copy()
        mask &= ~np.isin(mids, np.asarray([key[0] for key in selected], np.int64))
        if n == "dense":
            mask &= counts[:, plane] >= 24
        elif n == "large":
            mask &= counts[:, 0] >= np.quantile(counts[:, 0], .99)
            mask &= np.maximum(counts[:, 1], counts[:, 2]) >= 12
        else:
            mask &= counts[:, plane] == n
        candidates = all_gids[mask]
        if not len(candidates):
            raise ValueError(f"no independent query for {plane}/{n}")
        # Deterministic median corpus row among eligible records; persist the
        # stable pair identity rather than this corpus-local row number.
        gid = int(candidates[len(candidates) // 2])
        identity = (int(mids[gid]), int(conf_indices[gid]))
        selected.append(identity)
        xyz = catalog.coordinates(np.array([gid]))[0]
        ref = catalog.locate([gid])[0]
        _, pseudo, types = feature_payload(corpus, ref, xyz)
        plane_xyz = pseudo[types == plane] if plane else xyz
        rows.append(dict(identity=dict(mol_id=identity[0], conf_idx=identity[1]),
                         target=[plane, n],
                         population=len(candidates), counts=counts[gid].tolist(),
                         color_bbox_span_a=np.ptp(plane_xyz, axis=0).tolist()))
    generator = RDKitPharmacophoreGenerator()
    shapes, colors = [], []
    for identity in selected:
        gid = catalog.gid_for(*identity)
        xyz = catalog.coordinates([gid])[0]
        ref = catalog.locate([gid])[0]
        _, pseudo, types = feature_payload(corpus, ref, xyz)
        shapes.append(xyz[:ref.n_real])
        colors.append((pseudo, types))
    names = [f"{mol_id}:{conf_idx}" for mol_id, conf_idx in selected]
    queries = from_arrays(shapes, names, color=colors,
                          color_generator=generator, backend="cpu")
    np.savez(args.out / "queries.npz", coords=queries.coords, types=queries.types,
             n_real=queries.n_real, names=np.array(queries.names), vaa=queries.vaa,
             interaction_r=queries.interaction_r, interaction_p=queries.interaction_p)
    dump(args.out / "selection.json", dict(policy="count/size-only deterministic median conformer row; exclude frozen-panel molecules and previous selections; no outcome-based selection", queries=rows))
    print(json.dumps(rows), flush=True)


def query_identities(path):
    with np.load(path, allow_pickle=False) as q:
        return [tuple(map(int, str(value).split(":"))) for value in q["names"]]


def objective_rows(scores, mode):
    if mode == "rescore":
        scores = scores.copy()
        scores[:, 0] = scores[:, 1]
    return scores


def quality(args, corpus, catalog):
    generator = RDKitPharmacophoreGenerator()
    identities = query_identities(args.queries)
    rows, calls = [], []
    with ExitStack() as stack:
        stack.enter_context(resource_lock("cpu", 0, lambda: False))
        stack.enter_context(resource_lock("metal", 0, lambda: False))
        engines = {"analytic": Engine(backend="metal", backend_options=CONTROL),
                   "candidate": Engine(backend="metal", backend_options=None if args.default_candidate else CANDIDATE),
                   "cpu": Engine(backend="cpu")}
        for engine in engines.values():
            stack.callback(engine.close)
        for qi, identity in enumerate(identities):
            label = key_label(identity)
            query = load_query(args.queries, qi)
            for mode in args.modes:
                variants = ("analytic", "candidate", "cpu") if mode == "combo" else ("analytic", "candidate")
                for name in variants:
                    cfg = configuration(mode, 14 if name == "cpu" else None)
                    if name == "cpu":
                        cfg = replace(cfg, cpu_worker=replace(cfg.cpu_worker,
                                      self_overlap_policy="recompute", optimizer_exp_mode="baseline"))
                    start = time.perf_counter()
                    hit = screen(query, corpus, engine=engines[name], config=cfg,
                                 color_generator=generator, level="conformer",
                                 k=corpus.n_conf).hits[0]
                    np.savez(args.out / f"{mode}-{name}-{label}.npz",
                             mol_ids=hit.mol_id, conf_indices=hit.conf_idx,
                             scores=hit.scores)
                    call = dict(query_identity=dict(mol_id=identity[0], conf_idx=identity[1]),
                                mode=mode, variant=name,
                                wall_seconds=time.perf_counter()-start)
                    if name != "cpu":
                        call.update(dispatch(engines[name]._backend_context,
                                             CANDIDATE if name == "candidate" else CONTROL,
                                             query, mode))
                    calls.append(call)
                    dump(args.out / "dispatch.json", calls)
                mids, cis, a, b = aligned(
                    args.out / f"{mode}-analytic-{label}.npz",
                    args.out / f"{mode}-candidate-{label}.npz")
                comparisons = {"analytic_metal": compare(
                    objective_rows(a, mode), objective_rows(b, mode), mids, cis)}
                references = [a, b]
                if mode == "combo":
                    _, _, c, _ = aligned(args.out / f"{mode}-cpu-{label}.npz",
                                          args.out / f"{mode}-candidate-{label}.npz")
                    comparisons["scientific_cpu"] = compare(c, b, mids, cis)
                    comparisons["analytic_control_vs_cpu"] = compare(c, a, mids, cis)
                    references.append(c)
                selected = set(np.argsort(np.abs(a[:, 0].astype(float) - b[:, 0]))[-8:].tolist())
                pair_index = {(int(mid), int(ci)): i for i, (mid, ci) in enumerate(zip(mids, cis))}
                for values in references:
                    winner_mids, winner_cis = ranking(
                        objective_rows(values, mode), mids, cis, k=105)
                    selected.update(pair_index[(int(m), int(c))]
                                    for m, c in zip(winner_mids[:10], winner_cis[:10]))
                    selected.update(pair_index[(int(m), int(c))]
                                    for m, c in zip(winner_mids[94:105], winner_cis[94:105]))
                for comp in comparisons.values():
                    for change in comp["changes"]:
                        selected.add(pair_index[(change["molecule_id"], change["a_conf_idx"])])
                        selected.add(pair_index[(change["molecule_id"], change["b_conf_idx"])])
                ix = np.array(sorted(selected), np.int64)
                audit_pairs = list(zip(mids[ix].tolist(), cis[ix].tolist()))
                audit_gids = np.asarray([catalog.gid_for(*pair) for pair in audit_pairs], np.int64)
                audit = audit_poses(query, catalog, audit_gids, a[ix], b[ix], mode)
                row = dict(query_identity=dict(mol_id=identity[0], conf_idx=identity[1]),
                           mode=mode, comparisons=comparisons,
                           fixed_pose_audit=audit,
                           passed=all(ranking_passes(c) for c in comparisons.values()) and audit["passed"])
                rows.append(row)
                dump(args.out / "quality.json", dict(complete=False, queries=rows))
                print(json.dumps({"query_identity": identity, "mode": mode, "passed": row["passed"],
                      "comparisons": {k: [v['top1'], v['top10_order'], v['top100_overlap']] for k,v in comparisons.items()},
                      "audit": audit['passed']}), flush=True)
    dump(args.out / "quality.json", dict(complete=True, passed=all(r["passed"] for r in rows), queries=rows))


def parity(args, corpus, catalog):
    """Carry forward full oracle/ranking captures only for bitwise-equal output."""
    previous = json.loads((args.reference / "provenance.json").read_text())
    current = json.loads((args.out / "provenance.json").read_text())
    if (previous["queries_sha256"] != current["queries_sha256"]
        or previous["corpus"]["sha256"] != current["corpus"]["sha256"]
        or previous["metal"]["sources"] != current["metal"]["sources"]
        or previous["candidate"] != current["candidate"]):
        raise ValueError("parity reference inputs/shader/policy differ")
    reference_quality = json.loads((args.reference / "quality.json").read_text())
    if not reference_quality["complete"] or not reference_quality["passed"]:
        raise ValueError("parity reference must have passed complete quality gates")
    generator = RDKitPharmacophoreGenerator()
    rows = []
    with ExitStack() as stack:
        stack.enter_context(resource_lock("cpu", 0, lambda: False))
        stack.enter_context(resource_lock("metal", 0, lambda: False))
        engine = Engine(backend="metal")
        stack.callback(engine.close)
        for qi, identity in enumerate(query_identities(args.queries)):
            query = load_query(args.queries, qi)
            for mode in args.modes:
                hit = screen(query, corpus, engine=engine, config=configuration(mode),
                    color_generator=generator, level="conformer", k=corpus.n_conf).hits[0]
                path = args.reference / f"{mode}-candidate-{key_label(identity)}.npz"
                with np.load(path) as previous_hit:
                    a = np.argsort(canonical_ids(previous_hit["mol_ids"],
                                                 previous_hit["conf_indices"]))
                    b = np.argsort(canonical_ids(hit.mol_id, hit.conf_idx))
                    np.testing.assert_array_equal(hit.mol_id[b], previous_hit["mol_ids"][a])
                    np.testing.assert_array_equal(hit.conf_idx[b], previous_hit["conf_indices"][a])
                    np.testing.assert_array_equal(hit.scores[b], previous_hit["scores"][a])
                rows.append(dict(query_identity=dict(mol_id=identity[0], conf_idx=identity[1]),
                                 mode=mode, bitwise_equal=True,
                                 reference=str(path), **dispatch(engine._context(), CANDIDATE, query, mode)))
                dump(args.out / "parity.json", dict(complete=False, queries=rows))
                print(json.dumps(dict(query_identity=identity, mode=mode, bitwise_equal=True)), flush=True)
    dump(args.out / "parity.json", dict(complete=True, passed=True, queries=rows,
         reference_provenance_sha256=hashlib.sha256((args.reference / "provenance.json").read_bytes()).hexdigest(),
         reference_quality_sha256=hashlib.sha256((args.reference / "quality.json").read_bytes()).hexdigest()))


def performance(args, corpus, catalog):
    generator = RDKitPharmacophoreGenerator()
    identities = query_identities(args.queries)
    performance_keys = [(13625945, 0), (12673788, 0)]
    records, cold = [], []
    with ExitStack() as locks:
        locks.enter_context(resource_lock("cpu", 0, lambda: False))
        locks.enter_context(resource_lock("metal", 0, lambda: False))
        for mode in args.modes:
            for identity in performance_keys:
                label = key_label(identity)
                query = load_query(args.queries, identities.index(identity))
                with ExitStack() as stack:
                    engines = {"analytic": Engine(backend="metal", backend_options=CONTROL),
                               "candidate": Engine(backend="metal", backend_options=None if args.default_candidate else CANDIDATE),
                               "cpu14": Engine(backend="cpu")}
                    for e in engines.values(): stack.callback(e.close)
                    def call(name):
                        e = engines[name]
                        t = time.perf_counter()
                        result = screen(query, corpus, engine=e,
                            config=configuration(mode, 14 if name == "cpu14" else None),
                            color_generator=generator, level="molecule", k=100)
                        elapsed = time.perf_counter() - t
                        if len(result.hits[0].mol_id) != 100:
                            raise ValueError("top-K output incomplete")
                        return elapsed
                    # Cold first screen on fresh Engines, including runtime shader compile.
                    for name in engines:
                        elapsed = call(name)
                        cold.append(dict(mode=mode, query_identity=identity,
                                         variant=name, wall_seconds=elapsed))
                    dump(args.out / "cold.json", cold)
                    for _ in range(3):
                        for name in engines: call(name)
                    for repeat in range(5):
                        order = list(engines)
                        order = order[repeat % 3:] + order[:repeat % 3]
                        for name in order:
                            elapsed = call(name)
                            record = dict(mode=mode, query_identity=identity, variant=name,
                                          repeat=repeat, wall_seconds=elapsed)
                            records.append(record)
                            with (args.out / "timings.jsonl").open("a") as f:
                                f.write(json.dumps(record) + "\n")
                            print(json.dumps(record), flush=True)
    summary = []
    for mode in args.modes:
        aggregate = {}
        for name in ("analytic", "candidate", "cpu14"):
            times = []
            for identity in performance_keys:
                cell = [r['wall_seconds'] for r in records if r['mode']==mode and r['query_identity']==identity and r['variant']==name]
                if len(cell) != 5: raise ValueError("incomplete timing cell")
                med = float(np.median(cell)); times.append(med)
                summary.append(dict(mode=mode, query_identity=identity, variant=name, wall_seconds=med,
                                    conformers_per_second=corpus.n_conf / med))
            aggregate[name] = sum(times)
        summary.append(dict(mode=mode, aggregate_wall_seconds=aggregate,
                       cpu_speedup=aggregate['cpu14']/aggregate['candidate'],
                       analytic_metal_speedup=aggregate['analytic']/aggregate['candidate']))
    dump(args.out / "summary.json", summary)
    dump(args.out / "performance.json", dict(complete=True,
         cpu_speed_gate=all(x['cpu_speedup']>=1.5 for x in summary if 'cpu_speedup' in x),
         modes=[x for x in summary if 'cpu_speedup' in x], power_after=power_state()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('freeze','quality','performance','parity'), required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--queries', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--default-candidate', action='store_true')
    p.add_argument('--modes', nargs='+', choices=('shape','rescore','combo'), default=['shape','combo'])
    args=p.parse_args()
    if args.out.exists(): p.error('output directory must not exist')
    if args.phase != 'freeze' and args.queries is None: p.error('--queries is required')
    if args.phase == 'parity' and args.reference is None: p.error('--reference is required')
    args.out.mkdir(parents=True)
    corpus=Corpus(str(args.corpus)); catalog=Catalog(corpus)
    dump(args.out/'provenance.json',dict(phase=args.phase, corpus=corpus_fingerprint(args.corpus),
         queries_sha256=hashlib.sha256(args.queries.read_bytes()).hexdigest() if args.queries else None,
         panel_identities=[dict(mol_id=int(m), conf_idx=int(c)) for m, c in PANEL_KEYS],
         candidate=CANDIDATE, candidate_uses_defaults=args.default_candidate, control=CONTROL, metal=metal.build_stamp(), cpu=cpu.build_stamp(),
         driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), power_before=power_state(),
         modes=args.modes, shape_policy='existing width-dependent API policy',
         cpu_quality='14 workers; recomputed self; baseline exponentials',
         cpu_performance='public CPU policy; 14 workers; stored self allowed; default optimizer exponentials',
         performance=dict(warmups=3, repeats=5, cold_first_call_separate=True,
             queries=[dict(mol_id=m, conf_idx=c) for m, c in ((13625945, 0), (12673788, 0))],
             k=100, level='molecule',
             scope='full screen API on resident Corpus/QuerySet; includes source/decode/selection/readback; cold includes Engine context compile; excludes process import and file query preparation'),
         hardware='M4 Pro; uncontrolled frequency/core residency; no M1/M2 hardware'))
    {'freeze':freeze,'quality':quality,'performance':performance,'parity':parity}[args.phase](args,corpus,catalog)

if __name__ == '__main__': main()

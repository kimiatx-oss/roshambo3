"""Synthetic CPU binding concurrency control, separate from screening.

Geometry is reused from four small RDKit prototypes; names are synthetic unique
identities. No pose search, accuracy, or end-to-end throughput is measured.
Builds require a root-allocated build window; measurements require a released
quiet CPU window. Both are separate from --qualify's untimed correctness work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time

import numpy as np

from bench.format3_cpu_block_timing import (NATIVE_STAGES, _no_foreign_load,
    _write_line, affinity_for_workers, describe_arms)

SMILES = ('CO', 'CN', 'CCO', 'CCN')
STAGES = ('decode_wall_s', 'scorer_binding_wall_s', *NATIVE_STAGES)


def reused_geometry_inputs(n_molecules=2048, conformers=128):
    """Four embeddings, each jittered into one prototype, then cloned by name."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    if n_molecules < 4 or n_molecules % 4 or not 1 <= conformers <= 4096:
        raise ValueError('molecule count must be a positive multiple of four; conformers in [1,4096]')
    rng = np.random.default_rng(29556)
    prototypes = []
    for i, smiles in enumerate(SMILES):
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(molecule, randomSeed=295+i) != 0:
            raise RuntimeError('could not embed binding-control prototype ' + smiles)
        xyz = molecule.GetConformer().GetPositions()
        molecule.RemoveAllConformers()
        for _ in range(conformers):
            conf = Chem.Conformer(molecule.GetNumAtoms())
            for atom, point in enumerate(xyz + rng.normal(0, .08, xyz.shape)):
                conf.SetAtomPosition(atom, point.tolist())
            molecule.AddConformer(conf, assignId=True)
        prototypes.append(molecule)
    molecules = []
    for index in range(n_molecules):
        molecule = Chem.Mol(prototypes[index % len(prototypes)])
        molecule.SetProp('_Name', f'binding-synthetic-{index}')
        molecules.append(molecule)
    return molecules


def build_corpora(root, *, workers, report_path, n_molecules=2048, conformers=128):
    from bench.gitinfo import get_git_info
    from roshambo3.prep import prepare
    from roshambo3.pharmacophore import PharmacophoreGenerator
    from roshambo3.format.r3g.reader import open_corpus, verify
    git = get_git_info()
    if git.dirty or git.sha == 'unknown':
        raise RuntimeError('new corpus builds require a clean committed checkout')
    repo = Path(__file__).resolve().parents[1]
    if Path(root).resolve().is_relative_to(repo) or Path(report_path).resolve().is_relative_to(repo):
        raise ValueError('new corpus output and build report must live outside the worktree')
    paths = {n: Path(root) / f'block-{n}' for n in (1024, 2048, 4096)}
    if any(path.exists() for path in paths.values()) or Path(report_path).exists():
        raise ValueError('build output exists; select a fresh build root and report path')
    if not 1 <= workers <= 8:
        raise ValueError('binding-control builder uses at most eight workers')
    started = time.perf_counter()
    report = dict(source=git.as_dict(), status='building', begin_unix_s=time.time(), variants=[],
        synthetic_molecule_names=n_molecules, geometry_prototypes=len(SMILES),
        prototype_smiles=SMILES, conformers_per_prototype=conformers,
        geometry_reuse_per_prototype=n_molecules//len(SMILES),
        geometry='one embedding per SMILES plus deterministic jitter; prototypes cloned under unique names',
        claims='synthetic identities and repeated geometry; no chemical-diversity or screening claim')
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(report_path).open('x') as file:
        def publish():
            file.seek(0)
            json.dump(report, file, indent=2)
            file.write('\n')
            file.truncate()
            file.flush()
        publish()
        try:
            report['stage'] = 'prototype construction'
            prototype_start = time.perf_counter()
            molecules = reused_geometry_inputs(n_molecules, conformers)
            report['prototype_wall_s'] = time.perf_counter() - prototype_start
            generator = PharmacophoreGenerator()
            corpora = {}
            for nominal, path in paths.items():
                if get_git_info().dirty or get_git_info().sha != git.sha:
                    raise RuntimeError('checkout changed between matched corpus builds')
                variant = dict(block_rows=nominal, path=str(path), prepare_begin_unix_s=time.time())
                report['variants'].append(variant)
                report['stage'] = f'prepare block_rows={nominal}'
                publish()
                print(f'BEGIN prepare block_rows={nominal} path={path}', flush=True)
                stage_start = time.perf_counter()
                prepare(molecules, str(path), workers=workers, block_rows=nominal,
                        color_generator=generator)
                variant.update(prepare_end_unix_s=time.time(), prepare_wall_s=time.perf_counter()-stage_start,
                               deep_begin_unix_s=time.time())
                report['stage'] = f'deep validation block_rows={nominal}'
                publish()
                print(f'BEGIN deep block_rows={nominal} path={path}', flush=True)
                stage_start = time.perf_counter()
                verify(str(path), tier='deep')
                variant.update(deep_end_unix_s=time.time(), deep_wall_s=time.perf_counter()-stage_start)
                corpus = open_corpus(str(path))
                build = json.loads((path / 'build.json').read_text())
                if corpus.descriptor['prep']['commit'] != git.sha or build.get('commit') != git.sha:
                    raise AssertionError('built corpus has null or mismatched preparation provenance')
                corpora[nominal] = corpus
                variant['complete'] = True
                publish()
                print(f'BUILT block_rows={nominal} rows={corpus.n_conf} path={path}', flush=True)
            if get_git_info().dirty or get_git_info().sha != git.sha:
                raise RuntimeError('checkout changed before matched builds completed')
            metadata = describe_arms(corpora, paths)
            if min(v['physical_blocks'] for v in metadata.values()) < 56:
                raise AssertionError('component control needs at least 56 distinct physical blocks in every arm')
            report.update(arms=metadata, rows=n_molecules*conformers, status='complete', stage='complete')
        except BaseException as error:
            report.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
            raise
        finally:
            report.update(end_unix_s=time.time(), total_wall_s=time.perf_counter()-started,
                          output_paths_present=[str(path) for path in paths.values() if path.exists()])
            publish()
    return report


def check_consumed_values(record, *, mode):
    """Compare after worker clocks stop; no per-row verifier competes for GIL."""
    from roshambo3.numerics import SHAPE_PREFACTOR
    expected = record.pop('_expected_shape')
    actual = record.pop('_consumed_shape')
    if mode == 'combo':
        expected = (expected.astype(np.float64) / float(SHAPE_PREFACTOR)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), expected.view(np.uint32))
    record['shape_values_sha256'] = hashlib.sha256(np.asarray(actual).tobytes()).hexdigest()
    if mode == 'combo':
        expected_color = record.pop('_expected_color')
        consumed_color = record.pop('_consumed_color')
        if expected_color.dtype != np.float64 or consumed_color.dtype != np.float64:
            raise AssertionError('stored color self values lost their f64 precision')
        np.testing.assert_array_equal(consumed_color.view(np.uint64), expected_color.view(np.uint64))
        record['color_self_f64_sha256'] = hashlib.sha256(consumed_color.tobytes()).hexdigest()


def run_binding_pass(corpus, query, mode, workers, *, measure=False, timeout_s=30.):
    from roshambo3.backend import cpu
    from roshambo3.engine import R3gSource
    from roshambo3.engine.r3g_cpu import R3gWorkerSelfCache
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    if not math.isfinite(timeout_s) or timeout_s <= 0 or workers < 1:
        raise ValueError('barrier timeout and workers must be positive')
    color = mode == 'combo'
    source = R3gSource(corpus, tile=False, color=color,
        color_generator=generator_from_descriptor(corpus.color) if color else None)
    descriptors = source.cpu_block_descriptors()
    if len(descriptors) < workers:
        source.close()
        raise ValueError('not enough distinct physical blocks for all requested participants')
    qc = np.ascontiguousarray(query.coords if color else query.coords[..., :3], np.float32)
    qt = np.ascontiguousarray(query.types if color else np.zeros(qc.shape[:2], np.int32), np.int32)
    qn = np.ascontiguousarray(query.n_real, np.int32)
    r = np.ascontiguousarray(query.interaction_r if color else np.ones((1,1)), np.float32)
    p = np.ascontiguousarray(query.interaction_p if color else np.zeros((1,1)), np.float32)
    cache = R3gWorkerSelfCache(source, 'b'*64)
    claimed_gate = threading.Barrier(workers, timeout=timeout_s)
    decoded_gate = threading.Barrier(workers, timeout=timeout_s)
    lock = threading.Lock()
    participants, claimed, records, failures = {}, {}, [], []
    active = set()
    peak = 0

    def callback(worker_id, item_id):
        nonlocal peak
        descriptor = descriptors[item_id]
        key = descriptor.group_index, descriptor.block_index
        try:
            with lock:
                first = worker_id not in participants
                if key in claimed:
                    raise AssertionError('physical block claimed twice')
                claimed[key] = worker_id
                if first:
                    participants[worker_id] = dict(worker_id=worker_id,
                        thread_id=threading.get_ident(), descriptor=list(key))
            if first:
                claimed_gate.wait()
            start = time.perf_counter() if measure else None
            decoded = source.decode_cpu_block(descriptor)
            elapsed_decode = time.perf_counter() - start if measure else None
            if first:
                # Every participant now owns a distinct fully decoded block.
                # Waiting here does not contribute to either measured stage.
                decoded_gate.wait()
            with lock:
                active.add(worker_id)
                peak = max(peak, len(active))
            start = time.perf_counter() if measure else None
            try:
                result = cache.block(cpu, descriptor, decoded, decoded.result.coords,
                                     qc, qt, qn, r, p, mode=mode)
                elapsed_binding = time.perf_counter() - start if measure else None
            finally:
                with lock:
                    active.remove(worker_id)
            handle, reason, hit, replaced = result
            route = cache.routes[key]
            if handle is None or reason is not None or hit or replaced or route['vaa'] != 'verified_stored' or route['color_reason'] is not None:
                raise AssertionError('binding left the fresh verified-stored route')
            record = dict(group=key[0], block=key[1], rows=descriptor.n_conf,
                gid_base=descriptor.gid_base, L=descriptor.library_width,
                worker_id=worker_id, thread_id=threading.get_ident(),
                route=dict(vaa=route['vaa'], color=route['color'], color_reason=route['color_reason']),
                _expected_shape=decoded.result.vaa)
            if color:
                library = handle.library
                if (not library.library_shape_from_stored_volume or not library.library_color_from_sidecar
                        or library.shape_terms != 0 or library.color_terms != 0 or route['color'] != 'sidecar'):
                    raise AssertionError('component binding recomputed library self values')
                record.update(_consumed_shape=library.shape_values,
                              _expected_color=decoded.result.color_self, _consumed_color=library.color_values)
            else:
                record['_consumed_shape'] = handle.vaa
            if measure:
                record.update(decode_wall_s=elapsed_decode, scorer_binding_wall_s=elapsed_binding,
                    **{name: float(getattr(decoded.result.timings, attr)) for name, attr in NATIVE_STAGES.items()})
            with lock:
                records.append(record)
        except BaseException as error:
            with lock:
                if not failures:
                    failures.append((error, error.__traceback__))
            claimed_gate.abort()
            decoded_gate.abort()
            raise
        finally:
            cache.release_block(descriptor)

    executor = cpu.CPUComboExecutor(workers, claim_chunk_rows=128)
    try:
        start = time.perf_counter() if measure else None
        try:
            executor.run_indexed(len(descriptors), callback,
                rows_per_item=max(128, max(d.n_conf for d in descriptors)))
        except BaseException:
            if failures:
                error, traceback = failures[0]
                raise error.with_traceback(traceback)
            raise
        submission_wall = time.perf_counter() - start if measure else None
        executor_stats = executor.stats
        if not measure:
            executor_stats = {name: executor_stats[name] for name in (
                'submissions', 'work_items_completed', 'work_items_completed_by_worker',
                'claims_by_worker', 'last_claims_by_worker', 'worker_placement')}
    finally:
        executor.close()
        source.close()
    if len(participants) != workers or len({p['thread_id'] for p in participants.values()}) != workers:
        raise AssertionError('first-claim gate did not contain distinct native workers')
    if len({tuple(p['descriptor']) for p in participants.values()}) != workers:
        raise AssertionError('gate participants did not own distinct physical blocks')
    if len(records) != len(descriptors) or sum(b['rows'] for b in records) != corpus.n_conf:
        raise AssertionError('component pass did not cover the actual corpus once')
    worker_items = Counter(b['worker_id'] for b in records)
    if (executor_stats['submissions'] != 1 or executor_stats['work_items_completed'] != len(descriptors)
            or executor_stats['work_items_completed_by_worker'] != [worker_items[i] for i in range(workers)]):
        raise AssertionError('native ownership counters disagree with observed block callbacks')
    if active or cache.entries or cache.retained_bytes:
        raise AssertionError('component binding left an active claim or retained handle')
    for record in records:
        check_consumed_values(record, mode=mode)
    result = dict(rows=corpus.n_conf, blocks=len(records), n_queries=query.n_conformers,
        requested_workers=workers, first_claim_participants=sorted(participants.values(), key=lambda p: p['worker_id']),
        post_release_binding_invocation_peak=peak, barrier_timeout_s=timeout_s,
        worker_blocks=dict(Counter(b['worker_id'] for b in records)),
        stored_vaa_verified_blocks=len(records), color_self_section_blocks=len(records) if color else 0,
        library_shape_self_evaluations=0, library_color_self_evaluations=0,
        self_cache_peak_blocks=cache.peak_entries, retained_self_handle_bytes=0,
        consumed_shape_bits_equal=True, consumed_color_f64_bits_equal=color,
        executor_stats=executor_stats, block_records=sorted(records, key=lambda b: (b['group'], b['block'])))
    if measure:
        result['binding_submission_wall_s'] = submission_wall
        result['stage_us_per_row'] = {name: math.fsum(b[name] for b in records) / corpus.n_conf * 1e6 for name in STAGES}
    return result


def summarize_binding(passes):
    output = []
    for cell in sorted({(p['workers'], p['mode'], p['block_rows']) for p in passes}):
        selected = [p for p in passes if (p['workers'], p['mode'], p['block_rows']) == cell]
        output.append(dict(workers=cell[0], mode=cell[1], block_rows=cell[2], repeats=len(selected),
            stage_us_per_row={name: dict(median=statistics.median(p['stage_us_per_row'][name] for p in selected),
                min=min(p['stage_us_per_row'][name] for p in selected),
                max=max(p['stage_us_per_row'][name] for p in selected),
                values=[p['stage_us_per_row'][name] for p in selected]) for name in STAGES},
            binding_invocation_peak=[p['post_release_binding_invocation_peak'] for p in selected]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    build = commands.add_parser('build')
    build.add_argument('--root', type=Path, required=True)
    build.add_argument('--report', type=Path, required=True)
    build.add_argument('--workers', type=int, default=8)
    build.add_argument('--molecules', type=int, default=2048)
    build.add_argument('--conformers', type=int, default=128)
    for action in ('qualify', 'measure'):
        campaign = commands.add_parser(action)
        campaign.add_argument('--corpus', action='append', required=True, metavar='ROWS=PATH')
        campaign.add_argument('--queries', type=Path, required=True)
        campaign.add_argument('--workers', default='8,28,56')
        campaign.add_argument('--modes', default='shape,combo')
        campaign.add_argument('--warmup', type=int)
        campaign.add_argument('--repeats', type=int)
        campaign.add_argument('--barrier-timeout', type=float, default=30.)
        campaign.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'build':
        build_corpora(args.root, workers=args.workers, report_path=args.report,
                      n_molecules=args.molecules, conformers=args.conformers)
        return
    measure = args.action == 'measure'
    repeats = args.repeats if args.repeats is not None else (11 if measure else 1)
    warmup = args.warmup if args.warmup is not None else (3 if measure else 0)
    workers = [int(s) for s in args.workers.split(',')]
    modes = args.modes.split(',')
    if (repeats < 1 or warmup < 0 or any(w < 1 for w in workers)
            or any(mode not in ('shape', 'combo') for mode in modes)):
        parser.error('invalid repeats, warmup, workers or modes')
    paths = {int(s.split('=', 1)[0]): Path(s.split('=', 1)[1]) for s in args.corpus}
    if len(paths) != len(args.corpus):
        parser.error('duplicate nominal corpus label')
    from bench.cpu import cpu_identity
    from bench.cpu_run import CpuRun
    from bench.gitinfo import get_git_info
    from bench.numerics_backends import verified_build
    from roshambo3.api import load_queries
    from roshambo3.format.r3g.reader import open_corpus
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    if get_git_info().source_dirty:
        raise RuntimeError('component campaigns require committed clean source')
    cpu_build = verified_build('cpu')
    corpora = {n: open_corpus(str(path)) for n, path in paths.items()}
    metadata = describe_arms(corpora, paths)
    if min(m['physical_blocks'] for m in metadata.values()) < max(workers):
        raise ValueError('every arm needs a distinct physical block for every requested participant')
    reference = corpora[max(corpora)]
    if any(c.fingerprint != reference.fingerprint or c.n_conf != reference.n_conf for c in corpora.values()):
        raise AssertionError('component arms changed semantics or actual row count')
    generator = generator_from_descriptor(reference.color)
    queries = {mode: load_queries(args.queries, backend='cpu',
               color_generator=generator if mode == 'combo' else None) for mode in modes}
    output = {name: Path(str(args.output_prefix) + suffix) for name, suffix in
              (('passes', '-passes.jsonl'), ('runs', '-runs.jsonl'), ('summary', '-summary.json'))}
    if any(p.exists() for p in output.values()):
        raise ValueError('component output exists; select a fresh output prefix')
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    original_affinity = set(os.sched_getaffinity(0))
    identity = cpu_identity().as_dict()
    all_passes = []
    try:
        with output['passes'].open('x') as file:
            _write_line(file, dict(kind='header', action=args.action, args=vars(args), source=get_git_info().as_dict(),
                cpu_build=cpu_build, cpu_identity=identity, arms=metadata,
                query_sha256=hashlib.sha256(args.queries.read_bytes()).hexdigest(),
                claims='synthetic geometry-reuse binding component only; no screening or throughput claim'))
            for count in workers:
                cpus = affinity_for_workers(count, original_affinity)
                for mode in modes:
                    def execute():
                        sizes = sorted(corpora)
                        for rep in range(-warmup, repeats):
                            offset = (rep + warmup) % len(sizes)
                            for n in sizes[offset:] + sizes[:offset]:
                                state = _no_foreign_load() if measure else None
                                record = run_binding_pass(corpora[n], queries[mode], mode, count,
                                    measure=measure and rep >= 0, timeout_s=args.barrier_timeout)
                                if rep < 0:
                                    print(f'WARMUP binding {mode} workers={count} block_rows={n} rep={rep}', flush=True)
                                    continue
                                record.update(kind='pass', workers=count, mode=mode, block_rows=n, repeat=rep,
                                    cpu_state_before=state, affinity=cpus,
                                    physical_oversubscription=count > identity['physical_cores'],
                                    logical_oversubscription=count > identity['logical_cpus'])
                                _write_line(file, record)
                                all_passes.append(record)
                                print(f'PASS binding {mode} workers={count} block_rows={n} rep={rep}', flush=True)
                    if not measure:
                        os.sched_setaffinity(0, cpus)
                        execute()
                    else:
                        with CpuRun('format3-cpu-binding-control', decision_grade=True, cpus=cpus, threads=count,
                                    measurement_regime='calibration') as run:
                            if run.lock_busy_reason:
                                raise RuntimeError('CPU lock unavailable: ' + run.lock_busy_reason)
                            _no_foreign_load()
                            execute()
                            selected = [p for p in all_passes if p['workers'] == count and p['mode'] == mode]
                            run.timing = dict(n_warmup=warmup, n_repeats=repeats,
                                sustained_s=sum(p['binding_submission_wall_s'] for p in selected),
                                component_summary=summarize_binding(selected))
                            run.config = dict(mode=mode, workers=count, block_rows=sorted(corpora),
                                native_worker_barrier=True, max_results=0, no_pose_scoring=True)
                            run.dataset = dict(name='synthetic reused-geometry binding control',
                                n_conformers=reference.n_conf, paths={n: str(p) for n,p in paths.items()})
                            run.query_panel = dict(name=str(args.queries),
                                sha256=hashlib.sha256(args.queries.read_bytes()).hexdigest(), ids=list(queries[mode].names))
                            run.audits = dict(distinct_native_workers_and_descriptors=True,
                                consumed_color_self_f64_bits_equal=mode == 'combo', route='verified_stored')
                            run.notes = ('Native stages are decode subsets; barriers excluded from stage clocks. '
                                         'Submission wall includes synchronization. No end-to-end throughput claim.')
                        run.write(output['runs'])
                        _write_line(file, dict(kind='grade', workers=count, mode=mode, status=run.status))
            summary = dict(status='complete', kind=args.action, passes=len(all_passes),
                summary=summarize_binding(all_passes) if measure else [],
                no_pose_scoring=True, no_end_to_end_throughput_claim=True)
            output['summary'].write_text(json.dumps(summary, indent=2) + '\n')
            _write_line(file, {**summary, 'campaign_kind': summary['kind'], 'kind': 'footer'})
    finally:
        os.sched_setaffinity(0, original_affinity)


if __name__ == '__main__':
    main()

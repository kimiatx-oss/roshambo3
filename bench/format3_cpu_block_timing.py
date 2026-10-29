"""Format-3 CPU block timers and interleaved block-size campaign.

--qualify exercises the hooks and bitwise controls without publishing timing
values. --measure is reserved for a separately released quiet CPU window.
Existing corpora are read-only; this harness never builds or modifies them.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time

import numpy as np

NATIVE_STAGES = {'native_decode_s': 'decode', 'native_color_s': 'color',
                 'native_self_copy_s': 'self_overlap'}
BLOCK_STAGES = ('decode_wall_s', 'scorer_binding_wall_s', 'block_wall_s', *NATIVE_STAGES)


class BlockTimer:
    """Decode entry through the same owning worker's progress callback.

    Native decode/color/self-copy intervals are SUBSETS of decode_wall_s.
    Scorer binding is a separate stage after decode. No native interval is
    added to decode_wall_s to construct a total. block_wall_s
    includes scoring/top-K ingestion and the wait for progress publication.
    """
    def __init__(self, *, measure):
        self.measure = measure
        self.local = threading.local()
        self.done = []
        self.lock = threading.Lock()

    def start(self, descriptor):
        if getattr(self.local, 'block', None) is not None:
            raise AssertionError('worker started a block before ending its predecessor')
        block = dict(group=int(descriptor.group_index), block=int(descriptor.block_index),
                     rows=int(descriptor.n_conf), L=int(descriptor.library_width),
                     gid_base=int(descriptor.gid_base))
        if self.measure:
            block['_start'] = time.perf_counter()
        self.local.block = block
        return block

    def decoded(self, decoded, elapsed=None):
        block = self.local.block
        block['_native'] = {name: float(getattr(decoded.result.timings, attr))
                            for name, attr in NATIVE_STAGES.items()}
        if self.measure:
            block['decode_wall_s'] = elapsed
            block.update(block['_native'])
        if decoded.descriptor.gid_base != block['gid_base']:
            raise AssertionError('decoder changed the timed descriptor identity')

    def bound(self, cache, descriptor, result, elapsed=None):
        block = self.local.block
        if (block['group'], block['block']) != (descriptor.group_index, descriptor.block_index):
            raise AssertionError('binding belongs to another block')
        if 'route' in block:
            raise AssertionError('a block bound its self values more than once')
        handle, reason, hit, replaced = result
        if handle is None or reason is not None or hit or replaced:
            raise AssertionError('format-3 timed claim did not prepare its fresh self handle')
        route = cache.routes[descriptor.group_index, descriptor.block_index]
        block['route'] = dict(vaa=route['vaa'], color=route['color'],
                              color_reason=route['color_reason'])
        if self.measure:
            block['scorer_binding_wall_s'] = elapsed

    def end(self, _stats):
        block = self.local.block
        if block is None or 'route' not in block or '_native' not in block:
            raise AssertionError('progress does not match a decoded/bound block')
        if self.measure:
            block['block_wall_s'] = time.perf_counter() - block.pop('_start')
        self.local.block = None
        with self.lock:
            self.done.append(block)

    def finish(self, source, scan, *, color):
        expected = {(d.group_index, d.block_index): d.n_conf
                    for d in source.cpu_block_descriptors()}
        keys = Counter((b['group'], b['block']) for b in self.done)
        if keys != Counter({key: 1 for key in expected}):
            raise AssertionError('block timer did not see each selected block exactly once')
        if any(b['rows'] != expected[b['group'], b['block']] for b in self.done):
            raise AssertionError('block timer row count differs from its descriptor')
        if sum(b['rows'] for b in self.done) != scan['rows_decoded']:
            raise AssertionError('block timer rows do not cover the scored scan')
        for name, counter in (('native_decode_s', 'native_decode_s'),
                              ('native_color_s', 'native_color_derivation_s'),
                              ('native_self_copy_s', 'native_self_binding_s')):
            total = math.fsum(b['_native'][name] for b in self.done)
            if not math.isclose(total, float(scan[counter]), rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError('native timer totals disagree with engine counter: ' + name)
        for block in self.done:
            route = block['route']
            if route['vaa'] != 'verified_stored' or route['color_reason'] is not None:
                raise AssertionError('timed block left the authenticated stored route')
            if color and route['color'] != 'sidecar':
                raise AssertionError('combo timed block recomputed color self')
        # The qualification report contains no timing values, including native
        # clocks which production decoding necessarily collects internally.
        return [{k: v for k, v in b.items() if k != '_native'}
                for b in sorted(self.done, key=lambda b: (b['group'], b['block']))]


@contextmanager
def block_hooks(source, timer):
    """Only harness hooks; restore both methods on success, failure or cancel."""
    from roshambo3.engine.r3g_cpu import R3gWorkerSelfCache
    original_decode = source.decode_cpu_block
    had_override = 'decode_cpu_block' in vars(source)
    original_override = vars(source).get('decode_cpu_block')
    original_bind = R3gWorkerSelfCache.block

    def decode(descriptor):
        timer.start(descriptor)
        start = time.perf_counter() if timer.measure else None
        result = original_decode(descriptor)
        elapsed = time.perf_counter() - start if timer.measure else None
        timer.decoded(result, elapsed)
        return result

    def bind(cache, cpu, descriptor, *args, **kwargs):
        if cache.source is not source:
            return original_bind(cache, cpu, descriptor, *args, **kwargs)
        start = time.perf_counter() if timer.measure else None
        result = original_bind(cache, cpu, descriptor, *args, **kwargs)
        elapsed = time.perf_counter() - start if timer.measure else None
        timer.bound(cache, descriptor, result, elapsed)
        return result

    source.decode_cpu_block = decode
    R3gWorkerSelfCache.block = bind
    try:
        yield
    finally:
        if had_override:
            source.decode_cpu_block = original_override
        else:
            del source.decode_cpu_block
        R3gWorkerSelfCache.block = original_bind


def snapshot(rows, corpus):
    from roshambo3.format.catalog import Catalog
    from roshambo3.metrics import canonical_ids
    catalog = Catalog(corpus)
    output = []
    for scores, gids in rows:
        refs = catalog.locate(gids)
        pairs = np.asarray([(ref.mol_id, ref.conf_idx) for ref in refs], np.int64).reshape(-1, 2)
        output.append(dict(scores=np.asarray(scores, np.float32).copy(),
                           gids=np.asarray(gids, np.int64).copy(), pairs=pairs,
                           canonical=canonical_ids(pairs[:, 0], pairs[:, 1])))
    return output


def check_exact(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError('query count changed')
    for a, b in zip(actual, expected, strict=True):
        # Raw identities precede canonical keys; dense ranks alone can mask a
        # different molecule occupying the same relative sorted position.
        np.testing.assert_array_equal(a['pairs'], b['pairs'])
        np.testing.assert_array_equal(a['gids'], b['gids'])
        np.testing.assert_array_equal(a['canonical'], b['canonical'])
        np.testing.assert_array_equal(a['scores'].view(np.uint32), b['scores'].view(np.uint32))


def snapshot_record(result):
    return [dict(identity=r['pairs'].tolist(), gids=r['gids'].tolist(),
                 canonical_ids=r['canonical'].tolist(),
                 scores_sha256=hashlib.sha256(r['scores'].tobytes()).hexdigest()) for r in result]


def run_pass(corpus, query, mode, workers, *, k, level, instrument=True, measure=False):
    from roshambo3.engine import CPUWorkerOptions, Engine, R3gSource, ScreenConfig, TopK, MoleculeTopK
    from roshambo3.engine.config import ColorPlacement
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    color = mode == 'combo'
    generator = generator_from_descriptor(corpus.color) if color else None
    source = R3gSource(corpus, tile=False, color=color, color_generator=generator)
    config = ScreenConfig(color=color, mixing=.5 if color else 0.,
        color_placement=ColorPlacement.IN_GRADIENT if color else ColorPlacement.AT_POSE,
        cpu_worker=CPUWorkerOptions(self_overlap_policy='prepared', workers=workers))
    engine = Engine(backend='cpu')
    timer = BlockTimer(measure=measure) if instrument else None
    sink = MoleculeTopK(k) if level == 'molecule' else TopK(k)
    try:
        if timer is None:
            stats, rows = engine.run(source, query, config, sink)
        else:
            with block_hooks(source, timer):
                stats, rows = engine.run(source, query, config, sink, progress=timer.end)
        scan = stats.cpu_scan
        blocks = stats.n_blocks
        must_equal = dict(stored_vaa_verified_blocks=blocks,
            library_shape_self_evaluations=0, library_color_self_evaluations=0,
            library_color_self_binding_evaluations=0, color_self_binding_recompute_blocks=0,
            color_self_sidecar_blocks=blocks if color else 0, native_self_overlap_fallback_calls=0,
            prepared_self_retained_bytes=0, self_cache_hits=0)
        for name, expected in must_equal.items():
            if scan[name] != expected:
                raise AssertionError(f'{name}: {scan[name]!r} != {expected!r}')
        if scan['self_overlap_route'] != 'verified_stored':
            raise AssertionError('scan route is not verified_stored')
        result = snapshot(rows, corpus)
        record = dict(rows=stats.n_conformers, blocks=blocks, n_queries=query.n_conformers,
            routes=must_equal, result=snapshot_record(result),
            policy={name: scan[name] for name in ('shape_backend', 'shape_grain', 'shape_width',
                'shape_field_storage_stamps', 'field_spacing_by_group', 'field_width_by_group',
                'canonical_color_kernel', 'factor_canonical_torque', 'optimizer_exp_mode',
                'final_start_policy')},
            worker_blocks=[w['blocks'] for w in scan['worker_counters']],
            workers_with_claims=sum(w['blocks'] > 0 for w in scan['worker_counters']),
            max_live_decoded_blocks=scan['max_live_decoded_blocks'])
        if timer is not None:
            record['block_records'] = timer.finish(source, scan, color=color)
        if measure:
            record.update(scan_wall_s=scan['scan_wall_s'],
                full_setup_scan_s=scan['full_setup_scan_s'],
                descriptor_setup_s=scan['descriptor_setup_s'], query_context_setup_s=scan['query_context_setup_s'],
                engine_stage_s={name: scan[name] for name in ('decode_s', 'conversion_s',
                    'self_preparation_s', 'self_binding_s', 'scoring_s',
                    'native_decode_s', 'native_color_derivation_s', 'native_self_binding_s')})
            record['worker_counters'] = scan['worker_counters']
            record['totals'] = timing_totals(record)
        return result, record
    finally:
        engine.close()
        source.close()


def timing_totals(record):
    rows = record['rows']
    stages = {name: math.fsum(block[name] for block in record['block_records']) for name in BLOCK_STAGES}
    return dict(worker_stage_s=stages,
        worker_stage_us_per_row={name: value / rows * 1e6 for name, value in stages.items()},
        scan_conformers_per_s=rows / record['scan_wall_s'],
        scan_pairs_per_s=rows * record['n_queries'] / record['scan_wall_s'])


def summarize(passes):
    output = []
    cells = sorted({(r['workers'], r['mode'], r['block_rows']) for r in passes})
    for workers, mode, block_rows in cells:
        rows = [r for r in passes if (r['workers'], r['mode'], r['block_rows']) == (workers, mode, block_rows)]
        def spread(values):
            return dict(median=statistics.median(values), min=min(values), max=max(values), values=values)
        output.append(dict(workers=workers, mode=mode, block_rows=block_rows,
            repeats=len(rows), rows=rows[0]['rows'], blocks=rows[0]['blocks'],
            scan_conformers_per_s=spread([r['totals']['scan_conformers_per_s'] for r in rows]),
            stage_us_per_row={name: spread([r['totals']['worker_stage_us_per_row'][name] for r in rows])
                              for name in BLOCK_STAGES}))
    return output


def affinity_for_workers(workers, available):
    from bench.cpu import physical_core_cpus
    physical = [cpu for cpu in physical_core_cpus() if cpu in available]
    siblings = sorted(set(available) - set(physical))
    return (physical + siblings)[:min(workers, len(available))]


def describe_arms(corpora, paths):
    """Qualify distinct physical work lists, not just caller-supplied labels."""
    output, signatures = {}, {}
    for nominal, corpus in corpora.items():
        if not 1 <= nominal <= 4096:
            raise ValueError('nominal block-row targets must be in [1, 4096]')
        directory = corpus.block_directory('file')
        rows = directory['rows']
        if not len(rows) or int(rows.min()) < 1 or int(rows.max()) > nominal:
            raise ValueError(f'physical block rows exceed declared target {nominal}')
        signature = hashlib.sha256(directory.tobytes()).hexdigest()
        if signature in signatures:
            raise ValueError(f'identical physical block directories for nominal targets '
                             f'{signatures[signature]} and {nominal}; check corpus paths')
        signatures[signature] = nominal
        output[nominal] = dict(path=str(paths[nominal]), rows=corpus.n_conf,
            fingerprint=corpus.fingerprint, prep_commit=corpus.descriptor['prep']['commit'],
            physical_blocks=len(rows), min_block_rows=int(rows.min()),
            max_block_rows=int(rows.max()), directory_sha256=signature)
    return output


def _write_line(file, value):
    file.write(json.dumps(value, default=str) + '\n')
    file.flush()


def _no_foreign_load():
    from bench.cpu import query_cpu_state
    state = query_cpu_state()
    # In a user-authorized quiet window, load snapshots are diagnostic only. Preserve all foreign-process evidence;
    # CpuRun still records and grades the busy signal without modification.
    return state.as_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--qualify', action='store_true')
    action.add_argument('--measure', action='store_true')
    parser.add_argument('--corpus', action='append', required=True, metavar='ROWS=PATH')
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--workers', default='8,28,56')
    parser.add_argument('--modes', default='shape,combo')
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--level', choices=('conformer', 'molecule'), default='conformer')
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    repeats = args.repeats if args.repeats is not None else (1 if args.qualify else 11)
    warmup = args.warmup if args.warmup is not None else (0 if args.qualify else 3)
    workers = [int(x) for x in args.workers.split(',')]
    modes = args.modes.split(',')
    if repeats < 1 or warmup < 0 or args.k < 1 or any(n < 1 for n in workers):
        parser.error('repeats, workers and k must be positive; warmup must be nonnegative')
    if not modes or any(mode not in ('shape', 'combo') for mode in modes):
        parser.error('modes must be shape and/or combo')
    paths = {int(s.split('=', 1)[0]): Path(s.split('=', 1)[1]) for s in args.corpus}
    if len(paths) != len(args.corpus):
        parser.error('each block-size corpus may be specified only once')
    from bench.cpu import cpu_identity
    from bench.cpu_run import CpuRun
    from bench.gitinfo import get_git_info
    from bench.numerics_backends import verified_build
    from roshambo3.api import load_queries
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    from roshambo3.format.r3g.reader import open_corpus
    import roshambo3
    if get_git_info().source_dirty:
        raise RuntimeError('qualification and measurement require committed clean source')
    build = verified_build('cpu')
    corpora = {n: open_corpus(str(path)) for n, path in paths.items()}
    sizes = sorted(corpora)
    baseline = 4096 if 4096 in corpora else sizes[-1]
    reference = corpora[baseline]
    layout = [(g['L'], g['L_field_width'], g['n_conf']) for g in reference.group_table]
    for n, corpus in corpora.items():
        if [(g['L'], g['L_field_width'], g['n_conf']) for g in corpus.group_table] != layout:
            raise AssertionError('block-size arms changed group layout')
        if corpus.fingerprint != reference.fingerprint:
            raise AssertionError('block-size arms changed scoring semantics')
        if corpus.descriptor['prep']['commit'] is None:
            raise AssertionError('corpus preparation commit is null')
    arm_metadata = describe_arms(corpora, paths)
    generator = generator_from_descriptor(reference.color)
    queries = {mode: load_queries(args.queries, backend='cpu',
        color_generator=generator if mode == 'combo' else None) for mode in modes}
    outputs = {name: Path(str(args.output_prefix) + suffix) for name, suffix in
               (('passes', '-passes.jsonl'), ('runs', '-runs.jsonl'), ('summary', '-summary.json'))}
    if any(path.exists() for path in outputs.values()):
        raise RuntimeError('output already exists; choose a new output prefix')
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    identity = cpu_identity().as_dict()
    original_affinity = set(os.sched_getaffinity(0))
    all_passes = []
    header = dict(kind='header', action='qualification' if args.qualify else 'measurement',
        args=vars(args), source=get_git_info().as_dict(), cpu_build=build,
        package_file=roshambo3.__file__, cpu_identity=identity,
        queries_sha256=hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        corpus=arm_metadata,
        timing_contract=dict(native_stages_are_subsets_of_decode=True,
            block_span='decode entry through owning-worker progress after top-K ingestion',
            scorer_binding='R3gWorkerSelfCache.block wall interval',
            qualification_publishes_no_timings=True))
    try:
        with outputs['passes'].open('x') as file:
            _write_line(file, header)
            expected = {}
            # Independent uninstrumented controls catch hook-induced behavior
            # changes; all arms/counts/repeats then compare the same raw result.
            for mode in modes:
                expected[mode], control = run_pass(reference, queries[mode], mode, workers[0],
                    k=args.k, level=args.level, instrument=False)
                _write_line(file, dict(kind='control', mode=mode, block_rows=baseline, **control))
            for count in workers:
                cpus = affinity_for_workers(count, original_affinity)
                oversubscription = dict(physical=count > identity['physical_cores'],
                    logical=count > identity['logical_cpus'], affinity_cpus=len(cpus))
                for mode in modes:
                    def execute():
                        for rep in range(-warmup, repeats):
                            rotation = (rep + warmup) % len(sizes)
                            order = sizes[rotation:] + sizes[:rotation]
                            for n in order:
                                state = _no_foreign_load() if args.measure else None
                                result, record = run_pass(corpora[n], queries[mode], mode, count,
                                    k=args.k, level=args.level, measure=args.measure and rep >= 0)
                                check_exact(result, expected[mode])
                                if rep < 0:
                                    print(f'WARMUP {mode} workers={count} block_rows={n} rep={rep}', flush=True)
                                    continue
                                record.update(kind='pass', mode=mode, workers=count, block_rows=n,
                                    repeat=rep, affinity=cpus, oversubscription=oversubscription,
                                    cpu_state_before=state, identity_and_score_bits_equal=True)
                                _write_line(file, record)
                                all_passes.append(record)
                                print(f'PASS {mode} workers={count} block_rows={n} rep={rep}', flush=True)
                    if args.qualify:
                        os.sched_setaffinity(0, cpus)
                        execute()
                    else:
                        with CpuRun('format3-cpu-block-timing', decision_grade=True, cpus=cpus, threads=count,
                                    measurement_regime='calibration') as run:
                            if run.lock_busy_reason:
                                raise RuntimeError('CPU lock unavailable: ' + run.lock_busy_reason)
                            _no_foreign_load()
                            execute()
                            selected = [p for p in all_passes if p['workers'] == count and p['mode'] == mode]
                            run.timing = dict(n_repeats=repeats, n_warmup=warmup,
                                sustained_s=sum(p['scan_wall_s'] for p in selected),
                                summary=summarize(selected))
                            run.config = dict(workers=count, max_results=args.k, level=args.level,
                                block_rows=sizes, route='public_cpu_worker_prepared',
                                mode=mode, oversubscription=oversubscription)
                            run.dataset = dict(name='PUR2 format-3 matched block variants',
                                paths={n: str(p) for n, p in paths.items()}, n_conformers=reference.n_conf)
                            run.query_panel = dict(name=str(args.queries),
                                sha256=header['queries_sha256'], ids=list(queries[mode].names))
                            run.audits = dict(identity_and_score_bits_equal=True,
                                             route='verified_stored', native_timer_totals_qualified=True)
                            run.notes = ('Informational block-size throughput; native intervals are subsets '
                                         'of decode. Oversubscription is explicit, no clock is requested.')
                        run.write(outputs['runs'])
                        _write_line(file, dict(kind='grade', workers=count, mode=mode, status=run.status))
            summary = dict(kind='qualification' if args.qualify else 'measurement',
                source=get_git_info().as_dict(), identity_and_score_bits_equal=True,
                cells=len(all_passes), passes_file=str(outputs['passes']),
                summary=summarize(all_passes) if args.measure else [],
                status='complete', informational_throughput=True)
            outputs['summary'].write_text(json.dumps(summary, indent=2, default=str) + '\n')
            _write_line(file, dict(kind='footer', status='complete', passes=len(all_passes)))
    finally:
        os.sched_setaffinity(0, original_affinity)


if __name__ == '__main__':
    main()

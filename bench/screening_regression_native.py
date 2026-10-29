"""Source-verified native adapter and bounded runtime/fault lane for screening regression.

The panel/metric/cache contract lives in screening_regression. This adapter
executes production scorers, not a simulator. It supplies no stored self terms:
changed geometry/features therefore recompute their own denominators.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
EVAL_TARGETS = ('ADA', 'AMPC', 'MAPK2', 'ANDR', 'PARP1', 'RENI', 'EGFR', 'HS90A')


def select_checkout(native_paths):
    sys.meta_path[:] = [finder for finder in sys.meta_path if 'ScikitBuild' not in type(finder).__name__]
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
    import roshambo3
    if Path(roshambo3.__file__).resolve() != ROOT / 'src/roshambo3/__init__.py':
        raise RuntimeError('Python imported from another checkout')
    paths = [str(Path(path).resolve()) for path in native_paths]
    for name in ('cpu', 'cuda'):
        loaded = sys.modules.get('roshambo3._roshambo3_' + name)
        if loaded is not None and str(Path(loaded.__file__).resolve().parent) not in paths:
            raise RuntimeError('native extension imported before build selection')
    roshambo3.__path__[:] = paths + [p for p in roshambo3.__path__ if p not in paths]


def score_panel(meta, data, *, backend='cpu', lane='shape', fault=None):
    """Return native scores and source/config provenance for one 128-row panel.

    Caller owns GPU arbitration before entering this function. Shape CPU is the
    default analytic backend; combo is the explicit CUDA lane. Fault settings
    are separately recorded from the baseline configuration used for comparison.
    """
    import numpy as np
    from bench.numerics_backends import verified_build
    from bench.screening_regression import config_for
    from roshambo3.numerics import WIRE_QUANT_STEP
    from roshambo3.optimizer import lr_t_for_steps
    from roshambo3.backend import cpu, cuda
    from roshambo3.backend.spec import PoseSearch, ColorPlacement, CoordResidency

    if lane not in ('shape', 'combo') or backend not in ('cpu', 'cuda'):
        raise ValueError('unsupported lane/backend')
    if backend == 'cpu' and lane != 'shape':
        raise ValueError('CPU default covers shape; combo requires explicit CUDA lane')
    if fault not in (None, 'scale', 'zero_steps', 'omit_features'):
        raise ValueError('unknown native fault')
    if fault == 'omit_features' and lane != 'combo':
        raise ValueError('feature omission is a combo-only plant')
    module = cpu if backend == 'cpu' else cuda
    if not module.available():
        raise RuntimeError(backend + ' native backend unavailable')
    stamp = verified_build(backend)
    query = np.array(data['query_coords'], np.float32, copy=True)
    qtypes = np.array(data['query_types'], np.int32, copy=True)
    qn = np.array(data['query_n_atoms'], np.int32, copy=True)
    library = np.array(data['coords_i16'], np.int16, copy=True)
    counts = np.array(data['counts'], np.uint8, copy=True)
    color = lane == 'combo'
    if color and meta.get('library_color_offset') != -1:
        raise ValueError('native adapter requires the recorded merged color layout')
    if np.any(counts.astype(np.int64).sum(axis=1) > library.shape[1]):
        raise ValueError('counts exceed compact panel record width')
    r = np.asarray(data['interaction_r'], np.float32) if color else np.ones((1, 1), np.float32)
    p = np.asarray(data['interaction_p'], np.float32) if color else np.zeros((1, 1), np.float32)
    if not color:
        counts = counts[:, :1].copy()
        for qi, nr in enumerate(qn):
            query[qi, int(nr):] = 0
        qtypes[:] = 0
    if fault == 'scale':
        # Exact 1.5x scale in physical coordinates rounded back to the same wire
        # grid. Both shape and features move; native self terms are recomputed.
        scaled = np.rint(library.astype(np.float64) * 1.5)
        if np.max(np.abs(scaled)) > np.iinfo(np.int16).max:
            raise ValueError('scale plant would overflow int16')
        library = scaled.astype(np.int16)
    if fault == 'omit_features' and backend != 'cuda':
        counts[:, 1:] = 0
        qtypes[:] = 0
        for qi, nr in enumerate(qn):
            query[qi, int(nr):] = 0
    n = len(library)
    inputs = dict(query_coords=query, query_types=qtypes, query_n_atoms=qn,
                  library_coords=library, library_counts=counts,
                  pair_query_row=np.zeros(n, np.int32),
                  pair_library_row=np.arange(n, dtype=np.int32),
                  interaction_r=r, interaction_p=p)
    steps = 0 if fault == 'zero_steps' else 100
    settings = dict(steps=steps, lr_q=.1, lr_t=lr_t_for_steps(100), start_mode=1)
    launch = dict(settings)
    if backend == 'cpu':
        if fault == 'zero_steps':
            # The public optimizer-budget resolver refuses steps=0 before its
            # override. Execute the same compiled scorer's supported zero-step
            # entry directly, dequantizing exactly as cpu.score_pairs does.
            native_inputs = {key: inputs[key] for key in (
                'query_coords', 'query_n_atoms', 'library_counts',
                'pair_query_row', 'pair_library_row')}
            native_inputs['library_coords'] = library.astype(np.float32) * np.float32(WIRE_QUANT_STEP)
            rows = cpu.extension().score_pairs(**native_inputs, **settings, threads=1,
                                                final_start_policy='default')
        else:
            rows = cpu.score_pairs(**inputs, **settings, color=False, threads=1)
    else:
        from roshambo3.layout import tile32
        inputs['library_coords'] = tile32(library)
        from roshambo3.engine import ScreenConfig
        config = ScreenConfig(pose_search=PoseSearch.ANALYTIC, steps=100, start_mode=1,
            color=color, mixing=.5 if color else 0.,
            color_placement=ColorPlacement.IN_GRADIENT if color else ColorPlacement.AT_POSE,
            color_stage=0, color_residency=CoordResidency.GLOBAL)
        launch = config.for_width(int(library.shape[1]), query_atoms=int(query.shape[1]),
                                  n_features=int(counts.shape[1]),
                                  library_color_offset=-1 if color else 0)
        launch.update(settings)
        if fault == 'omit_features':
            # Preserve the public no-feature refusal separately; this bounded
            # plant mutates actual compiled inputs after clean validation.
            # No scorer, self term, compiler flag or production file changes.
            extension = cuda.extension()
            original = extension.optimize_pairs
            invocations = [0]
            def omit_at_native_input(*arguments, **keywords):
                actual = list(arguments)
                actual[0] = np.array(actual[0], copy=True)
                actual[1] = np.zeros_like(actual[1])
                actual[4] = np.array(actual[4], copy=True)
                actual[4][:, 1:] = 0
                for qi, nr in enumerate(actual[2]):
                    actual[0][qi, int(nr):] = 0
                invocations[0] += 1
                return original(*actual, **keywords)
            extension.optimize_pairs = omit_at_native_input
            try:
                rows = cuda.score_pairs(**inputs, **launch, check_self_overlap_bound=True)
            finally:
                extension.optimize_pairs = original
            if invocations[0] != 1:
                raise RuntimeError('compiled omitted-feature seam did not execute exactly once')
        else:
            rows = cuda.score_pairs(**inputs, **launch, check_self_overlap_bound=True)
    rows = np.asarray(rows)
    if rows.shape != (n, 20) or not np.isfinite(rows).all():
        raise ValueError('native scorer returned incomplete/nonfinite rows')
    scores = rows[:, 0 if color else 1].astype(np.float32)
    extension = module.extension()
    path = Path(extension.__file__).resolve()
    provenance = dict(backend=backend, lane=lane, config=config_for(lane, backend),
        candidate_source_sha256=stamp[backend + '_src_sha256'],
        extension_path=str(path), extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        harness_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        build=stamp, wire_step=float(WIRE_QUANT_STEP), native_fault=fault,
        fault_seam=('compiled optimize_pairs actual query/types/counts after clean wrapper validation'
                    if fault == 'omit_features' and backend == 'cuda' else
                    'compiled CPU score_pairs zero-step entry' if fault == 'zero_steps' and backend == 'cpu'
                    else 'production scorer input/settings'),
        actual_settings=settings, resolved_launch=launch,
        query_storage_width=int(query.shape[1]), library_storage_width=int(library.shape[1]),
        query_real_atoms=qn.tolist(), library_real_atom_range=[int(counts[:,0].min()), int(counts[:,0].max())],
        self_overlap_policy='native recompute from actual changed inputs',
        scope='production native scorer on bounded prepared panel; public pipeline invariants checked separately')
    return scores, provenance


@contextmanager
def arbitration(backend, output, *, correctness_only=False):
    """Own timing arbitration and machine records before any CUDA context."""
    if backend == 'cpu':
        from bench.cpu import physical_core_cpus
        from bench.cpu_run import CpuRun
        allowed = set(os.sched_getaffinity(0))
        chosen = next((c for c in physical_core_cpus() if c in allowed), None)
        if chosen is None:
            raise RuntimeError('no allowed physical CPU for routine timing')
        with CpuRun('screening-regression', decision_grade=True, cpus=[chosen], threads=1,
                    measurement_regime='headline') as run:
            if run.lock_busy_reason is not None:
                raise RuntimeError('CPU timing lock unavailable: ' + run.lock_busy_reason)
            yield run
        run.write(output / 'cpu-runtime.jsonl')
    else:
        from bench.gpu_lock import GpuLock, resolve_device, query_gpu_state
        from bench.clocks import ClockPowerSampler
        device = resolve_device(0)
        with GpuLock('screening-regression', device) as lock:
            state = query_gpu_state(device)
            preflight = dict(device=device.as_dict(), lock_path=str(lock.lock_path),
                             preflight=state.as_dict(), gpu_lock_held=True,
                             correctness_only=correctness_only,
                             timing_advisory=bool(correctness_only or state.compute_apps or state.utilization_gpu_pct > 0))
            (output / 'gpu-preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
            if (state.compute_apps or state.utilization_gpu_pct > 0) and not correctness_only:
                raise RuntimeError('GPU busy before bounded screening')
            with ClockPowerSampler(device=device) as sampler:
                receipt = preflight
                yield receipt
            receipt['clocks'] = sampler.summary()
            receipt['after'] = query_gpu_state(device).as_dict()
            (output / 'gpu-runtime.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-build', action='append', type=Path, required=True)
    parser.add_argument('--panel-dir', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--lane', choices=('shape', 'combo'), default='shape')
    parser.add_argument('--action', choices=('qualify', 'routine', 'sensitivity'), required=True)
    parser.add_argument('--targets', default=','.join(EVAL_TARGETS))
    parser.add_argument('--scenario', action='append', choices=('clean','stable_reorder','scale','zero_steps','omit_rows','bias','identities','omit_features'))
    parser.add_argument('--_inner', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--correctness-only', action='store_true',
                        help='allow recorded GPU contention; all runtime evidence stays advisory')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args._inner:
        import subprocess
        begin = time.perf_counter()
        invocation = [sys.executable, str(Path(__file__).resolve()),
                      *(sys.argv[1:] if argv is None else argv), '--_inner']
        completed = subprocess.run(invocation, check=False)
        elapsed = time.perf_counter() - begin
        envelope = dict(fresh_process_wall_s=elapsed, child_exitcode=completed.returncode,
                        action=args.action, includes='fresh interpreter/imports/setup/screen/report/teardown',
                        routine_total_budget_s=300,
                        total_budget_pass=elapsed <= 300 if args.action == 'routine' else None)
        (args.output / 'process-envelope.json').write_text(json.dumps(envelope, indent=2) + '\n')
        return completed.returncode if completed.returncode else int(args.action == 'routine' and elapsed > 300)
    select_checkout(args.native_build)
    from bench import screening_regression as core
    from roshambo3.metrics import canonical_ids
    import numpy as np
    targets = tuple(args.targets.split(','))
    scenarios = ('clean',) if args.action != 'sensitivity' else (
        'clean', 'stable_reorder', 'scale', 'zero_steps', 'omit_rows', 'bias', 'identities',
        *(['omit_features'] if args.lane == 'combo' else []))
    required_scenarios = scenarios
    if args.scenario:
        if args.action != 'sensitivity': raise ValueError('--scenario requires sensitivity action')
        scenarios = tuple(args.scenario)
    elapsed = dict(setup_s=0., screening_s=0., reporting_s=0.)
    reports = {}
    loaded = {}
    with arbitration(args.backend, args.output, correctness_only=args.correctness_only) as timing:
        begun = time.perf_counter()
        for target in targets:
            loaded[target] = core.load_panel(args.panel_dir, target)
        elapsed['setup_s'] = time.perf_counter() - STARTED
        clean = {}
        for scenario in scenarios:
            directory = args.output / scenario
            directory.mkdir(parents=True, exist_ok=True)
            refusals = []
            for target, (meta, data) in loaded.items():
                before = time.perf_counter()
                if scenario in ('stable_reorder', 'omit_rows', 'bias', 'identities'):
                    values, provenance = clean[target]
                    values, provenance = values.copy(), dict(provenance)
                else:
                    try:
                        values, provenance = score_panel(meta, data, backend=args.backend,
                            lane=args.lane, fault=None if scenario == 'clean' else scenario)
                    except ValueError as error:
                        if scenario != 'omit_features' or 'library_counts records no pharmacophore feature' not in str(error):
                            raise
                        elapsed['screening_s'] += time.perf_counter() - before
                        refusals.append(dict(target=target, detected=True,
                            route='production CUDA wrapper feature-completeness invariant',
                            reason=str(error), kernel_executed=False,
                            selection_metrics_status='unavailable_input_refused'))
                        continue
                    if scenario == 'clean': clean[target] = (values.copy(), dict(provenance))
                elapsed['screening_s'] += time.perf_counter() - before
                ids = canonical_ids(data['mol_id'], data['conf_idx'])
                if scenario == 'stable_reorder':
                    ids, values = ids[::-1], values[::-1]
                elif scenario == 'omit_rows':
                    keep = np.arange(len(ids)) % 7 != 6
                    ids, values = ids[keep], values[keep]
                elif scenario == 'bias': values = values + np.float32(.04)
                elif scenario == 'identities': ids = np.roll(ids, 1)
                before = time.perf_counter()
                stem = target + '.' + args.lane + '.' + args.backend
                npz = directory / (stem + '.npz')
                if npz.exists(): raise FileExistsError('candidate output already exists: ' + str(npz))
                np.savez_compressed(npz, ids=ids, scores=values)
                record = dict(provenance, protocol=core.PROTOCOL, role='candidate', target=target,
                    panel_sha256=meta['npz_sha256'], geometry_sha256=meta['geometry_sha256'],
                    npz_sha256=core.file_sha(npz), scenario=scenario,
                    plant_kind=('native changed input/settings' if scenario in ('scale','zero_steps','omit_features')
                                else 'output interface' if scenario != 'clean' else 'clean fresh native control'))
                mp = directory / (stem + '.json')
                mp.write_text(json.dumps(record, indent=2) + '\n')
                if args.action == 'qualify':
                    for role in ('fixed', 'current'):
                        core.import_cache(target, lane=args.lane, backend=args.backend, role=role,
                            source_npz=npz, source_json=mp, panel_dir=args.panel_dir,
                            output_dir=args.cache_root / role)
                elapsed['reporting_s'] += time.perf_counter() - before
            before = time.perf_counter()
            if args.action != 'qualify':
                if refusals:
                    report = dict(protocol=core.PROTOCOL, backend=args.backend, lane=args.lane,
                        scenario=scenario, status='complete' if len(refusals) == len(targets) else 'partial',
                        detector_pass=False, per_query=refusals,
                        detection_route='production input invariant; not native selection sensitivity',
                        selection_metrics_status='unavailable_input_refused')
                else:
                    report = core.evaluate(lane=args.lane, backend=args.backend, panel_dir=args.panel_dir,
                        fixed_dir=args.cache_root / 'fixed', current_dir=args.cache_root / 'current',
                        candidate_dir=directory, targets=targets)
                reports[scenario] = report
                (directory / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            elapsed['reporting_s'] += time.perf_counter() - before
            print(args.action, args.backend, args.lane, scenario, 'completed', flush=True)
        if args.backend == 'cpu':
            timing.timing = dict(wall_s=time.perf_counter() - begun, n_repeats=1, n_warmup=0)
            timing.config = dict(core.config_for(args.lane, args.backend), max_results=16,
                                 timing_purpose='absolute routine runtime, no speedup claim')
            timing.dataset = dict(name='bounded screening panel', n_conformers=len(targets)*128)
            timing.query_panel = dict(name='frozen held-out screening panel',
                ids=list(targets), sha256=hashlib.sha256(''.join(loaded[t][0]['npz_sha256'] for t in targets).encode()).hexdigest())
    elapsed['child_total_s'] = time.perf_counter() - STARTED
    phase_pass = elapsed['setup_s'] <= 120 and elapsed['screening_s'] <= 120 and elapsed['reporting_s'] <= 60
    cases = {scenario: dict(detector_pass=report['detector_pass'], status=report['status'],
        expected='pass' if scenario in ('clean','stable_reorder') else 'fail',
        expectation_met=(report['detector_pass'] if scenario in ('clean','stable_reorder') else
                         not report['detector_pass']) and report['status'] == 'complete')
        for scenario, report in reports.items()}
    if args.backend == 'cpu':
        # CpuRun's standard rate grade requires 3+11 repeats. An absolute
        # one-invocation budget is a separate measurement, never a headline rate.
        states = timing.record.operating_point
        quiet = not any(states[key].get('busy') or states[key].get('foreign_processes')
                        for key in ('cpu_state_before', 'cpu_state_after'))
        budget_environment = bool(states.get('cpu_lock_held') and states.get('cpu_affinity') and quiet)
        timing_grade = dict(standard_cpu_rate_status=timing.status,
            absolute_budget_environment_qualified=budget_environment,
            basis='budget-only wall invocation; standard rate receipt kept independently',
            advisory_reason=None if budget_environment else 'CPU contention or missing lock/pinning')
    else:
        budget_environment = not timing.get('timing_advisory')
        timing_grade = dict(absolute_budget_environment_qualified=budget_environment,
                           basis='canonical GPU lock/preflight and achieved clocks',
                           advisory_reason='correctness-only or contention' if not budget_environment else None)
    result = dict(action=args.action, backend=args.backend, lane=args.lane, targets=list(targets),
        phases=elapsed, timing_grade=timing_grade,
        routine_phase_budget_pass=phase_pass if args.action == 'routine' else None,
        routine_runtime_qualified=(phase_pass and budget_environment and not args.correctness_only)
            if args.action == 'routine' else None,
        created_initial_equal_references=args.action == 'qualify', cases=cases,
        sensitivity_qualified=(all(c['expectation_met'] for c in cases.values())
            and targets == EVAL_TARGETS and set(cases) == set(required_scenarios))
            if args.action == 'sensitivity' else None,
        runtime_note='qualify and sensitivity campaign costs separate from candidate-only routine invocation')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.action == 'qualify': return 0
    okay = all(c['expectation_met'] for c in cases.values())
    return 0 if okay and (phase_pass or args.action != 'routine') else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        import traceback
        core = sys.modules.get('bench.screening_regression')
        unavailable_type = getattr(core, 'Unavailable', ())
        unavailable = ((bool(unavailable_type) and isinstance(error, unavailable_type))
                       or isinstance(error, ImportError)
                       or 'native backend unavailable' in str(error))
        failure = dict(status='unavailable' if unavailable else 'error',
                       detector_pass=False, reason=str(error))
        if '--output' in sys.argv:
            directory = Path(sys.argv[sys.argv.index('--output') + 1])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        traceback.print_exc()
        raise SystemExit(2 if unavailable else 1)


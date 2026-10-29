"""Assemble flat screening-detector receipts without replacing original attempts or grades.

--results names a directory whose evaluation/ subdirectory holds the outputs of
bench/screening_regression_native.py: per-lane sensitivity runs
(cpu-shape-sensitivity, cuda-shape-sensitivity, cuda-combo-sensitivity, plus
cuda-combo-feature-native) with one <scenario>/evaluation.json per scenario,
and per-lane routine runs (cpu-shape-routine, cuda-shape-routine-advisory,
cuda-combo-routine-advisory) with summary.json, process-envelope.json and
clean/evaluation.json, plus cpu-runtime.jsonl (CPU) or gpu-runtime.json (CUDA).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = 'screening-regression-receipt-v1'
TARGETS = {'ADA','AMPC','MAPK2','ANDR','PARP1','RENI','EGFR','HS90A'}
CASES = ('clean','stable_reorder','scale','zero_steps','omit_rows','bias','identities')


def read(path):
    return json.loads(path.read_text())


def case_receipt(root, directory, scenario):
    path = root / directory / scenario / 'evaluation.json'
    data = read(path)
    expected_pass = scenario in ('clean','stable_reorder')
    actual_targets = {r['target'] for r in data['per_query']}
    valid = (data['status'] == 'complete' and actual_targets == TARGETS
             and data['detector_pass'] == expected_pass)
    native = scenario in ('scale','zero_steps','omit_features')
    source = root / directory / scenario
    if native:
        for target in TARGETS:
            record = next(source.glob(target + '.*.json'))
            artifact = read(record)
            if artifact.get('native_fault') != scenario:
                raise ValueError('native fault did not execute: ' + str(record))
            if scenario == 'omit_features' and not artifact.get('fault_seam','').startswith('compiled optimize_pairs'):
                raise ValueError('omitted-feature input guard cannot certify selection sensitivity')
    levels = data.get('comparisons', {}).get('fixed', {}).get('levels', {})
    return dict(artifact=str(path), artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected='pass' if expected_pass else 'fail', detector_pass=data['detector_pass'],
        expectation_met=valid, n_queries=len(actual_targets),
        route='native changed input/settings' if native else 'clean control' if expected_pass else 'output interface',
        selection_metrics_status='available' if levels else data.get('selection_metrics_status','unavailable'),
        fixed_reference_recall=levels,
        signed_bias=data.get('comparisons',{}).get('fixed',{}).get('signed_score_movement',{}))


def artifact_receipt(path):
    data = read(path)
    return dict(artifact=str(path), artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                protocol=data.get('protocol'), status=data.get('status'),
                detector_pass=data.get('detector_pass'))


def runtime(root, name, backend):
    base = root / name
    summary, envelope = read(base / 'summary.json'), read(base / 'process-envelope.json')
    clean = read(base / 'clean/evaluation.json')
    if backend == 'cpu':
        ledger = json.loads((base / 'cpu-runtime.jsonl').read_text().splitlines()[-1])
        grade = dict(standard_rate_status=ledger['status'], refusal_reason=ledger.get('refusal_reason'),
                     cpu_state=ledger['operating_point'])
    else:
        grade = read(base / 'gpu-runtime.json')
    qualified = bool(summary.get('routine_runtime_qualified')
                     and envelope['child_exitcode'] == 0 and envelope['fresh_process_wall_s'] <= 300)
    return dict(artifact=str(base), fresh_process_wall_s=envelope['fresh_process_wall_s'],
        phase_wall_s=summary['phases'], observed_total_budget_pass=envelope['fresh_process_wall_s'] <= 300,
        observed_phase_budget_pass=summary['routine_phase_budget_pass'],
        clean_detector_pass=clean['detector_pass'],
        actual_fixed_equals_current=all(r['fixed_equals_current'] for r in clean['per_query']),
        interpretation=('quiet absolute CPU budget qualified; no headline-rate claim' if qualified else
                        'advisory runtime only; not quiet budget qualification'),
        routine_runtime_qualified=qualified, machine_receipt=grade,
        raw_summary_annotation=('older CPU summary incorrectly labeled clean-only sensitivity qualified and actual fixed/current equality; use actual evidence fields in this receipt'
            if 'fixed_and_current_initially_equal' in summary else None))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='directory whose evaluation/ subdirectory holds the per-lane '
                             'sensitivity and routine outputs of screening_regression_native.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quiet-cpu-routine', type=Path,
                        help='optional: a CPU routine output directory (summary.json, '
                             'process-envelope.json, clean/evaluation.json, cpu-runtime.jsonl) '
                             'from a separate quiet machine, recorded as an extra runtime receipt')
    parser.add_argument('--accepted-change', type=Path,
                        help='optional: accepted-change.json from screening_accepted_change.py, '
                             'recorded by path and hash')
    parser.add_argument('--full-library', type=Path,
                        help='optional: a complete-library qualification receipt, '
                             'recorded by path and hash')
    args = parser.parse_args(argv)
    root = args.results / 'evaluation'
    lanes = {}
    for lane, directory in (('cpu-shape','cpu-shape-sensitivity'),
                            ('cuda-shape','cuda-shape-sensitivity'),
                            ('cuda-combo','cuda-combo-sensitivity')):
        cases = {scenario:case_receipt(root,directory,scenario) for scenario in CASES}
        if lane == 'cuda-combo':
            cases['omit_features'] = case_receipt(root,'cuda-combo-feature-native','omit_features')
        lanes[lane] = dict(cases=cases, all_frozen_expectations_met=all(r['expectation_met'] for r in cases.values()),
                          scope='frozen eight-query detector sensitivity; no universal optimizer equivalence')
    result = dict(protocol=PROTOCOL, lanes=lanes,
        runtime={name:runtime(root,directory,backend) for name,directory,backend in (
            ('cpu-shape','cpu-shape-routine','cpu'),
            ('cuda-shape','cuda-shape-routine-advisory','cuda'),
            ('cuda-combo','cuda-combo-routine-advisory','cuda'))})
    if args.quiet_cpu_routine is not None:
        quiet = args.quiet_cpu_routine
        result['runtime']['cpu-shape-quiet'] = runtime(quiet.parent, quiet.name, 'cpu')
    for key, path in (('accepted_numerical_change', args.accepted_change),
                      ('full_library', args.full_library)):
        if path is not None:
            result[key] = artifact_receipt(path)
    advisory = sorted(name for name, r in result['runtime'].items() if not r['routine_runtime_qualified'])
    limitations = []
    if advisory:
        limitations.append('Runtime advisory, not budget-qualified: ' + ', '.join(advisory))
    if any(r['actual_fixed_equals_current'] for r in result['runtime'].values()):
        limitations.append('Fixed and current references are equal; no independent truth implied')
    limitations.append('The challenger-enriched subset cannot certify whole-library recall')
    result['limitations'] = limitations
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('assembled sensitivity/runtime receipts:', args.output)
    return 0 if all(v['all_frozen_expectations_met'] for v in lanes.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())

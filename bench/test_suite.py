"""Run the complete suite serially or as four CPU workers plus one GPU lane."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bench.test_pytest import gpu_selectors
from bench.numerics_pytest import report_from_payloads

ROOT = Path(__file__).resolve().parents[1]


def plan(root=ROOT):
    files = sorted((root/'tests').rglob('test_*.py'))
    lanes = {'cpu': [], 'gpu': []}
    for path in files:
        selectors = gpu_selectors(path)
        if '*' not in selectors:
            lanes['cpu'].append(str(path.relative_to(root)))
        if selectors:
            lanes['gpu'].append(str(path.relative_to(root)))
    return lanes


def combine_outcomes(payloads):
    tests, durations, deselected, collection = {}, {}, set(), {}
    for payload in payloads:
        duplicate = tests.keys() & payload['tests'].keys()
        if duplicate:
            raise ValueError('suite lanes overlap: '+str(sorted(duplicate)))
        tests.update(payload['tests'])
        deselected.update(payload['deselected'])
        for nodeid, state in payload.get('collection', {}).items():
            if collection.setdefault(nodeid, state) != state:
                raise ValueError('suite lanes disagree on collection: '+nodeid)
        for name, value in payload['durations_by_file'].items():
            durations[name] = durations.get(name, 0)+value
    return dict(tests=dict(sorted(tests.items())), deselected=sorted(deselected),
                collection=dict(sorted(collection.items())), durations_by_file=durations)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path, help='compare with a prior serial/parallel output directory')
    parser.add_argument('--cache-root', type=Path, help='fresh private cache for this run; default OUTPUT/cache')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = (args.cache_root or output/'cache').resolve()
    if cache.exists():
        parser.error('acceptance runs require a fresh cache root')
    environment = dict(os.environ,
        CMAKE_BUILD_PARALLEL_LEVEL='2', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', ROSHAMBO3_TEST_CACHE=str(cache))
    # One merged report, even under CI; lane subprocesses never append summaries.
    environment.pop('GITHUB_STEP_SUMMARY', None)
    lanes = {'serial': ['tests']} if args.serial else plan()
    runs = []; numerical = []; outcomes = []
    started = time.monotonic()
    for lane, files in lanes.items():
        if not files:
            continue
        lane_dir = output/lane
        lane_dir.mkdir()
        command = [sys.executable, '-m', 'pytest', '--no-header', '--durations=0',
                   '--numerics-report-dir', str(lane_dir/'numerics'),
                   '--numerics-outcomes', str(lane_dir/'numerics-outcomes.json'),
                   '--suite-outcomes', str(lane_dir/'outcomes.json'),
                   '--suite-events', str(output/'events.jsonl'),
                   '--suite-cache-root', str(cache), '--junitxml', str(lane_dir/'junit.xml')]
        env = dict(environment)
        if lane == 'cpu':
            command += ['-n', str(args.workers), '--dist', 'loadfile', '--suite-lane', 'cpu']
            # CPU-only declarations cannot create CUDA contexts on CPU workers.
            # Outcome parity exposes any test that actually needs visibility.
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            command += ['-n', '0', '--suite-gpu-lock']
            if lane == 'gpu':
                command += ['--suite-lane', 'gpu']
        command += files
        print(f'{lane}: {len(files)} paths; log {lane_dir/"pytest.log"}', flush=True)
        before = time.monotonic()
        with (lane_dir/'pytest.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        runs.append(dict(lane=lane, command=command, exit_code=result.returncode, wall_seconds=time.monotonic()-before))
        if not (lane_dir/'numerics-outcomes.json').is_file() or not (lane_dir/'outcomes.json').is_file():
            (output/'run.json').write_text(json.dumps(dict(runs=runs, complete=False), indent=2))
            return result.returncode or 1
        numerical.append(json.loads((lane_dir/'numerics-outcomes.json').read_text()))
        outcomes.append(json.loads((lane_dir/'outcomes.json').read_text()))
    combined = combine_outcomes(outcomes)
    coverage = report_from_payloads(numerical)
    coverage.write(output/'numerics')
    (output/'outcomes.json').write_text(json.dumps(combined, indent=2)+'\n')
    top = sorted(combined['durations_by_file'].items(), key=lambda item: item[1], reverse=True)[:20]
    record = dict(runs=runs, complete=True, wall_seconds=time.monotonic()-started,
                  top_20_files=top, plan=lanes, environment={key:environment.get(key) for key in
                  ('ROSHAMBO3_CORPORA','ROSHAMBO3_TEST_SDF','CUDA_MPS_PIPE_DIRECTORY','ROSHAMBO3_GPU_LOCK','CMAKE_BUILD_PARALLEL_LEVEL',
                   'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    failed = any(run['exit_code'] for run in runs)
    if args.compare:
        baseline = json.loads((args.compare/'outcomes.json').read_text())
        differences = {nodeid: dict(before=baseline['tests'].get(nodeid), after=combined['tests'].get(nodeid))
                       for nodeid in baseline['tests'].keys()|combined['tests'].keys()
                       if baseline['tests'].get(nodeid)!=combined['tests'].get(nodeid)}
        record['parity'] = dict(test_differences=differences,
            deselected_equal=baseline['deselected']==combined['deselected'],
            collection_equal=baseline['collection']==combined['collection'],
            numerics_equal=json.loads((args.compare/'numerics/coverage.json').read_text())==coverage.as_dict())
        failed |= bool(differences) or not record['parity']['deselected_equal'] or not record['parity']['numerics_equal'] or not record['parity']['collection_equal']
    (output/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as stream:
            stream.write(coverage.markdown())
    print(json.dumps({key:value for key,value in record.items() if key not in ('plan','runs')}, indent=2), flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

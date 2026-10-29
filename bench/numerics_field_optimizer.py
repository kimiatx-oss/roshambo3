"""Preregistered FIELD native search order-controls and independent basins."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

CALIBRATION = range(271300, 271332)
HELDOUT = range(271400, 271416)
SAME_GEOMETRY_CALIBRATION = range(271600, 271632)
SAME_GEOMETRY_HELDOUT = range(271700, 271716)


def receipt_json(record):
    # The reused statistical helper calls its RMS proxy basin_change.
    # Reports must use the scientifically accurate name, including gate labels.
    return json.dumps(record, indent=2).replace('"basin_change"', '"raw_displacement_rate"') + '\n'


def assert_same_multisets(original, reordered):
    """Reject a control that changes geometry, weights, or typed features."""
    for key in original:
        if key not in ('query_coords', 'library_coords', 'query_types'):
            np.testing.assert_array_equal(original[key], reordered[key])

    def ordered(x):
        return x[np.lexsort(tuple(x[:, j] for j in range(x.shape[1])))]

    for prefix in ('query', 'library'):
        a = original[prefix + '_coords']
        b = reordered[prefix + '_coords']
        for i in range(len(a)):
            if prefix == 'query':
                shape_end = int(original['query_n_atoms'][i])
                # Shape occupancy is implicit in the prefix even when both
                # shape and padding have type zero. It must remain separate.
                if 'query_types' in original:
                    aa = np.column_stack((a[i], original['query_types'][i]))
                    bb = np.column_stack((b[i], reordered['query_types'][i]))
                else:
                    aa, bb = a[i], b[i]
                for segment in (slice(0, shape_end), slice(shape_end, None)):
                    np.testing.assert_array_equal(ordered(aa[segment]), ordered(bb[segment]))
            else:
                # Library feature types are implicit in contiguous segments,
                # not an explicit per-atom sidecar. Check each type separately.
                start = 0
                for count in original['library_counts'][i]:
                    end = start + int(count)
                    np.testing.assert_array_equal(ordered(a[i, start:end]), ordered(b[i, start:end]))
                    start = end
                np.testing.assert_array_equal(ordered(a[i, start:]), ordered(b[i, start:]))



def native_field(inputs, objective, *, lr_q=.1):
    from roshambo3.backend import cuda
    from roshambo3.backend.spec import PoseSearch, ColorPlacement
    from bench.numerics_cuda_field_winners import gpu_inputs
    from bench.numerics_optimizer_real import SETTINGS
    ids = np.full(len(inputs['pair_query_row']), -1, np.int32)
    sampled = np.full((len(ids), 4), np.nan, np.float32)
    opts = dict(pose_search=PoseSearch.FIELD, color=objective == 'combo', mixing=.5,
                color_placement=ColorPlacement.IN_GRADIENT)
    if objective == 'combo':
        opts.update(library_color_offset=-1, start_groups_per_row=4,
                    color_residency=0, color_stage=48)
    rows = cuda.score_pairs(**gpu_inputs(inputs), **{**SETTINGS, 'lr_q': lr_q},
                           **opts, winning_starts=ids, start_objectives=sampled)
    if not np.isfinite(sampled).all():
        raise AssertionError('nonfinite sampled search objective')
    np.testing.assert_array_equal(ids, np.argmax(sampled, axis=1))
    return rows, ids, sampled


def snapshot(args):
    import roshambo3
    from bench.run_numerics_checks import select_native_build
    select_native_build(roshambo3, args.build)
    from bench.gpu_lock import GpuLock, resolve_device, query_gpu_state
    device = resolve_device(0)
    with GpuLock('FIELD optimizer order controls', device) as lock:
        state = query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct > 0:
            raise RuntimeError('GPU busy before FIELD optimizer: ' + str(state))
        return regression_snapshot(args, lock_held=lock.acquired, preflight=state.as_dict())


def regression_snapshot(args, *, lock_held=None, preflight=None):
    """Run unchanged finite controls; caller owns GPU serialization before context.

    The default pytest runner already holds the physical GPU lock. Direct CLI
    execution obtains its lock in snapshot(). None means no lock claim here.
    """
    from bench.numerics_backends import verified_build
    from roshambo3.backend import cpu, cuda
    stamps = {k: verified_build(k) for k in ('cpu', 'cuda')}
    from bench.numerics_optimizer_real import panel, pack, stats, scored, freeze, exceeds, SETTINGS
    cases, oracle = panel()
    train = np.array([c['transform'] % 2 == 0 for c in cases])
    test = ~train
    calibration_seeds, heldout_seeds = CALIBRATION, HELDOUT
    protocol = 'FIELD_OPTIMIZER_PROTOCOL.md'
    if args.same_geometry:
        train = test = np.ones(len(cases), bool)
        calibration_seeds = SAME_GEOMETRY_CALIBRATION
        heldout_seeds = SAME_GEOMETRY_HELDOUT
        protocol = 'FIELD_OPTIMIZER_SAME_GEOMETRY_PROTOCOL.md'
    evidence = dict(protocol=protocol, qualified=False,
        heldout_scope=('arithmetic seeds on fixed geometry only' if args.same_geometry else 'odd transforms'),
        scope='finite FIELD order-control schedule canary; full optimizer unqualified',
        settings=SETTINGS, oracle_allowance=oracle, lock_held=lock_held,
        preflight=preflight, native=stamps,
        extensions={name: dict(path=str(Path(backend.extension().__file__).resolve()),
            sha256=hashlib.sha256(Path(backend.extension().__file__).read_bytes()).hexdigest())
            for name, backend in (('cpu', cpu), ('cuda', cuda))},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / 'doc/numerics' / protocol).read_bytes()).hexdigest(),
        raw_displacement_is_not_basin_classification=True, objectives={},
        gaps=['full3D texture arithmetic has no derived FIELD-vs-analytic optimizer allowance',
              'finite atom-order model does not cover arbitrary sampler arithmetic'])
    arrays = {}
    for objective in ('shape', 'combo'):
        inputs = pack(cases, objective)
        anchor, astarts, sampled = native_field(inputs, objective)
        anchor_scores = scored(cases, anchor, objective)
        arrays[objective + '_field_anchor'] = anchor
        arrays[objective + '_field_objectives'] = sampled
        def compare(rows, starts, mask):
            result = stats(cases, anchor, rows, objective, mask, astarts, starts,
                           anchor_scored=anchor_scores)
            return result
        calibration = []
        moved = []
        for seed in calibration_seeds:
            reordered = pack(cases, objective, seed)
            assert_same_multisets(inputs, reordered)
            rows, starts, _ = native_field(reordered, objective)
            calibration.append(dict(seed=seed, **compare(rows, starts, train)))
            moved.append(int(np.sum(np.any(rows[:, 9:20] != anchor[:, 9:20], axis=1))))
        limits = freeze(calibration, oracle)
        # This receipt exists before heldout/clean/plant observations are made.
        frozen = dict(objective=objective, limits=limits, calibration=calibration,
                      calibration_moved_pose_counts=moved,
                      protocol_sha256=evidence['protocol_sha256'])
        freeze_path = args.out.with_name(args.out.stem + '-' + objective + '-frozen.json')
        freeze_path.write_text(receipt_json(frozen))
        heldout = []
        for seed in heldout_seeds:
            reordered = pack(cases, objective, seed)
            assert_same_multisets(inputs, reordered)
            rows, starts, _ = native_field(reordered, objective)
            heldout.append(dict(seed=seed, **compare(rows, starts, test)))
        arms = {}
        for name, lr_q in (('clean', .1), ('plant', .11)):
            rows, starts, sampled = native_field(inputs, objective, lr_q=lr_q)
            arrays[objective + '_' + name] = rows
            arrays[objective + '_' + name + '_starts'] = starts
            arrays[objective + '_' + name + '_objectives'] = sampled
            record = compare(rows, starts, test)
            arms[name] = dict(metrics=record, exceeding=exceeds(record, limits))
        cpu_opts = dict(grain=-1, poly=False) if objective == 'shape' else dict(
            color=True, color_mode='combo_optimal', mixing=.5, color_grain=0)
        analytic, census = cpu.score_pairs(**inputs, **SETTINGS, **cpu_opts, with_census=True)
        arrays[objective + '_analytic_anchor'] = analytic
        analytic_stats = stats(cases, analytic, anchor, objective, test,
                               census['winning_starts'], astarts)
        heldout_valid = not any(exceeds(r, limits) for r in heldout)
        evidence['objectives'][objective] = dict(limits=limits,
            frozen_receipt_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            calibration=calibration, calibration_moved_pose_counts=moved,
            heldout=heldout, heldout_valid=heldout_valid, **arms,
            finite_schedule_canary_pass=bool(heldout_valid and not arms['clean']['exceeding']
                and arms['plant']['exceeding'] and any(moved)),
            analytic_anchor_comparison=analytic_stats,
            analytic_anchor_qualified=False, sampled_objective_argmax_verified=True)
        print(objective + ' native FIELD observations complete', flush=True)
    arrays['provenance'] = np.asarray(json.dumps(evidence))
    np.savez(args.scratch, **arrays)
    args.out.write_text(receipt_json(evidence))
    return evidence


def require_fresh_paths(scratch, output):
    if scratch.suffix != '.npz':
        raise ValueError('scratch must end in .npz')
    paths = [scratch, output, *[output.with_name(output.stem + '-' + objective + '-frozen.json')
                              for objective in ('shape', 'combo')]]
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise ValueError('refusing to overwrite experiment artifacts: ' + ', '.join(occupied))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--same-geometry', action='store_true')
    args = parser.parse_args()
    try:
        require_fresh_paths(args.scratch, args.out)
    except ValueError as error:
        parser.error(str(error))
    if args.worker:
        snapshot(args)
        return 0
    subprocess.run([sys.executable, '-S', '-m', 'bench.numerics_field_optimizer',
        '--build', str(args.build), '--scratch', str(args.scratch), '--out', str(args.out),
        '--worker', *(['--same-geometry'] if args.same_geometry else [])], check=True)
    # Native child has exited: no CUDA context or GPU lock during CPU polishing.
    from bench.numerics_optimizer_real import panel
    from bench.numerics_optimizer_basins import compare_basins
    cases, _ = panel()
    mask = np.array([c['transform'] % 2 == 1 for c in cases])
    evidence = json.loads(args.out.read_text())
    with np.load(args.scratch) as rows:
        for objective in ('shape', 'combo'):
            evidence['objectives'][objective]['analytic_endpoint_diagnostics'] = {
                name: compare_basins(cases, rows[objective + '_analytic_anchor'],
                    rows[objective + '_' + name], objective, mask=mask)
                for name in ('clean', 'plant')}
            print(objective + ' independent endpoint diagnostics complete', flush=True)
    evidence['polish_sha256'] = hashlib.sha256(
        Path(__file__).with_name('numerics_optimizer_basins.py').read_bytes()).hexdigest()
    evidence['snapshot_sha256'] = hashlib.sha256(args.scratch.read_bytes()).hexdigest()
    args.out.write_text(receipt_json(evidence))
    return 0 if all(x['finite_schedule_canary_pass'] for x in evidence['objectives'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())

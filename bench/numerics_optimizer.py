"""Live scalar-anchored optimizer pilot.

Exit 1 means unqualified, including a surviving schedule plant. JSON evidence
contains observations, never expected scores for later comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np

from roshambo3.metrics.floors import SCORE_STRATA, stratum_of
from roshambo3.optimizer import lr_t_for_steps
from roshambo3.reference.overlap import apply_pose, pair_overlap, self_overlap, tanimoto

METRICS = ('mean_loss', 'winning_start_disagreement', 'basin_change')
CALIBRATION_SEEDS = tuple(range(270000, 270016))
HELDOUT_SEEDS = tuple(range(270100, 270108))
BASIN_RESOLUTION = 0.1
KEYS = ('query_coords', 'query_n_atoms', 'library_coords', 'library_counts',
        'pair_query_row', 'pair_library_row')


def load_inputs(path: Path, limit=256):
    """Use input columns only. Expected scores and historical stamp are unread."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in KEYS}
    arrays['query_coords'][..., :3] = np.round(
        arrays['query_coords'][..., :3]  / WIRE_QUANT_STEP) * WIRE_QUANT_STEP
    for key in ('pair_query_row', 'pair_library_row'):
        arrays[key] = arrays[key][:limit]
    if arrays['library_coords'].dtype != np.int16:
        values = np.rint(arrays['library_coords'][..., :3].astype(float)  / WIRE_QUANT_STEP)
        if np.max(np.abs(values)) > np.iinfo(np.int16).max:
            raise ValueError('library outside declared int16 grid')
        arrays['library_coords'] = values.astype(np.int16)
    return arrays


def geometry(arrays, index):
    qi, li = (int(arrays[k][index]) for k in ('pair_query_row', 'pair_library_row'))
    a = arrays['query_coords'][qi, :int(arrays['query_n_atoms'][qi]), :3].astype(float)
    b = arrays['library_coords'][li, :int(arrays['library_counts'][li].item()), :3].astype(float) * float(WIRE_QUANT_STEP)
    return a, b


def rescore(arrays, rows):
    """Independent float64 value at each implementation's own physical pose."""
    values, poses = [], []
    if rows.shape != (len(arrays['pair_query_row']), 20) or not np.isfinite(rows).all():
        raise ValueError('nonfinite or malformed optimizer rows')
    for i, row in enumerate(rows):
        a, b = geometry(arrays, i)
        # CUDA returns optimizer pose in the rotated query frame. CPU folds
        # the start into pose and reports identity in START_QUAT.
        start_inverse = row[16:20].astype(float) * np.array([1, -1, -1, -1])
        posed = apply_pose(apply_pose(b, row[9:13], row[13:16]), start_inverse, np.zeros(3))
        values.append(tanimoto(pair_overlap(a, posed), self_overlap(a), self_overlap(b)))
        poses.append(posed)
    return np.asarray(values), poses


def statistics(arrays, anchor, candidate, mask, *, anchor_starts=None, candidate_starts=None):
    av, ap = rescore(arrays, anchor)
    cv, cp = rescore(arrays, candidate)
    basin = np.asarray([np.sqrt(np.mean(np.sum((a-b)**2, axis=1))) > BASIN_RESOLUTION
                        for a, b in zip(ap, cp)])
    labels = stratum_of(av)
    def summarize(selection):
        n = int(selection.sum())
        if not n:
            return dict(n=0, **{k: None for k in METRICS})
        return dict(n=n, mean_loss=float(np.mean((av-cv)[selection])),
                    winning_start_disagreement=(None if anchor_starts is None or candidate_starts is None
                        else float(np.mean((np.asarray(anchor_starts) != np.asarray(candidate_starts))[selection]))),
                    basin_change=float(np.mean(basin[selection])))
    return dict(overall=summarize(mask), by_stratum={name: summarize(mask & (labels == name))
                for name, _, _ in SCORE_STRATA})


def permuted(arrays, seed):
    """Only atom order changes. Ignore padding; no change to exact geometry."""
    rng = np.random.default_rng(seed)
    out = {key: value.copy() for key, value in arrays.items()}
    for coords, counts in (('query_coords', 'query_n_atoms'), ('library_coords', 'library_counts')):
        for row, n in enumerate(out[counts]):
            n = int(np.asarray(n).item())
            out[coords][row, :n] = out[coords][row, rng.permutation(n)]
    return out


def thresholds(calibration):
    if not calibration or any(row['overall']['n'] == 0 for row in calibration):
        raise ValueError('calibration panel must be nonempty')
    return {key: (None if any(row['overall'][key] is None for row in calibration)
                  else max(0., max(row['overall'][key] for row in calibration))) for key in METRICS}


def exceeding(stats, limits):
    return [key for key in METRICS if limits[key] is not None
            and stats['overall'][key] is not None and stats['overall'][key] > limits[key]]


def run(arrays, *, cuda=False):
    from roshambo3 import backend
    cpu = backend.select('cpu')
    settings = dict(steps=25, start_mode=1, lr_t=lr_t_for_steps(25), lr_q=.1)
    stamp = cpu.build_stamp()
    if stamp.get('cpu_src_sha256') in (None, '', 'unknown'):
        raise ValueError('live scalar source must be stamped')
    anchor = cpu.score_pairs(**arrays, **settings, grain=-1, width=1, poly=False)
    n = len(anchor)
    if n < 4:
        raise ValueError('at least four pair rows required')
    train = np.arange(n) % 2 == 0
    test = ~train
    calibration, heldout = [], []
    # Fix thresholds BEFORE executing held-out, production, or schedule plant.
    for seed in CALIBRATION_SEEDS:
        rows = cpu.score_pairs(**permuted(arrays, seed), **settings, grain=-1, width=1, poly=False)
        calibration.append(dict(seed=seed, **statistics(arrays, anchor, rows, train)))
    limits = thresholds(calibration)
    for seed in HELDOUT_SEEDS:
        rows = cpu.score_pairs(**permuted(arrays, seed), **settings, grain=-1, width=1, poly=False)
        stats = statistics(arrays, anchor, rows, test)
        heldout.append(dict(seed=seed, exceeding=exceeding(stats, limits), **stats))
    evidence = dict(protocol='optimizer-v1', settings=settings, scalar_stamp=stamp,
                    platform=dict(machine=platform.machine(), libc=platform.libc_ver()),
                    input_sha256=hashlib.sha256(b''.join(arrays[k].tobytes() for k in KEYS)).hexdigest(),
                    thresholds=limits, calibration=calibration, heldout=heldout, arms={})
    rows = cpu.score_pairs(**arrays, **settings, grain=2, poly=True)
    production = statistics(arrays, anchor, rows, test)
    plant = cpu.score_pairs(**arrays, **{**settings, 'lr_q': .11}, grain=2, poly=True)
    plant_stats = statistics(arrays, anchor, plant, test)
    evidence['arms']['cpu_analytic'] = dict(ran=True, production=production,
        exceeding=exceeding(production, limits), plant=plant_stats,
        plant_exceeding=exceeding(plant_stats, limits), qualified=False,
        why='reduction controls do not cover polynomial transcendental changes')
    evidence['arms']['cuda_analytic'] = dict(ran=False, qualified=False, why='CUDA not requested')
    if cuda:
        try:
            from roshambo3.backend.spec import PoseSearch
            from roshambo3.layout import tile32
            from roshambo3._build_info import get_build_info
            gpu = backend.select('cuda')
            inputs = dict(arrays, library_coords=tile32(arrays['library_coords']),
                query_types=np.zeros(arrays['query_coords'].shape[:2], np.int32),
                interaction_r=np.ones((1, 1), np.float32), interaction_p=np.zeros((1, 1), np.float32))
            rows = gpu.score_pairs(**inputs, **settings, pose_search=PoseSearch.ANALYTIC)
            plant = gpu.score_pairs(**inputs, **{**settings, 'lr_q': .11}, pose_search=PoseSearch.ANALYTIC)
            stats, ps = statistics(arrays, anchor, rows, test), statistics(arrays, anchor, plant, test)
            bi = get_build_info()
            evidence['arms']['cuda_analytic'] = dict(ran=True, qualified=False, production=stats,
                exceeding=exceeding(stats, limits), plant=ps, plant_exceeding=exceeding(ps, limits),
                stamp={k: getattr(bi, k, None) for k in ('cuda_src_sha256', 'nvcc_version', 'cuda_architectures')},
                why='reduction controls do not cover ex2.approx versus libm')
        except Exception as exc:
            evidence['arms']['cuda_analytic']['why'] = repr(exc)
    evidence['heldout_valid'] = not any(row['exceeding'] for row in heldout)
    evidence['qualified'] = False
    evidence['gaps'] = ['transcendental controls missing', 'real molecule/color optimizer panel not run',
                        'basin metric is an RMS proxy; independent basin assignment not established',
                        'CPU native rows do not expose winning-start IDs; column16 is quaternion w, never an ID']
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path('tests/data/anchor_fixture.npz'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    evidence = run(load_inputs(args.fixture), cuda=args.cuda)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + '\n')
    print('Optimizer NOT QUALIFIED; inspect', args.out)
    return 0 if evidence['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

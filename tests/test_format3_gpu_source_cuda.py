"""Untimed format-3 CUDA consumers of format-3 writer fixtures.

Acceptance tests on prepare()-built corpora are not here.
"""
from contextlib import closing

import numpy as np
import pytest

from roshambo3.backend import cuda, cpu
from roshambo3.engine import R3gSource, Engine, ScreenConfig, TopK, MoleculeTopK, DenseScores
from roshambo3.engine.query import from_arrays
from roshambo3.metrics.ranking import canonical_ids
from roshambo3.numerics import rank_order, WIRE_QUANT_STEP
from tests.test_format3_gpu_source import written, reference_group

pytestmark = pytest.mark.skipif(not cuda.available(), reason="CUDA extension unavailable")


def query_and_identity(corpus, color, gen):
    ref = reference_group(corpus, 0)
    counts = ref["counts"]
    real = int(counts[0, 0])
    merged = ref["merged"]
    coords = [merged[i, :, :real].T.astype(np.float32) * WIRE_QUANT_STEP for i in [0, 4]]
    colors = None
    if color:
        colors = []
        for i in [0, 4]:
            types = np.repeat(np.arange(1, counts.shape[1]), counts[i, 1:]).astype(np.int32)
            colors.append((merged[i, :, real:real + len(types)].T.astype(np.float32) * WIRE_QUANT_STEP, types))
    qs = from_arrays(coords, ["first", "second"], color_generator=gen if color else None,
                     color=colors, backend="cuda")
    confs = []
    with corpus.group(0) as group:
        for nc in group.runs["n_conf"]:
            confs.extend(range(int(nc)))
    ids = canonical_ids(ref["mol_id"], confs)
    return qs, ref, ids


@pytest.mark.parametrize("launch", [1, 50, 65536])
@pytest.mark.parametrize("depth", [3, 8])
def test_cuda_topk_identity_across_launches_and_prefetch(written, launch, depth):
    corpus, color, gen = written
    qs, _, identity = query_and_identity(corpus, color, gen)
    cfg = ScreenConfig(color=color, mixing=.5 if color else 0.)
    with closing(Engine(backend="cuda")) as engine:
        base_source = R3gSource(corpus, color=color, color_generator=gen,
                                conformers_per_launch=1, prefetch_depth=0)
        base_stats, base = engine.run(base_source, qs, cfg, TopK(k=20))
        source = R3gSource(corpus, color=color, color_generator=gen,
                           conformers_per_launch=launch, workers=8, prefetch_depth=depth)
        stats, actual = engine.run(source, qs, cfg, TopK(k=20))
        assert base_stats.pinned and stats.pinned
        for (bs, bi), (scores, gids) in zip(base, actual, strict=True):
            np.testing.assert_array_equal(identity[bi], identity[gids])
            np.testing.assert_array_equal(scores, bs)
        assert source._pool._q.qsize() == len(source._pool.buffers)


def test_cuda_molecule_topk_matches_dense_host_reduction(written):
    corpus, color, gen = written
    qs, ref, _ = query_and_identity(corpus, color, gen)
    cfg = ScreenConfig(color=color, mixing=.5 if color else 0.)
    with closing(Engine(backend="cuda")) as engine:
        source = R3gSource(corpus, color=color, color_generator=gen,
                           conformers_per_launch=1, workers=8, prefetch_depth=8)
        _, dense = engine.run(source, qs, cfg, DenseScores(max_bytes=1 << 20))
        _, actual = engine.run(source, qs, cfg, MoleculeTopK(k=5))
    for q, (scores, gids) in enumerate(actual):
        order = rank_order(dense.scores[q, :, 0], dense.gids)
        selected = []
        seen = set()
        for row in order:
            mid = int(ref["mol_id"][dense.gids[row]])
            if mid not in seen:
                selected.append(row)
                seen.add(mid)
            if len(selected) == 5:
                break
        selected = np.asarray(selected)
        np.testing.assert_array_equal(ref["mol_id"][gids], ref["mol_id"][dense.gids[selected]])
        np.testing.assert_array_equal(scores, dense.scores[q, selected])


def test_cpu_cuda_fixed_pose_scores_of_native_source(written):
    corpus, color, gen = written
    qs, ref, _ = query_and_identity(corpus, color, gen)
    source = R3gSource(corpus, color=color, color_generator=gen)
    block = source._decode(source.jobs[0])
    rows = block.n_conformers
    common = dict(query_coords=qs.coords, query_types=qs.types, query_n_atoms=qs.n_real,
                  library_counts=block.counts,
                  pair_query_row=np.zeros(rows, np.int32),
                  pair_library_row=np.arange(rows, dtype=np.int32),
                  interaction_r=qs.interaction_r, interaction_p=qs.interaction_p,
                  color=color, mixing=.5 if color else 0., steps=0, start_mode=0)
    gpu = cuda.score_pairs(library_coords=block.coords, **common,
                           pose_search=cuda.PoseSearch.ANALYTIC,
                           library_color_offset=-1 if color else 0, library_shape_width=block.L_shape)
    physical = np.ascontiguousarray(ref["merged"].transpose(0, 2, 1).astype(np.float32) * WIRE_QUANT_STEP)
    host_common = dict(common, steps=1)
    host = cpu.score_pairs(library_coords=physical, **host_common,
                           lr_q=0, lr_t=0, budget_override="fixed-pose source check",
                           color_mode="combo_optimal" if color else "shape_only")
    # Use the standing per-path arithmetic models, rather than adding a
    # universal CPU/CUDA epsilon. This is a fixed-pose source diagnostic; it
    # does not certify cross-backend optimizer trajectories.
    from bench.numerics_backends import cpu_contract, cuda_contract, COLUMNS
    from roshambo3.reference.numerics_contracts import (
        fixed_pose_row, residual_bounds, assert_residuals)
    contracts = [cuda_contract(), cpu_contract(dict(query_coords=qs.coords,
                                                   library_coords=physical))]
    expected, limits = [], []
    for contract in contracts:
        oracle, bounds = [], []
        for i in range(rows):
            nr = int(block.counts[i, 0])
            ncol = int(block.counts[i, 1:].sum())
            qreal = int(qs.n_real[0])
            qtypes = qs.types[0, qreal:]
            qmask = qtypes > 0
            args = dict(query_color=qs.coords[0, qreal:, :3][qmask],
                        query_types=qtypes[qmask],
                        library_color=physical[i, nr:nr + ncol],
                        library_types=np.repeat(np.arange(1, block.counts.shape[1]),
                                                block.counts[i, 1:]),
                        interaction_r=qs.interaction_r, interaction_p=qs.interaction_p,
                        mixing=.5 if color else 0.)
            a, b = qs.coords[0, :qreal, :3], physical[i, :nr]
            oracle.append(fixed_pose_row(a, b, contract=contract, **args))
            bounds.append(residual_bounds(a, b, contract=contract, **args))
        expected.append(oracle)
        limits.append(bounds)
    for scores, oracle, bounds, contract in zip([gpu, host], expected, limits, contracts, strict=True):
        observed = [dict(zip(COLUMNS, map(float, row[:9]))) for row in scores]
        assert_residuals(observed, oracle, bounds, contract=contract)
    # Both paths agree within the sum of their individually derived bounds.
    bound = np.array([[limits[0][i][name] + limits[1][i][name] for name in COLUMNS]
                      for i in range(rows)])
    assert np.all(np.abs(gpu[:, :9] - host[:, :9]) <= bound)

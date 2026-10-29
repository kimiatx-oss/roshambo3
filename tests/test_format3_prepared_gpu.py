"""Prepare-built format-3 CPU/CUDA parity under the standing CPU/CUDA bounds."""
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from roshambo3.backend import cuda, cpu, PoseSearch, ColorPlacement
from roshambo3.engine import DenseScores, Engine, R3gSource, ScreenConfig
from tests.format3_gpu_fixtures import prepared_gpu_corpus
from gpu_device import device_present

pytestmark = pytest.mark.skipif(not cuda.available() or not cpu.available() or not device_present(),
                                reason='CPU/CUDA extensions and CUDA device required')


def test_prepared_cpu_cuda_screen_scores_at_a_fixed_pose(prepared_gpu_corpus):
    """Exercise ordinary source rows on both backends; do not compare optimizer basins.

    The existing path arithmetic envelopes bound each backend independently
    against float64; their sum is the CPU/CUDA parity bound. No new epsilon.
    """
    from bench.numerics_backends import cpu_contract, cuda_contract, COLUMNS
    from roshambo3.reference.numerics_contracts import fixed_pose_row, residual_bounds, assert_residuals
    fx = prepared_gpu_corpus
    color = fx.generator is not None
    cfg = ScreenConfig(color=color, mixing=.5 if color else 0., steps=1,
                       color_stage=0, color_residency=1,
                       start_mode=0, lr_q=0., lr_t=0., pose_search=PoseSearch.ANALYTIC,
                       color_placement=ColorPlacement.IN_GRADIENT if color else ColorPlacement.AT_POSE,
                       budget_override='fixed-pose source parity under existing CPU/CUDA bounds')
    results = []
    for name in ('cuda', 'cpu'):
        src = R3gSource(fx.corpus, color=color, color_generator=fx.generator,
                         tile=name == 'cuda', prefetch_depth=0, conformers_per_launch=10)
        with closing(Engine(backend=name)) as engine:
            if name == 'cpu' and color:
                # CPU color's supported scalar mode-3 entry consumes the
                # ordinary source's complete decoded blocks. This avoids the
                # preexisting ordinary Engine color-dispatch restriction and
                # preserves the explicit fixed-pose budget reason.
                batches, gids = [], []
                for block in src.blocks():
                    n = block.n_conformers
                    scores = cpu.score_pairs(
                        query_coords=fx.queries.coords, query_types=fx.queries.types,
                        query_n_atoms=fx.queries.n_real,
                        library_coords=block.coords, library_counts=block.counts,
                        pair_query_row=np.repeat(np.arange(fx.queries.n_conformers, dtype=np.int32), n),
                        pair_library_row=np.tile(np.arange(n, dtype=np.int32), fx.queries.n_conformers),
                        interaction_r=fx.queries.interaction_r, interaction_p=fx.queries.interaction_p,
                        color=True, color_mode='combo_optimal', mixing=.5,
                        steps=1, lr_q=0., lr_t=0., start_mode=0,
                        budget_override=cfg.budget_override)
                    batches.append(scores.reshape(fx.queries.n_conformers, n, -1))
                    gids.extend(range(block.gid_base, block.gid_base + n))
                result = SimpleNamespace(gids=np.asarray(gids), scores=np.concatenate(batches, axis=1))
            else:
                _, result = engine.run(src, fx.queries, cfg, DenseScores(max_bytes=1 << 25))
        results.append(result)
    gpu, host = results
    np.testing.assert_array_equal(fx.identity[gpu.gids], fx.identity[host.gids])
    assert len(gpu.gids) == fx.corpus.n_conf
    # Rows in each result are canonical corpus order, including across groups.
    bounds_by_backend = []
    for result, backend_name in zip(results, ('cuda', 'cpu'), strict=True):
        all_bounds = []
        for qi in range(fx.queries.n_conformers):
            expected, limits = [], []
            nrq = int(fx.queries.n_real[qi])
            qmask = fx.queries.types[qi] > 0
            for group in fx.rows.groups:
                physical = group.merged.transpose(0, 2, 1).astype(np.float32) / 256
                contract = (cuda_contract() if backend_name == 'cuda' else
                            cpu_contract(dict(query_coords=fx.queries.coords, library_coords=physical)))
                for i in range(group.n_conf):
                    nr = int(group.n_real[i])
                    types = np.repeat(np.arange(1, group.counts.shape[1]), group.counts[i, 1:])
                    args = dict(query_color=fx.queries.coords[qi, qmask, :3],
                                query_types=fx.queries.types[qi, qmask],
                                library_color=physical[i, nr:nr + len(types)], library_types=types,
                                interaction_r=fx.queries.interaction_r,
                                interaction_p=fx.queries.interaction_p, mixing=.5 if color else 0.)
                    a, b = fx.queries.coords[qi, :nrq, :3], physical[i, :nr]
                    expected.append(fixed_pose_row(a, b, contract=contract, **args))
                    limits.append(residual_bounds(a, b, contract=contract, **args))
            observed = [dict(zip(COLUMNS, map(float, row[:9]))) for row in result.scores[qi]]
            assert_residuals(observed, expected, limits, contract=contract)
            all_bounds.append([[bound[name] for name in COLUMNS] for bound in limits])
        bounds_by_backend.append(np.asarray(all_bounds))
    assert np.all(np.abs(gpu.scores[:, :, :9] - host.scores[:, :, :9]) <=
                  bounds_by_backend[0] + bounds_by_backend[1])

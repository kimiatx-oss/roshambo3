"""Same-build production analytic rows across repeated and partitioned launches."""
from dataclasses import asdict
import numpy as np
import pytest

from bench.numerics_backends import seeded_panel, cuda_fixed, cuda_contract, verified_build


@pytest.mark.numerics_coverage(backend='cuda', path='analytic', check='determinism', channel='shape')
@pytest.mark.numerics_coverage(backend='cuda', path='analytic', check='determinism', channel='color')
def test_cuda_analytic_repeated_and_partitioned_fixed_rows(numerics_coverage):
    from roshambo3.backend import cuda
    if not cuda.available():
        pytest.skip('CUDA unavailable; analytic launch/reduction axis not run')
    if cuda.extension().device_count() < 1:
        pytest.skip('CUDA device unavailable; analytic launch/reduction axis not run')
    build = verified_build('cuda')
    panel = seeded_panel(32)
    expected = cuda_fixed(panel)
    np.testing.assert_array_equal(cuda_fixed(panel).view(np.uint32), expected.view(np.uint32))
    layouts = ([3, 7, 22], [11, 1, 20])
    for sizes in layouts:
        observed = []
        begin = 0
        for size in sizes:
            portion = dict(panel)
            for key in ('pair_query_row', 'pair_library_row'):
                portion[key] = panel[key][begin:begin + size].copy()
            observed.append(cuda_fixed(portion))
            begin += size
        np.testing.assert_array_equal(np.concatenate(observed).view(np.uint32), expected.view(np.uint32))
    numerics_coverage.stamp(build=build, arithmetic=asdict(cuda_contract()), evidence=dict(
        rows=32, launch_partitions=layouts, repeated_full_launch=True,
        scope='analytic fixed-pose production score_pairs rows; optimizer, GPU-count and other axes separate'))

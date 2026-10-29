"""Live same-build GPU-count comparisons; no stored output bits."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os

import numpy as np
import pytest

from bench.numerics_backends import cuda_wire_inputs, seeded_panel, verified_build


@pytest.mark.parametrize('path', ['analytic', 'field'])
@pytest.mark.parametrize('color', [False, True], ids=['shape', 'color'])
def test_cuda_gpu_count_and_partition(path, color, record_property, request, numerics_coverage):
    request.node.add_marker(pytest.mark.numerics_coverage(
        backend="cuda", path=path, check="determinism", channel="color" if color else "shape"))
    indices = tuple(int(index) for index in
                    os.environ.get('ROSHAMBO3_NUMERICS_DEVICE_IDS', '0').split(','))
    if len(indices) < 2:
        pytest.skip('multi-GPU axis not run: use run_numerics_locked --devices 0,1')
    from roshambo3.backend import cuda
    if not cuda.available():
        pytest.fail('explicit multi-GPU qualification requested but CUDA is unavailable')
    extension = cuda.extension()
    if max(indices) >= extension.device_count():
        pytest.fail('requested multi-GPU ordinal is unavailable')
    build = verified_build('cuda')
    panel = seeded_panel(16)
    panel['interaction_r'][1:, 1:] = 1
    panel['interaction_p'][1:, 1:] = np.eye(2, dtype=np.float32)
    args = cuda_wire_inputs(panel)
    common = dict(steps=4, start_mode=1, color=color, mixing=.5 if color else 0.,
                  pose_search=cuda.PoseSearch.FIELD if path == 'field' else cuda.PoseSearch.ANALYTIC,
                  color_placement=cuda.ColorPlacement.IN_GRADIENT,
                  library_color_offset=-1 if color else 0)

    def score(device, rows):
        portion = dict(args)
        for key in ('pair_query_row', 'pair_library_row'):
            portion[key] = args[key][rows].copy()
        return cuda.score_pairs(**portion, **common, device_id=device)

    all_rows = np.arange(16)
    expected = score(indices[0], all_rows)
    # Test complete input on every physical GPU before combining partitions.
    for device in indices[1:]:
        np.testing.assert_array_equal(score(device, all_rows).view(np.uint32), expected.view(np.uint32))
    layouts = []
    for count in range(2, len(indices) + 1):
        for interleaved in (False, True):
            partitions = ([all_rows[rank::count] for rank in range(count)] if interleaved
                          else np.array_split(all_rows, count))
            with ThreadPoolExecutor(max_workers=count) as pool:
                jobs = [pool.submit(score, device, rows)
                        for device, rows in zip(indices, partitions)]
                actual = np.empty_like(expected)
                for rows, job in zip(partitions, jobs):
                    actual[rows] = job.result()
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            layouts.append(dict(device_count=count, interleaved=interleaved,
                                partition_sizes=list(map(len, partitions))))
    evidence = dict(
        build=build, path=path, color=color, devices=[extension.device_info(i) for i in indices],
        layouts=layouts, scope='direct score_pairs optimizer and final rows; engine scheduling separate')
    record_property('numerics_multigpu', json.dumps(evidence, sort_keys=True))
    from bench.numerics_backends import cuda_contract, cuda_field_contract
    contract = cuda_field_contract() if path == 'field' else cuda_contract()
    numerics_coverage.stamp(build=build, arithmetic=asdict(contract), evidence=evidence)
    # A missing GPU-count axis keeps the combined determinism cell not_run.
    # Passing this axis still requires the other scheduling tests and native
    # reduction fault evidence before the cell can certify.

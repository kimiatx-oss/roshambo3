"""Same-build score/pose invariance across production FIELD scheduling axes."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.numerics import rank_order

pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")


def population():
    rng = np.random.default_rng(304)
    q = np.zeros((2, 24, 4), np.float32)
    b = np.zeros((17, 24, 4), np.float32)
    q[:, :19, :3] = rng.normal(0, 1.7, (2, 19, 3))
    b[:, :19, :3] = rng.normal(0, 1.6, (17, 19, 3))
    q[:, :19, 3] = b[:, :19, 3] = 1
    qt = np.zeros((2, 24), np.int32)
    qt[:, 13:16] = 1
    qt[:, 16:19] = 2
    return dict(query_coords=q, query_types=qt,
                query_n_atoms=np.full(2, 13, np.int32), library_coords=b,
                library_counts=np.tile([13, 3, 3], (17, 1)).astype(np.uint8),
                interaction_r=np.ones((3, 3), np.float32),
                interaction_p=np.diag([0, 1, 1]).astype(np.float32))


def score(data, rows, *, combo=False, threads=1, query_rows=None):
    if "_fields" not in data:
        data["_fields"] = cpu.prepare_combo_shape_evaluators(
            data["query_coords"], data["query_n_atoms"],
            shape_backend="pipelined_field", shape_grain="library_atom",
            shape_width=8, field_spacing_angstrom=.5)
        data["_shared"] = cpu.prepare_combo_shared_pose_queries(
            data["query_coords"], data["query_types"], data["query_n_atoms"],
            data["interaction_r"], data["interaction_p"],
            shape_backend="pipelined_field", shape_grain="library_atom",
            shape_width=8, field_spacing_angstrom=.5,
            shape_evaluators=data["_fields"])
    args = {key: value for key, value in data.items() if not key.startswith("_")}
    args.update(pair_query_row=(np.zeros(len(rows), np.int32) if query_rows is None
                                else np.asarray(query_rows, np.int32)),
                pair_library_row=np.asarray(rows, np.int32),
                steps=25, start_mode=1, threads=threads)
    if combo:
        return np.asarray(cpu.score_pairs(**args, color=True, color_mode="combo_optimal",
            mixing=.5, color_grain=16, canonical_color_kernel=True,
            factor_canonical_torque=True, optimizer_exp_mode="rawbits_unclamped",
            shape_backend="pipelined_field", shape_grain="library_atom",
            shape_width=8, shape_field_spacing_angstrom=.5,
            prepared_shape_evaluators=data["_fields"],
            prepared_shared_pose_queries=data["_shared"]))
    for key in ("query_types", "interaction_r", "interaction_p"):
        args.pop(key)
    return np.asarray(cpu.score_pairs(**args, field_h=.5, field_width=8,
        field_unroll=0, field_pipeline=True, field_gridxf=1, field_allin=1,
        field_batched=True, prepared_query_fields=data["_fields"]))


def assert_same(a, b):
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint32),
                                  np.ascontiguousarray(b).view(np.uint32))


@pytest.mark.parametrize("combo", [False, True], ids=["shape", "combo"])
def test_lengths_positions_partitions_repeats_and_workers(combo):
    data = population()
    ids = np.arange(17, dtype=np.int32)
    expected = score(data, ids, combo=combo)
    assert np.isfinite(expected).all()
    for length in range(1, 18):
        # Prefix calls span every complete/partial-batch length. Partitioning
        # the full library also puts the same final pair at each tail offset.
        assert_same(expected[:length], score(data, ids[:length], combo=combo))
        chunks = [ids[i:i + length] for i in range(0, 17, length)]
        for workers in (1, 2, 3):
            with ThreadPoolExecutor(workers) as pool:
                actual = np.concatenate(list(pool.map(
                    lambda rows: score(data, rows, combo=combo), chunks)))
            assert_same(expected, actual)
            np.testing.assert_array_equal(rank_order(expected[:, 0], ids),
                                          rank_order(actual[:, 0], ids))
    for offset in range(8):
        rows = np.roll(ids, offset)
        for workers in (1, 2, 3):
            for _ in range(2):
                actual = score(data, rows, combo=combo, threads=workers)
                assert_same(expected[rows], actual)


@pytest.mark.parametrize("combo", [False, True], ids=["shape", "combo"])
def test_query_boundaries_do_not_select_other_arithmetic(combo):
    data = population()
    ids = np.arange(17, dtype=np.int32)
    queries = ids % 2
    expected = np.concatenate([score(data, [row], combo=combo,
        query_rows=[query]) for row, query in zip(ids, queries)])
    for workers in (1, 2, 3):
        assert_same(expected, score(data, ids, combo=combo, threads=workers,
                                   query_rows=queries))


def test_full_and_partial_shape_batches_match_independent_fp64_final_pose():
    from roshambo3 import reference
    data = population()
    for length in (1, 7, 8, 9, 16, 17):
        actual = score(data, np.arange(length, dtype=np.int32))
        a = data["query_coords"][0, :13, :3]
        vaa = reference.self_overlap(a)
        for row, result in enumerate(actual):
            b = data["library_coords"][row, :13, :3]
            vbb = reference.self_overlap(b)
            posed = reference.apply_pose(b, result[9:13].astype(np.float64),
                                         result[13:16].astype(np.float64))
            vab = reference.pair_overlap(a, posed)
            # Existing CPU oracle contract (test_cpu_kernel.py), unchanged.
            np.testing.assert_allclose(result[[3, 5, 6]], [vab, vaa, vbb],
                                       rtol=1e-5, atol=0)
            assert abs(float(result[1]) - reference.tanimoto(vab, vaa, vbb)) < 1e-5

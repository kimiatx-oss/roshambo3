"""Tiny real builder matrix complements the standalone deciding-cell harness."""
from bench.format3_cpu_matrix import run_matrix, synthetic_molecules
from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep import prepare


def test_builder_blocks_and_worker_counts_preserve_identity(tmp_path):
    inputs = synthetic_molecules(n_molecules=5, conformers=2)
    generator = PharmacophoreGenerator()
    paths = {}
    for rows in (4, 8, 16):
        paths[rows] = tmp_path / str(rows)
        prepare(inputs, str(paths[rows]), workers=2, block_rows=rows,
                color_generator=generator)
    report = run_matrix(paths, workers=[1, 2], k=5)
    assert len(report['cells']) == 12
    assert all(c['identity_equal'] and c['score_columns_bit_equal'] for c in report['cells'])
    assert {c['blocks'] for c in report['cells']} == {1, 2, 3}


def test_matrix_rejects_a_score_bit_change_hidden_by_float_equality(tmp_path, monkeypatch):
    import numpy as np
    import pytest
    from roshambo3.engine import Engine

    path = tmp_path / 'corpus'
    prepare(synthetic_molecules(n_molecules=3, conformers=2), str(path),
            workers=2, block_rows=4, color_generator=PharmacophoreGenerator())
    original = Engine.run_worker_owned_cpu
    calls = [0]

    def altered(engine, *args, **kwargs):
        stats, rows = original(engine, *args, **kwargs)
        if calls[0] == 1:
            scores, gids, mids = rows[0]
            changed = scores.copy()
            # Shape-only color is exactly zero. Flip its sign bit: all float
            # comparisons still agree, but a bit-identity claim must refuse.
            assert changed[0, 2] == 0
            changed.view(np.uint32)[0, 2] ^= np.uint32(1 << 31)
            np.testing.assert_array_equal(changed, scores)
            rows[0] = (changed, gids, mids)
        calls[0] += 1
        return stats, rows

    monkeypatch.setattr(Engine, 'run_worker_owned_cpu', altered)
    with pytest.raises(AssertionError):
        run_matrix({4: path}, workers=[1, 2], k=3)
    assert calls[0] == 2

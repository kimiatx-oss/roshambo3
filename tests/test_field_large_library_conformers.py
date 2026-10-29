"""The field path scores a library conformer of more than 64 atoms correctly.

The x86 field kernels copy each library conformer into a fixed-size SoA
record. Its capacity was 64 atoms: `build()` clamped the copy there, but every
gradient loop ran to the conformer's real atom count, so atoms past 64 were
read from the neighboring arrays (x[64..] is y[0..]) and, past 72, from beyond
the record. A 78-atom conformer in a 2K smoke corpus scored 0.47 against itself and
the screen occasionally crashed. Real corpora carry such conformers: 113,192
of one 55.8M-conformer corpus exceed 64 atoms, the largest 134.

The analytic path never had the record, so it is the reference: a self-match
on every field dispatch must agree with it, at sizes on both sides of the old
capacity and up to the largest count the uint8 corpus format can store.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("roshambo3._roshambo3_cpu")
import roshambo3._roshambo3_cpu as ext                          # noqa: E402
from cpu_test_capabilities import IS_ARM                        # noqa: E402

#: x86-only arms; AArch64 refuses them by name (interface.cpp).
X86_ONLY = ("field_affine", "field_pairaos", "field_edgefield")

SIZES = (40, 64, 65, 78, 134, 255)


def _self_pairs(sizes=SIZES, seed=11):
    """Each conformer paired with itself, padded to the widest one."""
    rng = np.random.default_rng(seed)
    width = max(sizes)
    coords = np.zeros((len(sizes), width, 4), np.float32)
    for i, n in enumerate(sizes):
        # Atom density close to a real molecule's: about 1.5 A between nearest
        # neighbors, so every atom overlaps several others.
        side = 1.5 * n ** (1 / 3)
        coords[i, :n, :3] = rng.uniform(-side / 2, side / 2, size=(n, 3))
        coords[i, :n, 3] = 1.0
    rows = np.arange(len(sizes), dtype=np.int32)
    return dict(
        query_coords=coords, query_n_atoms=np.asarray(sizes, np.int32),
        library_coords=coords,
        library_counts=np.asarray(sizes, np.uint8)[:, None],
        pair_query_row=rows, pair_library_row=rows)


# 25 steps, the production budget: at 8 the two paths are still converging on
# this random geometry and differ by 2e-2 even at 40 atoms.
BASE = dict(steps=25, lr_q=0.1, lr_t=0.4, start_mode=1, threads=1)
FIELD = dict(BASE, field_h=0.5, field_width=8, field_unroll=0,
             field_pipeline=True)


@pytest.mark.parametrize("dispatch", [
    pytest.param({}, id="plain-blocked"),
    pytest.param({"field_allin": True}, id="allin"),
    pytest.param({"field_gridxf": True}, id="gridxf"),
    pytest.param({"field_gridxf": True, "field_allin": True}, id="both"),
    pytest.param({"field_affine": True}, id="affine"),
    pytest.param({"field_pairaos": True}, id="pairaos"),
    pytest.param({"field_edgefield": True}, id="edgefield"),
    pytest.param({"field_unroll": 2}, id="unroll2"),
])
def test_self_match_agrees_with_the_analytic_path(dispatch):
    pairs = _self_pairs()
    analytic = ext.score_pairs(**BASE, **pairs)
    if IS_ARM and any(dispatch.get(key) for key in X86_ONLY):
        with pytest.raises(ValueError, match="FP32 stored-gradient"):
            ext.score_pairs(**{**FIELD, **dispatch}, **pairs)
        return
    field = ext.score_pairs(**{**FIELD, **dispatch}, **pairs)
    # Column 1 is the shape Tanimoto. A self-match is ~1 on both paths; the
    # bug put the 78-atom row at 0.47.
    assert np.all(analytic[:, 1] > 0.9999)
    np.testing.assert_allclose(field[:, 1], analytic[:, 1], atol=1e-3,
                               err_msg=f"sizes {SIZES}, dispatch {dispatch}")


@pytest.mark.parametrize("n_atoms", [40, 78, 134, 255])
def test_the_eight_pair_batch_scores_large_conformers_too(n_atoms):
    """The batched arm builds eight records at once; same contract.

    It takes eight consecutive pairs sharing a query, so the library is eight
    copies of the query: eight self-matches in one batch.
    """
    if IS_ARM:
        pytest.skip("the eight-pair batched arm is x86-only; AArch64 scores "
                    "these pairs one at a time, which the test above covers")
    one = _self_pairs((n_atoms,))
    pairs = dict(one, library_coords=np.repeat(one["library_coords"], 8, 0),
                 library_counts=np.repeat(one["library_counts"], 8, 0),
                 pair_query_row=np.zeros(8, np.int32),
                 pair_library_row=np.arange(8, dtype=np.int32))
    kw = dict(FIELD, field_gridxf=True, field_allin=True, field_batched=True)
    batched = ext.score_pairs(**kw, **pairs)
    analytic = ext.score_pairs(**BASE, **one)
    np.testing.assert_allclose(batched[:, 1], np.repeat(analytic[:, 1], 8),
                               atol=1e-3, err_msg=f"{n_atoms} atoms")

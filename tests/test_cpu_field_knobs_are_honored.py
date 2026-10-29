"""A knob that is accepted must change something, or be refused.

Found in review. `field_winner_only` and `field_report_field_score` were
forwarded by two of the three blocked-kernel dispatch macros. A run that set
them with neither `field_gridxf` nor `field_allin` took the third,
`R3_BLK_S`, which dropped them -- so it reported full-rescore numbers under a
winner-only label, silently.

The validation block above the dispatch refused the wrong KERNELS (fp16,
affine, coeff, ...) and said nothing about the right kernel's wrong dispatch,
which is why the guard did not catch it. This file tests the property the guard
was reaching for: **accepted implies honored.**

`interface.cpp`'s standing contract is that an inapplicable knob is refused
rather than ignored, so both branches are tested -- honored where supported,
raised where not.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("roshambo3._roshambo3_cpu")
from cpu_test_capabilities import IS_ARM
import roshambo3._roshambo3_cpu as ext                          # noqa: E402


def _tiny(n_q=2, n_lib=48, n_atoms=12, seed=5):
    rng = np.random.default_rng(seed)
    qc = np.zeros((n_q, n_atoms, 4), np.float32)
    qc[:, :, :3] = rng.normal(0, 2.0, size=(n_q, n_atoms, 3))
    qc[:, :, 3] = 1.0
    lc = np.zeros((n_lib, n_atoms, 4), np.float32)
    lc[:, :, :3] = rng.normal(0, 2.0, size=(n_lib, n_atoms, 3))
    lc[:, :, 3] = 1.0
    return dict(
        query_coords=qc, query_n_atoms=np.full(n_q, n_atoms, np.int32),
        library_coords=lc,
        library_counts=np.full((n_lib, 1), n_atoms, np.uint8),
        pair_query_row=np.repeat(np.arange(n_q, dtype=np.int32), n_lib),
        pair_library_row=np.tile(np.arange(n_lib, dtype=np.int32), n_q))


BASE = dict(steps=8, lr_q=0.1, lr_t=0.4, start_mode=1, threads=1,
            field_h=0.8, field_width=8, field_unroll=0, field_pipeline=True)


@pytest.mark.parametrize("dispatch", [
    pytest.param({}, id="plain-blocked"),
    pytest.param({"field_allin": True}, id="allin"),
    pytest.param({"field_gridxf": True}, id="gridxf"),
    pytest.param({"field_gridxf": True, "field_allin": True}, id="both"),
    # The null arms. They were refused here until it turned out the
    # refusal was the only thing stopping the composition question from being
    # asked -- all of them route through the same search harness, which is
    # what implements winner-only.
    pytest.param({"field_fp16": True}, id="fp16"),
    pytest.param({"field_affine": True}, id="affine"),
    pytest.param({"field_coeff": True}, id="coeff"),
    pytest.param({"field_pairaos": True}, id="pairaos"),
    pytest.param({"field_edgefield": True}, id="edgefield"),
    pytest.param({"field_unroll": 2}, id="unroll2"),
])
def test_winner_only_is_honored_on_every_supported_dispatch(dispatch):
    """The bug: only two of the three dispatch macros forwarded the flag."""
    a = _tiny()
    kw = {**BASE, **dispatch}          # dispatch may override BASE's own keys
    if IS_ARM and any(kw.get(key) for key in
                      ("field_fp16", "field_affine", "field_coeff", "field_pairaos", "field_edgefield")):
        with pytest.raises(ValueError, match="FP32 stored-gradient"):
            ext.score_pairs(**kw, **a)
        return
    off = ext.score_pairs(**kw, **a)
    on = ext.score_pairs(**kw, field_winner_only=True, **a)
    assert not np.array_equal(off, on), (
        f"field_winner_only was silently ignored on dispatch {dispatch} -- it "
        "was accepted, so it must change what runs or be refused")


@pytest.mark.parametrize("dispatch", [
    {}, {"field_allin": True}, {"field_gridxf": True}])
def test_reporting_the_field_score_is_honored_everywhere_too(dispatch):
    a = _tiny()
    exact = ext.score_pairs(**BASE, **dispatch, **a)
    if IS_ARM:
        with pytest.raises(ValueError, match="FP32 stored-gradient"):
            ext.score_pairs(**BASE, **dispatch, field_report_field_score=True, **a)
        return
    field = ext.score_pairs(**BASE, **dispatch,
                            field_report_field_score=True, **a)
    assert not np.array_equal(exact, field)
    # The field's own value approximates the exact one; a wildly different
    # scale would mean the two are not the same quantity, which is what arm 3's
    # whole negative result depends on.
    d = np.abs(field[:, 1] - exact[:, 1])
    assert d.max() < 0.2, f"field score is not the same quantity: max {d.max()}"


VARIANTS = [{"field_fp16": True}, {"field_affine": True},
            {"field_coeff": True}, {"field_pairaos": True},
            {"field_edgefield": True}]


@pytest.mark.parametrize("arm", VARIANTS)
def test_gridxf_allin_are_honored_on_the_variant_arms(arm):
    """And honored *correctly*, which is the half a change-detector misses.

    **This test cannot see a gradient-scale error, and it passed for the entire
    time one was live.** The `2*hs` torque rescale that `GRIDXF` owes was
    missing from all four variants from `05852b9` until `60efcc2`, and every
    assertion here stayed green throughout: per-component Adagrad divides a
    constant factor on the gradient out almost exactly, so the scores this
    compares are nearly invariant to it. The bar below is a bar on the *pose
    space* being right, not on the gradient being right. The gradient-scale
    gate is the instrument that can tell the difference -- a finite-difference check
    against `reference/overlap.py`, which shares code with neither kernel.

    Each variant already TOOK an ALLIN/GRIDXF template parameter and forwarded
    it to `prepare_block`, but none of them folded the grid transform into the
    pose -- that lived only in the shipped kernel. Switching one on would have
    fed a world-space pose to a grid-space consumer. So "it changed something"
    is not enough here: a wrong pose space changes things too, by about four
    orders more. The bar is the perturbation the SHIPPED arm shows for the
    same flip.
    """
    a = _tiny()
    kw = {**BASE, **arm}
    if IS_ARM:
        with pytest.raises(ValueError, match="FP32 stored-gradient"):
            ext.score_pairs(**kw, field_gridxf=0, field_allin=0, **a)
        return
    off = ext.score_pairs(**kw, field_gridxf=0, field_allin=0, **a)
    on = ext.score_pairs(**kw, field_gridxf=1, field_allin=1, **a)
    assert not np.array_equal(off, on), (
        f"gridxf+allin silently ignored on {arm}")
    ship_off = ext.score_pairs(**BASE, field_gridxf=0, field_allin=0, **a)
    ship_on = ext.score_pairs(**BASE, field_gridxf=1, field_allin=1, **a)
    bar = np.abs(ship_on[:, 1].astype(np.float64)
                 - ship_off[:, 1].astype(np.float64)).max()
    d = np.abs(on[:, 1].astype(np.float64) - off[:, 1].astype(np.float64)).max()
    assert d < max(bar * 20.0, 1e-3), (
        f"gridxf on {arm} moved scores by {d:.3e}, far past the shipped arm's "
        f"own {bar:.3e} -- the pose is probably in the wrong space")


@pytest.mark.parametrize("bad", [
    pytest.param({"field_width": 1}, id="scalar-sampler"),
    pytest.param({"field_unroll": 2, "field_gridxf": 1}, id="unrolled-gridxf"),
    pytest.param({"field_coeff": True, "field_allin": 1, "field_gridxf": 0},
                 id="variant-split-pair"),
    pytest.param({"field_edgefield": True, "field_pipeline": False,
                  "field_allin": 1, "field_gridxf": 1}, id="variant-no-pipe"),
    # Found in review: field_batched was accepted and silently ignored
    # outside the one configuration whose dispatch implements it, so a run
    # could report unbatched numbers under a batched label.
    pytest.param({"field_batched": True, "field_allin": 0, "field_gridxf": 1},
                 id="batched-without-allin"),
    pytest.param({"field_batched": True, "field_allin": 1, "field_gridxf": 0},
                 id="batched-without-gridxf"),
    pytest.param({"field_batched": True, "field_unroll": 2}, id="batched-unrolled"),
    pytest.param({"field_batched": True, "field_coeff": True},
                 id="batched-variant-arm"),
])
def test_unsupported_combinations_refuse_rather_than_ignore(bad):
    """What is still out of reach, and refused rather than quietly dropped.

    The scalar sampler does not route through the shared harness. The unrolled
    arm prepares one sample at a time and so shares no code with the blocked
    prepare, where gridxf and allin live. The variants carry the pair only,
    and only with the pipeline, because each extra instantiation in that
    dispatch has a measured cost to the shipping path.
    """
    a = _tiny()
    kw = {**BASE, **bad}
    # field_winner_only is added only where the case is not already about a
    # knob of its own; the batched cases carry their refusal in `bad`.
    extra = {} if "field_batched" in bad else {"field_winner_only": True}
    with pytest.raises(Exception):
        ext.score_pairs(**kw, **extra, **a)

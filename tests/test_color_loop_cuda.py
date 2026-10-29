"""The color loop's structure and the color evaluation's placement.

What is pinned here, and why each one is a test rather than a comment:

* **The structures are bit-identical over the terms they evaluate.** They
  differ only over terms that are exactly ``+0.0f`` -- a cross-type pair's
  height on the canonical matrix -- but the same-type form also hoists a
  reciprocal and a ``__fsqrt_rn`` out of the atom-pair loop, which changes what
  ptxas may contract into an FFMA, so a value-preserving change is checked
  bitwise.

  The reference arm is ``BOUNDED``. ``MASKED_PADDED`` is retired: it was the
  only structure whose loop reached into the padding, and ``BOUNDED``
  reproduced it bitwise.

  The step counts are parametrized rather than collapsed onto the shipped 25,
  because they localize a failure -- terms, first gradient, or trajectory --
  and because a state has been measured (the counts driving a loop that
  still walked slots) where the terms agreed exactly and the 25-step trajectory
  did not.

* **The same-type structures are refused on a non-canonical matrix.** They
  evaluate only ``ta == tb``; where an off-diagonal interaction has real height
  that is a term they never reach, so they would compute a different quantity
  and report success. Silence there is the failure mode this whole area keeps
  producing (a zero ``OVERLAP_COLOR``), so it is a named refusal.

* **``AT_POSE`` does not change the shape channel at all.** Mode 2's pose is the
  shape-optimal one, start selection included, so its shape Tanimoto must equal
  a shape-only screen's bit for bit. That is what makes the color term a rescore
  of a pose that was already found, which is what a survivor-only second pass
  needs.

The fixture is deliberate: the same molecules, same non-canonical matrix,
``mixing`` never 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import backend, scores as S
from roshambo3.backend import ColorLoop, ColorPlacement, PoseSearch, UpdateRule

from test_color_path_cuda import (INTERACTION_P, INTERACTION_R, MIXING,
                                  _arrays, needs_gpu)

#: The canonical diagonal matrices, which is what a corpus is built with and the
#: only configuration the same-type structures are exact for.
CANON_R = np.eye(3, dtype=np.float32)
CANON_P = np.eye(3, dtype=np.float32)
CANON_R[0, 0] = 1.0
CANON_P[0, 0] = 0.0
CANON_R[CANON_R == 0.0] = 1.0   # slot 0 and the off-diagonal: width 1, height 0


def _canonical_arrays(**kw):
    a = _arrays(**kw)
    a["interaction_r"] = CANON_R
    a["interaction_p"] = CANON_P
    return a


#: The structures a launch can still name. `ColorLoop.MASKED_PADDED` is
#: retired; it is refused, not instantiated.
LIVE_LOOPS = (ColorLoop.BOUNDED, ColorLoop.SAME_TYPE,
              ColorLoop.SAME_TYPE_NOW)


@needs_gpu
class TestTheStructuresAreBitIdentical:

    @pytest.mark.parametrize("steps,start_mode",
                             [(0, 0), (1, 1), (5, 1), (25, 1)])
    def test_every_structure_matches_the_bounded_loop_bit_for_bit(self, steps,
                                                                  start_mode):
        """The structures compute the same terms, and the same trajectory.

        At ``steps = 0`` this is the color TERMS alone: same slots, same order,
        same bits. The longer arms carry the gradient too, and the 25-step arm
        is the shipped configuration, where any difference in what ptxas
        contracts into an FFMA would have 25 Adagrad steps to amplify. The step
        counts are parametrized rather than collapsed because they localize a
        failure: terms, first gradient, or trajectory.
        """
        for placement in ColorPlacement:
            kw = dict(pose_search=PoseSearch.ANALYTIC, color=True, mixing=MIXING,
                      steps=steps, start_mode=start_mode,
                      update_rule=UpdateRule.ADAGRAD_QADD,
                      color_placement=placement,
                      **_canonical_arrays(library_row=1))
            ref = backend.score_pairs(color_loop=ColorLoop.BOUNDED, **kw)
            assert ref[0, S.OVERLAP_COLOR] > 0.0, "the fixture scored no color"
            for loop in LIVE_LOOPS:
                v = backend.score_pairs(color_loop=loop, **kw)
                assert np.array_equal(v.view(np.uint32), ref.view(np.uint32)), (
                    loop, placement)


@needs_gpu
class TestWhatEachStructureIsAllowedToAssume:

    def test_the_retired_masked_structure_is_refused_by_name(self):
        """It was the only structure whose loop reached into the padding,
        and therefore the last thing on the library side that read the 0/1 mask
        in the 4th coordinate component. Refused rather than renumbered away, so
        a stale 0 fails loudly instead of selecting a different structure."""
        with pytest.raises(ValueError, match="MASKED_PADDED was retired"):
            backend.score_pairs(
                pose_search=PoseSearch.ANALYTIC, color=True, mixing=MIXING,
                color_loop=ColorLoop.MASKED_PADDED, **_arrays())

    def test_a_same_type_structure_is_refused_on_a_non_canonical_matrix(self):
        for loop in (ColorLoop.SAME_TYPE, ColorLoop.SAME_TYPE_NOW):
            with pytest.raises(ValueError, match="same-type"):
                backend.score_pairs(
                    pose_search=PoseSearch.ANALYTIC, color=True, mixing=MIXING,
                    color_loop=loop, **_arrays())

    def test_an_unnamed_structure_resolves_to_one_the_matrix_admits(self):
        """Resolving a default is not ignoring an argument.

        This fixture's matrix is non-canonical, so the resolved structure must be
        BOUNDED -- the only one that evaluates a cross-type pair. That the call
        succeeds at all is half the test: had it resolved to a same-type
        structure, score_pairs would have refused it on this matrix. That it
        agrees with an explicit BOUNDED bit for bit is the other half, and says
        WHICH one it picked. Both calls pin the additive rule because the
        measurement arm SAME_TYPE is instantiated for that rule only.
        """
        kw = dict(pose_search=PoseSearch.ANALYTIC, color=True, mixing=MIXING,
                  update_rule=UpdateRule.ADAGRAD_QADD, **_arrays())
        v = backend.score_pairs(**kw)
        assert v[0, S.OVERLAP_COLOR] > 0.0
        ref = backend.score_pairs(color_loop=ColorLoop.BOUNDED, **kw)
        assert np.array_equal(v.view(np.uint32), ref.view(np.uint32))

    def test_the_default_rule_also_resolves_to_a_correct_structure(self):
        """A hole once briefly opened: a caller with their own
        non-canonical matrix reaches the color path at the DEFAULT update rule,
        where the chosen same-type structure is wrong for them. BOUNDED is
        therefore instantiated at every rule, not only the additive one."""
        v = backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=MIXING, **_arrays())
        assert v[0, S.OVERLAP_COLOR] > 0.0

    def test_naming_a_structure_for_a_shape_screen_is_refused(self):
        with pytest.raises(ValueError, match="color=False"):
            backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=False,
                                color_loop=ColorLoop.BOUNDED, stage_width=0,
                                **{k: v for k, v in _arrays().items()
                                   if k not in ("interaction_r", "interaction_p",
                                                "library_color_offset")},
                                interaction_r=INTERACTION_R,
                                interaction_p=INTERACTION_P)


@needs_gpu
class TestTheNoFeaturesGuardDoesNotFalsePositive:
    """The guard used to false-positive on a library narrower than a tile.

    In the TILED TYPE PLANE a conformer's slots were strided by BTILE within
    its tile, so a slice of the first ``n_library`` rows was not the first
    ``n_library`` conformers -- for a small library it was the shape region of
    tile 0, which is type 0 by construction. It was hit on a 5-molecule
    fixture and again on a six-conformer bucket of a whole-corpus screen.

    The count vector removes the confusion at its source: the count vector is one row per
    conformer and is not tiled, so a row slice means what it says. Both cases
    are kept here because the guard still has to fire on a library that
    genuinely carries nothing.
    """

    def _narrow(self, n_library):
        from roshambo3.layout import BTILE

        a = _canonical_arrays()
        # the fixture's library is one whole tile; keep the rows, shrink the
        # declared library so the OLD prefix slice would miss the features
        a["library_counts"] = a["library_counts"][:n_library]
        assert a["library_coords"].shape[0] == BTILE
        return a

    @pytest.mark.parametrize("n_library", [1, 6, 31])
    def test_a_library_narrower_than_one_tile_is_not_falsely_refused(self, n_library):
        a = self._narrow(n_library)
        v = backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=MIXING, steps=25, **a)
        assert v[0, S.OVERLAP_COLOR] > 0.0

    def test_a_library_that_genuinely_carries_no_feature_is_still_refused(self):
        a = self._narrow(6)
        a["library_counts"] = a["library_counts"].copy()
        a["library_counts"][:, 1:] = 0
        with pytest.raises(ValueError, match="no pharmacophore feature"):
            backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=MIXING, steps=25, **a)


@needs_gpu
class TestThePlacementDoesNotTouchTheShapeChannel:

    @pytest.mark.parametrize("steps,start_mode",
                             [(0, 0), (1, 1), (5, 1), (25, 1)])
    def test_at_pose_shape_tanimoto_equals_a_shape_only_screen(self, steps,
                                                               start_mode):
        """Mode 2's pose is the shape-optimal one, start selection included.

        Bit for bit, at every step count including the shipped 25.
        """
        a = _canonical_arrays(library_row=1)
        shape_args = {k: v for k, v in a.items()
                      if k != "library_color_offset"}
        common = dict(pose_search=PoseSearch.ANALYTIC, steps=steps,
                      start_mode=start_mode, update_rule=UpdateRule.ADAGRAD_QADD)
        color = backend.score_pairs(
            color=True, mixing=MIXING,
            color_placement=ColorPlacement.AT_POSE, **common, **a)
        shape = backend.score_pairs(
            color=False, mixing=0.0, stage_width=0, **common, **shape_args)
        assert (color[0, S.TANIMOTO_SHAPE].view(np.uint32)
                == shape[0, S.TANIMOTO_SHAPE].view(np.uint32))

    def test_in_gradient_is_still_reachable_and_optimizes_the_combination(self):
        """Mode 3 must be one argument away, not one rebuild away."""
        a = _canonical_arrays(library_row=1)
        common = dict(pose_search=PoseSearch.ANALYTIC, color=True,
                      mixing=MIXING, steps=25, start_mode=1,
                      update_rule=UpdateRule.ADAGRAD_QADD)
        combo = backend.score_pairs(
            color_placement=ColorPlacement.IN_GRADIENT, **common, **a)
        rescore = backend.score_pairs(
            color_placement=ColorPlacement.AT_POSE, **common, **a)
        # The combination is what mode 3 optimizes, so it may not be beaten by
        # the mode that does not optimize it. Not an equality: the two converge
        # to different poses on purpose.
        assert (combo[0, S.TANIMOTO_COMBINATION]
                >= rescore[0, S.TANIMOTO_COMBINATION] - 1e-6)

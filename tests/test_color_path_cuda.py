"""The color path, reached the way a caller reaches it.

Every test here goes through :func:`roshambo3.backend.score_pairs` -- not the
raw extension -- because "the kernel computes the color term correctly" was
already established (``test_reference.py``) at a point when no caller could
launch it. What was missing was a test of the *reachable* path, and that gap is
what let two independent defects sit behind each other:

1. ``stage_width == 0`` left the staging *format* flags at their pybind
   defaults, composing dispatch key 29, which is not instantiated. The launch
   threw.
2. Behind it, the library type plane was never put through
   :func:`roshambo3.layout.tile32`. The kernel read a library atom's type at
   the tile stride, so an untiled plane handed the color loop type 0 -- the
   ordinary-atom row of the interaction matrix, whose height is zero by
   construction -- and the color sum came out exactly zero.

**The second defect's whole surface is gone**: there is no library type
plane. The kernel takes the per-conformer count vector -- one 8-byte record per
conformer, one row per conformer, not tiled -- and recovers the run boundaries
itself. The class that pinned the untiled-plane mechanism is replaced by
``TestTheCountVectorIsLoadBearing``, which perturbs the counts and requires the
score to move, because a wrong count vector is a plausible wrong number rather
than the loud zero a wrong plane produced.

The fixture is built against known traps: ``mixing`` is never 0 (at 0 the
combination Tanimoto is exactly the shape Tanimoto and the test measures
nothing about color), the features are same-type at nonzero separation, and the
interaction matrix has off-diagonal entries and non-unit heights, which is the
configuration in which a spurious width factor does *not* cancel.
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import backend, scores as S
from roshambo3.numerics import WIRE_QUANT_STEP
from roshambo3.backend import PoseSearch
from roshambo3.engine import ScreenConfig
from roshambo3.layout import counts_from_types, tile32
from roshambo3.reference import overlap as R

needs_gpu = pytest.mark.skipif(not backend.available(),
                               reason="no CUDA extension in this build")

L = 8
N_REAL = 2
N_ROWS = 32
MIXING = 0.5

SHAPE_XYZ = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
#: Two same-type features at nonzero separation, plus one of a second type.
FEATURE_XYZ = np.array([[0.3, 0.0, 0.0], [1.6, 0.2, 0.0], [0.4, 1.1, -0.5]])
FEATURE_TYPES = np.array([1, 1, 2], np.int32)

#: Off-diagonal and non-unit: the canonical diagonal matrix is exactly the
#: configuration in which a spurious width factor cancels.
INTERACTION_R = np.array([[1.0, 1.0, 1.0],
                          [1.0, 1.3, 0.7],
                          [1.0, 0.7, 2.1]], np.float32)
INTERACTION_P = np.array([[0.0, 0.0, 0.0],
                          [0.0, 1.4, 0.6],
                          [0.0, 0.6, 0.9]], np.float32)


def _molecule():
    """``[L, 4]`` coordinates and ``[L]`` types: shape, then features, then padding."""
    coords = np.zeros((L, 4), np.float32)
    types = np.zeros(L, np.int32)
    coords[:N_REAL, :3] = SHAPE_XYZ
    coords[:N_REAL, 3] = 1.0
    n_c = FEATURE_XYZ.shape[0]
    coords[N_REAL:N_REAL + n_c, :3] = FEATURE_XYZ
    coords[N_REAL:N_REAL + n_c, 3] = 1.0
    types[N_REAL:N_REAL + n_c] = FEATURE_TYPES
    return coords, types


#: A second library molecule: same shape, features displaced. Row 0 is the
#: query's own twin (a self-match) and row 1 is not, which matters for any test
#: that needs the color channel to be DISTINGUISHABLE from the shape channel --
#: at a shape-optimal pose a self-match scores exactly 1.0 on both channels, so
#: the combination equals the shape score there for any mixing at all. That is
#: not a defect, it is what a self-match is; a mixing test written against
#: row 0 stops discriminating under the shape-optimal-pose default placement.
FEATURE_XYZ_2 = FEATURE_XYZ + np.array([0.45, -0.2, 0.15])


def _molecule2():
    coords, types = _molecule()
    n_c = FEATURE_XYZ_2.shape[0]
    coords[N_REAL:N_REAL + n_c, :3] = FEATURE_XYZ_2
    return coords, types


def _arrays(library_row: int = 0):
    coords, types = _molecule()
    lib_c = np.zeros((N_ROWS, L, 4), np.float32)
    lib_t = np.zeros((N_ROWS, L), np.int32)
    lib_c[0] = coords
    lib_t[0] = types
    c2, t2 = _molecule2()
    lib_c[1] = c2
    lib_t[1] = t2
    # The coordinates are tiled and the count vector is not -- it is one
    # row per conformer, already coalesced. There is no type plane to
    # mis-stride, so there is no untiled-plane arm.
    return dict(
        query_coords=coords[None].copy(), query_types=types[None].copy(),
        query_n_atoms=np.array([N_REAL], np.int32),
        library_coords=tile32(lib_c),
        library_counts=counts_from_types(
            lib_t, np.full(N_ROWS, N_REAL, np.int32), INTERACTION_R.shape[0]),
        library_color_offset=N_REAL,
        pair_query_row=np.zeros(1, np.int32),
        pair_library_row=np.full(1, library_row, np.int32),
        interaction_r=INTERACTION_R, interaction_p=INTERACTION_P)


#: The canonical diagonal matrix, which is what a
#: PharmacophoreGenerator emits and the only matrix the field path's same-type
#: structure is exact for. The rest of this file deliberately uses a
#: non-canonical one, for the reason its module docstring gives.
CANONICAL_R = np.ones((3, 3), np.float32)
CANONICAL_P = np.diag([0.0, 1.0, 1.0]).astype(np.float32)


def _canonical_arrays(**kw):
    a = _arrays(**kw)
    a["interaction_r"] = CANONICAL_R
    a["interaction_p"] = CANONICAL_P
    return a


def _oracle_color_terms():
    """The float64 color self-overlap of this molecule with itself."""
    coords, types = _molecule()
    n_c = FEATURE_XYZ.shape[0]
    feat = coords[N_REAL:N_REAL + n_c, :3].astype(np.float64)
    ft = types[N_REAL:N_REAL + n_c]
    staged = np.rint(feat.astype(np.float32) / WIRE_QUANT_STEP) * WIRE_QUANT_STEP
    return (R.color_overlap(staged, ft, feat, ft, INTERACTION_R, INTERACTION_P),
            R.color_overlap(staged, ft, staged, ft, INTERACTION_R, INTERACTION_P),
            R.color_overlap(feat, ft, feat, ft, INTERACTION_R, INTERACTION_P))


@needs_gpu
class TestTheColorPathIsReachable:

    def test_score_pairs_reaches_color_at_its_default_arguments(self):
        """The deciding measurement, in one assertion.

        Nothing here names a staging switch. This once raised
        "this switch combination is not instantiated (key 29)".
        """
        v = backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                               mixing=MIXING, steps=0, start_mode=0,
                               **_arrays())
        assert v[0, S.OVERLAP_COLOR] > 0.0
        assert v[0, S.TANIMOTO_COLOR] > 0.0

    def test_the_cross_color_term_agrees_with_the_float64_oracle(self):
        """At the identity pose the pair is the molecule against itself, so the
        cross term must equal the self term the oracle computes."""
        v = backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                               mixing=MIXING, steps=0, start_mode=0,
                               **_arrays())
        want = _oracle_color_terms()[0]
        got = float(v[0, S.OVERLAP_COLOR])
        assert abs(got - want) / want < 1e-6, (got, want)

    def test_the_self_color_columns_agree_too(self):
        """Both self columns, not only the query's: the library's is the one the
        tile layout of the type plane decides."""
        v = backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                               mixing=MIXING, steps=0, start_mode=0,
                               **_arrays())
        for col, want in zip((S.SELF_COLOR_QUERY, S.SELF_COLOR_LIBRARY), _oracle_color_terms()[1:]):
            got = float(v[0, col])
            assert abs(got - want) / want < 1e-6, (col, got, want)

    def test_a_nonzero_mixing_makes_the_combination_differ_from_the_shape_score(self):
        """The first trap: at mixing=0 the combination Tanimoto is exactly the
        shape Tanimoto, and a passing test has measured nothing about color."""
        # Library row 1, not 0: row 0 is the query's own twin, and at the
        # shape-optimal pose a self-match scores exactly 1.0 on BOTH channels,
        # so the combination equals the shape score there whatever the mixing.
        kw = dict(pose_search=PoseSearch.ANALYTIC, color=True, steps=25,
                  **_arrays(library_row=1))
        at_zero = backend.score_pairs(mixing=0.0, **kw)
        at_half = backend.score_pairs(mixing=MIXING, **kw)
        assert at_zero[0, S.TANIMOTO_COMBINATION] == at_zero[0, S.TANIMOTO_SHAPE]
        assert at_half[0, S.TANIMOTO_COMBINATION] != at_half[0, S.TANIMOTO_SHAPE]

    def test_a_screen_config_resolves_to_a_launchable_color_configuration(self):
        """``ScreenConfig`` is the other public route the ticket names. It
        already forced ``stage_width`` to 0 for color; what changed is that 0 is
        now launchable."""
        cfg = ScreenConfig(color=True, mixing=MIXING,
                           pose_search=PoseSearch.ANALYTIC, steps=25)
        # `for_width` resolves `library_color_offset` from the block's
        # SHAPE width, which for this hand-built fixture is where the features
        # start. Naming it here as well would be passing it twice.
        args = cfg.for_width(L, L_shape=N_REAL)
        assert args["stage_width"] == 0
        assert args["library_color_offset"] == N_REAL
        fx = _arrays()
        fx.pop("library_color_offset")
        v = backend.score_pairs(**fx, **args)
        assert v[0, S.OVERLAP_COLOR] > 0.0


@needs_gpu
class TestWhatIsStillRefusedAndWhy:

    def test_an_explicit_stage_width_with_color_is_still_refused(self):
        """Resolving ``stage_width=None`` to 0 for color must not turn into
        silently ignoring a width the caller named."""
        with pytest.raises(RuntimeError, match="staging is shape-only"):
            backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                               mixing=MIXING, stage_width=256, **_arrays())

    def test_a_library_with_no_feature_types_at_all_is_refused(self):
        """The failure mode: it would succeed and report zero for every pair."""
        kw = _arrays()
        kw["library_counts"] = kw["library_counts"].copy()
        kw["library_counts"][:, 1:] = 0
        with pytest.raises(ValueError, match="no pharmacophore feature"):
            backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                               mixing=MIXING, **kw)

    def test_a_color_run_that_overruns_the_plane_is_refused(self):
        """A run boundary that walks off the end of a conformer's record
        reads the NEXT conformer's slots -- another molecule's, in the tiled
        layout -- and nothing about the score says so. It is the one check that
        turns an off-by-one in the count vector from a plausible number into a
        refusal."""
        kw = _arrays()
        kw["library_counts"] = kw["library_counts"].copy()
        kw["library_counts"][0, 1] += L      # more features than the plane holds
        with pytest.raises(ValueError, match="color run reaches slot"):
            backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=MIXING, **kw)

    def test_a_real_atom_count_wider_than_the_plane_is_refused(self):
        """The shape half of the same statement, which `library_shape_width`
        already made: it defaults to `library_counts[:, 0].max()` and is checked
        against the plane's width, so a count vector claiming more real atoms
        than the record holds is refused there."""
        kw = _arrays()
        kw["library_counts"] = kw["library_counts"].copy()
        kw["library_counts"][0, 0] = L + 1
        with pytest.raises(ValueError, match="library_shape_width must cover"):
            backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=MIXING, **kw)

    def test_a_color_launch_without_a_color_offset_is_refused(self):
        """The offset says where a library conformer's color run begins,
        and the plausible default -- the largest real atom count -- is right
        only when every conformer shares one. Getting it wrong reads back as
        +0.0, so it is named rather than guessed."""
        kw = _arrays()
        kw.pop("library_color_offset")
        with pytest.raises(ValueError, match="library_color_offset"):
            backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=MIXING, **kw)

    def test_the_field_path_carries_color_now(self):
        """It used to refuse outright; what it refuses now is narrower.

        A CANONICAL matrix, unlike the rest of this file: the field path
        implements the same-type structure only, and this fixture's matrix is
        deliberately non-canonical (see the module docstring).
        """
        out = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                  mixing=MIXING, **_canonical_arrays())
        assert out[0, S.OVERLAP_COLOR] > 0.0
        assert out[0, S.TANIMOTO_COLOR] > 0.0

    def test_the_field_path_refuses_a_non_canonical_matrix(self):
        """It hoists the per-type interaction constants into a per-block table,
        which is exact for the canonical diagonal matrix and for no other. A
        caller who named no structure is told about the matrix -- the thing they
        chose -- rather than about a structure they never asked for."""
        with pytest.raises(ValueError, match="not canonical"):
            backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                mixing=MIXING, **_arrays())

    def test_the_field_path_refuses_the_register_tier_with_color(self):
        """Measured: 692 B/thread of spill at MINB = 4 before any color
        accumulator exists. The cell is worth measuring against a lowered MINB,
        which is a separate instantiation -- refused by name, not ignored."""
        from roshambo3.backend import CoordResidency
        with pytest.raises(ValueError, match="REGISTER"):
            backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                mixing=MIXING,
                                coord_residency=CoordResidency.REGISTER,
                                **_arrays())


@needs_gpu
class TestTheCountVectorIsLoadBearing:
    """The successor to the type-plane-layout gate.

    The old failure -- a library type plane left in per-conformer order while the
    coordinates were tiled, which handed the color loop type 0 and produced a
    color overlap of exactly +0.0 -- is not expressible any more: there is no
    plane. What CAN still go silently wrong is a wrong count vector, and unlike
    the type plane's failure it is a plausible number rather than a loud zero.

    So the gate is the other way round: perturb the counts and require the
    score to MOVE. A kernel that had quietly stopped reading them -- or that
    read them and recovered the run boundary some other way -- would pass every
    other test in this file and fail here.
    """

    def _score(self, counts, offset=N_REAL):
        kw = _arrays()
        kw["library_counts"] = counts
        kw["library_color_offset"] = offset
        return backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                   mixing=MIXING, steps=0, start_mode=0, **kw)

    def test_a_shifted_count_vector_changes_the_score(self):
        base = _arrays()["library_counts"]
        assert self._score(base)[0, S.OVERLAP_COLOR] > 0.0
        # One fewer feature of the first type it carries: the run boundary moves
        # by one slot, so one library feature is scored as the next type or not
        # at all.
        shifted = base.copy()
        t = int(np.flatnonzero(shifted[0, 1:] > 0)[0]) + 1
        shifted[0, t] -= 1
        assert (self._score(shifted)[0, S.OVERLAP_COLOR]
                != self._score(base)[0, S.OVERLAP_COLOR])

    def test_a_wrong_real_atom_count_changes_the_score(self):
        """Column 0 IS n_real -- the count vector replaced the separate array
        rather than joining it, so a change here has to reach the SHAPE term."""
        base = _arrays()["library_counts"]
        wrong = base.copy()
        wrong[:, 0] = N_REAL - 1
        assert (self._score(wrong)[0, S.OVERLAP_VOLUME]
                != self._score(base)[0, S.OVERLAP_VOLUME])

    def test_a_wrong_color_offset_changes_the_score(self):
        base = _arrays()["library_counts"]
        assert (self._score(base, offset=N_REAL + 1)[0, S.OVERLAP_COLOR]
                != self._score(base, offset=N_REAL)[0, S.OVERLAP_COLOR])

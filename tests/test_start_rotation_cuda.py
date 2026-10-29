"""The grid path's signed-permutation precondition must fail loudly.

The grid kernel conjugates the start rotation out of the pose
(`kernels.cu`, the block comment over `start_quaternion`), and that
reparameterization is exact only because `R^T` is a signed permutation for every
start: Adagrad keeps a per-component cache of `g[k]*g[k]`, and only a signed
permutation permutes that cache along with the gradient. A start rotation that
mixes components makes the grid trajectory and the production trajectory diverge
from the second step onward and returns a converged pose for a different
optimization problem -- with no error, no NaN, and a plausible score.

Until this ticket that precondition was enforced by a source comment. These
tests exercise the guard `optimize_pairs_field` now runs on every launch. They
call the *same* exported body, not a reimplementation of it, so a regression in
the guard fails here.

The traps this file is shaped around:

  * **Vary the start mode.** Every shipped mode is checked, not just the
    default. A defect visible only at `start_mode > 0` has been found
    before, and it is invisible to a test that leaves the default alone.
  * **Trip it on purpose.** A predicate that has never returned False is not
    known to be a predicate. The off-axis cases below are the rotations a new
    start set would plausibly introduce: inertia-eigenvalue frames stand
    in as an arbitrary proper rotation, and the small-angle cases pin down that
    the tolerance is not swallowing real violations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from roshambo3.backend import cuda as backend

pytestmark = pytest.mark.skipif(not backend.available(),
                                reason="no CUDA extension in this build")

#: The shipped modes and their start counts -- `start_mode_n` in kernels.cuh.
START_MODE_N = {0: 1, 1: 4, 2: 10}


def _quat(axis, angle_rad):
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    h = 0.5 * angle_rad
    return (math.cos(h), n[0] * math.sin(h), n[1] * math.sin(h), n[2] * math.sin(h))


class TestShippedStartModes:
    @pytest.mark.parametrize("start_mode", sorted(START_MODE_N))
    def test_every_shipped_mode_passes(self, start_mode):
        """All three modes, not just the default."""
        ext = backend.extension()
        assert ext.check_start_rotations(start_mode) == START_MODE_N[start_mode]

    def test_out_of_range_mode_is_refused(self):
        """`start_mode_n` is __constant__ memory: reading past it does not fault.

        So an unknown mode has to be rejected before it is indexed, and it has to
        be rejected by the same guard, not by a separate range check somewhere
        that a new caller can miss.
        """
        ext = backend.extension()
        for bad in (-1, 3, 99):
            with pytest.raises(Exception) as e:
                ext.check_start_rotations(bad)
            assert "start_mode" in str(e.value)


class TestPredicate:
    """The predicate itself, on rotations chosen to be near and far."""

    @pytest.mark.parametrize("axis,deg", [
        ((0, 0, 1), 0.0),      # identity
        ((1, 0, 0), 180.0),
        ((0, 1, 0), 180.0),
        ((0, 0, 1), 180.0),
        ((1, 0, 0), 90.0),
        ((0, 1, 0), -90.0),
        ((0, 0, 1), 90.0),
    ])
    def test_axis_aligned_quarter_turns_pass(self, axis, deg):
        ext = backend.extension()
        assert ext.rotation_is_signed_permutation(*_quat(axis, math.radians(deg)))

    def test_tetrahedral_three_cycle_passes(self):
        """120 degrees about (1,1,1) is the 3-cycle x->y->z, a permutation matrix.

        This is asserted rather than skipped because the opposite has been
        claimed, and a check written to agree with that claim would be wrong. The octahedral group -- all 24 proper signed
        permutations -- contains the tetrahedral group, so every element of the
        withdrawn offset's coset qualifies. Measured miss from {0, +-1}:
        1.8e-07, i.e. float32 noise. Nothing here
        endorses implementing that offset, which is withdrawn for a different
        and sufficient reason.
        """
        ext = backend.extension()
        assert ext.rotation_is_signed_permutation(
            *_quat((1, 1, 1), math.radians(120.0)))

    @pytest.mark.parametrize("axis,deg", [
        ((0, 0, 1), 45.0),     # the classic off-axis turn
        ((1, 1, 0), 90.0),     # axis-aligned angle, off-axis axis
        ((1, 1, 1), 30.0),
        ((0, 0, 1), 10.0),
        ((0, 0, 1), 1.0),      # 1 degree: still 175x outside the tolerance
    ])
    def test_off_axis_rotations_are_rejected(self, axis, deg):
        ext = backend.extension()
        assert not ext.rotation_is_signed_permutation(
            *_quat(axis, math.radians(deg)))

    def test_arbitrary_proper_rotation_is_rejected(self):
        """Inertia-tensor eigenframe starts are arbitrary rotations.

        Twenty random ones stand in for that whole class. A signed permutation
        has measure zero in SO(3), so every draw must be rejected; a predicate
        that accepted any of these would be accepting that start set.
        """
        ext = backend.extension()
        rng = np.random.default_rng(20260821)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            assert not ext.rotation_is_signed_permutation(*(float(v) for v in q))


class TestGuardIsOnTheLaunchPath:
    """The guard runs at launch, and it did not break a working configuration.

    A launch-time check that rejects the shipped start modes is worse than no
    check. Every mode is screened end to end on the field path, and the start
    modes are varied -- the trap every test here is shaped around.
    """

    def test_field_screen_runs_at_every_start_mode(self, smoke_corpus):
        import roshambo3 as r3
        from roshambo3.engine import Engine, R3gSource, ScreenConfig, TopK, from_arrays
        from roshambo3.backend.cuda import PoseSearch

        cat = r3.Catalog(smoke_corpus)
        gids = np.array([0, 1])
        qs = from_arrays(cat.coordinates(gids), ["q0", "q1"])
        for start_mode in sorted(START_MODE_N):
            res = Engine(device_id=0).run(
                R3gSource(smoke_corpus), qs,
                ScreenConfig(pose_search=PoseSearch.FIELD, start_mode=start_mode),
                TopK(k=5))
            assert res is not None

    def test_field_screen_refuses_an_unknown_start_mode(self, smoke_corpus):
        """The refusal reaches the caller as an error, not as a wrong pose."""
        import roshambo3 as r3
        from roshambo3.engine import Engine, R3gSource, ScreenConfig, TopK, from_arrays
        from roshambo3.backend.cuda import PoseSearch

        cat = r3.Catalog(smoke_corpus)
        qs = from_arrays(cat.coordinates(np.array([0])), ["q0"])
        with pytest.raises(Exception) as e:
            Engine(device_id=0).run(
                R3gSource(smoke_corpus), qs,
                ScreenConfig(pose_search=PoseSearch.FIELD, start_mode=3),
                TopK(k=5))
        assert "start_mode" in str(e.value)

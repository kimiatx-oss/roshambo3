"""The float64 oracle, and the kernel checked against it.

The point of this file is independence. ROSHAMBO2 checked its CUDA kernel against
a C++ backend and a PyTorch backend, and this is what that bought:
"'C++ and CUDA agree, so the math is right' -- they agree because they share the
typo." So the reference here is transcribed from the published expression rather
than from the kernel, and the tests below prefer properties the kernel could not
accidentally satisfy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from roshambo3 import numerics as N
from roshambo3 import reference as R
from roshambo3.backend import cuda as backend

from gpu_device import device_present

needs_gpu = pytest.mark.skipif(not backend.available() or not device_present(),
                               reason="no CUDA extension in this build, or no device")


class TestConstants:
    def test_the_reference_and_the_device_agree_on_the_constants(self):
        """Computed independently -- the reference from kappa and the carbon
        radius in float64, the device values pinned as float32 -- so agreeing to
        float32 precision is evidence, not a tautology."""
        assert np.isclose(R.ALPHA, float(N.A), rtol=1e-7)
        assert np.isclose(R.PAIR_PREFACTOR, float(N.SHAPE_PREFACTOR), rtol=1e-6)
        assert np.isclose(R.COLOR_PREFACTOR, float(N.COLOR_PREFACTOR), rtol=1e-6)

    def test_alpha_is_kappa_over_the_carbon_radius_squared(self):
        assert R.ALPHA == R.KAPPA / (R.CARBON_RADIUS ** 2)


class TestOverlapProperties:
    """Properties of the analytic expression, independent of any implementation."""

    def test_a_single_pair_at_zero_separation_is_the_prefactor(self):
        z = np.zeros((1, 3))
        assert np.isclose(R.pair_overlap(z, z), R.PAIR_PREFACTOR)

    def test_overlap_decays_with_separation(self):
        a = np.zeros((1, 3))
        vals = [R.pair_overlap(a, np.array([[d, 0.0, 0.0]]))
                for d in (0.0, 0.5, 1.0, 2.0, 4.0)]
        assert vals == sorted(vals, reverse=True)

    def test_the_decay_is_the_stated_gaussian(self):
        """V = P exp(-(alpha/2) d^2). Checking the exponent, not just monotonicity."""
        for d in (0.3, 1.1, 2.7):
            got = R.pair_overlap(np.zeros((1, 3)), np.array([[d, 0.0, 0.0]]))
            want = R.PAIR_PREFACTOR * math.exp(-0.5 * R.ALPHA * d * d)
            assert np.isclose(got, want, rtol=1e-12)

    def test_overlap_is_invariant_under_a_shared_rigid_motion(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(9, 3)), rng.normal(size=(11, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        shift = np.array([3.0, -1.0, 2.0])
        assert np.isclose(R.pair_overlap(a, b),
                          R.pair_overlap(a @ q.T + shift, b @ q.T + shift),
                          rtol=1e-12)

    def test_overlap_is_symmetric(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
        assert np.isclose(R.pair_overlap(a, b), R.pair_overlap(b, a), rtol=1e-12)

    def test_self_tanimoto_is_one(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(10, 3))
        v = R.self_overlap(a)
        assert np.isclose(R.tanimoto(v, v, v), 1.0)

    def test_the_min_volume_bound_is_false_for_this_function(self):
        """V_AB <= min(V_AA, V_BB) is the tempting bound and it does not hold:
        there is no inclusion-exclusion in the first-order expression, so a small
        compact molecule inside a larger one has each atom seeing several
        partners. A filter built on it discards real hits."""
        a = np.zeros((1, 3))                       # one atom
        b = np.zeros((6, 3))                       # six coincident atoms
        v_ab = R.pair_overlap(a, b)
        assert v_ab > min(R.self_overlap(a), R.self_overlap(b))

    def test_cauchy_schwarz_holds(self):
        """The provable bound: V_AB <= sqrt(V_AA V_BB), with equality only at
        self-match. All shape Gaussians share one width, so V_AB is an inner
        product in the kernel's RKHS."""
        rng = np.random.default_rng(3)
        for _ in range(40):
            a = rng.normal(scale=2.0, size=(rng.integers(2, 12), 3))
            b = rng.normal(scale=2.0, size=(rng.integers(2, 12), 3))
            v = R.pair_overlap(a, b)
            bound = math.sqrt(R.self_overlap(a) * R.self_overlap(b))
            assert v <= bound * (1 + 1e-12)


class TestQuaternion:
    def test_the_matrix_is_a_proper_rotation_even_for_an_unnormalized_input(self):
        """The defect this oracle exists to catch: a missing normalization made
        the applied transform non-orthogonal, with |q| reaching 1.216 and
        det(M) 1.313, so 1.2% of poses reported a Tanimoto no rigid motion can
        achieve."""
        rng = np.random.default_rng(4)
        for _ in range(50):
            q = rng.normal(size=4) * rng.uniform(0.2, 3.0)
            m = R.quaternion_matrix(q)
            assert np.isclose(np.linalg.det(m), 1.0)
            assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)

    def test_a_degenerate_quaternion_gives_the_identity(self):
        assert np.allclose(R.quaternion_matrix(np.zeros(4)), np.eye(3))

    def test_apply_pose_preserves_all_pairwise_distances(self):
        rng = np.random.default_rng(5)
        b = rng.normal(size=(9, 3))
        q = rng.normal(size=4)
        t = rng.normal(size=3)
        bt = R.apply_pose(b, q, t)
        d0 = np.linalg.norm(b[:, None] - b[None, :], axis=-1)
        d1 = np.linalg.norm(bt[:, None] - bt[None, :], axis=-1)
        assert np.allclose(d0, d1)


@needs_gpu
class TestKernelAgainstTheReference:
    def test_the_device_overlap_matches_the_oracle(self):
        """The check ROSHAMBO2 could not make: the kernel's own overlap routine
        against an implementation that shares no code with it."""
        ext = backend.extension()
        rng = np.random.default_rng(7)
        worst = 0.0
        for _ in range(10):
            nA, nB = int(rng.integers(4, 20)), int(rng.integers(4, 20))
            a = rng.normal(scale=2.0, size=(nA, 3))
            b = rng.normal(scale=2.0, size=(nB, 3))
            A = np.zeros((1, nA, 4), np.float32); A[0, :, :3] = a; A[0, :, 3] = 1
            B = np.zeros((1, nB, 4), np.float32); B[0, :, :3] = b; B[0, :, 3] = 1
            V = np.zeros((1, 1), np.float32)
            ext.test_overlap(A, B, V)
            want = R.pair_overlap(a, b)
            worst = max(worst, abs(float(V[0, 0]) - want) / want)
        # float32 accumulation over a few hundred pair terms
        assert worst < 1e-5, worst

    def test_the_reported_self_overlaps_match_the_oracle(self):
        """V_AA and V_BB are score columns and the Tanimoto denominator, so they
        are worth checking directly rather than only through the ratio."""
        from roshambo3 import scores as S
        from roshambo3.backend import PoseSearch, score_pairs
        from roshambo3.layout import tile32

        rng = np.random.default_rng(8)
        nA, nB, L = 11, 14, 16
        a = rng.normal(scale=2.0, size=(nA, 3)); a -= a.mean(0)
        b = rng.normal(scale=2.0, size=(nB, 3)); b -= b.mean(0)
        qc = np.zeros((1, L, 4), np.float32); qc[0, :nA, :3] = a; qc[0, :nA, 3] = 1
        tb = np.zeros((1, L, 4), np.float32); tb[0, :nB, :3] = b; tb[0, :nB, 3] = 1
        row = score_pairs(
            query_coords=qc, query_types=np.zeros((1, L), np.int32),
            query_n_atoms=np.array([nA], np.int32),
            library_coords=tile32(tb),
            library_counts=np.array([[nB]], np.uint8),
            pair_query_row=np.zeros(1, np.int32),
            pair_library_row=np.zeros(1, np.int32),
            interaction_r=np.ones((1, 1), np.float32),
            interaction_p=np.zeros((1, 1), np.float32),
            pose_search=PoseSearch.ANALYTIC, steps=25, lr_t=0.4, lr_q=0.1,
            start_mode=1)[0]
        assert np.isclose(row[S.SELF_OVERLAP_QUERY], R.self_overlap(np.rint(a.astype(np.float32) / N.WIRE_QUANT_STEP) * N.WIRE_QUANT_STEP), rtol=1e-5)
        assert np.isclose(row[S.SELF_OVERLAP_LIBRARY], R.self_overlap(b), rtol=1e-5)

    def test_the_translation_gradient_is_the_negative_of_a_central_difference(self):
        """The one gradient claim that holds exactly, and it pins two conventions
        at once: the pose applies to the LIBRARY molecule, and the kernel returns
        a DESCENT direction.

        Measured at cos = -1.000000000 with magnitude ratio 1.000000. The rotation
        components are deliberately not checked -- they are a rotation generator,
        not dV/dq_i, so no sign flip reconciles them and asserting on them would
        be asserting something false.
        """
        ext = backend.extension()
        rng = np.random.default_rng(11)
        for _ in range(5):
            nA, nB = 8, 10
            a = rng.normal(scale=1.5, size=(nA, 3))
            b = rng.normal(scale=1.5, size=(nB, 3))
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            t = rng.normal(scale=0.5, size=3)
            A = np.zeros((nA, 4), np.float32); A[:, :3] = a; A[:, 3] = 1
            B = np.zeros((nB, 4), np.float32); B[:, :3] = b; B[:, 3] = 1
            dev = np.asarray(ext.test_gradient(A, B, q.astype(np.float32),
                                               t.astype(np.float32)))[4:]
            ref = R.overlap_gradient(a, b, q, t)[4:]     # pose on the library mol
            assert np.allclose(dev, -ref, rtol=2e-4, atol=1e-6), (dev, -ref)

    def test_the_reported_pose_reproduces_the_reported_score(self):
        """Which molecule the pose acts on, settled by measurement rather than by
        reading: the transform applies to the TARGET, then the start-mode
        rotation. Getting this backwards would write out wrong poses while every
        score stayed right, so it is checked.
        """
        from roshambo3 import scores as S
        from roshambo3.backend import PoseSearch, score_pairs
        from roshambo3.layout import tile32

        rng = np.random.default_rng(9)
        nA, nB, L = 12, 15, 16
        a = rng.normal(scale=2.0, size=(nA, 3)); a -= a.mean(0)
        b = rng.normal(scale=2.0, size=(nB, 3)); b -= b.mean(0)
        qc = np.zeros((1, L, 4), np.float32); qc[0, :nA, :3] = a; qc[0, :nA, 3] = 1
        tb = np.zeros((1, L, 4), np.float32); tb[0, :nB, :3] = b; tb[0, :nB, 3] = 1
        tc = tile32(tb)
        row = score_pairs(
            query_coords=qc, query_types=np.zeros((1, L), np.int32),
            query_n_atoms=np.array([nA], np.int32), library_coords=tc,
            library_counts=np.array([[nB]], np.uint8),
            pair_query_row=np.zeros(1, np.int32),
            pair_library_row=np.zeros(1, np.int32),
            interaction_r=np.ones((1, 1), np.float32),
            interaction_p=np.zeros((1, 1), np.float32),
            pose_search=PoseSearch.ANALYTIC, steps=25, lr_t=0.4, lr_q=0.1,
            start_mode=1)[0]

        a = np.rint(a.astype(np.float32) / N.WIRE_QUANT_STEP) * N.WIRE_QUANT_STEP
        moved = R.apply_pose(R.apply_pose(b, row[S.POSE_QUAT], row[S.POSE_TRANSLATION]),
                             row[S.START_QUAT], np.zeros(3))
        got = R.tanimoto(R.pair_overlap(a, moved), R.self_overlap(a), R.self_overlap(b))
        assert np.isclose(got, float(row[S.TANIMOTO_SHAPE]), atol=1e-4), \
            f"pose gives {got}, kernel reported {row[S.TANIMOTO_SHAPE]}"

        # and the alternatives do not
        wrong = R.tanimoto(
            R.pair_overlap(a, R.apply_pose(b, row[S.POSE_QUAT], row[S.POSE_TRANSLATION])),
            R.self_overlap(a), R.self_overlap(b))
        assert not np.isclose(wrong, float(row[S.TANIMOTO_SHAPE]), atol=1e-4), \
            "the start-mode rotation made no difference, so this test proves nothing"


class TestTheColorTermFollowsTheDeclaredContract:
    """CPU-only, and the check that would have caught this without a GPU.

    `numerics.INV_TWO_ALPHA` states the contract both sides implement:
    ``exp(-(r/2) d^2) == exp2(-r * INV_TWO_ALPHA * d'^2)``. The oracle carried a
    spurious factor of ALPHA in that exponent from the port until it was fixed,
    and the 17 tests in this file did not touch `color_overlap` at all -- the
    oracle's color half was untested, in the module the numerics policy
    designates as the project's second opinion.

    The canonical interaction matrix is why nobody noticed: it couples only
    same-type pairs, so a molecule whose features all have distinct types has
    every cross term at zero width and every surviving term at ``d = 0``, where
    every exponent agrees. These tests use **same-type features at a nonzero
    separation**, which is the configuration that discriminates.
    """

    def test_a_single_pair_matches_the_closed_form(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.3, 0.0, 0.0]])
        t = np.array([0])
        r, p, d2 = 1.7, 0.9, 1.3 ** 2
        want = (float(R.COLOR_PREFACTOR) * math.exp(-0.5 * r * d2)
                * p * p * r ** -1.5)
        got = R.color_overlap(a, t, b, t,
                              np.array([[r]]), np.array([[p]]))
        assert math.isclose(got, want, rel_tol=1e-12), (got, want)

    def test_the_width_is_not_multiplied_by_alpha(self):
        """The specific defect, pinned as its own statement so a future
        refactor cannot reintroduce it silently."""
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        t = np.array([0])
        r = np.array([[1.0]])
        p = np.array([[1.0]])
        got = R.color_overlap(a, t, b, t, r, p)
        with_alpha = (float(R.COLOR_PREFACTOR)
                      * math.exp(-0.5 * 1.0 * float(N.ALPHA) * 1.0))
        assert not math.isclose(got, with_alpha, rel_tol=1e-6), (
            "color_overlap is applying ALPHA to the interaction width again; "
            "the width plays the role ALPHA plays for shape and is not scaled "
            "by it as well")

    def test_a_zero_width_pair_is_skipped_rather_than_dividing_by_zero(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        r = np.array([[0.0, 1.0], [1.0, 0.0]])
        p = np.ones((2, 2))
        assert R.color_overlap(a, np.array([0]), b, np.array([0]), r, p) == 0.0

    def test_it_is_symmetric(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(3, 3)), rng.normal(size=(4, 3))
        ta, tb = np.array([0, 1, 0]), np.array([1, 0, 1, 0])
        r = np.array([[1.0, 0.4], [0.4, 1.6]])
        p = np.array([[1.0, 0.5], [0.5, 0.8]])
        assert math.isclose(R.color_overlap(a, ta, b, tb, r, p),
                            R.color_overlap(b, tb, a, ta, r, p), rel_tol=1e-12)


@needs_gpu
class TestTheKernelColorTermAgainstTheOracle:
    """The comparison the oracle exists to make, on the color channel.

    Reached through the raw extension, with the staging switches named
    explicitly. That was originally a workaround -- the color path could not be
    launched through the public API at all -- and it is now deliberate: the
    public route is tested in `test_color_path_cuda.py`, and this class stays on
    the raw call so that a change to how `score_pairs` resolves its defaults
    cannot move the one comparison the oracle is here to make. Both were true
    for the same fixture at the time of writing.

    Color **self**-overlap rather than cross-overlap, because the self term is
    what this fixture exercises: the library block below is deliberately left
    untiled, so its coordinates are read at the wrong stride and the cross term
    is not the quantity to compare. The cross term against the oracle, with the
    library properly tiled, is in `test_color_path_cuda.py`.

    The self term is fully exercised as long as the features are same-type and
    apart, which is what makes this discriminating: measured, the pre-fix
    oracle was 18% high here while agreeing exactly on every configuration the
    canonical matrix produces.
    """

    def _kernel_self_color(self, color_xyz, types, rmat, pmat, n_real=2):
        ext = backend.extension()
        L = 8
        n_c = color_xyz.shape[0]
        assert n_real + n_c <= L
        # Type 0 is "not a feature" everywhere in the format -- it is the
        # ordinary-atom row of the interaction matrix, the marker
        # `combine_shape_color` writes over padding, and what an untiled type
        # plane degenerates to. The color loop is now BOUNDED by it rather
        # than masked over it, so a fixture that used 0 as a scoring
        # feature type against a 1x1 matrix -- which these tests did, and which
        # no corpus and no query can produce -- would have its "features" read
        # as padding. Promote the matrix by one row and index the features at
        # type 1, which is the same arithmetic on the same numbers: rmat[1,1] is
        # the width these tests pass and the oracle is still called with the 1x1
        # matrix at its own type 0, so the comparison is untouched.
        types = np.asarray(types, np.int32)
        if types.size and int(types.max()) == 0:
            nf = int(np.asarray(rmat).shape[0])
            r2 = np.ones((nf + 1, nf + 1))
            p2 = np.zeros((nf + 1, nf + 1))
            r2[1:, 1:] = rmat
            p2[1:, 1:] = pmat
            rmat, pmat = r2, p2
            types = types + 1
        A = np.zeros((1, L, 4), np.float32)
        B = np.zeros((32, L, 4), np.float32)
        for M in (A, B):
            M[0, :n_real, :3] = [[0, 0, 0], [1.2, 0, 0]][:n_real]
            M[0, :n_real, 3] = 1.0
            M[0, n_real:n_real + n_c, :3] = color_xyz
            M[0, n_real:n_real + n_c, 3] = 1.0
        AT = np.zeros((1, L), np.int32)
        BT = np.zeros((32, L), np.int32)
        AT[0, n_real:n_real + n_c] = types
        BT[0, n_real:n_real + n_c] = types
        # The library takes a per-conformer count vector, not a per-slot
        # type plane. `BT` is still built above because it is what the count
        # vector describes and `counts_from_types` is the inverse of the
        # expansion the format performs -- the plane is the statement of intent
        # and the counts are what travels.
        from roshambo3.layout import counts_from_types
        nf = int(np.asarray(rmat).shape[0])
        BCNT = counts_from_types(BT, np.full(32, n_real, np.int32), nf)
        V = np.zeros((1, 20), np.float32)
        ext.optimize_pairs(
            A, AT, np.array([n_real], np.int32),
            B, BCNT,
            np.asarray(rmat, np.float32), np.asarray(pmat, np.float32),
            np.zeros(1, np.int32), np.zeros(1, np.int32), V,
            True, 0.5, 0.1, 0.4, 0, 0,
            pose_search=1, stage_width=0, stage_int16=False,
            stage_xyz_only=False, library_color_offset=n_real)
        from roshambo3 import scores as S
        return float(V[0, S.SELF_COLOR_QUERY])

    @pytest.mark.parametrize("width", [1.0, 1.3, 2.1])
    @pytest.mark.parametrize("sep", [0.7, 1.3, 2.4])
    def test_same_type_features_apart_agree_with_the_oracle(self, width, sep):
        xyz = np.array([[0.0, 0.0, 0.0], [sep, 0.0, 0.0]])
        t = np.array([0, 0])
        rmat, pmat = np.array([[width]]), np.array([[1.0]])
        got = self._kernel_self_color(xyz, t, rmat, pmat)
        staged = np.rint(xyz.astype(np.float32) / N.WIRE_QUANT_STEP) * N.WIRE_QUANT_STEP
        want = R.color_overlap(staged, t, staged, t, rmat, pmat)
        assert math.isclose(got, want, rel_tol=1e-5), (got, want)

    def test_a_non_unit_height_is_squared_the_same_way_on_both_sides(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        t = np.array([0, 0])
        rmat, pmat = np.array([[1.4]]), np.array([[0.75]])
        got = self._kernel_self_color(xyz, t, rmat, pmat)
        staged = np.rint(xyz.astype(np.float32) / N.WIRE_QUANT_STEP) * N.WIRE_QUANT_STEP
        want = R.color_overlap(staged, t, staged, t, rmat, pmat)
        assert math.isclose(got, want, rel_tol=1e-5), (got, want)

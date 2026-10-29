"""Conformer preparation -- one implementation, used by queries and corpora alike."""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.prep import (GAUSS_ALPHA, make_right_handed, prep_conformer,
                            principal_axis_rotation, self_volume, stable_id)


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestPrincipalAxisRotation:
    def test_is_always_a_proper_rotation(self):
        """det == +1, never -1.

        A left-handed frame applied as ``r @ v.T`` is a reflection, replacing a
        molecule with its mirror image. The optimizer searches only proper
        rotations, so it can never be undone: the hit would be scored as the
        wrong enantiomer. ROSHAMBO2's corpus builder omitted this guarantee
        while its frontend applied it.
        """
        rng = _rng(1)
        for _ in range(200):
            r = rng.normal(size=(rng.integers(2, 40), 3))
            v = principal_axis_rotation(r)
            assert np.isclose(np.linalg.det(v), 1.0), np.linalg.det(v)
            assert np.allclose(v @ v.T, np.eye(3), atol=1e-10)

    def test_make_right_handed_flips_only_the_least_significant_axis(self):
        v = np.eye(3)
        v[2] = -v[2]                       # det = -1
        assert np.linalg.det(v) < 0
        w = make_right_handed(v)
        assert np.isclose(np.linalg.det(w), 1.0)
        assert np.array_equal(w[:2], v[:2])   # leading axes untouched

    def test_make_right_handed_is_a_no_op_when_already_proper(self):
        v = np.eye(3)
        assert make_right_handed(v) is v or np.array_equal(make_right_handed(v), v)

    @pytest.mark.parametrize("coords", [
        np.zeros((0, 3)),                       # empty
        np.zeros((1, 3)),                       # a single atom has no axes
        np.array([[np.nan, 0, 0], [1, 2, 3]]),  # non-finite
        np.array([[np.inf, 0, 0], [1, 2, 3]]),
    ])
    def test_degenerate_geometry_falls_back_to_identity(self, coords):
        """One bad molecule must never abort a batch."""
        assert np.array_equal(principal_axis_rotation(coords), np.eye(3))

    def test_collinear_atoms_do_not_raise(self):
        r = np.stack([np.arange(6.0)] * 3, axis=1)   # perfectly collinear
        v = principal_axis_rotation(r)
        assert np.all(np.isfinite(v))
        assert np.isclose(abs(np.linalg.det(v)), 1.0)


class TestPrepConformer:
    def test_centers_on_the_centroid(self):
        r = _rng(2).normal(size=(20, 3)) + 100.0
        assert np.allclose(prep_conformer(r).mean(axis=0), 0.0, atol=1e-10)

    def test_is_invariant_to_a_prior_rigid_motion(self):
        """Preparation defines a canonical frame, so translating and rotating
        the input must not change the output (up to sign conventions on the
        axes, which is why this compares the Gram matrix)."""
        r = _rng(3).normal(size=(25, 3))
        q, _ = np.linalg.qr(_rng(4).normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        moved = r @ q.T + np.array([7.0, -3.0, 11.0])
        a, b = prep_conformer(r), prep_conformer(moved)
        assert np.allclose(a @ a.T, b @ b.T, atol=1e-8)

    def test_align_ref_places_a_second_block_in_the_first_block_frame(self):
        """Color pseudo-atoms live in the real atoms' frame, never their own."""
        real = _rng(5).normal(size=(20, 3))
        color = _rng(6).normal(size=(4, 3))
        v = principal_axis_rotation(real)
        expected = (color - real.mean(axis=0)) @ v.T
        assert np.allclose(prep_conformer(color, align_ref=real), expected)


class TestSelfVolume:
    def test_is_invariant_under_rigid_motion(self):
        r = _rng(7).normal(size=(18, 3)) * 2
        q, _ = np.linalg.qr(_rng(8).normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        assert np.isclose(self_volume(r), self_volume(r @ q.T + 5.0), rtol=1e-12)

    def test_single_atom_is_the_bare_prefactor(self):
        from roshambo3.prep import GAUSS_CONST
        assert np.isclose(self_volume(np.zeros((1, 3))), 8.0 * GAUSS_CONST)

    def test_grows_with_atom_count(self):
        rng = _rng(9)
        vols = [self_volume(rng.normal(size=(n, 3))) for n in (5, 10, 20, 40)]
        assert vols == sorted(vols)

    def test_two_coincident_atoms_score_as_four_pair_terms(self):
        """V_AA sums over ALL ordered pairs, so two atoms at the same point give
        4x a single atom, not 2x. This is the convention the vaa sidecar and the
        Tanimoto denominator both assume."""
        from roshambo3.prep import GAUSS_CONST
        assert np.isclose(self_volume(np.zeros((2, 3))), 4 * 8.0 * GAUSS_CONST)


class TestStableId:
    def test_is_deterministic_and_non_negative(self):
        for s in ("compound-example", "vendor-example", ""):
            a, b = stable_id(s), stable_id(s)
            assert a == b and 0 <= a < 2 ** 63

    def test_distinguishes_ids_that_a_underscore_split_would_merge(self):
        """Vendor ids (Enamine-style) legitimately end in _<digits>; recovering a molecule id
        by rsplit('_', 1) silently merges distinct molecules."""
        a, b = "s_1____11111____222222____333333", "s_1____11111____222222____333334"
        assert a.rsplit("_", 1)[0] == b.rsplit("_", 1)[0]   # the trap
        assert stable_id(a) != stable_id(b)                 # the fix


def test_unkekulizable_molecule_is_rejected_not_fatal(tmp_path):
    """One bad molecule must not abort a build.

    `Chem.RemoveHs` raises `KekulizeException` rather than returning None for a
    molecule carrying aromatic flags that cannot be kekulized, and that
    exception used to escape `_scatter_unit` through the process pool and kill
    `prepare` outright -- a 10M build died ~25 minutes in on a single ring,
    discarding every shard already written.

    The exception is also a `ValueError`, which is why the guard catches
    RDKit's `MolSanitizeException` specifically: a `except ValueError: raise`
    clause meant to protect our own layout invariants would re-raise it.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import MolSanitizeException

    from roshambo3.prep.sources import _group_by_name

    good = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CO"))
    AllChem.EmbedMolecule(good, randomSeed=0xF00D)
    good.SetProp("_Name", "good")

    # Pyrrole embedded normally, then stripped of the H that makes its ring
    # kekulizable -- so it carries a real conformer and still refuses. Built in
    # this order because the embedder itself kekulizes, and a molecule bad
    # enough to fail RemoveHs cannot be embedded in the first place.
    bad = Chem.AddHs(Chem.MolFromSmiles("c1cc[nH]c1"))
    AllChem.EmbedMolecule(bad, randomSeed=0xF00D)
    n = next(a for a in bad.GetAtoms() if a.GetSymbol() == "N")
    n.SetNumExplicitHs(0)
    n.SetNoImplicit(True)
    h = next(nb.GetIdx() for nb in n.GetNeighbors() if nb.GetSymbol() == "H")
    ed = Chem.RWMol(bad)
    ed.RemoveAtom(h)
    bad = ed.GetMol()
    bad.SetProp("_Name", "bad")
    with pytest.raises(MolSanitizeException):
        Chem.RemoveHs(bad)          # the premise of this test, stated as a check

    failures = []
    recs = list(_group_by_name([bad, good], 0, None, failures))

    assert [r.ext_id for r in recs] == ["good"], "the good molecule must survive"
    assert len(failures) == 1 and failures[0][0] == "bad"
    assert "Kekulize" in failures[0][1]

"""The shipped pharmacophore numbering and emission order."""
from __future__ import annotations
import numpy as np
import pytest

EXPECTED_ORDER = ("Donor", "Acceptor", "PosIonizable", "NegIonizable",
                  "Aromatic", "Hydrophobe")

#: THE SHIPPED order, written out for the same reason.
WORK_DESCENDING = ("Hydrophobe", "Acceptor", "Aromatic", "Donor",
                   "PosIonizable", "NegIonizable")


def _generator():
    rdkit = pytest.importorskip("rdkit")
    del rdkit
    from roshambo3.pharmacophore import RDKitPharmacophoreGenerator
    return RDKitPharmacophoreGenerator()


class TestTheShippedNumberingIsWorkDescending:

    def test_the_shipped_order_is_the_work_descending_one(self):
        """The shipped reorder: 1.0226x on sm_70, 1.0315x on sm_120, above
        METRICS section 8's floor on both, and free in retrieval."""
        g = _generator()
        got = tuple(k for k, _ in sorted(g.FEATURES_ENUM.items(),
                                         key=lambda kv: kv[1]))
        assert got == WORK_DESCENDING, (
            "the color plane order is a shared-memory staging decision. Moving "
            "it refuses every corpus that does not record this numbering, "
            "which is the point -- see the comment at pharmacophore.FEATURES.")
        assert tuple(sorted(g.FEATURES_ENUM.values())) == (1, 2, 3, 4, 5, 6)

    def test_descriptor_records_the_generator_numbering(self, tmp_path):
        from tests.codec_fixtures import make_codec_fixture
        fx = make_codec_fixture(tmp_path / "color", color_layout="merged")
        assert tuple(fx.corpus.color["types"]) == WORK_DESCENDING

    def test_the_historical_order_is_named_and_not_installed(self):
        """The named historical order differs from the installed generator order."""
        from roshambo3.pharmacophore import (HISTORICAL_FEATURE_ORDER,
                                             WORK_DESCENDING_FEATURE_ORDER)
        assert tuple(n for n, _ in
                     WORK_DESCENDING_FEATURE_ORDER) == WORK_DESCENDING
        assert WORK_DESCENDING_FEATURE_ORDER != HISTORICAL_FEATURE_ORDER
        assert set(n for n, _ in WORK_DESCENDING_FEATURE_ORDER) == \
            set(n for n, _ in HISTORICAL_FEATURE_ORDER)
        g = _generator()
        assert tuple(g.FEATURES) == WORK_DESCENDING

    def test_the_interaction_matrix_is_still_canonical(self):
        """A diagonal matrix is its own permutation -- which is exactly why
        this check cannot catch a numbering mismatch, and why the corpus has to
        record the numbering instead."""
        g = _generator()
        assert g.interaction_matrix_report.canonical

    def test_emission_is_still_grouped_and_ascending(self):
        """`layout.py`'s invariant, which holds BY CONSTRUCTION because the
        order is the literal's and there is no permutation layer on top."""
        Chem = pytest.importorskip("rdkit.Chem")
        from rdkit.Chem import AllChem
        g = _generator()
        for smi in ("NCCCCC(N)C(=O)O", "c1ccccc1CCN",
                    "CC(C)Cc1ccc(cc1)C(C)C(=O)O"):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(mol, randomSeed=20260827)
            _, types = g.generate_color_atoms(mol)
            assert np.all(np.diff(types) >= 0), f"{smi}: {types.tolist()}"

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.prep.sources import _read_exact


class _ShortReader:
    def __init__(self, data: bytes, width: int):
        self.data, self.width = data, width

    def read(self, n: int) -> bytes:
        take = min(n, self.width)
        out, self.data = self.data[:take], self.data[take:]
        return out


def test_exact_read_handles_short_stream_reads():
    assert _read_exact(_ShortReader(b"abcdef", 2), 6, record=4, what="payload") == b"abcdef"


def test_exact_read_reports_a_truncated_record():
    with pytest.raises(ValueError, match="record 4"):
        _read_exact(_ShortReader(b"abc", 2), 4, record=4, what="payload")


def test_binary_source_exposes_parallel_payloads(tmp_path):
    from roshambo3.prep.sources import RdkitBinaryZstdSource

    p = tmp_path / "empty.bin.zst"
    p.write_bytes(b"")
    source = RdkitBinaryZstdSource(str(p))
    assert source.parallel_payloads is True
    assert source.units() == [(str(p), None)]


def test_binary_source_reads_embedded_conformers(tmp_path):
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    zstandard = pytest.importorskip("zstandard")
    import struct

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=7)
    blob = mol.ToBinary()
    p = tmp_path / "mols.bin.zst"
    p.write_bytes(zstandard.ZstdCompressor().compress(struct.pack("<I", len(blob)) + blob))

    from roshambo3.prep.sources import RdkitBinaryZstdSource
    source = RdkitBinaryZstdSource(str(p))
    recs = list(source.iter_unit(source.units()[0]))
    assert len(recs) == 1
    # Mol.ToBinary does not include ordinary molecule properties by default,
    # hence the stable canonical-SMILES fallback is the binary record id.
    assert recs[0].ext_id == Chem.MolToSmiles(mol)
    assert len(recs[0].conformers) == 2
    assert source.stats["source_compressed_bytes_read"] > 0


def test_default_feature_plan_matches_rdkit_for_each_conformer():
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    from roshambo3.pharmacophore import RDKitPharmacophoreGenerator

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1OCCN"))
    assert AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=7)
    generator = RDKitPharmacophoreGenerator()
    plan = generator.make_feature_plan(mol)
    assert plan is not None
    for conf in mol.GetConformers():
        one = Chem.Mol(mol)
        one.RemoveAllConformers()
        one.AddConformer(conf, assignId=True)
        expected_coords, expected_types = generator.generate_color_atoms(
            one)
        # The plan carries atoms only; format 3 places every feature at the
        # equal-weight mean of its atoms (FORMAT.md 8.3). BaseFeatures'
        # selected families declare equal weights, so RDKit's own positions
        # are exactly those means (in float; the corpus derives them from
        # the stored int16 atoms).
        positions = one.GetConformer().GetPositions()
        actual_coords = np.asarray([positions[list(a)].mean(axis=0)
                                    for a in plan.atom_ids], dtype=np.float32)
        assert plan.types.tolist() == expected_types.tolist()
        assert actual_coords == pytest.approx(expected_coords, abs=1e-6)


def test_explicit_fdef_has_the_same_plan():
    """Every fdef has a feature plan (format 3 stores only atoms;
    unequal weights are refused at build, ``definitions.unequal_weights``):
    naming the default file explicitly plans exactly as the default does."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit import RDConfig
    from roshambo3.pharmacophore import RDKitPharmacophoreGenerator

    generator = RDKitPharmacophoreGenerator(
        fdefName=str(RDConfig.RDDataDir + "/BaseFeatures.fdef"))
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1OCCN"))
    plan = generator.make_feature_plan(mol)
    default = RDKitPharmacophoreGenerator().make_feature_plan(mol)
    assert plan.atom_ids == default.atom_ids and plan.atom_ids
    assert plan.types.tolist() == default.types.tolist()

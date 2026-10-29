"""Color features are perceived on the stored SMILES graph (FORMAT.md 8.2),
and the SMILES round trip compares radical electrons.

The builder used to perceive on the input molecule as read. An SDF record is
sanitized on reading, so it carries RDKit's default aromaticity; a binary
record (``Chem.Mol(pickle)``) is not, and keeps whatever aromaticity model it
was written with. ``deep`` and query featurization perceive on
``AddHs(MolFromSmiles(stored SMILES))`` -- default aromaticity -- so a corpus
built from MMFF-aromatic pickles stored tables that ``deep`` refused
(``layout.features``). The builder now perceives on that same molecule.

All chemistry here is public and synthetic: 2-pyridone, which MMFF94 does not
call aromatic and RDKit's default model does, and small radicals.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3 import definitions
from roshambo3.format.r3g import reference as R
from roshambo3.format.r3g.validate import validate_corpus
from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep import prepare, sources
from tests.format3_rows import read_corpus

OPTIONS = dict(workers=1, records_per_unit=3, max_conf_per_group=16)
PYRIDONE = "O=c1cccc[nH]1"


def _mmff_aromatic(smiles: str, name: str, conformers: int = 2, seed: int = 7):
    """``smiles`` embedded with explicit hydrogens, then re-flagged under the
    MMFF94 aromaticity model and round-tripped through a pickle, as a binary
    input record carries it (``Chem.Mol(blob)`` does not re-sanitize)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert len(AllChem.EmbedMultipleConfs(mol, numConfs=conformers,
                                          randomSeed=seed)) == conformers
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MMFF94)
    mol.SetProp("_Name", name)
    return Chem.Mol(mol.ToBinary(Chem.PropertyPickleOptions.AllProps))


def _plain(smiles: str, name: str, conformers: int = 2, seed: int = 3):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert len(AllChem.EmbedMultipleConfs(mol, numConfs=conformers,
                                          randomSeed=seed)) == conformers
    mol.SetProp("_Name", name)
    return mol


def _write_binary(path: Path, mols) -> str:
    blobs = [m.ToBinary(Chem.PropertyPickleOptions.AllProps) for m in mols]
    path.write_bytes(zstd.ZstdCompressor().compress(
        b"".join(struct.pack("<I", len(b)) + b for b in blobs)))
    return str(path)


def _input_molecule_table(gen, mol):
    """The table the builder used to store: perceived on ``AddHs`` of
    the input molecule as read, remapped H-bearing -> heavy -> slot."""
    canonical = sources.canonical_heavy(mol)
    mh = Chem.AddHs(mol)
    plan = gen.make_feature_plan(mh)
    is_h = np.array([a.GetAtomicNum() == 1 for a in mh.GetAtoms()])
    heavy_index = np.cumsum(~is_h) - 1
    order = np.asarray(canonical[2])
    slot_of = np.empty(order.size, dtype=np.int64)
    slot_of[order] = np.arange(order.size)
    return R.canonical_feature_order(
        [(int(t), tuple(sorted(int(x) for x in slot_of[heavy_index[list(a)]])))
         for a, t in zip(plan.atom_ids, np.asarray(plan.types).tolist())])


def _deep_table(color, smiles):
    """What ``deep`` re-perceives: the stored SMILES, hydrogens added."""
    got = definitions.perceive(color, [smiles], check=False)[0]
    assert got.parsed
    return R.canonical_feature_order(got.features)


def test_the_premise_mmff_aromaticity_changes_the_table():
    """If this fails, RDKit or BaseFeatures changed and the build tests
    below need another molecule of the mismatch class."""
    gen = PharmacophoreGenerator()
    mol = _mmff_aromatic(PYRIDONE, "pyridone")
    assert not any(a.GetIsAromatic() for a in mol.GetAtoms())   # MMFF94: not aromatic
    smiles = sources.canonical_heavy(mol)[1]
    assert smiles == PYRIDONE                                   # stored with default aromaticity
    old = _input_molecule_table(gen, mol)
    deep = _deep_table(gen.descriptor_color(), smiles)
    assert old != deep and len(old) != len(deep)


@pytest.mark.parametrize("layout", ["one_record", "per_conformer"])
def test_a_pickled_mmff_record_stores_the_table_deep_perceives(tmp_path, layout):
    """Binary (pickled) input: the stored table equals ``deep``'s
    re-perception of the stored SMILES, and ``deep`` over every row passes.
    ``one_record`` carries both conformers in one Mol (the multi-conformer
    branch); ``per_conformer`` is an in-memory source with one record per
    conformer (the per-record branch)."""
    gen = PharmacophoreGenerator()
    mols = [_mmff_aromatic(PYRIDONE, "pyridone"), _plain("CCOc1ccccc1N", "anisidine")]
    out = tmp_path / "corpus"
    if layout == "one_record":
        prepare(_write_binary(tmp_path / "in.bin.zst", mols), str(out),
                color_generator=gen, **OPTIONS)
    else:
        records = []
        for m in mols:
            for c in m.GetConformers():
                one = Chem.Mol(m)
                one.RemoveAllConformers()
                one.AddConformer(Chem.Conformer(c), assignId=True)
                records.append(one)
        assert not any(a.GetIsAromatic() for a in records[0].GetAtoms())
        prepare(records, str(out), color_generator=gen, **OPTIONS)
    corpus = read_corpus(out, merged=False)
    color = corpus.descriptor["color"]
    smiles = dict(zip(np.asarray(corpus.molecules["mol_id"]).tolist(),
                      corpus.molecules["smiles"]))
    seen = 0
    for g in corpus.groups:
        for m, s0 in enumerate(g.run_start[:-1]):
            mol_id = int(g.mol_id[s0])
            stored = R.molecule_features(g.feature_table, m)
            assert stored == _deep_table(color, smiles[mol_id])
            if mol_id == sources.stable_id("pyridone"):
                assert stored != _input_molecule_table(gen, mols[0])
            seen += 1
    assert seen == 2
    validate_corpus(str(out), "deep", all_rows=True)


def test_perception_runs_once_per_molecule_after_the_round_trip_check(monkeypatch):
    """One perception per molecule however many records it has, on the
    parsed SMILES (no coordinates, atom ``i`` is slot ``i``); a molecule
    that fails the round-trip check is rejected and never perceived."""
    from tests.test_prepare_smiles_roundtrip import silicon
    gen = PharmacophoreGenerator()
    seen = []
    original = gen.make_feature_plan

    def counting(mol):
        seen.append((Chem.MolToSmiles(Chem.RemoveHs(mol)), mol.GetNumConformers()))
        return original(mol)

    monkeypatch.setattr(gen, "make_feature_plan", counting)
    records = []
    for c in _plain("CCOc1ccccc1N", "anisidine", conformers=3).GetConformers():
        one = _plain("CCOc1ccccc1N", "anisidine", conformers=1)
        one.RemoveAllConformers()
        one.AddConformer(Chem.Conformer(c), assignId=True)
        records.append(one)
    failures = []
    recs = list(sources._group_by_name(records + [silicon()], 0, gen, failures))
    assert [r.ext_id for r in recs] == ["anisidine"]
    assert len(recs[0].conformers) == 3
    assert [n for n, _ in failures] == ["silicon"]
    assert len(seen) == 1 and seen[0] == (recs[0].meta["smiles"], 0)


# -- radical electrons in the SMILES round trip -------------------------------

def _with_coordinates(mol, name):
    """``mol`` with one arbitrary non-degenerate 3D conformer (heavy atoms
    only; the builder reads nothing else) and a name."""
    rng = np.random.default_rng(5)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(rng.normal(scale=1.5, size=(mol.GetNumAtoms(), 3))):
        conf.SetAtomPosition(i, xyz.tolist())
    mol.AddConformer(conf, assignId=True)
    mol.SetProp("_Name", name)
    return Chem.Mol(mol.ToBinary(Chem.PropertyPickleOptions.AllProps))


def _thiyl_with_two_radicals():
    """CH3-SH with two radical electrons on sulfur: element, charge, total
    H and bonds survive the SMILES (``C[SH]``), the radicals do not -- read
    back, the sulfur has none."""
    rw = Chem.RWMol()
    s = rw.AddAtom(Chem.Atom("S"))
    c = rw.AddAtom(Chem.Atom("C"))
    rw.AddBond(s, c, Chem.BondType.SINGLE)
    atom = rw.GetAtomWithIdx(s)
    atom.SetNumRadicalElectrons(2)
    atom.SetNumExplicitHs(1)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return _with_coordinates(mol, "thiyl")


def _round_tripping_radicals():
    """Radicals the SMILES carries: ethyl (``[CH2]C``) and phenoxyl."""
    return [_with_coordinates(Chem.MolFromSmiles("[CH2]CO"), "ethyl"),
            _with_coordinates(Chem.MolFromSmiles("[O]c1ccc(C)cc1"), "phenoxyl")]


def test_a_lost_radical_fails_the_round_trip_check_by_name():
    mol = _thiyl_with_two_radicals()
    heavy, smiles, _ = sources.canonical_heavy(mol)
    assert [a.GetNumRadicalElectrons() for a in heavy.GetAtoms()] == [0, 2]
    parsed = Chem.MolFromSmiles(smiles)
    # The premise: everything else the check compares agrees.
    assert [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetTotalNumHs())
            for a in parsed.GetAtoms()] == \
        [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetTotalNumHs())
         for a in heavy.GetAtoms()]
    why = sources.smiles_slot_mismatch(smiles, heavy)
    assert why == (f"SMILES round-trip check: its canonical SMILES {smiles!r} does not "
                   f"reproduce the stored slots (slot 1: radical electrons 0 vs stored 2); "
                   f"slot i must be atom i of MolFromSmiles(smiles), element, charge, "
                   f"hydrogens, radical electrons and bonds (FORMAT.md 11.1)")
    for mol in _round_tripping_radicals():
        heavy, smiles, _ = sources.canonical_heavy(mol)
        assert any(a.GetNumRadicalElectrons() for a in heavy.GetAtoms())
        parsed, why = sources.parse_stored_smiles(smiles, heavy)
        assert why is None
        assert [a.GetNumRadicalElectrons() for a in parsed.GetAtoms()] == \
            [a.GetNumRadicalElectrons() for a in heavy.GetAtoms()]


@pytest.mark.parametrize("color", [False, True], ids=["shape", "color"])
def test_a_build_rejects_the_lost_radical_and_keeps_the_others(tmp_path, color):
    """Per molecule, in ``build.json``'s rejected list with the radical
    reason, as every round-trip failure is; the round-tripping radicals are
    stored."""
    mols = [_thiyl_with_two_radicals(), *_round_tripping_radicals()]
    out = tmp_path / "corpus"
    prepare(_write_binary(tmp_path / "in.bin.zst", mols), str(out),
            color_generator=PharmacophoreGenerator() if color else None, **OPTIONS)
    rejected = json.loads((out / "build.json").read_text())["rejected"]
    assert rejected["n"] == 1
    (entry,) = rejected["examples"]
    assert entry["id"] == "thiyl"
    assert "radical electrons 0 vs stored 2" in entry["why"]
    corpus = read_corpus(out, merged=False)
    assert sorted(corpus.molecules["ext_id"]) == ["ethyl", "phenoxyl"]
    validate_corpus(str(out), "deep", all_rows=True)

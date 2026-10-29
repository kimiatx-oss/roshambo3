"""A molecule whose canonical SMILES does not parse back to its stored slots
is rejected per molecule, not refused for the build (FORMAT.md 11.1).

The known case is PubChem 12157944: RDKit writes its canonical SMILES as
``Cn1ccn(C)[si+]1C`` (aromatic charged silicon) and cannot parse that string
back ("Can't kekulize mol"). A 10M-conformer PubChem library build stopped on it after
about 1.68M molecules. ``silicon()`` reproduces it: a Kekule-form record with
no aromatic flags, as a PubChem3D binary carries it, so the builder's own
sanitization perceives the aromatic ring and writes the unparsable SMILES.

An SDF record of the same molecule never reaches the SMILES check: the SDF
reader perceives aromaticity on reading, and the builder's ``RemoveAllHs``
then fails to kekulize it, which was already a per-conformer reject. That
path is covered too, so every source rejects the molecule and none refuses.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import zstandard as zstd
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3.prep import build, prepare
from roshambo3.prep import sources
from roshambo3.prep.conformer import stable_id
from roshambo3.prep.sources import (BuildRefused, canonical_heavy,
                                    check_smiles_slots, smiles_slot_mismatch)
from tests.format3_rows import read_corpus
from tests.prep_fixtures import molecule

SILICON_SMILES = "Cn1ccn(C)[si+]1C"
OPTIONS = dict(workers=1, records_per_unit=3, max_conf_per_group=16)


def silicon(name="silicon", conformers=2, seed=13):
    """The PubChem 12157944 graph in Kekule form, embedded without
    aromaticity perception (an aromatic [si+] ring cannot be embedded)."""
    mol = Chem.MolFromSmiles("C[Si+]1N(C)C=CN1C", sanitize=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                     ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    mol = Chem.AddHs(mol)
    assert len(AllChem.EmbedMultipleConfs(mol, numConfs=conformers,
                                          randomSeed=seed)) == conformers
    mol.SetProp("_Name", name)
    return mol


def normal():
    return [molecule("CCO", "ethanol"), molecule("c1ccccc1O", "phenol", conformers=2),
            molecule("CCCN", "amine", conformers=3)]


def _library():
    """Normal molecules on both sides of the silicon one, so it sits inside
    a work unit with neighbors that must survive."""
    first, *rest = normal()
    return [first, silicon(), *rest]


def _write_binary(path: Path, mols) -> Path:
    blobs = [m.ToBinary(Chem.PropertyPickleOptions.AllProps) for m in mols]
    path.write_bytes(zstd.ZstdCompressor().compress(
        b"".join(struct.pack("<I", len(b)) + b for b in blobs)))
    return path


def _write_sdf(path: Path, mols) -> Path:
    with Chem.SDWriter(str(path)) as writer:
        for mol in mols:
            for conf in mol.GetConformers():
                writer.write(mol, confId=conf.GetId())
    return path


def _write_sqlite(path: Path, mols) -> tuple[Path, dict]:
    """The PubChem3D layout (``mols`` only): integer ids, a property-free
    binary of every conformer. Returns the path and name -> id."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE mols (molecule_id INTEGER PRIMARY KEY, "
                "molbinary_zst BLOB NOT NULL)")
    c = zstd.ZstdCompressor()
    ids = {}
    for i, mol in enumerate(mols):
        mid = 12157944 if mol.GetProp("_Name") == "silicon" else 1000 + i
        ids[mol.GetProp("_Name")] = str(mid)
        con.execute("INSERT INTO mols VALUES (?, ?)", (mid, c.compress(mol.ToBinary())))
    con.commit()
    con.close()
    return path, ids


def _input(tmp_path, kind, mols):
    """``(inputs, names)``: what ``prepare`` takes, and molecule name -> the
    identifier the corpus records for it."""
    names = {m.GetProp("_Name"): m.GetProp("_Name") for m in mols}
    if kind == "rdkit":
        return list(mols), names
    if kind == "binary":
        return str(_write_binary(tmp_path / "input.bin.zst", mols)), names
    if kind == "sdf":
        return str(_write_sdf(tmp_path / "input.sdf", mols)), names
    path, ids = _write_sqlite(tmp_path / "input.sqlite", mols)
    return [str(path)], ids


def _build_json(corpus) -> dict:
    return json.loads((Path(corpus.root) / "build.json").read_text())


# -- the check itself ---------------------------------------------------------

def test_the_known_case_writes_a_smiles_rdkit_cannot_read():
    """The premise of every test below: this RDKit writes the canonical
    SMILES and cannot parse it. If a later RDKit reads it, this fails first
    and the build tests need another non-round-tripping molecule."""
    heavy, smiles, _ = canonical_heavy(silicon())
    assert smiles == SILICON_SMILES
    assert Chem.MolFromSmiles(smiles) is None


def test_mismatch_reasons_name_the_check_and_the_smiles():
    heavy, smiles, _ = canonical_heavy(molecule("CC(=O)[O-]", "acetate"))
    assert smiles_slot_mismatch(smiles, heavy) is None
    why = smiles_slot_mismatch("CC(=O)O", heavy)
    assert why.startswith("SMILES round-trip check: its canonical SMILES 'CC(=O)O' "
                          "does not reproduce the stored slots (slot ")
    heavy, smiles, _ = canonical_heavy(silicon())
    why = smiles_slot_mismatch(smiles, heavy)
    assert why == (f"SMILES round-trip check: its canonical SMILES {SILICON_SMILES!r} "
                   f"does not parse; slot i must be atom i of MolFromSmiles(smiles) "
                   f"(FORMAT.md 11.1)")
    # The raising form, kept for the census and spot-check scripts.
    with pytest.raises(BuildRefused, match=r"molecule 'silicon': SMILES round-trip check"):
        check_smiles_slots("silicon", smiles, heavy)


def test_record_merging_rejects_the_molecule_and_keeps_its_neighbors():
    """``_group_by_name``, which every source goes through: no record for
    the molecule, one ``failures`` entry naming it, its neighbors intact, and
    no conformer of it counted as dropped (the molecule is rejected whole,
    as ``_why_reject`` rejects at layout)."""
    failures, dropped = [], {}
    mols = _library()
    recs = list(sources._group_by_name(mols, 0, None, failures, dropped=dropped))
    assert [r.ext_id for r in recs] == ["ethanol", "phenol", "amine"]
    assert [len(r.conformers) for r in recs] == [1, 2, 3]
    assert len(failures) == 1
    name, why = failures[0]
    assert name == "silicon"
    assert "SMILES round-trip check" in why and SILICON_SMILES in why
    assert dropped == {}


def test_a_multi_conformer_binary_record_is_rejected_whole():
    """A PubChem3D record carries every conformer in one Mol (the
    multi-conformer branch): none of them is emitted."""
    failures, dropped = [], {}
    recs = list(sources._group_by_name([silicon(conformers=3)], 0, None, failures,
                                       contiguous_conformer_ids=True, dropped=dropped))
    assert recs == []
    assert [n for n, _ in failures] == ["silicon"] and dropped == {}


# -- the builder end to end ---------------------------------------------------

@pytest.mark.parametrize("kind", ["rdkit", "binary", "sqlite", "sdf"])
@pytest.mark.parametrize("color", [False, True], ids=["shape", "color"])
def test_a_build_with_the_molecule_completes_without_it(tmp_path, kind, color):
    from roshambo3.pharmacophore import PharmacophoreGenerator
    mols = _library()
    inputs, names = _input(tmp_path, kind, mols)
    corpus = prepare(inputs, str(tmp_path / "corpus"),
                     color_generator=PharmacophoreGenerator() if color else None,
                     **OPTIONS)
    rejected = _build_json(corpus)["rejected"]
    reasons = [e["why"] for e in rejected["examples"] if e["id"] == names["silicon"]]
    assert reasons, rejected
    assert {e["id"] for e in rejected["examples"]} == {names["silicon"]}
    if kind == "sdf":
        # Rejected before the SMILES check (see the module docstring), once
        # per conformer, as an unkekulizable record always was.
        assert all("KekulizeException" in w for w in reasons) and len(reasons) == 2
    else:
        assert reasons == [smiles_slot_mismatch(SILICON_SMILES, canonical_heavy(silicon())[0])]
    meta = pq.read_table(Path(corpus.root) / "molecules.parquet").to_pydict()
    assert sorted(meta["ext_id"]) == sorted(names[n] for n in ("ethanol", "phenol", "amine"))
    assert SILICON_SMILES not in meta["smiles"]
    rows = read_corpus(corpus.root, merged=False)
    assert rows.n_mol == 3 and rows.n_conf == 1 + 2 + 3
    assert stable_id(names["silicon"]) not in set(rows.mol_id.tolist())


# -- the id gates -------------------------------------------------------------

def _ids(names) -> str:
    ids = np.array(sorted(stable_id(n) for n in names), dtype="<i8")
    return hashlib.sha256(ids.tobytes()).hexdigest()


def test_the_id_hash_gate_counts_the_molecule_as_missing(tmp_path):
    """``expect_sha256_ids`` compares the built corpus's ids, which do not
    include a rejected molecule: a hash of the input's ids including it is
    refused, one without it passes, exactly as for a ``_why_reject``
    rejection."""
    source = _write_binary(tmp_path / "input.bin.zst", _library())
    kept = ["ethanol", "phenol", "amine"]
    good = prepare(str(source), str(tmp_path / "good"),
                   expect_sha256_ids=_ids(kept), **OPTIONS)
    assert _build_json(good)["id_hash_gate"]["matched"] is True
    with pytest.raises(BuildRefused, match="id-hash gate"):
        prepare(str(source), str(tmp_path / "bad"),
                expect_sha256_ids=_ids(kept + ["silicon"]), **OPTIONS)
    assert not (tmp_path / "bad").exists()


def test_the_id_reference_gate_names_the_rejected_molecule(tmp_path):
    """``expect_ids_from``: a reference that holds the molecule fails the
    work unit that rejected it, naming it with its reason code, never the
    reason text that embeds its SMILES; a reference built from the same
    input, which rejected it too, passes."""
    lib = _library()
    stand_in = molecule("CC[Si](C)(C)C", "silicon", conformers=2)
    reference = prepare(str(_write_binary(tmp_path / "ref.bin.zst",
                                          [lib[0], stand_in, *lib[2:]])),
                        str(tmp_path / "reference"), **OPTIONS)
    source = str(_write_binary(tmp_path / "input.bin.zst", lib))
    with pytest.raises(BuildRefused, match=r"rejected 1 molecules that the reference "
                                           r"holds \(first: \['silicon \(smiles_unparsable\)'\]"):
        prepare(source, str(tmp_path / "out"), expect_ids_from=reference.root, **OPTIONS)
    same = prepare(source, str(tmp_path / "same"), **OPTIONS)
    again = prepare(source, str(tmp_path / "again"), expect_ids_from=same.root, **OPTIONS)
    assert _build_json(again)["id_reference"]["matched"] is True
    assert _build_json(again)["rejected"]["n"] == 1

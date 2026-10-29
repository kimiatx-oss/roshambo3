"""Builder review C1/C2: conformer ids, multi-conformer rejects and the
published reject list."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import MolSanitizeException

from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep import prepare
from roshambo3.prep.sources import _group_by_name


def embedded(smiles, name, n, seed=3):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert len(AllChem.EmbedMultipleConfs(mol, numConfs=n, randomSeed=seed)) == n
    mol.SetProp("_Name", name)
    return mol


def with_ids(mol, ids):
    """The same conformers, in the same order, carrying other ids; round
    tripped through the binary pickle a stream record is."""
    out = Chem.Mol(mol)
    confs = [Chem.Conformer(c) for c in mol.GetConformers()]
    out.RemoveAllConformers()
    for conf, cid in zip(confs, ids):
        conf.SetId(cid)
        out.AddConformer(conf, assignId=False)
    out = Chem.Mol(out.ToBinary(Chem.PropertyPickleOptions.AllProps))
    assert [c.GetId() for c in out.GetConformers()] == list(ids)
    return out


def unkekulizable(n):
    """Pyrrole with real conformers, then stripped of the N-H that makes the
    ring kekulizable (the recipe of test_prep's single-conformer test)."""
    bad = embedded("c1cc[nH]c1", "bad", n)
    nitrogen = next(a for a in bad.GetAtoms() if a.GetSymbol() == "N")
    nitrogen.SetNumExplicitHs(0)
    nitrogen.SetNoImplicit(True)
    h = next(nb.GetIdx() for nb in nitrogen.GetNeighbors() if nb.GetSymbol() == "H")
    ed = Chem.RWMol(bad)
    ed.RemoveAtom(h)
    bad = ed.GetMol()
    # Ring perception, which a parsed or unpickled sanitized molecule has; the
    # sanitize failure is then RemoveAllHs's kekulization, as in the wild.
    Chem.FastFindRings(bad)
    bad.SetProp("_Name", "bad")
    return bad


def write_stream(path, mols):
    blobs = []
    for mol in mols:
        blob = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
        blobs.append(struct.pack("<I", len(blob)) + blob)
    path.write_bytes(zstd.ZstdCompressor().compress(b"".join(blobs)))


def test_permuted_conformer_ids_keep_shape_and_color_together():
    """C1: a conformer's shape came from the position and its color from the
    id, so permuted ids paired conformer 0's color with conformer 1's shape.
    Color is one feature table per molecule and every conformer's
    pseudo-atoms are derived from its own stored atoms, so the pairing cannot
    recur; the table must not depend on the ids either."""
    gen = PharmacophoreGenerator()
    mol = embedded("c1ccccc1OCCNC(=O)C", "X", 3)
    reference = list(_group_by_name([with_ids(mol, [0, 1, 2])], 0, gen))[0]
    permuted = list(_group_by_name([with_ids(mol, [1, 0, 2])], 0, gen))[0]
    gapped = list(_group_by_name([with_ids(mol, [0, 2, 5])], 0, gen))[0]
    for record in (permuted, gapped):
        assert len(record.conformers) == 3
        # Content-defined identity is unaffected by the ids: the same
        # SMILES and the same canonical heavy-atom permutation per conformer.
        assert record.meta["smiles"] == reference.meta["smiles"]
        for a, b in zip(reference.atom_order, record.atom_order):
            np.testing.assert_array_equal(a, b)
        assert record.features == reference.features and record.features  # color
        for a, b in zip(reference.conformers, record.conformers):
            np.testing.assert_array_equal(a[0], b[0])      # shape
            assert a[2] == b[2]                            # self-overlap


@pytest.mark.parametrize("ids", [[1, 0, 2], [0, 2, 5]])
def test_binary_source_refuses_conformer_ids_other_than_record_order(tmp_path, ids):
    source = tmp_path / "input.bin.zst"
    write_stream(source, [embedded("CCO", "fine", 2),
                          with_ids(embedded("c1ccccc1OC", "odd", 3), ids)])
    with pytest.raises(ValueError, match=r"molecule 'odd' carries conformer ids"):
        prepare(source, str(tmp_path / "corpus"), workers=1,
                color_generator=PharmacophoreGenerator())
    assert not (tmp_path / "corpus").exists()


def test_conformer_ids_must_be_unique_everywhere():
    from roshambo3.prep.sources import _check_conformer_ids
    with pytest.raises(ValueError, match="duplicate conformer ids"):
        _check_conformer_ids("dup", [0, 1, 1], contiguous=False)
    _check_conformer_ids("ok", [3, 1, 2], contiguous=False)
    with pytest.raises(ValueError, match="not 0..2"):
        _check_conformer_ids("gap", [0, 1, 3], contiguous=True)


def test_multiconformer_sanitize_failure_is_a_reject_not_a_crash():
    """C2(b): strip_hydrogens sat outside the try in the multi-conformer
    branch, so a sanitize failure killed the build, on every resume."""
    bad = unkekulizable(3)
    with pytest.raises(MolSanitizeException):
        Chem.RemoveAllHs(bad)                # the premise, stated as a check
    good = embedded("CCOc1ccccc1", "good", 2)
    failures = []
    records = list(_group_by_name([bad, good], 0, PharmacophoreGenerator(), failures))
    assert [r.ext_id for r in records] == ["good"]
    assert len(failures) == 3, failures      # one entry per conformer, as per-conformer
    assert all(name == "bad" and "Kekulize" in why for name, why in failures)


def test_full_reject_list_is_published(tmp_path):
    """C2(b) end to end on the binary path, and C2(c): every reject is in the
    published list, which build.json names. In format 3 the corpus
    directory holds only its four names, so the full list sits beside it,
    ``<out>.rejected.jsonl``; build.json keeps the count and the examples."""
    source = tmp_path / "input.bin.zst"
    write_stream(source, [embedded("CCO", "first", 2), unkekulizable(3),
                          embedded("CCN", "last", 2)])
    corpus = prepare(source, str(tmp_path / "corpus"), workers=1,
                     color_generator=PharmacophoreGenerator())
    build = json.loads((Path(corpus.root) / "build.json").read_text())
    rejected = build["rejected"]
    assert rejected["n"] == 3 and rejected["file"] == "corpus.rejected.jsonl"
    lines = (tmp_path / "corpus.rejected.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == rejected["examples"]
    assert all(json.loads(line)["id"] == "bad" for line in lines)
    assert corpus.n_mol == 2
    assert not (Path(corpus.root) / "rejected.jsonl").exists()


def test_no_rejects_writes_no_file(tmp_path):
    source = tmp_path / "input.bin.zst"
    write_stream(source, [embedded("CCO", "first", 2)])
    corpus = prepare(source, str(tmp_path / "corpus"), workers=1)
    build = json.loads((Path(corpus.root) / "build.json").read_text())
    assert build["rejected"] == {"n": 0, "reasons": [], "examples": [], "file": None}
    assert not (tmp_path / "corpus.rejected.jsonl").exists()


def test_conformer_id_refusal_is_not_retryable(tmp_path):
    """A deterministic input fault found after work was committed is recorded
    ``retryable: false``, so the supervisor stops instead of resuming into the
    same molecule again (it lost 7-10 h per loop on the largest corpus build)."""
    from roshambo3.prep import build
    source = tmp_path / "input.bin.zst"
    fine = [embedded("CCO", f"fine-{i}", 2) for i in range(4)]       # slot 0
    write_stream(source, fine + [with_ids(embedded("c1ccccc1OC", "odd", 3), [1, 0, 2])])
    out = tmp_path / "corpus"
    for resume in (False, True):
        with pytest.raises(build.BuildRefused, match=r"molecule 'odd' carries conformer ids"):
            prepare(source, str(out), workers=1, records_per_unit=4, resume=resume)
        record = json.loads(Path(str(out) + ".progress.json").read_text())
        assert record["stage"] == "failed" and record["retryable"] is False
        assert "'odd'" in record["error"]
        assert Path(str(out) + ".partial").is_dir()        # committed work kept


def test_heavy_atom_mismatch_is_a_refusal():
    """Records of one id with different heavy-atom counts are different
    graphs: refused as a variant, before the n_real check is reached."""
    from roshambo3.prep.build import BuildRefused
    a = embedded("CCO", "X", 1)
    b = embedded("CCCO", "X", 1)
    with pytest.raises(BuildRefused, match=r"molecule 'X': its records differ .*'CCO'.*'CCCO'.*own id") as e:
        list(_group_by_name([a, b], 0, None))
    assert e.value.retryable is False
    assert isinstance(e.value, ValueError)      # callers matching ValueError still do


def test_inconsistent_shard_is_a_refusal(tmp_path):
    from roshambo3.prep import build
    (tmp_path / "n003_s00000.met").write_bytes(b"\0" * build.SHARD_META_DTYPE.itemsize)
    (tmp_path / "n003_s00000.xyz").write_bytes(b"\0" * 5)
    with pytest.raises(build.BuildRefused, match="coordinate bytes"):
        build._shard_index(str(tmp_path))

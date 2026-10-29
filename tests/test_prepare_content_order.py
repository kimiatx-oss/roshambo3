"""Corpus order and identity are functions of content.

* Order is ``(bucket, n_real, mol_id, conf_idx)`` and does not depend on the
  input's molecule order, the worker count or an interruption (FORMAT.md 9.2).
* Heavy atoms are stored in canonical, non-isomeric SMILES order, and the
  stored SMILES parses back to the slot elements (FORMAT.md 11.1).
* Charge and protonation variants under one id are refused, a duplicate
  ``(mol_id, conf_idx)`` across work units is refused, both by name.
* Stored int16 coordinates are on the 2^-8 A grid (FORMAT.md 4.1).
* A resume refuses changed input content or changed builder code.
"""
from __future__ import annotations

import json
import os
import random
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd
from rdkit import Chem

from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep import build, prepare
from roshambo3.prep import sources
from roshambo3.prep.conformer import strip_hydrogens
from roshambo3.prep.sources import BuildRefused, _prep_raw, canonical_heavy
from tests.format3_rows import read_corpus
from tests.prep_fixtures import (assert_same_rows, corpus_rows, molecule,
                                 synthetic_library, write_sdf)


def _build_record(corpus) -> dict:
    return json.loads((Path(corpus.root) / "build.json").read_text())

REPO = Path(__file__).resolve().parent.parent
#: Small work units, so the build has many of them and the workers and an
#: interruption really interleave; a group cap small enough to split buckets.
OPTIONS = dict(records_per_unit=10, max_conf_per_group=4)


@pytest.fixture(scope="module")
def library():
    return synthetic_library(40, seed=291, max_conformers=3)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, library):
    """The library in input order and with its molecules shuffled (each
    molecule's records keep their order, which defines conf_idx)."""
    root = tmp_path_factory.mktemp("order")
    ordered, shuffled = root / "ordered.sdf", root / "shuffled.sdf"
    write_sdf(ordered, library)
    mols = list(library)
    random.Random(7).shuffle(mols)
    write_sdf(shuffled, mols)
    return {"ordered": ordered, "shuffled": shuffled}


@pytest.fixture(scope="module")
def reference(tmp_path_factory, inputs):
    out = tmp_path_factory.mktemp("reference") / "corpus"
    return prepare(inputs["ordered"], str(out), workers=1,
                   color_generator=PharmacophoreGenerator(), **OPTIONS)


def test_every_bucket_is_populated_and_order_is_by_content(reference):
    rows = corpus_rows(reference)
    assert sorted(set(rows["bucket"].tolist())) == list(range(13))
    key = np.stack([rows["bucket"], rows["n_real"].astype(np.int64),
                    rows["mol_id"], rows["conf_idx"].astype(np.int64)], axis=1)
    step = np.diff(key, axis=0)
    first = np.argmax(step != 0, axis=1)
    assert np.all(step[np.arange(len(step)), first] > 0)   # strictly ascending
    assert len(reference.group_table) > 13                 # the cap split buckets


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_order_does_not_depend_on_input_order_or_workers(tmp_path, inputs, reference,
                                                         workers):
    built = prepare(inputs["shuffled"], str(tmp_path / "corpus"), workers=workers,
                    color_generator=PharmacophoreGenerator(), **OPTIONS)
    assert_same_rows(corpus_rows(reference), corpus_rows(built))


RESUMABLE = """
import sys
from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep import prepare
from tests import scatter_delay
scatter_delay.install()
prepare(sys.argv[1], sys.argv[2], workers=2, records_per_unit=10,
        max_conf_per_group=4, color_generator=PharmacophoreGenerator())
"""


def _progress(out):
    try:
        return json.loads(Path(str(out) + ".progress.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def test_order_survives_sigterm_and_resume(tmp_path, inputs, reference):
    out = tmp_path / "corpus"
    env = dict(os.environ, ROSHAMBO3_TEST_SCATTER_DELAY="0.4")
    proc = subprocess.Popen([sys.executable, "-c", RESUMABLE,
                             str(inputs["shuffled"]), str(out)], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(600):                      # at most 60 s to commit two units
        if _progress(out).get("slots_done", 0) >= 2 or proc.poll() is not None:
            break
        time.sleep(0.1)
    assert proc.poll() is None, proc.communicate()[1].decode()[-2000:]
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=60)
    assert proc.returncode == 128 + signal.SIGTERM, err.decode()[-2000:]
    record = _progress(out)
    assert record["stage"] == "paused" and record["resumable"] is True
    state = json.loads((Path(str(out) + ".partial") / build.CHECKPOINT_NAME).read_text())
    n_units = len(sources.SdfSource([str(inputs["shuffled"])]).units(10))
    assert 2 <= len(state["scatter"]["slots"]) < n_units
    assert all(s["content_sha256"] for s in state["scatter"]["slots"].values())
    resumed = prepare(inputs["shuffled"], str(out), workers=8, resume=True,
                      color_generator=PharmacophoreGenerator(), **OPTIONS)
    assert _build_record(resumed)["resumes"] == 1
    assert_same_rows(corpus_rows(reference), corpus_rows(resumed))


# --------------------------------------------------------------------------
# the SMILES is the graph

def _deuterated():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
    from rdkit.Chem import AllChem
    assert AllChem.EmbedMolecule(mol, randomSeed=3) == 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetNeighbors()[0].GetAtomicNum() == 6:
            atom.SetIsotope(2)
    mol.SetProp("_Name", "deuterated")
    return mol


def test_stored_smiles_parses_to_the_slot_elements(tmp_path, library):
    """For every stored molecule, MolFromSmiles(smiles) has n_real atoms whose
    elements are the slot elements in order, and no stereo marks. The slot
    elements are established from the coordinates: slot i holds source heavy
    atom order[i] (checked within half a step), whose element is compared."""
    chiral = molecule("C[C@H](N)C(=O)O", "chiral", conformers=2)
    mols = list(library[:13]) + [_deuterated(), chiral]
    prepare(mols, str(tmp_path / "corpus"), workers=1)
    rows = read_corpus(tmp_path / "corpus", merged=False)
    by_name = {m.GetProp("_Name"): m for m in mols}
    step = 2.0 ** -8
    deut = []
    for gid, (mol_id, conf_idx, n_real) in enumerate(zip(rows.mol_id, rows.conf_idx,
                                                          rows.n_real)):
        n_real = int(n_real)
        row = rows.molecule_row(mol_id)
        name, smiles = row["ext_id"], row["smiles"]
        source = by_name[name]
        one = Chem.Mol(source, confId=source.GetConformers()[int(conf_idx)].GetId())
        heavy, want_smiles, order = canonical_heavy(one)
        assert smiles == want_smiles
        assert not any(mark in smiles for mark in "@/\\"), smiles
        params = Chem.SmilesParserParams()
        params.allowCXSMILES = False
        parsed = Chem.MolFromSmiles(smiles, params)
        assert parsed.GetNumAtoms() == n_real == int(row["hac"])
        raw = np.asarray(strip_hydrogens(one).GetConformer().GetPositions())
        prepared = _prep_raw(raw[order])[0]
        assert np.all(np.abs(rows.coordinates(gid) - prepared) <= step / 2)
        source_elements = [a.GetAtomicNum() for a in strip_hydrogens(one).GetAtoms()]
        assert [a.GetAtomicNum() for a in parsed.GetAtoms()] == \
            [source_elements[i] for i in order]
        if name == "deuterated":
            deut.append(smiles)
    assert deut and all("2H" not in s and "[H]" not in s for s in deut)


def test_a_smiles_that_does_not_reproduce_the_slots_is_rejected(tmp_path, monkeypatch):
    """A SMILES that parses to other slots rejects the molecule, not the
    build (FORMAT.md 11.1).
    The unparsable case is ``test_prepare_smiles_roundtrip.py``'s."""
    original = sources.canonical_heavy

    def wrong(mol):
        heavy, smiles, order = original(mol)
        return heavy, smiles[::-1] if smiles == "CCO" else smiles, order

    monkeypatch.setattr(sources, "canonical_heavy", wrong)
    mols = [molecule("CCN", "fine"), molecule("OCC", "garbled")]
    assert original(mols[1])[1] == "CCO"
    corpus = prepare(mols, str(tmp_path / "corpus"), workers=1)
    rejected = _build_record(corpus)["rejected"]
    assert rejected["n"] == 1
    assert rejected["examples"][0]["id"] == "garbled"
    assert rejected["examples"][0]["why"].startswith(
        "SMILES round-trip check: its canonical SMILES 'OCC' does not reproduce "
        "the stored slots")
    assert read_corpus(corpus.root, merged=False).n_mol == 1


# --------------------------------------------------------------------------
# refusals, end to end

@pytest.mark.parametrize("variant", [("CCCN", "CCC[NH3+]"), ("CC(=O)O", "CC(=O)[O-]")])
def test_a_variant_under_one_id_refuses_the_build(tmp_path, variant):
    mols = [molecule(variant[0], "VAR-7"), molecule("CCO", "other"),
            molecule(variant[1], "VAR-7", seed=5)]
    with pytest.raises(BuildRefused, match=r"molecule 'VAR-7'.*own id"):
        prepare(mols, str(tmp_path / "memory"), workers=1)
    sdf = tmp_path / "in.sdf"
    write_sdf(sdf, mols)
    with pytest.raises(BuildRefused, match=r"molecule 'VAR-7'.*own id"):
        prepare(sdf, str(tmp_path / "sdf"), workers=1,
                color_generator=PharmacophoreGenerator())


def test_a_duplicate_conformer_across_work_units_is_refused(tmp_path):
    """Two binary records named alike land in two work units; merged, their
    rows are one run with conf_idx 0 twice, which nothing else sees."""
    mols = [molecule("CCOc1ccccc1", "dup"), molecule("CCN", "a"), molecule("CCC", "b"),
            molecule("CCOc1ccccc1", "dup", seed=5)]
    blobs = b"".join(struct.pack("<I", len(b)) + b for b in
                     (m.ToBinary(Chem.PropertyPickleOptions.AllProps) for m in mols))
    source = tmp_path / "in.bin.zst"
    source.write_bytes(zstd.ZstdCompressor().compress(blobs))
    with pytest.raises(BuildRefused,
                       match=r"conformer \(mol_id \d+, conf_idx 0\) of molecule \['dup'\] appears twice"):
        prepare(source, str(tmp_path / "corpus"), workers=2, records_per_unit=2)


# --------------------------------------------------------------------------
# the coordinate grid

def test_every_stored_coordinate_is_on_the_grid(reference, inputs):
    """Dequantized at the step (v * 2^-8), each stored coordinate is within
    half a step of its prepared value, re-prepared here in canonical order
    with ``_prep_raw``; ties round half to even, as np.rint."""
    from roshambo3 import numerics
    from roshambo3.format.r3g.limits import STEP_LOG2
    assert STEP_LOG2 == -8 and float(numerics.WIRE_QUANT_STEP) == 2.0 ** -8
    records: dict = {}                       # the SDF it was built from
    for mol in Chem.ForwardSDMolSupplier(str(inputs["ordered"]), removeHs=False):
        records.setdefault(mol.GetProp("_Name"), []).append(mol)
    rows = corpus_rows(reference)
    c = read_corpus(reference.root, merged=False)
    for k, (mol_id, conf_idx) in enumerate(zip(c.mol_id, c.conf_idx)):
        v = np.asarray(c.coordinates(k), dtype=np.float64) * 2.0 ** 8
        assert np.array_equal(v, np.rint(v))                 # integers: on the grid
        name = c.molecule_row(mol_id)["ext_id"]
        heavy, _, _ = canonical_heavy(records[name][int(conf_idx)])
        prepared = _prep_raw(np.asarray(heavy.GetConformer().GetPositions()))[0]
        assert np.all(np.abs(v * 2.0 ** -8 - prepared) <= 2.0 ** -9)
        assert np.array_equal(v, np.rint(prepared * 256.0))
    assert all(x.dtype == np.int16 for x in rows["xyz"])


# --------------------------------------------------------------------------
# resume identity

KILL = """
import os, sys
from roshambo3.prep import build, prepare
original = build._BuildCheckpoint.commit_slot
calls = [0]
def killed(self, *args, **kwargs):
    original(self, *args, **kwargs)
    calls[0] += 1
    if calls[0] >= 2:
        os._exit(73)
build._BuildCheckpoint.commit_slot = killed
prepare(sys.argv[1], sys.argv[2], workers=1, records_per_unit=10, max_conf_per_group=4)
"""


def _interrupted(tmp_path, library):
    source = tmp_path / "input.sdf"
    write_sdf(source, library)
    out = tmp_path / "corpus"
    result = subprocess.run([sys.executable, "-c", KILL, str(source), str(out)],
                            cwd=REPO, capture_output=True)
    assert result.returncode == 73, result.stderr.decode()[-2000:]
    return source, out


def test_resume_refuses_changed_content_with_the_same_size_and_mtime(tmp_path, library):
    source, out = _interrupted(tmp_path, library)
    st = os.stat(source)
    data = bytearray(source.read_bytes())
    # The last decimal of the first atom's x of the first record (work unit
    # 0, committed): same length, so the same size.
    at = data.index(b"\n", data.index(b"V2000")) + 10
    assert chr(data[at]).isdigit()
    data[at] = ord("1") if data[at] != ord("1") else ord("2")
    source.write_bytes(bytes(data))
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(source).st_size == st.st_size
    assert os.stat(source).st_mtime_ns == st.st_mtime_ns
    with pytest.raises(BuildRefused, match=r"input content changed.*work unit 0"):
        prepare(source, str(out), workers=1, resume=True, **OPTIONS)
    assert Path(str(out) + ".partial").is_dir()          # kept, not destroyed


def test_resume_refuses_changed_builder_code(tmp_path, library, monkeypatch):
    source, out = _interrupted(tmp_path, library)
    real = build._code_digest()
    monkeypatch.setattr(build, "_code_digest",
                        lambda: {**real, "python_sha256": "0" * 64})
    with pytest.raises(BuildRefused, match=r"builder code changed \(code digest: python_sha256"):
        prepare(source, str(out), workers=1, resume=True, **OPTIONS)
    monkeypatch.setattr(build, "_code_digest",
                        lambda: {**real, "native_cpu_src_sha256": "unknown"})
    with pytest.raises(BuildRefused, match=r"code digest cannot be established"):
        prepare(source, str(out), workers=1, resume=True, **OPTIONS)
    monkeypatch.undo()
    resumed = prepare(source, str(out), workers=1, resume=True, **OPTIONS)
    assert _build_record(resumed)["resumes"] >= 1


def test_the_code_digest_is_of_file_content(tmp_path, monkeypatch):
    """A changed byte in any producer module changes the digest -- a dirty
    tree gets its own digest, never a clean commit's."""
    import shutil
    package = Path(build.__file__).resolve().parent.parent
    copy = tmp_path / "roshambo3"
    for part in build.CODE_DIGEST_PACKAGES:
        shutil.copytree(package / part, copy / part,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for name in build.CODE_DIGEST_MODULES:
        shutil.copy2(package / name, copy / name)
    monkeypatch.setattr(build, "__file__", str(copy / "prep" / "build.py"))
    before = build._code_digest()
    assert before == {**build._code_digest()}
    target = copy / "format" / "r3g" / "writer.py"
    target.write_bytes(target.read_bytes() + b"\n# edited\n")
    after = build._code_digest()
    assert after["python_sha256"] != before["python_sha256"]
    assert after["native_cpu_src_sha256"] == before["native_cpu_src_sha256"]


# --------------------------------------------------------------------------
# dropped conformers (item 2)

def test_dropped_conformers_are_counted_per_molecule(tmp_path):
    """A record RDKit cannot parse is dropped and renumbers the molecule's
    later conformers; the count per id reaches the rejects file, the
    checkpoint and build.json (and the full list beside the corpus)."""
    from tests.test_sdf_merge_conformers import record_blocks, unparseable_block
    blocks = record_blocks(molecule("CCOc1ccccc1", "lossy", conformers=3))
    path = tmp_path / "in.sdf"
    path.write_text(blocks[0] + unparseable_block("lossy") + "".join(blocks[1:])
                    + "".join(record_blocks(molecule("CCN", "whole", conformers=2))))
    corpus = prepare(path, str(tmp_path / "corpus"), workers=1)
    record = _build_record(corpus)["dropped_conformers"]
    assert record["n_conformers"] == 1 and record["examples"] == [{"id": "lossy", "n": 1}]
    assert record["file"] == "corpus.dropped_conformers.jsonl"
    lines = (tmp_path / "corpus.dropped_conformers.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in lines] == [{"id": "lossy", "n": 1}]
    rows = read_corpus(tmp_path / "corpus", merged=False)
    lossy_id = sources.stable_id("lossy")
    lossy = rows.conf_idx[rows.mol_id == lossy_id].tolist()
    assert lossy == [0, 1, 2]                    # accepted conformers, renumbered


# --------------------------------------------------------------------------
# review fixes

SHRINK_KILL = """
import os, sys
from roshambo3.prep import build, prepare
orig = build._BuildCheckpoint.commit_slot
n = [0]
def killed(self, *a, **k):
    r = orig(self, *a, **k); n[0] += 1
    if n[0] >= 3: os._exit(73)
    return r
build._BuildCheckpoint.commit_slot = killed
prepare(sys.argv[1], sys.argv[2], workers=1, records_per_unit=2, max_conf_per_group=4)
"""


def test_binary_resume_refuses_committed_slots_the_stream_no_longer_reaches(tmp_path):
    """The input shrinks to 2 molecules, padded to the same size
    with a zstd skippable frame and given back its mtime. The committed
    slots past its end must refuse the resume, not be spliced in."""
    def stream(mols):
        out = []
        for m in mols:
            b = m.ToBinary(Chem.PropertyPickleOptions.AllProps)
            out.append(struct.pack("<I", len(b)) + b)
        return zstd.ZstdCompressor().compress(b"".join(out))

    smis = ["CCO", "CCN", "CCCO", "CCCN", "CCCCO", "c1ccccc1", "CC(=O)O", "OCCO"]
    mols = [molecule(s, f"M{i}") for i, s in enumerate(smis)]
    source = tmp_path / "in.bin.zst"
    v1 = stream(mols)
    source.write_bytes(v1)
    out = tmp_path / "corpus"
    r = subprocess.run([sys.executable, "-c", SHRINK_KILL, str(source), str(out)],
                       cwd=REPO, capture_output=True)
    assert r.returncode == 73, r.stderr.decode()[-1500:]
    st = os.stat(source)
    v2 = stream(mols[:2])
    pad = len(v1) - len(v2) - 8
    v2 = v2 + struct.pack("<II", 0x184D2A50, pad) + b"\0" * pad     # skippable frame
    assert len(v2) == len(v1)
    source.write_bytes(v2)
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(BuildRefused, match=r"input content changed.*work unit \d+.*absent"):
        prepare(source, str(out), workers=1, records_per_unit=2, max_conf_per_group=4,
                resume=True)


def test_the_code_digest_names_the_rdkit_and_numpy_releases(tmp_path, library, monkeypatch):
    """Canonical order and perception are RDKit-release
    properties, so a resume under another release is refused."""
    from rdkit import rdBase
    digest = build._code_digest()
    assert digest["rdkit_version"] == rdBase.rdkitVersion
    assert digest["numpy_version"] == np.__version__
    for name in ("numerics.py", "layout.py", "pharmacophore.py"):
        assert name in build.CODE_DIGEST_MODULES
    source, out = _interrupted(tmp_path, library)
    monkeypatch.setattr(build, "_code_digest",
                        lambda: {**digest, "rdkit_version": "2099.01.1"})
    with pytest.raises(BuildRefused, match=r"code digest: rdkit_version"):
        prepare(source, str(out), workers=1, resume=True, **OPTIONS)


def test_the_smiles_check_compares_charge_hydrogens_and_bonds():
    """J7 (review): elements alone are not the slot rule."""
    from roshambo3.prep.sources import check_smiles_slots
    heavy, smiles, _ = canonical_heavy(molecule("CC(=O)[O-]", "acetate"))
    check_smiles_slots("acetate", smiles, heavy)                  # passes
    for wrong in ("CC(=O)O",        # charge and hydrogens differ, same elements
                  "CC(O)=O",        # bonds moved between the two oxygens
                  "C=C(O)[O-]"):    # bond orders and hydrogens differ
        with pytest.raises(BuildRefused, match=r"molecule 'acetate'.*does not reproduce"):
            check_smiles_slots("acetate", wrong, heavy)
    with pytest.raises(BuildRefused, match=r"does not parse"):
        check_smiles_slots("acetate", "C(", heavy)


def test_a_same_formula_variant_refusal_mentions_resonance_forms():
    """J3 (review): an amidinium drawn with the charge on either nitrogen is
    one protonation state; the refusal says so."""
    mols = [molecule("CC(N)=[NH2+]", "amidinium"), molecule("CC(=N)[NH3+]", "amidinium", seed=5)]
    src = sources.RDKitSource(mols)
    with pytest.raises(BuildRefused, match=r"resonance or Lewis forms"):
        [r for u in src.units() for r in src.iter_unit(u)]
    other = sources.RDKitSource([molecule("CCCN", "P"), molecule("CCC[NH3+]", "P")])
    with pytest.raises(BuildRefused) as e:
        [r for u in other.units() for r in other.iter_unit(u)]
    assert "resonance" not in str(e.value)       # different formula: no hint


def test_molecular_weight_does_not_depend_on_atom_order():
    """J4 (review): mw is computed on the canonical graph."""
    a = molecule("OC(=O)c1ccc(cc1)C(=O)NCCCCCl", "X")
    rng = random.Random(3)
    weights = set()
    for _ in range(8):
        order = list(range(a.GetNumAtoms()))
        rng.shuffle(order)
        b = Chem.RenumberAtoms(a, order)
        b.SetProp("_Name", "X")
        rec = next(iter(sources._group_by_name([b], 0, None)))
        weights.add(np.float64(rec.meta["mw"]).tobytes())
    assert len(weights) == 1

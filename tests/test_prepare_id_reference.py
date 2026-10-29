"""Builder review S3: ``expect_ids_from`` checks each work unit against a
reference corpus's ids before it is committed, and the final id hash.

Format 3: the id hash is ``ids_sha256``, the sorted unique
``mol_id`` column of ``molecules.parquet`` as little-endian int64 -- the same
bytes format 2 hashed as ``sha256_ids`` over ``ids.i64``. The reference is a
format-3 corpus or a format-2-shaped directory (``ids.i64`` and a
``manifest.json`` with ``sha256_ids``, nothing else read)."""
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3.prep import build, prepare
from roshambo3.prep.conformer import stable_id

N = 30
OPTIONS = dict(workers=1, records_per_unit=4, max_conf_per_group=20)


def _mols(prefix, n=N):
    templates = []
    for smiles in ("CCO", "CCN(C)C", "c1ccccc1O"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert len(AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=5)) == 2
        templates.append(mol)
    out = []
    for i in range(n):
        mol = Chem.Mol(templates[i % len(templates)])
        mol.SetProp("_Name", f"{prefix}-{i:03d}")
        out.append(mol)
    return out


def _write(path, mols):
    if path.suffix == ".sdf":
        with Chem.SDWriter(str(path)) as writer:
            for mol in mols:
                for conf in mol.GetConformers():
                    writer.write(mol, confId=conf.GetId())
    else:
        blobs = [m.ToBinary(Chem.PropertyPickleOptions.AllProps) for m in mols]
        path.write_bytes(zstd.ZstdCompressor().compress(
            b"".join(struct.pack("<I", len(b)) + b for b in blobs)))
    return path


def _build(corpus) -> dict:
    return json.loads((Path(corpus.root) / "build.json").read_text())


def _tsv(mol_id, ext_id) -> str:
    """One row of a scatter slot's metadata file (``_ShardSet.add``)."""
    return f"{int(mol_id)}\t{json.dumps(ext_id)}\t3\t46.0\t0\tCCO\n"


def _format2_reference(root, ids, *, n_ids=True):
    """A format-2-shaped directory: only ``ids.i64`` and a ``manifest.json``
    with ``sha256_ids`` (the bytes format 2 hashed)."""
    root.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(ids, dtype="<i8")
    ids.tofile(root / "ids.i64")
    manifest = {"sha256_ids": hashlib.sha256(ids.tobytes()).hexdigest()}
    if n_ids:
        manifest["n_ids"] = int(ids.size)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture(params=["binary", "sdf"])
def kind(request):
    return request.param


def _source(tmp_path, kind, mols, name="input"):
    return _write(tmp_path / (name + (".sdf" if kind == "sdf" else ".bin.zst")), mols)


def test_correct_reference_passes(tmp_path, kind):
    source = _source(tmp_path, kind, _mols("MOL"))
    reference = prepare(source, str(tmp_path / "reference"), **OPTIONS)
    again = prepare(source, str(tmp_path / "again"),
                    expect_ids_from=str(tmp_path / "reference"), **OPTIONS)
    record = _build(again)["id_reference"]
    assert record["matched"] is True and record["final_comparison"] == "ids_sha256"
    assert record["kind"] == "format 3"
    assert record["early_checked_molecules"] == N
    assert record["ids_sha256"] == reference.descriptor["ids_sha256"]
    assert _build(reference)["id_reference"] is None     # additive, opt-in


def test_wrong_reference_fails_on_the_first_unit(tmp_path, kind):
    prepare(_source(tmp_path, kind, _mols("OTHER"), "other"),
            str(tmp_path / "reference"), **OPTIONS)
    source = _source(tmp_path, kind, _mols("MOL"))
    called = []
    original = build._BuildCheckpoint.commit_slot
    t0 = time.monotonic()
    with pytest.raises(build.BuildRefused, match=r"work unit 0 \(not committed\)"):
        build._BuildCheckpoint.commit_slot = lambda *a, **k: called.append(a)
        try:
            prepare(source, str(tmp_path / "corpus"),
                    expect_ids_from=str(tmp_path / "reference"), **OPTIONS)
        finally:
            build._BuildCheckpoint.commit_slot = original
    assert time.monotonic() - t0 < 30
    assert called == []                                    # nothing committed
    # A fresh build that committed nothing leaves nothing that looks like one.
    for suffix in ("", ".partial", ".progress.json"):
        assert not Path(str(tmp_path / "corpus") + suffix).exists(), suffix


def test_partly_wrong_source_stops_at_the_first_bad_unit(tmp_path):
    reference_mols = _mols("MOL", 12)               # units 0..2 of 4 records each
    prepare(_source(tmp_path, "binary", reference_mols, "ref"),
            str(tmp_path / "reference"), **OPTIONS)
    source = _source(tmp_path, "binary", _mols("MOL"))
    out = tmp_path / "corpus"
    with pytest.raises(build.BuildRefused, match="work unit 3"):
        prepare(source, str(out), expect_ids_from=str(tmp_path / "reference"),
                **OPTIONS)
    partial = Path(str(out) + ".partial")
    state = json.loads((partial / build.CHECKPOINT_NAME).read_text())
    assert sorted(int(s) for s in state["scatter"]["slots"]) == [0, 1, 2]
    record = json.loads(Path(str(out) + ".progress.json").read_text())
    assert record["stage"] == "failed" and record["retryable"] is False
    assert "MOL-012" in record["error"]
    with pytest.raises(build.BuildRefused, match="work unit 3"):
        prepare(source, str(out), expect_ids_from=str(tmp_path / "reference"),
                resume=True, **OPTIONS)


def test_prefix_build_passes_and_skips_the_final_hash(tmp_path):
    source = _source(tmp_path, "binary", _mols("MOL"))
    prepare(source, str(tmp_path / "reference"), **OPTIONS)
    prefix = prepare(source, str(tmp_path / "prefix"), max_records=10,
                     expect_ids_from=str(tmp_path / "reference"), **OPTIONS)
    record = _build(prefix)["id_reference"]
    assert record["matched"] is None
    assert record["final_comparison"].startswith("skipped: a max_records=10 prefix")
    assert record["early_checked_molecules"] == 10


def test_stable_id_mismatch_and_bounded_reference(tmp_path):
    source = _source(tmp_path, "binary", _mols("MOL"))
    reference = prepare(source, str(tmp_path / "reference"), **OPTIONS)
    ref = build._IdReference(reference.root)
    ids = ref.ids
    assert ids.dtype == np.dtype("<i8") and np.all(np.diff(ids) > 0)
    name = next(f"MOL-{i:03d}" for i in range(N) if stable_id(f"MOL-{i:03d}") == ids[0])
    (tmp_path / "meta_00007.tsv").write_text(
        _tsv(ids[0], name) + _tsv(ids[1], "RENAMED"))
    with pytest.raises(build.BuildRefused, match=r"1 have mol_id != stable_id.*RENAMED"):
        ref.check_slot(str(tmp_path), 7)
    # A format-2 reference's ids.i64 is memory-mapped, never loaded.
    old = _format2_reference(tmp_path / "format2", ids)
    assert isinstance(build._IdReference(str(old)).ids, np.memmap)


def test_reference_must_be_a_corpus(tmp_path):
    with pytest.raises(ValueError, match="not a corpus"):
        prepare(_source(tmp_path, "binary", _mols("MOL")), str(tmp_path / "c"),
                expect_ids_from=str(tmp_path), **OPTIONS)


def test_cli_flag(tmp_path, capsys):
    from roshambo3.cli import main
    source = _source(tmp_path, "binary", _mols("MOL"))
    assert main(["prepare", str(source), "-o", str(tmp_path / "reference"),
                 "--records-per-unit", "4", "--workers", "1"]) == 0
    assert main(["prepare", str(source), "-o", str(tmp_path / "again"),
                 "--records-per-unit", "4", "--workers", "1",
                 "--expect-ids-from", str(tmp_path / "reference")]) == 0
    record = json.loads((tmp_path / "again" / "build.json").read_text())
    assert record["id_reference"]["matched"] is True
    assert "30 molecules checked" in capsys.readouterr().err


def test_rejected_molecule_the_reference_holds_fails_its_unit(tmp_path):
    """The early check covered only accepted molecules, so a molecule this
    build rejects but the reference holds surfaced only at the final
    sha256_ids comparison, a whole build later. It now fails its unit."""
    from tests.test_prepare_conformer_ids import embedded, unkekulizable
    fine = _mols("MOL", 4)                                  # unit 0
    tail = [embedded("CCN", "last", 2)]
    reference_mols = fine + [embedded("c1cc[nH]c1", "bad", 3)] + tail
    prepare(_source(tmp_path, "binary", reference_mols, "ref"),
            str(tmp_path / "reference"), **OPTIONS)
    source = _source(tmp_path, "binary", fine + [unkekulizable(3)] + tail)
    out = tmp_path / "corpus"
    for resume in (False, True):
        with pytest.raises(build.BuildRefused,
                           match=r"work unit 1 \(not committed\) rejected 1 molecules .*[\"']bad \("):
            prepare(source, str(out), expect_ids_from=str(tmp_path / "reference"),
                    resume=resume, **OPTIONS)
        record = json.loads(Path(str(out) + ".progress.json").read_text())
        assert record["stage"] == "failed" and record["retryable"] is False
    state = json.loads((Path(str(out) + ".partial") / build.CHECKPOINT_NAME).read_text())
    assert sorted(int(s) for s in state["scatter"]["slots"]) == [0]


def test_rejects_the_reference_also_lacks_pass(tmp_path):
    """The same reject against a reference built from the same input (which
    rejected it too) is not an error."""
    from tests.test_prepare_conformer_ids import unkekulizable
    source = _source(tmp_path, "binary", _mols("MOL", 4) + [unkekulizable(3)])
    prepare(source, str(tmp_path / "reference"), **OPTIONS)
    again = prepare(source, str(tmp_path / "again"),
                    expect_ids_from=str(tmp_path / "reference"), **OPTIONS)
    assert _build(again)["id_reference"]["matched"] is True
    assert _build(again)["rejected"]["n"] == 3


def test_partly_rejected_molecule_is_not_a_reject(tmp_path):
    """A molecule that lost some conformers but was accepted is present."""
    ref = build._IdReference.__new__(build._IdReference)
    ref.root, ref.checked = "ref", 0
    ref.ids = np.array(sorted([stable_id("A"), stable_id("B")]), dtype=np.int64)
    (tmp_path / "meta_00000.tsv").write_text(_tsv(stable_id("A"), "A"))
    ref.check_slot(str(tmp_path), 0, [("A", "one conformer failed")])
    with pytest.raises(build.BuildRefused, match=r"rejected 1 molecules .*'B \(other\)'"):
        ref.check_slot(str(tmp_path), 0, [("A", "x"), ("B", "why"), ("_record_9", "y")])


# --------------------------------------------------------------------------
# A format-2-shaped reference, the expected hash, the shared definition

def _ids_of(mols):
    return np.array(sorted(stable_id(m.GetProp("_Name")) for m in mols), dtype="<i8")


def test_a_format2_reference_passes_and_its_hash_is_ids_sha256(tmp_path):
    """The format-3 id hash is format 2's ``sha256_ids`` by definition: a
    format-2 directory written by the test (ids.i64 + manifest sha256_ids)
    matches a format-3 build of the same molecules."""
    mols = _mols("MOL")
    old = _format2_reference(tmp_path / "format2", _ids_of(mols), n_ids=False)
    source = _source(tmp_path, "binary", mols)
    again = prepare(source, str(tmp_path / "again"), expect_ids_from=str(old), **OPTIONS)
    record = _build(again)["id_reference"]
    assert record["matched"] is True and record["kind"] == "format 2"
    assert record["ids_sha256"] == again.descriptor["ids_sha256"] == json.loads(
        (old / "manifest.json").read_text())["sha256_ids"]


@pytest.mark.parametrize("kind", ["format 3", "format 2"])
def test_a_missing_or_extra_id_refuses_before_publication(tmp_path, kind):
    """The build holds an id the reference lacks: refused at its work unit.
    The reference holds an id the build lacks: every unit passes, and the
    final ids_sha256 comparison refuses it. Neither publishes."""
    mols = _mols("MOL", 12)
    source = _source(tmp_path, "binary", mols)
    if kind == "format 3":
        fewer = prepare(_source(tmp_path, "binary", mols[:-1], "fewer"),
                        str(tmp_path / "fewer"), **OPTIONS).root
        more = prepare(_source(tmp_path, "binary", mols + _mols("EXTRA", 1), "more"),
                       str(tmp_path / "more"), **OPTIONS).root
    else:
        fewer = _format2_reference(tmp_path / "fewer", _ids_of(mols[:-1]))
        more = _format2_reference(tmp_path / "more", _ids_of(mols + _mols("EXTRA", 1)))
    for ref, match in ((fewer, r"absent from the reference ids \(first: \['MOL-011'\]\)"),
                       (more, r"id reference: the built corpus's ids_sha256 is")):
        out = tmp_path / f"out-{Path(ref).name}"
        with pytest.raises(build.BuildRefused, match=match):
            prepare(source, str(out), expect_ids_from=str(ref), **OPTIONS)
        assert not out.exists()


def test_expect_ids_sha256(tmp_path):
    mols = _mols("MOL", 6)
    want = hashlib.sha256(_ids_of(mols).tobytes()).hexdigest()
    source = _source(tmp_path, "binary", mols)
    good = prepare(source, str(tmp_path / "good"), expect_sha256_ids=want, **OPTIONS)
    assert _build(good)["id_hash_gate"] == {"expected": want, "actual": want, "matched": True}
    out = tmp_path / "bad"
    with pytest.raises(build.BuildRefused, match="id-hash gate"):
        prepare(source, str(out), expect_sha256_ids="0" * 64, **OPTIONS)
    assert not out.exists()
    record = json.loads(Path(str(out) + ".progress.json").read_text())
    assert record["id_hash_gate"]["matched"] is False


def test_a_format2_reference_that_does_not_hash_to_its_manifest_is_refused(tmp_path):
    old = _format2_reference(tmp_path / "format2", _ids_of(_mols("MOL", 3)))
    manifest = json.loads((old / "manifest.json").read_text())
    manifest["sha256_ids"] = "0" * 64
    (old / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not hash to the manifest"):
        build._IdReference(str(old))

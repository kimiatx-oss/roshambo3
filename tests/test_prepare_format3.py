"""The builder emits format 3 (``doc/FORMAT.md``).

What a built corpus must be, end to end: only the four format-3 names, every
corpus passing ``roshambo3 verify --tier deep``, molecule-aligned blocks at
any ``block_rows`` with the per-molecule cap at the format's 4,096-row limit,
a fingerprint that is the descriptor's alone (``build.json`` never in it, the
same in another process), no private path in ``corpus.json``, and the
refusals the format needs (a ``mol_id`` outside ``[0, 2^63)``, an
out-of-range ``block_rows``). Synthetic molecules: correctness checks, not
chemistry or accuracy evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3.format.r3g.limits import LIMITS
from roshambo3.prep import prepare
from roshambo3.prep.build import BuildRefused
from tests.format3_rows import read_corpus
from tests.prep_fixtures import fixtures, molecule, write_source

CAP = LIMITS["n_conf_per_molecule"]
BLOCK_CAP = LIMITS["block_rows"]


def _verify_deep(root) -> None:
    from roshambo3.cli import main
    assert main(["verify", str(root), "--tier", "deep", "--all"]) == 0


def _many_conformers(name: str, n: int, smiles: str = "CC(=O)N", seed: int = 3):
    """One molecule with ``n`` conformers: one embedding, then small seeded
    displacements (the builder only needs distinct finite coordinates)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    base = np.asarray(mol.GetConformer().GetPositions())
    rng = np.random.default_rng(seed)
    mol.RemoveAllConformers()
    for _ in range(n):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, p in enumerate(base + rng.normal(scale=0.05, size=base.shape)):
            conf.SetAtomPosition(i, p.tolist())
        mol.AddConformer(conf, assignId=True)
    mol.SetProp("_Name", name)
    return mol


@pytest.fixture(scope="module")
def sdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("in") / "fixtures.sdf"
    write_source(path, fixtures() + [molecule("c1ccccc1CC(=O)O", "acid", conformers=5),
                                     molecule("CCCCCCCCCCCCCCN", "long", conformers=4)])
    return path


# ---------------------------------------------------------------------------
# What a built corpus is
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("color", [False, True])
def test_a_synthetic_build_passes_verify_deep_and_writes_only_format3(tmp_path, sdf, color):
    from roshambo3.pharmacophore import PharmacophoreGenerator
    out = tmp_path / "corpus"
    prepare([str(sdf)], str(out), workers=2, records_per_unit=3,
            color_generator=PharmacophoreGenerator() if color else None)
    _verify_deep(out)
    # No format-2 file; only the four names.
    assert sorted(os.listdir(out)) == ["build.json", "corpus.json", "groups",
                                       "molecules.parquet"]
    assert all(n.endswith(".r3g") for n in os.listdir(out / "groups"))
    text = (out / "corpus.json").read_text()
    for private in ("fdef_path", "/data/", "/home/", "/tmp", str(tmp_path)):
        assert private not in text, private
    # Private paths live in build.json only.
    build = json.loads((out / "build.json").read_text())
    assert str(sdf) in json.dumps(build["source"])
    rows = read_corpus(out)
    assert ("color" in rows.descriptor) is color
    if color:
        assert all(g.color_self is not None for g in rows.groups)


def test_the_fingerprint_is_the_descriptors_alone(tmp_path, sdf):
    """``build.json`` is never fingerprinted (FORMAT.md 10.5, 12.1): the
    fingerprint recomputed from ``corpus.json`` -- here, in another process --
    equals the one every group header carries, and the corpus still passes
    ``scan`` without ``build.json``."""
    out = tmp_path / "corpus"
    corpus = prepare([str(sdf)], str(out), workers=1)
    code = ("import json,sys; from roshambo3.format.r3g.descriptor import fingerprint; "
            "print(fingerprint(json.load(open(sys.argv[1]))))")
    other = subprocess.run([sys.executable, "-c", code, str(out / "corpus.json")],
                           check=True, capture_output=True, text=True).stdout.strip()
    assert other == corpus.fingerprint
    for g in read_corpus(out).groups:
        assert g.header["fingerprint"] == corpus.fingerprint
    os.unlink(out / "build.json")
    from roshambo3.format.r3g.validate import validate_corpus
    assert validate_corpus(str(out), "scan").fingerprint == corpus.fingerprint


def test_ids_sha256_and_molecules_sha256(tmp_path, sdf):
    out = tmp_path / "corpus"
    prepare([str(sdf)], str(out), workers=1)
    d = json.loads((out / "corpus.json").read_text())
    rows = read_corpus(out)
    ids = np.asarray(rows.molecules["mol_id"], dtype=np.uint64).astype("<i8")
    assert d["ids_sha256"] == hashlib.sha256(np.unique(ids).tobytes()).hexdigest()
    assert d["molecules_sha256"] == hashlib.sha256(
        (out / "molecules.parquet").read_bytes()).hexdigest()


def test_the_definition_hash_is_printed_at_the_end_of_a_color_build(tmp_path, sdf, capfd):
    """Handed to whoever screens the corpus."""
    from roshambo3 import definitions
    from roshambo3.pharmacophore import PharmacophoreGenerator
    gen = PharmacophoreGenerator()
    prepare([str(sdf)], str(tmp_path / "c"), workers=1, color_generator=gen)
    h = definitions.definition_hash_of(gen.descriptor_color())
    assert f"color definition sha256 {h}" in capfd.readouterr().err


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def _decoded(rows):
    """Everything a block size must not change: per group, the run table,
    the feature table, the decoded rows and the self-overlaps."""
    out = []
    for g in rows.groups:
        out.append((g.bucket, g.group, g.runs.tobytes(),
                    {k: v.tobytes() for k, v in g.feature_table.items()},
                    g.xyz.tobytes(), g.vaa.tobytes(),
                    None if g.color_self is None else g.color_self.tobytes(),
                    g.L, g.L_heavy, g.L_field_width))
    return out


def _check_block_rule(rows, block_rows):
    """FORMAT.md 6 rules 1-5: molecule-aligned, tiling, at most 4,096 rows,
    and the writer's greedy rule at ``block_rows``."""
    for g in rows.groups:
        start = g.run_start
        n_conf = g.runs["n_conf"].astype(int)
        first = 0
        for i, b in enumerate(g.blocks):
            m0, m1 = int(b["m0"]), int(b["m1"])
            assert int(b["first_row"]) == first == start[m0]
            assert int(b["rows"]) == start[m1] - start[m0]
            assert 1 <= int(b["rows"]) <= BLOCK_CAP
            assert int(b["rows"]) <= block_rows or m1 - m0 == 1
            if m1 < g.n_mol:            # the next molecule would not have fit
                assert int(b["rows"]) + n_conf[m1] > block_rows
            first += int(b["rows"])
        assert first == g.n_conf and int(g.blocks[-1]["m1"]) == g.n_mol


@pytest.mark.parametrize("color", [False, True])
def test_block_rows_changes_no_row_and_no_fingerprint(tmp_path, sdf, color):
    """The same input at block_rows 1,024 and 4,096 decodes
    to identical rows, with identical fingerprints and corpus semantics; and
    the tiny caps a fixture uses (1, 3) are legal (FORMAT.md B.8)."""
    from roshambo3.pharmacophore import PharmacophoreGenerator
    built = {}
    for block_rows in (BLOCK_CAP, 1024, 3, 1):
        out = tmp_path / f"b{block_rows}"
        c = prepare([str(sdf)], str(out), workers=1, block_rows=block_rows,
                    color_generator=PharmacophoreGenerator() if color else None)
        rows = read_corpus(out, merged=False)
        _check_block_rule(rows, block_rows)
        assert rows.build["parameters"]["block_rows"] == block_rows
        built[block_rows] = (c.fingerprint, _decoded(rows),
                             sum(len(g.blocks) for g in rows.groups))
        _verify_deep(out)
    for block_rows in (1024, 3, 1):
        assert built[block_rows][:2] == built[BLOCK_CAP][:2], block_rows
    assert built[1][2] > built[3][2] > built[BLOCK_CAP][2]


def test_a_molecule_of_the_cap_fills_one_block_and_one_more_is_refused(tmp_path):
    """FORMAT.md 6 rule 4: at most
    4,096 conformers per molecule, so every block holds at most 4,096 rows; a
    molecule of exactly the cap fills a block on its own, between blocks of
    its neighbors, and one more conformer rejects the molecule, naming the
    limit. This is the cap lift (256 -> 4,096) end to end."""
    from roshambo3.prep.conformer import stable_id
    # Names whose stable ids put the big molecule between the two small ones
    # in corpus order (n_real, mol_id): checked, not assumed.
    assert stable_id("small-15") < stable_id("big") < stable_id("small-0")
    small_a = _many_conformers("small-15", 3, seed=1)
    small_b = _many_conformers("small-0", 5, seed=2)
    big = _many_conformers("big", CAP, seed=3)
    over = _many_conformers("over", CAP + 1, seed=4)
    out = tmp_path / "cap"
    prepare([small_a, big, small_b, over], str(out), workers=1)
    rows = read_corpus(out, merged=False)
    assert rows.n_mol == 3
    (g,) = rows.groups
    _check_block_rule(rows, BLOCK_CAP)
    big_m = int(np.flatnonzero(g.runs["n_conf"] == CAP)[0])
    blocks = [b for b in g.blocks if int(b["m0"]) <= big_m < int(b["m1"])]
    assert len(blocks) == 1
    assert (int(blocks[0]["m1"]) - int(blocks[0]["m0"]), int(blocks[0]["rows"])) == (1, CAP)
    assert len(g.blocks) == 3
    build = rows.build
    assert build["rejected"]["n"] == 1
    why = build["rejected"]["examples"][0]["why"]
    assert build["rejected"]["examples"][0]["id"] == "over"
    assert f"{CAP + 1} conformers" in why and f"limit of {CAP}" in why
    rejected = (tmp_path / "cap.rejected.jsonl").read_text().splitlines()
    assert [json.loads(x)["id"] for x in rejected] == ["over"]
    _verify_deep(out)


def test_block_rows_out_of_range_is_refused(tmp_path, sdf):
    for bad in (0, BLOCK_CAP + 1, 2.5, True):
        with pytest.raises(ValueError, match="block_rows"):
            prepare([str(sdf)], str(tmp_path / f"bad{bad}"), block_rows=bad)
        assert not (tmp_path / f"bad{bad}").exists()


def test_cli_block_rows(tmp_path, sdf):
    from roshambo3.cli import main
    out = tmp_path / "cli"
    assert main(["prepare", str(sdf), "-o", str(out), "--block-rows", "2",
                 "--workers", "1"]) == 0
    rows = read_corpus(out, merged=False)
    assert rows.build["parameters"]["block_rows"] == 2
    _check_block_rule(rows, 2)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_a_mol_id_outside_int63_is_refused_naming_it(tmp_path, monkeypatch):
    """Format 3 stores mol_id as u64 below 2^63 (FORMAT.md 7,
    limit 35) and the id hash serializes it as int64; nothing checked it."""
    from roshambo3.prep import sources
    real = sources.stable_id
    monkeypatch.setattr(sources, "stable_id",
                        lambda text: (1 << 63) + 5 if text == "multi" else real(text))
    with pytest.raises(BuildRefused, match=r"mol_id 9223372036854775813, outside \[0, 2\^63\)"):
        prepare(fixtures(), str(tmp_path / "big-id"), workers=1)
    assert not (tmp_path / "big-id").exists()


def test_the_writers_refuse_a_mol_id_outside_int63(tmp_path):
    from roshambo3.format.r3g.parquet import MoleculesWriter
    from roshambo3.format.r3g.reference import pack_runs
    from roshambo3.format.r3g.writer import GroupWriter
    w = MoleculesWriter(str(tmp_path / "m.parquet"))
    with pytest.raises(ValueError, match="outside"):
        w.write([1 << 63], ["x"], ["C"], [1], [12.0], [0])
    with GroupWriter(str(tmp_path), bucket=0, group=0, fingerprint="0" * 64,
                     n_types=0) as g:
        with pytest.raises(ValueError, match="outside"):
            g.add(pack_runs([1 << 63], [1], [1]), np.zeros((1, 3, 1), np.int16), [1.0])
        g.abort()
    assert not (tmp_path / "groups" / "b00_g0000.r3g").exists()


# ---------------------------------------------------------------------------
# The streaming writers are the conformance writers
# ---------------------------------------------------------------------------

def test_the_streaming_parquet_writer_is_write_table_byte_for_byte(tmp_path):
    """``MoleculesWriter`` takes rows in batches of any size and must write
    the bytes ``pq.write_table(..., row_group_size=2^14)`` writes: several
    row groups here, the case the conformance corpora (one row group) miss."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from roshambo3.format.r3g.parquet import MoleculesWriter, _schema
    n = 2 * LIMITS["parquet_row_group"] + 777
    rng = np.random.default_rng(1)
    ids = np.sort(rng.choice(1 << 62, n, replace=False)).astype(np.uint64)
    ext = [f"id-{i}" for i in range(n)]
    smi = ["C" * (1 + i % 40) for i in range(n)]
    hac = (1 + np.arange(n) % 40).astype(np.uint8)
    mw = rng.uniform(10, 500, n)
    rot = (np.arange(n) % 9).astype(np.uint16)
    pq.write_table(pa.table({"mol_id": ids.tolist(), "ext_id": ext, "smiles": smi,
                             "hac": hac.tolist(), "mw": mw.tolist(), "rotors": rot.tolist()},
                            schema=_schema()),
                   str(tmp_path / "a.parquet"), compression="zstd", compression_level=9,
                   write_statistics=False, store_schema=False, use_dictionary=False,
                   row_group_size=LIMITS["parquet_row_group"])
    w = MoleculesWriter(str(tmp_path / "b.parquet"))
    for lo, hi in ((0, 5), (5, 20000), (20000, 20001), (20001, n)):
        w.write(ids[lo:hi], ext[lo:hi], smi[lo:hi], hac[lo:hi], mw[lo:hi], rot[lo:hi])
    w.close()
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


def test_group_writer_chunks_do_not_change_the_bytes(tmp_path):
    """The builder hands ``GroupWriter`` chunks of many molecules that cut
    across block boundaries; the bytes must be those of one molecule at a
    time (``write_group``, which the conformance tests pin to the reference)."""
    from roshambo3.format.r3g.reference import pack_runs
    from roshambo3.format.r3g.writer import GroupWriter, Molecule, write_group
    rng = np.random.default_rng(3)
    mols = []
    for i, nc in enumerate([3, 1, 4, 2, 7, 1, 1, 5]):
        xyz = rng.integers(-3000, 3000, size=(nc, 3, 13), dtype=np.int16)
        feats = sorted({(int(rng.integers(1, 3)), tuple(sorted({int(a) for a in
                         rng.integers(0, 13, size=int(rng.integers(1, 4)))})))
                        for _ in range(int(rng.integers(0, 4)))})
        mols.append(Molecule(100 + i, xyz, rng.uniform(1, 9, nc).astype(np.float32), feats,
                             rng.uniform(0, 3, nc), f"m{i}", "C", 1.0, 0))
    write_group(str(tmp_path / "one"), mols, bucket=1, group=0, fingerprint="a" * 64,
                n_types=2, with_color_self=True, block_rows=6)
    w = GroupWriter(str(tmp_path / "chunked"), bucket=1, group=0, fingerprint="a" * 64,
                    n_types=2, with_color_self=True, block_rows=6, check_decode=True)
    for lo, hi in ((0, 3), (3, 4), (4, 8)):
        part = mols[lo:hi]
        flat = [x for m in part for x in m.features]
        w.add(pack_runs([m.mol_id for m in part], [m.n_conf for m in part],
                        [m.n_real for m in part]),
              np.concatenate([m.coords for m in part]),
              np.concatenate([m.vaa for m in part]),
              np.concatenate([m.color_self for m in part]),
              ([len(m.features) for m in part], [t for t, _ in flat],
               [len(a) for _, a in flat], [x for _, a in flat for x in a]))
    w.close()
    name = "groups/b01_g0000.r3g"
    assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "chunked" / name).read_bytes()


# ---------------------------------------------------------------------------
# Salvaged from deleted format-2 tests
# ---------------------------------------------------------------------------

def test_stored_self_volume_matches_direct_physical_formula():
    """From ``test_prep_stored_vaa.py`` (deleted: format 3 has one stored
    frame, so the vaa-frame option is gone)."""
    from roshambo3.prep.conformer import stored_self_volume
    from roshambo3.reference.overlap import ALPHA, PAIR_PREFACTOR
    coords = np.array([[0.123, -0.5, 0.0], [0.456, 0.1, 0.2]], np.float32)
    delta = coords.astype(np.float64)[:, None, :] - coords.astype(np.float64)[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    expected = float(PAIR_PREFACTOR * np.exp(-0.5 * ALPHA * d2).sum())
    assert stored_self_volume(coords) == expected


def test_core_preparation_does_not_import_server_dependencies(tmp_path):
    """From ``test_prepare_topology.py`` (deleted with the structures store)."""
    code = """
import sys
from roshambo3.prep import prepare
from rdkit import Chem
from rdkit.Chem import AllChem
m=Chem.AddHs(Chem.MolFromSmiles('CCO')); AllChem.EmbedMolecule(m,randomSeed=13)
c=prepare([m],sys.argv[1],workers=1)
assert c.n_mol == 1
assert not any(k.startswith(('fastapi','uvicorn','roshambo3.web')) for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "core")], check=True)


def test_binary_prefix(tmp_path):
    """From ``test_prepare_topology.py``: a ``max_records`` prefix of a
    binary stream builds exactly those records."""
    source = tmp_path / "prefix.bin.zst"
    write_source(source, fixtures(), True)
    prefix = prepare(source, str(tmp_path / "prefix"), workers=1, max_records=2)
    assert prefix.n_mol == 2 and prefix.n_conf == 2


@pytest.mark.parametrize("color", [False, True])
def test_many_atom_counts_do_not_retain_shard_descriptors(tmp_path, color):
    """From ``test_prepare_topology.py``. Writer-resource fixture only;
    synthetic coordinates are not chemistry evidence."""
    import resource
    import psutil
    from roshambo3.prep.build import _ShardSet
    from roshambo3.prep.sources import MoleculeRecord
    original_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    baseline = psutil.Process().num_fds()
    # Relative to what is already open: a fixed 64 fell below the ~96
    # descriptors a CUDA context leaves open. 32 spare descriptors are far
    # fewer than the 80 atom counts below, so a writer that kept one per atom
    # count still exhausts the limit.
    highest = max(int(fd) for fd in os.listdir("/dev/fd"))
    limit = highest + 1 + 32
    if original_limit[0] != resource.RLIM_INFINITY:
        limit = min(limit, original_limit[0])
    shard = None
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, original_limit[1]))
        shard = _ShardSet(str(tmp_path), 0, n_types=6 if color else None)
        for n in range(1, 81):
            rec = MoleculeRecord(str(n), n, {"smiles": "C" * n, "mw": 12.0 * n,
                                             "rotor_count": 0.0},
                                 [(np.zeros((n, 3)), n, 1.0)],
                                 features=[] if color else None)
            assert shard.add(rec)
            shard.flush()
            assert psutil.Process().num_fds() <= baseline + 3
        assert shard.close() == (80, 80)
        shard = None
    finally:
        if shard is not None:
            shard.close()
        resource.setrlimit(resource.RLIMIT_NOFILE, original_limit)


# ---------------------------------------------------------------------------
# prep.commit (review fix): a commit only for this checkout's clean builder
# ---------------------------------------------------------------------------

def test_prep_commit_names_only_a_clean_checkout_that_tracks_the_builder(tmp_path):
    import shutil
    from roshambo3.prep import build

    def git(*args, cwd):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=cwd, check=True, capture_output=True)

    # Not in any repository.
    plain = tmp_path / "plain" / "src" / "roshambo3" / "prep"
    plain.mkdir(parents=True)
    assert build._source_commit(str(plain)) is None
    # A repository that encloses the package without tracking it (a wheel
    # installed in a venv inside an unrelated checkout).
    other = tmp_path / "other"
    pkg = other / "venv" / "site-packages" / "roshambo3" / "prep"
    pkg.mkdir(parents=True)
    (pkg / "build.py").write_text("# installed\n")
    (other / "README").write_text("x\n")
    git("init", "-q", cwd=other)
    (other / ".gitignore").write_text("venv/\n")
    git("add", "README", ".gitignore", cwd=other)
    git("commit", "-q", "-m", "x", cwd=other)
    assert build._source_commit(str(pkg)) is None
    # This checkout's builder, clean: its HEAD; dirty or untracked in src: null.
    repo = tmp_path / "repo"
    prep = repo / "src" / "roshambo3" / "prep"
    prep.mkdir(parents=True)
    shutil.copy(build.__file__, prep / "build.py")
    git("init", "-q", cwd=repo)
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "x", cwd=repo)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert build._source_commit(str(prep)) == head
    (prep / "build.py").write_text((prep / "build.py").read_text() + "# edit\n")
    assert build._source_commit(str(prep)) is None
    git("checkout", "--", ".", cwd=repo)
    assert build._source_commit(str(prep)) == head
    (prep / "new.py").write_text("x = 1\n")
    assert build._source_commit(str(prep)) is None


def test_keep_temp_without_temp_dir_is_refused(tmp_path, sdf):
    """Review fix: the default scratch is ``<out>.partial/_tmp``, which the
    publishing rename would carry into the corpus."""
    with pytest.raises(ValueError, match="keep_temp needs temp_dir"):
        prepare([str(sdf)], str(tmp_path / "c"), keep_temp=True)
    assert not (tmp_path / "c").exists() and not (tmp_path / "c.partial").exists()
    corpus = prepare([str(sdf)], str(tmp_path / "c"), keep_temp=True,
                     temp_dir=str(tmp_path / "scratch"))
    assert sorted(os.listdir(corpus.root)) == ["build.json", "corpus.json", "groups",
                                               "molecules.parquet"]
    assert any((tmp_path / "scratch").iterdir())              # kept, outside the corpus


@pytest.mark.parametrize("memory", [True, False])
def test_the_id_gate_refusal_says_where_the_build_is(tmp_path, sdf, memory):
    """Review fix: "left at <staging>" only when the staging is kept (path
    input); an in-memory build's staging is removed."""
    out = tmp_path / "gated"
    with pytest.raises(BuildRefused) as e:
        prepare(fixtures() if memory else [str(sdf)], str(out), workers=1,
                expect_sha256_ids="0" * 64)
    partial = tmp_path / "gated.partial"
    if memory:
        assert "cannot be resumed, so nothing is kept" in str(e.value)
        assert "left at" not in str(e.value) and not partial.exists()
    else:
        assert f"left at {partial}" in str(e.value) and partial.is_dir()

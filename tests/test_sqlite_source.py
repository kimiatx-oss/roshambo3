"""The read-only SQLite conformer source (prep/sqlite_source.py).

Fixtures are built here with RDKit from synthetic molecules and synthetic
ids, in the PubChem3D and text-id layouts. A real PubChem slice under
``$ROSHAMBO3_CORPORA`` and a real text-id database named by
``ROSHAMBO3_TEXT_ID_SQLITE`` are used when present and skipped cleanly
otherwise. No test writes to, or scans, a real source.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import zstandard as zstd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from roshambo3.paths import corpora_root
from roshambo3.prep import prepare
from roshambo3.prep.conformer import stable_id, strip_hydrogens
from roshambo3.prep.sqlite_source import (
    DEFAULT_SEED, SqliteSource, detect_kind, hash_unit, hash_unit_scalar,
    open_read_only, splitmix64)

REAL_SLICE = corpora_root() / "_prep_subset" / "conf1_150k.sqlite"
#: A real text-id database, opt-in: unset, the test that reads it skips.
REAL_TEXT_ID = (Path(os.environ["ROSHAMBO3_TEXT_ID_SQLITE"])
                if os.environ.get("ROSHAMBO3_TEXT_ID_SQLITE") else None)

SMILES = ["CCO", "CCN", "CCCO", "CC(C)O", "CCOC", "c1ccccc1", "Cc1ccccc1",
          "CC(=O)O", "CC(=O)N", "OCCO", "CCCCN", "C1CCCCC1", "CC(C)(C)O",
          "CCOC(=O)C", "NCC(=O)O", "c1ccncc1", "Oc1ccccc1", "CCCCCCO",
          "CC(N)C(=O)O", "C1CCOC1", "CCS", "CCCl", "FC(F)F", "CC#N",
          "C[C@H](O)C(=O)O", "CCCC(=O)O", "c1ccc2ccccc2c1", "OC1CCCCC1",
          "CCN(CC)CC", "COc1ccccc1"]
N_MOLECULES = 300


def _write_pubchem(path: Path, records, with_meta=True):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE mols (molecule_id INTEGER PRIMARY KEY, molbinary_zst BLOB NOT NULL)")
    if with_meta:
        con.execute("CREATE TABLE meta (molecule_id INTEGER PRIMARY KEY, hac INTEGER NOT NULL, "
                    "mw REAL NOT NULL, smiles TEXT NOT NULL, rotor_count REAL NOT NULL, "
                    "atom_def_stereo INTEGER NOT NULL, atom_udef_stereo INTEGER NOT NULL, "
                    "bond_def_stereo INTEGER NOT NULL, bond_udef_stereo INTEGER NOT NULL)")
        con.execute("CREATE INDEX molecules_hac ON meta(hac)")
    c = zstd.ZstdCompressor()
    for mid, mol in records:
        # Like PubChem3D: a property-free binary carrying every conformer.
        con.execute("INSERT INTO mols VALUES (?, ?)", (mid, c.compress(mol.ToBinary())))
        if with_meta:
            con.execute("INSERT INTO meta VALUES (?,?,?,?,?,0,0,0,0)", (
                mid, strip_hydrogens(mol).GetNumAtoms(), Descriptors.MolWt(mol),
                Chem.MolToSmiles(Chem.RemoveHs(mol)),
                float(rdMolDescriptors.CalcNumRotatableBonds(mol))))
    con.commit()
    con.close()


@pytest.fixture(scope="module")
def pubchem(tmp_path_factory):
    """A few hundred tiny molecules, 1-3 conformers each, at sparse ids."""
    rng = np.random.default_rng(2008)
    ids = np.cumsum(rng.integers(1, 40, size=N_MOLECULES)) + 1000
    records = []
    for i, mid in enumerate(ids.tolist()):
        mol = Chem.AddHs(Chem.MolFromSmiles(SMILES[i % len(SMILES)]))
        n = 1 + i % 3
        got = AllChem.EmbedMultipleConfs(mol, numConfs=n, randomSeed=int(mid))
        assert len(got) == n
        records.append((mid, mol))
    root = tmp_path_factory.mktemp("sqlite_source")
    full, bare = root / "pubchem.sqlite", root / "pubchem_mols_only.sqlite"
    _write_pubchem(full, records)
    _write_pubchem(bare, records, with_meta=False)
    return {"path": full, "bare": bare, "records": dict(records),
            "n_conformers": sum(m.GetNumConformers() for _, m in records)}


@pytest.fixture(scope="module")
def text_id(tmp_path_factory):
    path = tmp_path_factory.mktemp("text_id") / "text_id.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE mols (compound_id TEXT PRIMARY KEY, mol_blob BLOB NOT NULL, "
                "n_confs INTEGER NOT NULL, source TEXT NOT NULL)")
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
    con.executemany("INSERT INTO meta VALUES (?, ?)",
                    [("blob_compression", "zstd"), ("zstd_level", "19")])
    c = zstd.ZstdCompressor(level=19)
    names = []
    for i in range(60):
        mol = Chem.AddHs(Chem.MolFromSmiles(SMILES[i % len(SMILES)]))
        n = 1 + i % 3
        AllChem.EmbedMultipleConfs(mol, numConfs=n, randomSeed=100 + i)
        cid = f"CMPD-{900000 - 7 * i:09d}"     # inserted out of key order
        names.append(cid)
        con.execute("INSERT INTO mols VALUES (?,?,?,?)",
                    (cid, c.compress(mol.ToBinary()), n, "shard_0000.sdf.zst"))
    con.commit()
    con.close()
    return {"path": path, "names": names}


def emitted(source, records_per_unit=1000, failures=None):
    """Every unit's records, in unit order. ``failures`` collects each unit's
    rejects: the source resets its list per unit, as the builder expects."""
    out = []
    for unit in source.units(records_per_unit):
        out.extend(source.iter_unit(unit))
        if failures is not None:
            failures.extend(source.failures)
    return out


def fingerprint(records):
    return [(r.ext_id, r.mol_id, tuple(np.asarray(c[0]).tobytes() for c in r.conformers),
             tuple(float(c[2]) for c in r.conformers)) for r in records]


def file_state(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns, hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- the sampler ---------------------------------------------------------

#: Computed from ROSHAMBO2's tools/rsb3/hashsample.py (the hash the benchmark
#: tiers and query panels were drawn with), not from the code under test.
GOLDEN = [  # (key, h(key, DEFAULT_SEED), h(key, seed=7))
    (1, 0.7475982482104084, 0.7398170143550824),
    (2000001, 0.9371086130184861, 0.11533698361219769),
    (2204466, 0.8814480828361914, 0.8402608912101461),
    (50102162, 0.06590336078381209, 0.6738448164806127),
    (100000000, 0.41188741575163124, 0.33334871519981757),
    (stable_id("CMPD-000655445"), 0.4630714605199435, 0.5656150111556099),
    (stable_id("vendor-x17-0042"), 0.4217193070897618, 0.17888815164425276),
]


def test_hash_matches_the_legacy_hash_line():
    assert DEFAULT_SEED == 353584562739            # recorded in every tier manifest
    for key, h_default, h_seven in GOLDEN:
        assert hash_unit_scalar(key) == h_default
        assert hash_unit_scalar(key, 7) == h_seven
        assert float(hash_unit([key])[0]) == h_default
        assert float(hash_unit([key], 7)[0]) == h_seven


def test_hash_scalar_equals_vector_and_is_in_unit_interval():
    rng = np.random.default_rng(1)
    keys = np.concatenate([np.arange(0, 2000), rng.integers(0, 2**63 - 1, 2000)])
    vec = hash_unit(keys, 12345)
    assert np.all((vec >= 0.0) & (vec < 1.0))
    assert [hash_unit_scalar(int(k), 12345) for k in keys] == vec.tolist()
    # splitmix64 of 0 is the published first output of the generator seeded 0.
    assert int(splitmix64(np.uint64(0))) == 0xE220A8397B1DCDAF


@pytest.mark.parametrize("p_lo,p_hi", [(0.0, 0.01), (0.25, 0.5), (0.7545, 1.0), (0.0, 1.0)])
def test_sampler_fraction_within_binomial_bound(p_lo, p_hi):
    n = 1_000_000
    h = hash_unit(np.arange(1, n + 1, dtype=np.int64))
    k = int(np.count_nonzero((h >= p_lo) & (h < p_hi)))
    p = p_hi - p_lo
    assert abs(k - n * p) <= 5 * math.sqrt(n * p * (1 - p)) + 1e-9


def test_sampler_partitions_the_line_and_is_reproducible(pubchem):
    full = {r.ext_id for r in emitted(SqliteSource(pubchem["path"]))}
    parts = []
    for lo, hi in [(0.0, 0.3), (0.3, 0.31), (0.31, 1.0)]:
        a = fingerprint(emitted(SqliteSource(pubchem["path"], p_lo=lo, p_hi=hi)))
        b = fingerprint(emitted(SqliteSource(pubchem["path"], p_lo=lo, p_hi=hi), 7))
        assert a == b
        ids = {x[0] for x in a}
        expect = {e for e in full if lo <= hash_unit_scalar(int(e)) < hi}
        assert ids == expect
        parts.append(ids)
    assert set().union(*parts) == full and sum(map(len, parts)) == len(full)


# -- identity, order and determinism --------------------------------------

def test_ext_id_is_the_molecule_id(pubchem):
    recs = emitted(SqliteSource(pubchem["path"]))
    assert [r.ext_id for r in recs] == [str(m) for m in sorted(pubchem["records"])]
    for r in recs:
        assert r.mol_id == stable_id(r.ext_id)
        assert len(r.conformers) == pubchem["records"][int(r.ext_id)].GetNumConformers()
    assert sum(len(r.conformers) for r in recs) == pubchem["n_conformers"]


def test_order_and_set_do_not_depend_on_unit_size(pubchem):
    src = SqliteSource(pubchem["path"])
    ref = fingerprint(emitted(src, 1000))
    for rpu in (1, 4, 37):
        units = src.units(rpu)
        assert len(units) > 1
        assert all(isinstance(u, tuple) and len(pickle.dumps(u)) < 300 for u in units)
        assert fingerprint(emitted(src, rpu)) == ref


@pytest.mark.parametrize("p_lo,p_hi", [(0.1, 0.6), (0.2, 0.23)])  # joined scan / id lookups
def test_both_read_paths_and_both_tables_agree(pubchem, p_lo, p_hi):
    a = fingerprint(emitted(SqliteSource(pubchem["path"], p_lo=p_lo, p_hi=p_hi), 9))
    b = fingerprint(emitted(SqliteSource(pubchem["bare"], p_lo=p_lo, p_hi=p_hi), 9))
    assert a and a == b


def test_hac_window_pre_and_post_decode_agree(pubchem):
    kw = dict(p_lo=0.0, p_hi=0.8, hac_lo=4, hac_hi=6)
    with_meta = SqliteSource(pubchem["path"], **kw)
    bare = SqliteSource(pubchem["bare"], **kw)
    assert with_meta.hac_source == "meta.hac" and bare.hac_source == "prepared_n_real"
    a, b = emitted(with_meta), emitted(bare)
    assert fingerprint(a) == fingerprint(b)
    assert a and all(4 <= r.conformers[0][1] <= 6 for r in a)
    expect = {str(m) for m, mol in pubchem["records"].items()
              if 4 <= strip_hydrogens(mol).GetNumAtoms() <= 6
              and hash_unit_scalar(m) < 0.8}
    assert {r.ext_id for r in a} == expect


# -- read-only -----------------------------------------------------------

def test_source_is_read_only_and_unchanged(pubchem, tmp_path):
    path = tmp_path / "copy.sqlite"
    shutil.copy2(pubchem["path"], path)
    before = file_state(path)
    listing = sorted(os.listdir(tmp_path))
    con = open_read_only(str(path))
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO mols VALUES (1, x'00')")
    with pytest.raises(sqlite3.OperationalError):
        con.execute("CREATE TABLE scratch (x)")
    con.close()
    src = SqliteSource(path, p_lo=0.0, p_hi=0.5, hac_lo=1, hac_hi=8)
    emitted(src, 11)
    src.describe()
    assert file_state(path) == before
    assert sorted(os.listdir(tmp_path)) == listing      # no -wal / -shm / -journal


def test_uncheckpointed_wal_is_refused(pubchem, tmp_path):
    path = tmp_path / "walled.sqlite"
    shutil.copy2(pubchem["path"], path)
    Path(str(path) + "-wal").write_bytes(b"\0" * 32)
    with pytest.raises(ValueError, match="uncheckpointed"):
        SqliteSource(path)
    Path(str(path) + "-wal").write_bytes(b"")
    assert SqliteSource(path).wal_bytes == 0


def test_source_changed_under_reader_is_detected(pubchem, tmp_path):
    path = tmp_path / "moving.sqlite"
    shutil.copy2(pubchem["path"], path)
    src = SqliteSource(path)
    unit = src.units()[0]
    os.utime(path, ns=(0, 0))
    with pytest.raises(RuntimeError, match="changed"):
        next(src.iter_unit(unit))


def test_describe_records_the_tier(pubchem):
    src = SqliteSource(pubchem["path"], seed=7, p_lo=0.1, p_hi=0.4, hac_lo=3, hac_hi=9)
    d = json.loads(json.dumps(src.describe()))
    st = os.stat(pubchem["path"])
    assert d["kind"] == "sqlite:pubchem" and d["path"] == str(pubchem["path"])
    assert d["bytes"] == st.st_size and d["mtime"] == st.st_mtime
    assert d["sampler"] == {"hash": "splitmix64(key ^ seed) >> 11, / 2**53", "seed": 7,
                            "p_lo": 0.1, "p_hi": 0.4, "hac_window": [3, 9],
                            "hac_window_inclusive": True, "hac_source": "meta.hac"}
    assert d["table_counts"] == {"meta": N_MOLECULES, "mols": N_MOLECULES}
    assert d["key_min"] == min(pubchem["records"]) and d["key_max"] == max(pubchem["records"])
    assert d["complete"] is False and d["key_space"] == "molecule_id"
    assert SqliteSource(pubchem["path"]).describe()["complete"] is True


def test_row_count_hint_skips_counting(pubchem):
    src = SqliteSource(pubchem["path"], n_rows=40)
    units = src.units(10)
    assert src.describe()["table_counts"] == {} and src.describe()["n_rows_hint"] == 40
    assert src.stats["source_rows_estimate"] == 40 and len(units) == 4
    assert fingerprint(emitted(src, 10)) == fingerprint(emitted(SqliteSource(pubchem["path"])))


def test_source_pickles_without_a_connection(pubchem):
    src = SqliteSource(pubchem["path"], p_hi=0.5)
    units = src.units(50)
    clone = pickle.loads(pickle.dumps(src))
    assert fingerprint(clone.iter_unit(units[0])) == fingerprint(src.iter_unit(units[0]))


# -- text-id ---------------------------------------------------------------

def test_text_id_layout(text_id):
    assert detect_kind(text_id["path"]) == "text-id"
    src = SqliteSource(text_id["path"])
    recs = emitted(src, 7)
    assert [r.ext_id for r in recs] == text_id["names"]                 # rowid order
    assert all(r.mol_id == stable_id(r.ext_id) for r in recs)
    assert [len(r.conformers) for r in recs] == [1 + i % 3 for i in range(60)]
    part = {r.ext_id for r in emitted(SqliteSource(text_id["path"], p_lo=0.5, p_hi=1.0))}
    assert part == {n for n in text_id["names"] if hash_unit_scalar(stable_id(n)) >= 0.5}
    assert src.describe()["key_space"] == "stable_id(compound_id)"


def test_text_id_requires_declared_zstd(text_id, tmp_path):
    path = tmp_path / "undeclared.db"
    shutil.copy2(text_id["path"], path)
    con = sqlite3.connect(path)
    con.execute("UPDATE meta SET value = 'lz4' WHERE key = 'blob_compression'")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="blob_compression"):
        SqliteSource(path)


# -- the builder end to end ---------------------------------------------------

def _build(monkeypatch):
    """Record the SqliteSource that prepare() really constructs.

    The dispatch is the builder's own (``prep.build._open_source``); this only
    wraps the class so a test can see the instance the workers were given.
    """
    import roshambo3.prep.sqlite_source as module

    made = []
    real_init = module.SqliteSource.__init__

    # Wrap __init__ rather than substitute the class: pickle finds the class
    # by module and name, so under spawn (macOS's default) the workers must
    # receive the real SqliteSource, and a local subclass cannot be pickled.
    def recording(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(module.SqliteSource, "__init__", recording)
    return made


def _build_record(root) -> dict:
    import json
    return json.loads((Path(root) / "build.json").read_text())


def test_prepare_builds_a_corpus_from_sqlite(pubchem, tmp_path, monkeypatch):
    """Format 3: the vaa-frame option is gone (one stored frame) and
    so is the topology capture; two worker counts give byte-identical files."""
    from tests.format3_rows import read_corpus

    built = {}
    for workers, rpu in ((1, 1000), (2, 13)):
        made = _build(monkeypatch)
        out = tmp_path / f"w{workers}"
        built[workers] = prepare([str(pubchem["path"])], str(out), workers=workers,
                                 records_per_unit=rpu, max_conf_per_group=64)
        assert len(made) == 1
    a, b = built[1], built[2]
    _assert_payloads_equal(a, b)
    rows = read_corpus(a.root, merged=False)
    assert rows.n_conf == pubchem["n_conformers"]
    assert rows.n_mol == N_MOLECULES
    assert rows.build["rejected"]["n"] == 0
    meta = pq.read_table(Path(a.root) / "molecules.parquet").to_pydict()
    assert sorted(meta["ext_id"]) == sorted(str(m) for m in pubchem["records"])
    assert all(mid == stable_id(e) for mid, e in zip(meta["mol_id"], meta["ext_id"]))
    assert set(rows.mol_id.tolist()) == set(meta["mol_id"])


def test_prepare_with_sampler_and_color_under_forkserver(pubchem, tmp_path, monkeypatch):
    """Workers that do not inherit memory: the source arrives pickled."""
    import multiprocessing

    import roshambo3.prep.build as build
    from roshambo3.pharmacophore import PharmacophoreGenerator

    monkeypatch.setattr(build, "SCATTER_MP_CONTEXT", multiprocessing.get_context("forkserver"))
    corpus = prepare([str(pubchem["path"])], str(tmp_path / "c"), workers=2,
                     records_per_unit=17, color_generator=PharmacophoreGenerator(),
                     sqlite_source={"p_lo": 0.2, "p_hi": 0.7})
    expect = {str(m) for m in pubchem["records"] if 0.2 <= hash_unit_scalar(m) < 0.7}
    meta = pq.read_table(Path(corpus.root) / "molecules.parquet").to_pydict()
    assert set(meta["ext_id"]) == expect
    assert corpus.has_color
    assert _build_record(corpus.root)["source"]["sampler"]["p_lo"] == 0.2


def _assert_payloads_equal(left, right):
    from tests.prep_fixtures import assert_same_rows, corpus_rows
    assert_same_rows(corpus_rows(left), corpus_rows(right))


# -- real sources (skipped when absent) -----------------------------------

@pytest.mark.skipif(not REAL_SLICE.exists(), reason="150K PubChem slice not present")
def test_real_pubchem_slice_sample():
    """~1% of the real slice (mols only, no meta), checked against hash sampling."""
    src = SqliteSource(REAL_SLICE, p_lo=0.0, p_hi=0.01)
    assert src.kind == "pubchem" and not src.has_meta
    failures: list = []
    recs = emitted(src, 20_000, failures)
    con = open_read_only(str(REAL_SLICE))
    ids = np.array([r[0] for r in con.execute("SELECT molecule_id FROM mols")], np.int64)
    con.close()
    expect = ids[hash_unit(ids) < 0.01]
    got = [int(r.ext_id) for r in recs]
    rejected = {name for name, _ in failures}
    assert got == [int(i) for i in expect if str(i) not in rejected]
    assert 0 < len(got) and abs(len(expect) - 0.01 * ids.size) <= 5 * math.sqrt(ids.size * 0.0099)
    assert fingerprint(emitted(src, 300)) == fingerprint(recs)


@pytest.mark.skipif(REAL_TEXT_ID is None or not REAL_TEXT_ID.exists(),
                    reason="set ROSHAMBO3_TEXT_ID_SQLITE to a text-id database to run this")
def test_real_text_id_point_reads():
    """Three rows by rowid: the documented zstd multi-conformer binaries."""
    src = SqliteSource(REAL_TEXT_ID)
    assert src.kind == "text-id" and src.blob_compression == "zstd"
    recs = list(src.iter_unit((str(REAL_TEXT_ID), 1, 3)))
    con = open_read_only(str(REAL_TEXT_ID))
    rows = con.execute("SELECT compound_id, n_confs FROM mols WHERE rowid BETWEEN 1 AND 3 "
                       "ORDER BY rowid").fetchall()
    con.close()
    assert [r.ext_id for r in recs] == [cid for cid, _ in rows]
    assert [len(r.conformers) for r in recs] == [n for _, n in rows]
    assert all(r.mol_id == stable_id(r.ext_id) for r in recs)


# -- the real dispatch in prepare(): no monkeypatching --------------------------

def _expected_ids(pubchem, p_lo, p_hi):
    keys = np.array(sorted(pubchem["records"]), dtype=np.uint64)
    h = hash_unit(keys)
    return {str(int(k)) for k, v in zip(keys, h) if p_lo <= v < p_hi}


def test_prepare_dispatches_sqlite_and_records_the_tier(pubchem, tmp_path):
    out = tmp_path / "tier"
    prepare([str(pubchem["path"])], str(out), workers=1, max_conf_per_group=64,
            input_format="sqlite:pubchem", sqlite_source={"p_lo": 0.0, "p_hi": 0.5})
    src = _build_record(out)["source"]
    assert src["kind"] == "sqlite:pubchem" and src["sampler"]["p_hi"] == 0.5
    assert src["sampler"]["seed"] == DEFAULT_SEED and src["complete"] is False
    meta = pq.read_table(out / "molecules.parquet").to_pydict()
    assert set(meta["ext_id"]) == _expected_ids(pubchem, 0.0, 0.5)


def test_prepare_detects_a_sqlite_extension_under_auto(pubchem, tmp_path):
    path = pubchem["path"]
    assert str(path).lower().endswith((".sqlite", ".sqlite3", ".db"))
    out = tmp_path / "auto"
    corpus = prepare([str(path)], str(out), workers=1, max_conf_per_group=64)
    src = _build_record(out)["source"]
    assert src["kind"].startswith("sqlite:")
    assert corpus.n_mol == N_MOLECULES
    assert src["complete"] is True


def test_prepare_refuses_sampler_options_for_other_inputs(tmp_path):
    sdf = tmp_path / "x.sdf"
    sdf.write_text("")
    with pytest.raises(ValueError, match="SQLite"):
        prepare([str(sdf)], str(tmp_path / "o"), sqlite_source={"p_hi": 0.5})
    with pytest.raises(ValueError, match="max_records"):
        prepare([str(sdf)], str(tmp_path / "o2"), input_format="sqlite", max_records=5)


def test_dry_run_for_sqlite_reads_one_range_and_writes_nothing(pubchem, tmp_path):
    out = tmp_path / "never"
    report = prepare([str(pubchem["path"])], str(out), workers=1, dry_run=True,
                     sqlite_source={"p_lo": 0.0, "p_hi": 0.5, "n_rows": N_MOLECULES})
    assert not out.exists() and not Path(str(out) + ".partial").exists()
    assert report["wrote_nothing"] is True
    assert report["records"] == round(N_MOLECULES * 0.5)
    assert report["records_basis"].startswith("expected")
    full = pubchem["path"].stat().st_size
    assert report["bytes_to_read"] == int(full * 0.5)
    assert report["sample"]["molecules"] > 0 and report["conformers_estimate"] > 0


def test_resume_identity_covers_the_sqlite_file(pubchem):
    from roshambo3.prep.build import _input_paths

    source = SqliteSource(str(pubchem["path"]))
    assert _input_paths(source.describe()) == [str(pubchem["path"])]

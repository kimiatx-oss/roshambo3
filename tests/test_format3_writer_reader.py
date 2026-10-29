"""Format 3: the production writer and reader, the scan cache and ``verify``.

* the conformance corpus read back through :mod:`.reader` matches the
  committed decoded-array digests, through every available decode path;
* :func:`writer.write_corpus` rewrites it byte-identical to
  ``reference.write_corpus`` and to the committed files;
* randomized synthetic corpora (widths 0..16, partial final BP256 groups,
  ``n_real`` 1..255, empty color) round-trip exactly;
* the scan cache, the refusal of an unscanned corpus for screening, and the
  ``roshambo3 verify`` command.

Synthetic corpora are written to ``tmp_path``, never into the tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil

import numpy as np
import pytest

from roshambo3.format.r3g import cache, codec, reference as R, writer as W
from roshambo3.format.r3g.errors import SCAN_REQUIRED, CorpusValidationError
from roshambo3.format.r3g.reader import open_corpus, verify

DATA = pathlib.Path(__file__).resolve().parent / "data" / "format3"
EXPECTED = json.loads((DATA / "EXPECTED.json").read_text())


def _digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@pytest.fixture
def scanned(tmp_path):
    """A writable copy of a conformance corpus, scanned into a private cache."""
    def make(variant):
        root = tmp_path / variant
        shutil.copytree(DATA / variant, root)
        verify(str(root), "scan", cache_dir=str(tmp_path / "cache"))
        return root, str(tmp_path / "cache")
    return make


def _backends():
    out = ["reference"]
    os.environ.pop("ROSHAMBO3_R3G_CODEC", None)
    codec._native.cache_clear()
    if codec.backend() == "native":
        out.append("native")
    return out


@pytest.fixture(params=_backends())
def backend(request, monkeypatch):
    if request.param == "reference":
        monkeypatch.setenv("ROSHAMBO3_R3G_CODEC", "reference")
    else:
        monkeypatch.delenv("ROSHAMBO3_R3G_CODEC", raising=False)
    codec._native.cache_clear()
    assert codec.backend() == request.param
    yield request.param
    codec._native.cache_clear()


# ---------------------------------------------------------------------------
# The conformance corpus through the reader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["shape", "color"])
def test_reader_matches_the_committed_digests(variant, scanned, backend):
    root, cdir = scanned(variant)
    exp = EXPECTED[variant]
    c = open_corpus(str(root), cache_dir=cdir)
    assert c.fingerprint == exp["fingerprint"] and c.scanned
    assert len(c) == len(exp["groups"])
    for g in c.groups():
        name = R.group_file_name(g.bucket, g.group)
        e = exp["groups"][name]
        assert _digest(g.decode_all()) == e["heavy_sha256"], name
        assert _digest(g.counts) == e["counts_sha256"], name
        assert _digest(g.runs) == e["runs_sha256"] and _digest(g.vaa) == e["vaa_sha256"]
        assert [int(b["rows"]) for b in g.blocks] == e["block_rows"]
        assert not g.vaa.flags.writeable and not g.runs.flags.writeable
        nr = g.n_real
        assert nr.shape == (g.n_conf,) and np.array_equal(nr, g.counts[:, 0])
        if variant == "color":
            for k in ("feature_offsets", "feature_types", "feature_atom_offsets", "feature_atoms"):
                assert _digest(g.section(k)) == e[f"{k}_sha256"], (name, k)
            assert _digest(g.color_self) == e["color_self_sha256"]
            assert np.array_equal(g.n_color, g.counts[:, 1:].sum(axis=1))
        else:
            assert g.color_self is None and g.feature_types is None
            assert g.counts.shape == (g.n_conf, 1)


def test_block_directory(scanned):
    root, cdir = scanned("color")
    c = open_corpus(str(root), cache_dir=cdir)
    d = c.block_directory()
    assert d.size == sum(len(g["block_rows"]) for g in EXPECTED["color"]["groups"].values())
    assert int(d["rows"].sum()) == c.n_conf
    e = c.block_directory("expensive_first")
    assert np.all(np.diff(e["cost"]) <= 0)
    same = np.diff(e["cost"]) == 0
    key = e["group"].astype(np.int64) * 10**6 + e["block"]
    assert np.all(np.diff(key)[same] > 0)


# ---------------------------------------------------------------------------
# Writer: byte identity with the reference and the committed files
# ---------------------------------------------------------------------------

def molecules_from(root: str, cdir: str) -> tuple[dict, list]:
    """Rebuild writer input from a corpus through the production reader."""
    c = open_corpus(root, cache_dir=cdir)
    meta = {}
    for t in c.molecules():
        d = t.to_pydict()
        for i, e, s, mw, ro in zip(d["mol_id"], d["ext_id"], d["smiles"], d["mw"], d["rotors"]):
            meta[i] = (e, s, mw, ro)
    out = []
    for g in c.groups():
        heavy = g.decode_all()
        rs = g.run_start
        for m in range(g.n_mol):
            mid, nc, nr = int(g.mol_id[m]), int(g.molecule_n_conf[m]), int(g.molecule_n_real[m])
            feats = []
            if c.has_color:
                fo, fao = g.feature_offsets, g.feature_atom_offsets
                for f in range(int(fo[m]), int(fo[m + 1])):
                    feats.append((int(g.feature_types[f]),
                                  tuple(int(a) for a in g.feature_atoms[int(fao[f]):int(fao[f + 1])])))
            r0, r1 = int(rs[m]), int(rs[m + 1])
            e, s, mw, ro = meta[mid]
            out.append(W.Molecule(mid, heavy[r0:r1, :, :nr].copy(), np.array(g.vaa[r0:r1]), feats,
                                  None if g.color_self is None else np.array(g.color_self[r0:r1]),
                                  e, s, mw, ro))
    return c.descriptor, out


@pytest.mark.parametrize("variant", ["shape", "color"])
def test_writer_is_byte_identical_to_the_reference_and_the_committed_corpus(
        variant, scanned, tmp_path, backend):
    root, cdir = scanned(variant)
    d, mols = molecules_from(str(root), cdir)
    build = json.loads((DATA / variant / "build.json").read_bytes())
    kw = dict(color=d.get("color"), prep_version=d["prep"]["version"],
              prep_commit=d["prep"]["commit"], block_rows=build["block_rows"],
              max_conf_per_group=build["max_conf_per_group"], build=build)
    mine, ref = tmp_path / "mine", tmp_path / "ref"
    W.write_corpus(str(mine), mols, **kw)
    R.write_corpus(str(ref), [R.Molecule(**m.__dict__) for m in mols], **kw)
    exp = EXPECTED[variant]
    for name, e in exp["groups"].items():
        assert _sha(mine / name) == _sha(ref / name) == e["file_sha256"], name
    assert _sha(mine / "corpus.json") == _sha(ref / "corpus.json") == _sha(DATA / variant / "corpus.json")
    assert (mine / "build.json").read_bytes() == (DATA / variant / "build.json").read_bytes()
    # molecules.parquet: identical between the two writers; identical to the
    # committed file except the footer's created_by (the pyarrow release)
    assert _sha(mine / "molecules.parquet") == _sha(ref / "molecules.parquet")
    import pyarrow
    if pyarrow.__version__ == EXPECTED["provenance"].get("pyarrow"):
        assert _sha(mine / "molecules.parquet") == _sha(DATA / variant / "molecules.parquet")


def test_native_and_reference_encoders_write_identical_group_files(tmp_path, monkeypatch):
    """Runs only when the extension provides the native codec."""
    codec._native.cache_clear()
    if codec.backend() != "native":
        pytest.skip("the native format-3 codec is not in this build")
    mols = _random_molecules(np.random.default_rng(5), 40, color=True)
    W.write_corpus(str(tmp_path / "n"), mols, color=_color(), prep_version="t", prep_commit=None,
                   block_rows=7)
    monkeypatch.setenv("ROSHAMBO3_R3G_CODEC", "reference")
    codec._native.cache_clear()
    W.write_corpus(str(tmp_path / "r"), mols, color=_color(), prep_version="t", prep_commit=None,
                   block_rows=7)
    codec._native.cache_clear()
    for p in sorted((tmp_path / "r" / "groups").iterdir()):
        assert p.read_bytes() == (tmp_path / "n" / "groups" / p.name).read_bytes(), p.name


# ---------------------------------------------------------------------------
# Randomized synthetic corpora
# ---------------------------------------------------------------------------

TYPES = ["Donor", "Acceptor", "Aromatic"]
FDEF = ("AtomType C [#6]\n"
        + "".join(f"DefineFeature F{t} [{{C}}]\n  Family {t}\n  Weights 1.0\nEndFeature\n"
                  for t in TYPES))


def _color(off_diagonal: float = 0.0):
    n = len(TYPES)
    r = np.zeros((n + 1, n + 1), np.float32)
    for t in range(1, n + 1):
        r[t, t] = 1.0
    p = r.copy()
    p[1, 2] = off_diagonal
    return R.color_descriptor(generator="rdkit_fdef", definition=FDEF, types=TYPES, r=r, p=p,
                              rdkit_version="x")


def _random_molecules(rng, n, *, color: bool, width=None, empty_color=False):
    """Molecules over many buckets; with ``width`` every coordinate is in
    ``[lo, lo + 2^width)`` for one ``lo``, else each molecule draws its own
    width and ``lo``; random feature tables (none if empty_color)."""
    mols = []
    if width is not None:
        lo_all = int(rng.integers(-32768, 32768 - (1 << width) + 1)) if width < 16 else -32768
    for i in range(n):
        n_real = int(rng.choice([1, 2, 7, 11, 12, 30, 63, 64, 100, 200, 255]))
        n_conf = int(rng.integers(1, 6))
        w = int(rng.integers(0, 17)) if width is None else width
        lo = (int(rng.integers(-32768, 32768 - (1 << w) + 1)) if w < 16 else -32768) \
            if width is None else lo_all
        xyz = (lo + rng.integers(0, 1 << w, (n_conf, 3, n_real))).astype(np.int16)
        feats = []
        if color and not empty_color:
            k = int(rng.integers(0, min(40, 512 - n_real) + 1))
            for _ in range(k):
                na = int(rng.integers(1, min(16, n_real) + 1))
                feats.append((int(rng.integers(1, len(TYPES) + 1)),
                              tuple(rng.choice(n_real, na, replace=False).tolist())))
        mols.append(W.Molecule(
            mol_id=int(rng.integers(0, 1 << 63)), coords=xyz,
            vaa=rng.uniform(1, 500, n_conf).astype(np.float32), features=feats,
            color_self=rng.uniform(0, 50, n_conf) if color else None,
            ext_id=f"mol-{i}-é", smiles="C" * max(1, n_real % 50), mw=float(i),
            rotors=int(rng.integers(0, 9))))
    return mols


def _read_back(root, cdir):
    verify(str(root), "scan", cache_dir=cdir)
    c = open_corpus(str(root), cache_dir=cdir)
    got = {}
    for g in c.groups():
        heavy = g.decode_all()
        rs = g.run_start
        for m in range(g.n_mol):
            r0, r1 = int(rs[m]), int(rs[m + 1])
            nr = int(g.molecule_n_real[m])
            feats = []
            if c.has_color:
                fo, fao = g.feature_offsets, g.feature_atom_offsets
                for f in range(int(fo[m]), int(fo[m + 1])):
                    feats.append((int(g.feature_types[f]), tuple(
                        int(a) for a in g.feature_atoms[int(fao[f]):int(fao[f + 1])])))
            got[int(g.mol_id[m])] = (heavy[r0:r1, :, :nr], np.array(g.vaa[r0:r1]), feats,
                                     None if g.color_self is None else np.array(g.color_self[r0:r1]))
            assert not heavy[r0:r1, :, nr:].any()                  # padding is zero
    return c, got


@pytest.mark.parametrize("width", [*range(17), None])
def test_random_round_trip_every_width(width, tmp_path):
    rng = np.random.default_rng(100 + (width if width is not None else 99))
    color = width is None or width % 2 == 0
    mols = _random_molecules(rng, 12, color=color, width=width)
    root = tmp_path / "c"
    W.write_corpus(str(root), mols, color=_color() if color else None, prep_version="t",
                   prep_commit=None, block_rows=int(rng.integers(1, 9)), max_conf_per_group=13)
    c, got = _read_back(root, str(tmp_path / "cache"))
    widths = set()
    for g in c.groups():
        for i in range(g.n_blocks):
            widths.update(codec.block_widths(bytes(g.block_bytes(i))))
    if width is not None:
        assert width in widths and max(widths) == width
    for m in mols:
        xyz, vaa, feats, cs = got[m.mol_id]
        assert np.array_equal(xyz, m.coords)
        assert np.array_equal(vaa, m.vaa)
        assert feats == R.canonical_feature_order(m.features if color else [])
        if color:
            assert np.array_equal(cs, m.color_self)


def test_partial_and_full_bp256_groups(tmp_path):
    """Streams of 255, 256, 257 and 511 values (tails of 255, 0, 1, 255)."""
    mols = []
    for i, (n_conf, n_real) in enumerate([(1, 255), (16, 16), (1, 257 - 256), (7, 73)]):
        rng = np.random.default_rng(i)
        mols.append(W.Molecule(mol_id=i + 1, coords=rng.integers(-500, 500, (n_conf, 3, n_real),
                                                                   dtype=np.int16),
                               vaa=np.ones(n_conf, np.float32), ext_id=f"m{i}", smiles="C"))
    W.write_corpus(str(tmp_path / "c"), mols, color=None, prep_version="t", prep_commit=None)
    _, got = _read_back(tmp_path / "c", str(tmp_path / "cache"))
    for m in mols:
        assert np.array_equal(got[m.mol_id][0], m.coords)


def test_empty_color(tmp_path):
    """A color corpus in which no molecule has a feature: zero-length feature
    sections are legal (section 3.2) and round-trip."""
    mols = _random_molecules(np.random.default_rng(3), 6, color=True, empty_color=True)
    for m in mols:
        m.color_self = np.zeros(m.n_conf)
    W.write_corpus(str(tmp_path / "c"), mols, color=_color(), prep_version="t", prep_commit=None)
    c, got = _read_back(tmp_path / "c", str(tmp_path / "cache"))
    for g in c.groups():
        assert g.feature_types.size == 0 and g.feature_atoms.size == 0
        assert np.array_equal(g.counts[:, 1:], np.zeros((g.n_conf, len(TYPES)), np.uint8))
        assert g.L == g.L_heavy
    assert all(got[m.mol_id][2] == [] for m in mols)


def test_prefix_features_are_in_canonical_order(tmp_path):
    """Section 8.1 rule 4 is lexicographic: (t, (0,)) sorts before (t, (0, 1))."""
    m = W.Molecule(mol_id=1, coords=np.zeros((1, 3, 4), np.int16), vaa=np.ones(1, np.float32),
                   features=[(1, (0, 1)), (1, (0,)), (2, (1, 2, 3)), (2, (1, 2))],
                   color_self=np.zeros(1), ext_id="a", smiles="CCCC")
    W.write_corpus(str(tmp_path / "c"), [m], color=_color(), prep_version="t", prep_commit=None)
    _, got = _read_back(tmp_path / "c", str(tmp_path / "cache"))
    assert got[1][2] == [(1, (0,)), (1, (0, 1)), (2, (1, 2)), (2, (1, 2, 3))]


def test_n_real_1_to_255(tmp_path):
    rng = np.random.default_rng(11)
    mols = [W.Molecule(mol_id=n, coords=rng.integers(-9, 9, (1, 3, n), dtype=np.int16),
                       vaa=np.ones(1, np.float32), ext_id=f"m{n}", smiles="C")
            for n in range(1, 256)]
    W.write_corpus(str(tmp_path / "c"), mols, color=None, prep_version="t", prep_commit=None,
                   block_rows=50)
    c, got = _read_back(tmp_path / "c", str(tmp_path / "cache"))
    assert len({h["bucket"] for h in c.headers}) == 13
    for m in mols:
        assert np.array_equal(got[m.mol_id][0], m.coords)


class TestWriterRefusals:
    def _one(self, **kw):
        base = dict(mol_id=1, coords=np.zeros((1, 3, 2), np.int16), vaa=np.ones(1, np.float32),
                    ext_id="a", smiles="CC")
        base.update(kw)
        return W.Molecule(**base)

    @pytest.mark.parametrize("kw,match", [
        (dict(mol_id=1 << 63), "2\\^63"),
        (dict(coords=np.zeros((1, 3, 256), np.int16)), "n_real"),
        (dict(coords=np.zeros((1, 3, 2), np.int32)), "int16"),
        (dict(vaa=np.array([0.0], np.float32)), "vaa"),
        (dict(smiles="CC |$$|"), "CXSMILES"),
        (dict(ext_id="a‮b"), "bidi"),
        (dict(ext_id=""), "ext_id"),
    ])
    def test_unrepresentable_input(self, kw, match, tmp_path):
        with pytest.raises(ValueError, match=match):
            W.write_corpus(str(tmp_path / "c"), [self._one(**kw)], color=None, prep_version="t",
                           prep_commit=None)

    def test_color_self_needs_canonical_matrices(self, tmp_path):
        m = self._one(features=[(1, (0,))], color_self=np.zeros(1))
        with pytest.raises(ValueError, match="canonical"):
            W.write_corpus(str(tmp_path / "c"), [m], color=_color(0.5), prep_version="t",
                           prep_commit=None, with_color_self=True)
        W.write_corpus(str(tmp_path / "d"), [m], color=_color(0.5), prep_version="t",
                       prep_commit=None)                      # omitted by default
        assert "color_self" not in json.dumps(json.loads(
            (tmp_path / "d" / "corpus.json").read_bytes()))

    def test_duplicate_mol_id_and_bad_features(self, tmp_path):
        with pytest.raises(ValueError, match="repeats"):
            W.write_corpus(str(tmp_path / "c"), [self._one(), self._one()], color=None,
                           prep_version="t", prep_commit=None)
        with pytest.raises(ValueError, match="outside"):
            W.write_corpus(str(tmp_path / "c"), [self._one(features=[(1, (2,))],
                                                           color_self=np.zeros(1))],
                           color=_color(), prep_version="t", prep_commit=None)

    def test_a_bad_definition_is_refused(self, tmp_path):
        bomb = "AtomType A0 [#6]\n" + "".join(f"AtomType A{i} [{{A{i-1}}};R]\n" for i in range(1, 10))
        cd = _color()
        cd["definition"] = bomb + FDEF
        with pytest.raises(ValueError, match="definition_depth"):
            W.write_corpus(str(tmp_path / "c"), [self._one()], color=cd, prep_version="t",
                           prep_commit=None)


# ---------------------------------------------------------------------------
# Scan cache and the unvalidated-corpus refusal
# ---------------------------------------------------------------------------

class TestScanCache:
    def test_unscanned_corpus_is_refused_for_screening(self, tmp_path):
        root = tmp_path / "c"
        shutil.copytree(DATA / "shape", root)
        e = pytest.raises(CorpusValidationError, open_corpus, str(root),
                          cache_dir=str(tmp_path / "cache")).value
        assert e.code == SCAN_REQUIRED and "verify --tier scan" in str(e)
        # inspect works, but reads nothing beyond headers
        c = open_corpus(str(root), purpose="inspect", cache_dir=str(tmp_path / "cache"))
        assert not c.scanned and len(c.block_directory()) > 0
        with pytest.raises(CorpusValidationError):
            c.group(0)
        with pytest.raises(CorpusValidationError):
            next(c.molecules())

    def test_opens_after_roshambo3_verify(self, tmp_path, monkeypatch, capsys):
        from roshambo3 import cli
        root = tmp_path / "c"
        shutil.copytree(DATA / "color", root)
        monkeypatch.setenv(cache.ENV, str(tmp_path / "cache"))
        with pytest.raises(CorpusValidationError):
            open_corpus(str(root))
        assert cli.main(["verify", str(root), "--tier", "open"]) == 0
        with pytest.raises(CorpusValidationError):
            open_corpus(str(root))                      # open does not record a scan
        assert cli.main(["verify", str(root), "--tier", "scan"]) == 0
        assert "tier=scan  OK" in capsys.readouterr().out
        c = open_corpus(str(root))
        assert c.scanned and c.fingerprint == EXPECTED["color"]["fingerprint"]

    def test_second_scan_reads_no_section(self, tmp_path, monkeypatch):
        from roshambo3.format.r3g import fileio, validate
        root = tmp_path / "c"
        shutil.copytree(DATA / "color", root)
        cdir = str(tmp_path / "cache")
        first = verify(str(root), "scan", cache_dir=cdir)
        assert not first.cached
        reads = []
        orig = fileio.CorpusFile.read

        def spy(self, offset, length, **kw):
            reads.append((os.path.basename(self.path), offset, length))
            return orig(self, offset, length, **kw)
        monkeypatch.setattr(fileio.CorpusFile, "read", spy)
        monkeypatch.setattr(validate, "scan_group",
                            lambda *a, **k: pytest.fail("scan_group ran on a cached corpus"))
        second = verify(str(root), "scan", cache_dir=cdir)
        assert second.cached and second.fingerprint == first.fingerprint
        assert not any(n == "molecules.parquet" for n, _, _ in reads)
        # per group file: the prologue, the header and the blocks section, nothing else
        d = json.loads((root / "corpus.json").read_bytes())
        for g in d["groups"]:
            name = R.group_file_name(g["bucket"], g["group"]).split("/")[-1]
            og = validate.open_group(str(root), g)
            allowed = {(0, 64), (og.header_offset, len(og.header_bytes)),
                       (og.sections["blocks"]["offset"], og.sections["blocks"]["length"])}
            og.close()
            got = {(o, n) for f, o, n in reads if f == name}
            assert got <= allowed, (name, got - allowed)

    def test_one_byte_of_corpus_json_invalidates(self, scanned, tmp_path):
        root, cdir = scanned("shape")
        d = json.loads((root / "corpus.json").read_bytes())
        d["prep"]["version"] = d["prep"]["version"][:-1] + "1"   # not fingerprinted
        (root / "corpus.json").write_bytes(R.canonical_json(d))
        with pytest.raises(CorpusValidationError) as e:
            open_corpus(str(root), cache_dir=cdir)
        assert e.value.code == SCAN_REQUIRED
        assert not verify(str(root), "scan", cache_dir=cdir).cached
        open_corpus(str(root), cache_dir=cdir)

    def test_a_group_header_change_invalidates(self, tmp_path):
        """Two corpora identical but for one vaa value: the group header (its
        vaa sha256) and corpus.json both change; a scan of one does not
        admit the other."""
        mols = _random_molecules(np.random.default_rng(9), 5, color=False)
        W.write_corpus(str(tmp_path / "a"), mols, color=None, prep_version="t", prep_commit=None)
        mols[0].vaa = mols[0].vaa + np.float32(1)
        W.write_corpus(str(tmp_path / "b"), mols, color=None, prep_version="t", prep_commit=None)
        cdir = str(tmp_path / "cache")
        verify(str(tmp_path / "a"), "scan", cache_dir=cdir)
        shutil.rmtree(tmp_path / "a" / "groups")
        shutil.copytree(tmp_path / "b" / "groups", tmp_path / "a" / "groups")
        shutil.copy(tmp_path / "b" / "corpus.json", tmp_path / "a" / "corpus.json")
        with pytest.raises(CorpusValidationError):
            open_corpus(str(tmp_path / "a"), cache_dir=cdir)
        k1 = cache.scan_key("0" * 64, ["1" * 64, "2" * 64], [])
        assert k1 != cache.scan_key("0" * 64, ["1" * 64, "3" * 64], [])

    def test_rewritten_section_bytes_invalidate(self, scanned):
        """No hash in corpus.json or a header covers the run table's bytes;
        the file's size and mtime in the key do."""
        root, cdir = scanned("shape")
        p = root / "groups" / "b00_g0000.r3g"
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        with pytest.raises(CorpusValidationError):
            open_corpus(str(root), cache_dir=cdir)

    def test_a_group_changed_after_open_is_refused(self, scanned):
        root, cdir = scanned("shape")
        c = open_corpus(str(root), cache_dir=cdir)
        p = root / "groups" / "b00_g0000.r3g"
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        with pytest.raises(CorpusValidationError) as e:
            c.group(0)
        assert e.value.code == SCAN_REQUIRED


# ---------------------------------------------------------------------------
# roshambo3 verify
# ---------------------------------------------------------------------------

class TestVerifyCommand:
    def test_tiers_and_refusal(self, tmp_path, monkeypatch, capsys):
        from roshambo3 import cli
        monkeypatch.setenv(cache.ENV, str(tmp_path / "cache"))
        assert cli.main(["verify", str(DATA / "shape"), "--tier", "deep", "--all"]) == 0
        out = capsys.readouterr().out
        assert "tier=deep  OK" in out and "(all)" in out
        rc = cli.main(["verify", str(DATA / "adversarial" / "width_17")])
        err = capsys.readouterr().err
        assert rc == 1 and "code=limit.bit_width" in err and "tier=scan" in err

    def test_tier_flags_are_format_3_only(self, tmp_path, capsys):
        from roshambo3 import cli
        (tmp_path / "old").mkdir()
        assert cli.main(["verify", str(tmp_path / "old"), "--tier", "scan"]) == 1
        assert "corpus.json is missing" in capsys.readouterr().err

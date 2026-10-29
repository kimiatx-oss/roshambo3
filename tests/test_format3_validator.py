"""Format 3: the production validator (``roshambo3.format.r3g.validate``).

* every adversarial seed of ``tests/data/format3/adversarial`` is refused at
  its stated tier with its stated code, every lower tier accepting it, the
  message naming the file, within the ``open`` time bound;
* the validator agrees with ``r3g.reference`` (the behavioral oracle for
  ``open`` and ``scan``) on every seed and on seeded byte mutations, and
  never lets anything but ``CorpusValidationError`` escape;
* strict JSON, the Parquet page walker, file opening and the ``deep`` tier.

Synthetic corpora go to ``tmp_path``, never into the tree. No GPU, no corpus outside
the repository.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import time

import numpy as np
import pytest

from roshambo3.format.r3g import jsonstrict, reference as R, validate as V
from roshambo3.format.r3g import parquet as P
from roshambo3.format.r3g.errors import CODES, CorpusValidationError
from roshambo3.format.r3g.limits import DEFAULT_BOUNDS

DATA = pathlib.Path(__file__).resolve().parent / "data" / "format3"
ADV = DATA / "adversarial"
ADVERSARIAL = json.loads((ADV / "EXPECTED.json").read_text())
TIERS = ("open", "scan", "deep")


def _make_adversarial():
    sys.path.insert(0, str(ADV))
    try:
        import make_adversarial
    finally:
        sys.path.pop(0)
    return make_adversarial


def seed_root(name: str, tmp_path) -> pathlib.Path:
    if ADVERSARIAL[name]["generated"]:
        root = tmp_path / name
        _make_adversarial().build(name, str(root))
        return root
    return ADV / name


def is_prepass_seed(root: pathlib.Path) -> bool:
    return (root / "definition.json").exists() or (root / "definition.fdef").exists()


def refusal(fn, *a, **kw) -> CorpusValidationError:
    with pytest.raises(CorpusValidationError) as e:
        fn(*a, **kw)
    return e.value


# ---------------------------------------------------------------------------
# The adversarial seeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_seed_refused_at_its_tier_with_its_code(name, tmp_path):
    exp = ADVERSARIAL[name]
    root = seed_root(name, tmp_path)
    if is_prepass_seed(root):
        if (root / "definition.json").exists():
            d = json.loads((root / "definition.json").read_bytes())
            args = (d["definition"], d["generator"], d["types"])
        else:
            args = ((root / "definition.fdef").read_text(), "rdkit_fdef", None)
        t0 = time.monotonic()
        e = refusal(R.check_definition_text, *args)
        assert e.code == exp["code"], str(e)
        assert time.monotonic() - t0 < DEFAULT_BOUNDS.open_seconds
        return
    for tier in TIERS[:TIERS.index(exp["tier"])]:
        V.validate_corpus(str(root), tier, seed=0)          # lower tiers accept
    t0 = time.monotonic()
    e = refusal(V.validate_corpus, str(root), exp["tier"], seed=0)
    elapsed = time.monotonic() - t0
    assert type(e) is CorpusValidationError
    assert (e.tier, e.code) == (exp["tier"], exp["code"]), str(e)
    assert e.code in CODES
    # the message names the file (the field and limit are in the text)
    assert e.path and os.path.basename(e.path) in str(e) and len(str(e)) > len(e.path) + 20
    assert elapsed < DEFAULT_BOUNDS.open_seconds, (name, elapsed)


#: bare definitions (no corpus): both validators call reference.check_definition_text
PREPASS_SEEDS = {n for n in ADVERSARIAL if is_prepass_seed(ADV / n)} | {"definition_oversize_prepass"}


@pytest.mark.parametrize("name", sorted(n for n, v in ADVERSARIAL.items()
                                        if v["tier"] != "deep" and n not in PREPASS_SEEDS))
def test_seed_code_agrees_with_the_reference(name, tmp_path):
    root = seed_root(name, tmp_path)
    tier = ADVERSARIAL[name]["tier"]
    mine = refusal(V.validate_corpus, str(root), tier, seed=0)
    ref = refusal(R.validate_corpus, str(root), tier)
    assert mine.code == ref.code


# ---------------------------------------------------------------------------
# Differential: seeded single-byte mutations of group files
# ---------------------------------------------------------------------------

def _outcome(fn, root, tier):
    try:
        fn(root, tier)
        return "accept"
    except CorpusValidationError as e:
        return e.code


@pytest.mark.parametrize("variant", ["shape", "color"])
def test_mutations_agree_with_the_reference_and_raise_nothing_else(variant, tmp_path):
    """300 single-byte mutations per corpus, each in a fresh copy: the
    production validator and the reference give the same outcome at ``open``
    and ``scan``, and nothing but CorpusValidationError escapes either."""
    src = DATA / variant
    rng = np.random.default_rng(20261018 + (variant == "color"))
    groups = sorted((src / "groups").iterdir())
    root = tmp_path / "c"
    disagreements = []
    for trial in range(300):
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(src, root)
        g = groups[int(rng.integers(len(groups)))]
        p = root / "groups" / g.name
        data = bytearray(p.read_bytes())
        # half the mutations land in the prologue+header, half anywhere
        hdr_off = int.from_bytes(data[8:16], "little")
        if rng.random() < 0.5:
            pos = int(rng.choice(np.r_[0:64, hdr_off:len(data)]))
        else:
            pos = int(rng.integers(len(data)))
        data[pos] = (data[pos] + int(rng.integers(1, 256))) % 256
        p.write_bytes(bytes(data))
        for tier in ("open", "scan"):
            a = _outcome(lambda r, t: V.validate_corpus(r, t, seed=0), str(root), tier)
            b = _outcome(R.validate_corpus, str(root), tier)
            assert a == "accept" or a in CODES
            if a != b:
                disagreements.append((g.name, pos, tier, a, b))
    assert not disagreements, disagreements[:10]


# ---------------------------------------------------------------------------
# Strict JSON (section 16.1)
# ---------------------------------------------------------------------------

class TestStrictJson:
    @pytest.mark.parametrize("doc,code", [
        (b'{"a":1,"a":2}', "json.duplicate_key"),
        (b'{"a":{"b":1,"b":1}}', "json.duplicate_key"),
        (b'{"a":NaN}', "json.nan"),
        (b'{"a":Infinity}', "json.nan"),
        (b'{"a":-Infinity}', "json.nan"),
        (b'{"a":1e999}', "json.nan"),
        (b'{"a":' + b"[" * 32 + b"]" * 32 + b"}", "json.depth"),
        (b"[1]", "json.syntax"),
        (b"", "json.syntax"),
        (b'{"a":"\xff"}', "json.syntax"),
        (b'{"a":1', "json.syntax"),
    ])
    def test_refusals(self, doc, code):
        assert refusal(jsonstrict.loads, doc, what="x", path="p").code == code

    def test_depth_32_is_legal_and_brackets_in_strings_do_not_count(self):
        assert jsonstrict.loads(b'{"a":' + b"[" * 31 + b"]" * 31 + b"}")
        assert jsonstrict.loads(b'{"a":"' + b"[{" * 100 + b'"}')

    def test_size_is_checked_first(self):
        e = refusal(jsonstrict.loads, b"{" * 100, max_bytes=50)
        assert e.code == "json.size"

    def test_string_length_counts_raw_bytes_escapes_included(self):
        """Limit 4: the raw bytes between the quotes. 32,769 two-byte escapes
        are 65,538 bytes (refused); the reference counts 32,769."""
        doc = b'{"a":"' + b"\\n" * 32769 + b'"}'
        assert refusal(jsonstrict.loads, doc).code == "json.string_length"
        assert refusal(R.loads_strict, doc, what="x").code == "json.string_length"
        ok = b'{"a":"' + b"\\n" * 32768 + b'"}'
        assert jsonstrict.loads(ok)["a"] == "\n" * 32768

    def test_prescan_matches_a_naive_scan(self):
        """Exact on every input whose backslashes are all inside strings (all
        valid JSON). A backslash outside a string is a syntax error; there
        the pre-scan treats ``\\"`` as escaped, stays outside the string and
        counts more depth, never less, and the parser refuses the text."""
        def naive(data):
            depth = md = 0
            in_str = esc = False
            start = ms = 0
            for i, b in enumerate(data):
                if in_str:
                    if esc:
                        esc = False
                    elif b == 0x5C:
                        esc = True
                    elif b == 0x22:
                        in_str = False
                        ms = max(ms, i - start - 1)
                elif b == 0x22:
                    in_str, start = True, i
                elif b == 0x5C:
                    return None
                elif b in (0x7B, 0x5B):
                    depth += 1
                    md = max(md, depth)
                elif b in (0x7D, 0x5D):
                    depth -= 1
            if in_str:
                ms = max(ms, len(data) - start - 1)
            return md, ms
        rng = np.random.default_rng(7)
        alphabet = np.frombuffer(b'{}[]"\\\\\\ax,:', dtype=np.uint8)
        checked = 0
        for n in (1, 2, 5, 40, 300):
            for _ in range(300):
                data = alphabet[rng.integers(0, alphabet.size, n)].tobytes()
                want = naive(data)
                got = jsonstrict.scan_depth_and_strings(data)
                if want is None:
                    continue
                assert got == want, data
                checked += 1
        assert checked > 500
        # across the chunk boundary
        big = b'{"k":"' + b"\\\\" * 700000 + b'\\"' + b"x" * 10 + b'"}'
        assert jsonstrict.scan_depth_and_strings(big) == naive(big)

    def test_hostile_16_mib_documents_parse_within_the_open_bound(self):
        m = 16 * 1024 * 1024
        for doc in (b"{}" + b" " * (m - 2),
                    b'{"a":[' + b"{}," * ((m - 20) // 3) + b"{}]}",
                    b'{"a":"' + b"[{" * ((m - 20) // 2) + b'"}'):
            t0 = time.monotonic()
            try:
                jsonstrict.loads(doc)
            except CorpusValidationError:
                pass
            assert time.monotonic() - t0 < DEFAULT_BOUNDS.open_seconds

    def test_dumps_is_canonical(self):
        assert jsonstrict.dumps({"b": 1, "a": [True, None, "é"]}) == \
            b'{"a":[true,null,"\xc3\xa9"],"b":1}'
        with pytest.raises(ValueError):
            jsonstrict.dumps({"x": float("nan")})


# ---------------------------------------------------------------------------
# The ticket's named cases, each on its own
# ---------------------------------------------------------------------------

def _copy(variant_or_seed: pathlib.Path, tmp_path) -> pathlib.Path:
    dst = tmp_path / "corpus"
    shutil.copytree(variant_or_seed, dst)
    return dst


def _patch_header(path: pathlib.Path, fn):
    """Rewrite a group header through ``fn`` and re-bind corpus.json's
    header_sha256, so the case reaches the check it targets."""
    data = path.read_bytes()
    off = int.from_bytes(data[8:16], "little")
    h = json.loads(data[off:])
    fn(h)
    hb = R.canonical_json(h)
    path.write_bytes(data[:off] + hb)
    root = path.parent.parent
    d = json.loads((root / "corpus.json").read_bytes())
    for g in d["groups"]:
        if R.group_file_name(g["bucket"], g["group"]).endswith(path.name):
            g["header_sha256"] = R.sha256_hex(hb)
    (root / "corpus.json").write_bytes(R.canonical_json(d))


class TestNamedCases:
    def test_duplicate_key(self):
        e = refusal(V.validate_corpus, str(ADV / "json_duplicate_key"), "open")
        assert e.code == "json.duplicate_key" and "corpus.json" in str(e)

    def test_duplicate_key_in_a_group_header(self, tmp_path):
        root = _copy(DATA / "shape", tmp_path)
        p = root / "groups" / "b00_g0000.r3g"
        data = p.read_bytes()
        off = int.from_bytes(data[8:16], "little")
        p.write_bytes(data[:off] + data[off:-1] + b',"n_mol":1}')
        e = refusal(V.validate_corpus, str(root), "open")
        assert e.code == "json.duplicate_key" and e.path.endswith("b00_g0000.r3g")

    def test_nan(self):
        assert refusal(V.validate_corpus, str(ADV / "json_nan"), "open").code == "json.nan"

    def test_deep_nesting(self):
        assert refusal(V.validate_corpus, str(ADV / "json_depth"), "open").code == "json.depth"

    def test_oversized_header_is_refused_before_it_is_read(self, tmp_path):
        root = seed_root("oversize_header", tmp_path)
        t0 = time.monotonic()
        e = refusal(V.validate_corpus, str(root), "open")
        assert e.code == "json.size" and "limit 2" in str(e)
        assert time.monotonic() - t0 < 0.5

    def test_overlapping_section(self):
        e = refusal(V.validate_corpus, str(ADV / "section_overlap"), "open")
        assert e.code == "layout.sections"

    def test_out_of_file_section(self):
        e = refusal(V.validate_corpus, str(ADV / "section_outside_file"), "open")
        assert e.code == "layout.sections"

    def test_width_17(self):
        V.validate_corpus(str(ADV / "width_17"), "open")
        e = refusal(V.validate_corpus, str(ADV / "width_17"), "scan")
        assert e.code == "limit.bit_width" and "block 0" in str(e)

    def test_n_real_exceeds_L(self):
        for seed in ("n_real_exceeds_L_heavy", "n_color_exceeds_L"):
            e = refusal(V.validate_corpus, str(ADV / seed), "scan")
            assert e.code == "layout.widths", seed

    def test_atom_index_at_least_n_real(self):
        e = refusal(V.validate_corpus, str(ADV / "feature_atom_out_of_range"), "scan")
        assert e.code == "layout.features" and "n_real" in str(e)

    def test_symlinked_group_file_escaping_the_root(self, tmp_path):
        e = refusal(V.validate_corpus, str(seed_root("group_symlink_escapes_root", tmp_path)), "open")
        assert e.code == "io.not_regular_file"

    def test_symlinked_group_file_inside_the_root(self, tmp_path):
        root = _copy(DATA / "shape", tmp_path)
        p = root / "groups" / "b00_g0000.r3g"
        os.replace(p, root / "elsewhere.r3g")
        os.symlink("../elsewhere.r3g", p)
        assert refusal(V.validate_corpus, str(root), "open").code == "io.not_regular_file"

    def test_symlinked_groups_directory(self, tmp_path):
        root = _copy(DATA / "shape", tmp_path)
        os.replace(root / "groups", tmp_path / "outside_groups")
        os.symlink(tmp_path / "outside_groups", root / "groups")
        assert refusal(V.validate_corpus, str(root), "open").code == "io.not_regular_file"

    def test_corpus_root_may_be_a_symlink(self, tmp_path):
        root = _copy(DATA / "shape", tmp_path)
        os.symlink(root, tmp_path / "link")
        V.validate_corpus(str(tmp_path / "link"), "scan")

    def test_dot_dot_and_path_entries(self):
        e = refusal(V.validate_corpus, str(ADV / "group_path_entry"), "open")
        assert e.code == "schema.unknown_key" and "'file'" in str(e)
        e = refusal(V.validate_corpus, str(ADV / "group_index_string"), "open")
        assert e.code == "schema.type"

    def test_names_never_come_from_a_file(self, tmp_path):
        from roshambo3.format.r3g.fileio import open_corpus_file
        for bad in ("..", "../corpus.json", "a/b", "", "."):
            e = refusal(open_corpus_file, str(DATA / "shape"), "groups", bad)
            assert e.code == "io.not_regular_file"

    @pytest.mark.parametrize("seed", ["group_is_fifo", "corpus_json_is_fifo"])
    def test_fifo_is_refused_not_waited_on(self, seed, tmp_path):
        t0 = time.monotonic()
        e = refusal(V.validate_corpus, str(seed_root(seed, tmp_path)), "open")
        assert e.code == "io.not_regular_file" and time.monotonic() - t0 < 1.0

    def test_missing_group_file(self, tmp_path):
        root = _copy(DATA / "shape", tmp_path)
        (root / "groups" / "b12_g0000.r3g").unlink()
        assert refusal(V.validate_corpus, str(root), "open").code == "io.missing_file"

    def test_truncated_group_file(self, tmp_path):
        root = _copy(DATA / "shape", tmp_path)
        p = root / "groups" / "b00_g0000.r3g"
        p.write_bytes(p.read_bytes()[:40])
        assert refusal(V.validate_corpus, str(root), "open").code == "prologue.magic"

    def test_open_reads_no_section_but_blocks(self, tmp_path, monkeypatch):
        """Section 16.2: open is O(header) plus the block table."""
        from roshambo3.format.r3g import fileio
        reads = []
        orig = fileio.CorpusFile.read

        def spy(self, offset, length, **kw):
            reads.append((os.path.basename(self.path), offset, length))
            return orig(self, offset, length, **kw)
        monkeypatch.setattr(fileio.CorpusFile, "read", spy)
        V.validate_corpus(str(DATA / "color"), "open")
        for name, off, ln in reads:
            if name.endswith(".r3g"):
                assert off == 0 and ln == 64 or ln % 64 == 0 or off > 64, (name, off, ln)
        assert sum(ln for _, _, ln in reads) < 64 * 1024

    def test_pre_pass_refusal_names_the_file_and_field(self):
        e = refusal(V.validate_corpus, str(ADV / "macro_bomb"), "scan")
        assert e.code == "limit.definition_depth"
        assert e.path.endswith("corpus.json") and "color.definition" in str(e)


# ---------------------------------------------------------------------------
# The block-table and block-header checks are vectorized: many blocks
# ---------------------------------------------------------------------------

def test_many_blocks_validate_quickly(tmp_path):
    """4,096 one-row blocks (block_rows = 1): open and scan stay well inside
    the bound because neither walks blocks in Python."""
    from roshambo3.format.r3g import writer as W
    mols = [W.Molecule(mol_id=i + 1, coords=np.full((1, 3, 1), i % 7, np.int16),
                       vaa=np.array([20.6], np.float32), ext_id=f"m{i}", smiles="C", mw=16.0)
            for i in range(4096)]
    W.write_corpus(str(tmp_path / "c"), mols, color=None, prep_version="t", prep_commit=None,
                   block_rows=1)
    t0 = time.monotonic()
    V.validate_corpus(str(tmp_path / "c"), "scan")
    assert time.monotonic() - t0 < DEFAULT_BOUNDS.open_seconds
    # and a tampered block deep in the table is found with the reference's code
    p = tmp_path / "c" / "groups" / "b00_g0000.r3g"
    og = V.open_group(str(tmp_path / "c"), json.loads((tmp_path / "c" / "corpus.json")
                                                     .read_bytes())["groups"][0])
    off = og.sections["coords"]["offset"] + int(og.blocks[3000]["offset"])
    og.close()
    data = bytearray(p.read_bytes())
    data[off + 20] = 1
    p.write_bytes(bytes(data))
    e = refusal(V.validate_corpus, str(tmp_path / "c"), "scan")
    assert e.code == "layout.block_header" and "block 3000" in str(e)
    assert refusal(R.validate_corpus, str(tmp_path / "c"), "scan").code == e.code


# ---------------------------------------------------------------------------
# Parquet page headers (section 11 item 3)
# ---------------------------------------------------------------------------

class TestPageWalker:
    def _chunks(self):
        import pyarrow.parquet as pq
        path = DATA / "color" / "molecules.parquet"
        data = path.read_bytes()
        g = pq.ParquetFile(str(path)).metadata.row_group(0)
        return data, g

    def test_accepts_the_conformance_pages(self):
        data, g = self._chunks()
        for c in range(g.num_columns):
            col = g.column(c)
            P.check_pages(data[col.data_page_offset:col.data_page_offset
                               + col.total_compressed_size], g.num_rows, codec=col.compression)

    def test_refuses_a_dictionary_page_and_a_short_chunk(self):
        data, g = self._chunks()
        col = g.column(2)
        chunk = bytearray(data[col.data_page_offset:col.data_page_offset + col.total_compressed_size])
        chunk[1] = 0x04                                  # page type 2: DICTIONARY_PAGE
        with pytest.raises(ValueError, match="not a data page"):
            P.check_pages(bytes(chunk), g.num_rows)
        with pytest.raises(ValueError):
            P.check_pages(bytes(chunk[:-3]), g.num_rows)

    def test_a_list_of_booleans_is_refused_at_once(self):
        """The first reference's ``_thrift_skip`` spun on this: a list of
        2^62 booleans consumes no bytes per element (amendment S2)."""
        hdr = bytes([0x15, 0x00, 0x15, 0x08, 0x15, 0x08,   # type 0, sizes 4, 4
                     0x69, 0xF1]) + b"\xff" * 8 + b"\x3f"  # field 9: list<bool>, huge size
        t0 = time.monotonic()
        with pytest.raises(ValueError, match="not allowed|thrift type"):
            P.check_pages(hdr + b"\0" * 16, 1)
        assert time.monotonic() - t0 < 0.1

    def test_unknown_field_and_wrong_type_are_refused(self):
        with pytest.raises(ValueError, match="not allowed"):
            P.check_pages(bytes([0x15, 0x00, 0xA5, 0x02, 0x00]), 1)
        with pytest.raises(ValueError, match="thrift type"):
            P.check_pages(bytes([0x18, 0x00, 0x00]), 1)  # field 1 as binary

    def test_page_header_size_is_capped_and_statistics_must_be_empty(self):
        # a Statistics max_value binary (amendment S2: only an empty Statistics)
        n = 300
        varint = bytearray()
        x = n
        while True:
            varint.append((x & 0x7F) | (0x80 if x > 0x7F else 0))
            x >>= 7
            if not x:
                break
        hdr = bytes([0x15, 0x00, 0x15, 0x02, 0x15, 0x02, 0x2C, 0x15, 0x02, 0x15, 0x00,
                     0x3C, 0x58]) + bytes(varint) + b"x" * n
        with pytest.raises(ValueError, match="not allowed|longer than 256"):
            P.check_pages(hdr, 1)


# ---------------------------------------------------------------------------
# deep (section 16.4)
# ---------------------------------------------------------------------------

class TestDeep:
    def test_shape_corpus_passes_with_every_row(self):
        r = V.validate_corpus(str(DATA / "shape"), "deep", all_rows=True)
        assert r.sampled_molecules == 47

    def test_conformance_color_corpus_passes_reperception(self):
        r = V.validate_corpus(str(DATA / "color"), "deep", all_rows=True)
        assert r.sampled_molecules == 47

    def test_tampered_tail_is_refused(self, tmp_path):
        from roshambo3.format.r3g import writer as W
        xyz = np.zeros((1, 3, 3), np.int16)
        xyz[0, 0] = [0, 1, 3]                          # w_x = 2, 3 of 256 positions used
        m = W.Molecule(mol_id=5, coords=xyz, vaa=np.array([1.0], np.float32), ext_id="x",
                       smiles="CCC", mw=1.0)
        root = tmp_path / "c"
        W.write_corpus(str(root), [m], color=None, prep_version="t", prep_commit=None)
        p = root / "groups" / "b00_g0000.r3g"
        og = V.open_group(str(root), json.loads((root / "corpus.json").read_bytes())["groups"][0])
        base = og.sections["coords"]["offset"]
        og.close()
        data = bytearray(p.read_bytes())
        data[base + 32 + 2 * 15] = 1                   # lane 15, position 0: unused
        # re-bind every hash in front of the tail check
        blk = bytes(data[base:base + 96])
        _rehash_block_and_sections(root, p, data, blk)
        e = refusal(V.validate_corpus, str(root), "deep")
        assert e.code == "integrity.padding" and "tail" in str(e)

    def test_sandbox_wall_clock_kill(self):
        from roshambo3.format.r3g import _sandbox
        with pytest.raises(_sandbox.SandboxFailure, match="killed"):
            _sandbox.run_child({"perceive_any": False, "generator": None, "definition": None,
                                "types": [], "molecules": [["C", False]] * 50000}, timeout=0.05)


def _rehash_block_and_sections(root, p, data, blk):
    """After editing a block in place, recompute its block-table hash, the
    section hashes and the header hash in corpus.json."""
    import hashlib
    off = int.from_bytes(data[8:16], "little")
    h = json.loads(bytes(data[off:]))
    sec = {e["name"]: e for e in h["sections"]}
    bt = sec["blocks"]["offset"]
    data[bt + 32:bt + 64] = hashlib.sha256(blk).digest()
    for e in h["sections"]:
        e["sha256"] = hashlib.sha256(bytes(data[e["offset"]:e["offset"] + e["length"]])).hexdigest()
    hb = R.canonical_json(h)
    p.write_bytes(bytes(data[:off]) + hb)
    d = json.loads((root / "corpus.json").read_bytes())
    d["groups"][0]["header_sha256"] = R.sha256_hex(hb)
    (root / "corpus.json").write_bytes(R.canonical_json(d))


# ---------------------------------------------------------------------------
# deep with RDKit on a corpus whose feature table is right
# ---------------------------------------------------------------------------

def build_rdkit_corpus(root, smiles_list, *, color=True, generator="rdkit_fdef", n_conf=2):
    """A small corpus built the way FORMAT.md says: canonical non-isomeric
    SMILES, slots in its atom order, features perceived on AddHs(parsed),
    self-overlaps from the stored integers."""
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, ChemicalFeatures
    from roshambo3.format.r3g import writer as W
    from roshambo3.prep.conformer import prep_conformer, stored_self_volume
    from roshambo3.reference.overlap import color_overlap
    types = ["Donor", "Acceptor", "Aromatic", "Hydrophobe"]
    n = len(types)
    r = np.zeros((n + 1, n + 1), np.float32)
    for t in range(1, n + 1):
        r[t, t] = 1.0
    if generator == "rdkit_fdef":
        text = open(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"), encoding="utf-8").read()
    else:
        text = {"Donor": ["[N;!H0]", "[O;H1]"], "Acceptor": ["[O,N;H0]"],
                "Aromatic": ["a1aaaaa1"], "Hydrophobe": ["[C;H3]"]}
    cd = R.color_descriptor(generator=generator, definition=text, types=types, r=r, p=r,
                            rdkit_version="x") if color else None
    fac = ChemicalFeatures.BuildFeatureFactoryFromString(text) if generator == "rdkit_fdef" else None
    mols = []
    for i, smi in enumerate(smiles_list):
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
        mol = Chem.MolFromSmiles(canon)
        mh = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, randomSeed=7 + i)
        heavy = [a.GetIdx() for a in mh.GetAtoms() if a.GetAtomicNum() > 1]
        assert heavy == list(range(mol.GetNumAtoms()))
        coords = np.stack([R.quantize(prep_conformer(np.asarray(c.GetPositions())[heavy])).T
                           for c in mh.GetConformers()]).astype(np.int16)
        feats = []
        if color and fac is not None:
            feats = [(types.index(f.GetFamily()) + 1, tuple(f.GetAtomIds()))
                     for f in fac.GetFeaturesForMol(mh) if f.GetFamily() in types]
        elif color:
            for fam, pats in text.items():
                seen = []
                for pt in pats:
                    for m in mh.GetSubstructMatches(Chem.MolFromSmarts(pt)):
                        if m not in seen:
                            seen.append(m)
                feats += [(types.index(fam) + 1, m) for m in seen]
        feats = R.canonical_feature_order(feats)
        vaa = np.array([stored_self_volume(R.dequantize(c).astype(np.float64).T) for c in coords],
                       np.float32)
        cs = None
        if color:
            tt = np.array([t for t, _ in feats], np.int64)
            def cself(c):
                pos = R.dequantize(R.derive_positions(c, feats)).astype(np.float64)
                return color_overlap(pos, tt, pos, tt, r, r) if feats else 0.0
            cs = np.array([cself(c) for c in coords])
        mols.append(W.Molecule(mol_id=1000 + i, coords=coords, vaa=vaa, features=feats,
                               color_self=cs, ext_id=f"m{i}", smiles=canon, mw=1.0))
    W.write_corpus(str(root), mols, color=cd, prep_version="t", prep_commit=None, block_rows=3)
    return mols, cd


SMILES = ["Nc1ccc(cc1)-c1ccc(N)cc1", "c1cnc[nH]1", "OC(=O)c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
          "CCO", "C"]


class TestDeepRdkit:
    pytest.importorskip("rdkit")

    @pytest.mark.parametrize("generator", ["rdkit_fdef", "smarts_list"])
    def test_a_correct_corpus_passes_every_row(self, generator, tmp_path):
        build_rdkit_corpus(tmp_path / "c", SMILES, generator=generator)
        r = V.validate_corpus(str(tmp_path / "c"), "deep", all_rows=True)
        assert r.sampled_molecules == len(SMILES)

    def test_a_wrong_feature_table_is_refused(self, tmp_path):
        from roshambo3.format.r3g import writer as W
        mols, cd = build_rdkit_corpus(tmp_path / "a", SMILES)
        mols[0].features = mols[0].features[1:]          # drop one feature
        W.write_corpus(str(tmp_path / "b"), mols, color=cd, prep_version="t", prep_commit=None)
        V.validate_corpus(str(tmp_path / "b"), "scan")
        e = refusal(V.validate_corpus, str(tmp_path / "b"), "deep", all_rows=True)
        assert e.code == "layout.features" and e.path.endswith(".r3g")

    def test_a_wrong_self_overlap_is_refused(self, tmp_path):
        from roshambo3.format.r3g import writer as W
        mols, cd = build_rdkit_corpus(tmp_path / "a", SMILES)
        mols[2].vaa = mols[2].vaa * np.float32(1.001)
        W.write_corpus(str(tmp_path / "b"), mols, color=cd, prep_version="t", prep_commit=None)
        e = refusal(V.validate_corpus, str(tmp_path / "b"), "deep", all_rows=True)
        assert e.code == "value.self_overlap" and "vaa" in str(e)

    def test_a_smiles_with_the_wrong_atom_count_is_refused(self, tmp_path):
        from roshambo3.format.r3g import writer as W
        mols, cd = build_rdkit_corpus(tmp_path / "a", SMILES, color=False)
        mols[4].smiles = "CCCO"
        W.write_corpus(str(tmp_path / "b"), mols, color=None, prep_version="t", prep_commit=None)
        V.validate_corpus(str(tmp_path / "b"), "scan")
        e = refusal(V.validate_corpus, str(tmp_path / "b"), "deep", all_rows=True)
        assert e.code == "limit.smiles" and "heavy atoms" in str(e)

    def test_an_unparsable_smiles_is_refused(self, tmp_path):
        from roshambo3.format.r3g import writer as W
        mols, _ = build_rdkit_corpus(tmp_path / "a", SMILES, color=False)
        mols[4].smiles = "C1CC"
        W.write_corpus(str(tmp_path / "b"), mols, color=None, prep_version="t", prep_commit=None)
        e = refusal(V.validate_corpus, str(tmp_path / "b"), "deep", all_rows=True)
        assert e.code == "limit.smiles" and "does not parse" in str(e)


# ---------------------------------------------------------------------------
# Amendments S7 and S8: short reads, int64 JSON integers
# ---------------------------------------------------------------------------

def test_a_file_truncated_after_open_is_a_short_read_not_sigbus(tmp_path):
    from roshambo3.format.r3g.fileio import open_corpus_file
    root = _copy(DATA / "shape", tmp_path)
    f = open_corpus_file(str(root), "groups", "b12_g0000.r3g")
    try:
        with open(root / "groups" / "b12_g0000.r3g", "r+b") as fh:
            fh.truncate(100)
        e = refusal(f.read, 0, f.size)
        assert e.code == "io.short_read"
    finally:
        f.close()


def test_a_json_integer_beyond_int64_is_refused(tmp_path):
    root = _copy(DATA / "shape", tmp_path)
    _patch_header(root / "groups" / "b12_g0000.r3g",
                  lambda h: h["sections"][0].__setitem__("offset", 1 << 64))
    e = refusal(V.validate_corpus, str(root), "open")
    assert e.code == "schema.value" and "64-bit" in str(e)
    assert refusal(R.validate_corpus, str(root), "open").code == e.code


# ---------------------------------------------------------------------------
# Limit 34 for every corpus seed, in a fresh interpreter
# ---------------------------------------------------------------------------

def test_every_corpus_seed_stays_within_the_open_memory_bound(tmp_path):
    """One child interpreter validates every (pre-materialized) corpus seed at
    its tier; its peak RSS above the post-import baseline must stay within
    256 MiB (limit 34; the seeds' fixed sections are a few hundred bytes).
    Linux: ``VmHWM`` after ``clear_refs`` -- ``ru_maxrss`` is inherited across
    ``execve`` from pytest, so it cannot measure the child; elsewhere
    ``ru_maxrss`` (bytes on macOS)."""
    import subprocess
    roots = []
    for name, exp in sorted(ADVERSARIAL.items()):
        root = seed_root(name, tmp_path)
        if not is_prepass_seed(root):
            roots.append((str(root), exp["tier"]))
    code = r"""
import json, resource, sys
import numpy, pyarrow.parquet, pyarrow.compute
from roshambo3.format.r3g import validate as V
from roshambo3.format.r3g.errors import CorpusValidationError
def st(f):
    try:
        for line in open("/proc/self/status"):
            if line.startswith(f + ":"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
def ru():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024
linux = st("VmHWM") is not None
if linux:
    try:
        open("/proc/self/clear_refs", "w").write("5")
    except OSError:
        linux = False
base = st("VmRSS") if linux else ru()
for root, tier in json.loads(sys.stdin.read()):
    try:
        V.validate_corpus(root, tier, seed=0)
    except CorpusValidationError:
        pass
print(json.dumps({"base": base, "peak": st("VmHWM") if linux else ru(), "linux": linux}))
"""
    out = subprocess.run([sys.executable, "-c", code], input=json.dumps(roots), text=True,
                         capture_output=True, timeout=600, check=True)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["peak"] - r["base"] <= DEFAULT_BOUNDS.rss_base_bytes, r

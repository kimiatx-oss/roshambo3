"""Corpus format 3: the reference codec against the spec and the conformance corpus.

``doc/FORMAT.md`` is the authority; ``src/roshambo3/format/r3g/reference.py``
is its pure-Python reference; ``tests/data/format3/`` is the conformance
corpus whose committed files are the oracle (``EXPECTED.json``) and whose
adversarial seeds must each be refused by the stated validator tier with the
stated refusal code, within the spec's time bound. No RDKit, no compiled
extension, no corpus outside the repository: this file runs on every CI job.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

from roshambo3.format.r3g import reference as R

DATA = pathlib.Path(__file__).resolve().parent / "data" / "format3"
EXPECTED = json.loads((DATA / "EXPECTED.json").read_text())
ADVERSARIAL = json.loads((DATA / "adversarial" / "EXPECTED.json").read_text())
BUDGET = 512 * 1024


def _digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# BP256-16: section 5
# ---------------------------------------------------------------------------

class TestBitPacking:
    def test_worked_example(self):
        """Appendix A, byte for byte."""
        xyz = np.zeros((1, 3, 8), np.int16)
        xyz[0, 0, :3] = [-1, 0, 2]
        xyz[0, 1, :3] = [5, 5, 5]
        xyz[0, 2, :3] = [-300, 100, 7]
        blk = R.encode_block(xyz, [3])
        assert len(blk) == 384 == R.block_length(3, 2, 0, 9)
        assert blk[:9].hex() == "ffff0500d4fe020009"
        assert not any(blk[9:32])
        assert blk[32:38].hex() == "000001000300" and not any(blk[38:96])
        assert blk[96:102].hex() == "000090013301" and not any(blk[102:])
        assert np.array_equal(R.decode_block(blk, [3], 8), xyz)

    def test_word_crossing_example(self):
        """Appendix A, second example: w = 3, lane 0 positions 5 and 6."""
        vals = np.zeros(256, np.int64)
        vals[16 * 5] = 7
        vals[16 * 6] = 5
        vals[1] = 7
        mn, w, stream = R.pack_axis(vals)
        words = np.frombuffer(stream, "<u2").reshape(w, 16)
        assert (mn, w) == (0, 3)
        assert words[0, 0] == 0x8000 and words[1, 0] == 0x17
        assert np.array_equal(R.unpack_axis(stream, w, mn, 256), vals)

    @pytest.mark.parametrize("w", range(17))
    def test_every_width_round_trips_with_tails(self, w):
        """Widths 0..16, streams of 1, 255, 256, 257 and ~1000 values (so the
        last group is full, one short, or a one-value tail), several rows."""
        rng = np.random.default_rng(1000 + w)
        for nrows in (1, 3, 9, 13):
            nr = rng.integers(1, 90, nrows)
            L = R.pad8(int(nr.max()))
            xyz = np.zeros((nrows, 3, L), np.int16)
            for a in range(3):
                lo = int(rng.integers(-32768, 32768 - (2 ** w - 1))) if w < 16 else -32768
                for j in range(nrows):
                    xyz[j, a, :nr[j]] = (lo + rng.integers(0, 2 ** w, nr[j])) if w else lo
                # pin the extremes so the width is exactly w
                xyz[0, a, 0] = lo
                xyz[-1, a, nr[-1] - 1] = lo + (2 ** w - 1)
            blk = R.encode_block(xyz, nr)
            hdr = R.parse_block_header(blk)
            assert list(hdr.widths) == [w, w, w]
            assert len(blk) == R.block_length(int(nr.sum()), *hdr.widths)
            assert np.array_equal(R.decode_block(blk, nr, L), xyz)

    def test_tail_positions_are_zero_and_ignored(self):
        xyz = np.zeros((1, 3, 8), np.int16)
        xyz[0, :, :5] = 7
        blk = R.encode_block(xyz, [5])
        hdr = R.parse_block_header(blk)
        assert list(hdr.widths) == [0, 0, 0] and len(blk) == 32
        xyz[0, 0, :5] = [1, 2, 3, 4, 5]
        blk = R.encode_block(xyz, [5])
        u = R.unpack_axis_raw(blk[32:32 + 3 * 32], 3, 5)
        assert u.size == 256 and not np.any(u[5:])

    @pytest.mark.parametrize("w", range(17))
    def test_vectorized_raw_unpack_matches_the_scalar_loop(self, w):
        rng = np.random.default_rng(77 + w)
        for total in (1, 255, 256, 700):
            stream = rng.integers(0, 1 << 16, (-(-total // 256)) * w * 16, dtype=np.uint16).astype("<u2").tobytes()
            assert np.array_equal(R.unpack_axis_raw_vectorized(stream, w, total),
                                  R.unpack_axis_raw(stream, w, total))

    @pytest.mark.parametrize("w", range(17))
    def test_decode_block_matches_the_scalar_section_5_3_loop(self, w):
        """``decode_block`` (vectorized unpack and row scatter) against a
        decode written as section 5.3's loop, position by position, on random
        stream bytes with ragged rows, random mins and the 5.6 wrap."""
        rng = np.random.default_rng(310 + w)
        for nrows in (1, 4, 11):
            nr = rng.integers(1, 120, nrows)
            total = int(nr.sum())
            L = R.pad8(int(nr.max()))
            groups = -(-total // 256)
            mins = [int(rng.integers(-32768, 32768)) for _ in range(3)]
            hdr = bytearray(32)
            hdr[0:6] = np.asarray(mins, "<i2").tobytes()
            hdr[6:9] = bytes([w, w, w])
            streams = [rng.integers(0, 1 << 16, groups * w * 16, dtype=np.uint16)
                       .astype("<u2").tobytes() for _ in range(3)]
            blk = bytes(hdr) + b"".join(streams)
            want = np.zeros((nrows, 3, L), np.int16)
            for a in range(3):
                u = R.unpack_axis_raw(streams[a], w, total)
                v = [np.uint16((int(x) + mins[a]) & 0xFFFF).view(np.int16) for x in u[:total]]
                p = 0
                for j in range(nrows):
                    for i in range(int(nr[j])):
                        want[j, a, i] = v[p]
                        p += 1
            assert np.array_equal(R.decode_block(blk, nr, L), want)

    def test_decoder_refusals(self):
        """Section 17: wrong input length, width 17, n_real > L, empty block."""
        xyz = np.zeros((2, 3, 8), np.int16)
        xyz[:, :, :4] = [1, 2, 3, 4]
        blk = R.encode_block(xyz, [4, 4])
        with pytest.raises(ValueError, match="bytes"):
            R.decode_block(blk[:-1], [4, 4], 8)
        with pytest.raises(ValueError, match="bytes"):
            R.decode_block(blk + b"\0", [4, 4], 8)
        bad = bytearray(blk)
        bad[6] = 17
        with pytest.raises(ValueError, match="width"):
            R.decode_block(bytes(bad), [4, 4], 8)
        with pytest.raises(ValueError, match="width"):
            R.decode_block(blk, [4, 9], 8)
        with pytest.raises(ValueError, match="row"):
            R.decode_block(blk, [], 8)
        with pytest.raises(ValueError, match="reserved"):
            bad = bytearray(blk)
            bad[20] = 1
            R.decode_block(bytes(bad), [4, 4], 8)

    def test_decoded_value_wraps_modulo_2_16(self):
        """Section 5.6: a tampered min + u wraps, never faults."""
        xyz = np.zeros((1, 3, 8), np.int16)
        xyz[0, 0, :2] = [32000, 32767]
        blk = bytearray(R.encode_block(xyz, [2]))
        blk[0:2] = (32700).to_bytes(2, "little", signed=True)   # min + 767 > 32767
        out = R.decode_block(bytes(blk), [2], 8)
        assert out[0, 0, 1] == np.int16(((32700 + 767) & 0xFFFF) - 0x10000)


# ---------------------------------------------------------------------------
# Derived color: section 8
# ---------------------------------------------------------------------------

class TestDerivedColor:
    @pytest.mark.parametrize("s,k,q", [(5, 2, 3), (-5, 2, -3), (7, 3, 2), (-7, 3, -2),
                                       (3, 6, 1), (-3, 6, -1), (0, 4, 0), (9, 1, 9),
                                       (-32768 * 16, 16, -32768), (32767 * 16, 16, 32767)])
    def test_integer_mean_half_away_from_zero(self, s, k, q):
        assert R.integer_mean_half_away(s, k) == q

    def test_single_atom_feature_is_exact_and_layout_follows_the_table(self):
        xyz = np.zeros((3, 8), np.int16)
        xyz[:, :4] = [[-7, 1, 2, 9], [5, 5, -6, 0], [3, 0, -1, -4]]
        feats = R.canonical_feature_order([(2, (3, 1, 0)), (1, (2,)), (1, (0,))])
        assert feats == [(1, (0,)), (1, (2,)), (2, (0, 1, 3))]
        pos = R.derive_positions(xyz, feats)
        assert pos.tolist() == [[-7, 5, 3], [2, -6, -1], [1, 3, 0]]   # 3/3=1, 10/3->3, -1/3->0
        row = R.kernel_layout(xyz, 4, feats, 8)
        assert row[:, 4:7].T.tolist() == pos.tolist() and not row[:, 7].any()
        assert R.derived_counts(4, feats, 3).tolist() == [4, 2, 1, 0]


# ---------------------------------------------------------------------------
# Descriptor: section 10
# ---------------------------------------------------------------------------

class TestParquetPages:
    def test_page_walk_accepts_the_corpus_and_refuses_a_forged_page(self):
        import pyarrow.parquet as pq
        path = DATA / "color" / "molecules.parquet"
        pf = pq.ParquetFile(str(path))
        data = path.read_bytes()
        g = pf.metadata.row_group(0)
        for c in range(g.num_columns):
            col = g.column(c)
            chunk = data[col.data_page_offset:col.data_page_offset + col.total_compressed_size]
            R.check_parquet_pages(chunk, g.num_rows, codec=col.compression)
        col = g.column(2)
        chunk = bytearray(data[col.data_page_offset:col.data_page_offset + col.total_compressed_size])
        # the first header byte is field 1 (type, i32 zigzag): make it a DICTIONARY_PAGE (2 -> zigzag 4)
        assert chunk[0] == 0x15 and chunk[1] == 0x00
        chunk[1] = 0x04
        with pytest.raises(ValueError, match="not a data page"):
            R.check_parquet_pages(bytes(chunk), g.num_rows)
        with pytest.raises(ValueError):
            R.check_parquet_pages(bytes(chunk[:-3]), g.num_rows)
        with pytest.raises(ValueError, match="values"):
            R.check_parquet_pages(data[col.data_page_offset:col.data_page_offset + col.total_compressed_size],
                                  g.num_rows + 1, codec=col.compression)


    def test_page_walk_refuses_containers_and_unknown_fields_fast(self):
        """Review 2: the walker is a closed schema. A list of 2^62 bools (no
        bytes per element) is refused at its field header, not skipped."""
        def varint(n):
            out = bytearray()
            while True:
                b, n = n & 0x7F, n >> 7
                out.append(b | 0x80 if n else b)
                if not n:
                    return bytes(out)
        t0 = time.monotonic()
        for ftype in (9, 10, 11):
            bomb = bytes([0x10 | ftype, 0xF1]) + varint(1 << 62) + b"\x00"
            with pytest.raises(ValueError, match="must be an i32"):
                R.check_parquet_pages(bomb, 1)
            # the same container under an unknown field id
            with pytest.raises(ValueError, match="not allowed"):
                R.check_parquet_pages(bytes([0x90 | ftype, 0xF1]) + varint(1 << 62) + b"\x00", 1)
        assert time.monotonic() - t0 < 0.5
        ok = bytes([0x15, 0x00, 0x15, 0x02, 0x15, 0x02, 0x2C, 0x15, 0x02, 0x15, 0x00,
                    0x15, 0x06, 0x15, 0x06, 0x1C, 0x00, 0x00, 0x00]) + b"\x07"
        R.check_parquet_pages(ok, 1)                          # one PLAIN page of 1 value, 1 byte
        hdr, n = R.parse_page_header(ok)
        assert n == len(ok) - 1 and hdr[5][1] == 1
        with pytest.raises(ValueError, match="repeated or out of order"):
            R.check_parquet_pages(bytes([0x15, 0x00, 0x05, 0x02]) + ok[2:], 1)
        with pytest.raises(ValueError, match="not allowed"):    # statistics with a field
            R.check_parquet_pages(ok[:15] + bytes([0x1C, 0x16, 0x02, 0x00]) + ok[17:], 1)
        with pytest.raises(ValueError, match="holds at least one"):
            R.check_parquet_pages(ok[:8] + b"\x00" + ok[9:], 1)
        with pytest.raises(ValueError, match="truncated or longer"):
            R.check_parquet_pages(ok[:10], 1)
        with pytest.raises(ValueError, match="varint longer"):
            R.check_parquet_pages(bytes([0x15]) + b"\xff" * 300, 1)
        assert R.PAGE_HEADER_MAX == 256

    def test_footer_cap(self):
        assert R.parquet_footer_cap(1) == (1, 64 * 1024 + 1024)
        assert R.parquet_footer_cap(1 << 14) == (1, 64 * 1024 + 1024)
        assert R.parquet_footer_cap((1 << 14) + 1) == (2, 64 * 1024 + 2048)


def test_deep_refuses_a_nonzero_tail_position(tmp_path):
    """The deep tier reads only each axis's last packing group for the tail
    check (5.4); a set bit at a tail position is still found, in a block of
    several groups."""
    rng = np.random.default_rng(5)
    xyz = rng.integers(-500, 500, (3, 3, 100)).astype(np.int16)        # 300 values: 2 groups, 212 tail
    mol = R.Molecule(mol_id=1, coords=xyz, vaa=np.ones(3, np.float32), smiles="C" * 100,
                     ext_id="t", mw=1.0)

    def gm(name, sections, fields, records):
        coords = bytearray(sections["coords"])
        w_x = coords[6]
        assert w_x > 0
        # group 1, last word, lane 15, top bit: position k = 15, j = 256 + 255 >= 300
        coords[32 + 1 * w_x * 32 + 32 * (w_x - 1) + 2 * 15 + 1] |= 0x80
        sections["coords"] = bytes(coords)
        records[0]["sha256"] = R.sha256_hex(sections["coords"])
    root = tmp_path / "tail"
    R.write_corpus(str(root), [mol], color=None, prep_version="0", prep_commit=None, group_mutate=gm)
    R.validate_corpus(str(root), "scan")
    with pytest.raises(R.CorpusValidationError) as e:
        R.validate_corpus(str(root), "deep")
    assert e.value.code == "integrity.padding"


@pytest.mark.parametrize("key,value", [("offset", 1 << 64), ("length", (1 << 63)), ("shape", [1 << 70])])
def test_header_integers_must_fit_int64(tmp_path, key, value):
    """Section 17 item 11: a header integer a native reader would wrap is
    refused by every reader."""
    mol = R.Molecule(mol_id=1, coords=np.zeros((1, 3, 2), np.int16), vaa=np.ones(1, np.float32),
                     smiles="CC", ext_id="x", mw=1.0)

    def hm(name, h):
        h["sections"][-1][key] = value
    root = tmp_path / "big"
    R.write_corpus(str(root), [mol], color=None, prep_version="0", prep_commit=None, header_mutate=hm)
    with pytest.raises(R.CorpusValidationError) as e:
        R.validate_corpus(str(root), "open")
    assert e.value.code in ("schema.value", "schema.type"), str(e.value)
    assert "64-bit" in str(e.value) or "shape" in str(e.value)


class TestFileAccess:
    def test_a_file_truncated_after_open_is_a_refusal_not_sigbus(self, tmp_path):
        """Section 1 item 4: reads are pread with a short-read check."""
        import shutil
        root = tmp_path / "c"
        shutil.copytree(DATA / "shape", root)
        descriptor, fp = R.read_descriptor(str(root))
        entry = descriptor["groups"][0]
        path = root / R.group_file_name(entry["bucket"], entry["group"])
        gf = R.open_group(str(path), root=str(root), descriptor=descriptor, group_entry=entry,
                          fingerprint_hex=fp)
        try:
            os.truncate(path, 64)
            with pytest.raises(R.CorpusValidationError) as e:
                gf.section_bytes("vaa")
            assert e.value.code == "io.short_read"
        finally:
            gf.close()

    def test_corpus_files_open_relative_to_directory_fds(self, tmp_path):
        with R.CorpusDir(str(DATA / "shape")) as cd:
            os.close(cd.open("corpus.json"))
            for bad in ("../shape/corpus.json", "groups/../corpus.json", "groups/x/y.r3g", ".."):
                with pytest.raises(R.CorpusValidationError) as e:
                    cd.open(bad)
                assert e.value.code in ("io.not_regular_file", "io.missing_file")
        # a symlinked corpus root is the user's choice and is followed
        link = tmp_path / "link"
        link.symlink_to(DATA / "shape")
        R.validate_corpus(str(link), "open")


class TestDescriptor:
    def test_fingerprint_is_order_and_rdkit_version_invariant(self):
        d = json.loads((DATA / "color" / "corpus.json").read_bytes())
        fp = R.fingerprint(d)
        assert fp == EXPECTED["color"]["fingerprint"]
        shuffled = json.loads(json.dumps(d, sort_keys=False)[::-1][::-1])
        shuffled["color"] = dict(reversed(list(d["color"].items())))
        assert R.fingerprint(shuffled) == fp
        other = json.loads(json.dumps(d))
        other["color"]["rdkit_version"] = "1999.01.1"
        assert R.fingerprint(other) == fp
        other["color"]["r"]["values"][1][1] = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
        assert R.fingerprint(other) != fp
        other = json.loads(json.dumps(d))
        other["color"]["definition"] += " "
        assert R.fingerprint(other) != fp

    def test_fingerprint_is_identical_in_a_separate_process(self):
        code = ("import json,sys; from roshambo3.format.r3g import reference as R; "
                "print(R.fingerprint(json.loads(open(sys.argv[1],'rb').read())))")
        for variant in ("shape", "color"):
            out = subprocess.run([sys.executable, "-c", code, str(DATA / variant / "corpus.json")],
                                 check=True, capture_output=True, text=True,
                                 env={**os.environ, "PYTHONHASHSEED": "random"})
            assert out.stdout.strip() == EXPECTED[variant]["fingerprint"]

    def test_definition_hash_normalizes_line_endings_only(self):
        assert R.definition_hash("a \r\nb\rc") == R.definition_hash("a \nb\nc")
        assert R.definition_hash("a\nb") != R.definition_hash("a \nb")
        assert R.definition_hash({"b": ["C"], "a": ["N"]}) == R.sha256_hex(b'{"a":["N"],"b":["C"]}')
        text = json.loads((DATA / "color" / "corpus.json").read_bytes())["color"]["definition"]
        assert R.definition_hash(text) == (
            "766f9790513f8f94c67ec6d1c8b6d5bede59b81f6da8c2081f57e49fb28d9784")

    def test_canonical_json(self):
        assert R.canonical_json({"b": 1, "a": [True, None, "é"]}) == b'{"a":[true,null,"\xc3\xa9"],"b":1}'
        with pytest.raises(ValueError):
            R.canonical_json({"x": float("nan")})

    def test_strict_json_refusals(self):
        with pytest.raises(R.CorpusValidationError) as e:
            R.loads_strict(b'{"a":1,"a":2}', what="x")
        assert e.value.code == "json.duplicate_key"
        with pytest.raises(R.CorpusValidationError) as e:
            R.loads_strict(b'{"a":NaN}', what="x")
        assert e.value.code == "json.nan"
        with pytest.raises(R.CorpusValidationError) as e:
            R.loads_strict(b'{"a":' + b"[" * 33 + b"]" * 33 + b"}", what="x")
        assert e.value.code == "json.depth"
        with pytest.raises(R.CorpusValidationError) as e:
            R.loads_strict(b"[1]", what="x")
        assert e.value.code == "json.syntax"
        assert R.loads_strict(b'{"a":' + b"[" * 31 + b"]" * 31 + b"}", what="x") is not None


# ---------------------------------------------------------------------------
# Definition pre-pass: section 15
# ---------------------------------------------------------------------------

class TestDefinitionPrePass:
    def test_base_features_passes(self):
        text = json.loads((DATA / "color" / "corpus.json").read_bytes())["color"]["definition"]
        info = R.check_definition_text(text, "rdkit_fdef")
        # 35, not 33: RDKit's keywords are case-insensitive and BaseFeatures
        # spells two of them "Atomtype"
        assert info["atomtype_lines"] == 35 and info["definefeature_lines"] == 27
        assert info["expanded_longest"] < 1024 and info["expanded_total"] < 8192

    def test_macro_bomb_is_refused_fast(self):
        n = 18
        lines = ["AtomType A0 [#6]"] + [f"AtomType A{i} [{{A{i-1}}},{{A{i-1}}}]" for i in range(1, n + 1)]
        lines += [f"DefineFeature F [{{A{n}}}]", "  Family Hydrophobe", "  Weights 1.0", "EndFeature"]
        t0 = time.monotonic()
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text("\n".join(lines) + "\n", "rdkit_fdef")
        # RDKit's $([...]) wrapping makes the bomb reach depth 9 at level 9,
        # before its expansion passes 8 KiB at level 10
        assert e.value.code == "limit.definition_depth"
        assert time.monotonic() - t0 < R.LIMITS["refuse_seconds"]

    def test_chained_stars_and_nesting(self):
        star = "DefineFeature X " + "~".join(["*"] * 18) + "\n  Family H\n  Weights 1.0\nEndFeature\n"
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text(star, "rdkit_fdef")
        assert e.value.code == "limit.definition_atoms"
        ok = "DefineFeature X " + "~".join(["*"] * 16) + "\n  Family H\nEndFeature\n"
        R.check_definition_text(ok, "rdkit_fdef")
        nest = "DefineFeature X " + "[$(" * 9 + "C" + ")]" * 9 + "\n  Family H\nEndFeature\n"
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text(nest, "rdkit_fdef")
        assert e.value.code == "limit.definition_depth"

    def test_smarts_structure_counts_top_level_atoms_only(self):
        assert R.smarts_structure("[{AromR8}]1:[{AromR8}]:[{AromR8}]:1") == (0, 3)
        assert R.smarts_structure("[$([N;H0]S(=O)(=O))]") == (1, 1)
        assert R.smarts_structure("[#6]-P(=O)(-O)-[C,O,N]-[C,H]") == (0, 6)
        assert R.smarts_structure("Clc1ccccc1Br") == (0, 8)

    def test_depth_counts_recursions_through_branches(self):
        """Review 2: every '(' is pushed (recursion or branch) and every ')'
        pops, so a branch inside a bracket inside a recursion cannot reset
        the depth."""
        for k in (1, 8, 9, 12):
            assert R.smarts_structure("[$(*(*)" * k + "*" + ")]" * k) == (k, 1)
        assert R.smarts_structure("C$(C(C)(C))") == (1, 1)            # branches are not depth
        assert R.smarts_structure("C%(12)CC%(12)") == (0, 3)          # extended ring closure
        assert R.smarts_structure("C))[$($(C))]") == (2, 2)           # stray ')' ignored
        with pytest.raises(R.CorpusValidationError) as e:
            R._check_pattern("[$(*(*)" * 9 + "*" + ")]" * 9, "x")
        assert e.value.code == "limit.definition_depth"

    def test_closed_grammar(self):
        """Section 15: case-insensitive keywords, no continuation lines, no
        stray lines, !Name counts toward Name, {Name} wraps as $([...])."""
        ok = "atomtype X [#6]\nATOMTYPE !X [#7]\nDefineFeature F [{X}]\n Family H\n Weights 1.0\nendfeature\n"
        R.check_definition_text(ok, "rdkit_fdef")
        for bad, code in (("AtomType X \\\n[#6]\n", "value.definition"),
                          ("Include x\n", "value.definition"),
                          ("DefineFeature F [#6]\n Family H\n", "value.definition"),
                          ("Family H\n", "value.definition"),
                          ("DefineFeature F [{Y}]\n Family H\nEndFeature\n", "value.definition")):
            with pytest.raises(R.CorpusValidationError) as e:
                R.check_definition_text(bad, "rdkit_fdef")
            assert e.value.code == code, bad
        deep = "AtomType A0 [#6]\n" + "".join(f"AtomType A{i} [{{A{i-1}}};R]\n" for i in range(1, 10))
        deep += "DefineFeature F [{A9}]\n Family H\nEndFeature\n"
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text(deep, "rdkit_fdef")
        assert e.value.code == "limit.definition_depth"
        # the pattern is the third token; trailing tokens are ignored, as RDKit does
        R.check_definition_text("DefineFeature F [#7] C(=O) # note\n Family H\nEndFeature\n", "rdkit_fdef")
        # blank and comment lines inside a block are fine
        R.check_definition_text("DefineFeature F [#6]\n\n# c\n Family H\nEndFeature\n", "rdkit_fdef")
        # every type must be a Family
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text("DefineFeature F [#6]\n Family H\nEndFeature\n", "rdkit_fdef", ["H", "X"])
        assert e.value.code == "value.definition"

    def test_macro_expansion_is_rdkits(self):
        """Measured on RDKit 2026.09.1: GetFeatureDefs
        of the same definitions."""
        assert R.expand_macro(["[#6,#7]"], []) == "$([#6,#7])"
        assert R.expand_macro(["C"], []) == "$(C)"
        assert R.expand_macro(["[#6]", "[#7]"], ["[#8]", "[O]"]) == "$([!$([O]);!$([#8]);#6,$([#7])])"
        assert R.expand_macro(["[#6]", "[$([#6]);R]"], []) == "$([#6,$([$([#6]);R])])"

    def test_atom_counting_closes_the_cold_read_bypasses(self):
        for pat in ("~".join(["A"] * 18), "C#" + "~".join(["*"] * 18), "~".join(["a"] * 17)):
            with pytest.raises(R.CorpusValidationError) as e:
                R.check_definition_text(f"DefineFeature X {pat}\n Family H\nEndFeature\n", "rdkit_fdef")
            assert e.value.code == "limit.definition_atoms", pat
        assert R.smarts_structure("N#C") == (0, 2) and R.smarts_structure("[se]1ccccc1") == (0, 6)

    def test_smarts_list_limits(self):
        good = {"A": ["[#6]", "c1ccccc1"]}
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text({"A": []}, "smarts_list")
        assert e.value.code == "value.definition"
        assert R.check_definition_text(good, "smarts_list")["patterns"] == 2
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text({"A": ["C"] * 257}, "smarts_list")
        assert e.value.code == "limit.definition_patterns"
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text("[#6]", "smarts_list")
        assert e.value.code == "value.definition"
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text(good, "smarts_list", types=["A", "B"])
        assert e.value.code == "value.definition"
        assert R.check_definition_text(good, "smarts_list", types=["A"])["families"] == 1


# ---------------------------------------------------------------------------
# The conformance corpus
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["shape", "color"])
class TestConformanceCorpus:
    def test_validates_at_every_tier(self, variant):
        for tier in ("open", "scan", "deep"):
            report = R.validate_corpus(str(DATA / variant), tier)
            assert report.fingerprint == EXPECTED[variant]["fingerprint"]

    def test_decoded_arrays_match_expected(self, variant):
        root = DATA / variant
        descriptor, fp = R.read_descriptor(str(root))
        exp = EXPECTED[variant]
        assert descriptor["ids_sha256"] == exp["ids_sha256"]
        assert descriptor["molecules_sha256"] == exp["molecules_sha256"]
        assert set(exp["groups"]) == {R.group_file_name(g["bucket"], g["group"]) for g in descriptor["groups"]}
        for entry in descriptor["groups"]:
            name = R.group_file_name(entry["bucket"], entry["group"])
            path = root / name
            g = exp["groups"][name]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == g["file_sha256"]
            gf = R.open_group(str(path), root=str(root), descriptor=descriptor,
                              group_entry=entry, fingerprint_hex=fp)
            try:
                d = R.decode_group(gf, descriptor)
                for key in ("heavy", "merged", "counts", "pseudo", "runs", "vaa"):
                    assert _digest(d[key]) == g[f"{key}_sha256"], (name, key)
                if d["features"] is not None:
                    for k, v in d["features"].items():
                        assert _digest(v) == g[f"{k}_sha256"], (name, k)
                if d["color_self"] is not None:
                    assert _digest(d["color_self"]) == g["color_self_sha256"]
                blocks = gf.section_array("blocks")
                assert [int(b["rows"]) for b in blocks] == g["block_rows"]
                assert [list(R.parse_block_header(gf.block_bytes(i)).widths)
                        for i in range(blocks.size)] == g["block_widths"]
            finally:
                gf.close()

    def test_required_cases_are_present(self, variant):
        """The ticket's list: a 1-atom molecule, a w = 0 axis, a partial tail
        group, a block of exactly 4,096 rows in each corpus, an
        n_real = 255 molecule, multi-atom features, several groups and buckets."""
        exp = EXPECTED[variant]
        rows = [r for g in exp["groups"].values() for r in g["block_rows"]]
        widths = [w for g in exp["groups"].values() for ws in g["block_widths"] for w in ws]
        assert 0 in widths and 16 >= max(widths) >= 9
        assert 4096 in rows and max(rows) <= 4096
        assert len(exp["groups"]) >= 8
        assert len({n.split("_")[0] for n in exp["groups"]}) >= 8          # buckets
        assert max(g["L_heavy"] for g in exp["groups"].values()) == 256   # n_real 255
        root = DATA / variant
        descriptor, _ = R.read_descriptor(str(root))
        n_real = []
        multi = 0
        tails = 0
        for entry in descriptor["groups"]:
            gf = R.open_group(str(root / R.group_file_name(entry["bucket"], entry["group"])))
            runs = gf.section_array("runs")
            n_real += runs["n_real"].tolist()
            for b in gf.section_array("blocks"):
                total = int((runs["n_conf"][b["m0"]:b["m1"]].astype(int)
                             * runs["n_real"][b["m0"]:b["m1"]].astype(int)).sum())
                tails += total % 256 != 0
            if variant == "color":
                fao = gf.section_array("feature_atom_offsets")
                multi += int((np.diff(fao) > 1).sum())
            gf.close()
        assert min(n_real) == 1 and max(n_real) == 255 and tails > 0
        if variant == "color":
            assert multi > 0
            assert max(g["L"] for g in exp["groups"].values()) == 512


WIDTHS = json.loads((DATA / "widths" / "EXPECTED.json").read_text())


@pytest.mark.parametrize("name", sorted(WIDTHS))
def test_width_vector_decodes_to_its_digest(name):
    """Per-width oracle blocks for the SIMD paths, including the wrap."""
    v = WIDTHS[name]
    blk = (DATA / "widths" / name).read_bytes()
    assert list(R.parse_block_header(blk).widths) == v["widths"]
    out = R.decode_block(blk, v["n_real"], v["L"])
    assert _digest(out) == v["decoded_sha256"]
    if "decoded" in v:
        assert out[0].tolist() == v["decoded"]
        assert out[0, 0, 2] < 0            # 32600 + 767 wrapped


def test_open_tier_is_fast_on_every_conformance_file():
    """Limit 33: 20 s per file for open; the whole corpus here takes well under."""
    for variant in ("shape", "color"):
        t0 = time.monotonic()
        R.validate_corpus(str(DATA / variant), "open")
        assert time.monotonic() - t0 < R.LIMITS["refuse_seconds"]


def test_directory_is_within_the_byte_budget():
    total = sum(p.stat().st_size for p in DATA.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts)
    assert total <= BUDGET, total


# ---------------------------------------------------------------------------
# Adversarial seeds
# ---------------------------------------------------------------------------

TIERS = ("open", "scan", "deep")


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_seed_is_refused_by_the_stated_tier(name, tmp_path):
    exp = ADVERSARIAL[name]
    if exp["generated"]:
        sys.path.insert(0, str(DATA / "adversarial"))
        try:
            import make_adversarial
        finally:
            sys.path.pop(0)
        root = tmp_path / name
        make_adversarial.build(name, str(root))
    else:
        root = DATA / "adversarial" / name
    if (root / "definition.fdef").exists() or (root / "definition.json").exists():
        # not a corpus: a definition handed straight to the pre-pass, as the
        # trust path does
        if (root / "definition.json").exists():
            d = json.loads((root / "definition.json").read_bytes())
            args = (d["definition"], d["generator"], d["types"])
        else:
            args = ((root / "definition.fdef").read_text(), "rdkit_fdef", None)
        t0 = time.monotonic()
        with pytest.raises(R.CorpusValidationError) as e:
            R.check_definition_text(*args)
        assert e.value.code == exp["code"], str(e.value)
        assert time.monotonic() - t0 < R.LIMITS["refuse_seconds"]
        return
    # every tier below the stated one accepts the seed: it fails the check it
    # targets, not one in front of it
    for tier in TIERS[:TIERS.index(exp["tier"])]:
        R.validate_corpus(str(root), tier)
    t0 = time.monotonic()
    with pytest.raises(R.CorpusValidationError) as e:
        R.validate_corpus(str(root), exp["tier"])
    elapsed = time.monotonic() - t0
    assert e.value.code == exp["code"], str(e.value)
    assert e.value.tier == exp["tier"] or exp["tier"] == "deep" and e.value.tier in ("scan", "deep")
    assert elapsed < R.LIMITS["refuse_seconds"], (name, elapsed)


def test_every_threat_model_row_has_a_seed():
    rows = " ".join(v["threat_model"] for v in ADVERSARIAL.values())
    for needle in ("bit width", "stream holds", "n_real > L_heavy", "device L", "block rows",
                   "FIFO", "symlink", "macro bomb", "lowercase", "negation", "continu",
                   "A/a primitives", "triple bond", "every type is a Family",
                   "offset outside", "overlap", "unaligned", "prod(shape)", "run sums",
                   "do not tile", "feature type", "atom index >= n_real", "atoms per centroid: 0",
                   "atoms per centroid: huge", "duplicate keys", "NaN", "uncapped header",
                   "path-like", "unknown major version", "macro bomb", "18 chained", "nested"):
        assert needle in rows, needle


def test_codes_are_from_the_closed_enumeration():
    spec = (pathlib.Path(__file__).resolve().parent.parent / "doc" / "FORMAT.md").read_text()
    for v in ADVERSARIAL.values():
        code = v["code"]
        head = code.split(".")[0] + "." + ("<name>" if code.startswith("limit.") else code.split(".")[1])
        assert f"`{head}`" in spec, code
        if code.startswith("limit."):
            assert f"`{code.split('.')[1]}`" in spec, code

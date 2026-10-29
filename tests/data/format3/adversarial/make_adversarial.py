#!/usr/bin/env python3
"""Write the format-3 adversarial seed set (``doc/FORMAT.md`` section 16).

One minimal corpus directory per row of the format threat model's section
2.2 table and per container rule of its section 2.3, plus the two
definition-text attacks of its appendix. Every seed is a tiny valid corpus
(two one-conformer molecules, or a six-atom molecule with two features for
the color cases) with exactly one thing wrong, produced through the reference
writer's mutation hooks so that every hash *in front of* the targeted check is
correct: a seed must fail the check it targets, not an integrity check.

``EXPECTED.json`` maps each seed to the validator tier that must refuse it and
the refusal code. Seeds marked ``generated`` are too large to commit and are
materialized by ``build(name, directory)`` at test time.

    PYTHONPATH=src python tests/data/format3/adversarial/make_adversarial.py
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from roshambo3.format.r3g import reference as R  # noqa: E402

TINY_FDEF = """# a minimal definition for the adversarial seeds
AtomType Carbon [#6]
DefineFeature Hphobe [{Carbon}]
  Family Hydrophobe
  Weights 1.0
EndFeature
DefineFeature Acid [C,S](=[O,S,P])-[O;H1,H0&-1]
  Family NegIonizable
  Weights 1.0,1.0,1.0
EndFeature
"""
TYPES = ["Hydrophobe", "NegIonizable"]
#: appended to every custom rdkit_fdef seed text so that each of TYPES is a
#: Family of the definition (FORMAT.md 10.2; the pre-pass checks it last)
ACID_BLOCK = """DefineFeature Acid [C,S](=[O,S,P])-[O;H1,H0&-1]
  Family NegIonizable
  Weights 1.0,1.0,1.0
EndFeature
"""


def _base_molecules(color: bool) -> list[R.Molecule]:
    """Two molecules: a 1-atom one and a 4-atom acetate-like one (with a
    1-atom Hydrophobe and a 3-atom NegIonizable feature when ``color``)."""
    a = R.Molecule(mol_id=11, coords=np.zeros((1, 3, 1), np.int16),
                   vaa=np.array([1.5], np.float32), ext_id="one", smiles="C",
                   mw=16.0, rotors=0, color_self=np.zeros(1) if color else None)
    xyz = np.array([[[-3, 0, 3, 1], [-2, 2, -1, 4], [0, 0, 0, 0]]], np.int16)   # small widths
    feats = [(1, (0,)), (2, (1, 2, 3))] if color else []
    b = R.Molecule(mol_id=22, coords=xyz, vaa=np.array([6.25], np.float32),
                   features=feats, ext_id="acetate", smiles="CC(=O)[O-]",
                   mw=59.0, rotors=0,
                   color_self=np.array([3.0]) if color else None)
    return [a, b]


def _color(definition=TINY_FDEF, off_diagonal=0.0):
    n = len(TYPES)
    r = np.zeros((n + 1, n + 1), np.float32)
    p = np.zeros((n + 1, n + 1), np.float32)
    for t in range(1, n + 1):
        r[t, t] = p[t, t] = 1.0
    p[1, 2] = off_diagonal
    return R.color_descriptor(generator="rdkit_fdef", definition=definition, types=TYPES,
                              r=r, p=p, rdkit_version="2026.09.1")


def _write_definition(root, definition, generator="rdkit_fdef", types=TYPES):
    """A bare pre-pass seed: no corpus, just the definition and its types, as
    the trust path hands them to `check_definition_text` (~1 KB instead
    of a ~4 KB corpus)."""
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    with open(os.path.join(root, "definition.json"), "wb") as fh:
        fh.write(R.canonical_json({"generator": generator, "definition": definition, "types": list(types)}))


def _write(root, *, color=False, molecules=None, definition=TINY_FDEF, off_diagonal=0.0,
           raw_definition=None, **hooks):
    shutil.rmtree(root, ignore_errors=True)
    mols = molecules if molecules is not None else _base_molecules(color)
    cd = _color(definition, off_diagonal) if color else None
    if raw_definition is not None:
        cd["definition"] = raw_definition       # bypass the writer's normalization
    return R.write_corpus(root, mols, color=cd,
                          prep_version="0.1.0.dev0", prep_commit=None,
                          block_rows=4096, **hooks)


def _patch_file(path, offset, data: bytes):
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(data)


def _group0(root):
    return os.path.join(root, R.group_file_name(0, 0))


def _patch_block(sections, records, i, offset, data: bytes):
    """Edit block ``i``'s bytes and recompute its hash."""
    coords = bytearray(sections["coords"])
    start = records[i]["offset"] + offset
    coords[start:start + len(data)] = data
    sections["coords"] = bytes(coords)
    blk = sections["coords"][records[i]["offset"]:records[i]["offset"] + records[i]["length"]]
    records[i]["sha256"] = R.sha256_hex(blk)


def _descriptor_patch(fn):
    """Rewrite corpus.json after the fact (no hash covers it)."""
    def patch(root):
        p = os.path.join(root, "corpus.json")
        d = json.loads(open(p, "rb").read().decode("utf-8"))
        text = fn(d)
        with open(p, "wb") as fh:
            fh.write(text if isinstance(text, bytes) else R.canonical_json(d))
    return patch


# --------------------------------------------------------------------------
# The seeds. Each entry: name -> (builder, tier, code, threat-model row).
# --------------------------------------------------------------------------

def seed_magic_rsb3(root):
    _write(root)
    _patch_file(_group0(root), 0, b"RSB3")


def seed_major_version_4(root):
    _write(root)
    _patch_file(_group0(root), 4, struct.pack("<I", 4))


def seed_descriptor_major_4(root):
    _write(root)
    _descriptor_patch(lambda d: d["format"].__setitem__("major", 4))(root)


def seed_reserved_prologue_nonzero(root):
    _write(root)
    _patch_file(_group0(root), 40, b"\x01")


def seed_truncated_file(root):
    _write(root)
    p = _group0(root)
    size = os.path.getsize(p)
    with open(p, "r+b") as fh:
        fh.truncate(100)            # header_offset now points past EOF


def seed_oversize_header(root):
    """Generated: the header is padded past 16 MiB with spaces (JSON allows
    trailing whitespace, so only the size check can refuse it cheaply)."""
    _write(root)
    with open(_group0(root), "ab") as fh:
        fh.write(b" " * (16 * 1024 * 1024 + 1))


def seed_json_duplicate_key(root):
    _write(root)
    _descriptor_patch(lambda d: R.canonical_json(d)[:-1] + b',"n_mol":2}')(root)


def seed_json_nan(root):
    _write(root, color=True)
    def fn(d):
        text = R.canonical_json(d)
        return text.replace(b'"values":[[0.0,0.0,0.0]', b'"values":[[NaN,0.0,0.0]', 1)
    _descriptor_patch(fn)(root)


def seed_json_depth(root):
    _write(root)
    _descriptor_patch(lambda d: d.__setitem__("x", json.loads("[" * 33 + "]" * 33)))(root)


def seed_group_path_entry(root):
    _write(root)
    _descriptor_patch(lambda d: d["groups"][0].__setitem__("file", "../../etc/passwd"))(root)


def seed_group_index_string(root):
    _write(root)
    _descriptor_patch(lambda d: d["groups"][0].__setitem__("group", "../x"))(root)


def seed_encoding_unknown(root):
    _write(root)
    _descriptor_patch(lambda d: d["format"].__setitem__("encoding", "zstd-19"))(root)


def seed_step_wrong(root):
    _write(root)
    _descriptor_patch(lambda d: d["format"].__setitem__("step_log2", -10))(root)


def seed_section_unknown(root):
    def hm(name, h):
        h["sections"][-1]["name"] = "counts"
    _write(root, header_mutate=hm)


def seed_section_outside_file(root):
    def hm(name, h):
        h["sections"][-1]["offset"] = 1 << 40
    _write(root, header_mutate=hm)


def seed_section_overlap(root):
    def hm(name, h):
        h["sections"][1]["offset"] = h["sections"][0]["offset"]
    _write(root, header_mutate=hm)


def seed_section_misaligned(root):
    def hm(name, h):
        for e in h["sections"][1:]:
            e["offset"] += 8
    _write(root, header_mutate=hm)


def seed_section_length_mismatch(root):
    def hm(name, h):
        h["sections"][2]["length"] += 1
    _write(root, header_mutate=hm)


def seed_width_17(root):
    def gm(name, sections, fields, records):
        _patch_block(sections, records, 0, 6, b"\x11")
    _write(root, group_mutate=gm)


def seed_block_length_short(root):
    def gm(name, sections, fields, records):
        # drop the last 32 bytes of the single block; keep the tables tiling
        coords = sections["coords"][:-32]
        sections["coords"] = coords
        records[0]["length"] -= 32
        records[0]["sha256"] = R.sha256_hex(coords)
    _write(root, group_mutate=gm)


def seed_block_reserved_nonzero(root):
    def gm(name, sections, fields, records):
        _patch_block(sections, records, 0, 20, b"\x07")
    _write(root, group_mutate=gm)


def seed_block_rows_4097(root):
    """Two one-atom molecules (4,096 + 1 conformers, every coordinate zero so
    the payload is 32 bytes) merged into one 4,097-row block."""
    a = R.Molecule(mol_id=11, coords=np.zeros((4096, 3, 1), np.int16),
                   vaa=np.full(4096, 1.5, np.float32), ext_id="one", smiles="C", mw=16.0)
    b = R.Molecule(mol_id=22, coords=np.zeros((1, 3, 1), np.int16),
                   vaa=np.array([1.5], np.float32), ext_id="two", smiles="N", mw=17.0)
    def gm(name, sections, fields, records):
        data = R.encode_block(np.zeros((4097, 3, 8), np.int16), np.ones(4097, np.int64))
        sections["coords"] = data
        records[:] = [{"first_row": 0, "rows": 4097, "offset": 0, "length": len(data),
                       "m0": 0, "m1": 2, "sha256": R.sha256_hex(data)}]
    _write(root, molecules=[a, b], group_mutate=gm)


def seed_block_molecule_gap(root):
    def gm(name, sections, fields, records):
        records[0]["m1"] = 1          # the block claims one molecule, holds two
    _write(root, group_mutate=gm)


def seed_feature_prefix_reversed(root):
    mols = _base_molecules(True)
    mols[1].features = [(1, (0,)), (1, (0, 1))]
    def gm(name, sections, fields, records):
        # Writer sorts prefix first; replace only the feature CSR with reversed order.
        for key, arr in R.pack_features([[], [(1, (0, 1)), (1, (0,))]]).items():
            sections[key] = arr.tobytes()
    _write(root, color=True, molecules=mols, group_mutate=gm)


def seed_block_rows_mismatch(root):
    def gm(name, sections, fields, records):
        runs = np.frombuffer(sections["runs"], dtype=R.RUN_DTYPE).copy()
        runs[0]["n_conf"] = 2         # the run table now sums to 3 rows
        sections["runs"] = runs.tobytes()
    _write(root, group_mutate=gm)


def seed_n_real_zero(root):
    def gm(name, sections, fields, records):
        runs = np.frombuffer(sections["runs"], dtype=R.RUN_DTYPE).copy()
        runs[0]["n_real"] = 0
        sections["runs"] = runs.tobytes()
    _write(root, group_mutate=gm)


def seed_n_real_exceeds_L_heavy(root):
    def hm(name, h):
        h["L_heavy"] = 8
        h["L"] = 8
    mols = _base_molecules(False)
    mols[1].coords = np.zeros((1, 3, 9), np.int16)   # n_real 9 -> L_heavy 16
    _write(root, molecules=mols, header_mutate=hm,
           descriptor_mutate=lambda d: [g.update(L_heavy=8, L=8) for g in d["groups"]])


def seed_n_color_exceeds_L(root):
    def hm(name, h):
        h["L"] = 8                   # n_real 4 + 2 features fits 8; make it not
    mols = _base_molecules(True)
    mols[1].features = [(1, (0,)), (1, (1,)), (1, (2,)), (1, (3,)), (2, (1, 2, 3))]  # 4 + 5 = 9 -> L 16
    _write(root, color=True, molecules=mols, header_mutate=hm,
           descriptor_mutate=lambda d: [g.update(L=8) for g in d["groups"]])


def seed_L_field_width_wrong(root):
    _write(root, descriptor_mutate=lambda d: [g.update(L_field_width=16) for g in d["groups"]])


def seed_runs_unsorted(root):
    mols = _base_molecules(False)
    mols[0].coords = np.zeros((1, 3, 4), np.int16)   # same n_real as molecule 22, higher id
    mols[0].mol_id = 33
    def gm(name, sections, fields, records):
        runs = np.frombuffer(sections["runs"], dtype=R.RUN_DTYPE).copy()
        sections["runs"] = runs[::-1].copy().tobytes()
    _write(root, molecules=mols, group_mutate=gm)


def seed_mol_id_2_63(root):
    def gm(name, sections, fields, records):
        runs = np.frombuffer(sections["runs"], dtype=R.RUN_DTYPE).copy()
        runs[1]["mol_id"] = 1 << 63
        sections["runs"] = runs.tobytes()
    _write(root, group_mutate=gm)


def seed_feature_type_out_of_range(root):
    def gm(name, sections, fields, records):
        ft = np.frombuffer(sections["feature_types"], dtype="u1").copy()
        ft[-1] = 3
        sections["feature_types"] = ft.tobytes()
    _write(root, color=True, group_mutate=gm)


def seed_feature_atom_out_of_range(root):
    def gm(name, sections, fields, records):
        fa = np.frombuffer(sections["feature_atoms"], dtype="u1").copy()
        fa[-1] = 4                     # n_real is 4
        sections["feature_atoms"] = fa.tobytes()
    _write(root, color=True, group_mutate=gm)


def seed_feature_zero_atoms(root):
    def gm(name, sections, fields, records):
        fao = np.frombuffer(sections["feature_atom_offsets"], dtype="<u4").copy()
        fao[1] = fao[0]                # first feature now has zero atoms
        sections["feature_atom_offsets"] = fao.tobytes()
    _write(root, color=True, group_mutate=gm)


def seed_feature_too_many_atoms(root):
    mols = _base_molecules(True)
    mols[1].coords = np.zeros((1, 3, 20), np.int16)
    mols[1].features = [(2, tuple(range(17)))]
    _write(root, color=True, molecules=mols)


def seed_features_unsorted(root):
    def gm(name, sections, fields, records):
        ft = np.frombuffer(sections["feature_types"], dtype="u1").copy()
        fao = np.frombuffer(sections["feature_atom_offsets"], dtype="<u4").copy()
        fa = np.frombuffer(sections["feature_atoms"], dtype="u1").copy()
        # swap the two features of molecule 22: (2, (1,2,3)) before (1, (0,))
        sections["feature_types"] = ft[::-1].copy().tobytes()
        sections["feature_atoms"] = np.concatenate([fa[1:4], fa[:1]]).tobytes()
        sections["feature_atom_offsets"] = np.array([0, 3, 4], "<u4").tobytes()
    _write(root, color=True, group_mutate=gm)


def seed_n_types_7(root):
    def dm(d):
        d["color"]["types"] = ["A", "B", "C", "D", "E", "F", "G"]
        d["color"]["r"] = {"dtype": "f32", "shape": [8, 8], "values": [[0.0] * 8] * 8}
        d["color"]["p"] = {"dtype": "f32", "shape": [8, 8], "values": [[0.0] * 8] * 8}
    _write(root, color=True, descriptor_mutate=dm)


def seed_matrix_negative_width(root):
    _write(root, color=True,
           descriptor_mutate=lambda d: d["color"]["r"]["values"][1].__setitem__(1, -1.0))


def seed_vaa_negative(root):
    def gm(name, sections, fields, records):
        v = np.frombuffer(sections["vaa"], dtype="<f4").copy()
        v[0] = -1.0
        sections["vaa"] = v.tobytes()
    _write(root, group_mutate=gm)


def seed_color_self_nan(root):
    def gm(name, sections, fields, records):
        v = np.frombuffer(sections["color_self"], dtype="<f8").copy()
        v[0] = np.nan
        sections["color_self"] = v.tobytes()
    _write(root, color=True, group_mutate=gm)


def seed_fingerprint_mismatch(root):
    _write(root, header_mutate=lambda name, h: h.__setitem__("fingerprint", "0" * 64))


def seed_header_hash_mismatch(root):
    _write(root, descriptor_mutate=lambda d: d["groups"][0].__setitem__("header_sha256", "f" * 64))


def seed_block_hash_mismatch(root):
    def gm(name, sections, fields, records):
        # flip a stream byte without updating the block hash; the section hash
        # (computed later, over the flipped bytes) stays right
        coords = bytearray(sections["coords"])
        coords[32] ^= 0xFF
        sections["coords"] = bytes(coords)
    _write(root, group_mutate=gm)


def seed_section_hash_mismatch(root):
    _write(root)
    p = _group0(root)
    d = R.open_group(p)
    off = d.sections["vaa"]["offset"]
    d.close()
    _patch_file(p, off, b"\x00\x00\x20\x40")     # 2.5 as f32: finite, positive, wrong


def seed_ids_hash_mismatch(root):
    _write(root, descriptor_mutate=lambda d: d.__setitem__("ids_sha256", "a" * 64))


def _rewrite_parquet(root, fn, **write_kw):
    """Rewrite molecules.parquet through ``fn(table) -> table`` and re-record
    its hash in corpus.json, so the seed reaches the check it targets."""
    import pyarrow.parquet as pq
    p = os.path.join(root, "molecules.parquet")
    t = fn(pq.read_table(p))
    kw = dict(compression="none", write_statistics=False, store_schema=False, use_dictionary=False)
    kw.update(write_kw)
    pq.write_table(t, p, **kw)
    _descriptor_patch(lambda d: d.__setitem__("molecules_sha256", R.sha256_hex(open(p, "rb").read())))(root)


def seed_parquet_schema(root):
    _write(root)
    import pyarrow as pa
    _rewrite_parquet(root, lambda t: t.append_column("extra", pa.array([0, 0], pa.int32())))


def seed_parquet_unsupported_codec(root):
    _write(root)
    _rewrite_parquet(root, lambda t: t, compression="snappy")


def seed_parquet_dictionary(root):
    _write(root)
    _rewrite_parquet(root, lambda t: t, use_dictionary=True)


def seed_parquet_nullable(root):
    _write(root)
    import pyarrow as pa
    def fn(t):
        schema = pa.schema([pa.field(f.name, f.type, nullable=True) for f in t.schema])
        return t.cast(schema)
    _rewrite_parquet(root, fn)


def seed_parquet_row_count(root):
    _write(root)
    _rewrite_parquet(root, lambda t: t.slice(0, 1))


def seed_parquet_set_mismatch(root):
    _write(root)
    import pyarrow as pa
    def fn(t):
        ids = t.column("mol_id").to_pylist()
        ids[-1] += 1
        return t.set_column(0, pa.field("mol_id", pa.uint64(), nullable=False), pa.array(ids, pa.uint64()))
    _rewrite_parquet(root, fn)


def seed_parquet_hash_mismatch(root):
    _write(root, descriptor_mutate=lambda d: d.__setitem__("molecules_sha256", "b" * 64))


def _patch_parquet_page(root, column, edit):
    """Edit the page bytes of ``column``'s chunk in row group 0 in place --
    the footer is left exactly as written (honest) -- and re-record
    ``molecules_sha256`` so the seed reaches the page-header walk.
    ``edit(chunk: bytearray)`` mutates the chunk's bytes, keeping its length."""
    import pyarrow.parquet as pq
    p = os.path.join(root, "molecules.parquet")
    col = pq.ParquetFile(p).metadata.row_group(0).column(
        [n for n, _ in R.PARQUET_COLUMNS].index(column))
    data = bytearray(open(p, "rb").read())
    off, ln = col.data_page_offset, col.total_compressed_size
    chunk = bytearray(data[off:off + ln])
    edit(chunk)
    assert len(chunk) == ln
    data[off:off + ln] = chunk
    with open(p, "wb") as fh:
        fh.write(bytes(data))
    _descriptor_patch(lambda d: d.__setitem__("molecules_sha256", R.sha256_hex(bytes(data))))(root)


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


#: the first bytes of every page header pyarrow writes for the seeds: field 1
#: (type, i32) = DATA_PAGE, then field 2 (uncompressed size) and field 3
#: (compressed size), each a one- or two-byte zigzag varint
def _page_header_prefix(chunk):
    assert chunk[0] == 0x15 and chunk[1] == 0x00, chunk[:4].hex()


def seed_parquet_page_bool_list(root):
    """The footer is honest; the first page header of the mol_id chunk is
    replaced by field 1 as a compact-protocol list of 2^62 bools. A bool
    element consumes no bytes, so a generic Thrift skipper loops ~2^62 times
    on 12 bytes (review 2: hours at 10^7 elements per 0.45 s)."""
    _write(root)

    def edit(chunk):
        _page_header_prefix(chunk)
        bomb = bytes([0x19, 0xF1]) + _varint(1 << 62) + b"\x00"
        chunk[:len(bomb)] = bomb
    _patch_parquet_page(root, "mol_id", edit)


def seed_parquet_page_type_dictionary(root):
    """Footer honest (no dictionary page, PLAIN); the page header itself says
    DICTIONARY_PAGE (type 2, zigzag 4)."""
    _write(root)

    def edit(chunk):
        _page_header_prefix(chunk)
        chunk[1] = 0x04
    _patch_parquet_page(root, "smiles", edit)


def seed_parquet_page_size_mismatch(root):
    """Footer honest; the page header's uncompressed_page_size differs from
    its compressed_page_size (a compressed page in disguise)."""
    _write(root)

    def edit(chunk):
        _page_header_prefix(chunk)
        assert chunk[2] == 0x15 and chunk[3] < 0x80      # field 2, one-byte varint
        chunk[3] += 2                                     # +1 in zigzag
    _patch_parquet_page(root, "mol_id", edit)


def seed_parquet_page_encoding_dictionary(root):
    """Footer honest (encodings PLAIN/RLE); the data page header's encoding is
    RLE_DICTIONARY (8, zigzag 0x10)."""
    _write(root)

    def edit(chunk):
        _page_header_prefix(chunk)
        i = chunk.index(bytes([0x2C, 0x15]))             # field 5 (struct), then its field 1
        j = i + 2
        while chunk[j] & 0x80:                           # skip num_values
            j += 1
        j += 1
        assert chunk[j] == 0x15 and chunk[j + 1] == 0x00  # field 2: encoding = PLAIN
        chunk[j + 1] = 0x10
    _patch_parquet_page(root, "ext_id", edit)


def seed_parquet_footer_oversize(root):
    """Generated: the 4-byte footer length claims 70 KiB (junk in front of
    the real footer), beyond the 64 KiB + 1 KiB-per-row-group bound for two
    molecules. Refused from the trailing 8 bytes, before pyarrow parses it."""
    _write(root)
    p = os.path.join(root, "molecules.parquet")
    data = open(p, "rb").read()
    flen = struct.unpack("<I", data[-8:-4])[0]
    junk = b"\x00" * (70 * 1024 - flen)
    out = data[:-8 - flen] + junk + data[-8 - flen:-8] + struct.pack("<I", len(junk) + flen) + b"PAR1"
    with open(p, "wb") as fh:
        fh.write(out)
    _descriptor_patch(lambda d: d.__setitem__("molecules_sha256", R.sha256_hex(out)))(root)


def seed_groups_dir_symlink(root):
    """Generated: groups/ itself is a symlink to a directory outside the
    corpus. The file names are derived, but a path resolved through groups/
    would follow it; opening groups/ with O_NOFOLLOW relative to the root fd
    refuses it."""
    _write(root)
    outside = os.path.join(os.path.dirname(os.path.abspath(root)), "outside_groups_" + os.path.basename(root))
    shutil.rmtree(outside, ignore_errors=True)
    os.replace(os.path.join(root, "groups"), outside)
    os.symlink(outside, os.path.join(root, "groups"))


def seed_hac_mismatch(root):
    _write(root)
    import pyarrow as pa
    _rewrite_parquet(root, lambda t: t.set_column(3, pa.field("hac", pa.uint8(), nullable=False),
                                                  pa.array([1, 5], pa.uint8())))


def seed_smiles_too_long(root):
    mols = _base_molecules(False)
    mols[1].smiles = "C" * 5000
    _write(root, molecules=mols)


def seed_smiles_cx(root):
    mols = _base_molecules(False)
    mols[1].smiles = "CC(=O)[O-] |$;;;;$|"
    _write(root, molecules=mols)


def seed_ext_id_control(root):
    mols = _base_molecules(False)
    mols[1].ext_id = "acetate\x1b[31m"
    _write(root, molecules=mols)


def seed_macro_bomb(root):
    n = 18
    lines = ["AtomType A0 [#6]"] + [f"AtomType A{i} [{{A{i-1}}},{{A{i-1}}}]" for i in range(1, n + 1)]
    lines += [f"DefineFeature F [{{A{n}}}]", "  Family Hydrophobe", "  Weights 1.0", "EndFeature"]
    text = "\n".join(lines) + "\n" + ACID_BLOCK
    _write(root, color=True, definition=text)


def seed_chained_star_18(root):
    text = ("DefineFeature Chain " + "~".join(["*"] * 18) + "\n  Family Hydrophobe\n"
            "  Weights " + ",".join(["1.0"] * 18) + "\nEndFeature\n" + ACID_BLOCK)
    _write(root, color=True, definition=text)


def seed_nested_smarts(root):
    d = 100
    text = ("DefineFeature Nest " + "[$(" * d + "C" + ")]" * d + "\n  Family Hydrophobe\n"
            "  Weights 1.0\nEndFeature\n" + ACID_BLOCK)
    _write_definition(root, text)


def seed_definition_oversize(root):
    """Generated: a definition past the 64 KiB text limit. Stored in
    corpus.json it is caught by the JSON string limit (the same 64 KiB) at
    ``open``; the pre-pass's own size limit is reachable only from text that
    did not come through strict JSON (the file path)."""
    text = TINY_FDEF + "# " + "x" * (64 * 1024) + "\n"
    _write(root, color=True, definition=text, descriptor_mutate=lambda d: None)


def seed_definition_oversize_prepass(root):
    """Generated: not a corpus -- a definition text handed straight to the
    pre-pass (the file path), 64 KiB + 1 of comment."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "definition.fdef"), "w", encoding="utf-8") as fh:
        fh.write(TINY_FDEF + "# " + "x" * (64 * 1024) + "\n")


def seed_definition_undefined_macro(root):
    text = "DefineFeature F [{Missing}]\n  Family Hydrophobe\n  Weights 1.0\nEndFeature\n" + ACID_BLOCK
    _write_definition(root, text)


def seed_feature_offsets_bad(root):
    def gm(name, sections, fields, records):
        fo = np.frombuffer(sections["feature_offsets"], dtype="<u4").copy()
        fo[1] = fo[2] + 1                    # not monotone
        sections["feature_offsets"] = fo.tobytes()
    _write(root, color=True, group_mutate=gm)


def seed_features_256(root):
    mols = _base_molecules(True)
    mols[1].coords = np.zeros((1, 3, 16), np.int16)
    mols[1].features = [(1, (a % 16,)) for a in range(256)]
    _write(root, color=True, molecules=mols)


def seed_n_conf_zero(root):
    def gm(name, sections, fields, records):
        runs = np.frombuffer(sections["runs"], dtype=R.RUN_DTYPE).copy()
        runs[0]["n_conf"] = 0
        sections["runs"] = runs.tobytes()
    _write(root, group_mutate=gm)


def seed_duplicate_mol_id_across_groups(root):
    mols = _base_molecules(False)
    big = R.Molecule(mol_id=22, coords=np.zeros((1, 3, 30), np.int16),
                     vaa=np.array([9.0], np.float32), ext_id="dup", smiles="C" * 30, mw=1.0)
    # Targets the cross-group duplicate check of 16.3 item 8 (layout.runs). The
    # parquet is rewritten to one row per distinct id while n_mol stays 3, so
    # the parquet row-count check (value.parquet) would fire too; the spec's
    # order -- run tables first, parquet last -- is what decides the code, and
    # the test pins it.
    _write(root, molecules=mols + [big],
           descriptor_mutate=lambda d: None)
    import pyarrow.parquet as pq
    _rewrite_parquet(root, lambda t: t.slice(0, 2))
    _descriptor_patch(lambda d: d.__setitem__("n_mol", 3))(root)


def seed_color_self_noncanonical(root):
    _write(root, color=True, with_color_self=True, off_diagonal=0.5)


def seed_definition_cr(root):
    _write_definition(root, TINY_FDEF.replace("\n", "\r\n", 1))


def seed_definition_lowercase_bomb(root):
    n = 18
    lines = ["atomtype A0 [#6]"] + [f"atomtype A{i} [{{A{i-1}}},{{A{i-1}}}]" for i in range(1, n + 1)]
    lines += [f"definefeature F [{{A{n}}}]", "  Family Hydrophobe", "  Weights 1.0", "EndFeature"]
    _write_definition(root, "\n".join(lines) + "\n" + ACID_BLOCK)


def seed_definition_negation_bomb(root):
    n = 18
    lines = ["AtomType A0 [#6]"]
    for i in range(1, n + 1):
        lines += [f"AtomType A{i} [{{A{i-1}}}]", f"AtomType !A{i} [{{A{i-1}}}]"]
    lines += [f"DefineFeature F [{{A{n}}}]", "  Family Hydrophobe", "  Weights 1.0", "EndFeature"]
    _write_definition(root, "\n".join(lines) + "\n" + ACID_BLOCK)


def seed_definition_continuation_star(root):
    text = ("DefineFeature Chain \\\n" + "~".join(["*"] * 18) + "\n  Family Hydrophobe\n"
            "  Weights " + ",".join(["1.0"] * 18) + "\nEndFeature\n" + ACID_BLOCK)
    _write_definition(root, text)


def seed_definition_lowercase_star(root):
    text = ("definefeature Chain " + "~".join(["*"] * 18) + "\n  Family Hydrophobe\n"
            "  Weights " + ",".join(["1.0"] * 18) + "\nEndFeature\n" + ACID_BLOCK)
    _write_definition(root, text)


def seed_definition_any_atom_chain(root):
    text = ("DefineFeature Chain " + "~".join(["A"] * 18) + "\n  Family Hydrophobe\n"
            "  Weights " + ",".join(["1.0"] * 18) + "\nEndFeature\n" + ACID_BLOCK)
    _write_definition(root, text)


def seed_definition_triple_bond_chain(root):
    """``#`` outside brackets is the triple bond, never a comment."""
    text = ("DefineFeature Chain C#" + "~".join(["*"] * 18) + "\n  Family Hydrophobe\n"
            "  Weights " + ",".join(["1.0"] * 19) + "\nEndFeature\n" + ACID_BLOCK)
    _write_definition(root, text)


def seed_definition_type_without_family(root):
    text = "DefineFeature F [#6]\n  Family Hydrophobe\n  Weights 1.0\nEndFeature\n"
    _write_definition(root, text)      # TYPES names NegIonizable too


def seed_definition_stray_line(root):
    _write_definition(root, TINY_FDEF + "Include evil.fdef\n")


def seed_definition_macro_depth(root):
    """Nested macro references: RDKit wraps each in $([...]), so nine levels
    of {Name} exceed the recursion-depth limit without any explicit $(."""
    lines = ["AtomType A0 [#6]"] + [f"AtomType A{i} [{{A{i-1}}};R]" for i in range(1, 10)]
    lines += ["DefineFeature F [{A9}]", "  Family Hydrophobe", "  Weights 1.0", "EndFeature"]
    _write_definition(root, "\n".join(lines) + "\n" + ACID_BLOCK)


def seed_definition_branch_depth(root):
    """Review 2: a branch ``(`` inside a bracket atom inside a recursion was
    not pushed but its ``)`` was popped, so ``[$(*(*)`` nested nine deep
    reported depth 1. Nine real levels; limit 28 is 8."""
    k = 9
    text = ("DefineFeature Nest " + "[$(*(*)" * k + "*" + ")]" * k + "\n  Family Hydrophobe\n"
            "  Weights 1.0\nEndFeature\n" + ACID_BLOCK)
    _write_definition(root, text)


def _smarts_color(definition):
    n = len(TYPES)
    r = np.zeros((n + 1, n + 1), np.float32)
    p = np.zeros((n + 1, n + 1), np.float32)
    for t in range(1, n + 1):
        r[t, t] = p[t, t] = 1.0
    return R.color_descriptor(generator="smarts_list", definition=definition, types=TYPES,
                              r=r, p=p, rdkit_version="2026.09.1")


def _write_smarts(root, definition, **hooks):
    shutil.rmtree(root, ignore_errors=True)
    return R.write_corpus(root, _base_molecules(True), color=_smarts_color(definition),
                          prep_version="0.1.0.dev0", prep_commit=None, block_rows=4096, **hooks)


def seed_smarts_list_too_many(root):
    _write_definition(root, generator="smarts_list", definition={"Hydrophobe": ["[#6]"] * 257, "NegIonizable": ["[O-]"]})


def seed_smarts_list_star_chain(root):
    _write_definition(root, generator="smarts_list", definition={"Hydrophobe": ["~".join(["*"] * 18)], "NegIonizable": ["[O-]"]})


def seed_smarts_list_nested(root):
    _write_definition(root, generator="smarts_list", definition={"Hydrophobe": ["[$(" * 9 + "C" + ")]" * 9], "NegIonizable": ["[O-]"]})


def seed_smarts_list_missing_family(root):
    _write_definition(root, generator="smarts_list", definition={"Hydrophobe": ["[#6]"]})


def seed_smarts_list_not_object(root):
    _write_smarts(root, {"Hydrophobe": ["[#6]"], "NegIonizable": ["[O-]"]},
                  descriptor_mutate=lambda d: d["color"].__setitem__("definition", "[#6]"))


def seed_group_is_fifo(root):
    """Generated: the group file is a FIFO; an open() without O_NONBLOCK hangs."""
    _write(root)
    p = _group0(root)
    os.unlink(p)
    os.mkfifo(p)


def seed_corpus_json_is_fifo(root):
    """Generated."""
    _write(root)
    p = os.path.join(root, "corpus.json")
    os.unlink(p)
    os.mkfifo(p)


def seed_group_symlink_escapes_root(root):
    """Generated: the group file is a symlink to a file outside the corpus."""
    _write(root)
    p = _group0(root)
    outside = os.path.join(os.path.dirname(os.path.abspath(root)), "outside_" + os.path.basename(root) + ".r3g")
    os.replace(p, outside)
    os.symlink(outside, p)


def seed_single_molecule_4097_rows(root):
    """Generated: one molecule with 4,097 conformers; the row cap
    cannot hold it, so the writer is bypassed: the block table claims
    4,097 rows for one molecule."""
    a = R.Molecule(mol_id=11, coords=np.zeros((1, 3, 1), np.int16),
                   vaa=np.array([1.5], np.float32), ext_id="one", smiles="C", mw=16.0)
    def gm(name, sections, fields, records):
        data = R.encode_block(np.zeros((4097, 3, 8), np.int16), np.ones(4097, np.int64))
        sections["coords"] = data
        sections["vaa"] = np.full(4097, 1.5, "<f4").tobytes()
        fields["n_conf"] = 4097
        records[:] = [{"first_row": 0, "rows": 4097, "offset": 0, "length": len(data),
                       "m0": 0, "m1": 1, "sha256": R.sha256_hex(data)}]
    _write(root, molecules=[a], group_mutate=gm,
           descriptor_mutate=lambda d: (d["groups"][0].__setitem__("n_conf", 4097), d.__setitem__("n_conf", 4097)))


def seed_empty_group(root):
    def hm(name, h):
        h["n_mol"] = 0
    _write(root, header_mutate=hm, descriptor_mutate=lambda d: d["groups"][0].__setitem__("n_mol", 0))


SEEDS = {
    # name: (builder, tier, code, threat-model row, generated)
    "magic_rsb3": (seed_magic_rsb3, "open", "prologue.magic", "2.3 containers: magic/version", False),
    "major_version_4": (seed_major_version_4, "open", "prologue.major_version", "2.3 unknown major version (prologue)", False),
    "descriptor_major_4": (seed_descriptor_major_4, "open", "prologue.major_version", "2.3 unknown major version (descriptor)", False),
    "reserved_prologue_nonzero": (seed_reserved_prologue_nonzero, "open", "prologue.reserved", "2.3 containers", False),
    "truncated_file": (seed_truncated_file, "open", "prologue.header_offset", "2.3 header extent (truncation)", False),
    "oversize_header": (seed_oversize_header, "open", "json.size", "2.3 JSON: uncapped header read", True),
    "json_duplicate_key": (seed_json_duplicate_key, "open", "json.duplicate_key", "2.3 JSON: duplicate keys", False),
    "json_nan": (seed_json_nan, "open", "json.nan", "2.3 JSON: NaN", False),
    "json_depth": (seed_json_depth, "open", "json.depth", "2.3 JSON: deep nesting", False),
    "group_path_entry": (seed_group_path_entry, "open", "schema.unknown_key", "2.3 paths: a path-like group entry", False),
    "group_index_string": (seed_group_index_string, "open", "schema.type", "2.3 paths: a non-integer group index", False),
    "encoding_unknown": (seed_encoding_unknown, "open", "schema.type", "3 P0: selectors are closed enumerations", False),
    "step_wrong": (seed_step_wrong, "open", "schema.value", "3 P0: closed schema", False),
    "section_unknown": (seed_section_unknown, "open", "schema.type", "3 P0: closed enumerations", False),
    "section_outside_file": (seed_section_outside_file, "open", "layout.sections", "2.2 section offset outside the file", False),
    "section_overlap": (seed_section_overlap, "open", "layout.sections", "2.2 overlapping sections", False),
    "section_misaligned": (seed_section_misaligned, "open", "layout.sections", "2.2 unaligned section", False),
    "section_length_mismatch": (seed_section_length_mismatch, "open", "layout.sections", "2.2 len != prod(shape)*size", False),
    "width_17": (seed_width_17, "scan", "limit.bit_width", "2.2 bit width > 16", False),
    "block_length_short": (seed_block_length_short, "scan", "limit.block_length", "2.2 more bits than the stream holds", False),
    "block_reserved_nonzero": (seed_block_reserved_nonzero, "scan", "layout.block_header", "2.2 codec params", False),
    "block_rows_4097": (seed_block_rows_4097, "open", "limit.block_rows", "2.2 block rows over the cap", True),
    "block_molecule_gap": (seed_block_molecule_gap, "open", "layout.blocks", "2.2 molecule ranges do not tile", False),
    "block_rows_mismatch": (seed_block_rows_mismatch, "scan", "layout.runs", "2.2 run sums wrong", False),
    "n_real_zero": (seed_n_real_zero, "scan", "limit.n_real", "2.2 n_real out of range", False),
    "n_real_exceeds_L_heavy": (seed_n_real_exceeds_L_heavy, "scan", "layout.widths", "2.2 n_real > L_heavy", False),
    "n_color_exceeds_L": (seed_n_color_exceeds_L, "scan", "layout.widths", "2.2 n_real + sum(color) > device L", False),
    "L_field_width_wrong": (seed_L_field_width_wrong, "scan", "layout.widths", "2.2 widths (bucket-wide)", False),
    "runs_unsorted": (seed_runs_unsorted, "scan", "layout.runs", "2.2 molecule runs (order)", False),
    "mol_id_2_63": (seed_mol_id_2_63, "scan", "limit.mol_id", "T293 scope 10: mol_id range", False),
    "feature_type_out_of_range": (seed_feature_type_out_of_range, "scan", "layout.features", "2.2 feature type not in [1, n_types]", False),
    "feature_atom_out_of_range": (seed_feature_atom_out_of_range, "scan", "layout.features", "2.2 atom index >= n_real", False),
    "feature_zero_atoms": (seed_feature_zero_atoms, "scan", "layout.features", "2.2 atoms per centroid: 0", False),
    "feature_too_many_atoms": (seed_feature_too_many_atoms, "scan", "limit.atoms_per_feature", "2.2 atoms per centroid: huge", True),
    "feature_prefix_reversed": (seed_feature_prefix_reversed, "scan", "layout.features", "section 8.1: prefix tuple must sort first", False),
    "features_unsorted": (seed_features_unsorted, "scan", "layout.features", "FORMAT.md 8.1 rule 4", False),
    "n_types_7": (seed_n_types_7, "open", "limit.n_types", "2.4 resource exhaustion: n_types", True),
    "matrix_negative_width": (seed_matrix_negative_width, "open", "value.matrix", "2.5 matrices: finite, >= 0", False),
    "vaa_negative": (seed_vaa_negative, "scan", "value.self_overlap", "3 scan: self-overlaps finite and > 0", False),
    "color_self_nan": (seed_color_self_nan, "scan", "value.self_overlap", "3 scan: self-overlaps finite", False),
    "fingerprint_mismatch": (seed_fingerprint_mismatch, "open", "integrity.fingerprint", "T281 design point 2: separated group file", False),
    "header_hash_mismatch": (seed_header_hash_mismatch, "open", "integrity.header_hash", "3 P1: header hashes", False),
    "block_hash_mismatch": (seed_block_hash_mismatch, "deep", "integrity.block_hash", "3 P1: block hashes", False),
    "section_hash_mismatch": (seed_section_hash_mismatch, "deep", "integrity.section_hash", "3 P1: section hashes", False),
    "ids_hash_mismatch": (seed_ids_hash_mismatch, "deep", "integrity.ids_hash", "T293 scope 10: ids_sha256", False),
    "parquet_schema": (seed_parquet_schema, "scan", "value.parquet", "2.3 Parquet: schema checked before data", False),
    "parquet_unsupported_codec": (seed_parquet_unsupported_codec, "scan", "value.parquet", "section 11: unsupported compression codec", False),
    "parquet_dictionary": (seed_parquet_dictionary, "scan", "value.parquet", "2.3 Parquet: PLAIN only, bytes on disk bound memory", False),
    "parquet_nullable": (seed_parquet_nullable, "scan", "value.parquet", "FORMAT.md 11: non-null columns", False),
    "parquet_row_count": (seed_parquet_row_count, "scan", "value.parquet", "3 scan: parquet row count equals n_mol", False),
    "parquet_set_mismatch": (seed_parquet_set_mismatch, "scan", "value.parquet", "3 scan: parquet mol_id set equals the run tables", False),
    "parquet_hash_mismatch": (seed_parquet_hash_mismatch, "scan", "integrity.molecules_hash", "FORMAT.md 10.7: molecules_sha256", False),
    "parquet_page_bool_list": (seed_parquet_page_bool_list, "scan", "value.parquet", "review 2: page header with a list of 2^62 bools (thrift skip loop)", False),
    "parquet_page_type_dictionary": (seed_parquet_page_type_dictionary, "scan", "value.parquet", "review 2: footer honest, page header says DICTIONARY_PAGE", False),
    "parquet_page_size_mismatch": (seed_parquet_page_size_mismatch, "scan", "value.parquet", "review 2: footer honest, page header sizes differ (compressed page)", False),
    "parquet_page_encoding_dictionary": (seed_parquet_page_encoding_dictionary, "scan", "value.parquet", "review 2: footer honest, page header encoding RLE_DICTIONARY", False),
    "parquet_footer_oversize": (seed_parquet_footer_oversize, "scan", "value.parquet", "review 2: footer length bounded before the footer is parsed", True),
    "groups_dir_symlink": (seed_groups_dir_symlink, "open", "io.not_regular_file", "review 2: groups/ is a symlink (dir-fd opens)", True),
    "hac_mismatch": (seed_hac_mismatch, "scan", "limit.hac", "FORMAT.md 11: hac == n_real", False),
    "feature_offsets_bad": (seed_feature_offsets_bad, "scan", "layout.features", "2.2 feature offsets not a valid CSR", False),
    "features_256": (seed_features_256, "scan", "limit.features_per_molecule", "2.2 per-type counts overflow u8", True),
    "n_conf_zero": (seed_n_conf_zero, "scan", "limit.n_conf_per_molecule", "2.2 molecule runs: n_conf = 0", False),
    "duplicate_mol_id_across_groups": (seed_duplicate_mol_id_across_groups, "scan", "layout.runs", "FORMAT.md 7 rule 2: mol_id unique across the corpus", False),
    "color_self_noncanonical": (seed_color_self_noncanonical, "open", "layout.sections", "FORMAT.md 18.2: color_self only with canonical matrices", False),
    "definition_cr": (seed_definition_cr, "scan", "value.definition", "FORMAT.md 10.3: no carriage return", False),
    "definition_lowercase_bomb": (seed_definition_lowercase_bomb, "scan", "limit.definition_depth", "review S1: lowercase keyword macro bomb", False),
    "definition_negation_bomb": (seed_definition_negation_bomb, "scan", "limit.definition_depth", "review S1: !Name negation bomb", False),
    "definition_any_atom_chain": (seed_definition_any_atom_chain, "scan", "limit.definition_atoms", "cold read 2 A20: A/a primitives count as atoms", False),
    "definition_triple_bond_chain": (seed_definition_triple_bond_chain, "scan", "limit.definition_atoms", "cold read 2 A8: # outside brackets is the triple bond, not a comment", False),
    "definition_type_without_family": (seed_definition_type_without_family, "scan", "value.definition", "FORMAT.md 10.2: every type is a Family of the definition", False),
    "definition_continuation_star": (seed_definition_continuation_star, "scan", "value.definition", "review S1: backslash-continued 18-star chain", False),
    "definition_lowercase_star": (seed_definition_lowercase_star, "scan", "limit.definition_atoms", "review S1: lowercase definefeature 18-star chain", False),
    "definition_stray_line": (seed_definition_stray_line, "scan", "value.definition", "review S1: closed grammar", False),
    "definition_macro_depth": (seed_definition_macro_depth, "scan", "limit.definition_depth", "review S1: RDKit's $([...]) wrapping counts toward depth", False),
    "smarts_list_too_many": (seed_smarts_list_too_many, "scan", "limit.definition_patterns", "FORMAT.md 15: smarts_list pattern count", False),
    "smarts_list_star_chain": (seed_smarts_list_star_chain, "scan", "limit.definition_atoms", "FORMAT.md 15: smarts_list top-level atoms", False),
    "smarts_list_nested": (seed_smarts_list_nested, "scan", "limit.definition_depth", "FORMAT.md 15: smarts_list recursion depth", False),
    "smarts_list_missing_family": (seed_smarts_list_missing_family, "scan", "value.definition", "FORMAT.md 15: every type has patterns", False),
    "smarts_list_not_object": (seed_smarts_list_not_object, "open", "schema.type", "FORMAT.md 10.2: smarts_list definition is an object", False),
    "group_is_fifo": (seed_group_is_fifo, "open", "io.not_regular_file", "review S2: a FIFO must not hang open()", True),
    "corpus_json_is_fifo": (seed_corpus_json_is_fifo, "open", "io.not_regular_file", "review S2: a FIFO must not hang open()", True),
    "group_symlink_escapes_root": (seed_group_symlink_escapes_root, "open", "io.not_regular_file", "2.3 paths: symlink out of the root", True),
    "single_molecule_4097_rows": (seed_single_molecule_4097_rows, "open", "limit.block_rows", "FORMAT.md 6 rule 4: single-molecule cap", True),
    "smiles_too_long": (seed_smiles_too_long, "scan", "limit.smiles", "2.3 strings: SMILES length", True),
    "smiles_cx": (seed_smiles_cx, "scan", "limit.smiles", "2.3 strings: CXSMILES", False),
    "ext_id_control": (seed_ext_id_control, "scan", "limit.ext_id", "2.3 strings: control characters", False),
    "macro_bomb": (seed_macro_bomb, "scan", "limit.definition_depth", "2.1 / appendix: AtomType macro bomb (RDKit's $([...]) wrapping reaches depth 9 at level 9, before the size limit)", False),
    "chained_star_18": (seed_chained_star_18, "scan", "limit.definition_atoms", "2.1 / appendix: 18 chained * atoms", False),
    "nested_smarts": (seed_nested_smarts, "scan", "limit.definition_depth", "2.1 / appendix: nested $( SMARTS", False),
    "definition_oversize": (seed_definition_oversize, "open", "json.string_length", "3 P0: definition text 64 KiB (caught by the JSON string limit)", True),
    "definition_oversize_prepass": (seed_definition_oversize_prepass, "scan", "limit.definition_size", "3 P0: the pre-pass's own size limit, exercised directly", True),
    "definition_branch_depth": (seed_definition_branch_depth, "scan", "limit.definition_depth", "review 2: branch inside a bracket inside a recursion hid the depth", False),
    "definition_undefined_macro": (seed_definition_undefined_macro, "scan", "value.definition", "2.1 fdef parser", False),
    "empty_group": (seed_empty_group, "open", "schema.value", "FORMAT.md 9.3: an empty group is invalid", False),
}


def build(name: str, root: str) -> None:
    """Materialize seed ``name`` into directory ``root``."""
    SEEDS[name][0](root)


def main() -> None:
    expected = {}
    for name, (fn, tier, code, row, generated) in SEEDS.items():
        root = os.path.join(HERE, name)
        if generated:
            shutil.rmtree(root, ignore_errors=True)
        else:
            fn(root)
        expected[name] = {"tier": tier, "code": code, "threat_model": row, "generated": generated}
        if name == "block_molecule_gap":
            expected[name]["native_code"] = "layout.runs"
        elif name in ("section_misaligned", "section_overlap"):
            expected[name]["native_code"] = "layout.blocks"
    with open(os.path.join(HERE, "EXPECTED.json"), "w", encoding="utf-8") as fh:
        json.dump(expected, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(SEEDS)} seeds ({sum(1 for v in SEEDS.values() if v[4])} generated at test time)")


if __name__ == "__main__":
    main()

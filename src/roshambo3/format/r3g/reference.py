"""Reference implementation of corpus format 3 (``doc/FORMAT.md``).

Pure Python plus NumPy: no compiled roshambo3 code, no RDKit. This module
is the **oracle**, not the production path -- clarity beats speed, and every
loop is written the way the specification states it. It implements:

* the BP256-16 block encoder and decoder and the block length formula
  (FORMAT.md section 5);
* the block, run and feature tables (sections 6-8) and the derivation of
  pseudo-atom positions and the kernel layout (section 8.3-8.5);
* canonical JSON, the fingerprint, ``definition_hash`` and ``ids_sha256``
  (section 10);
* the RDKit-free definition-text pre-pass (section 15), as a pure function
  that the definition sandbox reuses;
* a writer that assembles group files, ``corpus.json`` and
  ``molecules.parquet`` from in-memory molecules, so the conformance corpus
  and the adversarial seeds are produced by one code path;
* the ``open`` and ``scan`` validator tiers for a whole corpus, and the
  RDKit-free part of ``deep`` (section 16).

The ``deep`` items that need RDKit (SMILES parsing, feature re-perception)
belong to the production validator and are stubbed here (``deep_rdkit_checks``).

Section numbers in comments refer to ``doc/FORMAT.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants (FORMAT.md section 2)
# ---------------------------------------------------------------------------

MAGIC = b"R3G\0"
MAJOR = 3
STEP_LOG2 = -8
STEP = 2.0 ** STEP_LOG2
ENCODING = "bp256-16"
BUCKET_EDGES = (0, 12, 16, 20, 24, 28, 32, 38, 44, 50, 56, 64, 72, 256)
N_BUCKETS = len(BUCKET_EDGES) - 1
L_ALIGN = 8
PROLOGUE_SIZE = 16
FIRST_SECTION_OFFSET = 64
BLOCK_HEADER_SIZE = 32
BLOCK_RECORD_SIZE = 64
RUN_RECORD_SIZE = 16
SIGNATURE_FILE = "corpus.json.sig"
INT64_MAX = (1 << 63) - 1
#: features per chunk of the scan's feature-table checks (limit 34)
FEATURE_CHUNK = 1 << 16

#: dtype name -> (numpy dtype, itemsize). Closed enumeration (section 3.3).
DTYPES = {
    "u8": np.dtype("u1"), "u16": np.dtype("<u2"), "u32": np.dtype("<u4"),
    "u64": np.dtype("<u8"), "i16": np.dtype("<i2"), "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
}

#: Section order and dtype (section 3.3). ``color`` marks color-only sections.
SECTION_ORDER = (
    ("coords", "u8", False),
    ("blocks", "u8", False),
    ("runs", "u8", False),
    ("feature_offsets", "u32", True),
    ("feature_types", "u8", True),
    ("feature_atom_offsets", "u32", True),
    ("feature_atoms", "u8", True),
    ("vaa", "f32", False),
    ("color_self", "f64", True),
)

#: The limits table (section 14.1), by the name used in ``limit.<name>`` codes.
LIMITS = {
    "json_bytes": 16 * 1024 * 1024,
    "json_depth": 32,
    "json_string": 64 * 1024,
    "groups": 1 << 16,
    "blocks": 1 << 20,
    "n_conf": 1 << 24,
    "n_real": 255,
    "n_conf_per_molecule": 4096,
    "L_heavy": 256,
    "L": 512,
    "L_field_width": 256,
    "n_types": 6,
    "features_per_type": 255,
    "features_per_molecule": 255,
    "atoms_per_feature": 16,
    "block_rows": 4096,
    "block_rows_single_molecule": 4096,
    "bit_width": 16,
    "definition_size": 64 * 1024,
    "definition_lines": 4096,
    "definition_line_count": 256,         # AtomType and DefineFeature lines, each
    "definition_expansion": 8 * 1024,     # per pattern
    "definition_expansion_total": 128 * 1024,
    "definition_depth": 8,
    "definition_atoms": 16,
    "definition_patterns": 256,
    "definition_pattern_size": 1024,
    "definition_patterns_total": 64 * 1024,
    "n_feat_atoms": (1 << 32) - 1,
    "parquet_row_group": 1 << 14,       # rows per row group (formerly 2^16)
    "parquet_footer_base": 64 * 1024,   # footer bytes: base + per row group, before parsing
    "parquet_footer_per_row_group": 1024,
    "parquet_row_bytes": 4608,          # per-row cap: 4,371 B of column caps, rounded up to 4.5 KiB
    "parquet_overhead": 1 << 20,
    "smiles": 4096,
    "ext_id": 256,
    "mol_id": 1 << 63,
    "refuse_seconds": 20.0,                # limit 33 (relaxed)
    "scan_bytes_per_second": 1 << 20,       # limit 33 throughput floor, plus refuse_seconds per file
    "rss_bytes": 2 << 30,
}

CONVENTIONS = {
    "hydrogens": "RemoveAllHs",
    "features_perceived": "with_hydrogens",
    "atom_order": "canonical_smiles",
    "protonation": "as_given",
    "standardization": "none",
    "conformers": "as_given",
}
FRAME = {"center": "heavy_atom_centroid", "axes": "principal_axes",
         "handedness": "right"}
FORMAT_OBJECT = {"major": MAJOR, "encoding": ENCODING, "step_log2": STEP_LOG2}
GENERATORS = ("rdkit_fdef", "smarts_list")
POSITIONING = "centroid"

PARQUET_COLUMNS = (("mol_id", "uint64"), ("ext_id", "string"),
                   ("smiles", "string"), ("hac", "uint8"),
                   ("mw", "double"), ("rotors", "uint16"))
PARQUET_ENCODINGS = {"PLAIN", "RLE"}      # RLE carries the (all-defined) levels only
#: per-column byte caps per row (section 11 rule 3): the PLAIN encoding of
#: each column, strings with their 4-byte length prefix, u8/u16 stored as INT32
PARQUET_COLUMN_BYTES = {"mol_id": 8, "ext_id": 256 + 4, "smiles": 4096 + 4, "hac": 4, "mw": 8, "rotors": 4}
_SMILES_OK = re.compile(r"^[!-{}-~]+$")      # printable ASCII, no space, no '|' (0x7C)

_BIDI = {0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
         0x2066, 0x2067, 0x2068, 0x2069}


# ---------------------------------------------------------------------------
# Errors (section 16)
# ---------------------------------------------------------------------------

class CorpusValidationError(ValueError):
    """The one exception a validator may raise. ``code`` is from section 16.5."""

    def __init__(self, code: str, message: str, *, tier: str = "open",
                 path: Optional[str] = None):
        self.code = code
        self.tier = tier
        self.path = path
        where = f"{path}: " if path else ""
        super().__init__(f"{where}{message} [{code}]")


def _refuse(code: str, message: str, *, tier: str = "open",
            path: Optional[str] = None) -> CorpusValidationError:
    return CorpusValidationError(code, message, tier=tier, path=path)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def pad8(n: int) -> int:
    """``max(8, ceil(n / 8) * 8)`` (section 2)."""
    return max(L_ALIGN, -(-int(n) // L_ALIGN) * L_ALIGN)


def roundup64(x: int) -> int:
    return -(-int(x) // 64) * 64


def bucket_of(n_real: int) -> int:
    """Bucket index of a heavy-atom count (section 9.1)."""
    n = int(n_real)
    if not 1 <= n <= 255:
        raise ValueError(f"n_real {n} outside [1, 255]")
    for b in range(N_BUCKETS):
        if BUCKET_EDGES[b] <= n < BUCKET_EDGES[b + 1]:
            return b
    raise AssertionError("unreachable")


def group_file_name(bucket: int, group: int) -> str:
    """``groups/b{bucket:02d}_g{group:04d}.r3g`` (section 1)."""
    return f"groups/b{int(bucket):02d}_g{int(group):04d}.r3g"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def quantize(coords: np.ndarray) -> np.ndarray:
    """``rint(x * 256)`` as int16, round half to even (section 4.1)."""
    x = np.asarray(coords, dtype=np.float64)
    if x.size and not np.all(np.isfinite(x)):
        raise ValueError("coordinates must be finite")
    q = np.rint(x * (1.0 / STEP))
    if q.size and (q.max() > 32767 or q.min() < -32768):
        raise OverflowError("coordinate outside the int16 range at 2^-8 A")
    return q.astype(np.int16)


def dequantize(values: np.ndarray) -> np.ndarray:
    """``float32(v) * float32(2^-8)``, exact (section 4.1)."""
    return np.asarray(values, dtype=np.float32) * np.float32(STEP)


# ---------------------------------------------------------------------------
# Canonical and strict JSON (sections 10.5.1, 16.1)
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> bytes:
    """Section 10.5.1."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _scan_json_bytes(data: bytes) -> tuple[int, int]:
    """Max bracket depth and longest string (bytes), by one linear pass.

    Done on the raw bytes so a hostile depth never reaches the recursive
    parser; strings and escapes are tracked so brackets inside strings do
    not count.
    """
    depth = max_depth = containers = 0
    in_str = False
    esc = False
    str_len = max_str = 0
    for b in data:
        if in_str:
            if esc:
                esc = False
                str_len += 1
            elif b == 0x5C:  # backslash
                esc = True
                str_len += 1
            elif b == 0x22:  # quote
                in_str = False
                if str_len > max_str:
                    max_str = str_len
            else:
                str_len += 1
        elif b == 0x22:
            in_str = True
            str_len = 0
        elif b in (0x7B, 0x5B):  # { [
            containers += 1
            if containers > (1 << 18):
                raise _refuse("json.containers", "JSON container count exceeds 262144")
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif b in (0x7D, 0x5D):  # } ]
            depth -= 1
    return max_depth, max_str


def loads_strict(data: bytes, *, what: str, path: Optional[str] = None,
                 max_bytes: int = LIMITS["json_bytes"]) -> dict:
    """Strict JSON (section 16.1): bounded, no duplicate keys, no NaN,
    depth <= 32, strings <= 64 KiB, top level an object."""
    if len(data) > max_bytes:
        raise _refuse("json.size", f"{what} is {len(data)} bytes, limit {max_bytes}",
                      path=path)
    try:
        depth, max_str = _scan_json_bytes(data)
    except CorpusValidationError as e:
        raise _refuse(e.code, f"{what}: {e}", path=path) from None
    if depth > LIMITS["json_depth"]:
        raise _refuse("json.depth", f"{what} nests {depth} deep, limit "
                      f"{LIMITS['json_depth']}", path=path)
    if max_str > LIMITS["json_string"]:
        raise _refuse("json.string_length", f"{what} holds a string of {max_str} "
                      f"bytes, limit {LIMITS['json_string']}", path=path)

    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise _refuse("json.duplicate_key", f"{what}: duplicate key {k!r}",
                              path=path)
            d[k] = v
        return d

    def constant(name):
        raise _refuse("json.nan", f"{what}: {name} is not allowed", path=path)

    def parse_float(text):
        v = float(text)
        if not math.isfinite(v):
            raise _refuse("json.nan", f"{what}: float overflows to infinity", path=path)
        return v

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _refuse("json.syntax", f"{what} is not UTF-8: {e}", path=path) from None
    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=parse_float)
    except CorpusValidationError:
        raise
    except (ValueError, RecursionError) as e:
        raise _refuse("json.syntax", f"{what} is not valid JSON: {e}", path=path) from None
    if not isinstance(obj, dict):
        raise _refuse("json.syntax", f"{what} must be a JSON object", path=path)
    from .jsonstrict import check_unicode_scalars
    check_unicode_scalars(obj, what=what, path=path)
    return obj


def _require_keys(obj: dict, required: Iterable[str], optional: Iterable[str],
                  *, what: str, path: Optional[str], tier: str = "open") -> None:
    required = set(required)
    allowed = required | set(optional)
    for k in obj:
        if k not in allowed:
            raise _refuse("schema.unknown_key", f"{what}: unknown key {k!r}",
                          path=path, tier=tier)
    for k in required:
        if k not in obj:
            raise _refuse("schema.missing_key", f"{what}: missing key {k!r}",
                          path=path, tier=tier)


def _int(obj: dict, key: str, *, what: str, path: Optional[str],
         lo: int = 0, hi: Optional[int] = None, limit: Optional[str] = None) -> int:
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise _refuse("schema.type", f"{what}.{key} must be an integer", path=path)
    if v < lo:
        raise _refuse("schema.value", f"{what}.{key} = {v} is below {lo}", path=path)
    if v > INT64_MAX:
        # section 17 item 11: a native reader would wrap; every reader refuses
        raise _refuse("schema.value", f"{what}.{key} = {v} does not fit a signed 64-bit integer",
                      path=path)
    if hi is not None and v > hi:
        raise _refuse(f"limit.{limit}" if limit else "schema.value",
                      f"{what}.{key} = {v} exceeds {hi}", path=path)
    return v


def _hex64(obj: dict, key: str, *, what: str, path: Optional[str]) -> str:
    v = obj[key]
    if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise _refuse("schema.type", f"{what}.{key} must be 64 lowercase hex characters",
                      path=path)
    return v


# ---------------------------------------------------------------------------
# BP256-16 (section 5)
# ---------------------------------------------------------------------------

def bit_length(x: int) -> int:
    return int(x).bit_length()


def block_length(total: int, w_x: int, w_y: int, w_z: int) -> int:
    """``32 + 32 * G * (w_x + w_y + w_z)`` with ``G = ceil(total / 256)``."""
    groups = -(-int(total) // 256)
    return BLOCK_HEADER_SIZE + 32 * groups * (int(w_x) + int(w_y) + int(w_z))


def pack_axis(values: np.ndarray) -> tuple[int, int, bytes]:
    """Encode one axis stream: returns ``(min, w, stream_bytes)`` (section 5.3, 5.5)."""
    v = np.asarray(values, dtype=np.int64).reshape(-1)
    if v.size == 0:
        raise ValueError("an axis stream has at least one value")
    mn = int(v.min())
    w = bit_length(int(v.max()) - mn)
    total = int(v.size)
    groups = -(-total // 256)
    if w == 0:
        return mn, 0, b""
    u = (v - mn).astype(np.uint32)
    words = np.zeros((groups, w, 16), dtype=np.uint16)  # [group][word i][lane l]
    for j in range(total):
        g, r = divmod(j, 256)
        k, l = divmod(r, 16)
        val = int(u[j])
        for bit in range(w):
            if (val >> bit) & 1:
                b = k * w + bit
                words[g, b // 16, l] |= np.uint16(1 << (b % 16))
    return mn, w, words.astype("<u2").tobytes()


def unpack_axis(stream: bytes, w: int, mn: int, total: int) -> np.ndarray:
    """Decode one axis stream to ``int16[total]`` (section 5.3 pseudo-code).

    The bit rule is applied to every group and lane at once
    (:func:`unpack_axis_raw_vectorized`); :func:`unpack_axis_raw` is the
    position-by-position loop of section 5.3 it is tested against. Both are
    NumPy written from the spec, independent of the native codec. (Running
    the scalar loop here costs ~0.27 s per axis of a full block, which
    would make ``deep``'s reference cross-decode most of its time.)"""
    groups = -(-int(total) // 256)
    expected = groups * int(w) * 32
    if len(stream) != expected:
        raise ValueError(f"axis stream is {len(stream)} bytes, expected {expected}")
    out = unpack_axis_raw_vectorized(stream, int(w), int(total))
    # Section 5.6: modulo 2^16 into int16, bit-identical with a SIMD add.
    vals = ((out[:total] + mn) & 0xFFFF).astype(np.uint16).view(np.int16)
    return vals.copy()


def unpack_axis_raw(stream: bytes, w: int, total: int) -> np.ndarray:
    """The unsigned ``u`` values of every position of every group (tail
    included), by the scalar loop of section 5.3. The oracle for
    ``unpack_axis_raw_vectorized``."""
    groups = -(-int(total) // 256)
    out = np.zeros(groups * 256, dtype=np.int64)
    if w:
        words = np.frombuffer(stream, dtype="<u2").reshape(groups, w, 16)
        for g in range(groups):
            for k in range(16):
                for l in range(16):
                    u = 0
                    for bit in range(w):
                        b = k * w + bit
                        u |= ((int(words[g, b // 16, l]) >> (b % 16)) & 1) << bit
                    out[g * 256 + 16 * k + l] = u
    return out


def unpack_axis_raw_vectorized(stream: bytes, w: int, total: int) -> np.ndarray:
    """``unpack_axis_raw`` with the same bit rule applied to every group and
    lane at once: one NumPy operation per lane bit (``16 * w <= 256`` of
    them), so the ``deep`` tier's tail check runs at array speed on a
    100 MB block (limit 33). Bit-identical to the scalar loop (tested)."""
    groups = -(-int(total) // 256)
    if w == 0:
        return np.zeros(groups * 256, dtype=np.int64)
    words = np.frombuffer(stream, dtype="<u2").reshape(groups, w, 16).astype(np.uint32)
    u = np.zeros((groups, 16, 16), dtype=np.uint32)          # [group][position k][lane l]
    for b in range(16 * w):
        k, bit = divmod(b, w)
        u[:, k, :] |= ((words[:, b // 16, :] >> (b % 16)) & 1) << bit
    return u.reshape(groups * 256).astype(np.int64)


def encode_block(coords: np.ndarray, n_real: np.ndarray) -> bytes:
    """Encode a block from padded ``int16[rows, 3, L]`` and per-row ``n_real``."""
    xyz = np.asarray(coords, dtype=np.int16)
    nr = np.asarray(n_real, dtype=np.int64).reshape(-1)
    if xyz.ndim != 3 or xyz.shape[1] != 3 or xyz.shape[0] != nr.size or nr.size == 0:
        raise ValueError("encode_block takes int16[rows, 3, L] and n_real[rows], rows >= 1")
    if nr.min() < 1 or nr.max() > xyz.shape[2]:
        raise ValueError("every n_real must be in [1, L]")
    header = bytearray(BLOCK_HEADER_SIZE)
    streams = []
    for a in range(3):
        values = np.concatenate([xyz[j, a, :nr[j]].astype(np.int64) for j in range(nr.size)])
        mn, w, stream = pack_axis(values)
        struct.pack_into("<h", header, 2 * a, mn)
        header[6 + a] = w
        streams.append(stream)
    return bytes(header) + b"".join(streams)


@dataclass
class BlockHeader:
    mins: tuple[int, int, int]
    widths: tuple[int, int, int]


def parse_block_header(data: bytes) -> BlockHeader:
    if len(data) < BLOCK_HEADER_SIZE:
        raise ValueError("block shorter than its 32-byte header")
    mins = struct.unpack_from("<hhh", data, 0)
    widths = (data[6], data[7], data[8])
    return BlockHeader(mins, widths)


def decode_block(data: bytes, n_real: np.ndarray, L: Optional[int] = None,
                 *, check_reserved: bool = True) -> np.ndarray:
    """Decode a block to padded ``int16[rows, 3, L]`` (zeros beyond ``n_real``).

    Every check of FORMAT.md section 17 that the reference can express is
    here, as a ``ValueError`` naming the failed check.
    """
    nr = np.asarray(n_real, dtype=np.int64).reshape(-1)
    rows = int(nr.size)
    if rows == 0:
        raise ValueError("decode: a block has at least one row")
    if nr.min() < 1 or nr.max() > LIMITS["n_real"]:
        raise ValueError("decode: n_real outside [1, 255]")
    if L is None:
        L = pad8(int(nr.max()))
    if int(nr.max()) > L:
        raise ValueError("decode: a row exceeds the output width")
    hdr = parse_block_header(data)
    for w in hdr.widths:
        if w > LIMITS["bit_width"]:
            raise ValueError(f"decode: bit width {w} > 16")
    if check_reserved and any(data[9:BLOCK_HEADER_SIZE]):
        raise ValueError("decode: reserved block-header bytes are not zero")
    total = int(nr.sum())
    need = block_length(total, *hdr.widths)
    if len(data) != need:
        raise ValueError(f"decode: block is {len(data)} bytes, the header implies {need}")
    out = np.zeros((rows, 3, L), dtype=np.int16)
    groups = -(-total // 256)
    pos = BLOCK_HEADER_SIZE
    # row j's real atoms are the next nr[j] values of each axis stream: in C
    # order, the mask's True cells are exactly that sequence
    real = np.arange(L)[None, :] < nr[:, None]
    for a in range(3):
        size = groups * hdr.widths[a] * 32
        vals = unpack_axis(data[pos:pos + size], hdr.widths[a], hdr.mins[a], total)
        pos += size
        plane = out[:, a, :]
        plane[real] = vals
    return out


# ---------------------------------------------------------------------------
# Tables (sections 6-8)
# ---------------------------------------------------------------------------

BLOCK_DTYPE = np.dtype([("first_row", "<u4"), ("rows", "<u4"), ("offset", "<u8"),
                        ("length", "<u8"), ("m0", "<u4"), ("m1", "<u4"),
                        ("sha256", "u1", 32)])
RUN_DTYPE = np.dtype([("mol_id", "<u8"), ("n_conf", "<u2"), ("n_real", "u1"),
                      ("reserved", "u1", 5)])
assert BLOCK_DTYPE.itemsize == BLOCK_RECORD_SIZE and RUN_DTYPE.itemsize == RUN_RECORD_SIZE


def pack_runs(mol_id, n_conf, n_real) -> np.ndarray:
    runs = np.zeros(len(mol_id), dtype=RUN_DTYPE)
    runs["mol_id"] = np.asarray(mol_id, dtype=np.uint64)
    runs["n_conf"] = np.asarray(n_conf, dtype=np.uint16)
    runs["n_real"] = np.asarray(n_real, dtype=np.uint8)
    return runs


def pack_blocks(records: list[dict]) -> np.ndarray:
    blocks = np.zeros(len(records), dtype=BLOCK_DTYPE)
    for i, r in enumerate(records):
        for k in ("first_row", "rows", "offset", "length", "m0", "m1"):
            blocks[i][k] = r[k]
        blocks[i]["sha256"] = np.frombuffer(bytes.fromhex(r["sha256"]), dtype=np.uint8)
    return blocks


def canonical_feature_order(features: list[tuple[int, tuple[int, ...]]]
                            ) -> list[tuple[int, tuple[int, ...]]]:
    """Section 8.1 rule 3-4: atoms ascending within a feature, features by
    ``(type, atom tuple)``; duplicates kept."""
    out = [(int(t), tuple(sorted(int(a) for a in atoms))) for t, atoms in features]
    return sorted(out)


def pack_features(per_molecule: list[list[tuple[int, tuple[int, ...]]]]) -> dict:
    """CSR arrays from per-molecule ``[(type, atoms), ...]`` (already canonical)."""
    offsets = [0]
    types: list[int] = []
    atom_offsets = [0]
    atoms: list[int] = []
    for feats in per_molecule:
        for t, ats in feats:
            types.append(t)
            atoms.extend(ats)
            atom_offsets.append(len(atoms))
        offsets.append(len(types))
    return {
        "feature_offsets": np.asarray(offsets, dtype="<u4"),
        "feature_types": np.asarray(types, dtype="u1"),
        "feature_atom_offsets": np.asarray(atom_offsets, dtype="<u4"),
        "feature_atoms": np.asarray(atoms, dtype="u1"),
    }


def molecule_features(tables: dict, m: int) -> list[tuple[int, tuple[int, ...]]]:
    fo = tables["feature_offsets"]
    out = []
    for f in range(int(fo[m]), int(fo[m + 1])):
        a0, a1 = int(tables["feature_atom_offsets"][f]), int(tables["feature_atom_offsets"][f + 1])
        out.append((int(tables["feature_types"][f]),
                    tuple(int(a) for a in tables["feature_atoms"][a0:a1])))
    return out


def integer_mean_half_away(s: int, k: int) -> int:
    """``sign(s) * ((2|s| + k) // (2k))`` (section 8.3)."""
    s, k = int(s), int(k)
    if s == 0:
        return 0
    q = (2 * abs(s) + k) // (2 * k)
    return q if s > 0 else -q


def derive_positions(xyz: np.ndarray, features: list[tuple[int, tuple[int, ...]]]
                     ) -> np.ndarray:
    """Pseudo-atom positions ``int16[n_feat, 3]`` of one row from its decoded
    heavy atoms ``xyz`` (``[3, >= n_real]`` int16) and its molecule's features."""
    out = np.zeros((len(features), 3), dtype=np.int16)
    for i, (_, atoms) in enumerate(features):
        for a in range(3):
            s = sum(int(xyz[a, idx]) for idx in atoms)
            out[i, a] = integer_mean_half_away(s, len(atoms))
    return out


def derived_counts(n_real: int, features: list[tuple[int, tuple[int, ...]]],
                   n_types: int) -> np.ndarray:
    """The per-row count vector ``u8[n_types + 1]`` (section 8.5)."""
    counts = np.zeros(n_types + 1, dtype=np.int64)
    counts[0] = n_real
    for t, _ in features:
        counts[t] += 1
    if counts.max() > 255:
        raise ValueError("a count exceeds 255")
    return counts.astype(np.uint8)


def kernel_layout(xyz: np.ndarray, n_real: int,
                  features: list[tuple[int, tuple[int, ...]]], L: int) -> np.ndarray:
    """One row's merged record ``int16[3, L]`` (section 8.4)."""
    out = np.zeros((3, L), dtype=np.int16)
    out[:, :n_real] = xyz[:, :n_real]
    pos = derive_positions(xyz, features)
    if n_real + len(features) > L:
        raise ValueError("row exceeds the device width")
    out[:, n_real:n_real + len(features)] = pos.T
    return out


# ---------------------------------------------------------------------------
# Descriptor: fingerprint, definition hash, ids hash (section 10)
# ---------------------------------------------------------------------------

def matrix_bytes(values) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype="<f4")).tobytes()


def fingerprint(descriptor: dict) -> str:
    """Section 10.5."""
    f = {"format": descriptor["format"], "frame": descriptor["frame"],
         "prep_conventions": descriptor["prep"]["conventions"]}
    color = descriptor.get("color")
    if color is not None:
        c = {k: v for k, v in color.items() if k != "rdkit_version"}
        if c.get("generator") == "rdkit_fdef":
            # Section 10.3 normalizes before both storage and fingerprinting.
            # Keep the input descriptor untouched for callers that reuse it.
            c["definition"] = normalize_definition(c["definition"])
        for key in ("r", "p"):
            m = color[key]
            c[key] = {"dtype": "f32", "shape": list(m["shape"]),
                      "sha256": sha256_hex(matrix_bytes(m["values"]))}
        f["color"] = c
    return sha256_hex(canonical_json(f))


def normalize_definition(text: str) -> str:
    """Section 10.3: CRLF and CR to LF, nothing else."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def definition_hash(definition) -> str:
    """Section 10.3: SHA-256 of the LF-normalized fdef text, or of the
    canonical JSON of a ``smarts_list`` object."""
    if isinstance(definition, str):
        return sha256_hex(normalize_definition(definition).encode("utf-8"))
    return sha256_hex(canonical_json(definition))


def ids_sha256(mol_ids) -> str:
    """Section 10.6: sorted unique ``mol_id`` as little-endian int64."""
    ids = np.unique(np.asarray(mol_ids, dtype=np.int64))
    return sha256_hex(ids.astype("<i8").tobytes())


#: The smallest nonzero interaction width any corpus or query may carry. A
#: width is exactly 0 (non-interacting: every backend skips the pair) or at
#: least this: a tiny positive width overflows ``r^-1.5`` and yields NaN in
#: CUDA (below about 3e-39) and Metal (below about 1e-30). Compared in float32.
MIN_INTERACTION_WIDTH = 1e-6


def interaction_matrix_problems(r, p) -> list[str]:
    """Why ``(r, p)`` cannot be carried by any corpus or query, or ``[]``.

    The one statement of the validity rule, every entry,
    row and column 0 included, as float32: square matrices of one shape;
    every value finite; every width ``r`` exactly 0 or ``>= 1e-6``; every
    height ``p`` ``>= 0``."""
    r = np.asarray(r, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape != p.shape:
        return [f"the interaction matrices must be square and the same shape; got "
                f"r{r.shape} and p{p.shape}"]
    out = []
    with np.errstate(over="ignore", invalid="ignore"):
        r32, p32 = r.astype(np.float32), p.astype(np.float32)
    bad = np.argwhere(~np.isfinite(r32))
    if bad.size:
        out.append(f"non-finite widths at r{bad[:4].tolist()}")
    bad = np.argwhere(~np.isfinite(p32))
    if bad.size:
        out.append(f"non-finite heights at p{bad[:4].tolist()}")
    fin = np.isfinite(r32)
    bad = np.argwhere(fin & (r32 != 0) & ~(r32 >= np.float32(MIN_INTERACTION_WIDTH)))
    if bad.size:
        out.append(f"widths must be exactly 0 or >= {MIN_INTERACTION_WIDTH:g} (a tiny "
                   f"positive width gives NaN on device): r{bad[:4].tolist()}")
    bad = np.argwhere(np.isfinite(p32) & (p32 < 0))
    if bad.size:
        out.append(f"heights must be >= 0: p{bad[:4].tolist()}")
    return out


def canonical_matrix_reasons(r, p) -> list[str]:
    """Why ``(r, p)`` is not canonical, or ``[]`` -- the one definition
    (FORMAT.md sections 10.2.1 and 18.2). Indices
    ``1..n`` are the color types; row and column 0 are ignored. Canonical iff
    the matrices are valid (``interaction_matrix_problems``), and, as exact
    float32 values, ``p[t][t] == 1`` and ``p[t][t'] == 0`` for ``t != t'``
    and ``r[t][t] == 1``, for ``t, t'`` in ``1..n``. Off-diagonal widths are
    ignored: with a zero height their term is exactly zero."""
    problems = interaction_matrix_problems(r, p)
    if problems:
        return problems
    r32 = np.asarray(r, dtype=np.float64).astype(np.float32)[1:, 1:]
    p32 = np.asarray(p, dtype=np.float64).astype(np.float32)[1:, 1:]
    n = r32.shape[0]
    off = ~np.eye(n, dtype=bool)
    out = []
    if not np.all(np.diag(r32) == np.float32(1)):
        out.append(f"diagonal interaction widths are not all exactly 1: {np.diag(r32).tolist()}")
    if not np.all(np.diag(p32) == np.float32(1)):
        out.append(f"diagonal interaction heights are not all exactly 1: {np.diag(p32).tolist()}")
    if np.any(p32[off] != 0):
        idx = np.argwhere(off & (p32 != 0))
        out.append(f"{len(idx)} off-diagonal feature pairs interact (p != 0), e.g. "
                   f"{[tuple(int(v) + 1 for v in i) for i in idx[:4]]}")
    return out


def canonical_matrices(r, p, n_types: int) -> bool:
    """Whether ``(r, p)``, of shape ``[n_types + 1]^2``, are canonical
    (``canonical_matrix_reasons``)."""
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (n_types + 1, n_types + 1):
        return False
    return not canonical_matrix_reasons(r, p)


def color_descriptor(*, generator: str, definition, types: list[str],
                     r, p, rdkit_version: str) -> dict:
    n = len(types)
    if generator == "rdkit_fdef":
        definition = normalize_definition(definition)
    return {
        "generator": generator,
        "definition": definition,
        "types": list(types),
        "positioning": POSITIONING,
        "r": {"dtype": "f32", "shape": [n + 1, n + 1],
              "values": np.asarray(r, dtype=np.float32).astype(np.float64).tolist()},
        "p": {"dtype": "f32", "shape": [n + 1, n + 1],
              "values": np.asarray(p, dtype=np.float32).astype(np.float64).tolist()},
        "rdkit_version": rdkit_version,
    }


# ---------------------------------------------------------------------------
# Definition-text pre-pass (section 15)
# ---------------------------------------------------------------------------

_ORGANIC = ("Cl", "Br", "se", "as", "B", "C", "N", "O", "P", "S", "F", "I",
            "b", "c", "n", "o", "p", "s", "*", "A", "a")


def smarts_structure(pattern: str) -> tuple[int, int]:
    """``(max $( depth, top-level atom count)`` of a SMARTS string (section 15
    rule 5).

    Depth is tracked with one parenthesis stack: **every** ``(`` is pushed
    with a flag saying whether it opened a recursion (the ``(`` of ``$(``) or
    not (a branch, or the ``%(`` of an extended ring closure), **every** ``)``
    pops one entry, and the depth is the number of recursion entries on the
    stack. A ``)`` with nothing to pop is ignored (RDKit rejects the pattern
    anyway). Bracket atoms ``[...]`` -- which may contain recursions with
    nested brackets -- organic-subset and aromatic symbols and ``*``, ``A``,
    ``a`` count as atoms when they occur outside every recursion and every
    bracket. Bond and ring-closure characters are skipped. Linear in the
    pattern; the stack is at most ``len(pattern)`` entries.
    """
    stack: list[bool] = []   # True: this "(" opened a $( recursion
    depth = 0                # number of True entries on the stack
    max_depth = 0
    atoms = 0
    bracket = 0              # nesting of [ ] (nested only inside recursions)
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "$" and i + 1 < n and pattern[i + 1] == "(":
            stack.append(True)
            depth += 1
            max_depth = max(max_depth, depth)
            i += 2
            continue
        if ch == "(":
            stack.append(False)
            i += 1
            continue
        if ch == ")":
            if stack and stack.pop():
                depth -= 1
            i += 1
            continue
        if ch == "[":
            if depth == 0 and bracket == 0:
                atoms += 1
            bracket += 1
            i += 1
            continue
        if ch == "]":
            bracket = max(0, bracket - 1)
            i += 1
            continue
        if depth == 0 and bracket == 0:
            matched = False
            for sym in _ORGANIC:
                if pattern.startswith(sym, i):
                    atoms += 1
                    i += len(sym)
                    matched = True
                    break
            if matched:
                continue
        i += 1
    return max_depth, atoms


def _check_pattern(pattern: str, where: str) -> None:
    if len(pattern.encode("utf-8")) > LIMITS["definition_expansion"]:
        raise _refuse("limit.definition_expansion",
                      f"{where}: expanded pattern is {len(pattern)} bytes, limit "
                      f"{LIMITS['definition_expansion']}", tier="scan")
    depth, atoms = smarts_structure(pattern)
    if depth > LIMITS["definition_depth"]:
        raise _refuse("limit.definition_depth",
                      f"{where}: $( recursion depth {depth}, limit "
                      f"{LIMITS['definition_depth']}", tier="scan")
    if atoms > LIMITS["definition_atoms"]:
        raise _refuse("limit.definition_atoms",
                      f"{where}: {atoms} top-level atoms, limit "
                      f"{LIMITS['definition_atoms']}", tier="scan")


_FDEF_KEYWORDS = ("atomtype", "definefeature", "family", "weights", "endfeature")




def expand_macro(positives: list[str], negatives: list[str]) -> str:
    """How RDKit's fdef parser expands ``{Name}`` (section 15 rule 4), measured
    on 2026.09.1: ``$([!$(n_k);...;!$(n_1);p_1',$(p_2),...,$(p_j)])`` where
    ``p_1'`` is the first positive definition with one outer ``[...]`` removed
    when it has one, every later positive and every negation (newest first)
    is wrapped in its own ``$(...)``; a lone unbracketed positive gives
    ``$(p_1)``."""
    if not negatives and len(positives) == 1 and not (
            positives[0].startswith("[") and positives[0].endswith("]")):
        return "$(" + positives[0] + ")"
    parts = ["!$(" + n + ")" for n in reversed(negatives)]
    pos_part = ""
    if positives:
        first = positives[0]
        if first.startswith("[") and first.endswith("]"):
            first = first[1:-1]
        pos_part = first + "".join(",$(" + d + ")" for d in positives[1:])
    body = ";".join(parts + ([pos_part] if pos_part else []))
    return "$([" + body + "])"


def check_definition_text(definition, generator: str,
                          types: Optional[Iterable[str]] = None) -> dict:
    """The RDKit-free pre-pass (section 15). Returns summary counts on
    success; raises ``CorpusValidationError`` naming the limit or the rule.
    ``types`` (the corpus's ``color.types``) must each be a ``Family`` of a
    ``DefineFeature`` block (``rdkit_fdef``) or a key (``smarts_list``).

    Pure: no RDKit, no file access, bounded work on any input. The grammar
    it accepts is closed (section 15): keywords are case-insensitive whole
    tokens as in RDKit, the pattern is the third whitespace-separated token
    (RDKit ignores anything after it), a backslash continuation is refused,
    ``!Name`` definitions are stored with ``Name``, and ``{Name}`` expands as
    RDKit expands it (``expand_macro``), so sizes and recursion depth are
    counted the way RDKit will see them.
    """
    if generator not in GENERATORS:
        raise _refuse("schema.type", f"unknown generator {generator!r}", tier="scan")
    if generator == "smarts_list":
        return _check_smarts_list(definition, types)
    text = definition
    if not isinstance(text, str):
        raise _refuse("value.definition", "an rdkit_fdef definition is not a string", tier="scan")
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError:
        raise _refuse("value.definition", "definition is not encodable as UTF-8 (lone surrogate)",
                      tier="scan") from None
    if "\r" in text:
        raise _refuse("value.definition", "definition contains a carriage return", tier="scan")
    if len(raw) > LIMITS["definition_size"]:
        raise _refuse("limit.definition_size", f"definition is {len(raw)} bytes, "
                      f"limit {LIMITS['definition_size']}", tier="scan")
    lines = text.split("\n")
    if len(lines) > LIMITS["definition_lines"]:
        raise _refuse("limit.definition_lines", f"definition has {len(lines)} lines, "
                      f"limit {LIMITS['definition_lines']}", tier="scan")
    for lineno, line in enumerate(lines, 1):
        if line.rstrip().endswith("\\"):
            raise _refuse("value.definition", f"line {lineno}: backslash continuation lines "
                          f"are not accepted", tier="scan")
    positives: dict[str, list[str]] = {}
    negatives: dict[str, list[str]] = {}
    families: set = set()
    n_atomtype = n_define = 0
    total_expanded = 0
    longest = 0
    in_block = False

    def expand(pattern: str, where: str) -> str:
        nonlocal total_expanded, longest
        out: list[str] = []
        size = 0
        i = 0
        while i < len(pattern):
            if pattern[i] == "{":
                j = pattern.find("}", i + 1)
                if j < 0:
                    raise _refuse("value.definition", f"{where}: unterminated '{{'", tier="scan")
                name = pattern[i + 1:j]
                if name not in positives and name not in negatives:
                    raise _refuse("value.definition",
                                  f"{where}: AtomType {name!r} is not defined before use",
                                  tier="scan")
                piece = expand_macro(positives.get(name, []), negatives.get(name, []))
                i = j + 1
            else:
                piece = pattern[i]
                i += 1
            out.append(piece)
            size += len(piece.encode("utf-8"))
            if size > LIMITS["definition_expansion"]:      # bounded as it grows
                raise _refuse("limit.definition_expansion",
                              f"{where}: expansion exceeds {LIMITS['definition_expansion']} bytes",
                              tier="scan")
        expanded = "".join(out)
        total_expanded += size
        longest = max(longest, size)
        if total_expanded > LIMITS["definition_expansion_total"]:
            raise _refuse("limit.definition_expansion",
                          f"{where}: total expansion exceeds "
                          f"{LIMITS['definition_expansion_total']} bytes", tier="scan")
        return expanded

    for lineno, line in enumerate(lines, 1):
        where = f"line {lineno}"
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue                                       # blank or comment, anywhere
        kw = parts[0].lower()
        if kw not in _FDEF_KEYWORDS:
            raise _refuse("value.definition", f"{where}: {parts[0]!r} is not an fdef keyword",
                          tier="scan")
        if kw == "atomtype":
            if in_block:
                raise _refuse("value.definition", f"{where}: AtomType inside a DefineFeature block",
                              tier="scan")
            n_atomtype += 1
            if n_atomtype > LIMITS["definition_line_count"]:
                raise _refuse("limit.definition_line_count",
                              f"more than {LIMITS['definition_line_count']} AtomType lines", tier="scan")
            if len(parts) < 3:
                raise _refuse("value.definition", f"{where}: AtomType needs a name and a pattern",
                              tier="scan")
            name = parts[1]
            negated = name.startswith("!")
            name = name[1:] if negated else name
            if not name:
                raise _refuse("value.definition", f"{where}: empty AtomType name", tier="scan")
            expanded = expand(parts[2], where)
            _check_pattern(expanded, where)
            (negatives if negated else positives).setdefault(name, []).append(expanded)
        elif kw == "definefeature":
            if in_block:
                raise _refuse("value.definition", f"{where}: DefineFeature without EndFeature",
                              tier="scan")
            n_define += 1
            if n_define > LIMITS["definition_line_count"]:
                raise _refuse("limit.definition_line_count",
                              f"more than {LIMITS['definition_line_count']} DefineFeature lines",
                              tier="scan")
            if len(parts) < 3:
                raise _refuse("value.definition", f"{where}: DefineFeature needs a name and a pattern",
                              tier="scan")
            expanded = expand(parts[2], where)
            _check_pattern(expanded, where)
            in_block = True
        elif kw == "family":
            if not in_block:
                raise _refuse("value.definition", f"{where}: Family outside a DefineFeature block",
                              tier="scan")
            if len(parts) >= 2:
                families.add(parts[1])
        elif kw == "weights":
            if not in_block:
                raise _refuse("value.definition", f"{where}: Weights outside a DefineFeature block",
                              tier="scan")
        else:  # endfeature
            if not in_block:
                raise _refuse("value.definition", f"{where}: EndFeature without DefineFeature",
                              tier="scan")
            in_block = False
    if in_block:
        raise _refuse("value.definition", "the last DefineFeature block is not closed", tier="scan")
    for t in (types or ()):
        if t not in families:
            raise _refuse("value.definition", f"color.types names {t!r}, which no DefineFeature "
                          f"block of the definition declares as its Family", tier="scan")
    return {"atomtype_lines": n_atomtype, "definefeature_lines": n_define,
            "expanded_total": total_expanded, "expanded_longest": longest,
            "families": sorted(families)}


def _check_smarts_list(obj, types: Optional[Iterable[str]]) -> dict:
    if not isinstance(obj, dict):
        raise _refuse("value.definition", "a smarts_list definition is a JSON object "
                      "{family: [pattern, ...]}", tier="scan")
    if len(canonical_json(obj)) > LIMITS["definition_patterns_total"]:
        raise _refuse("limit.definition_patterns", f"smarts_list exceeds "
                      f"{LIMITS['definition_patterns_total']} bytes", tier="scan")
    n = 0
    for fam, pats in obj.items():
        if not isinstance(fam, str) or not fam or not isinstance(pats, list) or not pats:
            raise _refuse("value.definition", "smarts_list must map non-empty family names to "
                          "non-empty lists", tier="scan")
        for pat in pats:
            if not isinstance(pat, str):
                raise _refuse("value.definition", f"family {fam!r}: pattern is not a string",
                              tier="scan")
            n += 1
            if n > LIMITS["definition_patterns"]:
                raise _refuse("limit.definition_patterns",
                              f"more than {LIMITS['definition_patterns']} patterns", tier="scan")
            if len(pat.encode("utf-8")) > LIMITS["definition_pattern_size"]:
                raise _refuse("limit.definition_patterns",
                              f"family {fam!r}: a pattern exceeds "
                              f"{LIMITS['definition_pattern_size']} bytes", tier="scan")
            _check_pattern(pat, f"family {fam!r}")      # `{` is literal here: no macros
    for t in (types or ()):
        if t not in obj:
            raise _refuse("value.definition", f"family {t!r} of color.types has no "
                          f"patterns in the smarts_list definition", tier="scan")
    return {"patterns": n, "families": len(obj)}


# ---------------------------------------------------------------------------
# Writer (sections 3, 6-11): in-memory molecules -> files
# ---------------------------------------------------------------------------

@dataclass
class Molecule:
    """One molecule as the writer takes it. ``coords`` is ``int16[n_conf, 3, n_real]``
    in stored atom order; ``features`` are ``(type, atoms)`` tuples in heavy-atom
    indices (any order; canonicalized by the writer)."""
    mol_id: int
    coords: np.ndarray
    vaa: np.ndarray
    features: list = field(default_factory=list)
    color_self: Optional[np.ndarray] = None
    ext_id: str = ""
    smiles: str = ""
    mw: float = 0.0
    rotors: int = 0

    @property
    def n_conf(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_real(self) -> int:
        return int(self.coords.shape[2])


def corpus_order(molecules: list[Molecule]) -> list[Molecule]:
    """Section 9.2: ``(bucket, n_real, mol_id)``; conformers keep their order."""
    return sorted(molecules, key=lambda m: (bucket_of(m.n_real), m.n_real, m.mol_id))


def split_runs(items: list, sizes: list[int], cap: int) -> list[list]:
    """The write rule of sections 6 (rule 5) and 9.3: append while
    ``size + item <= cap``, a lone item may exceed ``cap``."""
    out: list[list] = []
    cur: list = []
    cur_size = 0
    for item, size in zip(items, sizes):
        if cur and cur_size + size > cap:
            out.append(cur)
            cur, cur_size = [], 0
        cur.append(item)
        cur_size += size
    if cur:
        out.append(cur)
    return out


def encode_group(molecules: list[Molecule], *, bucket: int, group: int,
                 n_types: int, with_color_self: bool, block_rows: int,
                 L_field_width: int, fingerprint_hex: str) -> tuple[dict, dict, list]:
    """Encode one group's sections. Returns ``(sections, fields, block_records)``
    where ``sections`` maps name -> bytes (in section order), ``fields`` is the
    header without ``sections``, and ``block_records`` are the block table
    entries. Kept separate from assembly so a test can mutate before hashing."""
    if not molecules:
        raise ValueError("a group has at least one molecule")
    if any(not 1 <= m.n_conf <= LIMITS["n_conf_per_molecule"] for m in molecules):
        raise ValueError("n_conf per molecule must be in [1, 4096]")
    n_conf = sum(m.n_conf for m in molecules)
    n_mol = len(molecules)
    color = n_types > 0
    feats = [canonical_feature_order(m.features) if color else [] for m in molecules]
    L_heavy = pad8(max(m.n_real for m in molecules))
    L = pad8(max(m.n_real + len(f) for m, f in zip(molecules, feats)))
    blocks = split_runs(list(range(n_mol)), [m.n_conf for m in molecules], block_rows)
    if len(blocks) > LIMITS["blocks"]:
        raise ValueError("group exceeds the block count limit")
    coords_parts: list[bytes] = []
    records: list[dict] = []
    first_row = 0
    offset = 0
    for members in blocks:
        m0, m1 = members[0], members[-1] + 1
        rows = sum(molecules[m].n_conf for m in members)
        xyz = np.zeros((rows, 3, L_heavy), dtype=np.int16)
        n_real = np.zeros(rows, dtype=np.int64)
        r = 0
        for m in members:
            mol = molecules[m]
            xyz[r:r + mol.n_conf, :, :mol.n_real] = mol.coords
            n_real[r:r + mol.n_conf] = mol.n_real
            r += mol.n_conf
        data = encode_block(xyz, n_real)
        records.append({"first_row": first_row, "rows": rows, "offset": offset,
                        "length": len(data), "m0": m0, "m1": m1,
                        "sha256": sha256_hex(data)})
        coords_parts.append(data)
        first_row += rows
        offset += len(data)
    sections: dict[str, bytes] = {}
    sections["coords"] = b"".join(coords_parts)
    sections["blocks"] = pack_blocks(records).tobytes()
    sections["runs"] = pack_runs([m.mol_id for m in molecules],
                                 [m.n_conf for m in molecules],
                                 [m.n_real for m in molecules]).tobytes()
    if color:
        for name, arr in pack_features(feats).items():
            sections[name] = arr.tobytes()
    sections["vaa"] = np.concatenate([np.asarray(m.vaa, dtype="<f4") for m in molecules]).tobytes()
    if color and with_color_self:
        sections["color_self"] = np.concatenate(
            [np.asarray(m.color_self, dtype="<f8") for m in molecules]).tobytes()
    fields = {"major": MAJOR, "fingerprint": fingerprint_hex, "bucket": bucket,
              "group": group, "n_conf": n_conf, "n_mol": n_mol, "n_types": n_types,
              "L_heavy": L_heavy, "L": L}
    return sections, fields, records


def section_shape(name: str, data: bytes) -> list[int]:
    dt = dict((n, d) for n, d, _ in SECTION_ORDER)[name]
    item = DTYPES[dt].itemsize
    if name == "blocks":
        return [len(data) // BLOCK_RECORD_SIZE, BLOCK_RECORD_SIZE]
    if name == "runs":
        return [len(data) // RUN_RECORD_SIZE, RUN_RECORD_SIZE]
    return [len(data) // item]


def assemble_group(sections: dict[str, bytes], fields: dict,
                   *, header_mutate: Optional[Callable[[dict], None]] = None) -> bytes:
    """Lay out sections (section 3.2), build the header (3.4) and the file.
    ``header_mutate`` edits the header dict before serialization (for
    adversarial seeds); the section hashes are of the bytes as given."""
    offset = FIRST_SECTION_OFFSET
    table = []
    body = bytearray()
    names = [n for n, _, _ in SECTION_ORDER if n in sections]
    dt = dict((n, d) for n, d, _ in SECTION_ORDER)
    for name in names:
        data = sections[name]
        entry = {"name": name, "offset": offset, "length": len(data),
                 "dtype": dt[name], "shape": section_shape(name, data),
                 "sha256": sha256_hex(data)}
        table.append(entry)
        body += data
        end = roundup64(offset + len(data))
        body += b"\0" * (end - offset - len(data))
        offset = end
    header = dict(fields)
    header["sections"] = table
    if header_mutate is not None:
        header_mutate(header)
    header_bytes = canonical_json(header)
    if header_mutate is None:
        loads_strict(header_bytes, what="group header")
    header_offset = FIRST_SECTION_OFFSET + len(body)
    prologue = MAGIC + struct.pack("<IQ", MAJOR, header_offset) + b"\0" * 48
    return bytes(prologue) + bytes(body) + header_bytes


def header_sha256_of_file(data: bytes) -> str:
    header_offset = struct.unpack_from("<Q", data, 8)[0]
    return sha256_hex(data[header_offset:])



def write_parquet(path: str, molecules: list[Molecule]) -> None:
    """Section 11: REQUIRED columns, ZSTD level 9, PLAIN, no statistics, no
    embedded Arrow schema, no dictionary pages, row groups of at most 2^14
    rows; declared uncompressed-size bounds are best-effort."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    mols = sorted(molecules, key=lambda m: m.mol_id)
    schema = pa.schema([pa.field(n, t, nullable=False) for n, t in (
        ("mol_id", pa.uint64()), ("ext_id", pa.string()), ("smiles", pa.string()),
        ("hac", pa.uint8()), ("mw", pa.float64()), ("rotors", pa.uint16()))])
    table = pa.table({
        "mol_id": [m.mol_id for m in mols], "ext_id": [m.ext_id for m in mols],
        "smiles": [m.smiles for m in mols], "hac": [m.n_real for m in mols],
        "mw": [float(m.mw) for m in mols], "rotors": [int(m.rotors) for m in mols],
    }, schema=schema)
    pq.write_table(table, path, compression="zstd", compression_level=9, write_statistics=False,
                   store_schema=False, use_dictionary=False,
                   row_group_size=LIMITS["parquet_row_group"])


def write_corpus(root: str, molecules: list[Molecule], *, color: Optional[dict],
                 prep_version: str, prep_commit: Optional[str],
                 block_rows: int = 4096, max_conf_per_group: int = 2_000_000,
                 build: Optional[dict] = None, with_color_self: Optional[bool] = None,
                 group_mutate: Optional[Callable[[str, dict, dict, list], None]] = None,
                 header_mutate: Optional[Callable[[str, dict], None]] = None,
                 descriptor_mutate: Optional[Callable[[dict], None]] = None) -> dict:
    """Write a whole corpus (sections 1, 9-12). Returns the descriptor.

    The three ``*_mutate`` hooks exist for the adversarial seeds: they edit a
    group's sections/fields/records, a group header, or the descriptor before
    the hashes that cover them are computed, so a seed fails the check it
    targets and not an integrity check in front of it.
    """
    if not 1 <= block_rows <= LIMITS["block_rows"]:
        raise ValueError("block_rows must be in [1, 4096]")
    n_types = len(color["types"]) if color else 0
    if with_color_self is None:
        with_color_self = bool(color) and canonical_matrices(
            color["r"]["values"], color["p"]["values"], n_types)
    descriptor: dict = {"format": dict(FORMAT_OBJECT), "frame": dict(FRAME)}
    if color:
        descriptor["color"] = color
    descriptor["prep"] = {"version": prep_version, "commit": prep_commit,
                          "conventions": dict(CONVENTIONS)}
    fp = fingerprint(descriptor)
    ordered = corpus_order(molecules)
    os.makedirs(os.path.join(root, "groups"), exist_ok=True)
    groups_table = []
    by_bucket: dict[int, list[Molecule]] = {}
    for m in ordered:
        by_bucket.setdefault(bucket_of(m.n_real), []).append(m)
    for b in sorted(by_bucket):
        mols = by_bucket[b]
        L_field_width = pad8(max(m.n_real for m in mols))
        groups = split_runs(mols, [m.n_conf for m in mols], max_conf_per_group)
        for g, members in enumerate(groups):
            name = group_file_name(b, g)
            sections, fields, records = encode_group(
                members, bucket=b, group=g, n_types=n_types,
                with_color_self=with_color_self, block_rows=block_rows,
                L_field_width=L_field_width, fingerprint_hex=fp)
            if group_mutate is not None:
                group_mutate(name, sections, fields, records)
                sections["blocks"] = pack_blocks(records).tobytes()
            hm = (lambda h, _n=name: header_mutate(_n, h)) if header_mutate else None
            data = assemble_group(sections, fields, header_mutate=hm)
            with open(os.path.join(root, name), "wb") as fh:
                fh.write(data)
            entry = {k: fields[k] for k in ("bucket", "group", "L_heavy", "L",
                                            "n_conf", "n_mol")}
            entry["L_field_width"] = L_field_width
            entry["header_sha256"] = header_sha256_of_file(data)
            groups_table.append(entry)
    descriptor["groups"] = groups_table
    descriptor["n_conf"] = sum(e["n_conf"] for e in groups_table)
    descriptor["n_mol"] = sum(e["n_mol"] for e in groups_table)
    descriptor["ids_sha256"] = ids_sha256([m.mol_id for m in molecules])
    pq_path = os.path.join(root, "molecules.parquet")
    write_parquet(pq_path, molecules)
    with open(pq_path, "rb") as fh:
        descriptor["molecules_sha256"] = sha256_hex(fh.read())
    if descriptor_mutate is not None:
        descriptor_mutate(descriptor)
    with open(os.path.join(root, "corpus.json"), "wb") as fh:
        data = canonical_json(descriptor)
        if descriptor_mutate is None:
            loads_strict(data, what="corpus.json")
        fh.write(data)
    if build is not None:
        with open(os.path.join(root, "build.json"), "wb") as fh:
            fh.write(canonical_json(build))
    return descriptor


# ---------------------------------------------------------------------------
# Reader + validator (section 16)
# ---------------------------------------------------------------------------

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _open_at(name: str, flags: int, dir_fd: Optional[int], path: str, *, directory: bool) -> int:
    """``os.open(name, flags, dir_fd=dir_fd)`` with the refusals of section 1."""
    try:
        return os.open(name, flags, dir_fd=dir_fd) if dir_fd is not None else os.open(name, flags)
    except FileNotFoundError:
        raise _refuse("io.missing_file", "file is missing", path=path) from None
    except OSError as e:
        what = "a directory" if directory else "a regular file"
        raise _refuse("io.not_regular_file", f"cannot open as {what}: {e}", path=path) from None


def _open_regular_at(name: str, dir_fd: Optional[int], path: str) -> int:
    """Open one file (section 1 item 2): ``O_RDONLY | O_NOFOLLOW | O_NONBLOCK``,
    relative to ``dir_fd``; ``fstat`` and refuse anything but ``S_ISREG``;
    then clear ``O_NONBLOCK``. ``name`` is a single path component."""
    # O_NONBLOCK: an open() on a FIFO blocks until a writer appears; with it
    # the open returns and fstat then refuses the non-regular file.
    fd = _open_at(name, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd, path, directory=False)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise _refuse("io.not_regular_file", "not a regular file", path=path)
    if _O_NONBLOCK:
        # a regular file never blocks; clear the flag so reads are ordinary
        import fcntl
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~_O_NONBLOCK)
    return fd


class CorpusDir:
    """The corpus directory as file descriptors (section 1 item 2).

    The root is the path the user named and is opened once with
    ``O_DIRECTORY`` (following a symlink there is the user's choice: corpora
    are shared by symlink). ``groups/`` is opened relative to the root fd with
    ``O_DIRECTORY | O_NOFOLLOW``, and every corpus file relative to its
    directory fd with ``O_NOFOLLOW``. Names are derived (section 1 item 1) and
    are single components, so no path is ever resolved through a symlink the
    corpus controls, and nothing can race a containment check: there is none
    to race."""

    def __init__(self, root: str):
        self.root = root
        self.fd = _open_at(root, os.O_RDONLY | _O_DIRECTORY, None, root, directory=True)
        self._groups_fd: Optional[int] = None

    def groups_fd(self) -> int:
        if self._groups_fd is None:
            self._groups_fd = _open_at("groups", os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, self.fd,
                                       os.path.join(self.root, "groups"), directory=True)
        return self._groups_fd

    def open(self, rel: str) -> int:
        """Open ``corpus.json``, ``molecules.parquet``, ``build.json`` or
        ``groups/<name>.r3g`` (a derived name, never one read from a file)."""
        parts = rel.split("/")
        path = os.path.join(self.root, *parts)
        if len(parts) == 1 and parts[0] not in ("", ".", ".."):
            return _open_regular_at(parts[0], self.fd, path)
        if len(parts) == 2 and parts[0] == "groups" and parts[1] not in ("", ".", ".."):
            return _open_regular_at(parts[1], self.groups_fd(), path)
        raise _refuse("io.not_regular_file", f"{rel!r} is not a corpus file name", path=path)

    def exists(self, name: str) -> bool:
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False

    def close(self) -> None:
        for fd in (self._groups_fd, self.fd):
            if fd is not None:
                os.close(fd)
        self._groups_fd = None
        self.fd = -1

    def __enter__(self) -> "CorpusDir":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _open_regular(path: str, root: Optional[str] = None) -> int:
    """Open one corpus file per section 1 item 2. With ``root``, ``path`` must
    be ``root/<file>`` or ``root/groups/<file>`` and is opened through
    directory fds (``CorpusDir``); without it (a standalone group file) the
    path is the user's and only its last component is ``O_NOFOLLOW``."""
    if root is None:
        return _open_regular_at(path, None, path)
    rel = os.path.relpath(path, root)
    if os.path.isabs(rel) or rel.startswith(".."):
        raise _refuse("io.not_regular_file", "path escapes the corpus directory", path=path)
    with CorpusDir(root) as cd:
        return cd.open(rel.replace(os.sep, "/"))


def _pread_exact(fd: int, n: int, offset: int, path: str) -> bytes:
    """Read exactly ``n`` bytes at ``offset`` (section 1 item 4: ``pread``
    into a buffer sized from checked values, never ``mmap``). A short read
    -- the file shrank after ``fstat`` -- is a refusal, not a ``SIGBUS``."""
    parts = []
    got = 0
    while got < n:
        chunk = os.pread(fd, min(n - got, 1 << 30), offset + got)
        if not chunk:
            raise _refuse("io.short_read", f"read {got} of {n} bytes at offset {offset}: "
                          f"the file is shorter than when it was opened", path=path)
        parts.append(chunk)
        got += len(chunk)
    return parts[0] if len(parts) == 1 else b"".join(parts)


@dataclass
class GroupFile:
    path: str
    size: int
    header_offset: int
    header: dict
    header_bytes: bytes
    sections: dict          # name -> section table entry
    fd: int                 # the descriptor everything is read from (pread)
    _blocks: Optional[np.ndarray] = None

    def read(self, offset: int, n: int) -> bytes:
        return _pread_exact(self.fd, n, offset, self.path)

    def section_bytes(self, name: str) -> bytes:
        e = self.sections[name]
        return self.read(e["offset"], e["length"])

    def section_array(self, name: str) -> np.ndarray:
        e = self.sections[name]
        buf = self.section_bytes(name)
        if name == "blocks":
            return np.frombuffer(buf, dtype=BLOCK_DTYPE)
        if name == "runs":
            return np.frombuffer(buf, dtype=RUN_DTYPE)
        return np.frombuffer(buf, dtype=DTYPES[e["dtype"]])

    def blocks(self) -> np.ndarray:
        if self._blocks is None:
            self._blocks = self.section_array("blocks")
        return self._blocks

    def block_bytes(self, i: int) -> bytes:
        b = self.blocks()[i]
        base = self.sections["coords"]["offset"]
        return self.read(base + int(b["offset"]), int(b["length"]))

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def open_group(path: str, *, root: Optional[str] = None, descriptor: Optional[dict] = None,
               group_entry: Optional[dict] = None, fingerprint_hex: Optional[str] = None,
               corpus_dir: Optional["CorpusDir"] = None) -> GroupFile:
    """The ``open`` tier for one group file (section 16.2). With
    ``corpus_dir`` the file is opened relative to its ``groups/`` fd."""
    if corpus_dir is not None:
        fd = corpus_dir.open("groups/" + os.path.basename(path))
    else:
        fd = _open_regular(path, root)
    try:
        size = os.fstat(fd).st_size
        if size < FIRST_SECTION_OFFSET:
            raise _refuse("prologue.magic", f"file is {size} bytes, shorter than the prologue",
                          path=path)
        return _open_group_data(path, size, fd, descriptor, group_entry, fingerprint_hex)
    except BaseException:
        os.close(fd)
        raise


def _open_group_data(path, size, fd, descriptor, group_entry, fingerprint_hex) -> GroupFile:
    pro = _pread_exact(fd, FIRST_SECTION_OFFSET, 0, path)
    if pro[:4] != MAGIC:
        raise _refuse("prologue.magic", f"magic {pro[:4]!r} is not {MAGIC!r}", path=path)
    major, header_offset = struct.unpack_from("<IQ", pro, 4)
    if major != MAJOR:
        raise _refuse("prologue.major_version", f"major version {major}; this reader "
                      f"implements {MAJOR} only", path=path)
    if not FIRST_SECTION_OFFSET <= header_offset <= size:
        raise _refuse("prologue.header_offset", f"header_offset {header_offset} outside "
                      f"[64, {size}]", path=path)
    if any(pro[PROLOGUE_SIZE:]):
        raise _refuse("prologue.reserved", "reserved prologue bytes are not zero", path=path)
    header_len = size - header_offset
    if header_len > LIMITS["json_bytes"]:
        raise _refuse("json.size", f"header is {header_len} bytes, limit "
                      f"{LIMITS['json_bytes']}", path=path)
    header_bytes = _pread_exact(fd, header_len, header_offset, path)      # bounded read
    header = loads_strict(header_bytes, what="group header", path=path)
    what = "header"
    # The header hash binds everything below to corpus.json: check it before
    # trusting any header value (section 16.2 item 1).
    if group_entry is not None and sha256_hex(header_bytes) != group_entry["header_sha256"]:
        raise _refuse("integrity.header_hash", "header bytes do not hash to the group "
                      "table's header_sha256", path=path)
    _require_keys(header, ("major", "fingerprint", "bucket", "group", "n_conf", "n_mol",
                           "n_types", "L_heavy", "L", "sections"), (),
                  what=what, path=path)
    if canonical_json(header) != header_bytes:
        raise _refuse("schema.value", "header is not canonical JSON", path=path)
    if _int(header, "major", what=what, path=path) != MAJOR:
        raise _refuse("prologue.major_version", f"header major {header['major']} != {MAJOR}",
                      path=path)
    fp = _hex64(header, "fingerprint", what=what, path=path)
    bucket = _int(header, "bucket", what=what, path=path, hi=N_BUCKETS - 1)
    group = _int(header, "group", what=what, path=path, hi=LIMITS["groups"] - 1, limit="groups")
    n_conf = _int(header, "n_conf", what=what, path=path, lo=1, hi=LIMITS["n_conf"], limit="n_conf")
    n_mol = _int(header, "n_mol", what=what, path=path, lo=1, hi=n_conf, limit="n_mol")
    n_types = _int(header, "n_types", what=what, path=path, hi=LIMITS["n_types"], limit="n_types")
    L_heavy = _int(header, "L_heavy", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L_heavy"], limit="L_heavy")
    L = _int(header, "L", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L"], limit="L")
    for v, nm in ((L_heavy, "L_heavy"), (L, "L")):
        if v % L_ALIGN:
            raise _refuse("schema.value", f"{nm} = {v} is not a multiple of 8", path=path)
    if L < L_heavy or (n_types == 0 and L != L_heavy):
        raise _refuse("layout.widths", f"L = {L} inconsistent with L_heavy = {L_heavy}", path=path)
    # --- section table (3.2, 3.3)
    expected = [(n, d) for n, d, c in SECTION_ORDER if (not c) or n_types > 0]
    table = header["sections"]
    if not isinstance(table, list):
        raise _refuse("schema.type", "sections must be a list", path=path)
    sections: dict[str, dict] = {}
    offset = FIRST_SECTION_OFFSET
    names_seen = []
    for i, e in enumerate(table):
        if not isinstance(e, dict):
            raise _refuse("schema.type", f"sections[{i}] must be an object", path=path)
        _require_keys(e, ("name", "offset", "length", "dtype", "shape", "sha256"), (),
                      what=f"sections[{i}]", path=path)
        name = e["name"]
        if not isinstance(name, str) or name not in dict((n, d) for n, d, _ in SECTION_ORDER):
            raise _refuse("schema.type", f"sections[{i}]: unknown section {name!r}", path=path)
        if name in sections:
            raise _refuse("layout.sections", f"section {name!r} listed twice", path=path)
        if not isinstance(e["dtype"], str) or e["dtype"] not in DTYPES:
            raise _refuse("schema.type", f"section {name!r}: unknown dtype {e['dtype']!r}", path=path)
        off = _int(e, "offset", what=name, path=path)
        ln = _int(e, "length", what=name, path=path)
        shape = e["shape"]
        if (not isinstance(shape, list) or not shape
                or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= INT64_MAX
                       for s in shape)):
            raise _refuse("schema.type", f"section {name!r}: shape must be a list of non-negative integers",
                          path=path)
        _hex64(e, "sha256", what=name, path=path)
        want_dt = dict((n, d) for n, d, _ in SECTION_ORDER)[name]
        if e["dtype"] != want_dt:
            raise _refuse("layout.sections", f"section {name!r}: dtype {e['dtype']} != {want_dt}", path=path)
        if len(shape) != (2 if name in ("blocks", "runs") else 1):
            raise _refuse("layout.sections", f"section {name!r}: invalid shape rank", path=path)
        prod = 1
        for s in shape:
            prod *= s
        if ln != prod * DTYPES[e["dtype"]].itemsize:
            raise _refuse("layout.sections", f"section {name!r}: length {ln} != prod(shape) * itemsize",
                          path=path)
        if off % 64:
            raise _refuse("layout.sections", f"section {name!r}: offset {off} not 64-aligned", path=path)
        if off != offset:
            raise _refuse("layout.sections", f"section {name!r}: offset {off}, expected {offset} "
                          f"(sections must tile the payload)", path=path)
        if off + ln > header_offset:
            raise _refuse("layout.sections", f"section {name!r} runs into the header", path=path)
        offset = roundup64(off + ln)
        sections[name] = e
        names_seen.append(name)
    if offset != header_offset:
        raise _refuse("layout.sections", f"sections end at {offset}, header at {header_offset}",
                      path=path)
    want_names = [n for n, _ in expected]
    if "color_self" in want_names and "color_self" not in sections:
        want_names.remove("color_self")          # optional (section 18.2)
    if names_seen != want_names:
        raise _refuse("layout.sections", f"sections {names_seen} != expected {want_names}", path=path)
    # shapes
    def shape_check(name, want):
        got = sections[name]["shape"]
        if got != want:
            raise _refuse("layout.sections", f"section {name!r}: shape {got} != {want}", path=path)
    n_blocks = sections["blocks"]["shape"][0]
    if n_blocks < 1 or n_blocks > LIMITS["blocks"]:
        raise _refuse("limit.blocks", f"{n_blocks} blocks", path=path)
    shape_check("blocks", [n_blocks, BLOCK_RECORD_SIZE])
    shape_check("runs", [n_mol, RUN_RECORD_SIZE])
    shape_check("coords", [sections["coords"]["length"]])
    shape_check("vaa", [n_conf])
    if n_types:
        shape_check("feature_offsets", [n_mol + 1])
        n_feat = sections["feature_types"]["shape"][0]
        n_feat_atoms = sections["feature_atoms"]["shape"][0]
        if max(n_feat, n_feat_atoms) > LIMITS["n_feat_atoms"]:
            raise _refuse("limit.n_feat_atoms", f"{n_feat} features / {n_feat_atoms} feature atoms; "
                          f"the u32 CSR offsets hold at most {LIMITS['n_feat_atoms']}", path=path)
        shape_check("feature_types", [n_feat])
        shape_check("feature_atom_offsets", [n_feat + 1])
        shape_check("feature_atoms", [sections["feature_atoms"]["shape"][0]])
        if "color_self" in sections:
            shape_check("color_self", [n_conf])
    # --- block table (16.2 item 4); its section hash is verified first, so
    # what is walked is what the (already verified) header describes
    gf = GroupFile(path, size, header_offset, header, header_bytes, sections, fd)
    if sha256_hex(gf.section_bytes("blocks")) != sections["blocks"]["sha256"]:
        raise _refuse("integrity.section_hash", "the blocks section does not hash to the header",
                      path=path)
    blocks = gf.blocks()
    row = 0
    m = 0
    boff = 0
    for i in range(n_blocks):
        b = blocks[i]
        rows, m0, m1 = int(b["rows"]), int(b["m0"]), int(b["m1"])
        if int(b["first_row"]) != row or m0 != m or int(b["offset"]) != boff:
            raise _refuse("layout.blocks", f"block {i} does not tile (row {b['first_row']}/{row}, "
                          f"m0 {m0}/{m}, offset {b['offset']}/{boff})", path=path)
        if rows < 1:
            raise _refuse("layout.blocks", f"block {i} is empty", path=path)
        if m1 <= m0:
            raise _refuse("layout.blocks", f"block {i}: m1 {m1} <= m0 {m0}", path=path)
        if m1 - m0 == 1:
            if rows > LIMITS["block_rows_single_molecule"]:
                raise _refuse("limit.block_rows", f"block {i}: {rows} rows for one molecule, limit "
                              f"{LIMITS['block_rows_single_molecule']}", path=path)
        elif rows > LIMITS["block_rows"]:
            raise _refuse("limit.block_rows", f"block {i}: {rows} rows, limit {LIMITS['block_rows']}",
                          path=path)
        ln = int(b["length"])
        if ln < BLOCK_HEADER_SIZE or ln % 32:
            raise _refuse("layout.blocks", f"block {i}: length {ln} is not a valid block length",
                          path=path)
        row += rows
        m = m1
        boff += ln
    if row != n_conf or m != n_mol or boff != sections["coords"]["length"]:
        raise _refuse("layout.blocks", f"blocks cover {row} rows / {m} molecules / {boff} bytes; "
                      f"header says {n_conf} / {n_mol} / {sections['coords']['length']}", path=path)
    # --- binding to the descriptor (16.2 item 5)
    if fingerprint_hex is not None and fp != fingerprint_hex:
        raise _refuse("integrity.fingerprint", f"group fingerprint {fp[:12]}... != corpus "
                      f"{fingerprint_hex[:12]}...", path=path)
    if descriptor is not None:
        want_types = len(descriptor["color"]["types"]) if "color" in descriptor else 0
        if n_types != want_types:
            raise _refuse("schema.value", f"n_types {n_types} != descriptor's {want_types}", path=path)
        if want_types and "color_self" in sections:
            c = descriptor["color"]
            if not canonical_matrices(c["r"]["values"], c["p"]["values"], want_types):
                raise _refuse("layout.sections", "color_self is present but the interaction "
                              "matrices are not canonical", path=path)
    if group_entry is not None:
        for k in ("bucket", "group", "n_conf", "n_mol", "L_heavy", "L"):
            if header[k] != group_entry[k]:
                raise _refuse("schema.value", f"header {k} = {header[k]} != group table "
                              f"{group_entry[k]}", path=path)
    return gf


def read_descriptor(root: str, corpus_dir: Optional[CorpusDir] = None) -> tuple[dict, str]:
    """``open`` tier for ``corpus.json`` (section 16.2). Returns ``(descriptor, fingerprint)``."""
    path = os.path.join(root, "corpus.json")
    if corpus_dir is None:
        with CorpusDir(root) as cd:
            return read_descriptor(root, cd)
    fd = corpus_dir.open("corpus.json")
    try:
        size = os.fstat(fd).st_size
        if size > LIMITS["json_bytes"]:
            raise _refuse("json.size", f"corpus.json is {size} bytes, limit {LIMITS['json_bytes']}",
                          path=path)
        data = _pread_exact(fd, size, 0, path)
    finally:
        os.close(fd)
    d = loads_strict(data, what="corpus.json", path=path)
    _require_keys(d, ("format", "frame", "prep", "groups", "n_conf", "n_mol", "ids_sha256",
                      "molecules_sha256"), ("color",), what="corpus.json", path=path)
    for key, want in (("format", FORMAT_OBJECT), ("frame", FRAME)):
        obj = d[key]
        if not isinstance(obj, dict):
            raise _refuse("schema.type", f"{key} must be an object", path=path)
        _require_keys(obj, want.keys(), (), what=key, path=path)
        for k, v in want.items():
            if obj[k] != v or type(obj[k]) is not type(v):
                code = "prologue.major_version" if k == "major" else "schema.type" if isinstance(v, str) else "schema.value"
                raise _refuse(code, f"{key}.{k} = {obj[k]!r}, this document requires {v!r}", path=path)
    prep = d["prep"]
    if not isinstance(prep, dict):
        raise _refuse("schema.type", "prep must be an object", path=path)
    _require_keys(prep, ("version", "commit", "conventions"), (), what="prep", path=path)
    if not isinstance(prep["version"], str) or not (prep["commit"] is None or isinstance(prep["commit"], str)):
        raise _refuse("schema.type", "prep.version must be a string, prep.commit a string or null", path=path)
    conv = prep["conventions"]
    if not isinstance(conv, dict):
        raise _refuse("schema.type", "prep.conventions must be an object", path=path)
    _require_keys(conv, CONVENTIONS.keys(), (), what="prep.conventions", path=path)
    for k, v in CONVENTIONS.items():
        if conv[k] != v:
            raise _refuse("schema.type", f"prep.conventions.{k} = {conv[k]!r}, requires {v!r}", path=path)
    n_types = 0
    if "color" in d:
        c = d["color"]
        if not isinstance(c, dict):
            raise _refuse("schema.type", "color must be an object", path=path)
        _require_keys(c, ("generator", "definition", "types", "positioning", "r", "p", "rdkit_version"),
                      (), what="color", path=path)
        if c["generator"] not in GENERATORS:
            raise _refuse("schema.type", f"color.generator {c['generator']!r} unknown", path=path)
        if c["positioning"] != POSITIONING:
            raise _refuse("schema.type", f"color.positioning {c['positioning']!r} != {POSITIONING!r}", path=path)
        if not isinstance(c["rdkit_version"], str):
            raise _refuse("schema.type", "color.rdkit_version must be a string", path=path)
        want = str if c["generator"] == "rdkit_fdef" else dict
        if not isinstance(c["definition"], want):
            raise _refuse("schema.type", f"color.definition must be a {want.__name__} for "
                          f"generator {c['generator']}", path=path)
        types = c["types"]
        if (not isinstance(types, list) or not types or any(not isinstance(t, str) for t in types)
                or len(set(types)) != len(types)):
            raise _refuse("schema.type", "color.types must be a non-empty list of unique strings", path=path)
        n_types = len(types)
        if n_types > LIMITS["n_types"]:
            raise _refuse("limit.n_types", f"{n_types} color types, limit {LIMITS['n_types']}", path=path)
        for key in ("r", "p"):
            m = c[key]
            if not isinstance(m, dict):
                raise _refuse("schema.type", f"color.{key} must be an object", path=path)
            _require_keys(m, ("dtype", "shape", "values"), (), what=f"color.{key}", path=path)
            if m["dtype"] != "f32" or m["shape"] != [n_types + 1, n_types + 1]:
                raise _refuse("value.matrix", f"color.{key}: dtype/shape must be f32 [{n_types + 1}, {n_types + 1}]",
                              path=path)
            vals = m["values"]
            if (not isinstance(vals, list) or len(vals) != n_types + 1
                    or any(not isinstance(rw, list) or len(rw) != n_types + 1 for rw in vals)):
                raise _refuse("value.matrix", f"color.{key}: values must be {n_types + 1} x {n_types + 1}", path=path)
            for rw in vals:
                for x in rw:
                    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                        raise _refuse("value.matrix", f"color.{key}: non-finite or non-numeric value", path=path)
                    if float(np.float32(x)) != float(x):
                        raise _refuse("value.matrix", f"color.{key}: {x!r} is not float32-representable", path=path)
                    if key == "r" and x < 0:
                        raise _refuse("value.matrix", f"color.r: negative width {x!r}", path=path)
        problems = interaction_matrix_problems(c["r"]["values"], c["p"]["values"])
        if problems:
            raise _refuse("value.matrix", "color matrices: " + "; ".join(problems), path=path)
    groups = d["groups"]
    if not isinstance(groups, list) or not groups:
        raise _refuse("schema.type", "groups must be a non-empty list", path=path)
    if len(groups) > LIMITS["groups"]:
        raise _refuse("limit.groups", f"{len(groups)} groups, limit {LIMITS['groups']}", path=path)
    last = (-1, -1)
    tot_conf = tot_mol = 0
    for i, g in enumerate(groups):
        what = f"groups[{i}]"
        if not isinstance(g, dict):
            raise _refuse("schema.type", f"{what} must be an object", path=path)
        _require_keys(g, ("bucket", "group", "L_heavy", "L", "L_field_width", "n_conf", "n_mol",
                          "header_sha256"), (), what=what, path=path)
        b = _int(g, "bucket", what=what, path=path, hi=N_BUCKETS - 1)
        gi = _int(g, "group", what=what, path=path)
        key = (b, gi)
        if key <= last or (b == last[0] and gi != last[1] + 1) or (b != last[0] and gi != 0):
            raise _refuse("schema.value", f"{what}: groups out of order or with a gap at {key}", path=path)
        last = key
        _int(g, "L_heavy", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L_heavy"], limit="L_heavy")
        _int(g, "L", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L"], limit="L")
        _int(g, "L_field_width", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L_field_width"], limit="L_field_width")
        tot_conf += _int(g, "n_conf", what=what, path=path, lo=1, hi=LIMITS["n_conf"], limit="n_conf")
        tot_mol += _int(g, "n_mol", what=what, path=path, lo=1)
        _hex64(g, "header_sha256", what=what, path=path)
    if _int(d, "n_conf", what="corpus.json", path=path) != tot_conf:
        raise _refuse("schema.value", f"n_conf {d['n_conf']} != sum of groups {tot_conf}", path=path)
    if _int(d, "n_mol", what="corpus.json", path=path) != tot_mol:
        raise _refuse("schema.value", f"n_mol {d['n_mol']} != sum of groups {tot_mol}", path=path)
    _hex64(d, "ids_sha256", what="corpus.json", path=path)
    _hex64(d, "molecules_sha256", what="corpus.json", path=path)
    return d, fingerprint(d)


# ---- scan tier ------------------------------------------------------------

def _scan_group(gf: GroupFile, descriptor: dict) -> dict:
    """Section 16.3 for one group, vectorized so its cost is linear in the
    section bytes with no per-molecule Python object. Returns derived facts."""
    path = gf.path
    h = gf.header
    n_types = h["n_types"]
    runs = gf.section_array("runs")
    n_mol = int(runs.size)
    if np.any(runs["reserved"]):
        raise _refuse("layout.runs", "reserved run-table bytes are not zero", tier="scan", path=path)
    n_real = runs["n_real"].astype(np.int64)
    n_conf = runs["n_conf"].astype(np.int64)
    mol_id = runs["mol_id"].astype(np.uint64)
    if n_real.min() < 1:
        raise _refuse("limit.n_real", "a molecule has n_real = 0", tier="scan", path=path)
    if n_conf.min() < 1 or n_conf.max() > LIMITS["n_conf_per_molecule"]:
        raise _refuse("limit.n_conf_per_molecule", "a molecule has n_conf outside [1, 4096]", tier="scan", path=path)
    if int(mol_id.max()) >= LIMITS["mol_id"]:
        raise _refuse("limit.mol_id", "a mol_id is >= 2^63", tier="scan", path=path)
    # sorted by (n_real, mol_id), strictly: consecutive keys ascend
    key_ok = (np.diff(n_real) > 0) | ((np.diff(n_real) == 0) & (np.diff(mol_id.astype(np.int64)) > 0))
    if not np.all(key_ok):
        raise _refuse("layout.runs", "run table is not sorted by (n_real, mol_id) or repeats a mol_id",
                      tier="scan", path=path)
    b = h["bucket"]
    if n_real.min() < BUCKET_EDGES[b] or n_real.max() >= BUCKET_EDGES[b + 1]:
        raise _refuse("layout.runs", f"a molecule's n_real is outside bucket {b}", tier="scan", path=path)
    if int(n_conf.sum()) != h["n_conf"]:
        raise _refuse("layout.runs", f"run table sums to {int(n_conf.sum())} rows, header says {h['n_conf']}",
                      tier="scan", path=path)
    run_start = np.concatenate([[0], np.cumsum(n_conf)])
    # feature table (8.1), vectorized and chunked: the stored arrays stay in
    # their stored dtypes and every int64 temporary is per chunk of
    # FEATURE_CHUNK features (<= 16 x that many atom entries), so the scan's
    # memory is the sections plus a constant (limit 34)
    n_color = np.zeros(n_mol, dtype=np.int64)
    feats = None
    if n_types:
        feats = {name: gf.section_array(name) for name in
                 ("feature_offsets", "feature_types", "feature_atom_offsets", "feature_atoms")}
        fo = feats["feature_offsets"].astype(np.int64)            # n_mol + 1 entries
        fao, ft, fa = feats["feature_atom_offsets"], feats["feature_types"], feats["feature_atoms"]
        n_feat = int(ft.size)
        if fo[0] != 0 or np.any(np.diff(fo) < 0) or fo[-1] != n_feat:
            raise _refuse("layout.features", "feature_offsets are not a valid CSR", tier="scan", path=path)
        if int(fao[0]) != 0 or int(fao[-1]) != fa.size or np.any(fao[1:] <= fao[:-1]):
            raise _refuse("layout.features", "feature_atom_offsets are not a valid CSR", tier="scan", path=path)
        n_color = np.diff(fo)
        if n_mol and n_color.max() > LIMITS["features_per_molecule"]:
            raise _refuse("limit.features_per_molecule", "a molecule has more than 255 features",
                          tier="scan", path=path)
        # (limit 15, <= 255 features of one type per molecule, is implied by
        # limit 16's <= 255 features per molecule, checked just above)
        if n_feat and (int(ft.min()) < 1 or int(ft.max()) > n_types):
            raise _refuse("layout.features", "a feature type is outside [1, n_types]", tier="scan", path=path)
        width = LIMITS["atoms_per_feature"] + 1
        for f0 in range(0, n_feat, FEATURE_CHUNK):
            # features [f0, f1) are checked; [f0, f2) with one more feature
            # so the order check spans the chunk boundary
            f1 = min(n_feat, f0 + FEATURE_CHUNK)
            f2 = min(n_feat, f1 + 1)
            ao = fao[f0:f2 + 1].astype(np.int64)
            k = np.diff(ao)                                         # atoms per feature
            if np.any(k < 1):
                raise _refuse("layout.features", "feature_atom_offsets are not a valid CSR (a feature "
                              "has no atoms)", tier="scan", path=path)
            if int(k.max()) > LIMITS["atoms_per_feature"]:
                raise _refuse("limit.atoms_per_feature", "a feature has more than 16 atoms", tier="scan",
                              path=path)
            a0, a1 = int(ao[0]), int(ao[-1])
            atoms = fa[a0:a1].astype(np.int64)
            local = np.repeat(np.arange(f2 - f0), k)                # chunk feature of each atom entry
            first = np.zeros(atoms.size, dtype=bool)
            first[ao[:-1] - a0] = True
            if np.any(atoms[1:][~first[1:]] <= atoms[:-1][~first[1:]]):
                raise _refuse("layout.features", "feature atoms are not strictly ascending",
                              tier="scan", path=path)
            mol = np.searchsorted(fo, np.arange(f0, f2), side="right") - 1   # molecule of each feature
            if np.any(atoms >= n_real[mol[local]]):
                raise _refuse("layout.features", "a feature atom index is >= its molecule's n_real",
                              tier="scan", path=path)
            # canonical order within a molecule: consecutive features of one
            # molecule compare <= on (type, atoms padded with -1)
            pad = np.full((f2 - f0, width), -1, dtype=np.int16)
            pad[:, 0] = ft[f0:f2]
            pad[local, np.arange(atoms.size) - (ao[local] - a0) + 1] = atoms
            same = mol[1:] == mol[:-1]
            x, y = pad[:-1][same], pad[1:][same]
            if x.size:
                ne = x != y
                has = ne.any(axis=1)
                col = np.argmax(ne, axis=1)
                r_ = np.arange(x.shape[0])
                if np.any(has & (x[r_, col] > y[r_, col])):
                    raise _refuse("layout.features", "features of a molecule are not in canonical "
                                  "(type, atom tuple) order", tier="scan", path=path)
    # widths (8.6)
    L_heavy = pad8(int(n_real.max()))
    L = pad8(int((n_real + n_color).max()))
    if int(n_real.max()) > LIMITS["n_real"] or L_heavy != h["L_heavy"] or L != h["L"]:
        raise _refuse("layout.widths", f"recomputed L_heavy {L_heavy} / L {L} != declared "
                      f"{h['L_heavy']} / {h['L']}", tier="scan", path=path)
    # blocks vs runs and the 32-byte block headers
    blocks = gf.blocks()
    base = gf.sections["coords"]["offset"]
    m0 = blocks["m0"].astype(np.int64)
    m1 = blocks["m1"].astype(np.int64)
    rows_of = run_start[m1] - run_start[m0]
    if np.any(rows_of != blocks["rows"].astype(np.int64)) or np.any(run_start[m0] != blocks["first_row"].astype(np.int64)):
        raise _refuse("layout.blocks", "a block's rows/first_row disagree with the run table",
                      tier="scan", path=path)
    atoms_cum = np.concatenate([[0], np.cumsum(n_conf * n_real)])
    totals = atoms_cum[m1] - atoms_cum[m0]
    # one 32-byte read per block; at the 2^20-block limit this loop is ~1 s,
    # linear in the block table, within limit 33
    for i in range(blocks.size):
        bl = blocks[i]
        off = base + int(bl["offset"])
        hdr = gf.read(off, BLOCK_HEADER_SIZE)
        bh = parse_block_header(hdr)
        if max(bh.widths) > LIMITS["bit_width"]:
            raise _refuse("limit.bit_width", f"block {i}: bit width {max(bh.widths)} > 16", tier="scan", path=path)
        if any(hdr[9:]):
            raise _refuse("layout.block_header", f"block {i}: reserved header bytes not zero", tier="scan", path=path)
        need = block_length(int(totals[i]), *bh.widths)
        if need != int(bl["length"]):
            raise _refuse("limit.block_length", f"block {i}: length {int(bl['length'])} != closed form {need}",
                          tier="scan", path=path)
    vaa = gf.section_array("vaa")
    if not np.all(np.isfinite(vaa)) or np.any(vaa <= 0):
        raise _refuse("value.self_overlap", "a vaa is not finite or <= 0", tier="scan", path=path)
    if "color_self" in gf.sections:
        cs = gf.section_array("color_self")
        if not np.all(np.isfinite(cs)) or np.any(cs < 0):
            raise _refuse("value.self_overlap", "a color_self is not finite or < 0", tier="scan", path=path)
    return {"mol_id": mol_id, "n_real": n_real, "n_conf": n_conf, "n_color": n_color,
            "features": feats, "run_start": run_start}


# ---- Parquet page headers (Thrift compact protocol), section 11 ----------
#
# The footer's encodings and sizes are declarations. pyarrow decodes whatever
# pages a chunk really holds, so the validator walks every page header of
# every column chunk itself -- a bounded loop over bytes on disk -- and
# refuses any page that is not a supported-codec PLAIN data page before the
# library reads the row group.
#
# The walker is a closed schema, not a generic Thrift skipper. In the
# parquet-format Thrift definition a PageHeader and the two data-page
# sub-headers a legal file may carry hold only i32 and bool fields and an
# empty Statistics struct: no list, set, map, binary or double (writers MUST
# NOT write statistics, section 11). So a container type (9, 10, 11), a binary or
# double, an unknown field id, a field of the wrong type, a repeated or
# out-of-order field id, and a header longer than PAGE_HEADER_MAX bytes are
# all refused outright. A generic skipper is what made a 7-byte header
# declaring a list of 2^62 bools loop for hours (a bool element consumes no
# bytes in the compact protocol).

#: Largest legal page header, from the schema below: PageHeader's four i32
#: fields (type, sizes, crc) at <= 9 bytes each (a field header of <= 4 bytes
#: in the long form, a varint of <= 5) = 36; one sub-struct field header 4;
#: DataPageHeaderV2's six i32 fields 54, one bool 4, an empty statistics
#: struct 4 + 1 and its stop byte 1 (the larger sub-header; DataPageHeader is
#: 4 x 9 + 5 + 1 = 42); PageHeader's stop byte 1: 105 bytes. Capped at 256.
PAGE_HEADER_MAX = 256

_CT_BOOL_TRUE, _CT_BOOL_FALSE, _CT_I32, _CT_STRUCT = 1, 2, 5, 12
_I32, _BOOL = "i32", "bool"
#: Statistics, allowed only empty: pyarrow writes ``1c 00`` (an empty
#: struct) in every data page header even with statistics disabled. Any
#: field inside it (min/max are binary) is refused.
_EMPTY_STATISTICS = ({}, set())
#: DataPageHeader: num_values, encoding, definition_level_encoding,
#: repetition_level_encoding, statistics (empty)
_DATA_PAGE_HEADER = ({1: _I32, 2: _I32, 3: _I32, 4: _I32, 5: _EMPTY_STATISTICS}, {1, 2, 3, 4})
#: DataPageHeaderV2: num_values, num_nulls, num_rows, encoding,
#: definition_levels_byte_length, repetition_levels_byte_length,
#: is_compressed, statistics (empty)
_DATA_PAGE_HEADER_V2 = ({1: _I32, 2: _I32, 3: _I32, 4: _I32, 5: _I32, 6: _I32, 7: _BOOL,
                         8: _EMPTY_STATISTICS}, {1, 2, 3, 4, 5, 6})
#: PageHeader: type, uncompressed_page_size, compressed_page_size, crc,
#: data_page_header, data_page_header_v2 (fields 6 and 7, the index and
#: dictionary page headers, are refused: neither page type is legal)
_PAGE_HEADER = ({1: _I32, 2: _I32, 3: _I32, 4: _I32, 5: _DATA_PAGE_HEADER, 8: _DATA_PAGE_HEADER_V2},
                {1, 2, 3})


def _ct_byte(buf: bytes, pos: int) -> int:
    if pos >= len(buf):
        raise ValueError("page header is truncated or longer than "
                         f"{PAGE_HEADER_MAX} bytes")
    return buf[pos]


def _ct_varint(buf: bytes, pos: int, max_bytes: int) -> tuple[int, int]:
    result = shift = 0
    for _ in range(max_bytes):
        b = _ct_byte(buf, pos)
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
    raise ValueError(f"varint longer than {max_bytes} bytes")


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _ct_struct(buf: bytes, pos: int, schema: tuple) -> tuple[dict, int]:
    """One compact-protocol struct against a closed ``(fields, required)``
    schema; every byte read is bounds-checked, every loop is bounded by the
    buffer (each field consumes at least one byte)."""
    allowed, required = schema
    fields: dict = {}
    last = 0
    while True:
        b = _ct_byte(buf, pos)
        pos += 1
        ftype = b & 0xF
        if ftype == 0:
            missing = required - fields.keys()
            if missing:
                raise ValueError(f"page header lacks required field(s) {sorted(missing)}")
            return fields, pos
        delta = b >> 4
        if delta:
            fid = last + delta
        else:
            z, pos = _ct_varint(buf, pos, 3)          # i16 field id
            fid = _zigzag(z)
        if fid <= last:
            raise ValueError(f"field id {fid} repeated or out of order")
        last = fid
        kind = allowed.get(fid)
        if kind is None:
            raise ValueError(f"field id {fid} (thrift type {ftype}) is not allowed in a page header")
        if kind == _BOOL:
            if ftype not in (_CT_BOOL_TRUE, _CT_BOOL_FALSE):
                raise ValueError(f"field {fid} must be a bool, got thrift type {ftype}")
            fields[fid] = ftype == _CT_BOOL_TRUE
        elif kind == _I32:
            if ftype != _CT_I32:
                raise ValueError(f"field {fid} must be an i32, got thrift type {ftype}")
            z, pos = _ct_varint(buf, pos, 5)
            v = _zigzag(z)
            if not -(1 << 31) <= v < (1 << 31):
                raise ValueError(f"field {fid}: {v} is outside the i32 range")
            fields[fid] = v
        else:
            if ftype != _CT_STRUCT:
                raise ValueError(f"field {fid} must be a struct, got thrift type {ftype}")
            fields[fid], pos = _ct_struct(buf, pos, kind)


def parse_page_header(buf: bytes) -> tuple[dict, int]:
    """Parse one page header from at most ``PAGE_HEADER_MAX`` bytes. Returns
    ``(fields, header_length)``; raises ``ValueError`` on anything outside
    the closed schema."""
    return _ct_struct(bytes(buf[:PAGE_HEADER_MAX]), 0, _PAGE_HEADER)


def walk_parquet_pages(read: Callable[[int, int], bytes], length: int, n_rows: int, *,
                       codec: str = "UNCOMPRESSED", uncompressed_cap: Optional[int] = None) -> None:
    """Walk one column chunk of ``length`` bytes, reading through
    ``read(offset, n)`` (offsets relative to the chunk), so memory is one
    page header at a time. Each page MUST be a DATA_PAGE (v1 or v2), PLAIN,
    ZSTD or uncompressed as declared by the chunk codec, with no nulls, hold at least
    one value, and the pages MUST tile the chunk exactly with ``num_values``
    summing to ``n_rows``. At most ``n_rows`` pages are walked. Raises
    ``ValueError`` naming the first violation."""
    pos = 0
    values = 0
    declared_size = 0
    if codec not in ("UNCOMPRESSED", "ZSTD"):
        raise ValueError("unsupported page codec")
    while pos < length:
        buf = read(pos, min(PAGE_HEADER_MAX, length - pos))
        try:
            hdr, hlen = parse_page_header(buf)
        except ValueError as e:
            raise ValueError(f"page header at {pos}: {e}") from None
        ptype, usize, csize = hdr[1], hdr[2], hdr[3]
        if ptype not in (0, 3):
            raise ValueError(f"page at {pos} is not a data page (type {ptype})")
        declared_size += hlen + usize
        if uncompressed_cap is not None and declared_size > uncompressed_cap:
            raise ValueError("declared uncompressed pages exceed the column bound")
        if usize < 0 or csize < 0 or (codec == "UNCOMPRESSED" and usize != csize):
            raise ValueError(f"page at {pos}: uncompressed size {usize} != compressed {csize}")
        if ptype == 0:
            dph = hdr.get(5)
            if dph is None or 8 in hdr:
                raise ValueError(f"page at {pos}: a DATA_PAGE carries exactly a data_page_header")
            if dph[2] != 0:
                raise ValueError(f"page at {pos}: data page encoding is not PLAIN")
            if dph[3] not in (3, 4) or dph[4] not in (3, 4):
                raise ValueError(f"page at {pos}: level encoding not RLE/BIT_PACKED")
            n = dph[1]
        else:
            v2 = hdr.get(8)
            if v2 is None or 5 in hdr:
                raise ValueError(f"page at {pos}: a DATA_PAGE_V2 carries exactly a data_page_header_v2")
            if v2[4] != 0:
                raise ValueError(f"page at {pos}: data page v2 encoding is not PLAIN")
            if (codec == "UNCOMPRESSED" and v2.get(7, True)) or v2[2] != 0:
                raise ValueError(f"page at {pos}: data page v2 is compressed or has nulls")
            if not v2.get(7, True) and usize != csize:
                raise ValueError("uncompressed v2 page has unequal sizes")
            if v2[5] < 0 or v2[6] < 0 or v2[5] + v2[6] > csize:
                raise ValueError(f"page at {pos}: data page v2 level lengths exceed the page")
            n = v2[1]
        if n < 1:
            raise ValueError(f"page at {pos} holds {n} values; every page holds at least one")
        values += n
        if values > n_rows:
            raise ValueError(f"pages hold more than the {n_rows} values of the row group")
        pos += hlen + csize
    if pos != length:
        raise ValueError("pages do not tile the column chunk")
    if values != n_rows:
        raise ValueError(f"pages hold {values} values for {n_rows} rows")


def check_parquet_pages(chunk: bytes, n_rows: int, *, codec: str = "UNCOMPRESSED") -> None:
    """``walk_parquet_pages`` over an in-memory column chunk."""
    walk_parquet_pages(lambda off, n: bytes(chunk[off:off + n]), len(chunk), n_rows, codec=codec)


def parquet_footer_cap(n_mol: int) -> tuple[int, int]:
    """``(max row groups, max footer bytes)`` for ``n_mol`` rows (section 11
    step 1): at most ``ceil(n_mol / 2^14)`` row groups, and a footer of at
    most 64 KiB plus 1 KiB per row group (a legal footer is ~200 bytes plus
    ~280 per row group, measured with pyarrow 26)."""
    n_rg = max(1, -(-int(n_mol) // LIMITS["parquet_row_group"]))
    return n_rg, LIMITS["parquet_footer_base"] + n_rg * LIMITS["parquet_footer_per_row_group"]


def _scan_parquet(root: str, descriptor: dict, all_ids: np.ndarray, all_n_real: np.ndarray,
                  cd: Optional[CorpusDir] = None) -> None:
    """Section 16.3 item 7: bounded by construction. One descriptor (FIFO-safe),
    file size capped by ``n_mol`` and the footer length bounded before the
    footer is parsed, footer rules checked, every page header walked with
    ``pread`` (``walk_parquet_pages``), then each column of each row group
    read on its own with vectorized checks."""
    if cd is None:
        with CorpusDir(root) as own:
            return _scan_parquet(root, descriptor, all_ids, all_n_real, own)
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    path = os.path.join(root, "molecules.parquet")
    fd = cd.open("molecules.parquet")

    def refuse(msg: str, code: str = "value.parquet") -> CorpusValidationError:
        return _refuse(code, msg, tier="scan", path=path)

    try:
        n_mol = descriptor["n_mol"]
        size = os.fstat(fd).st_size
        cap = n_mol * LIMITS["parquet_row_bytes"] + LIMITS["parquet_overhead"]
        if size > cap:
            raise refuse(f"molecules.parquet is {size} bytes, cap {cap} for {n_mol} molecules")
        digest = hashlib.sha256()
        for off in range(0, size, 1 << 20):
            digest.update(_pread_exact(fd, min(1 << 20, size - off), off, path))
        if digest.hexdigest() != descriptor["molecules_sha256"]:
            raise refuse("molecules.parquet does not hash to corpus.json's molecules_sha256",
                         "integrity.molecules_hash")
        # Step 1: the footer length, bounded before any library parses it.
        max_rg, footer_cap = parquet_footer_cap(n_mol)
        if size < 12:
            raise refuse(f"molecules.parquet is {size} bytes, too short for a parquet file")
        head = _pread_exact(fd, 4, 0, path)
        tail = _pread_exact(fd, 8, size - 8, path)
        footer_len = struct.unpack("<I", tail[:4])[0]
        if head != b"PAR1" or tail[4:] != b"PAR1":
            raise refuse("molecules.parquet does not begin and end with PAR1")
        if footer_len > footer_cap or footer_len + 12 > size:
            raise refuse(f"footer of {footer_len} bytes exceeds {min(footer_cap, size - 12)} "
                         f"(64 KiB + 1 KiB x {max_rg} row groups, within the file)")
        from .parquet_footer import check_footer
        try:
            check_footer(_pread_exact(fd, footer_len, size - 8 - footer_len, path), n_mol=n_mol, max_row_groups=max_rg)
        except (ValueError, UnicodeError) as e:
            raise refuse(f"invalid parquet footer: {e}") from None
        with os.fdopen(os.dup(fd), "rb") as fh:
            try:
                pf = pq.ParquetFile(fh, thrift_string_size_limit=footer_cap,
                                    thrift_container_size_limit=max(1024, max_rg))
            except Exception as e:  # pyarrow raises its own hierarchy
                raise refuse(f"cannot read parquet footer: {e}") from None
            md = pf.metadata
            if md.metadata and b"ARROW:schema" in md.metadata:
                raise refuse("an embedded Arrow schema is not allowed")
            got = [(f.name, str(f.type), f.nullable) for f in pf.schema_arrow]
            if got != [(n, t, False) for n, t in PARQUET_COLUMNS]:
                raise refuse(f"schema {got} != {list(PARQUET_COLUMNS)} (all non-null)")
            for c in range(len(PARQUET_COLUMNS)):
                if pf.schema.column(c).max_definition_level != 0 or pf.schema.column(c).max_repetition_level != 0:
                    raise refuse(f"column {c} is not REQUIRED")
            if md.num_rows != n_mol:
                raise refuse(f"{md.num_rows} rows != n_mol {n_mol}")
            if md.num_row_groups > max_rg:
                raise refuse(f"{md.num_row_groups} row groups, at most ceil(n_mol / "
                             f"{LIMITS['parquet_row_group']}) = {max_rg}")
            rows_sum = 0
            for rg in range(md.num_row_groups):
                g = md.row_group(rg)
                rows_sum += g.num_rows
                if g.num_rows > LIMITS["parquet_row_group"]:
                    raise refuse(f"row group {rg} has {g.num_rows} rows, limit {LIMITS['parquet_row_group']}")
                for c in range(g.num_columns):
                    col = g.column(c)
                    name = PARQUET_COLUMNS[c][0]
                    if col.compression not in ("UNCOMPRESSED", "ZSTD") or not set(col.encodings) <= PARQUET_ENCODINGS:
                        raise refuse(f"row group {rg} column {name}: codec {col.compression}, "
                                     f"encodings {col.encodings}; only ZSTD/UNCOMPRESSED and PLAIN are allowed")
                    if col.has_dictionary_page or col.dictionary_page_offset is not None:
                        raise refuse(f"row group {rg} column {name} has a dictionary page")
                    if col.is_stats_set:
                        raise refuse(f"row group {rg} column {name} carries statistics")
                    cap_c = g.num_rows * PARQUET_COLUMN_BYTES[name] + 4096
                    if col.total_uncompressed_size > cap_c or col.total_compressed_size > cap_c:
                        raise refuse(f"row group {rg} column {name} is larger than its rows allow")
                    off, ln = col.data_page_offset, col.total_compressed_size
                    if off < 4 or ln < 0 or off + ln > size - 8 - footer_len:
                        raise refuse(f"row group {rg} column {name}: byte range outside the data")
                    try:
                        walk_parquet_pages(lambda o, n, _b=off: _pread_exact(fd, n, _b + o, path),
                                           ln, g.num_rows, codec=col.compression,
                                           uncompressed_cap=cap_c)
                    except ValueError as e:
                        raise refuse(f"row group {rg} column {name}: {e}") from None
            if rows_sum != md.num_rows:
                raise refuse("row groups do not sum to the footer's row count")
            seen = 0
            prev_last = None

            def column(rg: int, name: str):
                # one column chunk at a time: the largest is smiles, at most
                # 2^14 x 4,100 B + 4 KiB = 67 MB (limit 34)
                try:
                    result = pf.read_row_group(rg, columns=[name]).column(0)
                    if len(result) != md.row_group(rg).num_rows or result.null_count:
                        raise ValueError("column length/null count differs from REQUIRED row group")
                    if name in ("ext_id", "smiles"):
                        try:
                            result.validate(full=True)
                        except Exception:
                            raise refuse("invalid UTF-8", "limit." + name) from None
                    return result
                except CorpusValidationError:
                    raise
                except Exception as e:
                    raise refuse(f"cannot read row group {rg} column {name}: {e}") from None

            for rg in range(md.num_row_groups):
                ids = column(rg, "mol_id").to_numpy().astype(np.int64)
                if ids.size and (ids.min() < 0):       # u64 >= 2^63 reads as negative int64
                    raise refuse("a parquet mol_id is >= 2^63", "limit.mol_id")
                if ids.size and (np.any(np.diff(ids) <= 0)
                                 or (prev_last is not None and ids[0] <= prev_last)):
                    raise refuse("mol_id is not strictly ascending")
                if ids.size:
                    prev_last = int(ids[-1])
                if not np.array_equal(ids, all_ids[seen:seen + ids.size]):
                    raise refuse("mol_id set differs from the run tables")
                hac = column(rg, "hac").to_numpy().astype(np.int64)
                if not np.array_equal(hac, all_n_real[seen:seen + ids.size]):
                    raise refuse("a hac differs from the molecule's n_real", "limit.hac")
                seen += ids.size
                del ids, hac
                smiles = column(rg, "smiles")
                sl = pc.binary_length(smiles).to_numpy()
                if sl.size and (sl.max() > LIMITS["smiles"] or sl.min() < 1
                                or not pc.all(pc.match_substring_regex(smiles, _SMILES_OK.pattern)).as_py()):
                    raise refuse("a smiles is empty, too long, or holds a character outside printable "
                                 "ASCII without '|' or whitespace (CXSMILES)", "limit.smiles")
                del smiles, sl
                ext_id = column(rg, "ext_id")
                el = pc.binary_length(ext_id).to_numpy()
                if el.size and (el.max() > LIMITS["ext_id"] or el.min() < 1):
                    raise refuse("an ext_id is empty or too long", "limit.ext_id")
                bad = pc.match_substring_regex(          # RE2 syntax
                    ext_id, "[\\x00-\\x1f\\x7f-\\x9f\\x{200e}\\x{200f}\\x{202a}-\\x{202e}\\x{2066}-\\x{2069}]")
                if el.size and pc.any(bad).as_py():
                    raise refuse("an ext_id carries a control or bidi character", "limit.ext_id")
                del ext_id, el, bad
                mw = column(rg, "mw").to_numpy()
                if not np.all(np.isfinite(mw)) or np.any(mw < 0):
                    raise refuse("mw must be finite and >= 0")
                rotors = column(rg, "rotors").to_numpy()
                if np.any(rotors < 0) or np.any(rotors > 65535):
                    raise refuse("rotors outside uint16")
            if seen != all_ids.size:
                raise refuse("mol_id set differs from the run tables")
    finally:
        os.close(fd)


def _deep_group(gf: GroupFile, facts: dict) -> None:
    """Section 16.4 item 1 (RDKit-free): hashes, padding, tails."""
    path = gf.path
    for name, e in gf.sections.items():
        digest = hashlib.sha256()                  # chunked: never a whole section in memory
        for off in range(0, e["length"], 1 << 24):
            digest.update(gf.read(e["offset"] + off, min(1 << 24, e["length"] - off)))
        if digest.hexdigest() != e["sha256"]:
            raise _refuse("integrity.section_hash", f"section {name!r} hash mismatch", tier="deep", path=path)
        end = e["offset"] + e["length"]
        pad = gf.read(end, roundup64(end) - end)
        if any(pad):
            raise _refuse("integrity.padding", f"section {name!r}: alignment padding not zero",
                          tier="deep", path=path)
    blocks = gf.blocks()
    n_conf, n_real = facts["n_conf"], facts["n_real"]
    for i in range(blocks.size):
        bl = blocks[i]
        start = gf.sections["coords"]["offset"] + int(bl["offset"])
        length = int(bl["length"])
        digest = hashlib.sha256()                  # chunked: one block never needs to be resident
        for off in range(0, length, 1 << 24):
            digest.update(gf.read(start + off, min(1 << 24, length - off)))
        if digest.digest() != bytes(bl["sha256"]):
            raise _refuse("integrity.block_hash", f"block {i} hash mismatch", tier="deep", path=path)
        m0, m1 = int(bl["m0"]), int(bl["m1"])
        total = int((n_conf[m0:m1] * n_real[m0:m1]).sum())
        bh = parse_block_header(gf.read(start, BLOCK_HEADER_SIZE))
        groups = -(-total // 256)
        pos = start + BLOCK_HEADER_SIZE
        for a in range(3):
            w = bh.widths[a]
            # tail positions (5.4) exist only in the last packing group: read
            # and unpack just its w * 32 bytes
            last = gf.read(pos + (groups - 1) * w * 32, w * 32)
            u = unpack_axis_raw_vectorized(last, w, total - (groups - 1) * 256)
            pos += groups * w * 32
            if np.any(u[total - (groups - 1) * 256:]):
                raise _refuse("integrity.padding", f"block {i}: tail positions not zero", tier="deep", path=path)


def deep_rdkit_checks(root: str, descriptor: dict, *, sample: Optional[int] = None) -> None:
    """Section 16.4 items 3-4: SMILES parse and feature re-perception. They
    need RDKit and belong to the production validator; the reference leaves them unimplemented."""
    raise NotImplementedError("the production validator implements the RDKit half of the deep tier")


@dataclass
class ValidationReport:
    tier: str
    fingerprint: str
    descriptor: dict
    groups: list = field(default_factory=list)


def validate_corpus(root: str, tier: str = "scan") -> ValidationReport:
    """Run the ``open``, ``scan`` or ``deep`` tier over a corpus directory
    (sections 16.2-16.4; ``deep`` without the RDKit items)."""
    if tier not in ("open", "scan", "deep"):
        raise ValueError(f"tier must be open, scan or deep, got {tier!r}")
    try:
        return _validate(root, tier)
    except CorpusValidationError:
        raise
    except Exception as e:  # a validator bug is still a refusal, never a crash
        raise CorpusValidationError("internal.error", f"unexpected {type(e).__name__}: {e}",
                                    tier=tier, path=root) from e


def _validate(root: str, tier: str) -> ValidationReport:
    with CorpusDir(root) as cd:
        return _validate_dir(root, cd, tier)


def _validate_dir(root: str, cd: CorpusDir, tier: str) -> ValidationReport:
    descriptor, fp = read_descriptor(root, cd)
    report = ValidationReport(tier, fp, descriptor)
    ids_parts: list[np.ndarray] = []
    nreal_parts: list[np.ndarray] = []
    lfw_declared: dict = {}
    lfw_actual: dict = {}
    for entry in descriptor["groups"]:
        path = os.path.join(root, group_file_name(entry["bucket"], entry["group"]))
        gf = open_group(path, descriptor=descriptor, group_entry=entry, fingerprint_hex=fp,
                        corpus_dir=cd)
        try:
            if tier in ("scan", "deep"):
                facts = _scan_group(gf, descriptor)
                ids_parts.append(facts["mol_id"].astype(np.int64))
                nreal_parts.append(facts["n_real"])
                b = entry["bucket"]
                lfw_declared.setdefault(b, set()).add(entry["L_field_width"])
                lfw_actual[b] = max(lfw_actual.get(b, 0), gf.header["L_heavy"])
                if tier == "deep":
                    _deep_group(gf, facts)
            report.groups.append(gf.header)
        finally:
            gf.close()
    if tier in ("scan", "deep"):
        for b, declared in lfw_declared.items():
            if declared != {lfw_actual[b]}:
                raise _refuse("layout.widths", f"bucket {b}: L_field_width {sorted(declared)} != bucket-wide "
                              f"{lfw_actual[b]}", tier="scan", path=root)
        ids = np.concatenate(ids_parts) if ids_parts else np.zeros(0, np.int64)
        n_real = np.concatenate(nreal_parts) if nreal_parts else np.zeros(0, np.int64)
        order = np.argsort(ids, kind="stable")
        ids, n_real = ids[order], n_real[order]
        if ids.size and np.any(np.diff(ids) == 0):
            raise _refuse("layout.runs", "a mol_id appears in more than one group", tier="scan", path=root)
        if "color" in descriptor:
            check_definition_text(descriptor["color"]["definition"],
                                  descriptor["color"]["generator"], descriptor["color"]["types"])
        _scan_parquet(root, descriptor, ids, n_real, cd)
        if tier == "deep":
            if ids_sha256(ids) != descriptor["ids_sha256"]:
                raise _refuse("integrity.ids_hash", "ids_sha256 mismatch", tier="deep", path=root)
            bpath = os.path.join(root, "build.json")
            if cd.exists("build.json"):
                fd = cd.open("build.json")
                try:
                    size = os.fstat(fd).st_size
                    if size > LIMITS["json_bytes"]:
                        raise _refuse("json.size", f"build.json is {size} bytes", tier="deep", path=bpath)
                    data = _pread_exact(fd, size, 0, bpath)
                finally:
                    os.close(fd)
                loads_strict(data, what="build.json", path=bpath)
    return report


# ---------------------------------------------------------------------------
# Convenience: decode a whole group to the kernel layout
# ---------------------------------------------------------------------------

def decode_group(gf: GroupFile, descriptor: Optional[dict] = None) -> dict:
    """Decode every block of an opened group: heavy coordinates in the padded
    disk layout, the merged kernel layout, derived pseudo-atoms, counts.
    Used by the conformance digests; not a production path."""
    h = gf.header
    runs = gf.section_array("runs")
    n_conf = runs["n_conf"].astype(np.int64)
    n_real = runs["n_real"].astype(np.int64)
    n_types = h["n_types"]
    feats = None
    if n_types:
        feats = {name: gf.section_array(name) for name in
                 ("feature_offsets", "feature_types", "feature_atom_offsets", "feature_atoms")}
    blocks = gf.blocks()
    heavy = np.zeros((h["n_conf"], 3, h["L_heavy"]), dtype=np.int16)
    merged = np.zeros((h["n_conf"], 3, h["L"]), dtype=np.int16)
    counts = np.zeros((h["n_conf"], n_types + 1), dtype=np.uint8)
    pseudo: list[np.ndarray] = []
    row_mol = np.repeat(np.arange(runs.size), n_conf)
    for i in range(blocks.size):
        bl = blocks[i]
        m0, m1 = int(bl["m0"]), int(bl["m1"])
        nr = np.repeat(n_real[m0:m1], n_conf[m0:m1])
        xyz = decode_block(gf.block_bytes(i), nr, h["L_heavy"])
        r0 = int(bl["first_row"])
        heavy[r0:r0 + xyz.shape[0]] = xyz
    for row in range(h["n_conf"]):
        m = int(row_mol[row])
        fl = molecule_features(feats, m) if feats is not None else []
        merged[row] = kernel_layout(heavy[row], int(n_real[m]), fl, h["L"])
        counts[row] = derived_counts(int(n_real[m]), fl, n_types)
        pseudo.append(derive_positions(heavy[row], fl))
    return {"heavy": heavy, "merged": merged, "counts": counts,
            "pseudo": np.concatenate(pseudo) if pseudo else np.zeros((0, 3), np.int16),
            "runs": runs, "features": feats, "vaa": gf.section_array("vaa"),
            "color_self": gf.section_array("color_self") if "color_self" in gf.sections else None}

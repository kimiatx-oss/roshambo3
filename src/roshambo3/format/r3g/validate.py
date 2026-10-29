"""The format-3 validator: ``open``, ``scan`` and ``deep`` (``doc/FORMAT.md`` 16).

Pure Python and NumPy; runs before any native code sees a byte. Every refusal
is a :class:`CorpusValidationError` with a section-16.5 code, the tier and the
path, and a message naming the file, the field and the limit; no other
exception type escapes :func:`validate_corpus` on any input.

The checks and their **order** follow the pure-Python reference
(``reference.validate_corpus``, the behavioral oracle for ``open`` and
``scan``), so a corpus wrong in several ways is refused with the same code.
What differs is how they run:

* every read is a bounded ``pread`` through :mod:`.fileio` (opened with
  directory descriptors and ``O_NOFOLLOW``), never an ``mmap``;
* the per-block loops of the reference (the ``open`` block-table walk and
  the ``scan`` 32-byte block-header loop) are vectorized, so the 2^20-block
  limit costs NumPy work, not 2^20 Python iterations;
* per-feature checks run in bounded chunks (limit 34);
* the Parquet page walk is ``reference.walk_parquet_pages`` (the closed-schema
  walker section 11 step 3 names as the definition; called, like the
  definition pre-pass, not re-written), fed by ``pread``;
* limit 4 counts escapes as the spec states (:mod:`.jsonstrict`).

The ``deep`` tier's RDKit items run in a subprocess (:mod:`._sandbox`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import math
import os
import time
from typing import Any, Optional

import numpy as np

from . import codec, jsonstrict, reference
from .errors import CorpusValidationError
from .fileio import CorpusFile, exists_in_corpus, open_corpus_file
from .limits import (INT64_MAX, BLOCK_DTYPE, BLOCK_HEADER_SIZE, BLOCK_RECORD_SIZE, BUCKET_EDGES,
                     CONVENTIONS, DEFAULT_BOUNDS, DTYPES, FIRST_SECTION_OFFSET,
                     FORMAT_OBJECT, FRAME, GENERATORS, LIMITS, L_ALIGN, MAGIC, MAJOR,
                     N_BUCKETS, OPTIONAL_SECTIONS, POSITIONING, PROLOGUE_SIZE, RUN_DTYPE,
                     RUN_RECORD_SIZE, SECTION_DTYPE, Bounds, expected_sections,
                     group_file_name, pad8, roundup64)
from .parquet import scan_molecules

TIERS = ("open", "scan", "deep")
#: Bumped whenever a check changes, so a cached ``scan`` from an older
#: validator is not trusted (:mod:`.cache` keys on it).
VALIDATOR_VERSION = 1
#: ``deep``: molecules re-perceived (and self-overlaps recomputed) without
#: ``--all``, drawn with a seed from ``os.urandom`` (section 16.4 item 4).
DEEP_SAMPLE_MOLECULES = 256
#: ``deep`` with the native decoder: blocks additionally decoded by the
#: reference and compared bit for bit, **per corpus** (section 16.4 item 1),
#: not per group: a per-group count grows with the group count and not the
#: assurance. Spread over the groups in proportion to their rows.
DEEP_REFERENCE_SAMPLE_BLOCKS = 32
#: ``deep`` without ``--all``: sampled molecules whose ``color_self`` is
#: recomputed by the Python reference as well as the native path, which it
#: cross-checks (section 16.4 item 4). ``--all``: every molecule.
DEEP_REFERENCE_SELF_OVERLAP_MOLECULES = 16
#: ``deep``: relative tolerance for recomputed self-overlaps (section 18).
SELF_OVERLAP_RTOL = 1e-5
#: Bytes of the ``coords`` section read at a time when gathering block headers.
_COORDS_CHUNK = 64 << 20


def _refuse(code: str, message: str, *, tier: str = "open",
            path: Optional[str] = None) -> CorpusValidationError:
    return CorpusValidationError(code, message, tier=tier, path=path)


# ---------------------------------------------------------------------------
# Schema helpers (as the reference's, so codes agree)
# ---------------------------------------------------------------------------

def _require_keys(obj: dict, required, optional=(), *, what: str, path: str,
                  tier: str = "open") -> None:
    required = list(required)
    allowed = set(required) | set(optional)
    for k in obj:
        if k not in allowed:
            raise _refuse("schema.unknown_key", f"{what}: unknown key {k!r}", path=path, tier=tier)
    for k in required:
        if k not in obj:
            raise _refuse("schema.missing_key", f"{what}: missing key {k!r}", path=path, tier=tier)


def _int(obj: dict, key: str, *, what: str, path: str, lo: int = 0,
         hi: Optional[int] = None, limit: Optional[str] = None) -> int:
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise _refuse("schema.type", f"{what}.{key} must be an integer", path=path)
    if v < lo:
        raise _refuse("schema.value", f"{what}.{key} = {v} is below {lo}", path=path)
    if v > INT64_MAX:
        # section 17 item 11 (S8): a native reader would wrap; every reader refuses
        raise _refuse("schema.value", f"{what}.{key} = {v} does not fit a signed 64-bit "
                      f"integer", path=path)
    if hi is not None and v > hi:
        raise _refuse(f"limit.{limit}" if limit else "schema.value",
                      f"{what}.{key} = {v} exceeds {hi}" + (f" (limit.{limit})" if limit else ""),
                      path=path)
    return v


def _hex64(obj: dict, key: str, *, what: str, path: str) -> str:
    v = obj[key]
    if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        raise _refuse("schema.type", f"{what}.{key} must be 64 lowercase hex characters", path=path)
    return v


def corpus_fingerprint(descriptor: dict) -> str:
    """The semantic fingerprint (section 10.5). A dedicated module's
    ``fingerprint`` is used when present, the reference's otherwise."""
    if importlib.util.find_spec(__package__ + ".descriptor") is not None:   # pragma: no cover
        from .descriptor import fingerprint  # type: ignore[import-not-found]
        return fingerprint(descriptor)
    return reference.fingerprint(descriptor)


def canonical_matrices(color: dict) -> bool:
    n = len(color["types"])
    return reference.canonical_matrices(color["r"]["values"], color["p"]["values"], n)


# ---------------------------------------------------------------------------
# open: corpus.json (section 16.2, last paragraph)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Descriptor:
    data: dict
    fingerprint: str
    sha256: str          # of the corpus.json bytes as stored
    path: str


def read_descriptor(root: str) -> Descriptor:
    """``open`` for ``corpus.json``: strict JSON, closed schema, every value."""
    f = open_corpus_file(root, "corpus.json")
    path = f.path
    try:
        if f.size > LIMITS["json_bytes"]:
            raise _refuse("json.size", f"corpus.json is {f.size} bytes, limit "
                          f"{LIMITS['json_bytes']} (limit 1)", path=path)
        raw = f.read(0, f.size)
    finally:
        f.close()
    d = jsonstrict.loads(raw, what="corpus.json", path=path)
    _require_keys(d, ("format", "frame", "prep", "groups", "n_conf", "n_mol", "ids_sha256",
                      "molecules_sha256"), ("color",), what="corpus.json", path=path)
    for key, want in (("format", FORMAT_OBJECT), ("frame", FRAME)):
        obj = d[key]
        if not isinstance(obj, dict):
            raise _refuse("schema.type", f"{key} must be an object", path=path)
        _require_keys(obj, want.keys(), (), what=key, path=path)
        for k, v in want.items():
            if obj[k] != v or type(obj[k]) is not type(v):
                code = ("prologue.major_version" if k == "major"
                        else "schema.type" if isinstance(v, str) else "schema.value")
                raise _refuse(code, f"{key}.{k} = {obj[k]!r}; this reader requires {v!r}", path=path)
    prep = d["prep"]
    if not isinstance(prep, dict):
        raise _refuse("schema.type", "prep must be an object", path=path)
    _require_keys(prep, ("version", "commit", "conventions"), (), what="prep", path=path)
    if not isinstance(prep["version"], str) or not (prep["commit"] is None
                                                      or isinstance(prep["commit"], str)):
        raise _refuse("schema.type", "prep.version must be a string, prep.commit a string or "
                      "null", path=path)
    conv = prep["conventions"]
    if not isinstance(conv, dict):
        raise _refuse("schema.type", "prep.conventions must be an object", path=path)
    _require_keys(conv, CONVENTIONS.keys(), (), what="prep.conventions", path=path)
    for k, v in CONVENTIONS.items():
        if conv[k] != v:
            raise _refuse("schema.type", f"prep.conventions.{k} = {conv[k]!r}; requires {v!r}",
                          path=path)
    if "color" in d:
        _check_color(d["color"], path)
    groups = d["groups"]
    if not isinstance(groups, list) or not groups:
        raise _refuse("schema.type", "groups must be a non-empty list", path=path)
    if len(groups) > LIMITS["groups"]:
        raise _refuse("limit.groups", f"{len(groups)} groups, limit {LIMITS['groups']} (limit 5)",
                      path=path)
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
            raise _refuse("schema.value", f"{what}: groups out of order or with a gap at "
                          f"(bucket {b}, group {gi})", path=path)
        last = key
        _int(g, "L_heavy", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L_heavy"], limit="L_heavy")
        _int(g, "L", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L"], limit="L")
        _int(g, "L_field_width", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L_field_width"],
             limit="L_field_width")
        tot_conf += _int(g, "n_conf", what=what, path=path, lo=1, hi=LIMITS["n_conf"],
                         limit="n_conf")
        tot_mol += _int(g, "n_mol", what=what, path=path, lo=1)
        _hex64(g, "header_sha256", what=what, path=path)
    if _int(d, "n_conf", what="corpus.json", path=path) != tot_conf:
        raise _refuse("schema.value", f"n_conf {d['n_conf']} != the group table's sum {tot_conf}",
                      path=path)
    if _int(d, "n_mol", what="corpus.json", path=path) != tot_mol:
        raise _refuse("schema.value", f"n_mol {d['n_mol']} != the group table's sum {tot_mol}",
                      path=path)
    _hex64(d, "ids_sha256", what="corpus.json", path=path)
    _hex64(d, "molecules_sha256", what="corpus.json", path=path)
    return Descriptor(d, corpus_fingerprint(d), hashlib.sha256(raw).hexdigest(), path)


def _check_color(c: Any, path: str) -> None:
    if not isinstance(c, dict):
        raise _refuse("schema.type", "color must be an object", path=path)
    _require_keys(c, ("generator", "definition", "types", "positioning", "r", "p",
                      "rdkit_version"), (), what="color", path=path)
    if c["generator"] not in GENERATORS:
        raise _refuse("schema.type", f"color.generator {c['generator']!r} is not one of "
                      f"{GENERATORS}", path=path)
    if c["positioning"] not in POSITIONING:
        raise _refuse("schema.type", f"color.positioning {c['positioning']!r} is not one of "
                      f"{POSITIONING}", path=path)
    if not isinstance(c["rdkit_version"], str):
        raise _refuse("schema.type", "color.rdkit_version must be a string", path=path)
    want = str if c["generator"] == "rdkit_fdef" else dict
    if not isinstance(c["definition"], want):
        raise _refuse("schema.type", f"color.definition must be a JSON "
                      f"{'string' if want is str else 'object'} for generator "
                      f"{c['generator']}", path=path)
    types = c["types"]
    if (not isinstance(types, list) or not types or any(not isinstance(t, str) for t in types)
            or len(set(types)) != len(types)):
        raise _refuse("schema.type", "color.types must be a non-empty list of unique strings",
                      path=path)
    n_types = len(types)
    if n_types > LIMITS["n_types"]:
        raise _refuse("limit.n_types", f"color.types has {n_types} entries, limit "
                      f"{LIMITS['n_types']} (limit 14)", path=path)
    for key in ("r", "p"):
        m = c[key]
        if not isinstance(m, dict):
            raise _refuse("schema.type", f"color.{key} must be an object", path=path)
        _require_keys(m, ("dtype", "shape", "values"), (), what=f"color.{key}", path=path)
        if m["dtype"] != "f32" or m["shape"] != [n_types + 1, n_types + 1]:
            raise _refuse("value.matrix", f"color.{key}: dtype/shape must be f32 "
                          f"[{n_types + 1}, {n_types + 1}]", path=path)
        vals = m["values"]
        if (not isinstance(vals, list) or len(vals) != n_types + 1
                or any(not isinstance(rw, list) or len(rw) != n_types + 1 for rw in vals)):
            raise _refuse("value.matrix", f"color.{key}.values must be {n_types + 1} x "
                          f"{n_types + 1}", path=path)
        for rw in vals:
            for x in rw:
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                    raise _refuse("value.matrix", f"color.{key}: a non-finite or non-numeric value",
                                  path=path)
                if float(np.float32(x)) != float(x):
                    raise _refuse("value.matrix", f"color.{key}: {x!r} is not exactly a float32",
                                  path=path)
                if key == "r" and x < 0:
                    raise _refuse("value.matrix", f"color.r: negative width {x!r}", path=path)
    problems = reference.interaction_matrix_problems(c["r"]["values"], c["p"]["values"])
    if problems:
        raise _refuse("value.matrix", "color matrices: " + "; ".join(problems), path=path)


# ---------------------------------------------------------------------------
# open: one group file (section 16.2 items 1-5)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class OpenGroup:
    """A group file that passed ``open``: its descriptor, header and tables."""
    file: CorpusFile
    path: str
    entry: Optional[dict]
    header: dict
    header_bytes: bytes
    header_offset: int
    sections: dict           # name -> section-table entry
    blocks: np.ndarray       # BLOCK_DTYPE, read and walked at open

    def read_section(self, name: str, *, tier: str = "scan") -> bytes:
        """The section's ``length`` data bytes (``pread``)."""
        e = self.sections[name]
        return self.file.read(e["offset"], e["length"], tier=tier)

    def section_array(self, name: str, *, tier: str = "scan") -> np.ndarray:
        e = self.sections[name]
        dt = (RUN_DTYPE if name == "runs" else BLOCK_DTYPE if name == "blocks"
              else DTYPES[e["dtype"]])
        return np.frombuffer(self.read_section(name, tier=tier), dtype=dt)

    def close(self) -> None:
        self.file.close()


def open_group(root: str, entry: Optional[dict], *, descriptor: Optional[dict] = None,
               fingerprint: Optional[str] = None, bucket: Optional[int] = None,
               group: Optional[int] = None) -> OpenGroup:
    """``open`` for one group file, located from its group-table entry (or from
    ``bucket``/``group`` for a standalone file)."""
    b = entry["bucket"] if entry is not None else bucket
    g = entry["group"] if entry is not None else group
    f = open_corpus_file(root, "groups", group_file_name(b, g))
    try:
        return _open_group_file(f, entry, descriptor, fingerprint)
    except BaseException:
        f.close()
        raise


def _open_group_file(f: CorpusFile, entry, descriptor, fingerprint) -> OpenGroup:
    path, size = f.path, f.size
    if size < FIRST_SECTION_OFFSET:
        raise _refuse("prologue.magic", f"file is {size} bytes, shorter than the 64-byte "
                      f"prologue", path=path)
    pro = f.read(0, FIRST_SECTION_OFFSET)
    if pro[:4] != MAGIC:
        raise _refuse("prologue.magic", f"magic {pro[:4]!r} is not {MAGIC!r}", path=path)
    major = int.from_bytes(pro[4:8], "little")
    header_offset = int.from_bytes(pro[8:16], "little")
    if major != MAJOR:
        raise _refuse("prologue.major_version", f"major version {major}; this reader "
                      f"implements {MAJOR} only", path=path)
    if not FIRST_SECTION_OFFSET <= header_offset <= size:
        raise _refuse("prologue.header_offset", f"header_offset {header_offset} outside "
                      f"[64, {size}]", path=path)
    if any(pro[PROLOGUE_SIZE:]):
        raise _refuse("prologue.reserved", "reserved prologue bytes 16-63 are not zero", path=path)
    header_len = size - header_offset
    if header_len > LIMITS["json_bytes"]:
        raise _refuse("json.size", f"header is {header_len} bytes, limit {LIMITS['json_bytes']} "
                      f"(limit 2)", path=path)
    header_bytes = f.read(header_offset, header_len)            # bounded read
    header = jsonstrict.loads(header_bytes, what="group header", path=path)
    # the header hash binds everything below to corpus.json: checked before
    # any header value is used (section 16.2 item 1)
    if entry is not None and hashlib.sha256(header_bytes).hexdigest() != entry["header_sha256"]:
        raise _refuse("integrity.header_hash", "header bytes do not hash to the group table's "
                      "header_sha256", path=path)
    what = "header"
    _require_keys(header, ("major", "fingerprint", "bucket", "group", "n_conf", "n_mol",
                           "n_types", "L_heavy", "L", "sections"), (), what=what, path=path)
    try:
        canonical = jsonstrict.dumps(header) == header_bytes
    except (ValueError, UnicodeEncodeError):
        canonical = False
    if not canonical:
        raise _refuse("schema.value", "header is not canonical JSON (section 10.5.1)", path=path)
    if _int(header, "major", what=what, path=path) != MAJOR:
        raise _refuse("prologue.major_version", f"header major {header['major']} != {MAJOR}",
                      path=path)
    fp = _hex64(header, "fingerprint", what=what, path=path)
    _int(header, "bucket", what=what, path=path, hi=N_BUCKETS - 1)
    _int(header, "group", what=what, path=path, hi=LIMITS["groups"] - 1, limit="groups")
    n_conf = _int(header, "n_conf", what=what, path=path, lo=1, hi=LIMITS["n_conf"], limit="n_conf")
    n_mol = _int(header, "n_mol", what=what, path=path, lo=1, hi=n_conf, limit="n_mol")
    n_types = _int(header, "n_types", what=what, path=path, hi=LIMITS["n_types"], limit="n_types")
    L_heavy = _int(header, "L_heavy", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L_heavy"],
                   limit="L_heavy")
    L = _int(header, "L", what=what, path=path, lo=L_ALIGN, hi=LIMITS["L"], limit="L")
    for v, nm in ((L_heavy, "L_heavy"), (L, "L")):
        if v % L_ALIGN:
            raise _refuse("schema.value", f"header.{nm} = {v} is not a multiple of 8", path=path)
    if L < L_heavy or (n_types == 0 and L != L_heavy):
        raise _refuse("layout.widths", f"header L = {L} inconsistent with L_heavy = {L_heavy}",
                      path=path)
    sections = _check_section_table(header, header_offset, n_types, n_conf, n_mol, path)
    # the blocks section's hash, then its walk (section 16.2 item 4)
    e = sections["blocks"]
    raw = f.read(e["offset"], e["length"])
    if hashlib.sha256(raw).hexdigest() != e["sha256"]:
        raise _refuse("integrity.section_hash", "the blocks section does not hash to the "
                      "header's sha256", path=path)
    blocks = np.frombuffer(raw, dtype=BLOCK_DTYPE)
    _walk_blocks(blocks, n_conf, n_mol, sections["coords"]["length"], path)
    # binding to the descriptor (16.2 item 5)
    if fingerprint is not None and fp != fingerprint:
        raise _refuse("integrity.fingerprint", f"header fingerprint {fp[:12]}... != corpus "
                      f"fingerprint {fingerprint[:12]}...", path=path)
    if descriptor is not None:
        color = descriptor.get("color")
        want_types = len(color["types"]) if color else 0
        if n_types != want_types:
            raise _refuse("schema.value", f"header.n_types {n_types} != the descriptor's "
                          f"{want_types}", path=path)
        if want_types and "color_self" in sections and not canonical_matrices(color):
            raise _refuse("layout.sections", "a color_self section is present but the "
                          "interaction matrices are not canonical (section 18.2)", path=path)
    if entry is not None:
        for k in ("bucket", "group", "n_conf", "n_mol", "L_heavy", "L"):
            if header[k] != entry[k]:
                raise _refuse("schema.value", f"header.{k} = {header[k]} != the group table's "
                              f"{entry[k]}", path=path)
    return OpenGroup(f, path, entry, header, header_bytes, header_offset, sections, blocks)


def _check_section_table(header: dict, header_offset: int, n_types: int, n_conf: int,
                         n_mol: int, path: str) -> dict:
    """Section 16.2 item 3: names, dtypes, shapes, lengths, alignment, tiling."""
    table = header["sections"]
    if not isinstance(table, list):
        raise _refuse("schema.type", "header.sections must be a list", path=path)
    sections: dict = {}
    names_seen: list = []
    offset = FIRST_SECTION_OFFSET
    for i, e in enumerate(table):
        if not isinstance(e, dict):
            raise _refuse("schema.type", f"sections[{i}] must be an object", path=path)
        _require_keys(e, ("name", "offset", "length", "dtype", "shape", "sha256"), (),
                      what=f"sections[{i}]", path=path)
        name = e["name"]
        if not isinstance(name, str) or name not in SECTION_DTYPE:
            raise _refuse("schema.type", f"sections[{i}]: unknown section name {name!r}", path=path)
        if name in sections:
            raise _refuse("layout.sections", f"section {name!r} is listed twice", path=path)
        if not isinstance(e["dtype"], str) or e["dtype"] not in DTYPES:
            raise _refuse("schema.type", f"section {name!r}: unknown dtype {e['dtype']!r}", path=path)
        off = _int(e, "offset", what=name, path=path)
        ln = _int(e, "length", what=name, path=path)
        shape = e["shape"]
        if (not isinstance(shape, list) or not shape
                or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= INT64_MAX
                       for s in shape)):
            raise _refuse("schema.type", f"section {name!r}: shape must be a list of non-negative "
                          f"integers", path=path)
        _hex64(e, "sha256", what=name, path=path)
        if e["dtype"] != SECTION_DTYPE[name]:
            raise _refuse("layout.sections", f"section {name!r}: dtype {e['dtype']} != "
                          f"{SECTION_DTYPE[name]}", path=path)
        if len(shape) != (2 if name in ("blocks", "runs") else 1):
            raise _refuse("layout.sections", f"section {name!r}: invalid shape rank", path=path)
        prod = 1
        for s in shape:
            prod *= s                                   # Python ints: no overflow
        if ln != prod * DTYPES[e["dtype"]].itemsize:
            raise _refuse("layout.sections", f"section {name!r}: length {ln} != prod(shape) x "
                          f"itemsize", path=path)
        if off % 64:
            raise _refuse("layout.sections", f"section {name!r}: offset {off} is not 64-aligned",
                          path=path)
        if off != offset:
            raise _refuse("layout.sections", f"section {name!r}: offset {off}, expected {offset} "
                          f"(sections must tile [64, header_offset) by extent)", path=path)
        if off + ln > header_offset:
            raise _refuse("layout.sections", f"section {name!r} runs into the header at "
                          f"{header_offset}", path=path)
        offset = roundup64(off + ln)
        sections[name] = e
        names_seen.append(name)
    if offset != header_offset:
        raise _refuse("layout.sections", f"sections end at {offset}, the header starts at "
                      f"{header_offset}", path=path)
    want = expected_sections(n_types)
    want = [n for n in want if n not in OPTIONAL_SECTIONS or n in sections]
    if names_seen != want:
        raise _refuse("layout.sections", f"sections {names_seen} != expected {want}", path=path)

    def shape_is(name, expect):
        if sections[name]["shape"] != expect:
            raise _refuse("layout.sections", f"section {name!r}: shape {sections[name]['shape']} "
                          f"!= {expect}", path=path)

    n_blocks = sections["blocks"]["shape"][0]
    if n_blocks < 1 or n_blocks > LIMITS["blocks"]:
        raise _refuse("limit.blocks", f"{n_blocks} blocks, limit 1..{LIMITS['blocks']} (limit 6)",
                      path=path)
    shape_is("blocks", [n_blocks, BLOCK_RECORD_SIZE])
    shape_is("runs", [n_mol, RUN_RECORD_SIZE])
    shape_is("coords", [sections["coords"]["length"]])
    shape_is("vaa", [n_conf])
    if n_types:
        shape_is("feature_offsets", [n_mol + 1])
        n_feat = sections["feature_types"]["shape"][0]
        n_atoms = sections["feature_atoms"]["shape"][0]
        if max(n_feat, n_atoms) > LIMITS["n_feat_atoms"]:
            raise _refuse("limit.n_feat_atoms", f"{n_feat} features / {n_atoms} feature atoms, "
                          f"limit {LIMITS['n_feat_atoms']} (limit 20)", path=path)
        shape_is("feature_types", [n_feat])
        shape_is("feature_atom_offsets", [n_feat + 1])
        shape_is("feature_atoms", [n_atoms])
        if "color_self" in sections:
            shape_is("color_self", [n_conf])
    return sections


def _walk_blocks(blocks: np.ndarray, n_conf: int, n_mol: int, coords_len: int, path: str) -> None:
    """The block table tiles rows, molecules and payload bytes; the row cap
    (section 6 rules 1-4, 16.2 item 4). Vectorized; for the first block that
    fails, the code is the one the reference's per-block loop would give."""
    n = blocks.size
    rows = blocks["rows"].astype(np.int64)
    first = blocks["first_row"].astype(np.int64)
    m0 = blocks["m0"].astype(np.int64)
    m1 = blocks["m1"].astype(np.int64)
    off_u = blocks["offset"]
    len_u = blocks["length"]
    # u64 fields compared in a range where int64 cannot overflow: anything
    # beyond the coords section cannot tile and is clamped just past it
    cap = coords_len + 64
    off = np.minimum(off_u, np.uint64(cap)).astype(np.int64)
    ln = np.minimum(len_u, np.uint64(cap)).astype(np.int64)
    exp_row = np.concatenate([[0], np.cumsum(rows)[:-1]])
    exp_m = np.concatenate([[0], m1[:-1]])
    exp_off = np.concatenate([[0], np.cumsum(ln)[:-1]])
    tile_bad = (first != exp_row) | (m0 != exp_m) | (off != exp_off) | (off_u > np.uint64(cap))
    empty = rows < 1
    no_mol = m1 <= m0
    single = (m1 - m0) == 1
    cap_bad = np.where(single, rows > LIMITS["block_rows_single_molecule"],
                       rows > LIMITS["block_rows"])
    len_bad = (len_u < np.uint64(BLOCK_HEADER_SIZE)) | (len_u % np.uint64(32) != 0)
    bad = tile_bad | empty | no_mol | cap_bad | len_bad
    if bad.any():
        i = int(np.argmax(bad))
        b = blocks[i]
        if tile_bad[i]:
            raise _refuse("layout.blocks", f"block {i} does not tile: first_row "
                          f"{int(b['first_row'])} (expected {int(exp_row[i])}), m0 {int(m0[i])} "
                          f"(expected {int(exp_m[i])}), offset {int(b['offset'])} (expected "
                          f"{int(exp_off[i])})", path=path)
        if empty[i]:
            raise _refuse("layout.blocks", f"block {i} is empty", path=path)
        if no_mol[i]:
            raise _refuse("layout.blocks", f"block {i}: m1 {int(m1[i])} <= m0 {int(m0[i])}",
                          path=path)
        if cap_bad[i]:
            lim = LIMITS["block_rows_single_molecule" if single[i] else "block_rows"]
            raise _refuse("limit.block_rows", f"block {i}: {int(rows[i])} rows for "
                          f"{int(m1[i] - m0[i])} molecule(s), limit {lim} (limit 18)", path=path)
        raise _refuse("layout.blocks", f"block {i}: length {int(b['length'])} is not a valid "
                      f"block length", path=path)
    end_row = int(rows.sum())
    end_m = int(m1[-1])
    end_off = int(off[-1]) + int(ln[-1])
    if end_row != n_conf or end_m != n_mol or end_off != coords_len:
        raise _refuse("layout.blocks", f"blocks cover {end_row} rows / {end_m} molecules / "
                      f"{end_off} bytes; the header says {n_conf} / {n_mol} / {coords_len}",
                      path=path)


# ---------------------------------------------------------------------------
# scan: one group (section 16.3 items 1-5)
# ---------------------------------------------------------------------------

#: Features per chunk of the per-feature checks (limit 34; amendment S9).
_FEATURE_CHUNK = 1 << 16


def scan_group(og: OpenGroup) -> dict:
    """Section 16.3 for one group, vectorized and chunked. Returns the facts
    the corpus-level checks need (``mol_id``, ``n_real``, ``n_conf``)."""
    path = og.path
    h = og.header
    n_types = h["n_types"]

    def refuse(code, msg):
        return _refuse(code, msg, tier="scan", path=path)

    runs = og.section_array("runs")
    n_mol = int(runs.size)
    if np.any(runs["reserved"]):
        raise refuse("layout.runs", "reserved run-table bytes 11-15 are not zero")
    n_real = runs["n_real"].astype(np.int64)
    n_conf = runs["n_conf"].astype(np.int64)
    mol_id = runs["mol_id"].astype(np.uint64)
    del runs
    if n_real.min() < 1:
        raise refuse("limit.n_real", "a molecule has n_real = 0, limit 1..255 (limit 9)")
    if n_conf.min() < 1 or n_conf.max() > LIMITS["n_conf_per_molecule"]:
        raise refuse("limit.n_conf_per_molecule", "a molecule has n_conf outside [1, 4096], limit 1..4096 "
                     "(limit 10)")
    if int(mol_id.max()) >= LIMITS["mol_id"]:
        raise refuse("limit.mol_id", "a run-table mol_id is >= 2^63 (limit 35)")
    ids = mol_id.astype(np.int64)
    dn = np.diff(n_real)
    if not np.all((dn > 0) | ((dn == 0) & (np.diff(ids) > 0))):
        raise refuse("layout.runs", "the run table is not sorted by (n_real, mol_id) or repeats "
                     "a mol_id")
    del dn
    b = h["bucket"]
    if n_real.min() < BUCKET_EDGES[b] or n_real.max() >= BUCKET_EDGES[b + 1]:
        raise refuse("layout.runs", f"a molecule's n_real is outside bucket {b} "
                     f"[{BUCKET_EDGES[b]}, {BUCKET_EDGES[b + 1]})")
    if int(n_conf.sum()) != h["n_conf"]:
        raise refuse("layout.runs", f"the run table sums to {int(n_conf.sum())} rows, the "
                     f"header says {h['n_conf']}")
    run_start = np.concatenate([[0], np.cumsum(n_conf)])
    n_color = np.zeros(n_mol, dtype=np.int64)
    if n_types:
        n_color = _scan_features(og, n_real, n_types, refuse)
    L_heavy = pad8(int(n_real.max()))
    L = pad8(int((n_real + n_color).max()))
    if L_heavy != h["L_heavy"] or L != h["L"]:
        raise refuse("layout.widths", f"recomputed L_heavy {L_heavy} / L {L} != declared "
                     f"{h['L_heavy']} / {h['L']} (limits 11-12)")
    blocks = og.blocks
    m0 = blocks["m0"].astype(np.int64)
    m1 = blocks["m1"].astype(np.int64)
    if (np.any(run_start[m1] - run_start[m0] != blocks["rows"].astype(np.int64))
            or np.any(run_start[m0] != blocks["first_row"].astype(np.int64))):
        raise refuse("layout.blocks", "a block's rows or first_row disagree with the run table")
    atoms_cum = np.concatenate([[0], np.cumsum(n_conf * n_real)])
    totals = atoms_cum[m1] - atoms_cum[m0]
    _scan_block_headers(og, totals, refuse)
    vaa = og.section_array("vaa")
    if not np.all(np.isfinite(vaa)) or np.any(vaa <= 0):
        raise refuse("value.self_overlap", "a vaa is not finite or is <= 0 (section 18.1)")
    del vaa
    if "color_self" in og.sections:
        cs = og.section_array("color_self")
        if not np.all(np.isfinite(cs)) or np.any(cs < 0):
            raise refuse("value.self_overlap", "a color_self is not finite or is < 0 "
                         "(section 18.2)")
    return {"mol_id": ids, "n_real": n_real, "n_conf": n_conf, "n_color": n_color,
            "run_start": run_start}


def _scan_features(og: OpenGroup, n_real: np.ndarray, n_types: int, refuse) -> np.ndarray:
    """Section 16.3 item 4 (8.1 rules 1-5) in the reference's order: the
    whole-table CSR ends, the per-molecule total (limit 16, which implies
    limit 15), the type range, then per chunk of ``_FEATURE_CHUNK`` features
    (one more for the boundary): atoms per feature, ascending atoms, atom
    index < n_real, canonical order. Every int64 temporary is per chunk, so
    memory is the sections plus a constant (limit 34; amendment S9)."""
    n_mol = n_real.size
    fo = og.section_array("feature_offsets").astype(np.int64)           # n_mol + 1
    fao = og.section_array("feature_atom_offsets")
    ft = og.section_array("feature_types")
    fa = og.section_array("feature_atoms")
    n_feat = int(ft.size)
    if fo[0] != 0 or np.any(np.diff(fo) < 0) or fo[-1] != n_feat:
        raise refuse("layout.features", "feature_offsets are not a valid CSR (section 8.1 rule 1)")
    if int(fao[0]) != 0 or int(fao[-1]) != fa.size or np.any(fao[1:] <= fao[:-1]):
        raise refuse("layout.features", "feature_atom_offsets are not a valid CSR (section 8.1 "
                     "rule 2)")
    n_color = np.diff(fo)
    if n_mol and int(n_color.max()) > LIMITS["features_per_molecule"]:
        raise refuse("limit.features_per_molecule", f"a molecule has {int(n_color.max())} "
                     f"features, limit {LIMITS['features_per_molecule']} (limit 16)")
    if n_feat and (int(ft.min()) < 1 or int(ft.max()) > n_types):
        raise refuse("layout.features", f"a feature type is outside [1, {n_types}]")
    width = LIMITS["atoms_per_feature"] + 1
    for f0 in range(0, n_feat, _FEATURE_CHUNK):
        f1 = min(n_feat, f0 + _FEATURE_CHUNK)
        f2 = min(n_feat, f1 + 1)                 # one more: the order check spans chunks
        ao = fao[f0:f2 + 1].astype(np.int64)
        k = np.diff(ao)
        if np.any(k < 1):
            raise refuse("layout.features", "feature_atom_offsets are not a valid CSR (a feature "
                         "has no atom; section 8.1 rule 2)")
        if int(k.max()) > LIMITS["atoms_per_feature"]:
            raise refuse("limit.atoms_per_feature", f"a feature has {int(k.max())} atoms, limit "
                         f"{LIMITS['atoms_per_feature']} (limit 17)")
        a0, a1 = int(ao[0]), int(ao[-1])
        atoms = fa[a0:a1].astype(np.int64)
        local = np.repeat(np.arange(f2 - f0), k)               # chunk feature of each atom
        first = np.zeros(atoms.size, dtype=bool)
        first[ao[:-1] - a0] = True
        if np.any(atoms[1:][~first[1:]] <= atoms[:-1][~first[1:]]):
            raise refuse("layout.features", "feature atoms are not strictly ascending within a "
                         "feature (section 8.1 rule 3)")
        mol = np.searchsorted(fo, np.arange(f0, f2), side="right") - 1
        if np.any(atoms >= n_real[mol[local]]):
            raise refuse("layout.features", "a feature atom index is >= its molecule's n_real")
        # canonical order within a molecule (rule 4): consecutive features of
        # one molecule compare <= on (type, atom tuple), lexicographic, so a
        # tuple that is a prefix of another sorts first (padding with -1).
        pad = np.full((f2 - f0, width), -1, dtype=np.int16)
        pad[:, 0] = ft[f0:f2]
        pad[local, np.arange(atoms.size) - (ao[local] - a0) + 1] = atoms
        same = mol[1:] == mol[:-1]
        x, y = pad[:-1][same], pad[1:][same]
        if x.size:
            ne = x != y
            col = np.argmax(ne, axis=1)
            r = np.arange(x.shape[0])
            if np.any(ne.any(axis=1) & (x[r, col] > y[r, col])):
                raise refuse("layout.features", "features of a molecule are not in canonical "
                             "(type, atom tuple) order (section 8.1 rule 4)")
    return n_color


def _scan_block_headers(og: OpenGroup, totals: np.ndarray, refuse) -> None:
    """Every block's 32-byte header: widths <= 16, reserved bytes zero, and
    ``length`` equal to the closed form (limits 19, 22). The headers are
    gathered from the ``coords`` section in bounded chunks."""
    blocks = og.blocks
    n = blocks.size
    base = og.sections["coords"]["offset"]
    clen = og.sections["coords"]["length"]
    offs = blocks["offset"].astype(np.int64)        # tiled at open: within coords
    hdrs = np.empty((n, BLOCK_HEADER_SIZE), dtype=np.uint8)
    i = 0
    while i < n:
        c0 = int(offs[i])
        c1 = min(clen, c0 + _COORDS_CHUNK)
        j = int(np.searchsorted(offs, c1 - BLOCK_HEADER_SIZE, side="right"))
        j = max(j, i + 1)
        end = min(clen, max(c1, int(offs[j - 1]) + BLOCK_HEADER_SIZE))
        buf = np.frombuffer(og.file.read(base + c0, end - c0, tier="scan"), dtype=np.uint8)
        rel = offs[i:j] - c0
        hdrs[i:j] = buf[rel[:, None] + np.arange(BLOCK_HEADER_SIZE)]
        i = j
    w = hdrs[:, 6:9].astype(np.int64)
    w_bad = w.max(axis=1) > LIMITS["bit_width"]
    res_bad = hdrs[:, 9:].any(axis=1)
    groups = -(-totals // 256)
    need = BLOCK_HEADER_SIZE + 32 * groups * w.sum(axis=1)
    len_u = blocks["length"]
    len_bad = (len_u > np.uint64(1 << 62)) | (need != np.minimum(len_u, np.uint64(1 << 62)).astype(np.int64))
    bad = w_bad | res_bad | len_bad
    if bad.any():
        i = int(np.argmax(bad))
        if w_bad[i]:
            raise refuse("limit.bit_width", f"block {i}: bit width {int(w[i].max())}, limit 16 "
                         f"(limit 22)")
        if res_bad[i]:
            raise refuse("layout.block_header", f"block {i}: reserved header bytes 9-31 are not "
                         f"zero")
        raise refuse("limit.block_length", f"block {i}: length {int(len_u[i])} != the closed form "
                     f"{int(need[i])} (limit 19)")


# ---------------------------------------------------------------------------
# deep: one group (section 16.4 item 1, and the sampled self-overlaps of 4)
# ---------------------------------------------------------------------------

def deep_group(og: OpenGroup, facts: dict, *, sample_mols: np.ndarray,
               color: Optional[dict], rng: np.random.Generator,
               reference_blocks: Optional[int], pending: list,
               reference_mols: Optional[np.ndarray] = None) -> set:
    """Section hashes, alignment padding, block hashes, every block decoded
    (tails zero), and for the sampled molecules (indices into this group's run
    table) ``vaa`` / ``color_self`` recomputed from the decoded coordinates.
    A self-overlap mismatch (section 16.4 item 4) is appended to ``pending``
    rather than raised, so items 2-3 refuse first, in the spec's order.

    With the native decoder, ``reference_blocks`` blocks of this group (drawn
    with ``rng``; every block when ``None``) are also decoded by the reference
    and compared bit for bit. Returns the indices of those blocks."""
    path = og.path

    def refuse(code, msg):
        return _refuse(code, msg, tier="deep", path=path)

    f = og.file
    for name, e in og.sections.items():
        dg = hashlib.sha256()
        for o in range(0, e["length"], 1 << 24):
            dg.update(f.read(e["offset"] + o, min(1 << 24, e["length"] - o), tier="deep"))
        if dg.hexdigest() != e["sha256"]:
            raise refuse("integrity.section_hash", f"section {name!r} does not hash to the "
                         f"header's sha256")
        end = e["offset"] + e["length"]
        if any(f.read(end, roundup64(end) - end, tier="deep")):
            raise refuse("integrity.padding", f"section {name!r}: alignment padding is not zero")
    runs = og.section_array("runs", tier="deep")
    n_conf, n_real = facts["n_conf"], facts["n_real"]
    run_start = facts["run_start"]
    L_heavy = og.header["L_heavy"]
    base = og.sections["coords"]["offset"]
    blocks = og.blocks
    want = np.zeros(n_real.size, dtype=bool)
    want[sample_mols] = True
    # None: every sampled molecule's color_self is also recomputed by the reference
    ref_mols = None if reference_mols is None else set(int(m) for m in reference_mols)
    vaa = og.section_array("vaa", tier="deep") if sample_mols.size else None
    cself = (og.section_array("color_self", tier="deep")
             if sample_mols.size and "color_self" in og.sections else None)
    feats = None
    if color is not None and cself is not None:
        feats = {n: og.section_array(n, tier="deep") for n in
                 ("feature_offsets", "feature_types", "feature_atom_offsets", "feature_atoms")}
    native = codec.backend() == "native"
    cross: set = set()
    if native:
        n_check = blocks.size if reference_blocks is None else min(blocks.size,
                                                                   int(reference_blocks))
        if n_check == blocks.size:
            cross = set(range(blocks.size))
        elif n_check:
            cross = set(rng.choice(blocks.size, size=n_check, replace=False).tolist())
    for i in range(blocks.size):
        bl = blocks[i]
        data = f.read(base + int(bl["offset"]), int(bl["length"]), tier="deep")
        if hashlib.sha256(data).digest() != bytes(bl["sha256"]):
            raise refuse("integrity.block_hash", f"block {i} does not hash to the block table's "
                         f"sha256")
        m0, m1 = int(bl["m0"]), int(bl["m1"])
        total = int((n_conf[m0:m1] * n_real[m0:m1]).sum())
        w = codec.block_widths(data)
        pos = BLOCK_HEADER_SIZE
        G = -(-total // 256)
        for a in range(3):
            size = G * w[a] * 32
            if np.any(codec.tail_values(data[pos:pos + size], w[a], total)):
                raise refuse("integrity.padding", f"block {i}: unused tail positions of axis "
                             f"{'xyz'[a]} are not zero (section 5.4)")
            pos += size
        xyz = _decode_checked(data, i, runs[m0:m1], n_conf[m0:m1], n_real[m0:m1], L_heavy,
                              i in cross, refuse)
        if want[m0:m1].any() and not pending:
            _recompute_self_overlaps(pending, xyz, m0, m1, want, run_start, n_real, vaa, cself, feats,
                                     color, int(bl["first_row"]), refuse, ref_mols)
    return cross


def _decode_checked(data, i, runs, n_conf, n_real, L_heavy, cross: bool, refuse):
    """Block ``i`` decoded by the dispatch decoder; with ``cross``, also by the
    reference and compared bit for bit (section 16.4 item 1)."""
    try:
        xyz = codec.decode_block(data, runs, L_heavy)
        if cross:                 # pragma: no cover - native only
            ref = reference.decode_block(data, np.repeat(n_real, n_conf), L_heavy)
            if not np.array_equal(ref, xyz):
                raise refuse("internal.error", f"block {i}: the native and reference "
                             f"decoders disagree")
    except ValueError as e:
        if isinstance(e, CorpusValidationError):
            raise
        raise refuse("internal.error", f"block {i} passed scan but the decoder refused it: "
                     f"{e}") from None
    return xyz


def _cross_decode_more(root: str, entry: dict, descriptor: dict, fingerprint: str,
                       done: set, budget: int, rng: np.random.Generator) -> int:
    """Up to ``budget`` more blocks of one group, not in ``done``, drawn with
    ``rng``, decoded by both decoders and compared (section 16.4 item 1): the
    allotment the walk could not use, so a corpus of at most K blocks has
    every block cross-decoded. The group is reopened (``open`` re-runs) and
    each block re-read and re-hashed. Returns how many were decoded."""
    og = open_group(root, entry, descriptor=descriptor, fingerprint=fingerprint)
    try:
        def refuse(code, msg):
            return _refuse(code, msg, tier="deep", path=og.path)

        rest = [i for i in range(og.blocks.size) if i not in done]
        if budget <= 0 or not rest:
            return 0
        pick = sorted(rest[j] for j in rng.choice(len(rest), size=min(budget, len(rest)),
                                                  replace=False).tolist())
        runs = og.section_array("runs", tier="deep")
        n_conf = runs["n_conf"].astype(np.int64)
        n_real = runs["n_real"].astype(np.int64)
        base = og.sections["coords"]["offset"]
        for i in pick:
            bl = og.blocks[i]
            data = og.file.read(base + int(bl["offset"]), int(bl["length"]), tier="deep")
            if hashlib.sha256(data).digest() != bytes(bl["sha256"]):
                raise refuse("integrity.block_hash", f"block {i} does not hash to the block "
                             f"table's sha256")
            m0, m1 = int(bl["m0"]), int(bl["m1"])
            _decode_checked(data, i, runs[m0:m1], n_conf[m0:m1], n_real[m0:m1],
                            og.header["L_heavy"], True, refuse)
        return len(pick)
    finally:
        og.close()


def _recompute_self_overlaps(pending, xyz, m0, m1, want, run_start, n_real, vaa, cself, feats, color,
                             first_row, refuse, ref_mols=None) -> None:
    """Section 16.4 item 4's self-overlaps for the sampled molecules of one
    block, checked row by row in corpus order (``vaa``, then ``color_self``);
    the first mismatch is appended to ``pending``.

    ``vaa`` is ``prep.conformer.stored_self_volume``, as the builder computes
    it. ``color_self`` is computed for every sampled row of the block at once
    by the native CPU path the builder uses
    (``backend.cpu.prepare_combo_library_self_overlaps``, section 18.2); for
    the molecules in ``ref_mols`` (a sub-sample) it is also computed by the
    float64 Python reference (``reference.overlap.color_overlap``, pseudo-atoms
    from ``reference.derive_positions``) and the two must agree within
    ``SELF_OVERLAP_RTOL``, else ``internal.error`` (a validator defect). Without
    the CPU extension every row uses the reference. (The reference on
    every sampled row was ~14 s of an 81 s deep run on a 5,014-molecule corpus.)"""
    from ...prep.conformer import stored_self_volume
    r = p = None
    if cself is not None and feats is not None:
        r = np.asarray(color["r"]["values"], dtype=np.float64)
        p = np.asarray(color["p"]["values"], dtype=np.float64)
    mols = [m for m in range(m0, m1) if want[m]]
    flists = {m: _feature_list(feats, m) if r is not None else [] for m in mols}
    native = None
    if r is not None:
        native = _native_color_self(xyz, mols, run_start, n_real, flists, first_row, r, p)
    for m in mols:
        nr = int(n_real[m])
        flist = flists[m]
        check_ref = r is not None and (native is None or ref_mols is None or m in ref_mols)
        types = np.asarray([t for t, _ in flist], dtype=np.int64)
        for row in range(int(run_start[m]), int(run_start[m + 1])):
            heavy = xyz[row - first_row]
            got = stored_self_volume(reference.dequantize(heavy[:, :nr]).astype(np.float64).T)
            if not _close(got, float(vaa[row])):
                return pending.append(refuse(
                    "value.self_overlap", f"row {row}: stored vaa {float(vaa[row])!r} != "
                    f"recomputed {got!r} (relative tolerance {SELF_OVERLAP_RTOL})"))
            if r is None:
                continue
            got = None if native is None else float(native[row])
            if check_ref:
                ref = _reference_color_self(heavy, flist, types, r, p)
                if got is not None and not _close(got, ref):
                    return pending.append(refuse(
                        "internal.error", f"row {row}: the native color self-overlap {got!r} "
                        f"and the reference's {ref!r} disagree (relative tolerance "
                        f"{SELF_OVERLAP_RTOL})"))
                got = ref if got is None else got
            if not _close(got, float(cself[row])):
                return pending.append(refuse(
                    "value.self_overlap", f"row {row}: stored color_self "
                             f"{float(cself[row])!r} != recomputed {got!r}"))


def _feature_list(feats: dict, m: int) -> list:
    """Molecule ``m``'s stored features as ``[(type, atom tuple), ...]``."""
    fo = feats["feature_offsets"]
    fao = feats["feature_atom_offsets"]
    out = []
    for fi in range(int(fo[m]), int(fo[m + 1])):
        a0, a1 = int(fao[fi]), int(fao[fi + 1])
        out.append((int(feats["feature_types"][fi]),
                    tuple(int(x) for x in feats["feature_atoms"][a0:a1])))
    return out


def _reference_color_self(heavy, flist, types, r, p) -> float:
    from ...reference.overlap import color_overlap
    if not flist:
        return 0.0
    pos = reference.dequantize(reference.derive_positions(heavy, flist)).astype(np.float64)
    return color_overlap(pos, types, pos, types, r, p)


def _native_color_self(xyz, mols, run_start, n_real, flists, first_row, r, p):
    """``{row: color_self}`` for every row of ``mols`` by the builder's native
    path, or ``None`` when the CPU extension is not built. The pseudo-atoms are
    derived from the decoded heavy atoms by section 8.3's integer rule and laid
    out as the builder lays them out (after the row's real atoms, in feature
    order, which is ascending type; counts per type)."""
    try:
        from ...backend import cpu
        cpu.extension()
    except ImportError:
        return None
    rows = [(m, row) for m in mols for row in range(int(run_start[m]), int(run_start[m + 1]))]
    if not rows:
        return {}
    n_types = r.shape[0] - 1
    width = pad8(max(int(n_real[m]) + len(flists[m]) for m in mols))
    coords = np.zeros((len(rows), 3, width), dtype=np.int16)
    counts = np.zeros((len(rows), n_types + 1), dtype=np.uint8)
    i = 0
    for m in mols:
        nr, flist = int(n_real[m]), flists[m]
        k = int(run_start[m + 1] - run_start[m])
        block = xyz[int(run_start[m]) - first_row:int(run_start[m + 1]) - first_row]
        coords[i:i + k, :, :nr] = block[:, :, :nr]
        for j, (_t, atoms) in enumerate(flist):
            s = block[:, :, list(atoms)].astype(np.int64).sum(axis=2)        # [k, 3]
            n = len(atoms)
            q = (2 * np.abs(s) + n) // (2 * n)                                 # half away from 0
            coords[i:i + k, :, nr + j] = np.where(s < 0, -q, q)
        counts[i:i + k, 0] = nr
        for t, _ in flist:
            counts[i:i + k, t] += 1
        i += k
    stored = (np.ascontiguousarray(coords.transpose(0, 2, 1)).astype(np.float32)
              * np.float32(reference.STEP))
    prepared = cpu.prepare_combo_library_self_overlaps(
        library_coords=stored, library_counts=counts, interaction_r=r.astype(np.float32),
        interaction_p=p.astype(np.float32), shape=False, color=True)
    if not prepared.color_prepared:
        raise RuntimeError("the native path did not prepare color self-overlaps")
    values = np.asarray(prepared.color_values, dtype=np.float64)
    return {row: float(values[j]) for j, (_m, row) in enumerate(rows)}


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= SELF_OVERLAP_RTOL * max(abs(a), abs(b))


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ValidationReport:
    tier: str
    root: str
    fingerprint: str
    descriptor: dict
    descriptor_sha256: str
    header_sha256: list            # actual group header hashes, group-table order
    groups: list                   # group headers
    seconds: float = 0.0
    sampled_molecules: int = 0
    backend: str = "reference"
    extra_files: list = dataclasses.field(default_factory=list)
    #: ``(name, size, mtime_ns)`` of every group file and of molecules.parquet
    #: as validated: part of the scan-cache key (:mod:`.cache`)
    file_stats: list = dataclasses.field(default_factory=list)
    #: a ``scan`` answered from the scan cache (only ``open`` ran)
    cached: bool = False
    #: ``deep``: sandbox children run at once for the RDKit items 
    workers: int = 0
    #: ``deep``, native decoder: blocks also decoded by the reference, and the
    #: sample size asked for (``None``: every block, ``--all``)
    reference_blocks: int = 0
    reference_sample: Optional[int] = None
    #: wall seconds per phase: ``groups`` (open, scan and deep per group file),
    #: ``molecules`` (the parquet scan and ids hash), ``rdkit`` (deep items 3-4)
    phase_seconds: dict = dataclasses.field(default_factory=dict)


def validate_corpus(root: str, tier: str = "scan", *, all_rows: bool = False,
                    bounds: Bounds = DEFAULT_BOUNDS, seed: Optional[int] = None,
                    sandbox_timeout: Optional[float] = None,
                    workers: Optional[int] = None) -> ValidationReport:
    """Run ``open``, ``scan`` or ``deep`` over a corpus directory. Raises
    :class:`CorpusValidationError` and nothing else; returns a report.

    ``all_rows`` makes ``deep`` cover every molecule and block instead of a
    sample; ``seed`` fixes the sample (default: ``os.urandom``); ``workers``
    is how many sandbox children ``deep`` runs at once (default
    ``min(16, os.cpu_count())``; the checks and the refusal do not depend on it)."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int)
                                or workers < 1):
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    try:
        return _validate(root, tier, all_rows, seed, sandbox_timeout, workers)
    except CorpusValidationError:
        raise
    except MemoryError:
        raise _refuse("internal.error", "the validator ran out of memory", tier=tier,
                      path=root) from None
    except Exception as e:      # a validator defect is still a refusal, never a crash
        raise _refuse("internal.error", f"unexpected {type(e).__name__}: {str(e)[:300]}",
                      tier=tier, path=root) from e


def _validate(root: str, tier: str, all_rows: bool, seed: Optional[int],
              sandbox_timeout: Optional[float], workers: Optional[int]) -> ValidationReport:
    t0 = time.monotonic()
    desc = read_descriptor(root)
    d, fp = desc.data, desc.fingerprint
    report = ValidationReport(tier, root, fp, d, desc.sha256, [], [], backend=codec.backend())
    color = d.get("color")
    rng = np.random.default_rng(seed if seed is not None
                                else int.from_bytes(os.urandom(8), "little"))
    sample_global = np.zeros(0, dtype=np.int64)
    if tier == "deep":
        n_total = d["n_mol"]
        k = n_total if all_rows else min(n_total, DEEP_SAMPLE_MOLECULES)
        sample_global = np.sort(rng.choice(n_total, size=k, replace=False))
        report.sampled_molecules = int(k)
    ref_self_global = None              # None: every sampled molecule (--all)
    if tier == "deep" and not all_rows:
        ref_self_global = np.sort(rng.choice(
            sample_global, size=min(sample_global.size, DEEP_REFERENCE_SELF_OVERLAP_MOLECULES),
            replace=False))
    # the reference cross-decode sample (section 16.4 item 1): K blocks per
    # corpus, allotted to the groups in proportion to their rows (the group
    # table's n_conf, checked at open), then drawn uniformly within a group;
    # an allotment larger than a group's block count carries to the next group,
    # and what is left after the last group goes back to the earlier groups'
    # unchecked blocks: a corpus of at most K blocks has every block checked
    ref_alloc = None
    ref_carry = 0
    ref_done: list = []
    if tier == "deep" and not all_rows and report.backend == "native":
        report.reference_sample = DEEP_REFERENCE_SAMPLE_BLOCKS
        rows = np.asarray([g["n_conf"] for g in d["groups"]], dtype=np.float64)
        ref_alloc = rng.multinomial(DEEP_REFERENCE_SAMPLE_BLOCKS, rows / rows.sum())
    ids_parts: list = []
    nreal_parts: list = []
    stored_feats: dict = {}
    lfw_declared: dict = {}
    lfw_actual: dict = {}
    mol_base = 0
    pending: list = []          # deferred item-4 refusals (deep_group)
    for gi, entry in enumerate(d["groups"]):
        og = open_group(root, entry, descriptor=d, fingerprint=fp)
        try:
            report.header_sha256.append(hashlib.sha256(og.header_bytes).hexdigest())
            report.file_stats.append((group_file_name(entry["bucket"], entry["group"]),
                                      *og.file.stat_key))
            if tier in ("scan", "deep"):
                facts = scan_group(og)
                ids_parts.append(facts["mol_id"])
                nreal_parts.append(facts["n_real"])
                b = entry["bucket"]
                lfw_declared.setdefault(b, set()).add(entry["L_field_width"])
                lfw_actual[b] = max(lfw_actual.get(b, 0), og.header["L_heavy"])
                if tier == "deep":
                    n_mol = og.header["n_mol"]
                    lo, hi = np.searchsorted(sample_global, [mol_base, mol_base + n_mol])
                    local = sample_global[lo:hi] - mol_base
                    want_ref = None if ref_alloc is None else int(ref_alloc[gi]) + ref_carry
                    ref_local = None
                    if ref_self_global is not None:
                        a, b = np.searchsorted(ref_self_global, [mol_base, mol_base + n_mol])
                        ref_local = ref_self_global[a:b] - mol_base
                    cross = deep_group(og, facts, sample_mols=local, color=color, rng=rng,
                                       reference_blocks=want_ref, pending=pending,
                                       reference_mols=ref_local)
                    report.reference_blocks += len(cross)
                    if want_ref is not None:
                        ref_carry = want_ref - len(cross)
                        ref_done.append((entry, cross))
                    if color is not None and local.size:
                        stored_feats.update(_stored_features(og, facts, local))
                    mol_base += n_mol
            report.groups.append(og.header)
        finally:
            og.close()
    for entry, done in ref_done:
        if ref_carry <= 0:
            break
        n = _cross_decode_more(root, entry, d, fp, done, ref_carry, rng)
        report.reference_blocks += n
        ref_carry -= n
    report.phase_seconds["groups"] = round(time.monotonic() - t0, 3)
    if tier in ("scan", "deep"):
        for b, declared in lfw_declared.items():
            if declared != {lfw_actual[b]}:
                raise _refuse("layout.widths", f"bucket {b}: group-table L_field_width "
                              f"{sorted(declared)} != the bucket-wide max L_heavy {lfw_actual[b]} "
                              f"(limit 13)", tier="scan", path=desc.path)
        ids = np.concatenate(ids_parts)
        n_real = np.concatenate(nreal_parts)
        del ids_parts, nreal_parts
        order = np.argsort(ids, kind="stable")
        ids, n_real = ids[order], n_real[order]
        del order
        if ids.size and np.any(np.diff(ids) == 0):
            dup = int(ids[:-1][np.diff(ids) == 0][0])
            raise _refuse("layout.runs", f"mol_id {dup} appears in more than one group",
                          tier="scan", path=root)
        if color is not None:
            _definition_prepass(color, desc.path)
        ids_digest = hashlib.sha256() if tier == "deep" else None
        t_mol = time.monotonic()
        pf = open_corpus_file(root, "molecules.parquet", tier="scan")
        report.file_stats.append(("molecules.parquet", *pf.stat_key))
        try:
            scan_molecules(pf, n_mol=d["n_mol"], molecules_sha256=d["molecules_sha256"],
                           all_ids=ids, all_n_real=n_real, ids_digest=ids_digest)
            if tier == "deep":
                if ids_digest.hexdigest() != d["ids_sha256"]:
                    raise _refuse("integrity.ids_hash", "the sorted mol_id column of "
                                  "molecules.parquet does not hash to corpus.json's ids_sha256",
                                  tier="deep", path=desc.path)
                from . import _sandbox
                report.workers = _sandbox.default_workers() if workers is None else workers
                report.phase_seconds["molecules"] = round(time.monotonic() - t_mol, 3)
                t_rd = time.monotonic()
                _sandbox.deep_rdkit_checks(pf, color=color, ids=ids, n_real=n_real,
                                           stored_features=stored_feats,
                                           timeout=sandbox_timeout, workers=report.workers)
                report.phase_seconds["rdkit"] = round(time.monotonic() - t_rd, 3)
                if pending:
                    raise pending[0]
        finally:
            pf.close()
        if tier == "deep" and exists_in_corpus(root, "build.json"):
            bf = open_corpus_file(root, "build.json", tier="deep")
            try:
                if bf.size > LIMITS["json_bytes"]:
                    raise _refuse("json.size", f"build.json is {bf.size} bytes, limit "
                                  f"{LIMITS['json_bytes']}", tier="deep", path=bf.path)
                raw = bf.read(0, bf.size, tier="deep")
            finally:
                bf.close()
            jsonstrict.loads(raw, what="build.json", path=bf.path, tier="deep")
    try:
        report.extra_files = sorted(
            n for n in os.listdir(root)
            if n not in ("corpus.json", "molecules.parquet", "build.json", "groups",
                         "corpus.json.sig"))
    except OSError:
        pass
    report.seconds = time.monotonic() - t0
    return report


def _definition_prepass(color: dict, path: str) -> None:
    """Section 15 through the one implementation, re-raised naming the file
    and the field (the pre-pass itself knows neither)."""
    try:
        reference.check_definition_text(color["definition"], color["generator"], color["types"])
    except CorpusValidationError as e:
        msg = str(e)
        suffix = f" [{e.code}]"
        if msg.endswith(suffix):
            msg = msg[:-len(suffix)]
        raise CorpusValidationError(e.code, f"color.definition: {msg}", tier="scan",
                                    path=path) from None


def _stored_features(og: OpenGroup, facts: dict, local: np.ndarray) -> dict:
    """``{mol_id: (group path, [(type, atoms), ...])}`` for the sampled molecules."""
    fo = og.section_array("feature_offsets", tier="deep")
    fao = og.section_array("feature_atom_offsets", tier="deep")
    ft = og.section_array("feature_types", tier="deep")
    fa = og.section_array("feature_atoms", tier="deep")
    out = {}
    for m in local.tolist():
        feats = []
        for f in range(int(fo[m]), int(fo[m + 1])):
            feats.append((int(ft[f]), tuple(int(x) for x in fa[int(fao[f]):int(fao[f + 1])])))
        out[int(facts["mol_id"][m])] = (og.path, feats)
    return out

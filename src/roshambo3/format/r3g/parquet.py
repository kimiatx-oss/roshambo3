"""``molecules.parquet`` (``doc/FORMAT.md`` section 11): writing it, and the
``scan``-tier checks that run before and while any data page is decoded.

Page headers are walked with ``reference.walk_parquet_pages`` (section 11
step 3 names it as the definition: a closed schema, at most 256 bytes per
header, read one header at a time with ``pread``), the one implementation, as
the definition pre-pass is. The footer's length is bounded from the file's
first 4 and last 8 bytes before pyarrow parses it, and pyarrow's own Thrift
limits are set. Row groups (at most 2^14 rows) are read **one column chunk at
a time**.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
from typing import Optional

import numpy as np

from .errors import CorpusValidationError
from .fileio import CorpusFile
from .limits import LIMITS
from .reference import parquet_footer_cap, walk_parquet_pages

COLUMNS = (("mol_id", "uint64"), ("ext_id", "string"), ("smiles", "string"),
           ("hac", "uint8"), ("mw", "double"), ("rotors", "uint16"))
ENCODINGS = {"PLAIN", "RLE"}
#: Per-row PLAIN byte caps per column (section 11 item 2).
COLUMN_BYTES = {"mol_id": 8, "ext_id": 256 + 4, "smiles": 4096 + 4, "hac": 4, "mw": 8,
                "rotors": 4}
COLUMN_SLACK = 4096

SMILES_OK = r"^[!-{}-~]+$"         # printable ASCII 0x21-0x7E without '|' (0x7C)
EXT_ID_BAD = ("[\\x00-\\x1f\\x7f-\\x9f\\x{200e}\\x{200f}\\x{202a}-\\x{202e}"
              "\\x{2066}-\\x{2069}]")                                  # RE2 syntax
_SMILES_RE = re.compile(SMILES_OK)


# ---------------------------------------------------------------------------
# Writing (section 11, writer rules)
# ---------------------------------------------------------------------------

def _schema():
    import pyarrow as pa
    return pa.schema([pa.field("mol_id", pa.uint64(), nullable=False),
                      pa.field("ext_id", pa.string(), nullable=False),
                      pa.field("smiles", pa.string(), nullable=False),
                      pa.field("hac", pa.uint8(), nullable=False),
                      pa.field("mw", pa.float64(), nullable=False),
                      pa.field("rotors", pa.uint16(), nullable=False)])


class MoleculesWriter:
    """Write ``molecules.parquet`` from rows that arrive in ascending
    ``mol_id`` order, in batches of any size, holding at most one row group
    (2^14 rows): the writer rules of section 11 -- REQUIRED columns in the
    spec's order, ZSTD level 9, PLAIN, no statistics, no embedded Arrow
    schema, no dictionary pages, row groups of exactly 2^14 rows but the
    last. :func:`write_molecules` is this class, so both write the same
    bytes for the same rows.

    Refuses (``ValueError``) a row the format cannot hold: ``mol_id``
    outside ``[0, 2^63)`` or not strictly ascending across all batches."""

    def __init__(self, path: str):
        import pyarrow.parquet as pq
        self.path = path
        self._schema = _schema()
        self._writer = pq.ParquetWriter(
            path, self._schema, compression="zstd", compression_level=9,
            write_statistics=False, store_schema=False, use_dictionary=False)
        self._rows = LIMITS["parquet_row_group"]
        self._buf: list = []
        self._buffered = 0
        self._last: Optional[int] = None
        self.n_rows = 0

    def write(self, mol_id, ext_id, smiles, hac, mw, rotors) -> None:
        ids = np.asarray(mol_id, dtype=np.uint64).reshape(-1)
        n = ids.size
        if not (len(ext_id) == len(smiles) == len(hac) == len(mw) == len(rotors) == n):
            raise ValueError("molecules.parquet: columns of different lengths")
        if n == 0:
            return
        if int(ids.max()) >= LIMITS["mol_id"]:
            bad = int(ids[ids >= np.uint64(LIMITS["mol_id"])][0])
            raise ValueError(f"mol_id {bad} outside [0, 2^63) (limit 35)")
        if np.any(ids[1:] <= ids[:-1]) or (self._last is not None and int(ids[0]) <= self._last):
            raise ValueError("molecules.parquet rows must arrive in strictly ascending mol_id")
        self._last = int(ids[-1])
        self._buf.append((ids, list(ext_id), list(smiles), np.asarray(hac),
                          np.asarray(mw, dtype=np.float64), np.asarray(rotors)))
        self._buffered += n
        while self._buffered >= self._rows:
            self._flush(self._rows)

    def _flush(self, n: int) -> None:
        import pyarrow as pa
        ids = np.concatenate([b[0] for b in self._buf])
        ext = [x for b in self._buf for x in b[1]]
        smi = [x for b in self._buf for x in b[2]]
        hac = np.concatenate([b[3] for b in self._buf])
        mw = np.concatenate([b[4] for b in self._buf])
        rot = np.concatenate([b[5] for b in self._buf])
        table = pa.table({
            "mol_id": ids[:n].tolist(), "ext_id": ext[:n], "smiles": smi[:n],
            "hac": [int(x) for x in hac[:n]], "mw": [float(x) for x in mw[:n]],
            "rotors": [int(x) for x in rot[:n]]}, schema=self._schema)
        self._writer.write_table(table, row_group_size=self._rows)
        self.n_rows += n
        rest = ids.size - n
        self._buf = ([(ids[n:], ext[n:], smi[n:], hac[n:], mw[n:], rot[n:])]
                     if rest else [])
        self._buffered = rest

    def close(self) -> None:
        if self._buffered:
            self._flush(self._buffered)
        self._writer.close()


def write_molecules(path: str, mol_id, ext_id, smiles, hac, mw, rotors) -> None:
    """Write ``molecules.parquet`` (:class:`MoleculesWriter`): rows sorted by
    ``mol_id`` here, so the input may be in any order."""
    ids = np.asarray(mol_id, dtype=np.uint64)
    order = np.argsort(ids, kind="stable")
    pick = lambda seq: [seq[i] for i in order]          # noqa: E731
    w = MoleculesWriter(path)
    w.write(ids[order], pick(list(ext_id)), pick(list(smiles)), np.asarray(hac)[order],
            np.asarray(mw, dtype=np.float64)[order], np.asarray(rotors)[order])
    w.close()


# ---------------------------------------------------------------------------
# Page headers (section 11 step 3)
# ---------------------------------------------------------------------------

def check_pages(chunk: bytes, n_rows: int, *, codec: str = "UNCOMPRESSED") -> None:
    """Walk one in-memory column chunk (``reference.walk_parquet_pages``, the
    closed-schema walker section 11 step 3 names as the definition). Raises
    ``ValueError`` naming the first violation."""
    walk_parquet_pages(lambda off, n: bytes(chunk[off:off + n]), len(chunk), n_rows, codec=codec)


# ---------------------------------------------------------------------------
# The scan tier over molecules.parquet (section 11 items 1-5, 16.3 item 7)
# ---------------------------------------------------------------------------

def scan_molecules(f: CorpusFile, *, n_mol: int, molecules_sha256: str,
                   all_ids: np.ndarray, all_n_real: np.ndarray,
                   ids_digest: Optional["hashlib._Hash"] = None) -> None:
    """Check ``molecules.parquet`` (open as ``f``) against the descriptor's
    ``n_mol`` and ``molecules_sha256`` and the sorted union of the run tables
    (``all_ids`` int64 ascending, ``all_n_real`` aligned with it). When
    ``ids_digest`` is given, the ``mol_id`` column is fed to it as int64 LE
    (``deep``'s ``ids_sha256``, computed in the same pass)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    path = f.path

    def refuse(code: str, msg: str) -> CorpusValidationError:
        return CorpusValidationError(code, f"molecules.parquet: {msg}", tier="scan", path=path)

    size = f.size
    cap = n_mol * LIMITS["parquet_row_bytes"] + LIMITS["parquet_overhead"]
    if size > cap:
        raise refuse("value.parquet", f"{size} bytes, cap {cap} for {n_mol} molecules (limit 21)")
    digest = hashlib.sha256()
    for off in range(0, size, 1 << 20):
        digest.update(f.read(off, min(1 << 20, size - off), tier="scan"))
    if digest.hexdigest() != molecules_sha256:
        raise refuse("integrity.molecules_hash", "does not hash to corpus.json's molecules_sha256")
    # step 1: the footer length, bounded before any library parses it (S3)
    max_rg, footer_cap = parquet_footer_cap(n_mol)
    if size < 12:
        raise refuse("value.parquet", f"{size} bytes, too short for a parquet file")
    head = f.read(0, 4, tier="scan")
    tail = f.read(size - 8, 8, tier="scan")
    footer_len = struct.unpack("<I", tail[:4])[0]
    if head != b"PAR1" or tail[4:] != b"PAR1":
        raise refuse("value.parquet", "does not begin and end with PAR1")
    if footer_len > footer_cap or footer_len + 12 > size:
        raise refuse("value.parquet", f"footer of {footer_len} bytes exceeds "
                     f"{min(footer_cap, size - 12)} (64 KiB + 1 KiB x {max_rg} row groups, "
                     f"within the file; limit 21)")
    from .parquet_footer import check_footer
    try:
        check_footer(f.read(size - 8 - footer_len, footer_len, tier="scan"), n_mol=n_mol, max_row_groups=max_rg)
    except (ValueError, UnicodeError) as e:
        raise refuse("value.parquet", f"invalid parquet footer: {e}") from None
    fh = os.fdopen(os.dup(f.fd), "rb")
    try:
        try:
            pf = pq.ParquetFile(fh, thrift_string_size_limit=footer_cap,
                                thrift_container_size_limit=max(1024, max_rg))
            md = pf.metadata
        except Exception as e:      # pyarrow raises its own hierarchy
            raise refuse("value.parquet", f"cannot read the footer: {str(e)[:200]}") from None
        if md.metadata and b"ARROW:schema" in md.metadata:
            raise refuse("value.parquet", "an embedded Arrow schema is not allowed")
        got = [(fl.name, str(fl.type), fl.nullable) for fl in pf.schema_arrow]
        if got != [(n, t, False) for n, t in COLUMNS]:
            raise refuse("value.parquet", f"schema {got} != {list(COLUMNS)} (all non-null)")
        for c in range(len(COLUMNS)):
            col = pf.schema.column(c)
            if col.max_definition_level != 0 or col.max_repetition_level != 0:
                raise refuse("value.parquet", f"column {COLUMNS[c][0]} is not REQUIRED")
        if md.num_rows != n_mol:
            raise refuse("value.parquet", f"{md.num_rows} rows != n_mol {n_mol}")
        if md.num_row_groups > max_rg:
            raise refuse("value.parquet", f"{md.num_row_groups} row groups, at most ceil(n_mol / "
                         f"{LIMITS['parquet_row_group']}) = {max_rg}")
        rows_sum = 0
        for rg in range(md.num_row_groups):
            g = md.row_group(rg)
            rows_sum += g.num_rows
            if g.num_rows > LIMITS["parquet_row_group"]:
                raise refuse("value.parquet", f"row group {rg} has {g.num_rows} rows, limit "
                             f"{LIMITS['parquet_row_group']}")
            for c in range(g.num_columns):
                col = g.column(c)
                name = COLUMNS[c][0]
                if col.compression not in ("UNCOMPRESSED", "ZSTD") or not set(col.encodings) <= ENCODINGS:
                    raise refuse("value.parquet", f"row group {rg} column {name}: codec "
                                 f"{col.compression}, encodings {col.encodings}; only "
                                 f"ZSTD/UNCOMPRESSED and PLAIN are allowed")
                if col.has_dictionary_page or col.dictionary_page_offset is not None:
                    raise refuse("value.parquet", f"row group {rg} column {name} has a "
                                 f"dictionary page")
                if col.is_stats_set:
                    raise refuse("value.parquet", f"row group {rg} column {name} carries "
                                 f"statistics")
                cap_c = g.num_rows * COLUMN_BYTES[name] + COLUMN_SLACK
                if col.total_uncompressed_size > cap_c or col.total_compressed_size > cap_c:
                    raise refuse("value.parquet", f"row group {rg} column {name} is larger "
                                 f"than its rows allow ({cap_c} bytes)")
                off, ln = col.data_page_offset, col.total_compressed_size
                if off < 4 or ln < 0 or off + ln > size - 8 - footer_len:
                    raise refuse("value.parquet", f"row group {rg} column {name}: byte range "
                                 f"outside the data")
                try:
                    walk_parquet_pages(lambda o, n, _b=off: f.read(_b + o, n, tier="scan"),
                                       ln, g.num_rows, codec=col.compression,
                                       uncompressed_cap=cap_c)
                except ValueError as e:
                    if isinstance(e, CorpusValidationError):
                        raise
                    raise refuse("value.parquet", f"row group {rg} column {name}: {e}") from None
        if rows_sum != md.num_rows:
            raise refuse("value.parquet", "row groups do not sum to the footer's row count")
        seen = 0
        prev_last = None

        def column(rg: int, name: str):
            try:
                result = pf.read_row_group(rg, columns=[name]).column(0)
                if len(result) != md.row_group(rg).num_rows or result.null_count:
                    raise ValueError("column length/null count differs from REQUIRED row group")
                if name in ("ext_id", "smiles"):
                    try:
                        result.validate(full=True)
                    except Exception:
                        raise refuse("limit." + name, "invalid UTF-8") from None
                return result
            except CorpusValidationError:
                raise
            except Exception as e:
                raise refuse("value.parquet", f"cannot read row group {rg} column {name}: "
                             f"{str(e)[:200]}") from None

        for rg in range(md.num_row_groups):
            ids = column(rg, "mol_id").to_numpy().astype(np.int64)
            n = ids.size
            if n and ids.min() < 0:          # a u64 >= 2^63 reads as negative int64
                raise refuse("limit.mol_id", f"row group {rg}: a mol_id is >= 2^63 (limit 35)")
            if n and (np.any(np.diff(ids) <= 0) or (prev_last is not None and ids[0] <= prev_last)):
                raise refuse("value.parquet", f"row group {rg}: mol_id is not strictly ascending")
            if n:
                prev_last = int(ids[-1])
            if not np.array_equal(ids, all_ids[seen:seen + n]):
                raise refuse("value.parquet", f"row group {rg}: the mol_id set differs from "
                             f"the run tables")
            if ids_digest is not None:
                ids_digest.update(ids.astype("<i8").tobytes())
            del ids
            hac = column(rg, "hac").to_numpy().astype(np.int64)
            if not np.array_equal(hac, all_n_real[seen:seen + n]):
                raise refuse("limit.hac", f"row group {rg}: a hac differs from its molecule's "
                             f"n_real (limit 36)")
            del hac
            seen += n
            smiles = column(rg, "smiles")
            sl = pc.binary_length(pc.cast(smiles, "binary")).to_numpy()
            if sl.size and (sl.max() > LIMITS["smiles"] or sl.min() < 1
                            or not pc.all(pc.match_substring_regex(smiles, SMILES_OK)).as_py()):
                raise refuse("limit.smiles", f"row group {rg}: a smiles is empty, longer than "
                             f"{LIMITS['smiles']} bytes, or holds a character outside printable "
                             f"ASCII without '|' or whitespace (CXSMILES; limit 31)")
            del smiles, sl
            ext_id = column(rg, "ext_id")
            el = pc.binary_length(pc.cast(ext_id, "binary")).to_numpy()
            if el.size and (el.max() > LIMITS["ext_id"] or el.min() < 1):
                raise refuse("limit.ext_id", f"row group {rg}: an ext_id is empty or longer "
                             f"than {LIMITS['ext_id']} bytes (limit 32)")
            if el.size and pc.any(pc.match_substring_regex(ext_id, EXT_ID_BAD)).as_py():
                raise refuse("limit.ext_id", f"row group {rg}: an ext_id carries a C0/C1 "
                             f"control or bidi character (limit 32)")
            del ext_id, el
            mw = column(rg, "mw").to_numpy()
            if not np.all(np.isfinite(mw)) or np.any(mw < 0):
                raise refuse("value.parquet", f"row group {rg}: mw must be finite and >= 0")
            rotors = column(rg, "rotors").to_numpy()
            if np.any(rotors < 0) or np.any(rotors > 65535):
                raise refuse("value.parquet", "rotors outside uint16")
        if seen != all_ids.size:
            raise refuse("value.parquet", "the mol_id set differs from the run tables")
    finally:
        fh.close()


def read_column_strings(f: CorpusFile, name: str):
    """Yield ``(row_offset, list[str])`` of one string column, one row group
    at a time (for ``deep``'s SMILES parse). Call only after ``scan``."""
    import pyarrow.parquet as pq
    fh = os.fdopen(os.dup(f.fd), "rb")
    try:
        pf = pq.ParquetFile(fh)
        seen = 0
        for rg in range(pf.metadata.num_row_groups):
            vals = pf.read_row_group(rg, columns=[name]).column(0).to_pylist()
            yield seen, vals
            seen += len(vals)
    finally:
        fh.close()

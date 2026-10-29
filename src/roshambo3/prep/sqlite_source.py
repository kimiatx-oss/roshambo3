"""Molecules read directly from a conformer SQLite database.

Two layouts are supported, both read strictly read-only:

``pubchem``
    ``mols(molecule_id INTEGER PRIMARY KEY, molbinary_zst BLOB)``, optionally
    with ``meta(molecule_id INTEGER PRIMARY KEY, hac, mw, smiles, rotor_count,
    ...)`` -- the PubChem3D ``conf1.sqlite`` / ``conf10.sqlite`` stores. A blob
    is one zstd frame holding an RDKit ``Mol`` binary with every conformer of
    the molecule and no properties at all, so identity comes from the SQL key.

``text-id``
    ``mols(compound_id TEXT PRIMARY KEY, mol_blob BLOB, n_confs, source)`` plus
    ``meta(key, value)`` declaring ``blob_compression = zstd``. A blob is the
    same thing: one zstd frame of a multi-conformer RDKit ``Mol`` binary.

Records are built from the decoded RDKit molecule directly. They never pass
through an SDF, whose V2000 ``F10.4`` coordinate field would round every input
coordinate by up to 5e-5 A.

Identity. ``ext_id`` is the source key as text -- ``str(molecule_id)`` or the
``compound_id`` -- and ``mol_id`` is ``stable_id(ext_id)``, exactly as for the
SDF, in-memory and binary sources: the molecule's name is set to ``ext_id``
and the shared per-molecule preparation (``sources._group_by_name``) does the
rest. Note that the legacy PubChem tiers built by ROSHAMBO2's ``rsb3.tiers``
used ``mol_id = molecule_id`` with a NULL ``ext_id``; ids from this source do
not coincide with theirs, so join the two on ``ext_id`` / ``molecule_id``.

Sampling. A molecule is in the tier iff

    h(key) = (splitmix64(key XOR seed) >> 11) / 2**53   lies in [p_lo, p_hi)

and, when a heavy-atom window is given, ``hac_lo <= hac <= hac_hi`` (both ends
inclusive). ``key`` is the integer ``molecule_id`` for PubChem and
``stable_id(compound_id)`` for text-id. This is bit-for-bit ROSHAMBO2's
``rsb3/hashsample.py`` with the same default seed, deliberately: ranges of
this hash line are already allocated (queries come from ``[0.7545, 1)``),
so a tier drawn with the same seed stays disjoint from those panels. The
variate is uniform over whichever keys exist, needs no RNG state, and gives
disjoint tiers for free by partitioning ``[0, 1)``. The defaults, ``p_lo = 0``
and ``p_hi = 1`` with no window, select the whole database.

With a PubChem ``meta`` table the population is the ``meta`` rows (each must
have a ``mols`` blob, or reading raises); without one it is the ``mols`` rows.
``hac`` is ``meta.hac`` when a PubChem ``meta`` table exists, so rejected
molecules are never decompressed; otherwise (text-id, or a ``mols``-only PubChem
slice) it is the prepared heavy-atom count, applied after decoding. The two
agree on PubChem: ``meta.hac`` equals the ``strip_hydrogens`` heavy-atom count.

Work units are ``(path, lo, hi)`` key ranges (``molecule_id`` for PubChem,
``rowid`` for text-id) of equal width, planned from MIN/MAX and a row count --
never a scan of the blob table. Each unit is read in ascending key order;
the set of emitted records depends on neither the worker count nor the unit
size, and the builder orders the corpus by content (``(n_real, mol_id,
conf_idx)``), so their order in the corpus does not either.

Identity on resume: besides the ``(size, mtime_ns)`` check below, each work
unit's content digest (``sha256`` over the selected ``(key, blob)`` rows,
computed as the unit is read) is committed with its slot, and a resume
re-reads the committed units' rows -- no decompression, no RDKit -- to check
them.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import time
from typing import Iterator, Optional

import numpy as np

from .conformer import stable_id, strip_hydrogens
from .sources import MoleculeRecord, _group_by_name, _hash_framed

#: The project-wide sampling seed, ``"RSHB3"`` as ASCII. Every built tier's
#: manifest records it; changing it moves every molecule on the hash line.
DEFAULT_SEED = 0x5253484233

_M64 = 0xFFFFFFFFFFFFFFFF

#: Above this selectivity a joined range scan of ``meta`` and ``mols`` beats
#: an id-list scan followed by batched primary-key lookups of the blobs.
FULLSCAN_THRESHOLD = 0.05

#: Primary keys per ``IN (...)`` lookup; below SQLite's 999-variable floor.
IN_BATCH = 900

#: File size above which ``COUNT(*)`` of the blob table is not attempted: with
#: no secondary index SQLite answers it by walking every leaf page.
COUNT_SCAN_MAX_BYTES = 2 << 30

KINDS = ("pubchem", "text-id")


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer over ``uint64``, wrapping mod 2**64."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hash_unit(keys, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Uniform variates in ``[0, 1)`` for integer keys, as float64.

    Uses the top 53 bits, so the division by 2**53 is exact.
    """
    k = np.asarray(keys, dtype=np.int64).astype(np.uint64)
    h = splitmix64(k ^ np.uint64(seed & _M64))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def hash_unit_scalar(key: int, seed: int = DEFAULT_SEED) -> float:
    """Scalar form of :func:`hash_unit`; identical results."""
    x = (((key & _M64) ^ (seed & _M64)) + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    x ^= x >> 31
    return (x >> 11) * (1.0 / 9007199254740992.0)


def open_read_only(path: str) -> sqlite3.Connection:
    """Open ``path`` so that no write is possible.

    ``mode=ro`` refuses writes at the VFS, ``immutable=1`` skips locking and
    the journal (nothing, not even a ``-shm`` file, is created next to the
    source), and ``query_only`` refuses writes at the SQL layer as well.
    """
    from pathlib import Path

    uri = f"{Path(path).resolve().as_uri()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.execute("PRAGMA query_only = 1")
    if con.execute("PRAGMA query_only").fetchone()[0] != 1:
        con.close()
        raise RuntimeError(f"could not open {path} query-only")
    return con


def _tables(con) -> dict:
    return {name: sql for name, sql in con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table'")}


def _columns(con, table: str) -> list[str]:
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def detect_kind(path: str) -> str:
    """``"pubchem"`` or ``"text-id"`` from the ``mols`` table's columns."""
    con = open_read_only(path)
    try:
        cols = set(_columns(con, "mols"))
    finally:
        con.close()
    if {"molecule_id", "molbinary_zst"} <= cols:
        return "pubchem"
    if {"compound_id", "mol_blob"} <= cols:
        return "text-id"
    raise ValueError(f"{path}: 'mols' has columns {sorted(cols)}, which match "
                     f"neither the PubChem nor the text-id layout")


class SqliteSource:
    """A read-only SQLite conformer database as a ``prepare`` source.

    Same contract as the other sources in ``prep/sources.py``: ``units``,
    ``iter_unit``, ``color_generator``, ``failures`` and ``stats``, plus the
    ``vaa_frame`` attribute the builder sets. Units
    are read by the workers themselves, so ``parallel_payloads`` is False and
    no connection is held on the instance (it is pickled into each worker).
    """

    parallel_payloads = False

    def __init__(self, path: str, kind: Optional[str] = None, color_generator=None,
                 *, seed: int = DEFAULT_SEED, p_lo: float = 0.0, p_hi: float = 1.0,
                 hac_lo: Optional[int] = None, hac_hi: Optional[int] = None,
                 n_rows: Optional[int] = None):
        self.path = os.path.abspath(str(path))
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        wal = self.path + "-wal"
        self.wal_bytes = os.path.getsize(wal) if os.path.exists(wal) else None
        if self.wal_bytes:
            # immutable=1 ignores the write-ahead log, so committed rows that
            # were never checkpointed would be silently invisible.
            raise ValueError(
                f"{wal} holds {self.wal_bytes} bytes of uncheckpointed writes; "
                f"a read-only immutable open would not see them. Checkpoint "
                f"the database (from its owner) first.")
        self.kind = kind or detect_kind(self.path)
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if not (0.0 <= p_lo <= p_hi <= 1.0):
            raise ValueError(f"need 0 <= p_lo <= p_hi <= 1, got [{p_lo}, {p_hi})")
        if (hac_lo is None) != (hac_hi is None):
            raise ValueError("give both hac_lo and hac_hi, or neither")
        if hac_lo is not None and hac_lo > hac_hi:
            raise ValueError(f"empty heavy-atom window [{hac_lo}, {hac_hi}]")
        self.color_generator = color_generator
        self.seed = int(seed)
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.hac_lo = hac_lo
        self.hac_hi = hac_hi
        self.n_rows_hint = n_rows
        self.failures: list[tuple[str, str]] = []
        self.dropped_conformers: dict = {}
        self.unit_sha256: Optional[str] = None
        self.stats: dict = {}
        st = os.stat(self.path)
        self._identity = (st.st_size, st.st_mtime_ns)

        con = open_read_only(self.path)
        try:
            tables = _tables(con)
            if "mols" not in tables:
                raise ValueError(f"{self.path} has no 'mols' table")
            cols = set(_columns(con, "mols"))
            if self.kind == "pubchem":
                if not {"molecule_id", "molbinary_zst"} <= cols:
                    raise ValueError(f"{self.path}: not a PubChem layout ({sorted(cols)})")
                self.has_meta = ("meta" in tables and
                                 {"molecule_id", "hac"} <= set(_columns(con, "meta")))
                self.blob_compression = "zstd"
            else:
                if not {"compound_id", "mol_blob"} <= cols:
                    raise ValueError(f"{self.path}: not a text-id layout ({sorted(cols)})")
                self.has_meta = False
                declared = {}
                if "meta" in tables:
                    declared = {k: (v.decode() if isinstance(v, bytes) else v)
                                for k, v in con.execute("SELECT key, value FROM meta")}
                self.blob_compression = declared.get("blob_compression")
                if self.blob_compression != "zstd":
                    raise ValueError(
                        f"{self.path}: meta.blob_compression is "
                        f"{self.blob_compression!r}; only 'zstd' is understood")
                self.declared_meta = declared
        finally:
            con.close()
        self._plan = None

    # -- identity --------------------------------------------------------

    @property
    def key_space(self) -> str:
        return "molecule_id" if self.kind == "pubchem" else "stable_id(compound_id)"

    @property
    def range_column(self) -> str:
        return "molecule_id" if self.kind == "pubchem" else "rowid"

    @property
    def hac_source(self) -> Optional[str]:
        if self.hac_lo is None:
            return None
        return "meta.hac" if self.has_meta else "prepared_n_real"

    def _check_unchanged(self) -> None:
        st = os.stat(self.path)
        if (st.st_size, st.st_mtime_ns) != self._identity:
            raise RuntimeError(f"{self.path} changed while being read")

    # -- planning (parent process) --------------------------------------

    def _plan_ranges(self) -> dict:
        if self._plan is not None:
            return self._plan
        t0 = time.perf_counter()
        con = open_read_only(self.path)
        try:
            col = self.range_column
            lo, hi = con.execute(f"SELECT MIN({col}), MAX({col}) FROM mols").fetchone()
            counts: dict = {}
            if self.has_meta and self.n_rows_hint is None:
                # Index-covered by meta's secondary indexes; never walks blobs,
                # but on a large database it walks a whole index. A caller
                # who knows the row count passes n_rows to skip it.
                counts["meta"] = con.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
            if (self.n_rows_hint is None
                    and os.path.getsize(self.path) <= COUNT_SCAN_MAX_BYTES):
                counts["mols"] = con.execute("SELECT COUNT(*) FROM mols").fetchone()[0]
            elif self.kind == "text-id" and lo is not None:
                # Rowids of an append-only table; an upper bound, not a count.
                counts["mols_rowid_span"] = hi - lo + 1
        finally:
            con.close()
        if self.n_rows_hint is not None:
            n_est = int(self.n_rows_hint)
        elif "mols" in counts:
            n_est = counts["mols"]
        elif "meta" in counts:
            n_est = counts["meta"]
        else:
            # Unknown density: plan as if every key existed. Units then hold
            # at most records_per_unit rows, never more.
            n_est = 0 if lo is None else hi - lo + 1
        self._plan = {"key_min": lo, "key_max": hi, "table_counts": counts,
                      "n_rows_estimate": n_est,
                      "plan_s": time.perf_counter() - t0}
        return self._plan

    def units(self, target_records: int = 20_000) -> list[tuple]:
        """Equal-width ``(path, lo, hi)`` key ranges, both ends inclusive."""
        if target_records < 1:
            raise ValueError("target_records must be positive")
        plan = self._plan_ranges()
        lo, hi = plan["key_min"], plan["key_max"]
        if lo is None:
            out = []
        else:
            n_units = max(1, math.ceil(plan["n_rows_estimate"] / target_records))
            span = hi - lo + 1
            n_units = min(n_units, span)
            edges = [lo + (span * i) // n_units for i in range(n_units + 1)]
            out = [(self.path, a, b - 1) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        self.stats = {"source_key_min": lo, "source_key_max": hi,
                      "source_rows_estimate": plan["n_rows_estimate"],
                      "source_units": len(out), "source_plan_s": plan["plan_s"]}
        return out

    def describe(self) -> dict:
        """The corpus manifest's ``source`` block: everything defining the tier."""
        plan = self._plan_ranges()
        st = os.stat(self.path)
        return {
            "kind": f"sqlite:{self.kind}",
            "path": self.path,
            "bytes": st.st_size,
            "mtime": st.st_mtime,
            "wal_bytes": self.wal_bytes,
            "blob_compression": self.blob_compression,
            "key_space": self.key_space,
            "range_column": self.range_column,
            "ext_id": "str(molecule_id)" if self.kind == "pubchem" else "compound_id",
            "mol_id": "stable_id(ext_id)",
            "sampler": {
                "hash": "splitmix64(key ^ seed) >> 11, / 2**53",
                "seed": self.seed, "p_lo": self.p_lo, "p_hi": self.p_hi,
                "hac_window": (None if self.hac_lo is None
                               else [int(self.hac_lo), int(self.hac_hi)]),
                "hac_window_inclusive": True,
                "hac_source": self.hac_source,
            },
            "key_min": plan["key_min"], "key_max": plan["key_max"],
            "table_counts": plan["table_counts"],
            "n_rows_hint": self.n_rows_hint,
            "population": "meta rows (each with a mols blob)" if self.has_meta else "mols rows",
            "complete": (self.p_lo == 0.0 and self.p_hi == 1.0
                         and self.hac_lo is None),
        }

    # -- reading (worker processes) --------------------------------------

    def _keep(self, keys: np.ndarray) -> np.ndarray:
        if self.p_lo == 0.0 and self.p_hi == 1.0:
            return np.ones(len(keys), dtype=bool)
        h = hash_unit(keys, self.seed)
        return (h >= self.p_lo) & (h < self.p_hi)

    def _hac_ok(self, hac) -> bool:
        return self.hac_lo is None or (hac is not None and self.hac_lo <= hac <= self.hac_hi)

    def _selective(self) -> bool:
        return (self.p_hi - self.p_lo) < FULLSCAN_THRESHOLD

    def _rows(self, con, lo: int, hi: int) -> Iterator[tuple]:
        """``(ext_id, blob, prefiltered_hac)`` for selected rows, ascending key."""
        if self.kind == "text-id":
            cur = con.execute("SELECT compound_id, mol_blob FROM mols "
                              "WHERE rowid BETWEEN ? AND ? ORDER BY rowid", (lo, hi))
            while rows := cur.fetchmany(1000):
                keys = np.fromiter((stable_id(r[0]) for r in rows), np.int64, len(rows))
                for (cid, blob), keep in zip(rows, self._keep(keys)):
                    if keep:
                        yield cid, blob, False
            return
        if self.has_meta and not self._selective():
            cur = con.execute(
                "SELECT m.molecule_id, m.hac, x.molbinary_zst FROM meta m "
                "LEFT JOIN mols x ON x.molecule_id = m.molecule_id "
                "WHERE m.molecule_id BETWEEN ? AND ? ORDER BY m.molecule_id", (lo, hi))
            while rows := cur.fetchmany(2000):
                keys = np.fromiter((r[0] for r in rows), np.int64, len(rows))
                for (mid, hac, blob), keep in zip(rows, self._keep(keys)):
                    if keep and self._hac_ok(hac):
                        if blob is None:
                            raise ValueError(f"{self.path}: meta row {mid} has no mols blob")
                        yield str(mid), blob, True
            return
        if self.has_meta or self._selective():
            # Choose ids first, then touch only the selected blobs' pages.
            if self.has_meta:
                cur = con.execute("SELECT molecule_id, hac FROM meta WHERE molecule_id "
                                  "BETWEEN ? AND ? ORDER BY molecule_id", (lo, hi))
            else:
                cur = con.execute("SELECT molecule_id, NULL FROM mols WHERE molecule_id "
                                  "BETWEEN ? AND ? ORDER BY molecule_id", (lo, hi))
            while rows := cur.fetchmany(100_000):
                keys = np.fromiter((r[0] for r in rows), np.int64, len(rows))
                selected = [r[0] for r, keep in zip(rows, self._keep(keys))
                            if keep and (not self.has_meta or self._hac_ok(r[1]))]
                for i in range(0, len(selected), IN_BATCH):
                    chunk = selected[i:i + IN_BATCH]
                    q = ("SELECT molecule_id, molbinary_zst FROM mols WHERE molecule_id IN ("
                         + ",".join("?" * len(chunk)) + ") ORDER BY molecule_id")
                    got = con.execute(q, chunk).fetchall()
                    if len(got) != len(chunk):
                        missing = sorted(set(chunk) - {g[0] for g in got})[:5]
                        raise ValueError(f"{self.path}: meta rows without a mols blob, "
                                         f"first {missing}")
                    for mid, blob in got:
                        yield str(mid), blob, self.has_meta
            return
        cur = con.execute("SELECT molecule_id, molbinary_zst FROM mols "
                          "WHERE molecule_id BETWEEN ? AND ? ORDER BY molecule_id", (lo, hi))
        while rows := cur.fetchmany(2000):
            keys = np.fromiter((r[0] for r in rows), np.int64, len(rows))
            for (mid, blob), keep in zip(rows, self._keep(keys)):
                if keep:
                    yield str(mid), blob, False

    @staticmethod
    def _hash_row(h, ext_id: str, blob) -> None:
        _hash_framed(h, ext_id.encode())
        _hash_framed(h, bytes(blob))

    def unit_digest(self, unit) -> str:
        """The content digest ``iter_unit`` computes for ``unit``: its
        selected rows re-read, never decompressed or prepared."""
        path, lo, hi = unit
        h = hashlib.sha256(b"roshambo3.unit.sqlite.v1")
        con = open_read_only(self.path)
        try:
            for ext_id, blob, _ in self._rows(con, lo, hi):
                self._hash_row(h, ext_id, blob)
        finally:
            con.close()
        return h.hexdigest()

    def iter_unit(self, unit) -> Iterator[MoleculeRecord]:
        try:
            import zstandard
            from rdkit import Chem
        except ImportError as e:
            raise ImportError("SQLite input requires the 'zstandard' and 'rdkit' "
                              "runtime dependencies") from e
        path, lo, hi = unit
        if os.path.abspath(path) != self.path:
            raise ValueError(f"unit for {path} given to the source for {self.path}")
        self._check_unchanged()
        self.failures = []
        self.dropped_conformers = {}
        self.unit_sha256 = None
        digest = hashlib.sha256(b"roshambo3.unit.sqlite.v1")
        decoder = zstandard.ZstdDecompressor()
        con = open_read_only(self.path)
        try:
            for ext_id, blob, hac_checked in self._rows(con, lo, hi):
                self._hash_row(digest, ext_id, blob)
                try:
                    mol = Chem.Mol(decoder.decompress(blob))
                except Exception as e:
                    self.failures.append((ext_id, f"{type(e).__name__}: {e}"))
                    continue
                if mol is None:
                    self.failures.append((ext_id, "RDKit returned None for binary payload"))
                    continue
                # The name is the identity: _group_by_name turns it into
                # ext_id and mol_id = stable_id(ext_id), as for every source.
                mol.SetProp("_Name", ext_id)
                if not hac_checked and self.hac_lo is not None:
                    # The prepared n_real is this count (prep keeps every
                    # heavy atom), so filter before paying for preparation.
                    if not self._hac_ok(strip_hydrogens(mol).GetNumAtoms()):
                        continue
                yield from _group_by_name([mol], 0, self.color_generator,
                                          self.failures,
                                          contiguous_conformer_ids=True,
                                          dropped=self.dropped_conformers)
        finally:
            con.close()
        self._check_unchanged()
        self.unit_sha256 = digest.hexdigest()

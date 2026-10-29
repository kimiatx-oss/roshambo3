"""The format-3 reader (``doc/FORMAT.md``).

:func:`open_corpus` always runs the ``open`` tier on ``corpus.json`` and on
every group file before it returns anything. Opened for a screen (the
default, ``purpose="screen"``) it also requires a cached passing ``scan`` of
this exact content (:mod:`.cache`) and refuses otherwise; ``roshambo3 verify
--tier scan`` (or :func:`verify`) is what records one. ``purpose="inspect"``
gives the descriptor, the group headers and the block directory of an
unscanned corpus, and refuses anything that reads beyond them.

A :class:`Group` exposes the run table, the feature table and the
fixed-width sections (``vaa`` f32, ``color_self`` f64) as read-only arrays
(``pread`` copies: section 1 item 4 forbids ``mmap`` without ``SIGBUS``
handling, which Python cannot do; :data:`.fileio.READER_ACCESS`), the derived
per-row arrays -- ``n_real``, ``n_color`` and the count vectors (section
8.5) -- and per-block decode through :mod:`.codec`. Format 3 stores no
per-conformer counts; the derived arrays are computed from the run and
feature tables after re-checking the cheap invariants that bound them (a
section edited after its scan cannot turn ``np.repeat`` into an allocation
bomb).
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, Optional

import numpy as np

from . import cache, codec
from .errors import SCAN_REQUIRED, CorpusValidationError
from .fileio import READER_ACCESS, open_corpus_file
from .limits import (BLOCK_DTYPE, DEFAULT_BOUNDS, DTYPES, LIMITS, RUN_DTYPE, STEP_LOG2, Bounds,
                     group_file_name)
from .validate import (OpenGroup, ValidationReport, open_group, read_descriptor,
                       validate_corpus)

PURPOSES = ("screen", "inspect")


def verify(root: str, tier: str = "scan", *, all_rows: bool = False,
           cache_dir: Optional[str] = None, seed: Optional[int] = None,
           bounds: Bounds = DEFAULT_BOUNDS, use_cache: bool = True,
           workers: Optional[int] = None) -> ValidationReport:
    """Validate ``root`` at ``tier`` and, when ``scan`` or ``deep`` passes,
    record it in the scan cache. A ``scan`` of content already in the cache
    runs ``open`` only (every header is still read and checked) and returns
    a report with ``cached=True``. ``workers`` bounds the sandbox children
    ``deep`` runs at once. Raises :class:`CorpusValidationError`."""
    if tier == "scan" and use_cache:
        c = open_corpus(root, purpose="inspect", cache_dir=cache_dir)
        if c.scanned:
            return ValidationReport("scan", root, c.fingerprint, c.descriptor, c._descriptor_sha256,
                                    list(c._header_sha256), list(c.headers), cached=True,
                                    file_stats=[(n, *st) for n, st in c._stats.items()])
    report = validate_corpus(root, tier, all_rows=all_rows, seed=seed, bounds=bounds,
                             workers=workers)
    if tier in ("scan", "deep"):
        key = cache.scan_key(report.descriptor_sha256, report.header_sha256, report.file_stats)
        cache.record(key, tier=tier, fingerprint=report.fingerprint, root=root,
                     directory=cache_dir)
    return report


BLOCK_DIRECTORY_DTYPE = np.dtype([("group", "<i4"), ("block", "<i4"), ("first_row", "<i8"),
                                  ("rows", "<i8"), ("cost", "<i8")])


class Corpus:
    """A format-3 corpus opened through validation and the scan-cache gate.

    ``Corpus(path)`` is the public convenience constructor. ``open_corpus``
    also installs already validated descriptor state through this class.
    """

    def __init__(self, root: str, *, purpose: str = "screen",
                 cache_dir: Optional[str] = None):
        opened = open_corpus(str(root), purpose=purpose, cache_dir=cache_dir)
        self.__dict__.update(opened.__dict__)

    @classmethod
    def _from_validated(cls, root: str, descriptor: dict, fingerprint: str,
                        entries: list, headers: list, blocks: list, stats: dict,
                        scanned: bool, purpose: str, descriptor_sha256: str,
                        header_sha256: tuple):
        """Install state only after open_corpus has validated and gated it."""
        self = object.__new__(cls)
        self._descriptor_sha256 = descriptor_sha256
        self._header_sha256 = tuple(header_sha256)
        self.root = root
        self.descriptor = descriptor
        self.fingerprint = fingerprint
        self.group_table = entries
        self.headers = headers
        self._blocks = blocks
        self._stats = stats
        self.scanned = scanned
        self.purpose = purpose
        return self

    # -- descriptor ---------------------------------------------------------
    @property
    def has_color(self) -> bool:
        return "color" in self.descriptor

    @property
    def n_types(self) -> int:
        return len(self.descriptor["color"]["types"]) if self.has_color else 0

    @property
    def color(self) -> Optional[dict]:
        return self.descriptor.get("color")

    @property
    def n_conf(self) -> int:
        return int(self.descriptor["n_conf"])

    @property
    def n_mol(self) -> int:
        return int(self.descriptor["n_mol"])

    @property
    def n_conformers(self) -> int:
        """Number of conformers (public spelling of ``n_conf``)."""
        return self.n_conf

    @property
    def n_molecules(self) -> int:
        """Number of molecules (public spelling of ``n_mol``)."""
        return self.n_mol

    @property
    def step(self) -> float:
        return 2.0 ** STEP_LOG2

    def __len__(self) -> int:
        return len(self.group_table)

    # -- groups -------------------------------------------------------------
    def group(self, i: int) -> "Group":
        """Open group ``i`` (group-table order). The file is opened again and
        passes ``open`` again; a file whose size or mtime changed since the
        corpus was opened is refused."""
        if self.purpose != "screen" and not self.scanned:
            raise CorpusValidationError(
                SCAN_REQUIRED, "group sections are read only from a scanned corpus; run "
                "`roshambo3 verify --tier scan` first", tier="scan", path=self.root)
        entry = self.group_table[i]
        og = open_group(self.root, entry, descriptor=self.descriptor, fingerprint=self.fingerprint)
        name = group_file_name(entry["bucket"], entry["group"])
        if og.file.stat_key != self._stats[name]:
            og.close()
            raise CorpusValidationError(
                SCAN_REQUIRED, f"{name} changed since the corpus was opened; validate it again",
                tier="scan", path=og.path)
        return Group(og, i)

    def groups(self) -> Iterator["Group"]:
        for i in range(len(self.group_table)):
            g = self.group(i)
            try:
                yield g
            finally:
                g.close()

    def block_directory(self, order: str = "file") -> np.ndarray:
        """Every block of every group (section 9.4): ``group``, ``block``,
        ``first_row``, ``rows`` and ``cost = rows * L``. ``order="file"``
        (the GPU producer's) or ``"expensive_first"`` (CPU schedulers:
        descending cost, then group, then block)."""
        parts = []
        for gi, (h, b) in enumerate(zip(self.headers, self._blocks)):
            d = np.zeros(b.size, dtype=BLOCK_DIRECTORY_DTYPE)
            d["group"] = gi
            d["block"] = np.arange(b.size)
            d["first_row"] = b["first_row"]
            d["rows"] = b["rows"]
            d["cost"] = d["rows"] * int(h["L"])
            parts.append(d)
        out = np.concatenate(parts)
        if order == "file":
            return out
        if order == "expensive_first":
            return out[np.lexsort((out["block"], out["group"], -out["cost"]))]
        raise ValueError("order is 'file' or 'expensive_first'")

    # -- molecules.parquet --------------------------------------------------
    def molecules(self, columns: Optional[list] = None):
        """Yield ``molecules.parquet`` one row group at a time (a pyarrow
        Table each). Only from a scanned corpus: the parquet is read only
        after ``scan`` has bounded it (section 11)."""
        import os
        import pyarrow.parquet as pq
        if not self.scanned:
            raise CorpusValidationError(
                SCAN_REQUIRED, "molecules.parquet is read only from a scanned corpus; run "
                "`roshambo3 verify --tier scan` first", tier="scan", path=self.root)
        f = open_corpus_file(self.root, "molecules.parquet", tier="scan")
        try:
            if f.stat_key != self._stats["molecules.parquet"]:
                raise CorpusValidationError(
                    SCAN_REQUIRED, "molecules.parquet changed since the corpus was opened",
                    tier="scan", path=f.path)
            fh = os.fdopen(os.dup(f.fd), "rb")
            try:
                pf = pq.ParquetFile(fh)
                for rg in range(pf.metadata.num_row_groups):
                    yield pf.read_row_group(rg, columns=columns)
            finally:
                fh.close()
        finally:
            f.close()


def open_corpus(root: str, *, purpose: str = "screen",
                cache_dir: Optional[str] = None) -> Corpus:
    """Open a corpus directory: ``open`` on ``corpus.json`` and every group
    file, then (``purpose="screen"``) the scan-cache check. Raises
    :class:`CorpusValidationError` and nothing else on a bad corpus."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose is one of {PURPOSES}")
    try:
        return _open_corpus(root, purpose, cache_dir)
    except CorpusValidationError:
        raise
    except Exception as e:
        raise CorpusValidationError("internal.error", f"unexpected {type(e).__name__}: "
                                    f"{str(e)[:300]}", tier="open", path=root) from e


def _open_corpus(root: str, purpose: str, cache_dir: Optional[str]) -> Corpus:
    import hashlib
    desc = read_descriptor(root)
    entries, headers, blocks, header_sha = [], [], [], []
    stats: dict = {}
    for entry in desc.data["groups"]:
        og = open_group(root, entry, descriptor=desc.data, fingerprint=desc.fingerprint)
        try:
            entries.append(entry)
            headers.append(og.header)
            blocks.append(og.blocks.copy())
            header_sha.append(hashlib.sha256(og.header_bytes).hexdigest())
            stats[group_file_name(entry["bucket"], entry["group"])] = og.file.stat_key
        finally:
            og.close()
    pf = open_corpus_file(root, "molecules.parquet")
    try:
        stats["molecules.parquet"] = pf.stat_key
    finally:
        pf.close()
    key = cache.scan_key(desc.sha256, header_sha,
                         [(n, *s) for n, s in stats.items()])
    scanned = cache.lookup(key, cache_dir) is not None
    if purpose == "screen" and not scanned:
        raise CorpusValidationError(
            SCAN_REQUIRED, "this corpus has not passed the scan tier (no cached scan for its "
            "content); run `roshambo3 verify --tier scan` on it before screening",
            tier="scan", path=root)
    return Corpus._from_validated(root, desc.data, desc.fingerprint, entries, headers, blocks, stats,
                  scanned, purpose, desc.sha256, header_sha)


class Group:
    """One open group file of a :class:`Corpus`. Close it when done."""

    def __init__(self, og: OpenGroup, index: int):
        self._og = og
        self.index = index
        self.header = og.header
        self.entry = og.entry
        self.path = og.path
        self.blocks = og.blocks
        self._map = og.file.map() if READER_ACCESS == "mmap" else None
        self._cache: dict = {}

    # -- header fields --------------------------------------------------------
    bucket = property(lambda self: int(self.header["bucket"]))
    group = property(lambda self: int(self.header["group"]))
    n_conf = property(lambda self: int(self.header["n_conf"]))
    n_mol = property(lambda self: int(self.header["n_mol"]))
    n_types = property(lambda self: int(self.header["n_types"]))
    L_heavy = property(lambda self: int(self.header["L_heavy"]))
    L = property(lambda self: int(self.header["L"]))
    L_field_width = property(lambda self: int(self.entry["L_field_width"]))
    n_blocks = property(lambda self: int(self.blocks.size))

    # -- sections -----------------------------------------------------------
    def section(self, name: str) -> Optional[np.ndarray]:
        """A stored section as a read-only array (``None`` when the section is
        absent: ``color_self``, or the feature sections of a shape corpus)."""
        if name in self._cache:
            return self._cache[name]
        e = self._og.sections.get(name)
        if e is None:
            return None
        dt = (RUN_DTYPE if name == "runs" else BLOCK_DTYPE if name == "blocks"
              else DTYPES[e["dtype"]])
        count = e["length"] // dt.itemsize
        if self._map is not None:
            arr = np.frombuffer(self._map, dtype=dt, count=count, offset=e["offset"])
        else:
            arr = np.frombuffer(self._og.read_section(name, tier="open"), dtype=dt)
        self._cache[name] = arr
        return arr

    runs = property(lambda self: self.section("runs"))
    vaa = property(lambda self: self.section("vaa"))
    color_self = property(lambda self: self.section("color_self"))
    feature_offsets = property(lambda self: self.section("feature_offsets"))
    feature_types = property(lambda self: self.section("feature_types"))
    feature_atom_offsets = property(lambda self: self.section("feature_atom_offsets"))
    feature_atoms = property(lambda self: self.section("feature_atoms"))

    @property
    def mol_id(self) -> np.ndarray:
        return self.runs["mol_id"]

    @property
    def molecule_n_conf(self) -> np.ndarray:
        return self.runs["n_conf"]

    @property
    def molecule_n_real(self) -> np.ndarray:
        return self.runs["n_real"]

    # -- derived (section 7 rule 1, 8.5) --------------------------------------
    def _refuse(self, code: str, msg: str) -> CorpusValidationError:
        return CorpusValidationError(code, msg, tier="scan", path=self.path)

    def _checked_runs(self) -> tuple[np.ndarray, np.ndarray]:
        if "_runs_checked" not in self._cache:
            n_conf = self.runs["n_conf"].astype(np.int64)
            n_real = self.runs["n_real"].astype(np.int64)
            if n_conf.size != self.n_mol or n_conf.min() < 1 or n_conf.max() > LIMITS["n_conf_per_molecule"] or n_real.min() < 1 \
                    or n_real.max() > self.L_heavy or int(n_conf.sum()) != self.n_conf:
                raise self._refuse("layout.runs", "the run table no longer agrees with the header")
            self._cache["_runs_checked"] = (n_conf, n_real)
        return self._cache["_runs_checked"]

    @property
    def run_start(self) -> np.ndarray:
        """``int64[n_mol + 1]``: molecule ``m``'s rows are ``[run_start[m], run_start[m+1])``."""
        n_conf, _ = self._checked_runs()
        return np.concatenate([[0], np.cumsum(n_conf)])

    def row_molecule(self, rows) -> np.ndarray:
        """Molecule index of each row (``searchsorted`` over run starts)."""
        return np.searchsorted(self.run_start, np.asarray(rows), side="right") - 1

    @property
    def n_real(self) -> np.ndarray:
        """Derived ``uint8[n_conf]``: each row's heavy-atom count."""
        n_conf, n_real = self._checked_runs()
        return np.repeat(n_real.astype(np.uint8), n_conf)

    def _molecule_counts(self) -> np.ndarray:
        """``uint8[n_mol, n_types + 1]``: column 0 ``n_real``, column t the
        features of type t (section 8.5), after re-checking the CSR."""
        if "_mol_counts" in self._cache:
            return self._cache["_mol_counts"]
        n_conf, n_real = self._checked_runs()
        nt = self.n_types
        counts = np.zeros((self.n_mol, nt + 1), dtype=np.int64)
        counts[:, 0] = n_real
        if nt:
            fo = self.feature_offsets.astype(np.int64)
            ft = self.feature_types
            if fo.size != self.n_mol + 1 or fo[0] != 0 or np.any(np.diff(fo) < 0) \
                    or fo[-1] != ft.size or (ft.size and (ft.min() < 1 or ft.max() > nt)):
                raise self._refuse("layout.features", "the feature table no longer agrees with "
                                   "the header")
            n_color = np.diff(fo)
            mol = np.repeat(np.arange(self.n_mol), n_color)
            np.add.at(counts, (mol, ft.astype(np.int64)), 1)
            if counts[:, 1:].sum(axis=1).max(initial=0) > LIMITS["features_per_molecule"] \
                    or (n_real + n_color).max() > self.L:
                raise self._refuse("layout.features", "a molecule's features exceed 255 or the "
                                   "device width")
        out = counts.astype(np.uint8)
        self._cache["_mol_counts"] = out
        return out

    @property
    def molecule_n_color(self) -> np.ndarray:
        return self._molecule_counts()[:, 1:].sum(axis=1, dtype=np.int64)

    @property
    def counts(self) -> np.ndarray:
        """Derived ``uint8[n_conf, n_types + 1]``: each row's count vector."""
        n_conf, _ = self._checked_runs()
        return np.repeat(self._molecule_counts(), n_conf, axis=0)

    @property
    def n_color(self) -> np.ndarray:
        """Derived ``uint8[n_conf]``: each row's feature count."""
        n_conf, _ = self._checked_runs()
        return np.repeat(self.molecule_n_color.astype(np.uint8), n_conf)

    # -- blocks -------------------------------------------------------------
    def block_bytes(self, i: int):
        """Block ``i``'s stored bytes (a read-only view, or a copy)."""
        b = self.blocks[i]
        start = self._og.sections["coords"]["offset"] + int(b["offset"])
        if self._map is not None:
            return memoryview(self._map)[start:start + int(b["length"])]
        return self._og.file.read(start, int(b["length"]), tier="open")

    def block_runs(self, i: int) -> np.ndarray:
        b = self.blocks[i]
        return self.runs[int(b["m0"]):int(b["m1"])]

    def decode_block(self, i: int) -> np.ndarray:
        """Block ``i`` decoded to ``int16[rows, 3, L_heavy]`` (section 4.3)
        through the codec dispatch (native when available)."""
        return codec.decode_block(self.block_bytes(i), self.block_runs(i), self.L_heavy)

    def decode_all(self) -> np.ndarray:
        """Every block, ``int16[n_conf, 3, L_heavy]`` (test-sized groups)."""
        out = np.zeros((self.n_conf, 3, self.L_heavy), dtype=np.int16)
        for i in range(self.n_blocks):
            r0 = int(self.blocks[i]["first_row"])
            xyz = self.decode_block(i)
            out[r0:r0 + xyz.shape[0]] = xyz
        return out

    def close(self) -> None:
        self._cache.clear()
        if self._map is not None:
            try:
                self._map.close()
            except BufferError:
                pass            # a view a caller still holds keeps it mapped
            self._map = None
        self._og.close()

    def __enter__(self) -> "Group":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

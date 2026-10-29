"""The format-3 writer (``doc/FORMAT.md`` sections 1, 3, 5-12).

:func:`write_corpus` writes ``groups/*.r3g``, ``molecules.parquet``,
``corpus.json`` and (optionally) ``build.json`` from in-memory molecules;
:func:`write_group` writes one group file. Given the same molecules,
``block_rows`` and ``max_conf_per_group`` the output is byte-identical to
``reference.write_corpus`` (the parquet up to its footer's ``created_by``,
which names the pyarrow release): canonical JSON, the block and group write
rules of sections 6 and 9.3, BP256-16 through :mod:`.codec`.

The writer refuses inputs it cannot represent with ``ValueError`` (this is
the builder's side, not a validation of untrusted bytes), so everything it
emits passes ``scan``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import struct
from typing import Iterable, Optional, Sequence

import numpy as np

from . import codec, jsonstrict, reference
from .errors import CorpusValidationError
from .limits import (BLOCK_DTYPE, BUCKET_EDGES, CONVENTIONS, FIRST_SECTION_OFFSET, FORMAT_OBJECT, FRAME,
                     LIMITS, MAGIC, MAJOR, RUN_DTYPE, SECTION_DTYPE, SECTION_SCHEMA,
                     bucket_of, group_file_name, pad8, roundup64)
from .parquet import write_molecules


@dataclasses.dataclass
class Molecule:
    """One molecule as the writer takes it (the reference's ``Molecule``
    fields; either type is accepted). ``coords`` is ``int16[n_conf, 3,
    n_real]`` in stored atom order (section 4.2); ``features`` are ``(type,
    atoms)`` in heavy-atom indices, any order (the writer canonicalizes,
    section 8.1 rule 4); ``vaa`` ``f32[n_conf]``; ``color_self``
    ``f64[n_conf]`` or ``None``."""
    mol_id: int
    coords: np.ndarray
    vaa: np.ndarray
    features: list = dataclasses.field(default_factory=list)
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


def _n_conf(m) -> int:
    return int(np.asarray(m.coords).shape[0])


def _n_real(m) -> int:
    return int(np.asarray(m.coords).shape[2])


def split_runs(sizes: Sequence[int], cap: int) -> list[tuple[int, int]]:
    """The write rule of sections 6 (rule 5) and 9.3 as index ranges: append
    while ``size + item <= cap``; a lone item may exceed ``cap``."""
    out: list = []
    start, cur = 0, 0
    for i, s in enumerate(sizes):
        if i > start and cur + s > cap:
            out.append((start, i))
            start, cur = i, 0
        cur += s
    if len(sizes) > start:
        out.append((start, len(sizes)))
    return out


# ---------------------------------------------------------------------------
# Input checks (ValueError: the caller's data cannot be represented)
# ---------------------------------------------------------------------------

def _check_molecule(m, n_types: int, with_color_self: bool) -> list:
    c = np.asarray(m.coords)
    if c.dtype != np.int16 or c.ndim != 3 or c.shape[1] != 3:
        raise ValueError(f"mol_id {m.mol_id}: coords must be int16[n_conf, 3, n_real]")
    n_conf, n_real = c.shape[0], c.shape[2]
    if not 1 <= n_conf <= LIMITS["n_conf_per_molecule"]:
        raise ValueError(f"mol_id {m.mol_id}: n_conf {n_conf} outside [1, 4096]")
    if not 1 <= n_real <= LIMITS["n_real"]:
        raise ValueError(f"mol_id {m.mol_id}: n_real {n_real} outside [1, 255]")
    if isinstance(m.mol_id, bool) or not 0 <= int(m.mol_id) < LIMITS["mol_id"]:
        raise ValueError(f"mol_id {m.mol_id} outside [0, 2^63)")
    vaa = np.asarray(m.vaa)
    if vaa.shape != (n_conf,) or not np.all(np.isfinite(vaa.astype(np.float32))) \
            or np.any(vaa.astype(np.float32) <= 0):
        raise ValueError(f"mol_id {m.mol_id}: vaa must be n_conf finite values > 0")
    if n_types and with_color_self:
        cs = np.asarray(m.color_self, dtype=np.float64) if m.color_self is not None else None
        if cs is None or cs.shape != (n_conf,) or not np.all(np.isfinite(cs)) or np.any(cs < 0):
            raise ValueError(f"mol_id {m.mol_id}: color_self must be n_conf finite values >= 0")
    feats = list(m.features or []) if n_types else []
    if not n_types and m.features:
        raise ValueError(f"mol_id {m.mol_id}: features given for a shape-only corpus")
    canon = reference.canonical_feature_order(feats)
    if len(canon) > LIMITS["features_per_molecule"]:
        raise ValueError(f"mol_id {m.mol_id}: {len(canon)} features, limit 255")
    per_type: dict = {}
    for t, atoms in canon:
        if not 1 <= t <= n_types:
            raise ValueError(f"mol_id {m.mol_id}: feature type {t} outside [1, {n_types}]")
        if not 1 <= len(atoms) <= LIMITS["atoms_per_feature"] or len(set(atoms)) != len(atoms):
            raise ValueError(f"mol_id {m.mol_id}: a feature needs 1..16 distinct atoms")
        if atoms[0] < 0 or atoms[-1] >= n_real:
            raise ValueError(f"mol_id {m.mol_id}: a feature atom is outside [0, n_real)")
        per_type[t] = per_type.get(t, 0) + 1
    if per_type and max(per_type.values()) > LIMITS["features_per_type"]:
        raise ValueError(f"mol_id {m.mol_id}: more than 255 features of one type")
    if n_real + len(canon) > LIMITS["L"]:
        raise ValueError(f"mol_id {m.mol_id}: n_real + n_color exceeds the device width 512")
    for name, s, cap in (("smiles", m.smiles, LIMITS["smiles"]), ("ext_id", m.ext_id,
                                                                   LIMITS["ext_id"])):
        if not isinstance(s, str) or not 1 <= len(s.encode("utf-8")) <= cap:
            raise ValueError(f"mol_id {m.mol_id}: {name} must be 1..{cap} bytes of UTF-8")
    if not reference._SMILES_OK.match(m.smiles):
        raise ValueError(f"mol_id {m.mol_id}: smiles must be printable ASCII without '|' "
                         f"or whitespace (no CXSMILES)")
    if any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F or ord(ch) in reference._BIDI
           for ch in m.ext_id):
        raise ValueError(f"mol_id {m.mol_id}: ext_id carries a control or bidi character")
    if not np.isfinite(float(m.mw)) or float(m.mw) < 0:
        raise ValueError(f"mol_id {m.mol_id}: mw must be finite and >= 0")
    if not 0 <= int(m.rotors) <= 65535:
        raise ValueError(f"mol_id {m.mol_id}: rotors outside u16")
    return canon


def _check_color(color: dict) -> None:
    from .validate import _check_color as check
    try:
        check(color, "<color descriptor>")
        reference.check_definition_text(color["definition"], color["generator"], color["types"])
    except CorpusValidationError as e:
        raise ValueError(f"the color descriptor is not valid: {e}") from None


# ---------------------------------------------------------------------------
# One group file
# ---------------------------------------------------------------------------

def _write_bytes(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
                 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _section_shape(name: str, data: bytes) -> list:
    if name == "blocks":
        return [len(data) // 64, 64]
    if name == "runs":
        return [len(data) // 16, 16]
    return [len(data) // _ITEMSIZE[SECTION_DTYPE[name]]]


_ITEMSIZE = {"u8": 1, "u16": 2, "u32": 4, "u64": 8, "i16": 2, "f32": 4, "f64": 8}


class GroupWriter:
    """Write one group file as its molecules arrive, in corpus order, without
    holding the group's coordinates: each block is encoded and written as
    soon as the block write rule (section 6 rule 5) closes it, and only the
    fixed-width sections (run table, block table, feature table, ``vaa``,
    ``color_self``) are kept until :meth:`close` lays them out after
    ``coords``, appends the canonical header and fills in the prologue.

    The layout is section 3.2's, and :func:`write_group` is this class, so the conformance tests that pin
    ``write_group`` and ``write_corpus`` to the reference pin it.

    Input is array-shaped, for the builder: :meth:`add` takes a chunk of
    whole molecules as their run records, their rows' padded heavy-atom
    coordinates and self-overlaps, and their feature table as per-molecule
    feature counts plus each feature's type, atom count and atoms -- already
    in the canonical order of section 8.1 rule 4 (:func:`write_group`
    canonicalizes per molecule). The structural rules are checked here,
    vectorized; rule 4's order is not (the ``scan`` tier checks it, and the
    builder runs ``scan`` on the finished corpus before publishing it).

    ``check_decode`` decodes every block just encoded and compares it with
    the rows it was given (the native decoder costs a few ns per conformer):
    the encoder is new code, and a block that does not decode to its input
    must never be published.
    """

    def __init__(self, root: str, *, bucket: int, group: int, fingerprint: str,
                 n_types: int, with_color_self: bool = False, block_rows: int = 4096,
                 check_decode: bool = False):
        if not 1 <= int(block_rows) <= LIMITS["block_rows"]:
            raise ValueError(f"block_rows must be in [1, {LIMITS['block_rows']}]")
        if not 0 <= int(n_types) <= LIMITS["n_types"]:
            raise ValueError(f"n_types must be in [0, {LIMITS['n_types']}]")
        if with_color_self and not n_types:
            raise ValueError("color_self exists only in a color corpus (section 18.2)")
        self.bucket, self.group = int(bucket), int(group)
        self.fingerprint = fingerprint
        self.n_types = int(n_types)
        self.with_color_self = bool(with_color_self)
        self.block_rows = int(block_rows)
        self.check_decode = bool(check_decode)
        os.makedirs(os.path.join(root, "groups"), exist_ok=True)
        self.path = os.path.join(root, "groups", group_file_name(bucket, group))
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                     | getattr(os, "O_NOFOLLOW", 0), 0o644)
        self._fh = os.fdopen(fd, "wb")
        self._closed = False
        self._fh.write(b"\0" * FIRST_SECTION_OFFSET)
        self._coords_sha = hashlib.sha256()
        self._coords_len = 0
        self._block_records: list = []
        self._block_rows_done = 0
        self._block_mols_done = 0
        self._runs: list = []
        self._vaa: list = []
        self._color_self: list = []
        self._feat_per_mol: list = []
        self._feat_types: list = []
        self._feat_natoms: list = []
        self._feat_atoms: list = []
        # the open block, as (runs, xyz) segments, and its row count
        self._pending: list = []
        self._pending_rows = 0
        self._n_mol = self._n_conf = 0
        self._max_n_real = self._max_total = 0
        self._last_key: Optional[tuple] = None

    # -- input ------------------------------------------------------------
    def add(self, runs: np.ndarray, xyz: np.ndarray, vaa, color_self=None,
            features: Optional[tuple] = None) -> None:
        """Append whole molecules. ``runs``: ``RUN_DTYPE[k]`` in corpus order
        (reserved bytes are written zero); ``xyz``: ``int16[rows, 3, W]``
        with ``W >= max n_real`` and ``rows = sum n_conf`` (slots beyond a
        row's ``n_real`` are ignored); ``vaa`` ``f32[rows]``; ``color_self``
        ``f64[rows]`` when the group carries it; ``features`` on a color
        group: ``(n_features[k], types[F], n_atoms[F], atoms[A])``."""
        if self._closed:
            raise ValueError("the group file is closed")
        runs = np.asarray(runs)
        if runs.dtype != RUN_DTYPE or runs.ndim != 1:
            raise ValueError("runs must be a RUN_DTYPE array")
        k = int(runs.size)
        if k == 0:
            return
        mol_id = runs["mol_id"].astype(np.uint64)
        n_conf = runs["n_conf"].astype(np.int64)
        n_real = runs["n_real"].astype(np.int64)
        cap = LIMITS["n_conf_per_molecule"]
        if n_conf.min() < 1 or n_conf.max() > cap:
            bad = int(mol_id[(n_conf < 1) | (n_conf > cap)][0])
            raise ValueError(f"mol_id {bad}: n_conf outside [1, {cap}] (section 7, limit 10)")
        if n_real.min() < 1 or n_real.max() > LIMITS["n_real"]:
            raise ValueError("n_real outside [1, 255] (section 7, limit 9)")
        if int(mol_id.max()) >= LIMITS["mol_id"]:
            bad = int(mol_id[mol_id >= np.uint64(LIMITS["mol_id"])][0])
            raise ValueError(f"mol_id {bad} outside [0, 2^63) (section 7, limit 35)")
        buckets = np.searchsorted(np.asarray(BUCKET_EDGES[1:-1]), n_real, side="right")
        if np.any(buckets != self.bucket):
            raise ValueError(f"every molecule of the group must be in bucket {self.bucket}")
        later = (n_real[1:] > n_real[:-1]) | ((n_real[1:] == n_real[:-1])
                                              & (mol_id[1:] > mol_id[:-1]))
        first = (int(n_real[0]), int(mol_id[0]))
        if not np.all(later) or (self._last_key is not None and first <= self._last_key):
            raise ValueError("a group's molecules must be in corpus order (n_real, mol_id), "
                             "unique (section 7 rule 2)")
        rows = int(n_conf.sum())
        xyz = np.asarray(xyz)
        if (xyz.dtype != np.int16 or xyz.ndim != 3 or xyz.shape[0] != rows
                or xyz.shape[1] != 3 or xyz.shape[2] < int(n_real.max())):
            raise ValueError("xyz must be int16[sum n_conf, 3, W >= max n_real]")
        vaa = np.asarray(vaa, dtype=np.float32).reshape(-1)
        if vaa.size != rows or not np.all(np.isfinite(vaa)) or np.any(vaa <= 0):
            raise ValueError("vaa must be one finite value > 0 per row (section 18.1)")
        cs = None
        if self.with_color_self:
            cs = (np.asarray(color_self, dtype=np.float64).reshape(-1)
                  if color_self is not None else None)
            if cs is None or cs.size != rows or not np.all(np.isfinite(cs)) or np.any(cs < 0):
                raise ValueError("color_self must be one finite value >= 0 per row "
                                 "(section 18.2)")
        elif color_self is not None:
            raise ValueError("color_self given for a group written without it")
        n_color = np.zeros(k, dtype=np.int64)
        feats = None
        if self.n_types:
            if features is None:
                raise ValueError("a color group needs every molecule's feature table")
            per_mol, types, natoms, atoms = (np.asarray(a).astype(np.int64).reshape(-1)
                                             for a in features)
            if (per_mol.size != k or int(per_mol.sum()) != types.size
                    or natoms.size != types.size or int(natoms.sum()) != atoms.size):
                raise ValueError("the feature table's sizes disagree")
            if per_mol.max() > LIMITS["features_per_molecule"]:
                raise ValueError("more than 255 features in one molecule (limit 16)")
            if types.size:
                if types.min() < 1 or types.max() > self.n_types:
                    raise ValueError(f"feature type outside [1, {self.n_types}]")
                if natoms.min() < 1 or natoms.max() > LIMITS["atoms_per_feature"]:
                    raise ValueError("a feature needs 1..16 atoms (limit 17)")
                atom_mol = np.repeat(np.repeat(np.arange(k), per_mol), natoms)
                if atoms.min() < 0 or np.any(atoms >= n_real[atom_mol]):
                    raise ValueError("a feature atom is outside [0, n_real) "
                                     "(section 8.1 rule 3)")
                start = np.zeros(types.size, dtype=np.int64)
                np.cumsum(natoms[:-1], out=start[1:])
                inner = np.ones(atoms.size, dtype=bool)
                inner[start] = False
                if np.any(inner[1:] & (atoms[1:] <= atoms[:-1])):
                    raise ValueError("a feature's atoms are not strictly ascending "
                                     "(section 8.1 rule 3)")
            n_color = per_mol
            feats = (per_mol, types.astype(np.uint8), natoms, atoms.astype(np.uint8))
        elif features is not None and np.asarray(features[1]).size:
            raise ValueError("features given for a shape-only corpus")
        if int((n_real + n_color).max()) > LIMITS["L"]:
            raise ValueError("n_real + n_color exceeds the device width 512 (limit 12)")
        if self._n_conf + rows > LIMITS["n_conf"]:
            raise ValueError(f"a group holds at most {LIMITS['n_conf']} conformers (limit 7)")
        # Accepted: record it.
        rec = np.zeros(k, dtype=RUN_DTYPE)
        rec["mol_id"], rec["n_conf"], rec["n_real"] = mol_id, n_conf, n_real
        self._runs.append(rec)
        self._vaa.append(vaa.astype("<f4"))
        if cs is not None:
            self._color_self.append(cs.astype("<f8"))
        if feats is not None:
            for acc, part in zip((self._feat_per_mol, self._feat_types, self._feat_natoms,
                                  self._feat_atoms), feats):
                acc.append(part)
        self._n_mol += k
        self._n_conf += rows
        self._max_n_real = max(self._max_n_real, int(n_real.max()))
        self._max_total = max(self._max_total, int((n_real + n_color).max()))
        self._last_key = (int(n_real[-1]), int(mol_id[-1]))
        self._cut_blocks(rec, xyz)

    def _cut_blocks(self, rec: np.ndarray, xyz: np.ndarray) -> None:
        """Section 6 rule 5 over this chunk, continuing the open block: a
        molecule joins the open block if it is empty or ``rows + n_conf <=
        block_rows``, else the block closes and the molecule starts the next."""
        n_conf = rec["n_conf"].astype(np.int64)
        row0 = np.zeros(n_conf.size + 1, dtype=np.int64)
        np.cumsum(n_conf, out=row0[1:])
        cuts = []
        rows = self._pending_rows
        for i, nc in enumerate(n_conf.tolist()):
            if rows and rows + nc > self.block_rows:
                cuts.append(i)
                rows = 0
            rows += nc
        bounds = [0] + cuts + [n_conf.size]
        for j in range(len(bounds) - 1):
            a, b = bounds[j], bounds[j + 1]
            if j > 0:
                self._flush_block()
            if b > a:
                self._pending.append((rec[a:b], xyz[row0[a]:row0[b]]))
                self._pending_rows += int(row0[b] - row0[a])

    def _flush_block(self) -> None:
        if not self._pending:
            return
        runs = np.concatenate([r for r, _ in self._pending])
        width = pad8(int(runs["n_real"].max()))
        rows = self._pending_rows
        xyz = np.zeros((rows, 3, width), dtype=np.int16)
        r = 0
        for _, seg in self._pending:
            n, w = seg.shape[0], min(width, seg.shape[2])
            xyz[r:r + n, :, :w] = seg[:, :, :w]
            r += n
        data = codec.encode_block(xyz, runs)
        if self.check_decode:
            got = codec.decode_block(data, runs, width)
            nr = np.repeat(runs["n_real"].astype(np.int64), runs["n_conf"].astype(np.int64))
            real = np.arange(width)[None, None, :] < nr[:, None, None]
            if not np.array_equal(np.where(real, xyz, 0), got):
                raise RuntimeError(f"{self.path}: block {len(self._block_records)} does not "
                                   f"decode to the rows it was encoded from")
        if len(self._block_records) + 1 > LIMITS["blocks"]:
            raise ValueError(f"group exceeds the block count limit ({LIMITS['blocks']} "
                             f"blocks per group, limit 6)")
        blk = np.zeros((), dtype=BLOCK_DTYPE)
        blk["first_row"], blk["rows"] = self._block_rows_done, rows
        blk["offset"], blk["length"] = self._coords_len, len(data)
        blk["m0"], blk["m1"] = self._block_mols_done, self._block_mols_done + runs.size
        blk["sha256"] = np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8)
        self._block_records.append(blk)
        self._block_rows_done += rows
        self._block_mols_done += int(runs.size)
        self._fh.write(data)
        self._coords_sha.update(data)
        self._coords_len += len(data)
        self._pending, self._pending_rows = [], 0

    # -- output -----------------------------------------------------------
    def close(self, L_field_width: Optional[int] = None) -> dict:
        """Finish the file and return its group-table entry
        (``L_field_width`` defaults to the group's own ``L_heavy``)."""
        if self._closed:
            raise ValueError("the group file is already closed")
        try:
            if self._n_mol == 0:
                raise ValueError("a group has at least one molecule (section 9.3)")
            self._flush_block()
            n_mol, n_conf = self._n_mol, self._n_conf
            sections: dict = {
                "blocks": np.array(self._block_records, dtype=BLOCK_DTYPE).tobytes(),
                "runs": np.concatenate(self._runs).tobytes()}
            if self.n_types:
                per_mol = np.concatenate(self._feat_per_mol)
                natoms = np.concatenate(self._feat_natoms)
                if max(int(per_mol.sum()), int(natoms.sum())) > LIMITS["n_feat_atoms"]:
                    raise ValueError("group exceeds the feature CSR limit (limit 20)")
                fo = np.zeros(n_mol + 1, dtype="<u4")
                fo[1:] = np.cumsum(per_mol)
                fao = np.zeros(natoms.size + 1, dtype="<u4")
                fao[1:] = np.cumsum(natoms)
                sections["feature_offsets"] = fo.tobytes()
                sections["feature_types"] = np.concatenate(self._feat_types).astype("u1").tobytes()
                sections["feature_atom_offsets"] = fao.tobytes()
                sections["feature_atoms"] = np.concatenate(self._feat_atoms).astype("u1").tobytes()
            sections["vaa"] = np.concatenate(self._vaa).astype("<f4").tobytes()
            if self.with_color_self:
                sections["color_self"] = np.concatenate(self._color_self).astype("<f8").tobytes()
            table = [{"name": "coords", "offset": FIRST_SECTION_OFFSET,
                      "length": self._coords_len, "dtype": "u8",
                      "shape": [self._coords_len], "sha256": self._coords_sha.hexdigest()}]
            offset = roundup64(FIRST_SECTION_OFFSET + self._coords_len)
            self._fh.write(b"\0" * (offset - FIRST_SECTION_OFFSET - self._coords_len))
            for name, _, _ in SECTION_SCHEMA[1:]:
                if name not in sections:
                    continue
                data = sections[name]
                table.append({"name": name, "offset": offset, "length": len(data),
                              "dtype": SECTION_DTYPE[name], "shape": _section_shape(name, data),
                              "sha256": hashlib.sha256(data).hexdigest()})
                end = roundup64(offset + len(data))
                self._fh.write(data)
                self._fh.write(b"\0" * (end - offset - len(data)))
                offset = end
            L_heavy, L = pad8(self._max_n_real), pad8(self._max_total)
            header = {"major": MAJOR, "fingerprint": self.fingerprint, "bucket": self.bucket,
                      "group": self.group, "n_conf": n_conf, "n_mol": n_mol,
                      "n_types": self.n_types, "L_heavy": L_heavy, "L": L, "sections": table}
            header_bytes = jsonstrict.dumps(header)
            jsonstrict.loads(header_bytes, what="group header")
            self._fh.write(header_bytes)
            self._fh.seek(0)
            self._fh.write(MAGIC + struct.pack("<IQ", MAJOR, offset) + b"\0" * 48)
            self._fh.close()
        except BaseException:
            self.abort()
            raise
        self._closed = True
        return {"bucket": self.bucket, "group": self.group, "L_heavy": L_heavy, "L": L,
                "n_conf": n_conf, "n_mol": n_mol,
                "L_field_width": int(L_field_width if L_field_width is not None else L_heavy),
                "header_sha256": hashlib.sha256(header_bytes).hexdigest()}

    def abort(self) -> None:
        """Close and remove the partial file (an interrupted group is never
        left looking like a group file)."""
        if not self._closed:
            self._closed = True
            try:
                self._fh.close()
            finally:
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass

    def __enter__(self) -> "GroupWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()


def feature_arrays(features: Sequence[list]) -> tuple:
    """:meth:`GroupWriter.add`'s ``features`` from per-molecule ``[(type,
    atoms), ...]`` lists already in canonical order (section 8.1 rules 3-4)."""
    flat = [x for f in features for x in f]
    return (np.asarray([len(f) for f in features], dtype=np.int64),
            np.asarray([t for t, _ in flat], dtype=np.int64),
            np.asarray([len(a) for _, a in flat], dtype=np.int64),
            np.asarray([x for _, a in flat for x in a], dtype=np.int64))


def write_group(root: str, molecules: Sequence, *, bucket: int, group: int,
                fingerprint: str, n_types: int, with_color_self: bool = False,
                block_rows: int = 4096, L_field_width: Optional[int] = None,
                _checked: Optional[list] = None) -> dict:
    """Write ``groups/b{bucket:02d}_g{group:04d}.r3g`` from one group's
    molecules (in corpus order, all in ``bucket``). Returns its group-table
    entry (``L_field_width`` defaults to the group's own ``L_heavy``; the
    corpus writer passes the bucket-wide value). Through
    :class:`GroupWriter`, one molecule at a time."""
    if not molecules:
        raise ValueError("a group has at least one molecule (section 9.3)")
    if not 1 <= block_rows <= LIMITS["block_rows"]:
        raise ValueError("block_rows must be in [1, 4096]")
    feats = _checked if _checked is not None else [
        _check_molecule(m, n_types, with_color_self) for m in molecules]
    keys = [(_n_real(m), int(m.mol_id)) for m in molecules]
    if keys != sorted(keys) or len(set(k[1] for k in keys)) != len(keys):
        raise ValueError("a group's molecules must be in corpus order (n_real, mol_id), unique")
    if any(bucket_of(k[0]) != bucket for k in keys):
        raise ValueError(f"every molecule of the group must be in bucket {bucket}")
    with GroupWriter(root, bucket=bucket, group=group, fingerprint=fingerprint,
                     n_types=n_types, with_color_self=with_color_self,
                     block_rows=block_rows) as w:
        for m, f in zip(molecules, feats):
            c = np.asarray(m.coords)
            w.add(reference.pack_runs([int(m.mol_id)], [c.shape[0]], [c.shape[2]]), c,
                  m.vaa, m.color_self if (n_types and with_color_self) else None,
                  feature_arrays([f]) if n_types else None)
        return w.close(L_field_width)


# ---------------------------------------------------------------------------
# A whole corpus
# ---------------------------------------------------------------------------

def descriptor_skeleton(color: Optional[dict], prep_version: str,
                        prep_commit: Optional[str]) -> dict:
    d: dict = {"format": dict(FORMAT_OBJECT), "frame": dict(FRAME)}
    if color:
        d["color"] = color
    d["prep"] = {"version": prep_version, "commit": prep_commit,
                 "conventions": dict(CONVENTIONS)}
    return d


GROUP_TABLE_KEYS = ("bucket", "group", "L_heavy", "L", "n_conf", "n_mol", "L_field_width",
                    "header_sha256")


def write_descriptor(root: str, *, color: Optional[dict], prep_version: str,
                     prep_commit: Optional[str], groups: Sequence[dict], ids_sha256: str) -> dict:
    """Write ``corpus.json`` (section 10) for group files and a
    ``molecules.parquet`` already in ``root``; return the descriptor.

    ``groups`` are the group-table entries :class:`GroupWriter` returned, in
    ``(bucket, group)`` order; ``ids_sha256`` is section 10.6's hash, which
    the caller computes from the ids it wrote to the parquet (the writer
    never reads them back); ``molecules_sha256`` is hashed from the file.
    Canonical JSON, strict-parsed before it is written."""
    if not groups:
        raise ValueError("a corpus has at least one group")
    if len(groups) > LIMITS["groups"]:
        raise ValueError(f"{len(groups)} groups, limit {LIMITS['groups']} (limit 5)")
    table = [{k: e[k] for k in GROUP_TABLE_KEYS} for e in groups]
    keys = [(e["bucket"], e["group"]) for e in table]
    if keys != sorted(keys) or len(set(keys)) != len(keys) or any(
            g != (0 if i == 0 or keys[i - 1][0] != b else keys[i - 1][1] + 1)
            for i, (b, g) in enumerate(keys)):
        raise ValueError("the group table must list (bucket, group) in order, groups of a "
                         "bucket numbered 0, 1, ... (section 10.4)")
    descriptor = descriptor_skeleton(color, prep_version, prep_commit)
    descriptor["groups"] = table
    descriptor["n_conf"] = sum(int(e["n_conf"]) for e in table)
    descriptor["n_mol"] = sum(int(e["n_mol"]) for e in table)
    descriptor["ids_sha256"] = ids_sha256
    h = hashlib.sha256()
    with open(os.path.join(root, "molecules.parquet"), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    descriptor["molecules_sha256"] = h.hexdigest()
    descriptor_bytes = jsonstrict.dumps(descriptor)
    jsonstrict.loads(descriptor_bytes, what="corpus.json")
    _write_bytes(os.path.join(root, "corpus.json"), descriptor_bytes)
    return descriptor


def write_corpus(root: str, molecules: Iterable, *, color: Optional[dict],
                 prep_version: str, prep_commit: Optional[str], block_rows: int = 4096,
                 max_conf_per_group: int = 2_000_000, build: Optional[dict] = None,
                 with_color_self: Optional[bool] = None) -> dict:
    """Write a whole corpus directory; return the descriptor (``corpus.json``).

    ``color`` is the section-10.2 object (``reference.color_descriptor``
    builds one) or ``None`` for a shape-only corpus. ``with_color_self``
    defaults to "when the matrices are canonical"; asking for it with
    non-canonical matrices is refused (section 18.2)."""
    from .validate import corpus_fingerprint
    molecules = list(molecules)
    if not molecules:
        raise ValueError("a corpus has at least one molecule")
    if not 1 <= block_rows <= LIMITS["block_rows"]:
        raise ValueError("block_rows must be in [1, 4096]")
    if not 1 <= max_conf_per_group <= LIMITS["n_conf"]:
        raise ValueError(f"max_conf_per_group must be in [1, {LIMITS['n_conf']}]")
    n_types = 0
    canonical = False
    if color:
        _check_color(color)
        n_types = len(color["types"])
        canonical = reference.canonical_matrices(color["r"]["values"], color["p"]["values"],
                                                 n_types)
    if with_color_self is None:
        with_color_self = bool(color) and canonical
    elif with_color_self and not (color and canonical):
        raise ValueError("color_self may be written only for a color corpus with canonical "
                         "interaction matrices (section 18.2)")
    fp = corpus_fingerprint(descriptor_skeleton(color, prep_version, prep_commit))
    ids = [int(m.mol_id) for m in molecules]
    if len(set(ids)) != len(ids):
        raise ValueError("mol_id repeats in the input (section 7 rule 2)")
    checked = {id(m): _check_molecule(m, n_types, with_color_self) for m in molecules}
    ordered = sorted(molecules, key=lambda m: (bucket_of(_n_real(m)), _n_real(m), int(m.mol_id)))
    by_bucket: dict = {}
    for m in ordered:
        by_bucket.setdefault(bucket_of(_n_real(m)), []).append(m)
    os.makedirs(os.path.join(root, "groups"), exist_ok=True)
    table = []
    for b in sorted(by_bucket):
        mols = by_bucket[b]
        lfw = pad8(max(_n_real(m) for m in mols))
        for g, (i0, i1) in enumerate(split_runs([_n_conf(m) for m in mols], max_conf_per_group)):
            members = mols[i0:i1]
            table.append(write_group(root, members, bucket=b, group=g, fingerprint=fp,
                                     n_types=n_types, with_color_self=with_color_self,
                                     block_rows=block_rows, L_field_width=lfw,
                                     _checked=[checked[id(m)] for m in members]))
    pq_path = os.path.join(root, "molecules.parquet")
    write_molecules(pq_path, ids, [m.ext_id for m in molecules], [m.smiles for m in molecules],
                    [_n_real(m) for m in molecules], [float(m.mw) for m in molecules],
                    [int(m.rotors) for m in molecules])
    descriptor = write_descriptor(
        root, color=color, prep_version=prep_version, prep_commit=prep_commit, groups=table,
        ids_sha256=hashlib.sha256(np.unique(np.asarray(ids, dtype=np.int64))
                                  .astype("<i8").tobytes()).hexdigest())
    if build is not None:
        _write_bytes(os.path.join(root, "build.json"), jsonstrict.dumps(build))
    return descriptor

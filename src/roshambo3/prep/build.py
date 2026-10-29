"""Build a format-3 corpus (``doc/FORMAT.md``) from molecules.

    from roshambo3.prep import prepare
    corpus = prepare(["library.sdf"], "$EXTERNAL_DATA")

Pipeline, in three passes, because the middle one is what makes the third cheap:

1. **Plan.** Split the input into work units of roughly equal record count. For
   an SDF that means indexing the ``$$$$`` delimiters once, sequentially, so a
   split can never land inside a record.

2. **Scatter, in parallel.** Each worker preps its molecules -- heavy atoms in
   canonical SMILES order, centroid, principal-axis frame, int16 quantization
   at the 2^-8 A step, the self-overlap of the quantized coordinates, the
   color feature table -- and appends to shard files keyed on the **exact**
   ``n_real``, not on the bucket. When a work unit closes, each of its shards
   is sorted by ``(mol_id, conf_idx)``. Sharding by the exact count is the
   whole trick: within a bucket, visiting the counts in ascending order and
   merging each count's sorted shards by key *is* the corpus order, so the
   gather never sorts more than one count's metadata. All conformers of a
   molecule share ``n_real``, so they can never be separated by it.

3. **Gather.** Per bucket, merge its shards in ``(n_real, mol_id, conf_idx)``
   order (FORMAT.md section 9.2: a function of content, not of
   worker scheduling) and write the format-3 group files through the format's
   own writer (``format.r3g.writer.GroupWriter``): groups by the write rule
   of FORMAT.md section 9.3 (``max_conf_per_group``), molecule-aligned blocks
   by section 6 rule 5 (``block_rows``), the bit-packed coordinates, the
   molecule run table, the feature table, ``vaa`` and ``color_self``. Then
   ``molecules.parquet`` (sorted by ``mol_id``, with the duplicate-id refusal),
   ``corpus.json`` (the descriptor and the group table) and ``build.json``
   (provenance), and the whole corpus passes the ``scan`` tier
   (``format.r3g.validate``) before it is published.

Why the group cap exists: a screen allocates ``(n_queries, n_library, 20)``
float32 on host and device, which is 80 B per query-conformer, so a 32-query
panel against a 2e6-conformer group is 2.56 GB per allocation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import time
import fcntl
import multiprocessing
import signal
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np

from ..format import constants as fmt
from ..format.constants import (BUCKET_EDGES, MAX_CONF_PER_GROUP,
                                MAX_COLOR_TYPE_COUNT, bucket_of)
from ..format.r3g.limits import LIMITS as _R3G_LIMITS
from ..format.r3g.limits import pad8
from ..format.r3g.writer import GroupWriter  # noqa: F401  (the gather's writer; tests patch it)
from .conformer import make_right_handed, principal_axis_rotation
from .sources import BuildRefused  # noqa: F401  (re-exported: build.BuildRefused)

# Intermediate preparation records, not distributed corpus fields.
SHARD_META_DTYPE = np.dtype(
    [("mol_id", "<i8"), ("conf_idx", "<u2"), ("vaa", "<f4")])

FLUSH_BYTES = 8 << 20

#: ``BuildOptions.feature_order`` values.
FEATURE_ORDERS = ("canonical", "enumeration")

#: The block row cap's default and maximum: FORMAT.md section 6 rule 4 (every
#: block holds at most 4,096 rows), from the format's one statement of it.
BLOCK_ROWS = _R3G_LIMITS["block_rows"]


@dataclass
class BuildOptions:
    name: str = "corpus"
    workers: int = 0
    """0 picks ``min(8, cpu_count)``."""
    records_per_unit: int = 20_000
    max_conf_per_group: int = MAX_CONF_PER_GROUP
    """The group write rule's cap (FORMAT.md section 9.3): a molecule joins
    the open group if it is empty or the group stays within this many
    conformers. Recorded in ``build.json``, never in ``corpus.json``."""
    block_rows: int = BLOCK_ROWS
    """The block write rule's cap (FORMAT.md section 6 rule 5), ``1 <=
    block_rows <= 4096``: a block is cut at the last molecule boundary at or
    before this many rows, and a molecule with more conformers gets a block
    of its own (at most 4,096 rows, the run table's per-molecule limit).
    Changes no score and no fingerprint; recorded in ``build.json``."""
    note: str = ""
    feature_order: str = "canonical"
    """Order of a molecule's color features within a type. ``"canonical"``
    (the format-3 rule, FORMAT.md section 8.1 rule 4): sorted by ``(type,
    slots)``, so the table and the kernel's color slot order are a function
    of the graph and the definition alone. ``"enumeration"`` keeps the
    generator's match order within a type, to measure the within-type sort
    as its own numerics component. Its table is not valid format 3,
    so the format-3 builder refuses it (``prepare``)."""


#: ``n_real`` is bounded at 255 (u8 in the format-3 run table). The bound was
#: once unchecked and did not wrap loudly: a 300-heavy-atom conformer became
#: ``n_real = 44``, the kernel scored 44 of its atoms and reported a confident
#: wrong number, and the re-gather truncated its pose the same way.
#:
#: Conformers per molecule are bounded by the format-3 run table's limit
#: (``n_conf`` u16 with ``1 <= n_conf <= 4096``: FORMAT.md section 7,
#: so every molecule-aligned block holds at most 4,096 rows).
#: It was 256, the format-2 uint8 ``conf_idx`` sidecar's, after a molecule
#: with more produced duplicate ``(mol_id, conf_idx)`` pairs that nothing
#: detected; the shard metadata carries ``conf_idx`` as u2 and the gather
#: refuses a duplicate pair by name.
MAX_N_REAL = _R3G_LIMITS["n_real"]
MAX_CONFORMERS_PER_MOLECULE = _R3G_LIMITS["n_conf_per_molecule"]
#: Color feature table limits (FORMAT.md section 8.1 rules 2 and 5, limits
#: table section 14), from the format's one statement of them.
MAX_FEATURES_PER_MOLECULE = _R3G_LIMITS["features_per_molecule"]
MAX_ATOMS_PER_FEATURE = _R3G_LIMITS["atoms_per_feature"]
#: ``molecules.parquet`` string limits (FORMAT.md section 11, limits 31-32).
MAX_SMILES_BYTES = _R3G_LIMITS["smiles"]
MAX_EXT_ID_BYTES = _R3G_LIMITS["ext_id"]
#: The int16 range of a stored coordinate, in steps (FORMAT.md section 4.1:
#: [-128, +127.99609375] A at 2^-8 A).
_INT16_MIN, _INT16_MAX = -32768, 32767


def _why_not_representable_text(ext_id: str, smiles) -> Optional[str]:
    """Why a molecule's identifier or SMILES cannot go into
    ``molecules.parquet`` (FORMAT.md section 11, limits 31-32), or None."""
    from ..format.r3g.reference import _BIDI, _SMILES_OK
    if not isinstance(ext_id, str) or not ext_id:
        return "it has no identifier (ext_id)"
    if len(ext_id.encode("utf-8", "surrogatepass")) > MAX_EXT_ID_BYTES:
        return (f"its identifier is {len(ext_id.encode('utf-8', 'surrogatepass'))} bytes; "
                f"molecules.parquet holds at most {MAX_EXT_ID_BYTES} (FORMAT.md section 11)")
    if any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F or ord(ch) in _BIDI
           or 0xD800 <= ord(ch) <= 0xDFFF for ch in ext_id):
        return ("its identifier carries a control, bidi or surrogate character, which "
                "molecules.parquet refuses (FORMAT.md section 11)")
    if not isinstance(smiles, str) or not smiles:
        return "it has no SMILES"
    if len(smiles) > MAX_SMILES_BYTES or not _SMILES_OK.match(smiles):
        return (f"its SMILES ({len(smiles)} characters) is not 1..{MAX_SMILES_BYTES} "
                f"bytes of printable ASCII without '|' (FORMAT.md section 11.1)")
    return None


def _why_reject(rec) -> Optional[str]:
    """Why this molecule cannot be stored, or None if it can.

    Non-finite coordinates are the case worth spelling out. Preparation
    tolerates them by design -- ``principal_axis_rotation`` falls back to the
    identity frame so one bad molecule cannot abort a batch -- and the
    quantizer's range guard cannot catch what arrives, because ``NaN > 32767``
    is False. A NaN therefore reached ``astype(np.int16)`` and became a
    plausible-looking atom at -128.0 A. A NaN also makes ``q.max()`` NaN,
    so it hid an Inf in the same conformer that would otherwise have been
    caught. Tolerating the *frame* is right; storing the coordinates is not.
    A coordinate outside the int16 range of the 2^-8 A step is refused
    here too, per molecule, never clipped (FORMAT.md section 4.1).
    """
    n_conf = len(rec.conformers)
    if n_conf == 0:
        return "no conformers"
    if n_conf > MAX_CONFORMERS_PER_MOLECULE:
        return (f"{n_conf} conformers exceeds the run table's limit of "
                f"{MAX_CONFORMERS_PER_MOLECULE} conformers per molecule "
                f"(1 <= n_conf <= {MAX_CONFORMERS_PER_MOLECULE}: FORMAT.md "
                f"section 7)")
    n_reals = {int(c[1]) for c in rec.conformers}
    if len(n_reals) != 1:
        return (f"its conformers disagree on the heavy-atom count {sorted(n_reals)}; "
                f"bucketing keeps a molecule together by that count")
    n_real = n_reals.pop()
    if n_real < 1:
        return "no heavy atoms"
    if n_real > MAX_N_REAL:
        return (f"{n_real} heavy atoms exceeds the {MAX_N_REAL} that the u8 "
                f"n_real of the run table can represent (FORMAT.md section 7)")

    # Color: one feature table per molecule (FORMAT.md section 8.1),
    # its limits checked here, where a violation rejects the molecule
    # rather than wrapping a u8 count or a u8 atom index downstream.
    features = getattr(rec, "features", None)
    if features is not None:
        if len(features) > MAX_FEATURES_PER_MOLECULE:
            return (f"{len(features)} color features exceeds the "
                    f"{MAX_FEATURES_PER_MOLECULE} a molecule can carry "
                    f"(FORMAT.md section 8.1 rule 5)")
        per_type: dict = {}
        for t, atoms in features:
            per_type[t] = per_type.get(t, 0) + 1
            if not 1 <= len(atoms) <= MAX_ATOMS_PER_FEATURE:
                return (f"a color feature on {len(atoms)} atoms; a feature has "
                        f"1 to {MAX_ATOMS_PER_FEATURE} (FORMAT.md section 8.1 "
                        f"rule 2)")
            if max(atoms) >= n_real:
                raise RuntimeError(
                    f"molecule {rec.ext_id or rec.mol_id}: feature atom "
                    f"{max(atoms)} is not one of its {n_real} heavy atoms")
        worst = max(per_type.values(), default=0)
        if worst > MAX_COLOR_TYPE_COUNT:
            return (f"{worst} features of one type exceeds the "
                    f"{MAX_COLOR_TYPE_COUNT} the uint8 per-type count "
                    f"can represent (FORMAT.md section 8.1 rule 5)")

    # The molecule table (FORMAT.md section 11): every column is REQUIRED.
    meta = rec.meta or {}
    why = _why_not_representable_text(rec.ext_id, meta.get("smiles"))
    if why is not None:
        return why
    mw = meta.get("mw")
    if mw is None or not math.isfinite(float(mw)) or float(mw) < 0:
        return f"its molecular weight {mw!r} is not a finite value >= 0"
    rot = meta.get("rotor_count")
    if rot is None or not math.isfinite(float(rot)) or float(rot) != int(float(rot)) \
            or not 0 <= int(float(rot)) <= 65535:
        return f"its rotatable-bond count {rot!r} is not an integer in [0, 65535]"

    for ci, conf in enumerate(rec.conformers):
        r = np.asarray(conf[0])
        if r.shape != (n_real, 3):
            return f"conformer {ci} has shape {r.shape}, expected ({n_real}, 3)"
        if not np.all(np.isfinite(r)):
            return f"conformer {ci} has non-finite coordinates"
        if not np.isfinite(conf[2]):
            return f"conformer {ci} has a non-finite self-overlap"
        q = np.rint(np.asarray(r, dtype=np.float64) / fmt.COORD_SCALE)
        if q.size and (q.max() > _INT16_MAX or q.min() < _INT16_MIN):
            return (f"conformer {ci} has a coordinate at "
                    f"{float(np.abs(r).max()):.3f} A from its centroid, outside the "
                    f"int16 range of the {fmt.COORD_SCALE} A step "
                    f"([-128, 127.996] A; FORMAT.md section 4.1)")
    return None


def molecule_runs(mol_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(start, end)`` of each maximal run of equal ``mol_id``.

    The one definition of "a molecule is a contiguous run of conformers", used
    by the builder's own check and by ``roshambo3 verify``.
    """
    m = np.asarray(mol_id)
    if m.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    start = np.flatnonzero(np.concatenate([[True], m[1:] != m[:-1]]))
    end = np.append(start[1:], m.size)
    return start.astype(np.int64), end.astype(np.int64)


def check_molecule_grouping(where: str, mol_id: np.ndarray, n_real: np.ndarray,
                            previous_ids: Optional[np.ndarray] = None
                            ) -> np.ndarray:
    """Check one group's two contiguity guarantees. Returns its distinct ids.

    Molecule-level results downstream are a segmented reduction over runs of
    equal ``mol_id`` in global id order (see
    ``backend/_cuda_src/molecule_topk.cuh``), so they are correct only while:

    1. **a molecule's conformers are one contiguous run** inside a group, and
    2. **a molecule lies wholly inside one group** -- checked here as
       disjointness from the previous group's ids, which is what the gather's
       "extend to a molecule boundary before closing a group" produces.

    Both were guaranteed by construction and checked nowhere. The failure mode
    is a molecule counted twice in a molecule-level answer, which looks exactly
    like a legitimate result -- so they are asserted while writing, not trusted.
    ``all conformers of a molecule share n_real`` is checked here too: it is
    what keeps bucketing from separating them, and ``_why_reject`` only enforces
    it on the way in.
    """
    mol_id = np.asarray(mol_id)
    n_real = np.asarray(n_real)
    start, end = molecule_runs(mol_id)
    distinct = np.unique(mol_id)
    if start.size != distinct.size:
        raise RuntimeError(
            f"{where}: {start.size} runs of equal mol_id for {distinct.size} "
            f"distinct molecules, so at least one molecule's conformers are not "
            f"contiguous. Every molecule-level result is a segmented reduction "
            f"over those runs and would count such a molecule twice.")
    first = n_real[start]
    if not np.array_equal(n_real, np.repeat(first, end - start)):
        bad = int(np.flatnonzero(n_real != np.repeat(first, end - start))[0])
        raise RuntimeError(
            f"{where}: conformer {bad} disagrees with its molecule's heavy-atom "
            f"count, so bucketing could separate a molecule's conformers")
    if previous_ids is not None and previous_ids.size and distinct.size:
        shared = np.intersect1d(previous_ids, distinct, assume_unique=True)
        if shared.size:
            raise RuntimeError(
                f"{where}: {shared.size} molecule(s) also appear in the "
                f"previous group (first {int(shared[0])}), so a molecule "
                f"straddles a group boundary. The gather is supposed to extend "
                f"a group to a molecule boundary before closing it.")
    return distinct


def feature_mol_dtype(n_types: int) -> np.dtype:
    """One molecule's row of a color shard's ``.fmol`` file: its id, how many
    feature atoms it has, and its per-type feature counts (``counts[t - 1]``
    for type ``t``). Packed: ``10 + n_types`` bytes."""
    return np.dtype([("mol_id", "<i8"), ("n_atoms", "<u2"),
                     ("counts", "u1", (int(n_types),))])


def _exclusive_cumsum(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    out = np.zeros(a.size, dtype=np.int64)
    if a.size > 1:
        np.cumsum(a[:-1], out=out[1:])
    return out


def _ragged_take(values: np.ndarray, lengths, order) -> np.ndarray:
    """The segments of ``values`` (consecutive, of ``lengths``) in ``order``,
    concatenated."""
    lengths = np.asarray(lengths, dtype=np.int64)
    order = np.asarray(order, dtype=np.int64)
    take = lengths[order]
    gather = (np.repeat(_exclusive_cumsum(lengths)[order] - _exclusive_cumsum(take), take)
              + np.arange(int(take.sum()), dtype=np.int64))
    return values[gather]


class _ShardSet:
    """Per-worker shard files, one set per exact ``n_real``.

    Every shard has ``.xyz`` (int16 coordinates, planar within a conformer)
    and ``.met`` (``SHARD_META_DTYPE`` per conformer). A color build adds the
    molecule's **feature table**, once per molecule, never a coordinate
    (FORMAT.md section 8): ``.fmol`` (``feature_mol_dtype``: id, feature-atom
    count, per-type counts), ``.fnat`` (u8 atoms per feature) and ``.fatm``
    (u8 slot indices), the features of each molecule grouped by ascending type
    (so a feature's type is implied by the counts). Pseudo-atom positions are
    derived from the stored heavy atoms at gather, exactly as a reader derives
    them (``derive_feature_positions``).
    """

    def __init__(self, root: str, slot: int, n_types: Optional[int] = None,
                 feature_order: str = "canonical"):
        self.dir = root
        os.makedirs(root, exist_ok=True)
        self.slot = slot
        if feature_order not in FEATURE_ORDERS:
            raise ValueError(f"feature_order must be one of {FEATURE_ORDERS}")
        self.feature_order = feature_order
        self.n_types = None if n_types is None else int(n_types)
        self.fmol_dtype = (feature_mol_dtype(self.n_types)
                           if self.n_types is not None else None)
        self.buf: dict = {}
        self.pending = 0
        self.tsv = open(os.path.join(root, f"meta_{slot:05d}.tsv"), "w")
        self.n_conf = 0
        self.n_mol = 0
        self.rejected: list[tuple[str, str]] = []
        self.has_color: Optional[bool] = None
        self.written: set = {os.path.join(root, f"meta_{slot:05d}.tsv")}

    def _feature_rows(self, rec, n_real: int):
        """The molecule's ``.fmol`` row and its ``.fnat`` / ``.fatm`` arrays."""
        features = list(rec.features)
        if self.feature_order == "canonical":
            features.sort()                    # FORMAT.md 8.1 rule 4
        types = np.asarray([t for t, _ in features], dtype=np.int64)
        if types.size and (int(types.min()) < 1 or int(types.max()) > self.n_types
                           or np.any(np.diff(types) < 0)):
            raise RuntimeError(
                f"molecule {rec.ext_id or rec.mol_id}: feature types "
                f"{types.tolist()} are not grouped in 1..{self.n_types}")
        nat = np.asarray([len(a) for _, a in features], dtype=np.uint8)
        atoms = np.asarray([x for _, a in features for x in a], dtype=np.uint8)
        counts = np.bincount(types, minlength=self.n_types + 1)[1:]
        row = np.zeros(1, dtype=self.fmol_dtype)
        row["mol_id"], row["n_atoms"], row["counts"][0] = rec.mol_id, atoms.size, counts
        return row, nat, atoms

    def add(self, rec) -> bool:
        """Accept a molecule, or reject it whole. Returns whether it was kept.

        All-or-nothing per *molecule*, never per conformer: bucketing keeps a
        molecule's conformers together because they share ``n_real``, and
        molecule-level results downstream depend on that. Half a molecule in
        the corpus would break both quietly.
        """
        why = _why_reject(rec)
        if why is not None:
            self.rejected.append((rec.ext_id or str(rec.mol_id), why))
            return False
        has_color = getattr(rec, "features", None) is not None
        if self.has_color is None:
            self.has_color = has_color
        elif self.has_color != has_color:
            # _why_reject only checks agreement *within* one molecule; this
            # is the stronger property the gather phase relies on -- one
            # source is color-built or it is not, never a mix.
            raise RuntimeError(
                "some molecules carry color and others do not within one "
                "build -- a color_generator must be given for every "
                "molecule or for none")
        if has_color and self.n_types is None:
            raise RuntimeError("a color molecule reached a shape-only shard set")
        if not 0 <= int(rec.mol_id) < _R3G_LIMITS["mol_id"]:
            # Every source derives mol_id = stable_id(ext_id), which is in
            # range; an id outside it means the identity scheme itself broke,
            # so the build stops rather than storing it (FORMAT.md section 7,
            # limit 35; the id gate's int64 hash needs it).
            raise BuildRefused(
                f"molecule {rec.ext_id!r} has mol_id {int(rec.mol_id)}, outside "
                f"[0, 2^63): format 3 stores mol_id as u64 below 2^63 (FORMAT.md "
                f"section 7, limit 35) and the id hash serializes it as int64")
        from .conformer import stored_self_volume
        self.n_mol += 1
        n_real = int(rec.conformers[0][1])
        b = self.buf.setdefault(n_real, ([], [], [], [], []))
        for ci, conf in enumerate(rec.conformers):
            r = conf[0]
            q = fmt.quantize(r)                    # [n, 3] int16
            # vaa describes the stored coordinates, the ones every reader
            # scores (FORMAT.md section 18.1): format 3 has one frame.
            vaa = stored_self_volume(q.astype(np.float32) * np.float32(fmt.COORD_SCALE))
            b[0].append(np.ascontiguousarray(q.T))   # [3, n] planar within a conformer
            b[1].append((rec.mol_id, ci, vaa))
            self.pending += 6 * n_real + SHARD_META_DTYPE.itemsize
            self.n_conf += 1
        if has_color:
            row, nat, atoms = self._feature_rows(rec, n_real)
            b[2].append(row)
            b[3].append(nat)
            b[4].append(atoms)
            self.pending += row.nbytes + nat.size + atoms.size
        # molecules.parquet's row (FORMAT.md section 11); ``_why_reject``
        # has checked that each field is representable. ``hac`` is n_real by
        # definition; the identifier is JSON-quoted so no character of it can
        # break the line (a SMILES is printable ASCII).
        m = rec.meta
        self.tsv.write("%d\t%s\t%d\t%r\t%d\t%s\n" % (
            rec.mol_id, json.dumps(rec.ext_id), n_real, float(m["mw"]),
            int(float(m["rotor_count"])), m["smiles"]))
        if self.pending >= FLUSH_BYTES:
            self.flush()
        return True

    def flush(self) -> None:
        for n_real, (xs, ms, fmols, nats, atoms) in self.buf.items():
            if not xs:
                continue
            handles = self._files(n_real)
            try:
                np.stack(xs).tofile(handles[0])
                np.array(ms, dtype=SHARD_META_DTYPE).tofile(handles[1])
                if self.has_color:
                    np.concatenate(fmols).tofile(handles[2])
                    np.concatenate(nats).astype(np.uint8).tofile(handles[3])
                    np.concatenate(atoms).astype(np.uint8).tofile(handles[4])
                for part in (xs, ms, fmols, nats, atoms):
                    part.clear()
            finally:
                for handle in handles:
                    if handle is not None:
                        handle.close()
        self.pending = 0

    def _files(self, n_real):
        # Append and close on each flush: descriptors do not scale with the
        # number of distinct atom counts a worker sees.
        base = os.path.join(self.dir, f"n{n_real:03d}_s{self.slot:05d}")
        exts = [".xyz", ".met"] + ([".fmol", ".fnat", ".fatm"] if self.has_color else [])
        handles = []
        try:
            for ext in exts:
                handles.append(open(base + ext, "ab"))
            self.written.update(h.name for h in handles)
            return tuple(handles)
        except BaseException:
            for handle in handles:
                handle.close()
            raise

    def close(self, sort: bool = True) -> tuple[int, int]:
        """Flush, sort (unless the unit failed: its files are never
        committed) and close."""
        try:
            self.flush()
            if sort:
                self._sort_shards()
        finally:
            self.tsv.close()
        return self.n_mol, self.n_conf

    def _sort_shards(self) -> None:
        """Rewrite each of this unit's ``n_real`` shards in ``(mol_id,
        conf_idx)`` order (FORMAT.md section 9.2).

        A unit's shards hold at most one unit's molecules and were appended
        flush by flush in the order the source emitted them; sorted here, at
        close, the gather can merge a bucket's shards by key instead of
        concatenating them in slot order -- a function of scheduling. One
        stable permutation of the rows rewrites ``.met`` and ``.xyz``; on a
        color build one stable permutation of the molecules, by ``mol_id``,
        rewrites the feature table (``.fmol`` and the ragged ``.fnat`` and
        ``.fatm``). Runs before ``sync``, so the committed slot is the sorted
        one; a rewrite goes through a sibling file and ``os.replace``, so an
        interruption leaves either file, both uncommitted.
        """
        for n_real in sorted(self.buf):
            base = os.path.join(self.dir, f"n{n_real:03d}_s{self.slot:05d}")
            if not os.path.exists(base + ".met"):
                continue
            met = np.fromfile(base + ".met", dtype=SHARD_META_DTYPE)
            n = met.size
            if n == 0:
                continue
            rewritten = {}
            perm = np.lexsort((met["conf_idx"], met["mol_id"]))   # stable
            if not np.array_equal(perm, np.arange(n)):
                xyz = np.fromfile(base + ".xyz", dtype="<i2").reshape(n, 3 * n_real)
                rewritten.update({".met": met[perm], ".xyz": xyz[perm]})
            if self.has_color:
                fmol = np.fromfile(base + ".fmol", dtype=self.fmol_dtype)
                order = np.argsort(fmol["mol_id"], kind="stable")
                if not np.array_equal(order, np.arange(fmol.size)):
                    n_feat = fmol["counts"].sum(axis=1, dtype=np.int64)
                    nat = np.fromfile(base + ".fnat", dtype="u1")
                    atoms = np.fromfile(base + ".fatm", dtype="u1")
                    if nat.size != int(n_feat.sum()) or atoms.size != int(
                            fmol["n_atoms"].sum(dtype=np.int64)):
                        raise RuntimeError(
                            f"{base}: the feature table's sizes disagree with "
                            f"its molecule rows")
                    rewritten.update({
                        ".fmol": fmol[order],
                        ".fnat": _ragged_take(nat, n_feat, order),
                        ".fatm": _ragged_take(atoms, fmol["n_atoms"], order)})
            for ext, values in rewritten.items():
                tmp = base + ext + ".sorting"
                np.ascontiguousarray(values).tofile(tmp)
                os.replace(tmp, base + ext)

    def sync(self) -> None:
        """Make this slot's files durable before a checkpoint names the slot
        complete: a resumed build trusts committed slots without re-reading."""
        paths = set(self.written)
        for path in sorted(paths):
            if os.path.exists(path):
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)


def _source_shard_set(tmp_dir: str, slot: int) -> "_ShardSet":
    """A scatter worker's shard set, configured from the source the build
    attached its options to (``prepare``)."""
    generator = getattr(_SOURCE, "color_generator", None)
    return _ShardSet(tmp_dir, slot, n_types=(len(generator.FEATURES_ENUM)
                              if generator is not None else None),
                     feature_order=getattr(_SOURCE, "feature_order", "canonical"))


_SOURCE = None
_PAYLOADS = None


def _worker_init(source):
    global _SOURCE
    _SOURCE = source


#: How often a scatter worker checks that the process that started it is
#: still there.
WORKER_PARENT_POLL_S = 1.0


def _pool_worker_init(source, payloads=None):
    # A forked worker inherits the parent's SIGTERM handler, which would turn
    # a termination into an exception mid-unit; workers just die.
    global _PAYLOADS
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _exit_with_parent()
    _PAYLOADS = payloads
    _worker_init(source)


def _exit_with_parent() -> None:
    """Exit once the process that started this worker is gone.

    An executor worker blocks on its task queue, whose write end every forked
    sibling also holds, so the parent's death (``kill -9``, an OOM kill)
    never reaches it as end-of-file: without this, orphaned workers would sit
    forever, holding their memory and any pipe the parent's caller reads.
    The parent's death reparents the worker at once (even while the parent is
    an unreaped zombie), so a changed ``getppid()`` is the signal. Under
    forkserver the parent is the fork server, which exits with the build.
    """
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(WORKER_PARENT_POLL_S)
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


#: The multiprocessing context of the scatter's workers; None is
#: multiprocessing's default. Tests set a forkserver context to check that
#: the source arrives pickled.
SCATTER_MP_CONTEXT = None


class WorkerDied(RuntimeError):
    """A scatter worker died (killed, most often by the kernel's OOM killer).
    Its work unit is lost; committed units are kept, so resuming continues."""
    retryable = True


class BuildTerminated(SystemExit):
    """SIGTERM, handled as Ctrl-C is: the build is recorded ``paused`` and
    the workers stopped. Exits with status 143 (128 + SIGTERM)."""

    def __init__(self, signum: int = signal.SIGTERM):
        super().__init__(128 + int(signum))
        self.signum = int(signum)

    def __str__(self):
        return f"terminated by {signal.Signals(self.signum).name}"


#: The signal :func:`_terminate_as_exit` turned into :class:`BuildTerminated`
#: during the running build, or None.
_TERMINATED_BY: dict = {"signum": None}


@contextmanager
def _terminate_as_exit():
    """Turn SIGTERM into :class:`BuildTerminated` while a build runs (main
    thread only; elsewhere signals cannot be handled and nothing changes)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        _TERMINATED_BY["signum"] = signum
        raise BuildTerminated(signum)

    _TERMINATED_BY["signum"] = None
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        _TERMINATED_BY["signum"] = None
        # None: the previous handler was not installed from Python; the
        # default is the closest thing to restore.
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


class _ScatterPool:
    """The scatter's worker processes.

    A ``ProcessPoolExecutor``, not a ``multiprocessing.Pool``: a ``Pool`` whose
    worker dies never completes that worker's task and says nothing (it starts
    a replacement), so the parent waited forever; and its ``terminate()``
    itself hangs when the dead worker held the task queue's lock, as an idle
    worker does. The executor marks itself broken the moment a worker dies,
    fails every outstanding future with ``BrokenProcessPool`` and cleans up
    without that lock.

    The worker processes are tracked through the public
    ``multiprocessing.active_children`` (the processes started since this pool
    was made), so an abort can stop them at once instead of letting each
    finish its unit, and a death can be reported by pid and signal.
    """

    def __init__(self, n_workers: int, source, payloads: bool = False):
        from concurrent.futures import ProcessPoolExecutor
        self._before = {p.pid for p in multiprocessing.active_children()}
        self._seen: dict = {}
        # Large work-unit inputs (the binary stream's raw batches, ~127 MiB
        # each on the largest corpus) travel through this queue, not as task arguments: the
        # executor keeps a task's arguments until its result arrives, which
        # would hold every in-flight batch in the parent (24 workers: ~3 GB);
        # the queue lets go of each once it is written to a worker.
        context = SCATTER_MP_CONTEXT or multiprocessing.get_context()
        self._payloads = context.Queue() if payloads else None
        self._executor = ProcessPoolExecutor(
            n_workers, mp_context=SCATTER_MP_CONTEXT,
            initializer=_pool_worker_init, initargs=(source, self._payloads))

    def _track(self) -> None:
        for p in multiprocessing.active_children():
            if p.pid not in self._before:
                self._seen.setdefault(p.pid, p)

    @property
    def workers(self) -> list:
        self._track()
        return list(self._seen.values())

    def submit(self, fn, arg, payload=None):
        """Run ``fn(arg)`` in a worker; with a ``payload``, the task's worker
        takes the next payload from the queue (``_next_payload``) instead.
        Tasks are submitted before their payloads are queued, so (under
        fork) workers are forked before the queue's feeder thread exists."""
        future = self._executor.submit(fn, arg)
        self._track()
        if payload is not None:
            self._payloads.put(payload)
        return future

    def result(self, future):
        from concurrent.futures.process import BrokenProcessPool
        try:
            return future.result()
        except BrokenProcessPool as e:
            raise WorkerDied(self._death_report()) from e

    def _death_report(self) -> str:
        self._track()
        for p in self._seen.values():
            p.join(5)
        codes = {p.pid: p.exitcode for p in self._seen.values()}
        # The executor terminates the survivors (SIGTERM) once it is broken,
        # so the victim is the one that died another way.
        victims = [pid for pid, code in codes.items()
                   if code is not None and code != -signal.SIGTERM] or list(codes)
        how = []
        for pid in victims[:4]:
            code = codes[pid]
            how.append(f"pid {pid} was killed by {signal.Signals(-code).name}"
                       if code is not None and code < 0 and -code in signal.valid_signals()
                       else f"pid {pid} exited with status {code}")
        return (f"a scatter worker died ({'; '.join(how)}; the kernel's OOM "
                f"killer sends SIGKILL). Its work unit is lost and the pool "
                f"cannot recover it. Committed work units are kept: resume "
                f"with --resume (resume=True).")

    def close(self, abort: bool) -> None:
        if abort:
            # Stop at once rather than wait for every running unit: kill the
            # workers (the executor then sees a broken pool and cancels the
            # rest) and wait until they are gone, so nothing they hold -- the
            # output's writer lock among it, inherited across fork -- outlives
            # this call.
            workers = self.workers
            for p in workers:
                if p.is_alive():
                    p.kill()
            for p in workers:
                p.join(10)
        if self._payloads is not None:
            if abort:
                # Nobody will read what is left; do not wait to flush it.
                self._payloads.cancel_join_thread()
            self._payloads.close()
            self._payloads.join_thread()
        self._executor.shutdown(wait=True, cancel_futures=abort)


def _scatter_unit(args):
    slot, unit, tmp_dir = args
    shards = _source_shard_set(tmp_dir, slot)
    try:
        for rec in _SOURCE.iter_unit(unit):
            shards.add(rec)
    except BaseException:
        shards.close(sort=False)
        raise
    # Chemistry the source could not process at all -- rejected per molecule
    # rather than fatally, and reported alongside the layout rejections so the
    # build record's reject count means "did not reach the corpus" and not merely
    # "was rejected by _why_reject".
    rejected = list(getattr(_SOURCE, "failures", ())) + list(shards.rejected)
    extra = {"content_sha256": getattr(_SOURCE, "unit_sha256", None),
             "dropped_conformers": dict(getattr(_SOURCE, "dropped_conformers", None) or {})}
    n_mol, n_conf = shards.close()
    shards.sync()
    return slot, n_mol, n_conf, rejected, extra


def _scatter_next_batch(tmp_dir):
    """A queued raw batch (see ``_ScatterPool``), prepared by this worker."""
    slot, first_record, blobs = _PAYLOADS.get()
    return _scatter_binary_batch((slot, first_record, blobs, tmp_dir))


def _scatter_binary_batch(args):
    """Deserialize raw binary records inside the worker that writes its shard."""
    from rdkit import Chem
    from .sources import _group_by_name, payload_digest

    slot, first_record, blobs, tmp_dir = args
    shards = _source_shard_set(tmp_dir, slot)
    failures = []
    dropped: dict = {}
    try:
        for i, blob in enumerate(blobs):
            record = first_record + i
            try:
                mol = Chem.Mol(blob)
            except Exception as e:
                failures.append((f"_record_{record}", f"{type(e).__name__}: {e}"))
                continue
            if mol is None:
                failures.append((f"_record_{record}", "RDKit returned None for binary payload"))
                continue
            for rec in _group_by_name([mol], record, _SOURCE.color_generator, failures,
                                     contiguous_conformer_ids=True, dropped=dropped):
                shards.add(rec)
    except BaseException:
        shards.close(sort=False)
        raise
    n_mol, n_conf = shards.close()
    shards.sync()
    return (slot, n_mol, n_conf, failures + list(shards.rejected),
            {"content_sha256": payload_digest(blobs), "dropped_conformers": dropped})


class _Shard(NamedTuple):
    """One committed scatter shard: ``base`` + ``.xyz`` / ``.met`` (and, on a
    color build, ``.fmol`` / ``.fnat`` / ``.fatm``), ``n`` conformers."""
    base: str
    n: int
    color: bool

    @property
    def xyz(self) -> str:
        return self.base + ".xyz"

    @property
    def met(self) -> str:
        return self.base + ".met"


def _shard_index(tmp_dir: str, n_types: Optional[int] = None
                 ) -> tuple[dict, Optional[int]]:
    """``{n_real: [_Shard, ...]}``, in slot order (each shard sorted by
    ``(mol_id, conf_idx)``; the gather merges them, ``_shard_chunks``), and
    ``n_types`` -- the build's number of color feature types, ``None`` for a
    shape-only build -- echoed back.

    A shard that disagrees with itself or with the build is refused, not
    retried: the shards read here are committed, and a resume trusts them
    again."""
    idx: dict = {}
    for fn in sorted(os.listdir(tmp_dir)):
        if not fn.endswith(".met"):
            continue
        base = os.path.join(tmp_dir, fn[:-4])
        n_real = int(fn[1:4])
        n = os.path.getsize(base + ".met") // SHARD_META_DTYPE.itemsize
        if n == 0:
            continue
        expect = n * 6 * n_real
        got = os.path.getsize(base + ".xyz")
        if got != expect:
            raise BuildRefused(
                f"shard {base} holds {got} coordinate bytes but its metadata "
                f"says {n} conformers at n_real={n_real} ({expect} bytes)")
        has_color = os.path.exists(base + ".fmol")
        if has_color != (n_types is not None):
            raise BuildRefused(
                f"shard {base} {'carries' if has_color else 'lacks'} a color "
                f"feature table, but this build is "
                f"{'shape-only' if n_types is None else 'a color build'}")
        if has_color:
            size = os.path.getsize(base + ".fmol")
            if size % feature_mol_dtype(n_types).itemsize:
                raise BuildRefused(
                    f"shard {base}.fmol is {size} bytes, not a whole number of "
                    f"{n_types}-type molecule rows")
        idx.setdefault(n_real, []).append(_Shard(base, n, has_color))
    return idx, n_types


class _FeatureRows:
    """Where one ``n_real``'s rows find their molecule's feature table:
    ``shard_no`` and ``mol`` per merged row (the molecule's index in its
    shard's ``.fmol``), and per shard the molecules' first feature and first
    feature atom (``feature_start``, ``atom_start``, each with a final total)."""

    def __init__(self, shards, shard_no, mol, feature_start, atom_start):
        self.shards = shards
        self.shard_no = shard_no
        self.mol = mol
        self.feature_start = feature_start
        self.atom_start = atom_start

    def take(self, sl: slice) -> "_FeatureRows":
        return _FeatureRows(self.shards, self.shard_no[sl], self.mol[sl],
                            self.feature_start, self.atom_start)


def _merged_rows(n_real: int, shards: list, n_types: Optional[int]):
    """Every row of one ``n_real`` across ``shards``, in ``(mol_id,
    conf_idx)`` order: ``(shard_no, offset, met, counts, features)``, each
    indexed by merged position (``counts`` the ``[n, n_types]`` per-type
    feature counts and ``features`` a ``_FeatureRows``; both ``None`` on a
    shape-only build).

    Each shard is already sorted (``_ShardSet._sort_shards``; checked here),
    so this is a k-way merge of sorted runs. It is computed as one stable
    ``lexsort`` of the concatenated keys, which yields exactly the merge's
    order -- ties keep shard (slot) order, and within a shard rows keep
    ascending offsets -- and runs vectorized, where a Python heap would pay a
    microsecond per row. Memory is bounded by one ``n_real``'s rows, never
    the corpus's, but not small: the merged metadata, offsets, shard numbers
    and counts retain about 36 B per row, and the concatenate-and-permute
    peaks near 86 B per row on a color build with 6 types (measured): about 2.7 GiB for the largest ``n_real`` of a 540M-row
    corpus, reached twice (planning pass and writing pass). The feature
    table adds 8 B per row (its molecule index) and 16 B per molecule.

    Per-type counts are a property of the molecule (its ``.fmol`` row),
    expanded here to the molecule's rows.
    """
    mets, counts, shard_no, offset, mols = [], [], [], [], []
    feature_start, atom_start = [], []
    for k, shard in enumerate(shards):
        mp, n = shard.met, shard.n
        met = np.fromfile(mp, dtype=SHARD_META_DTYPE, count=n)
        if met.size != n:
            raise BuildRefused(f"shard {mp} holds {met.size} rows, its index {n}")
        if n > 1:
            mid, cix = met["mol_id"], met["conf_idx"]
            if not np.all((mid[1:] > mid[:-1])
                          | ((mid[1:] == mid[:-1]) & (cix[1:] >= cix[:-1]))):
                raise BuildRefused(
                    f"shard {mp} is not in (mol_id, conf_idx) order; every "
                    f"committed shard is sorted when its work unit closes")
        mets.append(met)
        shard_no.append(np.full(n, k, dtype=np.int64))
        offset.append(np.arange(n, dtype=np.int64))
        if shard.color:
            fmol = np.fromfile(shard.base + ".fmol", dtype=feature_mol_dtype(n_types))
            fid = fmol["mol_id"]
            if fid.size > 1 and np.any(fid[1:] <= fid[:-1]):
                raise BuildRefused(
                    f"shard {shard.base}.fmol is not in ascending molecule order "
                    f"or names molecule {int(fid[1:][fid[1:] <= fid[:-1]][0])} "
                    f"twice: two input records carry one molecule id (or two "
                    f"ids collide). Give each molecule its own id.")
            mi = np.searchsorted(fid, met["mol_id"])
            if (np.any(mi >= fid.size) or np.any(fid[np.minimum(mi, fid.size - 1)]
                                                 != met["mol_id"])
                    or np.unique(mi).size != fid.size):
                raise BuildRefused(
                    f"shard {shard.base}: its feature table and its conformer "
                    f"rows name different molecules")
            n_feat = fmol["counts"].sum(axis=1, dtype=np.int64)
            fs = np.concatenate([[0], np.cumsum(n_feat)])
            at = np.concatenate([[0], np.cumsum(fmol["n_atoms"], dtype=np.int64)])
            if (os.path.getsize(shard.base + ".fnat") != int(fs[-1])
                    or os.path.getsize(shard.base + ".fatm") != int(at[-1])):
                raise BuildRefused(
                    f"shard {shard.base}: its feature files' sizes disagree "
                    f"with its molecule rows")
            counts.append(fmol["counts"][mi])
            mols.append(mi.astype(np.int64))
            feature_start.append(fs)
            atom_start.append(at)
    met = np.concatenate(mets)
    perm = np.lexsort((met["conf_idx"], met["mol_id"]))
    met = met[perm]
    shard_no = np.concatenate(shard_no)[perm]
    features = None
    if counts:
        features = _FeatureRows(shards, shard_no, np.concatenate(mols)[perm],
                                feature_start, atom_start)
    return (shard_no, np.concatenate(offset)[perm], met,
            np.concatenate(counts)[perm] if counts else None, features)


def derive_feature_positions(xyz: np.ndarray, row_feature_start, row_n_features,
                             n_atoms, atom_start, atoms,
                             batch_rows: int = 1 << 15) -> np.ndarray:
    """Pseudo-atom positions derived from stored heavy atoms, every row's
    features in row order: ``int16 [sum(row_n_features), 3]``.

    The format-3 rule (FORMAT.md section 8.3), per axis: the
    **integer mean of the features' atoms' int16 coordinates, rounded half
    away from zero**, ``sign(s) * ((2|s| + k) // (2k))``; a one-atom feature
    copies its atom exactly. Integer arithmetic only, so it is the same
    everywhere; ``format.r3g.reference.derive_positions`` is the oracle.

    ``xyz`` is ``int16 [n_rows, 3, >= n_real]`` stored coordinates (step
    units). Row ``r``'s features are ``row_feature_start[r] ..
    + row_n_features[r]`` of a feature table given as ``n_atoms`` (atoms per
    feature), ``atom_start`` (each feature's first index into ``atoms``) and
    ``atoms`` (slot indices).
    """
    row_feature_start = np.asarray(row_feature_start, dtype=np.int64)
    row_n_features = np.asarray(row_n_features, dtype=np.int64)
    n_atoms = np.asarray(n_atoms)
    atom_start = np.asarray(atom_start, dtype=np.int64)
    out = np.empty((int(row_n_features.sum()), 3), dtype=np.int16)
    done = 0
    for r0 in range(0, row_n_features.size, batch_rows):
        cnt = row_n_features[r0:r0 + batch_rows]
        n_rf = int(cnt.sum())
        if n_rf == 0:
            continue
        rows = np.repeat(np.arange(r0, r0 + cnt.size, dtype=np.int64), cnt)
        feat = (np.repeat(row_feature_start[r0:r0 + cnt.size] - _exclusive_cumsum(cnt), cnt)
                + np.arange(n_rf, dtype=np.int64))
        k = n_atoms[feat].astype(np.int64)
        if int(k.min()) < 1:
            raise RuntimeError("a color feature without atoms")
        first = _exclusive_cumsum(k)
        member = atoms[np.repeat(atom_start[feat] - first, k)
                       + np.arange(int(k.sum()), dtype=np.int64)].astype(np.intp)
        values = xyz[np.repeat(rows, k), :, member].astype(np.int64)      # [A, 3]
        s = np.add.reduceat(values, first, axis=0)                          # [n_rf, 3]
        kk = k[:, None]
        q = (2 * np.abs(s) + kk) // (2 * kk)
        out[done:done + n_rf] = np.where(s < 0, -q, q)
        done += n_rf
    return out


def _refuse_duplicate_conformer(n_real: int, shards: list, shard_no, met,
                                at: int) -> None:
    """A ``(mol_id, conf_idx)`` met twice in the merged order: refused by name.

    A merge joins rows of two work units into one run, which
    ``check_molecule_grouping`` cannot tell from one molecule's, and the
    metadata merge only sees duplicate *molecule* ids; this is the per-
    conformer check."""
    mol_id, conf_idx = int(met["mol_id"][at]), int(met["conf_idx"][at])
    paths = sorted({shards[int(shard_no[at - 1])].met, shards[int(shard_no[at])].met})
    names = []
    for path in paths:
        slot = _slot_of(os.path.basename(path))
        if slot is not None:
            names += _slot_names(os.path.dirname(path), slot, {mol_id})
    raise BuildRefused(
        f"conformer (mol_id {mol_id}, conf_idx {conf_idx}) of molecule "
        f"{sorted(set(names)) or mol_id} appears twice at n_real={n_real} "
        f"(shards {', '.join(os.path.basename(p) for p in paths)}): two input "
        f"records carry one molecule id (or two ids collide), so its "
        f"conformers cannot be told apart. Give each molecule its own id.")


def _shard_chunks(idx: dict, members: list, n_types: Optional[int],
                  read_chunk: int):
    """Walk a bucket's shards in corpus order, yielding METADATA chunks only.

    Corpus order within a bucket is ``(n_real, mol_id, conf_idx)``, a
    function of content (FORMAT.md section 9.2): the shards of
    each ``n_real`` are merged by key (``_merged_rows``), not concatenated in
    slot order. Each chunk is ``(n_real, segments, met, counts_chunk,
    features)``: ``met`` is the chunk's ``SHARD_META_DTYPE`` rows,
    ``counts_chunk`` its ``[cnt, n_types]`` per-type feature counts and
    ``features`` its rows' ``_FeatureRows`` (both ``None`` on a shape-only
    build), all in merged order, and ``segments`` the ordered list of
    ``(shard, off, cnt)`` contiguous row ranges of single shards that make up
    the chunk -- a chunk interleaves shards, so the writer gathers its
    payload segment by segment, in this order. The merge consumes every shard
    in ascending offset order, so a reader that takes every segment in order,
    skipping none, can keep a running per-shard offset into a ragged file.

    Two passes need this order: the planner, which decides where the group
    boundaries fall and how wide each group is, and the writer, which then
    fills them. **They must agree exactly** -- a chunk boundary moves where a
    molecule-boundary extension can reach, and a plan derived from a different
    chunking would size a group's record for conformers it does not hold. One
    generator, called twice, rather than two loops that have to match.

    A molecule never straddles a chunk (a chunk is cut back to the last
    molecule boundary, or extended to the molecule's end if it has none). A
    duplicate ``(mol_id, conf_idx)`` is refused, naming it.

    It reads 14 B/conformer of ``.met`` plus ``10 + n_types`` bytes per
    molecule of ``.fmol`` and never opens a ``.xyz``, so the planning pass
    costs a metadata scan against the payload's ``6 * n_real`` bytes per
    conformer.
    """
    for n_real in members:
        shards = idx[n_real]
        shard_no, offset, met, counts, features = _merged_rows(n_real, shards, n_types)
        n = met.size
        mol_id = met["mol_id"]
        dup = np.flatnonzero((mol_id[1:] == mol_id[:-1])
                             & (met["conf_idx"][1:] == met["conf_idx"][:-1]))
        if dup.size:
            _refuse_duplicate_conformer(n_real, shards, shard_no, met, int(dup[0]) + 1)
        pos = 0
        while pos < n:
            end = min(pos + read_chunk, n)
            if end < n and mol_id[end] == mol_id[end - 1]:
                # never let a molecule straddle a chunk boundary
                chg = np.flatnonzero(mol_id[pos + 1:end] != mol_id[pos:end - 1])
                if chg.size:
                    end = pos + int(chg[-1]) + 1
                else:
                    rest = np.flatnonzero(mol_id[end:] != mol_id[end - 1])
                    end = end + int(rest[0]) if rest.size else n
            s, o = shard_no[pos:end], offset[pos:end]
            cuts = np.flatnonzero((s[1:] != s[:-1]) | (o[1:] != o[:-1] + 1)) + 1
            starts = np.concatenate([[0], cuts])
            stops = np.concatenate([cuts, [end - pos]])
            segments = [(shards[int(s[a])], int(o[a]), int(b - a))
                        for a, b in zip(starts, stops)]
            yield (n_real, segments, met[pos:end],
                   counts[pos:end] if counts is not None else None,
                   features.take(slice(pos, end)) if features is not None else None)
            pos = end


def _plan_bucket(idx: dict, members: list, n_types: Optional[int],
                 max_conf_per_group: int, read_chunk: int, heartbeat=None) -> list[dict]:
    """Where this bucket's groups end: one ``{"n_conf", "n_mol"}`` per group.

    The group write rule of FORMAT.md section 9.3, which is normative for
    writers: walk the bucket's molecules in corpus order; a molecule joins
    the open group if the group is empty or ``n_conf + n_conf(molecule) <=
    max_conf_per_group``, else the group closes and the molecule opens the
    next. A molecule therefore never straddles two groups, and two builds
    with the same ``max_conf_per_group`` cut the same groups. Computed from
    the shard metadata alone (``_shard_chunks``, never a ``.xyz``) before
    anything is written, so a resumed gather can check its committed groups
    against it and know exactly which molecules they hold.
    """
    plan: list[dict] = []
    cur_conf = cur_mol = 0
    for _, _, met, _, _ in _shard_chunks(idx, members, n_types, read_chunk):
        if heartbeat is not None:
            heartbeat()
        _, n_conf = _chunk_molecules(met)
        for nc in n_conf.tolist():
            if cur_mol and cur_conf + nc > max_conf_per_group:
                plan.append({"n_conf": cur_conf, "n_mol": cur_mol})
                cur_conf = cur_mol = 0
            cur_conf += nc
            cur_mol += 1
    if cur_mol:
        plan.append({"n_conf": cur_conf, "n_mol": cur_mol})
    # A bucket with conformers always yields at least one group; a bucket
    # without them never reaches here (`_shard_index` drops empty shards).
    return plan


def _chunk_molecules(met: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(first_row, n_conf)`` of each molecule of a chunk. A chunk never
    splits a molecule (``_shard_chunks``), and its rows are in ``(mol_id,
    conf_idx)`` order, so each molecule is one run whose ``conf_idx`` counts
    0, 1, ...: checked, because the run table stores ``n_conf`` alone and a
    reader recovers ``conf_idx`` as the offset in the run (FORMAT.md section 7 rule 1)."""
    mid = met["mol_id"]
    first = np.flatnonzero(np.concatenate([[True], mid[1:] != mid[:-1]]))
    n_conf = np.diff(np.append(first, mid.size))
    expect = np.arange(mid.size) - np.repeat(first, n_conf)
    if not np.array_equal(met["conf_idx"].astype(np.int64), expect):
        bad = int(np.flatnonzero(met["conf_idx"].astype(np.int64) != expect)[0])
        start = int(first[np.searchsorted(first, bad, side="right") - 1])
        raise BuildRefused(
            f"molecule {int(mid[bad])}: its conformers are numbered "
            f"{met['conf_idx'][start:bad + 1].tolist()[:8]}, not 0, 1, ...; the "
            f"run table stores a molecule's conformer count and a reader numbers "
            f"them by position (FORMAT.md section 7)")
    return first, n_conf


def _color_self_values(coords: np.ndarray, counts: np.ndarray,
                       r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Library color self-overlaps in the physical frame, via the native path
    every CPU consumer uses (``prepare_combo_library_self_overlaps``)."""
    from ..backend import cpu
    prepared = cpu.prepare_combo_library_self_overlaps(
        library_coords=coords, library_counts=counts,
        interaction_r=r, interaction_p=p, shape=False, color=True)
    if not prepared.color_prepared:
        raise RuntimeError("the native path did not prepare color self-overlaps")
    return np.array(prepared.color_values, dtype="<f8", copy=True)


def _color_self_plan(color_generator, enabled: bool) -> tuple[Optional[dict], dict]:
    """Whether to write the ``color_self`` section, and the ``build.json``
    record saying so.

    Written only for canonical interaction matrices (FORMAT.md section 18.2:
    the section is optional, and allowed only then): the values are
    matrix-specific, and a non-canonical build writes none and says why, so
    a consumer recomputes rather than trusting a term that was never
    universal. The rule is the one definition of canonical matrices
    (``format.r3g.reference.canonical_matrix_reasons``, which
    ``numerics.check_interaction_matrix`` calls): exact diagonal widths and
    heights of 1 and zero off-diagonal heights on the color types;
    off-diagonal widths and row/column 0 are ignored. ``--no-color-self``
    (``enabled=False``) omits it for any matrix.
    """
    from ..numerics import check_interaction_matrix
    if color_generator is None:
        return None, {}
    r = np.ascontiguousarray(color_generator.interaction_matrix_r, dtype=np.float32)
    p = np.ascontiguousarray(color_generator.interaction_matrix_p, dtype=np.float32)
    record = {"algorithm": "native prepare_combo_library_self_overlaps over the stored heavy "
                           "atoms and the derived pseudo-atoms, dequantized at the step "
                           "(FORMAT.md section 18.2)"}
    reason = None
    report = check_interaction_matrix(r, p)
    record["matrices_canonical"] = bool(report.canonical)
    if not enabled:
        reason = "disabled by the caller (color_self=False, --no-color-self)"
    elif not report.canonical:
        reason = ("non-canonical interaction matrix; a stored color self-overlap would "
                  "be matrix-specific, so readers recompute: " + "; ".join(report.reasons))
    else:
        from ..backend import cpu
        if not cpu.available():
            reason = "the native CPU extension is not available to compute it"
    if reason is not None:
        return None, {**record, "status": "not_built", "reason": reason}
    return {"r": r, "p": p}, {**record, "status": "built"}


def _chunk_features(counts_chunk, row_feature_start, chunk_nat, chunk_astart,
                    chunk_atoms, first: np.ndarray, n_types: int) -> tuple:
    """``GroupWriter.add``'s feature table for the molecules whose first rows
    in the chunk are ``first``: per-molecule feature counts, then each
    feature's type, atom count and atoms, in the shards' order -- grouped by
    ascending type and, on a ``"canonical"`` build, sorted by ``(type,
    slots)`` (FORMAT.md 8.1 rule 4, made by ``_ShardSet._feature_rows``)."""
    mol_counts = counts_chunk[first].astype(np.int64)            # [k, n_types]
    per_mol = mol_counts.sum(axis=1)
    types = np.repeat(np.tile(np.arange(1, n_types + 1, dtype=np.int64), first.size),
                      mol_counts.reshape(-1))
    feats = (np.repeat(row_feature_start[first] - _exclusive_cumsum(per_mol), per_mol)
             + np.arange(int(per_mol.sum()), dtype=np.int64))
    natoms = chunk_nat[feats].astype(np.int64)
    atoms = chunk_atoms[np.repeat(chunk_astart[feats] - _exclusive_cumsum(natoms), natoms)
                        + np.arange(int(natoms.sum()), dtype=np.int64)]
    return per_mol, types, natoms, atoms


def _gather(out_dir: str, tmp_dir: str, opts: BuildOptions, fingerprint: str,
            read_chunk: int = 200_000, color_self: Optional[dict] = None,
            checkpoint: Optional["_BuildCheckpoint"] = None,
            heartbeat=None, n_types: Optional[int] = None) -> list[dict]:
    """Write every group file (FORMAT.md section 3) under ``out_dir/groups``
    and return the group-table entries, in ``(bucket, group)`` order.

    Each group goes through the format's writer (``format.r3g.writer.
    GroupWriter``), which cuts its molecule-aligned blocks by
    ``opts.block_rows`` and encodes them with the native BP256-16 codec,
    decoding every block it wrote back against its input. With a
    ``checkpoint``, each finished group is committed as it closes, and a
    resumed gather starts at the first group not committed: the plan is a
    pure function of the shard index, so the skipped molecules are exactly
    the committed groups' (checked, not assumed).

    ``n_types`` is the color generator's number of feature types (``None``
    for a shape-only build). On a color build each molecule's feature table
    goes into the group file and, for ``color_self``, every row's
    pseudo-atoms are **derived** from its stored heavy atoms
    (``derive_feature_positions``; FORMAT.md section 8.3), exactly as a
    reader derives them."""
    from ..format.r3g.reference import pack_runs
    idx, n_types = _shard_index(tmp_dir, n_types)
    has_color = n_types is not None
    if not idx:
        raise RuntimeError(f"no molecules were prepared (no shards in {tmp_dir})")
    n_reals = sorted(idx)
    bidx = bucket_of(np.array(n_reals))
    groups: list[dict] = []
    os.makedirs(os.path.join(out_dir, "groups"), exist_ok=True)

    for b in sorted(set(bidx.tolist())):
        members = [n for n, bb in zip(n_reals, bidx) if bb == b]
        # The bucket-wide query-field width (FORMAT.md 8.6, Appendix B.15):
        # pad8 of the bucket's largest n_real, the same for every group.
        L_field = pad8(max(members))
        total = sum(shard.n for m in members for shard in idx[m])
        plan = _plan_bucket(idx, members, n_types, int(opts.max_conf_per_group),
                            read_chunk, heartbeat)
        if sum(p["n_conf"] for p in plan) != total:
            raise RuntimeError(
                f"bucket {b}: the group plan accounts for "
                f"{sum(p['n_conf'] for p in plan):,} conformers and the shard "
                f"index says {total:,}")
        done = checkpoint.bucket_groups(b) if checkpoint is not None else []
        if done:
            if len(done) > len(plan) or any(
                    int(g["n_conf"]) != int(p["n_conf"]) or int(g["n_mol"]) != int(p["n_mol"])
                    or int(g["L_field_width"]) != L_field
                    for g, p in zip(done, plan)):
                raise RuntimeError(
                    f"bucket {b}: the committed groups do not match the plan "
                    f"recomputed from the shards; the scratch changed under "
                    f"the checkpoint")
            groups.extend(done)
            if len(done) == len(plan):
                continue
        skip_mols = sum(int(g["n_mol"]) for g in done)
        state = {"gi": len(done), "in_group": 0, "w": None}

        def _open():
            state["w"] = GroupWriter(
                out_dir, bucket=b, group=state["gi"], fingerprint=fingerprint,
                n_types=n_types or 0, with_color_self=color_self is not None,
                block_rows=int(opts.block_rows), check_decode=True)
            state["in_group"] = 0

        def _close():
            w = state["w"]
            entry = w.close(L_field)
            p = plan[state["gi"]]
            if (entry["n_conf"], entry["n_mol"]) != (p["n_conf"], p["n_mol"]):
                raise RuntimeError(
                    f"{w.path}: wrote {entry['n_conf']:,} conformers of "
                    f"{entry['n_mol']:,} molecules; the plan said {p['n_conf']:,} "
                    f"of {p['n_mol']:,}")
            entry["file_bytes"] = os.path.getsize(w.path)
            entry["coords_bytes"] = int(w._coords_len)
            groups.append(entry)
            if checkpoint is not None:
                checkpoint.commit_group(b, entry, w.path)
            state["gi"] += 1
            state["w"] = None

        bucket_mol = 0
        next_row: dict = {}         # per shard: the next row the merge must hand us
        try:
            for n_real, segments, met, counts_chunk, frows in _shard_chunks(
                    idx, members, n_types, read_chunk):
                if heartbeat is not None:
                    heartbeat()
                cnt = met.size
                first, n_conf = _chunk_molecules(met)
                chunk_mol0, bucket_mol = bucket_mol, bucket_mol + first.size
                # The merge consumes each shard in ascending offset order, so a
                # chunk's rows from one shard are one contiguous range, and its
                # molecules one contiguous range of the shard's feature table.
                # Checked, not assumed. Each shard's range is read ONCE per
                # chunk and scattered into merged order by index.
                per_shard: dict = {}           # shard -> [first row, rows, dest rows]
                dest = 0
                for shard, off, n in segments:
                    if off != next_row.get(shard, 0):
                        raise RuntimeError(
                            f"the merge handed rows {off}.. of {shard.met}, but row "
                            f"{next_row.get(shard, 0)} is next; a shard must be "
                            f"consumed in ascending order, none skipped")
                    next_row[shard] = off + n
                    entry = per_shard.setdefault(shard, [off, 0, []])
                    entry[1] += n
                    entry[2].append(np.arange(dest, dest + n))
                    dest += n
                if bucket_mol <= skip_mols:
                    continue                   # wholly inside committed groups
                xyz = np.empty((cnt, 3, n_real), dtype="<i2")
                if has_color:
                    row_feature_start = np.empty(cnt, dtype=np.int64)
                    row_n_features = counts_chunk.sum(axis=1, dtype=np.int64)
                    nat_parts, astart_parts, atom_parts = [], [], []
                    base_f = base_a = 0
                for shard, (off, n, rows) in per_shard.items():
                    rows = np.concatenate(rows)          # merged positions, shard order
                    xyz[rows] = np.fromfile(shard.xyz, dtype="<i2", count=n * 3 * n_real,
                                            offset=off * 6 * n_real).reshape(n, 3, n_real)
                    if has_color:
                        k = int(frows.shard_no[rows[0]])
                        mi = frows.mol[rows]
                        fs, at = frows.feature_start[k], frows.atom_start[k]
                        m0, m1 = int(mi.min()), int(mi.max())
                        f0, f1 = int(fs[m0]), int(fs[m1 + 1])
                        a0, a1 = int(at[m0]), int(at[m1 + 1])
                        nat = np.fromfile(shard.base + ".fnat", dtype="u1",
                                          count=f1 - f0, offset=f0)
                        atm = np.fromfile(shard.base + ".fatm", dtype="u1",
                                          count=a1 - a0, offset=a0)
                        if nat.size != f1 - f0 or atm.size != a1 - a0:
                            raise RuntimeError(f"{shard.base}: short feature-table read")
                        row_feature_start[rows] = base_f + fs[mi] - f0
                        nat_parts.append(nat)
                        astart_parts.append(base_a + _exclusive_cumsum(nat))
                        atom_parts.append(atm)
                        base_f += f1 - f0
                        base_a += a1 - a0
                if has_color:
                    chunk_nat = (np.concatenate(nat_parts) if nat_parts
                                 else np.zeros(0, np.uint8))
                    chunk_astart = (np.concatenate(astart_parts) if astart_parts
                                    else np.zeros(0, np.int64))
                    chunk_atoms = (np.concatenate(atom_parts) if atom_parts
                                   else np.zeros(0, np.uint8))
                row_of = np.append(first, cnt)
                m = max(0, skip_mols - chunk_mol0)    # molecule index within the chunk
                while m < first.size:
                    if state["w"] is None:
                        _open()
                    want = plan[state["gi"]]["n_mol"] - state["in_group"]
                    m_end = min(first.size, m + want)
                    r0, r1 = int(row_of[m]), int(row_of[m_end])
                    runs = pack_runs(met["mol_id"][row_of[m:m_end]], n_conf[m:m_end],
                                     np.full(m_end - m, n_real))
                    features = cself = None
                    if has_color:
                        features = _chunk_features(
                            counts_chunk, row_feature_start, chunk_nat, chunk_astart,
                            chunk_atoms, first[m:m_end], n_types)
                    if color_self is not None:
                        # Pseudo-atoms derived from the stored heavy atoms, as a
                        # format-3 reader derives them (FORMAT.md section 8.3),
                        # in the slots after each row's real atoms (section 8.4).
                        take = r1 - r0
                        nf = row_n_features[r0:r1]
                        width = pad8(n_real + int(nf.max(initial=0)))
                        padded = np.zeros((take, 3, width), dtype=np.int16)
                        padded[:, :, :n_real] = xyz[r0:r1]
                        derived = derive_feature_positions(
                            xyz[r0:r1], row_feature_start[r0:r1], nf,
                            chunk_nat, chunk_astart, chunk_atoms)
                        rr = np.repeat(np.arange(take, dtype=np.int64), nf)
                        slot = (n_real + np.arange(rr.size, dtype=np.int64)
                                - np.repeat(_exclusive_cumsum(nf), nf))
                        padded[rr, :, slot] = derived
                        full_counts = np.empty((take, n_types + 1), np.uint8)
                        full_counts[:, 0] = n_real
                        full_counts[:, 1:] = counts_chunk[r0:r1]
                        stored_xyz = (np.ascontiguousarray(padded.transpose(0, 2, 1))
                                      .astype(np.float32) * np.float32(fmt.COORD_SCALE))
                        cself = _color_self_values(stored_xyz, full_counts,
                                                   color_self["r"], color_self["p"])
                    state["w"].add(runs, xyz[r0:r1], met["vaa"][r0:r1], cself, features)
                    state["in_group"] += m_end - m
                    m = m_end
                    if state["in_group"] == plan[state["gi"]]["n_mol"]:
                        _close()
            if state["w"] is not None:
                raise RuntimeError(
                    f"bucket {b}: group {state['gi']} received {state['in_group']} of "
                    f"its planned {plan[state['gi']]['n_mol']} molecules")
        except BaseException:
            if state["w"] is not None:
                state["w"].abort()
            raise
        if state["gi"] != len(plan):
            raise RuntimeError(
                f"bucket {b}: the plan named {len(plan)} groups and the "
                f"gather wrote {state['gi']}")
    return groups


def _merge_meta(out_dir: str, tmp_dir: str, heartbeat=None) -> tuple[int, str]:
    """Write ``molecules.parquet`` (FORMAT.md section 11) from the scatter's
    per-slot metadata and return ``(n_ids, ids_sha256)``.

    Without retaining one Python object per molecule: SQLite's on-disk
    primary-key index gives the duplicate-id refusal, and the rows leave it
    in ``mol_id`` order, in bounded batches, into the format's own parquet
    writer (``format.r3g.parquet.MoleculesWriter``: ZSTD level 9, PLAIN, row
    groups of 2^14 rows). The same ordered cursor feeds the id hash, so
    ``ids_sha256`` is by construction the sha256 of the sorted, unique
    ``mol_id`` column as little-endian int64 (FORMAT.md 10.6) -- the hash
    format 2 wrote as ``sha256_ids`` over its ``ids.i64``, which came from the
    same cursor.
    """
    from ..format.r3g.parquet import MoleculesWriter

    db_path = os.path.join(tmp_dir, "metadata.sqlite")
    if os.path.exists(db_path):
        os.unlink(db_path)             # an interrupted earlier merge: redo it
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("""CREATE TABLE metadata (
        mol_id INTEGER PRIMARY KEY, ext_id TEXT NOT NULL, hac INTEGER NOT NULL,
        mw REAL NOT NULL, rotors INTEGER NOT NULL, smiles TEXT NOT NULL)""")
    batch = []

    def insert_batch() -> None:
        nonlocal batch
        if not batch:
            return
        try:
            con.executemany("INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?)", batch)
        except sqlite3.IntegrityError as e:
            raise BuildRefused(
                "a molecule id appears more than once: two input records carry "
                "one molecule name in different work units (binary records must "
                "have unique _Name values), or two stable ids collided. Give each "
                "molecule its own id.") from e
        con.commit()
        batch = []
        if heartbeat is not None:
            heartbeat()

    try:
        for fn in sorted(os.listdir(tmp_dir)):
            if not (fn.startswith("meta_") and fn.endswith(".tsv")):
                continue
            with open(os.path.join(tmp_dir, fn)) as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    batch.append((int(p[0]), json.loads(p[1]), int(p[2]), float(p[3]),
                                  int(p[4]), p[5]))
                    if len(batch) >= 100_000:
                        insert_batch()
        insert_batch()

        out = os.path.join(out_dir, "molecules.parquet")
        writer = MoleculesWriter(out)
        ids_hash = hashlib.sha256()
        n_ids = 0
        try:
            cur = con.execute("SELECT mol_id, ext_id, hac, mw, rotors, smiles "
                              "FROM metadata ORDER BY mol_id")
            while rows := cur.fetchmany(1 << 16):
                columns = list(zip(*rows))
                ids = np.asarray(columns[0], dtype="<i8")
                writer.write(ids, columns[1], columns[5], columns[2], columns[3], columns[4])
                ids_hash.update(ids.tobytes())
                n_ids += ids.size
                if heartbeat is not None:
                    heartbeat()
        finally:
            writer.close()
    finally:
        con.close()
    return n_ids, ids_hash.hexdigest()


def sha256_ids(ids: np.ndarray) -> str:
    """Hash ids in bounded chunks as little-endian int64 (FORMAT.md 10.6's
    ``ids_sha256``, format 2's ``sha256_ids``); ``ids`` may be a memmap."""
    h = hashlib.sha256()
    for start in range(0, len(ids), 1_000_000):
        h.update(np.ascontiguousarray(ids[start:start + 1_000_000], dtype="<i8").tobytes())
    return h.hexdigest()


def _corpus_statistics(root: str, descriptor: dict, heartbeat=None) -> dict:
    """Per-conformer statistics for ``build.json``, read back from the group
    files just written (``n_real`` from each run table, ``vaa`` from each
    ``vaa`` section): ``hac_hist``, ``hac_mean`` and ``hac_sd`` over
    conformers, the ``vaa`` median and 1% / 99% quantiles, and the bytes.

    Two columns, each read once into one preallocated array: at full scale
    (374M conformers) 0.37 GB of n_real and 1.5 GB of vaa; the statistics are bitwise those of the copying numpy calls
    (``_hac_statistics``)."""
    from ..format.r3g.validate import open_group
    n_conf = int(descriptor["n_conf"])
    nr = np.empty(n_conf, dtype=np.uint8)
    vaa = np.empty(n_conf, dtype=np.float32)
    row = 0
    group_bytes = 0
    for entry in descriptor["groups"]:
        og = open_group(root, entry, descriptor=descriptor)
        try:
            runs = og.section_array("runs")
            n = int(entry["n_conf"])
            nr[row:row + n] = np.repeat(runs["n_real"], runs["n_conf"].astype(np.int64))
            vaa[row:row + n] = og.section_array("vaa").reshape(-1)
            group_bytes += og.file.size
        finally:
            og.close()
        row += n
        if heartbeat is not None:
            heartbeat()
    if row != n_conf:
        raise RuntimeError(f"the group files hold {row:,} rows; corpus.json says {n_conf:,}")
    pq_bytes = os.path.getsize(os.path.join(root, "molecules.parquet"))
    desc_bytes = os.path.getsize(os.path.join(root, "corpus.json"))
    return {
        **_hac_statistics(nr),
        # In place: vaa is this function's own array. Selection picks the
        # same order statistics whatever the array's order.
        "vaa_median": float(np.median(vaa, overwrite_input=True)),
        "vaa_q01": float(np.quantile(vaa, 0.01, overwrite_input=True)),
        "vaa_q99": float(np.quantile(vaa, 0.99, overwrite_input=True)),
        "bytes": {"groups": int(group_bytes), "molecules_parquet": int(pq_bytes),
                  "corpus_json": int(desc_bytes),
                  "total": int(group_bytes + pq_bytes + desc_bytes),
                  "per_conformer": (group_bytes + pq_bytes + desc_bytes) / n_conf
                  if n_conf else 0.0},
    }


def _pairwise_sum(segment, start: int, n: int, leaf: int = 1 << 20) -> float:
    """numpy's float64 sum of a length-``n`` array, bit for bit, without
    materializing it: numpy sums a contiguous array by pairwise recursion,
    splitting ``n`` at ``n // 2`` rounded down to a multiple of 8, so
    recursing the same way and letting numpy sum each leaf (``segment(start,
    m)``, a contiguous float64 array) reproduces its result exactly. Checked
    against ``np.add.reduce`` and ``np.std`` in tests.

    ``leaf`` must be at least numpy's pairwise block (128): below it numpy
    stops splitting and sums with eight interleaved accumulators instead."""
    if leaf < 128:
        raise ValueError("leaf must be at least 128")
    if n <= leaf:
        return float(np.add.reduce(segment(start, n)))
    half = n // 2
    half -= half % 8
    return (_pairwise_sum(segment, start, half, leaf)
            + _pairwise_sum(segment, start + half, n - half, leaf))


def _hac_statistics(nr: np.ndarray) -> dict:
    """``hac_hist``, ``hac_mean`` and ``hac_sd`` equal to
    ``np.bincount(nr.astype(np.int64))``, ``nr.mean()`` and ``nr.std()``, without
    their n-sized int64 and float64 temporaries (3 GB each at 374M rows).

    The histogram is summed per chunk (integers). The mean: numpy sums uint8
    into float64, where every partial sum is an exact integer below 2**53, so
    it is the exact sum over n. The standard deviation: numpy squares the
    float64 deviations from that mean and sums them pairwise; the same
    deviations, summed by :func:`_pairwise_sum`, give the same bits.
    """
    if nr.size == 0:
        return {"hac_hist": np.bincount(nr.astype(np.int64)).tolist(),
                "hac_mean": float(nr.mean()), "hac_sd": float(nr.std())}
    hist = np.zeros(256, dtype=np.int64)
    total = 0
    for start in range(0, nr.size, 1 << 22):
        chunk = nr[start:start + (1 << 22)]
        hist += np.bincount(chunk, minlength=256)
        total += int(chunk.sum(dtype=np.int64))
    mean = np.float64(total) / nr.size
    center = np.array([mean])

    def squared_deviations(start, m):
        x = nr[start:start + m] - center
        return np.multiply(x, x, out=x)

    var = np.float64(_pairwise_sum(squared_deviations, 0, nr.size)) / nr.size
    return {"hac_hist": hist[:int(np.flatnonzero(hist)[-1]) + 1].tolist(),
            "hac_mean": float(mean), "hac_sd": float(np.sqrt(var))}


#: Planning figure for the stored-frame shape self-overlap, per conformer on
#: one core (the float64 self-overlap over the dequantized coordinates), as
#: scoped; not measured by this module. The dry run reports it beside
#: the cost it measures on a sample of the actual input.
STORED_VAA_SECONDS_PER_CONFORMER = 11e-6

#: Records the dry run prepares in memory to measure the per-record cost.
DRY_RUN_SAMPLE_RECORDS = 64

CHECKPOINT_NAME = "_build_checkpoint.json"

#: The longest a running build leaves ``<out>.progress.json`` unrefreshed
#: while it is making progress (see ``prepare``'s heartbeat).
HEARTBEAT_S = 30.0


#: v3: the format-3 builder. A v2 checkpoint describes format-2
#: groups, so a resume across the switch is refused (the identity differs).
CHECKPOINT_FORMAT = "roshambo3.prepare.checkpoint.v3"


def _open_source(inputs, color_generator, input_format, max_records,
                 sqlite_source=None):
    """``(source, source_info, default_name)`` for the accepted input kinds."""
    from .sources import RDKitSource, RdkitBinaryZstdSource, SdfSource

    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    inputs = list(inputs)
    if not inputs:
        raise ValueError("no inputs given")

    if all(isinstance(p, (str, os.PathLike)) for p in inputs):
        paths = [str(p) for p in inputs]
        if input_format not in {"auto", "sdf", "rdkit-binary-zstd", "sqlite",
                                "sqlite:pubchem", "sqlite:text-id"}:
            raise ValueError(f"unknown input_format {input_format!r}")
        bad = [p for p in paths if p.lower().endswith((".smi", ".smiles"))]
        if bad:
            raise ValueError(
                f"{bad[0]} is a SMILES file, which has no 3D coordinates. "
                f"Generate conformers first and pass the SDF.")
        binary = input_format == "rdkit-binary-zstd" or (
            input_format == "auto" and len(paths) == 1 and
            paths[0].lower().endswith(".bin.zst"))
        sqlite = input_format.startswith("sqlite") or (
            input_format == "auto" and len(paths) == 1 and
            paths[0].lower().endswith((".sqlite", ".sqlite3", ".db")))
        if sqlite_source is not None and not sqlite:
            raise ValueError("sampler options (sqlite_source) apply only to SQLite input")
        if sqlite:
            if len(paths) != 1 or max_records is not None:
                raise ValueError(
                    "sqlite input accepts exactly one database and no max_records")
            from .sqlite_source import SqliteSource
            kind = input_format.partition(":")[2] or None  # None: detect from the schema
            source = SqliteSource(paths[0], kind=kind, color_generator=color_generator,
                                  **(sqlite_source or {}))
            source_info = source.describe()
        elif binary:
            if len(paths) != 1:
                raise ValueError("rdkit-binary-zstd input accepts exactly one solid stream")
            source = RdkitBinaryZstdSource(paths[0], color_generator=color_generator,
                                           max_records=max_records)
            source_info = {"kind": "rdkit-binary-zstd", "path": paths[0],
                           "bytes": os.path.getsize(paths[0]),
                           "max_records": max_records,
                           "complete": max_records is None}
        else:
            if max_records is not None:
                raise ValueError("max_records is currently supported only for rdkit-binary-zstd")
            source = SdfSource(paths, color_generator=color_generator)
            source_info = {"kind": "sdf", "paths": paths,
                           "bytes": sum(os.path.getsize(p) for p in paths)}
        default_name = os.path.basename(paths[0]).rsplit(".", 1)[0]
    else:
        if input_format != "auto" or max_records is not None:
            raise ValueError("input_format and max_records require path inputs")
        source = RDKitSource(inputs, color_generator=color_generator)
        source_info = {"kind": "rdkit", "n_molecules": len(inputs)}
        default_name = "rdkit"
    return source, source_info, default_name


def _input_paths(source_info: dict) -> list[str]:
    if source_info["kind"] == "sdf":
        return list(source_info["paths"])
    if source_info["kind"] == "rdkit-binary-zstd" or source_info["kind"].startswith("sqlite"):
        return [source_info["path"]]
    return []


#: What produces a corpus's stored bytes, for the code digest: the builder and the formats it writes, the color feature
#: generator, under the ``roshambo3`` package directory; plus the native
#: self-overlap producer, by its compiled source stamp.
CODE_DIGEST_PACKAGES = ("prep", "format")
CODE_DIGEST_MODULES = ("pharmacophore.py", "numerics.py", "layout.py")


def _code_digest() -> dict:
    """The code identity a resume must share: a sha256 over the **bytes** of
    every ``.py`` file of ``CODE_DIGEST_PACKAGES`` and ``CODE_DIGEST_MODULES``
    as they are on disk beside the loaded package, and the CPU extension's
    ``cpu_src_sha256`` (a sha256 over the bytes of every file it was compiled
    from, which holds the color self-overlap producer).

    Content, never a commit or ``__version__`` (a static string): a build
    from a dirty tree gets the digest of what it runs, never a clean one.
    The RDKit and numpy releases are part of it. ``native_cpu_src_sha256`` is ``"unavailable"`` without the
    extension (then nothing native produces stored bytes) and ``"unknown"``
    when the extension cannot say, which a resume refuses.
    """
    from .sources import _hash_framed
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = []
    for package in CODE_DIGEST_PACKAGES:
        for directory, dirs, names in os.walk(os.path.join(root, package)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            files += [os.path.join(directory, n) for n in names if n.endswith(".py")]
    files += [os.path.join(root, m) for m in CODE_DIGEST_MODULES]
    h = hashlib.sha256(b"roshambo3.prepare.code.v1")
    for path in sorted(files, key=lambda p: os.path.relpath(p, root)):
        with open(path, "rb") as f:
            data = f.read()
        _hash_framed(h, os.path.relpath(path, root).replace(os.sep, "/").encode())
        _hash_framed(h, data)
    native = "unavailable"
    try:
        from ..backend import cpu
        if cpu.available():
            native = str(cpu.build_stamp().get("cpu_src_sha256") or "unknown")
    except Exception:
        native = "unknown"
    # Canonical SMILES order and feature perception are RDKit-release
    # properties (FORMAT.md 13.3), and the arithmetic is numpy's: a resume
    # from another environment must run the same releases.
    import rdkit
    from rdkit import rdBase
    return {"python_sha256": h.hexdigest(), "native_cpu_src_sha256": native,
            "rdkit_version": str(getattr(rdBase, "rdkitVersion", rdkit.__version__)),
            "numpy_version": np.__version__}


def _build_identity(source_info, opts, batch_records,
                    color_self_enabled, color_generator, tmp_dir) -> dict:
    """Everything a resumed build must share with the build it resumes.

    Input files are identified here by path, size and modification time -- the
    same identity an indexer resumes on -- not re-hashed: hashing a
    multi-hundred-gigabyte input to resume would cost what resuming saves.
    Their *content* is identified per work unit instead: each committed slot
    carries the digest of the bytes it read, checked on resume
    (``_scatter``). The code is identified by ``_code_digest``.
    """
    from .. import __version__
    inputs = []
    for path in _input_paths(source_info):
        st = os.stat(path)
        inputs.append({"path": os.path.abspath(path), "bytes": st.st_size,
                       "mtime_ns": st.st_mtime_ns})
    color = None
    if color_generator is not None:
        from ..backend.cpu import _array_digest
        from ..format.r3g.reference import definition_hash
        color = {
            "class": type(color_generator).__name__,
            "fdef_path": getattr(color_generator, "fdefName", None),
            "definition_sha256": definition_hash(
                color_generator.descriptor_color()["definition"]),
            "feature_order": opts.feature_order,
            "features_enum": dict(color_generator.FEATURES_ENUM),
            "interaction_r_sha256": _array_digest("interaction_r", np.ascontiguousarray(
                color_generator.interaction_matrix_r, dtype=np.float32)),
            "interaction_p_sha256": _array_digest("interaction_p", np.ascontiguousarray(
                color_generator.interaction_matrix_p, dtype=np.float32)),
        }
    return {
        "format": CHECKPOINT_FORMAT,
        "roshambo3_version": __version__,
        "code": _code_digest(),
        "source": {k: v for k, v in source_info.items() if k != "bytes"},
        "inputs": inputs,
        "options": {"name": opts.name,
                    "records_per_unit": int(opts.records_per_unit),
                    "batch_records": batch_records,
                    "max_conf_per_group": int(opts.max_conf_per_group),
                    "block_rows": int(opts.block_rows),
                    "note": opts.note,
                    "color_self": bool(color_self_enabled)},
        "color": color,
        "scratch": tmp_dir,
    }


def _same_file_content(a, b) -> bool:
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                x, y = fa.read(1 << 20), fb.read(1 << 20)
                if x != y:
                    return False
                if not x:
                    return True
    except (OSError, TypeError):
        return False


def _same_build_elsewhere(stored, identity, report) -> bool:
    """Whether two identities differ only in where the color feature
    definitions file was found, and the two files are byte-identical.

    The path is the RDKit data directory of whichever environment ran the
    build, so resuming from another checkout's environment changes it while
    the definitions -- and the generator class, feature numbering and R/P
    digests, which the identity also pins -- are unchanged. Content, not
    location, is what a resume must share. Anything else still refuses.
    """
    if not isinstance(stored, dict) or not isinstance(stored.get("color"), dict):
        return False
    color = identity.get("color")
    if not isinstance(color, dict):
        return False
    old_path, new_path = stored["color"].get("fdef_path"), color.get("fdef_path")
    if old_path == new_path:
        return False
    if ({**stored, "color": {**stored["color"], "fdef_path": new_path}} != identity
            or not _same_file_content(old_path, new_path)):
        return False
    print(f"roshambo3: resuming with color feature definitions at {new_path}; "
          f"byte-identical to the interrupted build's {old_path}", file=sys.stderr)
    report("resume", fdef_path_relocated=True)
    return True


def _fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: str, record: dict, *, durable: bool) -> None:
    """Write via a sibling temporary and ``os.replace``: a reader never sees a
    torn record, and a crash leaves the previous one."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=1)
            if durable:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    if durable:
        _fsync_path(directory)


class _BuildCheckpoint:
    """The durable progress record of one resumable build.

    Lives inside the ``.partial`` staging directory, so it is published with
    nothing and removed with it. A slot (scatter work unit) or a group is
    recorded only after its files are durable; anything on disk that the record
    does not name is from an interrupted step and is deleted on resume.
    """

    def __init__(self, staging: str, identity: dict, *, resume: bool,
                 report=None):
        self.path = os.path.join(staging, CHECKPOINT_NAME)
        self.report = report or (lambda *a, **k: None)
        if resume:
            try:
                with open(self.path) as f:
                    self.state = json.load(f)
            except FileNotFoundError:
                raise ValueError(
                    f"{staging} holds no build checkpoint, so there is nothing "
                    f"to resume; remove it and build afresh") from None
            stored = self.state.get("identity")
            if "unknown" in (identity.get("code") or {}).values():
                raise BuildRefused(
                    "resume refused: the code digest cannot be established "
                    f"({identity.get('code')}): the native extension does not "
                    "report its source stamp, so nothing shows the interrupted "
                    "build ran the same producer. Build afresh.")
            if stored != identity and not _same_build_elsewhere(stored, identity,
                                                                 self.report):
                differ = sorted(k for k in set(stored or {}) | set(identity)
                                if (stored or {}).get(k) != identity.get(k))
                why = ""
                if "code" in differ:
                    was = (stored or {}).get("code") or {}
                    now = identity.get("code") or {}
                    parts = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
                    why = (" The builder code changed (code digest: "
                           + ", ".join(parts or ["absent from the interrupted "
                                                 "build's record"])
                           + "; covers prep/, format/, pharmacophore.py and the "
                             "native self-overlap producer).")
                raise BuildRefused(
                    "resume refused: the inputs, options, scratch location or "
                    "code differ from the interrupted build's "
                    f"(differing: {', '.join(differ)})." + why +
                    " Resuming would splice two different builds together.")
            self.state["resumes"] = int(self.state.get("resumes", 0)) + 1
        else:
            self.state = {"identity": identity, "resumes": 0,
                          "scatter": {"complete": False, "slots": {}},
                          "gather": {"complete": False, "buckets": {}}}
        self.save()

    def save(self) -> None:
        _write_json_atomic(self.path, self.state, durable=True)

    @property
    def committed(self) -> bool:
        return bool(self.state["scatter"]["slots"])

    # -- scatter -------------------------------------------------------
    def done_slots(self) -> dict:
        return {int(k): v for k, v in self.state["scatter"]["slots"].items()}

    def commit_slot(self, tmp_dir: str, slot: int, n_mol: int, n_conf: int,
                    rejected: list, extra: Optional[dict] = None) -> None:
        """Record a durable work unit: its counts, its rejects and dropped
        conformers (``rejects_NNNNN.json``) and its content digest, which a
        resume checks the input against."""
        extra = extra or {}
        dropped = {str(k): int(v) for k, v in
                   sorted((extra.get("dropped_conformers") or {}).items())}
        path = os.path.join(tmp_dir, f"rejects_{slot:05d}.json")
        _write_json_atomic(path, {"rejected": [list(r) for r in rejected],
                                  "dropped_conformers": dropped}, durable=True)
        self.state["scatter"]["slots"][str(slot)] = {
            "n_mol": int(n_mol), "n_conf": int(n_conf),
            "n_rejected": len(rejected),
            "n_dropped_conformers": int(sum(dropped.values())),
            "content_sha256": extra.get("content_sha256")}
        self.save()

    def _slot_record(self, tmp_dir: str, slot: int) -> dict:
        with open(os.path.join(tmp_dir, f"rejects_{slot:05d}.json")) as f:
            return json.load(f)

    def slot_rejects(self, tmp_dir: str, slot: int) -> list:
        return [tuple(r) for r in self._slot_record(tmp_dir, slot)["rejected"]]

    def slot_dropped(self, tmp_dir: str, slot: int) -> dict:
        return dict(self._slot_record(tmp_dir, slot)["dropped_conformers"])

    def finish_scatter(self, n_units: int) -> None:
        self.state["scatter"].update({"complete": True, "n_units": int(n_units)})
        self.save()

    # -- gather --------------------------------------------------------
    def bucket_groups(self, bucket: int) -> list:
        return list(self.state["gather"]["buckets"].get(str(bucket), []))

    def commit_group(self, bucket: int, group: dict, path: str) -> None:
        """Record a finished group file (its group-table entry) once it and
        its directory entry are durable."""
        _fsync_path(path)
        _fsync_path(os.path.dirname(path))
        self.state["gather"]["buckets"].setdefault(str(bucket), []).append(group)
        self.save()
        self.report("gather", groups=sum(
            len(v) for v in self.state["gather"]["buckets"].values()))

    def committed_group_files(self) -> set:
        from ..format.r3g.limits import group_file_name
        return {group_file_name(g["bucket"], g["group"])
                for groups in self.state["gather"]["buckets"].values() for g in groups}


def _slot_of(name: str) -> Optional[int]:
    """The scatter slot a scratch file belongs to, or None for a shared file."""
    stem = name.split(".", 1)[0]
    for prefix in ("meta_", "rejects_"):
        if stem.startswith(prefix):
            return int(stem[len(prefix):])
    if stem.startswith("n") and "_s" in stem:
        return int(stem.rsplit("_s", 1)[1])
    return None


def _discard_uncommitted(staging: str, tmp_dir: str,
                         checkpoint: _BuildCheckpoint) -> None:
    """Delete everything an interrupted step left that the checkpoint does not
    name, so the resumed build writes only into a state it fully knows."""
    done = checkpoint.done_slots()
    if os.path.isdir(tmp_dir):
        for name in os.listdir(tmp_dir):
            slot = _slot_of(name)
            if slot is None or slot not in done:
                path = os.path.join(tmp_dir, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
    keep = {CHECKPOINT_NAME, "groups", "_tmp"}
    for name in os.listdir(staging):
        if name not in keep:
            path = os.path.join(staging, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
    groups = os.path.join(staging, "groups")
    if os.path.isdir(groups):
        committed = checkpoint.committed_group_files()
        for name in os.listdir(groups):
            if name not in committed:
                path = os.path.join(groups, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)


def _existing_ancestor(path: str) -> str:
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _count_sdf_records(paths, budget_bytes: int = 64 << 20) -> tuple[Optional[int], str]:
    """Exact when every file fits in the read budget, else extrapolated from
    the delimiter density of a prefix (and labeled as such)."""
    total = 0
    exact = True
    for path in paths:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(min(size, budget_bytes))
        n = head.count(b"$$$$")
        if len(head) < size:
            exact = False
            n = int(round(n * size / max(1, len(head))))
        total += n
    return total, ("counted" if exact else "extrapolated from a 64 MiB prefix")


def _sample_stats(records, n_inputs: int, failures: int, prep_s: float) -> dict:
    """The dry-run sample's figures, with the stored-frame ``vaa`` timed (the
    builder computes it from the quantized coordinates of every conformer)."""
    from .conformer import stored_self_volume
    confs = [c for rec in records for c in rec.conformers]
    t0 = time.perf_counter()
    for conf in confs:
        q = fmt.quantize(conf[0])
        stored_self_volume(q.astype(np.float32) * np.float32(fmt.COORD_SCALE))
    vaa_s = time.perf_counter() - t0
    n_color = [len(rec.features) for rec in records for _ in rec.conformers
               if getattr(rec, "features", None) is not None]
    return {"input_records": n_inputs, "molecules": len(records),
            "conformers": len(confs), "rejected": failures,
            "preparation_s": prep_s, "stored_vaa_s": vaa_s,
            "s_per_input_record": (prep_s + vaa_s) / n_inputs,
            "conformers_per_input_record": len(confs) / n_inputs,
            "conformers_per_molecule": len(confs) / max(1, len(records)),
            "mean_heavy_atoms": float(np.mean([c[1] for c in confs])) if confs else None,
            "mean_color_features": float(np.mean(n_color)) if n_color else 0.0}


def _sample_sqlite_preparation(source, k: int) -> Optional[dict]:
    """The dry-run sample for a SQLite source: time its first ``k`` selected
    molecules end to end (read, decode, prepare), which is what a worker pays.
    Reads one key range, never the table; writes nothing."""
    import itertools
    units = source.units(k)
    if not units:
        return None
    t0 = time.perf_counter()
    records = list(itertools.islice(source.iter_unit(units[0]), k))
    prep_s = time.perf_counter() - t0
    if not records:
        return None
    return _sample_stats(records, len(records), len(source.failures), prep_s)


def _sqlite_expected_records(source_info: dict):
    """Expected molecule count for a sampled SQLite tier, or ``(None, why)``.

    Source rows x the sampler fraction. The rows come from a ``source_rows``
    hint or the table counts the source already took; a heavy-atom window
    narrows the tier by an amount nothing here knows, so it is an upper bound.
    """
    sampler = source_info["sampler"]
    rows = source_info.get("n_rows_hint")
    if rows is None:
        counts = source_info.get("table_counts") or {}
        rows = counts.get("meta") or counts.get("mols") or counts.get("mols_rowid_span")
    if rows is None:
        return None, "unknown: pass a source row count (--source-rows) to avoid counting the table"
    frac = float(sampler["p_hi"]) - float(sampler["p_lo"])
    bound = " (upper bound: heavy-atom window)" if sampler["hac_window"] else ""
    return int(round(int(rows) * frac)), f"expected: {int(rows):,} source rows x sampler fraction {frac:.6g}{bound}"


def _sample_preparation(source, source_info, color_generator,
                        k: int = DRY_RUN_SAMPLE_RECORDS) -> Optional[dict]:
    """Prepare up to ``k`` input records in memory and time it. Writes nothing."""
    from rdkit import Chem
    from .sources import _group_by_name

    mols = []
    if source_info["kind"] == "rdkit":
        mols = [m for m in source.mols[:k] if m is not None]
    elif source_info["kind"] == "sdf":
        with open(source_info["paths"][0], "rb") as f:
            supplier = Chem.ForwardSDMolSupplier(f, removeHs=False, sanitize=True)
            for mol in supplier:
                if mol is not None:
                    mols.append(mol)
                if len(mols) >= k:
                    break
    elif source_info["kind"].startswith("sqlite"):
        return _sample_sqlite_preparation(source, k)
    else:
        for _, _, blobs in source.iter_payload_batches(k):
            mols = [Chem.Mol(b) for b in blobs]
            break
    if not mols:
        return None
    failures: list = []
    t0 = time.perf_counter()
    records = list(_group_by_name(mols, 0, color_generator, failures,
                                  contiguous_conformer_ids=(
                                      source_info["kind"] == "rdkit-binary-zstd")))
    prep_s = time.perf_counter() - t0
    return _sample_stats(records, len(mols), len(failures), prep_s)


def _dry_run(source, source_info, out_dir, temp_dir, opts, color_generator,
             n_workers) -> dict:
    """What a build would read, write and cost. Creates and writes nothing."""
    kind = source_info["kind"]
    if kind == "rdkit":
        records, basis = len(source.mols), "counted (in-memory molecules)"
    elif kind == "sdf":
        records, basis = _count_sdf_records(source_info["paths"])
    elif kind.startswith("sqlite"):
        records, basis = _sqlite_expected_records(source_info)
    elif source_info.get("max_records") is not None:
        records, basis = int(source_info["max_records"]), "max_records (upper bound)"
    else:
        records, basis = None, ("unknown: a solid zstd stream cannot be counted "
                                "without reading all of it")
    sample = _sample_preparation(source, source_info, color_generator)
    conformers = None
    if records is not None and sample:
        # An SDF record is one conformer; a binary/in-memory record is a
        # molecule carrying any number of them.
        conformers = (records if kind == "sdf" else
                      int(round(records * sample["conformers_per_input_record"])))
    cpu = {"stored_vaa_s": None, "preparation_s": None, "total_s": None,
           "wall_s_at_workers": None, "workers": n_workers,
           "stored_vaa_s_per_conformer": STORED_VAA_SECONDS_PER_CONFORMER}
    if conformers is not None:
        cpu["stored_vaa_s"] = conformers * STORED_VAA_SECONDS_PER_CONFORMER
    if records is not None and sample:
        cpu["preparation_s"] = records * sample["preparation_s"] / sample["input_records"]
        cpu["total_s"] = cpu["preparation_s"] + (cpu["stored_vaa_s"] or 0.0)
        cpu["wall_s_at_workers"] = cpu["total_s"] / max(1, n_workers)
    written = {"corpus_bytes": None, "scratch_bytes": None,
               "basis": "upper estimate from the sample's mean atom and feature "
                        "counts: 16-bit coordinates (bit packing stores fewer), "
                        "vaa, color_self, the run and feature tables"}
    if conformers is not None and sample and sample["mean_heavy_atoms"]:
        heavy = sample["mean_heavy_atoms"]
        n_feat = sample["mean_color_features"]
        per_mol = 16 + (n_feat * 6 + 4 if color_generator else 0) + 100   # + parquet row
        per_conf = 6 * heavy + 4 + (8 if color_generator else 0)
        written["corpus_bytes"] = int(conformers * per_conf + conformers / max(
            1.0, sample["conformers_per_molecule"]) * per_mol)
        n_types = len(color_generator.FEATURES_ENUM) if color_generator else 0
        written["scratch_bytes"] = int(conformers * (6 * heavy + 14)
                                       + conformers * (10 + n_types + 4 * n_feat))
    dest = _existing_ancestor(out_dir)
    scratch_root = _existing_ancestor(temp_dir) if temp_dir else dest
    free = {"destination": shutil.disk_usage(dest).free,
            "destination_path": dest,
            "scratch": shutil.disk_usage(scratch_root).free,
            "scratch_path": scratch_root}
    need_dest = written["corpus_bytes"]
    if need_dest is not None and scratch_root == dest:
        need_dest += written["scratch_bytes"]
    state = ("absent" if not os.path.exists(out_dir) else
             "empty" if not os.listdir(out_dir) else "not_empty")
    return {
        "dry_run": True, "wrote_nothing": True,
        "out_dir": out_dir, "out_dir_state": state,
        "would_refuse": state == "not_empty",
        "interrupted_build_present": os.path.exists(out_dir.rstrip(os.sep) + ".partial"),
        "source": source_info, "bytes_to_read": _input_bytes(source_info),
        "records": records, "records_basis": basis,
        "conformers_estimate": conformers, "sample": sample,
        "cpu_estimate": cpu, "bytes_to_write": written, "free_bytes": free,
        "fits": (None if need_dest is None else need_dest < free["destination"]),
        "options": {"color": color_generator is not None,
                    "block_rows": int(opts.block_rows),
                    "max_conf_per_group": int(opts.max_conf_per_group)},
    }


def _input_bytes(source_info: dict) -> Optional[int]:
    size = source_info.get("bytes")
    if size is not None and source_info["kind"].startswith("sqlite"):
        # A sampled tier fetches only its selected rows' blobs (plus the key
        # index), so the file size is an upper bound, scaled by the fraction.
        sampler = source_info["sampler"]
        size = int(size * (float(sampler["p_hi"]) - float(sampler["p_lo"])))
    return size


@contextmanager
def _writer_lock(path: str):
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"another prepare owns {path[:-len('.lock')]}; one writer per "
                f"output directory") from None
        yield


def _publish(staging: str, final_dir: str) -> None:
    """Rename the finished staging directory into place, never over content.

    ``rename(2)`` of a directory fails with ENOTEMPTY/EEXIST onto a non-empty
    directory and ENOTDIR onto a file, so a corpus or anything else that
    appeared at the target meanwhile is never overwritten; the staging
    directory survives the refusal intact.
    """
    if os.path.isdir(final_dir) and not os.listdir(final_dir):
        os.rmdir(final_dir)
    try:
        os.rename(staging, final_dir)
    except OSError as e:
        raise FileExistsError(
            f"{final_dir} appeared while this build ran; the finished build is "
            f"left at {staging} and was not published over it") from e


def _content_refusal(slot: int, why: str) -> BuildRefused:
    return BuildRefused(
        f"resume refused: the input content changed under the interrupted "
        f"build -- work unit {slot}'s content digest {why} (path, size and "
        f"mtime may all be unchanged). Resuming would splice two different "
        f"inputs into one corpus; remove the partial build and build afresh.")


def _check_committed_content(stored: Optional[str], actual: Optional[str],
                             slot: int) -> None:
    """A committed work unit's content digest against the input as it is
    now."""
    if stored is None and actual is None:
        return
    if stored is None or actual is None:
        raise _content_refusal(slot, "cannot be compared (recorded "
                                     f"{stored!r}, now {actual!r})")
    if stored != actual:
        raise _content_refusal(slot, f"was {stored[:16]}..., is now {actual[:16]}...")


def _scatter(source, opts, n_workers, tmp_dir, checkpoint, progress,
             report, heartbeat=None, id_reference=None
             ) -> tuple[int, int, list, int, int, dict]:
    """Prepare every input record into scratch shards, one slot per work unit.

    Slots already committed by ``checkpoint`` are skipped; each newly finished
    slot is committed as it completes, with its content digest (computed as
    the unit was read). On a resume every committed unit's digest is checked
    against the input before anything new is read: re-reading the committed
    units' bytes for a re-readable input (SDF byte ranges, SQLite rows; no
    RDKit work), and during the replay a solid stream needs anyway. Returns
    ``(n_mol, n_conf, rejected, n_units, records_per_unit, dropped)``, totals
    over every slot in slot order; ``dropped`` is ``{molecule name:
    conformers dropped}``.
    """
    done = checkpoint.done_slots() if checkpoint is not None else {}
    results: dict = {}

    def finish(slot, m, c, bad, extra=None):
        if id_reference is not None:
            # Before the commit: a slot that fails is never trusted later.
            id_reference.check_slot(tmp_dir, slot, bad)
        extra = extra or {}
        results[slot] = (m, c, bad, extra.get("dropped_conformers") or {})
        if checkpoint is not None:
            checkpoint.commit_slot(tmp_dir, slot, m, c, bad, extra)
        n_mol = sum(v[0] for v in results.values()) + sum(
            v["n_mol"] for k, v in done.items() if k not in results)
        n_conf = sum(v[1] for v in results.values()) + sum(
            v["n_conf"] for k, v in done.items() if k not in results)
        report("scatter", slots_done=len(results) + len(done),
               molecules=n_mol, conformers=n_conf)
        return n_mol, n_conf

    if getattr(source, "parallel_payloads", False):
        # Keep at most one raw batch in flight per worker.  A solid zstd
        # stream cannot be seek-sharded, so this is the bounded handoff
        # between its sole reader and worker-local RDKit deserialization.
        # A prefix build smaller than the normal 20k batch must still
        # create enough independent jobs to exercise its requested pool.
        # The complete solid stream keeps its 20k batches (about 127 MiB
        # large raw payloads); a capped prefix is reduced only
        # as far as needed for two batches per worker.
        batch_records = _batch_records(source, opts, n_workers)
        # A resumed build still streams through committed batches (the
        # stream cannot seek) but never deserializes or prepares them.
        def uncommitted():
            from .sources import payload_digest
            replayed = set()
            for slot, first, blobs in source.iter_payload_batches(batch_records):
                if slot in done:
                    # The replay reads these bytes anyway: check them.
                    _check_committed_content(done[slot].get("content_sha256"),
                                             payload_digest(blobs), slot)
                    replayed.add(slot)
                    if heartbeat is not None:
                        heartbeat(replayed_slot=slot)
                    continue
                yield slot, first, blobs
            # A committed slot the stream no longer reaches is stale content
            # (the input shrank; size and mtime can still match): refused, or
            # its records would be spliced into the corpus.
            for slot in sorted(set(done) - replayed):
                raise _content_refusal(
                    slot, "is absent: the input stream now ends before it")

        batches = uncommitted()
        n_slots = len(done)
        completed_batches = 0
        from concurrent.futures import FIRST_COMPLETED, wait
        pool = _ScatterPool(n_workers, source, payloads=True)
        aborted = True
        try:
            pending = set()

            def dispatch() -> bool:
                try:
                    batch = next(batches)
                except StopIteration:
                    return False
                pending.add(pool.submit(_scatter_next_batch, tmp_dir, payload=batch))
                return True

            for _ in range(n_workers):
                if not dispatch():
                    break
            while pending:
                # Committed as each completes: a resume takes any set of
                # committed slots, and slot order is restored at the end.
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.discard(future)
                    slot, m, c, bad, extra = pool.result(future)
                    n_slots += 1
                    n_mol, n_conf = finish(slot, m, c, bad, extra)
                    completed_batches += 1
                    # Before every worker has completed a first batch, the
                    # aggregate count omits work already under way in the
                    # other workers and ``n_mol / elapsed`` is a uselessly
                    # low startup rate.  Report after that first wave instead.
                    if progress and completed_batches >= n_workers:
                        progress(n_mol, n_conf)
                    dispatch()
            aborted = False
        finally:
            pool.close(abort=aborted)
        n_units = n_slots
        records_per_unit = batch_records
    else:
        units = source.units(opts.records_per_unit)
        digest = getattr(source, "unit_digest", None)
        for slot in sorted(done):
            if heartbeat is not None:
                heartbeat(verified_slot=slot)
            _check_committed_content(
                done[slot].get("content_sha256"),
                (digest(units[slot]) if digest is not None and slot < len(units)
                 else None if digest is None else "absent: the input now has "
                 f"only {len(units)} work units"), slot)
        jobs = [(i, u, tmp_dir) for i, u in enumerate(units) if i not in done]
        pool = None
        if n_workers == 1 or len(jobs) <= 1:
            _worker_init(source)
            output = map(_scatter_unit, jobs)
        else:
            from concurrent.futures import as_completed
            pool = _ScatterPool(n_workers, source)
            output = (pool.result(f) for f in as_completed(
                [pool.submit(_scatter_unit, job) for job in jobs]))
        aborted = True
        try:
            for slot, m, c, bad, extra in output:
                n_mol, n_conf = finish(slot, m, c, bad, extra)
                if progress:
                    progress(n_mol, n_conf)
            aborted = False
        finally:
            if pool is not None:
                pool.close(abort=aborted)
        n_units = len(units)
        records_per_unit = opts.records_per_unit
    if not results and not done:
        raise ValueError("the input contains no records")
    n_mol = n_conf = 0
    rejected: list = []
    dropped: dict = {}
    # Slot order, not completion order: the reject report is deterministic
    # and identical whether or not the build was resumed.
    for slot in sorted(set(results) | set(done)):
        if slot in results:
            m, c, bad, lost = results[slot]
        else:
            m, c = done[slot]["n_mol"], done[slot]["n_conf"]
            bad = checkpoint.slot_rejects(tmp_dir, slot)
            lost = checkpoint.slot_dropped(tmp_dir, slot)
        n_mol += m
        n_conf += c
        rejected.extend(bad)
        _add_dropped(dropped, lost)
    return n_mol, n_conf, rejected, n_units, records_per_unit, dropped


def _add_dropped(total: dict, more: dict) -> None:
    for name, n in more.items():
        total[name] = total.get(name, 0) + int(n)


class _IdReference:
    """An existing corpus whose molecule ids a build must reproduce, checked
    as the build goes rather than ~15 h in (``expect_ids_from``).

    The reference is a format-3 corpus directory (its ``molecules.parquet``
    ``mol_id`` column, read after the corpus passes the ``scan`` tier, whose
    sha256 must be its ``corpus.json`` ``ids_sha256``) or a format-2
    directory, of which only ``ids.i64`` (a raw sorted little-endian int64
    file) and ``manifest.json``'s ``sha256_ids`` are read; no format-2 reader
    is needed. Both id lists are the sorted, unique ``mol_id`` set hashed as
    little-endian int64, so one hash compares with the other.

    Each scatter slot, before it is committed, must satisfy ``mol_id ==
    stable_id(ext_id)`` for every molecule it accepted, and every ``mol_id``
    must be in the reference's sorted ids; no molecule it rejected may be
    (by ``stable_id`` of the identifier its reject list records, unless the
    same unit also accepted that molecule). A format-2 ``ids.i64`` is
    memory-mapped, never loaded, and read once in 8 MB chunks to confirm it
    is sorted and hashes to ``sha256_ids``; a format-3 id list is read into
    memory (8 B per molecule). After the build, ``ids_sha256`` must equal the
    reference's, unless ``max_records`` made this a prefix build (which
    cannot match; the comparison is skipped and ``build.json`` says so).
    """

    def __init__(self, corpus_dir: str):
        self.root = os.path.abspath(str(corpus_dir))
        if os.path.exists(os.path.join(self.root, "corpus.json")):
            self.kind = "format 3"
            self.ids, self.sha256_ids = self._format3_ids(self.root)
        elif os.path.exists(os.path.join(self.root, "manifest.json")):
            self.kind = "format 2"
            self.ids, self.sha256_ids = self._format2_ids(self.root)
        else:
            raise ValueError(f"expect_ids_from: {self.root} holds neither a corpus.json "
                             f"(format 3) nor a manifest.json (format 2); it is not a corpus")
        self.checked = 0

    @staticmethod
    def _format3_ids(root: str) -> tuple[np.ndarray, str]:
        import pyarrow.parquet as pq
        from ..format.r3g.errors import CorpusValidationError
        from ..format.r3g.validate import validate_corpus
        try:
            report = validate_corpus(root, "scan")
        except CorpusValidationError as e:
            raise ValueError(f"expect_ids_from: {root} is not a valid format-3 corpus "
                             f"(scan refused it: {e.code}: {e})") from None
        expected = report.descriptor["ids_sha256"]
        # The scan has bounded and checked molecules.parquet (sorted, unique,
        # in range, equal to the run tables); read only its mol_id column,
        # one row group at a time.
        parts = []
        with open(os.path.join(root, "molecules.parquet"), "rb") as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.metadata.num_row_groups):
                parts.append(np.asarray(pf.read_row_group(rg, columns=["mol_id"])
                                        .column(0).to_numpy(), dtype=np.uint64))
        ids = (np.concatenate(parts) if parts else np.zeros(0, np.uint64)).astype("<i8")
        if sha256_ids(ids) != expected:
            raise ValueError(f"expect_ids_from: {root}'s molecules.parquet mol_id column "
                             f"does not hash to its corpus.json ids_sha256 {expected}")
        return ids, expected

    @staticmethod
    def _format2_ids(root: str) -> tuple[np.ndarray, str]:
        with open(os.path.join(root, "manifest.json")) as f:
            manifest = json.load(f)
        expected = manifest.get("sha256_ids") if isinstance(manifest, dict) else None
        path = os.path.join(root, "ids.i64")
        if not os.path.exists(path) or not expected:
            raise ValueError(f"expect_ids_from: {root} lacks ids.i64 or a "
                             f"manifest sha256_ids")
        if os.path.getsize(path) % 8:
            raise ValueError(f"expect_ids_from: {path} is not a whole number of int64 ids")
        ids = np.memmap(path, dtype="<i8", mode="r") if os.path.getsize(path) else \
            np.zeros(0, dtype="<i8")
        if "n_ids" in manifest and int(manifest["n_ids"]) != ids.size:
            raise ValueError(f"expect_ids_from: {path} holds {ids.size:,} ids "
                             f"but the manifest says {manifest.get('n_ids')}")
        last = None
        h = hashlib.sha256()
        for start in range(0, ids.size, 1 << 20):
            chunk = np.asarray(ids[start:start + (1 << 20)])
            if (chunk.size > 1 and not np.all(chunk[1:] > chunk[:-1])) or (
                    last is not None and chunk.size and chunk[0] <= last):
                raise ValueError(f"expect_ids_from: {path} is not strictly "
                                 f"increasing, so it cannot be searched")
            if chunk.size and chunk[0] < 0:
                raise ValueError(f"expect_ids_from: {path} holds a negative id")
            if chunk.size:
                last = chunk[-1]
            h.update(chunk.astype("<i8").tobytes())
        if h.hexdigest() != expected:
            raise ValueError(f"expect_ids_from: {path} does not hash to the manifest's "
                             f"sha256_ids {expected}")
        return ids, expected

    def check_slot(self, tmp_dir: str, slot: int, rejected=()) -> None:
        """``rejected``: the unit's ``(identifier, reason)`` reject list, as
        committed beside it; a rejected molecule the reference holds fails
        the unit too."""
        from .conformer import stable_id
        from .rejects import reason_code
        mol_ids, bad_hash = [], []
        with open(os.path.join(tmp_dir, f"meta_{slot:05d}.tsv")) as f:
            for line in f:
                mol_id, ext_id = line.split("\t", 2)[:2]
                mol_id, ext_id = int(mol_id), json.loads(ext_id)
                mol_ids.append(mol_id)
                if stable_id(ext_id) != mol_id:
                    bad_hash.append(ext_id)
        found = np.asarray(mol_ids, dtype=np.int64)
        at = np.searchsorted(self.ids, found)
        present = at < self.ids.size
        present[present] = np.asarray(self.ids[at[present]]) == found[present]
        absent = found[~present]
        if bad_hash or absent.size:
            names = _slot_names(tmp_dir, slot, set(absent.tolist()))
            raise BuildRefused(
                f"id check against {self.root}: work unit {slot} (not committed) "
                f"holds {len(mol_ids):,} molecules, of which {absent.size:,} are "
                f"absent from the reference ids (first: {names[:5]}) and "
                f"{len(bad_hash):,} have mol_id != stable_id(ext_id) (first: "
                f"{bad_hash[:5]}). The input is not the reference's source, or "
                f"the reference is the wrong corpus; a resume would stop here "
                f"again. A fresh build that committed nothing leaves nothing "
                f"behind; otherwise the partial build holds only work units "
                f"that passed this check.")
        # A molecule this unit rejected that the reference holds would be
        # missing from the finished corpus, which only the final sha256_ids
        # comparison would notice, a whole build later. A name both accepted
        # and rejected here (a molecule that lost some conformers) is present.
        # The identifiers are those the reject list records: the source's
        # name, else ``str(mol_id)`` or ``_record_<n>`` for an unparsed record,
        # which hash to ids no reference holds.
        accepted = set(mol_ids)
        lost = {}
        for ident, why in rejected:
            mol_id = stable_id(str(ident))
            if mol_id not in accepted:
                lost.setdefault(mol_id, (str(ident), str(why)))
        if lost:
            wanted = np.fromiter(sorted(lost), dtype=np.int64, count=len(lost))
            at = np.searchsorted(self.ids, wanted)
            present = at < self.ids.size
            present[present] = np.asarray(self.ids[at[present]]) == wanted[present]
            held = [lost[int(i)] for i in wanted[present]]
            if held:
                raise BuildRefused(
                    f"id check against {self.root}: work unit {slot} (not "
                    f"committed) rejected {len(held):,} molecules that the "
                    f"reference holds (first: "
                    f"{[f'{n} ({reason_code(n, w)})' for n, w in held[:5]]}, "
                    f"by reason code). This build cannot "
                    f"reproduce the reference's ids; a resume would stop here "
                    f"again. A fresh build that committed nothing leaves nothing "
                    f"behind; otherwise the partial build holds only work units "
                    f"that passed this check.")
        self.checked += len(mol_ids)

    def record(self, actual_sha256: str, max_records) -> dict:
        if max_records is not None:
            return {"corpus": self.root, "kind": self.kind, "ids_sha256": self.sha256_ids,
                    "early_checked_molecules": self.checked, "matched": None,
                    "final_comparison": f"skipped: a max_records={max_records} "
                                        f"prefix build cannot reproduce the "
                                        f"reference's ids"}
        return {"corpus": self.root, "kind": self.kind, "ids_sha256": self.sha256_ids,
                "early_checked_molecules": self.checked,
                "matched": actual_sha256 == self.sha256_ids,
                "final_comparison": "ids_sha256"}


def _slot_names(tmp_dir: str, slot: int, wanted: set) -> list:
    out = []
    with open(os.path.join(tmp_dir, f"meta_{slot:05d}.tsv")) as f:
        for line in f:
            mol_id, ext_id = line.split("\t", 2)[:2]
            if int(mol_id) in wanted:
                out.append(json.loads(ext_id))
                if len(out) >= 5:
                    break
    return out


#: The full reject and dropped-conformer lists, one JSON object per line,
#: written **beside** the corpus as ``<out_dir>.rejected.jsonl`` and
#: ``<out_dir>.dropped_conformers.jsonl`` (a corpus directory holds only
#: ``corpus.json``, ``molecules.parquet``, ``build.json`` and its group files;
#: FORMAT.md section 1). ``build.json`` keeps the counts, the reasons and 50
#: examples, and names the files.
REJECTS_SUFFIX = ".rejected.jsonl"
DROPPED_SUFFIX = ".dropped_conformers.jsonl"

#: Files written into the staging directory after the gather, which the
#: ``assembled`` checkpoint vouches for (the group files are made durable as
#: each group commits).
ASSEMBLED_FILES = ("molecules.parquet", "corpus.json", "build.json")


def _sync_assembled(staging: str, heartbeat=None) -> None:
    for name in ASSEMBLED_FILES:
        path = os.path.join(staging, name)
        if os.path.exists(path):
            _fsync_path(path)
            if heartbeat is not None:
                heartbeat()
    if os.path.isdir(os.path.join(staging, "groups")):
        _fsync_path(os.path.join(staging, "groups"))
    _fsync_path(staging)


def _remove_scratch(tmp_dir: str, heartbeat=None) -> None:
    """Delete the build's scratch, entry by entry with a heartbeat (it holds
    one set of shard files per work unit: thousands at full scale). Tolerates
    a scratch that is already partly or wholly gone, as an interrupted
    removal leaves it."""
    try:
        entries = list(os.scandir(tmp_dir))
    except FileNotFoundError:
        return
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
        except FileNotFoundError:
            pass
        if heartbeat is not None:
            heartbeat()
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)


def _reconcile_counts(n_mol: int, n_conf: int, groups: list, n_ids: int,
                      staging: str) -> None:
    """The scatter's committed totals must be what the gather wrote.

    Committed slots are trusted on resume without re-reading, so a lost or
    damaged shard would otherwise publish a corpus short of conformers (the
    ids gate cannot see it: the ids come from the metadata, not the shards).
    """
    written = sum(int(g["n_conf"]) for g in groups)
    molecules = sum(int(g["n_mol"]) for g in groups)
    if written != int(n_conf) or int(n_ids) != int(n_mol) or molecules != int(n_mol):
        raise BuildRefused(
            f"count reconciliation: the scatter committed {int(n_mol):,} "
            f"molecules and {int(n_conf):,} conformers, but the gather wrote "
            f"{written:,} conformers of {molecules:,} molecules in {len(groups)} "
            f"groups and the metadata holds {int(n_ids):,} ids. Committed scratch "
            f"is missing or damaged, and resuming would trust it again: not "
            f"published; remove {staging} and build afresh.")


def _write_lines_atomic(path: str, lines) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            for record in lines:
                f.write(json.dumps(record) + "\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _source_commit(package_dir: Optional[str] = None) -> Optional[str]:
    """``prep.commit`` (FORMAT.md 10.1: a git commit or null): the commit of
    the checkout the running builder was loaded from, **only** when that
    checkout tracks this very file and its ``src/`` tree has no modified,
    staged or untracked (non-ignored) file; otherwise ``None``.

    ``git -C <dir> rev-parse HEAD`` alone names the HEAD of whatever
    repository encloses the package -- an installed wheel inside an unrelated
    checkout's venv, say -- and does not see a dirty tree, so it could record
    a commit that did not build the corpus. A dirty or unknown tree records
    null; ``build.json``'s code digest identifies what ran either way.
    Informational; not fingerprinted."""
    import subprocess
    here = os.path.abspath(package_dir or os.path.dirname(os.path.abspath(__file__)))

    def git(*args):
        try:
            r = subprocess.run(["git", "-C", here, *args], capture_output=True, text=True,
                               timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    if not top:
        return None
    top = top.strip()
    rel = os.path.relpath(here, top)
    if rel.startswith("..") or git("ls-files", "--error-unmatch", "--",
                                   os.path.join(here, "build.py")) is None:
        return None                       # not this checkout's own builder
    src = os.path.join(top, "src")
    status = git("status", "--porcelain", "--untracked-files=normal", "--",
                 src if os.path.isdir(src) else here)
    if status is None or status.strip():
        return None                       # a dirty tree: no commit describes it
    sha = (git("rev-parse", "HEAD") or "").strip()
    return sha if len(sha) == 40 else None


class _AlreadyPublished(Exception):
    """Control flow: a resumed build found nothing left but publication."""


def _batch_records(source, opts, n_workers) -> int:
    batch_records = opts.records_per_unit
    max_records = getattr(source, "max_records", None)
    if max_records is not None:
        batch_records = min(
            batch_records, max(1, math.ceil(max_records / (2 * n_workers))))
    return batch_records


def _build_color_descriptor(color_generator) -> dict:
    """The build's color descriptor (FORMAT.md section 10.2), or a
    ``BuildRefused`` saying why this generator's definition cannot make a
    format-3 corpus.

    * The generator must describe itself (``descriptor_color``: an RDKit fdef
      or a SMARTS list); the corpus records the definition **text**, never a
      path into the build environment.
    * At most ``LIMITS["n_types"]`` (6) types: the count word (FORMAT.md 8.5).
    * The text must pass the RDKit-free pre-pass (FORMAT.md section 15), or
      every reader would refuse the corpus.
    * A selected family whose blocks declare anything but all-ones atom
      weights is refused, naming the family and the weights: the format's one
      positioning rule is the equal-weight centroid, and no weights are
      stored (FORMAT.md 10.2, Appendix B.11; ``definitions.unequal_weights``).

    A build from the user's own definition needs no trust flag; its hash is
    printed when the build ends.
    """
    from .. import definitions
    from ..format.r3g.errors import CorpusValidationError
    from ..format.r3g.reference import check_definition_text

    describe = getattr(color_generator, "descriptor_color", None)
    if describe is None:
        raise BuildRefused(
            f"color generator {type(color_generator).__name__} cannot describe "
            f"its definition (no descriptor_color): a format-3 corpus records "
            f"the definition text, an RDKit fdef or a SMARTS list")
    color = describe()
    n_types = len(color["types"])
    if not 1 <= n_types <= _R3G_LIMITS["n_types"]:
        raise BuildRefused(
            f"color definition selects {n_types} feature types; a corpus "
            f"carries 1 to {_R3G_LIMITS['n_types']} (FORMAT.md section 8.5)")
    try:
        check_definition_text(color["definition"], color["generator"], color["types"])
    except CorpusValidationError as e:
        raise BuildRefused(
            f"color definition {definitions.definition_hash_of(color)[:16]} fails "
            f"the definition pre-pass every reader runs (FORMAT.md section 15): "
            f"{e}") from None
    bad = definitions.unequal_weights(color)
    if bad:
        raise BuildRefused(
            "color definition declares atom weights format 3 cannot represent "
            "for a selected family: a feature is placed at the equal-weight "
            "mean of its atoms, no weights are stored, and every block of a "
            "selected family must declare all-ones weights (a block without a "
            "Weights line cannot be placed by RDKit at query time) (FORMAT.md "
            "sections 8.3 and 10.2, Appendix B.11): "
            + "; ".join(f"family {fam!r} (feature {name!r}) declares "
                        + ("no Weights line" if w is None else f"Weights {w}")
                        for fam, name, w in bad))
    if color["generator"] == "smarts_list":
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        try:
            for family in color["types"]:
                for pattern in color["definition"][family]:
                    if Chem.MolFromSmarts(pattern) is None:
                        raise BuildRefused(
                            f"color definition: family {family!r}'s SMARTS pattern "
                            f"{pattern!r} does not parse (the validator's "
                            f"re-perception refuses it too)")
        finally:
            RDLogger.EnableLog("rdApp.*")
    from ..format.r3g.reference import interaction_matrix_problems
    problems = interaction_matrix_problems(color["r"]["values"], color["p"]["values"])
    if problems:
        raise BuildRefused("color interaction matrices: " + "; ".join(problems))
    return color


def prepare(inputs, out_dir: str, *, name: Optional[str] = None,
            color_generator=None, workers: int = 0,
            records_per_unit: int = 20_000,
            max_conf_per_group: int = MAX_CONF_PER_GROUP,
            block_rows: int = BLOCK_ROWS,
            note: str = "", keep_temp: bool = False, progress=None,
            input_format: str = "auto", max_records: Optional[int] = None,
            temp_dir: Optional[str] = None, color_self: bool = True,
            resume: bool = False, dry_run: bool = False,
            expect_sha256_ids: Optional[str] = None,
            sqlite_source: Optional[dict] = None,
            expect_ids_from: Optional[str] = None,
            verbose_rejects: bool = False,
            _feature_order: str = "canonical"):
    """Build a format-3 corpus (``doc/FORMAT.md``) at ``out_dir`` and return
    it opened (``format.r3g.reader.open_corpus(out_dir, purpose="inspect")``;
    it has passed ``scan``, recorded in the scan cache when that is writable).

    ``inputs`` is a path, a sequence of paths, or a sequence of RDKit molecules
    with 3D conformers. SMILES are refused: a query and a library both need 3D
    coordinates, and conformer generation is deliberately not part of this
    function.

    The corpus directory holds ``corpus.json``, ``molecules.parquet``,
    ``build.json`` and ``groups/*.r3g``, nothing else. ``out_dir`` must not
    already exist with anything in it. The build runs in the sibling
    ``<out_dir>.partial`` under an exclusive ``<out_dir>.lock``, passes the
    ``scan`` tier there, and is renamed into place only then, so an
    interrupted build leaves no directory that a later run -- or a later
    reader -- can mistake for a corpus. The full lists of rejected molecules
    and dropped conformers, when there are any, are written beside it as
    ``<out_dir>.rejected.jsonl`` and ``<out_dir>.dropped_conformers.jsonl``.

    ``block_rows`` (1..4,096) and ``max_conf_per_group`` are the block and
    group write rules' caps (FORMAT.md sections 6 rule 5 and 9.3); blocks
    never split a molecule. Neither changes a score or the fingerprint; both
    are recorded in ``build.json``. Two builds of the same records with the
    same caps, in any order, with any worker count, resumed or not, write
    byte-identical group files, ``molecules.parquet`` and ``corpus.json``
    given the same roshambo3 release, RDKit and numpy releases and Parquet
    writer release (``molecules.parquet`` up to its ``created_by`` string,
    ``corpus.json`` up to ``prep.commit``); a development version with a null
    ``prep.commit`` claims this only for the same source tree (FORMAT.md 13.3).

    **Resume** (path inputs): every finished scatter work unit and every
    finished group is checkpointed in ``<out_dir>.partial``. After an
    interruption that left committed work -- a kill, a crash, Ctrl-C -- the
    partial build is kept, and ``resume=True`` continues it with identical
    inputs and options, producing the same corpus an uninterrupted build
    would. Without ``resume`` an existing partial build is refused rather than
    silently discarded. Progress is mirrored to ``<out_dir>.progress.json``.

    ``dry_run=True`` returns a dict of the record count (when cheaply
    knowable), estimated CPU time, bytes to read and write and the free space
    at the destination, and creates or writes nothing.

    ``expect_sha256_ids``: the build fails, unpublished, if the corpus's
    ``ids_sha256`` (FORMAT.md 10.6: the sorted ``mol_id`` column as
    little-endian int64, the same hash as a format-2 ``sha256_ids``) differs.
    Expected and actual are recorded either way (``build.json``'s
    ``id_hash_gate``, or the progress record).

    ``expect_ids_from``: a corpus whose molecule ids this build must
    reproduce -- a format-3 corpus directory, or a format-2 one (only its
    ``ids.i64`` and ``manifest.json`` ``sha256_ids`` are read). Every scatter
    work unit is checked against it before it is committed (``mol_id ==
    stable_id(ext_id)``, every id present, no rejected molecule the reference
    holds), so a wrong source or a rejection the reference did not make
    stops at its unit, and the finished build's ``ids_sha256`` must equal
    the reference's unless ``max_records`` makes it a prefix (recorded in
    ``build.json``'s ``id_reference``).

    ``verbose_rejects``: the console reports rejected molecules and dropped
    conformers as counts (by reason code) and a pointer to ``build.json`` and
    the lists beside the corpus, never an identifier or a SMILES, so the log
    of a build that completes names no molecule. A refused build does: its
    message names the molecule (and, for a variant conflict, its SMILES), on
    the console and in ``<out_dir>.progress.json``, so inspect such a log
    before committing it. ``True`` adds the first few identifiers and reasons
    inline, for a public corpus. Neither changes ``build.json`` or the lists.

    ``sqlite_source`` is the sampler for SQLite input, a dict of ``seed``,
    ``p_lo``, ``p_hi``, ``hac_lo``, ``hac_hi`` and ``n_rows`` (see
    :class:`~roshambo3.prep.sqlite_source.SqliteSource`); a tier is defined by
    the source, seed and hash range, which ``build.json`` records.

    ``color_self`` writes each row's color self-overlap (the ``color_self``
    section, FORMAT.md 18.2) for a color build with canonical interaction
    matrices; a non-canonical matrix writes none and ``build.json`` says why.

    **Color.** Each molecule's features are stored once, as a table
    of ``(type, heavy-atom slots)`` (FORMAT.md section 8), perceived on its
    stored SMILES with hydrogens added (as ``deep`` re-perceives it);
    every conformer's pseudo-atoms are
    derived from its stored heavy atoms (integer mean, FORMAT.md 8.3), and
    ``color_self`` is computed on those. The definition text is recorded in
    ``corpus.json``'s ``color`` and its hash printed at the end. The private
    ``_feature_order="enumeration"`` (a measurement switch) is refused:
    such a table is not valid format 3.
    """
    from .. import __version__
    from ..format.r3g import cache as scan_cache
    from ..format.r3g import jsonstrict
    from ..format.r3g.errors import CorpusValidationError
    from ..format.r3g.reader import open_corpus
    from ..format.r3g.validate import corpus_fingerprint, validate_corpus
    from ..format.r3g.writer import descriptor_skeleton, write_descriptor
    from . import rejects as rejects_mod

    out_dir = str(out_dir)
    if expect_sha256_ids is not None:
        expect_sha256_ids = str(expect_sha256_ids).strip().lower()
        if len(expect_sha256_ids) != 64 or any(
                c not in "0123456789abcdef" for c in expect_sha256_ids):
            raise ValueError("expect_sha256_ids must be a 64-digit hex SHA-256")
    if resume and dry_run:
        raise ValueError("dry_run and resume are exclusive")
    if keep_temp and temp_dir is None:
        # Without temp_dir the scratch is <out>.partial/_tmp, which the
        # publishing rename would carry into the corpus directory.
        raise ValueError("keep_temp needs temp_dir: the default scratch lives inside the "
                         "build's staging directory and would be published with the corpus")
    if isinstance(block_rows, bool) or int(block_rows) != block_rows \
            or not 1 <= int(block_rows) <= BLOCK_ROWS:
        raise ValueError(f"block_rows must be an integer in [1, {BLOCK_ROWS}] "
                         f"(FORMAT.md section 6 rule 5)")
    if isinstance(max_conf_per_group, bool) or int(max_conf_per_group) != max_conf_per_group \
            or not 1 <= int(max_conf_per_group) <= _R3G_LIMITS["n_conf"]:
        raise ValueError(f"max_conf_per_group must be an integer in "
                         f"[1, {_R3G_LIMITS['n_conf']}] (FORMAT.md section 9.3, limit 7)")
    feature_order = _feature_order
    if feature_order not in FEATURE_ORDERS:
        raise ValueError(f"_feature_order must be one of {FEATURE_ORDERS}")
    if feature_order != "canonical":
        raise BuildRefused(
            f"_feature_order={feature_order!r} keeps the generator's within-type "
            f"feature order, which is not a valid format-3 feature table (FORMAT.md "
            f"section 8.1 rule 4): the format-3 builder refuses to emit it.")
    id_reference = (_IdReference(expect_ids_from)
                    if expect_ids_from is not None and not dry_run else None)

    color_descriptor = (_build_color_descriptor(color_generator)
                        if color_generator is not None else None)

    source, source_info, default_name = _open_source(
        inputs, color_generator, input_format, max_records, sqlite_source)
    source.feature_order = feature_order
    opts = BuildOptions(name=name or default_name, workers=workers,
                        records_per_unit=records_per_unit,
                        max_conf_per_group=int(max_conf_per_group),
                        block_rows=int(block_rows), note=note,
                        feature_order=feature_order)
    n_workers = (opts.workers or (os.cpu_count() or 1)
                 if getattr(source, "parallel_payloads", False)
                 else opts.workers or min(8, os.cpu_count() or 1))

    if dry_run:
        return _dry_run(source, source_info, out_dir, temp_dir, opts,
                        color_generator, n_workers)

    resumable = source_info["kind"] != "rdkit"
    if resume and not resumable:
        raise ValueError(
            "resume needs path inputs: in-memory molecules have no identity a "
            "later process can check")

    final_dir = out_dir.rstrip(os.sep) or out_dir
    staging = final_dir + ".partial"
    progress_path = final_dir + ".progress.json"
    parent = os.path.dirname(os.path.abspath(final_dir))
    os.makedirs(parent, exist_ok=True)
    started = time.perf_counter()
    status: dict = {}
    current = {"stage": "start", "written": time.monotonic()}

    def report(stage, **extra):
        status.update(extra)
        current.update(stage=stage, written=time.monotonic())
        record = {"stage": stage, "pid": os.getpid(), "updated": time.time(),
                  "elapsed_s": time.perf_counter() - started,
                  "out_dir": final_dir, "staging": staging, **status}
        _write_json_atomic(progress_path, record, durable=False)

    def heartbeat(**extra):
        # Long phases with no other report (a resume replaying the committed
        # prefix of a solid stream, the metadata merge, the validation) still
        # refresh ``updated``, so a watchdog keyed on its staleness never
        # mistakes slow work for a hang. Called from the work loops
        # themselves, so a stuck loop stops it.
        if time.monotonic() - current["written"] >= HEARTBEAT_S:
            report(current["stage"], **extra)

    scan_report = None
    with _writer_lock(final_dir + ".lock"), _terminate_as_exit():
        if os.path.exists(final_dir) and os.listdir(final_dir):
            # Deliberately stricter than "does it have a corpus.json": the
            # descriptor is written last, so keying on it meant a build killed
            # during the gather left group files behind AND passed this check
            # on the next run, which then overwrote them by name.
            raise FileExistsError(
                f"{final_dir} is not empty. If it holds a corpus, choose another "
                f"directory; if it is the wreckage of an interrupted build, "
                f"remove it -- this function will not build on top of either.")
        if os.path.exists(staging) and not resume:
            raise FileExistsError(
                f"{staging} holds an interrupted build. Resume it with "
                f"resume=True (roshambo3 prepare --resume) and the same inputs "
                f"and options, or remove it to start again.")
        if resume and not os.path.isdir(staging):
            raise ValueError(f"resume=True but there is no {staging} to resume")

        if temp_dir is None:
            tmp_dir = os.path.join(staging, "_tmp")
        else:
            # Stable per output, so a resumed build finds its scratch again.
            digest = hashlib.sha256(os.path.abspath(final_dir).encode()).hexdigest()
            tmp_dir = os.path.join(os.path.abspath(temp_dir),
                                   f"roshambo3-prepare-{digest[:16]}")
        batch_records = (_batch_records(source, opts, n_workers)
                         if getattr(source, "parallel_payloads", False) else None)
        color_plan, color_record = _color_self_plan(color_generator, color_self)
        identity = _build_identity(source_info, opts, batch_records,
                                   color_self, color_generator, tmp_dir)
        if not resume:
            os.makedirs(staging)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        checkpoint = None
        try:
            if (resume and not os.path.exists(os.path.join(staging, CHECKPOINT_NAME))
                    and os.path.exists(os.path.join(staging, "corpus.json"))
                    and os.path.exists(os.path.join(staging, "build.json"))):
                # Interrupted between assembly and the rename: build.json is
                # written last and the checkpoint removed only after it, so
                # this is a finished build awaiting publication, and
                # publishing it is all that is left -- once the scratch
                # directory recreated above is gone again. Nothing on disk
                # vouches for these files any more (the checkpoint is gone),
                # so they pass the scan tier again before the rename.
                if not keep_temp:
                    _remove_scratch(tmp_dir, heartbeat)
                report("validate")
                try:
                    scan_report = validate_corpus(staging, "scan")
                except CorpusValidationError as e:
                    raise BuildRefused(
                        f"resume: the finished build awaiting publication in {staging} "
                        f"fails the scan tier ({e.code}: {e}); not published. Remove it "
                        f"and build afresh.") from e
                _publish(staging, final_dir)
                raise _AlreadyPublished
            checkpoint = _BuildCheckpoint(staging, identity, resume=resume,
                                          report=report)
            if checkpoint.state.get("assembled"):
                # The interrupted build may have died removing its scratch;
                # finish that first, or a ``_tmp`` inside the staging
                # directory (and anything left in it) is published as part
                # of the corpus.
                if not keep_temp:
                    _remove_scratch(tmp_dir, heartbeat)
                os.unlink(checkpoint.path)
                _publish(staging, final_dir)
                raise _AlreadyPublished
            if resume:
                _discard_uncommitted(staging, tmp_dir, checkpoint)
            report("scatter", resumes=checkpoint.state["resumes"])

            t0 = time.perf_counter()
            if checkpoint.state["scatter"]["complete"]:
                # Nothing left to scatter: do not re-index the input or
                # re-stream a solid zstd file just to skip every unit.
                done = checkpoint.done_slots()
                n_mol = sum(v["n_mol"] for v in done.values())
                n_conf = sum(v["n_conf"] for v in done.values())
                rejected = [r for slot in sorted(done)
                            for r in checkpoint.slot_rejects(tmp_dir, slot)]
                dropped: dict = {}
                for slot in sorted(done):
                    _add_dropped(dropped, checkpoint.slot_dropped(tmp_dir, slot))
                n_units = int(checkpoint.state["scatter"]["n_units"])
                unit_records = (batch_records if batch_records is not None
                                else opts.records_per_unit)
            else:
                n_mol, n_conf, rejected, n_units, unit_records, dropped = _scatter(
                    source, opts, n_workers, tmp_dir, checkpoint, progress, report,
                    heartbeat, id_reference)
                checkpoint.finish_scatter(n_units)
            t_scatter = time.perf_counter() - t0
            if rejected:
                # Reported, never silent. A build that quietly drops molecules
                # produces a corpus whose molecule count nobody can reconcile
                # with its input. Counts by reason code only: build logs are
                # committed as evidence, and a reject's reason text embeds its
                # SMILES (ids and reasons stay in build.json).
                print(rejects_mod.rejected_message(
                          rejected, final_dir, final_dir + REJECTS_SUFFIX,
                          verbose=verbose_rejects),
                      file=sys.stderr)
            if dropped:
                # A dropped conformer renumbers its molecule's later
                # conformers: conf_idx counts the accepted ones.
                print(rejects_mod.dropped_message(
                          dropped, final_dir, final_dir + DROPPED_SUFFIX,
                          verbose=verbose_rejects),
                      file=sys.stderr)

            prep_commit = _source_commit()
            fingerprint = corpus_fingerprint(
                descriptor_skeleton(color_descriptor, __version__, prep_commit))
            report("gather")
            t0 = time.perf_counter()
            groups = _gather(staging, tmp_dir, opts, fingerprint, color_self=color_plan,
                             checkpoint=checkpoint, heartbeat=heartbeat,
                             n_types=(len(color_generator.FEATURES_ENUM)
                                      if color_generator is not None else None))
            t_gather = time.perf_counter() - t0
            report("metadata")
            t0 = time.perf_counter()
            n_ids, actual_ids = _merge_meta(staging, tmp_dir, heartbeat=heartbeat)
            t_metadata = time.perf_counter() - t0
            _reconcile_counts(n_mol, n_conf, groups, n_ids, staging)

            # A path build keeps its staging (it holds committed work); an
            # in-memory one cannot be resumed, so its staging is removed.
            left = (f"the build is left at {staging} (resume=True / --resume after "
                    f"fixing the cause, or remove it)" if resumable else
                    "an in-memory build cannot be resumed, so nothing is kept")
            gate = {"expected": expect_sha256_ids, "actual": actual_ids,
                    "matched": (None if expect_sha256_ids is None
                                else actual_ids == expect_sha256_ids)}
            if gate["matched"] is False:
                report("failed", id_hash_gate=gate,
                       error="ids_sha256 does not match the expected value")
                raise BuildRefused(
                    f"id-hash gate: the built corpus's ids_sha256 is "
                    f"{actual_ids}, expected {expect_sha256_ids}. Not "
                    f"published; {left}.")
            reference_record = None
            if id_reference is not None:
                reference_record = id_reference.record(actual_ids, max_records)
                print(f"roshambo3: id reference {id_reference.root} "
                      f"({id_reference.kind}): {id_reference.checked:,} molecules "
                      f"checked as they were prepared; final ids_sha256 "
                      f"{reference_record['final_comparison']}"
                      + ("" if reference_record["matched"] is None else
                         f" -> {'match' if reference_record['matched'] else 'MISMATCH'}"),
                      file=sys.stderr)
                if reference_record["matched"] is False:
                    report("failed", id_reference=reference_record,
                           error="ids_sha256 does not match the reference corpus")
                    raise BuildRefused(
                        f"id reference: the built corpus's ids_sha256 is "
                        f"{actual_ids}, but {id_reference.root} has "
                        f"{id_reference.sha256_ids}. Not published; {left}.")

            report("descriptor")
            descriptor = write_descriptor(
                staging, color=color_descriptor, prep_version=__version__,
                prep_commit=prep_commit, groups=groups, ids_sha256=actual_ids)

            # The native encoder is new code: every group file, the
            # descriptor and molecules.parquet pass the scan tier before the
            # build is declared finished (each block was also decoded back as
            # it was written).
            report("validate")
            t0 = time.perf_counter()
            try:
                scan_report = validate_corpus(staging, "scan")
            except CorpusValidationError as e:
                raise BuildRefused(
                    f"the built corpus fails the scan tier ({e.code}: {e}); this is a "
                    f"builder defect. Not published; {left}.") from e
            if scan_report.fingerprint != fingerprint:
                raise RuntimeError("the validated fingerprint differs from the one the "
                                   "group files were written with")
            t_validate = time.perf_counter() - t0
            stats = _corpus_statistics(staging, descriptor, heartbeat=heartbeat)

            report("build.json")
            ordered = sorted(dropped.items())
            build = {
                "schema": "roshambo3.build.v1",
                "name": opts.name, "note": opts.note,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "host": os.uname().nodename,
                "roshambo3_version": __version__, "commit": prep_commit,
                "source": source_info,
                "parameters": {
                    "block_rows": int(opts.block_rows),
                    "max_conf_per_group": int(opts.max_conf_per_group),
                    "records_per_unit": int(unit_records), "n_units": int(n_units),
                    "color_self": bool(color_self),
                    "feature_order": opts.feature_order,
                },
                # How this run went, which changes nothing in the corpus.
                "run": {"workers": int(n_workers)},
                "timing": {"scatter_s": t_scatter, "gather_s": t_gather,
                           "metadata_s": t_metadata, "validate_s": t_validate,
                           **getattr(source, "stats", {})},
                "identity": identity,
                "resumes": int(checkpoint.state["resumes"]),
                "fingerprint": fingerprint,
                "validation": {"tier": "scan", "seconds": scan_report.seconds,
                               "backend": scan_report.backend},
                "statistics": stats,
                "id_hash_gate": gate,
                "id_reference": reference_record,
                "rejected": {
                    "n": len(rejected),
                    "reasons": sorted({w for _, w in rejected})[:20],
                    "examples": [{"id": i, "why": w} for i, w in rejected[:50]],
                    "file": (os.path.basename(final_dir) + REJECTS_SUFFIX
                             if rejected else None)},
                "dropped_conformers": {
                    "n_conformers": int(sum(dropped.values())),
                    "n_molecules": len(dropped),
                    "conf_idx": "0-based index among the molecule's accepted "
                                "conformers, in source order; a dropped conformer "
                                "renumbers the ones after it",
                    "examples": [{"id": i, "n": int(n)} for i, n in ordered[:50]],
                    "file": (os.path.basename(final_dir) + DROPPED_SUFFIX
                             if dropped else None)},
                "scratch": {"external": temp_dir is not None, "path": tmp_dir,
                            "bytes": sum(os.path.getsize(os.path.join(dp, fn))
                                         for dp, _, fns in os.walk(tmp_dir) for fn in fns)},
            }
            if color_descriptor is not None:
                from .. import definitions
                h = definitions.definition_hash_of(color_descriptor)
                build["color"] = {
                    "definition_sha256": h,
                    "shipped": bool(color_descriptor["generator"] == "rdkit_fdef"
                                    and definitions.is_shipped(h)),
                    "generator_class": type(color_generator).__name__,
                    "fdef_path": getattr(color_generator, "fdefName", None),
                    "color_self": color_record,
                }
            build_bytes = jsonstrict.dumps(build)
            jsonstrict.loads(build_bytes, what="build.json")
            # The full lists beside the corpus; a list left there by an
            # earlier build of this output is removed when this one has none.
            for suffix, lines in ((REJECTS_SUFFIX, [{"id": i, "why": w} for i, w in rejected]),
                                  (DROPPED_SUFFIX, [{"id": i, "n": int(n)} for i, n in ordered])):
                if lines:
                    _write_lines_atomic(final_dir + suffix, lines)
                elif os.path.exists(final_dir + suffix):
                    os.unlink(final_dir + suffix)
            with open(os.path.join(staging, "build.json"), "wb") as f:
                f.write(build_bytes)

            # Durable before the checkpoint says so: a resume after this point
            # only publishes, without looking at these files again.
            _sync_assembled(staging, heartbeat)
            # Assembled: from here a resume only publishes.
            checkpoint.state["assembled"] = True
            checkpoint.save()
            if not keep_temp:
                _remove_scratch(tmp_dir, heartbeat)
            os.unlink(checkpoint.path)
            report("publish")
            _publish(staging, final_dir)
        except _AlreadyPublished:
            pass
        except BaseException as exc:
            # A resumed build never deletes what it was asked to resume -- not
            # even when it refuses to (an identity mismatch). A fresh build
            # keeps its staging only once it holds committed work.
            keep = resumable and (resume or (
                checkpoint is not None and checkpoint.committed))
            if keep:
                # Committed work survives for resume=True; the record says why
                # the build stopped. (A kill -9 never reaches here, and leaves
                # the same resumable state.)
                stage = ("paused" if isinstance(exc, (KeyboardInterrupt, SystemExit))
                         else "failed")
                print(f"roshambo3: build {stage} ({type(exc).__name__}: {exc}); "
                      f"committed work is kept in {staging}"
                      + ("" if getattr(exc, "retryable", True) is False else
                         "; continue it with --resume (resume=True)"),
                      file=sys.stderr)
                try:
                    if (checkpoint is not None and os.path.isdir(staging)
                            and not os.path.exists(checkpoint.path)):
                        checkpoint.save()
                    report(stage, error=f"{type(exc).__name__}: {exc}",
                           resumable=os.path.isdir(staging),
                           retryable=bool(getattr(exc, "retryable", True)))
                except Exception:
                    pass
            else:
                # Nothing worth resuming: no litter beside the target that a
                # later run would have to explain.
                shutil.rmtree(staging, ignore_errors=True)
                if temp_dir is not None:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                if os.path.exists(progress_path):
                    os.unlink(progress_path)
            raise
        if os.path.exists(progress_path):
            os.unlink(progress_path)
    try:
        os.unlink(final_dir + ".lock")
    except FileNotFoundError:
        pass
    if scan_report is not None:
        # The scan passed on these exact files (a rename changes neither
        # their bytes nor their size and mtime, which the cache key holds), so
        # the published corpus opens for a screen without a second scan.
        scan_cache.record(scan_cache.scan_key(scan_report.descriptor_sha256,
                                              scan_report.header_sha256,
                                              scan_report.file_stats),
                          tier="scan", fingerprint=scan_report.fingerprint, root=final_dir)
    if color_descriptor is not None:
        _print_definition_hash(color_descriptor)
    return open_corpus(final_dir, purpose="inspect")


def _print_definition_hash(color: dict) -> None:
    """The definition's hash, at the end of a color build, so it can be
    handed to whoever screens the corpus (FORMAT.md 10.3)."""
    from .. import definitions
    h = definitions.definition_hash_of(color)
    if color["generator"] == "rdkit_fdef" and definitions.is_shipped(h):
        what = "a shipped RDKit definition; readers need no trust"
    else:
        what = (f"a custom definition; to screen this corpus elsewhere, "
                f"`roshambo3 definitions trust {h}` there")
    print(f"roshambo3: color definition sha256 {h} ({what})", file=sys.stderr)

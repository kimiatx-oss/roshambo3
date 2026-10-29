"""Resolve corpus-local rows and durable molecule/conformer identities.

Metadata joins use mol_id, never molecule-table row order. Coordinates are
read through the validated group API and decoded once per touched block.
The optional reverse identity index is built only when requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np


if TYPE_CHECKING:  # pragma: no cover
    from .r3g.reader import Corpus


@dataclass(frozen=True)
class ConformerRef:
    """Where a global conformer id lives, and what molecule it belongs to."""

    gid: int
    group_index: int
    row: int
    """Row within the group, i.e. ``gid`` minus the group's first gid."""
    mol_id: int
    conf_idx: int
    n_real: int
    vaa: float
    """Precomputed Gaussian self-overlap of this conformer."""


class Catalog:
    """Random access from global conformer ids into a corpus.

        catalog = Catalog(corpus)
        refs = catalog.locate(gids)             # ids -> group/row/molecule
        meta = catalog.molecules(gids)          # ids -> ext_id, smiles, ...
        xyz  = catalog.coordinates(gids)        # ids -> float32 [n, L_i, 3] (ragged)

    Run tables are read lazily per group. Coordinates are decoded only for
    requested rows; metadata access does not decode coordinate payloads.
    """

    def __init__(self, corpus: "Corpus", run_tables: Optional[dict] = None):
        """``run_tables`` optionally maps a group index to that group's run
        table (its ``runs`` section) as a caller already read it -- a screen's
        source holds every group's -- so resolving a screen's hits does not
        reopen every group a second time. Each table is checked
        against the group's descriptor entry before it is used."""
        self.corpus = corpus
        counts = np.asarray([int(entry["n_conf"]) for entry in corpus.group_table],
                            dtype=np.int64)
        #: First gid of each group; ``offsets[-1]`` is the corpus total.
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        self._sidecars: list[Optional[dict]] = [None] * counts.size
        self._meta_index: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._columns: Optional[list[str]] = None
        self._rg_bounds: Optional[np.ndarray] = None
        # Format-3 run tables are loaded per group, on first use. A Catalog
        # used for one small result must not map/copy every molecule table.
        # The reverse identity index is even more expensive and is built only
        # if gid_for() is actually requested (web overlays).
        self._lookup_runs: Optional[np.ndarray] = None
        self._run_tables: dict = dict(run_tables or {})

    def _build_reverse_lookup(self) -> None:
        """Build the optional reverse index in one bounded-memory pass.

        The unaligned structured row is 18 bytes (mol_id, gid_start, n_conf).
        Filling it in place avoids the three full-size concatenations and the
        argsort permutation that otherwise multiply peak memory on large
        corpora. Group files are opened and closed through the format-3 reader.
        """
        if self._lookup_runs is not None:
            return
        dtype = np.dtype([("mol_id", "<i8"), ("gid_start", "<i8"),
                          ("n_conf", "<u2")], align=False)
        lookup = np.empty(int(self.corpus.n_mol), dtype=dtype)
        cursor = 0
        for gi in range(len(self.offsets) - 1):
            with self.corpus.group(gi) as group:
                runs = group.runs
                n = int(runs.size)
                lookup["mol_id"][cursor:cursor + n] = runs["mol_id"]
                lookup["gid_start"][cursor:cursor + n] = (
                    self.offsets[gi] + group.run_start[:-1])
                lookup["n_conf"][cursor:cursor + n] = runs["n_conf"]
            cursor += n
        if cursor != lookup.size:
            raise ValueError(
                f"run tables contain {cursor} molecules; descriptor says {lookup.size}")
        lookup.sort(order="mol_id", kind="stable")
        self._lookup_runs = lookup

    @property
    def n_conformers(self) -> int:
        return int(self.offsets[-1])

    # -- gid -> position ----------------------------------------------------

    def _split(self, gids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(group_index, row)`` for each gid, validating the range."""
        gids = np.asarray(gids, dtype=np.int64)
        if gids.size and (gids.min() < 0 or gids.max() >= self.n_conformers):
            bad = gids[(gids < 0) | (gids >= self.n_conformers)][:5]
            raise IndexError(
                f"conformer id(s) {bad.tolist()} outside this corpus "
                f"[0, {self.n_conformers})")
        gi = np.searchsorted(self.offsets, gids, side="right") - 1
        return gi, gids - self.offsets[gi]

    def _group_sidecars(self, gi: int) -> dict:
        if self._sidecars[gi] is None and gi in self._run_tables:
            # Popped: the table is either adopted below or rejected, and in
            # neither case does the Catalog keep a second reference to it.
            runs = self._run_tables.pop(gi)
            entry = self.corpus.group_table[gi]
            n_conf = np.asarray(runs["n_conf"], dtype=np.int64)
            # The checks Group.run_start makes, against the descriptor rather
            # than the header the caller already validated. A table that fails
            # them is not used: the group is read below as it always was.
            if (n_conf.size == int(entry["n_mol"]) and n_conf.size
                    and int(n_conf.min()) >= 1
                    and int(n_conf.sum()) == int(entry["n_conf"])):
                self._sidecars[gi] = {
                    "runs": runs,
                    "run_start": np.concatenate([[0], np.cumsum(n_conf)]),
                }
        if self._sidecars[gi] is None:
            with self.corpus.group(gi) as group:
                # Own compact arrays rather than leaving mmap-backed Group
                # objects open in a long-lived Catalog. BoundedCatalog can
                # evict this dictionary without invalidating caller views.
                self._sidecars[gi] = {
                    "runs": group.runs.copy(),
                    "run_start": group.run_start.copy(),
                }
        return self._sidecars[gi]

    def locate(self, gids: Sequence[int] | np.ndarray) -> list[ConformerRef]:
        """Resolve gids to :class:`ConformerRef`, preserving input order."""
        gids = np.asarray(gids, dtype=np.int64)
        gi, row = self._split(gids)
        mol_id = np.empty(gids.size, dtype=np.int64)
        conf_idx = np.empty(gids.size, dtype=np.int64)
        n_real = np.empty(gids.size, dtype=np.int64)
        for g in np.unique(gi):
            sel = np.flatnonzero(gi == g)
            sidecars = self._group_sidecars(int(g))
            runs = sidecars["runs"]
            starts = sidecars["run_start"]
            m = np.searchsorted(starts, row[sel], side="right") - 1
            mol_id[sel] = runs["mol_id"][m]
            conf_idx[sel] = row[sel] - starts[m]
            n_real[sel] = runs["n_real"][m]
        vaa = np.empty(gids.size, dtype=np.float32)
        for g in np.unique(gi):
            sel = np.flatnonzero(gi == g)
            with self.corpus.group(int(g)) as group:
                vaa[sel] = group.vaa[row[sel]]
        return [
            ConformerRef(gid=int(gid), group_index=int(g), row=int(r),
                         mol_id=int(mid), conf_idx=int(ci), n_real=int(nr),
                         vaa=float(v))
            for gid, g, r, mid, ci, nr, v in zip(
                gids.tolist(), gi.tolist(), row.tolist(), mol_id.tolist(),
                conf_idx.tolist(), n_real.tolist(), vaa.tolist())
        ]

    def mol_ids(self, gids: Sequence[int] | np.ndarray) -> np.ndarray:
        """``mol_id`` for each gid -- the cheap path when only identity matters."""
        gids = np.asarray(gids, dtype=np.int64)
        gi, row = self._split(gids)
        out = np.empty(gids.size, dtype=np.int64)
        for g in np.unique(gi):
            sel = np.flatnonzero(gi == g)
            sidecars = self._group_sidecars(int(g))
            starts = sidecars["run_start"]
            m = np.searchsorted(starts, row[sel], side="right") - 1
            out[sel] = sidecars["runs"]["mol_id"][m]
        return out

    def conf_indices(self, gids: Sequence[int] | np.ndarray) -> np.ndarray:
        """``conf_idx`` within each molecule's run for every gid."""
        gids = np.asarray(gids, dtype=np.int64)
        gi, row = self._split(gids)
        out = np.empty(gids.size, dtype=np.int64)
        for g in np.unique(gi):
            sel = np.flatnonzero(gi == g)
            starts = self._group_sidecars(int(g))["run_start"]
            m = np.searchsorted(starts, row[sel], side="right") - 1
            out[sel] = row[sel] - starts[m]
        return out

    def gid_for(self, mol_id: int, conf_idx: int) -> int:
        """Resolve a durable ``(mol_id, conf_idx)`` identity to this corpus's gid."""
        if isinstance(mol_id, (bool, np.bool_)) or isinstance(conf_idx, (bool, np.bool_)):
            raise TypeError("mol_id and conf_idx must be integers")
        try:
            mid, ci = int(mol_id), int(conf_idx)
        except (TypeError, ValueError, OverflowError):
            raise TypeError("mol_id and conf_idx must be integers") from None
        if mid != mol_id or ci != conf_idx:
            raise TypeError("mol_id and conf_idx must be integers")
        self._build_reverse_lookup()
        lookup = self._lookup_runs
        ids = lookup["mol_id"]
        i = int(np.searchsorted(ids, mid))
        if i >= ids.size or int(ids[i]) != mid:
            raise KeyError(f"mol_id {mid} is not in this corpus")
        n_conf = int(lookup["n_conf"][i])
        if not 0 <= ci < n_conf:
            raise IndexError(f"conf_idx {ci} outside molecule {mid}'s [0, {n_conf}) run")
        return int(lookup["gid_start"][i]) + ci

    # -- molecule metadata --------------------------------------------------

    def _meta_ids(self) -> tuple[np.ndarray, np.ndarray]:
        """``(sorted mol_id, its row in molecules.parquet)``, read once.

        Only the ``mol_id`` column is read here. The columns a caller actually
        asked for are read separately and taken by row, because on a
        540M-conformer library the table is 54,029,188 rows and materializing a
        whole string column to answer a 1,000-row query costs about fifteen
        seconds.
        """
        if self._meta_index is None:
            ids_by_group = []
            bounds = [0]
            for table in self.corpus.molecules(columns=["mol_id"]):
                ids_by_group.append(table.column("mol_id").to_numpy().astype(
                    np.int64, copy=False))
                bounds.append(bounds[-1] + table.num_rows)
            ids = (np.concatenate(ids_by_group) if ids_by_group
                   else np.empty(0, dtype=np.int64))
            self._rg_bounds = np.asarray(bounds[:-1], dtype=np.int64)
            order = np.argsort(ids, kind="stable")
            self._meta_index = (ids[order], order)
        return self._meta_index

    def molecules(self, gids: Sequence[int] | np.ndarray,
                  columns: Optional[Sequence[str]] = None) -> dict:
        """Per-hit molecule metadata, joined on ``mol_id``.

        Returns a dict of column name -> list, aligned with ``gids``, plus a
        boolean ``found`` list.

        A ``mol_id`` absent from ``molecules.parquet`` yields ``None`` in every column
        and ``found=False``, rather than the wrong neighboring row -- a missing
        molecule is a corpus defect worth seeing, not something to paper over.
        ``found`` exists because ``None`` is otherwise ambiguous: ``ext_id`` is
        legitimately null for every PubChem-sourced molecule, so a null column
        value is not evidence the join failed.
        """
        want = self.mol_ids(gids)
        sorted_ids, order = self._meta_ids()

        pos = np.clip(np.searchsorted(sorted_ids, want), 0, max(sorted_ids.size - 1, 0))
        hit = sorted_ids[pos] == want if sorted_ids.size else np.zeros(want.size, bool)
        rows = order[pos]

        out = {"mol_id": want.tolist(), "found": hit.tolist()}
        cols = [c for c in (columns or self._all_columns()) if c not in ("mol_id", "found")]
        if not cols:
            return out

        picked = self._read_rows(rows[hit], cols)
        for name in cols:
            values = picked.get(name, [])
            it = iter(values)
            out[name] = [next(it) if h else None for h in hit.tolist()]
        return out

    def _read_rows(self, rows: np.ndarray, cols: Sequence[str]) -> dict:
        """Read selected molecule-table rows through the validated reader."""
        if rows.size == 0:
            return {c: [] for c in cols}
        return self._read_r3g_rows(rows, cols)

    def _read_r3g_rows(self, rows: np.ndarray, cols: Sequence[str]) -> dict:
        """Select hit metadata from the format-3 reader's streamed row groups."""
        import pyarrow as pa

        bounds = self._row_group_bounds()
        rg_of = np.searchsorted(bounds, rows, side="right") - 1
        values: dict[str, list] = {c: [None] * rows.size for c in cols}
        for rg, table in enumerate(self.corpus.molecules(columns=list(cols))):
            sel = np.flatnonzero(rg_of == rg)
            if not sel.size:
                continue
            local = rows[sel] - bounds[rg]
            picked = table.take(pa.array(local, type=pa.int64()))
            for name in cols:
                for k, value in zip(sel.tolist(), picked.column(name).to_pylist()):
                    values[name][k] = value
        return values

    def _row_group_bounds(self) -> np.ndarray:
        """First global row index of each molecule-table row group."""
        if self._rg_bounds is None:
            self._meta_ids()
        return self._rg_bounds

    def _all_columns(self) -> list[str]:
        if self._columns is None:
            self._columns = ["mol_id", "ext_id", "smiles", "hac", "mw", "rotors"]
        return self._columns

    # -- coordinates --------------------------------------------------------

    def coordinates(self, gids: Sequence[int] | np.ndarray,
                    trim: bool = True) -> list[np.ndarray]:
        """Decode each gid's coordinates, in Angstrom, in input order.

        With ``trim`` (the default) each array is ``[n_real, 3]`` -- the
        conformer's real atoms only. Without it, the group's full padded width
        ``[L, 3]`` is returned, padding included.

        Blocks are decoded once and shared across all gids that land in them, so
        asking for a scattered top-K costs one decode per distinct block rather
        than one per hit.
        """
        gids = np.asarray(gids, dtype=np.int64)
        gi, row = self._split(gids)

        return self._r3g_coordinates(gids, gi, row, trim)

    def _r3g_coordinates(self, gids: np.ndarray, gi: np.ndarray, row: np.ndarray,
                         trim: bool) -> list[np.ndarray]:
        """Decode each touched block once through the reader's Group API."""
        want: dict[int, dict[int, list[int]]] = {}
        row_in_block = np.empty(gids.size, dtype=np.int64)
        out: list[Optional[np.ndarray]] = [None] * gids.size
        for group_index in np.unique(gi):
            selected = np.flatnonzero(gi == group_index)
            with self.corpus.group(int(group_index)) as group:
                blocks = group.blocks
                for k in selected.tolist():
                    group_row = int(row[k])
                    b = int(np.searchsorted(
                        blocks["first_row"], group_row, side="right") - 1)
                    if b < 0 or group_row >= int(
                            blocks[b]["first_row"] + blocks[b]["rows"]):
                        raise IndexError(
                            f"row {group_row} is not covered by a block in group {group_index}")
                    row_in_block[k] = group_row - int(blocks[b]["first_row"])
                    want.setdefault(int(group_index), {}).setdefault(b, []).append(k)

        for group_index, blocks_wanted in want.items():
            with self.corpus.group(group_index) as group:
                sidecars = self._group_sidecars(group_index)
                scale = np.float32(self.corpus.step)
                for block_index, positions in blocks_wanted.items():
                    xyz = group.decode_block(block_index)  # int16[rows, 3, L_heavy]
                    for k in positions:
                        coords = xyz[int(row_in_block[k])].T.astype(np.float32) * scale
                        if trim:
                            run = np.searchsorted(sidecars["run_start"], row[k],
                                                  side="right") - 1
                            coords = coords[:int(sidecars["runs"]["n_real"][run])]
                        out[k] = coords
        return out  # type: ignore[return-value]

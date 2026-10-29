"""What a screen returns.

Scores and ids by default; poses and molecules on request.

That default is not a limitation, it is the point. The device-resident top-K
returns score columns and an int64 global conformer id per hit, which is what
keeps the dense block off the bus -- 64 GB of copy-back eliminated on a
1e8-conformer tier. Materializing poses for every retained row would put the
per-batch deep copy the streaming engine exists to avoid straight back in.

So :class:`ScreenResult` holds the table, and reaches back into the corpus
through :class:`roshambo3.format.Catalog` only for the rows a caller actually
asks about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from . import scores as S


def spreadsheet_safe_text(value):
    """Prefix spreadsheet formula markers in corpus-derived text cells."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


@dataclass
class QueryHits:
    """One query's hits, in canonical order (score descending, ties by id)."""

    query_name: str
    query_index: int
    gids: np.ndarray
    """Global conformer ids."""
    scores: np.ndarray
    """``[n, 20]`` float32; see :mod:`roshambo3.scores` for the columns."""
    rank_col: int = 0
    """Which of the 20 score columns the rows were ranked on.

    It cannot be recovered from the arrays -- every row carries all the columns,
    so a list ranked on column 7 looks exactly like one ranked on column 0 --
    and a consumer that re-ranks on the wrong one silently returns a different
    order. :meth:`ScreenResult.distinct_molecules_in_hits` is such a consumer."""
    level: str = "conformer"
    """What one row is: ``"conformer"`` or ``"molecule"``.

    Not cosmetic. At ``"conformer"`` the rows are the best conformers and a
    molecule may hold several of them, so this list is **not** the top-n
    molecules and no compound-level statistic can be read off it. At
    ``"molecule"`` each row is a distinct molecule scored by its best
    conformer, and ``gids`` names that conformer."""
    mol_id: Optional[np.ndarray] = None
    """Durable molecule identity for each hit, parallel to ``gids``."""
    conf_idx: Optional[np.ndarray] = None
    """Conformer offset within its molecule, parallel to ``gids``."""

    def __len__(self) -> int:
        return int(self.gids.size)

    def __repr__(self) -> str:
        return f"<QueryHits {self.query_name!r} n={len(self)}>"

    @property
    def tanimoto_shape(self) -> np.ndarray:
        return self.scores[:, S.TANIMOTO_SHAPE]

    @property
    def tanimoto_color(self) -> np.ndarray:
        return self.scores[:, S.TANIMOTO_COLOR]

    @property
    def tanimoto_combination(self) -> np.ndarray:
        return self.scores[:, S.TANIMOTO_COMBINATION]


@dataclass
class ScreenResult:
    """A screen's hits, with the corpus kept to hand for re-gathering."""

    hits: list[QueryHits]
    corpus: Any = None
    stats: Any = None
    _catalog: Any = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, i) -> QueryHits:
        return self.hits[i]

    def __iter__(self):
        return iter(self.hits)

    def __repr__(self) -> str:
        return (f"<ScreenResult {len(self.hits)} queries, "
                f"{sum(len(h) for h in self.hits)} hits>")

    @property
    def catalog(self):
        if self._catalog is None:
            if self.corpus is None:
                raise RuntimeError(
                    "this result has no corpus attached, so hits cannot be "
                    "resolved back to molecules")
            from .format import Catalog
            self._catalog = Catalog(self.corpus)
        return self._catalog

    def _ensure_hit_identity(self, hit: QueryHits) -> None:
        """Fill durable ids for manually assembled results that lack them."""
        if hit.mol_id is not None and hit.conf_idx is not None:
            return
        if self.corpus is None:
            raise RuntimeError(
                "this result has no corpus identity and its QueryHits do not carry "
                "mol_id/conf_idx")
        refs = self.catalog.locate(hit.gids)
        hit.mol_id = np.asarray([ref.mol_id for ref in refs], dtype=np.int64)
        hit.conf_idx = np.asarray([ref.conf_idx for ref in refs], dtype=np.int64)

    # -- re-gather ---------------------------------------------------------

    def hit_molecules(self, query: int = 0, n: Optional[int] = None,
                      columns: Optional[Sequence[str]] = None) -> dict:
        """Metadata for one query's top ``n`` hits -- **one row per hit**.

        Joined on ``mol_id``, never on row order: ``meta.parquet`` has one row
        per molecule and the sidecars one per conformer.

        On a conformer-level result a molecule with *k* conformers among the
        hits appears *k* times, with *k* identical ``ext_id``/``smiles`` values
        and *k* distinct ``gid``s. That is what the rows *are*, and the name
        says so: this method used to be called ``molecules()``, which read as
        molecule-level and was not, so "the top 100 molecules" silently returned
        fewer than 100 distinct molecules with no indication of how many fewer.
        Use :meth:`molecules` for the molecule-level answer and
        :meth:`distinct_molecules_in_hits` to see how far apart the two are.
        """
        h = self.hits[query]
        self._ensure_hit_identity(h)
        gids = h.gids if n is None else h.gids[:n]
        out = self.catalog.molecules(gids, columns=columns)
        out["gid"] = gids.tolist()
        out["mol_id"] = np.asarray(h.mol_id[:len(gids)], dtype=np.int64).tolist()
        out["conf_idx"] = np.asarray(h.conf_idx[:len(gids)], dtype=np.int64).tolist()
        return out

    def molecules(self, query: int = 0, n: Optional[int] = None,
                  columns: Optional[Sequence[str]] = None) -> dict:
        """The top ``n`` **molecules**, one row each, best conformer first.

        Requires a molecule-level screen (``r3.screen(..., level="molecule")``
        or :class:`~roshambo3.engine.MoleculeTopK`). A conformer-level result
        does not contain the answer -- not merely in the wrong shape: a molecule
        that would rank 40th on its best conformer can be absent from a
        conformer-level top 100 that another molecule's near-duplicates filled.
        So this refuses rather than deduplicating and calling the result a
        top-n.

        ``gid`` names the conformer that scored best, so a pose is still
        recoverable through :meth:`poses` and :meth:`aligned_poses`.
        """
        h = self.hits[query]
        if h.level != "molecule":
            raise ValueError(
                f"this result is {h.level}-level, so it does not contain the "
                f"top-n molecules: a molecule ranked inside the top n on its "
                f"best conformer can be missing from a conformer-level top n "
                f"that another molecule's near-duplicates filled. Re-screen "
                f"with level='molecule' (or engine.MoleculeTopK) for the exact "
                f"answer, or call distinct_molecules_in_hits() for the "
                f"molecules that happen to be present in this list.")
        return self.hit_molecules(query, n, columns)

    def distinct_molecules_in_hits(self, query: int = 0,
                                   n: Optional[int] = None,
                                   columns: Optional[Sequence[str]] = None
                                   ) -> dict:
        """The distinct molecules present in one query's top ``n`` hits.

        Each molecule once, keeping its best-scoring hit, in canonical order
        (descending score, ties by ascending conformer id). ``n`` cuts the
        **hit** list before the deduplication, because that is the operation
        being described: "I asked for n hits, how many molecules did I get".

        This is **not** the top-n molecules unless the result is already
        molecule-level -- see :meth:`molecules`. It exists to make the gap
        visible: ``n_hits`` and ``n_molecules`` are both returned, and
        ``n_conformers_in_hits`` says how many slots each molecule took.
        """
        from .numerics import rank_order

        h = self.hits[query]
        cut = h.gids.size if n is None else min(int(n), h.gids.size)
        self._ensure_hit_identity(h)
        gids = np.asarray(h.gids[:cut], dtype=np.int64)
        rank = np.asarray(h.scores[:cut, h.rank_col], dtype=np.float32)
        mol = np.asarray(h.mol_id[:cut], dtype=np.int64)
        conf = np.asarray(h.conf_idx[:cut], dtype=np.int64)

        # Preserve the screen's ranking policy within this corpus. Durable
        # (mol_id, conf_idx) keys are emitted for cross-build comparisons,
        # where gid must not be used as the join key.
        order = rank_order(rank, gids)
        _, first = np.unique(mol[order], return_index=True)
        keep = order[np.sort(first)]
        uniq, inv = np.unique(mol, return_inverse=True)
        counts = np.bincount(inv, minlength=uniq.size)

        out = self.catalog.molecules(gids[keep], columns=columns)
        out["gid"] = gids[keep].tolist()
        out["conf_idx"] = conf[keep].tolist()
        out["n_conformers_in_hits"] = counts[inv[keep]].tolist()
        out["n_hits"] = int(gids.size)
        out["n_molecules"] = int(keep.size)
        return out
        for c in range(h.scores.shape[1]):
            col = h.scores[:, c]
            if np.all(np.diff(col.astype(np.float64)) <= 0):
                return c
        return 0

    def poses(self, query: int = 0, n: Optional[int] = None) -> list[np.ndarray]:
        """Hit coordinates in the corpus frame, for one query's top ``n``.

        These are the *stored* coordinates, not yet moved onto the query. Use
        :meth:`aligned_poses` for that.
        """
        h = self.hits[query]
        gids = h.gids if n is None else h.gids[:n]
        return self.catalog.coordinates(gids)

    def aligned_poses(self, query: int = 0, n: Optional[int] = None
                      ) -> list[np.ndarray]:
        """Hit coordinates moved onto the query, by the pose the optimizer found.

        Applies ``r @ R(q).T + t`` and then the start-mode rotation, which is the
        transform the score columns describe. On the field path the start-mode
        rotation is the identity, because it is already folded into the reported
        pose.

        The result is in the *query's prepared frame*. Putting it back into the
        query's original input frame additionally needs that query's preparation
        transform, which a caller who read the query from a file has and this
        object does not.
        """
        h = self.hits[query]
        k = len(h) if n is None else min(n, len(h))
        coords = self.poses(query, k)
        out = []
        for i in range(k):
            row = h.scores[i]
            q = row[S.POSE_QUAT]
            t = row[S.POSE_TRANSLATION]
            qs = row[S.START_QUAT]
            r = np.asarray(coords[i], dtype=np.float64)
            out.append(_apply(_quat_matrix(qs), _apply(_quat_matrix(q), r) + t))
        return out

    def to_table(self, query: Optional[int] = None) -> dict:
        """A flat, plain-Python table of hits. Column names, not indices."""
        rows = self.hits if query is None else [self.hits[query]]
        cols: dict[str, list] = {"query": [], "gid": [], "mol_id": [], "conf_idx": []}
        for name in S.COLUMN_INDEX:
            cols[name] = []
        for h in rows:
            self._ensure_hit_identity(h)
            for i in range(len(h)):
                cols["query"].append(h.query_name)
                cols["gid"].append(int(h.gids[i]))
                cols["mol_id"].append(int(h.mol_id[i]))
                cols["conf_idx"].append(int(h.conf_idx[i]))
                for name, idx in S.COLUMN_INDEX.items():
                    cols[name].append(float(h.scores[i, idx]))
        return cols

    def to_pandas(self, query: Optional[int] = None):
        """The same table as a DataFrame. pandas is an optional dependency."""
        import pandas as pd
        return pd.DataFrame(self.to_table(query))


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a quaternion ``(w, x, y, z)``, normalized first.

    Normalizing here is not defensive tidying. A quaternion normalization defect
    in ROSHAMBO2 made the applied transform non-orthogonal -- ``|q|`` reached
    1.216 and ``det(M)`` 1.313 -- so 1.2% of returned poses reported a Tanimoto
    no rigid motion can achieve. The fix belongs in the kernel, and is there;
    this is the reader's half of it.
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n == 0:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _apply(m: np.ndarray, r: np.ndarray) -> np.ndarray:
    return r @ m.T

"""What happens to scores.

``Sink`` is the engine's second extension point, and the boundary is narrow: a
device pointer to the ``[n_rows, 20]`` score block the kernel just wrote, plus
the maps from output row to (query, library) and the block's ``gid_base``. That is
the same contract ``topk_add_block`` already takes, which is why adding a sink is
cheap.

Why the top-K is downstream of the kernel rather than fused into it
------------------------------------------------------------------
It reads a block the scoring kernel has already written, in a separate launch.
Fusing the two would save writing and re-reading that block -- at a
5e8-conformer corpus about 328 GB each way, roughly 0.4 s against a 135 s wall,
on a device whose DRAM sits at 1.2-2.5% of peak. Under 1%, in exchange for
coupling the reducer to every scoring kernel that will ever exist.

The measured 1.6x the device top-K bought came from **not crossing PCIe** -- 64 GB
of copy-back eliminated on a 1e8-conformer tier, where the host reduction that
consumed it was 36.8% of the critical path. It did not come from avoiding a
device round trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from ..numerics import rank_order
from ..scores import N_SCORE_COLUMNS


class Sink(Protocol):
    """Consumes each launch's scores and produces the screen's result.

    ``max_block_conformers`` is the largest block the source will yield. It is
    passed to :meth:`prepare` rather than discovered from the first block
    because a device-side state has to be sized for the worst case up front --
    blocks are not all the same size, the last of every group being short --
    and a state sized from whichever block happened to arrive first refuses the
    first larger one partway through a screen.

    A sink that needs something staged on the device *before* the launch (the
    molecule-level top-K needs the block's ``mol_id``) may also implement
    ``before_block(block)``; the engine calls it when it is present.
    """

    def prepare(self, n_queries: int, device_id: int,
                max_block_conformers: int = 0) -> None: ...
    def add_block(self, block, scores: Optional[np.ndarray]) -> None: ...
    def finalize(self) -> Any: ...

    @property
    def wants_dense(self) -> bool:
        """True if the engine must copy the score block back to the host."""
        ...


class TopK:
    """Device-resident running top-K. The dense block never crosses PCIe.

    The output is a pure function of the **set** of conformers screened -- not of
    block size, grid size, launch configuration, batch split or GPU count. See
    ``backend/_cuda_src/topk.cuh`` for why: the admitted set is order-independent
    even though `atomicAdd` slot assignment is not, selection is a function of the
    set because ids are unique and both sort passes are stable, and any threshold
    at or below the true k-th score gives the same final answer.

    One state per device. To screen on several GPUs, run one engine and one
    ``TopK`` each and merge with :func:`roshambo3.numerics.rank_order` --
    :meth:`merge` does it.
    """

    wants_dense = False

    device_level = "conformer"
    """What one retained row is. ``score_pairs`` routes on this: a
    conformer-level state takes the dense block directly, a molecule-level one
    reduces it first."""

    def __init__(self, k: int, *, rank_col: int = 0, keep_columns: bool = True):
        if int(k) <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not 0 <= int(rank_col) < N_SCORE_COLUMNS:
            raise ValueError(
                f"rank_col {rank_col} is not a score column; "
                f"see roshambo3.scores for the {N_SCORE_COLUMNS} names")
        if not keep_columns:
            # The device state supports dropping the columns and keeping only
            # the ranking score, which would save cap*dv floats. The readback
            # does not: topk_result writes n floats at stride k on that path
            # while the pybind binding copies n*dv out of a [nq, k, dv]
            # buffer, so the caller receives mostly uninitialized memory. And a
            # correct version would return [n, 1] rows, which every consumer of
            # a score row indexes by column name. Refused rather than fixed
            # halfway.
            raise NotImplementedError(
                "keep_columns=False returns uninitialized memory: the device "
                "writes one float per row and the binding reads a full score "
                "row. Use keep_columns=True.")
        self.k = int(k)
        self.rank_col = int(rank_col)
        self.keep_columns = bool(keep_columns)
        self._state = None
        self._device_id = 0
        self.rows_offered = 0

    def prepare(self, n_queries: int, device_id: int,
                max_block_conformers: int = 0) -> None:
        from ..backend.cuda import extension

        self._device_id = int(device_id)
        self._state = extension().DeviceTopK(
            n_query=int(n_queries), k=self.k, rank_col=self.rank_col,
            dv=N_SCORE_COLUMNS, keep_cols=self.keep_columns,
            device_id=self._device_id)

    @property
    def handle(self):
        if self._state is None:
            raise RuntimeError("TopK.prepare() has not been called")
        return self._state.handle

    def add_block(self, block, scores: Optional[np.ndarray]) -> None:
        # Nothing to do: the kernel already folded this block in on the device,
        # because the engine passed `handle` to the launch. Note the name: the
        # sink protocol's method is `add_block`, not `handle`, precisely so it
        # cannot collide with the device pointer property above -- which it did,
        # silently, on the first attempt.
        self.rows_offered += block.n_conformers

    def counters(self) -> dict:
        return {} if self._state is None else dict(self._state.counters())

    def thresholds(self) -> np.ndarray:
        return self._state.thresholds()

    def finalize(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """``[(scores[n, 20], gids[n])]`` per query, in canonical order."""
        if self._state is None:
            return []
        return list(self._state.result(self._device_id))

    @staticmethod
    def merge(parts: list[list[tuple[np.ndarray, np.ndarray]]], k: int,
              *, rank_col: int = 0):
        """Merge per-device results into one, under the canonical tie-break.

        This is what makes multi-GPU a host-side concern. Top-N lists were
        measured to merge correctly even across heterogeneous hardware.

        ``rank_col`` **must match the column the parts were ranked on**. It
        cannot be inferred from the arrays -- every part carries all the score
        columns, so a list ranked on column 7 looks exactly like one ranked on
        column 0. Merging on the wrong column silently returns a differently
        ordered answer, not an error, which is why :meth:`merge_sinks` exists
        and is the safer call.
        """
        if not parts:
            return []
        n_q = len(parts[0])
        for i, p in enumerate(parts):
            if len(p) != n_q:
                raise ValueError(
                    f"every part must cover the same queries: part 0 has "
                    f"{n_q}, part {i} has {len(p)}. A short part used to raise "
                    f"IndexError and a long one was silently truncated.")
        out = []
        for q in range(n_q):
            s = np.concatenate([p[q][0] for p in parts], axis=0)
            g = np.concatenate([p[q][1] for p in parts], axis=0)
            if s.ndim != 2 or not 0 <= rank_col < s.shape[1]:
                raise ValueError(
                    f"rank_col {rank_col} is out of range for score rows of "
                    f"shape {s.shape}")
            order = rank_order(s[:, rank_col], g)[:k]
            out.append((s[order], g[order]))
        return out

    @staticmethod
    def merge_sinks(sinks: "Sequence[TopK]", k: Optional[int] = None):
        """Merge finalized :class:`TopK` sinks, taking ``rank_col`` from them.

        Prefer this to :meth:`merge`: the sinks know which column they ranked
        on, so the one parameter that cannot be recovered from the result
        arrays does not have to be restated by the caller.
        """
        sinks = list(sinks)
        if not sinks:
            return []
        levels = {getattr(s, "device_level", "conformer") for s in sinks}
        if len(levels) != 1:
            raise ValueError(
                f"these sinks retained different things ({sorted(levels)}): one "
                f"holds the best conformers and another the best molecules, so "
                f"the merged list is neither. Merge each level separately.")
        cols = {s.rank_col for s in sinks}
        if len(cols) != 1:
            raise ValueError(
                f"these sinks ranked on different columns ({sorted(cols)}), so "
                f"their results are not a single ranking and cannot be merged.")
        ks = {s.k for s in sinks}
        if k is None:
            if len(ks) != 1:
                raise ValueError(
                    f"these sinks kept different numbers of candidates "
                    f"({sorted(ks)}), so there is no k that is safe to infer. "
                    f"The largest asks for a global top-k that the smaller "
                    f"partitions never retained enough rows to supply, and the "
                    f"result would be silently short and not the true top-k -- "
                    f"`rank_order(...)[:k]` simply yields fewer rows. Pass k "
                    f"explicitly, at most {min(ks)}.")
            k = ks.pop()
        elif k > min(ks):
            raise ValueError(
                f"k={k} exceeds the {min(ks)} candidates the smallest sink "
                f"retained, so the merged list cannot be the true global "
                f"top-{k}: a molecule ranked {min(ks) + 1}st within its own "
                f"partition was already discarded there.")
        return TopK.merge([s.finalize() for s in sinks], k,
                          rank_col=cols.pop())



class HostTopK:
    """A running top-K on the host, satisfying the same contract as :class:`TopK`.

    Exists because :class:`TopK` and :class:`MoleculeTopK` reach into the
    CUDA extension for ``DeviceTopK``, so a backend that is not CUDA has no
    reducer at all -- and ``DenseScores`` is not the answer, because it keeps
    every row and carries a size guard for exactly that reason.

    **The contract is the determinism argument, not the API.** The output must
    be a pure function of the *set* of conformers screened -- not of block size,
    launch configuration, batch split or thread count. ``topk.cuh:33-58`` states
    the three facts that give the device that property, and it was written so a
    host reducer could reproduce it. Reproduced here:

    1. **The admitted set is order-independent.** A row is admitted iff its
       ranking score is at or above the running threshold, and the threshold
       only ever rises. A row rejected against a higher threshold would also
       have been cut by the final selection, so admitting on a stale lower
       threshold cannot change the answer -- only how much is buffered.
    2. **Selection is a function of the set**, because ids are unique and the
       sort is total: :func:`roshambo3.numerics.rank_order` is descending score
       with ties by *ascending id*, via ``np.lexsort``, which is stable.
       ``np.argsort``'s default is not, and would break this.
    3. **Any threshold at or below the true k-th score yields the same final
       answer**, which is what leaves the periodic reselection below free to
       happen whenever it likes.

    The tie-break id is the **global conformer id**, matching what the device
    reduces on (``topk.cuh``'s "ascending by conformer id"). That is *not*
    ``metrics.canonical_ids(mol_id, conf_idx)``, which is the tie-break for
    comparing two ranked lists; the two agree on which rows are retained
    because gid is unique, and a caller comparing this against a gold list must
    still re-rank both under the canonical key.
    """

    wants_dense = True
    """The engine must hand this sink the score block. On a CPU backend that is
    not a copy -- the scores were computed into host memory."""

    device_level = "conformer"

    #: Reselect once a query's buffer exceeds this multiple of ``k``. Mirrors
    #: the device's periodic radix reselection; fact 3 is what makes the choice
    #: of multiple a memory/time trade rather than a correctness one.
    RESELECT_AT = 4

    def __init__(self, k: int, *, rank_col: int = 0):
        if int(k) <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not 0 <= int(rank_col) < N_SCORE_COLUMNS:
            raise ValueError(
                f"rank_col {rank_col} is not a score column; "
                f"see roshambo3.scores for the {N_SCORE_COLUMNS} names")
        self.k = int(k)
        self.rank_col = int(rank_col)
        self.rows_offered = 0
        self._n_q = 0
        self._scores: list = []
        self._gids: list = []
        self._threshold: list = []

    def prepare(self, n_queries: int, device_id: int,
                max_block_conformers: int = 0) -> None:
        self._n_q = int(n_queries)
        self._scores = [[] for _ in range(self._n_q)]
        self._gids = [[] for _ in range(self._n_q)]
        self._threshold = [-np.inf] * self._n_q
        self.rows_offered = 0

    @property
    def handle(self):
        """No device state to hand the backend. ``score_pairs`` routes on this
        being None, which is how a dense-wanting sink is told apart from one
        that owns a device reducer."""
        return None

    def add_block(self, block, scores) -> None:
        if scores is None:
            raise ValueError(
                "HostTopK needs the dense score block and the engine did not "
                "provide one. It advertises wants_dense=True, so this means "
                "the engine routed it as a device sink.")
        n = int(block.n_conformers)
        self.rows_offered += int(scores.shape[0])
        # Row r is (query r // n, library r % n) -- the dense Cartesian
        # enumeration the engine builds. Reshaping is its inverse and costs
        # nothing; it must not be re-derived from a pair index, which a sink is
        # not given.
        rows = np.asarray(scores).reshape(self._n_q, n, -1)
        gids = int(block.gid_base) + np.arange(n, dtype=np.int64)
        for q in range(self._n_q):
            keep = rows[q][:, self.rank_col] >= self._threshold[q]
            if not keep.any():
                continue
            self._scores[q].append(rows[q][keep])
            self._gids[q].append(gids[keep])
            if sum(len(a) for a in self._gids[q]) > self.RESELECT_AT * self.k:
                self._reselect(q)

    def _reselect(self, q: int) -> None:
        s = np.concatenate(self._scores[q], axis=0)
        g = np.concatenate(self._gids[q], axis=0)
        order = rank_order(s[:, self.rank_col], g)[:self.k]
        s, g = s[order], g[order]
        self._scores[q], self._gids[q] = [s], [g]
        if len(g) >= self.k:
            # Fact 3: raising the threshold to the k-th retained score cannot
            # change the answer, and it is what bounds the buffer.
            self._threshold[q] = float(s[-1, self.rank_col])

    def counters(self) -> dict:
        return {"rows_offered": self.rows_offered,
                "held": [sum(len(a) for a in g) for g in self._gids]}

    def thresholds(self) -> list:
        return list(self._threshold)

    def finalize(self) -> "list[tuple[np.ndarray, np.ndarray]]":
        out = []
        for q in range(self._n_q):
            if not self._gids[q]:
                out.append((np.zeros((0, N_SCORE_COLUMNS), np.float32),
                            np.zeros(0, np.int64)))
                continue
            s = np.concatenate(self._scores[q], axis=0)
            g = np.concatenate(self._gids[q], axis=0)
            order = rank_order(s[:, self.rank_col], g)[:self.k]
            out.append((np.ascontiguousarray(s[order], dtype=np.float32),
                        np.ascontiguousarray(g[order], dtype=np.int64)))
        return out


class MoleculeTopK:
    """Device-resident running top-K over **molecules**, not conformers.

        sink = MoleculeTopK(k=100)
        stats, per_query = engine.run(source, queries, config, sink)

    A molecule scores as the best of its conformers, and the returned gid names
    the conformer that achieved it -- so a pose is still recoverable, and the
    score row is that conformer's whole row, not a synthetic one.

    **Why this is not a display fix.** The top 100 conformers and the top 100
    molecules are different *sets*: a molecule that would rank 40th on its best
    conformer can be absent from a conformer-level top 100 that another
    molecule's near-duplicates filled. Enrichment is a compound-level
    statistic, so it cannot be computed from the conformer-level list at all.
    The conformer/molecule distinction is kept in the result API rather than in
    an analysis script.

    **What it costs.** Nothing on the bus and one small kernel on the device:
    the reduction runs on the dense score block where it already lives, before
    the ordinary device top-K sees it, and the only new traffic is the block's
    ``mol_id`` sidecar going *up* -- 8 B per conformer, which nothing else in
    the pipeline sends. Overfetching ``k * max_conformers`` conformers and
    deduplicating on the host is the fallback, not the design: it multiplies the
    top-K state and its readback by up to 256, and it is exact only while that
    bound holds.

    **What it assumes, and where that is checked.** All of a molecule's
    conformers are contiguous in global id order. ``prep/build.py`` asserts it
    while writing each group, ``roshambo3 verify`` checks it corpus-wide, and
    this sink additionally checks the device's reading of each block's ``mol_id``
    against the sidecar's. A silent violation would count a molecule twice,
    which looks exactly like a legitimate result.

    See ``backend/_cuda_src/molecule_topk.cuh`` for the segmented argmax, the
    boundary carry and the determinism argument.
    """

    wants_dense = False

    device_level = "molecule"

    def __init__(self, k: int, *, rank_col: int = 0):
        if int(k) <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not 0 <= int(rank_col) < N_SCORE_COLUMNS:
            raise ValueError(
                f"rank_col {rank_col} is not a score column; "
                f"see roshambo3.scores for the {N_SCORE_COLUMNS} names")
        self.k = int(k)
        self.rank_col = int(rank_col)
        self.keep_columns = True
        self._state = None
        self._device_id = 0
        self.rows_offered = 0
        self.blocks_seen = 0
        #: Global id of the first conformer of the molecule left open by the
        #: previous block, or None. It is what ``base_gid`` has to be: the
        #: winning conformer of a carried molecule precedes the current block,
        #: so a base taken from the current block's ``gid_base`` could not
        #: express it as a non-negative offset.
        self._open_molecule_gid: Optional[int] = None

    def prepare(self, n_queries: int, device_id: int,
                max_block_conformers: int = 0) -> None:
        from ..backend.cuda import extension

        if int(max_block_conformers) <= 0:
            raise ValueError(
                "MoleculeTopK needs the source's largest block size, which the "
                "engine passes to prepare(). A state sized from the first block "
                "refuses the first larger one partway through a screen.")
        self._device_id = int(device_id)
        self._open_molecule_gid = None
        self._state = extension().DeviceMoleculeTopK(
            n_query=int(n_queries), k=self.k, rank_col=self.rank_col,
            dv=N_SCORE_COLUMNS, keep_cols=True, device_id=self._device_id,
            max_block_conformers=int(max_block_conformers))

    @property
    def handle(self):
        if self._state is None:
            raise RuntimeError("MoleculeTopK.prepare() has not been called")
        return self._state.handle

    def before_block(self, block) -> None:
        """Stage this block's ``mol_id`` and the boundary bookkeeping.

        Called by the engine before the launch, because the reduction runs
        inside it. Everything here is host arithmetic over one mmapped sidecar
        slice plus an 8 B/conformer upload.
        """
        if self._state is None:
            raise RuntimeError("MoleculeTopK.prepare() has not been called")
        mol = getattr(block, "mol_id", None)
        if mol is None:
            raise RuntimeError(
                "this block carries no mol_id, so molecules cannot be "
                "identified. A source feeding MoleculeTopK must set "
                "ConformerBlock.mol_id and tail_molecule_open from the "
                "corpus's mol_id sidecar.")
        mol = np.ascontiguousarray(mol, dtype=np.int64)
        nb = int(block.n_conformers)
        if mol.shape[0] != nb:
            raise ValueError(
                f"this block has {nb} conformers but {mol.shape[0]} mol_ids")
        tail_open = bool(getattr(block, "tail_molecule_open", False))

        base = (int(block.gid_base) if self._open_molecule_gid is None
                else int(self._open_molecule_gid))
        self._state.stage_block(mol_id=mol, gid_base=int(block.gid_base),
                                tail_open=tail_open, base_gid=base,
                                device_id=self._device_id)

        # The device read the same array to find the segments. Check its answer
        # against the sidecar rather than trusting that two readings of one
        # array agree -- mol_ids are unique per molecule, so `mol == mol[-1]`
        # selects exactly the trailing molecule's rows.
        tail_start = nb - int(np.count_nonzero(mol == mol[-1]))
        n_seg = 1 + int(np.count_nonzero(mol[1:] != mol[:-1]))
        seen = self._state.last_block()
        if (seen["n_segments"] != n_seg
                or seen["tail_segment_start"] != tail_start):
            raise RuntimeError(
                f"the device segmented this block into {seen['n_segments']} "
                f"molecules with the last starting at row "
                f"{seen['tail_segment_start']}, and the sidecar says {n_seg} "
                f"and {tail_start}. The two readings of one mol_id array "
                f"disagree, so a molecule's conformers are not the contiguous "
                f"run this reduction is built on -- run `roshambo3 verify`.")

        if tail_open:
            if not (tail_start == 0 and self._open_molecule_gid is not None):
                self._open_molecule_gid = int(block.gid_base) + tail_start
        else:
            self._open_molecule_gid = None

    def add_block(self, block, scores: Optional[np.ndarray]) -> None:
        # The reduction and the top-K both ran inside the launch, because the
        # engine passed `handle`. Same shape as TopK.add_block.
        self.rows_offered += block.n_conformers
        self.blocks_seen += 1

    def counters(self) -> dict:
        return {} if self._state is None else dict(self._state.counters())

    def thresholds(self) -> np.ndarray:
        return self._state.thresholds()

    def finalize(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """``[(scores[n, 20], winning_gids[n])]`` per query, canonical order.

        Raises if the screen ended in the middle of a molecule: that molecule is
        absent from this answer and the partition that continues it would emit
        it as well. A group boundary is always a molecule boundary, which is why
        splitting by group is safe and splitting inside one is not.
        """
        if self._state is None:
            return []
        return list(self._state.result(self._device_id))

    @staticmethod
    def merge(parts: list[list[tuple[np.ndarray, np.ndarray]]], k: int,
              *, rank_col: int = 0):
        """Merge per-device molecule-level results under the canonical tie-break.

        Identical to :meth:`TopK.merge` -- the rows are already one per molecule
        and their gids are unique -- and correct only because **each partition
        covers whole molecules**. ``R3gSource(groups=...)`` splits by group and
        a molecule never straddles a group, so a molecule appears in exactly one
        partition. Each state also refuses to finalize with a molecule still
        open, which is what a split through the middle of one looks like.
        """
        return TopK.merge(parts, k, rank_col=rank_col)

    @staticmethod
    def merge_sinks(sinks, k: Optional[int] = None):
        return TopK.merge_sinks(sinks, k)


@dataclass
class DenseResult:
    """The full score matrix, with the mapping to conformers made explicit.

    ``scores`` is ``[n_queries, n_conformers, 20]`` and ``gids`` is
    ``[n_conformers]``, so ``scores[q, i]`` is the score of query ``q`` against
    the conformer whose global id is ``gids[i]``. Both are needed: a screen may
    cover a subset of a corpus, in which case ``gids`` is not ``arange``.
    """

    scores: np.ndarray
    gids: np.ndarray

    def __repr__(self) -> str:
        return (f"<DenseResult {self.scores.shape[0]} queries x "
                f"{self.scores.shape[1]} conformers>")


class DenseScores:
    """Return the raw score matrix, with no reduction of any kind.

    The all-pairs mode: every pair's full score row, for a corpus small enough
    that the matrix itself is the deliverable.

    It necessarily re-introduces the device-to-host copy the top-K exists to
    avoid, so it carries a size guard. A score row is 80 B, so a 1e6-conformer
    corpus against 8 queries is 6.4 GB -- which is why the limit is a limit and
    not a warning. Raise ``max_bytes`` deliberately, having decided the copy is
    what you want.

    On the output layout, because getting it wrong is easy and silent: each
    launch writes rows query-major within its own block, so naively
    concatenating blocks gives block-major-then-query-major, which is neither
    corpus order nor query order. :class:`DenseResult` therefore reassembles
    into an explicit ``[n_queries, n_conformers, 20]`` rather than handing back
    a flat array whose ordering has to be reasoned about.
    """

    wants_dense = True

    def __init__(self, max_bytes: int = 4 << 30):
        self.max_bytes = int(max_bytes)
        self._chunks: list[tuple[int, int, np.ndarray]] = []
        self._bytes = 0
        self._n_queries = 0

    def prepare(self, n_queries: int, device_id: int,
                max_block_conformers: int = 0) -> None:
        self._chunks.clear()
        self._bytes = 0
        self._n_queries = int(n_queries)

    def add_block(self, block, scores: Optional[np.ndarray]) -> None:
        if scores is None:
            raise RuntimeError("DenseScores needs the dense block; "
                               "the engine did not copy it back")
        self._bytes += scores.nbytes
        if self._bytes > self.max_bytes:
            raise MemoryError(
                f"DenseScores would hold {self._bytes / 2**30:.3f} GiB, over the "
                f"{self.max_bytes / 2**30:.3f} GiB limit. A score row is "
                f"{N_SCORE_COLUMNS * 4} B, so an all-pairs matrix grows as "
                f"n_queries * n_conformers -- use TopK for a screen, or raise "
                f"max_bytes if the matrix really is what you want.")
        self._chunks.append((block.gid_base, block.n_conformers, scores.copy()))

    def finalize(self) -> DenseResult:
        nq = self._n_queries
        if not self._chunks:
            return DenseResult(np.empty((nq, 0, N_SCORE_COLUMNS), np.float32),
                               np.empty(0, np.int64))
        self._chunks.sort(key=lambda c: c[0])
        gids = np.concatenate([np.arange(g, g + n, dtype=np.int64)
                               for g, n, _ in self._chunks])
        total = int(sum(n for _, n, _ in self._chunks))
        out = np.empty((nq, total, N_SCORE_COLUMNS), np.float32)
        at = 0
        for _, n, sc in self._chunks:
            # Each launch wrote query-major within its own block.
            out[:, at:at + n, :] = sc.reshape(nq, n, N_SCORE_COLUMNS)
            at += n
        return DenseResult(out, gids)


class Callback:
    """Hand each launch's scores to a function as they complete.

    For a future server, and for a caller who wants to stream hits out rather
    than accumulate them. ``fn(block, scores)`` is called on the engine's thread,
    so it must be quick or hand off.
    """

    wants_dense = True

    def __init__(self, fn: Callable[[Any, np.ndarray], None]):
        self.fn = fn
        self.n_blocks = 0

    def prepare(self, n_queries: int, device_id: int,
                max_block_conformers: int = 0) -> None:
        self.n_blocks = 0

    def add_block(self, block, scores: Optional[np.ndarray]) -> None:
        self.n_blocks += 1
        self.fn(block, scores)

    def finalize(self) -> int:
        return self.n_blocks

"""Where conformers come from.

``Source`` is the engine extension point for deterministic conformer blocks.
The format-3 implementation reads a prepared corpus; future sources may generate
conformers on-device or consume a queue fed by a long-lived server.

**Block order is load-bearing.** A global conformer id is a *position* in corpus
order, so a screen that visited blocks in a different order would assign
different ids and, on a tie, select a different conformer. Every source must
therefore yield in a deterministic order, and the prefetcher below preserves it.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Protocol, Sequence

import numpy as np



@dataclass
class ConformerBlock:
    """One launch's worth of library molecules, staged and kernel-ready.

    **Deliberately "no checks here".** Every
    field below is consumed by exactly one function, ``backend.score_pairs``,
    and every invariant that matters is checked there against the arguments it
    is actually combined with: ``counts``' row count against ``coords``', the
    real-atom counts against the padded widths, and an explicit (sparse) pair
    list against both. A dense launch -- the engine's -- has no pair list; its
    width is ``counts``' row count, and the engine's launch loop refuses a block
    whose ``n_conformers`` disagrees with it. A ``__post_init__`` here could only re-state a subset of that, from a
    position that cannot see the query it will be paired with -- and a second,
    weaker copy of a rule is drift, not a second line
    of defense. What a block *cannot* self-check is the interesting part:
    whether ``n_conformers`` is right is a claim about the producer, and the
    boundary catches its consequence.

    ``coords`` is ``[n_rows, L, 3]`` **int16** in the transposed tile layout --
    the wire record, the corpus's own shorts at the wire-step, 6 B/atom instead of
    the 16 a dequantized four-component record would cost -- and ``n_rows`` may
    exceed ``n_conformers`` because the final tile is zero-padded.
    Size the library count from ``n_conformers``, never from ``coords.shape[0]``:
    the padding lanes hold no molecule and no output row may reference them.
    """

    coords: np.ndarray
    counts: np.ndarray
    """``[n_conformers, n_features]`` uint8, the per-conformer count vector:
    column 0 the real atom count, columns 1.. the per-type feature counts.

    This is what the kernel reads. It is one row per conformer and is **not**
    tiled, unlike ``coords`` -- a warp's 32 lanes hold 32 consecutive rows, so
    8 B per conformer is already a coalesced load."""
    n_conformers: int
    L: int
    bucket: int
    gid_base: int
    """Global id of this block's first conformer."""
    L_shape: int = 0
    """The width the per-bucket QUERY-FIELD policies resolve from.

    It exists because ``L`` is two different things to two different readers.
    To anything sizing the record the kernel reads -- the staged buffer, the
    pinned pool -- ``L`` is the padded width and is correct. To the per-bucket
    query-field policy it was a proxy for HOW BIG THE MOLECULES ARE, and the
    padded width has now stopped being that proxy twice. First a color plane
    joined it: the same molecules resolved h = 0.5 A shape-built at
    L = 32 and h = 0.6 A color-built at L = 72, so a color screen drew a
    coarser lattice than a shape screen of the same molecules. Then the padded
    width became per GROUP rather than per bucket, which would move the
    lattice again -- and a lattice change is a numerics change, not a
    data-movement one.

    So this is the bucket's ``pad_L(max n_real)``, recorded in the manifest as
    ``L_field_width`` and read back by ``Group.L_field_width``: exactly what
    the physical atom width used before padding. 0 means
    "same as ``L``"."""
    buffer: Optional[np.ndarray] = None
    """The pinned buffer backing ``coords``, to be released after the launch."""
    mol_id: Optional[np.ndarray] = None
    """``[n_conformers]`` int64 ``mol_id`` sidecar slice, or None.

    A molecule-level reduction needs to know which conformers belong together,
    and this is the only place that says so. It is 8 B per conformer against
    the coordinates' hundreds, and nothing else in the pipeline sends it."""
    tile_mode: int = 1
    """Which tile ``coords`` is in: 1 the interleaved
    ``[tile][atom][lane][component]``, -1 the planar
    ``[tile][atom][component][lane]``.

    It travels with the block rather than with the ScreenConfig because the
    party that packed the buffer is the only one that knows how it is laid out,
    and nothing about the array says so -- reading one layout at the other's
    stride returns another conformer's coordinates and does not raise. One
    definition, carried, rather than two that have to agree."""
    color_offset: int = 0
    """Where this block's color runs begin, in the kernel's own convention.

    ``0`` -- a shape-only block, where no color run is present. ``-1`` -- a color
    block, where the color run starts at each conformer's own real atom count
    (``library_color_offset < 0`` is also how ``engine/query.py`` lays a query
    out).

    It travels with the block, beside :attr:`tile_mode`, for the same reason:
    the party that packed the buffer is the only one that knows, and nothing
    about the array says so. A wrong value here reads one molecule's padding as
    another's features and returns a plausible score rather than raising."""
    tail_molecule_open: bool = False
    """Whether this block's last molecule continues into the next block.

    Only a source can know: it needs the sidecar entry one past the end of the
    block. It is False at the end of a group, because a molecule never straddles
    one -- the builder extends a group to a molecule boundary before closing
    it."""
    @property
    def n_real(self) -> np.ndarray:
        """``[n_conformers]`` real atom counts -- ``counts[:, 0]``.

        A view, not a second array: there is one record; this keeps the name the ~15 external consumers use."""
        return self.counts[:, 0]


class Source(Protocol):
    """Yields :class:`ConformerBlock` in a deterministic order."""

    def blocks(self) -> Iterator[ConformerBlock]: ...

    @property
    def n_conformers(self) -> int: ...


class Prefetcher:
    """Run ``fn`` over ``jobs`` on a thread pool, ``depth`` ahead, in order.

    ``wait_s`` accumulates the consumer's blocked time, which is the headline
    number: it is how much of the wall the GPU spent waiting for the host.

    **``depth = 0`` is the only true serial baseline.** It runs ``fn`` inline on
    the consumer's thread. ``depth = 1`` is *not* that -- one job is already
    submitted before the consumer is handed the previous one, so a single-deep
    queue hides the whole decode stage. An early ablation ladder used depth 1 as
    its "serial" rung and measured it 1.55x faster than the real serial loop for
    exactly that reason.
    """

    def __init__(self, jobs: Sequence, fn: Callable, n_workers: int, depth: int,
                 on_discard: Optional[Callable] = None):
        self.jobs = list(jobs)
        self.fn = fn
        self.n_workers = max(1, int(n_workers))
        self.depth = max(0, int(depth))
        self.wait_s = 0.0
        self.on_discard = on_discard
        """Called with each decoded item that is never yielded, when the
        consumer stops early. Without it a screen that raises leaves every
        in-flight block's pinned buffer checked out of a pool that outlives the
        run."""

    def __iter__(self):
        if self.depth == 0:
            for job in self.jobs:
                t0 = time.perf_counter()
                item = self.fn(job)
                self.wait_s += time.perf_counter() - t0
                yield item
            return
        yield from self._pipelined()

    def _pipelined(self):
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            pending = []
            it = iter(self.jobs)
            try:
                for _ in range(self.depth):
                    try:
                        pending.append(ex.submit(self.fn, next(it)))
                    except StopIteration:
                        break
                while pending:
                    fut = pending.pop(0)
                    t0 = time.perf_counter()
                    item = fut.result()
                    self.wait_s += time.perf_counter() - t0
                    # `item` is off `pending` and not yet yielded, so `_drain`
                    # cannot see it and this frame is its only owner. A submit
                    # can fail here -- a shut-down executor, a thread that
                    # cannot be created -- and strand it.
                    #
                    # The `yield` is deliberately *not* covered: from there the
                    # consumer owns the buffer and releases it in its own
                    # `finally`, so releasing here too is a double release. That
                    # is not a theoretical distinction -- an earlier draft of
                    # this fix wrapped the yield as well, and the pool's
                    # ownership check caught it handing one buffer to two
                    # blocks.
                    try:
                        try:
                            pending.append(ex.submit(self.fn, next(it)))
                        except StopIteration:
                            pass
                    except BaseException:
                        if self.on_discard is not None:
                            self.on_discard(item)
                        raise
                    yield item
            finally:
                # Reached when the consumer stops early -- an exception in the
                # screen loop, or a caller who breaks out of it. Cancel what has
                # not started, then hand back what did: a decode that finished
                # but was never yielded still holds a pinned buffer, and the
                # pool outlives this run.
                self._drain(pending)

    def _drain(self, pending: list) -> None:
        """Hand back what finished but was never yielded.

        Note what this cannot do: a job that *raised* has no item, so its
        buffer -- acquired before the failure -- is not represented by anything
        here. Releasing it is the decode's own responsibility, and `_decode`
        does it.
        """
        for fut in pending:
            if fut.cancel():
                continue
            try:
                item = fut.result()
            except Exception:
                continue
            if self.on_discard is not None:
                try:
                    self.on_discard(item)
                except Exception:
                    # One failed release must not strand every buffer behind
                    # it. This runs while an exception is already propagating,
                    # so there is nothing useful to raise it at.
                    continue



def _check_groups(groups: Sequence[int], n_groups: int) -> list[int]:
    """Validate a group selection. Neither failure below is loud on its own.

    A **duplicate** index enqueues that group's blocks twice with the same
    ``gid_base``, which puts the same global conformer id into the top-K twice.
    The determinism proof in ``backend/_cuda_src/topk.cuh`` rests on ids being
    unique -- with duplicates the id sort is no longer a total order, and one
    conformer can occupy two slots in the answer.

    A **negative** index is worse, because Python resolves it: ``groups=[-1]``
    reads the last group's blocks but takes its base from ``offsets[-1]``,
    which is the corpus grand total rather than that group's offset. Every id
    it emits then points past the end of the corpus, and nothing downstream
    notices -- the re-gather simply resolves the wrong molecule.
    """
    out = []
    for g in groups:
        gi = int(g)
        if gi < 0 or gi >= n_groups:
            raise ValueError(
                f"group index {gi} is out of range for a corpus with "
                f"{n_groups} groups. Negative indices are refused rather than "
                f"wrapped: a wrapped index reads the right blocks under the "
                f"wrong global ids.")
        out.append(gi)
    if len(set(out)) != len(out):
        dupes = sorted({g for g in out if out.count(g) > 1})
        raise ValueError(
            f"group indices must be unique; {dupes} repeated. A repeated group "
            f"is screened twice under the same global conformer ids, which the "
            f"device top-K's determinism argument does not survive.")
    return out


DEFAULT_PREFETCH_DEPTH = 3
"""How many decoded runs a source keeps in flight ahead of the consumer.

**This is also the cap on how many of ``workers`` threads can ever run.**
``Prefetcher._pipelined`` keeps at most ``depth`` futures outstanding, so the
historical ``workers=8, prefetch_depth=3`` pairing gave five threads no work at
all. Raising the depth raises the pool with it -- ``Engine._ensure_pool`` takes
``max(pool_buffers, prefetch_depth + 1)`` -- and at the coalesced launch size a
buffer is tens of MiB, so the depth is a memory knob as much as a concurrency
one."""

DEFAULT_CONFORMERS_PER_LAUNCH = 65_536
"""Conformers a source aims to put in one ``score_pairs`` launch.

Without it a launch is one on-disk block -- ``format.BLOCK_CONF`` = 4,096
conformers. Measured on a 10M-conformer corpus against 8 queries with a top-K
sink, in one binary and one clock session, with the query-field cache.
Speedup over 4,096:

===========  =========  =========  =========  =========
launch size  g0 (L=16)  g2 (L=24)  g5 (L=32)  g7 (L=40)
===========  =========  =========  =========  =========
16,384       2.29x      1.87x      1.79x      1.77x
32,768       2.83x      2.15x      2.05x      2.22x
65,536       3.22x      2.30x      2.19x      2.42x
full         3.63x      2.52x      2.37x      2.57x
===========  =========  =========  =========  =========

**The knee is at 32,768-65,536, and 65,536 captures ~95% of full aggregation**
-- which is why the default is a bounded coalesce and not "the whole group".
The remaining 5% costs pinned memory and device top-K capacity linearly, and
both are paid whether or not a given corpus is large enough to use them.

Two mechanisms, not one, and only the second is a host-side cost: at 8 queries
a 4,096-conformer launch is 32,768 pairs, 244x under ``bench/rules.py``'s
``MIN_CONFORMERS_PER_LAUNCH``, so the *kernel itself* runs 1.7x longer split 60
ways. No amount of host prefetch recovers that.

``conformers_per_launch=BLOCK_CONF`` (4,096) reproduces one-block-per-launch behavior
exactly, block for block, and is what the identity tests pin against."""


def _bounded_launch_size(target: int, n_queries: int, k: int, *,
                         budget_bytes: int, dv: int) -> int:
    """Largest launch size at or below ``target`` whose top-K state fits.

    The device top-K sizes its candidate buffer as ``cap = k + 2 *
    max_rows_per_query`` and, with ``keep_cols``, allocates ``n_queries * cap *
    dv`` floats -- so the cost of coalescing is **linear in the query count as
    well as in the launch size**, and the launch size alone does not bound it.
    At 8 queries the state is 5.8 MB at 4,096 conformers and 84.5 MB at 65,536;
    at a 480-conformer query panel the same 65,536 would ask for 5.1 GB.

    Failing late is the thing to avoid: the capacity is fixed on the **first**
    launch and a later block that does not fit is a hard error, so a screen that
    started fine dies partway through. Hence a bound applied before any device
    memory is touched.

    Never returns less than one on-disk block: below that, coalescing is not
    what is wrong with the configuration.
    """
    from ..format import BLOCK_CONF

    n_queries = max(1, int(n_queries))
    rows = (budget_bytes / (4.0 * dv * n_queries) - int(k)) / 2.0
    return int(max(BLOCK_CONF, min(int(target), int(rows))))


def conformers_per_launch(n_queries: int, k: int, *,
                          target: int = DEFAULT_CONFORMERS_PER_LAUNCH,
                          budget_bytes: int = 256 << 20,
                          dv: Optional[int] = None) -> int:
    """The launch size to hand :class:`R3gSource`, bounded by the query count.

    ``api.screen`` calls this rather than passing
    :data:`DEFAULT_CONFORMERS_PER_LAUNCH` straight through, because it is the
    layer that knows ``k`` and the number of queries and the source is not.
    A bench script that builds its own source should call it too.
    """
    from ..scores import N_SCORE_COLUMNS

    if dv is None:
        dv = N_SCORE_COLUMNS
    return _bounded_launch_size(target, n_queries, k,
                                budget_bytes=budget_bytes, dv=int(dv))


def _padded_rows(n: int, tile: bool) -> int:
    from ..layout import BTILE
    return (((int(n) + BTILE - 1) // BTILE) * BTILE) if tile else int(n)


@dataclass
class _Job:
    """One **launch**: a run of consecutive blocks from a single group.

    A run never spans a group. ``config.for_width(block.L)`` makes the launch
    arguments a function of the group's atom width, ``coord_scale`` is a
    per-group quantity, and "a molecule never straddles a group" is the premise
    ``MoleculeTopK.merge`` and the device ``moltopk_finish`` rest on. Job
    construction flushes at every group boundary for those three reasons, and
    :meth:`R3gSource._decode` re-checks the width of every constituent block
    rather than trusting that it did.
    """

    group_index: int
    block0: int
    """First constituent block's index within its group."""
    n_blocks: int
    row0: int
    """Row of this run's first conformer within its group."""
    gid_base: int
    n_conf: int
    """Conformers in the whole run.

    The rows this run stages are **contiguous**: ``gid = gid_base + out_t[r]``
    (``backend/_cuda_src/topk.cu:123``), so a padding row between two
    constituent blocks would shift every later global id by the size of the
    hole -- silently, onto ids belonging to other conformers, which the
    determinism proof in ``topk.cuh:44-51`` does not survive because it needs
    ids to be unique. That is safe only because every block but the last of a
    group is exactly ``BLOCK_CONF`` rows and ``BLOCK_CONF`` is a multiple of
    ``BTILE``; the decode asserts it rather than assuming it."""



def make_source(corpus, **kwargs):
    """Construct the format-3 source for an opened corpus."""
    from .r3g_source import R3gSource
    return R3gSource(corpus, **kwargs)

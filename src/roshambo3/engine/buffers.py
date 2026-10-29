"""Pinned host staging: the ring pool and the packer.

Why pinned memory at all, and it is not mainly bandwidth: **a copy out of
pageable memory cannot be asynchronous.** For as long as the producer decodes
into ordinary NumPy arrays there is no version of this pipeline in which the
host-to-device copy overlaps anything. Handing the producer a page-locked buffer
to write into costs nothing extra -- it has to write somewhere, and this is
simply where.
"""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np

from ..layout import BTILE


class PinnedPool:
    """A fixed ring of page-locked buffers, handed out and returned.

    Sized in *elements*, not shape: one block's coordinates are ``n * L * 3``
    int16 and both ``n`` and ``L``
    vary by size bucket, so the pool holds flat buffers and callers take a
    reshaped view of the prefix they need.

    ``dtype`` is the element type of the record the pool stages. It is int16 for
    the library wire record and the parameter exists rather than being hard
    wired because the pool is also what an ablation would use to stage the
    float32 record for comparison.

    ``pinned=False`` falls back to ordinary arrays, which exists so an ablation
    can measure what pinning is worth rather than assuming it.
    """

    def __init__(self, n_buffers: int, n_elems: int, pinned: bool = True,
                 dtype=np.int16):
        from ..backend.cuda import extension

        self.pinned = bool(pinned)
        self.n_elems = int(n_elems)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.int16), np.dtype(np.float32)):
            raise ValueError(
                f"a pinned pool stages the int16 library record or the float32 "
                f"one; {self.dtype} is neither")
        self._q: "queue.Queue" = queue.Queue()
        self.buffers = []
        ext = extension() if self.pinned else None
        alloc = (None if ext is None
                 else (ext.pinned_empty_i16 if self.dtype == np.int16
                       else ext.pinned_empty_f32))
        for _ in range(n_buffers):
            b = (alloc([self.n_elems]) if self.pinned
                 else np.empty(self.n_elems, self.dtype))
            self.buffers.append(b)
            self._q.put(b)
        # Verified, not assumed: a staging copy somewhere in the producer that
        # silently un-pins the buffer is exactly the defect this catches, and it
        # is one that was found in ROSHAMBO2 rather than imagined.
        self.is_pinned = bool(self.pinned and self.buffers
                              and ext is not None and ext.is_pinned(self.buffers[0]))

    #: Seconds ``acquire`` waits before deciding the pool has leaked.
    #: Long enough that a slow but healthy screen never trips it -- one launch
    #: over a large block is seconds, not minutes -- and short enough that a
    #: leak surfaces as a diagnosable error rather than a process that hangs.
    ACQUIRE_TIMEOUT_S = 600.0

    def acquire(self) -> np.ndarray:
        """Take a buffer, or fail with a diagnosis rather than hanging.

        An untimed ``Queue.get`` here is how a leaked buffer turns into a
        process that never exits and never says why: the worker blocks, the
        executor's ``shutdown(wait=True)`` joins it, and the exception that
        caused the leak is never printed. The bound converts that into a
        message naming the pool.
        """
        try:
            return self._q.get(timeout=self.ACQUIRE_TIMEOUT_S)
        except queue.Empty:
            raise RuntimeError(
                f"no pinned buffer became available within "
                f"{self.ACQUIRE_TIMEOUT_S:.0f}s. All {len(self.buffers)} are "
                f"checked out, which means either the pool is smaller than the "
                f"number of blocks in flight, or a buffer was leaked by an "
                f"error on a path that did not release it.") from None

    def release(self, b: np.ndarray) -> None:
        """Hand a buffer back, refusing one this pool does not own.

        A bare ``put`` accepts anything, including a buffer released twice --
        which grows the queue past ``len(self.buffers)`` and then hands the same
        memory to two consumers at once, so one block's coordinates are packed
        over another's while both are in flight. That corrupts scores rather
        than raising, and it is the failure the release paths added around every
        acquire could plausibly introduce.

        Identity, not equality: these are large arrays and ``in`` on a list of
        them would compare element-wise.
        """
        if not any(b is x for x in self.buffers):
            raise ValueError(
                "released a buffer this pool does not own. Buffers must be "
                "returned to the pool they came from.")
        # Under the queue's own lock, and the check and the put together.
        # `Queue.get` pops under this mutex, and releases run concurrently by
        # design -- a worker releases on a failed decode while the consumer
        # releases in its `finally`. Scanning `self._q.queue` outside the lock
        # can raise "deque mutated during iteration" *before* the put, leaking
        # the very buffer this check protects and, since it runs inside a
        # `finally`, replacing the real exception with a bookkeeping one. A
        # check that is not atomic with the put would also let the double
        # release it guards against slip between them.
        with self._q.mutex:
            if any(b is x for x in self._q.queue):
                raise ValueError(
                    "released a buffer that is already in the pool. A double "
                    "release hands the same memory to two blocks at once, "
                    "which corrupts scores silently instead of raising.")
            self._q.queue.append(b)
            self._q.unfinished_tasks += 1
            self._q.not_empty.notify()

    @property
    def available(self) -> int:
        """Buffers not currently checked out. For tests and diagnostics."""
        return self._q.qsize()

    def __len__(self) -> int:
        return len(self.buffers)


# There is no all-zero atom-type plane to upload (at L = 40 and 65,536
# conformers per launch one would be 10.5 MB, ~20% of the H2D bytes): the
# kernel takes the per-conformer count vector, and a shape-only corpus's is one
# byte wide.


#: The quantization step the int16 LIBRARY WIRE RECORD means, in Angstrom.
#: It is ``ROSHAMBO3_STAGE_INV_Q_XYZ`` in ``kernels.cuh`` and
#: ``roshambo3.numerics.WIRE_QUANT_STEP`` pins the two together. The corpus
#: format's own step (``format.constants.COORD_SCALE``) happens to be the same
#: number, and that coincidence is what makes shipping the stored shorts raw an
#: encoding rather than a conversion -- so it is CHECKED here rather than
#: assumed, on every block.


def require_wire_quantization(coord_scale: float) -> None:
    """Refuse a corpus whose int16 step is not the wire record's.

    The shorts on the wire are read by the kernel as
    ``ROSHAMBO3_STAGE_INV_Q_XYZ * (float)v``. A corpus stored at a different
    step would be scaled by the wrong constant with no other symptom -- every
    coordinate uniformly wrong, every score plausible. With no host-side
    dequantization, this equality is load-bearing, so it is asserted
    at the one place the shorts are handed over.
    """
    from ..numerics import compiled_wire_constants, WireQuantizationError

    wire_step, _ = compiled_wire_constants()
    if np.float32(coord_scale) != np.float32(wire_step):
        raise WireQuantizationError(
            f"this corpus quantizes coordinates at {coord_scale} A and the "
            f"int16 wire record means {wire_step} A "
            f"(ROSHAMBO3_STAGE_INV_Q_XYZ). Shipping its shorts raw would scale "
            f"every coordinate by the wrong constant.")


def pack_into(xyz: np.ndarray, L: int, dst_flat: np.ndarray,
              coord_scale: float, tile: bool = True, planar: bool = False
              ) -> tuple[np.ndarray, int]:
    """Lay one block's int16 planes into ``dst_flat`` in tile order, unchanged.

    ``xyz`` is the ``[nb, 3, L]`` int16 planar array as stored on disk. The
    result is the ``[n_rows, L, 3]`` **int16** view the kernel takes, written
    directly into the pinned buffer -- the transposition into tile order happens
    on the way in rather than as a second full-array copy afterwards, which is
    what would throw the pinning away.

    Returns ``(view, nb)``. ``nb`` is the real conformer count; ``view`` may be
    longer, padded up to a whole number of tiles.

    ``planar`` writes the PLANAR tile -- ``[tile][atom][component][lane]``
    rather than ``[tile][atom][lane][component]``. The buffer is the same size
    and holds the same values; only the stride between a row's components
    changes, from 1 to ``BTILE``. It is the layout
    :func:`roshambo3.layout.tile32` produces with ``planar=True``, addressed
    directly here for the same reason the interleaved one is. A block packed
    this way must be scored with ``library_tile_mode < 0``: nothing about the
    array says which layout it is in, and reading one at the other's stride
    reads another conformer's coordinates without raising.

    **This converts nothing.** Dequantizing to float32 with a fourth component
    would carry 16 B/atom for a record the disk holds in 6. What crosses the wire is the
    corpus's own shorts, and the kernel applies ``ROSHAMBO3_STAGE_INV_Q_XYZ``
    once, at the point of use (``r3_fetch`` in ``kernels.cuh``). That is the
    same two roundings a float32 record would go through -- the host's
    ``float32(v) * float32(WIRE_QUANT_STEP)`` and then the kernel's ``cscale *`` -- and
    NOT one fused rounding, which ``doc/NUMERICS_POLICY.md`` 4c records costing
    17.7% of pairs their bits.

    ``coord_scale`` is therefore not applied; it is **checked**. The wire
    record's meaning is "int16 at ``ROSHAMBO3_STAGE_INV_Q_XYZ``", so a corpus
    quantized at any other step may not be shipped raw, and this refuses rather
    than shipping coordinates the kernel would scale by the wrong constant.

    **There is no fourth (weight) component.** It would be 1.0 inside
    ``j < n_real`` and 0.0 outside -- a per-slot expansion of exactly the count
    vector already shipped (``format/dataset.py::combine_shape_color`` writes
    ``cols < n_real``). The kernel reconstructs it from the count word where
    anything reads it (``r3_slot_weight``); the only reader on a library record
    is ``get_gradient``'s ``NOW = false`` arm, whose loop is bounded by
    ``n_real`` so the answer is always 1.0f. ``volume_single``'s shape loop is
    not a library-side reader. ``volume_test_kernel``'s call IS still such a reader -- it passes the padded
    width for the real count -- but it is the PyTorch-oracle wrapper and takes
    the float32 record, not this one.

    What this function does do is write the pinned buffer **once**. The tiled
    layout is reached by addressing the destination in tile coordinates, not by
    building the untiled array and permuting it afterwards: that costs a
    scratch array, a copy, and an in-place ``np.copyto`` whose aliased source
    numpy silently buffers -- four traversals and two allocations for one pass
    of work.
    """
    require_wire_quantization(coord_scale)
    nb = int(xyz.shape[0])
    n_rows = ((nb + BTILE - 1) // BTILE) * BTILE if tile else nb
    need = n_rows * L * 3
    if dst_flat.dtype != np.int16:
        raise ValueError(
            f"the library wire record is int16; the buffer is {dst_flat.dtype}")
    if dst_flat.size < need:
        raise ValueError(f"pinned buffer holds {dst_flat.size} int16, "
                         f"block needs {need}")
    out = dst_flat[:need].reshape(n_rows, L, 3)
    if nb == 0:
        out[...] = 0
        return out, nb

    if not tile:
        out[:nb] = xyz.transpose(0, 2, 1)
        if n_rows != nb:
            out[nb:] = 0
        return out, nb

    n_tiles = n_rows // BTILE
    # [tile][atom][lane][component] -- ROSHAMBO3_BTILE's layout -- or, with
    # `planar`, [tile][atom][component][lane]. Addressed directly either way;
    # roshambo3.layout.tile32 is the definition this mirrors.
    dst = out.reshape(n_tiles, L, 3, BTILE) if planar \
        else out.reshape(n_tiles, L, BTILE, 3)
    n_full = nb // BTILE
    rem = nb - n_full * BTILE

    if n_full:
        # [conf, 3, L] -> [tile, lane, 3, L] -> [tile, L, lane, 3]
        #                                    or [tile, L, 3, lane]
        x = xyz[:n_full * BTILE].reshape(n_full, BTILE, 3, L)
        dst[:n_full] = x.transpose(0, 3, 2, 1) if planar \
            else x.transpose(0, 3, 1, 2)
    if rem:
        # The last tile is partial. Its unused lanes are the only padding in
        # the buffer, and they are zeroed here in *tile* coordinates. The form
        # this replaces zeroed rows `nb:` in untiled coordinates and let the
        # permutation carry them into place; addressing the destination
        # directly means the zeroing has to be addressed directly too.
        d = dst[n_full]
        if planar:
            # [conf, 3, L] -> [L, 3, conf]: the lane axis is last.
            d[:, :, :rem] = xyz[n_full * BTILE:nb].transpose(2, 1, 0)
            d[:, :, rem:] = 0
        else:
            d[:, :rem, :] = xyz[n_full * BTILE:nb].transpose(2, 0, 1)
            d[:, rem:, :] = 0
    return out, nb

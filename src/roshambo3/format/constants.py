"""Shared coordinate and preparation helpers for corpus format 3."""
from __future__ import annotations

import numpy as np

from .r3g.limits import BUCKET_EDGES, L_ALIGN, LIMITS, MAGIC, MAJOR, STEP_LOG2

COORD_SCALE = 2.0 ** STEP_LOG2
VERSION = MAJOR
BLOCK_CONF = LIMITS["block_rows"]
MAX_CONF_PER_GROUP = 2_000_000
MAX_COLOR_TYPE_COUNT = LIMITS["features_per_type"]


def bucket_of(n_real: np.ndarray, edges=BUCKET_EDGES) -> np.ndarray:
    """Bucket indices; interior edges belong to the next bucket."""
    return np.searchsorted(np.asarray(edges[1:-1]), np.asarray(n_real), side="right")


def pad_L(n_max: int, align: int = L_ALIGN) -> int:
    """Pad a coordinate width to its storage alignment."""
    return int((n_max + align - 1) // align * align)


def quantize(coords_f: np.ndarray, scale: float = COORD_SCALE) -> np.ndarray:
    """Round-to-nearest int16 quantization.  Raises if anything would clip.

    The finiteness check is not defensive tidying, it is what makes the range
    check work. ``NaN > 32767`` is False, so a NaN passed the guard and reached
    ``astype(np.int16)`` -- an undefined cast that in practice produced a
    plausible-looking atom at -128.0 A. And ``np.max`` propagates NaN, so one
    NaN made the whole comparison False and hid a co-resident Inf that would
    otherwise have been caught.
    """
    q = np.rint(np.asarray(coords_f, dtype=np.float64) / scale)
    if q.size and not np.all(np.isfinite(q)):
        n = int((~np.isfinite(q)).sum())
        raise ValueError(
            f"{n} of {q.size} coordinates are not finite. They cannot be "
            f"quantized -- an int16 cast of NaN is undefined and silently "
            f"yields a coordinate that looks real.")
    if q.size and (q.max() > 32767 or q.min() < -32768):
        raise OverflowError(
            f"coordinate out of int16 range at scale {scale}: "
            f"[{q.min()*scale:.3f}, {q.max()*scale:.3f}] A")
    return q.astype(np.int16)

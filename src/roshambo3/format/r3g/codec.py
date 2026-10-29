"""The one block encode/decode entry point of the format-3 reader and writer.

Every block the writer encodes and every block the reader or the ``deep``
validator decodes goes through :func:`encode_block` / :func:`decode_block`.
They take the block's **run-table slice** -- the ``runs`` records of its
molecule range ``[m0, m1)`` -- because that is what ``doc/FORMAT.md``
section 17 and the native codec take, never a per-row array.

Two implementations sit behind them:

* the native codec (``roshambo3.format.r3g.native``, typed by
  ``native.pyi``: ``decode_block(block, runs, rows, L_heavy)`` and
  ``encode_block(coords, runs)``), resolved once by :func:`_native`;
* the pure-Python reference (``reference.encode_block`` /
  ``reference.decode_block``), used when the extension does not provide the
  native codec. It is the oracle and is slow: test-sized corpora only.

Both MUST produce identical bytes and identical decoded arrays (section 17
item 9). When the ``native`` module is absent, or the extension
was built without the codec, :func:`_native` returns ``None`` and the
reference is used.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Optional

import numpy as np

from . import reference
from .limits import BLOCK_HEADER_SIZE, LIMITS, RUN_DTYPE, block_length


@functools.lru_cache(maxsize=None)
def _native() -> Optional[Any]:
    """The native codec module, or ``None`` when this tree or its
    extension does not provide it. ``ROSHAMBO3_R3G_CODEC=reference`` forces
    the reference (tests compare the two paths)."""
    if os.environ.get("ROSHAMBO3_R3G_CODEC", "") == "reference":
        return None
    try:
        from . import native  # type: ignore[attr-defined]
        native._ext()         # ImportError when the extension has no r3g codec
    except ImportError:
        return None
    return native


def backend() -> str:
    """``"native"`` or ``"reference"``: which implementation the dispatch uses."""
    return "native" if _native() is not None else "reference"


def _run_columns(runs) -> tuple[np.ndarray, np.ndarray]:
    """``(n_conf, n_real)`` as int64 from a ``RUN_DTYPE`` slice (or a pair)."""
    if isinstance(runs, np.ndarray) and runs.dtype == RUN_DTYPE:
        return runs["n_conf"].astype(np.int64), runs["n_real"].astype(np.int64)
    n_conf, n_real = runs
    return np.asarray(n_conf, np.int64).reshape(-1), np.asarray(n_real, np.int64).reshape(-1)


def _rows_n_real(runs) -> np.ndarray:
    """Per-row ``n_real`` of a block, after the section 17 item 5/10 checks
    that bound the expansion: at least one molecule, every ``n_conf`` and
    ``n_real`` at least 1 and in range, at most 4,096 rows."""
    n_conf, n_real = _run_columns(runs)
    if n_conf.size == 0 or n_conf.size != n_real.size:
        raise ValueError("decode: the run-table slice is empty or ragged")
    if n_conf.min() < 1 or n_conf.max() > LIMITS["n_conf_per_molecule"]:
        raise ValueError("decode: n_conf outside [1, 4096]")
    if n_real.min() < 1 or n_real.max() > LIMITS["n_real"]:
        raise ValueError("decode: n_real outside [1, 255]")
    rows = int(n_conf.sum())
    if rows > LIMITS["block_rows_single_molecule"]:
        raise ValueError(f"decode: {rows} rows exceed the block row cap")
    return np.repeat(n_real, n_conf)


def decode_block(data, runs, L_heavy: int) -> np.ndarray:
    """Decode one block to the padded ``int16[rows, 3, L_heavy]`` layout
    (section 4.3). ``runs`` is the block's run-table slice. Raises
    ``ValueError`` naming the failed check on any malformed input."""
    nr = _rows_n_real(runs)
    if not 1 <= int(L_heavy) <= LIMITS["L_heavy"]:
        raise ValueError(f"decode: L_heavy {L_heavy} out of range")
    nat = _native()
    if nat is not None:
        return nat.decode_block(data, _run_records(runs), int(nr.size), int(L_heavy))
    return reference.decode_block(bytes(data), nr, int(L_heavy))


def encode_block(coords: np.ndarray, runs) -> bytes:
    """Encode one block from padded ``int16[rows, 3, L]`` and its run-table
    slice: per-axis minimum and minimal width, BP256-16, zero tails
    (section 5.5). Byte-identical across implementations."""
    nr = _rows_n_real(runs)
    xyz = np.asarray(coords)
    if xyz.dtype != np.int16 or xyz.ndim != 3 or xyz.shape[0] != nr.size or xyz.shape[1] != 3:
        raise ValueError("encode: coords must be int16[rows, 3, L] with one row per conformer")
    nat = _native()
    if nat is not None:
        return nat.encode_block(np.ascontiguousarray(xyz), _run_records(runs))
    return reference.encode_block(xyz, nr)


def _run_records(runs) -> np.ndarray:
    """The run slice as contiguous ``RUN_DTYPE`` records (what the native
    binding takes); a ``(n_conf, n_real)`` pair is packed into records."""
    if isinstance(runs, np.ndarray) and runs.dtype == RUN_DTYPE:
        return np.ascontiguousarray(runs)
    n_conf, n_real = _run_columns(runs)
    rec = np.zeros(n_conf.size, dtype=RUN_DTYPE)
    rec["n_conf"], rec["n_real"] = n_conf, n_real
    return rec


def block_total(runs) -> int:
    """Real atoms in a block: ``sum n_conf * n_real`` over its molecules (5.2)."""
    n_conf, n_real = _run_columns(runs)
    return int((n_conf * n_real).sum())


def tail_values(stream: bytes, w: int, total: int) -> np.ndarray:
    """The stored ``u`` of the unused positions ``[total, 256 * G)`` of an axis
    stream's last packing group (section 5.4): all zero in a valid block.
    Vectorized over the 16 lanes and ``w`` words of that one group."""
    groups = -(-int(total) // 256)
    r = int(total) - 256 * (groups - 1)            # values used in the last group
    if w == 0 or r == 256:
        return np.zeros(0, dtype=np.int64)
    words = np.frombuffer(stream, dtype="<u2", count=w * 16,
                          offset=(groups - 1) * w * 32).reshape(w, 16).astype(np.int64)
    # lane bitstream of each lane as a Python-free bit matrix [16 lanes, 16*w bits]
    bits = ((words.T[:, :, None] >> np.arange(16)) & 1).reshape(16, 16 * w)
    k = np.arange(16)
    pos_bits = bits[:, (k[:, None] * w + np.arange(w)).reshape(-1)].reshape(16, 16, w)
    u = (pos_bits << np.arange(w)).sum(axis=2)        # [lane l, position k]
    values = u.T.reshape(256)                           # index 16*k + l
    return values[r:]


def block_widths(data) -> tuple[int, int, int]:
    """``(w_x, w_y, w_z)`` from a block's 32-byte header (unchecked)."""
    b = bytes(data[6:9])
    return b[0], b[1], b[2]


def expected_block_length(data, runs) -> int:
    return block_length(block_total(runs), *block_widths(data))


__all__ = ["backend", "decode_block", "encode_block", "block_total", "tail_values",
           "BLOCK_HEADER_SIZE"]

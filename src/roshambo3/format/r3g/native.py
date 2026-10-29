"""The native format-3 coordinate codec: thin Python wrappers.

The typed interface, with every argument's meaning, is ``native.pyi`` next to
this file; the C++ contract is ``backend/_cpu_src/r3g_codec.hpp``. This module
only normalizes arguments, allocates the small per-row outputs a caller did not
provide, and calls the GIL-releasing functions bound into
``roshambo3._roshambo3_cpu`` (``r3g_*``). Every refusal is a
:class:`CodecError`, raised by the binding itself.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

BTILE = 32
BLOCK_HEADER_SIZE = 32
RUN_RECORD_SIZE = 16

__all__ = ["BTILE", "BLOCK_HEADER_SIZE", "RUN_RECORD_SIZE", "CodecError",
           "FeatureSlice", "SelfOverlaps", "Timings", "BlockResult",
           "block_length", "gpu_layout_elements", "decode_block",
           "decode_merged", "write_gpu_block", "write_gpu_launch", "write_cpu_block",
           "encode_block", "read_into", "isa", "compiled_isas", "force_isa"]


class CodecError(ValueError):
    """A refusal by the native codec; ``code`` names the failed check."""

    def __init__(self, message: str, code: str = "decode.argument", status: int = 1):
        super().__init__(message)
        self.code = code
        self.status = status


class FeatureSlice(NamedTuple):
    offsets: np.ndarray
    types: np.ndarray
    atom_offsets: np.ndarray
    atoms: np.ndarray


class SelfOverlaps(NamedTuple):
    vaa: np.ndarray
    color_self: Optional[np.ndarray]
    first_row: int


class Timings(NamedTuple):
    decode: float
    color: float
    self_overlap: float


class BlockResult(NamedTuple):
    coords: np.ndarray
    n_rows: int
    rows: int
    counts: np.ndarray
    count_words: np.ndarray
    vaa: Optional[np.ndarray]
    color_self: Optional[np.ndarray]
    timings: Timings


_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        try:
            from roshambo3 import _roshambo3_cpu as ext
        except ImportError as e:  # pragma: no cover - build problem, not a refusal
            raise ImportError("the native format-3 codec needs roshambo3's CPU "
                              "extension (_roshambo3_cpu), which did not import") from e
        if not hasattr(ext, "r3g_decode_heavy"):  # pragma: no cover
            raise ImportError("this _roshambo3_cpu build has no r3g codec "
                              "(built before the codec existed)")
        _EXT = ext
    return _EXT


def _runs_bytes(runs) -> np.ndarray:
    a = np.asarray(runs)
    if not a.flags.c_contiguous:
        raise CodecError("decode.argument: the run slice must be C-contiguous")
    if a.nbytes % RUN_RECORD_SIZE:
        raise CodecError(f"decode.argument: the run slice is {a.nbytes} bytes, "
                         f"not a multiple of {RUN_RECORD_SIZE}")
    return a.reshape(-1).view(np.uint8)


def _block_bytes(block):
    if isinstance(block, np.ndarray):
        if not block.flags.c_contiguous:
            raise CodecError("decode.argument: the block must be C-contiguous")
        return block.reshape(-1).view(np.uint8)
    return memoryview(block).cast("B")


def _features(features, n_types):
    if features is None:
        return None
    f = FeatureSlice(*features)
    return (np.ascontiguousarray(f.offsets, dtype="<u4"),
            np.ascontiguousarray(f.types, dtype=np.uint8),
            np.ascontiguousarray(f.atom_offsets, dtype="<u4"),
            np.ascontiguousarray(f.atoms, dtype=np.uint8))


def _out(arr, dtype, n, name):
    if arr is None:
        return np.empty(n, dtype=dtype)
    if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype(dtype) \
            or not arr.flags.c_contiguous or not arr.flags.writeable:
        raise CodecError(f"decode.argument: {name} must be a writeable C-contiguous "
                         f"{np.dtype(dtype).name} array")
    if arr.size < n:
        raise CodecError(f"decode.output_capacity: {name} holds {arr.size} elements, "
                         f"the block needs {n}", "decode.output_capacity", 16)
    return arr


def _so(self_overlaps, rows, vaa_out, color_self_out):
    if self_overlaps is None:
        return None, None, None, None, 0
    so = SelfOverlaps(*self_overlaps)
    vaa = np.ascontiguousarray(so.vaa, dtype="<f4")
    cs = None if so.color_self is None else np.ascontiguousarray(so.color_self, dtype="<f8")
    vo = _out(vaa_out, np.float32, rows, "vaa_out")
    co = None if cs is None else _out(color_self_out, np.float64, rows, "color_self_out")
    return vaa, cs, vo, co, int(so.first_row)


def block_length(total: int, w_x: int, w_y: int, w_z: int) -> int:
    return int(_ext().r3g_block_length(int(total), int(w_x), int(w_y), int(w_z)))


def gpu_layout_elements(rows: int, L: int) -> tuple[int, int]:
    _check_shape(int(rows), 1, int(L))
    n_rows, n = _ext().r3g_gpu_layout_elements(int(rows), int(L))
    return int(n_rows), int(n)


def _check_shape(rows, L_heavy, L=None, n_types=0):
    checks = ((rows < 1, "layout.blocks", 2),
              (not 1 <= L_heavy <= 256, "limit.L_heavy", 8),
              (L is not None and not 1 <= L <= 512, "limit.L", 9),
              (not 0 <= n_types <= 6, "limit.n_types", 10),
              (rows > 4096, "limit.block_rows", 4))
    for bad, code, status in checks:
        if bad:
            raise CodecError(f"{code}: invalid block dimensions", code, status)


def decode_block(block, runs, rows: int, L_heavy: int, out=None) -> np.ndarray:
    rows, L_heavy = int(rows), int(L_heavy)
    _check_shape(rows, L_heavy)
    n = rows * 3 * L_heavy
    o = _out(out, np.int16, n, "out")
    _ext().r3g_decode_heavy(_block_bytes(block), _runs_bytes(runs), rows, L_heavy, o)
    return o.reshape(-1)[:n].reshape(rows, 3, L_heavy)


def decode_merged(block, runs, rows: int, L_heavy: int, L: int, n_types: int,
                  features, out=None):
    rows, L_heavy, L, n_types = int(rows), int(L_heavy), int(L), int(n_types)
    _check_shape(rows, L_heavy, L, n_types)
    n = rows * 3 * L
    o = _out(out, np.int16, n, "out")
    counts = np.empty((rows, n_types + 1), dtype=np.uint8)
    t = _ext().r3g_decode_merged(_block_bytes(block), _runs_bytes(runs), rows,
                                 int(L_heavy), L, n_types, _features(features, n_types),
                                 o, counts)
    return o.reshape(-1)[:n].reshape(rows, 3, L), counts, Timings(*t)


def _common_outputs(rows, n_types, counts_out, count_words_out):
    counts = _out(counts_out, np.uint8, rows * (n_types + 1), "counts_out")
    words = _out(count_words_out, np.uint64, rows, "count_words_out")
    return counts, words


def write_gpu_block(block, runs, rows: int, L_heavy: int, L: int, n_types: int,
                    features, out, *, tile_mode: int = 1, counts_out=None,
                    count_words_out=None, self_overlaps=None, vaa_out=None,
                    color_self_out=None) -> BlockResult:
    rows, L_heavy, L, n_types = int(rows), int(L_heavy), int(L), int(n_types)
    _check_shape(rows, L_heavy, L, n_types)
    n_rows, n = gpu_layout_elements(rows, L)
    o = _out(out, np.int16, n, "out")
    counts, words = _common_outputs(rows, n_types, counts_out, count_words_out)
    vaa, cs, vo, co, first_row = _so(self_overlaps, rows, vaa_out, color_self_out)
    t, cs_present = _ext().r3g_write_gpu(
        _block_bytes(block), _runs_bytes(runs), rows, int(L_heavy), L, n_types,
        _features(features, n_types), int(tile_mode), o, counts, words,
        vaa, cs, first_row, vo, co)
    return BlockResult(o.reshape(-1)[:n].reshape(n_rows, L, 3), n_rows, rows,
                       counts.reshape(-1)[:rows * (n_types + 1)].reshape(rows, n_types + 1),
                       words.reshape(-1)[:rows],
                       None if vo is None else vo.reshape(-1)[:rows],
                       co.reshape(-1)[:rows] if (co is not None and cs_present) else None,
                       Timings(*t))


def write_gpu_launch(blocks, rows: int, L_heavy: int, L: int, n_types: int,
                     out, *, tile_mode: int = 1, counts_out=None,
                     mol_ids_out=None):
    """Write whole blocks on one launch-relative tile grid, expanding IDs.

    ``blocks`` contains ``(bytes, runs, rows, FeatureSlice | None)`` tuples.
    Physical blocks retain the format's 4096-row limit; a launch may be larger.
    All output capacities and constituent inputs are checked natively.
    Returns ``(coords, counts, mol_id)`` views backed by caller-owned outputs.
    """
    rows, L_heavy, L, n_types = int(rows), int(L_heavy), int(L), int(n_types)
    if rows < 1:
        raise CodecError("layout.blocks: a launch must contain rows", "layout.blocks", 2)
    _check_shape(1, L_heavy, L, n_types)
    # gpu_layout_elements' public wrapper enforces a *block* row limit.
    n_rows, n = _ext().r3g_gpu_layout_elements(rows, L)
    o = _out(out, np.int16, n, "out")
    counts = _out(counts_out, np.uint8, rows * (n_types + 1), "counts_out")
    ids = _out(mol_ids_out, np.int64, rows, "mol_ids_out")
    inputs = [(_block_bytes(b), _runs_bytes(r), int(nr), _features(f, n_types))
              for b, r, nr, f in blocks]
    _ext().r3g_write_gpu_launch(inputs, rows, L_heavy, L, n_types, int(tile_mode),
                              o, counts, ids)
    return (o.reshape(-1)[:n].reshape(n_rows, L, 3),
            counts.reshape(-1)[:rows * (n_types + 1)].reshape(rows, n_types + 1),
            ids.reshape(-1)[:rows])


def write_cpu_block(block, runs, rows: int, L_heavy: int, L: int, n_types: int,
                    features, coord_scale: float, *, out=None, counts_out=None,
                    count_words_out=None, self_overlaps=None, vaa_out=None,
                    color_self_out=None) -> BlockResult:
    rows, L_heavy, L, n_types = int(rows), int(L_heavy), int(L), int(n_types)
    _check_shape(rows, L_heavy, L, n_types)
    n = rows * L * 3
    o = _out(out, np.float32, n, "out")
    counts, words = _common_outputs(rows, n_types, counts_out, count_words_out)
    vaa, cs, vo, co, first_row = _so(self_overlaps, rows, vaa_out, color_self_out)
    t, cs_present = _ext().r3g_write_cpu(
        _block_bytes(block), _runs_bytes(runs), rows, int(L_heavy), L, n_types,
        _features(features, n_types), float(coord_scale), o, counts, words,
        vaa, cs, first_row, vo, co)
    return BlockResult(o.reshape(-1)[:n].reshape(rows, L, 3), rows, rows,
                       counts.reshape(-1)[:rows * (n_types + 1)].reshape(rows, n_types + 1),
                       words.reshape(-1)[:rows],
                       None if vo is None else vo.reshape(-1)[:rows],
                       co.reshape(-1)[:rows] if (co is not None and cs_present) else None,
                       Timings(*t))


def encode_block(coords, runs) -> bytes:
    c = np.asarray(coords)
    if c.dtype != np.int16 or c.ndim != 3 or c.shape[1] != 3:
        raise CodecError("decode.argument: encode_block takes int16[rows, 3, L_heavy]")
    c = np.ascontiguousarray(c)
    return _ext().r3g_encode(c, int(c.shape[0]), int(c.shape[2]), _runs_bytes(runs))


def read_into(fd: int, offset: int, length: int, out=None) -> np.ndarray:
    """Read exactly ``length`` bytes at ``offset`` of ``fd`` with bounded
    ``pread`` (GIL released); a short read is ``io.short_read``."""
    length = int(length)
    o = _out(out, np.uint8, length, "out")
    _ext().r3g_read_into(int(fd), int(offset), length, o)
    return o.reshape(-1)[:length]


def isa() -> str:
    return str(_ext().r3g_isa())


def compiled_isas() -> tuple[str, ...]:
    return tuple(_ext().r3g_compiled_isas())


def force_isa(name: str) -> None:
    _ext().r3g_force_isa(str(name))


def _set_builder_mode(mode: str) -> None:
    """Private test/benchmark override: auto, baseline, or fused.

    Process-wide; never mutate during concurrent calls. Auto enables the measured
    NEON GPU writer; CPU and other ISA/consumer combinations retain baseline.
    Public writer signatures are unchanged.
    """
    _ext().r3g_set_builder_mode(str(mode))


def _builder_info(consumer: str, coord_scale: float = 2.0 ** -8) -> dict:
    """Describe normal nonalias dispatch, including the selected scale path.

    Hostile native aliases use the guarded baseline regardless of the override;
    Python bindings reject aliases. GPU scales describe integer output.
    """
    return dict(_ext().r3g_builder_info(str(consumer), float(coord_scale)))

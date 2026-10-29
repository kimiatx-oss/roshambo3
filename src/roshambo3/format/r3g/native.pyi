"""Typed interface of the native format-3 coordinate codec.

The implementation is C++ (``backend/_cpu_src/r3g_codec.hpp``), bound into
``roshambo3._roshambo3_cpu`` and wrapped by ``roshambo3.format.r3g.native``.
The reader, the GPU source and the CPU workers code against THIS file;
changing a name, an argument's order, dtype or shape, or a return value
requires changing all of them.

Conventions shared by every function
------------------------------------
* ``block`` is one block's bytes exactly as stored (32-byte header plus the
  three axis streams): any C-contiguous buffer (``bytes``, ``memoryview``,
  ``uint8`` array). It must come from a bounded ``pread`` (:func:`read_into`)
  or from an ``mmap`` only where ``SIGBUS`` is handled (FORMAT.md 17 item 12). Its length must equal FORMAT.md 5.1's
  ``block_length``; nothing about it is trusted.
* ``runs`` is the block's **run-table slice** ``runs[m0:m1]`` of the group's
  ``runs`` section: ``uint8[n_molecules, 16]`` or the reference's structured
  ``RUN_DTYPE[n_molecules]`` (any C-contiguous array whose byte size is a
  multiple of 16). ``(m0, m1)`` come from the block table.
* ``rows``, ``L_heavy``, ``L`` (device width) and ``n_types`` come from the
  group table / validated group header -- never from the block. ``rows`` must
  equal the run slice's ``n_conf`` sum; a zero-row block is refused.
* ``features`` is the block's **feature-table slice** (a :class:`FeatureSlice`):
  ``offsets = feature_offsets[m0 : m1 + 1]`` **not rebased** -- its values are
  group-wide feature indices -- and the other three arrays are the group's
  whole ``feature_types``, ``feature_atom_offsets`` and ``feature_atoms``
  sections. Every offset is checked against the length of the array it
  indexes, and monotone, before it is used. ``None`` iff ``n_types == 0``.
* Every refusal raises :class:`CodecError` (a ``ValueError``) whose message
  begins with the failed check's code (``"limit.bit_width: ..."``); ``code``
  is the FORMAT.md 16.5 refusal name where one exists (``limit.bit_width``,
  ``limit.block_length``, ``limit.block_rows``, ``limit.n_real``,
  ``limit.n_conf_per_molecule``, ``limit.L_heavy``, ``limit.L``,
  ``limit.n_types``, ``limit.atoms_per_feature``, ``limit.features_per_type``,
  ``limit.features_per_molecule``, ``layout.blocks``, ``layout.widths``,
  ``layout.features``, ``layout.block_header``) and otherwise a
  ``decode.<name>`` code (``decode.overflow``, ``decode.output_capacity``,
  ``decode.argument``, ``decode.self_overlap_range``, ``decode.allocation``,
  ``decode.isa_unavailable``, ``decode.features_missing``,
  ``encode.range``, ``io.short_read``, ``io.read_error``). Arguments of the
  wrong dtype or shape raise ``TypeError``
  or :class:`CodecError` (``decode.argument``).
* Inputs need no alignment (a slice of any file buffer is accepted); outputs
  passed in must be naturally aligned (numpy arrays are).
* :func:`decode_block` (the oracle-shaped heavy decode) returns no timings;
  the three consumer calls (:func:`decode_merged`, :func:`write_gpu_block`,
  :func:`write_cpu_block`) return the (decode, color, self-overlap) triple.
* Calls release the GIL for the native work. Outputs passed in are written in
  place and must be C-contiguous with the stated dtype and at least the
  stated number of elements; only the stated prefix is written.
* Decoded values are ``int16_wrap(min + u)`` (FORMAT.md 5.6) on every ISA path;
  all paths are bit-identical.
"""

from typing import NamedTuple, Optional, Union

import numpy as np
from numpy.typing import NDArray

_Buffer = Union[bytes, bytearray, memoryview, NDArray[np.uint8]]

#: Rows per GPU tile (``layout.BTILE``).
BTILE: int  # = 32
#: Bytes in a block header.
BLOCK_HEADER_SIZE: int  # = 32
#: Bytes in a run-table record.
RUN_RECORD_SIZE: int  # = 16


class CodecError(ValueError):
    """A refusal by the native codec. ``code`` names the failed check
    (FORMAT.md 16.5 where it overlaps, else ``decode.*`` / ``encode.*``);
    ``status`` is the native enum value (``r3g_codec.hpp::Status``)."""
    code: str
    status: int


class FeatureSlice(NamedTuple):
    """The block's feature-table slice (FORMAT.md 8.1; section 17 item 7)."""
    offsets: NDArray[np.uint32]       # feature_offsets[m0 : m1 + 1], group-wide values
    types: NDArray[np.uint8]          # the group's whole feature_types
    atom_offsets: NDArray[np.uint32]  # the group's whole feature_atom_offsets
    atoms: NDArray[np.uint8]          # the group's whole feature_atoms


class SelfOverlaps(NamedTuple):
    """The group's self-overlap sections, whole (FORMAT.md 18), and the block's
    ``first_row`` from the block table. ``color_self`` is ``None`` when the
    section is absent (18.2); the binding then reports it absent."""
    vaa: NDArray[np.float32]                    # f32[n_conf]
    color_self: Optional[NDArray[np.float64]]   # f64[n_conf] or None
    first_row: int


class Timings(NamedTuple):
    """Wall seconds of one block call, attributed to the stages below."""
    decode: float        # baseline: checks/unpack/layout; fused: validation/unpack
    color: float         # baseline: feature checks/derive; fused: derive/counts/emit
    self_overlap: float  # the vaa / color_self binding (0.0 when not asked)


class BlockResult(NamedTuple):
    """What a writer call produced.

    ``coords``: GPU -- the ``int16[n_rows, L, 3]`` view of ``out`` in tile order
    (reinterpret per ``tile_mode``; same contract as ``buffers.pack_into``'s
    return); CPU -- ``float32[rows, L, 3]``.
    ``n_rows``: rows in ``coords`` (GPU: ``ceil(rows / 32) * 32``; CPU: ``rows``).
    ``rows``: real conformers in the block.
    ``counts``: ``uint8[rows, n_types + 1]``, FORMAT.md 8.5 count vectors.
    ``count_words``: ``uint64[rows]``, the device count word (byte 0 ``n_real``,
    bytes 1..6 per-type counts, zero past ``n_types``, byte 7 ``n_color``).
    ``vaa``: ``float32[rows]`` or ``None`` when ``self_overlaps`` was not given.
    ``color_self``: ``float64[rows]``, or ``None`` when not given or absent.
    """
    coords: NDArray
    n_rows: int
    rows: int
    counts: NDArray[np.uint8]
    count_words: NDArray[np.uint64]
    vaa: Optional[NDArray[np.float32]]
    color_self: Optional[NDArray[np.float64]]
    timings: Timings


def block_length(total: int, w_x: int, w_y: int, w_z: int) -> int:
    """``32 + 32 * ceil(total / 256) * (w_x + w_y + w_z)``, overflow-checked;
    a width above 16 raises ``limit.bit_width``."""


def gpu_layout_elements(rows: int, L: int) -> tuple[int, int]:
    """``(n_rows, n_rows * L * 3)`` with ``n_rows = ceil(rows / 32) * 32``:
    the int16 element count the GPU writer needs in ``out``."""


def decode_block(block: _Buffer, runs: NDArray, rows: int, L_heavy: int,
                 out: Optional[NDArray[np.int16]] = None) -> NDArray[np.int16]:
    """Decode to the padded heavy layout ``int16[rows, 3, L_heavy]`` (zeros past
    ``n_real``; FORMAT.md 4.3) -- ``reference.decode_block(block, n_real,
    L_heavy)`` bit for bit. ``out``, if given, needs ``rows * 3 * L_heavy``
    elements; the result is a view of its prefix."""


def decode_merged(block: _Buffer, runs: NDArray, rows: int, L_heavy: int,
                  L: int, n_types: int, features: Optional[FeatureSlice],
                  out: Optional[NDArray[np.int16]] = None
                  ) -> tuple[NDArray[np.int16], NDArray[np.uint8], Timings]:
    """The merged record ``int16[rows, 3, L]`` (FORMAT.md 8.4: real atoms,
    derived pseudo-atoms in stored feature order, zeros) and the count vectors
    ``uint8[rows, n_types + 1]`` (8.5). Equals ``reference.kernel_layout`` /
    ``reference.derived_counts`` per row."""


def write_gpu_block(block: _Buffer, runs: NDArray, rows: int, L_heavy: int,
                    L: int, n_types: int, features: Optional[FeatureSlice],
                    out: NDArray[np.int16], *, tile_mode: int = 1,
                    counts_out: Optional[NDArray[np.uint8]] = None,
                    count_words_out: Optional[NDArray[np.uint64]] = None,
                    self_overlaps: Optional[SelfOverlaps] = None,
                    vaa_out: Optional[NDArray[np.float32]] = None,
                    color_self_out: Optional[NDArray[np.float64]] = None
                    ) -> BlockResult:
    """Decode, derive color and write the tiled int16 device layout straight
    into ``out`` (the producer's pinned pool buffer; 1-D or any C-contiguous
    int16 array with at least ``gpu_layout_elements(rows, L)[1]`` elements).
    ``tile_mode`` 1 is ``[tile][atom][lane][xyz]`` (``pack_into`` default),
    -1 is ``[tile][atom][xyz][lane]`` (``pack_into(planar=True)``); padding
    lanes and slots are zeroed. Values are the stored int16 in grid units.
    The untiled layout is not offered (the CUDA kernel refuses it).
    Nothing large is allocated here; ``counts`` / ``count_words`` / ``vaa`` /
    ``color_self`` are allocated when their ``*_out`` is not given."""


def write_gpu_launch(blocks: list[tuple[_Buffer, NDArray, int, Optional[FeatureSlice]]],
                     rows: int, L_heavy: int, L: int, n_types: int,
                     out: NDArray[np.int16], *, tile_mode: int = 1,
                     counts_out: Optional[NDArray[np.uint8]] = None,
                     mol_ids_out: Optional[NDArray[np.int64]] = None
                     ) -> tuple[NDArray[np.int16], NDArray[np.uint8], NDArray[np.int64]]:
    """Consecutive whole blocks on one launch-relative tile grid.

    ``blocks``: ``(bounded block bytes, run slice, block rows, feature slice)``.
    Common widths and type count come from the validated group table. ``rows``
    is the sum of block rows (may exceed 4096); each block retains all existing
    codec bounds. Writes directly into ``out`` with only final launch padding,
    native per-row count and signed molecule-ID expansion. Returns coordinates,
    counts, IDs, all views of output prefixes. Checks all inputs/capacities and
    rejects input/output and output/output aliases before writing. Releases GIL.
    An error makes the entire launch unusable (no partial result is returned).
    """


def write_cpu_block(block: _Buffer, runs: NDArray, rows: int, L_heavy: int,
                    L: int, n_types: int, features: Optional[FeatureSlice],
                    coord_scale: float, *,
                    out: Optional[NDArray[np.float32]] = None,
                    counts_out: Optional[NDArray[np.uint8]] = None,
                    count_words_out: Optional[NDArray[np.uint64]] = None,
                    self_overlaps: Optional[SelfOverlaps] = None,
                    vaa_out: Optional[NDArray[np.float32]] = None,
                    color_self_out: Optional[NDArray[np.float64]] = None
                    ) -> BlockResult:
    """Decode, derive color and write ``float32[rows, L, 3]`` (atom-major,
    ``float32(v) * float32(coord_scale)``, zeros past ``n_real + n_color``)
    -- exactly what ``prepare_cpu_block_coords(merged, counts, L, coord_scale,
    merged_color=n_types > 0)`` returns for the merged record. ``coord_scale``
    is the corpus step (``2 ** format.step_log2``), finite and > 0."""


def encode_block(coords: NDArray[np.int16], runs: NDArray) -> bytes:
    """Encode ``int16[rows, 3, L_heavy]`` (padding ignored) for the run slice
    into the exact FORMAT.md 5.5 bytes -- ``reference.encode_block(coords,
    n_real_per_row)`` byte for byte."""


def read_into(fd: int, offset: int, length: int,
              out: Optional[NDArray[np.uint8]] = None) -> NDArray[np.uint8]:
    """Read exactly ``length`` bytes at ``offset`` of the open descriptor ``fd``
    into ``out`` (allocated if None) with bounded ``pread`` calls, GIL released.
    A file that ends first (e.g. truncated after it was opened) raises
    ``CodecError`` ``io.short_read``; any other failure ``io.read_error``;
    ``offset + length`` past a signed 64-bit offset ``decode.overflow``.
    FORMAT.md section 17 item 12 / Appendix B S7: block bytes handed to the
    decoders must come from bounded reads like this one, or from an mmap only
    where SIGBUS is converted into an error -- the decoders themselves do no
    I/O and cannot defend against a mapping that faults."""


def isa() -> str:
    """The unpack path in use: ``"scalar"``, ``"avx2"`` or ``"neon"``."""


def compiled_isas() -> tuple[str, ...]:
    """The unpack paths in this build (``"scalar"`` always)."""


def force_isa(name: str) -> None:
    """Force an unpack path (``"scalar"``, ``"avx2"``, ``"neon"``) or restore
    the default (``"auto"``), process-wide; for tests. The environment variable
    ``ROSHAMBO3_R3G_ISA`` sets the initial value (read once). Forcing a path
    this build does not contain raises ``CodecError`` (``decode.isa_unavailable``)."""


def _set_builder_mode(mode: str) -> None:
    """Private process-wide test/benchmark override (auto|baseline|fused).
    Do not mutate during concurrent calls. Auto resolves to baseline until
    paired measurements approve a default change."""


def _builder_info(consumer: str, coord_scale: float = 2.0 ** -8) -> dict[str, object]:
    """Normal nonalias dispatch: requested_mode, isa, implementation,
    scale_specialization, tile_lanes, consumer, coord_scale, alias_fallback.
    Aliased C++ inputs/outputs fall back to baseline; bindings reject aliases.
    """

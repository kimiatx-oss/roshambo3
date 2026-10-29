"""Validated wrapper over CPU shape search and scalar color modes.

**What this backend is, and is not.** It computes the Tier-3 specification: the
exact analytic Grant-Pickup pair sum in the reduced frame, at a pose found by
Adagrad on the exponential map, over the canonical start set, with field,
cutoff, SIMD and threading for shape. Color stays scalar and
separate from that shape dispatch so the shape trajectory cannot move.

**It is not bit-identical to the CUDA path and does not try to be.** Acceptance
is recall and top-end ranking; scores are not
bitwise portable even between two GPUs -- 0.12% of pairs identical after 100
Adagrad steps -- while the ranked output survived intact, so bitwise
comparison is not an acceptance criterion (``doc/NUMERICS_POLICY.md``
section 8). A CPU is one more architecture.

**Every knob it cannot honor is refused by name**, in the extension rather than
here, because that is where the list of things a CUDA launch means lives. An
ignored knob is a silent wrong answer.
"""

from __future__ import annotations

import hashlib
import json
import operator
import threading
from .._cpu_platform import scoped_qos, validate_qos
from functools import wraps
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .. import numerics

from ..layout import check_interaction_matrices, library_counts_u8
from .. import scores as _scores
from ..scores import N_SCORE_COLUMNS, check_self_overlap_bound
from .color_field_policy import (CPUColorFieldPolicy,
                                 ColorPlaneFieldCalibration)
from .self_overlap_bindings import (COLOR_SELF_ALGORITHM, COLOR_SELF_CONTRACT,
                                    COLOR_SELF_SCHEMA, VAA_SCHEMA,
                                    color_rows_sha256, shape_rows_sha256)

_ext = None


def _thread_count(value) -> int:
    """Require an actual positive integer worker count; do not truncate."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("threads must be an integer >= 1, not a boolean")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError("threads must be an integer >= 1") from exc
    if count < 1:
        raise ValueError("threads must be >= 1")
    return int(count)


@dataclass(frozen=True)
class CPUWorkerLocalTopKConfig:
    """Exact opt-in worker-local mode-3 top-K result configuration.

    The selected score column ranks candidates; complete 20-column score/pose
    rows and global IDs are returned per query. ``state_budget_bytes`` is a
    conservative planning estimate, not a strict allocator cap.
    """

    k: int
    rank_column: int = 0
    level: str = "conformer"
    state_budget_bytes: int = 256 << 20

    def __post_init__(self) -> None:
        for name, value, minimum in (
                ("k", self.k, 0),
                ("rank_column", self.rank_column, 0),
                ("state_budget_bytes", self.state_budget_bytes, 1)):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            try:
                integer = operator.index(value)
            except TypeError as exc:
                raise ValueError(
                    f"{name} must be an integer >= {minimum}") from exc
            if integer < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(integer))
        if self.rank_column >= N_SCORE_COLUMNS:
            raise ValueError("rank_column must be in [0, 19]")
        if self.level not in ("conformer", "molecule"):
            raise ValueError("level must be 'conformer' or 'molecule'")


def extension():
    """Import and return the CPU extension, or raise with a useful message."""
    global _ext
    if _ext is None:
        try:
            from .. import _roshambo3_cpu as mod
        except ImportError as e:                            # pragma: no cover
            raise ImportError(
                "the roshambo3 CPU kernel is not built. It is built "
                "unconditionally by CMake when the sources are present, so "
                "this means either an old wheel or a source tree without "
                f"src/roshambo3/backend/_cpu_src/. Original error: {e}") from e
        _ext = mod
    return _ext


class CPUComboExecutor:
    """Explicitly owned persistent native workers for mode-3 CPU scoring.

    A session has a fixed worker count and serializes submissions made from
    multiple caller threads. Use it as a context manager, and pass its matching
    worker count to :func:`score_pairs`; no process-global pool is created.
    """

    def __init__(self, workers: int, *, claim_chunk_rows: int = 256,
                 require_worker_affinity: bool = False, cpu_qos: str = "inherit"):
        validate_qos(cpu_qos)
        self._cpu_qos = cpu_qos
        worker_count = _thread_count(workers)
        if not isinstance(require_worker_affinity, (bool, np.bool_)):
            raise ValueError("require_worker_affinity must be a boolean")
        if isinstance(claim_chunk_rows, (bool, np.bool_)):
            raise ValueError("claim_chunk_rows must be an integer row count")
        try:
            chunk_rows = operator.index(claim_chunk_rows)
        except TypeError as exc:
            raise ValueError(
                "claim_chunk_rows must be an integer row count") from exc
        if chunk_rows < 128 or chunk_rows > 512 or chunk_rows % 16 != 0:
            raise ValueError(
                "claim_chunk_rows must be a multiple of 16 in [128, 512]")
        self._call_lock = threading.Lock()
        self._native = extension()._CPUComboExecutor(
            worker_count, int(chunk_rows), bool(require_worker_affinity), cpu_qos)

    @property
    def threads(self) -> int:
        return int(self._native.threads)

    @property
    def cpu_qos(self) -> str:
        """Requested scheduling policy, fixed for this executor's lifetime."""
        return self._cpu_qos

    @property
    def claim_chunk_rows(self) -> int:
        return int(self._native.claim_chunk_rows)

    @property
    def require_worker_affinity(self) -> bool:
        return bool(self._native.require_worker_affinity)

    @property
    def closed(self) -> bool:
        return bool(self._native.closed)

    @property
    def stats(self) -> dict:
        """Lifetime work and per-worker claim counts, including the last job."""
        with self._call_lock:
            return dict(self._native.stats)

    def close(self) -> None:
        """Wait for any active submission and idempotently join all workers."""
        # Lock acquisition may block; CPython releases the GIL while waiting.
        # This keeps close from owning the GIL while an active native call
        # returns from its GIL-free worker section.
        with self._call_lock:
            self._native.close()

    @scoped_qos
    def run_indexed(self, work_items: int, callback, *,
                    rows_per_item: int = 1, cpu_qos: str = "inherit") -> None:
        """Run one indexed job on these persistent native workers.

        ``callback(worker_id, item_id)`` runs on the owning native worker.
        The waiting caller releases the GIL; each worker holds it only while
        executing this Python callback. A callback may decode a block and then
        call a native scorer, but must not submit nested work to this executor.
        """
        if cpu_qos != "inherit" and cpu_qos != self.cpu_qos:
            raise ValueError("cpu_qos must match the executor policy")
        if not callable(callback):
            raise TypeError("callback must be callable")
        for name, value in (("work_items", work_items),
                            ("rows_per_item", rows_per_item)):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a non-negative integer")
            try:
                numeric = operator.index(value)
            except TypeError as exc:
                raise ValueError(
                    f"{name} must be a non-negative integer") from exc
            if numeric < (1 if name == "rows_per_item" else 0):
                raise ValueError(
                    f"{name} must be {'positive' if name == 'rows_per_item' else 'non-negative'}")
            if name == "work_items":
                item_count = numeric
            else:
                row_count = numeric
        with self._call_lock:
            self._native.run_python_items(
                callback, item_count, row_count)

    def __enter__(self) -> "CPUComboExecutor":
        if self.closed:
            raise RuntimeError("combo executor is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def _serialize_combo_executor_call(function):
    """Serialize all Python preparation and native stages for one session."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        executor = kwargs.get("executor")
        if not isinstance(executor, CPUComboExecutor):
            return function(*args, **kwargs)
        # Lock acquisition releases the GIL while waiting. Keep this outside
        # the native call so preparation, every native stage and output copy
        # share one session-level serialization interval.
        with executor._call_lock:
            return function(*args, **kwargs)
    return wrapped


_SELF_OVERLAP_CONTRACT = (
    "shared_fp32_reduced_v1|color_view_t154_ordered_types_1_to_n_minus_1|"
    "shape_ordered_fp32_reduced_v1|library_physical_angstrom_v1")
_VAA_SCHEMA = VAA_SCHEMA
_COLOR_SELF_SCHEMA = COLOR_SELF_SCHEMA
_COLOR_SELF_ALGORITHM = COLOR_SELF_ALGORITHM


def _fingerprint_arrays(domain: str, arrays) -> str:
    """Hash normalized array contents, shape and dtype, never object identity."""
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    for name, value in arrays:
        array = np.asarray(value)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _normalize_self_overlap_inputs(
        query_coords, query_types, query_n_atoms, library_coords,
        library_counts, interaction_r, interaction_p):
    """Match score_pairs' normalized wire and matrix representations."""
    from ..numerics import WIRE_QUANT_STEP

    qc = np.array(query_coords, dtype=np.float32, order="C", copy=True)
    qt = np.array(query_types, dtype=np.int32, order="C", copy=True)
    qn = np.array(query_n_atoms, dtype=np.int32, order="C", copy=True)
    raw_library = np.asarray(library_coords)
    if raw_library.dtype == np.int16:
        lc = (np.asarray(raw_library, dtype=np.float32) *
              np.float32(WIRE_QUANT_STEP))
        lc = np.ascontiguousarray(lc, dtype=np.float32)
    else:
        lc = np.array(raw_library, dtype=np.float32, order="C", copy=True)
    lcnt = library_counts_u8(library_counts).copy()
    r = np.array(interaction_r, dtype=np.float32, order="C", copy=True)
    p = np.array(interaction_p, dtype=np.float32, order="C", copy=True)
    if qc.ndim != 3 or qc.shape[2] not in (3, 4):
        raise ValueError("query_coords must be [n_q,L,3|4]")
    if qt.shape != qc.shape[:2] or qn.shape != (qc.shape[0],):
        raise ValueError("query types and atom counts must match query coordinates")
    if lc.ndim != 3 or lc.shape[2] not in (3, 4):
        raise ValueError("library_coords must be [n_library,L,3|4]")
    if lcnt.ndim != 2 or lcnt.shape[0] != lc.shape[0]:
        raise ValueError("library_counts must match library coordinate rows")
    if r.ndim != 2 or p.shape != r.shape or r.shape[0] != r.shape[1]:
        raise ValueError("interaction matrices must be matching square arrays")
    from ..layout import check_color_layout
    check_color_layout(
        r, p, np.asarray(query_types, dtype=np.int64), qn,
        int(qc.shape[1]), lcnt, int(lc.shape[1]),
        library_weight=(lc[:, :, 3] if lc.shape[2] == 4 else None))
    return qc, qt, qn, lc, lcnt, r, p


def _library_self_fingerprint(library_coords, library_counts,
                              interaction_r, interaction_p) -> str:
    return _fingerprint_arrays(
        _SELF_OVERLAP_CONTRACT + "|library-v1",
        (("library_coords", library_coords), ("library_counts", library_counts),
         ("interaction_r", interaction_r), ("interaction_p", interaction_p)))


def _shape_library_fingerprint(library_coords, library_counts) -> str:
    """Hash only real shape atoms, in the exact group-row sidecar order."""
    coords = np.asarray(library_coords)
    counts = np.asarray(library_counts)
    if coords.ndim != 3 or coords.shape[2] not in (3, 4):
        raise ValueError("library coordinates must be [n_rows,L,3|4]")
    if counts.ndim != 2 or counts.shape[0] != coords.shape[0] or counts.shape[1] < 1:
        raise ValueError("library counts must match coordinate rows")
    n_real = counts[:, 0].astype(np.int64)
    if np.any(n_real < 0) or np.any(n_real > coords.shape[1]):
        raise ValueError("library n_real is outside its coordinate row")
    # Vectorized, byte-identical to hashing row by row.
    return shape_rows_sha256(coords, n_real)


def _combo_self_fingerprint(query_coords, query_types, query_n_atoms,
                            library_coords, library_counts,
                            interaction_r, interaction_p) -> str:
    return _fingerprint_arrays(
        _SELF_OVERLAP_CONTRACT + "|query-library-v1",
        (("query_coords", query_coords), ("query_types", query_types),
         ("query_n_atoms", query_n_atoms),
         ("library_coords", library_coords), ("library_counts", library_counts),
         ("interaction_r", interaction_r), ("interaction_p", interaction_p)))


def _validate_stored_vaa_metadata(metadata, values, library_coords,
                                  library_counts) -> dict:
    """Every contract field of a stored-frame VAA record, against these rows.

    This is for caller-owned arrays. Format-3 corpus values enter through the
    authenticated native group decoder instead.
    """
    if not isinstance(metadata, dict):
        raise ValueError(
            "library_vaa requires explicit stored-frame metadata; legacy values are not verified")
    required = {
        "schema": _VAA_SCHEMA,
        "frame": "stored",
        "units": "angstrom^3",
        "coordinate_units": "angstrom",
        "wire_quantization_step_angstrom": float(numerics.WIRE_QUANT_STEP),
        "algorithm": "self_overlap_from_stored_quantized_xyz_v1",
        "row_order": "group_coordinate_row",
        "n_rows": int(library_coords.shape[0]),
        "shape_coordinates_sha256": _shape_library_fingerprint(
            library_coords, library_counts),
        "values_sha256": hashlib.sha256(
            np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")
        ).hexdigest(),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"library_vaa metadata {key!r} must be {expected!r}; "
                "legacy/prepared-frame values are not safe for combo normalization")
    if metadata.get("verified") is not True:
        raise ValueError("library_vaa metadata must declare verified=true")
    return dict(metadata)


def _array_digest(name: str, value) -> str:
    return _fingerprint_arrays("roshambo3.array.v1", ((name, value),))


def _validate_color_self_metadata(metadata, values, library_coords,
                                  library_counts, interaction_r,
                                  interaction_p) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("caller-supplied color self-overlaps require metadata")
    if "binding" in metadata:
        raise ValueError(
            "format-3 corpus color self values use authenticated native handles")
    vals = np.ascontiguousarray(values, dtype=np.float64)
    expected = {
        **COLOR_SELF_CONTRACT,
        "library_fingerprint": _library_self_fingerprint(
            library_coords, library_counts, interaction_r, interaction_p),
        "library_coords_counts_sha256": _fingerprint_arrays(
            "roshambo3.combo_library_rows.v1",
            (("library_coords", library_coords),
             ("library_counts", library_counts))),
        "interaction_r_sha256": _array_digest("interaction_r", interaction_r),
        "interaction_p_sha256": _array_digest("interaction_p", interaction_p),
        "values_sha256": _array_digest("color_self_f64", vals),
        "n_rows": int(library_coords.shape[0]),
        "n_features": int(interaction_r.shape[0]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"color self-overlap sidecar {key!r} does not match current inputs")
    return dict(metadata)


@dataclass(frozen=True)
class PreparedComboLibrarySelfOverlaps:
    """Caller-owned immutable denominator snapshot reusable across queries."""

    _native: object
    fingerprint: str
    values_fingerprint: str
    shape_values: np.ndarray
    color_values: np.ndarray
    n_features: int
    n_library: int
    shape_preparation_seconds: float
    color_preparation_seconds: float
    shape_terms: int
    color_terms: int
    shape_prepared: bool
    color_prepared: bool
    library_shape_from_stored_volume: bool
    library_color_from_sidecar: bool


@dataclass(frozen=True)
class PreparedComboSelfOverlaps:
    """Immutable query/library self-overlaps bound to exact normalized inputs."""

    _native: object
    input_fingerprint: str
    native_digest: str
    library_values_fingerprint: str
    n_queries: int
    n_library: int
    n_features: int
    shape_prepared: bool
    color_prepared: bool
    shape_preparation_seconds: float
    color_preparation_seconds: float
    library: PreparedComboLibrarySelfOverlaps


def prepare_combo_library_self_overlaps(
        *, library_coords: np.ndarray, library_counts: np.ndarray,
        interaction_r: np.ndarray, interaction_p: np.ndarray,
        library_vaa: Optional[np.ndarray] = None,
        library_vaa_metadata: Optional[dict] = None,
        library_color_self: Optional[np.ndarray] = None,
        library_color_metadata: Optional[dict] = None,
        shape: bool = True,
        color: bool = True
) -> PreparedComboLibrarySelfOverlaps:
    """Compute library self terms once for reuse across query sets.

    ``library_vaa`` is physical Å³ and is accepted only with explicit verified
    stored-coordinate metadata. Color values are matrix-specific; pass either
    a validated sidecar/value pair or let the native baseline compute them.

    """
    from ..numerics import assert_arithmetic_pinned

    assert_arithmetic_pinned()
    if not isinstance(shape, (bool, np.bool_)) or not isinstance(
            color, (bool, np.bool_)):
        raise TypeError("shape and color must be booleans")
    shape, color = bool(shape), bool(color)
    if not shape and not color:
        raise ValueError("at least one self-overlap channel must be prepared")
    if not shape and (library_vaa is not None or
                      library_vaa_metadata is not None):
        raise ValueError("shape sidecar values require shape=True")
    if not color and (library_color_self is not None or
                      library_color_metadata is not None):
        raise ValueError("color sidecar values require color=True")
    # The query arrays are empty by construction: this object owns library
    # values only, so computing it never repeats per-query self work.
    qc = np.zeros((0, 0, 3), dtype=np.float32)
    qt = np.zeros((0, 0), dtype=np.int32)
    qn = np.zeros((0,), dtype=np.int32)
    raw_library = np.asarray(library_coords)
    if raw_library.dtype == np.int16:
        from ..numerics import WIRE_QUANT_STEP
        lc = np.ascontiguousarray(
            np.asarray(raw_library, dtype=np.float32) * np.float32(WIRE_QUANT_STEP))
    else:
        lc = np.array(raw_library, dtype=np.float32, order="C", copy=True)
    lcnt = library_counts_u8(library_counts).copy()
    r = np.array(interaction_r, dtype=np.float32, order="C", copy=True)
    p = np.array(interaction_p, dtype=np.float32, order="C", copy=True)
    if lc.ndim != 3 or lc.shape[2] not in (3, 4):
        raise ValueError("library_coords must be [n_library,L,3|4]")
    if lcnt.ndim != 2 or lcnt.shape[0] != lc.shape[0]:
        raise ValueError("library_counts must match library coordinate rows")
    if r.ndim != 2 or p.shape != r.shape or r.shape[0] != r.shape[1]:
        raise ValueError("interaction matrices must be matching square arrays")
    if lcnt.shape[1] != r.shape[0]:
        raise ValueError("library count width must match interaction matrices")
    library_fingerprint = _library_self_fingerprint(lc, lcnt, r, p)
    vaa_values = None
    if library_vaa is not None:
        vaa_input = np.asarray(library_vaa, dtype=np.float64)
        if vaa_input.shape != (lc.shape[0],):
            raise ValueError("library_vaa must have one value per library row")
        if not np.all(np.isfinite(vaa_input)) or np.any(vaa_input < 0.0):
            raise ValueError("library_vaa must be finite and nonnegative")
        # Stored VAA sidecars are declared as little-endian float32.  Validate
        # and consume that exact representation: otherwise a float64 caller
        # value can hash to the certified float32 bytes while still changing
        # the native shape denominator.
        with np.errstate(over="ignore", invalid="ignore"):
            vaa_values = np.ascontiguousarray(
                vaa_input.astype(np.float32).astype(np.float64))
        if not np.all(np.isfinite(vaa_values)):
            raise ValueError("library_vaa must be representable as finite float32")
        _validate_stored_vaa_metadata(
            library_vaa_metadata, vaa_values, lc, lcnt)
    elif library_vaa_metadata is not None:
        raise ValueError("library_vaa_metadata requires library_vaa values")
    color_values = None
    if library_color_self is not None:
        color_values = np.ascontiguousarray(library_color_self, dtype=np.float64)
        _validate_color_self_metadata(
            library_color_metadata, color_values, lc, lcnt, r, p)
        if color_values.shape != (lc.shape[0],):
            raise ValueError("library_color_self must have one value per library row")
        if not np.all(np.isfinite(color_values)) or np.any(color_values < 0.0):
            raise ValueError("library_color_self must be finite and nonnegative")
    elif library_color_metadata is not None:
        raise ValueError("library_color_metadata requires library_color_self values")

    source_digest = _fingerprint_arrays(
        "roshambo3.combo_library_self_snapshot.v1",
        (("library_fingerprint", np.frombuffer(
            bytes.fromhex(library_fingerprint), dtype=np.uint8)),
         ("vaa", np.zeros((0,), np.float64) if vaa_values is None else vaa_values),
         ("color", np.zeros((0,), np.float64) if color_values is None else color_values)))
    native = extension().prepare_combo_self_overlaps(
        qc, qt, qn, lc, lcnt, r, p, source_digest,
        None, vaa_values, color_values, shape, color)
    shape_values = np.array(native.shape_library_values, dtype=np.float32,
                            order="C", copy=True)
    color_values_native = np.array(native.color_library_values, dtype=np.float64,
                                   order="C", copy=True)
    shape_values.flags.writeable = False
    color_values_native.flags.writeable = False
    values_fingerprint = _fingerprint_arrays(
        "roshambo3.combo_library_self_values.v1",
        (("shape_reduced_f32", shape_values),
         ("color_angstrom3_f64", color_values_native)))
    return PreparedComboLibrarySelfOverlaps(
        native, library_fingerprint, values_fingerprint,
        shape_values, color_values_native, int(r.shape[0]), int(lc.shape[0]),
        float(native.shape_preparation_seconds),
        float(native.color_preparation_seconds),
        int(native.shape_library_terms), int(native.color_library_terms),
        bool(native.shape_prepared), bool(native.color_prepared),
        bool(native.library_shape_from_stored_volume),
        bool(native.library_color_from_sidecar))


def prepare_combo_self_overlaps(
        *, query_coords: np.ndarray, query_types: np.ndarray,
        query_n_atoms: np.ndarray, library_coords: np.ndarray,
        library_counts: np.ndarray, interaction_r: np.ndarray,
        interaction_p: np.ndarray,
        prepared_library: Optional[PreparedComboLibrarySelfOverlaps] = None,
        library_vaa: Optional[np.ndarray] = None,
        library_vaa_metadata: Optional[dict] = None,
        library_color_self: Optional[np.ndarray] = None,
        library_color_metadata: Optional[dict] = None,
        shape: bool = True,
        color: bool = True
) -> PreparedComboSelfOverlaps:
    """Prepare query self terms and bind reusable library terms to exact inputs.

    Pair maps are deliberately absent: rows index query/library arrays and
    remain free to be sparse, duplicated, reversed or reordered at scoring.
    """
    from ..numerics import assert_arithmetic_pinned

    assert_arithmetic_pinned()
    if not isinstance(shape, (bool, np.bool_)) or not isinstance(
            color, (bool, np.bool_)):
        raise TypeError("shape and color must be booleans")
    shape, color = bool(shape), bool(color)
    if not shape and not color:
        raise ValueError("at least one self-overlap channel must be prepared")
    qc, qt, qn, lc, lcnt, r, p = _normalize_self_overlap_inputs(
        query_coords, query_types, query_n_atoms, library_coords,
        library_counts, interaction_r, interaction_p)
    if prepared_library is None:
        prepared_library = prepare_combo_library_self_overlaps(
            library_coords=lc, library_counts=lcnt,
            interaction_r=r, interaction_p=p,
            library_vaa=library_vaa,
            library_vaa_metadata=library_vaa_metadata,
            library_color_self=library_color_self,
            library_color_metadata=library_color_metadata,
            shape=shape, color=color)
    elif any(value is not None for value in
             (library_vaa, library_vaa_metadata, library_color_self,
              library_color_metadata)):
        raise ValueError(
            "sidecar values belong to prepared_library or a new library preparation, not both")
    expected_library = _library_self_fingerprint(lc, lcnt, r, p)
    if prepared_library.fingerprint != expected_library:
        raise ValueError(
            "prepared library self-overlaps do not match coordinates/counts/R/P")
    if (prepared_library.n_library != lc.shape[0] or
            prepared_library.n_features != r.shape[0]):
        raise ValueError("prepared library self-overlap row or feature count mismatch")
    if ((shape and not prepared_library.shape_prepared) or
            (color and not prepared_library.color_prepared)):
        raise ValueError(
            "prepared library self-overlaps do not contain every requested channel")
    current_values_fingerprint = _fingerprint_arrays(
        "roshambo3.combo_library_self_values.v1",
        (("shape_reduced_f32", prepared_library.shape_values),
         ("color_angstrom3_f64", prepared_library.color_values)))
    if current_values_fingerprint != prepared_library.values_fingerprint:
        raise ValueError("prepared library self-overlap values were mutated")
    input_fingerprint = _combo_self_fingerprint(
        qc, qt, qn, lc, lcnt, r, p)
    native_digest = _fingerprint_arrays(
        "roshambo3.combo_self_handle.v1",
        (("inputs", np.frombuffer(bytes.fromhex(input_fingerprint),
                                  dtype=np.uint8)),
         ("library_values", np.frombuffer(
             bytes.fromhex(prepared_library.values_fingerprint),
             dtype=np.uint8))))
    native = extension().prepare_combo_self_overlaps(
        qc, qt, qn, lc, lcnt, r, p, native_digest,
        prepared_library.shape_values if shape else None, None,
        prepared_library.color_values if color else None, shape, color)
    return PreparedComboSelfOverlaps(
        native, input_fingerprint, native_digest,
        prepared_library.values_fingerprint, int(qc.shape[0]), int(lc.shape[0]),
        int(r.shape[0]), bool(native.shape_prepared),
        bool(native.color_prepared), float(native.shape_preparation_seconds),
        float(native.color_preparation_seconds), prepared_library)


def write_combo_color_self_sidecar(path, *, values, library_coords,
                                   library_counts, interaction_r,
                                   interaction_p) -> dict:
    """Write a versioned, matrix-specific color sidecar without overwriting."""
    lc = np.asarray(library_coords)
    if lc.dtype == np.int16:
        from ..numerics import WIRE_QUANT_STEP
        lc = np.ascontiguousarray(lc.astype(np.float32) * WIRE_QUANT_STEP)
    else:
        lc = np.ascontiguousarray(lc, dtype=np.float32)
    counts = library_counts_u8(library_counts)
    r = np.ascontiguousarray(interaction_r, dtype=np.float32)
    p = np.ascontiguousarray(interaction_p, dtype=np.float32)
    vals = np.ascontiguousarray(values, dtype=np.float64)
    if vals.shape != (lc.shape[0],):
        raise ValueError("color self-overlap values must have one entry per library row")
    if not np.all(np.isfinite(vals)) or np.any(vals < 0.0):
        raise ValueError("color self-overlap values must be finite and nonnegative")
    metadata = {
        "schema": _COLOR_SELF_SCHEMA,
        "version": 1,
        "algorithm": _COLOR_SELF_ALGORITHM,
        "units": "angstrom^3",
        "coordinate_frame": "library_physical_angstrom_v1",
        "coordinate_precision": "float32_angstrom",
        "accumulator_precision": "float64",
        "feature_semantics": "color_view_t154_ordered_types_1_to_n_minus_1",
        "row_order": "library_coordinate_row",
        "library_fingerprint": _library_self_fingerprint(lc, counts, r, p),
        "library_coords_counts_sha256": _fingerprint_arrays(
            "roshambo3.combo_library_rows.v1",
            (("library_coords", lc), ("library_counts", counts))),
        "interaction_r_sha256": _array_digest("interaction_r", r),
        "interaction_p_sha256": _array_digest("interaction_p", p),
        "values_sha256": _array_digest("color_self_f64", vals),
        "n_rows": int(lc.shape[0]),
        "n_features": int(r.shape[0]),
    }
    target = Path(path)
    with target.open("xb") as stream:
        np.savez(stream, color_self=vals,
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    return metadata


def load_combo_color_self_sidecar(path, *, library_coords, library_counts,
                                  interaction_r, interaction_p):
    """Load and validate a versioned color sidecar against exact current inputs."""
    lc = np.asarray(library_coords)
    if lc.dtype == np.int16:
        from ..numerics import WIRE_QUANT_STEP
        lc = np.ascontiguousarray(lc.astype(np.float32) * WIRE_QUANT_STEP)
    else:
        lc = np.ascontiguousarray(lc, dtype=np.float32)
    counts = library_counts_u8(library_counts)
    r = np.ascontiguousarray(interaction_r, dtype=np.float32)
    p = np.ascontiguousarray(interaction_p, dtype=np.float32)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"color_self", "metadata"}:
            raise ValueError("unrecognized color self-overlap sidecar members")
        values = np.array(archive["color_self"], dtype=np.float64, order="C", copy=True)
        metadata = json.loads(str(archive["metadata"].item()))
    _validate_color_self_metadata(metadata, values, lc, counts, r, p)
    if values.shape != (lc.shape[0],):
        raise ValueError("color sidecar must contain one value per library row")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("color sidecar values must be finite and nonnegative")
    values.flags.writeable = False
    return values, metadata


def inert_launch_knobs() -> frozenset:
    """Launch knobs this backend refuses by name -- they describe CUDA hardware.

    Published so a caller can tell a knob it *chose* from one
    ``ScreenConfig.for_width`` emits unconditionally because that is what a
    CUDA launch takes. The first must be refused; the second was never a
    request. Refusing both is what made this backend unreachable through
    ``Engine``: the engine forwards the whole launch
    dict, so a screen died on ``smem_carveout`` before it scored a pair.

    The list lives in the extension, beside the refusal itself, so there is one
    of it.
    """
    return frozenset(extension().inert_knob_names())


#: Below this many library conformers in a launch, the field path loses to the
#: analytic conformer grain: the query field is built once per call by a serial
#: host loop, and a small launch cannot amortize it. Measured on 4 cores,
#: one evaluation group -- 0.26x at 128 conformers per query, 0.76x at
#: 512, 1.57x at 2,048, 2.29x at 32,768, `decision`-graded at four of
#: five points.
#: The crossover is bracketed between 512 and 2,048 and this is the conservative
#: end of that bracket, not a measured optimum.
#:
#: The underlying defect is that the spacing policy has no scale term, which
#: is what bites once this is wired into the Engine. This threshold is a floor under that gap, not a fix for it.
FIELD_PATH_MIN_LIBRARY = 2048

#: And below this many query atoms the field path also loses, for a different
#: and more fundamental reason: its advantage is deleting the `N_A` factor from
#: `O(N_A N_B)`, and at one query atom there is no factor to delete while the
#: grid build is still paid. The deciding measurement was **the crossover in
#: query atoms** -- 21.8 on the GPU against a panel mean of 22.6 -- and this
#: default originally ignored it, thresholding on library size alone.
#:
#: Measured on 4 cores, library fixed at 8,192: field/analytic is 0.54x
#: at 1 query atom, 0.83x at 4, 1.07x at 8, and 1.67-1.78x from 16 to 32, which
#: is where real panels sit. 8 is the first winning point measured, so it is
#: the threshold; the crossover is bracketed between 4 and 8.
FIELD_PATH_MIN_QUERY_ATOMS = 8


def _analytic_grain_knobs() -> dict:
    """The analytic configuration: conformer grain, widest available, poly.

    Used below the field path's crossover. `width=8` needs AVX2 and `width=16`
    needs AVX-512, and the extension refuses a width its binary does not carry,
    so the width is read from the build rather than assumed.
    """
    try:
        targets = set(extension().compiled_targets().split(","))
    except Exception:
        return {"grain": 2, "width": 1, "poly": True}
    width = 8 if "X86_64_V3" in targets else 1
    return {"grain": 2, "width": width, "poly": True}


def default_screen_knobs(n_library: int, spacing: float,
                         query_atoms: int = 1 << 30) -> dict:
    """The fast configuration, when this launch is big enough to want it.

    Published for the Engine to ask for, rather than applied inside
    :func:`score_pairs`, for a reason that is not style: **the extension's own
    defaults are the CPU reference anchor**, and `tests/test_anchor.py`
    reaches it through this seam. Moving the seam's defaults would move the
    stamped reference that both backends are compared against. So a caller who
    asks for nothing still gets the scalar exact path, and the fast
    configuration is something the Engine selects.

    This conservative helper does not select the public prepared-screen
    policy. That route defaults to fresh converged-objective selection and one
    accurate winner final; ``final_start_policy="all_start"`` retains accurate
    all-start selection. `library_vaa` is absent because the block does not carry
    the sidecar yet -- worth about 1.08x when it does.
    """
    grain = _analytic_grain_knobs()
    if (int(n_library) < FIELD_PATH_MIN_LIBRARY
            or int(query_atoms) < FIELD_PATH_MIN_QUERY_ATOMS):
        # NOT `{}`. Returning nothing let the extension fall through to its own
        # defaults -- `grain=-1, width=1`, which is the CPU
        # REFERENCE ANCHOR, not a screening path. Measured: 12.2k pairs/s
        # against the analytic conformer grain's 82.5k on the same panel, so
        # every small Engine screen ran at a seventh of the available rate,
        # silently. The crossover that set the threshold compared the field
        # path against the conformer grain, not against the anchor, so the
        # threshold was right and the fallback below it was not.
        return grain
    # The field sampler is implemented on the scalar path only (`grain=-1`),
    # which `interface.cpp` enforces, so the field configuration carries no
    # grain knobs.
    return {"field_h": float(spacing), "field_width": 8, "field_unroll": 0,
            "field_pipeline": True, "field_gridxf": 1, "field_allin": 1,
            "field_batched": True}


def available() -> bool:
    try:
        extension()
        return True
    except Exception:
        return False


def build_stamp() -> dict:
    """What a harness needs to grade a number from this backend.

    ``compiled_targets`` and ``selected_target`` are two different questions and
    both are asked. As with a stale GPU binary, asking the binary what it contains
    does not tell you what the process chose, and a build that silently ran a
    baseline scalar path looks like a slow kernel rather than a broken build.
    """
    ext = extension()
    return {
        "compiled_targets": [t for t in ext.compiled_targets().split(",") if t],
        "compiler": getattr(ext, "__compiler__", "unknown"),
        "selected_target": ext.selected_target(),
        # The ISA and the width, both, in every record. The width is
        # the SIMD layer's, which is what fixes the reduction order and so what
        # decides whether two CPU scores may be compared bitwise at all.
        "vector_width": int(ext.__vector_width__),
        # The anchor's identity is the CONTENT of its source, not the commit
        # it was checked out from: a binary can stamp one commit while
        # containing another's code. For a reference anchor that is fatal rather than merely untidy: a
        # fixture is only interpretable if it can name the exact source that
        # produced it.
        "cpu_src_sha256": str(ext.__cpu_src_sha256__),
        "cpu_src_tree": str(ext.__cpu_src_tree__),
        "cpu_src_dirty": int(ext.__cpu_src_dirty__),
        "cpu_src_dirty_files": [f for f in
                                str(ext.__cpu_src_dirty_files__).split(",") if f],
        "source_stamp_phase": str(ext.__source_stamp_phase__),
        "build_type": ext.__build_type__,
        "cxx_flags": ext.__cxx_flags__,
        "coord_scale": float(ext.__coord_scale__),
        "shape_prefactor": float(ext.__shape_prefactor__),
        # The stamp exposes the color ABI/feature-count limit and the
        # selected ISA. Selected ISA is already
        # above (`selected_target`); these two are the color half.
        "color_max_features": int(ext.__color_max_features__),
        "color_abi_version": int(ext.__color_abi_version__),
    }


def color_build_stamp(color_grain: int = 0) -> dict:
    """Return the loaded CPU and selected analytic-color arm provenance.

    The extension-wide dispatch target does not identify which color code ran.
    This stamp names that arm, its lane width and reduction policy,
    compilation status, and current-process ISA availability alongside the
    ordinary source/build identity a reference record requires. Grains other
    than 0, 11 and 16 are refused.
    """
    stamp = build_stamp()
    arm = dict(extension().color_arm_stamp(int(color_grain)))
    stamp.update(arm)
    return stamp


def combo_thread_config_stamp(*, threads: int, color_grain: int,
                              start_mode: int, n_pairs: int,
                              require_worker_affinity: bool = False,
                              claim_chunk_rows: int = 256) -> dict:
    """Describe work units, coarse claims, parallelism bound and affinity.

    ``workers_used`` is retained as the legacy one-shot pool-capacity estimate;
    it does not claim that every worker receives a coarse scheduling claim.
    Use ``worker_parallelism_limit`` for the maximum pool workers that can
    receive a claim and score rows.
    """
    if isinstance(claim_chunk_rows, (bool, np.bool_)):
        raise ValueError("claim_chunk_rows must be an integer row count")
    try:
        chunk_rows = operator.index(claim_chunk_rows)
    except TypeError as exc:
        raise ValueError(
            "claim_chunk_rows must be an integer row count") from exc
    if chunk_rows < 128 or chunk_rows > 512 or chunk_rows % 16 != 0:
        raise ValueError(
            "claim_chunk_rows must be a multiple of 16 in [128, 512]")
    return dict(extension().combo_thread_config_stamp(
        _thread_count(threads), int(color_grain), int(start_mode), int(n_pairs),
        bool(require_worker_affinity), int(chunk_rows)))


def build_color_view(*, query_coords: np.ndarray, query_types: np.ndarray,
                     query_n_atoms: np.ndarray, library_coords: np.ndarray,
                     library_counts: np.ndarray, n_features: int) -> dict:
    """The one read-only CPU color view, built by the C++ decoders.

    Thin pass-through to the extension's ``build_color_view`` -- not called
    by anything on the scoring path (color is still refused, see
    ``score_pairs`` below), and it exists for exactly one reason: so
    ``tests/test_color_view.py`` can check that the SAME segment-offset
    derivation this seam's future scalar, SIMD and field color paths will all
    share agrees, on the same inputs, with :func:`roshambo3.layout`'s Python
    decoders (:func:`roshambo3.layout.color_view_from_types` /
    :func:`color_view_from_counts`) -- the round-trip fixture.

    Returns a dict with ``"query"`` and ``"library"`` sub-dicts, each holding
    ``n_real``, ``n_color``, ``seg_offset``, ``seg_count`` and ``color_xyz``
    (the color coordinates the C++ decoder actually read, zero-padded to that
    side's own max ``n_color`` -- the padded-width histogram, kept rather
    than assumed constant).
    """
    return dict(extension().build_color_view(
        query_coords=np.ascontiguousarray(query_coords, dtype=np.float32),
        query_types=np.ascontiguousarray(query_types, dtype=np.int32),
        query_n_atoms=np.ascontiguousarray(query_n_atoms, dtype=np.int32),
        library_coords=np.ascontiguousarray(library_coords, dtype=np.float32),
        library_counts=library_counts_u8(library_counts),
        n_features=int(n_features)))


_SHAPE_COMBO_BACKENDS = {
    "scalar_analytic": 0,
    "analytic_simd": 1,
    "scalar_field": 2,
    "paired_channel_field": 3,
    "pipelined_field": 4,
}
_SHAPE_COMBO_GRAINS = {
    "scalar": 0,
    "query_atom": 1,
    "library_atom": 2,
    "trajectory": 3,
}


def _shape_combo_codes(shape_backend: str, shape_grain: str) -> tuple[int, int]:
    try:
        backend = _SHAPE_COMBO_BACKENDS[shape_backend]
    except KeyError as exc:
        raise ValueError(
            "shape_backend must be one of "
            f"{', '.join(_SHAPE_COMBO_BACKENDS)}") from exc
    try:
        grain = _SHAPE_COMBO_GRAINS[shape_grain]
    except KeyError as exc:
        raise ValueError(
            "shape_grain must be one of "
            f"{', '.join(_SHAPE_COMBO_GRAINS)}") from exc
    return backend, grain


def _shape_field_spacing(backend: int,
                         spacing: Optional[float]) -> float:
    """Require caller-resolved spacing for shape-field backends."""
    if backend in (2, 3, 4) and spacing is None:
        raise ValueError(
            "field_spacing_angstrom is required for a shape-field backend; "
            "resolve it from Group.L_field_width using "
            "engine.config.FIELD_SPACING_BY_WIDTH")
    return 0.5 if spacing is None else float(spacing)


def _shape_field_storage(storage: str) -> int:
    if storage == "fp32":
        return 0
    if storage == "fp16":
        return 1
    raise ValueError("field_storage must be 'fp32' or 'fp16'")


def _shape_field_interpolation(interpolation: str, storage: str) -> int:
    if interpolation == "fp32":
        return 0
    if interpolation == "mixed_half_yz":
        if storage != "fp16":
            raise ValueError(
                "mixed_half_yz interpolation requires field_storage='fp16'")
        return 1
    raise ValueError(
        "field_interpolation must be 'fp32' or 'mixed_half_yz'")


def prepare_combo_shape_evaluators(
        query_coords: np.ndarray, query_n_atoms: np.ndarray, *,
        shape_backend: str, shape_grain: str, shape_width: int = 1,
        field_spacing_angstrom: Optional[float] = None,
        field_padding_angstrom: float = 8.0,
        field_storage: str = "fp32",
        field_interpolation: str = "fp32"):
    """Build reusable query-scoped shape evaluators and query field grids.

    The handle owns reduced-coordinate snapshots and any query-prepared fields.
    Its build is outside :func:`score_pairs` so primary optimizer rates can
    exclude query field construction while still reporting it. For field
    backends, pass spacing resolved from `Group.L_field_width` through
    `engine.config.FIELD_SPACING_BY_WIDTH` (including its 0.8 A fallback):
    the low-level 0.5 A default is not an automatic width-policy selection.
    Field padding is the existing CPU shape-field halo, 8 A.
    The private ``mixed_half_yz`` experiment is AArch64-only and requires
    ``field_storage="fp16"``; query preparation selects FP32 fallback when
    native half arithmetic or the conservative finite-difference bound fails.
    Pipelined/library-atom width-eight handles also serve the legacy field
    shape route through ``score_pairs(prepared_query_fields=handle)``.
    """
    from ..numerics import assert_arithmetic_pinned

    assert_arithmetic_pinned()
    backend, grain = _shape_combo_codes(shape_backend, shape_grain)
    spacing = _shape_field_spacing(backend, field_spacing_angstrom)
    storage = _shape_field_storage(field_storage)
    interpolation = _shape_field_interpolation(field_interpolation,
                                               field_storage)
    qc = np.ascontiguousarray(query_coords, dtype=np.float32)
    qn = np.ascontiguousarray(query_n_atoms, dtype=np.int32)
    prepare = extension().prepare_combo_shape_evaluators
    args = (qc, qn, backend, grain, int(shape_width),
            spacing, float(field_padding_angstrom))
    # Keep the new FP32-default Python wrapper compatible with a frozen
    # mainline DSO that predates the optional storage argument. FP16 itself is
    # accepted only by a DSO whose binding implements the storage selector.
    if interpolation != 0:
        return prepare(*args, storage, interpolation)
    return prepare(*args) if storage == 0 else prepare(*args, storage)


def prepare_combo_shared_pose_queries(
        query_coords: np.ndarray, query_types: np.ndarray,
        query_n_atoms: np.ndarray, interaction_r: np.ndarray,
        interaction_p: np.ndarray, *, shape_backend: str,
        shape_grain: str, shape_width: int = 1,
        field_spacing_angstrom: Optional[float] = None,
        field_padding_angstrom: float = 8.0,
        shape_evaluators: Optional[object] = None,
        field_storage: str = "fp32",
        field_interpolation: str = "fp32"):
    """Own reduced-frame query colors and the matching shape handle.

    The returned object is immutable and reusable across calls/threads. It
    snapshots the query, type layout, shape configuration, and interaction
    matrix. Query colors are multiplied by ``COORD_SCALE`` once here; library
    coordinates stay physical and are posed by the shared FP32 ``M=sR,t'=s*t``
    bytes during scoring.
    """
    from ..numerics import assert_arithmetic_pinned

    assert_arithmetic_pinned()
    backend, grain = _shape_combo_codes(shape_backend, shape_grain)
    spacing = _shape_field_spacing(backend, field_spacing_angstrom)
    qc = np.ascontiguousarray(query_coords, dtype=np.float32)
    qt = np.ascontiguousarray(query_types, dtype=np.int32)
    qn = np.ascontiguousarray(query_n_atoms, dtype=np.int32)
    r = np.ascontiguousarray(interaction_r, dtype=np.float32)
    p = np.ascontiguousarray(interaction_p, dtype=np.float32)
    if qc.ndim != 3 or qc.shape[2] not in (3, 4):
        raise ValueError("query_coords must be [n_q,L,3|4]")
    if qt.shape != qc.shape[:2] or qn.shape != (qc.shape[0],):
        raise ValueError("query types and atom counts must match query coordinates")
    from ..layout import check_color_layout
    check_color_layout(r, p, np.asarray(qt, dtype=np.int64), qn,
                       int(qc.shape[1]), np.zeros((0, r.shape[0]), np.uint8), 0)
    if shape_evaluators is None:
        shape_evaluators = prepare_combo_shape_evaluators(
            qc, qn, shape_backend=shape_backend, shape_grain=shape_grain,
            shape_width=shape_width, field_spacing_angstrom=spacing,
            field_padding_angstrom=field_padding_angstrom,
            field_storage=field_storage,
            field_interpolation=field_interpolation)
    return extension().prepare_combo_shared_pose_queries(
        qc, qt, qn, r, p, shape_evaluators)


def combo_shape_config_stamp(*, shape_backend: str, shape_grain: str,
                             shape_width: int = 1,
                             field_spacing_angstrom: Optional[float] = None,
                             field_padding_angstrom: float = 8.0,
                             field_storage: str = "fp32",
                             field_interpolation: str = "fp32") -> dict:
    """Return the selected fixed-pose shape evaluator's truthful code stamp."""
    backend, grain = _shape_combo_codes(shape_backend, shape_grain)
    spacing = _shape_field_spacing(backend, field_spacing_angstrom)
    storage = _shape_field_storage(field_storage)
    interpolation = _shape_field_interpolation(field_interpolation,
                                               field_storage)
    stamp = extension().combo_shape_config_stamp
    args = (backend, grain, int(shape_width), spacing,
            float(field_padding_angstrom))
    if interpolation != 0:
        return dict(stamp(*args, storage, interpolation))
    return dict(stamp(*args) if storage == 0 else stamp(*args, storage))


def prepare_color_fields(
        *, query_coords: np.ndarray, query_types: np.ndarray,
        query_n_atoms: np.ndarray, interaction_r: np.ndarray,
        interaction_p: np.ndarray, backend_codes: np.ndarray):
    """Build a query-scoped color field handle outside optimizer calls.

    ``backend_codes`` may be a legacy one-dimensional vector broadcast to all
    query rows or a row-specific ``[n_query,n_features]`` matrix. The returned
    native handle owns selected fields for every query row. Pass
    it as ``prepared_color_fields`` to repeated mode-3 ``score_pairs`` calls
    with exactly the same query and interaction inputs. Construction and
    packing happen here; a mismatch is rejected rather than silently reusing
    the wrong query fields. Geometry is fixed at 0.5 A spacing and 4 A padding.
    """
    from ..numerics import assert_arithmetic_pinned

    assert_arithmetic_pinned()
    qc = np.ascontiguousarray(query_coords, dtype=np.float32)
    qt = np.ascontiguousarray(query_types, dtype=np.int32)
    qn = np.ascontiguousarray(query_n_atoms, dtype=np.int32)
    r = np.ascontiguousarray(interaction_r, dtype=np.float32)
    p = np.ascontiguousarray(interaction_p, dtype=np.float32)
    from .color_field_policy import normalize_backend_matrix
    if qc.ndim != 3 or qc.shape[2] not in (3, 4):
        raise ValueError("query_coords must be [n_q,L,3|4]")
    if qt.shape != qc.shape[:2] or qn.shape != (qc.shape[0],):
        raise ValueError("query types and atom counts must match query coordinates")
    nf = int(r.shape[0])
    codes = normalize_backend_matrix(
        backend_codes, int(qc.shape[0]), nf, require_selected=True)
    from ..layout import check_color_layout
    check_color_layout(
        r, p, np.asarray(qt, dtype=np.int64), qn, int(qc.shape[1]),
        np.zeros((0, nf), dtype=np.uint8), 0)
    return extension().prepare_color_fields(qc, qt, qn, r, p, codes, 0.5, 4.0)


@_serialize_combo_executor_call
@scoped_qos
def score_pairs(
        *,
        query_coords: np.ndarray,
        query_types: Optional[np.ndarray] = None,
        query_n_atoms: np.ndarray,
        library_coords: np.ndarray,
        library_counts: np.ndarray,
        pair_query_row: Optional[np.ndarray],
        pair_library_row: Optional[np.ndarray],
        interaction_r: Optional[np.ndarray] = None,
        interaction_p: Optional[np.ndarray] = None,
        steps: int = 25,
        lr_q: float = 0.1,
        lr_t: Optional[float] = None,
        start_mode: int = 1,
        budget_override: Optional[str] = None,
        color: bool = False,
        color_mode: Optional[str] = None,
        color_grain: int = 0,
        threads: int = 1,
        cpu_qos: str = "inherit",
        executor: Optional[CPUComboExecutor] = None,
        row_sink: Optional[object] = None,
        row_sink_only: bool = False,
        color_field_backends: Optional[np.ndarray] = None,
        prepared_color_fields: Optional[object] = None,
        shape_backend: str = "scalar_analytic",
        shape_grain: str = "scalar",
        shape_width: int = 1,
        shape_field_spacing_angstrom: Optional[float] = None,
        shape_field_padding_angstrom: float = 8.0,
        prepared_shape_evaluators: Optional[object] = None,
        prepared_query_fields: Optional[object] = None,
        pose_contract: Optional[str] = None,
        prepared_shared_pose_queries: Optional[object] = None,
        prepared_self_overlaps: Optional[Union[
            PreparedComboSelfOverlaps,
            PreparedComboLibrarySelfOverlaps]] = None,
        library_vaa: Optional[np.ndarray] = None,
        library_vaa_metadata: Optional[dict] = None,
        use_color_invariant_plan: bool = False,
        reuse_exact_final_scratch: bool = False,
        factor_canonical_torque: bool = False,
        optimizer_exp_mode: str = "baseline",
        canonical_color_kernel: bool = False,
        final_start_policy: Optional[str] = None,
        mixing: float = 0.0,
        out: Optional[np.ndarray] = None,
        topk=None,
        with_census: bool = False,
        with_phase_timers: bool = False,
        with_worker_timers: bool = False,
        library_self_timing: Optional[dict] = None,
        library_self_counts: Optional[dict] = None,
        require_worker_affinity: bool = False,
        worker_local_topk: Optional[CPUWorkerLocalTopKConfig] = None,
        library_conformer_ids: Optional[np.ndarray] = None,
        library_molecule_ids: Optional[np.ndarray] = None,
        check_stored_self: bool = True,
        **inert,
) -> Union[np.ndarray, list]:
    """Score ``[n_pairs, 20]`` on the CPU.

    ``library_self_timing`` is a caller-owned diagnostic dict for mode 3.
    It receives coarse shape/color library fallback elapsed spans and counts
    without changing the dense/sink return contract. It does not enable the
    per-pair phase timers; use a distinct dict for each concurrent call. These
    spans include fallback vector allocation and shape scaling, exclude query
    self work, and must not be summed as scan wall time across workers.

    ``lr_t`` may be omitted, in which case it is derived from ``steps`` --
    the same coupling ``optimizer.resolve_optimizer_settings``
    enforces: the step count and the translation learning rate are ONE knob
    through ``budget = lr_t * sum k^-0.5``, held at 4.0 A. A step-count change
    that does not carry its ``lr_t`` is not a speedup, it is a silent accuracy
    loss that looks exactly like one.

    ``color_field_backends`` is an experimental mode-3 override indexed by
    library target feature type, either broadcast from ``[n_features]`` or
    specified per query as ``[n_query,n_features]``: 0=analytic, 1=fp32 AoS4
    nodes, 2=fp16 AoS4 nodes, 3=fp32 polynomial bricks. Brick coefficients are packed from
    half-rounded AoS4 nodes, matching the reviewed primitive. It leaves
    the shipping default untouched, uses the fixed color field geometry
    (0.5 A spacing, 4 A padding), and must be paired with
    ``color_grain`` 11 or 16.
    Final reported scores are still the exact analytic rescore.
    ``prepared_color_fields`` reuses a query-scoped handle made by
    ``prepare_color_fields``; its build and pack work is outside this optimizer
    call and is reported separately by the handle.

    ``prepared_self_overlaps`` reuses immutable shape and/or color denominator
    terms made by ``prepare_combo_self_overlaps``; an unprepared channel keeps
    the reference recomputation path. Prepare selected matrix/library terms
    once with ``prepare_combo_library_self_overlaps`` and reuse that library
    handle across queries. Handles bind normalized coordinate/count/type and
    R/P contents, not array identity or pair maps; mutations are rejected. A
    physical ``library_vaa`` is accepted only with the matching group's
    verified stored-frame metadata (including coordinate/value hashes); a
    legacy or prepared-frame sidecar is never used for combo normalization.
    On the shape path a ``library_vaa`` is the stored value the scorer trusts,
    so every returned row is checked against the self-overlap bound and an
    inconsistent one raises ``InconsistentSelfOverlapError``. Pass
    ``check_stored_self=False`` only when a native top-K collector downstream
    checks the same rows (the engine's worker route does).

    ``shape_backend``/``shape_grain``/``shape_width`` explicitly select the
    mode-3 shape evaluator. Defaults retain scalar analytic shape. Supported
    names are ``scalar_analytic``, ``analytic_simd``, ``scalar_field``,
    ``paired_channel_field``, and ``pipelined_field``; grains are ``scalar``,
    ``query_atom``, ``library_atom``, or ``trajectory``. Use
    ``prepare_combo_shape_evaluators`` to build query-scoped field state once
    and pass it as ``prepared_shape_evaluators`` for repeated alignments.
    The same pipelined/library-atom width-eight handle can be passed as
    ``prepared_query_fields`` to the legacy FP32 field shape route
    (``field_h>0, field_width=8, field_unroll=0, field_pipeline=True``).
    Query coordinates/counts and field spacing/padding must match exactly;
    the grid is borrowed read-only, not copied or rebuilt per call.
    Field backends require an explicit spacing resolved from
    ``Group.L_field_width`` using ``engine.config.FIELD_SPACING_BY_WIDTH``;
    this API cannot infer that metadata from combined coordinates. Unspecified
    spacing remains 0.5 A only for non-field evaluators, where it is metadata
    and does not change arithmetic. Field padding defaults to the existing 8 A
    CPU shape-field halo.

    ``threads`` is honored by mode-3 combo scoring as independent native
    pair/batch workers. Each worker owns complete optimizer state; prepared
    query evaluators are shared read-only. The default remains one thread.
    Pass a context-managed ``CPUComboExecutor`` with a matching worker count to
    reuse persistent native workers across synchronous calls. Calls using one
    executor serialize. ``row_sink`` is a native-only observer called once per
    completed 20-column dense result row; its score pointer is borrowed only
    for the callback, and workers never invoke Python callbacks.
    ``row_sink_only=True`` is the opt-in block path: it requires a native sink,
    suppresses pair-dense output allocation, and returns ``None`` after all
    final rows have been offered. It cannot use a nested executor.
    Parallel combo calls reject census and phase-timer instrumentation because
    those counters are serial-only; use ``combo_thread_config_stamp`` to record
    the requested workers, work units, schedule and affinity plan.
    ``with_worker_timers=True`` is a separate parallel-only diagnostic that
    returns ``(scores, stats)`` with per-worker range elapsed times; it is never
    a rate call and cannot be combined with census or phase timers.
    ``require_worker_affinity=True`` makes a parallel combo call fail if a
    worker cannot be pinned to its requested logical CPU. This is intended for
    decision-grade topology experiments; the default remains advisory.

    ``worker_local_topk=CPUWorkerLocalTopKConfig(...)`` is a separate opt-in
    mode-3 path. It returns one tuple of bounded result arrays per query rather
    than a dense pair matrix; each tuple contains complete score/pose rows and
    canonical conformer IDs, plus molecule IDs when supplied. The native
    collector copies/normalizes the integer ID arrays into an owned immutable
    bundle after its checked memory-budget preflight. The default remains the
    existing dense result.
    """
    from ..numerics import assert_arithmetic_pinned
    from ..optimizer import resolve_optimizer_settings

    # The arithmetic the loaded .so was compiled with is a checked invariant,
    # not a convention, and it runs at every backend entry, this one
    # included. Cached after the first call, so it is free. check_build()
    # reads the CPU
    # extension's own flags, which is what makes calling it here mean
    # something rather than re-checking the CUDA extension.
    assert_arithmetic_pinned()

    if topk is not None:
        raise NotImplementedError(
            "the cpu backend has no device top-K to hand a handle to. Use "
            "engine.HostTopK, which advertises wants_dense=True and reduces on "
            "the host under numerics.rank_order.")
    if worker_local_topk is not None and not isinstance(
            worker_local_topk, CPUWorkerLocalTopKConfig):
        raise TypeError(
            "worker_local_topk must be a CPUWorkerLocalTopKConfig")

    # It takes a dict, not kwargs -- and it is the same call the engine makes,
    # so the coupling is enforced identically on both backends rather than
    # re-implemented here.
    settings = {"steps": int(steps), "lr_q": float(lr_q)}
    if lr_t is not None:
        settings["lr_t"] = float(lr_t)
    if budget_override is not None:
        # The coupling's escape hatch, and it is a STRING because the point is to say
        # why. An off-curve (steps, lr_t) pair is refused rather than warned
        # about, because the failure it catches -- cutting the step count and
        # leaving lr_t behind -- looks like a speedup and measures as an
        # accuracy loss. The only legitimate uses are a deliberately frozen
        # optimizer (isolating the volume sums from any trajectory) and
        # reproducing the grandfathered original pair.
        settings["budget_override"] = str(budget_override)
    opt = resolve_optimizer_settings(settings)

    qc = np.ascontiguousarray(query_coords, dtype=np.float32)
    lcoords = np.ascontiguousarray(library_coords, dtype=np.float32)
    if lcoords.dtype != np.float32:                         # pragma: no cover
        lcoords = lcoords.astype(np.float32)
    qn = np.ascontiguousarray(query_n_atoms, dtype=np.int32)
    lcnt = library_counts_u8(library_counts)
    if (pair_query_row is None) != (pair_library_row is None):
        raise ValueError(
            "pass both pair_query_row and pair_library_row, or neither for a "
            "dense launch")
    if pair_query_row is None:
        # A dense launch (every query against every library row,
        # query-major), the form the engine now passes. The native CPU scorer
        # takes an explicit pair list, so the list is written out here: the
        # CPU path's per-pair cost dwarfs it, and the native code and its
        # inputs stay exactly what they were.
        n_q_dense, n_lib_dense = int(qc.shape[0]), int(lcnt.shape[0])
        pq = np.repeat(np.arange(n_q_dense, dtype=np.int32), n_lib_dense)
        pt = np.tile(np.arange(n_lib_dense, dtype=np.int32), n_q_dense)
    else:
        pq = np.ascontiguousarray(pair_query_row, dtype=np.int32)
        pt = np.ascontiguousarray(pair_library_row, dtype=np.int32)

    # The int16 wire record is dequantized here rather than in the kernel. The
    # device does it at the point of use because that saves bus bytes; there is
    # no bus here, so doing it once on the way in is strictly fewer roundings.
    # Reduced coordinates use the wire step pinned by format.constants.COORD_SCALE
    # and numerics.WIRE_QUANT_STEP. It is distinct from the reduced frame's
    # approximately 0.77687365 scale; confusing the two scales every library
    # coordinate without another visible symptom.
    if np.asarray(library_coords).dtype == np.int16:
        from ..numerics import WIRE_QUANT_STEP
        lcoords = (np.asarray(library_coords, dtype=np.float32)
                   * np.float32(WIRE_QUANT_STEP))

    if qc.ndim != 3 or qc.shape[2] not in (3, 4):
        raise ValueError(f"query_coords must be [n_q, L, 3|4], got {qc.shape}")
    if lcoords.ndim != 3 or lcoords.shape[2] not in (3, 4):
        raise ValueError(
            f"library_coords must be [n_rows, L, 3|4], got {lcoords.shape}")
    if pq.shape != pt.shape:
        raise ValueError(
            "pair_query_row and pair_library_row must be the same length")
    if lcnt.ndim != 2:
        raise ValueError(
            "library_counts must be [n_library, N_features] uint8 with column "
            "0 the real atom count; a shape-only library is "
            "n_real.reshape(-1, 1)")
    if qn.shape[0] != qc.shape[0]:
        raise ValueError(
            f"query_n_atoms must be one count per query row: expected "
            f"({qc.shape[0]},), got {qn.shape}")
    # A count past the padded width would read the next row's atoms -- or past
    # the buffer, for the last row -- and score them as this conformer's. So
    # would a count row with no coordinate row behind it: pair rows are
    # bounded by the count rows. The CUDA wrapper refuses both.
    if lcnt.shape[0] > lcoords.shape[0]:
        raise ValueError(
            f"library_counts has {lcnt.shape[0]} rows but library_coords has "
            f"{lcoords.shape[0]}")
    if lcnt.shape[0] and int(lcnt[:, 0].max()) > lcoords.shape[1]:
        raise ValueError(
            f"library_counts[:, 0] reaches {int(lcnt[:, 0].max())} atoms but "
            f"library_coords rows hold {lcoords.shape[1]}")
    if qn.size and (int(qn.min()) < 0 or int(qn.max()) > qc.shape[1]):
        raise ValueError(
            f"query_n_atoms must lie in [0, {qc.shape[1]}], the query width; "
            f"got [{int(qn.min())}, {int(qn.max())}]")
    if pq.size:
        if int(pq.min()) < 0 or int(pq.max()) >= qc.shape[0]:
            raise ValueError("pair_query_row is out of range")
        if int(pt.min()) < 0 or int(pt.max()) >= lcnt.shape[0]:
            raise ValueError("pair_library_row is out of range")
    if interaction_r is not None and interaction_p is not None:
        # Shape-only, so there is no type plane to bound -- but the matrices
        # are still checked for shape, through the one definition in
        # `layout`, so a caller cannot learn a different contract here.
        check_interaction_matrices(interaction_r, interaction_p,
                                   np.zeros(0, np.int32))

    from ..reference.color import (MODE_SHAPE_ONLY,
                                   MODE_SHAPE_POSE_COLOR_RESCORE,
                                   MODE_COMBO_OPTIMAL)
    if color_mode is None:
        color_mode = (MODE_SHAPE_POSE_COLOR_RESCORE if color
                      else MODE_SHAPE_ONLY)
    if color_mode not in (MODE_SHAPE_ONLY, MODE_SHAPE_POSE_COLOR_RESCORE,
                          MODE_COMBO_OPTIMAL):
        raise NotImplementedError(
            f"CPU color mode {color_mode!r} is not implemented")
    if not 0.0 <= float(mixing) <= 1.0:
        raise ValueError("mixing must be in [0, 1]")
    thread_count = _thread_count(threads)
    if with_worker_timers and color_mode != MODE_COMBO_OPTIMAL:
        raise ValueError("worker timers are available only for mode-3 combo scoring")
    if with_worker_timers and thread_count < 2:
        raise ValueError("worker timers require threads >= 2")
    if require_worker_affinity and color_mode != MODE_COMBO_OPTIMAL:
        raise ValueError(
            "required worker affinity is available only for mode-3 combo scoring")
    if require_worker_affinity and thread_count < 2:
        raise ValueError("required worker affinity requires threads >= 2")
    if with_worker_timers and (with_census or with_phase_timers):
        raise ValueError(
            "worker timers are a separate diagnostic from census and phase timers")
    if executor is not None:
        if not isinstance(executor, CPUComboExecutor):
            raise TypeError("executor must be a CPUComboExecutor")
        if color_mode != MODE_COMBO_OPTIMAL:
            raise ValueError(
                "CPUComboExecutor is available only for mode-3 combo scoring")
        if cpu_qos != "inherit" and executor.cpu_qos != cpu_qos:
            raise ValueError("cpu_qos must match the executor policy")
        if executor.closed:
            raise RuntimeError("combo executor is closed")
        if executor.threads != thread_count:
            raise ValueError(
                "threads must match the CPUComboExecutor worker count")
        if require_worker_affinity and not executor.require_worker_affinity:
            raise ValueError(
                "executor must be created with required worker affinity")
    if row_sink is not None and color_mode != MODE_COMBO_OPTIMAL:
        raise ValueError("row_sink is available only for mode-3 combo scoring")
    if row_sink_only:
        if row_sink is None:
            raise ValueError("row_sink_only requires a native row_sink")
        if worker_local_topk is not None or out is not None:
            raise ValueError(
                "row_sink_only cannot be combined with worker_local_topk or out=")
        if executor is not None:
            raise ValueError(
                "row_sink_only is a block callback path and cannot nest an executor")
        if with_census or with_phase_timers or with_worker_timers:
            raise ValueError(
                "row_sink_only cannot be combined with score diagnostics")
    worker_topk_sink = None
    if worker_local_topk is not None:
        if color_mode != MODE_COMBO_OPTIMAL:
            raise ValueError(
                "worker_local_topk is available only for mode-3 combo scoring")
        if out is not None:
            raise ValueError(
                "worker_local_topk returns bounded per-query arrays and cannot "
                "be combined with out=")
        if row_sink is not None:
            raise ValueError(
                "worker_local_topk owns its native row sink; do not pass row_sink")
        if with_census or with_phase_timers or with_worker_timers:
            raise ValueError(
                "worker_local_topk sink-only output cannot be combined with "
                "census, phase timers, or worker timers")
        if library_conformer_ids is None:
            raise ValueError(
                "worker_local_topk requires library_conformer_ids")
        if not isinstance(library_conformer_ids, np.ndarray):
            raise TypeError("library_conformer_ids must be a NumPy integer array")
        if (library_conformer_ids.ndim != 1 or
                library_conformer_ids.shape[0] != lcnt.shape[0]):
            raise ValueError(
                "library_conformer_ids must have one ID per library row")
        if library_conformer_ids.dtype.kind not in "iu":
            raise TypeError(
                "library_conformer_ids must have a signed or unsigned integer dtype")
        if library_molecule_ids is not None:
            if not isinstance(library_molecule_ids, np.ndarray):
                raise TypeError(
                    "library_molecule_ids must be a NumPy integer array")
            if (library_molecule_ids.ndim != 1 or
                    library_molecule_ids.shape[0] != lcnt.shape[0]):
                raise ValueError(
                    "library_molecule_ids must have one ID per library row")
            if library_molecule_ids.dtype.kind not in "iu":
                raise TypeError(
                    "library_molecule_ids must have a signed or unsigned "
                    "integer dtype")
        if (worker_local_topk.level == "molecule" and
                library_molecule_ids is None):
            raise ValueError(
                "molecule worker_local_topk requires library_molecule_ids")

    elif library_conformer_ids is not None or library_molecule_ids is not None:
        raise ValueError(
            "library ID maps are accepted only with worker_local_topk")
    use_color = color_mode != MODE_SHAPE_ONLY
    shape_backend_code, shape_grain_code = _shape_combo_codes(
        shape_backend, shape_grain)
    effective_shape_field_spacing = _shape_field_spacing(
        shape_backend_code, shape_field_spacing_angstrom)
    shape_options_requested = (
        shape_backend_code != 0 or shape_grain_code != 0 or
        int(shape_width) != 1 or effective_shape_field_spacing != 0.5 or
        float(shape_field_padding_angstrom) != 8.0 or
        prepared_shape_evaluators is not None)
    if (shape_options_requested and
            color_mode != MODE_COMBO_OPTIMAL):
        raise ValueError(
            "combo shape evaluator options are available only in mode-3 "
            "combo scoring")
    if color and not use_color:
        raise ValueError("color=True requires a color scoring mode")
    if not use_color and float(mixing) != 0.0:
        raise ValueError("shape_only mode requires mixing=0")
    # Color grains: 0 is the unchanged scalar path; 11 runs eight independent
    # AVX2 conformer lanes; 16 maps each pair's four standard starts to
    # independent start lanes, two pairs per eight-lane block. Both vector
    # grains preserve independent optimizer/cache lanes and exact scalar
    # per-start and winner rescoring. Every other grain id is refused.
    # Color-field backend policy is separate and does not change these grains.
    if color_grain not in (0, 11, 16):
        raise ValueError(
            f"unsupported CPU color grain {color_grain}: color_grain must be "
            "0 (scalar), 11 (avx2 conformer8) or 16 (scheduled "
            "standard-start lanes)")
    if not use_color and color_grain:
        raise ValueError("color_grain requires a color scoring mode")
    field_backend_codes = None
    if color_field_backends is not None:
        if color_mode != MODE_COMBO_OPTIMAL:
            raise NotImplementedError(
                "color_field_backends is available only in mode-3 combo scoring")
        nf = int(np.asarray(interaction_r).shape[0]) if interaction_r is not None else 0
        from .color_field_policy import normalize_backend_matrix
        codes = normalize_backend_matrix(
            color_field_backends, int(qc.shape[0]), nf)
        if np.any(codes[:, 1:] != 0) and color_grain not in (11, 16):
            raise ValueError(
                "color_field_backends currently requires color_grain=11 or "
                "start-lane grain 16")
        field_backend_codes = codes
    if prepared_color_fields is not None and field_backend_codes is None:
        raise ValueError(
            "prepared_color_fields requires color_field_backends")
    if use_color:
        # Color validation still runs before any shape work or color
        # dispatch. A bare color=True is incomplete input, not a shape-only
        # fallback; the query type plane cannot be inferred from counts.
        if query_types is None:
            raise ValueError("color=True requires query_types")
        if interaction_r is None or interaction_p is None:
            raise ValueError("color=True requires interaction_r and interaction_p")
        from ..layout import check_color_layout
        qt = np.ascontiguousarray(query_types, dtype=np.int64)
        library_weight = (lcoords[:, :, 3] if lcoords.shape[2] == 4
                          else None)
        check_color_layout(interaction_r, interaction_p, qt, qn,
                           int(qc.shape[1]), lcnt, int(lcoords.shape[1]),
                           library_weight=library_weight)
        if (int(inert.get("grain", -1)) != -1 or
                int(inert.get("width", 1)) != 1 or
                float(inert.get("field_h", 0.0)) != 0.0):
            raise NotImplementedError(
                "CPU color rescore supports the scalar analytic path "
                "(grain=-1, width=1, field_h=0) only")

    if (library_self_timing is not None or library_self_counts is not None) and color_mode != MODE_COMBO_OPTIMAL:
        raise ValueError("library_self_timing requires mode-3 combo scoring")
    if color_mode == MODE_COMBO_OPTIMAL:
        if library_self_counts is not None and not isinstance(library_self_counts, dict):
            raise TypeError("library_self_counts must be a caller-owned dict")
        if library_self_timing is not None and not isinstance(library_self_timing, dict):
            raise TypeError("library_self_timing must be a caller-owned dict")
        # This separate scalar search does not pass through interface.cpp's
        # knob refusal, so explicitly reject every unsupported request.
        allowed = {"grain": -1, "width": 1, "field_h": 0.0,
                   "poly": False}
        for name, value in inert.items():
            if name not in allowed or value != allowed[name]:
                raise NotImplementedError(
                f"CPU combo optimizer does not support {name}={value!r}")

    if pose_contract is None:
        # Ordinary mode-3 calls use the reviewed shared FP32 pose. Other
        # modes retain their existing path unless a caller explicitly asks
        # for the mode-3-only shared contract.
        pose_contract = ("shared_fp32_reduced_v1"
                         if color_mode == MODE_COMBO_OPTIMAL
                         else "legacy_physical_fp64_pose")
    if pose_contract not in ("legacy_physical_fp64_pose",
                             "shared_fp32_reduced_v1"):
        raise ValueError("unknown combo pose_contract")
    shared_pose = pose_contract == "shared_fp32_reduced_v1"
    if canonical_color_kernel and (
            color_mode != MODE_COMBO_OPTIMAL or not shared_pose or color_grain != 16):
        raise ValueError("canonical_color_kernel requires shared-pose combo color_grain=16")
    if final_start_policy is not None and final_start_policy not in ("winner", "all_start"):
        raise ValueError("final_start_policy must be winner or all_start")
    native_final_start_policy = "default" if final_start_policy is None else final_start_policy
    if optimizer_exp_mode not in ("baseline", "rawbits", "rawbits_unclamped"):
        raise ValueError("optimizer_exp_mode must be baseline or rawbits (or rawbits_unclamped)")
    if optimizer_exp_mode != "baseline" and (
            color_mode != MODE_COMBO_OPTIMAL or not shared_pose or color_grain != 16):
        raise ValueError("rawbits requires shared-pose combo color_grain=16")
    if factor_canonical_torque and (
            color_mode != MODE_COMBO_OPTIMAL or not shared_pose or color_grain != 16):
        raise ValueError("factor_canonical_torque requires shared-pose combo color_grain=16")
    combo_qt32 = (np.ascontiguousarray(query_types, dtype=np.int32)
                  if query_types is not None else None)
    verified_vaa = None
    if library_vaa is not None:
        if color_mode == MODE_COMBO_OPTIMAL:
            raise ValueError(
                "mode-3 combo uses prepared_self_overlaps; raw library_vaa is "
                "accepted only with the shape or rescore route")
        if prepared_self_overlaps is not None:
            raise ValueError("pass library VAA through one prepared-self route")
        values = np.asarray(library_vaa)
        if values.dtype != np.dtype("float32") or values.shape != (lcoords.shape[0],):
            raise ValueError("library_vaa must be float32 with one value per row")
        _validate_stored_vaa_metadata(
            library_vaa_metadata, values, lcoords, lcnt)
        verified_vaa = np.ascontiguousarray(values, dtype=np.float32)
    elif library_vaa_metadata is not None:
        raise ValueError("library_vaa_metadata requires library_vaa values")
    prepared_library = None
    if prepared_self_overlaps is not None:
        if isinstance(prepared_self_overlaps, PreparedComboSelfOverlaps):
            if color_mode != MODE_COMBO_OPTIMAL or not shared_pose:
                raise ValueError(
                    "query/library prepared_self_overlaps requires mode-3 "
                    "shared_fp32_reduced_v1")
            if (not prepared_self_overlaps.shape_prepared and
                    not prepared_self_overlaps.color_prepared):
                raise ValueError(
                    "prepared_self_overlaps must contain at least one channel")
            if (prepared_self_overlaps._native.shape_prepared !=
                    prepared_self_overlaps.shape_prepared or
                    prepared_self_overlaps._native.color_prepared !=
                    prepared_self_overlaps.color_prepared):
                raise ValueError(
                    "prepared self-overlap channel metadata does not match its native handle")
            library = prepared_self_overlaps.library
            current_fingerprint = _combo_self_fingerprint(
                qc, combo_qt32, qn, lcoords, lcnt,
                np.ascontiguousarray(interaction_r, dtype=np.float32),
                np.ascontiguousarray(interaction_p, dtype=np.float32))
            if current_fingerprint != prepared_self_overlaps.input_fingerprint:
                raise ValueError(
                    "prepared self-overlaps are stale: normalized coordinates, "
                    "counts/types or interaction matrices changed")
            if (prepared_self_overlaps.n_queries != qc.shape[0] or
                    prepared_self_overlaps.n_library != lcoords.shape[0] or
                    prepared_self_overlaps.n_features != int(interaction_r.shape[0])):
                raise ValueError("prepared self-overlap handle dimensions do not match")
            prepared_library = library
        elif isinstance(prepared_self_overlaps, PreparedComboLibrarySelfOverlaps):
            if color_mode == MODE_COMBO_OPTIMAL:
                raise TypeError(
                    "mode-3 combo needs query/library PreparedComboSelfOverlaps")
            library = prepared_self_overlaps
            expected_library = _library_self_fingerprint(
                lcoords, lcnt,
                np.ascontiguousarray(interaction_r, dtype=np.float32),
                np.ascontiguousarray(interaction_p, dtype=np.float32))
            if library.fingerprint != expected_library:
                raise ValueError(
                    "prepared library self-overlaps do not match "
                    "coordinates/counts/R/P")
            if (library.n_library != lcoords.shape[0] or
                    library.n_features != int(interaction_r.shape[0])):
                raise ValueError("prepared library self-overlap row or feature count mismatch")
            if not library.shape_prepared:
                raise ValueError("prepared library self-overlaps lack shape values")
            if use_color and not library.color_prepared:
                raise ValueError("prepared library self-overlaps lack color values")
            if (library._native.shape_prepared != library.shape_prepared or
                    library._native.color_prepared != library.color_prepared or
                    library._native.library_shape_from_stored_volume !=
                    library.library_shape_from_stored_volume or
                    library._native.library_color_from_sidecar !=
                    library.library_color_from_sidecar):
                raise ValueError(
                    "prepared library self metadata does not match its native handle")
            prepared_library = library
        else:
            raise TypeError(
                "prepared_self_overlaps must come from a prepared combo self API")
        current_values_fingerprint = _fingerprint_arrays(
            "roshambo3.combo_library_self_values.v1",
            (("shape_reduced_f32", library.shape_values),
             ("color_angstrom3_f64", library.color_values)))
        expected_values_fingerprint = (
            prepared_self_overlaps.library_values_fingerprint
            if isinstance(prepared_self_overlaps, PreparedComboSelfOverlaps)
            else library.values_fingerprint)
        if (current_values_fingerprint != expected_values_fingerprint or
                library.values_fingerprint != expected_values_fingerprint):
            raise ValueError("prepared library self-overlap values were mutated")
    if worker_local_topk is not None:
        # The native factory performs the checked P*Q*K plus sidecar estimate
        # before it normalizes/copies the IDs or allocates collector shards.
        # Every collected row is checked against the self-overlap bound for its
        # verified stored values or values computed from these coordinates.
        prepared_stored = bool(
            prepared_library is not None and
            prepared_library.library_shape_from_stored_volume)
        if prepared_stored or verified_vaa is not None:
            self_route = _scores.ROUTE_VERIFIED_STORED
        else:
            self_route = _scores.ROUTE_RECOMPUTE
        worker_topk_sink = extension().make_worker_local_topk_row_sink(
            thread_count, int(qc.shape[0]), worker_local_topk.k,
            worker_local_topk.rank_column, worker_local_topk.level,
            int(lcnt.shape[0]), library_conformer_ids,
            library_molecule_ids,
            worker_local_topk.state_budget_bytes,
            self_overlap_bound_margin=_scores.self_overlap_bound_margin(
                self_route))

    if shared_pose:
        if color_mode != MODE_COMBO_OPTIMAL:
            raise NotImplementedError(
                "shared_fp32_reduced_v1 is available only for mode-3 combo scoring")
        if color_grain == 16 and int(start_mode) != 1:
            raise NotImplementedError(
                "shared color_grain 16 requires the standard four-start mode")
        if prepared_shared_pose_queries is None:
            # Preparation is query-owned and deliberately per-call unless the
            # caller supplies a reusable immutable handle. Reuse a caller's
            # shape evaluator set so field construction is not duplicated.
            prepared_shared_pose_queries = prepare_combo_shared_pose_queries(
                qc, np.ascontiguousarray(query_types, dtype=np.int32), qn,
                np.ascontiguousarray(interaction_r, dtype=np.float32),
                np.ascontiguousarray(interaction_p, dtype=np.float32),
                shape_backend=shape_backend, shape_grain=shape_grain,
                shape_width=int(shape_width),
                field_spacing_angstrom=effective_shape_field_spacing,
                field_padding_angstrom=float(shape_field_padding_angstrom),
                shape_evaluators=prepared_shape_evaluators)
    elif prepared_shared_pose_queries is not None:
        raise ValueError(
            "prepared_shared_pose_queries requires pose_contract="
            "'shared_fp32_reduced_v1'")

    kw = dict(
        query_coords=qc, query_n_atoms=qn,
        library_coords=lcoords, library_counts=lcnt,
        pair_query_row=pq, pair_library_row=pt,
        threads=thread_count,
        steps=int(opt["steps"]), lr_q=float(opt["lr_q"]),
        lr_t=float(opt["lr_t"]), start_mode=int(start_mode))
    kw["final_start_policy"] = native_final_start_policy
    if "prepared_shape_library_self_reduced" in inert:
        raise ValueError(
            "prepared_shape_library_self_reduced is an internal prepared-handle value")
    kw.update(inert)
    if verified_vaa is not None:
        kw["library_vaa"] = verified_vaa
    if prepared_library is not None:
        # This value has already been validated against the caller's exact
        # coordinates, counts and matrices above, and the format-3 source
        # constructs its native handle only from authenticated decoded rows.
        kw["prepared_shape_library_self_reduced"] = prepared_library.shape_values
    if prepared_query_fields is not None:
        if color_mode == MODE_COMBO_OPTIMAL or float(inert.get("field_h", 0.)) <= 0.:
            raise ValueError("prepared_query_fields requires the legacy field shape route")
        kw["prepared_query_fields"] = prepared_query_fields

    if sum((bool(with_census), bool(with_phase_timers),
            bool(with_worker_timers))) > 1:
        raise ValueError(
            "census, phase timers and worker timers are separate diagnostic "
            "entry points and cannot be combined")

    if color_mode == MODE_COMBO_OPTIMAL:
        native_self = prepared_self_overlaps
        try:
            scores, stats = extension().optimize_scalar_combo(
                qc, np.ascontiguousarray(query_types, dtype=np.int32), qn,
                lcoords, lcnt, pq, pt,
                np.ascontiguousarray(interaction_r, dtype=np.float32),
                np.ascontiguousarray(interaction_p, dtype=np.float32),
                int(opt["steps"]), np.float32(opt["lr_q"]),
                np.float32(opt["lr_t"]), int(start_mode), np.float32(mixing),
                with_census, with_phase_timers, color_grain,
                field_backend_codes, 0.5, 4.0, prepared_color_fields,
                shape_backend_code, shape_grain_code, int(shape_width),
                np.float32(effective_shape_field_spacing),
                np.float32(shape_field_padding_angstrom),
                prepared_shape_evaluators, thread_count,
                bool(with_worker_timers), bool(require_worker_affinity),
                prepared_shared_pose_queries, pose_contract,
                bool(use_color_invariant_plan),
                bool(reuse_exact_final_scratch),
                (native_self._native if native_self is not None else None),
                (native_self.native_digest if native_self is not None else ""),
                (executor._native if executor is not None else None),
                (worker_topk_sink if worker_topk_sink is not None else row_sink),
                worker_topk_sink is None and not row_sink_only,
                library_self_timing is not None, bool(factor_canonical_torque),
                optimizer_exp_mode, bool(canonical_color_kernel), native_final_start_policy)
        except BaseException:
            if worker_topk_sink is not None:
                worker_topk_sink.discard_after_failure()
            raise
        if library_self_timing is not None:
            library_self_timing.update({key: stats[key] for key in (
                "library_shape_self_seconds", "library_color_self_seconds",
                "library_shape_self_evaluations", "library_color_self_evaluations")})
        if library_self_counts is not None:
            library_self_counts.update({key: stats[key] for key in (
                "library_shape_self_evaluations", "library_color_self_evaluations")})
        if worker_topk_sink is not None:
            worker_topk_sink.mark_scoring_succeeded()
            return worker_topk_sink.finalize()
        if row_sink_only:
            return None
        if with_census or with_phase_timers or with_worker_timers:
            return scores, dict(stats)
        if out is not None:
            out[:scores.shape[0], :N_SCORE_COLUMNS] = scores
            return out
        return scores

    def refuse_inconsistent_stored_self(shape_scores):
        # Every row is in hand here, so verified stored values that do not
        # describe their coordinates are refused wherever those rows rank.
        prepared_stored = (
            prepared_library is not None and
            prepared_library.library_shape_from_stored_volume)
        if (verified_vaa is not None or prepared_stored) and check_stored_self:
            check_self_overlap_bound(shape_scores, pt, id_label="library row",
                                     route=_scores.ROUTE_VERIFIED_STORED)

    def color_rescore(shape_scores, stats=None):
        if not use_color:
            return (shape_scores, stats) if stats is not None else shape_scores
        rescored, color_stats = extension().rescore_scalar_color(
            qc, np.ascontiguousarray(query_types, dtype=np.int32), qn,
            lcoords, lcnt, pq, pt,
            np.ascontiguousarray(interaction_r, dtype=np.float32),
            np.ascontiguousarray(interaction_p, dtype=np.float32),
            shape_scores, np.float32(mixing), with_census,
            with_phase_timers, False, color_grain, None,
            (prepared_library.color_values
             if prepared_library is not None and prepared_library.color_prepared
             else None))
        if stats is not None:
            stats = dict(stats)
            for name, value in color_stats.items():
                if with_census and not name.startswith("color_"):
                    continue
                if with_phase_timers and name.startswith("color_"):
                    continue
                if name in stats and name in ("query_prep", "query_self_overlap",
                                              "library_self_overlap"):
                    stats[name] += value
                else:
                    stats[name] = value
            return rescored, stats
        return rescored

    if with_census:
        # A separate entry point, and it returns a TUPLE. The counters cost
        # something, so a rate taken with them on is a rate of the
        # instrumented kernel; and a conditional return type on the seam's own
        # function would break the seam, which the engine calls at one site
        # expecting an array. `out=` is not honored here -- a census pass is
        # not a screening pass.
        scores, census = extension().score_pairs_with_census(**kw)
        refuse_inconsistent_stored_self(scores)
        return color_rescore(scores, dict(census))

    if with_phase_timers:
        # Same shape as `with_census` above and for the
        # same reason: pricing structure is not the shipping path, and
        # "an instrumented run is never a rate run" applies here exactly as
        # it does to the census -- a wall-clock measurement taken with these
        # timers on prices the PHASES, and must never be quoted as this
        # backend's rate.
        scores, timers = extension().score_pairs_with_phase_timers(**kw)
        refuse_inconsistent_stored_self(scores)
        return color_rescore(scores, dict(timers))

    scores = extension().score_pairs(**kw)
    refuse_inconsistent_stored_self(scores)
    scores = color_rescore(scores)
    if out is not None:
        out[:scores.shape[0], :N_SCORE_COLUMNS] = scores
        return out
    return scores

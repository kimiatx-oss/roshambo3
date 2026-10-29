"""Content bindings for caller-owned CPU self-overlap snapshots.

These byte contracts cover in-memory arrays passed to the public CPU scoring
helpers. Corpus groups use format-3's authenticated native reader path instead
of this metadata interface.
"""

from __future__ import annotations

import hashlib

import numpy as np

VAA_SCHEMA = "roshambo3.vaa.v1"
SHAPE_ROWS_DOMAIN = b"roshambo3.vaa.shape_rows.v1\0"
COLOR_SELF_SCHEMA = "roshambo3.combo_color_self.v1"
COLOR_SELF_ALGORITHM = "native_scalar_analytic_self_overlap_v1"
COLOR_ROWS_DOMAIN = b"roshambo3.color_self.rows.v1\0"

COLOR_SELF_CONTRACT = {
    "schema": COLOR_SELF_SCHEMA,
    "version": 1,
    "algorithm": COLOR_SELF_ALGORITHM,
    "units": "angstrom^3",
    "coordinate_frame": "library_physical_angstrom_v1",
    "coordinate_precision": "float32_angstrom",
    "accumulator_precision": "float64",
    "feature_semantics": "color_view_t154_ordered_types_1_to_n_minus_1",
    "row_order": "library_coordinate_row",
}


def _interleaved(header: np.ndarray, coords: np.ndarray,
                 n_slots: np.ndarray) -> np.ndarray:
    """Build the original per-row header + real-coordinate digest stream."""
    header = np.ascontiguousarray(header, dtype=np.uint8)
    n = header.shape[0]
    xyz = np.ascontiguousarray(coords[:, :, :3], dtype="<f4")
    n_slots = np.asarray(n_slots, dtype=np.int64)
    if xyz.shape[0] != n or n_slots.shape != (n,):
        raise ValueError("row header, coordinates and slot counts disagree")
    if n == 0:
        return np.zeros(0, np.uint8)
    if np.any(n_slots < 0) or np.any(n_slots > xyz.shape[1]):
        raise ValueError("a row's real-slot count is outside its coordinate row")
    header_bytes = header.shape[1]
    real = np.arange(xyz.shape[1])[None, :] < n_slots[:, None]
    data = xyz[real].view(np.uint8).reshape(-1)
    record = header_bytes + 12 * n_slots
    starts = np.concatenate(([0], np.cumsum(record)[:-1]))
    out = np.empty(int(record.sum()), np.uint8)
    is_data = np.ones(out.size, bool)
    for column in range(header_bytes):
        out[starts + column] = header[:, column]
        is_data[starts + column] = False
    out[is_data] = data
    return out


def update_shape_rows(digest, coords: np.ndarray, n_real: np.ndarray) -> None:
    """Feed float32 Angstrom rows into a padding-independent shape digest."""
    n_real = np.asarray(n_real, dtype=np.int64)
    header = np.ascontiguousarray(n_real.astype("<u2")).view(np.uint8).reshape(-1, 2)
    digest.update(_interleaved(header, coords, n_real))


def update_color_rows(digest, coords: np.ndarray, counts: np.ndarray) -> None:
    """Feed full count rows and their real merged xyz slots into a digest."""
    counts = np.ascontiguousarray(counts, dtype=np.uint8)
    if counts.ndim != 2 or counts.shape[1] < 1:
        raise ValueError("color rows need the [n, n_features] count vector")
    digest.update(_interleaved(counts, coords, counts.astype(np.int64).sum(axis=1)))


def shape_rows_sha256(coords, n_real) -> str:
    digest = hashlib.sha256(SHAPE_ROWS_DOMAIN)
    update_shape_rows(digest, coords, n_real)
    return digest.hexdigest()


def color_rows_sha256(coords, counts) -> str:
    digest = hashlib.sha256(COLOR_ROWS_DOMAIN)
    update_color_rows(digest, coords, counts)
    return digest.hexdigest()

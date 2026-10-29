"""Static, query-scoped CPU color-plane field selection.

This module only plans a fixed backend matrix before optimizer work begins.
It never samples poses, times the active scan, or changes a choice mid-run.
An empty calibration table is deliberately analytic-only: the integrated
field cells did not establish a positive finite-reuse crossover, and the
primitive brackets are not a grain-16 production cutoff.
Policy rows and timing qualification are FP32-only (backend codes 1
and 3); legacy explicit FP16 code 2 remains outside that qualification.
For the retained grain-16 conformer route, native bounds preflight is per
target-type/pair-lane: if any posed target sample is out of the selected
query-field grid, that lane's whole target-type plane uses the existing
analytic SIMD path. The exposed out-of-grid sample/term counts are aggregate,
not plane/lane-resolved. This bounded fallback is partial scope and does not
promote fields by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import re
from typing import Any

import numpy as np


COLOR_FIELD_ANALYTIC = 0
COLOR_FIELD_NODE_FP32 = 1
COLOR_FIELD_BRICK_FP32 = 3
MAX_CHANNEL_VALUES_PER_QUERY = 32_000_000


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


@dataclass(frozen=True)
class ColorPlaneFieldCalibration:
    """One measured representation row for a fixed color evaluator.

    Timing units are nanoseconds. ``analytic_ns_per_interaction`` prices one
    query-feature/target-feature evaluation; ``field_ns_per_target_feature``
    prices one interpolated target feature per color evaluation. Build terms
    are measured separately as a fixed cost, per generated grid node, and per
    contributing query source. ``first_winning_query_features`` is the first
    integer query-plane count that passed the recorded paired timing rule;
    the cost equation below is an additional finite-screen guard, not a
    replacement for that measured bracket.

    Rows must describe the same production grain, field geometry, target
    distribution, ISA and correctness-qualified stored-gradient primitive
    used by the intended dispatch. ``source_sha256`` and ``calibration_id``
    make that provenance explicit. The runtime refuses to infer these values.
    """

    backend_code: int
    first_winning_query_features: int
    analytic_ns_per_interaction: float
    field_ns_per_target_feature: float
    build_fixed_ns: float
    build_ns_per_node: float
    build_ns_per_source: float
    reference_library_conformers: int
    max_plane_bytes: int
    calibration_id: str
    source_sha256: str
    measurement_scope: str

    def __post_init__(self) -> None:
        code = _integer("backend_code", self.backend_code, 1)
        if code not in (COLOR_FIELD_NODE_FP32, COLOR_FIELD_BRICK_FP32):
            raise ValueError("CPU calibration backend_code must be 1 or 3")
        object.__setattr__(self, "backend_code", code)
        for name, minimum in (
                ("first_winning_query_features", 1),
                ("reference_library_conformers", 1),
                ("max_plane_bytes", 1)):
            object.__setattr__(self, name,
                               _integer(name, getattr(self, name), minimum))
        for name in ("analytic_ns_per_interaction",
                     "field_ns_per_target_feature", "build_fixed_ns",
                     "build_ns_per_node", "build_ns_per_source"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.analytic_ns_per_interaction <= 0.0 or \
                self.field_ns_per_target_feature <= 0.0:
            raise ValueError("measured alignment costs must be positive")
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("calibration_id must identify the measurement")
        digest = str(self.source_sha256).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("source_sha256 must be a 64-character SHA-256")
        object.__setattr__(self, "source_sha256", digest)
        if not isinstance(self.measurement_scope, str) or not self.measurement_scope.strip():
            raise ValueError("measurement_scope must describe evaluator/workload")

    def cache_key(self) -> tuple:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class CPUColorFieldPolicy:
    """Immutable threshold table plus explicit retained/build byte ceilings.

    The defaults select no fields. To opt in, callers must supply measured
    calibration rows and an explicit color-field storage ceiling. Selection
    also respects the enclosing Engine's joint memory preflight, the measured
    per-plane working-set ceiling, and the native builder's 32M channel-value
    per-query guard.

    ``evaluations_per_conformer`` is the number of optimizer color-gradient
    evaluations paid by each conformer under the calibration's schedule. It
    must be supplied with a nonempty table; for public grain 16 the caller can
    use four starts times the configured optimizer steps, matching the actual
    loop. This is a static cost-model input, not an online counter.
    """

    calibrations: tuple[ColorPlaneFieldCalibration, ...] = ()
    evaluations_per_conformer: int = 0
    max_retained_bytes: int = 0
    max_build_peak_bytes: int = 0

    def __post_init__(self) -> None:
        rows = tuple(self.calibrations)
        if any(not isinstance(row, ColorPlaneFieldCalibration) for row in rows):
            raise TypeError("calibrations must contain ColorPlaneFieldCalibration rows")
        if len({row.backend_code for row in rows}) != len(rows):
            raise ValueError("at most one calibration row per backend is supported")
        object.__setattr__(self, "calibrations", rows)
        object.__setattr__(self, "evaluations_per_conformer",
                           _integer("evaluations_per_conformer",
                                    self.evaluations_per_conformer, 0))
        object.__setattr__(self, "max_retained_bytes",
                           _integer("max_retained_bytes",
                                    self.max_retained_bytes, 0))
        object.__setattr__(self, "max_build_peak_bytes",
                           _integer("max_build_peak_bytes",
                                    self.max_build_peak_bytes, 0))
        if rows:
            if self.evaluations_per_conformer < 1:
                raise ValueError(
                    "a calibrated policy requires evaluations_per_conformer")
            if self.max_retained_bytes < 1 or self.max_build_peak_bytes < 1:
                raise ValueError(
                    "a calibrated policy requires explicit retained and build byte ceilings")

    def cache_key(self) -> tuple:
        return (self.evaluations_per_conformer, self.max_retained_bytes,
                self.max_build_peak_bytes,
                tuple(row.cache_key() for row in self.calibrations))

    def select(self, *, query_coords, query_types, query_n_atoms,
               interaction_r, interaction_p, library_target_counts,
               library_conformers: int, spacing_angstrom: float = 0.5,
               padding_angstrom: float = 4.0,
               available_build_bytes: int | None = None):
        """Return immutable ``[query,target_type]`` codes and plan metadata.

        Only exact diagonal interaction matrices qualify. General matrices,
        inactive planes, empty source planes, unsupported/oversized geometries,
        non-amortizing planes and planes over any byte ceiling remain analytic.
        Candidate planes are allocated by estimated saved nanoseconds per
        retained byte; ties are deterministic by query row, target type and
        backend code. The field storage estimate includes the conservative
        geometry, final payload and selected-builder temporary peak.
        """
        qc = np.asarray(query_coords)
        qt = np.asarray(query_types)
        qn = np.asarray(query_n_atoms)
        r = np.asarray(interaction_r)
        p = np.asarray(interaction_p)
        target_counts = np.asarray(library_target_counts, dtype=np.float64)
        if qc.ndim != 3 or qc.shape[2] not in (3, 4):
            raise ValueError("query_coords must have shape [n_query,L,3|4]")
        if qt.shape != qc.shape[:2] or qn.shape != (qc.shape[0],):
            raise ValueError("query types/atom counts must match query coordinates")
        if r.ndim != 2 or r.shape[0] != r.shape[1] or p.shape != r.shape:
            raise ValueError("interaction matrices must be matching square arrays")
        nf = int(r.shape[0])
        if qt.size and (int(np.min(qt)) < 0 or int(np.max(qt)) >= nf):
            raise ValueError("query type is outside the interaction matrix")
        if target_counts.shape != (nf,) or not np.isfinite(target_counts).all() or \
                np.any(target_counts < 0):
            raise ValueError("library_target_counts must be a finite nonnegative [n_features] vector")
        total = _integer("library_conformers", library_conformers, 0)
        spacing = float(spacing_angstrom)
        padding = float(padding_angstrom)
        if not math.isfinite(spacing) or spacing <= 0.0 or \
                not math.isfinite(padding) or padding < 0.0:
            raise ValueError("field spacing must be positive and padding nonnegative")
        available = self.max_build_peak_bytes
        if available_build_bytes is not None:
            available = min(available, _integer(
                "available_build_bytes", available_build_bytes, 0))

        codes = np.zeros((qc.shape[0], nf), dtype=np.int32)
        info = {
            "policy": "static_color_plane_v1",
            "calibration_ids": tuple(row.calibration_id
                                      for row in self.calibrations),
            "calibration_source_sha256": tuple(row.source_sha256
                                                for row in self.calibrations),
            "measurement_scope": tuple(row.measurement_scope
                                        for row in self.calibrations),
            "general_matrix_analytic_fallback": False,
            "selected_planes": [],
            "field_bytes_estimated": 0,
            "field_build_peak_bytes_estimated": 0,
            "estimated_alignment_savings_ns": 0.0,
            "non_amortizing_planes": 0,
            "byte_guarded_planes": 0,
        }
        if not self.calibrations:
            info["selection_reason"] = "no_calibrated_rows_analytic_default"
            codes.flags.writeable = False
            return codes, info
        if total == 0:
            info["selection_reason"] = "empty_library_analytic"
            codes.flags.writeable = False
            return codes, info

        # The policy is qualified only for the repository's canonical active color
        # matrix: R == 1 and P == identity on feature types 1..nf-1. Type 0 is
        # the reserved "no feature" slot, so its row/column do not participate
        # (canonical generators conventionally set P[0,0] to zero). Use exact
        # value comparison: a near-canonical/general matrix stays analytic.
        active_r = r[1:, 1:]
        active_p = p[1:, 1:]
        canonical_active_r = np.ones_like(active_r)
        canonical_active_p = np.eye(nf - 1, dtype=active_p.dtype)
        if not np.array_equal(active_r, canonical_active_r) or \
                not np.array_equal(active_p, canonical_active_p):
            info["general_matrix_analytic_fallback"] = True
            info["selection_reason"] = "general_matrix_analytic"
            codes.flags.writeable = False
            return codes, info

        from ..layout import counts_from_types
        query_counts = counts_from_types(
            qt, qn, nf, require_contiguous_from_n_real=True)
        candidates = []
        for query_row in range(qc.shape[0]):
            n_real = int(qn[query_row])
            if n_real < 0 or n_real > qc.shape[1]:
                raise ValueError("query atom count is outside its coordinate row")
            type_plane = np.asarray(qt[query_row, n_real:], dtype=np.int64)
            for target_type in range(1, nf):
                m = int(query_counts[query_row, target_type])
                if m <= 0 or float(r[target_type, target_type]) <= 0.0 or \
                        float(p[target_type, target_type]) == 0.0:
                    continue
                ell = float(target_counts[target_type])
                if ell <= 0.0:
                    continue
                source_xyz = np.asarray(
                    qc[query_row, n_real:, :3][type_plane == target_type],
                    dtype=np.float64)
                if source_xyz.shape[0] != m or not np.isfinite(source_xyz).all():
                    raise ValueError("selected query color sources must be finite")
                geometry = _plane_geometry(source_xyz, spacing, padding)
                if geometry is None:
                    continue
                cells, nodes = geometry
                query_candidates = []
                for row in self.calibrations:
                    if m < row.first_winning_query_features:
                        continue
                    retained, temporary = _representation_bytes(
                        row.backend_code, cells, nodes)
                    if retained > row.max_plane_bytes:
                        info["byte_guarded_planes"] += 1
                        continue
                    channel_values = nodes * 4
                    delta_ns = (m * row.analytic_ns_per_interaction -
                                row.field_ns_per_target_feature)
                    if delta_ns <= 0.0:
                        info["non_amortizing_planes"] += 1
                        continue
                    build_ns = (row.build_fixed_ns + nodes * row.build_ns_per_node +
                                m * row.build_ns_per_source)
                    alignment_saving_ns = (
                        total * self.evaluations_per_conformer * ell * delta_ns)
                    saving_ns = alignment_saving_ns - build_ns
                    if saving_ns <= 0.0:
                        info["non_amortizing_planes"] += 1
                        continue
                    query_candidates.append({
                        "query_row": query_row,
                        "target_type": target_type,
                        "backend_code": row.backend_code,
                        "query_features": m,
                        "field_bytes": retained,
                        "build_temporary_bytes": temporary,
                        "channel_values": channel_values,
                        "estimated_build_ns": build_ns,
                        "estimated_savings_ns": saving_ns,
                        "calibration_id": row.calibration_id,
                    })
                if query_candidates:
                    # Compare calibrated total benefit, then storage. Backend
                    # code is the final deterministic tie-breaker.
                    chosen = max(query_candidates, key=lambda item: (
                        item["estimated_savings_ns"],
                        -item["field_bytes"], -item["backend_code"]))
                    candidates.append(chosen)

        candidates.sort(key=lambda item: (
            -(item["estimated_savings_ns"] / item["field_bytes"]),
            item["query_row"], item["target_type"], item["backend_code"]))
        accepted = []
        query_retained = [0] * int(qc.shape[0])
        query_channels = [0] * int(qc.shape[0])
        query_temp = [0] * int(qc.shape[0])
        for item in candidates:
            qrow = item["query_row"]
            t = item["target_type"]
            retained = item["field_bytes"]
            temp = item["build_temporary_bytes"]
            # Only one representation can own one (query,target) plane.
            if codes[qrow, t] != COLOR_FIELD_ANALYTIC:
                continue
            next_total = sum(query_retained) + retained
            next_temp = max(query_temp[qrow], temp)
            peak = next_total + max(max(query_temp, default=0), temp)
            native_metadata = _native_handle_payload_bytes(
                qc, qt, qn, r, p, query_counts)
            retained_with_metadata = next_total + native_metadata
            peak += native_metadata
            if (retained_with_metadata > self.max_retained_bytes or
                    peak > available or
                    query_channels[qrow] + item["channel_values"] >
                    MAX_CHANNEL_VALUES_PER_QUERY):
                info["byte_guarded_planes"] += 1
                continue
            codes[qrow, t] = item["backend_code"]
            query_retained[qrow] += retained
            query_temp[qrow] = next_temp
            query_channels[qrow] += item["channel_values"]
            accepted.append(item)

        field_payload_total = sum(query_retained)
        native_metadata = _native_handle_payload_bytes(
            qc, qt, qn, r, p, query_counts) if accepted else 0
        retained_total = field_payload_total + native_metadata
        peak_total = retained_total + max(query_temp, default=0)
        info["selected_planes"] = accepted
        info["field_bytes_estimated"] = field_payload_total
        info["native_retained_payload_bytes_estimated"] = native_metadata
        info["retained_bytes_estimated"] = retained_total
        info["field_build_peak_bytes_estimated"] = peak_total
        info["estimated_alignment_savings_ns"] = float(sum(
            item["estimated_savings_ns"] for item in accepted))
        info["selection_reason"] = (
            "calibrated_static_selection" if accepted else
            "no_plane_passed_count_cost_and_byte_guards")
        codes.flags.writeable = False
        return codes, info


def normalize_backend_matrix(backend_codes, n_queries: int, n_features: int,
                             *, require_selected: bool = False) -> np.ndarray:
    """Validate 1-D broadcast or immutable per-query backend code matrix."""
    codes = np.asarray(backend_codes)
    if codes.ndim == 1:
        if codes.shape != (n_features,):
            raise ValueError(
                "backend_codes needs one code per target type with shape "
                "[n_features]")
        codes = np.broadcast_to(codes, (n_queries, n_features))
    elif codes.ndim != 2 or codes.shape != (n_queries, n_features):
        raise ValueError(
            "backend_codes must be [n_features] or [n_query,n_features]")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("backend_codes must contain integer codes")
    if np.any((codes < 0) | (codes > 3)):
        raise ValueError("codes must be 0, 1, 2 or 3")
    # Validate before narrowing so wide values cannot wrap to a known code.
    result = np.ascontiguousarray(codes, dtype=np.int32)
    if n_features and np.any(result[:, 0] != 0):
        raise ValueError("target type zero must use backend code 0")
    if require_selected and not np.any(result[:, 1:] != 0):
        raise ValueError("at least one target plane must select a field backend")
    result.flags.writeable = False
    return result


def estimate_color_field_storage(backend_codes, query_coords, query_types,
                                 query_n_atoms, interaction_r,
                                 interaction_p, *, spacing_angstrom=0.5,
                                 padding_angstrom=4.0) -> dict:
    """Conservative field payload/build scratch for a normalized code matrix."""
    codes = normalize_backend_matrix(
        backend_codes, int(np.shape(query_coords)[0]),
        int(np.shape(interaction_r)[0]))
    # Reuse the policy geometry/count validation with a zero-work table; the
    # helper below is shared so explicit benchmark arms get the same estimate.
    qc = np.asarray(query_coords)
    qt = np.asarray(query_types)
    qn = np.asarray(query_n_atoms)
    r = np.asarray(interaction_r)
    p = np.asarray(interaction_p)
    from ..layout import counts_from_types
    counts = counts_from_types(qt, qn, r.shape[0],
                               require_contiguous_from_n_real=True)
    selected = []
    rows_retained = []
    rows_temp = []
    rows_channels = []
    spacing, padding = float(spacing_angstrom), float(padding_angstrom)
    for qrow in range(qc.shape[0]):
        retained = temp = channel_values = 0
        n_real = int(qn[qrow])
        types = np.asarray(qt[qrow, n_real:], dtype=np.int64)
        for target_type in range(1, r.shape[0]):
            code = int(codes[qrow, target_type])
            if code == 0:
                continue
            m = int(counts[qrow, target_type])
            source_xyz = np.asarray(
                qc[qrow, n_real:, :3][types == target_type], dtype=np.float64)
            if source_xyz.shape[0] != m or not np.isfinite(source_xyz).all():
                raise ValueError("selected query color sources must be finite")
            geometry = _plane_geometry(source_xyz, spacing, padding)
            if geometry is None:
                continue
            cells, nodes = geometry
            payload, temporary = _representation_bytes(code, cells, nodes)
            retained += payload
            temp = max(temp, temporary)
            channel_values += 4 * nodes
            selected.append({"query_row": qrow, "target_type": target_type,
                             "backend_code": code, "field_bytes": payload,
                             "nodes": nodes, "cells": cells})
        rows_retained.append(retained)
        rows_temp.append(temp)
        rows_channels.append(channel_values)
    retained_total = sum(rows_retained)
    native_metadata = _native_handle_payload_bytes(qc, qt, qn, r, p, counts)
    retained_with_metadata = retained_total + (native_metadata if selected else 0)
    return {
        "selected_planes": selected,
        "field_bytes_estimated": retained_total,
        "native_retained_payload_bytes_estimated": (
            native_metadata if selected else 0),
        "retained_bytes_estimated": retained_with_metadata,
        "field_build_peak_bytes_estimated": retained_with_metadata +
            max(rows_temp, default=0),
        "channel_values_by_query": rows_channels,
        "native_channel_limit_exceeded": any(
            count > MAX_CHANNEL_VALUES_PER_QUERY for count in rows_channels),
    }


def _native_handle_payload_bytes(qc, qt, qn, r, p, counts) -> int:
    """Payload copied/owned by the C++ handle and per-query bundles."""
    n_queries = int(np.shape(qc)[0])
    nf = int(np.shape(r)[0])
    n_sources = int(np.sum(np.asarray(counts)[:, 1:], dtype=np.uint64))
    # Query/R/P snapshots, row codes, per-query double coefficient/decay
    # matrices, ColorFieldSource records, shared_ptr slots, and a conservative
    # object/vector header allowance. This is owned payload, not RSS.
    return int(
        np.asarray(qc).nbytes + np.asarray(qt).nbytes + np.asarray(qn).nbytes +
        np.asarray(r).nbytes + np.asarray(p).nbytes +
        n_queries * nf * 4 + nf * 4 +
        n_queries * nf * nf * 2 * np.dtype(np.float64).itemsize +
        n_sources * 16 + n_queries * 1024)


def _plane_geometry(source_xyz: np.ndarray, spacing: float,
                    padding: float):
    if source_xyz.shape[0] == 0:
        return None
    spans = np.ptp(source_xyz, axis=0)
    extents = (spans + 2.0 * padding) / spacing
    if not np.isfinite(extents).all() or \
            np.any(extents >= np.iinfo(np.int64).max - 2):
        return None
    # One extra cell on every axis covers FP32 scale/ceil boundary drift in
    # the native COORD_SCALE geometry and intentionally errs high.
    cell_axes = np.maximum(1, np.ceil(extents).astype(np.int64) + 1)
    node_axes = cell_axes + 1
    cells = math.prod(int(value) for value in cell_axes)
    nodes = math.prod(int(value) for value in node_axes)
    return cells, nodes


def _representation_bytes(code: int, cells: int, nodes: int):
    if code == COLOR_FIELD_NODE_FP32:
        payload = nodes * 4 * np.dtype(np.float32).itemsize
        temporary_extra = 0
    elif code == 2:  # legacy explicit API arm; never emitted by policy rows
        payload = nodes * 4 * np.dtype(np.uint16).itemsize
        temporary_extra = nodes * 4 * np.dtype(np.float32).itemsize
    elif code == COLOR_FIELD_BRICK_FP32:
        payload = cells * 32 * np.dtype(np.float32).itemsize
        # While packing, both raw fp32 nodes and half-rounded nodes coexist
        # with the final fp32 polynomial bricks.
        temporary_extra = nodes * 4 * (
            np.dtype(np.float32).itemsize + np.dtype(np.uint16).itemsize)
    else:
        raise ValueError("unknown CPU color field backend code")
    return payload, temporary_extra

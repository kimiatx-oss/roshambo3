"""Format-3 physical-block CPU descriptors and verified self bindings.

Directory entries retain block metadata only. Authenticated in-memory digests
bind bounded per-block reads to the format's whole-section hashes; no full row
sidecar or expanded molecule-ID array lives on the coordinator.
"""
from dataclasses import dataclass
import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np

from ..format.r3g import native
from ..format.r3g.errors import CorpusValidationError
from ..format.r3g.fileio import open_corpus_file
from ..format.r3g.limits import DTYPES, RUN_DTYPE, group_file_name


def _batched_fingerprint_arrays(domain, arrays):
    """Public SHA-256 byte stream, one hash call per bounded block snapshot.

    Joining avoids repeated large-array HASH.update GIL release/reacquire
    cycles. The temporary is bounded by one block plus its query metadata;
    Engine's worker peak estimate includes it. No digest/check is omitted.
    """
    parts = [domain.encode("utf-8"), b"\0"]
    for name, value in arrays:
        array = np.asarray(value)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        parts.extend((name.encode("utf-8"), b"\0", array.dtype.str.encode("ascii"),
            b"\0", json.dumps(array.shape, separators=(",", ":")).encode(),
            b"\0", memoryview(array).cast("B")))
    return hashlib.sha256(b"".join(parts)).hexdigest()


def _refuse(group, block, code, message):
    return CorpusValidationError(code, f"group {group.index} block {block}: {message}",
                                 tier="deep", path=group.path)


@dataclass(frozen=True)
class CPUPhysicalBlockDescriptor:
    """Authenticated metadata for one format-3 CPU scan block."""

    group_index: int
    block_index: int
    group_row0: int
    compact_row0: int
    gid_base: int
    n_conf: int
    shape_record: tuple[int, int, int, int]
    shape_reader: object
    library_width: int
    physical_shape_width: int
    field_width: int
    coord_scale: float


@dataclass(frozen=True)
class CPUGroupMetadata:
    index: int
    path: str
    n_conf: int
    n_types: int
    L: int
    L_heavy: int
    L_field_width: int
    sections: dict
    file_name: str
    stat_key: tuple

    def close(self):
        """Metadata holds no open file or mapped payload."""


def _read_section_rows(group, name, first, count):
    section = group._og.sections[name]
    dtype = RUN_DTYPE if name == "runs" else DTYPES[section["dtype"]]
    data = group._og.file.read(section["offset"] + first * dtype.itemsize,
                               count * dtype.itemsize, tier="deep")
    return np.frombuffer(data, dtype=dtype)


def _authenticate_section(group, name, ranges, bindings, *, overlap=0):
    digest = hashlib.sha256()
    for bi, (first, count) in enumerate(ranges):
        values = _read_section_rows(group, name, first, count)
        digest.update(values[overlap if bi else 0:])
        bindings[bi][name] = (first, count, hashlib.sha256(values).digest())
        if name in ("feature_offsets", "feature_atom_offsets"):
            yield int(values[0]), int(values[-1])
    if digest.hexdigest() != group._og.sections[name]["sha256"]:
        raise _refuse(group, f"0..{group.n_blocks - 1}", "integrity.section_hash",
                      f"{name} section hash mismatch")


def _read_bound(group, bi, name, binding):
    first, count, expected = binding[name]
    values = _read_section_rows(group, name, first, count)
    if hashlib.sha256(values).digest() != expected:
        raise _refuse(group, bi, "integrity.section_hash",
                      f"{name} block slice changed since section authentication")
    return values


def build_cpu_block_descriptors(source):
    if source._cpu_descriptors is not None:
        return source._cpu_descriptors
    groups, bindings_by_key, totals = {}, {}, {}
    descriptors = []
    offsets = np.concatenate(([0], np.cumsum(
        [int(g["n_conf"]) for g in source.corpus.group_table], dtype=np.int64)))
    compact_offsets = {}
    compact = 0
    for gi in source.group_indices:
        compact_offsets[gi] = compact
        compact += int(source.corpus.group_table[gi]["n_conf"])
    for gi in source.group_indices:
        with source.corpus.group(gi) as group:
            metadata = CPUGroupMetadata(gi, group.path, group.n_conf, group.n_types,
                group.L, group.L_heavy, group.L_field_width, dict(group._og.sections),
                group_file_name(group.bucket, group.group), group._og.file.stat_key)
            groups[gi] = metadata
            blocks = group.blocks
            bindings = [{} for _ in blocks]
            run_ranges = [(int(b["m0"]), int(b["m1"] - b["m0"])) for b in blocks]
            row_ranges = [(int(b["first_row"]), int(b["rows"])) for b in blocks]
            # Every yielded digest is derived from the same bytes fed into the
            # whole-section hash. Exhaust the generator before trusting any.
            list(_authenticate_section(group, "runs", run_ranges, bindings))
            for name in ("vaa", "color_self"):
                if name in group._og.sections and (name == "vaa" or source.color):
                    list(_authenticate_section(group, name, row_ranges, bindings))
            if source.color:
                feature_ranges = list(_authenticate_section(group, "feature_offsets",
                    [(m, n + 1) for m, n in run_ranges], bindings, overlap=1))
                list(_authenticate_section(group, "feature_types",
                    [(f0, f1 - f0) for f0, f1 in feature_ranges], bindings))
                atom_ranges = list(_authenticate_section(group, "feature_atom_offsets",
                    [(f0, f1 - f0 + 1) for f0, f1 in feature_ranges], bindings, overlap=1))
                list(_authenticate_section(group, "feature_atoms",
                    [(a0, a1 - a0) for a0, a1 in atom_ranges], bindings))
            for bi, block in enumerate(blocks):
                rows, row0 = int(block["rows"]), int(block["first_row"])
                runs = _read_bound(group, bi, "runs", bindings[bi])
                if int(runs["n_conf"].sum(dtype=np.uint64)) != rows:
                    raise _refuse(group, bi, "layout.blocks", "a molecule spans blocks")
                if np.any(runs["mol_id"] >= (1 << 63)):
                    raise _refuse(group, bi, "limit.mol_id", "mol_id must be in [0, 2^63)")
                # The scan-certified run IDs are globally unique. A block's
                # complete run range plus its row-count equality above proves
                # that none of those molecules crosses a descriptor boundary.
                feature_totals = np.zeros(1 + (group.n_types if source.color else 0), np.uint64)
                feature_totals[0] = np.sum(runs["n_conf"].astype(np.uint64) * runs["n_real"])
                if source.color:
                    foffsets = _read_bound(group, bi, "feature_offsets", bindings[bi])
                    ftypes = _read_bound(group, bi, "feature_types", bindings[bi])
                    weights = np.repeat(runs["n_conf"], np.diff(foffsets).astype(np.int64))
                    feature_totals[:] += np.bincount(ftypes, weights=weights,
                        minlength=len(feature_totals)).astype(np.uint64)
                bindings[bi]["_coords"] = (int(block["offset"]), int(block["length"]),
                                             bytes(block["sha256"]))
                key = gi, bi
                totals[key] = feature_totals
                bindings_by_key[key] = bindings[bi]
                descriptors.append(CPUPhysicalBlockDescriptor(
                    group_index=gi, block_index=bi, group_row0=row0,
                    compact_row0=compact_offsets[gi] + row0,
                    gid_base=int(offsets[gi]) + row0, n_conf=rows,
                    shape_record=(int(block["offset"]), int(block["length"]),
                                  rows * group.L_heavy * 3 * 2, rows),
                    shape_reader=metadata, library_width=group.L,
                    physical_shape_width=group.L_heavy,
                    field_width=group.L_field_width,
                    coord_scale=source.corpus.step))
    positions = {(int(d["group"]), int(d["block"])): i for i, d in enumerate(
        source.corpus.block_directory("expensive_first"))}
    descriptors.sort(key=lambda d: positions[d.group_index, d.block_index])
    source._cpu_groups = groups
    source._cpu_bindings = bindings_by_key
    source._cpu_feature_totals = totals
    source._cpu_metadata_owned_bytes = (len(groups) * 512 + sum(
        128 * len(binding) + totals[key].nbytes for key, binding in bindings_by_key.items()))
    source._cpu_descriptors = tuple(descriptors)
    source._cpu_descriptor_index = {(d.group_index, d.block_index): d for d in descriptors}
    return source._cpu_descriptors


@dataclass(frozen=True)
class CPUPreparedBlock:
    descriptor: CPUPhysicalBlockDescriptor
    result: object
    molecule_ids: np.ndarray
    shape_compressed_bytes: int
    shape_raw_bytes: int
    color_compressed_bytes: int = 0
    color_raw_bytes: int = 0
    color_xyz_i16: object = None


def _decode_cpu_block(source, descriptor):
    source.cpu_block_descriptors()
    key = descriptor.group_index, descriptor.block_index
    if source._cpu_descriptor_index.get(key) is not descriptor:
        raise ValueError("descriptor is not owned by this source")
    bi = descriptor.block_index
    binding = source._cpu_bindings[key]
    metadata = descriptor.shape_reader
    # Reopen only the bounded file descriptor, never re-parse the group's
    # entire block table for each claim (which would be quadratic in blocks).
    with open_corpus_file(source.corpus.root, "groups", metadata.file_name, tier="deep") as file:
        group = SimpleNamespace(index=metadata.index, path=metadata.path,
            L_heavy=metadata.L_heavy, L=metadata.L, n_types=metadata.n_types,
            _og=SimpleNamespace(file=file, sections=metadata.sections))
        if file.stat_key != metadata.stat_key:
            raise _refuse(group, bi, "validation.scan_required", "group changed since corpus open")
        offset, length, expected = binding["_coords"]
        data = native.read_into(file.fd, metadata.sections["coords"]["offset"] + offset, length)
        if hashlib.sha256(data).digest() != expected:
            raise _refuse(group, bi, "integrity.block_hash", "coordinate block hash mismatch")
        runs = _read_bound(group, bi, "runs", binding)
        if np.any(runs["mol_id"] >= (1 << 63)):
            raise _refuse(group, bi, "limit.mol_id", "mol_id must be in [0, 2^63)")
        features = None
        if source.color:
            fo = _read_bound(group, bi, "feature_offsets", binding)
            ao = _read_bound(group, bi, "feature_atom_offsets", binding)
            features = native.FeatureSlice(fo - fo[0],
                _read_bound(group, bi, "feature_types", binding), ao - ao[0],
                _read_bound(group, bi, "feature_atoms", binding))
        vaa = _read_bound(group, bi, "vaa", binding)
        colors = (_read_bound(group, bi, "color_self", binding)
                  if source.color and "color_self" in binding else None)
        try:
            result = native.write_cpu_block(data, runs, descriptor.n_conf,
                group.L_heavy, group.L, group.n_types if source.color else 0,
                features, descriptor.coord_scale,
                self_overlaps=native.SelfOverlaps(vaa, colors, 0))
        except native.CodecError as exc:
            raise _refuse(group, bi, "value.self_overlap" if "self" in exc.code
                          else "layout.block_header", str(exc)) from exc
        mids = np.repeat(runs["mol_id"].astype(np.int64), runs["n_conf"])
        mids.flags.writeable = False
        return CPUPreparedBlock(descriptor, result, mids, len(data), descriptor.shape_record[2])


def decode_cpu_block(source, descriptor):
    try:
        return _decode_cpu_block(source, descriptor)
    except native.CodecError as exc:
        raise _refuse(descriptor.shape_reader, descriptor.block_index,
                      "io.short_read" if exc.code.startswith("io.") else "layout.block_header",
                      str(exc)) from exc
    except CorpusValidationError as exc:
        # Opening a changed/truncated group also identifies the attempted
        # physical block, even when the refusal precedes its payload read.
        raise _refuse(descriptor.shape_reader, descriptor.block_index,
                      exc.code, str(exc)) from exc


class R3gWorkerSelfCache:
    """Values-only scorer handles from authenticated native block output."""
    def __init__(self, source, query_key):
        self.source, self.query_key = source, query_key
        self.entries, self.routes = {}, {}
        self._lock = threading.Lock()
        self.peak_entries = 0
        self._query_lock = threading.Lock()
        self._query_native = self._query_signature = None
        self.query_preparations = 0
        self.query_shape_self_evaluations = self.query_color_self_evaluations = 0
        self.query_self_preparation_seconds = 0.0
        self.query_retained_bytes = 0

    @property
    def retained_bytes(self):
        return sum(self.routes[k]["bytes"] + 1024 for k in self.entries)

    @staticmethod
    def _query_bytes(arrays):
        dtypes = (np.float32, np.int32, np.int32, np.float32, np.float32)
        return tuple((a.shape, a.dtype.str, a.tobytes()) for a in
                     (np.asarray(a, dtype=t) for a, t in zip(arrays, dtypes)))

    def _query_self(self, cpu, qc, qt, qn, r, p):
        # Source + query_key scopes this cache. Validate once under a lock;
        # subsequent claims compare exact query bytes, including padding.
        # Small bytes comparisons avoid repeated hash GIL transitions.
        signature = self._query_bytes((qc, qt, qn, r, p))
        with self._query_lock:
            if self._query_native is None:
                empty = np.empty((0, 1, 3), np.float32)
                counts = np.empty((0, len(r)), np.uint8)
                q, types, n, lib, cnt, rm, pm = cpu._normalize_self_overlap_inputs(
                    qc, qt, qn, empty, counts, r, p)
                if self._query_bytes((q, types, n, rm, pm)) != signature:
                    raise ValueError("format-3 query changed while preparing self snapshot")
                from ..numerics import assert_arithmetic_pinned
                assert_arithmetic_pinned()
                native = cpu.extension().prepare_combo_self_overlaps(
                    q, types, n, lib, cnt, rm, pm, self.query_key,
                    None, None, None, True, True)
                self._query_signature = signature
                self._query_native = native
                self.query_preparations += 1
                self.query_shape_self_evaluations += len(n)
                self.query_color_self_evaluations += len(n)
                self.query_self_preparation_seconds += (
                    native.shape_preparation_seconds + native.color_preparation_seconds)
                self.query_retained_bytes = sum(len(item[2]) for item in signature) + len(n) * 12
            elif signature != self._query_signature:
                raise ValueError("format-3 self cache query inputs were mutated or replaced")
            return self._query_native

    def block(self, cpu, descriptor, decoded, coords, qc, qt, qn, r, p, *, mode="combo"):
        if decoded.descriptor is not descriptor:
            raise ValueError("worker decoder returned a different descriptor")
        if mode in ("combo", "rescore"):
            color = self.source.corpus.color
            for name, actual in (("r", r), ("p", p)):
                expected = np.asarray(color[name]["values"], dtype=np.float32)
                if not np.array_equal(actual, expected):
                    raise ValueError("query interaction matrix differs from corpus: " + name)
        key = descriptor.group_index, descriptor.block_index
        previous = self.entries.get(key)
        if previous is not None:
            return previous[1], None, True, False
        values, colors = decoded.result.vaa, decoded.result.color_self
        binding = hashlib.sha256((self.query_key + repr(key)).encode()).hexdigest()
        if mode != "combo":
            from .cpu_modes import prepare_block_self
            prepare_color = mode == "rescore"
            raw = cpu.extension()._prepare_r3g_library_self_overlaps(
                coords, decoded.result.counts, r, p, binding,
                np.asarray(values, dtype=np.float64),
                colors if prepare_color else None, True, prepare_color,
                shape_float32_reciprocal=True)
            shape_values = np.array(raw.shape_library_values, dtype=np.float32)
            color_values = np.array(raw.color_library_values, dtype=np.float64)
            shape_values.flags.writeable = color_values.flags.writeable = False
            values_fingerprint = _batched_fingerprint_arrays(
                "roshambo3.combo_library_self_values.v1",
                (("shape_reduced_f32", shape_values),
                 ("color_angstrom3_f64", color_values)))
            library = cpu.PreparedComboLibrarySelfOverlaps(
                raw,
                _batched_fingerprint_arrays(
                    cpu._SELF_OVERLAP_CONTRACT + "|library-v1",
                    (("library_coords", coords),
                     ("library_counts", decoded.result.counts),
                     ("interaction_r", r), ("interaction_p", p))),
                values_fingerprint, shape_values, color_values, len(r),
                descriptor.n_conf, raw.shape_preparation_seconds,
                raw.color_preparation_seconds, raw.shape_library_terms,
                raw.color_library_terms, bool(raw.shape_prepared),
                bool(raw.color_prepared),
                bool(raw.library_shape_from_stored_volume),
                bool(raw.library_color_from_sidecar))
            handle = prepare_block_self(cpu, coords, values, binding, library,
                                        rescore=prepare_color)
            retained = handle.retained_native_value_bytes
        else:
            # Authentication and native decode validated the block. Retain
            # the public scorer's input/value fingerprints while avoiding its
            # generic sidecar wrapper's repeated query/layout preparation.
            query_native = self._query_self(cpu, qc, qt, qn, r, p)
            raw = cpu.extension()._prepare_r3g_library_self_overlaps(coords,
                decoded.result.counts, r, p, binding,
                np.asarray(values, dtype=np.float64), colors, True, True,
                shape_float32_reciprocal=False)
            if (not raw.library_shape_from_stored_volume or
                    bool(raw.library_color_from_sidecar) != (colors is not None)):
                raise RuntimeError("native scorer did not consume authenticated stored self values")
            shape_values = np.array(raw.shape_library_values, dtype=np.float32)
            color_values = np.array(raw.color_library_values, dtype=np.float64)
            shape_values.flags.writeable = color_values.flags.writeable = False
            fingerprint = _batched_fingerprint_arrays("roshambo3.combo_library_self_values.v1",
                (("shape_reduced_f32", shape_values), ("color_angstrom3_f64", color_values)))
            library = cpu.PreparedComboLibrarySelfOverlaps(raw,
                _batched_fingerprint_arrays(cpu._SELF_OVERLAP_CONTRACT + "|library-v1",
                    (("library_coords", coords), ("library_counts", decoded.result.counts),
                     ("interaction_r", r), ("interaction_p", p))),
                fingerprint, shape_values, color_values, len(r), descriptor.n_conf,
                raw.shape_preparation_seconds, raw.color_preparation_seconds,
                raw.shape_library_terms, raw.color_library_terms,
                True, True, bool(raw.library_shape_from_stored_volume),
                bool(raw.library_color_from_sidecar))
            input_fingerprint = _batched_fingerprint_arrays(
                cpu._SELF_OVERLAP_CONTRACT + "|query-library-v1",
                (("query_coords", qc), ("query_types", qt), ("query_n_atoms", qn),
                 ("library_coords", coords), ("library_counts", decoded.result.counts),
                 ("interaction_r", r), ("interaction_p", p)))
            native_digest = _batched_fingerprint_arrays("roshambo3.combo_self_handle.v1",
                (("inputs", np.frombuffer(bytes.fromhex(input_fingerprint), np.uint8)),
                 ("library_values", np.frombuffer(bytes.fromhex(fingerprint), np.uint8))))
            combined = cpu.extension()._combine_r3g_self_overlaps(
                query_native, raw, native_digest)
            handle = cpu.PreparedComboSelfOverlaps(combined, input_fingerprint,
                native_digest, fingerprint, len(qn), descriptor.n_conf, len(r),
                True, True, raw.shape_preparation_seconds,
                raw.color_preparation_seconds, library)
            retained = (len(qn) + 3 * descriptor.n_conf) * 12
        self.entries[key] = (binding, handle)
        with self._lock:
            self.peak_entries = max(self.peak_entries, len(self.entries))
        self.routes[key] = {"vaa": "verified_stored",
            "color": ("unused" if mode == "shape" else
                      "sidecar" if colors is not None else "recomputed"),
            "color_reason": ("color_self_section_absent" if colors is None and
                             mode in ("combo", "rescore") else None), "bytes": retained}
        return handle, None, False, False

    def release_block(self, descriptor):
        """Release a values handle after its owning worker finished scoring."""
        key = descriptor.group_index, descriptor.block_index
        self.entries.pop(key, None)
        self.routes.pop(key, None)

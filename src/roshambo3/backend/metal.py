"""Apple Silicon Metal screening with an explicit FP32 numerical contract.

Shader compilation and GPU execution require macOS 15+ and a visible Apple GPU.
Imports and backend discovery remain usable on other platforms.
"""

from __future__ import annotations

import hashlib
import platform
import re
import threading
import time
from pathlib import Path

import numpy as np

from ..engine.config import FIELD_SPACING_BY_WIDTH
from .spec import ColorPlacement, PoseSearch, UpdateRule

_SOURCE = Path(__file__).with_name("_metal_src") / "kernels.metal"
_TEXTURE_BINDING_LIMIT = 128


_TOKEN = re.compile(r"\w+|[^\w\s]")


def _replace_once(text, old, new):
    """Replace the single occurrence of the token sequence ``old`` in ``text``.

    Matched by tokens, not characters, so a reformat of kernels.metal cannot
    silently turn a substitution into a no-op: whitespace may differ between
    tokens (and must separate two word tokens), and anything other than exactly
    one match is an error rather than an unmodified kernel.
    """
    tokens = _TOKEN.findall(old)
    pattern = ""
    for prev, tok in zip([None] + tokens, tokens):
        if prev is not None:
            word = re.match(r"\w", prev) and re.match(r"\w", tok)
            pattern += r"\s+" if word else r"\s*"
        pattern += re.escape(tok)
    found = len(re.findall(pattern, text))
    if found != 1:
        raise RuntimeError(
            f"kernels.metal: expected one {old!r} in the color-field template, "
            f"found {found}")
    return re.sub(pattern, lambda _: new, text)


def _wire_kernel_source(source):
    from ..numerics import WIRE_QUANT_STEP
    return _replace_once(source, "R3_WIRE_QUANT_STEP", f"{float(WIRE_QUANT_STEP):.9g}f")


def _color_kernel_source(source, planes):
    # Compile separate entry points for each possible selected-plane subset.
    # Thresholds can select a different subset on each query without routing
    # sparse planes through a field-only kernel.
    start = source.index("kernel void optimize_diagonal_factored(")
    end = source.index("kernel void rescore(", start)
    template = source[start:end]
    masks = [0]
    for plane in planes:
        masks += [m | (1 << plane) for m in masks]
    wrappers = []
    for mask in masks[1:]:
        wrapper = _replace_once(
            template, "optimize_diagonal_factored(",
            f"optimize_color_fields_m{mask}(")
        wrapper = _replace_once(
            wrapper, "device float* out,constant Params& p,",
            "device float* out,device const Grid* color_grids,device const float4* color_field,device const uint* color_slots,constant Params& p,",
        )
        wrapper = _replace_once(
            wrapper, "optimize_body<true,true>",
            f"optimize_body<true,true,{mask}>")
        wrapper = _replace_once(
            wrapper, "out,grids,field,n,p,",
            "out,color_grids,color_field,color_slots,p,")
        wrappers.append(wrapper)
    return source + "\n" + "\n".join(wrappers)


def _texture_batches(nq, shape_textures, color_textures, offsets, row_masks=None):
    """Compact selected planes and partition bindings; pair/output indices stay global."""
    begin = 0
    while begin < nq:
        end = begin
        while end < nq:
            if row_masks is not None and row_masks[end] != row_masks[begin]:
                break
            size = (end+1-begin if shape_textures else 0)
            size += int(offsets[end+1])-int(offsets[begin])
            if size > _TEXTURE_BINDING_LIMIT:
                break
            end += 1
        if end == begin:
            raise ValueError("A query exceeds the Metal texture binding limit")
        bound = shape_textures[begin:end] if shape_textures else []
        bound = bound + color_textures[int(offsets[begin]):int(offsets[end])]
        yield begin, end, bound
        begin = end


def _color_grid_record(points, spacing, padding, offset):
    scale = float(np.float32(0.77687365))
    points = points.astype(np.float64) * scale
    origin64 = points.min(axis=0) - padding * scale
    upper64 = points.max(axis=0) + padding * scale
    delta64 = spacing * scale
    extents = np.ceil((upper64 - origin64) / delta64)
    if (
        not np.all(np.isfinite(extents))
        or np.any(extents < 1)
        or np.any(extents >= 2**31 - 1)
    ):
        raise MemoryError("Metal color field geometry exceeds supported addressing")
    dims = extents.astype(np.uint64) + 1
    size = int(dims[0]) * int(dims[1]) * int(dims[2])
    if offset + size > 2**32 - 1:
        raise MemoryError("Metal color fields exceed 32-bit addressing")
    record = np.r_[
        np.r_[origin64, delta64].astype(np.float32).view(np.uint32), _u(*dims, offset)
    ]
    return record, size


def extension():
    try:
        from .. import _roshambo3_metal

        return _roshambo3_metal
    except ImportError as exc:
        raise ImportError(
            "Metal requires a native arm64 macOS build with ROSHAMBO3_METAL=ON (macOS 15+)"
        ) from exc


def available():
    try:
        extension().device_info(0)
        return True
    except (ImportError, ValueError, RuntimeError):
        return False


def build_stamp():
    ext = extension()
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(_SOURCE.parent.iterdir())
        if p.suffix in (".mm", ".metal")
    }
    return dict(
        backend="metal",
        status="experimental",
        build_type=ext.__build_type__,
        compiler=ext.__compiler__,
        os=platform.platform(),
        language="Metal 3.1",
        math_mode="safe",
        floating_point_functions="precise",
        accumulation="fp32",
        reduction="four_lane_xor_2_1",
        fields="laplacian_prefilter; manual_fp32, texture16, texture32",
        default_factor_color_torque=True,
        default_specialize_color=True,
        default_color_range_policy="active",
        default_field_sampler="texture16",
        default_color_field_min_sources=12,
        default_color_field_spacing_a=0.25,
        default_color_field_sampler="texture16",
        color_fields="default diagonal planes 1/2; cutoff 12; analytic bounds/budget fallback",
        search_exp2="metal::fast::exp2",
        final_exp2="metal::precise::exp2",
        source_sha256=hashlib.sha256(
            repr(sorted(sources.items())).encode()
        ).hexdigest(),
        sources=sources,
        python_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compiled_src_sha256=ext.__metal_src_sha256__,
        compiled_src_dirty=ext.__metal_src_dirty__,
        sdk_max_allowed=ext.__sdk_max_allowed__,
        extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
    )


def _u(*values):
    return np.asarray(values, np.uint32)


def _params(
    nq,
    qw,
    nl,
    width,
    nf,
    record,
    planar,
    color,
    offset,
    pairs,
    steps,
    starts,
    combo,
    mixing,
    lrq,
    lrt,
    padding,
):
    return np.concatenate(
        (
            _u(
                nq,
                qw,
                nl,
                width,
                nf,
                record,
                planar,
                color,
                offset & 0xFFFFFFFF,
                pairs,
                steps,
                starts,
                combo,
                0,
                0,
                0,
            ),
            np.asarray([mixing, lrq, lrt, padding], np.float32).view(np.uint32),
            _u(0),  # Pair-output offset for texture-batched optimizer launches.
        )
    )


class MetalContext:
    """Default half textures with dense color fields at cutoff 12 / 0.25 A.

    An omitted color cutoff disables fields when specialization or torque
    factoring is explicitly disabled. Zero explicitly requests analytic color.
    """
    def __init__(
        self,
        device_id=0,
        *,
        fast_search_exp2=True,
        field_sampler="texture16",
        profile_kernels=False,
        specialize_color=True,
        factor_color_torque=True,
        color_range_policy="active",
        color_field_min_sources=None,
        color_field_spacing_a=0.25,
        color_field_types=(1, 2),
        color_field_sampler="texture16",
        color_field_max_bytes=128 << 20,
    ):
        if field_sampler not in ("manual", "texture16", "texture32"):
            raise ValueError("field_sampler must be manual, texture16, or texture32")
        if color_range_policy not in ("active", "width"):
            raise ValueError("color_range_policy must be active or width")
        if color_field_min_sources is None:
            color_field_min_sources = 12 if specialize_color and factor_color_torque else 0
        if type(color_field_min_sources) is not int or color_field_min_sources < 0:
            raise ValueError("color_field_min_sources must be a nonnegative integer")
        if not np.isfinite(color_field_spacing_a) or color_field_spacing_a <= 0:
            raise ValueError("color_field_spacing_a must be positive and finite")
        planes = tuple(color_field_types)
        if (
            not planes
            or any(type(t) is not int or not 1 <= t <= 7 for t in planes)
            or len(set(planes)) != len(planes)
        ):
            raise ValueError("color_field_types must be unique feature IDs from 1 to 7")
        if color_field_sampler not in ("manual", "texture16", "texture32"):
            raise ValueError("invalid color_field_sampler")
        if type(color_field_max_bytes) is not int or color_field_max_bytes < 1:
            raise ValueError("color_field_max_bytes must be a positive integer")
        self.color_field_min_sources = color_field_min_sources
        self.color_field_spacing_a = float(color_field_spacing_a)
        self.color_field_types = planes
        self.color_field_sampler = color_field_sampler
        self.color_field_max_bytes = color_field_max_bytes
        self.color_cache = None
        self.last_color_fields = []
        self.color_field_build_wall_s = self.color_field_build_gpu_s = 0.0
        self.color_range_policy = color_range_policy
        self.last_optimizer_kernel = None
        self.last_optimizer_dispatches = []
        self.device_id = int(device_id)
        self.info = extension().device_info(device_id)
        started = time.perf_counter()
        self.fast_search_exp2 = bool(fast_search_exp2)
        self.field_sampler = field_sampler
        self.specialize_color = bool(specialize_color)
        self.factor_color_torque = bool(factor_color_torque)
        if (
            "texture32" in (field_sampler, color_field_sampler)
            and not self.info["float32_filtering"]
        ):
            raise ValueError("texture32 requires Apple GPU family 9 or newer")
        source = (
            f"#define R3_METAL_FAST_SEARCH_EXP2 {int(self.fast_search_exp2)}\n"
            f"#define R3_METAL_TEXTURE_FIELDS {int(field_sampler != 'manual')}\n"
            f"#define R3_METAL_COLOR_TEXTURE_FIELDS {int(color_field_sampler != 'manual')}\n"
            + _wire_kernel_source(_SOURCE.read_text())
        )
        source = _color_kernel_source(source, planes)
        self.native = extension().Context(device_id, source)
        self.native.profile_kernels = bool(profile_kernels)
        self.compile_s = time.perf_counter() - started
        self.buffers = {}
        self.query_key = None
        self.grids = {}
        self.textures = []
        self.lock = threading.RLock()
        self.closed = False

    def buffer(self, name, size):
        current = self.buffers.get(name)
        if current is None or current[0] < size:
            # Release an obsolete allocation before checking the new budget.
            self.buffers.pop(name, None)
            current = None
            current = (size, self.native.allocate(max(4, int(size))))
            self.buffers[name] = current
        return current[1]

    def put(self, name, array):
        array = np.ascontiguousarray(array)
        b = self.buffer(name, array.nbytes)
        b.write(array)
        return b

    def close(self):
        with self.lock:
            self.buffers.clear()
            self.grids.clear()
            self.textures.clear()
            self.color_cache = None
            self.last_color_fields = []
            for name in ("q", "qt", "qn", "r", "h", "qs"):
                self.__dict__.pop(name, None)
            self.native = None
            self.closed = True

    def timings(self):
        return dict(
            gpu_s=self.native.gpu_seconds,
            shader_compile_s=self.compile_s,
            field_sampler=self.field_sampler,
            specialize_color=self.specialize_color,
            factor_color_torque=self.factor_color_torque,
            color_range_policy=self.color_range_policy,
            last_optimizer_kernel=self.last_optimizer_kernel,
            optimizer_dispatches=self.last_optimizer_dispatches,
            color_field_min_sources=self.color_field_min_sources,
            color_field_spacing_a=self.color_field_spacing_a,
            color_field_types=self.color_field_types,
            color_field_sampler=self.color_field_sampler,
            color_field_max_bytes=self.color_field_max_bytes,
            color_fields=self.last_color_fields,
            color_field_build_wall_s=self.color_field_build_wall_s,
            color_field_build_gpu_s=self.color_field_build_gpu_s,
            kernel_gpu_s=dict(self.native.kernel_seconds),
            profiling_separate_commands=bool(self.native.profile_kernels),
            search_exp2="metal::fast::exp2"
            if self.fast_search_exp2
            else "metal::precise::exp2",
        )

    def query(self, qc, qt, qn, r, h, params, spacing, padding):
        digest = hashlib.sha256()
        for a in (qc, qt, qn, r, h):
            digest.update(str((a.shape, a.dtype)).encode())
            digest.update(a.tobytes())
        key = (digest.digest(), bool(params[7]), bool(params[13]))
        jobs = []
        texture_error = None
        # Query self-overlaps evaluated by this call: all of them when the
        # query changed, none when the cached ones are reused.
        self.last_query_self_evaluations = 0
        if key != self.query_key:
            self.last_query_self_evaluations = len(qn)
            self.color_cache = None
            self.last_color_fields = []
            for name in (
                "color_grid_metadata",
                "color_grid_field",
                "color_texture_error",
            ):
                self.buffers.pop(name, None)
            self.grids.clear()
            self.textures.clear()
            self.query_key = key
            prepared_q, prepared_types = qc.copy(), qt.copy()
            ranges = np.zeros((len(qn), len(r), 2), np.uint32)
            if params[13]:
                for row, nr in enumerate(qn):
                    order = np.argsort(
                        np.where(qt[row, int(nr) :] > 0, qt[row, int(nr) :], 255),
                        kind="stable",
                    ) + int(nr)
                    prepared_q[row, int(nr) :] = qc[row, order]
                    prepared_types[row, int(nr) :] = qt[row, order]
                    for feature in range(1, len(r)):
                        indices = np.flatnonzero(prepared_types[row] == feature)
                        if len(indices):
                            ranges[row, feature] = [indices[0], indices[-1] + 1]
            self.q = self.put("query", prepared_q)
            self.qt = self.put("types", prepared_types)
            self.qn = self.put(
                "nquery",
                np.r_[qn, np.uint32(qc.shape[1]), ranges.ravel()].astype(np.uint32),
            )
            # Prepare pair invariants once, outside every optimizer iteration.
            scale = np.float32(0.77687365)
            decay = r / np.float32(2 * scale * scale * np.float32(np.log(2.0)))
            amplitude = np.divide(
                h * h, r * np.sqrt(r), out=np.zeros_like(h), where=r > 0
            )
            self.r, self.h = self.put("r", decay), self.put("h", amplitude)
            self.qs = self.buffer("query_self", len(qn) * 8)
            jobs.append(
                (
                    "query_self",
                    [self.q, self.qt, self.qn, self.r, self.h, self.qs],
                    params,
                    len(qn),
                    64,
                )
            )
        grid_key = (float(spacing), float(padding))
        if grid_key not in self.grids:
            records = []
            offset = 0
            for row, n in enumerate(qn):
                points = qc[row, : int(n), :3] * np.float32(0.77687365)
                origin64 = points.min(axis=0).astype(np.float64) - float(
                    padding
                ) * float(np.float32(0.77687365))
                origin = origin64.astype(np.float32)
                upper64 = points.max(axis=0).astype(np.float64) + float(
                    padding
                ) * float(np.float32(0.77687365))
                delta64 = float(spacing) * float(np.float32(0.77687365))
                delta = np.float32(delta64)
                extents = np.ceil((upper64 - origin64) / delta64)
                if (
                    not np.all(np.isfinite(extents))
                    or np.any(extents < 1)
                    or np.any(extents >= 2**31 - 1)
                ):
                    raise MemoryError(
                        "Metal field geometry exceeds supported addressing"
                    )
                dims = extents.astype(np.uint64) + 1
                size = int(dims[0]) * int(dims[1]) * int(dims[2])
                if offset + size > 2**32 - 1:
                    raise MemoryError(
                        "Metal query fields exceed 32-bit grid addressing"
                    )
                record = np.concatenate(
                    (
                        np.r_[origin, delta].astype(np.float32).view(np.uint32),
                        _u(*dims, offset),
                    )
                )
                records.append(record)
                offset += size
            # Retain only one spacing at a time, bounding caches on varied corpora.
            self.grids.clear()
            self.textures.clear()
            meta = self.put("grid_metadata", np.array(records, np.uint32))
            field = self.buffer("grid_field", offset * 16)
            jobs.append(
                (
                    "build_field",
                    [self.q, self.qn, meta, field],
                    _u(len(qn)),
                    offset,
                    128,
                )
            )
            if self.field_sampler != "manual":
                texture_error = self.put("texture_error", _u(0))
                for row, record in enumerate(records):
                    dims = record[4:7]
                    tex = self.native.texture(
                        *(int(x) for x in dims), self.field_sampler == "texture16"
                    )
                    self.textures.append(tex)
                    jobs.append(
                        (
                            "pack_field",
                            [meta, field, texture_error],
                            _u(row, self.field_sampler == "texture16"),
                            int(np.prod(dims.astype(np.uint64))),
                            128,
                            [tex],
                        )
                    )
            self.grids[grid_key] = (meta, field)
        if jobs:
            self.native.submit(jobs)
        if texture_error is not None and texture_error.read(np.dtype("uint32"), [1])[0]:
            self.grids.clear()
            self.textures.clear()
            self.query_key = None
            raise FloatingPointError(
                "Metal field cannot be represented in the requested texture format"
            )
        return self.grids[grid_key]

    def color_query(self, qc, qt, qn, r, h, params, minimum, spacing, padding):
        self.last_color_fields = []
        self.color_field_build_wall_s = self.color_field_build_gpu_s = 0.0
        if not minimum or not params[12] or not params[13]:
            return None
        if not self.specialize_color or not self.factor_color_torque:
            raise ValueError(
                "Metal color fields require specialization and torque factoring"
            )
        key = (minimum, float(spacing), float(padding))
        if self.color_cache is not None and self.color_cache[0] == key:
            self.last_color_fields = self.color_cache[2]
            return self.color_cache[1]
        self.color_cache = None
        started = time.perf_counter()
        gpu_before = self.native.gpu_seconds
        for name in ("color_grid_metadata", "color_grid_field", "color_texture_error", "color_texture_slots"):
            self.buffers.pop(name, None)
        records = np.zeros((len(qn) * len(r), 8), np.uint32)
        planes, offset, mask = [], 0, 0
        textured = self.color_field_sampler != "manual"
        for row, nr in enumerate(qn):
            for plane in self.color_field_types:
                if plane >= len(r) or r[plane, plane] <= 0 or h[plane, plane] == 0:
                    continue
                indices = np.flatnonzero(qt[row, int(nr) :] == plane) + int(nr)
                if len(indices) < minimum:
                    continue
                record, size = _color_grid_record(
                    qc[row, indices, :3], spacing, padding, offset
                )
                if textured and np.any(record[4:7] > 2048):
                    continue  # This plane retains the analytic path.
                bytes_per_node = (
                    16 + (8 if self.color_field_sampler == "texture16" else 16)
                    if textured
                    else 16
                )
                if (offset + size) * bytes_per_node > self.color_field_max_bytes:
                    continue
                records[row * len(r) + plane] = record
                planes.append(
                    dict(query_row=row, plane=plane, sources=len(indices), nodes=size)
                )
                offset += size
                mask |= 1 << plane
        if not planes:
            self.color_cache = (key, None, [])
            return None
        meta = self.put("color_grid_metadata", records)
        field = self.buffer("color_grid_field", offset * 16)
        jobs = []
        textures = []
        slots = np.zeros(len(records), np.uint32)
        row_counts = np.zeros(len(qn), np.uint32)
        error = self.put("color_texture_error", _u(0))
        for selected in planes:
            row, plane = selected["query_row"], selected["plane"]
            index = row * len(r) + plane
            jobs.append(
                (
                    "build_color_field",
                    [self.q, self.qt, self.qn, self.r, self.h, meta, field, error],
                    _u(row, plane, qc.shape[1], len(r)),
                    selected["nodes"],
                    128,
                )
            )
            if textured:
                tex = self.native.texture(
                    *(int(x) for x in records[index, 4:7]),
                    self.color_field_sampler == "texture16",
                )
                slots[index] = len(textures)
                textures.append(tex)
                row_counts[row] += 1
                jobs.append(
                    (
                        "pack_field",
                        [meta, field, error],
                        _u(index, self.color_field_sampler == "texture16"),
                        selected["nodes"],
                        128,
                        [tex],
                    )
                )
        self.native.submit(jobs)
        if error.read(np.dtype("uint32"), [1])[0]:
            raise FloatingPointError(
                "Metal color field has nonfinite nodes or exceeds requested texture range"
            )
        offsets = np.r_[np.uint32(0), np.cumsum(row_counts, dtype=np.uint32)]
        # Type zero never samples color, so its metadata entry stores each row's
        # compact texture-prefix offset, including rows with no selected planes.
        slots[::len(r)] = offsets[:-1]
        slot_buffer = self.put("color_texture_slots", slots) if textured else self.qn
        row_masks = np.zeros(len(qn), np.uint32)
        for selected in planes:
            row_masks[selected["query_row"]] |= 1 << selected["plane"]
        result = (mask, meta, field, textures, slot_buffer, offsets, row_masks)
        self.last_color_fields = planes
        self.color_cache = (key, result, planes)
        self.color_field_build_wall_s = time.perf_counter() - started
        self.color_field_build_gpu_s = self.native.gpu_seconds - gpu_before
        return result


def create_context(
    device_id=0,
    *,
    fast_search_exp2=True,
    field_sampler="texture16",
    profile_kernels=False,
    specialize_color=True,
    factor_color_torque=True,
    color_range_policy="active",
    color_field_min_sources=None,
    color_field_spacing_a=0.25,
    color_field_types=(1, 2),
    color_field_sampler="texture16",
    color_field_max_bytes=128 << 20,
):
    return MetalContext(
        device_id,
        fast_search_exp2=fast_search_exp2,
        field_sampler=field_sampler,
        profile_kernels=profile_kernels,
        specialize_color=specialize_color,
        factor_color_torque=factor_color_torque,
        color_range_policy=color_range_policy,
        color_field_min_sources=color_field_min_sources,
        color_field_spacing_a=color_field_spacing_a,
        color_field_types=color_field_types,
        color_field_sampler=color_field_sampler,
        color_field_max_bytes=color_field_max_bytes,
    )


def screen_args(config, width, shape_width, color_offset, **unused):
    """Resolve scientific settings without querying CUDA launch capabilities."""
    from dataclasses import fields

    from ..engine.config import ScreenConfig

    baseline = ScreenConfig()
    supported = {
        "pose_search",
        "color",
        "mixing",
        "steps",
        "lr_q",
        "lr_t",
        "start_mode",
        "field_spacing_a",
        "field_pad_a",
        "update_rule",
        "color_placement",
        "budget_override",
        "color_field_min_sources",
        "color_field_spacing_a",
    }
    for item in fields(config):
        if item.name not in supported and getattr(config, item.name) != getattr(
            baseline, item.name
        ):
            raise NotImplementedError(
                f"Metal does not support ScreenConfig.{item.name}"
            )
    if PoseSearch(config.pose_search) != PoseSearch.FIELD:
        raise NotImplementedError("Metal supports pose_search=FIELD")
    if UpdateRule(config.update_rule) != UpdateRule.ADAGRAD_EXP:
        raise NotImplementedError("Metal supports update_rule=ADAGRAD_EXP")
    # Use the public width policy, independent of color padding.
    spacing = config.field_spacing_a
    if spacing is None:
        table = FIELD_SPACING_BY_WIDTH
        spacing = 0.8
        for edge in sorted(table):
            if edge <= shape_width:
                spacing = table[edge]
    opt = config.resolved_optimizer()
    return dict(
        pose_search=config.pose_search,
        color=config.color,
        mixing=config.mixing,
        steps=opt["steps"],
        lr_q=opt["lr_q"],
        lr_t=opt["lr_t"],
        start_mode=config.start_mode,
        field_spacing_a=spacing,
        field_pad_a=config.field_pad_a,
        update_rule=config.update_rule,
        color_placement=config.color_placement,
        library_color_offset=color_offset if config.color else 0,
        **(
            dict(
                color_field_min_sources=config.color_field_min_sources,
                color_field_spacing_a=config.color_field_spacing_a,
            )
            if config.color_field_min_sources is not None
            else (dict(color_field_spacing_a=config.color_field_spacing_a)
                  if config.color_field_spacing_a is not None else {})
        ),
    )


def _validate(
    query_coords,
    query_types,
    query_n_atoms,
    library_coords,
    library_counts,
    pair_query_row,
    pair_library_row,
    interaction_r,
    interaction_p,
    color,
    offset,
    tile,
):
    if (pair_query_row is None) != (pair_library_row is None):
        raise ValueError(
            "pass both pair_query_row and pair_library_row, or neither for a "
            "dense launch")
    dense = pair_query_row is None
    arrays = (
        query_coords,
        query_types,
        query_n_atoms,
        library_coords,
        library_counts,
        pair_query_row,
        pair_library_row,
        interaction_r,
        interaction_p,
    )
    # Only the pair list (indices 5 and 6) may be None, and only both together
    # (checked above): every other input must still be an array.
    if any(not isinstance(a, np.ndarray) for i, a in enumerate(arrays)
           if not (dense and i in (5, 6))):
        raise TypeError("Metal inputs must be NumPy arrays")
    q, types, n, b, counts, pq, pb, r, h = arrays
    if (
        q.dtype != np.float32
        or q.ndim != 3
        or q.shape[2] != 4
        or not q.flags.c_contiguous
    ):
        raise ValueError("query_coords must be contiguous float32 [queries,width,4]")
    if (
        types.shape != q.shape[:2]
        or types.dtype != np.int32
        or n.shape != (len(q),)
        or n.dtype != np.int32
    ):
        raise ValueError("query types/counts must match query coordinates (int32)")
    if np.any(n <= 0) or np.any(n > q.shape[1]) or not np.all(np.isfinite(q)):
        raise ValueError("invalid query counts or nonfinite coordinates")
    if (
        b.ndim != 3
        or (b.dtype, b.shape[2])
        not in ((np.dtype("int16"), 3), (np.dtype("float32"), 4))
        or not b.flags.c_contiguous
        or len(b) % 32
    ):
        raise ValueError(
            "library_coords must be contiguous tile32 int16[...,3] or float32[...,4]"
        )
    if tile not in (1, -1):
        raise ValueError("Metal requires an explicitly declared tiled library layout")
    if not np.all(np.isfinite(b)):
        raise ValueError("nonfinite library coordinates")
    if (
        counts.dtype != np.uint8
        or counts.ndim != 2
        or len(counts) > len(b)
        or np.any(counts[:, 0] > b.shape[1])
        or np.any(counts[:, 0] == 0)
    ):
        raise ValueError("invalid library count vector")
    if (
        r.dtype != np.float32
        or h.dtype != np.float32
        or r.ndim != 2
        or r.shape != h.shape
        or r.shape[0] != r.shape[1]
        or not 1 <= len(r) <= 8
    ):
        raise ValueError(
            "interaction matrices must be matching float32 square arrays with at most 8 types"
        )
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(h)) or np.any(r < 0):
        raise ValueError("invalid interaction matrices")
    if counts.shape[1] > len(r) or np.any(types < 0) or np.any(types >= len(r)):
        raise ValueError("feature numbering exceeds interaction matrices")
    if not dense and (
        pq.dtype != np.int32
        or pb.dtype != np.int32
        or pq.ndim != 1
        or pb.shape != pq.shape
    ):
        raise ValueError("pair indices must be matching int32 vectors")
    if not dense and (
        np.any(pq < 0)
        or np.any(pq >= len(q))
        or np.any(pb < 0)
        or np.any(pb >= len(counts))
    ):
        raise ValueError("pair index out of range")
    for row, nreal in enumerate(n):
        if np.any(types[row, :nreal] != 0):
            raise ValueError("shape atoms must have type zero")
    if color:
        if counts.shape[1] != len(r):
            raise ValueError("color screening requires feature counts")
        begin = (
            counts[:, 0].astype(np.int64)
            if offset == -1
            else np.full(len(counts), offset)
        )
        if (
            offset < -1
            or np.any(begin < counts[:, 0])
            or np.any(begin + counts[:, 1:].sum(axis=1) > b.shape[1])
        ):
            raise ValueError("color runs exceed declared library layout")
    return q, types, n, b, counts, pq, pb, r, h


def score_pairs(
    *,
    query_coords,
    query_types,
    query_n_atoms,
    library_coords,
    library_counts,
    pair_query_row,
    pair_library_row,
    interaction_r,
    interaction_p,
    out=None,
    pose_search=PoseSearch.FIELD,
    color=False,
    mixing=0.0,
    steps=25,
    lr_q=0.1,
    lr_t=0.4,
    start_mode=1,
    device_id=0,
    field_spacing_a=0.8,
    field_pad_a=8.0,
    color_field_min_sources=None,
    color_field_spacing_a=None,
    update_rule=UpdateRule.ADAGRAD_EXP,
    color_placement=ColorPlacement.AT_POSE,
    library_color_offset=0,
    library_tile_mode=1,
    topk=None,
    gid_base=0,
    max_rows_per_query=0,
    context=None,
    poses=None,
    check_self_overlap_bound=False,
    call_stats=None,
    **unsupported,
):
    if unsupported:
        raise NotImplementedError(
            "Unsupported Metal controls: " + ", ".join(sorted(unsupported))
        )
    if (
        PoseSearch(pose_search) != PoseSearch.FIELD
        or UpdateRule(update_rule) != UpdateRule.ADAGRAD_EXP
    ):
        raise NotImplementedError("Metal supports FIELD / ADAGRAD_EXP")
    if (
        start_mode not in (0, 1, 2)
        or type(start_mode) is bool
        or steps < 0
        or int(steps) != steps
    ):
        raise ValueError("invalid start mode or steps")
    if not 0 <= mixing <= 1 or not color and mixing != 0:
        raise ValueError("mixing must be in [0,1] and zero for shape-only")
    if not all(
        np.isfinite(x) and x > 0 for x in (lr_q, lr_t, field_spacing_a, field_pad_a)
    ):
        raise ValueError(
            "learning rates, spacing, and padding must be positive and finite"
        )
    placement = ColorPlacement(color_placement)
    q, qt, qn, b, counts, pq, pb, r, h = _validate(
        query_coords,
        query_types,
        query_n_atoms,
        library_coords,
        library_counts,
        pair_query_row,
        pair_library_row,
        interaction_r,
        interaction_p,
        color,
        library_color_offset,
        library_tile_mode,
    )
    # No pair list means a dense launch, query-major: row q * n + t is
    # query q against library row t. Nothing is built or uploaded for it.
    dense = pq is None
    n_pairs = len(q) * len(counts) if dense else len(pq)
    if out is not None and (
        out.dtype != np.float32
        or out.shape != (n_pairs, 20)
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise ValueError("out must be writable contiguous float32 [pairs,20]")
    if not n_pairs:
        return np.empty((0, 20), np.float32) if out is None else out
    owned = context is None
    ctx = context or create_context(device_id)
    try:
        with ctx.lock:
            ctx.last_optimizer_dispatches = []
            if ctx.closed or ctx.device_id != device_id:
                raise ValueError("Metal context is closed or belongs to another device")
            fullcounts = np.zeros((len(counts), len(r)), np.uint8)
            fullcounts[:, : counts.shape[1]] = counts
            params = _params(
                len(q),
                q.shape[1],
                len(counts),
                b.shape[1],
                len(r),
                b.dtype == np.int16,
                library_tile_mode == -1,
                color,
                library_color_offset,
                n_pairs,
                steps,
                (1, 4, 10)[start_mode],
                color and placement == ColorPlacement.IN_GRADIENT,
                mixing,
                lr_q,
                lr_t,
                field_pad_a,
            )
            # Both width and height define interaction eligibility. Standard
            # queries store nonzero off-diagonal widths with zero heights.
            offdiag = (
                (r > 0) & (h != 0) if ctx.color_range_policy == "active" else r != 0
            )
            np.fill_diagonal(offdiag, False)
            params[13] = int(not np.any(offdiag))
            grid, field = ctx.query(
                q, qt, qn.astype(np.uint32), r, h, params, field_spacing_a, field_pad_a
            )
            minimum = (
                ctx.color_field_min_sources
                if color_field_min_sources is None
                else color_field_min_sources
            )
            spacing = (
                ctx.color_field_spacing_a
                if color_field_spacing_a is None
                else color_field_spacing_a
            )
            if type(minimum) is not int or minimum < 0:
                raise ValueError(
                    "color_field_min_sources must be a nonnegative integer"
                )
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError("color_field_spacing_a must be positive and finite")
            color_fields = (
                ctx.color_query(q, qt, qn, r, h, params, minimum, spacing, field_pad_a)
                if poses is None
                else None
            )
            if poses is not None:
                ctx.last_color_fields = []
                ctx.color_field_build_wall_s = ctx.color_field_build_gpu_s = 0.0
            lb = ctx.put("library", b)
            cnt = ctx.put("counts", fullcounts)
            if dense:
                # Written on the device by `dense_pairs` (first job below);
                # the kernels then read them as they read an uploaded list.
                pqu = ctx.buffer("pairs_q", n_pairs * 4)
                pbu = ctx.buffer("pairs_b", n_pairs * 4)
            else:
                pqu = ctx.put("pairs_q", pq)
                pbu = ctx.put("pairs_b", pb)
            bs = ctx.buffer("library_self", len(counts) * 8)
            output = ctx.buffer("scores", n_pairs * 80)
            jobs = ([("dense_pairs", [pqu, pbu], _u(len(q), len(counts)), n_pairs, 128)]
                    if dense else [])
            jobs += [
                (
                    "library_self",
                    [lb, cnt, ctx.r, ctx.h, bs],
                    params,
                    len(counts) * 4,
                    128,
                )
            ]
            inputs = [
                ctx.q,
                ctx.qt,
                ctx.qn,
                lb,
                cnt,
                pqu,
                pbu,
                ctx.r,
                ctx.h,
                ctx.qs,
                bs,
            ]
            if poses is None:
                kernel = "optimize"
                if params[12] and ctx.specialize_color and params[13]:
                    kernel += "_diagonal"
                if params[12] and ctx.factor_color_torque:
                    kernel += "_factored"
                analytic_kernel = kernel
                extra, textures = [], ctx.textures
                if color_fields is not None:
                    mask, color_grid, color_field, color_textures, color_slots, offsets, row_masks = color_fields
                    kernel = f"optimize_color_fields_m{mask}"
                    extra = [color_grid, color_field, color_slots]
                ctx.last_optimizer_kernel = kernel
                color_textures = color_fields[3] if color_fields is not None else []
                offsets = color_fields[5] if color_fields is not None else np.zeros(len(qn)+1, np.uint32)
                ctx.last_optimizer_dispatches = []
                query_ordered = dense or not np.any(pq[1:] < pq[:-1])
                row_masks = color_fields[6] if color_fields is not None else None
                for begin, end, bound in _texture_batches(len(qn), ctx.textures, color_textures, offsets, row_masks):
                    batch_mask = int(row_masks[begin]) if row_masks is not None else 0
                    batch_kernel = f"optimize_color_fields_m{batch_mask}" if batch_mask else analytic_kernel
                    batch_params = params.copy()
                    batch_params[14:16] = [begin, end-begin]
                    if dense:
                        pair_begin, pair_end = begin * len(counts), end * len(counts)
                    else:
                        pair_begin, pair_end = (np.searchsorted(pq, [begin, end])
                                                if query_ordered else (0, len(pq)))
                    batch_params[9] = pair_end-pair_begin
                    batch_params[20] = pair_begin
                    ctx.last_optimizer_dispatches.append(dict(
                        query_begin=begin, query_end=end, bound_textures=len(bound),
                        kernel=batch_kernel, field_mask=batch_mask,
                        pairs_dispatched=int(pair_end-pair_begin)))
                    jobs.append((batch_kernel, inputs + [grid, field, output] + (extra if batch_mask else []),
                                 batch_params, int(pair_end-pair_begin)*4, 128, bound))
                kernels = {d["kernel"] for d in ctx.last_optimizer_dispatches}
                ctx.last_optimizer_kernel = kernels.pop() if len(kernels) == 1 else "mixed"
            else:
                poses = np.asarray(poses)
                if (
                    poses.dtype != np.float32
                    or poses.shape != (n_pairs, 7)
                    or not np.all(np.isfinite(poses))
                    or np.any(np.linalg.norm(poses[:, :4], axis=1) == 0)
                ):
                    raise ValueError(
                        "poses must be finite float32 [pairs,7] with nonzero quaternions"
                    )
                jobs.append(
                    (
                        "rescore",
                        inputs + [ctx.put("poses", poses), output],
                        params,
                        n_pairs * 4,
                        128,
                    )
                )
            error = ctx.put("error", _u(0))
            jobs.append(
                ("check_scores", [output, error], _u(n_pairs * 20), n_pairs * 20, 128)
            )
            # Self-overlap bound: opt-in, and nothing is dispatched when off.
            # UNVERIFIED: written on Linux, never compiled or run on Metal.
            bound = None
            if check_self_overlap_bound:
                from .. import scores as _scores

                # Metal computes self-overlaps in-kernel: the recompute
                # route's margin, read at call time.
                margin = np.float32(_scores.self_overlap_bound_margin(
                    _scores.ROUTE_RECOMPUTE))
                bound = ctx.put("self_bound", _u(0, 0xFFFFFFFF))
                jobs.append(
                    (
                        "count_self_overlap_bound",
                        [output, bound],
                        _u(n_pairs, int(np.array(margin).view(np.uint32))),
                        n_pairs,
                        128,
                    )
                )
            ctx.native.submit(jobs)
            if error.read(np.dtype("uint32"), [1])[0]:
                raise FloatingPointError("Metal scoring produced nonfinite values")
            if call_stats is not None:
                flagged, first = ((-1, -1) if bound is None else
                                  (int(x) for x in bound.read(np.dtype("uint32"), [2])))
                call_stats.update(
                    query_self_evaluations=int(ctx.last_query_self_evaluations),
                    library_self_evaluations=int(len(counts)),
                    library_self_supplied=False,
                    self_bound_rows_checked=(-1 if bound is None else int(n_pairs)),
                    self_bound_rows_flagged=flagged,
                    self_bound_first_row=(-1 if bound is None or flagged == 0
                                          else first),
                )
            if topk is not None:
                if topk.context is not ctx:
                    raise ValueError(
                        "Metal sink belongs to a different execution context"
                    )
                topk.ingest(output, pq, pb, len(counts), int(gid_base))
                return None
            values = output.read(np.dtype("float32"), [n_pairs, 20])
            if out is not None:
                out[:] = values
                return out
            return values
    except BaseException:
        ctx.query_key = None
        ctx.color_cache = None
        ctx.last_color_fields = []
        ctx.grids.clear()
        ctx.textures.clear()
        raise
    finally:
        if owned:
            ctx.close()


class MetalTopK:
    wants_dense = False

    def __init__(self, k, level, rank_col, context):
        if int(k) != k or k <= 0 or rank_col not in range(20):
            raise ValueError("invalid K or ranking column")
        self.k, self.rank_col, self.context = int(k), int(rank_col), context
        self.device_level = level
        self.rows_offered = 0
        self.states = []
        self._mols = None

    @property
    def handle(self):
        return self

    def prepare(self, n_queries, device_id, max_block_conformers=0):
        if device_id != self.context.device_id:
            raise ValueError("Metal sink device mismatch")
        self.states = [None] * n_queries
        self.rows_offered = 0

    def before_block(self, block):
        if self.device_level == "molecule" and block.mol_id is None:
            raise ValueError("molecule ranking requires molecule IDs")
        self._mols = block.mol_id

    def add_block(self, block, scores):
        self.rows_offered += block.n_conformers

    def ingest(self, scores, pq, pb, nlib, gid_base):
        """Fold one complete query-major block into the per-query state.

        ``pq``/``pb`` is the block's pair list, which must be the dense
        query-major one, or ``None``/``None`` for a dense launch. Either
        way row ``qi * nlib + t`` is library row ``t``, which is what
        ``join_rows`` computes, so no pair list is bound.
        """
        if gid_base < 0 or gid_base + nlib > 2**63 - 1:
            raise ValueError("GID range exceeds signed 64-bit output")
        nq = len(self.states)
        if (pq is None) != (pb is None):
            raise ValueError("pass both halves of the pair list, or neither")
        if pq is not None and (
            len(pq) != nq * nlib
            or not np.array_equal(pq, np.repeat(np.arange(nq), nlib))
            or not np.array_equal(pb, np.tile(np.arange(nlib), nq))
        ):
            raise ValueError("Metal top-K requires a complete query-major pair block")
        ctx = self.context
        mols = (
            np.arange(nlib, dtype=np.uint64)
            if self._mols is None
            else np.asarray(self._mols, dtype=np.uint64)
        )
        mb = ctx.put("molecule_ids", mols)
        for qi in range(nq):
            old = self.states[qi]
            oldn = old[0] if old else 0
            n = nlib + oldn
            joined = ctx.buffer("joined_scores", n * 80)
            ids = ctx.buffer("joined_ids", n * 8)
            mids = ctx.buffer("joined_mids", n * 8)
            dummy = ctx.buffer("empty", 80)
            oldbufs = list(old[1:]) if old else [dummy] * 3
            jobs = [
                (
                    "join_rows",
                    [scores, mb] + oldbufs + [joined, ids, mids],
                    _u(nlib, oldn, qi * nlib, gid_base & 0xFFFFFFFF, gid_base >> 32),
                    n,
                    128,
                )
            ]
            a = ctx.buffer("sort_a", n * 4)
            b = ctx.buffer("sort_b", n * 4)
            jobs.append(("init_indices", [a], _u(n), n, 128))
            for molecule in [1, 0] if self.device_level == "molecule" else [0]:
                width = 1
                while width < n:
                    jobs.append(
                        (
                            "merge_indices",
                            [a, b, joined, ids, mids],
                            _u(n, width, self.rank_col, molecule),
                            n,
                            128,
                        )
                    )
                    a, b = b, a
                    width *= 2
                if molecule:
                    jobs.append(
                        (
                            "deduplicate",
                            [a, joined, mids],
                            _u(n, 0, self.rank_col, 1),
                            n,
                            128,
                        )
                    )
            keep = min(self.k, n)
            # Allocate fresh retained state: the old buffers remain inputs to join.
            selected = ctx.native.allocate(keep * 80)
            selectedids = ctx.native.allocate(keep * 8)
            selectedmids = ctx.native.allocate(keep * 8)
            jobs.append(
                (
                    "gather",
                    [a, joined, ids, mids, selected, selectedids, selectedmids],
                    _u(n, keep),
                    keep,
                    128,
                )
            )
            ctx.native.submit(jobs)
            self.states[qi] = (keep, selected, selectedids, selectedmids)

    def counters(self):
        return dict(rows_offered=self.rows_offered)

    def finalize(self):
        results = []
        for state in self.states:
            if state is None:
                results.append((np.empty((0, 20), np.float32), np.empty(0, np.int64)))
                continue
            n, scores, ids, _ = state
            rows = scores.read(np.dtype("float32"), [n, 20])
            gids = ids.read(np.dtype("int64"), [n])
            valid = np.isfinite(rows[:, self.rank_col])
            results.append((rows[valid], gids[valid]))
        return results


def make_topk(*, k, level, rank_col, context):
    return MetalTopK(k, level, rank_col, context)


def launch_conformers(n_queries, k, max_width, *, device_id=0):
    """Conservative shared-memory working set, including sorting scratch."""
    budget = (
        int(extension().device_info(device_id)["recommended_working_set_bytes"]) // 2
    )
    retained = int(n_queries) * int(k) * 96 * 2
    reserve = 128 << 20  # Query fields and driver/pipeline overhead.
    per_row = int(max_width) * 6 + int(n_queries) * 88 + 128
    rows = (budget - retained - reserve) // max(1, per_row)
    if rows < 32:
        raise MemoryError("Metal top-K/query state exceeds working-set budget")
    return min(65536, rows // 32 * 32)

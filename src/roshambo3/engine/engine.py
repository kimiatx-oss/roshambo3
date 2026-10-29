"""The screen loop: one device, a source, a plan, a sink.

The engine owns the things that must be allocated once and reused -- the pinned
buffer ring, the reused zero type plane, the device arena -- and drives the loop
that keeps the GPU fed. It does not own the corpus (that is the source) or the
result (that is the sink).

**Multi-GPU is the caller's job, not this class's.** Run one engine per device
over disjoint block ranges and merge with :meth:`TopK.merge`. ROSHAMBO2 split
work inside C++ under OpenMP, in two places that disagreed, and had to forbid
the device top-K on more than one GPU because its state is per-device with its
own atomics. Splitting here removes all three problems, and the determinism
argument survives it: the answer is a function of the set of conformers screened,
not of how they were divided.
"""

from __future__ import annotations

import time
import operator
import platform
import threading
from functools import wraps
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from .._cpu_platform import scoped_qos

import numpy as np

from .. import backend as backend_registry
from ..backend.spec import ColorPlacement, PoseSearch, UpdateRule
from .. import scores as _scores
from ..scores import N_SCORE_COLUMNS, check_self_overlap_bound
from .buffers import PinnedPool
from .config import CPUWorkerOptions, ScreenConfig, unsupported_cells
from .sink import HostTopK, MoleculeTopK, Sink, TopK


def _clear_worker_caches_on_failure(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self._worker_query_cache = None
            self._worker_self_cache = None
            raise
    return guarded


@dataclass
class ScreenStats:
    """Where the wall went. The decomposition matters more than the total."""

    wall_s: float = 0.0
    producer_wait_s: float = 0.0
    """Seconds the loop was blocked waiting for a decoded block. This is the
    headline host number -- it is the part of the wall the GPU spent idle."""
    launch_s: float = 0.0
    sink_s: float = 0.0
    n_blocks: int = 0
    n_conformers: int = 0
    n_rows: int = 0
    device_stages: dict = field(default_factory=dict)
    arena: list = field(default_factory=list)
    pinned: bool = False
    cpu_scan: dict = field(default_factory=dict)

    @property
    def conformers_per_s(self) -> float:
        return self.n_conformers / self.wall_s if self.wall_s > 0 else 0.0


def _default_config():
    """A pristine `ScreenConfig`, built once. The yardstick for "was this
    knob chosen, or is it just what a CUDA launch takes by default?" """
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        from .config import ScreenConfig
        _DEFAULT_CONFIG = ScreenConfig()
    return _DEFAULT_CONFIG


_DEFAULT_CONFIG = None


def _resolve_public_shape_field_precision(policy, *, mode, shape_backend,
                                          shape_grain, shape_width,
                                          machine=None):
    """Resolve the public precision policy once, before query preparation.

    ``auto`` currently selects FP32 on all routes: matched native ARM tests
    favor prepared FP32 over native-half interpolation. Explicit low-level
    half requests retain their feature/range checks and actual fallback stamps.
    """
    if policy not in ("auto", "fp32"):
        raise ValueError(
            "shape_field_interpolation must be 'auto' or 'fp32'")
    return "fp32", "fp32"


def _check_returned_rows(result, queries) -> None:
    """Refuse a returned row whose self-overlaps cannot describe its pair.

    Covers the rows a sink returns -- the top-K sinks' per-query rows and a
    dense result's whole matrix. The device backends compute self-overlaps
    in-kernel, so for them this is a guard against bugs, not data. A
    `Callback` sink hands rows to the caller as they arrive and returns none.
    The CPU worker route checks every scored row natively.
    Every backend here computes its self-overlaps from the scored
    coordinates, so the rows take the recompute route's margin.
    """
    names = getattr(queries, "names", None)

    def label(i):
        return (f"query {names[i]!r}" if names is not None and i < len(names)
                else f"query {i}")

    scores = getattr(result, "scores", None)
    gids = getattr(result, "gids", None)
    if isinstance(scores, np.ndarray) and scores.ndim == 3:
        for i in range(scores.shape[0]):
            check_self_overlap_bound(scores[i], gids, query=label(i),
                                     route=_scores.ROUTE_RECOMPUTE)
        return
    if not isinstance(result, (list, tuple)):
        return
    for i, item in enumerate(result):
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        check_self_overlap_bound(item[0], item[1], query=label(i),
                                 route=_scores.ROUTE_RECOMPUTE)


#: ``library_self_evaluation_counter_source`` values (worker CPU scan). The
#: two ``*_recomputed_at_binding`` sources name a scan in which some block's
#: library color self terms were evaluated while binding its prepared self
#: handle rather than read from a validated sidecar; those evaluations are
#: counted in ``library_color_self_evaluations``.
COUNTER_SOURCE_NATIVE = "native"
COUNTER_SOURCE_NATIVE_AND_COLOR_RECOMPUTED_AT_BINDING = (
    "native_and_python_counted_color_recomputed_at_binding")
COUNTER_SOURCE_PYTHON_SUPPLIED = "python_supplied_values"
COUNTER_SOURCE_PYTHON_COLOR_RECOMPUTED_AT_BINDING = (
    "python_counted_color_recomputed_at_binding")


def _binding_color_self_evaluations(backend, handle) -> int:
    """Library color self terms evaluated to make a worker block's prepared
    self handle, rather than read from a validated sidecar."""
    if handle is None:
        return 0
    if hasattr(handle, "color_self_binding_evaluations"):  # shape / rescore
        return int(handle.color_self_binding_evaluations)
    if isinstance(handle, backend.PreparedComboSelfOverlaps):
        library = handle.library
        # Rows, not the native pairwise term count (color_terms).
        return (0 if library.library_color_from_sidecar or not library.color_prepared
                else int(library.n_library))
    raise TypeError(f"unsupported prepared self-overlap handle {type(handle).__name__}")


#: The physical-to-reduced shape conversion each mode applies to a stored
#: VAA value: shape and rescore multiply by the float32
#: reciprocal; combo divides in float64 and rounds once.
_STORED_VAA_CONVERSION = {
    "shape": "float32(VAA)*float32(1/SHAPE_PREFACTOR)",
    "rescore": "float32(VAA)*float32(1/SHAPE_PREFACTOR)",
    "combo": "float32(float64(VAA)/float64(SHAPE_PREFACTOR))",
}


def _stored_vaa_conversion(mode, route):
    """The conversion a scan actually applied to stored VAA, naming its
    route, or None when it used no stored values."""
    if route is None:
        return None
    return f"{_STORED_VAA_CONVERSION[mode]} on {route}"


def _differs(a, b) -> bool:
    """Value inequality that survives numpy arrays turning up in a launch dict."""
    if a is b:
        return False
    try:
        import numpy as _np
        if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
            return not (_np.shape(a) == _np.shape(b) and bool(_np.all(a == b)))
    except Exception:
        pass
    try:
        return bool(a != b)
    except Exception:
        return True


_CPU_WORKER_CONFIG_FIELDS = {
    "cpu_worker", "color", "mixing", "pose_search", "color_placement",
    "steps", "lr_t", "lr_q", "start_mode", "update_rule",
    "budget_override", "field_spacing_a", "field_pad_a",
}


def _cpu_worker_config_compatible(config) -> bool:
    """Implicit fast dispatch must not steal unsupported ordinary requests."""
    try:
        if (PoseSearch(config.pose_search) is not PoseSearch.FIELD or
                ColorPlacement(config.color_placement) not in
                (ColorPlacement.IN_GRADIENT, ColorPlacement.AT_POSE) or
                UpdateRule(config.update_rule) is not UpdateRule.ADAGRAD_EXP or
                isinstance(config.start_mode, (bool, np.bool_)) or
                operator.index(config.start_mode) != 1 or config.field_pad_a != 8.0):
            return False
    except (TypeError, ValueError):
        return False
    return all(setting.name in _CPU_WORKER_CONFIG_FIELDS or not _differs(
        getattr(config, setting.name), getattr(_default_config(), setting.name))
        for setting in fields(ScreenConfig))


class Engine:
    """Screens a source against a query set on one device.

        engine = Engine(device_id=0)
        stats, result = engine.run(source, queries, config, sink)

    ``queries`` is a :class:`QuerySet`: a small, already-prepared batch of query
    conformers padded to one width.
    """

    def __init__(self, *, device_id: int = 0, pinned: bool = True,
                 pool_buffers: int = 4, stage_timing: bool = False,
                 backend: "str | object | None" = None,
                 backend_options: Optional[dict] = None):
        # Per launch shape, what a DEFAULT ScreenConfig would emit -- the
        # reference for deciding whether an inert knob was chosen or inherited.
        # See `_launch_args_for_backend`.
        self._default_launch_args: dict = {}
        self._dropped_launch_keys: set = set()
        #: Real atom counts of the query set, set at the start of each launch.
        #: Defaults to a value that selects the field path, so a caller poking
        #: `_backend_screen_defaults` directly gets the library-size answer.
        self._query_n_real = (1 << 30,)
        self._backend_defaults_applied: set = set()
        # Resolved ONCE, here, and held -- not looked up per launch, so a
        # screen cannot change backend halfway through, and not a module-level
        # alias, which is what it was and what made `score_pairs` unsubstitutable.
        # A module may be passed directly, which is what lets a test or a
        # measurement arm supply one without registering it.
        self.backend = (backend if hasattr(backend, "score_pairs")
                        else backend_registry.select(backend))
        self._backend_options = dict(backend_options or {})
        if self._backend_options and not callable(getattr(self.backend, "create_context", None)):
            raise ValueError("selected backend does not accept context options")
        self.device_id = int(device_id)
        self.pinned = bool(pinned)
        self.pool_buffers = int(pool_buffers)
        self.stage_timing = bool(stage_timing)
        self._pool: Optional[PinnedPool] = None
        self._score_buf: Optional[np.ndarray] = None
        self._worker_self_cache = None
        self._worker_query_cache = None

    def make_topk(self, k: int, *, level: str = "conformer", rank_col: int = 0):
        """Construct a ranking request for this Engine's backend."""
        if level not in ("conformer", "molecule"):
            raise ValueError("level must be conformer or molecule")
        factory = getattr(self.backend, "make_topk", None)
        if factory is not None:
            return factory(k=k, level=level, rank_col=rank_col, context=self._context())
        if level == "molecule":
            return MoleculeTopK(k=k, rank_col=rank_col)
        return (HostTopK if getattr(self.backend, "__name__", "") == "roshambo3.backend.cpu" else TopK)(
            k=k, rank_col=rank_col)

    def _context(self):
        if not hasattr(self, "_backend_context"):
            factory = getattr(self.backend, "create_context", None)
            self._backend_context = factory(self.device_id, **self._backend_options) if factory else None
        return self._backend_context

    # -- lifecycle ---------------------------------------------------------

    def _ensure_pool(self, n_elems: int, n_buffers: int) -> PinnedPool:
        """The pool, grown if this screen needs wider or more buffers.

        ``n_buffers`` is not a preference. The producer keeps ``prefetch_depth``
        blocks in flight and the consumer holds one more, so a pool smaller than
        ``depth + 1`` deadlocks the moment anything stops releasing. That
        coupling used to be implicit -- the defaults happened to satisfy it
        exactly (4 buffers, depth 3), and ``api.screen`` exposed
        ``prefetch_depth`` but not ``pool_buffers``, so raising the depth alone
        was enough to break it.
        """
        if (self._pool is None or self._pool.n_elems < n_elems
                or len(self._pool) < n_buffers):
            # CPU has no host/device DMA and must not allocate via CUDA.
            pinned = (self.pinned and getattr(self.backend, "__name__", "")
                      == "roshambo3.backend.cuda")
            self._pool = PinnedPool(n_buffers, n_elems, pinned=pinned)
        return self._pool

    def _score_buffer(self, n_rows: int) -> np.ndarray:
        """A reused host score buffer.

        A screen is thousands of launches of the same shape, and allocating and
        zeroing a fresh ``[n_rows, 20]`` float32 each time is 160 MB per launch
        at a 250k block and 8 queries. The zeroing was never needed: the copy
        back overwrites every row the kernel owns.
        """
        need = n_rows * N_SCORE_COLUMNS
        if self._score_buf is None or self._score_buf.size < need:
            self._score_buf = np.empty(need, np.float32)
        return self._score_buf[:need].reshape(n_rows, N_SCORE_COLUMNS)

    def _backend_screen_defaults(self, args: dict, n_library: int) -> dict:
        """Let the backend fill in its own recommended configuration.

        A knob already present in `args` is left alone -- the caller's
        `ScreenConfig` wins, always. This only supplies what nothing else set,
        which is what makes it a default rather than an override.

        The CUDA path is untouched: a backend that publishes no
        `default_screen_knobs` gets `args` back unchanged.
        """
        fn = getattr(self.backend, "default_screen_knobs", None)
        if fn is None:
            return args
        spacing = args.get("field_h", 0.0) or args.get("lattice_spacing", 0.8)
        # The MEDIAN real atom count, not the padded width: a backend choosing
        # between an O(N_A N_B) kernel and one that deletes the N_A factor needs
        # to know how big N_A actually is, and padding does not tell it.
        extra = fn(int(n_library), float(spacing) or 0.8,
                   int(np.median(np.asarray(self._query_n_real))))
        if not extra:
            return args
        out = dict(args)
        for k, v in extra.items():
            out.setdefault(k, v)
            self._backend_defaults_applied.add(k)
        return out

    def _launch_args_for_backend(self, args: dict, shape: tuple,
                                 mk_default) -> dict:
        """Drop launch knobs this backend refuses -- but only at their default.

        `ScreenConfig.for_width` emits the argument list a CUDA launch takes,
        because that is the only launch there was when it was written. Handing
        all of it to a backend with no SM makes a screen die on
        `smem_carveout` before it scores a pair, which is why the CPU backend
        would be unreachable through `Engine` at all.

        Dropping them wholesale would be exactly the silent-ignore this
        boundary exists to refuse. So the test is whether the value was
        *chosen*: compare against what a default `ScreenConfig` emits for this
        same launch shape. Equal means nobody asked for it and it describes
        hardware that is not here -- drop it. Different means the caller moved
        it, so pass it through and let the backend refuse it by name, which is
        the behavior the contract wants.

        A backend that publishes no `inert_launch_knobs` gets the dict
        unchanged, so the CUDA path is untouched.
        """
        names = getattr(self.backend, "inert_launch_knobs", None)
        if names is None:
            return args
        inert = names()
        if not inert:
            return args
        ref = self._default_launch_args.get(shape)
        if ref is None:
            ref = self._default_launch_args[shape] = mk_default()
        out = {}
        for k, v in args.items():
            if k not in inert:
                out[k] = v
            elif k not in ref or _differs(ref[k], v):
                out[k] = v          # the caller moved it; the backend says no
            else:
                self._dropped_launch_keys.add(k)
        return out

    def close(self) -> None:
        """Hand cached device memory back to the driver."""
        self._worker_self_cache = None
        self._worker_query_cache = None
        context = getattr(self, "_backend_context", None)
        if context is not None:
            context.close()
            del self._backend_context
        if self.backend.available():
            trim = getattr(self.backend.extension(), "arena_trim", None)
            if trim is not None:
                trim()

    # -- the loop ----------------------------------------------------------

    def run(self, source, queries, config: Optional[ScreenConfig] = None,
            sink: Optional[Sink] = None, *, progress=None,
            canceled=None) -> tuple[ScreenStats, Any]:
        """Run a screen, optionally observing submitted work at block boundaries.

        ``progress(stats)`` sees cumulative submitted conformers, not necessarily
        completed device execution. ``canceled()`` may stop before the next
        launch by returning true; the normal stream cleanup still drains decode
        work and releases buffers. Both callbacks are optional.
        """
        config = config or ScreenConfig()
        # R3G fixes the coordinate and count projection when the source is
        # built. Reject a mismatched scorer request before feature-presence
        # checks can turn this configuration error into a misleading message.
        source_color = getattr(source, "color", None)
        if (source_color is not None and
                bool(config.color) != bool(source_color)):
            raise ValueError("config/source color layouts differ")
        sink = sink if sink is not None else self.make_topk(k=1000)

        if config.cpu_worker is not None:
            return self._run_cpu_worker_optin(
                source, queries, config, sink, progress=progress,
                canceled=canceled)
        # Switching away from the worker route releases its retained generation
        # rather than leaving unaccounted worker state behind an ordinary run.
        self._worker_query_cache = None
        self._worker_self_cache = None

        # Check every bucket this screen will touch, before the first
        # launch. The default build carries only the shipped configuration's
        # cells, and the failure that allows is a bucket no cell covers --
        # discovered mid-screen, as a wall of template parameters, after a
        # corpus has been opened and work has been done. A throw during a
        # production run is worse than a slow build.
        #
        # Every bucket, not the first one that fails: a screen that dies on its
        # ninth group after eight minutes has told the user one bucket's worth
        # of what is wrong with their build.
        resolver = getattr(self.backend, "screen_args", None)
        if resolver is not None:
            widths = getattr(source, "bucket_widths", ())
            widths = widths() if callable(widths) else widths
            for width in widths or (int(queries.coords.shape[1]),):
                resolver(config, int(width), int(width), int(width))
        else:
            self._check_cells_for(source, config, queries)

        # Only reached for a timed run. Asking for the extension unconditionally
        # made every engine run require a CUDA build even when the backend was
        # substituted, which is why the failure-lifecycle tests all needed a GPU
        # to say anything about host-side buffer ownership.
        ext = self.backend.extension() if self.stage_timing and resolver is None else None
        timing_was_enabled = False
        if self.stage_timing and ext is not None:
            # Saved and restored, not switched off: this flag is process-global,
            # and a bench script that enabled it around a screen used to have it
            # silently turned off by the screen.
            # The extension reports the flag back through stage_timings(),
            # which is the only way to read process-global state that Python
            # does not own.
            timing_was_enabled = bool(ext.stage_timings().get("enabled", False))
            ext.stage_timing_enable(True)
            ext.stage_timing_reset()

        try:
            return self._run(source, queries, config, sink, ext,
                             timing_was_enabled, progress, canceled)
        except BaseException:
            # Setup between enabling stage timing and entering the loop can
            # raise -- a pinned allocation that fails, a sink whose device state
            # will not fit. Global state must not survive it.
            if self.stage_timing and ext is not None:
                ext.stage_timing_enable(timing_was_enabled)
            raise

    @_clear_worker_caches_on_failure
    def _run_cpu_worker_optin(self, source, queries, config, sink, *,
                             progress=None, canceled=None):
        """Translate an explicit public request to the existing bounded scan.

        Sink objects supply k/rank/level only; their CUDA or host state is never
        prepared or mutated. The scan returns only after successful reduction
        and cleanup. Library normalization uses prepared descriptor-bound
        values by default; explicit recompute/fallback is reported in stats.
        """
        from .r3g_source import R3gSource

        options = config.cpu_worker
        if type(options) is not CPUWorkerOptions:
            raise TypeError("cpu_worker must be a CPUWorkerOptions")
        if options.shape_field_interpolation not in ("auto", "fp32"):
            raise ValueError(
                "shape_field_interpolation must be 'auto' or 'fp32'")
        if getattr(self.backend, "__name__", "") != "roshambo3.backend.cpu":
            raise ValueError("cpu_worker requires Engine(backend='cpu')")
        if self.device_id != 0:
            raise ValueError("cpu_worker requires device_id=0")
        if self.stage_timing:
            raise NotImplementedError("cpu_worker does not support stage_timing")
        if type(sink) not in (TopK, MoleculeTopK, HostTopK):
            raise NotImplementedError(
                "cpu_worker requires an exact TopK, MoleculeTopK or HostTopK "
                "request; dense, callback and subclass sinks are unsupported")
        if not isinstance(source, R3gSource):
            raise TypeError("cpu_worker source must be an R3gSource")
        if source.tile:
            raise ValueError("cpu_worker requires R3gSource(tile=False)")
        if bool(config.color) != bool(source.color):
            raise ValueError("cpu_worker config/source color layouts differ")
        if options.self_overlap_policy not in ("prepared", "recompute"):
            raise ValueError("cpu_worker self_overlap_policy must be 'prepared' or 'recompute'")
        if PoseSearch(config.pose_search) is not PoseSearch.FIELD:
            raise NotImplementedError("cpu_worker requires pose_search=FIELD")
        if ColorPlacement(config.color_placement) not in (ColorPlacement.IN_GRADIENT, ColorPlacement.AT_POSE):
            raise NotImplementedError("cpu_worker requires AT_POSE or IN_GRADIENT")
        if UpdateRule(config.update_rule) is not UpdateRule.ADAGRAD_EXP:
            raise NotImplementedError("cpu_worker requires update_rule=ADAGRAD_EXP")
        try:
            four_starts = (not isinstance(config.start_mode, (bool, np.bool_))
                           and operator.index(config.start_mode) == 1)
        except TypeError:
            four_starts = False
        if not four_starts:
            raise ValueError("cpu_worker color16 requires start_mode=1 (four starts)")
        if config.field_pad_a != 8.0:
            raise ValueError("cpu_worker requires field_pad_a=8.0")
        mode = ("shape" if not config.color else
                "combo" if ColorPlacement(config.color_placement) is ColorPlacement.IN_GRADIENT
                else "rescore")
        field_storage, field_interpolation = _resolve_public_shape_field_precision(
            options.shape_field_interpolation, mode=mode,
            shape_backend="pipelined_field", shape_grain="library_atom",
            shape_width=8)
        # Resolve AUTO once per scan, outside the block loop. Explicit
        # per-option values remain authoritative; the lower-level
        # run_worker_owned_cpu API retains its conservative defaults.
        factor_canonical_torque = options.factor_canonical_torque
        if factor_canonical_torque is None:
            factor_canonical_torque = mode == "combo"
        canonical_color_kernel = options.canonical_color_kernel
        if canonical_color_kernel is None:
            canonical_color_kernel = mode == "combo"
        optimizer_exp_mode = options.optimizer_exp_mode
        if optimizer_exp_mode is None:
            optimizer_exp_mode = "rawbits_unclamped" if mode == "combo" else "baseline"
        if optimizer_exp_mode not in ("baseline", "rawbits", "rawbits_unclamped"):
            raise ValueError("optimizer_exp_mode must be baseline or rawbits (or rawbits_unclamped)")
        for setting in fields(ScreenConfig):
            if setting.name not in _CPU_WORKER_CONFIG_FIELDS and _differs(
                    getattr(config, setting.name),
                    getattr(_default_config(), setting.name)):
                raise NotImplementedError(
                    f"cpu_worker does not support ScreenConfig.{setting.name}")
        from .config import _default_cpu_workers
        worker_count = options.workers if options.workers is not None else _default_cpu_workers()
        stats, results = self.run_worker_owned_cpu(
            source, queries, config, workers=worker_count,
            k=sink.k, rank_column=sink.rank_col,
            level="molecule" if type(sink) is MoleculeTopK else "conformer",
            memory_budget_bytes=options.memory_budget_bytes,
            topk_state_budget_bytes=options.topk_state_budget_bytes,
            color_grain=16, shape_backend="pipelined_field",
            shape_grain="library_atom", shape_width=8, canceled=canceled, progress=progress,
            self_overlap_policy=options.self_overlap_policy,
            factor_canonical_torque=factor_canonical_torque,
            optimizer_exp_mode=optimizer_exp_mode,
            canonical_color_kernel=canonical_color_kernel,
            final_start_policy=options.final_start_policy,
            field_storage=field_storage,
            field_interpolation=field_interpolation,
            **({"cpu_qos": options.cpu_qos} if options.cpu_qos != "inherit" else {}))
        stats.cpu_scan["shape_field_interpolation_policy"] = (
            options.shape_field_interpolation)
        stats.cpu_scan["shape_field_storage_request"] = field_storage
        stats.cpu_scan["shape_field_interpolation_request"] = field_interpolation
        return stats, [(result[0], result[1]) for result in results]

    def _check_cells_for(self, source, config, queries) -> None:
        """Refuse a screen whose buckets need cells this binary lacks.

        Reads widths from the format-3 source's group table. The launcher's
        ``!launched_`` ladder remains the backstop for any width the source
        cannot enumerate.
        """
        widths = getattr(source, "bucket_widths", None)
        if callable(widths):
            widths = widths()
        if not widths:
            return
        problems = []
        for L in sorted({int(w) for w in widths}):
            try:
                # The SAME arguments the loop below passes, so the tuple
                # checked here is the tuple that will launch. Resolving with a
                # different query panel would check a different stage width --
                # which is not in the cell table today, and is exactly the kind
                # of drift that makes a pre-flight check quietly stop matching
                # the flight.
                resolved = config.for_width(
                    L, device_id=self.device_id, L_shape=L,
                    query_atoms=int(queries.coords.shape[1]),
                    n_features=int(queries.interaction_r.shape[0]))
            except Exception:
                # A width this configuration cannot resolve at all is a
                # different failure and belongs to whoever raised it.
                continue
            problems.extend(unsupported_cells(resolved, bucket=L))
        if problems:
            raise ValueError(
                "this build carries no dispatch cell for "
                f"{len(problems)} of the resolved settings this screen needs:\n"
                + "\n".join(f"  - {p}" for p in problems))

    def _run(self, source, queries, config, sink, ext, timing_was_enabled,
             progress=None, canceled=None):
        metal_context = self._context() if getattr(self.backend, "screen_args", None) else None
        gpu_start = metal_context.native.gpu_seconds if metal_context is not None else 0.0
        # The pool must exist before the source starts decoding. It is sized
        # from the largest block the source will yield, and held to at least
        # one buffer more than the source keeps in flight.
        pool = self._ensure_pool(
            source.max_block_elements,
            max(self.pool_buffers, int(getattr(source, "prefetch_depth", 0)) + 1))
        source._pool = pool

        # The top-K's capacity is fixed on its first launch, so every launch
        # must declare the same per-query bound: the largest block, not this
        # one. Getting this wrong is not a silent overflow -- the state refuses
        # the first block that would not fit -- but it does mean a screen that
        # started fine can fail partway through.
        max_rows_per_query = getattr(source, "max_block_conformers", 0)

        sink.prepare(queries.n_conformers, self.device_id,
                     max_rows_per_query)
        # Routed on whether the sink OWNS DEVICE STATE, not on which of
        # two concrete classes it is. `isinstance(sink, (TopK, MoleculeTopK))`
        # made every host-side reducer -- including a CPU backend's -- look
        # like a dense sink that also wanted a device handle. `handle` is the honest
        # question: a sink that has one is reduced on the device, and a sink
        # that does not gets the dense block. Asked after `prepare`, because
        # that is when a device state exists to have a handle.
        topk_handle = getattr(sink, "handle", None)
        device_sink = sink if topk_handle is not None else None
        # A sink that has to put something on the device before the launch --
        # MoleculeTopK needs the block's mol_id, because the reduction runs
        # inside the launch, on the dense block, before it is discarded.
        before_block = getattr(sink, "before_block", None)
        wants_dense = getattr(sink, "wants_dense", True)

        stats = ScreenStats(pinned=pool.is_pinned)
        t_wall = time.perf_counter()

        # Everything from here is wrapped, because a buffer that is not
        # returned is not merely leaked: the pool is cached on this Engine and
        # only ever grows, so the loss is permanent, and after `pool_buffers`
        # failures the next screen blocks on an empty pool while the producer's
        # in-flight workers block on the same one -- with the executor's
        # shutdown(wait=True) joining them and the original exception never
        # printed. The failure to surface the error is worse than the leak.
        stream = source.blocks()
        try:
            for block in stream:
                try:
                    if canceled is not None and canceled():
                        raise InterruptedError("Screen canceled at a block boundary")
                    # Inside the try from the first statement: this block's
                    # buffer is already checked out, and `_drain` cannot see it
                    # -- the prefetcher handed it over and dropped its own
                    # reference.
                    n = block.n_conformers
                    # A dense launch's width is the count vector's row count
                    # (the backend derives every row from it), so it must be
                    # the block's: a wider count vector would score rows past
                    # the block and a dense top-K would admit them under the
                    # next block's gids.
                    if int(block.counts.shape[0]) != int(n):
                        raise ValueError(
                            f"block has n_conformers={n} but its count vector has "
                            f"{int(block.counts.shape[0])} rows; a dense launch "
                            f"scores every count row, so the two must agree")
                    n_rows = queries.n_conformers * n

                    resolver = getattr(self.backend, "screen_args", None)
                    if resolver is not None:
                        args = resolver(config, block.L,
                            getattr(block, "L_shape", 0) or block.L,
                            getattr(block, "color_offset", 0) or (getattr(block, "L_shape", 0) or block.L))
                        args["context"] = self._context()
                    else:
                        args = config.for_width(
                            block.L, device_id=self.device_id,
                            L_shape=getattr(block, "L_shape", 0) or block.L,
                            # Which layout the packer put this block in. A
                            # merged record's color runs start at each conformer's
                            # own real atom count, which no width can express.
                            library_color_offset=(
                                getattr(block, "color_offset", 0) or None),
                            # The stage width is chosen against the whole block's
                            # shared memory and the query record is part of it, so
                            # the policy is told which panel this launch carries
                            # rather than assuming one. The PADDED width, color
                            # plane included -- that is the launcher's `NmolA`.
                            query_atoms=int(queries.coords.shape[1]),
                            n_features=int(queries.interaction_r.shape[0]))
                        _L = int(block.L)
                        _Ls = int(getattr(block, "L_shape", 0) or block.L)
                        _qa = int(queries.coords.shape[1])
                        _nf = int(queries.interaction_r.shape[0])
                        self._query_n_real = queries.n_real
                        args = self._backend_screen_defaults(args, n)
                        args = self._launch_args_for_backend(
                            args, (_L, _Ls, _qa, _nf),
                            lambda: _default_config().for_width(
                                _L, device_id=self.device_id, L_shape=_Ls,
                                query_atoms=_qa, n_features=_nf))
                    out = (self._score_buffer(n_rows)
                           if (wants_dense and topk_handle is None) else None)
                    if before_block is not None:
                        before_block(block)

                    # The CPU wrapper revalidates the optimizer budget. Preserve
                    # an explicitly authored override after for_width resolves
                    # the numeric settings; device launchers do not accept it.
                    if (getattr(self.backend, "__name__", "") == "roshambo3.backend.cpu"
                            and config.budget_override):
                        args["budget_override"] = config.budget_override
                    t0 = time.perf_counter()
                    scores = self.backend.score_pairs(
                        query_coords=queries.coords,
                        query_types=queries.types,
                        query_n_atoms=queries.n_real,
                        library_coords=block.coords,
                        library_counts=block.counts,
                        # A dense launch -- every query against every
                        # row of the block, query-major. No pair index is
                        # built: the backend derives (query, library row) from
                        # the launch shape.
                        pair_query_row=None,
                        pair_library_row=None,
                        interaction_r=queries.interaction_r,
                        interaction_p=queries.interaction_p,
                        out=out,
                        device_id=self.device_id,
                        topk=device_sink,
                        gid_base=block.gid_base,
                        max_rows_per_query=max_rows_per_query or n,
                        # Which tile the block was packed in. It comes from
                        # the BLOCK, not from the ScreenConfig: the party that
                        # wrote the buffer is the only one that knows its
                        # layout, and a second copy of the answer on the config
                        # would be a second thing to keep in agreement.
                        library_tile_mode=int(getattr(block, "tile_mode", 1)),
                        **args)
                    stats.launch_s += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    sink.add_block(block, scores)
                    stats.sink_s += time.perf_counter() - t0
                finally:
                    if block.buffer is not None:
                        pool.release(block.buffer)

                stats.n_blocks += 1
                stats.n_conformers += n
                stats.n_rows += n_rows
                if progress is not None:
                    progress(stats)
        finally:
            # Closing the generator runs the producer's `with ThreadPoolExecutor`
            # exit here, at the raise site, rather than whenever the traceback
            # holding this frame is finally collected.
            # Nothing here may raise: it runs while an exception may already be
            # propagating, and a failure would replace the real error with a
            # bookkeeping one. `producer_wait_s` is not part of the Source
            # protocol -- a conforming custom source does not have it.
            try:
                stream.close()
            finally:
                stats.wall_s = time.perf_counter() - t_wall
                stats.producer_wait_s = float(
                    getattr(source, "producer_wait_s", 0.0) or 0.0)
                if self.stage_timing and ext is not None:
                    try:
                        stats.device_stages = dict(ext.stage_timings())
                        stats.arena = list(ext.arena_stats())
                    except Exception:
                        pass
                    ext.stage_timing_enable(timing_was_enabled)
        result = sink.finalize()
        _check_returned_rows(result, queries)
        context = getattr(self, "_backend_context", None)
        if self.stage_timing and context is not None:
            stats.device_stages = context.timings()
            stats.device_stages["gpu_s"] -= gpu_start
        return stats, result

    @_clear_worker_caches_on_failure
    @scoped_qos
    def run_worker_owned_cpu(
            self, source, queries, config: Optional[ScreenConfig] = None, *,
            workers: int = 1, k: int = 100, rank_column: int = 0,
            level: str = "conformer", block_descriptors=None,
            memory_budget_bytes: Optional[int] = None,
            topk_state_budget_bytes: Optional[int] = None,
            color_grain: int = 0, shape_backend: str = "scalar_analytic",
            shape_grain: str = "scalar", shape_width: int = 1,
            field_spacing_angstrom: Optional[float] = None,
            field_padding_angstrom: Optional[float] = None,
            field_storage: str = "fp32",
            field_interpolation: str = "fp32",
            color_field_backends=None, color_field_policy=None, canceled=None, progress=None,
            self_overlap_policy: str = "recompute",
            factor_canonical_torque: bool = False,
            optimizer_exp_mode: str = "baseline",
            canonical_color_kernel: bool = False,
            final_start_policy: str = "winner", cpu_qos: str = "inherit"):
        """Decode, prepare, score and reduce R3G groups in owning CPU workers.

        This explicit opt-in is separate from :meth:`run`: GPU staging,
        ordinary CPU dense output and callback sinks retain their existing
        behavior. The scan makes one indexed submission over physical R3G
        group descriptors. Each persistent worker reads and decodes a claimed group,
        converts it directly to the contiguous FP32 CPU layout, invokes the
        native mode-3 scorer with one internal thread, and offers completed
        rows to its shared scan-wide bounded top-K collector.

        ``block_descriptors`` may select an explicit subset of physical blocks
        returned by ``source.cpu_block_descriptors()``. Otherwise all selected
        source groups are scanned. ``memory_budget_bytes`` covers bounded
        worker scratch, query snapshots, descriptor objects and top-K state;
        file-backed mmap residency is OS-managed and is reported separately.
        Both budgets default to ``None`` (no configured cap), as do public
        CPUWorkerOptions. Explicit limits remain available
        for constrained deployments and reproducible experiments.

        The retained query path supports scalar analytic evaluation and the
        pipelined-field/library-atom/width-8 evaluator. Query handles are
        retained on this Engine as one bounded query/shape/build generation,
        with one handle per distinct spacing from selected ``L_field_width``.
        Matching later runs reuse them; query mutation, relevant shape/build
        changes, failure/cancellation and close invalidate them. Retained bytes
        are included in each run's memory estimate. The current
        native optimizer still performs its query-view setup per block.
        Direct calls retain FP32 field storage/interpolation by default.
        ``field_storage="fp16", field_interpolation="mixed_half_yz"`` is
        limited to combo pipelined-field/library-atom/width-8 and may resolve
        to native FP32 fallback when half arithmetic or the conservative
        finite-range guard is unavailable.
        The optional ``color_field_policy`` (or explicit
        ``color_field_backends`` matrix) is resolved once before block scoring;
        selected color fields are prepared once and the same native query
        handle is bound into each block's grain-16 call. Policy rows and
        timing qualification covers FP32 codes 1 and 3; explicit code 2 remains
        compatible but outside that qualification. Omitting both
        options keeps the analytic color path. On the retained grain-16
        conformer route, native bounds preflight is per target-type/pair-lane:
        any out-of-grid posed sample sends that lane's whole plane to the
        existing analytic SIMD path. Fallback sample/term counters are
        aggregate, not plane/lane-resolved. This partial policy path does not
        promote fields by default.
        ``self_overlap_policy="prepared"`` prepares self values on the owning
        worker's first decode and reuses them across this Engine's runs. Stored
        verified stored-frame VAA is accepted without precision certification;
        current-R/P color is computed once per binding.
        Missing/unsupported shape inputs explicitly fall back to recompute.
        Preparation is included in first-use scan wall and reported separately.
        Direct callers retain the explicit method's recompute default.
        Because the CPU entry is mode 3 with the native exponential-map
        optimizer, it requires ``PoseSearch.FIELD``,
        ``ColorPlacement.IN_GRADIENT`` and ``UpdateRule.ADAGRAD_EXP`` rather
        than silently accepting GPU/legacy settings the scorer cannot honor.
        """
        full_start = time.perf_counter()
        from ..reference.color import MODE_COMBO_OPTIMAL
        from .r3g_cpu import CPUPhysicalBlockDescriptor
        from .r3g_source import R3gSource
        if self_overlap_policy not in ("prepared", "recompute"):
            raise ValueError("self_overlap_policy must be 'prepared' or 'recompute'")
        if self_overlap_policy == "recompute":
            self._worker_self_cache = None

        if getattr(self.backend, "__name__", "") != "roshambo3.backend.cpu":
            raise ValueError(
                "run_worker_owned_cpu requires Engine(backend='cpu')")
        if not isinstance(source, R3gSource):
            raise TypeError("source must be an R3gSource")
        for name, value, minimum in (
                ("workers", workers, 1), ("k", k, 1),
                ("rank_column", rank_column, 0),
                ("memory_budget_bytes", memory_budget_bytes, 1),
                ("topk_state_budget_bytes", topk_state_budget_bytes, 1)):
            if value is None and name in ("memory_budget_bytes", "topk_state_budget_bytes"):
                continue
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            try:
                integer = operator.index(value)
            except TypeError as exc:
                raise ValueError(
                    f"{name} must be an integer >= {minimum}") from exc
            if integer < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            if name == "workers":
                workers = integer
            elif name == "k":
                k = integer
            elif name == "rank_column":
                rank_column = integer
            elif name == "memory_budget_bytes":
                memory_budget_bytes = integer
            else:
                topk_state_budget_bytes = integer
        if not 0 <= int(rank_column) < N_SCORE_COLUMNS:
            raise ValueError("rank_column must be in [0, 19]")
        if level not in ("conformer", "molecule"):
            raise ValueError("level must be 'conformer' or 'molecule'")
        if int(color_grain) not in (0, 11, 16):
            raise NotImplementedError(
                "worker-owned mode-3 scan currently supports retained color "
                "grains 0, 11, and 16 only")
        if canceled is not None and not callable(canceled):
            raise TypeError("canceled must be callable or None")
        if final_start_policy not in ("winner", "all_start"):
            raise ValueError("final_start_policy must be winner or all_start")
        if optimizer_exp_mode not in ("baseline", "rawbits", "rawbits_unclamped"):
            raise ValueError("optimizer_exp_mode must be baseline or rawbits (or rawbits_unclamped)")
        if (canonical_color_kernel or factor_canonical_torque or optimizer_exp_mode != "baseline") and int(color_grain) != 16:
            raise ValueError("analytic color options require color_grain=16")

        config = config or ScreenConfig(color=bool(source.color))
        mode = ("shape" if not config.color else
                "combo" if ColorPlacement(config.color_placement) is ColorPlacement.IN_GRADIENT
                else "rescore")
        if mode != "combo" and (canonical_color_kernel or factor_canonical_torque or optimizer_exp_mode != "baseline"):
            raise ValueError("analytic optimizer-color options require combo mode")
        if bool(config.color) != bool(source.color):
            raise ValueError(
                "ScreenConfig.color must match the R3gSource color layout")
        if PoseSearch(config.pose_search) is not PoseSearch.FIELD:
            raise NotImplementedError(
                "worker-owned CPU mode-3 scoring requires pose_search=FIELD")
        if ColorPlacement(config.color_placement) not in (ColorPlacement.IN_GRADIENT, ColorPlacement.AT_POSE):
            raise NotImplementedError(
                "worker-owned CPU scoring requires color_placement=IN_GRADIENT")
        if UpdateRule(config.update_rule) is not UpdateRule.ADAGRAD_EXP:
            raise NotImplementedError(
                "worker-owned CPU scoring implements update_rule=ADAGRAD_EXP")
        opt = config.resolved_optimizer()
        if not 0.0 <= float(config.mixing) <= 1.0:
            raise ValueError("mixing must be in [0, 1]")
        scalar_shape = (shape_backend, shape_grain, int(shape_width)) == (
            "scalar_analytic", "scalar", 1)
        retained_field_shape = (
            shape_backend, shape_grain, int(shape_width)) == (
                "pipelined_field", "library_atom", 8)
        if field_storage not in ("fp32", "fp16"):
            raise ValueError("field_storage must be 'fp32' or 'fp16'")
        if field_interpolation not in ("fp32", "mixed_half_yz"):
            raise ValueError(
                "field_interpolation must be 'fp32' or 'mixed_half_yz'")
        if ((field_storage, field_interpolation) not in (
                ("fp32", "fp32"), ("fp16", "mixed_half_yz"))):
            raise ValueError(
                "mixed_half_yz requires FP16 storage; FP32 storage requires "
                "FP32 interpolation")
        if field_storage == "fp16" and (mode != "combo" or
                                         not retained_field_shape):
            raise ValueError(
                "mixed FP16 Y/Z is limited to combo pipelined-field/library-atom/8")
        if not (scalar_shape or retained_field_shape):
            raise NotImplementedError(
                "worker-owned scan supports scalar_analytic/scalar/1 or the "
                "retained pipelined_field/library_atom/8 shape policy")
        if mode != "combo" and not retained_field_shape:
            raise NotImplementedError("shape/rescore workers require retained pipelined fields")
        if mode == "shape" and float(config.mixing) != 0.:
            raise ValueError("shape-only workers require mixing=0")
        if mode != "combo" and (color_field_backends is not None or color_field_policy is not None):
            raise NotImplementedError("optimizer color fields require combo mode")
        if color_field_backends is not None and color_field_policy is not None:
            raise ValueError(
                "pass either explicit color_field_backends or "
                "color_field_policy, not both")
        field_padding = (float(config.field_pad_a)
                         if field_padding_angstrom is None
                         else float(field_padding_angstrom))
        if not np.isfinite(field_padding) or field_padding <= 0.0:
            raise ValueError("field_padding_angstrom must be finite and positive")

        if config.color:
            from ..definitions import check_corpus_color
            check_corpus_color(source.corpus, source.color_generator,
                               trusted_definitions=source.trusted_definitions)
            from ..format.r3g.descriptor import DefinitionMismatch
            for name, actual in (("r", queries.interaction_r), ("p", queries.interaction_p)):
                expected = np.asarray(source.corpus.color[name]["values"], dtype=np.float32)
                if not np.array_equal(np.asarray(actual, dtype=np.float32), expected):
                    raise DefinitionMismatch(name)
        descriptor_setup_start = time.perf_counter()
        descriptors_all = source.cpu_block_descriptors()
        if block_descriptors is None:
            descriptors = tuple(descriptors_all)
        else:
            descriptors = tuple(block_descriptors)
        if any(not isinstance(item, CPUPhysicalBlockDescriptor)
               for item in descriptors):
            raise TypeError(
                "block_descriptors must come from cpu_block_descriptors()")
        source_descriptors = {
            (item.group_index, item.block_index): item
            for item in descriptors_all
        }
        seen_blocks = set()
        group_spans = {}
        group_counts = [int(g["n_conf"]) for g in source.corpus.group_table]
        full_offsets = np.concatenate((
            np.asarray([0], dtype=np.int64),
            np.cumsum(group_counts, dtype=np.int64)))
        for item in descriptors:
            key = (item.group_index, item.block_index)
            canonical = source_descriptors.get(key)
            if canonical is not item:
                raise ValueError(
                    "block descriptor is not owned by this source scan")
            if key in seen_blocks:
                raise ValueError("block_descriptors contains a duplicate block")
            seen_blocks.add(key)
            group_metadata = source._cpu_groups[item.group_index]
            if (item.shape_reader is not group_metadata or
                    item.group_row0 < 0 or item.n_conf <= 0 or
                    item.group_row0 + item.n_conf > group_counts[item.group_index]):
                raise ValueError("block descriptor has an invalid group row span")
            expected_gid = int(full_offsets[item.group_index] +
                                item.group_row0)
            if item.gid_base != expected_gid:
                raise ValueError("block descriptor GID does not match source order")
            group_spans.setdefault(item.group_index, []).append(
                (item.group_row0, item.group_row0 + item.n_conf))
        for spans in group_spans.values():
            ordered = sorted(spans)
            if any(left[1] > right[0]
                   for left, right in zip(ordered, ordered[1:])):
                raise ValueError("selected physical block row spans overlap")
        # Source order is the spec-owned expensive-first directory. Compact
        # rows and source GIDs remain independent of claim order.
        directory_order = {
            (d.group_index, d.block_index): i for i, d in enumerate(descriptors_all)}
        descriptors = tuple(sorted(descriptors, key=lambda d:
            directory_order[d.group_index, d.block_index]))
        descriptor_setup_s = time.perf_counter() - descriptor_setup_start
        # One active selection only; never accumulate handles for old subsets.
        if self._worker_self_cache is not None:
            selected_keys = {(d.group_index, d.block_index) for d in descriptors}
            self._worker_self_cache.entries = {
                key: value for key, value in self._worker_self_cache.entries.items()
                if key in selected_keys}

        qc = np.ascontiguousarray(queries.coords if mode == "combo" else queries.coords[..., :3], dtype=np.float32)
        qt = (np.ascontiguousarray(queries.types, dtype=np.int32) if config.color
              else np.zeros(qc.shape[:2], np.int32))
        qn = np.ascontiguousarray(queries.n_real, dtype=np.int32)
        interaction_r = np.ascontiguousarray(
            queries.interaction_r if config.color else np.ones((1, 1), np.float32), dtype=np.float32)
        interaction_p = np.ascontiguousarray(
            queries.interaction_p if config.color else np.zeros((1, 1), np.float32), dtype=np.float32)
        if (qc.ndim != 3 or qt.shape != qc.shape[:2] or
                qn.shape != (qc.shape[0],) or qc.shape[0] == 0):
            raise ValueError("queries must be a non-empty compatible QuerySet")
        if interaction_r.ndim != 2 or interaction_p.shape != interaction_r.shape:
            raise ValueError("query interaction matrices must have matching 2-D shapes")
        if int(rank_column) >= N_SCORE_COLUMNS:
            raise ValueError("rank_column must be in [0, 19]")
        if not descriptors:
            self._worker_query_cache = None
            self._worker_self_cache = None
            empty = []
            for _ in range(int(qc.shape[0])):
                if level == "molecule":
                    empty.append((np.zeros((0, N_SCORE_COLUMNS), np.float32),
                                  np.zeros(0, np.int64),
                                  np.zeros(0, np.int64)))
                else:
                    empty.append((np.zeros((0, N_SCORE_COLUMNS), np.float32),
                                  np.zeros(0, np.int64)))
            return ScreenStats(cpu_scan={
                "blocks": 0, "rows": 0, "scored_rows": 0,
                "workers": int(workers), "memory_peak_estimate_bytes": 0,
                "norm_contract": "shared_fp32_reduced_v1",
                "stored_vaa_used": False,
                "shape_field_storage_request": field_storage,
                "shape_field_interpolation_request": field_interpolation,
                "shape_field_storage_stamps": []}), empty

        n_queries = int(qc.shape[0])
        self_cache_invalidated = False
        if self_overlap_policy == "prepared":
            query_key = self.backend._fingerprint_arrays(
                "roshambo3.worker.self.query.v1." + mode, (
                    ("coords", qc), ("types", qt), ("n_real", qn),
                    ("R", interaction_r), ("P", interaction_p)))
            cached = self._worker_self_cache
            if (cached is None or cached.source is not source or
                    cached.query_key != query_key):
                self_cache_invalidated = cached is not None
                from .r3g_cpu import R3gWorkerSelfCache
                self._worker_self_cache = R3gWorkerSelfCache(source, query_key)
        self_cache = self._worker_self_cache
        if self_cache is not None:
            self_cache.peak_entries = 0
        prepared_self_bound = (min(int(workers), len(descriptors)) * max(
            ((n_queries + 3 * d.n_conf) * 12 + 1024 for d in descriptors), default=0)
            if self_cache is not None else 0)
        r3g_query_self_before = (0, 0, 0, 0.0)
        if self_cache is not None:
            r3g_query_self_before = (self_cache.query_preparations,
                self_cache.query_shape_self_evaluations,
                self_cache.query_color_self_evaluations,
                self_cache.query_self_preparation_seconds)
            if mode == "combo":
                # Cached exact query bytes plus native self vectors, including
                # the first preparation's normalized snapshot temporary.
                prepared_self_bound += 2 * (qc.nbytes + qt.nbytes + qn.nbytes +
                    interaction_r.nbytes + interaction_p.nbytes + n_queries * 12)
        total_rows = sum(int(item.n_conf) for item in descriptors)
        feature_count = int(interaction_r.shape[0])
        color_enabled = bool(source.color)
        if color_enabled and any(
                source._cpu_groups[item.group_index].n_types + 1 != feature_count
                for item in descriptors):
            raise ValueError(
                "query interaction width differs from source count features")

        from .config import FIELD_SPACING_BY_WIDTH, _by_width
        if retained_field_shape:
            spacing_by_group = {}
            for item in descriptors:
                if field_spacing_angstrom is not None:
                    spacing_by_group[item.group_index] = float(
                        field_spacing_angstrom)
                elif config.field_spacing_a is not None:
                    spacing_by_group[item.group_index] = float(
                        config.field_spacing_a)
                else:
                    spacing_by_group[item.group_index] = float(_by_width(
                        FIELD_SPACING_BY_WIDTH, item.field_width, 0.8))
        else:
            spacing = (float(field_spacing_angstrom)
                       if field_spacing_angstrom is not None
                       else (float(config.field_spacing_a)
                             if config.field_spacing_a is not None else 0.5))
            spacing_by_group = {
                item.group_index: spacing for item in descriptors}
        spacing_contexts = tuple(sorted(set(spacing_by_group.values())))
        if any(not np.isfinite(spacing) or spacing <= 0.0
               for spacing in spacing_contexts):
            raise ValueError(
                "field_spacing_angstrom must be finite and positive")

        query_setup_start = time.perf_counter()
        query_binding = self.backend._fingerprint_arrays(
            "roshambo3.worker.query.context.v1", (
                ("coords", qc), ("types", qt), ("n_real", qn),
                ("R", interaction_r), ("P", interaction_p)))
        native_identity = (str(self.backend.extension().__file__),
                           repr(sorted(self.backend.build_stamp().items())))
        previous_contexts = self._worker_query_cache

        # Bound concurrent private state. There is no decode queue: at most
        # min(workers, block_count) callbacks own a decoded block. Count both
        # the required int16->FP32 CPU-layout conversion and the native scorer's
        # aligned FP32 library snapshot; the former is an allocation, not zero-copy.
        def block_worker_peak(item):
            rows = int(item.n_conf)
            width = int(item.library_width)
            shape_i16 = rows * 3 * int(item.physical_shape_width) * 2
            decoded_i16 = shape_i16
            fp32_conversion = rows * width * 3 * 4
            aligned_native_width = ((width * 3 + 15) // 16) * 16
            native_library = rows * aligned_native_width * 4 + 64
            shape_workspace = (rows * 1024
                               if retained_field_shape else 0)
            native_query = n_queries * int(qc.shape[1]) * 3 * 4
            count_copy = rows * feature_count
            pair_index = n_queries * rows * 2 * np.dtype(np.int32).itemsize
            compressed = int(item.shape_record[1])
            raw = int(item.shape_record[2])
            decode_transient = compressed + raw
            fingerprint_scratch = (fp32_conversion + count_copy + qc.nbytes +
                qt.nbytes + qn.nbytes + interaction_r.nbytes + interaction_p.nbytes + 4096
                if mode == "combo" else 0)
            # The physical heavy-atom input and the expanded FP32 output are
            # live together during the native format-3 decode.
            return (decoded_i16 + fp32_conversion + native_library +
                    shape_workspace + (n_queries * rows * N_SCORE_COLUMNS * 4 * (2 if mode == "rescore" else 1) if mode != "combo" else 0) +
                    native_query + count_copy + pair_index + decode_transient + fingerprint_scratch)

        active_workers = min(int(workers), len(descriptors))
        max_worker_peak = max(block_worker_peak(item) for item in descriptors)
        shared_query_bytes = (qc.nbytes + qt.nbytes + qn.nbytes +
                              interaction_r.nbytes + interaction_p.nbytes)
        descriptor_bytes = len(descriptors_all) * 1024 + len(descriptors) * 512
        metadata_owned_bytes = int(getattr(
            source, "_cpu_metadata_owned_bytes", 0))
        def field_context_bound(spacing):
            if not retained_field_shape:
                return 0
            total = 0
            for query_row, n_real in enumerate(qn):
                n_real = int(n_real)
                xyz = np.asarray(qc[query_row, :n_real, :3],
                                 dtype=np.float64)
                if xyz.shape[0] == 0 or not np.isfinite(xyz).all():
                    raise ValueError(
                        "field query rows need non-empty finite shape coordinates")
                spans = np.ptp(xyz, axis=0)
                # Mirror plan_field's range + two halos and add one full cell
                # per axis for float-ceil boundary differences. This is a
                # conservative pre-context estimate, not an RSS guarantee.
                cell_extents = (spans + 2.0 * field_padding) / spacing
                if (not np.isfinite(cell_extents).all() or
                        np.any(cell_extents >= np.iinfo(np.int64).max - 2)):
                    raise ValueError(
                        "field spacing/padding implies an unrepresentable grid")
                cells = np.ceil(cell_extents)
                nodes = np.maximum(1, cells.astype(np.int64) + 2) + 1
                nx, ny, nz = (int(nodes[0]), int(nodes[1]), int(nodes[2]))
                zero_tail_floats = nx * 4 * ny + nx * 4 + 8
                total += ((nx * ny * nz * 4 + zero_tail_floats) * 4)
            return total

        query_context_bound = 0
        query_self_bytes_per_context = n_queries * (
            np.dtype(np.float32).itemsize + np.dtype(np.float64).itemsize)
        for spacing in spacing_contexts:
            query_context_bound += (
                2 * qc.nbytes + field_context_bound(spacing) +
                feature_count * feature_count * 16 + n_queries * 1024 +
                query_self_bytes_per_context)
        query_context_bound_without_color = query_context_bound
        base_pre_context_peak = (
            active_workers * max_worker_peak + shared_query_bytes +
            descriptor_bytes + metadata_owned_bytes +
            query_context_bound_without_color + prepared_self_bound)
        if memory_budget_bytes is not None and base_pre_context_peak > int(memory_budget_bytes):
            raise MemoryError(
                "worker-owned scan preflight needs approximately "
                f"{base_pre_context_peak:,} private bytes, above budget "
                f"{int(memory_budget_bytes):,}")
        ext = self.backend.extension()
        # The native byte limit is a checked size_t bound, not an allocation
        # size. No public cap still preserves representability/overflow checks.
        native_topk_budget = (int(np.iinfo(np.uintp).max)
            if topk_state_budget_bytes is None else int(topk_state_budget_bytes))
        topk_preflight_estimate = int(ext.estimate_worker_local_topk_direct_peak_bytes(
            int(workers), n_queries, int(k), int(rank_column), str(level),
            total_rows, True, native_topk_budget))
        base_pre_context_peak += topk_preflight_estimate

        from ..backend.color_field_policy import (
            CPUColorFieldPolicy, estimate_color_field_storage,
            normalize_backend_matrix)
        if color_field_policy is not None and not isinstance(
                color_field_policy, CPUColorFieldPolicy):
            raise TypeError("color_field_policy must be a CPUColorFieldPolicy")
        if color_field_policy is not None and color_grain != 16:
            raise NotImplementedError(
                "the static color-plane policy is qualified for public grain 16 only")
        target_feature_totals = np.zeros(feature_count, dtype=np.float64)
        for item in descriptors:
            target_feature_totals += (
                source._cpu_feature_totals[item.group_index, item.block_index])
        target_feature_means = (target_feature_totals / total_rows
                                if total_rows else target_feature_totals)
        color_field_spacing = 0.5
        color_field_padding = 4.0
        available_color_build_bytes = (int(np.iinfo(np.uintp).max)
            if memory_budget_bytes is None else
            max(0, int(memory_budget_bytes) - base_pre_context_peak))
        color_field_policy_info = {
            "policy": "explicit_backend_matrix" if color_field_backends is not None
            else "analytic_default",
            "selection_reason": "no_color_field_override",
            "selected_planes": [], "field_bytes_estimated": 0,
            "retained_bytes_estimated": 0,
            "field_build_peak_bytes_estimated": 0,
        }
        color_codes = None
        if color_field_policy is not None:
            color_codes, color_field_policy_info = color_field_policy.select(
                query_coords=qc, query_types=qt, query_n_atoms=qn,
                interaction_r=interaction_r, interaction_p=interaction_p,
                library_target_counts=target_feature_means,
                library_conformers=total_rows,
                spacing_angstrom=color_field_spacing,
                padding_angstrom=color_field_padding,
                available_build_bytes=available_color_build_bytes)
            if np.any(color_codes[:, 1:] != 0) and color_grain != 16:
                raise NotImplementedError(
                    "static color-plane dispatch requires color_grain=16")
        elif color_field_backends is not None:
            color_codes = normalize_backend_matrix(
                color_field_backends, n_queries, feature_count)
            if np.any(color_codes[:, 1:] != 0) and color_grain not in (11, 16):
                raise NotImplementedError(
                    "worker-owned color fields require retained grain 11 or 16")
        if color_codes is not None and np.any(color_codes[:, 1:] != 0):
            color_field_policy_info.update(estimate_color_field_storage(
                color_codes, qc, qt, qn, interaction_r, interaction_p,
                spacing_angstrom=color_field_spacing,
                padding_angstrom=color_field_padding))
            if color_field_policy_info.get("native_channel_limit_exceeded"):
                raise MemoryError(
                    "selected color fields exceed the native per-query "
                    "32M channel-value limit")
            if color_field_policy is None:
                color_field_policy_info["selection_reason"] = (
                    "explicit_backend_matrix")
            estimated_build_peak = int(color_field_policy_info.get(
                "field_build_peak_bytes_estimated", 0))
            if estimated_build_peak > available_color_build_bytes:
                raise MemoryError(
                    "selected color fields need approximately "
                    f"{estimated_build_peak:,} build/retained bytes after "
                    "the joint shape/query/worker budget, but only "
                    f"{available_color_build_bytes:,} bytes remain")
            if not color_field_policy_info.get("selected_planes"):
                # An explicit code for an empty/inactive plane does not create
                # useful storage. Keep it analytic and avoid native per-block
                # fallback builds for a handle that would contain no fields.
                color_codes = None
        else:
            color_codes = None
            if color_field_policy is not None:
                color_field_policy_info = dict(color_field_policy_info)

        if color_field_policy is not None:
            policy_cache_key = color_field_policy.cache_key()
        else:
            policy_cache_key = None
        color_codes_key = (None if color_codes is None else
                           (tuple(color_codes.shape), color_codes.tobytes()))
        query_context_key = (
            "t217-static-color-field-v1", mode, query_binding, native_identity,
            shape_backend, shape_grain, int(shape_width), field_storage,
            field_interpolation, spacing_contexts,
            field_padding, int(color_grain), policy_cache_key,
            color_codes_key, color_field_spacing, color_field_padding)
        query_context_cache_hit = (previous_contexts is not None and
                                   previous_contexts[0] == query_context_key)
        query_context_cache_invalidated = (previous_contexts is not None and
                                           not query_context_cache_hit)
        if not query_context_cache_hit:
            self._worker_query_cache = None
            previous_contexts = None  # Release the obsolete generation now.

        color_retained_bound = (int(color_field_policy_info.get(
            "retained_bytes_estimated", 0)) if color_codes is not None else 0)
        color_build_peak_bound = (int(color_field_policy_info.get(
            "field_build_peak_bytes_estimated", 0)) if color_codes is not None else 0)
        query_context_bound = (query_context_bound_without_color +
                               color_retained_bound)
        pre_context_peak = base_pre_context_peak + (
            color_retained_bound if query_context_cache_hit else
            color_build_peak_bound)
        if memory_budget_bytes is not None and pre_context_peak > int(memory_budget_bytes):
            raise MemoryError(
                "worker-owned scan preflight needs approximately "
                f"{pre_context_peak:,} private bytes including the bounded "
                "top-K state allowance, above budget "
                f"{int(memory_budget_bytes):,}; reduce workers/block selection "
                "or increase memory_budget_bytes")

        # One immutable shape/shared-pose context per resolved query-field
        # spacing. Scalar analytic handles are also shared scan-wide.
        contexts = previous_contexts[1] if query_context_cache_hit else {}
        newly_built_contexts = []
        for spacing in (() if query_context_cache_hit else spacing_contexts):
            shape_evaluators = self.backend.prepare_combo_shape_evaluators(
                qc, qn, shape_backend=shape_backend, shape_grain=shape_grain,
                shape_width=int(shape_width), field_spacing_angstrom=spacing,
                field_padding_angstrom=field_padding,
                field_storage=field_storage,
                field_interpolation=field_interpolation)
            if mode != "combo":
                from .cpu_modes import prepare_query_context
                shared_queries = prepare_query_context(self.backend, qc, qt, qn,
                    interaction_r, interaction_p, rescore=mode == "rescore")
            else:
                shared_queries = self.backend.prepare_combo_shared_pose_queries(
                qc, qt, qn, interaction_r, interaction_p,
                shape_backend=shape_backend, shape_grain=shape_grain,
                shape_width=int(shape_width), field_spacing_angstrom=spacing,
                field_padding_angstrom=field_padding,
                shape_evaluators=shape_evaluators,
                field_storage=field_storage,
                field_interpolation=field_interpolation)
            contexts[spacing] = (shape_evaluators, shared_queries)
            newly_built_contexts.append(shared_queries)
        color_fields = (previous_contexts[3]
                        if query_context_cache_hit and
                        len(previous_contexts) > 3 else None)
        color_field_build_s = 0.0
        color_field_builds_this_call = 0
        if color_codes is not None:
            if query_context_cache_hit:
                if color_fields is None:
                    raise RuntimeError(
                        "cached color-field policy key has no native handle")
                color_field_policy_info = dict(previous_contexts[4])
                color_field_policy_info["handle_reused"] = True
            else:
                color_build_start = time.perf_counter()
                color_fields = self.backend.prepare_color_fields(
                    query_coords=qc, query_types=qt, query_n_atoms=qn,
                    interaction_r=interaction_r, interaction_p=interaction_p,
                    backend_codes=color_codes)
                color_field_builds_this_call = 1
                color_field_build_s = time.perf_counter() - color_build_start
                if int(color_fields.field_bytes) > int(
                        color_field_policy_info.get("field_bytes_estimated", 0)):
                    raise MemoryError(
                        "native color fields exceeded the conservative "
                        "preflight payload estimate")
                if int(color_fields.retained_bytes) > int(
                        color_field_policy_info.get("retained_bytes_estimated", 0)):
                    raise MemoryError(
                        "native prepared color handle exceeded the conservative "
                        "retained-payload estimate")
                color_field_policy_info["field_bytes_actual"] = int(
                    color_fields.field_bytes)
                color_field_policy_info["retained_bytes_actual"] = int(
                    color_fields.retained_bytes)
                color_field_policy_info["build_seconds_actual"] = float(
                    color_fields.build_seconds)
                color_field_policy_info["pack_seconds_actual"] = float(
                    color_fields.pack_seconds)
                color_field_policy_info["handle_reused"] = False
        query_setup_s = time.perf_counter() - query_setup_start
        # Shared query self is lazy on some routes. Compare lifetime counters
        # before/after this scan, including a first preparation on a cache hit.
        query_self_before = {
            spacing: (bool(shared.query_self_prepared),
                int(shared.shape_query_self_evaluations),
                int(shared.color_query_self_evaluations),
                float(shared.query_self_preparation_seconds))
            if query_context_cache_hit else (False, 0, 0, 0.0)
            for spacing, (_, shared) in contexts.items()}
        query_context_bytes = 0
        for shape_evaluators, shared_queries in contexts.values():
            query_context_bytes += int(
                shape_evaluators.query_snapshot_bytes +
                shape_evaluators.query_workspace_bytes +
                shape_evaluators.field_bytes +
                shared_queries.reduced_color_bytes +
                feature_count * feature_count * 16 + n_queries * 1024 +
                query_self_bytes_per_context)
        if color_fields is not None:
            query_context_bytes += int(color_fields.retained_bytes)
        if query_context_bytes > query_context_bound:
            raise MemoryError(
                "query context exceeded its preflight bound: "
                f"{query_context_bytes:,} > {query_context_bound:,} bytes")
        self._worker_query_cache = (
            query_context_key, contexts, query_context_bytes,
            color_fields, dict(color_field_policy_info))
        planned_private_bytes = (
            active_workers * max_worker_peak + shared_query_bytes +
            descriptor_bytes + metadata_owned_bytes + query_context_bytes +
            prepared_self_bound)
        if memory_budget_bytes is not None and planned_private_bytes > int(memory_budget_bytes):
            raise MemoryError(
                "worker-owned scan preflight needs approximately "
                f"{planned_private_bytes:,} private bytes, above budget "
                f"{int(memory_budget_bytes):,}; reduce workers/block selection "
                "or increase memory_budget_bytes")

        collector = ext.make_worker_local_topk_direct_row_sink(
            int(workers), n_queries, int(k), int(rank_column), str(level),
            total_rows, True, native_topk_budget)
        topk_estimate = int(collector.memory_estimate["peak_bytes"])
        if topk_estimate != topk_preflight_estimate:
            collector.discard_after_failure()
            raise MemoryError("native top-K allocation estimate changed after preflight")
        planned_peak = max(pre_context_peak,
                           planned_private_bytes + topk_estimate)
        if memory_budget_bytes is not None and planned_peak > int(memory_budget_bytes):
            collector.discard_after_failure()
            raise MemoryError(
                "worker-owned scan including bounded top-K state needs "
                f"approximately {planned_peak:,} bytes, above budget "
                f"{int(memory_budget_bytes):,}")

        from ..backend.cpu import CPUComboExecutor
        # Each physical descriptor is already coarse-grained work. Give the
        # indexed cursor exactly one descriptor per claim so short-tail and
        # tiny-fixture blocks can still be owned by distinct persistent
        # workers; large corpus blocks also remain indivisible claims.
        descriptor_claim_rows = max(
            128, max(int(item.n_conf) for item in descriptors))
        executor = CPUComboExecutor(int(workers), claim_chunk_rows=128, cpu_qos=cpu_qos)
        stats = ScreenStats(pinned=False)
        counter_template = {
            "blocks": 0, "shape_blocks_decoded": 0,
            "color_blocks_decoded": 0, "rows_decoded": 0,
            "scored_rows": 0, "compressed_bytes_read": 0,
            "raw_bytes_decoded": 0, "fp32_layout_conversion_bytes": 0,
            "count_copy_bytes": 0, "pair_index_bytes": 0,
            "native_score_calls": 0,
            "native_self_overlap_fallback_calls": 0,
            "library_shape_self_evaluations": 0,
            "library_color_self_evaluations": 0,
            "self_preparation_calls": 0, "self_cache_hits": 0,
            "self_cache_replaced_blocks": 0,
            "self_preparation_s": 0.0, "self_binding_s": 0.0,
            "decode_s": 0.0, "conversion_s": 0.0,
            "native_decode_s": 0.0, "native_color_derivation_s": 0.0,
            "native_self_binding_s": 0.0,
            "scoring_s": 0.0,
            # Which verified stored-VAA blocks supplied self terms.
            "stored_vaa_verified_blocks": 0,
            "color_self_sidecar_blocks": 0,
            # Library color self terms evaluated while binding a block's
            # prepared self handle and blocks where color values were
            # recomputed instead of read from a verified sidecar.
            "library_color_self_binding_evaluations": 0,
            "color_self_binding_recompute_blocks": 0,
        }
        worker_counters = [dict(counter_template) for _ in range(int(workers))]
        failure = []
        fallback_reasons = {}
        color_self_fallback_reasons = {}
        # route -> blocks scored on it (scores.SELF_OVERLAP_MARGIN_POLICY).
        block_routes = {}
        route_lock = threading.Lock()
        progress_lock = threading.Lock()
        completed_blocks = completed_conformers = 0

        def score_block(worker_id, block_index):
            nonlocal completed_blocks, completed_conformers
            counters = worker_counters[int(worker_id)]
            try:
                if canceled is not None and canceled():
                    raise InterruptedError(
                        "worker-owned CPU scan canceled before block claim")
                descriptor = descriptors[block_index]
                decode_start = time.perf_counter()
                decoded = source.decode_cpu_block(descriptor)
                counters["decode_s"] += time.perf_counter() - decode_start
                counters["shape_blocks_decoded"] += 1
                counters["compressed_bytes_read"] += (
                    decoded.shape_compressed_bytes +
                    decoded.color_compressed_bytes)
                counters["raw_bytes_decoded"] += (
                    decoded.shape_raw_bytes + decoded.color_raw_bytes)

                counts = np.ascontiguousarray(decoded.result.counts, dtype=np.uint8)
                count_copy_bytes = int(counts.nbytes)
                library_coords = decoded.result.coords
                counters["native_decode_s"] += decoded.result.timings.decode
                counters["native_color_derivation_s"] += decoded.result.timings.color
                counters["native_self_binding_s"] += decoded.result.timings.self_overlap
                counters["color_blocks_decoded"] += int(source.color)
                if not library_coords.flags.c_contiguous:
                    raise AssertionError("CPU block conversion is not contiguous")
                pairs_q = np.repeat(np.arange(n_queries, dtype=np.int32),
                                    descriptor.n_conf)
                pairs_t = np.tile(np.arange(descriptor.n_conf, dtype=np.int32),
                                  n_queries)
                spacing = spacing_by_group[descriptor.group_index]
                shape_evaluators, shared_queries = contexts[spacing]
                self_kwargs = {}
                native_self_counts = {}
                if self_cache is not None:
                    preparation_start = time.perf_counter()
                    handle, reason, hit, replaced = self_cache.block(
                        self.backend, descriptor, decoded, library_coords,
                        qc, qt, qn, interaction_r, interaction_p, mode=mode)
                    elapsed = time.perf_counter() - preparation_start
                    counters["self_preparation_s" if handle is not None and not hit
                             else "self_binding_s"] += elapsed
                    if handle is not None:
                        counters["self_cache_hits"] += int(hit)
                        counters["self_preparation_calls"] += int(not hit)
                        counters["self_cache_replaced_blocks"] += int(replaced)
                        route = self_cache.routes[(descriptor.group_index,
                                                   descriptor.block_index)]
                        verified_route = route["vaa"] == "verified_stored"
                        if not verified_route:
                            raise RuntimeError("format-3 self cache returned an unverified VAA route")
                        counters["stored_vaa_verified_blocks"] += 1
                        counters["color_self_sidecar_blocks"] += int(
                            route["color"] == "sidecar")
                        binding_color = _binding_color_self_evaluations(
                            self.backend, handle)
                        counters["color_self_binding_recompute_blocks"] += int(
                            binding_color > 0 or
                            getattr(handle, "color_source", None) ==
                            "recomputed_at_binding")
                        if not hit:
                            # Evaluated now, for this binding; a cache hit
                            # reuses values evaluated by an earlier scan.
                            counters["library_color_self_binding_evaluations"] += (
                                binding_color)
                        if route["color_reason"] is not None:
                            color_self_fallback_reasons[(
                                descriptor.group_index,
                                descriptor.block_index)] = route["color_reason"]
                        if mode == "combo" and isinstance(
                                handle, self.backend.PreparedComboSelfOverlaps):
                            # Verified stored VAA (and color sidecar): the
                            # ordinary verified-handle API, no legacy flag.
                            self_kwargs = dict(prepared_self_overlaps=handle)
                        elif mode == "combo":
                            raise RuntimeError(
                                "format-3 combo self cache returned an unsupported handle")
                    else:
                        fallback_reasons[(descriptor.group_index, descriptor.block_index)] = reason
                    self_kwargs["library_self_counts"] = native_self_counts
                # The block's self-overlap route chooses its bound margin
                # (scores.SELF_OVERLAP_MARGIN_POLICY); the native sink checks
                # every row of the block against it.
                block_route = (
                    _scores.ROUTE_RECOMPUTE
                    if self_cache is None or handle is None else
                    self_cache.routes[(descriptor.group_index,
                                       descriptor.block_index)]["vaa"])
                with route_lock:
                    block_routes[block_route] = block_routes.get(block_route, 0) + 1
                    # While the tight tier is active a mixed scan is refused
                    # as soon as its second route appears.
                    _scores.check_scan_self_overlap_routes(block_routes)
                block_sink = ext.make_worker_local_topk_block_row_sink(
                    collector, int(worker_id), int(descriptor.gid_base),
                    int(descriptor.n_conf),
                    decoded.molecule_ids,
                    self_overlap_bound_margin=_scores.self_overlap_bound_margin(
                        block_route))
                score_start = time.perf_counter()
                if mode != "combo":
                    from .cpu_modes import score_block as score_shape_block
                    supplied = {}
                    rows = score_shape_block(self.backend, qc, qt, qn,
                        library_coords, counts, pairs_q, pairs_t,
                        interaction_r, interaction_p, shared_queries,
                        shape_evaluators, handle if self_cache is not None else None,
                        spacing=spacing, padding=field_padding, opt=opt,
                        start_mode=config.start_mode, mixing=config.mixing,
                        rescore=mode == "rescore",final_start_policy=final_start_policy,
                        library_self_supplied=supplied)
                    block_sink.ingest_scores(rows, pairs_q, pairs_t)
                    # Counted from the values actually handed to the native
                    # scorer (and, for shape, checked against every row's
                    # reported library self-overlap) -- not from whether a
                    # handle existed. The native shape/rescore paths expose
                    # no evaluation counter of their own. These are
                    # the evaluations made while scoring: color terms the
                    # handle evaluated at binding are handed over prepared
                    # and are counted, as evaluations, in
                    # library_color_self_binding_evaluations above.
                    handle_color = (mode == "rescore" and self_cache is not None
                                    and handle is not None
                                    and handle.color_self is not None)
                    native_self_counts.update(
                        library_shape_self_evaluations=descriptor.n_conf - supplied["shape_rows"],
                        library_color_self_evaluations=(
                            0 if mode != "rescore" or handle_color
                            else descriptor.n_conf - supplied["color_rows"]))
                else:
                    self.backend.score_pairs(
                    query_coords=qc, query_types=qt, query_n_atoms=qn,
                    library_coords=library_coords, library_counts=counts,
                    pair_query_row=pairs_q, pair_library_row=pairs_t,
                    interaction_r=interaction_r, interaction_p=interaction_p,
                    steps=int(opt["steps"]), lr_q=float(opt["lr_q"]),
                    lr_t=float(opt["lr_t"]),
                    budget_override=opt.get("budget_override"),
                    start_mode=int(config.start_mode),
                    color=True, color_mode=MODE_COMBO_OPTIMAL,
                    color_grain=int(color_grain), threads=1, executor=None,
                    factor_canonical_torque=bool(factor_canonical_torque),
                    optimizer_exp_mode=optimizer_exp_mode,
                    canonical_color_kernel=bool(canonical_color_kernel),
                    final_start_policy=final_start_policy,
                    mixing=float(config.mixing),
                    shape_backend=shape_backend, shape_grain=shape_grain,
                    shape_width=int(shape_width),
                    shape_field_spacing_angstrom=spacing,
                    shape_field_padding_angstrom=field_padding,
                    prepared_shape_evaluators=shape_evaluators,
                    pose_contract="shared_fp32_reduced_v1",
                    prepared_shared_pose_queries=shared_queries,
                    color_field_backends=color_codes,
                    prepared_color_fields=color_fields,
                    row_sink=block_sink, row_sink_only=True, **self_kwargs)
                counters["scoring_s"] += time.perf_counter() - score_start
                counters["native_score_calls"] += 1
                if self_cache is not None:
                    required_counts = {"library_shape_self_evaluations",
                                       "library_color_self_evaluations"}
                    if set(native_self_counts) != required_counts:
                        raise RuntimeError("native library self counter keys missing or unexpected")
                    for value in native_self_counts.values():
                        try:
                            valid = (not isinstance(value, (bool, np.bool_))
                                     and operator.index(value) >= 0)
                        except TypeError:
                            valid = False
                        if not valid:
                            raise RuntimeError("native library self counters must be nonnegative integers")
                for name, value in native_self_counts.items():
                    counters[name] += int(value)
                prepared_block = ("prepared_self_overlaps" in self_kwargs
                                  if mode == "combo" else
                                  self_cache is not None and handle is not None)
                if prepared_block and any(native_self_counts.values()):
                    raise RuntimeError("prepared worker scoring recomputed library self")
                counters["native_self_overlap_fallback_calls"] += int(not prepared_block)
                counters["blocks"] += 1
                counters["rows_decoded"] += int(descriptor.n_conf)
                counters["scored_rows"] += (
                    n_queries * int(descriptor.n_conf))
                counters["fp32_layout_conversion_bytes"] += int(
                    library_coords.nbytes)
                counters["count_copy_bytes"] += count_copy_bytes
                counters["pair_index_bytes"] += int(
                    pairs_q.nbytes + pairs_t.nbytes)
                if progress is not None:
                    # Serialize publication after ingestion. Each callback owns
                    # a snapshot; neither other workers nor finalization mutate it.
                    with progress_lock:
                        completed_blocks += 1
                        completed_conformers += int(descriptor.n_conf)
                        progress(ScreenStats(
                            wall_s=time.perf_counter() - scan_start,
                            n_blocks=completed_blocks,
                            n_conformers=completed_conformers,
                            n_rows=n_queries * completed_conformers,
                            pinned=False))
            except BaseException as exc:
                if not failure:
                    failure.append((exc, exc.__traceback__))
                raise
            finally:
                if self_cache is not None:
                    self_cache.release_block(descriptors[block_index])

        scan_start = time.perf_counter()
        try:
            executor.run_indexed(
                len(descriptors), score_block,
                rows_per_item=descriptor_claim_rows)
            executor_stats = executor.stats
            collector.mark_scoring_succeeded()
            result = collector.finalize()
        except BaseException:
            self._worker_self_cache = None
            collector.discard_after_failure()
            if failure:
                exc, traceback = failure[0]
                raise exc.with_traceback(traceback)
            raise
        finally:
            executor.close()

        scan_wall_s = time.perf_counter() - scan_start
        full_setup_scan_s = time.perf_counter() - full_start
        counters = {
            key: sum(worker.get(key, 0) for worker in worker_counters)
            for key in counter_template
        }
        # Verified and recomputed blocks describe their scored coordinates;
        # the route classifier records their provenance per scan.
        self_overlap_route = _scores.scan_self_overlap_route(block_routes)
        stored_vaa_route = (
            "verified_stored" if counters["stored_vaa_verified_blocks"] else None)
        per_worker_counters = []
        for worker in worker_counters:
            per_worker_counters.append({
                **{key: worker.get(key, 0) for key in counter_template},
            })
        stats.wall_s = full_setup_scan_s
        stats.launch_s = counters["scoring_s"]
        stats.n_blocks = counters["blocks"]
        stats.n_conformers = counters["rows_decoded"]
        stats.n_rows = counters["scored_rows"]
        query_self_contexts_built = sum(
            bool(shared.query_self_prepared) and not query_self_before[spacing][0]
            for spacing, (_, shared) in contexts.items())
        shared_query_shape_self_evaluations = sum(
            int(shared.shape_query_self_evaluations) - query_self_before[spacing][1]
            for spacing, (_, shared) in contexts.items())
        shared_query_color_self_evaluations = sum(
            int(shared.color_query_self_evaluations) - query_self_before[spacing][2]
            for spacing, (_, shared) in contexts.items())
        shared_query_self_preparation_s = sum(
            float(shared.query_self_preparation_seconds) - query_self_before[spacing][3]
            for spacing, (_, shared) in contexts.items())
        r3g_query_self_contexts_built = 0
        if self_cache is not None:
            r3g_query_self_contexts_built = (
                self_cache.query_preparations - r3g_query_self_before[0])
            query_self_contexts_built += r3g_query_self_contexts_built
            shared_query_shape_self_evaluations += (
                self_cache.query_shape_self_evaluations - r3g_query_self_before[1])
            shared_query_color_self_evaluations += (
                self_cache.query_color_self_evaluations - r3g_query_self_before[2])
            shared_query_self_preparation_s += (
                self_cache.query_self_preparation_seconds - r3g_query_self_before[3])
        stats.cpu_scan = {
            "mode": mode,
            "format3_query_self_contexts_built": r3g_query_self_contexts_built,
            "format3_query_self_retained_bytes": (self_cache.query_retained_bytes
                if self_cache is not None else 0),
            **counters,
            "workers": int(workers),
            "executor_submissions": 1,
            "descriptors_per_claim": 1,
            "active_worker_bound": active_workers,
            "worker_counters": per_worker_counters,
            "self_cache_peak_blocks": (self_cache.peak_entries
                if self_cache is not None else None),
            "cpu_qos": cpu_qos,
            "worker_placement": executor_stats.get("worker_placement", []),
            "query_contexts_built": len(newly_built_contexts),
            "query_context_cache_hits": len(contexts) if query_context_cache_hit else 0,
            "query_context_cache_invalidated": query_context_cache_invalidated,
            "query_context_retained_bytes": query_context_bytes,
            "color_field_policy": color_field_policy_info.get("policy"),
            "color_field_selection_reason": color_field_policy_info.get(
                "selection_reason"),
            "color_field_selected_planes": color_field_policy_info.get(
                "selected_planes", []),
            "color_field_codes_by_query": (
                None if color_codes is None else color_codes.tolist()),
            "color_field_policy_calibration_ids": color_field_policy_info.get(
                "calibration_ids", ()),
            "color_field_policy_source_sha256": color_field_policy_info.get(
                "calibration_source_sha256", ()),
            "color_field_policy_scope": color_field_policy_info.get(
                "measurement_scope", ()),
            "color_field_bytes_estimated": int(color_field_policy_info.get(
                "field_bytes_estimated", 0)),
            "color_field_bytes_actual": int(color_fields.field_bytes)
                if color_fields is not None else 0,
            "color_field_retained_bytes_estimated": int(
                color_field_policy_info.get("retained_bytes_estimated", 0)),
            "color_field_retained_bytes_actual": int(color_fields.retained_bytes)
                if color_fields is not None else 0,
            "color_field_build_peak_bytes_estimated": int(
                color_field_policy_info.get(
                    "field_build_peak_bytes_estimated", 0)),
            "color_field_build_seconds_actual": float(
                color_fields.build_seconds) if color_fields is not None else 0.0,
            "color_field_pack_seconds_actual": float(
                color_fields.pack_seconds) if color_fields is not None else 0.0,
            "color_field_build_wall_s": color_field_build_s,
            "color_field_builds_this_call": color_field_builds_this_call,
            "color_field_handle_reused": bool(
                color_fields is not None and
                color_field_policy_info.get("handle_reused", False)),
            "color_field_joint_budget_available_bytes":
                available_color_build_bytes,
            "query_self_contexts_built": query_self_contexts_built,
            "shape_query_self_evaluations":
                shared_query_shape_self_evaluations,
            "color_query_self_evaluations":
                shared_query_color_self_evaluations,
            "query_self_preparation_s": shared_query_self_preparation_s,
            "shape_backend": shape_backend,
            "shape_field_storage_request": field_storage,
            "shape_field_interpolation_request": field_interpolation,
            "shape_field_storage_stamps": [
                stamp.get("storage")
                for spacing in sorted(contexts)
                for stamp in contexts[spacing][0].stamps],
            "factor_canonical_torque": bool(factor_canonical_torque),
            "optimizer_exp_mode": optimizer_exp_mode,
            "final_start_policy": final_start_policy,
            "canonical_color_kernel": bool(canonical_color_kernel),
            "shape_grain": shape_grain,
            "shape_width": int(shape_width),
            "field_spacing_by_group": dict(spacing_by_group),
            "field_width_by_group": {
                item.group_index: int(item.field_width)
                for item in descriptors},
            "query_context_setup_s": query_setup_s,
            "descriptor_setup_s": descriptor_setup_s,
            "scan_wall_s": scan_wall_s,
            "full_setup_scan_s": full_setup_scan_s,
            "scan_conformers_per_s": (
                stats.n_conformers / scan_wall_s if scan_wall_s > 0 else 0.0),
            "full_setup_conformers_per_s": stats.conformers_per_s,
            "library_row_fp32_conversion_is_copy": True,
            "library_row_fp32_conversion_bytes":
                counters["fp32_layout_conversion_bytes"],
            "source_count_and_mid_maps_copied": False,
            "global_id_map_copied": False,
            "count_sidecar_copy_bytes": counters["count_copy_bytes"],
            "count_metadata_bytes_owned": metadata_owned_bytes,
            "topk_state_estimate_bytes": topk_estimate,
            "worker_private_peak_estimate_bytes": planned_private_bytes,
            "query_context_estimate_bytes": query_context_bytes,
            "memory_peak_estimate_bytes": planned_peak,
            "memory_estimate_not_hard_rss_cap": True,
            "memory_budget_bytes": memory_budget_bytes,
            "topk_state_budget_bytes": topk_state_budget_bytes,
            "norm_contract": "shared_fp32_reduced_v1" if mode == "combo" else "shape_fp32_physical_color_v1",
            "self_overlap_policy": self_overlap_policy,
            "library_self_evaluations_counted": self_cache is not None,
            "library_shape_self_evaluations": counters["library_shape_self_evaluations"] if self_cache else None,
            # Every library color self term this scan evaluated, in every
            # mode: those evaluated while scoring (combo's native counter;
            # the Python count for shape and rescore) plus those evaluated
            # while binding a block's prepared self handle, which are also
            # reported alone in library_color_self_binding_evaluations. A
            # binding-time recomputation is not a supplied value, so this reads 0 only when every block's terms came
            # from a validated color self sidecar or from an earlier scan's
            # binding (a cache hit) and scoring evaluated none.
            "library_color_self_evaluations": (
                None if not self_cache else
                counters["library_color_self_evaluations"] +
                counters["library_color_self_binding_evaluations"]),
            "self_cache_invalidated": self_cache_invalidated,
            "prepared_self_retained_bytes": self_cache.retained_bytes if self_cache else 0,
            "prepared_self_budget_bound_bytes": prepared_self_bound,
            "self_fallback_reasons": {f"{g}:{b}": reason for (g,b), reason in fallback_reasons.items()},
            # Every block's self-overlap route, the scan's classification
            # and the bound margin each route's rows were checked against
            # (scores.SELF_OVERLAP_MARGIN_POLICY).
            "self_overlap_route": self_overlap_route,
            "self_overlap_block_routes": dict(sorted(block_routes.items())),
            "self_overlap_bound_margins": {
                route: _scores.self_overlap_bound_margin(route)
                for route in sorted(block_routes)},
            "self_overlap_margin_tight_active": bool(
                _scores.SELF_OVERLAP_MARGIN_TIGHT_ACTIVE),
            "stored_vaa_used": stored_vaa_route is not None,
            # Route taken by the blocks this run prepared or reused. Frame
            # verification (stored frame, row-bound, validated at binding) is
            # not precision qualification, which stays a separate decision.
            "stored_vaa_route": stored_vaa_route,
            "stored_vaa_frame_verified": bool(
                counters["stored_vaa_verified_blocks"]),
            "stored_vaa_precision_qualified": False,
            # The conversion actually applied to stored values and the route
            # they came by; None when the scan used none.
            "stored_vaa_conversion": _stored_vaa_conversion(
                mode, stored_vaa_route),
            "self_preparation_included_in_scan_wall": True,
            "serialized_native_shape_self_values_used": False,
            "color_self_sidecar_used": bool(counters["color_self_sidecar_blocks"]),
            "color_self_fallback_reasons": {
                f"{g}:{b}": reason
                for (g, b), reason in color_self_fallback_reasons.items()},
            # Where the library self counters come from. A scan any of whose
            # blocks had its color self terms recomputed at binding says so
            # instead of reporting those values as supplied; for combo the scoring-time count is the native counter's
            # and the binding-time count is Python's.
            "library_self_evaluation_counter_source": (
                None if self_cache is None else
                COUNTER_SOURCE_NATIVE_AND_COLOR_RECOMPUTED_AT_BINDING
                if mode == "combo" and
                counters["color_self_binding_recompute_blocks"] else
                COUNTER_SOURCE_NATIVE if mode == "combo" else
                COUNTER_SOURCE_PYTHON_COLOR_RECOMPUTED_AT_BINDING
                if mode == "rescore" and
                counters["color_self_binding_recompute_blocks"] else
                COUNTER_SOURCE_PYTHON_SUPPLIED),
            "library_color_self_binding_evaluations":
                counters["library_color_self_binding_evaluations"]
                if self_cache else None,
            "color_self_binding_recompute_blocks":
                counters["color_self_binding_recompute_blocks"],
            "native_self_overlap_fallback_per_block": bool(counters["native_self_overlap_fallback_calls"]),
            "native_library_self_overlap_fallback_per_block": bool(counters["native_self_overlap_fallback_calls"]),
            "native_query_views_setup_per_block": True,
            "native_query_self_overlaps_reused_scan_wide": True,
            "decode_queue_depth": 0,
            "max_live_decoded_blocks": active_workers,
        }
        return stats, result

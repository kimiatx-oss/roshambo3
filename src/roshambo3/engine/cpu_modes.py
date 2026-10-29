"""Retained shape-search/rescore plumbing; no alternate optimizer/reducer.

Mode 2 color is evaluated exactly once at the shape-best of four starts.
It does not optimize color or use color to select a different start.
Shape self consumes the authenticated format-3 stored volume through the
float(VAA) * float(1/SHAPE_PREFACTOR) conversion. Physical color self is
read from a validated color self sidecar on the verified stored route, or
else evaluated once per Source/descriptor binding (and counted as evaluated);
it is never taken from combo's reduced frame values. Facade-created Sources
therefore still pay cold library prep off the verified route.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ShapeQueryContext:
    shape_self: np.ndarray
    color_self: object
    query_self_preparation_seconds: float
    reduced_color_bytes: int = 0
    query_self_prepared: bool = True

    @property
    def shape_query_self_evaluations(self):
        return len(self.shape_self)

    @property
    def color_query_self_evaluations(self):
        return 0 if self.color_self is None else len(self.color_self)


def prepare_query_context(cpu, qc, qt, qn, r, p, *, rescore):
    # The existing exact shape self helper uses the same scaled FP32
    # coordinates as legacy shape search. No color work for shape-only.
    shape_r = np.ones((1, 1), np.float32)
    shape_p = np.zeros((1, 1), np.float32)
    empty = np.empty((0, 1, 3), np.float32)
    counts = np.empty((0, 1), np.uint8)
    import time
    started = time.perf_counter()
    handle = cpu.extension().prepare_combo_self_overlaps(
        qc, np.zeros(qc.shape[:2], np.int32), qn, empty, counts,
        shape_r, shape_p, "T219-shape-query-self", None, None, None, True, False)
    shape = handle.shape_query_values
    color = None
    if rescore:
        color, _ = cpu.extension().prepare_rescore_color_self(
            qc, qt, qn, empty, np.empty((0, len(r)), np.uint8), r, p)
        color.flags.writeable = False
    shape.flags.writeable = False
    return ShapeQueryContext(shape, color, time.perf_counter() - started)


#: Where a rescore block's library color self terms came from.
COLOR_SELF_FROM_SIDECAR = "sidecar"
COLOR_SELF_RECOMPUTED_AT_BINDING = "recomputed_at_binding"


@dataclass
class ShapeBlockSelf:
    vaa: np.ndarray
    color_self: object
    descriptor_binding: str
    prepared_library_self_overlaps: object
    #: None (shape only), COLOR_SELF_FROM_SIDECAR or
    #: COLOR_SELF_RECOMPUTED_AT_BINDING.
    color_source: object = None
    #: Library color self terms evaluated to make this handle (0 when they
    #: were read from a validated sidecar).
    color_self_binding_evaluations: int = 0

    @property
    def retained_native_value_bytes(self):
        return self.vaa.nbytes + (0 if self.color_self is None else self.color_self.nbytes)


def prepare_block_self(cpu, coords, values, binding, prepared_library,
                       *, rescore):
    """Values-only self handle for one shape/rescore block.

    ``prepared_library`` is the native format-3 library handle produced from
    the authenticated decoder output. It carries the exact stored shape self
    values and, for rescore, either the matrix-qualified stored color self
    values or the native binding-time recomputation.
    """
    vaa = np.array(values, dtype=np.float32, copy=True)
    vaa.flags.writeable = False
    if not prepared_library.library_shape_from_stored_volume:
        raise ValueError("format-3 library self handle must use stored shape values")
    if len(prepared_library.shape_values) != len(vaa):
        raise ValueError("prepared shape self values must cover every block row")
    color = None
    source = None
    evaluations = 0
    if rescore:
        if not prepared_library.color_prepared:
            raise ValueError("rescore requires prepared library color self values")
        color = np.array(prepared_library.color_values, dtype=np.float64, copy=True)
        if color.shape != (len(vaa),):
            raise ValueError("prepared color self values must cover every block row")
        color.flags.writeable = False
        if prepared_library.library_color_from_sidecar:
            source = COLOR_SELF_FROM_SIDECAR
        else:
            source = COLOR_SELF_RECOMPUTED_AT_BINDING
            evaluations = int(len(vaa))
    return ShapeBlockSelf(vaa, color, binding, prepared_library,
                          source, evaluations)


def score_block(cpu, qc, qt, qn, coords, counts, pq, pl, r, p, context,
                fields, handle, *, spacing, padding, opt, start_mode,
                mixing, rescore, final_start_policy="winner",
                library_self_supplied=None):
    """Score one block. ``library_self_supplied``, a caller-owned dict, gets
    how many library rows' shape and color self terms were handed to the
    native scorer rather than left for it to compute. The shape count
    is checked against the scored rows: every row's reported library
    self-overlap must be its supplied stored value. Only color terms read
    from a validated sidecar count as supplied; terms the handle evaluated at
    binding are a recomputation and are counted by the caller as one."""
    supplied_shape = 0 if handle is None else int(len(handle.vaa))
    supplied_color = (0 if handle is None or not rescore or handle.color_self is None
                      or handle.color_source != COLOR_SELF_FROM_SIDECAR
                      else int(len(handle.color_self)))
    if handle is not None and supplied_shape != coords.shape[0]:
        raise ValueError("prepared shape self values must cover every block row")
    if supplied_color and supplied_color != coords.shape[0]:
        raise ValueError("prepared color self values must cover every block row")
    if (handle is not None and rescore and handle.color_self is not None
            and len(handle.color_self) != coords.shape[0]):
        raise ValueError("prepared color self values must cover every block row")
    rows = cpu.score_pairs(query_coords=qc, query_n_atoms=qn,
        library_coords=coords, library_counts=counts,
        pair_query_row=pq, pair_library_row=pl,
        # Prepared R3G library handles bind their content fingerprint to these
        # interaction matrices even when this call scores only shape.
        interaction_r=r, interaction_p=p, color=False, mixing=0.,
        steps=int(opt["steps"]), lr_q=float(opt["lr_q"]), lr_t=float(opt["lr_t"]),
        budget_override=opt.get("budget_override"), start_mode=int(start_mode),
        threads=1, prepared_query_fields=fields,
        final_start_policy=final_start_policy,
        prepared_shape_query_self=context.shape_self,
        # The worker's native collector checks every row these produce
        # against the self-overlap bound, naming the global conformer; checking here
        # too would name only the block-local row.
        **({"prepared_self_overlaps": handle.prepared_library_self_overlaps,
            "check_stored_self": False}
           if handle is not None else {}),
        field_h=spacing, field_pad=padding, field_width=8, field_unroll=0,
        field_pipeline=True, field_gridxf=1, field_allin=1, field_batched=True)
    if handle is not None:
        from ..scores import SELF_OVERLAP_LIBRARY
        reported = rows[:, SELF_OVERLAP_LIBRARY]
        # The shape path's documented conversion: reduced = float(VAA) *
        # float(1/SHAPE_PREFACTOR), reported back as reduced * prefactor, all
        # in float32. A row the scorer recomputed would not be bound to it.
        prefactor = float(cpu.extension().__shape_prefactor__)
        expected = ((np.asarray(handle.vaa, dtype=np.float32)[np.asarray(pl)]
                     * np.float32(1.0 / prefactor)) * np.float32(prefactor))
        if not np.array_equal(reported, expected):
            raise RuntimeError(
                "a scored row's library self-overlap is not its supplied "
                "stored value; the native shape path did not consume it")
    if library_self_supplied is not None:
        library_self_supplied.update(shape_rows=supplied_shape,
                                     color_rows=supplied_color)
    if rescore:
        shaped = rows
        rows, _ = cpu.extension().rescore_scalar_color(
            qc, qt, qn, coords, counts, pq, pl, r, p, shaped,
            float(mixing), False, False, False, 0,
            context.color_self, None if handle is None else handle.color_self)
        if not np.array_equal(rows[:, 9:20], shaped[:, 9:20]):
            raise RuntimeError("color rescore changed shape-selected pose/start")
    return rows

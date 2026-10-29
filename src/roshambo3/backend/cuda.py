"""Thin, validated wrapper over the CUDA extension.

There is deliberately very little here. ROSHAMBO2's equivalent grew to 561 lines
around an entry point with 11 method codes and roughly 20 keyword arguments,
because every measurement arm was added to the shipping signature and none were
ever removed -- its own docstring labels most of them "measurement arms". The
shipping surface here is one function and a handful of enums; arms live behind
``roshambo3.experimental``.

The extension is imported lazily, so the package stays usable with no GPU for
corpus preparation, format inspection and metrics.
"""

from __future__ import annotations

import enum
from typing import Mapping, Optional

import numpy as np

from ..layout import (BTILE, MAX_FEATURES, check_interaction_matrices,
                      library_counts_u8)
from ..numerics import assert_arithmetic_pinned, check_interaction_matrix
from .. import scores as _scores
from ..scores import N_SCORE_COLUMNS


from .spec import (  # noqa: F401  -- re-exported: `from .cuda import PoseSearch`
    ColorLoop,
    ColorPlacement,
    CoordResidency,
    DEFAULT_STAGE_WIDTH,
    PoseSearch,
    QUERY_ELEMENT_BYTES,
    QUERY_PLANES,
    REGISTER_WIDTHS,
    STAGED_CARVEOUT,
    STAGED_ELEMENT_BYTES,
    STAGED_PLANES,
    UpdateLanes,
    UpdateRule,
    analytic_block_shared_bytes,
    carveout_for,
)


#: The int64 per-call counters ``optimize_pairs(call_stats=...)`` writes, in
#: order. Must match the ``CALL_STATS_*`` enum in ``_cuda_src/interface.cpp``.
_CALL_STATS_FIELDS = (
    "query_self_evaluations",
    "library_self_evaluations",
    "library_self_supplied",
    "self_bound_rows_checked",
    "self_bound_rows_flagged",
    "self_bound_first_row",
)


def extension():
    """Import and return the CUDA extension, or raise with a useful message."""
    try:
        from .. import _roshambo3_cuda as ext  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            "roshambo3's CUDA extension is not available. The package installs "
            "without it when no CUDA toolkit is present, which is fine for "
            "corpus preparation and metrics but not for screening."
        ) from e
    return ext


def available() -> bool:
    try:
        extension()
        return True
    except ImportError:
        return False


def score_pairs(
    *,
    query_coords: np.ndarray,
    query_types: np.ndarray,
    query_n_atoms: np.ndarray,
    library_coords: np.ndarray,
    library_counts: np.ndarray,
    pair_query_row: Optional[np.ndarray],
    pair_library_row: Optional[np.ndarray],
    interaction_r: np.ndarray,
    interaction_p: np.ndarray,
    out: Optional[np.ndarray] = None,
    pose_search: PoseSearch = PoseSearch.FIELD,
    color: bool = False,
    mixing: float = 0.0,
    steps: int = 25,
    lr_q: float = 0.1,
    lr_t: float = 0.4,
    start_mode: int = 1,
    device_id: int = 0,
    field_spacing_a: float = 0.8,
    field_pad_a: float = 8.0,
    coord_residency: CoordResidency = CoordResidency.SHARED,
    min_blocks_per_sm: int = 4,
    smem_carveout: Optional[int] = None,
    update_lanes: UpdateLanes = UpdateLanes.PER_LANE,
    start_groups_per_row: Optional[int] = None,
    update_rule: UpdateRule = UpdateRule.ADAGRAD_EXP,
    stage_width: Optional[int] = None,
    color_loop: Optional[ColorLoop] = None,
    color_placement: ColorPlacement = ColorPlacement.AT_POSE,
    color_field_min_sources: int = 0,
    color_field_spacing_a: float = 0.0,
    lanes_per_group: int = 4,
    color_residency: int = 1,
    color_stage: int = 0,
    query_color_residency: int = 0,
    query_shape_width: int = 0,
    library_shape_width: int = 0,
    library_color_offset: int = 0,
    library_tile_mode: int = 1,
    color_ballot: bool = False,
    interleave: int = 0,
    stream_min_blocks_per_sm: Optional[int] = None,
    topk=None,
    gid_base: int = 0,
    max_rows_per_query: int = 0,
    check_self_overlap_bound: bool = False,
    call_stats: Optional[dict] = None,
    winning_starts: Optional[np.ndarray] = None,
    start_objectives: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Score (query row, library row) pairs on one device.

    ``pair_query_row`` and ``pair_library_row`` are the pair list. Both
    ``None`` means a **dense** launch: every query against every library row,
    query-major, so row ``q * n_library + t`` is query ``q`` against library
    row ``t``. A dense launch builds, scans and uploads no pair array; the
    device derives each row's query and library row from the launch shape.
    An explicit pair list is the sparse form and is read as given. The two
    forms score the same pairs bit-identically.

    Returns the ``[n_pairs, 20]`` score block, or ``None`` when ``topk`` is
    given -- in that case the dense block never leaves the device and there is
    nothing to return.

    ``winning_starts`` optionally receives native integer start IDs in a writable
    contiguous int32 ``[n_pairs]`` array. ``start_objectives`` receives the actual
    native final selection criterion for each start in writable contiguous
    float32 ``[n_pairs, n_starts]`` form. These diagnostics require a dense
    call. FIELD supports default dispatch cells and records its sampled search
    criterion separately from the final analytic rescore in the score columns.

    ``library_coords`` must be in the transposed 32-row tile layout
    (:func:`roshambo3.layout.tile32`); both kernels read it, so switching
    between them re-lays out nothing.

    ``library_tile_mode`` says WHICH tile: ``1`` the interleaved
    ``[tile][atom][lane][component]`` that is the default, ``-1`` the
    planar ``[tile][atom][component][lane]``
    (:func:`roshambo3.layout.tile32` with ``planar=True``). Nothing about the
    array itself says which one it is -- both are the same shape and the same
    size -- so reading one at the other's stride returns another conformer's
    coordinates and raises nothing. It is the caller's to state, and the caller
    is whoever packed the buffer. The two layouts read the same values in the
    same order, so they are bit-identical by construction; only the stride
    between a row's components changes.

    **Its dtype chooses the record.** ``int16 [n_rows, L, 3]`` is the wire
    record: the corpus's own coordinates at their stored wire step step, 6 B/atom,
    dequantized once on the device at the point of use. ``float32 [n_rows, L,
    4]`` is the four-component record, kept for callers whose coordinates are
    not on that grid -- the committed golden fixture, the ROSHAMBO2 port gate
    and the hand-built fixtures are all off-grid, and quantizing them would be a
    numerics change rather than an encoding. Nothing is converted here in either
    direction; the two are instantiations of the same kernels.

    ``library_counts`` is the per-conformer count vector, uint8
    ``[n_library, N_features]`` indexed by feature type id: **column 0 is the
    real atom count** and columns 1.. are the per-type feature counts. It
    replaces both the per-slot type plane and the separate ``library_n_atoms``
    array -- there is one number now rather than two that had to agree -- and,
    unlike the coordinates, it is **not tiled**: it is one 8-byte record per
    conformer, and a warp's 32 lanes hold 32 consecutive rows, so it is already
    coalesced. A type plane left in per-conformer order while the
    coordinates were tiled is not expressible here.

    ``library_color_offset`` is where a library conformer's color run begins.
    ``>= 0`` is an absolute slot, uniform over the launch, which is what a
    corpus block wants: every launch is single-bucket and
    ``format/dataset.py::combine_shape_color`` puts a library conformer's
    features at the group's PADDED shape width. ``-1`` means "immediately after
    each conformer's own real atoms", which is how ``engine/query.py`` lays a
    query out and how the hand-built fixtures lay a library out. It is
    **required for a color launch and refused for a shape one** rather than
    defaulted, because the natural-looking default -- the launch's largest real
    atom count -- is right only when every conformer has the same one, and
    silently wrong otherwise. The QUERY side needs no argument; its convention
    is fixed.

    ``topk`` is a :class:`~roshambo3.engine.TopK` or
    :class:`~roshambo3.engine.MoleculeTopK`; its ``device_level`` decides which
    of the two device states this block is folded into. Routing on the object
    rather than on two keyword arguments is deliberate -- a caller who passed a
    molecule-level state as ``topk=`` would get a conformer-level answer with no
    error, and every accuracy number after it would mean something else.

    Multi-GPU is the caller's job: run one of these per device with one top-K
    state each and merge with :func:`roshambo3.numerics.rank_order`. The top-K
    state holds its own atomics and cannot be shared across devices.

    ``check_self_overlap_bound`` (off by default) counts, on the device, every
    row of this launch whose pair overlap exceeds the mean of its self-overlaps
    by more than the recompute route's margin,
    ``roshambo3.scores.self_overlap_bound_margin("recompute")`` -- the
    self-overlap bound -- over the whole score block before any top-K reduction, so rows
    the denominator clamp reports as 0 are counted too. It only counts: what
    to do with a nonzero count belongs to the consumer that reads stored
    self-overlaps (none does on the GPU today, so on this
    recompute path it should be 0). Off, nothing is launched and the scores
    are bit-identical. ``call_stats``, if a dict, receives this call's
    counters: ``query_self_evaluations`` and ``library_self_evaluations``
    (self-overlaps evaluated on the device), ``library_self_supplied``
    (whether the library's were supplied rather than computed; always False
    today), and ``self_bound_rows_checked``, ``self_bound_rows_flagged`` and
    ``self_bound_first_row`` (each -1 when the scan was off; the first row is
    -1 when none was flagged). The same counts accumulate into
    ``extension().stage_timings()`` while stage accounting is enabled.
    """
    qc = np.ascontiguousarray(query_coords, dtype=np.float32)
    # The LIBRARY record's element type is the caller's, not this
    # function's. int16 [n_rows, L, 3] is the wire record -- the corpus's own
    # the wire-step quantization, 6 B/atom instead of 16, and what the engine sends.
    # float32 [n_rows, L, 4] is the four-component record. There is NO
    # conversion here in either direction: quantizing an off-grid float32
    # library would be a numerics change rather than an encoding (the golden
    # fixture, the ROSHAMBO2 port gate and the hand-built fixtures are all
    # off-grid), and widening the int16 one would put back the bytes the
    # wire record removed.
    tc = np.ascontiguousarray(library_coords)
    if tc.dtype != np.int16:
        tc = np.ascontiguousarray(tc, dtype=np.float32)
    qt = np.ascontiguousarray(query_types, dtype=np.int32)
    qn = np.ascontiguousarray(query_n_atoms, dtype=np.int32)
    tcnt = library_counts_u8(library_counts)
    if tcnt.ndim != 2:
        raise ValueError(
            f"library_counts must be [n_library, N_features] uint8 with column "
            f"0 the real atom count; got shape {tcnt.shape}. A shape-only "
            f"library is n_real.reshape(-1, 1).")
    if tcnt.shape[1] > MAX_FEATURES:
        # The device packs the vector into one 64-bit word and spends byte 7 on
        # the redundant color total, so seven type bytes is the ceiling.
        # Refused here, at the documented boundary, rather than only inside the
        # extension -- a caller with an eight-feature generator has a real
        # question and deserves to be told what the limit is and why.
        raise ValueError(
            f"library_counts is {tcnt.shape[1]} feature types wide. The "
            f"per-conformer count vector packs into one 64-bit word whose "
            f"byte 7 carries the redundant color total, so at most "
            f"{MAX_FEATURES} types fit.")
    ext = extension()
    # The arithmetic the loaded .so was compiled with is a checked invariant,
    # not a convention. Cached after the first call, so this is free.
    assert_arithmetic_pinned()

    # counts[:, 0] IS n_real. Named so the code below reads as it did, and
    # derived rather than passed so the two cannot disagree.
    tn = tcnt[:, 0].astype(np.int32)
    if (pair_query_row is None) != (pair_library_row is None):
        raise ValueError(
            "pass both pair_query_row and pair_library_row, or neither for a "
            "dense launch")
    dense = pair_query_row is None
    if not dense:
        pq = np.ascontiguousarray(pair_query_row, dtype=np.int32)
        pt = np.ascontiguousarray(pair_library_row, dtype=np.int32)
    rmat = np.ascontiguousarray(interaction_r, dtype=np.float32)
    pmat = np.ascontiguousarray(interaction_p, dtype=np.float32)

    if qc.ndim != 3 or qc.shape[2] != 4:
        raise ValueError(f"query_coords must be [n_q, L, 4], got {qc.shape}")
    lib_width = 3 if tc.dtype == np.int16 else 4
    if tc.ndim != 3 or tc.shape[2] != lib_width:
        raise ValueError(
            f"library_coords must be [n_rows, L, {lib_width}] for "
            f"{tc.dtype}, got {tc.shape}")
    if not dense and pq.shape != pt.shape:
        raise ValueError("pair_query_row and pair_library_row must be the same length")

    # ---- Relationships, not just shapes -------------------------------------
    #
    # The paths that BUILD these arrays are checked. This is the path that
    # CONSUMES them, and it has two ways around those checks: a
    # directly-constructed `QuerySet` (a plain dataclass, so `from_arrays`'
    # checks never ran) and a direct `score_pairs` call. Every check below
    # guards a read the device performs with no bounds check of its own, so the
    # failure mode without them is a wrong number or a segfault, not an
    # exception.
    #
    # Cost, measured rather than assumed: every check here is
    # O(1) except the two `pair_*` range checks, and those are the ones guarding
    # the read that actually crashes. `qn.max()` and `tn.max()` already run
    # unconditionally a few lines below, so the O(n) pass is not new to this
    # function.
    n_q_rows = int(qc.shape[0])
    if qn.ndim != 1 or qn.shape[0] != n_q_rows:
        raise ValueError(
            f"query_n_atoms must be one count per query row: expected "
            f"({n_q_rows},), got {qn.shape}")
    if qt.shape != qc.shape[:2]:
        raise ValueError(
            f"query_types must be [n_q, L] matching query_coords' first two "
            f"axes {qc.shape[:2]}, got {qt.shape}")
    if int(qn.min(initial=0)) < 0:
        raise ValueError("query_n_atoms has a negative count")
    if tcnt.shape[0] > int(tc.shape[0]):
        raise ValueError(
            f"library_counts has {tcnt.shape[0]} rows but library_coords has "
            f"{int(tc.shape[0])}; the per-row count would be read out of bounds")
    # NOT checked here, deliberately: `query_n_atoms.max()` against the padded
    # query width, and `library_counts[:, 0].max()` against the padded library
    # width. Both are already refused a few lines below by the
    # `query_shape_width` / `library_shape_width` bounds, on every path --
    # derived or explicitly passed. A second copy here would be two
    # definitions of one rule. `tests/test_color_path_cuda.py` pins the existing message.

    # The interaction matrices. `from_arrays` checks these, and only for the
    # queries it built, which leaves this path open. The device indexes `rmat[ta*N_features + tb]` with no bounds
    # check, so a short matrix is an out-of-bounds device read.
    check_interaction_matrices(rmat, pmat, qt)

    # The pair index. These are the only O(n) checks in this function, and
    # they guard the read that actually crashes: an index outside
    # range is an unchecked device gather.
    #
    # MEASURED, and it is why they are conditional. At a saturating launch
    # (62,077 conformers x 8 queries, GPU lock held) the three
    # range checks cost an upper bound of 1.13% of the launch wall.
    # That is above METRICS section 8's
    # 0.5% kernel-only resolvable delta, so it is not free and does not go
    # on the per-launch path unhoisted. At a
    # 15,364-conformer launch the same measurement reads 0.18%: the small block gives the wrong answer, and here it
    # gives it in the reassuring direction.
    #
    # THE CONTRACT, and it is narrow on purpose. A read-only pair array is a
    # statement by whoever built it that its range has already been checked and
    # that it cannot have changed since -- which is only assertable about an
    # array nobody can write to. (`Engine._pair_index` used to build exactly one
    # such pair per (n_q, n); the engine now launches dense and passes no
    # pair list at all, which has no index to range-check.)
    #
    # Deliberately NOT a `validate=` argument. A public flag that turns off a
    # safety check is a worse object than a check that costs 1%: it is one
    # keyword away from being wrong, and it would be wrong silently. Freezing
    # an array is a positive act with an unrelated cost of its own, so it is
    # not something a caller does by accident.
    #
    # Note `np.ascontiguousarray` returns the SAME object when the input is
    # already int32 and contiguous, preserving the flag; any conversion
    # produces a writeable copy, which is then checked. So the default
    # direction is to check.
    pair_index_is_frozen = dense or (
        (not pq.flags.writeable) and (not pt.flags.writeable))
    if not pair_index_is_frozen and pq.size:
        if int(pq.min()) < 0 or int(pq.max()) >= n_q_rows:
            raise ValueError(
                f"pair_query_row ranges over "
                f"[{int(pq.min())}, {int(pq.max())}] but there are only "
                f"{n_q_rows} query rows")
        n_lib_rows = int(tcnt.shape[0])
        if int(pt.min()) < 0 or int(pt.max()) >= n_lib_rows:
            raise ValueError(
                f"pair_library_row ranges over "
                f"[{int(pt.min())}, {int(pt.max())}] but there are only "
                f"{n_lib_rows} library conformers")
    if not query_shape_width:
        # A non-shared query-color arm stages only the contiguous real-atom
        # prefix. Its width is a property of the query panel, not of the
        # current library bucket's padded shape width.
        query_shape_width = (int(qc.shape[1]) if int(query_color_residency) == 0
                             else int(qn.max(initial=0)))
    if query_shape_width < int(qn.max(initial=0)) or query_shape_width > qc.shape[1]:
        raise ValueError(
            f"query_shape_width must cover every real query atom and be <= "
            f"the combined query width {qc.shape[1]}, got {query_shape_width}")
    if not library_shape_width:
        # The staged shape record is written only up to each row's real atom
        # count, but was sized by the COMBINED width L_shape + L_color -- so
        # 42-60% of it per bucket (57% conformer-weighted on the 10M color
        # corpus) was allocated and never touched, at the direct cost of the
        # blocks per SM the staged tier is limited by. The exact bound is the
        # largest real atom count in this launch, which is already to hand.
        library_shape_width = int(tn.max(initial=0))
    if (library_shape_width < int(tn.max(initial=0))
            or library_shape_width > tc.shape[1]):
        raise ValueError(
            f"library_shape_width must cover every real library atom and be <= "
            f"the combined library width {tc.shape[1]}, got {library_shape_width}")

    # The carveout is resolved FROM the residency tier and before anything
    # reads it. `None` does not mean "leave it alone" -- `cudaFuncSetAttribute`
    # is sticky, so an unset carveout is whatever the previous kernel left. It
    # means "the value that goes with this tier", and BOTH directions are
    # measured cells: a zero-shared tier at the staged arm's carveout 50 runs
    # 2.05-6.81x slower, and the staged tier at carveout 0 runs up to 2.4x
    # slower. A caller who names one and
    # not the other cannot construct either.
    coord_residency = CoordResidency(coord_residency)
    if smem_carveout is None:
        smem_carveout = carveout_for(coord_residency)

    n_library = tn.shape[0]
    padded = ((n_library + BTILE - 1) // BTILE) * BTILE
    if tc.shape[0] < padded:
        raise ValueError(
            f"library_coords has {tc.shape[0]} rows but {n_library} library molecules need "
            f"{padded} rows once padded to whole {BTILE}-row tiles -- pass the "
            f"output of roshambo3.layout.tile32")
    if int(library_tile_mode) == 0:
        raise ValueError(
            "library_tile_mode must be 1 (the interleaved tile) or -1 (the "
            "planar tile). Both pair kernels read a tile; 0 is "
            "the untiled per-conformer layout, which the CPU backend reads and "
            "neither of these kernels does.")

    # Library staging is shape-only -- the staged record holds coordinates and
    # the color loop still reads the type plane at the global stride -- so a
    # staging width is not a thing a color launch can have. `None` therefore
    # resolves to "the shipped width where staging applies, none where it does
    # not", which is what makes `score_pairs(color=True)` work at its defaults;
    # ScreenConfig.for_width already resolves it the same way. An *explicitly*
    # requested width is still refused rather than quietly ignored, because a
    # caller who named one has a wrong model of the kernel.
    if stage_width is None:
        stage_width = 0 if (color or PoseSearch(pose_search) is PoseSearch.FIELD) \
            else DEFAULT_STAGE_WIDTH

    pose_search = PoseSearch(pose_search)
    if int(color_field_min_sources) < 0:
        raise ValueError("color_field_min_sources must be >= 0")
    if int(color_field_min_sources) and not (
            color and pose_search is PoseSearch.FIELD
            and ColorPlacement(color_placement) is ColorPlacement.IN_GRADIENT):
        # A color field is only ever read from inside the optimizer loop:
        # at AT_POSE the color term is evaluated once, at the converged pose,
        # by the EXACT analytic pair sum, and on the analytic path there is no
        # texture to read at all. Refused rather than silently ignored -- a
        # caller who named it has a wrong model of where it applies.
        raise ValueError(
            "color_field_min_sources applies to PoseSearch.FIELD with "
            "color=True and ColorPlacement.IN_GRADIENT only: a color field is "
            "read from inside the optimizer loop, and at AT_POSE the color "
            "term is evaluated once at the converged pose by the exact "
            "analytic sum.")
    if pose_search is PoseSearch.FIELD:
        if UpdateRule(update_rule) not in (UpdateRule.ADAGRAD_QADD,
                                           UpdateRule.ADAGRAD_EXP):
            # The field kernel has an OPT template parameter, so both
            # SHIPPING rules reach it. The other three do not: Adam, QN2 and
            # BFGS each carry per-pose auxiliary state (8, 12 and 33 floats) on
            # top of `cache`, against a kernel pinned at 64 registers per thread
            # by __launch_bounds__(256, 4), and all three are analytic-path
            # measurement arms rather than shipping rules -- see
            # `experimental.SHIPPING_UPDATE_RULE`.
            raise ValueError(
                "the field path carries UpdateRule.ADAGRAD_QADD and "
                "UpdateRule.ADAGRAD_EXP only; ADAM_EXP, QN2_EXP and BFGS_EXP are "
                "instantiated on PoseSearch.ANALYTIC")
        if CoordResidency(coord_residency) is CoordResidency.REGISTER \
                and tc.shape[1] not in REGISTER_WIDTHS:
            raise ValueError(
                f"CoordResidency.REGISTER is instantiated only for padded widths "
                f"{REGISTER_WIDTHS}, not {tc.shape[1]}")

    # `None` resolves to "the structure this interaction matrix admits": the
    # same-type structures are exact only for the canonical diagonal matrix, so
    # a caller who supplied their own matrix and named no structure gets the one
    # that is right for it rather than a refusal. Resolving a default must not
    # become ignoring an argument, so an EXPLICIT same-type structure with
    # a non-canonical matrix is still refused below.
    canonical = check_interaction_matrix(rmat, pmat).canonical
    if color_loop is None:
        color_loop = (ColorLoop.SAME_TYPE_NOW if (canonical or not color)
                      else ColorLoop.BOUNDED)
    color_loop = ColorLoop(color_loop)
    if color_loop is ColorLoop.MASKED_PADDED:
        raise ValueError(
            "ColorLoop.MASKED_PADDED was retired. It was the only "
            "structure whose loop reached into the padding -- the last thing "
            "on the library side that read the 0/1 mask in the 4th coordinate "
            "component -- and ColorLoop.BOUNDED was measured "
            "reproducing it bitwise on the canonical matrix. Use "
            "ColorLoop.BOUNDED.")
    color_placement = ColorPlacement(color_placement)
    selected_combo_field = (color and pose_search is PoseSearch.FIELD
                            and color_placement is ColorPlacement.IN_GRADIENT)
    if start_groups_per_row is None:
        # S = 4 ships for the combo field path, fitted at start_mode 1's
        # four starts. It must be CLAMPED to the mode's start count, not applied
        # flat: start_mode 0 has a single start, so a flat 4 hands three of every
        # four groups a row and no work, and the launcher refuses it -- turning a
        # default into a hard error on a configuration that worked before.
        # The counts live in __constant__ memory and are not host-addressable,
        # so they are read through the extension's own accessor rather than
        # copied here; kernels.cu's note over require_signed_permutation_starts
        # is explicit that a second copy of those literals is a liability.
        if selected_combo_field:
            start_groups_per_row = min(4, int(ext.check_start_rotations(int(start_mode))))
        else:
            start_groups_per_row = 1
    if stream_min_blocks_per_sm is None:
        stream_min_blocks_per_sm = 2 if selected_combo_field else 0
    if color and color_loop in (ColorLoop.SAME_TYPE, ColorLoop.SAME_TYPE_NOW):
        # The same-type structures evaluate only ta == tb, so an off-diagonal
        # interaction is a term they never reach. On the canonical matrix that
        # term is exactly +0.0f and dropping it is value-preserving; on any
        # other matrix it is a real contribution and the structure would compute
        # a DIFFERENT quantity while reporting success. Refused rather than
        # silently downgraded to BOUNDED: a caller who set a non-canonical
        # matrix and a same-type loop has two intentions that cannot both hold.
        report = check_interaction_matrix(rmat, pmat)
        if not canonical:
            raise ValueError(
                f"color_loop={color_loop.name} evaluates only same-type feature "
                f"pairs, which is exact for the canonical diagonal interaction "
                f"matrix and wrong for any other. This matrix is not canonical: "
                f"{'; '.join(report.reasons)}. Use ColorLoop.BOUNDED, which "
                f"evaluates every pair and is still bounded to the real slots.")
        # Both same-type structures index the query's per-type segment
        # positionally, so a query plane whose color types are interleaved
        # rather than one ascending run per type would silently score a
        # different set of pairs. `generate_color_atoms` asserts this as a
        # postcondition; assert it again here, because this is the boundary the
        # kernel's assumption actually lives at and the arrays are tiny.
        for r in range(qt.shape[0]):
            row = qt[r][int(qn[r]):]
            nz = row[row != 0]
            if nz.size and not bool(np.all(nz[1:] >= nz[:-1])):
                raise ValueError(
                    f"query row {r}'s color types are not one ascending run per "
                    f"type ({nz.tolist()}), which color_loop="
                    f"{color_loop.name} indexes positionally. See "
                    f"roshambo3.pharmacophore.generate_color_atoms.")
    # Which structures are reachable where. `SAME_TYPE_NOW` is the default and is
    # everywhere; `BOUNDED` is everywhere too, because it is the only structure
    # that evaluates a cross-type pair and is therefore what a non-canonical
    # matrix resolves to at any update rule. `SAME_TYPE` is a measurement arm and
    # is instantiated for the additive rule only, so that the comparison between
    # structures cannot straddle two optimizers -- the same pin
    # bench/xcheck_roshambo2.py makes. (`MASKED_PADDED` was the third; it
    # is retired.)
    _everywhere = (ColorLoop.SAME_TYPE_NOW, ColorLoop.BOUNDED)
    # The field path implements SAME_TYPE_NOW and only that structure: it
    # hoists the per-type interaction constants into a per-block table, which is
    # exact for the canonical diagonal matrix and wrong for any other. Checked
    # AFTER the resolution above, so a caller who named nothing and supplied a
    # non-canonical matrix is told about the matrix -- the thing they actually
    # chose -- rather than about a structure they never asked for.
    if (color and pose_search is PoseSearch.FIELD
            and color_loop is not ColorLoop.SAME_TYPE_NOW):
        raise ValueError(
            f"PoseSearch.FIELD implements ColorLoop.SAME_TYPE_NOW only, and this "
            f"launch resolves to {color_loop.name}"
            + ("" if canonical else
               " because the interaction matrix is not canonical (its "
               "off-diagonal heights are not all zero), and a same-type "
               "structure would not evaluate the cross-type pairs it carries")
            + ". Use PoseSearch.ANALYTIC for the other structures.")

    if color_loop not in _everywhere and (
            not color or UpdateRule(update_rule) is not UpdateRule.ADAGRAD_QADD):
        raise ValueError(
            f"color_loop={color_loop.name} is a measurement arm and is "
            f"instantiated only for color=True with "
            f"update_rule=ADAGRAD_QADD -- got color={color}, "
            f"update_rule={UpdateRule(update_rule).name}. A shape launch never "
            f"enters the color loop at all, and the structure comparison is "
            f"pinned to the additive rule so it cannot straddle two "
            f"optimizers (the same pin bench/xcheck_roshambo2.py makes). "
            f"ColorLoop.BOUNDED and ColorLoop.SAME_TYPE_NOW are available at "
            f"every rule.")
    if not color and color_loop != ColorLoop.SAME_TYPE_NOW:
        raise ValueError(
            f"color_loop={color_loop.name} was named with color=False. A shape "
            f"launch never enters the color loop, so naming a structure for it "
            f"means the caller has the wrong model of the kernel.")
    # The one boundary that can still see an all-zero color library: a color
    # launch whose library carries no feature type at all returns a color
    # Tanimoto of exactly zero for every pair and reports success. Checked only
    # when color is on, so the shape path pays nothing.
    #
    # AFTER the pose-search checks above, deliberately. A caller asking for color
    # on the field path has made a wrong request and must hear that, not a
    # complaint about their data -- same ordering as api._check_color.
    #
    if color and not int(library_color_offset):
        # Refused, not defaulted. The natural-looking default -- the largest
        # real atom count -- is right only when every conformer in the launch
        # has the same one, and a launch is a whole size bucket. Getting it
        # wrong reads back as a color overlap of exactly +0.0 or as one
        # molecule's features scored against another's, neither of which
        # raises.
        raise ValueError(
            "color=True needs library_color_offset: the slot where a library "
            "conformer's color run begins. For a corpus block that is the "
            "group's PADDED shape width -- combine_shape_color puts a library "
            "conformer's features there, not immediately after its real atoms "
            "-- and ScreenConfig.for_width supplies it from the block's "
            "L_shape. Pass -1 for a plane whose features follow each "
            "conformer's own real atoms.")
    if not color and int(library_color_offset):
        raise ValueError(
            "library_color_offset was named with color=False. A shape launch "
            "has no color run, so naming where one begins means the caller has "
            "the wrong model of the kernel.")
    # The color half of the count vector has to FIT the plane it describes.
    # (The shape half already does: `library_shape_width` above defaults to
    # `counts[:, 0].max()` and is checked against the plane's width.) This is
    # where an off-by-one in a COLOR run boundary becomes loud: a run that walks
    # off the end of a conformer's record reads the NEXT conformer's slots --
    # another molecule's, in the tiled layout -- and nothing about the
    # resulting score says so. One pass over n_t x N_features bytes per launch,
    # which at 65,536 conformers is 458 KB against the launch's ~8 MB of H2D.
    #
    # `roshambo3 verify` makes the same check against the corpus; this one is at
    # the kernel boundary, where a hand-built library also arrives.
    if color:
        n_color = tcnt[:, 1:].sum(axis=1, dtype=np.int64)
        if int(library_color_offset) >= 0:
            reach = int(library_color_offset) + int(n_color.max(initial=0))
        else:
            # A negative offset puts each conformer's run after its OWN real
            # atoms, so the bound is per conformer, not per launch.
            reach = int((tn.astype(np.int64) + n_color).max(initial=0))
        if reach > tc.shape[1]:
            raise ValueError(
                f"a library conformer's color run reaches slot {reach} and the "
                f"library plane is {tc.shape[1]} wide. Either the count vector "
                f"does not describe this plane, or library_color_offset "
                f"({library_color_offset}) is wrong for it.")
    # Against the COUNTS, so the row slice means what it says. A tiled type
    # plane would need one whole TILE here, because there a conformer's
    # slots are strided by BTILE and `tt[:n_library]` is not "the first
    # n_library conformers" at all -- it is tile 0's first n_library*L
    # elements, which for a library narrower than one tile are all in the shape
    # region and therefore all type 0. The count vector is one row per conformer and
    # is not tiled, so `library_counts[:BTILE, 1:]` is exactly the first 32
    # conformers' feature counts and nothing else.
    if color and n_library and not tcnt[:BTILE, 1:].any():
        raise ValueError(
            f"color=True but library_counts records no pharmacophore feature "
            f"at all over the first {min(n_library, BTILE)} of {n_library} "
            f"library conformers, so every color overlap would be exactly "
            f"zero. This library carries no features.")

    n_pairs = n_q_rows * int(tcnt.shape[0]) if dense else int(pq.shape[0])
    metadata = {}
    if winning_starts is not None or start_objectives is not None:
        if topk is not None:
            raise ValueError("CUDA start metadata requires a dense call")
        if int(start_mode) not in (0, 1, 2):
            raise ValueError("invalid start_mode for CUDA start metadata")
        n_starts = (1, 4, 10)[int(start_mode)]
        for name, array, dtype, shape in (
            ("winning_starts", winning_starts, np.dtype(np.int32), (n_pairs,)),
            ("start_objectives", start_objectives, np.dtype(np.float32), (n_pairs, n_starts)),
        ):
            if array is None:
                continue
            if (not isinstance(array, np.ndarray) or array.dtype != dtype
                    or array.shape != shape or not array.flags.c_contiguous
                    or not array.flags.writeable):
                raise ValueError(f"{name} must be writable contiguous {dtype} {shape}")
            metadata[name] = array
        outputs = list(metadata.values()) + ([] if out is None else [out])
        if any(np.shares_memory(a, b) for i, a in enumerate(outputs) for b in outputs[i+1:]):
            raise ValueError("CUDA start metadata output buffers must not overlap")

    level = "none" if topk is None else str(getattr(topk, "device_level",
                                                   "conformer"))
    if level not in ("none", "conformer", "molecule"):
        raise ValueError(
            f"topk.device_level must be 'conformer' or 'molecule', got "
            f"{level!r}; see roshambo3.engine.sink")
    if topk is not None:
        # With a device top-K attached the dense block is never copied back, so
        # there is nothing for `out` to receive. A one-row dummy satisfies the
        # extension's shape assertion without allocating a block to throw away
        # -- at a 250k-conformer batch and 8 queries that would be 160 MB per
        # launch.
        v = np.zeros((1, N_SCORE_COLUMNS), dtype=np.float32)
    elif out is not None:
        v = out
        if v.shape != (n_pairs, N_SCORE_COLUMNS) or v.dtype != np.float32:
            raise ValueError(
                f"out must be [{n_pairs}, {N_SCORE_COLUMNS}] float32, "
                f"got {v.shape} {v.dtype}")
    else:
        # Not zeroed: the copy back overwrites every row the kernel owns.
        v = np.empty((n_pairs, N_SCORE_COLUMNS), dtype=np.float32)

    stats = (np.zeros(len(_CALL_STATS_FIELDS), dtype=np.int64)
             if call_stats is not None else None)
    ext.optimize_pairs(
        qc, qt, qn, tc, tcnt, rmat, pmat,
        None if dense else pq, None if dense else pt, v,
        optim_color=bool(color),
        mixing_param=float(mixing),
        lr_q=float(lr_q), lr_t=float(lr_t), nsteps=int(steps),
        start_mode=int(start_mode),
        pose_search=int(pose_search),
        device_id=int(device_id),
        field_spacing_a=float(field_spacing_a),
        field_pad_a=float(field_pad_a),
        update_lanes=int(UpdateLanes(update_lanes)),
        start_groups_per_row=int(start_groups_per_row),
        smem_carveout=int(smem_carveout),
        coord_residency=int(CoordResidency(coord_residency)),
        min_blocks_per_sm=int(min_blocks_per_sm),
        update_rule=int(UpdateRule(update_rule)),
        stage_width=int(stage_width),
        color_loop=int(color_loop),
        color_placement=int(color_placement),
        color_field_min_sources=int(color_field_min_sources),
        color_field_spacing_a=float(color_field_spacing_a),
        lanes_per_group=int(lanes_per_group),
        color_residency=int(color_residency),
        color_stage=int(color_stage),
        query_color_residency=int(query_color_residency),
        query_shape_width=int(query_shape_width),
        library_shape_width=int(library_shape_width),
        library_color_offset=int(library_color_offset),
        color_ballot=int(bool(color_ballot)),
        interleave=int(interleave),
        stream_min_blocks_per_sm=int(stream_min_blocks_per_sm),
        topk=(0 if level != "conformer" else int(topk.handle)),
        gid_base=int(gid_base),
        max_rows_per_query=int(max_rows_per_query or n_pairs),
        topk_molecule=(0 if level != "molecule" else int(topk.handle)),
        library_tile_mode=int(library_tile_mode),
        check_self_overlap_bound=bool(check_self_overlap_bound),
        # Read at call time from the one definition, not copied: CUDA
        # computes self-overlaps in-kernel, the recompute route.
        self_overlap_bound_margin=_scores.self_overlap_bound_margin(
            _scores.ROUTE_RECOMPUTE),
        call_stats=stats,
        **metadata,
    )
    if call_stats is not None:
        call_stats.update(zip(_CALL_STATS_FIELDS, (int(x) for x in stats)))
        call_stats["library_self_supplied"] = bool(
            call_stats["library_self_supplied"])
    return None if topk is not None else v

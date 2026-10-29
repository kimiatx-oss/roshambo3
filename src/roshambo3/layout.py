"""Coordinate layouts shared by the on-disk format, the producer and the kernel.

This module is the **single definition** of the transposed tile layout. The
format reader, the streaming producer and the CUDA kernel all have to agree
about the order of bytes in a library block, and nothing checks at runtime that
they do -- so there is exactly one implementation here and everything imports
it. It depends on nothing but NumPy, so both ``roshambo3.format`` and
``roshambo3.engine`` can use it without an import cycle.

``BTILE`` must equal ``ROSHAMBO3_BTILE`` in ``backend/_cuda_src/kernels.cuh``.
Changing one without the other is a silent data corruption, not a build error.

Why the layout exists
---------------------
The natural layout packs each library's atoms contiguously, so the 32 threads of
a warp -- which hold 32 *different* library molecules -- read 32 addresses ``L * 16`` bytes
apart, and every warp-wide 16-byte load touches 32 separate cache lines. Grouping
32 rows into a tile and transposing to ``[tile][atom][lane]`` puts lane ``i``'s
read of atom ``j`` next to lane ``i+1``'s, so the load coalesces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Rows per tile. Must match ``ROSHAMBO3_BTILE`` in ``kernels.cuh``.
BTILE = 32

#: The largest value a library count vector may hold -- column 0 the real
#: atom count, columns 1.. the per-type feature counts. The corpus stores them
#: as uint8 (``prep.build.MAX_N_REAL``) and every backend reads them as uint8,
#: so this is the largest library conformer any backend can score.
MAX_LIBRARY_COUNT = 255


def library_counts_u8(counts, name: str = "library_counts") -> np.ndarray:
    """``counts`` as a C-contiguous uint8 array, refusing what does not fit.

    A plain ``astype(np.uint8)`` wraps: a 256-atom conformer became 0 atoms
    and a 257-atom one 1, scored and reported with no error -- the failure
    ``prep.build`` already refuses at corpus build time, reachable again by
    any caller passing arrays.
    """
    a = np.asarray(counts)
    if a.dtype != np.uint8:
        if a.dtype.kind not in "biuf":
            raise TypeError(f"{name} must be integer counts, got {a.dtype}")
        wide = a.astype(np.int64)
        if a.dtype.kind == "f" and not np.array_equal(wide, a):
            raise ValueError(f"{name} must be whole numbers")
        if wide.size and (int(wide.min()) < 0
                          or int(wide.max()) > MAX_LIBRARY_COUNT):
            raise ValueError(
                f"{name} must lie in [0, {MAX_LIBRARY_COUNT}]; got "
                f"[{int(wide.min())}, {int(wide.max())}]. Counts are uint8 on "
                f"every backend, so a larger conformer cannot be scored.")
    return np.ascontiguousarray(a, dtype=np.uint8)


def tile32(a: np.ndarray, btile: int = BTILE, planar: bool = False) -> np.ndarray:
    """Transpose a ``[n, L, ...]`` per-conformer array into the tile layout.

    Groups of ``btile`` rows become ``[tile][atom][lane][...]`` -- that is,
    ``tile = row // btile`` and ``lane = row % btile`` -- with the trailing rows
    of the last tile zero-padded.

    The result keeps its original shape ``[n_padded, L, ...]``: the buffer it
    describes is the same size, only the order of the bytes inside it differs,
    and it is the kernel's address arithmetic that knows the difference.

    With ``planar``, the trailing axis moves inside the lane axis
    (``[tile][atom][component][lane]``), so a warp-wide load of one component
    covers 128 contiguous bytes with every byte used, rather than 512 bytes of
    which a quarter is used.
    """
    n, L = a.shape[0], a.shape[1]
    n_tiles = (n + btile - 1) // btile
    n_pad = n_tiles * btile
    tail = a.shape[2:]

    out = np.zeros((n_pad, L) + tail, dtype=a.dtype)
    out[:n] = a

    if planar and tail:
        perm = (0, 2) + tuple(range(3, 3 + len(tail))) + (1,)
    else:
        perm = (0, 2, 1) + tuple(range(3, 3 + len(tail)))

    out = np.ascontiguousarray(out.reshape((n_tiles, btile, L) + tail).transpose(perm))
    return out.reshape((n_pad, L) + tail)


#: Feature types a per-conformer count vector can carry. One 64-bit word
#: holds eight bytes and byte 7 is the redundant color total, so seven type
#: bytes fit. Must match ``ROSHAMBO3_MAX_FEATURES`` in ``kernels.cuh``.
MAX_FEATURES = 7


def check_interaction_matrices(interaction_r: np.ndarray,
                               interaction_p: np.ndarray,
                               types: np.ndarray) -> None:
    """The interaction matrices are square, agree, and index every type present.

    **One definition, called from both doors.** These checks were written in
    ``engine.query.from_arrays``, which guards only the queries it built --
    a directly-constructed ``QuerySet`` and a raw ``backend.score_pairs`` call
    both went round them. They are needed at the consuming boundary too, and a
    second copy there is exactly the drift to avoid, so the
    rule lives here and both callers import it. This module already holds the
    tile layout for the same reason: it depends on nothing but NumPy, so the
    format, the engine and the backend can all use it without an import cycle.

    Why each one is a memory-safety check and not a style check: the device
    indexes ``rmat[ta * N_features + tb]`` **with no bounds check**, so a type
    id outside the matrix -- in either direction -- is an out-of-bounds device
    read, not a wrong number.
    """
    r = np.asarray(interaction_r)
    q = np.asarray(interaction_p)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape != q.shape:
        raise ValueError(
            f"the interaction matrices must be square and the same shape; got "
            f"r{r.shape} and p{q.shape}")
    # Finiteness. A NaN or Inf width/height is not a
    # numbering or bounds problem -- it passes every check below -- and it
    # reaches `r^-1.5` or a term multiply and turns one cell's incoherence
    # into a NaN that contaminates an entire pair's Tanimoto, silently,
    # because IEEE 754 propagates NaN through `+` without raising anything.
    # Checked here, once, rather than by every consumer that happens to
    # divide by a width.
    if not (np.isfinite(r).all() and np.isfinite(q).all()):
        bad_r = np.argwhere(~np.isfinite(r))
        bad_p = np.argwhere(~np.isfinite(q))
        raise ValueError(
            "interaction_r/interaction_p must be finite; found non-finite "
            f"cells at r{bad_r[:4].tolist()} p{bad_p[:4].tolist()}")
    # The color prefactor contains r**-1.5. Zero is the documented
    # non-interacting sentinel; a negative width has no real-valued Gaussian
    # and would otherwise propagate NaN into the reported score.
    if (r < 0).any():
        bad = np.argwhere(r < 0)
        raise ValueError(
            "interaction_r widths must be nonnegative; found negative "
            f"cells at r{bad[:4].tolist()}")
    # The full validity range, stated once with the format: widths exactly 0 or >= 1e-6 (a tiny positive width gives
    # NaN on device), heights >= 0.
    from .format.r3g.reference import interaction_matrix_problems
    problems = interaction_matrix_problems(r, q)
    if problems:
        raise ValueError("interaction matrices: " + "; ".join(problems))
    ty = np.asarray(types)
    if not ty.size:
        return
    lo, hi = int(ty.min()), int(ty.max())
    if lo < 0:
        raise ValueError(
            f"color type {lo} is negative; feature types index the "
            f"interaction matrix and must be in [0, {r.shape[0] - 1}].")
    if hi >= r.shape[0]:
        raise ValueError(
            f"color type {hi} is outside a {r.shape[0]}x{r.shape[0]} "
            f"interaction matrix. Pass the same color_generator that assigned "
            f"the features.")


def counts_from_types(types: np.ndarray, n_real: np.ndarray,
                      n_features: int, *,
                      require_contiguous_from_n_real: bool = False) -> np.ndarray:
    """The per-conformer count vector for an explicit per-slot type plane.

    ``[n, n_features]`` uint8: column 0 the real atom count, column ``t`` the
    number of slots carrying feature type ``t``. This describes the same per-row prefix as the format-3 decoder, and
    it is what a caller with a hand-built plane -- a test fixture, a
    cross-check, an ad-hoc library -- hands to ``backend.score_pairs`` in place
    of the plane.

    **The count vector describes a plane whose feature slots form one
    contiguous ascending run per type, and no other**, so that is checked here
    rather than assumed. A plane that interleaves types cannot be represented,
    and the failure it would otherwise cause is a silent one: donors scored
    against acceptors. ``pharmacophore.generate_color_atoms`` sorts its output
    and asserts the same postcondition at generation time; this is the same
    invariant at the other end.

    Slots at or past ``n_real`` carrying type 0 are padding and are ignored;
    slots BEFORE ``n_real`` are real atoms and their type is not consulted.

    ``require_contiguous_from_n_real`` (default ``False``): additionally
    refuse a PADDING GAP -- a zero slot that precedes a nonzero one, inside
    ``[n_real, L)``. Off by default because this function also serves
    SPLIT-record callers (color pseudo-atoms at a launch-wide
    ``L_shape``, not at each row's own ``n_real`` -- ``test_color_field_cuda
    .py``'s fixture is one), for whom the padding between ``n_real`` and the
    color block is the SHAPE plane's own padding, not a gap, and where an
    offset carried separately (``library_color_offset``) locates the run --
    this function only has to report how many of each type. The two
    MERGED-record consumers (:func:`color_view_from_types`,
    :func:`check_color_layout`), which derive absolute positions from
    ``n_real`` alone with no separate offset, pass ``True``.
    """
    types = np.asarray(types, dtype=np.int64)
    n_real = np.asarray(n_real, dtype=np.int64)
    if types.ndim != 2:
        raise ValueError(f"types must be [n, L]; got shape {types.shape}")
    if n_real.shape != (types.shape[0],):
        raise ValueError(
            f"n_real must be [{types.shape[0]}]; got shape {n_real.shape}")
    n_features = int(n_features)
    if n_features < 1 or n_features > MAX_FEATURES:
        raise ValueError(
            f"n_features must be in [1, {MAX_FEATURES}]; got {n_features}. "
            f"One 64-bit word holds eight bytes and byte 7 carries the "
            f"redundant color total.")
    out = np.zeros((types.shape[0], n_features), np.uint8)
    out[:, 0] = n_real
    for r in range(types.shape[0]):
        row = types[r][int(n_real[r]):]
        nz_idx = np.flatnonzero(row)
        if nz_idx.size:
            if require_contiguous_from_n_real:
                # Reviewer correction: a padding (type 0) slot that precedes
                # a nonzero one inside [n_real, L) is a GAP a MERGED-record
                # consumer cannot tolerate -- it derives absolute positions
                # as n_real + (a running sum of counts), which assumes the
                # nonzero slots occupy EXACTLY [n_real, n_real+n_color) with
                # nothing skipped. [1, 0, 1] used to filter the zero out and
                # see [1, 1], trivially non-decreasing, and accept it; the
                # count vector that produced ({t=1: 2}) then made a merged
                # consumer read the padding slot as the second feature and
                # miss the real one past it.
                last = int(nz_idx[-1])
                if nz_idx.size != last + 1:
                    raise ValueError(
                        f"row {r}'s color run has a padding gap: nonzero "
                        f"slots at {nz_idx.tolist()} (relative to n_real) "
                        f"are not contiguous from the start of the color "
                        f"run. A merged-record consumer requires real "
                        f"features to occupy exactly "
                        f"[n_real, n_real+n_color) with no padding slot "
                        f"before the last one -- see "
                        f"roshambo3.pharmacophore.generate_color_atoms.")
            nz = row[row != 0]
            if not bool(np.all(nz[1:] >= nz[:-1])):
                raise ValueError(
                    f"row {r}'s color types are not one contiguous ascending "
                    f"run per type ({nz.tolist()}). The per-conformer count "
                    f"vector cannot represent interleaving -- see "
                    f"roshambo3.pharmacophore.generate_color_atoms.")
            if int(nz.max()) >= n_features or int(nz.min()) < 0:
                raise ValueError(
                    f"row {r} carries feature type {int(nz.max())}, outside "
                    f"[1, {n_features - 1}]")
            for t in range(1, n_features):
                out[r, t] = int((nz == t).sum())
    return out


# ---------------------------------------------------------------------------
# The one read-only CPU color view, and the boundary validation.
#
# Query types are explicit (a per-atom type array that must be DECODED into
# counts and validated for contiguity); library types are already a count
# vector (there is no per-atom array to decode -- the counts ARE the input).
# That is a real asymmetry, and a trap: the two sides
# cannot share one decoding call, only the offset arithmetic once the counts
# exist. `segment_offsets_from_counts` is that one shared derivation;
# `color_view_from_types` and `color_view_from_counts` are the two decoders
# that both end by calling it, so a later scalar, SIMD or field path that
# wants segment offsets gets them from here and never re-derives them.
# ---------------------------------------------------------------------------

def segment_offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    """The ONE segment-offset derivation, shared by both decode paths below.

    ``counts`` is ``[..., n_features]`` (the count vector: column 0 is ``n_real``,
    never a feature). Returns ``offsets`` of the same shape and dtype
    ``int64``: ``offsets[..., t]`` is feature type ``t``'s start, relative to
    the conformer's OWN color run -- so absolute column ``n_real +
    offsets[..., t]`` is where type ``t``'s slots begin. ``offsets[..., 0]``
    is always 0 and is never read (type 0 is never a feature).

    A cumulative sum over columns ``1..``, computed once. Both
    :func:`color_view_from_types` (which must first turn per-atom types into
    a count vector) and :func:`color_view_from_counts` (which is handed one
    directly) end here, so there is exactly one place that turns "how many of
    each type" into "where each type starts".
    """
    counts = np.asarray(counts)
    if counts.shape[-1] < 1:
        raise ValueError(
            f"counts must carry at least column 0 (n_real); got shape "
            f"{counts.shape}")
    offsets = np.zeros(counts.shape, dtype=np.int64)
    if counts.shape[-1] > 1:
        c = counts[..., 1:].astype(np.int64)
        cum = np.cumsum(c, axis=-1)
        offsets[..., 1:] = cum - c
    return offsets


@dataclass(frozen=True)
class ColorView:
    """The one read-only CPU view of a batch of color conformers.

    Shape coordinates are ``xyz[:, :n_real]`` of whatever coordinate array
    this view was built alongside; color coordinates are the contiguous run
    ``xyz[:, n_real : n_real + n_color]``. This dataclass carries the
    bookkeeping (counts, offsets, totals) and not the coordinates themselves,
    so it applies identically to a query row (``[L, 3]`` or ``[L, 4]``) and a
    library row of a different padded width -- the L histogram is a
    property of the coordinate array, not of this view.

    Attributes:
        n_real: ``[n]`` int64, real (shape) atom count.
        counts: ``[n, n_features]`` uint8, the count vector (column 0 is
            ``n_real`` again, so a consumer that wants one array has it).
        offsets: ``[n, n_features]`` int64, from
            :func:`segment_offsets_from_counts`.
        n_color: ``[n]`` int64, ``counts[:, 1:].sum(axis=1)`` -- the total
            REAL feature count. Never conflate this with a padded width:
            a batched kernel's max feature count across a
            group is a property of the group, not of one conformer, and is
            not carried here.
    """
    n_real: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    n_color: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.counts.shape[1])

    def __len__(self) -> int:
        return int(self.counts.shape[0])


def color_view_from_counts(library_counts: np.ndarray) -> ColorView:
    """The LIBRARY decoder: the count vector already IS the counts.

    No per-atom type array exists on this side to decode -- library feature
    types are reconstructed from the contiguous-run convention the count
    vector encodes, never read explicitly. Do not call
    :func:`color_view_from_types` here and discard the type array it expects;
    that is exactly the "one decoder called twice" trap.
    """
    counts = np.asarray(library_counts, dtype=np.uint8)
    if counts.ndim != 2:
        raise ValueError(
            f"library_counts must be [n_library, n_features]; got shape "
            f"{counts.shape}")
    if counts.shape[1] > MAX_FEATURES:
        raise ValueError(
            f"library_counts carries {counts.shape[1]} columns, more than "
            f"MAX_FEATURES={MAX_FEATURES} (one 64-bit word, byte 7 "
            f"reserved for the redundant color total)")
    n_real = counts[:, 0].astype(np.int64)
    n_color = (counts[:, 1:].astype(np.int64).sum(axis=1)
              if counts.shape[1] > 1 else np.zeros(counts.shape[0], np.int64))
    offsets = segment_offsets_from_counts(counts)
    return ColorView(n_real=n_real, counts=counts.copy(), offsets=offsets,
                     n_color=n_color)


def color_view_from_types(query_types: np.ndarray, query_n_atoms: np.ndarray,
                          n_features: int) -> ColorView:
    """The QUERY decoder: per-atom types are explicit and must be decoded.

    Reuses :func:`counts_from_types` for the contiguity and bounds check --
    the same "one contiguous ascending run per type" invariant the library
    side is handed already satisfied -- and then the one shared offset
    derivation. This is a different function from
    :func:`color_view_from_counts`, deliberately: there is no count vector to
    hand it here, only a per-atom type plane, and that is the asymmetry
    itself, not an oversight to be papered over by one shared entry point.
    """
    counts = counts_from_types(query_types, query_n_atoms, n_features,
                              require_contiguous_from_n_real=True)
    n_color = (counts[:, 1:].astype(np.int64).sum(axis=1)
              if n_features > 1 else np.zeros(counts.shape[0], np.int64))
    offsets = segment_offsets_from_counts(counts)
    return ColorView(n_real=np.asarray(query_n_atoms, dtype=np.int64),
                     counts=counts, offsets=offsets, n_color=n_color)


def color_loop_is_same_type_safe(interaction_r: np.ndarray,
                                 interaction_p: np.ndarray) -> bool:
    """Whether a same-type-only ("canonical fast path") color loop is exact
    here.

    **This is the fast-path predicate scope item 3 asks for, and it is not a
    new one**: `numerics.check_interaction_matrix` already answers this exact
    question for the CUDA seam's `color_loop` resolution
    (`backend/cuda.py:428`, `ColorLoop.SAME_TYPE`/`SAME_TYPE_NOW`), so this is
    that function under the name a CPU dispatcher reads at a decision point,
    not a fork of its contract.

    A same-type loop visits only ``ta == tb`` segments. That is exact iff
    every OFF-diagonal interaction height is zero: an off-diagonal cell with
    nonzero width is still VISITED under the general loop and, on a
    canonical matrix, contributes exactly zero because its height is zero --
    it is not the same statement as "skipped for zero width"
    (``doc/COLOR_CONTRACT.md`` section 1). A matrix that is canonical except
    for one off-diagonal height returns ``False`` here, and a same-type loop
    run against it anyway silently drops that cell's real contribution --
    which is exactly the failure `tests/test_color_census.py` demonstrates by
    comparing this function's answer against a term count that does and does
    not include that cell.
    """
    from .numerics import check_interaction_matrix
    return bool(check_interaction_matrix(interaction_r, interaction_p).canonical)


def check_color_layout(interaction_r: np.ndarray, interaction_p: np.ndarray,
                       query_types: Optional[np.ndarray],
                       query_n_atoms: np.ndarray, query_L: int,
                       library_counts: np.ndarray, library_L: int, *,
                       library_weight: Optional[np.ndarray] = None) -> None:
    """The one color-input validation at the Python/backend boundary.

    Every scope-item-2 check, in order, each raising ``ValueError`` naming
    what failed -- called BEFORE a color request reaches the extension, so a
    malformed input fails here rather than being silently scored shape-only
    or reaching a device-side unbounded index.

    1. **Matrix shape and finiteness, and the query's feature ids are in
       bounds** -- :func:`check_interaction_matrices`, not forked: this
       function is the one place both checks live, and this validator calls
       it rather than repeating it.
    2. **The query type plane matches the query coordinate rows and width**;
       the color decoder walks to ``query_L`` and cannot safely read a
       shorter plane. **The feature-count width is bounded** (7 type columns fit in the
       reserved word) and **query and library agree on the enumeration
       WIDTH**: ``library_counts`` must have exactly as many columns as the
       interaction matrix has rows, or a column read as one feature type on
       one side is read as a different type -- or padding -- on the other.
       This is a shape check, not a semantic one: it cannot confirm the two
       sides were assigned under the SAME ``FEATURES_ENUM`` ordering, only
       that they could not disagree on how many types there are. Numeric
       type equality without an enumeration stamp is not semantic
       equality, and closing that gap for real is still open.
    3. **Contiguous runs**, on the query's explicit per-atom types --
       :func:`counts_from_types`'s own check, surfaced here rather than
       repeated.
    4. **``n_real + sum(color_counts) <= L``**, on both sides. This is a
       memory-safety bound, not a style check: nothing downstream
       bounds-checks a count vector that claims more feature slots than the
       padded row actually has room for.
    5. **The merged-record offset**, when the caller can state it: if
       ``library_weight`` (the coordinate plane's 4th, occupancy, component)
       is given, no weight may be set at or beyond ``n_real + n_color``. A
       weighted slot out there is a GAP between the shape run and the color
       run -- a split record's signature, and this seam reads only the
       merged one (shape then color, contiguous, per conformer). Refused
       with a migration message rather than silently read as if contiguous.

    **Empty planes are legal input and pass every check above with no
    special-casing.** A query with zero color atoms, or a library row whose
    count-vector columns ``1..`` are all zero, is a zero-feature
    molecule -- the guard that turns that into a finite zero color score
    lives one level up, in :func:`reference.color.guarded_tanimoto`, and does
    not belong in a layout check.
    """
    r = np.asarray(interaction_r)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        # check_interaction_matrices raises the precise message; this branch
        # only protects the n_features read two lines down from an IndexError
        # on a malformed matrix, which would otherwise be a confusing crash
        # instead of check_interaction_matrices's own ValueError.
        check_interaction_matrices(interaction_r, interaction_p, np.zeros(0))
    n_features = int(r.shape[0])
    if n_features > MAX_FEATURES:
        raise ValueError(
            f"the interaction matrix is {n_features}x{n_features}, more "
            f"than MAX_FEATURES={MAX_FEATURES} columns a count vector "
            f"can carry (one 64-bit word, byte 7 reserved)")

    qt_plane = (np.asarray(query_types) if query_types is not None else None)
    qt = (qt_plane.reshape(-1) if qt_plane is not None
          else np.zeros(0, np.int64))
    check_interaction_matrices(interaction_r, interaction_p, qt)
    if qt_plane is not None:
        expected = (np.asarray(query_n_atoms).size, int(query_L))
        if qt_plane.shape != expected:
            raise ValueError(
                "query_types must match query_coords' [n_query, L] shape; "
                f"got {qt_plane.shape} for coordinates {expected}")

    lc = np.asarray(library_counts)
    if lc.ndim != 2:
        raise ValueError(
            f"library_counts must be [n_library, n_features]; got shape "
            f"{lc.shape}")
    if lc.shape[1] != n_features:
        raise ValueError(
            f"library_counts carries {lc.shape[1]} columns but the "
            f"interaction matrix is {n_features}x{n_features}: query and "
            f"library must be read against the SAME feature enumeration "
            f"width, or a column read as one type on one side is read as a "
            f"different type (or nothing) on the other.")

    if lc.shape[1] > 1:
        n_real_l = lc[:, 0].astype(np.int64)
        n_color_l = lc[:, 1:].astype(np.int64).sum(axis=1)
    else:
        n_real_l = lc[:, 0].astype(np.int64)
        n_color_l = np.zeros(lc.shape[0], np.int64)
    over_l = (n_real_l + n_color_l) > int(library_L)
    if over_l.any():
        bad = np.argwhere(over_l)[:4, 0].tolist()
        raise ValueError(
            f"library rows {bad} claim n_real + sum(color_counts) > "
            f"L={library_L}; the count vector describes more feature slots "
            f"than the padded record has room for")

    if qt.size:
        qn = np.asarray(query_n_atoms, dtype=np.int64)
        # Raises ValueError by itself on a non-contiguous run or an
        # out-of-range type -- scope items 2's bounds check and 3's
        # contiguity check, both already implemented there and reused here
        # rather than repeated.
        qcounts = counts_from_types(np.asarray(query_types), qn, n_features,
                                  require_contiguous_from_n_real=True)
        n_color_q = (qcounts[:, 1:].astype(np.int64).sum(axis=1)
                    if n_features > 1 else np.zeros(qcounts.shape[0], np.int64))
        over_q = (qn + n_color_q) > int(query_L)
        if over_q.any():
            bad = np.argwhere(over_q)[:4, 0].tolist()
            raise ValueError(
                f"query rows {bad} claim n_real + sum(color_counts) > "
                f"L={query_L}")

    if library_weight is not None:
        w = np.asarray(library_weight)
        end = n_real_l + n_color_l
        cols = np.arange(w.shape[1])[None, :]
        gap = (cols >= end[:, None]) & (w > 0)
        if gap.any():
            bad = np.argwhere(gap.any(axis=1))[:4, 0].tolist()
            raise ValueError(
                f"library rows {bad} carry an occupied coordinate slot at or "
                f"beyond n_real + n_color. The library layout requires a "
                f"contiguous heavy-atom prefix followed by the color features "
                f"and zero-weight padding.")

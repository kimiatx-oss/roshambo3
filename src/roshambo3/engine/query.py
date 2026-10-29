"""The query side: a small batch of prepared conformers, padded to one width.

Queries are few -- a screen is one to a few dozen of them against a corpus of
1e8 to 1e11 conformers -- so nothing here needs to be clever. They are prepared
in process, by the same :mod:`roshambo3.prep` the corpus builder uses, because
the query and the library have to be in the same coordinate frame or the score is
meaningless. ROSHAMBO2 had two implementations of that frame and they differed.

**The query lives on the library's grid.** Every query coordinate -- shape atoms
and color pseudo-atoms alike -- is put on the library wire record's wire-step grid
here, once, by exactly the format's operations (:func:`snap_to_wire_grid`), and
the query's self-overlap is computed from those coordinates. Otherwise the CUDA
analytic kernel reads the query as an int16 record on that grid while
``V_AA`` comes from the unrounded float32 query, so a query not already on the
grid has its Tanimoto numerator and denominator describe two coordinate sets
(a +3.2e-5 mean bias on the golden queries). Snapping here gives
every backend -- CUDA field and analytic, CPU, Metal -- one query coordinate
set. A query taken from a corpus is already on the grid and is unchanged.

The CUDA analytic path's rotating starts re-quantize the rotated query at
the wire step. The shipped starts are signed permutations, so this reproduces
the snapped integers exactly: numerator and self term describe the same
coordinates. A start set containing other rotations would need a separate
representation check. Analytic query self terms are also reconstructed through
the wire record for raw direct backend.score_pairs calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..layout import BTILE, check_interaction_matrices
from ..prep import prep_conformer
from ..prep.conformer import stored_self_volume


def snap_to_wire_grid(xyz, what: str = "query") -> np.ndarray:
    """Put coordinates on the library wire record's wire-step grid, as float32.

    Exactly the format's two operations, in the corpus builder's order
    (``prep/build.py``: ``fmt.quantize(r)``, then
    ``q.astype(np.float32) * np.float32(fmt.COORD_SCALE)``): round to nearest
    int16 at the format's step, then dequantize with one float32 multiply by
    ``numerics.WIRE_QUANT_STEP``. That constant is never folded into the
    reduced frame's ``COORD_SCALE``: the device applies the two as two separate
    multiplies (``r3_query_coord``), and a folded constant rounds differently
    on 31.9% of int16 values (NUMERICS_POLICY section 4c).

    The identity on coordinates already on the grid -- every int16 value
    survives the round trip bit for bit -- so a query taken from a corpus is
    unchanged. A coordinate past the int16 range (x outside [-128.0, +127.99609375] A) raises: the
    query cannot be represented on the grid the kernel reads it at, and
    clipping it would move an atom without saying so.
    """
    from ..format.constants import quantize
    from ..numerics import WIRE_QUANT_STEP

    try:
        q = quantize(xyz)
    except OverflowError as exc:
        raise ValueError(
            f"{what} has a coordinate outside the wire step wire grid's int16 "
            f"range ([-128.0, +127.99609375] A in the prepared frame): {exc}. Queries are "
            f"put on the library's grid so every backend scores one coordinate "
            f"set; a coordinate there cannot be represented, and is refused "
            f"rather than clipped.") from exc
    return q.astype(np.float32) * np.float32(WIRE_QUANT_STEP)


def check_on_wire_grid(coords) -> None:
    """Refuse query coordinates that are not on the wire grid.

    ``coords`` is a padded ``[n_q, L, 3|4]`` query record; its xyz, as the
    float32 every backend reads, must be what :func:`snap_to_wire_grid` makes
    of them (padding is 0 and stays 0). A non-finite or out-of-range
    coordinate raises as it does in :func:`from_arrays`.
    """
    a = np.asarray(coords)
    if a.ndim != 3 or a.shape[2] < 3 or a.size == 0:
        return  # malformed shapes are refused where the record is consumed
    xyz = np.asarray(a[..., :3], dtype=np.float32)
    snapped = snap_to_wire_grid(xyz, "QuerySet.coords")
    off = snapped != xyz
    if off.any():
        query = int(np.argwhere(off)[0][0])
        raise ValueError(
            f"QuerySet.coords has {int(off.sum())} of {xyz.size} coordinates "
            f"off the library wire record's wire-step grid (first in query "
            f"{query}). Every backend must score one query coordinate set: "
            f"build the set with from_arrays or from_rdkit, which snap the "
            f"coordinates and compute vaa from the snapped ones, or snap them "
            f"with snap_to_wire_grid (and recompute vaa) before constructing "
            f"a QuerySet by hand.")


@dataclass
class QuerySet:
    """Prepared query conformers, ready for the kernel.

    ``coords`` is ``[n_q, L, 4]`` float32 with the fourth column marking real
    atoms, ``types`` is ``[n_q, L]`` int32 (0 for a real atom, >= 1 for a color
    feature), and ``n_real`` is the real-atom count per query. Built by
    :func:`from_arrays` (or :func:`from_rdkit`), every coordinate lies on the
    library wire record's wire-step grid (:func:`snap_to_wire_grid`).
    """

    coords: np.ndarray
    types: np.ndarray
    n_real: np.ndarray
    names: list[str]
    interaction_r: np.ndarray
    interaction_p: np.ndarray
    vaa: np.ndarray
    """Gaussian self-overlap per query in physical cubic Angstrom, for reference
    and filter calibration: ``prep.conformer.stored_self_volume`` of the snapped
    shape atoms in ``coords``. No backend reads it; each computes its own query
    self-overlap from ``coords``, so all of them describe one coordinate set."""

    def __post_init__(self):
        # Every construction goes through here, so a QuerySet built by hand
        # (or handed straight to api.load_queries / Engine.run) cannot carry
        # coordinates off the grid that from_arrays snaps to. It is a refusal,
        # not a snap: moving the coordinates here would leave a caller's
        # ``vaa`` describing other coordinates. Arrays mutated after
        # construction are not re-checked.
        check_on_wire_grid(self.coords)

    @property
    def n_conformers(self) -> int:
        return int(self.coords.shape[0])

    @property
    def L(self) -> int:
        return int(self.coords.shape[1])


def from_arrays(coords: Sequence[np.ndarray], names: Sequence[str],
                *, color_generator=None, backend=None,
                color: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None
                ) -> QuerySet:
    """Build a :class:`QuerySet` from already-prepared coordinate blocks.

    Each entry of ``coords`` is ``[n_real_i, 3]`` in the prepared frame. They are
    padded to a common width here; the width is *not* rounded up to a tile,
    because queries are indexed individually and never read as a warp-wide tile.

    Every coordinate, shape and color, is snapped to the library wire record's
    the wire-step grid (:func:`snap_to_wire_grid`) from the precision it is passed in
    -- float64 straight out of preparation rounds exactly as the corpus builder
    rounds it -- and ``vaa`` is computed from the snapped coordinates. Queries
    already on the grid are unchanged; a coordinate past [-128.0, +127.99609375] A raises.
    """
    # Kept at the caller's floating precision (anything else becomes float64)
    # so the snap below rounds from what preparation produced.
    raw = [a if a.dtype.kind == "f" else a.astype(np.float64)
           for a in (np.asarray(x) for x in coords)]
    if not raw:
        raise ValueError("a query set needs at least one conformer")
    for i, b in enumerate(raw):
        if b.ndim != 2 or b.shape[1] != 3:
            # A 1-D (3,) array used to succeed: nr became 3, and
            # `out[i, :3, :3] = b` broadcast one row into three, fabricating a
            # 3-atom query out of a single point with no error anywhere.
            raise ValueError(
                f"query {i} has shape {b.shape}; each block must be "
                f"[n_atoms, 3] in the prepared frame")
        if b.shape[0] == 0:
            raise ValueError(f"query {i} has no atoms")
        if not np.all(np.isfinite(b)):
            # A NaN here does not raise downstream, it *disappears*: the device
            # top-K drops a NaN score rather than ranking it, so the screen
            # returns an empty hit list and reports no error at all.
            raise ValueError(f"query {i} has non-finite coordinates")
    # The one place a query's coordinates are fixed. Everything below --
    # the padded record every backend reads and `vaa` -- uses these.
    blocks = [snap_to_wire_grid(b, f"query {i}") for i, b in enumerate(raw)]
    names = list(names)
    if len(names) != len(blocks):
        raise ValueError(
            f"{len(names)} names for {len(blocks)} queries; the mismatch used "
            f"to surface much later as an IndexError while building results")
    n_real = np.array([b.shape[0] for b in blocks], dtype=np.int32)

    if color is None:
        color = [(np.zeros((0, 3), np.float32), np.zeros(0, np.int32))] * len(blocks)
    else:
        color = list(color)
        if len(color) != len(blocks):
            # Both zips below truncate to the shorter list, so a short `color`
            # made L a function of a prefix of the queries and a long one was
            # dropped -- neither said anything.
            raise ValueError(
                f"{len(color)} color entries for {len(blocks)} queries")
        for i, (cc, ct) in enumerate(color):
            cc = np.asarray(cc)
            ct = np.asarray(ct)
            if cc.size and (cc.ndim != 2 or cc.shape[1] != 3):
                raise ValueError(
                    f"query {i} color coordinates have shape {cc.shape}, "
                    f"expected [n_features, 3]")
            if ct.shape[0] != cc.shape[0]:
                raise ValueError(
                    f"query {i} has {cc.shape[0]} color coordinates but "
                    f"{ct.shape[0]} types")
            if cc.size and not np.all(np.isfinite(cc)):
                raise ValueError(f"query {i} has non-finite color coordinates")
            # Color pseudo-atoms go on the same grid: the library's color
            # record is quantized at the same step (prep/build.py).
            color[i] = (snap_to_wire_grid(cc.reshape(-1, 3), f"query {i} color")
                        if cc.size else np.zeros((0, 3), np.float32), ct)
    n_total = np.array([b.shape[0] + np.asarray(c[0]).shape[0]
                        for b, c in zip(blocks, color)], dtype=np.int32)
    L = int(n_total.max())

    out = np.zeros((len(blocks), L, 4), np.float32)
    types = np.zeros((len(blocks), L), np.int32)
    for i, (b, (cc, ct)) in enumerate(zip(blocks, color)):
        nr = b.shape[0]
        out[i, :nr, :3] = b
        out[i, :nr, 3] = 1.0
        if cc.shape[0]:
            out[i, nr:nr + cc.shape[0], :3] = cc
            out[i, nr:nr + cc.shape[0], 3] = 1.0
            types[i, nr:nr + cc.shape[0]] = ct

    if color_generator is not None:
        rmat = np.ascontiguousarray(color_generator.interaction_matrix_r, np.float32)
        pmat = np.ascontiguousarray(color_generator.interaction_matrix_p, np.float32)
    else:
        # Shape only: a 1x1 identity interaction, which the kernel never reads
        # because the color loop is bounded by the real atom count.
        rmat = np.ones((1, 1), np.float32)
        pmat = np.zeros((1, 1), np.float32)
    # One definition, in roshambo3.layout, called from here and from
    # backend.score_pairs, so a directly-constructed QuerySet and a raw
    # score_pairs call cannot go round it.
    check_interaction_matrices(rmat, pmat, types)

    # The query's own bound, checked here rather than discovered at
    # screen time after a corpus has been opened and a panel built. A query has
    # no uint8 sidecar, so none of `prep.build._why_reject`'s library limits
    # apply to it and MAX_N_REAL = 255 would be a wrong bound that looks
    # principled. What bounds a query is shared memory.
    #
    # Checked at the MOST FAVORABLE configuration -- the unstaged rung, where
    # the staged library costs nothing -- so this refuses only a query no
    # configuration on this device could run. A query inside this bound but
    # large enough to crowd the staged library is not refused: the launcher
    # steps `stage_width` down the ladder and it runs, more slowly. That is a
    # throughput consequence, not a correctness one, and refusing it here would
    # reject a query the engine could actually screen.
    backend_name = getattr(backend, "__name__", backend)
    if backend_name not in ("cpu", "roshambo3.backend.cpu", "metal", "roshambo3.backend.metal"):
        from .config import query_atom_limit, shared_memory_budget
        limit = query_atom_limit(shared_memory_budget(0),
                                 n_features=int(rmat.shape[0]))
        if int(out.shape[1]) > limit:
            widest = int(np.argmax([b.shape[0] for b in blocks]))
            raise ValueError(
                f"query {names[widest]!r} pads the panel to {int(out.shape[1])} "
                f"atoms, past the {limit} this device's {shared_memory_budget(0):,}-byte "
                f"shared-memory budget allows even unstaged. The panel is padded to "
                f"its widest member, so one oversized query bounds the whole set.")

    # From the snapped coordinates, with the direct-difference evaluation the
    # stored-frame library sidecars use, not the norm/dot-product
    # identity of the historical prepared-frame `self_volume`.
    vaa = np.array([stored_self_volume(b) for b in blocks], np.float32)
    return QuerySet(coords=out, types=types, n_real=n_real,
                    names=names, interaction_r=rmat, interaction_p=pmat,
                    vaa=vaa)


def from_rdkit(mols, *, color_generator=None, names=None, backend=None) -> QuerySet:
    """Prepare RDKit molecules into a :class:`QuerySet`.

    One conformer per molecule entry. Hydrogens are removed for the shape
    calculation, as everywhere; when a color generator is given, features are
    assigned before removal so hydrogens can inform them, and the pseudo-atoms
    are placed in the *real atoms'* frame rather than their own.
    """
    from rdkit import Chem

    from ..prep.conformer import strip_hydrogens

    coords, out_names, color = [], [], []
    for i, m in enumerate(mols):
        heavy = strip_hydrogens(m)
        raw = np.asarray(heavy.GetConformer().GetPositions(), dtype=np.float64)
        prepared = prep_conformer(raw)
        # float64 on purpose: from_arrays snaps it to the wire grid from the
        # precision the corpus builder rounds from, so a query prepared from
        # the same conformer is bit-identical to its library record.
        coords.append(prepared)
        nm = (names[i] if names is not None
              else (m.GetProp("_Name") if m.HasProp("_Name") else f"query_{i}"))
        out_names.append(nm)
        if color_generator is not None:
            cc, ct = color_generator.generate_color_atoms(m)
            cc = np.asarray(cc, dtype=np.float64)
            placed = (prep_conformer(cc, align_ref=raw)
                      if cc.shape[0] else np.zeros((0, 3), np.float64))
            color.append((placed, np.asarray(ct, dtype=np.int32)))
        else:
            color.append((np.zeros((0, 3), np.float32), np.zeros(0, np.int32)))

    return from_arrays(coords, out_names, color_generator=color_generator,
                       color=color, backend=backend)

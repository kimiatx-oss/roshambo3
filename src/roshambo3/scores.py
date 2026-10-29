"""The score row, given a name.

A result row is 20 float32 columns. In ROSHAMBO2 that ``20`` was an untyped
integer that leaked everywhere -- a ``V.shape(1) == 20`` assertion in C++, a
``dv=20`` default on the top-K constructor, pandas conversion by integer index
(``scores[:, 3]``), and a "pass a one-row dummy to satisfy the assertion" hack
in the Python wrapper. Naming the columns costs nothing and makes all of that
readable.

Two column groups are dead on the production path and worth about 20% of every
top-K row and every dense score copy:

* ``START_QUAT`` is always ``(1, 0, 0, 0)`` on the field path, because the
  start-mode rotation is folded into the reported pose;
* the four color columns are zero until color joins the exact rescore.

Shrinking the row is therefore a real optimization, and it is deliberately
*not* done yet: the first bit-identity comparison against ROSHAMBO2 has to run
at the same width, or a mismatch cannot be told apart from a layout change.
``dv`` is a runtime parameter everywhere for that reason.
"""

from __future__ import annotations

import numpy as np

#: Columns in a score row.
N_SCORE_COLUMNS = 20

# Scalars.
TANIMOTO_COMBINATION = 0
TANIMOTO_SHAPE = 1
TANIMOTO_COLOR = 2
OVERLAP_VOLUME = 3
OVERLAP_COLOR = 4
SELF_OVERLAP_QUERY = 5
SELF_OVERLAP_LIBRARY = 6
SELF_COLOR_QUERY = 7
SELF_COLOR_LIBRARY = 8

# The pose. Applied as r @ R(q).T + t, then the start-mode rotation.
POSE_QUAT = slice(9, 13)
POSE_TRANSLATION = slice(13, 16)
START_QUAT = slice(16, 20)

#: Structured view of one row, for reading a result without integer indices.
SCORE_DTYPE = np.dtype([
    ("tanimoto_combination", "<f4"),
    ("tanimoto_shape", "<f4"),
    ("tanimoto_color", "<f4"),
    ("overlap_volume", "<f4"),
    ("overlap_color", "<f4"),
    ("self_overlap_query", "<f4"),
    ("self_overlap_library", "<f4"),
    ("self_color_query", "<f4"),
    ("self_color_library", "<f4"),
    ("pose_quat", "<f4", (4,)),
    ("pose_translation", "<f4", (3,)),
    ("start_quat", "<f4", (4,)),
])
assert SCORE_DTYPE.itemsize == N_SCORE_COLUMNS * 4

#: Column name -> index, for building a table without a magic number in sight.
COLUMN_INDEX = {
    "tanimoto_combination": TANIMOTO_COMBINATION,
    "tanimoto_shape": TANIMOTO_SHAPE,
    "tanimoto_color": TANIMOTO_COLOR,
    "overlap_volume": OVERLAP_VOLUME,
    "overlap_color": OVERLAP_COLOR,
    "self_overlap_query": SELF_OVERLAP_QUERY,
    "self_overlap_library": SELF_OVERLAP_LIBRARY,
    "self_color_query": SELF_COLOR_QUERY,
    "self_color_library": SELF_COLOR_LIBRARY,
}


#: How far a row's overlap may exceed the mean of its two self-overlaps before
#: :func:`check_self_overlap_bound` refuses it, relative. Gaussian overlaps
#: obey ``V_AB <= sqrt(V_AA V_BB) <= (V_AA + V_BB) / 2``, so only rounding
#: and caller-supplied stored self-overlaps from a different coordinate
#: preparation can cross the bound. The margin admits known reduced-coordinate
#: precision differences and catches larger inconsistencies.
#:
#: This is the ONE definition of the value. Every bound check reads it through
#: :data:`SELF_OVERLAP_MARGIN_POLICY` and :func:`self_overlap_bound_margin`:
#: the native CPU worker sink (``worker_local_topk_sink.hpp``) receives it per
#: call or per block, the CUDA bound scan and the Metal float32 predicate
#: receive it per call, and a native caller that passes no margin gets it
#: read from here at call time. ``tests/test_self_overlap_routes.py`` fails if
#: another copy of the literal appears.
SELF_OVERLAP_BOUND_MARGIN = 1e-3

#: The tight tier (accepted, NOT active): for
#: self-overlaps that describe the scored coordinates, only rounding separates
#: them from the bound -- a synthetic 255-atom self-match reaches +2e-6,
#: and 1e-5 keeps 5x headroom.
SELF_OVERLAP_BOUND_MARGIN_TIGHT = 1e-5

#: Self-overlap routes: where a scored row's self-overlaps come from.
#:
#: * ``verified_stored`` -- format-3 stored-frame values, row-bound and
#:   validated at binding by the CPU worker route;
#: * ``recompute`` -- values computed from the scored coordinates by direct CPU
#:   calls, CUDA and Metal.
ROUTE_VERIFIED_STORED = "verified_stored"
ROUTE_RECOMPUTE = "recompute"

#: Whether the tight tier is in force. It switches on only once all five of
#: its activation conditions hold (among them a LIT-PCBA rerun, the color
#: channel's measured excess, and a premerge run) and the maintainer says so. While False, every route uses
#: SELF_OVERLAP_BOUND_MARGIN.
SELF_OVERLAP_MARGIN_TIGHT_ACTIVE = False

#: The margin each route's bound check uses: ``route -> (while the tight tier
#: is inactive, once it is active)``. The one place a margin is chosen.
SELF_OVERLAP_MARGIN_POLICY = {
    ROUTE_VERIFIED_STORED: (SELF_OVERLAP_BOUND_MARGIN, SELF_OVERLAP_BOUND_MARGIN_TIGHT),
    ROUTE_RECOMPUTE: (SELF_OVERLAP_BOUND_MARGIN, SELF_OVERLAP_BOUND_MARGIN_TIGHT),
}


def self_overlap_bound_margin(route: str) -> float:
    """The relative bound margin for rows whose self-overlaps took ``route``,
    under the policy in force now (read at call time)."""
    try:
        inactive, active = SELF_OVERLAP_MARGIN_POLICY[route]
    except KeyError:
        raise ValueError(
            f"unknown self-overlap route {route!r}; expected one of "
            f"{sorted(SELF_OVERLAP_MARGIN_POLICY)}") from None
    return float(active if SELF_OVERLAP_MARGIN_TIGHT_ACTIVE else inactive)


class InconsistentSelfOverlapError(ValueError):
    """A scored row's self-overlaps do not describe the conformers scored."""


def scan_self_overlap_route(block_routes) -> "str | None":
    """Classify a scan from its blocks' routes (``route -> block count``).

    None for no blocks; the one route when every block took it; otherwise the
    combination of verified-stored and recomputed blocks. Both routes describe
    the scored coordinates and use the same margin.
    """
    present = sorted(route for route, count in dict(block_routes).items() if count)
    for route in present:
        if route not in SELF_OVERLAP_MARGIN_POLICY:
            raise ValueError(f"unknown self-overlap route {route!r}")
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return "+".join(route for route in
                    (ROUTE_VERIFIED_STORED, ROUTE_RECOMPUTE) if route in present)


def check_scan_self_overlap_routes(block_routes) -> "str | None":
    """Classify a scan (:func:`scan_self_overlap_route`)."""
    return scan_self_overlap_route(block_routes)


def check_self_overlap_bound(scores: np.ndarray, conformer_ids=None,
                             query: str = "", *,
                             id_label: str = "conformer",
                             route: str) -> None:
    """Refuse rows whose overlap exceeds the mean of their self-overlaps.

    ``2 V_AB > V_AA + V_BB`` is exactly a shape Tanimoto above 1. It also
    holds wherever the Tanimoto denominator ``V_AA + V_BB - V_AB`` is not
    positive, which every backend reports as a Tanimoto of 0 -- so one test
    covers the inflated rows that rank first and the clamped ones that rank
    last. Empty rows (all three zero) pass. ``route`` names where the rows'
    self-overlaps came from and so which margin applies
    (:data:`SELF_OVERLAP_MARGIN_POLICY`). It is required: a default would be
    one tier for every caller that forgot it, and the only safe default
    (the loosest) would silently admit what a tighter route refuses once
    the tight tier is active.
    """
    a = np.asarray(scores)
    if a.ndim != 2 or a.shape[1] != N_SCORE_COLUMNS or a.shape[0] == 0:
        return
    margin = self_overlap_bound_margin(route)
    overlap = a[:, OVERLAP_VOLUME].astype(np.float64)
    selves = (a[:, SELF_OVERLAP_QUERY].astype(np.float64)
              + a[:, SELF_OVERLAP_LIBRARY].astype(np.float64))
    bad = np.flatnonzero(2.0 * overlap > selves * (1.0 + margin))
    if bad.size == 0:
        return
    i = int(bad[0])
    who = (f"{id_label} {int(np.asarray(conformer_ids)[i])}"
           if conformer_ids is not None else f"row {i}")
    raise InconsistentSelfOverlapError(
        f"{bad.size} scored row(s){' for ' + query if query else ''} report a "
        f"pair overlap above the mean of their self-overlaps, which consistent "
        f"inputs cannot produce; first: {who}, overlap {a[i, OVERLAP_VOLUME]:g}, "
        f"self-overlaps {a[i, SELF_OVERLAP_QUERY]:g} (query) and "
        f"{a[i, SELF_OVERLAP_LIBRARY]:g} (library). The self-overlaps do not "
        "describe the scored coordinates -- usually a stored self-overlap "
        "sidecar from a different preparation.")


def as_records(scores: np.ndarray) -> np.ndarray:
    """View an ``[n, 20]`` float32 block as an ``[n]`` array of named records.

    A view, not a copy: the underlying buffer must be C-contiguous float32.
    """
    a = np.ascontiguousarray(scores, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != N_SCORE_COLUMNS:
        raise ValueError(f"expected [n, {N_SCORE_COLUMNS}] float32, got {a.shape}")
    return a.view(SCORE_DTYPE).reshape(a.shape[0])


def tanimoto_combo_legacy(scores: np.ndarray) -> np.ndarray:
    """ROSHAMBO2's ``tanimoto_combo_legacy``: shape + color, so out of 2.

    Kept because published comparisons quote it. It is not the combination
    Tanimoto in column 0, which is a properly mixed single score.
    """
    a = np.asarray(scores)
    return a[:, TANIMOTO_SHAPE] + a[:, TANIMOTO_COLOR]

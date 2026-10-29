"""The closed-form CPU color operation census.

**Not a kernel, and not derived from what a kernel "should" do.** A shape
census derived that way was 26.6% low because a derivation of
the loop's own behavior is exactly the kind of estimate that goes wrong
quietly, and nothing could fail on it because an error that loosens a
denominator is invisible to every ratio computed from it. There is no color
kernel yet to instrument, so this module computes what a CORRECT one would visit, directly
from the color view (:class:`roshambo3.layout.ColorView`) and the
interaction matrices -- the ceiling future instrumented counters are
checked against, and the object
``tests/test_color_census.py`` checks against hand-worked canonical,
general, sparse and empty cases before anything downstream is allowed to
trust it.

Two things this module is careful to keep apart:

* **Terms versus visited slots.** Every count here is of REAL
  feature-pair terms. A batched or SIMD implementation additionally visits
  PADDING slots up to the group's max feature count; that quantity is not
  computed here (there is no batch shape in a per-pair closed form) and must
  never be added into a terms counter -- an implementation that pads reports
  its own ``color_padding_visits`` separately, against this ceiling.
* **Skipped-for-zero-width versus contributes-zero (COLOR_CONTRACT.md
  section 1).** An interaction cell with ``r == 0`` is never visited at all;
  a cell with ``r != 0, p == 0`` is visited and its term is exactly zero.
  Only the first is ``color_zero_width_terms``; the second is counted as an
  ordinary (zero-valued) term, because a same-type fast path that skips it
  is skipping something a general loop would have visited, and that is
  precisely what makes the fast path unsafe on a non-canonical matrix (see
  :func:`roshambo3.layout.color_loop_is_same_type_safe`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..layout import ColorView


def color_term_census(query_view: ColorView, pair_query_row: np.ndarray,
                      library_view: ColorView, pair_library_row: np.ndarray,
                      interaction_r: np.ndarray, interaction_p: np.ndarray, *,
                      same_type_fast_path: bool,
                      color_search_steps: int = 0,
                      evaluate_gradient_every_step: bool = False) -> dict:
    """Closed-form color term counts for a batch of (query, library) pairs.

    Args:
        query_view / library_view: built by
            :func:`roshambo3.layout.color_view_from_types` /
            :func:`color_view_from_counts`.
        pair_query_row / pair_library_row: the pair list, indexing the two
            views -- the same convention ``backend.cpu.score_pairs`` takes.
        interaction_r / interaction_p: the ``[n_features, n_features]``
            matrices, ``[query_type, library_type]``.
        same_type_fast_path: price the canonical same-type loop (visits only
            ``ta == tb``) rather than the general one (every ``(ta, tb)``
            the matrix's own nonzero-width structure admits). Callers should
            gate this with :func:`roshambo3.layout.color_loop_is_same_type_safe`
            rather than deciding it themselves -- passing ``True`` on a
            non-canonical matrix here reproduces the exact undercount an
            unsafe fast-path loop would, which is the point of the
            ``test_same_type_fast_path_undercounts_a_near_canonical_matrix``
            test, not a bug in this function.
        color_search_steps: how many optimizer steps evaluate color (0 for
            ``shape_only``/``shape_pose_color_rescore``, which do not score
            color during the search at all -- COLOR_CONTRACT.md section 7 --
            and the mode's own step count for ``combo_optimal``).
        evaluate_gradient_every_step: whether each priced step also evaluates
            the color GRADIENT (true only for ``combo_optimal``, whose
            objective the gradient must follow every step; false rescores
            the value only, once, winner-only).

    Scope item 4 asks for a bucket for "per-start final scoring" as well as
    "winner-only exact rescore" -- this function has only the latter
    (``color_final_rescore_terms``). There is deliberately no separate
    per-start bucket: a caller pricing "evaluate color once per START rather
    than winner-only" gets that number from THIS SAME function by passing
    ``pair_query_row``/``pair_library_row`` expanded by the start count
    (one repeated (query, library) entry per start) rather than one entry
    per pair -- the per-pair loop above does not know or care how many
    times one (query, library) combination appears in the pair list, so
    that expansion is exactly what prices "per start" instead of
    "winner-only" with no new code path.

    Returns:
        A dict of scalar counts plus ``"color_plane_terms"``, an
        ``[n_features, n_features]`` int64 array of REAL term counts keyed by
        ``(query_type, library_type)`` (a merged color record has
        its own histogram, kept per-type rather than collapsed to one
        number).
    """
    r = np.asarray(interaction_r, dtype=np.float64)
    p = np.asarray(interaction_p, dtype=np.float64)
    if r.shape != p.shape or r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"interaction matrices must be square and agree; "
                         f"got r{r.shape} p{p.shape}")
    n_features = r.shape[0]
    if query_view.n_features != n_features or library_view.n_features != n_features:
        raise ValueError(
            f"view/matrix enumeration mismatch: query n_features="
            f"{query_view.n_features}, library n_features="
            f"{library_view.n_features}, matrix is {n_features}x{n_features}")

    qi = np.asarray(pair_query_row, dtype=np.int64)
    li = np.asarray(pair_library_row, dtype=np.int64)
    if qi.shape != li.shape:
        raise ValueError("pair_query_row and pair_library_row must be the "
                         "same length")

    nonzero_width = (r != 0.0)                     # doc contract section 1
    if same_type_fast_path:
        cell_mask = np.eye(n_features, dtype=bool) & nonzero_width
    else:
        cell_mask = nonzero_width.copy()
    # Type 0 is never a feature (COLOR_CONTRACT.md section 2); a real (query,
    # library) pair can carry it as padding only, and a padding row already
    # reads zero out of `.counts`, but the mask is zeroed explicitly so a
    # matrix that carries garbage in row/column 0 cannot leak a term.
    cell_mask[0, :] = False
    cell_mask[:, 0] = False
    zero_width_mask = np.zeros_like(cell_mask)
    if not same_type_fast_path:
        zero_width_mask = (~nonzero_width)
        zero_width_mask[0, :] = False
        zero_width_mask[:, 0] = False

    qc = query_view.counts.astype(np.int64)
    lc = library_view.counts.astype(np.int64)

    plane_terms = np.zeros((n_features, n_features), dtype=np.int64)
    zero_width_terms = 0
    value_terms_per_pair = np.zeros(qi.shape[0], dtype=np.int64)

    for row in range(qi.shape[0]):
        # Outer product of one pair's per-type counts -- [n_features,
        # n_features], term count per (ta, tb) cell before any mask.
        outer = np.outer(qc[int(qi[row])], lc[int(li[row])]).astype(np.int64)
        plane_terms += outer * cell_mask
        value_terms_per_pair[row] = int((outer * cell_mask).sum())
        if not same_type_fast_path:
            zero_width_terms += int((outer * zero_width_mask).sum())

    n_final_rescore = int(value_terms_per_pair.sum())     # winner-only, once/pair
    n_value_search = n_final_rescore * int(color_search_steps)
    n_gradient_search = (n_value_search if evaluate_gradient_every_step else 0)

    q_used = np.unique(qi) if qi.size else np.zeros(0, np.int64)
    l_used = np.unique(li) if li.size else np.zeros(0, np.int64)
    # One-time work: paid once per ROW that appears in the pair list, however
    # many pairs use it -- not once per pair, which is what makes it
    # "one-time" rather than a per-pair term.
    query_prep_terms = int(query_view.n_color[q_used].sum()) if q_used.size else 0

    def _self_color(counts_used: np.ndarray):
        """The ordered self double sum for each row in ``counts_used``
        (``[n_used, n_features]``), masked EXACTLY like a pair term: by
        ``cell_mask`` (same-type-only when the caller priced that loop, the
        matrix's own nonzero-width structure otherwise) and by
        ``zero_width_mask`` for the tally of what a general loop would have
        visited-and-skipped.

        Reviewer correction: this used to be the unconditional
        ``n_color ** 2`` -- right only when every present type pair is
        same-type and nonzero-width, which a single-type molecule can never
        distinguish from the general case (the reason no test caught it).
        Self-color of a row with per-type counts ``c`` is
        ``sum_{ta,tb} c[ta] * c[tb] * cell_mask[ta, tb]``: the row plays
        BOTH the query and the library side of its own ordered pair sum,
        exactly as ``reference.color``'s ``S_AA`` does.
        """
        real = 0
        zero = 0
        for c in counts_used:
            outer = np.outer(c, c)
            real += int((outer * cell_mask).sum())
            zero += int((outer * zero_width_mask).sum())
        return real, zero

    query_self_color_terms, query_self_color_zero_width_terms = (
        _self_color(qc[q_used]) if q_used.size else (0, 0))
    library_self_color_terms, library_self_color_zero_width_terms = (
        _self_color(lc[l_used]) if l_used.size else (0, 0))

    # 4 bytes/float, 3 components, real color coordinates only -- read once
    # per ROW that is used, same one-time accounting as the self-color terms
    # (a batched kernel that streams a row once per PAIR instead is a
    # different, worse, and separately reportable number).
    color_bytes_read = 12 * (int(query_view.n_color[q_used].sum()) if q_used.size else 0)
    color_bytes_read += 12 * (int(library_view.n_color[l_used].sum()) if l_used.size else 0)

    return {
        "n_pairs": int(qi.shape[0]),
        "same_type_fast_path": bool(same_type_fast_path),
        "query_prep_terms": query_prep_terms,
        "query_self_color_terms": query_self_color_terms,
        "query_self_color_zero_width_terms": query_self_color_zero_width_terms,
        "library_self_color_terms": library_self_color_terms,
        "library_self_color_zero_width_terms": library_self_color_zero_width_terms,
        "color_value_search_terms": int(n_value_search),
        "color_gradient_search_terms": int(n_gradient_search),
        "color_final_rescore_terms": n_final_rescore,
        "color_zero_width_terms": int(zero_width_terms),
        "color_plane_terms": plane_terms,
        "color_bytes_read": int(color_bytes_read),
    }


def shape_l_histogram(n_atoms: np.ndarray, n_color: Optional[np.ndarray] = None
                      ) -> dict:
    """Matched L histograms, made checkable: a shape-only ``L`` histogram against a
    color one.

    **Returns an actual histogram, not a three-number summary.** Two
    distributions can share all three (a bimodal one and a
    unimodal one at the same mean, for instance) while having genuinely
    different padded-batch costs, so "identical L histograms" cannot be checked by comparing summaries. ``counts`` is
    ``np.bincount`` of the REAL width (``n_atoms`` for shape-only,
    ``n_atoms + n_color`` for a merged color record), index ``i`` holding how
    many rows have width exactly ``i``; two configurations compared for
    throughput must have this array equal, or the comparison is between data
    layouts, not kernels. min/mean/max are kept alongside for a quick human
    read, not as the check itself.
    """
    n_atoms = np.asarray(n_atoms, dtype=np.int64)
    total = (n_atoms if n_color is None
            else n_atoms + np.asarray(n_color, dtype=np.int64))
    counts = np.bincount(total) if total.size else np.zeros(0, np.int64)
    return {
        "n_rows": int(total.shape[0]),
        "min": int(total.min()) if total.size else 0,
        "mean": float(total.mean()) if total.size else 0.0,
        "max": int(total.max()) if total.size else 0,
        "has_color": n_color is not None,
        "counts": counts,
    }

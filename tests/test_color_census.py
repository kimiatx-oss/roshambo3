"""The closed-form CPU color operation census.

A shape census that was once 26.6% low is the reason this file exists: no
downstream ratio could have caught it, because an error
that loosens a denominator is invisible to every ratio computed from it.
`reference.color_census.color_term_census` is checked here against
HAND-COMPUTED counts on small, exact cases rather than only against itself
or against a kernel.
"""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import layout
from roshambo3.reference.color_census import (color_term_census,
                                              shape_l_histogram)


def _matrix(n_features, canonical=True, perturb_off_diagonal=None):
    r = np.ones((n_features, n_features))
    p = np.zeros((n_features, n_features))
    for t in range(1, n_features):
        p[t, t] = 1.0
    if perturb_off_diagonal is not None:
        p[perturb_off_diagonal] = 0.5
    return r, p


class TestHandComputedCases:
    """Canonical, general, sparse, empty -- worked out by hand in each
    docstring/comment, then checked against the function."""

    def test_canonical_two_types_general_vs_fast_path(self):
        """query: type1 x3, type2 x1. library: type1 x3, type2 x1.

        General (every (ta,tb) with r!=0, which is every cell of a 3x3
        matrix here): (1,1)=3*3=9, (1,2)=3*1=3, (2,1)=1*3=3, (2,2)=1*1=1;
        total 16. Fast (same-type only): (1,1)=9, (2,2)=1; total 10. Worked
        by hand and asserted exactly -- this is the number the module must
        reproduce, not merely be self-consistent about.
        """
        n_features = 3
        r, p = _matrix(n_features)
        lv = layout.color_view_from_counts(np.array([[2, 3, 1]], np.uint8))
        qv = layout.color_view_from_types(
            np.array([[0, 0, 1, 1, 1, 2]], np.int64), np.array([2]), n_features)
        qi, li = np.array([0]), np.array([0])

        gen = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=False)
        fast = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=True)
        assert gen["color_final_rescore_terms"] == 16
        assert fast["color_final_rescore_terms"] == 10
        assert gen["color_plane_terms"].tolist() == [[0, 0, 0],
                                                      [0, 9, 3],
                                                      [0, 3, 1]]
        assert fast["color_plane_terms"].tolist() == [[0, 0, 0],
                                                       [0, 9, 0],
                                                       [0, 0, 1]]
        # Canonical matrix: r != 0 everywhere, so nothing is EVER skipped for
        # zero width under the general loop -- the off-diagonal cells are
        # visited and contribute exactly zero (p=0), which is a different
        # statement (COLOR_CONTRACT.md section 1).
        assert gen["color_zero_width_terms"] == 0

    def test_general_matrix_with_a_true_zero_width_gap(self):
        """query: type2 x1 only. library: type1 x1 only. r[2,1] = 0 (an
        identity matrix off that one cell), so the pair is SKIPPED, not
        visited-and-zero. Hand count: 1 zero-width term, 0 real terms."""
        n_features = 3
        r = np.eye(n_features)
        p = np.eye(n_features)
        lv = layout.color_view_from_counts(np.array([[4, 1, 0]], np.uint8))
        qv = layout.color_view_from_types(
            np.array([[0, 0, 0, 0, 2]], np.int64), np.array([4]), n_features)
        qi, li = np.array([0]), np.array([0])
        c = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=False)
        assert c["color_final_rescore_terms"] == 0
        assert c["color_zero_width_terms"] == 1

    def test_sparse_disjoint_types_no_shared_type_at_all(self):
        """query type1 (count 2) only, library type2 (count 2) only, on a
        plain identity matrix: same-type fast path visits NOTHING (no shared
        type -- 0 real terms). The general loop visits (1,2) (r[1,2]=0, so
        skipped) -- count(1)*count(2) = 2*2 = 4 terms accounted as
        ZERO-WIDTH, none as real work; (2,1) contributes 0*0=0 the same way.
        Both report zero REAL work either way."""
        n_features = 3
        r = np.eye(n_features)
        p = np.eye(n_features)
        lv = layout.color_view_from_counts(np.array([[3, 0, 2]], np.uint8))
        qv = layout.color_view_from_types(
            np.array([[0, 0, 1, 1]], np.int64), np.array([2]), n_features)
        qi, li = np.array([0]), np.array([0])
        fast = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=True)
        gen = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=False)
        assert fast["color_final_rescore_terms"] == 0
        assert gen["color_final_rescore_terms"] == 0
        assert gen["color_zero_width_terms"] == 4   # (1,2): count(1)*count(2)

    def test_empty_color_both_sides_is_all_zero(self):
        n_features = 3
        r, p = _matrix(n_features)
        lv = layout.color_view_from_counts(np.array([[5, 0, 0]], np.uint8))
        qv = layout.color_view_from_types(
            np.zeros((1, 5), np.int64), np.array([5]), n_features)
        qi, li = np.array([0]), np.array([0])
        c = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=True)
        for key in ("query_prep_terms", "query_self_color_terms",
                   "library_self_color_terms", "color_final_rescore_terms",
                   "color_zero_width_terms", "color_bytes_read"):
            assert c[key] == 0, key


class TestOneTimeVersusPerPairWork:
    """query_prep / self-color terms are paid ONCE per row that appears in
    the pair list, not once per pair -- the census item 4 separation between
    one-time preparation and per-pair work."""

    def test_self_color_and_prep_do_not_scale_with_pair_count(self):
        n_features = 3
        r, p = _matrix(n_features)
        lv = layout.color_view_from_counts(np.array([[2, 2, 1]], np.uint8))
        qv = layout.color_view_from_types(
            np.array([[0, 0, 1, 1, 2]], np.int64), np.array([2]), n_features)
        one = color_term_census(qv, np.array([0]), lv, np.array([0]),
                                r, p, same_type_fast_path=True)
        many = color_term_census(qv, np.array([0, 0, 0]), lv,
                                 np.array([0, 0, 0]), r, p,
                                 same_type_fast_path=True)
        assert one["query_prep_terms"] == many["query_prep_terms"]
        assert one["query_self_color_terms"] == many["query_self_color_terms"]
        assert one["library_self_color_terms"] == many["library_self_color_terms"]
        assert one["color_bytes_read"] == many["color_bytes_read"]
        # But the per-pair final rescore DOES scale linearly with pair count.
        assert many["color_final_rescore_terms"] == 3 * one["color_final_rescore_terms"]

    def test_self_color_is_the_ordered_square_not_the_triangular_count(self):
        """3 features of one type: the ordered self sum is 3*3=9, not the
        6-pair triangular count (COLOR_CONTRACT.md section 3: two coincident
        features contribute FOUR times one feature's self term, i.e. the
        full n^2, not n(n-1)/2 or n(n+1)/2)."""
        n_features = 2
        r, p = _matrix(n_features)
        lv = layout.color_view_from_counts(np.array([[1, 3]], np.uint8))
        qv = layout.color_view_from_types(
            np.array([[0, 1, 1, 1]], np.int64), np.array([1]), n_features)
        c = color_term_census(qv, np.array([0]), lv, np.array([0]), r, p,
                              same_type_fast_path=True)
        assert c["library_self_color_terms"] == 9
        assert c["query_self_color_terms"] == 9

    def test_multi_type_self_color_is_masked_like_a_pair_term(self):
        """Reviewer correction: self-color used to be the unconditional
        `n_color ** 2` regardless of `same_type_fast_path` or the matrix's
        own zero-width structure -- right only for a SINGLE-type molecule,
        which is exactly the regime the previous test (necessarily) used and
        so could not catch it.

        counts {t1: 3, t2: 1} (n_color = 4, so unconditional n_color**2 = 16):

        * canonical matrix (r=ones, p=diag), same_type_fast_path=True: only
          the diagonal is visited -- 3*3 + 1*1 = 10, not 16.
        * r=eye (so only the diagonal is nonzero-width), general loop
          (same_type_fast_path=False): the loop VISITS every cell (r!=0 or
          not), so it still finds only the diagonal nonzero -- 10 real terms
          -- but now ALSO tallies the two off-diagonal cells it visited and
          skipped for zero width: 3*1 + 1*3 = 6. 10 + 6 = 16 = n_color**2,
          which is the sanity identity: every one of the n_color**2 ordered
          atom pairs is accounted for exactly once, as real work or as a
          zero-width skip, never both and never neither.
        """
        n_features = 3
        counts_row = np.array([[2, 3, 1]], np.uint8)   # n_real=2, t1=3, t2=1
        lv = layout.color_view_from_counts(counts_row)
        qv = layout.color_view_from_types(
            np.array([[0, 0, 1, 1, 1, 2]], np.int64), np.array([2]), n_features)
        qi, li = np.array([0]), np.array([0])

        r_canon, p_canon = _matrix(n_features)
        fast = color_term_census(qv, qi, lv, li, r_canon, p_canon,
                                 same_type_fast_path=True)
        assert fast["library_self_color_terms"] == 10
        assert fast["query_self_color_terms"] == 10
        assert fast["library_self_color_zero_width_terms"] == 0
        assert fast["query_self_color_zero_width_terms"] == 0

        r_eye, p_eye = np.eye(n_features), np.eye(n_features)
        gen = color_term_census(qv, qi, lv, li, r_eye, p_eye,
                                same_type_fast_path=False)
        assert gen["library_self_color_terms"] == 10
        assert gen["library_self_color_zero_width_terms"] == 6
        assert (gen["library_self_color_terms"]
               + gen["library_self_color_zero_width_terms"]) == 4 ** 2
        assert gen["query_self_color_terms"] == 10
        assert gen["query_self_color_zero_width_terms"] == 6


def _brute_force_term_counts(counts_a, counts_b, r, p, same_type_fast_path):
    """The reviewer's instrument, committed as a test oracle: enumerate
    every INDIVIDUAL feature-atom pair literally (not the outer-product
    shortcut `color_term_census` takes), apply the `r == 0` skip and the
    same-type restriction one atom pair at a time, and count.

    Deliberately independent of `color_term_census`'s implementation -- a
    vectorized outer product and this nested loop can agree by construction
    on a shared bug only if
    neither is derived from the other, and this one is not: it does not
    know what an outer product is. Used both for pair terms (``counts_a`` a
    query row, ``counts_b`` a library row) and for self-color (``counts_a
    is counts_b``, the SAME row played on both sides of its own ordered
    sum -- COLOR_CONTRACT.md section 3's full n**2, i==j included, falls out
    of this loop for free because it never special-cases "the same atom").

    Returns ``(real_terms, zero_width_terms)``.
    """
    r = np.asarray(r, dtype=np.float64)
    n_features = r.shape[0]
    real = 0
    zero_width = 0
    for ta in range(1, n_features):
        for _ in range(int(counts_a[ta])):
            for tb in range(1, n_features):
                for _ in range(int(counts_b[tb])):
                    if same_type_fast_path and ta != tb:
                        continue
                    if r[ta, tb] == 0.0:
                        if not same_type_fast_path:
                            zero_width += 1
                        continue
                    real += 1
    return real, zero_width


class TestTheBruteForceOracle:
    """The instrument that caught the self-color bug, committed so the
    census cannot drift from the specification again without a test
    noticing. Randomized cases, cross-checked against
    `color_term_census` for both pair terms and self-color, in both loop
    structures."""

    @pytest.mark.parametrize("same_type_fast_path", [True, False])
    def test_pair_terms_match_the_brute_force_enumeration(self, same_type_fast_path):
        rng = np.random.default_rng(0)
        n_features = 4
        for trial in range(25):
            r = rng.choice([0.0, 1.0], size=(n_features, n_features),
                           p=[0.4, 0.6])
            r[0, :] = r[:, 0] = 0.0
            p = rng.standard_normal((n_features, n_features))
            counts_q = rng.integers(0, 4, size=n_features)
            counts_q[0] = 0
            counts_l = rng.integers(0, 4, size=n_features)
            counts_l[0] = 0
            lv = layout.color_view_from_counts(counts_l[None, :].astype(np.uint8))
            # Build a query type plane whose counts match counts_q exactly,
            # in ascending type order (a valid, contiguous run).
            types = [0, 0]        # two real atoms, arbitrary
            for t in range(1, n_features):
                types += [t] * int(counts_q[t])
            qv = layout.color_view_from_types(
                np.array([types], np.int64), np.array([2]), n_features)

            got = color_term_census(qv, np.array([0]), lv, np.array([0]),
                                    r, p, same_type_fast_path=same_type_fast_path)
            want_real, want_zero = _brute_force_term_counts(
                counts_q, counts_l, r, p, same_type_fast_path)
            assert got["color_final_rescore_terms"] == want_real, trial
            assert got["color_zero_width_terms"] == want_zero, trial

    @pytest.mark.parametrize("same_type_fast_path", [True, False])
    def test_self_color_matches_the_brute_force_enumeration(self, same_type_fast_path):
        rng = np.random.default_rng(1)
        n_features = 4
        for trial in range(25):
            r = rng.choice([0.0, 1.0], size=(n_features, n_features),
                           p=[0.4, 0.6])
            r[0, :] = r[:, 0] = 0.0
            p = rng.standard_normal((n_features, n_features))
            counts = rng.integers(0, 4, size=n_features)
            counts[0] = 0
            lv = layout.color_view_from_counts(counts[None, :].astype(np.uint8))
            types = [0, 0]
            for t in range(1, n_features):
                types += [t] * int(counts[t])
            qv = layout.color_view_from_types(
                np.array([types], np.int64), np.array([2]), n_features)

            got = color_term_census(qv, np.array([0]), lv, np.array([0]),
                                    r, p, same_type_fast_path=same_type_fast_path)
            want_real, want_zero = _brute_force_term_counts(
                counts, counts, r, p, same_type_fast_path)
            assert got["library_self_color_terms"] == want_real, trial
            assert got["library_self_color_zero_width_terms"] == want_zero, trial
            assert got["query_self_color_terms"] == want_real, trial
            assert got["query_self_color_zero_width_terms"] == want_zero, trial


class TestTheFastPathPredicate:
    """Scope item 3: the fast-path predicate must be explicit, stamped, and
    tested against a near-canonical matrix differing in one cell."""

    def test_same_type_fast_path_undercounts_a_near_canonical_matrix(self):
        """A matrix that is canonical except ONE off-diagonal height is
        nonzero: `color_loop_is_same_type_safe` must say False, and a
        same-type-only term count must differ from the general one on
        exactly the perturbed cell -- otherwise a same-type fast path could
        silently drop a real, nonzero contribution."""
        n_features = 3
        r, p = _matrix(n_features, perturb_off_diagonal=(1, 2))
        assert layout.color_loop_is_same_type_safe(r, p) is False
        # The un-perturbed matrix (from the SAME constructor) IS canonical:
        r0, p0 = _matrix(n_features)
        assert layout.color_loop_is_same_type_safe(r0, p0) is True

        lv = layout.color_view_from_counts(np.array([[2, 3, 1]], np.uint8))
        qv = layout.color_view_from_types(
            np.array([[0, 0, 1, 1, 1, 2]], np.int64), np.array([2]), n_features)
        qi, li = np.array([0]), np.array([0])
        gen = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=False)
        fast = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=True)
        # r is unchanged by the perturbation (only p moved), so term COUNTS
        # match the canonical case above exactly: general visits both
        # off-diagonal cells, (1,2)=3*1=3 and (2,1)=1*3=3 terms; same-type
        # fast path visits neither.
        assert gen["color_plane_terms"][1, 2] == 3
        assert gen["color_plane_terms"][2, 1] == 3
        assert fast["color_plane_terms"][1, 2] == 0
        assert fast["color_plane_terms"][2, 1] == 0
        assert gen["color_final_rescore_terms"] == fast["color_final_rescore_terms"] + 6
        # And this is the case that actually MATTERS: cell (1,2) carries a
        # real, nonzero height (p[1,2]=0.5) once perturbed, so the term the
        # fast path skips is not merely uncounted work -- it is a real
        # contribution to V_color that a same-type loop would silently drop.
        assert p[1, 2] != 0.0

    def test_predicate_reuses_numerics_check_interaction_matrix(self):
        """Not forked: this is the same object `backend.cuda`'s color_loop
        resolution already reads (backend/cuda.py:428)."""
        from roshambo3.numerics import check_interaction_matrix
        n_features = 3
        r, p = _matrix(n_features, perturb_off_diagonal=(1, 2))
        assert (layout.color_loop_is_same_type_safe(r, p)
               == check_interaction_matrix(r, p).canonical)


class TestPitfall18PaddingIsNotTerms:
    """Padding visits are not color work. A batch whose rows carry different
    n_color must not have its real term count inflated toward a padded
    common width."""

    def test_unequal_n_color_across_a_batch_reports_only_real_terms(self):
        n_features = 2
        r, p = _matrix(n_features)
        # Two library rows with DIFFERENT n_color (2 and 5) paired against
        # one query row with n_color=1.
        lv = layout.color_view_from_counts(np.array([[3, 2], [3, 5]], np.uint8))
        qv = layout.color_view_from_types(
            np.array([[0, 0, 0, 1]], np.int64), np.array([3]), n_features)
        qi = np.array([0, 0])
        li = np.array([0, 1])
        c = color_term_census(qv, qi, lv, li, r, p, same_type_fast_path=True)
        # Real terms: 1*2 + 1*5 = 7. A padded loop visiting max_n_color=5 for
        # BOTH pairs would report 1*5 + 1*5 = 10 -- strictly more, and this
        # module must not report that inflated number.
        assert c["color_final_rescore_terms"] == 7
        padded_if_batched_to_max = 1 * 5 + 1 * 5
        assert c["color_final_rescore_terms"] < padded_if_batched_to_max


class TestPitfall8TheLHistogramDiffers:
    def test_merged_color_record_has_a_different_l_histogram_than_shape_only(self):
        n_atoms = np.array([10, 20, 30])
        n_color = np.array([0, 5, 2])
        shape_only = shape_l_histogram(n_atoms)
        merged = shape_l_histogram(n_atoms, n_color)
        # Row totals: shape-only [10, 20, 30]; merged [10, 25, 32]. Both the
        # MAX and the MEAN move once color is folded in -- the point
        # is not that the max necessarily comes from a different row, only
        # that treating width as constant across the two is wrong.
        assert shape_only["max"] == 30
        assert merged["max"] == 32
        assert shape_only["mean"] != merged["mean"]
        assert shape_only != merged
        assert merged["has_color"] and not shape_only["has_color"]

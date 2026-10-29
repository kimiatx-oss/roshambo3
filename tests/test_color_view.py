"""The one read-only CPU color view, and its round-trip fixture.

The deciding measurement: every real and color coordinate,
type segment, count and matrix cell C++'s decoders see must match what
Python's own decoders (`roshambo3.layout.color_view_from_types` /
`color_view_from_counts`) compute on the SAME inputs. If it does not, two
future kernels (one scalar, one SIMD) could each pass their own tests while
disagreeing with each other on where a feature's atoms are -- silently,
because nothing else in the pipeline would notice.

Four regimes:
canonical (every type present, one run each), general (types out of the
canonical diagonal order or with gaps), sparse (types present on one side
only), and empty (no color features at all). Each is checked as a
hand-computed case, not just "the two implementations agree with each
other" -- agreement between two backends that share a bug proves nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import layout

cpu = pytest.importorskip("roshambo3.backend.cpu")
pytestmark = pytest.mark.skipif(not cpu.available(),
                                reason="the CPU kernel is not built")


def _check_round_trip(qc, qt, qn, lc, lcnt, n_features):
    """Compare the C++ view (via `cpu.build_color_view`) against Python's
    `roshambo3.layout` decoders, on every field, including a direct slice
    of the raw coordinate arrays."""
    out = cpu.build_color_view(query_coords=qc, query_types=qt,
                               query_n_atoms=qn, library_coords=lc,
                               library_counts=lcnt, n_features=n_features)
    qv = layout.color_view_from_types(qt, qn, n_features)
    lv = layout.color_view_from_counts(lcnt)

    np.testing.assert_array_equal(out["query"]["n_real"], qv.n_real)
    np.testing.assert_array_equal(out["query"]["n_color"], qv.n_color)
    np.testing.assert_array_equal(out["query"]["seg_offset"], qv.offsets)
    np.testing.assert_array_equal(out["query"]["seg_count"][:, 1:],
                                  qv.counts[:, 1:])
    np.testing.assert_array_equal(out["library"]["n_real"], lv.n_real)
    np.testing.assert_array_equal(out["library"]["n_color"], lv.n_color)
    np.testing.assert_array_equal(out["library"]["seg_offset"], lv.offsets)
    np.testing.assert_array_equal(out["library"]["seg_count"][:, 1:],
                                  lv.counts[:, 1:])

    for i in range(qc.shape[0]):
        nr, nc = int(qv.n_real[i]), int(qv.n_color[i])
        np.testing.assert_array_equal(
            out["query"]["color_xyz"][i, :nc], qc[i, nr:nr + nc, :3])
    for i in range(lc.shape[0]):
        nr, nc = int(lv.n_real[i]), int(lv.n_color[i])
        np.testing.assert_array_equal(
            out["library"]["color_xyz"][i, :nc], lc[i, nr:nr + nc, :3])
    return out, qv, lv


class TestTheRoundTripFixture:
    """Canonical, general, sparse and empty -- the ticket's own four cases."""

    def test_canonical_every_type_present_once_each_side(self):
        """One feature of each of 3 types, in enumeration order, both sides."""
        n_features = 4
        rng = np.random.default_rng(1)
        qc = np.zeros((1, 9, 3), np.float32)
        qc[0, 3:9] = rng.standard_normal((6, 3))
        qt = np.array([[0, 0, 0, 1, 1, 2, 2, 3, 3]], np.int32)
        qn = np.array([3], np.int32)

        lc = np.zeros((1, 9, 3), np.float32)
        lc[0, 2:9] = rng.standard_normal((7, 3))
        lcnt = np.array([[2, 2, 2, 3]], np.uint8)   # n_real=2, 2+2+3=7 features

        _check_round_trip(qc, qt, qn, lc, lcnt, n_features)

    def test_general_types_with_gaps_and_unequal_multiplicity(self):
        """Type 2 absent on the query; unequal per-type counts both sides."""
        n_features = 4
        rng = np.random.default_rng(2)
        qc = np.zeros((2, 10, 3), np.float32)
        qc[0, 4:10] = rng.standard_normal((6, 3))
        qc[1, 2:10] = rng.standard_normal((8, 3))
        qt = np.array([
            [0, 0, 0, 0, 1, 1, 1, 3, 3, 3],       # type 2 absent
            [0, 0, 1, 3, 3, 3, 3, 3, 3, 3],       # n_real=2, type1 x1, type3 x7
        ], np.int32)
        qn = np.array([4, 2], np.int32)

        lc = np.zeros((3, 10, 3), np.float32)
        lc[0, 5:10] = rng.standard_normal((5, 3))
        lc[1, 3:10] = rng.standard_normal((7, 3))
        lc[2, 6:10] = rng.standard_normal((4, 3))
        lcnt = np.array([
            [5, 3, 0, 2],
            [3, 0, 4, 3],
            [6, 4, 0, 0],
        ], np.uint8)
        _check_round_trip(qc, qt, qn, lc, lcnt, n_features)

    def test_sparse_disjoint_types_between_sides(self):
        """Query carries only type 1; library only type 2 -- no shared type."""
        n_features = 3
        rng = np.random.default_rng(3)
        qc = np.zeros((1, 6, 3), np.float32)
        qc[0, 4:6] = rng.standard_normal((2, 3))
        qt = np.array([[0, 0, 0, 0, 1, 1]], np.int32)
        qn = np.array([4], np.int32)

        lc = np.zeros((1, 6, 3), np.float32)
        lc[0, 5:6] = rng.standard_normal((1, 3))
        lcnt = np.array([[5, 0, 1]], np.uint8)
        _check_round_trip(qc, qt, qn, lc, lcnt, n_features)

    def test_empty_color_both_sides(self):
        """A zero-feature molecule: no color slots at all, either side."""
        n_features = 3
        qc = np.zeros((1, 5, 3), np.float32)
        qt = np.zeros((1, 5), np.int32)
        qn = np.array([5], np.int32)
        lc = np.zeros((1, 5, 3), np.float32)
        lcnt = np.array([[5, 0, 0]], np.uint8)
        out, qv, lv = _check_round_trip(qc, qt, qn, lc, lcnt, n_features)
        assert int(qv.n_color[0]) == 0
        assert int(lv.n_color[0]) == 0


class TestTheAsymmetry:
    """Query types are explicit and decoded; library types are counts,
    already decoded. The two are DIFFERENT functions, and this class pins
    why: one of them has a contiguity check to run and one structurally
    cannot need one."""

    def test_query_decoder_refuses_a_non_contiguous_run(self):
        n_features = 4
        qc = np.zeros((1, 8, 3), np.float32)
        qn = np.array([3], np.int32)
        # type 2 appears, then type 1 -- not one ascending run per type.
        qt = np.array([[0, 0, 0, 2, 2, 1, 0, 0]], np.int32)
        lc = np.zeros((1, 4, 3), np.float32)
        lcnt = np.array([[3, 0, 0, 0]], np.uint8)

        with pytest.raises(ValueError, match="contiguous ascending run"):
            layout.color_view_from_types(qt, qn, n_features)
        with pytest.raises(ValueError, match="contiguous ascending run"):
            cpu.build_color_view(query_coords=qc, query_types=qt,
                                 query_n_atoms=qn, library_coords=lc,
                                 library_counts=lcnt, n_features=n_features)

    def test_library_side_has_no_contiguity_to_check_only_bounds(self):
        """A count vector cannot even EXPRESS interleaving -- it is
        already "how many of each type", not a per-slot type array -- so
        there is no analogous check to run on it. What it DOES bound-check
        is the column count against MAX_FEATURES, which is a property no
        per-atom type array could violate either."""
        too_wide = np.zeros((1, layout.MAX_FEATURES + 1), np.uint8)
        with pytest.raises(ValueError, match="MAX_FEATURES"):
            layout.color_view_from_counts(too_wide)

    def test_query_decoder_refuses_a_padding_gap_inside_the_color_run(self):
        """Reviewer-found: `types = [pad, pad, 1, pad, 1, pad]`, n_real=2 --
        real features sit at slots 2 and 4 with a PADDING slot at 3 between
        them. `nz = row[row != 0]` used to filter the zero out and see
        `[1, 1]`, trivially non-decreasing, and accept it -- deriving
        offsets that assume contiguity from n_real (reading slot 3, the
        padding, as the second feature; the real feature at slot 4 was never
        read). Both decoders must refuse this, not just one -- a fixture
        where BOTH implementations make the same wrong assumption would pass
        a round-trip check while returning coordinate garbage."""
        n_features = 3
        qn = np.array([2], np.int32)
        qt = np.array([[0, 0, 1, 0, 1, 0]], np.int32)
        qc = np.zeros((1, 6, 3), np.float32)
        qc[0, 2] = [9.0, 9.0, 9.0]     # the real feature the gap would hide
        qc[0, 4] = [7.0, 7.0, 7.0]     # the one a contiguity bug would miss
        lc = np.zeros((1, 4, 3), np.float32)
        lcnt = np.array([[4, 0, 0]], np.uint8)

        with pytest.raises(ValueError, match="padding gap"):
            layout.color_view_from_types(qt, qn, n_features)
        with pytest.raises(ValueError, match="padding gap"):
            cpu.build_color_view(query_coords=qc, query_types=qt,
                                 query_n_atoms=qn, library_coords=lc,
                                 library_counts=lcnt, n_features=n_features)

    def test_a_gap_at_the_very_first_color_slot_is_also_refused(self):
        """The degenerate case: n_real correct, but the FIRST color slot is
        itself padding and a real feature follows it. No "last contiguous
        run" exists at all here, so this checks the boundary rather than
        the general case above."""
        n_features = 2
        qn = np.array([2], np.int32)
        qt = np.array([[0, 0, 0, 1, 0]], np.int32)   # slot 2 (=n_real) is padding
        with pytest.raises(ValueError, match="padding gap"):
            layout.color_view_from_types(qt, qn, n_features)


class TestTheStamp:
    """Acceptance: 'the loaded extension stamp exposes the color
    ABI/feature-count limit and selected ISA.'"""

    def test_color_max_features_matches_python_and_the_wire_limit(self):
        ext = cpu.extension()
        assert int(ext.__color_max_features__) == layout.MAX_FEATURES

    def test_color_abi_version_is_stamped(self):
        ext = cpu.extension()
        assert isinstance(int(ext.__color_abi_version__), int)
        assert int(ext.__color_abi_version__) >= 1

    def test_build_stamp_carries_both(self):
        s = cpu.build_stamp()
        assert s["color_max_features"] == layout.MAX_FEATURES
        assert s["color_abi_version"] >= 1
        assert "selected_target" in s

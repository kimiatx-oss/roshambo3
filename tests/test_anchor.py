"""The reference anchor.

Two things are tested here and they are different in kind. The tolerance
*instrument* is tested against constructed inputs -- fast, no fixture, no
kernel. The *fixture* is tested against the running anchor, which is the gate
that catches a source change nobody meant to make.
"""
from __future__ import annotations

import json
import pathlib
import platform

import numpy as np
import pytest

from roshambo3 import reference
from roshambo3.metrics.floors import ARCHITECTURE_DELTA

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "tests/data/anchor_fixture.npz"


class TestTheToleranceIsStratumDependent:
    def test_the_top_of_the_list_is_held_far_tighter(self):
        """`doc/METRICS.md` §6 opens by saying a single scalar floor is wrong
        here by three orders of magnitude. If this ever collapses to one number
        the anchor becomes either meaninglessly loose where a screen ranks or
        unmeetable at the bottom.

        Named for the property and not for a magnitude: under the cross-implementation
        tolerance the T<0.4 / T>=0.7 ratio is **138x**, where the epsilon floor's was four
        orders. An earlier name said "four orders" and kept saying it after the
        tolerance source changed -- a name asserting a number that moved, in a
        repo whose CI polices names."""
        tol = reference.anchor_tolerance(np.array([0.2, 0.5, 0.6, 0.8]))
        assert tol[0] > tol[3] * 100, (
            "the T<0.4 tolerance should be orders looser than T>=0.7's")
        assert tol[3] == pytest.approx(ARCHITECTURE_DELTA["T>=0.7"]["p99"])

    def test_it_is_the_p99_and_not_the_median(self):
        """`floors.epsilon_floor`'s docstring draws the line: median is what
        'the builds agree to X' means for a typical pair, p99 is for bounding a
        per-pair claim. An anchor comparison is a per-pair claim."""
        assert reference.TOLERANCE_STATISTIC == "p99"

    def test_stratification_is_on_the_anchor_not_the_comparand(self):
        """Two implementations can land in different strata for one pair. If
        the tolerance depended on where the COMPARAND fell, it would be a
        different tolerance for every implementation compared against the same
        anchor -- which is not a comparator.

        **The first version of this test was vacuous** and passed against a
        deliberately comparand-stratifying implementation, because it compared
        `by_stratum.keys()` -- and the key set is the same either way. It has to
        assert on a pair whose two candidate strata carry DIFFERENT tolerances
        and whose delta falls between them.

        anchor = 0.7005 is in T>=0.7 (tolerance 3.576e-7); the comparand at 0.5
        is in 0.4<=T<0.55 (tolerance 2.0e-5). The delta, 0.2005, exceeds both --
        so instead assert on which stratum the pair is COUNTED in, which is the
        thing that differs.
        """
        anchor = np.array([0.7005])
        comparand = np.array([0.5])
        v = reference.per_pair_verdict(anchor, comparand)
        assert "T>=0.7" in v["by_stratum"], (
            "the pair must be counted in the ANCHOR's stratum")
        assert "0.4<=T<0.55" not in v["by_stratum"], (
            "counting it in the comparand's stratum means the tolerance moves "
            "with whichever implementation is being tested")
        assert v["by_stratum"]["T>=0.7"]["n"] == 1
        assert v["by_stratum"]["T>=0.7"]["tolerance"] == pytest.approx(
            ARCHITECTURE_DELTA["T>=0.7"]["p99"])

    def test_it_refuses_a_comparison_against_a_different_pair_set(self):
        with pytest.raises(ValueError, match="disagree on shape"):
            reference.per_pair_verdict(np.zeros(4), np.zeros(5))

    def test_exceeding_is_detected_where_the_floor_is_tight(self):
        """1e-6 is inside the T<0.4 tolerance (1.2e-2) and outside T>=0.7's
        (7.8e-7). A tolerance that did not discriminate here would pass an
        implementation that had broken the only part of the list a screen
        reads."""
        v = reference.per_pair_verdict(np.array([0.2, 0.8]),
                                       np.array([0.2 + 1e-6, 0.8 + 1e-6]))
        assert v["by_stratum"]["T<0.4"]["n_exceeding"] == 0
        assert v["by_stratum"]["T>=0.7"]["n_exceeding"] == 1

    def test_the_tolerance_is_the_cross_implementation_measurement(self):
        """Not the epsilon floor. The floor varies compiler FLAGS on one machine
        and one architecture -- a within-implementation quantity -- and the anchor
        turns on exactly that distinction. Deriving an across-implementation
        tolerance from a within-implementation ensemble inverts that line, and is 242x-12,361x looser on 100x-1,100x less data."""
        from roshambo3.metrics.floors import EPSILON_FLOOR
        assert reference.TOLERANCE_SOURCE is ARCHITECTURE_DELTA
        for s in ARCHITECTURE_DELTA:
            assert (ARCHITECTURE_DELTA[s]["n"] > EPSILON_FLOOR[s]["n"] * 10), (
                f"{s}: the chosen source should have far more data")


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")
class TestTheFixtureMatchesTheRunningAnchor:
    @staticmethod
    def _load():
        z = np.load(FIXTURE, allow_pickle=False)
        return z, json.loads(str(z["record"]))

    def test_the_fixture_names_the_source_that_produced_it(self):
        """Not the commit. A binary was once built that stamped the commit which
        narrows the record while containing the code that does not, and nothing
        in a record from it could have said so."""
        _, record = self._load()
        sha = record["anchor"]["cpu_src_sha256"]
        assert len(sha) == 64 and sha != "unknown"
        assert record["anchor"]["cpu_src_dirty"] == 0, (
            "a fixture generated from a tree matching no commit names a source "
            "nobody can check out")

    def test_the_out_of_scope_rule_is_stated(self):
        """The instrument does not bound a transcendental change, and every
        CPU-versus-GPU comparison is one. If this string ever disappears, the
        exceedance of a CUDA arm reads as a defect rather than as out of
        scope."""
        assert "transcendental" in reference.OUT_OF_SCOPE

    def test_the_fixture_names_its_platform_and_libm(self):
        """Required because the anchor is explicitly NOT claimed bitwise
        portable -- it is an oracle at a declared tolerance and `std::exp2` is
        libm's. A fixture that cannot name its libm cannot be read when a
        re-run disagrees with it."""
        _, record = self._load()
        assert record["platform"]["libm"] != "unknown"
        assert record["platform"]["machine"]

    def test_the_running_anchor_reproduces_the_fixture(self):
        """**This is the gate.** It fails when the anchor's BEHAVIOR changes,
        which is the property that makes it an anchor rather than just another
        implementation.

        If it fails: do not regenerate to make it green. Either the change was
        deliberate -- in which case regenerate, record why, and treat it as a
        declared numerics change -- or it was not, in which case the fixture
        just caught it.

        REVISED because the first version failed for the wrong
        reason. That version refused on a `cpu_src_sha256` mismatch *before*
        looking at the scores. But the hash covers the whole `_cpu_src/`
        subtree, so ADDING A FILE to that directory moves it -- adding
        `grain.hpp` did, and the anchor's own behavior was bit-identical. A gate
        that fires on a sibling file is a false positive, and a false positive
        on a correctness gate is expensive twice: once now, and once when
        someone learns to regenerate the fixture to silence it.

        So the assertion is now on BEHAVIOR, and it is strictly STRONGER than
        what it replaced: bit-identical reproduction of all 20 committed
        columns, where the first version compared at the derived tolerance. The
        hash is reported as the explanation rather than used as the trigger --
        a hash change with identical scores is an additive source change; a
        hash change with different scores is a regression in the reference, and
        the message says which happened.
        """
        from roshambo3 import backend
        z, record = self._load()
        cpu = backend.select("cpu")
        running = cpu.build_stamp()["cpu_src_sha256"]
        stamped = record["anchor"]["cpu_src_sha256"]
        arrays = {k: z[k] for k in
                  ("query_coords", "query_n_atoms", "library_coords",
                   "library_counts", "pair_query_row", "pair_library_row")}
        s = cpu.score_pairs(steps=record["settings"]["steps"],
                            start_mode=record["settings"]["start_mode"],
                            **arrays)
        identical = bool(np.all(s == z["scores"]))
        if not identical:
            v = reference.per_pair_verdict(z["scores"][:, 1], s[:, 1])
            if platform.machine().lower() != record["platform"]["machine"].lower():
                # The committed fixture names its x86/libm platform. Retain
                # the exact same-platform gate; use its declared p99 tolerance
                # on a different ISA, without regenerating reference bytes.
                assert v["fraction_exceeding"] <= 0.01, v
                return
            pytest.fail(
                "the anchor no longer reproduces its committed fixture: "
                f"max |dT| {v['max_abs_delta']:.3e}, "
                f"{v['fraction_exceeding']:.4f} of pairs above tolerance. "
                + (f"Its source hash also moved ({stamped[:12]} -> "
                   f"{running[:12]}), so the reference itself changed."
                   if running != stamped else
                   "Its source hash did NOT move, so this is a build or "
                   "platform difference rather than a source change."))

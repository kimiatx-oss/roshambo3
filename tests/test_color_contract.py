"""The CPU color contract (doc/COLOR_CONTRACT.md), executable.

Three kinds of test, different in kind:

* **Frozen-fixture tests** gate the reference itself: the committed float64
  values, gradients and tolerances must reproduce from the committed inputs.
  This is what makes the fixture an anchor rather than a snapshot.
* **Convention tests** make each deliberately wrong convention fail a test
  that names it: swapped type numbering, wrong pose direction, omitted self
  cross-terms, the legacy combination formula. A contract nothing can violate
  has not been pinned.
* **Cross-checks** compare the CUDA analytic path to the fixture at the
  derived fixed-pose tolerance. The CUDA path is a cross-check of the port,
  never the oracle -- the fixture was generated and committed
  before any of these ran, and nothing here feeds back into it.
"""
from __future__ import annotations

import json
import math
import pathlib

from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np
import pytest

from roshambo3 import backend, scores as S
from roshambo3 import reference as R
from roshambo3.reference import color as C

FIXTURE = (pathlib.Path(__file__).resolve().parents[1]
           / "tests/data/color_contract_fixture.npz")

needs_gpu = pytest.mark.skipif(not backend.available(),
                               reason="no CUDA extension in this build")

#: Deliberately wrong conventions must miss the fixture by at least this
#: multiple of the derived tolerance, so "the gate discriminates" is an
#: assertion rather than prose.
WRONG_CONVENTION_MARGIN = 10.0


def _load():
    z = np.load(FIXTURE, allow_pickle=False)
    return z, json.loads(str(z["record"]))


def _case(z, name: str) -> dict:
    out = {}
    prefix = name + "__"
    for key in z.files:
        if key.startswith(prefix):
            out[key[len(prefix):]] = z[key]
    return out


def _columns_dict(arr, record) -> dict:
    return dict(zip(record["scalar_columns"], np.asarray(arr, np.float64)))


def _recompute_columns(c, mixing: float) -> dict:
    """The nine columns from the committed pre-posed inputs -- the exact
    evaluation the fixture froze."""
    return C.score_columns(
        c["query_shape"], c["library_shape_posed"],
        c["query_color"], c["query_types"],
        c["library_color_posed"], c["library_types"],
        c["interaction_r"], c["interaction_p"], mixing)


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")
class TestTheFixtureIsStampedAndReproduces:
    def test_the_fixture_names_what_produced_it(self):
        """Source hashes, git sha, platform, libm, library versions, the
        feature enumeration by name, and the SDF hash. A fixture that cannot
        name what produced it cannot be read when a re-run disagrees."""
        _, record = self._rec()
        gen = record["generator"]
        assert len(gen["reference_color_sha256"]) == 64
        assert len(gen["reference_overlap_sha256"]) == 64
        assert len(gen["git_sha"]) == 40
        assert record["platform"]["libm"] != "unknown"
        assert record["platform"]["machine"]
        enum = record["inputs"]["feature_enumeration"]
        assert enum and 0 not in enum.values(), (
            "type 0 is a normal atom; a feature enumerated 0 contradicts the "
            "contract's type numbering")
        assert len(record["inputs"]["sdf_sha256"]) == 64

    @staticmethod
    def _rec():
        return _load()

    def test_the_running_reference_reproduces_every_committed_column(self):
        """**This is the gate.** It fails when the reference's behavior
        changes, which is what makes the fixture an anchor rather than a
        snapshot. If it fails, do not regenerate to make it green: either the
        change was a deliberate, recorded numerics change (then regenerate and
        say so) or the fixture just caught an accident.

        Behavior-gated, not hash-gated: the source
        hash is the *explanation* in the failure message, never the trigger,
        so an additive or cosmetic source change cannot false-positive."""
        z, record = _load()
        rtol = record["tolerance"]["float64_reproduction_rtol"]
        color_sha = C.__dict__ and __import__("hashlib").sha256(
            pathlib.Path(C.__file__).read_bytes()).hexdigest()
        stamped = record["generator"]["reference_color_sha256"]
        for name, meta in record["cases"].items():
            c = _case(z, name)
            got = _recompute_columns(c, meta["mixing"])
            for i, col in enumerate(record["scalar_columns"]):
                want = c["columns"][i]
                ok = (got[col] == want or
                      abs(got[col] - want) <= rtol * max(abs(want), 1e-300))
                if not ok:
                    explain = (
                        f"reference/color.py's hash moved ({stamped[:12]} -> "
                        f"{color_sha[:12]}), so the reference itself changed."
                        if color_sha != stamped else
                        "the source hash did NOT move: a platform or library "
                        "difference larger than the derived float64 bound.")
                    pytest.fail(
                        f"{name}.{col}: committed {want!r}, recomputed "
                        f"{got[col]!r} (|d| {abs(got[col] - want):.3e} > "
                        f"rtol {rtol:g}). {explain}")

    def test_the_committed_posed_coordinates_reproduce_bitwise(self):
        """The pose application is elementwise IEEE float64 with one rounding
        to float32, so it is deterministic across platforms; the posed arrays
        are therefore held bit-exactly, unlike the libm-dependent sums."""
        z, record = _load()
        for name in record["cases"]:
            c = _case(z, name)
            for src, posed in (("library_shape", "library_shape_posed"),
                               ("library_color", "library_color_posed")):
                got = C.apply_pose_elementwise(
                    c[src], c["pose_quat"],
                    c["pose_translation"]).astype(np.float32)
                assert np.array_equal(got, c[posed]), (
                    f"{name}: {posed} does not reproduce from {src} + pose")

    def test_the_committed_tolerances_reproduce_from_the_inputs(self):
        """The tolerance is derived from the case's own inputs by committed
        code. If the model changes, this fails loudly instead of the gate
        silently loosening or tightening."""
        z, record = _load()
        for name, meta in record["cases"].items():
            c = _case(z, name)
            for flavor, flag in (("tolerance", False),
                                 ("tolerance_input_rounding", True)):
                tol = C.fixed_pose_tolerance(
                    c["query_shape"], c["library_shape_posed"],
                    c["query_color"], c["query_types"],
                    c["library_color_posed"], c["library_types"],
                    c["interaction_r"], c["interaction_p"], meta["mixing"],
                    include_input_rounding=flag)
                got = np.array([tol[k] for k in record["scalar_columns"]])
                assert np.allclose(got, c[flavor], rtol=1e-9, atol=0.0), (
                    f"{name}.{flavor}: the derivation no longer reproduces "
                    f"the committed tolerance -- a model change must be "
                    f"deliberate and the fixture regenerated with it")

    def test_the_committed_gradients_reproduce(self):
        z, record = _load()
        h = record["gradient_h"][0]
        for name, meta in record["cases"].items():
            if not meta["gradients"]:
                continue
            c = _case(z, name)
            got = C.color_overlap_gradient(
                c["query_color"], c["query_types"],
                c["library_color"], c["library_types"],
                c["interaction_r"], c["interaction_p"],
                c["pose_quat"], c["pose_translation"], h=h)
            assert np.allclose(got, c["color_gradient_h0"],
                               rtol=1e-9, atol=1e-12), name

    def test_the_derivation_string_is_the_module_constant(self):
        """The record carries the derivation so the fixture is readable
        standalone; it must be the module's, not a paraphrase that can
        drift."""
        _, record = _load()
        assert (record["tolerance"]["fixed_pose"]
                == C.FIXED_POSE_TOLERANCE_DERIVATION)


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")
class TestThePhysicalFormula:
    def test_one_feature_pair_matches_the_closed_form(self):
        """Independent of the reference module: the published expression,
        one term, evaluated inline."""
        z, record = _load()
        c = _case(z, "one_feature_each")
        d2 = float(((np.asarray(c["query_color"][0], np.float64)
                     - np.asarray(c["library_color_posed"][0], np.float64))
                    ** 2).sum())
        r = float(c["interaction_r"][1, 1])
        p = float(c["interaction_p"][1, 1])
        want = ((math.pi / 2.0) ** 1.5 * p * p * r ** -1.5
                * math.exp(-0.5 * r * d2))
        got = _columns_dict(c["columns"], record)["overlap_color"]
        assert got == pytest.approx(want, rel=1e-14)

    def test_a_zero_width_pair_is_skipped_not_evaluated(self):
        """r[1,2] = 0 with p[1,2] != 0. Skipping the pair and zeroing its
        height are the same value, so equality against a zero-height
        evaluation at a NONZERO width proves the cell contributed nothing --
        and the committed columns are finite, which an evaluated zero width
        (0^-1.5) could not be."""
        z, record = _load()
        c = _case(z, "zero_width_interaction")
        meta = record["cases"]["zero_width_interaction"]
        assert np.isfinite(c["columns"]).all()
        r2 = np.array(c["interaction_r"], np.float64)
        p2 = np.array(c["interaction_p"], np.float64)
        assert r2[1, 2] == 0.0 and p2[1, 2] != 0.0, "fixture premise"
        r2[1, 2] = r2[2, 1] = 1.0
        p2[1, 2] = p2[2, 1] = 0.0
        alt = C.score_columns(
            c["query_shape"], c["library_shape_posed"],
            c["query_color"], c["query_types"],
            c["library_color_posed"], c["library_types"],
            r2, p2, meta["mixing"])
        for i, col in enumerate(record["scalar_columns"]):
            assert alt[col] == pytest.approx(c["columns"][i], rel=1e-14,
                                             abs=0.0), col

    def test_the_self_overlap_is_ordered_four_x_for_coincident_features(self):
        """Two identical coincident features: the full ordered sum has four
        unit terms. A triangular or diagonal-only sum reports 2x and fails
        here before it fails the named negative test below."""
        z, record = _load()
        c = _case(z, "coincident_repeated_features")
        cols = _columns_dict(c["columns"], record)
        want = 4.0 * (math.pi / 2.0) ** 1.5
        assert cols["self_color_library"] == pytest.approx(want, rel=1e-14)
        assert cols["self_color_query"] == pytest.approx(
            (math.pi / 2.0) ** 1.5, rel=1e-14)

    def test_a_missing_query_type_reaches_only_the_library_self_column(self):
        """The library's type-2 features have no query partner under the
        canonical matrix: deleting them must not move the cross overlap, and
        must move the library self-color."""
        z, record = _load()
        c = _case(z, "missing_query_type")
        meta = record["cases"]["missing_query_type"]
        keep = c["library_types"] == 1
        alt = C.score_columns(
            c["query_shape"], c["library_shape_posed"],
            c["query_color"], c["query_types"],
            c["library_color_posed"][keep], c["library_types"][keep],
            c["interaction_r"], c["interaction_p"], meta["mixing"])
        cols = _columns_dict(c["columns"], record)
        assert alt["overlap_color"] == pytest.approx(cols["overlap_color"],
                                                     rel=1e-14)
        assert alt["self_color_library"] < cols["self_color_library"]

    def test_empty_color_is_finite_zero_and_preserves_shape(self):
        """The acceptance bar verbatim: finite zero color scores, shape
        untouched, no denominator NaN/Inf -- and column 0 is
        (1-mixing)*T_shape, which is the formula's answer for a featureless
        pair, not a defect."""
        z, record = _load()
        for name in ("empty_color_both_sides", "empty_color_one_side"):
            c = _case(z, name)
            cols = _columns_dict(c["columns"], record)
            assert np.isfinite(c["columns"]).all(), name
            assert cols["tanimoto_color"] == 0.0
            assert cols["overlap_color"] == 0.0
            assert cols["self_color_library"] == 0.0
            shape_only = C.shape_only_columns(c["query_shape"],
                                              c["library_shape_posed"])
            for col in ("tanimoto_shape", "overlap_volume",
                        "self_overlap_query", "self_overlap_library"):
                assert cols[col] == shape_only[col], (name, col)
            m = record["cases"][name]["mixing"]
            assert cols["tanimoto_combination"] == pytest.approx(
                (1.0 - m) * cols["tanimoto_shape"], rel=1e-15)

    def test_the_guard_reports_zero_not_one_for_two_empty_planes(self):
        assert C.guarded_tanimoto(0.0, 0.0, 0.0) == 0.0
        assert C.guarded_tanimoto(0.0, 1.5, 0.0) == 0.0


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")
class TestTheNineColumns:
    def test_column_zero_is_the_mixed_score_for_every_case(self):
        z, record = _load()
        for name, meta in record["cases"].items():
            cols = _columns_dict(_case(z, name)["columns"], record)
            m = meta["mixing"]
            assert cols["tanimoto_combination"] == pytest.approx(
                (1.0 - m) * cols["tanimoto_shape"]
                + m * cols["tanimoto_color"], rel=1e-14, abs=1e-300), name

    def test_the_mixing_endpoints(self):
        """mixing 0 reports the shape Tanimoto in column 0 exactly, mixing 1
        the color Tanimoto -- and mixing 0 is also why no test at mixing 0
        can discriminate a color implementation (module docstring)."""
        z, _ = _load()
        c = _case(z, "asymmetric_matrix_indexing")
        at0 = _recompute_columns(c, 0.0)
        at1 = _recompute_columns(c, 1.0)
        assert at0["tanimoto_combination"] == at0["tanimoto_shape"]
        assert at1["tanimoto_combination"] == at1["tanimoto_color"]

    def test_mixing_outside_the_unit_interval_is_refused(self):
        z, _ = _load()
        c = _case(z, "one_feature_each")
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError, match="mixing"):
                _recompute_columns(c, bad)

    def test_a_feature_numbered_zero_is_refused(self):
        """Type 0 is a normal atom; an implementation that numbers features
        from 0 has the off-by-one this refusal exists to catch."""
        z, _ = _load()
        c = _case(z, "one_feature_each")
        with pytest.raises(ValueError, match="type 0"):
            C.score_columns(
                c["query_shape"], c["library_shape_posed"],
                c["query_color"], np.zeros_like(c["query_types"]),
                c["library_color_posed"], c["library_types"],
                c["interaction_r"], c["interaction_p"], 0.5)

    def test_the_volume_columns_carry_the_physical_prefactors(self):
        """A single self-feature's color self-overlap is exactly
        (pi/2)^1.5; the shape self-overlap of one atom is exactly the shape
        prefactor. Both pin that the committed volumes are physical, not
        reduced."""
        one = np.zeros((1, 3))
        t1 = np.array([1])
        r = np.ones((2, 2))
        p = np.diag([0.0, 1.0])
        cols = C.score_columns(one, one, one, t1, one, t1, r, p, 0.5)
        assert cols["self_color_query"] == pytest.approx(
            (math.pi / 2.0) ** 1.5, rel=1e-14)
        assert cols["self_overlap_query"] == pytest.approx(
            float(R.PAIR_PREFACTOR), rel=1e-14)


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")
class TestTheDeliberatelyWrongConventions:
    """Each wrong convention the deciding measurement names, failing a test
    that names it. Every assertion demands a miss of at least
    WRONG_CONVENTION_MARGIN times the derived tolerance: the gate does not
    merely reject these, it rejects them with two orders of headroom."""

    def test_swapped_type_numbering_fails_the_fixture(self):
        """Renumbering features in a different order than the shipped
        enumeration (types 1 and 2 exchanged, matrices left in shipped
        order) is a real hazard: numeric type equality without an
        enumeration stamp is not semantic equality."""
        z, record = _load()
        c = _case(z, "asymmetric_matrix_indexing")
        meta = record["cases"]["asymmetric_matrix_indexing"]
        swap = {1: 2, 2: 1}
        qt = np.array([swap[t] for t in c["query_types"]], np.int64)
        lt = np.array([swap[t] for t in c["library_types"]], np.int64)
        wrong = C.score_columns(
            c["query_shape"], c["library_shape_posed"],
            c["query_color"], qt, c["library_color_posed"], lt,
            c["interaction_r"], c["interaction_p"], meta["mixing"])
        cols = _columns_dict(c["columns"], record)
        tol = _columns_dict(c["tolerance"], record)
        for col in ("tanimoto_color", "overlap_color", "tanimoto_combination"):
            assert abs(wrong[col] - cols[col]) > (
                WRONG_CONVENTION_MARGIN * tol[col]), (
                f"{col}: swapped type numbering lands inside the tolerance; "
                f"the fixture cannot discriminate it")

    def test_wrong_pose_direction_fails_the_fixture(self):
        """Two wrong flavors: the pose applied to the query instead of the
        library, and the inverse rotation applied to the library. The correct
        direction reproduces the committed columns from the ORIGINAL library
        coordinates; each wrong one misses by orders of magnitude."""
        z, record = _load()
        c = _case(z, "asymmetric_matrix_indexing")
        meta = record["cases"]["asymmetric_matrix_indexing"]
        q, t = c["pose_quat"], c["pose_translation"]
        cols = _columns_dict(c["columns"], record)
        tol = _columns_dict(c["tolerance"], record)

        right = C.score_columns(
            c["query_shape"], c["library_shape"],
            c["query_color"], c["query_types"],
            c["library_color"], c["library_types"],
            c["interaction_r"], c["interaction_p"], meta["mixing"], q, t)
        # The committed columns are at posed-then-rounded float32
        # coordinates; the float64 pose differs from them by one float32
        # rounding of each coordinate, far inside the derived tolerance.
        for col in ("tanimoto_shape", "tanimoto_color",
                    "tanimoto_combination"):
            assert abs(right[col] - cols[col]) <= tol[col], (
                f"{col}: the CORRECT pose direction does not reproduce the "
                f"fixture -- the fixture itself is broken")

        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        wrong_inverse = C.score_columns(
            c["query_shape"], c["library_shape"],
            c["query_color"], c["query_types"],
            c["library_color"], c["library_types"],
            c["interaction_r"], c["interaction_p"], meta["mixing"], q_inv, t)
        wrong_side = C.score_columns(
            C.apply_pose_elementwise(c["query_shape"], q, t),
            c["library_shape"],
            C.apply_pose_elementwise(c["query_color"], q, t),
            c["query_types"],
            c["library_color"], c["library_types"],
            c["interaction_r"], c["interaction_p"], meta["mixing"])
        for wrong, label in ((wrong_inverse, "inverse rotation"),
                             (wrong_side, "pose applied to the query")):
            misses = [col for col in ("tanimoto_shape", "tanimoto_color")
                      if abs(wrong[col] - cols[col])
                      > WRONG_CONVENTION_MARGIN * tol[col]]
            assert misses, (
                f"{label} lands inside the tolerance on every column; the "
                f"fixture cannot discriminate the pose direction")

    def test_omitted_self_cross_terms_fail_the_fixture(self):
        """A self-color that keeps only the i == j diagonal reports half the
        ordered sum on the coincident-features case (2 of 4 unit terms) and
        the wrong color Tanimoto with it."""
        z, record = _load()
        c = _case(z, "coincident_repeated_features")
        meta = record["cases"]["coincident_repeated_features"]
        cols = _columns_dict(c["columns"], record)
        tol = _columns_dict(c["tolerance"], record)

        def diagonal_only_self(coords, types):
            total = 0.0
            r = np.asarray(c["interaction_r"], np.float64)
            p = np.asarray(c["interaction_p"], np.float64)
            for i in range(coords.shape[0]):
                w = r[types[i], types[i]]
                if w == 0.0:
                    continue
                h = p[types[i], types[i]]
                total += h * h * w ** -1.5      # exp(0) at d = 0
            return (math.pi / 2.0) ** 1.5 * total

        wrong_self = diagonal_only_self(
            np.asarray(c["library_color_posed"], np.float64),
            c["library_types"])
        assert abs(wrong_self - cols["self_color_library"]) > (
            WRONG_CONVENTION_MARGIN * tol["self_color_library"])
        assert wrong_self == pytest.approx(
            cols["self_color_library"] / 2.0, rel=1e-12), (
            "the coincident case should make the omission exactly a factor "
            "of two, or the case no longer tests what it says")
        wrong_tc = C.guarded_tanimoto(cols["overlap_color"],
                                      cols["self_color_query"], wrong_self)
        assert abs(wrong_tc - cols["tanimoto_color"]) > (
            WRONG_CONVENTION_MARGIN * tol["tanimoto_color"])

    def test_the_legacy_combination_formula_fails_the_fixture(self):
        """Column 0 is (1-mixing)*T_shape + mixing*T_color. The legacy
        T_shape + T_color (out of 2) must fail against every case with a
        nonzero score -- including at mixing exactly 0.5, where it is wrong
        by a factor of two, not by a rounding."""
        z, record = _load()
        checked = 0
        for name, meta in record["cases"].items():
            c = _case(z, name)
            cols = _columns_dict(c["columns"], record)
            tol = _columns_dict(c["tolerance"], record)
            legacy = cols["tanimoto_shape"] + cols["tanimoto_color"]
            if cols["tanimoto_combination"] == 0.0:
                continue
            assert abs(legacy - cols["tanimoto_combination"]) > (
                WRONG_CONVENTION_MARGIN
                * max(tol["tanimoto_combination"], 1e-12)), (
                f"{name}: the legacy sum lands inside the tolerance")
            checked += 1
        assert checked >= 8, "the legacy formula went untested"


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")
class TestTheModesAreDistinguished:
    """The three color modes, discriminated at nonzero mixing -- the acceptance
    criterion verbatim. All committed values come from the reference's own
    global search, converged (16 vs 64 restarts) before freezing."""

    #: Both mode cases. The first has byte-identical shapes (its shape
    #: optimum is the identity at exactly 1.0 -- doc/COLOR_CONTRACT.md
    #: section 7 says what that does and does not exercise); the second has
    #: genuinely distinct shapes and is the one that carries mode 2's
    #: acceptance bar.
    MODE_CASES = ("mode_discrimination", "mode_discrimination_distinct_shapes")

    @staticmethod
    def _modes(name="mode_discrimination_distinct_shapes"):
        _, record = _load()
        return record["cases"][name]["mode_results"]

    def test_mixing_zero_cannot_discriminate_and_is_why_these_run_at_half(self):
        """At mixing 0 the mixed objective IS the shape objective, so a test
        there measures nothing about color. Pinned so nobody 'simplifies' the
        fixture to mixing 0 and keeps green tests that test nothing."""
        z, record = _load()
        for name in self.MODE_CASES:
            c = _case(z, name)
            at0 = _recompute_columns(c, 0.0)
            assert at0["tanimoto_combination"] == at0["tanimoto_shape"], name
            assert record["cases"][name]["mixing"] > 0.0, name

    def test_the_rescore_mode_keeps_the_shape_pose_and_shape_score(self):
        """shape_pose_color_rescore's defining property: pose and shape score
        identical to shape_only's -- here literally the same floats, which is
        the semantics the CPU backend must reproduce bit-for-bit."""
        for name in self.MODE_CASES:
            m = self._modes(name)
            a = m[C.MODE_SHAPE_ONLY]
            b = m[C.MODE_SHAPE_POSE_COLOR_RESCORE]
            assert a["pose_quat"] == b["pose_quat"], name
            assert a["pose_translation"] == b["pose_translation"], name
            assert (a["columns"]["tanimoto_shape"]
                    == b["columns"]["tanimoto_shape"]), name

    def test_the_shape_search_has_something_to_find(self):
        """Review correction 1 (blocking). The original mode case has
        byte-identical shapes, so its shape optimum is the identity at
        exactly 1.0 and any implementation that returns the identity pose
        WITHOUT SEARCHING satisfies its mode-2 rows -- the exact failure the
        mode-2 bar exists to catch. The distinct-shapes case is required to
        have a shape optimum that is not the identity and not 1.0, so the
        bar is satisfiable only by running the search. Both halves pinned:
        the degenerate case must STAY degenerate (it is the
        perfect-shape-match control and its docstring says so), and the
        distinct case must stay discriminating."""
        z, record = _load()
        deg = _case(z, "mode_discrimination")
        assert np.array_equal(deg["query_shape"], deg["library_shape"]), (
            "the control case's shapes are no longer identical; its"
            " documented role changed underneath its description")

        m = self._modes("mode_discrimination_distinct_shapes")
        a = m[C.MODE_SHAPE_ONLY]
        assert a["columns"]["tanimoto_shape"] < 0.999, (
            "the distinct-shapes case scores ~1.0 on shape; it no longer "
            "exercises the search")
        q = np.array(a["pose_quat"])
        t = np.array(a["pose_translation"])
        # Compare against the identity as a ROTATION (q and -q are the same
        # rotation), and require a visible margin on at least one component.
        rot_from_identity = 1.0 - abs(float(q[0]))
        assert rot_from_identity > 1e-4 or np.abs(t).max() > 1e-3, (
            "the distinct-shapes shape optimum IS the identity pose; the "
            "mode-2 bar is once again satisfiable without searching")

    def test_shape_only_reports_zero_color_and_column_zero_is_the_shape_score(self):
        for name in self.MODE_CASES:
            cols = self._modes(name)[C.MODE_SHAPE_ONLY]["columns"]
            for col in ("tanimoto_color", "overlap_color", "self_color_query",
                        "self_color_library"):
                assert cols[col] == 0.0, name
            assert (cols["tanimoto_combination"]
                    == cols["tanimoto_shape"]), name

    def test_the_rescore_mode_populates_color_at_the_shape_pose(self):
        """Both color modes differ from shape-only: real color columns, and a
        column 0 that is no longer the shape score."""
        for name in self.MODE_CASES:
            m = self._modes(name)
            for mode in (C.MODE_SHAPE_POSE_COLOR_RESCORE,
                         C.MODE_COMBO_OPTIMAL):
                cols = m[mode]["columns"]
                assert cols["tanimoto_color"] > 0.0, (name, mode)
                assert cols["overlap_color"] > 0.0, (name, mode)
                assert (cols["tanimoto_combination"]
                        != cols["tanimoto_shape"]), (name, mode)

    def test_the_combo_mode_differs_from_the_rescore_mode_at_nonzero_mixing(self):
        """Mode 3 against mode 2, with margins far above every tolerance in
        the fixture: a better mixed score, bought with shape, at a different
        pose. An implementation that runs mode 2 and calls it mode 3 (or the
        reverse) fails here."""
        for name in self.MODE_CASES:
            m = self._modes(name)
            rescore = m[C.MODE_SHAPE_POSE_COLOR_RESCORE]
            combo = m[C.MODE_COMBO_OPTIMAL]
            assert (combo["columns"]["tanimoto_combination"]
                    > rescore["columns"]["tanimoto_combination"] + 0.1), name
            assert (combo["columns"]["tanimoto_shape"]
                    < rescore["columns"]["tanimoto_shape"] - 0.1), name
            dq = np.abs(np.array(combo["pose_quat"])
                        - np.array(rescore["pose_quat"])).max()
            assert dq > 0.01, (
                f"{name}: the two modes converged to the same pose")

    def test_the_combo_pose_is_a_stationary_point_of_column_zero(self):
        """The combo gradient is the derivative of the reported mixed score,
        so at the combo optimum it vanishes -- and at the shape pose it does
        not. An optimizer following anything else (raw overlap sums, a
        single-channel gradient) is optimizing a different function and
        fails one of the two halves."""
        z, record = _load()
        for name in self.MODE_CASES:
            c = _case(z, name)
            m = self._modes(name)
            mixing = record["cases"][name]["mixing"]

            def grad_at(mode):
                return C.mixed_tanimoto_gradient(
                    c["query_shape"], c["library_shape"],
                    c["query_color"], c["query_types"],
                    c["library_color"], c["library_types"],
                    c["interaction_r"], c["interaction_p"], mixing,
                    np.array(m[mode]["pose_quat"]),
                    np.array(m[mode]["pose_translation"]), h=1e-5)

            at_combo = np.linalg.norm(grad_at(C.MODE_COMBO_OPTIMAL))
            at_shape = np.linalg.norm(grad_at(C.MODE_SHAPE_POSE_COLOR_RESCORE))
            assert at_combo < 1e-4, (
                f"{name}: not stationary at its own optimum")
            assert at_shape > 100 * at_combo, (
                f"{name}: the mixed gradient vanishes at the shape pose too; "
                f"the case no longer discriminates the objectives")

    def test_the_cpu_backend_refuses_incomplete_color_and_shape_mixing(self):
        """The CPU backend scores complete color input; incomplete input cannot fall back."""
        from roshambo3.backend import cpu
        if not cpu.available():
            pytest.skip("no CPU extension in this build")
        kw = dict(query_coords=np.zeros((1, 4, 4), np.float32),
                  query_n_atoms=np.array([2], np.int32),
                  library_coords=np.zeros((1, 4, 4), np.float32),
                  library_counts=np.array([[2]], np.uint8),
                  pair_query_row=np.zeros(1, np.int32),
                  pair_library_row=np.zeros(1, np.int32))
        with pytest.raises(ValueError, match="requires query_types"):
            cpu.score_pairs(color=True, **kw)
        with pytest.raises(ValueError, match="shape_only mode requires mixing=0"):
            cpu.score_pairs(mixing=0.5, **kw)


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")
class TestTheNumericalGradient:
    def test_the_two_committed_step_sizes_agree(self):
        """Central differences at h and h/2: O(h^2) truncation means the two
        must agree to ~1e-8 relative, and their difference is the committed
        uncertainty a consumer reads."""
        z, record = _load()
        for name, meta in record["cases"].items():
            if not meta["gradients"]:
                continue
            c = _case(z, name)
            for stem in ("color_gradient", "mixed_gradient"):
                a, b = c[f"{stem}_h0"], c[f"{stem}_h1"]
                # Relative agreement where the gradient is alive, an absolute
                # floor where it is not: the mode case's committed pose is the
                # combo OPTIMUM, so its mixed gradient there is ~h^2 times the
                # third derivative -- pure truncation residual, both estimates
                # are "zero", and a relative criterion at a stationary point
                # compares noise against noise. 5e-8 admits that residual at
                # h = 1e-4 and is still ~zero against every live gradient in
                # the fixture.
                bound = max(1e-6 * float(np.abs(a).max()), 5e-8)
                assert float(np.abs(a - b).max()) < bound, (name, stem)

    def test_the_translation_gradient_points_from_library_toward_query(self):
        """One feature each: moving the library feature toward the query
        raises the overlap, so the translation gradient of V_color points
        along (query - posed_library)."""
        z, _ = _load()
        c = _case(z, "one_feature_each")
        g = c["color_gradient_h0"][3:]
        direction = (np.asarray(c["query_color"][0], np.float64)
                     - np.asarray(c["library_color_posed"][0], np.float64))
        cos = float(g @ direction / (np.linalg.norm(g)
                                     * np.linalg.norm(direction)))
        assert cos > 0.999, cos

    def test_the_rotation_components_are_the_axis_angle_chart(self):
        """Reproduce the committed rotation components by different machinery
        -- Rodrigues' formula directly, no quaternions anywhere -- to pin that
        the committed chart is the space-frame axis-angle increment and not
        an artifact of the quaternion route. A raw quaternion-component
        difference would differ by the chart's Jacobian; this one must not
        differ at all."""
        z, record = _load()
        c = _case(z, "one_feature_each")
        h = record["gradient_h"][0]
        q, t = c["pose_quat"], c["pose_translation"]
        rot = R.quaternion_matrix(q)
        a = np.asarray(c["query_color"], np.float64)
        b = np.asarray(c["library_color"], np.float64)

        def rodrigues(axis, angle):
            k = np.zeros((3, 3))
            k[0, 1], k[0, 2] = -axis[2], axis[1]
            k[1, 0], k[1, 2] = axis[2], -axis[0]
            k[2, 0], k[2, 1] = -axis[1], axis[0]
            return (np.eye(3) + math.sin(angle) * k
                    + (1 - math.cos(angle)) * (k @ k))

        def value(pre):
            posed = b @ (pre @ rot).T + t
            return R.color_overlap(a, c["query_types"], posed,
                                   c["library_types"],
                                   c["interaction_r"], c["interaction_p"])

        for k in range(3):
            axis = np.eye(3)[k]
            got = (value(rodrigues(axis, +h))
                   - value(rodrigues(axis, -h))) / (2 * h)
            assert got == pytest.approx(float(c["color_gradient_h0"][k]),
                                        rel=1e-6, abs=1e-10), k

    def test_the_mixed_gradient_is_the_gradient_of_column_zero(self):
        """Linearity identity: grad(col0) = (1-m) grad(T_shape) +
        m grad(T_color), each side by independent central differences -- and
        NOT the mixed raw-overlap gradient, which is what the uncorrected
        legacy combo weighting effectively follows. Both halves, so a future
        analytic combo gradient has a yes and a no to check against."""
        z, record = _load()
        c = _case(z, "one_feature_each")
        m = record["cases"]["one_feature_each"]["mixing"]
        q, t = c["pose_quat"], c["pose_translation"]

        def channel(col):
            def value(qq, tt):
                return C.score_columns(
                    c["query_shape"], c["library_shape"],
                    c["query_color"], c["query_types"],
                    c["library_color"], c["library_types"],
                    c["interaction_r"], c["interaction_p"], m, qq, tt)[col]
            return C.pose_gradient_numerical(value, q, t, h=1e-4)

        want = (1 - m) * channel("tanimoto_shape") + m * channel(
            "tanimoto_color")
        got = c["mixed_gradient_h0"]
        assert np.allclose(got, want, rtol=1e-5, atol=1e-9)

        raw = (1 - m) * channel("overlap_volume") + m * channel(
            "overlap_color")
        raw_dir = raw / np.linalg.norm(raw)
        got_dir = got / np.linalg.norm(got)
        assert float(raw_dir @ got_dir) < 0.999, (
            "the raw-overlap blend points where the Tanimoto gradient "
            "points on this case; it no longer discriminates the wrong "
            "combo weighting")


# ---------------------------------------------------------------------------
# The CUDA cross-check: the current analytic path against the fixture
# ---------------------------------------------------------------------------

def _cuda_launch_arrays(c: dict, n_features: int):
    """Pack one fixture case the way `backend.score_pairs` expects: query
    features immediately after the query's real atoms, library row 0 the
    PRE-POSED library with features at `library_color_offset`, and -- when the
    case's library carries no features -- a helper row with one feature so
    the launch passes the no-features-at-all refusal. The helper row is
    cross-check plumbing, never part of the case."""
    from roshambo3.layout import counts_from_types, tile32

    qn = c["query_shape"].shape[0]
    qc_n = c["query_color"].shape[0]
    ln = c["library_shape_posed"].shape[0]
    lc_n = c["library_color_posed"].shape[0]
    L = max(qn + qc_n, ln + max(lc_n, 1))

    qcoords = np.zeros((1, L, 4), np.float32)
    qtypes = np.zeros((1, L), np.int32)
    qcoords[0, :qn, :3] = c["query_shape"]
    qcoords[0, :qn, 3] = 1.0
    qcoords[0, qn:qn + qc_n, :3] = c["query_color"]
    qcoords[0, qn:qn + qc_n, 3] = 1.0
    qtypes[0, qn:qn + qc_n] = c["query_types"]

    n_rows = 32
    lib = np.zeros((n_rows, L, 4), np.float32)
    ltypes = np.zeros((n_rows, L), np.int32)
    lib[0, :ln, :3] = c["library_shape_posed"]
    lib[0, :ln, 3] = 1.0
    lib[0, ln:ln + lc_n, :3] = c["library_color_posed"]
    lib[0, ln:ln + lc_n, 3] = 1.0
    ltypes[0, ln:ln + lc_n] = c["library_types"]
    if lc_n == 0:
        lib[1] = lib[0]
        lib[1, ln, :3] = 0.0
        lib[1, ln, 3] = 1.0
        ltypes[1] = 0
        ltypes[1, ln] = 1
    return dict(
        query_coords=qcoords, query_types=qtypes,
        query_n_atoms=np.array([qn], np.int32),
        library_coords=tile32(lib),
        library_counts=counts_from_types(
            ltypes, np.full(n_rows, ln, np.int32), n_features),
        library_color_offset=ln,
        pair_query_row=np.zeros(1, np.int32),
        pair_library_row=np.zeros(1, np.int32),
        interaction_r=np.asarray(c["interaction_r"], np.float32),
        interaction_p=np.asarray(c["interaction_p"], np.float32))


@needs_gpu
@pytest.mark.skipif(not FIXTURE.exists(), reason=f"no fixture at {FIXTURE}")
class TestTheCudaAnalyticPathIsCrossChecked:
    """The deciding measurement's cross-check half. Agreement here is
    evidence about the CUDA port at a fixed pose, not about the mathematics;
    the fixture was committed before any of this ran."""

    def test_every_supported_case_agrees_at_the_derived_tolerance(self):
        from roshambo3.backend import PoseSearch
        z, record = _load()
        failures = []
        for name, meta in record["cases"].items():
            c = _case(z, name)
            if "unguarded" in meta["cuda"]:
                continue        # the named-defect test below owns it
            arrays = _cuda_launch_arrays(c, c["interaction_r"].shape[0])
            v = backend.score_pairs(pose_search=PoseSearch.ANALYTIC,
                                    color=True, mixing=meta["mixing"],
                                    steps=0, start_mode=0, **arrays)
            for i, col in enumerate(record["scalar_columns"]):
                got, want, tol = float(v[0, i]), c["columns"][i], \
                    c["tolerance"][i]
                # float32 storage of the committed float64 value costs up to
                # half an ulp of the value itself, on top of the arithmetic
                # bound, which is stated in float64.
                slack = tol + abs(want) * 2.0 ** -24
                if not abs(got - want) <= slack:
                    failures.append(f"{name}.{col}: cuda {got!r} vs fixture "
                                    f"{want!r} (|d| {abs(got - want):.3e} > "
                                    f"{slack:.3e})")
        assert not failures, "\n".join(failures)

    def test_the_analytic_path_consumes_the_query_at_the_wire_grid(self):
        """Not a defect -- a design decision that had to be rediscovered
        by bisection, pinned here so nobody pays for it twice.
        The analytic path stages the query (shape and color) as three int16
        planes at the corpus's wire-step step; on-grid coordinates round-trip
        exactly, off-grid ones are consumed at the grid. So: an
        off-grid query feature must make CUDA agree with the reference
        evaluated at the SNAPPED query -- tightly -- and disagree with the
        reference at the raw query by the quantization's own size. The
        library side is float32-exact on a color launch, which the same
        construction verifies by leaving it off-grid.

        This is why every input coordinate in the committed fixture is a
        wire-grid value (doc/COLOR_CONTRACT.md section 9), and why a CPU
        implementation -- which has no bus and no staging -- must consume
        float32 at float32 rather than importing this representation."""
        from roshambo3.backend import PoseSearch
        z, record = _load()
        base = _case(z, "one_feature_each")
        c = dict(base)
        # One off-grid query feature at moderate separation: the exponent is
        # small enough that the term is O(1) and the quantization visible.
        c["query_color"] = np.array([[0.30047, 0.20031, 0.10057]], np.float32)
        c["library_color_posed"] = np.array([[0.9, 0.2, 0.0]], np.float32)
        arrays = _cuda_launch_arrays(c, c["interaction_r"].shape[0])
        v = backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=0.5, steps=0, start_mode=0, **arrays)
        got = float(v[0, S.OVERLAP_COLOR])

        def ref_at(query_color):
            return C.score_columns(
                c["query_shape"], c["library_shape_posed"],
                query_color, c["query_types"],
                c["library_color_posed"], c["library_types"],
                c["interaction_r"], c["interaction_p"], 0.5)["overlap_color"]

        snapped = (np.round(c["query_color"].astype(np.float64) / float(WIRE_QUANT_STEP))
                   .astype(np.float32) * WIRE_QUANT_STEP).astype(np.float32)
        at_snapped = ref_at(snapped)
        at_raw = ref_at(c["query_color"])
        assert abs(got - at_snapped) < 1e-5 * abs(at_snapped), (
            "CUDA no longer matches the reference at the wire-grid-snapped "
            "query: either the query staging changed (a declared numerics "
            "change -- record it and update doc/COLOR_CONTRACT.md section 9) "
            "or something else moved")
        assert abs(at_raw - at_snapped) > 20 * abs(got - at_snapped), (
            "the constructed feature no longer discriminates the grid from "
            "float32; move it further off-grid")

    @pytest.mark.parametrize("pose_search", [
        pytest.param("FIELD", id="field"),
        pytest.param("ANALYTIC", id="analytic"),
    ])
    def test_empty_empty_color_tanimoto_is_zero_on_both_kernels(self,
                                                                 pose_search):
        """An absent color representation is zero similarity on both kernels,
        not a perfect match and not an undefined NaN."""
        from roshambo3.backend import PoseSearch
        pose_search = getattr(PoseSearch, pose_search)
        z, record = _load()
        c = _case(z, "empty_color_both_sides")
        meta = record["cases"]["empty_color_both_sides"]
        arrays = _cuda_launch_arrays(c, c["interaction_r"].shape[0])
        v = backend.score_pairs(pose_search=pose_search, color=True,
                                mixing=meta["mixing"], steps=0, start_mode=0,
                                **arrays)
        assert float(v[0, S.TANIMOTO_COLOR]) == 0.0
        assert float(v[0, S.TANIMOTO_COMBINATION]) == pytest.approx(
            (1.0 - meta["mixing"]) * float(v[0, S.TANIMOTO_SHAPE]))
        assert float(v[0, S.TANIMOTO_SHAPE]) == pytest.approx(
            _columns_dict(c["columns"], record)["tanimoto_shape"], rel=1e-5)

    def test_the_reported_pose_reproduces_the_reported_color_score(self):
        """End to end through the real optimizer: run the analytic combo path
        for 25 steps, read the pose back out of the score row, and evaluate
        the reference at that pose from the ORIGINAL coordinates. This is the
        one check that exercises the kernel's own pose application against
        the contract's pose convention, so it uses the tolerance variant
        that budgets for the kernel rotating in float32."""
        from roshambo3.backend import ColorPlacement, PoseSearch
        z, record = _load()
        name = "asymmetric_matrix_indexing"
        c = _case(z, name)
        meta = record["cases"][name]
        # Unposed library: let the kernel find its own pose.
        unposed = dict(c)
        unposed["library_shape_posed"] = c["library_shape"]
        unposed["library_color_posed"] = c["library_color"]
        arrays = _cuda_launch_arrays(unposed, c["interaction_r"].shape[0])
        v = backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=meta["mixing"], steps=25, start_mode=1,
                                color_placement=ColorPlacement.IN_GRADIENT,
                                **arrays)
        q = np.asarray(v[0, S.POSE_QUAT], np.float64)
        t = np.asarray(v[0, S.POSE_TRANSLATION], np.float64)
        start = np.asarray(v[0, S.START_QUAT], np.float64)
        # The reported pose composes with the start rotation: the start-mode
        # rotation is applied to the QUERY, so the library-frame transform to
        # reproduce the score applies the start's inverse on the way in.
        sq = np.array([start[0], -start[1], -start[2], -start[3]])
        q_total = C.quat_multiply(sq, q)
        t_total = C.apply_pose_elementwise(t[None], sq, np.zeros(3))[0]
        got = C.score_columns(
            c["query_shape"], c["library_shape"],
            c["query_color"], c["query_types"],
            c["library_color"], c["library_types"],
            c["interaction_r"], c["interaction_p"], meta["mixing"],
            q_total, t_total)
        tol = C.fixed_pose_tolerance(
            c["query_shape"],
            C.apply_pose_elementwise(c["library_shape"], q_total, t_total),
            c["query_color"], c["query_types"],
            C.apply_pose_elementwise(c["library_color"], q_total, t_total),
            c["library_types"],
            c["interaction_r"], c["interaction_p"], meta["mixing"],
            include_input_rounding=True)
        for col, idx in (("tanimoto_combination", S.TANIMOTO_COMBINATION),
                         ("tanimoto_shape", S.TANIMOTO_SHAPE),
                         ("tanimoto_color", S.TANIMOTO_COLOR),
                         ("overlap_color", S.OVERLAP_COLOR)):
            slack = tol[col] + abs(got[col]) * 2.0 ** -23
            assert abs(float(v[0, idx]) - got[col]) <= slack, (
                f"{col}: reported {float(v[0, idx])!r}, reference at the "
                f"reported pose {got[col]!r}")

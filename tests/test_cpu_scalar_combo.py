"""Scalar combo objective and gradient against the independent anchor."""
import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.reference import color as C
from roshambo3.numerics import COORD_SCALE

from test_cpu_scalar_color import cases, packed

pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")


@pytest.mark.parametrize("name", ["mode_discrimination_distinct_shapes",
                                  "asymmetric_matrix_indexing"])
def test_color_descent_matches_independent_axis_angle_finite_difference(name):
    _, meta, case, _ = next(x for x in cases() if x[0] == name)
    kw = packed(case, posed=False)
    q = np.asarray(case["pose_quat"], np.float32)
    t = np.asarray(case["pose_translation"], np.float32)
    value, descent = cpu.extension().scalar_color_value_gradient(
        kw["query_coords"][0], kw["query_types"][0],
        int(kw["query_n_atoms"][0]), kw["library_coords"][0],
        kw["library_counts"][0], kw["interaction_r"],
        kw["interaction_p"], q, t)
    numerical = C.color_overlap_gradient(
        case["query_color"], case["query_types"],
        case["library_color"], case["library_types"],
        case["interaction_r"], case["interaction_p"], q, t,
        h=1e-4)
    # The optimizer's rotation chart is quaternion HALF-angle; the oracle's
    # numerical derivative is full axis-angle. Translation is reduced frame.
    comparable = -np.r_[descent[1:4]/2, descent[4:7]*COORD_SCALE]
    np.testing.assert_allclose(comparable, numerical, rtol=3e-4, atol=2e-5)
    assert value > 0


@pytest.mark.parametrize("name", ["mode_discrimination_distinct_shapes",
                                  "asymmetric_matrix_indexing"])
@pytest.mark.parametrize("mixing", [0.25, 0.5, 0.75])
def test_weighted_combo_direction_matches_numerical_reported_score(name, mixing):
    _, _, case, _ = next(x for x in cases() if x[0] == name)
    kw = packed(case, posed=False)
    q = np.asarray(case["pose_quat"], np.float32)
    t = np.asarray(case["pose_translation"], np.float32)
    descent = cpu.extension().scalar_combo_direction(
        kw["query_coords"][0], kw["query_types"][0],
        int(kw["query_n_atoms"][0]), kw["library_coords"][0],
        kw["library_counts"][0], kw["interaction_r"],
        kw["interaction_p"], q, t, np.float32(mixing))
    numerical = C.mixed_tanimoto_gradient(
        case["query_shape"], case["library_shape"],
        case["query_color"], case["query_types"],
        case["library_color"], case["library_types"],
        case["interaction_r"], case["interaction_p"], mixing, q, t,
        h=1e-4)
    comparable = -np.r_[descent[1:4]/2, descent[4:7]*COORD_SCALE]
    np.testing.assert_allclose(comparable, numerical, rtol=3e-3, atol=2e-5)


@pytest.mark.parametrize("name", ["mode_discrimination_distinct_shapes",
                                  "asymmetric_matrix_indexing"])
def test_tiny_combined_step_increases_the_reported_objective(name):
    _, _, case, _ = next(x for x in cases() if x[0] == name)
    kw = packed(case, posed=False)
    base = dict(color=True, color_mode=C.MODE_COMBO_OPTIMAL, mixing=0.5,
                start_mode=0, lr_q=1e-5, lr_t=1e-5,
                budget_override="T156 local derivative check")
    before = cpu.score_pairs(steps=1, **base, **kw)
    after = cpu.score_pairs(steps=2, **base, **kw)
    assert after[0, 0] > before[0, 0]


@pytest.mark.parametrize("name", ["mode_discrimination_distinct_shapes",
                                  "asymmetric_matrix_indexing"])
def test_combo_counts_every_start_and_reuses_the_winning_score(name):
    _, _, case, _ = next(x for x in cases() if x[0] == name)
    kw = packed(case, posed=False)
    out, counts = cpu.score_pairs(
        color=True, color_mode=C.MODE_COMBO_OPTIMAL, mixing=0.5,
        steps=3, start_mode=1, with_census=True, **kw)
    assert counts["color_gradient_evaluations"] == 4*3
    assert counts["color_per_start_final_evaluations"] == 4
    assert counts["color_winner_rescore_evaluations"] == 0
    assert counts["color_evaluations"] == 4*(3+1)
    assert 0 <= counts["winning_starts"][0] < 4
    assert np.isfinite(out).all()
    assert np.isclose(out[0, 0], 0.5*(out[0, 1]+out[0, 2]), atol=1e-6)
    oracle = C.score_columns(
        case["query_shape"], case["library_shape"],
        case["query_color"], case["query_types"],
        case["library_color"], case["library_types"],
        case["interaction_r"], case["interaction_p"], 0.5,
        out[0, 9:13], out[0, 13:16])
    posed_s = C.apply_pose_elementwise(case["library_shape"],
                                       out[0, 9:13], out[0, 13:16])
    posed_c = C.apply_pose_elementwise(case["library_color"],
                                       out[0, 9:13], out[0, 13:16])
    tol = C.fixed_pose_tolerance(
        case["query_shape"], posed_s, case["query_color"],
        case["query_types"], posed_c, case["library_types"],
        case["interaction_r"], case["interaction_p"], 0.5,
        include_input_rounding=True)
    for key, col in (("tanimoto_color", 2), ("overlap_color", 4),
                     ("self_color_query", 7), ("self_color_library", 8)):
        assert abs(float(out[0, col])-oracle[key]) <= max(tol[key], 1e-7), key


def test_empty_color_is_finite_and_mixing_zero_is_only_a_control():
    _, _, case, _ = next(x for x in cases()
                         if x[0] == "empty_color_both_sides")
    kw = packed(case, posed=False)
    out = cpu.score_pairs(color=True, color_mode=C.MODE_COMBO_OPTIMAL,
                          mixing=0.5, steps=3, start_mode=0, **kw)
    assert np.isfinite(out).all()
    assert np.array_equal(out[:, [2, 4, 7, 8]], np.zeros((1, 4), np.float32))
    control = cpu.score_pairs(color=True, color_mode=C.MODE_COMBO_OPTIMAL,
                              mixing=0.0, steps=3, start_mode=0, **kw)
    assert np.array_equal(control[:, 0], control[:, 1])


def test_mode3_moves_to_a_better_mixed_pose_than_mode2_on_named_case():
    _, _, case, _ = next(x for x in cases()
                         if x[0] == "mode_discrimination_distinct_shapes")
    kw = packed(case, posed=False)
    rescore = cpu.score_pairs(color=True, mixing=0.5, steps=25, **kw)
    combo = cpu.score_pairs(color=True, color_mode=C.MODE_COMBO_OPTIMAL,
                            mixing=0.5, steps=25, **kw)
    assert combo[0, 0] > rescore[0, 0] + 0.05
    assert not np.array_equal(combo[0, 9:16], rescore[0, 9:16])


def test_combo_phase_decomposition_is_present_but_not_a_rate():
    _, _, case, _ = next(x for x in cases() if x[0] == "one_feature_each")
    kw = packed(case, posed=False)
    out, phases = cpu.score_pairs(
        color=True, color_mode=C.MODE_COMBO_OPTIMAL, mixing=0.5,
        steps=2, start_mode=0, with_phase_timers=True, **kw)
    for name in ("shape_gradient", "color_gradient", "combo_weighting",
                 "pose_update", "per_start_final_value",
                 "exact_final_rescore"):
        assert name in phases and phases[name] >= 0.0
    assert np.isfinite(out).all()


def test_combo_keeps_the_budget_guard_and_refuses_unimplemented_grain():
    _, _, case, _ = next(x for x in cases() if x[0] == "one_feature_each")
    kw = packed(case, posed=False)
    opts = dict(color=True, color_mode=C.MODE_COMBO_OPTIMAL,
                mixing=0.5, steps=10)
    with pytest.raises(ValueError, match="budget"):
        cpu.score_pairs(**opts, lr_t=0.4, **kw)
    with pytest.raises(NotImplementedError, match="grain"):
        cpu.score_pairs(**opts, grain=2, **kw)


@pytest.mark.parametrize("mode", [C.MODE_SHAPE_POSE_COLOR_RESCORE,
                                    C.MODE_COMBO_OPTIMAL])
def test_color_modes_reject_short_query_type_plane(mode):
    _, _, case, _ = next(x for x in cases() if x[0] == "one_feature_each")
    kw = packed(case, posed=False)
    kw["query_types"] = kw["query_types"][:, :-1]
    with pytest.raises(ValueError, match="query_types must match"):
        cpu.score_pairs(color=True, color_mode=mode, mixing=0.5, **kw)


def test_scalar_extension_rejects_short_query_type_plane():
    _, _, case, _ = next(x for x in cases() if x[0] == "one_feature_each")
    kw = packed(case, posed=False)
    kw["query_types"] = kw["query_types"][:, :-1]
    args = [kw[k] for k in ("query_coords", "query_types", "query_n_atoms",
                            "library_coords", "library_counts",
                            "pair_query_row", "pair_library_row",
                            "interaction_r", "interaction_p")]
    shape = np.zeros((1, 20), np.float32)
    with pytest.raises(ValueError, match="query types must match"):
        cpu.extension().rescore_scalar_color(
            *args, shape, np.float32(0.5), False, False)
    with pytest.raises(ValueError, match="query types must match"):
        cpu.extension().optimize_scalar_combo(
            *args, 0, np.float32(0.1), np.float32(0.1), 0,
            np.float32(0.5), False, False)


@pytest.mark.parametrize("mode", [C.MODE_SHAPE_POSE_COLOR_RESCORE,
                                    C.MODE_COMBO_OPTIMAL])
def test_color_modes_reject_negative_interaction_width(mode):
    _, _, case, _ = next(x for x in cases() if x[0] == "one_feature_each")
    kw = packed(case, posed=False)
    kw["interaction_r"][1, 1] = -1.0
    with pytest.raises(ValueError, match="widths must be nonnegative"):
        cpu.score_pairs(color=True, color_mode=mode, mixing=0.5, **kw)

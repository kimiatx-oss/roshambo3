"""Compatible vector pairs match independent four-start singleton evaluations."""
import numpy as np
import pytest
import platform
from roshambo3.backend import cpu
from test_cpu_color_start_lanes import _population

pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")

@pytest.mark.parametrize("factor", [False, True])
@pytest.mark.parametrize("exponential", ["baseline", "rawbits", "rawbits_unclamped"])
@pytest.mark.parametrize("policy", ["winner", "all_start"])
def test_compatible_pair_matches_singletons(factor, exponential, policy):
    kw = _population(5, "canonical", shared_queries=True)
    # Distinct coordinates with equal shape and full color signatures in the
    # first pair, a mismatched second pair, and a singleton tail.
    kw["library_counts"][1] = kw["library_counts"][0]
    kw["library_coords"][1] = kw["library_coords"][0]
    kw["library_coords"][1, :, :3] += np.float32(0.013)
    opts = dict(color=True, color_mode="combo_optimal", mixing=0.5, steps=3,
                budget_override="T231 pair arithmetic fixture", start_mode=1,
                color_grain=16, pose_contract="shared_fp32_reduced_v1",
                canonical_color_kernel=True, factor_canonical_torque=factor,
                optimizer_exp_mode=exponential, final_start_policy=policy,
                shape_backend="pipelined_field", shape_grain="library_atom",
                shape_width=8, shape_field_spacing_angstrom=0.5)
    got, stats = cpu.score_pairs(**kw, **opts, with_census=True)
    rows = []
    starts = []
    for row in range(5):
        one = dict(kw, pair_query_row=kw["pair_query_row"][row:row+1],
                   pair_library_row=kw["pair_library_row"][row:row+1])
        score, census = cpu.score_pairs(**one, **opts, with_census=True)
        rows.append(score)
        starts.extend(census["winning_starts"])
    np.testing.assert_array_equal(got, np.concatenate(rows))
    np.testing.assert_array_equal(stats["winning_starts"], starts)
    assert np.isfinite(got).all()
    # Existing color census counts optimizer evaluations; the final pass deliberately
    # passes no color census to the fresh converged SEARCH evaluation.
    evaluations = 3
    enabled = cpu.color_build_stamp(16)["color_compatible_pair_policy_enabled"]
    x86 = platform.machine() in ("x86_64", "AMD64")
    arm = platform.machine().lower() in ("aarch64", "arm64")
    streams = arm and cpu.color_build_stamp(16).get("color_neon_streams", 1) == 2
    assert stats["color_compatible_ymm8_evaluations"] == (evaluations if enabled else 0)
    assert stats.get("color_compatible_neon4_two_stream_evaluations", 0) == (evaluations if streams else 0)
    expected = (3 if enabled or streams else 5)*evaluations
    assert stats["color_singleton_four_start_evaluations"] == expected
    assert stats["color_singleton_xmm4_evaluations"] == (expected if x86 else 0)
    assert stats["color_singleton_neon4_evaluations"] == (expected if arm else 0)

def test_empty_planes_keep_full_signature_and_tail():
    kw = _population(3, "canonical", shared_queries=True)
    kw["library_counts"][:, 1:] = 0
    kw["library_coords"][:, 2:] = 0
    opts = dict(color=True, color_mode="combo_optimal", mixing=0.5, steps=2,
                budget_override="T231 empty planes", start_mode=1,
                color_grain=16, pose_contract="shared_fp32_reduced_v1",
                canonical_color_kernel=True, with_census=True)
    got, stats = cpu.score_pairs(**kw, **opts)
    assert np.isfinite(got).all()
    assert stats["color_start_lane_scalar_fallback_pairs"] == 0

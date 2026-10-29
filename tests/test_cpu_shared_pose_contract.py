"""One FP32 reduced pose drives supported CPU shape/color consumers."""

import numpy as np
import pytest

from roshambo3.backend import cpu
from cpu_test_capabilities import IS_ARM, RETAINED_COLOR_GRAIN
from roshambo3.reference.color import MODE_COMBO_OPTIMAL


def _inputs(*, same_query=False, canonical=False):
    q = np.zeros((2, 4, 4), dtype=np.float32)
    q[0, 0] = (0.0, 0.0, 0.0, 1.0)
    q[0, 1] = (0.5, 0.1, 0.0, 1.0)
    q[0, 2] = (-0.2, 0.4, 0.1, 1.0)
    q[1, 0] = (0.1, -0.1, 0.0, 1.0)
    q[1, 1] = (0.3, 0.2, -0.1, 1.0)
    q[1, 2] = (9.0, 9.0, 9.0, 1.0)  # padding, not a second color feature
    qt = np.array([[0, 1, 2, 0], [0, 1, 0, 0]], dtype=np.int32)
    qn = np.array([1, 1], dtype=np.int32)
    if same_query:
        qt[1] = qt[0]
        q[1, 1:3] = q[0, 1:3]

    library = np.zeros((3, 3, 4), dtype=np.float32)
    library[0, 0, :3] = (0.0, 0.0, 0.0)
    library[0, 1, :3] = (0.4, -0.1, 0.0)
    library[0, 2, :3] = (-0.1, 0.5, 0.2)
    library[1, 0, :3] = (0.2, 0.1, -0.1)
    library[1, 1, :3] = (0.1, 0.2, 0.3)
    library[1, 2, :3] = (-0.3, 0.6, -0.2)
    library[2, 0, :3] = (-0.2, 0.1, 0.1)
    library[2, 1, :3] = (0.7, -0.2, 0.2)
    library[2, 2, :3] = (4.0, 4.0, 4.0)  # masked absent feature
    counts = np.array([[1, 1, 1], [1, 2, 0], [1, 1, 0]], dtype=np.uint8)

    widths = np.zeros((3, 3), dtype=np.float32)
    heights = np.zeros((3, 3), dtype=np.float32)
    if canonical:
        widths[1:, 1:] = 1.0
        heights[1, 1] = heights[2, 2] = 1.0
    else:
        widths[1:, 1:] = [[0.8, 1.2], [1.5, 0.7]]
        heights[1:, 1:] = [[1.0, 0.7], [0.4, 0.9]]
    return q, qt, qn, library, counts, widths, heights


def _prepare(data, backend="scalar_analytic", grain="scalar"):
    q, qt, qn, _, _, widths, heights = data
    width = 8 if backend in ("analytic_simd", "paired_channel_field",
                             "pipelined_field") else 1
    return cpu.prepare_combo_shared_pose_queries(
        q, qt, qn, widths, heights, shape_backend=backend,
        shape_grain=grain, shape_width=width,
        field_spacing_angstrom=(0.5 if "field" in backend else None))


def _score(data, prepared, *, grain=RETAINED_COLOR_GRAIN, use_plan=True, reuse=True,
           backend="scalar_analytic", shape_grain="scalar", census=True,
           factor_torque=False, optimizer_exp_mode="baseline",
           canonical_color_kernel=False):
    q, qt, qn, library, counts, widths, heights = data
    queries = np.array([0, 0, 0] if shape_grain == "trajectory"
                       else [0, 1, 0], dtype=np.int32)
    targets = np.array([0, 1, 2], dtype=np.int32)
    return cpu.score_pairs(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        pair_query_row=queries, pair_library_row=targets,
        interaction_r=widths, interaction_p=heights,
        color=True, color_mode=MODE_COMBO_OPTIMAL, color_grain=grain,
        steps=2, start_mode=1, mixing=0.5,
        shape_backend=backend, shape_grain=shape_grain,
        shape_width=8 if backend in ("analytic_simd", "paired_channel_field",
                                     "pipelined_field") else 1,
        shape_field_spacing_angstrom=(0.5 if "field" in backend else None),
        prepared_shared_pose_queries=prepared,
        pose_contract="shared_fp32_reduced_v1",
        use_color_invariant_plan=use_plan,
        reuse_exact_final_scratch=reuse, with_census=census,
        factor_canonical_torque=factor_torque,
        optimizer_exp_mode=optimizer_exp_mode,
        canonical_color_kernel=canonical_color_kernel)


def test_general_interactions_keep_original_path():
    data=_inputs(canonical=False)
    prepared=_prepare(data)
    original,_=_score(data,prepared,grain=16,factor_torque=False)
    factored,_=_score(data,prepared,grain=16,factor_torque=True)
    np.testing.assert_array_equal(original,factored)


def test_selector_refuses_unimplemented_color_grain():
    data=_inputs(canonical=True)
    prepared=_prepare(data)
    with pytest.raises(ValueError,match="factor_canonical_torque requires"):
        _score(data,prepared,grain=11,factor_torque=True)


@pytest.mark.parametrize("grain", [11, 16])
@pytest.mark.parametrize("canonical", [False, True])
def test_shared_pose_color11_and_color16_plan_and_exact_scratch_parity(
        grain, canonical):
    data = _inputs(canonical=canonical)
    prepared = _prepare(data)
    scores_on, stats_on = _score(data, prepared, grain=grain)
    scores_scratch_off, stats_scratch_off = _score(
        data, prepared, grain=grain, use_plan=True, reuse=False)
    scores_plan_off, stats_plan_off = _score(
        data, prepared, grain=grain, use_plan=False, reuse=True)

    np.testing.assert_array_equal(scores_on, scores_scratch_off)
    np.testing.assert_array_equal(scores_on, scores_plan_off)
    assert stats_on["pose_contract"] == "shared_fp32_reduced_v1"
    assert stats_on["color_invariant_plan_enabled"] is True
    assert stats_scratch_off["color_invariant_plan_enabled"] is True
    assert stats_plan_off["color_invariant_plan_enabled"] is False
    assert stats_on["shared_pose_preparations"] == 3 * (2 * 4 + 4)
    assert stats_on["color_per_start_final_evaluations"] == 3 * 4
    assert stats_on["color_winner_rescore_evaluations"] == 0
    assert stats_scratch_off["color_per_start_final_evaluations"] == 3 * 4
    np.testing.assert_array_equal(stats_on["winning_starts"],
                                  stats_scratch_off["winning_starts"])
    np.testing.assert_array_equal(stats_on["winning_starts"],
                                  stats_plan_off["winning_starts"])
    # Exact per-start scoring still reuses worker-local scratch, but winner
    # serialization now consumes the saved score and retains no coordinates.
    assert stats_on["exact_final_scratch_enabled"] is True
    assert stats_on["exact_final_winner_candidates_retained"] == 0
    assert stats_on["exact_final_library_points_prepared"] > 0
    assert stats_scratch_off["exact_final_scratch_buffer_growths"] > 0
    assert stats_scratch_off["exact_final_scratch_enabled"] is False
    assert stats_plan_off["exact_final_scratch_enabled"] is True


@pytest.mark.parametrize("backend,grain", [
    ("scalar_analytic", "scalar"),
    ("analytic_simd", "library_atom"),
    ("scalar_field", "scalar"),
    ("paired_channel_field", "library_atom"),
    ("pipelined_field", "library_atom"),
])
def test_shared_pose_shape_backends(backend, grain):
    data = _inputs(same_query=(grain == "trajectory"))
    if grain == "library_atom":
        data = _inputs(same_query=True)
    if IS_ARM and backend == "analytic_simd":
        with pytest.raises(ValueError, match="unavailable AVX2 target"):
            _prepare(data, backend, grain)
        return
    prepared = _prepare(data, backend, grain)
    scores, stats = _score(data, prepared, grain=RETAINED_COLOR_GRAIN, backend=backend,
                           shape_grain=grain, census=True)
    assert np.isfinite(scores).all()
    assert stats["shared_pose_preparations"] == 3 * (2 * 4 + 4)


def test_shared_pose_rejects_storage_equal_interpolation_mismatch():
    data = _inputs()
    q, qt, qn, library, counts, widths, heights = data
    shape_options = dict(shape_backend="pipelined_field",
                         shape_grain="library_atom", shape_width=8,
                         field_spacing_angstrom=0.5,
                         field_padding_angstrom=8.0)
    storage_only = cpu.prepare_combo_shape_evaluators(
        q, qn, **shape_options, field_storage="fp16",
        field_interpolation="fp32")
    mixed = cpu.prepare_combo_shape_evaluators(
        q, qn, **shape_options, field_storage="fp16",
        field_interpolation="mixed_half_yz")
    shared = cpu.prepare_combo_shared_pose_queries(
        q, qt, qn, widths, heights, shape_backend="pipelined_field",
        shape_grain="library_atom", shape_width=8,
        field_spacing_angstrom=0.5, field_padding_angstrom=8.0,
        shape_evaluators=storage_only)

    with pytest.raises(ValueError,
                       match="prepared shared-pose queries do not match inputs/config"):
        cpu.score_pairs(
            query_coords=q, query_types=qt, query_n_atoms=qn,
            library_coords=library, library_counts=counts,
            pair_query_row=np.array([0, 1, 0], dtype=np.int32),
            pair_library_row=np.array([0, 1, 2], dtype=np.int32),
            interaction_r=widths, interaction_p=heights,
            color=True, color_mode=MODE_COMBO_OPTIMAL, color_grain=RETAINED_COLOR_GRAIN,
            steps=2, start_mode=1, mixing=0.5,
            shape_backend="pipelined_field", shape_grain="library_atom",
            shape_width=8, shape_field_spacing_angstrom=0.5,
            shape_field_padding_angstrom=8.0,
            prepared_shape_evaluators=mixed,
            prepared_shared_pose_queries=shared,
            pose_contract="shared_fp32_reduced_v1")


@pytest.mark.skipif(IS_ARM, reason="trajectory shape grain requires AVX2")
def test_shared_pose_trajectory_analytic_shapes_use_the_same_pose():
    data = _inputs(same_query=True)
    prepared = _prepare(data, "analytic_simd", "trajectory")
    scores, stats = _score(data, prepared, grain=RETAINED_COLOR_GRAIN,
                           backend="analytic_simd", shape_grain="trajectory")
    assert np.isfinite(scores).all()
    assert stats["shared_pose_preparations"] == 3 * (2 * 4 + 4)


def test_default_combo_selects_shared_pose_and_counts_timer_only_preparations():
    data = _inputs()
    q, qt, qn, library, counts, widths, heights = data
    query_rows = np.array([0, 1, 0], np.int32)
    library_rows = np.array([0, 1, 2], np.int32)
    common = dict(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        pair_query_row=query_rows, pair_library_row=library_rows,
        interaction_r=widths, interaction_p=heights,
        color=True, color_mode=MODE_COMBO_OPTIMAL, color_grain=0,
        steps=2, start_mode=1, mixing=0.5)
    default_scores, default_stats = cpu.score_pairs(
        **common, with_census=True)
    prepared_scores, prepared_stats = cpu.score_pairs(
        **common, with_census=True,
        pose_contract="shared_fp32_reduced_v1",
        prepared_shared_pose_queries=_prepare(data),
        use_color_invariant_plan=False,
        reuse_exact_final_scratch=False)
    np.testing.assert_array_equal(default_scores, prepared_scores)
    assert default_stats["pose_contract"] == "shared_fp32_reduced_v1"
    assert default_stats["shared_pose_preparations"] == 36
    assert default_stats["color_per_start_final_evaluations"] == 12
    assert default_stats["color_winner_rescore_evaluations"] == 0
    assert default_stats["color_invariant_plan_enabled"] is False
    assert default_stats["exact_final_scratch_enabled"] is False
    assert prepared_stats["pose_contract"] == default_stats["pose_contract"]

    _, timer_stats = cpu.score_pairs(**common, with_phase_timers=True)
    assert timer_stats["shared_pose_preparations"] == 36
    assert timer_stats["exact_final_rescore"] == 0.0


def test_direct_native_default_auto_prepares_shared_queries():
    data = _inputs()
    q, qt, qn, library, counts, widths, heights = data
    query_rows = np.array([0, 1, 0], np.int32)
    library_rows = np.array([0, 1, 2], np.int32)
    native_args = (q, qt, qn, library, counts, query_rows, library_rows,
                   widths, heights, 1, np.float32(0.1), np.float32(0.1),
                   1, np.float32(0.5), True, False)

    # Omitted pose_contract and omitted shared-query handle must select the
    # ordinary shared path at the native boundary too, including scalar grain.
    scalar_scores, scalar_stats = cpu.extension().optimize_scalar_combo(
        *native_args, color_grain=0)
    assert scalar_stats["pose_contract"] == "shared_fp32_reduced_v1"
    assert scalar_stats["shared_pose_preparations"] == 3 * (4 + 4)
    assert np.isfinite(scalar_scores).all()

    # Native auto-preparation must also reuse a caller-supplied shape handle
    # without requiring a second composite shared-query handle.
    shape_handle = cpu.prepare_combo_shape_evaluators(
        q, qn, shape_backend="scalar_analytic", shape_grain="scalar",
        field_spacing_angstrom=0.0)
    batch_scores, batch_stats = cpu.extension().optimize_scalar_combo(
        *native_args, color_grain=RETAINED_COLOR_GRAIN,
        prepared_shape_evaluators=shape_handle)
    assert batch_stats["pose_contract"] == "shared_fp32_reduced_v1"
    assert batch_stats["shared_pose_preparations"] == 3 * (4 + 4)
    assert batch_stats["shape_query_preparation_in_call"] is False
    assert np.isfinite(batch_scores).all()


def test_shared_pose_adapts_field_grains():
    data = _inputs()
    q, qt, qn, library, counts, widths, heights = data
    prepared = _prepare(data)
    kwargs = dict(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        pair_query_row=np.array([0], np.int32),
        pair_library_row=np.array([0], np.int32),
        interaction_r=widths, interaction_p=heights,
        color=True, color_mode=MODE_COMBO_OPTIMAL, steps=1,
        pose_contract="shared_fp32_reduced_v1",
        prepared_shared_pose_queries=prepared)

    field_codes = np.array([0, 1, 0], np.int32)
    fields = cpu.prepare_color_fields(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        interaction_r=widths, interaction_p=heights,
        backend_codes=field_codes)
    for grain in (11, 16):
        if not cpu.color_build_stamp(grain).get("color_runtime_available", True):
            with pytest.raises(ValueError, match="unavailable"):
                cpu.score_pairs(**kwargs, color_grain=grain,
                                color_field_backends=field_codes,
                                prepared_color_fields=fields, with_census=True)
            continue
        field_scores, field_stats = cpu.score_pairs(
            **kwargs, color_grain=grain, color_field_backends=field_codes,
            prepared_color_fields=fields, with_census=True)
        assert np.isfinite(field_scores).all()
        if IS_ARM:
            assert field_stats["color_field_samples"] == 0
            assert field_stats["color_start_lane_scalar_fallback_pairs"] > 0
        else:
            assert field_stats["color_field_samples"] > 0


def test_shared_query_handle_rejects_mutated_source_query():
    data = _inputs()
    q, qt, qn, library, counts, widths, heights = data
    prepared = _prepare(data)
    changed = q.copy()
    changed[0, 1, 0] += 0.01
    with pytest.raises(ValueError, match="do not match"):
        cpu.score_pairs(
            query_coords=changed, query_types=qt, query_n_atoms=qn,
            library_coords=library, library_counts=counts,
            pair_query_row=np.array([0], np.int32),
            pair_library_row=np.array([0], np.int32),
            interaction_r=widths, interaction_p=heights,
            color=True, color_mode=MODE_COMBO_OPTIMAL, color_grain=RETAINED_COLOR_GRAIN,
            steps=1, pose_contract="shared_fp32_reduced_v1",
            prepared_shared_pose_queries=prepared)

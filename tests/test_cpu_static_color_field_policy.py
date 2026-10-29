"""Static per-query color-plane policy, handles and byte guards."""

from dataclasses import replace

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.backend.color_field_policy import (
    CPUColorFieldPolicy, ColorPlaneFieldCalibration,
    estimate_color_field_storage,
)
from roshambo3.backend.spec import ColorPlacement
from roshambo3.engine import Engine, ScreenConfig
from roshambo3.layout import counts_from_types
from roshambo3.reference.color import MODE_COMBO_OPTIMAL
from tests.codec_fixtures import FEATURES_ENUM, make_codec_fixture
from tests.test_cpu_worker_owned_scan import _query_set, _source


def _calibration(*, first=2, code=1, build_fixed=0.0,
                 source_hash="a" * 64):
    return ColorPlaneFieldCalibration(
        backend_code=code,
        first_winning_query_features=first,
        analytic_ns_per_interaction=100.0,
        field_ns_per_target_feature=1.0,
        build_fixed_ns=build_fixed,
        build_ns_per_node=0.001,
        build_ns_per_source=0.1,
        reference_library_conformers=100,
        max_plane_bytes=8 << 20,
        calibration_id="T217 synthetic unit-test row",
        source_sha256=source_hash,
        measurement_scope="synthetic test; not production calibration",
    )


def _synthetic_queries():
    # Real atom at slot 0, then sorted color features. Query row 0 is dense in
    # type 1; row 1 is dense in type 2. Coordinates have a compact field shape.
    coords = np.zeros((2, 4, 3), dtype=np.float32)
    coords[0, 1:] = [[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]]
    coords[1, 1:] = [[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]]
    types = np.array([[0, 1, 1, 2], [0, 1, 2, 2]], dtype=np.int32)
    n_real = np.array([1, 1], dtype=np.int32)
    # Canonical active matrix: all R entries are one, active P is identity;
    # reserved type 0 has no self-interaction.
    r = np.ones((3, 3), dtype=np.float32)
    p = np.eye(3, dtype=np.float32)
    p[0, 0] = 0.0
    return coords, types, n_real, r, p


def test_static_policy_selects_only_dense_amortizing_query_planes():
    coords, types, n_real, r, p = _synthetic_queries()
    counts = counts_from_types(types, n_real, 3,
                               require_contiguous_from_n_real=True)
    policy = CPUColorFieldPolicy(
        calibrations=(_calibration(first=2),),
        evaluations_per_conformer=100,
        max_retained_bytes=4 << 20,
        max_build_peak_bytes=8 << 20,
    )
    codes, info = policy.select(
        query_coords=coords, query_types=types, query_n_atoms=n_real,
        interaction_r=r, interaction_p=p,
        library_target_counts=np.array([0.0, 5.0, 5.0]),
        library_conformers=1000, spacing_angstrom=0.5,
        padding_angstrom=4.0, available_build_bytes=8 << 20)

    np.testing.assert_array_equal(codes, [[0, 1, 0], [0, 0, 1]])
    assert info["selection_reason"] == "calibrated_static_selection"
    assert [(item["query_row"], item["target_type"])
            for item in info["selected_planes"]] == [(0, 1), (1, 2)]
    assert info["field_bytes_estimated"] > 0
    assert info["retained_bytes_estimated"] >= info["field_bytes_estimated"]
    assert info["field_build_peak_bytes_estimated"] >= \
        info["retained_bytes_estimated"]
    assert not codes.flags.writeable
    assert counts[0, 1] == 2 and counts[1, 2] == 2


def test_policy_is_analytic_for_general_matrices_unamortized_or_over_budget():
    coords, types, n_real, r, p = _synthetic_queries()
    policy = CPUColorFieldPolicy(
        calibrations=(_calibration(first=1, build_fixed=1.0e12),),
        evaluations_per_conformer=1,
        max_retained_bytes=4 << 20,
        max_build_peak_bytes=8 << 20,
    )
    general_r = r.copy()
    general_r[1, 2] = 1.25
    general_r[2, 1] = 0.75
    general_p = p.copy()
    general_p[1, 2] = 0.35
    general_p[2, 1] = 0.20
    codes, info = policy.select(
        query_coords=coords, query_types=types, query_n_atoms=n_real,
        interaction_r=general_r, interaction_p=general_p,
        library_target_counts=np.array([0.0, 5.0, 5.0]),
        library_conformers=1000, available_build_bytes=8 << 20)
    assert not np.any(codes)
    assert info["general_matrix_analytic_fallback"] is True

    tiny = CPUColorFieldPolicy(
        calibrations=(_calibration(first=1),),
        evaluations_per_conformer=100,
        max_retained_bytes=1,
        max_build_peak_bytes=1,
    )
    codes, info = tiny.select(
        query_coords=coords, query_types=types, query_n_atoms=n_real,
        interaction_r=r, interaction_p=p,
        library_target_counts=np.array([0.0, 5.0, 5.0]),
        library_conformers=1000, available_build_bytes=1)
    assert not np.any(codes)
    assert info["byte_guarded_planes"] > 0

    no_calibration = CPUColorFieldPolicy()
    codes, info = no_calibration.select(
        query_coords=coords, query_types=types, query_n_atoms=n_real,
        interaction_r=r, interaction_p=p,
        library_target_counts=np.array([0.0, 5.0, 5.0]),
        library_conformers=1000)
    assert not np.any(codes)
    assert info["selection_reason"] == "no_calibrated_rows_analytic_default"


@pytest.mark.parametrize("matrix", ["R", "P"])
def test_policy_refuses_near_canonical_active_matrix(matrix):
    coords, types, n_real, r, p = _synthetic_queries()
    if matrix == "R":
        r[1, 2] = np.nextafter(np.float32(1.0), np.float32(2.0))
    else:
        p[1, 2] = np.finfo(np.float32).eps
    policy = CPUColorFieldPolicy(
        calibrations=(_calibration(first=1),),
        evaluations_per_conformer=100,
        max_retained_bytes=4 << 20,
        max_build_peak_bytes=8 << 20)
    codes, info = policy.select(
        query_coords=coords, query_types=types, query_n_atoms=n_real,
        interaction_r=r, interaction_p=p,
        library_target_counts=np.array([0.0, 5.0, 5.0]),
        library_conformers=1000)
    assert not np.any(codes)
    assert info["general_matrix_analytic_fallback"] is True
    assert info["selection_reason"] == "general_matrix_analytic"


def test_policy_cost_table_and_per_query_backend_matrices_are_bound():
    coords, types, n_real, r, p = _synthetic_queries()
    first = _calibration(code=1, source_hash="b" * 64)
    policy = CPUColorFieldPolicy(
        calibrations=(first,), evaluations_per_conformer=100,
        max_retained_bytes=4 << 20, max_build_peak_bytes=8 << 20)
    changed = replace(policy, evaluations_per_conformer=101)
    assert policy.cache_key() != changed.cache_key()

    codes = np.array([[0, 1, 0], [0, 0, 3]], dtype=np.int32)
    estimate = estimate_color_field_storage(
        codes, coords, types, n_real, r, p)
    assert len(estimate["selected_planes"]) == 2
    assert estimate["field_bytes_estimated"] > 0
    with pytest.raises(ValueError, match=r"\[n_features\].*\[n_query"):
        estimate_color_field_storage(codes[:, :2], coords, types, n_real, r, p)


def test_per_query_handle_matches_dynamic_build_and_exact_final_rescore():
    query_coords = np.zeros((2, 4, 4), dtype=np.float32)
    query_coords[:, 0, 3] = 1.0
    query_coords[0, 1:, :3] = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
                               [0.0, 0.2, 0.0]]
    query_coords[1, 1:, :3] = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
                               [0.0, 0.2, 0.0]]
    query_coords[:, 1:, 3] = 1.0
    query_types = np.array([[0, 1, 1, 2], [0, 1, 2, 2]], dtype=np.int32)
    query_n_atoms = np.array([1, 1], dtype=np.int32)
    interaction_r = np.ones((3, 3), dtype=np.float32)
    interaction_p = np.eye(3, dtype=np.float32)
    interaction_p[0, 0] = 0.0
    library_coords = np.zeros((2, 4, 4), dtype=np.float32)
    library_coords[:, 0, 3] = 1.0
    library_coords[0, 1:, :3] = [[0.1, 0.0, 0.0], [0.3, 0.0, 0.0],
                                 [0.0, 0.1, 0.0]]
    library_coords[1, 1:, :3] = [[0.2, 0.0, 0.0], [0.0, 0.1, 0.0],
                                 [0.0, 0.3, 0.0]]
    library_coords[:, 1:, 3] = 1.0
    library_counts = np.array([[1, 2, 1], [1, 1, 2]], dtype=np.uint8)
    pair_rows = np.arange(2, dtype=np.int32)
    codes = np.array([[0, 1, 0], [0, 0, 3]], dtype=np.int32)
    prepared = cpu.prepare_color_fields(
        query_coords=query_coords, query_types=query_types,
        query_n_atoms=query_n_atoms, interaction_r=interaction_r,
        interaction_p=interaction_p, backend_codes=codes)
    assert prepared.backend_codes_by_query == codes.tolist()

    options = dict(
        color=True, color_mode=MODE_COMBO_OPTIMAL, color_grain=16,
        color_field_backends=codes, mixing=0.5, steps=1,
        lr_q=1e-3, lr_t=0.05, start_mode=1,
        budget_override="T217 fixed per-query handle correctness check",
        with_census=True)
    dynamic, dynamic_stats = cpu.score_pairs(
        **options, query_coords=query_coords, query_types=query_types,
        query_n_atoms=query_n_atoms, library_coords=library_coords,
        library_counts=library_counts, pair_query_row=pair_rows,
        pair_library_row=pair_rows, interaction_r=interaction_r,
        interaction_p=interaction_p)
    cached, cached_stats = cpu.score_pairs(
        **options, prepared_color_fields=prepared,
        query_coords=query_coords, query_types=query_types,
        query_n_atoms=query_n_atoms, library_coords=library_coords,
        library_counts=library_counts, pair_query_row=pair_rows,
        pair_library_row=pair_rows, interaction_r=interaction_r,
        interaction_p=interaction_p)
    np.testing.assert_array_equal(cached, dynamic)
    assert dynamic_stats["color_field_backend_codes_by_query"] == codes.tolist()
    assert cached_stats["color_field_prepared_handle_used"] is True
    assert cached_stats["color_field_build_seconds"] == 0.0
    assert cached_stats["color_field_pack_seconds"] == 0.0

    # The legacy physical-FP64 rescorer reconstructs a different rounded
    # pose from the serialized quaternion. It is not a bit-identity oracle for
    # shared_fp32_reduced_v1. Dynamic/cached identity is checked above; check
    # score meaning independently using the contract's input-rounding budget.
    from tests.color_score_oracle import assert_reported_color_scores
    assert_reported_color_scores(cached, query_coords, query_types,
        query_n_atoms, library_coords, library_counts, pair_rows, pair_rows,
        interaction_r, interaction_p)
    assert np.isfinite(cached).all()

    # Explicit legacy FP16 code 2 is still accepted by the per-query binding
    # API, but it is not emitted by the FP32-only policy or qualified by
    # its calibration/timing claims.
    legacy_codes = codes.copy()
    legacy_codes[0, 1] = 2
    legacy_prepared = cpu.prepare_color_fields(
        query_coords=query_coords, query_types=query_types,
        query_n_atoms=query_n_atoms, interaction_r=interaction_r,
        interaction_p=interaction_p, backend_codes=legacy_codes)
    legacy_options = dict(options, color_field_backends=legacy_codes)
    legacy_dynamic, _ = cpu.score_pairs(
        **legacy_options, query_coords=query_coords, query_types=query_types,
        query_n_atoms=query_n_atoms, library_coords=library_coords,
        library_counts=library_counts, pair_query_row=pair_rows,
        pair_library_row=pair_rows, interaction_r=interaction_r,
        interaction_p=interaction_p)
    legacy_cached, _ = cpu.score_pairs(
        **legacy_options, prepared_color_fields=legacy_prepared,
        query_coords=query_coords, query_types=query_types,
        query_n_atoms=query_n_atoms, library_coords=library_coords,
        library_counts=library_counts, pair_query_row=pair_rows,
        pair_library_row=pair_rows, interaction_r=interaction_r,
        interaction_p=interaction_p)
    np.testing.assert_array_equal(legacy_cached, legacy_dynamic)

    # Grain 16 preflights each target-type/pair lane. Keep the real atom in
    # place for the standard start, but move every library color point well
    # outside the query-built field. That lane's whole plane must take the
    # existing analytic SIMD fallback and match the analytic-only run exactly.
    far_library_coords = library_coords.copy()
    far_library_coords[:, 1:, :3] += np.float32(100.0)
    far_field, far_stats = cpu.score_pairs(
        **options, prepared_color_fields=prepared,
        query_coords=query_coords, query_types=query_types,
        query_n_atoms=query_n_atoms, library_coords=far_library_coords,
        library_counts=library_counts, pair_query_row=pair_rows,
        pair_library_row=pair_rows, interaction_r=interaction_r,
        interaction_p=interaction_p)
    far_analytic, _ = cpu.score_pairs(
        **dict(options, color_field_backends=np.zeros_like(codes)),
        query_coords=query_coords, query_types=query_types,
        query_n_atoms=query_n_atoms, library_coords=far_library_coords,
        library_counts=library_counts, pair_query_row=pair_rows,
        pair_library_row=pair_rows, interaction_r=interaction_r,
        interaction_p=interaction_p)
    from cpu_test_capabilities import IS_ARM
    if IS_ARM:
        assert far_stats["color_field_interpolated_samples"] == 0
        assert far_stats["color_field_out_of_grid_fallback_samples"] == 0
    else:
        assert far_stats["color_field_out_of_grid_fallback_samples"] > 0
    if IS_ARM:
        assert far_stats["color_field_analytic_fallback_terms"] == 0
    else:
        assert far_stats["color_field_analytic_fallback_terms"] > 0
    np.testing.assert_array_equal(far_field, far_analytic)


def test_worker_engine_reuses_native_per_query_handle_and_invalidates_bindings(
        tmp_path, monkeypatch):
    pytest.importorskip("roshambo3._roshambo3_cpu")
    fx = make_codec_fixture(tmp_path / "t217", color_layout="merged")
    source = _source(fx.corpus)
    queries = _query_set(fx, row_indices=(0, 4))
    nf = int(queries.interaction_r.shape[0])
    diagonal = np.arange(nf)
    queries.interaction_r[:] = 1.0
    queries.interaction_p[:] = 0.0
    queries.interaction_p[diagonal[1:], diagonal[1:]] = 1.0
    queries.interaction_p[0, 0] = 0.0
    qcounts = counts_from_types(
        queries.types, queries.n_real, nf,
        require_contiguous_from_n_real=True)
    codes = np.zeros((queries.coords.shape[0], nf), dtype=np.int32)
    for row in range(queries.coords.shape[0]):
        present = np.flatnonzero(qcounts[row, 1:]) + 1
        if present.size:
            codes[row, int(present[0])] = 1 if row == 0 else 3
    assert np.any(codes[0] != codes[1])

    config = ScreenConfig(color=True, color_placement=ColorPlacement.IN_GRADIENT,
                          mixing=0.5, steps=7, start_mode=1)
    engine = Engine(backend="cpu", pinned=False)
    score_handles = []
    native_score_pairs = cpu.score_pairs

    def capture_score_handle(*args, **kwargs):
        score_handles.append((kwargs["prepared_color_fields"],
                              np.asarray(kwargs["color_field_backends"]).copy()))
        return native_score_pairs(*args, **kwargs)

    monkeypatch.setattr(cpu, "score_pairs", capture_score_handle)
    options = dict(workers=1, k=2, color_grain=16,
                   color_field_backends=codes,
                   memory_budget_bytes=64 << 20,
                   topk_state_budget_bytes=1 << 20)
    first, _ = engine.run_worker_owned_cpu(source, queries, config, **options)
    first_handle = engine._worker_query_cache[3]
    assert first_handle is not None
    assert first_handle.backend_codes_by_query == codes.tolist()
    assert first.cpu_scan["color_field_codes_by_query"] == codes.tolist()
    assert first.cpu_scan["color_field_bytes_actual"] == first_handle.field_bytes
    assert first.cpu_scan["color_field_handle_reused"] is False
    assert first.cpu_scan["color_field_builds_this_call"] == 1
    assert first.cpu_scan["memory_peak_estimate_bytes"] <= 64 << 20
    assert first.cpu_scan["color_field_retained_bytes_actual"] == \
        first_handle.retained_bytes
    assert first.cpu_scan["query_context_retained_bytes"] == \
        engine._worker_query_cache[2]
    assert first.cpu_scan["query_context_retained_bytes"] >= \
        first_handle.retained_bytes
    block_count = len(source.cpu_block_descriptors())
    assert block_count > 1
    assert len(score_handles) == block_count
    assert all(handle is first_handle and np.array_equal(selected, codes)
               for handle, selected in score_handles)
    assert engine._worker_query_cache[0][-2:] == (0.5, 4.0)

    score_handles.clear()
    second, _ = engine.run_worker_owned_cpu(source, queries, config, **options)
    assert engine._worker_query_cache[3] is first_handle
    assert second.cpu_scan["color_field_handle_reused"] is True
    assert second.cpu_scan["color_field_build_wall_s"] == 0.0
    assert second.cpu_scan["color_field_builds_this_call"] == 0
    assert len(score_handles) == block_count
    assert all(handle is first_handle and np.array_equal(selected, codes)
               for handle, selected in score_handles)

    changed = codes.copy()
    chosen = np.argwhere(changed[:, 1:] != 0)[0]
    changed[int(chosen[0]), int(chosen[1]) + 1] = 3
    third, _ = engine.run_worker_owned_cpu(
        source, queries, config, **dict(options, color_field_backends=changed))
    assert third.cpu_scan["query_context_cache_invalidated"] is True
    matrix_handle = engine._worker_query_cache[3]
    assert matrix_handle is not first_handle
    assert third.cpu_scan["color_field_builds_this_call"] == 1
    assert all(handle is matrix_handle and np.array_equal(selected, changed)
               for handle, selected in score_handles[-block_count:])

    queries.coords[0, 0, 0] += np.float32(0.125)
    fourth, _ = engine.run_worker_owned_cpu(
        source, queries, config,
        **dict(options, color_field_backends=changed))
    query_handle = engine._worker_query_cache[3]
    assert fourth.cpu_scan["query_context_cache_invalidated"] is True
    assert query_handle is not matrix_handle
    assert fourth.cpu_scan["color_field_builds_this_call"] == 1

    # The real policy path selects fixed canonical planes before the block
    # loop, builds one native handle, and reuses that handle for every block
    # and warm Engine call. Calibration values are synthetic test inputs, not
    # a production cutoff claim.
    policy = CPUColorFieldPolicy(
        calibrations=(_calibration(first=1),),
        evaluations_per_conformer=10_000,
        max_retained_bytes=32 << 20,
        max_build_peak_bytes=32 << 20)
    alternate_policy = replace(
        policy, evaluations_per_conformer=10_001)
    policy_selections = []
    native_policy_select = CPUColorFieldPolicy.select

    def count_policy_selection(self, *args, **kwargs):
        policy_selections.append(self.cache_key())
        return native_policy_select(self, *args, **kwargs)

    monkeypatch.setattr(CPUColorFieldPolicy, "select", count_policy_selection)
    score_handles.clear()
    first_policy, _ = engine.run_worker_owned_cpu(
        source, queries, config,
        **dict(options, color_field_backends=None,
               color_field_policy=policy))
    policy_codes = np.asarray(
        first_policy.cpu_scan["color_field_codes_by_query"], dtype=np.int32)
    assert np.any(policy_codes[:, 1:] != 0)
    assert first_policy.cpu_scan["color_field_selected_planes"]
    policy_handle = engine._worker_query_cache[3]
    assert policy_handle.backend_codes_by_query == policy_codes.tolist()
    assert first_policy.cpu_scan["color_field_builds_this_call"] == 1
    assert len(score_handles) == block_count
    assert all(handle is policy_handle and np.array_equal(selected, policy_codes)
               for handle, selected in score_handles)
    policy_key = engine._worker_query_cache[0][-4]
    score_handles.clear()
    second_policy, _ = engine.run_worker_owned_cpu(
        source, queries, config,
        **dict(options, color_field_backends=None,
               color_field_policy=policy))
    assert first_policy.cpu_scan["query_context_cache_invalidated"] is True
    assert second_policy.cpu_scan["query_context_cache_invalidated"] is False
    assert engine._worker_query_cache[3] is policy_handle
    assert second_policy.cpu_scan["color_field_handle_reused"] is True
    assert second_policy.cpu_scan["color_field_builds_this_call"] == 0
    assert len(score_handles) == block_count
    assert all(handle is policy_handle and np.array_equal(selected, policy_codes)
               for handle, selected in score_handles)
    third_policy, _ = engine.run_worker_owned_cpu(
        source, queries, config,
        **dict(options, color_field_backends=None,
               color_field_policy=alternate_policy))
    assert third_policy.cpu_scan["query_context_cache_invalidated"] is True
    assert engine._worker_query_cache[0][-4] != policy_key
    alternate_handle = engine._worker_query_cache[3]
    assert alternate_handle is not policy_handle
    assert third_policy.cpu_scan["color_field_builds_this_call"] == 1
    alternate_codes = np.asarray(
        third_policy.cpu_scan["color_field_codes_by_query"], dtype=np.int32)
    assert alternate_handle.backend_codes_by_query == alternate_codes.tolist()
    assert all(handle is alternate_handle and
               np.array_equal(selected, alternate_codes)
               for handle, selected in score_handles[-block_count:])
    assert policy_selections == [policy.cache_key(), policy.cache_key(),
                                 alternate_policy.cache_key()]
    engine.close()


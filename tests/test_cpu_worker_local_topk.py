"""Exact worker-local mode-3 top-K scorer integration."""

import inspect

import numpy as np
import pytest

from roshambo3.backend import cpu
from cpu_test_capabilities import IS_ARM, RETAINED_COLOR_GRAIN
from roshambo3.reference.color import MODE_COMBO_OPTIMAL

from test_cpu_color_start_lanes import _population


cpu.extension()  # This test requires the built CPU extension.


def _panel():
    # A repeated sparse pair map forces duplicate callbacks while preserving
    # the original non-identity pair order. The middle query intentionally has
    # no rows; each molecule is represented by multiple library conformers.
    n_library = 65
    kw = _population(n_library, "general")
    library_order = np.arange(n_library - 1, -1, -1, dtype=np.int32)
    query_order = np.where(np.arange(n_library) % 2 == 0, 0, 2).astype(np.int32)
    kw["pair_library_row"] = np.tile(library_order, 2)
    kw["pair_query_row"] = np.tile(query_order, 2)
    gids = (1009 + 37 * np.arange(n_library)).astype(np.int64)
    molecule_ids = (np.arange(n_library) % 13).astype(np.int64)
    return kw, gids, molecule_ids


def _options():
    return dict(
        color=True,
        color_mode=MODE_COMBO_OPTIMAL,
        color_grain=RETAINED_COLOR_GRAIN,
        mixing=0.45,
        steps=2,
        start_mode=1,
    )


def _dense_oracle(dense, pair_query, pair_library, gids, molecule_ids,
                  query_count, k, rank_column, level):
    results = []
    for query_row in range(query_count):
        best_by_gid = {}
        for output_row, (query, library) in enumerate(
                zip(pair_query, pair_library)):
            if int(query) != query_row:
                continue
            gid = int(gids[int(library)])
            candidate = dense[output_row]
            previous = best_by_gid.get(gid)
            if previous is not None:
                np.testing.assert_array_equal(candidate, previous[0])
            else:
                best_by_gid[gid] = (candidate, int(molecule_ids[int(library)]))

        candidates = [
            (gid, row, molid)
            for gid, (row, molid) in best_by_gid.items()
        ]
        if level == "molecule":
            best_by_molecule = {}
            for gid, row, molid in candidates:
                current = best_by_molecule.get(molid)
                if (current is None or row[rank_column] > current[1][rank_column]
                        or (row[rank_column] == current[1][rank_column]
                            and gid < current[0])):
                    best_by_molecule[molid] = (gid, row, molid)
            candidates = list(best_by_molecule.values())

        candidates.sort(key=lambda item: (-float(item[1][rank_column]), item[0]))
        selected = candidates[:k]
        scores = (np.stack([row for _, row, _ in selected]).astype(np.float32)
                  if selected else np.empty((0, dense.shape[1]), np.float32))
        selected_gids = np.asarray([gid for gid, _, _ in selected], np.int64)
        selected_molecules = np.asarray([mol for _, _, mol in selected], np.int64)
        results.append((scores, selected_gids, selected_molecules))
    return results


def _assert_result(actual, expected, *, has_molecule_ids):
    assert len(actual) == len(expected)
    for actual_query, expected_query in zip(actual, expected):
        if has_molecule_ids:
            scores, gids, molecule_ids = actual_query
        else:
            scores, gids = actual_query
        expected_scores, expected_gids, expected_molecules = expected_query
        np.testing.assert_array_equal(scores, expected_scores)
        np.testing.assert_array_equal(gids, expected_gids)
        if has_molecule_ids:
            np.testing.assert_array_equal(molecule_ids, expected_molecules)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("level", ["conformer", "molecule"])
def test_dense_observer_and_sink_only_match_exact_dense_reduction(threads, level):
    kw, gids, molecule_ids = _panel()
    options = _options()
    k = 7
    config = cpu.CPUWorkerLocalTopKConfig(
        k=k, rank_column=0, level=level)
    executor_context = (
        cpu.CPUComboExecutor(2, claim_chunk_rows=128)
        if threads == 2 else None)

    def score(**extra):
        call = dict(threads=threads, executor=executor_context,
                    **options, **extra, **kw)
        if extra.get("worker_local_topk") is not None:
            call["library_conformer_ids"] = gids
            call["library_molecule_ids"] = molecule_ids
        return cpu.score_pairs(**call)

    try:
        dense = score()
        expected = _dense_oracle(
            dense, kw["pair_query_row"], kw["pair_library_row"], gids,
            molecule_ids, kw["query_coords"].shape[0], k, 0, level)

        native_sink = cpu.extension().make_worker_local_topk_row_sink(
            threads, int(kw["query_coords"].shape[0]), k, 0, level,
            len(gids), gids, molecule_ids)
        dense_with_sink = score(row_sink=native_sink)
        native_sink.mark_scoring_succeeded()
        dense_sink_results = native_sink.finalize()
        np.testing.assert_array_equal(dense_with_sink, dense)
        _assert_result(dense_sink_results, expected, has_molecule_ids=True)

        sink_only = score(worker_local_topk=config)
        _assert_result(sink_only, expected, has_molecule_ids=True)
        assert sink_only[1][0].shape == (0, 20)
        assert sink_only[1][1].shape == (0,)
    finally:
        if executor_context is not None:
            executor_context.close()


def _assert_production_sink_only_parity(*, threads, color_grain,
                                        shape_backend, shape_grain,
                                        shape_width, shape_spacing,
                                        pair_count):
    kw, gids, _ = _panel()
    base_library_order = np.arange(len(gids) - 1, -1, -1, dtype=np.int32)
    base_query_order = np.where(np.arange(len(gids)) % 2 == 0, 0, 2).astype(
        np.int32)
    repetitions = (pair_count + len(gids) - 1) // len(gids)
    kw["pair_library_row"] = np.tile(base_library_order, repetitions)[:pair_count]
    kw["pair_query_row"] = np.tile(base_query_order, repetitions)[:pair_count]
    # One molecule has 20 conformers; every other library row is its own
    # molecule. The partial maxima for the large molecule cross worker chunks.
    molecule_ids = np.where(np.arange(len(gids)) < 20, 0,
                            np.arange(len(gids)) - 19).astype(np.int64)

    shape_evaluators = cpu.prepare_combo_shape_evaluators(
        kw["query_coords"], kw["query_n_atoms"],
        shape_backend=shape_backend, shape_grain=shape_grain,
        shape_width=shape_width, field_spacing_angstrom=shape_spacing,
        field_padding_angstrom=8.0)
    shared_poses = cpu.prepare_combo_shared_pose_queries(
        kw["query_coords"], kw["query_types"], kw["query_n_atoms"],
        kw["interaction_r"], kw["interaction_p"],
        shape_backend=shape_backend, shape_grain=shape_grain,
        shape_width=shape_width, field_spacing_angstrom=shape_spacing,
        field_padding_angstrom=8.0, shape_evaluators=shape_evaluators)
    executor_context = (
        cpu.CPUComboExecutor(threads, claim_chunk_rows=128)
        if threads > 1 else None)
    common = dict(
        threads=threads, executor=executor_context,
        color=True, color_mode=MODE_COMBO_OPTIMAL,
        color_grain=color_grain, mixing=0.45, steps=2, start_mode=1,
        shape_backend=shape_backend, shape_grain=shape_grain,
        shape_width=shape_width,
        shape_field_spacing_angstrom=shape_spacing,
        shape_field_padding_angstrom=8.0,
        pose_contract="shared_fp32_reduced_v1",
        prepared_shared_pose_queries=shared_poses)

    try:
        dense = cpu.score_pairs(**common, **kw)
        assert dense.shape == (pair_count, 20)
        expected = _dense_oracle(
            dense, kw["pair_query_row"], kw["pair_library_row"], gids,
            molecule_ids, kw["query_coords"].shape[0], 7, 0, "molecule")

        actual = cpu.score_pairs(
            **common,
            worker_local_topk=cpu.CPUWorkerLocalTopKConfig(
                k=7, rank_column=0, level="molecule"),
            library_conformer_ids=gids,
            library_molecule_ids=molecule_ids,
            **kw)
        _assert_result(actual, expected, has_molecule_ids=True)

        if threads == 4:
            claims = executor_context.stats["last_claims_by_worker"]
            assert sum(claims) == 3
            assert len(claims) == threads
            assert all(count >= 0 for count in claims)
            # Dynamic claiming guarantees once-only work, not a minimum
            # number of active workers for three short tasks.
    finally:
        if executor_context is not None:
            executor_context.close()


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("color_grain", [11, 16])
@pytest.mark.parametrize("shape_backend,shape_grain,shape_width,shape_spacing", [
    ("scalar_analytic", "scalar", 1, None),
    ("pipelined_field", "library_atom", 8, 0.5),
])
def test_production_shared_pose_dense_and_sink_only_parity(
        threads, color_grain, shape_backend, shape_grain, shape_width,
        shape_spacing):
    _assert_production_sink_only_parity(
        threads=threads, color_grain=color_grain,
        shape_backend=shape_backend, shape_grain=shape_grain,
        shape_width=shape_width, shape_spacing=shape_spacing,
        pair_count=130)


def test_four_worker_pipelined_field_color16_dense_and_sink_only_parity():
    _assert_production_sink_only_parity(
        threads=4, color_grain=16, shape_backend="pipelined_field",
        shape_grain="library_atom", shape_width=8, shape_spacing=0.5,
        pair_count=257)


def test_conformer_only_results_omit_optional_molecule_id_array():
    kw, gids, _ = _panel()
    scores = cpu.score_pairs(
        threads=1,
        worker_local_topk=cpu.CPUWorkerLocalTopKConfig(k=3),
        library_conformer_ids=gids,
        **_options(), **kw)
    assert all(len(query_result) == 2 for query_result in scores)


@pytest.mark.parametrize("level", ["conformer", "molecule"])
@pytest.mark.parametrize("pair_order", [
    (3, 0, 2, 1, 3, 0, 2, 1),
    (1, 2, 0, 3, 1, 2, 0, 3),
])
def test_equal_scores_use_global_id_tiebreak_not_pair_order(level, pair_order):
    kw = _population(4, "general")
    kw["library_coords"][:] = kw["library_coords"][:1]
    kw["library_counts"][:] = kw["library_counts"][:1]
    kw["pair_library_row"] = np.asarray(pair_order, dtype=np.int32)
    kw["pair_query_row"] = np.zeros(len(pair_order), dtype=np.int32)
    gids = np.asarray([40, 10, 30, 20], dtype=np.int64)
    molecule_ids = np.asarray([7, 7, 9, 9], dtype=np.int64)
    k = 4 if level == "conformer" else 2

    dense = cpu.score_pairs(threads=1, **_options(), **kw)
    np.testing.assert_array_equal(dense[:, 0], np.full(len(pair_order), dense[0, 0]))
    expected = _dense_oracle(
        dense, kw["pair_query_row"], kw["pair_library_row"], gids,
        molecule_ids, kw["query_coords"].shape[0], k, 0, level)

    actual = cpu.score_pairs(
        threads=1,
        worker_local_topk=cpu.CPUWorkerLocalTopKConfig(
            k=k, rank_column=0, level=level),
        library_conformer_ids=gids,
        library_molecule_ids=molecule_ids,
        **_options(), **kw)
    _assert_result(actual, expected, has_molecule_ids=True)
    expected_tie_order = np.asarray([10, 20, 30, 40] if level == "conformer"
                                    else [10, 20], dtype=np.int64)
    np.testing.assert_array_equal(actual[0][1], expected_tie_order)


def test_worker_local_topk_validates_mode_diagnostics_and_budget_preflight():
    kw, gids, molecule_ids = _panel()
    config = cpu.CPUWorkerLocalTopKConfig(k=2, state_budget_bytes=1)
    with pytest.raises(ValueError, match="exceeds the configured budget"):
        cpu.score_pairs(
            threads=1, worker_local_topk=config,
            library_conformer_ids=gids,
            library_molecule_ids=molecule_ids,
            **_options(), **kw)

    with pytest.raises(ValueError, match="cannot be combined with census"):
        cpu.score_pairs(
            threads=1, worker_local_topk=cpu.CPUWorkerLocalTopKConfig(k=2),
            library_conformer_ids=gids, with_census=True,
            **_options(), **kw)

    with pytest.raises(ValueError, match="only for mode-3"):
        cpu.score_pairs(
            threads=1, worker_local_topk=cpu.CPUWorkerLocalTopKConfig(k=2),
            library_conformer_ids=gids, color_mode="shape_only",
            **kw)

    grain12_options = _options()
    grain12_options["color_grain"] = 12
    with pytest.raises(ValueError, match="unsupported CPU color grain"):
        cpu.score_pairs(
            threads=1, worker_local_topk=cpu.CPUWorkerLocalTopKConfig(k=2),
            library_conformer_ids=gids,
            **grain12_options, **kw)


def test_legacy_dense_call_and_keyword_binding_still_work(
        monkeypatch):
    kw = _population(1, "general")
    native = cpu.extension()
    original = native.optimize_scalar_combo
    legacy_kwargs = {**kw, **_options(), "threads": 1,
                     "with_census": False, "with_phase_timers": False,
                     "with_worker_timers": False,
                     "require_worker_affinity": False}
    signature = inspect.signature(cpu.score_pairs)
    bound_python_call = signature.bind(**legacy_kwargs)
    bound_python_call.apply_defaults()
    assert bound_python_call.arguments["worker_local_topk"] is None

    def call_legacy_native_abi(*args):
        # Six optional controls now follow row_sink: dense output, self
        # timing, torque factoring, exp mode, color kernel, and final policy.
        # Omit them all to exercise the older row_sink-ending call shape.
        assert args[-6] is True
        assert args[-5] is False
        return original(*args[:-6])

    monkeypatch.setattr(native, "optimize_scalar_combo",
                        call_legacy_native_abi)
    dense = cpu.score_pairs(**legacy_kwargs)
    assert isinstance(dense, np.ndarray)
    assert dense.shape == (1, 20)


def test_native_sink_only_guard_rejects_pair_ordering_grain_before_dispatch():
    kw = _population(1, "general")
    native = cpu.extension()
    with pytest.raises(ValueError, match="supports retained color grains"):
        native.optimize_scalar_combo(
            kw["query_coords"], kw["query_types"].astype(np.int32),
            kw["query_n_atoms"], kw["library_coords"],
            kw["library_counts"], kw["pair_query_row"],
            kw["pair_library_row"], kw["interaction_r"],
            kw["interaction_p"], 1, 0.0, 0.0, 1, 0.45, False, False,
            color_grain=12, return_dense=False)


def test_exception_after_rows_discarded_sink_without_publishing_results(
        monkeypatch):
    kw, gids, _ = _panel()
    native = cpu.extension()
    original = native.optimize_scalar_combo
    observed_sinks = []

    def score_then_fail(*args):
        sink = args[-7]
        assert args[-6] is False
        assert args[-5] is False
        observed_sinks.append(sink)
        original(*args)
        raise RuntimeError("injected failure after native score completion")

    monkeypatch.setattr(native, "optimize_scalar_combo", score_then_fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        cpu.score_pairs(
            threads=1,
            worker_local_topk=cpu.CPUWorkerLocalTopKConfig(k=3),
            library_conformer_ids=gids,
            **_options(), **kw)
    assert len(observed_sinks) == 1
    assert observed_sinks[0].memory_estimate["peak_bytes"] == 0
    with pytest.raises(RuntimeError, match="only after successful scoring"):
        observed_sinks[0].finalize()


def test_zero_k_and_empty_pair_batch_return_well_shaped_query_results():
    kw, gids, _ = _panel()
    kw["pair_query_row"] = np.empty(0, np.int32)
    kw["pair_library_row"] = np.empty(0, np.int32)
    result = cpu.score_pairs(
        threads=1,
        worker_local_topk=cpu.CPUWorkerLocalTopKConfig(k=0),
        library_conformer_ids=gids,
        **_options(), **kw)
    assert len(result) == kw["query_coords"].shape[0]
    for scores, out_gids in result:
        assert scores.shape == (0, 20)
        assert out_gids.shape == (0,)

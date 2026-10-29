"""End-to-end checks for the opt-in worker-owned CPU R3G scan."""

from __future__ import annotations

import gc
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.backend.spec import ColorPlacement, PoseSearch, UpdateRule
from roshambo3.engine import Engine, R3gSource, ScreenConfig
from roshambo3.engine.query import from_arrays
from roshambo3.numerics import WIRE_QUANT_STEP, rank_order
from roshambo3.reference.color import MODE_COMBO_OPTIMAL
from tests.codec_fixtures import (
    FEATURES_ENUM,
    codec_support_reason,
    make_codec_fixture,
    ENCODING,
)


def _extension():
    ext = cpu.extension()
    assert hasattr(ext, "prepare_cpu_block_coords"), (
        "CPU extension is missing the worker-owned block converter")
    return ext


def _source(corpus):
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    return R3gSource(
        corpus, color=True, color_generator=generator_from_descriptor(corpus.color),
        tile=False, workers=1, prefetch_depth=0)


def _query_set(fx, row_indices=(0,)):
    step = np.float32(WIRE_QUANT_STEP)
    shapes = []
    colors = []
    for row in row_indices:
        n_real = int(fx.n_real[row])
        color_count = np.asarray(fx.color_counts[row], dtype=np.int64)
        n_color = int(color_count.sum())
        shapes.append(
            fx.shape_xyz_i16[row, :, :n_real].T.astype(np.float32) * step)
        raw_color = (
            fx.shape_xyz_i16[row, :, n_real:n_real + n_color]
            if fx.color_layout == "merged" else
            fx.color_xyz_i16[row, :, :n_color])
        color = raw_color.T.astype(np.float32) * step
        color_types = np.repeat(
            np.arange(1, color_count.size + 1, dtype=np.int32), color_count)
        colors.append((color, color_types))
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    matrices = generator_from_descriptor(fx.corpus.color)
    return from_arrays(
        shapes, [f"t207-query-{row}" for row in row_indices],
        color_generator=matrices, color=colors)


def _compact_reference(source, queries, config, color_grain=0, *,
                       shape_backend="scalar_analytic",
                       shape_grain="scalar", shape_width=1,
                       field_spacing_by_group=None, scoring_options=None):
    """Materialize the same CPU-native layout, then run the dense scorer."""
    ext = _extension()
    blocks = []
    counts = []
    gids = []
    mids = []
    row_groups = []
    for descriptor in source.cpu_block_descriptors():
        decoded = source.decode_cpu_block(descriptor)
        block_counts = decoded.result.counts
        coords = decoded.result.coords
        blocks.append(coords)
        counts.append(block_counts)
        gids.append(descriptor.gid_base + np.arange(
            descriptor.n_conf, dtype=np.int64))
        mids.append(decoded.molecule_ids)
        row_groups.append(np.full(
            descriptor.n_conf, descriptor.group_index, dtype=np.int32))
    library_coords = np.ascontiguousarray(np.concatenate(blocks, axis=0))
    library_counts = np.ascontiguousarray(np.concatenate(counts, axis=0))
    global_ids = np.concatenate(gids)
    molecule_ids = np.concatenate(mids)
    group_ids = np.concatenate(row_groups)
    n_rows = int(library_counts.shape[0])
    optimizer = config.resolved_optimizer()
    if field_spacing_by_group is None:
        spacing_by_group = {int(group_id): 0.5
                            for group_id in np.unique(group_ids)}
    else:
        spacing_by_group = dict(field_spacing_by_group)
    row_spacings = np.asarray(
        [spacing_by_group[int(group_id)] for group_id in group_ids],
        dtype=np.float64)
    n_queries = int(queries.coords.shape[0])
    scores = np.empty((n_queries, n_rows, 20), dtype=np.float32)
    for spacing in np.unique(row_spacings):
        rows = np.flatnonzero(row_spacings == spacing)
        pairs_q = np.repeat(np.arange(n_queries, dtype=np.int32), rows.size)
        pairs_t = np.tile(np.arange(rows.size, dtype=np.int32), n_queries)
        evaluators = cpu.prepare_combo_shape_evaluators(
            queries.coords, queries.n_real, shape_backend=shape_backend,
            shape_grain=shape_grain, shape_width=shape_width,
            field_spacing_angstrom=float(spacing),
            field_padding_angstrom=config.field_pad_a)
        shared = cpu.prepare_combo_shared_pose_queries(
            queries.coords, queries.types, queries.n_real,
            queries.interaction_r, queries.interaction_p,
            shape_backend=shape_backend, shape_grain=shape_grain,
            shape_width=shape_width, field_spacing_angstrom=float(spacing),
            field_padding_angstrom=config.field_pad_a,
            shape_evaluators=evaluators)
        scores[:, rows] = np.asarray(cpu.score_pairs(
            query_coords=queries.coords, query_types=queries.types,
            query_n_atoms=queries.n_real,
            library_coords=np.ascontiguousarray(library_coords[rows]),
            library_counts=np.ascontiguousarray(library_counts[rows]),
            pair_query_row=pairs_q, pair_library_row=pairs_t,
            interaction_r=queries.interaction_r,
            interaction_p=queries.interaction_p,
            steps=optimizer["steps"], lr_q=optimizer["lr_q"],
            lr_t=optimizer["lr_t"],
            budget_override=optimizer.get("budget_override"),
            start_mode=config.start_mode, color=True,
            color_mode=MODE_COMBO_OPTIMAL, color_grain=color_grain,
            threads=1, executor=None, mixing=config.mixing,
            shape_backend=shape_backend, shape_grain=shape_grain,
            shape_width=shape_width,
            shape_field_spacing_angstrom=float(spacing),
            shape_field_padding_angstrom=config.field_pad_a,
            prepared_shape_evaluators=evaluators,
            pose_contract="shared_fp32_reduced_v1",
            prepared_shared_pose_queries=shared, **(scoring_options or {}))).reshape(
                n_queries, rows.size, 20)
    if n_queries == 1:
        scores = scores[0]
    return np.asarray(scores), global_ids, molecule_ids


def _reference_topk(scores, gids, mids, *, k, level, rank_column):
    if level == "conformer":
        rows = rank_order(scores[:, rank_column], gids)[:k]
        return scores[rows], gids[rows], mids[rows]
    winners = []
    for mid in np.unique(mids):
        rows = np.flatnonzero(mids == mid)
        winners.append(rows[rank_order(scores[rows, rank_column], gids[rows])[0]])
    winners = np.asarray(winners, dtype=np.int64)
    rows = winners[rank_order(scores[winners, rank_column], gids[winners])[:k]]
    return scores[rows], gids[rows], mids[rows]


def _assert_result_matches(result, expected):
    actual_scores, actual_gids, actual_mids = result[0]
    expected_scores, expected_gids, expected_mids = expected
    np.testing.assert_array_equal(actual_gids, expected_gids)
    np.testing.assert_array_equal(actual_mids, expected_mids)
    np.testing.assert_array_equal(actual_scores, expected_scores)


@pytest.mark.parametrize(
    "codec,shuffle,color_layout",
    [(ENCODING, False, "merged")], ids=["bp256-derived-color"])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("level", ["conformer", "molecule"])
def test_worker_owned_scan_matches_dense_scores_and_topk(
        codec, shuffle, color_layout, workers, level, tmp_path, monkeypatch):
    reason = codec_support_reason(codec)
    assert reason is None, reason
    ext = _extension()
    fx = make_codec_fixture(
        tmp_path / f"{codec}-{shuffle}-{color_layout}", codec=codec,
        shuffle=shuffle, color_layout=color_layout, block_conf=3)
    source = _source(fx.corpus)
    queries = _query_set(fx)
    config = ScreenConfig(
        color=True, color_placement=ColorPlacement.IN_GRADIENT,
        mixing=0.5, steps=25, lr_q=0.1, start_mode=1,
        field_spacing_a=0.5, field_pad_a=8.0)

    # Independently compare every decoded slot against the fixture's derived
    # kernel layout, including padding and the per-row real/color boundary.
    first = source.cpu_block_descriptors()[0]
    decoded = source.decode_cpu_block(first)
    expected = fx.shape_xyz_i16[first.gid_base:first.gid_base + first.n_conf].transpose(0, 2, 1)
    np.testing.assert_array_equal(decoded.result.coords,
                                  expected.astype(np.float32) * np.float32(WIRE_QUANT_STEP))

    dense_scores, gids, mids = _compact_reference(source, queries, config)
    # Synchronize the first two W=2 callbacks after claim. The blocking barrier
    # releases the GIL, proving separate persistent workers own descriptors.
    original_decode = source.decode_cpu_block
    barrier = threading.Barrier(2) if workers > 1 else None
    arrival_lock = threading.Lock()
    arrivals = 0

    def delayed_decode(descriptor):
        nonlocal arrivals
        should_wait = False
        if barrier is not None:
            with arrival_lock:
                if arrivals < 2:
                    arrivals += 1
                    should_wait = True
        if should_wait:
            barrier.wait(timeout=5)
        return original_decode(descriptor)

    monkeypatch.setattr(source, "decode_cpu_block", delayed_decode)
    engine = Engine(backend="cpu", pinned=False)
    stats, result = engine.run_worker_owned_cpu(
        source, queries, config, workers=workers, k=6, rank_column=0,
        level=level, memory_budget_bytes=64 << 20,
        topk_state_budget_bytes=1 << 20)
    _assert_result_matches(
        result, _reference_topk(dense_scores, gids, mids, k=6,
                                level=level, rank_column=0))

    counters = stats.cpu_scan
    assert counters["executor_submissions"] == 1
    assert counters["blocks"] == 3
    assert counters["shape_blocks_decoded"] == 3
    assert counters["color_blocks_decoded"] == 3
    assert counters["rows_decoded"] == 6
    assert counters["scored_rows"] == 6
    assert counters["native_score_calls"] == 3
    assert counters["fp32_layout_conversion_bytes"] > 0
    assert counters["memory_peak_estimate_bytes"] <= counters["memory_budget_bytes"]
    assert counters["memory_estimate_not_hard_rss_cap"] is True
    assert counters["library_row_fp32_conversion_is_copy"] is True
    assert counters["native_self_overlap_fallback_per_block"] is True
    assert counters["native_library_self_overlap_fallback_per_block"] is True
    assert counters["query_self_contexts_built"] == counters["query_contexts_built"]
    assert counters["shape_query_self_evaluations"] == counters["query_contexts_built"]
    assert counters["color_query_self_evaluations"] == counters["query_contexts_built"]
    assert counters["query_context_setup_s"] >= 0
    assert counters["scan_wall_s"] >= 0
    worker_counters = counters["worker_counters"]
    assert len(worker_counters) == workers
    assert sum(worker["blocks"] for worker in worker_counters) == 3
    if workers > 1:
        assert all(worker["blocks"] > 0 for worker in worker_counters)
    if level == "molecule":
        # MID 900 has rows on both sides of the first block boundary.
        mids_out = result[0][2]
        assert len(np.unique(mids_out)) == mids_out.size
        assert set(mids_out.tolist()) == {17, 41, 900}


def test_sparse_selected_group_preserves_source_global_gids(tmp_path):
    _extension()
    fx = make_codec_fixture(tmp_path / "sparse", color_layout="merged")
    source = R3gSource(
        fx.corpus, groups=[1], color=True,
        color_generator=__import__("roshambo3.format.r3g.descriptor",
            fromlist=["generator_from_descriptor"]).generator_from_descriptor(fx.corpus.color),
        tile=False, prefetch_depth=0)
    queries = _query_set(fx)
    config = ScreenConfig(color=True,
                          color_placement=ColorPlacement.IN_GRADIENT,
                          mixing=0.5)
    stats, result = Engine(backend="cpu", pinned=False).run_worker_owned_cpu(
        source, queries, config, workers=2, k=6, memory_budget_bytes=64 << 20,
        topk_state_budget_bytes=1 << 20)
    assert stats.cpu_scan["rows_decoded"] == 2
    np.testing.assert_array_equal(np.sort(result[0][1]), [4, 5])


def test_memory_budget_refuses_before_payload_decode_or_query_preparation(
        tmp_path, monkeypatch):
    _extension()
    fx = make_codec_fixture(tmp_path / "too-small-budget",
                             color_layout="merged")
    source = _source(fx.corpus)
    queries = _query_set(fx)
    config = ScreenConfig(color=True,
                          color_placement=ColorPlacement.IN_GRADIENT,
                          mixing=0.5)
    decoded = []

    def unexpected_decode(descriptor):
        decoded.append(descriptor)
        raise AssertionError("preflight must reject before payload reads")

    def unexpected_query_prep(*args, **kwargs):
        raise AssertionError("preflight must reject before query preparation")

    monkeypatch.setattr(source, "decode_cpu_block", unexpected_decode)
    monkeypatch.setattr(cpu, "prepare_combo_shape_evaluators",
                        unexpected_query_prep)
    with pytest.raises(MemoryError, match="above budget"):
        Engine(backend="cpu", pinned=False).run_worker_owned_cpu(
            source, queries, config, workers=2, k=3,
            memory_budget_bytes=1, topk_state_budget_bytes=1)
    assert decoded == []


def test_retained_field_policy_caches_width_aware_query_contexts(tmp_path):
    _extension()
    fx = make_codec_fixture(tmp_path / "field-widths", color_layout="merged")
    # Simulate the distinct selected corpus buckets used by the retained
    # recipe while keeping this test's real rows, codecs, and sidecars tiny.
    fx.corpus.group_table[0]["L_field_width"] = 32
    fx.corpus.group_table[1]["L_field_width"] = 40
    source = _source(fx.corpus)
    queries = _query_set(fx)
    config = ScreenConfig(
        color=True, color_placement=ColorPlacement.IN_GRADIENT,
        mixing=0.5, steps=25, lr_q=0.1, start_mode=1,
        field_spacing_a=None, field_pad_a=8.0)
    spacing_by_group = {0: 0.5, 1: 0.6}
    dense_scores, gids, mids = _compact_reference(
        source, queries, config, color_grain=16,
        shape_backend="pipelined_field", shape_grain="library_atom",
        shape_width=8, field_spacing_by_group=spacing_by_group)

    stats, result = Engine(backend="cpu", pinned=False).run_worker_owned_cpu(
        source, queries, config, workers=2, k=6, rank_column=0,
        level="conformer", color_grain=16,
        shape_backend="pipelined_field", shape_grain="library_atom",
        shape_width=8, memory_budget_bytes=64 << 20,
        topk_state_budget_bytes=1 << 20)
    _assert_result_matches(
        result, _reference_topk(dense_scores, gids, mids, k=6,
                                level="conformer", rank_column=0))
    counters = stats.cpu_scan
    assert counters["query_contexts_built"] == 2
    assert counters["shape_backend"] == "pipelined_field"
    assert counters["shape_grain"] == "library_atom"
    assert counters["shape_width"] == 8
    assert counters["field_spacing_by_group"] == spacing_by_group
    assert counters["field_width_by_group"] == {0: 32, 1: 40}


def test_multiquery_worker_scan_offers_one_complete_row_per_query(tmp_path):
    _extension()
    fx = make_codec_fixture(tmp_path / "two-queries", color_layout="merged")
    source = _source(fx.corpus)
    queries = _query_set(fx, row_indices=(0, 1))
    config = ScreenConfig(
        color=True, color_placement=ColorPlacement.IN_GRADIENT,
        mixing=0.5, steps=25, lr_q=0.1, start_mode=1,
        field_spacing_a=0.5, field_pad_a=8.0)
    dense_scores, gids, mids = _compact_reference(
        source, queries, config, color_grain=16,
        shape_backend="pipelined_field", shape_grain="library_atom",
        shape_width=8)

    stats, result = Engine(backend="cpu", pinned=False).run_worker_owned_cpu(
        source, queries, config, workers=2, k=3, rank_column=0,
        level="conformer", color_grain=16,
        shape_backend="pipelined_field", shape_grain="library_atom",
        shape_width=8, memory_budget_bytes=64 << 20,
        topk_state_budget_bytes=1 << 20)
    assert len(result) == 2
    for query_row in range(2):
        _assert_result_matches(
            [result[query_row]],
            _reference_topk(dense_scores[query_row], gids, mids, k=3,
                            level="conformer", rank_column=0))
        assert len(result[query_row][1]) == 3
    counters = stats.cpu_scan
    assert counters["blocks"] == 3
    assert counters["rows_decoded"] == 6
    assert counters["scored_rows"] == 12
    assert counters["native_score_calls"] == 3
    assert counters["query_contexts_built"] == 1
    # Q=2 query self terms are prepared once in the shared context rather than
    # recomputed for each of the three block-owned native scorer calls.
    assert counters["query_self_contexts_built"] == 1
    assert counters["shape_query_self_evaluations"] == 2
    assert counters["color_query_self_evaluations"] == 2


@pytest.mark.parametrize(
    "field_spacing,field_padding",
    [(float("nan"), 8.0), (0.5, 0.0), (float("inf"), 8.0)])
def test_retained_field_policy_rejects_invalid_grid_parameters(
        field_spacing, field_padding, tmp_path):
    _extension()
    fx = make_codec_fixture(tmp_path / "bad-field", color_layout="merged")
    source = _source(fx.corpus)
    with pytest.raises(ValueError, match="finite and positive"):
        Engine(backend="cpu", pinned=False).run_worker_owned_cpu(
            source, _query_set(fx),
            ScreenConfig(color=True,
                         color_placement=ColorPlacement.IN_GRADIENT), workers=1,
            shape_backend="pipelined_field", shape_grain="library_atom",
            shape_width=8, field_spacing_angstrom=field_spacing,
            field_padding_angstrom=field_padding,
            memory_budget_bytes=64 << 20,
            topk_state_budget_bytes=1 << 20)


@pytest.mark.parametrize(
    "override,expected",
    [({"pose_search": PoseSearch.ANALYTIC}, "pose_search=FIELD"),
     ({"color_placement": ColorPlacement.AT_POSE,
           "pose_search": PoseSearch.FIELD}, "require retained pipelined fields"),
     ({"update_rule": UpdateRule.ADAGRAD_QADD,
       "pose_search": PoseSearch.FIELD}, "update_rule=ADAGRAD_EXP")])
def test_worker_owned_scan_rejects_unimplemented_optimizer_settings(
        override, expected, tmp_path):
    _extension()
    fx = make_codec_fixture(tmp_path / "unsupported-config",
                             color_layout="merged")
    source = _source(fx.corpus)
    options = {"color": True,
               "color_placement": ColorPlacement.IN_GRADIENT}
    options.update(override)
    config = ScreenConfig(**options)
    with pytest.raises(NotImplementedError, match=expected):
        Engine(backend="cpu", pinned=False).run_worker_owned_cpu(
            source, _query_set(fx), config, workers=1,
            memory_budget_bytes=64 << 20,
            topk_state_budget_bytes=1 << 20)


@pytest.mark.parametrize(
    "level,expected_gids,expected_mids",
    [("conformer", [0, 1, 2], [41, 41, 41]),
     ("molecule", [0, 3, 4], [41, 900, 17])])
def test_exact_rank_ties_use_source_gid_across_worker_shards(
        level, expected_gids, expected_mids, tmp_path, monkeypatch):
    _extension()
    fx = make_codec_fixture(tmp_path / level, color_layout="merged")
    source = _source(fx.corpus)
    queries = _query_set(fx)
    config = ScreenConfig(color=True,
                          color_placement=ColorPlacement.IN_GRADIENT,
                          mixing=0.5)
    dense_scores, gids, mids = _compact_reference(source, queries, config)
    # The query self-overlap is the same for every candidate. Ranking by this
    # column forces a complete tie, so retained GIDs decide both shard-local
    # and final global order.
    np.testing.assert_array_equal(dense_scores[:, 5],
                                  np.repeat(dense_scores[0, 5], len(gids)))
    engine = Engine(backend="cpu", pinned=False)
    original_decode = source.decode_cpu_block
    original_score = engine.backend.score_pairs
    block_context = threading.local()
    first_block_waiting = threading.Event()
    later_block_scored = threading.Event()

    def mark_decode(descriptor):
        block_context.gid_base = int(descriptor.gid_base)
        return original_decode(descriptor)

    def complete_out_of_order(**kwargs):
        gid_base = int(block_context.gid_base)
        if gid_base == 0:
            first_block_waiting.set()
            if not later_block_scored.wait(timeout=15):
                raise AssertionError("later block did not complete first")
        try:
            return original_score(**kwargs)
        finally:
            if gid_base != 0:
                later_block_scored.set()

    monkeypatch.setattr(source, "decode_cpu_block", mark_decode)
    monkeypatch.setattr(engine.backend, "score_pairs", complete_out_of_order)
    stats, result = engine.run_worker_owned_cpu(
        source, queries, config, workers=2, k=3, rank_column=5,
        level=level, memory_budget_bytes=64 << 20,
        topk_state_budget_bytes=1 << 20)
    assert first_block_waiting.is_set()
    assert later_block_scored.is_set()
    np.testing.assert_array_equal(result[0][1], expected_gids)
    np.testing.assert_array_equal(result[0][2], expected_mids)
    assert stats.cpu_scan["blocks"] == 3


def test_empty_final_group_is_refused_by_format3_fixture(tmp_path):
    with pytest.raises(ValueError, match="empty group"):
        make_codec_fixture(tmp_path / "empty", empty_final_group=True)


def test_cancel_and_failures_discard_partial_scan_and_recover(
        tmp_path, monkeypatch):
    ext = _extension()
    fx = make_codec_fixture(tmp_path / "failure", color_layout="merged")
    source = _source(fx.corpus)
    queries = _query_set(fx)
    config = ScreenConfig(color=True,
                          color_placement=ColorPlacement.IN_GRADIENT,
                          mixing=0.5)
    engine = Engine(backend="cpu", pinned=False)

    real_sink_factory = ext.make_worker_local_topk_direct_row_sink
    created_sinks = []

    def capture_sink(*args, **kwargs):
        sink = real_sink_factory(*args, **kwargs)
        created_sinks.append(sink)
        return sink

    def assert_latest_sink_discarded():
        sink = created_sinks[-1]
        assert sink.memory_estimate["peak_bytes"] == 0
        with pytest.raises(RuntimeError, match="only after successful scoring"):
            sink.finalize()

    monkeypatch.setattr(ext, "make_worker_local_topk_direct_row_sink",
                        capture_sink)
    with pytest.raises(InterruptedError, match="canceled"):
        engine.run_worker_owned_cpu(
            source, queries, config, workers=2, k=3,
            memory_budget_bytes=64 << 20, topk_state_budget_bytes=1 << 20,
            canceled=lambda: True)
    assert_latest_sink_discarded()

    # A single worker makes the second cancellation check happen only after
    # one complete descriptor has decoded and scored. The failed call must not
    # return the partial local top-K as a completed search.
    original_decode = source.decode_cpu_block
    decoded_count = 0

    def count_decode(descriptor):
        nonlocal decoded_count
        decoded_count += 1
        return original_decode(descriptor)

    cancel_checks = 0

    def cancel_after_one_block():
        nonlocal cancel_checks
        cancel_checks += 1
        return cancel_checks > 1

    monkeypatch.setattr(source, "decode_cpu_block", count_decode)
    with pytest.raises(InterruptedError, match="canceled"):
        engine.run_worker_owned_cpu(
            source, queries, config, workers=1, k=3,
            memory_budget_bytes=64 << 20, topk_state_budget_bytes=1 << 20,
            canceled=cancel_after_one_block)
    assert decoded_count == 1
    assert cancel_checks == 2
    assert_latest_sink_discarded()
    monkeypatch.setattr(source, "decode_cpu_block", original_decode)

    # Force one worker to be inside decode while the other worker raises. The
    # executor must join the in-flight peer before the scan discards its sink.
    peer_started = threading.Event()
    primary_failed = threading.Event()
    release_peer = threading.Event()
    peer_finished = threading.Event()
    peer_score_finished = threading.Event()
    worker_context = threading.local()
    original_score = engine.backend.score_pairs

    def fail_first_after_peer(descriptor):
        gid_base = int(descriptor.gid_base)
        worker_context.gid_base = gid_base
        if gid_base == 0:
            if not peer_started.wait(timeout=15):
                raise AssertionError("peer never entered descriptor decode")
            primary_failed.set()
            raise RuntimeError("injected asynchronous decode failure")
        if gid_base == 4:
            peer_started.set()
            if not primary_failed.wait(timeout=15):
                raise AssertionError("primary callback did not fail")
            if not release_peer.wait(timeout=15):
                raise AssertionError("test did not release in-flight peer")
            try:
                return original_decode(descriptor)
            finally:
                peer_finished.set()
        return original_decode(descriptor)

    def record_peer_score_completion(**kwargs):
        try:
            return original_score(**kwargs)
        finally:
            if getattr(worker_context, "gid_base", None) == 4:
                peer_score_finished.set()

    monkeypatch.setattr(source, "decode_cpu_block", fail_first_after_peer)
    monkeypatch.setattr(engine.backend, "score_pairs",
                        record_peer_score_completion)
    async_failures = []

    def run_async_failure_scan():
        try:
            engine.run_worker_owned_cpu(
                source, queries, config, workers=2, k=3,
                memory_budget_bytes=64 << 20,
                topk_state_budget_bytes=1 << 20)
        except BaseException as exc:  # surfaced on the test thread below
            async_failures.append(exc)

    scan_thread = threading.Thread(
        target=run_async_failure_scan, name="t207-async-failure-scan")
    scan_thread.start()
    try:
        assert primary_failed.wait(timeout=15)
        assert peer_started.is_set()
        assert not peer_finished.is_set()
        assert scan_thread.is_alive(), (
            "scan returned while the peer decode callback was still blocked")
    finally:
        release_peer.set()
        scan_thread.join(timeout=20)
    assert not scan_thread.is_alive(), "failing scan did not join"
    assert peer_finished.is_set(), "in-flight peer callback did not finish"
    assert peer_score_finished.is_set(), (
        "in-flight peer callback was not joined through scoring")
    assert len(async_failures) == 1
    assert isinstance(async_failures[0], RuntimeError)
    assert str(async_failures[0]) == "injected asynchronous decode failure"
    assert_latest_sink_discarded()
    monkeypatch.setattr(source, "decode_cpu_block", original_decode)

    # Exercise the production bounded read, rather than an obsolete reader's
    # block_bytes method (format-3 descriptors retain metadata, not readers).
    from roshambo3.format.r3g import native
    original_read = native.read_into
    def short_read(fd, offset, length):
        return original_read(fd, offset, length)[:-1]
    monkeypatch.setattr(native, "read_into", short_read)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        engine.run_worker_owned_cpu(
            source, queries, config, workers=2, k=3,
            memory_budget_bytes=64 << 20, topk_state_budget_bytes=1 << 20)
    assert_latest_sink_discarded()
    monkeypatch.setattr(native, "read_into", original_read)

    # A failed/canceled submission leaves the new explicit executor and source
    # usable; successful completion retains a live, finalizable sink.
    stats, result = engine.run_worker_owned_cpu(
        source, queries, config, workers=1, k=3,
        memory_budget_bytes=64 << 20, topk_state_budget_bytes=1 << 20)
    assert stats.cpu_scan["blocks"] == 3
    assert result[0][1].size == 3
    assert created_sinks[-1].memory_estimate["peak_bytes"] > 0


def test_native_decoded_buffers_live_through_scoring_then_release(
        tmp_path, monkeypatch):
    _extension()
    fx = make_codec_fixture(tmp_path / "decoded-lifetime",
                            color_layout="merged")
    source = _source(fx.corpus)
    queries = _query_set(fx)
    config = ScreenConfig(color=True,
                          color_placement=ColorPlacement.IN_GRADIENT,
                          mixing=0.5)
    engine = Engine(backend="cpu", pinned=False)
    original_decode = source.decode_cpu_block
    original_score = engine.backend.score_pairs
    buffer_refs = []
    scoring_entered = threading.Event()
    release_scoring = threading.Event()

    def capture_decode(descriptor):
        decoded = original_decode(descriptor)
        buffer_refs.append((weakref.ref(decoded.result.coords),
                            weakref.ref(decoded.result.counts)))
        return decoded

    first_score = True

    def pause_inside_score(**kwargs):
        nonlocal first_score
        if first_score:
            first_score = False
            scoring_entered.set()
            if not release_scoring.wait(timeout=15):
                raise AssertionError("test did not release in-flight scoring")
        return original_score(**kwargs)

    monkeypatch.setattr(source, "decode_cpu_block", capture_decode)
    monkeypatch.setattr(engine.backend, "score_pairs", pause_inside_score)
    outcome = []
    failures = []

    def run_scan():
        try:
            outcome.append(engine.run_worker_owned_cpu(
                source, queries, config, workers=1, k=3,
                memory_budget_bytes=64 << 20,
                topk_state_budget_bytes=1 << 20))
        except BaseException as exc:  # surfaced on the test thread below
            failures.append(exc)

    scan_thread = threading.Thread(target=run_scan)
    scan_thread.start()
    try:
        assert scoring_entered.wait(timeout=15)
        gc.collect()
        assert len(buffer_refs) == 1
        assert all(reference() is not None for reference in buffer_refs[0])
    finally:
        release_scoring.set()
        scan_thread.join(timeout=20)
    assert not scan_thread.is_alive(), "worker-owned scan did not join"
    assert failures == []
    assert len(outcome) == 1
    assert outcome[0][0].cpu_scan["blocks"] == 3
    gc.collect()
    assert all(reference() is None
               for pair in buffer_refs for reference in pair)


def test_shape_only_scalar_scan_is_rejected_without_changing_decoder_support(tmp_path):
    _extension()
    fx = make_codec_fixture(tmp_path / "shape-only", color_layout="none")
    source = R3gSource(fx.corpus, color=False, tile=False, prefetch_depth=0)
    assert len(source.cpu_block_descriptors()) == 3
    with pytest.raises(NotImplementedError, match="require retained pipelined fields"):
        Engine(backend="cpu", pinned=False).run_worker_owned_cpu(
            source, None, ScreenConfig(color=False), workers=1)

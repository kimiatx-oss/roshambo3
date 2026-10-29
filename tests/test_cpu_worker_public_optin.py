"""Public worker-owned opt-in: translation, refusals and exact tiny parity."""

from dataclasses import fields, replace

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.backend.spec import ColorPlacement, PoseSearch, UpdateRule
from roshambo3.engine import (
    CPUWorkerOptions, DenseScores, Engine, HostTopK, MoleculeTopK,
    R3gSource, ScreenConfig, ScreenStats, TopK, from_arrays,
)
from roshambo3.engine import engine as engine_module
from roshambo3.numerics import WIRE_QUANT_STEP, rank_order
from tests.codec_fixtures import make_codec_fixture
from tests.test_cpu_worker_owned_scan import (
    _compact_reference, _query_set, _reference_topk, _source,
)


def _config(**kwargs):
    return ScreenConfig(
        color=True, mixing=0.5,
        color_placement=ColorPlacement.IN_GRADIENT,
        cpu_worker=CPUWorkerOptions("recompute", workers=1),
        **kwargs)


@pytest.fixture
def inputs(tmp_path):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    return _source(fx.corpus), _query_set(fx, (0, 4))


@pytest.mark.parametrize("sink_type,level", [
    (TopK, "conformer"), (HostTopK, "conformer"),
    (MoleculeTopK, "molecule"),
])
def test_explicit_route_uses_existing_scan_without_sink_state(
        inputs, monkeypatch, sink_type, level):
    source, queries = inputs
    engine = Engine(backend=cpu)
    sink = sink_type(2, rank_col=1)
    rows = np.zeros((1, 20), np.float32)
    gids = np.array([9], np.int64)
    mids = np.array([70], np.int64)
    calls = []

    def scan(source_arg, queries_arg, config_arg, **kwargs):
        assert source_arg is source and queries_arg is queries
        calls.append(kwargs)
        return ScreenStats(), [(rows, gids, mids), (rows, gids)]

    monkeypatch.setattr(engine, "run_worker_owned_cpu", scan)
    monkeypatch.setattr(engine, "_check_cells_for", lambda *a: pytest.fail("CUDA preflight"))
    monkeypatch.setattr(sink, "prepare", lambda *a: pytest.fail("sink state"))
    _, result = engine.run(source, queries, _config(), sink)
    assert calls == [dict(
        workers=1, k=2, rank_column=1, level=level,
        memory_budget_bytes=None, topk_state_budget_bytes=None,
        color_grain=16, shape_backend="pipelined_field",
        shape_grain="library_atom", shape_width=8, canceled=None, progress=None,
        self_overlap_policy="recompute", factor_canonical_torque=True,
        optimizer_exp_mode="rawbits_unclamped", canonical_color_kernel=True,
        final_start_policy="winner", field_storage="fp32", field_interpolation="fp32")]
    assert all(len(item) == 2 for item in result)
    assert result[0][0] is rows and result[0][1] is gids


def test_public_precision_policy_passes_fp32_auto_and_explicit_request(
        inputs, monkeypatch):
    source, queries = inputs
    engine = Engine(backend=cpu)
    monkeypatch.setattr(engine_module.platform, "machine", lambda: "aarch64")
    rows = np.zeros((1, 20), np.float32)
    gids = np.array([9], np.int64)
    calls = []

    def scan(source_arg, queries_arg, config_arg, **kwargs):
        calls.append(kwargs)
        return ScreenStats(), [(rows, gids)]

    monkeypatch.setattr(engine, "run_worker_owned_cpu", scan)
    _, _ = engine.run(source, queries, _config(), TopK(2))
    assert calls[-1]["field_storage"] == "fp32"
    assert calls[-1]["field_interpolation"] == "fp32"

    config = _config()
    optout = replace(config, cpu_worker=replace(
        config.cpu_worker, shape_field_interpolation="fp32"))
    stats, _ = engine.run(source, queries, optout, TopK(2))
    assert calls[-1]["field_storage"] == "fp32"
    assert calls[-1]["field_interpolation"] == "fp32"
    assert stats.cpu_scan["shape_field_interpolation_policy"] == "fp32"


def test_unset_option_preserves_ordinary_default_and_dense_route(inputs, monkeypatch):
    source, queries = inputs
    engine = Engine(backend=cpu)
    calls = []
    monkeypatch.setattr(engine, "_check_cells_for", lambda *a: None)
    monkeypatch.setattr(engine, "_run", lambda s, q, c, sink, *a:
                        calls.append((c, sink)) or (ScreenStats(), "ordinary"))
    for config in (ScreenConfig(), replace(_config(), cpu_worker=None)):
        route_source = R3gSource(
            source.corpus, color=config.color,
            color_generator=source.color_generator if config.color else None,
            tile=False, workers=1, prefetch_depth=0)
        sink = DenseScores()
        assert engine.run(route_source, queries, config, sink)[1] == "ordinary"
        assert calls[-1] == (config, sink)


def test_native_shape_default_still_matches_explicit_dense_ranking(tmp_path):
    fx = make_codec_fixture(tmp_path / "shape", color_layout="none")
    queries = from_arrays(
        [fx.shape_xyz_i16[0, :, :int(fx.n_real[0])].T.astype(np.float32)
         * np.float32(WIRE_QUANT_STEP)], ["shape-query"])
    source = R3gSource(fx.corpus, color=False, tile=False,
                        workers=1, prefetch_depth=0, conformers_per_launch=3)
    engine = Engine(backend=cpu, pinned=False)
    _, dense = engine.run(source, queries, ScreenConfig(), DenseScores())
    _, result = engine.run(source, queries, ScreenConfig(), HostTopK(3))
    order = rank_order(dense.scores[0, :, 0], dense.gids)[:3]
    np.testing.assert_array_equal(result[0][0], dense.scores[0, order])
    np.testing.assert_array_equal(result[0][1], dense.gids[order])


SUPPORTED = {
    "cpu_worker", "color", "mixing", "pose_search", "color_placement",
    "steps", "lr_t", "lr_q", "start_mode", "update_rule", "budget_override",
    "field_spacing_a", "field_pad_a",
}


@pytest.mark.parametrize("name", [f.name for f in fields(ScreenConfig)
                                  if f.name not in SUPPORTED])
def test_each_changed_cuda_setting_refuses_by_name(inputs, name):
    source, queries = inputs
    config = _config()
    value = getattr(config, name)
    setattr(config, name, 99 if value is None else value + 1)
    with pytest.raises(NotImplementedError, match=name):
        Engine(backend=cpu).run(source, queries, config, TopK(2))


@pytest.mark.parametrize("change,match", [
    ({"color": False}, "color layouts differ"),
    ({"start_mode": 0}, "start_mode"),
    ({"start_mode": 1.0}, "start_mode"),
    ({"start_mode": True}, "start_mode"),
    ({"field_pad_a": 4.0}, "field_pad_a"),
    ({"pose_search": PoseSearch.ANALYTIC}, "pose_search"),
    ({"update_rule": UpdateRule(0)}, "update_rule"),
])
def test_unsupported_scoring_contract_refuses(inputs, change, match):
    source, queries = inputs
    with pytest.raises((ValueError, NotImplementedError), match=match):
        Engine(backend=cpu).run(source, queries, replace(_config(), **change), TopK(2))


@pytest.mark.parametrize("policy", [None, "algorithm-design-legacy-prepared", "verified"])
def test_unadmitted_self_policy_refuses_before_scan(inputs, policy):
    source, queries = inputs
    config = _config()
    config.cpu_worker = replace(config.cpu_worker, self_overlap_policy=policy)
    with pytest.raises(ValueError, match="self_overlap_policy"):
        Engine(backend=cpu).run(source, queries, config, TopK(2))


def test_format3_uses_descriptor_generator_and_refuses_tiled_workers(inputs):
    source, queries = inputs
    config = _config()
    stats, _ = Engine(backend=cpu).run(source, queries, config, TopK(2))
    assert stats.cpu_scan["rows_decoded"] == source.n_conformers
    source.tile = True
    with pytest.raises(ValueError, match="tile=False"):
        Engine(backend=cpu).run(source, queries, _config(), TopK(2))


def test_dense_subclasses_refuse(inputs):
    source, queries = inputs
    class DerivedTopK(TopK):
        pass
    for sink in (DenseScores(), DerivedTopK(2)):
        with pytest.raises(NotImplementedError, match="sink|request"):
            Engine(backend=cpu).run(source, queries, _config(), sink)


@pytest.mark.parametrize("setting", ["workers", "memory_budget_bytes", "topk_state_budget_bytes"])
@pytest.mark.parametrize("value", [True, 1.5, 0])
def test_worker_limits_refuse_invalid_values(inputs, setting, value):
    source, queries = inputs
    config = _config()
    config.cpu_worker = replace(config.cpu_worker, **{setting: value})
    with pytest.raises(ValueError, match=setting):
        Engine(backend=cpu).run(source, queries, config, TopK(2))


@pytest.mark.parametrize("kwargs,match", [
    ({"device_id": 1}, "device_id"),
    ({"stage_timing": True}, "stage_timing"),
])
def test_engine_options_refuse(inputs, kwargs, match):
    source, queries = inputs
    with pytest.raises((ValueError, NotImplementedError), match=match):
        Engine(backend=cpu, **kwargs).run(source, queries, _config(), TopK(2))


def test_incompatible_backend_refuses(inputs):
    from roshambo3.backend import cuda
    source, queries = inputs
    with pytest.raises(ValueError, match="backend='cpu'"):
        Engine(backend=cuda).run(source, queries, _config(), TopK(2))


def test_invalid_matrix_refuses_before_rows(inputs):
    source, queries = inputs
    queries.interaction_r[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite|matrix|matrices|descriptor differs"):
        Engine(backend=cpu).run(source, queries, _config(), TopK(2))


@pytest.mark.parametrize("layout", ["merged"])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("sink_type,level", [(TopK, "conformer"), (MoleculeTopK, "molecule")])
def test_native_public_topk_matches_exact_dense(
        tmp_path, layout, workers, sink_type, level):
    # A missing private DSO is a test failure, never a green skip for this tranche.
    cpu.extension()
    fx = make_codec_fixture(tmp_path / "corpus", color_layout=layout)
    source = _source(fx.corpus)
    queries = _query_set(fx, (0, 4))  # includes an empty-color query
    config = _config(field_spacing_a=0.5)
    config.cpu_worker = replace(config.cpu_worker, workers=workers)
    dense, gids, mids = _compact_reference(
        source, queries, config, 16, shape_backend="pipelined_field",
        shape_grain="library_atom", shape_width=8,
        scoring_options=dict(factor_canonical_torque=True,
            optimizer_exp_mode="rawbits_unclamped", canonical_color_kernel=True,
            final_start_policy="winner"))
    engine = Engine(backend=cpu)
    for repeat in range(2):
        stats, result = engine.run(source, queries, config, sink_type(3))
        for q, actual in enumerate(result):
            expected = _reference_topk(dense[q], gids, mids, k=3,
                                       level=level, rank_column=0)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
        assert stats.cpu_scan["query_contexts_built"] == (0 if repeat else 1)
        assert stats.cpu_scan["query_context_cache_hits"] == (1 if repeat else 0)
        assert stats.cpu_scan["stored_vaa_used"] is False
        assert stats.cpu_scan["native_library_self_overlap_fallback_per_block"] is True
        assert stats.cpu_scan["norm_contract"] == "shared_fp32_reduced_v1"
        assert stats.n_conformers == len(gids)
        assert stats.n_rows == 2 * len(gids)
        assert stats.n_blocks == len(source.cpu_block_descriptors())


@pytest.mark.parametrize("failure", ["cancel", "decode"])
def test_public_failure_cleans_up_and_reuse_has_no_partial_result(inputs, monkeypatch, failure):
    source, queries = inputs
    engine = Engine(backend=cpu)
    original = source.decode_cpu_block
    calls = []
    closed = []
    original_close = cpu.CPUComboExecutor.close
    def close(executor):
        original_close(executor)
        closed.append(executor)
    monkeypatch.setattr(cpu.CPUComboExecutor, "close", close)
    def decode(descriptor):
        calls.append(descriptor)
        if failure == "decode" and len(calls) == 2:
            raise ValueError("public fixture decode failure")
        return original(descriptor)
    monkeypatch.setattr(source, "decode_cpu_block", decode)
    canceled = (lambda: len(calls) >= 1) if failure == "cancel" else None
    error = InterruptedError if failure == "cancel" else ValueError
    with pytest.raises(error, match="canceled|fixture decode failure"):
        engine.run(source, queries, _config(), TopK(2), canceled=canceled)
    assert len(closed) == 1 and closed[0].closed
    monkeypatch.setattr(source, "decode_cpu_block", original)
    stats, result = engine.run(source, queries, _config(), TopK(2))
    assert stats.n_conformers == sum(g["n_conf"] for g in source.corpus.group_table)
    assert len(result) == 2 and all(len(gids) == 2 for _, gids in result)
    assert len(closed) == 2 and all(executor.closed for executor in closed)

"""Fast facade defaults, optional limits and native preflight accounting."""
from dataclasses import replace
import numpy as np
import pytest

from roshambo3.api import screen
from roshambo3.backend import cpu
from roshambo3.backend.spec import ColorPlacement, UpdateRule
from roshambo3.engine import CPUWorkerOptions, Engine, ScreenConfig, ScreenStats
from roshambo3.engine import config as config_module
from tests.codec_fixtures import make_codec_fixture
from tests.test_cpu_worker_owned_scan import _query_set


@pytest.mark.parametrize("mode", ["shape", "rescore", "combo"])
@pytest.mark.parametrize("level", ["conformer", "molecule"])
def test_facade_fast_defaults_all_modes_and_levels(tmp_path, monkeypatch, mode, level):
    from roshambo3.engine.query import from_rdkit
    from tests.test_self_overlap_routes import build_inputs
    corpus, base, generator = build_inputs(tmp_path / "corpus", noncanonical=True)
    query = from_rdkit([base], color_generator=generator)
    monkeypatch.setattr(config_module, "_default_cpu_workers", lambda: 2)
    color = mode != "shape"
    config = ScreenConfig(color=color, mixing=.5 if color else 0.,
        field_spacing_a=.5, color_placement=ColorPlacement.IN_GRADIENT
        if mode == "combo" else ColorPlacement.AT_POSE)
    engine = Engine(backend="cpu")
    kwargs = dict(backend="cpu", engine=engine, k=3, level=level,
        prefetch_depth=0, color_generator=generator if color else None)
    try:
        auto = screen(query, corpus, config=config, **kwargs)
        stats = auto.stats.cpu_scan
        assert stats["mode"] == mode
        assert stats["workers"] == 2
        assert stats["memory_budget_bytes"] is None
        assert stats["topk_state_budget_bytes"] is None
        assert stats["topk_state_estimate_bytes"] > 0
        assert stats["shape_backend"] == "pipelined_field"
        assert stats["optimizer_exp_mode"] == (
            "rawbits_unclamped" if mode == "combo" else "baseline")
        assert stats["library_shape_self_evaluations"] == 0
        # This noncanonical descriptor intentionally has no stored color-self
        # section, so rescore and combo count each binding-time evaluation.
        assert stats["library_color_self_evaluations"] == (
            auto.stats.n_conformers if mode != "shape" else 0)
        explicit = replace(config, cpu_worker=CPUWorkerOptions(
            workers=1,
            memory_budget_bytes=64 << 20, topk_state_budget_bytes=1 << 20))
        limited = screen(query, corpus, config=explicit, workers=2, **kwargs)
        assert limited.stats.cpu_scan["workers"] == 1
        assert limited.stats.cpu_scan["query_contexts_built"] == 0
        np.testing.assert_array_equal(limited.hits[0].scores, auto.hits[0].scores)
        np.testing.assert_array_equal(limited.hits[0].gids, auto.hits[0].gids)
        # An options object used just to opt out of exp must not accidentally
        # force one worker: the facade keyword fills its omitted worker count.
        inherited = replace(explicit, cpu_worker=replace(explicit.cpu_worker, workers=None))
        result = screen(query, corpus, config=inherited, workers=2, **kwargs)
        assert result.stats.cpu_scan["workers"] == 2
    finally:
        engine.close()
    assert config.cpu_worker is None  # No mutation of caller configuration.


def test_topk_ceiling_is_not_charged_as_allocation(tmp_path):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    result = screen(query, fx.corpus, backend="cpu", workers=2, k=3,
        prefetch_depth=0, config=ScreenConfig(field_spacing_a=.5,
            cpu_worker=CPUWorkerOptions(workers=2, memory_budget_bytes=64 << 20,
                topk_state_budget_bytes=1 << 40)))
    assert result.stats.cpu_scan["memory_peak_estimate_bytes"] < 64 << 20
    assert result.stats.cpu_scan["topk_state_estimate_bytes"] < 1 << 20


@pytest.mark.parametrize("level", ["conformer", "molecule"])
def test_native_pure_estimator_agrees_with_collector(level):
    ext = cpu.extension()
    args = (2, 3, 5, 0, level, 16, True, int(np.iinfo(np.uintp).max))
    peak = ext.estimate_worker_local_topk_direct_peak_bytes(*args)
    collector = ext.make_worker_local_topk_direct_row_sink(*args)
    assert peak == collector.memory_estimate["peak_bytes"]
    collector.discard_after_failure()
    with pytest.raises((ValueError, MemoryError), match="budget"):
        ext.estimate_worker_local_topk_direct_peak_bytes(*args[:-1], 1)


def test_auto_workers_respect_affinity_and_fallback(monkeypatch):
    monkeypatch.setattr(config_module.os, "sched_getaffinity", lambda _: {2, 5, 8}, raising=False)
    assert config_module._default_cpu_workers() == 3
    def unavailable(_):
        raise OSError("affinity unavailable")
    monkeypatch.setattr(config_module.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 4)
    assert config_module._default_cpu_workers() == 4
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: None)
    assert config_module._default_cpu_workers() == 1


def test_direct_worker_entry_has_no_implicit_memory_caps():
    import inspect
    parameters = inspect.signature(Engine.run_worker_owned_cpu).parameters
    assert parameters["memory_budget_bytes"].default is None
    assert parameters["topk_state_budget_bytes"].default is None


@pytest.mark.parametrize("override", [
    {"start_mode": 0}, {"update_rule": UpdateRule.ADAGRAD_QADD},
    {"field_pad_a": 9.0}, {"color_ballot": True},
])
def test_unsupported_implicit_config_preserves_ordinary_route(tmp_path, monkeypatch, override):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    engine = Engine(backend="cpu")
    observed = []
    def ordinary(source, queries, config, sink, **kwargs):
        observed.append(config.cpu_worker)
        return ScreenStats(), [(np.zeros((0, 20), np.float32), np.zeros(0, np.int64))]
    monkeypatch.setattr(engine, "run", ordinary)
    try:
        screen(query, fx.corpus, backend="cpu", workers=1, engine=engine,
            config=ScreenConfig(**override), prefetch_depth=0)
        assert observed == [None]
    finally:
        engine.close()


@pytest.mark.parametrize("engine_kwargs", [{"stage_timing": True}, {"device_id": 1}])
def test_unsupported_implicit_engine_preserves_ordinary_route(tmp_path, monkeypatch, engine_kwargs):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    engine = Engine(backend="cpu", **engine_kwargs)
    observed = []
    def ordinary(source, queries, config, sink, **kwargs):
        observed.append(config.cpu_worker)
        return ScreenStats(), [(np.zeros((0, 20), np.float32), np.zeros(0, np.int64))]
    monkeypatch.setattr(engine, "run", ordinary)
    try:
        screen(query, fx.corpus, backend="cpu", workers=1, engine=engine,
            config=ScreenConfig(), prefetch_depth=0)
        assert observed == [None]
    finally:
        engine.close()

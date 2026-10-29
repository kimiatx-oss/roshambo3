"""Public CPU facade and one-generation immutable query-context reuse."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from roshambo3.api import screen
from roshambo3.backend import cpu, cuda
from roshambo3.engine import engine as engine_module
from roshambo3.engine import (CPUWorkerOptions, Engine, HostTopK, MoleculeTopK,
                             ScreenConfig, ScreenStats, TopK, from_arrays)
from roshambo3.backend.spec import ColorPlacement
from roshambo3.numerics import WIRE_QUANT_STEP
from tests.test_cpu_worker_public_optin import _config
from tests.test_cpu_worker_owned_scan import _query_set, _source
from tests.codec_fixtures import make_codec_fixture


def _color_generator(fx):
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    return generator_from_descriptor(fx.corpus.color)


@pytest.fixture
def inputs(tmp_path):
    cpu.extension()  # Missing native backend must not produce a green skip.
    fx = make_codec_fixture(tmp_path/"corpus", color_layout="merged")
    return fx, _query_set(fx, (0, 4))


@pytest.mark.parametrize("level", ["conformer", "molecule"])
@pytest.mark.parametrize("explicit", [False, True])
def test_cpu_facade_native_parity_and_session_reuse(inputs, monkeypatch, level, explicit):
    fx, query = inputs
    config = replace(_config(field_spacing_a=.5), cpu_worker=(
        CPUWorkerOptions(workers=2) if explicit else None))
    # screen() resolves both omitted combo optimizations to the mode-aware
    # worker defaults. Match those exact settings in the direct Engine path.
    expected_config = replace(config, cpu_worker=CPUWorkerOptions(
        workers=2, factor_canonical_torque=True,
        canonical_color_kernel=True))
    expected_engine = Engine(backend="cpu")
    _, expected = expected_engine.run(_source(fx.corpus), query, expected_config,
        TopK(3) if level == "conformer" else MoleculeTopK(3))
    expected_engine.close()
    monkeypatch.setattr(cuda, "extension", lambda: pytest.fail("CPU screen invoked CUDA"))
    generator = _color_generator(fx)
    engine = Engine(backend="cpu")
    first_contexts = None
    for repeat in range(2):
        actual = screen(query, fx.corpus, config=config, engine=engine, backend="cpu",
            color_generator=generator, workers=2, k=3, level=level, prefetch_depth=0)
        for got, (scores, gids) in zip(actual.hits, expected):
            np.testing.assert_array_equal(got.scores, scores)
            np.testing.assert_array_equal(got.gids, gids)
        stats = actual.stats.cpu_scan
        assert stats["workers"] == 2
        assert stats["query_contexts_built"] == (0 if repeat else 1)
        assert stats["query_context_cache_hits"] == (1 if repeat else 0)
        # Each public screen creates a fresh format-3 source binding, so its
        # authenticated query self values are prepared once per query.
        assert stats["shape_query_self_evaluations"] == query.n_conformers
        assert stats["color_query_self_evaluations"] == query.n_conformers
        assert stats["query_self_preparation_s"] > 0
        assert stats["query_context_retained_bytes"] == stats["query_context_estimate_bytes"]
        assert stats["memory_budget_bytes"] is None
        assert stats["topk_state_budget_bytes"] is None
        assert stats["memory_peak_estimate_bytes"] > 0
        if repeat:
            assert engine._worker_query_cache[1] is first_contexts
        first_contexts = engine._worker_query_cache[1]
    assert (config.cpu_worker is not None) == explicit
    engine.close()
    assert engine._worker_query_cache is None and engine._worker_self_cache is None


@pytest.mark.parametrize("part", ["coords", "types", "n_real", "spacing", "native"])
def test_query_context_complete_binding_invalidates(inputs, monkeypatch, part):
    fx, query = inputs
    config = replace(_config(field_spacing_a=.5), cpu_worker=CPUWorkerOptions())
    engine = Engine(backend="cpu")
    source = _source(fx.corpus)
    engine.run(source, query, config, TopK(2))
    old = engine._worker_query_cache[1]
    if part == "coords":
        query.coords[0,0,0] += .1
    elif part == "types":
        query.types[0,int(query.n_real[0])] = 1
    elif part == "n_real":
        query.n_real[0] += 1
    elif part == "spacing":
        config = replace(config, field_spacing_a=.6)
    else:
        stamp = cpu.build_stamp()
        monkeypatch.setattr(cpu, "build_stamp", lambda: dict(stamp, test_native_identity="changed"))
    stats, _ = engine.run(source, query, config, TopK(2))
    assert stats.cpu_scan["query_context_cache_invalidated"]
    assert stats.cpu_scan["query_contexts_built"] == 1
    assert stats.cpu_scan["query_context_cache_hits"] == 0
    assert engine._worker_query_cache[1] is not old
    engine.close()


@pytest.mark.parametrize("matrix", ["interaction_r", "interaction_p"])
def test_format3_query_matrices_must_match_corpus_descriptor(inputs, matrix):
    from roshambo3.format.r3g.descriptor import DefinitionMismatch

    fx, query = inputs
    config = replace(_config(field_spacing_a=.5), cpu_worker=CPUWorkerOptions())
    engine = Engine(backend="cpu")
    source = _source(fx.corpus)
    engine.run(source, query, config, TopK(2))
    getattr(query, matrix)[1, 1] += .2
    field = "r" if matrix == "interaction_r" else "p"
    with pytest.raises(DefinitionMismatch, match=field):
        engine.run(source, query, config, TopK(2))
    engine.close()


def test_lazy_query_self_work_counted_once_even_on_reused_context(inputs):
    fx, query = inputs
    engine = Engine(backend="cpu")
    source = _source(fx.corpus)
    config = replace(_config(field_spacing_a=.5), cpu_worker=CPUWorkerOptions())
    first, _ = engine.run(source, query, config, TopK(2))
    assert first.cpu_scan["shape_query_self_evaluations"] == query.n_conformers
    config = replace(config, cpu_worker=replace(config.cpu_worker, self_overlap_policy="recompute"))
    for repeat in range(2):
        stats, _ = engine.run(source, query, config, TopK(2))
        assert stats.cpu_scan["query_contexts_built"] == 0
        assert stats.cpu_scan["query_self_contexts_built"] == (0 if repeat else 1)
        assert stats.cpu_scan["shape_query_self_evaluations"] == (0 if repeat else query.n_conformers)
        assert stats.cpu_scan["color_query_self_evaluations"] == (0 if repeat else query.n_conformers)
        if repeat:
            assert stats.cpu_scan["query_self_preparation_s"] == 0
    engine.close()


@pytest.mark.parametrize("failure", ["validation", "memory", "preparation", "scan", "cancel"])
def test_failed_scan_clears_retained_query_contexts(inputs, monkeypatch, failure):
    fx, query = inputs
    config = replace(_config(field_spacing_a=.5), cpu_worker=CPUWorkerOptions())
    engine = Engine(backend="cpu")
    source = _source(fx.corpus)
    engine.run(source, query, config, TopK(2))
    assert engine._worker_query_cache is not None
    kwargs = {}
    if failure == "validation":
        config = replace(config, field_pad_a=9.)
    elif failure == "memory":
        config = replace(config, cpu_worker=replace(config.cpu_worker, memory_budget_bytes=1))
    elif failure == "preparation":
        query.coords[0,0,0] += .1
        monkeypatch.setattr(cpu, "prepare_combo_shape_evaluators", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("prepare failed")))
    elif failure == "scan":
        monkeypatch.setattr(cpu, "score_pairs", lambda **k: (_ for _ in ()).throw(RuntimeError("scan failed")))
    elif failure == "cancel":
        kwargs["canceled"] = lambda: True
    with pytest.raises((ValueError, MemoryError, RuntimeError, InterruptedError)):
        engine.run(source, query, config, TopK(2), **kwargs)
    assert engine._worker_query_cache is None and engine._worker_self_cache is None


def test_explicit_options_win_and_api_conflicts_refuse(inputs):
    fx, query = inputs
    generator = _color_generator(fx)
    config = replace(_config(field_spacing_a=.5), cpu_worker=CPUWorkerOptions(workers=1))
    engine = Engine(backend="cpu")
    actual = screen(query, fx.corpus, config=config, engine=engine, workers=2,
                    k=2, color_generator=generator)
    assert actual.stats.cpu_scan["workers"] == 1
    for kwargs in ({"backend":"cuda"}, {"device_id":1}, {"stage_timing":True}):
        with pytest.raises(ValueError, match="conflicts"):
            screen(query, fx.corpus, config=config, engine=engine,
                   color_generator=generator, **kwargs)
    wrong = _color_generator(fx)
    wrong.interaction_matrix_p[1, 2] += .2
    with pytest.raises(ValueError):
        screen(query, fx.corpus, config=config, engine=engine, color_generator=wrong)
    with pytest.raises(ValueError, match="requires the color_generator"):
        screen(query, fx.corpus, config=replace(config, cpu_worker=None), backend="cpu",
               color_generator=None)
    with pytest.raises(TypeError, match="CPUWorkerOptions"):
        screen(query, fx.corpus, config=replace(config, cpu_worker="wrong"),
               backend="cpu", color_generator=generator)
    engine.close()


def test_complete_spacing_set_has_one_bounded_generation(inputs):
    fx, query = inputs
    generator = _color_generator(fx)
    config = replace(_config(), cpu_worker=CPUWorkerOptions())
    engine = Engine(backend="cpu")
    first = screen(query, fx.corpus, engine=engine, config=config, k=2,
                   color_generator=generator)
    expected = len({int(group["L_field_width"]) for group in fx.corpus.group_table})
    assert first.stats.cpu_scan["query_contexts_built"] == expected
    assert len(engine._worker_query_cache[1]) == expected
    second = screen(query, fx.corpus, engine=engine, config=config, k=2,
                    groups=[0], color_generator=generator)
    # Group selection does not alter the field geometry or query definition;
    # this corpus's selected groups share the same spacing, so the complete
    # prepared query context remains reusable.
    assert not second.stats.cpu_scan["query_context_cache_invalidated"]
    assert second.stats.cpu_scan["query_contexts_built"] == 0
    assert second.stats.cpu_scan["query_context_cache_hits"] == 1
    assert len(engine._worker_query_cache[1]) == expected
    assert second.stats.cpu_scan["query_context_retained_bytes"] == first.stats.cpu_scan["query_context_retained_bytes"]
    engine.close()


def test_public_shape_precision_default_fallback_optout_and_cache_key(
        inputs, monkeypatch):
    fx, query = inputs
    source = _source(fx.corpus)
    config = replace(_config(field_spacing_a=.5), cpu_worker=CPUWorkerOptions(workers=1))
    engine = Engine(backend="cpu")
    # The measured ARM common prepared loop now favors FP32. Auto and explicit
    # FP32 share effective geometry/cache identity, while request stamps differ.
    monkeypatch.setattr(engine_module.platform, "machine", lambda: "aarch64")

    first, _ = engine.run(source, query, config, TopK(2))
    assert first.cpu_scan["shape_field_interpolation_policy"] == "auto"
    assert first.cpu_scan["shape_field_storage_request"] == "fp32"
    assert first.cpu_scan["shape_field_interpolation_request"] == "fp32"
    assert first.cpu_scan["query_contexts_built"] == 1
    assert first.cpu_scan["shape_field_storage_stamps"]
    storage_stamps = first.cpu_scan["shape_field_storage_stamps"]
    assert all(stamp == "FP32 values" for stamp in storage_stamps)

    reused, _ = engine.run(source, query, config, TopK(2))
    assert reused.cpu_scan["query_contexts_built"] == 0
    assert reused.cpu_scan["query_context_cache_hits"] == 1
    assert reused.cpu_scan["shape_field_storage_request"] == "fp32"
    assert reused.cpu_scan["shape_field_storage_stamps"] == (
        first.cpu_scan["shape_field_storage_stamps"])

    optout = replace(config, cpu_worker=replace(
        config.cpu_worker, shape_field_interpolation="fp32"))
    explicit, _ = engine.run(source, query, optout, TopK(2))
    assert explicit.cpu_scan["shape_field_interpolation_policy"] == "fp32"
    assert explicit.cpu_scan["shape_field_storage_request"] == "fp32"
    assert explicit.cpu_scan["shape_field_interpolation_request"] == "fp32"
    assert not explicit.cpu_scan["query_context_cache_invalidated"]
    assert explicit.cpu_scan["query_contexts_built"] == 0
    assert explicit.cpu_scan["query_context_cache_hits"] == 1
    assert all(stamp == "FP32 values"
               for stamp in explicit.cpu_scan["shape_field_storage_stamps"])

    explicit_reused, _ = engine.run(source, query, optout, TopK(2))
    assert explicit_reused.cpu_scan["query_context_cache_hits"] == 1
    assert not explicit_reused.cpu_scan["query_context_cache_invalidated"]
    engine.close()


def test_public_shape_precision_auto_is_arch_and_mode_bounded():
    resolve = engine_module._resolve_public_shape_field_precision
    field_route = dict(shape_backend="pipelined_field",
                       shape_grain="library_atom", shape_width=8)
    assert resolve("auto", mode="combo", machine="aarch64",
                   **field_route) == ("fp32", "fp32")
    assert resolve("auto", mode="combo", machine="x86_64",
                   **field_route) == ("fp32", "fp32")
    assert resolve("auto", mode="rescore", machine="aarch64",
                   **field_route) == ("fp32", "fp32")
    assert resolve("auto", mode="combo", machine="aarch64",
                   shape_backend="scalar_analytic", shape_grain="scalar",
                   shape_width=1) == ("fp32", "fp32")
    assert resolve("fp32", mode="combo", machine="aarch64",
                   **field_route) == ("fp32", "fp32")


def test_cpu_shape_facade_stays_shape_and_uses_fast_reduction(inputs, monkeypatch):
    fx, _ = inputs
    xyz = fx.shape_xyz_i16[0,:,:int(fx.n_real[0])].T.astype(np.float32)*np.float32(WIRE_QUANT_STEP)
    query = from_arrays([xyz], ["shape"])
    config = ScreenConfig()
    engine = Engine(backend="cpu")
    monkeypatch.setattr(cuda, "extension", lambda: pytest.fail("CPU shape invoked CUDA"))
    actual = screen(query, fx.corpus, config=config, engine=engine, k=2, workers=1,
        prefetch_depth=0, launch_conformers=3)
    source = __import__("roshambo3.engine", fromlist=["R3gSource"]).R3gSource(
        fx.corpus, tile=False, color=False, workers=1, prefetch_depth=0, conformers_per_launch=3)
    # The old ordinary route recomputed self-overlap; the production default
    # uses accepted stored volumes (this synthetic fixture deliberately stores
    # nonphysical VAA). Compare with the explicit prepared worker contract.
    explicit = replace(config, cpu_worker=CPUWorkerOptions(workers=1))
    _, expected = engine.run(source, query, explicit, HostTopK(2))
    np.testing.assert_array_equal(actual.hits[0].scores, expected[0][0])
    assert actual.stats.cpu_scan["mode"] == "shape" and config.cpu_worker is None
    molecules = screen(query, fx.corpus, config=config, engine=engine,
        workers=1, level="molecule", k=2, prefetch_depth=0)
    assert molecules.stats.cpu_scan["mode"] == "shape"
    assert molecules.hits[0].scores.shape == (2, 20)
    engine.close()


def test_cuda_default_translation_unchanged(inputs, monkeypatch):
    from roshambo3 import api
    fx, query = inputs
    config = ScreenConfig()
    observed = {}
    class DefaultEngine:
        def make_topk(self, k, level):
            return TopK(k)
        def close(self): pass
        def __init__(self, **kwargs):
            observed["kwargs"] = kwargs
            self.backend = SimpleNamespace(__name__="roshambo3.backend.cuda")
        def run(self, source, queries, cfg, sink):
            assert source.tile and cfg is config and type(sink) is TopK
            return ScreenStats(), [(np.zeros((0,20), np.float32),
                np.zeros(0, np.int64)) for _ in range(queries.n_conformers)]
    monkeypatch.setattr(api, "Engine", DefaultEngine)
    result = api.screen(query, fx.corpus, config=config)
    assert observed["kwargs"] == {"device_id":0, "stage_timing":False, "backend":None}
    assert len(result.hits) == query.n_conformers and config.cpu_worker is None


def test_direct_worker_shape_policy_and_padding_replace_generation(inputs):
    fx, query = inputs
    source, engine = _source(fx.corpus), Engine(backend="cpu")
    config = replace(_config(), cpu_worker=None)
    options = dict(workers=1, k=2, color_grain=16,
        shape_backend="pipelined_field", shape_grain="library_atom", shape_width=8,
        field_spacing_angstrom=.5)
    engine.run_worker_owned_cpu(source, query, config, **options)
    stats, _ = engine.run_worker_owned_cpu(source, query, config,
        field_padding_angstrom=9., **options)
    assert stats.cpu_scan["query_context_cache_invalidated"]
    assert stats.cpu_scan["query_contexts_built"] == 1
    options.update(shape_backend="scalar_analytic", shape_grain="scalar", shape_width=1)
    stats, _ = engine.run_worker_owned_cpu(source, query, config, **options)
    assert stats.cpu_scan["query_context_cache_invalidated"]
    assert len(engine._worker_query_cache[1]) == 1
    engine.close()

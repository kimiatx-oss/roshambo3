"""Actual public mode dispatch, format-3 self bindings and bounded lifecycle."""
from dataclasses import replace
import numpy as np
import pytest
from roshambo3.api import screen
from roshambo3.backend import cpu
from roshambo3.backend.spec import ColorPlacement
from roshambo3.engine import CPUWorkerOptions, Engine, R3gSource, ScreenConfig
from tests.test_cpu_worker_owned_scan import _query_set
from tests.codec_fixtures import make_codec_fixture


@pytest.mark.parametrize("rescore", [False, True])
def test_actual_public_shape_modes_exact_query_and_self_reuse(tmp_path, rescore):
    fixture = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fixture, (0,))
    config = ScreenConfig(color=rescore, mixing=.5 if rescore else 0.,
        color_placement=ColorPlacement.AT_POSE, field_spacing_a=.5,
        cpu_worker=CPUWorkerOptions(workers=2))
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    generator = generator_from_descriptor(fixture.corpus.color)
    source = R3gSource(fixture.corpus, color=rescore,
        color_generator=generator if rescore else None,
        tile=False, workers=1, prefetch_depth=0)
    engine = Engine(backend="cpu")
    try:
        stats, expected = engine.run_worker_owned_cpu(source, query, config,
            workers=2, k=100, color_grain=16, shape_backend="pipelined_field",
            shape_grain="library_atom", shape_width=8, self_overlap_policy="prepared")
        contexts = engine._worker_query_cache[1]
        second_stats, second = engine.run_worker_owned_cpu(source, query, config,
            workers=2, k=100, color_grain=16, shape_backend="pipelined_field",
            shape_grain="library_atom", shape_width=8, self_overlap_policy="prepared")
        assert second_stats.cpu_scan["query_contexts_built"] == 0
        # Query context persists across runs; block-local self handles are
        # released after scoring so retained denominator state stays worker-bounded.
        assert second_stats.cpu_scan["self_preparation_calls"] == stats.n_blocks
        assert second_stats.cpu_scan["self_cache_hits"] == 0
        assert second_stats.cpu_scan["library_shape_self_evaluations"] == 0
        assert second_stats.cpu_scan["library_color_self_evaluations"] == 0
        assert engine._worker_query_cache[1] is contexts
        for first, later in zip(expected[0], second[0]):
            np.testing.assert_array_equal(first, later)
        # Recompute from the same native-decoded rows to verify the explicit
        # fallback remains available. Stored VAA and fresh recomputation can
        # differ by their documented frame-conversion precision.
        fallback_stats, fallback = engine.run_worker_owned_cpu(source, query, config,
            workers=2, k=100, color_grain=16, shape_backend="pipelined_field",
            shape_grain="library_atom", shape_width=8,
            self_overlap_policy="recompute")
        assert not fallback_stats.cpu_scan["stored_vaa_used"]
        assert not fallback_stats.cpu_scan["library_self_evaluations_counted"]
        assert fallback_stats.cpu_scan["native_self_overlap_fallback_calls"] == (
            fallback_stats.n_blocks)
        assert all(np.isfinite(item[0]).all() for item in fallback)
        # api.screen must genuinely select persistent workers and reuse the
        # query generation even though its fresh Source invalidates self.
        public = screen(query, fixture.corpus, config=config, engine=engine,
            backend="cpu", workers=2, k=100, prefetch_depth=0,
            color_generator=generator if rescore else None)
        assert public.stats.cpu_scan["mode"] == ("rescore" if rescore else "shape")
        assert public.stats.cpu_scan["query_contexts_built"] == 0
        assert public.stats.cpu_scan["library_shape_self_evaluations"] == 0
        assert public.stats.cpu_scan["library_color_self_evaluations"] == 0
        assert public.stats.cpu_scan["color_self_sidecar_blocks"] == (
            public.stats.n_blocks if rescore else 0)
        np.testing.assert_array_equal(public.hits[0].scores, expected[0][0])
        np.testing.assert_array_equal(public.hits[0].gids, expected[0][1])
        query.coords[0, 0, 0] += .125
        changed = screen(query, fixture.corpus, config=config, engine=engine,
            backend="cpu", k=100, prefetch_depth=0,
            color_generator=generator if rescore else None)
        assert changed.stats.cpu_scan["query_contexts_built"] == 1
    finally:
        engine.close()
    assert engine._worker_query_cache is None
    assert engine._worker_self_cache is None


@pytest.mark.parametrize("bad", ["query", "negative", "library", "width", "length"])
def test_block_ingestion_rejects_before_any_row_is_offered(bad):
    ext = cpu.extension()
    collector = ext.make_worker_local_topk_direct_row_sink(1, 1, 2, 0, "conformer", 2, False, 1 << 20)
    sink = ext.make_worker_local_topk_block_row_sink(collector, 0, 10, 2)
    values = np.zeros((2, 20), np.float32)
    values[:, 0] = (2, 1)
    qi = np.zeros(2, np.int32)
    li = np.arange(2, dtype=np.int32)
    if bad == "query": qi[1] = 1
    if bad == "negative": qi[1] = -1
    if bad == "library": li[1] = 2
    if bad == "width": values = values[:, :19].copy()
    if bad == "length": qi = qi[:1]
    with pytest.raises((ValueError, IndexError)):
        sink.ingest_scores(values, qi, li)
    collector.mark_scoring_succeeded()
    assert collector.finalize()[0][0].shape == (0, 20)


@pytest.mark.parametrize("bad", ["negative_query", "long_query", "long_library", "nonfinite"])
def test_rescore_self_preparation_rejects_invalid_bounds(bad):
    xyz = np.zeros((1, 2, 3), np.float32)
    qt = np.zeros((1, 2), np.int32)
    qn = np.array([2], np.int32)
    counts = np.array([[2, 0]], np.uint8)
    if bad == "negative_query": qn[0] = -1
    if bad == "long_query": qn[0] = 3
    if bad == "long_library": counts[0, 1] = 1
    if bad == "nonfinite": xyz[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        cpu.extension().prepare_rescore_color_self(xyz, qt, qn, xyz,
            counts, np.ones((2, 2), np.float32), np.eye(2, dtype=np.float32))


@pytest.mark.parametrize("flag", ["factor_canonical_torque", "optimizer_exp_mode", "canonical_color_kernel"])
@pytest.mark.parametrize("rescore", [False, True])
def test_noncombo_rejects_inapplicable_optimizer_color_options(tmp_path, flag, rescore):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    options = CPUWorkerOptions(**{flag: "rawbits" if flag == "optimizer_exp_mode" else True})
    config = ScreenConfig(color=rescore, mixing=.5 if rescore else 0., cpu_worker=options)
    engine = Engine(backend="cpu")
    try:
        with pytest.raises(ValueError, match="require combo mode"):
            from roshambo3.format.r3g.descriptor import generator_from_descriptor
            screen(query, fx.corpus, config=config, engine=engine, k=2, prefetch_depth=0,
                   color_generator=generator_from_descriptor(fx.corpus.color)
                   if rescore else None)
        assert engine._worker_query_cache is None
        assert engine._worker_self_cache is None
    finally:
        engine.close()


def test_shape_corpus_needs_no_color_enum_and_molecule_ranking_is_native(tmp_path):
    from roshambo3.engine import from_arrays
    fx = make_codec_fixture(tmp_path / "shape", color_layout="none")
    xyz = fx.shape_xyz_i16[0, :, :int(fx.n_real[0])].T.astype(np.float32) * np.float32(.001)
    query = from_arrays([xyz], ["shape"])
    config = ScreenConfig(field_spacing_a=.5, cpu_worker=CPUWorkerOptions(workers=2))
    engine = Engine(backend="cpu")
    try:
        conf = screen(query, fx.corpus, engine=engine, config=config,
            k=100, workers=2, backend="cpu", prefetch_depth=0)
        mol = screen(query, fx.corpus, engine=engine, config=config,
            k=100, workers=2, backend="cpu", level="molecule", prefetch_depth=0)
        assert conf.stats.cpu_scan["mode"] == "shape"
        assert conf.stats.cpu_scan["color_blocks_decoded"] == 0
        assert conf.stats.cpu_scan["color_query_self_evaluations"] == 0
        assert conf.stats.cpu_scan["color_field_builds_this_call"] == 0
        assert mol.stats.cpu_scan["query_contexts_built"] == 0
        mids = fx.molecule_ids
        wanted = []
        seen = set()
        for row, gid in zip(conf.hits[0].scores, conf.hits[0].gids):
            if int(mids[gid]) not in seen:
                seen.add(int(mids[gid]))
                wanted.append((row, gid))
        np.testing.assert_array_equal(mol.hits[0].gids, [gid for _, gid in wanted])
        np.testing.assert_array_equal(mol.hits[0].scores, [row for row, _ in wanted])
    finally:
        engine.close()


@pytest.mark.parametrize("failure", ["memory", "score", "validation", "cancel"])
def test_failed_shape_worker_run_clears_cached_state(tmp_path, monkeypatch, failure):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="none")
    from roshambo3.engine import from_arrays
    query = from_arrays([np.zeros((3, 3), np.float32)], ["shape"])
    config = ScreenConfig(field_spacing_a=.5, cpu_worker=CPUWorkerOptions(workers=2))
    engine = Engine(backend="cpu")
    try:
        screen(query, fx.corpus, engine=engine, config=config, k=2, prefetch_depth=0)
        source = R3gSource(fx.corpus, color=False, tile=False, workers=1, prefetch_depth=0)
        kwargs = {}
        if failure == "memory":
            config = replace(config, cpu_worker=replace(config.cpu_worker, memory_budget_bytes=1))
        elif failure == "score":
            monkeypatch.setattr(cpu, "score_pairs", lambda **k: (_ for _ in ()).throw(RuntimeError("score failed")))
        elif failure == "validation":
            config = replace(config, field_pad_a=9.)
        else:
            kwargs["canceled"] = lambda: True
        with pytest.raises((ValueError, MemoryError, RuntimeError, InterruptedError)):
            engine.run(source, query, config, __import__("roshambo3.engine", fromlist=["TopK"]).TopK(2), **kwargs)
        assert engine._worker_query_cache is None
        assert engine._worker_self_cache is None
    finally:
        engine.close()

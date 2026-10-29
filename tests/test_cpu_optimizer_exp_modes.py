"""Call-local approximate optimizer selection, exact finals and fallback."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from roshambo3.backend import cpu
from test_cpu_shared_pose_contract import _inputs, _prepare, _score


@pytest.mark.parametrize("level", ["conformer", "molecule"])
@pytest.mark.parametrize("canonical_kernel", [False, True])
def test_public_options_reach_worker_and_reuse_query_cache(tmp_path, monkeypatch, level, canonical_kernel):
    from dataclasses import replace
    from roshambo3.api import screen
    from roshambo3.engine import CPUWorkerOptions, Engine
    from tests.test_cpu_worker_public_optin import _config
    from tests.test_cpu_worker_owned_scan import _query_set
    from tests.codec_fixtures import make_codec_fixture
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0, 4))
    generator = generator_from_descriptor(fx.corpus.color)
    engine = Engine(backend="cpu")
    original = cpu.score_pairs
    calls = []
    def record(**kwargs):
        calls.append((kwargs["optimizer_exp_mode"], kwargs["factor_canonical_torque"],
                      kwargs["canonical_color_kernel"]))
        return original(**kwargs)
    monkeypatch.setattr(cpu, "score_pairs", record)
    config = replace(_config(field_spacing_a=.5), cpu_worker=CPUWorkerOptions(
        workers=2, optimizer_exp_mode="rawbits",
        factor_canonical_torque=True, canonical_color_kernel=canonical_kernel))
    try:
        for repeat in range(2):
            actual = screen(query, fx.corpus, config=config, engine=engine,
                            backend="cpu", color_generator=generator, k=3,
                            level=level, prefetch_depth=0)
            assert actual.stats.cpu_scan["query_contexts_built"] == (0 if repeat else 1)
            assert actual.stats.cpu_scan["query_context_cache_hits"] == (1 if repeat else 0)
            if repeat:
                for first, second in zip(saved.hits, actual.hits):
                    np.testing.assert_array_equal(first.scores, second.scores)
                    np.testing.assert_array_equal(first.gids, second.gids)
            saved = actual
        assert calls and set(calls) == {("rawbits", True, canonical_kernel)}
        # Optimizer trajectory policy does not alter cached query geometry.
        config = replace(config, cpu_worker=replace(config.cpu_worker,
                         optimizer_exp_mode="baseline", factor_canonical_torque=False,
                         canonical_color_kernel=not canonical_kernel))
        actual = screen(query, fx.corpus, config=config, engine=engine,
                        backend="cpu", color_generator=generator, k=3,
                        level=level, prefetch_depth=0)
        assert actual.stats.cpu_scan["query_context_cache_hits"] == 1
        assert calls[-1] == ("baseline", False, not canonical_kernel)
    finally:
        engine.close()
    assert engine._worker_query_cache is None


@pytest.mark.parametrize("canonical_kernel", [False, True])
def test_concurrent_approximate_and_baseline_calls_are_isolated(canonical_kernel):
    data = _inputs(canonical=True)
    prepared = _prepare(data)
    def score(mode, factor):
        return _score(data, prepared, grain=16, use_plan=False, reuse=False,
                      optimizer_exp_mode=mode, factor_torque=factor,
                      canonical_color_kernel=canonical_kernel)
    settings = [("baseline", False), ("rawbits", False), ("rawbits", True)]
    expected = [score(*setting)[0] for setting in settings]
    from cpu_test_capabilities import IS_ARM
    if IS_ARM and not canonical_kernel:
        np.testing.assert_array_equal(expected[0], expected[1])
    else:
        assert not np.array_equal(expected[0], expected[1])
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(score, *settings[i % 3]) for i in range(9)]
        for i, future in enumerate(futures):
            rows, stats = future.result()
            np.testing.assert_array_equal(rows, expected[i % 3])
            assert np.isfinite(rows).all()
            assert stats["color_per_start_final_evaluations"] == 12
            assert stats["color_winner_rescore_evaluations"] == 0


@pytest.mark.parametrize("factor", [False, True])
@pytest.mark.parametrize("canonical_kernel", [False, True])
def test_general_fallback_is_exact(factor, canonical_kernel):
    data = _inputs(canonical=False)
    prepared = _prepare(data)
    baseline, _ = _score(data, prepared, grain=16, factor_torque=factor)
    approximate, _ = _score(data, prepared, grain=16, factor_torque=factor,
                            optimizer_exp_mode="rawbits",
                            canonical_color_kernel=canonical_kernel)
    np.testing.assert_array_equal(baseline, approximate)


@pytest.mark.parametrize("factor", [False, True])
@pytest.mark.parametrize("canonical_kernel", [False, True])
def test_approximate_winner_color_is_rescored_exactly(factor, canonical_kernel):
    data = _inputs(canonical=True)
    q, qt, qn, library, counts, widths, heights = data
    rows, _ = _score(data, _prepare(data), grain=16, factor_torque=factor,
                     optimizer_exp_mode="rawbits",
                     canonical_color_kernel=canonical_kernel)
    exact, _ = cpu.extension().rescore_scalar_color(
        q, qt, qn, library, counts, np.array([0, 1, 0], np.int32),
        np.arange(3, dtype=np.int32), widths, heights, rows,
        np.float32(.5), False, False, True, 0)
    # Independent final replay uses the existing physical-frame exact scorer.
    np.testing.assert_allclose(rows[:, [2, 4, 7, 8]],
                               exact[:, [2, 4, 7, 8]], rtol=3e-6, atol=3e-7)


def test_unknown_mode_and_unsupported_grain_refused():
    data = _inputs(canonical=True)
    prepared = _prepare(data)
    with pytest.raises(ValueError, match="must be baseline or rawbits"):
        _score(data, prepared, optimizer_exp_mode="polishedbits")
    with pytest.raises(ValueError, match="rawbits requires"):
        _score(data, prepared, grain=11, optimizer_exp_mode="rawbits")


@pytest.mark.parametrize("use_plan", [False, True])
def test_canonical_policies_repeat_and_report_exact_score_meaning(use_plan):
    from tests.color_score_oracle import assert_reported_color_scores
    data = _inputs(canonical=True)
    prepared = _prepare(data)
    # Canonical normalization/fusion is a declared trajectory change.
    # Each policy must repeat exactly and report the correct score at its own
    # returned pose; different search policies need not return identical poses.
    for canonical in (False, True):
        options = dict(grain=16, use_plan=use_plan,
                       canonical_color_kernel=canonical)
        first, _ = _score(data, prepared, **options)
        repeated, _ = _score(data, prepared, **options)
        np.testing.assert_array_equal(first, repeated)
        q, qt, qn, library, counts, widths, heights = data
        assert_reported_color_scores(first, q, qt, qn, library, counts,
            np.array([0, 1, 0]), np.arange(3), widths, heights)


def test_canonical_selector_defaults_off_and_unsupported_route_refused():
    from roshambo3.engine import CPUWorkerOptions
    options = CPUWorkerOptions()
    assert options.factor_canonical_torque is None
    assert options.optimizer_exp_mode is None
    assert options.canonical_color_kernel is None
    data = _inputs(canonical=True)
    with pytest.raises(ValueError, match="canonical_color_kernel requires"):
        _score(data, _prepare(data), grain=11, canonical_color_kernel=True)

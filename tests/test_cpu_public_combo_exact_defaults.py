"""Fast defaults with exact canonical eligibility on public CPU combo."""
import numpy as np
import pytest

from roshambo3.api import screen
from roshambo3.backend import cpu
from roshambo3.backend.spec import ColorPlacement
from roshambo3.engine import CPUWorkerOptions, Engine, ScreenConfig
from tests.test_cpu_worker_owned_scan import _query_set
from tests.codec_fixtures import make_codec_fixture


def _generator(fx):
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    return generator_from_descriptor(fx.corpus.color)


def _combo_config(*, cpu_worker=None, placement=ColorPlacement.IN_GRADIENT):
    return ScreenConfig(color=True, mixing=.5, color_placement=placement,
        field_spacing_a=.5, cpu_worker=cpu_worker)


def test_public_auto_combo_selects_unclamped_rawbits_and_exact_canonical_factorization(tmp_path):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    engine = Engine(backend="cpu")
    try:
        result = screen(query, fx.corpus, config=_combo_config(), engine=engine,
            color_generator=_generator(fx),
            backend="cpu", workers=1, k=3, prefetch_depth=0)
        stats = result.stats.cpu_scan
        assert stats["mode"] == "combo"
        assert stats["canonical_color_kernel"] is True
        assert stats["factor_canonical_torque"] is True
        assert stats["optimizer_exp_mode"] == "rawbits_unclamped"

        # Explicit options with their public defaults resolve identically to
        # the API-generated worker options and reach the same native route.
        explicit_engine = Engine(backend="cpu")
        try:
            explicit = screen(query, fx.corpus,
                config=_combo_config(cpu_worker=CPUWorkerOptions(
                    workers=1)),
                engine=explicit_engine,
                color_generator=_generator(fx),
                backend="cpu", workers=1, k=3, prefetch_depth=0)
            assert explicit.stats.cpu_scan["optimizer_exp_mode"] == "rawbits_unclamped"
            assert explicit.stats.cpu_scan["factor_canonical_torque"] is True
            assert explicit.stats.cpu_scan["canonical_color_kernel"] is True
            np.testing.assert_array_equal(explicit.hits[0].gids, result.hits[0].gids)
            np.testing.assert_array_equal(explicit.hits[0].scores, result.hits[0].scores)
        finally:
            explicit_engine.close()
    finally:
        engine.close()


def test_explicit_worker_options_override_auto_combo_policy(tmp_path):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    options = CPUWorkerOptions(workers=1,
        self_overlap_policy="prepared", factor_canonical_torque=False,
        optimizer_exp_mode="baseline", canonical_color_kernel=False)
    engine = Engine(backend="cpu")
    try:
        result = screen(query, fx.corpus, config=_combo_config(cpu_worker=options),
            engine=engine,
            color_generator=_generator(fx),
            backend="cpu", workers=1, k=3, prefetch_depth=0)
        stats = result.stats.cpu_scan
        assert stats["canonical_color_kernel"] is False
        assert stats["factor_canonical_torque"] is False
        assert stats["optimizer_exp_mode"] == "baseline"
    finally:
        engine.close()


def test_baseline_exp_opt_out_keeps_exact_combo_optimizations(tmp_path):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    options = CPUWorkerOptions(workers=1,
        optimizer_exp_mode="baseline")
    engine = Engine(backend="cpu")
    try:
        result = screen(query, fx.corpus,
            config=_combo_config(cpu_worker=options), engine=engine,
            color_generator=_generator(fx),
            backend="cpu", workers=1, k=3, prefetch_depth=0)
        stats = result.stats.cpu_scan
        assert stats["optimizer_exp_mode"] == "baseline"
        assert stats["factor_canonical_torque"] is True
        assert stats["canonical_color_kernel"] is True
    finally:
        engine.close()


def test_unknown_exp_mode_is_not_hidden_by_auto_policy(tmp_path):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    options = CPUWorkerOptions(workers=1,
        optimizer_exp_mode="unknown")
    engine = Engine(backend="cpu")
    try:
        with pytest.raises(
                ValueError, match="optimizer_exp_mode must be baseline or rawbits"):
            screen(query, fx.corpus,
                config=_combo_config(cpu_worker=options), engine=engine,
                color_generator=_generator(fx),
                backend="cpu", workers=1, k=3, prefetch_depth=0)
    finally:
        engine.close()


def test_public_auto_combo_general_matrix_keeps_exact_fallback(tmp_path, monkeypatch):
    from roshambo3.engine.query import from_rdkit
    from tests.test_self_overlap_routes import build_inputs
    corpus, base, generator = build_inputs(tmp_path / "corpus", noncanonical=True)
    query = from_rdkit([base], color_generator=generator)
    original = cpu.score_pairs
    observed = []

    def capture(**kwargs):
        observed.append((kwargs["optimizer_exp_mode"],
                         kwargs["factor_canonical_torque"],
                         kwargs["canonical_color_kernel"]))
        return original(**kwargs)

    monkeypatch.setattr(cpu, "score_pairs", capture)
    auto_engine = Engine(backend="cpu")
    base_engine = Engine(backend="cpu")
    try:
        auto = screen(query, corpus, config=_combo_config(), engine=auto_engine,
            color_generator=generator,
            backend="cpu", workers=1, k=5, prefetch_depth=0)
        assert auto.stats.cpu_scan["optimizer_exp_mode"] == "rawbits_unclamped"
        assert auto.stats.cpu_scan["factor_canonical_torque"] is True
        assert auto.stats.cpu_scan["canonical_color_kernel"] is True
        baseline_options = CPUWorkerOptions(workers=1,
            self_overlap_policy="prepared", optimizer_exp_mode="baseline",
            factor_canonical_torque=False, canonical_color_kernel=False)
        baseline = screen(query, corpus,
            config=_combo_config(cpu_worker=baseline_options), engine=base_engine,
            color_generator=generator,
            backend="cpu", workers=1, k=5, prefetch_depth=0)
        assert observed and set(observed) == {
            ("rawbits_unclamped", True, True), ("baseline", False, False)}
        assert auto.stats.cpu_scan["color_field_builds_this_call"] == 0
        assert auto.stats.cpu_scan["color_field_selected_planes"] == []
        np.testing.assert_array_equal(auto.hits[0].gids, baseline.hits[0].gids)
        np.testing.assert_array_equal(auto.hits[0].scores, baseline.hits[0].scores)
    finally:
        auto_engine.close()
        base_engine.close()


def test_stored_vaa_conversion_metadata_matches_mode(tmp_path):
    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    query = _query_set(fx, (0,))
    engine = Engine(backend="cpu")
    generator = _generator(fx)
    try:
        shape = screen(query, fx.corpus,
            config=ScreenConfig(field_spacing_a=.5,
                cpu_worker=CPUWorkerOptions(workers=1)),
            engine=engine, backend="cpu", workers=1, k=2, prefetch_depth=0)
        assert shape.stats.cpu_scan["stored_vaa_conversion"] == (
            "float32(VAA)*float32(1/SHAPE_PREFACTOR) on verified_stored")
        assert shape.stats.cpu_scan["factor_canonical_torque"] is False
        assert shape.stats.cpu_scan["optimizer_exp_mode"] == "baseline"
        assert shape.stats.cpu_scan["canonical_color_kernel"] is False

        rescore = screen(query, fx.corpus,
            config=_combo_config(cpu_worker=CPUWorkerOptions(
                workers=1), placement=ColorPlacement.AT_POSE),
            engine=engine, color_generator=generator, backend="cpu",
            workers=1, k=2, prefetch_depth=0)
        assert rescore.stats.cpu_scan["mode"] == "rescore"
        assert rescore.stats.cpu_scan["stored_vaa_conversion"] == (
            "float32(VAA)*float32(1/SHAPE_PREFACTOR) on verified_stored")
        assert rescore.stats.cpu_scan["factor_canonical_torque"] is False
        assert rescore.stats.cpu_scan["optimizer_exp_mode"] == "baseline"
        assert rescore.stats.cpu_scan["canonical_color_kernel"] is False

        combo = screen(query, fx.corpus, config=_combo_config(), engine=engine,
            color_generator=generator, backend="cpu", workers=1, k=2,
            prefetch_depth=0)
        assert combo.stats.cpu_scan["stored_vaa_conversion"] == (
            "float32(float64(VAA)/float64(SHAPE_PREFACTOR)) on verified_stored")
        assert combo.stats.cpu_scan["optimizer_exp_mode"] == "rawbits_unclamped"
        assert combo.stats.cpu_scan["factor_canonical_torque"] is True
        assert combo.stats.cpu_scan["canonical_color_kernel"] is True
    finally:
        engine.close()

"""Portable routing checks for the manual Apple numerical qualification run."""
from types import SimpleNamespace

import pytest

from bench import numerics_end_to_end as e2e


@pytest.mark.parametrize('sampler', ['manual', 'texture16', 'texture32'])
@pytest.mark.parametrize('mode', ['shape', 'rescore', 'combo'])
def test_public_metal_panel_selects_field_and_closes_engine(monkeypatch, sampler, mode):
    import roshambo3 as r3
    import roshambo3.engine as engine_module
    from roshambo3.backend.spec import PoseSearch
    events = []
    class Engine:
        def __init__(self, **kwargs):
            events.append(kwargs)
        def close(self):
            events.append('closed')
    monkeypatch.setattr(engine_module, 'Engine', Engine)
    monkeypatch.setattr(e2e, 'prepared_queries', lambda *a, **kw: 'queries')
    def screen(queries, corpus, **kwargs):
        assert queries == 'queries'
        assert corpus == 'corpus'
        assert kwargs['config'].pose_search == PoseSearch.FIELD
        assert kwargs['config'].color == (mode != 'shape')
        # The panel must leave CUDA-only launch knobs at their defaults; the
        # platform-independent resolver still rejects explicitly unsupported
        # settings below.
        from roshambo3.backend import metal
        metal.screen_args(kwargs['config'], 32, 32, 32)
        assert isinstance(kwargs['engine'], Engine)
        return 'result'
    monkeypatch.setattr(r3, 'screen', screen)
    panel = SimpleNamespace(corpus='corpus', records={1: None}, generator=None)
    assert e2e.screen_panel(panel, backend='metal', path=sampler, mode=mode) == ('queries', 'result')
    assert events == [dict(backend='metal', backend_options=dict(
        field_sampler=sampler, color_field_sampler=sampler, color_field_min_sources=2)), 'closed']


def test_metal_panel_refuses_unsupported_analytic_optimizer():
    with pytest.raises(ValueError, match='explicit FIELD sampler'):
        e2e.screen_panel(None, backend='metal', path='analytic')


def test_cuda_panel_retains_cuda_residency_settings(monkeypatch):
    import roshambo3 as r3
    from roshambo3.backend.spec import CoordResidency

    monkeypatch.setattr(e2e, 'prepared_queries', lambda *a, **kw: 'queries')
    seen = {}
    def screen(queries, corpus, **kwargs):
        seen.update(kwargs['config'].__dict__)
        return 'result'
    monkeypatch.setattr(r3, 'screen', screen)
    panel = SimpleNamespace(corpus='corpus', records={1: None}, generator=None)
    assert e2e.screen_panel(panel, backend='cuda', path='analytic') == ('queries', 'result')
    assert seen['color_stage'] == 0
    assert seen['color_residency'] == CoordResidency.GLOBAL


def test_metal_compiled_source_must_match_checkout(monkeypatch, tmp_path):
    from bench import numerics_backends
    from roshambo3.backend import metal
    from scripts.source_stamp import source_content_hash
    root = tmp_path / 'src/roshambo3/backend/_metal_src'
    root.mkdir(parents=True)
    shader = root / 'kernels.metal'
    shader.write_text('original shader')
    compiled = source_content_hash(root)
    monkeypatch.setattr(numerics_backends, 'ROOT', tmp_path)
    monkeypatch.setattr(metal, 'build_stamp', lambda: dict(compiled_src_sha256=compiled))
    assert numerics_backends.verified_build('metal')['metal_src_sha256'] == compiled
    shader.write_text('changed shader')
    with pytest.raises(AssertionError, match='loaded extension source differs'):
        numerics_backends.verified_build('metal')

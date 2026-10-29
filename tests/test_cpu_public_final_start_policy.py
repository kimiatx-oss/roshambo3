"""Approved public default, exact all-start optout and policy-neutral ownership."""
from dataclasses import replace
import numpy as np
import pytest
from roshambo3.api import screen
from roshambo3.backend import cpu
from roshambo3.backend.spec import ColorPlacement
from roshambo3.engine import CPUWorkerOptions,Engine,ScreenConfig,R3gSource
from tests.test_cpu_worker_owned_scan import _query_set
from tests.codec_fixtures import make_codec_fixture
from tests.test_cpu_color_start_lanes import _population
from tests.test_cpu_winner_only_final import run

def test_same_binary_public_all_start_optout_is_accurate_four_finals(monkeypatch):
    kw=_population(3,'canonical')
    default,stats,_=run(kw)
    winner,winning,_=run(kw,final_start_policy='winner')
    old,control,_=run(kw,final_start_policy='all_start')
    np.testing.assert_array_equal(default,winner)
    assert winning['winner_only_final_pairs']==3
    assert winning['color_winner_rescore_evaluations']==3
    assert winning['analytic_final_shape_evaluations']==3
    assert winning['final_rescore_mode']==0
    assert control['winner_only_final_pairs']==0
    assert control['converged_objective_evaluations']==0
    assert control['color_per_start_final_evaluations']==12
    assert control['analytic_final_shape_evaluations']==12
    assert np.all(old[:,0]>=winner[:,0])

@pytest.mark.parametrize('mode',['shape','rescore','combo'])
def test_public_policy_flip_reuses_same_engine_query_and_self_cache(tmp_path,monkeypatch,mode):
    fx=make_codec_fixture(tmp_path/'corpus',color_layout='merged')
    query=_query_set(fx,(0,))
    color=mode!='shape'
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    generator=generator_from_descriptor(fx.corpus.color)
    config=ScreenConfig(color=color,mixing=.5 if color else 0.,field_spacing_a=.5,
        color_placement=ColorPlacement.IN_GRADIENT if mode=='combo' else ColorPlacement.AT_POSE)
    observed=[];original=cpu.score_pairs
    def spy(**kwargs):
        observed.append(kwargs['final_start_policy'])
        return original(**kwargs)
    monkeypatch.setattr(cpu,'score_pairs',spy)
    engine=Engine(backend='cpu')
    common=dict(backend='cpu',engine=engine,workers=1,k=3,prefetch_depth=0,
        color_generator=generator if color else None)
    try:
        first=screen(query,fx.corpus,config=config,**common)
        policies=[]
        for policy in ('all_start','winner'):
            explicit=replace(config,cpu_worker=CPUWorkerOptions(
                workers=1,final_start_policy=policy))
            result=screen(query,fx.corpus,config=explicit,**common)
            assert result.stats.cpu_scan['final_start_policy']==policy
            assert result.stats.cpu_scan['query_contexts_built']==0
            # Each facade invocation owns a NEW Source; library handles must
            # not cross that identity boundary, unlike query geometry.
            assert result.stats.cpu_scan['self_cache_hits']==0
            assert result.stats.cpu_scan['self_cache_invalidated']
            assert result.stats.cpu_scan['library_shape_self_evaluations']==0
            assert result.stats.cpu_scan['library_color_self_evaluations']==0
            policies.append(result)
        np.testing.assert_array_equal(first.hits[0].gids,policies[1].hits[0].gids)
        np.testing.assert_array_equal(first.hits[0].scores,policies[1].hits[0].scores)
        assert first.stats.cpu_scan['final_start_policy']=='winner'
        assert set(observed)=={'winner','all_start'}
        source=R3gSource(fx.corpus,color=color,color_generator=generator if color else None,
            tile=False,workers=1,prefetch_depth=0)
        try:
            initial=None
            for policy in ('winner','all_start','winner'):
                stats,hits=engine.run_worker_owned_cpu(source,query,config,
                    workers=1,k=3,color_grain=16,shape_backend='pipelined_field',
                    shape_grain='library_atom',shape_width=8,
                    self_overlap_policy='prepared',final_start_policy=policy,
                    factor_canonical_torque=mode=='combo',canonical_color_kernel=mode=='combo',
                    optimizer_exp_mode='rawbits_unclamped' if mode=='combo' else 'baseline')
                # Query contexts persist across scans; block-local self handles
                # are released after scoring, keeping retained state worker-bounded.
                assert stats.cpu_scan['self_cache_hits']==0
                assert stats.cpu_scan['self_preparation_calls']==stats.n_blocks
                assert stats.cpu_scan['query_contexts_built']==0
                assert stats.cpu_scan['library_shape_self_evaluations']==0
                assert stats.cpu_scan['library_color_self_evaluations']==0
                if initial is None:
                    initial=hits
        finally:
            # R3gSource owns no public close operation; Engine owns workers.
            pass
    finally:engine.close()
    assert config.cpu_worker is None

@pytest.mark.parametrize('backend,matrix',[
    ('scalar_analytic','canonical'),('pipelined_field','general')])
def test_explicit_policy_does_not_remove_existing_fallback(backend,matrix):
    kw=_population(3,matrix)
    winner,stats,_=run(kw,backend=backend,final_start_policy='winner')
    old,_,_=run(kw,backend=backend,final_start_policy='all_start')
    assert stats['winner_only_final_pairs']==0
    np.testing.assert_array_equal(winner,old)

def test_invalid_public_policy_refuses():
    with pytest.raises(ValueError,match='final_start_policy'):
        run(_population(1,'canonical'),final_start_policy='not-a-policy')

"""Fresh last-pose selection, stable ties, exact winner and actual final counts."""
import numpy as np
import pytest
from roshambo3.backend import cpu
from roshambo3.numerics import COORD_SCALE
from test_cpu_color_start_lanes import _population

def run(kw, *, steps=2, start=1, backend='pipelined_field', mixing=.5,
        exp_mode='rawbits_unclamped',factor=True,final_start_policy=None):
    handle=cpu.prepare_combo_shared_pose_queries(
        kw['query_coords'],kw['query_types'],kw['query_n_atoms'],
        kw['interaction_r'],kw['interaction_p'],shape_backend=backend,
        shape_grain='library_atom' if 'field' in backend else 'scalar',
        shape_width=8 if 'field' in backend else 1,
        field_spacing_angstrom=.5 if 'field' in backend else None)
    if steps==0:
        scores,stats=cpu.extension().optimize_scalar_combo(
            kw['query_coords'],kw['query_types'],kw['query_n_atoms'],
            kw['library_coords'],kw['library_counts'],kw['pair_query_row'],kw['pair_library_row'],
            kw['interaction_r'],kw['interaction_p'],0,.1,.1,start,mixing,True,False,
            color_grain=16,shape_backend=4,shape_grain=2,shape_width=8,
            shape_field_spacing=.5,shape_field_padding=8.,prepared_shared_pose_queries=handle,
            canonical_color_kernel=True,factor_canonical_torque=True,optimizer_exp_mode='rawbits_unclamped',
            final_start_policy='default' if final_start_policy is None else final_start_policy)
        return scores,stats,handle
    scores,stats=cpu.score_pairs(**kw,color=True,color_mode='combo_optimal',
        color_grain=16 if start==1 else 0,steps=steps,lr_t=.1,
        budget_override='bounded fixed-pose/update fixture',start_mode=start,mixing=mixing,
        shape_backend=backend,shape_grain='library_atom' if 'field' in backend else 'scalar',
        shape_width=8 if 'field' in backend else 1,
        shape_field_spacing_angstrom=.5 if 'field' in backend else None,
        prepared_shared_pose_queries=handle,canonical_color_kernel=start==1,
        factor_canonical_torque=factor and start==1,optimizer_exp_mode=exp_mode if start==1 else 'baseline',
        with_census=True,final_start_policy=final_start_policy)
    return scores,stats,handle

@pytest.mark.parametrize('pairs',[1,2,3,9])
def test_actual_final_counts_and_exact_recorded_winner(pairs,monkeypatch):
    monkeypatch.setenv('ROSHAMBO3_FINAL_SELECTION_TRACE','1')
    kw=_population(pairs,'canonical')
    scores,stats,handle=run(kw)
    assert np.isfinite(scores).all()
    if not cpu.extension().__final_winner_only__:
        assert stats['color_per_start_final_evaluations']==pairs*4
        assert stats['color_winner_rescore_evaluations']==0
        return
    assert stats['winner_only_final_pairs']==pairs
    assert stats['converged_objective_evaluations']==pairs*4
    assert stats['analytic_final_shape_evaluations']==pairs
    assert stats['color_winner_rescore_evaluations']==pairs
    assert stats['color_per_start_final_evaluations']==0
    chosen=np.argmax(stats['converged_objectives'],axis=1)
    np.testing.assert_array_equal(chosen,stats['winning_starts'])
    for row,start in enumerate(chosen):
        pose=np.ascontiguousarray(stats['converged_pose_bytes'][row,start])
        np.testing.assert_array_equal(scores[row,9:13],pose[12:16])
        np.testing.assert_array_equal(scores[row,13:16],pose[9:12]/np.float32(COORD_SCALE))
        exact=cpu.extension()._replay_fixed_pose_cross(handle,int(kw['pair_query_row'][row]),
            kw['library_coords'][kw['pair_library_row'][row]],
            kw['library_counts'][kw['pair_library_row'][row]],pose)
        np.testing.assert_array_equal(exact,scores[row,[3,4]])

def test_last_update_changes_the_recorded_selection_pose(monkeypatch):
    monkeypatch.setenv('ROSHAMBO3_FINAL_SELECTION_TRACE','1')
    kw=_population(3,'canonical')
    _,zero,_=run(kw,steps=0)
    _,one,_=run(kw,steps=1)
    if not cpu.extension().__final_winner_only__:pytest.skip('winner-only private control off')
    assert not np.array_equal(zero['converged_pose_bytes'],one['converged_pose_bytes'])
    assert not np.array_equal(zero['converged_objectives'],one['converged_objectives'])

def test_fixed_pose_replay_rejects_out_of_bounds_color_counts():
    kw=_population(1,'canonical')
    _,_,handle=run(kw)
    counts=kw['library_counts'][0].copy()
    counts[1]=255
    pose=np.zeros(16,dtype=np.float32)
    with pytest.raises(ValueError,match='shape/color counts'):
        cpu.extension()._replay_fixed_pose_cross(handle,0,kw['library_coords'][0],counts,pose)

@pytest.mark.parametrize('exp_mode,factor',[
    ('baseline',False),('baseline',True),('rawbits',False),
    ('rawbits',True),('rawbits_unclamped',True)])
def test_fresh_objective_replays_actual_search_exp_and_reduction(exp_mode,factor,monkeypatch):
    monkeypatch.setenv('ROSHAMBO3_FINAL_SELECTION_TRACE','1')
    kw=_population(3,'canonical')
    _,stats,handle=run(kw,exp_mode=exp_mode,factor=factor)
    if not cpu.extension().__final_winner_only__:
        pytest.skip('winner-only private control off')
    different_from_value_only=False
    for row in range(3):
        args=(handle,int(kw['pair_query_row'][row]),
            kw['library_coords'][kw['pair_library_row'][row]],
            kw['library_counts'][kw['pair_library_row'][row]],
            np.ascontiguousarray(stats['converged_pose_bytes'][row]),
            np.ascontiguousarray(stats['converged_objective_self'][row]),
            .5,exp_mode,factor)
        search=cpu.extension()._replay_fixed_pose_search(*args,True)
        value_only=cpu.extension()._replay_fixed_pose_search(*args,False)
        np.testing.assert_array_equal(stats['converged_objectives'][row],search)
        different_from_value_only |= not np.array_equal(search,value_only)
    # This fixture must discriminate the old buggy gradient=false rawbits call,
    # not merely replay a coincidentally equal exponential/reduction schedule.
    if exp_mode!='baseline':assert different_from_value_only

def test_equal_search_objectives_choose_earliest_start(monkeypatch):
    monkeypatch.setenv('ROSHAMBO3_FINAL_SELECTION_TRACE','1')
    kw=_population(1,'canonical')
    kw['query_coords'][...,:3]=0;kw['library_coords'][...,:3]=0
    _,stats,_=run(kw,steps=0)
    assert stats['winning_starts'].tolist()==[0]
    if cpu.extension().__final_winner_only__:
        values=stats['converged_objectives'][0]
        np.testing.assert_array_equal(values,np.repeat(values[0],4))

@pytest.mark.parametrize('matrix,backend,start',[
    ('general','pipelined_field',1),('canonical','scalar_analytic',1),
    ('canonical','pipelined_field',0),('canonical','pipelined_field',2)])
def test_explicit_unsupported_routes_keep_all_start_control(matrix,backend,start):
    _,stats,_=run(_population(3,matrix),backend=backend,start=start)
    assert stats['winner_only_final_pairs']==0
    assert stats['converged_objective_evaluations']==0
    assert stats['color_winner_rescore_evaluations']==0

def test_shape_rescore_cannot_use_color_to_choose_start():
    kw=_population(3,'canonical')
    fields=cpu.prepare_combo_shape_evaluators(kw['query_coords'],kw['query_n_atoms'],
        shape_backend='pipelined_field',shape_grain='library_atom',shape_width=8,
        field_spacing_angstrom=.5,field_padding_angstrom=8.)
    base={k:kw[k] for k in ['query_coords','query_n_atoms','library_coords',
        'library_counts','pair_query_row','pair_library_row']}
    knobs=dict(steps=2,start_mode=1,field_h=.5,field_pad=8.,field_width=8,
        field_unroll=0,field_pipeline=True,prepared_query_fields=fields)
    shape=cpu.score_pairs(**base,**knobs)
    for factor in [.1,10.]:
        rescore,_=cpu.extension().rescore_scalar_color(
            kw['query_coords'],kw['query_types'],kw['query_n_atoms'],
            kw['library_coords'],kw['library_counts'],kw['pair_query_row'],kw['pair_library_row'],
            kw['interaction_r'],kw['interaction_p']*np.float32(factor),
            shape,.5,False,False,False,0)
        np.testing.assert_array_equal(shape[:,1],rescore[:,1])
        np.testing.assert_array_equal(shape[:,9:20],rescore[:,9:20])

def test_unsafe_canonical_inputs_keep_existing_analytic_fallback():
    kw=_population(1,'canonical')
    kw['library_coords'][0,2:8,:3]=np.float32(2e9)
    rows,stats,_=run(kw)
    assert np.isfinite(rows).all()
    assert stats['winner_only_final_pairs']==0
    assert stats['color_per_start_final_evaluations']==4

def test_empty_color_planes_stay_finite_with_single_accurate_final():
    kw=_population(3,'canonical')
    kw['query_types'][:]=0;kw['library_counts'][:,1:]=0
    kw['query_coords'][:,2:]=0;kw['library_coords'][:,2:]=0
    rows,stats,_=run(kw)
    assert np.isfinite(rows).all()
    np.testing.assert_array_equal(rows[:,[2,4,7,8]],np.zeros((3,4),np.float32))
    if cpu.extension().__final_winner_only__:
        assert stats['color_winner_rescore_evaluations']==3


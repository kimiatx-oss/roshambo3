"""Lifetime receipts and actual current-canonical field/remainder execution."""
import numpy as np
import pytest
from roshambo3.backend import cpu
from roshambo3.reference.color import MODE_COMBO_OPTIMAL
from test_cpu_shared_pose_contract import _inputs, _prepare

@pytest.mark.parametrize('backend',[1,2,3])
def test_retained_field_generation_and_pack_counts(backend):
    q,qt,qn,lib,counts,r,p=_inputs(canonical=True)
    codes=np.array([0,backend,0],dtype=np.int32)
    before=cpu.extension().ColorFieldBundle.construction_count
    h=cpu.prepare_color_fields(query_coords=q,query_types=qt,query_n_atoms=qn,
        interaction_r=r,interaction_p=p,backend_codes=codes)
    generation=list(h.bundle_generations)
    packs=h.bundle_pack_counts
    assert len(set(generation))==len(q)
    assert cpu.extension().ColorFieldBundle.construction_count==before+len(q)
    kw=dict(query_coords=q,query_types=qt,query_n_atoms=qn,library_coords=lib,
        library_counts=counts,interaction_r=r,interaction_p=p,
        pair_query_row=np.array([0,1,0],dtype=np.int32),
        pair_library_row=np.array([0,1,2],dtype=np.int32),
        color=True,color_mode=MODE_COMBO_OPTIMAL,color_grain=16,
        steps=2,start_mode=1,mixing=.5,shape_backend='scalar_analytic',
        shape_grain='scalar',shape_width=1,pose_contract='shared_fp32_reduced_v1',
        prepared_shared_pose_queries=_prepare((q,qt,qn,lib,counts,r,p)),
        canonical_color_kernel=True,factor_canonical_torque=True,
        optimizer_exp_mode='rawbits_unclamped',color_field_backends=codes,
        prepared_color_fields=h,with_census=True)
    first,stats=cpu.score_pairs(**kw)
    second,_=cpu.score_pairs(**kw)
    np.testing.assert_array_equal(first,second)
    assert np.isfinite(first).all()
    assert h.bundle_generations==generation and h.bundle_pack_counts==packs
    assert cpu.extension().ColorFieldBundle.construction_count==before+len(q)
    assert stats['color_field_prepared_handle_used']
    assert stats['color_field_build_seconds']==stats['color_field_pack_seconds']==0
    if cpu.extension().__color_field_hybrid__:
        assert stats['color_field_current_four_start_field_groups']>0
        assert stats['color_field_current_four_start_remainder_groups']>0
        assert stats['color_field_interpolated_samples']>0
    assert stats['color_field_samples']==stats['color_field_interpolated_samples']+stats['color_field_analytic_selected_samples']+stats['color_field_analytic_fallback_samples']

@pytest.mark.parametrize('backend',[1,2,3])
def test_current_canonical_outside_group_keeps_analytic_remainder(backend):
    if not cpu.extension().__color_field_hybrid__:
        pytest.skip('isolated arithmetic control does not enable current mixed route')
    data=_inputs(canonical=True)
    q,qt,qn,lib,counts,r,p=data
    lib=lib.copy()
    lib[:,1:,:3]=100
    codes=np.array([0,backend,0],dtype=np.int32)
    h=cpu.prepare_color_fields(query_coords=q,query_types=qt,query_n_atoms=qn,
        interaction_r=r,interaction_p=p,backend_codes=codes)
    common=dict(query_coords=q,query_types=qt,query_n_atoms=qn,library_coords=lib,
        library_counts=counts,interaction_r=r,interaction_p=p,
        pair_query_row=np.array([0,1,0],dtype=np.int32),
        pair_library_row=np.array([0,1,2],dtype=np.int32),
        color=True,color_mode=MODE_COMBO_OPTIMAL,color_grain=16,
        steps=2,start_mode=1,mixing=.5,shape_backend='scalar_analytic',
        shape_grain='scalar',shape_width=1,pose_contract='shared_fp32_reduced_v1',
        prepared_shared_pose_queries=_prepare(data),canonical_color_kernel=True,
        factor_canonical_torque=True,optimizer_exp_mode='rawbits_unclamped',with_census=True)
    analytic,_=cpu.score_pairs(**common)
    field,stats=cpu.score_pairs(**common,color_field_backends=codes,prepared_color_fields=h)
    np.testing.assert_array_equal(field,analytic)
    assert stats['color_field_out_of_grid_fallback_samples']>0
    assert stats['color_field_interpolated_samples']==0
    assert stats['color_field_current_four_start_remainder_groups']>0

@pytest.mark.parametrize('backend',[1,2,3])
def test_nonfinite_inverse_geometry_stays_exact_and_rejects_nan(backend):
    coefficient=np.zeros((2,2),dtype=np.float64)
    decay=np.zeros_like(coefficient)
    coefficient[1,1]=1;decay[1,1]=-.5
    bundle=cpu.extension().ColorFieldBundle(np.zeros((1,3),dtype=np.float32),
        np.ones(1,dtype=np.int32),coefficient,decay,
        np.array([0,backend],dtype=np.int32),1e-40,1e-40)
    assert bundle.field_bytes==0
    result=bundle.sample(1,np.zeros((1,3),dtype=np.float32))
    np.testing.assert_array_equal(result['value'],[1.])
    assert not result['used_field'][0]
    with pytest.raises(ValueError,match='finite'):
        bundle.sample(1,np.array([[0,np.nan,0]],dtype=np.float32))

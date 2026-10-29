"""Focused native ARM dispatch, isolation and exact-fallback assertions."""
import platform

import numpy as np
import pytest

from roshambo3.backend import cpu
from test_cpu_shared_pose_contract import _inputs, _prepare, _score
from test_cpu_optimizer_exp_modes import test_concurrent_approximate_and_baseline_calls_are_isolated as _concurrent_check

pytestmark=pytest.mark.skipif(platform.machine().lower() not in ('aarch64','arm64'),
                            reason='native NEON screening acceptance')

def test_native_retained_color_executes_neon_and_accurate_optout():
    data=_inputs(canonical=True)
    prepared=_prepare(data)
    kwargs=dict(grain=16,use_plan=False,reuse=False,canonical_color_kernel=True)
    baseline,base_stats=_score(data,prepared,**kwargs)
    retained,stats=_score(data,prepared,**kwargs,factor_torque=True,optimizer_exp_mode='rawbits')
    assert np.isfinite(baseline).all() and np.isfinite(retained).all()
    assert base_stats['color_start_lane_vector_evaluations']>0
    assert stats['color_start_lane_vector_evaluations']>0
    assert stats['color_start_lane_scalar_fallback_pairs']==0
    stamp=cpu.color_build_stamp(16)
    assert stamp['color_isa']=='NEON' and stamp['color_vector_width']==4
    again,_=_score(data,prepared,**kwargs)
    np.testing.assert_array_equal(again,baseline)

def test_native_call_local_exp_and_factoring_are_thread_isolated():
    _concurrent_check(True)

def test_native_general_matrix_uses_exact_fallback():
    data=_inputs(canonical=False)
    prepared=_prepare(data)
    kwargs=dict(grain=16,use_plan=False,reuse=False,canonical_color_kernel=True)
    exact,stats=_score(data,prepared,**kwargs)
    approximate,_=_score(data,prepared,**kwargs,factor_torque=True,optimizer_exp_mode='rawbits')
    assert stats['color_start_lane_vector_evaluations']==0
    assert stats['color_start_lane_scalar_fallback_pairs']>0
    np.testing.assert_array_equal(approximate,exact)


def test_native_query_eligibility_cache_is_plane_local_and_refreshed():
    # Shape uses atom0. Feature1 deliberately exceeds the color range gate.
    unsafe = _inputs(canonical=True)
    unsafe[0][:, 1, 0] = 2.e9
    kwargs = dict(grain=16, use_plan=False, reuse=False,
                  canonical_color_kernel=True, factor_torque=True,
                  optimizer_exp_mode='rawbits_unclamped')
    exact, stats = _score(unsafe, _prepare(unsafe), **kwargs)
    reference, _ = _score(unsafe, _prepare(unsafe), grain=16, use_plan=False,
                          reuse=False, canonical_color_kernel=False)
    assert stats['color_start_lane_vector_evaluations'] == 0
    assert stats['color_start_lane_scalar_fallback_pairs'] > 0
    np.testing.assert_array_equal(exact, reference)
    # The next packing of retained worker scratch must refresh cached results.
    ordinary = _inputs(canonical=True)
    _, stats = _score(ordinary, _prepare(ordinary), **kwargs)
    assert stats['color_start_lane_vector_evaluations'] > 0
    assert stats['color_start_lane_scalar_fallback_pairs'] == 0
    # An unsafe query plane without any matching target features is inactive.
    inactive = _inputs(canonical=True)
    inactive[0][0, 2, 0] = 2.e9
    inactive[4][:, 2] = 0
    _, stats = _score(inactive, _prepare(inactive), **kwargs)
    assert stats['color_start_lane_vector_evaluations'] > 0
    assert stats['color_start_lane_scalar_fallback_pairs'] == 0

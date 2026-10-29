"""Independent-pair final arithmetic, real tails and accurate fallback."""
import numpy as np
import pytest
from roshambo3.backend import cpu
from roshambo3.numerics import COORD_SCALE

def fixture():
    counts=np.array([[1,1],[3,1],[7,2],[9,0],[0,0]],np.int32)
    q=np.zeros((1,16,3),np.float32)
    b=np.zeros((5,16,3),np.float32)
    for j in range(5):
        b[j,:,0]=np.arange(16,dtype=np.float32)*.13
        b[j,:,1]=-.07*j
    return dict(query_reduced=q,query_counts=np.array([[1,1]],np.int32),
        library_physical=b,library_counts=counts,pair_query=np.zeros(5,np.int32),
        pair_library=np.arange(5,dtype=np.int32),matrices=np.tile(
            np.eye(3,dtype=np.float32)*np.float32(COORD_SCALE),(5,1,1)),
        translations=np.zeros((5,3),np.float32),coefficients=np.diag([0.,1.9687012432153024]),
        physical_decays=np.full((2,2),-.5,np.float64),canonical=True)

@pytest.mark.parametrize('mode',range(4))
def test_real_pair_tails_and_counted_timed_agreement(mode):
    f=fixture();native=cpu.extension()
    reference=native.final_rescore_fixed(**f,mode=0)
    value=native.final_rescore_fixed(**f,mode=mode,repeats=2)
    for channel in ('shape','color'):
        np.testing.assert_allclose(value[channel],reference[channel],rtol=3e-5,atol=2e-4)
    assert value['pairs']==5 and value['timing_has_counters'] is False
    assert value['counted_shape_max_abs_drift']==value['counted_color_max_abs_drift']==0
    assert value['shape'][-1]==value['color'][-1]==0

@pytest.mark.parametrize('mode',range(4))
def test_nonfinite_after_finite_unsafe_still_refuses(mode):
    f=fixture();f['library_physical'][1,0,0]=2e9
    f['library_physical'][1,2,2]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='nonfinite'):
        cpu.extension().final_rescore_fixed(**f,mode=mode)

@pytest.mark.parametrize('mode',range(4))
def test_general_color_exact_fallback(mode):
    f=fixture();f['canonical']=False;f['coefficients'][1,0]=-.3
    reference=cpu.extension().final_rescore_fixed(**f,mode=0)
    result=cpu.extension().final_rescore_fixed(**f,mode=mode)
    for channel in ('shape','color'):np.testing.assert_array_equal(result[channel],reference[channel])
    if mode:assert result['accurate_fallback_pairs']==5

def test_every_mode_counts_only_the_color_terms_it_issues():
    # Two color types under a canonical plan: every final route visits only the
    # diagonal planes (type1 x type1, type2 x type2). The accurate route once
    # multiplied counts over the plan's active_interactions(), which also lists
    # the zero-coefficient type1 x type2 pair, and reported 14 instead of 10.
    q=np.zeros((1,16,3),np.float32);q[0,:,0]=np.arange(16)*.11
    b=np.zeros((2,16,3),np.float32)
    for j in range(2):
        b[j,:,0]=np.arange(16)*.13;b[j,:,1]=-.07*j
    f=dict(query_reduced=q,query_counts=np.array([[2,1,2]],np.int32),
        library_physical=b,library_counts=np.array([[4,2,3],[3,0,1]],np.int32),
        pair_query=np.zeros(2,np.int32),pair_library=np.arange(2,dtype=np.int32),
        matrices=np.tile(np.eye(3,dtype=np.float32)*np.float32(COORD_SCALE),(2,1,1)),
        translations=np.zeros((2,3),np.float32),
        coefficients=np.diag([0.,1.9687012432153024,1.9687012432153024]),
        physical_decays=np.full((3,3),-.5,np.float64),canonical=True)
    diagonal=(1*2+2*3)+(1*0+2*1)
    for mode in range(4):
        result=cpu.extension().final_rescore_fixed(**f,mode=mode,repeats=1)
        assert result['color_terms']==diagonal,mode
        # SIMD modes also count padded lanes as issued; scalar modes do not.
        if mode<2:assert result['color_issued_lanes']==diagonal,mode

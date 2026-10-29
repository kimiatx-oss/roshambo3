import numpy as np
import pytest
from bench.numerics_color_controls import instrument
from bench.numerics_optimizer_basins import compare_basins, polish


def test_color_instrument_targets_only_search_exponential():
    source='''double self() {return std::exp(0.);}
ColorEvaluation color_value_gradient_pose(int a) {
return std::exp(-a);
}
double final() {return std::exp(0.);}
'''
    actual=instrument(source)
    assert actual.count('color_control::exp(')==1
    assert actual.count('std::exp(')==2
    with pytest.raises(ValueError,match='seam changed'):
        instrument(source.replace('return std::exp(-a);','return a;'))


def asymmetric_case():
    shape=np.array([[0.,0.,0.],[1.,.2,0.],[.1,1.5,.3],[-.2,.4,1.2]])
    return dict(a=shape,b=shape.copy(),ac=shape[:2],bc=shape[:2].copy(),
        at=np.array([1,2]),bt=np.array([1,2]),r=np.diag([0.,1.,1.]),p=np.diag([0.,1.,1.]))


@pytest.mark.parametrize('objective',['shape','combo'])
def test_independent_polish_separates_raw_displacement_from_attraction_basin(objective):
    anchor=np.zeros((1,20));anchor[:,9]=anchor[:,16]=1
    candidate=anchor.copy();candidate[:,13]=.2
    result=compare_basins([asymmetric_case()],anchor,candidate,objective)
    assert result['unresolved']==0
    assert result['different_rate']==0
    assert result['records'][0]['endpoint_rms_angstrom']<result['endpoint_neighborhood_angstrom']


def test_unfinished_polish_remains_unresolved():
    row=np.zeros(20);row[9]=row[16]=1;row[13]=3.
    result,_,_=polish(asymmetric_case(),row,'combo',maxiter=0)
    assert not result['stationary']

"""Operation-derived shape mean controls, not stored native output fixtures."""
import math
import numpy as np
import pytest
from roshambo3.reference.field_intervals import Interval
from roshambo3.reference.shape_intervals import (
    _reduce,constant,overlap_interval,aggregate_report,prefactor_plant_model)
from bench.numerics_shape_mean import PATHS,panel,preregistration,libm_identity


@pytest.fixture(scope='module')
def declared():
    cases=panel()
    return cases,preregistration(cases)


@pytest.mark.parametrize('grain',['scalar','query_atom','library_atom'])
def test_declared_reduction_encloses_exact_binary32_execution(grain):
    terms=np.random.default_rng(277).uniform(0,1,(11,13)).astype(np.float32)
    if grain=='scalar':
        actual=float(np.cumsum(terms.T.ravel(),dtype=np.float32)[-1])
    else:
        streams=[[] for _ in range(8)]
        for j in range(terms.shape[1]):
            for i in range(terms.shape[0]):
                streams[(i if grain=='query_atom' else j)%8].append((j if grain=='query_atom' else j//8,i,terms[i,j]))
        lanes=np.array([np.cumsum([item[2] for item in sorted(stream)],dtype=np.float32)[-1] for stream in streams],np.float32)
        four=(lanes[:4]+lanes[4:]).astype(np.float32)
        actual=float(np.float32(np.float32(four[0]+four[2])+np.float32(four[1]+four[3])))
    expected=_reduce(Interval(terms),grain)
    assert float(expected.lo)==actual
    lo=_reduce(Interval(np.nextafter(terms,np.float32(-np.inf)),np.nextafter(terms,np.float32(np.inf))),grain)
    assert float(lo.lo)<=float(expected.lo)<=float(lo.hi)
    # Exact singleton endpoint operations stay singleton; no n*u padding was
    # added to reproduce the generic row-implied mean radius.
    assert expected.lo==expected.hi


def test_preregistered_window_uses_only_declared_bounds(declared):
    cases,record=declared
    assert len(cases)==36 and record['smaller_bias'] is not None
    amplitude=record['smaller_bias']['amplitude']
    for rows in record['paths'].values():
        for sign in (-1,1):
            plants=[prefactor_plant_model(row,1+sign*amplitude) for row in rows]
            assert all(row['physical']-row['row_bound']<=row['lower']<=row['upper']<=row['physical']+row['row_bound'] for row in plants)
            if sign>0:
                assert math.fsum(row['lower']-row['physical'] for row in plants)>math.fsum(row['upper']-row['physical'] for row in rows)
            else:
                assert math.fsum(row['upper']-row['physical'] for row in plants)<math.fsum(row['lower']-row['physical'] for row in rows)
        assert any(row['term_count']>1000 for row in rows)


def test_correlated_repetitions_do_not_narrow_mean_interval():
    a=np.array([[.3,-.1,.2],[.9,.2,-.4]],np.float32)
    model=overlap_interval(a,a)
    value=(model['lower']+model['upper'])/2
    one=aggregate_report([value],[model])
    for count in (2,32,100):
        repeated=aggregate_report([value]*count,[model]*count)
        np.testing.assert_allclose(repeated['mean_interval'],one['mean_interval'],rtol=0,atol=1e-15)
        assert repeated['mean_pass']


def test_domain_and_observation_fail_closed():
    for a in (np.zeros((0,3),np.float32),np.ones((2,3)),np.full((2,3),np.nan,np.float32),np.full((2,3),33,np.float32)):
        with pytest.raises(ValueError):overlap_interval(a,a)
    a=np.zeros((1,3),np.float32)
    for matrix in (np.zeros((2,2)),np.full((3,3),np.nan),np.full((3,3),3.)):
        with pytest.raises(ValueError):overlap_interval(a,a,matrix)
    for values in ([],[np.nan],[[1]]):
        with pytest.raises(ValueError):aggregate_report(values,[overlap_interval(a,a)])


def test_scalar_authority_refuses_an_unreadable_mapped_libm(tmp_path):
    from bench.numerics_shape_mean import libm_image_sha256
    with pytest.raises(RuntimeError, match="not a readable file for the explicit scalar authority"):
        libm_image_sha256(tmp_path / "shared-cache-libm")


@pytest.mark.parametrize('path',['cpu_scalar','cpu_query_atom','cpu_library_atom'])
def test_native_cpu_affine_shape_directed_mean(path,declared):
    from roshambo3.backend import cpu
    if not cpu.available():pytest.skip('CPU extension unavailable')
    from bench.numerics_backends import verified_build,cpu_evaluator_capability
    verified_build('cpu')
    grain,exponential,backend,code,width=PATHS[path]
    capability=cpu_evaluator_capability(backend,code,width)
    if not capability['available']:pytest.skip(capability['reason'])
    if path=='cpu_scalar':
        try:libm_identity()
        except RuntimeError as error:pytest.skip(str(error))
    cases,record=declared;values=[]
    for case in cases:
        result=cpu.extension().numerics_shape_evaluate(case['query'],case['library'],backend,code,width,.25,
            physical_matrix=case['matrix'].astype(np.float32),physical_translation=case['translation'].astype(np.float32))
        assert result['backend']==capability['stamp']['backend']
        assert result['exponential']==capability['stamp']['exponential']
        values.append(result['overlap_volume'])
    report=aggregate_report(values,record['paths'][path])
    assert report['rows_pass'] and report['mean_pass'],report


def test_native_cuda_arbitrary_coordinate_shape_mean(declared):
    from roshambo3.backend import cuda
    if not cuda.available():pytest.skip('CUDA extension unavailable')
    if cuda.extension().device_count()<1:pytest.skip('CUDA device unavailable')
    from bench.numerics_backends import verified_build,cuda_shape_overlap
    verified_build('cuda')
    cases,record=declared;values=[]
    for case in cases:
        b=(case['library'].astype(float)@case['matrix'].T+case['translation']).astype(np.float32)
        values.append(cuda_shape_overlap(cuda.extension(),case['query'],b))
    report=aggregate_report(values,record['paths']['cuda_direct'])
    assert report['rows_pass'] and report['mean_pass'],report


def test_scalar_authority_requires_exact_version_image_route_and_code():
    from bench.numerics_shape_mean import libm_authority
    for version,digest,offset,code in (
        ('2.39','fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0',514880,
         '3f5f3cc5a6e1a7ee90ef94bb53312b093df40b1a58f09d2ba2edad9f59947fd7'),
        ('2.41','a8afb2d1f4a0c5e14c0fc82f794ec839eafe29738907aa92895410f57630bbe2',270336,
         '2e454ff95c036e1e05ba348dbe82e8842f5f134a5361cd6daaa9fbfce52ee542')):
        assert 'two-ULP' in libm_authority(('glibc',version),digest,offset,code)
        for values in ((('glibc','0'),digest,offset,code),(('glibc',version),'0'*64,offset,code),
                       (('glibc',version),digest,offset+4,code),(('glibc',version),digest,offset,'0'*64)):
            with pytest.raises(RuntimeError):libm_authority(*values)

"""Portable mathematical controls; no native FIELD qualification is emitted."""
import numpy as np
import pytest
from roshambo3.reference.field_intervals import Interval, gaussian_prefilter_intervals, assert_interval


def test_signed_interval_operations_include_separate_and_contracted_results():
    rng=np.random.default_rng(272)
    a,b,c=rng.uniform(-4,4,(3,1024)).astype(np.float32)
    value=Interval(a)*Interval(b)+Interval(c)
    separate=(a*b+c).astype(float)
    contracted=(a.astype(float)*b.astype(float)+c).astype(np.float32).astype(float)
    for actual in (separate,contracted):
        assert np.all(actual>=value.lo)
        assert np.all(actual<=value.hi)


def test_gaussian_single_center_value_and_gradient_have_correct_symmetry():
    q=np.array([[0,0,0,1]],np.float32)
    expected=gaussian_prefilter_intervals(q,[-.5]*3,.25,[4]*3)
    center=2
    # At d=0 the Laplacian is -6 ln(2); prefilter raises F by hs² ln(2)/2.
    value=np.float16(1+.25**2*np.log(2)/2)
    assert expected.lo[center,center,center,0]<=value<=expected.hi[center,center,center,0]
    assert np.all(expected.lo[center,center,center,1:]<=0)
    assert np.all(expected.hi[center,center,center,1:]>=0)
    np.testing.assert_allclose(expected.lo[...,0],expected.lo[::-1,::-1,::-1,0],atol=.001)


@pytest.mark.parametrize('shape,factor',[(True,1.),(False,.7),(False,1.3)])
def test_prefilter_intervals_reject_width_plant_without_changing_bounds(shape,factor):
    query=np.random.default_rng(273).uniform(-1,1,(5,4)).astype(np.float32)
    query[:,3]=1
    clean=gaussian_prefilter_intervals(query,[-1]*3,.25,[8]*3,shape=shape,factor=factor)
    # Source-side invocation control, distinct from an executed native plant.
    shifted=query.copy(); shifted[:,:3]*=np.float32(1.0001)
    plant=gaussian_prefilter_intervals(shifted,[-1]*3,.25,[8]*3,shape=shape,factor=factor)
    actual=(plant.lo+plant.hi)/2
    with pytest.raises(AssertionError,match='outside directed'):
        assert_interval(actual,clean)


def test_interval_shape_and_nonfinite_fail_closed():
    expected=Interval(np.zeros((2,4)))
    for actual in (np.zeros((4,2)),np.full((2,4),np.nan)):
        with pytest.raises(AssertionError): assert_interval(actual,expected)


@pytest.mark.parametrize('shape,factor',[(True,1.),(False,.8),(False,1.3)])
def test_direct_intervals_enclose_independent_physical_value_and_gradient(shape,factor):
    from roshambo3.reference.overlap import ALPHA
    scale=np.sqrt(ALPHA/(2*np.log(2)))
    q=np.array([[-.3,.2,.7,.6],[.4,-.6,.1,1.]],np.float32)
    origin=np.array([-.5]*3,np.float32);spacing=.25;cells=[4]*3
    interval=gaussian_prefilter_intervals(q,origin,spacing,cells,shape=shape,factor=factor)
    z,y,x=np.indices((5,5,5)); points=np.stack((x,y,z),axis=-1)*spacing+origin
    # Evaluate the continuous fp64 physical Gaussian and its Laplacian. The
    # gradient is independently a central difference of the scalar prefilter.
    width=ALPHA/2*(1 if shape else np.float32(factor))
    h=spacing/scale
    def scalar(reduced):
        d2=np.sum((reduced[...,None,:]/scale-q[:,:3].astype(float))**2,axis=-1)
        f=np.exp(-width*d2)*q[:,3]
        return np.sum(f*(1-h*h/12*(4*width*width*d2-6*width)),axis=-1)
    values=[scalar(points)]
    for axis in range(3):
        delta=np.eye(3)[axis]*1e-5
        values.append((scalar(points+delta)-scalar(points-delta))/(2e-5))
    expected=np.stack(values,axis=-1).astype(np.float16).astype(float)
    assert_interval(expected,interval)


@pytest.mark.numerics_coverage(backend='cuda',path='field',check='fixed_pose',channel='shape')
@pytest.mark.numerics_coverage(backend='cuda',path='field',check='fixed_pose',channel='color')
def test_native_gaussian_builders_and_bounded_full3D_samples(numerics_coverage):
    try:
        from roshambo3 import _roshambo3_cuda as ext
    except ImportError:
        pytest.skip('CUDA extension unavailable; Gaussian field fixed pose not run')
    try:
        if ext.device_count()<1: pytest.skip('CUDA device unavailable; Gaussian field fixed pose not run')
    except RuntimeError as error:
        pytest.skip('CUDA runtime unavailable: '+str(error))
    from bench.numerics_backends import verified_build,sampler_panel,check_sampler
    from bench.numerics_cuda_field_arithmetic import panels
    from roshambo3.reference.field_intervals import convex_filter_intervals,gaussian_physical_values,CUDA_LINEAR_FILTER_SOURCE
    build=verified_build('cuda')
    origin=np.array([-1]*3,np.float32);spacing=np.float32(.25);cells=[8]*3
    vertices=np.array([[x,y,z] for z in range(9) for y in range(9) for x in range(9)],np.float32)
    points=np.random.default_rng(275).integers(1,2048,(512,3)).astype(np.float32)/256
    points=(origin+spacing*points).astype(np.float32)
    evidence={}
    for name,q,shape,factor in panels():
        bounds=gaussian_prefilter_intervals(q,origin,spacing,cells,shape=shape,factor=factor)
        bits=ext.numerics_build_field(q,origin,spacing,cells,shape,factor)
        nodes=bits.view(np.float16)
        node_report=assert_interval(nodes,bounds)
        actual=ext.numerics_sample_field(bits,(origin+spacing*vertices).astype(np.float32),origin,spacing).reshape(nodes.shape)
        assert_interval(actual,bounds)
        sampled=ext.numerics_sample_field(bits,points,origin,spacing)
        hull=convex_filter_intervals(bounds,points,origin,spacing)
        sample_report=assert_interval(sampled,hull)
        reference=gaussian_physical_values(q,points,shape=shape,factor=factor)
        row_bounds=np.maximum(np.abs(reference-hull.lo[:,0]),np.abs(hull.hi[:,0]-reference))
        residual=sampled[:,0].astype(float)-reference
        assert np.all(np.abs(residual)<=row_bounds)
        assert abs(residual.mean())<=row_bounds.mean()
        evidence[name]=dict(nodes=node_report,samples=sample_report,
            fp64_continuous_mean_residual=float(residual.mean()),mean_sample_bound=float(row_bounds.mean()),
            max_sample_bound=float(row_bounds.max()),scope='seeded weighted Gaussian component; no optimizer or complete color matrix')
    ramp,ramp_points=sampler_panel()
    ramp_origin=np.array([2,-3,4],np.float32);ramp_spacing=np.float32(.5)
    check_sampler(ext.numerics_sample_field(ramp.view(np.uint16),(ramp_origin+ramp_spacing*ramp_points).astype(np.float32),ramp_origin,ramp_spacing),ramp_points)
    numerics_coverage.stamp(build=build,arithmetic=dict(
        name='Gaussian_prefilter_directed_intervals+convex_full3D_ranges+axis8',
        source=CUDA_LINEAR_FILTER_SOURCE,combined_weight_model='range preserving documented convex filter',
        sampler_weight_bits=8,mean='directed Gaussian node means plus conservative fractional sample means'),evidence=evidence)


def test_conservative_filter_bounds_all_corner_allocations_and_border():
    from roshambo3.reference.field_intervals import convex_filter_intervals
    rng=np.random.default_rng(276)
    nodes=rng.uniform(-2,2,(2,2,2,4)).astype(np.float16).astype(float)
    points=np.array([[.125,.25,.5],[-4,-4,-4],[4,4,4]],np.float32)
    bound=convex_filter_intervals(Interval(nodes),points)
    for corner in nodes.reshape(-1,4):
        assert np.all(corner>=bound.lo[0]) and np.all(corner<=bound.hi[0])
    np.testing.assert_array_equal(bound.lo[1:],0)
    np.testing.assert_array_equal(bound.hi[1:],0)


def test_physical_to_lattice_transform_and_wrong_origin_are_observable():
    from roshambo3.reference.field_intervals import convex_filter_intervals
    z,y,x=np.indices((5,5,5))
    nodes=np.repeat(x[...,None],4,axis=-1).astype(float)
    points=np.array([[11.5,21.5,31.5]],np.float32)
    expected=convex_filter_intervals(Interval(nodes),points,[10,20,30],.5)
    assert np.all(expected.lo<=3) and np.all(expected.hi>=3)
    wrong=convex_filter_intervals(Interval(nodes),points,[8,20,30],.5)
    assert np.all(wrong.hi<expected.lo)

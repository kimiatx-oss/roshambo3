"""Directed fixed-pose shape intervals from declared physical inputs.

No production constants or native observations enter this model. Primitive
outward intervals include real and binary32-rounded operations, covering FMA
contraction alternatives. Explicit serial/lane reductions then use monotone
binary32 addition on their binary32 endpoints. Correlation is never discounted.
"""
from __future__ import annotations
import math
import numpy as np
from .field_intervals import Interval
from .overlap import ALPHA, PAIR_PREFACTOR, pair_overlap
from .numerics_contracts import PathArithmetic, residual_bounds

CPU_EXP2_SOURCE = 'https://sourceware.org/legacy-ml/libc-alpha/2017-09/msg00720.html'
CUDA_EXP2_SOURCE = 'https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html'


def polynomial_error():
    """Declared d5 coefficient/Taylor distance and Lagrange remainder on [-.5,.5]."""
    coefficients=[1.,.693147182,.240226507,.0555051871,.00961799621,.00133978436]
    distance=math.fsum(abs(float(np.float32(c))-math.log(2)**i/math.factorial(i))*.5**i
                       for i,c in enumerate(coefficients))
    remainder=math.sqrt(2)*(math.log(2)/2)**6/math.factorial(6)
    return math.nextafter(math.sqrt(2)*(distance+remainder)+8*2**-24,math.inf)


def constant(value):
    # Existing singleton controls admit two binary32 ULPs around published
    # physical constants. This also covers the pinned reduced-frame scale,
    # which is not always the nearest binary32 to the full physical expression.
    point=np.float32(value)
    lower=upper=point
    for _ in range(2):
        lower=np.nextafter(lower,np.float32(-np.inf))
        upper=np.nextafter(upper,np.float32(np.inf))
    return Interval(float(lower),float(upper))


def _points(value):
    value=np.asarray(value)
    if value.dtype!=np.float32 or value.ndim!=2 or value.shape[1]!=3 or not len(value) or not np.isfinite(value).all():
        raise ValueError('nonempty finite float32 [n,3] points required')
    if len(value)>256 or np.max(np.abs(value))>32:
        raise ValueError('declared shape panel domain is <=256 atoms and |coordinate|<=32A')
    return value


def _sum(a,b):
    # Both endpoints are binary32 numbers represented exactly in binary64.
    # Binary64 addition supplies enough precision to determine binary32 RN,
    # including a halfway sum of two binary32 operands. No FMA can change a
    # source-order sum containing no multiply.
    return Interval((a.lo+b.lo).astype(np.float32).astype(float),
                    (a.hi+b.hi).astype(np.float32).astype(float))


def _reduce(terms,grain):
    nq,nb=terms.lo.shape
    if grain=='scalar':
        result=Interval(0.)
        for j in range(nb):
            for i in range(nq):result=_sum(result,Interval(terms.lo[i,j],terms.hi[i,j]))
        return result
    if grain not in ('query_atom','library_atom'):
        raise ValueError('unknown declared reduction grain')
    result=Interval(np.zeros(8))
    if grain=='query_atom':
        groups=(([(i,j) if i<nq else None for i in range(begin,begin+8)])
                for j in range(nb) for begin in range(0,nq,8))
    else:
        groups=(([(i,j) if j<nb else None for j in range(begin,begin+8)])
                for begin in range(0,nb,8) for i in range(nq))
    for group in groups:
        lo=np.array([terms.lo[index] if index is not None else 0. for index in group])
        hi=np.array([terms.hi[index] if index is not None else 0. for index in group])
        result=_sum(result,Interval(lo,hi))
    four=_sum(Interval(result.lo[:4],result.hi[:4]),Interval(result.lo[4:],result.hi[4:]))
    two=_sum(Interval(four.lo[:2],four.hi[:2]),Interval(four.lo[2:],four.hi[2:]))
    return _sum(Interval(two.lo[0],two.hi[0]),Interval(two.lo[1],two.hi[1]))


def overlap_interval(query,library,matrix=None,translation=None,*,grain='scalar',exponential='glibc_exp2',prefactor_multiplier=1.):
    """Actual affine scoring arithmetic; physical matrix/translation cast to fp32.

    Native query preparation scales physical fp32 points. Library affine matrix
    and translation are independently supplied physical values, cast to fp32,
    then scaled by the native constant. This covers the evaluator at a declared
    pose, not quaternion construction or optimizer-generated pose selection.
    """
    a,b=_points(query),_points(library)
    matrix=np.eye(3) if matrix is None else np.asarray(matrix,np.float64)
    translation=np.zeros(3) if translation is None else np.asarray(translation,np.float64)
    if matrix.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(matrix).all() or not np.isfinite(translation).all():
        raise ValueError('finite physical 3x3 matrix and translation required')
    if np.max(np.abs(matrix))>2 or np.max(np.abs(translation))>16:
        raise ValueError('declared matrix/translation domain exceeded')
    if not math.isfinite(prefactor_multiplier) or prefactor_multiplier<=0:
        raise ValueError('positive finite prefactor multiplier required')
    scale=constant(math.sqrt(ALPHA/(2*math.log(2))))
    q=Interval(a)*scale
    m=Interval(matrix.astype(np.float32))*scale
    t=Interval(translation.astype(np.float32))*scale
    posed=[]
    for axis in range(3):
        posed.append(Interval(b[:,0])*Interval(m.lo[axis,0],m.hi[axis,0])
            +Interval(b[:,1])*Interval(m.lo[axis,1],m.hi[axis,1])
            +Interval(b[:,2])*Interval(m.lo[axis,2],m.hi[axis,2]))
    square=[]
    for axis in range(3):
        delta=(Interval(q.lo[:,axis,None],q.hi[:,axis,None])-Interval(t.lo[axis],t.hi[axis]))-Interval(posed[axis].lo[None,:],posed[axis].hi[None,:])
        # A squared difference is nonnegative even if its interval crosses zero.
        low=np.where((delta.lo<=0)&(delta.hi>=0),0,np.minimum(delta.lo**2,delta.hi**2))
        square.append(Interval.rounded(low,np.maximum(delta.lo**2,delta.hi**2)))
    distance=square[0]+square[1]+square[2]
    distance=Interval(np.maximum(0,distance.lo),np.maximum(0,distance.hi))
    if exponential=='glibc_exp2':relative=2**-22
    elif exponential=='cuda_exp2':relative=2**-21
    elif exponential=='minimax_d5':relative=polynomial_error()
    else:raise ValueError('unknown declared exponential')
    # One binary64 evaluation ULP is dominated by the binary32 implementation
    # allowance. Normal/subnormal tails are covered absolutely, with no fitted
    # input-domain residuals.
    tiny=np.finfo(np.float32).tiny
    terms=Interval.rounded(np.maximum(0,np.exp2(-distance.hi)*(1-relative)-tiny),np.exp2(-distance.lo)*(1+relative)+tiny)
    if exponential=='minimax_d5':
        terms=Interval(np.where(distance.hi>24,0,terms.lo),np.where(distance.lo>24,0,terms.hi))
    summed=_reduce(terms,grain)
    prefactor=constant(PAIR_PREFACTOR)
    if prefactor_multiplier!=1.:
        prefactor=prefactor*np.float32(prefactor_multiplier)
    result=summed*prefactor
    physical=pair_overlap(a.astype(float),b.astype(float)@matrix.T+translation)
    maximum=max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))*float(np.max(np.abs(matrix).sum(axis=1)))+float(np.max(np.abs(translation))))
    # Bound the reference's own affine/subtraction cancellation absolutely:
    # <=16 binary64 roundings at magnitude M per component. The Gaussian
    # gradient norm is globally <=sqrt(ALPHA/e), so its value perturbation
    # is bounded without dividing by a possibly tiny pair value. Another
    # 128*u64 per term covers width/coefficient formation, squared-norm/
    # exponent evaluation and the existing reference's one-ULP libm model.
    # The final positive sum has n+8 roundings. These are operation counts
    # and global derivative bounds, independent of native observations.
    count=len(a)*len(b);u64=2**-53
    coordinate_error64=16*u64*maximum
    sum_gamma=(count+8)*u64/(1-(count+8)*u64)
    oracle_error=math.nextafter(PAIR_PREFACTOR*count*(128*u64+math.sqrt(3*ALPHA/math.e)*coordinate_error64)
        +sum_gamma*max(1.,abs(physical)),math.inf)
    contract=PathArithmetic('public_fixed_affine_shape',coordinate_error=64*2**-24*maximum,
        exp2_relative_error=relative,argument_roundings=16,coefficient_roundings=8,
        operation_model_derivation='Existing public final physical-pose envelope:64*u*M coordinate preparation; <=16 exponent and8 coefficient roundings; positive serial n*m reduction. SIMD adds declared d5 and cutoff envelope.')
    broad=residual_bounds(a.astype(float),b.astype(float)@matrix.T+translation,contract=contract)['overlap_volume']
    if exponential=='minimax_d5':broad+=PAIR_PREFACTOR*len(a)*len(b)*2**-24
    return dict(lower=float(result.lo)-oracle_error,upper=float(result.hi)+oracle_error,physical=physical,
        row_bound=broad,oracle_error=oracle_error,oracle_coordinate_error=coordinate_error64,oracle_operation_count=128,grain=grain,exponential=exponential,
        term_count=len(a)*len(b),sum_lower=float(summed.lo),sum_upper=float(summed.hi),scope='fixed declared affine shape overlap only')


def aggregate_report(values,models):
    values=np.asarray(values,np.float64)
    if values.ndim!=1 or len(values)!=len(models) or not len(values) or not np.isfinite(values).all():
        raise ValueError('one finite observation per declared row required')
    residual=[float(value)-model['physical'] for value,model in zip(values,models)]
    lower=math.nextafter(math.fsum(model['lower']-model['physical'] for model in models)/len(models),-math.inf)
    upper=math.nextafter(math.fsum(model['upper']-model['physical'] for model in models)/len(models),math.inf)
    mean=math.fsum(residual)/len(values)
    return dict(rows=len(values),mean_bias=mean,mean_interval=[lower,upper],mean_pass=lower<=mean<=upper,
        rows_pass=all(abs(error)<=model['row_bound'] for error,model in zip(residual,models)),
        generic_mean_radius=math.fsum(model['row_bound'] for model in models)/len(models),
        arithmetic='Directed affine/exponent/lane/serial intervals averaged with arbitrary row correlation; no sqrt(N)')


def prefactor_plant_model(model,multiplier):
    """Declared source coefficient multiplication; reuses no observed values."""
    result=Interval(model['sum_lower'],model['sum_upper'])*(constant(PAIR_PREFACTOR)*np.float32(multiplier))
    return dict(model,lower=float(result.lo)-model['oracle_error'],upper=float(result.hi)+model['oracle_error'])

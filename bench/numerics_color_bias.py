"""Preregistered exact-origin typed-color aggregate control (not SEARCH/FIELD)."""
import hashlib
import math
import numpy as np
from roshambo3.reference.overlap import COLOR_PREFACTOR
from roshambo3.reference.numerics_contracts import PathArithmetic, residual_bounds

DERIVATION = ('typed origin single-pair: exact coordinate/exponent operations; '
    'CPU exp(0)=1; coefficient two binary32 ULP plus64 double u; '
    'dyadic height/width scale; outward binary32 product; arbitrary correlation')


def color_panel(repeats=32):
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError('positive integer repeat count required')
    pairs = np.tile(np.array([(a,b) for a in range(1,7) for b in range(1,7)],np.int32),(repeats,1))
    n=len(pairs)
    coords=np.zeros((n,8,4),np.float32)
    coords[:,:2,3]=1
    types=np.zeros((n,8),np.int32); types[:,1]=pairs[:,0]
    counts=np.zeros((n,7),np.uint8); counts[:,0]=1
    counts[np.arange(n),pairs[:,1]]=1
    widths=np.zeros((7,7),np.float32); heights=np.zeros_like(widths)
    for a in range(1,7):
        for b in range(1,7):
            widths[a,b]=0 if (a+b)%5==0 else 4 if (a+b)%3==0 else 1
            heights[a,b]=0 if (a+2*b)%7==0 else 2.**(((a+b)%3)-1)
    return dict(query_coords=coords,query_types=types,query_n_atoms=np.ones(n,np.int32),
        library_coords=coords.copy(),library_counts=counts,
        pair_query_row=np.arange(n,dtype=np.int32),pair_library_row=np.arange(n,dtype=np.int32),
        interaction_r=widths,interaction_p=heights), pairs


def panel_hash(panel):
    h=hashlib.sha256()
    for name,value in sorted(panel.items()):
        h.update(name.encode()+b'\0'+value.dtype.str.encode()+b'\0')
        h.update(str(value.shape).encode()+b'\0'+value.tobytes())
    return h.hexdigest()


def coefficient_interval(width,height,*,backend='cpu'):
    if backend not in ('cpu','cuda'):
        raise ValueError('CPU/CUDA only')
    if width not in (0,1,4) or height not in (0,.5,1,2):
        raise ValueError('outside preregistered coefficient domain')
    if width==0 or height==0:return 0.,0.
    p=COLOR_PREFACTOR
    f=np.float32(p)
    ulp=max(float(np.nextafter(f,np.float32(math.inf)))-float(f),
            float(f)-float(np.nextafter(f,np.float32(-math.inf))))
    e=2*ulp+64*2**-53*p
    scale=height*height/width**1.5
    ee=0 if backend=='cpu' else 2*2**-23
    lo=np.nextafter(np.float32((p-e)*scale*(1-ee)),np.float32(-math.inf))
    hi=np.nextafter(np.float32((p+e)*scale*(1+ee)),np.float32(math.inf))
    return float(lo),float(hi)


def color_report(values,*,backend='cpu',repeats=32):
    panel,pairs=color_panel(repeats)
    values=np.asarray(values,np.float64)
    if values.shape!=(len(pairs),) or not np.isfinite(values).all():
        raise ValueError('one finite observation required for every preregistered row')
    r,p=panel['interaction_r'],panel['interaction_p']
    origin=np.zeros((1,3),np.float32)
    contract=PathArithmetic('typed_origin_'+backend,
        exp2_relative_error=2**(-22 if backend=='cpu' else -21),
        operation_model_derivation=DERIVATION)
    strata=[]
    lower=[];upper=[]; row_pass=[]
    for a in range(1,7):
        for b in range(1,7):
            at=np.flatnonzero((pairs[:,0]==a)&(pairs[:,1]==b))
            width,height=float(r[a,b]),float(p[a,b])
            exact=0. if width==0 else COLOR_PREFACTOR*height*height/width**1.5
            limit=residual_bounds(origin,origin,contract=contract,
                query_color=origin,query_types=[a],library_color=origin,library_types=[b],
                interaction_r=r,interaction_p=p)['overlap_color']
            lo,hi=coefficient_interval(width,height,backend=backend)
            mean=math.fsum(map(float,values[at]))/len(at)
            ok=bool(np.all(np.abs(values[at]-exact)<=limit))
            row_pass.append(ok)
            lower.extend([lo]*len(at));upper.extend([hi]*len(at))
            strata.append(dict(query_type=a,library_type=b,rows=len(at),physical=exact,
                row_bound=limit,rows_pass=ok,mean=mean,interval=[lo,hi],mean_pass=lo<=mean<=hi))
    lo=math.nextafter(math.fsum(lower)/len(values),-math.inf)
    hi=math.nextafter(math.fsum(upper)/len(values),math.inf)
    mean=math.fsum(map(float,values))/len(values)
    return dict(rows=len(values),panel_sha256=panel_hash(panel),rows_pass=all(row_pass),
        mean=mean,mean_interval=[lo,hi],mean_pass=lo<=mean<=hi,strata=strata,
        derivation=DERIVATION,scope='exact-final typed-color single-pair origin; not molecular/SEARCH/FIELD')


def native_color(backend='cpu',repeats=32):
    panel,pairs=color_panel(repeats)
    if backend=='cpu':
        from roshambo3.backend import cpu
        scores=cpu.score_pairs(**panel,steps=1,lr_q=0,lr_t=0,start_mode=0,
            budget_override='typed-color origin aggregate arithmetic control',
            color=True,color_mode='combo_optimal',mixing=.5,threads=1)
    elif backend=='cuda':
        from bench.numerics_backends import cuda_fixed
        scores=cuda_fixed(panel)
    else:raise ValueError('CPU/CUDA only')
    np.testing.assert_array_equal(scores[:,9:13],np.tile([1,0,0,0],(len(pairs),1)))
    np.testing.assert_array_equal(scores[:,13:16],np.zeros((len(pairs),3)))
    return scores[:,4]

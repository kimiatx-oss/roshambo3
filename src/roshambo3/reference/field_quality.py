"""Independent shape FIELD quality ingredients; no native observations as budgets.

The full3D contract is range-preserving interpolation, not exact tensor8 hardware.
The tensor8 sampler below generates independent search endpoints only.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
from .field_intervals import Interval, gaussian_prefilter_intervals
from .overlap import ALPHA, PAIR_PREFACTOR, apply_pose, self_overlap
from ..numerics import COORD_SCALE, WIRE_QUANT_STEP

INV_COORD_SCALE = np.float32(1)/COORD_SCALE


def lattice(query, spacing_a=.8, pad_a=8.):
    """Production host geometry, independently transcribed in double precision."""
    query=np.asarray(query,np.float32)
    scale=float(COORD_SCALE); hs=float(np.float32(spacing_a))*scale
    pad=float(np.float32(pad_a))*scale
    lo=query[:,:3].astype(float).min(axis=0)*scale
    hi=query[:,:3].astype(float).max(axis=0)*scale
    return (lo-pad).astype(np.float32),np.float32(hs),np.maximum(1,np.ceil((hi-lo+2*pad)/hs)).astype(int)


def nodes(query):
    origin,spacing,cells=lattice(query)
    return gaussian_prefilter_intervals(query,origin,spacing,cells),origin,spacing


def tensor8(field, points, origin, spacing):
    """Ideal tensor product, solely an endpoint generator (not a CUDA oracle)."""
    points=np.asarray(points,float)
    address=(points-origin)/float(spacing)
    base=np.floor(address).astype(int)
    frac=np.rint((address-base)*256)/256
    out=np.zeros(points.shape[:-1]+(4,))
    for dx,dy,dz in itertools.product((0,1),repeat=3):
        ijk=base+np.array([dx,dy,dz])
        valid=np.all((ijk>=0)&(ijk<np.array(field.shape[:3][::-1])),axis=-1)
        weight=np.prod(np.where(np.array([dx,dy,dz]),frac,1-frac),axis=-1)
        x,y,z=np.moveaxis(ijk[valid],-1,0)
        out[valid]+=weight[valid,None]*field[z,y,x]
    return out


def matrix(q):
    """Unit quaternion matrices for independent double precision search."""
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y,
        2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x,
        2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y),axis=-1).reshape(q.shape[:-1]+(3,3))


def compose(a,b):
    aw,av=a[...,:1],a[...,1:];bw,bv=b[...,:1],b[...,1:]
    return np.concatenate((aw*bw-np.sum(av*bv,axis=-1,keepdims=True),aw*bv+bw*av+np.cross(av,bv)),axis=-1)


def search(query, library, field, origin, spacing, *, steps=25):
    """Four-start fp64 Adagrad exponential-map endpoint generator.

    Each bounded node realization is selected before search; its stored gradients
    drive this independent search. Selection uses the same software field value.
    No native optimized endpoint is an input.
    """
    b=np.asarray(library,float)*float(COORD_SCALE)
    q=np.eye(4);t=np.zeros((4,3));cache=np.zeros((4,6))
    rawself=(self_overlap(query[:,:3])+self_overlap(library))/PAIR_PREFACTOR
    for step in range(steps+1):
        rotated=np.einsum('sij,nj->sni',matrix(q),b)
        sampled=tensor8(field,rotated+t[:,None,:],origin,spacing)
        if step==steps:break
        grad=-sampled[:,:,1:]/rawself
        g=np.concatenate((np.sum(np.cross(2*rotated,grad),axis=1),np.sum(grad,axis=1)),axis=-1)
        cache+=g*g
        delta=-g/(np.sqrt(cache)+1e-8)*np.array([.1]*3+[.4*float(COORD_SCALE)]*3)
        angle=np.linalg.norm(delta[:,:3],axis=-1,keepdims=True)
        dq=np.concatenate((np.cos(angle),np.sinc(angle/math.pi)*delta[:,:3]),axis=-1)
        q=compose(dq,q);q/=np.linalg.norm(q,axis=-1,keepdims=True)
        t+=delta[:,3:]
    values=sampled[:,:,0].sum(axis=1)
    winner=int(np.argmax(values))
    return np.concatenate((q[winner],t[winner]/float(COORD_SCALE))),winner,values


def staged(cloud):
    """Actual float input -> short -> float library staging, in physical units."""
    cloud=np.asarray(cloud,np.float32)
    integers=np.rint(cloud/WIRE_QUANT_STEP)
    if np.any(abs(integers)>32767):raise ValueError('short staging range exceeded')
    return integers.astype(np.int16).astype(np.float32)*WIRE_QUANT_STEP


def native_points(cloud,q,t, *, returned_translation=False):
    """Enclose fp32 pose matrix, short staging, FMA, and translation recovery.

    q is the actual reported native quaternion, without normalization. A returned
    physical translation is the rounded product INV_SCALE*t_reduced; its entire
    rounding preimage is retained before recovering reduced translation.
    """
    w,x,y,z=[Interval(np.float32(v)) for v in q]
    m=[[Interval(1)-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
       [2*x*y+2*w*z,Interval(1)-2*x*x-2*z*z,2*y*z-2*w*x],
       [2*x*z-2*w*y,2*y*z+2*w*x,Interval(1)-2*x*x-2*y*y]]
    t=np.asarray(t,np.float32)
    if returned_translation:
        prev=np.nextafter(t,np.float32(-np.inf)).astype(float)
        nxt=np.nextafter(t,np.float32(np.inf)).astype(float)
        ti=Interval(np.nextafter((prev+t.astype(float))/2/float(INV_COORD_SCALE),-np.inf),
                    np.nextafter((nxt+t.astype(float))/2/float(INV_COORD_SCALE),np.inf))
    else:ti=Interval(t)
    b=staged(cloud)
    result=[]
    for k in range(3):
        row=[v*COORD_SCALE for v in m[k]]
        result.append(row[0]*b[:,0]+row[1]*b[:,1]+row[2]*b[:,2]+Interval(ti.lo[k],ti.hi[k]))
    return Interval(np.stack([v.lo for v in result],axis=-1),np.stack([v.hi for v in result],axis=-1))


def hull(field,points,origin,spacing):
    """Full3D convex range for interval-valued points, including border zero."""
    inv=Interval.rounded(1/float(spacing),1/float(spacing))
    coordinate=(points-Interval(origin))*inv+.5-.5
    dims=np.array(field.lo.shape[:3][::-1])
    lo=np.floor(np.clip(coordinate.lo,-2,dims+1)).astype(int)
    hi=np.floor(np.clip(coordinate.hi,-2,dims+1)).astype(int)
    lower=[];upper=[]
    for a,b in zip(lo,hi):
        values=[]
        for x,y,z in itertools.product(*(range(max(-1,int(a[k])),min(int(dims[k]),int(b[k])+1)+1) for k in range(3))):
            values.append((field.lo[z,y,x],field.hi[z,y,x]) if np.all(np.array([x,y,z])>=0) and np.all(np.array([x,y,z])<dims) else (np.zeros(4),np.zeros(4)))
        lower.append(np.min([v[0] for v in values],axis=0) if values else np.zeros(4))
        upper.append(np.max([v[1] for v in values],axis=0) if values else np.zeros(4))
    return Interval(lower,upper)


def overlap_interval(field,points,origin,spacing):
    values=hull(field,points,origin,spacing)
    # Actual WA=4 serial lane accumulation then XOR offsets 1,2. All atom
    # amplitudes are one in this public panel. Interval addition covers FMA too.
    lanes=[Interval(0.) for _ in range(4)]
    for i in range(len(values.lo)):
        lanes[i%4]=lanes[i%4]+Interval(values.lo[i,0],values.hi[i,0])
    for offset in (1,2):lanes=[lanes[k]+lanes[k^offset] for k in range(4)]
    return lanes[0]


def phi(raw,total_self):
    raw=np.asarray(raw,float)
    if np.any(raw>=total_self):raise ValueError('sampled overlap reaches Tanimoto pole')
    return raw/(total_self-raw)


def phi_interval(raw,total_self):
    return Interval(np.nextafter(phi(raw.lo,total_self),-np.inf),np.nextafter(phi(raw.hi,total_self),np.inf))


def own_pose(cloud,pose):
    return apply_pose(cloud,np.asarray(pose[:4]),np.asarray(pose[4:]))


def staging_slack(field,cloud,pose,origin,spacing,total_self):
    """Independent two-sided scalar bound between rigid and re-input poses.

    Convex hulls at both positions make this conservative across texture-cell
    discontinuities. A Gaussian Lipschitz bound alone would not cover an
    undocumented hardware weight allocation. No residuals enter this slack.
    """
    moved=own_pose(cloud,pose)
    rigid=phi_interval(overlap_interval(field,Interval(moved*float(COORD_SCALE)),origin,spacing),total_self)
    reinput=phi_interval(overlap_interval(field,native_points(moved,[1,0,0,0],[0,0,0]),origin,spacing),total_self)
    return float(max(abs(rigid.lo-reinput.hi),abs(rigid.hi-reinput.lo)))

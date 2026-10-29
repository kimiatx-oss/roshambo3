"""Direct point controls for independently preregistered FIELD search quality."""
from __future__ import annotations
import math
import numpy as np
from . import field_quality as f
from .overlap import PAIR_PREFACTOR,self_overlap
from ..numerics import COORD_SCALE


def search_all(query,library,field,origin,spacing):
    """Independent fp64 search; retain all four endpoints before native probes."""
    b=np.asarray(library,float)*float(COORD_SCALE)
    q=np.eye(4);t=np.zeros((4,3));cache=np.zeros((4,6))
    total=(self_overlap(query[:,:3])+self_overlap(library))/PAIR_PREFACTOR
    for step in range(25):
        rotated=np.einsum('sij,nj->sni',f.matrix(q),b)
        gradient=-f.tensor8(field,rotated+t[:,None,:],origin,spacing)[:,:,1:]/total
        g=np.concatenate((np.sum(np.cross(2*rotated,gradient),axis=1),np.sum(gradient,axis=1)),axis=-1)
        cache+=g*g
        delta=-g/(np.sqrt(cache)+1e-8)*np.array([.1]*3+[.4*float(COORD_SCALE)]*3)
        angle=np.linalg.norm(delta[:,:3],axis=-1,keepdims=True)
        dq=np.concatenate((np.cos(angle),np.sinc(angle/math.pi)*delta[:,:3]),axis=-1)
        q=f.compose(dq,q);q/=np.linalg.norm(q,axis=-1,keepdims=True)
        t+=delta[:,3:]
    return np.concatenate((q,t/float(COORD_SCALE)),axis=-1)


def point_control(cloud,pose):
    """Declared rounded endpoint points, with an unfitted displacement guard."""
    pose=np.asarray(pose,float)
    if not np.isfinite(pose).all() or np.linalg.norm(pose[4:])>16:raise ValueError('control extent exceeded')
    physical=f.own_pose(cloud,pose)
    reduced=physical*float(COORD_SCALE)
    points=np.ascontiguousarray(reduced,dtype=np.float32)
    # The computed fp64 reduced point is the declared endpoint. This only
    # bounds its subsequent fp32 rounding, not error relative to an exact real
    # quaternion/rotation. No cancellation-sensitive pose envelope is claimed.
    per_axis=(2**-24*np.abs(reduced)+2**-150)/float(COORD_SCALE)
    # Outward fp64 evaluation margin for these positive products/sums/norm,
    # independently of the unbounded rotation/translation cancellation ratio.
    bound=np.nextafter(np.linalg.norm(per_axis,axis=-1)*(1+32*np.finfo(float).eps),np.inf)
    displacement=np.linalg.norm((points.astype(float)-reduced)/float(COORD_SCALE),axis=-1)
    if np.any(displacement>bound):raise AssertionError('point conversion exceeds independent rounding guard')
    return points,dict(max_displacement=float(displacement.max()),max_bound=float(bound.max()),
        scope='fp32 rounding of the computed fp64 reduced endpoint only; no exact real pose envelope')


def wa4(values):
    """Positive unit-weight field sum in production's WA4 operation order."""
    values=np.asarray(values,np.float32)
    lanes=np.zeros(4,np.float32)
    for i,value in enumerate(values):lanes[i%4]=np.float32(lanes[i%4]+value)
    for offset in (1,2):lanes=np.array([np.float32(lanes[k]+lanes[k^offset]) for k in range(4)])
    return lanes[0]

"""Directed intervals for the CUDA Gaussian prefilter, before texture sampling.

All constants derive from the fp64 physical reference. This does not model the
undocumented combined weights or accumulation of full3D texture filtering.
Intervals cover separate fp32 roundings and FMA contraction: every primitive's
outward enclosure includes both the exact real operation and its fp32 rounding.
No backend observations or measured residual budgets enter this construction.
"""
from __future__ import annotations
import math
import numpy as np
from .overlap import ALPHA


class Interval:
    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, np.float64)
        self.hi = np.asarray(lo if hi is None else hi, np.float64)
        if np.any(self.lo > self.hi) or not (np.isfinite(self.lo).all() and np.isfinite(self.hi).all()):
            raise ValueError("finite ordered interval required")

    @staticmethod
    def rounded(lo, hi):
        # A whole fp32 ULP also covers double endpoint evaluation error. The
        # exact endpoint remains inside: fused and unfused expressions qualify.
        lower = np.nextafter(np.asarray(lo, np.float32), np.float32(-np.inf)).astype(float)
        upper = np.nextafter(np.asarray(hi, np.float32), np.float32(np.inf)).astype(float)
        return Interval(lower, upper)

    def __add__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        return self.rounded(self.lo+other.lo, self.hi+other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -(other if isinstance(other, Interval) else Interval(other))

    def __mul__(self, other):
        other = other if isinstance(other, Interval) else Interval(other)
        candidates = np.stack(np.broadcast_arrays(self.lo*other.lo, self.lo*other.hi,
                                                   self.hi*other.lo, self.hi*other.hi))
        return self.rounded(candidates.min(axis=0), candidates.max(axis=0))

    __rmul__ = __mul__

    def exp2(self):
        lo, hi = np.exp2(self.lo), np.exp2(self.hi)
        # ex2.approx's declared two-ULP envelope is relative <=2^-21;
        # CUDA Programming Guide Table 7 declares exp2f <=2 ULP:
        # https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#mathematical-functions-appendix
        # FTZ can discard each fp32 subnormal. Include it absolutely.
        tiny = np.finfo(np.float32).tiny
        return self.rounded(np.maximum(0, lo*(1-2**-21)-tiny), hi*(1+2**-21)+tiny)

    def half(self):
        # Nearest-even conversion is monotone; no observed half bits enter.
        return Interval(self.lo.astype(np.float16).astype(float), self.hi.astype(np.float16).astype(float))


def gaussian_prefilter_intervals(query, origin, spacing, cells, *, shape=True, factor=1.):
    """Enclose production half4 Gaussian nodes (value and reduced gradient).

    query is physical fp32 [n,4], xyz and amplitude; geometry is exact fp32
    reduced-frame input. Shape uses exp(-ALPHA*d_physical²/2); color uses the
    declared factor multiplying that width. This is a builder-only contract.
    """
    q = np.asarray(query)
    origin = np.asarray(origin)
    cells = np.asarray(cells)
    if q.dtype != np.float32 or q.ndim != 2 or q.shape[1] != 4 or not len(q) or not np.isfinite(q).all():
        raise ValueError("nonempty finite float32 query [n,4] required")
    if origin.shape != (3,) or not np.isfinite(origin).all() or cells.shape != (3,) or np.any(cells<1) or np.any(cells>128) or np.any(cells!=cells.astype(int)):
        raise ValueError("finite origin and integer 1..128 cells required")
    if not np.isfinite(spacing) or spacing<=0 or not np.isfinite(factor) or factor<=0:
        raise ValueError("positive finite spacing and factor required")
    origin = origin.astype(np.float32)
    spacing = np.float32(spacing)
    factor = np.float32(factor)
    z,y,x = np.indices(tuple((cells+1)[::-1].astype(int)))
    xyz = [Interval(origin[k]) + Interval(index)*spacing for k,index in enumerate((x,y,z))]
    # Stored physical constants differ from mathematical values by <=one
    # fp32 rounding. The interval includes the real constant for FMA safety.
    scale = Interval.rounded(math.sqrt(ALPHA/(2*math.log(2))), math.sqrt(ALPHA/(2*math.log(2))))
    ln2 = Interval.rounded(math.log(2), math.log(2))
    b = ln2 if shape else ln2*factor
    b2, b3 = b*b, b*b*b
    accum = [Interval(np.zeros(x.shape)) for _ in range(8)]
    for atom in q:
        d = [xyz[k]-scale*atom[k] for k in range(3)]
        d2 = d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
        exponent = -d2 if shape else -d2*factor
        e = exponent.exp2()*atom[3]
        accum[0] = accum[0]+e
        accum[1] = accum[1]+e*(4*b2*d2-6*b)
        cg,cl = -2*b*e, e*(20*b2-8*b3*d2)
        for k in range(3):
            accum[2+k] = accum[2+k]+cg*d[k]
            accum[5+k] = accum[5+k]+cl*d[k]
    # hs*hs rounds before division by 12; enclose both operations.
    hs2 = Interval(spacing)*spacing
    c = Interval.rounded(hs2.lo/12, hs2.hi/12)
    out = [accum[0]-c*accum[1]]+[accum[2+k]-c*accum[5+k] for k in range(3)]
    out = [value.half() for value in out]
    return Interval(np.stack([v.lo for v in out],axis=-1), np.stack([v.hi for v in out],axis=-1))


def assert_interval(actual, expected):
    """Check every element and the signed aggregate against the same interval."""
    actual = np.asarray(actual, np.float64)
    if actual.shape != expected.lo.shape or not np.isfinite(actual).all():
        raise AssertionError("field result shape or finite-value contract")
    assert_signed_mean(actual,expected)
    if np.any(actual<expected.lo) or np.any(actual>expected.hi):
        raise AssertionError("Gaussian prefilter node outside directed arithmetic interval")
    # Signed aggregate, no independence or cancellation assumption. Keep
    # channel axes; this is not a mean budget estimated from a sample variance.
    axes = tuple(range(actual.ndim-1))
    midpoint = (expected.lo+expected.hi)/2
    return dict(elements=int(actual.size), ambiguous_elements=int(np.count_nonzero(expected.lo!=expected.hi)),
        mean_residual=(actual-midpoint).mean(axis=axes).tolist(),
        mean_lower=(expected.lo-midpoint).mean(axis=axes).tolist(),
        mean_upper=(expected.hi-midpoint).mean(axis=axes).tolist(),
        max_interval_width=float(np.max(expected.hi-expected.lo)))


def assert_signed_mean(actual, expected):
    """Independent directed signed-mean gate; no uniform maximum row budget."""
    actual=np.asarray(actual,np.float64)
    if actual.shape!=expected.lo.shape or not np.isfinite(actual).all():
        raise AssertionError('field signed mean shape or finite-value contract')
    axes=tuple(range(actual.ndim-1))
    mean=actual.mean(axis=axes)
    lower=expected.lo.mean(axis=axes);upper=expected.hi.mean(axis=axes)
    if np.any(mean<lower) or np.any(mean>upper):
        raise AssertionError('Gaussian field signed mean outside directed arithmetic interval')


CUDA_LINEAR_FILTER_SOURCE = 'https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#linear-filtering'


def convex_filter_intervals(nodes, points, origin=(0.,0.,0.), spacing=1.):
    """Conservative full3D ranges, independent of combined-weight allocation.

    CUDA's documented trilinear definition has nonnegative coefficients with
    unit mass; hence a sample lies in its eight-node convex hull. This retains
    that range property without assuming the rejected exact tensor weights.
    The declaration assumes a range-preserving implementation of the documented
    filter. It is tested explicitly, and is separate from the measured 8-bit
    one-axis precision declaration. It cannot itself detect that precision plant.

    Input nodes are *independent builder intervals*, never observed texels.
    Address arithmetic is enclosed operation by operation, including +half and
    the hardware center subtraction. Border corners contribute exact zero.
    """
    points=np.asarray(points)
    origin=np.asarray(origin,np.float32)
    spacing=np.float32(spacing)
    if points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('finite float32 points [n,3] required')
    if origin.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(spacing) or spacing<=0 or not np.isfinite(np.float32(1)/spacing):
        raise ValueError('finite origin and positive finite reciprocal spacing required')
    inv=Interval.rounded(1/float(spacing),1/float(spacing))
    coordinate=(Interval(points)-Interval(origin))*inv+.5
    # x_B=x_texture-0.5: exact subtraction of dyadic values suffices; enlarge
    # by its fp32 rounding too rather than assuming an internal address format.
    coordinate=coordinate-.5
    dimensions=np.array(nodes.lo.shape[:3][::-1])
    # Far outside addresses are all border zero; clip before integer casting
    # so finite large fp32 coordinates never overflow host integer indexing.
    lo=np.floor(np.clip(coordinate.lo,-2,dimensions+1)).astype(np.int64)
    hi=np.floor(np.clip(coordinate.hi,-2,dimensions+1)).astype(np.int64)
    lower=np.empty((len(points),4));upper=np.empty_like(lower)
    import itertools
    for i in range(len(points)):
        # Clamp enumeration to border and valid indices. This remains bounded
        # even for far outside addresses; all such indices read border zero.
        axes=[range(max(-1,int(lo[i,k])),min(int(dimensions[k]),int(hi[i,k])+1)+1) for k in range(3)]
        lows=[];highs=[]
        for x,y,z in itertools.product(*axes):
            if 0<=x<dimensions[0] and 0<=y<dimensions[1] and 0<=z<dimensions[2]:
                lows.append(nodes.lo[z,y,x]);highs.append(nodes.hi[z,y,x])
            else:
                lows.append(np.zeros(4));highs.append(np.zeros(4))
        lower[i]=np.min(lows,axis=0) if lows else 0
        upper[i]=np.max(highs,axis=0) if highs else 0
    return Interval(lower,upper)


def gaussian_physical_values(query, reduced_points, *, shape=True, factor=1.):
    """Live fp64 continuous Gaussian value, independently in physical units."""
    scale=math.sqrt(ALPHA/(2*math.log(2)))
    query=np.asarray(query,np.float64)
    physical=np.asarray(reduced_points,np.float64)/scale
    width=ALPHA/2*(1. if shape else float(np.float32(factor)))
    d2=np.sum((physical[:,None,:]-query[None,:,:3])**2,axis=-1)
    return np.sum(query[None,:,3]*np.exp(-width*d2),axis=1)

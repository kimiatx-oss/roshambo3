"""Directed singleton shape control for correlated aggregate error.

This deliberately narrow operation model does not qualify other geometries,
color, fields, or an optimizer. The protocol is summarized here.
"""
import math
import numpy as np
from roshambo3.reference.overlap import PAIR_PREFACTOR
from roshambo3.reference.numerics_contracts import residual_report, residual_bounds, PathArithmetic

DERIVATION = ('origin singleton: exact zero coordinate operations and exponent; '
    'exp2f <= two ULP at one; published prefactor within two binary32 ULP; '
    'one binary32 product rounded outward; arbitrary correlation, no sqrt(N)')


def singleton_interval():
    """Conservative physical-volume interval, independently of observations.

    Two ULP are measured with the larger adjacent spacing at each exact value.
    The prefactor allowance includes the published decimal implementation's
    conversion error; it is audited algebraically, never from native outputs.
    The fp64 reference-expression allowance is 64 double unit roundoffs.
    """
    p = PAIR_PREFACTOR
    def ulp(x):
        f = np.float32(x)
        return max(float(np.nextafter(f, np.float32(math.inf)))-float(f),
                   float(f)-float(np.nextafter(f, np.float32(-math.inf))))
    ep = 2*ulp(p) + 64*2**-53*abs(p)
    ee = 2*ulp(1.)
    # All quantities are positive. Outward binary32 neighbors include final
    # multiplication rounding and fp64 endpoint evaluation uncertainty.
    lo = np.nextafter(np.float32((p-ep)*(1-ee)), np.float32(-math.inf))
    hi = np.nextafter(np.float32((p+ep)*(1+ee)), np.float32(math.inf))
    return float(lo), float(hi)


def singleton_report(volumes, *, backend):
    """Compare existing row gates and the independent directed mean interval."""
    values = np.asarray(volumes, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('nonempty finite one-dimensional observations required')
    if backend not in ('cpu', 'cuda'):
        raise ValueError('only declared analytic CPU/CUDA paths are supported')
    # Unchanged generic operation counts and exp assumptions from existing
    # adapters. At the origin their coordinate uncertainty contributes zero.
    contract = PathArithmetic('singleton_'+backend,
        exp2_relative_error=2**(-22 if backend=='cpu' else -21),
        operation_model_derivation=DERIVATION)
    origin = np.zeros((1,3), np.float32)
    limit = residual_bounds(origin, origin, contract=contract)['overlap_volume']
    old = residual_report([{'volume':v} for v in values],
        [{'volume':PAIR_PREFACTOR} for _ in values],
        [{'volume':limit} for _ in values])['volume']
    lo, hi = singleton_interval()
    # math.fsum prevents the reporting reduction itself creating apparent bias.
    mean = math.fsum(map(float, values))/len(values)
    return dict(rows=len(values), rows_pass=old['rows_pass'], row_bound=limit,
        mean_volume=mean, mean_interval=[lo,hi], mean_pass=lo<=mean<=hi,
        mean_model_derivation=DERIVATION, scope='analytic singleton shape origin only')


def native_singletons(backend, count=32):
    """Exercise the production evaluator directly, before any final rescore."""
    origin = np.zeros((1,3), np.float32)
    if backend == 'cpu':
        from roshambo3.backend import cpu
        return [cpu.extension().numerics_shape_evaluate(origin,origin,0,0,1,.25)
                ['overlap_volume'] for _ in range(count)]
    if backend == 'cuda':
        from roshambo3 import _roshambo3_cuda as ext
        from bench.numerics_backends import cuda_shape_overlap
        return [cuda_shape_overlap(ext,origin,origin) for _ in range(count)]
    raise ValueError('unknown backend')

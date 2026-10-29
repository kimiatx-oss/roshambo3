"""Mathematical and execution checks for the directed shape bias control."""
import numpy as np
import pytest
from bench.numerics_bias import singleton_interval, singleton_report, native_singletons
from roshambo3.reference.overlap import PAIR_PREFACTOR


def test_prefactor_literal_has_independent_two_ulp_allowance():
    # Audit the protocol's literal representation, never a backend score.
    stored = float(np.float32('20.5795288'))
    spacing = float(np.spacing(np.float32(PAIR_PREFACTOR)))
    assert abs(stored-PAIR_PREFACTOR) <= 2*spacing


def test_correlated_bias_is_detected_without_sample_size_gain():
    lo,hi = singleton_interval()
    # Synthetic boundary check; native qualification must be a separate receipt.
    baseline = float(np.float32('20.5795288'))
    assert lo <= baseline <= hi
    biased = float(np.float32(baseline*np.float32(1+2**-21)))
    for count in (1,32,100):
        report = singleton_report([biased]*count, backend='cpu')
        assert report['rows_pass']
        assert not report['mean_pass']
        assert report['mean_interval']==[lo,hi]


def test_invalid_bias_observations_refused():
    for values in ([],[np.nan],[[1]]):
        with pytest.raises(ValueError):singleton_report(values,backend='cpu')


def test_cpu_directed_singleton_control():
    from roshambo3.backend import cpu
    if not cpu.available():pytest.skip('CPU extension unavailable')
    from bench.numerics_backends import verified_build
    verified_build('cpu')
    report=singleton_report(native_singletons('cpu'),backend='cpu')
    assert report['rows_pass'] and report['mean_pass'],report


def test_cuda_directed_singleton_control():
    from roshambo3.backend import cuda
    if not cuda.available():pytest.skip('CUDA extension unavailable')
    if cuda.extension().device_count()==0:pytest.skip('CUDA device unavailable')
    from bench.numerics_backends import verified_build
    verified_build('cuda')
    report=singleton_report(native_singletons('cuda'),backend='cuda')
    assert report['rows_pass'] and report['mean_pass'],report

"""Independent color control bounds and live native gate."""
import math
import numpy as np
import pytest
from bench.numerics_color_bias import color_panel,coefficient_interval,color_report,native_color
from roshambo3.reference.overlap import COLOR_PREFACTOR


def physical_values(repeats=32):
    panel,pairs=color_panel(repeats)
    r,p=panel['interaction_r'],panel['interaction_p']
    return np.array([0 if r[a,b]==0 else COLOR_PREFACTOR*float(p[a,b])**2/float(r[a,b])**1.5
                     for a,b in pairs])


def test_physical_coefficient_intervals_and_literal_audit():
    assert abs(float(np.float32('1.9687012'))-COLOR_PREFACTOR)<=2*float(np.spacing(np.float32(COLOR_PREFACTOR)))
    for width in (0,1,4):
        for height in (0,.5,1,2):
            lo,hi=coefficient_interval(width,height)
            physical=0 if width==0 else COLOR_PREFACTOR*height**2/width**1.5
            assert lo<=physical<=hi
            if physical==0:assert lo==hi==0


def test_correlated_color_bias_both_signs_without_sample_size_gain():
    for repeats in (1,32):
        baseline=physical_values(repeats).astype(np.float32)
        clean=color_report(baseline,repeats=repeats)
        assert clean['rows_pass'] and clean['mean_pass']
        for sign in (-1,1):
            biased=(baseline*np.float32(1+sign*2**-21)).astype(np.float32)
            report=color_report(biased,repeats=repeats)
            assert report['rows_pass'] and not report['mean_pass']
            assert report['mean_interval']==clean['mean_interval']
            assert all(not s['mean_pass'] for s in report['strata'] if s['physical']>0)
            assert all(s['mean_pass'] for s in report['strata'] if s['physical']==0)


def test_malformed_color_observations_refused():
    for values in ([],[math.nan],np.zeros((1152,1))):
        with pytest.raises(ValueError):color_report(values)
    for count in (0,-1,1.5,True):
        with pytest.raises(ValueError):color_panel(count)
    with pytest.raises(ValueError):coefficient_interval(2,1)


def test_cpu_directed_typed_color_control():
    from roshambo3.backend import cpu
    if not cpu.available():pytest.skip('CPU extension unavailable')
    from bench.numerics_backends import verified_build
    verified_build('cpu')
    report=color_report(native_color())
    assert report['rows_pass'] and report['mean_pass'],report
    assert all(s['mean_pass'] for s in report['strata']),report


def test_cuda_color_interval_and_amplitudes_before_native_qualification():
    baseline=physical_values().astype(np.float32)
    clean=color_report(baseline,backend='cuda')
    assert clean['rows_pass'] and clean['mean_pass']
    for sign in (-1,1):
        report=color_report((baseline*np.float32(1+sign*2**-20)).astype(np.float32),backend='cuda')
        assert report['rows_pass'] and not report['mean_pass']
        assert all(not s['mean_pass'] for s in report['strata'] if s['physical']>0)


def test_cuda_directed_typed_color_control():
    from roshambo3.backend import cuda
    if not cuda.available():pytest.skip('CUDA extension unavailable')
    if cuda.extension().device_count()==0:pytest.skip('CUDA device unavailable')
    from bench.numerics_backends import verified_build
    verified_build('cuda')
    report=color_report(native_color('cuda'),backend='cuda')
    assert report['rows_pass'] and report['mean_pass'],report
    assert all(s['mean_pass'] for s in report['strata']),report

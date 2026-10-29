"""Full canonical molecules: derive reduction count without observing residuals."""
import numpy as np
import pytest
from bench.numerics_molecular_color_bias import molecular_model,molecular_report,native_molecular_color


def test_full_molecular_issued_count_and_preregistered_window():
    panel,strata=molecular_model(1)
    assert len(strata)==4
    for i,s in enumerate(strata):
        counts=panel['library_counts'][i,1:].astype(int)
        assert s['issued_terms_per_row']==sum(counts*counts)
        assert s['aggregate_bound']/s['physical']<2**-15<s['row_bound']/s['physical']
        assert s['binding']['source_sdf_sha256']
    baseline=np.array([s['physical'] for s in strata],np.float32)
    for sign in (-1,1):
        report=molecular_report((baseline*np.float32(1+sign*2**-15)).astype(np.float32),1)
        assert report['rows_pass'] and not report['mean_pass']
        assert all(not s['mean_pass'] for s in report['strata'])


def test_molecular_observation_and_repeat_guards():
    for values in ([],[np.nan],np.zeros((128,1))):
        with pytest.raises(ValueError):molecular_report(values)
    for repeats in (0,-1,True):
        with pytest.raises(ValueError):molecular_model(repeats)


def test_cpu_canonical_molecular_color_aggregate():
    import platform
    if platform.libc_ver()!=('glibc','2.39'):
        pytest.skip('molecular expf model currently qualified for glibc2.39 only')
    from roshambo3.backend import cpu
    if not cpu.available():pytest.skip('CPU extension unavailable')
    from bench.numerics_backends import verified_build
    verified_build('cpu')
    values,census=native_molecular_color()
    report=molecular_report(values)
    assert report['rows_pass'] and report['mean_pass'],report
    assert all(s['mean_pass'] for s in report['strata']),report

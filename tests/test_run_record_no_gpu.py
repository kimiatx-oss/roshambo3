"""CPU-only ledger metadata must not require an NVIDIA driver utility."""
from types import SimpleNamespace
import subprocess

from bench.run_record import gpu_info


def test_missing_nvidia_smi_records_explicit_missing_telemetry(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(subprocess, 'run', missing)
    record = gpu_info(None, False)
    assert record['name'] == record['uuid'] == record['driver'] == 'unknown'
    assert record['power_limit'] == 'unknown'
    assert record['clocks_locked'] is False
    assert record['telemetry_error'] == 'nvidia-smi not installed'


def test_available_nvidia_smi_keeps_existing_gpu_metadata(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout='Example GPU, GPU-example, 123.45, 300 W\n', returncode=0))
    record = gpu_info(None, True)
    assert record == dict(name='Example GPU', uuid='GPU-example',
                          driver='123.45', clocks_locked=True, power_limit='300 W')

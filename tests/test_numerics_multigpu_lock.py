"""Multi-device qualification must hold every lock before native execution."""
from types import SimpleNamespace
import argparse
import os

import pytest

from bench import gpu_lock
from bench.run_numerics_locked import device_indices, run_locked


@pytest.mark.parametrize('value', ['0,0', '-1', '', 'x', '0,', '1', '1,2'])
def test_refuse_invalid_device_selection(value):
    with pytest.raises(argparse.ArgumentTypeError):
        device_indices(value)


def fake_devices(monkeypatch, *, busy=None):
    active = set()
    events = []
    def resolve(index):
        return SimpleNamespace(uuid=f'GPU-{index}', visible_index=index)
    class Lock:
        def __init__(self, ticket, device):
            self.device = device
            self.acquired = False
        def __enter__(self):
            active.add(self.device.uuid)
            events.append(('acquire', self.device.uuid))
            self.acquired = True
            return self
        def __exit__(self, *args):
            active.remove(self.device.uuid)
            events.append(('release', self.device.uuid))
            self.acquired = False
    def state(device):
        assert active == {'GPU-0', 'GPU-1'}
        occupied = device.visible_index == busy
        return SimpleNamespace(compute_apps=[SimpleNamespace(pid='123')] if occupied else [],
                               utilization_gpu_pct=0, as_dict=lambda: dict(occupied=occupied))
    monkeypatch.setattr(gpu_lock, 'resolve_device', resolve)
    monkeypatch.setattr(gpu_lock, 'GpuLock', Lock)
    monkeypatch.setattr(gpu_lock, 'query_gpu_state', state)
    return active, events


def test_all_locks_precede_runtime_and_environment_is_restored(monkeypatch):
    active, events = fake_devices(monkeypatch)
    monkeypatch.setenv('ROSHAMBO3_NUMERICS_DEVICE_IDS', '7')
    def run():
        assert active == {'GPU-0', 'GPU-1'}
        assert os.environ['ROSHAMBO3_NUMERICS_DEVICE_IDS'] == '1,0'
        return 0
    code, receipt = run_locked((1, 0), run)
    assert code == 0
    assert [entry['visible_index'] for entry in receipt] == [1, 0]
    assert all(entry['lock_held'] for entry in receipt)
    assert events[:2] == [('acquire', 'GPU-0'), ('acquire', 'GPU-1')]
    assert not active
    assert os.environ['ROSHAMBO3_NUMERICS_DEVICE_IDS'] == '7'


def test_busy_second_device_prevents_runtime_and_releases_all(monkeypatch):
    active, _ = fake_devices(monkeypatch, busy=1)
    with pytest.raises(RuntimeError, match='GPU busy.*GPU-1'):
        run_locked((0, 1), lambda: pytest.fail('must not create a CUDA context'))
    assert not active


def test_runtime_failure_releases_all_and_removes_opt_in(monkeypatch):
    active, _ = fake_devices(monkeypatch)
    monkeypatch.delenv('ROSHAMBO3_NUMERICS_DEVICE_IDS', raising=False)
    def fail():
        raise RuntimeError('native failure')
    with pytest.raises(RuntimeError, match='native failure'):
        run_locked((0, 1), fail)
    assert not active
    assert 'ROSHAMBO3_NUMERICS_DEVICE_IDS' not in os.environ


def test_explicit_existing_allocation_is_preserved_in_receipt(monkeypatch):
    active, _ = fake_devices(monkeypatch, busy=1)
    code, receipt = run_locked((0, 1), lambda: 0, allowed_pids=[123],
                              allocation_reason='User approved existing idle allocation')
    assert code == 0 and not active
    assert receipt[1]['preflight']['occupied'] is True
    assert receipt[1]['allowed_existing_pids'] == ['123']
    assert receipt[1]['correctness_only'] is True
    assert receipt[1]['allocation_exception_reason']


def test_allowing_one_pid_does_not_allow_other_allocations(monkeypatch):
    active, _ = fake_devices(monkeypatch, busy=1)
    with pytest.raises(RuntimeError, match='GPU busy'):
        run_locked((0, 1), lambda: pytest.fail('unexpected runtime'),
                   allowed_pids=[456], allocation_reason='A different approved PID')
    assert not active


def test_allocation_exception_requires_reason_before_acquiring_locks(monkeypatch):
    active, events = fake_devices(monkeypatch)
    with pytest.raises(ValueError, match='explicit reason'):
        run_locked((0, 1), lambda: pytest.fail('unexpected runtime'), allowed_pids=[123])
    assert not active and not events

"""OS services are explicit and scoped, including persistent worker policy."""
import sys
import threading

import pytest

from roshambo3._cpu_platform import qos_scope, validate_qos
from roshambo3.backend import cpu


def test_platform_affinity_and_capabilities():
    info = cpu.extension().cpu_platform_info()
    assert info["affinity_supported"] == (sys.platform == "linux")
    assert isinstance(info["fp16"], bool)
    assert isinstance(info["fp16_fml"], bool)
    if sys.platform == "darwin":
        assert info["allowed_cpus"] == []
        with pytest.raises(RuntimeError, match="affinity"):
            cpu.CPUComboExecutor(2, require_worker_affinity=True)


def test_unknown_policy_and_unsupported_platform_refused():
    with pytest.raises(ValueError, match="cpu_qos"):
        validate_qos("p_core")
    if sys.platform != "darwin":
        with pytest.raises(ValueError, match="macOS"):
            cpu.CPUComboExecutor(2, cpu_qos="background")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS QoS")
def test_unspecified_caller_qos_is_refused_before_mutation(monkeypatch):
    native = cpu.extension()
    monkeypatch.setattr(native, "_qos_snapshot", lambda: (0, 0))
    monkeypatch.setattr(native, "_set_qos", lambda _: pytest.fail("changed caller QoS"))
    with pytest.raises(RuntimeError, match="unspecified"):
        with qos_scope("background"):
            pytest.fail("entered unsupported scope")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS QoS")
@pytest.mark.parametrize("policy", ["background", "user_initiated"])
def test_qos_restored_after_failure_and_workers_keep_their_policy(policy):
    native = cpu.extension()
    before = native._qos_snapshot()
    with pytest.raises(RuntimeError, match="deliberate"):
        with qos_scope(policy):
            assert native._current_qos() == policy
            with cpu.CPUComboExecutor(2, cpu_qos=policy) as executor:
                with pytest.raises(AttributeError):
                    executor.cpu_qos = "inherit"
                observations = []
                executor.run_indexed(4, lambda worker, item:
                    observations.append(native._current_qos()), rows_per_item=128)
                assert observations == [policy] * 4
                assert all(p["qos_observed"] == policy
                           for p in executor.stats["worker_placement"])
                with pytest.raises(ValueError, match="match"):
                    executor.run_indexed(1, lambda *args: None,
                        cpu_qos="background" if policy == "user_initiated" else "user_initiated")
            raise RuntimeError("deliberate")
    assert native._qos_snapshot() == before


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS QoS")
def test_concurrent_policies_are_thread_local():
    barrier = threading.Barrier(2)
    observations, errors = [], []
    def worker(policy):
        try:
            with qos_scope(policy):
                barrier.wait(timeout=5)
                observations.append((policy, cpu.extension()._current_qos()))
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=worker, args=(policy,))
               for policy in ("background", "user_initiated")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors
    assert sorted(observations) == [("background", "background"),
                                    ("user_initiated", "user_initiated")]

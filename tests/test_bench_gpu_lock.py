"""Per-device resolution, the compute-apps filter, and the per-device
lock.

Most of this module's logic is pure ``nvidia-smi``-output parsing plus
string/env handling, so it is tested here without any GPU at all -- by
monkeypatching :func:`bench.gpu_lock._list_physical_gpus` (the one function
that shells out) and, separately, :func:`bench.gpu_lock._run_nvidia_smi`.
Tests that instead exercise the real ``nvidia-smi`` binary carry ``cuda`` in
their name (bare substring, per this project's ``pytest -k "not cuda"``
convention) and skip cleanly with a reason when it is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from bench import gpu_lock

TWO_TITAN_V = [(0, "GPU-synthetic-device-0"),
               (1, "GPU-synthetic-device-1")]
ONE_BLACKWELL = [(0, "GPU-synthetic-device-2")]


def _have_nvidia_smi() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
    except OSError:
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


needs_real_gpu = pytest.mark.skipif(
    not _have_nvidia_smi(),
    reason="no nvidia-smi / no GPU visible on this host")


class TestResolveDeviceUnset:
    """CUDA_VISIBLE_DEVICES unset: nvidia-smi's own order is the process's."""

    def test_single_gpu(self, monkeypatch):
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: ONE_BLACKWELL)
        d = gpu_lock.resolve_device(visible_index=0, cuda_visible_devices=None)
        assert d.visible_index == 0
        assert d.physical_index == 0
        assert d.uuid == ONE_BLACKWELL[0][1]

    def test_out_of_range_index_refuses_rather_than_guessing(self, monkeypatch):
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: ONE_BLACKWELL)
        with pytest.raises(gpu_lock.DeviceResolutionError):
            gpu_lock.resolve_device(visible_index=5, cuda_visible_devices=None)


class TestResolveDeviceRenumbering:
    """The trap this ticket exists to close: CUDA_VISIBLE_DEVICES renumbers
    devices, so the process's visible index is not the nvidia-smi index."""

    def test_single_entry_that_is_not_index_zero(self, monkeypatch):
        """The bug this ticket's original defect looked like from outside:
        CUDA_VISIBLE_DEVICES=1 means the process's device 0 IS the
        machine's device 1, not device 0."""
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: TWO_TITAN_V)
        d = gpu_lock.resolve_device(visible_index=0, cuda_visible_devices="1")
        assert d.visible_index == 0
        assert d.physical_index == 1
        assert d.uuid == TWO_TITAN_V[1][1]

    def test_multi_entry_list_reorders(self, monkeypatch):
        """The SUBTLER re-introduction of the bug the ticket warns about: a
        naive fix that reads CUDA_VISIBLE_DEVICES's raw text as if it were
        already the nvidia-smi index breaks the moment the list has more
        than one entry, because the process's device N is entries[N], not
        the literal Nth nvidia-smi index."""
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: TWO_TITAN_V)
        d0 = gpu_lock.resolve_device(visible_index=0, cuda_visible_devices="1,0")
        d1 = gpu_lock.resolve_device(visible_index=1, cuda_visible_devices="1,0")
        assert (d0.physical_index, d0.uuid) == (1, TWO_TITAN_V[1][1])
        assert (d1.physical_index, d1.uuid) == (0, TWO_TITAN_V[0][1])
        # And a naive "just use the raw entry" reading would have gotten d0
        # right (entries[0] == "1") but silently wrong on d1's physical
        # index if the code assumed visible_index==0 always -- the whole
        # point of visible_index being an explicit argument, not implied.
        assert d0.physical_index != d1.physical_index

    def test_uuid_entries_are_accepted_directly(self, monkeypatch):
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: TWO_TITAN_V)
        uuid1 = TWO_TITAN_V[1][1]
        d = gpu_lock.resolve_device(visible_index=0, cuda_visible_devices=uuid1)
        assert d.physical_index == 1
        assert d.uuid == uuid1

    def test_unresolvable_uuid_refuses(self, monkeypatch):
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: TWO_TITAN_V)
        with pytest.raises(gpu_lock.DeviceResolutionError):
            gpu_lock.resolve_device(visible_index=0, cuda_visible_devices="GPU-doesnotexist")

    def test_garbage_entry_refuses_with_a_reason(self, monkeypatch):
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: TWO_TITAN_V)
        with pytest.raises(gpu_lock.DeviceResolutionError, match="neither an integer index"):
            gpu_lock.resolve_device(visible_index=0, cuda_visible_devices="not-a-device")

    def test_visible_index_beyond_the_list_refuses(self, monkeypatch):
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: TWO_TITAN_V)
        with pytest.raises(gpu_lock.DeviceResolutionError):
            gpu_lock.resolve_device(visible_index=2, cuda_visible_devices="0,1")

    def test_reads_the_real_environment_by_default(self, monkeypatch):
        """cuda_visible_devices=_UNSET (the default) means "read
        os.environ", distinct from an explicit None (unset-and-known)."""
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: TWO_TITAN_V)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        d = gpu_lock.resolve_device(visible_index=0)
        assert d.physical_index == 1


class TestResolveDefaultDevice:
    def test_refuses_to_guess_on_a_multi_gpu_host(self, monkeypatch):
        """Silently reporting device 0 on a two-GPU host, whatever card ran
        the kernel, is the exact defect -- resolve_default_device must
        refuse instead of guessing."""
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: TWO_TITAN_V)
        with pytest.raises(gpu_lock.DeviceResolutionError):
            gpu_lock.resolve_default_device()

    def test_resolves_on_a_single_gpu_host(self, monkeypatch):
        monkeypatch.setattr(gpu_lock, "_list_physical_gpus", lambda: ONE_BLACKWELL)
        d = gpu_lock.resolve_default_device()
        assert d.physical_index == 0
        assert d.uuid == ONE_BLACKWELL[0][1]


class TestQueryGpuStateIsPerDevice:
    """The ticket's core bug, reproduced with a fabricated two-GPU
    ``nvidia-smi`` transcript rather than requiring a second physical card."""

    def test_only_this_devices_index_is_queried(self, monkeypatch):
        device = gpu_lock.DeviceIdentity(visible_index=0, physical_index=1,
                                         uuid=TWO_TITAN_V[1][1])
        seen_args: list[list[str]] = []

        def fake_run(args):
            seen_args.append(args)
            if "--query-compute-apps" in args[0]:
                return ("pid, process_name, used_gpu_memory [MiB], gpu_uuid\n")
            # --query-gpu path: assert -i names the PHYSICAL index, not 0.
            assert "-i" in args
            assert args[args.index("-i") + 1] == "1"
            return "utilization.gpu [%], memory.used [MiB], power.draw [W]\n5, 100, 50.0\n"

        monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", fake_run)
        st = gpu_lock.query_gpu_state(device)
        assert st.utilization_gpu_pct == 5.0
        assert st.power_draw_w == 50.0

    def test_a_process_on_the_other_card_is_not_contention(self, monkeypatch):
        """The other half of the ticket's bug: compute-apps has no device
        column by default, so a stale allocation on the OTHER card must be
        filtered out by gpu_uuid, not read as contention here."""
        device = gpu_lock.DeviceIdentity(visible_index=0, physical_index=1,
                                         uuid=TWO_TITAN_V[1][1])

        def fake_run(args):
            if "--query-compute-apps" in args[0]:
                other_uuid = TWO_TITAN_V[0][1]
                return (
                    "pid, process_name, used_gpu_memory [MiB], gpu_uuid\n"
                    f"12345, some_other_process, 400, {other_uuid}\n"
                )
            return "utilization.gpu [%], memory.used [MiB], power.draw [W]\n0, 0, 20.0\n"

        monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", fake_run)
        st = gpu_lock.query_gpu_state(device)
        assert not st.other_process_present
        assert st.compute_apps == []

    def test_a_process_on_this_card_IS_contention(self, monkeypatch):
        device = gpu_lock.DeviceIdentity(visible_index=0, physical_index=1,
                                         uuid=TWO_TITAN_V[1][1])

        def fake_run(args):
            if "--query-compute-apps" in args[0]:
                return (
                    "pid, process_name, used_gpu_memory [MiB], gpu_uuid\n"
                    f"12345, someone_elses_job, 400, {device.uuid}\n"
                )
            return "utilization.gpu [%], memory.used [MiB], power.draw [W]\n80, 400, 250.0\n"

        monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", fake_run)
        st = gpu_lock.query_gpu_state(device, exclude_pids=set())
        assert st.other_process_present
        assert st.compute_apps[0].pid == "12345"

    def test_our_own_pid_is_excluded_from_contention(self, monkeypatch):
        import os

        device = gpu_lock.DeviceIdentity(visible_index=0, physical_index=1,
                                         uuid=TWO_TITAN_V[1][1])
        own_pid = os.getpid()

        def fake_run(args):
            if "--query-compute-apps" in args[0]:
                return (
                    "pid, process_name, used_gpu_memory [MiB], gpu_uuid\n"
                    f"{own_pid}, pytest, 400, {device.uuid}\n"
                )
            return "utilization.gpu [%], memory.used [MiB], power.draw [W]\n0, 400, 30.0\n"

        monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", fake_run)
        st = gpu_lock.query_gpu_state(device)
        assert not st.other_process_present


class TestPerDeviceLock:
    """The lock is keyed on (host, device UUID) -- see
    roshambo3.paths.gpu_lock_path -- so it never needs a real GPU to test."""

    def _dev(self, uuid):
        return gpu_lock.DeviceIdentity(visible_index=0, physical_index=0, uuid=uuid)

    def test_lock_path_is_keyed_by_device_uuid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROSHAMBO3_GPU_LOCK", str(tmp_path / "gpu.lock"))
        a = gpu_lock.GpuLock("T73-test", self._dev("GPU-aaaa"))
        b = gpu_lock.GpuLock("T73-test", self._dev("GPU-bbbb"))
        assert a.lock_path != b.lock_path
        assert "GPU-aaaa" in str(a.lock_path)
        assert "GPU-bbbb" in str(b.lock_path)

    def test_two_different_devices_do_not_serialize(self, tmp_path, monkeypatch):
        """Lane 5 on GPU 0 and Lane 6 on GPU 1 must not block each other."""
        monkeypatch.setenv("ROSHAMBO3_GPU_LOCK", str(tmp_path / "gpu.lock"))
        lock_a = gpu_lock.GpuLock("lane5", self._dev("GPU-aaaa"), blocking=False)
        lock_b = gpu_lock.GpuLock("lane6", self._dev("GPU-bbbb"), blocking=False)
        lock_a.acquire()
        try:
            lock_b.acquire()  # must NOT raise GpuLockBusyError
            lock_b.release()
        finally:
            lock_a.release()

    def test_a_second_claimant_on_the_SAME_device_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROSHAMBO3_GPU_LOCK", str(tmp_path / "gpu.lock"))
        first = gpu_lock.GpuLock("lane5", self._dev("GPU-aaaa"), blocking=False)
        first.acquire()
        try:
            second = gpu_lock.GpuLock("lane5b", self._dev("GPU-aaaa"), blocking=False)
            with pytest.raises(gpu_lock.GpuLockBusyError):
                second.acquire()
        finally:
            first.release()

    def test_lock_file_records_the_device(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROSHAMBO3_GPU_LOCK", str(tmp_path / "gpu.lock"))
        lock = gpu_lock.GpuLock("T73-test", self._dev("GPU-cccc"))
        with lock:
            content = lock.lock_path.read_text()
        assert "GPU-cccc" in content
        assert "device_physical_index=0" in content


class TestRealNvidiaSmiCuda:
    """The one class in this module that talks to a real GPU."""

    @needs_real_gpu
    def test_resolve_default_device_matches_nvidia_smi_l(self):
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                              "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        if len(out.strip().splitlines()) > 1:
            with pytest.raises(gpu_lock.DeviceResolutionError, match="resolved explicitly"):
                gpu_lock.resolve_default_device()
        else:
            assert gpu_lock.resolve_default_device().uuid in out

    @needs_real_gpu
    def test_query_gpu_state_cuda_runs_end_to_end(self):
        device = gpu_lock.resolve_device(visible_index=0)
        st = gpu_lock.query_gpu_state(device)
        assert st.device.uuid == device.uuid
        assert st.utilization_gpu_pct >= 0.0

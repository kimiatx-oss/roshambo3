"""GPU arbitration protocol -- the shared-machine lock.

This machine's GPU is used by other human users and by other ROSHAMBO3
runs at the same time as any benchmark we run. The protocol has two
layers, and both are mandatory for anything whose number will be quoted:

1. **The flock lock** (:func:`roshambo3.paths.gpu_lock_path`) serializes
   *our own* timed runs against each other, across every run that uses
   this harness. It is a cooperative lock: it only protects agents that go
   through :class:`GpuLock`.
2. **The ``nvidia-smi --query-compute-apps`` check** detects *anyone else's*
   use of the GPU, cooperative or not (other human users, containers outside
   our PID namespace, etc.) -- this is not something a flock can arbitrate,
   only observe.

Failing (2) does not necessarily abort the run --
"a run that finds the GPU busy and proceeds anyway is advisory, never
decision" -- but failing (1) (another *decision-grade* run already holds the
lock) is a hard refusal: two ROSHAMBO3 decision-grade timed runs must never
overlap on the *same GPU*.

Per-device correctness
----------------------

An earlier version of this module was device-blind in three ways:

* ``query_gpu_state()`` called ``nvidia-smi --query-gpu=...`` with **no
  ``-i``**, so on a two-GPU host it always reported device 0 -- whatever
  card actually ran the kernel.
* the compute-apps check spanned every device with no device column, so a
  stale allocation on the card *not* in use read as contention on a run that
  never touched it.
* the lock file (``$ROSHAMBO3_GPU_LOCK``) was one literal path with no host
  or device component, so two independent lanes on a two-GPU host's cards would
  have serialized against each other for no reason, and a lock on one host
  said nothing about another host sharing the same path over NFS.

:class:`DeviceIdentity` and :func:`resolve_device` fix the first two by
making every ``nvidia-smi`` call in this module take an explicit,
UUID-anchored device rather than trusting positional defaults or
``CUDA_VISIBLE_DEVICES`` alone (that variable renumbers devices -- the
process's device 0 may be the machine's device 1, and a "fix" that reads the
variable's raw text as if it were already an ``nvidia-smi`` index
reintroduces the same bug in a subtler form the moment the variable holds
more than one entry). :func:`roshambo3.paths.gpu_lock_path` fixes the third
by keying the lock file on host **and** device UUID.

All device resolution here goes through ``nvidia-smi`` subprocess calls only
-- never a CUDA runtime/driver call -- so it can run, and the per-device lock
can be acquired, **before any CUDA context exists** (a per-device lock
that needs a device query to pick its path must make that query without
creating a context, or lock and context deadlock).
"""
from __future__ import annotations

import csv
import fcntl
import io
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from roshambo3 import paths

#: Sentinel distinguishing "the caller did not pass this" from "the caller
#: explicitly passed None" (a real, valid state: CUDA_VISIBLE_DEVICES unset).
_UNSET = object()


class GpuLockBusyError(RuntimeError):
    """Raised when a decision-grade run cannot acquire the flock because
    another ROSHAMBO3 run currently holds it (on the same device)."""


class DeviceResolutionError(ValueError):
    """Raised when a device cannot be unambiguously resolved -- e.g. an
    out-of-range ``visible_index``, or a ``CUDA_VISIBLE_DEVICES`` entry
    ``nvidia-smi`` does not recognise. Deliberately loud: a silent fallback
    here (e.g. "assume index 0") is exactly the bug this module exists to
    close."""


@dataclass(frozen=True)
class DeviceIdentity:
    """Which physical GPU a process means when it says "the GPU it's using".

    Three numbers, because two of them are ambiguous on a multi-GPU host:

    * ``visible_index`` -- the CUDA ordinal *inside a process* (what
      ``cudaGetDevice``/``torch.cuda.current_device()`` would report for
      device 0). Meaningless outside that process: ``CUDA_VISIBLE_DEVICES``
      renumbers it.
    * ``physical_index`` -- the ``nvidia-smi`` index for the same card. This
      is what every ``-i`` argument in this module takes, and getting it
      confused with ``visible_index`` is the device-blindness bug (and the subtler
      re-introduction of it: reading ``CUDA_VISIBLE_DEVICES``'s raw text and
      using it as this value directly, which breaks the moment the variable
      lists more than one entry or lists UUIDs).
    * ``uuid`` -- the identifier that survives a ``CUDA_VISIBLE_DEVICES``
      remap and a reboot's PCI re-enumeration. Used to key the per-device
      lock file (:func:`roshambo3.paths.gpu_lock_path`) and to filter the
      compute-apps contention check; recorded on every run record so a
      reader never has to trust an index alone.
    """

    visible_index: int
    physical_index: int
    uuid: str

    def as_dict(self) -> dict:
        return {
            "visible_index": self.visible_index,
            "physical_index": self.physical_index,
            "uuid": self.uuid,
        }


def _run_nvidia_smi(args: list[str]) -> str:
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, check=True
    ).stdout


def _list_physical_gpus() -> list[tuple[int, str]]:
    """``[(physical_index, uuid), ...]`` in ``nvidia-smi``'s own enumeration
    order. A pure query: no CUDA context, no environment interpretation --
    safe to call before a CUDA context exists."""
    out = _run_nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader"])
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1]))
    return rows


def resolve_device(visible_index: int = 0, cuda_visible_devices=_UNSET) -> DeviceIdentity:
    """Resolve which physical GPU a process sees as ``cuda:<visible_index>``,
    without creating a CUDA context.

    **The trap this function exists to close.** ``CUDA_VISIBLE_DEVICES``
    renumbers devices: with it set to ``"1"``, the process's device 0 is the
    machine's device 1. The ticket's original bug was code that queried
    ``nvidia-smi`` with no ``-i`` at all (always device 0, regardless of
    which card ran the kernel). The *subtler* re-introduction of the same
    bug is code that "fixes" that by reading ``CUDA_VISIBLE_DEVICES`` and
    using its raw text directly as the ``nvidia-smi`` index -- correct only
    by accident when the variable holds exactly one integer entry equal to
    ``visible_index``'s target, and wrong the moment it holds more than one
    entry (``visible_index`` other than 0) or a UUID (``GPU-...`` is a legal
    ``CUDA_VISIBLE_DEVICES`` entry, and not an index at all).

    Parameters
    ----------
    visible_index:
        The CUDA ordinal *inside the environment being resolved* -- 0 for a
        process that does not further restrict itself. This must come from
        an explicit argument (the caller's own knowledge of which device it
        means to use), never inferred.
    cuda_visible_devices:
        The ``CUDA_VISIBLE_DEVICES`` value to resolve against. Defaults to
        reading it from this process's own environment; pass it explicitly
        to resolve a *different* environment (e.g. one about to be handed to
        a child process) without mutating ``os.environ`` first.

    Returns
    -------
    DeviceIdentity
        Carrying both the visible index and the resolved physical index and
        UUID, so a run record can state all three rather than only the one
        that happens to be unambiguous.

    Notes
    -----
    If ``CUDA_VISIBLE_DEVICES`` is unset, every ``nvidia-smi``-enumerated GPU
    is visible, and this function takes ``nvidia-smi``'s own enumeration
    order as the process's order. That is the conventional assumption and
    holds on any host with identical GPUs, but it is
    not something this function -- or anything running outside the CUDA
    runtime -- can verify: CUDA's *unrestricted* default device order is
    "fastest first" by compute capability, which need not match
    ``nvidia-smi``'s PCI-bus order on a host with non-identical cards. The
    working rule for a multi-GPU host is therefore to set
    ``CUDA_VISIBLE_DEVICES`` explicitly rather than rely on this fallback.
    """
    if cuda_visible_devices is _UNSET:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    gpus = _list_physical_gpus()
    by_index = {i: u for i, u in gpus}
    by_uuid = {u: i for i, u in gpus}

    if cuda_visible_devices is None or not cuda_visible_devices.strip():
        if visible_index not in by_index:
            raise DeviceResolutionError(
                f"visible_index={visible_index} but nvidia-smi only "
                f"enumerates physical indices {sorted(by_index)} and "
                "CUDA_VISIBLE_DEVICES is unset")
        physical_index = visible_index
        return DeviceIdentity(visible_index, physical_index, by_index[physical_index])

    entries = [e.strip() for e in cuda_visible_devices.split(",") if e.strip()]
    if visible_index >= len(entries):
        raise DeviceResolutionError(
            f"visible_index={visible_index} but CUDA_VISIBLE_DEVICES="
            f"{cuda_visible_devices!r} lists only {len(entries)} device(s)")
    entry = entries[visible_index]

    if entry.upper().startswith("GPU-") or entry.upper().startswith("MIG-"):
        if entry not in by_uuid:
            raise DeviceResolutionError(
                f"CUDA_VISIBLE_DEVICES entry {entry!r} (visible_index="
                f"{visible_index}) is not a UUID nvidia-smi enumerates "
                f"({sorted(by_uuid)})")
        return DeviceIdentity(visible_index, by_uuid[entry], entry)

    try:
        physical_index = int(entry)
    except ValueError:
        raise DeviceResolutionError(
            f"CUDA_VISIBLE_DEVICES entry {entry!r} (visible_index="
            f"{visible_index}) is neither an integer index nor a GPU-/MIG- "
            "UUID") from None
    if physical_index not in by_index:
        raise DeviceResolutionError(
            f"CUDA_VISIBLE_DEVICES names physical index {physical_index} "
            f"(visible_index={visible_index}), which nvidia-smi does not "
            f"enumerate (has {sorted(by_index)})")
    return DeviceIdentity(visible_index, physical_index, by_index[physical_index])


def resolve_default_device() -> DeviceIdentity:
    """:func:`resolve_device` for ``visible_index=0`` when that is
    unambiguous -- i.e. exactly one GPU is enumerated by ``nvidia-smi``
    regardless of ``CUDA_VISIBLE_DEVICES``.

    This exists only to keep single-GPU hosts working without every
    caller having to pass a device explicitly, while making a multi-GPU host
    refuse to guess. Raise :class:`DeviceResolutionError` on more
    than one enumerated GPU rather than silently picking one -- silently
    picking device 0 on a two-GPU host, whatever card the kernel actually
    ran on, is the entire defect this guards against.
    """
    gpus = _list_physical_gpus()
    if len(gpus) != 1:
        raise DeviceResolutionError(
            f"{len(gpus)} GPUs are enumerated by nvidia-smi; a device must be "
            "resolved explicitly via resolve_device() on a multi-GPU host, "
            "not defaulted."
        )
    return resolve_device(visible_index=0)


@dataclass(frozen=True)
class ComputeApp:
    pid: str
    process_name: str
    used_gpu_memory: str


@dataclass(frozen=True)
class GpuState:
    device: DeviceIdentity
    compute_apps: list[ComputeApp]
    utilization_gpu_pct: float
    memory_used_mib: float
    power_draw_w: float

    @property
    def other_process_present(self) -> bool:
        """True if any *foreign* process (excluding the caller's own pid/ppid,
        per :func:`query_gpu_state`'s ``exclude_pids``) is currently allocated
        on **this device**. Processes on the other card do not count -- that
        is exactly the per-device contention fix."""
        return len(self.compute_apps) > 0

    def as_dict(self) -> dict:
        return {
            "device": self.device.as_dict(),
            "compute_apps": [
                {"pid": a.pid, "process_name": a.process_name,
                 "used_gpu_memory": a.used_gpu_memory}
                for a in self.compute_apps
            ],
            "other_process_present": self.other_process_present,
            "utilization_gpu_pct": self.utilization_gpu_pct,
            "memory_used_mib": self.memory_used_mib,
            "power_draw_w": self.power_draw_w,
        }


def query_gpu_state(device: DeviceIdentity | None = None,
                     exclude_pids: set[int] | None = None) -> GpuState:
    """Run the two ``nvidia-smi`` checks the arbitration protocol requires
    before any timed run, scoped to one physical GPU.

    ``device`` defaults to :func:`resolve_default_device` -- i.e. it resolves
    automatically **only** when exactly one GPU is enumerated at all. On a host with more than one GPU this
    raises :class:`DeviceResolutionError` rather than silently reporting
    device 0: that silent default, whatever card actually ran the kernel, is
    the exact defect this guards against. A caller on a multi-GPU
    host must resolve and pass ``device`` explicitly.

    The pre-fix version of this function took no device at all and
    read ``nvidia-smi``'s first row (device 0) and every device's
    compute-apps rows indiscriminately, so on a two-GPU host it reported
    device 0's utilization/power whatever card ran the kernel, and a stale
    allocation on the *other* card read as contention on a run that never
    touched it. ``device`` (from :func:`resolve_device`) fixes both: the
    ``--query-gpu`` call is scoped with ``-i <physical_index>``, and the
    compute-apps rows are filtered to this device's ``gpu_uuid`` column
    in the shared harness.

    ``exclude_pids`` defaults to ``{os.getpid(), os.getppid()}``: without
    this, a process's own CUDA allocation
    on the card it is about to benchmark reads as somebody else's
    contention. The rule "fixing the contention check makes it more
    likely to fire correctly, not less" means this exclusion must be exact
    and narrow -- it removes exactly the caller's own two PIDs, nothing else.
    """
    if device is None:
        device = resolve_default_device()
    if exclude_pids is None:
        exclude_pids = {os.getpid(), os.getppid()}

    apps_csv = _run_nvidia_smi(
        ["--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
         "--format=csv"]
    )
    apps = []
    reader = csv.reader(io.StringIO(apps_csv))
    rows = list(reader)
    for row in rows[1:]:
        if len(row) < 4:
            continue
        pid_s, name, mem, uuid = (x.strip() for x in row[:4])
        if uuid != device.uuid:
            continue  # a process on the OTHER card is not contention
        if pid_s.isdigit() and int(pid_s) in exclude_pids:
            continue  # our own allocation is not foreign contention
        apps.append(ComputeApp(pid_s, name, mem))

    gpu_csv = _run_nvidia_smi(
        ["--query-gpu=utilization.gpu,memory.used,power.draw",
         "--format=csv,nounits", "-i", str(device.physical_index)]
    )
    reader = csv.reader(io.StringIO(gpu_csv))
    rows = list(reader)
    util, mem_used, power = 0.0, 0.0, 0.0
    if len(rows) > 1:
        util, mem_used, power = (float(x.strip()) for x in rows[1][:3])

    return GpuState(
        device=device,
        compute_apps=apps,
        utilization_gpu_pct=util,
        memory_used_mib=mem_used,
        power_draw_w=power,
    )


class GpuLock:
    """flock-based mutual exclusion for GPU-timed runs on **one physical
    device**, holding the ticket id and PID in the lock file for the run's
    duration, as the arbitration protocol requires.

    Usage as a context manager::

        device = resolve_device(visible_index=0)   # or resolve_default_device()
        with GpuLock("my-run", device, blocking=False) as lock:
            ...run the timed benchmark on that device...

    ``blocking=False`` (the default for decision-grade runs) raises
    :class:`GpuLockBusyError` immediately if another process holds the lock
    for this device, rather than silently queueing behind it -- a benchmark
    that blocks invisibly for an unknown time is worse than one that fails
    fast and lets the caller decide whether to wait or run advisory-unlocked.

    ``device`` (a :class:`DeviceIdentity`, from :func:`resolve_device`)
    now identifies which physical card this lock arbitrates, and the lock
    path defaults to :func:`roshambo3.paths.gpu_lock_path` keyed on that
    device's UUID and this host's name -- **not** a single shared path. Two
    lanes on a two-GPU host's cards therefore do not serialize against each other,
    while a second claimant on the *same* card still excludes cleanly.

    ``device`` defaults to :func:`resolve_default_device`, which resolves
    automatically only when exactly one GPU is enumerated at all. On a
    multi-GPU host it
    raises :class:`DeviceResolutionError` rather than silently locking device
    0's lock file regardless of which card is actually in use -- a caller
    there must resolve and pass ``device`` explicitly. Device resolution
    happens purely through ``nvidia-smi`` subprocess calls (see
    :func:`resolve_device`), so the lock can be, and must be, acquired
    **before any CUDA context exists** -- resolving it does not create
    one.
    """

    def __init__(
        self,
        ticket_id: str,
        device: DeviceIdentity | None = None,
        lock_path: str | None = None,
        blocking: bool = False,
        timeout_s: float | None = None,
    ):
        self.ticket_id = ticket_id
        self.device = device if device is not None else resolve_default_device()
        self.lock_path = Path(lock_path) if lock_path is not None else paths.gpu_lock_path(self.device.uuid)
        self.blocking = blocking
        self.timeout_s = timeout_s
        self._fh = None
        self.acquired = False
        self.was_held = False
        """True from the moment acquire() succeeds onward, and *never*
        reset by release() -- unlike .acquired, which release() clears (by
        design: .acquired means "currently holding", which is what a second
        caller deciding whether to block needs). This was a real
        footgun: a run record was built after the finally-block that calls
        release(), read .acquired at that point, and recorded
        gpu_lock_held=False for runs that held the lock for their entire
        timed region. A
        run-record field that answers "did this run ever hold the lock" must
        survive release(); use .was_held for that, not .acquired."""

    def __enter__(self) -> "GpuLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.lock_path, "a+")
        flags = fcntl.LOCK_EX
        if not self.blocking:
            flags |= fcntl.LOCK_NB

        deadline = None if self.timeout_s is None else time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(self._fh.fileno(), flags)
                break
            except BlockingIOError:
                if self.blocking and (deadline is None or time.monotonic() < deadline):
                    time.sleep(0.5)
                    continue
                self._fh.close()
                self._fh = None
                held_by = self._read_holder(self.lock_path)
                raise GpuLockBusyError(
                    f"{self.lock_path} (device uuid={self.device.uuid}, "
                    f"physical_index={self.device.physical_index}) is held by "
                    f"another run ({held_by}); refusing to start a second "
                    "decision-grade GPU run concurrently on this device. Run "
                    "advisory-unlocked if this is intentional."
                )

        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(
            f"ticket={self.ticket_id} pid={os.getpid()} "
            f"host={os.uname().nodename} device_uuid={self.device.uuid} "
            f"device_physical_index={self.device.physical_index} "
            f"time={time.time()}\n"
        )
        self._fh.flush()
        self.acquired = True
        self.was_held = True

    @staticmethod
    def _read_holder(path: Path) -> str:
        try:
            return path.read_text().strip() or "(empty lock file)"
        except OSError:
            return "(unreadable)"

    def release(self) -> None:
        if self._fh is not None:
            try:
                self._fh.seek(0)
                self._fh.truncate()
            finally:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                self._fh.close()
                self._fh = None
        self.acquired = False

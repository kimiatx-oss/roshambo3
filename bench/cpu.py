"""CPU-side measurement provenance: identity, contention, affinity, frequency.

The GPU half of this harness is `bench/clocks.py` and `bench/gpu_lock.py`, and
every rule in them is about a GPU. This is the CPU analogue, under the same
rules: clock discipline, the calibration/headline regimes, lock arbitration,
and "ask the binary what it contains".

**The one structural difference, and it shapes everything here: we cannot lock
CPU clocks and cannot change SMT on these machines.** The answer for the GPU
is to control the clock and record what was achieved; here only the second half
is available. What we *can* control is affinity. So:

* frequency is **recorded, never requested** -- and a run that did not sample it
  cannot claim a clock;
* affinity is **pinned and recorded**, and an unpinned timed run is advisory;
* SMT is **observed** and recorded as a property of the run, not a knob.

**Do not read the boost ceiling off ``lscpu``.** It reports
``CPU max MHz: 7022.94`` on one Zen 3 host, a driver-derived figure no Zen 3
part reaches; the sampler sees 4.5 GHz under load. That is the CPU form of the
rule that a clock nobody configured is not a measurement.
"""
from __future__ import annotations

import fcntl
import glob
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

_POLLER = str(Path(__file__).resolve().with_name("cpu_poller.py"))


# ---------------------------------------------------------------------------
# Identity: what this machine is, asked rather than assumed
# ---------------------------------------------------------------------------

def _lscpu() -> dict:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:
        return {}
    got = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            got[k.strip()] = v.strip()
    return got


#: ISA extensions worth recording for this workload. Not an exhaustive flag
#: dump: these are the ones a kernel's dispatch target turns on, so a record
#: that carries them can be read against the target that was selected.
ISA_FLAGS_OF_INTEREST = (
    "sse4_2", "avx", "avx2", "fma", "f16c", "bmi1", "bmi2",
    "avx512f", "avx512dq", "avx512bw", "avx512vl", "avx512vnni",
    "avx512_bf16", "avx512_fp16", "amx_tile", "amx_bf16",
    # aarch64, for when the ladder reaches it
    "asimd", "asimdhp", "sve", "sve2", "i8mm", "bf16",
)


@dataclass
class CpuIdentity:
    """What the CPU is, for the run record. Asked of the machine, not inferred.

    Hardware tables have been wrong about these machines before; nothing here
    is taken from a document.
    """

    host: str = ""
    model: str = ""
    arch: str = ""
    performance_cores: int | None = None
    efficiency_cores: int | None = None
    physical_cores: int = 0
    logical_cpus: int = 0
    threads_per_core: int = 1
    sockets: int = 1
    numa_nodes: int = 1
    l1d_bytes: int = 0
    l2_bytes: int = 0
    l3_bytes: int = 0
    l3_instances: int = 0
    isa_flags: list = field(default_factory=list)
    governor: str = ""
    scaling_driver: str = ""
    boost_enabled: "bool | None" = None
    lscpu_max_mhz: "float | None" = None
    """Recorded so a reader can see it was IGNORED, not overlooked. It is
    driver-derived and unreachable; the achieved distribution is what counts."""

    @property
    def smt_enabled(self) -> bool:
        return self.threads_per_core > 1

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["smt_enabled"] = self.smt_enabled
        return d


def _cache_bytes(text: str) -> tuple:
    """`lscpu`'s "768 KiB (24 instances)" -> (786432, 24)."""
    if not text:
        return 0, 0
    m = re.match(r"([\d.]+)\s*([KMG])iB(?:\s*\((\d+) instances?\))?", text)
    if not m:
        return 0, 0
    mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]
    return int(float(m.group(1)) * mult), int(m.group(3) or 1)


def _online_cpu_ids() -> list:
    """Online CPU ids, for displacing a helper process off the measured set."""
    out = []
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
        m = re.fullmatch(r"cpu(\d+)", os.path.basename(path))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def cpu_identity() -> CpuIdentity:
    """Identify this machine's CPU."""
    if sys.platform == "darwin":
        from .cpu_darwin import identity
        return CpuIdentity(host=socket.gethostname(), arch=platform.machine(),
                           **identity())
    L = _lscpu()
    l1d, _ = _cache_bytes(L.get("L1d cache", ""))
    l2, _ = _cache_bytes(L.get("L2 cache", ""))
    l3, l3n = _cache_bytes(L.get("L3 cache", ""))
    per_socket = int(L.get("Core(s) per socket", 0) or 0)
    sockets = int(L.get("Socket(s)", 1) or 1)
    flags = set((L.get("Flags", "") or "").split())
    boost = _read("/sys/devices/system/cpu/cpufreq/boost")
    try:
        maxmhz = float(L.get("CPU max MHz", "") or 0) or None
    except ValueError:
        maxmhz = None
    return CpuIdentity(
        host=socket.gethostname(),
        model=L.get("Model name", "") or platform.processor(),
        arch=L.get("Architecture", "") or platform.machine(),
        physical_cores=per_socket * sockets,
        logical_cpus=int(L.get("CPU(s)", os.cpu_count() or 0) or 0),
        threads_per_core=int(L.get("Thread(s) per core", 1) or 1),
        sockets=sockets,
        numa_nodes=int(L.get("NUMA node(s)", 1) or 1),
        l1d_bytes=l1d, l2_bytes=l2, l3_bytes=l3, l3_instances=l3n,
        isa_flags=[f for f in ISA_FLAGS_OF_INTEREST if f in flags],
        governor=_read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        scaling_driver=_read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
        boost_enabled=(None if boost == "" else boost == "1"),
        lscpu_max_mhz=maxmhz,
    )


# ---------------------------------------------------------------------------
# Affinity: the one thing we can actually control
# ---------------------------------------------------------------------------

def current_affinity() -> list:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):                                  # pragma: no cover
        return []


def pin_to(cpus) -> list:
    """Pin this process to ``cpus`` and return the affinity actually set.

    Returns rather than asserts, because the kernel may narrow the request
    (a cpuset, an offline CPU) and a caller that recorded what it *asked for*
    would be recording a claim rather than a fact.
    """
    if not hasattr(os, "sched_setaffinity"):
        raise NotImplementedError("strict CPU affinity is unsupported on this OS")
    want = sorted({int(c) for c in cpus})
    os.sched_setaffinity(0, set(want))
    return current_affinity()


def logical_cpus() -> list:
    """Every online logical CPU, siblings included.

    The counterpart to :func:`physical_core_cpus`, and the pair of them is how
    SMT is varied here: these machines cannot have SMT disabled and clocks
    cannot be locked, so "SMT off" is one thread pinned per physical core and
    "SMT on" is both siblings. Affinity is what can be controlled; the rest is
    recorded rather than controlled.
    """
    return list(_online_cpu_ids())


def physical_core_cpus() -> list:
    """One logical CPU per physical core -- the first sibling of each.

    The default pinning for a throughput measurement that is not itself about
    SMT: the prep pool was measured peaking at physical cores and getting
    *slower* past them, so a run that silently used both siblings of half the
    cores is measuring something else.
    """
    seen, out = set(), []
    for path in sorted(glob.glob(
            "/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list")):
        sibs = _read(path)
        if not sibs or sibs in seen:
            continue
        seen.add(sibs)
        first = sibs.replace("-", ",").split(",")[0]
        try:
            out.append(int(first))
        except ValueError:
            pass
    return sorted(out)


# ---------------------------------------------------------------------------
# Contention: "is anyone else using the machine?", asked of the CPU
# ---------------------------------------------------------------------------

@dataclass
class CpuContention:
    loadavg_1m: float
    loadavg_5m: float
    logical_cpus: int
    foreign_processes: list
    """(pid, %cpu, command) for processes over ``BUSY_PCT`` that are not us."""

    #: A process using at least this much of one core counts as foreign load.
    BUSY_PCT = 20.0

    @property
    def busy(self) -> bool:
        """Load above half the logical CPUs, or any foreign process running.

        Deliberately not "load > 0": this machine runs a desktop and a shell,
        and a threshold that fires always is one people learn to pass with a
        flag."""
        return (self.loadavg_1m > 0.5 * max(1, self.logical_cpus)
                or bool(self.foreign_processes))

    def as_dict(self) -> dict:
        return {"loadavg_1m": self.loadavg_1m, "loadavg_5m": self.loadavg_5m,
                "logical_cpus": self.logical_cpus,
                "foreign_processes": self.foreign_processes,
                "busy": self.busy}


def query_cpu_state() -> CpuContention:
    """Who else is on this machine.

    A GPU screen holding the GPU lock is heavy CPU load and shows up here,
    which is the case a GPU-only contention check cannot see at all.
    """
    try:
        la1, la5, _ = os.getloadavg()
    except OSError:                                         # pragma: no cover
        la1 = la5 = float("nan")
    # Us, our parent, and anything we spawned. `ps` itself lands in its own
    # output at a spuriously high pcpu -- it is a fresh process whose CPU time
    # is divided by an almost-zero lifetime -- and reporting the measurement's
    # own probe as foreign load would make every check fail.
    mine = {os.getpid(), os.getppid()}
    foreign = []
    try:
        command = (["ps", "-axo", "pid,ppid,pcpu,comm"] if sys.platform == "darwin"
                   else ["ps", "-eo", "pid,ppid,pcpu,comm", "--sort=-pcpu"])
        out = subprocess.run(command,
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines()[1:]:
            parts = line.split(None, 3)
            if len(parts) != 4:
                continue
            pid, ppid, pcpu, comm = parts
            try:
                pid_i, ppid_i, pcpu_f = int(pid), int(ppid), float(pcpu)
            except ValueError:
                continue
            if pid_i in mine or ppid_i in mine:
                continue
            if pcpu_f >= CpuContention.BUSY_PCT:
                foreign.append([pid_i, pcpu_f, comm.strip()])
    except Exception:
        pass
    return CpuContention(loadavg_1m=la1, loadavg_5m=la5,
                         logical_cpus=os.cpu_count() or 0,
                         foreign_processes=foreign)


# ---------------------------------------------------------------------------
# The achieved frequency distribution
# ---------------------------------------------------------------------------

class CpuFrequencySampler:
    """Sample per-core frequency for the duration of a timed run.

    Mirrors :class:`bench.clocks.ClockPowerSampler`, including running in its
    own OS process: a sampling thread competes for the very cores being
    measured, and does so hardest exactly when the machine is busiest.

    ``summary()`` reports the **achieved** distribution and never a requested
    value, because there is no requested value to report -- see this module's
    docstring. It also reports ``n_samples`` explicitly, so a one-sample
    distribution reads as one sample rather than silently as a clock.
    """

    #: The sampler was measured perturbing a pinned FMA benchmark by **2.7x**
    #: at 48 CPUs and 0.5 s, because under ``acpi-cpufreq`` each
    #: ``scaling_cur_freq`` read IPIs the target core. Both defaults below exist
    #: to keep that cost off the measurement: sample only the CPUs the caller
    #: pinned to, and sample them rarely. A caller that genuinely wants the
    #: whole machine's distribution must ask for it and should not be timing
    #: anything while it does.
    DEFAULT_INTERVAL_S = 2.0

    def __init__(self, interval_s: float = DEFAULT_INTERVAL_S, cpus=None):
        self.interval_s = float(interval_s)
        # None means "the CPUs this process is pinned to", not "all of them".
        if cpus is None:
            cpus = current_affinity()
            # An unpinned process is affine to everything, and sampling
            # everything is the expensive case. Fall back to the whole machine
            # only when affinity says nothing useful, and say so in the summary.
            if len(cpus) > 8:
                cpus = None
        self.cpus = None if cpus is None else set(int(c) for c in cpus)
        self.rows: list = []
        self._proc = None
        self._path = None
        self._t0 = None
        self._displaced_to = None

    def __enter__(self) -> "CpuFrequencySampler":
        self._t0 = time.monotonic()
        if sys.platform == "darwin":
            return self
        fd, self._path = tempfile.mkstemp(prefix="roshambo3-cpusample-",
                                          suffix=".csv")
        os.close(fd)
        self._fh = open(self._path, "w")
        cmd = [sys.executable, _POLLER, "--interval", str(self.interval_s)]
        if self.cpus:
            cmd += ["--cpus", ",".join(str(c) for c in sorted(self.cpus))]
        self._proc = subprocess.Popen(cmd, stdout=self._fh,
                                      stderr=subprocess.DEVNULL)
        # THE CHILD INHERITS OUR AFFINITY, WHICH IS THE ONE THING IT MUST NOT
        # HAVE. A pinned measurement pins its children too, so the poller's
        # ~40 ms of Python startup lands on exactly the core being measured.
        # This was measured as a 2.7x hit on the scalar FMA arms and none at all
        # on the AVX-512-width one -- because that arm runs last and startup was
        # over by then, which is precisely the shape of bug that reads as a real
        # effect. Push the child onto everything we are NOT measuring.
        self._displaced_to = None
        try:
            online = set(_online_cpu_ids())
            others = online - (self.cpus or set())
            if others:
                os.sched_setaffinity(self._proc.pid, others)
                self._displaced_to = sorted(others)
        except Exception:                                   # pragma: no cover
            # Better to sample with a perturbation we can name than not at all;
            # `summary()` reports whether the displacement took.
            pass
        return self

    def __exit__(self, *exc) -> None:
        if sys.platform == "darwin":
            return
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:               # pragma: no cover
                self._proc.kill()
        try:
            self._fh.close()
        except Exception:                                   # pragma: no cover
            pass
        self._read_back()

    def _read_back(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        with open(self._path) as fh:
            header = fh.readline()
            if not header.startswith("t_rel_s"):
                return
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                try:
                    t, cpu, khz, load = (float(parts[0]), int(parts[1]),
                                         float(parts[2]), float(parts[3]))
                except ValueError:
                    continue
                if self.cpus is not None and cpu not in self.cpus:
                    continue
                self.rows.append((t, cpu, khz, load))
        try:
            os.unlink(self._path)
        except OSError:                                     # pragma: no cover
            pass

    def summary(self) -> dict:
        if not self.rows:
            return {"n_samples": 0,
                    "note": "no frequency samples; this run may not claim a clock"}
        mhz = [r[2] / 1000.0 for r in self.rows]
        loads = [r[3] for r in self.rows]
        mhz_sorted = sorted(mhz)
        return {
            "n_samples": len(self.rows),
            "n_cpus_seen": len({r[1] for r in self.rows}),
            "cpus_sampled": (sorted(self.cpus) if self.cpus else "all online"),
            "interval_s": self.interval_s,
            # Sampling every core and taking the median describes the
            # IDLE cores, not the working one. On a single-threaded run with 47
            # cores at 1800 MHz the median IS 1800, which reads as a throttled
            # measurement and is nothing of the kind.
            "median_is_over_sampled_cpus_only": bool(self.cpus),
            # False means the poller shared the measured cores and the numbers
            # it accompanies carry its cost. Reported, never assumed.
            "poller_displaced_off_measured_cpus": self._displaced_to is not None,
            "duration_s": max(r[0] for r in self.rows),
            "achieved_mhz": {
                "min": mhz_sorted[0], "max": mhz_sorted[-1],
                "median": statistics.median(mhz),
                "mean": statistics.fmean(mhz),
                "p10": mhz_sorted[int(0.10 * (len(mhz_sorted) - 1))],
                "p90": mhz_sorted[int(0.90 * (len(mhz_sorted) - 1))],
            },
            # The rule that numbers from different clocks may not be
            # combined needs a way to say HOW different. This is it: a spread
            # is a property of the run, reported rather than asserted on.
            "achieved_mhz_spread": (mhz_sorted[-1] / mhz_sorted[0]
                                    if mhz_sorted[0] else None),
            "loadavg_1m": {"min": min(loads), "max": max(loads),
                           "median": statistics.median(loads)},
        }


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------

class CpuLockBusyError(RuntimeError):
    """Another run holds this host's CPU lock."""


class CpuLock:
    """flock-based mutual exclusion for CPU-timed runs on **one host**.

    The twin of :class:`bench.gpu_lock.GpuLock`, and it should have been
    written at the same time as the rule that requires it:
    ``rules.RunContext.cpu_lock_held`` is a *hard refusal* for a decision-grade
    CPU run. Without an implementation to call, 4 of 31 scripts referenced the
    lock at all, each by hand-rolling ``flock`` in a shell wrapper, and 0 of 28
    artifacts recorded that it was held.

    That is why this class exists rather than a note in a README: a rule whose
    obedient path is harder than its disobedient one selects for disobedience.

    Usage::

        with CpuLock("my-run") as lock:
            ...run the timed region...
        # lock.was_held is what a run record must carry -- see below.

    ``blocking=False`` (the default) raises :class:`CpuLockBusyError`
    immediately rather than queueing invisibly, matching ``GpuLock``: a
    benchmark that blocks for an unknown time is worse than one that fails fast
    and lets the caller choose to wait or to run advisory-unlocked.

    **Host-scoped, and nothing finer.** ``paths.cpu_lock_path`` explains why:
    two runs pinned to disjoint cores still contend for last-level cache,
    memory bandwidth and the boost budget, so a per-core-set lock would
    arbitrate the wrong thing while reading as safety.

    **Holding the lock does not mean the machine is yours**, only that no other
    *quoted* run is competing. Pair it with :func:`query_cpu_state`, which is
    what sees foreign processes (20 were once recorded while holding it).
    """

    def __init__(
        self,
        ticket_id: str,
        lock_path: "str | None" = None,
        blocking: bool = False,
        timeout_s: "float | None" = None,
    ):
        from roshambo3 import paths

        self.ticket_id = ticket_id
        self.lock_path = (Path(lock_path) if lock_path is not None
                          else paths.cpu_lock_path())
        self.blocking = blocking
        self.timeout_s = timeout_s
        self._fh = None
        self.acquired = False
        self.was_held = False
        """True from the moment :meth:`acquire` succeeds onward, and **never**
        reset by :meth:`release` -- unlike :attr:`acquired`, which release()
        clears because "am I currently holding it" is what a second claimant
        needs. A run record is built *after* the finally-block that releases,
        so a record that reads ``.acquired`` there records ``False`` for a run
        that held the lock throughout its timed region. That is not
        hypothetical: the GPU side once shipped exactly that defect.
        **Record ``.was_held``.**"""

    def __enter__(self) -> "CpuLock":
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

        deadline = (None if self.timeout_s is None
                    else time.monotonic() + self.timeout_s)
        while True:
            try:
                fcntl.flock(self._fh.fileno(), flags)
                break
            except BlockingIOError:
                if self.blocking and (deadline is None
                                      or time.monotonic() < deadline):
                    time.sleep(0.5)
                    continue
                self._fh.close()
                self._fh = None
                raise CpuLockBusyError(
                    f"{self.lock_path} is held by another run "
                    f"({self._read_holder(self.lock_path)}); refusing to start "
                    "a second decision-grade CPU run concurrently on this "
                    "host. Run advisory-unlocked if this is intentional."
                )

        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(
            f"ticket={self.ticket_id} pid={os.getpid()} "
            f"host={socket.gethostname()} time={time.time()}\n"
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

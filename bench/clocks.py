"""Clock locking and the 1 Hz clocks/power/temp/throttle log.

The ticket requires ``nvidia-smi -pm 1``, ``--lock-gpu-clocks``, and
``--lock-memory-clocks`` to be wired into the harness's setup path, and a
1 Hz log of ``clocks.sm``, ``power.draw``, ``temperature.gpu``, and
``clocks_throttle_reasons.active`` for the duration of every timed run.

On this machine those ``nvidia-smi`` clock-locking calls fail with
"Insufficient Permissions" for a non-root user (verified: ``nvidia-smi -lgc
2272,2272`` -> exit 4, "The current user does not have permission to change
clocks"). The harness still *issues* the commands (wiring them into the setup
path, as required) and records whether they actually took effect
(``gpu.clocks_locked``) rather than assuming success -- an unenforceable
lock silently believed to be enforced is worse than an honestly-recorded
unlocked run. Per the rules in :mod:`bench.rules`,
unlocked clocks alone do not downgrade a run to advisory; only unlocked
clocks *combined with observed throttling >5%* do.

**Power cap and profiler detection**: alongside the generic throttle-reasons mask, the
sampler now also tracks ``clocks_event_reasons.sw_power_cap`` specifically
(a run can report ``clocks_locked=True`` -- i.e. the lock command
itself succeeded -- and still not hold that clock, because the power ceiling
binds instead; that is a *power-limited* run, not a *clock-locked* one, and
the two must not be reported as the same claim). It also stamps the achieved
``clocks.sm`` distribution explicitly as min/median/max (never the
requested value -- that is exactly the error behind the
project's wrong "5.6% of the MUFU.EX2 ceiling" figure, see
:mod:`bench.rules`), and provides best-effort detection of whether
an ``ncu``/``nsys`` profiler is holding GPU clock control during the run
(``ncu --clock-control`` defaults to ``base`` and silently overrides
any external lock for the run's duration).

**Per-device, and a real subprocess sampler.**

Every ``nvidia-smi`` call below used to have no ``-i`` at all, so on a
two-GPU host every function here silently acted on (or reported) device 0,
whatever card actually ran the kernel. Every clock-affecting and
clock-reading function now takes an explicit ``device``
(:class:`bench.gpu_lock.DeviceIdentity`), defaulting to
:func:`bench.gpu_lock.resolve_default_device` -- which resolves
automatically only when exactly one GPU is enumerated at all, and refuses to
guess otherwise.

``ClockPowerSampler`` no longer samples from a background *thread* in this
process, because that
was wrong: a C++/CUDA extension is not required to release the GIL around a
synchronous call, so a thread sampler can go unscheduled for the entire
timed region -- it was measured at **one sample over a two minute benchmark**.
Sampling now happens in a genuinely separate OS process
(:mod:`bench.gpu_poller`, pinned to ``device`` with ``-i``), so its
scheduling does not depend on what this process's extension is doing.
Every :meth:`ClockPowerSampler.summary` carries ``n_samples`` explicitly, so
a one-sample run is visible as one sample, never mistaken for a clock
distribution with only one point plotted.
"""
from __future__ import annotations

import csv
import io
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from bench.gpu_lock import DeviceIdentity, resolve_default_device

_POLLER_SCRIPT = str(Path(__file__).resolve().with_name("gpu_poller.py"))


def _resolve(device: DeviceIdentity | None) -> DeviceIdentity:
    return device if device is not None else resolve_default_device()


def enable_persistence_mode(device: DeviceIdentity | None = None) -> bool:
    device = _resolve(device)
    r = subprocess.run(["nvidia-smi", "-i", str(device.physical_index), "-pm", "1"],
                       capture_output=True, text=True)
    return r.returncode == 0


def lock_clocks(gpu_clock_mhz: int | None = None, mem_clock_mhz: int | None = None,
                 device: DeviceIdentity | None = None) -> bool:
    """Best-effort: attempt to lock GPU and memory clocks at their reported
    max (or a caller-supplied value) **on ``device``**. Returns True iff
    nvidia-smi reports success for both -- False (with the attempt still
    made, satisfying the "wired into the setup path" requirement) if this
    user lacks permission, which is the observed case on shared machines
    without root.

    **Returns False when it did nothing.** A bare ``lock_clocks()`` (no clock
    values) resolves the max itself; if that cannot be read, it reports failure
    rather than success, so a run record never carries ``clocks_locked: true``
    having issued no command at all.

    Without root, ``nvidia-smi -lgc`` is refused ("The current user does not
    have permission to change clocks"). Then what keeps numbers comparable is
    clocks pinned *externally*, and what keeps a ratio honest is running its
    arms back-to-back in one session with the achieved distribution recorded
    (the headline regime) rather than this function.
    """
    device = _resolve(device)
    # A caller who names no clock means "lock at max", which is what the
    # docstring has always said. Resolve it rather than silently doing nothing.
    attempted = False
    if gpu_clock_mhz is None and mem_clock_mhz is None:
        try:
            gpu_clock_mhz = max_gpu_clock_mhz(device=device)
            mem_clock_mhz = max_mem_clock_mhz(device=device)
        except Exception:
            return False
        if gpu_clock_mhz is None and mem_clock_mhz is None:
            return False
    ok = True
    if gpu_clock_mhz is not None:
        attempted = True
        r = subprocess.run(
            ["nvidia-smi", "-i", str(device.physical_index),
             "-lgc", f"{gpu_clock_mhz},{gpu_clock_mhz}"],
            capture_output=True, text=True,
        )
        ok = ok and r.returncode == 0 and "not have permission" not in (
            r.stdout + r.stderr)
    if mem_clock_mhz is not None:
        attempted = True
        r = subprocess.run(
            ["nvidia-smi", "-i", str(device.physical_index),
             "-lmc", f"{mem_clock_mhz},{mem_clock_mhz}"],
            capture_output=True, text=True,
        )
        ok = ok and r.returncode == 0 and "not have permission" not in (
            r.stdout + r.stderr)
    # Never report success for an attempt that was never made.
    return bool(ok and attempted)


def reset_clocks(device: DeviceIdentity | None = None) -> None:
    device = _resolve(device)
    subprocess.run(["nvidia-smi", "-i", str(device.physical_index), "-rgc"],
                   capture_output=True, text=True)
    subprocess.run(["nvidia-smi", "-i", str(device.physical_index), "-rmc"],
                   capture_output=True, text=True)


def max_gpu_clock_mhz(device: DeviceIdentity | None = None) -> int | None:
    device = _resolve(device)
    r = subprocess.run(
        ["nvidia-smi", "-i", str(device.physical_index),
         "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    try:
        return int(float(r.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        return None


def max_mem_clock_mhz(device: DeviceIdentity | None = None) -> int | None:
    device = _resolve(device)
    r = subprocess.run(
        ["nvidia-smi", "-i", str(device.physical_index),
         "--query-gpu=clocks.max.mem", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    try:
        return int(float(r.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        return None


_FIELDS = [
    "clocks.sm", "clocks.mem", "power.draw", "temperature.gpu",
    "utilization.gpu", "clocks_throttle_reasons.active",
    "clocks_event_reasons.sw_power_cap",
]


@dataclass
class ClockSample:
    t_rel_s: float
    clocks_sm_mhz: float
    clocks_mem_mhz: float
    power_draw_w: float
    temperature_gpu_c: float
    utilization_gpu_pct: float
    throttle_reasons_active: str
    sw_power_cap_active: bool = False
    """``clocks_event_reasons.sw_power_cap`` specifically (both the old
    ``clocks_throttle_reasons.*`` and new ``clocks_event_reasons.*`` names
    resolve to the same counter on this driver; queried under the latter
    name). Distinguished from the generic
    throttle mask because "power-limited" and "clock-locked" are different
    claims (1800 MHz requested was measured at 1762 with this Active -- the
    lock did not bind)."""


class ClockPowerSampler:
    """Subprocess-sampled ``nvidia-smi`` log at ~1 Hz for the duration of a
    timed run, pinned to one physical GPU. Also integrates power draw into
    an energy (joules) total via the trapezoidal rule -- logged from day one
    per the ticket's explicit instruction, never gated on.

    Sampling happens in :mod:`bench.gpu_poller`, a separate OS process
    launched with ``-i <device.physical_index>``, not a background thread in
    this process. See the module docstring for why a thread sampler was
    measured to under-sample by up to 2 orders of magnitude on a real run.
    ``device`` defaults to :func:`bench.gpu_lock.resolve_default_device`
    (single-GPU hosts only); a multi-GPU host must pass it explicitly.
    """

    def __init__(self, device: DeviceIdentity | None = None, interval_s: float = 1.0):
        self.device = _resolve(device)
        self.interval_s = interval_s
        self.samples: list[ClockSample] = []
        self._proc: subprocess.Popen | None = None
        self._tmp_path: str | None = None
        self._t0 = None

    def __enter__(self) -> "ClockPowerSampler":
        self._t0 = time.monotonic()
        fd, self._tmp_path = tempfile.mkstemp(prefix="roshambo3-clocksample-",
                                              suffix=".csv")
        os.close(fd)
        out = open(self._tmp_path, "w")
        try:
            self._proc = subprocess.Popen(
                [sys.executable, _POLLER_SCRIPT,
                 "--index", str(self.device.physical_index),
                 "--fields", ",".join(_FIELDS),
                 "--interval", str(self.interval_s)],
                stdout=out, stderr=subprocess.DEVNULL,
            )
        finally:
            out.close()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=max(2.0, self.interval_s * 3))
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
                self._proc.wait(timeout=5)
            self._proc = None
        if self._tmp_path is not None:
            self._load(self._tmp_path)
            try:
                os.unlink(self._tmp_path)
            except OSError:  # pragma: no cover
                pass
            self._tmp_path = None
        return False

    def _load(self, path: str) -> None:
        """Parse the poller's output file into :attr:`samples`.

        Each line is ``"<poller's own time.monotonic()>,<nvidia-smi CSV
        row>"``. The poller's monotonic reading is directly comparable to
        this process's own (``CLOCK_MONOTONIC`` is machine-wide, not
        per-process), so ``t_rel_s`` is computed against *this* sampler's
        ``__enter__`` time, exactly as the old thread-based sampler defined
        it -- the timeline a reader sees is unchanged even though the
        sampling process is not this one.
        """
        try:
            with open(path) as fh:
                lines = fh.readlines()
        except OSError:  # pragma: no cover
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = next(csv.reader(io.StringIO(line)), None)
            if not row or len(row) < 1 + len(_FIELDS):
                continue
            try:
                poller_t = float(row[0])
                sample = ClockSample(
                    t_rel_s=poller_t - self._t0,
                    clocks_sm_mhz=float(row[1]),
                    clocks_mem_mhz=float(row[2]),
                    power_draw_w=float(row[3]),
                    temperature_gpu_c=float(row[4]),
                    utilization_gpu_pct=float(row[5]),
                    throttle_reasons_active=row[6].strip(),
                    sw_power_cap_active=(row[7].strip() == "Active"
                                         if len(row) > 7 else False),
                )
            except ValueError:
                continue
            self.samples.append(sample)

    def energy_joules(self) -> float:
        """Trapezoidal integral of power draw (W) over elapsed time (s)."""
        if len(self.samples) < 2:
            return 0.0
        j = 0.0
        for a, b in zip(self.samples, self.samples[1:]):
            dt = b.t_rel_s - a.t_rel_s
            j += 0.5 * (a.power_draw_w + b.power_draw_w) * dt
        return j

    def max_throttle_fraction(self) -> float:
        """Fraction of samples with a non-zero (non-"0x0000...0") active
        throttle-reason mask."""
        if not self.samples:
            return 0.0
        thr = sum(
            1 for s in self.samples
            if s.throttle_reasons_active not in ("0x0000000000000000", "N/A", "")
        )
        return thr / len(self.samples)

    def sw_power_cap_fraction(self) -> float:
        """Fraction of samples with ``sw_power_cap`` Active,
        tracked separately from the generic throttle mask above. A run
        whose fraction exceeds
        :data:`bench.rules.SW_POWER_CAP_ADVISORY_THRESHOLD` is
        *power-limited*, a distinct claim from "clock-locked" even when
        ``clocks_locked`` (the lock command's own reported success) is
        True -- that combination is exactly the measured 1800-requested/
        1762-achieved case."""
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s.sw_power_cap_active) / len(self.samples)

    def summary(self) -> dict:
        if not self.samples:
            # Report the sample count explicitly even when
            # it is zero, rather than an empty dict a caller might read as
            # "not sampled" (indistinguishable from "sampled, saw nothing").
            return {"n_samples": 0, "device": self.device.as_dict()}
        powers = [s.power_draw_w for s in self.samples]
        sms = [s.clocks_sm_mhz for s in self.samples]
        temps = [s.temperature_gpu_c for s in self.samples]
        return {
            "device": self.device.as_dict(),
            # On every record, so a one-sample
            # distribution reads as one sample rather than silently as a
            # clock -- a headline run's sampler must not repeat this failure in the other direction (a wide but
            # *real* distribution must not be flagged as an error either;
            # it is reported, not asserted).
            "n_samples": len(self.samples),
            "duration_s": self.samples[-1].t_rel_s,
            "power_draw_w": {"mean": sum(powers) / len(powers), "max": max(powers), "min": min(powers)},
            "clocks_sm_mhz": {"mean": sum(sms) / len(sms), "max": max(sms), "min": min(sms)},
            # The *achieved* clocks.sm distribution -- min/median/max --
            # never the requested value. Any "% of peak" figure derived from
            # this run must cite this distribution's clock (or at minimum its
            # median) as its normalizing clock; see
            # rules.find_percent_of_peak_fields_missing_clock.
            "achieved_clock_mhz": {
                "min": min(sms), "median": statistics.median(sms), "max": max(sms),
            },
            "temperature_gpu_c": {"mean": sum(temps) / len(temps), "max": max(temps)},
            "energy_j": self.energy_joules(),
            "throttle_fraction": self.max_throttle_fraction(),
            "sw_power_cap_fraction": self.sw_power_cap_fraction(),
        }

    def to_csv_rows(self) -> list[dict]:
        return [
            {
                "t_rel_s": s.t_rel_s, "clocks_sm_mhz": s.clocks_sm_mhz,
                "clocks_mem_mhz": s.clocks_mem_mhz, "power_draw_w": s.power_draw_w,
                "temperature_gpu_c": s.temperature_gpu_c,
                "utilization_gpu_pct": s.utilization_gpu_pct,
                "throttle_reasons_active": s.throttle_reasons_active,
                "sw_power_cap_active": s.sw_power_cap_active,
            }
            for s in self.samples
        ]


# ---------------------------------------------------------------------------
# Profiler clock-control detection
# ---------------------------------------------------------------------------

_PROFILER_BASENAMES = ("ncu", "nsys")
_CLOCK_CONTROL_RE = re.compile(r"--clock-control[= ](\S+)")


@dataclass(frozen=True)
class ProfilerProcess:
    tool: str
    pid: str
    clock_control: str | None
    """The tool's ``--clock-control``-equivalent setting, where determinable.
    For ``ncu`` specifically, an absent flag means the *default* value
    ``"base"`` -- not "unset"; ``ncu`` always seizes the clock unless
    told ``--clock-control=none``. ``nsys`` does not expose a consistently
    parseable equivalent, so this is left ``None`` for it (tool presence is
    still recorded)."""
    cmdline: str


@dataclass(frozen=True)
class ProfilerControlState:
    checked: bool
    """False iff the detection command itself could not be run (e.g. ``ps``
    missing/failed) -- see :attr:`state`."""
    processes: list[ProfilerProcess] = field(default_factory=list)

    @property
    def state(self) -> str:
        """One of ``"none"`` (checked, nothing found), ``"profiler-active"``
        (checked, found ncu/nsys), or ``"undetermined"`` (the check itself
        did not run). Per the ticket's own trap: a false negative in
        *detection* (profiler running but its process missed by this
        best-effort scan) must not block a run and is simply reported as
        "none" -- there is no way to rule that out from userspace. But if
        the check could not run at all, that is a stronger and different
        failure and must say "undetermined", never be silently folded into
        "none" as if it had been checked and found clean."""
        if not self.checked:
            return "undetermined"
        return "profiler-active" if self.processes else "none"

    @property
    def tool(self) -> str | None:
        return self.processes[0].tool if self.processes else None

    @property
    def clock_control_setting(self) -> str | None:
        return self.processes[0].clock_control if self.processes else None

    def as_dict(self) -> dict:
        return {
            "checked": self.checked,
            "state": self.state,
            "tool": self.tool,
            "clock_control_setting": self.clock_control_setting,
            "processes": [
                {"tool": p.tool, "pid": p.pid, "clock_control": p.clock_control,
                 "cmdline": p.cmdline}
                for p in self.processes
            ],
        }


def detect_profiler_clock_control() -> ProfilerControlState:
    """Best-effort detection of an ``ncu``/``nsys`` process holding GPU clock
    control during this run (this was diagnosed the hard way -- a
    locked-clock run showed 1740-1792 MHz and looked like a broken lock,
    when a concurrent ``ncu`` run was actually pinning the card at
    2265 MHz for its own duration). Matches on the process's own argv[0]
    basename (``ncu``/``nsys``), not a substring grep of the full command
    line, to avoid false positives from unrelated commands that merely
    mention "ncu"/"nsys" in an argument (e.g. a grep for them, or a
    docstring). A false negative here (the tool is running but this scan
    misses it) must not block a run -- it is recorded as ``"none"``, since
    there is no stronger claim available from userspace. A failure of the
    detection mechanism itself is different and is recorded as
    ``"undetermined"`` rather than assumed clean.

    Deliberately not device-scoped: ``ncu``/``nsys`` seize *the driver's*
    clock-control state, which (on the hardware this project has profiled
    so far) is a single knob, not one per GPU. If that stops being true on
    a host with independently-controllable per-device clock domains, this
    function's result should be intersected with which device the detected
    process's ``--clock-control`` argument (if any) actually named -- not
    assumed today, since ``ncu``'s CLI does not expose a per-device flag for
    it.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return ProfilerControlState(checked=False)
        lines = out.stdout.splitlines()[1:]  # drop the "PID COMMAND" header
    except (OSError, subprocess.SubprocessError):
        return ProfilerControlState(checked=False)

    procs: list[ProfilerProcess] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid, args = parts
        try:
            tokens = shlex.split(args)
        except ValueError:
            tokens = args.split()
        if not tokens:
            continue
        basename = tokens[0].rsplit("/", 1)[-1]
        if basename not in _PROFILER_BASENAMES:
            continue
        m = _CLOCK_CONTROL_RE.search(args)
        if m:
            clock_control = m.group(1)
        elif basename == "ncu":
            clock_control = "base"  # ncu's own documented default
        else:
            clock_control = None
        procs.append(ProfilerProcess(tool=basename, pid=pid, clock_control=clock_control,
                                      cmdline=args[:300]))
    return ProfilerControlState(checked=True, processes=procs)

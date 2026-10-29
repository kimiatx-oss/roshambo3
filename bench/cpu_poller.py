#!/usr/bin/env python3
"""Sample per-core CPU frequency and load into a CSV, from its own process.

The GPU sampler runs outside a background thread because a thread was
measured to under-sample by up to two orders of magnitude on a real run. The
argument is stronger here and for a different reason: a sampling thread does
not merely lose samples, it **competes for the very cores being measured**, and
it does so most when the machine is busiest -- which is exactly when the
distribution matters.

One row per (t_rel_s, cpu, khz, plus a machine-wide load average). Written
line-buffered so a run killed mid-flight still leaves usable samples, because
partial results are results.

Frequency comes from ``/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq``
where the driver exposes it, and falls back to ``/proc/cpuinfo``'s "cpu MHz"
otherwise -- amd_pstate and intel_pstate both expose scaling_cur_freq, but a
container or a VM may not.

**READING THAT FILE IS NOT FREE, AND ON SOME DRIVERS IT IS EXPENSIVE.** Under
``acpi-cpufreq`` a read of ``scaling_cur_freq`` issues an IPI to the
target CPU and reads its APERF/MPERF counters there. Polling every online CPU
interrupts every core, including the one being measured. This was measured to cost
at **2.7x on a pinned FMA benchmark** with 48 CPUs at 0.5 s, which is why
``--cpus`` exists and why callers should pass the set they actually pinned to.
This is the CPU analogue of "profilers seize the clock", and it is worse
in one respect: a profiler changes the clock, this steals the core.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

_SYS = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq"


def _online_cpus() -> list[int]:
    out = []
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
        m = re.fullmatch(r"cpu(\d+)", os.path.basename(path))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _read_sysfs(cpus):
    got = {}
    for c in cpus:
        try:
            with open(_SYS.format(c)) as fh:
                got[c] = float(fh.read().strip())        # kHz
        except OSError:
            pass
    return got


def _read_proc_cpuinfo():
    """Fallback: /proc/cpuinfo's "cpu MHz", in the order the file lists them."""
    got, idx = {}, 0
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("processor"):
                    idx = int(line.split(":")[1])
                elif line.startswith("cpu MHz"):
                    got[idx] = float(line.split(":")[1]) * 1000.0
    except OSError:
        pass
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--cpus", type=str, default="",
                    help="comma-separated CPUs to sample; empty = all online")
    a = ap.parse_args()

    cpus = _online_cpus()
    if a.cpus:
        want = {int(c) for c in a.cpus.split(",") if c != ""}
        cpus = [c for c in cpus if c in want]
    t0 = time.monotonic()
    print("t_rel_s,cpu,khz,loadavg_1m", flush=True)
    while True:
        khz = _read_sysfs(cpus) or _read_proc_cpuinfo()
        try:
            load = os.getloadavg()[0]
        except OSError:
            load = float("nan")
        t = time.monotonic() - t0
        for c in cpus:
            v = khz.get(c)
            if v is not None:
                print(f"{t:.3f},{c},{v:.0f},{load:.2f}", flush=True)
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)

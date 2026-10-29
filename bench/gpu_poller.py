#!/usr/bin/env python3
"""Standalone subprocess poller for :class:`bench.clocks.ClockPowerSampler`.

Why a subprocess and not a background thread in the sampling process's own
interpreter: the process being sampled may hold the GIL for the *entire*
duration of a timed kernel launch, inside a C++/CUDA extension that is not
required to release it around a synchronous CUDA call. This project's own
prior experience is that a
background *thread* sampler got exactly **one sample over a two-minute
benchmark** for precisely this reason -- the interpreter running the thread
never got scheduled, because nothing releases the GIL for it to run in. A
separate OS process has its own interpreter and its own GIL, so its
scheduling does not depend on what the parent's extension is doing.

Deliberately dependency-free (stdlib only) and standalone -- it does not
import ``roshambo3`` or ``bench`` -- so it runs unmodified on a bare-Python
host with no package installed at all (e.g. a machine mid-setup, or a CI image
that never installs this project).

    python3 gpu_poller.py --index 1 --fields clocks.sm,power.draw --interval 1.0

Writes one CSV line per sample to stdout, flushed immediately after each
sample (``nvidia-smi -l`` block-buffers when stdout is not a tty, so this
reimplements the poll loop itself rather than using it). Each
line is prefixed with this process's own ``time.monotonic()`` reading,
comparable against the parent's: ``CLOCK_MONOTONIC`` is a machine-wide clock,
not a per-process one, so two different processes on the same host can
compare readings from it directly. Runs until killed (the parent's
``ClockPowerSampler.__exit__`` sends SIGTERM, then SIGKILL if that does not
land in time).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True,
                     help="nvidia-smi physical device index (NOT a CUDA "
                          "visible-device ordinal -- see bench.gpu_lock."
                          "resolve_device)")
    ap.add_argument("--fields", required=True,
                     help="comma-separated --query-gpu field list")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args(argv)

    while True:
        t = time.monotonic()
        r = subprocess.run(
            ["nvidia-smi", "-i", args.index, f"--query-gpu={args.fields}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        out = r.stdout.strip()
        line = out.splitlines()[0] if out else ""
        if line:
            sys.stdout.write(f"{t!r},{line}\n")
            sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Repeat/median/dispersion mechanics and the start-pairs/s invariant unit.

``start_mode`` changes how many pose starts each (query, library) pair costs:
0 -> 1 start, 1 -> 4 starts, 2 -> 10 starts. Reporting throughput in
conformer-pairs/s alone conflates these -- start_mode=0 looks 4x faster than
start_mode=1 while doing 4x less work. Every throughput number this harness
emits is denominated in start-pairs/s, computed via :func:`n_starts`.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

N_STARTS_BY_MODE = {0: 1, 1: 4, 2: 10}


def n_starts(start_mode: int) -> int:
    try:
        return N_STARTS_BY_MODE[start_mode]
    except KeyError:
        raise ValueError(f"unknown start_mode {start_mode!r}; expected 0, 1, or 2")


@dataclass
class RepeatedTiming:
    warmup_s: list[float]
    repeat_s: list[float]

    @property
    def median_s(self) -> float:
        return statistics.median(self.repeat_s)

    @property
    def dispersion(self) -> float:
        """(max - min) / median, the dispersion metric the ticket requires
        reported alongside the median. Measured historically in the 0.1%-1.4%
        range with unlocked clocks on 0.03-0.8s kernels."""
        med = self.median_s
        if med == 0:
            return float("nan")
        return (max(self.repeat_s) - min(self.repeat_s)) / med

    @property
    def total_repeat_s(self) -> float:
        """Sum of all timed-repeat durations -- the "sustained kernel time"
        the burst/non-burst rule checks against, not any single repeat."""
        return sum(self.repeat_s)

    def as_dict(self) -> dict:
        return {
            "warmup_s": self.warmup_s,
            "repeat_s": self.repeat_s,
            "median_s": self.median_s,
            "dispersion": self.dispersion,
            "total_repeat_s": self.total_repeat_s,
        }


def timed_repeats(
    fn: Callable[[], None],
    n_warmup: int,
    n_repeats: int,
) -> RepeatedTiming:
    """Run ``fn`` ``n_warmup`` times (untimed, to burn through PTX JIT and
    CUDA-context creation) then ``n_repeats`` times, timing each with
    ``time.perf_counter``. ``fn`` is responsible for its own
    synchronization (e.g. calling into a function that internally
    cudaDeviceSynchronize()s) so the measured interval is a valid wall-clock
    number."""
    warmup_s = []
    for _ in range(n_warmup):
        t0 = time.perf_counter()
        fn()
        warmup_s.append(time.perf_counter() - t0)

    repeat_s = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        repeat_s.append(time.perf_counter() - t0)

    return RepeatedTiming(warmup_s=warmup_s, repeat_s=repeat_s)


def start_pairs_per_s(n_pairs: int, start_mode: int, seconds: float) -> float:
    """The invariant throughput unit: (conformer pairs) x (starts/pair) / time."""
    if seconds <= 0:
        return float("nan")
    return n_pairs * n_starts(start_mode) / seconds

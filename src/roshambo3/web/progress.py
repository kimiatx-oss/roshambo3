"""Constant-space, throttled estimates of remaining submitted screen work."""
from math import exp


class SubmissionProgress:
    """Use monotonic screening time, never preparation/queue or client clocks.

    Estimates concern submission, not device completion or finalization. The
    caller supplies one clock sample per block; sub-interval calls do no rate
    calculation or persistence. Defaults publish at most twice a second, smooth
    over five seconds, and wait for two positive samples/two seconds of warmup.
    """

    def __init__(self, total, started, *, interval=0.5, warmup=2.0, smoothing=5.0):
        self.total = max(0, int(total))
        self.started = self.last = started
        self.interval = interval
        self.warmup = warmup
        self.smoothing = smoothing
        self.submitted = 0
        self.rate = None
        self.samples = 0

    def update(self, submitted, now):
        dt = now - self.last
        if dt < self.interval:
            return None
        submitted = min(self.total, max(self.submitted, int(submitted)))
        instant = (submitted - self.submitted) / dt
        if instant > 0:
            self.samples += 1
        self.rate = instant if self.rate is None else self.rate + (
            1 - exp(-dt / self.smoothing)) * (instant - self.rate)
        self.last, self.submitted = now, submitted
        elapsed = now - self.started
        remaining = self.total - submitted
        eta = remaining / self.rate if (
            elapsed >= self.warmup and self.samples >= 2
            and self.rate > 0 and remaining > 0) else None
        return {"submitted_conformers": submitted,
                "progress": submitted / self.total if self.total else 0.0,
                "screen_elapsed_s": elapsed, "submission_rate": self.rate,
                "eta_seconds": eta}

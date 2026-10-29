"""The benchmark harness: enforced methodology, not convenience.

Dev-only -- not part of the installed package. Its job is to make a
non-comparable measurement fail rather than be recorded, because the failure
mode being defended against is a plausible number that nobody can reproduce.

What is here is the methodology core:

  rules.py        the enforced rules; refuses a run or tags it `advisory`
  clocks.py       clock and power sampling, and profiler-seizure detection
  gpu_lock.py     the flock protocol on a shared card
  run_record.py   the run-record schema
  build_check.py  the loaded .so is the build you think it is
  demo_refuse.py  deliberately non-compliant runs, each shown being refused

ROSHAMBO2's benchmark *drivers* are deliberately not ported. They were written
against an API that no longer exists, and the engine now reports its own
six-stage breakdown through `ScreenStats` and `stage_timings()` -- so a driver
here should be written against that rather than translated from one that
predates it.
"""

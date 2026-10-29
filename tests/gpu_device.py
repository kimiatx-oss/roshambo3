"""Whether a CUDA device could be reached, decided without touching CUDA.

`backend.available()` answers whether the extension imports, which a CUDA build
does on a machine with no GPU at all -- the CI `build with CUDA` job. Tests that
launch kernels need a device too. This check makes no CUDA call: creating a
context is what blocks under a foreign MPS daemon (CLAUDE.md), and a skip
predicate must never hang collection.
"""
import os
import sys


def device_present() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False  # explicitly hidden
    if sys.platform.startswith("linux"):
        return os.path.exists("/dev/nvidiactl")  # created by the loaded driver
    return True  # let the test say so on other platforms

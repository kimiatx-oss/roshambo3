"""Where the machine-specific things live.

Nothing in this package should hardcode a path that is true only on one host.
Two of them are genuinely deployment-specific and both are read from the
environment, with defaults chosen to be obviously-a-default rather than
obviously-somebody's-laptop:

``ROSHAMBO3_CORPORA``
    Directory holding prepared corpora. Bulk data belongs on a fast, large
    volume and should exist exactly once -- a corpus is tens of gigabytes and
    copying one into a working tree is the mistake this variable exists to make
    unnecessary.

``ROSHAMBO3_BUILD_REGISTRY``
    Directory holding already-built extensions, keyed by the source they were
    compiled from and the flags they were compiled with. Consulted by
    the build harness before it configures, so an A/B arm that
    rebuilds *identical* source in a different worktree is a lookup rather than
    a compile. Shared by every worktree on a host, which is the point: 34 build
    directories existed across this machine's worktrees and nothing reused any
    of them.

``ROSHAMBO3_CPU_LOCK``
    Base path for the lock file serializing timed runs on a shared CPU.
    Host-scoped, and nothing finer: two runs pinned to disjoint cores still
    contend for last-level cache, memory bandwidth and the boost budget, so a
    per-core-set lock would arbitrate the wrong thing while reading as safety.
    See :func:`cpu_lock_path` and ``bench/cpu.py``.

``ROSHAMBO3_GPU_LOCK``
    Base path for the lock file(s) serializing timed runs on a shared GPU. Any
    measurement whose number will be quoted has to hold one; correctness work
    does not. See ``bench/gpu_lock.py``.

    A single literal path with no host or device component would be wrong
    on any machine with more than one GPU or shared between hosts over a
    common filesystem. :func:`gpu_lock_path` now
    derives one lock file per **(host, GPU UUID)** pair from this base, so two
    machines never arbitrate each other's card by accident, and two
    cards in one host run as two independent lanes rather than serializing on one
    file -- while a second claimant on the *same* card still excludes cleanly.
    The UUID, not the ``nvidia-smi`` index, is what identifies the device:
    ``CUDA_VISIBLE_DEVICES`` renumbers indices, so an index-keyed lock file can
    silently name the wrong card.
"""

from __future__ import annotations

import os
import pathlib
import socket

#: Default corpus root. Deliberately generic: a real deployment sets
#: ROSHAMBO3_CORPORA, and a default that looked like a real path would get
#: copied into scripts and quietly become one.
DEFAULT_CORPORA = "/var/lib/roshambo3/corpora"

#: Default build-registry root. Generic for the same reason as the corpus root:
#: a default that looked like a real path would be copied into scripts. Under
#: /var/lib rather than /tmp because a compiled extension is expensive enough
#: that losing the cache on reboot is a real cost, and because it must be shared
#: between worktrees rather than between processes.
DEFAULT_BUILD_REGISTRY = "/var/lib/roshambo3/build-registry"

#: Default GPU lock base path. Under /tmp so an unconfigured machine still gets
#: correct mutual exclusion between processes, just not across reboots. Never
#: used directly as a lock file path any more -- see :func:`gpu_lock_path`,
#: which appends the host name and device UUID.
DEFAULT_GPU_LOCK = "/tmp/roshambo3.gpu.lock"

DEFAULT_CPU_LOCK = "/tmp/roshambo3.cpu.lock"
"""The CPU is a shared resource and a timed CPU number
needs the same arbitration a timed GPU number gets -- more, if anything, since
a GPU screen running under a held GPU lock is itself heavy CPU load and would
otherwise be invisible to a CPU run's contention check."""


def corpora_root() -> pathlib.Path:
    """Directory holding prepared corpora, from ``ROSHAMBO3_CORPORA``."""
    return pathlib.Path(os.environ.get("ROSHAMBO3_CORPORA", DEFAULT_CORPORA))


def corpus_path(name: str) -> pathlib.Path:
    """Path to a named corpus under :func:`corpora_root`."""
    return corpora_root() / name


def build_registry_root() -> pathlib.Path:
    """Directory of reusable built extensions, from ``ROSHAMBO3_BUILD_REGISTRY``.

    Read here and nowhere else, per this module's own docstring. The build
    driver loads this file directly by path rather than importing the package,
    because it has to work in a tree where the package is not yet installed --
    which is precisely when it is about to build one.
    """
    return pathlib.Path(os.environ.get("ROSHAMBO3_BUILD_REGISTRY",
                                       DEFAULT_BUILD_REGISTRY))


def cpu_lock_path() -> pathlib.Path:
    """Per-host lock file for timed CPU runs.

    Host-scoped and nothing finer, which is the difference from
    :func:`gpu_lock_path`: a machine has one CPU in the sense that matters here.
    Two runs pinned to disjoint cores still contend for last-level cache,
    memory bandwidth and the boost budget -- L3 instances are shared
    across several cores, and every core's achievable clock depends on what the
    others are doing -- so a per-core-set lock would arbitrate the wrong thing
    and read as safety.

    The base comes from ``ROSHAMBO3_CPU_LOCK``, read here and nowhere else, and
    is a *base* rather than a literal path for the same reason
    ``ROSHAMBO3_GPU_LOCK`` is one: two machines sharing a filesystem
    must not arbitrate each other.
    """
    base = os.environ.get("ROSHAMBO3_CPU_LOCK", DEFAULT_CPU_LOCK)
    return pathlib.Path(f"{base}.{socket.gethostname()}")


def gpu_lock_path(device_uuid: str) -> pathlib.Path:
    """Per-host, per-device lock file for timed GPU runs.

    ``device_uuid`` is required, not defaulted: a lock file that does not name
    the device it arbitrates is exactly the defect of a single
    ``ROSHAMBO3_GPU_LOCK`` path with no host *or* device component (two runs
    on a host's GPU 0 and GPU 1 serialized against each other for no reason,
    and a lock taken on one host said nothing about the other host's card).

    The base path comes from ``ROSHAMBO3_GPU_LOCK`` (unchanged env var name;
    it is now a *base* for a family of lock files rather than one literal
    path) -- read here and nowhere else, per this module's own docstring.
    The host name is appended so two machines sharing a filesystem never
    arbitrate each other's card by accident. The UUID -- never the
    ``nvidia-smi`` index -- is appended last: ``CUDA_VISIBLE_DEVICES``
    renumbers indices, so an index-keyed lock file can silently name the
    wrong card once a caller's environment restricts visibility.
    """
    base = os.environ.get("ROSHAMBO3_GPU_LOCK", DEFAULT_GPU_LOCK)
    host = socket.gethostname()
    # GPU UUIDs from nvidia-smi are already filesystem-safe ("GPU-<hex>"),
    # but sanitize defensively rather than assume that never changes.
    safe_uuid = "".join(c if (c.isalnum() or c in "-_") else "_" for c in device_uuid)
    return pathlib.Path(f"{base}.{host}.{safe_uuid}")

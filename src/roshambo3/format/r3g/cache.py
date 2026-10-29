"""The ``scan`` cache: a corpus is scanned once per content, not per screen.

A passing ``scan`` (or ``deep``) is recorded under a key that is the SHA-256
of:

* :data:`validate.VALIDATOR_VERSION` (a validator fix invalidates every entry);
* the SHA-256 of the ``corpus.json`` bytes;
* the SHA-256 of every group header, as read (``open`` checks each against
  ``corpus.json``'s ``header_sha256``);
* ``(name, size, mtime_ns)`` of every group file and of ``molecules.parquet``.

The first three are the reader's own keys; the stat triple is added because no
hash in ``corpus.json`` or a group header covers the bytes of the run, feature
and self-overlap sections, so a section edited after the scan would otherwise
still hit. Changing one byte of ``corpus.json`` or a group header changes the
key; so does rewriting any file.

Entries are small JSON files under :func:`cache_dir`: an argument, else
``$ROSHAMBO3_SCAN_CACHE``, else ``$XDG_CACHE_HOME/roshambo3/scan``, else
``~/.cache/roshambo3/scan``. The cache is the user's own state, trusted like
the rest of their home directory; a cache that cannot be written is skipped
(the corpus is then scanned again next time), never an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Iterable, Optional

ENV = "ROSHAMBO3_SCAN_CACHE"


def cache_dir(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get(ENV)
    if env:
        return env
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "roshambo3", "scan")


def scan_key(descriptor_sha256: str, header_sha256: Iterable[str],
             file_stats: Iterable[tuple]) -> str:
    from .validate import VALIDATOR_VERSION
    h = hashlib.sha256()
    h.update(f"r3g-scan-cache\0{VALIDATOR_VERSION}\0{descriptor_sha256}\0".encode())
    for x in header_sha256:
        h.update(f"h:{x}\0".encode())
    for name, size, mtime_ns in sorted(file_stats):
        h.update(f"f:{name}:{int(size)}:{int(mtime_ns)}\0".encode())
    return h.hexdigest()


def lookup(key: str, directory: Optional[str] = None) -> Optional[dict]:
    """The recorded entry for ``key``, or ``None``."""
    path = os.path.join(cache_dir(directory), key + ".json")
    try:
        with open(path, "rb") as fh:
            data = fh.read(1 << 16)
        entry = json.loads(data.decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("key") != key:
        return None
    return entry


def record(key: str, *, tier: str, fingerprint: str, root: str,
           directory: Optional[str] = None) -> bool:
    """Record a passing ``scan``/``deep``; ``False`` when the cache cannot be
    written (the corpus is simply scanned again next time)."""
    d = cache_dir(directory)
    entry = {"key": key, "tier": tier, "fingerprint": fingerprint,
             "root": os.path.abspath(root)}
    try:
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, sort_keys=True)
        os.replace(tmp, os.path.join(d, key + ".json"))
    except OSError:
        return False
    return True

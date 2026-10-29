"""Opening and reading corpus files (``doc/FORMAT.md`` section 1 item 2).

Two policies live here and nowhere else, so a spec change touches one file:

* **How a file is opened.** :func:`open_corpus_file` walks from the corpus
  root with directory file descriptors: every component below the root is
  opened relative to its parent's descriptor with ``O_NOFOLLOW`` (so no
  symlink anywhere below the root is followed, and nothing can resolve
  outside it), the last with ``O_NONBLOCK`` (so a FIFO cannot hang the
  ``open()``), then ``fstat`` + ``S_ISREG``, then ``O_NONBLOCK`` cleared.
  Every later read uses that one descriptor; nothing reopens by path. The
  component names are always derived by the reader (``corpus.json``,
  ``groups/b00_g0000.r3g``), never read from a file.
* **How section bytes are accessed.** :meth:`CorpusFile.read` (``pread``
  into memory): a file truncated under it gives a short read and the refusal
  ``io.short_read``, never a ``SIGBUS`` (section 1 item 4). The validator and,
  by default, the reader use it; :data:`READER_ACCESS` is the one switch.
"""

from __future__ import annotations

import errno
import mmap
import os
import stat
from typing import Optional

from .errors import CorpusValidationError

#: The reader's section access. Section 1 item 4 (amendment S7): a reader MUST
#: NOT ``mmap`` a corpus file unless it converts the ``SIGBUS`` of a file
#: truncated after mapping into an error, which Python cannot do; so the
#: reader copies with ``pread`` into read-only arrays. ``"mmap"`` remains for
#: callers that guarantee the files are not truncated (it is not the default).
READER_ACCESS = "pread"

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def _refuse(code: str, message: str, path: str, tier: str) -> CorpusValidationError:
    return CorpusValidationError(code, message, tier=tier, path=path)


def _os_refusal(e: OSError, path: str, tier: str, what: str) -> CorpusValidationError:
    if e.errno == errno.ENOENT:
        return _refuse("io.missing_file", f"{what} is missing", path, tier)
    if e.errno in (errno.ELOOP, errno.EMLINK):
        return _refuse("io.not_regular_file", f"{what} is a symlink", path, tier)
    return _refuse("io.not_regular_file", f"{what} cannot be opened as a regular file "
                   f"({errno.errorcode.get(e.errno, e.errno)})", path, tier)


class CorpusFile:
    """One open regular file of a corpus: a descriptor, its size at open, and
    bounded reads. Close it (or use it as a context manager)."""

    def __init__(self, fd: int, path: str, size: int, mtime_ns: int = 0):
        self.fd = fd
        self.path = path
        self.size = size
        self.mtime_ns = mtime_ns

    @property
    def stat_key(self) -> tuple[int, int]:
        """``(size, mtime_ns)`` at open: what the scan cache binds to."""
        return (self.size, self.mtime_ns)

    def read(self, offset: int, length: int, *, tier: str = "open") -> bytes:
        """Exactly ``length`` bytes at ``offset`` (``pread``), or a refusal if
        the file is shorter than its size at open said."""
        if offset < 0 or length < 0 or offset + length > self.size:
            raise _refuse("internal.error", f"read [{offset}, {offset + length}) outside the "
                          f"file's {self.size} bytes", self.path, tier)
        parts = []
        got = 0
        while got < length:
            chunk = os.pread(self.fd, min(length - got, 1 << 30), offset + got)
            if not chunk:
                raise _refuse("io.short_read", f"read {got} of {length} bytes at offset "
                              f"{offset}: the file is shorter than when it was opened",
                              self.path, tier)
            parts.append(chunk)
            got += len(chunk)
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def map(self) -> Optional[mmap.mmap]:
        """A read-only map of the whole file (``None`` for an empty file)."""
        if self.size == 0:
            return None
        return mmap.mmap(self.fd, self.size, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "CorpusFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_corpus_file(root: str, *components: str, tier: str = "open") -> CorpusFile:
    """Open ``root/<components...>`` as a regular file inside the corpus.

    ``root`` itself may be a symlink (corpora are shared that way); every
    component below it is opened with ``O_NOFOLLOW`` relative to its parent's
    descriptor. Refusals: ``io.missing_file`` (absent), ``io.not_regular_file``
    (a symlink, FIFO, socket, device or directory)."""
    path = os.path.join(root, *components)
    what = "/".join(components)
    for c in components:
        if not c or c in (".", "..") or "/" in c or "\0" in c:
            raise _refuse("io.not_regular_file", f"{what!r} is not a plain name inside the "
                          f"corpus directory", path, tier)
    try:
        dir_fd = os.open(root, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC)
    except OSError as e:
        raise _os_refusal(e, path, tier, "the corpus directory") from None
    try:
        for c in components[:-1]:
            try:
                nxt = os.open(c, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                              dir_fd=dir_fd)
            except OSError as e:
                raise _os_refusal(e, path, tier, f"directory {c!r}") from None
            os.close(dir_fd)
            dir_fd = nxt
        try:
            fd = os.open(components[-1], os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC,
                         dir_fd=dir_fd)
        except OSError as e:
            raise _os_refusal(e, path, tier, what) from None
    finally:
        os.close(dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _refuse("io.not_regular_file", f"{what} is not a regular file", path, tier)
        if _O_NONBLOCK:
            import fcntl
            fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~_O_NONBLOCK)
    except BaseException:
        os.close(fd)
        raise
    return CorpusFile(fd, path, st.st_size, st.st_mtime_ns)


def exists_in_corpus(root: str, *components: str) -> bool:
    """Whether a directory entry exists (``lstat``, following nothing)."""
    return os.path.lexists(os.path.join(root, *components))

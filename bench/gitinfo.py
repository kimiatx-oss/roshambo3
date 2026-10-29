"""Git repository state, for the run-record ``git`` block.

A benchmark run's provenance is incomplete without knowing exactly what
source produced it. ``dirty=True`` alone does not invalidate a run, but
combined with the other auto-tag rules in :mod:`bench.rules` it is
one of the signals that downgrades a run to ``advisory``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from dataclasses import dataclass


def _git(*args: str, cwd: str | None = None) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


#: Tracked paths that a run WRITES rather than reads. Dirtiness here says
#: nothing about whether the run is reproducible -- it is the run's own output.
#:
#: The concrete failure this exists to stop: ``run.write()`` appends the graded
#: record to a ``runs.jsonl`` under ``results/``, which is tracked, so **recording a
#: graded run dirtied the tree and downgraded the next one**. It once cost a full
#: run graded ``advisory`` on nothing but its predecessor's ledger entry.
#:
#: Deliberately narrow. Only ``results/`` is listed, because only ``results/``
#: is unambiguously output. Documentation and prose are left IN the
#: source set: they do not affect a number, but excluding them is a judgement
#: about what counts as the run rather than a fact about it, and the check
#: should not quietly grow into "dirtiness we would rather ignore".
OUTPUT_PATH_PREFIXES = ("results/",)


def _split_status(status: str):
    """``(source_dirty_paths, output_dirty_paths)`` from ``--porcelain``."""
    src, out = [], []
    for line in status.splitlines():
        if not line.strip():
            continue
        # NOT a fixed offset. Porcelain is "XY path", but `_git` strips its
        # output, which eats the leading space of the FIRST line and shifts
        # every offset by one -- it silently produced "ench/gitinfo.py" for
        # "bench/gitinfo.py", which would misclassify any path whose prefix
        # match depended on its first character.
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        path = parts[1]
        # Renames read "old -> new"; the destination is what was touched.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        (out if path.startswith(OUTPUT_PATH_PREFIXES) else src).append(path)
    return src, out


@dataclass(frozen=True)
class GitInfo:
    sha: str
    dirty: bool
    branch: str
    source_dirty: bool = True
    """Whether anything OUTSIDE :data:`OUTPUT_PATH_PREFIXES` is modified.

    This is the one that bears on reproducibility, and the one the harness
    grades on. :attr:`dirty` stays whole-tree and stays recorded, so a reader
    can still see the full state and nothing is hidden by the narrowing."""

    source_dirty_paths: tuple = ()
    output_dirty_paths: tuple = ()

    def as_dict(self) -> dict:
        return {"sha": self.sha, "dirty": self.dirty, "branch": self.branch,
                "source_dirty": self.source_dirty,
                # Capped: a reader needs to see WHICH paths, not all of them.
                "source_dirty_paths": list(self.source_dirty_paths[:20]),
                "output_dirty_paths": list(self.output_dirty_paths[:20])}


def get_git_info(cwd: str | None = None) -> GitInfo:
    expected = Path(cwd).resolve() if cwd else Path(__file__).resolve().parents[1]
    try:
        actual = Path(_git("rev-parse", "--show-toplevel", cwd=str(expected))).resolve()
        if actual != expected:
            return GitInfo(sha="unknown", dirty=True, branch="unknown", source_dirty=True)
        cwd = str(expected)
        sha = _git("rev-parse", "HEAD", cwd=cwd)
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
        status = _git("status", "--porcelain", cwd=cwd)
        dirty = bool(status.strip())
        src, out = _split_status(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fail closed: if git cannot be read, assume the worst on both.
        return GitInfo(sha="unknown", dirty=True, branch="unknown",
                       source_dirty=True)
    return GitInfo(sha=sha, dirty=dirty, branch=branch,
                   source_dirty=bool(src),
                   source_dirty_paths=tuple(src), output_dirty_paths=tuple(out))

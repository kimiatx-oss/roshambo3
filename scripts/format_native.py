#!/usr/bin/env python3
"""Format the native CPU and Metal sources with clang-format.

    python scripts/format_native.py            # rewrite in place
    python scripts/format_native.py --check    # list files that would change

Covers ``src/roshambo3/backend/_cpu_src`` (x86 and ARM) and ``_metal_src``.
The CUDA sources are deliberately excluded (``backend/.clang-format-ignore``):
nothing may be added under ``_cuda_src/``, whose every file is hashed into the
build stamp by ``scripts/source_stamp.py``.
The style lives in ``src/roshambo3/backend/.clang-format``.

``.metal`` is not a suffix clang-format recognizes, so it is passed through
stdin with ``--assume-filename``; Metal Shading Language is C++ for layout
purposes. A reformat must change whitespace alone -- prove it with
``scripts/verify_layout_only.py`` against a copy of the previous tree.

Needs ``clang-format`` on PATH or in ``$CLANG_FORMAT`` (``pip install
clang-format`` provides one). Results depend on the clang-format version; the
committed layout was produced with 23.1.3.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKEND = ROOT / "src" / "roshambo3" / "backend"
DIRS = (BACKEND / "_cpu_src", BACKEND / "_metal_src")
SUFFIXES = {".cpp", ".hpp", ".h", ".mm", ".metal"}


def clang_format() -> str:
    exe = os.environ.get("CLANG_FORMAT") or shutil.which("clang-format")
    if not exe:
        sys.exit("clang-format not found: install it or set $CLANG_FORMAT")
    return exe


def formatted(exe: str, path: pathlib.Path) -> bytes:
    # Metal is laid out as C++; the assumed name only selects the language and
    # the .clang-format lookup directory.
    assume = path.with_suffix(".cpp") if path.suffix == ".metal" else path
    return subprocess.run(
        [exe, "--style=file", f"--assume-filename={assume}"],
        input=path.read_bytes(), stdout=subprocess.PIPE, check=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report files that are not formatted; change nothing")
    args = ap.parse_args()
    exe = clang_format()
    files = sorted(p for d in DIRS for p in d.rglob("*")
                   if p.suffix in SUFFIXES and p.is_file())
    stale = []
    for path in files:
        new = formatted(exe, path)
        if new != path.read_bytes():
            stale.append(path)
            if not args.check:
                path.write_bytes(new)
    verb = "need formatting" if args.check else "reformatted"
    for path in stale:
        print(path.relative_to(ROOT))
    print(f"{len(stale)} of {len(files)} files {verb}", file=sys.stderr)
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prove that a change to the native sources moved layout or comments only.

    python scripts/verify_layout_only.py BEFORE AFTER            # layout only
    python scripts/verify_layout_only.py BEFORE AFTER --comments # comments only

BEFORE and AFTER are git revisions; AFTER may be ``WORKTREE`` for the files on
disk. Every C/C++/Objective-C++/Metal/CUDA file under ``src/roshambo3/backend``
that exists in both is checked.

Layout mode (a reformat) requires two independent things of every file:

1. The bytes are identical once all whitespace, and every backslash-newline
   line splice, is deleted. Splices are removed in translation phase 2, before
   tokenization, so moving one inside a macro is layout.
2. libclang's raw lexer yields the same token sequence. Comment tokens are
   compared with their internal whitespace deleted, so a re-indented block
   comment is still the same comment. This catches what (1) cannot: whitespace
   inside a string literal, or two tokens fused into one.

Comments mode (an annotation pass) requires the token sequence *excluding*
comments to be identical: a comment may be added, edited or removed, and
nothing else may change.

Neither check compiles anything, so it holds for the NEON and Metal sources
that cannot be built on an x86 host. Needs libclang (``pip install libclang``).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

try:
    import clang.cindex as ci
except ImportError:
    sys.exit("needs libclang: pip install libclang")

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKEND = "src/roshambo3/backend"
SUFFIXES = {".cpp", ".hpp", ".h", ".mm", ".metal", ".cu", ".cuh"}
LAYOUT = re.compile(rb"\\\r?\n|\s+")


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          stdout=subprocess.PIPE).stdout


def files_at(rev: str) -> dict[str, bytes]:
    if rev == "WORKTREE":
        base = ROOT / BACKEND
        return {str(p.relative_to(ROOT)): p.read_bytes()
                for p in base.rglob("*") if p.suffix in SUFFIXES}
    names = git("ls-tree", "-r", "--name-only", rev, "--", BACKEND).decode()
    return {n: git("show", f"{rev}:{n}") for n in names.split()
            if pathlib.PurePath(n).suffix in SUFFIXES}


def tokens(index, path: pathlib.Path, keep_comments: bool) -> list:
    lang = "objective-c++" if path.suffix == ".mm" else "c++"
    # -nostdinc and a skipped parse: only the raw lexer's view of this one file
    # matters, and it does not depend on headers resolving.
    tu = index.parse(str(path), args=["-x", lang, "-std=c++20", "-nostdinc"],
                     options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                     | ci.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
                     | ci.TranslationUnit.PARSE_INCOMPLETE)
    out = []
    for t in tu.cursor.get_tokens():
        if t.location.file is None or pathlib.Path(t.location.file.name) != path:
            continue
        if t.kind == ci.TokenKind.COMMENT:
            if keep_comments:
                out.append(("comment", re.sub(r"\s+", "", t.spelling)))
        else:
            out.append((t.kind.name.lower(), t.spelling))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--comments", action="store_true",
                    help="allow comment changes; require identical code tokens")
    args = ap.parse_args()
    before, after = files_at(args.before), files_at(args.after)
    common = sorted(set(before) & set(after))
    index = ci.Index.create()
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for name in common:
            b, a = before[name], after[name]
            if b == a:
                continue
            if not args.comments and LAYOUT.sub(b"", b) != LAYOUT.sub(b"", a):
                print(f"non-whitespace change: {name}")
                failures += 1
                continue
            # Same file name in two directories: the suffix picks the language.
            pb, pa = tmp / "b" / name, tmp / "a" / name
            for p, data in ((pb, b), (pa, a)):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            tb = tokens(index, pb, not args.comments)
            ta = tokens(index, pa, not args.comments)
            if not tb:
                print(f"no tokens lexed (cannot verify): {name}")
                failures += 1
            elif tb != ta:
                i = next((i for i, (x, y) in enumerate(zip(tb, ta)) if x != y),
                         min(len(tb), len(ta)))
                print(f"token stream differs: {name} at token {i}: "
                      f"{tb[i:i + 3]} -> {ta[i:i + 3]}")
                failures += 1
    changed = sum(before[n] != after[n] for n in common)
    mode = "comments-only" if args.comments else "layout-only"
    print(f"{changed} changed files of {len(common)} checked; "
          f"{failures} not {mode}", file=sys.stderr)
    only = sorted(set(before) ^ set(after))
    if only:
        print(f"not compared (present on one side only): {only}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

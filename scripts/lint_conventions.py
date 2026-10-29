#!/usr/bin/env python3
"""Enforce the two naming conventions in src/.

1. No ticket ids in identifiers. `T36GridGeom`, `t44_topk_create`, `t27_stage_w`
   name history that will be absent post-release, and at least one of them --
   ROSHAMBO2's `grid_tier`, a coordinate-residency knob -- read as
   if it named a level of the Tier-0..3 cascade, an unrelated concept. The name
   says what it does; a comment says where it came from. Comments and docstrings
   are therefore exempt, and are where provenance belongs.

2. American English. `color`, `optimizer`, `quantize`, `normalize`.

Both are scoped to the code this project authors -- src/, scripts/, bench/ and
tests/ -- plus the **active** prose: the top-level guides and `doc/`. The
spelling rule applies to docs because CLAUDE.md says it does ("identifiers,
comments and docs"), and until now nothing checked them: `doc/NUMERICS_POLICY.md`
alone carried 21 British spellings.

tickets/, decisions/ and results/ are the verbatim historical archive: they
quote measured numbers, and rewriting them would make the archive disagree with
the run records it describes.

Ticket ids are only checked in *identifiers*, so prose is scanned for spelling
alone -- a document naming T44 is doing exactly what the rule asks for.

Where the ticket-id rule applies depends on the repository. In a repository
that carries the ticket archive (`tickets/`, i.e. this internal one), a name
like `bench/t210_existing_evidence.py` still resolves to its
ticket, so the rule covers the shipped package, src/, only; research harnesses
and tests in scripts/, bench/ and tests/ may keep ticket names. In a release
candidate there is no archive to resolve them against, so the rule covers all
four directories. `--strict` applies the release scope anywhere, e.g. to
preview what an export will be held to. Spelling is checked everywhere in both.

    python scripts/lint_conventions.py [--strict]
"""
from __future__ import annotations

import pathlib
import re
import sys
import tokenize

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src", "scripts", "bench", "tests")
#: The ticket-id rule's scope in a repository that keeps the ticket archive;
#: see the module docstring. Release candidates use all of SCAN_DIRS.
INTERNAL_TICKET_ID_DIRS = ("src",)
SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".metal", ".mm"}

#: Active prose: spelling only, no identifier rule.
PROSE_DIRS = ("doc",)
PROSE_FILES = ("README.md", "CLAUDE.md", "INSTALL.md", "USER_GUIDE.md",
               "DEV_GUIDE.md", "NOTICE")
#: doc/PLAN-ORIGINAL.md is the plan as written, archived for the same reason
#: tickets/ is -- it is the authority of record, not a document to edit.
PROSE_EXEMPT = {"doc/PLAN-ORIGINAL.md"}

TICKET_ID = re.compile(r"\b[tT]\d{2}[A-Za-z0-9_]*|\b[A-Za-z_][A-Za-z0-9_]*_[tT]\d{2}\b"
                       r"|\b[A-Za-z_][A-Za-z0-9_]*_[tT]\d{2,3}_[A-Za-z0-9_]+")

BRITISH = {
    "colour": "color", "colours": "colors", "coloured": "colored",
    "optimise": "optimize", "optimised": "optimized", "optimiser": "optimizer",
    "optimisation": "optimization",
    "quantise": "quantize", "quantised": "quantized", "quantisation": "quantization",
    "normalise": "normalize", "normalised": "normalized", "normalisation": "normalization",
    "initialise": "initialize", "initialised": "initialized",
    "serialise": "serialize", "summarise": "summarize", "analyse": "analyze",
    "behaviour": "behavior", "centred": "centered", "standardise": "standardize",
    "parameterise": "parameterize", "characterise": "characterize",
}
BRITISH_RE = re.compile(r"\b(" + "|".join(sorted(BRITISH)) + r")\b", re.I)

#: A line carrying this marker is exempt from both checks. It exists for the one
#: legitimate case: naming a ROSHAMBO2 identifier or spelling on purpose, e.g. in
#: a comment recording where code came from.
ALLOW = "lint-conventions: allow"

#: This file necessarily contains every word it forbids.
SELF = pathlib.Path(__file__).resolve()


def code_only_python(path: pathlib.Path) -> list[tuple[int, str]]:
    """Lines of a .py file with comments and string literals blanked out.

    Only the token's own character span is blanked, never the whole line -- a
    trailing comment must not erase the code in front of it.
    """
    blanked = path.read_text().splitlines()
    # PEP 701 (Python 3.12+) tokenizes an f-string as FSTRING_START /
    # FSTRING_MIDDLE / FSTRING_END rather than one STRING, so the literal text
    # inside it is NOT a STRING token. Without FSTRING_MIDDLE here, the same
    # file passes on 3.11 and fails on 3.12.
    #
    # Known limitation, in the strict-on-newer direction: on 3.11 the whole
    # f-string is one STRING token, so an identifier used inside a `{...}`
    # expression is blanked along with the literal text and escapes the scan.
    # On 3.12+ it is caught. CI runs the conventions job on 3.12 so the stricter
    # reading is the one that gates. A ticket-named identifier can in any case
    # only be *used* in an f-string if it was *defined* somewhere, and the
    # definition is caught on every version.
    string_types = {tokenize.COMMENT, tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, name):
            string_types.add(getattr(tokenize, name))
    try:
        with path.open("rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type not in string_types:
                    continue
                (r0, c0), (r1, c1) = tok.start, tok.end
                for ln in range(r0, min(r1, len(blanked)) + 1):
                    line = blanked[ln - 1]
                    lo = c0 if ln == r0 else 0
                    hi = c1 if ln == r1 else len(line)
                    blanked[ln - 1] = line[:lo] + " " * (hi - lo) + line[hi:]
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass
    return list(enumerate(blanked, 1))


def code_only_c(path: pathlib.Path) -> list[tuple[int, str]]:
    """Lines of a C/C++/CUDA file with // and /* */ comments and strings blanked."""
    text = path.read_text()
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        line = re.sub(r"//.*$", "", line)
        line = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)
        out.append((i, line))
    return out


def _prose_files():
    """Active prose: the top-level guides and doc/, never the archive."""
    out = []
    for name in PROSE_FILES:
        p = ROOT / name
        if p.is_file():
            out.append(p)
    for d in PROSE_DIRS:
        base = ROOT / d
        if base.exists():
            out += [p for p in sorted(base.rglob("*.md")) if p.is_file()]
    return out


def ticket_id_dirs(strict: bool) -> tuple[str, ...]:
    """Directories whose identifiers must not carry ticket ids."""
    if strict or not (ROOT / "tickets").is_dir():
        return SCAN_DIRS
    return INTERNAL_TICKET_ID_DIRS


def main() -> int:
    strict = "--strict" in sys.argv[1:]
    unknown = [a for a in sys.argv[1:] if a != "--strict"]
    if unknown:
        print(f"usage: lint_conventions.py [--strict]; unknown {unknown}",
              file=sys.stderr)
        return 2
    ticket_dirs = ticket_id_dirs(strict)
    failures: list[str] = []
    scanned = 0

    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            scanned += 1
            rel = path.relative_to(ROOT)
            code = code_only_python(path) if path.suffix == ".py" else code_only_c(path)
            raw = path.read_text().splitlines()

            check_ticket_ids = d in ticket_dirs
            for lineno, line in code:
                if not check_ticket_ids:
                    break
                # The ALLOW marker lives in a comment, and `line` has had its
                # comments blanked -- so it has to be looked for in the RAW line.
                # Checking the blanked one silently ignores every marker, which
                # is what this did at first.
                if ALLOW in raw[lineno - 1]:
                    continue
                for m in TICKET_ID.finditer(line):
                    failures.append(
                        f"{rel}:{lineno}: ticket id in identifier {m.group(0)!r} "
                        f"-- rename it; put the provenance in a comment")

            # Spelling applies to the whole file, comments included.
            if path.resolve() == SELF:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if ALLOW in line:
                    continue
                for m in BRITISH_RE.finditer(line):
                    w = m.group(1)
                    failures.append(
                        f"{rel}:{lineno}: British spelling {w!r} "
                        f"-- use {BRITISH[w.lower()]!r}")

    for path in _prose_files():
        rel = path.relative_to(ROOT)
        if str(rel).replace("\\", "/") in PROSE_EXEMPT:
            continue
        scanned += 1
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if ALLOW in line:
                continue
            for m in BRITISH_RE.finditer(line):
                w = m.group(1)
                failures.append(
                    f"{rel}:{lineno}: British spelling {w!r} "
                    f"-- use {BRITISH[w.lower()]!r}")

    if failures:
        print(f"lint_conventions: {len(failures)} problem(s) in {scanned} file(s)\n",
              file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    scope = "everywhere" if ticket_dirs == SCAN_DIRS else "in " + ", ".join(ticket_dirs)
    print(f"lint_conventions: OK ({scanned} source files scanned; "
          f"ticket ids checked {scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

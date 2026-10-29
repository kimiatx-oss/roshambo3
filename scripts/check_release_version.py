#!/usr/bin/env python3
"""Refuse to package a release whose tag and version disagree.

A corpus built from an installed release records ``prep.commit: null`` and is
identified by ``prep.version`` alone, so a release build must carry a real,
tag-matched version. ``roshambo3.__version__`` is single-sourced from the
package metadata, which comes from ``pyproject.toml``'s ``[project] version``.

The release tag is ``v<version>`` (for example ``v1.0.0`` for version
``1.0.0``). It is taken from ``--tag``, else ``$GITHUB_REF_NAME`` when
``$GITHUB_REF_TYPE`` is ``tag``, else ``git describe --exact-match --tags HEAD``.
The check fails when no tag is found, when the tag is not ``v`` + the version,
or when the version is a development version (``.dev``). It reads no ``.git``
when ``--tag`` is given, so it also works from a source tarball.

Exit status 0 means the tree may be packaged as that release.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def project_version(root: Path = ROOT) -> str:
    return tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]


def source_version(root: Path = ROOT) -> str:
    """The fallback literal in ``roshambo3/__init__.py`` (no metadata case)."""
    text = (root / "src" / "roshambo3" / "__init__.py").read_text()
    match = re.search(r'^_SOURCE_VERSION = "([^"]+)"$', text, re.M)
    if not match:
        raise SystemExit("cannot find _SOURCE_VERSION in src/roshambo3/__init__.py")
    return match.group(1)


def discover_tag(root: Path = ROOT) -> str | None:
    if os.environ.get("GITHUB_REF_TYPE") == "tag" and os.environ.get("GITHUB_REF_NAME"):
        return os.environ["GITHUB_REF_NAME"]
    try:
        out = subprocess.run(["git", "-C", str(root), "describe", "--exact-match", "--tags", "HEAD"],
                             capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def problems(tag: str | None, version: str, fallback: str) -> list[str]:
    found = []
    if ".dev" in version:
        found.append(f"pyproject.toml version {version!r} is a development version")
    if fallback != version:
        found.append(f"src/roshambo3/__init__.py _SOURCE_VERSION {fallback!r} != "
                     f"pyproject.toml version {version!r}")
    if tag is None:
        found.append("no release tag: pass --tag, or build from a commit tagged v<version>")
    elif tag != f"v{version}":
        found.append(f"release tag {tag!r} != 'v' + pyproject.toml version {version!r}")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="release tag, e.g. v1.0.0 (default: discover it)")
    parser.add_argument("--root", type=Path, default=ROOT, help="source tree (default: this one)")
    args = parser.parse_args(argv)
    version = project_version(args.root)
    tag = args.tag if args.tag is not None else discover_tag(args.root)
    found = problems(tag, version, source_version(args.root))
    if found:
        for problem in found:
            print(f"release version check: {problem}", file=sys.stderr)
        print("refusing to build release artifacts", file=sys.stderr)
        return 1
    print(f"release version check: tag {tag} matches version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

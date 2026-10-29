"""The build stamp from a source tree with no ``.git``: a release tarball.

A released roshambo3 is built from an sdist or a ``git archive`` extract, not a
checkout. CMake runs ``scripts/source_stamp.py`` at build time, and it must
not fail the build there: the git fields degrade to ``"unknown"`` and the dirty
marker to -1 ("nobody looked", never "clean"), while ``cuda_src_sha256`` and
``cpu_src_sha256`` are still content hashes, equal to the checkout's for the
same bytes. Unlike ``test_source_stamp.py`` this needs no git, so it also runs
from a tarball.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = "scripts/source_stamp.py"
SUBTREES = {"cuda": "src/roshambo3/backend/_cuda_src",
            "cpu": "src/roshambo3/backend/_cpu_src"}


def _load_source_stamp():
    spec = importlib.util.spec_from_file_location("_source_stamp_without_git", REPO / SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_stamp = _load_source_stamp()


@pytest.fixture
def extract(tmp_path) -> pathlib.Path:
    """The stamp script and both compiled subtrees, copied with no ``.git``."""
    root = tmp_path / "roshambo3-src"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / SCRIPT, root / SCRIPT)
    for subtree in SUBTREES.values():
        shutil.copytree(REPO / subtree, root / subtree)
    return root


def _environment(tmp_path, with_git: bool) -> dict:
    env = dict(os.environ)
    # Never let git find an enclosing repository above the extract.
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    if not with_git:
        empty = tmp_path / "no-git-bin"
        empty.mkdir(exist_ok=True)
        env["PATH"] = str(empty)
    return env


def _run(extract, env, *args) -> subprocess.CompletedProcess:
    # As CMake runs it (CMakeLists.txt, roshambo3_source_stamp targets).
    return subprocess.run([sys.executable, "-I", str(extract / SCRIPT), "--repo", str(extract),
                           "--phase", "build", *args],
                          capture_output=True, text=True, env=env, timeout=120, check=False)


@pytest.mark.parametrize("with_git", [True, False], ids=["git-installed", "no-git"])
@pytest.mark.parametrize("prefix", sorted(SUBTREES))
def test_a_tree_without_git_still_stamps(extract, tmp_path, with_git, prefix):
    env = _environment(tmp_path, with_git)
    subtree = SUBTREES[prefix]

    result = _run(extract, env, "--subtree", subtree, "--prefix", prefix, "--json")
    assert result.returncode == 0, result.stderr
    s = json.loads(result.stdout)
    assert s["git_sha"] == "unknown"
    assert s[f"{prefix}_src_tree"] == "unknown"
    assert s[f"{prefix}_src_dirty"] == source_stamp.DIRTY_UNKNOWN
    assert s[f"{prefix}_src_dirty_files"] == []
    assert s[f"{prefix}_src_dirty_file_count"] == 0
    assert s["source_stamp_phase"] == "build"
    # Content-addressed: the same bytes as this tree, so the same hash,
    # with or without git.
    assert re.fullmatch(r"[0-9a-f]{64}", s[f"{prefix}_src_sha256"])
    assert s[f"{prefix}_src_sha256"] == source_stamp.source_content_hash(REPO / subtree)

    header = tmp_path / "build" / f"roshambo3_{prefix}_source_stamp.h"
    result = _run(extract, env, "--subtree", subtree, "--prefix", prefix,
                  "--header", str(header))
    assert result.returncode == 0, result.stderr
    text = header.read_text()
    up = prefix.upper()
    assert f'#define ROSHAMBO3_{up}_SRC_SHA256 "{s[f"{prefix}_src_sha256"]}"' in text
    assert f'#define ROSHAMBO3_{up}_SRC_TREE "unknown"' in text
    assert f"#define ROSHAMBO3_{up}_SRC_DIRTY -1" in text

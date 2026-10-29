"""The build stamp has to name the source that was compiled, not a commit.

`ROSHAMBO3_GIT_SHA` is `git rev-parse HEAD` at configure time. It names an
index, and the A/B arms this project measures are routinely built from a tree
that departs from its index. Restoring an older `kernels.cu` into a worktree
whose HEAD already carried the three-plane narrowing once gave a binary stamping
the commit that narrows the record while containing the code that does not --
and nothing in a run record from it could have said so.

These tests build that exact situation in a throwaway git repository and assert
the stamp reports it. They need no GPU and no build: the defect is in what the
stamp *computes*.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="git is not installed")


def _load(name: str, relative: str):
    """Load a module from THIS tree by path, not by import.

    One module under test is a build-time script, not importable as a package,
    and the other must come from the worktree rather than from whatever the shared venv's redirecting
    finder resolves `roshambo3` to. Registered in `sys.modules` because
    `dataclasses` looks the module up by name while processing annotations.
    """
    import sys

    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source_stamp = _load("_source_stamp_under_test", "scripts/source_stamp.py")


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True).stdout.strip()


def _repo_with_a_compiled_subtree(root: pathlib.Path) -> pathlib.Path:
    """A miniature of this repository: a git checkout with a compiled subtree."""
    subtree = root / source_stamp.CUDA_SRC_RELATIVE
    subtree.mkdir(parents=True)
    (subtree / "kernels.cu").write_text("// four planes\nint planes = 4;\n")
    (subtree / "kernels.cuh").write_text("#pragma once\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "four planes")
    return subtree




class TestTheDefectThisStampFixes:
    def test_an_uncommitted_edit_moves_the_content_hash_and_nothing_else(
            self, tmp_path):
        """The whole defect, in four lines.

        A commit and a tree hash both describe the index, so neither moves when
        a file is edited in place -- which is what makes a stamp built only from
        them able to name source the binary does not contain.
        """
        subtree = _repo_with_a_compiled_subtree(tmp_path)

        before = source_stamp.stamp(tmp_path, phase="build")
        (subtree / "kernels.cu").write_text("// three planes\nint planes = 3;\n")
        after = source_stamp.stamp(tmp_path, phase="build")

        assert after["git_sha"] == before["git_sha"]
        assert after["cuda_src_tree"] == before["cuda_src_tree"]
        # ... and the one field that does move.
        assert after["cuda_src_sha256"] != before["cuda_src_sha256"]
        assert before["cuda_src_dirty"] == source_stamp.DIRTY_CLEAN
        assert after["cuda_src_dirty"] == source_stamp.DIRTY_YES
        assert after["cuda_src_dirty_files"] == [
            source_stamp.CUDA_SRC_RELATIVE + "/kernels.cu"]

    def test_a_restored_file_under_a_later_commit_is_reported(self, tmp_path):
        """The restored-file case exactly, and the one the old stamp could not express.

        Build from a tree whose HEAD carries the narrowing but whose
        `kernels.cu` is the four-plane one restored from the earlier commit. The
        commit and the tree hash both name the narrowing; the content hash names
        the four-plane source, and equals what the four-plane commit hashed to.
        """
        subtree = _repo_with_a_compiled_subtree(tmp_path)
        four_planes = source_stamp.stamp(tmp_path, phase="build")

        (subtree / "kernels.cu").write_text("// three planes\nint planes = 3;\n")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", "narrow the record to three planes")
        three_planes = source_stamp.stamp(tmp_path, phase="build")

        # The restore: main's kernels.cu back into a tree whose HEAD narrows.
        _git(tmp_path, "checkout", "-q", "HEAD~1", "--",
             source_stamp.CUDA_SRC_RELATIVE + "/kernels.cu")
        control_arm = source_stamp.stamp(tmp_path, phase="build")

        # What the old stamp saw: indistinguishable from the narrowed build.
        assert control_arm["git_sha"] == three_planes["git_sha"]
        assert control_arm["cuda_src_tree"] == three_planes["cuda_src_tree"]

        # What the new stamp says.
        assert control_arm["cuda_src_dirty"] == source_stamp.DIRTY_YES
        assert control_arm["cuda_src_sha256"] == four_planes["cuda_src_sha256"]
        assert control_arm["cuda_src_sha256"] != three_planes["cuda_src_sha256"]

    def test_a_clean_build_is_reported_clean(self, tmp_path):
        _repo_with_a_compiled_subtree(tmp_path)
        s = source_stamp.stamp(tmp_path, phase="build")
        assert s["cuda_src_dirty"] == source_stamp.DIRTY_CLEAN
        assert s["cuda_src_dirty_files"] == []
        assert len(s["cuda_src_sha256"]) == 64
        assert len(s["cuda_src_tree"]) == 40


class TestNobodyLookedIsNotClean:
    def test_outside_a_git_checkout_the_dirty_marker_is_unknown(self, tmp_path):
        """An sdist, or a tarball. `git status` cannot answer, and the answer
        it must not give is "clean" -- the same defect once fixed for
        `checked_a_gpu_build` and the architecture probe."""
        subtree = tmp_path / source_stamp.CUDA_SRC_RELATIVE
        subtree.mkdir(parents=True)
        (subtree / "kernels.cu").write_text("int x = 1;\n")

        s = source_stamp.stamp(tmp_path, phase="build")
        assert s["cuda_src_dirty"] == source_stamp.DIRTY_UNKNOWN
        assert s["git_sha"] == "unknown"
        assert s["cuda_src_tree"] == "unknown"
        # The content hash needs no git at all, which is the point of it.
        assert len(s["cuda_src_sha256"]) == 64

    def test_a_missing_subtree_does_not_raise(self, tmp_path):
        """It cannot fail a build, the same rule as the cell registry's."""
        s = source_stamp.stamp(tmp_path, phase="build")
        assert s["cuda_src_sha256"] == "unknown"


class TestTheGeneratedHeader:
    def test_it_is_rewritten_only_when_the_stamp_moves(self, tmp_path):
        """Otherwise every build relinks the extension for a stamp that did not
        move, and the build-time regeneration would cost what it is worth."""
        subtree = _repo_with_a_compiled_subtree(tmp_path)
        header = tmp_path / "b" / "roshambo3_source_stamp.h"

        assert source_stamp.write_header(header, source_stamp.stamp(tmp_path))
        assert not source_stamp.write_header(header,
                                             source_stamp.stamp(tmp_path))

        (subtree / "kernels.cu").write_text("// edited\n")
        assert source_stamp.write_header(header, source_stamp.stamp(tmp_path))

    def test_it_defines_every_macro_the_extension_reads(self):
        """The header and interface.cpp are two files that have to agree, and
        the failure mode if they do not is a stamp that silently falls back to
        "unknown" -- a defect that looks exactly like a wrong-source stamp."""
        interface = (REPO / "src/roshambo3/backend/_cuda_src/interface.cpp"
                     ).read_text()
        header = source_stamp.header_text(source_stamp.stamp(REPO))
        for macro, attr in (
                ("ROSHAMBO3_CUDA_SRC_SHA256", "__cuda_src_sha256__"),
                ("ROSHAMBO3_CUDA_SRC_TREE", "__cuda_src_tree__"),
                ("ROSHAMBO3_CUDA_SRC_DIRTY", "__cuda_src_dirty__"),
                ("ROSHAMBO3_CUDA_SRC_DIRTY_FILES", "__cuda_src_dirty_files__"),
                ("ROSHAMBO3_SOURCE_STAMP_PHASE", "__source_stamp_phase__")):
            assert f"#define {macro} " in header, macro
            # And a fallback in interface.cpp, so a compile without the header
            # says "unknown"/-1 rather than failing to build.
            assert f"#ifndef {macro}\n" in interface, macro
            assert f'm.attr("{attr}")' in interface, attr
        assert 'include "roshambo3_source_stamp.h"' in interface
        assert '__has_include("roshambo3_source_stamp.h")' in interface

    def test_a_dirty_path_with_a_quote_in_it_does_not_break_the_header(self):
        s = {"cuda_src_sha256": 'a"b\\c', "cuda_src_tree": "t",
             "cuda_src_dirty": 1, "cuda_src_dirty_files": ['x"y'],
             "source_stamp_phase": "build"}
        text = source_stamp.header_text(s)
        assert r'"a\"b\\c"' in text
        assert r'"x\"y"' in text

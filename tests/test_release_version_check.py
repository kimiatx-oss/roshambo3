"""The packaging gate: a release tag must equal ``v`` + pyproject's version."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_release_version",
                                              ROOT / "scripts" / "check_release_version.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_matching_tag_passes():
    assert check.problems("v1.0.0", "1.0.0", "1.0.0") == []


def test_mismatched_tag_is_refused():
    assert any("!= 'v'" in p for p in check.problems("v1.0.1", "1.0.0", "1.0.0"))


def test_dev_version_is_refused_even_with_a_matching_tag():
    assert any("development" in p for p in check.problems("v1.0.0.dev0", "1.0.0.dev0", "1.0.0.dev0"))


def test_missing_tag_is_refused():
    assert any("no release tag" in p for p in check.problems(None, "1.0.0", "1.0.0"))


def test_fallback_literal_must_match():
    assert any("_SOURCE_VERSION" in p for p in check.problems("v1.0.0", "1.0.0", "0.1.0rc1"))


def test_this_tree_is_self_consistent():
    assert check.source_version() == check.project_version()


def test_command_line_accepts_this_trees_own_tag(capsys):
    tag = f"v{check.project_version()}"
    assert check.main(["--tag", tag]) == 0
    assert f"tag {tag} matches" in capsys.readouterr().out


def test_command_line_refuses_a_wrong_tag(capsys):
    assert check.main(["--tag", "v0.0.0-not-this"]) == 1
    assert "refusing" in capsys.readouterr().err

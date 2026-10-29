"""An export inside another repository must not borrow its provenance."""
import importlib.util
from pathlib import Path
import subprocess
import pytest
from bench.gitinfo import get_git_info

def test_enclosing_git_repository_is_not_the_export(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                       capture_output=True, text=True)
    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "parent.txt").write_text("unrelated source")
    git("add", "parent.txt")
    git("commit", "-q", "-m", "parent")
    export = tmp_path / "candidate"
    export.mkdir()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("export_stamp", root / "scripts/source_stamp.py")
    stamp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stamp)
    provenance = stamp.git_provenance(export)
    assert provenance["git_sha"] == "unknown"
    assert provenance["cuda_src_dirty"] == stamp.DIRTY_UNKNOWN
    record = get_git_info(str(export))
    assert record.sha == "unknown" and record.source_dirty
    assert get_git_info(str(tmp_path)).sha != "unknown"

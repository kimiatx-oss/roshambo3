"""Builder review S1 (scatter/gather count reconciliation) and S2
(assembled files are durable before the checkpoint says so), for the
format-3 file set: group files, molecules.parquet, corpus.json,
build.json."""
import json
from pathlib import Path

import pytest

from roshambo3.prep import build, prepare
from tests.prep_fixtures import fixtures, write_source

OPTIONS = dict(workers=1, records_per_unit=2, max_conf_per_group=2)


@pytest.fixture
def sdf(tmp_path):
    path = tmp_path / "input.sdf"
    write_source(path, fixtures())
    return path


def _interrupt_before_first_group(monkeypatch):
    def stop(self, *args, **kwargs):
        raise KeyboardInterrupt("stopped before any group committed")
    monkeypatch.setattr(build.GroupWriter, "add", stop)


@pytest.mark.parametrize("lose", ["shard", "metadata"])
def test_lost_committed_scratch_is_refused_not_published(tmp_path, sdf, monkeypatch, lose):
    out = tmp_path / "corpus"
    _interrupt_before_first_group(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        prepare(sdf, str(out), **OPTIONS)
    monkeypatch.undo()
    partial = Path(str(out) + ".partial")
    state = json.loads((partial / build.CHECKPOINT_NAME).read_text())
    assert state["scatter"]["complete"] and not state["gather"]["buckets"]
    tmp = partial / "_tmp"
    if lose == "shard":
        # One committed slot's conformers at one heavy-atom count.
        victims = sorted(tmp.glob("n*_s00001.met"))[:1]
        assert victims
        for met in victims:
            for path in tmp.glob(met.name[:-4] + ".*"):
                path.unlink()
    else:
        (tmp / "meta_00001.tsv").unlink()
    with pytest.raises(build.BuildRefused, match="count reconciliation"):
        prepare(sdf, str(out), resume=True, **OPTIONS)
    assert not out.exists()
    record = json.loads(Path(str(out) + ".progress.json").read_text())
    assert record["stage"] == "failed" and record["retryable"] is False
    assert "count reconciliation" in record["error"]


def test_assembled_files_are_durable_before_the_checkpoint(tmp_path, sdf, monkeypatch):
    events = []
    original_fsync = build._fsync_path
    original_save = build._BuildCheckpoint.save

    def fsync(path):
        events.append(("fsync", Path(path).name))
        return original_fsync(path)

    def save(self):
        events.append(("assembled" if self.state.get("assembled") else "save", None))
        return original_save(self)

    monkeypatch.setattr(build, "_fsync_path", fsync)
    monkeypatch.setattr(build._BuildCheckpoint, "save", save)
    out = tmp_path / "corpus"
    prepare(sdf, str(out), **OPTIONS)
    assembled = events.index(("assembled", None))
    before = [name for kind, name in events[:assembled] if kind == "fsync"]
    names = ("molecules.parquet", "corpus.json", "build.json")
    for name in names:
        assert name in before, name
    # Every group file as its group commits, before the assembly.
    assert sum(name.endswith(".r3g") for name in before) == len(list((out / "groups").iterdir()))
    # The directories themselves, after their files: names are durable too.
    last_file = max(before.index(n) for n in names)
    assert "groups" in before[last_file + 1:]
    assert "corpus.partial" in before[last_file + 1:]

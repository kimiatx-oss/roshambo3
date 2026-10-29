"""<out>.progress.json keeps being refreshed through every long phase,
so a watchdog keyed on its staleness never mistakes slow work for a hang."""
import subprocess
import sys
from pathlib import Path

from roshambo3.prep import build, prepare
from tests.test_prepare_resume import SCRIPT
from tests.prep_fixtures import fixtures, write_source


def _records(monkeypatch, out):
    seen = []
    original = build._write_json_atomic

    def spy(path, record, *, durable):
        if str(path) == str(out) + ".progress.json":
            seen.append(dict(record))
        return original(path, record, durable=durable)
    monkeypatch.setattr(build, "_write_json_atomic", spy)
    monkeypatch.setattr(build, "HEARTBEAT_S", 0.0)
    return seen


def test_every_phase_refreshes_progress(tmp_path, monkeypatch):
    source = tmp_path / "input.sdf"
    write_source(source, fixtures())
    out = tmp_path / "corpus"
    seen = _records(monkeypatch, out)
    prepare(source, str(out), workers=1, records_per_unit=2, max_conf_per_group=2)
    stages = [r["stage"] for r in seen]
    for stage in ("scatter", "gather", "metadata", "validate", "build.json"):
        assert stages.count(stage) >= 2, (stage, stages)
    for stage in ("descriptor", "publish"):
        assert stage in stages, (stage, stages)


def test_resume_replay_of_a_solid_stream_refreshes_progress(tmp_path, monkeypatch):
    source = tmp_path / "input.bin.zst"
    write_source(source, fixtures(), binary=True)
    out = tmp_path / "corpus"
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(source), str(out),
                             "_BuildCheckpoint", "commit_slot", "3", "7"],
                            capture_output=True)
    assert result.returncode == 73, result.stderr.decode()[-2000:]
    seen = _records(monkeypatch, out)
    from roshambo3.pharmacophore import PharmacophoreGenerator
    prepare(source, str(out), color_generator=PharmacophoreGenerator(), resume=True,
            workers=1, records_per_unit=2, max_conf_per_group=2, max_records=7)
    replayed = [r["replayed_slot"] for r in seen if "replayed_slot" in r]
    assert replayed[:3] == [0, 1, 2]
    assert not Path(str(out) + ".partial").exists()


def test_late_phases_heartbeat(tmp_path, monkeypatch):
    """The group planning, the statistics read back from the group files,
    the assembled-file fsyncs and the scratch removal each refresh the record
    (they were silent: the largest corpus build spent tens of minutes in them unseen)."""
    calls = {}
    for name in ("_plan_bucket", "_corpus_statistics", "_sync_assembled", "_remove_scratch"):
        original = getattr(build, name)

        def counting(*args, _name=name, _original=original, **kwargs):
            hb = kwargs.get("heartbeat") or args[-1]
            assert callable(hb), _name

            def beat(**k):
                calls[_name] = calls.get(_name, 0) + 1
                return hb(**k)
            if "heartbeat" in kwargs:
                kwargs["heartbeat"] = beat
            else:
                args = args[:-1] + (beat,)
            return _original(*args, **kwargs)
        monkeypatch.setattr(build, name, counting)
    source = tmp_path / "input.sdf"
    write_source(source, fixtures())
    prepare(source, str(tmp_path / "corpus"), workers=1, records_per_unit=2,
            max_conf_per_group=2)
    assert calls["_plan_bucket"] >= 1
    assert calls["_corpus_statistics"] >= 2              # one per group
    assert calls["_sync_assembled"] >= 2                 # parquet, corpus.json, build.json
    assert calls["_remove_scratch"] >= 2                 # one per scratch entry

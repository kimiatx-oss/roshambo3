"""Resumable prepare, no-overwrite publication, dry run and the id-hash gate."""
import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from roshambo3.backend import cpu
from roshambo3.prep import build, prepare
from tests.prep_fixtures import fixtures, write_source

#: build.json fields that describe one particular run rather than the corpus:
#: when, how long, how often resumed, and where its scratch was.
RUN_FIELDS = {"created", "timing", "run", "resumes", "validation", "scratch", "identity"}

OPTIONS = dict(workers=1, records_per_unit=2, max_conf_per_group=2)


@pytest.fixture
def sdf(tmp_path):
    path = tmp_path / "input.sdf"
    write_source(path, fixtures())
    return path


def _files(root):
    root = Path(root)
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}


def _provenance(path):
    record = json.loads(path.read_text())
    for key in ("rejected", "dropped_conformers"):
        record[key]["file"] = None             # named after the output directory
    return {k: v for k, v in record.items() if k not in RUN_FIELDS}


def assert_same_corpus(left, right):
    """Every file byte-identical, except build.json's
    run fields."""
    a, b = _files(left), _files(right)
    assert sorted(a) == sorted(b)
    for name in a:
        if name == "build.json":
            assert _provenance(a[name]) == _provenance(b[name])
        else:
            assert a[name].read_bytes() == b[name].read_bytes(), name


def _build(corpus) -> dict:
    return json.loads((Path(corpus.root) / "build.json").read_text())


def _no_litter(out):
    out = Path(out)
    for suffix in (".partial", ".lock", ".progress.json"):
        assert not Path(str(out) + suffix).exists(), suffix


KILLS = {
    # After the second scatter work unit is durable.
    "scatter": ("_BuildCheckpoint", "commit_slot", 2),
    # Right after the first group is committed.
    "gather_commit": ("_BuildCheckpoint", "commit_group", 1),
    # Mid-group: a group file is being written (the third GroupWriter.add),
    # so it exists on disk uncommitted and must be discarded on resume.
    "gather_mid": ("GroupWriter", "add", 3),
}

SCRIPT = """
import os, sys
from roshambo3.prep import build, prepare
from roshambo3.pharmacophore import PharmacophoreGenerator
cls, name, after = sys.argv[3], sys.argv[4], int(sys.argv[5])
original = getattr(getattr(build, cls), name)
calls = [0]
def killed(self, *args, **kwargs):
    if name == "add" and calls[0] + 1 >= after:
        os._exit(73)
    result = original(self, *args, **kwargs)
    calls[0] += 1
    if name != "add" and calls[0] >= after:
        os._exit(73)
    return result
setattr(getattr(build, cls), name, killed)
prepare(sys.argv[1], sys.argv[2], workers=1, records_per_unit=2,
        max_conf_per_group=2, color_generator=PharmacophoreGenerator(),
        max_records=(int(sys.argv[6]) if len(sys.argv) > 6 else None))
"""


@pytest.mark.parametrize("kill", sorted(KILLS))
def test_killed_build_resumes_to_the_uninterrupted_corpus(tmp_path, sdf, kill):
    cpu.extension()
    from roshambo3.pharmacophore import PharmacophoreGenerator
    reference = prepare(sdf, str(tmp_path / "reference"),
                        color_generator=PharmacophoreGenerator(), **OPTIONS)
    out = tmp_path / "resumed"
    cls, name, after = KILLS[kill]
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(sdf), str(out),
                             cls, name, str(after)], capture_output=True)
    assert result.returncode == 73, result.stderr.decode()[-2000:]
    assert not out.exists()
    partial = Path(str(out) + ".partial")
    assert (partial / build.CHECKPOINT_NAME).exists()
    state = json.loads((partial / build.CHECKPOINT_NAME).read_text())
    assert state["scatter"]["slots"]

    # Neither a fresh build nor a resume with other options may touch it.
    with pytest.raises(FileExistsError, match="interrupted build"):
        prepare(sdf, str(out), color_generator=PharmacophoreGenerator(), **OPTIONS)
    with pytest.raises(ValueError, match="resume refused"):
        prepare(sdf, str(out), color_generator=PharmacophoreGenerator(),
                resume=True, **{**OPTIONS, "records_per_unit": 3})
    assert (partial / build.CHECKPOINT_NAME).exists()

    resumed = prepare(sdf, str(out), color_generator=PharmacophoreGenerator(),
                      resume=True, **OPTIONS)
    assert _build(resumed)["resumes"] >= 1
    assert len(resumed.group_table) > 2
    assert_same_corpus(reference.root, resumed.root)
    _no_litter(out)


def test_killed_binary_stream_build_resumes(tmp_path):
    """The solid zstd stream cannot seek: a resume streams through committed
    batches without preparing them again."""
    cpu.extension()
    from roshambo3.pharmacophore import PharmacophoreGenerator
    source = tmp_path / "input.bin.zst"
    write_source(source, fixtures(), binary=True)
    options = dict(OPTIONS, max_records=7)
    reference = prepare(source, str(tmp_path / "reference"),
                        color_generator=PharmacophoreGenerator(), **options)
    out = tmp_path / "resumed"
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(source), str(out),
                             "_BuildCheckpoint", "commit_slot", "2", "7"],
                            capture_output=True)
    assert result.returncode == 73, result.stderr.decode()[-2000:]
    resumed = prepare(source, str(out), color_generator=PharmacophoreGenerator(),
                      resume=True, **options)
    assert _build(resumed)["resumes"] == 1
    assert_same_corpus(reference.root, resumed.root)
    _no_litter(out)


def test_interrupted_scatter_with_nothing_committed_leaves_no_litter(tmp_path, sdf, monkeypatch):
    def fail(*args, **kwargs):
        raise KeyboardInterrupt("before any unit finished")
    monkeypatch.setattr(build, "_scatter_unit", fail)
    out = tmp_path / "corpus"
    with pytest.raises(KeyboardInterrupt):
        prepare(sdf, str(out), **OPTIONS)
    assert not out.exists() and not Path(str(out) + ".partial").exists()
    assert not Path(str(out) + ".progress.json").exists()


def test_failure_after_committed_work_is_kept_and_reported(tmp_path, sdf, monkeypatch):
    def fail(*args, **kwargs):
        raise KeyboardInterrupt("paused")
    monkeypatch.setattr(build, "_merge_meta", fail)
    out = tmp_path / "corpus"
    with pytest.raises(KeyboardInterrupt):
        prepare(sdf, str(out), **OPTIONS)
    assert not out.exists()
    record = json.loads(Path(str(out) + ".progress.json").read_text())
    assert record["stage"] == "paused" and record["resumable"] is True
    monkeypatch.undo()

    def no_rescan(*args, **kwargs):
        raise AssertionError("a finished scatter must not be walked again")
    monkeypatch.setattr(build, "_scatter", no_rescan)
    corpus = prepare(sdf, str(out), resume=True, **OPTIONS)
    assert _build(corpus)["resumes"] == 1
    _no_litter(out)


def test_publication_never_overwrites_and_the_build_stays_resumable(tmp_path, sdf, monkeypatch):
    out = tmp_path / "corpus"
    original = build.os.rename

    def competing_writer(source, target):
        Path(target).mkdir(exist_ok=True)
        (Path(target) / "someone-elses").write_text("theirs")
        return original(source, target)
    monkeypatch.setattr(build.os, "rename", competing_writer)
    with pytest.raises(FileExistsError, match="appeared while this build ran"):
        prepare(sdf, str(out), **OPTIONS)
    assert (out / "someone-elses").read_text() == "theirs"
    assert sorted(p.name for p in out.iterdir()) == ["someone-elses"]
    assert Path(str(out) + ".partial").is_dir()
    monkeypatch.undo()
    (out / "someone-elses").unlink()
    corpus = prepare(sdf, str(out), resume=True, **OPTIONS)
    assert corpus.n_conf > 0
    _no_litter(out)


def test_one_writer_per_output(tmp_path, sdf):
    out = tmp_path / "corpus"
    with open(str(out) + ".lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another prepare owns"):
            prepare(sdf, str(out), **OPTIONS)
    assert not out.exists() and not Path(str(out) + ".partial").exists()


def test_resume_without_a_partial_or_with_memory_inputs_is_refused(tmp_path, sdf):
    with pytest.raises(ValueError, match="no .* to resume"):
        prepare(sdf, str(tmp_path / "corpus"), resume=True, **OPTIONS)
    with pytest.raises(ValueError, match="path inputs"):
        prepare(fixtures(), str(tmp_path / "memory"), resume=True, **OPTIONS)


def test_dry_run_writes_nothing(tmp_path, sdf, capsys):
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    out = tmp_path / "nested" / "corpus"
    report = prepare(sdf, str(out), dry_run=True, **OPTIONS)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert report["wrote_nothing"] and report["out_dir_state"] == "absent"
    assert report["records"] == sum(m.GetNumConformers() for m in fixtures())
    assert report["records_basis"] == "counted"
    assert report["sample"]["input_records"] > 0
    assert report["cpu_estimate"]["total_s"] > 0
    assert report["cpu_estimate"]["stored_vaa_s"] == pytest.approx(
        report["conformers_estimate"] * build.STORED_VAA_SECONDS_PER_CONFORMER)
    assert report["bytes_to_read"] == sdf.stat().st_size
    assert report["bytes_to_write"]["corpus_bytes"] > 0
    assert report["free_bytes"]["destination"] > 0 and report["fits"] is True

    from roshambo3.cli import main
    assert main(["prepare", str(sdf), "-o", str(out), "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["dry_run"] is True and printed["records"] == report["records"]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_id_hash_gate(tmp_path, sdf):
    good = prepare(sdf, str(tmp_path / "first"), **OPTIONS)
    expected = good.descriptor["ids_sha256"]
    assert _build(good)["id_hash_gate"] == {"expected": None, "actual": expected,
                                            "matched": None}
    out = tmp_path / "gated"
    wrong = "0" * 64
    with pytest.raises(ValueError, match="id-hash gate"):
        prepare(sdf, str(out), expect_sha256_ids=wrong, **OPTIONS)
    assert not out.exists()
    record = json.loads(Path(str(out) + ".progress.json").read_text())
    assert record["id_hash_gate"] == {"expected": wrong, "actual": expected,
                                      "matched": False}
    passed = prepare(sdf, str(tmp_path / "passed"),
                     expect_sha256_ids=expected.upper(), **OPTIONS)
    assert _build(passed)["id_hash_gate"]["matched"] is True
    with pytest.raises(ValueError, match="64-digit"):
        prepare(sdf, str(tmp_path / "bad"), expect_sha256_ids="abc", **OPTIONS)

    from roshambo3.cli import build_parser
    args = build_parser().parse_args(
        ["prepare", str(sdf), "-o", "x", "--expect-ids-sha256", expected,
         "--resume", "--no-color-self"])
    assert args.expect_ids_sha256 == expected and args.resume and args.no_color_self


def test_resume_refuses_a_format2_checkpoint(tmp_path, sdf):
    """A gather resumed across the format switch must not publish one corpus
    in two formats: a checkpoint written by the format-2 builder (v2) names
    other options and another format, and the resume refuses it whole."""
    from roshambo3.pharmacophore import PharmacophoreGenerator
    out = tmp_path / "resumed"
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(sdf), str(out),
                             "_BuildCheckpoint", "commit_group", "2"],
                            capture_output=True)
    assert result.returncode == 73, result.stderr.decode()[-2000:]
    path = Path(str(out) + ".partial") / build.CHECKPOINT_NAME
    state = json.loads(path.read_text())
    assert state["identity"]["format"] == build.CHECKPOINT_FORMAT
    state["identity"]["format"] = "roshambo3.prepare.checkpoint.v2"
    path.write_text(json.dumps(state))
    with pytest.raises(build.BuildRefused, match=r"resume refused.*differing: format"):
        prepare(sdf, str(out), color_generator=PharmacophoreGenerator(), resume=True,
                **OPTIONS)
    assert path.exists()                                  # kept, not destroyed


def test_resume_accepts_identical_feature_definitions_found_elsewhere(tmp_path):
    """Another checkout's environment finds the RDKit feature definitions at
    another path; the same bytes resume, different bytes do not."""
    fdef = tmp_path / "a" / "BaseFeatures.fdef"
    fdef.parent.mkdir()
    fdef.write_text("DefineFeature Donor [N;!H0]\n")
    same = tmp_path / "b" / "BaseFeatures.fdef"
    same.parent.mkdir()
    shutil.copy(fdef, same)
    other = tmp_path / "c" / "BaseFeatures.fdef"
    other.parent.mkdir()
    other.write_text("DefineFeature Donor [O;!H0]\n")
    identity = {"format": build.CHECKPOINT_FORMAT, "options": {"name": "x"},
                "color": {"class": "G", "fdef_path": str(fdef),
                          "interaction_r_sha256": "r"}}
    staging = tmp_path / "staging"
    staging.mkdir()
    build._BuildCheckpoint(str(staging), identity, resume=False)
    moved = {**identity, "color": {**identity["color"], "fdef_path": str(same)}}
    assert build._BuildCheckpoint(str(staging), moved, resume=True).state[
        "identity"] == identity
    changed = {**identity, "color": {**identity["color"], "fdef_path": str(other)}}
    with pytest.raises(ValueError, match="resume refused.*differing: color"):
        build._BuildCheckpoint(str(staging), changed, resume=True)
    # A path change does not excuse any other difference.
    both = {**moved, "options": {"name": "y"}}
    with pytest.raises(ValueError, match="differing: color, options"):
        build._BuildCheckpoint(str(staging), both, resume=True)


#: Kill a build after its ``assembled`` checkpoint: partway through deleting
#: its scratch, or after that and the checkpoint's removal, just before the
#: rename that publishes it (os._exit, as with kill -9).
LATE_KILL_SCRIPT = """
import os, sys
from roshambo3.prep import prepare
out, where = sys.argv[2], sys.argv[3]
unlink, rename = os.unlink, os.rename
def killing_unlink(path, *args, **kwargs):
    name = os.path.basename(os.fsdecode(path))
    if where == "scratch" and name.startswith("n") and name.endswith(".xyz"):
        os._exit(73)                      # the first shard of the scratch removal
    return unlink(path, *args, **kwargs)
def killing_rename(src, dst, *args, **kwargs):
    if where == "publish" and os.fsdecode(dst) == out:
        os._exit(73)
    return rename(src, dst, *args, **kwargs)
os.unlink, os.rename = killing_unlink, killing_rename
prepare(sys.argv[1], out, workers=1, records_per_unit=2, max_conf_per_group=2)
"""


@pytest.mark.parametrize("where", ["scratch", "publish"])
def test_resume_after_assembly_publishes_no_scratch(tmp_path, sdf, where):
    """A build killed after its ``assembled`` checkpoint is published by the
    resume -- without the scratch directory (``_tmp``) the interrupted build
    was deleting, or that the resume itself recreated."""
    reference = prepare(sdf, str(tmp_path / "reference"), **OPTIONS)
    out = tmp_path / "corpus"
    result = subprocess.run([sys.executable, "-c", LATE_KILL_SCRIPT, str(sdf), str(out),
                             where], capture_output=True)
    assert result.returncode == 73, result.stderr.decode()[-2000:]
    partial = Path(str(out) + ".partial")
    if where == "scratch":
        state = json.loads((partial / build.CHECKPOINT_NAME).read_text())
        assert state["assembled"] is True
        assert (partial / "_tmp").is_dir()              # the premise: scratch left
    else:
        assert not (partial / build.CHECKPOINT_NAME).exists()
        assert (partial / "corpus.json").exists() and (partial / "build.json").exists()
    resumed = prepare(sdf, str(out), resume=True, **OPTIONS)
    assert not (Path(resumed.root) / "_tmp").exists()
    assert sorted(p.name for p in Path(resumed.root).iterdir()) == \
        sorted(p.name for p in Path(reference.root).iterdir())
    assert_same_corpus(reference.root, resumed.root)
    _no_litter(out)


def test_scratch_removal_heartbeats_and_tolerates_a_partial_scratch(tmp_path):
    scratch = tmp_path / "_tmp"
    (scratch / "structures_build").mkdir(parents=True)
    for i in range(5):
        (scratch / f"n003_s{i:05d}.xyz").write_bytes(b"x")
    beats = []
    build._remove_scratch(str(scratch), lambda **k: beats.append(k))
    assert not scratch.exists() and len(beats) == 6
    build._remove_scratch(str(scratch))                 # already gone: no error


def test_a_staged_build_without_its_checkpoint_is_validated_before_publication(tmp_path, sdf):
    """With no checkpoint, a staged ``corpus.json`` and
    ``build.json`` were published as "a finished build awaiting publication"
    without a look. They must pass ``scan`` first; stale ones are refused."""
    from roshambo3.pharmacophore import PharmacophoreGenerator
    out = tmp_path / "corpus"
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(sdf), str(out),
                             "_BuildCheckpoint", "commit_group", "2"], capture_output=True)
    assert result.returncode == 73, result.stderr.decode()[-2000:]
    partial = Path(str(out) + ".partial")
    (partial / "corpus.json").write_bytes(b"{}")
    (partial / "build.json").write_bytes(b"{}")
    (partial / build.CHECKPOINT_NAME).unlink()
    with pytest.raises(build.BuildRefused, match="fails the scan tier"):
        prepare(sdf, str(out), color_generator=PharmacophoreGenerator(), resume=True,
                **OPTIONS)
    assert not out.exists() and partial.is_dir()


def test_a_genuine_staged_build_without_its_checkpoint_publishes_scanned(tmp_path, sdf):
    """The legitimate case of the same path (killed between the checkpoint's
    removal and the rename) still publishes, now scanned."""
    reference = prepare(sdf, str(tmp_path / "reference"), **OPTIONS)
    out = tmp_path / "corpus"
    result = subprocess.run([sys.executable, "-c", LATE_KILL_SCRIPT, str(sdf), str(out),
                             "publish"], capture_output=True)
    assert result.returncode == 73, result.stderr.decode()[-2000:]
    resumed = prepare(sdf, str(out), resume=True, **OPTIONS)
    assert resumed.scanned
    assert_same_corpus(reference.root, resumed.root)
    _no_litter(out)

"""Builder review M1: a dead scatter worker is a loud, fast failure, and
SIGTERM is handled like Ctrl-C. Both leave a build that resumes to the corpus
an uninterrupted build writes."""
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
import zstandard as zstd
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3.prep import prepare
from tests.test_prepare_resume import _no_litter, assert_same_corpus

N_MOLECULES = 48
OPTIONS = dict(records_per_unit=3, max_conf_per_group=40)

#: Runs a build whose work units each take ``ROSHAMBO3_TEST_SCATTER_DELAY``
#: seconds longer, so a worker can be killed while units are still
#: outstanding. The slowdown lives in an importable module so that it also
#: reaches workers started with ``spawn`` (macOS's default).
SCRIPT = """
import sys
from roshambo3.prep import prepare
from tests import scatter_delay
scatter_delay.install()
source, out = sys.argv[1], sys.argv[2]
prepare(source, out, workers=2, records_per_unit=3, max_conf_per_group=40)
"""
REPO = Path(__file__).resolve().parent.parent


def _templates():
    out = []
    for smiles in ("CCOc1ccccc1", "CC(=O)NC", "c1ccncc1CO", "OCCN"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert len(AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=11)) == 2
        out.append(mol)
    return out


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("sources")
    templates = _templates()
    mols = []
    for i in range(N_MOLECULES):
        mol = Chem.Mol(templates[i % len(templates)])
        mol.SetProp("_Name", f"SYN-{i:04d}")
        mols.append(mol)
    binary = root / "input.bin.zst"
    blobs = b"".join(struct.pack("<I", len(b)) + b for b in
                     (m.ToBinary(Chem.PropertyPickleOptions.AllProps) for m in mols))
    binary.write_bytes(zstd.ZstdCompressor().compress(blobs))
    sdf = root / "input.sdf"
    with Chem.SDWriter(str(sdf)) as writer:
        for mol in mols:
            writer.write(mol, confId=0)       # one record = one conformer
    return {"binary": binary, "sdf": sdf}


def _progress(out):
    try:
        return json.loads(Path(str(out) + ".progress.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _start(kind, source, out, delay=0.4):
    env = dict(os.environ, ROSHAMBO3_TEST_SCATTER_DELAY=str(delay))
    proc = subprocess.Popen([sys.executable, "-c", SCRIPT, str(source), str(out)],
                            cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(600):                      # at most 60 s to commit two units
        if _progress(out).get("slots_done", 0) >= 2 or proc.poll() is not None:
            break
        time.sleep(0.1)
    assert proc.poll() is None, proc.communicate()[1].decode()[-2000:]
    assert _progress(out).get("slots_done", 0) >= 2
    workers = [p for p in psutil.Process(proc.pid).children()
               if "resource_tracker" not in " ".join(p.cmdline())]
    assert len(workers) == 2, workers
    return proc, workers


def _finish_within(proc, seconds):
    t0 = time.monotonic()
    try:
        _, err = proc.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        pytest.fail(f"the build was still running {seconds} s later")
    return time.monotonic() - t0, err.decode()


@pytest.mark.parametrize("kind", ["binary", "sdf"])
def test_killed_worker_fails_fast_and_resumes(tmp_path, sources, kind):
    reference = prepare(sources[kind], str(tmp_path / "reference"), workers=1, **OPTIONS)
    out = tmp_path / "corpus"
    proc, workers = _start(kind, sources[kind], out)
    workers[0].send_signal(signal.SIGKILL)
    elapsed, err = _finish_within(proc, 30)
    assert proc.returncode != 0
    assert elapsed < 15, elapsed
    assert "WorkerDied" in err and "--resume" in err, err[-2000:]
    record = _progress(out)
    assert record["stage"] == "failed" and record["retryable"] is True
    assert f"pid {workers[0].pid} was killed by SIGKILL" in record["error"]
    assert record["resumable"] is True
    assert not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                   for p in workers[1:])
    resumed = prepare(sources[kind], str(out), workers=2, resume=True, **OPTIONS)
    assert json.loads((Path(resumed.root) / "build.json").read_text())["resumes"] == 1
    assert_same_corpus(reference.root, resumed.root)
    _no_litter(out)


@pytest.mark.parametrize("kind", ["binary", "sdf"])
def test_sigterm_pauses_and_resumes(tmp_path, sources, kind):
    reference = prepare(sources[kind], str(tmp_path / "reference"), workers=1, **OPTIONS)
    out = tmp_path / "corpus"
    proc, workers = _start(kind, sources[kind], out)
    proc.send_signal(signal.SIGTERM)
    elapsed, err = _finish_within(proc, 30)
    assert proc.returncode == 128 + signal.SIGTERM, err[-2000:]
    assert elapsed < 15, elapsed
    record = _progress(out)
    assert record["stage"] == "paused", record
    assert record["error"] == "BuildTerminated: terminated by SIGTERM"
    assert record["pid"] == proc.pid and record["resumable"] is True
    for worker in workers:
        assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
    resumed = prepare(sources[kind], str(out), workers=2, resume=True, **OPTIONS)
    assert_same_corpus(reference.root, resumed.root)
    _no_litter(out)


def test_dead_worker_is_reported_not_awaited():
    """The pool: a killed worker fails the outstanding unit in seconds, by
    pid and signal, and closing the broken pool does not hang either --
    even when the victim was idle (multiprocessing.Pool's terminate() did,
    on the task-queue lock the idle worker held)."""
    import os
    import time as clock
    from roshambo3.prep import build
    for busy in (False, True):
        pool = build._ScatterPool(3, None)
        try:
            # Every worker started. Under fork the executor starts all three
            # at once; under spawn and forkserver it starts one per submission
            # that finds no idle worker, so keep three busy at the same time.
            for future in [pool.submit(clock.sleep, 0.5) for _ in range(3)]:
                pool.result(future)
            workers = pool.workers
            assert len(workers) == 3
            future = pool.submit(clock.sleep, 30 if busy else 0)
            if busy:
                clock.sleep(0.5)
            victim = workers[0]
            os.kill(victim.pid, signal.SIGKILL)
            if not busy:
                future = pool.submit(clock.sleep, 30)
            t0 = clock.monotonic()
            with pytest.raises(build.WorkerDied, match=f"pid {victim.pid} was killed by SIGKILL"):
                pool.result(future)
            assert clock.monotonic() - t0 < 10
        finally:
            t0 = clock.monotonic()
            pool.close(abort=True)
            assert clock.monotonic() - t0 < 10
        assert not any(p.is_alive() for p in workers)


def test_workers_exit_when_the_parent_is_killed(tmp_path, sources):
    """kill -9 of the parent must not orphan its workers (they would hold
    their memory, and the pipes of whoever started the build, forever)."""
    out = tmp_path / "corpus"
    proc, workers = _start("binary", sources["binary"], out, delay=2.0)
    proc.send_signal(signal.SIGKILL)
    proc.wait(10)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(
            w.is_running() and w.status() != psutil.STATUS_ZOMBIE for w in workers):
        time.sleep(0.2)
    assert not any(w.is_running() and w.status() != psutil.STATUS_ZOMBIE
                   for w in workers)
    proc.communicate(timeout=10)          # the pipes reach end-of-file
    resumed = prepare(sources["binary"], str(out), workers=2, resume=True, **OPTIONS)
    assert json.loads((Path(resumed.root) / "build.json").read_text())["resumes"] == 1

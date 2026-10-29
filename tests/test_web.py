"""Web contracts and chemistry correctness; all fixtures run without CUDA."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from rdkit import Chem
from rdkit.Chem import AllChem
from roshambo3.web.config import Database, Settings, load_config
from roshambo3.web.store import Store
from roshambo3.web.chemistry import prepare_query, molblock_at
from roshambo3.web.screening import merge_hits
from roshambo3.web.app import create_app


def molecule(smiles="CCO", name="ethanol"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=13) == 0
    mol.SetProp("_Name", name)
    return mol


@pytest.fixture(scope="session")
def web_corpus(tmp_path_factory):
    from roshambo3.prep import prepare
    root = tmp_path_factory.mktemp("web-corpus") / "corpus"
    return prepare([molecule()], str(root), workers=1)


@pytest.fixture
def settings(tmp_path, web_corpus):
    return Settings({"test": Database("test", "Test database", web_corpus.root)},
                    tmp_path / "jobs")


def test_ensemble_merge_matches_exhaustive_max_pool():
    # Includes ties, two conformers per molecule, and molecule IDs too large
    # for JavaScript's exact numeric range.
    rng = np.random.default_rng(42)
    mids = np.repeat(np.arange(25, dtype=np.int64) + 2**60, 2)
    gids = np.arange(len(mids))
    scores = rng.integers(1, 9, size=(4, len(mids), 20)).astype(np.float32)
    for col in (0, 1):
        for k in (1, 7, 25, 50):
            gold = {}
            parts, ids, conf_indices = [], [], []
            for qi, rows in enumerate(scores):
                local = {}
                for row, gid, mid in zip(rows, gids, mids):
                    key = (-float(row[col]), int(gid), qi)
                    if mid not in local or key < local[mid][0]:
                        local[mid] = (key, row, gid)
                    if mid not in gold or key < gold[mid][0]:
                        gold[mid] = (key, row, gid)
                selected = sorted(local.items(), key=lambda x: x[1][0])[:k]
                parts.append((np.array([v[1] for _, v in selected]), np.array([v[2] for _, v in selected])))
                ids.append(np.array([mid for mid, _ in selected]))
                conf_indices.append(np.array([int(v[2]) % 2 for _, v in selected]))
            got = merge_hits(parts, ids, k, col, conf_indices=conf_indices)
            want = sorted(gold.items(), key=lambda x: x[1][0])[:k]
            assert got[3].tolist() == [int(mid) for mid, _ in want]
            assert got[1].tolist() == [int(v[2]) for _, v in want]
            assert got[2].tolist() == [v[0][2] for _, v in want]


def test_ensemble_merge_requires_conformer_identity():
    with pytest.raises(ValueError, match="requires conformer identities"):
        merge_hits([], [], 1, 0)


def test_query_generation_and_mixed_graph_refusal(tmp_path):
    req = {"smiles": "CCO", "conformers": 4, "generate": True, "minimize": True}
    details = prepare_query(tmp_path, req, 50)
    assert 1 <= details["conformers"] <= 4
    assert details["seed"] == 0xC0FFEE
    read = list(Chem.SDMolSupplier(str(tmp_path / "query.sdf"), removeHs=False))
    assert all(m.GetNumConformers() == 1 for m in read)
    with Chem.SDWriter(str(tmp_path / "input.sdf")) as writer:
        writer.write(molecule())
        writer.write(molecule("CCN"))
    with pytest.raises(ValueError, match="mixed molecules"):
        prepare_query(tmp_path, {"suffix": ".sdf"}, 50)


def test_query_2d_requires_generation(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    AllChem.Compute2DCoords(mol)
    Chem.MolToMolFile(mol, str(tmp_path / "input.mol"))
    with pytest.raises(ValueError, match="no 3D"):
        prepare_query(tmp_path, {"suffix": ".mol"}, 50)


def test_pose_serializer_preserves_chirality_and_atom_order():
    from roshambo3.prep.conformer import strip_hydrogens, prep_conformer
    mol = molecule("C[C@H](O)C(=O)O")
    mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    heavy = strip_hydrogens(mol)
    xyz = prep_conformer(heavy.GetConformer().GetPositions())
    reread = Chem.MolFromMolBlock(molblock_at(mol, xyz), removeHs=False)
    assert [a.GetAtomicNum() for a in reread.GetAtoms()] == [a.GetAtomicNum() for a in heavy.GetAtoms()]
    assert Chem.MolToSmiles(reread) == Chem.MolToSmiles(heavy)
    np.testing.assert_allclose(reread.GetConformer().GetPositions(), xyz, atol=0.00006)


def test_store_restart_and_expiration(settings):
    store = Store(settings.jobs)
    active = store.create({})["id"]
    waiting = store.create({})["id"]
    interrupted_upload = store.create({})["id"]
    store.update(active, "screening")
    store.update(waiting, "queued")
    store.recover()
    assert store.get(active)["state"] == "failed"
    assert store.get(interrupted_upload)["state"] == "failed"
    assert store.next() == waiting
    with store.connection() as con:
        con.execute("UPDATE jobs SET updated=0")
    store.expire(7)
    assert store.next() == waiting
    with pytest.raises(KeyError):
        store.get(active)


def test_http_submission_validation_and_history(settings, monkeypatch):
    app = create_app(settings, supervise=False)
    monkeypatch.setattr(app.state.supervisor, "launch", lambda *args: None)
    with TestClient(app) as client:
        assert client.get("/api/databases").json()["databases"][0]["modes"] == ["shape"]
        assert client.post("/api/queries", data={"smiles": "CCO", "conformers": 100}).status_code == 422
        assert client.post("/api/queries", files={"file": ("bad.exe", b"xx")}).status_code == 422
        j = client.post("/api/queries", data={"smiles": "CCO"}).json()
        ident = j["id"]
        store = Store(settings.jobs)
        store.update(ident, "ready")
        assert client.post(f"/api/jobs/{ident}/screen", json={"database": "test", "mode": "combo"}).status_code == 422
        assert client.post(f"/api/jobs/{ident}/screen", json={"database": "test", "k": 10001}).status_code == 422
        assert client.post(f"/api/jobs/{ident}/screen", json={"database": "test"}).status_code == 202
        assert client.post(f"/api/jobs/{ident}/cancel").json()["state"] == "canceled"
        assert client.get("/api/jobs/nonexistent").status_code == 404
        assert client.get(f"/api/jobs/{ident}/hits").status_code == 409


def test_worker_preparation_failure_is_persisted(settings):
    from roshambo3.web.jobs import child
    store = Store(settings.jobs)
    ident = store.create({"smiles": "nonsense", "conformers": 1})["id"]
    child(settings, ident, "prepare")
    assert store.get(ident)["state"] == "failed"


def test_preparation_process_start_failure_is_terminal(settings, monkeypatch):
    app = create_app(settings, supervise=False)
    process = SimpleNamespace(close=lambda: closed.append(True))
    def fail():
        raise OSError("injected process start failure")
    process.start = fail
    closed = []
    from roshambo3.web import jobs
    monkeypatch.setattr(jobs.mp, "get_context", lambda _: SimpleNamespace(Process=lambda **kwargs: process))
    with TestClient(app) as client:
        response = client.post("/api/queries", data={"smiles": "CCO"})
        assert response.status_code == 503
        latest = client.get("/api/jobs").json()[0]
        assert latest["state"] == "failed"
        assert "process start failure" in latest["error"]
        assert app.state.supervisor.children == {}
    assert closed == [True]


def test_cancel_does_not_overwrite_concurrent_completion(settings, monkeypatch):
    store = Store(settings.jobs)
    ident = store.create({})["id"]
    store.update(ident, "queued")
    original_touch = Path.touch
    def complete_on_touch(path, *args, **kwargs):
        if path.name == "cancel":
            store.update(ident, "completed", hits=1)
        return original_touch(path, *args, **kwargs)
    monkeypatch.setattr(Path, "touch", complete_on_touch)
    with TestClient(create_app(settings, supervise=False)) as client:
        assert client.post(f"/api/jobs/{ident}/cancel").json()["state"] == "completed"
    assert store.get(ident)["state"] == "completed"


def test_upload_read_failure_is_terminal(settings, monkeypatch):
    from starlette.datastructures import UploadFile
    async def fail(*args):
        raise OSError("injected upload read failure")
    monkeypatch.setattr(UploadFile, "read", fail)
    app = create_app(settings, supervise=False)
    with TestClient(app) as client:
        response = client.post("/api/queries", files={"file": ("query.sdf", b"incomplete")})
        assert response.status_code == 503
        latest = client.get("/api/jobs").json()[0]
        assert latest["state"] == "failed"
        assert "upload read failure" in latest["error"]
        assert app.state.supervisor.children == {}


def test_config_relative_paths_and_duplicate_ids(tmp_path, web_corpus):
    config = tmp_path / "web.toml"
    relative = os.path.relpath(web_corpus.root, tmp_path)
    entry = f'[[databases]]\nid="test"\nname="Test"\npath="{relative}"\n'
    config.write_text(entry)
    assert load_config(config).databases["test"].path == Path(web_corpus.root)
    config.write_text(entry + entry)
    with pytest.raises(ValueError, match="unique"):
        load_config(config)


def test_config_rejects_manifest_only_format2_database(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text("{}")
    config = tmp_path / "web.toml"
    config.write_text('[server]\n[[databases]]\nid="legacy"\nname="Legacy"\n'
                      'path="legacy"\n')

    with pytest.raises(ValueError, match="missing corpus.json"):
        load_config(config)


def test_engine_callbacks_release_buffers(tmp_path):
    from roshambo3.prep import prepare
    from roshambo3.engine import Engine, make_source, ScreenConfig
    from roshambo3.engine import from_arrays
    corpus = prepare([molecule()], str(tmp_path / "corpus"), workers=1)
    queries = from_arrays([np.zeros((3, 3), np.float32)], ["query"])
    calls = []
    def score_pairs(**kwargs):
        calls.append(1)
        # The engine launches dense -- no pair list.
        assert kwargs["pair_query_row"] is None and kwargs["pair_library_row"] is None
        return np.zeros((kwargs["query_coords"].shape[0]
                          * kwargs["library_counts"].shape[0], 20), np.float32)
    backend = SimpleNamespace(score_pairs=score_pairs, available=lambda: False)
    class Sink:
        def prepare(self, *args): pass
        def add_block(self, *args): pass
        def finalize(self): return "done"
    engine = Engine(pinned=False, backend=backend)
    def source(): return make_source(corpus, workers=2, prefetch_depth=1)
    with pytest.raises(InterruptedError):
        engine.run(source(), queries, ScreenConfig(), Sink(), canceled=lambda: True)
    assert not calls
    assert engine._pool.available == len(engine._pool)
    def failing_progress(stats): raise ValueError("callback failed")
    with pytest.raises(ValueError, match="callback failed"):
        engine.run(source(), queries, ScreenConfig(), Sink(), progress=failing_progress)
    assert engine._pool.available == len(engine._pool)
    seen = []
    stats, result = engine.run(source(), queries, ScreenConfig(), Sink(),
                              progress=lambda s: seen.append(s.n_conformers))
    assert seen == [1] and stats.n_conformers == 1 and result == "done"


def test_preparation_cancellation_and_atomic_submission(settings):
    from roshambo3.web.jobs import child
    store = Store(settings.jobs)
    ident = store.create({"smiles": "CCO", "conformers": 1})["id"]
    (settings.jobs / ident / "cancel").touch()
    child(settings, ident, "prepare")
    assert store.get(ident)["state"] == "canceled"
    store.update(ident, "ready")
    assert store.update(ident, "queued", expected="ready")
    assert not store.update(ident, "queued", expected="ready")


def test_supervisor_prepares_in_spawned_process(settings):
    import time
    from roshambo3.web.jobs import Supervisor
    supervisor = Supervisor(settings)
    supervisor.start()
    try:
        ident = supervisor.store.create({"smiles": "CCO", "conformers": 1})["id"]
        supervisor.launch(ident, "prepare")
        deadline = time.monotonic() + 15
        while supervisor.store.get(ident)["state"] in ("draft", "preparing") and time.monotonic() < deadline:
            supervisor.stop.wait(0.05)
        assert supervisor.store.get(ident)["state"] == "ready"
    finally:
        supervisor.close()


def test_gpu_lock_pins_resolved_uuid_and_restores_visibility(tmp_path, monkeypatch):
    import os
    from roshambo3.web import gpu
    monkeypatch.setattr(gpu.subprocess, "check_output", lambda *a, **k: "0, GPU-aaa\n1, GPU-bbb\n")
    monkeypatch.setattr(gpu, "gpu_lock_path", lambda uuid: tmp_path / uuid)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert gpu.device_uuid(0) == "GPU-bbb"
    with gpu.gpu_lock(0, lambda: False) as uuid:
        assert uuid == "GPU-bbb"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == uuid
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b")
    assert gpu.device_uuid(0) == "GPU-bbb"
    with pytest.raises(InterruptedError):
        with gpu.gpu_lock(0, lambda: True): pass
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="disables"):
        gpu.device_uuid(0)
    with pytest.raises(ValueError, match="outside"):
        gpu.device_uuid(2)


def test_sidecar_caches_are_bounded_and_preserve_live_views():
    import threading
    from roshambo3.web.storage import BoundedSource, BoundedCatalog

    class Group:
        def __init__(self, i):
            self.runs = np.array([(i, 1, 1)], dtype=[
                ("mol_id", "<i8"), ("n_conf", "<u2"), ("n_real", "<u2")])
            self.run_start = np.array([0, 1], dtype="<u4")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def section(self, name):
            assert name == "runs"
            return self.runs

    class Corpus:
        descriptor = {"format": {"major": 3}}
        group_table = [{"n_conf": 1} for _ in range(100)]
        n_mol = 100

        def group(self, i):
            return Group(i)

    corpus = Corpus()
    catalog = BoundedCatalog(corpus)
    source = BoundedSource.__new__(BoundedSource)
    source.corpus, source._sidecars = corpus, {}
    source._lock, source.color = threading.RLock(), False
    live = source._group_sidecars(0)["runs"]["mol_id"]
    for i in range(100):
        source._group_sidecars(i)
    assert len(source._sidecars) == 16
    assert live.tolist() == [0]

    catalog_live = catalog._group_sidecars(0)["runs"]["mol_id"]
    for i in range(100):
        catalog._group_sidecars(i)
    assert sum(s is not None for s in catalog._sidecars) == 16
    assert catalog_live.tolist() == [0]


@pytest.mark.parametrize("mode", ["shape", "rescore", "combo"])
def test_screen_adapter_and_persisted_overlays(tmp_path, monkeypatch, mode):
    """Real corpus/query/pose/HTTP path with only the device scorer replaced."""
    from contextlib import contextmanager
    import roshambo3
    import roshambo3.engine as engine_module
    from roshambo3.prep import prepare
    from roshambo3.format import Catalog
    from roshambo3.pharmacophore import PharmacophoreGenerator
    from roshambo3.web import gpu
    from roshambo3.web.screening import run_screen
    from roshambo3.backend import ColorPlacement
    sdf = tmp_path / "original.sdf"
    with Chem.SDWriter(str(sdf)) as writer:
        writer.write(molecule())
        writer.write(molecule("CCN", "amine"))
    corpus = prepare(str(sdf), str(tmp_path / "corpus"), workers=1,
                     color_generator=PharmacophoreGenerator())
    settings = Settings({"test": Database("test", "Test", corpus.root)}, tmp_path / "jobs")
    store = Store(settings.jobs)
    ident = store.create({"suffix": ".sdf", "database": "test", "mode": mode, "k": 2, "mixing": 0.5})["id"]
    directory = settings.jobs / ident
    with Chem.SDWriter(str(directory / "input.sdf")) as writer:
        writer.write(molecule())
        second = molecule()
        second.GetConformer().SetAtomPosition(0, (3, 1, 2))
        writer.write(second)
    details = prepare_query(directory, store.get(ident)["request"], 50)
    store.update(ident, "queued", preparation=details)
    @contextmanager
    def lock(*args): yield "test-gpu"
    monkeypatch.setattr(gpu, "gpu_lock", lock)
    monkeypatch.setattr(roshambo3, "load_queries", lambda *a, **k: SimpleNamespace(n_conformers=2))
    class Engine:
        def __init__(self, **kwargs): assert kwargs["backend"] == "cuda"
        def close(self): pass
        def make_topk(self, k, level, rank_col):
            return engine_module.MoleculeTopK(k=k, rank_col=rank_col)
        def run(self, source, queries, config, sink, **kwargs):
            assert config.color == (mode != "shape")
            assert config.color_placement == (ColorPlacement.IN_GRADIENT if mode == "combo" else ColorPlacement.AT_POSE)
            assert sink.rank_col == (1 if mode == "shape" else 0)
            scores = np.zeros((2, 20), np.float32)
            scores[:, 9] = scores[:, 16] = 1
            scores[:, :3] = [[0.6, 0.6, 0.6], [0.5, 0.5, 0.5]]
            second = scores.copy()
            second[0, :3] = 0.9
            second[0, 13:16] = [1, 2, 3]
            kwargs["progress"](engine_module.ScreenStats(n_conformers=2))
            return engine_module.ScreenStats(n_conformers=2), [(scores, np.array([0, 1])), (second, np.array([0, 1]))]
    monkeypatch.setattr(engine_module, "Engine", Engine)
    run_screen(settings, ident)
    assert store.get(ident)["state"] == "completed"
    assert store.get(ident)["eta_seconds"] is None
    assert store.get(ident)["submitted_conformers"] == 2
    assert store.get(ident)["screen_elapsed_s"] >= 0
    with TestClient(create_app(settings, supervise=False)) as client:
        rows = client.get(f"/api/jobs/{ident}/hits").json()["rows"]
        assert len(rows) == 2 and rows[0]["query_index"] == 1
        response = client.get(f"/api/jobs/{ident}/hits/1/overlay")
        assert response.status_code == 200
        overlay = response.json()
        hit = Chem.MolFromMolBlock(overlay["hit"])
        expected = Catalog(corpus).coordinates([0])[0] + [1, 2, 3]
        np.testing.assert_allclose(hit.GetConformer().GetPositions(), expected, atol=0.00006)
        from roshambo3.web.chemistry import payload
        queries = list(Chem.SDMolSupplier(str(directory / "query.sdf"), removeHs=False))
        assert overlay["query"] == payload(queries[1])
        for kind in ("csv", "manifest", "sdf?ranks=1,2"):
            assert client.get(f"/api/jobs/{ident}/export/{kind}").status_code == 200

        # A previously viewed hit must still validate the format-3 corpus.
        molecules_path = Path(corpus.root) / "molecules.parquet"
        original_molecules = molecules_path.read_bytes()
        molecules_path.write_bytes(original_molecules[:-1] +
                                   bytes([original_molecules[-1] ^ 1]))
        assert client.get(f"/api/jobs/{ident}/hits/1/overlay").status_code == 422
        assert client.get(f"/api/jobs/{ident}/export/sdf?ranks=1").status_code == 422
        molecules_path.write_bytes(original_molecules)
        from roshambo3.format.r3g.reader import verify
        verify(corpus.root, "scan")
        restored = client.get(f"/api/jobs/{ident}/hits/1/overlay")
        assert restored.status_code == 200, restored.text


@pytest.mark.parametrize("mode", ["shape", "rescore", "combo"])
def test_cuda_screen_overlay_and_exports(tmp_path, mode):
    """Explicit host-side smoke test, including real color preparation."""
    import os
    if os.environ.get("ROSHAMBO3_WEB_GPU_TEST") != "1":
        pytest.skip("Set ROSHAMBO3_WEB_GPU_TEST=1 on a CUDA screening host")
    from roshambo3.backend import cuda
    if not cuda.available():
        pytest.skip("CUDA extension is unavailable")
    from roshambo3.prep import prepare
    from roshambo3.pharmacophore import PharmacophoreGenerator
    from roshambo3.web.screening import run_screen
    sdf = tmp_path / "original.sdf"
    with Chem.SDWriter(str(sdf)) as writer:
        writer.write(molecule())
        writer.write(molecule("CCN", "amine"))
    corpus = prepare(str(sdf), str(tmp_path / "corpus"), workers=1,
                     color_generator=PharmacophoreGenerator())
    settings = Settings({"test": Database("test", "Test", corpus.root)}, tmp_path / "jobs")
    store = Store(settings.jobs)
    ident = store.create({"smiles": "CCO", "generate": True, "conformers": 2,
                          "database": "test", "mode": mode, "k": 2, "mixing": 0.5})["id"]
    details = prepare_query(settings.jobs / ident, store.get(ident)["request"], 50)
    store.update(ident, "queued", preparation=details)
    run_screen(settings, ident)
    assert store.get(ident)["state"] == "completed"
    with TestClient(create_app(settings, supervise=False)) as client:
        rows = client.get(f"/api/jobs/{ident}/hits").json()["rows"]
        assert len(rows) == 2 and len({r["mol_id"] for r in rows}) == 2
        overlay = client.get(f"/api/jobs/{ident}/hits/1/overlay")
        assert overlay.status_code == 200
        query = Chem.MolFromMolBlock(overlay.json()["query"])
        hit = Chem.MolFromMolBlock(overlay.json()["hit"])
        assert query is not None and hit is not None
        assert rows[0]["shape"] > 0.99
        for kind in ("csv", "manifest", "sdf?ranks=1,2"):
            exported = client.get(f"/api/jobs/{ident}/export/{kind}")
            assert exported.status_code == 200
        exported = client.get(f"/api/jobs/{ident}/export/sdf?ranks=1,2")
        import io
        assert len(list(Chem.ForwardSDMolSupplier(io.BytesIO(exported.content)))) == 4

@pytest.mark.parametrize('backend,metal_sampler', [
    ('cpu', None), ('metal', None), ('metal', 'manual'), ('metal', 'texture16')])
@pytest.mark.parametrize('mode', ['shape', 'rescore', 'combo'])
def test_native_screen_and_overlays_without_cuda(tmp_path, monkeypatch, mode, backend, metal_sampler):
    if backend == 'metal':
        from roshambo3.backend import metal
        if not metal.available():
            if os.environ.get('ROSHAMBO3_REQUIRE_METAL') == '1':
                pytest.fail('Metal GPU required')
            pytest.skip('Metal unavailable')
    from roshambo3.backend import cuda
    from roshambo3.prep import prepare
    from roshambo3.pharmacophore import PharmacophoreGenerator
    from roshambo3.web.chemistry import prepare_query
    from roshambo3.web.screening import run_screen
    from roshambo3.web import gpu
    from dataclasses import replace
    monkeypatch.setattr(cuda, 'extension', lambda: pytest.fail('CPU web invoked CUDA'))
    monkeypatch.setattr(gpu, 'device_uuid', lambda *a: pytest.fail('CPU web invoked NVIDIA discovery'))
    monkeypatch.setenv('ROSHAMBO3_CPU_LOCK', str(tmp_path / 'cpu.lock'))
    monkeypatch.setenv('ROSHAMBO3_GPU_LOCK', str(tmp_path / 'gpu.lock'))
    sdf = tmp_path / 'library.sdf'
    with Chem.SDWriter(str(sdf)) as writer:
        writer.write(molecule())
        writer.write(molecule('CCN', 'amine'))
    corpus = prepare(str(sdf), str(tmp_path / 'corpus'), workers=1,
                     color_generator=PharmacophoreGenerator())
    settings = Settings({'test': Database('test', 'Test', corpus.root)},
                        tmp_path / 'jobs', backend=backend, cpu_workers=2,
                        metal_field_sampler=metal_sampler)
    store = Store(settings.jobs)
    ident = store.create({'suffix': '.sdf', 'database': 'test', 'mode': mode, 'k': 2, 'mixing': .5})['id']
    directory = settings.jobs / ident
    with Chem.SDWriter(str(directory / 'input.sdf')) as writer:
        writer.write(molecule())
    details = prepare_query(directory, store.get(ident)['request'], 50)
    store.update(ident, 'queued', preparation=details)
    run_screen(settings, ident)
    job = store.get(ident)
    assert job['state'] == 'completed'
    assert job['progress_basis'] == ('completed' if backend == 'cpu' else 'submitted')
    assert job['completed_conformers'] == 2
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['backend'] == backend
    if backend == 'cpu':
        assert manifest['gpu_uuid'] is None
        assert manifest['stats']['cpu_scan']['workers'] == 2
    else:
        assert manifest['resource_identity'].startswith('metal-')
        policy = manifest['backend_options']
        assert policy['field_sampler'] == (metal_sampler or 'texture16')
        assert policy['color_field_sampler'] == 'texture16'
        assert policy['color_field_min_sources'] == 12
        assert policy['color_field_spacing_a'] == 0.25
        assert policy['color_field_types'] == [1, 2]
    with TestClient(create_app(settings, supervise=False)) as client:
        rows = client.get(f'/api/jobs/{ident}/hits').json()['rows']
        assert len(rows) == 2
        assert client.get(f'/api/jobs/{ident}/hits/1/overlay').status_code == 200


def test_resource_lock_cpu_cancellation_and_visibility(tmp_path, monkeypatch):
    from roshambo3.web.gpu import resource_lock
    monkeypatch.setenv('ROSHAMBO3_CPU_LOCK', str(tmp_path / 'cpu.lock'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'unchanged')
    with resource_lock('cpu', 0, lambda: False) as identity:
        assert identity == 'cpu'
        with pytest.raises(InterruptedError):
            with resource_lock('cpu', 0, lambda: True):
                pytest.fail('entered canceled lock')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'unchanged'

@pytest.mark.parametrize('entry', ['backend = "bogus"', 'cpu_workers = 0',
    'cpu_workers = true', 'cpu_memory_budget_bytes = -1',
    'cpu_topk_state_budget_bytes = 0', 'cpu_qos = "bogus"',
    'backend = "cpu"\ndevice = 1', 'metal_field_sampler = "bogus"',
    'backend = "cpu"\nmetal_field_sampler = "texture16"'])
def test_invalid_compute_server_settings(tmp_path, web_corpus, entry):
    config = tmp_path / 'server.toml'
    config.write_text('[server]\n' + entry + f'\n[[databases]]\nid="test"\nname="Test"\npath="{web_corpus.root}"\n')
    with pytest.raises(ValueError):
        load_config(config)

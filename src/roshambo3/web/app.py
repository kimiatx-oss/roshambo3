"""HTTP surface for the trusted shared lab workspace."""
from contextlib import asynccontextmanager
from pathlib import Path
import csv
import io
import json
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .jobs import Supervisor
from .store import Store


class ScreenRequest(BaseModel):
    database: str
    mode: str = "shape"
    k: int = 1000
    mixing: float = 0.5


def create_app(settings, supervise=True):
    store = Store(settings.jobs)
    supervisor = Supervisor(settings)
    @asynccontextmanager
    async def lifespan(app):
        if supervise:
            supervisor.start()
        try:
            yield
        finally:
            if supervise:
                supervisor.close()
    app = FastAPI(title="ROSHAMBO3", lifespan=lifespan)
    app.state.supervisor = supervisor

    def job(ident):
        try:
            return store.get(ident)
        except KeyError:
            raise HTTPException(404, "Job not found")

    def directory(ident):
        job(ident)  # UUIDs come from SQLite, never from a filesystem path.
        return settings.jobs / ident

    def complete(ident):
        j = job(ident)
        if j["state"] != "completed":
            raise HTTPException(409, "Results are not available yet")
        return j

    @app.get("/api/databases")
    def databases():
        return {"databases": [db.public() for db in settings.databases.values()],
                "limits": {"max_k": settings.max_k, "max_conformers": settings.max_conformers,
                           "upload_bytes": settings.upload_bytes, "retention_days": settings.retention_days}}

    @app.get("/api/jobs")
    def history():
        return store.list()

    @app.get("/api/jobs/{ident}")
    def status(ident: str):
        return job(ident)

    @app.post("/api/queries", status_code=202)
    async def prepare(file: UploadFile | None = File(None), smiles: str = Form(""),
                      generate: bool = Form(False), conformers: int = Form(10), minimize: bool = Form(False)):
        if bool(file) == bool(smiles.strip()):
            raise HTTPException(422, "Supply either an SDF/MOL file or SMILES")
        if not 1 <= conformers <= settings.max_conformers:
            raise HTTPException(422, "Conformer count exceeds the configured limit")
        suffix = Path(file.filename or "").suffix.lower() if file else ".smi"
        if file and suffix not in (".sdf", ".sd", ".mol"):
            raise HTTPException(422, "Upload SDF/SD or MOL")
        if len(smiles.encode()) > settings.upload_bytes:
            raise HTTPException(413, "SMILES input exceeds the configured size limit")
        j = store.create({"suffix": suffix, "smiles": smiles.strip(), "generate": generate,
                          "conformers": conformers, "minimize": minimize})
        ident = j["id"]
        if file:
            size = 0
            try:
                with (directory(ident) / ("input" + suffix)).open("wb") as out:
                    while chunk := await file.read(65536):
                        size += len(chunk)
                        if size > settings.upload_bytes:
                            store.update(ident, "failed", error="Upload exceeded the configured size limit")
                            raise HTTPException(413, "Upload exceeds the configured size limit")
                        out.write(chunk)
            except HTTPException:
                raise
            except BaseException as exc:
                store.update(ident, "failed", error=f"Upload did not complete: {exc}")
                if isinstance(exc, Exception):
                    raise HTTPException(503, "Upload did not complete; please try again") from exc
                raise
            finally:
                await file.close()
        store.update(ident, "preparing")
        try:
            supervisor.launch(ident, "prepare")
        except ValueError as exc:
            store.update(ident, "failed", error=str(exc))
            raise HTTPException(429, str(exc))
        except Exception as exc:
            store.update(ident, "failed", error=f"Could not start preparation worker: {exc}")
            raise HTTPException(503, "Could not start preparation worker; please try again")
        return job(ident)

    @app.post("/api/jobs/{ident}/screen", status_code=202)
    def screen(ident: str, request: ScreenRequest):
        j = job(ident)
        if j["state"] != "ready":
            raise HTTPException(409, "Prepare a query before screening")
        db = settings.databases.get(request.database)
        if not db:
            raise HTTPException(422, "Unknown database")
        public_db = db.public()
        if request.mode not in public_db["modes"]:
            raise HTTPException(
                422, public_db.get("color_error") or
                "Scoring mode is unavailable for this database")
        if not 1 <= request.k <= settings.max_k or not 0 <= request.mixing <= 1:
            raise HTTPException(422, "Invalid K or color mixing")
        if not store.update(ident, "queued", expected="ready", request={**j["request"], **request.model_dump()}):
            raise HTTPException(409, "This query has already been submitted or canceled")
        return job(ident)

    @app.post("/api/jobs/{ident}/cancel")
    def cancel(ident: str):
        j = job(ident)
        if j["state"] in ("completed", "failed", "canceled"):
            return j
        (directory(ident) / "cancel").touch(exist_ok=True)
        if j["state"] in ("draft", "ready", "queued"):
            store.update(ident, "canceled", expected=j["state"])
        return job(ident)

    @app.get("/api/jobs/{ident}/hits")
    def hits(ident: str, offset: int = 0, limit: int = 50):
        complete(ident)
        if offset < 0 or not 1 <= limit <= 200:
            raise HTTPException(422, "Invalid page")
        rows = json.loads((directory(ident) / "hits.json").read_text())
        return {"total": len(rows), "rows": rows[offset:offset + limit]}

    def load_overlay(ident, rank):
        j = complete(ident)
        root = directory(ident)
        rows = json.loads((root / "hits.json").read_text())
        if not 1 <= rank <= len(rows):
            raise HTTPException(404, "Hit not found")
        row = rows[rank - 1]
        db = settings.databases[j["request"]["database"]]
        from .chemistry import molblock_at, payload, thumbnail
        from rdkit import Chem
        manifest = json.loads((root / "manifest.json").read_text())
        # Corpus-local gids never cross a rebuild. Result rows carry the
        # canonical (mol_id, conf_idx) identity used to recover this pose.
        from roshambo3 import open_corpus
        from roshambo3.format import Catalog
        corpus = open_corpus(str(db.path))
        if manifest["corpus_sha256"] != corpus.fingerprint:
            raise ValueError(
                "This database was changed after the screen; restore the original "
                "corpus to view its overlays")
        catalog = Catalog(corpus)
        gid = catalog.gid_for(int(row["mol_id"]), int(row["conf_idx"]))
        metadata = catalog.molecules([gid], columns=["smiles"])
        smiles = metadata["smiles"][0]
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None:
            raise ValueError("The format-3 hit has no parseable canonical SMILES")
        with np.load(root / "poses.npz") as poses:
            offsets = poses["offsets"]
            xyz = poses["xyz"][offsets[rank - 1]:offsets[rank]]
        queries = list(Chem.SDMolSupplier(str(root / "query.sdf"), removeHs=False))
        return {"hit": molblock_at(mol, xyz, preserve_atoms=True), "query": payload(queries[row["query_index"]]),
                "thumbnail": thumbnail(mol), "row": row}

    @app.get("/api/jobs/{ident}/hits/{rank}/overlay")
    def overlay(ident: str, rank: int):
        complete(ident)
        try:
            # Validate the corpus on every request, including SDF export.
            # Caching a serialized pose must not bypass mutable-file validation.
            return load_overlay(ident, rank)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            raise HTTPException(422, str(exc))

    @app.get("/api/jobs/{ident}/hits/{rank}/thumbnail")
    def hit_thumbnail(ident: str, rank: int):
        complete(ident)
        rows = json.loads((directory(ident) / "hits.json").read_text())
        if not 1 <= rank <= len(rows):
            raise HTTPException(404, "Hit not found")
        from .chemistry import thumbnail
        from rdkit import Chem
        mol = Chem.MolFromSmiles(rows[rank - 1]["smiles"] or "")
        if mol is None:
            raise HTTPException(422, "No molecular structure available")
        return Response(thumbnail(mol), media_type="image/svg+xml")

    @app.get("/api/jobs/{ident}/export/{kind}")
    def export(ident: str, kind: str, ranks: str = ""):
        complete(ident)
        root = directory(ident)
        if kind == "manifest":
            return FileResponse(root / "manifest.json", filename="screen-manifest.json")
        if kind == "csv":
            rows = json.loads((root / "hits.json").read_text())
            out = io.StringIO()
            writer = csv.DictWriter(
                out, fieldnames=list(rows[0]) if rows else ["rank", "mol_id", "conf_idx"])
            writer.writeheader()
            # Spreadsheet exports must not treat molecule names as formulas.
            from roshambo3.results import spreadsheet_safe_text
            writer.writerows({k: spreadsheet_safe_text(v) for k, v in row.items()}
                             for row in rows)
            return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=hits.csv"})
        if kind == "sdf":
            try:
                selected = [int(r) for r in ranks.split(",") if r]
            except ValueError:
                raise HTTPException(422, "Invalid selected ranks")
            if not 1 <= len(selected) <= 100:
                raise HTTPException(422, "Select between 1 and 100 hits")
            records = []
            for rank in selected:
                item = overlay(ident, rank)
                for key in ("query", "hit"):
                    block = item[key]
                    block = f"{key}-rank-{rank}\n" + block.split("\n", 1)[1]
                    props = {"record_role": key, "query_conformer": item["row"]["query_index"] + 1,
                             **{k: item["row"][k] for k in (
                                 "rank", "gid", "mol_id", "conf_idx", "shape", "color",
                                 "combination") if k in item["row"]}}
                    records.append(block + "\n" + "".join(f"> <{k}>\n{v}\n\n" for k, v in props.items()) + "$$$$\n")
            return Response("".join(records), media_type="chemical/x-mdl-sdfile",
                            headers={"Content-Disposition": "attachment; filename=overlays.sdf"})
        raise HTTPException(404, "Unknown export")

    static = Path(__file__).parent / "static"
    if (static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    else:
        @app.get("/")
        def missing_frontend():
            raise HTTPException(503, "Build the frontend with npm ci && npm run build in web/")
    return app


def serve(args):
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError("Install the optional web dependencies: pip install 'roshambo3[web]'")
    from .config import load_config
    uvicorn.run(create_app(load_config(args.config)), host=args.host, port=args.port, workers=1)
    return 0

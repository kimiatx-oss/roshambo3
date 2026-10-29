"""Administrator-controlled database registry and resource limits."""
from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Database:
    id: str
    name: str
    path: Path
    description: str = ""

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
    def public(self):
        color_error = None
        from roshambo3.format.r3g.reader import open_corpus
        corpus = open_corpus(str(self.path), purpose="inspect")
        color = corpus.color
        molecules, conformers = corpus.n_mol, corpus.n_conf
        overlays = (self.path / "molecules.parquet").is_file()
        modes = ["shape"]
        if color:
            try:
                # This checks trust before it builds a shipped factory or
                # enables the isolated custom-query adapter.
                from roshambo3.format.r3g.descriptor import isolated_generator_from_descriptor
                from roshambo3.definitions import check_corpus_color
                generator = isolated_generator_from_descriptor(color)
                check_corpus_color(corpus, generator)
                modes = ["shape", "rescore", "combo"]
            except (ValueError, OSError) as exc:
                color_error = str(exc)
        return {"id": self.id, "name": self.name, "description": self.description,
                "molecules": molecules, "conformers": conformers,
                "modes": modes, "color_error": color_error,
                "overlays": overlays, "overlay_error": None}


@dataclass(frozen=True)
class Settings:
    databases: dict[str, Database]
    jobs: Path
    device: int = 0
    max_k: int = 10000
    max_conformers: int = 50
    upload_bytes: int = 10 * 1024 * 1024
    retention_days: int = 7
    job_timeout_s: int = 86400
    cancel_timeout_s: int = 15
    backend: str = "cuda"
    cpu_workers: int | None = None
    cpu_qos: str = "inherit"
    cpu_memory_budget_bytes: int | None = None
    cpu_topk_state_budget_bytes: int | None = None
    metal_field_sampler: str | None = None


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as f:
        data = tomllib.load(f)
    def resolve(value):
        p = Path(value).expanduser()
        return (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
    server = data.get("server", {})
    backend = server.get("backend", "cuda")
    if backend not in ("cuda", "metal", "cpu"):
        raise ValueError("server.backend must be cuda, metal, or cpu")
    sampler = server.get("metal_field_sampler")
    if sampler not in (None, "manual", "texture16", "texture32"):
        raise ValueError("invalid server.metal_field_sampler")
    if sampler not in (None, "manual") and backend != "metal":
        raise ValueError("server.metal_field_sampler requires backend=metal")
    cpu_values = {}
    for key in ("cpu_workers", "cpu_memory_budget_bytes", "cpu_topk_state_budget_bytes"):
        if key in server:
            value = server[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"server.{key} must be a positive integer")
            cpu_values[key] = value
    qos = server.get("cpu_qos", "inherit")
    if qos not in ("inherit", "background", "user_initiated"):
        raise ValueError("invalid server.cpu_qos")
    databases = {}
    for item in data.get("databases", []):
        ident = item["id"]
        if not ident or ident in databases:
            raise ValueError("database IDs must be unique and nonempty")
        db = Database(ident, item["name"], resolve(item["path"]),
                      item.get("description", ""))
        if not (db.path / "corpus.json").is_file():
            raise ValueError(f"database {ident}: missing corpus.json")
        databases[ident] = db
    if not databases:
        raise ValueError("configure at least one database")
    values = {k: int(server[k]) for k in ("device", "max_k", "max_conformers", "upload_bytes",
              "retention_days", "job_timeout_s", "cancel_timeout_s") if k in server}
    if any(v < (0 if k == "device" else 1) for k, v in values.items()):
        raise ValueError("server resource limits must be positive; device must be nonnegative")
    if backend == "cpu" and values.get("device", 0) != 0:
        raise ValueError("CPU backend requires device=0")
    return Settings(databases, resolve(server.get("jobs", "jobs")),
                    backend=backend, metal_field_sampler=sampler, cpu_qos=qos, **cpu_values, **values)

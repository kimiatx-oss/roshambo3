"""Persistent FIFO job state. Each operation owns and closes its connection."""
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time
import uuid

TERMINAL = {"completed", "failed", "canceled"}


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, "
                        "created REAL, updated REAL, state TEXT, payload TEXT)")

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.root / "jobs.sqlite", timeout=30)
        try:
            with con:
                yield con
        finally:
            con.close()

    def create(self, request):
        ident = uuid.uuid4().hex
        now = time.time()
        directory = self.root / ident
        directory.mkdir()
        payload = {"request": request, "progress": 0, "submitted_conformers": 0}
        with self.connection() as con:
            con.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
                        (ident, now, now, "draft", json.dumps(payload)))
        return self.get(ident)

    def get(self, ident):
        with self.connection() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        return {"id": row[0], "created": row[1], "updated": row[2], "state": row[3],
                **json.loads(row[4])}

    def update(self, ident, state=None, expected=None, **changes):
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT state,payload FROM jobs WHERE id=?", (ident,)).fetchone()
            if row is None:
                raise KeyError(ident)
            if expected is not None and row[0] != expected:
                return False
            payload = json.loads(row[1])
            payload.update(changes)
            if state in TERMINAL:
                payload["eta_seconds"] = None
            con.execute("UPDATE jobs SET state=?,updated=?,payload=? WHERE id=?",
                        (state or row[0], time.time(), json.dumps(payload), ident))
        return True

    def list(self, limit=100):
        with self.connection() as con:
            rows = con.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "created": r[1], "updated": r[2], "state": r[3],
                 **json.loads(r[4])} for r in rows]

    def next(self):
        with self.connection() as con:
            row = con.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created,id LIMIT 1").fetchone()
        return row[0] if row else None

    def recover(self):
        with self.connection() as con:
            ids = [r[0] for r in con.execute("SELECT id FROM jobs WHERE state NOT IN "
                    "('ready','queued','completed','failed','canceled')")]
        for ident in ids:
            self.update(ident, "failed", error="Server restarted during this job. Submit a new screen.")

    def expire(self, days):
        import shutil
        cutoff = time.time() - days * 86400
        with self.connection() as con:
            ids = [r[0] for r in con.execute("SELECT id FROM jobs WHERE updated < ? AND state IN "
                    "('draft','ready','completed','failed','canceled')", (cutoff,))]
            for ident in ids:
                shutil.rmtree(self.root / ident, ignore_errors=True)
                con.execute("DELETE FROM jobs WHERE id=?", (ident,))

"""Single-server supervisor: FIFO screens, isolated children, restart recovery."""
import fcntl
import multiprocessing as mp
import time
import threading
from .store import Store, TERMINAL


def child(settings, ident, action):
    store = Store(settings.jobs)
    try:
        if action == "prepare":
            from .chemistry import prepare_query
            store.update(ident, "preparing")
            details = prepare_query(settings.jobs / ident, store.get(ident)["request"], settings.max_conformers)
            if (settings.jobs / ident / "cancel").exists():
                raise InterruptedError("Canceled")
            store.update(ident, "ready", preparation=details)
        else:
            from .screening import run_screen
            run_screen(settings, ident)
    except InterruptedError:
        store.update(ident, "canceled")
    except Exception as exc:
        store.update(ident, "failed", error=str(exc))


class Supervisor:
    def __init__(self, settings):
        self.settings = settings
        self.store = Store(settings.jobs)
        self.stop = threading.Event()
        self.lock = None
        self.thread = None
        self.children = {}
        self.mutex = threading.Lock()

    def start(self):
        self.lock = (self.settings.jobs / "server.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("A web server already owns this job directory. Use one API process.")
        self.store.recover()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def launch(self, ident, action):
        with self.mutex:
            if action == "prepare" and sum(v[2] == "prepare" for v in self.children.values()) >= 2:
                raise ValueError("Two queries are already being prepared; try again shortly")
            proc = mp.get_context("spawn").Process(target=child, args=(self.settings, ident, action))
            try:
                proc.start()
            except BaseException:
                proc.close()
                raise
            self.children[ident] = (proc, time.monotonic(), action)

    def loop(self):
        last_cleanup = 0
        while not self.stop.wait(0.2):
            with self.mutex:
                for ident, (proc, started, action) in list(self.children.items()):
                    cancel = (self.settings.jobs / ident / "cancel").exists()
                    expired = time.monotonic() - started > (120 if action == "prepare" else self.settings.job_timeout_s)
                    if (cancel and time.time() - (self.settings.jobs / ident / "cancel").stat().st_mtime > self.settings.cancel_timeout_s) or expired:
                        proc.terminate()
                        proc.join(timeout=3)
                        if proc.is_alive():
                            proc.kill()
                            proc.join()
                        self.store.update(ident, "canceled" if cancel else "failed", error="Worker exceeded its time limit")
                    if not proc.is_alive():
                        proc.join()
                        proc.close()
                        if cancel:
                            self.store.update(ident, "canceled", expected="ready")
                        if self.store.get(ident)["state"] not in TERMINAL | {"ready"}:
                            self.store.update(ident, "canceled" if cancel else "failed", error="Worker stopped before completing the job")
                        del self.children[ident]
                busy = any(v[2] == "screen" for v in self.children.values())
            if not busy:
                ident = self.store.next()
                if ident:
                    if self.store.update(ident, "starting", expected="queued"):
                        try:
                            self.launch(ident, "screen")
                        except Exception as exc:
                            self.store.update(ident, "failed", error=f"Could not start worker: {exc}")
            if time.monotonic() - last_cleanup > 60:
                self.store.expire(self.settings.retention_days)
                last_cleanup = time.monotonic()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        with self.mutex:
            for proc, _, _ in self.children.values():
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                proc.close()
            self.children.clear()
        if self.lock:
            self.lock.close()

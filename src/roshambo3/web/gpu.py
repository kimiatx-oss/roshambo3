"""Installed-package GPU lock matching the repository benchmark protocol."""
from contextlib import contextmanager
import csv
import fcntl
import io
import os
import subprocess
import time
from roshambo3.paths import gpu_lock_path


@contextmanager
def resource_lock(backend, device, canceled):
    if backend == "cuda":
        with gpu_lock(device, canceled) as identity:
            yield identity
        return
    if backend == "cpu":
        from roshambo3.paths import cpu_lock_path
        identity, path = "cpu", cpu_lock_path()
    elif backend == "metal":
        from roshambo3.backend.metal import extension
        info = extension().device_info(device)
        identity = f"metal-{info['registry_id']}"
        path = gpu_lock_path(identity)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        while True:
            if canceled():
                raise InterruptedError("Canceled while waiting for compute resources")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.2)
        try:
            f.seek(0)
            f.truncate()
            f.write(f"web pid={os.getpid()} backend={backend} identity={identity}\n")
            f.flush()
            yield identity
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def device_uuid(device):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=10)
    rows = list(csv.reader(io.StringIO(output)))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device < 0 or (visible is not None and device >= len(visible.split(","))):
        raise ValueError("Configured GPU is outside CUDA_VISIBLE_DEVICES")
    selected = visible.split(",")[device].strip() if visible is not None else str(device)
    if not selected or selected == "-1":
        raise ValueError("CUDA_VISIBLE_DEVICES disables the configured GPU")
    matches = [r[1].strip() for r in rows if r[0].strip() == selected or r[1].strip().startswith(selected)]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely resolve the configured visible GPU")
    return matches[0]


@contextmanager
def gpu_lock(device, canceled):
    uuid = device_uuid(device)
    path = gpu_lock_path(uuid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        while True:
            if canceled():
                raise InterruptedError("Canceled while waiting for the GPU")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.2)
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            f.seek(0)
            f.truncate()
            f.write(f"web pid={os.getpid()} uuid={uuid}\n")
            f.flush()
            # Pin by UUID before CUDA initialization: the worker's device 0
            # must be exactly the physical card whose lock we hold.
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            yield uuid
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
            fcntl.flock(f, fcntl.LOCK_UN)

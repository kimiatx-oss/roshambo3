"""The run-record schema: one JSON object per run, appended to a JSONL file.

Every quoted number should cite a ``run_id`` in this file.
The schema is deliberately wide because a downstream reader who wasn't there when the run happened
must be able to reconstruct *why* a number can or cannot be trusted, without
re-running anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .build_check import VerifiedBuild
from .gitinfo import GitInfo


def numerics_stamp() -> dict:
    """The numerics stamp, or an explicit marker that it could not be read.

    Kept behind a try/except because the harness must still be able to write a
    *refused* record on a machine where the GPU extension is absent -- but
    "we don't know what arithmetic this was" is itself recorded, rather than the
    field silently going missing.
    """
    try:
        from roshambo3.numerics import numerics_stamp as _stamp

        return _stamp()
    except Exception as e:                       # pragma: no cover
        return {"error": f"numerics stamp unavailable: {e}"}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-serializable config object, used for
    ``config_hash`` -- lets two run records be checked for "same config" by
    hash rather than deep-diffing dicts."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def padding_table_from_corpus(corpus) -> list:
    """The ``(bucket, L, L_color, n_conf)`` rows for a prepared corpus.

    Padding width, in the one form two runs can be compared on. It is
    derived from the format-3 group table alone -- no payload is read -- and it is
    canonicalized by :func:`bench.rules.canonical_padding_table`, so a corpus
    whose builder split a bucket into two groups compares equal to one that did
    not, while a corpus whose *widths* differ does not.

    ``L_color`` is 0 on a merged-color corpus, where there is one width rather
    than two; the slot count per conformer is then ``L`` alone. That the two
    layouts reach different slot counts for the same molecules is precisely
    what a throughput comparison between them must not hide.
    """
    from .rules import canonical_padding_table

    return canonical_padding_table(
        {"bucket": g["bucket"], "L": g["L"], "L_color": 0,
         "n_conf": g["n_conf"]}
        for g in corpus.group_table)


def host_info() -> dict:
    """What machine this ran on.

    Includes the CPU's identity, because ``cpu_count`` alone cannot
    distinguish 24 Zen 3 cores from 28 Cascade Lake ones -- measured at 1,824
    against 931 conformers/s/core, a 1.96x difference. The ISA flags matter
    for the same reason on the other axis: the Zen 3 host has no AVX-512.

    Best-effort. A record from a machine whose ``lscpu`` is unavailable keeps
    the four fields it always had rather than failing to be written.
    """
    out = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        from bench.cpu import cpu_identity
        out["cpu"] = cpu_identity().as_dict()
    except Exception as e:                                  # pragma: no cover
        out["cpu"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def gpu_info(build: VerifiedBuild, clocks_locked: bool) -> dict:
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,uuid,driver_version,power.limit",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        # CPU-only hosts must still be able to write their run ledger.
        # Missing GPU telemetry is explicit, never an invented GPU reading.
        return {
            "name": "unknown", "uuid": "unknown", "driver": "unknown",
            "clocks_locked": clocks_locked, "power_limit": "unknown",
            "telemetry_error": "nvidia-smi not installed",
        }
    name, uuid_, driver, power_limit = ("unknown",) * 4
    line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 4:
        name, uuid_, driver, power_limit = parts[:4]
    return {
        "name": name,
        "uuid": uuid_,
        "driver": driver,
        "clocks_locked": clocks_locked,
        "power_limit": power_limit,
    }


@dataclass
class RunRecord:
    run_id: str
    utc: str
    preregistration_id: str | None
    git: dict
    build: dict
    numerics: dict
    """The numerics stamp: the pinned CUDA toolkit, the nvcc flags actually
    compiled in, the reduced-frame constants, and whether the arithmetic
    invariant passed. A number that cannot say which arithmetic produced it is
    not reproducible, and "which arithmetic" is a specific short list
    -- see roshambo3.numerics and doc/NUMERICS_POLICY.md."""
    host: dict
    gpu: dict
    gpu_lock_held: bool
    dataset: dict
    query_panel: dict
    config: dict
    config_hash: str
    gold_ref_run_id: str | None
    timing: dict
    metrics: dict
    audits: dict
    artifacts: dict
    status: str
    """One of "decision", "advisory", "refused"."""
    advisory_reasons: list[str] = field(default_factory=list)
    refusal_reason: str | None = None
    notes: str = ""
    padding: dict = field(default_factory=dict)
    """The padded-width histogram of the library this run
    screened, and whatever else describes the slot count it paid for. Expected
    shape (all optional): ``{"table": [[bucket, L, L_color, n_conf], ...],
    "slots", "slot_efficiency", "merged_color", "corpus"}``.

    Carried because the harness cannot refuse a comparison it cannot see:
    ``bench/rules.py`` checked clock, architecture, regime and device and
    nothing about padding, so the one pitfall that is about the LIBRARY's shape
    was the one with no enforcement. Defaults to ``{}`` so every caller that
    predates the field keeps working, and ``{}`` means "not recorded" rather
    than "the padding matched" -- see ``check_padding_comparable``."""
    operating_point: dict = field(default_factory=dict)
    """The achieved (never requested) clock/power-cap/profiler-
    control state this run was measured under -- stamped into every run
    record. Expected shape (all optional, present
    when known):
    ``{"achieved_clock_mhz": {"min", "median", "max"}, "sw_power_cap_fraction",
    "power_limited", "clocks_locked_requested_succeeded", "profiler_control":
    {"state", "tool", "clock_control_setting"}}``. Defaults to ``{}`` so every
    existing caller of build_run_record() that predates this field keeps
    working unmodified."""

    def as_dict(self) -> dict:
        return asdict(self)


def new_run_id() -> str:
    return f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_jsonl(record: RunRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record.as_dict(), default=str) + "\n")


def build_run_record(
    *,
    ticket_id: str,
    git: GitInfo,
    build: VerifiedBuild,
    clocks_locked: bool,
    gpu_lock_held: bool,
    dataset: dict,
    query_panel: dict,
    config: dict,
    timing: dict,
    metrics: dict,
    audits: dict,
    artifacts: dict,
    status: str,
    advisory_reasons: list[str] | None = None,
    refusal_reason: str | None = None,
    preregistration_id: str | None = None,
    gold_ref_run_id: str | None = None,
    notes: str = "",
    operating_point: dict | None = None,
    padding: dict | None = None,
    run_id: str | None = None,
    utc: str | None = None,
) -> RunRecord:
    """Build a run record.

    ``run_id`` and ``utc`` default to "now", which is right for a run being
    recorded as it happens and wrong for a **backfill** -- a record whose
    timestamp is the moment it was reconstructed says the run happened then,
    which it did not. Pass both explicitly when reclassifying an existing
    artifact, with a provenance block saying what the timestamp actually
    denotes. A backfilled grade is a
    reclassification, not a measurement, and the record must not read as one.
    """
    return RunRecord(
        run_id=run_id if run_id is not None else new_run_id(),
        utc=utc if utc is not None else utc_now_iso(),
        preregistration_id=preregistration_id,
        git=git.as_dict(),
        build=build.as_dict(),
        numerics=numerics_stamp(),
        host=host_info(),
        gpu=gpu_info(build, clocks_locked),
        gpu_lock_held=gpu_lock_held,
        dataset=dataset,
        query_panel=query_panel,
        config=config,
        config_hash=sha256_json(config),
        gold_ref_run_id=gold_ref_run_id,
        timing=timing,
        metrics=metrics,
        audits=audits,
        artifacts=artifacts,
        status=status,
        advisory_reasons=advisory_reasons or [],
        refusal_reason=refusal_reason,
        notes=notes,
        operating_point=operating_point or {},
        padding=padding or {},
    )

"""The ``deep`` tier's RDKit items (``doc/FORMAT.md`` section 16.4 items 3-4),
run in a time- and memory-limited subprocess, never in the validator's process.

Section 16.4 puts every factory built from corpus text inside this sandboxed
"build factory and perceive" call. :func:`run_child` is its one
implementation: ``RLIMIT_AS``, ``RLIMIT_CPU``, a wall-clock kill of the whole
process group, JSON over pipes, an isolated interpreter that imports nothing
of roshambo3, no CUDA. ``roshambo3.definitions.perceive`` (query featurization)
and :func:`deep_rdkit_checks` both call it, and the limits apply to trusted
definitions too: trust removes the refusal, not the bounds.

What runs in the child, per molecule: ``MolFromSmiles`` with CXSMILES
disabled; the parsed heavy-atom count; and, for the sampled molecules of a
color corpus, ``AddHs`` then perception with the stored definition --
``BuildFeatureFactoryFromString`` for ``rdkit_fdef``, and for ``smarts_list``
the matching ``CustomSMARTSPharmacophoreGenerator.assign_features`` does
(``GetSubstructMatches`` per pattern, identical matches within a family
kept once; a pattern that does not parse refuses the definition, as the
builder does) -- returning only integers.
"""

from __future__ import annotations

import functools
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import resource
except ImportError:                       # not POSIX
    resource = None  # type: ignore

import numpy as np

from .errors import CorpusValidationError

#: Child limits (defaults): address space, CPU seconds, wall seconds.
RLIMIT_AS_BYTES = 2 << 30
RLIMIT_CPU_SECONDS = 30
WALL_SECONDS = 60.0
#: Molecules per child process. Small on purpose: it bounds the child's reply and
#: keeps the per-batch budget below proportionate to the work asked of it. A
#: ``deep`` batch re-perceives at most this many molecules.
BATCH = 250
#: Molecules per ``deep`` child when at most :data:`BATCH` of them are
#: re-perceived and the rest are only parsed. A child costs ~72 ms to
#: start and ~0.1 ms per parse, so at 250 the start-up was ~80% of the work.
#: The budgets still scale per batch (:class:`Limits`). A batch is whole
#: :data:`BATCH`-row pieces of the serial walk's grid, and one that fails as a
#: whole is replayed piece by piece, so the refusal reported is the one
#: 250-molecule batches would give. Should be a multiple of :data:`BATCH`.
PARSE_BATCH = 2000
#: Reply bytes allowed per molecule that is parsed but not re-perceived
#: (``[true, 255, null, 1234]`` and separators).
PARSE_REPLY_BYTES = 48
#: Most sandbox children ``deep`` runs at once (``--workers``).
MAX_DEFAULT_WORKERS = 16
#: Extra budget per molecule in the batch, on top of the base limits: shipped
#: BaseFeatures costs ~15 ms per drug-sized molecule with hydrogens, so a fixed
#: budget would fail a legitimate full batch of larger molecules.
CPU_PER_MOLECULE = 0.05
WALL_PER_MOLECULE = 0.1
EXIT_MEMORY = 97
EXIT_FEATURES = 98
EXIT_DEFINITION = 96
STDERR_KEEP = 64 << 10


@dataclass(frozen=True)
class Limits:
    """The child limits; any may be lowered (tests) or raised. The CPU and wall
    budgets of a batch are ``base + per_molecule * n_molecules`` (set the per-
    molecule terms to 0 for a fixed budget); memory is not scaled."""
    memory_bytes: int = RLIMIT_AS_BYTES
    cpu_seconds: float = RLIMIT_CPU_SECONDS
    wall_seconds: float = WALL_SECONDS
    cpu_per_molecule: float = CPU_PER_MOLECULE
    wall_per_molecule: float = WALL_PER_MOLECULE


_CHILD = r"""
import json, resource, sys
if sys.argv[1]:
    sys.path.append(sys.argv[1])          # where the parent found RDKit (python -I drops PYTHONPATH)
def lim(which, v, hard_v=None):
    try:
        soft, hard = resource.getrlimit(which)
        if hard != resource.RLIM_INFINITY and hard < v:
            v = hard
        h = hard
        if hard_v is not None and (hard == resource.RLIM_INFINITY or hard_v <= hard):
            h = hard_v
        resource.setrlimit(which, (v, h))
        return True
    except (ValueError, OSError):
        return False
req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
lim(resource.RLIMIT_AS, req["rlimit_as"])
# SIGXCPU at the soft limit, SIGKILL one second later at the hard one.
lim(resource.RLIMIT_CPU, req["rlimit_cpu"], req["rlimit_cpu"] + 1)
MAXF, MAXT, MAXA = req["max_features"], req["max_per_type"], req["max_atoms"]
def too_many(feats):
    if len(feats) > MAXF:
        return "a molecule has more than %d features" % MAXF
    per = {}
    for t, atoms in feats:
        per[t] = per.get(t, 0) + 1
        if per[t] > MAXT:
            return "a molecule has more than %d features of one type" % MAXT
        if len(atoms) > MAXA:
            return "a feature has more than %d atoms" % MAXA
    return None
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    types = req["types"]
    tid = {t: i + 1 for i, t in enumerate(types)}
    factory = None
    smarts = None
    if req["perceive_any"]:
        if req["generator"] == "rdkit_fdef":
            from rdkit.Chem import ChemicalFeatures
            factory = ChemicalFeatures.BuildFeatureFactoryFromString(req["definition"])
        else:
            smarts = {fam: [Chem.MolFromSmarts(p) for p in pats]
                      for fam, pats in req["definition"].items() if fam in tid}
            for fam, pats in req["definition"].items():
                for p, q in zip(pats, smarts.get(fam, ())):
                    if q is None:
                        # Refused, never skipped: the builder refuses the same
                        # definition, so build and validator agree.
                        sys.stderr.write("family %r: SMARTS pattern %r does not parse" % (fam, p))
                        sys.exit(@EXIT_DEFINITION@)
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = False
    molblock = req.get("input") == "molblock"
    out = []
    for text, perceive in req["molecules"]:
        if molblock:
            mol = Chem.MolFromMolBlock(text, sanitize=True, removeHs=False)
        else:
            mol = Chem.MolFromSmiles(text, params)
        if mol is None:
            out.append([False, 0, None, 0])
            continue
        n = (sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1) if molblock
             else mol.GetNumAtoms())
        feats = None
        nh = mol.GetNumAtoms()
        if perceive:
            mh = Chem.AddHs(mol)           # keeps the input's atom order; new hydrogens are appended
            nh = mh.GetNumAtoms()
            feats = []
            if factory is not None:
                for f in factory.GetFeaturesForMol(mh):
                    fam = f.GetFamily()
                    if fam in tid:
                        feats.append([tid[fam], [int(a) for a in f.GetAtomIds()]])
            else:
                for fam, pats in smarts.items():
                    seen = []
                    for p in pats:
                        for m in mh.GetSubstructMatches(p):
                            if m not in seen:
                                seen.append(m)
                    for m in seen:
                        feats.append([tid[fam], [int(a) for a in m]])
                    if len(feats) > MAXF:
                        break
            why = too_many(feats)
            if why:
                sys.stderr.write("feature limit: " + why)
                sys.exit(@EXIT_FEATURES@)
        out.append([True, n, feats, nh])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sys.stdout.write(json.dumps({"results": out, "maxrss": rss}))
except MemoryError:
    sys.stderr.write("MemoryError")
    sys.exit(@EXIT_MEMORY@)
""".replace("@EXIT_DEFINITION@", str(EXIT_DEFINITION)).replace("@EXIT_FEATURES@", str(EXIT_FEATURES)).replace("@EXIT_MEMORY@", str(EXIT_MEMORY))


class SandboxFailure(RuntimeError):
    """The child was killed, ran out of a limit, or answered garbage.

    ``limit`` names the bound that was hit (``"cpu"``, ``"wall"``, ``"memory"``,
    ``"features"``, ``"reply"``) or is ``None`` for any other failure;
    ``peak_rss_bytes`` is that child's own peak resident set (measured by
    ``wait4`` on every outcome, None if unavailable); ``seconds`` is the wall
    time it ran."""

    def __init__(self, message: str, *, limit: Optional[str] = None,
                 peak_rss_bytes: Optional[int] = None, seconds: float = 0.0):
        super().__init__(message)
        self.limit = limit
        self.peak_rss_bytes = peak_rss_bytes
        self.seconds = seconds


#: Peak resident set (bytes) of the most recent child, or None. Informational
#: (timing tables); not thread-safe.
last_peak_rss_bytes: Optional[int] = None


def _rss_scale() -> int:
    return 1 if sys.platform == "darwin" else 1024        # ru_maxrss: bytes vs KiB


def _child_rss_bytes(pid: int) -> Optional[int]:
    """Current RSS of ``pid`` through ``ps`` (the parent-side memory watch used
    where ``RLIMIT_AS`` is not enforced, i.e. macOS)."""
    try:
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                           timeout=2)
        return int(r.stdout.split()[0]) * 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _kill_group(proc: "subprocess.Popen") -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)       # child leads its own session
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


@functools.lru_cache(maxsize=None)
def _rdkit_site_dir() -> str:
    """The directory holding the ``rdkit`` package, as the parent resolved it. The
    child runs ``python -I`` (no PYTHONPATH, no cwd), so it is passed explicitly."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("rdkit")
        if spec and spec.submodule_search_locations:
            return os.path.dirname(list(spec.submodule_search_locations)[0])
    except (ImportError, ValueError):
        pass
    return ""


def default_workers() -> int:
    """``min(16, CPUs this process may run on)``: concurrent sandbox children
    in ``deep`` (``os.sched_getaffinity`` where available, else
    ``os.cpu_count()``)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(MAX_DEFAULT_WORKERS, n))


def max_reply_bytes(n_molecules: int) -> int:
    """Upper bound on a valid reply: every molecule at the feature limits, each
    integer at most 7 characters plus separator."""
    from .reference import LIMITS
    per_feature = LIMITS["atoms_per_feature"] * 8 + 16
    return n_molecules * (LIMITS["features_per_molecule"] * per_feature + 48) + 4096


def run_child(request: dict, *, timeout: Optional[float] = None,
              limits: Optional[Limits] = None,
              cancel: Optional[threading.Event] = None) -> list:
    """Run one batch in the limited child; return its per-molecule results
    ``[parsed, n_heavy, features | None, n_with_hydrogens]``.

    For a request that perceives (``perceive_any``) the RDKit-free pre-pass
    (``reference.check_definition_text``) runs **here first**, so this one place
    cannot be reached with unchecked text (``CorpusValidationError`` propagates).

    Bounds: ``RLIMIT_AS`` and ``RLIMIT_CPU`` set in the child before RDKit is
    imported, and a wall-clock deadline enforced here by killing the child's whole
    process group (CPU and wall budgets scale with the batch, see
    :class:`Limits`). On macOS ``RLIMIT_AS`` is not enforced, so the parent polls
    the child's RSS instead. The child is always reaped. The child stops with an
    error when a molecule has more than 255 features (in total or of one type) or
    a feature more than 16 atoms (FORMAT.md section 14), and the parent reads the
    reply incrementally under a size cap derived from those limits, killing the
    child if it is exceeded. The reply is then validated like a corpus table.
    ``request["input"]`` is ``"smiles"`` (default) or ``"molblock"``; ``timeout``
    overrides the whole wall budget. Setting ``cancel`` kills the child and
    raises :class:`SandboxFailure` with ``limit="cancelled"`` (a parallel
    ``deep`` that has already found its refusal stops its other children).

    The reply cap counts a molecule that is not re-perceived (``[text, False]``,
    or any molecule of a request without ``perceive_any``) at
    :data:`PARSE_REPLY_BYTES`, since its features are ``None``."""
    global last_peak_rss_bytes
    from .reference import LIMITS, check_definition_text
    lim = limits or Limits()
    mols = request["molecules"]
    n = len(mols)
    if request.get("perceive_any") and request.get("generator"):
        check_definition_text(request["definition"], request["generator"], request.get("types"))
    cpu_budget = float(lim.cpu_seconds) + float(lim.cpu_per_molecule) * n
    wall = float(timeout) if timeout else float(lim.wall_seconds) + float(lim.wall_per_molecule) * n
    req = dict(request)
    req["rlimit_as"] = int(lim.memory_bytes)
    req["rlimit_cpu"] = max(1, int(math.ceil(cpu_budget)))
    req["max_features"] = LIMITS["features_per_molecule"]
    req["max_per_type"] = LIMITS["features_per_type"]
    req["max_atoms"] = LIMITS["atoms_per_feature"]
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "LC_ALL",
                                                         "SYSTEMROOT", "TMPDIR")}
    env["CUDA_VISIBLE_DEVICES"] = ""          # no GPU context in the child
    n_perceive = (sum(1 for m in mols if m[1]) if request.get("perceive_any") else 0)
    reply_cap = max_reply_bytes(n_perceive) + PARSE_REPLY_BYTES * (n - n_perceive)
    start = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-I", "-c", _CHILD, _rdkit_site_dir()],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env, cwd=os.path.abspath(os.sep),
                            start_new_session=(os.name == "posix"))
    payload = json.dumps(req).encode("utf-8")
    out_buf = bytearray()
    err_buf = bytearray()
    over = threading.Event()

    def pump_in():
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (OSError, ValueError):
            pass

    def pump_out():
        try:
            while True:
                chunk = proc.stdout.read1(1 << 16)
                if not chunk:
                    return
                out_buf.extend(chunk)
                if len(out_buf) > reply_cap:
                    over.set()
                    return
        except (OSError, ValueError):
            pass

    def pump_err():
        try:
            while True:
                chunk = proc.stderr.read1(1 << 14)
                if not chunk:
                    return
                err_buf.extend(chunk)
                if len(err_buf) > 2 * STDERR_KEEP:           # keep the tail, bounded
                    del err_buf[:len(err_buf) - STDERR_KEEP]
        except (OSError, ValueError):
            pass

    threads = [threading.Thread(target=f, daemon=True) for f in (pump_in, pump_out, pump_err)]
    for t in threads:
        t.start()
    killed: Optional[str] = None
    status = None
    rusage = None
    peak_seen = 0
    watch_rss = sys.platform == "darwin"
    last_watch = 0.0
    try:
        while True:
            if os.name == "posix":
                pid_, st, ru = os.wait4(proc.pid, os.WNOHANG)
                if pid_:
                    status, rusage = st, ru
                    break
            elif proc.poll() is not None:
                break
            if over.is_set():
                killed = "reply"
                break
            if cancel is not None and cancel.is_set():
                killed = "cancelled"
                break
            now = time.monotonic()
            if now - start > wall:
                killed = "wall"
                break
            if watch_rss and now - last_watch > 0.1:
                last_watch = now
                rss = _child_rss_bytes(proc.pid)
                if rss:
                    peak_seen = max(peak_seen, rss)
                    if rss > lim.memory_bytes:
                        killed = "memory"
                        break
            over.wait(0.01)
    finally:
        if status is None:                       # not yet reaped: kill, then reap
            _kill_group(proc)
            try:
                if os.name == "posix":
                    _, status, rusage = os.wait4(proc.pid, 0)
                else:
                    proc.wait()
            except ChildProcessError:
                pass
        if os.name == "posix":
            if status is not None:
                proc.returncode = os.waitstatus_to_exitcode(status)
            try:
                os.killpg(proc.pid, signal.SIGKILL)       # nothing the child spawned lives on
            except (ProcessLookupError, PermissionError, OSError):
                pass
        for t in threads:
            t.join(timeout=5)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except OSError:
                pass
    seconds = time.monotonic() - start
    peak = peak_seen or None
    if rusage is not None:
        peak = int(rusage.ru_maxrss) * _rss_scale()
    last_peak_rss_bytes = peak
    if killed == "cancelled":
        raise SandboxFailure("cancelled", limit="cancelled", peak_rss_bytes=peak,
                             seconds=seconds)
    if killed == "wall":
        raise SandboxFailure(f"killed after the {wall:g} s wall-clock limit", limit="wall",
                             peak_rss_bytes=peak, seconds=seconds)
    if killed == "memory":
        raise SandboxFailure(f"killed above the {lim.memory_bytes >> 20} MiB memory limit",
                             limit="memory", peak_rss_bytes=peak, seconds=seconds)
    if killed == "reply" or over.is_set():
        raise SandboxFailure(f"killed: the reply exceeded {reply_cap} bytes", limit="reply",
                             peak_rss_bytes=peak, seconds=seconds)
    rc = proc.returncode if proc.returncode is not None else -1
    if rc != 0:
        tail = err_buf.decode("utf-8", "replace").strip().splitlines()[-1:] or ["?"]
        tail = _printable(tail[0])[:200]
        if rc == -signal.SIGXCPU or (rc == -signal.SIGKILL
                                     and seconds >= cpu_budget * 0.9):
            raise SandboxFailure(f"killed at the {cpu_budget:g} s CPU limit after "
                                 f"{seconds:.1f} s", limit="cpu", peak_rss_bytes=peak,
                                 seconds=seconds)
        if rc == EXIT_DEFINITION:
            raise SandboxFailure(f"definition refused: {tail}", peak_rss_bytes=peak,
                                 seconds=seconds)
        if rc == EXIT_FEATURES:
            raise SandboxFailure(f"stopped: {tail}", limit="features", peak_rss_bytes=peak,
                                 seconds=seconds)
        if rc == EXIT_MEMORY or (rc in (-signal.SIGABRT, -signal.SIGSEGV)
                                 and peak and peak >= 0.5 * lim.memory_bytes):
            raise SandboxFailure(f"ran out of memory at the {lim.memory_bytes >> 20} MiB limit",
                                 limit="memory", peak_rss_bytes=peak, seconds=seconds)
        raise SandboxFailure(f"exited {rc}: {tail}", peak_rss_bytes=peak, seconds=seconds)
    try:
        reply = json.loads(out_buf.decode("utf-8"))
        out = reply["results"]
    except (ValueError, KeyError, TypeError):
        raise SandboxFailure("answered something that is not JSON", seconds=seconds) from None
    return _validate_reply(out, request, seconds)


def _printable(text: str) -> str:
    """Escape control characters: RDKit echoes raw bytes of what it was given."""
    return "".join(c if c.isprintable() else f"\\x{ord(c):02x}" for c in text)


def _validate_reply(out, request: dict, seconds: float) -> list:
    """Integers only, shaped like a corpus feature table; anything else is a failure."""
    from .reference import LIMITS

    def bad(why):
        raise SandboxFailure(f"answered malformed results ({why})", seconds=seconds)

    def is_int(x):
        return isinstance(x, int) and not isinstance(x, bool)

    if not isinstance(out, list) or len(out) != len(request["molecules"]):
        raise SandboxFailure("answered the wrong number of results", seconds=seconds)
    n_types = len(request.get("types") or [])
    res = []
    for item in out:
        if not (isinstance(item, list) and len(item) == 4):
            bad("record shape")
        ok, n, feats, nh = item
        if not (isinstance(ok, bool) and is_int(n) and is_int(nh) and 0 <= n <= nh):
            bad("counts")
        if feats is not None:
            if not isinstance(feats, list) or len(feats) > LIMITS["features_per_molecule"]:
                bad("features")
            per: dict = {}
            for f in feats:
                if not (isinstance(f, list) and len(f) == 2 and is_int(f[0])
                        and 1 <= f[0] <= n_types and isinstance(f[1], list) and f[1]
                        and len(f[1]) <= LIMITS["atoms_per_feature"]
                        and all(is_int(a) and 0 <= a < nh for a in f[1])):
                    bad("feature type, size or atom index out of range")
                per[f[0]] = per.get(f[0], 0) + 1
                if per[f[0]] > LIMITS["features_per_type"]:
                    bad("too many features of one type")
        res.append([ok, n, feats, nh])
    return res


def deep_rdkit_checks(parquet_file, *, color: Optional[dict], ids: np.ndarray,
                      n_real: np.ndarray, stored_features: dict,
                      timeout: Optional[float] = None, limits: Optional[Limits] = None,
                      workers: Optional[int] = None) -> None:
    """Items 3-4: every SMILES parses with CXSMILES off to exactly ``n_real``
    heavy atoms; the sampled molecules' features (``stored_features``,
    ``{mol_id: (group path, [(type, atoms)])}``) equal the re-perceived, canonicalized
    table. ``ids``/``n_real`` are aligned with the parquet's row order.

    The rows are cut into consecutive batches of whole :data:`BATCH`-row
    pieces of the serial walk (``[250k, 250k + 250)``) -- at most
    :data:`PARSE_BATCH` molecules, of which at most :data:`BATCH`
    re-perceived -- and the batches
    run in up to ``workers`` sandbox children at once (default
    :func:`default_workers`). The threads here only feed and wait on the
    children; RDKit and every factory built from corpus text run in a child.
    Results are checked **in row order**, so the refusal is the one a serial
    walk reports first: the lowest row, then the check order. A batch of more
    than :data:`BATCH` molecules whose child fails as a whole is re-run in
    :data:`BATCH`-molecule pieces before it is reported, so a whole-batch
    failure is attributed as 250-molecule batches would attribute it."""
    from .parquet import read_column_strings
    from .reference import canonical_feature_order
    n_workers = default_workers() if workers is None else int(workers)
    if n_workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers!r}")
    path = parquet_file.path
    perceive_ids = np.fromiter(stored_features, dtype=np.int64, count=len(stored_features))
    perceive_ids.sort()
    base_req = {"perceive_any": False, "generator": None, "definition": None, "types": []}
    if color is not None and perceive_ids.size:
        base_req.update(generator=color["generator"], definition=color["definition"],
                        types=list(color["types"]))

    def request(smiles, perceive):
        req = dict(base_req)
        req["perceive_any"] = bool(any(perceive)) and color is not None
        req["molecules"] = [[s, bool(p)] for s, p in zip(smiles, perceive)]
        return req

    def run(batch, cancel=None):
        """The child's results, or the SandboxFailure it ended with."""
        _row0, smiles, perceive = batch
        try:
            return run_child(request(smiles, perceive), timeout=timeout, limits=limits,
                             cancel=cancel)
        except SandboxFailure as e:
            return e

    def check(batch, out) -> None:
        row0, smiles, perceive = batch
        if isinstance(out, SandboxFailure):
            if len(smiles) > BATCH:
                for k in range(0, len(smiles), BATCH):
                    piece = (row0 + k, smiles[k:k + BATCH], perceive[k:k + BATCH])
                    check(piece, run(piece))
                return
            pa = bool(any(perceive)) and color is not None
            code = "value.definition" if pa else "limit.smiles"
            raise CorpusValidationError(code, f"deep: RDKit {'perception' if pa else 'SMILES parsing'} "
                                        f"in the limited subprocess failed ({out})", tier="deep",
                                        path=path) from None
        for j, (ok, n, feats, _nh) in enumerate(out):
            row = row0 + j
            mid = int(ids[row])
            if not ok:
                raise CorpusValidationError("limit.smiles", f"row {row} (mol_id {mid}): the "
                                            f"smiles does not parse with CXSMILES disabled",
                                            tier="deep", path=path)
            if int(n) != int(n_real[row]):
                raise CorpusValidationError("limit.smiles", f"row {row} (mol_id {mid}): the "
                                            f"smiles has {n} heavy atoms, the run table says "
                                            f"n_real {int(n_real[row])}", tier="deep", path=path)
            if feats is not None:
                if any(a >= n for _, atoms in feats for a in atoms):
                    raise CorpusValidationError("layout.features", f"mol_id {mid}: re-perception "
                                                f"yields a feature on a hydrogen", tier="deep",
                                                path=stored_features[mid][0])
                gpath, stored = stored_features[mid]
                got = canonical_feature_order([(t, tuple(a)) for t, a in feats])
                want = [(t, tuple(a)) for t, a in stored]
                if got != want:
                    raise CorpusValidationError("layout.features", f"mol_id {mid}: the stored "
                                                f"feature table differs from the re-perceived "
                                                f"one (stored {len(want)} features, perceived "
                                                f"{len(got)}; first difference at feature "
                                                f"{next(i for i, (x, y) in enumerate(zip(want + [None], got + [None])) if x != y)})",
                                                tier="deep", path=gpath)

    def pieces():
        """The serial walk's batches: rows ``[k * BATCH, (k + 1) * BATCH)``."""
        row0, smiles, perceive = 0, [], []
        for c0, chunk in read_column_strings(parquet_file, "smiles"):
            chunk = list(chunk)
            if perceive_ids.size:
                flags = np.isin(ids[c0:c0 + len(chunk)].astype(np.int64), perceive_ids).tolist()
            else:
                flags = [False] * len(chunk)
            for j, s in enumerate(chunk):
                if not smiles:
                    row0 = c0 + j
                smiles.append(s)
                perceive.append(flags[j])
                if len(smiles) == BATCH:
                    yield (row0, smiles, perceive)
                    smiles, perceive = [], []
        if smiles:
            yield (row0, smiles, perceive)

    def batches():
        """Whole consecutive pieces joined while the batch stays within
        PARSE_BATCH rows and BATCH re-perceived molecules. Every batch, and so
        every replay piece, starts on the serial walk's BATCH-row grid."""
        row0, smiles, perceive, n_p = 0, [], [], 0
        for r0, sm, pe in pieces():
            k = sum(pe)
            if smiles and (len(smiles) + len(sm) > PARSE_BATCH or n_p + k > BATCH):
                yield (row0, smiles, perceive)
                smiles, perceive, n_p = [], [], 0
            if not smiles:
                row0 = r0
            smiles = smiles + sm
            perceive = perceive + pe
            n_p += k
        if smiles:
            yield (row0, smiles, perceive)

    if n_workers == 1:
        for batch in batches():
            check(batch, run(batch))
        return
    import collections
    from concurrent.futures import ThreadPoolExecutor
    cancel = threading.Event()
    window: collections.deque = collections.deque()
    pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="r3g-deep-sandbox")
    try:
        for batch in batches():
            window.append((batch, pool.submit(run, batch, cancel)))
            while len(window) >= 2 * n_workers:
                b, fut = window.popleft()
                check(b, fut.result())
        while window:
            b, fut = window.popleft()
            check(b, fut.result())
    finally:
        cancel.set()                    # a refusal stops the children still running
        pool.shutdown(wait=True, cancel_futures=True)

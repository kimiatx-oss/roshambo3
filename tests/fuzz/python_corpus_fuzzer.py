"""Atheris smoke target for the complete format-3 Python reader path.

Each input selects a committed conformance/adversarial corpus, copies the
whole directory to a temporary root, and optionally flips up to eight bytes
in one file. The target runs open -> scan -> native decode of every block and
compares every successful decode with the pure-Python reference. It deliberately
uses a raw file mutator: hash-chain refusals are useful coverage for this short
smoke target, while the existing adversarial seeds exercise deeper checks.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

np = codec = reference = validate = CorpusValidationError = None


def _load_target(atheris_module=None) -> None:
    """Load the reader normally for manual smoke, instrumented for Atheris.

    Seed writing deliberately does not call this, so it remains stdlib-only.
    NumPy is imported before the Atheris import hook: feedback should focus on
    roshambo3's parser, validator, and codec wrapper rather than NumPy internals.
    """
    global np, codec, reference, validate, CorpusValidationError
    import numpy as _np

    if atheris_module is None:
        from roshambo3.format.r3g import codec as _codec, reference as _reference, validate as _validate
        from roshambo3.format.r3g.errors import CorpusValidationError as _error
    else:
        with atheris_module.instrument_imports(include=["roshambo3"]):
            from roshambo3.format.r3g import codec as _codec, reference as _reference, validate as _validate
            from roshambo3.format.r3g.errors import CorpusValidationError as _error

    np, codec, reference, validate, CorpusValidationError = (
        _np, _codec, _reference, _validate, _error)

DATA = ROOT / "tests/data/format3"
_TIMING_PATH = os.environ.get("ROSHAMBO3_FUZZ_TIMINGS")
_TIMING_FILE = open(_TIMING_PATH, "a", encoding="utf-8") if _TIMING_PATH else None
_TIMINGS = {"inputs": 0, "accepted": 0, "refused": 0, "total_seconds": 0.0,
            "max_seconds": 0.0, "over_20_seconds": 0}


def seed_roots() -> list[Path]:
    roots = [DATA / "shape", DATA / "color"]
    adversarial = DATA / "adversarial"
    for root in sorted(adversarial.iterdir()):
        descriptor = root / "corpus.json"
        if (root.is_dir() and not root.is_symlink() and descriptor.is_file()
                and not descriptor.is_symlink()):
            roots.append(root)
    return roots


SEEDS = seed_roots()


def _copy_tree_safely(source: Path, target: Path) -> None:
    """Copy corpus seeds without opening a FIFO or following a symlink."""
    target.mkdir()
    for entry in os.scandir(source):
        src, dst = Path(entry.path), target / entry.name
        mode = src.lstat().st_mode
        if stat.S_ISDIR(mode):
            _copy_tree_safely(src, dst)
        elif stat.S_ISREG(mode):
            shutil.copyfile(src, dst, follow_symlinks=False)
        elif stat.S_ISLNK(mode):
            os.symlink(os.readlink(src), dst)
        elif stat.S_ISFIFO(mode):
            os.mkfifo(dst, stat.S_IMODE(mode))
        # Sockets and device nodes are left absent; they are outside the
        # corpus-reader contract and must never be opened by the harness.


def _candidate_files(root: Path) -> list[Path]:
    files = []
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
        for name in sorted(names):
            path = Path(base) / name
            try:
                mode = path.lstat().st_mode
                if stat.S_ISREG(mode) and path.stat().st_size <= (64 << 20):
                    files.append(path)
            except OSError:
                continue
    return files


def _mutate_one_file(root: Path, data: bytes) -> None:
    if len(data) < 3:
        return
    files = _candidate_files(root)
    if not files:
        return
    target = files[data[1] % len(files)]
    try:
        raw = bytearray(target.read_bytes())
    except OSError:
        return
    if not raw:
        return
    payload = data[2:]
    edits = min(8, 1 + (payload[0] % 8 if payload else data[2] % 8))
    for i in range(edits):
        b = payload[(i * 2) % len(payload)] if payload else data[(i + 2) % len(data)]
        pos = (b + i * 131) % len(raw)
        raw[pos] ^= (1 << ((b + i) % 8))
    target.write_bytes(raw)


def _decode_all_blocks(root: Path, report) -> None:
    if codec.backend() != "native":
        raise RuntimeError("Python corpus fuzzing requires the native CPU extension")
    for entry in report.descriptor["groups"]:
        og = validate.open_group(str(root), entry, descriptor=report.descriptor,
                                 fingerprint=report.fingerprint)
        try:
            h = og.header
            runs = og.section_array("runs")
            coords_offset = int(og.sections["coords"]["offset"])
            for block in og.blocks:
                m0, m1 = int(block["m0"]), int(block["m1"])
                rows = int(block["rows"])
                start = coords_offset + int(block["offset"])
                data = og.file.read(start, int(block["length"]), tier="scan")
                native = codec.decode_block(data, runs[m0:m1], h["L_heavy"])
                n_conf = runs["n_conf"][m0:m1].astype("int64")
                n_real = runs["n_real"][m0:m1].astype("int64")
                if int(n_conf.sum()) != rows:
                    raise AssertionError("validated run slice disagrees with block row count")
                expected = reference.decode_block(
                    data, np.repeat(n_real, n_conf), h["L_heavy"])
                if native.shape != expected.shape or native.tobytes() != expected.tobytes():
                    raise AssertionError("native block decode differs from reference.decode_block")
        finally:
            og.close()


def _exercise(data: bytes) -> str:
    source = SEEDS[data[0] % len(SEEDS)] if data else SEEDS[0]
    with tempfile.TemporaryDirectory(prefix="r3g_python_fuzz_") as tmp:
        root = Path(tmp) / "corpus"
        _copy_tree_safely(source, root)
        _mutate_one_file(root, data)
        validate.validate_corpus(str(root), tier="open")
        report = validate.validate_corpus(str(root), tier="scan", seed=0)
        _decode_all_blocks(root, report)
    return "accepted"


def TestOneInput(data: bytes) -> None:
    start = time.monotonic()
    outcome = "accepted"
    try:
        _exercise(bytes(data))
    except CorpusValidationError as exc:
        outcome = f"refused:{exc.tier}:{exc.code}"
    except Exception:
        outcome = "unexpected_exception"
        raise
    finally:
        elapsed = time.monotonic() - start
        _TIMINGS["inputs"] += 1
        _TIMINGS["total_seconds"] += elapsed
        _TIMINGS["max_seconds"] = max(_TIMINGS["max_seconds"], elapsed)
        if elapsed > 20.0:
            _TIMINGS["over_20_seconds"] += 1
        if outcome == "accepted":
            _TIMINGS["accepted"] += 1
        else:
            _TIMINGS["refused"] += 1
        if _TIMING_FILE is not None:
            _TIMING_FILE.write(json.dumps({"seconds": elapsed, "outcome": outcome,
                                           "input_bytes": len(data)}, sort_keys=True) + "\n")
            _TIMING_FILE.flush()


def write_seed_corpus(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, root in enumerate(SEEDS):
        (directory / f"seed_{i:03d}_{root.name}").write_bytes(bytes([i & 0xFF]))


def _report() -> None:
    if _TIMING_FILE is not None:
        _TIMING_FILE.close()
    print(json.dumps(_TIMINGS, sort_keys=True), file=sys.stderr)


def _instrument_fuzz_functions(atheris_module) -> None:
    # Import instrumentation covers the reader. Instrumenting these helpers
    # also gives mutations feedback for the seed selector and byte mutator.
    for function in (_copy_tree_safely, _candidate_files, _mutate_one_file,
                     _decode_all_blocks, _exercise, TestOneInput):
        atheris_module.instrument_func(function)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-seeds":
        write_seed_corpus(Path(sys.argv[2]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--smoke":
        _load_target()
        import atexit
        atexit.register(_report)
        seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        stop = time.monotonic() + max(0.0, seconds)
        i = 0
        while time.monotonic() < stop:
            TestOneInput(bytes([i % len(SEEDS)]))
            i += 1
    else:
        import atheris
        import atexit
        _load_target(atheris)
        _instrument_fuzz_functions(atheris)
        atexit.register(_report)
        atheris.Setup(sys.argv, TestOneInput)
        atheris.Fuzz()

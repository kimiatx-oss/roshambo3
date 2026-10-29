"""Atheris smoke target for FORMAT.md section 15's RDKit-free pre-pass.

This target never imports or invokes RDKit. It combines curated grammar cases
with mutations of the bare definition.json adversarial seeds, then accepts
only a summary or the documented value/limit CorpusValidationError codes.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

reference = CorpusValidationError = None


def _load_target(atheris_module=None) -> None:
    """Import the definition pre-pass under Atheris' import hook for fuzzing."""
    global reference, CorpusValidationError
    if atheris_module is None:
        from roshambo3.format.r3g import reference as _reference
        from roshambo3.format.r3g.errors import CorpusValidationError as _error
    else:
        with atheris_module.instrument_imports(include=["roshambo3"]):
            from roshambo3.format.r3g import reference as _reference
            from roshambo3.format.r3g.errors import CorpusValidationError as _error
    reference, CorpusValidationError = _reference, _error

DATA = ROOT / "tests/data/format3/adversarial"
TIMING_PATH = os.environ.get("ROSHAMBO3_DEFINITION_FUZZ_TIMINGS")
TIMING_FILE = open(TIMING_PATH, "a", encoding="utf-8") if TIMING_PATH else None
TIMINGS = {"inputs": 0, "accepted": 0, "refused": 0, "total_seconds": 0.0,
           "max_seconds": 0.0, "over_20_seconds": 0}

# Cover mixed-case keywords, continuations, positive/negative !Name macros,
# nested references, and long wildcard/atom chains before Atheris mutates them.
BASE = [
    ("rdkit_fdef", None,
     "aToMtYpE Carbon [#6]\nDeFiNeFeAtUrE Hphobe [{Carbon}]\n"
     "  fAmIlY Hydrophobe\n  Weights 1.0\nEnDfEaTuRe\n"),
    ("rdkit_fdef", None,
     "AtomType Carbon [#6]\\\n  \nDefineFeature Hphobe [{Carbon}]\n"
     "  Family Hydrophobe\n  Weights 1.0\nEndFeature\n"),
    ("rdkit_fdef", None,
     "AtomType !Neg [#8-]\nAtomType Carbon [#6]\n"
     "DefineFeature Hphobe [{Carbon};!{Neg}]\n  Family Hydrophobe\n"
     "  Weights 1.0\nEndFeature\n"),
    ("rdkit_fdef", None,
     "AtomType A0 [#6]\nAtomType A1 [{A0};R]\nAtomType A2 [{A1};R]\n"
     "AtomType A3 [{A2};R]\nDefineFeature Nested [{A3}]\n"
     "  Family Hydrophobe\n  Weights 1.0\nEndFeature\n"),
    ("rdkit_fdef", None,
     "DefineFeature Wild *~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*\n"
     "  Family Hydrophobe\n  Weights 1.0\nEndFeature\n"),
    ("rdkit_fdef", None,
     "DefineFeature Atoms A~A~A~A~A~A~A~A~A~A~A~A~A~A~A~A\n"
     "  Family Hydrophobe\n  Weights 1.0\nEndFeature\n"),
]


def _seed_definitions() -> list[tuple[str, object, object]]:
    seeds = list(BASE)
    for path in sorted(DATA.glob("*/definition.json")):
        try:
            record = json.loads(path.read_bytes())
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("generator") in ("rdkit_fdef", "smarts_list"):
            seeds.append((record["generator"], record.get("types"), record.get("definition")))
    return seeds


SEEDS = _seed_definitions()


def _definition_for(data: bytes):
    selector = data[0] if data else 0
    generator, types, original = SEEDS[selector % len(SEEDS)]
    payload = data[1:]
    if generator == "rdkit_fdef":
        # Insert raw fuzz bytes at a deterministic grammar boundary. UTF-8
        # replacement is intentional: lone surrogates and malformed bytes are
        # part of the pre-pass's input domain, not a reason to skip the target.
        addition = payload.decode("utf-8", "replace")
        text = str(original)
        if payload:
            at = payload[0] % (len(text) + 1)
            text = text[:at] + addition + text[at:]
        return generator, types, text

    # Keep smarts_list definitions structured so mutations reach its pattern
    # and size checks instead of failing only at the top-level JSON shape.
    obj = copy.deepcopy(original)
    if not isinstance(obj, dict):
        obj = {}
    if payload:
        families = sorted(k for k, v in obj.items() if isinstance(v, list) and v)
        family = families[payload[0] % len(families)] if families else "Hydrophobe"
        pattern = payload.decode("utf-8", "replace")
        obj.setdefault(family, []).append(pattern)
    return generator, types, obj


def TestOneInput(data: bytes) -> None:
    start = time.monotonic()
    outcome = "accepted"
    try:
        generator, types, definition = _definition_for(bytes(data))
        reference.check_definition_text(definition, generator, types)
    except CorpusValidationError as exc:
        if not (exc.code == "value.definition" or exc.code.startswith("limit.definition_")):
            outcome = f"unexpected_refusal:{exc.code}"
            raise AssertionError(f"pre-pass returned an undocumented refusal: {exc.code}") from exc
        outcome = f"refused:{exc.code}"
    except Exception:
        outcome = "unexpected_exception"
        raise
    finally:
        elapsed = time.monotonic() - start
        TIMINGS["inputs"] += 1
        TIMINGS["total_seconds"] += elapsed
        TIMINGS["max_seconds"] = max(TIMINGS["max_seconds"], elapsed)
        if elapsed > 20.0:
            TIMINGS["over_20_seconds"] += 1
        if outcome == "accepted":
            TIMINGS["accepted"] += 1
        else:
            TIMINGS["refused"] += 1
        if TIMING_FILE is not None:
            TIMING_FILE.write(json.dumps({"seconds": elapsed, "outcome": outcome,
                                          "input_bytes": len(data)}, sort_keys=True) + "\n")
            TIMING_FILE.flush()


def write_seed_corpus(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, (generator, _, _) in enumerate(SEEDS):
        (directory / f"seed_{i:03d}_{generator}").write_bytes(bytes([i & 0xFF]))


def _report() -> None:
    if TIMING_FILE is not None:
        TIMING_FILE.close()
    print(json.dumps(TIMINGS, sort_keys=True), file=sys.stderr)


def _instrument_fuzz_functions(atheris_module) -> None:
    for function in (_definition_for, TestOneInput):
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

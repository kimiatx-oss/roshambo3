"""The native format-3 codec's standalone C++ fixture, compiled with
g++ the way tests/test_shape_field_evaluator.py compiles its own: the core
sources plus this host's unpack path (AVX2 on x86-64 at x86-64-v3, NEON on
aarch64 at armv8-a+simd), no Python and no pybind11.

Two builds run: plain -O2, and -fsanitize=address,undefined. Each runs the
built-in exhaustive tests (every width 0..16, tails 1..255, n_real 1..255 x
every L, writers, every refusal path) and then every block of the conformance
corpus, the per-width vectors and every adversarial seed through every
compiled ISA path (tests/format3_native_cases.py), whose outputs are checked
against EXPECTED.json and against each other.

The sanitizer build runs by default on Linux. On macOS it runs only with
ROSHAMBO3_R3G_SANITIZE=1 (Apple's toolchain support for the combination on
the CI image is untested here); it is not skipped silently on Linux.
"""

from __future__ import annotations

import os
import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import format3_native_cases as cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CPU_SRC = ROOT / "src/roshambo3/backend/_cpu_src"
CORE = ("r3g_codec.cpp", "r3g_unpack_scalar.cpp", "r3g_build_scalar.cpp", "r3g_encode.cpp")
ARM = platform.machine() in ("aarch64", "arm64")


def _build(tmp_path: Path, extra: list[str], name: str) -> Path:
    arch = "-march=armv8-a+simd" if ARM else "-march=x86-64-v3"
    isa_src = ("r3g_unpack_neon.cpp", "r3g_build_neon.cpp") if ARM else ("r3g_unpack_avx2.cpp", "r3g_build_avx2.cpp")
    exe = tmp_path / name
    cmd = ["g++", "-std=c++17", arch, "-Wall", "-Wextra", *extra, "-I", str(CPU_SRC),
           *(str(CPU_SRC / f) for f in (*CORE, *isa_src)), str(ROOT / "tests/r3g_codec.cpp"),
           "-o", str(exe)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, " ".join(cmd) + "\n" + r.stderr
    return exe


def _run(exe: Path, case_dir: Path, env=None) -> str:
    r = subprocess.run([str(exe), "--cases", str(case_dir)], cwd=ROOT, capture_output=True,
                       text=True, env=env, timeout=1800)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "failures: 0" in r.stdout, r.stdout
    return r.stdout


def _expected_isas() -> list[str]:
    return sorted(["scalar", "neon" if ARM else "avx2"])


def _check(case_dir: Path) -> dict:
    report = cases.check_all(case_dir, isas=_expected_isas())
    expected = json.loads((cases.DATA / "EXPECTED.json").read_bytes())
    assert len(report["conformance"]) == sum(len(expected[v]["groups"]) for v in ("shape", "color"))
    assert len(report["widths"]) == 11
    # A generated seed whose builder failed would otherwise just have no cases.
    errors = {k: v["error"] for k, v in report["adversarial"].items() if "error" in v}
    assert not errors, errors
    # The decoder-relevant seeds are refused by the native code with the same
    # refusal name the validator uses (FORMAT.md 16.5).
    adversarial = json.loads((cases.DATA / "adversarial/EXPECTED.json").read_bytes())
    for seed in ("block_rows_mismatch", "width_17", "block_length_short", "block_reserved_nonzero", "block_rows_4097",
                 "single_molecule_4097_rows", "n_conf_zero", "n_real_zero",
                 "n_real_exceeds_L_heavy", "n_color_exceeds_L", "feature_atom_out_of_range",
                 "feature_offsets_bad", "feature_type_out_of_range", "feature_zero_atoms",
                 "feature_too_many_atoms", "features_256", "block_molecule_gap"):
        got = report["adversarial"][seed]
        assert adversarial[seed].get("native_code", got["validator_code"]) in got["native_codes"], (seed, got)
    return report


def test_fixture_plain(tmp_path):
    exe = _build(tmp_path, ["-O2"], "r3g_codec_test")
    case_dir = tmp_path / "cases"
    cases.write_all(case_dir)
    out = _run(exe, case_dir)
    assert "compiled ISAs: " + " ".join(["scalar", "neon" if ARM else "avx2"]) in out
    _check(case_dir)


@pytest.mark.skipif(platform.system() != "Linux" and os.environ.get("ROSHAMBO3_R3G_SANITIZE") != "1",
                    reason="sanitizer build runs on Linux, elsewhere with ROSHAMBO3_R3G_SANITIZE=1")
def test_fixture_asan_ubsan(tmp_path):
    exe = _build(tmp_path, ["-O1", "-g", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"],
                 "r3g_codec_test_san")
    case_dir = tmp_path / "cases"
    cases.write_all(case_dir)
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:abort_on_error=0",
               UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1")
    _run(exe, case_dir, env=env)
    _check(case_dir)

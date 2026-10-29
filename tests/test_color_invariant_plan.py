"""Compile and exercise the owning color-invariant snapshot."""

from pathlib import Path
import platform
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_color_invariant_plan_snapshots_matrix_and_isa(tmp_path):
    source = ROOT / "src/roshambo3/backend/_cpu_src"
    targets = [("scalar", [])]
    if platform.machine().lower() in ("x86_64", "amd64"):
        targets.append(("avx2", ["-mavx2", "-mfma"]))
    for name, target_flags in targets:
        executable = tmp_path / f"color-invariant-plan-{name}"
        compile_result = subprocess.run(
            ["g++", "-O2", "-std=c++17", "-pthread", *target_flags,
             "-ffunction-sections", "-fdata-sections",
             "-I", str(source),
             str(ROOT / "tests/color_invariant_plan.cpp"),
             str(source / "color_invariant_plan.cpp"),
             str(source / "color_canonical_four_start.cpp"),
             *(["-Wl,-dead_strip"] if platform.system() == "Darwin"
               else ["-Wl,--gc-sections"]),
             "-o", str(executable)],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert compile_result.returncode == 0, compile_result.stderr
        result = subprocess.run(
            [str(executable)], cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASS T188 color invariant plan" in result.stdout

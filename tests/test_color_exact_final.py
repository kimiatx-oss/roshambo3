"""Compile and exercise the exact shared-pose scorer and worker scratch."""

from pathlib import Path
import subprocess

from tests.shared_cache import cached_compile


ROOT = Path(__file__).resolve().parents[1]
CPU_SRC = ROOT / "src/roshambo3/backend/_cpu_src"


def test_exact_final_scratch_parity_and_lifetime(tmp_path):
    executable = tmp_path / "color-exact-final"
    command = [
        "g++", "-std=c++17", "-O2", "-fno-fast-math",
        "-fno-associative-math", "-pthread", f"-I{CPU_SRC}",
        str(ROOT / "tests/color_exact_final.cpp"),
        str(CPU_SRC / "color_exact_final.cpp"),
        str(CPU_SRC / "color_invariant_plan.cpp"),
        str(CPU_SRC / "reduced_pose.cpp"),
        "-o", str(executable),
    ]
    built = cached_compile(command, cwd=ROOT, capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr

    result = subprocess.run(
        [str(executable)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS T192 exact-final scratch parity/lifetime" in result.stdout

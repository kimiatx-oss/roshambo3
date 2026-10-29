from pathlib import Path
import subprocess
import pytest
from test_reduced_pose_color_cpp import _runtime_has_avx2_fma

pytestmark = pytest.mark.skipif(not _runtime_has_avx2_fma(),
                              reason="standalone AVX2 fixture requires AVX2+FMA")

ROOT = Path(__file__).resolve().parents[1]


def test_reduced_force_torque_finite_difference(tmp_path):
    source = ROOT / "src/roshambo3/backend/_cpu_src"
    executable = tmp_path / "torque-factoring"
    subprocess.run(["g++", "-O2", "-std=c++17", "-mavx2", "-mfma",
                    "-I", str(source), str(ROOT / "tests/color_torque_factoring.cpp"),
                    "-o", str(executable)], check=True, capture_output=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True,
                            text=True)
    assert "PASS T215" in result.stdout

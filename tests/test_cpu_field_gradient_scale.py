"""Gradient scale gate, backed by independent overlap-value differences."""

from pathlib import Path
import subprocess
import pytest
from test_reduced_pose_color_cpp import _runtime_has_avx2_fma

pytestmark = pytest.mark.skipif(not _runtime_has_avx2_fma(),
                              reason="standalone AVX2 fixture requires AVX2+FMA")

import numpy as np

from roshambo3.reference.overlap import PAIR_PREFACTOR, pair_overlap


ROOT = Path(__file__).resolve().parents[1]


def test_field_gradient_scale(tmp_path):
    executable = tmp_path / "field-gradient-scale"
    compile_result = subprocess.run(
        ["g++", "-O2", "-march=x86-64-v3", "-std=c++17",
         "-I", str(ROOT / "src/roshambo3/backend/_cpu_src"),
         str(ROOT / "tests/field_gradient_scale.cpp"),
         "-o", str(executable)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    smoke = subprocess.run(
        [str(executable), "--oracle-smoke"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    a = np.array([[0.2, -0.1, 0.7], [-0.5, 0.4, 0.3]])
    b = np.array([[0.6, 0.2, -0.3]]) + [0.1, -0.2, 0.05]
    expected = pair_overlap(a, b) / PAIR_PREFACTOR
    assert abs(float(smoke.stdout) - expected) < 1e-7
    run_result = subprocess.run(
        [str(executable)], cwd=ROOT, capture_output=True, text=True,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "PASS T147 gradient scale" in run_result.stdout
    perpendicular = subprocess.run(
        [str(executable), "--perp-control"], cwd=ROOT,
        capture_output=True, text=True,
    )
    assert perpendicular.returncode != 0
    assert "FAIL convergence pose=0 h=0.2 arm=pairaos setting=3" in perpendicular.stdout
    assert "FAIL pose=0 h=0.2 arm=pairaos setting=3 torque_scale=" not in perpendicular.stdout

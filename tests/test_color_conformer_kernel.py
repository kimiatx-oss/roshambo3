"""Compile and exercise the isolated fixed-pose conformer kernel."""

from pathlib import Path
import subprocess
import pytest
from test_reduced_pose_color_cpp import _runtime_has_avx2_fma

pytestmark = pytest.mark.skipif(not _runtime_has_avx2_fma(),
                              reason="standalone AVX2 fixture requires AVX2+FMA")


ROOT = Path(__file__).resolve().parents[1]


def test_color_conformer_kernel_fixed_pose(tmp_path):
    source = ROOT / "src/roshambo3/backend/_cpu_src"
    for name, target_flags in (("avx2", ["-mavx2", "-mfma"]),
                               ("scalar", [])):
        executable = tmp_path / f"color-conformer-kernel-{name}"
        compile_result = subprocess.run(
            ["g++", "-O2", "-std=c++17", *target_flags,
             "-I", str(source),
             str(ROOT / "tests/color_conformer_kernel.cpp"),
             str(ROOT / "tests/color_field_bundle_link_stubs.cpp"),
             str(source / "color_conformer_kernel.cpp"),
             str(source / "color_canonical_kernel.cpp"),
             str(source / "color_invariant_plan.cpp"),
             str(source / "reduced_pose.cpp"),
             "-o", str(executable)],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert compile_result.returncode == 0, compile_result.stderr
        result = subprocess.run(
            [str(executable)], cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASS conformer kernel" in result.stdout

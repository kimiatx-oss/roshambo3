"""Native gate for prepared scalar and AVX2 shape-field evaluators."""

import platform
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CPU_SRC = ROOT / "src/roshambo3/backend/_cpu_src"


def test_shape_field_evaluator(tmp_path):
    executable = tmp_path / "shape-field-evaluator"
    if platform.machine() in ("aarch64", "arm64"):
        base = ["g++", "-O2", "-std=c++17", "-march=armv8-a+simd",
                "-I", str(CPU_SRC), "-c"]
        sources = ("shape_evaluator.cpp", "shape_field_evaluator.cpp",
                   "field_neon.cpp", "cpu_platform.cpp")
        objects = []
        for name in sources:
            obj = tmp_path / f"{name}.o"
            result = subprocess.run(base + [str(CPU_SRC / name), "-o", str(obj)],
                                    cwd=ROOT, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            objects.append(str(obj))
        fp16_obj = tmp_path / "field_neon_fp16.cpp.o"
        result = subprocess.run(
            ["g++", "-O2", "-std=c++17", "-march=armv8.2-a+fp16",
             "-I", str(CPU_SRC), "-c",
             str(CPU_SRC / "field_neon_fp16.cpp"), "-o", str(fp16_obj)],
            cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        objects.append(str(fp16_obj))
        test_obj = tmp_path / "shape_field_evaluator_test.o"
        result = subprocess.run(
            base + [str(ROOT / "tests/shape_field_evaluator.cpp"),
                    "-o", str(test_obj)], cwd=ROOT,
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        objects.append(str(test_obj))
        link_result = subprocess.run(["g++", *objects, "-o", str(executable)],
                                     cwd=ROOT, capture_output=True, text=True)
    else:
        link_result = subprocess.run(
            ["g++", "-O2", "-march=x86-64-v3", "-std=c++17",
             "-I", str(CPU_SRC),
             str(CPU_SRC / "shape_evaluator.cpp"),
             str(CPU_SRC / "shape_field_evaluator.cpp"),
             str(ROOT / "tests/shape_field_evaluator.cpp"),
             "-o", str(executable)],
            cwd=ROOT, capture_output=True, text=True)
    assert link_result.returncode == 0, link_result.stderr
    run_result = subprocess.run(
        [str(executable)], cwd=ROOT, capture_output=True, text=True,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "PASS T185 shape field evaluator" in run_result.stdout

"""Standalone ASan/UBSan check of the new native GPU launch entry point."""
import os
import platform
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "src/roshambo3/backend/_cpu_src"


@pytest.mark.skipif(platform.system() != "Linux", reason="ASan/UBSan exercised on Linux")
def test_native_launch_asan_ubsan(tmp_path):
    arm = platform.machine() in {"arm64", "aarch64"}
    sources = ["r3g_codec.cpp", "r3g_encode.cpp", "r3g_unpack_scalar.cpp", "r3g_build_scalar.cpp"]
    sources += (["r3g_unpack_neon.cpp", "r3g_build_neon.cpp"] if arm else
                ["r3g_unpack_avx2.cpp", "r3g_build_avx2.cpp"])
    exe = tmp_path / "launch-sanitized"
    cmd = ["g++", "-std=c++17", "-O1", "-g", "-Wall", "-Wextra",
           "-march=armv8-a+simd" if arm else "-march=x86-64-v3",
           "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
           "-fno-omit-frame-pointer", "-I", str(CPU),
           *(str(CPU / s) for s in sources), str(ROOT / "tests/r3g_gpu_launch.cpp"),
           "-o", str(exe)]
    # Share the same build arbitration as editable/native project builds.
    build_lock = os.environ.get("ROSHAMBO3_BUILD_LOCK", str(tmp_path / "build.lock"))
    compiled = subprocess.run(["flock", build_lock, *cmd],
                              capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stderr
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1", UBSAN_OPTIONS="halt_on_error=1")
    result = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all checks passed" in result.stdout

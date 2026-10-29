"""Compile the standalone shared-pose/color integration fixture."""

from pathlib import Path
import platform
import subprocess

from tests.shared_cache import cached_compile

import pytest


ROOT = Path(__file__).resolve().parents[1]
CPU_SRC = ROOT / "src/roshambo3/backend/_cpu_src"


def _runtime_has_avx2_fma():
    """Never execute an AVX2-targeted fixture on unsupported hardware."""
    if platform.machine().lower() not in {"x86_64", "amd64", "i386", "i686"}:
        return False
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return False
    flags = set()
    for line in cpuinfo.splitlines():
        if line.startswith(("flags", "Features")) and ":" in line:
            flags.update(line.split(":", 1)[1].split())
    return {"avx2", "fma"} <= flags


@pytest.mark.parametrize("name,target_flags", [
    ("scalar", []), ("avx2", ["-mavx2", "-mfma"]),
])
def test_reduced_pose_color_fixture(tmp_path, name, target_flags):
    if name == "avx2" and not _runtime_has_avx2_fma():
        pytest.skip("AVX2+FMA fixture requires runtime CPU support")
    fixture = ROOT / "tests/reduced_pose_color.cpp"
    stubs = ROOT / "tests/color_field_bundle_link_stubs.cpp"
    common_sources = [
        fixture,
        stubs,
        CPU_SRC / "color_conformer_kernel.cpp",
        CPU_SRC / "color_canonical_kernel.cpp",
        CPU_SRC / "color_exact_final.cpp",
        CPU_SRC / "color_invariant_plan.cpp",
        CPU_SRC / "reduced_pose.cpp",
    ]

    if platform.machine().lower() in {"aarch64", "arm64"}:
        common_sources.append(CPU_SRC / "color_canonical_neon.cpp")

    executable = tmp_path / f"reduced-pose-color-{name}"
    command = [
        "g++", "-std=c++17", "-O2", "-fno-fast-math",
        "-fno-associative-math", "-pthread", *target_flags,
        f"-I{CPU_SRC}", *(str(source) for source in common_sources),
        "-o", str(executable),
    ]
    built = cached_compile(command, cwd=ROOT, capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr

    result = subprocess.run([str(executable)], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS T191 reduced pose/color" in result.stdout

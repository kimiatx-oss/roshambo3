"""The build system's architecture discovery, against real toolkit output.

`nvcc --list-gpu-arch` is the toolkit describing itself, so the build asks it
rather than carrying a CUDA-version-to-architecture table that would need
maintaining forever and would be wrong for as long as nobody noticed.

These tests drive the real CMake logic with a **fake nvcc** that replays output
captured from actual toolkits -- CUDA 12.9 and 13.2, in `tests/data/`. That is deliberately not a unit test of a Python reimplementation:
the thing that has to be right is the CMake, and a second implementation agreeing
with itself proves nothing.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "tests" / "data"

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None,
                                reason="cmake is not installed")

#: Extracted verbatim from CMakeLists.txt so the test cannot drift from it.
_MARKER = "function(roshambo3_supported_archs out_var)"


def _extract_cmake_logic() -> str:
    """The discovery block, lifted out of the real CMakeLists.txt."""
    text = (REPO / "CMakeLists.txt").read_text()
    start = text.index("set(ROSHAMBO3_ARCH_WISHLIST")
    end = text.index("include(CheckLanguage)")
    wishlist = text[start:end]
    fstart = text.index(_MARKER)
    fend = text.index("if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)", fstart)
    func = text[fstart:fend]
    sstart = fend
    send = text.index("\nset(_r3_src", sstart)
    select = text[sstart:send]
    return wishlist + func + select


def _run(tmp_path: pathlib.Path, arch_file: pathlib.Path) -> str:
    """Configure a throwaway project whose 'nvcc' replays ``arch_file``."""
    fake = tmp_path / "fake_nvcc"
    fake.write_text("#!/bin/sh\n"
                    'if [ "$1" = "--list-gpu-arch" ]; then\n'
                    f'  cat "{arch_file}"\n'
                    "  exit 0\n"
                    "fi\n"
                    "exit 1\n")
    fake.chmod(0o755)

    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(archprobe NONE)\n"
        f'set(CMAKE_CUDA_COMPILER "{fake}")\n'
        + _extract_cmake_logic()
        + '\nmessage(STATUS "SELECTED=${CMAKE_CUDA_ARCHITECTURES}")\n')

    build = tmp_path / "b"
    build.mkdir()
    r = subprocess.run(["cmake", str(tmp_path)], cwd=build,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for line in r.stdout.splitlines():
        if "SELECTED=" in line:
            return line.split("SELECTED=", 1)[1].strip()
    raise AssertionError("the project did not report a selection:\n" + r.stdout)


def test_cuda_129_includes_volta(tmp_path):
    """CUDA 12.9 still supports sm_70, and nothing in this code needs Ampere --
    zero `__CUDA_ARCH__` guards -- so Volta should be targeted, and is measured
    working."""
    got = _run(tmp_path, DATA / "nvcc_list_gpu_arch_cuda129.txt")
    assert "70-real" in got, got
    assert got.endswith("120-virtual"), got


def test_cuda_132_drops_volta_without_failing(tmp_path):
    """CUDA 13.x rejects sm_70 outright. The build must simply omit it rather
    than fail, which is the whole point of discovering instead of hardcoding."""
    got = _run(tmp_path, DATA / "nvcc_list_gpu_arch_cuda132.txt")
    assert "70-real" not in got, got
    assert "75-real" in got, got
    assert got.endswith("120-virtual"), got


@pytest.mark.parametrize("toolkit", ["cuda129", "cuda132"])
def test_only_the_newest_architecture_gets_ptx(tmp_path, toolkit):
    """PTX for every architecture would bloat the binary for nothing -- the
    driver only ever JITs from the newest it can use."""
    got = _run(tmp_path, DATA / f"nvcc_list_gpu_arch_{toolkit}.txt")
    virtual = [a for a in got.split(";") if a.endswith("-virtual")]
    assert virtual == ["120-virtual"], got


@pytest.mark.parametrize("toolkit", ["cuda129", "cuda132"])
def test_every_selection_is_supported_by_the_toolkit(tmp_path, toolkit):
    """The selection must be a subset of what nvcc reported, or the build fails
    at compile time instead of at configure time -- which is much later and much
    more confusing."""
    name = f"nvcc_list_gpu_arch_{toolkit}.txt"
    got = _run(tmp_path, DATA / name)
    supported = {l.strip().replace("compute_", "")
                 for l in (DATA / name).read_text().splitlines() if l.strip()}
    for entry in got.split(";"):
        arch = entry.split("-")[0]
        assert arch in supported, f"{arch} not in {name}: {got}"

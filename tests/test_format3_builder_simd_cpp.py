"""Exhaustive direct SIMD builder arithmetic, layout and tail checks."""
import os
import platform
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARM = platform.machine() in ('aarch64', 'arm64')


@pytest.mark.parametrize('sanitized', [False, True])
def test_builder_primitives(tmp_path, sanitized):
    if sanitized and platform.system() != 'Linux' and os.environ.get('ROSHAMBO3_R3G_SANITIZE') != '1':
        pytest.skip('sanitizers require Linux or explicit opt-in')
    src = ROOT / 'src/roshambo3/backend/_cpu_src'
    isa = 'neon' if ARM else 'avx2'
    exe = tmp_path / 'builder'
    flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
             '-fno-omit-frame-pointer'] if sanitized else ['-O2']
    cmd = ['g++', '-std=c++17', '-march=armv8-a+simd' if ARM else '-march=x86-64-v3',
           '-Wall', '-Wextra', *flags, '-I', str(src), str(src / f'r3g_build_{isa}.cpp'),
           str(ROOT / 'tests/r3g_builder_simd.cpp'), '-o', str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout + build.stderr
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1', UBSAN_OPTIONS='halt_on_error=1')
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS' in result.stdout, result.stdout

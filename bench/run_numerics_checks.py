"""Run the portable numerical family against this checkout and a named build.

Usage: python bench/run_numerics_checks.py --native-build build/t270
A local build is optional for pure reference tests, but unavailable backends
remain not-run. This never mutates an environment's editable installation.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def select_native_build(package, build):
    """Prefer the named build and refuse native modules already loaded elsewhere."""
    build = build.resolve()
    for name in ('cpu', 'cpu_probe', 'cuda', 'metal'):
        module = sys.modules.get('roshambo3._roshambo3_' + name)
        if module is not None and Path(module.__file__).resolve().parent != build:
            raise RuntimeError('native extension already imported from outside requested build: ' + module.__file__)
    package.__path__[:] = [str(build), *(p for p in package.__path__ if Path(p).resolve() != build)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-build', type=Path)
    parser.add_argument('--report-dir', type=Path, default=Path('numerics-report'))
    parser.add_argument('--profile', choices=('cpu-cuda', 'cpu-cuda-metal'), default='cpu-cuda')
    parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    sys.meta_path[:] = [finder for finder in sys.meta_path
                       if 'ScikitBuild' not in type(finder).__name__]
    sys.path[:0] = [str(root / 'src'), str(root)]
    import roshambo3
    if Path(roshambo3.__file__).resolve() != root / 'src/roshambo3/__init__.py':
        raise RuntimeError('roshambo3 was imported from another checkout')
    print('Numerics Python source:', roshambo3.__file__, flush=True)
    if args.native_build:
        build = args.native_build.resolve()
        if not build.is_dir():
            parser.error('native build directory does not exist')
        select_native_build(roshambo3, build)
        from bench.numerics_backends import verified_build
        from roshambo3.backend import cpu, cuda, metal
        for name, backend in (('cpu', cpu), ('cuda', cuda), ('metal', metal)):
            if backend.available():
                if Path(backend.extension().__file__).resolve().parent != build:
                    raise RuntimeError(f'{name} extension was not loaded from requested build')
                verified_build(name)
                print(f'Numerics {name} extension:', backend.extension().__file__, flush=True)
    # The source stamp binds test/generator code; this binds committed input
    # bytes and the preregistered real-panel selection protocol separately.
    digest = hashlib.sha256()
    for path in sorted((root / 'tests/data/alignment').glob('*')):
        if path.is_file():
            digest.update(path.name.encode() + b'\0' + path.read_bytes())
    panel_stamp = 'oracle-numerics-v1:' + digest.hexdigest()
    tests = args.pytest_args
    if tests and tests[0] == '--':
        tests = tests[1:]
    if not tests:
        tests = [str(p) for p in sorted((root / 'tests').glob('test_numerics_*.py'))]
    import pytest
    return pytest.main(['--no-header', '--numerics-report-dir', str(args.report_dir),
                        '--numerics-panel-stamp', panel_stamp,
                        '--numerics-profile', args.profile, *tests])


if __name__ == '__main__':
    raise SystemExit(main())

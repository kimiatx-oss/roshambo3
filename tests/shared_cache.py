"""Private content-addressed test artifacts, shared across workers, never hardlinked."""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]


def digest_files(paths):
    digest = hashlib.sha256()
    for path in sorted(map(Path, paths)):
        digest.update(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(None)
def source_key():
    import numpy, rdkit
    return dict(source=digest_files([*ROOT.glob('src/roshambo3/**/*.py'),
        *(p for p in ROOT.glob('src/roshambo3/backend/_cpu_src/*') if p.is_file()),
        *ROOT.glob('tests/*.hpp'), *ROOT.glob('tests/*.h'), ROOT/'tests/shared_cache.py']),
        numpy=numpy.__version__, rdkit=rdkit.__version__, machine=platform.machine())


@contextmanager
def compile_lock():
    # The singleton makes a helper safely reentrant under the test-level guard.
    path = os.environ.get('ROSHAMBO3_TEST_BUILD_LOCK', str(Path(tempfile.gettempdir())/'roshambo3-test-build.lock'))
    with FileLock(path, is_singleton=True):
        yield


def cached_directory(kind, inputs, build):
    root = os.environ.get('ROSHAMBO3_TEST_CACHE')
    if not root:
        raise RuntimeError('shared test cache fixture is not active')
    key = hashlib.sha256(json.dumps(dict(kind=kind, inputs=inputs, source=source_key()),
                                   sort_keys=True).encode()).hexdigest()
    directory = Path(root)/kind/key
    directory.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory)+'.lock'):
        marker = directory/'complete.json'
        if not marker.is_file():
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir()
            try:
                build(directory)
                manifest = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in directory.rglob('*') if p.is_file()}
                marker.write_text(json.dumps(manifest, sort_keys=True))
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
        return directory


@lru_cache(None)
def compiler_identity(executable):
    compiler = shutil.which(executable)
    if not compiler:
        raise FileNotFoundError(executable)
    return dict(path=compiler, sha256=hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
                version=subprocess.check_output([compiler, '--version'], text=True))


def cached_compile(command, **kwargs):
    """Cache standalone oracle setup; compile-behavior tests keep subprocess.run."""
    index = command.index('-o')+1
    destination = Path(command[index])
    normalized = list(map(str, command))
    normalized[index] = '<output>'
    paths = [Path(arg) for arg in command[1:] if str(arg).endswith(('.cpp', '.cc', '.h', '.hpp'))]
    inputs = dict(command=normalized, compiler=compiler_identity(command[0]), sources=digest_files(paths))
    result = None
    flags = [arg for i, arg in enumerate(command) if i not in (index-1, index)
             and Path(str(arg)) not in paths]
    environment = kwargs.get('env', os.environ)
    inputs['environment'] = {key: environment.get(key) for key in
                             ('CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'GCC_EXEC_PREFIX')}
    def build(directory):
        nonlocal result
        objects = []
        for source in paths:
            def compile_object(target, source=source):
                cmd = [*flags, '-c', str(source), '-o', str(target/'fixture.o')]
                with compile_lock():
                    built = subprocess.run(cmd, **kwargs)
                if built.returncode:
                    raise RuntimeError((built.stdout or '')+(built.stderr or ''))
            obj = cached_directory('native-object', dict(flags=flags,
                compiler=inputs['compiler'], environment=inputs['environment'],
                source=digest_files([source])), compile_object)
            objects.append(str(obj/'fixture.o'))
        cmd = [*flags, *objects, '-o', str(directory/'fixture')]
        with compile_lock():
            result = subprocess.run(cmd, **kwargs)
        if result.returncode:
            raise RuntimeError((result.stdout or '')+(result.stderr or ''))
    directory = cached_directory('native', inputs, build)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(directory/'fixture', destination)
    return result or subprocess.CompletedProcess(command, 0, '', '')

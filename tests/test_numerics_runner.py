"""Named native builds must control module resolution before execution."""
import importlib.machinery
import sys
from types import SimpleNamespace

import pytest

from bench.run_numerics_checks import select_native_build


def test_named_build_takes_precedence_over_package_extension(tmp_path, monkeypatch):
    for backend in ("cpu", "cpu_probe", "cuda", "metal"):
        monkeypatch.delitem(sys.modules, "roshambo3._roshambo3_" + backend, raising=False)
    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir(); build.mkdir()
    name = "_roshambo3_cpu"
    filename = name + importlib.machinery.EXTENSION_SUFFIXES[0]
    (source / filename).touch()
    (build / filename).touch()
    package = SimpleNamespace(__path__=[str(source)])
    select_native_build(package, build)
    found = importlib.machinery.PathFinder.find_spec(name, package.__path__)
    assert found.origin == str(build / filename)


def test_cached_extension_from_other_build_is_refused(tmp_path, monkeypatch):
    package = SimpleNamespace(__path__=[str(tmp_path / "source")])
    module = SimpleNamespace(__file__=str(tmp_path / "other" / "_roshambo3_cpu.so"))
    monkeypatch.setitem(sys.modules, "roshambo3._roshambo3_cpu", module)
    with pytest.raises(RuntimeError, match="outside requested build"):
        select_native_build(package, tmp_path / "requested")

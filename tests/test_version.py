"""One version: ``roshambo3.__version__`` is ``pyproject.toml``'s ``version``.

``prep.version`` in every ``corpus.json`` and ``roshambo3_version`` in every
``build.json`` are ``__version__``, and FORMAT.md 13.3 rests rebuild identity
on it, so a second copy that drifts would put a wrong release into corpora.
``__version__`` comes from the installed distribution's metadata, falling back
to one literal when there is none.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import pathlib
import tomllib

import pytest

import roshambo3

PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject_version() -> str:
    if not PYPROJECT.is_file():
        pytest.skip("no pyproject.toml beside the tests (an installed test tree)")
    return tomllib.loads(PYPROJECT.read_text())["project"]["version"]


def test_version_is_pyprojects(pyproject_version):
    """Fails after an edit to ``pyproject.toml``'s version until the package
    is reinstalled: an editable install's metadata is written at install time
    and would otherwise report the old version."""
    assert roshambo3.__version__ == pyproject_version, (
        "stale install metadata: reinstall with pip install -e .")


def test_the_fallback_literal_is_pyprojects(pyproject_version):
    """The literal used with no distribution metadata (``PYTHONPATH=src``)."""
    assert roshambo3._SOURCE_VERSION == pyproject_version


def test_without_metadata_the_fallback_is_used(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    assert roshambo3._installed_version() == roshambo3._SOURCE_VERSION


def _fake_distribution(site: pathlib.Path, version: str, direct_url: dict | None = None):
    """A ``roshambo3-<version>.dist-info`` under ``site``, owning
    ``site/roshambo3/__init__.py`` as a regular install would."""
    info = site / f"roshambo3-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: roshambo3\nVersion: {version}\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))
    return importlib.metadata.PathDistribution(info)


@pytest.fixture
def first_on_path(monkeypatch):
    """Put a fake distribution first on ``sys.path``, where
    ``importlib.metadata`` looks."""
    def install(site: pathlib.Path):
        monkeypatch.syspath_prepend(str(site))
        importlib.invalidate_caches()
    return install


def test_metadata_of_another_installation_is_not_trusted(tmp_path, first_on_path):
    """Development code run against an environment holding a release (say
    ``PYTHONPATH=src`` over a venv with 0.2.0 installed) must not record that
    release's version: the metadata found first on ``sys.path`` does not own
    the module that was imported."""
    _fake_distribution(tmp_path / "site", "0.2.0")
    first_on_path(tmp_path / "site")
    assert importlib.metadata.version("roshambo3") == "0.2.0"
    assert roshambo3._installed_version() == roshambo3._SOURCE_VERSION


def test_a_regular_install_owns_its_module(tmp_path, first_on_path):
    site = tmp_path / "site"
    _fake_distribution(site, "0.2.0")
    (site / "roshambo3").mkdir()
    (site / "roshambo3" / "__init__.py").write_text("")
    first_on_path(site)
    assert roshambo3._installed_version(str(site / "roshambo3" / "__init__.py")) == "0.2.0"


def test_an_editable_install_owns_its_source_tree(tmp_path, first_on_path):
    tree = tmp_path / "tree"
    module = tree / "src" / "roshambo3" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    _fake_distribution(tmp_path / "site", "0.2.0.dev3",
                       {"url": tree.as_uri(), "dir_info": {"editable": True}})
    first_on_path(tmp_path / "site")
    assert roshambo3._installed_version(str(module)) == "0.2.0.dev3"
    # An editable install of another tree does not own this one.
    other = tmp_path / "other" / "src" / "roshambo3" / "__init__.py"
    other.parent.mkdir(parents=True)
    other.write_text("")
    assert roshambo3._installed_version(str(other)) == roshambo3._SOURCE_VERSION
    # Nor does it own a worktree nested inside it.
    nested = tree / ".claude" / "worktrees" / "w" / "src" / "roshambo3" / "__init__.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("")
    assert roshambo3._installed_version(str(nested)) == roshambo3._SOURCE_VERSION


def test_the_builder_records_this_version(tmp_path):
    """``prep.version`` and ``build.json``'s ``roshambo3_version`` are
    ``roshambo3.__version__``, from a real (one-molecule) build."""
    from roshambo3.prep import prepare

    sdf = pathlib.Path(__file__).resolve().parent / "data" / "alignment" / "ADRB2_3p0g.sdf"
    out = tmp_path / "corpus"
    prepare(str(sdf), str(out), name="VERSION", workers=1)
    descriptor = json.loads((out / "corpus.json").read_text())
    build = json.loads((out / "build.json").read_text())
    assert descriptor["prep"]["version"] == roshambo3.__version__
    assert build["roshambo3_version"] == roshambo3.__version__

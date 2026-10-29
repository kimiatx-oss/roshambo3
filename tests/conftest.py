"""Shared fixtures.

Tests must run from anywhere -- ROSHAMBO2's suite only passed with the working
directory set to ``test/``, which meant CI carried a ``working-directory:``
line and a developer running ``pytest`` from the repo root got failures that
looked like real ones.
"""

from __future__ import annotations

import pathlib

import pytest

from roshambo3.paths import corpora_root

pytest_plugins = ["bench.numerics_pytest", "bench.test_pytest"]

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Corpora are large and live once, outside the repo. Set
#: ROSHAMBO3_CORPORA to point at them; tests needing one skip cleanly otherwise,
#: so a checkout with no data still has a passing suite.
CORPORA = corpora_root()


def pytest_runtest_setup(item):
    """Skip a grain-parametrized test when this host cannot run that grain.

    The vector color grains are compiled for one ISA each (AVX2+FMA on x86,
    NEON on ARM); a host without it refuses them. Stamp tests and the refusal
    tests in test_cpu_color_build_option.py still run, since they hold whether
    or not the grain can run here.
    """
    if "stamp" in item.name or item.path.name == "test_cpu_color_build_option.py":
        return
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return
    for key in ("grain", "color_grain"):
        value = callspec.params.get(key)
        if value is None:
            continue
        try:
            grain = int(value)
        except (TypeError, ValueError):
            continue
        if not 0 <= grain <= 17:
            continue
        from roshambo3.backend import cpu
        if not cpu.available():
            return
        stamp = cpu.color_build_stamp(grain)
        if not (stamp.get("color_target_compiled", True) and
                stamp.get("color_runtime_available", True)):
            pytest.skip(f"CPU color grain {grain} requires an unavailable ISA")


@pytest.fixture(scope="session", autouse=True)
def _private_scan_cache(tmp_path_factory):
    """Keep the format-3 scan cache out of the user's home directory.

    ``prepare`` and ``roshambo3 verify`` record a passing scan in the
    scan cache (``format.r3g.cache``: ``$ROSHAMBO3_SCAN_CACHE``, else
    ``~/.cache/roshambo3/scan``); a test run would otherwise leave one entry
    per corpus it built there. A test that sets the variable itself still
    wins (``monkeypatch.setenv``)."""
    import os
    if os.environ.get("ROSHAMBO3_SCAN_CACHE"):
        yield
        return
    os.environ["ROSHAMBO3_SCAN_CACHE"] = str(tmp_path_factory.mktemp("scan-cache"))
    try:
        yield
    finally:
        os.environ.pop("ROSHAMBO3_SCAN_CACHE", None)


def _corpus_or_skip(name: str):
    root = CORPORA / name
    if not (root / "corpus.json").exists():
        pytest.skip(f"corpus {name} not present at {root}")
    from roshambo3.format import Corpus
    from roshambo3.format.r3g.reader import verify
    verify(str(root), tier="scan")
    return Corpus(str(root))


@pytest.fixture(scope="session")
def smoke_corpus():
    """D0-SMOKE-2K: 1,966 conformers, small enough to sweep exhaustively."""
    return _corpus_or_skip("format3/D0-SMOKE-2K")


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def dev_corpus():
    """D1: 199,734 molecules/conformers, 10 groups and 53 physical blocks.

    Molecule-aligned blocks exercise launch coalescing without fixed-size
    physical block assumptions.
    """
    return _corpus_or_skip("format3/D1-DEV-200K")


@pytest.fixture(scope="session")
def multiconformer_corpus():
    """D8: 520,630 molecules, 998,173 conformers, 12 groups and 252 blocks.

    Molecules never straddle blocks. One source molecule was rejected by the
    canonical-SMILES round-trip gate.
    """
    return _corpus_or_skip("format3/D8-ADV-1M")


@pytest.fixture(scope="session")
def color_corpus():
    """PUR2: 811 molecules / 48,660 conformers, six color groups / 15 blocks.

    One canonical rebuild serves fixture and attribution roles. Its
    descriptor records definition content, matrices and type numbering.
    """
    return _corpus_or_skip("format3/dudez-PUR2-color-T299-w1")

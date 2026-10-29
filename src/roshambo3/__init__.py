"""ROSHAMBO3 -- ultra-scale 3D molecular shape screening on GPUs.

    import roshambo3 as r3
    hits = r3.screen("queries.sdf", "$EXTERNAL_DATA", k=100_000)

The public surface is deliberately small, and the pieces it assembles are all
reachable directly: an Engine, a Source, a Sink and a ScreenConfig. See
``doc/ARCHITECTURE.md`` for how they fit together, and
``doc/NUMERICS_POLICY.md`` before changing anything that produces a score.

Importing this module must not import the CUDA extension: corpus preparation,
format inspection and metrics all work without a GPU.
"""

from __future__ import annotations

from . import metrics, numerics, optimizer, paths, scores
from ._build_info import BuildInfo, get_build_info
from .api import load_queries, open_corpus, score_matrix, screen
from .engine import (CPUWorkerOptions, Callback, DenseResult, DenseScores, Engine, MoleculeTopK,
                     QuerySet, R3gSource, ScreenConfig, ScreenStats, TopK)
from .format import Catalog, Corpus
from .layout import BTILE, tile32
from .numerics import assert_arithmetic_pinned, check_build, rank_order
from .results import QueryHits, ScreenResult
from .scores import SCORE_DTYPE

_SOURCE_VERSION = "1.0.0"
"""``pyproject.toml``'s ``version``, for a source tree run with no distribution
metadata (``PYTHONPATH=src``). The one copy outside ``pyproject.toml``;
``tests/test_version.py`` holds the two equal."""


def _installed_version(module_file: str = __file__) -> str:
    """The version of the distribution that owns this module, else
    :data:`_SOURCE_VERSION`.

    Packaging writes the metadata from ``pyproject.toml`` (and, once release
    builds take it from the tag, from the tag). The metadata is trusted only
    if its distribution owns ``module_file``, because
    ``importlib.metadata`` returns the first ``roshambo3`` metadata on
    ``sys.path`` whatever code was imported: development code run with
    ``PYTHONPATH=src`` against an environment holding an installed release
    would otherwise record that release's version (no ``.dev`` suffix) in
    ``prep.version``, claiming an identity it does not have. A regular install
    owns it when ``locate_file`` resolves to ``module_file``; an editable one
    when ``direct_url.json`` marks it editable and ``module_file`` is that
    tree's own ``src/roshambo3/__init__.py`` (not merely somewhere under it: a
    worktree can be nested inside another checkout).

    An editable install's metadata is written at install time and is not
    refreshed by editing ``pyproject.toml``: until the next install it reports
    the old version, and ``tests/test_version.py`` fails rather than letting a
    corpus record it. ``prep.version`` and ``build.json`` read this value.
    """
    import json
    from importlib.metadata import PackageNotFoundError, distribution
    from pathlib import Path
    from urllib.parse import unquote, urlparse
    try:
        dist = distribution("roshambo3")
    except PackageNotFoundError:
        return _SOURCE_VERSION
    here = Path(module_file).resolve()
    try:
        if Path(dist.locate_file("roshambo3/__init__.py")).resolve() == here:
            return dist.version
        direct = json.loads(dist.read_text("direct_url.json") or "null")
        if (isinstance(direct, dict)
                and (direct.get("dir_info") or {}).get("editable")):
            url = urlparse(direct.get("url", ""))
            root = Path(unquote(url.path)).resolve()
            if url.scheme == "file" and here == root / "src" / "roshambo3" / "__init__.py":
                return dist.version
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return _SOURCE_VERSION


__version__ = _installed_version()

__all__ = [
    # the common case
    "screen", "score_matrix", "load_queries", "open_corpus",
    # the pieces
    "Engine", "ScreenConfig", "ScreenStats", "CPUWorkerOptions",
    "R3gSource", "TopK", "MoleculeTopK", "DenseScores", "DenseResult",
    "Callback",
    "QuerySet", "Corpus", "Catalog",
    # results
    "ScreenResult", "QueryHits", "SCORE_DTYPE", "scores",
    # policy
    "numerics", "optimizer", "metrics", "paths", "rank_order",
    "check_build", "assert_arithmetic_pinned",
    # build
    "BuildInfo", "get_build_info",
    "tile32", "BTILE",
    "__version__",
]

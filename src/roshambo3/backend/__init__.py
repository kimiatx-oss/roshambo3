"""Backend selection, and the vocabulary a configuration is written in.

Two things live here, and they are separate on purpose.

**The vocabulary** -- the enums and the two shared-memory helpers -- comes from
:mod:`roshambo3.backend.spec` and is importable with no GPU build. That is what
lets a ``ScreenConfig`` be constructed on a machine with no CUDA toolkit, which
requires that ``engine.config`` not import it from ``backend.cuda``.

**The backend** is resolved by :func:`select`, and there is exactly one seam it
has to satisfy: ``engine.engine`` calls ``backend.score_pairs`` at one call
site and nothing else in the package calls a kernel at all. A backend module
must provide ``score_pairs``, ``available()`` and ``extension()``.

Selection order, most specific first:

1. the ``backend=`` argument to :class:`roshambo3.engine.Engine`;
2. the ``ROSHAMBO3_BACKEND`` environment variable;
3. ``"cuda"``.

**An unknown or unbuildable name raises rather than falling back.** A silent
fallback to a different backend is the same class of failure as a stale binary -- the
run produces numbers, and nothing in them says which code produced them.
"""

from __future__ import annotations

import importlib
import os
from typing import Optional

from .spec import (
    ColorLoop,
    ColorPlacement,
    CoordResidency,
    PoseSearch,
    UpdateLanes,
    UpdateRule,
    analytic_block_shared_bytes,
    carveout_for,
)

#: Env var naming the default backend. Read once per :func:`select` call rather
#: than cached, so a test can set it; the resolved module is what gets cached.
BACKEND_ENV = "ROSHAMBO3_BACKEND"

DEFAULT_BACKEND = "cuda"

#: Importable backend modules, by name. A new backend is one entry here plus a
#: module satisfying the seam -- deliberately a table and not a plugin
#: registry: with two backends a registry is indirection with nothing behind it.
_BACKENDS = {"cuda": "roshambo3.backend.cuda",
             "cpu": "roshambo3.backend.cpu",
             "metal": "roshambo3.backend.metal"}

_cache: dict = {}


def select(name: Optional[str] = None):
    """Resolve a backend module.

    ``name`` overrides the environment, which overrides
    :data:`DEFAULT_BACKEND`.
    """
    chosen = name or os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    chosen = str(chosen).strip().lower()
    if chosen in _cache:
        return _cache[chosen]
    if chosen not in _BACKENDS:
        raise ValueError(
            f"unknown backend {chosen!r}; known backends are "
            f"{sorted(_BACKENDS)}. Set {BACKEND_ENV} or pass "
            f"Engine(backend=...). There is deliberately no fallback: a screen "
            f"that quietly ran a different backend produces numbers with "
            f"nothing in them to say which code made them.")
    mod = importlib.import_module(_BACKENDS[chosen])
    for required in ("score_pairs", "available", "extension"):
        if not hasattr(mod, required):
            raise TypeError(
                f"backend {chosen!r} ({mod.__name__}) does not provide "
                f"{required!r}. The seam is score_pairs / available / "
                f"extension; see roshambo3.backend.__doc__.")
    _cache[chosen] = mod
    return mod


def backend_names() -> list:
    """Every backend name :func:`select` will accept."""
    return sorted(_BACKENDS)


# The module-level surface stays what it was: `from roshambo3 import backend;
# backend.score_pairs(...)` resolves the default, and `from roshambo3.backend
# import score_pairs` binds the default backend's own function.
#
# Resolved through PEP 562's module __getattr__ rather than a forwarding
# wrapper, because **the signature is part of the surface**: a `def
# score_pairs(*a, **kw)` shim would have hidden it, and
# `test_exp_map_field_cuda.py::test_the_backend_entry_default` introspects
# `sig.parameters["update_rule"].default` to pin the default to the exponential
# map. It caught exactly that, which is what it is for.
def __getattr__(name):
    if name in ("score_pairs", "extension"):
        return getattr(select(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def available() -> bool:
    """Whether the default backend can actually run.

    False rather than raising when the backend cannot even be imported: a
    machine with no CUDA toolkit is a supported configuration for everything
    except screening, and ~18 test modules gate on this.
    """
    try:
        return bool(select().available())
    except Exception:
        return False


__all__ = [
    "score_pairs", "available", "extension", "select", "backend_names",
    "BACKEND_ENV", "DEFAULT_BACKEND",
    "carveout_for", "analytic_block_shared_bytes",
    "PoseSearch", "CoordResidency", "UpdateLanes", "UpdateRule",
    "ColorLoop", "ColorPlacement",
]

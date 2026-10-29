"""The streaming screen loop: sources, sinks, pinned buffers and the arena."""

from __future__ import annotations

from .buffers import PinnedPool, pack_into
from .config import CPUWorkerOptions, ScreenConfig
from .engine import Engine, ScreenStats
from .query import QuerySet, from_arrays, from_rdkit
from .sink import (Callback, DenseResult, DenseScores, HostTopK,
                   MoleculeTopK, Sink, TopK)
from .r3g_source import R3gSource
from .source import (DEFAULT_CONFORMERS_PER_LAUNCH, DEFAULT_PREFETCH_DEPTH,
                     ConformerBlock, Prefetcher, Source,
                     conformers_per_launch, make_source)

__all__ = [
    "Engine", "ScreenStats", "ScreenConfig", "CPUWorkerOptions",
    "Source", "R3gSource", "make_source", "ConformerBlock", "Prefetcher",
    "conformers_per_launch", "DEFAULT_CONFORMERS_PER_LAUNCH",
    "DEFAULT_PREFETCH_DEPTH",
    "Sink", "TopK", "HostTopK", "MoleculeTopK", "DenseScores", "DenseResult",
    "Callback",
    "QuerySet", "from_arrays", "from_rdkit",
    "PinnedPool", "pack_into",
]

"""The engine refuses a block whose count vector is not exactly its width.

A dense launch derives every row from the launch shape, and the shape's
library width is the count vector's row count. A count vector longer than
``n_conformers`` would score rows past the block, and a dense top-K would admit
them under the next block's gids, so the engine refuses it before launching.
CPU-only: the backend is a fake that must never be reached.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from roshambo3.engine import Engine, ScreenConfig, from_arrays
from roshambo3.engine.source import ConformerBlock


class OneBlock:
    prefetch_depth = 0
    producer_wait_s = 0.

    def __init__(self, n_conformers, n_count_rows):
        self.block = ConformerBlock(
            coords=np.zeros((32, 4, 3), np.int16),
            counts=np.full((n_count_rows, 1), 3, np.uint8),
            n_conformers=n_conformers, L=4, bucket=4, gid_base=0)
        self.max_block_conformers = n_conformers
        self.max_block_elements = 32 * 4 * 3

    def blocks(self):
        yield self.block


class Sink:
    def prepare(self, *args): pass
    def add_block(self, *args): pass
    def finalize(self): return "done"


@pytest.mark.parametrize("n_count_rows", [6, 4])
def test_a_count_vector_that_is_not_the_block_width_is_refused(n_count_rows):
    calls = []
    backend = SimpleNamespace(score_pairs=lambda **kw: calls.append(kw),
                              available=lambda: False)
    queries = from_arrays([np.zeros((3, 3), np.float32)], ["query"])
    engine = Engine(pinned=False, backend=backend)
    with pytest.raises(ValueError, match="count vector has"):
        engine.run(OneBlock(5, n_count_rows), queries, ScreenConfig(), Sink())
    assert not calls, "the backend must not be reached"

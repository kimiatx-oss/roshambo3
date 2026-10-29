"""CPU query preparation must not discover a CUDA device."""
import numpy as np
import pytest

from roshambo3.backend import cpu, cuda
from roshambo3.engine.query import from_arrays


@pytest.mark.parametrize("backend", ["cpu", cpu])
def test_cpu_query_preparation_does_not_probe_cuda(monkeypatch, backend):
    monkeypatch.setattr(cuda, "extension", lambda: pytest.fail("CPU query probed CUDA"))
    from roshambo3.engine import config
    monkeypatch.setattr(config, "_smem_budget_cache", {})
    query = from_arrays([np.array([[0, 0, 0], [1, 0, 0]], np.float32)],
                        ["cpu"], backend=backend)
    assert query.n_real.tolist() == [2]
    assert query.names == ["cpu"]

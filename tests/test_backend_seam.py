"""The backend is selectable, and the host reducer keeps the contract.

Neither `backend/__init__.py` nor `engine/engine.py` may bind `.cuda` at
module scope, and the top-K sinks must not reach into the extension for
`DeviceTopK`.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from roshambo3 import backend
from roshambo3.engine.sink import HostTopK
from roshambo3.scores import N_SCORE_COLUMNS


class TestTheVocabularyDoesNotNeedAGpuBuild:
    """`engine.config` imports every enum at module scope, so a
    `ScreenConfig` must be constructible without importing the CUDA
    wrapper."""

    def test_spec_imports_nothing_from_the_extension(self):
        import roshambo3.backend.spec as spec
        src = __import__("pathlib").Path(spec.__file__).read_text()
        assert "extension()" not in src
        assert "import cuda" not in src and "from .cuda" not in src

    def test_config_takes_its_vocabulary_from_spec(self):
        """At MODULE SCOPE, which is the whole point: an import inside a
        function runs only when that function is called, and
        `shared_memory_budget`'s `extension()` import is deliberately lazy and
        wrapped -- asking the device is the right thing when there is a device
        and must not be a hard dependency when there is not."""
        import ast
        import pathlib as _p
        import roshambo3.engine.config as cfg
        tree = ast.parse(_p.Path(cfg.__file__).read_text())
        top = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
        mods = {n.module for n in top}
        assert "..backend.spec".lstrip(".") in {m.lstrip(".") for m in mods
                                                if m}
        assert not any((m or "").endswith("backend.cuda") for m in mods), (
            "ScreenConfig must be constructible without importing the CUDA "
            "wrapper; a lazy import inside a function is fine, a module-scope "
            "one is not")

    def test_the_enums_are_the_same_objects_either_way(self):
        """Re-exported, not redefined. Two enum classes with equal members
        compare unequal, so a duplicate here would break every isinstance and
        every `is` in the dispatch."""
        from roshambo3.backend import cuda, spec
        for name in ("PoseSearch", "CoordResidency", "UpdateRule",
                     "UpdateLanes", "ColorLoop", "ColorPlacement"):
            assert getattr(cuda, name) is getattr(spec, name), name
            assert getattr(backend, name) is getattr(spec, name), name


class TestSelect:
    def test_the_default_is_cuda(self, monkeypatch):
        monkeypatch.delenv(backend.BACKEND_ENV, raising=False)
        assert backend.select().__name__ == "roshambo3.backend.cuda"

    def test_the_environment_names_it(self, monkeypatch):
        monkeypatch.setenv(backend.BACKEND_ENV, "cuda")
        assert backend.select().__name__ == "roshambo3.backend.cuda"

    def test_an_explicit_name_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv(backend.BACKEND_ENV, "nonexistent")
        assert backend.select("cuda").__name__ == "roshambo3.backend.cuda"

    def test_an_unknown_backend_raises_rather_than_falling_back(self, monkeypatch):
        """A silent fallback is the classic silent failure: the run produces numbers and
        nothing in them says which code produced them."""
        monkeypatch.delenv(backend.BACKEND_ENV, raising=False)
        with pytest.raises(ValueError, match="unknown backend"):
            backend.select("cpu-that-does-not-exist-yet")

    def test_score_pairs_keeps_its_signature_through_the_selector(self):
        """The signature is part of the surface -- `test_exp_map_field_cuda`
        introspects `update_rule`'s default to pin it. A forwarding
        `def score_pairs(*a, **kw)` shim would hide it, which is why the module
        resolves through PEP 562's __getattr__ instead."""
        import inspect
        from roshambo3.backend import cuda
        sig = inspect.signature(backend.score_pairs)
        assert "update_rule" in sig.parameters
        assert backend.score_pairs is cuda.score_pairs

    def test_available_is_false_rather_than_raising_with_no_backend(
            self, monkeypatch):
        """~18 test modules gate on this, and a machine with no CUDA toolkit is
        a supported configuration for everything but screening."""
        monkeypatch.setenv(backend.BACKEND_ENV, "nope")
        assert backend.available() is False


class TestEngineHoldsAResolvedBackend:
    def test_a_module_can_be_passed_directly(self):
        """What lets a measurement arm supply a backend without registering
        one."""
        from roshambo3.engine import Engine
        from roshambo3.backend import cuda
        fake = types.ModuleType("fake_backend")
        fake.score_pairs = cuda.score_pairs
        fake.available = cuda.available
        fake.extension = cuda.extension
        e = Engine(backend=fake)
        assert e.backend is fake

    def test_it_is_resolved_once_at_construction(self, monkeypatch):
        """Not looked up per launch: a screen must not change backend halfway
        through."""
        from roshambo3.engine import Engine
        monkeypatch.delenv(backend.BACKEND_ENV, raising=False)
        e = Engine()
        held = e.backend
        monkeypatch.setenv(backend.BACKEND_ENV, "nope")
        assert e.backend is held


def _block(n, gid_base=0):
    b = types.SimpleNamespace()
    b.n_conformers = n
    b.gid_base = gid_base
    return b


def _rows(n_q, n, rng):
    s = np.zeros((n_q * n, N_SCORE_COLUMNS), np.float32)
    s[:, 0] = rng.random(n_q * n).astype(np.float32)
    return s


class TestHostTopKReproducesTheDeterminismArgument:
    """`topk.cuh:33-58` states three facts that make the device's output a pure
    function of the SET of conformers screened. The host reducer was written to
    that argument rather than to the device's algorithm, so these test the
    argument."""

    def _run(self, k, n_q, chunks, rng_seed=7):
        rng = np.random.default_rng(rng_seed)
        total = sum(chunks)
        full = _rows(n_q, total, rng)
        by_q = full.reshape(n_q, total, -1)
        sink = HostTopK(k=k)
        sink.prepare(n_q, 0)
        off = 0
        for n in chunks:
            part = by_q[:, off:off + n, :].reshape(n_q * n, -1)
            sink.add_block(_block(n, gid_base=off), np.ascontiguousarray(part))
            off += n
        return sink.finalize(), by_q

    def test_the_answer_does_not_depend_on_block_size(self):
        a, _ = self._run(k=10, n_q=3, chunks=[64, 64, 64, 64])
        b, _ = self._run(k=10, n_q=3, chunks=[256])
        c, _ = self._run(k=10, n_q=3, chunks=[1] * 256)
        for q in range(3):
            assert np.array_equal(a[q][1], b[q][1]), "block size changed the ids"
            assert np.array_equal(a[q][1], c[q][1])
            assert np.array_equal(a[q][0], b[q][0])

    def test_it_returns_the_true_top_k(self):
        k, n_q = 10, 3
        got, by_q = self._run(k=k, n_q=n_q, chunks=[97, 97, 62])
        for q in range(n_q):
            want = np.sort(by_q[q][:, 0])[::-1][:k]
            assert np.allclose(got[q][0][:, 0], want)
            assert len(got[q][1]) == k

    def test_ties_break_on_ascending_id(self):
        """Float32 ties are common at scale -- 50,737 in one measured run -- and
        an unstable sort manufactures rank changes out of nothing."""
        n = 8
        s = np.zeros((n, N_SCORE_COLUMNS), np.float32)
        s[:, 0] = 1.0                       # every score identical
        sink = HostTopK(k=4)
        sink.prepare(1, 0)
        sink.add_block(_block(n, gid_base=100), s)
        (_, gids), = sink.finalize()
        assert list(gids) == [100, 101, 102, 103]

    def test_fewer_rows_than_k_is_not_an_error(self):
        sink = HostTopK(k=1000)
        sink.prepare(1, 0)
        sink.add_block(_block(5), _rows(1, 5, np.random.default_rng(1)))
        (scores, gids), = sink.finalize()
        assert len(gids) == 5 and scores.shape == (5, N_SCORE_COLUMNS)

    def test_a_query_that_saw_nothing_returns_empty_not_garbage(self):
        sink = HostTopK(k=4)
        sink.prepare(2, 0)
        (s0, g0), (s1, g1) = sink.finalize()
        assert g0.size == 0 and s0.shape == (0, N_SCORE_COLUMNS)

    def test_the_threshold_only_rises(self):
        """Fact 1: a row rejected against a higher threshold would also have
        been cut by the final selection."""
        rng = np.random.default_rng(3)
        sink = HostTopK(k=5)
        sink.prepare(1, 0)
        seen = [-np.inf]
        for i in range(20):
            sink.add_block(_block(40, gid_base=40 * i), _rows(1, 40, rng))
            seen.append(sink.thresholds()[0])
        assert all(b >= a for a, b in zip(seen, seen[1:])), seen

    def test_it_wants_the_dense_block_and_owns_no_device_state(self):
        """How the engine tells a host reducer from a device one -- routed on
        `handle`, not on isinstance."""
        sink = HostTopK(k=4)
        assert sink.wants_dense is True
        assert sink.handle is None

    def test_it_refuses_to_be_handed_no_scores(self):
        sink = HostTopK(k=4)
        sink.prepare(1, 0)
        with pytest.raises(ValueError, match="needs the dense score block"):
            sink.add_block(_block(4), None)

    def test_k_must_be_positive_and_rank_col_a_real_column(self):
        with pytest.raises(ValueError, match="k must be positive"):
            HostTopK(k=0)
        with pytest.raises(ValueError, match="not a score column"):
            HostTopK(k=4, rank_col=N_SCORE_COLUMNS)

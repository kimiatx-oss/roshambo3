"""The CPU extension's own arithmetic is inspected, and the backend asks.

Two defects of one shape, both found by wiring the CPU ladder into the
harness rather than by reading:

1. ``assert_arithmetic_pinned()`` had exactly one call site, ``cuda.py``, while
   ``CLAUDE.md`` said it "runs at every backend entry". The CPU backend shipped
   without it.
2. Even called, it would have inspected the wrong artifact.
   ``numerics.check_build()`` reads ``get_build_info()``, which is the **CUDA**
   extension's stamp; the CPU extension is a separate compilation with its own
   flags, and nothing read them.

Together those mean ``SUSPECT_CXX_FLAGS`` never applied to the binary that does
the CPU arithmetic. CPU policy **defers** ``-ffast-math`` and friends rather
than refusing them -- they remain an explicit future arm -- which makes
noticing them the entire safeguard, and there was none.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from roshambo3 import numerics
from roshambo3.backend import cpu


def _cpu_extension_present() -> bool:
    try:
        cpu.build_stamp()
        return True
    except Exception:                                       # noqa: BLE001
        return False


needs_cpu_ext = pytest.mark.skipif(
    not _cpu_extension_present(), reason="no CPU extension in this build")


@needs_cpu_ext
class TestTheCpuExtensionsFlagsAreRead:

    def test_a_relaxed_flag_in_the_cpu_extension_is_noticed(self, monkeypatch):
        """The case that motivated this: a CPU build with -ffast-math.

        Simulated at the stamp rather than by rebuilding, because the thing
        under test is whether anything READS the CPU stamp -- and once
        nothing did, so this test would have passed vacuously with an empty
        advisory list.
        """
        real = cpu.build_stamp
        monkeypatch.setattr(cpu, "build_stamp", lambda: dict(
            real(), cxx_flags=" -O3 -DNDEBUG -march=x86-64-v3 -ffast-math"))

        with_cpu, advisories = numerics._cpu_flag_advisories()

        assert with_cpu
        assert any("-ffast-math" in a for a in advisories)
        # And it must name WHICH artifact, or a reader cannot act on it: the
        # CUDA extension's host half and the CPU extension are both "host
        # flags" and only one of them is the problem.
        assert any("CPU EXTENSION" in a for a in advisories)

    def test_the_shipped_cpu_build_is_clean(self):
        """The real one, unmocked. `-fno-fast-math -fno-associative-math` are
        what the CPU build sets as the current
        setting -- so this asserts the check does not cry wolf on it."""
        with_cpu, advisories = numerics._cpu_flag_advisories()
        assert with_cpu
        assert advisories == []

    def test_an_unstamped_cpu_extension_says_it_could_not_check(self,
                                                                monkeypatch):
        """"Nothing to check" and "checked and clean" must not look alike.

        The same distinction ``checked_a_gpu_build`` exists for, and the same
        one the golden fixture once failed to draw.
        """
        real = cpu.build_stamp
        monkeypatch.setattr(cpu, "build_stamp",
                            lambda: dict(real(), cxx_flags="none"))

        with_cpu, advisories = numerics._cpu_flag_advisories()

        assert not with_cpu
        assert advisories and "nothing to check" in advisories[0]

    def test_check_build_reports_that_a_cpu_build_was_checked(self):
        res = numerics.check_build()
        assert res.checked_a_cpu_build

    def test_a_missing_cpu_extension_is_not_an_exception(self, monkeypatch):
        """A tree with no CPU extension is legitimate and must stay
        recordable. Raising here would break the CUDA path, which calls
        check_build() too."""
        def boom():
            raise ImportError("no CPU extension")
        monkeypatch.setattr(cpu, "build_stamp", boom)

        with_cpu, advisories = numerics._cpu_flag_advisories()

        assert not with_cpu
        assert advisories == []
        # ...and check_build() then says so rather than staying silent.
        res = numerics.check_build()
        assert not res.checked_a_cpu_build
        assert any("no CPU extension is present" in a for a in res.advisories)


@needs_cpu_ext
class TestTheBackendAsks:

    def test_score_pairs_asserts_the_arithmetic_is_pinned(self, monkeypatch):
        """The invariant CLAUDE.md states, made true for this backend.

        Asserted by observing the call rather than by grepping the source: a
        test that greps passes on a call inside an `if False`.
        """
        calls = []
        import roshambo3.numerics as N
        monkeypatch.setattr(N, "_checked", False)
        real = N.assert_arithmetic_pinned

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(N, "assert_arithmetic_pinned", spy)

        rs = np.random.RandomState(0)
        q = np.zeros((1, 4, 4), dtype=np.float32)
        q[0, :, :3] = rs.randn(4, 3)
        q[0, :, 3] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cpu.score_pairs(
                query_coords=q, query_n_atoms=np.array([4], dtype=np.int32),
                library_coords=np.ascontiguousarray(q.copy()),
                library_counts=np.array([[4]], dtype=np.uint8),
                pair_query_row=np.array([0], dtype=np.int32),
                pair_library_row=np.array([0], dtype=np.int32), steps=3)

        assert calls, ("the CPU backend did not assert the arithmetic was "
                       "pinned; that invariant is what CLAUDE.md claims runs "
                       "at every backend entry")

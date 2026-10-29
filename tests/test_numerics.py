"""The numerics policy, as code. See doc/NUMERICS_POLICY.md.

These run without a GPU and are the half of the policy that CI can enforce on
every push.
"""

from __future__ import annotations

import math
import pathlib
import re

import numpy as np
import pytest

from roshambo3 import numerics as N
from roshambo3._build_info import BuildInfo


def _build(**kw):
    base = dict(build_type="Release", git_sha="0" * 40, with_cuda=True,
                cuda_architectures="80-real,120-real,120-virtual",
                nvcc_version=N.PINNED_CUDA_TOOLKIT,
                cxx_flags="-O3", cuda_flags="-O3 -lineinfo --ptxas-options=-v")
    base.update(kw)
    return BuildInfo(**base)


class TestConstants:
    def test_derived_constants_follow_from_kappa_and_the_carbon_radius(self):
        a = np.float32(np.float32(N.KAPPA) / np.float32(np.float32(N.CARBON_RADIUS) ** 2))
        assert N.A == a
        assert N.ALPHA == np.float32(np.float32(a * a) / np.float32(a + a))

    def test_coord_scale_makes_the_pair_term_exactly_exp2(self):
        """The reduced frame's whole point: coordinates premultiplied by
        COORD_SCALE make the pair term exp2(-|d|^2), so the kernel needs no
        multiply inside the inner loop."""
        assert N.COORD_SCALE == np.float32(math.sqrt(float(N.ALPHA) * math.log2(math.e)))

    def test_shape_prefactor_uses_full_precision_pi(self):
        """ROSHAMBO2's corpus builder used a truncated PI = 3.14159265358 while
        this constant used math.pi, so the vaa sidecar and the kernel disagreed.
        roshambo3.prep now uses math.pi too."""
        from roshambo3.prep import GAUSS_ALPHA, GAUSS_CONST
        # N.A is float32 (it is a device constant); prep works in float64. The
        # agreement to float32 precision is the point -- they are the same
        # quantity, computed from the same pi.
        assert np.isclose(float(N.A), GAUSS_ALPHA, rtol=1e-7)
        assert np.isclose(float(N.SHAPE_PREFACTOR), 8.0 * GAUSS_CONST, rtol=1e-6)

    def test_device_constants_are_float32(self):
        """They are compared against float literals in a CUDA header, so a
        float64 here would be a false mismatch."""
        for name, v in N.DEVICE_CONSTANTS.items():
            assert isinstance(v, np.float32), name


class TestForbiddenFlags:
    @pytest.mark.parametrize("flag", ["--use_fast_math", "-fmad=false",
                                      "-ftz=true", "-prec-div=false",
                                      "-prec-sqrt=false"])
    def test_each_measured_no_go_flag_is_refused(self, flag):
        """All six compiler axes land on the same output distribution, so there
        is no small-perturbation regime: the policy pins rather than tolerates."""
        assert N.check_cuda_flags(f"-O3 {flag}"), flag

    def test_the_shipped_flags_are_clean(self):
        assert N.check_cuda_flags("-O3 -lineinfo --ptxas-options=-v") == []

    def test_absent_flags_are_clean(self):
        assert N.check_cuda_flags("none") == []
        assert N.check_cuda_flags("") == []

    def test_every_forbidden_flag_carries_its_measurement(self):
        """A rule with no number behind it is a preference. Each entry must
        state what it does to the shape Tanimoto."""
        for f in N.FORBIDDEN_CUDA_FLAGS:
            assert f.why
            assert f.max_dT > 0
            assert 0 < f.basin_flip_rate < 1


class TestCheckBuild:
    def test_a_clean_release_build_passes_with_no_advisories(self):
        r = N.check_build(_build())
        assert r.ok and r.checked_a_gpu_build and not r.advisories

    def test_a_forbidden_flag_is_a_violation_not_an_advisory(self):
        r = N.check_build(_build(cuda_flags="-O3 --use_fast_math"))
        assert not r.ok and len(r.violations) == 1

    def test_an_unpinned_toolkit_is_an_advisory_not_a_violation(self):
        """The toolkit is a real but much weaker axis: 92.95% of pairs stay
        bit-identical between 13.2 and 12.9."""
        r = N.check_build(_build(nvcc_version="12.9"))
        assert r.ok and r.advisories

    def test_an_unvalidated_architecture_is_an_advisory(self):
        r = N.check_build(_build(cuda_architectures="80-real"))
        assert r.ok and any("sm_120" in a for a in r.advisories)

    def test_ok_is_vacuous_without_a_cuda_build(self):
        """The trap this guards: `ok` means "found no violations", which is
        trivially true when there is no extension to inspect. A harness must
        require checked_a_gpu_build as well."""
        r = N.check_build(_build(with_cuda=False, cuda_flags="none",
                                 nvcc_version="none", cuda_architectures="none"))
        assert r.ok
        assert not r.checked_a_gpu_build
        assert any("no CUDA extension" in a for a in r.advisories)

    def test_an_empty_host_flag_advisory_list_is_vacuous_without_host_flags(self):
        """The same trap on the host side.

        `check_cxx_flags` returns [] both when it inspected the stamp and found
        none of SUSPECT_CXX_FLAGS and when the stamp carried no host flags at
        all. From the empty list a caller cannot tell which happened, so the
        fact is carried separately rather than inferred.
        """
        clean = N.check_build(_build())
        assert clean.checked_host_flags

        r = N.check_build(_build(cxx_flags="none"))
        assert r.ok, "missing host flags are not a violation"
        assert not r.checked_host_flags
        assert any("no host compiler flags" in a for a in r.advisories)

    def test_host_flags_seen_is_about_the_stamp_not_about_cleanliness(self):
        """A build WITH host flags that happen to be clean must report that it
        was checked -- otherwise the advisory fires on every compliant build
        and people learn to ignore it."""
        r = N.check_build(_build(cxx_flags="-O3 -DNDEBUG"))
        assert r.checked_host_flags
        assert not any("no host compiler flags" in a for a in r.advisories)

    def test_a_suspect_host_flag_is_still_only_an_advisory(self):
        """Host flags are not a measured axis on this project, so they are
        reported and never refused -- SUSPECT_CXX_FLAGS says so in its own
        docstring, and widening it would claim a number nobody has taken."""
        r = N.check_build(_build(cxx_flags="-O3 -ffast-math"))
        assert r.ok
        assert r.checked_host_flags
        assert any("-ffast-math" in a for a in r.advisories)

    def test_the_stamp_records_whether_anyone_looked(self):
        """A record carrying `arithmetic_ok` alone can be
        read as verified when nothing was verified."""
        s = N.numerics_stamp(_build())
        assert s["arithmetic_ok"] is True
        assert s["arithmetic_checked_a_gpu_build"] is True
        assert s["arithmetic_checked_host_flags"] is True

        s = N.numerics_stamp(_build(with_cuda=False, cuda_flags="none",
                                    cxx_flags="none", nvcc_version="none",
                                    cuda_architectures="none"))
        assert s["arithmetic_ok"] is True, "still vacuously true, by design"
        assert s["arithmetic_checked_a_gpu_build"] is False
        assert s["arithmetic_checked_host_flags"] is False


class TestRankOrder:
    def test_sorts_by_descending_score(self):
        s = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        assert N.rank_order(s, np.arange(3)).tolist() == [1, 2, 0]

    def test_breaks_ties_by_ascending_id(self):
        """Float32 ties are common at scale -- one run had 50,737 of them.
        An unstable sort is a correctness bug, not a performance detail."""
        s = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        ids = np.array([30, 10, 20])
        assert ids[N.rank_order(s, ids)].tolist() == [10, 20, 30]

    def test_is_a_total_order_so_the_result_cannot_depend_on_input_order(self):
        rng = np.random.default_rng(0)
        s = rng.choice([0.1, 0.5, 0.9], size=200).astype(np.float32)
        ids = np.arange(200)
        base = ids[N.rank_order(s, ids)]
        for _ in range(5):
            p = rng.permutation(200)
            assert np.array_equal(ids[p][N.rank_order(s[p], ids[p])], base)

    def test_is_stable_across_a_batch_split(self):
        """The determinism claim the device top-K makes: the answer is a
        function of the SET screened, not of how it was chunked."""
        rng = np.random.default_rng(1)
        s = rng.choice([0.2, 0.7], size=100).astype(np.float32)
        ids = np.arange(100)
        whole = ids[N.rank_order(s, ids)][:10]
        halves = np.concatenate([ids[:50][N.rank_order(s[:50], ids[:50])][:10],
                                 ids[50:][N.rank_order(s[50:], ids[50:])][:10]])
        merged = halves[N.rank_order(s[halves], halves)][:10]
        assert np.array_equal(whole, merged)


class TestTheDeviceHeaderMatchesThePythonDerivation:
    """The check ROSHAMBO2 had and the port dropped.

    ``TestConstants`` verifies the Python constants follow from kappa and the
    carbon radius. Nothing verified that ``kernels.cuh``'s float literals are
    those constants -- so the derivation could be right, the kernel could be
    wrong, and every test would pass. ``DEVICE_CONSTANTS`` has claimed in its
    own docstring since the port that a test pins it; this is that test, and
    until now the claim was false (``doc/NUMERICS_POLICY.md`` said so in one
    place and contradicted itself in another).

    The literals are typed into the header by hand and cannot be generated:
    ``nvcc`` sees the header, not Python.
    """

    def _header(self):
        hdr = (pathlib.Path(__file__).resolve().parent.parent
               / "src/roshambo3/backend/_cuda_src/kernels.cuh")
        if not hdr.exists():
            pytest.skip("kernels.cuh not present")
        return hdr.read_text()

    def test_every_pinned_constant_is_in_the_header_and_equal(self):
        src = self._header()
        for macro, value in N.DEVICE_CONSTANTS.items():
            m = re.search(
                rf"#define\s+{macro}\s+\(?\s*(-?[0-9.]+(?:[eE][-+]?\d+)?)f\s*\)?",
                src)
            assert m, f"{macro} is not defined in kernels.cuh"
            in_header = np.float32(m.group(1))
            # Exact: both sides are float32 and the header literal is the
            # decimal expansion of the Python value. A tolerance here would
            # defeat the purpose -- the whole point is that the two agree
            # bit for bit in the arithmetic the kernel performs.
            assert in_header == np.float32(value), (
                f"{macro}: header has {in_header!r}, Python derives "
                f"{np.float32(value)!r}")

    def test_the_derived_reciprocal_still_follows(self):
        """INV_COORD_SCALE is defined in the header as an expression rather
        than a literal, so what has to hold is the relation, not a digit
        string."""
        src = self._header()
        assert re.search(
            r"#define\s+ROSHAMBO3_INV_COORD_SCALE\s+\(1\.0f/ROSHAMBO3_COORD_SCALE\)",
            src), "INV_COORD_SCALE is no longer derived from COORD_SCALE"


@pytest.mark.parametrize("name", ["limits", "reference"])
def test_format3_step_mismatch_is_refused(monkeypatch, name):
    from roshambo3.format.r3g import limits, reference
    monkeypatch.setattr({"limits": limits, "reference": reference}[name], "STEP_LOG2", -9)
    with pytest.raises(RuntimeError, match=rf"WIRE_QUANT_STEP=.*{name}.STEP_LOG2"):
        N.assert_arithmetic_pinned(_build(), force=True)


def test_compiled_wire_step_mismatch_is_refused(monkeypatch):
    monkeypatch.setattr(N, "compiled_wire_constants", lambda: (0.001, 1000.0))
    with pytest.raises(RuntimeError, match="0.00390625.*compiled wire step=0.001"):
        N.assert_arithmetic_pinned(_build(), force=True)


def test_compiled_reciprocal_must_match(monkeypatch):
    monkeypatch.setattr(N, "compiled_wire_constants", lambda: (2.0 ** -8, 1000.0))
    with pytest.raises(RuntimeError, match="product must equal 1"):
        N.assert_arithmetic_pinned(_build(), force=True)


def test_reader_uses_compiled_step_without_a_cuda_context(monkeypatch):
    from roshambo3.engine.buffers import require_wire_quantization
    monkeypatch.setattr(N, "compiled_wire_constants", lambda: (2.0 ** -8, 256.0))
    with pytest.raises(N.WireQuantizationError, match="0.001.*0.00390625"):
        require_wire_quantization(0.001)


def test_metal_shader_substitutes_the_python_step():
    from roshambo3.backend import metal
    source = metal._wire_kernel_source(metal._SOURCE.read_text())
    assert "R3_WIRE_QUANT_STEP" not in source
    assert f"{float(N.WIRE_QUANT_STEP):.9g}f;" in source

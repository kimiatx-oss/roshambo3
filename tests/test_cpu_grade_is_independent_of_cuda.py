"""A CPU number's grade must not depend on whether CUDA was compiled beside it.

Found on a machine with no CUDA extension, where every CPU run
drew two advisories that are about the CUDA extension -- and since any advisory
downgrades `decision` to `advisory`, **no CPU measurement on a CUDA-less machine
could ever be decision-grade.** Identical CPU runs elsewhere graded `decision`
only because a CUDA extension happened to exist in the same tree.

Two causes, both a scoping mechanism applied in one direction only:

1. ``check_build`` appended "no CUDA extension is present" to ``advisories``
   but not to ``device_advisories``, so the CPU filter in
   ``RunContext.from_build`` could not drop it. The CPU-side twin, "no CPU
   extension is present", *was* marked, so the defect only pointed one way.
2. ``checked_host_flags`` is read off ``get_build_info()``, the CUDA stamp. On a
   CUDA-less tree that is False -- so a CPU run was told its host flags had not
   been inspected when ``_cpu_flag_advisories`` had just read them off the CPU
   extension's own stamp.

Neither weakens anything: a CPU run is still refused outright if its own
extension was never inspected (``checked_a_cpu_build``), and the CUDA advisories
still fire on GPU runs. What changes is which artifact each check reads.
"""
from __future__ import annotations

import dataclasses

import pytest

from roshambo3 import numerics


class _NoCudaBuild:
    """A build stamp for a tree with no CUDA extension."""
    with_cuda = False
    cuda_flags = "none"
    cxx_flags = "none"
    nvcc_version = "none"
    cuda_architectures = ""


class TestTheNoCudaAdvisoryIsDeviceScoped:

    def test_it_is_marked_as_a_device_advisory(self):
        c = numerics.check_build(_NoCudaBuild())
        hits = [a for a in c.advisories if "no CUDA extension is present" in a]
        assert hits, "expected the advisory on a CUDA-less build"
        assert all(a in c.device_advisories for a in hits), (
            "the no-CUDA advisory must be device-scoped, or a CPU run cannot "
            "filter it and a CUDA-less machine becomes ungradeable")

    def test_the_cpu_twin_is_still_host_scoped(self):
        """The mirror check, so this fix is not applied in one direction only."""
        c = numerics.check_build(_NoCudaBuild())
        for a in c.advisories:
            if "no CPU extension is present" in a:
                assert a in c.host_advisories


class TestHostFlagsAreJudgedOnTheRightArtifact:

    def test_cpu_host_flags_come_from_the_cpu_stamp(self, monkeypatch):
        c = numerics.check_build(_NoCudaBuild())
        # The CUDA stamp records nothing -- that is what _NoCudaBuild means.
        assert c.checked_host_flags is False

        # Asserting `checked_cpu_host_flags == checked_a_cpu_build` would be
        # TRUE BY CONSTRUCTION -- both are the same `with_cpu` variable in
        # `check_build`'s return -- so it would stay green under either future
        # breakage. Found in review. Drive the CPU stamp instead.
        monkeypatch.setattr(
            numerics, "_cpu_flag_advisories", lambda: (False, []))
        assert numerics.check_build(_NoCudaBuild()).checked_cpu_host_flags \
            is False
        monkeypatch.setattr(
            numerics, "_cpu_flag_advisories", lambda: (True, []))
        assert numerics.check_build(_NoCudaBuild()).checked_cpu_host_flags \
            is True

    def test_a_present_but_unstamped_cpu_extension_is_refused(self,
                                                              monkeypatch):
        """The gate must mean what its own refusal text says.

        `ctx.checked_a_cpu_build` was wired from `build.with_cpu`, which is
        PRESENCE, while the refusal it guards says the flags "were never read".
        An extension present but carrying no flags satisfied the first and not
        the second, so it was merely advisory-capped under a field whose name
        and refusal both promised more.
        """
        from bench import rules

        # A CPU extension that exists but records no compiler flags.
        monkeypatch.setattr(
            numerics, "_cpu_flag_advisories",
            lambda: (False, ["a CPU extension is present but its stamp "
                             "records no compiler flags"]))
        check = numerics.check_build(_NoCudaBuild())
        assert check.checked_a_cpu_build is False

        ctx = rules.RunContext.from_build(
            _fake_cpu_verified_build(), check, backend_name="cpu",
            **_runtime())
        assert ctx.checked_a_cpu_build is False, (
            "the context must carry the check's meaning, not the build's "
            "mere presence flag")
        res = rules.enforce(_req(), ctx)
        assert res.hard_refuse, (
            "a CPU run whose extension's flags were never read must be "
            "refused, which is what the refusal text already claims")


@pytest.mark.skipif(
    not hasattr(numerics.check_build(), "checked_cpu_host_flags"),
    reason="build predates the CPU host-flag split")
class TestACpuRunOnACudaLessTreeCanBeGraded:

    def test_no_cuda_advisory_survives_into_a_cpu_run_context(self):
        from bench import rules

        check = numerics.check_build(_NoCudaBuild())
        build = _fake_cpu_verified_build()
        ctx = rules.RunContext.from_build(
            build, check, backend_name="cpu", **_runtime())
        leaked = [a for a in ctx.numerics_advisories
                  if "no CUDA extension is present" in a]
        assert not leaked, (
            f"a CUDA advisory reached a CPU run's context: {leaked}")
        assert ctx.checked_host_flags is True, (
            "a CPU run with a stamped CPU extension has had its host flags "
            "read; judging it on the absent CUDA stamp is the bug")

    def test_a_gpu_run_still_sees_it(self):
        """The GPU path must not change."""
        from bench import rules

        check = numerics.check_build(_NoCudaBuild())
        build = _fake_cpu_verified_build()
        ctx = rules.RunContext.from_build(
            build, check, backend_name="cuda", **_runtime())
        assert any("no CUDA extension is present" in a
                   for a in ctx.numerics_advisories), (
            "a GPU run on a tree with no CUDA extension must still be told so")


def _req():
    """A compliant timing request, so only the build facts decide the outcome."""
    from bench import rules
    return rules.RunRequest(n_repeats=11, n_warmup=3,
                            decision_grade=True, n_conformers=65536,
                            max_results=100, claim_non_burst=False)


def _runtime():
    """The runtime half of a RunContext, none of which this test is about."""
    return dict(git_dirty=False, gpu_lock_held=True,
                gpu_lock_acquisition_failed=False, other_process_on_gpu=False,
                clocks_locked=False, throttle_fraction=0.0,
                sustained_kernel_s=120.0, measurement_regime="calibration")


def _fake_cpu_verified_build():
    """A `VerifiedBuild` describing an inspected CPU extension, no CUDA."""
    from bench import build_check

    fields = {f.name for f in dataclasses.fields(build_check.VerifiedBuild)}
    # Required first, then the CPU facts this test is about. Built by name so
    # the test does not silently pass a positional into the wrong slot if
    # VerifiedBuild gains a field.
    kw = {"build_type": "Release", "cxx_flags": "none", "cuda_flags": "none",
          "with_cuda": False, "cuda_architectures_requested": "",
          "nvcc_version": "none", "git_sha": "0" * 40}
    for name, value in (("with_cpu", True),
                        ("cpu_src_sha256", "0" * 64), ("cpu_src_dirty", False),
                        ("cpu_src_tree", "0" * 40),
                        ("cpu_build_type", "Release"),
                        ("cpu_cxx_flags",
                         "-O3 -march=x86-64-v3 -fno-fast-math"),
                        ("cpu_selected_target", "X86_64_V3"),
                        ("cpu_compiled_targets", ["X86_64_V3"]),
                        ("cpu_vector_width", 8)):
        if name in fields:
            kw[name] = value
    return build_check.VerifiedBuild(**{k: v for k, v in kw.items()
                                        if k in fields})

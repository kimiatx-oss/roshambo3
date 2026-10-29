"""The CPU shape kernel.

Validated against `roshambo3.reference`, the float64 oracle -- never against
the CUDA kernel. The trap: "C++ and CUDA agree, so the math is right -- they
agree because they share the typo", and a quaternion normalization defect
survived in ROSHAMBO2's two backends for as long as they were each other's only
check.
"""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import backend, numerics, reference

cpu = pytest.importorskip("roshambo3.backend.cpu")
pytestmark = pytest.mark.skipif(not cpu.available(),
                                reason="the CPU kernel is not built")

#: An off-budget (steps, lr_t) pair is refused unless the caller says why.
#: Freezing the optimizer to isolate the volume sums is the legitimate use.
FROZEN = "test: lr=0 freezes the optimizer so volumes are compared at a fixed pose"


def _pack(a, b):
    L = max(a.shape[0], b.shape[0])
    qc = np.zeros((1, L, 4), np.float32)
    qc[0, :a.shape[0], :3] = a
    qc[0, :a.shape[0], 3] = 1.0
    lc = np.zeros((1, L, 4), np.float32)
    lc[0, :b.shape[0], :3] = b
    lc[0, :b.shape[0], 3] = 1.0
    return dict(query_coords=qc,
                query_n_atoms=np.array([a.shape[0]], np.int32),
                library_coords=lc,
                library_counts=np.array([[b.shape[0]]], np.uint8),
                pair_query_row=np.array([0], np.int32),
                pair_library_row=np.array([0], np.int32))


def _pair(seed=11, na=9, nb=11, scale=2.2):
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, scale, size=(na, 3)).astype(np.float32),
            rng.normal(0.0, scale, size=(nb, 3)).astype(np.float32))


class TestTheConstantsCannotDrift:
    def test_the_kernel_carries_the_python_derivation(self):
        """A drift here would scale every score with no other symptom."""
        s = cpu.build_stamp()
        assert np.float32(s["coord_scale"]) == numerics.COORD_SCALE
        assert np.float32(s["shape_prefactor"]) == numerics.SHAPE_PREFACTOR

    def test_it_reports_both_what_it_holds_and_what_it_chose(self):
        """Asking the binary what targets it contains is
        a different question from asking the process what it dispatched to, and
        a decision-grade CPU number is refused without the second."""
        s = cpu.build_stamp()
        assert s["compiled_targets"]
        assert s["selected_target"] in s["compiled_targets"]

    def test_the_stamp_names_the_isa_and_the_width_and_the_flags(self):
        """Every CPU run record carries the ISA and the vector width, exactly as
        it carries `cuda_architectures`, and a decision-grade number is refused
        without them.

        The flags are asserted alongside because they are where CONTRACTION is
        determined, and contraction is settled by recording it rather than
        by pinning it -- so a stamp that omits them leaves nothing to record it.
        That failed once for real: `ROSHAMBO3_CXX_FLAGS` was built from the
        GLOBAL CMake flags, so a binary compiled at `-march=x86-64-v3` stamped
        " -O3 -DNDEBUG" and did not mention the flag that changed its
        arithmetic."""
        s = cpu.build_stamp()
        assert isinstance(s["vector_width"], int) and s["vector_width"] >= 1
        assert s["cxx_flags"].strip(), "a stamp with no flags records no contraction"
        assert "-fno-fast-math" in s["cxx_flags"]
        assert "-fno-associative-math" in s["cxx_flags"], (
            "this is what keeps the accumulation in source order, which is what "
            "lets this kernel be the reference anchor")


class TestAgainstTheFloat64Oracle:
    def test_self_overlap_sums_over_ordered_pairs(self):
        """Two coincident atoms give FOUR times one atom, not twice --
        `reference.self_overlap` says so and summing i<j is a quiet way to be
        wrong by 2x."""
        a, b = _pair()
        s = cpu.score_pairs(steps=1, lr_q=0.0, lr_t=0.0, start_mode=0,
                            budget_override=FROZEN, **_pack(a, b))
        assert abs(s[0, 5] - reference.self_overlap(a)) / reference.self_overlap(a) < 1e-5
        assert abs(s[0, 6] - reference.self_overlap(b)) / reference.self_overlap(b) < 1e-5

    def test_the_pair_sum_matches_at_the_reported_pose(self):
        """Also the check that the reported pose reproduces the reported score
        -- getting the pose convention backwards would write out wrong poses
        while every score stayed right."""
        a, b = _pair(seed=12)
        s = cpu.score_pairs(steps=1, lr_q=0.0, lr_t=0.0, start_mode=0,
                            budget_override=FROZEN, **_pack(a, b))
        q, t = s[0, 9:13].astype(np.float64), s[0, 13:16].astype(np.float64)
        want = reference.pair_overlap(a, reference.apply_pose(b, q, t))
        assert abs(s[0, 3] - want) / want < 1e-5

    def test_the_tanimoto_is_the_grant_pickup_one(self):
        a, b = _pair(seed=13)
        s = cpu.score_pairs(steps=1, lr_q=0.0, lr_t=0.0, start_mode=0,
                            budget_override=FROZEN, **_pack(a, b))
        want = reference.tanimoto(float(s[0, 3]), float(s[0, 5]), float(s[0, 6]))
        assert abs(s[0, 1] - want) < 1e-6

    def test_a_conformer_scored_against_itself_reaches_one(self):
        a, _ = _pair(seed=14)
        s = cpu.score_pairs(steps=1, lr_q=0.0, lr_t=0.0, start_mode=0,
                            budget_override=FROZEN, **_pack(a, a))
        assert abs(s[0, 1] - 1.0) < 1e-5


class TestTheInvariantsThatShareNoCodeWithIt:
    def test_check_result_invariants_passes(self):
        """Unit pose quaternion and Cauchy-Schwarz on both channels. It shares
        no code with either kernel, which is the entire point of it."""
        a, b = _pair(seed=15, na=18, nb=20)
        s = cpu.score_pairs(steps=25, start_mode=1, **_pack(a, b))
        assert numerics.check_result_invariants(s) == []

    def test_the_pose_quaternion_is_unit(self):
        a, b = _pair(seed=16)
        s = cpu.score_pairs(steps=25, start_mode=1, **_pack(a, b))
        assert abs(np.linalg.norm(s[0, 9:13]) - 1.0) < 1e-5


class TestTheSearch:
    def test_more_steps_never_makes_it_worse(self):
        """The sign check that matters: if the gradient or the update were
        inverted the Tanimoto would FALL with more steps, and every fixed-pose
        test above would still pass."""
        a, b = _pair(seed=17, na=16, nb=18)
        got = [float(cpu.score_pairs(steps=n, start_mode=1, **_pack(a, b))[0, 1])
               for n in (1, 5, 25)]
        assert got[1] > got[0] and got[2] >= got[1] - 1e-6, got

    def test_four_starts_are_at_least_as_good_as_one(self):
        a, b = _pair(seed=18, na=16, nb=18)
        one = cpu.score_pairs(steps=25, start_mode=0, **_pack(a, b))[0, 1]
        four = cpu.score_pairs(steps=25, start_mode=1, **_pack(a, b))[0, 1]
        assert four >= one - 1e-6

    def test_the_start_modes_are_the_canonical_counts(self):
        for mode in (0, 1, 2):
            a, b = _pair(seed=19)
            cpu.score_pairs(steps=25, start_mode=mode, **_pack(a, b))
        with pytest.raises(Exception):
            cpu.score_pairs(steps=25, start_mode=3, **_pack(*_pair(seed=19)))

    def test_it_is_deterministic(self):
        a, b = _pair(seed=20, na=16, nb=18)
        kw = _pack(a, b)
        first = cpu.score_pairs(steps=25, start_mode=1, **kw)
        second = cpu.score_pairs(steps=25, start_mode=1, **kw)
        assert np.array_equal(first, second)


class TestTheBoundaryRefusesWhatItCannotHonor:
    """An ignored knob is a silent wrong answer. Half of what
    `ScreenConfig.for_width()` emits is about shared memory, warps and
    carveouts, and none of it means anything here."""

    @pytest.mark.parametrize("knob,value", [
        ("smem_carveout", 25), ("min_blocks_per_sm", 4), ("stage_width", 256),
        ("coord_residency", 0), ("lanes_per_group", 4),
        ("start_groups_per_row", 1), ("library_tile_mode", 1),
    ])
    def test_a_gpu_only_knob_is_refused_by_name(self, knob, value):
        a, b = _pair()
        with pytest.raises(Exception) as exc:
            cpu.score_pairs(steps=25, **{knob: value}, **_pack(a, b))
        assert knob in str(exc.value)

    def test_color_is_refused_rather_than_silently_shape_only(self):
        a, b = _pair()
        with pytest.raises(ValueError, match="requires query_types"):
            cpu.score_pairs(steps=25, color=True, mixing=0.5, **_pack(a, b))

    def test_a_device_topk_handle_is_refused_with_the_alternative_named(self):
        a, b = _pair()
        with pytest.raises(NotImplementedError, match="HostTopK"):
            cpu.score_pairs(steps=25, topk=object(), **_pack(a, b))

    def test_an_out_of_range_pair_index_is_refused(self):
        a, b = _pair()
        kw = _pack(a, b)
        kw["pair_library_row"] = np.array([7], np.int32)
        with pytest.raises(ValueError, match="pair_library_row"):
            cpu.score_pairs(steps=25, **kw)


class TestTheD019CouplingIsEnforcedHereToo:
    def test_omitting_lr_t_derives_it(self):
        a, b = _pair()
        cpu.score_pairs(steps=25, **_pack(a, b))       # no lr_t: derived

    def test_an_off_budget_pair_is_refused_without_a_reason(self):
        """A step-count change that does not carry its lr_t is not a speedup,
        it is a silent accuracy loss that looks exactly like one."""
        a, b = _pair()
        with pytest.raises(ValueError, match="budget"):
            cpu.score_pairs(steps=25, lr_t=0.1, **_pack(a, b))

    def test_a_stated_reason_admits_it(self):
        a, b = _pair()
        cpu.score_pairs(steps=25, lr_t=0.1, budget_override="test", **_pack(a, b))


class TestItIsReachableThroughTheSeam:
    def test_select_resolves_it(self):
        assert backend.select("cpu").__name__ == "roshambo3.backend.cpu"

    def test_it_satisfies_the_seam(self):
        mod = backend.select("cpu")
        for name in ("score_pairs", "available", "extension"):
            assert hasattr(mod, name), name

"""The exponential map on the production field kernel: the default, and its two
structural claims.

`optimize_pairs_field_kernel` takes the `OPT` template parameter, and
 `UpdateRule.ADAGRAD_EXP` the default on both kernels. Two things about
that change are *structural* rather than statistical, so they are asserted here
rather than measured somewhere and quoted:

  1. **`||q||` stays unit BY CONSTRUCTION.** The exp map composes on the
     manifold, `q <- exp(d) (x) q`, so the pose is a product of unit quaternions
     and the only deviation is fp32 rounding of the composition. Measured:
     `max | ||q|| - 1 | = 2.9e-6` at 100 steps against a 1e-5 gate; that gate is
     the one used below.

  2. **The renormalization the additive rule needs is gone.** Not "cheap" --
     absent. The bug class here is a renormalization that is subtly the wrong one,
     and the way not to have that bug is not to need the step. So the source is
     checked: every quaternion renormalization left in either kernel must sit
     inside an `OPT == UpdateRule::AdagradQAdd` guard.

**The start mode is varied throughout.** The original defect was found only
because it was, and a check that runs at one start mode is the trap, not the
test. The input is the seeded fixture from `bench/make_golden.py`, so this file
needs no corpus.

Nothing here is an accuracy claim. Recall at the shipped configuration is a
measured tie -- 0.9852 additive against 0.9851 exp map -- and the production
per-step cost is deliberately not measured here.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from roshambo3 import scores as S
from roshambo3.backend import cuda as backend
from roshambo3.backend.cuda import PoseSearch, UpdateRule

pytestmark = pytest.mark.skipif(not backend.available(),
                                reason="no CUDA extension in this build")

START_MODES = (0, 1, 2)

#: The gate on the composition's fp32 drift, measured at 100 steps.
UNIT_NORM_GATE = 1e-5

CUDA_SRC = (pathlib.Path(__file__).resolve().parent.parent
            / "src/roshambo3/backend/_cuda_src/kernels.cu")


@pytest.fixture(scope="module")
def seeded_pairs():
    """The `bench/make_golden.py` input, laid out for `score_pairs`.

    Reused rather than re-invented: it is generated from a fixed seed, it is
    molecule-shaped rather than uniform noise (so the optimizer has a basin to
    find), and it needs no corpus, which is what lets this file run anywhere.
    """
    import bench.make_golden as mk
    from roshambo3.layout import tile32

    coords, n_real = mk.make_input()
    nq, L = mk.N_QUERIES, mk.L
    qc = np.ascontiguousarray(coords[:nq])
    qn = np.ascontiguousarray(n_real[:nq])
    tc = tile32(np.ascontiguousarray(coords[nq:]))
    tn = np.ascontiguousarray(n_real[nq:])
    n_lib = mk.N_LIBRARY
    return dict(
        query_coords=qc,
        query_types=np.zeros((nq, L), np.int32),
        query_n_atoms=qn,
        library_coords=tc,
        # A shape-only library's count vector is one column -- the real
        # atom count -- and there is no type plane at all.
        library_counts=tn.astype(np.uint8).reshape(-1, 1),
        pair_query_row=np.repeat(np.arange(nq, dtype=np.int32), n_lib),
        pair_library_row=np.tile(np.arange(n_lib, dtype=np.int32), nq),
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.zeros((1, 1), np.float32),
    )


def _run(seeded_pairs, **kw):
    from roshambo3.backend import score_pairs

    # steps and lr_t are ONE knob; 100 steps is where the drift gate was
    # measured, so the norm check runs at the step count its gate belongs to.
    args = dict(steps=100, lr_t=0.2, lr_q=0.1, pose_search=PoseSearch.FIELD)
    args.update(kw)
    return score_pairs(**seeded_pairs, **args)


def _norm_error(v):
    q = v[:, S.POSE_QUAT].astype(np.float64)
    return np.abs(np.linalg.norm(q, axis=1) - 1.0)


class TestTheQuaternionStaysUnitByConstruction:
    """Claim 1. Asserted at every start mode, including 0, where the additive
    rule's shrinkage is not masked by the four pi-flip starts."""

    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_exp_map_holds_the_unit_norm_without_renormalizing(
            self, seeded_pairs, start_mode):
        v = _run(seeded_pairs, start_mode=start_mode,
                 update_rule=UpdateRule.ADAGRAD_EXP)
        err = _norm_error(v)
        assert err.max() < UNIT_NORM_GATE, (
            f"max | ||q|| - 1 | = {err.max():.3e} at start_mode={start_mode}, "
            f"over the {UNIT_NORM_GATE:.0e} "
            f"gate. The exp map's pose is a product of unit quaternions, so an "
            f"excursion here is a composition bug, not drift.")

    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_the_drift_is_nonzero_so_the_gate_is_measuring_something(
            self, seeded_pairs, start_mode):
        """A gate no value has ever approached certifies nothing.

        fp32 rounding of `nsteps` quaternion products cannot come out exactly
        zero over 2,048 pairs; if it did, either the optimizer took no steps or
        something normalized the result, and both would make the assertion above
        vacuous.
        """
        v = _run(seeded_pairs, start_mode=start_mode,
                 update_rule=UpdateRule.ADAGRAD_EXP)
        err = _norm_error(v)
        assert err.max() > 0.0, (
            "every returned pose is exactly unit norm -- the composition is "
            "either not running or is being renormalized")


class TestTheRenormalizationIsGoneOnTheExpPath:
    """Claim 2, checked at the source, because "nothing calls it" is a statement
    about the code and not about any output.

    A behavioral check cannot see this: a renormalization applied to a
    quaternion that is already unit is very nearly the identity, so it would
    leave the numbers above looking exactly the same while reintroducing the step
    that bug class lives in.
    """

    def _kernel_source(self):
        if not CUDA_SRC.exists():
            pytest.skip("kernels.cu not present in this checkout")
        return CUDA_SRC.read_text()

    def test_every_quaternion_renormalization_is_guarded_by_the_additive_rule(self):
        src = self._kernel_source()
        # The renormalization's signature is a magnitude over all four
        # quaternion components. Both kernels write it the same way.
        pattern = re.compile(
            r"sqrt\(\s*(\w+)\[0\]\*\1\[0\]\s*\+\s*\1\[1\]\*\1\[1\]"
            r"\s*\+\s*\1\[2\]\*\1\[2\]\s*\+\s*\1\[3\]\*\1\[3\]\s*\)")
        hits = list(pattern.finditer(src))
        assert hits, ("no quaternion renormalization found at all -- this test "
                      "has stopped matching the source it is checking")
        for m in hits:
            # The guard must be the nearest enclosing condition, so look back a
            # short way rather than anywhere in the file.
            before = src[max(0, m.start() - 700):m.start()]
            assert "OPT == UpdateRule::AdagradQAdd" in before, (
                f"the renormalization at offset {m.start()} is not inside an "
                f"`OPT == UpdateRule::AdagradQAdd` guard, so the exponential "
                f"map is still paying for a correction it does not need -- and "
                f"a correction applied to a unit quaternion is silently a "
                f"different optimizer.")

    def test_the_exp_map_composes_through_exactly_one_implementation(self):
        """Pitfall 5: two copies of a quaternion update agree because they share
        a typo. There is one definition of the composition in this file, and both
        the per-lane and the distributed update call it."""
        src = self._kernel_source()
        assert len(re.findall(r"__device__[^\n]*\bquat_left_exp\b", src)) == 1
        # `apply_update` (UPD 0) and `apply_update_lanes` (UPD 2/3) are the two
        # callers, and neither may spell the product out again.
        assert src.count("quat_left_exp(") == 3, (
            "the number of `quat_left_exp` mentions changed: expected one "
            "definition and two call sites (apply_update, apply_update_lanes)")


class TestTheFieldPathHonorsTheRuleItIsGiven:

    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_the_two_rules_do_not_return_the_same_bits(self, seeded_pairs,
                                                       start_mode):
        """A rule that is accepted and then ignored is the defect the previous
        refusal existed to prevent, wearing a different hat."""
        a = _run(seeded_pairs, start_mode=start_mode,
                 update_rule=UpdateRule.ADAGRAD_QADD)
        b = _run(seeded_pairs, start_mode=start_mode,
                 update_rule=UpdateRule.ADAGRAD_EXP)
        assert not np.array_equal(a, b), (
            "the exponential map returned bit-identical scores to the additive "
            "step on the field path -- it was accepted and then ignored")
        # Two parameterizations of one search: nearby poses, not equal ones.
        ta, tb = a[:, S.TANIMOTO_SHAPE], b[:, S.TANIMOTO_SHAPE]
        assert np.nanmax(np.abs(ta - tb)) < 0.35, (
            "the two parameterizations should converge to nearby poses")

    @pytest.mark.parametrize("rule", [UpdateRule.ADAM_EXP, UpdateRule.QN2_EXP,
                                      UpdateRule.BFGS_EXP])
    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_the_analytic_only_rules_are_refused_by_name(self, seeded_pairs,
                                                         rule, start_mode):
        """Refused, not ignored, and refused the same way at every start mode.

        Adam, QN2 and BFGS carry 8, 12 and 33 floats of per-pose auxiliary state
        against a kernel pinned at 64 registers per thread, and none of them is a
        shipping rule. Accepting the argument and scoring with a different
        optimizer is the silent wrong answer this project keeps guarding against.
        """
        with pytest.raises(Exception) as e:
            _run(seeded_pairs, start_mode=start_mode, update_rule=rule)
        msg = str(e.value)
        assert "field path" in msg
        assert "ADAGRAD_EXP" in msg or "update_rule 1" in msg


class TestTheDefaultIsTheExponentialMapEverywhere:
    """Three defaults had to move together, and the one that decides is not the
    one it looks like.

    `optimizer.DEFAULT_OPT_MODE` is dead for engine callers:
    `ScreenConfig.resolved_optimizer()` always puts `opt_mode` in the settings
    dict explicitly, so the module-level default never fills it in. Flipping one
    and not the other would leave the shipped default unchanged in practice,
    which is a change that reads as landed and is not.
    """

    def test_the_module_default(self):
        from roshambo3 import optimizer

        assert optimizer.DEFAULT_OPT_MODE == int(UpdateRule.ADAGRAD_EXP)
        assert (optimizer.default_optimizer_settings()["opt_mode"]
                == int(UpdateRule.ADAGRAD_EXP))

    def test_the_engine_default_that_actually_reaches_the_kernel(self):
        from roshambo3.engine import ScreenConfig

        cfg = ScreenConfig()
        assert cfg.update_rule is UpdateRule.ADAGRAD_EXP
        assert (cfg.resolved_optimizer()["opt_mode"]
                == int(UpdateRule.ADAGRAD_EXP))

    def test_the_backend_entry_default(self, seeded_pairs):
        """`score_pairs` is a third default, and the one every bench script
        inherits. It must agree with the other two, or one settings dict means
        different arithmetic depending on which door it came through."""
        import inspect

        from roshambo3.backend import score_pairs

        sig = inspect.signature(score_pairs)
        assert (sig.parameters["update_rule"].default
                is UpdateRule.ADAGRAD_EXP)

    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_the_default_launch_computes_the_exp_map(self, seeded_pairs,
                                                    start_mode):
        """The end-to-end version of the three assertions above: an unspecified
        rule must produce the exp map's bits, not the additive rule's."""
        default = _run(seeded_pairs, start_mode=start_mode)
        exp = _run(seeded_pairs, start_mode=start_mode,
                   update_rule=UpdateRule.ADAGRAD_EXP)
        assert np.array_equal(default, exp)

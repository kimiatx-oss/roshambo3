"""Which update rules each scoring path actually honors.

Two kernels ship. Only one of them carries an optimizer mode:

  * the **analytic** path has an ``OPT`` template parameter, instantiated for
    all five modes on the shipped cell and for mode 0 everywhere else;
  * the **field** (grid) path has none. ``optimize_pairs_field`` does not take
    ``update_rule`` as an argument at all.

Asking the field path for anything but mode 0 is refused, in
``backend/cuda.py`` and again in ``interface.cpp`` for a caller who reaches the
extension directly. Neither refusal had a test. These add one, because the
alternative -- accepting the argument and scoring with a different optimizer --
is exactly the silent wrong answer the start-rotation guard exists to prevent,
and an untested refusal is one refactor away from being that.

The start mode is varied throughout: the refusal must not depend on it, and a
check that only ever runs at the default start mode can hide a defect.
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.backend import cuda as backend

pytestmark = pytest.mark.skipif(not backend.available(),
                                reason="no CUDA extension in this build")

START_MODES = (0, 1, 2)


@pytest.fixture()
def two_queries(smoke_corpus):
    import roshambo3 as r3
    from roshambo3.engine import from_arrays
    cat = r3.Catalog(smoke_corpus)
    gids = np.array([0, 1])
    return from_arrays(cat.coordinates(gids), ["q0", "q1"])


class TestFieldPathRefusesOtherRules:
    """Rules 2-4 stay refused on the field path; rule 1 no longer is.

    Narrowed when the exponential map became the shipped rule on **both**
    kernels. `ADAGRAD_EXP` (1) is now instantiated on the field path, so asserting
    it is refused would assert the old behavior and fail. `ADAM_EXP`,
    `QN2_EXP` and `BFGS_EXP` (2-4) are still analytic-only -- reachable through
    `roshambo3.experimental`, not through `ScreenConfig` -- and their refusal is
    what keeps one settings dict from meaning different arithmetic depending on
    which kernel it reaches.

    The companion positive case is `TestTheFieldPathNowRunsTheExponentialMap`
    below: a rule that is accepted and then ignored is the same defect wearing a
    different hat.
    """

    @pytest.mark.parametrize("rule", [2, 3, 4])
    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_refused(self, smoke_corpus, two_queries, rule, start_mode):
        from roshambo3.backend.cuda import PoseSearch, UpdateRule
        from roshambo3.engine import Engine, R3gSource, ScreenConfig, TopK

        with pytest.raises(Exception) as e:
            Engine(device_id=0).run(
                R3gSource(smoke_corpus), two_queries,
                ScreenConfig(pose_search=PoseSearch.FIELD,
                             start_mode=start_mode,
                             update_rule=UpdateRule(rule)),
                TopK(k=5))
        msg = str(e.value)
        assert "field path" in msg
        assert "ADAGRAD_QADD" in msg or "update_rule 0" in msg

    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_mode_zero_still_runs(self, smoke_corpus, two_queries, start_mode):
        """The refusal must not have closed the door the production path uses."""
        from roshambo3.backend.cuda import PoseSearch, UpdateRule
        from roshambo3.engine import Engine, R3gSource, ScreenConfig, TopK

        res = Engine(device_id=0).run(
            R3gSource(smoke_corpus), two_queries,
            ScreenConfig(pose_search=PoseSearch.FIELD, start_mode=start_mode,
                         update_rule=UpdateRule.ADAGRAD_QADD),
            TopK(k=5))
        assert res is not None


class TestTheFieldPathNowRunsTheExponentialMap:
    """The other half of the narrowing above.

    An update-rule template parameter was added to `optimize_pairs_field_kernel`,
    which previously inlined the additive Adagrad step and had no `OPT` at all.
    So this asserts two things the refusal test can no longer cover: that rule 1
    is reachable on the production path, and that it is not silently ignored once
    accepted.
    """

    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_the_exp_map_runs_on_the_field_path(self, smoke_corpus, two_queries,
                                                start_mode):
        from roshambo3.backend.cuda import PoseSearch, UpdateRule
        from roshambo3.engine import Engine, R3gSource, ScreenConfig, TopK

        res = Engine(device_id=0).run(
            R3gSource(smoke_corpus), two_queries,
            ScreenConfig(pose_search=PoseSearch.FIELD, start_mode=start_mode,
                         update_rule=UpdateRule.ADAGRAD_EXP),
            TopK(k=5))
        assert res is not None

    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_it_is_not_a_no_op_on_the_field_path(self, smoke_corpus, two_queries,
                                                 start_mode):
        """A different optimizer must produce a different trajectory.

        Not an accuracy claim -- recall measured as a tie at the shipped
        configuration. This only pins that the parameter is *wired*: a
        rule accepted and then discarded would pass every test above.
        """
        import numpy as np
        from roshambo3.backend.cuda import PoseSearch, UpdateRule
        from roshambo3.engine import DenseScores, Engine, R3gSource, ScreenConfig

        def run(rule):
            sink = DenseScores()
            Engine(device_id=0).run(
                R3gSource(smoke_corpus), two_queries,
                ScreenConfig(pose_search=PoseSearch.FIELD, start_mode=start_mode,
                             update_rule=rule, steps=25),
                sink)
            # finalize() returns a DenseResult; .scores is the [nq, n, 20] block.
            return np.asarray(sink.finalize().scores)

        add = run(UpdateRule.ADAGRAD_QADD)
        exp = run(UpdateRule.ADAGRAD_EXP)
        assert add.shape == exp.shape
        assert not np.array_equal(add, exp), \
            "the field path accepted ADAGRAD_EXP and computed the additive step"


class TestAnalyticPathHonorsTheExponentialMap:
    @pytest.mark.parametrize("start_mode", START_MODES)
    def test_exp_map_runs_and_changes_the_answer(self, smoke_corpus, two_queries,
                                                 start_mode):
        """Mode 1 must be reachable, and it must not be a no-op.

        A mode that is accepted and then ignored is the defect above wearing a
        different hat, so this asserts the scores actually move. It is not an
        accuracy claim -- the recall question was measured as a tie.
        """
        from roshambo3.backend.cuda import PoseSearch, UpdateRule
        from roshambo3.engine import DenseScores, Engine, R3gSource, ScreenConfig

        from roshambo3 import scores as S

        def run(rule):
            return Engine(device_id=0).run(
                R3gSource(smoke_corpus), two_queries,
                ScreenConfig(pose_search=PoseSearch.ANALYTIC,
                             start_mode=start_mode, update_rule=rule),
                DenseScores())[1]

        a = run(UpdateRule.ADAGRAD_QADD).scores[:, :, S.TANIMOTO_COMBINATION]
        b = run(UpdateRule.ADAGRAD_EXP).scores[:, :, S.TANIMOTO_COMBINATION]
        assert a.shape == b.shape
        # Two different parameterizations of the same search: the converged
        # scores are close but not equal. Equality here would mean mode 1 was
        # accepted and ignored, which is the defect this file is about.
        assert not np.array_equal(a, b), (
            "the exponential map returned bit-identical scores to the shipped "
            "Adagrad step -- it was accepted and then ignored")
        assert np.nanmax(np.abs(a - b)) < 0.2, (
            "the two parameterizations should converge to nearby poses")

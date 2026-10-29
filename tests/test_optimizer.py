"""The step count and the translation learning rate are ONE knob.

The finding this module exists to enforce: 25 steps at lr_t 0.4 BEATS 100 steps
at lr_t 0.1 (recall@N' 0.9851 vs 0.9568) at 1.95x the throughput -- but 25 steps
at the *old* lr_t 0.1 reaches only 1.5 A, below the shipped kernel's 2.0 A, and
is a silent accuracy regression that looks exactly like a speedup.
"""

from __future__ import annotations

import math

import pytest

from roshambo3 import optimizer as O


class TestBudgetArithmetic:
    def test_travel_sum_is_the_adagrad_partial_sum(self):
        for n in (1, 5, 25, 100):
            assert math.isclose(O.adagrad_travel_sum(n),
                                sum(k ** -0.5 for k in range(1, n + 1)), rel_tol=1e-12)

    def test_budget_is_linear_in_the_learning_rate(self):
        assert math.isclose(O.translation_budget(0.4, 25),
                            2 * O.translation_budget(0.2, 25), rel_tol=1e-12)

    def test_derived_lr_lands_inside_the_accepted_band(self):
        lo, hi = O.BUDGET_BAND_A
        for n in (10, 25, 50, 100, 200):
            assert lo <= O.translation_budget(O.lr_t_for_steps(n), n) <= hi

    def test_the_shipped_roshambo2_pair_is_off_budget(self):
        """This is the finding, not a bug: 100 steps at lr_t 0.1 buys 1.86 A
        while the reference optimum's own |t| distribution has p90 = 2.67 A, so
        22.7% of pairs want to travel further than the optimizer can go."""
        assert O.translation_budget(0.1, 100) < O.BUDGET_BAND_A[0]


class TestResolve:
    def test_defaults_are_on_the_budget_curve(self):
        s = O.default_optimizer_settings()
        lo, hi = O.BUDGET_BAND_A
        assert lo <= O.translation_budget(s["lr_t"], s["steps"]) <= hi

    def test_steps_alone_derives_the_learning_rate(self):
        """The coupling carried by construction: there is nothing to forget."""
        s = O.resolve_optimizer_settings({"steps": 50})
        assert math.isclose(s["lr_t"], O.lr_t_for_steps(50), rel_tol=1e-12)

    def test_a_learning_rate_without_a_step_count_is_refused(self):
        """A learning rate has no meaning without the step count whose budget
        it was set for."""
        with pytest.raises(Exception):
            O.resolve_optimizer_settings({"lr_t": 0.4})

    def test_the_silent_regression_is_refused(self):
        """25 steps at the old lr_t 0.1 is a 0.86 A budget -- worse than the
        100-step kernel it appears to accelerate."""
        with pytest.raises(Exception):
            O.resolve_optimizer_settings({"steps": 25, "lr_t": 0.1})

    def test_the_historical_control_is_grandfathered(self):
        """Dozens of committed harnesses reproduce (100, 0.1). It is off-budget
        -- that is the finding -- so it is permitted by name rather than by
        silently widening the band."""
        s = O.resolve_optimizer_settings({"steps": 100, "lr_t": 0.1})
        assert s["steps"] == 100 and s["lr_t"] == 0.1

    def test_an_off_curve_pair_can_be_forced_only_with_a_stated_reason(self):
        with pytest.raises(Exception):
            O.resolve_optimizer_settings({"steps": 40, "lr_t": 0.9})
        s = O.resolve_optimizer_settings(
            {"steps": 40, "lr_t": 0.9, "budget_override": "measuring the band edge"})
        assert s["lr_t"] == 0.9

    def test_the_resolved_settings_report_the_budget_they_bought(self):
        s = O.resolve_optimizer_settings({"steps": 25})
        assert math.isclose(s["translation_budget_A"],
                            O.translation_budget(s["lr_t"], 25), rel_tol=1e-12)

    def test_default_update_rule_is_the_shipped_one(self):
        """opt_mode 1 -- the exponential map -- everywhere.

        The assertion this replaces read `== 0`, and the reason it could is that
        the field kernel had no OPT template parameter at all, so mode 1 would
        have meant different arithmetic depending on which kernel a settings dict
        reached. The field kernel now has that parameter, so the dict means one
        thing again and the default moved to the exponential map. The invariant
        the old assertion was protecting -- one dict, one arithmetic -- is
        unchanged and is what `tests/test_exp_map_field_cuda.py` checks across all
        three defaults that carry it.
        """
        assert O.resolve_optimizer_settings({})["opt_mode"] == 1
        assert O.DEFAULT_OPT_MODE == 1

    def test_the_additive_rule_is_still_reachable_by_asking_for_it(self):
        """Flipping a default must not remove the control. `bench/xcheck_roshambo2.py`
        and the committed golden fixture both pin mode 0, so it stays a first-class
        request rather than becoming a legacy path."""
        assert O.resolve_optimizer_settings({"opt_mode": 0})["opt_mode"] == 0

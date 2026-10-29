"""Optimizer defaults, and the one coupling that must not be broken.

One measured finding is load-bearing behind this module. In one line:

    the step count and the translation learning rate are ONE knob, not two.

The optimizer's translation step is Adagrad's, so with a roughly constant
gradient its magnitude at step ``k`` is ``lr_t/sqrt(k)`` and the total distance
it can travel per coordinate over ``n`` steps is

    budget(lr_t, n) = lr_t * sum_{k<=n} k**-0.5

That integral -- not ``lr_t``, and not ``steps`` -- is the physical quantity.
The reference optimum's own ``|t|`` distribution (median 0.99 A,
p90 2.67 A, p99 5.15 A) was measured; the shipped ``lr_t = 0.1`` at 100 steps buys 1.86 A,
so **22.7% of pairs want to travel further than the optimizer can go**.

What that costs on ranked output was then measured, and the result is the
reason this module exists rather than a docstring somewhere:

  * 25 steps with ``lr_t = 0.4`` BEATS the shipped 100 steps with ``lr_t = 0.1``
    -- recall@N' 0.9851 against 0.9568 on a 96-query x 9,994,493-conformer
    screen -- at 1.95x the end-to-end throughput.
  * 25 steps with the shipped ``lr_t = 0.1`` has a 0.86 A budget and a measured
    reachable displacement of 1.5 A, **below the shipped kernel's 2.0 A**. It is
    a regression, and it is most of an apparent "25 steps gives up 1.4e-2 of mean
    Tanimoto" loss.
  * Raising ``lr_q`` instead of ``lr_t`` is a regression too (0.9308 -> 0.9217).
    The rotational budget was never the binding one.

So a step-count change that does not carry its ``lr_t`` is not a speedup, it is
a silent accuracy loss -- and it looks exactly like a speedup. This module makes
that shape of change fail loudly:

  * give ``steps`` and omit ``lr_t``  -> ``lr_t`` is DERIVED. The coupling is
    carried by construction and there is nothing to forget.
  * give ``lr_t`` and omit ``steps``  -> error. A learning rate has no meaning
    without the step count whose budget it was set for.
  * give both, off the budget curve   -> error, naming the on-curve value,
    unless the caller states a reason in ``'budget_override'``.

What this module does **not** change is the update rule, but the rule it hands
out did change: ``opt_mode`` defaults to **1**, the exponential map, on every
path. See ``DEFAULT_OPT_MODE`` for the argument, for the measured recall tie
that makes the flip safe, and for what is still open.

The one grandfathered point is the earlier shipped pair (100 steps, lr_t 0.1).
It is off-budget -- that is the finding -- but it is the control that dozens of
committed harnesses reproduce, so it is permitted without ceremony and named
here rather than special-cased silently.
"""

import math

# --------------------------------------------------------------------------
# The budget constants. Changing any of these changes the shipped answer.
# --------------------------------------------------------------------------

#: The translation travel budget, in Angstrom, that the learning rate is set to
#: hold. **Fitted on one tuning corpus and on nothing else.** It is a property
#: of how far the molecules in a corpus actually have to move, so re-derive it
#: on other corpora before treating it as universal, and re-derive it if `prepare`'s centroid alignment
#: ever changes. This is the single place to change when that happens.
TRANSLATION_BUDGET_A = 4.0

#: The band of implied budgets actually measured as near-optimal, across a
#: 10x range of step counts (10 to 200). A pair outside this band is not
#: refused because it is wrong -- it is refused because it is un-budgeted, and
#: the caller has to say that they meant it.
BUDGET_BAND_A = (3.0, 5.8)

#: The earlier shipped configuration: 100 steps at lr_t = 0.1, a 1.86 A budget.
#: Off-band by 1.6x, permitted anyway, because it is the control every prior
#: measurement in this project is quoted against.
LEGACY_SHIPPED_STEPS = 100
LEGACY_SHIPPED_LR_T = 0.1

#: The measured recommendation. 30 steps (lr_t 0.365) is the documented alternative if the T >= 0.7 stratum
#: is judged to matter more than throughput; it is a one-token change here and
#: lr_t follows automatically.
DEFAULT_STEPS = 25
DEFAULT_LR_Q = 0.1

#: ``opt_mode = 1``, the exponential map, IS the default on both shipped
#: kernels. This reverses an earlier additive-rule default, so the reasoning is
#: kept.
#:
#: **Why the exp map.** The additive rule adds the torque to ``q[1:4]`` and
#: renormalizes, which is the rotation the optimizer asked for only at
#: ``q = (1,0,0,0)``. In general the applied rotation is a pose-dependent linear
#: map of the step -- shrunk by ~cos(theta/2) along v, and skewed by a -d x v
#: term (``kernels.cu``, the block over ``quat_left_exp``) -- so the further
#: the pose travels the smaller and more crooked its steps become, while
#: Adagrad's cache keeps accumulating the *nominal* gradient. The exp map
#: composes on the manifold instead, ``q <- exp(d) (x) q``: a uniform tangent
#: step at every pose, unit by construction, with no renormalization to get
#: subtly wrong.
#:
#: **Why flipping is safe.** Recall@N' at the shipped configuration
#: -- 25 steps, ``lr_t`` 0.4, ``lr_q`` 0.1 -- measured **0.9852 (mode 0) against
#: 0.9851 (mode 1)**: a tie, on the primary metric. So the flip is not bought
#: with accuracy, and the recall headline remains the translation budget.
#:
#: **Why it is worth flipping into a tie.** The tie is measured at
#: ``start_mode = 1``, whose four pi-flip starts keep every trajectory near
#: identity -- which is exactly the regime where the additive rule's shrinkage
#: does not bite. Remove that cover and the gap appears: measured
#: **+0.0123 at 50 steps and +0.0148 at 100** at ``start_mode = 0``, growing
#: with step count. A parameterization that only works near identity is a
#: standing constraint on every future start set, and the exp map removes it.
#:
#: **What the previous entry argued, and why it no longer applies.** It objected
#: that modes 1-4 were instantiated for only two kernel arms, so defaulting to 1
#: would make one settings dict mean different arithmetic on different arms and
#: break ``test_sparse_matches_dense``. That test and those arms -- the dense
#: kernel, the threadwise kernel, cooperating-lane arms, layout variants --
#: were never ported to ROSHAMBO3. What remains is two kernels, and the field
#: kernel now has the ``OPT`` template parameter it lacked, so both
#: now carry mode 1 and the dict means one thing.
#:
#: **What is NOT settled.** ``lr_q = 0.1`` is the *additive* rule's swept
#: optimum (under ``start_mode = 0``) and the exp map is deliberately shipped
#: at it, because that is the setting the tie was measured at. The two rules do not share a rotational calibration -- mode 0's applied
#: rotation shrinks as the pose leaves identity, mode 1's does not -- and no
#: rotational budget has been derived. That, and the production per-step cost,
#: remain open; the cost was deliberately not measured before flipping.
DEFAULT_OPT_MODE = 1

#: Rotational learning rate. Held at 0.1: a sweep showed raising it alone is
#: a regression (0.9308 -> 0.9217 at 0.2, -> 0.8756 at 0.4).
_LR_Q_SWEPT_OPTIMUM = 0.1


def adagrad_travel_sum(steps):
    """``sum_{k<=n} k**-0.5`` -- the integral of the Adagrad step-size schedule.

    This is the quantity that has to be held fixed, and the only reason
    ``lr_t`` has a preferred value at all.
    """
    n = int(steps)
    if n < 1:
        raise ValueError("steps must be >= 1, got %r" % (steps,))
    return sum(k ** -0.5 for k in range(1, n + 1))


def translation_budget(lr_t, steps):
    """Total Adagrad translation reach, in Angstrom, for this (lr_t, steps)."""
    return float(lr_t) * adagrad_travel_sum(steps)


def lr_t_for_steps(steps):
    """The budget-matched translation learning rate for ``steps`` steps.

    The closed form, ``lr_t ~= 2.0/sqrt(n)``, which is
    ``TRANSLATION_BUDGET_A / sum_{k<=n} k**-0.5`` to within one grid point of
    the measured argmax at every step count swept. The closed form is used
    rather than the exact quotient because it reproduces the measured cells
    exactly -- 0.2 at 100, 0.3 at 50, 0.4 at 25 -- and those are the numbers the
    recall figures were taken at.
    """
    n = int(steps)
    if n < 1:
        raise ValueError("steps must be >= 1, got %r" % (steps,))
    return 2.0 / math.sqrt(n)


def default_optimizer_settings():
    """A fresh dict of the recommended configuration."""
    return {
        'lr_q': DEFAULT_LR_Q,
        'lr_t': lr_t_for_steps(DEFAULT_STEPS),
        'steps': DEFAULT_STEPS,
        'opt_mode': DEFAULT_OPT_MODE,
    }


def resolve_optimizer_settings(settings, supports_opt_mode=True):
    """Fill in, couple, and check an ``optimizer_settings`` dict.

    Args:
        settings: the caller's dict, or ``None`` for the shipped default.
        supports_opt_mode: whether the kernel this is bound for carries an
            optimizer mode other than 0. Only ``method='blockwise'`` and the
            shipped analytic cell do; every other arm is a measurement
            variant instantiated at the shipped update rule and nothing
            else. When this is False, an **explicitly requested** non-zero
            ``opt_mode`` is an error -- never quietly dropped -- while the
            *default* resolves to 0, because a default that cannot be honored
            is not a default.

    Returns:
        A new dict with 'lr_q', 'lr_t', 'steps', 'opt_mode' and any other keys
        the caller passed through (e.g. 'legacy_combo_gradient'), plus
        'translation_budget_A' recording what the pair actually buys.
    """
    if settings is None:
        settings = {}
    out = dict(settings)

    override = out.pop('budget_override', None)

    has_steps = 'steps' in out and out['steps'] is not None
    has_lr_t = 'lr_t' in out and out['lr_t'] is not None

    if has_lr_t and not has_steps:
        raise ValueError(
            "optimizer_settings gives 'lr_t' but not 'steps'. A "
            "translation learning rate has no meaning on its own: what it buys is "
            "lr_t * sum_{k<=n} k**-0.5, so it is only interpretable against the step "
            "count it was budgeted for. State 'steps' as well, or omit 'lr_t' and let "
            "it be derived.")

    if not has_steps:
        out['steps'] = DEFAULT_STEPS
    out['steps'] = int(out['steps'])

    if not has_lr_t:
        # The coupling, carried by construction: change 'steps' and lr_t moves
        # with it. This is the path a caller should normally be on.
        out['lr_t'] = lr_t_for_steps(out['steps'])
    else:
        out['lr_t'] = float(out['lr_t'])
        _check_budget(out['lr_t'], out['steps'], override)

    out.setdefault('lr_q', DEFAULT_LR_Q)
    out['lr_q'] = float(out['lr_q'])

    explicit_opt = 'opt_mode' in out and out['opt_mode'] is not None
    opt_mode = int(out['opt_mode']) if explicit_opt else DEFAULT_OPT_MODE
    if opt_mode not in (0, 1, 2, 3, 4):
        raise ValueError("optimizer_settings['opt_mode'] must be 0..4, got %r"
                         % (out.get('opt_mode'),))
    if not supports_opt_mode:
        if explicit_opt and opt_mode != 0:
            raise ValueError(
                "optimizer_settings['opt_mode'] = %d was requested, but this kernel "
                "arm is only instantiated for opt_mode 0. Modes 1-4 are compiled for "
                "method='blockwise' and for the shipped analytic cell only. Refusing "
                "rather than scoring with a different optimizer than was asked for."
                % opt_mode)
        opt_mode = 0
    out['opt_mode'] = opt_mode

    out['translation_budget_A'] = translation_budget(out['lr_t'], out['steps'])
    return out


def _check_budget(lr_t, steps, override):
    """Refuse an (lr_t, steps) pair that is not on the budget curve."""
    if steps == LEGACY_SHIPPED_STEPS and lr_t == LEGACY_SHIPPED_LR_T:
        # The earlier shipped control. Off-budget by design; see the module
        # docstring. Permitted so that every committed harness reproducing the
        # shipped configuration keeps running unchanged.
        return

    budget = translation_budget(lr_t, steps)
    lo, hi = BUDGET_BAND_A
    if lo <= budget <= hi:
        return
    if override:
        return

    want = lr_t_for_steps(steps)
    raise ValueError(
        "optimizer_settings pairs steps=%d with lr_t=%g, an Adagrad translation "
        "budget of %.2f A, outside the %.1f-%.1f A band measured as "
        "near-optimal. The budget-matched value at %d steps is "
        "lr_t = %.4g (the rule is lr_t ~= 2.0/sqrt(steps), i.e. %.1f A of travel). "
        "This is refused rather than warned about because the failure it catches -- "
        "cutting the step count and leaving lr_t behind -- looks like a speedup and "
        "measures as an accuracy loss: 25 steps at lr_t 0.1 reaches 1.5 A, less than "
        "the shipped 100-step kernel's 2.0 A. Omit 'lr_t' to have it derived, or set "
        "optimizer_settings['budget_override'] to a string saying why this pair is "
        "meant." % (steps, lr_t, budget, lo, hi, steps, want,
                    TRANSLATION_BUDGET_A))

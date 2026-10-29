"""Measurement arms. Nothing here is a supported interface.

This module exists to be a *place*, and the place matters. ROSHAMBO2's scoring
entry point reached 11 method codes and roughly 20 keyword switches because every
experiment was added to the shipping signature and none were ever removed -- its
own docstring labels most of them "measurement arms". A user reading that
signature could not tell which arguments were the product and which were somebody
mid-investigation.

So arms live here, behind a name that says what they are. Two rules:

1. **Nothing here may produce a reported number.** `doc/NUMERICS_POLICY.md` makes
   the shipped Tier-3 arithmetic the specification; an arm is for finding out
   whether something is worth shipping, not for scoring a library.
2. **An arm that wins gets promoted and its switch deleted**, rather than
   accumulating alongside the thing it replaced.

## What is reachable

The optimizer update rules beyond the two that ship. All five are compiled --
they share one device body, so there is no second copy of the optimizer to keep
correct -- but only `ADAGRAD_QADD` and `ADAGRAD_EXP` are shipping configurations.

    from roshambo3.backend import UpdateRule
    from roshambo3.experimental import ARMS_UPDATE_RULE

`ADAM_EXP` is the fix for the translation-reach limit an Adagrad step has, since
its step decays as `lr/sqrt(k)` and Adam's does not; `QN2_EXP` is a
block-diagonal Barzilai-Borwein quasi-Newton; `BFGS_EXP` is damped BFGS on the
6-vector. All three go through the exponential map. They are instantiated only
for the analytic kernel -- the field path refuses anything but `ADAGRAD_QADD`.

Note the interaction that is easy to get wrong: an optimizer change alters how
far the pose can travel, and the step count and translation learning rate are a
single knob (``roshambo3.optimizer``). An arm compared at a fixed `lr_t` is being compared at a
different travel budget, which is not the comparison anyone intends.

## What is not here, deliberately

The nine kernels ROSHAMBO2 shipped that were arms -- dense, threadwise, warpwise,
blockwise-warp, the plain blockwise variants, the `xhoist` arms, a probe tap
and prefix pruning. They were not ported at all. The code was not
worth carrying, and carrying it would have re-created exactly the surface this
module exists to prevent.
"""

from __future__ import annotations

from ..backend.cuda import UpdateRule

#: Update rules that are compiled but not shipping configurations.
ARMS_UPDATE_RULE = (UpdateRule.ADAM_EXP, UpdateRule.QN2_EXP, UpdateRule.BFGS_EXP)

#: Update rules that ship.
SHIPPING_UPDATE_RULE = (UpdateRule.ADAGRAD_QADD, UpdateRule.ADAGRAD_EXP)

__all__ = ["ARMS_UPDATE_RULE", "SHIPPING_UPDATE_RULE"]

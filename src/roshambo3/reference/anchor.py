"""The reference anchor: what two implementations are compared *against*.

There are two reference objects in this package and this
is the second one; the split is deliberate and is about what each can anchor.

`reference/overlap.py` is a float64 oracle for the **analytic expression at a
fixed pose**. It cannot anchor an end-to-end pose search, because its optimizer
is Powell with 64 restarts -- deliberately unrelated to the shipped Adagrad, so
that agreement means the algebra is right rather than that two implementations
of one schedule share a defect.

This module anchors the **end-to-end result**: the scalar CPU kernel running the
shipped schedule, frozen and stamped, with the fixtures it generates.

Why a third thing is needed at all
----------------------------------
A CPU score legitimately depends on the **vector width** it was computed at.
A GPU score legitimately depends on the **architecture** -- Volta against Blackwell, identical source and
bit-identical inputs, only 0.12% of pairs bit-identical after 100 Adagrad steps
(`doc/NUMERICS_POLICY.md` §8).

Both are fine individually. Together they mean **neither optimized
implementation can be the reference for the other**, and until this module
existed each effectively was: the only portable bit-level check,
`tests/data/golden_scores.npz`, is generated *by the GPU* and gated on its
architecture list, so it could not anchor a CPU arm at all -- and it had not
been anchoring the GPU either, having skipped silently on every
`ROSHAMBO3_ARCH_NATIVE` build.

That is a known trap at the scale of the project rather than of a function: two
implementations that agree because they were only ever checked against each
other. It is not hypothetical -- a quaternion normalization defect survived in
ROSHAMBO2's C++ and CUDA backends for exactly as long as they were each other's
only check.

**It is not a second specification.** Tier 3 remains the specification and it remains the CUDA kernel's. Nothing here is the source of a
reported number. This is a comparator.
"""
from __future__ import annotations

import numpy as np

from ..metrics.floors import (ARCHITECTURE_DELTA, EPSILON_FLOOR,  # noqa: F401
                              stratum_of)

#: The order statistic used as the per-pair tolerance.
TOLERANCE_STATISTIC = "p99"

#: Where the tolerance comes from. **Derived, not chosen** -- the
#: distinction matters, so the derivation is written out.
#:
#: The question a tolerance answers is: *is this implementation further from the
#: anchor than legitimate implementation differences can explain?* The repo has
#: two measured candidates and they are not interchangeable.
#:
#: ======================  ==================  =================  ==============
#: axis varied             epsilon floor       architecture       this anchor
#: ======================  ==================  =================  ==============
#: compiler flags          yes (7 builds)      no (identical)     yes
#: architecture / ISA      no (one machine)    **yes**            yes
#: reduction order         yes (`arith_revred`) **no**            yes
#: FMA contraction         yes (`arith_nofmad`) no                yes
#: transcendental impl     no                  no                 **yes**
#: n per stratum           43 - 283            5,272 - 209,231    --
#: ======================  ==================  =================  ==============
#:
#: **The architecture measurement does not vary reduction order**: it ran
#: identical source on both GPUs, so both reduce in the same order. The
#: measurement that varies reduction order is the floor's ``arith_revred``,
#: exposed as
#: :data:`~roshambo3.metrics.floors.REDUCTION_ORDER_DELTA_ALL_PAIRS`, and it is
#: the honest citation for the **vector-width** claim below -- a width
#: change is a reassociation, not an architecture change. At the same panel and
#: step count the two are 2.83e-5 and 2.60e-5: the same distribution.
#:
#: **The architecture delta is used, and the epsilon floor is not.** The floor varies compiler
#: flags on ONE machine and ONE architecture: it is a *within-implementation*
#: quantity, and the distinction matters -- the numerics policy pins arithmetic within one implementation, while an anchor compares
#: across them. Deriving an across-implementation tolerance from a
#: within-implementation ensemble inverts that line. It is
#: also 242x to 12,361x looser per stratum, on 100x to 1,100x less data.
#:
#: **Why not the epsilon floor.** One could argue the mismatch is safe because a 7-build ensemble spread must bound a 2-sample
#: difference. That argument fails: it holds only if the ensemble's axes
#: cover the comparison's, and they do not -- the ensemble never changes
#: architecture. It also fails where it is testable, the CUDA ANALYTIC arm's
#: max at T >= 0.7 (1.073e-6) exceeding the entire seven-build ensemble's own
#: max (8.941e-7).
TOLERANCE_SOURCE = ARCHITECTURE_DELTA

#: What this instrument does NOT bound, and it is the reason it has a scope
#: rather than a universal claim.
#:
#: **Neither candidate measurement varies the transcendental implementation**,
#: and a change of `exp2` implementation is a *larger*
#: numerics change than a reassociation -- larger than anything either bounds.
#: The anchor calls libm's `exp2f`; the CUDA kernel compiles to
#: `ex2.approx.ftz.f32`, a ~2-ULP hardware approximation with denormals flushed.
#:
#: **Measured, not argued** (a static difference probe): with `lr_t` pinned on both sides, CPU against CUDA ANALYTIC at
#: **steps=1** -- before the optimizer amplifies anything -- is nonzero on 73%
#: of pairs with p99 **5.5 ULP** and max ~2e4 ULP, against the architecture axis
#: at steps=1 at 12%, **1.5 ULP**, 5 ULP. The cross-implementation difference is
#: outside `ARCHITECTURE_DELTA`'s distribution at the source.
#:
#: What that probe does **not** establish is that the transcendental is the
#: *dominant* term: its axis bundles gcc-against-nvcc codegen, contraction and
#: the transcendental together, and nothing here isolates them. Read it as "the
#: static difference is already out of scope", not as an attribution.
#:
#: So a comparison that changes the transcendental -- **which every CPU-versus-
#: GPU comparison does** -- is outside this instrument's scope, and its
#: exceedance is not evidence of a defect. That is not a dodge to make an
#: inconvenient number go away; recall and top-end ranking are the acceptance criterion precisely because
#: perturbations in this class have to be judged on ranked output.
#:
#: The instrument's real consumer is a CPU vector-width comparison: two widths
#: differ by reduction order alone, which is bounded by ``arith_revred`` (see
#: the note on the table above -- **not** by the architecture delta, which holds
#: reduction order fixed). A ranked-output comparison is the first thing a
#: second width owes.
OUT_OF_SCOPE = (
    "a comparison that changes the transcendental implementation (libm exp2f "
    "against ex2.approx), which every CPU-versus-GPU comparison does")


def anchor_tolerance(t_ref):
    """Per-pair tolerance on |dT| against the anchor, elementwise by stratum.

    **Stratum-dependent and not a scalar**, because `doc/METRICS.md` §6 opens by
    saying a single scalar floor is wrong here *by three orders of magnitude*.
    A single number would be meaninglessly loose at the top of the list -- the
    only part a screen ranks -- or unmeetable at the bottom.

    **Read `n` before quoting a stratum.** `METRICS.md` §6 and `floors.py` both
    warn about the 0.55-0.7 row of the *epsilon floor*, where a p99 rests on 43
    samples. The architecture delta's n is 5,272 to 209,231, which is the other reason it is used
    here -- but the top stratum is still the smallest, and it is the one that
    matters most.
    """
    lab = stratum_of(np.asarray(t_ref, dtype=np.float64))
    out = np.empty(np.asarray(t_ref).shape, np.float64)
    for name, row in TOLERANCE_SOURCE.items():
        out[lab == name] = row[TOLERANCE_STATISTIC]
    return out


def per_pair_verdict(t_anchor, t_other):
    """Does `t_other` sit within build-to-build distance of the anchor?

    Returns the fraction of pairs exceeding their stratum's tolerance, the
    per-stratum breakdown, and the expected exceedance.

    **Read the asymmetry before using this.** The tolerance is a p99 of a
    measured cross-architecture distribution, so:

    * **exceeding it is evidence**, but only for a comparison inside the
      instrument's scope -- see :data:`OUT_OF_SCOPE`. A comparison that changes
      the transcendental implementation exceeds it for reasons the measurement
      never bounded.
    * **being within it is weak evidence of agreement**, because it is what is
      *measurable*, not what is *achievable*.

    So this can falsify agreement; it cannot establish it. What establishes it
    is the ranked-output comparison, which is the acceptance criterion. This function is a diagnostic that runs beside it.

    **Report it stratum-first.** An aggregate exceedance is dominated by
    whichever stratum holds most of the pairs, and the strata differ by orders;
    a single percentage hides which part of the list moved, and the only part a
    screen reads is the top.

    By construction ~1% of pairs are *expected* to exceed a p99 even for two
    legitimate builds, so the caller compares `fraction_exceeding` against
    `expected_exceeding` rather than against zero.
    """
    t_anchor = np.asarray(t_anchor, dtype=np.float64)
    t_other = np.asarray(t_other, dtype=np.float64)
    if t_anchor.shape != t_other.shape:
        raise ValueError(
            f"anchor and comparand disagree on shape: {t_anchor.shape} vs "
            f"{t_other.shape}. A fixture compared against a different set of "
            f"pairs is not a comparison.")

    # Stratify on the ANCHOR's score, not the comparand's. Two implementations
    # can land on opposite sides of a stratum boundary for one pair, and a
    # tolerance that depends on which side the comparand fell on would be a
    # different tolerance for every implementation compared.
    delta = np.abs(t_other - t_anchor)
    tol = anchor_tolerance(t_anchor)
    exceed = delta > tol
    labels = stratum_of(t_anchor)

    by_stratum = {}
    for name in TOLERANCE_SOURCE:
        sel = labels == name
        n = int(sel.sum())
        if not n:
            continue
        by_stratum[name] = {
            "n": n,
            "tolerance": float(TOLERANCE_SOURCE[name][TOLERANCE_STATISTIC]),
            "tolerance_n": int(TOLERANCE_SOURCE[name]["n"]),
            "n_exceeding": int(exceed[sel].sum()),
            "fraction_exceeding": float(exceed[sel].mean()),
            "max_abs_delta": float(delta[sel].max()),
            "median_abs_delta": float(np.median(delta[sel])),
        }

    return {
        "n_pairs": int(delta.size),
        "statistic": TOLERANCE_STATISTIC,
        "fraction_exceeding": float(exceed.mean()),
        "expected_exceeding": 0.01,      # what "p99" means
        "n_exceeding": int(exceed.sum()),
        "max_abs_delta": float(delta.max()),
        "median_abs_delta": float(np.median(delta)),
        "by_stratum": by_stratum,
        "tolerance_source": ("cross-architecture per-pair |dT|, "
                             "metrics.floors.ARCHITECTURE_DELTA"),
        "out_of_scope": OUT_OF_SCOPE,
        "note": ("Read BY STRATUM, not in aggregate: an aggregate exceedance is "
                 "dominated by whichever stratum holds most pairs and hides "
                 "which part of the list moved. Exceeding is evidence only for "
                 "a comparison inside the instrument's scope; being within it "
                 "is weak evidence of agreement, being what is measurable "
                 "rather than achievable. Acceptance is the "
                 "ranked-output comparison; this is the "
                 "diagnostic beside it."),
    }

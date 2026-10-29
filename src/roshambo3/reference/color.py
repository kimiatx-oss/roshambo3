"""The CPU color contract, in float64: the nine columns, the three modes,
a numerical color gradient, and a derived fixed-pose tolerance.

This module is the executable half of ``doc/COLOR_CONTRACT.md``. It exists so
that the first CPU color implementation is written *against* a contract rather
than inferring one from the optimized CUDA body: two
implementations once agreed on the same error for as long as they were each
other's only check. Everything here is transcribed from the published
Grant-Pickup expression and ``reference/overlap.py``; the CUDA source was read
for **conventions** (type numbering, where the pose applies, which matrix axis
is the query's) and never for the algebra.

What is pinned here, in one place:

* **The physical color formula.** ``V_color = (pi/2)^1.5 * sum_ij
  p_ij^2 r_ij^-1.5 exp(-(r_ij/2) |a_i - b_j|^2)`` over feature pairs whose
  interaction width ``r_ij = interaction_r[type_a, type_b]`` is nonzero, in
  physical Angstrom. ``interaction_p`` is the height. A zero **width** means
  the two feature types do not interact: the pair is skipped, because
  evaluating it would divide by zero. A zero **height** contributes exactly
  zero. Those are different statements and both are contract.
* **Type numbering.** Type 0 is a normal atom and never a color feature;
  feature types are ``1..N_features-1`` in the shipped enumeration
  (``pharmacophore.WORK_DESCENDING_FEATURE_ORDER``). The
  interaction matrices are ``[N_features, N_features]`` including row/column 0
  and are indexed ``[query_type, library_type]``.
* **The ordered self-overlap.** ``S_AA`` sums over all *ordered* pairs,
  ``i == j`` included, so two coincident identical features contribute four
  times one feature's self term, not twice.
* **The pose.** Applied to the **library** molecule, shape and color
  coordinates alike: ``x -> R(q) x + t`` with ``q = (w, x, y, z)`` normalized
  first, translation in Angstrom. The query is never moved.
* **The nine score columns**, ``roshambo3.scores`` 0..8, and in particular
  column 0 ``= (1 - mixing) * T_shape + mixing * T_color``. It is **not** the
  legacy sum out of two (``scores.tanimoto_combo_legacy``).
* **The guard.** Every Tanimoto denominator is guarded: ``den <= 0`` reports
  ``0.0``, never NaN or Inf. Empty color on either or both sides gives finite
  zero color scores and leaves the shape output untouched.

It is **not a second specification**: Tier 3 remains
the CUDA kernel's, and no reported number comes from this module. It is the
comparator the first CPU color implementation is measured against, and the
generator of ``tests/data/color_contract_fixture.npz``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .overlap import (color_overlap, pair_overlap, quaternion_matrix,
                      self_overlap, _CANONICAL_STARTS, _axis_angle_quat)

# ---------------------------------------------------------------------------
# The three public modes, named without CUDA vocabulary
# ---------------------------------------------------------------------------
#
# ``doc/COLOR_CONTRACT.md`` numbers them 1/2/3; the CUDA path spells modes 2 and 3
# `ColorPlacement.AT_POSE` / `IN_GRADIENT`, which describe where a kernel
# evaluates a term rather than what a caller asked for. The public names
# describe the objective and are what the CPU seam uses.
#
# * SHAPE_ONLY -- the pose is optimized on shape and color is not scored at
#   all. The color columns (2, 4, 7, 8) are 0.0 and column 0 equals the shape
#   Tanimoto; `mixing` must be 0 in this mode and a nonzero value is refused,
#   because column 0 would otherwise silently report (1-mixing)*T_shape.
# * SHAPE_POSE_COLOR_RESCORE -- mode 2. The pose search is *exactly*
#   the shape-only one: same trajectory, same per-start winner, selected on
#   the shape Tanimoto (ties go to the earliest start in the canonical start
#   order). Color is then evaluated exactly once at that converged pose and
#   every color column is populated; column 0 is the mixed score. The
#   property that makes this mode worth having -- and that a test must pin --
#   is that its pose, winning start and shape score are identical to a
#   shape-only run's on the same inputs.
# * COMBO_OPTIMAL -- mode 3. The optimizer's objective is the reported
#   column 0 itself: the gradient is the derivative of
#   (1-mixing)*T_shape + mixing*T_color, each channel weighted by its own
#   dT/dV before mixing (the corrected `combo_gradient_weights`).
#   Start selection and the final winner are on the mixed score, and the
#   reported row is an exact evaluation of both channels at the converged
#   pose -- the final rescore, not a value left over from the search.
MODE_SHAPE_ONLY = "shape_only"
MODE_SHAPE_POSE_COLOR_RESCORE = "shape_pose_color_rescore"
MODE_COMBO_OPTIMAL = "combo_optimal"

COLOR_MODES = (MODE_SHAPE_ONLY, MODE_SHAPE_POSE_COLOR_RESCORE,
               MODE_COMBO_OPTIMAL)


def guarded_tanimoto(v_ab: float, v_aa: float, v_bb: float) -> float:
    """``V_AB / (V_AA + V_BB - V_AB)``, and 0.0 when the denominator is not
    positive.

    In exact arithmetic the denominator is strictly positive whenever either
    molecule has any density at all (Cauchy-Schwarz gives
    ``V_AB <= sqrt(V_AA V_BB) <= (V_AA + V_BB)/2``), so ``den <= 0`` means
    "both sides are empty on this channel" and the only honest similarity to
    report is none. **The guard is the contract**: an empty color plane on
    both sides is a legal input (zero-feature molecules are
    representable) and must produce finite zero color scores, never 0/0.
    ``0.0`` and not ``1.0``, deliberately: reporting two empty planes as a
    perfect match would rank every featureless pair above every real one.

    The kernel's field path already guards its color divide exactly this way
    (``(den_c > 0.0f) ? vcsum/den_c : 0.0f``); the shape divide does not
    (an open defect), and the analytic path's color
    divide does not either -- both are defects to fix toward this contract,
    not precedents to import from it.
    """
    den = v_aa + v_bb - v_ab
    return float(v_ab / den) if den > 0.0 else 0.0


def _as_f64(a, width: int) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64)
    if out.size == 0:
        return out.reshape(0, width) if width > 1 else out.reshape(0)
    return out


def apply_pose_elementwise(coords: np.ndarray, q: np.ndarray,
                           t: np.ndarray) -> np.ndarray:
    """``x -> R(q) x + t`` on ``[n, 3]``, written as elementwise float64 ops.

    Same transform as :func:`reference.overlap.apply_pose` (``b @ R(q).T + t``)
    -- the pose applies to the **library** molecule -- but computed without a
    matrix product, so no BLAS is in the loop and the result is a fixed
    sequence of IEEE-754 elementwise operations. That is what lets a committed
    fixture carry pre-posed float32 coordinates and a test reproduce them
    exactly: a BLAS matmul's summation order is a library detail, and a
    last-bit float64 difference could (rarely, but detectably) survive the
    rounding to float32.
    """
    c = _as_f64(coords, 3)
    if c.size == 0:
        return c.reshape(0, 3)
    m = quaternion_matrix(q)
    tt = np.asarray(t, dtype=np.float64)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    out = np.empty_like(c)
    out[:, 0] = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + tt[0]
    out[:, 1] = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + tt[1]
    out[:, 2] = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z + tt[2]
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of ``(w, x, y, z)`` quaternions."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def score_columns(query_shape, library_shape,
                  query_color, query_types,
                  library_color, library_types,
                  interaction_r, interaction_p,
                  mixing: float,
                  pose_quat=None, pose_translation=None) -> dict:
    """The nine scalar score columns, in float64, at a fixed pose.

    The single authority for what each of ``roshambo3.scores`` columns 0..8
    *means* on a color run. Volumes are physical (the shape and color
    prefactors applied), Tanimotos are guarded per :func:`guarded_tanimoto`,
    and column 0 is the properly mixed single score.

    Args:
        query_shape: ``[nA, 3]`` heavy-atom coordinates, Angstrom.
        library_shape: ``[nB, 3]``.
        query_color / library_color: ``[nCA, 3]`` / ``[nCB, 3]`` feature
            pseudo-atom coordinates; either may be empty.
        query_types / library_types: matching integer feature types, in
            ``1..N_features-1`` (0 is a normal atom and never a feature).
        interaction_r / interaction_p: ``[N_features, N_features]`` width and
            height, indexed ``[query_type, library_type]``.
        mixing: the blend weight in ``[0, 1]``; refused outside it.
        pose_quat / pose_translation: if given, applied to the **library**
            molecule (shape and color alike) in float64 before scoring. The
            self-overlaps are computed from the coordinates as scored, which
            for the library means the posed ones -- an exact rigid motion
            leaves them mathematically unchanged, so this is a statement about
            which floats are summed, not about the physics.

    Returns:
        dict keyed by the ``roshambo3.scores.COLUMN_INDEX`` names.
    """
    m = float(mixing)
    if not 0.0 <= m <= 1.0:
        raise ValueError(f"mixing must be in [0, 1], got {m}")

    a_s = _as_f64(query_shape, 3)
    b_s = _as_f64(library_shape, 3)
    a_c = _as_f64(query_color, 3)
    b_c = _as_f64(library_color, 3)
    ta = np.asarray(query_types, dtype=np.int64).reshape(-1)
    tb = np.asarray(library_types, dtype=np.int64).reshape(-1)
    if a_c.shape[0] != ta.shape[0] or b_c.shape[0] != tb.shape[0]:
        raise ValueError("feature coordinates and types disagree on length")
    if ((ta == 0).any() or (tb == 0).any()) and (ta.size or tb.size):
        raise ValueError(
            "feature type 0 is a normal atom, never a color feature; the "
            "shipped enumeration numbers features from 1")

    if pose_quat is not None:
        b_s = apply_pose_elementwise(b_s, pose_quat, pose_translation)
        b_c = apply_pose_elementwise(b_c, pose_quat, pose_translation)

    v_ab = pair_overlap(a_s, b_s) if a_s.size and b_s.size else 0.0
    v_aa = self_overlap(a_s) if a_s.size else 0.0
    v_bb = self_overlap(b_s) if b_s.size else 0.0

    def _color(x, tx, y, ty):
        if x.size == 0 or y.size == 0:
            return 0.0
        return color_overlap(x, tx, y, ty, interaction_r, interaction_p)

    c_ab = _color(a_c, ta, b_c, tb)
    c_aa = _color(a_c, ta, a_c, ta)
    c_bb = _color(b_c, tb, b_c, tb)

    t_shape = guarded_tanimoto(v_ab, v_aa, v_bb)
    t_color = guarded_tanimoto(c_ab, c_aa, c_bb)

    return {
        # Column 0. The mixed single score -- NOT tanimoto_shape +
        # tanimoto_color, which is the legacy sum out of two and lives (only)
        # in scores.tanimoto_combo_legacy for published comparisons.
        "tanimoto_combination": (1.0 - m) * t_shape + m * t_color,
        "tanimoto_shape": t_shape,
        "tanimoto_color": t_color,
        "overlap_volume": v_ab,
        "overlap_color": c_ab,
        "self_overlap_query": v_aa,
        "self_overlap_library": v_bb,
        "self_color_query": c_aa,
        "self_color_library": c_bb,
    }


def shape_only_columns(query_shape, library_shape,
                       pose_quat=None, pose_translation=None) -> dict:
    """MODE_SHAPE_ONLY's nine columns: color identically zero, column 0 the
    shape Tanimoto itself. ``mixing`` does not appear because the mode pins it
    to zero."""
    empty = np.zeros((0, 3))
    none = np.zeros(0, np.int64)
    r1 = np.zeros((1, 1))
    return score_columns(query_shape, library_shape, empty, none, empty, none,
                         r1, r1, 0.0, pose_quat, pose_translation)


# ---------------------------------------------------------------------------
# The numerical color gradient (scope item 3)
# ---------------------------------------------------------------------------

def pose_gradient_numerical(value: Callable[[np.ndarray, np.ndarray], float],
                            pose_quat, pose_translation,
                            h: float = 1e-4) -> np.ndarray:
    """Six-component pose derivative of ``value(q, t)`` by central differences.

    **Deliberately numerical**: an analytic gradient here would be
    a second chance to make the same algebra mistake a kernel might have made,
    and agreeing with it would prove nothing. A central difference of the
    *value* shares nothing with any kernel's derivation but the value itself.
    If a future refactor rearranges this into the analytic expression it has
    built a tautology, and the fixture it stamps means nothing.

    **The rotation is perturbed in axis-angle, not in quaternion components**,
    so the result compares like coordinates with a rotation generator.
    Components 0..2 are the derivative along a space-frame axis-angle
    increment applied on top of the current rotation:

        d/d omega_k  value( exp([omega_k e_k]) R(q), t )  at omega = 0,

    realized as the quaternion left-product ``axis_angle(h, e_k) * q``. A
    central difference in raw ``q`` components is **not** this quantity -- it
    differs by the (pose-dependent) Jacobian of the chart, and looks plausible
    while being wrong by exactly that factor. Components 3..5 are the
    derivative in the translation, in the same units ``t`` is in (Angstrom
    here; a consumer comparing against a reduced-frame implementation owns the
    frame conversion, `doc/NUMERICS_POLICY.md` section 4).

    Note the perturbed rotation acts about the space origin on the *rotated,
    untranslated* molecule -- the same composition the pose itself uses
    (``x -> R x + t``), so rotation and translation components stay
    independent coordinates.

    Sign convention: this is the **ascent** gradient of ``value``. The shipped
    optimizer consumes a descent direction; ``reference.overlap
    .overlap_gradient``'s docstring records the measured sign relation.
    """
    q = np.asarray(pose_quat, dtype=np.float64)
    t = np.asarray(pose_translation, dtype=np.float64)
    g = np.zeros(6)
    axes = np.eye(3)
    for k in range(3):
        up = quat_multiply(_axis_angle_quat(+h, *axes[k]), q)
        dn = quat_multiply(_axis_angle_quat(-h, *axes[k]), q)
        g[k] = (value(up, t) - value(dn, t)) / (2.0 * h)
    for k in range(3):
        up, dn = t.copy(), t.copy()
        up[k] += h
        dn[k] -= h
        g[3 + k] = (value(q, up) - value(q, dn)) / (2.0 * h)
    return g


def color_overlap_gradient(query_color, query_types,
                           library_color, library_types,
                           interaction_r, interaction_p,
                           pose_quat, pose_translation,
                           h: float = 1e-4) -> np.ndarray:
    """Numerical pose gradient of the physical color overlap ``V_color``."""
    a_c = _as_f64(query_color, 3)
    ta = np.asarray(query_types, dtype=np.int64).reshape(-1)
    b_c = _as_f64(library_color, 3)
    tb = np.asarray(library_types, dtype=np.int64).reshape(-1)

    def value(q, t):
        if a_c.size == 0 or b_c.size == 0:
            return 0.0
        posed = apply_pose_elementwise(b_c, q, t)
        return color_overlap(a_c, ta, posed, tb, interaction_r, interaction_p)

    return pose_gradient_numerical(value, pose_quat, pose_translation, h)


def mixed_tanimoto_gradient(query_shape, library_shape,
                            query_color, query_types,
                            library_color, library_types,
                            interaction_r, interaction_p, mixing: float,
                            pose_quat, pose_translation,
                            h: float = 1e-4) -> np.ndarray:
    """Numerical pose gradient of the reported column 0.

    This is the derivative MODE_COMBO_OPTIMAL's optimizer must follow: the
    mixed *Tanimoto*, each channel implicitly weighted by its own dT/dV.
    Adding the raw overlap gradients instead optimizes neither reported
    Tanimoto; the corrected per-channel weighting is exactly
    what makes an analytic implementation agree with this central difference.
    """
    def value(q, t):
        return score_columns(query_shape, library_shape, query_color,
                             query_types, library_color, library_types,
                             interaction_r, interaction_p, mixing,
                             q, t)["tanimoto_combination"]

    return pose_gradient_numerical(value, pose_quat, pose_translation, h)


# ---------------------------------------------------------------------------
# Reference optimization for the modes (fixture generation only)
# ---------------------------------------------------------------------------

def optimize_pose_on(value: Callable[[np.ndarray, np.ndarray], float], *,
                     restarts: int = 64, seed: int = 0,
                     maxiter: int = 40000, maxfev: Optional[int] = None) -> dict:
    """Maximize ``value(q, t)`` by the oracle's own global search.

    The same Powell-with-canonical-seeds scheme as
    :func:`reference.overlap.optimize_pose` (whose docstring carries the
    measured convergence ladder), generalized to any objective so the three
    modes can share one search. Deliberately unrelated to the shipped
    Adagrad schedule: the reference pins each mode's *objective and selection
    rule*, never the trajectory -- an oracle whose optimizer matched the
    kernel's could only confirm that two copies of one schedule agree
    (see the two-reference-objects split in ``reference/anchor.py``).
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    seeds = [np.concatenate([_axis_angle_quat(*s), np.zeros(3)])
             for s in _CANONICAL_STARTS]
    while len(seeds) < restarts:
        q = rng.normal(size=4)
        seeds.append(np.concatenate([q / np.linalg.norm(q),
                                     rng.normal(scale=1.5, size=3)]))

    def neg(x):
        return -value(x[:4], x[4:])

    best_x, best_f = None, np.inf
    for x0 in seeds[:max(restarts, len(_CANONICAL_STARTS))]:
        options = {"xtol": 1e-10, "ftol": 1e-12, "maxiter": maxiter}
        if maxfev is not None:
            options["maxfev"] = maxfev
        res = minimize(neg, x0, method="Powell", options=options)
        if res.fun < best_f:
            best_f, best_x = float(res.fun), res.x.copy()

    q = best_x[:4] / np.linalg.norm(best_x[:4])
    return {"pose_quat": q, "pose_translation": best_x[4:].copy(),
            "objective": -best_f}


def score_modes(query_shape, library_shape,
                query_color, query_types,
                library_color, library_types,
                interaction_r, interaction_p, mixing: float, *,
                restarts: int = 64, seed: int = 0) -> dict:
    """All three modes of one pair, each at its own optimal pose.

    * ``shape_only``: pose maximizes the shape Tanimoto; color unscored.
    * ``shape_pose_color_rescore``: the **same pose** (by construction here,
      literally the same floats -- which is the mode's defining property, not
      an optimization), with one exact color evaluation at it.
    * ``combo_optimal``: pose maximizes the mixed column 0.

    Discrimination between the modes needs ``mixing > 0``: at 0 the mixed
    objective is the shape objective and all three collapse.
    """
    def shape_value(q, t):
        return shape_only_columns(query_shape, library_shape,
                                  q, t)["tanimoto_shape"]

    def mixed_value(q, t):
        return score_columns(query_shape, library_shape, query_color,
                             query_types, library_color, library_types,
                             interaction_r, interaction_p, mixing,
                             q, t)["tanimoto_combination"]

    shape_opt = optimize_pose_on(shape_value, restarts=restarts, seed=seed)
    combo_opt = optimize_pose_on(mixed_value, restarts=restarts, seed=seed)

    out = {}
    out[MODE_SHAPE_ONLY] = {
        "pose_quat": shape_opt["pose_quat"],
        "pose_translation": shape_opt["pose_translation"],
        "columns": shape_only_columns(query_shape, library_shape,
                                      shape_opt["pose_quat"],
                                      shape_opt["pose_translation"]),
    }
    out[MODE_SHAPE_POSE_COLOR_RESCORE] = {
        "pose_quat": shape_opt["pose_quat"],
        "pose_translation": shape_opt["pose_translation"],
        "columns": score_columns(query_shape, library_shape, query_color,
                                 query_types, library_color, library_types,
                                 interaction_r, interaction_p, mixing,
                                 shape_opt["pose_quat"],
                                 shape_opt["pose_translation"]),
    }
    out[MODE_COMBO_OPTIMAL] = {
        "pose_quat": combo_opt["pose_quat"],
        "pose_translation": combo_opt["pose_translation"],
        "columns": score_columns(query_shape, library_shape, query_color,
                                 query_types, library_color, library_types,
                                 interaction_r, interaction_p, mixing,
                                 combo_opt["pose_quat"],
                                 combo_opt["pose_translation"]),
    }
    return out


# ---------------------------------------------------------------------------
# The derived fixed-pose tolerance (scope item 5; "derive, do not discover")
# ---------------------------------------------------------------------------

#: Unit roundoff of float32.
_U32 = 2.0 ** -24

#: Relative tolerance for REPRODUCING a committed float64 fixture value with
#: this module on another platform. The fixture is deliberately not claimed
#: bitwise portable, for the same reason as the shape
#: anchor (``reference/anchor.py``): the float64 ``exp`` is libm's. Derived: <= 1 ulp of float64 per
#: transcendental (2^-52 relative), plus n-term float64 accumulation
#: (n * 2^-52, n <= 4096 on any fixture case), through the Tanimoto's
#: 2 T (1+T) <= 4 amplification: 4 * 4097 * 2^-52 = 3.6e-12; held at 1e-11
#: for the same declared headroom the float32 bound carries. On the
#: generating platform reproduction is expected bit-exact and reported as
#: such, but the *gate* is this tolerance.
FLOAT64_REPRODUCTION_RTOL = 1e-11

#: Relative error allowed the transcendental: covers CUDA's
#: ``ex2.approx.ftz.f32`` (documented <= 2 ulp) and any faithful libm
#: ``exp2f``, with 4x headroom on that axis alone. 2 ulp of a result in
#: [0.5, 1) is 2^-23; this is 2^-21.
_EXP2_REL = 2.0 ** -21

#: Bound on the relative error of the float32 exponent *argument* arising
#: from the argument arithmetic itself: two coordinate-scale multiplies, a
#: subtract, three squares and two adds, one width multiply -- at most ~8
#: roundings on the critical path; 16 is deliberate slack, still first
#: principles rather than a fit.
_ARG_ROUNDINGS = 16.0

#: Roundings in a color term's pair coefficient ``p^2 * r^-1.5`` (square,
#: reciprocal, square root, two multiplies -- ``__fsqrt_rn`` and the CPU's
#: sqrt are correctly rounded).
_COEFF_ROUNDINGS = 8.0

#: Multiplier on the whole bound. Declared here, before any comparison ran
#: (the commit history is the evidence), to cover rounding paths the model
#: does not enumerate: the start-mode transform's pre-scale, the mask-weight
#: multiplies, FMA regrouping the compiler is free to choose, and the final
#: prefactor multiply. It is headroom on a worst-case bound, not a fitted
#: constant, and it must never be raised to admit a failing candidate --
#: a candidate outside the bound is a finding, not a calibration problem.
_MODEL_HEADROOM = 4.0

#: Provenance string carried into fixtures and verdicts.
FIXED_POSE_TOLERANCE_DERIVATION = (
    "first-principles forward-error bound on a float32 evaluation of the "
    "fixed-pose sums against their float64 values: per term, "
    "ln2 * x * 16u + 2^-21 (transcendental) [+ input rounding when the "
    "implementation applies the pose itself] + 8u (pair coefficient); per "
    "sum, + n*u accumulation (worst-case serial, positive terms); Tanimoto "
    "via |dT| <= 2 T (1+T) rho; column 0 by linear mixing; all times a "
    "declared headroom of 4. u = 2^-24. Derived from the case's own inputs "
    "before any candidate was compared -- never fitted to one.")


def _channel_relative_bound(exponents: np.ndarray, terms: np.ndarray,
                            input_arg_coeff: float) -> float:
    """Relative forward-error bound on one float32 pair sum.

    ``exponents``/``terms`` are the per-pair exp2 arguments and term values in
    float64. The bound is a sum over terms of each term's own relative error
    (a large-exponent term has a larger relative error but a proportionally
    smaller weight, and weighting by the term handles that exactly), plus
    worst-case serial accumulation.

    ``input_arg_coeff`` carries the pose-application rounding for a comparand
    that is handed the *unposed* library and rotates it in float32 itself, as
    an exponent perturbation ``|dx| <= input_arg_coeff * sqrt(x)``:

        x = c d^2  =>  |dx| = c |d(d^2)| <= c * 2 sqrt(3) d eps
                            = 2 sqrt(3) eps sqrt(c) sqrt(x),

    with ``eps`` the absolute coordinate error of the float32 pose
    application and ``c`` the channel's exponent constant -- so the caller
    passes ``2 sqrt(3) eps sqrt(c_max)`` and 0.0 when the comparand receives
    pre-posed coordinates (the fixture's arrangement, where the inputs are
    bit-identical on both sides).
    """
    s = float(terms.sum())
    if s <= 0.0 or terms.size == 0:
        return 0.0
    d_arg = _ARG_ROUNDINGS * _U32 * exponents
    if input_arg_coeff > 0.0:
        d_arg = d_arg + input_arg_coeff * np.sqrt(exponents)
    per_term_rel = math.log(2.0) * d_arg + _EXP2_REL + _COEFF_ROUNDINGS * _U32
    abs_err = float((terms * per_term_rel).sum())
    abs_err += terms.size * _U32 * s          # serial accumulation
    return abs_err / s


def _shape_pairs(a: np.ndarray, b: np.ndarray):
    """(exponents, terms) of the shape channel in the reduced frame."""
    if a.size == 0 or b.size == 0:
        return np.zeros(0), np.zeros(0)
    from .overlap import ALPHA
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).reshape(-1)
    x = (0.5 * ALPHA / math.log(2.0)) * d2      # exp2 argument
    return x, np.exp2(-x)


def _color_pairs(a, ta, b, tb, rmat, pmat):
    """(exponents, terms) of the color channel, zero-width pairs skipped."""
    r = np.asarray(rmat, dtype=np.float64)
    p = np.asarray(pmat, dtype=np.float64)
    xs, ts = [], []
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            width = r[ta[i], tb[j]]
            if width == 0.0:
                continue
            height = p[ta[i], tb[j]]
            d2 = float(((a[i] - b[j]) ** 2).sum())
            x = 0.5 * width * d2 / math.log(2.0)
            xs.append(x)
            ts.append(math.exp2(-x) * height * height * width ** -1.5)
    return np.asarray(xs), np.asarray(ts)


def fixed_pose_tolerance(query_shape, library_shape,
                         query_color, query_types,
                         library_color, library_types,
                         interaction_r, interaction_p, mixing: float, *,
                         include_input_rounding: bool = False) -> dict:
    """Per-column absolute tolerances for a float32 implementation at a fixed
    pose, derived from this case's own inputs.

    **This is the declared fixed-pose color tolerance**,
    and it is not the anchor's: that one is the measured *end-to-end*
    cross-architecture |dT| distribution, which bounds what 100 optimizer
    steps do to a last-bit difference and says nothing about a single
    fixed-pose evaluation. Here there is no optimizer in the loop, so the
    honest bound is the forward rounding error of the arithmetic itself --
    see :data:`FIXED_POSE_TOLERANCE_DERIVATION`. It is a worst-case bound and
    therefore loose against typical behavior (measured fixed-pose CUDA-vs-
    oracle agreement runs ~1e-6 relative where this admits ~1e-4): what makes
    it usable as a gate is that every wrong-convention case it must catch --
    swapped types, wrong pose direction, dropped self cross-terms, the legacy
    combination -- moves the score by orders of magnitude more, and the
    contract's negative tests verify that margin rather than assume it.

    The library coordinates must be the **posed** ones when the comparand is
    handed posed coordinates (the fixture's arrangement); pass
    ``include_input_rounding=True`` only for a comparand that applies the
    pose itself in float32.
    """
    a_s = _as_f64(query_shape, 3)
    b_s = _as_f64(library_shape, 3)
    a_c = _as_f64(query_color, 3)
    b_c = _as_f64(library_color, 3)
    ta = np.asarray(query_types, dtype=np.int64).reshape(-1)
    tb = np.asarray(library_types, dtype=np.int64).reshape(-1)
    m = float(mixing)

    cols = score_columns(a_s, b_s, a_c, ta, b_c, tb,
                         interaction_r, interaction_p, m)

    # The pose-application coordinate error, when the comparand rotates the
    # library itself in float32: ~8 roundings per produced component (three
    # multiplies, two adds, the translation add, slack), each relative to the
    # coordinate magnitude in play.
    if include_input_rounding:
        cmax = max((float(np.abs(x).max()) for x in (a_s, b_s, a_c, b_c)
                    if x.size), default=0.0)
        eps = 8.0 * _U32 * cmax
        from .overlap import ALPHA
        c_shape = 0.5 * ALPHA / math.log(2.0)
        r_all = np.asarray(interaction_r, dtype=np.float64)
        c_color = 0.5 * float(r_all.max(initial=0.0)) / math.log(2.0)
        shape_coeff = 2.0 * math.sqrt(3.0) * eps * math.sqrt(c_shape)
        color_coeff = 2.0 * math.sqrt(3.0) * eps * math.sqrt(max(c_color,
                                                                 0.0))
    else:
        shape_coeff = color_coeff = 0.0

    def rel(coeff, pairs_fn, *args):
        x, terms = pairs_fn(*args)
        return _channel_relative_bound(x, terms, coeff) * _MODEL_HEADROOM

    rho = {
        "overlap_volume": rel(shape_coeff, _shape_pairs, a_s, b_s),
        "self_overlap_query": rel(shape_coeff, _shape_pairs, a_s, a_s),
        "self_overlap_library": rel(shape_coeff, _shape_pairs, b_s, b_s),
        "overlap_color": rel(color_coeff, _color_pairs, a_c, ta, b_c, tb,
                             interaction_r, interaction_p),
        "self_color_query": rel(color_coeff, _color_pairs, a_c, ta, a_c, ta,
                                interaction_r, interaction_p),
        "self_color_library": rel(color_coeff, _color_pairs, b_c, tb, b_c, tb,
                                  interaction_r, interaction_p),
    }
    tol = {k: abs(cols[k]) * r for k, r in rho.items()}

    # |dT| <= 2 T (1 + T) rho_max: T = V/(SA+SB-V), dT/dV = (SA+SB)/den^2 and
    # |dT/dSA| = |dT/dSB| = V/den^2, so with a shared relative bound rho on
    # all three volumes, |dT| <= rho * 2 V (SA+SB)/den^2 = 2 rho T (1+T).
    def dt(t_val, *cols_of_channel):
        r = max(rho[c] for c in cols_of_channel)
        return 2.0 * r * t_val * (1.0 + t_val)

    tol["tanimoto_shape"] = dt(cols["tanimoto_shape"], "overlap_volume",
                               "self_overlap_query", "self_overlap_library")
    tol["tanimoto_color"] = dt(cols["tanimoto_color"], "overlap_color",
                               "self_color_query", "self_color_library")
    tol["tanimoto_combination"] = ((1.0 - m) * tol["tanimoto_shape"]
                                   + m * tol["tanimoto_color"]
                                   + 4.0 * _U32 * abs(
                                       cols["tanimoto_combination"]))
    return tol

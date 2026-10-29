"""A float64 reference implementation of the scoring function.

This is the oracle. It is slow, obvious, and written to be read rather than to
be fast: every quantity is a direct transcription of the Grant-Pickup
first-order Gaussian overlap in physical units, in float64, with no reduced
frame, no staging, no tiling and no fused anything.

Why it exists, in its own right
-------------------------------
ROSHAMBO2 checked its CUDA kernel against a C++/OpenMP backend and a PyTorch
autograd backend. Both are gone here, and neither is missed, but the *function*
they served must not be: without an independent implementation there is no test
of the kernel at this level, only end-to-end comparisons that cannot distinguish
a defect the two share.

That is not hypothetical. "'C++ and CUDA agree, so the math is right' -- they
agree because they share the typo": a quaternion normalization defect survived
in both for as long as they were each other's only check. An oracle transcribed from the
published expression, in a different language, by a different route, is the thing
that catches that class of error.

What it is not
--------------
It is **not** the specification. ``doc/NUMERICS_POLICY.md`` makes the CUDA
kernel's Tier-3 arithmetic the specification, deliberately, because the reported
score has to come from one place. This module answers "is the analytic result
right", not "is the last bit right" -- for that, see the bit-identity gate in
``bench/xcheck_roshambo2.py``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: The Grant-Pickup constants, in float64. These are the same numbers
#: ``roshambo3.numerics`` pins for the device, computed here without the float32
#: rounding the device constants carry.
KAPPA = 2.41798793102
CARBON_RADIUS = 1.7
ALPHA = KAPPA / (CARBON_RADIUS * CARBON_RADIUS)
"""Per-atom Gaussian width, kappa / r_C^2 = 0.836674 A^-2."""

PAIR_PREFACTOR = 8.0 * (math.pi / (2.0 * ALPHA)) ** 1.5
"""Volume prefactor of a single atom pair at zero separation."""

COLOR_PREFACTOR = (math.pi / 2.0) ** 1.5
"""Extracted from the color pair term, where it cancels for the canonical
interaction matrix. It only cancels for that one, which is why
``numerics.check_interaction_matrix`` refuses others."""


def pair_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Shape overlap volume between two heavy-atom point sets, in cubic Angstrom.

        V_AB = 8 (pi/2alpha)^{3/2} sum_i sum_j exp(-(alpha/2)|a_i - b_j|^2)

    Every shape atom shares one width, which is the fact the whole engine is
    built on: it makes the pair exponent a constant, it is why a precomputed
    query field on a lattice is an identity rather than an approximation, and it
    is why Cauchy-Schwarz is the provable filter bound.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(PAIR_PREFACTOR * np.exp(-0.5 * ALPHA * d2).sum())


def self_overlap(a: np.ndarray) -> float:
    """``V_AA``. Note it sums over all *ordered* pairs, so two coincident atoms
    give four times one atom, not twice."""
    return pair_overlap(a, a)


def tanimoto(v_ab: float, v_aa: float, v_bb: float) -> float:
    """The Grant-Pickup Tanimoto: ``V_AB / (V_AA + V_BB - V_AB)``.

    There is no inclusion-exclusion in the first-order expression, which is why
    ``V_AB <= min(V_AA, V_BB)`` is **false** here -- measured up to 2.25 -- and
    why a filter built on it discards real hits. Cauchy-Schwarz
    (``V_AB <= sqrt(V_AA V_BB)``) is the provable bound, and it is weak.
    """
    den = v_aa + v_bb - v_ab
    # A non-positive denominator is 0 on every backend: consistent inputs
    # cannot produce one.
    return float(v_ab / den) if den > 0 else 0.0


def color_overlap(a: np.ndarray, a_types: np.ndarray,
                  b: np.ndarray, b_types: np.ndarray,
                  interaction_r: np.ndarray,
                  interaction_p: np.ndarray) -> float:
    """Color overlap between two sets of pharmacophore pseudo-atoms.

    Unlike the shape term, color widths are **pair-dependent**: the width comes
    from ``interaction_r[ta, tb]`` and the height from ``interaction_p[ta, tb]``.
    A zero width means those two feature types do not interact at all and the
    pair is skipped -- evaluating it would divide by zero.

    This pair-dependence is exactly why the query field cannot carry color: one
    scalar field cannot represent a family of widths, so a color-capable field
    means one field per feature type.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ta = np.asarray(a_types, dtype=np.int64)
    tb = np.asarray(b_types, dtype=np.int64)
    r = np.asarray(interaction_r, dtype=np.float64)
    p = np.asarray(interaction_p, dtype=np.float64)

    total = 0.0
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            width = r[ta[i], tb[j]]
            if width == 0.0:
                continue
            height = p[ta[i], tb[j]]
            d2 = float(((a[i] - b[j]) ** 2).sum())
            # exp(-(r/2) d^2). The width `r` plays the role ALPHA plays for
            # shape (see `pair_overlap` above, exp(-0.5*ALPHA*d2)), so it is
            # not multiplied by it as well -- this line carried a spurious
            # factor of ALPHA once, which the canonical
            # interaction matrix hid: it couples only same-type pairs, and a
            # molecule whose features all have distinct types has every
            # cross term at zero width and every self term at d = 0, where
            # any exponent agrees. `numerics.py`'s INV_TWO_ALPHA docstring
            # states the contract the kernel implements:
            # exp(-(r/2) d^2) == exp2(-r * INV_TWO_ALPHA * d'^2).
            total += (math.exp(-0.5 * width * d2)
                      * height * height * width ** -1.5)
    return float(COLOR_PREFACTOR * total)


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of ``(w, x, y, z)``, normalized first.

    Normalizing is the point, not hygiene: the defect this oracle exists to
    catch was a missing normalization that made the applied transform
    non-orthogonal, with ``|q|`` reaching 1.216 and ``det(M)`` 1.313, so 1.2% of
    poses reported a Tanimoto no rigid motion can achieve.
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0 or not np.isfinite(n):
        return np.eye(3)
    w, x, y, z = q / n
    m = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return m


def apply_pose(b: np.ndarray, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``b @ R(q).T + t`` -- the transform the score columns describe."""
    return np.asarray(b, dtype=np.float64) @ quaternion_matrix(q).T + np.asarray(t, np.float64)


def score_at_pose(a: np.ndarray, b: np.ndarray, q: np.ndarray,
                  t: np.ndarray) -> dict:
    """Shape overlap and Tanimoto for a given pose. No optimization."""
    bt = apply_pose(b, q, t)
    v_ab = pair_overlap(a, bt)
    v_aa = self_overlap(a)
    v_bb = self_overlap(b)
    return {"overlap_volume": v_ab, "self_overlap_query": v_aa,
            "self_overlap_library": v_bb,
            "tanimoto_shape": tanimoto(v_ab, v_aa, v_bb)}


def overlap_gradient(a: np.ndarray, b: np.ndarray, q: np.ndarray,
                     t: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Numerical gradient of ``V_AB`` in ``(q, t)``, by central differences.

    The pose applies to ``b`` -- the library molecule -- which is the kernel's
    convention too (``get_gradient``'s ``// molB needs to be transformed by q``).

    Deliberately numerical. An analytic gradient here would be a second chance
    to make the same algebra mistake the kernel might have made, and agreeing
    with it would prove nothing -- two implementations can share a typo. A central
    difference of the *value* shares nothing with the kernel's derivation but
    the value itself.

    **How this compares to the kernel's ``test_gradient``, measured:**

    * the **translation** components are the exact negative of the kernel's --
      ``cos = -1.000000000`` and magnitude ratio ``1.000000`` over five random
      pairs. The sign is a convention: the kernel returns a *descent* direction,
      which is what ``adagrad_step`` consumes.
    * the **rotation** components are **not** comparable. The kernel's are a
      rotation generator (a torque on ``q[1:4]``, with ``q[0]`` unused and the
      update applied as ``q <- normalize(q + (0, d))``), not ``dV/dq_i`` of the
      raw components. Flipping the sign does not reconcile them and nothing here
      should pretend otherwise; ROSHAMBO2 validated that parameterization against
      PyTorch autograd, and that is where the claim rests.
    """
    q = np.asarray(q, dtype=np.float64).copy()
    t = np.asarray(t, dtype=np.float64).copy()
    g = np.zeros(7)

    def value(qq, tt):
        return pair_overlap(a, apply_pose(b, qq, tt))

    for i in range(4):
        up, dn = q.copy(), q.copy()
        up[i] += h
        dn[i] -= h
        g[i] = (value(up, t) - value(dn, t)) / (2 * h)
    for i in range(3):
        up, dn = t.copy(), t.copy()
        up[i] += h
        dn[i] -= h
        g[4 + i] = (value(q, up) - value(q, dn)) / (2 * h)
    return g


#: The canonical start rotations, as axis-angle. The same set the kernel uses:
#: identity, 180 degrees about each axis, then +-90 about each. They matter as
#: *seeds* here because the objective is badly multi-modal -- two molecules can
#: be flipped relative to each other and a local optimizer started at the
#: identity will never find it.
_CANONICAL_STARTS = [
    (0.0, 0.0, 0.0, 0.0),
    (np.pi, 1.0, 0.0, 0.0), (np.pi, 0.0, 1.0, 0.0), (np.pi, 0.0, 0.0, 1.0),
    (np.pi / 2, 1.0, 0.0, 0.0), (-np.pi / 2, 1.0, 0.0, 0.0),
    (np.pi / 2, 0.0, 1.0, 0.0), (-np.pi / 2, 0.0, 1.0, 0.0),
    (np.pi / 2, 0.0, 0.0, 1.0), (-np.pi / 2, 0.0, 0.0, 1.0),
]


def _axis_angle_quat(angle: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(angle / 2.0) / n
    return np.array([math.cos(angle / 2.0), x * s, y * s, z * s])


def optimize_pose(a: np.ndarray, b: np.ndarray, *, restarts: int = 64,
                  seed: int = 0, polish: bool = False) -> dict:
    """Find the best rigid pose of ``b`` onto ``a`` by independent global search.

    This is how ground truth is generated: float64, and with no relationship to
    the kernel's Adagrad-with-a-travel-budget, so the two agree only if both are
    right.

    **Seeding is the whole difficulty.** The objective is badly multi-modal --
    two molecules may be flipped relative to each other, and a local optimizer
    started at the identity will never find that. So the canonical start
    rotations are used as deterministic seeds, and random restarts are added on
    top. With only random seeding and a handful of restarts this function
    returns an answer that *looks* plausible and is worse than the kernel's,
    which is evidence the reference is unconverged rather than the
    kernel being good.

    Use :func:`optimization_is_converged` to establish that ``restarts`` is
    enough for a given pair, rather than assuming it.

    **Why ``polish`` defaults off, measured rather than guessed.** A Nelder-Mead
    polish after each Powell run costs about 30x and, on one pair of three,
    improved the answer by 4.5e-4 -- which is right at the 5.8e-4 gauge floor, so
    not negligible. But spending that time on *more restarts instead* does
    strictly better: on the pair where the polish mattered,

        16 restarts + polish   0.47609069   ~30 s
        16 restarts, no polish 0.47564183    1.1 s
        64 restarts, no polish 0.47641197    4.9 s   <- better, and 6x faster
       128 restarts, no polish 0.47641197   10.2 s   <- identical
       256 restarts, no polish 0.47641197   20.3 s   <- identical

    The objective is multi-modal, not ill-conditioned: what this needs is more
    basins visited, not each basin resolved more finely. 64 restarts is where it
    plateaus, hence the default.
    """
    from scipy.optimize import minimize

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v_aa, v_bb = self_overlap(a), self_overlap(b)
    rng = np.random.default_rng(seed)

    def neg(x):
        v = pair_overlap(a, apply_pose(b, x[:4], x[4:]))
        return -tanimoto(v, v_aa, v_bb)

    seeds = [np.concatenate([_axis_angle_quat(*s), np.zeros(3)])
             for s in _CANONICAL_STARTS]
    while len(seeds) < restarts:
        q = rng.normal(size=4)
        seeds.append(np.concatenate([q / np.linalg.norm(q),
                                     rng.normal(scale=1.5, size=3)]))

    best_x, best_f = None, np.inf
    for x0 in seeds[:max(restarts, len(_CANONICAL_STARTS))]:
        r = minimize(neg, x0, method="Powell",
                     options={"xtol": 1e-10, "ftol": 1e-12, "maxiter": 40000})
        if polish:
            r = minimize(neg, r.x, method="Nelder-Mead",
                         options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000})
        if r.fun < best_f:
            best_f, best_x = float(r.fun), r.x.copy()

    q, t = best_x[:4], best_x[4:]
    out = score_at_pose(a, b, q, t)
    out["pose_quat"] = q / np.linalg.norm(q)
    out["pose_translation"] = t
    out["restarts"] = len(seeds[:max(restarts, len(_CANONICAL_STARTS))])
    return out


def optimization_is_converged(a: np.ndarray, b: np.ndarray, *,
                              restarts: Sequence[int] = (16, 64),
                              tol: float = 1e-6) -> tuple[bool, dict]:
    """Whether :func:`optimize_pose` has stopped improving with more restarts.

    An unconverged reference is worse than none: it looks like a gold standard
    and quietly makes a correct candidate appear to beat it. The only honest way
    to claim convergence is to show that spending more search does not change
    the answer, which is what this does.
    """
    scores = {}
    for r in restarts:
        scores[r] = optimize_pose(a, b, restarts=r)["tanimoto_shape"]
    vals = [scores[r] for r in restarts]
    return (max(vals) - min(vals) <= tol), scores

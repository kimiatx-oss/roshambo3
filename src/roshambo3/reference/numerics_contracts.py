"""Declared fixed-pose arithmetic and independent forward-error contracts.

This is a comparator, never a production scoring implementation. Physical
constants come from reference.overlap's published expression. No production
constants, measured residuals, or expected output arrays enter these bounds.
A declaration is not qualified merely because it has been instantiated: backend
fault injections must establish that its bounds detect the intended defects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import numpy as np

from .. import numerics

from .color import apply_pose_elementwise, score_columns
from .overlap import ALPHA, COLOR_PREFACTOR, PAIR_PREFACTOR

UNIT_ROUNDOFF = 2.0 ** -24


@dataclass(frozen=True)
class CoordinateGrid:
    """Nearest-even wire grid, physical Angstrom; None means no wire grid.

    The reciprocal and step are separate fp32 operations, as on the wire.
    Overflow is refused rather than silently wrapped or clipped.
    """
    step: float = float(numerics.WIRE_QUANT_STEP)
    bits: int = 16

    def __post_init__(self):
        if not math.isfinite(self.step) or self.step <= 0 or self.bits < 2:
            raise ValueError("grid requires a positive finite step and bits >= 2")


def reconstruct_coordinates(coords, grid: CoordinateGrid | None = None):
    """Reconstruct physical coordinates, retaining each fp32 wire rounding."""
    c = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(c).all():
        raise ValueError("coordinates must be finite")
    if grid is None:
        return c.copy()
    step = np.float32(grid.step)
    wire = np.rint(c.astype(np.float32) * np.float32(1.0 / grid.step))
    if np.any(wire < -(2 ** (grid.bits - 1))) or np.any(wire >= 2 ** (grid.bits - 1)):
        raise ValueError("coordinates overflow the declared wire grid")
    return (wire.astype(np.float32) * step).astype(np.float64)


@dataclass(frozen=True)
class PathArithmetic:
    """Arithmetic facts, supplied by the backend arm rather than host names.

    ``argument_roundings`` bounds the critical exponent path; 16 covers the
    usual scale/subtract/square/add path. ``coefficient_roundings`` covers
    coefficient formation and reporting prefactors. ``sum_roundings`` is an
    optional declared reduction depth; None uses the worst-case serial count.
    ``exp2_relative_error`` must name the actual implementation's guarantee;
    2^-21 conservatively covers two-ULP ex2.approx and faithful libm exp2f.
    Grid reconstruction is separate from pose arithmetic. Color coordinates
    are not put on the atom wire grid: derived centroids need not be on it.
    """
    name: str
    library_grid: CoordinateGrid | None = None
    query_grid: CoordinateGrid | None = None
    rotated_query_grid: CoordinateGrid | None = None
    library_color_grid: CoordinateGrid | None = None
    query_color_grid: CoordinateGrid | None = None
    exp2_relative_error: float = 2.0 ** -21
    argument_roundings: int = 16
    coefficient_roundings: int = 8
    sum_roundings: int | None = None
    pose_contract: str = "physical_fp64"
    # Uniform absolute coordinate uncertainty, Angstrom, for a backend that
    # applies a rounded prepared pose. Must be derived by its adapter.
    coordinate_error: float = 0.0
    operation_model_derivation: str = ""

    def validate_for_certification(self):
        """Refuse an unstated operation model or prepared-pose diagnostic."""
        if not self.operation_model_derivation.strip():
            raise ValueError("certification requires a backend operation-model derivation")
        if self.pose_contract == "shared_fp32_reduced_v1":
            raise ValueError("prepared-byte pose oracle is diagnostic; independently validate pose first")

    def __post_init__(self):
        if self.pose_contract not in ("physical_fp64", "shared_fp32_reduced_v1"):
            raise ValueError("unknown pose contract")
        if not self.name or self.argument_roundings < 0 or self.coefficient_roundings < 0:
            raise ValueError("invalid arithmetic declaration")
        if self.sum_roundings is not None and self.sum_roundings < 0:
            raise ValueError("negative reduction depth")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.exp2_relative_error, self.coordinate_error)):
            raise ValueError("invalid arithmetic error bound")


def scored_coordinates(query, library, *, contract: PathArithmetic,
                       pose_quat=None, pose_translation=None,
                       query_rotation=None, color=False,
                       prepared_matrix=None, prepared_translation=None):
    """Coordinates the cross term and both self terms must describe.

    Physical poses apply to the library. For shared_fp32_reduced_v1 the caller
    supplies the exact prepared reduced matrix/translation bytes; rebuilding
    a quaternion here would hide a split-frame defect. Divide by an independent
    mathematical scale to return physical coordinates for the fp64 oracle.
    This prepared-byte mode is diagnostic: it cannot certify that the backend
    prepared the correct scale or normalized pose. Certification requires an
    independent check of those bytes against the intended physical pose.
    ``query_rotation`` is an optional start-frame rotation before the separate
    rotated-query wire grid. It never implies a color wire grid.
    """
    qgrid = contract.query_color_grid if color else contract.query_grid
    lgrid = contract.library_color_grid if color else contract.library_grid
    a = reconstruct_coordinates(query, qgrid)
    b = reconstruct_coordinates(library, lgrid)
    if query_rotation is not None:
        a = a @ np.asarray(query_rotation, dtype=np.float64).T
        if not color:
            a = reconstruct_coordinates(a, contract.rotated_query_grid)
    if contract.pose_contract == "shared_fp32_reduced_v1":
        if prepared_matrix is None or prepared_translation is None:
            raise ValueError("shared pose requires exact prepared matrix and translation")
        scale = math.sqrt(ALPHA / (2.0 * math.log(2.0)))
        b = (b @ np.asarray(prepared_matrix, dtype=np.float64).T
             + np.asarray(prepared_translation, dtype=np.float64)) / scale
    elif pose_quat is not None:
        if pose_translation is None:
            raise ValueError("pose translation is required with quaternion")
        b = apply_pose_elementwise(b, pose_quat, pose_translation)
    return a, b


def fixed_pose_row(query_shape, library_shape, *, contract: PathArithmetic,
                   query_color=(), query_types=(), library_color=(), library_types=(),
                   interaction_r=None, interaction_p=None, mixing=0.0, **pose):
    """All nine fp64 columns, with same-frame self terms recomputed live."""
    a, b = scored_coordinates(query_shape, library_shape, contract=contract, **pose)
    ac, bc = scored_coordinates(query_color, library_color, contract=contract,
                                color=True, **pose)
    r = np.zeros((1, 1)) if interaction_r is None else interaction_r
    p = np.zeros((1, 1)) if interaction_p is None else interaction_p
    return score_columns(a, b, ac, query_types, bc, library_types, r, p, mixing)


def _gamma(n):
    nu = n * UNIT_ROUNDOFF
    if nu >= 1:
        raise ValueError("rounding model requires n*u < 1")
    return nu / (1.0 - nu)


def _sum_bound(a, b, widths, coefficients, prefactor, contract):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    c = np.broadcast_to(np.asarray(widths) * 0.5, d2.shape)
    coeff = np.broadcast_to(coefficients, d2.shape)
    if np.any(c < 0) or np.any(coeff < 0):
        raise ValueError("positive Gaussian sum required by this error model")
    x = c * d2
    terms = coeff * np.exp(-x)
    # |delta d| <= 2 sqrt(3) eps if each side's components err by eps.
    # Input casting is an absolute coordinate effect. A relative bound in x
    # alone fails for large nearly coincident coordinates (cancellation).
    cast_error = max(float(np.max(np.abs(z.astype(np.float32).astype(np.float64)-z)))
                     for z in (a, b))
    e = 2.0 * math.sqrt(3.0) * (contract.coordinate_error + cast_error)
    dx = _gamma(contract.argument_roundings) * x + c * (2 * np.sqrt(d2) * e + e * e)
    rel = np.expm1(dx) + contract.exp2_relative_error * np.exp(dx)
    rel = (1 + rel) * (1 + _gamma(contract.coefficient_roundings)) - 1
    n = terms.size if contract.sum_roundings is None else contract.sum_roundings
    err = np.sum(terms * rel) + _gamma(n) * np.sum(terms * (1 + rel))
    # FTZ may lose every subnormal term; this additive bound covers even zero
    # fp64 terms, and is negligible at ordinary molecular overlap scales.
    return float(prefactor * (err + terms.size * np.finfo(np.float32).tiny))


def _ratio_bound(v, aa, bb, ev, ea, eb):
    d = aa + bb - v
    ed = ea + eb + ev
    if v == aa == bb == 0:
        return 0.0
    if d <= ed:
        return math.inf  # The rounding interval crosses the denominator guard.
    lo = max(0.0, v - ev) / (d + ed)
    hi = (v + ev) / (d - ed)
    return max(v / d - lo, hi - v / d) + _gamma(4) * hi


def residual_bounds(query_shape, library_shape, *, contract: PathArithmetic,
                    query_color=(), query_types=(), library_color=(), library_types=(),
                    interaction_r=None, interaction_p=None, mixing=0.0, **pose):
    """Per-column absolute rounding bounds, derived before observing a backend.

    Quantization is evaluated exactly in fixed_pose_row, rather than budgeted
    as unexplained error. A missing query grid therefore produces a measurable
    signed residual instead of being admitted by a generic quantization budget.
    """
    a, b = scored_coordinates(query_shape, library_shape, contract=contract, **pose)
    ac, bc = scored_coordinates(query_color, library_color, contract=contract, color=True, **pose)
    ta, tb = np.asarray(query_types, dtype=int), np.asarray(library_types, dtype=int)
    row = fixed_pose_row(query_shape, library_shape, contract=contract,
                         query_color=query_color, query_types=ta, library_color=library_color,
                         library_types=tb, interaction_r=interaction_r,
                         interaction_p=interaction_p, mixing=mixing, **pose)
    def shape(x, y):
        return _sum_bound(x, y, ALPHA, 1.0, PAIR_PREFACTOR, contract)
    def color(x, tx, y, ty):
        if len(x) == 0 or len(y) == 0:
            return 0.0
        r = np.asarray(interaction_r, dtype=np.float64)[tx[:, None], ty[None, :]]
        p = np.asarray(interaction_p, dtype=np.float64)[tx[:, None], ty[None, :]]
        if np.any(r < 0):
            raise ValueError("negative interaction width")
        coeff = np.zeros_like(r)
        np.divide(p*p, np.where(r > 0, r, 1)**1.5, out=coeff, where=r > 0)
        return _sum_bound(x, y, r, coeff, COLOR_PREFACTOR, contract)
    out = dict(zip(("overlap_volume", "self_overlap_query", "self_overlap_library",
                    "overlap_color", "self_color_query", "self_color_library"),
                   (shape(a, b), shape(a, a), shape(b, b), color(ac, ta, bc, tb),
                    color(ac, ta, ac, ta), color(bc, tb, bc, tb))))
    for channel, keys in (("shape", ("overlap_volume", "self_overlap_query", "self_overlap_library")),
                          ("color", ("overlap_color", "self_color_query", "self_color_library"))):
        out["tanimoto_" + channel] = _ratio_bound(*(row[k] for k in keys), *(out[k] for k in keys))
    out["tanimoto_combination"] = ((1-mixing)*out["tanimoto_shape"] + mixing*out["tanimoto_color"]
                                      + _gamma(3)*abs(row["tanimoto_combination"]))
    return out


def residual_report(observed: Sequence[Mapping], expected: Sequence[Mapping],
                    bounds: Sequence[Mapping], *, signed_offsets=None):
    """Report row and mean gates without assuming independent rounding errors.

    Optional signed_offsets are independently computed declared coordinate
    effects (declared oracle minus original-frame oracle). Their signed mean
    shifts the permitted bias interval. Arithmetic uncertainty is correlated:
    its mean bound is the sum of absolute error bounds / N, never / sqrt(N).
    This deterministic bound may be too broad to detect a tiny bias; a backend
    must demonstrate its bias plant before claiming qualification.
    """
    if not observed or len(observed) != len(expected) or len(bounds) != len(expected):
        raise ValueError("nonempty equal row counts required")
    if signed_offsets is not None and len(signed_offsets) != len(expected):
        raise ValueError("signed offset row counts disagree")
    keys = set(expected[0])
    if not keys:
        raise ValueError("nonempty oracle columns required")
    for group in (observed, expected, bounds, signed_offsets or []):
        for row in group:
            if set(row) != keys:
                raise ValueError("inconsistent oracle column sets")
            if not all(np.isscalar(value) and np.isfinite(value) for value in row.values()):
                raise ValueError("finite scalar oracle values required")
    report = {}
    for key in expected[0]:
        residual = np.asarray([o[key]-e[key] for o, e in zip(observed, expected)])
        limit = np.asarray([b[key] for b in bounds])
        offset = np.zeros_like(residual) if signed_offsets is None else np.asarray([s[key] for s in signed_offsets])
        if np.any(limit < 0) or not np.isfinite(limit).all():
            raise ValueError("finite nonnegative bounds required for a qualified gate")
        mean = float(np.mean(residual))
        center, radius = float(np.mean(offset)), float(np.mean(limit))
        report[key] = {"max_absolute_residual": float(np.max(np.abs(residual-offset))),
                       "mean_residual": mean, "mean_interval": (center-radius, center+radius),
                       "mean_bias_qualified": False,
                       "mean_model": "row-implied diagnostic; no independent aggregate bound",
                       "rows_pass": bool(np.all(np.abs(residual-offset) <= limit)),
                       "mean_pass": bool(center-radius <= mean <= center+radius)}
    return report


def assert_residuals(observed, expected, bounds, *, contract=None, certification=False, **kwargs):
    """Assert diagnostic bounds; certification additionally requires a stated model."""
    if certification:
        if contract is None:
            raise ValueError("certification requires an arithmetic contract")
        contract.validate_for_certification()
    report = residual_report(observed, expected, bounds, **kwargs)
    failed = [k for k, v in report.items() if not v["rows_pass"] or not v["mean_pass"]]
    if failed:
        raise AssertionError("fixed-pose residual contract failed: " + ", ".join(failed))
    return report


def fixed_pose_uncertainty(query_shape, library_shape, *, contract: PathArithmetic,
                           **kwargs):
    """Absolute grid displacement plus rounding at this particular fixed pose.

    This is for a physical-input quality qualifier. It is deliberately distinct
    from residual_bounds, whose oracle already applies the declared grid. Grid
    effects are computed live with the independent fp64 expression, not fitted
    from backend residuals. This does not bound an entire optimizer trajectory.
    """
    from dataclasses import replace
    declared = fixed_pose_row(query_shape, library_shape, contract=contract, **kwargs)
    physical = fixed_pose_row(query_shape, library_shape, contract=replace(
        contract, query_grid=None, library_grid=None, rotated_query_grid=None,
        query_color_grid=None, library_color_grid=None), **kwargs)
    arithmetic = residual_bounds(query_shape, library_shape, contract=contract, **kwargs)
    return {k: abs(declared[k] - physical[k]) + arithmetic[k] for k in declared}


# These are diagnostic starting declarations, not qualified backend guarantees.
# An adapter must supply operation_model_derivation before certification. In
# particular vector exp2 polynomials and fields require additional bounds.
CPU_SCALAR_ANALYTIC = PathArithmetic(
    "cpu_scalar_analytic_libm", library_grid=CoordinateGrid(),
    exp2_relative_error=2.0**-22,
)
CUDA_ANALYTIC = PathArithmetic(
    "cuda_analytic_ex2", library_grid=CoordinateGrid(),
    rotated_query_grid=CoordinateGrid(), exp2_relative_error=2.0**-21,
)

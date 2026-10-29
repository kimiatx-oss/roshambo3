"""Coordinate-aware fixed-pose color comparisons for FIELD and ANALYTIC.

The paths may score different query coordinates. Callers must reconstruct the
coordinates each path actually scores, then compare each path with a separate
fp64 oracle whose cross and self terms all use that same coordinate set.
"""
from __future__ import annotations

from roshambo3.reference.numerics_contracts import (
    PathArithmetic,
    assert_residuals,
    fixed_pose_row,
    residual_bounds,
)

COLOR_ORACLE_KEYS = (
    "overlap_color",
    "self_color_query",
    "self_color_library",
    "tanimoto_color",
)


def snap_query_like_cuda(coords, *, step, inverse):
    """Apply the active CUDA wire constants with its float32 rounding points."""
    import numpy as np

    coordinates = np.asarray(coords, dtype=np.float32)
    return (np.rint(coordinates * np.float32(inverse)).astype(np.float32)
            * np.float32(step)).astype(np.float32)


def color_path_contract(name: str) -> PathArithmetic:
    """Fixed-pose color rounding model, with coordinates supplied.

    Grid movement is evaluated from the actual path coordinates before this
    contract is used. The remaining bound covers fp32 argument/coefficient and
    reduction rounding plus the conservative 2^-21 exponential allowance for
    faithful exp2f and CUDA ex2.approx.
    """
    return PathArithmetic(
        name=name,
        exp2_relative_error=2.0 ** -21,
        argument_roundings=16,
        coefficient_roundings=8,
        sum_roundings=16,
        operation_model_derivation=(
            "T270 fixed-pose color expression; exact float32 coordinates are "
            "reconstructed per path before comparison; at most 4 query and 4 "
            "library features yield at most 16 ordered terms; canonical "
            "interaction values; fp32 arithmetic with the T270 2^-21 "
            "exponential allowance."
        ),
    )


def color_oracle_row(query_xyz, query_types, library_xyz, library_types,
                     interaction_r, interaction_p, *, mixing=0.5):
    """Independent fp64 score columns at one path's physical coordinates."""
    import numpy as np

    empty = np.empty((0, 3), dtype=np.float64)
    return fixed_pose_row(
        empty,
        empty,
        contract=color_path_contract("fp64-oracle"),
        query_color=np.asarray(query_xyz, dtype=np.float32),
        query_types=np.asarray(query_types, dtype=np.int32),
        library_color=np.asarray(library_xyz, dtype=np.float32),
        library_types=np.asarray(library_types, dtype=np.int32),
        interaction_r=interaction_r,
        interaction_p=interaction_p,
        mixing=mixing,
    )


def color_row_bounds(query_xyz, query_types, library_xyz, library_types,
                     interaction_r, interaction_p, *, mixing=0.5, name):
    """Absolute per-column arithmetic bounds, independent of observed output."""
    import numpy as np

    empty = np.empty((0, 3), dtype=np.float64)
    return residual_bounds(
        empty,
        empty,
        contract=color_path_contract(name),
        query_color=np.asarray(query_xyz, dtype=np.float32),
        query_types=np.asarray(query_types, dtype=np.int32),
        library_color=np.asarray(library_xyz, dtype=np.float32),
        library_types=np.asarray(library_types, dtype=np.int32),
        interaction_r=interaction_r,
        interaction_p=interaction_p,
        mixing=mixing,
    )


def assert_color_row_matches(observed, expected, bounds):
    """Check relevant color columns against path-local fp64 expectations."""
    keys = COLOR_ORACLE_KEYS
    return assert_residuals(
        [{key: float(observed[key]) for key in keys}],
        [{key: float(expected[key]) for key in keys}],
        [{key: float(bounds[key]) for key in keys}],
    )


def color_denominator_diagnostics(expected, bounds):
    """Return absolute and relative conditioning diagnostics for the ratio."""
    denominator = (expected["self_color_query"]
                   + expected["self_color_library"]
                   - expected["overlap_color"])
    denominator_error = (bounds["self_color_query"]
                         + bounds["self_color_library"]
                         + bounds["overlap_color"])
    scale = (abs(expected["self_color_query"])
             + abs(expected["self_color_library"])
             + abs(expected["overlap_color"]))
    return {
        "denominator": float(denominator),
        "denominator_abs_error_bound": float(denominator_error),
        "denominator_condition_number": float(scale / abs(denominator))
        if denominator != 0 else None,
        "tanimoto_abs_error_bound": float(bounds["tanimoto_color"]),
    }


def assert_cross_path_residual(field_value, analytic_value,
                               field_expected, analytic_expected,
                               field_bound, analytic_bound):
    """Check arithmetic after subtracting the independently computed movement.

    The expected FIELD-to-ANALYTIC difference uses the actual raw and snapped
    coordinates. The remaining residual is only arithmetic movement and gets
    the sum of the two independent absolute bounds.
    """
    residual = ((float(field_value) - float(analytic_value))
                - (float(field_expected) - float(analytic_expected)))
    limit = float(field_bound) + float(analytic_bound)
    if abs(residual) > limit:
        raise AssertionError(
            f"cross-path arithmetic residual {residual:.9g} exceeds "
            f"coordinate-specific bound {limit:.9g}"
        )
    return {"residual": residual, "absolute_bound": limit}

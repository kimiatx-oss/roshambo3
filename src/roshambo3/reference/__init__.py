"""A float64 NumPy/SciPy reference implementation, for tests and ground truth.

Slow, obvious, and independent of the kernel by construction. It replaces
ROSHAMBO2's C++/OpenMP and PyTorch backends, whose real job was never to be a
second engine but to be a second opinion -- and which failed at that for as long
as they were each other's only check, because they shared a typo.

It is not the specification: `doc/NUMERICS_POLICY.md` makes the kernel's Tier-3
arithmetic the specification, so that the reported score comes from one place.
"""

from __future__ import annotations

from .anchor import (OUT_OF_SCOPE, TOLERANCE_SOURCE, TOLERANCE_STATISTIC,
                     anchor_tolerance, per_pair_verdict)
from .color import (COLOR_MODES, FIXED_POSE_TOLERANCE_DERIVATION,
                    FLOAT64_REPRODUCTION_RTOL, MODE_COMBO_OPTIMAL,
                    MODE_SHAPE_ONLY, MODE_SHAPE_POSE_COLOR_RESCORE,
                    apply_pose_elementwise, color_overlap_gradient,
                    fixed_pose_tolerance, guarded_tanimoto,
                    mixed_tanimoto_gradient, pose_gradient_numerical,
                    quat_multiply, score_columns, score_modes,
                    shape_only_columns)
from .overlap import (ALPHA, CARBON_RADIUS, COLOR_PREFACTOR, KAPPA,
                      PAIR_PREFACTOR, apply_pose, color_overlap,
                      optimization_is_converged, optimize_pose, overlap_gradient,
                      pair_overlap,
                      quaternion_matrix, score_at_pose, self_overlap, tanimoto)

__all__ = [
    "OUT_OF_SCOPE", "TOLERANCE_SOURCE", "TOLERANCE_STATISTIC",
    "anchor_tolerance", "per_pair_verdict",
    "pair_overlap", "self_overlap", "tanimoto", "color_overlap",
    "quaternion_matrix", "apply_pose", "score_at_pose",
    "overlap_gradient", "optimize_pose", "optimization_is_converged",
    "KAPPA", "CARBON_RADIUS", "ALPHA", "PAIR_PREFACTOR", "COLOR_PREFACTOR",
    "COLOR_MODES", "MODE_SHAPE_ONLY", "MODE_SHAPE_POSE_COLOR_RESCORE",
    "MODE_COMBO_OPTIMAL", "FLOAT64_REPRODUCTION_RTOL",
    "FIXED_POSE_TOLERANCE_DERIVATION", "apply_pose_elementwise",
    "color_overlap_gradient", "fixed_pose_tolerance", "guarded_tanimoto",
    "mixed_tanimoto_gradient", "pose_gradient_numerical", "quat_multiply",
    "score_columns", "score_modes", "shape_only_columns",
]

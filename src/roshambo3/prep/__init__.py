"""Conformer preparation: the coordinate frame, self-volumes and color atoms."""

from __future__ import annotations

from .build import BuildOptions, prepare, sha256_ids
from .conformer import (
    GAUSS_ALPHA,
    GAUSS_CONST,
    KAPPA,
    make_right_handed,
    prep_conformer,
    prep_rdkit_mol,
    principal_axis_rotation,
    self_volume,
    stable_id,
)

__all__ = [
    "prepare", "BuildOptions", "sha256_ids",
    "principal_axis_rotation", "make_right_handed", "prep_conformer",
    "prep_rdkit_mol", "self_volume", "stable_id",
    "KAPPA", "GAUSS_ALPHA", "GAUSS_CONST",
]

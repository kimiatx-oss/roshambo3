"""Color descriptors and semantic fingerprints for format 3.

The reference module defines the on-disk bytes. This module provides the
query-side API and intentionally delegates both hashes to that reference.
"""

from __future__ import annotations

import warnings
from typing import Any

from . import reference

# These are re-exports by design: the reference owns the canonical definition.
fingerprint = reference.fingerprint
definition_hash = reference.definition_hash


class DefinitionMismatch(ValueError):
    """A corpus and query use different color definitions or parameters."""

    def __init__(self, field: str, message: str | None = None):
        self.field = field
        super().__init__(message or f"color descriptor differs at {field}")


class RDKitVersionWarning(UserWarning):
    """The stored and local RDKit versions differ, but meanings still match."""


def _definition_bytes(color: dict[str, Any]) -> bytes:
    definition = color["definition"]
    if color["generator"] == "rdkit_fdef":
        return reference.normalize_definition(definition).encode("utf-8")
    return reference.canonical_json(definition)


def _matrix_bytes(color: dict[str, Any], name: str) -> bytes:
    return reference.matrix_bytes(color[name]["values"])


def _same_matrix(left: dict[str, Any], right: dict[str, Any], name: str) -> bool:
    try:
        a, b = left[name], right[name]
        return (a["dtype"] == b["dtype"] and a["shape"] == b["shape"]
                and _matrix_bytes(left, name) == _matrix_bytes(right, name))
    except (KeyError, TypeError, ValueError):
        return False


def compare(corpus_color: dict[str, Any] | None,
            query_color: dict[str, Any] | None) -> None:
    """Refuse the first semantic difference between two section-10.2 colors.

    The caller should validate a corpus descriptor before comparison. Query
    descriptors come from ``descriptor_color()`` on a generator.
    """
    if corpus_color is None or query_color is None:
        if corpus_color is not query_color:
            raise DefinitionMismatch("color", "one side has color and the other does not")
        return None

    if corpus_color["types"] != query_color["types"]:
        raise DefinitionMismatch("types")

    if corpus_color["generator"] != query_color["generator"]:
        raise DefinitionMismatch("generator")
    try:
        same_definition = _definition_bytes(corpus_color) == _definition_bytes(query_color)
    except (KeyError, TypeError, ValueError, UnicodeError):
        same_definition = False
    if not same_definition:
        raise DefinitionMismatch("definition")

    if corpus_color["positioning"] != query_color["positioning"]:
        raise DefinitionMismatch("positioning")
    for name in ("r", "p"):
        if not _same_matrix(corpus_color, query_color, name):
            raise DefinitionMismatch(name)

    if corpus_color["rdkit_version"] != query_color["rdkit_version"]:
        warnings.warn(
            "corpus and query use different RDKit versions: "
            f"{corpus_color['rdkit_version']} != {query_color['rdkit_version']}",
            RDKitVersionWarning,
            stacklevel=2,
        )
    return None


def _interaction_pairs(color: dict[str, Any], types: list[str]) -> list[tuple]:
    """Build the generator's interaction list in the descriptor's type order."""
    r = color["r"]["values"]
    p = color["p"]["values"]
    return [
        (types[i - 1], types[j - 1], float(r[i][j]), float(p[i][j]))
        for i in range(1, len(types) + 1)
        for j in range(1, len(types) + 1)
    ]


def _restore_matrices(generator, color: dict[str, Any]) -> None:
    """Restore every stored value, including the unused type-zero cells."""
    import numpy as np

    from roshambo3.numerics import check_interaction_matrix

    generator.interaction_matrix_r = np.asarray(color["r"]["values"], dtype=np.float64)
    generator.interaction_matrix_p = np.asarray(color["p"]["values"], dtype=np.float64)
    generator.interaction_matrix_report = check_interaction_matrix(
        generator.interaction_matrix_r, generator.interaction_matrix_p)


def generator_from_descriptor(color: dict[str, Any]):
    """Build an in-process query generator from stored descriptor text.

    Only a **verified-shipped** RDKit definition is accepted: the gate
    (``roshambo3.definitions.check_definition``, which also runs the pre-pass) must
    report status ``"shipped"``, and the factory is built from the **packaged**
    file, never from the corpus's own text. Anything else raises: an untrusted
    custom definition ``UntrustedDefinitionError``, and a trusted custom one or a
    ``smarts_list`` ``ValueError``, because custom text is perceived only in the
    limited subprocess (``roshambo3.definitions.perceive``).
    """
    return _generator_from_descriptor(color)


def isolated_generator_from_descriptor(color: dict[str, Any], *,
                                       trusted_definitions=()):
    """Build a query generator while keeping custom perception in the sandbox child.

    Shipped defaults use :func:`generator_from_descriptor`. A trusted custom
    definition gets a lightweight adapter whose ``generate_color_atoms`` calls
    :func:`roshambo3.definitions.perceive`; it never builds a feature factory or
    SMARTS molecule in this process. The definition is checked here before the
    adapter can be used, and again by the bounded perception API for each query.
    """
    from roshambo3 import definitions

    verdict = definitions.check_definition(color, trusted=trusted_definitions)
    if verdict.status == "shipped" and color["generator"] == "rdkit_fdef":
        return _generator_from_descriptor(color, verdict=verdict)
    return _IsolatedDescriptorGenerator(color, trusted_definitions)


def _generator_from_descriptor(color: dict[str, Any], *, verdict=None):
    from roshambo3 import definitions

    verdict = verdict or definitions.check_definition(color)
    if verdict.status != "shipped" or color["generator"] != "rdkit_fdef":
        raise ValueError(
            "generator_from_descriptor builds in-process generators for RDKit's shipped "
            "definitions only; a custom definition is perceived through "
            "roshambo3.definitions.perceive (limited subprocess)")
    from roshambo3.pharmacophore import (
        CustomSMARTSPharmacophoreGenerator,
        RDKitPharmacophoreGenerator,
    )

    types = list(color["types"])
    interactions = _interaction_pairs(color, types)
    if color["generator"] == "rdkit_fdef":
        from rdkit.Chem import ChemicalFeatures

        definition = definitions.shipped_text(verdict.hash)     # the packaged file
        factory = ChemicalFeatures.BuildFeatureFactoryFromString(definition)
        features = {family: "rdkit" for family in types}
        generator = RDKitPharmacophoreGenerator(
            features=features,
            interactions=interactions,
            factory=factory,
            definition=definition,
        )
        _restore_matrices(generator, color)
        return generator
    if color["generator"] == "smarts_list":
        definition = color["definition"]
        missing = [family for family in types if family not in definition]
        if missing:
            raise ValueError(f"descriptor types missing from SMARTS definition: {missing}")
        generator = CustomSMARTSPharmacophoreGenerator(
            features=definition,
            interactions=interactions,
            types=types,
        )
        _restore_matrices(generator, color)
        return generator
    raise ValueError(f"unknown color generator {color['generator']!r}")


class _IsolatedDescriptorGenerator:
    """Centroid feature adapter that delegates all custom matching to the sandbox."""

    def __init__(self, color: dict[str, Any], trusted_definitions):
        import copy
        import numpy as np

        from roshambo3.numerics import check_interaction_matrix

        # The adapter can outlive the reader's descriptor object; callers must
        # not be able to mutate the meaning after the trust/compare gate.
        self._color = copy.deepcopy(color)
        self._trusted_definitions = tuple(trusted_definitions)
        self.FEATURES_ENUM = {family: i + 1
                              for i, family in enumerate(self._color["types"])}
        self.interaction_matrix_r = np.asarray(
            self._color["r"]["values"], dtype=np.float64)
        self.interaction_matrix_p = np.asarray(
            self._color["p"]["values"], dtype=np.float64)
        self.interaction_matrix_report = check_interaction_matrix(
            self.interaction_matrix_r, self.interaction_matrix_p)

    def descriptor_color(self):
        import copy
        return copy.deepcopy(self._color)

    def generate_color_atoms(self, mol):
        import numpy as np
        from rdkit import Chem

        from roshambo3 import definitions

        if mol.GetNumConformers() == 0:
            raise ValueError("custom color perception requires a 3D query conformer")
        # Add coordinates to any implicit hydrogens so the limited child sees
        # exactly the atom order used to calculate feature centroids here.
        positioned = Chem.AddHs(Chem.Mol(mol), addCoords=True)
        conformer = positioned.GetConformer()
        block = Chem.MolToMolBlock(positioned, confId=conformer.GetId())
        (perception,) = definitions.perceive(
            self._color, molblocks=[block], trusted=self._trusted_definitions)
        if not perception.parsed:
            raise ValueError("RDKit could not parse a query molecule for color perception")
        positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
        heavy_ids = {atom.GetIdx() for atom in positioned.GetAtoms()
                     if atom.GetAtomicNum() != 1}
        if perception.n_heavy != len(heavy_ids):
            raise ValueError(
                "limited color perception returned a different heavy-atom count")
        found = []
        for feature_type, atom_ids in perception.features or ():
            atom_ids = tuple(int(i) for i in atom_ids)
            if (not atom_ids or len(atom_ids) > 16 or len(set(atom_ids)) != len(atom_ids)
                    or any(i not in heavy_ids for i in atom_ids)):
                raise ValueError(
                    "limited color perception returned a feature with invalid or hydrogen atoms")
            if any(i < 0 or i >= len(positions) for i in atom_ids):
                raise ValueError("limited color perception returned invalid atom indices")
            typ = int(feature_type)
            if not 1 <= typ <= len(self.FEATURES_ENUM):
                raise ValueError("limited color perception returned an invalid feature type")
            atoms = tuple(sorted(atom_ids))
            found.append((typ, atoms, np.mean(positions[list(atoms)], axis=0)))
        found.sort(key=lambda item: (item[0], item[1]))
        coords = np.asarray([point for _, _, point in found], dtype=np.float32).reshape(-1, 3)
        types = np.asarray([typ for typ, _, _ in found], dtype=np.int32)
        return coords, types

"""The one refusal type of format 3 (``doc/FORMAT.md`` section 16).

There is exactly one ``CorpusValidationError`` class: the pure-Python reference
defines it and its definition pre-pass (``reference.check_definition_text``,
the single implementation section 15 names) raises it, so the production
validator re-exports that class rather than defining a second one that an
``except`` clause could miss.
"""

from __future__ import annotations

from .reference import CorpusValidationError

#: Section 16.5, the closed enumeration of ``CorpusValidationError.code``
#: (``limit.<name>`` expanded), including its amendments.
LIMIT_NAMES = (
    "groups", "blocks", "n_conf", "n_mol", "n_real", "n_conf_per_molecule",
    "L_heavy", "L", "L_field_width", "n_types", "features_per_type",
    "features_per_molecule", "atoms_per_feature", "n_feat_atoms", "block_rows",
    "block_length", "bit_width", "definition_size", "definition_lines",
    "definition_line_count", "definition_expansion", "definition_depth",
    "definition_atoms", "definition_patterns", "smiles", "ext_id", "mol_id", "hac",
)
CODES = frozenset((
    "io.not_regular_file", "io.short_read", "io.missing_file", "prologue.magic",
    "prologue.major_version", "prologue.header_offset", "prologue.reserved",
    "json.size", "json.syntax", "json.duplicate_key", "json.nan", "json.depth",
    "json.containers", "json.string_length", "schema.unknown_key", "schema.missing_key", "schema.type",
    "schema.value", "layout.sections", "layout.blocks", "layout.runs",
    "layout.features", "layout.widths", "layout.block_header", "value.self_overlap",
    "value.matrix", "value.definition", "value.parquet", "integrity.fingerprint",
    "integrity.header_hash", "integrity.section_hash", "integrity.block_hash",
    "integrity.padding", "integrity.ids_hash", "integrity.molecules_hash",
    "internal.error", "validation.scan_required",
) + tuple(f"limit.{n}" for n in LIMIT_NAMES))

#: Section 16.5: the reader refuses to hand a corpus to a
#: screen when no ``scan`` result is cached for its content.
SCAN_REQUIRED = "validation.scan_required"

__all__ = ["CorpusValidationError", "CODES", "LIMIT_NAMES", "SCAN_REQUIRED"]

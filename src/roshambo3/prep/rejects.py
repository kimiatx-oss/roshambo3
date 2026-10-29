"""What the builder says on stdout and stderr about the molecules it rejected
and the conformers it dropped: counts by a stable reason code, never a
molecule's identifier or SMILES.

A reject's reason is free text built where the molecule fails
(``sources.parse_stored_smiles``, ``build._why_reject``, an RDKit sanitize
exception), and the SMILES round-trip reasons embed the canonical SMILES.
Build logs are committed as evidence, and an internal corpus's identifiers
and structures must never enter committed text, so the console carries only
``reason_code`` counts and a pointer to the private record: ``build.json``
(never distributed; FORMAT.md 12.1) and the full lists beside the corpus
(``<out>.rejected.jsonl``, ``<out>.dropped_conformers.jsonl``), which keep
every identifier and reason unchanged. ``verbose=True`` (``prepare
--verbose-rejects``, for public corpora) adds the first few inline, as the
builder printed them before.

The code is derived from the reason text alone, so a resumed build -- whose
rejects come back from the checkpoint as ``(identifier, reason)`` pairs --
summarizes them exactly as an uninterrupted one does. A SMILES round-trip
failure's code is chosen from the first differing slot, the only one its
reason text names: ``smiles_radical_mismatch`` means that slot differs in
radical electrons alone, not that no other slot differs otherwise.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

#: ``(code, pattern)`` in match order: the first whose pattern matches the
#: start of the reason wins. Each pattern is anchored on the fixed opening
#: words of one producer's text; a reason none matches is ``other``.
REASON_CODES: tuple = (
    # sources.parse_stored_smiles (FORMAT.md 11.1)
    ("smiles_unparsable",
     r"SMILES round-trip check: its canonical SMILES .* does not parse;"),
    # Chosen from the first differing slot only (the reason text names
    # only that one): radical electrons are the first difference.
    ("smiles_radical_mismatch",
     r"SMILES round-trip check: its canonical SMILES .* does not reproduce the "
     r"stored slots \(slot \d+: radical electrons -?\d+ vs stored -?\d+\);"),
    ("smiles_slot_mismatch",
     r"SMILES round-trip check: its canonical SMILES .* does not reproduce the "
     r"stored slots"),
    # RDKit's sanitize family, as ``f"{type(e).__name__}: {e}"`` (sources)
    ("kekulize", r"(Atom)?KekulizeException: "),
    ("valence", r"AtomValenceException: "),
    ("sanitize", r"(AtomSanitizeException|MolSanitizeException): "),
    # a binary or SQLite payload RDKit could not read: it returned None
    # (here) or raised (any other exception, last row of this table)
    ("unreadable_record", r"RDKit returned None for binary payload"),
    # build._why_not_representable_text (FORMAT.md section 11)
    ("identifier_missing", r"it has no identifier"),
    ("identifier_too_long", r"its identifier is \d+ bytes"),
    ("identifier_characters", r"its identifier carries a control"),
    ("smiles_missing", r"it has no SMILES"),
    ("smiles_not_representable", r"its SMILES \(\d+ characters\) is not"),
    # build._why_reject
    ("no_conformers", r"no conformers$"),
    ("too_many_conformers", r"\d+ conformers exceeds"),
    ("heavy_atom_count_disagrees", r"its conformers disagree on the heavy-atom count"),
    ("no_heavy_atoms", r"no heavy atoms$"),
    ("too_many_heavy_atoms", r"\d+ heavy atoms exceeds"),
    ("too_many_features", r"\d+ color features exceeds"),
    ("feature_atom_count", r"a color feature on \d+ atoms"),
    ("too_many_features_of_one_type", r"\d+ features of one type exceeds"),
    ("molecular_weight", r"its molecular weight "),
    ("rotor_count", r"its rotatable-bond count "),
    ("conformer_shape", r"conformer \d+ has shape "),
    ("nonfinite_coordinates", r"conformer \d+ has non-finite coordinates"),
    ("nonfinite_self_overlap", r"conformer \d+ has a non-finite self-overlap"),
    ("coordinate_out_of_range", r"conformer \d+ has a coordinate at "),
    # Any other exception a source records as ``f"{type(e).__name__}: {e}"``.
    # The sanitize family has its codes above; every other one comes from
    # reading a payload (``Chem.Mol(blob)`` in the binary sources, zstd and
    # ``Chem.Mol`` in the SQLite source), so one failure gets one code
    # whether RDKit raised or returned None.
    ("unreadable_record", r"[A-Za-z_][A-Za-z0-9_]*(Error|Exception): "),
)

_COMPILED = tuple((code, re.compile(pattern, re.DOTALL)) for code, pattern in REASON_CODES)

#: The code of a reason no pattern recognizes.
OTHER = "other"

def reason_code(ident: Optional[str], why: str) -> str:
    """The stable code of one reject's ``(identifier, reason)``. The code
    depends on the reason alone; ``ident`` is accepted so a caller can pass
    a reject list's pairs as they are."""
    why = str(why)
    for code, pattern in _COMPILED:
        if pattern.match(why):
            return code
    return OTHER


def reason_counts(rejected: Iterable) -> list:
    """``[(code, count), ...]``, most frequent first, ties by code."""
    counts = Counter(reason_code(i, w) for i, w in rejected)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def rejected_message(rejected: list, final_dir: str, rejects_file: str,
                     verbose: bool = False) -> str:
    """The console line for the build's rejects: counts by reason code and
    where the identifiers and reasons are; with ``verbose``, the first three
    inline as well (identifiers and reason text, SMILES included)."""
    codes = ", ".join(f"{code} {n}" for code, n in reason_counts(rejected))
    line = (f"roshambo3: rejected {len(rejected)} molecules ({codes}); ids and "
            f"reasons in {final_dir}/build.json (rejected) and {rejects_file}")
    if verbose:
        line += "; first few: " + "; ".join(f"{i} ({w})" for i, w in rejected[:3])
    return line


def dropped_message(dropped: dict, final_dir: str, dropped_file: str,
                    verbose: bool = False) -> str:
    """The console line for dropped conformers: counts, and where the
    identifiers are; with ``verbose``, the first three identifiers inline.
    A conformer dropped for a sanitize failure is also in the reject list,
    counted there under its reason code; an SDF record RDKit could not parse
    at all is counted only here."""
    line = (f"roshambo3: dropped {sum(dropped.values())} conformers of "
            f"{len(dropped)} molecules; ids and counts in {final_dir}/build.json "
            f"(dropped_conformers) and {dropped_file}")
    if verbose:
        line += "; first few: " + "; ".join(
            f"{i} ({n})" for i, n in sorted(dropped.items())[:3])
    return line

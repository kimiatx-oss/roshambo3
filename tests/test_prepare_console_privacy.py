"""The builder's console output names no molecule: rejected molecules and
dropped conformers are reported as counts by a stable reason code, with a
pointer to ``build.json`` and the lists beside the corpus, which keep every
identifier and reason (SMILES included) unchanged. ``--verbose-rejects``
restores the first few inline, for public corpora.

Build logs may be committed as evidence, and an internal corpus's identifiers and structures must never enter
committed text.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from roshambo3 import cli
from roshambo3.prep import build, prepare, rejects, sources
from roshambo3.prep.sources import canonical_heavy, smiles_slot_mismatch
from tests.prep_fixtures import molecule
from tests.test_prepare_perceive_stored_smiles import _thiyl_with_two_radicals
from tests.test_prepare_smiles_roundtrip import (SILICON_SMILES, _write_binary,
                                                 _write_sdf, silicon)

#: Identifiers no path, option or count in the output could contain by chance.
SILICON_ID = "PRIVATE-ID-Q7XK2"
THIYL_ID = "PRIVATE-ID-R4WZ9"
OPTIONS = ["--workers", "1", "--records-per-unit", "3"]


def _thiyl():
    mol = _thiyl_with_two_radicals()
    mol.SetProp("_Name", THIYL_ID)
    return mol


def _library(radical: bool):
    lib = [molecule("CCO", "ethanol"), silicon(name=SILICON_ID),
           molecule("c1ccccc1O", "phenol", conformers=2)]
    return lib + ([_thiyl()] if radical else [])


def _prepare_cli(capfd, source: str, out: Path, *extra) -> str:
    """Run ``roshambo3 prepare`` and return everything it wrote to file
    descriptors 1 and 2 (RDKit's own log included)."""
    capfd.readouterr()
    assert cli.main(["prepare", source, "-o", str(out), "--color", *OPTIONS, *extra]) == 0
    captured = capfd.readouterr()
    return captured.out + captured.err


def _private_text(out: Path) -> str:
    """build.json and the lists beside the corpus: the private record."""
    return "".join(p.read_text() for p in (out / "build.json",
                                           Path(str(out) + build.REJECTS_SUFFIX),
                                           Path(str(out) + build.DROPPED_SUFFIX))
                   if p.exists())


def _secrets(radical: bool) -> list:
    out = [SILICON_ID, SILICON_SMILES]
    if radical:
        out += [THIYL_ID, canonical_heavy(_thiyl())[1]]
    return out


# -- end to end ---------------------------------------------------------------

def test_binary_build_prints_reason_codes_and_no_id_or_smiles(tmp_path, capfd):
    """A binary stream: the silicon molecule's canonical SMILES does not
    parse and the thiyl's loses its radicals. Neither id nor SMILES reaches
    the console (RDKit's parse-error log, which echoes the SMILES, included);
    build.json and the reject list keep both, with the full reason text."""
    source = str(_write_binary(tmp_path / "input.bin.zst", _library(radical=True)))
    out = tmp_path / "corpus"
    text = _prepare_cli(capfd, source, out)
    for secret in _secrets(radical=True):
        assert secret not in text, secret
    assert f" {rejects.OTHER} " not in text
    assert ("roshambo3: rejected 2 molecules (smiles_radical_mismatch 1, "
            "smiles_unparsable 1); ids and reasons in "
            f"{out}/build.json (rejected) and {out}{build.REJECTS_SUFFIX}") in text

    record = json.loads((out / "build.json").read_text())["rejected"]
    whys = {e["id"]: e["why"] for e in record["examples"]}
    assert set(whys) == {SILICON_ID, THIYL_ID} and record["n"] == 2
    assert whys[SILICON_ID] == smiles_slot_mismatch(
        SILICON_SMILES, canonical_heavy(silicon())[0])
    assert canonical_heavy(_thiyl())[1] in whys[THIYL_ID]
    assert sorted(record["reasons"]) == sorted(whys.values())
    lines = [json.loads(x) for x in Path(str(out) + build.REJECTS_SUFFIX)
             .read_text().splitlines()]
    assert {(e["id"], e["why"]) for e in lines} == set(whys.items())


def test_sdf_build_reports_dropped_conformers_without_ids(tmp_path, capfd):
    """An SDF record of the silicon molecule fails RDKit's kekulization per
    conformer: two rejects (kekulize) and two dropped conformers, all by
    count; build.json and both lists name the molecule."""
    source = str(_write_sdf(tmp_path / "input.sdf", _library(radical=False)))
    out = tmp_path / "corpus"
    text = _prepare_cli(capfd, source, out)
    for secret in _secrets(radical=False):
        assert secret not in text, secret
    assert f" {rejects.OTHER} " not in text
    assert "roshambo3: rejected 2 molecules (kekulize 2); ids and reasons in" in text
    assert ("roshambo3: dropped 2 conformers of 1 molecules; ids and counts in "
            f"{out}/build.json (dropped_conformers) and "
            f"{out}{build.DROPPED_SUFFIX}") in text
    record = json.loads((out / "build.json").read_text())
    assert {e["id"] for e in record["rejected"]["examples"]} == {SILICON_ID}
    assert all(e["why"].startswith("KekulizeException: ")
               for e in record["rejected"]["examples"])
    assert record["dropped_conformers"]["examples"] == [{"id": SILICON_ID, "n": 2}]
    assert SILICON_ID in _private_text(out)


def test_verbose_rejects_prints_the_first_few_inline(tmp_path, capfd):
    """``--verbose-rejects`` (public corpora) adds what the builder printed
    by default; build.json is the same either way."""
    source = str(_write_sdf(tmp_path / "input.sdf", _library(radical=False)))
    text = _prepare_cli(capfd, source, tmp_path / "loud", "--verbose-rejects")
    assert f"; first few: {SILICON_ID} (KekulizeException: " in text
    assert f"; first few: {SILICON_ID} (2)" in text
    _prepare_cli(capfd, source, tmp_path / "quiet")

    def content(out):
        b = json.loads((out / "build.json").read_text())
        return b["rejected"]["examples"], b["dropped_conformers"]["examples"]

    assert content(tmp_path / "loud") == content(tmp_path / "quiet")


def test_library_prepare_prints_no_id_or_smiles(tmp_path, capfd):
    """The library call (in-memory molecules) prints the same way."""
    out = tmp_path / "corpus"
    capfd.readouterr()
    prepare(_library(radical=True), str(out), workers=1, records_per_unit=3)
    captured = capfd.readouterr()
    for secret in _secrets(radical=True):
        assert secret not in captured.out + captured.err, secret
    assert "smiles_unparsable 1" in captured.err
    assert f" {rejects.OTHER} " not in captured.err
    assert SILICON_ID in _private_text(out) and SILICON_SMILES in _private_text(out)


# -- the reason codes ---------------------------------------------------------

def _record(**change):
    """A minimal storable record for ``build._why_reject``."""
    rec = SimpleNamespace(ext_id="m", mol_id=1, features=None,
                          meta={"smiles": "C", "mw": 16.0, "rotor_count": 0},
                          conformers=[(np.zeros((1, 3)), 1, 1.0)])
    for k, v in change.items():
        setattr(rec, k, v)
    return rec


def _why_reject_samples() -> dict:
    """``code -> reason``, each produced by ``build._why_reject`` itself."""
    big = build.MAX_CONFORMERS_PER_MOLECULE + 1
    n = build.MAX_N_REAL + 1
    cases = {
        "no_conformers": _record(conformers=[]),
        "too_many_conformers": _record(conformers=[(np.zeros((1, 3)), 1, 1.0)] * big),
        "heavy_atom_count_disagrees": _record(conformers=[(np.zeros((1, 3)), 1, 1.0),
                                                          (np.zeros((2, 3)), 2, 1.0)]),
        "no_heavy_atoms": _record(conformers=[(np.zeros((0, 3)), 0, 1.0)]),
        "too_many_heavy_atoms": _record(conformers=[(np.zeros((n, 3)), n, 1.0)]),
        "too_many_features": _record(features=[(1, (0,))] * (build.MAX_FEATURES_PER_MOLECULE + 1)),
        "feature_atom_count": _record(features=[(1, ())]),
        "identifier_missing": _record(ext_id=""),
        "identifier_too_long": _record(ext_id="x" * (build.MAX_EXT_ID_BYTES + 1)),
        "identifier_characters": _record(ext_id="a\x07b"),
        "smiles_missing": _record(meta={"smiles": "", "mw": 1.0, "rotor_count": 0}),
        "smiles_not_representable": _record(meta={"smiles": "C|C", "mw": 1.0,
                                                  "rotor_count": 0}),
        "molecular_weight": _record(meta={"smiles": "C", "mw": float("nan"),
                                          "rotor_count": 0}),
        "rotor_count": _record(meta={"smiles": "C", "mw": 1.0, "rotor_count": -1}),
        "conformer_shape": _record(conformers=[(np.zeros((2, 3)), 1, 1.0)]),
        "nonfinite_coordinates": _record(conformers=[(np.full((1, 3), np.nan), 1, 1.0)]),
        "nonfinite_self_overlap": _record(conformers=[(np.zeros((1, 3)), 1, np.inf)]),
        "coordinate_out_of_range": _record(conformers=[(np.full((1, 3), 500.0), 1, 1.0)]),
    }
    out = {}
    for code, rec in cases.items():
        why = build._why_reject(rec)
        assert why is not None, code
        out[code] = why
    # Unreachable while the per-molecule feature limit is no larger than the
    # per-type one (both 255); lift the first to reach the second's text.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build, "MAX_FEATURES_PER_MOLECULE", 10_000)
        out["too_many_features_of_one_type"] = build._why_reject(
            _record(features=[(1, (0,))] * (build.MAX_COLOR_TYPE_COUNT + 1)))
    return out


def _source_samples() -> dict:
    """``code -> (identifier, reason)`` from the sources' own producers."""
    from tests.test_prepare_conformer_ids import unkekulizable
    out = {}
    heavy, smiles, _ = canonical_heavy(silicon())
    out["smiles_unparsable"] = ("m", smiles_slot_mismatch(smiles, heavy))
    heavy, smiles, _ = canonical_heavy(molecule("CC(=O)[O-]", "acetate"))
    out["smiles_slot_mismatch"] = ("m", smiles_slot_mismatch("CC(=O)O", heavy))
    heavy, smiles, _ = canonical_heavy(_thiyl())
    out["smiles_radical_mismatch"] = ("m", smiles_slot_mismatch(smiles, heavy))
    failures: list = []
    list(sources._group_by_name([unkekulizable(1)], 0, None, failures))
    out["kekulize"] = failures[0]
    failures = []
    bad = Chem.RWMol(molecule("CC", "valence"))
    bad.GetAtomWithIdx(0).SetFormalCharge(-3)
    bad.GetAtomWithIdx(0).SetNoImplicit(True)
    list(sources._group_by_name([bad.GetMol()], 0, None, failures))
    out["valence"] = failures[0]
    out["unreadable_record"] = ("_record_3", "RDKit returned None for binary payload")
    return out


def test_every_producer_maps_to_its_code():
    samples = {code: ("m", why) for code, why in _why_reject_samples().items()}
    samples.update(_source_samples())
    for code, (ident, why) in samples.items():
        assert rejects.reason_code(ident, why) == code, (code, why)


def test_every_code_has_a_producer_or_a_rule():
    """No code in the table is dead: each is produced above, or is one of the
    exception-class codes whose producer is RDKit itself."""
    produced = set(_why_reject_samples()) | set(_source_samples())
    by_rule = {"sanitize"}
    assert {c for c, _ in rejects.REASON_CODES} <= produced | by_rule


@pytest.mark.parametrize("ident, why, code", [
    ("m", "AtomKekulizeException: non-ring atom 3 marked aromatic", "kekulize"),
    ("m", "AtomValenceException: Explicit valence for atom # 0 C, 5", "valence"),
    ("m", "MolSanitizeException: something", "sanitize"),
    ("m", "AtomSanitizeException: something", "sanitize"),
    ("_record_7", "RuntimeError: Invalid pickle", "unreadable_record"),
    # the SQLite source keeps the row's own id; zstd or RDKit raised
    ("12157944", "ZstdError: could not decompress chunk 0", "unreadable_record"),
    ("12157944", "RDKit returned None for binary payload", "unreadable_record"),
    ("m", "no reason this table knows", rejects.OTHER),
])
def test_exception_and_unknown_reasons(ident, why, code):
    assert rejects.reason_code(ident, why) == code


def test_counts_are_ordered_by_frequency_then_code():
    rejected = [("a", "no conformers"), ("b", "KekulizeException: x"),
                ("c", "KekulizeException: y"), ("d", "no heavy atoms")]
    assert rejects.reason_counts(rejected) == [
        ("kekulize", 2), ("no_conformers", 1), ("no_heavy_atoms", 1)]


def test_the_id_gate_refusal_carries_codes_not_reason_text(tmp_path):
    """``expect_ids_from``'s refusal names the molecule (the operator must
    find it) but not its reason text, which embeds the SMILES."""
    ref = build._IdReference.__new__(build._IdReference)
    ref.root, ref.checked = "ref", 0
    from roshambo3.prep.conformer import stable_id
    ref.ids = np.array([stable_id("B")], dtype=np.int64)
    (tmp_path / "meta_00000.tsv").write_text("")
    why = smiles_slot_mismatch(SILICON_SMILES, canonical_heavy(silicon())[0])
    with pytest.raises(build.BuildRefused) as e:
        ref.check_slot(str(tmp_path), 0, [("B", why)])
    assert "'B (smiles_unparsable)'" in str(e.value)
    assert SILICON_SMILES not in str(e.value)


def test_an_unparsable_smiles_is_not_echoed_by_rdkit(capfd):
    """RDKit logs a SMILES with a syntax error verbatim ("Failed parsing
    SMILES '...'"); the round-trip check parses with its log blocked, and the
    reason, in the private record, carries it instead."""
    heavy = canonical_heavy(molecule("CCO", "ethanol"))[0]
    capfd.readouterr()
    parsed, why = sources.parse_stored_smiles("CC(O", heavy)
    captured = capfd.readouterr()
    assert parsed is None and "'CC(O'" in why
    assert "CC(O" not in captured.out + captured.err
    assert rejects.reason_code("m", why) == "smiles_unparsable"


def _returned_texts(func) -> list:
    """Every string ``func`` returns, read from its source, with each
    ``{...}`` field rendered as ``1`` (the patterns match digits there)."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out.append(v.value)
        elif isinstance(v, ast.JoinedStr):
            out.append("".join(p.value if isinstance(p, ast.Constant) else "1"
                               for p in v.values))
    return out


@pytest.mark.parametrize("func", [build._why_reject, build._why_not_representable_text])
def test_every_returned_reason_has_a_named_code(func):
    """A new ``return "..."`` in the layout checks without a code would be
    summarized as ``other``; this finds it from the source."""
    texts = _returned_texts(func)
    assert len(texts) >= 5
    for text in texts:
        assert rejects.reason_code("m", text) != rejects.OTHER, text

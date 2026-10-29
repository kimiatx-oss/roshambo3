"""Format-3 descriptor naming, content comparison, and query reconstruction."""

from __future__ import annotations

import builtins
import copy
import json
import pathlib
import shutil
import tempfile

import numpy as np
import pytest

from roshambo3 import pharmacophore
from roshambo3.format.r3g import descriptor, reference, validate


DATA = pathlib.Path(__file__).resolve().parent / "data" / "format3"
EXPECTED = json.loads((DATA / "EXPECTED.json").read_text())


def _base_color():
    return pharmacophore.PharmacophoreGenerator().descriptor_color()


class TestFingerprint:
    def test_conformance_fingerprints_match(self):
        for variant in ("shape", "color"):
            corpus = json.loads((DATA / variant / "corpus.json").read_bytes())
            assert descriptor.fingerprint(corpus) == EXPECTED[variant]["fingerprint"]

    def test_order_version_and_fdef_line_endings_do_not_change_fingerprint(self):
        corpus = json.loads((DATA / "color" / "corpus.json").read_bytes())
        expected = descriptor.fingerprint(corpus)

        reversed_order = dict(reversed(list(corpus.items())))
        reversed_order["format"] = dict(reversed(list(corpus["format"].items())))
        reversed_order["frame"] = dict(reversed(list(corpus["frame"].items())))
        reversed_order["prep"] = dict(reversed(list(corpus["prep"].items())))
        reversed_order["prep"]["conventions"] = dict(
            reversed(list(corpus["prep"]["conventions"].items())))
        reversed_order["color"] = dict(reversed(list(corpus["color"].items())))
        assert descriptor.fingerprint(reversed_order) == expected

        for ending in ("\r\n", "\r"):
            changed = copy.deepcopy(corpus)
            changed["color"]["definition"] = changed["color"]["definition"].replace(
                "\n", ending)
            definition_before_hash = changed["color"]["definition"]
            assert descriptor.fingerprint(changed) == expected
            assert changed["color"]["definition"] == definition_before_hash

        changed = copy.deepcopy(corpus)
        changed["color"]["rdkit_version"] = "0.0.0"
        assert descriptor.fingerprint(changed) == expected

        changed = copy.deepcopy(corpus)
        changed["color"]["r"]["values"][1][1] = float(
            np.nextafter(np.float32(1.0), np.float32(2.0)))
        assert descriptor.fingerprint(changed) != expected

    def test_definition_hash_is_the_reference_hash(self):
        color = _base_color()
        assert descriptor.definition_hash(color["definition"]) == (
            "766f9790513f8f94c67ec6d1c8b6d5bede59b81f6da8c2081f57e49fb28d9784")
        assert descriptor.definition_hash("a\r\nb\rc") == descriptor.definition_hash("a\nb\nc")


class TestDescriptorColor:
    def test_default_generator_emits_the_shipped_color_descriptor(self):
        generator = pharmacophore.PharmacophoreGenerator()
        color = generator.descriptor_color()
        with open(generator.fdefName, encoding="utf-8") as definition_file:
            expected_definition = reference.normalize_definition(definition_file.read())
        assert color["definition"] == expected_definition
        assert descriptor.definition_hash(color["definition"]) == (
            "766f9790513f8f94c67ec6d1c8b6d5bede59b81f6da8c2081f57e49fb28d9784")
        assert color["types"] == [name for name, _ in pharmacophore.WORK_DESCENDING_FEATURE_ORDER]
        validate._check_color(color, "test color")

    def test_generator_refuses_noncontiguous_feature_ids(self):
        generator = pharmacophore.PharmacophoreGenerator()
        generator.FEATURES_ENUM["Hydrophobe"] = 9
        with pytest.raises(ValueError, match="exactly 1..n"):
            generator.descriptor_color()


def _perturb(color, field):
    if field == "types":
        color["types"][0], color["types"][1] = color["types"][1], color["types"][0]
    elif field == "definition":
        color["definition"] += "\n# content change"
    elif field == "positioning":
        color["positioning"] = "not-centroid"
    elif field in ("r", "p"):
        color[field]["values"][1][1] = float(
            np.nextafter(np.float32(color[field]["values"][1][1]), np.float32(2.0)))
    else:
        raise AssertionError(field)


@pytest.mark.parametrize(
    "field,side",
    [(field, side) for field in ("types", "definition", "positioning", "r", "p")
     for side in ("corpus", "query")],
    ids=[f"{field}-{side}" for field in ("types", "definition", "positioning", "r", "p")
         for side in ("corpus", "query")],
)
def test_compare_names_the_first_mismatch_on_either_side(field, side):
    corpus_color = _base_color()
    query_color = copy.deepcopy(corpus_color)
    _perturb(corpus_color if side == "corpus" else query_color, field)

    with pytest.raises(descriptor.DefinitionMismatch) as error:
        descriptor.compare(corpus_color, query_color)
    assert error.value.field == field


def test_compare_checks_generator_and_color_presence():
    color = _base_color()
    other = copy.deepcopy(color)
    other["generator"] = "smarts_list"
    with pytest.raises(descriptor.DefinitionMismatch) as error:
        descriptor.compare(color, other)
    assert error.value.field == "generator"

    with pytest.raises(descriptor.DefinitionMismatch) as error:
        descriptor.compare(None, color)
    assert error.value.field == "color"
    assert descriptor.compare(None, None) is None


def test_compare_canonicalizes_smarts_definition_objects():
    r = {"dtype": "f32", "shape": [3, 3], "values": [[0.0] * 3 for _ in range(3)]}
    p = copy.deepcopy(r)
    left = {
        "generator": "smarts_list",
        "definition": {"Beta": ["[O]"], "Alpha": ["[N]"]},
        "types": ["Alpha", "Beta"],
        "positioning": "centroid",
        "r": r,
        "p": p,
        "rdkit_version": "2026.09.1",
    }
    right = copy.deepcopy(left)
    right["definition"] = {"Alpha": ["[N]"], "Beta": ["[O]"]}
    assert descriptor.compare(left, right) is None


def test_rdkit_version_difference_warns_once_without_refusal():
    corpus_color = _base_color()
    query_color = copy.deepcopy(corpus_color)
    query_color["rdkit_version"] = "0.0.0"
    with pytest.warns(descriptor.RDKitVersionWarning) as caught:
        assert descriptor.compare(corpus_color, query_color) is None
    assert len(caught) == 1


def test_scan_still_refuses_carriage_returns_in_stored_fdef(tmp_path):
    from roshambo3.format.r3g import jsonstrict
    from roshambo3.format.r3g.errors import CorpusValidationError

    root = tmp_path / "color"
    shutil.copytree(DATA / "color", root)
    path = root / "corpus.json"
    corpus = json.loads(path.read_bytes())
    corpus["color"]["definition"] = corpus["color"]["definition"].replace("\n", "\r\n")
    path.write_bytes(jsonstrict.dumps(corpus))

    with pytest.raises(CorpusValidationError) as error:
        validate.validate_corpus(str(root), "scan", seed=0)
    assert error.value.code == "value.definition"


def test_group_with_mismatched_fingerprint_is_refused_by_reader():
    from roshambo3.format.r3g.reader import open_corpus
    from roshambo3.format.r3g.errors import CorpusValidationError

    root = DATA / "adversarial" / "fingerprint_mismatch"
    with pytest.raises(CorpusValidationError) as error:
        open_corpus(str(root))
    assert error.value.code == "integrity.fingerprint"


def _embedded_molecules():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecules = []
    for seed, smiles in enumerate(("CCO", "c1ccccc1", "NCC(=O)O"), start=17):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
        molecules.append(mol)
    return molecules


def _feature_signature(generator, mol):
    return [
        (family, tuple(int(i) for i in atom_ids), tuple(float(x) for x in position))
        for family, atom_ids, position in generator.assign_features(mol)
    ]


def test_generator_from_fdef_descriptor_round_trips_without_files(monkeypatch):
    source = pharmacophore.PharmacophoreGenerator()
    color = source.descriptor_color()
    real_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError("generator_from_descriptor must not open a file")

    def forbidden_temp(*args, **kwargs):
        raise AssertionError("generator_from_descriptor must not create a temp file")

    from roshambo3 import definitions
    definitions.shipped_definitions()          # the packaged text is read once, here
    monkeypatch.setattr(builtins, "open", forbidden_open)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", forbidden_temp)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", forbidden_temp)
    reconstructed = descriptor.generator_from_descriptor(color)
    monkeypatch.setattr(builtins, "open", real_open)

    assert reconstructed.FEATURES_ENUM == source.FEATURES_ENUM
    np.testing.assert_array_equal(reconstructed.interaction_matrix_r, source.interaction_matrix_r)
    np.testing.assert_array_equal(reconstructed.interaction_matrix_p, source.interaction_matrix_p)
    for mol in _embedded_molecules():
        assert _feature_signature(reconstructed, mol) == _feature_signature(source, mol)


def test_generator_from_smarts_descriptor_is_refused_in_process():
    """Custom text is perceived only in the limited subprocess."""
    definition = {
        "Unused": ["[S]"],
        "Oxygen": ["[O]"],
        "Nitrogen": ["[N]"],
    }
    selected = ["Nitrogen", "Oxygen"]
    interactions = [
        (a, b, 1.0 if a == b else 0.75, 1.0 if a == b else 0.0)
        for a in selected for b in selected
    ]
    source = pharmacophore.CustomSMARTSPharmacophoreGenerator(
        features=definition, interactions=interactions, types=selected)
    color = source.descriptor_color()
    validate._check_color(color, "source SMARTS color")
    from roshambo3 import definitions
    with pytest.raises(definitions.UntrustedDefinitionError):
        descriptor.generator_from_descriptor(color)


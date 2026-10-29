"""Format-3 regression coverage for results identity."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roshambo3.format import Catalog
from roshambo3.format.r3g import reference, writer
from roshambo3.format.r3g.reader import open_corpus, verify
from roshambo3.results import QueryHits, ScreenResult


def _molecule(mol_id, *, n_conf=2, n_real=3, ext_id="name", smiles="CCO"):
    coords = np.empty((n_conf, 3, n_real), dtype=np.int16)
    for ci in range(n_conf):
        for axis in range(3):
            for atom in range(n_real):
                coords[ci, axis, atom] = (mol_id + ci * 7 + axis * 3 + atom) % 100
    return writer.Molecule(
        mol_id=mol_id, coords=coords, vaa=np.ones(n_conf, dtype=np.float32),
        ext_id=ext_id, smiles=smiles, mw=46.0, rotors=1)


def _write(root, molecules):
    writer.write_corpus(str(root), molecules, color=None, prep_version="t297",
                        prep_commit=None, block_rows=1, max_conf_per_group=2)
    cache = root.parent / "scan-cache"
    verify(str(root), "scan", cache_dir=str(cache))
    return open_corpus(str(root), cache_dir=str(cache))


def test_format3_catalog_round_trips_run_identity_metadata_and_coordinates(tmp_path):
    # Input Parquet order differs from format-3's (bucket, n_real, mol_id) run order.
    corpus = _write(tmp_path / "corpus", [
        _molecule(900, ext_id="=HYPERLINK(\"https://bad\")"),
        _molecule(200),
    ])
    catalog = Catalog(corpus)
    gids = np.arange(catalog.n_conformers, dtype=np.int64)
    refs = catalog.locate(gids)

    assert [(ref.mol_id, ref.conf_idx) for ref in refs] == [
        (200, 0), (200, 1), (900, 0), (900, 1)]
    assert [catalog.gid_for(ref.mol_id, ref.conf_idx) for ref in refs] == gids.tolist()
    assert catalog.mol_ids(gids).tolist() == [200, 200, 900, 900]
    assert catalog.conf_indices(gids).tolist() == [0, 1, 0, 1]

    meta = catalog.molecules([2, 0], columns=["ext_id", "smiles", "hac"])
    assert meta["mol_id"] == [900, 200]
    assert meta["ext_id"][0] == '=HYPERLINK("https://bad")'
    assert meta["smiles"] == ["CCO", "CCO"]
    assert meta["hac"] == [3, 3]

    decoded = catalog.coordinates([3, 0, 2])
    for gid, xyz in zip((3, 0, 2), decoded):
        ref = refs[gid]
        mid = ref.mol_id
        raw = np.asarray([[(mid + ref.conf_idx * 7 + axis * 3 + atom) % 100
                           for atom in range(3)] for axis in range(3)], dtype=np.int16)
        np.testing.assert_array_equal(xyz, raw.T.astype(np.float32) * corpus.step)


def test_format3_reverse_lookup_tracks_identity_when_an_earlier_molecule_shifts_gids(tmp_path):
    base = Catalog(_write(tmp_path / "base", [_molecule(200), _molecule(900)]))
    shifted = Catalog(_write(tmp_path / "shifted", [
        _molecule(100), _molecule(200), _molecule(900)]))

    assert base.gid_for(200, 1) == 1
    assert shifted.gid_for(200, 1) == 3
    assert (base.locate([base.gid_for(200, 1)])[0].mol_id,
            base.locate([base.gid_for(200, 1)])[0].conf_idx) == (200, 1)
    assert (shifted.locate([shifted.gid_for(200, 1)])[0].mol_id,
            shifted.locate([shifted.gid_for(200, 1)])[0].conf_idx) == (200, 1)


def test_screen_result_emits_durable_identity_and_preserves_gid_tie_order(tmp_path):
    corpus = _write(tmp_path / "corpus", [
        _molecule(900, n_conf=1, n_real=1, smiles="C"),
        _molecule(200, n_real=3, smiles="CCO"),
    ])
    catalog = Catalog(corpus)
    gids = np.asarray([catalog.gid_for(900, 0), catalog.gid_for(200, 1)])
    scores = np.zeros((2, 20), dtype=np.float32)
    hit = QueryHits("q", 0, gids, scores, rank_col=0, level="conformer",
                    mol_id=np.asarray([900, 200]), conf_idx=np.asarray([0, 1]))
    result = ScreenResult([hit], corpus=corpus, _catalog=catalog)

    table = result.to_table()
    assert table["gid"] == gids.tolist()
    assert table["mol_id"] == [900, 200]
    assert table["conf_idx"] == [0, 1]
    distinct = result.distinct_molecules_in_hits()
    assert distinct["mol_id"] == [900, 200]
    assert distinct["conf_idx"] == [0, 1]


def test_cli_csv_keeps_formula_like_ext_id_as_text(tmp_path):
    from roshambo3.cli import _write_csv

    corpus = _write(tmp_path / "corpus", [
        _molecule(200, ext_id='=HYPERLINK("https://bad")')])
    catalog = Catalog(corpus)
    gid = np.asarray([catalog.gid_for(200, 0)])
    scores = np.zeros((1, 20), dtype=np.float32)
    result = ScreenResult([
        QueryHits("query", 0, gid, scores, mol_id=np.asarray([200]),
                  conf_idx=np.asarray([0]))], corpus=corpus, _catalog=catalog)
    path = tmp_path / "hits.csv"
    _write_csv(result, str(path), with_names=True)

    with path.open(newline="") as f:
        header, row = list(csv.reader(f))
    assert row[header.index("ext_id")] == "'=HYPERLINK(\"https://bad\")"
    assert row[header.index("mol_id")] == "200"
    assert row[header.index("conf_idx")] == "0"


def test_formula_escape_covers_all_spreadsheet_prefixes():
    from roshambo3.results import spreadsheet_safe_text

    for value in ("=x", "+x", "-x", "@x", "\tx", "\rx"):
        assert spreadsheet_safe_text(value) == "'" + value
    assert spreadsheet_safe_text("plain") == "plain"
    assert spreadsheet_safe_text(-1.0) == -1.0


def test_web_merge_keeps_gid_rank_order_and_compares_by_canonical_identity():
    from roshambo3.web.screening import merge_hits
    from roshambo3.metrics.ranking import canonical_ids

    scores = np.zeros((2, 20), dtype=np.float32)
    mol_ids = [np.asarray([900, 200])]
    conf_indices = [np.asarray([0, 1])]
    first = merge_hits([(scores, np.asarray([7, 8]))], mol_ids, 2, 0,
                       conf_indices=conf_indices)
    shifted = merge_hits([(scores, np.asarray([108, 107]))], mol_ids, 2, 0,
                         conf_indices=conf_indices)

    # Ties preserve the existing gid order, which can change across builds.
    # Compare hit identity through canonical_ids, never through gid.
    np.testing.assert_array_equal(first[3], [900, 200])
    np.testing.assert_array_equal(shifted[3], [200, 900])
    np.testing.assert_array_equal(
        np.sort(canonical_ids(first[3], first[4])),
        np.sort(canonical_ids(shifted[3], shifted[4])))


def test_format3_overlay_uses_smiles_atom_order_and_decoded_coordinate_slots():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from roshambo3.web.chemistry import molblock_at

    mol = Chem.MolFromSmiles("C[C@H](O)F")
    assert AllChem.EmbedMolecule(mol, randomSeed=731) == 0
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    block = molblock_at(mol, xyz, preserve_atoms=True)
    overlaid = Chem.MolFromMolBlock(block, removeHs=False)

    assert [atom.GetAtomicNum() for atom in overlaid.GetAtoms()] == [
        atom.GetAtomicNum() for atom in mol.GetAtoms()]
    np.testing.assert_allclose(overlaid.GetConformer().GetPositions(), xyz, atol=1e-3)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.AssignStereochemistry(overlaid, cleanIt=True, force=True)
    assert mol.GetAtomWithIdx(1).GetProp("_CIPCode") == \
        overlaid.GetAtomWithIdx(1).GetProp("_CIPCode")


def _mock_format3_corpus(color):
    return SimpleNamespace(has_color=True,
                           descriptor={"format": {"major": 3}, "color": color},
                           root="synthetic-format3")


def test_color_gate_compares_descriptor_then_requires_trust():
    color_corpus = open_corpus(
        str(Path(__file__).parent / "data" / "format3" / "color"),
        purpose="inspect")
    color = copy.deepcopy(color_corpus.color)
    custom = copy.deepcopy(color)
    custom["definition"] += "\n# T297 custom marker\n"
    fake_corpus = _mock_format3_corpus(custom)
    fake_generator = SimpleNamespace(descriptor_color=lambda: copy.deepcopy(custom))
    from roshambo3 import definitions
    from roshambo3.format.r3g.descriptor import DefinitionMismatch, generator_from_descriptor

    with pytest.raises(definitions.UntrustedDefinitionError):
        definitions.check_corpus_color(fake_corpus, fake_generator)
    digest = definitions.definition_hash_of(custom)
    verdict = definitions.check_corpus_color(
        fake_corpus, fake_generator, trusted_definitions=[digest])
    assert verdict.status == "trusted" and not verdict.in_process

    mismatched = copy.deepcopy(custom)
    mismatched["positioning"] = "not-the-same"
    with pytest.raises(DefinitionMismatch, match="positioning"):
        definitions.check_corpus_color(
            fake_corpus, SimpleNamespace(descriptor_color=lambda: mismatched),
            trusted_definitions=[digest])
    with pytest.raises(ValueError, match="requires the color generator"):
        definitions.check_corpus_color(fake_corpus, None,
                                       trusted_definitions=[digest])

    shipped = generator_from_descriptor(color_corpus.color)
    assert definitions.check_corpus_color(color_corpus, shipped).status == "shipped"


def test_shape_only_screen_bypasses_color_gate(monkeypatch):
    from roshambo3 import definitions
    from roshambo3.api import _check_color
    from roshambo3.engine import ScreenConfig

    def fail_if_called(*args, **kwargs):
        raise AssertionError("shape-only screens must bypass the color gate")

    monkeypatch.setattr(definitions, "check_corpus_color", fail_if_called)
    _check_color(ScreenConfig(color=False), object(), None)


def _custom_color_corpus(root):
    conformance = open_corpus(
        str(Path(__file__).parent / "data" / "format3" / "color"),
        purpose="inspect")
    color = copy.deepcopy(conformance.color)
    color["definition"] += "\n# T297 isolated custom definition\n"
    writer.write_corpus(str(root), [_molecule(200, n_conf=1)], color=color,
                        prep_version="t297", prep_commit=None, block_rows=1,
                        with_color_self=False)
    cache = root.parent / "custom-scan-cache"
    verify(str(root), "scan", cache_dir=str(cache))
    return open_corpus(str(root), cache_dir=str(cache)), color


def test_api_descriptor_generator_checks_trust_before_query_perception(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from roshambo3 import definitions
    from roshambo3.api import _check_color, _format3_color_generator
    from roshambo3.engine import ScreenConfig

    corpus, color = _custom_color_corpus(tmp_path / "custom")
    digest = definitions.definition_hash_of(color)
    with pytest.raises(definitions.UntrustedDefinitionError):
        _format3_color_generator(corpus, None, ())

    generator = _format3_color_generator(corpus, None, [digest])
    _check_color(ScreenConfig(color=True), corpus, generator,
                 trusted_definitions=[digest])
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=127) == 0
    mol.GetConformer().SetId(17)
    coords, types = generator.generate_color_atoms(mol)
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert types.ndim == 1 and len(types) == len(coords)
    assert set(types.tolist()) <= set(generator.FEATURES_ENUM.values())


def test_isolated_generator_rejects_hydrogen_features_and_sorts_matches(
        tmp_path, monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from roshambo3 import definitions
    from roshambo3.format.r3g.descriptor import _IsolatedDescriptorGenerator

    corpus, color = _custom_color_corpus(tmp_path / "custom")
    generator = _IsolatedDescriptorGenerator(color, ())
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=128) == 0
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
    monkeypatch.setattr(definitions, "perceive", lambda *a, **k: [SimpleNamespace(
        parsed=True, n_heavy=len(heavy), features=[(2, [heavy[1]]), (1, [heavy[2], heavy[0]])])])
    coords, types = generator.generate_color_atoms(mol)
    assert types.tolist() == [1, 2]
    expected = np.asarray(mol.GetConformer().GetPositions())
    np.testing.assert_allclose(coords[0], expected[heavy[0:1] + [heavy[2]]].mean(axis=0))

    hydrogen = next(atom.GetIdx() for atom in mol.GetAtoms()
                    if atom.GetAtomicNum() == 1)
    monkeypatch.setattr(definitions, "perceive", lambda *a, **k: [SimpleNamespace(
        parsed=True, n_heavy=len(heavy), features=[(1, [hydrogen])])])
    with pytest.raises(ValueError, match="hydrogen atoms"):
        generator.generate_color_atoms(mol)


def test_api_prepared_queryset_requires_its_real_generator_and_compares_descriptor(
        tmp_path, monkeypatch):
    from roshambo3 import definitions
    from roshambo3 import backend, api
    from roshambo3.engine import ScreenConfig
    from roshambo3.engine.query import QuerySet
    from roshambo3.format.r3g.descriptor import DefinitionMismatch

    corpus, color = _custom_color_corpus(tmp_path / "custom")
    digest = definitions.definition_hash_of(color)
    # A prebuilt QuerySet carries numeric arrays only, not the descriptor that
    # assigned its color ids. It must not be certified with an inferred corpus
    # generator.
    queries = QuerySet(
        coords=np.zeros((1, 2, 4), dtype=np.float32),
        types=np.zeros((1, 2), dtype=np.int32),
        n_real=np.asarray([1], dtype=np.int32), names=["q"],
        interaction_r=np.asarray(color["r"]["values"], dtype=np.float64),
        interaction_p=np.asarray(color["p"]["values"], dtype=np.float64),
        vaa=np.asarray([1.0], dtype=np.float64))
    monkeypatch.setattr(api, "open_corpus", lambda _: corpus)
    monkeypatch.setattr(backend, "select", lambda _: SimpleNamespace(__name__="fake.backend"))
    with pytest.raises(ValueError, match="prepared QuerySet requires"):
        api.screen(queries, corpus.root,
                   config=ScreenConfig(color=True), backend="cpu")

    generator = api._format3_color_generator(corpus, None, [digest])
    mismatched = copy.deepcopy(color)
    mismatched["positioning"] = "different"
    fake = SimpleNamespace(descriptor_color=lambda: mismatched)
    with pytest.raises(DefinitionMismatch, match="positioning"):
        api.screen(queries, corpus.root, config=ScreenConfig(color=True),
                   color_generator=fake, backend="cpu")
    # A matching, explicitly supplied descriptor passes the API's pre-query
    # gate; query execution is then left to the regular backend path.
    assert api._resolve_color_generator(queries, corpus, generator, [digest]) is generator
    api._check_color(ScreenConfig(color=True), corpus, generator,
                     trusted_definitions=[digest])


def test_web_custom_definition_requires_trust_then_enables_color_modes(
        tmp_path, monkeypatch, capsys):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from roshambo3 import definitions, cli
    from roshambo3.web.app import create_app
    from roshambo3.web.config import Database, Settings
    from roshambo3.web.store import Store

    corpus, color = _custom_color_corpus(tmp_path / "custom")
    db = Database("custom", "Custom", Path(corpus.root))
    jobs = tmp_path / "jobs"
    settings = Settings({"custom": db}, jobs, backend="cpu")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    public = db.public()
    assert public["modes"] == ["shape"]
    assert "definitions trust" in public["color_error"]

    app = create_app(settings, supervise=False)
    with TestClient(app) as client:
        ident = Store(jobs).create({})["id"]
        Store(jobs).update(ident, "ready")
        refused = client.post(
            f"/api/jobs/{ident}/screen",
            json={"database": "custom", "mode": "combo"})
        assert refused.status_code == 422
        assert "definitions trust" in refused.json()["detail"]

        assert cli.main(["definitions", "trust",
                         definitions.definition_hash_of(color)]) == 0
        assert "trusted" in capsys.readouterr().out
        assert db.public()["modes"] == ["shape", "rescore", "combo"]
        accepted = client.post(
            f"/api/jobs/{ident}/screen",
            json={"database": "custom", "mode": "combo"})
        assert accepted.status_code == 202


def test_cli_threads_run_scoped_definition_trust(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from roshambo3 import definitions
    import roshambo3
    from roshambo3 import backend
    from roshambo3 import cli
    from roshambo3.api import _format3_color_generator
    from roshambo3.results import QueryHits

    corpus, color = _custom_color_corpus(tmp_path / "custom")
    digest = definitions.definition_hash_of(color)
    monkeypatch.setattr(backend, "select", lambda _: SimpleNamespace(__name__="fake.backend"))

    class Result:
        stats = SimpleNamespace(n_conformers=1, wall_s=0.01, conformers_per_s=100.0)
        hits = [QueryHits("query", 0, np.asarray([0]), np.zeros((1, 20), np.float32),
                          level="molecule", mol_id=np.asarray([200]),
                          conf_idx=np.asarray([0]))]

        def __iter__(self):
            return iter(self.hits)

        def __len__(self):
            return len(self.hits)

    def screened(*args, **kwargs):
        assert kwargs["color_generator"] is None
        generator = _format3_color_generator(corpus, None, kwargs["trust_definitions"])
        assert generator.descriptor_color()["types"] == color["types"]
        return Result()

    monkeypatch.setattr(roshambo3, "screen", screened)
    argv = ["screen", "unused.sdf", corpus.root, "--mode", "combo",
            "--level", "molecule", "--trust-definition", digest]
    assert cli.main(argv) == 0
    assert "gid=0  mol_id=200  conf_idx=0" in capsys.readouterr().out

    def untrusted(*args, **kwargs):
        return _format3_color_generator(corpus, None, ())

    monkeypatch.setattr(roshambo3, "screen", untrusted)
    assert cli.main(["screen", "unused.sdf", corpus.root, "--mode", "combo"]) == 1
    assert "definitions trust" in capsys.readouterr().err

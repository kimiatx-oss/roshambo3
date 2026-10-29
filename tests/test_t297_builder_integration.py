"""Builder checks through the real ``prepare()`` output."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from rdkit import Chem

from roshambo3.format import Catalog
from roshambo3.prep import prepare
from tests.prep_fixtures import molecule, write_source


def test_builder_color_overlay_uses_canonical_smiles_slots_without_build_json(tmp_path):
    """Every builder molecule overlays its decoded heavy-atom slots in order."""
    from roshambo3.format.r3g.reader import open_corpus, verify
    from roshambo3.pharmacophore import PharmacophoreGenerator
    from roshambo3.web.chemistry import molblock_at

    chiral = molecule("C[C@H](O)C(=O)O", "T297-chiral", conformers=2, seed=731)
    reordered = Chem.RenumberAtoms(chiral, list(reversed(range(chiral.GetNumAtoms()))))
    reordered.SetProp("_Name", "T297-reordered")
    source_mols = [molecule("CO", "T297-methanol", seed=733),
                   molecule("CCN", "T297-amine", seed=739), chiral, reordered]
    root = tmp_path / "color-corpus"
    built = prepare(source_mols, str(root), workers=1,
                    color_generator=PharmacophoreGenerator())
    assert built.has_color
    assert (root / "build.json").is_file()

    # A corpus distributed without provenance must still open and support its
    # catalog. Format 3 requires only corpus.json, molecules.parquet, groups/.
    (root / "build.json").unlink()
    assert sorted(p.name for p in root.iterdir()) == [
        "corpus.json", "groups", "molecules.parquet"]
    verify(str(root), "scan", cache_dir=str(tmp_path / "scan"))
    corpus = open_corpus(str(root), cache_dir=str(tmp_path / "scan"))
    catalog = Catalog(corpus)

    import pyarrow.parquet as pq
    mids = pq.read_table(root / "molecules.parquet", columns=["mol_id"]).column(
        "mol_id").to_pylist()
    for mol_id in mids:
        gid = catalog.gid_for(int(mol_id), 0)
        smiles = catalog.molecules([gid], columns=["smiles"])["smiles"][0]
        source = Chem.MolFromSmiles(smiles)
        assert source is not None
        xyz = catalog.coordinates([gid])[0]
        block = molblock_at(source, xyz, preserve_atoms=True)
        overlay = Chem.MolFromMolBlock(block, removeHs=False)
        assert overlay is not None
        assert [a.GetAtomicNum() for a in overlay.GetAtoms()] == [
            a.GetAtomicNum() for a in source.GetAtoms()]
        np.testing.assert_allclose(overlay.GetConformer().GetPositions(), xyz,
                                   rtol=0, atol=5.1e-5)


def test_builder_screen_identity_is_stable_when_an_earlier_run_shifts_gids(tmp_path):
    """Screen both real builds fully, then compare common hits by pair identity."""
    from roshambo3 import screen
    from roshambo3.metrics.ranking import canonical_ids

    common = [molecule("CCO", "T297-common-ethanol", conformers=2, seed=743),
              molecule("CCN", "T297-common-amine", conformers=2, seed=751),
              molecule("c1ccccc1", "T297-common-benzene", seed=757)]
    earlier = molecule("C", "T297-earlier", conformers=1, seed=761)
    common_sdf, earlier_sdf, query_sdf = (tmp_path / name for name in
        ("common.sdf", "earlier.sdf", "query.sdf"))
    write_source(common_sdf, common)
    write_source(earlier_sdf, [earlier])
    write_source(query_sdf, [molecule("CCO", "T297-query", seed=743)])

    base = prepare(str(common_sdf), str(tmp_path / "base"), workers=1)
    shifted = prepare([str(earlier_sdf), str(common_sdf)],
                      str(tmp_path / "shifted"), workers=1)
    base_catalog, shifted_catalog = Catalog(base), Catalog(shifted)
    base_ids = {int(mid) for mid in base_catalog.mol_ids(np.arange(base.n_conf))}
    common_ids = base_ids
    assert all(shifted_catalog.gid_for(mid, 0) != base_catalog.gid_for(mid, 0)
               for mid in common_ids)

    # K exceeds either corpus size, so the inserted molecule cannot crowd a
    # common pair out of the result. Both screens consume the same query file.
    first = screen(str(query_sdf), base, backend="cpu", workers=1, k=64)
    second = screen(str(query_sdf), shifted, backend="cpu", workers=1, k=64)
    h1, h2 = first[0], second[0]
    mask1 = np.isin(h1.mol_id, list(common_ids))
    mask2 = np.isin(h2.mol_id, list(common_ids))
    pairs1 = list(zip(h1.mol_id[mask1].astype(int), h1.conf_idx[mask1].astype(int)))
    pairs2 = list(zip(h2.mol_id[mask2].astype(int), h2.conf_idx[mask2].astype(int)))

    # Compare raw pairs as well as dense canonical ids: the latter alone could
    # conceal a substitution of one molecule/conformer for another.
    assert pairs1 == pairs2
    dense1 = canonical_ids(h1.mol_id[mask1], h1.conf_idx[mask1])
    dense2 = canonical_ids(h2.mol_id[mask2], h2.conf_idx[mask2])
    np.testing.assert_array_equal(dense1, dense2)
    np.testing.assert_array_equal(h1.scores[mask1], h2.scores[mask2])
    gids1 = dict(zip(pairs1, h1.gids[mask1].astype(int)))
    gids2 = dict(zip(pairs2, h2.gids[mask2].astype(int)))
    assert all(gids1[pair] != gids2[pair] for pair in pairs1)


def test_builder_cli_csv_keeps_formula_like_ext_id_as_text(tmp_path, capsys):
    """The formula escape applies to ext_id values emitted by prepare()."""
    from roshambo3.cli import main

    names_and_smiles = [
        ('=HYPERLINK("https://bad")', "CCO"),
        ("+SUM(1,1)", "CC"),
        ("-1+2", "CCN"),
        ("@SUM(1,1)", "CO"),
    ]
    library = [molecule(smiles, name, seed=769 + i)
               for i, (name, smiles) in enumerate(names_and_smiles)]
    query = Chem.Mol(library[0])
    query.SetProp("_Name", "T297-query")
    library_sdf, query_sdf = tmp_path / "library.sdf", tmp_path / "query.sdf"
    write_source(library_sdf, library)
    write_source(query_sdf, [query])
    corpus = prepare(str(library_sdf), str(tmp_path / "corpus"), workers=1)
    output = tmp_path / "hits.csv"
    argv = ["screen", str(query_sdf), corpus.root, "--backend", "cpu",
            "--mode", "shape", "--workers", "1", "--steps", "25",
            "-k", "8", "--out", str(output), "--names"]

    assert main(argv) == 0, capsys.readouterr().err
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["ext_id"] for row in rows} == {
        "'" + name for name, _ in names_and_smiles}
    assert all(row["conf_idx"] == "0" for row in rows)
    assert {row["mol_id"] for row in rows} == {
        str(int(mol_id)) for mol_id in Catalog(corpus).mol_ids(
            np.arange(corpus.n_conf))}

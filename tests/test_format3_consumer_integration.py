"""Public format-3 consumer acceptance without the pending builder.

The writer fixture uses real RDKit features and scored-coordinate self values;
these are correctness checks, not accuracy or throughput measurements.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3 import definitions
from roshambo3.format.r3g.reader import open_corpus, verify
from roshambo3.format.r3g.writer import Molecule, write_corpus
from roshambo3.pharmacophore import RDKitPharmacophoreGenerator
from roshambo3.prep.conformer import stored_self_volume


@pytest.fixture
def custom_screen_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('ROSHAMBO3_SCAN_CACHE', str(tmp_path / 'scan'))
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=127) == 0
    mol.SetProp('_Name', 'query')
    generator = RDKitPharmacophoreGenerator()
    for matrix in (generator.interaction_matrix_r, generator.interaction_matrix_p):
        matrix[:] = np.eye(len(matrix))
        matrix[0, 0] = 0
    color = copy.deepcopy(generator.descriptor_color())
    color['definition'] += '\n# Consumer integration custom definition\n'
    digest = definitions.definition_hash_of(color)
    perception, = definitions.perceive(color, molblocks=[Chem.MolToMolBlock(mol)], trusted=[digest])
    features = sorted((int(t), tuple(sorted(map(int, atoms))))
                      for t, atoms in perception.features)
    assert features and all(max(atoms) < 3 for _, atoms in features)
    xyz = np.rint(np.asarray(mol.GetConformer().GetPositions())[:3] * 256).astype(np.int16)
    coords = xyz[None].transpose(0, 2, 1).copy()
    vaa = np.asarray([stored_self_volume(xyz.astype(np.float32) / 256)], np.float32)
    root = tmp_path / 'corpus'
    write_corpus(str(root), [Molecule(200, coords, vaa, features,
        ext_id='=HYPERLINK("example")', smiles='CCO', mw=46.0, rotors=0)],
        color=color, prep_version='consumer-test', prep_commit=None,
        block_rows=1, with_color_self=False)
    verify(str(root), 'scan')
    sdf = tmp_path / 'query.sdf'
    with Chem.SDWriter(str(sdf)) as writer:
        writer.write(mol)
    return open_corpus(str(root)), mol, sdf, digest


def test_public_api_custom_trust_and_durable_hits(custom_screen_corpus):
    from roshambo3 import screen
    from roshambo3.backend import ColorPlacement
    from roshambo3.engine import ScreenConfig
    corpus, mol, _, digest = custom_screen_corpus
    config = ScreenConfig(color=True, color_placement=ColorPlacement.IN_GRADIENT,
                          mixing=.5, steps=25)
    with pytest.raises(definitions.UntrustedDefinitionError):
        screen([mol], corpus, backend='cpu', config=config, workers=1, k=1)
    for persistent in (False, True):
        if persistent:
            definitions.trust(digest)
        result = screen([mol], corpus, backend='cpu', config=config, workers=1,
                        k=1, trust_definitions=() if persistent else [digest])
        assert result[0].mol_id.tolist() == [200]
        assert result[0].conf_idx.tolist() == [0]
        assert np.isfinite(result[0].scores).all()
        assert result.to_table()['mol_id'] == [200]
        assert result.hit_molecules()['smiles'] == ['CCO']
        assert result.aligned_poses()[0].shape == (3, 3)


def test_real_cli_custom_trust_csv(custom_screen_corpus, tmp_path, capsys):
    from roshambo3.cli import main
    corpus, _, sdf, digest = custom_screen_corpus
    output = tmp_path / 'hits.csv'
    argv = ['screen', str(sdf), corpus.root, '--backend', 'cpu', '--mode', 'combo',
            '--workers', '1', '--steps', '25', '-k', '1', '--out', str(output), '--names']
    assert main(argv) == 1
    assert 'trust' in capsys.readouterr().err.lower()
    for persistent in (False, True):
        if persistent:
            assert main(['definitions', 'trust', digest]) == 0
        assert main(argv + ([] if persistent else ['--trust-definition', digest])) == 0
        with output.open(newline='') as stream:
            row, = list(csv.DictReader(stream))
        assert row['mol_id'] == '200' and row['conf_idx'] == '0'
        assert row['ext_id'].startswith("'=HYPERLINK")


def test_real_web_screen_persists_identity_and_semantic_fingerprint(custom_screen_corpus, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from roshambo3.web.config import Database, Settings
    from roshambo3.web.screening import run_screen
    from roshambo3.web.store import Store
    corpus, _, sdf, digest = custom_screen_corpus
    definitions.trust(digest)
    jobs = tmp_path / 'jobs'
    settings = Settings({'custom': Database('custom', 'Custom', Path(corpus.root))},
                        jobs, backend='cpu', cpu_workers=1)
    store = Store(jobs)
    ident = store.create({'database': 'custom', 'mode': 'combo', 'k': 1, 'mixing': .5})['id']
    directory = jobs / ident
    (directory / 'query.sdf').write_bytes(sdf.read_bytes())
    store.update(ident, 'queued', preparation={})
    run_screen(settings, ident)
    assert store.get(ident)['state'] == 'completed', store.get(ident)
    rows = json.loads((directory / 'hits.json').read_text())
    assert rows[0]['mol_id'] == '200' and rows[0]['conf_idx'] == 0
    for filename in ('scores.npz', 'poses.npz'):
        with np.load(directory / filename) as data:
            assert data['mol_ids'].tolist() == [200]
            assert data['conf_indices'].tolist() == [0]
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['corpus_sha256'] == corpus.fingerprint
    # Exercise the real format-3 overlay route, including durable reverse
    # lookup and SMILES atom order, rather than its molblock helper alone.
    from fastapi.testclient import TestClient
    from roshambo3.web.app import create_app
    import io
    with TestClient(create_app(settings, supervise=False)) as client:
        response = client.get(f'/api/jobs/{ident}/hits/1/overlay')
        assert response.status_code == 200, response.text
        hit = Chem.MolFromMolBlock(response.json()['hit'], removeHs=False)
        assert [a.GetSymbol() for a in hit.GetAtoms()] == ['C', 'C', 'O']
        with np.load(directory / 'poses.npz') as data:
            expected = data['xyz'].copy()
        # Molblock serializes decimal coordinates to four places.
        np.testing.assert_allclose(hit.GetConformer().GetPositions(), expected,
                                   rtol=0, atol=5.1e-5)
        response = client.get(f'/api/jobs/{ident}/export/csv')
        assert response.status_code == 200
        row, = list(csv.DictReader(io.StringIO(response.text)))
        assert row['mol_id'] == '200' and row['conf_idx'] == '0'
        assert row['name'].startswith("'=HYPERLINK")
        response = client.get(f'/api/jobs/{ident}/export/sdf?ranks=1')
        assert response.status_code == 200, response.text
        records = list(Chem.ForwardSDMolSupplier(io.BytesIO(response.content)))
        assert len(records) == 2
        assert all(m.GetProp('mol_id') == '200' and m.GetProp('conf_idx') == '0'
                   for m in records)

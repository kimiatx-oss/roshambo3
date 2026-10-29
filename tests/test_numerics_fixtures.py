"""Input provenance/feature and independent ambiguity checks (CPU only)."""
import json

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("scipy")

from bench.numerics_fixtures import (AlignmentFixture, DATA, FAMILIES, load_panel,
                                      objective_function, oracle_uncertainty,
                                      qualify_fixture)
from roshambo3.reference.color import score_columns


def test_real_panel_is_tiny_attributed_and_covers_feature_geometry():
    manifest = json.loads((DATA / "manifest.json").read_text())
    files = list(DATA.glob("*"))
    assert sum(p.stat().st_size for p in files if p.is_file()) <= 1024 * 1024
    assert all(p.stat().st_size <= 256 * 1024 for p in files if p.is_file())
    assert manifest["source"]["sha256"]
    assert manifest["license"]["status"]
    assert manifest["recipe"]
    fixtures = load_panel()
    assert len(fixtures) == 4
    assert len({e["canonical_isomeric_smiles"] for e in manifest["molecules"]}) == 4
    assert set(np.concatenate([f.types for f in fixtures])) == set(range(1, 7))
    for entry, fixture in zip(manifest["molecules"], fixtures):
        assert entry["pdb_id"] in entry["source_member"]
        assert len(entry["source_member_sha256"]) == 64
        assert len(fixture.shape) == entry["heavy_atoms"] <= 48
        assert np.max(np.abs(fixture.shape.mean(axis=0))) < 1e-12
        assert all((fixture.types == i).sum() == entry["features"][family]
                   for i, family in enumerate(FAMILIES, 1))
        singular = np.linalg.svd(fixture.shape, compute_uv=False)
        assert singular[-1] / singular[0] >= .08
        assert any(len(ids) > 1 for ids in fixture.feature_atom_ids)
        assert any(len(ids) == 1 for ids in fixture.feature_atom_ids)
        assert max(np.bincount(fixture.types)) >= 2


def test_fixed_pose_objectives_reuse_existing_color_reference():
    from scipy.spatial.transform import Rotation
    pose = np.array([.4, -.2, .3, .7, -.5, .1])
    q_xyzw = Rotation.from_rotvec(pose[:3]).as_quat()
    q = q_xyzw[[3, 0, 1, 2]]
    for fixture in load_panel():
        row = score_columns(fixture.shape, fixture.shape, fixture.color, fixture.types,
                            fixture.color, fixture.types, fixture.interaction_r,
                            fixture.interaction_p, .5, q, pose[3:])
        assert objective_function(fixture, "shape")(pose) == pytest.approx(row["tanimoto_shape"], abs=1e-14)
        assert objective_function(fixture, "combo")(pose) == pytest.approx(row["tanimoto_combination"], abs=1e-14)
        for objective in ("shape", "combo"):
            assert 0 < oracle_uncertainty(fixture, objective) < 1e-6
            # Identity=1 is a mathematical self-overlap invariant, not stored output.
            assert objective_function(fixture, objective)(np.zeros(6)) == pytest.approx(1., abs=1e-14)


@pytest.mark.parametrize("scale,perturbation", [(1.,0.), (1.,.001), (.05,0.)])
@pytest.mark.parametrize("objective", ["shape", "combo"])
def test_nonplanar_symmetric_and_nearly_symmetric_controls_are_rejected(scale, perturbation, objective):
    shape = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
    shape *= scale
    shape[0, 0] += perturbation
    shape -= shape.mean(axis=0)
    fixture = AlignmentFixture("tetrahedral_control", shape, shape.copy(), np.ones(4, dtype=np.int64))
    receipt = qualify_fixture(fixture, objective)
    assert receipt["complete"]
    assert receipt["outside_count"] > 0
    assert receipt["outside_gap"] < receipt["minimum_gap_design"]
    assert not receipt["qualified"]
    assert not receipt["production_qualified"]


def test_smoke_search_never_claims_full_qualification():
    fixture = load_panel()[0]
    receipt = qualify_fixture(fixture, "shape", starts=2)
    assert not receipt["qualified"]
    assert not receipt["production_qualified"]
    assert receipt["missing"]


def test_recovery_neighborhood_has_independent_design_and_grid_derivation():
    from bench.numerics_fixtures import RECOVERY_NEIGHBORHOOD, protocol
    assert RECOVERY_NEIGHBORHOOD == .01 + 2*np.sqrt(3)*.0005
    assert protocol()["pose_neighborhood_rmsd_angstrom"] == RECOVERY_NEIGHBORHOOD
    assert RECOVERY_NEIGHBORHOOD < .05


def test_portable_qualification_rejects_changed_inputs_protocol_and_endpoint_bytes(tmp_path, monkeypatch):
    import shutil
    from dataclasses import replace
    from bench.numerics_fixtures import write_qualification, load_qualification

    for path in DATA.iterdir():
        if path.is_file():
            shutil.copy2(path, tmp_path / path.name)
    fixture = load_panel(tmp_path)[0]
    smoke = qualify_fixture(fixture, "shape", starts=2)
    write_qualification([smoke],tmp_path)
    # A smoke panel cannot be relabeled as a full128-start qualification.
    with pytest.raises(ValueError,match="incomplete"):
        load_qualification(fixture,"shape",tmp_path)
    # A release-name change alone does not change the certified mathematics.
    index_path=tmp_path/"qualification.json"
    index=json.loads(index_path.read_text())
    index["receipts"][0]["binding"]["rdkit_version"]="other release, identical content"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError,match="incomplete"):
        load_qualification(fixture,"shape",tmp_path)
    with monkeypatch.context() as patch:
        original=fixture.interaction_r
        patch.setattr(AlignmentFixture,"interaction_r",property(lambda self:original*.9))
        with pytest.raises(ValueError,match="binding mismatch"):
            load_qualification(fixture,"shape",tmp_path)
    for attribute in ("shape","color","types"):
        changed = getattr(fixture,attribute).copy()
        changed.flat[0] += 1
        altered = replace(fixture,**{attribute:changed})
        with pytest.raises(ValueError,match="binding mismatch"):
            load_qualification(altered,"shape",tmp_path)
    index_path=tmp_path/"qualification.json"
    index=json.loads(index_path.read_text())
    artifact=tmp_path/index["receipts"][0]["endpoints_file"]
    payload=artifact.read_bytes()
    artifact.write_bytes(payload+b"tampered")
    with pytest.raises(ValueError,match="checksum mismatch"):
        load_qualification(fixture,"shape",tmp_path)
    artifact.write_bytes(payload)
    protocol_path=tmp_path/"protocol.json"
    protocol_path.write_text(protocol_path.read_text()+"\n")
    with pytest.raises(ValueError,match="binding mismatch"):
        load_qualification(fixture,"shape",tmp_path)


def test_endpoint_summaries_are_recomputed_instead_of_trusting_flags():
    from bench.numerics_fixtures import summarize_endpoints
    fixture=load_panel()[0]
    # Planted claims describe a nonstationary pose as converged and perfect.
    endpoints=[dict(pose_rotvec=[.4,-.2,.3],pose_translation=[.7,-.5,.1],
                    score=1.,rmsd=0.,gradient_inf=0.,stationary=True,
                    translation_in_envelope=True,optimizer_success=True,iterations=1)]*128
    receipt=summarize_endpoints(fixture,"shape",endpoints)
    assert not receipt["complete"]
    assert not receipt["qualified"]
    assert all(not e["stationary"] for e in receipt["endpoints"])


def test_all_portable_real_qualifications_recompute_complete_and_separated():
    from bench.numerics_fixtures import load_qualification, RECOVERY_NEIGHBORHOOD
    for fixture in load_panel():
        for objective in ("shape","combo"):
            receipt=load_qualification(fixture,objective)
            assert receipt["complete"] and receipt["qualified"]
            assert receipt["starts"] == len(receipt["endpoints"]) == 128
            assert receipt["pose_neighborhood_rmsd_angstrom"] == RECOVERY_NEIGHBORHOOD
            assert receipt["outside_count"] > 0
            assert receipt["outside_gap"] > receipt["minimum_gap_design"]+2*receipt["oracle_uncertainty"]


def test_canonical_centroids_derive_from_source_atoms_and_check_rdkit_roundoff():
    import math
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
    from bench.numerics_fixtures import canonical_centroid, centroid_roundoff_bound
    manifest=json.loads((DATA/"manifest.json").read_text())
    for entry,fixture in zip(manifest["molecules"],load_panel()):
        mol=next(iter(Chem.SDMolSupplier(str(DATA/entry["file"]),removeHs=False)))
        source=np.asarray(mol.GetConformer().GetPositions())
        heavy=source[[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum()>1]]
        center=np.array([math.fsum(map(float,heavy[:,axis]))/len(heavy) for axis in range(3)])
        assert np.array_equal(fixture.shape,heavy-center)
        for point,ids in zip(fixture.color,fixture.feature_atom_ids):
            atoms=source[list(ids)]
            expected=np.array([math.fsum(map(float,atoms[:,axis]))/len(atoms) for axis in range(3)])
            assert np.array_equal(point,expected-center)
            # fsum mean does not depend on an atom-ID enumeration direction.
            assert np.array_equal(canonical_centroid(atoms),canonical_centroid(atoms[::-1]))
            assert 0<centroid_roundoff_bound(atoms)<1e-10


@pytest.mark.parametrize("displacement", [1e-9, float("nan"), None])
def test_changed_centroid_semantics_cannot_hide_behind_coordinate_hashes(monkeypatch, displacement):
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from bench.numerics_fixtures import uniform_feature_weights,from_molecule
    with pytest.raises(ValueError,match="nonuniform"):
        uniform_feature_weights("DefineFeature Changed [C][O]\n Family Aromatic\n Weights 1,2\nEndFeature")
    entry=json.loads((DATA/"manifest.json").read_text())["molecules"][0]
    mol=next(iter(Chem.SDMolSupplier(str(DATA/entry["file"]),removeHs=False)))
    original=ChemicalFeatures.BuildFeatureFactory
    class ChangedFeature:
        def __init__(self,feature): self.feature=feature
        def __getattr__(self,name): return getattr(self.feature,name)
        def GetPos(self):
            point=np.asarray(self.feature.GetPos(),dtype=np.float64)
            return point[:2] if displacement is None else point+np.array([displacement,0.,0.])
    class ChangedFactory:
        def __init__(self,factory): self.factory=factory
        def GetFeaturesForMol(self,molecule):
            return [ChangedFeature(f) for f in self.factory.GetFeaturesForMol(molecule)]
    monkeypatch.setattr(ChemicalFeatures,"BuildFeatureFactory",lambda path:ChangedFactory(original(path)))
    with pytest.raises(ValueError,match="beyond fp64 rounding"):
        from_molecule(mol,entry["name"])

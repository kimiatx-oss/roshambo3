"""Portable guards for preregistered direct-point FIELD controls."""
from argparse import Namespace
import json
from pathlib import Path
import numpy as np
import pytest
from bench import numerics_field_quality_v2 as v2
from roshambo3.reference.field_quality_v2 import point_control,wa4,search_all
from roshambo3.reference import field_quality as f


def test_direct_point_rounding_has_derived_physical_displacement_bound():
    rng=np.random.default_rng(276100)
    cloud=rng.normal(size=(30,3));pose=np.r_[rng.normal(size=4),rng.uniform(-1,1,3)]
    points,guard=point_control(cloud,pose)
    assert points.dtype==np.float32 and points.flags.c_contiguous
    assert 0<guard['max_displacement']<=guard['max_bound']<1e-5
    with pytest.raises(ValueError,match='extent'):point_control(cloud,[1,0,0,0,17,0,0])


def test_wa4_matches_declared_serial_xor_order_and_not_sorted_sum():
    values=np.array([2**24,1,2,4],np.float32)
    # Production XOR1 then XOR2 gives (2^24 + 1) + (2 + 4).
    assert wa4(values)==np.float32(16777222)
    # The previously incorrect XOR2 then XOR1 gives (2^24 + 2) + (1 + 4).
    reverse=np.float32(np.float32(values[0]+values[2])+np.float32(values[1]+values[3]))
    assert reverse==np.float32(16777224) and reverse!=wa4(values)


def test_native_probe_packs_each_original_endpoint_and_keeps_four_starts():
    class Extension:
        def numerics_sample_field(self,bits,points,origin,spacing):
            assert points.dtype==np.float32
            return np.column_stack((points[:,0],np.zeros((len(points),3)))).astype(np.float32)
    cases=[dict(a=np.zeros((1,3)),b=np.array([[1.,0,0],[2.,0,0]]))]
    poses=np.tile([1.,0,0,0,0,0,0],(1,4,1));poses[0,:,4]=np.arange(4)
    result=v2.probe(cases,poses,[(np.zeros((2,2,2,4),np.uint16),np.zeros(3),1.)],Extension())
    expected=np.array([[wa4(point_control(cases[0]['b'],p)[0][:,0]) for p in poses[0]]])
    np.testing.assert_array_equal(result,expected)


def test_four_endpoint_search_retains_start_identity_on_zero_gradient():
    field=np.zeros((4,4,4,4));query=np.array([[0.,0,0,1]])
    poses=search_all(query,np.array([[0.,0,0]]),field,np.array([-1.,-1,-1]),1.)
    np.testing.assert_array_equal(poses[:,:4],np.eye(4));np.testing.assert_array_equal(poses[:,4:],0)


def test_frozen_limits_keep_signed_regret_and_cannot_use_candidate_values():
    controls=[dict(overall=dict(sampled_regret=-.2,start_disagreement=.1,basin_disagreement=0.)),
              dict(overall=dict(sampled_regret=-.1,start_disagreement=.2,basin_disagreement=.1))]
    bounds=v2.limits(controls)
    assert bounds==dict(overall=dict(sampled_regret=-.1,start_disagreement=.2,basin_disagreement=.1))
    assert v2.exceeding(dict(overall=dict(sampled_regret=0.,start_disagreement=.2,basin_disagreement=.1)),bounds)==['overall/sampled_regret']


def test_strata_use_fixed_original_anchor_scores_and_basin_subset():
    data=dict(anchor_values=np.array([.2,.5,.6,.9]),anchor_F=np.ones(4),self_sums=np.full(4,10.),
        anchor_starts=np.zeros(4,int),basin_mask=np.array([False,True,False,True]))
    record=v2.metrics(data,np.array([1.,1.,1.,1.]),np.array([1,0,0,1]),np.array([False,True,False,False]))
    assert record['overall']['start_disagreement']==.5
    assert record['overall']['basin_disagreement']==.5
    assert record['T<0.4']['basin_disagreement'] is None
    assert record['T>=0.7']['start_disagreement']==1.


def test_basin_classifier_refuses_unresolved_cases_and_uses_fixed_neighborhood():
    cases=[dict(b=np.zeros((2,3)),transform=1)]
    anchor=np.zeros((1,2,3));candidate=np.ones((1,2,3))
    assert v2.basin_flags(cases,anchor,[True],candidate,[True]).tolist()==[True]
    with pytest.raises(ValueError,match='unresolved'):v2.basin_flags(cases,anchor,[True],candidate,[False])


def test_saved_state_refuses_model_edits_and_dependency_replacement(tmp_path,monkeypatch):
    # Saved-state identity is the subject here; keep revision deterministic so
    # this unit test also runs from exported source archives without .git.
    monkeypatch.setattr(v2.old,'revision',lambda:'test-revision')
    captured=v2.identity();args=Namespace(state=tmp_path,identity=captured)
    v2.save(args,'parent',dict(native_optimized_observed=False),dict(a=np.arange(3)))
    parent=tmp_path/'parent.json'
    v2.save(args,'child',dict(parent_sha256=v2.old.digest(parent)))
    v2.read(args,'child')
    parent.write_text('replaced')
    with pytest.raises(ValueError,match='dependency identity'):v2.read(args,'child')
    monkeypatch.setattr(v2,'identity',lambda:{'changed':'model'})
    with pytest.raises(ValueError,match='model/driver/fixture'):v2.read(args,'child')


def test_fresh_v2_seeds_and_geometry_disjoint_from_every_prior_field_model():
    from bench import numerics_field_quality as v1
    from bench import numerics_field_optimizer as order
    seeds=[set(x) for x in (v2.CALIBRATION,v2.HELDOUT,v1.CALIBRATION,v1.HELDOUT,
        order.CALIBRATION,order.HELDOUT,order.SAME_GEOMETRY_CALIBRATION,order.SAME_GEOMETRY_HELDOUT)]
    assert list(map(len,seeds))==[32,16]*4
    assert sum(map(len,seeds))==len(set.union(*seeds))
    assert len(v2.panel())==128


def test_nonfinite_actual_native_points_are_rejected_before_selection():
    class Extension:
        def numerics_sample_field(self,*args):return np.full((1,4),np.nan,np.float32)
    cases=[dict(a=np.zeros((1,3)),b=np.zeros((1,3)))]
    with pytest.raises(AssertionError,match='nonfinite production point'):
        v2.probe(cases,np.array([[[1.,0,0,0,0,0,0]]]),[(None,np.zeros(3),1.)],Extension())


def test_orphan_heldout_refuses_freeze_before_any_calibration_access(tmp_path):
    orphan=tmp_path/'heldout.npz';orphan.write_bytes(b'preserve')
    with pytest.raises(FileExistsError,match='preserving existing evidence'):
        v2.freeze(Namespace(state=tmp_path))
    assert orphan.read_bytes()==b'preserve' and not (tmp_path/'frozen.json').exists()


def test_orphan_optimized_output_refuses_calibration_before_native_context(tmp_path):
    orphan=tmp_path/'observed.json';orphan.write_text('preserve native output')
    with pytest.raises(FileExistsError,match='preserving existing evidence'):
        v2.calibration(Namespace(state=tmp_path))
    assert orphan.read_text()=='preserve native output'

"""Portable checks for independently derived FIELD quality ingredients."""
import numpy as np
import pytest
from roshambo3.reference.field_intervals import Interval, convex_filter_intervals
from roshambo3.reference.field_quality import (hull,lattice,matrix,native_points,
    overlap_interval,phi,staged,tensor8,INV_COORD_SCALE)
from roshambo3.numerics import COORD_SCALE, WIRE_QUANT_STEP


def test_interval_points_agree_with_existing_convex_model():
    rng=np.random.default_rng(275100)
    nodes=Interval(rng.normal(size=(5,6,7,4)))
    points=rng.uniform(-2,8,(40,3)).astype(np.float32)
    old=convex_filter_intervals(nodes,points)
    new=hull(nodes,Interval(points),np.zeros(3,np.float32),np.float32(1))
    np.testing.assert_array_equal(new.lo,old.lo)
    np.testing.assert_array_equal(new.hi,old.hi)


def test_wa4_encloses_all_positive_sample_sums():
    field=Interval(np.full((4,4,4,4),.25),np.full((4,4,4,4),.5))
    points=Interval(np.full((11,3),1.2))
    bound=overlap_interval(field,points,np.zeros(3),np.float32(1))
    assert bound.lo<=11*.25 and bound.hi>=11*.5


def test_returned_translation_preimage_encloses_actual_reduced_values():
    rng=np.random.default_rng(275101)
    cloud=np.zeros((3,3))
    for _ in range(30):
        reduced=rng.uniform(-8,8,3).astype(np.float32)
        returned=INV_COORD_SCALE*reduced
        points=native_points(cloud,[1,0,0,0],returned,returned_translation=True)
        assert np.all(points.lo<=reduced) and np.all(points.hi>=reduced)


def test_native_matrix_retains_nonunit_quaternion_and_staging():
    q=np.array([1.001,.001,.002,.003],np.float32)
    cloud=np.array([[.0006,.0014,-.0016]],np.float32)
    points=native_points(cloud,q,[0,0,0])
    # Normalizing would silently change native sample locations.
    exact=np.array([[1-2*q[2]**2-2*q[3]**2,2*q[1]*q[2]-2*q[0]*q[3],2*q[1]*q[3]+2*q[0]*q[2]],
        [2*q[1]*q[2]+2*q[0]*q[3],1-2*q[1]**2-2*q[3]**2,2*q[2]*q[3]-2*q[0]*q[1]],
        [2*q[1]*q[3]-2*q[0]*q[2],2*q[2]*q[3]+2*q[0]*q[1],1-2*q[1]**2-2*q[2]**2]])
    value=staged(cloud)@exact.T*float(COORD_SCALE)
    assert np.all(points.lo<=value) and np.all(points.hi>=value)
    assert np.max(abs(exact-matrix(q)))>1e-7


def test_tensor_endpoint_sampler_stays_in_declared_hull():
    rng=np.random.default_rng(275102)
    field=rng.normal(size=(4,5,6,4))
    points=rng.uniform(-1,6,(100,3)).astype(np.float32)
    bound=hull(Interval(field),Interval(points),np.zeros(3),np.float32(1))
    sampled=tensor8(field,points,np.zeros(3),1)
    assert np.all(sampled>=bound.lo) and np.all(sampled<=bound.hi)


def test_tanimoto_mapping_checks_pole_and_monotonicity():
    assert phi(2,10)<phi(3,10)
    with pytest.raises(ValueError,match='pole'):phi(10,10)


def test_lattice_and_staging_are_input_derived():
    query=np.array([[0,0,0,1],[2,1,3,1]],np.float32)
    origin,spacing,cells=lattice(query)
    assert origin.dtype==np.float32 and spacing.dtype==np.float32
    np.testing.assert_array_equal(cells,[23,22,24])
    np.testing.assert_array_equal(staged([[.6*WIRE_QUANT_STEP,-.6*WIRE_QUANT_STEP,0]]),
                                  np.array([[WIRE_QUANT_STEP,-WIRE_QUANT_STEP,0]],np.float32))


def test_native_fixed_controls_select_actual_int16_wire(monkeypatch):
    from bench.numerics_field_quality import native
    from bench.numerics_optimizer_real import pack
    from bench.numerics_field_quality import panel
    from roshambo3.backend import cuda
    inputs=pack(panel()[:1],'shape')
    inputs['library_coords'][0,0,0]=.6*WIRE_QUANT_STEP
    def score(**kwargs):
        assert kwargs['library_coords'].dtype==np.int16
        assert kwargs['steps']==0 and kwargs['start_mode']==0
        assert kwargs['field_spacing_a']==.8 and kwargs['field_pad_a']==8.
        assert 1 in kwargs['library_coords'].reshape(-1)
        kwargs['winning_starts'][:]=0;kwargs['start_objectives'][:]=1
        rows=np.zeros((1,20),np.float32);rows[:,9]=1;rows[:,16]=1
        return rows
    monkeypatch.setattr(cuda,'score_pairs',score)
    native(inputs,fixed=True)


def test_native_rejects_nonidentity_query_frame(monkeypatch):
    from bench.numerics_field_quality import native,panel
    from bench.numerics_optimizer_real import pack
    from roshambo3.backend import cuda
    def score(**kwargs):
        kwargs['winning_starts'][:]=0;kwargs['start_objectives'][:]=1
        return np.zeros((1,20),np.float32)
    monkeypatch.setattr(cuda,'score_pairs',score)
    with pytest.raises(AssertionError,match='nonidentity query frame'):
        native(pack(panel()[:1],'shape'),fixed=True)


def test_existing_receipts_are_preserved_including_orphan_frozen(tmp_path):
    from bench.numerics_field_quality import fresh
    frozen=tmp_path/'receipt.json.frozen.json'
    frozen.write_text('preregistered')
    with pytest.raises(FileExistsError,match='preserving existing evidence'):
        fresh(tmp_path/'receipt.json',tmp_path/'observations.npz',frozen)
    assert frozen.read_text()=='preregistered'


def test_preregistered_controls_are_disjoint_from_previous_experiments():
    from bench.numerics_field_quality import CALIBRATION,HELDOUT
    from bench.numerics_field_optimizer import CALIBRATION as OLD_CAL,HELDOUT as OLD_HELD,SAME_GEOMETRY_CALIBRATION,SAME_GEOMETRY_HELDOUT
    sets=list(map(set,(CALIBRATION,HELDOUT,OLD_CAL,OLD_HELD,SAME_GEOMETRY_CALIBRATION,SAME_GEOMETRY_HELDOUT)))
    assert list(map(len,sets))==[32,16,32,16,32,16]
    assert sum(map(len,sets))==len(set.union(*sets))


def test_frozen_model_identity_refuses_current_source_changes(monkeypatch):
    from bench import numerics_field_quality as q
    record=dict(source_identity=q.source_identity(),protocol_sha256=q.digest(q.PROTOCOL))
    q.assert_identity(record)
    changed=dict(record['source_identity']);changed['src/roshambo3/reference/field_quality.py']='changed'
    monkeypatch.setattr(q,'source_identity',lambda:changed)
    with pytest.raises(ValueError,match='model/driver/fixture'):q.assert_identity(record)


def test_midrun_identity_guard_refuses_relabeling(monkeypatch):
    from bench import numerics_field_quality as q
    captured=q.source_identity()
    q.unchanged(captured)
    monkeypatch.setattr(q,'source_identity',lambda:{'edited':'during execution'})
    with pytest.raises(RuntimeError,match='source changed during'):q.unchanged(captured)


def test_native_learning_rate_plant_uses_exact_declared_increment(monkeypatch):
    from bench.numerics_field_quality import native,panel
    from bench.numerics_optimizer_real import pack
    from roshambo3.backend import cuda
    def score(**kwargs):
        assert kwargs['lr_q']==.11 and kwargs['lr_t']==.4
        assert kwargs['steps']==25 and kwargs['start_mode']==1
        kwargs['winning_starts'][:]=0;kwargs['start_objectives'][:]=1
        rows=np.zeros((1,20),np.float32);rows[:,16]=1
        return rows
    monkeypatch.setattr(cuda,'score_pairs',score)
    native(pack(panel()[:1],'shape'),lr_q=.11)

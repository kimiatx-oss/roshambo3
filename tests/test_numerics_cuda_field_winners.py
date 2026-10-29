"""FIELD IDs reflect sampled search selection, separate from analytic final rescore."""
import numpy as np
import pytest
from bench.numerics_cuda_field_winners import snapshot,gpu_inputs
from bench.numerics_backends import seeded_panel


@pytest.fixture(scope='module')
def live_cuda():
    from roshambo3.backend import cuda
    if not cuda.available():pytest.skip('CUDA extension unavailable')
    from bench.numerics_backends import verified_build
    verified_build('cuda')
    if cuda.extension().device_count()==0:pytest.skip('CUDA device unavailable')
    return cuda


def test_actual_native_integer_ids_match_native_selection(live_cuda):
    _,observations=snapshot(metadata=True)
    assert all(item['ids_match_native_objective_argmax'] for item in observations)


@pytest.mark.parametrize('name,bad',[('winning_starts',np.zeros(32,np.int64)),
    ('winning_starts',np.zeros(64,np.int32)[::2]),('winning_starts',np.zeros((32,1),np.int32)),
    ('start_objectives',np.zeros((32,4),np.float64)),('start_objectives',np.zeros((32,3),np.float32)),
    ('start_objectives',np.zeros((32,8),np.float32)[:,::2])])
def test_metadata_refuses_wrong_dtype_shape_and_stride(live_cuda,name,bad):
    from roshambo3.backend.spec import PoseSearch
    with pytest.raises(ValueError,match=name):
        live_cuda.score_pairs(**gpu_inputs(seeded_panel()),pose_search=PoseSearch.FIELD,**{name:bad})


@pytest.mark.parametrize('name,shape,dtype',[('winning_starts',(32,),np.int32),('start_objectives',(32,4),np.float32)])
def test_metadata_refuses_readonly_outputs(live_cuda,name,shape,dtype):
    from roshambo3.backend.spec import PoseSearch
    array=np.zeros(shape,dtype);array.flags.writeable=False
    with pytest.raises(ValueError,match=name):
        live_cuda.score_pairs(**gpu_inputs(seeded_panel()),pose_search=PoseSearch.FIELD,**{name:array})


def test_field_metadata_refuses_unsupported_dispatch(live_cuda):
    from roshambo3.backend.spec import PoseSearch, CoordResidency
    before=live_cuda.extension().arena_stats()
    with pytest.raises(ValueError,match='supported default dispatch cell'):
        live_cuda.score_pairs(**gpu_inputs(seeded_panel()),pose_search=PoseSearch.FIELD,
            coord_residency=CoordResidency.GLOBAL,winning_starts=np.empty(32,np.int32))
    assert live_cuda.extension().arena_stats()==before


def test_native_validation_cannot_be_bypassed_by_calling_extension(live_cuda,monkeypatch):
    from roshambo3.backend.spec import PoseSearch
    extension=live_cuda.extension();calls=[]
    class Capture:
        def __getattr__(self,name):return getattr(extension,name)
        def optimize_pairs(self,*args,**kwargs):
            calls.append((args,kwargs));return extension.optimize_pairs(*args,**kwargs)
    monkeypatch.setattr(live_cuda,'extension',lambda:Capture())
    live_cuda.score_pairs(**gpu_inputs(seeded_panel()),pose_search=PoseSearch.FIELD)
    args,kwargs=calls[-1]
    before=extension.arena_stats()
    with pytest.raises(ValueError,match='supported default dispatch cell'):
        extension.optimize_pairs(*args,**{**kwargs,'coord_residency':1,
            'winning_starts':np.empty(32,np.int32)})
    assert extension.arena_stats()==before
    for name,bad in [('winning_starts',np.zeros(32,np.int64)),
                     ('winning_starts',np.zeros(64,np.int32)[::2]),
                     ('start_objectives',np.zeros((32,3),np.float32))]:
        with pytest.raises(ValueError,match=name):
            extension.optimize_pairs(*args,**{**kwargs,name:bad})
    objectives=np.empty((32,4),np.float32)
    ids=objectives.reshape(-1).view(np.int32)[:32]
    with pytest.raises(ValueError,match='overlap'):
        extension.optimize_pairs(*args,**{**kwargs,'winning_starts':ids,'start_objectives':objectives})
    with pytest.raises(ValueError,match='dense call'):
        extension.optimize_pairs(*args,**{**kwargs,'topk':1,'winning_starts':np.empty(32,np.int32)})


def test_native_objectives_can_be_requested_without_ids(live_cuda):
    from roshambo3.backend.spec import PoseSearch
    inputs=gpu_inputs(seeded_panel())
    objectives=np.full((32,4),np.nan,np.float32)
    rows=live_cuda.score_pairs(**inputs,pose_search=PoseSearch.FIELD,start_objectives=objectives)
    assert np.isfinite(objectives).all()
    # Objectives are raw sampled overlaps, while score columns hold the final
    # analytic rescore. They have separate semantics; no equality is required.
    assert np.isfinite(rows).all()


def test_metadata_rejects_overlapping_outputs(live_cuda):
    from roshambo3.backend.spec import PoseSearch
    inputs=gpu_inputs(seeded_panel())
    objectives=np.empty((32,4),np.float32)
    ids=objectives.reshape(-1).view(np.int32)[:32]
    with pytest.raises(ValueError,match='overlap'):
        live_cuda.score_pairs(**inputs,pose_search=PoseSearch.FIELD,
            winning_starts=ids,start_objectives=objectives)
    rows=np.empty((32,20),np.float32)
    with pytest.raises(ValueError,match='overlap'):
        live_cuda.score_pairs(**inputs,pose_search=PoseSearch.FIELD,out=rows,
            winning_starts=rows.reshape(-1).view(np.int32)[:32])


def test_metadata_refuses_topk_before_accessing_a_device_handle(live_cuda):
    from roshambo3.backend.spec import PoseSearch
    with pytest.raises(ValueError,match='dense call'):
        live_cuda.score_pairs(**gpu_inputs(seeded_panel()),pose_search=PoseSearch.FIELD,
            topk=object(),winning_starts=np.empty(32,np.int32))

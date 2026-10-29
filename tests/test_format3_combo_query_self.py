"""Format-3 query reuse, exact input/value binding and mutation guards."""
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import pytest
from roshambo3.backend import cpu
from roshambo3.engine import R3gSource
from roshambo3.engine.r3g_cpu import R3gWorkerSelfCache
from tests.test_self_overlap_routes import build_inputs, _inputs

@pytest.fixture
def binding(tmp_path,monkeypatch):
    monkeypatch.setenv('ROSHAMBO3_SCAN_CACHE',str(tmp_path/'scan-cache'))
    built = build_inputs(tmp_path / 'corpus')
    from roshambo3.format.r3g.reader import verify, open_corpus
    verify(built[0].root, tier='scan')
    built = (open_corpus(built[0].root), built[1], built[2])
    source, query, _ = _inputs(*built, 'combo')
    arrays = tuple(np.array(a, dtype=t, order='C', copy=True) for a,t in zip(
        (query.coords,query.types,query.n_real,query.interaction_r,query.interaction_p),
        (np.float32,np.int32,np.int32,np.float32,np.float32)))
    cache = R3gWorkerSelfCache(source, 'a'*64)
    yield source, cache, arrays
    source.close()

def bind(source, cache, arrays, descriptor):
    decoded=source.decode_cpu_block(descriptor)
    handle, reason, hit, replaced=cache.block(cpu,descriptor,decoded,decoded.result.coords,*arrays)
    assert reason is None and not hit and not replaced
    return decoded,handle

def score(decoded, handle, arrays, **replace):
    qc,qt,qn,r,p=arrays
    kwargs=dict(query_coords=qc,query_types=qt,query_n_atoms=qn,
        library_coords=decoded.result.coords,library_counts=decoded.result.counts,
        pair_query_row=np.zeros(decoded.descriptor.n_conf,np.int32),
        pair_library_row=np.arange(decoded.descriptor.n_conf,dtype=np.int32),
        interaction_r=r,interaction_p=p,color=True,color_mode='combo_optimal',
        pose_contract='shared_fp32_reduced_v1',color_grain=16,mixing=.5,
        steps=1,lr_q=0.,lr_t=0.,budget_override='T295 fixed-pose binding regression',
        prepared_self_overlaps=handle)
    kwargs.update(replace)
    return cpu.score_pairs(**kwargs)

def test_query_preparation_once_across_concurrent_blocks(binding,monkeypatch):
    source,cache,arrays=binding
    descriptors=source.cpu_block_descriptors()
    original=cpu.extension().prepare_combo_self_overlaps
    calls=[]
    def observed(qc,qt,qn,lc,*args,**kwargs):
        calls.append((qc.shape[0],lc.shape[0]))
        return original(qc,qt,qn,lc,*args,**kwargs)
    monkeypatch.setattr(cpu.extension(),'prepare_combo_self_overlaps',observed)
    original_library=cpu.extension()._prepare_r3g_library_self_overlaps
    def observed_library(lc,*args,**kwargs):
        calls.append((0,lc.shape[0]))
        return original_library(lc,*args,**kwargs)
    monkeypatch.setattr(cpu.extension(),'_prepare_r3g_library_self_overlaps',observed_library)
    gate=threading.Barrier(2,timeout=5)
    def claim(descriptor):
        if descriptor in descriptors[:2]:gate.wait()
        decoded,handle=bind(source,cache,arrays,descriptor)
        np.testing.assert_array_equal(handle.library.color_values,decoded.result.color_self)
        assert handle.library.shape_terms==handle.library.color_terms==0
        cache.release_block(descriptor)
    with ThreadPoolExecutor(2) as workers:list(workers.map(claim,descriptors))
    assert sum(q>0 for q,l in calls)==1
    assert sum(l>0 for q,l in calls)==len(descriptors)
    assert cache.query_preparations==1
    assert cache.query_shape_self_evaluations==cache.query_color_self_evaluations==len(arrays[2])
    assert not cache.entries and cache.retained_bytes==0

@pytest.mark.parametrize('field',[0,1,2,3,4])
def test_query_snapshot_mutation_refused(binding,field):
    source,cache,arrays=binding
    descriptors=source.cpu_block_descriptors()
    bind(source,cache,arrays,descriptors[0]);cache.release_block(descriptors[0])
    arrays[field].flat[0]+=1
    with pytest.raises(ValueError,match='mutated|interaction matrix'):
        bind(source,cache,arrays,descriptors[1])

def test_composed_handle_exact_scores_and_public_mutation_guards(binding):
    source,cache,arrays=binding
    descriptor=source.cpu_block_descriptors()[0]
    decoded,handle=bind(source,cache,arrays,descriptor)
    qc,qt,qn,r,p=arrays
    reference=cpu.prepare_combo_self_overlaps(query_coords=qc,query_types=qt,query_n_atoms=qn,
        library_coords=decoded.result.coords,library_counts=decoded.result.counts,
        interaction_r=r,interaction_p=p,prepared_library=handle.library)
    np.testing.assert_array_equal(score(decoded,handle,arrays).view(np.uint32),
                                  score(decoded,reference,arrays).view(np.uint32))
    changed=decoded.result.coords.copy();changed.flat[0]+=.125
    with pytest.raises(ValueError,match='stale'):
        score(decoded,handle,arrays,library_coords=changed)
    altered=qc.copy();altered.flat[0]+=.125
    with pytest.raises(ValueError,match='stale'):
        score(decoded,handle,arrays,query_coords=altered)
    handle.library.color_values.flags.writeable=True
    handle.library.color_values[0]=np.nextafter(handle.library.color_values[0],np.inf)
    with pytest.raises(ValueError,match='values were mutated'):
        score(decoded,handle,arrays)
    cache.release_block(descriptor)

def test_native_combine_rejects_library_as_query_snapshot(binding):
    source,cache,arrays=binding
    decoded,handle=bind(source,cache,arrays,source.cpu_block_descriptors()[0])
    with pytest.raises(ValueError,match='incompatible'):
        cpu.extension()._combine_r3g_self_overlaps(handle.library._native,handle.library._native,'b'*64)

def test_engine_counts_query_snapshot_once_and_reuses_it(tmp_path,monkeypatch):
    from roshambo3.engine import Engine
    from tests.test_self_overlap_routes import _run
    monkeypatch.setenv('ROSHAMBO3_SCAN_CACHE',str(tmp_path/'scan-cache'))
    built=build_inputs(tmp_path/'corpus')
    from roshambo3.format.r3g.reader import verify,open_corpus
    verify(built[0].root,tier='scan')
    built=(open_corpus(built[0].root),built[1],built[2])
    inputs=_inputs(*built,'combo')
    engine=Engine(backend=cpu)
    try:
        first,_=_run(built,'combo',engine=engine,inputs=inputs)
        second,_=_run(built,'combo',engine=engine,inputs=inputs)
        assert first.cpu_scan['format3_query_self_contexts_built']==1
        assert first.cpu_scan['shape_query_self_evaluations']==inputs[1].n_conformers
        assert first.cpu_scan['color_query_self_evaluations']==inputs[1].n_conformers
        assert second.cpu_scan['format3_query_self_contexts_built']==0
        assert second.cpu_scan['shape_query_self_evaluations']==0
        assert second.cpu_scan['color_query_self_evaluations']==0
        assert first.cpu_scan['format3_query_self_retained_bytes']>0
        assert first.cpu_scan['prepared_self_retained_bytes']==second.cpu_scan['prepared_self_retained_bytes']==0
    finally:
        engine.close();inputs[0].close()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8, np.int32])
@pytest.mark.parametrize("strided", [False, True])
def test_batched_fingerprints_match_public_stream_exactly(dtype, strided):
    from roshambo3.engine.r3g_cpu import _batched_fingerprint_arrays
    array = np.arange(72, dtype=dtype).reshape(3, 8, 3)
    if strided:
        array = array[:, ::2, :]
    values = np.array([0., -0., 1., np.nextafter(1., 2.)], np.float64)
    arrays = (("coords", array), ("self", values))
    expected = cpu._fingerprint_arrays("T295.exact.public.stream", arrays)
    assert _batched_fingerprint_arrays("T295.exact.public.stream", arrays) == expected
    values[1] = 0.
    changed = _batched_fingerprint_arrays("T295.exact.public.stream", arrays)
    assert changed != expected
    assert changed == cpu._fingerprint_arrays("T295.exact.public.stream", arrays)

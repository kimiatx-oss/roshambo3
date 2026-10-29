"""Fused coordinate builders feed the existing CPU/GPU consumers unchanged.

Synthetic raw blocks have no on-disk scale descriptor. GPU checks use the
current consumer wire step; provisional binary scales are CPU writer checks.
"""
import numpy as np
import pytest
from roshambo3.format.r3g import native as N, reference as R


@pytest.fixture(autouse=True)
def restore_builder():
    yield
    N._set_builder_mode('auto')
    N.force_isa('auto')


def _block(color, rows=33):
    rng = np.random.default_rng(20261019)
    confs = [min(rows, 17), max(rows-17, 0)]
    nr = np.repeat([5, 7], confs)
    xyz = rng.integers(-1800, 1801, size=(rows, 3, 8), dtype=np.int16)
    xyz[np.broadcast_to(np.arange(8)[None,None,:] >= nr[:,None,None], xyz.shape)] = 0
    use = 1 if confs[1] == 0 else 2
    runs = R.pack_runs(np.arange(use), confs[:use], [5,7][:use])
    features = None
    if color:
        features = N.FeatureSlice(np.array([0,2,4][:use+1], np.uint32),
            np.array([1,2,1,2],np.uint8), np.array([0,1,4,6,7],np.uint32),
            np.array([1,0,2,4,1,3,2],np.uint8))
    return R.encode_block(xyz, nr), runs, rows, 8, 16 if color else 8, 2 if color else 0, features


def _write(args, consumer, mode, scale, tile_mode=1):
    N._set_builder_mode(mode)
    if consumer == 'cpu': return N.write_cpu_block(*args, scale)
    n = N.gpu_layout_elements(args[2], args[4])[1]
    return N.write_gpu_block(*args, np.empty(n, np.int16), tile_mode=tile_mode)


def _query(library, counts, color):
    L = library.shape[1]
    q = np.zeros((1,L,4),np.float32)
    occupied = int(counts[0].sum())
    q[0,:occupied,:3] = library[0,:occupied]
    q[0,:occupied,3] = 1.
    qt = np.zeros((1,L),np.int32)
    if color:
        nr = int(counts[0,0]); qt[0,nr:nr+2] = [1,2]
    ntypes = counts.shape[1]
    r = np.ones((ntypes,ntypes),np.float32)
    p = np.zeros_like(r)
    if color: p[1,1] = p[2,2] = 1.
    return q,qt,np.array([counts[0,0]],np.int32),r,p


@pytest.mark.parametrize('isa', N.compiled_isas())
@pytest.mark.parametrize('color',[False,True])
@pytest.mark.parametrize('scale',[2.**-10,2.**-8,.001])
def test_cpu_builder_scores_match_reference(isa,color,scale):
    from roshambo3.backend import cpu
    N.force_isa(isa)
    args = _block(color)
    baseline = _write(args,'cpu','baseline',scale)
    fused = _write(args,'cpu','fused',scale)
    assert fused.coords.tobytes() == baseline.coords.tobytes()
    assert fused.counts.tobytes() == baseline.counts.tobytes()
    q,qt,qn,r,p = _query(baseline.coords,baseline.counts,color)
    kw = dict(query_coords=q,query_types=qt,query_n_atoms=qn,library_counts=baseline.counts,
              pair_query_row=np.zeros(args[2],np.int32),pair_library_row=np.arange(args[2],dtype=np.int32),
              interaction_r=r,interaction_p=p,steps=2,start_mode=1,threads=1)
    if color:
        shared=cpu.prepare_combo_shared_pose_queries(q,qt,qn,r,p,
            shape_backend='pipelined_field',shape_grain='library_atom',shape_width=8,
            field_spacing_angstrom=.8)
        kw.update(color=True,color_mode='combo_optimal',color_grain=16,mixing=.5,
                  shape_backend='pipelined_field',shape_grain='library_atom',shape_width=8,
                  shape_field_spacing_angstrom=.8,pose_contract='shared_fp32_reduced_v1',
                  prepared_shared_pose_queries=shared,factor_canonical_torque=True,
                  canonical_color_kernel=True,optimizer_exp_mode='rawbits_unclamped')
    old=cpu.score_pairs(library_coords=baseline.coords,**kw)
    new=cpu.score_pairs(library_coords=fused.coords,**kw)
    assert np.isfinite(old).all()
    assert new.tobytes() == old.tobytes()


@pytest.mark.parametrize('isa',N.compiled_isas())
@pytest.mark.parametrize('color',[False,True])
@pytest.mark.parametrize('tile_mode',[1,-1])
@pytest.mark.parametrize('rows',[7,33])
def test_gpu_builder_scores_match_reference(isa,color,tile_mode,rows):
    from roshambo3.backend import cuda
    from roshambo3.numerics import WIRE_QUANT_STEP
    from tests.gpu_device import device_present
    if not cuda.available() or not device_present(): pytest.skip('CUDA consumer unavailable')
    N.force_isa(isa)
    args = _block(color, rows)
    # Use exactly the current consumer's physical interpretation, not -10/-8
    # provisional writer experiments with a differently scaled kernel.
    scale=float(WIRE_QUANT_STEP)
    cpu_ref=_write(args,'cpu','baseline',scale)
    old=_write(args,'gpu','baseline',scale,tile_mode)
    new=_write(args,'gpu','fused',scale,tile_mode)
    assert old.coords.tobytes() == new.coords.tobytes()
    if tile_mode == -1:
        # No planar scoring cell is instantiated; the coordinate bytes are
        # the claim for the planar layout.
        return
    q,qt,qn,r,p=_query(cpu_ref.coords,cpu_ref.counts,color)
    kw=dict(query_coords=q,query_types=qt,query_n_atoms=qn,library_counts=old.counts,
            pair_query_row=np.zeros(rows,np.int32),pair_library_row=np.arange(rows,dtype=np.int32),
            interaction_r=r,interaction_p=p,color=color,mixing=.5 if color else 0.,
            steps=2,start_mode=1,pose_search=cuda.PoseSearch.ANALYTIC,
            library_color_offset=-1 if color else 0,library_shape_width=8,library_tile_mode=tile_mode)
    old_scores=cuda.score_pairs(library_coords=old.coords,**kw)
    new_scores=cuda.score_pairs(library_coords=new.coords,**kw)
    assert np.isfinite(old_scores).all()
    assert new_scores.tobytes() == old_scores.tobytes()

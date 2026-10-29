"""Direct FIELD search start-ID diagnostics; separate from analytic final rescore."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np


def scenarios():
    from bench.numerics_backends import seeded_panel
    from bench.numerics_optimizer_real import panel, pack
    from roshambo3.backend.spec import ColorPlacement
    small=seeded_panel(rows=32)
    real,_=panel()
    for kind,inputs in [('seeded',small),('real',None)]:
        for mode in ('shape','rescore','combo'):
            data=pack(real,'shape' if mode=='shape' else 'combo') if kind=='real' else inputs
            for starts in ((0,1,2) if kind=='seeded' else (1,)):
                settings=dict(start_mode=starts,color=mode!='shape',
                    mixing=0. if mode=='shape' else .5,
                    color_placement=ColorPlacement.IN_GRADIENT if mode=='combo' else ColorPlacement.AT_POSE,
                    library_color_offset=-1 if mode!='shape' else 0)
                yield kind+'-'+mode+'-'+str(starts),data,settings
                if mode=='combo':
                    yield kind+'-'+mode+'-'+str(starts)+'-staged',data,dict(settings,
                        color_residency=0,color_stage=48)
    # An exact tie, without geometry-dependent rounding in the start rotations.
    tie=seeded_panel(rows=1)
    tie['query_coords'][:,:,:3]=0;tie['library_coords'][:,:,:3]=0
    yield 'exact-tie',tie,dict(start_mode=1,color=False,mixing=0.,lr_q=0.,lr_t=0.)
    empty={k:v.copy() for k,v in small.items()}
    empty['pair_query_row']=empty['pair_query_row'][:0];empty['pair_library_row']=empty['pair_library_row'][:0]
    yield 'empty',empty,dict(start_mode=1,color=False,mixing=0.)


def gpu_inputs(inputs,integer=False):
    from roshambo3.layout import tile32
    data=dict(inputs)
    # FIELD requires canonical diagonal feature interactions. Keep the seeded
    # geometry/padding but declare the supported objective before either build.
    if 'interaction_r' in data and data['interaction_r'].shape == (3,3):
        data['interaction_r']=np.ones((3,3),np.float32)
        data['interaction_p']=np.diag(np.array([0,1,1],np.float32))
    library=inputs['library_coords']
    if integer:
        library=np.rint(library[:,:,:3] / WIRE_QUANT_STEP).astype(np.int16)
    data['library_coords']=tile32(library)
    if 'query_types' not in data:
        data.update(query_types=np.zeros(inputs['query_coords'].shape[:2],np.int32),
            interaction_r=np.ones((1,1),np.float32),interaction_p=np.zeros((1,1),np.float32))
    return data


def snapshot(*,metadata):
    from roshambo3.backend import cuda
    from roshambo3.backend.spec import PoseSearch
    results={};receipts=[]
    for name,inputs,settings in scenarios():
        for integer in (False,True):
            label=name+('-int16' if integer else '-float32')
            data=gpu_inputs(inputs,integer)
            count=len(data['pair_query_row']);starts=(1,4,10)[settings['start_mode']]
            kwargs=dict(pose_search=PoseSearch.FIELD,steps=25,lr_q=.1,lr_t=.4)
            kwargs.update(settings)
            kwargs['start_groups_per_row']=min(4,starts) if '-combo-' in name else 1
            clean=cuda.score_pairs(**data,**kwargs)
            results[label]=clean
            item=dict(name=label,n_pairs=count,n_starts=starts,
                start_groups_per_row=kwargs['start_groups_per_row'],
                color_residency=kwargs.get('color_residency',1),color_stage=kwargs.get('color_stage',0),
                search_selection='mixed_sampled_tanimoto' if '-combo-' in name else 'raw_sampled_overlap')
            if metadata:
                ids=np.full(count,-777,np.int32);objectives=np.full((count,starts),np.nan,np.float32)
                actual=cuda.score_pairs(**data,**kwargs,winning_starts=ids,start_objectives=objectives)
                np.testing.assert_array_equal(actual.view(np.uint32),clean.view(np.uint32),err_msg=label+' metadata changed score rows')
                assert np.isfinite(objectives).all(),label
                np.testing.assert_array_equal(ids,np.argmax(objectives,axis=1) if count else np.empty(0,np.int64),
                    err_msg=label+' ID differs from native selection criterion')
                assert np.all((ids>=0)&(ids<starts)),label
                if name=='exact-tie':
                    np.testing.assert_array_equal(objectives,objectives[:,:1]*np.ones((1,starts),np.float32))
                    assert ids.tolist()==[0]
                if name.startswith('real-'):
                    assert np.any(ids!=0),'real panel must exercise nonzero winning IDs'
                item.update(id_counts=np.bincount(ids,minlength=starts).tolist(),
                    ids_match_native_objective_argmax=True,metadata_score_bits_unchanged=True)
            receipts.append(item)
    return results,receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rows',type=Path,required=True)
    parser.add_argument('--metadata',action='store_true')
    parser.add_argument('--baseline-rows',type=Path)
    parser.add_argument('--pytest-report',type=Path,help='also execute native boundary tests under this same GPU lock')
    parser.add_argument('--source-commit',help='verify a historical baseline against this exact git source tree')
    args=parser.parse_args()
    import roshambo3
    from bench.run_numerics_checks import select_native_build
    select_native_build(roshambo3,args.build)
    from bench.gpu_lock import GpuLock,resolve_device,query_gpu_state
    device=resolve_device(0)
    with GpuLock('numerics-field-start-metadata',device) as lock:
        state=query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct>0:
            raise RuntimeError('GPU busy before diagnostic: '+str(state))
        from roshambo3.backend import cuda
        ext=cuda.extension()
        from roshambo3 import get_build_info
        if args.source_commit:
            subtree='src/roshambo3/backend/_cuda_src/'
            paths=subprocess.check_output(['git','ls-tree','-r','--name-only',args.source_commit,'--',subtree],text=True).splitlines()
            if not paths:raise RuntimeError('historical CUDA source tree missing')
            digest=hashlib.sha256()
            for path in sorted(paths):
                content=subprocess.check_output(['git','show',args.source_commit+':'+path])
                digest.update(path.removeprefix(subtree).encode()+b'\0')
                digest.update(hashlib.sha256(content).hexdigest().encode()+b'\0')
            if vars(get_build_info())['cuda_src_sha256']!=digest.hexdigest():
                raise RuntimeError('loaded baseline extension differs from declared git source tree')
        else:
            from bench.numerics_backends import verified_build
            verified_build('cuda')
        rows,observations=snapshot(metadata=args.metadata)
        if args.baseline_rows:
            with np.load(args.baseline_rows) as baseline:
                assert set(rows)==set(baseline.files)
                for name,array in rows.items():
                    np.testing.assert_array_equal(array.view(np.uint32),baseline[name].view(np.uint32),err_msg=name+' changed vs unmodified baseline')
        np.savez(args.rows,**rows)
        from roshambo3 import get_build_info
        report=dict(search_objective='raw FIELD overlap for shape/rescore; mixed FIELD Tanimoto for combo', final_score='analytic winner rescore', protocol='FIELD_WINNER_PROTOCOL.md',metadata=args.metadata,
            source_commit=subprocess.check_output(['git','rev-parse',args.source_commit or 'HEAD'],text=True).strip(),
            source_content_verified=True,
            build=vars(get_build_info()),extension=str(Path(ext.__file__).resolve()),
            binary_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            lock_held=lock.acquired,preflight=state.as_dict(),device=ext.device_info(0),
            baseline_score_bits_unchanged=bool(args.baseline_rows),observations=observations)
        if args.pytest_report:
            import pytest
            report['pytest_exitcode']=pytest.main(['--no-header','tests/test_numerics_cuda_field_winners.py',
                '--numerics-report-dir',str(args.pytest_report)])
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        if report.get('pytest_exitcode',0):return int(report['pytest_exitcode'])
    return 0

if __name__=='__main__':raise SystemExit(main())

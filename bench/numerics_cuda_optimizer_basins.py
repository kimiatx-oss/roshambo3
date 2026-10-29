"""Fresh CUDA own-pose basin diagnostics using the frozen fp64 polish protocol.

A short child snapshots CPU/CUDA poses under the GPU lock, then exits its CUDA
context before the parent performs the independent CPU-only endpoint polish.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def snapshot(args):
    import roshambo3
    from bench.run_numerics_checks import select_native_build
    select_native_build(roshambo3,args.cpu_build)
    select_native_build(roshambo3,args.cuda_build)
    from bench.gpu_lock import GpuLock,resolve_device,query_gpu_state
    device=resolve_device(0)
    with GpuLock('numerics-CUDA-independent-basins',device) as lock:
        state=query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct>0:
            raise RuntimeError('GPU busy before snapshot: '+str(state))
        from roshambo3.backend import cpu,cuda
        from roshambo3.backend.spec import PoseSearch,ColorPlacement
        from bench.numerics_backends import verified_build
        from bench.numerics_optimizer_real import panel,pack,SETTINGS
        from bench.numerics_cuda_winners import gpu_inputs
        cases,_=panel();arrays={}
        for objective in ('shape','combo'):
            inputs=pack(cases,objective)
            cpu_opts=dict(grain=-1,poly=False) if objective=='shape' else dict(
                color=True,color_mode='combo_optimal',mixing=.5,color_grain=0)
            arrays[objective+'_anchor']=cpu.score_pairs(**inputs,**SETTINGS,**cpu_opts)
            gpu_opts=dict(pose_search=PoseSearch.ANALYTIC,color=objective=='combo',mixing=.5,
                color_placement=ColorPlacement.IN_GRADIENT)
            if objective=='combo':gpu_opts['library_color_offset']=-1
            for name,lr_q in (('production',.1),('plant',.11)):
                ids=np.full(len(cases),-1,np.int32)
                arrays[objective+'_'+name]=cuda.score_pairs(**gpu_inputs(inputs),
                    **{**SETTINGS,'lr_q':lr_q},**gpu_opts,winning_starts=ids)
                if np.any((ids<0)|(ids>3)):raise RuntimeError('invalid native CUDA IDs')
                arrays[objective+'_'+name+'_starts']=ids
        metadata=dict(cpu=verified_build('cpu'),cuda=verified_build('cuda'),
            extensions={name:str(Path(backend.extension().__file__).resolve())
                for name,backend in (('cpu',cpu),('cuda',cuda))},
            lock_held=lock.acquired,preflight=state.as_dict(),settings=SETTINGS)
        arrays['provenance']=np.asarray(json.dumps(metadata))
        np.savez(args.scratch,**arrays)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-build',type=Path,required=True)
    parser.add_argument('--cpu-build',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        snapshot(args);return 0
    if args.scratch.exists():parser.error('scratch must be a fresh .npz path')
    if args.scratch.suffix!='.npz':parser.error('scratch must end in .npz')
    command=[sys.executable,'-S','-m','bench.numerics_cuda_optimizer_basins',
        '--cuda-build',str(args.cuda_build),'--cpu-build',str(args.cpu_build),
        '--scratch',str(args.scratch),'--out',str(args.out),'--worker']
    subprocess.run(command,check=True)
    print('CUDA snapshot complete; native worker exited before CPU polish',flush=True)
    from bench.numerics_optimizer_real import panel
    from bench.numerics_optimizer_basins import compare_basins
    cases,_=panel();mask=np.array([c['transform']%2==1 for c in cases])
    evidence=dict(protocol='OPTIMIZER_SUPPLEMENT_PROTOCOL.md',qualified=False,
        scope='independent empirical stationary-endpoint diagnostic; no topological or global proof',
        snapshot_sha256=hashlib.sha256(args.scratch.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        polish_sha256=hashlib.sha256(Path(__file__).with_name('numerics_optimizer_basins.py').read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),objectives={})
    with np.load(args.scratch) as rows:
        evidence['provenance']=json.loads(str(rows['provenance']))
        for objective in ('shape','combo'):
            evidence['objectives'][objective]={name:compare_basins(cases,
                rows[objective+'_anchor'],rows[objective+'_'+name],objective,mask=mask)
                for name in ('production','plant')}
            print(objective+' stationary-endpoint comparison complete',flush=True)
    args.out.write_text(json.dumps(evidence,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())

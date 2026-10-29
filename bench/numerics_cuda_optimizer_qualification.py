"""Run the full frozen optimizer protocol with direct native CUDA start IDs."""
from __future__ import annotations
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-build',type=Path,required=True)
    parser.add_argument('--cpu-build',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.scratch.mkdir(parents=True,exist_ok=True)
    import roshambo3
    from bench.run_numerics_checks import select_native_build
    select_native_build(roshambo3,args.cpu_build)
    select_native_build(roshambo3,args.cuda_build)
    from bench.gpu_lock import GpuLock,resolve_device,query_gpu_state
    device=resolve_device(0)
    with GpuLock('numerics-native-start-optimizer',device) as lock:
        state=query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct>0:
            raise RuntimeError('GPU busy before qualification: '+str(state))
        import pybind11
        from roshambo3.backend import cpu
        from bench.numerics_backends import verified_build
        verified_build('cpu');verified_build('cuda')
        root=Path(__file__).resolve().parents[1]
        source=root/'bench/optimizer_control.cpp'
        output=args.scratch/('_numerics_optimizer_control'+sysconfig.get_config_var('EXT_SUFFIX'))
        flags=shlex.split(cpu.build_stamp()['cxx_flags'].split('[')[0])
        command=[os.environ.get('CXX','c++'),*flags,'-std=c++17',
            *shlex.split(sysconfig.get_config_var('LDSHARED'))[1:],'-fPIC',
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),str(source),'-o',str(output)]
        subprocess.run(command,check=True)
        spec=importlib.util.spec_from_file_location('_numerics_optimizer_control',output)
        probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
        probe.__compiler_command__=command;probe.__compiler_flags__=flags
        sys.modules[spec.name]=probe
        from bench.numerics_optimizer_real import run
        evidence=run(cuda=True)
        evidence['gpu_lock_held']=lock.acquired;evidence['gpu_preflight']=state.as_dict()
        from roshambo3.backend import cuda
        evidence['native_extensions']={name:str(Path(backend.extension().__file__).resolve())
            for name,backend in (('cpu',cpu),('cuda',cuda))}
        evidence['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        evidence['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        args.out.write_text(json.dumps(evidence,indent=2)+'\n')
        for objective,item in evidence['objectives'].items():
            assert item['heldout_valid'],objective+' held-out controls failed'
            assert not item['cpu']['exceeding'],objective+' CPU production failed'
            assert item['cpu']['plant_exceeding'],objective+' CPU plant survived'
            gpu=item['cuda']
            assert gpu['ran'],(objective,gpu)
            assert not gpu['exceeding'],(objective,gpu['production'])
            assert gpu['plant_exceeding'],objective+' CUDA plant survived'
            assert gpu['production']['overall']['winning_start_disagreement'] is not None
    return 0

if __name__=='__main__':raise SystemExit(main())

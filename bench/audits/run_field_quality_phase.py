"""Record real lock/preflight provenance around the frozen FIELD quality driver.

This adds observation only: it calls the original lock acquire/release and GPU
state query, and does not relax refusal or replace any numerical code.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from bench import gpu_lock
from bench import numerics_field_quality as q


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-report',type=Path,required=True)
    parser.add_argument('driver_args',nargs=argparse.REMAINDER)
    args=parser.parse_args();q.fresh(args.lock_report)
    records=[];locks=[]
    real_query=gpu_lock.query_gpu_state
    class ObservedLock(gpu_lock.GpuLock):
        def __enter__(self):
            item=dict(device=asdict(self.device),path=str(self.lock_path),
                cuda_extension_loaded_before_lock='roshambo3._roshambo3_cuda' in sys.modules)
            records.append(item);locks.append(self)
            result=super().__enter__();item['acquired']=self.acquired
            return result
        def __exit__(self,kind,error,tb):
            records[-1]['held_until_exit']=self.acquired
            records[-1]['was_held']=self.was_held
            return super().__exit__(kind,error,tb)
    def observed_query(device):
        state=real_query(device)
        records[-1]['preflight']=asdict(state)
        return state
    gpu_lock.GpuLock=ObservedLock;gpu_lock.query_gpu_state=observed_query
    original=sys.argv;status=0
    try:
        sys.argv=['numerics_field_quality',*args.driver_args]
        if args.driver_args and args.driver_args[0]=='v2':
            from bench.numerics_field_quality_v2 import main as v2_main
            sys.argv=['numerics_field_quality_v2',*args.driver_args[1:]]
            v2_main()
        else:q.main()
    except BaseException as error:
        status=1
        failure=repr(error)
        raise
    finally:
        sys.argv=original
        gpu_lock.GpuLock=ObservedLock.__bases__[0];gpu_lock.query_gpu_state=real_query
        q.write_json(args.lock_report,dict(source=q.revision(),wrapper_sha256=q.digest(__file__),
            driver_args=args.driver_args,exit_status=status,error=locals().get('failure'),
            locks=records,lock_base=os.environ.get('ROSHAMBO3_GPU_LOCK'),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            mps_pipe_directory=os.environ.get('CUDA_MPS_PIPE_DIRECTORY')))

if __name__=='__main__':main()

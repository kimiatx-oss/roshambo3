"""Native directed aggregate detector; run separately for control and mutant."""
import argparse
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-build',type=Path,required=True)
parser.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
parser.add_argument('--plant',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root)]
import roshambo3
from bench.run_numerics_checks import select_native_build
select_native_build(roshambo3, args.native_build)
from bench.numerics_bias import native_singletons,singleton_report
from bench.numerics_backends import verified_build
from bench.numerics_report import source_stamp
from contextlib import nullcontext
context = nullcontext()
if args.backend == 'cuda':
    from bench.gpu_lock import GpuLock, resolve_device, query_gpu_state
    device = resolve_device(0)
    context = GpuLock('numerics-bias', device)
with context as lock:
    hardware = {}
    if args.backend == 'cuda':
        state = query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct > 0:
            raise RuntimeError('GPU busy before aggregate control: '+str(state))
        hardware = dict(lock_held=lock.acquired, preflight=state.as_dict())
    report=singleton_report(native_singletons(args.backend),backend=args.backend)
    if args.backend == 'cuda':
        from roshambo3 import _roshambo3_cuda as ext
        hardware['device'] = ext.device_info(0)
    report.update(hardware=hardware)
report.update(build=verified_build(args.backend),source=source_stamp(root),plant=args.plant)
print(json.dumps(report,indent=2,allow_nan=False))
assert report['rows_pass'], 'plant exceeded existing row envelope'
assert report['mean_pass'] is not args.plant, 'mean gate did not distinguish control/plant'

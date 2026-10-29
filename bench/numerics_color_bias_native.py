"""Source-bound native typed-color aggregate receipt."""
import argparse
import ctypes
import ctypes.util
import json
import hashlib
import platform
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-build',type=Path,required=True)
parser.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
parser.add_argument('--molecular',action='store_true')
parser.add_argument('--plant-sign',type=int,choices=(-1,0,1),default=0)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root)]
import roshambo3
from bench.run_numerics_checks import select_native_build
select_native_build(roshambo3,args.native_build)
# Mathematical function controls, independent of production evaluations.
libm_name=ctypes.util.find_library('m')
libm=ctypes.CDLL(libm_name)
pow_function=libm.pow
pow_function.argtypes=[ctypes.c_double,ctypes.c_double]
pow_function.restype=ctypes.c_double
pow_controls={str(w):pow_function(w,1.5) for w in (1.,4.)}
assert pow_controls=={'1.0':1.,'4.0':8.}, 'libm is outside declared finite coefficient model'
from bench.numerics_backends import verified_build
from bench.numerics_report import source_stamp
from bench.numerics_color_bias import color_report,native_color
build=verified_build(args.backend)
if args.backend=='cpu':
    from roshambo3.backend import cpu
    binary=Path(cpu.extension().__file__).resolve()
else:
    from roshambo3 import _roshambo3_cuda as ext
    binary=Path(ext.__file__).resolve()
build['extension_path']=str(binary)
build['extension_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
detectors={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
    ('bench/numerics_color_bias.py','bench/numerics_color_bias_native.py',
     'bench/run_numerics_color_bias_plant.py','doc/numerics/COLOR_BIAS_PROTOCOL.md')}
if args.molecular:
    for name in ('bench/numerics_molecular_color_bias.py','bench/numerics_fixtures.py',
                 'bench/numerics_optimizer_real.py','src/roshambo3/reference/numerics_contracts.py',
                 'src/roshambo3/reference/overlap.py','doc/numerics/MOLECULAR_COLOR_BIAS_PROTOCOL.md'):
        detectors[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
libm_paths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if '/libm.so' in line and line.split()[-1].startswith('/')})
libm_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in libm_paths}
from contextlib import nullcontext
context=nullcontext()
if args.backend=='cuda':
    from bench.gpu_lock import GpuLock,resolve_device,query_gpu_state
    device=resolve_device(0)
    context=GpuLock('typed-color-bias',device)
with context as lock:
    hardware={}
    if args.backend=='cuda':
        state=query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct>0:
            raise RuntimeError('GPU busy before color control: '+str(state))
        hardware=dict(lock_held=lock.acquired,preflight=state.as_dict())
    if args.molecular:
        if args.backend!='cpu':raise ValueError('molecular model is CPU only')
        assert libm.fegetround()==0, 'nearest rounding required'
        assert platform.libc_ver()==('glibc','2.39'), 'current molecular expf authority is glibc2.39 only'
        from bench.numerics_molecular_color_bias import molecular_report,native_molecular_color
        values,census=native_molecular_color()
        report=molecular_report(values)
        report['issued_term_census']=census
        report['rounding_mode']=libm.fegetround()
    else:
        report=color_report(native_color(args.backend),backend=args.backend)
    if args.backend=='cuda':hardware['device']=ext.device_info(0)
report['hardware']=hardware
report.update(build=build,source=source_stamp(root),plant_sign=args.plant_sign,
    detector_sha256=detectors, libm=dict(library=libm_name,libc=platform.libc_ver(),pow_controls=pow_controls,loaded_files_sha256=libm_hashes))
print(json.dumps(report,indent=2,allow_nan=False))
assert report['rows_pass'], 'plant exceeded original row bounds'
assert report['mean_pass']==(args.plant_sign==0), 'aggregate did not distinguish plant'
assert all(s['mean_pass']==(args.plant_sign==0 or s['physical']==0) for s in report['strata']), 'type stratum distinction failed'

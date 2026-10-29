"""Preregistered general shape fixed-pose means from directed operation intervals."""
from __future__ import annotations
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import numpy as np
from roshambo3.reference.shape_intervals import overlap_interval,aggregate_report,prefactor_plant_model

ROOT=Path(__file__).resolve().parents[1]
PATHS={'cpu_scalar':('scalar','glibc_exp2',0,0,1),
       'cpu_query_atom':('query_atom','minimax_d5',1,1,8),
       'cpu_library_atom':('library_atom','minimax_d5',1,2,8),
       'cuda_direct':('scalar','cuda_exp2',None,None,None)}


def panel():
    from scipy.spatial.transform import Rotation
    from bench.numerics_fixtures import load_panel,fixture_binding
    rng=np.random.default_rng(276)
    geometries=[]
    for count in (5,9,17,33,65):
        geometries.append(dict(name='asymmetric_'+str(count),query=rng.normal(0,1.1,(count,3)).astype(np.float32),
            library=rng.normal(0,1.2,(count+2,3)).astype(np.float32),binding=dict(seed=276)))
    for fixture in load_panel():
        geometries.append(dict(name=fixture.name,query=fixture.shape.astype(np.float32),
            library=fixture.shape.astype(np.float32),binding=fixture_binding(fixture)))
    matrices=[np.eye(3),Rotation.from_rotvec([.31,-.22,.17]).as_matrix(),
              Rotation.from_rotvec([-.43,.19,.27]).as_matrix(),np.array([[1.1,.08,0],[0,.9,-.05],[.03,0,1.]])]
    translations=[np.zeros(3),np.array([.31,-.27,.19]),np.array([-1.1,.7,.4]),np.array([.2,-.4,.3])]
    result=[]
    for geometry in geometries:
        for index,(matrix,translation) in enumerate(zip(matrices,translations)):
            result.append(dict(geometry,pose=index,matrix=matrix,translation=translation))
    return result


def models(cases,path):
    grain,exponential,*_=PATHS[path]
    result=[]
    for case in cases:
        b=case['library'];matrix=case['matrix'];translation=case['translation']
        if path=='cuda_direct':
            # The production float-record volume seam accepts prepared point
            # clouds, not a pose matrix. This checks the resulting arbitrary
            # physical coordinates, not CUDA's transform-generation arithmetic.
            b=(b.astype(float)@matrix.T+translation).astype(np.float32)
            matrix=np.eye(3);translation=np.zeros(3)
        result.append(overlap_interval(case['query'],b,matrix,translation,grain=grain,exponential=exponential))
    return result


def preregistration(cases):
    declared={path:models(cases,path) for path in PATHS}
    chosen=None
    candidates=sorted((multiple*2.**exponent,exponent,multiple)
        for exponent in range(-22,-9) for multiple in (1.,1.25,1.5,1.75))
    for amplitude,exponent,multiple in candidates:
        good=True
        for rows in declared.values():
            clean_hi=math.fsum(row['upper']-row['physical'] for row in rows)/len(rows)
            clean_lo=math.fsum(row['lower']-row['physical'] for row in rows)/len(rows)
            for sign in (-1,1):
                plant=[prefactor_plant_model(row,1+sign*amplitude) for row in rows]
                plant_lo=math.fsum(row['lower']-row['physical'] for row in plant)/len(rows)
                plant_hi=math.fsum(row['upper']-row['physical'] for row in plant)/len(rows)
                within=all(row['lower']>=row['physical']-row['row_bound'] and row['upper']<=row['physical']+row['row_bound'] for row in plant)
                good &= within and (plant_lo>clean_hi if sign>0 else plant_hi<clean_lo)
        if good:
            chosen=dict(amplitude=amplitude,power=exponent,multiple=multiple,
                recipe='smallest quarter-dyadic amplitude multiple*2^power, multiple in {1,1.25,1.5,1.75}, power -22..-10 whose BOTH signed plant intervals stay inside every broad row bound and whose mean intervals exclude the clean mean interval')
            break
    digest=hashlib.sha256()
    for case in cases:
        digest.update(case['name'].encode()+b'\0')
        for key in ('query','library','matrix','translation'):digest.update(np.ascontiguousarray(case[key]).tobytes())
    return dict(panel_sha256=digest.hexdigest(),rows=len(cases),fixed_paired_constant_amplitude=1e-4,
        smaller_bias=chosen,paths=declared,full_qualified=False,
        gaps=['fixed-pose overlap component, not optimizer selection or trajectory','CUDA direct coordinate overlap does not qualify native affine transforms',
              'CPU libm allowance conditional on exact reviewed glibc image/code and nearest-rounding exp2f implementation','Other SIMD widths, trajectory grain, FIELD and arbitrary feature matrices are outside this shape-only panel'])


def libm_authority(libc,digest,offset,code_digest):
    profiles={
        (('glibc','2.39'),'fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0',514880,
         '3f5f3cc5a6e1a7ee90ef94bb53312b093df40b1a58f09d2ba2edad9f59947fd7'):
            'Ubuntu glibc2.39-0ubuntu8.9 x86_64 exp2f@GLIBC_2.27 FMA IFUNC target',
        (('glibc','2.41'),'a8afb2d1f4a0c5e14c0fc82f794ec839eafe29738907aa92895410f57630bbe2',270336,
         '2e454ff95c036e1e05ba348dbe82e8842f5f134a5361cd6daaa9fbfce52ee542'):
            'Debian glibc2.41-12+deb13u4 aarch64 exp2f@GLIBC_2.27 scalar table/FMA implementation',
    }
    key=(tuple(libc),digest,offset,code_digest)
    if key not in profiles:
        raise RuntimeError('exp2f DSO/resolved code not covered by the explicit scalar authority')
    return profiles[key]+'; published .502ULP nearest, conditional conservative two-ULP allowance'


def libm_image_sha256(path):
    """Hash the mapped libm image only when it exists as an ordinary file.

    Some systems expose libm through a shared cache with a synthetic loader
    path. That path cannot establish the explicit file-and-code authority this
    scalar check requires, so report the route as unqualified instead of
    guessing a digest.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        raise RuntimeError(
            f"resolved exp2f image is not a readable file for the explicit scalar authority: {path}"
        ) from None


def libm_identity():
    library=ctypes.CDLL(ctypes.util.find_library('m'))
    library.fegetround.restype=ctypes.c_int
    if library.fegetround()!=0:raise RuntimeError('nearest-even rounding required')
    class DlInfo(ctypes.Structure):
        _fields_=[('filename',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
    info=DlInfo();loader=ctypes.CDLL(None)
    loader.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)]
    pointer=ctypes.cast(library.exp2f,ctypes.c_void_p)
    if not loader.dladdr(pointer,ctypes.byref(info)):raise RuntimeError('cannot resolve actual exp2f implementation')
    path=Path(info.filename.decode()).resolve()
    digest=libm_image_sha256(path)
    offset=pointer.value-info.base
    code_digest=hashlib.sha256(ctypes.string_at(pointer,128)).hexdigest()
    authority=libm_authority(platform.libc_ver(),digest,offset,code_digest)
    return dict(libc=platform.libc_ver(),rounding='nearest_even',library=str(path),sha256=digest,
        resolved_symbol=info.symbol.decode() if info.symbol else None,resolved_offset=offset,
        resolved_code_sha256=code_digest,authority=authority)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--expect-constant-rejection',action='store_true')
    parser.add_argument('--expect-mean-only',action='store_true')
    parser.add_argument('--paths',nargs='+',choices=list(PATHS),default=list(PATHS))
    args=parser.parse_args()
    if args.out.exists():parser.error('output must be fresh')
    cases=panel();record=preregistration(cases)
    record.update(protocol_sha256=hashlib.sha256((ROOT/'doc/numerics/GENERIC_SHAPE_MEAN_PROTOCOL.md').read_bytes()).hexdigest(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
        model_sha256=hashlib.sha256((ROOT/'src/roshambo3/reference/shape_intervals.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        if not args.declare_only:
            if args.build is None:parser.error('--build required for native observations')
            import roshambo3
            from bench.run_numerics_checks import select_native_build
            from bench.numerics_backends import verified_build,cpu_evaluator_capability,cuda_shape_overlap
            select_native_build(roshambo3,args.build)
            from roshambo3.backend import cpu,cuda
            record['observations']={}
            for path in args.paths:
                values=[];metadata={}
                if path.startswith('cpu_'):
                    if not cpu.available():
                        record['observations'][path]=dict(status='not_run',reason='CPU extension unavailable');continue
                    grain,_,backend,code,width=PATHS[path]
                    capability=cpu_evaluator_capability(backend,code,width)
                    if not capability['available']:
                        record['observations'][path]=dict(status='not_run',capability=capability);continue
                    binary=Path(cpu.extension().__file__).resolve()
                    metadata=dict(build=verified_build('cpu'),capability=capability,
                        extension_path=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
                    if path=='cpu_scalar':
                        try:metadata['libm']=libm_identity()
                        except RuntimeError as error:
                            record['observations'][path]=dict(status='not_run',reason=str(error));continue
                    for case in cases:
                        result=cpu.extension().numerics_shape_evaluate(case['query'],case['library'],backend,code,width,.25,
                            physical_matrix=case['matrix'].astype(np.float32),physical_translation=case['translation'].astype(np.float32))
                        assert result['backend']==capability['stamp']['backend']
                        assert result['exponential']==capability['stamp']['exponential']
                        values.append(result['overlap_volume'])
                else:
                    if not cuda.available():
                        record['observations'][path]=dict(status='not_run',reason='CUDA extension unavailable');continue
                    from bench.gpu_lock import GpuLock,resolve_device,query_gpu_state
                    device=resolve_device(0)
                    with GpuLock('directed generic shape means',device) as lock:
                        state=query_gpu_state(device)
                        if state.compute_apps or state.utilization_gpu_pct>0:raise RuntimeError('GPU busy before shape means')
                        stamp=verified_build('cuda');binary=Path(cuda.extension().__file__).resolve()
                        metadata=dict(build=stamp,lock_held=lock.acquired,preflight=state.as_dict(),
                            extension_path=str(binary),extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
                        for case in cases:
                            b=(case['library'].astype(float)@case['matrix'].T+case['translation']).astype(np.float32)
                            values.append(cuda_shape_overlap(cuda.extension(),case['query'],b))
                report=aggregate_report(values,record['paths'][path])
                record['observations'][path]=dict(metadata,report=report,values=values)
                if args.expect_mean_only:
                    assert report['rows_pass'] and not report['mean_pass'],(path,report)
                elif args.expect_constant_rejection:
                    assert not(report['rows_pass'] and report['mean_pass']),(path,report)
                else:
                    assert report['rows_pass'] and report['mean_pass'],(path,report)
    except BaseException as error:
        record['execution_error']=dict(kind=type(error).__name__,message=str(error))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(record,indent=2)+'\n')
        raise
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    if record['smaller_bias'] is None:raise RuntimeError('no preregistered theoretical mean-only window; do not fit observed losses')


if __name__=='__main__':main()

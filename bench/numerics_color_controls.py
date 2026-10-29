"""Isolated test-only color SEARCH exponential controls; supplement protocol.

Builds two CPU extensions from copied source, never modifies production files.
The receipt is supplementary evidence and cannot upgrade v2 qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import numpy as np

from bench.numerics_optimizer_real import (
    CALIBRATION, HELDOUT, SETTINGS, panel, pack, scored, stats, freeze, exceeds,
    assert_zero_instrument, seed_intervals)
from bench.numerics_optimizer import METRICS

ROOT = Path(__file__).resolve().parents[1]
HOOK = r'''
#include <cmath>
#include <cstring>
#include <cstdint>
namespace color_control {
inline uint32_t seed = 0;
inline double amplitude = 0.;
inline double exp(double exponent) {
    const double value = std::exp(exponent);
    if (amplitude == 0.) return value;
    const float x = float(exponent);
    uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
    uint32_t h = bits ^ seed;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
    return value * (1. + ((h & 1u) ? amplitude : -amplitude));
}
}
'''


def instrument(source):
    """Replace exactly the scalar SEARCH Gaussian seam, never self/final sums."""
    begin = source.index('ColorEvaluation color_value_gradient_pose(')
    end = source.index('\n}', begin) + 2
    function = source[begin:end]
    if function.count('std::exp(') != 1:
        raise ValueError('scalar color SEARCH seam changed; review instrument')
    return source[:begin] + function.replace('std::exp(', 'color_control::exp(') + source[end:]


def build_variants(destination, jobs=4):
    """Source copies are small code only; no corpora or result fixtures copied."""
    import pybind11
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('build destination must not exist; reuse is not provenance')
    source = destination / 'source'
    source.mkdir(parents=True)
    for name in ('src', 'scripts'):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'CMakeLists.txt', source / 'CMakeLists.txt')
    commands = []
    source_manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob('*')) if p.is_file()}
    def build(name):
        directory = destination / name
        command = ['cmake', '-S', str(source), '-B', str(directory),
            '-DROSHAMBO3_CUDA=OFF', '-DCMAKE_BUILD_TYPE=Release',
            '-DPython_EXECUTABLE='+sys.executable, '-Dpybind11_DIR='+pybind11.get_cmake_dir()]
        commands.append(command); subprocess.run(command, check=True)
        command = ['cmake', '--build', str(directory), '--target', '_roshambo3_cpu', '-j', str(jobs)]
        commands.append(command); subprocess.run(command, check=True)
        return next(directory.glob('_roshambo3_cpu*'+sysconfig.get_config_var('EXT_SUFFIX')))
    baseline = build('baseline')
    native = source / 'src/roshambo3/backend/_cpu_src'
    scalar = native / 'color_scalar.cpp'
    original = scalar.read_text()
    (native / 'color_control.hpp').write_text(HOOK)
    scalar.write_text('#include "color_control.hpp"\n'+instrument(original))
    interface = native / 'interface.cpp'
    text = interface.read_text()
    marker = 'PYBIND11_MODULE(_roshambo3_cpu, m) {'
    if text.count(marker) != 1: raise ValueError('module binding changed')
    text = '#include "color_control.hpp"\n' + text.replace(marker, marker + '''
    m.def("_set_color_control", [](uint32_t seed, double amplitude) {
        color_control::seed = seed; color_control::amplitude = amplitude;
    });
''')
    interface.write_text(text)
    variant = build('instrument')
    metadata = dict(commands=commands, source_manifest=source_manifest,
        compiler_version=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
        cmake_version=subprocess.check_output(['cmake','--version'],text=True).splitlines()[0], original_scalar_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrument_scalar_sha256=hashlib.sha256(scalar.read_bytes()).hexdigest(),
        hook_sha256=hashlib.sha256(HOOK.encode()).hexdigest(),
        baseline_binary_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        instrument_binary_sha256=hashlib.sha256(variant.read_bytes()).hexdigest())
    (destination / 'build.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return baseline, variant, metadata


def worker(extension, output, instrumented):
    """Separate process prevents mixing native type registries or cached setup."""
    import roshambo3
    spec = importlib.util.spec_from_file_location('roshambo3._roshambo3_cpu', extension)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    roshambo3._roshambo3_cpu = module
    from roshambo3.backend import cpu
    cpu._ext = module
    cases, _ = panel()
    def native(seed=None, lr_q=.1, production=False):
        return cpu.score_pairs(**pack(cases,'combo',seed), **{**SETTINGS,'lr_q':lr_q},
            color=True, color_mode='combo_optimal', mixing=.5,
            color_grain=16 if production else 0, with_census=True)
    arrays = {}
    def save(key, value):
        rows, details = value
        starts = np.asarray(details['winning_starts'])
        if not np.issubdtype(starts.dtype,np.integer) or np.any((starts<0)|(starts>3)):
            raise ValueError('invalid native integer winning starts')
        arrays[key] = rows; arrays[key+'_starts'] = starts
    save('zero', native())
    if not instrumented:
        save('production',native(production=True)); save('plant',native(production=True,lr_q=.11))
    for seed in (*CALIBRATION,*HELDOUT):
        if instrumented:
            module._set_color_control(seed,2**-18)
            save(str(seed),native())
        else: save(str(seed),native(seed))
    arrays['_native_stamp_json']=np.asarray(json.dumps(cpu.build_stamp()))
    np.savez(output,**arrays)
    Path(str(output)+'.stamp.json').write_text(json.dumps(cpu.build_stamp(),indent=2)+'\n')


def evaluate(baseline, variant, metadata):
    cases, oracle = panel(); train=np.array([c['transform']%2==0 for c in cases]); test=~train
    anchor=baseline['zero']; astarts=baseline['zero_starts']
    anchor_values=scored(cases,anchor,'combo')
    assert_zero_instrument(anchor_values,scored(cases,variant['zero'],'combo'))
    np.testing.assert_array_equal(astarts,variant['zero_starts'])
    def compare(rows,starts,mask):
        return stats(cases,anchor,rows,'combo',mask,astarts,starts,anchor_scored=anchor_values)
    calibration=[]; heldout=[]
    for seeds,mask,out in ((CALIBRATION,train,calibration),(HELDOUT,test,heldout)):
        for seed in seeds:
            key=str(seed)
            records=[compare(z[key],z[key+'_starts'],mask) for z in (baseline,variant)]
            changed=int(np.sum(np.any(variant[key][:,9:16]!=variant['zero'][:,9:16],axis=1)))
            summary={k:max(abs(r['overall'][k]) if k=='mean_loss' else r['overall'][k] for r in records) for k in METRICS}
            out.append(dict(seed=seed,overall=dict(n=int(mask.sum()),**summary),
                reduction_order=records[0],color_exponential=records[1],
                color_instrument_pose_rows_changed=changed))
        if seeds is CALIBRATION: limits=freeze(calibration,oracle)
    if not any(r['color_instrument_pose_rows_changed'] for r in calibration):
        raise ValueError('nonzero color instrument changed no pose; control seam may be inactive')
    production=compare(baseline['production'],baseline['production_starts'],test)
    plant=compare(baseline['plant'],baseline['plant_starts'],test)
    heldout_valid=not any(exceeds(r,limits) for r in heldout)
    passing=heldout_valid and not exceeds(production,limits) and bool(exceeds(plant,limits))
    return dict(protocol='optimizer-color-supplement-v1',qualified=False,
        qualification_scope='finite color SEARCH exponential and reduction-order model only',
        gaps=['finite model has no proof of architecture coverage','CUDA metadata not exercised by this CPU-only supplement'],
        build=metadata,native_stamps={name:json.loads(str(z['_native_stamp_json']))
            for name,z in (('baseline',baseline),('instrument',variant))},zero_instrument=dict(exact_per_pair_match=True,exact_integer_winners=True),
        limits=limits,calibration=calibration,heldout=heldout,heldout_valid=heldout_valid,
        control_mean_intervals=seed_intervals(calibration),production=production,
        exceeding=exceeds(production,limits),plant=plant,plant_exceeding=exceeds(plant,limits),
        qualified_under_declared_model=passing)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--worker',type=Path);parser.add_argument('--instrumented',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.out,args.instrumented);return 0
    if args.build_dir is None:parser.error('--build-dir is required unless --worker is given')
    baseline,variant,metadata=build_variants(args.build_dir,args.jobs)
    for name,path in (('baseline',baseline),('instrument',variant)):
        command=[sys.executable,'-S','-m','bench.numerics_color_controls','--worker',str(path),
            '--out',str(args.build_dir/(name+'.npz'))]
        if name=='instrument':command.append('--instrumented')
        subprocess.run(command,check=True)
    with np.load(args.build_dir/'baseline.npz') as a, np.load(args.build_dir/'instrument.npz') as b:
        evidence=evaluate(a,b,metadata)
    args.out.write_text(json.dumps(evidence,indent=2)+'\n')
    return 0 if evidence['qualified_under_declared_model'] else 1

if __name__=='__main__':raise SystemExit(main())

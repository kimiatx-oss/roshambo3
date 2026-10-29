"""Execute both preregistered actual CPU/CUDA coefficient plants, restoring source."""
import argparse
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--disposable-worktree',type=Path,required=True)
p.add_argument('--molecular',action='store_true')
p.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=a.disposable_worktree.resolve()
if root==Path(__file__).resolve().parents[1] or not (root/'.git').is_file():
    p.error('separate disposable worktree required')
branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
if not branch.startswith('ticket/T270-fault-'):p.error('disposable ticket/T270-fault-* branch required')
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():
    p.error('clean tracked disposable worktree required')
build=root/'build/t270'
if 'CMAKE_HOME_DIRECTORY:INTERNAL='+str(root) not in (build/'CMakeCache.txt').read_text():
    p.error('build must belong to disposable worktree')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
seams=[('src/roshambo3/backend/_cpu_src/color_scalar.cpp','constexpr double COLOR_PREFACTOR = 1.9687012432153024;'),
       ('src/roshambo3/backend/_cpu_src/shared_pose_api.cpp','constexpr double kColorPrefactor = 1.9687012432153024;')]
if a.molecular:
    if a.backend!='cpu':p.error('molecular model is CPU only')
    seams.extend([('src/roshambo3/backend/_cpu_src/color_field.cpp','constexpr double kColorPrefactor = 1.9687012432153024;'),
                  ('src/roshambo3/backend/_cpu_src/color_default_parameters.hpp','static constexpr double physical_coefficient = 1.9687012432153024;')])
if a.backend=='cuda':
    seams=[('src/roshambo3/backend/_cuda_src/kernels.cuh','#define ROSHAMBO3_COLOR_PREFACTOR 1.9687012f')]
originals={relative:(root/relative).read_text() for relative,_ in seams}
for relative,needle in seams:
    if originals[relative].count(needle)!=1:raise RuntimeError('native coefficient seam changed')

def compile_native(name):
    with (out/(name+'-build.txt')).open('w') as log:
        subprocess.run(['cmake','--build',str(build),'--target','_roshambo3_'+a.backend,'-j','8'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)

def detect(name,sign=0):
    command=[sys.executable,'-S',str(root/'bench/numerics_color_bias_native.py'),'--native-build',str(build),'--plant-sign',str(sign),'--backend',a.backend]
    if a.molecular:command.append('--molecular')
    run=subprocess.run(command,cwd=root,text=True,capture_output=True,env=os.environ.copy())
    (out/(name+'.json')).write_text(run.stdout)
    (out/(name+'-stderr.txt')).write_text(run.stderr)
    if run.returncode:raise RuntimeError(name+' detector failed; see retained receipt')

try:
    detect('control')
    for name,sign in (('positive',1),('negative',-1)):
        patch=[]
        for relative,needle in seams:
            exponent=15 if a.molecular else 21
            replacement=(needle.replace('1.9687012432153024;',f'(1.9687012432153024 * (1.0 {"+" if sign>0 else "-"} 0x1p-{exponent}));')
                         if a.backend=='cpu' else needle.replace('1.9687012f',f'(1.9687012f * (1.0f {"+" if sign>0 else "-"} 0x1p-20f))'))
            mutant=originals[relative].replace(needle,replacement)
            patch.extend(difflib.unified_diff(originals[relative].splitlines(True),mutant.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
            (root/relative).write_text(mutant)
        (out/(name+'.patch')).write_text(''.join(patch))
        compile_native(name)
        detect(name,sign)
finally:
    for relative,text in originals.items():(root/relative).write_text(text)
    compile_native('restored')
    detect('restored')

# Enforce the frozen restoration/provenance requirements, independently of
# numerical assertions. A successful detector alone is not a restoration check.
receipts={name:json.loads((out/(name+'.json')).read_text())
          for name in ('control','positive','negative','restored')}
key=a.backend+'_src_sha256'
control,restored=receipts['control'],receipts['restored']
if control['build'][key]!=restored['build'][key]:
    raise RuntimeError('restored native source differs from clean control')
if control['build']['extension_sha256']!=restored['build']['extension_sha256']:
    raise RuntimeError('restored native binary differs from clean control')
if len({receipts[name]['build'][key] for name in ('control','positive','negative')})!=3:
    raise RuntimeError('native plants require three distinct source-content hashes')
if len({receipts[name]['build']['extension_sha256'] for name in ('control','positive','negative')})!=3:
    raise RuntimeError('native plants require three distinct binary hashes')
if any(receipt['detector_sha256']!=control['detector_sha256'] for receipt in receipts.values()):
    raise RuntimeError('detector/protocol source changed during native replay')

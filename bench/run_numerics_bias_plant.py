"""Reproduce the preregistered native analytic aggregate plant in a disposable tree.

Configure/build its build/t270 first; the detector and this script must already
be committed. Output must be a new directory. Production source is restored in
finally and a matching clean binary and control receipt are always rebuilt.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--disposable-worktree',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
a=p.parse_args()
root=a.disposable_worktree.resolve()
if root == Path(__file__).resolve().parents[1] or not (root/'.git').is_file():
    p.error('separate disposable worktree required')
branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
if not branch.startswith('ticket/T270-fault-'):
    p.error('disposable ticket/T270-fault-* branch required')
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():
    p.error('clean tracked disposable source required')
build=root/'build/t270'
if 'CMAKE_HOME_DIRECTORY:INTERNAL='+str(root) not in (build/'CMakeCache.txt').read_text():
    p.error('build must belong to disposable worktree')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
relative = ('src/roshambo3/backend/_cpu_src/kernel.hpp' if a.backend=='cpu'
            else 'src/roshambo3/backend/_cuda_src/kernels.cuh')
source=root/relative
original=source.read_text()
needle=('constexpr float SHAPE_PREFACTOR = 20.5795288f;' if a.backend=='cpu'
        else '#define ROSHAMBO3_SHAPE_PREFACTOR 20.579529f')
if original.count(needle)!=1:raise RuntimeError('source seam changed')
replacement=('constexpr float SHAPE_PREFACTOR = (20.5795288f * (1.0f + 0x1p-21f));'
             if a.backend=='cpu' else '#define ROSHAMBO3_SHAPE_PREFACTOR (20.579529f * (1.0f + 0x1p-21f))')
mutant=original.replace(needle,replacement)
(out/'prefactor.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),mutant.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative)))

def compile_native(name):
    with (out/(name+'-build.txt')).open('w') as log:
        subprocess.run(['cmake','--build',str(build),'--target','_roshambo3_'+a.backend,'-j','8'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)

def detect(name,plant=False):
    command=[sys.executable,'-S',str(root/'bench/numerics_bias_native.py'),'--native-build',str(build),'--backend',a.backend]
    if plant:command.append('--plant')
    run=subprocess.run(command,cwd=root,text=True,capture_output=True,env=os.environ.copy())
    (out/(name+'.json')).write_text(run.stdout)
    (out/(name+'-stderr.txt')).write_text(run.stderr)
    if run.returncode:raise RuntimeError(name+' detector failed; see receipt')

try:
    detect('control')
    source.write_text(mutant)
    compile_native('mutant')
    detect('mutant',True)
finally:
    source.write_text(original)
    compile_native('restored')
    detect('restored')

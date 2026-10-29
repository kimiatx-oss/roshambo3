"""Independent fp64 attraction-basin diagnostic for chosen optimizer poses.

Polish each physical placement with SciPy BFGS, then compare stationary endpoints.
This is an empirical local-search diagnostic, not topological certification.
"""
from __future__ import annotations
import numpy as np
from bench.numerics_fixtures import central_gradient, RECOVERY_NEIGHBORHOOD
from roshambo3.reference.overlap import (
    apply_pose, quaternion_matrix, pair_overlap, self_overlap, color_overlap)
from roshambo3.reference.color import guarded_tanimoto


def physical_clouds(case,row):
    inverse=np.asarray(row[16:20],float)*[1,-1,-1,-1]
    def move(cloud):
        return apply_pose(apply_pose(cloud,row[9:13],row[13:16]),inverse,np.zeros(3))
    return move(case['b']),move(case['bc']),quaternion_matrix(inverse)@row[13:16]


def polish(case,row,objective,*,maxiter=500):
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    if objective not in ('shape','combo'):raise ValueError('unknown objective')
    shape,color,initial_translation=physical_clouds(case,row)
    saa,sbb=self_overlap(case['a']),self_overlap(case['b'])
    if objective=='combo':
        caa=color_overlap(case['ac'],case['at'],case['ac'],case['at'],case['r'],case['p'])
        cbb=color_overlap(case['bc'],case['bt'],case['bc'],case['bt'],case['r'],case['p'])
    def value(pose):
        matrix=Rotation.from_rotvec(pose[:3]).as_matrix()
        moved=shape@matrix.T+pose[3:]
        ts=guarded_tanimoto(pair_overlap(case['a'],moved),saa,sbb)
        if objective=='shape':return ts
        moved_color=color@matrix.T+pose[3:]
        cross=color_overlap(case['ac'],case['at'],moved_color,case['bt'],case['r'],case['p'])
        return .5*(ts+guarded_tanimoto(cross,caa,cbb))
    fit=minimize(lambda p:-value(p),np.zeros(6),jac=lambda p:-central_gradient(value,p,h=1e-5),
        method='BFGS',options=dict(gtol=1e-8,maxiter=maxiter))
    gradient=float(np.max(np.abs(central_gradient(value,fit.x,h=1e-5))))
    matrix=Rotation.from_rotvec(fit.x[:3]).as_matrix()
    endpoint_shape=shape@matrix.T+fit.x[3:]
    endpoint_color=color@matrix.T+fit.x[3:]
    translation=matrix@initial_translation+fit.x[3:]
    stationary=bool(np.isfinite(fit.x).all() and np.isfinite(gradient) and gradient<=1e-6
                    and np.linalg.norm(translation)<=16.)
    return dict(stationary=stationary,gradient_inf=gradient,score=float(value(fit.x)),
        iterations=int(fit.nit),optimizer_success=bool(fit.success),message=str(fit.message),
        translation_norm=float(np.linalg.norm(translation)),
        incremental_pose=fit.x.tolist()),endpoint_shape,endpoint_color


def compare_basins(cases,anchor,candidate,objective,*,mask=None):
    if len(cases)!=len(anchor) or len(anchor)!=len(candidate):raise ValueError('row count mismatch')
    selected=np.ones(len(cases),bool) if mask is None else np.asarray(mask,bool)
    records=[]
    for index,(case,a,b) in enumerate(zip(cases,anchor,candidate)):
        if not selected[index]:continue
        aa,ash,ac=polish(case,a,objective);bb,bsh,bc=polish(case,b,objective)
        clouds_a=[ash] if objective=='shape' else [ash,ac]
        clouds_b=[bsh] if objective=='shape' else [bsh,bc]
        rms=float(np.sqrt(np.mean(np.sum((np.concatenate(clouds_a)-np.concatenate(clouds_b))**2,axis=1))))
        resolved=aa['stationary'] and bb['stationary']
        records.append(dict(index=index,anchor=aa,candidate=bb,endpoint_rms_angstrom=rms,
            same_attraction_basin=(bool(rms<=RECOVERY_NEIGHBORHOOD) if resolved else None)))
    unresolved=sum(r['same_attraction_basin'] is None for r in records)
    different=sum(r['same_attraction_basin'] is False for r in records)
    return dict(method='independent fp64 BFGS polish of own physical placements',
        scope='empirical attraction basin; global/topological identity unproved',
        endpoint_neighborhood_angstrom=RECOVERY_NEIGHBORHOOD,n=len(records),
        unresolved=unresolved,different=different,
        different_rate=(different/len(records) if records and not unresolved else None),records=records)


def main():
    import argparse
    import json
    import hashlib
    from pathlib import Path
    from bench.numerics_optimizer_real import panel
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--rows',type=Path,
        help='fresh baseline.npz produced by numerics_color_controls; combo only')
    source.add_argument('--extension',type=Path,help='isolated CPU extension for a fresh native snapshot')
    parser.add_argument('--objective',choices=('shape','combo'),default='combo')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cases,_=panel();mask=np.array([c['transform']%2==1 for c in cases])
    if args.rows:
        if args.objective!='combo':parser.error('color-control row archives contain combo poses only')
        with np.load(args.rows) as archive:
            rows={name:archive[name] for name in ('zero','production','plant')}
        provenance=dict(fresh_pose_archive_sha256=hashlib.sha256(args.rows.read_bytes()).hexdigest())
    else:
        import importlib.util
        import sys
        import roshambo3
        from bench.numerics_optimizer_real import pack, SETTINGS
        spec=importlib.util.spec_from_file_location('roshambo3._roshambo3_cpu',args.extension)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        sys.modules[spec.name]=module;roshambo3._roshambo3_cpu=module
        from roshambo3.backend import cpu
        cpu._ext=module
        inputs=pack(cases,args.objective)
        rows={}
        for name,production,lr_q in (('zero',False,.1),('production',True,.1),('plant',True,.11)):
            opts=dict(grain=2 if production else -1,poly=production) if args.objective=='shape' else dict(
                color=True,color_mode='combo_optimal',mixing=.5,color_grain=16 if production else 0)
            rows[name]=cpu.score_pairs(**inputs,**{**SETTINGS,'lr_q':lr_q},**opts)
        provenance=dict(native_build=cpu.build_stamp(),
            extension_binary_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest())
    evidence={name:compare_basins(cases,rows['zero'],rows[name],args.objective,mask=mask)
              for name in ('production','plant')}
    evidence['protocol']='optimizer-supplement-independent-basin-v1'
    evidence['qualified']=False
    evidence['objective']=args.objective
    evidence['provenance']=provenance
    evidence['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(evidence,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())

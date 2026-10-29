"""Real-molecule optimizer protocol v2 (explicitly conditional control model)."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np
from bench.numerics_fixtures import load_panel, oracle_uncertainty
from bench.numerics_optimizer import METRICS, BASIN_RESOLUTION
from roshambo3.reference.overlap import apply_pose, pair_overlap, self_overlap, tanimoto
from roshambo3.reference.color import score_columns
from roshambo3.metrics.floors import SCORE_STRATA, stratum_of
from roshambo3.optimizer import lr_t_for_steps

CALIBRATION = range(270300,270332)
HELDOUT = range(270400,270416)
SETTINGS = dict(steps=25,start_mode=1,lr_q=.1,lr_t=lr_t_for_steps(25))


def panel():
    from scipy.spatial.transform import Rotation
    fixtures = load_panel()
    rng=np.random.default_rng(270200)
    rotations=Rotation.random(8,random_state=rng).as_matrix()
    translations=rng.uniform(-1,1,(8,3))
    cases=[]
    def wire(x):
        return (np.rint(x / WIRE_QUANT_STEP).astype(np.float32)*WIRE_QUANT_STEP).astype(float)
    for transform,(rotation,translation) in enumerate(zip(rotations,translations)):
        for a in fixtures:
            for b in fixtures:
                cases.append(dict(a=wire(a.shape),b=wire(b.shape@rotation.T+translation),
                    ac=wire(a.color),bc=wire(b.color@rotation.T+translation),
                    at=a.types,bt=b.types, r=a.interaction_r,p=a.interaction_p,
                    transform=transform, query=a.name, library=b.name))
    uncertainty=2*max(oracle_uncertainty(f,'combo') for f in fixtures)
    return cases,uncertainty


def pack(cases, objective, seed=None):
    rng=np.random.default_rng(seed)
    width=max(max(len(c['a'])+len(c['ac']),len(c['b'])+len(c['bc'])) for c in cases)
    q=np.zeros((len(cases),width,4),np.float32); b=np.zeros_like(q)
    qt=np.zeros(q.shape[:2],np.int32)
    qn=np.array([len(c['a']) for c in cases],np.int32)
    counts=np.zeros((len(cases),7 if objective=='combo' else 1),np.uint8)
    for i,c in enumerate(cases):
        for prefix,out in (('a',q),('b',b)):
            order=rng.permutation(len(c[prefix])) if seed is not None else np.arange(len(c[prefix]))
            cloud=c[prefix][order]
            out[i,:len(cloud),:3]=cloud;out[i,:len(cloud),3]=1
            if objective=='combo':
                offset=len(cloud)
                orderc=np.arange(len(c[prefix+'c']))
                # Library color features are segmented by type; preserve segment layout.
                if seed is not None:
                    orderc=np.concatenate([rng.permutation(np.flatnonzero(c[prefix+'t']==t))
                                           for t in range(1,7)])
                out[i,offset:offset+len(orderc),:3]=c[prefix+'c'][orderc]
                out[i,offset:offset+len(orderc),3]=1
                if prefix=='a': qt[i,offset:offset+len(orderc)]=c['at'][orderc]
        counts[i,0]=len(c['b'])
        if objective=='combo':
            for t in range(1,7): counts[i,t]=np.sum(c['bt']==t)
    inputs=dict(query_coords=q,query_n_atoms=qn,library_coords=b,library_counts=counts,
                pair_query_row=np.arange(len(cases),dtype=np.int32),
                pair_library_row=np.arange(len(cases),dtype=np.int32))
    if objective=='combo': inputs.update(query_types=qt,interaction_r=cases[0]['r'].astype(np.float32),
                                        interaction_p=cases[0]['p'].astype(np.float32))
    return inputs


def scored(cases, rows, objective):
    values=[];posed=[]
    if not np.isfinite(rows).all(): raise ValueError('nonfinite rows')
    for c,row in zip(cases,rows):
        if np.linalg.norm(row[13:16])>16: raise ValueError('oracle extent exceeded')
        sq=row[16:20]*np.array([1,-1,-1,-1])
        def transform(cloud):return apply_pose(apply_pose(cloud,row[9:13],row[13:16]),sq,np.zeros(3))
        moved=transform(c['b'])
        if objective=='shape':
            values.append(tanimoto(pair_overlap(c['a'],moved),self_overlap(c['a']),self_overlap(moved)))
            posed.append(moved)
            continue
        movedc=transform(c['bc'])
        # score_columns identity pose on already transformed geometry avoids
        # sharing the native prepared pose algebra or exact rescore routine.
        value=score_columns(c['a'],moved,c['ac'],c['at'],movedc,c['bt'],
                            c['r'],c['p'],.5 if objective=='combo' else 0.,np.array([1.,0,0,0]),np.zeros(3))
        values.append(value['tanimoto_combination'] if objective=='combo' else value['tanimoto_shape'])
        posed.append(moved)
    return np.array(values),posed


def stats(cases, anchor, candidate, objective, mask, astarts, cstarts, *, anchor_scored=None):
    av,ap=scored(cases,anchor,objective) if anchor_scored is None else anchor_scored
    cv,cp=scored(cases,candidate,objective)
    flips=np.array([np.sqrt(np.mean(np.sum((a-b)**2,axis=1)))>BASIN_RESOLUTION for a,b in zip(ap,cp)])
    labels=stratum_of(av)
    def reduce(sel):
        n=int(np.sum(sel))
        if not n:return dict(n=0,**{k:None for k in METRICS})
        return dict(n=n,mean_loss=float(np.mean((av-cv)[sel])),
            winning_start_disagreement=(None if astarts is None or cstarts is None else
                float(np.mean(np.array(astarts)[sel]!=np.array(cstarts)[sel]))),
            basin_change=float(np.mean(flips[sel])))
    return dict(overall=reduce(mask),by_stratum={name:reduce(mask&(labels==name)) for name,_,_ in SCORE_STRATA})


def freeze(controls,oracle):
    return {k:(None if any(c['overall'][k] is None for c in controls) else
        max(abs(c['overall'][k]) if k=='mean_loss' else c['overall'][k] for c in controls)
        +(oracle if k=='mean_loss' else 0)) for k in METRICS}


def exceeds(record,limits):
    return [k for k in METRICS if limits[k] is not None and record['overall'][k] is not None
            and record['overall'][k]>limits[k]]


def seed_intervals(controls):
    rng=np.random.default_rng(270500)
    indices=rng.integers(0,len(controls),(2000,len(controls)))
    result={}
    for key in METRICS:
        values=[c['overall'][key] for c in controls]
        result[key]=None if any(v is None for v in values) else np.quantile(
            np.asarray(values)[indices].mean(axis=1),[.025,.975]).tolist()
    return result


def assert_zero_instrument(anchor_scored, control_scored):
    """The zero-amplitude instrument must preserve every score and physical pose."""
    np.testing.assert_array_equal(control_scored[0], anchor_scored[0],
                                  err_msg='zero instrument changed per-pair scores')
    for anchor_pose, control_pose in zip(anchor_scored[1], control_scored[1], strict=True):
        np.testing.assert_array_equal(control_pose, anchor_pose,
                                      err_msg='zero instrument changed physical pose')


def run(*,cuda=False,control_module='_numerics_optimizer_control'):
    from roshambo3.backend import cpu
    try:
        probe=importlib.import_module(control_module)
    except ModuleNotFoundError:
        raise RuntimeError('compile bench/optimizer_control.cpp as _numerics_optimizer_control; see optimizer README') from None
    cases,oracle=panel()
    train=np.array([c['transform']%2==0 for c in cases]);test=~train
    result=dict(protocol='optimizer-real-v2',settings=SETTINGS,oracle_allowance=oracle,
        control_quantile=.9,control_confidence=1-.9**len(CALIBRATION),
        scalar_stamp=cpu.build_stamp(),
        control_stamp=dict(source_sha256=hashlib.sha256(Path(__file__).with_name('optimizer_control.cpp').read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
            compiler_command=getattr(probe,'__compiler_command__',None),
            compiler_flags=getattr(probe,'__compiler_flags__',None)),
        qualified=False,objectives={})
    def native(inputs,objective,*,production=False,lr_q=.1):
        opts={} if objective=='shape' else dict(color=True,color_mode='combo_optimal',mixing=.5,
                                               color_grain=16 if production else 0)
        if objective=='shape':opts.update(grain=2 if production else -1,poly=production)
        rows,details=cpu.score_pairs(**inputs,**{**SETTINGS,'lr_q':lr_q},**opts,with_census=True)
        winners=details.get('winning_starts')
        if winners is not None and (np.any(np.array(winners)<0) or np.any(np.array(winners)>3)):
            winners=None
        return rows,winners
    for objective in ('shape','combo'):
        inputs=pack(cases,objective)
        anchor,astarts=native(inputs,objective)
        if astarts is None: raise ValueError('scalar winning starts unavailable')
        anchor_values=scored(cases,anchor,objective)
        def compare(rows,mask,starts):
            return stats(cases,anchor,rows,objective,mask,astarts,starts,anchor_scored=anchor_values)
        calibration=[];heldout=[]
        for seeds,mask,out in ((CALIBRATION,train,calibration),(HELDOUT,test,heldout)):
            for seed in seeds:
                rows,starts=native(pack(cases,objective,seed),objective)
                records=[compare(rows,mask,starts)]
                if objective=='shape':
                    rows=[];starts=[]
                    for c in cases:
                        row,start=probe.score(c['a'],c['b'],seed,2**-18,25,.1,SETTINGS['lr_t'])
                        rows.append(row);starts.append(start)
                    records.append(compare(np.asarray(rows),mask,starts))
                # Preserve both instruments, and calibrate on their envelope.
                summary={k:max(abs(r['overall'][k]) if k=='mean_loss' else r['overall'][k]
                               for r in records) for k in METRICS}
                out.append(dict(seed=seed,overall=dict(n=int(mask.sum()),**summary),instruments=records))
            if seeds is CALIBRATION:
                limits=freeze(calibration,oracle)  # BEFORE held-out/production/plant
        zero=[];zero_starts=[]
        if objective=='shape':
            for c in cases:
                row,start=probe.score(c['a'],c['b'],0,0.,25,.1,SETTINGS['lr_t'])
                zero.append(row);zero_starts.append(start)
            zero_rows=np.asarray(zero)
            assert_zero_instrument(anchor_values, scored(cases,zero_rows,objective))
            zero_stats=compare(zero_rows,test,zero_starts)
            zero_stats['exact_per_pair_match']=True
        else:zero_stats=None
        rows,starts=native(inputs,objective,production=True)
        production=compare(rows,test,starts)
        rows,starts=native(inputs,objective,production=True,lr_q=.11)
        plant=compare(rows,test,starts)
        item=dict(limits=limits,calibration=calibration,heldout=heldout,
            control_mean_intervals=seed_intervals(calibration),zero_instrument=zero_stats,
            heldout_valid=not any(exceeds(r,limits) for r in heldout),
            cpu=dict(production=production,exceeding=exceeds(production,limits),plant=plant,
                     plant_exceeding=exceeds(plant,limits)),cuda=dict(ran=False,why='not requested'),
            qualified=False,gaps=['finite perturbation ensemble has no proof of production coverage'])
        if result['control_stamp']['compiler_command'] is None:
            item['gaps'].append('control instrument compiler invocation not recorded')
        if objective=='combo':item['gaps'].append('color exponential perturbation control absent')
        if cuda:
            item['cuda']=cuda_observation(cases,inputs,objective,anchor,astarts,test,limits)
        item['control_model_pass'] = (item['heldout_valid'] and not item['cpu']['exceeding']
            and bool(item['cpu']['plant_exceeding']) and all(limits[k] is not None for k in METRICS)
            and (zero_stats is None or not exceeds(zero_stats,limits)))
        item['qualified_under_declared_model'] = (item['control_model_pass'] and objective=='shape'
            and result['control_stamp']['compiler_command'] is not None)
        item['qualification_scope'] = 'finite shape perturbation model only; no universal architecture claim'
        item['settings_control_sha256'] = hashlib.sha256(json.dumps(SETTINGS,sort_keys=True).encode()).hexdigest()
        item['settings_mutant_sha256'] = hashlib.sha256(json.dumps({**SETTINGS,'lr_q':.11},sort_keys=True).encode()).hexdigest()
        result['objectives'][objective]=item
    return result


def cuda_observation(cases,inputs,objective,anchor,astarts,mask,limits):
    """Run both live CUDA arms against already frozen control limits."""
    try:
        from roshambo3 import backend
        from roshambo3.backend.spec import PoseSearch,ColorPlacement
        from roshambo3.layout import tile32
        from bench.numerics_backends import verified_build
        gpu=backend.select('cuda')
        g=dict(inputs,library_coords=tile32(inputs['library_coords']))
        if objective=='shape':g.update(query_types=np.zeros(inputs['query_coords'].shape[:2],np.int32),
            interaction_r=np.ones((1,1),np.float32),interaction_p=np.zeros((1,1),np.float32))
        opts=dict(pose_search=PoseSearch.ANALYTIC,color=objective=='combo',mixing=.5,
                  color_placement=ColorPlacement.IN_GRADIENT)
        if objective=='combo':opts['library_color_offset']=-1
        starts=np.full(len(cases),-1,np.int32)
        plant_starts=np.full(len(cases),-1,np.int32)
        rows=gpu.score_pairs(**g,**SETTINGS,**opts,winning_starts=starts)
        plant=gpu.score_pairs(**g,**{**SETTINGS,'lr_q':.11},**opts,winning_starts=plant_starts)
        if np.any((starts<0)|(starts>3)) or np.any((plant_starts<0)|(plant_starts>3)):
            raise ValueError('CUDA returned invalid native winning-start metadata')
        gs=stats(cases,anchor,rows,objective,mask,astarts,starts)
        gp=stats(cases,anchor,plant,objective,mask,astarts,plant_starts)
        return dict(ran=True,stamp=verified_build('cuda'),production=gs,exceeding=exceeds(gs,limits),
            plant=gp,plant_exceeding=exceeds(gp,limits),qualified=False,
            winning_start_source='native strict-improvement selection branch; int32 sidecar',
            why='finite perturbation-model coverage only; FIELD metadata not implemented')
    except Exception as exc:
        return dict(ran=False,qualified=False,why=repr(exc))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args();evidence=run(cuda=args.cuda)
    args.out.write_text(json.dumps(evidence,indent=2,default=lambda x:x.item())+'\n')
    print('Optimizer real panel NOT QUALIFIED:',args.out)
    return 1

if __name__=='__main__':raise SystemExit(main())

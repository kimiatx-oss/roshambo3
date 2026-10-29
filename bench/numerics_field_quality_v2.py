"""Preregistered direct-point FIELD controls; frozen before heldout/native search."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import numpy as np
from bench import numerics_field_quality as old
from roshambo3.reference import field_quality as f
from roshambo3.reference.field_quality_v2 import point_control,search_all,wa4
from roshambo3.reference.field_intervals import assert_interval
from roshambo3.reference.overlap import PAIR_PREFACTOR,self_overlap
from roshambo3.metrics.floors import SCORE_STRATA,stratum_of

CALIBRATION=tuple(range(276300,276332));HELDOUT=tuple(range(276400,276416))
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'doc/numerics/FIELD_QUALITY_V2_PROTOCOL.md'


def identity():
    paths=[PROTOCOL,Path(__file__).resolve(),ROOT/'bench/numerics_field_quality.py',
        ROOT/'bench/numerics_optimizer_real.py',ROOT/'bench/numerics_optimizer_basins.py',
        ROOT/'bench/numerics_fixtures.py',ROOT/'bench/numerics_cuda_field_winners.py',
        ROOT/'bench/numerics_backends.py',ROOT/'bench/run_numerics_checks.py',ROOT/'bench/gpu_lock.py',
        ROOT/'bench/audits/run_field_quality_phase.py',ROOT/'scripts/source_stamp.py']
    paths+=sorted((ROOT/'src/roshambo3').rglob('*.py'))
    paths+=sorted(p for p in (ROOT/'tests/data/alignment').glob('*') if p.is_file())
    return {str(p.relative_to(ROOT)):old.digest(p) for p in paths}


def unchanged(captured):
    if captured!=identity():raise ValueError('frozen v2 model/driver/fixture changed')


PHASES=('prepared','calibration','frozen','heldout','validation','observed','report')


def phase_fresh(args,name):
    old.fresh(*(args.state/(later+suffix) for later in PHASES[PHASES.index(name):] for suffix in ('.json','.npz')))


def read(args,name):
    record=json.loads((args.state/(name+'.json')).read_text());unchanged(record['identity'])
    for path,sha in record.get('artifacts',{}).items():
        if old.digest(args.state/path)!=sha:raise ValueError('artifact identity mismatch: '+path)
    for key,sha in record.items():
        if key.endswith('_sha256') and sha is not None:
            dependency=args.state/(key[:-7]+'.json')
            if not dependency.is_file() or old.digest(dependency)!=sha:raise ValueError('dependency identity mismatch: '+key)
    return record


def save(args,name,record,arrays=None):
    unchanged(args.identity)
    report=args.state/(name+'.json');old.fresh(report)
    artifact=dict(record.pop('artifacts',{}))
    if arrays is not None:
        array_path=args.state/(name+'.npz');old.fresh(array_path)
        with array_path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        artifact[array_path.name]=old.digest(array_path)
    old.write_json(report,dict(identity=args.identity,source=old.revision(),artifacts=artifact,**record))


def panel():
    from scipy.spatial.transform import Rotation
    from bench.numerics_fixtures import load_panel
    fixtures=load_panel();rng=np.random.default_rng(276200)
    rotations=Rotation.random(8,random_state=rng).as_matrix();translations=rng.uniform(-1,1,(8,3))
    cases=[]
    for transform,(rotation,translation) in enumerate(zip(rotations,translations)):
        for a in fixtures:
            for b in fixtures:
                cases.append(dict(a=f.staged(a.shape).astype(float),b=f.staged(b.shape@rotation.T+translation).astype(float),
                    ac=f.staged(a.color).astype(float),bc=f.staged(b.color@rotation.T+translation).astype(float),
                    at=a.types,bt=b.types,r=a.interaction_r,p=a.interaction_p,transform=transform,query=a.name,library=b.name))
    return cases


def rows(poses):
    out=np.zeros((len(poses),20));out[:,9:13]=poses[:,:4];out[:,13:16]=poses[:,4:];out[:,16]=1
    return out


def _polish(task):
    from bench.numerics_optimizer_basins import polish
    index,case,row=task;report,cloud,_=polish(case,row,'shape')
    return index,report,cloud


def polished(cases,pose_rows):
    maximum=max(len(c['b']) for c in cases)
    endpoints=np.full((len(cases),maximum,3),np.nan);stationary=np.zeros(len(cases),bool);records=[]
    tasks=[(i,c,row) for i,(c,row) in enumerate(zip(cases,pose_rows)) if c['transform']%2==1]
    # CPU-only phase in a fresh process, after the prior native child exited.
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as executor:
        for index,report,cloud in executor.map(_polish,tasks):
            endpoints[index,:len(cloud)]=cloud;stationary[index]=report['stationary']
            records.append(dict(index=index,**report))
    return endpoints,stationary,records


def basin_flags(cases,anchor_endpoints,anchor_stationary,endpoints,stationary):
    from bench.numerics_fixtures import RECOVERY_NEIGHBORHOOD
    flags=np.zeros(len(cases),bool)
    for i,case in enumerate(cases):
        if case['transform']%2==0:continue
        if not(anchor_stationary[i] and stationary[i]):raise ValueError('unresolved independent attraction basin')
        n=len(case['b']);distance=np.sqrt(np.mean(np.sum((anchor_endpoints[i,:n]-endpoints[i,:n])**2,axis=1)))
        flags[i]=distance>RECOVERY_NEIGHBORHOOD
    return flags


def masks(data):
    labels=stratum_of(data['anchor_values'])
    return dict(overall=np.ones(len(labels),bool),**{name:labels==name for name,_,_ in SCORE_STRATA})


def metrics(data,raw,starts,basins):
    if not np.isfinite(raw).all():raise AssertionError('nonfinite actual sampled objective')
    value=f.phi(raw,data['self_sums']);anchor=f.phi(data['anchor_F'],data['self_sums'])
    result={}
    for name,mask in masks(data).items():
        bm=mask&data['basin_mask'];n=int(mask.sum());nb=int(bm.sum())
        result[name]=dict(n=n,basin_n=nb,
            sampled_regret=float(np.mean((anchor-value)[mask])) if n else None,
            start_disagreement=float(np.mean((starts!=data['anchor_starts'])[mask])) if n else None,
            basin_disagreement=float(np.mean(basins[bm])) if nb else None)
    return result


def limits(controls):
    return {name:{metric:(None if records[0][metric] is None else max(r[metric] for r in records))
        for metric in ('sampled_regret','start_disagreement','basin_disagreement')}
        for name in controls[0] for records in [[r[name] for r in controls]]}


def exceeding(record,bounds):
    return [name+'/'+metric for name in bounds for metric,limit in bounds[name].items()
        if limit is not None and record[name][metric]>limit]


def prepare(args):
    from bench.run_numerics_checks import select_native_build
    import roshambo3
    select_native_build(roshambo3,args.build)
    from bench.numerics_optimizer_real import pack,SETTINGS,scored
    from roshambo3.backend import cpu
    args.state.mkdir(exist_ok=False)
    cases=panel();inputs=pack(cases,'shape');fields=old.geometry(cases)
    stamp=old.loaded_stamp('cpu',args.build)
    anchor_rows,details=cpu.score_pairs(**inputs,**SETTINGS,grain=-1,poly=False,with_census=True)
    anchor_poses=np.array([old.row_pose(row) for row in anchor_rows]);values,_=scored(cases,anchor_rows,'shape')
    endpoints,stationary,polish_records=polished(cases,anchor_rows)
    poses=[];guards=[]
    for seed in CALIBRATION+HELDOUT:
        rng=np.random.default_rng(seed);realizations={};items=[];guard=[]
        for case,(field,origin,spacing) in zip(cases,fields):
            key=case['a'].tobytes()
            if key not in realizations:realizations[key]=np.where(rng.integers(0,2,field.lo.shape,dtype=np.uint8),field.hi,field.lo)
            all_poses=search_all(np.column_stack((case['a'],np.ones(len(case['a'])))),case['b'],realizations[key],origin,spacing)
            items.append(all_poses);guard.append([point_control(case['b'],pose)[1] for pose in all_poses])
        poses.append(items);guards.append(guard);print('prepared v2 independent seed',seed,flush=True)
    anchor_guard=[point_control(c['b'],p)[1] for c,p in zip(cases,anchor_poses)]
    arrays=dict(inputs,anchor_rows=anchor_rows,anchor_poses=anchor_poses,anchor_values=values,
        anchor_starts=np.array(details['winning_starts']),self_sums=old.self_sums(cases),control_poses=poses,
        anchor_endpoints=endpoints,anchor_stationary=stationary,
        basin_mask=np.array([c['transform']%2==1 for c in cases]),seeds=CALIBRATION+HELDOUT)
    if old.loaded_stamp('cpu',args.build)!=stamp:raise ValueError('CPU anchor binary changed during preparation')
    save(args,'prepared',dict(cpu_stamp=stamp,anchor_polish=polish_records,point_conversion_guards=guards,
        anchor_conversion_guards=anchor_guard,native_controls_observed=False,native_optimized_observed=False),arrays)


def built_fields(cases,ext):
    cache={};lookup=[];bits={}
    for case in cases:
        query=np.column_stack((case['a'],np.ones(len(case['a'])))).astype(np.float32);key=query.tobytes()
        if key not in cache:
            origin,spacing,cells=f.lattice(query)
            node_bits=ext.numerics_build_field(query,origin.tolist(),float(spacing),cells.tolist(),True,1.)
            expected,_,_=f.nodes(query);assert_interval(node_bits.view(np.float16).astype(float),expected)
            cache[key]=(node_bits,origin,spacing);bits['query_'+str(len(cache)-1)]=node_bits
        lookup.append(cache[key])
    return lookup,bits


def probe(cases,poses,fields,ext):
    # One native sampler call per unique query; retain per-row/start WA4 sums.
    poses=np.asarray(poses);result=np.zeros(poses.shape[:2],np.float32)
    groups={}
    for i,case in enumerate(cases):groups.setdefault(case['a'].tobytes(),[]).append(i)
    for indices in groups.values():
        clouds=[];sizes=[]
        for i in indices:
            for pose in poses[i]:
                cloud,_=point_control(cases[i]['b'],pose);clouds.append(cloud);sizes.append(len(cloud))
        bits,origin,spacing=fields[indices[0]]
        sampled=ext.numerics_sample_field(bits,np.concatenate(clouds),origin.tolist(),float(spacing))[:,0]
        if not np.isfinite(sampled).all():raise AssertionError('nonfinite production point sample')
        offset=0;k=0
        for i in indices:
            for start in range(poses.shape[1]):
                n=sizes[k];result[i,start]=wa4(sampled[offset:offset+n]);offset+=n;k+=1
    return result


def _native_points(args,heldout=False):
    phase_fresh(args,'heldout' if heldout else 'calibration')
    prepared=read(args,'prepared');d=np.load(args.state/'prepared.npz');cases=panel()
    if heldout:read(args,'frozen')
    name='heldout' if heldout else 'calibration';old.fresh(args.state/(name+'.json'),args.state/(name+'.npz'))
    def operation(stamp):
        from roshambo3.backend import cuda
        ext=cuda.extension();fields,node_bits=built_fields(cases,ext)
        if old.loaded_stamp('cpu',args.build)!=prepared['cpu_stamp']:raise ValueError('scalar anchor build changed')
        if heldout:
            cal=read(args,'calibration');saved=np.load(args.state/'calibration.npz')
            if stamp!=cal['native_stamp']:raise ValueError('native calibration binary changed')
            for key,bits in node_bits.items():np.testing.assert_array_equal(bits,saved[key])
        else:
            # Exact same production builder/sampler geometry at original identity.
            from bench.numerics_optimizer_real import pack
            _,_,actual=old.native(pack(cases,'shape'),fixed=True)
            identity=np.tile([1.,0,0,0,0,0,0],(len(cases),1))[:,None,:]
            sampled=probe(cases,identity,fields,ext)
            np.testing.assert_array_equal(actual.view(np.uint32),sampled.view(np.uint32),err_msg='production point sampler identity differs from native FIELD search')
        anchor=probe(cases,d['anchor_poses'][:,None,:],fields,ext)[:,0]
        if heldout:np.testing.assert_array_equal(anchor,saved['anchor_F'])
        selected=range(32,48) if heldout else range(32)
        objectives=np.array([probe(cases,d['control_poses'][i],fields,ext) for i in selected])
        if old.loaded_stamp('cuda',args.build)!=stamp:raise ValueError('native binary changed during point probes')
        save(args,name,dict(native_stamp=stamp,prepared_sha256=old.digest(args.state/'prepared.json'),
            frozen_sha256=old.digest(args.state/'frozen.json') if heldout else None,
            native_optimized_observed=False,identity_metadata_exact=not heldout),dict(anchor_F=anchor,control_F=objectives,**node_bits))
    old.lock_run(args,operation)


def calibration(args):_native_points(args)
def heldout(args):_native_points(args,True)


def control_records(args,start,end,base):
    read(args,'prepared');d=dict(np.load(args.state/'prepared.npz'));native=read(args,base)
    observed=np.load(args.state/(base+'.npz'));d['anchor_F']=observed['anchor_F']
    cases=panel();records=[];polishes=[];selected_poses=[];all_endpoints=[];all_stationary=[]
    for local,index in enumerate(range(start,end)):
        objectives=observed['control_F'][local];starts=np.argmax(objectives,axis=1)
        chosen=d['control_poses'][index,np.arange(len(cases)),starts]
        endpoints,stationary,report=polished(cases,rows(chosen))
        basins=basin_flags(cases,d['anchor_endpoints'],d['anchor_stationary'],endpoints,stationary)
        raw=objectives[np.arange(len(cases)),starts]
        records.append(metrics(d,raw,starts,basins));polishes.append(report);selected_poses.append(chosen)
        all_endpoints.append(endpoints);all_stationary.append(stationary)
        print('classified independent control basin seed',int(d['seeds'][index]),flush=True)
    return records,dict(selected_poses=selected_poses,endpoints=all_endpoints,stationary=all_stationary),polishes,native


def freeze(args):
    phase_fresh(args,'frozen')
    old.fresh(args.state/'frozen.json',args.state/'frozen.npz')
    records,arrays,polishes,native=control_records(args,0,32,'calibration')
    save(args,'frozen',dict(limits=limits(records),calibration=records,polishes=polishes,
        prepared_sha256=old.digest(args.state/'prepared.json'),calibration_sha256=old.digest(args.state/'calibration.json'),
        native_stamp=native['native_stamp'],heldout_observed=False,native_optimized_observed=False),arrays)


def validate(args):
    phase_fresh(args,'validation')
    frozen=read(args,'frozen');read(args,'heldout')
    records,arrays,polishes,_=control_records(args,32,48,'heldout')
    failures=[exceeding(r,frozen['limits']) for r in records]
    save(args,'validation',dict(heldout=records,heldout_exceeding=failures,heldout_pass=not any(failures),polishes=polishes,
        frozen_sha256=old.digest(args.state/'frozen.json'),native_optimized_observed=False),arrays)


def observe(args):
    phase_fresh(args,'observed')
    frozen=read(args,'frozen');validation=read(args,'validation');prepared=read(args,'prepared');cal=read(args,'calibration')
    old.fresh(args.state/'observed.json',args.state/'observed.npz')
    d=np.load(args.state/'prepared.npz');cases=panel()
    def operation(stamp):
        if stamp!=frozen['native_stamp'] or old.loaded_stamp('cpu',args.build)!=prepared['cpu_stamp']:raise ValueError('frozen binary changed')
        from bench.numerics_optimizer_real import pack
        arrays={}
        for label,plant,lr in [('clean',False,.1),('zero_lr_plant',True,0.),('lr_q11_plant',False,.11)]:
            result,starts,sampled=old.native(pack(cases,'shape'),plant=plant,lr_q=lr)
            arrays.update({label+'_rows':result,label+'_starts':starts,label+'_sampled':sampled})
        if old.loaded_stamp('cuda',args.build)!=stamp:raise ValueError('native binary changed during optimized search')
        save(args,'observed',dict(native_stamp=stamp,validation_sha256=old.digest(args.state/'validation.json'),
            frozen_sha256=old.digest(args.state/'frozen.json'),invoked={'clean':[25,.1,.4],'lr_q11_plant':[25,.11,.4],'zero_lr_plant':[25,0.,0.]}),arrays)
    old.lock_run(args,operation)


def report(args):
    phase_fresh(args,'report')
    frozen=read(args,'frozen');validation=read(args,'validation');read(args,'observed');read(args,'prepared');read(args,'calibration')
    d=dict(np.load(args.state/'prepared.npz'));d['anchor_F']=np.load(args.state/'calibration.npz')['anchor_F']
    observed=np.load(args.state/'observed.npz');cases=panel();fields=old.geometry(cases);records={};polishes={};saved={}
    from bench.numerics_optimizer_real import scored
    for label in ('clean','zero_lr_plant','lr_q11_plant'):
        rr=observed[label+'_rows'];starts=observed[label+'_starts'];sampled=observed[label+'_sampled']
        np.testing.assert_array_equal(starts,np.argmax(sampled,axis=1));raw=sampled[np.arange(len(cases)),starts]
        own,_=scored(cases,rr,'shape');ends,stationary,polish=polished(cases,rr)
        basins=basin_flags(cases,d['anchor_endpoints'],d['anchor_stationary'],ends,stationary)
        record=metrics(d,raw,starts,basins);upper=[];inside=[]
        for i,(case,(field,origin,spacing),row,total) in enumerate(zip(cases,fields,rr,d['self_sums'])):
            bound=f.overlap_interval(field,f.native_points(case['b'],row[9:13],row[13:16],returned_translation=True),origin,spacing)
            inside.append(bool(bound.lo<=raw[i]<=bound.hi));upper.append(float(f.phi_interval(bound,total).hi-own[i]))
        signed=d['anchor_values']-f.phi(d['anchor_F'],d['self_sums'])
        for name,mask in masks(d).items():
            if not mask.any():continue
            allowance=float(np.mean(signed[mask]))+frozen['limits'][name]['sampled_regret']+float(np.mean(np.array(upper)[mask]))
            record[name].update(mean_loss=float(np.mean((d['anchor_values']-own)[mask])),allowance=allowance,
                informative=allowance<float(np.mean(d['anchor_values'][mask])),
                quality_pass=float(np.mean((d['anchor_values']-own)[mask]))<=allowance,
                distortion_rows_pass=bool(np.all(np.array(inside)[mask])))
        failed=exceeding(record,frozen['limits'])
        records[label]=dict(metrics=record,exceeding=failed,
            quality_informative=all(r.get('informative',True) for r in record.values()),
            quality_pass=all(r.get('quality_pass',True) and r.get('distortion_rows_pass',True) for r in record.values()))
        polishes[label]=polish;saved.update({label+'_endpoints':ends,label+'_stationary':stationary,label+'_own':own})
    clean=records['clean'];rate=records['lr_q11_plant'];zero=records['zero_lr_plant']
    passed=validation['heldout_pass'] and not clean['exceeding'] and clean['quality_informative'] and clean['quality_pass'] and bool(rate['exceeding']) and (any(x.endswith('/sampled_regret') for x in zero['exceeding']) or not zero['quality_pass'])
    save(args,'report',dict(records=records,polishes=polishes,heldout_pass=validation['heldout_pass'],model_pass=passed,
        qualified=False,scope='fixed geometry finite independently derived controls; shape only; conditional scalar consequence',
        observed_sha256=old.digest(args.state/'observed.json'),frozen_sha256=old.digest(args.state/'frozen.json')),saved)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','calibration','freeze','heldout','validate','observe','report'])
    parser.add_argument('--build',type=Path,required=True);parser.add_argument('--state',type=Path,required=True)
    args=parser.parse_args();args.identity=identity();globals()[args.phase](args)

if __name__=='__main__':main()

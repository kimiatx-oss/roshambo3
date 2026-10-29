"""Frozen independent FIELD endpoint controls and scalar-anchor quality gates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from roshambo3.reference import field_quality as fq
from roshambo3.reference.overlap import PAIR_PREFACTOR,pair_overlap,self_overlap,tanimoto

CALIBRATION=tuple(range(275300,275332))
HELDOUT=tuple(range(275400,275416))
PROTOCOL=Path(__file__).resolve().parents[1]/'doc/numerics/FIELD_QUALITY_PROTOCOL.md'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    with Path(path).open('x') as stream:json.dump(data,stream,indent=2);stream.write('\n')
def fresh(*paths):
    for path in paths:
        if Path(path).exists():raise FileExistsError('preserving existing evidence: '+str(path))
def revision():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def source_identity():
    root=PROTOCOL.parents[2]
    files=[PROTOCOL,Path(__file__).resolve(),root/'bench/numerics_fixtures.py',
        root/'bench/numerics_optimizer_real.py',root/'bench/numerics_cuda_field_winners.py',
        root/'src/roshambo3/reference/field_quality.py',root/'src/roshambo3/reference/field_intervals.py',
        root/'src/roshambo3/reference/overlap.py',root/'src/roshambo3/numerics.py']
    files+=sorted((root/'src/roshambo3').rglob('*.py'))
    files += [root/'bench/run_numerics_checks.py',root/'bench/numerics_backends.py',root/'bench/gpu_lock.py',root/'scripts/source_stamp.py']
    files+=sorted(p for p in (root/'tests/data/alignment').glob('*') if p.is_file())
    return {str(p.relative_to(root)):digest(p) for p in files}


def assert_identity(record):
    if record['source_identity']!=source_identity():raise ValueError('frozen model/driver/fixture identity mismatch')
    if record['protocol_sha256']!=digest(PROTOCOL):raise ValueError('frozen protocol identity mismatch')


def unchanged(identity):
    if identity!=source_identity():raise RuntimeError('source changed during frozen experiment')


def loaded_stamp(kind,build):
    from bench.numerics_backends import verified_build
    from roshambo3.backend import cpu,cuda
    stamp=verified_build(kind)
    extension=Path((cpu if kind=='cpu' else cuda).extension().__file__).resolve()
    if extension.parent!=build.resolve():raise RuntimeError('loaded extension outside requested build')
    return json.loads(json.dumps(dict(stamp,loaded_extension=str(extension),binary_sha256=digest(extension))))



def panel():
    from scipy.spatial.transform import Rotation
    from bench.numerics_fixtures import load_panel
    rng=np.random.default_rng(275200)
    rotations=Rotation.random(8,random_state=rng).as_matrix()
    translations=rng.uniform(-1,1,(8,3));cases=[]
    def wire(x):return fq.staged(x).astype(float)
    for transform,(rotation,translation) in enumerate(zip(rotations,translations)):
        for a in load_panel():
            for b in load_panel():
                cases.append(dict(a=wire(a.shape),b=wire(b.shape@rotation.T+translation),
                    ac=a.color,bc=b.color,at=a.types,bt=b.types,r=a.interaction_r,p=a.interaction_p,
                    transform=transform,query=a.name,library=b.name))
    return cases


def geometry(cases):
    cache={};lookup=[]
    for case in cases:
        query=np.column_stack((case['a'],np.ones(len(case['a'])))).astype(np.float32)
        key=query.tobytes()
        if key not in cache:cache[key]=fq.nodes(query)
        lookup.append(cache[key])
    return lookup


def self_sums(cases):
    return np.array([(self_overlap(c['a'])+self_overlap(c['b']))/PAIR_PREFACTOR for c in cases])


def row_pose(row):
    # Scalar's prepared signed-axis frame is explicitly undone in physical units.
    from scipy.spatial.transform import Rotation
    from roshambo3.reference.overlap import quaternion_matrix
    matrix=quaternion_matrix(row[16:20]*[1,-1,-1,-1])@quaternion_matrix(row[9:13])
    q=Rotation.from_matrix(matrix).as_quat()[[3,0,1,2]]
    t=quaternion_matrix(row[16:20]*[1,-1,-1,-1])@row[13:16]
    return np.r_[q,t]


def prepare(args):
    fresh(args.data,args.report)
    identity=source_identity()
    import roshambo3
    from bench.run_numerics_checks import select_native_build
    select_native_build(roshambo3,args.build)
    from roshambo3.backend import cpu
    from bench.numerics_optimizer_real import pack,SETTINGS,scored
    cases=panel();inputs=pack(cases,'shape');fields=geometry(cases);sums=self_sums(cases)
    stamp=loaded_stamp('cpu',args.build)
    rows,details=cpu.score_pairs(**inputs,**SETTINGS,grain=-1,poly=False,with_census=True)
    anchors=np.array([row_pose(row) for row in rows])
    anchor_values,_=scored(cases,rows,'shape')
    # Independently reconstructed anchor poses must agree with the existing oracle.
    reconstructed=np.array([tanimoto(pair_overlap(c['a'],fq.own_pose(c['b'],p)),
        self_overlap(c['a']),self_overlap(c['b'])) for c,p in zip(cases,anchors)])
    np.testing.assert_allclose(reconstructed,anchor_values,rtol=0,atol=2e-14)
    anchor_slack=np.array([fq.staging_slack(f,c['b'],p,o,h,s) for (f,o,h),c,p,s in zip(fields,cases,anchors,sums)])
    controls=[];starts=[];slacks=[]
    for seed in CALIBRATION+HELDOUT:
        rng=np.random.default_rng(seed);realizations={};poses=[];winners=[];bounds=[]
        for c,(f,o,h),total in zip(cases,fields,sums):
            key=c['a'].tobytes()
            if key not in realizations:
                realizations[key]=np.where(rng.integers(0,2,f.lo.shape,dtype=np.uint8),f.hi,f.lo)
            pose,winner,_=fq.search(np.column_stack((c['a'],np.ones(len(c['a'])))),c['b'],realizations[key],o,h)
            poses.append(pose);winners.append(winner)
            bounds.append(fq.staging_slack(f,c['b'],pose,o,h,total))
        controls.append(poses);starts.append(winners);slacks.append(bounds)
        print('prepared independent control seed',seed,flush=True)
    if identity!=source_identity():raise RuntimeError('source changed while generating controls')
    # Open exclusively: no overwrite of a previous preparation or preregistration.
    with Path(args.data).open('xb') as stream:
        np.savez_compressed(stream,**inputs,anchor_rows=rows,anchor_poses=anchors,
            anchor_values=anchor_values,anchor_starts=np.array(details['winning_starts']),
            anchor_slack=anchor_slack,control_poses=np.asarray(controls),control_starts=starts,
            control_slack=slacks,self_sums=sums,seeds=CALIBRATION+HELDOUT)
    write_json(args.report,dict(protocol_sha256=digest(PROTOCOL),source=revision(),
        data_sha256=digest(args.data),cpu_stamp=stamp,source_identity=identity,calibration=list(CALIBRATION),heldout=list(HELDOUT),
        geometry_seed=275200,pairs=len(cases),native_optimized_observed=False,qualified=False))


def native(inputs, *, fixed=False, plant=False, lr_q=.1):
    from roshambo3.backend import cuda
    from roshambo3.backend.spec import PoseSearch
    from bench.numerics_cuda_field_winners import gpu_inputs
    starts=1 if fixed else 4
    ids=np.full(len(inputs['pair_query_row']),-1,np.int32)
    sampled=np.full((len(ids),starts),np.nan,np.float32)
    rows=cuda.score_pairs(**gpu_inputs(inputs,integer=True),pose_search=PoseSearch.FIELD,color=False,
        library_color_offset=0,steps=0 if fixed else 25,start_mode=0 if fixed else 1,
        lr_q=0. if plant else lr_q,lr_t=0. if plant else .4,field_spacing_a=.8,field_pad_a=8.,
        winning_starts=ids,start_objectives=sampled)
    if not np.isfinite(sampled).all():raise AssertionError('nonfinite actual sampled objective')
    np.testing.assert_array_equal(rows[:,16:20],np.tile([1,0,0,0],(len(rows),1)),
        err_msg='FIELD returned nonidentity query frame')
    np.testing.assert_array_equal(ids,np.argmax(sampled,axis=1))
    return rows,ids,sampled


def moved_inputs(data,poses):
    keys=('query_coords','query_n_atoms','library_coords','library_counts','pair_query_row','pair_library_row')
    inputs={key:data[key].copy() for key in keys}
    for i,pose in enumerate(poses):
        n=int(inputs['library_counts'][i,0])
        inputs['library_coords'][i,:n,:3]=fq.own_pose(data['library_coords'][i,:n,:3],pose).astype(np.float32)
    return inputs


def lock_run(args,operation):
    import roshambo3
    from bench.run_numerics_checks import select_native_build
    select_native_build(roshambo3,args.build)
    from bench.gpu_lock import GpuLock,resolve_device,query_gpu_state
    device=resolve_device(0)
    with GpuLock('independent FIELD scalar quality',device):
        state=query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct>0:raise RuntimeError('GPU busy before context: '+str(state))
        stamp=loaded_stamp('cuda',args.build)
        operation(stamp)


def controls(args):
    frozen=Path(str(args.report)+'.frozen.json')
    fresh(args.report,args.observations,frozen)
    prepared=json.loads(Path(args.prepared).read_text())
    assert_identity(prepared)
    identity=source_identity()
    if prepared['protocol_sha256']!=digest(PROTOCOL) or prepared['data_sha256']!=digest(args.data):raise ValueError('preparation identity mismatch')
    data=np.load(args.data)
    def run(stamp):
        if loaded_stamp('cpu',args.build)!=prepared['cpu_stamp']:raise ValueError('scalar anchor binary/source changed')
        _,_,a=native(moved_inputs(data,data['anchor_poses']),fixed=True)
        af=fq.phi(a[:,0],data['self_sums']);values=[];records=[]
        for i,seed in enumerate(CALIBRATION+HELDOUT):
            _,_,c=native(moved_inputs(data,data['control_poses'][i]),fixed=True)
            cf=fq.phi(c[:,0],data['self_sums']);values.append(c[:,0])
            discrepancy=float(np.mean(af-cf+data['anchor_slack']+data['control_slack'][i]))
            records.append(dict(seed=seed,mean_discrepancy=discrepancy))
            if i==len(CALIBRATION)-1:
                unchanged(identity)
                if loaded_stamp('cuda',args.build)!=stamp:raise RuntimeError('native binary changed during calibration')
                regret=max(r['mean_discrepancy'] for r in records)
                write_json(frozen,dict(protocol_sha256=digest(PROTOCOL),data_sha256=digest(args.data),
                    source=revision(),source_identity=identity,native_stamp=stamp,R=regret,calibration=records.copy(),
                    native_optimized_observed=False,heldout_observed=False))
        unchanged(identity)
        if loaded_stamp('cuda',args.build)!=stamp:raise RuntimeError('native binary changed during controls')
        with Path(args.observations).open('xb') as stream:np.savez_compressed(stream,anchor_F=a[:,0],control_F=values)
        write_json(args.report,dict(protocol_sha256=digest(PROTOCOL),data_sha256=digest(args.data),
            observations_sha256=digest(args.observations),frozen_sha256=digest(frozen),native_stamp=stamp,
            source_identity=identity,cpu_stamp=prepared['cpu_stamp'],prepared_sha256=digest(args.prepared),
            R=regret,heldout=records[len(CALIBRATION):],heldout_pass=all(r['mean_discrepancy']<=regret for r in records[len(CALIBRATION):]),
            signed_anchor_distortion=float(np.mean(data['anchor_values']-af)),
            mean_anchor_staging_slack=float(data['anchor_slack'].mean()),qualified=False))
    lock_run(args,run)


def observe(args):
    fresh(args.report,args.observations)
    control=json.loads(Path(args.controls).read_text())
    assert_identity(control)
    identity=source_identity()
    frozen=Path(str(args.controls)+'.frozen.json')
    frozen_record=json.loads(frozen.read_text());assert_identity(frozen_record)
    if digest(frozen)!=control['frozen_sha256'] or digest(args.data)!=control['data_sha256']:raise ValueError('frozen identity mismatch')
    data=np.load(args.data);control_values=np.load(args.control_observations)
    if digest(args.control_observations)!=control['observations_sha256']:raise ValueError('native control identity mismatch')
    cases=panel();fields=geometry(cases)
    def run(stamp):
        if stamp!=control['native_stamp']:raise ValueError('native control binary/source changed')
        if loaded_stamp('cpu',args.build)!=control['cpu_stamp']:raise ValueError('scalar anchor binary/source changed')
        from bench.numerics_optimizer_real import scored
        inputs=moved_inputs(data,np.tile([1.,0,0,0,0,0,0],(len(cases),1)))
        records={};saved={}
        for label,plant,lr_q in [('clean',False,.1),('zero_lr_plant',True,0.),('lr_q11_plant',False,.11)]:
            rows,starts,sampled=native(inputs,plant=plant,lr_q=lr_q)
            raw=sampled[np.arange(len(starts)),starts];normalized=fq.phi(raw,data['self_sums'])
            own,_=scored(cases,rows,'shape');upper=[];inside=[]
            for i,(c,(field,origin,spacing),row,total) in enumerate(zip(cases,fields,rows,data['self_sums'])):
                points=fq.native_points(c['b'],row[9:13],row[13:16],returned_translation=True)
                bound=fq.overlap_interval(field,points,origin,spacing)
                inside.append(bool(bound.lo<=raw[i]<=bound.hi))
                upper.append(float(fq.phi_interval(bound,total).hi-own[i]))
            regret=float(np.mean(fq.phi(control_values['anchor_F'],data['self_sums'])-normalized))
            allowance=control['signed_anchor_distortion']+control['R']+float(np.mean(upper))
            loss=float(np.mean(data['anchor_values']-own))
            informative=allowance<float(np.mean(data['anchor_values']))
            passed=all(inside) and regret<=control['R'] and loss<=allowance and informative
            records[label]=dict(invoked=dict(steps=25,start_mode=1,lr_q=lr_q,lr_t=0. if plant else .4),mean_loss=loss,mean_allowance=allowance,mean_candidate_distortion_bound=float(np.mean(upper)),
                mean_sampled_regret=regret,regret_pass=regret<=control['R'],distortion_rows_pass=all(inside),
                informative=informative,quality_pass=loss<=allowance,pass_all=passed,
                winning_start_counts=np.bincount(starts,minlength=4).tolist())
            saved.update({label+'_rows':rows,label+'_starts':starts,label+'_sampled':sampled,label+'_own':own})
        unchanged(identity)
        if loaded_stamp('cuda',args.build)!=stamp:raise RuntimeError('native binary changed during observations')
        with Path(args.observations).open('xb') as stream:np.savez_compressed(stream,**saved)
        write_json(args.report,dict(protocol_sha256=digest(PROTOCOL),data_sha256=digest(args.data),source=revision(),native_stamp=stamp,
            controls_sha256=digest(args.controls),observations_sha256=digest(args.observations),records=records,
            source_identity=identity,scalar_gate_scope='conditional consequence of sampled regret and directed distortion; not a second independent detector',
            model_pass=control['heldout_pass'] and records['clean']['pass_all'] and all(
                not(records[label]['regret_pass'] and records[label]['quality_pass'] and records[label]['distortion_rows_pass'])
                for label in ('zero_lr_plant','lr_q11_plant')),
            qualified=False,gaps=['finite endpoint control ensemble has no universal native trajectory coverage',
                'winning-start and basin acceptance not derived by scalar gate']))
    lock_run(args,run)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','controls','observe'])
    parser.add_argument('--build',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--observations',type=Path)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--controls',type=Path);parser.add_argument('--control-observations',type=Path)
    args=parser.parse_args()
    if args.phase!='prepare' and args.observations is None:parser.error('--observations required')
    if args.phase=='controls' and args.prepared is None:parser.error('--prepared required')
    if args.phase=='observe' and (args.controls is None or args.control_observations is None):parser.error('control receipts required')
    globals()[args.phase](args)

if __name__=='__main__':main()

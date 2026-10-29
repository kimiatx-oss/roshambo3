"""Canonical full-molecule color aggregate model from actually issued terms."""
from dataclasses import replace
import math
import numpy as np
from bench.numerics_fixtures import load_panel,fixture_binding
from bench.numerics_optimizer_real import pack
from bench.numerics_color_bias import panel_hash
from roshambo3.reference.overlap import color_overlap
from roshambo3.reference.numerics_contracts import PathArithmetic,residual_bounds

DERIVATION=('canonical molecular color: all off-type zero-width interactions skipped; '
    'K=sum_type nQ_type*nB_type actual positive serial additions; gamma(K), '
    'unchanged physical preparation/exponent/coefficient allowances; arbitrary correlation')


def molecular_model(repeats=32):
    if not isinstance(repeats,int) or isinstance(repeats,bool) or repeats<1:
        raise ValueError('positive integer repeats required')
    cases=[]; strata=[]
    for fixture in load_panel():
        s=fixture.shape.astype(np.float32);c=fixture.color.astype(np.float32)
        types=fixture.types
        r,p=fixture.interaction_r.astype(np.float32),fixture.interaction_p.astype(np.float32)
        maximum=max(float(np.max(np.abs(s))),float(np.max(np.abs(c))))
        count=sum(int(n)**2 for n in np.bincount(types,minlength=7)[1:])
        generic=PathArithmetic('canonical_molecular_color_generic',
            coordinate_error=2*2**-24*maximum,exp2_relative_error=2**-22,
            operation_model_derivation=DERIVATION)
        args=dict(query_color=c,query_types=types,library_color=c,library_types=types,
                  interaction_r=r,interaction_p=p)
        physical=color_overlap(c,types,c,types,r,p)
        row=residual_bounds(s,s,contract=generic,**args)['overlap_color']
        bound=residual_bounds(s,s,contract=replace(generic,sum_roundings=count),**args)['overlap_color']
        strata.append(dict(name=fixture.name,physical=physical,row_bound=row,
            aggregate_bound=bound,issued_terms_per_row=count,rows=repeats,
            binding=fixture_binding(fixture)))
        case=dict(a=s,b=s,ac=c,bc=c,at=types,bt=types,r=r,p=p)
        cases.extend([case]*repeats)
    return pack(cases,'combo'),strata


def molecular_report(values,repeats=32):
    panel,strata=molecular_model(repeats)
    values=np.asarray(values,np.float64)
    if values.shape!=(len(strata)*repeats,) or not np.isfinite(values).all():
        raise ValueError('one finite observation per full molecular row required')
    residuals=[];bounds=[];rows_pass=[]
    for i,stratum in enumerate(strata):
        observed=values[i*repeats:(i+1)*repeats]
        errors=[float(v)-stratum['physical'] for v in observed]
        mean=math.fsum(errors)/repeats
        radius=math.nextafter(stratum['aggregate_bound'],math.inf)
        row_pass=all(abs(v)<=stratum['row_bound'] for v in errors)
        stratum.update(mean_bias=mean,mean_bound=radius,mean_pass=abs(mean)<=radius,
            rows_pass=row_pass,observed_mean=math.fsum(map(float,observed))/repeats)
        residuals.extend(errors);bounds.extend([radius]*repeats);rows_pass.append(row_pass)
    bias=math.fsum(residuals)/len(values)
    radius=math.nextafter(math.fsum(bounds)/len(values),math.inf)
    return dict(rows=len(values),rows_pass=all(rows_pass),mean_bias=bias,mean_bound=radius,
        mean_pass=abs(bias)<=radius,strata=strata,panel_sha256=panel_hash(panel),
        derivation=DERIVATION,scope='CPU canonical full four-molecule fixed identity final color; not shape/SEARCH/FIELD')


def native_molecular_color(repeats=32):
    from roshambo3.backend import cpu
    panel,strata=molecular_model(repeats)
    scores,census=cpu.score_pairs(**panel,steps=1,lr_q=0,lr_t=0,start_mode=0,
        budget_override='canonical molecular color aggregate arithmetic control',
        color=True,color_mode='combo_optimal',mixing=.5,threads=1,with_census=True)
    np.testing.assert_array_equal(scores[:,9:13],np.tile([1,0,0,0],(len(scores),1)))
    np.testing.assert_array_equal(scores[:,13:16],np.zeros((len(scores),3)))
    issued=sum(s['issued_terms_per_row']*repeats for s in strata)
    assert census['color_per_start_final_terms']==issued,(census,issued)
    assert census['color_per_start_final_evaluations']==len(scores)
    return scores[:,4],dict(expected_final_terms=issued,
        native_final_terms=census['color_per_start_final_terms'],
        native_final_evaluations=census['color_per_start_final_evaluations'])

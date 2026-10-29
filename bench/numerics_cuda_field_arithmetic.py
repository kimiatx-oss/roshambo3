"""Actual CUDA Gaussian builders and exact-vertex sampler interval controls.

The full3D residual is characterization only: an unknown hardware combined-weight
model must not silently become a fitted tolerance or production SEARCH coverage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from roshambo3.reference.field_intervals import gaussian_prefilter_intervals,assert_interval,convex_filter_intervals,CUDA_LINEAR_FILTER_SOURCE
from bench.numerics_cuda_texture import tensor_model,residual_report


def panels():
    rng=np.random.default_rng(274)
    query=np.ones((5,4),np.float32)
    query[:,:3]=rng.uniform(-1,1,(5,3))
    yield 'shape',query,True,1.
    # Typed-color planes: each target type has its own Gaussian width and
    # source amplitude; production builds exactly one such plane per type.
    for target,factor,weights in ((1,.8,[1.,.6,.9]),(2,1.3,[.6,.9,1.])):
        color=query[:3].copy(); color[:,3]=np.array(weights,np.float32)
        yield 'color_target_'+str(target),color,False,factor


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rows',type=Path,help='temporary observations for independent review; never expected fixtures')
    ap.add_argument('--require-mean-only',action='store_true',help='require unchanged broad rows pass and focused node mean rejects')
    ap.add_argument('--expect-sampler-rejection',action='store_true')
    ap.add_argument('--expect-node-rejection',action='store_true',help='native plant run; keep the clean intervals unchanged')
    args=ap.parse_args()
    if args.output.exists():ap.error('output must be fresh')
    import roshambo3
    from bench.run_numerics_checks import select_native_build
    select_native_build(roshambo3,args.build.resolve())
    from bench.gpu_lock import GpuLock,resolve_device,query_gpu_state
    device=resolve_device(0)
    with GpuLock('numerics-Gaussian-field-intervals',device) as lock:
        state=query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct>0:raise RuntimeError('GPU busy before field diagnostic')
        from roshambo3 import _roshambo3_cuda as ext
        from bench.numerics_backends import verified_build
        stamp=verified_build('cuda')
        if Path(ext.__file__).resolve().parent!=args.build.resolve():
            raise RuntimeError('extension did not load from requested build')
        from bench.numerics_backends import sampler_panel,check_sampler
        ramp,ramp_points=sampler_panel()
        ramp_origin=np.array([2,-3,4],np.float32);ramp_spacing=np.float32(.5)
        ramp_physical=(ramp_origin+ramp_spacing*ramp_points).astype(np.float32)
        sampler_error=None
        try: check_sampler(ext.numerics_sample_field(ramp.view(np.uint16),ramp_physical,ramp_origin,ramp_spacing),ramp_points)
        except AssertionError as error: sampler_error=str(error)
        if args.expect_sampler_rejection!=(sampler_error is not None):
            raise AssertionError('unexpected sampler detector outcome: '+str(sampler_error))
        origin=np.array([-1]*3,np.float32);spacing=np.float32(.25);cells=np.array([8]*3)
        points=np.random.default_rng(275).integers(1,2048,(512,3)).astype(np.float32)/256
        vertices=np.array([[x,y,z] for z in range(9) for y in range(9) for x in range(9)],np.float32)
        report=dict(scope='seeded Gaussian builder means, full3D convex range rows and physical-to-lattice sampling; SEARCH not qualified',
            production_search_qualified=False,full3D_model='conservative documented convex range; exact combined weights/value rounding unresolved',
            lock_held=lock.acquired,device=ext.device_info(0),preflight=state.as_dict(),build=stamp,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            interval_sha256=hashlib.sha256(Path('src/roshambo3/reference/field_intervals.py').read_bytes()).hexdigest(),
            extension_path=str(Path(ext.__file__).resolve()),
            sampler_precision=dict(expected_rejection=args.expect_sampler_rejection,detector=sampler_error,origin=ramp_origin.tolist(),spacing=float(ramp_spacing)),
            convexity_source=CUDA_LINEAR_FILTER_SOURCE,
            extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),panels={})
        rejected=0
        observations={}
        for name,query,shape,factor in panels():
            expected=gaussian_prefilter_intervals(query,origin,spacing,cells,shape=shape,factor=factor)
            nodebits=ext.numerics_build_field(query,origin,float(spacing),cells.tolist(),shape,factor)
            nodes=nodebits.view(np.float16)
            try:
                checked=assert_interval(nodes,expected)
                node_status='passed'
            except AssertionError as error:
                if not args.expect_node_rejection:raise
                checked=dict(detector=str(error));node_status='rejected';rejected+=1
            # Invocation control: execute the same native builder on a known
            # 1e-4 coordinate-scale perturbation, keeping every clean interval
            # unchanged. This is not claimed as a native source fault plant.
            shifted=query.copy();shifted[:,:3]*=np.float32(1.0001)
            shifted_bits=ext.numerics_build_field(shifted,origin,float(spacing),cells.tolist(),shape,factor)
            shifted_nodes=shifted_bits.view(np.float16).astype(float)
            outside=(shifted_nodes<expected.lo)|(shifted_nodes>expected.hi)
            if not args.expect_node_rejection and not np.any(outside):
                raise AssertionError('Gaussian invocation perturbation was not detected')
            signed_lower=expected.lo.mean(axis=(0,1,2));signed_upper=expected.hi.mean(axis=(0,1,2))
            shifted_mean=shifted_nodes.mean(axis=(0,1,2))
            invocation=dict(scope='native invocation control; not a native source plant',
                coordinate_multiplier=1.0001,unchanged_bounds=True,
                outside_elements=int(np.count_nonzero(outside)),
                signed_mean_detected=((shifted_mean<signed_lower)|(shifted_mean>signed_upper)).tolist())
            sampled_vertices=ext.numerics_sample_field(nodebits,(origin+spacing*vertices).astype(np.float32),origin,spacing).reshape(nodes.shape)
            # Fetching a vertex should reproduce the actual half4 texel, with
            # no combined fractional-weight assumption or output bit fixture.
            np.testing.assert_array_equal(sampled_vertices,nodes.astype(np.float32))
            physical=(origin+spacing*points).astype(np.float32)
            actual=ext.numerics_sample_field(nodebits,physical,origin,spacing)
            hull=convex_filter_intervals(expected,physical,origin,spacing)
            # Evaluate broad fractional rows independently of focused node
            # rows/means, including on a known defective builder. No assertion
            # ordering or narrower node gate can stand in for this outcome.
            rows_pass=bool(np.all((actual>=hull.lo)&(actual<=hull.hi)))
            node_mean=nodes.astype(float).mean(axis=(0,1,2))
            mean_lower=expected.lo.mean(axis=(0,1,2));mean_upper=expected.hi.mean(axis=(0,1,2))
            mean_pass=bool(np.all((node_mean>=mean_lower)&(node_mean<=mean_upper)))
            vertex_hull=convex_filter_intervals(expected,(origin+spacing*vertices).astype(np.float32),origin,spacing)
            vertex_values=sampled_vertices.reshape(-1,4)
            vertex_rows_pass=bool(np.all((vertex_values>=vertex_hull.lo)&(vertex_values<=vertex_hull.hi)))
            # The focused mean and this original broad row arm use the SAME
            # 729 lattice points, with different independently derived bounds.
            bounds_digest=hashlib.sha256()
            for array in (expected.lo,expected.hi,hull.lo,hull.hi,vertex_hull.lo,vertex_hull.hi):
                bounds_digest.update(np.ascontiguousarray(array,dtype='<f8').tobytes())
            hull_report=dict(gate_scope="broad full3D convex-range fractional and same-vertex rows versus focused Gaussian-node means",
                rows_pass=rows_pass,vertex_rows_pass=vertex_rows_pass,mean_pass=mean_pass,
                bounds_sha256=bounds_digest.hexdigest(),
                unchanged_bounds=True,node_mean=node_mean.tolist(),
                node_mean_lower=mean_lower.tolist(),node_mean_upper=mean_upper.tolist(),
                scalar_mean_pass=bool(mean_lower[0]<=node_mean[0]<=mean_upper[0]),
                max_row_width=float(np.max(hull.hi-hull.lo)))
            if not args.expect_node_rejection and not rows_pass:
                raise AssertionError('full3D convex row gate failed')
            if args.require_mean_only and (not rows_pass or not vertex_rows_pass or mean_pass or hull_report['scalar_mean_pass']):
                raise AssertionError('plant did not demonstrate rows_pass=true and scalar mean_pass=false')
            observations.update({name+'_nodes':nodes.astype(float),name+'_node_lower':expected.lo,
                name+'_node_upper':expected.hi,name+'_samples':actual,
                name+'_sample_lower':hull.lo,name+'_sample_upper':hull.hi,
                name+'_vertex_lower':vertex_hull.lo,name+'_vertex_upper':vertex_hull.hi})
            diagnostic=residual_report(actual,tensor_model(nodes,points))
            report['panels'][name]=dict(builder=node_status,intervals=checked,full3D_convex_ranges=hull_report,invocation_control=invocation,vertices='passed',
                vertex_count=len(vertices),full3D_vs_rejected_tensor8=diagnostic,
                source_count=len(query),factor=factor)
        if args.expect_node_rejection and rejected==0:raise AssertionError('native Gaussian plant was not detected')
        report['native_plant_expected']=args.expect_node_rejection
        report['rejected_panels']=rejected
        if args.rows:
            if args.rows.exists():raise RuntimeError('temporary row output must be fresh')
            np.savez(args.rows,**observations)
            report['temporary_rows_sha256']=hashlib.sha256(args.rows.read_bytes()).hexdigest()
            report['temporary_rows_path']=str(args.rows.resolve())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

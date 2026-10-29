"""Recover effective corner coefficients without assuming separable weights."""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
from bench.gpu_lock import GpuLock, resolve_device, query_gpu_state
from bench.numerics_backends import verified_build
from bench.numerics_cuda_texture import tensor_model, texture_panel, residual_report


def allocated_weights(points, order=(2,1,0), tie="even"):
    """Unqualified hypothesis: conserve an integer256 quota at every split."""
    weights={(0,0,0):np.full(len(points),256,dtype=np.float64)}
    for axis in order:
        split={}
        for corner,quota in weights.items():
            raw=quota*(1-points[:,axis])
            low=np.rint(raw) if tie=="even" else np.floor(raw+.5)
            high=list(corner);high[axis]=1
            split[corner]=low
            split[tuple(high)]=quota-low
        weights=split
    return np.stack([weights[c]/256 for c in itertools.product((0,1),repeat=3)],axis=1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('output must be fresh')
    import roshambo3
    roshambo3.__path__.append(str(args.build.resolve()))
    device=resolve_device(0)
    with GpuLock('numerics-texture-basis',device) as lock:
        preflight=query_gpu_state(device)
        if preflight.compute_apps or preflight.utilization_gpu_pct>0:raise RuntimeError('GPU busy')
        from roshambo3 import _roshambo3_cuda as ext
        stamp=verified_build('cuda')
        points=np.random.default_rng(271).integers(1,256,(1024,3)).astype(np.float32)/256
        corners=list(itertools.product((0,1),repeat=3))
        basis=np.zeros((1024,8),np.float64)
        exact=np.zeros_like(basis)
        for start in (0,4):
            nodes=np.zeros((2,2,2,4),np.float16)
            for ch,(x,y,z) in enumerate(corners[start:start+4]):nodes[z,y,x,ch]=1
            basis[:,start:start+4]=ext.numerics_sample_field(nodes.view(np.uint16),points)
            exact[:,start:start+4]=tensor_model(nodes,points)
        reports={}
        for kind in ('ramp','constant','random','positive'):
            nodes,_=texture_panel(kind)
            actual=ext.numerics_sample_field(nodes.view(np.uint16),points)
            vals=np.array([nodes[z,y,x] for x,y,z in corners],np.float64)
            reports[kind]=dict(basis_linear=residual_report(actual,basis@vals), basis_final_half=residual_report(actual,(basis@vals).astype(np.float16).astype(np.float64)))
        allocation_candidates={''.join(map(str,order))+'_'+tie:residual_report(basis,allocated_weights(points,order,tie)) for order in itertools.permutations(range(3)) for tie in ('even','up')}
        record=dict(allocation_candidates=allocation_candidates,scope='diagnostic only; basis model not qualified',device=ext.device_info(0),build=stamp,lock_held=lock.acquired,preflight=preflight.as_dict(),basis_vs_tensor=residual_report(basis,exact),sum_residual=float(np.max(np.abs(basis.sum(axis=1)-1))),lattice={str(bits):int(np.count_nonzero(basis*2**bits!=np.rint(basis*2**bits))) for bits in (8,9,10,12,16,20,24)},panels=reports,examples=[dict(point=points[i].tolist(),basis=basis[i].tolist(),tensor=exact[i].tolist()) for i in range(8)])
        args.output.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()

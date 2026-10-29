"""CUDA half4 filter characterization; candidate models are hypotheses only.

This intentionally does not certify the Gaussian field path. The documented
8-bit axis weights do not specify precision of the internal 3D arithmetic.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np


def texture_panel(kind, count=1024):
    rng = np.random.default_rng(270)
    z, y, x = np.indices((3, 3, 3))
    if kind == "ramp":
        nodes = np.stack((x, y, z, x+2*y+4*z), axis=-1)
    elif kind == "constant":
        nodes = np.broadcast_to([1., 3., -2., .125], (3, 3, 3, 4)).copy()
    elif kind == "random":
        nodes = rng.uniform(-8, 8, (3, 3, 3, 4))
    elif kind == "positive":
        nodes = rng.uniform(.01, 8, (3, 3, 3, 4))
    else:
        raise ValueError(kind)
    # Exact dyadic coordinates avoid the address +0.5 rounding and weight
    # tie-rule ambiguity. Every coordinate is strictly within the grid.
    points = rng.integers(1, 512, (count, 3)).astype(np.float32)/256
    return nodes.astype(np.float16), points


def tensor_model(nodes, points, weight_bits=8, product_bits=None):
    """Border-addressed exact tensor product of quantized per-axis weights.

    product_bits is a diagnostic hypothesis: round each *combined* weight.
    It is not the CUDA documented model and is never an accepted bound.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    points = np.asarray(points, dtype=np.float32)
    # The production sampler adds a half texel in fp32 before the fetch.
    lattice = (points + np.float32(.5)).astype(np.float64)-.5
    cells = np.floor(lattice).astype(int)
    weights = np.rint((lattice-cells)*2**weight_bits)/2**weight_bits
    out = np.zeros((len(points), 4), np.float64)
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        indices = cells + [dx, dy, dz]
        valid = np.all((indices >= 0) & (indices < np.array(nodes.shape[:3])[::-1]), axis=1)
        w = np.prod(np.where([dx, dy, dz], weights, 1-weights), axis=1)
        if product_bits is not None:
            w = np.rint(w*2**product_bits)/2**product_bits
        ix, iy, iz = indices[valid].T
        out[valid] += w[valid, None]*nodes[iz, iy, ix]
    return out


def staged_model(nodes, points, order=(0, 1, 2), half_stages=False):
    """Hypothesis: three nested lerps, optionally rounded to half each stage."""
    nodes = np.asarray(nodes, np.float64)
    cells = np.floor(points).astype(int)
    weights = np.rint((points-cells)*256)/256
    values = {}
    for corner in itertools.product((0, 1), repeat=3):
        ix, iy, iz = (cells + corner).T
        values[corner] = nodes[iz, iy, ix]
    for axis in order:
        reduced = {}
        for corner in values:
            if corner[axis]:
                continue
            other = list(corner); other[axis] = 1
            value = ((1-weights[:,axis,None])*values[corner] +
                     weights[:,axis,None]*values[tuple(other)])
            reduced[corner] = value.astype(np.float16).astype(np.float64) if half_stages else value
        values = reduced
    return values[(0, 0, 0)]


def residual_report(actual, expected):
    error = np.asarray(actual, np.float64)-expected
    return dict(max_abs_by_channel=np.max(np.abs(error), axis=0).tolist(),
                mean_by_channel=error.mean(axis=0).tolist(),
                mismatching_channels=int(np.count_nonzero(error)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be fresh")
    import roshambo3
    roshambo3.__path__.append(str(args.build.resolve()))
    from bench.gpu_lock import GpuLock, resolve_device, query_gpu_state
    device = resolve_device(0)
    with GpuLock("numerics-texture", device) as lock:
        state = query_gpu_state(device)
        if state.compute_apps or state.utilization_gpu_pct > 0:
            raise RuntimeError("GPU busy before diagnostic: "+str(state))
        from roshambo3 import _roshambo3_cuda as ext
        from bench.numerics_backends import verified_build, sampler_panel, check_sampler
        build = verified_build("cuda")
        axis_nodes, axis_points = sampler_panel()
        report = dict(scope="diagnostic; no full3D or Gaussian field certification",
                      device=ext.device_info(0), build=build, lock_held=lock.acquired,
                      preflight=state.as_dict(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      axis=check_sampler(ext.numerics_sample_field(axis_nodes.view(np.uint16), axis_points), axis_points),
                      panels={})
        for kind in ("ramp", "constant", "random", "positive"):
            nodes, points = texture_panel(kind)
            actual = ext.numerics_sample_field(nodes.view(np.uint16), points)
            expected = tensor_model(nodes, points)
            item = dict(tensor8=residual_report(actual, expected),
                        final_half=residual_report(actual, expected.astype(np.float16).astype(np.float64)),
                        candidates={})
            for bits in (8, 9, 10, 12, 16):
                item["candidates"]["product"+str(bits)] = residual_report(actual, tensor_model(nodes, points, product_bits=bits))
            for order in itertools.permutations(range(3)):
                item["candidates"]["half_stages_"+"".join(map(str,order))] = residual_report(actual, staged_model(nodes, points, order, True))
            item["vertex"] = residual_report(ext.numerics_sample_field(nodes.view(np.uint16), np.array(list(itertools.product(range(3), repeat=3)), np.float32)), nodes.transpose(2,1,0,3).reshape(-1,4).astype(np.float64))
            report["panels"][kind] = item
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()

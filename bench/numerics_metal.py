"""Numerics Metal adapters. Sampler measurements never qualify Gaussian fields.

Metal has FIELD optimization only; ``poses=`` is its analytic final scorer.
Unknown Apple families have no declared fractional-bit guarantee. Their direct
probes use the conservative convex-interpolation envelope and visibly report
that limitation. A measured weight lattice is evidence, not a fitted bound.
"""
from dataclasses import asdict
import itertools
import numpy as np

from bench.numerics_backends import COLUMNS, PANEL_STAMP, reference_rows
from roshambo3.reference.numerics_contracts import PathArithmetic, assert_residuals


def analytic_contract(panel):
    maximum = max(float(np.max(np.abs(panel[k][:, :8, :3])))
                  for k in ("query_coords", "library_coords"))
    return PathArithmetic("metal_precise_final_identity", library_grid=None,
        coordinate_error=4 * 2**-24 * maximum, exp2_relative_error=2**-21,
        argument_roundings=16, coefficient_roundings=8,
        operation_model_derivation=("physical fp32 input; identity pose; reduced-scale "
            "multiplication and reconstruction <=4u per coordinate; <=16 exponent "
            "and <=8 coefficient roundings; precise exp2 conservative two ULP; "
            "four-lane positive sum bounded by serial pair count"))


def fixed(panel, context):
    from roshambo3.backend import metal
    from roshambo3.layout import tile32
    args = dict(panel, library_coords=tile32(panel["library_coords"]),
                library_color_offset=5, color=True, mixing=.5)
    poses = np.zeros((len(panel["pair_query_row"]), 7), np.float32)
    poses[:, 0] = 1
    return metal.score_pairs(**args, context=context, poses=poses)


def check_fixed(panel, scores):
    contract = analytic_contract(panel)
    expected, bounds = reference_rows(panel, contract)
    np.testing.assert_array_equal(scores[:, 9:13], np.tile([1, 0, 0, 0], (len(scores), 1)))
    np.testing.assert_array_equal(scores[:, 13:16], np.zeros((len(scores), 3)))
    observed = [dict(zip(COLUMNS, map(float, row[:9]))) for row in scores]
    return assert_residuals(observed, expected, bounds, contract=contract, certification=True)


def stamp(observation, context, panel, **evidence):
    from roshambo3.backend import metal
    # Metal kernels are compiled from package source at context creation.
    # Preserve those actual source hashes rather than inventing a CUDA stamp.
    observation.stamp(build=dict(metal.build_stamp(), device=context.info),
        arithmetic=asdict(analytic_contract(panel)),
        evidence=dict(panel=PANEL_STAMP, rows=len(panel["pair_query_row"]),
                      qualification_status="execution only; native mutation qualification pending", **evidence))


def sampler_panel():
    """Noncubic, signed four-channel nodes; exact and off-lattice positions."""
    rng = np.random.default_rng(270)
    nodes = rng.uniform(-2, 3, (5, 4, 3, 4)).astype(np.float32)
    # Include all centers, boundaries/skirt, far outside, and 3D interiors.
    centers = np.array(list(itertools.product(range(3), range(4), range(5))), np.float32)
    points = np.concatenate((centers, rng.uniform([.05, .05, .05], [1.95, 2.95, 3.95], (257, 3)),
        [[-.25, 1, 1], [2.5, 1, 1], [-1, 1, 1], [3, 1, 1], [1e30, 0, 0]]))
    return nodes, points.astype(np.float32)


def interpolate(nodes, points, *, sampler="manual", weight_bits=None):
    """Independent fp64 trilinear model, including distinct production borders."""
    nodes = np.asarray(nodes, np.float64)
    dims = np.array(nodes.shape[2::-1])
    out = np.zeros((len(points), 4), np.float64)
    for i, p in enumerate(np.asarray(points, np.float64)):
        if not np.all(np.isfinite(p)):
            continue
        if sampler == "manual":
            if np.any(p < 0) or np.any(p >= dims-1):
                continue
        elif np.any(p <= -1) or np.any(p >= dims):
            continue
        lo = np.floor(p).astype(int)
        w = p-lo
        if weight_bits is not None:
            w = np.rint(w * 2**weight_bits) / 2**weight_bits
        for x, y, z in itertools.product((0, 1), repeat=3):
            ix = lo + [x, y, z]
            if np.all(ix >= 0) and np.all(ix < dims):
                out[i] += nodes[ix[2], ix[1], ix[0]] * np.prod(np.where([x, y, z], w, 1-w))
    return out


def _half_conversion_error(nodes):
    """Bound FP32-to-finite-FP16 conversion by its adjacent representables.

    Metal's documented ``rte`` and ``rtz`` texture-write modes can select
    either adjacent finite binary16 value around an FP32 input. The default
    ``native`` mode does not declare which strategy the hardware uses, so the
    bound admits either endpoint instead of assuming NumPy's nearest-even
    conversion. Values outside the finite binary16 range are rejected because
    their conversion may overflow to infinity rather than either neighbor.
    """
    values = np.asarray(nodes, np.float64)
    max_finite = float(np.finfo(np.float16).max)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > max_finite):
        raise ValueError("texture16 rounding bound requires finite in-range FP32 texels")

    nearest = values.astype(np.float16)
    nearest64 = nearest.astype(np.float64)
    with np.errstate(over="ignore"):
        below = np.nextafter(nearest, np.full_like(nearest, -np.inf))
        above = np.nextafter(nearest, np.full_like(nearest, np.inf))
    lower = np.where(nearest64 <= values, nearest, below).astype(np.float64)
    upper = np.where(nearest64 >= values, nearest, above).astype(np.float64)
    return np.maximum(values - lower, upper - values)


def sampler_bound(nodes, *, sampler, weight_bits=None):
    """Absolute per-channel arithmetic envelope, derived before observations.

    Trilinear interpolation is convex; each axis has Lipschitz constant at
    most max(node)-min(node), including zero border. A nearest weight lattice
    changes each weight by <=2^(-bits-1), yielding 3*span*delta. Unknown
    precision uses delta=1, the complete convex envelope (explicitly weak).
    Half conversion adds the farther distance to the two adjacent finite
    binary16 values around each FP32 texel. This covers the documented
    nearest-even and toward-zero modes without declaring the default native
    hardware strategy; convex weights sum to <=1. FP32 coordinate
    normalization, products and eight-term accumulation add gamma(64)*maxabs;
    grid extent here is <=5, so normalized-coordinate reconstruction
    contributes <=16u.
    """
    nodes = np.asarray(nodes, np.float64)
    magnitude = np.max(np.abs(nodes), axis=(0, 1, 2))
    span = np.maximum(0, nodes.max(axis=(0, 1, 2))) - np.minimum(0, nodes.min(axis=(0, 1, 2)))
    texel = np.max(_half_conversion_error(nodes), axis=(0, 1, 2)) if sampler == "texture16" else 0
    delta = 0 if sampler == "manual" else (1 if weight_bits is None else 2.**(-weight_bits-1))
    u = 2**-24
    return texel + 3*span*(delta + 16*u) + 64*u/(1-64*u)*magnitude


def check_sampler(nodes, points, samples, *, sampler, weight_bits=None):
    expected = interpolate(nodes, points, sampler=sampler)
    bound = sampler_bound(nodes, sampler=sampler, weight_bits=weight_bits)
    errors = np.asarray(samples, float)-expected
    assert np.all(np.abs(errors) <= bound), "sampler interpolation arithmetic"
    assert np.all(np.abs(errors.mean(axis=0)) <= bound), "sampler mean bias"
    # Exact texel centers and completely out-of-range points remain strong
    # checks even when a family's filtering precision is undeclared.
    p = np.asarray(points)
    exact = np.all(p == np.floor(p), axis=1)
    exact |= np.any((p <= -1) | (p >= np.array(nodes.shape[2::-1])), axis=1)
    center_bound = sampler_bound(nodes, sampler=sampler, weight_bits=24)
    assert np.all(np.abs(errors[exact]) <= center_bound), "sampler center/border"
    return dict(max_error=np.max(np.abs(errors), axis=0).tolist(), bound=bound.tolist(),
        fractional_weight_bits=weight_bits,
        texture_write_rounding=(
            "native default; exact hardware strategy undeclared"
            if sampler == "texture16" else "not applicable"),
        precision_status="undeclared family: conservative convex bound" if weight_bits is None and sampler != "manual" else "declared arithmetic")


def weight_sweep(context, sampler):
    """Measure candidate nearest-weight lattices; never silently assume CUDA8.

    Exactly representable ramps isolate one weight at a time. Report all
    matching candidates, or none if the device uses a different rule. The
    result is not promoted to an arithmetic guarantee by this function.
    """
    z, y, x = np.indices((3, 3, 3))
    nodes = np.stack((x, y, z, x+2*y+4*z), axis=-1).astype(np.float32)
    p = np.zeros((1023, 3), np.float32)
    p[:, 0] = np.arange(1, 1024, dtype=np.float32)/1024 + 2**-15
    actual = probe(context, sampler, nodes, p)
    candidates = []
    for bits in range(1, 17):
        expected = interpolate(nodes, p, sampler=sampler, weight_bits=bits)
        if np.max(np.abs(actual-expected)) <= 64*2**-24*8:
            candidates.append(bits)
    return dict(nearest_weight_bit_candidates=candidates, measured_points=len(p),
                guarantee="measurement only; no family declaration inferred")


def probe(context, sampler, nodes, points):
    from roshambo3.backend import metal
    nodes = np.asarray(nodes, np.float32)
    records = np.zeros((1, 8), np.uint32)
    records[0, :4] = np.array([0, 0, 0, 1], np.float32).view(np.uint32)
    records[0, 4:7] = nodes.shape[2::-1]
    pos = np.zeros((len(points), 4), np.float32); pos[:, :3] = points
    grid = context.put("t270_grid", records)
    field = context.put("t270_field", nodes)
    positions = context.put("t270_positions", pos)
    out = context.buffer("t270_output", len(points)*16)
    if sampler == "manual":
        context.native.submit([("probe_field", [positions, grid, field, out], metal._u(0, len(points)), len(points), 64)])
    else:
        texture = context.native.texture(*map(int, nodes.shape[2::-1]), sampler == "texture16")
        error = context.put("t270_error", metal._u(0))
        context.native.submit([
            ("pack_field", [grid, field, error], metal._u(0, sampler == "texture16"), int(np.prod(nodes.shape[:3])), 64, [texture]),
            ("probe_texture", [positions, grid, out], metal._u(0, len(points)), len(points), 64, [texture])])
        assert error.read(np.dtype("uint32"), [1])[0] == 0
    return out.read(np.dtype("float32"), [len(points), 4])


def check_weight_lattice(nodes, points, samples, *, weight_bits):
    """Check a declared ramp lattice, independently of the convex error bound.

    FP32 coordinate conversion, weight formation and interpolation have at
    most 64 roundings here; gamma(64)*maxabs bounds the roundoff. Quantization
    itself is represented exactly in the fp64 oracle. This detects one bit
    coarser weights on a seeded dense sweep. A native call requires a separate
    family declaration; a measured best fit alone does not authorize one.
    """
    expected = interpolate(nodes, points, sampler="texture32", weight_bits=weight_bits)
    u = 2**-24
    bound = 64*u/(1-64*u) * max(1., float(np.max(np.abs(nodes))))
    assert np.all(np.abs(np.asarray(samples)-expected) <= bound), "sampler weight precision"
    return dict(weight_bits=weight_bits, roundoff_bound=bound)


def check_determinism(actual, expected):
    """Same-build dispatch partition is declared bit-identical."""
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),
                                  np.asarray(expected).view(np.uint32),
                                  err_msg="Metal dispatch partition changed output bits")

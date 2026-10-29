"""Seeded numerics backend adapters; all references are evaluated live.

A path name in a receipt is restricted to arithmetic actually exercised. In
particular sampler calibration does not certify the Gaussian field builder,
and an analytic final rescore does not certify its field search.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
from roshambo3.numerics import WIRE_QUANT_STEP

from roshambo3.reference.numerics_contracts import (
    PathArithmetic, fixed_pose_row, residual_bounds, assert_residuals)
from roshambo3.reference.overlap import ALPHA, PAIR_PREFACTOR

COLUMNS = ("tanimoto_combination", "tanimoto_shape", "tanimoto_color",
           "overlap_volume", "overlap_color", "self_overlap_query",
           "self_overlap_library", "self_color_query", "self_color_library")
PANEL_STAMP = "numerics-panel-seed270-count5+3-fp32-v1"
ROOT = Path(__file__).resolve().parents[1]


def seeded_panel(rows=32):
    """Asymmetric typed features, poisoned padding, off-wire atom coordinates."""
    rng = np.random.default_rng(270)
    q = np.zeros((rows, 10, 4), np.float32)
    b = np.zeros_like(q)
    q[:, :8, :3] = rng.normal(0, 1.3, (rows, 8, 3))
    b[:, :8, :3] = rng.normal(0, 1.1, (rows, 8, 3))
    q[:, :8, 3] = b[:, :8, 3] = 1
    q[:, 8:, :3] = 99
    b[:, 8:, :3] = -99
    r = np.zeros((3, 3), np.float32)
    p = np.zeros_like(r)
    r[1:, 1:] = [[.8, 1.3], [1.3, .7]]
    p[1:, 1:] = [[1, .6], [.6, .9]]
    return dict(query_coords=q, query_types=np.tile([0]*5+[1,1,2]+[0]*2, (rows,1)).astype(np.int32),
                query_n_atoms=np.full(rows,5,np.int32), library_coords=b,
                library_counts=np.tile([5,2,1],(rows,1)).astype(np.uint8),
                pair_query_row=np.arange(rows,dtype=np.int32),
                pair_library_row=np.arange(rows,dtype=np.int32),
                interaction_r=r, interaction_p=p)


def cpu_contract(panel):
    # Physical input arrays already carry fp32 input conversion. The scalar
    # combo final scorer multiplies query and posed library by sqrt(alpha/2ln2),
    # then subtracts, squares and sums in fp32; exp2f and serial accumulation.
    # Identity rotation and zero translation are required by this adapter.
    # Scale multiplication rounding + stored-scale error are each <= u*|x|;
    # returning to physical units gives 2*u*max|x| per coordinate.
    maximum = max(float(np.max(np.abs(panel[k][:,:8,:3])))
                  for k in ("query_coords", "library_coords"))
    return PathArithmetic("cpu_scalar_fixed_identity_libm", library_grid=None,
        exp2_relative_error=2**-22, coordinate_error=2*2**-24*maximum,
        argument_roundings=16, coefficient_roundings=8,
        operation_model_derivation="physical fp32 inputs; identity pose; reduced scale (two u absolute coordinate error); subtract/square/add/exponent <=16 roundings; coefficient <=8; serial positive sum; libm exp2f <=2 ULP")


def reference_rows(panel, contract):
    expected, bounds = [], []
    for i in range(len(panel["pair_query_row"])):
        qi, bi = int(panel["pair_query_row"][i]), int(panel["pair_library_row"][i])
        args = dict(query_color=panel["query_coords"][qi,5:8,:3],query_types=[1,1,2],
            library_color=panel["library_coords"][bi,5:8,:3],library_types=[1,1,2],
            interaction_r=panel["interaction_r"],interaction_p=panel["interaction_p"],mixing=.5)
        a,b=panel["query_coords"][qi,:5,:3],panel["library_coords"][bi,:5,:3]
        expected.append(fixed_pose_row(a,b,contract=contract,**args))
        bounds.append(residual_bounds(a,b,contract=contract,**args))
    return expected,bounds


def cpu_fixed(panel, *, threads=1, executor=None):
    from roshambo3.backend import cpu
    return cpu.score_pairs(**panel, steps=1, lr_q=0, lr_t=0,
        budget_override="numerics identity fixed-pose diagnostic", start_mode=0,
        color=True, color_mode="combo_optimal", mixing=.5,
        threads=threads, executor=executor)


def check_cpu_fixed(panel, scores):
    # Verify the independently intended pose, not merely whatever pose bytes
    # the backend returns. That prevents prepared-frame errors being hidden.
    np.testing.assert_array_equal(scores[:,9:13],np.tile([1,0,0,0],(len(scores),1)))
    np.testing.assert_array_equal(scores[:,13:16],np.zeros((len(scores),3)))
    contract=cpu_contract(panel)
    expected,bounds=reference_rows(panel,contract)
    observed=[dict(zip(COLUMNS,map(float,row[:9]))) for row in scores]
    report=assert_residuals(observed,expected,bounds,contract=contract,certification=True)
    return report


def verified_build(backend):
    """Refuse a stale extension by content, independent of checkout location."""
    from scripts.source_stamp import source_content_hash
    if backend=="cpu":
        from roshambo3.backend import cpu
        stamp=cpu.build_stamp()
    elif backend=="metal":
        from roshambo3.backend import metal
        stamp=metal.build_stamp()
        stamp["metal_src_sha256"]=stamp["compiled_src_sha256"]
    else:
        from roshambo3 import get_build_info
        stamp=vars(get_build_info())
    key=backend+"_src_sha256"
    assert stamp[key]==source_content_hash(ROOT/"src/roshambo3/backend"/("_"+backend+"_src")), "loaded extension source differs from tested checkout"
    return stamp


def stamp_cpu(observation, panel, report=None, **extra):
    observation.stamp(build=verified_build("cpu"), arithmetic=asdict(cpu_contract(panel)),
        evidence=dict(panel=PANEL_STAMP, rows=len(panel["pair_query_row"]),residuals=report or {},**extra))


def cuda_shape_overlap(extension, a, b):
    """Direct production volume_single diagnostic, no optimizer or wire grid."""
    a=np.asarray(a,np.float32); b=np.asarray(b,np.float32)
    qa=np.ones((1,len(a),4),np.float32); qb=np.ones((1,len(b),4),np.float32)
    qa[0,:,:3]=a; qb[0,:,:3]=b
    out=np.empty((1,1),np.float32)
    extension.test_overlap(qa,qb,out)
    return float(out[0,0])


def sampler_panel():
    # Dyadic half4 ramps isolate weights from texel quantization. One axis
    # varies at a time: multidimensional hardware filter arithmetic needs a
    # separate qualification, and is deliberately not inferred from this probe.
    z,y,x=np.indices((3,3,3))
    nodes=np.stack((x,y,z,x+2*y+4*z),axis=-1).astype(np.float16)
    rng=np.random.default_rng(270)
    points=np.zeros((129,3),np.float32)
    points[np.arange(len(points)),np.arange(len(points))%3]=rng.uniform(.05,1.95,len(points))
    return nodes,points


def ramp_model(points, weight_bits=8):
    points=np.asarray(points,dtype=np.float64)
    cell=np.floor(points); w=points-cell
    w=np.rint(w*2**weight_bits)/2**weight_bits
    a=cell+w
    return np.column_stack((a,a[:,0]+2*a[:,1]+4*a[:,2]))


def check_sampler(samples, points, weight_bits=8):
    expected=ramp_model(points,weight_bits)
    # Half texels and fixed-point weights are dyadic, and this ramp's sums
    # are exact at fp32 for the one-axis probe; allow 8*u final rounding.
    # This contract does not cover simultaneous fractional x/y/z filtering.
    bound=8*2**-24*np.maximum(1,np.abs(expected))
    difference=np.asarray(samples)-expected
    assert np.all(np.abs(difference)<=bound), "sampler weight precision differs from declared fixed-point model"
    assert np.all(np.abs(difference.mean(axis=0))<=bound.mean(axis=0)), "sampler mean bias"
    # Measure uniqueness rather than accepting the conservative width alone.
    assert not np.allclose(samples,ramp_model(points,weight_bits-1),rtol=0,atol=bound.max()), "sampler precision probe does not distinguish a one-bit truncation"
    return {"fractional_weight_bits":weight_bits,"max_residual":float(np.max(np.abs(difference))),"mean_residual":difference.mean(axis=0).tolist()}


def cuda_contract():
    from roshambo3.reference.numerics_contracts import CoordinateGrid
    return PathArithmetic("cuda_analytic_t106_fixed_identity",query_grid=CoordinateGrid(),
        query_color_grid=CoordinateGrid(),library_grid=None,
        operation_model_derivation="physical fp32 library; nearest-even int16 query shape and color then separate wire-step dequantization and reduced scale; identity pose; <=16 exponent operations, <=8 coefficient operations; ex2.approx two ULP; serial positive atom-pair reduction")


def cuda_fixed(panel, *, pose_search=None, **options):
    from roshambo3.backend import cuda
    from roshambo3.layout import tile32
    args=dict(panel)
    args["library_coords"]=tile32(panel["library_coords"],planar=options.get("library_tile_mode",1)==-1)
    return cuda.score_pairs(**args,steps=0,start_mode=0,color=True,mixing=.5,
        pose_search=cuda.PoseSearch.ANALYTIC if pose_search is None else pose_search,
        library_color_offset=-1,**options)


def check_cuda_fixed(panel,scores):
    contract=cuda_contract()
    expected,bounds=reference_rows(panel,contract)
    observed=[dict(zip(COLUMNS,map(float,row[:9]))) for row in scores]
    return assert_residuals(observed,expected,bounds,contract=contract,certification=True)


def polynomial_relative_bound():
    # Declared degree-five approximation coefficients, not backend outputs.
    # Bound on [-1/2,1/2]: distance from Taylor polynomial plus Lagrange
    # remainder e^(ln2/2)*(ln2/2)^6/6!, divided by min 2^f=1/sqrt2.
    c=[1.,.693147182,.240226507,.0555051871,.00961799621,.00133978436]
    error=sum(abs(np.float32(v).item()-math.log(2)**i/math.factorial(i))*.5**i
              for i,v in enumerate(c))
    error+=math.sqrt(2)*(math.log(2)/2)**6/math.factorial(6)
    # Horner 5 FMA + scale multiply, coefficient and argument errors outside.
    return math.sqrt(2)*error+8*2**-24


def cuda_wire_inputs(panel, *, planar=False):
    from roshambo3.layout import tile32
    args=dict(panel)
    query=np.zeros_like(panel["query_coords"])
    query[:,:8,:3]=np.rint(panel["query_coords"][:,:8,:3]/WIRE_QUANT_STEP)*WIRE_QUANT_STEP
    query[:,:8,3]=1
    library=np.zeros(panel["library_coords"].shape[:2]+(3,),np.int16)
    library[:,:8]=np.rint(panel["library_coords"][:,:8,:3]/WIRE_QUANT_STEP).astype(np.int16)
    args["query_coords"]=query
    args["library_coords"]=tile32(library,planar=planar)
    return args


def cuda_field_contract():
    from dataclasses import replace
    from roshambo3.reference.numerics_contracts import CoordinateGrid
    return replace(cuda_contract(),name="cuda_field_exact_final_wire_identity",
        library_grid=CoordinateGrid(),library_color_grid=CoordinateGrid(),
        operation_model_derivation="wire fp32 query and int16 library; separate wire-step dequantization and reduced scale; identity intended pose; production field winner-only analytic final ex2 evaluation; <=16 exponent and8 coefficient roundings; serial positive atom sum; sampler calibration is separately required for field fixed-pose certification")


def cpu_evaluator_capability(backend, grain, width, *, spacing=.25):
    """Ask the production factory for support; never infer it from host names.

    Analytic atom-grain evaluators currently have an AVX2 implementation only.
    Field width8 is a scheduling group; its native stamp may describe NEON
    four-lane instructions. An unavailable analytic factory is not a fallback.
    """
    from roshambo3.backend import cpu
    backends={0:"scalar_analytic",1:"analytic_simd",2:"scalar_field",
              3:"paired_channel_field",4:"pipelined_field"}
    grains={0:"scalar",1:"query_atom",2:"library_atom",3:"trajectory"}
    options=dict(shape_backend=backends[backend],shape_grain=grains[grain],shape_width=width)
    if backend>=2:options["field_spacing_angstrom"]=spacing
    try:
        stamp=cpu.combo_shape_config_stamp(**options)
    except ValueError as error:
        # Only a native missing-implementation refusal is an unavailable arm;
        # bad widths/configurations and unexpected errors remain failures.
        if "unavailable AVX2 target" not in str(error):
            raise
        return dict(available=False,reason=str(error),requested=options)
    return dict(available=True,stamp=stamp,requested=options)


def cpu_fixed_color_selection():
    """Choose an actually compiled retained grain and its supported start mode.

    Grain11 is retained on AVX2; grain16's native implementation can use NEON
    and requires four starts. The scalar grain remains explicit if neither
    native retained vector arm is supported. Callers verify census selection.
    """
    from roshambo3.backend import cpu
    for grain,start_mode in ((11,0),(16,1),(0,0)):
        stamp=cpu.color_build_stamp(grain)
        if (stamp.get("color_compiled",False) and stamp.get("color_target_compiled",False)
                and stamp.get("color_runtime_available",False)):
            return dict(grain=grain,start_mode=start_mode,stamp=stamp)
    raise RuntimeError("no compiled and runtime-available CPU color grain")

"""Executed backend checks; unsupported/unqualified arms stay visible."""
import numpy as np
import pytest

from bench.numerics_backends import (seeded_panel,cpu_fixed,check_cpu_fixed,
    stamp_cpu,verified_build,cuda_shape_overlap,sampler_panel,check_sampler,ramp_model)
from roshambo3.backend import cpu
from roshambo3.reference.numerics_contracts import PathArithmetic, residual_bounds
from roshambo3.reference.overlap import pair_overlap


def require_cpu():
    if not cpu.available():
        pytest.skip("CPU extension unavailable; seeded CPU checks not run")


@pytest.mark.numerics_coverage(backend="cpu",path="scalar",check="fixed_pose",channel="shape")
@pytest.mark.numerics_coverage(backend="cpu",path="scalar",check="fixed_pose",channel="color")
@pytest.mark.numerics_coverage(backend="cpu",path="scalar",check="frame",channel="shape")
@pytest.mark.numerics_coverage(backend="cpu",path="scalar",check="frame",channel="color")
def test_cpu_scalar_seeded_fixed_pose_and_same_frame(numerics_coverage):
    require_cpu()
    panel=seeded_panel()
    scores=cpu_fixed(panel)
    report=check_cpu_fixed(panel,scores)
    stamp_cpu(numerics_coverage,panel,report,qualification_status="execution only; native plants required")


@pytest.mark.numerics_coverage(backend="cpu",path="scalar",check="determinism",channel="shape")
@pytest.mark.numerics_coverage(backend="cpu",path="scalar",check="determinism",channel="color")
def test_cpu_scalar_workers_and_claim_chunking(numerics_coverage):
    require_cpu()
    panel=seeded_panel(528)
    expected=cpu_fixed(panel)
    configurations=[]
    for workers in (1,2,4):
        for chunk in (128,256,512):
            with cpu.CPUComboExecutor(workers,claim_chunk_rows=chunk) as executor:
                actual=cpu_fixed(panel,threads=workers,executor=executor)
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
            configurations.append([workers,chunk])
    stamp_cpu(numerics_coverage,panel,axes=configurations,
              qualification_status="fixed-pose scheduling only; optimizer trajectories not certified")


def cuda_extension():
    try:
        from roshambo3 import _roshambo3_cuda as ext
    except ImportError:
        pytest.skip("CUDA extension unavailable")
    try:
        if ext.device_count()<1:
            pytest.skip("CUDA device unavailable")
    except RuntimeError as error:
        pytest.skip("CUDA runtime unavailable: "+str(error))
    return ext


def test_cuda_direct_shape_fixed_pose(numerics_coverage):
    ext=cuda_extension()
    panel=seeded_panel()
    contract=PathArithmetic("cuda_direct_volume_single",library_grid=None,
        operation_model_derivation="test_overlap invokes production volume_single: physical fp32 input; reduced-distance scaling; <=16 exponent roundings, <=8 coefficient roundings; exp2f two ULP; serial atom-pair sum")
    observed=[]; expected=[]; bounds=[]
    for a,b in zip(panel["query_coords"][:,:5,:3],panel["library_coords"][:,:5,:3]):
        observed.append({"overlap_volume":cuda_shape_overlap(ext,a,b)})
        expected.append({"overlap_volume":pair_overlap(a,b)})
        bounds.append({"overlap_volume":residual_bounds(a,b,contract=contract)["overlap_volume"]})
    from roshambo3.reference.numerics_contracts import assert_residuals
    from dataclasses import asdict
    report=assert_residuals(observed,expected,bounds,contract=contract,certification=True)
    build=verified_build("cuda")
    build["device"]=ext.device_info(0)
    numerics_coverage.stamp(build=build,arithmetic=asdict(contract),evidence=dict(
        rows=len(observed),residuals=report,scope="direct shape overlap, no query wire grid, no optimizer; production output frame not certified"))


def test_cuda_production_sampler_measures_eight_fractional_bits():
    ext=cuda_extension()
    if not hasattr(ext,"numerics_sample_field"):
        pytest.skip("CUDA direct sampler diagnostic absent in this build")
    verified_build("cuda")
    nodes,points=sampler_panel()
    samples=ext.numerics_sample_field(nodes.view(np.uint16),points)
    report=check_sampler(samples,points)
    assert report["fractional_weight_bits"]==8
    # A direct texture probe alone does not qualify a Gaussian field path;
    # field fixed_pose receipts deliberately remain not_run.


def test_sampler_detector_rejects_one_bit_coarser_weights():
    _,points=sampler_panel()
    check_sampler(ramp_model(points,8),points)
    with pytest.raises(AssertionError,match="sampler weight precision"):
        check_sampler(ramp_model(points,7),points)


def test_cpu_frame_detector_rejects_recomputed_self_in_different_frame():
    require_cpu()
    panel=seeded_panel()
    scores=cpu_fixed(panel)
    split=seeded_panel()
    # Source-side invocation plant: recompute all production self terms after
    # rounding query coordinates to a 0.1 A grid, leave original cross term.
    split["query_coords"][:,:5,:3]=np.rint(split["query_coords"][:,:5,:3]*10)/10
    alternate=cpu_fixed(split)
    scores[:,5]=alternate[:,5]
    with pytest.raises(AssertionError,match="self_overlap_query"):
        check_cpu_fixed(panel,scores)


def test_cpu_direct_color_field_samples_with_derived_interpolation_bound():
    require_cpu()
    from roshambo3.reference.overlap import color_overlap,COLOR_PREFACTOR
    panel=seeded_panel(1)
    query=panel["query_coords"][0,5:8,:3].copy()
    types=np.array([1,1,2],np.int32)
    r,p=panel["interaction_r"],panel["interaction_p"]
    spacing=.25
    # This release has no ColorFieldSet (that experiment stays in the paper
    # repository); exercise the retained production bundle instead, with
    # physical Gaussian parameters. Color Gaussian widths are in Angstrom.
    coefficients=np.zeros_like(r,dtype=np.float64)
    positive=r>0
    coefficients[positive]=COLOR_PREFACTOR*p[positive].astype(float)**2/r[positive].astype(float)**1.5
    decays=-r.astype(float)/2
    codes=np.array([0,1,1],np.int32)
    field=cpu.extension().ColorFieldBundle(query,types,coefficients,decays,codes,spacing,4.)
    points=np.random.default_rng(271).uniform(-1,1,(48,3)).astype(np.float32)
    for target in (1,2):
        actual=field.sample(target,points)["value"]
        expected=np.array([color_overlap(query,types,z[None,:],[target],r,p) for z in points])
        # A positive Gaussian of width r has |d²f/dx²| <= r*amplitude.
        # Successive convex 1D interpolation contributes <= h²/8 on each
        # of 3 axes. Node fp32 conversion and interpolation add <=32*u.
        widths=r[types,target].astype(float)
        amplitudes=COLOR_PREFACTOR*p[types,target].astype(float)**2/widths**1.5
        bound=3*spacing**2/8*np.sum(widths*amplitudes)+32*2**-24*np.sum(amplitudes)
        assert np.all(np.abs(actual-expected)<=bound)
        assert abs(np.mean(actual-expected))<=bound
    # This direct production bundle sampler check is execution evidence only;
    # field production-path and weight fault qualification stay not_run.


@pytest.mark.numerics_coverage(backend="cuda",path="analytic",check="fixed_pose",channel="shape")
@pytest.mark.numerics_coverage(backend="cuda",path="analytic",check="fixed_pose",channel="color")
@pytest.mark.numerics_coverage(backend="cuda",path="analytic",check="frame",channel="shape")
@pytest.mark.numerics_coverage(backend="cuda",path="analytic",check="frame",channel="color")
def test_cuda_complete_fixed_pose_same_frame(numerics_coverage):
    ext=cuda_extension()
    from bench.numerics_backends import cuda_fixed,check_cuda_fixed,cuda_contract
    from dataclasses import asdict
    panel=seeded_panel()
    report=check_cuda_fixed(panel,cuda_fixed(panel))
    build=verified_build("cuda");build["device"]=ext.device_info(0)
    numerics_coverage.stamp(build=build,arithmetic=asdict(cuda_contract()),
                            evidence=dict(residuals=report,rows=32))


@pytest.mark.parametrize("configuration",[
    pytest.param((1,1,8),marks=pytest.mark.numerics_coverage(backend="cpu",path="vector",check="fixed_pose",channel="shape")),
    pytest.param((1,2,8),marks=pytest.mark.numerics_coverage(backend="cpu",path="vector",check="fixed_pose",channel="shape")),
    (2,0,1),(3,2,8),(4,2,8)],
                         ids=["query-atom","library-atom","scalar-field","paired-field","pipeline-field"])
def test_cpu_direct_production_shape_evaluators(configuration,numerics_coverage):
    require_cpu()
    if not hasattr(cpu.extension(),"numerics_shape_evaluate"):
        pytest.skip("direct CPU shape evaluator diagnostic unavailable")
    from dataclasses import replace
    from bench.numerics_backends import cpu_contract,polynomial_relative_bound
    from roshambo3.reference.overlap import ALPHA,PAIR_PREFACTOR
    panel=seeded_panel(16)
    backend,grain,width=configuration
    from bench.numerics_backends import cpu_evaluator_capability
    capability=cpu_evaluator_capability(backend,grain,width)
    if not capability["available"]:
        pytest.skip("production shape evaluator unavailable: "+capability["reason"])
    contract=replace(cpu_contract(panel),name="cpu_vector_declared_d5",
                     exp2_relative_error=polynomial_relative_bound())
    residuals=[];limits=[]
    for a,b in zip(panel["query_coords"][:,:5,:3],panel["library_coords"][:,:5,:3]):
        result=cpu.extension().numerics_shape_evaluate(a.copy(),b.copy(),backend,grain,width,.25)
        expected=pair_overlap(a,b)
        arithmetic=residual_bounds(a,b,contract=contract)["overlap_volume"]
        if backend>=2:
            assert result["field_samples"]==len(b)
            assert result["fallback_evaluations"]==0
            # |f_xx|<=alpha*amplitude on each axis. Trilinear interpolation
            # <=3*h²/8, Laplacian prefilter <=3*h²/12; together 5*h²/8.
            bound=arithmetic+5*.25**2/8*ALPHA*PAIR_PREFACTOR*len(a)*len(b)
        else:
            assert result["backend"]=="analytic_simd"
            assert result["exponential"]=="minimax_d5"
            bound=arithmetic+PAIR_PREFACTOR*len(a)*len(b)*2**-24
        residuals.append(result["overlap_volume"]-expected);limits.append(bound)
    assert np.all(np.abs(residuals)<=limits)
    assert abs(np.mean(residuals))<=np.mean(limits)
    if backend==1:
        numerics_coverage.stamp(build=verified_build("cpu"),
            arithmetic=dict(family="CPU analytic atom grains",configs=[[1,1,8],[1,2,8]],
                            exp_relative_bound=polynomial_relative_bound(),
                            operation_model_derivation=contract.operation_model_derivation),
            evidence=dict(selected=capability["stamp"],rows=len(residuals),
                          residual_mean=float(np.mean(residuals))))


@pytest.mark.parametrize("shape_backend,shape_grain",[("analytic_simd","query_atom"),
    ("analytic_simd","library_atom"),("scalar_field","scalar"),
    ("paired_channel_field","library_atom"),("pipelined_field","library_atom")])
def test_cpu_canonical_vector_color_and_field_final_frame(shape_backend,shape_grain):
    require_cpu()
    from bench.numerics_backends import cpu_evaluator_capability,cpu_fixed_color_selection
    backend_codes={"analytic_simd":1,"scalar_field":2,"paired_channel_field":3,"pipelined_field":4}
    grain_codes={"scalar":0,"query_atom":1,"library_atom":2}
    width=1 if shape_backend=="scalar_field" else 8
    capability=cpu_evaluator_capability(backend_codes[shape_backend],grain_codes[shape_grain],width)
    if not capability["available"]:
        pytest.skip("production shape evaluator unavailable: "+capability["reason"])
    selected=cpu_fixed_color_selection()
    panel=seeded_panel(16)
    if selected["start_mode"]==1:
        # The retained four-start grain has no identity-only mode. Identical
        # asymmetric conformers independently force the identity winner,
        # while all four native start evaluations still execute.
        panel["library_coords"]=panel["query_coords"].copy()
    panel["interaction_r"][1:,1:]=1
    panel["interaction_p"][1:,1:]=np.eye(2,dtype=np.float32)
    options=dict(shape_backend=shape_backend,shape_grain=shape_grain,
        shape_width=width)
    if "field" in shape_backend:options["shape_field_spacing_angstrom"]=.25
    scores,census=cpu.score_pairs(**panel,steps=1,lr_q=0,lr_t=0,
        budget_override="T270 native path fixed pose",start_mode=selected["start_mode"],color=True,
        color_mode="combo_optimal",mixing=.5,color_grain=selected["grain"],with_census=True,**options)
    assert census["color_grain"]==selected["grain"]
    starts=4 if selected["start_mode"]==1 else 1
    if "field" in shape_backend:assert census["shape_field_samples"]==16*5*starts
    check_cpu_fixed(panel,scores)
    # Exact final columns are a frame check; direct evaluator tests above
    # exercise search arithmetic, so final rescoring cannot conceal a sampler.


@pytest.mark.numerics_coverage(backend="cuda",path="field",check="frame",channel="shape")
@pytest.mark.numerics_coverage(backend="cuda",path="field",check="frame",channel="color")
def test_cuda_field_complete_final_same_frame(numerics_coverage):
    ext=cuda_extension()
    from roshambo3.backend import cuda
    from bench.numerics_backends import cuda_wire_inputs,cuda_field_contract,reference_rows
    from dataclasses import asdict
    from roshambo3.reference.numerics_contracts import assert_residuals
    panel=seeded_panel(16)
    panel["interaction_r"][1:,1:]=1
    panel["interaction_p"][1:,1:]=np.eye(2,dtype=np.float32)
    scores=cuda.score_pairs(**cuda_wire_inputs(panel),steps=0,start_mode=0,color=True,
        mixing=.5,pose_search=cuda.PoseSearch.FIELD,library_color_offset=-1)
    expected,bounds=reference_rows(panel,cuda_field_contract())
    observed=[dict(zip(("tanimoto_combination","tanimoto_shape","tanimoto_color", "overlap_volume","overlap_color","self_overlap_query","self_overlap_library","self_color_query","self_color_library"),map(float,z[:9]))) for z in scores]
    report=assert_residuals(observed,expected,bounds,contract=cuda_field_contract(),certification=True)
    build=verified_build("cuda");build["device"]=ext.device_info(0)
    numerics_coverage.stamp(build=build,arithmetic=asdict(cuda_field_contract()),
        evidence=dict(residuals=report,scope="exact final score frame; field sampler qualification separate"))


@pytest.mark.numerics_coverage(backend="cuda",path="field",check="determinism",channel="shape")
@pytest.mark.numerics_coverage(backend="cuda",path="field",check="determinism",channel="color")
def test_cuda_field_optimizer_supported_scheduling(numerics_coverage):
    ext=cuda_extension()
    from roshambo3.backend import cuda
    from bench.numerics_backends import cuda_wire_inputs,cuda_field_contract
    from dataclasses import asdict
    panel=seeded_panel(16)
    panel["interaction_r"][1:,1:]=1
    panel["interaction_p"][1:,1:]=np.eye(2,dtype=np.float32)
    common=dict(steps=4,start_mode=1,color=True,mixing=.5,pose_search=cuda.PoseSearch.FIELD,
                color_placement=cuda.ColorPlacement.IN_GRADIENT,library_color_offset=-1)
    args=cuda_wire_inputs(panel)
    expected=cuda.score_pairs(**args,**common)
    axes=[]
    # SHARED is the one coordinate residency tier the build compiles.
    for residency in (cuda.CoordResidency.SHARED,):
        for blocks in (2,4):
            actual=cuda.score_pairs(**args,**common,coord_residency=residency,min_blocks_per_sm=blocks)
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
            axes.append([int(residency),blocks])
    # Launch coalescing: identical complete input records, same indexed pair
    # rows, split into independently launched groups and reassembled by row.
    split=[]
    for lo,hi in ((0,3),(3,10),(10,16)):
        portion=dict(args)
        portion["pair_query_row"]=args["pair_query_row"][lo:hi].copy()
        portion["pair_library_row"]=args["pair_library_row"][lo:hi].copy()
        split.append(cuda.score_pairs(**portion,**common))
    np.testing.assert_array_equal(np.concatenate(split).view(np.uint32),expected.view(np.uint32))
    build=verified_build("cuda");build["device"]=ext.device_info(0)
    numerics_coverage.stamp(build=build,arithmetic=asdict(cuda_field_contract()),
        evidence=dict(steps=4,axes=axes,coalescing=[3,7,6],device_count=ext.device_count(),
                      scope="one available GPU; cross-device-count coverage remains absent"))


@pytest.mark.parametrize("backend,shape_grain_id,width",[(2,0,1),(3,2,8),(4,2,8)])
def test_cpu_field_lattice_nodes_against_fp64_prefilter(backend,shape_grain_id,width):
    require_cpu()
    from roshambo3.reference.overlap import ALPHA,PAIR_PREFACTOR
    import math
    scale=math.sqrt(ALPHA/(2*math.log(2)))
    panel=seeded_panel(8)
    for query in panel["query_coords"][:,:5,:3]:
        query=query.copy()
        geometry=cpu.extension().numerics_shape_evaluate(query,query[:1].copy(),backend,shape_grain_id,width,.25)
        origin=np.array(geometry["grid_origin_reduced"],float)
        step=float(geometry["grid_spacing_reduced"])
        # Choose interior lattice nodes near atoms, then request the actual
        # production sampler there. This removes the O(h²) interpolant budget.
        indices=np.rint((query.astype(float)*scale-origin)/step).astype(int)
        nodes=(origin+indices*step)/scale
        library=nodes.astype(np.float32)
        d2=np.sum((query.astype(float)[:,None,:]-nodes[None,:,:])**2,axis=-1)
        h=step/scale
        terms=np.exp(-ALPHA*d2/2)
        laplacian=(ALPHA**2*d2-3*ALPHA)*terms
        expected=PAIR_PREFACTOR*np.sum(terms-h*h/12*laplacian)
        actual=cpu.extension().numerics_shape_evaluate(query,library,backend,shape_grain_id,width,.25)
        assert actual["field_samples"]==len(library)
        assert actual["fallback_evaluations"]==0
        # <=128*u covers reduced-coordinate scaling, stored ln2, exp2,
        # Laplacian formation, prefilter, trilinear endpoint arithmetic and
        # serial accumulation, on this <=25-term panel. Absolute term envelope
        # bounds cancellation in the prefilter; no observed residual enters.
        u=2**-24
        envelope=PAIR_PREFACTOR*(len(query)*len(library))*(1+5*ALPHA*h*h/12)
        rounding=(128*u)/(1-128*u)*envelope
        # Rounded physical sample points can sit a few u off a node. Gaussian
        # slope <=sqrt(alpha), prefilter slope <=5*alpha^1.5*h²/12;
        # sum the absolute movement in each physical coordinate.
        eps=8*u*max(1,float(np.max(np.abs(library))))
        coordinate=PAIR_PREFACTOR*len(query)*len(library)*3*eps*(math.sqrt(ALPHA)+5*ALPHA**1.5*h*h/12)
        assert abs(actual["overlap_volume"]-expected)<=rounding+coordinate

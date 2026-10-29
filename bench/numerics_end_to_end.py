"""Real LIT-PCBA public preparation, scoring and exported-pose checks.

Execution receipts are deliberately not native fault qualification. Interface
faults exercise public seams, and never claim to prove a native mutation caught.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from roshambo3.numerics import WIRE_QUANT_STEP

from bench.numerics_fixtures import DATA, load_panel, oracle_uncertainty
from roshambo3.reference.color import score_columns
from roshambo3.reference.numerics_contracts import PathArithmetic, residual_bounds

# Design requirement: optimizer displacement <= 0.01 A RMS, plus independent
# nearest-wire error for both sides. This is an acceptance requirement, not an
# error model inferred from production endpoints or from the observed basin gap.
OPTIMIZER_RMSD_REQUIREMENT = .01
WIRE_DISPLACEMENT = np.sqrt(3.) * float(WIRE_QUANT_STEP) / 2
# Keep the preregistered optimizer recovery gate fixed across wire steps.
RECOVERY_WIRE_ALLOWANCE = np.sqrt(3.) * .0005
POSE_RMSD_LIMIT = OPTIMIZER_RMSD_REQUIREMENT + 2 * RECOVERY_WIRE_ALLOWANCE


def transforms():
    from scipy.spatial.transform import Rotation
    return {
        "identity": (np.eye(3), np.zeros(3)),
        "grid": (np.diag([-1., -1., 1.]), np.array([.127, -.213, .319])),
        "arbitrary": (Rotation.from_rotvec([.31, -.23, .17]).as_matrix(),
                      np.array([.1732, -.2194, .1371])),
    }


@dataclass
class PreparedPanel:
    corpus: object
    generator: object
    records: dict
    base_gids: list
    fixture_names: list
    source_errors: dict


def prepare_panel(directory):
    """Prepare actual source molecules, including raw rigid transforms.

    Raw transforms can disappear under principal-axis preparation. Queries are
    therefore transformed *again after readback*, to exercise a nonidentity
    physical optimizer problem rather than just testing canonicalization.
    """
    from rdkit import Chem
    from roshambo3.prep import prepare
    from roshambo3.pharmacophore import PharmacophoreGenerator

    manifest = json.loads((DATA / "manifest.json").read_text())
    load_panel()  # Validate all source hashes before preparation.
    from roshambo3.prep.sources import canonical_heavy
    mols = []
    orders = {}
    for entry in manifest["molecules"]:
        source = next(iter(Chem.SDMolSupplier(str(DATA / entry["file"]), removeHs=False)))
        # The builder stores heavy atoms in canonical SMILES order; the
        # fixture's are in source order.
        orders[entry["name"]] = canonical_heavy(source)[2]
        for label, (rotation, translation) in transforms().items():
            mol = Chem.Mol(source)
            mol.SetProp("_Name", entry["name"] + "__" + label)
            xyz = np.asarray(source.GetConformer().GetPositions()) @ rotation.T + translation
            for atom, point in enumerate(xyz):
                mol.GetConformer().SetAtomPosition(atom, point)
            mols.append(mol)
    generator = PharmacophoreGenerator()
    prepare(mols, str(directory), workers=1, records_per_unit=5,
            color_generator=generator)
    # Read back through the format-3 reader, as a consumer would:
    # reopened, scanned, every block decoded with the codec dispatch, each
    # row's pseudo-atoms derived from its stored heavy atoms (FORMAT.md 8.3)
    # in the kernel layout's feature-table order (8.4).
    from roshambo3.format.r3g.reader import open_corpus
    from roshambo3.format.r3g.reference import derive_positions, molecule_features
    corpus = open_corpus(str(directory))  # Exercise reopening, not builder objects.
    records = {}
    row_mol_id = {}
    gid_base = 0
    for gi in range(len(corpus.group_table)):
        with corpus.group(gi) as g:
            heavy = g.decode_all()
            n_conf = g.molecule_n_conf.astype(np.int64)
            run_start = np.concatenate([[0], np.cumsum(n_conf)])
            tables = {k: np.asarray(getattr(g, k)) for k in (
                "feature_offsets", "feature_types", "feature_atom_offsets", "feature_atoms")}
            for m in range(g.n_mol):
                feats = molecule_features(tables, m)
                types = np.asarray([t for t, _ in feats], dtype=np.int64)
                nr = int(g.molecule_n_real[m])
                for row in range(int(run_start[m]), int(run_start[m + 1])):
                    xyz = heavy[row]
                    records[gid_base + row] = (
                        xyz[:, :nr].T.astype(np.float32) * WIRE_QUANT_STEP,
                        derive_positions(xyz, feats).astype(np.float32) * WIRE_QUANT_STEP,
                        types)
                    row_mol_id[gid_base + row] = int(g.mol_id[m])
            gid_base += g.n_conf
    ext_of = {}
    for table in corpus.molecules(columns=["mol_id", "ext_id"]):
        ext_of.update(zip(table.column("mol_id").to_pylist(), table.column("ext_id").to_pylist()))
    lookup = {ext_of[row_mol_id[gid]]: gid for gid in records}
    fixture_names = [entry["name"] for entry in manifest["molecules"]]
    base = [lookup[name + "__identity"] for name in fixture_names]
    assert len(base) == 4 and len(records) == 12
    # Independent source-to-wire semantic check. Fit a proper rigid transform
    # using heavy atoms only; feature centroids must follow that same frame.
    raw = {fixture.name: fixture for fixture in load_panel()}
    source_errors = {}
    for name, gid in lookup.items():
        fixture = raw[name.rsplit("__", 1)[0]]
        fixture_shape = fixture.shape[orders[name.rsplit("__", 1)[0]]]
        shape, color, types = records[gid]
        np.testing.assert_array_equal(types, fixture.types,
                                      err_msg="feature assignment/count changed during prepare")
        assert shape.shape == fixture_shape.shape
        centered = shape.astype(np.float64) - shape.mean(axis=0, dtype=np.float64)
        u, _, vt = np.linalg.svd(fixture_shape.T @ centered)
        fix = np.eye(3); fix[2,2] = np.linalg.det(u @ vt)
        rotation = u @ fix @ vt
        translation = shape.mean(axis=0, dtype=np.float64)
        shape_error = np.linalg.norm(fixture_shape @ rotation + translation - shape, axis=1)
        # The corpus stores each molecule's features in FORMAT.md 8.1 rule-4
        # order, (type, atoms), not RDKit's enumeration order, so
        # features are paired within each type by minimum total distance.
        from scipy.optimize import linear_sum_assignment
        moved = fixture.color @ rotation + translation
        color_error = np.zeros(len(types))
        for t in np.unique(types):
            ours, theirs = np.flatnonzero(types == t), np.flatnonzero(fixture.types == t)
            cost = np.linalg.norm(moved[theirs][:, None, :] - color[ours][None, :, :], axis=2)
            ri, ci = linear_sum_assignment(cost)
            color_error[ours[ci]] = cost[ri, ci]
        # Nearest grid plus propagated independent-fit error. Kabsch minimizes
        # squared error; max point fit error is bounded by sqrt(n)*wire_error;
        # float32 centroid/preparation inputs add 64*u times source extent.
        # A pseudo-atom is derived from its stored atoms (integer mean, half
        # away from zero: FORMAT.md 8.3), which adds at most half a step per
        # axis to the wire error of its atoms: one more WIRE_DISPLACEMENT.
        extent = max(1., np.max(np.abs(fixture.shape)), np.max(np.abs(fixture.color)))
        semantic_budget = (1 + np.sqrt(len(shape))) * WIRE_DISPLACEMENT + 64*2**-24*extent
        assert shape_error.max() <= semantic_budget
        assert color_error.max() <= semantic_budget + WIRE_DISPLACEMENT
        source_errors[gid] = dict(shape=float(shape_error.max()), color=float(color_error.max()),
                                  semantic_budget=float(semantic_budget))
    return PreparedPanel(corpus, generator, records, base, fixture_names, source_errors)


def prepared_queries(panel, transform, backend, *, include_color=True):
    from roshambo3.engine import from_arrays
    rotation, translation = transforms()[transform]
    shapes, colors = [], []
    for gid in panel.base_gids:
        shape, color, types = panel.records[gid]
        shapes.append(shape.astype(np.float64) @ rotation.T + translation)
        colors.append((color.astype(np.float64) @ rotation.T + translation, types))
    return from_arrays(shapes, panel.fixture_names, color=colors if include_color else None,
                       color_generator=panel.generator if include_color else None, backend=backend)


def screen_panel(panel, *, backend="cpu", path="analytic", mode="shape",
                 transform="arbitrary", starts=1, batch=5):
    import roshambo3 as r3
    from roshambo3.backend.spec import PoseSearch, ColorPlacement, CoordResidency
    if backend == "cpu" and mode != "shape":
        path = "field"
    if backend == "metal" and path not in ("manual", "texture16", "texture32"):
        raise ValueError("Metal end-to-end requires an explicit FIELD sampler")
    qs = prepared_queries(panel, transform, backend, include_color=mode != "shape")
    # Residency/staging describe CUDA launch resources. Metal has its own
    # sampler selection and deliberately rejects those CUDA-only config knobs.
    knobs = (dict(color_stage=0, color_residency=CoordResidency.GLOBAL)
             if backend == "cuda" else {})
    config = r3.ScreenConfig(pose_search=PoseSearch.ANALYTIC if path == "analytic" else PoseSearch.FIELD,
        color=mode != "shape", mixing=.5 if mode == "combo" else 0.,
        color_placement=ColorPlacement.IN_GRADIENT if mode == "combo" else ColorPlacement.AT_POSE,
        start_mode=starts, steps=200, **knobs,
        budget_override="numerics strict real-pose convergence requirement")
    options = dict(k=len(panel.records), config=config, backend=backend, workers=1,
                   launch_conformers=batch, color_generator=panel.generator)
    if backend == "metal":
        from roshambo3.engine import Engine
        engine = Engine(backend="metal", backend_options=dict(
            field_sampler=path, color_field_sampler=path, color_field_min_sources=2))
        try:
            result = r3.screen(qs, panel.corpus, engine=engine, **options)
        finally:
            engine.close()
    else:
        result = r3.screen(qs, panel.corpus, **options)
    return qs, result


def apply_reported_pose(points, row):
    """Independent quaternion-to-matrix calculation (SciPy, physical units)."""
    from scipy.spatial.transform import Rotation
    def matrix(quat):
        w, x, y, z = np.asarray(quat, np.float64)
        return Rotation.from_quat([x, y, z, w]).as_matrix()
    return (np.asarray(points, np.float64) @ matrix(row[9:13]).T + row[13:16]) @ matrix(row[16:20]).T


def arithmetic_contract(qs, panel, result):
    """Operation-count envelope for public analytic final scoring.

    Max absolute coordinates + translation bound scale/rotation construction:
    64*u*M per component covers normalized quaternion formation, two 3x3
    rotations, reduced-coordinate scale and inverse, translation and output
    float32 storage (each <= 16 operations). Scalar/vector/CUDA final positive
    sums use <=n*m serial roundings; exp2 error <=2^-21. This bound does not
    certify any field optimizer or hardware sampler.
    """
    maximum = max(1., float(np.max(np.abs(qs.coords[..., :3]))),
                  *(float(np.max(np.abs(r[:2][0]))) for r in panel.records.values()))
    maximum += max(float(np.max(np.abs(hit.scores[:,13:16]))) for hit in result)
    return PathArithmetic("public_final_analytic_physical_pose", coordinate_error=64*2**-24*maximum,
        exp2_relative_error=2**-21, argument_roundings=16, coefficient_roundings=8,
        operation_model_derivation=arithmetic_contract.__doc__)


def validate_result(panel, qs, result, *, mode="shape", transform="arbitrary", receipts=None):
    """Check execution on the actual geometry; report fixture transfer separately.

    A conservative source-to-prepared bound cannot establish an optimizer loss.
    Recovery, rows, feature semantics and export checks still assert as before.
    Call require_fixture_qualification on this report when a uniqueness claim is
    needed; routine execution receipts never certify that stronger claim.
    """
    from roshambo3 import scores as S
    contract = arithmetic_contract(qs, panel, result)
    measurements = []
    for qi, hit in enumerate(result):
        assert len(hit) == len(panel.records), "lost/duplicated result rows"
        assert set(map(int, hit.gids)) == set(panel.records), "gid association"
        exported = result.aligned_poses(query=qi)
        nr = int(qs.n_real[qi]); mask = qs.types[qi] > 0
        a = qs.coords[qi,:nr,:3].astype(np.float64)
        ac, at = qs.coords[qi,mask,:3].astype(np.float64), qs.types[qi,mask]
        if mode == "shape":
            from roshambo3.engine.query import snap_to_wire_grid
            rotation, translation = transforms()[transform]
            _, reference_color, at = panel.records[panel.base_gids[qi]]
            ac = snap_to_wire_grid(reference_color.astype(np.float64) @ rotation.T + translation)
        for i, (gid, row) in enumerate(zip(hit.gids, hit.scores)):
            b, bc, bt = panel.records[int(gid)]
            moved = apply_reported_pose(b, row)
            np.testing.assert_allclose(exported[i], moved, atol=2e-12, rtol=0,
                                       err_msg="public exported pose disagrees with physical transform")
            moved_color = apply_reported_pose(bc, row)
            args = dict(query_color=ac if mode != "shape" else (),
                query_types=at if mode != "shape" else (),
                library_color=moved_color if mode != "shape" else (),
                library_types=bt if mode != "shape" else (),
                interaction_r=qs.interaction_r, interaction_p=qs.interaction_p,
                mixing=.5 if mode == "combo" else 0.)
            expected = score_columns(a, moved, args["query_color"], args["query_types"],
                args["library_color"], args["library_types"], qs.interaction_r, qs.interaction_p, args["mixing"])
            bound = residual_bounds(a, moved, contract=contract, **args)
            for name, column in S.COLUMN_INDEX.items():
                assert abs(float(row[column])-expected[name]) <= bound[name], (
                    qi, int(gid), name, float(row[column]), expected[name], bound[name])
            if int(gid) == panel.base_gids[qi]:
                heavy_rmsd = float(np.sqrt(np.mean(np.sum((exported[i]-a)**2, axis=1))))
                color_rmsd = float(np.sqrt(np.mean(np.sum((moved_color-ac)**2, axis=1))))
                assert heavy_rmsd <= POSE_RMSD_LIMIT, ("heavy RMSD", heavy_rmsd, POSE_RMSD_LIMIT)
                assert color_rmsd <= POSE_RMSD_LIMIT, ("color RMSD", color_rmsd, POSE_RMSD_LIMIT)
                gap = qualify_scored_representation(panel, qi, mode, contract)
                measurements.append(dict(name=panel.fixture_names[qi], gap=gap, heavy_rmsd=heavy_rmsd,
                    color_rmsd=color_rmsd, score_allowance=bound["tanimoto_combination"]))
    return dict(measurements=measurements, arithmetic=asdict(contract),
                fixture_qualification={
                    "all_production_qualified": all(m["gap"]["production_qualified"] for m in measurements),
                    "scope": "source-geometry basin-bound transfer; independent prepared-geometry search not run",
                    "cases": [{"name": m["name"], **m["gap"]} for m in measurements]},
                qualification="execution only; native fault plants required", transform=transform)


def qualify_scored_representation(panel, qi, mode, contract):
    """Uniform Gaussian Lipschitz + arithmetic allowance, independent of gaps.

    For exp(-a*d²/2), sup norm of its spatial derivative is sqrt(a/e).
    Moving either endpoint by delta changes each term by at most this times
    the sum of endpoint displacements. Uniform exponent rounding contributes
    gamma16/e because sup x*exp(-x)=1/e; serial positive sums contribute
    gamma(n²). Divide by the actual independent fp64 source self-overlap S:
    for self-matching clouds d=AA+BB-AB>=S, AB<=S, and perturbing all three
    terms by E gives |delta T| <=4E/(S-3E). No observed score residual or basin
    gap enters E. Query snapping and source->prepared representation movement
    are included as coordinate displacement. Finite multistart gaps remain
    empirical evidence; this cannot prove global uniqueness.
    """
    from roshambo3.reference.overlap import (ALPHA, PAIR_PREFACTOR, COLOR_PREFACTOR,
                                            self_overlap, color_overlap)
    fixture = load_panel()[qi]
    gid = panel.base_gids[qi]
    u = 2**-24
    def gamma(n):
        return n*u/(1-n*u)
    # The already-derived posed-coordinate model applies to both endpoints.
    def channel(points, alpha, prefactor, self_value, discrepancy, types=None):
        if types is None:
            terms = len(points)**2
        else:
            _, counts = np.unique(types, return_counts=True)
            terms = int(np.sum(counts**2))
        radius = max(np.linalg.norm(fixture.shape, axis=1).max(), np.linalg.norm(fixture.color, axis=1).max())
        uniform_coordinate_error = 64*u*(radius+16.)
        delta = 2*discrepancy + WIRE_DISPLACEMENT + 2*np.sqrt(3)*uniform_coordinate_error
        term_error = np.sqrt(alpha/np.e)*delta + gamma(16)/(np.e*(1-gamma(16))) + 2**-21
        per_term = (1+term_error)*(1+gamma(8))*(1+gamma(terms))-1
        error = terms * prefactor * per_term
        assert self_value > 3*error, "representation bound crosses denominator"
        return float(4*error/(self_value-3*error))
    shape = channel(fixture.shape, ALPHA, PAIR_PREFACTOR, self_overlap(fixture.shape),
                    panel.source_errors[gid]["shape"])
    color = 0.
    if mode != "shape":
        r, p = panel.generator.interaction_matrix_r, panel.generator.interaction_matrix_p
        cs = color_overlap(fixture.color, fixture.types, fixture.color, fixture.types, r, p)
        color = channel(fixture.color, 1., COLOR_PREFACTOR, cs,
                        panel.source_errors[gid]["color"], fixture.types)
    allowance = .5*(shape+color) if mode == "combo" else shape
    objective = "combo" if mode == "combo" else "shape"
    from bench.numerics_fixtures import load_qualification
    receipt = load_qualification(fixture, objective)
    assert receipt["qualified"] and receipt["complete"]
    oracle = oracle_uncertainty(fixture, objective)
    required_gap = receipt["minimum_gap_design"] + 2*(allowance+oracle)
    transferred = receipt["outside_gap"] > required_gap
    return dict(outside_gap=receipt["outside_gap"], representation_allowance=allowance,
                minimum_gap_design=receipt["minimum_gap_design"], required_gap=required_gap,
                oracle_uncertainty=oracle, production_qualified=bool(transferred),
                status="qualified" if transferred else "unqualified",
                reason=("source bound transfers" if transferred else
                        "source bound does not establish prepared-geometry basin separation; optimizer loss unobserved"),
                limitation="empirical finite search; global uniqueness unproved")


def require_fixture_qualification(report):
    """Fail closed for the stronger fixture-uniqueness obligation, separately."""
    qualification = report.get("fixture_qualification", {})
    cases = qualification.get("cases", [])
    assert cases and all(case.get("production_qualified") is True for case in cases), (
        "prepared fixture qualification unavailable", qualification)



def real_color_gradient(panel):
    """Probe production scalar color derivative off optimum against fp64 FD.

    This exposes the retained scalar evaluator; it does not certify the
    distinct field/retained-vector optimizer gradient implementation.
    """
    from roshambo3.backend import cpu
    from roshambo3.reference.color import color_overlap_gradient
    from roshambo3.reference.overlap import COLOR_PREFACTOR, ALPHA
    qs = prepared_queries(panel, "arbitrary", "cpu")
    quat = np.array([.97, .11, -.13, .09], np.float32)
    translation = np.array([.21, -.17, .13], np.float32)
    reports = []
    for qi, gid in enumerate(panel.base_gids):
        shape, color, types = panel.records[gid]
        library = np.ones((len(shape)+len(color),4), np.float32)
        library[:len(shape),:3] = shape
        library[len(shape):,:3] = color
        counts = np.bincount(types, minlength=7).astype(np.uint8)
        counts[0] = len(shape)
        value, descent, terms = cpu.extension().scalar_color_simd_probe(
            qs.coords[qi], qs.types[qi], int(qs.n_real[qi]), library, counts,
            qs.interaction_r, qs.interaction_p, quat, translation, 0, True)
        mask = qs.types[qi] > 0
        h = 1e-5
        reference = color_overlap_gradient(qs.coords[qi,mask,:3], qs.types[qi,mask],
            color, types, qs.interaction_r, qs.interaction_p, quat, translation, h=h)
        # Production rotation uses half-angle chart and negative derivative;
        # its translation descent is with respect to reduced coordinates.
        scale = np.sqrt(ALPHA/(2*np.log(2.)))
        expected = np.r_[0., -2*reference[:3], -reference[3:]/scale]
        radius = max(np.linalg.norm(color, axis=1).max(),
                     np.linalg.norm(qs.coords[qi,mask,:3], axis=1).max()) + np.linalg.norm(translation)
        # Three rigid-motion derivatives each bring <=(1+R); derivative of
        # exp(-d²/2) through order three is bounded by32 in these units.
        # Central difference remainder <=h²/6 times that third derivative.
        truncation = terms*COLOR_PREFACTOR*32*(1+radius)**3*h*h/6
        cancellation = terms*COLOR_PREFACTOR*256*np.finfo(float).eps*(1+radius)**2/h
        bound = 2*(truncation+cancellation)  # half-angle chart contributes2
        assert np.linalg.norm(expected[1:]) > 10*bound, "off-optimum probe has insufficient gradient signal"
        np.testing.assert_allclose(descent, expected, atol=bound, rtol=0)
        reports.append(dict(name=panel.fixture_names[qi], value=float(value), terms=int(terms),
            gradient_norm=float(np.linalg.norm(expected[1:])), absolute_bound=float(bound)))
    return reports

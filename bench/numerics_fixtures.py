"""Small real alignment inputs and an independent, empirical basin qualifier.

The fp64 scorer is the existing reference; SciPy's BFGS and Haar starts are
independent of production starts/optimizers. Receipts are evidence, never expected
score fixtures. A finite multistart search cannot prove global uniqueness.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import gzip
import json
import math
from pathlib import Path

import numpy as np

from roshambo3.pharmacophore import WORK_DESCENDING_FEATURE_ORDER
from roshambo3.reference.color import guarded_tanimoto
from roshambo3.reference.overlap import color_overlap, pair_overlap, self_overlap

DATA = Path(__file__).resolve().parents[1] / "tests/data/alignment"
FAMILIES = tuple(name for name, _ in WORK_DESCENDING_FEATURE_ORDER)
RECOVERY_NEIGHBORHOOD = .01 + 2 * np.sqrt(3) * .0005


def protocol(data=DATA):
    return json.loads((data / "protocol.json").read_text())


@dataclass(frozen=True)
class AlignmentFixture:
    name: str
    shape: np.ndarray
    color: np.ndarray
    types: np.ndarray
    feature_atom_ids: tuple[tuple[int, ...], ...] = ()

    @property
    def interaction_r(self):
        return np.diag([0.] + [1.] * 6)

    @property
    def interaction_p(self):
        return self.interaction_r


def canonical_centroid(points):
    """Uniform source-coordinate centroid with deterministic fp64 summation."""
    coordinates = np.asarray(points,dtype=np.float64).reshape(-1,3)
    if not len(coordinates):
        raise ValueError("centroid needs source atoms")
    return np.array([math.fsum(float(x) for x in coordinates[:,axis])/len(coordinates)
                     for axis in range(3)],dtype=np.float64)


def centroid_roundoff_bound(points):
    """Uniform-weight RDKit arithmetic versus fsum centroid, physical Angstrom.

    Normalized weights, weighted products and sequential summation, plus the
    canonical sum/division, use fewer than2*n+8 rounding stages. gamma bounds
    their absolute error by that factor times max source coordinate magnitude.
    This conservative bound derives from operations, not observed deltas.
    """
    coordinates=np.asarray(points,dtype=np.float64).reshape(-1,3)
    u=2.**-53
    stages=2*len(coordinates)+8
    gamma=stages*u/(1-stages*u)
    return gamma*max(1.,float(np.max(np.abs(coordinates))))


def uniform_feature_weights(fdef_text):
    """Parse default selected feature definitions; refuse changed semantics."""
    definitions={}
    for section in fdef_text.split("DefineFeature ")[1:]:
        lines=section.split("EndFeature",1)[0].splitlines()
        name=lines[0].split()[0]
        family=next((line.split()[1] for line in lines if line.strip().startswith("Family ")),None)
        if family not in FAMILIES:
            continue
        weights=next((line.split(None,1)[1] for line in lines if line.strip().startswith("Weights ")),None)
        if weights is None:
            raise ValueError("selected fdef lacks explicit centroid weights")
        values=[float(w) for w in weights.split(",")]
        if not values or not all(np.isfinite(w) and w>0 and w==values[0] for w in values):
            raise ValueError("selected fdef requires nonuniform centroid semantics")
        definitions[name]=len(values)
    return definitions


def from_molecule(mol, name: str) -> AlignmentFixture:
    """Keep explicit source H for feature typing; shape uses heavy atoms only.

    No AddHs, protonation/tautomer change, embedding, or force-field minimization.
    Default uniform-weight features use canonical fsum source-atom centroids;
    live RDKit positions must agree within a separately derived rounding bound.
    """
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    fdef_path=Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    definitions=uniform_feature_weights(fdef_path.read_text())
    factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_path))
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError("source atom coordinates must be finite")
    shape = positions[[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]]
    center = canonical_centroid(shape)
    features = [f for f in factory.GetFeaturesForMol(mol) if f.GetFamily() in FAMILIES]
    features.sort(key=lambda f: FAMILIES.index(f.GetFamily()))
    color=[]
    for feature in features:
        ids=feature.GetAtomIds()
        if definitions.get(feature.GetType()) != len(ids):
            raise ValueError("feature atom IDs differ from uniform fdef weight count")
        source_atoms=positions[list(ids)]
        centroid=canonical_centroid(source_atoms)
        reported=np.asarray(feature.GetPos(),dtype=np.float64)
        if reported.shape != (3,) or not np.isfinite(reported).all() or np.max(np.abs(reported-centroid)) > centroid_roundoff_bound(source_atoms):
            raise ValueError("RDKit feature position differs from canonical centroid beyond fp64 rounding")
        color.append(centroid)
    color=np.asarray(color,dtype=np.float64).reshape(-1,3)
    return AlignmentFixture(name, shape - center, color - center,
                            np.asarray([1 + FAMILIES.index(f.GetFamily()) for f in features], dtype=np.int64),
                            tuple(tuple(f.GetAtomIds()) for f in features))


def load_panel(data: Path = DATA) -> list[AlignmentFixture]:
    from rdkit import Chem

    manifest = json.loads((data / "manifest.json").read_text())
    panel = []
    for entry in manifest["molecules"]:
        path = data / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"fixture checksum mismatch: {path}")
        mols = list(Chem.SDMolSupplier(str(path), removeHs=False))
        if len(mols) != 1 or mols[0] is None:
            raise ValueError(f"expected one sanitized molecule: {path}")
        panel.append(from_molecule(mols[0], entry["name"]))
    return panel


def oracle_uncertainty(fixture: AlignmentFixture, objective: str, *, translation_extent=16.) -> float:
    """Conservative conditional fp64 score envelope, independent of found gaps.

    For the centered cloud radius R and terminal translations <=T, all
    coordinate/rotation/distance operations are bounded in magnitude by
    O((1+R+T)^2). Allocate 256 eps*(1+R+T)^2 absolute error to each Gaussian
    term, including a <=2 ulp libm exponential assumption. Positive n^2-term
    sums add gamma(n^2); self sums >=n give normalized channel error <=n
    times this term envelope. Differentiating x/(a+b-x), x<=a=b, bounds
    score error by 4 eta/(1-3 eta); use twice that for rounding slack.

    This is an explicitly assumed ordinary fp64 operation model, not a
    libm proof, a search-completeness bound, or a production fp32 envelope.
    A terminal translation beyond T must be reported as unqualified.
    """
    if objective not in ("shape", "combo"):
        raise ValueError("objective must be shape or combo")
    eps = np.finfo(np.float64).eps
    n = len(fixture.shape)
    clouds = [fixture.shape]
    if objective == "combo":
        n = max(n, len(fixture.color))
        clouds.append(fixture.color)
    radius = max((float(np.max(np.linalg.norm(c, axis=1))) for c in clouds if len(c)), default=0.)
    eta = n * (256 * (1 + radius + translation_extent) ** 2 + n * n) * eps / (1 - n * n * eps)
    return float(8 * eta / (1 - 3 * eta))


def objective_function(fixture: AlignmentFixture, objective: str):
    """Cached self terms, existing fp64 overlap expressions, physical Angstrom."""
    from scipy.spatial.transform import Rotation

    if objective not in ("shape", "combo"):
        raise ValueError("objective must be shape or combo")
    ss = self_overlap(fixture.shape)
    cs = color_overlap(fixture.color, fixture.types, fixture.color, fixture.types,
                       fixture.interaction_r, fixture.interaction_p) if objective == "combo" else 0.

    def value(pose):
        rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
        moved = fixture.shape @ rotation.T + pose[3:]
        shape_score = guarded_tanimoto(pair_overlap(fixture.shape, moved), ss, ss)
        if objective == "shape":
            return shape_score
        moved_color = fixture.color @ rotation.T + pose[3:]
        cross = color_overlap(fixture.color, fixture.types, moved_color, fixture.types,
                              fixture.interaction_r, fixture.interaction_p)
        return .5 * (shape_score + guarded_tanimoto(cross, cs, cs))

    return value


def central_gradient(value, pose, h=1e-5):
    basis = np.eye(6) * h
    return np.array([(value(pose + step) - value(pose - step)) / (2 * h) for step in basis])


def qualify_fixture(fixture: AlignmentFixture, objective="shape", *, starts=128,
                    scored_uncertainty=None) -> dict:
    """Broad reproducible multistart; report unresolved endpoints as unqualified.

    ``scored_uncertainty`` is the caller's derived end-to-end arithmetic and
    representation envelope. Without it this only qualifies the fp64 search;
    ``production_qualified`` stays false and the missing allowance is visible.
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    settings = protocol()
    rng = np.random.default_rng(settings["seed"])
    rotations = Rotation.random(starts, random_state=rng).as_rotvec()
    translations = rng.uniform(-settings["translation_start_angstrom"], settings["translation_start_angstrom"], (starts, 3))
    value = objective_function(fixture, objective)
    uncertainty = oracle_uncertainty(fixture, objective)
    endpoints = []
    for rotation, translation in zip(rotations, translations):
        start = np.concatenate((rotation, translation))
        result = minimize(lambda p: -value(p), start,
                          jac=lambda p: -central_gradient(value, p), method="BFGS",
                          options={"gtol": settings["gtol"], "maxiter": settings["maxiter"]})
        gradient = float(np.max(np.abs(central_gradient(value, result.x))))
        moved = fixture.shape @ Rotation.from_rotvec(result.x[:3]).as_matrix().T + result.x[3:]
        rmsd = float(np.sqrt(np.mean(np.sum((moved - fixture.shape) ** 2, axis=1))))
        endpoints.append({"score": float(value(result.x)), "rmsd": rmsd,
                          "gradient_inf": gradient,
                          "stationary": bool(np.isfinite(gradient) and gradient <= 1e-6),
                          "optimizer_success": bool(result.success), "iterations": int(result.nit),
                          "pose_rotvec": result.x[:3].tolist(), "pose_translation": result.x[3:].tolist(),
                          "translation_in_envelope": bool(np.linalg.norm(result.x[3:]) <= 16.)})
    return summarize_endpoints(fixture, objective, endpoints, starts=starts,
                               scored_uncertainty=scored_uncertainty)


def summarize_endpoints(fixture, objective, endpoints, *, starts=128,
                        scored_uncertainty=None, data=DATA):
    """Recompute every endpoint from bound poses; never trust stored flags/gaps."""
    from scipy.spatial.transform import Rotation
    settings = protocol(data)
    value = objective_function(fixture, objective)
    uncertainty = oracle_uncertainty(fixture, objective)
    radius = settings["pose_neighborhood_rmsd_angstrom"]
    if radius != RECOVERY_NEIGHBORHOOD:
        raise ValueError("protocol recovery neighborhood differs from independently derived requirement")
    evaluated = []
    for endpoint in endpoints:
        pose = np.r_[endpoint["pose_rotvec"], endpoint["pose_translation"]].astype(np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("invalid qualification endpoint")
        moved = fixture.shape @ Rotation.from_rotvec(pose[:3]).as_matrix().T + pose[3:]
        rmsd = float(np.sqrt(np.mean(np.sum((moved - fixture.shape) ** 2, axis=1))))
        gradient = float(np.max(np.abs(central_gradient(value, pose, settings["central_gradient_step"]))))
        evaluated.append(dict(endpoint, score=float(value(pose)), rmsd=rmsd,
            gradient_inf=gradient, stationary=bool(np.isfinite(gradient) and gradient <= settings["stationarity_max_gradient"]),
            translation_in_envelope=bool(np.linalg.norm(pose[3:]) <= settings["translation_extent_angstrom"])))
    inside = [e for e in evaluated if e["rmsd"] <= radius and e["stationary"]]
    outside = [e for e in evaluated if e["rmsd"] > radius and e["stationary"]]
    gap = 1. - max((e["score"] for e in outside), default=0.)
    singular = np.linalg.svd(fixture.shape, compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] else 0.
    complete = len(evaluated) == starts and all(e["stationary"] and e["translation_in_envelope"] for e in evaluated)
    qualified = bool(starts >= settings["starts"] and len(inside) >= 4 and complete
                     and ratio >= settings["nonplanarity_minimum"]
                     and gap > settings["minimum_gap_design"] + 2 * uncertainty)
    production_qualified = bool(qualified and scored_uncertainty is not None
                                and np.isfinite(scored_uncertainty) and scored_uncertainty >= 0
                                and gap > settings["minimum_gap_design"] + 2 * (uncertainty + scored_uncertainty))
    return {"name": fixture.name, "objective": objective, "starts": starts,
            "qualified": qualified, "production_qualified": production_qualified,
            "missing": [] if scored_uncertainty is not None else ["end-to-end scored representation allowance"],
            "oracle_uncertainty": uncertainty, "scored_uncertainty": scored_uncertainty,
            "minimum_gap_design": settings["minimum_gap_design"], "pose_neighborhood_rmsd_angstrom": radius,
            "outside_gap": gap, "inside_count": len(inside), "outside_count": len(outside),
            "complete": complete, "nonplanarity_ratio": ratio, "endpoints": evaluated,
            "scope": "empirical finite search; global uniqueness unproved"}


def fixture_binding(fixture, data=DATA):
    """Bind scientific evidence to the exact current source and live chemistry."""
    from rdkit import RDConfig, rdBase
    def digest_array(array, dtype):
        a = np.asarray(array, dtype=dtype)
        return hashlib.sha256(json.dumps(list(a.shape)).encode() + a.tobytes(order="C")).hexdigest()
    entry = next((e for e in json.loads((data / "manifest.json").read_text())["molecules"]
                  if e["name"] == fixture.name), None)
    if entry is None:
        raise ValueError("qualification requires an attributed source molecule")
    source = data / entry["file"]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != entry["sha256"]:
        raise ValueError("source SDF differs from manifest")
    return {"name":fixture.name, "source_sdf":entry["file"], "source_sdf_sha256":source_hash,
            "source_member_sha256":entry["source_member_sha256"],
            "shape_sha256":digest_array(fixture.shape,"<f8"),
            "color_sha256":digest_array(fixture.color,"<f8"),
            "types_sha256":digest_array(fixture.types,"<i8"),
            "interaction_r_sha256":digest_array(fixture.interaction_r,"<f8"),
            "interaction_p_sha256":digest_array(fixture.interaction_p,"<f8"),
            "feature_atom_ids_sha256":hashlib.sha256(json.dumps(fixture.feature_atom_ids).encode()).hexdigest(),
            "rdkit_version":rdBase.rdkitVersion,
            "fdef_sha256":hashlib.sha256(Path(RDConfig.RDDataDir,"BaseFeatures.fdef").read_bytes()).hexdigest(),
            "protocol_sha256":hashlib.sha256((data / "protocol.json").read_bytes()).hexdigest(),
            "protocol_settings":protocol(data)}


def write_qualification(receipts, data=DATA):
    """Write portable scientific endpoint evidence, never production outputs."""
    import scipy
    import platform
    fixtures = {f.name:f for f in load_panel(data)}
    index = {"execution_environment": {"numpy_version":np.__version__, "scipy_version":scipy.__version__,
                                       "python_version":platform.python_version()}, "kind":"independent fp64 scientific basin qualification; no expected production scores", "receipts":[]}
    for receipt in receipts:
        name, objective = receipt["name"], receipt["objective"]
        endpoints = [{k:e[k] for k in ("pose_rotvec","pose_translation","optimizer_success","iterations")}
                     for e in receipt["endpoints"]]
        payload = gzip.compress(json.dumps(endpoints,separators=(",",":")).encode(),mtime=0)
        filename = f"{name}.{objective}.endpoints.json.gz"
        (data / filename).write_bytes(payload)
        index["receipts"].append({"name":name,"objective":objective,
            "binding":fixture_binding(fixtures[name],data),"endpoints_file":filename,
            "endpoints_sha256":hashlib.sha256(payload).hexdigest(),"endpoint_count":len(endpoints)})
    (data / "qualification.json").write_text(json.dumps(index,indent=2)+"\n")


def load_qualification(fixture, objective, data=DATA):
    """Validate all bindings and independently recompute stationary/basin claims."""
    index = json.loads((data / "qualification.json").read_text())
    matches = [r for r in index["receipts"] if r["name"] == fixture.name and r["objective"] == objective]
    if len(matches) != 1:
        raise ValueError("missing or duplicate qualification receipt")
    receipt = matches[0]
    current_binding = fixture_binding(fixture,data)
    # RDKit release is provenance. Identical actual factory bytes and canonical
    # geometry/feature/matrix content describe the same mathematical problem.
    recorded_semantics = {k:v for k,v in receipt["binding"].items() if k != "rdkit_version"}
    current_semantics = {k:v for k,v in current_binding.items() if k != "rdkit_version"}
    if recorded_semantics != current_semantics:
        raise ValueError("qualification source/geometry/chemistry/protocol binding mismatch")
    path = data / receipt["endpoints_file"]
    if path.parent != data or path.name != receipt["endpoints_file"]:
        raise ValueError("invalid endpoint artifact path")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt["endpoints_sha256"]:
        raise ValueError("qualification endpoint artifact checksum mismatch")
    endpoints = json.loads(gzip.decompress(payload))
    if len(endpoints) != receipt["endpoint_count"] or len(endpoints) != protocol(data)["starts"]:
        raise ValueError("qualification endpoint panel incomplete")
    result = summarize_endpoints(fixture,objective,endpoints,starts=len(endpoints),data=data)
    result["recorded_rdkit_version"] = receipt["binding"]["rdkit_version"]
    result["current_rdkit_version"] = current_binding["rdkit_version"]
    return result


def extract_archive(archive: Path, data: Path = DATA):
    """Reproduce the tiny factual SDF inputs using manifest source members.

    Extraction never runs the production scorer or changes molecular geometry.
    The manifest identifies original bytes and records the feature policy.
    """
    import tarfile
    from rdkit import Chem, RDConfig, rdBase

    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["source"]["sha256"]:
        raise ValueError("source archive checksum mismatch")
    with tarfile.open(archive) as source:
        for entry in manifest["molecules"]:
            raw = source.extractfile(entry["source_member"]).read()
            if hashlib.sha256(raw).hexdigest() != entry["source_member_sha256"]:
                raise ValueError("source member checksum mismatch")
            mol = Chem.MolFromMol2Block(raw.decode(), removeHs=False)
            if mol is None:
                raise ValueError(f"RDKit could not sanitize {entry['source_member']}")
            mol.SetProp("_Name", entry["name"])
            path = data / entry["file"]
            with Chem.SDWriter(str(path)) as writer:
                writer.write(mol)
            restored = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)))
            if not np.array_equal(mol.GetConformer().GetPositions(), restored.GetConformer().GetPositions()):
                raise ValueError("SDF conversion changed source coordinates")
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["feature_policy"]["rdkit_version"] = rdBase.rdkitVersion
    manifest["feature_policy"]["fdef_sha256"] = hashlib.sha256(
        Path(RDConfig.RDDataDir, "BaseFeatures.fdef").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extract", type=Path, help="reproduce SDF inputs from checked upstream archive")
    parser.add_argument("--starts", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--portable", action="store_true", help="write bound compressed endpoint artifacts")
    args = parser.parse_args()
    if args.extract:
        extract_archive(args.extract)
        return
    if args.output is None:
        parser.error("--output is required for qualification")
    import scipy
    import rdkit
    from concurrent.futures import ProcessPoolExecutor, as_completed
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(qualify_fixture, fixture, objective, starts=args.starts)
                for fixture in load_panel() for objective in ("shape", "combo")]
        for job in as_completed(jobs):
            receipt = job.result()
            receipts.append(receipt)
            print(f"{receipt['name']} {receipt['objective']}: qualified={receipt['qualified']} gap={receipt['outside_gap']:.6g}", flush=True)
            args.output.write_text(json.dumps({"rdkit": rdkit.__version__, "scipy": scipy.__version__,
                                               "receipts": receipts}, indent=2) + "\n")
    if args.portable:
        write_qualification(receipts)


if __name__ == "__main__":
    main()

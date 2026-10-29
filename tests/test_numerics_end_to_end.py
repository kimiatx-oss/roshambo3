"""LIT-PCBA through public prepare/read/screen/export interfaces."""
import importlib
import numpy as np
import pytest

from bench.numerics_end_to_end import prepare_panel, screen_panel, validate_result
from bench.numerics_backends import verified_build


@pytest.fixture(scope="module")
def real_panel(tmp_path_factory):
    pytest.importorskip("rdkit")
    return prepare_panel(tmp_path_factory.mktemp("lit-pcba") / "corpus")


def require_backend(backend):
    module = importlib.import_module("roshambo3.backend." + backend)
    if not module.available():
        pytest.skip(f"{backend} backend unavailable; real end-to-end arm NOT RUN")
    return module


@pytest.mark.parametrize("backend,path", [pytest.param("cpu", "analytic", marks=pytest.mark.numerics_coverage(
    backend="cpu", path="vector", check="end_to_end", channel="shape")),
    pytest.param("cuda", "analytic", marks=pytest.mark.numerics_coverage(
    backend="cuda", path="analytic", check="end_to_end", channel="shape")),
    *[pytest.param(backend, "field", marks=pytest.mark.numerics_coverage(
    backend=backend, path="field", check="end_to_end", channel="shape"))
      for backend in ("cpu", "cuda")],
    *[pytest.param("metal", sampler, marks=pytest.mark.numerics_coverage(
    backend="metal", path=sampler, check="end_to_end", channel="shape"))
      for sampler in ("manual", "texture16", "texture32")]])
@pytest.mark.parametrize("transform,starts,batch", [("identity",0,5),("grid",1,3),("arbitrary",1,5)])
def test_real_shape_public_pipeline(real_panel, backend, path, transform, starts, batch, numerics_coverage):
    module = require_backend(backend)
    if backend == "metal" and path == "texture32" and not module.extension().device_info(0)["float32_filtering"]:
        pytest.skip("RGBA32Float filtering unsupported by runtime; texture32 not run")
    qs, result = screen_panel(real_panel, backend=backend, path=path, transform=transform,
                              starts=starts, batch=batch)
    report = validate_result(real_panel, qs, result, transform=transform)
    stamp = verified_build(backend)
    numerics_coverage.stamp(build=stamp, arithmetic={"name": report["arithmetic"]["name"], "derivation": report["arithmetic"]["operation_model_derivation"]}, evidence=report)


@pytest.mark.parametrize("backend,path", [pytest.param("cpu", "field", marks=pytest.mark.numerics_coverage(
    backend="cpu", path="field", check="end_to_end", channel="color")),
    pytest.param("cuda", "analytic", marks=pytest.mark.numerics_coverage(
    backend="cuda", path="analytic", check="end_to_end", channel="color")),
    pytest.param("cuda", "field", marks=pytest.mark.numerics_coverage(
    backend="cuda", path="field", check="end_to_end", channel="color")),
    *[pytest.param("metal", sampler, marks=pytest.mark.numerics_coverage(
    backend="metal", path=sampler, check="end_to_end", channel="color"))
      for sampler in ("manual", "texture16", "texture32")]])
@pytest.mark.parametrize("mode", ["rescore", "combo"])
def test_real_color_public_pipeline(real_panel, backend, path, mode, numerics_coverage):
    module = require_backend(backend)
    if backend == "metal" and path == "texture32" and not module.extension().device_info(0)["float32_filtering"]:
        pytest.skip("RGBA32Float filtering unsupported by runtime; texture32 not run")
    qs, result = screen_panel(real_panel, backend=backend, path=path, mode=mode)
    report = validate_result(real_panel, qs, result, mode=mode)
    stamp = verified_build(backend)
    numerics_coverage.stamp(build=stamp, arithmetic={"name": report["arithmetic"]["name"], "derivation": report["arithmetic"]["operation_model_derivation"]}, evidence=report)


@pytest.mark.parametrize("fault", ["pose_inverse", "translation_scale", "row_association"])
def test_export_interface_faults_are_detected(real_panel, monkeypatch, fault):
    """Mutate actual export interfaces; this is explicitly not a native plant."""
    require_backend("cpu")
    import roshambo3.results as exports
    qs, result = screen_panel(real_panel)
    validate_result(real_panel, qs, result)  # Clean control through identical checker.
    if fault == "pose_inverse":
        original = exports._quat_matrix
        monkeypatch.setattr(exports, "_quat_matrix", lambda q: original(q).T)
    elif fault == "translation_scale":
        original = exports._apply
        invocation = [0]
        def scaled_translation(matrix, points):
            index = invocation[0]; invocation[0] += 1
            if index % 2:
                # The outer export receives R*shape+t: scale only its actual
                # physical translation at this interface, leaving rotation intact.
                points = points + .01*result[0].scores[index//2,13:16]
            return original(matrix, points)
        monkeypatch.setattr(exports, "_apply", scaled_translation)
    else:
        original = result.catalog.coordinates
        monkeypatch.setattr(result.catalog, "coordinates", lambda gids: list(reversed(original(gids))))
    with pytest.raises((AssertionError, ValueError)):
        validate_result(real_panel, qs, result)


def test_real_color_gradient_off_optimum(real_panel):
    require_backend("cpu")
    from bench.numerics_end_to_end import real_color_gradient
    verified_build("cpu")
    assert len(real_color_gradient(real_panel)) == 4


def test_prepare_feature_omission_at_generator_interface_is_detected(tmp_path, monkeypatch):
    """Actual preparation input-interface plant; no native qualification claim."""
    from roshambo3.pharmacophore import PharmacophoreGenerator, RDKitFeaturePlan
    original_plan = PharmacophoreGenerator.make_feature_plan
    def omit_plan(self, *args, **kwargs):
        # Every build perceives through the feature plan.
        plan = original_plan(self, *args, **kwargs)
        return RDKitFeaturePlan(plan.atom_ids[1:], plan.types[1:])
    monkeypatch.setattr(PharmacophoreGenerator, "make_feature_plan", omit_plan)
    with pytest.raises(AssertionError, match="feature assignment/count changed"):
        prepare_panel(tmp_path / "omitted-feature")

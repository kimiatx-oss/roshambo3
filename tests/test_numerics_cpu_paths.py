"""Live CPU path scheduling and final-frame coverage, without saved outputs."""
import numpy as np
import pytest

from bench.numerics_backends import (
    cpu_evaluator_capability, cpu_fixed_color_selection, seeded_panel, verified_build)
from roshambo3.backend import cpu
from test_numerics_backends import (
    test_cpu_canonical_vector_color_and_field_final_frame as check_final_frame)


# Actual factories, not architecture-name skips. The field final columns are
# analytically rescored; their frame checks do not qualify sampled SEARCH.
CONFIGURATIONS = [
    ("scalar_analytic", "scalar", 0, 0, 1, "scalar"),
    ("analytic_simd", "query_atom", 1, 1, 8, "vector"),
    ("analytic_simd", "library_atom", 1, 2, 8, "vector"),
    ("scalar_field", "scalar", 2, 0, 1, "field"),
    ("paired_channel_field", "library_atom", 3, 2, 8, "field"),
    ("pipelined_field", "library_atom", 4, 2, 8, "field"),
]


def parameters(check, *, include_scalar=True):
    return [pytest.param(config, id=config[0]+"-"+config[1], marks=[
        pytest.mark.numerics_coverage(backend="cpu", path=config[-1],
                                     check=check, channel=channel)
        for channel in ("shape", "color")])
        for config in CONFIGURATIONS if include_scalar or config[-1] != "scalar"]


def require_configuration(config):
    if not cpu.available():
        pytest.skip("CPU extension unavailable")
    verified_build("cpu")
    capability = cpu_evaluator_capability(*config[2:5])
    if not capability["available"]:
        pytest.skip("production CPU evaluator unavailable: "+capability["reason"])
    return capability


@pytest.mark.parametrize("configuration", parameters("frame", include_scalar=False))
def test_cpu_selected_final_frame(configuration, numerics_coverage):
    capability = require_configuration(configuration)
    check_final_frame(*configuration[:2])
    numerics_coverage.stamp(build=verified_build("cpu"), arithmetic=dict(
            family=configuration[-1],
            configs=[list(c[:5]) for c in CONFIGURATIONS if c[-1] == configuration[-1]],
            comparison="all output columns; same-build exact equality for scheduling, fp64 same-frame bounds for final scoring"),
        evidence=dict(selected_shape=capability, scope="selected production path exact final frame; not FIELD search arithmetic",
                      selected_color=cpu_fixed_color_selection(), rows=16))


@pytest.mark.parametrize("configuration", parameters("determinism", include_scalar=False))
def test_cpu_optimizer_workers_and_claim_chunks(configuration, numerics_coverage):
    capability = require_configuration(configuration)
    backend, grain, _, _, width, _ = configuration
    selected = cpu_fixed_color_selection()
    panel = seeded_panel(16)
    # Canonical typed interactions are required by retained native color grains.
    panel["interaction_r"][1:, 1:] = 1
    panel["interaction_p"][1:, 1:] = np.eye(2, dtype=np.float32)
    # 528 rows cross every supported chunk boundary, with a short final chunk.
    # Only 16 query fields are needed; repeated pair IDs still run complete
    # independent native optimizer states on each claimed row.
    panel["pair_query_row"] = np.arange(528, dtype=np.int32) % 16
    panel["pair_library_row"] = (np.arange(528, dtype=np.int32)*7+3) % 16
    options = dict(steps=12, start_mode=1, color=True, color_mode="combo_optimal",
                   color_grain=selected["grain"], mixing=.5,
                   shape_backend=backend, shape_grain=grain, shape_width=width)
    if "field" in backend:
        options["shape_field_spacing_angstrom"] = .25
    expected, census = cpu.score_pairs(**panel, **options, threads=1, with_census=True)
    assert np.isfinite(expected).all()
    assert np.any(expected[:, 13:16] != 0), "optimizer did not move from its initial pose"
    assert census["color_grain"] == selected["grain"]
    for key in ("backend", "grain", "width", "exponential", "reduction"):
        assert census["shape_evaluator_stamp"][key] == capability["stamp"][key]
    assert census["shape_gradient_evaluations"] > 0
    assert census["color_gradient_evaluations"] > 0
    if "field" in backend:
        assert census["shape_field_samples"] > 0
        assert census["shape_fallback_evaluations"] == 0
    axes = []
    for workers in (1, 2, 4):
        for chunk in (128, 256, 512):
            with cpu.CPUComboExecutor(workers, claim_chunk_rows=chunk) as executor:
                actual = cpu.score_pairs(**panel, **options, threads=workers, executor=executor)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            axes.append(dict(workers=workers, claim_chunk_rows=chunk))
    numerics_coverage.stamp(build=verified_build("cpu"), arithmetic=dict(
            family=configuration[-1],
            configs=[list(c[:5]) for c in CONFIGURATIONS if c[-1] == configuration[-1]],
            comparison="all output columns; same-build exact equality for scheduling, fp64 same-frame bounds for final scoring"),
        evidence=dict(selected_shape=capability, scope="same-build complete combo optimizer output scheduling equality",
                      rows=528, steps=12, starts=4, axes=axes,
                      selected_color=selected, shape_field_samples=census["shape_field_samples"],
                      shape_gradient_evaluations=census["shape_gradient_evaluations"],
                      color_gradient_evaluations=census["color_gradient_evaluations"],
                      native_fault_qualification="separate receipt required"))

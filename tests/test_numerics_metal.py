"""Portable detector controls and actual Metal runtime receipts."""
import os
import json
import numpy as np
import pytest

from bench.numerics_backends import seeded_panel
from bench.numerics_metal import (fixed, check_fixed, stamp, sampler_panel,
    interpolate, check_sampler, probe, weight_sweep, analytic_contract, check_weight_lattice, check_determinism)
from roshambo3.backend import metal


@pytest.fixture
def ctx():
    if not metal.available():
        if os.environ.get("ROSHAMBO3_REQUIRE_METAL") == "1":
            pytest.fail("Metal required; numerical family cannot run")
        pytest.skip("Metal unavailable; native numerical checks not run")
    with_context = metal.create_context()
    try:
        yield with_context
    finally:
        with_context.close()


def marks(path, *checks):
    return [pytest.mark.numerics_coverage(backend="metal", path=path, check=c, channel=ch)
            for c in checks for ch in ("shape", "color")]


@pytest.mark.numerics_coverage(backend="metal", path="analytic", check="fixed_pose", channel="shape")
@pytest.mark.numerics_coverage(backend="metal", path="analytic", check="fixed_pose", channel="color")
@pytest.mark.numerics_coverage(backend="metal", path="analytic", check="frame", channel="shape")
@pytest.mark.numerics_coverage(backend="metal", path="analytic", check="frame", channel="color")
def test_analytic_final_fixed_pose_same_frame(ctx, numerics_coverage):
    panel = seeded_panel()
    rows = fixed(panel, ctx)
    report = check_fixed(panel, rows)
    stamp(numerics_coverage, ctx, panel, residuals=report,
          scope="analytic final rescore only; no field optimizer arithmetic")


@pytest.mark.numerics_coverage(backend="metal", path="analytic", check="determinism", channel="shape")
@pytest.mark.numerics_coverage(backend="metal", path="analytic", check="determinism", channel="color")
def test_analytic_final_partition_determinism(ctx, numerics_coverage):
    panel = seeded_panel()
    expected = fixed(panel, ctx)
    for size in (1, 3, 11):
        actual = []
        for start in range(0, len(expected), size):
            part = dict(panel)
            for key in ("pair_query_row", "pair_library_row"):
                part[key] = panel[key][start:start+size]
            actual.append(fixed(part, ctx))
        check_determinism(np.concatenate(actual), expected)
    stamp(numerics_coverage, ctx, panel, partitions=[1, 3, 11])


@pytest.mark.parametrize("sampler", ["manual", "texture16", "texture32"])
def test_native_direct_sampler(ctx, sampler, record_property):
    if sampler == "texture32" and not ctx.info["float32_filtering"]:
        pytest.skip("RGBA32Float filtering unsupported by runtime")
    nodes, points = sampler_panel()
    report = check_sampler(nodes, points, probe(ctx, sampler, nodes, points), sampler=sampler)
    if sampler != "manual":
        report.update(weight_sweep(ctx, sampler))
    record_property("metal_sampler_measurement", json.dumps(report, sort_keys=True))
    # This deliberately does not emit field fixed_pose coverage: synthetic
    # texels exercise actual sampler arithmetic, not Gaussian construction.


@pytest.mark.parametrize("sampler", [pytest.param(s, marks=marks(s, "determinism"))
    for s in ("manual", "texture16", "texture32")])
def test_field_dispatch_partition_shape_and_color(ctx, sampler, monkeypatch, numerics_coverage):
    from roshambo3.layout import tile32
    from roshambo3.backend.spec import ColorPlacement
    if sampler == "texture32" and not ctx.info["float32_filtering"]:
        pytest.skip("RGBA32Float filtering unsupported by runtime")
    panel = seeded_panel(12)
    # Force a dense diagonal color plane to exercise color-field dispatch too.
    panel["interaction_r"][:] = np.diag([0, .8, .7])
    panel["interaction_p"][:] = np.diag([0, 1, .9])
    candidate = metal.create_context(field_sampler=sampler, color_field_sampler=sampler,
                                     color_field_min_sources=2)
    try:
        args = dict(panel, library_coords=tile32(panel["library_coords"]), library_color_offset=5,
                    color=True, mixing=.5, color_placement=ColorPlacement.IN_GRADIENT,
                    steps=3, start_mode=0, field_pad_a=2)
        monkeypatch.setattr(metal, "_TEXTURE_BINDING_LIMIT", 128)
        expected = metal.score_pairs(**args, context=candidate)
        original_dispatches = len(candidate.last_optimizer_dispatches)
        monkeypatch.setattr(metal, "_TEXTURE_BINDING_LIMIT", 4)
        actual = metal.score_pairs(**args, context=candidate)
        if sampler == "manual":
            # Manual interpolation binds no textures, so texture-binding
            # limits cannot split its query batches.
            assert len(candidate.last_optimizer_dispatches) == original_dispatches
            scope = "manual dispatch invariant to texture-binding limits"
        else:
            assert len(candidate.last_optimizer_dispatches) > original_dispatches
            scope = "field optimizer dispatch partition"
        assert candidate.last_color_fields, "color field path must actually execute"
        check_determinism(actual, expected)
        stamp(numerics_coverage, candidate, panel, binding_limits=[128, 4],
              scope=f"{scope}; final scores analytic",
              sampler=sampler, dispatches=candidate.last_optimizer_dispatches)
    finally:
        candidate.close()


def test_sampler_model_manual_and_texture_borders_differ():
    nodes = np.ones((3, 3, 3, 4), np.float32)
    p = np.array([[-.25, 1, 1], [2, 1, 1], [3, 1, 1]])
    np.testing.assert_array_equal(interpolate(nodes, p, sampler="manual"), 0)
    np.testing.assert_array_equal(interpolate(nodes, p, sampler="texture16")[:, 0], [.75, 1, 0])


@pytest.mark.parametrize("sampler", ["manual", "texture16", "texture32"])
def test_sampler_model_control_and_axis_fault(sampler):
    nodes, points = sampler_panel()
    stored = nodes.astype(np.float16).astype(np.float32) if sampler == "texture16" else nodes
    samples = interpolate(stored, points, sampler=sampler)
    check_sampler(nodes, points, samples, sampler=sampler)
    with pytest.raises(AssertionError, match="sampler"):
        check_sampler(nodes, points, samples[:, [1, 0, 2, 3]], sampler=sampler)


def test_half_write_interval_covers_rte_rtz_and_rejects_corruption():
    nodes, points = sampler_panel()
    nearest = nodes.astype(np.float16)
    nearest32 = nearest.astype(np.float32)
    toward_zero = np.nextafter(nearest, np.zeros_like(nearest))
    move_toward_zero = (
        ((nodes > 0) & (nearest32 > nodes))
        | ((nodes < 0) & (nearest32 < nodes))
    )
    toward_zero = np.where(move_toward_zero, toward_zero, nearest).astype(np.float32)

    nearest_samples = interpolate(nearest32, points, sampler="texture16")
    nearest_report = check_sampler(nodes, points, nearest_samples, sampler="texture16")
    assert nearest_report["texture_write_rounding"] == (
        "native default; exact hardware strategy undeclared")
    toward_zero_samples = interpolate(toward_zero, points, sampler="texture16")
    check_sampler(nodes, points, toward_zero_samples, sampler="texture16")

    # A write one representable half beyond the adjacent-value interval must
    # still fail the exact-center detector; this guards against loosening it.
    small = np.zeros((3, 3, 3, 4), np.float32)
    small[1, 1, 1, 0] = 1.0001
    stored = small.astype(np.float16)
    upper = np.nextafter(stored[1, 1, 1, 0], np.float16(np.inf))
    corrupt = np.nextafter(upper, np.float16(np.inf))
    assert (
        float(stored[1, 1, 1, 0])
        < float(small[1, 1, 1, 0])
        < float(upper)
        < float(corrupt)
    )
    stored[1, 1, 1, 0] = corrupt
    corrupted_sample = interpolate(
        stored.astype(np.float32), np.array([[1, 1, 1]], np.float32),
        sampler="texture16")
    with pytest.raises(AssertionError, match="sampler center/border"):
        check_sampler(small, np.array([[1, 1, 1]], np.float32),
                      corrupted_sample, sampler="texture16")


def test_declared_weight_detector_catches_coarser_lattice():
    # This is a synthetic declared arithmetic control, NOT an Apple8 claim.
    z, y, x = np.indices((3, 3, 3))
    nodes = np.stack((x, y, z, x), axis=-1).astype(np.float32)
    points = np.zeros((4096, 3), np.float32)
    points[:, 0] = np.arange(4096)/4096
    check_weight_lattice(nodes, points,
        interpolate(nodes, points, sampler="texture32", weight_bits=8), weight_bits=8)
    with pytest.raises(AssertionError, match="sampler weight precision"):
        check_weight_lattice(nodes, points,
            interpolate(nodes, points, sampler="texture32", weight_bits=7), weight_bits=8)



def test_metal_fixed_detector_catches_frame_and_constant_faults():
    from bench.numerics_backends import reference_rows, COLUMNS
    panel = seeded_panel()
    expected, _ = reference_rows(panel, analytic_contract(panel))
    rows = np.zeros((len(expected), 20), np.float32)
    rows[:, :9] = [[r[k] for k in COLUMNS] for r in expected]
    rows[:, 9] = 1
    check_fixed(panel, rows)
    split = rows.copy(); split[:, 5] *= 1.001
    with pytest.raises(AssertionError, match="self_overlap_query"):
        check_fixed(panel, split)
    shifted = rows.copy(); shifted[:, 3] *= 1.0001
    with pytest.raises(AssertionError, match="overlap_volume"):
        check_fixed(panel, shifted)


def test_partition_detector_rejects_one_ulp_reduction_plant():
    rows = np.array([1., .5, .25], np.float32)
    check_determinism(rows.copy(), rows)
    altered = rows.copy()
    altered[1] = np.nextafter(altered[1], np.float32(1))
    with pytest.raises(AssertionError, match="dispatch partition"):
        check_determinism(altered, rows)

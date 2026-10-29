"""Immutable reusable combo self-overlap handles."""

import hashlib

from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np
import pytest

from roshambo3.backend import cpu
from cpu_test_capabilities import IS_ARM, RETAINED_COLOR_GRAIN
from roshambo3.reference.color import MODE_COMBO_OPTIMAL
from roshambo3.prep.conformer import stored_self_volume

from test_cpu_shared_pose_contract import _inputs


def _combo(data, *, grain=RETAINED_COLOR_GRAIN, prepared=None, **overrides):
    q, qt, qn, library, counts, widths, heights = data
    args = dict(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        # Deliberately repeated, sparse, and out of order: self terms index
        # source rows, never pair positions.
        pair_query_row=np.array([1, 0, 1, 0], dtype=np.int32),
        pair_library_row=np.array([2, 0, 2, 1], dtype=np.int32),
        interaction_r=widths, interaction_p=heights,
        color=True, color_mode=MODE_COMBO_OPTIMAL, color_grain=grain,
        steps=2, start_mode=1, mixing=0.5, with_census=True)
    args.update(overrides)
    if prepared is not None:
        args["prepared_self_overlaps"] = prepared
    return cpu.score_pairs(**args)


@pytest.mark.parametrize("grain", [11, 16])
@pytest.mark.parametrize("canonical", [False, True])
def test_prepared_combo_self_overlaps_are_exact_and_skip_all_self_work(
        grain, canonical):
    data = _inputs(canonical=canonical)
    q, qt, qn, library, counts, widths, heights = data
    library_self = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights)
    prepared = cpu.prepare_combo_self_overlaps(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        prepared_library=library_self)

    reference, fallback = _combo(data, grain=grain)
    actual, stats = _combo(data, grain=grain, prepared=prepared)
    np.testing.assert_array_equal(actual, reference)
    assert fallback["shape_query_self_evaluations"] == q.shape[0]
    assert fallback["shape_library_self_evaluations"] == library.shape[0]
    assert fallback["color_query_self_evaluations"] == q.shape[0]
    assert fallback["color_library_self_evaluations"] == library.shape[0]
    assert fallback["shape_query_self_terms"] > 0
    assert fallback["color_library_self_terms"] > 0
    assert stats["prepared_self_overlaps_used"] is True
    for channel in ("shape_query", "shape_library",
                    "color_query", "color_library"):
        assert stats[f"prepared_{channel}_self_used"] is True
        assert stats[f"{channel}_self_evaluations"] == 0
        assert stats[f"{channel}_self_terms"] == 0
    remapped, remap_stats = _combo(
        data, grain=grain, prepared=prepared,
        pair_query_row=np.array([0, 1], dtype=np.int32),
        pair_library_row=np.array([1, 2], dtype=np.int32))
    assert np.isfinite(remapped).all()
    assert remap_stats["prepared_self_overlaps_used"] is True


@pytest.mark.parametrize(
    ("prepare_shape", "prepare_color", "prepared_channel", "recomputed_channel"),
    [(True, False, "shape", "color"), (False, True, "color", "shape")])
def test_partial_combo_self_overlaps_are_exact_and_skip_only_selected_channel(
        prepare_shape, prepare_color, prepared_channel, recomputed_channel):
    data = _inputs(canonical=False)
    q, qt, qn, library, counts, widths, heights = data
    library_self = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        shape=prepare_shape, color=prepare_color)
    prepared = cpu.prepare_combo_self_overlaps(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        prepared_library=library_self,
        shape=prepare_shape, color=prepare_color)

    reference, fallback = _combo(data, grain=RETAINED_COLOR_GRAIN)
    actual, stats = _combo(data, grain=RETAINED_COLOR_GRAIN, prepared=prepared)
    np.testing.assert_array_equal(actual, reference)
    assert prepared.shape_prepared is prepare_shape
    assert prepared.color_prepared is prepare_color
    assert stats["prepared_self_overlaps_used"] is True
    assert stats[f"prepared_{prepared_channel}_query_self_used"] is True
    assert stats[f"prepared_{prepared_channel}_library_self_used"] is True
    assert stats[f"{prepared_channel}_query_self_evaluations"] == 0
    assert stats[f"{prepared_channel}_library_self_evaluations"] == 0
    assert stats[f"{prepared_channel}_query_self_terms"] == 0
    assert stats[f"{prepared_channel}_library_self_terms"] == 0
    assert stats[f"prepared_{recomputed_channel}_query_self_used"] is False
    assert stats[f"prepared_{recomputed_channel}_library_self_used"] is False
    assert stats[f"{recomputed_channel}_query_self_evaluations"] == q.shape[0]
    assert stats[f"{recomputed_channel}_library_self_evaluations"] == library.shape[0]
    assert fallback[f"{recomputed_channel}_query_self_evaluations"] == q.shape[0]
    assert fallback[f"{recomputed_channel}_library_self_evaluations"] == library.shape[0]


def test_combo_self_overlap_preparation_requires_a_selected_channel():
    data = _inputs(canonical=False)
    _, _, _, library, counts, widths, heights = data
    with pytest.raises(ValueError, match="at least one"):
        cpu.prepare_combo_library_self_overlaps(
            library_coords=library, library_counts=counts,
            interaction_r=widths, interaction_p=heights,
            shape=False, color=False)


def test_library_handle_reused_for_multiple_query_sets_without_recomputing():
    data = _inputs(canonical=False)
    q, qt, qn, library, counts, widths, heights = data
    library_self = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights)
    original_values = (library_self.shape_values.copy(),
                       library_self.color_values.copy())
    first = cpu.prepare_combo_self_overlaps(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        prepared_library=library_self)
    second = cpu.prepare_combo_self_overlaps(
        query_coords=q[:1].copy(), query_types=qt[:1].copy(),
        query_n_atoms=qn[:1].copy(), library_coords=library,
        library_counts=counts, interaction_r=widths, interaction_p=heights,
        prepared_library=library_self)
    np.testing.assert_array_equal(first.library.shape_values, original_values[0])
    np.testing.assert_array_equal(second.library.shape_values, original_values[0])
    np.testing.assert_array_equal(first.library.color_values, original_values[1])
    np.testing.assert_array_equal(second.library.color_values, original_values[1])
    assert library_self.shape_terms > 0
    assert library_self.color_terms > 0


def test_prepared_handle_rejects_mutated_coordinates_types_or_matrix():
    data = _inputs(canonical=False)
    q, qt, qn, library, counts, widths, heights = data
    prepared = cpu.prepare_combo_self_overlaps(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights)
    common = dict(prepared=prepared)
    _combo(data, **common)

    q[0, 0, 0] += 0.125
    with pytest.raises(ValueError, match="stale"):
        _combo(data, **common)
    q[0, 0, 0] -= 0.125
    qt[0, 1] = 2
    with pytest.raises(ValueError, match="stale"):
        _combo(data, **common)
    qt[0, 1] = 1
    counts[2, 2] = 1
    with pytest.raises(ValueError, match="stale"):
        _combo(data, **common)
    counts[2, 2] = 0
    library[0, 0, 0] += 0.125
    with pytest.raises(ValueError, match="stale"):
        _combo(data, **common)
    library[0, 0, 0] -= 0.125
    widths[1, 1] += 0.1
    with pytest.raises(ValueError, match="stale"):
        _combo(data, **common)
    widths[1, 1] -= 0.1
    heights[1, 1] += 0.1
    with pytest.raises(ValueError, match="stale"):
        _combo(data, **common)


def test_prepared_handle_rejects_mutated_python_snapshot_values():
    data = _inputs(canonical=False)
    q, qt, qn, library, counts, widths, heights = data
    prepared = cpu.prepare_combo_self_overlaps(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights)
    stored = prepared.library.shape_values
    stored.flags.writeable = True
    original = float(stored[0])
    stored[0] += 0.25
    with pytest.raises(ValueError, match="self-overlap values were mutated"):
        _combo(data, prepared=prepared)
    stored[0] = original
    stored.flags.writeable = False


def test_self_overlap_preparation_validates_unused_padding_values():
    data = _inputs(canonical=False)
    q, qt, qn, library, counts, widths, heights = data
    bad_query = q.copy()
    bad_query[0, -1, 0] = np.nan
    with pytest.raises(ValueError, match="coordinates/weights must be finite"):
        cpu.prepare_combo_self_overlaps(
            query_coords=bad_query, query_types=qt, query_n_atoms=qn,
            library_coords=library, library_counts=counts,
            interaction_r=widths, interaction_p=heights)

    bad_library = library.copy()
    bad_library[2, -1, 3] = -0.25
    with pytest.raises(ValueError, match="weights must be nonnegative"):
        cpu.prepare_combo_library_self_overlaps(
            library_coords=bad_library, library_counts=counts,
            interaction_r=widths, interaction_p=heights)


def test_array_vaa_requires_authenticated_stored_frame_and_declared_units():
    data = _inputs(canonical=False)
    q, qt, qn, library, counts, widths, heights = data
    values = np.array([
        stored_self_volume(row[:int(count[0]), :3])
        for row, count in zip(library, counts)], dtype=np.float64)
    with pytest.raises(ValueError, match="explicit stored-frame"):
        cpu.prepare_combo_library_self_overlaps(
            library_coords=library, library_counts=counts,
            interaction_r=widths, interaction_p=heights,
            library_vaa=values)
    stored = {
        "schema": "roshambo3.vaa.v1", "frame": "stored",
        "units": "angstrom^3", "coordinate_units": "angstrom",
        "wire_quantization_step_angstrom": float(WIRE_QUANT_STEP),
        "algorithm": "self_overlap_from_stored_quantized_xyz_v1",
        "verified": True,
        "row_order": "group_coordinate_row",
        "n_rows": len(library),
        "shape_coordinates_sha256": cpu._shape_library_fingerprint(
            library, counts),
        "values_sha256": hashlib.sha256(
            values.astype("<f4").tobytes(order="C")).hexdigest(),
    }
    library_self = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        library_vaa=values, library_vaa_metadata=stored)
    assert library_self.library_shape_from_stored_volume is True

    # The sidecar contract certifies float32 bytes.  A float64 value can
    # quantize to those same bytes yet produce a different native denominator
    # unless consumption uses the certified representation too.
    certified_values = values.astype(np.float32).astype(np.float64)
    certified_values[0] = 1000.0
    certified_metadata = dict(stored)
    certified_metadata["values_sha256"] = hashlib.sha256(
        certified_values.astype("<f4").tobytes(order="C")).hexdigest()
    off_grid_values = certified_values.copy()
    off_grid_values[0] = 1000.00001
    assert off_grid_values[0] != certified_values[0]
    assert np.float32(off_grid_values[0]) == np.float32(certified_values[0])
    certified_prepared = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        library_vaa=certified_values,
        library_vaa_metadata=certified_metadata,
        color=False)
    off_grid_prepared = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        library_vaa=off_grid_values,
        library_vaa_metadata=certified_metadata,
        color=False)
    expected_shape_values = (
        certified_values / cpu.extension().__shape_prefactor__).astype(np.float32)
    np.testing.assert_array_equal(off_grid_prepared.shape_values,
                                  certified_prepared.shape_values)
    np.testing.assert_array_equal(off_grid_prepared.shape_values,
                                  expected_shape_values)

    wrong_rows = library.copy()
    wrong_rows[0, 0, 0] += float(WIRE_QUANT_STEP)
    with pytest.raises(ValueError, match="shape_coordinates_sha256"):
        cpu.prepare_combo_library_self_overlaps(
            library_coords=wrong_rows, library_counts=counts,
            interaction_r=widths, interaction_p=heights,
            library_vaa=values, library_vaa_metadata=stored)
    wrong_values = values.copy()
    wrong_values[0] += 0.25
    with pytest.raises(ValueError, match="values_sha256"):
        cpu.prepare_combo_library_self_overlaps(
            library_coords=library, library_counts=counts,
            interaction_r=widths, interaction_p=heights,
            library_vaa=wrong_values, library_vaa_metadata=stored)
    recomputed = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights)
    np.testing.assert_allclose(library_self.shape_values,
                               recomputed.shape_values,
                               rtol=2e-6, atol=2e-6)
    prepared = cpu.prepare_combo_self_overlaps(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        prepared_library=library_self)
    common = dict(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        pair_query_row=np.array([0, 1, 0], np.int32),
        pair_library_row=np.array([2, 1, 0], np.int32),
        interaction_r=widths, interaction_p=heights,
        color=True, color_mode=MODE_COMBO_OPTIMAL, color_grain=RETAINED_COLOR_GRAIN,
        steps=1, lr_q=0.0, lr_t=0.0,
        budget_override="T204 frozen-pose denominator comparison",
        start_mode=1, mixing=0.5)
    recomputed_scores = cpu.score_pairs(**common)
    stored_scores = cpu.score_pairs(**common,
                                    prepared_self_overlaps=prepared)
    np.testing.assert_allclose(stored_scores[:, :9],
                               recomputed_scores[:, :9],
                               rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(np.argsort(-stored_scores[:, 0]),
                                  np.argsort(-recomputed_scores[:, 0]))


def test_versioned_color_self_sidecar_is_matrix_and_library_specific(tmp_path):
    data = _inputs(canonical=False)
    _, _, _, library, counts, widths, heights = data
    native = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights)
    path = tmp_path / "color-self.npz"
    metadata = cpu.write_combo_color_self_sidecar(
        path, values=native.color_values, library_coords=library,
        library_counts=counts, interaction_r=widths, interaction_p=heights)
    values, loaded_metadata = cpu.load_combo_color_self_sidecar(
        path, library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights)
    assert loaded_metadata == metadata
    imported = cpu.prepare_combo_library_self_overlaps(
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights,
        library_color_self=values, library_color_metadata=loaded_metadata)
    assert imported.library_color_from_sidecar is True
    np.testing.assert_array_equal(imported.color_values, native.color_values)
    altered = widths.copy()
    altered[1, 1] += 0.01
    with pytest.raises(ValueError, match="library_fingerprint"):
        cpu.load_combo_color_self_sidecar(
            path, library_coords=library, library_counts=counts,
            interaction_r=altered, interaction_p=heights)
    with pytest.raises(FileExistsError):
        cpu.write_combo_color_self_sidecar(
            path, values=native.color_values, library_coords=library,
            library_counts=counts, interaction_r=widths, interaction_p=heights)


def test_empty_color_rows_prepare_and_score_as_zero():
    q, qt, qn, library, counts, widths, heights = _inputs(canonical=False)
    q = q[:1].copy()
    q[:, 1:, :3] = 0.0
    qt = np.zeros_like(qt[:1])
    qn = np.array([1], dtype=np.int32)
    library = library[:1].copy()
    counts = np.array([[1, 0, 0]], dtype=np.uint8)
    prepared = cpu.prepare_combo_self_overlaps(
        query_coords=q, query_types=qt, query_n_atoms=qn,
        library_coords=library, library_counts=counts,
        interaction_r=widths, interaction_p=heights)
    data = (q, qt, qn, library, counts, widths, heights)
    scores, stats = _combo(
        data, grain=RETAINED_COLOR_GRAIN, prepared=prepared,
        pair_query_row=np.array([0], np.int32),
        pair_library_row=np.array([0], np.int32))
    assert np.isfinite(scores).all()
    assert scores[0, 4] == 0.0
    assert scores[0, 7] == 0.0
    assert scores[0, 8] == 0.0
    assert stats["prepared_self_overlaps_used"] is True

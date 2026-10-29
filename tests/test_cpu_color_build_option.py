"""Compiled-grain reachability and explicit refusal contract."""

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.reference.color import MODE_SHAPE_POSE_COLOR_RESCORE


pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")

UNSUPPORTED_GRAINS = (1, 3, 12, 17, 18)
UNSUPPORTED = "unsupported CPU color grain"


def _inputs():
    query = np.zeros((1, 2, 3), dtype=np.float32)
    query[0, 0] = (0.1, 0.2, -0.1)
    query[0, 1] = (0.4, -0.2, 0.3)
    library = np.zeros((1, 2, 3), dtype=np.float32)
    library[0, 0] = (-0.1, 0.3, 0.2)
    library[0, 1] = (0.2, 0.1, -0.4)
    widths = np.ones((3, 3), dtype=np.float32)
    heights = np.eye(3, dtype=np.float32)
    return dict(
        query_coords=query,
        query_types=np.array([[0, 1]], dtype=np.int32),
        query_n_atoms=np.array([1], dtype=np.int32),
        library_coords=library,
        library_counts=np.array([[1, 1, 0]], dtype=np.uint8),
        pair_query_row=np.array([0], dtype=np.int32),
        pair_library_row=np.array([0], dtype=np.int32),
        interaction_r=widths,
        interaction_p=heights,
    )


def test_stamp_reports_supported_grains_and_refuses_others():
    for grain in (0, 11, 16):
        stamp = cpu.color_build_stamp(grain)
        assert stamp["color_compiled"] is True
        # Compile availability is independent of the target CPU's runtime ISA.
        assert isinstance(stamp["color_target_compiled"], bool)
        assert isinstance(stamp["color_runtime_available"], bool)
    for grain in UNSUPPORTED_GRAINS:
        with pytest.raises(ValueError, match=UNSUPPORTED):
            cpu.color_build_stamp(grain)


def test_mode2_default_and_retained_start_rescore_remap_are_unchanged():
    kwargs = _inputs()
    options = dict(color=True, color_mode=MODE_SHAPE_POSE_COLOR_RESCORE,
                   steps=1, start_mode=0)
    default = cpu.score_pairs(**options, **kwargs)
    scheduled_start_request = cpu.score_pairs(
        **options, color_grain=16, **kwargs)
    np.testing.assert_array_equal(scheduled_start_request, default)


@pytest.mark.parametrize("grain", [11, 16])
def test_scalar_simd_probe_accepts_only_the_scalar_grain(grain):
    kwargs = _inputs()
    with pytest.raises(
            ValueError,
            match="scalar_color_simd_probe supports color grain 0 only"):
        cpu.extension().scalar_color_simd_probe(
            kwargs["query_coords"][0], kwargs["query_types"][0],
            int(kwargs["query_n_atoms"][0]), kwargs["library_coords"][0],
            kwargs["library_counts"][0], kwargs["interaction_r"],
            kwargs["interaction_p"], np.array([1, 0, 0, 0], dtype=np.float32),
            np.zeros(3, dtype=np.float32), grain, False)


def test_unsupported_grains_fail_at_public_and_native_boundaries():
    kwargs = _inputs()
    with pytest.raises(ValueError, match=UNSUPPORTED):
        cpu.score_pairs(color=True, color_mode=MODE_SHAPE_POSE_COLOR_RESCORE,
                        color_grain=17, steps=1, start_mode=0, **kwargs)

    ext = cpu.extension()
    with pytest.raises(ValueError, match=UNSUPPORTED):
        ext.combo_thread_config_stamp(1, 12, 0, 1, False)

    with pytest.raises(ValueError, match=UNSUPPORTED):
        ext.optimize_scalar_combo(
            kwargs["query_coords"], kwargs["query_types"],
            kwargs["query_n_atoms"], kwargs["library_coords"],
            kwargs["library_counts"], kwargs["pair_query_row"],
            kwargs["pair_library_row"], kwargs["interaction_r"],
            kwargs["interaction_p"], 1, 0.1, 0.1, 1, 0.5,
            False, False, color_grain=17)

    with pytest.raises(ValueError, match=UNSUPPORTED):
        ext.rescore_scalar_color(
            kwargs["query_coords"], kwargs["query_types"],
            kwargs["query_n_atoms"], kwargs["library_coords"],
            kwargs["library_counts"], kwargs["pair_query_row"],
            kwargs["pair_library_row"], kwargs["interaction_r"],
            kwargs["interaction_p"], np.zeros((1, 20), dtype=np.float32),
            np.float32(0.5), False, False, False, 17)

    with pytest.raises(ValueError, match="supports color grain 0 only"):
        ext.scalar_color_simd_probe(
            kwargs["query_coords"][0], kwargs["query_types"][0], 1,
            kwargs["library_coords"][0], kwargs["library_counts"][0],
            kwargs["interaction_r"], kwargs["interaction_p"],
            np.array([1, 0, 0, 0], dtype=np.float32),
            np.zeros(3, dtype=np.float32), 3, True)

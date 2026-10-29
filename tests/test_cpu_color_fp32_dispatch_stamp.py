"""The retained grain16 stamp describes the prepared canonical FP32 route."""

import os

import pytest

import roshambo3

extension_dir = os.environ.get("ROSHAMBO3_CPU_EXTENSION_DIR")
if extension_dir:
    roshambo3.__path__.append(extension_dir)

from roshambo3.backend import cpu


pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")


def test_grain16_stamp_declares_the_fp32_route_and_retained_fallback():
    stamp = cpu.color_build_stamp(16)

    assert stamp["color_prepared_canonical_route"] is True
    assert stamp["color_isa"] in ("AVX2+FMA", "NEON")
    assert "shared FP32 reduced pose" in stamp["color_pose_transform"]
    assert "FP32 transform" in stamp["color_value_precision"]
    assert "accurate analytic final" in stamp["color_value_precision"]
    assert "FP64" in stamp["color_declared_numerics_change"]
    assert "all planes in FP32 per start" in stamp["color_reduction"]
    assert "no cross-start reduction" in stamp["color_reduction"]
    assert stamp["color_analytic_final_exp"].startswith("std::exp(float)")
    assert "exact FP64 scalar fallback" in stamp["color_out_of_range_policy"]

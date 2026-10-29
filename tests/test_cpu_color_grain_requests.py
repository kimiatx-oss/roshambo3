"""The scalar color arm's stamp, and the color-grain requests every build refuses.

Grain ids other than 0, 11 and 16 are refused in test_cpu_color_build_option.
"""

import numpy as np
import pytest

from roshambo3.backend import cpu
from test_cpu_scalar_color import cases, packed


pytestmark = pytest.mark.skipif(not cpu.available(), reason="CPU extension absent")


def test_color_arm_stamp_names_color_isa_width_and_reduction_policy():
    scalar = cpu.color_build_stamp(0)
    assert scalar["color_compiled"] is True
    assert scalar["color_arm"] == "scalar"
    assert scalar["color_isa"] == "SCALAR"
    assert scalar["color_vector_width"] == 1
    assert scalar["color_reduction_width"] == 1
    assert scalar["color_scalar_default"] is True
    assert scalar["cpu_src_sha256"]


def test_unknown_grain_is_rejected():
    _, _, case, _ = next(x for x in cases() if x[0] == "one_feature_each")
    kw = packed(case, posed=False)
    with pytest.raises(ValueError, match="grain"):
        cpu.extension().scalar_color_simd_probe(
            kw["query_coords"][0], kw["query_types"][0],
            int(kw["query_n_atoms"][0]), kw["library_coords"][0],
            kw["library_counts"][0], kw["interaction_r"],
            kw["interaction_p"], np.array([1, 0, 0, 0], np.float32),
            np.zeros(3, np.float32), 18, True)


def test_public_color_grain_refuses_shape_only_and_unknown_values():
    _, _, case, _ = next(x for x in cases() if x[0] == "one_feature_each")
    kw = packed(case, posed=False)
    with pytest.raises(ValueError, match="requires a color"):
        cpu.score_pairs(color_grain=16, **kw)
    with pytest.raises(ValueError, match="color_grain must be"):
        cpu.score_pairs(color=True, mixing=0.5, color_grain=18, **kw)

"""A dispatch cell that is not compiled must refuse by name, not substitute.

The build compiles only the dispatch cells a shipped configuration can select.
That is only safe because the ladder ends in a throw: an absent cell becomes a
loud, diagnosable failure naming every knob and its value, rather than a
silent fall-through to a different schedule -- a failure mode this project has
already paid for once.
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3 import backend
from roshambo3.backend import ColorPlacement, CoordResidency, PoseSearch

from test_color_field_cuda import _fixture

needs_gpu = pytest.mark.skipif(not backend.available(),
                               reason="no CUDA extension in this build")

#: Knob names the throw must carry. A message that merely says "not
#: instantiated" would leave a user with no way to tell WHICH knob to move.
KNOBS = ("lanes_per_group", "start_groups_per_row", "coord_residency",
         "color_residency", "color_stage", "min_blocks_per_sm",
         "stream_min_blocks_per_sm", "update_lanes", "update_rule",
         "color_mode", "interleave", "query_color_residency", "color_ballot")


def _shape_launch(**kw):
    """A shape-only field launch at a padded width every tier supports."""
    fx = _fixture()["shape"]
    return backend.score_pairs(pose_search=PoseSearch.FIELD, steps=5,
                               start_mode=1, **kw, **fx)


class TestAnUncompiledCellRefusesByName:

    @needs_gpu
    def test_the_register_tier_refuses_and_names_every_knob(self):
        """The register coordinate tier was measured and is not shipped, so
        it has no cell."""
        with pytest.raises(Exception) as exc:
            _shape_launch(coord_residency=CoordResidency.REGISTER,
                          min_blocks_per_sm=4, start_groups_per_row=1)
        message = str(exc.value)
        assert "not instantiated" in message
        for knob in KNOBS:
            assert knob in message, f"the throw does not name {knob}"
        assert "coord_residency=3" in message

    @needs_gpu
    def test_a_tuning_knob_refuses_rather_than_launching_something_else(self):
        """`lanes_per_group` away from the shipped 4 has no cell."""
        with pytest.raises(Exception) as exc:
            _shape_launch(lanes_per_group=2, start_groups_per_row=1)
        assert "not instantiated" in str(exc.value)
        assert "lanes_per_group=2" in str(exc.value)

    @needs_gpu
    def test_the_shipped_streamed_minimum_still_launches(self):
        """The combined path ships ``stream_min_blocks_per_sm=2``, and those
        cells read as search space only because the ladder's `defaults_`
        test predates that default. Pruning them broke four tests once; this is the
        gate that says it stays fixed."""
        fx = _fixture()["color"]
        # IN_GRADIENT, because that is the path that default applies to:
        # `cuda.py::score_pairs` resolves stream_min_blocks_per_sm to 2 only on
        # `selected_combo_field`. The AT_POSE cross is not instantiated at
        # MINB_S = 2 on either side of this change.
        v = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                mixing=0.5, steps=5, start_mode=1,
                                color_placement=ColorPlacement.IN_GRADIENT,
                                stream_min_blocks_per_sm=2, **fx)
        assert np.isfinite(v).all()

    @needs_gpu
    def test_the_combo_field_default_resolves_into_a_compiled_cell(self):
        """The shipped combined-path schedule with nothing named: this is the launch that four
        tests failed on when the extended ladder was guarded wholesale."""
        fx = _fixture()["color"]
        v = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                mixing=0.5, steps=5, start_mode=1,
                                color_placement=ColorPlacement.IN_GRADIENT,
                                **fx)
        assert np.isfinite(v).all()

    @needs_gpu
    def test_the_shared_residency_tier_launches(self):
        """SHARED is the one coordinate residency tier the build compiles."""
        v = _shape_launch(coord_residency=CoordResidency.SHARED)
        assert np.isfinite(v).all()

    @needs_gpu
    @pytest.mark.parametrize("tier", [CoordResidency.GLOBAL,
                                      CoordResidency.GLOBAL_LDCG,
                                      CoordResidency.GLOBAL_LDCS])
    def test_a_global_residency_tier_refuses_by_name(self, tier):
        with pytest.raises(Exception) as exc:
            _shape_launch(coord_residency=tier)
        assert "not instantiated" in str(exc.value)
        assert f"coord_residency={int(tier)}" in str(exc.value)

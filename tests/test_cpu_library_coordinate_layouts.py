"""Wire-format and strided library coordinates score as their float32 equivalent.

`cpu.score_pairs` accepts the corpus's int16 wire record (wire-step steps) and
arrays in any memory layout, converting both to contiguous float32 before the
native call. This checks the outcome, not an intercepted native entry point.
"""
import numpy as np
import pytest

from roshambo3.numerics import WIRE_QUANT_STEP

pytest.importorskip("roshambo3._roshambo3_cpu")


def _scores(library, data, color):
    from roshambo3.backend import cpu
    q, qt, qn, _, counts, r, p = data
    kw = dict(query_coords=q, query_n_atoms=qn, library_coords=library,
              library_counts=counts, pair_query_row=np.array([0, 1, 0], np.int32),
              pair_library_row=np.arange(3, dtype=np.int32), steps=25,
              start_mode=1, threads=1)
    if color:
        from test_cpu_shared_pose_contract import _prepare
        kw.update(query_types=qt, interaction_r=r, interaction_p=p, color=True,
                  color_mode="combo_optimal", color_grain=16, mixing=.5,
                  shape_backend="pipelined_field", shape_grain="library_atom",
                  shape_width=8, shape_field_spacing_angstrom=.5,
                  prepared_shared_pose_queries=_prepare(
                      data, "pipelined_field", "library_atom"),
                  pose_contract="shared_fp32_reduced_v1")
    return cpu.score_pairs(**kw)


@pytest.mark.parametrize("color", [False, True], ids=["shape", "combo"])
@pytest.mark.parametrize("layout", ["int16", "strided", "strided-int16"])
def test_library_layouts_score_as_their_float32_equivalent(layout, color):
    from test_cpu_shared_pose_contract import _inputs
    data = _inputs()
    library = data[3]
    if "int16" in layout:
        library = np.rint(library / np.float32(WIRE_QUANT_STEP)).astype(np.int16)
    equivalent = np.ascontiguousarray(library, dtype=np.float32)
    if library.dtype == np.int16:
        equivalent = equivalent * np.float32(WIRE_QUANT_STEP)
    if layout.startswith("strided"):
        library = np.ascontiguousarray(library.swapaxes(1, 2)).swapaxes(1, 2)
        assert not library.flags.c_contiguous
    expected = _scores(equivalent, data, color)
    actual = _scores(library, data, color)
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(actual, expected)

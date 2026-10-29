"""Color inside the field kernel.

The field is still the query's SHAPE field and there is still no color field.
The color *gradient* never needed one: it
is an analytic pair sum over a small same-type set, and it is evaluated in the
same loop that samples the shape texture.

Three properties are pinned here, and they are different in kind:

1. **A shape-only field screen is untouched.** The ``COLOR`` template parameter
   is 0 there and every line of the kernel it reaches is the line that shipped,
   which is what keeps the residency characterization and lane numbers
   valid for mode 1.
2. **``AT_POSE``'s shape channel is bit-identical to a shape-only field
   screen's.** This is the acceptance criterion, and it is a much
   sharper check on the implementation than any tolerance: the pose search at
   ``COLOR = 1`` is textually the ``COLOR = 0`` loop, so if a single bit moves,
   the color planes have leaked into the shape search.
3. **The field path's color agrees with the analytic path's.** Not bitwise --
   the color slots are split over the four cooperating lanes here and walked by
   one thread there, so the sums are grouped differently -- and at a pose the
   two paths do not share. Checked at ``steps = 0``, where both evaluate at the
   identity pose and only the grouping differs.

``mixing`` is never 0: at 0 the combination Tanimoto is exactly the shape
Tanimoto and the test has measured nothing about color.
"""

from __future__ import annotations

from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np
import pytest

from roshambo3 import backend, scores as S
from roshambo3.backend import ColorPlacement, PoseSearch
from tests.color_coordinate_comparison import (
    assert_color_row_matches,
    assert_cross_path_residual,
    color_denominator_diagnostics,
    color_oracle_row,
    color_row_bounds,
    snap_query_like_cuda,
)

needs_gpu = pytest.mark.skipif(not backend.available(),
                               reason="no CUDA extension in this build")

L = 16
N_REAL = 6
N_ROWS = 32
MIXING = 0.5

#: The canonical diagonal matrix: all widths 1.0 (the measured interaction
#: width matrix as all ones, which is what makes a per-type constant hoistable
#: at all), heights 1.0 on the diagonal and 0.0 off it.
N_FEATURES = 7
CANONICAL_R = np.ones((N_FEATURES, N_FEATURES), np.float32)
CANONICAL_P = np.diag([0.0] + [1.0] * (N_FEATURES - 1)).astype(np.float32)


def _rng_molecule(rng, n_real=N_REAL, n_color=4, jitter=0.0):
    """One molecule in the LIBRARY layout: atoms, shape padding, features."""
    coords = np.zeros((L, 4), np.float32)
    types = np.zeros(L, np.int32)
    coords[:n_real, :3] = rng.normal(0.0, 1.6, (n_real, 3)) + jitter
    coords[:n_real, 3] = 1.0
    # Contiguous, ascending in type -- which the generator asserts as a
    # postcondition and the same-type loop bound relies on -- and DOMINATED BY
    # ONE PLANE, which is the shape of the real distribution and the whole
    # reason a per-plane field is worth building: Hydrophobe is at mean
    # 11.47 of a mean 18.88 total and at 88.2% of all same-type pair work.
    ft = np.array([1]*(n_color - 1) + [2], np.int32)[:n_color]
    return coords, types, ft, rng.normal(0.0, 1.4, (n_color, 3)) + jitter


def _fixture(seed=7, n_color=4):
    """Matched color-built and shape-only arrays for the same molecules.

    The library layout puts features at the shape plane's PADDED width and the
    query layout puts them immediately after the real atoms; the two are
    deliberately different, because that asymmetry is what the loop bound has to
    derive rather than assume.
    """
    from roshambo3.layout import counts_from_types, tile32
    rng = np.random.default_rng(seed)

    L_shape, L_color = L, 8
    lib_c = np.zeros((N_ROWS, L_shape + L_color, 4), np.float32)
    lib_t = np.zeros((N_ROWS, L_shape + L_color), np.int32)
    shp_c = np.zeros((N_ROWS, L_shape, 4), np.float32)
    q_c = np.zeros((1, N_REAL + n_color, 4), np.float32)
    q_t = np.zeros((1, N_REAL + n_color), np.int32)
    qs_c = np.zeros((1, L_shape, 4), np.float32)

    for r in range(N_ROWS):
        coords, types, ft, fxyz = _rng_molecule(rng, jitter=0.35 * r)
        lib_c[r, :L_shape] = coords
        lib_t[r, :L_shape] = types
        lib_c[r, L_shape:L_shape + n_color, :3] = fxyz
        lib_c[r, L_shape:L_shape + n_color, 3] = 1.0
        lib_t[r, L_shape:L_shape + n_color] = ft
        shp_c[r] = coords
        if r == 0:
            q_c[0, :N_REAL] = coords[:N_REAL]
            q_c[0, N_REAL:N_REAL + n_color, :3] = fxyz
            q_c[0, N_REAL:N_REAL + n_color, 3] = 1.0
            q_t[0, N_REAL:N_REAL + n_color] = ft
            qs_c[0] = coords

    n_real = np.full(N_ROWS, N_REAL, np.int32)
    pairs_q = np.zeros(N_ROWS, np.int32)
    pairs_t = np.arange(N_ROWS, dtype=np.int32)
    # The coordinates are tiled, the count vector is not. This fixture's
    # library puts its features at the shape plane's PADDED width, exactly as a
    # corpus block does, so `library_color_offset` is L_shape and NOT the real
    # atom count -- the asymmetry the loop bound used to derive by scanning the
    # type plane is now the argument it is given.
    lib_cnt = counts_from_types(lib_t, n_real, CANONICAL_R.shape[0])
    return dict(
        color=dict(query_coords=q_c, query_types=q_t,
                   query_n_atoms=np.array([N_REAL], np.int32),
                   library_coords=tile32(lib_c), library_counts=lib_cnt,
                   library_color_offset=L_shape,
                   pair_query_row=pairs_q, pair_library_row=pairs_t,
                   interaction_r=CANONICAL_R, interaction_p=CANONICAL_P),
        shape=dict(query_coords=qs_c, query_types=np.zeros((1, L_shape), np.int32),
                   query_n_atoms=np.array([N_REAL], np.int32),
                   library_coords=tile32(shp_c),
                   library_counts=counts_from_types(
                       np.zeros((N_ROWS, L_shape), np.int32), n_real,
                       CANONICAL_R.shape[0]),
                   pair_query_row=pairs_q, pair_library_row=pairs_t,
                   interaction_r=CANONICAL_R, interaction_p=CANONICAL_P))


def _untile32(coords):
    """Recover the contiguous row layout from the kernel's 32-lane tile."""
    a = np.asarray(coords)
    n_pad, width = a.shape[:2]
    tail = a.shape[2:]
    return np.ascontiguousarray(
        a.reshape((n_pad // 32, width, 32) + tail)
        .transpose((0, 2, 1) + tuple(range(3, 3 + len(tail))))
        .reshape((n_pad, width) + tail)
    )


SHAPE_CHANNEL = ([S.TANIMOTO_SHAPE, S.OVERLAP_VOLUME, S.SELF_OVERLAP_QUERY,
                  S.SELF_OVERLAP_LIBRARY]
                 + list(range(9, 20)))


@needs_gpu
class TestTheFieldPathCarriesColor:

    def test_the_color_channel_is_not_zero(self):
        """The one failure mode that reads as success: a color sum that is
        computed, written, and exactly zero."""
        fx = _fixture()
        v = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                mixing=MIXING, steps=25, start_mode=1,
                                **fx["color"])
        # Not every row: the fixture walks the library away from the query, and
        # a far pair's color overlap underflows to exactly zero, correctly.
        assert np.count_nonzero(v[:, S.OVERLAP_COLOR]) >= 16
        assert v[:, S.TANIMOTO_COLOR].max() > 0.0

    def test_at_pose_shape_channel_is_bit_identical_at_the_evaluated_pose(self):
        """The acceptance criterion, at the pose where it is exactly checkable.

        At ``COLOR = 1`` the step loop is textually the ``COLOR = 0`` loop, so
        the color planes must not reach the shape search. At ``steps = 0`` there
        is no optimizer trajectory to amplify anything, so this is a bit-for-bit
        statement about the shape ARITHMETIC and a single moved bit means color
        reached it. Run it at every start mode, since the start set is the other
        thing the shape search owns.
        """
        fx = _fixture()
        for start_mode in (0, 1):
            shape = backend.score_pairs(pose_search=PoseSearch.FIELD, color=False,
                                        mixing=0.0, steps=0, start_mode=start_mode,
                                        **fx["shape"])
            color = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                        mixing=MIXING, steps=0, start_mode=start_mode,
                                        color_placement=ColorPlacement.AT_POSE,
                                        **fx["color"])
            a = shape[:, SHAPE_CHANNEL].view(np.uint32)
            b = color[:, SHAPE_CHANNEL].view(np.uint32)
            assert np.array_equal(a, b), (
                f"start_mode {start_mode}: "
                f"{int(np.count_nonzero(a != b))} of {a.size} shape-channel "
                f"words moved")

    def test_at_pose_shape_channel_tracks_a_shape_only_screen_over_a_trajectory(self):
        """The same criterion at the shipped 25 steps, where it is a BOUND.

        It is not a bit-identity assertion, and **not because any arithmetic
        changed**. The two arms are two
        instantiations of one kernel, and replacing the library type plane with
        a count word changed what ptxas spills and therefore which multiply-add
        it contracts into an FFMA: measured with `cuobjdump -res-usage`, the
        shipped color cell's stack frame moved from 104 B to 88 B at the
        identical 64-register budget. A contracted and an uncontracted FFMA
        differ in the last bit, and 25 Adagrad steps amplify that, which is the
        mechanism doc/NUMERICS_POLICY.md section 2 describes, also seen
        for an inlined-vs-called `adagrad_step`.

        Measured here: 2 of 480 shape-channel words move, max |dT_shape| =
        6.0e-8 and max |dV_AB| = 7.6e-6, on 1 of 32 conformers. That is four
        orders inside the epsilon budget (2e-3 with no overfetch). The
        bit-exact half of the criterion is the test above, at steps = 0.

        **The FIELD path is the only one where this had to be relaxed.** The
        analytic path's equivalents -- the three color structures against each
        other, and AT_POSE against a shape-only screen -- are bit-identical at
        the shipped 25 steps.
        """
        fx = _fixture()
        shape = backend.score_pairs(pose_search=PoseSearch.FIELD, color=False,
                                    mixing=0.0, steps=25, start_mode=1,
                                    **fx["shape"])
        color = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                    mixing=MIXING, steps=25, start_mode=1,
                                    color_placement=ColorPlacement.AT_POSE,
                                    **fx["color"])
        d = np.abs(shape[:, S.TANIMOTO_SHAPE].astype(np.float64)
                   - color[:, S.TANIMOTO_SHAPE].astype(np.float64))
        assert d.max() < 1e-5, f"max |dT_shape| {d.max():.3e}"
        # The ordering is what the product requires, and it does not move.
        assert np.array_equal(np.argsort(-shape[:, S.TANIMOTO_SHAPE], kind="stable"),
                              np.argsort(-color[:, S.TANIMOTO_SHAPE], kind="stable"))

    def test_in_gradient_moves_the_pose_and_at_pose_does_not(self):
        """The two placements must differ, or the combination objective is not
        being optimized and `mixing` has measured nothing."""
        fx = _fixture()
        common = dict(pose_search=PoseSearch.FIELD, color=True, mixing=MIXING,
                      steps=25, start_mode=1)
        at = backend.score_pairs(color_placement=ColorPlacement.AT_POSE,
                                 **common, **fx["color"])
        ing = backend.score_pairs(color_placement=ColorPlacement.IN_GRADIENT,
                                  **common, **fx["color"])
        assert np.any(at[:, S.TANIMOTO_SHAPE] != ing[:, S.TANIMOTO_SHAPE])
        # The combination objective must do at least as well as the shape
        # objective does ON the combination, on average -- that is what
        # optimizing it means.
        assert (ing[:, S.TANIMOTO_COMBINATION].mean()
                >= at[:, S.TANIMOTO_COMBINATION].mean())

    def _compare_coordinate_cell(self, grid_aligned):
        """Keep off-grid semantics, and compare each path at its own coordinates.

        FIELD publicly consumes the float32 query. ANALYTIC consumes the query
        reconstructed from the active compiled wire inverse and step. The
        independent oracle therefore computes FIELD's numerator and query
        self-overlap at the raw coordinates, then ANALYTIC's numerator and
        query self-overlap at the snapped coordinates. Both use the same
        float32 library record and its matching self-overlap.

        For off-grid input the raw cross-path delta is reported, not gated by
        the old 0.005 arithmetic allowance. Its independently computed
        representation component is removed first; only the remaining
        same-coordinate arithmetic residual is checked. For grid-aligned input
        the representation component is exactly zero and the existing 1e-5
        relative agreement contract remains in force.
        """
        from pathlib import Path
        from scripts.source_stamp import source_content_hash

        from roshambo3.backend import cuda as cuda_backend
        from roshambo3.format import COORD_SCALE

        fx = _fixture()
        extension = cuda_backend.extension()
        source = Path(__file__).resolve().parents[1] / "src/roshambo3/backend/_cuda_src"
        source_sha = source_content_hash(source)
        assert extension.__cuda_src_sha256__ == source_sha, (
            "loaded CUDA module was built from different CUDA source bytes: "
            f"{extension.__cuda_src_sha256__} != {source_sha}")
        step = np.float32(extension.compiled_wire_step())
        inverse = np.float32(extension.compiled_wire_inverse_step())
        assert step == np.float32(COORD_SCALE) == np.float32(WIRE_QUANT_STEP)
        assert inverse * step == np.float32(1.0)

        args = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in fx["color"].items()
        }
        if grid_aligned:
            for key in ("query_coords", "library_coords"):
                args[key][..., :3] = snap_query_like_cuda(
                    args[key][..., :3], step=step, inverse=inverse)

        n_real = int(args["query_n_atoms"][0])
        query_types = args["query_types"][0]
        query_types = query_types[query_types > 0]
        query_field = args["query_coords"][0, n_real:n_real + len(query_types), :3].copy()
        query_analytic = snap_query_like_cuda(query_field, step=step, inverse=inverse)
        if grid_aligned:
            assert np.array_equal(query_field, query_analytic)
        else:
            assert not np.array_equal(query_field, query_analytic), (
                "the public off-grid fixture no longer exercises ANALYTIC snapping")
            displacement = np.abs(query_field.astype(np.float64)
                                  - query_analytic.astype(np.float64))
            assert displacement.max() <= float(step) / 2 + 1e-6

        library = _untile32(args["library_coords"])
        counts = args["library_counts"]
        offset = int(args["library_color_offset"])
        library_xyz, library_types = [], []
        for row in range(len(counts)):
            types = np.repeat(np.arange(1, counts.shape[1], dtype=np.int32),
                              counts[row, 1:])
            library_types.append(types)
            library_xyz.append(library[row, offset:offset + len(types), :3].copy())

        common = dict(color=True, mixing=MIXING, steps=0, start_mode=0)
        field = backend.score_pairs(pose_search=PoseSearch.FIELD, **common, **args)
        analytic = backend.score_pairs(pose_search=PoseSearch.ANALYTIC,
                                       **common, **args)

        def observed(row):
            return {
                "overlap_color": float(row[S.OVERLAP_COLOR]),
                "self_color_query": float(row[S.SELF_COLOR_QUERY]),
                "self_color_library": float(row[S.SELF_COLOR_LIBRARY]),
                "tanimoto_color": float(row[S.TANIMOTO_COLOR]),
            }

        field_overlap = field[:, S.OVERLAP_COLOR].astype(np.float64)
        analytic_overlap = analytic[:, S.OVERLAP_COLOR].astype(np.float64)
        native_delta = field_overlap - analytic_overlap
        relative_native_delta = np.abs(native_delta) / np.maximum(
            np.abs(analytic_overlap), 1e-6)
        representation_delta = []
        maximum_condition = 0.0
        minimum_denominator = np.inf
        max_field_abs_error = 0.0
        max_analytic_abs_error = 0.0
        field_denominator_diagnostics = []
        analytic_denominator_diagnostics = []
        field_observed_denominator_residual = 0.0
        analytic_observed_denominator_residual = 0.0
        field_errors = {key: 0.0 for key in (
            "overlap_color", "self_color_query", "self_color_library", "tanimoto_color")}
        analytic_errors = dict(field_errors)
        field_bound_max = dict(field_errors)
        analytic_bound_max = dict(field_errors)
        field_residual_bound_ratio = dict(field_errors)
        analytic_residual_bound_ratio = dict(field_errors)
        maximum_cross_path_residual = 0.0
        maximum_cross_path_bound = 0.0
        maximum_tanimoto_bound = 0.0

        for i, (types, xyz) in enumerate(zip(library_types, library_xyz, strict=True)):
            field_expected = color_oracle_row(
                query_field, query_types, xyz, types,
                args["interaction_r"], args["interaction_p"], mixing=MIXING)
            analytic_expected = color_oracle_row(
                query_analytic, query_types, xyz, types,
                args["interaction_r"], args["interaction_p"], mixing=MIXING)
            field_bounds = color_row_bounds(
                query_field, query_types, xyz, types,
                args["interaction_r"], args["interaction_p"], mixing=MIXING,
                name="cuda_field_fixed_pose_color")
            analytic_bounds = color_row_bounds(
                query_analytic, query_types, xyz, types,
                args["interaction_r"], args["interaction_p"], mixing=MIXING,
                name="cuda_analytic_fixed_pose_color")
            field_report = assert_color_row_matches(
                observed(field[i]), field_expected, field_bounds)
            analytic_report = assert_color_row_matches(
                observed(analytic[i]), analytic_expected, analytic_bounds)
            cross_path = assert_cross_path_residual(
                field_overlap[i], analytic_overlap[i],
                field_expected["overlap_color"],
                analytic_expected["overlap_color"],
                field_bounds["overlap_color"],
                analytic_bounds["overlap_color"])
            maximum_cross_path_residual = max(
                maximum_cross_path_residual, abs(cross_path["residual"]))
            maximum_cross_path_bound = max(
                maximum_cross_path_bound, cross_path["absolute_bound"])
            field_conditioning = color_denominator_diagnostics(
                field_expected, field_bounds)
            analytic_conditioning = color_denominator_diagnostics(
                analytic_expected, analytic_bounds)
            for conditioning in (field_conditioning, analytic_conditioning):
                assert conditioning["denominator"] > conditioning["denominator_abs_error_bound"]
                assert conditioning["denominator_condition_number"] is not None
            field_native_denominator = (
                float(field[i, S.SELF_COLOR_QUERY])
                + float(field[i, S.SELF_COLOR_LIBRARY])
                - float(field[i, S.OVERLAP_COLOR]))
            analytic_native_denominator = (
                float(analytic[i, S.SELF_COLOR_QUERY])
                + float(analytic[i, S.SELF_COLOR_LIBRARY])
                - float(analytic[i, S.OVERLAP_COLOR]))
            field_observed_denominator_residual = max(
                field_observed_denominator_residual,
                abs(field_native_denominator - field_conditioning["denominator"]))
            analytic_observed_denominator_residual = max(
                analytic_observed_denominator_residual,
                abs(analytic_native_denominator - analytic_conditioning["denominator"]))
            assert abs(field_native_denominator - field_conditioning["denominator"]) <= (
                field_conditioning["denominator_abs_error_bound"])
            assert abs(analytic_native_denominator - analytic_conditioning["denominator"]) <= (
                analytic_conditioning["denominator_abs_error_bound"])

            representation_delta.append(
                field_expected["overlap_color"] - analytic_expected["overlap_color"])
            maximum_condition = max(
                maximum_condition,
                field_conditioning["denominator_condition_number"],
                analytic_conditioning["denominator_condition_number"])
            minimum_denominator = min(
                minimum_denominator,
                field_conditioning["denominator"],
                analytic_conditioning["denominator"])
            field_denominator_diagnostics.append(field_conditioning)
            analytic_denominator_diagnostics.append(analytic_conditioning)
            max_field_abs_error = max(
                max_field_abs_error,
                field_report["overlap_color"]["max_absolute_residual"])
            max_analytic_abs_error = max(
                max_analytic_abs_error,
                analytic_report["overlap_color"]["max_absolute_residual"])
            for key in field_errors:
                field_residual = field_report[key]["max_absolute_residual"]
                analytic_residual = analytic_report[key]["max_absolute_residual"]
                field_errors[key] = max(
                    field_errors[key], field_residual)
                analytic_errors[key] = max(
                    analytic_errors[key], analytic_residual)
                field_bound_max[key] = max(field_bound_max[key], field_bounds[key])
                analytic_bound_max[key] = max(analytic_bound_max[key],
                                              analytic_bounds[key])
                field_residual_bound_ratio[key] = max(
                    field_residual_bound_ratio[key],
                    field_residual / field_bounds[key])
                analytic_residual_bound_ratio[key] = max(
                    analytic_residual_bound_ratio[key],
                    analytic_residual / analytic_bounds[key])
            maximum_tanimoto_bound = max(
                maximum_tanimoto_bound, field_bounds["tanimoto_color"],
                analytic_bounds["tanimoto_color"])

        if grid_aligned:
            # Preserve the original same-coordinate tolerance and its absolute
            # floor for low-overlap rows.
            assert relative_native_delta.max() < 1e-5, (
                f"grid-aligned max relative overlap delta "
                f"{relative_native_delta.max():.3e} against 1e-5")
            assert np.max(np.abs(representation_delta)) == 0.0
        else:
            assert np.max(np.abs(representation_delta)) > 0.0
        assert np.isfinite(maximum_condition)
        assert minimum_denominator > 0.0
        assert max_field_abs_error >= 0.0 and max_analytic_abs_error >= 0.0
        return {
            "grid_aligned": bool(grid_aligned),
            "step": float(step),
            "inverse": float(inverse),
            "n_rows": int(len(field)),
            "query_max_displacement_angstrom": float(np.max(np.abs(
                query_field.astype(np.float64) - query_analytic.astype(np.float64)))),
            "native_max_abs_overlap_delta": float(np.max(np.abs(native_delta))),
            "native_max_relative_overlap_delta_floor_1e-6": float(
                np.max(relative_native_delta)),
            "fp64_max_representation_overlap_delta": float(np.max(np.abs(
                representation_delta))),
            "path_local_max_abs_residual": {
                "field": field_errors, "analytic": analytic_errors,
            },
            "path_local_max_error_bound": {
                "field": field_bound_max, "analytic": analytic_bound_max,
            },
            "path_local_max_residual_over_bound": {
                "field": field_residual_bound_ratio,
                "analytic": analytic_residual_bound_ratio,
            },
            "cross_path_arithmetic": {
                "maximum_absolute_residual_after_representation_removal":
                    maximum_cross_path_residual,
                "maximum_absolute_bound": maximum_cross_path_bound,
            },
            "denominator_conditioning": {
                "field": {
                    "minimum": min(x["denominator"] for x in field_denominator_diagnostics),
                    "maximum_condition_number": max(
                        x["denominator_condition_number"]
                        for x in field_denominator_diagnostics),
                    "maximum_absolute_error_bound": max(
                        x["denominator_abs_error_bound"]
                        for x in field_denominator_diagnostics),
                    "maximum_native_denominator_residual":
                        field_observed_denominator_residual,
                },
                "analytic": {
                    "minimum": min(x["denominator"] for x in analytic_denominator_diagnostics),
                    "maximum_condition_number": max(
                        x["denominator_condition_number"]
                        for x in analytic_denominator_diagnostics),
                    "maximum_absolute_error_bound": max(
                        x["denominator_abs_error_bound"]
                        for x in analytic_denominator_diagnostics),
                    "maximum_native_denominator_residual":
                        analytic_observed_denominator_residual,
                },
            },
            "minimum_color_denominator": float(minimum_denominator),
            "maximum_denominator_condition_number": float(maximum_condition),
            "maximum_tanimoto_absolute_bound": float(maximum_tanimoto_bound),
            "same_grid_1e-5_contract_pass": (
                bool(np.max(relative_native_delta) < 1e-5) if grid_aligned else None),
        }

    @pytest.mark.parametrize("grid_aligned", [False, True])
    def test_the_field_and_analytic_color_agree_at_the_identity_pose(self, grid_aligned):
        """The off-grid and same-grid cells use their separate contracts."""
        self._compare_coordinate_cell(grid_aligned)

    def test_a_shape_only_field_screen_is_untouched(self):
        """``COLOR = 0`` must reach the kernel that shipped. Pinned against the
        golden fixture elsewhere; here, against the analytic path's own answer at
        zero steps, which is the one pose both paths share exactly."""
        fx = _fixture()
        v = backend.score_pairs(pose_search=PoseSearch.FIELD, color=False,
                                mixing=0.0, steps=25, start_mode=1,
                                **fx["shape"])
        assert np.all(v[:, S.TANIMOTO_COLOR] == 0.0)
        assert np.all(v[:, S.OVERLAP_COLOR] == 0.0)
        assert np.array_equal(v[:, S.TANIMOTO_COMBINATION],
                              v[:, S.TANIMOTO_SHAPE])


@needs_gpu
class TestTheLaneAssignmentAndResidencyStillApply:
    """A color launch reaches the compiled start grouping and residency tier."""

    def test_start_groups_per_row_reaches_the_color_kernel(self):
        from roshambo3.backend import CoordResidency
        fx = _fixture()
        v = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                mixing=MIXING, steps=10, start_mode=1,
                                start_groups_per_row=1,
                                coord_residency=CoordResidency.SHARED,
                                **fx["color"])
        assert np.count_nonzero(v[:, S.OVERLAP_COLOR]) >= 16

    def test_the_shared_residency_tier_reaches_the_color_kernel(self):
        from roshambo3.backend import CoordResidency
        fx = _fixture()
        v = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                mixing=MIXING, steps=10, start_mode=1,
                                coord_residency=CoordResidency.SHARED,
                                **fx["color"])
        assert np.count_nonzero(v[:, S.OVERLAP_COLOR]) >= 16


@needs_gpu
class TestThePerPlaneColorField:
    """A field for the one color plane whose query count can pay for one.

    The threshold is applied per (query, plane) on the host, so the two arms are
    ONE binary separated by a runtime argument -- which is what makes the
    comparison between them a causal ablation of the method rather than of a
    rebuild.
    """

    #: The fixture's queries carry three features of type 1 and one of type 2,
    #: so a threshold of 3 serves exactly the dominant plane and 99 serves none
    #: -- which is the split the real distribution has (one plane at mean
    #: 11.47 and five totalling ~7.4).
    SERVED, UNSERVED = 3, 99

    def _common(self, fx):
        return dict(pose_search=PoseSearch.FIELD, color=True, mixing=MIXING,
                    color_placement=ColorPlacement.IN_GRADIENT, **fx["color"])

    def test_the_field_is_read_only_inside_the_optimizer_loop(self):
        """At zero steps the pose never moves, and the reported color overlap is
        the EXACT analytic pair sum at that pose -- so the two
        arms must agree BIT FOR BIT.

        This is the sharp statement of what the field does and does not touch:
        it is an approximation to the pose search and to nothing that is
        reported. If a single bit moves here, the field has leaked into the
        reported score and the accuracy gate would be a gate on the wrong thing.
        """
        fx = _fixture()
        c = self._common(fx)
        off = backend.score_pairs(steps=0, start_mode=0,
                                  color_field_min_sources=0, **c)
        on = backend.score_pairs(steps=0, start_mode=0,
                                 color_field_min_sources=self.SERVED,
                                 color_field_spacing_a=0.6, **c)
        assert np.array_equal(off.view(np.uint32), on.view(np.uint32))

    def test_the_field_moves_the_pose_and_stays_close(self):
        """With steps > 0 it must change the trajectory -- otherwise it is not
        being read at all -- and must land near the analytic arm."""
        fx = _fixture()
        c = self._common(fx)
        off = backend.score_pairs(steps=25, start_mode=1,
                                  color_field_min_sources=0, **c)
        on = backend.score_pairs(steps=25, start_mode=1,
                                 color_field_min_sources=self.SERVED,
                                 color_field_spacing_a=0.6, **c)
        assert np.any(off[:, S.TANIMOTO_COMBINATION]
                      != on[:, S.TANIMOTO_COMBINATION])
        live = off[:, S.OVERLAP_COLOR] > 1e-3
        d = np.abs(off[live, S.TANIMOTO_COMBINATION]
                   - on[live, S.TANIMOTO_COMBINATION])
        assert d.max() < 0.15, f"max |dT| {d.max():.4f}"

    def test_a_threshold_no_plane_clears_is_the_analytic_arm(self):
        """Bit-identical, because no field was built at all."""
        fx = _fixture()
        c = self._common(fx)
        off = backend.score_pairs(steps=25, start_mode=1,
                                  color_field_min_sources=0, **c)
        none = backend.score_pairs(steps=25, start_mode=1,
                                   color_field_min_sources=self.UNSERVED,
                                   color_field_spacing_a=0.6, **c)
        assert np.array_equal(off.view(np.uint32), none.view(np.uint32))

    def test_it_is_refused_where_it_would_never_be_read(self):
        """A color field is read from inside the optimizer loop only. At
        AT_POSE the color term is evaluated once at the converged pose by the
        exact analytic sum, and the analytic kernel has no texture at all --
        so naming it there is a refusal, not a silent no-op."""
        fx = _fixture()
        with pytest.raises(ValueError, match="IN_GRADIENT"):
            backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                mixing=MIXING, steps=25,
                                color_placement=ColorPlacement.AT_POSE,
                                color_field_min_sources=self.SERVED,
                                **fx["color"])
        with pytest.raises(ValueError, match="IN_GRADIENT"):
            backend.score_pairs(pose_search=PoseSearch.ANALYTIC, color=True,
                                mixing=MIXING, steps=25,
                                color_placement=ColorPlacement.IN_GRADIENT,
                                color_field_min_sources=self.SERVED,
                                **fx["color"])

    def test_the_shape_channel_gate_still_holds_with_a_color_field(self):
        """The color field is a declared numerics change for its own plane and
        for nothing else. AT_POSE cannot carry one at all, so the acceptance
        criterion is untouched -- asserted here so a future change that let a
        color field into the shape search fails at this line."""
        fx = _fixture()
        # steps = 0, for the reason the AT_POSE gate above gives: at the shipped
        # 25 the criterion is a bound rather than an equality, and the
        # bit-exact statement lives where there is no trajectory to amplify.
        shape = backend.score_pairs(pose_search=PoseSearch.FIELD, color=False,
                                    mixing=0.0, steps=0, start_mode=1,
                                    **fx["shape"])
        color = backend.score_pairs(pose_search=PoseSearch.FIELD, color=True,
                                    mixing=MIXING, steps=0, start_mode=1,
                                    color_placement=ColorPlacement.AT_POSE,
                                    **fx["color"])
        assert np.array_equal(shape[:, SHAPE_CHANNEL].view(np.uint32),
                              color[:, SHAPE_CHANNEL].view(np.uint32))


class TestTheLatticePolicyFollowsTheMolecules:
    """`L` is two different things to two different readers.

    To anything sizing the record the kernel reads -- the staged buffer, the
    pinned pool -- `L` is the padded width and is correct. To the per-bucket
    query-field policy it was a proxy for HOW BIG THE MOLECULES ARE, and a color
    plane joining the padded width silently broke that proxy.

    No GPU: this is a policy resolution, not a launch.
    """

    def test_a_color_block_resolves_the_same_lattice_as_its_shape_twin(self):
        from roshambo3.engine import ScreenConfig
        cfg = ScreenConfig()
        shape = cfg.for_width(32)
        # The same molecules, color-built: L = 32 shape + 40 color.
        color = cfg.for_width(72, L_shape=32)
        assert color["field_spacing_a"] == shape["field_spacing_a"]
        assert color["min_blocks_per_sm"] == shape["min_blocks_per_sm"]
        # And the defect this replaced: resolving from the padded width alone
        # gives a different lattice for the same molecules.
        assert cfg.for_width(72)["field_spacing_a"] != shape["field_spacing_a"]

    def test_a_shape_block_is_unchanged(self):
        from roshambo3.engine import ScreenConfig
        cfg = ScreenConfig()
        for L in (16, 24, 32, 40, 48, 56, 64, 88):
            assert cfg.for_width(L) == cfg.for_width(L, L_shape=L)

    def test_no_L_dependent_knob_reaches_a_color_launch_through_the_padded_width(self):
        """The invariant behind the blast-radius audit, pinned.

        `for_width` has exactly three `L`-dependent outputs. For a COLOR launch
        none of them may depend on the color plane's width: the two field
        policies now resolve from `L_shape`, and `stage_width` is forced to 0
        because staging is shape-only. If a fourth `L`-dependent output is ever
        added, this fails and whoever adds it has to say what it does to a color
        screen.
        """
        from roshambo3.engine import ScreenConfig
        for L_color in (16, 40, 64):
            a = ScreenConfig(color=True).for_width(32, L_shape=32)
            b = ScreenConfig(color=True).for_width(32 + L_color, L_shape=32)
            assert a == b, f"a color launch moved with L_color={L_color}"
            assert b["stage_width"] == 0

    def test_an_explicit_spacing_still_wins(self):
        from roshambo3.engine import ScreenConfig
        cfg = ScreenConfig(field_spacing_a=0.35, min_blocks_per_sm=2)
        a = cfg.for_width(72, L_shape=32)
        assert a["field_spacing_a"] == 0.35
        assert a["min_blocks_per_sm"] == 2

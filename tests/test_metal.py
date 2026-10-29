"""Actual Apple GPU tests; set ROSHAMBO3_REQUIRE_METAL=1 to forbid skips."""

import os
from types import SimpleNamespace

from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np
import pytest

from roshambo3.backend import metal
from roshambo3.backend.spec import ColorPlacement
from roshambo3.layout import tile32
from roshambo3.reference.color import score_columns
from roshambo3.scores import COLUMN_INDEX


@pytest.fixture
def ctx():
    if not metal.available():
        if os.environ.get("ROSHAMBO3_REQUIRE_METAL") == "1":
            pytest.fail("Apple GPU is required for this test run")
        pytest.skip("Apple GPU unavailable")
    c = metal.create_context(factor_color_torque=False)
    yield c
    c.close()


def fixture(planar=False, wire=False, general=False):
    rng = np.random.default_rng(574)
    q = np.zeros((2, 12, 4), np.float32)
    q[:, :8, :3] = rng.normal(size=(2, 8, 3))
    q[:, :8, 3] = 1
    qt = np.zeros((2, 12), np.int32)
    qt[:, 5:8] = [1, 1, 2]
    qn = np.array([5, 5], np.int32)
    b = np.zeros((32, 16, 3 if wire else 4), np.int16 if wire else np.float32)
    coords = np.round(rng.normal(size=(7, 8, 3)), 3).astype(np.float32)
    if wire:
        encoded = np.rint(coords / WIRE_QUANT_STEP).astype(np.int16)
        b[:7, :8, :3] = encoded
        # The independent oracle must score the decoded format-3 payload,
        # which is the representation consumed by the Metal scorer.
        coords = encoded.astype(np.float32) * np.float32(WIRE_QUANT_STEP)
    else:
        b[:7, :8, :3] = coords
        b[:7, :8, 3] = 1
    counts = np.tile(np.array([5, 2, 1], np.uint8), (7, 1))
    r = np.diag([0, 1, 1]).astype(np.float32)
    h = r.copy()
    if general:
        r[1, 2] = r[2, 1] = 0.7
        h[1, 2] = h[2, 1] = 0.4
    return dict(
        query_coords=q,
        query_types=qt,
        query_n_atoms=qn,
        library_coords=tile32(b, planar=planar),
        library_counts=counts,
        pair_query_row=np.repeat(np.arange(2, dtype=np.int32), 7),
        pair_library_row=np.tile(np.arange(7, dtype=np.int32), 2),
        interaction_r=r,
        interaction_p=h,
        color=True,
        mixing=0.5,
        library_color_offset=5,
        library_tile_mode=-1 if planar else 1,
    ), coords


@pytest.mark.parametrize(
    "wire,planar,general",
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
        (False, True, True),
    ],
)
def test_fixed_pose_against_independent_oracle(ctx, wire, planar, general):
    a, b = fixture(planar, wire, general)
    rng = np.random.default_rng(7)
    poses = rng.normal(size=(14, 7)).astype(np.float32)
    rows = metal.score_pairs(**a, context=ctx, poses=poses)
    names = list(COLUMN_INDEX)[:9]
    for i, row in enumerate(rows):
        qi = i // 7
        bi = i % 7
        want = score_columns(
            a["query_coords"][qi, :5, :3],
            b[bi, :5],
            a["query_coords"][qi, 5:8, :3],
            [1, 1, 2],
            b[bi, 5:8],
            [1, 1, 2],
            a["interaction_r"],
            a["interaction_p"],
            0.5,
            poses[i, :4],
            poses[i, 4:],
        )
        for name in names:
            np.testing.assert_allclose(
                row[COLUMN_INDEX[name]], want[name], rtol=1e-5, atol=2e-6
            )


def test_rescore_preserves_shape_search_and_context_reuse(ctx):
    a, _ = fixture()
    rescore = metal.score_pairs(**a, context=ctx)
    key = ctx.query_key
    repeat = metal.score_pairs(**a, context=ctx)
    np.testing.assert_array_equal(rescore, repeat)
    assert ctx.query_key == key
    shape = metal.score_pairs(**dict(a, color=False, mixing=0), context=ctx)
    np.testing.assert_array_equal(
        rescore[:, [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15]],
        shape[:, [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15]],
    )
    combo = metal.score_pairs(
        **a, context=ctx, color_placement=ColorPlacement.IN_GRADIENT
    )
    assert np.all(np.isfinite(combo))
    np.testing.assert_allclose(np.linalg.norm(combo[:, 9:13], axis=1), 1, atol=1e-5)
    replay = metal.score_pairs(**a, context=ctx, poses=combo[:, 9:16].copy())
    np.testing.assert_allclose(combo[:, :9], replay[:, :9], rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("level", ["conformer", "molecule"])
def test_device_topk_ties_large_ids_split_molecules(ctx, level):
    rng = np.random.default_rng(18)
    sink = metal.make_topk(k=9, level=level, rank_col=0, context=ctx)
    sink.prepare(2, 0)
    allrows = [[], []]
    for size, base in [(37, 2**40), (19, 2**40 + 37), (65, 2**40 + 56)]:
        rows = rng.normal(size=(2 * size, 20)).astype(np.float32)
        rows[:, 0] = rng.integers(0, 4, 2 * size)
        mids = (np.arange(size) + base) // 3
        block = SimpleNamespace(mol_id=mids, n_conformers=size)
        sink.before_block(block)
        for qi in range(2):
            allrows[qi].extend(
                (row.copy(), base + i, int(mids[i]))
                for i, row in enumerate(rows[qi * size : (qi + 1) * size])
            )
        # Alternate the explicit query-major pair list and the dense
        # (no pair list) form; both must fold identically.
        dense = size == 19
        sink.ingest(
            ctx.put("test_scores", rows),
            None if dense else np.repeat(np.arange(2, dtype=np.int32), size),
            None if dense else np.tile(np.arange(size, dtype=np.int32), 2),
            size,
            base,
        )
        sink.add_block(block, None)
    for (rows, gids), candidates in zip(sink.finalize(), allrows):
        candidates.sort(key=lambda x: (-x[0][0], x[1]))
        if level == "molecule":
            seen = set()
            candidates = [x for x in candidates if not (x[2] in seen or seen.add(x[2]))]
        candidates = candidates[:9]
        np.testing.assert_array_equal(gids, [x[1] for x in candidates])
        np.testing.assert_array_equal(rows, np.array([x[0] for x in candidates]))


@pytest.mark.parametrize("mode", ["shape", "rescore", "combo"])
def test_public_screen_dense_and_engine_reuse(ctx, tmp_path, mode):
    from roshambo3.api import score_matrix, screen
    from roshambo3.engine import Engine, ScreenConfig
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    from tests.codec_fixtures import make_codec_fixture
    from tests.test_cpu_worker_owned_scan import _query_set

    fx = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    q = _query_set(fx, (0, 4))
    config = ScreenConfig(
        color=mode != "shape",
        mixing=0.5 if mode != "shape" else 0,
        color_placement=ColorPlacement.IN_GRADIENT
        if mode == "combo"
        else ColorPlacement.AT_POSE,
    )
    generator = generator_from_descriptor(fx.corpus.color)
    engine = Engine(backend="metal", stage_timing=True)
    for level in ["molecule", "conformer"]:
        first = screen(
            q,
            fx.corpus,
            engine=engine,
            config=config,
            k=3,
            level=level,
            color_generator=generator,
            workers=1,
            prefetch_depth=0,
            launch_conformers=32,
        )
        second = screen(
            q,
            fx.corpus,
            engine=engine,
            config=config,
            k=3,
            level=level,
            color_generator=generator,
            workers=1,
            prefetch_depth=0,
            launch_conformers=64,
        )
        for a, b in zip(first.hits, second.hits):
            np.testing.assert_array_equal(a.gids, b.gids)
            np.testing.assert_array_equal(a.scores, b.scores)
        assert first.stats.device_stages["gpu_s"] >= 0
    engine.close()
    dense = score_matrix(
        q, fx.corpus, backend="metal", config=config, color_generator=generator
    )
    assert dense.scores.shape[0] == q.n_conformers


def test_prefiltered_field_values_and_gradients_against_differences(ctx):
    a, _ = fixture()
    metal.score_pairs(**a, context=ctx, steps=0, field_spacing_a=0.5)
    metadata, field = ctx.grids[(0.5, 8.0)]
    records = metadata.read(np.dtype("uint32"), [2, 8])
    total = sum(int(np.prod(rec[4:7].astype(np.uint64))) for rec in records)
    values = field.read(np.dtype("float32"), [total, 4])
    scale = float(np.float32(0.77687365))
    for qi, rec in enumerate(records):
        origin = rec[:3].view(np.float32).astype(np.float64)
        h = float(rec[3:4].view(np.float32)[0])
        dims = rec[4:7]
        points = (a["query_coords"][qi, :5, :3] * np.float32(scale)).astype(np.float64)

        def gaussian(p):
            return np.exp2(-np.sum((points - p) ** 2, axis=1)).sum()

        eps = 0.002
        axes = np.eye(3) * eps

        def prefiltered(p):
            lap = sum(
                gaussian(p + d) + gaussian(p - d) - 2 * gaussian(p) for d in axes
            ) / (eps * eps)
            return gaussian(p) - h * h / 12 * lap

        for atom in points:
            xyz = np.rint((atom - origin) / h).astype(int)
            pos = origin + h * xyz
            index = (
                int(rec[7]) + xyz[0] + int(dims[0]) * (xyz[1] + int(dims[1]) * xyz[2])
            )
            expected = [prefiltered(pos)] + [
                (prefiltered(pos + d) - prefiltered(pos - d)) / (2 * eps) for d in axes
            ]
            np.testing.assert_allclose(values[index], expected, rtol=3e-5, atol=1e-5)


def test_fast_exp2_is_search_only(ctx):
    a, _ = fixture(general=True)
    poses = np.tile(np.array([1, 0.2, -0.1, 0.3, 0.1, -0.2, 0.3], np.float32), (14, 1))
    control = metal.create_context(fast_search_exp2=False)
    try:
        fast = metal.score_pairs(**a, context=ctx, poses=poses)
        precise = metal.score_pairs(**a, context=control, poses=poses)
        np.testing.assert_array_equal(fast, precise)
        fast_combo = metal.score_pairs(
            **a, context=ctx, color_placement=ColorPlacement.IN_GRADIENT
        )
        accurate = metal.score_pairs(
            **a, context=control, poses=fast_combo[:, 9:16].copy()
        )
        np.testing.assert_allclose(
            fast_combo[:, :9], accurate[:, :9], rtol=1e-5, atol=2e-6
        )
    finally:
        control.close()


def test_failed_allocation_invalidates_cache_and_allows_retry(ctx, monkeypatch):
    a, _ = fixture()
    original = ctx.buffer

    def fail(name, size):
        if name == "grid_field":
            raise MemoryError("injected field budget failure")
        return original(name, size)

    with monkeypatch.context() as patch:
        patch.setattr(ctx, "buffer", fail)
        with pytest.raises(MemoryError, match="injected"):
            metal.score_pairs(**a, context=ctx)
    assert ctx.query_key is None and ctx.grids == {}
    rows = metal.score_pairs(**a, context=ctx)
    assert np.all(np.isfinite(rows))


def test_configuration_never_probes_cuda(monkeypatch):
    from dataclasses import replace

    from roshambo3.backend import cuda
    from roshambo3.engine import ScreenConfig
    from roshambo3.engine.query import from_arrays

    monkeypatch.setattr(cuda, "extension", lambda: pytest.fail("Metal queried CUDA"))
    cfg = ScreenConfig()
    metal.screen_args(cfg, 32, 32, 32)
    from_arrays([np.zeros((4, 3), np.float32)], ["test"], backend="metal")
    with pytest.raises(NotImplementedError, match="coord_residency"):
        metal.screen_args(replace(cfg, coord_residency=2), 32, 32, 32)


@pytest.mark.parametrize(
    "change",
    [
        dict(start_mode=3),
        dict(mixing=2),
        dict(field_spacing_a=float("nan")),
        dict(query_n_atoms=np.array([0, 5], np.int32)),
        dict(pair_library_row=np.array([100] * 14, np.int32)),
    ],
)
def test_invalid_inputs_fail_before_gpu_allocation(monkeypatch, change):
    a, _ = fixture()
    a.update(change)
    monkeypatch.setattr(
        metal, "create_context", lambda *a: pytest.fail("allocated before validation")
    )
    with pytest.raises((ValueError, NotImplementedError)):
        metal.score_pairs(**a)


def test_extreme_coordinates_never_convert_to_invalid_grid_indices(ctx):
    a, _ = fixture()
    # Valid FP32 input far outside the query field must sample zero, without
    # overflowing an integer address or reading an unrelated node.
    raw = np.zeros((32, 16, 4), np.float32)
    raw[:7, :8, :3] = np.float32(1e20)
    a["library_coords"] = tile32(raw)
    with np.errstate(over="ignore"):
        rows = metal.score_pairs(**a, context=ctx, start_mode=0)
    assert np.all(np.isfinite(rows))
    np.testing.assert_array_equal(rows[:, :5], np.zeros((14, 5), np.float32))


def test_close_releases_query_buffer_references(ctx):
    a, _ = fixture()
    metal.score_pairs(**a, context=ctx)
    assert hasattr(ctx, "q")
    ctx.close()
    assert ctx.closed and ctx.native is None and not ctx.buffers and not ctx.grids
    assert not hasattr(ctx, "q")


@pytest.mark.parametrize("sampler", ["texture16", "texture32"])
def test_native_texture_centres_interpolation_and_zero_borders(ctx, sampler):
    if sampler == "texture32" and not ctx.info["float32_filtering"]:
        pytest.skip("RGBA32Float filtering is unavailable")
    c = metal.create_context(field_sampler=sampler)
    try:
        # Dyadic sample fractions avoid assuming undocumented fractional-weight
        # precision. Non-cubic dimensions and a nonzero field offset catch axis
        # and query-row mistakes. Channels vary independently.
        dims = (3, 4, 5)
        nodes = np.array(
            [[x, y, z, 1] for z in range(5) for y in range(4) for x in range(3)],
            np.float32,
        )
        records = np.zeros((2, 8), np.uint32)
        records[:, :4] = np.array([0, 0, 0, 1], np.float32).view(np.uint32)
        records[:, 4:7] = dims
        records[1, 7] = 7
        values = np.concatenate([np.zeros((7, 4), np.float32), nodes])
        points = np.array(
            [
                [0, 0, 0, 0],
                [2, 3, 4, 0],
                [0.25, 1.5, 2.75, 0],
                [-0.25, 0, 0, 0],
                [2.5, 3, 4, 0],
                [-1, 0, 0, 0],
                [3, 4, 5, 0],
                [1e30, 0, 0, 0],
            ],
            np.float32,
        )
        want = np.array(
            [
                [0, 0, 0, 1],
                [2, 3, 4, 1],
                [0.25, 1.5, 2.75, 1],
                [0, 0, 0, 0.75],
                [1, 1.5, 2, 0.5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            np.float32,
        )
        tex = c.native.texture(*dims, sampler == "texture16")
        grid = c.put("test_grid", records)
        field = c.put("test_field", values)
        positions = c.put("test_positions", points)
        error = c.put("test_error", metal._u(0))
        out = c.buffer("test_samples", len(points) * 16)
        c.native.submit(
            [
                (
                    "pack_field",
                    [grid, field, error],
                    metal._u(1, sampler == "texture16"),
                    len(nodes),
                    128,
                    [tex],
                ),
                (
                    "probe_texture",
                    [positions, grid, out],
                    metal._u(1, len(points)),
                    len(points),
                    128,
                    [tex],
                ),
            ]
        )
        assert error.read(np.dtype("uint32"), [1])[0] == 0
        np.testing.assert_allclose(
            out.read(np.dtype("float32"), want.shape), want, atol=3e-6, rtol=2e-6
        )
    finally:
        c.close()


@pytest.mark.parametrize("sampler", ["texture16", "texture32"])
def test_texture_optimizer_replay_multiquery_and_cache_lifetime(ctx, sampler):
    if sampler == "texture32" and not ctx.info["float32_filtering"]:
        pytest.skip("RGBA32Float filtering is unavailable")
    c = metal.create_context(field_sampler=sampler)
    try:
        a, _ = fixture()
        rows = metal.score_pairs(
            **a, context=c, color_placement=ColorPlacement.IN_GRADIENT
        )
        assert len(c.textures) == 2
        textures = c.textures.copy()
        again = metal.score_pairs(
            **a, context=c, color_placement=ColorPlacement.IN_GRADIENT
        )
        np.testing.assert_array_equal(rows, again)
        assert c.textures == textures
        replay = metal.score_pairs(**a, context=ctx, poses=rows[:, 9:16].copy())
        np.testing.assert_allclose(rows[:, :9], replay[:, :9], rtol=1e-6, atol=1e-6)
        shape = metal.score_pairs(**dict(a, color=False, mixing=0), context=c)
        rescore = metal.score_pairs(**a, context=c)
        np.testing.assert_array_equal(shape[:, 9:16], rescore[:, 9:16])
        changed = dict(a, field_spacing_a=0.5)
        metal.score_pairs(**changed, context=c)
        assert len(c.grids) == 1
        assert c.textures != textures
    finally:
        c.close()
    assert not c.textures and not c.grids


def test_texture_limits_and_format_range_failure_are_recoverable(ctx):
    with pytest.raises(ValueError, match="dimensions"):
        ctx.native.texture(2049, 2, 2, True)
    c = metal.create_context(field_sampler="texture16")
    try:
        a, _ = fixture()
        # A huge spacing keeps the origin next to the query at the first node
        # while making the prefilter unrepresentable in half storage.
        a["query_coords"] = np.zeros_like(a["query_coords"])
        with pytest.raises(FloatingPointError, match="texture format"):
            metal.score_pairs(**a, context=c, field_spacing_a=1e5, field_pad_a=1e-3)
        assert c.query_key is None and not c.textures
        good, _ = fixture()
        assert np.isfinite(metal.score_pairs(**good, context=c)).all()
    finally:
        c.close()


def test_opt_in_kernel_profiling_preserves_results(ctx):
    a, _ = fixture()
    expected = metal.score_pairs(**a, context=ctx)
    c = metal.create_context(profile_kernels=True)
    try:
        np.testing.assert_array_equal(metal.score_pairs(**a, context=c), expected)
        stats = c.timings()
        assert stats["profiling_separate_commands"]
        assert stats["kernel_gpu_s"]["optimize"] > 0
        assert abs(sum(stats["kernel_gpu_s"].values()) - stats["gpu_s"]) < 1e-8
    finally:
        c.close()


def test_engine_texture_options_are_owned_and_survive_close(ctx):
    from roshambo3.engine import Engine

    options = dict(field_sampler="texture16")
    e = Engine(backend="metal", backend_options=options)
    options["field_sampler"] = "invalid"
    try:
        first = e._context()
        assert first.field_sampler == "texture16"
        e.close()
        second = e._context()
        assert first is not second
        assert second.field_sampler == "texture16"
    finally:
        e.close()


@pytest.mark.parametrize("sampler", ["manual", "texture16"])
@pytest.mark.parametrize("case", ["diagonal", "inactive", "general"])
def test_diagonal_specialization_matches_existing_optimizer(ctx, sampler, case):
    a, _ = fixture(wire=True, planar=True, general=case == "general")
    if case == "diagonal":
        a["interaction_r"][1, 1], a["interaction_r"][2, 2] = 0.7, 1.3
        a["interaction_p"][1, 1], a["interaction_p"][2, 2] = 0.4, 1.2
    elif case == "inactive":
        a["interaction_r"][1, 1] = 0
        a["interaction_p"][2, 2] = 0
    baseline = metal.create_context(field_sampler=sampler, factor_color_torque=False)
    candidate = metal.create_context(
        field_sampler=sampler,
        specialize_color=True,
        factor_color_torque=False,
        profile_kernels=True,
    )
    try:
        for start_mode in (0, 1, 2):
            expected = metal.score_pairs(
                **a,
                context=baseline,
                start_mode=start_mode,
                color_placement=ColorPlacement.IN_GRADIENT,
            )
            actual = metal.score_pairs(
                **a,
                context=candidate,
                start_mode=start_mode,
                color_placement=ColorPlacement.IN_GRADIENT,
            )
            np.testing.assert_array_equal(actual, expected)
        selected = "optimize" if case == "general" else "optimize_diagonal"
        assert selected in candidate.timings()["kernel_gpu_s"]
        assert candidate.timings()["specialize_color"]
    finally:
        baseline.close()
        candidate.close()


@pytest.mark.parametrize("specialize", [False, True])
@pytest.mark.parametrize("general", [False, True])
def test_factored_torque_has_accurate_final_scores_and_generic_fallback(
    ctx, specialize, general
):
    a, _ = fixture(wire=True, general=general)
    # This regression pins the original FP32 torque-comparison policy; half
    # interpolation can amplify reassociation into a different optimizer basin.
    baseline = metal.create_context(field_sampler="manual", factor_color_torque=False)
    candidate = metal.create_context(
        field_sampler="manual",
        specialize_color=specialize, factor_color_torque=True, profile_kernels=True
    )
    try:
        for start_mode in (0, 1, 2):
            expected = metal.score_pairs(
                **a,
                context=baseline,
                start_mode=start_mode,
                color_placement=ColorPlacement.IN_GRADIENT,
            )
            actual = metal.score_pairs(
                **a,
                context=candidate,
                start_mode=start_mode,
                color_placement=ColorPlacement.IN_GRADIENT,
            )
            np.testing.assert_allclose(
                actual[:, :9], expected[:, :9], rtol=2e-5, atol=2e-5
            )
            replay = metal.score_pairs(**a, context=baseline, poses=actual[:, 9:16].copy())
            np.testing.assert_allclose(
                actual[:, :9], replay[:, :9], rtol=1e-6, atol=1e-6
            )
        name = (
            "optimize_diagonal_factored"
            if specialize and not general
            else "optimize_factored"
        )
        assert name in candidate.timings()["kernel_gpu_s"]
        assert candidate.timings()["factor_color_torque"]
        # AT_POSE/shape kernels must not acquire the candidate's reassociation.
        np.testing.assert_array_equal(
            metal.score_pairs(**a, context=candidate),
            metal.score_pairs(**a, context=baseline),
        )
    finally:
        candidate.close()
        baseline.close()


def test_factored_torque_handles_empty_feature_segments(ctx):
    a, _ = fixture()
    a["library_counts"][:, 1:] = 0
    candidate = metal.create_context(specialize_color=True, factor_color_torque=True)
    try:
        expected = metal.score_pairs(
            **a, context=ctx, color_placement=ColorPlacement.IN_GRADIENT
        )
        actual = metal.score_pairs(
            **a, context=candidate, color_placement=ColorPlacement.IN_GRADIENT
        )
        np.testing.assert_array_equal(actual, expected)
    finally:
        candidate.close()


@pytest.mark.parametrize("sampler", ["manual", "texture16", "texture32"])
@pytest.mark.parametrize("wire,planar", [(False, False), (True, True)])
def test_factored_directed_color_against_nine_column_oracle(ctx, sampler, wire, planar):
    if sampler == "texture32" and not ctx.info["float32_filtering"]:
        pytest.skip("RGBA32Float filtering unavailable")
    a, coords = fixture(wire=wire, planar=planar, general=True)
    a["interaction_r"][2, 1] = 1.1
    a["interaction_p"][2, 1] = 0.9
    c = metal.create_context(field_sampler=sampler, factor_color_torque=True)
    try:
        result = metal.score_pairs(
            **a, context=c, color_placement=ColorPlacement.IN_GRADIENT
        )
        for i, row in enumerate(result):
            qi, bi = i // 7, i % 7
            oracle = score_columns(
                a["query_coords"][qi, :5, :3],
                coords[bi, :5],
                a["query_coords"][qi, 5:8, :3],
                [1, 1, 2],
                coords[bi, 5:8],
                [1, 1, 2],
                a["interaction_r"],
                a["interaction_p"],
                0.5,
                row[9:13],
                row[13:16],
            )
            for name in list(COLUMN_INDEX)[:9]:
                np.testing.assert_allclose(
                    row[COLUMN_INDEX[name]], oracle[name], rtol=1e-5, atol=2e-6
                )
    finally:
        c.close()


@pytest.mark.parametrize("mixing", [0.0, 1.0])
def test_factored_torque_mixing_endpoints_and_zero_step(ctx, mixing):
    a, _ = fixture()
    a["mixing"] = mixing
    baseline = metal.create_context(factor_color_torque=False)
    candidate = metal.create_context(factor_color_torque=True)
    try:
        # No update uses the final gradient, so factoring must preserve the
        # selected start and complete row bit for bit at zero steps.
        kwargs = dict(color_placement=ColorPlacement.IN_GRADIENT, steps=0)
        np.testing.assert_array_equal(
            metal.score_pairs(**a, context=baseline, **kwargs),
            metal.score_pairs(**a, context=candidate, **kwargs),
        )
        actual = metal.score_pairs(
            **a, context=candidate, color_placement=ColorPlacement.IN_GRADIENT
        )
        replay = metal.score_pairs(**a, context=baseline, poses=actual[:, 9:16].copy())
        np.testing.assert_allclose(actual[:, :9], replay[:, :9], rtol=1e-6, atol=1e-6)
    finally:
        baseline.close()
        candidate.close()


def test_default_optimizer_policy_matches_explicit_specialization_and_keeps_control(
    ctx,
):
    from bench.metal import VARIANTS
    from roshambo3.engine import Engine

    a, _ = fixture()
    e = Engine(backend="metal")
    control = metal.create_context(**VARIANTS["baseline"])
    explicit = metal.create_context(specialize_color=True, factor_color_torque=True)
    try:
        default = e._context()
        assert default.factor_color_torque and default.specialize_color
        assert not control.factor_color_torque and not control.specialize_color
        assert explicit.factor_color_torque and explicit.specialize_color
        kwargs = dict(color_placement=ColorPlacement.IN_GRADIENT)
        np.testing.assert_array_equal(
            metal.score_pairs(**a, context=default, **kwargs),
            metal.score_pairs(**a, context=explicit, **kwargs),
        )
        np.testing.assert_array_equal(
            metal.score_pairs(**a, context=control, **kwargs),
            metal.score_pairs(**a, context=ctx, **kwargs),
        )
    finally:
        e.close()
        control.close()
        explicit.close()


@pytest.mark.parametrize("active_offdiagonal", [False, True])
def test_color_range_detection_uses_height_and_dispatches_specialization(
    ctx, active_offdiagonal
):
    a, coords = fixture()
    a["interaction_r"][:] = 1
    if active_offdiagonal:
        a["interaction_p"][2, 1] = 0.6
    legacy = metal.create_context(color_range_policy="width", specialize_color=True)
    active = metal.create_context(color_range_policy="active", specialize_color=True)
    try:
        kwargs = dict(color_placement=ColorPlacement.IN_GRADIENT)
        old = metal.score_pairs(**a, context=legacy, **kwargs)
        new = metal.score_pairs(**a, context=active, **kwargs)
        assert legacy.last_optimizer_kernel == "optimize_factored"
        expected = (
            "optimize_factored" if active_offdiagonal else "optimize_diagonal_factored"
        )
        assert active.last_optimizer_kernel == expected
        np.testing.assert_array_equal(new, old)
        # Independent physical score check at every candidate pose.
        for i, row in enumerate(new):
            qi, bi = i // 7, i % 7
            oracle = score_columns(
                a["query_coords"][qi, :5, :3],
                coords[bi, :5],
                a["query_coords"][qi, 5:8, :3],
                [1, 1, 2],
                coords[bi, 5:8],
                [1, 1, 2],
                a["interaction_r"],
                a["interaction_p"],
                0.5,
                row[9:13],
                row[13:16],
            )
            for name in list(COLUMN_INDEX)[:9]:
                np.testing.assert_allclose(
                    row[COLUMN_INDEX[name]], oracle[name], rtol=1e-5, atol=2e-6
                )
    finally:
        legacy.close()
        active.close()


@pytest.mark.parametrize("sampler", ["manual", "texture16", "texture32"])
def test_dense_color_planes_mixed_dispatch_cache_and_exact_final_scores(ctx, sampler):
    a, _ = fixture()
    ctx.factor_color_torque = True
    # Query 0 has two type-1 sources; query 1 has two type-2 sources.
    a["query_types"][1, 5] = 2
    candidate = metal.create_context(
        color_field_min_sources=2, color_field_sampler=sampler
    )
    try:
        kwargs = dict(color_placement=ColorPlacement.IN_GRADIENT)
        result = metal.score_pairs(**a, context=candidate, **kwargs)
        assert candidate.last_optimizer_kernel == "mixed"
        assert [d["kernel"] for d in candidate.last_optimizer_dispatches] == [
            "optimize_color_fields_m2", "optimize_color_fields_m4"]
        assert [(p["query_row"], p["plane"]) for p in candidate.last_color_fields] == [
            (0, 1),
            (1, 2),
        ]
        cache = candidate.color_cache
        np.testing.assert_array_equal(
            result, metal.score_pairs(**a, context=candidate, **kwargs)
        )
        assert candidate.color_cache is cache
        replay = metal.score_pairs(**a, context=ctx, poses=result[:, 9:16].copy())
        np.testing.assert_allclose(result[:, :9], replay[:, :9], rtol=1e-6, atol=1e-6)
        # Threshold zero and below-threshold planes retain the analytic kernel.
        for minimum in (0, 3):
            expected = metal.score_pairs(**a, context=ctx, **kwargs)
            actual = metal.score_pairs(
                **a, context=candidate, color_field_min_sources=minimum, **kwargs
            )
            np.testing.assert_array_equal(actual, expected)
            assert not candidate.last_color_fields
            assert candidate.last_optimizer_kernel == "optimize_diagonal_factored"
        # Shape/rescore do not build or sample color fields.
        metal.score_pairs(**a, context=candidate)
        assert not candidate.last_color_fields
        assert candidate.last_optimizer_kernel == "optimize"
    finally:
        candidate.close()


@pytest.mark.parametrize("sampler", ["manual", "texture16", "texture32"])
def test_color_field_values_and_gradients_against_independent_gaussians(ctx, sampler):
    a, _ = fixture()
    a["interaction_r"][1, 1] = 0.7
    a["interaction_p"][1, 1] = 0.8
    candidate = metal.create_context(
        color_field_min_sources=2,
        color_field_types=(1,),
        color_field_spacing_a=0.1,
        color_field_sampler=sampler,
    )
    try:
        metal.score_pairs(
            **a,
            context=candidate,
            color_placement=ColorPlacement.IN_GRADIENT,
            steps=0,
            field_pad_a=2,
        )
        assert candidate.last_optimizer_kernel == "optimize_color_fields_m2"
        _, meta, field, textures, slots, _, _ = candidate.color_cache[1]
        positions = np.zeros((4, 4), np.float32)
        positions[:, :3] = np.array(
            [[0.1, 0.2, -0.2], [0.5, -0.1, 0.3], [-0.3, 0.4, 0.1], [0.2, 0.2, 0.2]]
        )
        pos = candidate.put("probe_color_positions", positions)
        output = candidate.buffer("probe_color_output", 4 * 16)
        index = 1  # query 0, plane 1
        if sampler == "manual":
            job = ("probe_field", [pos, meta, field, output], metal._u(index, 4), 4, 64)
        else:
            job = (
                "probe_texture",
                [pos, meta, output],
                metal._u(index, 4),
                4,
                64,
                [textures[int(slots.read(np.dtype("uint32"), [1], index * 4)[0])]],
            )
        candidate.native.submit([job])
        actual = output.read(np.dtype("float32"), [4, 4])
        scale = float(np.float32(0.77687365))
        sources = a["query_coords"][0, 5:7, :3].astype(np.float64) * scale
        delta = positions[:, None, :3].astype(np.float64) - sources
        beta = 0.7 / (2 * scale * scale)
        amplitude = 0.8**2 / 0.7**1.5
        values = amplitude * np.exp(-beta * (delta * delta).sum(axis=2))
        expected = np.c_[
            values.sum(axis=1), (-2 * beta * values[:, :, None] * delta).sum(axis=1)
        ]
        # Bound approximation error separately from field-construction and
        # sampling correctness. Linear interpolation is still approximate even
        # with the mean-error Laplacian prefilter.
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-3 if sampler == "manual" else 2e-3
        )
        if sampler == "manual":
            record = meta.read(np.dtype("uint32"), [6, 8])[index]
            origin_h = record[:4].view(np.float32).astype(np.float64)
            dims = record[4:7].astype(np.int64)
            interpolated = np.zeros((len(positions), 4), np.float64)
            for pi, position in enumerate(positions[:, :3]):
                v = (position - origin_h[:3]) / origin_h[3]
                lo, frac = np.floor(v).astype(np.int64), v - np.floor(v)
                for z in range(2):
                    for y in range(2):
                        for x in range(2):
                            node = lo + [x, y, z]
                            d = origin_h[:3] + origin_h[3] * node - sources
                            d2 = (d * d).sum(axis=1)
                            val = amplitude * np.exp(-beta * d2)
                            f = np.r_[
                                val.sum(), (-2 * beta * val[:, None] * d).sum(axis=0)
                            ]
                            lap = np.r_[
                                (val * (4 * beta**2 * d2 - 6 * beta)).sum(),
                                (
                                    val[:, None]
                                    * (20 * beta**2 - 8 * beta**3 * d2)[:, None]
                                    * d
                                ).sum(axis=0),
                            ]
                            corrected = f - origin_h[3] ** 2 / 12 * lap
                            address = int(
                                record[7]
                                + node[0]
                                + dims[0] * (node[1] + dims[1] * node[2])
                            )
                            stored = field.read(np.dtype("float32"), [4], address * 16)
                            np.testing.assert_allclose(
                                stored, corrected, rtol=1e-5, atol=3e-6
                            )
                            weight = np.prod(np.where([x, y, z], frac, 1 - frac))
                            interpolated[pi] += weight * corrected
            np.testing.assert_allclose(actual, interpolated, rtol=1e-5, atol=3e-6)
    finally:
        candidate.close()


def test_color_fields_general_matrix_budget_and_outside_bounds_fall_back(ctx):
    ctx.factor_color_torque = True
    for general, options in [(True, {}), (False, {"color_field_max_bytes": 1})]:
        a, _ = fixture(general=general)
        c = metal.create_context(color_field_min_sources=2, **options)
        try:
            expected = metal.score_pairs(
                **a, context=ctx, color_placement=ColorPlacement.IN_GRADIENT
            )
            actual = metal.score_pairs(
                **a, context=c, color_placement=ColorPlacement.IN_GRADIENT
            )
            np.testing.assert_array_equal(actual, expected)
            assert not c.last_color_fields
        finally:
            c.close()
    # A nonzero color interaction outside a deliberately small plane volume
    # must run analytically, not turn into a zero-border field sample.
    a, _ = fixture()
    a["query_coords"][:, 5:8, :3] = 0
    # Float interleaved tile32: flattening the untiled view is unnecessary.
    b = (
        a["library_coords"]
        .reshape(1, 16, 32, 4)
        .transpose(0, 2, 1, 3)
        .reshape(32, 16, 4)
        .copy()
    )
    b[:7, 5:8, :3] = [1, 0, 0]
    a["library_coords"] = tile32(b)
    c = metal.create_context(color_field_min_sources=2)
    try:
        kwargs = dict(
            color_placement=ColorPlacement.IN_GRADIENT,
            field_pad_a=0.1,
            steps=1,
            start_mode=0,
        )
        expected = metal.score_pairs(**a, context=ctx, **kwargs)
        actual = metal.score_pairs(**a, context=c, **kwargs)
        assert c.last_color_fields
        np.testing.assert_array_equal(actual, expected)
    finally:
        c.close()


@pytest.mark.parametrize(
    "options",
    [
        {"color_field_min_sources": -1},
        {"color_field_min_sources": True},
        {"color_field_spacing_a": 0},
        {"color_field_spacing_a": float("nan")},
        {"color_field_types": (0,)},
        {"color_field_types": (1, 1)},
        {"color_field_sampler": "unsupported"},
        {"color_field_max_bytes": 0},
    ],
)
def test_invalid_color_field_context_options_are_rejected(options):
    with pytest.raises(ValueError):
        metal.create_context(**options)


def test_color_field_screen_configuration_and_failure_recovery(ctx, monkeypatch):
    from dataclasses import replace
    from roshambo3.engine import ScreenConfig

    cfg = replace(ScreenConfig(), color_field_min_sources=12, color_field_spacing_a=0.3)
    kwargs = metal.screen_args(cfg, 32, 32, 32)
    assert kwargs["color_field_min_sources"] == 12
    assert kwargs["color_field_spacing_a"] == 0.3
    assert "color_field_min_sources" not in metal.screen_args(ScreenConfig(), 32, 32, 32)
    disabled = metal.screen_args(replace(ScreenConfig(), color_field_min_sources=0), 32, 32, 32)
    assert disabled["color_field_min_sources"] == 0
    spacing_only = metal.screen_args(replace(ScreenConfig(), color_field_spacing_a=0.2), 32, 32, 32)
    assert spacing_only["color_field_spacing_a"] == 0.2
    a, _ = fixture()
    c = metal.create_context(color_field_min_sources=2)
    original = c.buffer
    try:

        def fail(name, size):
            if name == "color_grid_field":
                raise MemoryError("injected color allocation failure")
            return original(name, size)

        with monkeypatch.context() as patch:
            patch.setattr(c, "buffer", fail)
            with pytest.raises(MemoryError, match="injected"):
                metal.score_pairs(
                    **a, context=c, color_placement=ColorPlacement.IN_GRADIENT
                )
        assert c.color_cache is None and c.query_key is None
        actual = metal.score_pairs(
            **a, context=c, color_placement=ColorPlacement.IN_GRADIENT
        )
        assert c.last_color_fields and np.all(np.isfinite(actual))
        # Changing the query invalidates the retained plane cache.
        cache = c.color_cache
        a["query_coords"][0, 5, 0] += 0.2
        metal.score_pairs(**a, context=c, color_placement=ColorPlacement.IN_GRADIENT)
        assert c.color_cache is not cache
    finally:
        c.close()


def test_nonfinite_color_field_nodes_fail_explicitly_and_recover(ctx):
    a, _ = fixture()
    c = metal.create_context(color_field_min_sources=2)
    try:
        a["interaction_r"][1, 1] = 1e20
        a["interaction_p"][1, 1] = 1e10
        with pytest.raises(FloatingPointError, match="nonfinite nodes"):
            metal.score_pairs(
                **a, context=c, color_placement=ColorPlacement.IN_GRADIENT, steps=0
            )
        assert c.query_key is None and c.color_cache is None
        a["interaction_r"][1, 1] = a["interaction_p"][1, 1] = 1
        result = metal.score_pairs(
            **a, context=c, color_placement=ColorPlacement.IN_GRADIENT
        )
        assert c.color_field_build_gpu_s > 0
        metal.score_pairs(**a, context=c, poses=result[:, 9:16].copy())
        assert (
            not c.last_color_fields
            and c.color_field_build_gpu_s == c.color_field_build_wall_s == 0
        )
    finally:
        c.close()


def test_default_half_fields_dispatch_and_explicit_policy_match(ctx, monkeypatch):
    from roshambo3.engine import Engine

    # The default policy must not require Apple9 float32 filtering.
    native = metal.extension()
    info = dict(native.device_info(0), float32_filtering=False)
    monkeypatch.setattr(metal, "extension", lambda: SimpleNamespace(
        device_info=lambda device: info, Context=native.Context))
    a, _ = fixture()
    rng = np.random.default_rng(8472)
    q = np.zeros((2, 20, 4), np.float32)
    q[:, :5] = a["query_coords"][:, :5]
    q[:, 5:17, :3] = rng.normal(size=(2, 12, 3))
    q[:, 5:17, 3] = 1
    types = np.zeros((2, 20), np.int32)
    types[:, 5:17] = 1
    a.update(query_coords=q, query_types=types)
    default = Engine(backend="metal")
    explicit = metal.create_context(field_sampler="texture16",
        color_field_sampler="texture16", color_field_min_sources=12,
        color_field_spacing_a=0.25, color_field_types=(1, 2))
    try:
        c = default._context()
        kwargs = dict(color_placement=ColorPlacement.IN_GRADIENT)
        actual = metal.score_pairs(**a, context=c, **kwargs)
        expected = metal.score_pairs(**a, context=explicit, **kwargs)
        np.testing.assert_array_equal(actual, expected)
        assert c.last_optimizer_kernel == "optimize_color_fields_m2"
        assert len(c.last_color_fields) == 2
        cache = c.color_cache
        np.testing.assert_array_equal(actual, metal.score_pairs(**a, context=c, **kwargs))
        assert c.color_cache is cache
        replay = metal.score_pairs(**a, context=ctx, poses=actual[:, 9:16].copy())
        np.testing.assert_allclose(actual[:, :9], replay[:, :9], rtol=1e-6, atol=1e-6)
        # Direct-call zero remains an explicit analytic override.
        metal.score_pairs(**a, context=c, color_field_min_sources=0, **kwargs)
        assert not c.last_color_fields
        assert c.last_optimizer_kernel == "optimize_diagonal_factored"
    finally:
        default.close()
        explicit.close()


@pytest.mark.parametrize("shape_sampler", ["manual", "texture16"])
def test_compact_color_bindings_and_split_dispatch_preserve_pairs_and_topk(ctx, monkeypatch, shape_sampler):
    a, _ = fixture()
    # Old reservation used 33*(3+1)=132 slots, even with only plane 1 served.
    a["query_coords"] = np.repeat(a["query_coords"][:1], 33, axis=0)
    a["query_types"] = np.repeat(a["query_types"][:1], 33, axis=0)
    a["query_n_atoms"] = np.repeat(a["query_n_atoms"][:1], 33)
    order = np.random.default_rng(715).permutation(66)
    a["pair_query_row"] = np.repeat(np.arange(33, dtype=np.int32), 2)[order]
    a["pair_library_row"] = np.tile(np.array([0, 1], np.int32), 33)[order]
    c = metal.create_context(field_sampler=shape_sampler, color_field_min_sources=2,
                             color_field_max_bytes=1 << 30)
    try:
        kw = dict(color_placement=ColorPlacement.IN_GRADIENT, steps=2, field_pad_a=2)
        expected = metal.score_pairs(**a, context=c, **kw)
        assert len(c.last_color_fields) == 33
        assert len(c.color_cache[1][3]) == 33  # No placeholders for type 0 or 2.
        assert len(c.last_optimizer_dispatches) == 1
        assert c.last_optimizer_dispatches[0]["bound_textures"] == (66 if shape_sampler != "manual" else 33)
        cache = c.color_cache
        monkeypatch.setattr(metal, "_TEXTURE_BINDING_LIMIT", 8)
        actual = metal.score_pairs(**a, context=c, **kw)
        np.testing.assert_array_equal(actual, expected)
        assert c.color_cache is cache
        assert len(c.last_optimizer_dispatches) > 1
        assert all(d["bound_textures"] <= 8 for d in c.last_optimizer_dispatches)
        assert c.last_optimizer_kernel == "optimize_color_fields_m2"
        # Device selection requires a complete query-major library block.
        full = dict(a, pair_query_row=np.repeat(np.arange(33, dtype=np.int32), 7),
                    pair_library_row=np.tile(np.arange(7, dtype=np.int32), 33))
        monkeypatch.setattr(metal, "_TEXTURE_BINDING_LIMIT", 128)
        full_reference = metal.score_pairs(**full, context=c, **kw)
        monkeypatch.setattr(metal, "_TEXTURE_BINDING_LIMIT", 8)
        base = 2**40
        sink = metal.make_topk(k=1, level="conformer", rank_col=0, context=c)
        sink.prepare(33, 0)
        metal.score_pairs(**full, context=c, topk=sink, gid_base=base, **kw)
        for qi, (rows, gids) in enumerate(sink.finalize()):
            indices = np.flatnonzero(full["pair_query_row"] == qi)
            best = min(indices, key=lambda i: (-float(full_reference[i, 0]), int(full["pair_library_row"][i])))
            np.testing.assert_array_equal(rows[0], full_reference[best])
            assert gids.tolist() == [base + int(full["pair_library_row"][best])]
    finally:
        c.close()


def test_shape_ensemble_over_128_splits_without_repeating_library_self(ctx):
    a, _ = fixture()
    n = 129
    a["query_coords"] = np.repeat(a["query_coords"][:1], n, axis=0)
    a["query_types"] = np.repeat(a["query_types"][:1], n, axis=0)
    a["query_n_atoms"] = np.repeat(a["query_n_atoms"][:1], n)
    a["pair_query_row"] = np.arange(n, dtype=np.int32)
    a["pair_library_row"] = np.arange(n, dtype=np.int32) % 7
    c = metal.create_context()
    try:
        calls = []
        original = c.native
        class RecordingContext:
            def __getattr__(self, name):
                return getattr(original, name)
            def submit(self, jobs):
                calls.extend(job[0] for job in jobs)
                return original.submit(jobs)
        c.native = RecordingContext()
        kw = dict(color_placement=ColorPlacement.IN_GRADIENT, steps=1, field_pad_a=1)
        actual = metal.score_pairs(**a, context=c, **kw)
        assert [d["bound_textures"] for d in c.last_optimizer_dispatches] == [128, 1]
        assert sum(d["pairs_dispatched"] for d in c.last_optimizer_dispatches) == n
        assert calls.count("library_self") == 1
        assert calls.count("optimize_diagonal_factored") == 2
        one = dict(a, query_coords=a["query_coords"][:1].copy(),
                   query_types=a["query_types"][:1].copy(),
                   query_n_atoms=a["query_n_atoms"][:1].copy(),
                   pair_query_row=np.zeros(n, np.int32))
        expected = metal.score_pairs(**one, context=c, **kw)
        np.testing.assert_array_equal(actual, expected)
    finally:
        c.close()


def test_budget_limited_rows_use_separate_analytic_kernel(ctx):
    a, _ = fixture()
    _, nodes = metal._color_grid_record(a["query_coords"][0, 5:7, :3], 0.25, 2, 0)
    c = metal.create_context(color_field_min_sources=2, color_field_types=(1,),
                             color_field_max_bytes=nodes * 24)
    try:
        kw = dict(color_placement=ColorPlacement.IN_GRADIENT, steps=2, field_pad_a=2)
        actual = metal.score_pairs(**a, context=c, **kw)
        assert [p["query_row"] for p in c.last_color_fields] == [0]
        assert [d["field_mask"] for d in c.last_optimizer_dispatches] == [2, 0]
        assert [d["kernel"] for d in c.last_optimizer_dispatches] == [
            "optimize_color_fields_m2", "optimize_diagonal_factored"]
        assert c.last_optimizer_kernel == "mixed"
        assert sum(d["pairs_dispatched"] for d in c.last_optimizer_dispatches) == len(a["pair_query_row"])
        analytic = metal.score_pairs(**a, context=c, color_field_min_sources=0, **kw)
        np.testing.assert_array_equal(actual[7:], analytic[7:])
    finally:
        c.close()

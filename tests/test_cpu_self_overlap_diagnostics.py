"""CPU self-overlap counters and authenticated format-3 bindings."""
import numpy as np
import pytest

from roshambo3.backend import cpu

cpu.extension()  # Missing native code is a test failure, never a green skip.


@pytest.mark.parametrize("threads", [1, 2])
def test_optional_native_self_diagnostics_preserve_exact_results(threads):
    from roshambo3.backend import cpu
    from roshambo3.reference.color import MODE_COMBO_OPTIMAL
    from test_cpu_shared_pose_contract import _inputs, _prepare
    q, qt, qn, library, counts, r, p = data = _inputs()
    prepared = _prepare(data)
    kwargs = dict(query_coords=q, query_types=qt, query_n_atoms=qn,
                  library_coords=library, library_counts=counts,
                  pair_query_row=np.array([0, 1, 0], dtype=np.int32),
                  pair_library_row=np.arange(3, dtype=np.int32),
                  interaction_r=r, interaction_p=p, color=True, color_mode=MODE_COMBO_OPTIMAL,
                  color_grain=16, steps=2, start_mode=1, mixing=0.5,
                  prepared_shared_pose_queries=prepared,
                  pose_contract="shared_fp32_reduced_v1", threads=threads)
    baseline = cpu.score_pairs(**kwargs)
    timing = {}
    instrumented = cpu.score_pairs(**kwargs, library_self_timing=timing)
    np.testing.assert_array_equal(baseline, instrumented)
    assert timing["library_shape_self_evaluations"] == 3
    assert timing["library_color_self_evaluations"] == 3
    assert timing["library_shape_self_seconds"] > 0
    assert timing["library_color_self_seconds"] > 0


@pytest.mark.parametrize("threads", [1, 2])
def test_format3_native_handle_binds_exact_rows_and_evaluates_no_library_self(
        tmp_path, threads):
    from roshambo3.backend import cpu
    from roshambo3.engine.r3g_cpu import R3gWorkerSelfCache
    from tests.test_self_overlap_routes import build_inputs, _inputs

    corpus, base, generator = build_inputs(tmp_path / "corpus")
    source, query, _ = _inputs(corpus, base, generator, "combo")
    descriptor = source.cpu_block_descriptors()[0]
    decoded = source.decode_cpu_block(descriptor)
    cache = R3gWorkerSelfCache(source, "a" * 64)
    handle, reason, hit, replaced = cache.block(
        cpu, descriptor, decoded, decoded.result.coords, query.coords, query.types,
        query.n_real, query.interaction_r, query.interaction_p)
    assert reason is None and not hit and not replaced
    assert handle._native.library_shape_from_stored_volume
    assert handle._native.library_color_from_sidecar
    assert handle.library.shape_terms == handle.library.color_terms == 0
    n = descriptor.n_conf
    kwargs = dict(query_coords=query.coords, query_types=query.types,
        query_n_atoms=query.n_real, library_coords=decoded.result.coords,
        library_counts=decoded.result.counts,
        pair_query_row=np.zeros(n, dtype=np.int32),
        pair_library_row=np.arange(n, dtype=np.int32),
        interaction_r=query.interaction_r, interaction_p=query.interaction_p,
        color=True, color_mode="combo_optimal", color_grain=16, steps=2,
        start_mode=1, mixing=0.5, threads=threads,
        pose_contract="shared_fp32_reduced_v1", prepared_self_overlaps=handle)
    counts_only, timers = {}, {}
    baseline = cpu.score_pairs(**kwargs, library_self_counts=counts_only)
    instrumented = cpu.score_pairs(**kwargs, library_self_timing=timers)
    np.testing.assert_array_equal(baseline, instrumented)
    assert counts_only == {"library_shape_self_evaluations": 0,
                           "library_color_self_evaluations": 0}
    assert all(value == 0 for value in timers.values())
    with pytest.raises(ValueError, match="stale|interaction matrix"):
        cpu.score_pairs(**dict(kwargs,
            interaction_r=query.interaction_r * np.float32(1.01)))
    cache.release_block(descriptor)

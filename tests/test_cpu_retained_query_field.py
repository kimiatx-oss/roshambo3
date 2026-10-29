"""The retained query field on the CPU shape route: reuse, refusal, GIL.

Product checks extracted from the mode-screening tests.
"""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import copy

import numpy as np
import pytest

from roshambo3.backend import cpu
from tests.cpu_retained_field import FAST_SHAPE, shape_then_rescore
from tests.codec_fixtures import make_codec_fixture
from tests.test_cpu_worker_owned_scan import _query_set, _source

cpu.extension()  # Required native code is a test prerequisite, never a skip.


def test_rescore_preserves_shape_winning_pose_and_shape_values(tmp_path):
    fx = make_codec_fixture(tmp_path/"corpus", color_layout="merged")
    source, query = _source(fx.corpus), _query_set(fx)
    d = source.cpu_block_descriptors()[0]
    raw = source.decode_cpu_block(d)
    coords, counts, vaa = raw.result.coords, raw.result.counts, raw.result.vaa
    shape = shape_then_rescore(cpu, query, coords, counts, vaa)
    rescore = shape_then_rescore(cpu, query, coords, counts, vaa, rescore=True)
    np.testing.assert_array_equal(shape[:,9:20], rescore[:,9:20])
    np.testing.assert_array_equal(shape[:,1], rescore[:,1])
    assert np.isfinite(rescore).all()
    diagnostic, phases = shape_then_rescore(cpu, query, coords, counts, vaa,
                                                   rescore=True, diagnostic=True)
    np.testing.assert_array_equal(diagnostic, rescore)
    assert phases["shape_phases"] and phases["color_phases"]


@pytest.mark.parametrize("rescore", [False, True])
def test_reused_query_field_exact_rows_and_concurrent_lifetime(tmp_path, rescore):
    fx = make_codec_fixture(tmp_path/"corpus", color_layout="merged")
    source, query = _source(fx.corpus), _query_set(fx)
    d = source.cpu_block_descriptors()[0]
    raw = source.decode_cpu_block(d)
    coords, counts, vaa = raw.result.coords, raw.result.counts, raw.result.vaa
    fields = cpu.prepare_combo_shape_evaluators(query.coords, query.n_real,
        shape_backend="pipelined_field", shape_grain="library_atom", shape_width=8,
        field_spacing_angstrom=.5, field_padding_angstrom=8.)
    # Reused fields must reproduce the fresh route exactly under the same final
    # start policy. The prepared-field route defaults to
    # winner selection while the fresh route scores every start, so the exact
    # comparison pins all_start; the default winner policy is then checked for
    # thread-safe reuse against itself.
    expected = shape_then_rescore(cpu, query, coords, counts, vaa,
        rescore=rescore, final_start_policy="all_start")
    def score(policy):
        return shape_then_rescore(cpu, query, coords, counts, vaa,
            rescore=rescore, query_fields=fields, final_start_policy=policy)
    winner = score(None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, rows in enumerate(pool.map(score, ["all_start", None] * 2)):
            np.testing.assert_array_equal(rows, expected if i % 2 == 0 else winner)
    diagnostic, _ = shape_then_rescore(cpu, query, coords, counts, vaa,
        rescore=rescore, query_fields=fields, diagnostic=True,
        final_start_policy="all_start")
    np.testing.assert_array_equal(diagnostic, expected)
    changed = copy(query)
    changed.coords = query.coords.copy()
    changed.coords[0, 0, 0] += .125
    with pytest.raises(ValueError, match="binding mismatch"):
        shape_then_rescore(cpu, changed, coords, counts, vaa, query_fields=fields)
    changed = copy(query)
    changed.n_real = query.n_real.copy()
    changed.n_real[0] -= 1
    with pytest.raises(ValueError, match="binding mismatch"):
        shape_then_rescore(cpu, changed, coords, counts, vaa, query_fields=fields)


@pytest.mark.parametrize("overrides, message", [
    ({"field_h":0.}, "requires the legacy field"),
    ({"field_h":.6}, "binding mismatch"),
    ({"field_pad":9.}, "binding mismatch"),
    ({"field_pipeline":False}, "pipelined register-resident"),
    ({"field_width":4}, "need field_h > 0 with field_width=8"),
])
def test_prepared_query_field_refuses_unsupported_or_mismatched_route(tmp_path, overrides, message):
    fx = make_codec_fixture(tmp_path/"corpus", color_layout="merged")
    query = _query_set(fx)
    fields = cpu.prepare_combo_shape_evaluators(query.coords, query.n_real,
        shape_backend="pipelined_field", shape_grain="library_atom", shape_width=8,
        field_spacing_angstrom=.5, field_padding_angstrom=8.)
    knobs = dict(FAST_SHAPE, **overrides)
    with pytest.raises(ValueError, match=message):
        cpu.score_pairs(query_coords=query.coords, query_n_atoms=query.n_real,
            library_coords=query.coords, library_counts=np.array([[int(query.n_real[0])]], np.uint8),
            pair_query_row=np.array([0], np.int32), pair_library_row=np.array([0], np.int32),
            prepared_query_fields=fields, **knobs)


@pytest.mark.parametrize("rescore", [False, True])
def test_native_numerical_call_releases_gil(tmp_path, rescore):
    fx = make_codec_fixture(tmp_path/"corpus", color_layout="merged")
    source, query = _source(fx.corpus), _query_set(fx)
    d = source.cpu_block_descriptors()[0]
    raw = source.decode_cpu_block(d)
    coords, counts, vaa = raw.result.coords, raw.result.counts, raw.result.vaa
    repeat = 10000 if rescore else 100
    coords, counts, vaa = (np.tile(coords, (repeat,1,1)),
        np.tile(counts, (repeat,1)), np.tile(vaa, repeat))
    if rescore:
        shapes = shape_then_rescore(cpu, query, coords[:d.n_conf], raw.result.counts,
                                    vaa[:d.n_conf])
        shapes = np.tile(shapes, (repeat,1))
        pq = np.zeros(len(counts), np.int32)
        pl = np.arange(len(counts), dtype=np.int32)
        call = lambda: cpu.extension().rescore_scalar_color(query.coords, query.types,
            query.n_real, coords, counts, pq, pl, query.interaction_r,
            query.interaction_p, shapes, .5, False, False, False, 0)
    else:
        # The native scorer itself, on arrays already in its layout, so no
        # Python-side numpy copy (which also releases the GIL) can satisfy
        # the probe on the scorer's behalf.
        from roshambo3.optimizer import lr_t_for_steps
        native = dict(query_coords=np.ascontiguousarray(query.coords, np.float32),
            query_n_atoms=np.ascontiguousarray(query.n_real, np.int32),
            library_coords=np.ascontiguousarray(coords, np.float32),
            library_counts=np.ascontiguousarray(counts, np.uint8),
            pair_query_row=np.zeros(len(counts), np.int32),
            pair_library_row=np.arange(len(counts), dtype=np.int32),
            steps=25, lr_q=.1, lr_t=lr_t_for_steps(25), start_mode=1,
            threads=1,
            **FAST_SHAPE)
        call = lambda: cpu.extension().score_pairs(**native)

    def probe():
        """True if another Python thread ran while `call` was in native code."""
        ready, begin, progressed = (threading.Event(), threading.Event(),
                                    threading.Event())

        def observer():
            ready.set()
            begin.wait()
            progressed.set()
        thread = threading.Thread(target=observer)
        thread.start()
        assert ready.wait(5)
        old_interval = sys.getswitchinterval()
        try:
            # Prevent an ordinary interpreter timeslice from satisfying it.
            sys.setswitchinterval(60)
            begin.set()
            call()
            return progressed.is_set()
        finally:
            sys.setswitchinterval(old_interval)
            begin.set()
            thread.join(5)
            assert not thread.is_alive()

    # A scorer holding the GIL fails every attempt. One that releases it can
    # still miss on a loaded host, when the observer is not scheduled inside
    # the native window (seen 2 times in 28 single-attempt runs).
    assert any(probe() for _ in range(5)), "native numerical work held the GIL"

"""Qualification of harness hooks, accounting, bit checks and restoration."""
from types import SimpleNamespace

import numpy as np
import pytest

from bench.format3_cpu_block_timing import BlockTimer, block_hooks, check_exact, run_pass


def _descriptor(index=0):
    return SimpleNamespace(group_index=0, block_index=index, n_conf=2, library_width=8,
                           gid_base=index*2)


def _decoded(descriptor):
    return SimpleNamespace(descriptor=descriptor,
        result=SimpleNamespace(timings=SimpleNamespace(decode=.1, color=.2, self_overlap=.3)))


def test_qualification_requires_every_claim_and_native_counter_total():
    d = _descriptor()
    source = SimpleNamespace(cpu_block_descriptors=lambda: [d])
    timer = BlockTimer(measure=False)
    timer.start(d)
    timer.decoded(_decoded(d))
    cache = SimpleNamespace(routes={(0, 0): dict(vaa='verified_stored', color='sidecar', color_reason=None)})
    timer.bound(cache, d, (object(), None, False, False))
    timer.end(None)
    scan = dict(rows_decoded=2, native_decode_s=.1, native_color_derivation_s=.2,
                native_self_binding_s=.3)
    rows = timer.finish(source, scan, color=True)
    assert rows == [dict(group=0, block=0, rows=2, L=8, gid_base=0,
                         route=dict(vaa='verified_stored', color='sidecar', color_reason=None))]
    with pytest.raises(AssertionError, match='timer totals'):
        timer.finish(source, {**scan, 'native_decode_s': 1.}, color=True)
    source.cpu_block_descriptors = lambda: [d, _descriptor(1)]
    with pytest.raises(AssertionError, match='exactly once'):
        timer.finish(source, scan, color=True)


def test_hooks_restore_a_class_decoder_and_global_binding_on_failure():
    from roshambo3.engine.r3g_cpu import R3gWorkerSelfCache
    class Source:
        def decode_cpu_block(self, descriptor):
            raise RuntimeError('injected decoder failure')
    source = Source()
    original_bind = R3gWorkerSelfCache.block
    with pytest.raises(RuntimeError, match='injected decoder failure'):
        with block_hooks(source, BlockTimer(measure=False)):
            source.decode_cpu_block(_descriptor())
    assert 'decode_cpu_block' not in vars(source)
    assert R3gWorkerSelfCache.block is original_bind


def test_exact_control_refuses_signed_zero_and_substituted_identity():
    expected = dict(scores=np.zeros((1, 20), np.float32), gids=np.array([0]),
                    pairs=np.array([[19, 0]]), canonical=np.array([0]))
    signed = {**expected, 'scores': expected['scores'].copy()}
    signed['scores'].view(np.uint32)[0, 0] ^= np.uint32(1 << 31)
    np.testing.assert_array_equal(signed['scores'], expected['scores'])
    with pytest.raises(AssertionError):
        check_exact([signed], [expected])
    substituted = {**expected, 'pairs': np.array([[20, 0]])}
    with pytest.raises(AssertionError):
        check_exact([substituted], [expected])


@pytest.mark.parametrize('mode', ['shape', 'combo'])
def test_real_public_worker_hook_is_transparent_and_reports_no_qualification_times(tmp_path, mode):
    from bench.format3_cpu_matrix import _query
    from roshambo3.engine import R3gSource
    from tests.test_self_overlap_routes import build_inputs
    corpus, _, generator = build_inputs(tmp_path / 'corpus')
    source = R3gSource(corpus, color=mode == 'combo', color_generator=generator, tile=False)
    try:
        query = _query(source)
    finally:
        source.close()
    expected, _ = run_pass(corpus, query, mode, 2, k=3, level='conformer', instrument=False)
    actual, record = run_pass(corpus, query, mode, 2, k=3, level='conformer')
    check_exact(actual, expected)
    assert record['workers_with_claims'] <= 2
    assert sum(record['worker_blocks']) == record['blocks']
    assert len(record['block_records']) == record['blocks']
    for block in record['block_records']:
        assert not any(key.endswith('_s') or key == '_native' for key in block)
    assert 'scan_wall_s' not in record and 'engine_stage_s' not in record


def test_campaign_refuses_wrong_arm_labels_and_duplicate_work_lists():
    from bench.format3_cpu_block_timing import describe_arms
    dtype = [('group', '<i4'), ('block', '<i4'), ('first_row', '<i8'),
             ('rows', '<i8'), ('cost', '<i8')]
    def corpus(counts):
        directory = np.asarray([(0, i, sum(counts[:i]), n, n*8)
                                for i, n in enumerate(counts)], dtype=dtype)
        return SimpleNamespace(block_directory=lambda order: directory, n_conf=sum(counts),
            fingerprint='a'*64, descriptor={'prep': {'commit': 'b'*40}})
    with pytest.raises(ValueError, match='exceed declared target'):
        describe_arms({4: corpus([8, 2])}, {4: 'wrong'})
    with pytest.raises(ValueError, match='identical physical block directories'):
        describe_arms({4: corpus([4, 2]), 8: corpus([4, 2])}, {4: 'a', 8: 'b'})
    arms = describe_arms({4: corpus([4, 4, 2]), 8: corpus([8, 2])}, {4: 'a', 8: 'b'})
    assert arms[4]['physical_blocks'] == 3
    assert arms[8]['min_block_rows'] == 2 and arms[8]['max_block_rows'] == 8
    assert arms[4]['directory_sha256'] != arms[8]['directory_sha256']

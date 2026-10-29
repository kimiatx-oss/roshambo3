"""Authenticated format-3 section slices replace format-2 sidecar receipts.

Each source authenticates complete sections once before making descriptors.
Every claim rechecks its bounded slices, including when size/mtime is unchanged.
No receipt or native values handle outlives a changed source's content.
"""
import gc
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from roshambo3.backend import cpu
from roshambo3.engine import Engine, r3g_cpu
from roshambo3.format.r3g.errors import CorpusValidationError
from tests.test_self_overlap_routes import build_inputs, _inputs, _run, _block_inputs


@pytest.fixture
def built(tmp_path):
    return build_inputs(tmp_path / 'corpus')


def _flip(source, name, *, descriptor=None, preserve_stat=True):
    d = descriptor or source.cpu_block_descriptors()[0]
    with source.corpus.group(d.group_index) as group:
        path = Path(group.path)
        section = group._og.sections[name]
        offset = section['offset']
        if name == 'coords':
            offset += int(group.blocks[d.block_index]['offset'])
        elif name in ('vaa', 'color_self'):
            offset += d.group_row0 * (4 if name == 'vaa' else 8)
    stat = path.stat()
    with path.open('r+b') as file:
        file.seek(offset)
        value = file.read(1)
        file.seek(offset)
        file.write(bytes([value[0] ^ 1]))
    if preserve_stat:
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    return path


def test_setup_authenticates_each_section_once_and_claims_recheck_slices(built, monkeypatch):
    source, query, config = _inputs(*built, 'combo')
    counts = Counter()
    authenticate = r3g_cpu._authenticate_section
    read_bound = r3g_cpu._read_bound
    def counted_auth(group, name, *args, **kwargs):
        counts['section:' + name] += 1
        return authenticate(group, name, *args, **kwargs)
    def counted_bound(group, bi, name, *args):
        counts['slice:' + name] += 1
        return read_bound(group, bi, name, *args)
    monkeypatch.setattr(r3g_cpu, '_authenticate_section', counted_auth)
    monkeypatch.setattr(r3g_cpu, '_read_bound', counted_bound)
    descriptors = source.cpu_block_descriptors()
    expected = {'runs', 'vaa', 'color_self', 'feature_offsets', 'feature_types',
                'feature_atom_offsets', 'feature_atoms'}
    assert {k for k in counts if k.startswith('section:')} == {'section:' + n for n in expected}
    assert all(counts['section:' + n] == len(source.group_indices) for n in expected)
    engine = Engine(backend=cpu)
    try:
        previous = None
        for _ in range(2):
            counts.clear()
            stats, rows = _run(built, 'combo', engine=engine, inputs=(source, query, config))
            assert not any(k.startswith('section:') for k in counts)
            assert all(counts['slice:' + n] == len(descriptors) for n in expected)
            assert stats.cpu_scan['stored_vaa_verified_blocks'] == len(descriptors)
            assert stats.cpu_scan['prepared_self_retained_bytes'] == 0
            if previous is not None:
                for a, b in zip(previous[0], rows[0]):
                    np.testing.assert_array_equal(a, b)
            previous = rows
    finally:
        engine.close()
        source.close()


@pytest.mark.parametrize('name', ['vaa', 'color_self', 'runs', 'feature_types', 'coords'])
def test_tampering_after_setup_refuses_the_claimed_group_and_block(built, name):
    source, query, config = _inputs(*built, 'combo')
    descriptor = source.cpu_block_descriptors()[0]
    _flip(source, name, descriptor=descriptor)
    with pytest.raises(CorpusValidationError, match=f'group {descriptor.group_index} block {descriptor.block_index}'):
        Engine(backend=cpu).run_worker_owned_cpu(source, query, config,
            workers=2, k=3, self_overlap_policy='prepared')
    source.close()


@pytest.mark.parametrize('name', ['vaa', 'color_self'])
def test_tampering_before_authentication_refuses_a_fresh_source(built, name):
    source, _, _ = _inputs(*built, 'combo')
    # Locate the file without creating/authenticating descriptors.
    with source.corpus.group(0) as group:
        path = Path(group.path)
        offset = group._og.sections[name]['offset']
    stat = path.stat()
    with path.open('r+b') as file:
        file.seek(offset)
        value = file.read(1)
        file.seek(offset)
        file.write(bytes([value[0] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(CorpusValidationError, match='group 0 block.*' + name):
        source.cpu_block_descriptors()
    source.close()


@pytest.mark.parametrize('name', ['vaa', 'color_self'])
def test_preserved_stat_tampering_refused_by_a_new_process(built, name):
    source, _, _ = _inputs(*built, 'combo')
    _flip(source, name)
    code = '''
import sys
from roshambo3.format.r3g.reader import verify, open_corpus
from roshambo3.engine import R3gSource
from roshambo3.format.r3g.errors import CorpusValidationError
try:
    verify(sys.argv[1], use_cache=False, cache_dir=sys.argv[2])
    corpus = open_corpus(sys.argv[1], cache_dir=sys.argv[2])
    source = R3gSource(corpus, color=True, tile=False)
    source.cpu_block_descriptors()
except CorpusValidationError as error:
    print(error.code)
    sys.exit(3)
'''
    result = subprocess.run([sys.executable, '-c', code, source.corpus.root,
                             str(Path(source.corpus.root).parent / 'fresh-cache')],
                            capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    assert 'integrity.section_hash' in result.stdout
    source.close()


def test_decoded_rows_cannot_be_bound_as_another_descriptor(built):
    source, descriptor, decoded, _ = _block_inputs(built)
    _, query, _ = _inputs(*built, 'combo')
    cache = r3g_cpu.R3gWorkerSelfCache(source, 'a'*64)
    with pytest.raises(ValueError, match='different descriptor'):
        cache.block(cpu, replace(descriptor, gid_base=descriptor.gid_base+1), decoded,
            decoded.result.coords, query.coords, query.types, query.n_real,
            query.interaction_r, query.interaction_p)
    with pytest.raises(ValueError, match='not owned'):
        source.decode_cpu_block(replace(descriptor))
    source.close()


def test_closed_source_does_not_retain_block_values(built):
    import weakref
    source, descriptor, decoded, _ = _block_inputs(built)
    refs = [weakref.ref(decoded.result.coords), weakref.ref(decoded.result.vaa)]
    del decoded
    source.close()
    gc.collect()
    assert all(ref() is None for ref in refs)

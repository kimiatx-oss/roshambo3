"""Cache reuse, isolation, invalidation and failed-publication controls."""
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.shared_cache import cached_directory, cached_compile


def _produce(cache, count):
    os.environ['ROSHAMBO3_TEST_CACHE'] = cache
    def build(directory):
        with open(count, 'a') as stream:
            stream.write('built\n')
        (directory/'value').write_text('complete')
    cached_directory('concurrent-control', {'input': 1}, build)


def test_workers_publish_once_and_input_changes_invalidate(tmp_path, monkeypatch):
    cache = tmp_path/'cache';count=tmp_path/'count'
    monkeypatch.setenv('ROSHAMBO3_TEST_CACHE', str(cache))
    context=multiprocessing.get_context('spawn')
    children=[context.Process(target=_produce,args=(str(cache),str(count))) for _ in range(2)]
    for child in children: child.start()
    for child in children:
        child.join(30)
        assert child.exitcode==0
    assert count.read_text()=='built\n'
    one=cached_directory('identity-control', {'input':1}, lambda path:(path/'value').write_text('one'))
    two=cached_directory('identity-control', {'input':2}, lambda path:(path/'value').write_text('two'))
    assert one!=two and (one/'value').read_text()=='one' and (two/'value').read_text()=='two'


def test_failed_build_is_never_reused(tmp_path, monkeypatch):
    monkeypatch.setenv('ROSHAMBO3_TEST_CACHE',str(tmp_path/'cache'))
    def fail(path):
        (path/'partial').write_text('incomplete')
        raise ValueError('planted interruption')
    with pytest.raises(ValueError,match='interruption'):
        cached_directory('failure-control',{},fail)
    complete=cached_directory('failure-control',{},lambda path:(path/'value').write_text('complete'))
    assert not (complete/'partial').exists()


def test_corpus_copies_are_private(tmp_path):
    from tests.codec_fixtures import make_codec_fixture
    one=make_codec_fixture(tmp_path/'one')
    two=make_codec_fixture(tmp_path/'two')
    path=one.physical_blocks[0].path
    sibling=two.root/path.relative_to(one.root)
    assert path.read_bytes()==sibling.read_bytes() and path.stat().st_ino!=sibling.stat().st_ino
    original=sibling.read_bytes()
    path.write_bytes(b'private corruption')
    assert sibling.read_bytes()==original


def test_common_native_translation_unit_is_built_once(tmp_path, monkeypatch):
    if not shutil.which('g++'):
        pytest.skip('native compiler unavailable')
    cache=tmp_path/'cache'
    monkeypatch.setenv('ROSHAMBO3_TEST_CACHE',str(cache))
    common=tmp_path/'common.cpp';common.write_text('int value() {return 4;}')
    for i in range(2):
        driver=tmp_path/f'driver{i}.cpp'
        driver.write_text('extern int value(); int main(){return value()!=4;}')
        executable=tmp_path/f'run{i}'
        cached_compile(['g++','-O2',str(common),str(driver),'-o',str(executable)],capture_output=True,text=True)
        assert subprocess.run([str(executable)]).returncode==0
    assert len(list((cache/'native-object').glob('*/complete.json')))==3


def _held_compile(lock, events):
    import time
    from tests.shared_cache import compile_lock
    os.environ['ROSHAMBO3_TEST_BUILD_LOCK']=lock
    with compile_lock():
        start=time.monotonic()
        time.sleep(.05)
        with open(events,'a') as stream:
            stream.write(f'{start} {time.monotonic()}\n')


def test_different_builds_never_compile_concurrently(tmp_path):
    lock=str(tmp_path/'build.lock');events=tmp_path/'events'
    context=multiprocessing.get_context('spawn')
    children=[context.Process(target=_held_compile,args=(lock,str(events))) for _ in range(2)]
    for child in children:child.start()
    for child in children:
        child.join(30)
        assert child.exitcode==0
    intervals=sorted(tuple(map(float,line.split())) for line in events.read_text().splitlines())
    assert len(intervals)==2 and intervals[0][1]<=intervals[1][0]

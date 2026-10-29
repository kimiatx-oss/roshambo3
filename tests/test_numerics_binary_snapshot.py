"""Fail-closed cleanup of the exact original artifact with restored native source."""
import pytest

from bench.numerics_binary_snapshot import snapshot_binary, restore_binary, file_sha256
from scripts.source_stamp import source_content_hash


def fixture(tmp_path):
    build = tmp_path/'build'; build.mkdir()
    native = tmp_path/'native'; native.mkdir()
    (native/'kernel.cu').write_text('declared original source')
    binary = build/'extension.so'; binary.write_bytes(b'original native artifact')
    return build, native, binary


def test_original_artifact_restored_after_source_identical_different_rebuild(tmp_path):
    build, native, binary = fixture(tmp_path)
    original = file_sha256(binary)
    source = source_content_hash(native)
    saved = snapshot_binary(binary, original, build=build, backup_parent=tmp_path/'snapshots')
    binary.write_bytes(b'different clean compiler artifact')
    restore_binary(saved, native_source=native, expected_source_sha256=source)
    assert file_sha256(binary) == original
    assert saved['backup_path'].startswith(str(tmp_path/'snapshots'))


def test_original_restore_refuses_changed_source(tmp_path):
    build, native, binary = fixture(tmp_path)
    saved = snapshot_binary(binary, file_sha256(binary), build=build, backup_parent=tmp_path/'snapshots')
    source = source_content_hash(native)
    (native/'kernel.cu').write_text('still mutated source')
    binary.write_bytes(b'mutant artifact')
    with pytest.raises(RuntimeError, match='different native source'):
        restore_binary(saved, native_source=native, expected_source_sha256=source)
    assert binary.read_bytes() == b'mutant artifact'


def test_original_restore_refuses_corrupt_snapshot(tmp_path):
    build, native, binary = fixture(tmp_path)
    saved = snapshot_binary(binary, file_sha256(binary), build=build, backup_parent=tmp_path/'snapshots')
    from pathlib import Path
    Path(saved['backup_path']).write_bytes(b'corrupted original')
    binary.write_bytes(b'rebuilt artifact')
    with pytest.raises(RuntimeError, match='snapshot was changed'):
        restore_binary(saved, native_source=native, expected_source_sha256=source_content_hash(native))
    assert binary.read_bytes() == b'rebuilt artifact'


def test_snapshot_must_be_outside_build_and_bind_current_bytes(tmp_path):
    build, native, binary = fixture(tmp_path)
    with pytest.raises(RuntimeError, match='outside the build'):
        snapshot_binary(binary, file_sha256(binary), build=build, backup_parent=build/'snapshots')
    with pytest.raises(RuntimeError, match='changed before snapshot'):
        snapshot_binary(binary, '0'*64, build=build, backup_parent=tmp_path/'snapshots')

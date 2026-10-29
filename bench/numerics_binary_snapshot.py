"""Exact native artifact cleanup, independent of compiler byte reproducibility."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from scripts.source_stamp import source_content_hash


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot_binary(binary, expected_sha256, *, build, backup_parent):
    binary, build, backup_parent = map(lambda p: Path(p).resolve(), (binary, build, backup_parent))
    if binary.parent != build:
        raise RuntimeError('clean binary must belong to the named build')
    if backup_parent == build or build in backup_parent.parents:
        raise RuntimeError('original binary backup must be outside the build')
    if file_sha256(binary) != expected_sha256:
        raise RuntimeError('clean binary changed before snapshot')
    backup_parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='original-cuda-', dir=backup_parent))
    backup = directory/binary.name
    shutil.copy2(binary, backup)
    if file_sha256(backup) != expected_sha256:
        raise RuntimeError('original binary snapshot hash mismatch')
    return dict(binary_path=str(binary), backup_path=str(backup), sha256=expected_sha256,
                mode=binary.stat().st_mode & 0o777, size=binary.stat().st_size)


def restore_binary(snapshot, *, native_source, expected_source_sha256):
    """Check restored source, then atomically reinstall the exact verified bytes."""
    if source_content_hash(Path(native_source)) != expected_source_sha256:
        raise RuntimeError('refusing original binary restore onto different native source')
    backup = Path(snapshot['backup_path'])
    binary = Path(snapshot['binary_path'])
    if file_sha256(backup) != snapshot['sha256']:
        raise RuntimeError('original binary snapshot was changed')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.native-restore-', dir=binary.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with backup.open('rb') as source:
                shutil.copyfileobj(source, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, snapshot['mode'])
        if file_sha256(temporary) != snapshot['sha256']:
            raise RuntimeError('temporary original binary hash mismatch')
        os.replace(temporary, binary)
        temporary = None
        if file_sha256(binary) != snapshot['sha256']:
            raise RuntimeError('atomic original binary restoration hash mismatch')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

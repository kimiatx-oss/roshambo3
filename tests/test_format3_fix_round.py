"""Regressions for the format-3 spec/reference fix round."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from roshambo3.format.r3g import jsonstrict, limits, native, reference as R, validate as V, writer as W

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'tests/data/format3'
spec = importlib.util.spec_from_file_location('adversarial_builder', DATA / 'adversarial/make_adversarial.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def refusal(fn, code, *args, **kwargs):
    with pytest.raises(R.CorpusValidationError) as exc:
        fn(*args, **kwargs)
    assert exc.value.code == code


@pytest.mark.parametrize('parser', [jsonstrict.loads, lambda data: R.loads_strict(data, what='test')])
@pytest.mark.parametrize('data,code', [
    (b'{"x":1e999}', 'json.nan'),
    (b'{"\\ud800":0}', 'json.syntax'),
    (b'{"x":"\\udfff"}', 'json.syntax'),
    (b'{"x":[' + b'[],' * (1 << 18) + b'[]]}', 'json.containers'),
])
def test_strict_json_before_allocation(parser, data, code):
    refusal(parser, code, data)


@pytest.mark.parametrize('implementation', [R, V])
@pytest.mark.parametrize('field,value,code', [
    ('name', [], 'schema.type'), ('dtype', {}, 'schema.type'),
    ('shape', [1 << 62] * 10000, 'layout.sections'),
])
def test_section_fields_refused_at_open(tmp_path, implementation, field, value, code):
    def mutate(name, header):
        header['sections'][0][field] = value
    M._write(str(tmp_path), header_mutate=mutate)
    refusal(implementation.validate_corpus, code, str(tmp_path), 'open')


@pytest.mark.parametrize('implementation', [R, V])
def test_csr_chunk_boundary_is_checked_before_slicing(tmp_path, implementation):
    mols = [R.Molecule(i, np.zeros((1, 3, 1), np.int16), np.ones(1, np.float32),
                       features=[(1, (0,))] * 255, color_self=np.ones(1),
                       ext_id=str(i), smiles='C', mw=16) for i in range(258)]
    def mutate(name, sections, fields, records):
        offsets = np.frombuffer(sections['feature_atom_offsets'], dtype='<u4').copy()
        offsets[:65538] = np.arange(65538)
        offsets[65538:] = 10
        sections['feature_atom_offsets'] = offsets.tobytes()
        sections['feature_atoms'] = bytes(10)
    M._write(str(tmp_path), color=True, molecules=mols, group_mutate=mutate)
    refusal(implementation.validate_corpus, 'layout.features', str(tmp_path), 'scan')


@pytest.mark.parametrize('implementation', [R, V])
def test_prefix_feature_order_round_trip(tmp_path, implementation):
    mols = M._base_molecules(True)
    mols[1].features = [(1, (0,)), (1, (0, 1))]
    M._write(str(tmp_path), color=True, molecules=mols)
    implementation.validate_corpus(str(tmp_path), 'scan')


@pytest.mark.parametrize('case', ['abort_case', 'abort_case2'])
@pytest.mark.parametrize('module', ['reference', 'validate'])
def test_footer_abort_reproducer_is_a_typed_refusal(tmp_path, case, module):
    # Put the old hostile footer behind valid current-limit groups/hashes.
    if case == 'abort_case':
        M._write(str(tmp_path))
    else:
        shutil.copytree(DATA / 'color', tmp_path, dirs_exist_ok=True)
    # Malformed-footer reproducers from a writer review.
    pq = DATA / 'repro' / {'abort_case': 'malformed_footer_shape.parquet',
                           'abort_case2': 'malformed_footer_color.parquet'}[case]
    shutil.copyfile(pq, tmp_path / 'molecules.parquet')
    d = json.loads((tmp_path / 'corpus.json').read_bytes())
    d['molecules_sha256'] = hashlib.sha256(pq.read_bytes()).hexdigest()
    (tmp_path / 'corpus.json').write_bytes(R.canonical_json(d))
    script = '''
import sys
from roshambo3.format.r3g import MODULE as m
try:
    m.validate_corpus(sys.argv[1], 'scan')
except m.CorpusValidationError as e:
    assert e.code == 'value.parquet', e
else:
    raise AssertionError('malformed footer accepted')
'''.replace('MODULE', module)
    run = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize('implementation', [R, V])
@pytest.mark.parametrize('codec', ['none', 'zstd'])
@pytest.mark.parametrize('page_version', ['1.0', '2.0'])
def test_parquet_supported_codecs_and_page_versions(tmp_path, implementation, codec, page_version):
    M._write(str(tmp_path))
    M._rewrite_parquet(str(tmp_path), lambda table: table, compression=codec,
                       data_page_version=page_version)
    implementation.validate_corpus(str(tmp_path), 'scan')


@pytest.mark.parametrize('implementation', [R, V])
@pytest.mark.parametrize('name,code', [('ext_id', 'limit.ext_id'), ('smiles', 'limit.smiles')])
def test_invalid_utf8_is_refused_at_scan(tmp_path, implementation, name, code):
    import pyarrow as pa
    M._write(str(tmp_path))
    def mutate(table):
        values = table[name].to_pylist()
        offsets = np.array([0, 1, 1 + len(values[1].encode())], dtype='<i4')
        array = pa.Array.from_buffers(pa.string(), 2,
                                     [None, pa.py_buffer(offsets), pa.py_buffer(b'\xff' + values[1].encode())])
        return table.set_column(table.schema.get_field_index(name), table.schema.field(name), array)
    M._rewrite_parquet(str(tmp_path), mutate)
    refusal(implementation.validate_corpus, code, str(tmp_path), 'scan')


def test_writer_enforces_block_and_feature_limits(tmp_path, monkeypatch):
    mols = M._base_molecules(False)
    monkeypatch.setitem(limits.LIMITS, 'blocks', 1)
    with pytest.raises(ValueError, match='block count'):
        W.write_corpus(str(tmp_path), mols, color=None, prep_version='test', prep_commit=None, block_rows=1)
    monkeypatch.setitem(limits.LIMITS, 'blocks', 1 << 20)
    monkeypatch.setitem(limits.LIMITS, 'n_feat_atoms', 1)
    with pytest.raises(ValueError, match='feature CSR'):
        W.write_corpus(str(tmp_path), M._base_molecules(True), color=M._color(),
                       prep_version='test', prep_commit=None)


def test_writer_checks_escaped_descriptor_string(tmp_path):
    text = M.TINY_FDEF + '# ' + '\t' * 40000 + '\n'
    with pytest.raises(ValueError, match='json.string_length'):
        W.write_corpus(str(tmp_path), M._base_molecules(True), color=M._color(text),
                       prep_version='test', prep_commit=None)


@pytest.mark.parametrize('call', [
    lambda: native.decode_block(b'', b'', 10**6, 8),
    lambda: native.decode_merged(b'', b'', 10**6, 8, 512, 6, None),
    lambda: native.write_cpu_block(b'', b'', 10**6, 8, 512, 6, None, 1),
    lambda: native.write_gpu_block(b'', b'', 10**6, 8, 512, 6, None, None),
])
def test_native_dimensions_checked_before_allocation(monkeypatch, call):
    def forbidden(*a, **kw):
        raise AssertionError('allocated before checking dimensions')
    monkeypatch.setattr(native.np, 'empty', forbidden)
    with pytest.raises(native.CodecError) as exc:
        call()
    assert exc.value.code == 'limit.block_rows'


def test_binding_rejects_feature_input_alias():
    # The review's public-binding reproducer: output overwrites atom offsets.
    mols = M._base_molecules(True)
    xyz = np.zeros((2, 3, 8), np.int16)
    xyz[1, :, :4] = mols[1].coords[0]
    runs = R.pack_runs([11, 22], [1, 1], [1, 4])
    block = native.encode_block(xyz, runs)
    fs = R.pack_features([[], mols[1].features])
    # Pad storage so the aliased words output has sufficient capacity.
    store = np.zeros(4, dtype='<u4')
    store[:3] = fs['feature_atom_offsets']
    feature = native.FeatureSlice(fs['feature_offsets'], fs['feature_types'], store, fs['feature_atoms'])
    with pytest.raises(native.CodecError) as exc:
        native.write_cpu_block(block, runs, 2, 8, 8, 2, feature, 1,
                               count_words_out=store.view(np.uint64))
    assert exc.value.code == 'decode.argument'


def test_conformance_native_gather_matches_section_8_3_formula():
    descriptor, fp = R.read_descriptor(str(DATA / 'color'))
    try:
        for isa in native.compiled_isas():
            native.force_isa(isa)
            for entry in descriptor['groups']:
                gf = R.open_group(str(DATA / 'color' / R.group_file_name(entry['bucket'], entry['group'])),
                                  descriptor=descriptor, group_entry=entry, fingerprint_hex=fp)
                try:
                    runs = gf.section_array('runs')
                    tables = {name: gf.section_array(name) for name in
                              ('feature_offsets', 'feature_types', 'feature_atom_offsets', 'feature_atoms')}
                    for index, block in enumerate(gf.blocks()):
                        m0, m1, rows = int(block['m0']), int(block['m1']), int(block['rows'])
                        features = native.FeatureSlice(tables['feature_offsets'][m0:m1 + 1],
                                                       tables['feature_types'], tables['feature_atom_offsets'],
                                                       tables['feature_atoms'])
                        merged, _, _ = native.decode_merged(gf.block_bytes(index), runs[m0:m1], rows,
                                                             gf.header['L_heavy'], gf.header['L'],
                                                             gf.header['n_types'], features)
                        row = 0
                        for mol in range(m0, m1):
                            nr, nc = int(runs[mol]['n_real']), int(runs[mol]['n_conf'])
                            # Independent integer transcription: never call derive_positions/mean helpers.
                            for offset, (_, atoms) in enumerate(R.molecule_features(tables, mol)):
                                sums = merged[row:row + nc, :, :nr][:, :, list(atoms)].astype(np.int64).sum(axis=2)
                                k = len(atoms)
                                want = np.sign(sums) * ((2 * np.abs(sums) + k) // (2 * k))
                                np.testing.assert_array_equal(merged[row:row + nc, :, nr + offset], want)
                            row += nc
                finally:
                    gf.close()
    finally:
        native.force_isa('auto')


@pytest.mark.parametrize('implementation', [R, V])
@pytest.mark.parametrize('name', ['mw', 'rotors'])
def test_scan_reads_every_column_and_checks_decoded_length(tmp_path, monkeypatch, implementation, name):
    import pyarrow.parquet as pq
    M._write(str(tmp_path))
    original = pq.ParquetFile
    class ShortColumn:
        def __init__(self, *args, **kwargs):
            self.wrapped = original(*args, **kwargs)
        def __getattr__(self, attr):
            return getattr(self.wrapped, attr)
        def read_row_group(self, *args, **kwargs):
            table = self.wrapped.read_row_group(*args, **kwargs)
            return table.slice(0, 0) if kwargs.get('columns') == [name] else table
    monkeypatch.setattr(pq, 'ParquetFile', ShortColumn)
    refusal(implementation.validate_corpus, 'value.parquet', str(tmp_path), 'scan')


def test_conformance_regeneration_is_byte_identical(tmp_path):
    import platform
    import pyarrow
    import rdkit
    generator_spec = importlib.util.spec_from_file_location('conformance_generator', DATA / 'make_conformance.py')
    generator = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator)
    generator.HERE = str(tmp_path)
    generator.main()
    def files():
        return {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    first = files()
    generator.main()
    assert first == files()
    pinned = json.loads((DATA / 'EXPECTED.json').read_bytes())['provenance']
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    pyarrow=pyarrow.__version__, rdkit=rdkit.__version__)
    if all(pinned[key] == value for key, value in versions.items()):
        assert all((DATA / name).read_bytes() == data for name, data in first.items())


def test_footer_rejects_byte_statistics_on_numeric_column(tmp_path):
    import struct
    from roshambo3.format.r3g.parquet_footer import check_footer
    M._write(str(tmp_path))
    data = (tmp_path / 'molecules.parquet').read_bytes()
    length = struct.unpack('<I', data[-8:-4])[0]
    footer = data[-8-length:-8]
    # First column (mol_id): add SizeStatistics.unencoded_byte_array_data_bytes.
    # Numeric columns cannot carry it: Arrow's SizeStatistics::Validate throws.
    encoding_stats = b'\x49\x1c\x15\x00\x15\x00\x15\x02\x00'
    assert encoding_stats in footer
    footer = footer.replace(encoding_stats, encoding_stats + b'\x3c\x16\x02\x00', 1)
    with pytest.raises(ValueError, match='BYTE_ARRAY'):
        check_footer(footer, n_mol=2, max_row_groups=1)

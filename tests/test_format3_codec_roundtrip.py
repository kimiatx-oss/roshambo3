"""Round-trip the production fixture through validated groups and native decode."""
from pathlib import Path

import numpy as np
import pytest
import roshambo3
from roshambo3.format import Catalog
from roshambo3.format.r3g import codec, native
from roshambo3.format.r3g.errors import CorpusValidationError
from roshambo3.format.r3g.validate import open_group
from tests.codec_fixtures import iter_codec_layouts, make_codec_fixture, truncate_block_record_copy


def test_runner_loaded_this_checkouts_package():
    expected = Path(__file__).resolve().parents[1] / 'src/roshambo3/__init__.py'
    assert Path(roshambo3.__file__).resolve().read_bytes() == expected.read_bytes()


def test_layout_iterator_names_only_the_production_encoding():
    layouts = tuple(iter_codec_layouts())
    assert {(x.codec, x.shuffle, x.color_layout) for x in layouts} == {
        ('bp256-16', False, 'none'), ('bp256-16', False, 'merged')}


@pytest.mark.parametrize('layout', tuple(iter_codec_layouts()), ids=lambda x: x.color_layout)
def test_roundtrip_coordinates_metadata_and_molecule_block_edges(tmp_path, layout):
    fx = make_codec_fixture(tmp_path / 'corpus', codec=layout.codec,
                            shuffle=layout.shuffle, color_layout=layout.color_layout)
    cat = Catalog(fx.corpus)
    np.testing.assert_array_equal(cat.mol_ids(fx.global_ids), [41, 41, 41, 900, 17, 17])
    np.testing.assert_array_equal(cat.conf_indices(fx.global_ids), [0, 1, 2, 0, 0, 1])
    assert fx.corpus.n_conf == 6
    assert [t.num_rows for t in fx.corpus.molecules()] == [3]
    blocks = fx.blocks_for('shape')
    assert [b.gid_slice for b in blocks] == [slice(0, 3), slice(3, 4), slice(4, 6)]
    assert [b.record[1] for b in blocks] == [3, 1, 2]
    # The first molecule ends exactly at the edge; no molecule straddles it.
    assert fx.molecule_ids[2] == 41 and fx.molecule_ids[3] == 900
    for b in blocks:
        with fx.corpus.group(b.group_index) as g:
            rec = g.blocks[b.block_index]
            assert tuple(int(rec[k]) for k in ("first_row", "rows", "offset", "length", "m0", "m1")) == b.record
            m0, m1 = int(rec['m0']), int(rec['m1'])
            runs = g.runs[m0:m1]
            data = g.block_bytes(b.block_index)
            np.testing.assert_array_equal(g.decode_block(b.block_index), fx.heavy_xyz_i16[b.gid_slice])
            fs = (native.FeatureSlice(g.feature_offsets[m0:m1 + 1], g.feature_types,
                                      g.feature_atom_offsets, g.feature_atoms)
                  if g.n_types else None)
            merged, counts, _ = native.decode_merged(data, runs, int(rec['rows']),
                g.header['L_heavy'], g.header['L'], g.n_types, fs)
            np.testing.assert_array_equal(merged, fx.shape_xyz_i16[b.gid_slice])
            np.testing.assert_array_equal(counts, fx.counts[b.gid_slice])
            # Counts delimit each row's own heavy/color prefix and zero tail.
            for row, count in zip(merged, counts):
                assert not row[:, int(count.sum()):].any()


def test_short_final_block_is_legal_but_empty_group_is_refused(tmp_path):
    fx = make_codec_fixture(tmp_path / 'short')
    assert [b.record[1] for b in fx.physical_blocks] == [3, 1, 2]
    with pytest.raises(ValueError, match='empty'):
        make_codec_fixture(tmp_path / 'empty', empty_final_group=True)


def test_truncated_directory_and_payload_are_refused(tmp_path):
    fx = make_codec_fixture(tmp_path / 'source')
    b = fx.physical_blocks[-1]
    damaged = truncate_block_record_copy(b, tmp_path / 'damaged' / 'groups' / b.path.name)
    with pytest.raises(CorpusValidationError):
        open_group(str(damaged.parent.parent), fx.corpus.group_table[b.group_index])
    with fx.corpus.group(b.group_index) as g:
        rec = g.blocks[b.block_index]
        runs = g.runs[int(rec['m0']):int(rec['m1'])]
        with pytest.raises(ValueError):
            codec.decode_block(g.block_bytes(b.block_index)[:-1], runs, g.header['L_heavy'])

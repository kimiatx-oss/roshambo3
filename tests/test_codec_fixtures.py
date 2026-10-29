"""``tests/codec_fixtures.py`` writes the format-3 corpus it describes: what
its arrays say is what a reader decodes, its blocks are the
documented ones, and the format-2 keywords are refused, never mapped
(FORMAT.md Appendix B.8)."""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3.format.r3g import codec, reference as R
from roshambo3.format.r3g.errors import CorpusValidationError
from tests.codec_fixtures import (ENCODING, FEATURES_ENUM, FixtureCorpus, iter_codec_layouts,
                                  make_codec_fixture, truncate_block_record_copy)


@pytest.mark.parametrize("layout", list(iter_codec_layouts()))
@pytest.mark.parametrize("block_conf", [3, 1, 5])
def test_the_fixture_is_what_a_reader_decodes(tmp_path, layout, block_conf):
    fx = make_codec_fixture(tmp_path / "c", codec=layout.codec, shuffle=layout.shuffle,
                            color_layout=layout.color_layout, block_conf=block_conf)
    assert isinstance(fx, FixtureCorpus) and fx.codec == ENCODING
    assert [b.row_stop - b.row_start for b in fx.blocks_for("shape")] == [block_conf, 1, 2]
    assert fx.expected_decode_counts == {"shape": 3}
    assert fx.molecule_ids.tolist() == [41] * block_conf + [900, 17, 17]
    assert fx.global_ids.tolist() == list(range(block_conf + 3))
    corpus = fx.corpus
    assert corpus.scanned and corpus.n_conf == block_conf + 3 and len(corpus) == 2
    color = layout.color_layout == "merged"
    assert corpus.has_color is color
    if color:
        assert {t: i + 1 for i, t in enumerate(corpus.color["types"])} == FEATURES_ENUM
    gid = 0
    for gi in range(len(corpus)):
        with corpus.group(gi) as g:
            heavy = g.decode_all()
            rows = slice(gid, gid + g.n_conf)
            assert np.array_equal(heavy, fx.heavy_xyz_i16[rows, :, :g.L_heavy])
            assert np.array_equal(g.n_real, fx.n_real[rows])
            assert np.array_equal(np.repeat(g.mol_id.astype(np.int64),
                                            g.molecule_n_conf.astype(np.int64)),
                                  fx.molecule_ids[rows])
            assert np.array_equal(g.counts, fx.counts[rows])
            assert np.array_equal(g.vaa, fx.vaa[rows])
            if color:
                assert np.array_equal(g.color_self, fx.color_self[rows])
                start = g.run_start
                for r in range(g.n_conf):
                    m = int(np.searchsorted(start, r, side="right") - 1)
                    feats = R.molecule_features(
                        {k: getattr(g, k) for k in ("feature_offsets", "feature_types",
                                                    "feature_atom_offsets",
                                                    "feature_atoms")}, m)
                    assert feats == fx.molecule_features[int(g.mol_id[m])]
                    merged = R.kernel_layout(heavy[r], int(g.n_real[r]), feats, 8)
                    assert np.array_equal(merged, fx.shape_xyz_i16[gid + r])
            else:
                assert g.color_self is None
            gid += g.n_conf
    assert fx.color_xyz_i16 is None
    if color:
        assert fx.counts[block_conf + 1, 1:].sum() == 0          # an empty-color row


@pytest.mark.parametrize("kwargs, match", [
    ({"color_layout": "split"}, "split"),
    ({"codec": "raw"}, "does not exist in format 3"),
    ({"codec": "lz4"}, "does not exist in format 3"),
    ({"shuffle": True}, "shuffle"),
    ({"empty_final_group": True}, "empty group"),
    ({"block_conf": 0}, "block_conf"),
])
def test_format2_keywords_are_refused_not_mapped(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_codec_fixture(tmp_path / "c", **kwargs)


def test_a_truncated_block_record_is_refused_by_open_and_by_the_decoder(tmp_path):
    from roshambo3.format.r3g.validate import open_group
    fx = make_codec_fixture(tmp_path / "c", color_layout="merged")
    block = fx.blocks_for("shape")[0]
    damaged = truncate_block_record_copy(block, tmp_path / "damaged" / "groups" /
                                         block.path.name)
    assert block.path.read_bytes() != damaged.read_bytes()
    with pytest.raises(CorpusValidationError) as e:
        open_group(str(tmp_path / "damaged"), None, bucket=0, group=0)
    assert e.value.code == "layout.blocks"
    with fx.corpus.group(0) as g:
        data = bytes(g.block_bytes(0))[:-1]
        with pytest.raises(ValueError):
            codec.decode_block(data, g.block_runs(0), g.L_heavy)


@pytest.mark.parametrize("layout", ["none", "merged"])
def test_the_fixture_passes_verify_deep(tmp_path, layout):
    """Review fix: the color fixture's feature tables are what BaseFeatures
    perceives on its SMILES, so ``deep`` (re-perception, every block decoded,
    self-overlaps recomputed) holds for both layouts."""
    from roshambo3.format.r3g.validate import validate_corpus
    from tests.codec_fixtures import COLOR_DEEP_VALID
    assert COLOR_DEEP_VALID
    fx = make_codec_fixture(tmp_path / "c", color_layout=layout)
    report = validate_corpus(str(fx.root), "deep", all_rows=True)
    assert report.sampled_molecules == 3

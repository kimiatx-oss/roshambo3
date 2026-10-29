"""Format-3 catalog lookups over corpora produced by ``prepare()``."""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.format import Catalog
from roshambo3.prep import prepare
from tests.prep_fixtures import molecule
from tests.format3_rows import read_corpus


@pytest.fixture(scope="module")
def builder_corpus(tmp_path_factory):
    """A small actual builder output with several runs and content ordering."""
    root = tmp_path_factory.mktemp("t297-catalog") / "corpus"
    # Gids sort by n_real first; the Parquet table sorts by stable mol_id.
    # These fixed names make those orders differ on purpose.
    mols = [molecule("CCCCCC", "T297-large", conformers=2),
            molecule("C", "T297-small", conformers=3),
            molecule("CCO", "T297-middle", conformers=4)]
    return prepare(mols, str(root), workers=1, block_rows=3)


@pytest.fixture(scope="module")
def catalog(builder_corpus):
    return Catalog(builder_corpus)


class TestLocate:
    def test_covers_exactly_the_corpus(self, catalog, builder_corpus):
        assert catalog.n_conformers == builder_corpus.n_conf
        assert catalog.offsets[0] == 0
        assert catalog.offsets[-1] == builder_corpus.n_conf

    def test_every_id_resolves_to_its_builder_run(self, catalog, builder_corpus):
        rows = read_corpus(builder_corpus.root, merged=False)
        refs = catalog.locate(np.arange(catalog.n_conformers))
        assert np.array_equal([r.mol_id for r in refs], rows.mol_id)
        assert np.array_equal([r.conf_idx for r in refs], rows.conf_idx)
        assert np.array_equal([r.n_real for r in refs], rows.n_real)
        assert np.array_equal([r.vaa for r in refs], rows.vaa)
        assert np.array_equal(catalog.mol_ids(np.arange(catalog.n_conformers)),
                              rows.mol_id)
        assert np.array_equal(catalog.conf_indices(np.arange(catalog.n_conformers)),
                              rows.conf_idx)

    def test_preserves_input_order(self, catalog):
        gids = np.array([catalog.n_conformers - 1, 2, 0, 2, 1])
        assert [r.gid for r in catalog.locate(gids)] == gids.tolist()

    def test_group_and_row_are_consistent_with_the_offsets(self, catalog):
        for r in catalog.locate(np.arange(0, catalog.n_conformers, 2)):
            assert catalog.offsets[r.group_index] + r.row == r.gid

    @pytest.mark.parametrize("bad", [-1, 10 ** 12])
    def test_out_of_range_ids_raise_rather_than_clamp(self, catalog, bad):
        with pytest.raises(IndexError):
            catalog.locate([bad])

    def test_empty_input_is_not_an_error(self, catalog):
        assert catalog.locate([]) == []
        assert catalog.mol_ids([]).size == 0
        assert catalog.conf_indices([]).size == 0


class TestMolecules:
    def test_joins_on_mol_id_not_row_order(self, catalog, builder_corpus):
        """The gid order differs from the molecule table's mol_id order."""
        import pyarrow.parquet as pq

        gids = np.arange(catalog.n_conformers)
        got = catalog.molecules(gids, columns=["hac", "smiles"])
        table = pq.read_table(f"{builder_corpus.root}/molecules.parquet")
        ref = dict(zip(table.column("mol_id").to_pylist(),
                       table.column("hac").to_pylist()))
        assert all(got["found"])
        assert got["hac"] == [ref[mid] for mid in got["mol_id"]]
        gid_order = list(dict.fromkeys(got["mol_id"]))
        metadata_order = table.column("mol_id").to_pylist()
        assert gid_order != metadata_order

    def test_defaults_to_every_column(self, catalog):
        got = catalog.molecules([0])
        for column in ("mol_id", "found", "hac", "smiles", "ext_id"):
            assert column in got

    def test_result_is_aligned_with_the_input(self, catalog):
        gids = np.array([catalog.n_conformers - 1, 0, 1, 0])
        got = catalog.molecules(gids, columns=["hac"])
        assert len(got["hac"]) == len(gids)
        assert got["mol_id"][1] == got["mol_id"][3]

    def test_found_distinguishes_a_missing_join_from_nonnullable_metadata(
            self, catalog, builder_corpus, monkeypatch):
        """Format 3 forbids null metadata; the defensive missing-join API remains."""
        import pyarrow.parquet as pq

        schema = pq.read_schema(f"{builder_corpus.root}/molecules.parquet")
        assert all(not field.nullable for field in schema)

        target_gid = 0
        target_mid = int(catalog.mol_ids([target_gid])[0])
        all_ids = set(map(int, catalog.mol_ids(np.arange(catalog.n_conformers))))
        other_mid = next(mid for mid in all_ids if mid != target_mid)

        present = catalog.molecules([target_gid], columns=["ext_id"])
        assert present["found"] == [True]
        assert isinstance(present["ext_id"][0], str) and present["ext_id"][0]

        # A verified corpus cannot contain a null ext_id. Hide this valid
        # molecule's key from the defensive join index to exercise the public
        # miss behavior without manufacturing invalid format-3 Parquet.
        monkeypatch.setattr(
            catalog, "_meta_ids",
            lambda: (np.asarray([other_mid], dtype=np.int64),
                     np.asarray([0], dtype=np.int64)))
        missing = catalog.molecules([target_gid], columns=["ext_id"])
        assert missing["found"] == [False]
        assert missing["ext_id"] == [None]


class TestCoordinates:
    def test_matches_a_sequential_builder_decode(self, catalog, builder_corpus):
        rows = read_corpus(builder_corpus.root, merged=False)
        gids = np.arange(0, catalog.n_conformers, 2)
        got = catalog.coordinates(gids)
        for xyz, gid in zip(got, gids):
            assert np.array_equal(xyz, rows.coordinates(int(gid))), f"gid {gid}"

    def test_trims_to_real_atoms_by_default(self, catalog):
        gids = np.arange(0, catalog.n_conformers, 2)
        for xyz, ref in zip(catalog.coordinates(gids), catalog.locate(gids)):
            assert xyz.shape == (ref.n_real, 3)

    def test_untrimmed_returns_the_group_heavy_width(self, catalog, builder_corpus):
        gids = np.arange(0, catalog.n_conformers, 2)
        rows = read_corpus(builder_corpus.root, merged=False)
        for xyz, ref in zip(catalog.coordinates(gids, trim=False), catalog.locate(gids)):
            assert xyz.shape == (rows.groups[ref.group_index].L_heavy, 3)

    def test_repeated_ids_decode_consistently(self, catalog):
        a, b = catalog.coordinates([1, 1])
        assert np.array_equal(a, b)

    def test_scattered_ids_decode_each_block_once(self, catalog, builder_corpus,
                                                 monkeypatch):
        from collections import Counter
        from roshambo3.format.r3g.reader import Group

        calls = []
        original = Group.decode_block

        def decode_spy(group, block_index):
            calls.append((group.index, int(block_index)))
            return original(group, block_index)

        monkeypatch.setattr(Group, "decode_block", decode_spy)
        gids = np.asarray([8, 0, 5, 2, 6, 3, 4, 1, 7])
        got = catalog.coordinates(gids)
        assert len(got) == len(gids)
        assert all(xyz.ndim == 2 and xyz.shape[1] == 3 for xyz in got)
        expected = []
        for group_index in range(len(builder_corpus.group_table)):
            with builder_corpus.group(group_index) as group:
                expected.extend((group_index, block)
                                for block in range(group.n_blocks))
        assert Counter(calls) == Counter(expected)
        assert len(calls) == len(set(calls))

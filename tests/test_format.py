"""Coordinate limits, layout constants, and the validated format-3 reader."""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.format import (BLOCK_CONF, BUCKET_EDGES, COORD_SCALE, L_ALIGN,
                              MAGIC, VERSION, bucket_of, pad_L, quantize)


class TestFormatConstants:
    def test_the_on_disk_contract_is_pinned(self):
        """These values are baked into every corpus already built on /data.
        Changing one without a format version bump silently misreads them."""
        assert MAGIC == b"R3G\0"
        assert VERSION == 3
        assert COORD_SCALE == 2.0 ** -8
        assert BLOCK_CONF == 4096
        assert L_ALIGN == 8

    def test_block_conf_is_a_whole_number_of_tiles(self):
        """Every block but a group's last tiles exactly, which is what lets the
        producer write the transposed layout straight into a pinned buffer."""
        from roshambo3.layout import BTILE
        assert BLOCK_CONF % BTILE == 0

    def test_bucket_edges_are_sorted_and_start_at_zero(self):
        assert BUCKET_EDGES[0] == 0
        assert list(BUCKET_EDGES) == sorted(BUCKET_EDGES)


class TestBucketing:
    def test_assigns_by_heavy_atom_count(self):
        assert bucket_of(np.array([0]))[0] == 0
        assert bucket_of(np.array([11]))[0] == 0
        assert bucket_of(np.array([12]))[0] == 1

    def test_is_monotonic(self):
        n = np.arange(0, 140)
        b = bucket_of(n)
        assert np.all(np.diff(b) >= 0)

    def test_equal_atom_counts_always_share_a_bucket(self):
        """All conformers of a molecule share n_real, so this is what keeps a
        molecule's conformers together -- which de-duplication relies on."""
        for n in range(1, 140):
            a = bucket_of(np.array([n, n, n]))
            assert len(set(a.tolist())) == 1


class TestPadL:
    @pytest.mark.parametrize("n,expect", [(1, 8), (8, 8), (9, 16), (32, 32), (33, 40)])
    def test_rounds_up_to_the_alignment(self, n, expect):
        assert pad_L(n) == expect

    def test_result_is_never_smaller_than_the_input(self):
        for n in range(0, 200):
            assert pad_L(n) >= n


class TestQuantize:
    def test_round_trips_within_half_a_step(self):
        r = np.random.default_rng(0).uniform(-20, 20, size=(100, 3))
        back = quantize(r).astype(np.float64) * COORD_SCALE
        assert np.abs(back - r).max() <= COORD_SCALE / 2 + 1e-12

    def test_refuses_to_clip(self):
        """int16 at the wire step covers [-128, +127.99609375] A. Conformers are centered and
        principal-axis aligned so the real extent is ~16 A, but a silent clip
        would be a wrong answer rather than a loud failure."""
        with pytest.raises(OverflowError):
            quantize(np.array([[140.0, 0.0, 0.0]]))


class TestCorpus:
    def test_reads_every_group_and_conformer(self, smoke_corpus):
        c = smoke_corpus
        assert c.n_conformers > 0
        assert sum(g.n_conf for g in c.groups()) == c.n_conformers
        assert len(c.group_table) == len(c.headers)
        seen = 0
        for group in c.groups():
            for i, block in enumerate(group.blocks):
                xyz = group.decode_block(i)
                assert xyz.dtype == np.int16
                assert xyz.shape == (int(block["rows"]), 3, group.header["L_heavy"])
                seen += xyz.shape[0]
        assert seen == c.n_conformers

    def test_sha256_ids_matches_the_descriptor(self, smoke_corpus):
        import hashlib
        ids = np.concatenate([t.column("mol_id").to_numpy()
                              for t in smoke_corpus.molecules(columns=["mol_id"])])
        ids.sort()
        got = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
        assert got == smoke_corpus.descriptor["ids_sha256"]

    def test_padding_is_zero_and_groups_have_one_size_bucket(self, smoke_corpus):
        for group in smoke_corpus.groups():
            nr = group.n_real
            assert len(set(bucket_of(nr).tolist())) == 1
            assert group.header["L_heavy"] >= int(nr.max())
            for i, block in enumerate(group.blocks):
                xyz = group.decode_block(i)
                r0 = int(block["first_row"])
                for row, count in zip(xyz, nr[r0:r0 + xyz.shape[0]]):
                    assert not row[:, int(count):].any()


class TestQuantizeRejectsWhatItCannotRepresent:
    """The range guard is only sound once non-finite values are gone.

    ``NaN > 32767`` is False, so a NaN passed the clip check and reached
    ``astype(np.int16)`` -- undefined, and in practice a plausible-looking atom
    at -128 A rather than an error.
    """

    def test_nan_is_refused(self):
        with pytest.raises(ValueError, match="not finite"):
            quantize(np.array([[0.0, 1.0, np.nan]]))

    def test_inf_is_refused(self):
        with pytest.raises(ValueError, match="not finite"):
            quantize(np.array([[0.0, 1.0, np.inf]]))

    def test_a_nan_does_not_hide_an_inf(self):
        """np.max propagates NaN, so one NaN made the whole range comparison
        False and let a co-resident Inf through with it."""
        with pytest.raises(ValueError, match="not finite"):
            quantize(np.array([[np.nan, 1.0, np.inf]]))

    def test_finite_coordinates_still_quantize(self):
        q = quantize(np.array([[0.0, 1.0, -2.5]]))
        assert q.dtype == np.int16
        assert q.tolist() == [[0, 256, -640]]


def test_public_constructor_does_not_accept_unvalidated_state(tmp_path):
    from roshambo3.format import Corpus
    with pytest.raises(TypeError):
        Corpus(str(tmp_path), descriptor={})

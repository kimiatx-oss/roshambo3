"""Molecule level vs conformer level, on the host.

The GPU half is ``test_molecule_topk_cuda.py``. Everything here is the part
that decides what the answer *means* -- the contiguity the segmented reduction
rests on, and a result API that cannot be misread -- and none of it needs a
device.
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.engine import MoleculeTopK, TopK
from roshambo3.engine.source import ConformerBlock
from roshambo3.prep.build import check_molecule_grouping, molecule_runs
from roshambo3.results import QueryHits, ScreenResult


class TestMoleculeRuns:
    def test_runs_of_equal_ids(self):
        start, end = molecule_runs(np.array([7, 7, 7, 9, 4, 4], np.int64))
        assert start.tolist() == [0, 3, 4]
        assert end.tolist() == [3, 4, 6]

    def test_empty(self):
        start, end = molecule_runs(np.empty(0, np.int64))
        assert start.size == 0 and end.size == 0


class TestContiguityIsChecked:
    """The two guarantees the molecule-level reduction is built on, which were
    produced by construction and asserted nowhere."""

    def test_a_well_formed_group_passes(self):
        mol = np.array([5, 5, 8, 9, 9, 9], np.int64)
        nr = np.array([12, 12, 7, 30, 30, 30], np.uint8)
        got = check_molecule_grouping("g", mol, nr)
        assert sorted(got.tolist()) == [5, 8, 9]

    def test_a_split_molecule_is_refused(self):
        """The failure this exists to catch: a molecule counted twice, which
        looks exactly like a legitimate result."""
        mol = np.array([5, 8, 5], np.int64)
        nr = np.array([12, 12, 12], np.uint8)
        with pytest.raises(RuntimeError, match="not contiguous"):
            check_molecule_grouping("g", mol, nr)

    def test_conformers_disagreeing_on_n_real_are_refused(self):
        mol = np.array([5, 5], np.int64)
        nr = np.array([12, 13], np.uint8)
        with pytest.raises(RuntimeError, match="heavy-atom"):
            check_molecule_grouping("g", mol, nr)

    def test_a_molecule_shared_with_the_previous_group_is_refused(self):
        mol = np.array([5, 6], np.int64)
        nr = np.array([12, 12], np.uint8)
        with pytest.raises(RuntimeError, match="straddles a group"):
            check_molecule_grouping("g", mol, nr,
                                    previous_ids=np.array([4, 5], np.int64))


class TestSinkGuards:
    def test_prepare_needs_the_sources_largest_block(self):
        """A device state sized from the first block refuses the first larger
        one partway through a screen, which is why the bound is not discovered."""
        with pytest.raises(ValueError, match="largest block"):
            MoleculeTopK(k=10).prepare(2, 0, 0)

    def test_before_block_needs_prepare(self):
        with pytest.raises(RuntimeError, match="prepare"):
            MoleculeTopK(k=10).before_block(object())

    def test_a_block_with_no_mol_id_is_refused(self):
        """A source that does not carry mol_id cannot identify molecules, and
        the sink must say so rather than reducing nothing."""
        sink = MoleculeTopK(k=10)
        sink._state = object()          # past the prepare() guard, not the GPU
        block = ConformerBlock(coords=np.zeros((32, 8, 4), np.float32),
                               counts=np.full((1, 1), 8, np.uint8), n_conformers=1,
                               L=8, bucket=0, gid_base=0)
        with pytest.raises(RuntimeError, match="mol_id"):
            sink.before_block(block)

    def test_the_two_levels_cannot_be_merged(self):
        """One list holds the best conformers and the other the best molecules;
        merging them gives neither."""
        with pytest.raises(ValueError, match="different things"):
            TopK.merge_sinks([TopK(k=10), MoleculeTopK(k=10)], k=10)

    def test_k_and_rank_col_are_validated(self):
        with pytest.raises(ValueError, match="k must be positive"):
            MoleculeTopK(k=0)
        with pytest.raises(ValueError, match="not a score column"):
            MoleculeTopK(k=10, rank_col=99)


class _Catalog:
    """Just enough Catalog for joins; durable IDs are carried on QueryHits."""

    def __init__(self, mol_of_gid):
        self.mol_of_gid = dict(mol_of_gid)

    def mol_ids(self, gids):
        return np.array([self.mol_of_gid[int(g)] for g in np.asarray(gids)],
                        np.int64)

    def molecules(self, gids, columns=None):
        m = self.mol_ids(gids)
        return {"mol_id": m.tolist(), "found": [True] * m.size,
                "ext_id": [f"m{v}" for v in m.tolist()]}


def _result(level, gids, scores, mol_of_gid):
    """Make a detached result with explicit format-3 pair identities."""
    gids = np.asarray(gids, np.int64)
    rows = np.zeros((len(gids), 20), np.float32)
    rows[:, 0] = scores
    mol_ids = np.asarray([mol_of_gid[int(gid)] for gid in gids], np.int64)
    conf_by_gid = {}
    grouped = {}
    for gid, mol_id in mol_of_gid.items():
        grouped.setdefault(int(mol_id), []).append(int(gid))
    for conformers in grouped.values():
        for conf_idx, gid in enumerate(sorted(conformers)):
            conf_by_gid[gid] = conf_idx
    conf_indices = np.asarray([conf_by_gid[int(gid)] for gid in gids], np.int64)
    hits = [QueryHits(query_name="q", query_index=0,
                      gids=gids, scores=rows, rank_col=0, level=level,
                      mol_id=mol_ids, conf_idx=conf_indices)]
    return ScreenResult(hits=hits, _catalog=_Catalog(mol_of_gid))


class TestResultApiSaysWhatItMeans:
    """`ScreenResult.molecules()` was the conformer-level join under a
    molecule-level name, so "give me the top 100 molecules" returned fewer than
    100 distinct molecules with no indication of how many fewer."""

    #: gid -> mol_id. Molecule 10 has three conformers, 11 has two.
    MOLS = {0: 10, 1: 10, 2: 10, 3: 11, 4: 11, 5: 12, 6: 13}

    def test_hit_molecules_is_one_row_per_hit(self):
        res = self._conformer_level()
        m = res.hit_molecules(n=4)
        assert m["mol_id"] == [10, 10, 10, 11]
        assert m["conf_idx"] == [0, 1, 2, 0]

    def test_molecules_refuses_a_conformer_level_result(self):
        res = self._conformer_level()
        with pytest.raises(ValueError, match="conformer-level"):
            res.molecules(n=3)

    def test_molecules_answers_a_molecule_level_result(self):
        res = _result("molecule", [0, 3, 5], [0.9, 0.8, 0.7], self.MOLS)
        assert res.molecules(n=2)["mol_id"] == [10, 11]

    def test_distinct_molecules_in_hits_reports_the_gap(self):
        res = self._conformer_level()
        d = res.distinct_molecules_in_hits(n=5)
        assert d["n_hits"] == 5
        assert d["n_molecules"] == 2
        assert d["mol_id"] == [10, 11]
        assert d["n_conformers_in_hits"] == [3, 2]

    def test_distinct_molecules_keeps_each_molecules_best_hit(self):
        res = self._conformer_level()
        d = res.distinct_molecules_in_hits()
        assert d["gid"] == [0, 3, 5, 6]

    def test_a_tie_is_broken_by_ascending_conformer_id(self):
        """Float32 ties are common at scale, and the representative conformer
        has to be chosen the same way everywhere."""
        res = _result("conformer", [4, 3, 5], [0.5, 0.5, 0.4], self.MOLS)
        d = res.distinct_molecules_in_hits()
        assert d["gid"] == [3, 5]
        assert d["conf_idx"] == [0, 0]

    def _conformer_level(self):
        return _result("conformer", [0, 1, 2, 3, 4, 5, 6],
                       [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], self.MOLS)


class TestQueryHitsCarriesItsLevel:
    def test_the_default_is_conformer(self):
        h = QueryHits("q", 0, np.zeros(1, np.int64), np.zeros((1, 20), np.float32))
        assert h.level == "conformer"
        assert h.rank_col == 0


class TestBothLevelsAreAlwaysReported:
    """The failure mode is an *omitted* level, not a miscalculated one:
    the caller asks for the level it is thinking about and the artifact simply
    does not carry the other."""

    def _pair(self):
        rng = np.random.default_rng(7)
        gold = rng.random((3, 40)).astype(np.float64)
        cand = gold + rng.normal(0, 0.01, gold.shape)
        mol = np.repeat(np.arange(10), 4)
        return gold, cand, mol

    def test_recall_report_both_reports_both(self):
        from roshambo3.metrics import recall_report_both

        gold, cand, mol = self._pair()
        out = recall_report_both(gold, cand, 5, mol_ids=mol)
        assert out["conformer"].level == "conformer"
        assert out["molecule"].level == "molecule"
        assert out["molecule"].m_targets == 10

    def test_a_missing_molecule_level_is_stated_not_omitted(self):
        from roshambo3.metrics import recall_report_both

        gold, cand, _ = self._pair()
        out = recall_report_both(gold, cand, 5)
        assert out["molecule"]["reported"] is False
        assert "mol_ids" in out["molecule"]["why"]

    def test_evaluate_primary_says_why_the_molecule_level_is_absent(self):
        import warnings

        from roshambo3.metrics import evaluate_primary

        gold, cand, _ = self._pair()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = evaluate_primary(gold, cand, n_prime=5, B=50)
        assert out["recall_molecule_level"]["reported"] is False
        assert any("METRICS.md section 3" in str(x.message) for x in w)

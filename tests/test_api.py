"""The public surface, the result object, and the CLI."""

from __future__ import annotations

import numpy as np
import pytest

import roshambo3 as r3
from roshambo3.backend import cuda as backend
from roshambo3.backend.cuda import PoseSearch

needs_gpu = pytest.mark.skipif(not backend.available(),
                               reason="no CUDA extension in this build")

@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """An actual format-3 builder corpus, independent of external datasets."""
    from roshambo3.prep import prepare
    from tests.prep_fixtures import molecule

    root = tmp_path_factory.mktemp("t297-api") / "corpus"
    mols = [molecule("CCO", "T297-api-ethanol", conformers=12),
            molecule("CCN", "T297-api-amine", conformers=12)]
    return prepare(mols, str(root), workers=1)


@pytest.fixture(scope="module")
def queries(corpus):
    """Use exact decoded corpus conformers as queries for known self hits."""
    from roshambo3.engine import from_arrays
    cat = r3.Catalog(corpus)
    gids = np.array([0, corpus.n_conf - 1])
    return gids, from_arrays(cat.coordinates(gids), [f"q{g}" for g in gids])


@pytest.fixture(scope="module")
def color_corpus(tmp_path_factory):
    from roshambo3.prep import prepare
    from tests.prep_fixtures import molecule
    from roshambo3.pharmacophore import PharmacophoreGenerator

    root = tmp_path_factory.mktemp("t298-api-color") / "corpus"
    return prepare([molecule("CCO", "T298-api-color")], str(root), workers=1,
                   color_generator=PharmacophoreGenerator())


class TestPublicSurface:
    def test_import_does_not_pull_the_cuda_extension(self):
        """The package must be usable for preparation and metrics with no GPU,
        so nothing on the import path may require the extension.

        In a subprocess, necessarily: another test in this session has almost
        certainly imported the extension already, and purging sys.modules to
        fake a clean import leaves live objects belonging to a stale module --
        which silently broke every isinstance check downstream when this test
        was first written that way.
        """
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys, roshambo3; "
             "assert 'roshambo3._roshambo3_cuda' not in sys.modules, "
             "'importing roshambo3 pulled in the CUDA extension'; "
             "assert hasattr(roshambo3, 'screen'); print('ok')"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "ok" in r.stdout

    def test_exports_the_pieces_not_just_the_facade(self):
        """A screen at scale is a system; a caller who needs to place its parts
        should not have to reach around a facade."""
        for name in ("Engine", "R3gSource", "TopK", "DenseScores",
                     "ScreenConfig", "Corpus", "Catalog", "rank_order"):
            assert hasattr(r3, name), name

    def test_load_queries_refuses_a_smiles_file(self, tmp_path):
        p = tmp_path / "q.smi"
        p.write_text("CCO ethanol\n")
        with pytest.raises(ValueError, match="3D"):
            r3.load_queries(p)

    def test_load_queries_passes_a_queryset_through(self, queries):
        _, qs = queries
        assert r3.load_queries(qs) is qs


class TestColorPreconditions:
    """``_check_color`` checks the ordinary preconditions of a color screen
    (a colored corpus and a matching generator), not a standing refusal.

    Once, every one of these raised unconditionally through one gate
    (``roshambo3/_color_gate.py``, now deleted) whose whole purpose was to
    stand until the corpus and producer sides were real. The corpus check
    below still refuses -- a shape-only corpus genuinely has no color to
    score -- but a corpus built *with* color, a real generator and the
    analytic path now succeed instead of raising.
    """

    def test_color_check_requires_a_colored_corpus_and_matching_generator(
            self, corpus, color_corpus):
        """The format-3 descriptor and query generator must agree before scoring."""
        from roshambo3.api import _check_color
        from roshambo3.pharmacophore import PharmacophoreGenerator

        gen = PharmacophoreGenerator()
        analytic = r3.ScreenConfig(color=True,
                                   pose_search=PoseSearch.ANALYTIC)

        with pytest.raises(ValueError, match="stores no color features"):
            _check_color(analytic, corpus, gen)
        with pytest.raises(ValueError, match="requires the color generator"):
            _check_color(analytic, color_corpus, None)
        assert _check_color(analytic, color_corpus, gen) is None
        assert _check_color(r3.ScreenConfig(color=True,
                                            pose_search=PoseSearch.FIELD),
                            color_corpus, gen) is None

    def test_a_shape_screen_passes_the_color_check_untouched(self, corpus):
        from roshambo3.api import _check_color
        assert _check_color(r3.ScreenConfig(), corpus, None) is None

    def test_screen_refuses_color_against_a_shape_only_corpus(self, corpus, queries):
        from roshambo3.pharmacophore import PharmacophoreGenerator
        _, qs = queries
        assert not corpus.has_color
        with pytest.raises(ValueError, match="no color features"):
            r3.screen(qs, corpus, k=5,
                      config=r3.ScreenConfig(color=True),
                      color_generator=PharmacophoreGenerator())

    def test_screen_refuses_color_without_a_generator(self, corpus, queries):
        _, qs = queries
        with pytest.raises(ValueError):
            r3.screen(qs, corpus, k=5, config=r3.ScreenConfig(color=True))

    def test_score_matrix_refuses_color_against_a_shape_only_corpus_too(
            self, corpus, queries):
        from roshambo3.pharmacophore import PharmacophoreGenerator
        _, qs = queries
        with pytest.raises(ValueError):
            r3.score_matrix(qs, corpus,
                            config=r3.ScreenConfig(
                                color=True,
                                pose_search=PoseSearch.ANALYTIC),
                            color_generator=PharmacophoreGenerator())

    def test_prepare_accepts_a_color_generator(self, tmp_path):
        """The other half of what changed: prepare() used to refuse a
        color_generator unconditionally (it would have discarded the
        features it generated). It now builds a color corpus -- see
        tests/test_prepare.py for the full round trip; this just checks the
        empty-input path no longer raises about color specifically."""
        from roshambo3.pharmacophore import PharmacophoreGenerator
        from roshambo3.prep import prepare
        with pytest.raises(ValueError, match="no inputs"):
            prepare([], str(tmp_path / "corpus"),
                    color_generator=PharmacophoreGenerator())


@pytest.fixture(scope="module")
def screened(corpus, queries):
    if not backend.available():
        pytest.skip("no CUDA extension in this build")
    gids, qs = queries
    return gids, r3.screen(qs, corpus, k=25)


@needs_gpu
class TestScreenResult:

    def test_each_query_finds_itself_first(self, screened):
        gids, res = screened
        for q, g in enumerate(gids):
            assert res[q].gids[0] == g
            assert res[q].tanimoto_shape[0] > 0.999

    def test_hits_are_in_canonical_order(self, screened):
        _, res = screened
        for h in res:
            s = h.tanimoto_combination
            assert np.all(np.diff(s) <= 0), "scores not descending"

    def test_hit_molecules_join_on_mol_id(self, screened):
        _, res = screened
        m = res.hit_molecules(query=0, n=5, columns=["ext_id", "hac", "smiles"])
        assert all(m["found"])
        assert len(m["gid"]) == 5
        assert all(isinstance(s, str) and s for s in m["smiles"])

    def test_molecules_refuses_a_conformer_level_result(self, screened):
        """`molecules()` used to be the conformer-level join under a
        molecule-level name, so "the top 100 molecules" quietly returned fewer
        than 100 distinct molecules. It now answers the question it is named
        for, and refuses a result that cannot answer it."""
        _, res = screened
        with pytest.raises(ValueError, match="conformer-level"):
            res.molecules(query=0, n=5)

    def test_distinct_molecules_in_hits_reports_both_counts(self, screened):
        _, res = screened
        d = res.distinct_molecules_in_hits(query=0, n=20)
        assert d["n_hits"] == 20
        assert 1 <= d["n_molecules"] <= 20
        assert len(d["gid"]) == d["n_molecules"]
        assert sum(d["n_conformers_in_hits"]) >= d["n_molecules"]
        assert len(set(d["mol_id"])) == d["n_molecules"]

    def test_poses_have_the_right_atom_counts(self, screened):
        _, res = screened
        poses = res.poses(query=0, n=5)
        cat = res.catalog
        for a, ref in zip(poses, cat.locate(res[0].gids[:5])):
            assert a.shape == (ref.n_real, 3)

    def test_aligned_poses_apply_a_proper_rotation(self, screened):
        """The transform must be a rigid motion. A quaternion normalization
        defect in ROSHAMBO2 made it non-orthogonal for 1.2% of poses, which
        reported Tanimotos no rigid motion can achieve."""
        _, res = screened
        raw = res.poses(query=0, n=5)
        aligned = res.aligned_poses(query=0, n=5)
        for a, b in zip(raw, aligned):
            # A rigid motion preserves all pairwise distances.
            da = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
            db = np.linalg.norm(b[:, None] - b[None, :], axis=-1)
            assert np.allclose(da, db, atol=1e-4)

    def test_self_hit_aligns_almost_exactly_onto_the_query(self, screened):
        """The best hit is the query's own conformer, so its aligned pose should
        land back on the query -- within int16 quantization and a finite
        optimizer."""
        gids, res = screened
        aligned = res.aligned_poses(query=0, n=1)[0]
        original = res.catalog.coordinates([gids[0]])[0]
        rmsd = float(np.sqrt(((aligned - original) ** 2).sum(1).mean()))
        assert rmsd < 0.1, f"self-hit RMSD {rmsd:.3f} A"

    def test_table_uses_column_names(self, screened):
        _, res = screened
        t = res.to_table(0)
        assert "tanimoto_shape" in t and "gid" in t and "query" in t
        assert len(t["gid"]) == len(res[0])


@needs_gpu
class TestScoreMatrix:
    def test_shape_and_gid_mapping(self, corpus, queries):
        _, qs = queries
        dense = r3.score_matrix(qs, corpus, groups=[0])
        with corpus.group(0) as group:
            n = group.n_conf
        assert dense.scores.shape == (qs.n_conformers, n, 20)
        assert dense.gids.shape == (n,)

    def test_size_guard_refuses_a_matrix_that_is_too_large(self, corpus, queries):
        _, qs = queries
        with pytest.raises(MemoryError):
            r3.score_matrix(qs, corpus, max_bytes=1000)


class TestCli:
    def test_cli_exposes_only_the_supported_commands(self):
        import argparse
        from roshambo3.cli import build_parser

        parser = build_parser()
        commands = next(action.choices for action in parser._actions
                        if isinstance(action, argparse._SubParsersAction))
        assert set(commands) == {"web", "info", "screen", "prepare", "verify", "definitions"}

    def test_info_prints_a_summary(self, corpus, capsys):
        from roshambo3.cli import main
        assert main(["info", corpus.root]) == 0
        out = capsys.readouterr().out
        assert "conformers" in out and "ids_sha256" in out
        assert "format          r3g v3" in out

    def test_verify_passes_on_a_built_corpus(self, corpus, capsys):
        from roshambo3.cli import main
        assert main(["verify", corpus.root, "--tier", "deep", "--all"]) == 0
        out = capsys.readouterr().out
        assert "format 3  tier=deep  OK" in out
        assert "recomputed from molecules.parquet: match" in out

    def test_verify_reports_corrupt_format3_group(self, corpus, tmp_path, capsys):
        """The CLI reports a changed group through the format-3 integrity chain."""
        import shutil
        from roshambo3.cli import main

        copy = tmp_path / "corpus"
        shutil.copytree(corpus.root, copy)
        group_path = next((copy / "groups").glob("*.r3g"))
        data = bytearray(group_path.read_bytes())
        data[64] ^= 1
        group_path.write_bytes(data)

        assert main(["verify", str(copy), "--tier", "deep", "--all"]) == 1
        captured = capsys.readouterr()
        out = captured.out + captured.err
        # The expected refusal is asserted by the format-3 validator tests;
        # here the public CLI contract is a concise refusal, never a traceback.
        assert "REFUSED" in out and "Traceback" not in out

    def test_a_library_error_becomes_a_message_not_a_traceback(self, capsys):
        from roshambo3.cli import main
        assert main(["info", "/nonexistent-corpus"]) == 1
        assert "roshambo3:" in capsys.readouterr().err

    def test_no_command_prints_help(self, capsys):
        from roshambo3.cli import main
        assert main([]) == 2
        assert "usage" in capsys.readouterr().out.lower()

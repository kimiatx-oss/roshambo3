"""Building a corpus from molecules.

The invariants here are the ones the reader, the bucketing policy and the global
conformer id all depend on, so they are checked against a corpus actually built
rather than against the format spec. The builder writes format 3: the
corpus is read back through the format-3 reader (``tests/format3_rows.py``).
"""

from __future__ import annotations

import hashlib
import os
import pathlib

import numpy as np
import pytest

from roshambo3.format.r3g.limits import BUCKET_EDGES, pad8
from roshambo3.prep import prepare, prep_conformer, self_volume, stable_id
from tests.format3_rows import STEP, corpus_files, read_corpus

COORD_SCALE = STEP


def bucket_of(n_real):
    return np.searchsorted(np.asarray(BUCKET_EDGES[1:-1]), np.asarray(n_real), side="right")

#: A multi-conformer SDF whose conformers are INTERLEAVED -- several molecules
#: appear in two or three separate runs with others in between, which is the
#: case the builder has to regroup by name rather than by adjacency. Point
#: ROSHAMBO3_TEST_SDF at one; these tests skip without it.
ORIGINAL_CONFS = pathlib.Path(os.environ.get("ROSHAMBO3_TEST_SDF", "/nonexistent/confs.sdf"))
pytestmark = pytest.mark.skipif(not ORIGINAL_CONFS.exists(),
                                reason="set ROSHAMBO3_TEST_SDF to a multi-conformer SDF to run these")


@pytest.fixture(scope="module")
def CONFS(tmp_path_factory):
    """The test SDF with each protonation state under its own id.

    ROSHAMBO2's ``test/confs.sdf`` puts two protonation states of 70 of its
    480 molecules under one name, which a format-3 build refuses; renaming the later states' title lines is the fix the refusal
    asks for, and keeps every coordinate byte and the interleaving."""
    from tests.prep_fixtures import split_variant_ids
    path = tmp_path_factory.mktemp("confs") / "confs_states.sdf"
    split_variant_ids(ORIGINAL_CONFS, path)
    return path


@pytest.fixture(scope="module")
def source_molecules(CONFS):
    """The source SDF, regrouped by name -- which is how the builder reads it."""
    from rdkit import Chem
    src: dict = {}
    for m in Chem.SDMolSupplier(str(CONFS), removeHs=False):
        if m is not None:
            src.setdefault(m.GetProp("_Name"), []).append(m)
    return src


@pytest.fixture(scope="module")
def built(tmp_path_factory, CONFS):
    out = tmp_path_factory.mktemp("corpus") / "built"
    prepare(str(CONFS), str(out), name="TEST", workers=4, records_per_unit=500)
    return read_corpus(out)


def prepared_in_canonical_order(mol):
    """What the builder stores for a single-conformer source record, before
    quantization: its heavy atoms put in canonical SMILES order through the
    permutation ``canonical_heavy`` exposes, then prepared."""
    from roshambo3.prep.conformer import strip_hydrogens
    from roshambo3.prep.sources import canonical_heavy
    _, _, order = canonical_heavy(mol)
    source = np.asarray(strip_hydrogens(mol).GetConformer().GetPositions(),
                        dtype=np.float64)
    return prep_conformer(source[order])


class TestCounts:
    def test_every_molecule_and_conformer_is_kept(self, built, source_molecules):
        assert built.n_mol == len(source_molecules)
        assert built.n_conf == sum(len(v) for v in source_molecules.values())

    def test_group_counts_sum_to_the_descriptor(self, built):
        assert sum(g.n_conf for g in built.groups) == built.n_conf
        assert sum(g.n_mol for g in built.groups) == built.n_mol

    def test_ids_sha256_is_the_hash_of_the_sorted_ids(self, built):
        """FORMAT.md 10.6, and the format-2 ``sha256_ids`` definition: the
        sorted unique mol_id list as little-endian int64."""
        ids = np.asarray(built.molecules["mol_id"], dtype=np.uint64)
        assert ids.size == built.n_mol
        assert np.all(np.diff(ids.astype(np.int64)) > 0), "ids must be sorted and unique"
        got = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
        assert got == built.descriptor["ids_sha256"]
        assert np.array_equal(np.sort(np.unique(built.mol_id)), ids.astype(np.int64))


class TestInterleavedConformers:
    """The defect this file was written to catch.

    Real SDFs interleave: in this one, 16 of 480 molecules have their conformers
    split across two or three runs with other molecules in between. Grouping by
    adjacency reads those as several single-conformer molecules that then collide
    on one mol_id.
    """

    def test_the_source_really_does_interleave(self, source_molecules, CONFS):
        from roshambo3.prep.sources import index_sdf
        _, titles = index_sdf(str(CONFS))
        runs = []
        for t in titles:
            if not runs or runs[-1] != t:
                runs.append(t)
        assert len(runs) > len(set(runs)), \
            "this test is pointless unless the fixture actually interleaves"

    def test_interleaved_conformers_end_up_in_one_molecule(self, built,
                                                           source_molecules):
        ids, counts = np.unique(built.mol_id, return_counts=True)
        by_id = dict(zip(ids.tolist(), counts.tolist()))
        for name, mols in source_molecules.items():
            assert by_id[stable_id(name)] == len(mols), name
        runs = np.concatenate([g.runs for g in built.groups])
        assert runs.size == np.unique(runs["mol_id"]).size, "one run per molecule"

    def test_conf_idx_is_contiguous_from_zero_per_molecule(self, built):
        for g in built.groups:
            for m in range(g.n_mol):
                r0, r1 = g.run_start[m], g.run_start[m + 1]
                assert g.conf_idx[r0:r1].tolist() == list(range(r1 - r0))


class TestBucketing:
    def test_every_group_holds_exactly_one_bucket(self, built):
        """Single-bucket launches are what make the per-bucket lattice spacing
        and register budget free."""
        for g in built.groups:
            assert set(bucket_of(g.n_real).tolist()) == {g.bucket}

    def test_padded_width_covers_the_widest_conformer(self, built):
        for g in built.groups:
            assert g.L_heavy == pad8(int(g.n_real.max()))
            assert int(g.n_real.max()) <= g.L

    def test_conformers_are_sorted_by_atom_count_within_a_group(self, built):
        """Sorting within a bucket is worth a measured 1.31-1.55x from reduced
        warp divergence, and it comes free from sharding on the exact count."""
        for g in built.groups:
            assert np.all(np.diff(g.n_real.astype(int)) >= 0)

    def test_rows_are_in_content_order(self, built):
        """Corpus order is ``(bucket, n_real, mol_id, conf_idx)``
        (FORMAT.md 9.2): within a group and across the groups of a bucket,
        not n_real, then worker slot, then source order."""
        key = np.stack([bucket_of(built.n_real), built.n_real.astype(np.int64),
                        built.mol_id.astype(np.int64), built.conf_idx.astype(np.int64)],
                       axis=1)
        step = np.diff(key, axis=0)
        first_change = np.argmax(step != 0, axis=1)
        assert np.all(step[np.arange(len(step)), first_change] > 0)

    def test_a_molecules_conformers_share_a_group(self, built):
        """Bucketing on n_real keeps them together; the group split must not
        undo it, because de-duplication downstream relies on it."""
        seen: dict = {}
        for gi, g in enumerate(built.groups):
            for mid in set(g.mol_id.tolist()):
                assert mid not in seen, f"molecule {mid} spans groups {seen[mid]} and {gi}"
                seen[mid] = gi

    def test_every_block_is_molecule_aligned(self, built):
        """FORMAT.md 6 rule 1: a block holds whole molecules; at the default
        cap of 4,096 rows."""
        from roshambo3.format.r3g.limits import LIMITS
        for g in built.groups:
            start = g.run_start
            for b in g.blocks:
                assert int(b["first_row"]) == start[int(b["m0"])]
                assert int(b["first_row"]) + int(b["rows"]) == start[int(b["m1"])]
                assert 1 <= int(b["rows"]) <= LIMITS["block_rows"]


class TestRoundTrip:
    def test_stored_coordinates_match_a_fresh_prep(self, built, source_molecules):
        """End to end: what came out of the builder is what preparing the source
        molecule again produces, to within int16 quantization."""
        rng = np.random.default_rng(0)
        gids = np.sort(rng.choice(built.n_conf, min(120, built.n_conf), replace=False))
        mol_id, conf_idx = built.mol_id, built.conf_idx
        worst = 0.0
        extent = 0.0
        for gid in gids:
            name = built.molecule_row(mol_id[gid])["ext_id"]
            want = prepared_in_canonical_order(source_molecules[name][int(conf_idx[gid])])
            stored = built.coordinates(int(gid))
            assert want.shape == stored.shape
            worst = max(worst, float(np.abs(want - stored).max()))
            extent = max(extent, float(np.abs(want).max()))
        # half the quantization step, plus what float32 dequantization adds at
        # this coordinate extent
        bound = COORD_SCALE / 2 + np.finfo(np.float32).eps * extent
        assert worst <= bound, f"{worst} > {bound}"

    def test_the_vaa_section_matches_a_recomputed_self_overlap(self, built,
                                                               source_molecules):
        """``vaa`` describes the stored (quantized) coordinates
        (FORMAT.md 18.1): one frame, recomputed here from the source."""
        mol_id, conf_idx, vaa = built.mol_id, built.conf_idx, built.vaa
        for gid in range(0, built.n_conf, 97):
            name = built.molecule_row(mol_id[gid])["ext_id"]
            want = prepared_in_canonical_order(source_molecules[name][int(conf_idx[gid])])
            stored = np.rint(want / COORD_SCALE).astype(np.int16).astype(np.float32) \
                * np.float32(COORD_SCALE)
            assert abs(self_volume(stored) - vaa[gid]) / vaa[gid] < 1e-6

    def test_hac_in_the_molecule_table_equals_n_real(self, built):
        """It does not on the prebuilt tiers because their source
        descriptor disagreed with the prepared geometry. A corpus built here
        computes both from the same molecule, so they must agree."""
        runs = np.concatenate([g.runs for g in built.groups])
        order = np.argsort(runs["mol_id"])
        assert np.array_equal(np.asarray(built.molecules["mol_id"]), runs["mol_id"][order])
        assert np.array_equal(np.asarray(built.molecules["hac"]), runs["n_real"][order])


class TestDeterminism:
    def test_the_build_does_not_depend_on_the_worker_count(self, tmp_path, CONFS):
        """A corpus rebuilt with different parallelism must be the same corpus,
        or no result can cite it: every file but ``build.json`` (provenance:
        timings, worker counts, paths) is byte-identical."""
        prepare(str(CONFS), str(tmp_path / "a"), name="A", workers=1, records_per_unit=500)
        prepare(str(CONFS), str(tmp_path / "b"), name="B", workers=4, records_per_unit=500)
        fa, fb = corpus_files(tmp_path / "a"), corpus_files(tmp_path / "b")
        assert sorted(fa) == sorted(fb)
        fa.pop("build.json"), fb.pop("build.json")
        assert len(fa) > 3
        for name in fa:
            assert fa[name] == fb[name], name


class TestRefusals:
    def test_a_smiles_file_is_refused(self, tmp_path):
        p = tmp_path / "in.smi"
        p.write_text("CCO ethanol\n")
        with pytest.raises(ValueError, match="no 3D coordinates"):
            prepare(str(p), str(tmp_path / "out"))

    def test_building_into_an_existing_corpus_is_refused(self, built, tmp_path, CONFS):
        """Overwriting in place would leave a directory that is half two
        datasets, described by neither descriptor."""
        with pytest.raises(FileExistsError):
            prepare(str(CONFS), built.root)

    def test_no_inputs_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            prepare([], str(tmp_path / "out"))

    def test_a_missing_input_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare(str(tmp_path / "absent.sdf"), str(tmp_path / "out"))


@pytest.mark.skipif(not __import__("roshambo3.backend.cuda",
                                   fromlist=["available"]).available(),
                    reason="no CUDA extension in this build")
class TestScreenAgainstABuiltCorpus:
    def test_a_source_conformer_finds_itself(self, built):
        """The whole loop: prepare, then screen, then the hit resolves back to
        the molecule it came from."""
        import roshambo3 as r3
        from roshambo3.engine import from_arrays
        from roshambo3.format.r3g.reader import open_corpus

        gids = np.array([5, 2000])
        qs = from_arrays([built.coordinates(int(g)) for g in gids], ["a", "b"])
        res = r3.screen(qs, open_corpus(built.root), k=5)
        for q, g in enumerate(gids):
            assert res[q].gids[0] == g
            assert res[q].tanimoto_shape[0] > 0.999


class TestPreparationIsTransactional:
    """A build that fails must leave nothing a later run can mistake for data.

    The old guard keyed on ``manifest.json`` (format 2's last file) -- so a
    build killed during the gather left group files behind AND passed the check
    on the next run, which overwrote them by name and orphaned any it did not
    reproduce.
    """

    def test_a_non_empty_directory_is_refused(self, tmp_path, CONFS):
        out = tmp_path / "corpus"
        out.mkdir()
        (out / "groups").mkdir()        # the wreckage of an interrupted build
        with pytest.raises(FileExistsError, match="not empty"):
            prepare(str(CONFS), str(out), name="X", records_per_unit=500)

    def test_a_failed_build_leaves_no_directory(self, tmp_path):
        out = tmp_path / "corpus"
        with pytest.raises(Exception):
            prepare(["/nonexistent/missing.sdf"], str(out), name="X")
        assert not out.exists(), "a failed build left a directory behind"
        leftovers = [p for p in tmp_path.iterdir() if "building" in p.name]
        assert not leftovers, f"staging directories left behind: {leftovers}"

    def test_a_successful_build_lands_at_the_target(self, built):
        """Only the four format-3 names (FORMAT.md 1)."""
        root = pathlib.Path(built.root)
        assert sorted(p.name for p in root.iterdir()) == [
            "build.json", "corpus.json", "groups", "molecules.parquet"]
        assert all(p.suffix == ".r3g" for p in (root / "groups").iterdir())
        assert not list(root.parent.glob("*.partial"))


class TestTheBuilderRejectsMoleculesItCannotStore:
    """Rejection is per molecule, never per conformer: bucketing keeps a
    molecule's conformers together because they share n_real, and molecule-level
    results depend on that. Half a molecule would break both quietly."""

    def _rec(self, n_conf=1, n_real=5, coords=None, ext_id="m1", **meta):
        import types
        confs = []
        for _ in range(n_conf):
            r = np.zeros((n_real, 3)) if coords is None else coords
            confs.append((r, n_real, 1.0))
        meta = {"hac": n_real, "smiles": "C" * n_real, "mw": 72.15, "rotor_count": 2.0,
                **meta}
        return types.SimpleNamespace(mol_id=1, ext_id=ext_id, conformers=confs, meta=meta)

    def test_a_storable_molecule_is_accepted(self):
        from roshambo3.prep.build import _why_reject
        assert _why_reject(self._rec()) is None

    def test_non_finite_coordinates_are_rejected(self):
        from roshambo3.prep.build import _why_reject
        r = np.zeros((5, 3))
        r[1, 1] = np.nan
        assert "non-finite" in _why_reject(self._rec(coords=r))

    def test_too_many_heavy_atoms_is_rejected(self):
        from roshambo3.prep.build import _why_reject
        assert "n_real of the run table" in _why_reject(self._rec(n_real=300))

    def test_too_many_conformers_is_rejected(self):
        """The format-3 run table's limit (FORMAT.md 7), named
        from the format's constant, never restated."""
        from roshambo3.format.r3g.limits import LIMITS
        from roshambo3.prep.build import _why_reject
        cap = LIMITS["n_conf_per_molecule"]
        assert _why_reject(self._rec(n_conf=cap)) is None
        why = _why_reject(self._rec(n_conf=cap + 1))
        assert f"{cap + 1} conformers" in why and f"limit of {cap}" in why
        assert "run table" in why

    def test_a_coordinate_outside_the_int16_range_is_rejected_not_clipped(self):
        """FORMAT.md 4.1: [-128, +127.996] A at 2^-8 A; refused per molecule
        at build (it was an OverflowError that stopped the build)."""
        from roshambo3.prep.build import _why_reject
        r = np.zeros((5, 3))
        r[2, 0] = 127.99
        assert _why_reject(self._rec(coords=r)) is None
        r[2, 0] = 128.0
        assert "int16 range" in _why_reject(self._rec(coords=r))
        r[2, 0] = -128.002
        assert "int16 range" in _why_reject(self._rec(coords=r))

    @pytest.mark.parametrize("ext_id,why", [
        ("", "no identifier"), ("a\x07b", "control"), ("x\u202ey", "bidi"),
        ("é" * 129, "at most 256")])
    def test_an_identifier_molecules_parquet_cannot_hold_is_rejected(self, ext_id, why):
        from roshambo3.prep.build import _why_reject
        assert why in _why_reject(self._rec(ext_id=ext_id))

    @pytest.mark.parametrize("smiles", ["", "CC |$;$|", "C C", "C" * 4097, "CCé"])
    def test_a_smiles_molecules_parquet_cannot_hold_is_rejected(self, smiles):
        from roshambo3.prep.build import _why_reject
        assert "SMILES" in _why_reject(self._rec(smiles=smiles))

    @pytest.mark.parametrize("field,value", [("mw", None), ("mw", float("nan")),
                                             ("rotor_count", None), ("rotor_count", 1.5),
                                             ("rotor_count", 70000.0)])
    def test_a_missing_or_unrepresentable_descriptor_is_rejected(self, field, value):
        from roshambo3.prep.build import _why_reject
        assert _why_reject(self._rec(**{field: value})) is not None

    def test_conformers_disagreeing_on_heavy_atoms_are_rejected(self):
        from roshambo3.prep.build import _why_reject
        rec = self._rec(n_conf=2)
        rec.conformers[1] = (np.zeros((6, 3)), 6, 1.0)
        assert "disagree" in _why_reject(rec)


@pytest.fixture(scope="module")
def color_built(tmp_path_factory, CONFS):
    """A color corpus built with a group cap small enough to split a bucket.

    Module scope, and a plain function rather than a class-scoped method:
    pytest warns that a class-scoped fixture defined as an instance method sees
    a different instance from the tests that use it, and a warning nobody reads
    is how a fixture that silently rebuilds per test gets missed.
    """
    pytest.importorskip("rdkit")
    from roshambo3.pharmacophore import PharmacophoreGenerator
    out = tmp_path_factory.mktemp("t88color") / "built"
    prepare(str(CONFS), str(out), name="T88-COLOR", workers=4,
            records_per_unit=500, max_conf_per_group=700,
            color_generator=PharmacophoreGenerator())
    return read_corpus(out)


def _bucket_groups(rows):
    by_bucket: dict = {}
    for g in rows.groups:
        by_bucket.setdefault(g.bucket, []).append(g)
    return by_bucket


class TestPerGroupWidthAndGroups:
    """Each group is padded to its own worst case, and groups follow
    the format-3 group write rule (FORMAT.md 9.3).

    The device width ``L`` is per group, ``pad8(max(n_real + n_color))``;
    the query-field width ``L_field_width`` is per bucket. Checking either
    needs a build with a group cap small enough to actually split a bucket.
    """

    def test_the_cap_actually_split_a_bucket(self, color_built):
        """Otherwise everything below passes vacuously: with one group per
        bucket, per-group L IS the bucket width and there is nothing to test."""
        assert max(len(v) for v in _bucket_groups(color_built).values()) > 1

    def test_groups_follow_the_write_rule(self, color_built):
        """A molecule joins the open group iff it is empty or the group stays
        within ``max_conf_per_group`` (700 here): so every group but a
        bucket's last is full, in the sense that its next molecule would not
        have fit."""
        cap = color_built.build["parameters"]["max_conf_per_group"]
        assert cap == 700
        for groups in _bucket_groups(color_built).values():
            for g, nxt in zip(groups, groups[1:]):
                assert g.n_conf <= cap or g.n_mol == 1
                assert g.n_conf + int(nxt.runs["n_conf"][0]) > cap

    def test_each_group_is_padded_to_its_own_worst_case(self, color_built):
        for g in color_built.groups:
            cnt = g.counts.astype(np.int64)
            assert g.L == pad8(int((cnt[:, 0] + cnt[:, 1:].sum(axis=1)).max()))
            assert g.L_heavy == pad8(int(cnt[:, 0].max()))

    def test_no_conformer_overflows_its_groups_record(self, color_built):
        for g in color_built.groups:
            cnt = g.counts.astype(np.int64)
            assert int((cnt[:, 0] + cnt[:, 1:].sum(axis=1)).max()) <= g.L

    def test_no_molecule_is_split_across_groups(self, color_built):
        seen: dict = {}
        for gi, g in enumerate(color_built.groups):
            for mid in set(g.mol_id.tolist()):
                assert mid not in seen, \
                    f"molecule {mid} spans groups {seen[mid]} and {gi}"
                seen[mid] = gi

    def test_the_field_policy_width_is_per_bucket_not_per_group(self, color_built):
        """`L` moved; the query-field policy's proxy for how big the molecules
        are must not (FORMAT.md 8.6: pad8 of the bucket's largest n_real)."""
        for b, groups in _bucket_groups(color_built).items():
            widths = {g.L_field_width for g in groups}
            assert widths == {pad8(max(int(g.n_real.max()) for g in groups))}, b

    def test_the_count_vector_is_the_feature_tables(self, color_built):
        """FORMAT.md 8.5: column 0 n_real, column t the molecule's type-t
        features; and the merged layout's pseudo-atoms are the derived ones."""
        for g in color_built.groups:
            for m, feats in enumerate(g.features):
                r0 = g.run_start[m]
                want = np.bincount([t for t, _ in feats], minlength=7)[1:]
                assert g.counts[r0, 0] == g.n_real[r0]
                assert np.array_equal(g.counts[r0, 1:], want)

    def test_verify_deep_passes(self, color_built):
        from roshambo3.cli import main
        assert main(["verify", color_built.root, "--tier", "deep", "--all"]) == 0

    def test_the_width_is_not_the_sum_of_the_two_maxima(self, color_built):
        """The defect this test exists for, caught once already: a planner that
        tracks `max(n_real)` and `max(n_color)` separately and adds them is
        charging the split record's width while writing the merged record's
        layout. Those are two independent worst cases and the group's widest
        molecule is usually not its most-featured one, so at least one group
        must come in strictly under the sum.
        """
        strictly_under = 0
        for g in color_built.groups:
            cnt = g.counts.astype(np.int64)
            total = int((cnt[:, 0] + cnt[:, 1:].sum(axis=1)).max())
            summed = int(cnt[:, 0].max()) + int(cnt[:, 1:].sum(axis=1).max())
            assert g.L == pad8(total) <= pad8(summed)
            strictly_under += int(total < summed)
        assert strictly_under > 0

"""An SDF molecule stored one record per conformer takes the
multi-conformer (feature-plan) branch, and nothing it produces changes.

Every test runs ``SdfSource`` with ``merge_conformers=True`` and ``False`` over
the same file and requires every ``MoleculeRecord`` field -- ext_id, mol_id,
meta, all five members of every conformer tuple, the per-conformer atom permutations -- and the ``failures`` list to be equal, bit for
bit.

Records of one name that differ only in atom order merge (each is
put in the first record's atom order first, on both paths); records that
differ in more than stereo or atom order -- a charge or protonation variant --
are refused with ``BuildRefused`` naming the id, on both paths, instead of
falling back to per-record preparation under one id.
"""
import io
import os

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import MolSanitizeException

from roshambo3.pharmacophore import (CustomSMARTSPharmacophoreGenerator,
                                     PharmacophoreGenerator)
from roshambo3.prep import sources
from roshambo3.prep.sources import MAX_MERGED_CONFORMERS, SdfSource

RDLogger.DisableLog("rdApp.*")

#: Real one-record-per-conformer SDF shards (e.g. a LIT-PCBA or DUD-E shard),
#: opt-in: ``ROSHAMBO3_SDF_SLICES`` is an ``os.pathsep``-separated list of
#: paths. Unset, the test that reads them skips.
REAL_SDF_SLICES = [p for p in os.environ.get("ROSHAMBO3_SDF_SLICES", "").split(os.pathsep) if p]


# --------------------------------------------------------------------------
# comparison

def _same_value(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return (a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes() == b.tobytes())
    if isinstance(a, float) or isinstance(b, float):
        return (type(a) is type(b)
                and np.float64(a).tobytes() == np.float64(b).tobytes())
    return type(a) is type(b) and a == b


def assert_records_identical(left, right):
    assert len(left) == len(right)
    for ra, rb in zip(left, right):
        assert ra.ext_id == rb.ext_id
        assert ra.mol_id == rb.mol_id
        assert ra.meta.keys() == rb.meta.keys(), ra.ext_id
        for k in ra.meta:
            assert _same_value(ra.meta[k], rb.meta[k]), (ra.ext_id, k)
        assert len(ra.conformers) == len(rb.conformers), ra.ext_id
        for ci, (ca, cb) in enumerate(zip(ra.conformers, rb.conformers)):
            assert len(ca) == len(cb)
            for j, (xa, xb) in enumerate(zip(ca, cb)):
                assert _same_value(xa, xb), (ra.ext_id, ci, j)
        assert len(ra.atom_order) == len(ra.conformers) == len(rb.atom_order)
        for oa, ob in zip(ra.atom_order, rb.atom_order):
            assert _same_value(oa, ob), ra.ext_id
        # The feature table, compared in FORMAT.md 8.1 rule 4 order:
        # the two paths may enumerate a type's features in different orders
        # and the builder sorts them.
        assert (ra.features is None) == (rb.features is None), ra.ext_id
        if ra.features is not None:
            assert sorted(ra.features) == sorted(rb.features), ra.ext_id


def run_source(path, generator, merge):
    src = SdfSource([str(path)], color_generator=generator,
                    merge_conformers=merge)
    recs, failures = [], []
    for unit in src.units(target_records=7):
        recs.extend(src.iter_unit(unit))
        failures.extend(src.failures)
    return recs, failures


def both_paths(path, generator):
    on = run_source(path, generator, True)
    off = run_source(path, generator, False)
    assert_records_identical(on[0], off[0])
    assert on[1] == off[1]
    return on


# --------------------------------------------------------------------------
# synthetic fixture

def embedded(smiles, name, n, seed=13):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n, randomSeed=seed)
    assert len(ids) == n
    mol.SetProp("_Name", name)
    return mol


def record_blocks(mol):
    """One SDF record per conformer, as LIT-PCBA and PUR2 store them."""
    out = []
    for conf in mol.GetConformers():
        sio = io.StringIO()
        with Chem.SDWriter(sio) as w:
            w.write(mol, confId=conf.GetId())
        out.append(sio.getvalue())
    return out


def unparseable_block(name):
    """A record that ForwardSDMolSupplier(sanitize=True) returns None for."""
    bad = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)   # 5-valent C
    bad.UpdatePropertyCache(strict=False)
    AllChem.Compute2DCoords(bad)
    bad.SetProp("_Name", name)
    sio = io.StringIO()
    with Chem.SDWriter(sio) as w:
        w.SetKekulize(False)
        w.write(bad)
    block = sio.getvalue()
    assert next(Chem.ForwardSDMolSupplier(io.BytesIO(block.encode()),
                                          removeHs=False, sanitize=True)) is None
    return block


def stereo_varying(name="stereo-varies", n=8):
    """An unspecified stereocenter embedded both ways: SDF parsing assigns a
    different chiral tag to different records of the one molecule."""
    mol = embedded("CC(N)C(=O)Oc1ccccc1", name, n, seed=7)
    blocks = record_blocks(mol)
    tags = set()
    for b in blocks:
        m = next(Chem.ForwardSDMolSupplier(io.BytesIO(b.encode()), removeHs=False))
        tags.add(tuple(int(a.GetChiralTag()) for a in m.GetAtoms()))
    assert len(tags) > 1, "fixture premise: the records disagree on chirality"
    return blocks


@pytest.fixture
def synthetic_sdf(tmp_path):
    multi = record_blocks(embedded("CCOc1ccccc1C(=O)N", "multi", 6))
    single = record_blocks(embedded("c1ccncc1CCO", "single", 1))
    # Two protonation states, each under its own id: they merge separately.
    # (Under one id they are refused: test_variant_under_one_id_is_refused.)
    proto = (record_blocks(embedded("CCCN", "proto", 2))
             + record_blocks(embedded("CCC[NH3+]", "proto-h", 2, seed=5)))
    # Same name, same atoms, different atom order: merges, the second
    # record put in the first record's atom order.
    a = embedded("OCCN", "order", 2)
    b = Chem.RenumberAtoms(a, list(reversed(range(a.GetNumAtoms()))))
    b.SetProp("_Name", "order")
    order = record_blocks(a) + record_blocks(b)
    inter_a = record_blocks(embedded("CC(=O)Nc1ccc(O)cc1", "inter-a", 4))
    inter_b = record_blocks(embedded("OC(=O)c1ccccc1", "inter-b", 3))
    stereo = stereo_varying()
    blocks = (multi[:3] + [unparseable_block("multi")] + multi[3:]
              + single + proto
              + inter_a[:2] + inter_b[:2] + inter_a[2:] + inter_b[2:]
              + order + stereo)
    path = tmp_path / "fixture.sdf"
    path.write_text("".join(blocks))
    return path


# --------------------------------------------------------------------------
# tests

@pytest.mark.parametrize("color", [True, False])
def test_merged_path_identical_on_fixture(synthetic_sdf, color):
    gen = PharmacophoreGenerator() if color else None
    recs, failures = both_paths(synthetic_sdf, gen)
    names = [r.ext_id for r in recs]
    assert sorted(names) == sorted(["multi", "single", "proto", "proto-h", "inter-a",
                                    "inter-b", "order", "stereo-varies"])
    by = {r.ext_id: r for r in recs}
    assert len(by["multi"].conformers) == 6          # the unparseable one skipped
    assert len(by["inter-a"].conformers) == 4
    assert len(by["proto"].conformers) == 2
    assert len(by["proto-h"].conformers) == 2
    assert len(by["order"].conformers) == 4
    # The atom-order variant's permutation is its own source order's: the
    # reversed record's order is the first's, read backwards.
    n = by["order"].conformers[0][1]
    assert np.array_equal(by["order"].atom_order[2], n - 1 - by["order"].atom_order[0])
    assert failures == []


def test_merge_is_taken_and_guard_falls_back(synthetic_sdf, monkeypatch):
    """The merged path is really exercised: the graph is canonicalized once
    per merged molecule, and per record for every molecule the guard
    refuses. Features are perceived once per molecule on either path, on the
    parsed stored SMILES (no conformers); positions are never computed
    here."""
    gen = PharmacophoreGenerator()
    calls = {"merged": 0, "per_record": 0, "perceived": 0, "positions": 0}
    orig_plan, orig_gen = gen.make_feature_plan, gen.generate_color_atoms
    orig_canonical = sources.canonical_heavy

    def canonical(mol):
        calls["merged" if mol.GetNumConformers() > 1 else "per_record"] += 1
        return orig_canonical(mol)

    def plan(mol):
        assert mol.GetNumConformers() == 0
        calls["perceived"] += 1
        return orig_plan(mol)

    def per_conf(mol):
        calls["positions"] += 1
        return orig_gen(mol)

    monkeypatch.setattr(sources, "canonical_heavy", canonical)
    monkeypatch.setattr(gen, "make_feature_plan", plan)
    monkeypatch.setattr(gen, "generate_color_atoms", per_conf)
    run_source(synthetic_sdf, gen, True)
    # merged: multi, inter-a, inter-b, stereo-varies, order (atom order
    # only), proto, proto-h -> one canonicalization each. per record: single
    # (1). Perceived: every one of the 8 molecules, once.
    assert calls == {"merged": 7, "per_record": 1, "perceived": 8, "positions": 0}


def test_guard_each_difference_refuses():
    base = embedded("CC(N)C(=O)O", "g", 2)
    ok = [next(Chem.ForwardSDMolSupplier(io.BytesIO(b.encode()), removeHs=False))
          for b in record_blocks(base)]
    merged, same = sources._merge_records(ok + [None], stereo_independent=True)
    assert merged is not None and merged.GetNumConformers() == 2
    assert [c.GetId() for c in merged.GetConformers()] == [0, 1]
    for ci, m in enumerate(ok):
        assert (merged.GetConformer(ci).GetPositions()
                == m.GetConformer().GetPositions()).all()

    def variant(edit):
        m = Chem.RWMol(ok[1])
        edit(m)
        m = m.GetMol()
        m.UpdatePropertyCache(strict=False)
        return m

    n_idx = next(a.GetIdx() for a in ok[0].GetAtoms() if a.GetSymbol() == "N")
    c_idx = next(a.GetIdx() for a in ok[0].GetAtoms() if a.GetSymbol() == "C")
    edits = {
        "charge": lambda m: m.GetAtomWithIdx(n_idx).SetFormalCharge(1),
        "isotope": lambda m: m.GetAtomWithIdx(c_idx).SetIsotope(13),
        "element": lambda m: m.GetAtomWithIdx(n_idx).SetAtomicNum(8),
        "explicit_h": lambda m: m.GetAtomWithIdx(n_idx).SetNumExplicitHs(3),
        "bond_type": lambda m: m.GetBondWithIdx(0).SetBondType(Chem.BondType.DOUBLE),
        "aromatic_flag": lambda m: m.GetBondWithIdx(0).SetIsAromatic(True),
    }
    # Not one graph: not merged here; the per-record path then refuses them
    # (test_variant_under_one_id_is_refused) -- the one place the variant
    # rule is decided, on the hydrogen-free graph.
    for what, edit in edits.items():
        assert sources._merge_records([ok[0], variant(edit)], stereo_independent=True) == (None, None), what
    # Atom order only: merged, in the first record's atom order.
    order = list(reversed(range(ok[1].GetNumAtoms())))
    perm = Chem.RenumberAtoms(ok[1], order)
    perm.SetProp("_Name", "g")
    # (Strict mode compares full keys, bond list order included, so there an
    # aligned record may still take the per-record path, which accepts it.)
    merged, same = sources._merge_records([ok[0], perm], stereo_independent=True)
    assert merged is not None and merged.GetNumConformers() == 2, "atom order"
    heavy = [a.GetIdx() for a in ok[1].GetAtoms() if a.GetAtomicNum() != 1]
    assert np.array_equal(merged.GetConformer(1).GetPositions()[heavy],
                          ok[1].GetConformer().GetPositions()[heavy])
    other_name = Chem.Mol(ok[1])
    other_name.SetProp("_Name", "h")
    assert sources._merge_records([ok[0], other_name], stereo_independent=True) == (None, None), "name"
    removed = Chem.RWMol(ok[1])
    removed.RemoveAtom(removed.GetNumAtoms() - 1)
    assert sources._merge_records([ok[0], removed.GetMol()], stereo_independent=True) == (None, None), "atom count"
    # Chirality differs: merged, but the stereo agreement flag is false.
    chiral = Chem.Mol(ok[1])
    center = chiral.GetAtomWithIdx(1)
    ref_tag = ok[0].GetAtomWithIdx(1).GetChiralTag()
    center.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                        if ref_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW
                        else Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    merged, same = sources._merge_records([ok[0], chiral], stereo_independent=True)
    assert merged is not None and same is False
    assert sources._merge_records([ok[0]], stereo_independent=True) == (None, None), "one record"


def _bad_name_raises(monkeypatch):
    """Inject RDKit's sanitize family into the heavy-atom view of the molecule
    named 'bad', keyed on the graph's name so both paths see it identically."""
    orig = sources.strip_hydrogens

    def strip(mol):
        if mol.HasProp("_Name") and mol.GetProp("_Name") == "bad":
            raise MolSanitizeException("injected: Can't kekulize mol")
        return orig(mol)

    monkeypatch.setattr(sources, "strip_hydrogens", strip)


def test_reject_semantics_identical_when_heavy_view_raises(tmp_path, monkeypatch):
    """(a): a molecule whose heavy-atom view raises is rejected per record,
    with the same failures entries, never a crash -- on both paths."""
    _bad_name_raises(monkeypatch)
    blocks = (record_blocks(embedded("CCOc1ccccc1", "good", 3))
              + record_blocks(embedded("c1ccc[nH]1", "bad", 4))
              + record_blocks(embedded("CCN", "after", 2)))
    path = tmp_path / "bad.sdf"
    path.write_text("".join(blocks))
    recs, failures = both_paths(path, PharmacophoreGenerator())
    assert [r.ext_id for r in recs] == ["good", "after"]
    assert failures == [("bad", "MolSanitizeException: injected: Can't kekulize mol")] * 4


def test_reject_semantics_identical_when_plan_raises(tmp_path, monkeypatch):
    """A sanitize-family error from perception rejects the molecule whole,
    one ``failures`` entry, on both paths: perception runs once per molecule,
    on its parsed stored SMILES, so it is keyed here on that graph."""
    gen = PharmacophoreGenerator()
    orig_plan = gen.make_feature_plan
    bad = Chem.CanonSmiles("CCOc1ccccc1")

    def boom(mol):
        if Chem.MolToSmiles(Chem.RemoveHs(mol)) == bad:
            raise MolSanitizeException("injected feature failure")
        return orig_plan(mol)

    monkeypatch.setattr(gen, "make_feature_plan", boom)
    path = tmp_path / "bad.sdf"
    path.write_text("".join(record_blocks(embedded("CCOc1ccccc1", "bad", 3))
                            + record_blocks(embedded("CCN", "ok", 3))))
    recs, failures = both_paths(path, gen)
    assert [r.ext_id for r in recs] == ["ok"]
    assert failures == [("bad", "MolSanitizeException: injected feature failure")]


def test_custom_smarts_generator_has_a_plan(tmp_path):
    """Every generator a corpus can be built with states its
    features as atom sets: a SMARTS-list generator takes the merged path and
    both paths give the same table."""
    gen = CustomSMARTSPharmacophoreGenerator(
        features={"Donor": ["[N;!H0]", "[O;H1]"], "Acceptor": ["[O]"]},
        interactions=[("Donor", "Donor", 1.0, 1.0), ("Acceptor", "Acceptor", 1.0, 1.0)])
    plan = gen.make_feature_plan(Chem.MolFromSmiles("CCO"))
    assert plan.types.tolist() == [1, 2] and plan.atom_ids == ((2,), (2,))
    path = tmp_path / "custom.sdf"
    path.write_text("".join(record_blocks(embedded("OCCN", "m", 4))))
    recs, _ = both_paths(path, gen)
    assert len(recs) == 1 and len(recs[0].conformers) == 4
    # OCCN: donors N and O, acceptor O, as slots of the canonical "NCCO".
    assert sorted(recs[0].features) == [(1, (0,)), (1, (3,)), (2, (3,))]


def test_generator_without_plan_is_refused(tmp_path):
    """A generator that cannot state its features as atoms cannot build a
    corpus: the merged path falls back, and the per-record path refuses."""
    class NoPlan(CustomSMARTSPharmacophoreGenerator):
        def make_feature_plan(self, mol):
            return None

    gen = NoPlan(features={"Donor": ["[N;!H0]"]},
                 interactions=[("Donor", "Donor", 1.0, 1.0)])
    path = tmp_path / "custom.sdf"
    path.write_text("".join(record_blocks(embedded("OCCN", "m", 4))))
    with pytest.raises(sources.BuildRefused, match="no feature plan"):
        run_source(path, gen, True)


def test_more_records_than_the_bound_stream_per_record(tmp_path, monkeypatch):
    """(d): past MAX_MERGED_CONFORMERS no merged Mol is built at all. The bound
    is 4,096, so it is lowered here rather than writing 4,099
    records."""
    bound = 8
    monkeypatch.setattr(sources, "MAX_MERGED_CONFORMERS", bound)
    n = bound + 3
    mol = embedded("CCO", "huge", 1)
    block = record_blocks(mol)[0]
    path = tmp_path / "huge.sdf"
    path.write_text(block * n + "".join(record_blocks(embedded("CCN", "small", 3))))
    merged_sizes = []
    orig = sources._merge_records

    def spy(mols, **kwargs):
        merged_sizes.append(len(mols))
        return orig(mols, **kwargs)

    monkeypatch.setattr(sources, "_merge_records", spy)
    recs, _ = both_paths(path, PharmacophoreGenerator())
    assert merged_sizes == [3]
    assert [len(r.conformers) for r in recs] == [n, 3]


def _counting_generator(monkeypatch):
    gen = PharmacophoreGenerator()
    calls = {"plan": 0, "per_conf": 0}
    orig_plan, orig_gen = gen.make_feature_plan, gen.generate_color_atoms

    def plan(m):
        calls["plan"] += 1
        return orig_plan(m)

    def per_conf(m):
        calls["per_conf"] += 1
        return orig_gen(m)

    monkeypatch.setattr(gen, "make_feature_plan", plan)
    monkeypatch.setattr(gen, "generate_color_atoms", per_conf)
    return gen, calls


def test_binary_zstd_and_sqlite_sources_already_take_plan_branch(tmp_path, monkeypatch):
    """Item 5: those sources hand _group_by_name one Mol per molecule with its
    conformers attached, so they already match features once per molecule;
    only SDF input (one record per conformer) took the per-conformer path."""
    import sqlite3
    import struct

    import zstandard as zstd

    from roshambo3.prep.sqlite_source import SqliteSource

    mols = [embedded("CCOc1ccccc1", "x", 5), embedded("OC(=O)c1ccccc1N", "y", 3)]

    gen, calls = _counting_generator(monkeypatch)
    blobs = b"".join(struct.pack("<I", len(b)) + b for b in
                     (m.ToBinary(Chem.PropertyPickleOptions.AllProps) for m in mols))
    path = tmp_path / "in.bin.zst"
    path.write_bytes(zstd.ZstdCompressor().compress(blobs))
    src = sources.RdkitBinaryZstdSource(str(path), color_generator=gen)
    recs = [r for u in src.units() for r in src.iter_unit(u)]
    assert [len(r.conformers) for r in recs] == [5, 3]
    assert calls == {"plan": 2, "per_conf": 0}

    gen, calls = _counting_generator(monkeypatch)
    db = tmp_path / "in.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE mols (molecule_id INTEGER PRIMARY KEY, molbinary_zst BLOB NOT NULL)")
    for mid, m in enumerate(mols, start=1):
        con.execute("INSERT INTO mols VALUES (?, ?)",
                    (mid, zstd.ZstdCompressor().compress(m.ToBinary())))
    con.commit()
    con.close()
    src = SqliteSource(str(db), kind="pubchem", color_generator=gen)
    recs = [r for u in src.units() for r in src.iter_unit(u)]
    assert sorted(len(r.conformers) for r in recs) == [3, 5]
    assert calls == {"plan": 2, "per_conf": 0}


# --------------------------------------------------------------------------
# real data

def _slice(src, dst, n_records):
    n = 0
    with open(src, "rb") as f, open(dst, "wb") as out:
        for line in f:
            out.write(line)
            if line.startswith(b"$$$$"):
                n += 1
                if n >= n_records:
                    break
    return dst


@pytest.mark.parametrize("path", REAL_SDF_SLICES or [None])
def test_real_slice_identical(tmp_path, path):
    n = 1200
    if path is None:
        pytest.skip("set ROSHAMBO3_SDF_SLICES to real SDF shards to run this")
    if not os.path.exists(path):
        pytest.skip(f"{path} absent")
    sl = _slice(path, tmp_path / "slice.sdf", n)
    recs, failures = both_paths(sl, PharmacophoreGenerator())
    assert sum(len(r.conformers) for r in recs) == n


def test_merge_bound_mirrors_the_layout_limit():
    from roshambo3.prep import build
    assert MAX_MERGED_CONFORMERS == build.MAX_CONFORMERS_PER_MOLECULE == 4096


# --------------------------------------------------------------------------
# variant refusal (FORMAT.md 11.1)

VARIANTS = {
    # name: (first record's SMILES, second record's SMILES)
    "charge": ("CC(=O)O", "CC(=O)[O-]"),
    "protonation": ("CCCN", "CCC[NH3+]"),
    "element": ("CCCO", "CCCN"),
    "atom-count": ("CCO", "CCCO"),
}


@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize("what", sorted(VARIANTS))
def test_variant_under_one_id_is_refused(tmp_path, what, merge):
    """Records of one id whose hydrogen-free graphs differ in more than stereo
    or atom order are refused, naming the id -- on the merged and the
    per-record SDF path alike."""
    first, second = VARIANTS[what]
    path = tmp_path / "variant.sdf"
    path.write_text("".join(record_blocks(embedded("c1ccccc1O", "before", 2))
                            + record_blocks(embedded(first, "VAR-1", 2))
                            + record_blocks(embedded(second, "VAR-1", 2, seed=5))))
    with pytest.raises(sources.BuildRefused, match=r"molecule 'VAR-1'.*own id"):
        run_source(path, PharmacophoreGenerator(), merge)
    # shape-only builds refuse too (no merge without a color generator)
    with pytest.raises(sources.BuildRefused, match=r"molecule 'VAR-1'"):
        run_source(path, None, merge)


@pytest.mark.parametrize("merge", [True, False])
def test_stereo_and_atom_order_variants_are_accepted(tmp_path, merge):
    stereo = (record_blocks(embedded("C[C@H](N)C(=O)O", "S", 2))
              + record_blocks(embedded("C[C@@H](N)C(=O)O", "S", 2, seed=3)))
    a = embedded("OCC(N)c1ccccc1", "A", 2)
    b = Chem.RenumberAtoms(a, list(reversed(range(a.GetNumAtoms()))))
    b.SetProp("_Name", "A")
    path = tmp_path / "ok.sdf"
    path.write_text("".join(stereo + record_blocks(a) + record_blocks(b)))
    recs, failures = run_source(path, PharmacophoreGenerator(), merge)
    by = {r.ext_id: r for r in recs}
    assert [len(by[k].conformers) for k in ("S", "A")] == [4, 4]
    assert by["S"].meta["smiles"] == "CC(N)C(=O)O"      # non-isomeric
    assert failures == []


def test_rdkit_source_refuses_a_variant_and_accepts_order_and_stereo():
    a = embedded("OCC(N)c1ccccc1", "A", 1)
    b = Chem.RenumberAtoms(a, list(reversed(range(a.GetNumAtoms()))))
    b.SetProp("_Name", "A")
    ok = [embedded("C[C@H](N)C(=O)O", "S", 1), embedded("C[C@@H](N)C(=O)O", "S", 1),
          a, b]
    src = sources.RDKitSource(ok)
    recs = [r for u in src.units() for r in src.iter_unit(u)]
    assert sorted((r.ext_id, len(r.conformers)) for r in recs) == [("A", 2), ("S", 2)]
    bad = sources.RDKitSource([embedded("CCCN", "P", 1), embedded("CCC[NH3+]", "P", 1)])
    with pytest.raises(sources.BuildRefused, match=r"molecule 'P'"):
        [r for u in bad.units() for r in bad.iter_unit(u)]

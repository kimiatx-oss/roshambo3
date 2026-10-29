"""The builder's color feature table (FORMAT.md section 8).

Each molecule's features are stored once, as ``(type, heavy-atom slots)``,
perceived on the stored SMILES with hydrogens added; every
conformer's pseudo-atoms are derived from its stored int16 heavy atoms
(integer mean, half away from zero)
and ``color_self`` is computed on those. The oracle for every derived number is
the pure-Python format-3 reference (``format/r3g/reference.py``), and the
oracle for every table is re-perception of the stored SMILES with hydrogens
added, using the stored definition text (``definitions.perceive``) -- the check
the ``deep`` validator makes.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3 import definitions
from roshambo3.format.r3g import reference as R
from roshambo3.pharmacophore import (CustomSMARTSPharmacophoreGenerator,
                                     PharmacophoreGenerator,
                                     RDKitPharmacophoreGenerator)
from roshambo3.prep import prepare
from roshambo3.prep import sources
from roshambo3.prep.build import BuildRefused, derive_feature_positions
from roshambo3.reference.overlap import color_overlap, self_overlap
from tests.format3_rows import read_corpus

SMILES = ["c1ccccc1OCCNC(=O)C", "OC(=O)c1ccc(N)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
          "c1ccc2[nH]ccc2c1CCN(C)C", "Clc1ccc(cc1)C(=O)NCC[NH3+]", "O=C([O-])CCc1ccncc1"]


def records(smiles=SMILES, n_conf=3, seed=0):
    """One SDF-like record per conformer, hydrogens explicit and every atom
    order shuffled, so the two remaps (H-bearing -> heavy -> slot) are
    exercised: hydrogens sit between heavy atoms and the heavy atoms are not
    in canonical order."""
    rng = np.random.default_rng(seed)
    out = []
    for i, smi in enumerate(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        assert AllChem.EmbedMultipleConfs(m, n_conf, randomSeed=i + 1) is not None
        m = Chem.RenumberAtoms(m, [int(x) for x in rng.permutation(m.GetNumAtoms())])
        for c in m.GetConformers():
            one = Chem.Mol(m)
            one.RemoveAllConformers()
            one.AddConformer(Chem.Conformer(c), assignId=True)
            one.SetProp("_Name", f"m{i}")
            out.append(one)
    return out


def build(tmp_path, name, generator, mols=None, **kw):
    """Build (format 3) and read the corpus back as rows."""
    out = tmp_path / name
    prepare(records() if mols is None else mols, str(out),
            color_generator=generator, workers=1, **kw)
    return read_corpus(out, merged=False)


def groups_of(corpus):
    """Per group: (group rows, feature table, decoded int16 heavy rows, the
    per-row ``mol_id`` and ``vaa``, derived counts, molecule runs)."""
    for g in corpus.groups:
        start = g.run_start
        yield (g, g.feature_table, g.xyz, {"mol_id": g.mol_id, "vaa": g.vaa}, g.counts,
               (start[:-1], start[1:]))


def check_corpus(corpus, *, canonical_order=True):
    """Every acceptance property a reader relies on, against the oracles."""
    color = corpus.descriptor["color"]
    with open(os.path.join(corpus.root, "corpus.json")) as f:
        assert "fdef_path" not in f.read()               # text, never a path
    smiles = dict(zip(np.asarray(corpus.molecules["mol_id"]).tolist(),
                      corpus.molecules["smiles"]))
    n_types = len(color["types"])
    r = np.asarray(color["r"]["values"])
    p = np.asarray(color["p"]["values"])
    n_mol = 0
    man = {"descriptor": corpus.descriptor, "build": corpus.build,
           "_color_self_rows_checked": 0}
    for g, table, xyz, sc, counts, (starts, ends) in groups_of(corpus):
        assert table["feature_offsets"].size == starts.size + 1
        perceived = definitions.perceive(
            color, [smiles[int(sc["mol_id"][s])] for s in starts], check=False)
        color_self = g.color_self
        for m, (s0, e0) in enumerate(zip(starts, ends)):
            feats = R.molecule_features(table, m)
            # Re-perception from the stored SMILES, hydrogens added, with the
            # stored definition: no feature on a hydrogen, the same table.
            got = perceived[m]
            assert got.parsed and all(max(a) < got.n_heavy for _, a in got.features)
            if canonical_order:
                assert feats == R.canonical_feature_order(got.features)
            else:
                assert sorted(R.canonical_feature_order(feats)) == \
                    R.canonical_feature_order(got.features)
            n_real = int(counts[s0, 0])
            for row in range(s0, e0):
                # The reader's derived count vector is the reference's (the
                # merged record is derived from the same table: FORMAT.md 8.4).
                assert np.array_equal(R.derived_counts(n_real, feats, n_types),
                                      counts[row])
                heavy = xyz[row][:, :n_real].T.astype(np.float64) * 2.0 ** -8
                v = self_overlap(heavy)
                assert abs(float(sc["vaa"][row]) - v) <= 1e-5 * v
                if color_self is not None:
                    pos = R.derive_positions(xyz[row], feats).astype(np.float64) * 2.0 ** -8
                    t = np.asarray([f[0] for f in feats], dtype=np.int64)
                    want = color_overlap(pos, t, pos, t, r, p)
                    assert abs(color_self[row] - want) <= 1e-5 * max(want, 1e-30)
                    man["_color_self_rows_checked"] += 1
            n_mol += 1
    assert n_mol == corpus.n_mol
    return man


def test_default_fdef(tmp_path):
    gen = PharmacophoreGenerator()
    corpus = build(tmp_path, "default", gen)
    man = check_corpus(corpus)
    assert man["descriptor"]["color"] == gen.descriptor_color()
    assert man["build"]["color"]["definition_sha256"] == R.definition_hash(
        gen.descriptor_color()["definition"])
    assert definitions.is_shipped(man["build"]["color"]["definition_sha256"])
    assert man["build"]["parameters"]["feature_order"] == "canonical"
    assert man["build"]["color"]["color_self"]["status"] == "built"
    assert man["_color_self_rows_checked"] == corpus.n_conf


def test_default_fdef_merged_and_binary_paths(tmp_path):
    """The multi-conformer branch (one Mol, conformers attached) builds the
    same table as the per-record branch."""
    per_record = build(tmp_path, "per-record", PharmacophoreGenerator())
    merged = []
    for i, m in enumerate(records()):
        if i % 3 == 0:
            merged.append(Chem.Mol(m))
        else:
            merged[-1].AddConformer(Chem.Conformer(m.GetConformer()), assignId=True)
    one_mol = build(tmp_path, "merged", PharmacophoreGenerator(), mols=merged)
    check_corpus(one_mol)
    for (_, ta, xa, *_), (_, tb, xb, *_) in zip(groups_of(per_record), groups_of(one_mol)):
        for name in ta:
            assert np.array_equal(ta[name], tb[name]), name
        assert np.array_equal(xa, xb)


CUSTOM_FDEF = """\
# Families declared in one order, selected in another.
AtomType Hal [F,Cl,Br,I]
DefineFeature Halogen [{Hal}]
  Family Halogen
  Weights 1.0
EndFeature
DefineFeature Carbonyl [#6]=[#8]
  Family Carbonyl
  Weights 1.0,1.0
EndFeature
DefineFeature Ring6 a1aaaaa1
  Family Ring
  Weights 1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
DefineFeature Amine [NX3,NX4+;!$(N-C=O)]
  Family Base
  Weights 1.0
EndFeature
DefineFeature Lever [#6]-[#6]
  Family Unselected
  Weights 3.0,1.0
EndFeature
"""


def custom_fdef_generator(tmp_path):
    path = tmp_path / "custom.fdef"
    path.write_text(CUSTOM_FDEF)
    families = ["Ring", "Base", "Carbonyl", "Halogen"]
    return RDKitPharmacophoreGenerator(
        features={f: "rdkit" for f in families},
        interactions=[(f, f, 1.0, 1.0) for f in families],
        fdefName=str(path))


def test_custom_fdef_whose_families_and_order_differ(tmp_path):
    gen = custom_fdef_generator(tmp_path)
    corpus = build(tmp_path, "custom", gen)
    man = check_corpus(corpus)
    color = man["descriptor"]["color"]
    assert color["generator"] == "rdkit_fdef"
    assert color["definition"] == CUSTOM_FDEF
    assert color["types"] == ["Ring", "Base", "Carbonyl", "Halogen"]
    assert not definitions.is_shipped(man["build"]["color"]["definition_sha256"])
    # Every family is present somewhere, numbered by the selection order.
    seen = set()
    for _, table, *_ in groups_of(corpus):
        seen.update(table["feature_types"].tolist())
    assert seen == {1, 2, 3, 4}


def test_custom_smarts_generator(tmp_path):
    families = {"Acceptor": ["[O;!H1]", "[n]"], "Donor": ["[N;!H0]", "[O;H1]", "[nH]"],
                "Ring": ["a1aaaaa1", "a1aaaa1"]}
    gen = CustomSMARTSPharmacophoreGenerator(
        features=families, types=["Ring", "Donor", "Acceptor"],
        interactions=[(f, f, 1.0, 1.0) for f in families])
    corpus = build(tmp_path, "smarts", gen)
    man = check_corpus(corpus)
    color = man["descriptor"]["color"]
    assert color["generator"] == "smarts_list" and color["definition"] == families
    assert color["types"] == ["Ring", "Donor", "Acceptor"]


def test_a_feature_on_a_hydrogen_is_refused(tmp_path):
    gen = CustomSMARTSPharmacophoreGenerator(
        features={"Polar": ["[#1]-[#7,#8]"], "Ring": ["a1aaaaa1"]},
        interactions=[("Polar", "Polar", 1.0, 1.0), ("Ring", "Ring", 1.0, 1.0)])
    with pytest.raises(BuildRefused, match=r"molecule 'm0': its 'Polar' feature on "
                                           r"atoms \[.*\] references a hydrogen"):
        build(tmp_path, "h", gen)
    assert not (tmp_path / "h").exists()


def test_a_hydrogen_fdef_feature_is_refused(tmp_path):
    path = tmp_path / "h.fdef"
    path.write_text("DefineFeature HX [#1]-[#6]\n  Family HX\n  Weights 1.0,1.0\nEndFeature\n")
    gen = RDKitPharmacophoreGenerator(features={"HX": "rdkit"},
                                      interactions=[("HX", "HX", 1.0, 1.0)],
                                      fdefName=str(path))
    with pytest.raises(BuildRefused, match=r"'HX' feature .* references a hydrogen"):
        build(tmp_path, "hx", gen)


def test_unequal_weights_are_refused(tmp_path):
    path = tmp_path / "w.fdef"
    path.write_text(CUSTOM_FDEF.replace("Weights 1.0,1.0\n", "Weights 2.0,1.0\n"))
    families = ["Ring", "Carbonyl"]
    gen = RDKitPharmacophoreGenerator(features={f: "rdkit" for f in families},
                                      interactions=[(f, f, 1.0, 1.0) for f in families],
                                      fdefName=str(path))
    with pytest.raises(BuildRefused, match=r"family 'Carbonyl' \(feature 'Carbonyl'\) "
                                           r"declares Weights 2.0,1.0"):
        build(tmp_path, "w", gen)
    assert not (tmp_path / "w").exists()


def test_weights_are_checked_only_for_selected_families():
    # CUSTOM_FDEF's 'Unselected' family declares 3,1, and BaseFeatures'
    # ZnBinder family 1,0: neither is selected, so neither is refused.
    color = PharmacophoreGenerator().descriptor_color()
    assert definitions.unequal_weights(color) == []
    assert definitions.unequal_weights({**color, "types": ["ZnBinder"]})
    custom = {"generator": "rdkit_fdef", "definition": CUSTOM_FDEF,
              "types": ["Ring", "Halogen"]}
    assert definitions.unequal_weights(custom) == []
    assert definitions.unequal_weights({**custom, "types": ["Unselected"]}) == [
        ("Unselected", "Lever", "3.0,1.0")]


def test_the_within_type_sort_and_its_measurement_switch(tmp_path):
    """FORMAT.md 8.1 rule 4 sorts each molecule's features by (type, atoms);
    the shard writer's ``feature_order="enumeration"`` keeps the generator's
    order within a type. The two differ only by a permutation within each
    type. Such a table is not valid format 3, so the format-3 builder refuses
    to emit it, before any work and leaving nothing."""
    import types as _types
    from roshambo3.prep.build import _ShardSet
    rec = _types.SimpleNamespace(
        mol_id=7, ext_id="m", features=[(1, (3,)), (1, (1, 2)), (1, (1,)), (2, (0,))])
    rows = {}
    for order in ("canonical", "enumeration"):
        s = _ShardSet(str(tmp_path / f"shards-{order}"), 0, n_types=2, feature_order=order)
        try:
            row, nat, atoms = s._feature_rows(rec, 4)
        finally:
            s.tsv.close()
        rows[order] = (row["counts"][0].tolist(), nat.tolist(), atoms.tolist())
    assert rows["canonical"] == ([3, 1], [1, 2, 1, 1], [1, 1, 2, 3, 0])
    assert rows["enumeration"] == ([3, 1], [1, 2, 1, 1], [3, 1, 2, 1, 0])
    with pytest.raises(BuildRefused, match="not a valid format-3 feature table"):
        build(tmp_path, "enumeration", PharmacophoreGenerator(), _feature_order="enumeration")
    assert not os.path.exists(tmp_path / "enumeration")
    assert not os.path.exists(tmp_path / "enumeration.partial")


def test_derive_feature_positions_matches_the_reference():
    """The vectorized derivation is the reference's integer mean, half away
    from zero, at the int16 extremes and on exact halves."""
    rng = np.random.default_rng(7)
    n_rows, n_real = 40, 20
    xyz = rng.integers(-32768, 32768, size=(n_rows, 3, n_real)).astype(np.int16)
    xyz[0, :, :4] = [[1, 2, -1, -2], [3, 0, 0, 0], [-32768, -32768, 32767, 32767]]
    features = [(1, (0, 1)), (1, (2, 3)), (2, (0, 1, 2, 3)), (3, (5,))]
    features += [(4, tuple(sorted(rng.choice(n_real, size=k, replace=False).tolist())))
                 for k in rng.integers(1, 17, size=30)]
    n_atoms = np.asarray([len(a) for _, a in features], dtype=np.uint8)
    atoms = np.asarray([x for _, a in features for x in a], dtype=np.uint8)
    atom_start = np.concatenate([[0], np.cumsum(n_atoms)[:-1]])
    got = derive_feature_positions(
        xyz, np.zeros(n_rows, np.int64), np.full(n_rows, len(features)),
        n_atoms, atom_start, atoms, batch_rows=7)
    want = np.concatenate([R.derive_positions(xyz[r], features) for r in range(n_rows)])
    assert np.array_equal(got, want)
    # (1+2)/2 = 1.5 -> 2; (-1-2)/2 = -1.5 -> -2; 3/4 -> 1; -2/4 = -0.5 -> -1.
    assert got[:3].tolist() == [[2, 2, -32768], [-2, 0, 32767], [0, 1, -1]]
    assert got[3].tolist() == xyz[0, :, 5].tolist()          # one atom: copied


def test_the_table_is_the_stored_smiles_reperceived():
    """Perceived on the parsed stored SMILES, whose atom ``i`` is
    slot ``i``: the table equals ``deep``'s re-perception of the stored
    SMILES with hydrogens added, for inputs whose hydrogens sit between heavy
    atoms and whose heavy atoms are not in canonical order."""
    gen = PharmacophoreGenerator()
    color = gen.descriptor_color()
    reordered = 0
    for mol in records(n_conf=1, seed=3):
        canonical = sources.canonical_heavy(mol)
        reordered += not np.array_equal(canonical[2], np.arange(canonical[2].size))
        assert mol.GetAtomWithIdx(0).GetAtomicNum() == 1 or any(
            a.GetAtomicNum() == 1 for a in list(mol.GetAtoms())[:canonical[2].size])
        parsed, why = sources.parse_stored_smiles(canonical[1], canonical[0])
        assert why is None
        table = sources.feature_table(gen, parsed, mol.GetProp("_Name"))
        again = definitions.perceive(color, [canonical[1]], check=False)[0]
        assert R.canonical_feature_order(table) == R.canonical_feature_order(again.features)
        rec, = sources._group_by_name([mol], 0, gen)
        assert rec.features == table
    assert reordered == len(SMILES)


def test_a_generator_that_cannot_describe_itself_is_refused(tmp_path):
    class Opaque(PharmacophoreGenerator):
        descriptor_color = None

    with pytest.raises(BuildRefused, match="cannot describe its definition"):
        build(tmp_path, "opaque", Opaque())


def test_the_default_generator_is_canonical_and_keeps_color_self(tmp_path):
    """One definition of canonical matrices: exact
    diagonal r = p = 1 and zero off-diagonal p on types 1..n; off-diagonal r
    and row/column 0 ignored. The default generator's r starts at all ones,
    which that definition accepts, so both tests agree and its corpora keep
    color_self."""
    from roshambo3.numerics import check_interaction_matrix
    gen = PharmacophoreGenerator()
    color = gen.descriptor_color()
    assert R.canonical_matrices(color["r"]["values"], color["p"]["values"], len(color["types"]))
    assert check_interaction_matrix(gen.interaction_matrix_r, gen.interaction_matrix_p).canonical
    corpus = build(tmp_path, "default", gen, mols=records(SMILES[:2]))
    record = corpus.build["color"]["color_self"]
    assert record["status"] == "built" and record["matrices_canonical"] is True
    assert all(g.color_self is not None for g in corpus.groups)


def _matrices(n=3):
    """The default generator's form: r all ones, p the identity on 1..n."""
    r = np.ones((n + 1, n + 1), np.float32)
    p = np.eye(n + 1, dtype=np.float32)
    p[0, 0] = 0
    return r, p


@pytest.mark.parametrize("edit, canonical", [
    (lambda r, p: None, True),
    (lambda r, p: r.__setitem__((1, 2), 0.5), True),          # off-diagonal r ignored
    (lambda r, p: r.__setitem__((0, 0), 7.0), True),          # row/column 0 ignored
    (lambda r, p: p.__setitem__((0, 3), 2.0), True),
    (lambda r, p: p.__setitem__((1, 2), 0.5), False),         # off-diagonal height
    (lambda r, p: r.__setitem__((2, 2), 1.0000001), False),   # exact, not allclose
    (lambda r, p: p.__setitem__((3, 3), 0.9999999), False),
])
def test_one_canonical_rule_everywhere(edit, canonical):
    from roshambo3.format.r3g import validate
    from roshambo3.layout import color_loop_is_same_type_safe
    from roshambo3.numerics import check_interaction_matrix
    r, p = _matrices()
    edit(r, p)
    assert R.canonical_matrices(r, p, 3) is canonical
    assert check_interaction_matrix(r, p).canonical is canonical
    assert color_loop_is_same_type_safe(r, p) is canonical
    color = R.color_descriptor(generator="smarts_list", definition={"A": ["C"], "B": ["N"], "C": ["O"]},
                               types=["A", "B", "C"], r=r, p=p, rdkit_version="x")
    assert validate.canonical_matrices(color) is canonical


@pytest.mark.parametrize("value, ok", [(0.0, True), (1e-6, True), (2.0, True),
                                       (1e-7, False), (1e-30, False), (float("nan"), False),
                                       (float("inf"), False), (-1.0, False)])
def test_interaction_width_validity(value, ok):
    """Every width is exactly 0 or >= 1e-6 (a tiny positive width is NaN on
    device), finite; every height finite and >= 0 -- for every matrix."""
    from roshambo3.layout import check_interaction_matrices
    r, p = _matrices()
    r[1, 2] = value
    assert (not R.interaction_matrix_problems(r, p)) is ok
    if ok:
        check_interaction_matrices(r, p, np.array([1, 2]))
    else:
        with pytest.raises(ValueError):
            check_interaction_matrices(r, p, np.array([1, 2]))
    r, p = _matrices()
    p[2, 3] = -0.5 if ok else 0.0
    assert bool(R.interaction_matrix_problems(r, p)) is ok


def _r3g_color(off_p=0.0, r_cell=None):
    r, p = _matrices(2)
    p[1, 2] = off_p
    if r_cell is not None:
        r[1, 2] = r_cell
    return R.color_descriptor(generator="smarts_list", definition={"A": ["[#6]"], "B": ["[#8]"]},
                              types=["A", "B"], r=r, p=p, rdkit_version="2026.09.1")


def _r3g_molecule():
    xyz = np.array([[[0, 256, 512], [0, 0, 0], [0, 0, 0]]], dtype=np.int16)
    return R.Molecule(mol_id=1, coords=xyz, vaa=np.ones(1, np.float32),
                      features=[(1, (0,)), (1, (1,)), (2, (2,))], color_self=np.ones(1),
                      smiles="CCO", ext_id="m1", mw=46.0)


def test_non_canonical_matrices_carry_no_color_self(tmp_path):
    from roshambo3.format.r3g import writer
    from roshambo3.format.r3g.errors import CorpusValidationError
    from roshambo3.format.r3g.validate import validate_corpus
    noncanonical = _r3g_color(off_p=0.5)
    d = writer.write_corpus(str(tmp_path / "a"), [_r3g_molecule()], color=noncanonical,
                            prep_version="0", prep_commit=None)
    validate_corpus(str(tmp_path / "a"), "scan")
    with pytest.raises(ValueError, match="canonical"):
        writer.write_corpus(str(tmp_path / "b"), [_r3g_molecule()], color=noncanonical,
                            prep_version="0", prep_commit=None, with_color_self=True)
    # A corpus that carries color_self anyway is refused by the validator.
    R.write_corpus(str(tmp_path / "c"), [_r3g_molecule()], color=noncanonical,
                   prep_version="0", prep_commit=None, with_color_self=True)
    with pytest.raises(CorpusValidationError) as e:
        validate_corpus(str(tmp_path / "c"), "open")
    assert e.value.code == "layout.sections"
    # Off-diagonal widths do not matter: still canonical, color_self kept.
    writer.write_corpus(str(tmp_path / "d"), [_r3g_molecule()], color=_r3g_color(r_cell=0.5),
                        prep_version="0", prep_commit=None, with_color_self=True)
    validate_corpus(str(tmp_path / "d"), "scan")


@pytest.mark.parametrize("r_cell, ok", [(1e-7, False), (0.0, True)])
def test_a_tiny_width_is_refused_by_writer_and_validator(tmp_path, r_cell, ok):
    from roshambo3.format.r3g import writer
    from roshambo3.format.r3g.errors import CorpusValidationError
    from roshambo3.format.r3g.validate import validate_corpus
    color = _r3g_color(r_cell=r_cell)
    if ok:
        writer.write_corpus(str(tmp_path / "w"), [_r3g_molecule()], color=color,
                            prep_version="0", prep_commit=None)
        validate_corpus(str(tmp_path / "w"), "scan")
        return
    with pytest.raises((ValueError, CorpusValidationError)):
        writer.write_corpus(str(tmp_path / "w"), [_r3g_molecule()], color=color,
                            prep_version="0", prep_commit=None)
    R.write_corpus(str(tmp_path / "r"), [_r3g_molecule()], color=color,
                   prep_version="0", prep_commit=None)
    with pytest.raises(CorpusValidationError) as e:
        validate_corpus(str(tmp_path / "r"), "open")
    assert e.value.code == "value.matrix"


def test_a_build_with_an_invalid_width_is_refused(tmp_path):
    families = ["Donor", "Acceptor"]
    gen = CustomSMARTSPharmacophoreGenerator(
        features={"Donor": ["[N;!H0]"], "Acceptor": ["[O]"]},
        interactions=[("Donor", "Donor", 1.0, 1.0), ("Acceptor", "Acceptor", 1.0, 1.0),
                      ("Donor", "Acceptor", 1e-7, 0.0)])
    with pytest.raises(BuildRefused, match=r"widths must be exactly 0 or >= 1e-06"):
        build(tmp_path, "tiny", gen)


NO_WEIGHTS_FDEF = """DefineFeature Ar6 c1ccccc1
  Family Aromatic
EndFeature
DefineFeature Don [N;!H0]
  Family Donor
EndFeature
DefineFeature Acc [O]
  Family Acceptor
  Weights 1.0
EndFeature
"""


def test_a_selected_family_without_weights_is_refused(tmp_path):
    """Every block of a selected family must declare its
    (all-ones) weights; RDKit cannot place a feature without them at query
    time. Unselected families may omit them."""
    path = tmp_path / "noweights.fdef"
    path.write_text(NO_WEIGHTS_FDEF)

    def gen(families):
        return RDKitPharmacophoreGenerator(features={f: "rdkit" for f in families},
                                           interactions=[(f, f, 1.0, 1.0) for f in families],
                                           fdefName=str(path))
    with pytest.raises(BuildRefused, match=r"family 'Aromatic' \(feature 'Ar6'\) declares "
                                           r"no Weights line; family 'Donor' \(feature 'Don'\)"):
        build(tmp_path, "noweights", gen(["Aromatic", "Donor"]))
    check_corpus(build(tmp_path, "selected-ok", gen(["Acceptor"])))
    color = {"generator": "rdkit_fdef", "definition": NO_WEIGHTS_FDEF, "types": ["Donor"]}
    assert definitions.unequal_weights(color) == [("Donor", "Don", None)]


def test_an_unparseable_smarts_pattern_is_refused_by_builder_and_validator(tmp_path):
    from roshambo3.format.r3g.errors import CorpusValidationError
    from roshambo3.format.r3g.validate import validate_corpus
    families = {"A": ["[#6]", "C1CC"], "B": ["[#8]"]}
    gen = CustomSMARTSPharmacophoreGenerator(
        features=families, interactions=[(f, f, 1.0, 1.0) for f in families])
    with pytest.raises(BuildRefused, match=r"family 'A''s SMARTS pattern 'C1CC' does not parse"):
        build(tmp_path, "bad-smarts", gen)
    # The same definition in a written corpus: `deep` refuses it rather than
    # skipping the pattern.
    r, p = _matrices(2)
    color = R.color_descriptor(generator="smarts_list", definition=families, types=["A", "B"],
                               r=r, p=p, rdkit_version="2026.09.1")
    R.write_corpus(str(tmp_path / "c"), [_r3g_molecule()], color=color,
                   prep_version="0", prep_commit=None)
    validate_corpus(str(tmp_path / "c"), "scan")
    with pytest.raises(CorpusValidationError) as e:
        validate_corpus(str(tmp_path / "c"), "deep", all_rows=True)
    assert e.value.code == "value.definition" and "does not parse" in str(e.value)


def _hydrogen_variants(tmp_path, kind):
    """SDF input with no explicit hydrogens (``none``) or with only the
    heteroatoms' (``partial``): one record per conformer."""
    path = tmp_path / f"{kind}.sdf"
    with Chem.SDWriter(str(path)) as w:
        for i, smi in enumerate(["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCCCCCN", "c1ccc2[nH]ccc2c1CCN(C)C"]):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            assert len(AllChem.EmbedMultipleConfs(m, 3, randomSeed=i + 5)) == 3
            if kind == "none":
                m = Chem.RemoveHs(m)
            else:
                rw = Chem.RWMol(m)
                for a in sorted((a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 1
                                 and a.GetNeighbors()[0].GetAtomicNum() == 6), reverse=True):
                    rw.RemoveAtom(a)
                m = rw.GetMol()
                Chem.SanitizeMol(m)
            m.SetProp("_Name", f"h{i}")
            for c in m.GetConformers():
                w.write(m, confId=c.GetId())
    return path


@pytest.mark.parametrize("kind", ["none", "partial"])
@pytest.mark.parametrize("merge", [True, False])
def test_input_without_explicit_hydrogens_is_perceived_with_them(tmp_path, kind, merge,
                                                                 monkeypatch):
    """Perception sees every hydrogen even when the input omits
    some. Without ``AddHs`` the Hydrophobe set is wrong (H-free ibuprofen:
    10 features stored against 17 re-perceived), and ``deep`` would refuse
    the corpus."""
    path = _hydrogen_variants(tmp_path, kind)
    if not merge:
        monkeypatch.setattr(sources.SdfSource, "__init__", _no_merge(sources.SdfSource.__init__))
    prepare([str(path)], str(tmp_path / f"{kind}-{merge}"),
            color_generator=PharmacophoreGenerator(), workers=1)
    corpus = read_corpus(tmp_path / f"{kind}-{merge}", merged=False)
    check_corpus(corpus)
    ibuprofen = sources.stable_id("h0")
    for _, table, _, sc, _, (starts, _) in groups_of(corpus):
        for m, s0 in enumerate(starts):
            if int(sc["mol_id"][s0]) == ibuprofen:
                assert len(R.molecule_features(table, m)) == 17


def _no_merge(init):
    def wrapped(self, *args, **kwargs):
        kwargs["merge_conformers"] = False
        return init(self, *args, **kwargs)
    return wrapped

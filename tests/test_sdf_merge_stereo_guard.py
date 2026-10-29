"""The SDF conformer merge respects stereochemistry conservatively.

``SdfSource`` merges a molecule's per-conformer records so the color feature
SMARTS are matched once per molecule. Records that differ in stereochemistry
may only be merged when the GENERATOR positively certifies that its patterns
cannot depend on stereo. Otherwise the merge is strict: known-different stereo
does not merge, assigned stereo does not merge with unspecified, and stereo
RDKit cannot report never merges. Every test that runs a source compares the
strict and the certified path field by field; they must agree bit for bit.
"""
import io

import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures

from roshambo3.pharmacophore import (BasePharmacophoreGenerator,
                                     CustomSMARTSPharmacophoreGenerator,
                                     PharmacophoreGenerator,
                                     RDKitPharmacophoreGenerator)
from roshambo3.prep import sources
from roshambo3.prep.sources import SdfSource

from test_sdf_merge_conformers import (assert_records_identical, embedded,
                                       record_blocks, run_source)

RDLogger.DisableLog("rdApp.*")


def parsed(mol):
    return [next(Chem.ForwardSDMolSupplier(io.BytesIO(b.encode()), removeHs=False))
            for b in record_blocks(mol)]


def flipped(mol, idx):
    out = Chem.Mol(mol)
    tag = out.GetAtomWithIdx(idx).GetChiralTag()
    out.GetAtomWithIdx(idx).SetChiralTag(
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW
        else Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return out


def unspecified(mol):
    out = Chem.Mol(mol)
    Chem.RemoveStereochemistry(out)
    return out


class UncertifiedGenerator(RDKitPharmacophoreGenerator):
    """The real generator and the real plan path, but it does not vouch."""
    def feature_patterns_stereo_free(self):
        return False


# --------------------------------------------------------------------------
# the certificate

def test_default_generator_certifies_its_patterns_stereo_free():
    gen = PharmacophoreGenerator()
    assert gen.feature_patterns_stereo_free() is True
    # Cached: the factory's definitions are not rescanned.
    assert gen._stereo_free_cache is True


def test_base_and_custom_smarts_generators_do_not_certify():
    assert BasePharmacophoreGenerator().feature_patterns_stereo_free() is False
    assert CustomSMARTSPharmacophoreGenerator.feature_patterns_stereo_free is \
        BasePharmacophoreGenerator.feature_patterns_stereo_free


@pytest.mark.parametrize("what,smarts,certified", [
    ("clean", "[#6]~[#7]", True),
    ("atom chirality", "[C@H](F)(Cl)Br", False),
    ("atom chirality, double", "[C@@H](F)(Cl)Br", False),
    ("bond direction /", "F/C=C/F", False),
    ("bond direction backslash", "F/C=C\\F", False),
    # Stereo inside a recursive pattern is invisible to a parsed-query check:
    ("chirality inside $()", "[$([C@H](F)(Cl)Br)]", False),
    # Documented false positive: '@' is also SMARTS' ring-bond primitive.
    # It costs speed, never correctness.
    ("ring bond", "[#6]@[#6]", False),
])
def test_text_scan_of_the_expanded_patterns(what, smarts, certified):
    class Factory:
        def GetFeatureDefs(self):
            return {"Donor.Probe": "[#7;H1]", "Probe.X": smarts}

    gen = RDKitPharmacophoreGenerator()
    gen.factory = Factory()
    assert gen.feature_patterns_stereo_free() is certified, what


def test_uncertifiable_cases_all_answer_false():
    gen = RDKitPharmacophoreGenerator()
    class Empty:
        def GetFeatureDefs(self):
            return {}
    gen.factory = Empty()
    assert gen.feature_patterns_stereo_free() is False, "no definitions is not 'none with stereo'"
    gen = RDKitPharmacophoreGenerator()
    class Broken:
        def GetFeatureDefs(self):
            raise RuntimeError("factory failed")
    gen.factory = Broken()
    assert gen.feature_patterns_stereo_free() is False
    # a caller-supplied fdef is never the default plan
    assert RDKitPharmacophoreGenerator(fdefName="/nonexistent.fdef").feature_patterns_stereo_free() is False


def test_the_source_helper_treats_anything_short_of_true_as_unknown():
    class NoMethod:
        pass

    class Raises:
        def feature_patterns_stereo_free(self):
            raise RuntimeError

    class Truthy:
        def feature_patterns_stereo_free(self):
            return 1

    for g in (None, NoMethod(), Raises(), Truthy(), BasePharmacophoreGenerator()):
        assert sources._generator_is_stereo_free(g) is False
    assert sources._generator_is_stereo_free(PharmacophoreGenerator()) is True


def test_a_stereo_bearing_custom_fdef_does_not_certify(tmp_path):
    fdef = tmp_path / "stereo.fdef"
    fdef.write_text("DefineFeature Probe [C@H](F)(Cl)Br\n  Family Donor\n"
                    "  Weights 1.0,1.0,1.0,1.0\nEndFeature\n")
    assert RDKitPharmacophoreGenerator(fdefName=str(fdef)).feature_patterns_stereo_free() is False


def test_rdkit_still_matches_features_without_chirality():
    """Documents an assumption this project does not control (see the docstring
    of feature_patterns_stereo_free): RDKit's feature factory ignores chirality
    and bond direction in a pattern. It is NOT what the merge relies on (the
    patterns are scanned), but if it ever changes, a stereo-bearing custom fdef
    would genuinely start to depend on stereo, and this test says so."""
    fdef = ("DefineFeature Probe [C@H](F)(Cl)Br\n  Family Probe\n"
            "  Weights 1.0,1.0,1.0,1.0\nEndFeature\n")
    factory = ChemicalFeatures.BuildFeatureFactoryFromString(fdef)
    counts = []
    for smi in ("F[C@H](Cl)Br", "F[C@@H](Cl)Br"):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=7)
        counts.append(len(factory.GetFeaturesForMol(m)))
    assert counts[0] == counts[1], "RDKit now honors chirality in feature patterns"


# --------------------------------------------------------------------------
# the merge guard

@pytest.fixture(scope="module")
def alanine():
    mol = embedded("C[C@H](N)C(=O)O", "ala", 3)
    return parsed(mol)


def center(mol):
    return next(a.GetIdx() for a in mol.GetAtoms()
                if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)


def test_identical_stereo_merges_in_both_modes(alanine):
    for flag in (True, False):
        merged, same = sources._merge_records(alanine, stereo_independent=flag)
        assert merged is not None and merged.GetNumConformers() == 3 and same is True


def test_known_different_stereo_merges_only_when_certified(alanine):
    pair = [alanine[0], flipped(alanine[1], center(alanine[0]))]
    merged, same = sources._merge_records(pair, stereo_independent=True)
    assert merged is not None and same is False
    assert sources._merge_records(pair, stereo_independent=False) == (None, None)


def test_unspecified_never_merges_with_assigned_when_strict(alanine):
    pair = [alanine[0], unspecified(alanine[1])]
    merged, same = sources._merge_records(pair, stereo_independent=True)
    assert merged is not None and same is False
    assert sources._merge_records(pair, stereo_independent=False) == (None, None)
    # either order
    assert sources._merge_records(pair[::-1], stereo_independent=False) == (None, None)


def test_stereo_that_cannot_be_reported_never_merges_when_strict(alanine, monkeypatch):
    monkeypatch.setattr(sources, "_stereo_signature", lambda mol: None)
    assert sources._merge_records(alanine, stereo_independent=False) == (None, None)
    # certified mode never asks, so it is unaffected
    merged, _ = sources._merge_records(alanine, stereo_independent=True)
    assert merged is not None


def test_bond_stereo_differences_are_seen_when_strict():
    mol = embedded("C/C=C/C(=O)O", "but", 2)
    a, b = parsed(mol)
    other = Chem.Mol(b)
    for bond in other.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREOZ if bond.GetStereo() == Chem.BondStereo.STEREOE
                           else Chem.BondStereo.STEREOE)
    assert sources._merge_records([a, other], stereo_independent=False) == (None, None)
    assert sources._merge_records([a, b], stereo_independent=False)[0] is not None


def test_stereo_is_a_required_argument():
    with pytest.raises(TypeError):
        sources._merge_records([])


# --------------------------------------------------------------------------
# through the source

@pytest.fixture(scope="module")
def stereo_sdf(tmp_path_factory):
    """A molecule whose records disagree on a chiral tag the way real conformer
    sets do (an unspecified center embedded in both handednesses; SDF parsing
    assigns the tag from the 3D coordinates), one whose records agree, and one
    stored as a single record."""
    ala = embedded("CC(N)C(=O)O", "ala", 8)
    tags = {m.GetAtomWithIdx(center_of(m)).GetChiralTag() for m in parsed(ala)}
    assert len(tags) > 1, "fixture must contain a genuine stereo disagreement"
    blocks = record_blocks(ala)
    blocks += record_blocks(embedded("c1ccccc1O", "phenol", 3))
    blocks += record_blocks(embedded("CC(C)O", "iprop", 1))
    path = tmp_path_factory.mktemp("stereo") / "mixed.sdf"
    path.write_text("".join(blocks))
    return path


def center_of(mol):
    """The one stereocenter-capable carbon of alanine (bonded to N and C(=O))."""
    patt = Chem.MolFromSmarts("[CX4]([NX3])([CX3]=O)")
    return mol.GetSubstructMatch(patt)[0]


def test_the_source_asks_the_generator_once_not_per_molecule(stereo_sdf):
    calls = []

    class Counting(RDKitPharmacophoreGenerator):
        def feature_patterns_stereo_free(self):
            calls.append(1)
            return super().feature_patterns_stereo_free()

    src = SdfSource([str(stereo_sdf)], color_generator=Counting())
    for unit in src.units(target_records=2):
        list(src.iter_unit(unit))
    assert len(calls) == 1


def test_strict_and_certified_paths_produce_identical_records(stereo_sdf):
    certified, c_fail = run_source(stereo_sdf, PharmacophoreGenerator(), True)
    strict, s_fail = run_source(stereo_sdf, UncertifiedGenerator(), True)
    per_record, p_fail = run_source(stereo_sdf, PharmacophoreGenerator(), False)
    assert_records_identical(certified, strict)
    assert_records_identical(certified, per_record)
    assert c_fail == s_fail == p_fail

def test_strict_mode_leaves_the_stereo_varying_molecule_on_the_per_record_path(stereo_sdf, monkeypatch):
    merged = []
    orig = sources.canonical_heavy

    def spy(mol):
        # Merged molecules only: the per-record path canonicalizes each
        # one-conformer record. (Features are perceived once per molecule on
        # either path, on the stored SMILES, so perception no
        # longer tells the paths apart.)
        if mol.GetNumConformers() > 1:
            merged.append(mol.GetProp("_Name"))
        return orig(mol)

    monkeypatch.setattr(sources, "canonical_heavy", spy)
    run_source(stereo_sdf, PharmacophoreGenerator(), True)
    # certified: both multi-record molecules are merged and canonicalized once
    assert sorted(merged) == ["ala", "phenol"]
    merged.clear()
    run_source(stereo_sdf, UncertifiedGenerator(), True)
    # strict: 'ala' (records disagree on stereo) is NOT merged, so it is not
    # canonicalized as a merged molecule; 'phenol' (identical stereo) still is.
    assert merged == ["phenol"]


def test_prepare_builds_the_same_corpus_either_way(stereo_sdf, tmp_path, monkeypatch):
    """Whether the merge is strict or certified changes speed, never bytes."""
    import os
    from roshambo3.prep import prepare

    def build(name, generator_cls):
        monkeypatch.setattr("roshambo3.prep.build.PharmacophoreGenerator", generator_cls, raising=False)
        out = tmp_path / name
        prepare([str(stereo_sdf)], str(out), color_generator=generator_cls(),
                workers=1, max_conf_per_group=64)
        return out

    a = build("certified", PharmacophoreGenerator)
    b = build("strict", UncertifiedGenerator)
    names = sorted(p for p in os.listdir(a / "groups"))
    assert names == sorted(os.listdir(b / "groups"))
    # The group files carry the payload and the format-3 feature table
    # (FORMAT.md 8.1's four arrays are sections of each group file).
    assert names and all(n.endswith(".r3g") for n in names)
    from roshambo3.format.r3g.reader import open_corpus
    corpus = open_corpus(str(a), purpose="inspect")
    with corpus.group(0) as g:
        assert g.feature_offsets is not None and g.feature_atoms is not None
    for n in names:
        assert (a / "groups" / n).read_bytes() == (b / "groups" / n).read_bytes(), n
    for n in ("corpus.json", "molecules.parquet"):
        assert (a / n).read_bytes() == (b / n).read_bytes(), n

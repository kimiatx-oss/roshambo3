"""Deuterium and tritium are hydrogens, and this is a shape model.

`Chem.RemoveHs` by design retains isotopically labeled hydrogens, so a
deuterated compound kept its deuteriums through preparation -- and because the
kernel uses a uniform carbon radius (the fourth coordinate is a padding
mask, not a radius), each retained deuterium was then scored as a **carbon**.

That is not a tolerable approximation whose cost should be measured. Deuterium
has the same van der Waals radius as protium; if protium is stripped for shape
purposes then so is deuterium, and retaining one while inflating it to carbon is
the preparation step failing at its only job.

Both strip sites are pinned here, because fixing one alone would be worse than
fixing neither: a deuterated *query* scored against a stripped *library* is an
asymmetry, where today both sides are wrong in the same direction.
"""
from __future__ import annotations

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem")


def _methane_with(isotope: int):
    m = Chem.AddHs(Chem.MolFromSmiles("C"))
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 1:
            a.SetIsotope(isotope)
            break
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE)
    return m


@pytest.mark.parametrize("isotope,name", [(2, "deuterium"), (3, "tritium")])
def test_the_prep_path_strips_isotopic_hydrogens(isotope, name):
    from roshambo3.prep.conformer import prep_rdkit_mol
    m = _methane_with(isotope)
    assert Chem.RemoveHs(m).GetNumAtoms() == 2, (
        f"RDKit no longer retains {name} under RemoveHs -- this test's premise "
        "is gone and the guard below may be unnecessary")
    out = prep_rdkit_mol(m, remove_hs=True)
    n_real = out[0][1]
    assert n_real == 1, (
        f"prep kept the {name}: {n_real} atoms where methane has one heavy "
        "atom. Under the uniform carbon radius that atom is scored as a carbon.")


@pytest.mark.parametrize("isotope,name", [(2, "deuterium"), (3, "tritium")])
def test_the_query_path_strips_them_the_same_way(isotope, name):
    """The asymmetry is the real hazard: one side stripping and one not."""
    from roshambo3.engine import query as q
    m = _methane_with(isotope)
    qs = q.from_rdkit([m], names=[name]) if hasattr(q, "from_rdkit") else None
    if qs is None:
        heavy = Chem.RemoveAllHs(m)
        assert heavy.GetNumAtoms() == 1
        pytest.skip("engine.query exposes no from_rdkit; strip verified directly")
    assert int(qs.n_real[0]) == 1, (
        f"the query path kept the {name} while prep strips it -- a deuterated "
        "query would be scored against a stripped library")


def test_both_paths_agree_because_there_is_only_one_policy():
    """The DRY guard, stated behaviorally rather than structurally.

    The library and query paths each carried their own `Chem.RemoveHs` call, so
    the isotope defect existed twice and had to be found twice. They now share
    `prep.conformer.strip_hydrogens`. This asserts the *consequence* -- same
    molecule, same heavy-atom count from both entry points -- so it keeps
    working if the shared function is renamed or moved, and fails if either
    caller ever grows its own policy again.
    """
    from roshambo3.engine import query as q
    from roshambo3.prep.conformer import prep_rdkit_mol

    for smiles in ("CCO", "[2H]C([2H])([2H])CO", "[3H]C([3H])([3H])CO",
                   "[2H]OC([2H])([2H])c1ccccc1"):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        from rdkit.Chem import AllChem
        assert AllChem.EmbedMolecule(m, randomSeed=20260907) == 0, smiles
        (_, lib_n, _), = prep_rdkit_mol(m)
        qs = q.from_rdkit([m], names=["q"])
        assert int(qs.n_real[0]) == lib_n, (
            f"{smiles}: query path placed {int(qs.n_real[0])} atoms and the "
            f"library path placed {lib_n} -- the two have diverged, which is "
            "the shape of the isotope defect")
        assert lib_n == Chem.MolFromSmiles(smiles).GetNumHeavyAtoms(), (
            f"{smiles}: heavy-atom count disagrees with RDKit's own")


def test_a_molecule_with_no_heavy_atoms_yields_no_atoms():
    """The one behavior change beyond isotopes, pinned so it is not a surprise.

    `RemoveHs` leaves H2 at two atoms; `RemoveAllHs` leaves zero. Under the old
    strip, H2 entered the corpus as two **carbon-sized** spheres -- a phantom
    two-carbon molecule. It now enters as nothing, which is what a molecule
    with no heavy atoms should be in a shape model.

    This pins `prep_rdkit_mol` in isolation, which emits a zero-atom conformer
    with self-volume 0.0. It is NOT what the corpus builder does: `prepare()`
    refuses such a molecule outright via `build.py::_why_reject`'s pre-existing
    "no heavy atoms" gate, which is pinned separately below. An earlier version
    of this docstring generalized from this function to the pipeline, and was
    wrong to -- this function has no production callers.
    """
    from rdkit.Chem import AllChem
    from roshambo3.prep.conformer import prep_rdkit_mol

    m = Chem.AddHs(Chem.MolFromSmiles("[H][H]"))
    assert AllChem.EmbedMolecule(m, randomSeed=7) == 0
    assert Chem.RemoveHs(m).GetNumAtoms() == 2, (
        "premise: RemoveHs keeps H2 whole; if RDKit changed, revisit this")
    (coords, n_real, vol), = prep_rdkit_mol(m)
    assert n_real == 0 and coords.shape == (0, 3) and vol == 0.0


def test_the_corpus_builder_strips_isotopes_on_the_path_it_actually_uses():
    """The regression test for the first fix's own miss.

    `prep_rdkit_mol` has no callers in `src/` outside its own export. Every
    corpus is built by `prepare()`, which reaches `sources.py::_prep_one` --
    a *different* strip call that the first fix left on `Chem.RemoveHs`, so
    deuteriums survived into real corpora while the unit tests all passed.
    This asserts through `prepare()` for that reason: a test that exercises
    anything else does not cover the defect.
    """
    import os
    import tempfile
    import pyarrow.parquet as pq
    from rdkit.Chem import AllChem
    from roshambo3.prep.build import prepare

    def embed(smiles, name):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(m, randomSeed=0xf00d) == 0
        AllChem.MMFFOptimizeMolecule(m)
        m.SetProp("_Name", name)
        return m

    mols = [embed("CCO", "ethanol"), embed("[2H]C([2H])([2H])CO", "d3-ethanol")]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "corpus")
        prepare(mols, out, name="isotope-check")
        hac = pq.read_table(os.path.join(out, "molecules.parquet")).to_pydict()["hac"]

    assert hac == [3, 3], (
        f"built corpus carries heavy-atom counts {hac}; deuterated ethanol must "
        "match protium ethanol at 3. A 6 here is the isotope defect intact -- the "
        "deuteriums survived and are being scored as carbon-sized Gaussians")


def test_the_corpus_builder_refuses_a_molecule_with_no_heavy_atoms():
    """H2's real disposition, which is refusal -- not the degenerate row that
    `prep_rdkit_mol` alone would emit.

    Stripping isotopes made this reachable: under `RemoveHs`, H2 kept two atoms
    and entered the corpus as a phantom two-carbon molecule, so `_why_reject`'s
    "no heavy atoms" gate could never fire for it. Now it does. Refusing is the
    better of the two, and it is pre-existing policy (USER_GUIDE.md: 1-255
    heavy atoms per conformer, enforced at prepare time), not something isotope
    stripping introduced.
    """
    import os
    import tempfile
    import pyarrow.parquet as pq
    from rdkit.Chem import AllChem
    from roshambo3.prep.build import prepare

    h2 = Chem.AddHs(Chem.MolFromSmiles("[H][H]"))
    assert AllChem.EmbedMolecule(h2, randomSeed=7) == 0
    h2.SetProp("_Name", "H2")
    ethanol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(ethanol, randomSeed=0xf00d) == 0
    ethanol.SetProp("_Name", "ethanol")

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "corpus")
        prepare([ethanol, h2], out, name="h2-check")
        hac = pq.read_table(os.path.join(out, "molecules.parquet")).to_pydict()["hac"]

    assert hac == [3], (
        f"expected H2 refused and only ethanol kept, got heavy-atom counts {hac}")


def test_the_binary_zstd_source_strips_isotopes_too(tmp_path):
    """The third source class, which no other test covers for isotopes.

    `RdkitBinaryZstdSource` shares `_group_by_name`/`_prep_one` with the SDF
    and in-memory sources, so it is correct for the same reason they are --
    but "correct for the same reason" is exactly the inference that failed
    before. The branch that was broken was the one no test exercised, so this
    exercises it.
    """
    import os
    import struct

    import pyarrow.parquet as pq
    zstandard = pytest.importorskip("zstandard")
    from rdkit.Chem import AllChem
    from roshambo3.prep.build import prepare

    mols = []
    for smiles, name in (("CCO", "ethanol"), ("[2H]C([2H])([2H])CO", "d3-ethanol")):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(m, randomSeed=0xf00d) == 0
        AllChem.MMFFOptimizeMolecule(m)
        m.SetProp("_Name", name)
        mols.append(m)

    src = tmp_path / "mols.bin.zst"
    with open(src, "wb") as fh:
        with zstandard.ZstdCompressor().stream_writer(fh) as w:
            for m in mols:
                blob = m.ToBinary()
                w.write(struct.pack("<I", len(blob)))
                w.write(blob)

    out = str(tmp_path / "corpus")
    prepare(str(src), out, name="binzst-isotope-check",
            input_format="rdkit-binary-zstd")
    hac = pq.read_table(os.path.join(out, "molecules.parquet")).to_pydict()["hac"]

    assert hac == [3, 3], (
        f"binary-zstd ingestion produced heavy-atom counts {hac}; a 6 is the "
        "isotope defect surviving in the one source class the suite did not cover")

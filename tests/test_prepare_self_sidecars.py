"""The builder's self-overlaps describe the stored coordinates (FORMAT.md
section 18), per molecule-aligned block.

Format 3 keeps ``vaa`` (f32) and ``color_self`` (f64) as sections of each group
file, in corpus row order; there are no sidecar files, no row-binding digests
and no block indexes (they bound format-2 sidecars to format-2 blocks). What
must hold instead: every stored value is the self-overlap of the row as a
reader decodes it -- ``vaa`` of the dequantized heavy atoms, ``color_self`` of
the derived pseudo-atoms (FORMAT.md 8.3) under the corpus's matrices -- block
by block, and ``color_self`` is written only for canonical matrices.

Engine-side binding of these values (receipts, routes) is tested in
``test_sidecar_receipts.py`` and ``test_self_overlap_routes.py``; the format-2
helpers they imported from here are gone with the format-2 builder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3.backend import cpu
from roshambo3.format.r3g import reference as R
from roshambo3.format.r3g.limits import LIMITS
from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep import prepare
from roshambo3.prep.conformer import stored_self_volume
from roshambo3.reference.overlap import color_overlap
from tests.format3_rows import read_corpus

STEP = 2.0 ** -8
BLOCK_CAP = LIMITS["block_rows"]


def _jittered_library(n_mols=17, n_confs=256, smiles="Oc1ccc(N)cc1C(=O)O", seed=7):
    base = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(base, randomSeed=13) == 0
    xyz = base.GetConformer().GetPositions()
    rng = np.random.default_rng(seed)
    mols = []
    for m in range(n_mols):
        mol = Chem.Mol(base)
        mol.RemoveAllConformers()
        for _ in range(n_confs):
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, p in enumerate(xyz + rng.normal(0, 0.15, xyz.shape)):
                conf.SetAtomPosition(i, p.tolist())
            mol.AddConformer(conf, assignId=True)
        mol.SetProp("_Name", f"mol-{m}")
        mols.append(mol)
    return base, mols


def _matrices(generator):
    return (np.ascontiguousarray(generator.interaction_matrix_r, np.float32),
            np.ascontiguousarray(generator.interaction_matrix_p, np.float32))


def _format2_helper_gone(name):
    def gone(*args, **kwargs):
        raise NotImplementedError(
            f"{name} read format-2 sidecars and block indexes; the builder writes "
            f"format 3 (self-overlaps are group-file sections); the engine-side "
            f"tests that used it must be ported.")
    gone.__name__ = name
    return gone


#: Imported by ``test_sidecar_receipts.py`` and ``test_self_overlap_routes.py``
#: (still to be ported): kept importable so those files fail at the call, by name,
#: rather than stopping a whole test run at collection.
_blocks = _format2_helper_gone("_blocks")
_block_inputs = _format2_helper_gone("_block_inputs")
_engine_inputs = _format2_helper_gone("_engine_inputs")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cpu.extension()  # no green skip without the native path
    root = tmp_path_factory.mktemp("t208")
    base, mols = _jittered_library()
    generator = PharmacophoreGenerator()
    prepare(mols, str(root / "corpus"), workers=2, color_generator=generator)
    return read_corpus(root / "corpus", merged=False), base, generator


def test_a_group_of_several_molecule_aligned_blocks(built):
    """17 molecules of 256 conformers in one group: the first block holds 16
    whole molecules (exactly 4,096 rows), the second the last molecule --
    never a molecule split at row 4,096 (format 2 cut there)."""
    rows, _, _ = built
    (group,) = rows.groups
    assert group.n_conf == 17 * 256 > BLOCK_CAP
    assert [(int(b["first_row"]), int(b["rows"]), int(b["m0"]), int(b["m1"]))
            for b in group.blocks] == [(0, BLOCK_CAP, 0, 16), (BLOCK_CAP, 256, 16, 17)]
    assert rows.build["color"]["color_self"]["status"] == "built"
    assert group.vaa.dtype == np.dtype("<f4") and group.vaa.shape == (group.n_conf,)
    assert group.color_self.dtype == np.dtype("<f8")
    assert group.color_self.shape == (group.n_conf,)


def test_every_block_holds_the_self_overlaps_of_its_decoded_rows(built):
    """Per molecule-aligned block, from the block as the native decoder
    returns it: ``vaa`` is ``stored_self_volume`` of the dequantized heavy
    atoms, bit for bit; ``color_self`` is the native path's value for the
    derived pseudo-atoms, bit for bit, and within FORMAT.md 18's 1e-5 of the
    float64 reference."""
    rows, _, generator = built
    r, p = _matrices(generator)
    from roshambo3.format.r3g.reader import open_corpus
    corpus = open_corpus(rows.root, purpose="inspect")
    checked = 0
    for gi, g in enumerate(rows.groups):
        with corpus.group(gi) as reader_group:
            for bi, b in enumerate(g.blocks):
                xyz = reader_group.decode_block(bi)             # native, as a reader
                r0, n = int(b["first_row"]), int(b["rows"])
                assert np.array_equal(xyz, g.xyz[r0:r0 + n])
                row_mol = g.run_start
                width = g.L
                library = np.zeros((n, width, 3), np.float32)
                counts = g.counts[r0:r0 + n]
                for k in range(n):
                    row = r0 + k
                    m = int(np.searchsorted(row_mol, row, side="right") - 1)
                    nr = int(g.n_real[row])
                    heavy = xyz[k, :, :nr].T.astype(np.float32) * np.float32(STEP)
                    assert np.float32(stored_self_volume(heavy)) == g.vaa[row]
                    feats = g.features[m]
                    pos = R.derive_positions(xyz[k], feats)
                    library[k, :nr] = heavy
                    library[k, nr:nr + len(feats)] = pos.astype(np.float32) * np.float32(STEP)
                    t = np.asarray([f[0] for f in feats], dtype=np.int64)
                    want = color_overlap(pos.astype(np.float64) * STEP, t,
                                         pos.astype(np.float64) * STEP, t, r, p)
                    assert abs(g.color_self[row] - want) <= 1e-5 * max(want, 1e-30)
                    checked += 1
                native = cpu.prepare_combo_library_self_overlaps(
                    library_coords=library, library_counts=np.ascontiguousarray(counts),
                    interaction_r=r, interaction_p=p, shape=False, color=True)
                np.testing.assert_array_equal(np.asarray(native.color_values, np.float64),
                                              g.color_self[r0:r0 + n])
    assert checked == rows.n_conf


def _non_canonical(p):
    p = np.array(p, dtype=np.float32, copy=True)
    p[1, 2] = p[2, 1] = 0.5
    return p


def test_non_canonical_or_disabled_build_persists_no_color_self(tmp_path):
    cpu.extension()
    mol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1N"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    mol.SetProp("_Name", "one")
    generator = PharmacophoreGenerator()
    generator.interaction_matrix_p = _non_canonical(generator.interaction_matrix_p)
    prepare([mol], str(tmp_path / "nc"), workers=1, color_generator=generator)
    nc = read_corpus(tmp_path / "nc", merged=False)
    record = nc.build["color"]["color_self"]
    assert record["status"] == "not_built" and "non-canonical" in record["reason"]
    assert record["matrices_canonical"] is False
    assert all(g.color_self is None for g in nc.groups)
    assert not R.canonical_matrices(nc.descriptor["color"]["r"]["values"],
                                    nc.descriptor["color"]["p"]["values"], 6)
    prepare([mol], str(tmp_path / "off"), workers=1,
            color_generator=PharmacophoreGenerator(), color_self=False)
    off = read_corpus(tmp_path / "off", merged=False)
    assert off.build["color"]["color_self"]["status"] == "not_built"
    assert "disabled" in off.build["color"]["color_self"]["reason"]
    assert all(g.color_self is None for g in off.groups)
    for root in (tmp_path / "nc", tmp_path / "off"):
        assert sorted(p.name for p in Path(root).iterdir()) == [
            "build.json", "corpus.json", "groups", "molecules.parquet"]
    from roshambo3.cli import build_parser
    assert build_parser().parse_args(["prepare", "x", "-o", "y", "--no-color-self"]).no_color_self

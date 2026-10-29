#!/usr/bin/env python3
"""Write the format-3 conformance corpus (``doc/FORMAT.md``).

Two corpora, ``shape/`` and ``color/``, from a fixed SMILES list embedded with
RDKit ETKDG at fixed seeds, plus synthetic molecules whose coordinates are
drawn from a seeded generator: a 255-heavy-atom chain (both), a two-atom
molecule with exactly 4,096 conformers (color: the 4,096-row block) and a
one-atom molecule with 4,096 conformers (shape: a single-molecule block at
the 4,096-row cap). Color
features are RDKit ``BaseFeatures.fdef`` perceived with hydrogens present.

**The committed files are the oracle; this script is provenance.** RDKit
releases differ in embedding and canonical atom order, so a regeneration is
byte-identical only under the recorded RDKit, NumPy and pyarrow versions
(``EXPECTED.json["provenance"]``). Run from the repository root:

    PYTHONPATH=src python tests/data/format3/make_conformance.py

It rewrites ``shape/``, ``color/`` and ``EXPECTED.json`` next to itself. The
self-overlaps are computed with the fp64 references
(``prep.conformer.stored_self_volume`` for ``vaa``,
``reference.overlap.color_overlap`` for ``color_self``), not the native path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from roshambo3.format.r3g import reference as R  # noqa: E402
from roshambo3.prep.conformer import (prep_conformer, stable_id,  # noqa: E402
                                      stored_self_volume)
from roshambo3.reference.overlap import color_overlap  # noqa: E402
from roshambo3.pharmacophore import WORK_DESCENDING_FEATURE_ORDER  # noqa: E402

TYPES = [name for name, _ in WORK_DESCENDING_FEATURE_ORDER]
TYPE_ID = {name: i + 1 for i, name in enumerate(TYPES)}

BLOCK_ROWS = 16
MAX_CONF_PER_GROUP = 40
SEED = 20261017

#: (ext_id, SMILES, conformers). Chosen to populate many buckets, to include a
#: 1-heavy-atom molecule, aromatic rings, a carboxylate, a guanidine, charged
#: groups, halogens, and a few large molecules.
SMILES = [
    ("methane", "C", 1),
    ("ethanol", "CCO", 2),
    ("acetate", "CC(=O)[O-]", 3),
    ("benzene", "c1ccccc1", 2),
    ("pyridine", "c1ccncc1", 3),
    ("urea", "NC(N)=O", 2),
    ("guanidinium", "NC(N)=[NH2+]", 2),
    ("methylammonium", "C[NH3+]", 2),
    ("toluene", "Cc1ccccc1", 3),
    ("phenol", "Oc1ccccc1", 3),
    ("benzoic_acid", "OC(=O)c1ccccc1", 4),
    ("benzoate", "[O-]C(=O)c1ccccc1", 4),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", 4),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", 3),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", 4),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1", 3),
    ("nicotine", "CN1CCC[C@H]1c1cccnc1", 3),
    ("chlorobenzene", "Clc1ccccc1", 2),
    ("bromophenol", "Oc1ccc(Br)cc1", 2),
    ("indole", "c1ccc2[nH]ccc2c1", 2),
    ("naphthalene", "c1ccc2ccccc2c1", 2),
    ("imidazole", "c1cnc[nH]1", 2),
    ("histamine", "NCCc1c[nH]cn1", 3),
    ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", 3),
    ("adenine", "Nc1ncnc2[nH]cnc12", 2),
    ("tryptophan", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O", 3),
    ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", 3),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", 3),
    ("sildenafil", "CCCc1nn(C)c2c(=O)[nH]c(nc12)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1", 2),
    ("imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(n1)-c1cccnc1", 2),
    ("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", 2),
    ("ritonavir", "CC(C)c1nc(CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)cs1", 2),
    ("cyclosporin_fragment", "CC[C@H](C)[C@H]1C(=O)N(C)[C@H](CC(C)C)C(=O)N(C)[C@H](CC(C)C)C(=O)N[C@H](C(C)C)C(=O)N(C)[C@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@H](CC(C)C)C(=O)N(C)[C@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C", 1),
    ("tetrapeptide", "CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O", 2),
    ("decapeptide_like", "CC[C@H](C)[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCNC(N)=N)NC(=O)[C@H](CO)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CCCCN)NC(=O)[C@H](C)NC(=O)[C@@H](N)CC(=O)O)C(=O)O", 1),
    ("pamoic_acid", "OC(=O)c1cc2ccccc2c(Cc2c(O)c(C(=O)O)cc3ccccc23)c1O", 2),
    ("fluoxetine", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1", 3),
    ("sulfonamide", "NS(=O)(=O)c1ccc(N)cc1", 2),
    ("nitrobenzene", "O=[N+]([O-])c1ccccc1", 2),
    ("thiophenol", "Sc1ccccc1", 2),
    ("tert_butylbenzene", "CC(C)(C)c1ccccc1", 2),
    ("cyclopropane", "C1CC1", 2),
    ("cyclohexane", "C1CCCCC1", 3),
    ("biphenyl_diamine", "Nc1ccc(cc1)-c1ccc(N)cc1", 2),
    ("azulene", "c1ccc2cccc2cc1", 2),
]

SYNTHETIC = [
    # (ext_id, smiles, n_conf, generator): coordinates from a seeded generator.
    ("chain255", "C" * 255, 1, "walk"),
]
#: the exactly-4,096-row single-molecule block (color only, for the byte budget)
COLOR_ONLY_SYNTHETIC = [
    ("ethane4096", "CC", 4096, "ethane"),
]
#: a single-molecule block above 4,096 rows (shape only); every coordinate is
#: zero, so the payload is the 32-byte header and the rows cost 4 bytes each
SHAPE_ONLY_SYNTHETIC = [
    ("one4096", "C", 4096, "zero"),
]


def canonical_heavy(mol):
    """``(mol_without_H renumbered into canonical SMILES order, smiles, perm)``;
    ``perm[old] = new`` over heavy atoms."""
    from rdkit import Chem
    heavy = Chem.RemoveAllHs(mol)
    smiles = Chem.MolToSmiles(heavy, isomericSmiles=False)   # section 11.1: no stereo
    order = [int(x) for x in heavy.GetProp("_smilesAtomOutputOrder").strip("[],").split(",") if x]
    assert sorted(order) == list(range(heavy.GetNumAtoms()))
    renum = Chem.RenumberAtoms(heavy, order)
    check = Chem.MolFromSmiles(smiles)
    assert check is not None and check.GetNumAtoms() == renum.GetNumAtoms()
    for a, b in zip(check.GetAtoms(), renum.GetAtoms()):
        assert a.GetAtomicNum() == b.GetAtomicNum(), (smiles, "element order")
    assert Chem.MolToSmiles(renum, isomericSmiles=False) == smiles
    perm = {old: new for new, old in enumerate(order)}
    return renum, smiles, perm


def perceive_features(heavy_mol, factory):
    """Features on an already canonical heavy molecule; AddHs preserves its indices."""
    from rdkit import Chem
    n_heavy = heavy_mol.GetNumAtoms()
    mol_h = Chem.AddHs(heavy_mol)
    feats = []
    for f in factory.GetFeaturesForMol(mol_h):
        fam = f.GetFamily()
        if fam not in TYPE_ID:
            continue
        ids = [int(i) for i in f.GetAtomIds()]
        assert max(ids) < n_heavy, (fam, "feature references a hydrogen")
        feats.append((TYPE_ID[fam], tuple(ids)))
    return R.canonical_feature_order(feats)


def embedded_coords(smiles, n_conf, seed):
    """int16[n_conf, 3, n_real] in canonical order, prepared and quantized."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=params))
    assert len(ids) == n_conf, (smiles, len(ids))
    heavy, canon, perm = canonical_heavy(mol)
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    # after RemoveAllHs the heavy atoms keep their relative order; RenumberAtoms
    # then applies `order`, so old heavy index i -> new index perm[i]
    out = []
    for cid in ids:
        pos = np.asarray(mol.GetConformer(cid).GetPositions(), dtype=np.float64)[heavy_idx]
        new = np.empty_like(pos)
        for old, nw in perm.items():
            new[nw] = pos[old]
        q = R.quantize(prep_conformer(new))
        out.append(q.T)
    return np.stack(out).astype(np.int16), heavy, canon, perm


def synthetic_coords(kind, n_conf, n_real, rng):
    """int16[n_conf, 3, n_real] integer coordinates (already on the grid)."""
    out = np.zeros((n_conf, 3, n_real), dtype=np.int64)
    if kind == "walk":
        # a 3-D random walk with 1.5 A steps and jitter: the 255-atom chain
        step = 1.5 / R.STEP
        for c in range(n_conf):
            d = rng.normal(size=(n_real, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            pos = np.cumsum(d * step, axis=0) + rng.normal(scale=20.0, size=(n_real, 3))
            pos -= pos.mean(axis=0)
            out[c] = np.rint(pos).T
    elif kind == "ethane":
        # two atoms on the x axis: y and z are identically zero (w = 0)
        half = rng.integers(180, 200, size=n_conf)
        out[:, 0, 0] = -half
        out[:, 0, 1] = half
    elif kind == "zero":
        pass
    else:
        raise ValueError(kind)
    assert out.min() >= -32768 and out.max() <= 32767
    return out.astype(np.int16)


def self_overlaps(coords, features, r, p):
    """``(vaa f32[n_conf], color_self f64[n_conf])`` from the stored integers."""
    vaa = np.empty(coords.shape[0], dtype=np.float32)
    cself = np.zeros(coords.shape[0], dtype=np.float64)
    types = np.asarray([t for t, _ in features], dtype=np.int64)
    for c in range(coords.shape[0]):
        xyz = R.dequantize(coords[c]).astype(np.float64)   # [3, n_real]
        vaa[c] = np.float32(stored_self_volume(xyz.T))
        if len(features):
            pos = R.dequantize(R.derive_positions(coords[c], features)).astype(np.float64)
            cself[c] = color_overlap(pos, types, pos, types, r, p)
    return vaa, cself


def build_molecules(color: bool):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit import RDConfig
    factory = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    n = len(TYPES)
    r = np.zeros((n + 1, n + 1), dtype=np.float32)
    p = np.zeros((n + 1, n + 1), dtype=np.float32)
    for t in range(1, n + 1):
        r[t, t] = p[t, t] = 1.0
    rng = np.random.default_rng(SEED)
    molecules = []
    extra = COLOR_ONLY_SYNTHETIC if color else SHAPE_ONLY_SYNTHETIC
    rows = list(SMILES) + [(e, s, k) for e, s, k, _ in SYNTHETIC + extra]
    synthetic = {e: g for e, _, _, g in SYNTHETIC + COLOR_ONLY_SYNTHETIC + SHAPE_ONLY_SYNTHETIC}
    for i, (ext_id, smiles, n_conf) in enumerate(rows):
        if ext_id in synthetic:
            heavy, canon, perm = canonical_heavy(Chem.MolFromSmiles(smiles))
            coords = synthetic_coords(synthetic[ext_id], n_conf, heavy.GetNumAtoms(), rng)
        else:
            coords, heavy, canon, perm = embedded_coords(smiles, n_conf, SEED + i)
        feats = perceive_features(heavy, factory) if color else []
        assert len(feats) <= 255 and heavy.GetNumAtoms() + len(feats) <= 512, ext_id
        vaa, cself = self_overlaps(coords, feats, r, p)
        molecules.append(R.Molecule(
            mol_id=stable_id(ext_id), coords=coords, vaa=vaa, features=feats,
            color_self=cself if color else None, ext_id=ext_id, smiles=canon,
            mw=float(Descriptors.MolWt(heavy)),
            rotors=int(rdMolDescriptors.CalcNumRotatableBonds(heavy))))
    color_desc = None
    if color:
        import rdkit
        text = open(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"), encoding="utf-8").read()
        color_desc = R.color_descriptor(generator="rdkit_fdef", definition=text, types=TYPES,
                                        r=r, p=p, rdkit_version=rdkit.__version__)
    return molecules, color_desc


def digest(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def expected_for(root: str) -> dict:
    """Decode a corpus with the reference and digest every array."""
    descriptor, fp = R.read_descriptor(root)
    out = {"fingerprint": fp, "ids_sha256": descriptor["ids_sha256"],
           "n_conf": descriptor["n_conf"], "n_mol": descriptor["n_mol"],
           "corpus_json_sha256": R.sha256_hex(open(os.path.join(root, "corpus.json"), "rb").read()),
           "molecules_sha256": descriptor["molecules_sha256"],
           "groups": {}}
    for entry in descriptor["groups"]:
        name = R.group_file_name(entry["bucket"], entry["group"])
        path = os.path.join(root, name)
        gf = R.open_group(path, root=root, descriptor=descriptor, group_entry=entry, fingerprint_hex=fp)
        try:
            d = R.decode_group(gf, descriptor)
            blocks = gf.section_array("blocks")
            widths = []
            for i in range(blocks.size):
                widths.append(list(R.parse_block_header(gf.block_bytes(i)).widths))
            g = {
                "file_sha256": R.sha256_hex(open(path, "rb").read()),
                "n_conf": gf.header["n_conf"], "n_mol": gf.header["n_mol"],
                "L_heavy": gf.header["L_heavy"], "L": gf.header["L"],
                "L_field_width": entry["L_field_width"],
                "block_rows": [int(b["rows"]) for b in blocks],
                "block_widths": widths,
                "heavy_sha256": digest(d["heavy"]),
                "merged_sha256": digest(d["merged"]),
                "counts_sha256": digest(d["counts"]),
                "pseudo_sha256": digest(d["pseudo"]),
                "runs_sha256": digest(d["runs"]),
                "vaa_sha256": digest(d["vaa"]),
            }
            if d["features"] is not None:
                for k, v in d["features"].items():
                    g[f"{k}_sha256"] = digest(v)
            if d["color_self"] is not None:
                g["color_self_sha256"] = digest(d["color_self"])
            out["groups"][name] = g
        finally:
            gf.close()
    return out


def write_width_vectors() -> dict:
    """``widths/wNN.bin``: one block per width 1-7 and 14-16 (one row of 16 atoms, so
    the stream is one packing group with a 240-position tail), plus ``wrap.bin``,
    a width-9 block whose ``min_x`` was raised after encoding so that
    ``min + u`` leaves int16 and the decoder's modulo-2^16 wrap is exercised.
    ``widths/EXPECTED.json`` carries each block's inputs and decoded digest."""
    root = os.path.join(HERE, "widths")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    rng = np.random.default_rng(SEED + 7)
    out = {}
    n_real = 16
    for w in (*range(1, 8), 14, 15, 16):      # 0 and 8-13 occur in the corpora
        xyz = np.zeros((1, 3, n_real), np.int16)
        for a in range(3):
            lo = int(rng.integers(-32768, 32768 - (2 ** w - 1))) if w < 16 else -32768
            vals = lo + rng.integers(0, 2 ** w, n_real) if w else np.full(n_real, lo)
            vals[0], vals[-1] = lo, lo + (2 ** w - 1)      # pin the extremes
            xyz[0, a] = vals
        blk = R.encode_block(xyz, [n_real])
        assert list(R.parse_block_header(blk).widths) == [w, w, w]
        name = f"w{w:02d}.bin"
        open(os.path.join(root, name), "wb").write(blk)
        out[name] = {"n_real": [n_real], "L": n_real, "widths": [w, w, w],
                     "decoded_sha256": digest(R.decode_block(blk, [n_real], n_real))}
    xyz = np.zeros((1, 3, 8), np.int16)
    xyz[0, 0, :3] = [32000, 32300, 32767]
    blk = bytearray(R.encode_block(xyz, [3]))
    blk[0:2] = (32600).to_bytes(2, "little", signed=True)     # min + 767 = 33367 wraps
    blk = bytes(blk)
    out["wrap.bin"] = {"n_real": [3], "L": 8, "widths": list(R.parse_block_header(blk).widths),
                       "decoded_sha256": digest(R.decode_block(blk, [3], 8)),
                       "decoded": R.decode_block(blk, [3], 8)[0].tolist()}
    open(os.path.join(root, "wrap.bin"), "wb").write(blk)
    with open(os.path.join(root, "EXPECTED.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return out


def main() -> None:
    import rdkit
    import pyarrow
    expected = {"provenance": {
        "rdkit": rdkit.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__,
        "python": platform.python_version(), "block_rows": BLOCK_ROWS,
        "max_conf_per_group": MAX_CONF_PER_GROUP, "seed": SEED,
        "self_overlaps": "fp64 references: prep.conformer.stored_self_volume (vaa), "
                         "reference.overlap.color_overlap (color_self)"}}
    for variant, color in (("shape", False), ("color", True)):
        root = os.path.join(HERE, variant)
        shutil.rmtree(root, ignore_errors=True)
        molecules, color_desc = build_molecules(color)
        R.write_corpus(root, molecules, color=color_desc, prep_version="0.1.0.dev0",
                       prep_commit=None, block_rows=BLOCK_ROWS,
                       max_conf_per_group=MAX_CONF_PER_GROUP,
                       build={"block_rows": BLOCK_ROWS, "max_conf_per_group": MAX_CONF_PER_GROUP,
                              "source": "tests/data/format3/make_conformance.py", "seed": SEED})
        R.validate_corpus(root, "deep")
        expected[variant] = expected_for(root)
    write_width_vectors()
    with open(os.path.join(HERE, "EXPECTED.json"), "w", encoding="utf-8") as fh:
        json.dump(expected, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")
    total = 0
    for dirpath, _, files in os.walk(HERE):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    print(f"wrote shape/ and color/; tests/data/format3 totals {total} bytes")


if __name__ == "__main__":
    main()

"""Shared builder fixtures.

``fixtures()`` holds no charge or protonation variant under one id: format 3
refuses one at build. It keeps the cases the builder must
accept under one id -- a stereo-only variant, an atom-order-only variant and
a deuterated molecule -- and puts the two protonation states of propylamine
under ids of their own, as a refused input is told to.
"""
from __future__ import annotations

import struct

from rdkit import Chem
from rdkit.Chem import AllChem
import zstandard as zstd


def molecule(smiles="CCO", name="ethanol", conformers=1, seed=13):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert len(AllChem.EmbedMultipleConfs(mol, numConfs=conformers, randomSeed=seed)) == conformers
    mol.SetProp("_Name", name)
    return mol


def fixtures():
    first = molecule("C[C@H](O)C(=O)O", "stereo")
    opposite = molecule("C[C@@H](O)C(=O)O", "stereo")
    order = list(reversed(range(first.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(first, order)
    reordered.SetProp("_Name", "stereo")
    return [first, molecule("CCCN", "amine"), molecule("[2H]OC", "isotope"),
            opposite, molecule("CCC[NH3+]", "ammonium"), reordered,
            molecule("CCO", "multi", conformers=3)]


def write_source(path, mols, binary=False):
    if binary:
        # A binary record is a molecule; duplicate names are rejected by the
        # existing binary builder. Use unique IDs, preserving multi-conformers.
        blobs = []
        for i, mol in enumerate(mols):
            one = Chem.Mol(mol)
            one.SetProp("_Name", f"binary-{i}")
            blob = one.ToBinary(Chem.PropertyPickleOptions.AllProps)
            blobs.append(struct.pack("<I", len(blob)) + blob)
        path.write_bytes(zstd.ZstdCompressor().compress(b"".join(blobs)))
    else:
        with Chem.SDWriter(str(path)) as writer:
            for mol in mols:
                for conf in mol.GetConformers():
                    writer.write(mol, confId=conf.GetId())


def split_variant_ids(src, dst) -> dict:
    """Copy the SDF ``src`` to ``dst``, giving every protonation (or other
    non-stereo) state of a molecule id its own id: the second and later
    distinct hydrogen-free graphs under one name are renamed
    ``<name>|state<k>``. Only title lines change; every other byte of every
    record is kept. Returns ``{name: number of states}`` for the renamed ids.

    ROSHAMBO2's ``test/confs.sdf`` puts protonation states of 70 of its 480
    molecules under one name, which a format-3 build refuses.
    """
    from roshambo3.prep.sources import _graph_key, canonical_heavy

    data = open(src, "rb").read()
    records = data.split(b"$$$$")
    tail = records.pop()                         # after the last delimiter
    states: dict = {}
    out = []
    for block in records:
        lead = block[:len(block) - len(block.lstrip(b"\r\n"))]
        body = block[len(lead):]
        mol = Chem.MolFromMolBlock(body.decode(), removeHs=False)
        title, _, rest = body.partition(b"\n")
        if mol is not None:
            name = title.decode().strip()
            key = _graph_key(canonical_heavy(mol)[0], stereo=False)
            keys = states.setdefault(name, [])
            if key not in keys:
                keys.append(key)
            k = keys.index(key)
            if k:
                title = f"{name}|state{k}".encode()
        out.append(lead + title + b"\n" + rest)
    with open(dst, "wb") as f:
        f.write(b"$$$$".join(out + [tail]))
    return {n: len(k) for n, k in states.items() if len(k) > 1}


# --------------------------------------------------------------------------
# content order: a library covering every bucket

#: ``format/r3g/limits.py::BUCKET_EDGES``.
BUCKET_EDGES = (0, 12, 16, 20, 24, 28, 32, 38, 44, 50, 56, 64, 72, 256)


def _chain_smiles(rng, n_heavy: int) -> str:
    """A valid acyclic-backbone SMILES of exactly ``n_heavy`` heavy atoms:
    carbons with some O/N, para-phenylene links and small substituents."""
    out, left = ["C"], n_heavy - 1
    while left > 0:
        r = rng.random()
        if left >= 7 and r < 0.15:
            out.append("c1ccc(cc1)")
            left -= 6
        elif left >= 3 and r < 0.25:
            out.append("C(=O)")
            left -= 2
        elif left >= 3 and r < 0.33:
            out.append("C(F)")
            left -= 2
        elif r < 0.45 and out[-1] not in ("O", "N"):
            out.append(rng.choice("ON"))
            left -= 1
        else:
            out.append("C")
            left -= 1
    return "".join(out)


def synthetic_library(n_molecules=40, seed=291, max_conformers=3, max_heavy=80):
    """``n_molecules`` molecules named ``SYN-NNNN`` cycling over every
    bucket (up to ``max_heavy`` heavy atoms, so bucket 12 -- 72 and up -- is
    populated), each with 1..``max_conformers`` embedded conformers."""
    import random
    rng = random.Random(seed)
    mols = []
    for i in range(n_molecules):
        b = i % (len(BUCKET_EDGES) - 1)
        lo = max(3, BUCKET_EDGES[b])
        hi = min(BUCKET_EDGES[b + 1] - 1, max_heavy)
        n_heavy = rng.randint(lo, hi)
        mol = Chem.AddHs(Chem.MolFromSmiles(_chain_smiles(rng, n_heavy)))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + i
        params.useRandomCoords = True
        n = rng.randint(1, max_conformers)
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
        assert len(ids) == n, (i, n_heavy)
        mol.SetProp("_Name", f"SYN-{i:04d}")
        mols.append(mol)
    return mols


def write_sdf(path, mols):
    """One record per conformer, a molecule's records together and in
    conformer order (which defines ``conf_idx``)."""
    with Chem.SDWriter(str(path)) as writer:
        for mol in mols:
            for conf in mol.GetConformers():
                writer.write(mol, confId=conf.GetId())


def corpus_rows(corpus) -> dict:
    """A built format-3 corpus's decoded rows in gid order: ``bucket``,
    ``n_real``, ``mol_id``, ``conf_idx``, ``vaa``, the kernel-layout ``int16
    [rows, 3, L]`` planes (heavy atoms and derived pseudo-atoms; per group,
    as a list) and ``L``, the molecule table, and the bytes of every file but
    ``build.json`` (provenance: timings, workers, paths) -- the content a
    rebuild must reproduce (byte-identical files)."""
    import numpy as np
    import pyarrow.parquet as pq
    from pathlib import Path
    from tests.format3_rows import corpus_files, read_corpus
    root = Path(getattr(corpus, "root", corpus))
    c = read_corpus(root)
    out = {"bucket": np.concatenate([np.full(g.n_conf, g.bucket) for g in c.groups]),
           "n_real": c.n_real, "mol_id": c.mol_id, "conf_idx": c.conf_idx, "vaa": c.vaa,
           "xyz": [g.merged for g in c.groups], "L": [g.L for g in c.groups],
           "molecules": pq.read_table(root / "molecules.parquet")}
    files = corpus_files(root)
    files.pop("build.json", None)
    out["files"] = files
    return out


def assert_same_rows(a: dict, b: dict) -> None:
    import numpy as np
    for k in ("bucket", "n_real", "mol_id", "conf_idx", "vaa"):
        assert np.array_equal(a[k], b[k]), k
    assert a["L"] == b["L"]
    assert len(a["xyz"]) == len(b["xyz"])
    for x, y in zip(a["xyz"], b["xyz"]):
        assert x.dtype == y.dtype and np.array_equal(x, y)
    assert a["molecules"].equals(b["molecules"])
    assert sorted(a["files"]) == sorted(b["files"])
    for name in a["files"]:
        assert a["files"][name] == b["files"][name], name

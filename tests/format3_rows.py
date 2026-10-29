"""Read a built format-3 corpus back as per-row arrays, for builder tests.

``read_corpus(root)`` opens the corpus through the format-3 reader (running
``verify --tier scan`` first if no cached scan covers it), decodes every
block, and returns per-group and corpus-wide row arrays in corpus order:
``n_real``, ``mol_id``, ``conf_idx`` (a row's offset in its molecule's run,
FORMAT.md section 7 rule 1), ``vaa``, ``color_self``, ``counts`` (the derived count vector,
8.5), ``xyz`` (``int16[rows, 3, L_heavy]``) and ``merged`` (the kernel layout,
8.4, derived pseudo-atoms included), plus ``molecules.parquet`` as columns
and ``build.json``. For test-sized corpora: everything is held in memory.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional

import numpy as np

from roshambo3.format.r3g.limits import group_file_name

STEP = 2.0 ** -8


@dataclasses.dataclass
class GroupRows:
    index: int
    bucket: int
    group: int
    path: str
    L: int
    L_heavy: int
    L_field_width: int
    n_conf: int
    n_mol: int
    blocks: np.ndarray              # BLOCK_DTYPE records
    runs: np.ndarray                # RUN_DTYPE records
    features: list                  # per molecule: [(type, atoms), ...]
    feature_table: dict             # the four CSR sections (FORMAT.md 8.1), raw
    n_real: np.ndarray
    mol_id: np.ndarray
    conf_idx: np.ndarray
    vaa: np.ndarray
    color_self: Optional[np.ndarray]
    counts: np.ndarray
    xyz: np.ndarray                 # int16[rows, 3, L_heavy]
    merged: np.ndarray              # int16[rows, 3, L]
    header: dict

    @property
    def run_start(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.runs["n_conf"].astype(np.int64))])


@dataclasses.dataclass
class CorpusRows:
    root: str
    descriptor: dict
    build: Optional[dict]
    groups: list
    molecules: dict                 # molecules.parquet, column -> list/array

    def _cat(self, name):
        return np.concatenate([getattr(g, name) for g in self.groups])

    n_real = property(lambda self: self._cat("n_real"))
    mol_id = property(lambda self: self._cat("mol_id"))
    conf_idx = property(lambda self: self._cat("conf_idx"))
    vaa = property(lambda self: self._cat("vaa"))

    @property
    def n_conf(self) -> int:
        return int(self.descriptor["n_conf"])

    @property
    def n_mol(self) -> int:
        return int(self.descriptor["n_mol"])

    def coordinates(self, gid: int) -> np.ndarray:
        """Row ``gid``'s heavy atoms in Angstrom, ``float32[n_real, 3]``
        (dequantized at the step, exactly)."""
        for g in self.groups:
            if gid < g.n_conf:
                n = int(g.n_real[gid])
                return g.xyz[gid, :, :n].T.astype(np.float32) * np.float32(STEP)
            gid -= g.n_conf
        raise IndexError(gid)

    def molecule_row(self, mol_id: int) -> dict:
        i = int(np.searchsorted(self.molecules["mol_id"], np.uint64(mol_id)))
        assert int(self.molecules["mol_id"][i]) == int(mol_id), mol_id
        return {k: v[i] for k, v in self.molecules.items()}


def merged_layout(xyz: np.ndarray, n_real: np.ndarray, features: list,
                  row_mol: np.ndarray, L: int) -> np.ndarray:
    """The kernel layout of FORMAT.md 8.4 from decoded heavy atoms and the
    feature table, with the format's own integer-mean rule (reference)."""
    from roshambo3.format.r3g.reference import kernel_layout
    out = np.zeros((xyz.shape[0], 3, L), dtype=np.int16)
    for r in range(xyz.shape[0]):
        out[r] = kernel_layout(xyz[r], int(n_real[r]), features[int(row_mol[r])], L)
    return out


def read_corpus(root, *, merged: bool = True) -> CorpusRows:
    from roshambo3.format.r3g.reader import open_corpus, verify
    root = str(root)
    c = open_corpus(root, purpose="inspect")
    if not c.scanned:
        verify(root, "scan")
        c = open_corpus(root, purpose="inspect")
    groups = []
    for gi in range(len(c.group_table)):
        with c.group(gi) as g:
            runs = np.array(g.runs)
            n_conf = runs["n_conf"].astype(np.int64)
            start = np.concatenate([[0], np.cumsum(n_conf)])
            row_mol = np.repeat(np.arange(runs.size), n_conf)
            feats = []
            table = {}
            if g.n_types:
                fo, ft = np.array(g.feature_offsets), np.array(g.feature_types)
                fao, fa = np.array(g.feature_atom_offsets), np.array(g.feature_atoms)
                table = {"feature_offsets": fo, "feature_types": ft,
                         "feature_atom_offsets": fao, "feature_atoms": fa}
                for m in range(runs.size):
                    feats.append([(int(ft[f]), tuple(int(a) for a in fa[fao[f]:fao[f + 1]]))
                                  for f in range(int(fo[m]), int(fo[m + 1]))])
            else:
                feats = [[] for _ in range(runs.size)]
            xyz = g.decode_all()
            n_real = np.array(g.n_real)
            groups.append(GroupRows(
                index=gi, bucket=g.bucket, group=g.group,
                path=os.path.join(root, "groups", group_file_name(g.bucket, g.group)),
                L=g.L, L_heavy=g.L_heavy, L_field_width=g.L_field_width,
                n_conf=g.n_conf, n_mol=g.n_mol, blocks=np.array(g.blocks), runs=runs,
                features=feats, feature_table=table, n_real=n_real,
                mol_id=np.repeat(runs["mol_id"].astype(np.int64), n_conf),
                conf_idx=np.arange(g.n_conf) - start[row_mol],
                vaa=np.array(g.vaa),
                color_self=None if g.color_self is None else np.array(g.color_self),
                counts=np.array(g.counts), xyz=xyz,
                merged=(merged_layout(xyz, n_real, feats, row_mol, g.L) if merged
                        else None),
                header=dict(g.header)))
    import pyarrow.parquet as pq
    table = pq.read_table(os.path.join(root, "molecules.parquet"))
    molecules = {name: (table.column(name).to_numpy() if name not in ("ext_id", "smiles")
                        else table.column(name).to_pylist()) for name in table.column_names}
    build = None
    if os.path.exists(os.path.join(root, "build.json")):
        with open(os.path.join(root, "build.json"), "rb") as f:
            build = json.load(f)
    return CorpusRows(root, c.descriptor, build, groups, molecules)


def corpus_files(root) -> dict:
    """``{relative path: bytes}`` of every file under a corpus directory."""
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            with open(p, "rb") as f:
                out[os.path.relpath(p, root)] = f.read()
    return out

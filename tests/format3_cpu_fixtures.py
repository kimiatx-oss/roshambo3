"""Writer-built CPU fixtures, independent of the legacy builder fixtures."""
from pathlib import Path

import numpy as np

from roshambo3.engine.r3g_source import R3gSource
from roshambo3.format.r3g.descriptor import generator_from_descriptor
from roshambo3.format.r3g.reader import open_corpus, verify
from roshambo3.format.r3g.writer import Molecule, write_corpus
from roshambo3.pharmacophore import RDKitPharmacophoreGenerator
from roshambo3.prep.conformer import stored_self_volume
from roshambo3.reference.overlap import color_overlap


class PhysicalCPUFixtureSource(R3gSource):
    """Real format-3 source wired to the authenticated CPU block path."""

    def __init__(self, corpus, *, color=False, group_indices=None):
        generator = generator_from_descriptor(corpus.color) if color else None
        super().__init__(corpus, groups=group_indices, tile=False,
                         color=color, color_generator=generator)


def make_cpu_corpus(path, *, color=False, block_rows=4, with_color_self=True,
                    sizes=(2, 3, 1, 2), n_real=(2, 2, 2, 2), tied=False):
    generator = RDKitPharmacophoreGenerator() if color else None
    if color:
        for matrix in (generator.interaction_matrix_r, generator.interaction_matrix_p):
            matrix[:] = np.eye(len(matrix))
            matrix[0, 0] = 0
    molecules = []
    for i, (rows, atoms) in enumerate(zip(sizes, n_real)):
        xyz = np.zeros((rows, 3, atoms), np.int16)
        xyz[:, 0] = np.arange(atoms) * 200
        if not tied:
            xyz[:, 1] = np.arange(atoms) * (31 + i * 9)
        coords = xyz.transpose(0, 2, 1).astype(np.float32) * np.float32(2 ** -8)
        vaa = np.asarray([stored_self_volume(row) for row in coords], np.float32)
        features = [(1, (0,))] if color else []
        color_self = None
        if color:
            color_self = np.asarray([color_overlap(row[:1], np.asarray([1]), row[:1],
                np.asarray([1]), generator.interaction_matrix_r,
                generator.interaction_matrix_p) for row in coords], np.float64)
        molecules.append(Molecule(100 + i * 7, xyz, vaa, features, color_self,
                                  ext_id=f"mol-{i}", smiles="C" * atoms))
    write_corpus(str(path), molecules, color=generator.descriptor_color() if color else None,
                 prep_version="cpu-fixture", prep_commit=None, block_rows=block_rows,
                 with_color_self=with_color_self if color else False)
    cache = str(Path(path).parent / 'scan-cache')
    verify(str(path), cache_dir=cache)
    return open_corpus(str(path), cache_dir=cache)

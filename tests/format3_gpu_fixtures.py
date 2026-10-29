"""Small prepare-built libraries for format-3 GPU acceptance.

Real RDKit geometries cover one and multiple conformers, multiple physical
blocks and groups, and donor/acceptor/aromatic/multi-atom color features.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from roshambo3.engine import from_arrays
from roshambo3.format import Catalog
from roshambo3.metrics.ranking import canonical_ids
from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep.build import prepare
from tests.prep_fixtures import molecule
from tests.format3_rows import read_corpus


@dataclass
class PreparedGPUFixture:
    corpus: object
    generator: object
    rows: object
    queries: object
    identity: np.ndarray


@pytest.fixture(scope="module", params=[False, True], ids=["shape", "color"])
def prepared_gpu_corpus(tmp_path_factory, request):
    color = request.param
    if isinstance(color, str):
        from pathlib import Path
        import os
        from roshambo3.format.r3g.reader import open_corpus, verify
        from roshambo3.format.r3g.descriptor import generator_from_descriptor
        from roshambo3.paths import corpora_root
        if color == 'rebuilt-smoke':
            variable, default = 'ROSHAMBO3_FORMAT3_D0', 'D0-SMOKE-2K'
        elif color == 'rebuilt-pur2':
            variable, default = 'ROSHAMBO3_FORMAT3_PUR2', 'dudez-PUR2-color'
        else:
            raise ValueError(color)
        root = Path(os.environ.get(variable, str(corpora_root() / 'format3' / default)))
        assert (root / 'corpus.json').exists(), f'original-input rebuild required: {root}'
        verify(str(root), 'scan')
        corpus = open_corpus(str(root))
        rows = read_corpus(corpus.root)
        generator = generator_from_descriptor(corpus.color) if corpus.has_color else None
        gids = np.array([10, 900], np.int64)
        coords, colors = [], []
        catalog = Catalog(corpus)
        for gid in gids:
            group_i, local = catalog._split(np.asarray([gid]))
            group, row = rows.groups[int(group_i[0])], int(local[0])
            nr = int(group.n_real[row])
            coords.append(group.merged[row, :, :nr].T.astype(np.float32) / 256)
            types = np.repeat(np.arange(1, group.counts.shape[1]), group.counts[row, 1:])
            colors.append((group.merged[row, :, nr:nr + len(types)].T.astype(np.float32) / 256,
                           types.astype(np.int32)))
        queries = from_arrays(coords, ['original-10', 'original-900'],
                              color_generator=generator, color=colors if generator else None)
        return PreparedGPUFixture(corpus, generator, rows, queries,
                                  canonical_ids(rows.mol_id, rows.conf_idx))
    root = tmp_path_factory.mktemp("gpu-prepared")
    generator = PharmacophoreGenerator() if color else None
    smiles = ["CCO", "CCCN", "c1ccccc1O", "CC(=O)O", "CCOC", "CCNCC"]
    molecules = [molecule(smi, f"GPU-{i:03d}", conformers=1 + i % 3, seed=294 + i)
                 for i, smi in enumerate(smiles * 3)]
    corpus = prepare(molecules, str(root / "corpus"), workers=1,
                     color_generator=generator, block_rows=5, max_conf_per_group=12)
    rows = read_corpus(corpus.root)
    assert set(np.unique(rows.conf_idx)) >= {0, 1, 2}
    assert len(rows.groups) > 1
    assert sum(len(g.blocks) for g in rows.groups) > len(rows.groups)
    ids = np.array([0, corpus.n_conf // 2], np.int64)
    # Query color positions come from the same scored-coordinate layout.
    coords, colors = [], []
    for gid in ids:
        group_i, local = Catalog(corpus)._split(np.asarray([gid]))
        group, row = rows.groups[int(group_i[0])], int(local[0])
        nr = int(group.n_real[row])
        coords.append(group.merged[row, :, :nr].T.astype(np.float32) / 256)
        types = np.repeat(np.arange(1, group.counts.shape[1]), group.counts[row, 1:])
        colors.append((group.merged[row, :, nr:nr + len(types)].T.astype(np.float32) / 256,
                       types.astype(np.int32)))
    queries = from_arrays(coords, ["first", "middle"], color_generator=generator,
                          color=colors if color else None)
    return PreparedGPUFixture(corpus, generator, rows, queries,
                              canonical_ids(rows.mol_id, rows.conf_idx))

"""All shipped modes can consume a color corpus with the cells the build compiles."""
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from roshambo3.backend import ColorLoop, ColorPlacement, CoordResidency, PoseSearch, UpdateRule
from roshambo3.engine import Engine, R3gSource, ScreenConfig, from_rdkit
from roshambo3.engine.config import unsupported_cells
from roshambo3.layout import check_color_layout
from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep import prepare

@pytest.fixture
def color_corpus(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1OCCN"))
    assert AllChem.EmbedMolecule(mol, randomSeed=13) == 0
    mol.SetProp("_Name", "color-molecule")
    generator = PharmacophoreGenerator()
    corpus = prepare([mol], str(tmp_path / "corpus"), workers=1, color_generator=generator)
    return corpus, mol, generator


@pytest.mark.parametrize("mode", ["shape", "rescore", "combo"])
def test_real_color_corpus_crosses_engine_boundary_in_all_modes(color_corpus, mode):
    corpus, mol, generator = color_corpus
    color = mode != "shape"
    config = ScreenConfig(color=color, mixing=0.5 if color else 0.0,
                          color_placement=ColorPlacement.IN_GRADIENT if mode == "combo" else ColorPlacement.AT_POSE)
    queries = from_rdkit([mol], color_generator=generator if color else None)
    seen = []
    def score_pairs(**kw):
        expected_width = 7 if color else 1
        assert kw["library_counts"].shape[1] == expected_width
        assert kw["interaction_r"].shape == (expected_width, expected_width)
        check_color_layout(kw["interaction_r"], kw["interaction_p"], kw["query_types"],
                           kw["query_n_atoms"], kw["query_coords"].shape[1],
                           kw["library_counts"], kw["library_coords"].shape[1])
        seen.append(kw)
        # The engine launches dense -- no pair list.
        assert kw["pair_query_row"] is None and kw["pair_library_row"] is None
        return np.zeros((kw["query_coords"].shape[0]
                          * kw["library_counts"].shape[0], 20), np.float32)
    class Sink:
        def prepare(self, *args): pass
        def add_block(self, *args): pass
        def finalize(self): return "done"
    backend = SimpleNamespace(score_pairs=score_pairs, available=lambda: False)
    engine = Engine(pinned=False, backend=backend)
    stats, result = engine.run(R3gSource(corpus, color=color), queries, config, Sink())
    assert result == "done" and stats.n_conformers == 1 and len(seen) == 1
    from roshambo3.format.r3g.descriptor import generator_from_descriptor
    color_source = R3gSource(corpus, color=True,
        color_generator=generator_from_descriptor(corpus.color), tile=False,
        workers=1, prefetch_depth=0)
    descriptor = color_source.cpu_block_descriptors()[0]
    assert color_source.decode_cpu_block(descriptor).result.counts.shape[1] == 7


@pytest.mark.parametrize("mode", ["shape", "rescore", "combo"])
@pytest.mark.parametrize("width", [16, 40, 64, 136])
def test_shipped_mode_is_supported(mode, width):
    config = ScreenConfig(color=mode != "shape", mixing=0.5,
                          color_placement=ColorPlacement.IN_GRADIENT if mode == "combo" else ColorPlacement.AT_POSE)
    resolved = config.for_width(width, query_atoms=32, n_features=7 if mode != "shape" else 1)
    assert unsupported_cells(resolved, bucket=width) == []


@pytest.mark.parametrize("change", [
    {"color_stage": 32}, {"lanes_per_group": 2}, {"start_groups_per_row": 2},
    {"stream_min_blocks_per_sm": 3}, {"coord_residency": CoordResidency.GLOBAL},
    {"pose_search": PoseSearch.ANALYTIC}, {"interleave": 1},
    {"color_placement": ColorPlacement.AT_POSE},
])
def test_combo_exemption_does_not_admit_other_schedules(change):
    resolved = ScreenConfig(color=True, color_placement=ColorPlacement.IN_GRADIENT).for_width(40)
    resolved.update(change)
    assert unsupported_cells(resolved)


@pytest.mark.parametrize("start_mode", [0, 1, 2])
@pytest.mark.parametrize("update_rule", [UpdateRule.ADAGRAD_QADD, UpdateRule.ADAGRAD_EXP])
def test_all_shipped_combo_start_and_update_cells_are_supported(start_mode, update_rule):
    resolved = ScreenConfig(color=True, color_placement=ColorPlacement.IN_GRADIENT,
                            start_mode=start_mode, update_rule=update_rule).for_width(40)
    assert unsupported_cells(resolved) == []


def test_explicit_shipped_combo_schedule_is_supported():
    resolved = ScreenConfig(color=True, color_placement=ColorPlacement.IN_GRADIENT,
                            start_groups_per_row=4, stream_min_blocks_per_sm=2,
                            color_loop=ColorLoop.SAME_TYPE_NOW).for_width(40)
    assert unsupported_cells(resolved) == []


def test_shape_source_reads_only_real_atom_counts(color_corpus):
    corpus, _, generator = color_corpus
    shape_source = R3gSource(corpus, color=False, tile=False,
                             workers=1, prefetch_depth=0)
    color_source = R3gSource(corpus, color=True, color_generator=generator,
                             tile=False, workers=1, prefetch_depth=0)
    shape_descriptors = {(d.group_index, d.block_index): d
                         for d in shape_source.cpu_block_descriptors()}
    color_descriptors = {(d.group_index, d.block_index): d
                         for d in color_source.cpu_block_descriptors()}
    assert shape_descriptors.keys() == color_descriptors.keys()
    key = next(iter(shape_descriptors))
    shape = shape_source.decode_cpu_block(shape_descriptors[key]).result
    color = color_source.decode_cpu_block(color_descriptors[key]).result
    assert shape.counts.flags.c_contiguous
    assert shape.counts.shape[1] == 1
    assert color.counts.shape[1] == 1 + len(generator.FEATURES_ENUM)
    np.testing.assert_array_equal(shape.counts, color.counts[:, :1])

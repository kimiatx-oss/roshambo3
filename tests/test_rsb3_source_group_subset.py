"""A dense-route source over a group subset keeps the corpus's global ids.

Extracted from the reference-pipeline tests. The worker route's descriptors are covered by
test_cpu_worker_owned_scan; this is the dense route's `R3gSource` job list
and native GPU tile decode.
"""
import numpy as np

from roshambo3.engine import R3gSource
from tests.codec_fixtures import make_codec_fixture
from roshambo3.pharmacophore import PharmacophoreGenerator


def test_a_selected_merged_color_group_decodes_without_compacting_gids(tmp_path):
    fx = make_codec_fixture(tmp_path / "merged", shuffle=False,
                            color_layout="merged", block_conf=3)
    source = R3gSource(fx.corpus, groups=[1], tile=True, color=True,
                        color_generator=PharmacophoreGenerator(), workers=1,
                        prefetch_depth=0)
    assert len(source.jobs) == 1
    job = source.jobs[0]
    # Group 1 starts at global conformer 4 (group 0 holds block_conf + 1).
    assert (job.group_index, job.block0, job.row0, job.gid_base,
            job.n_conf, job.n_blocks) == (1, 0, 0, 4, 2, 1)
    block = source._decode(job)
    assert block.tile_mode == 1
    assert block.gid_base == 4
    assert block.color_offset == -1
    np.testing.assert_array_equal(block.mol_id, [17, 17])
    np.testing.assert_array_equal(block.counts, fx.counts[4:6])
    np.testing.assert_array_equal(
        block.coords.reshape(-1, block.L, 32, 3).transpose(0, 2, 1, 3)
        .reshape(-1, block.L, 3)[:2], fx.shape_xyz_i16[4:6].transpose(0, 2, 1))

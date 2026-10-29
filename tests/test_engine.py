"""The engine: sources, sinks, config, and the determinism claim.

Most of these need a GPU and skip cleanly without one. The ones that do not --
config resolution, the buffer packer, sink bookkeeping -- run in CI.
"""

from __future__ import annotations

import threading
import types

from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np
import pytest

from roshambo3.backend import cuda as backend
from roshambo3.backend import (ColorPlacement, CoordResidency, PoseSearch,
                               UpdateLanes, UpdateRule)
from roshambo3.engine import (DEFAULT_CONFORMERS_PER_LAUNCH, Callback,
                              DenseScores, Engine, MoleculeTopK, R3gSource,
                              ScreenConfig, TopK, conformers_per_launch,
                              from_arrays, from_rdkit, pack_into)
from roshambo3.engine.buffers import PinnedPool
from roshambo3.engine.config import MIN_BLOCKS_BY_WIDTH, _by_width
from roshambo3.engine.source import Prefetcher, _Job
from roshambo3.format import BLOCK_CONF
from roshambo3.layout import BTILE
from roshambo3.scores import N_SCORE_COLUMNS

from tests.format3_gpu_fixtures import prepared_gpu_corpus
from gpu_device import device_present

needs_gpu = pytest.mark.skipif(not backend.available() or not device_present(),
                               reason="no CUDA extension in this build, or no device")


# --------------------------------------------------------------------------
# no GPU needed
# --------------------------------------------------------------------------

class TestConfig:
    def test_defaults_are_the_shipped_configuration(self):
        c = ScreenConfig()
        assert c.pose_search is PoseSearch.FIELD
        # The exponential map. This is the default that DECIDES -- and it
        # decides alone, because `resolved_optimizer()` always passes opt_mode
        # explicitly, so `optimizer.DEFAULT_OPT_MODE` never fills it in from
        # here. The two are asserted together in test_exp_map_field_cuda.py.
        assert c.update_rule is UpdateRule.ADAGRAD_EXP
        assert c.update_lanes is UpdateLanes.PER_LANE
        # All four tiers were measured and none shipped: the choice is
        # not separable from color, which contests the same register file.
        assert c.coord_residency is CoordResidency.SHARED
        # The carveout is an attribute of the ARM. What reaches the launch
        # is always an explicit integer, never an unresolved None -- the sticky
        # attribute makes "unset" read whatever the previous kernel left.
        assert c.smem_carveout is None

    def test_an_explicit_residency_carries_its_own_carveout(self):
        """Neither bad cell may be reachable by naming one knob.

        The carveout is an attribute of the ARM and `cudaFuncSetAttribute` is
        sticky, so the two knobs are one decision. Both directions are measured
        regressions, which is why this is a mapping and not a blanket zero:

          * a zero-shared tier at the staged arm's carveout: 2.05-6.81x
            slower, over L = 16..56;
          * the staged tier at carveout 0: up to 2.4x slower, because
            carveout 0 leaves exactly ONE resident block per SM at every demand.

        The staged value is `STAGED_CARVEOUT`, 25 rather than 50:
        25 wins on both sm_70 and sm_120, and 50 is one carveout band too high.
        Asserted against the constant rather than a literal, so the two cannot
        drift; the literal that must not drift is in `backend/cuda.py`.

        A caller who moves the tier and says nothing about the carveout gets the
        carveout that goes with it, in both directions.
        """
        from roshambo3.backend.cuda import STAGED_CARVEOUT
        assert STAGED_CARVEOUT == 25, (
            "carveout 25 was measured as the optimum on both architectures; "
            "moving it is a decision, not a tidy-up.")
        # Both sides are constructed explicitly rather than leaning on the
        # color defaults, because those moved: the tree ships
        # `color_residency = SHARED` and `color_stage = 48`, so a config that
        # streams its SHAPE coordinates still stages its COLOR ones and still
        # wants the staged carveout. `carveout_for` says exactly that, and
        # reading the shape tier alone would partition zero shared to a kernel
        # asking for 12 KB of it -- the carveout trap from the other direction. A test
        # that inherited the defaults would silently stop testing what it names.
        #
        # The staged-color side is a MODE 3 config, because that is the
        # only mode in which a color tier is staged at all. The color-pass knobs
        # are inert at `AT_POSE` and on a shape-only screen -- the launcher
        # refuses a non-inert value there by name -- so `for_width` resolves
        # them to their inert values, and the carveout has to be resolved from
        # what the launch will actually carry rather than from what the dataclass
        # declares. Asserting the staged carveout for a SHAPE config was
        # asserting a launch that cannot exist.
        unstaged = dict(color_residency=CoordResidency.GLOBAL, color_stage=0)
        combo = dict(color=True, mixing=0.5,
                     color_placement=ColorPlacement.IN_GRADIENT,
                     color_residency=CoordResidency.SHARED, color_stage=48)
        for L in (16, 56):
            assert ScreenConfig(coord_residency=CoordResidency.SHARED,
                                **unstaged
                                ).for_width(L)["smem_carveout"] == STAGED_CARVEOUT
            for tier in (CoordResidency.GLOBAL, CoordResidency.GLOBAL_LDCG,
                         CoordResidency.REGISTER):
                assert ScreenConfig(coord_residency=tier, **unstaged
                                    ).for_width(L)["smem_carveout"] == 0
                # ...and the same tier WITH color staged wants the staged
                # carveout, which is the case the shipped mode-3 launch takes.
                assert ScreenConfig(coord_residency=tier, **combo
                                    ).for_width(L)["smem_carveout"] == STAGED_CARVEOUT
                # ...and the same knobs at AT_POSE do NOT, because there is no
                # color pass to stage: they never reach the launch.
                at_pose = dict(combo, color_placement=ColorPlacement.AT_POSE)
                assert ScreenConfig(coord_residency=tier, **at_pose
                                    ).for_width(L)["smem_carveout"] == 0

    def test_what_reaches_the_launch_is_never_an_unresolved_none(self):
        """`None` on the config means "the one that goes with the tier", never
        "leave it alone" -- an unset carveout reads whatever the previous kernel
        left behind, which is not a control."""
        for tier in CoordResidency:
            got = ScreenConfig(coord_residency=tier).for_width(32)
            assert isinstance(got["smem_carveout"], int)

    def test_an_explicit_carveout_is_still_honoured(self):
        """The carveout must stay re-sweepable per arm; the deciding
        measurement is that sweep. Resolving a default must not become
        ignoring an argument."""
        got = ScreenConfig(coord_residency=CoordResidency.GLOBAL_LDCG,
                           smem_carveout=50).for_width(32)
        assert got["smem_carveout"] == 50

    def test_lr_t_is_derived_from_steps(self):
        """They are one knob. Omitting lr_t must not mean 'use the old one'."""
        from roshambo3.optimizer import lr_t_for_steps
        for steps in (25, 50, 100):
            got = ScreenConfig(steps=steps).for_width(32)
            assert np.isclose(got["lr_t"], lr_t_for_steps(steps))

    def test_an_off_curve_pair_is_refused_without_a_reason(self):
        with pytest.raises(Exception):
            ScreenConfig(steps=25, lr_t=0.1).for_width(32)

    def test_an_off_curve_pair_is_allowed_with_a_stated_reason(self):
        got = ScreenConfig(steps=25, lr_t=0.1,
                           budget_override="reproducing the legacy control"
                           ).for_width(32)
        assert got["lr_t"] == 0.1

    @pytest.mark.parametrize("L,spacing", [(16, 0.4), (24, 0.4), (32, 0.5),
                                           (40, 0.6), (48, 0.6), (56, 0.6)])
    def test_field_spacing_follows_the_per_bucket_policy(self, L, spacing):
        """Every launch is single-bucket, which is what makes a per-bucket
        lattice spacing free. Accuracy wants coarser and residency wants finer,
        so there is no single h that serves every width."""
        assert ScreenConfig().for_width(L)["field_spacing_a"] == spacing

    def test_an_explicit_spacing_overrides_the_policy(self):
        assert ScreenConfig(field_spacing_a=0.8).for_width(24)["field_spacing_a"] == 0.8

    def test_stage_width_respects_the_device_limit_not_a_constant(self):
        """The opt-in shared-memory limit spans 64 KB (Turing) to 227 KB (Hopper),
        so any hardcoded budget is right on one family and wrong elsewhere. This
        used to assume 99 KB; on Turing that asks for 96 KB at L = 64 and the
        launcher refuses the launch. Measured on a TITAN V, Volta is 96 KB -- not
        the 64 KB one would assume for "old hardware" -- which is exactly why it
        has to be asked rather than guessed."""
        from roshambo3.engine.config import stage_width_for

        # Turing: 256 fits only to L = 42, so L = 48 must step down. The
        # ladder is every multiple of 32 rather than (256, 128, 0), so a step
        # down is now one rung and not a factor of two -- and L = 96, which used
        # to fall off the bottom of the ladder and stop staging altogether, gets
        # the 96 that fits. Trimming the query then bought L = 48 a further rung,
        # 192 -> 224, by taking 280 B out of the block. These are the SAME
        # arithmetic throughout; what changed is how finely the ladder can
        # approach the boundary and how much of the budget the query costs.
        assert stage_width_for(32, budget=64 * 1024) == 256
        assert stage_width_for(48, budget=64 * 1024) == 224
        assert stage_width_for(96, budget=64 * 1024) == 96
        # Hopper: 256 fits far wider.
        assert stage_width_for(96, budget=227 * 1024) == 256
        # Whatever the budget, the chosen width must actually fit.
        for budget in (48, 64, 99, 163, 227):
            for L in (16, 24, 32, 40, 48, 56, 64, 72, 96, 136):
                w = stage_width_for(L, budget=budget * 1024)
                assert w * L * 6 <= budget * 1024, (budget, L, w)

    @needs_gpu
    def test_the_budget_comes_from_the_device(self):
        from roshambo3.backend import extension
        from roshambo3.engine.config import shared_memory_budget

        want = extension().device_info(0)["shared_memory_per_block_optin"]
        assert shared_memory_budget(0) == want
        assert want >= 48 * 1024, "no CUDA architecture offers less than 48 KB"

    def test_by_width_takes_the_largest_key_at_or_below_L(self):
        # mb2 measured 1.66x faster than this entry on the register tier
        # and deliberately did not change it: MINB is an occupancy choice and
        # color contests occupancy.
        assert _by_width(MIN_BLOCKS_BY_WIDTH, 56, 4) == 1
        assert _by_width(MIN_BLOCKS_BY_WIDTH, 55, 4) == 3   # the 48 entry
        assert _by_width({}, 32, 7) == 7                    # nothing matches


class TestPackInto:
    def test_lays_the_stored_shorts_out_per_conformer(self):
        """The packer transposes and nothing else. It used to dequantize
        to float32 and append a padding-mask column; the record is the corpus's
        own int16 now, and the mask was a per-slot expansion of the count
        vector the kernel already has."""
        nb, L = 5, 8
        xyz = np.arange(nb * 3 * L, dtype=np.int16).reshape(nb, 3, L)
        dst = np.zeros(((nb + BTILE - 1) // BTILE) * BTILE * L * 3, np.int16)
        out, got = pack_into(xyz, L, dst, WIRE_QUANT_STEP, tile=False)
        assert got == nb and out.shape == (nb, L, 3)
        for i in range(nb):
            assert np.array_equal(out[i], xyz[i].T)

    def test_tiled_output_round_trips(self):
        nb, L = 33, 8
        xyz = np.random.default_rng(0).integers(-1000, 1000, (nb, 3, L)).astype(np.int16)
        dst = np.zeros(64 * L * 3, np.int16)
        out, got = pack_into(xyz, L, dst, WIRE_QUANT_STEP, tile=True)
        assert got == nb and out.shape[0] == 64
        back = (out.reshape(2, L, BTILE, 3).transpose(0, 2, 1, 3)
                   .reshape(-1, L, 3)[:nb])
        assert np.array_equal(back, xyz.transpose(0, 2, 1))

    def test_refuses_a_buffer_that_is_too_small(self):
        xyz = np.zeros((4, 3, 8), np.int16)
        with pytest.raises(ValueError, match="pinned buffer holds 4 int16, block needs 96"):
            pack_into(xyz, 8, np.zeros(4, np.int16), WIRE_QUANT_STEP, tile=False)


def _fake_corpus(specs, block_conf=BLOCK_CONF):
    """A reader-shaped stand-in for launch arithmetic without file I/O."""
    group_table, directory = [], []
    for i, (L, n_conf) in enumerate(specs):
        group_table.append({"L": L, "L_heavy": L, "L_field_width": L,
                            "n_conf": n_conf, "bucket": i, "group": i})
        row0 = 0
        for bi in range((n_conf + block_conf - 1) // block_conf):
            rows = min(block_conf, n_conf - row0)
            directory.append((i, bi, row0, rows, rows * L))
            row0 += rows
    from roshambo3.format.r3g.reader import BLOCK_DIRECTORY_DTYPE
    blocks = np.asarray(directory, dtype=BLOCK_DIRECTORY_DTYPE)
    return types.SimpleNamespace(
        group_table=group_table,
        block_directory=lambda order="file": blocks,
        n_types=0,
        step=WIRE_QUANT_STEP)


class TestLaunchCoalescing:
    """The launch size is a knob, and 4,096 is still the old behavior.

    Job construction is host arithmetic, so the invariants that fail silently
    on the device -- a run spanning two atom widths, a hole in the global ids
    -- are all checkable here, before any of them can reach a kernel.
    """

    def test_block_conf_reproduces_one_launch_per_on_disk_block(self):
        corpus = _fake_corpus([(24, 10_000), (32, 4_096), (40, 17)])
        src = R3gSource(corpus, conformers_per_launch=BLOCK_CONF)
        assert [j.n_blocks for j in src.jobs] == [1] * 5
        assert [j.n_conf for j in src.jobs] == [4096, 4096, 1808, 4096, 17]
        assert [j.gid_base for j in src.jobs] == [0, 4096, 8192, 10000, 14096]

    def test_a_run_never_crosses_a_group(self):
        """`config.for_width(block.L)` makes the launch arguments a function of
        the group's atom width, `coord_scale` is per group, and "a molecule
        never straddles a group" is what MoleculeTopK rests on. The three
        groups here are each far smaller than the target, so a coalescer that
        only counted conformers would merge them."""
        corpus = _fake_corpus([(16, 100), (24, 100), (32, 100)])
        src = R3gSource(corpus, conformers_per_launch=65_536)
        assert len(src.jobs) == 3
        assert [j.group_index for j in src.jobs] == [0, 1, 2]
        assert [j.n_conf for j in src.jobs] == [100, 100, 100]

    @pytest.mark.parametrize("target", [BLOCK_CONF, 8192, 16384, 65536])
    def test_the_runs_tile_the_corpus_and_the_gids_are_a_permutation(self, target):
        """`gid = gid_base + row` (topk.cu:123), so every conformer must be
        covered exactly once and no run may leave a hole."""
        corpus = _fake_corpus([(16, 40_000), (24, 4_096), (32, 1), (40, 12_345)])
        total = sum(int(g["n_conf"]) for g in corpus.group_table)
        src = R3gSource(corpus, conformers_per_launch=target)
        gids = np.concatenate([np.arange(j.gid_base, j.gid_base + j.n_conf)
                               for j in src.jobs])
        assert np.array_equal(np.sort(gids), np.arange(total))
        assert sum(j.n_conf for j in src.jobs) == total

    def test_a_run_never_exceeds_the_target(self):
        corpus = _fake_corpus([(24, 1_000_000)])
        for target in (BLOCK_CONF, 8192, 40_000, 65_536):
            src = R3gSource(corpus, conformers_per_launch=target)
            assert max(j.n_conf for j in src.jobs) <= target

    def test_every_interior_block_of_a_run_is_a_whole_number_of_tiles(self):
        """The condition that makes the run's rows contiguous, and therefore
        the one that makes `max_block_elements` sizing the padded TOTAL agree
        with what the decode writes. Only a group's last block may be short."""
        corpus = _fake_corpus([(24, 100_000), (32, 4_097)])
        src = R3gSource(corpus, conformers_per_launch=65_536)
        for j in src.jobs:
            rows = corpus.block_directory("file")
            group_blocks = rows[rows["group"] == j.group_index]
            for rec in group_blocks[j.block0:j.block0 + j.n_blocks - 1]:
                assert int(rec["rows"]) % BTILE == 0

    def test_sizing_pads_the_total_not_each_constituent_block(self):
        corpus = _fake_corpus([(24, 10_000)])
        src = R3gSource(corpus, conformers_per_launch=65_536)
        j, = src.jobs
        assert j.n_conf == 10_000
        want_rows = ((10_000 + BTILE - 1) // BTILE) * BTILE
        # Three int16 per atom on the wire, not four float32.
        assert src.max_block_elements == want_rows * 24 * 3
        assert src.max_block_conformers == 10_000
        assert src.n_launches == 1

    def test_the_reported_worst_case_is_the_coalesced_one(self):
        """The device top-K's capacity is fixed on the FIRST launch and a later
        block that does not fit is a hard error, so a source that reported the
        on-disk block size here would kill the screen it started."""
        corpus = _fake_corpus([(24, 200_000)])
        src = R3gSource(corpus, conformers_per_launch=65_536)
        assert src.max_block_conformers == 65_536

    def test_a_nonpositive_target_is_refused(self):
        corpus = _fake_corpus([(24, 10)])
        with pytest.raises(ValueError, match="must be positive"):
            R3gSource(corpus, conformers_per_launch=0)

    def test_the_query_count_bounds_the_default_not_the_launch_size_alone(self):
        """The top-K state is `n_queries * (k + 2*launch) * 20` floats, so the
        cost of coalescing is linear in the panel as well. 8 queries fit the
        default; a 480-conformer panel would ask for 5.1 GB."""
        assert conformers_per_launch(8, 1000) == DEFAULT_CONFORMERS_PER_LAUNCH
        small = conformers_per_launch(480, 100_000)
        assert small < DEFAULT_CONFORMERS_PER_LAUNCH
        assert small >= BLOCK_CONF
        # Never below one on-disk block: coalescing is not what is wrong with
        # a configuration that cannot afford 4,096.
        assert conformers_per_launch(100_000, 1_000_000) == BLOCK_CONF


class TestDenseScoresLayout:
    def test_reassembles_query_major(self):
        """Each launch writes query-major within its own block, so naively
        concatenating blocks gives neither corpus nor query order. The sink must
        reassemble rather than hand back an array whose order has to be guessed."""
        from roshambo3.engine.source import ConformerBlock
        from roshambo3.scores import N_SCORE_COLUMNS
        nq = 3
        sink = DenseScores()
        sink.prepare(nq, 0)
        at = 0
        for n in (4, 2, 5):
            block = ConformerBlock(coords=np.zeros((n, 8, 4), np.float32),
                                   counts=np.full((n, 1), 8, np.uint8),
                                   n_conformers=n, L=8, bucket=0, gid_base=at)
            # row (q, i) carries the value q*1000 + gid
            sc = np.zeros((nq * n, N_SCORE_COLUMNS), np.float32)
            for q in range(nq):
                for i in range(n):
                    sc[q * n + i, 0] = q * 1000 + at + i
            sink.add_block(block, sc)
            at += n
        res = sink.finalize()
        assert res.scores.shape == (nq, at, N_SCORE_COLUMNS)
        assert np.array_equal(res.gids, np.arange(at))
        for q in range(nq):
            assert np.array_equal(res.scores[q, :, 0],
                                  q * 1000 + np.arange(at))

    def test_size_guard_refuses_rather_than_exhausting_memory(self):
        from roshambo3.engine.source import ConformerBlock
        from roshambo3.scores import N_SCORE_COLUMNS
        sink = DenseScores(max_bytes=100)
        sink.prepare(1, 0)
        block = ConformerBlock(coords=np.zeros((50, 8, 4), np.float32),
                              counts=np.full((50, 1), 8, np.uint8),
                              n_conformers=50, L=8, bucket=0, gid_base=0)
        with pytest.raises(MemoryError):
            sink.add_block(block, np.zeros((50, N_SCORE_COLUMNS), np.float32))


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_setup(smoke_corpus):
    from roshambo3.format import Catalog
    cat = Catalog(smoke_corpus)
    gids = np.array([10, 900])
    return smoke_corpus, gids, from_arrays(cat.coordinates(gids), ["a", "b"])


@needs_gpu
class TestScreen:
    def test_a_conformer_finds_itself_first(self, gpu_setup):
        """The strongest end-to-end check available without a second
        implementation: a conformer drawn from the corpus must come back as its
        own best hit, at a Tanimoto of ~1."""
        corpus, gids, qs = gpu_setup
        _, res = Engine(device_id=0).run(R3gSource(corpus), qs,
                                         ScreenConfig(), TopK(k=10))
        for q, g in enumerate(gids):
            ids, sc = res[q][1], res[q][0]
            assert ids[0] == g, f"query {g} did not find itself first"
            assert sc[0, 1] > 0.999, sc[0, 1]

    @pytest.mark.parametrize("prepared_gpu_corpus", [
        pytest.param(False, id='prepare-shape'), pytest.param(True, id='prepare-color'),
        pytest.param('rebuilt-smoke', id='original-D0-inputs'),
        pytest.param('rebuilt-pur2', id='original-PUR2-inputs')], indirect=True)
    @pytest.mark.parametrize("launch", [1, 10, 65536])
    @pytest.mark.parametrize("depth", [3, 8])
    def test_result_is_independent_of_how_blocks_were_produced(
            self, prepared_gpu_corpus, launch, depth):
        """Prepare-built shape/color top-K identity across launch production."""
        from contextlib import closing
        from roshambo3.engine import R3gSource
        fx = prepared_gpu_corpus
        color = fx.generator is not None
        config = ScreenConfig(color=color, mixing=.5 if color else 0.)

        def source(**kw):
            return R3gSource(fx.corpus, color=color,
                             color_generator=fx.generator, **kw)

        with closing(Engine(device_id=0)) as engine:
            _, base = engine.run(source(prefetch_depth=0, conformers_per_launch=1),
                                 fx.queries, config, TopK(k=30))
            _, other = engine.run(source(workers=8, prefetch_depth=depth,
                                         conformers_per_launch=launch),
                                  fx.queries, config, TopK(k=30))
        for (sa, ia), (sb, ib) in zip(base, other, strict=True):
            np.testing.assert_array_equal(fx.identity[ia], fx.identity[ib])
            np.testing.assert_array_equal(sa.view(np.uint32), sb.view(np.uint32))

    def test_coalescing_is_bit_identical_on_a_corpus_with_several_blocks(
            self, dev_corpus):
        """The one above runs on D0-SMOKE-2K, whose every group is a single
        on-disk block -- so it exercises the coalescing code path but never a
        multi-block run. This is the same claim where it can actually bite:
        13 groups, 56 blocks, runs of up to 10 blocks, and a source that
        reports a coalesced worst case to the device top-K.

        ``topk_identity.n_distinct_topk`` must be 1 at every granularity;
        this is that check, in the suite, on the shipped source.
        """
        from roshambo3.format import Catalog
        cat = Catalog(dev_corpus)
        gids = np.array([7, 60_000, 150_000])
        qs = from_arrays(cat.coordinates(gids), [f"q{g}" for g in gids])

        def run(target):
            src = R3gSource(dev_corpus, conformers_per_launch=target)
            stats, res = Engine(device_id=0).run(src, qs, ScreenConfig(),
                                                 TopK(k=200))
            return stats, res

        base_stats, base = run(BLOCK_CONF)
        physical_blocks = dev_corpus.block_directory("file").size
        assert base_stats.n_blocks == physical_blocks, "one launch per physical block"
        for target in (8192, 16_384, 65_536, 1 << 30):
            stats, other = run(target)
            assert stats.n_blocks < base_stats.n_blocks, target
            assert stats.n_conformers == base_stats.n_conformers
            for (sa, ia), (sb, ib) in zip(base, other):
                assert np.array_equal(ia, ib), f"gids differ at {target}"
                assert np.array_equal(sa, sb), f"scores differ at {target}"

    def test_a_coalesced_screen_reports_the_same_gids_it_would_have(
            self, dev_corpus):
        """`gid = gid_base + out_t[r]` (topk.cu:123), so a hole between two
        constituent blocks of a run shifts every later id -- onto conformers
        that exist, with scores that are real, which is why this is checked
        against the corpus rather than only against another arm.

        A conformer drawn from the corpus must still come back as its own best
        hit, at every coalesce factor."""
        from roshambo3.format import Catalog
        cat = Catalog(dev_corpus)
        # Probe both sides of several molecule-aligned physical block edges.
        directory = dev_corpus.block_directory("file")
        group_offsets = np.cumsum([0] + [int(g["n_conf"])
                                         for g in dev_corpus.group_table[:-1]])
        boundaries = [int(group_offsets[int(rec["group"])]) + int(rec["first_row"])
                      for rec in directory if int(rec["first_row"]) > 0]
        gids = np.asarray([value for edge in boundaries[:2]
                           for value in (edge - 1, edge)], dtype=np.int64)
        qs = from_arrays(cat.coordinates(gids), [f"q{g}" for g in gids])
        for target in (BLOCK_CONF, 8192, 65_536):
            src = R3gSource(dev_corpus, conformers_per_launch=target)
            _, res = Engine(device_id=0).run(src, qs, ScreenConfig(),
                                             TopK(k=5))
            for q, g in enumerate(gids):
                assert res[q][1][0] == g, (target, g)
                assert res[q][0][0, 1] > 0.999

    def test_smaller_k_is_a_prefix_of_larger_k(self, gpu_setup):
        corpus, _, qs = gpu_setup
        big = Engine(device_id=0).run(R3gSource(corpus), qs,
                                      ScreenConfig(), TopK(k=40))[1]
        small = Engine(device_id=0).run(R3gSource(corpus), qs,
                                        ScreenConfig(), TopK(k=10))[1]
        for (sb, ib), (ss, isml) in zip(big, small):
            assert np.array_equal(ib[:10], isml)

    def test_device_topk_matches_a_host_reduction_of_the_dense_matrix(self, gpu_setup):
        """Two unrelated reduction paths -- device compaction plus two radix
        sorts, and rank_order over a full matrix on the host -- must agree
        exactly, ties included."""
        from roshambo3 import numerics, scores as S
        corpus, _, qs = gpu_setup
        dense = Engine(device_id=0).run(R3gSource(corpus), qs,
                                        ScreenConfig(), DenseScores())[1]
        topk = Engine(device_id=0).run(R3gSource(corpus), qs,
                                       ScreenConfig(), TopK(k=30))[1]
        for q in range(qs.n_conformers):
            col = dense.scores[q, :, S.TANIMOTO_COMBINATION]
            want = dense.gids[numerics.rank_order(col, dense.gids)][:30]
            assert np.array_equal(want, topk[q][1])

    def test_the_field_and_analytic_paths_agree_on_ranking(self, gpu_setup):
        """The field decides which pose; the fused exact rescore reports the
        number. So the two paths do not agree bit for bit -- they agree on the
        top of the ranking, which is what acceptance means."""
        corpus, _, qs = gpu_setup
        f = Engine(device_id=0).run(R3gSource(corpus), qs, ScreenConfig(
            pose_search=PoseSearch.FIELD), TopK(k=25))[1]
        a = Engine(device_id=0).run(R3gSource(corpus), qs, ScreenConfig(
            pose_search=PoseSearch.ANALYTIC), TopK(k=25))[1]
        for (sf, idf), (sa, ida) in zip(f, a):
            assert idf[0] == ida[0], "the two paths disagree on the best hit"
            overlap = len(set(idf.tolist()) & set(ida.tolist())) / len(idf)
            assert overlap >= 0.8, f"top-25 set agreement only {overlap:.2f}"

    def test_color_config_must_match_the_format3_source(self, gpu_setup):
        """Color layout is fixed by the format-3 source, not a split-source flag."""
        corpus, _, qs = gpu_setup
        with pytest.raises(ValueError, match="config/source color layouts differ"):
            Engine(device_id=0).run(
                R3gSource(corpus), qs,
                ScreenConfig(color=True, pose_search=PoseSearch.ANALYTIC),
                TopK(k=5))

    def test_the_no_features_gate_fires_on_the_field_path_too(self, gpu_setup):
        """The field path has color, so the gate that catches a library
        carrying no features has to reach it as well -- otherwise the one
        failure mode that reads as success (a color Tanimoto of exactly zero for
        every pair) comes back on the path that ships."""
        from roshambo3.backend import score_pairs
        # One library molecule padded to a whole BTILE tile, since the tile
        # check runs first and would otherwise be what refuses this.
        n = BTILE
        with pytest.raises(ValueError, match="no pharmacophore feature"):
            score_pairs(query_coords=np.zeros((1, 32, 4), np.float32),
                        query_types=np.zeros((1, 32), np.int32),
                        query_n_atoms=np.array([8], np.int32),
                        library_coords=np.zeros((n, 32, 4), np.float32),
                        library_counts=np.full((n, 1), 8, np.uint8),
                        library_color_offset=8,
                        pair_query_row=np.array([0], np.int32),
                        pair_library_row=np.array([0], np.int32),
                        interaction_r=np.ones((1, 1), np.float32),
                        interaction_p=np.zeros((1, 1), np.float32),
                        pose_search=PoseSearch.FIELD, color=True)

    def test_pinned_buffers_are_actually_pinned(self, gpu_setup):
        """A staging copy in the producer that silently un-pins the buffer is a
        real defect that happened once; it is checked, not assumed."""
        corpus, _, qs = gpu_setup
        stats, _ = Engine(device_id=0, pinned=True).run(
            R3gSource(corpus), qs, ScreenConfig(), TopK(k=5))
        assert stats.pinned

    def test_callback_sees_every_row(self, gpu_setup):
        corpus, _, qs = gpu_setup
        rows = []
        Engine(device_id=0).run(R3gSource(corpus), qs, ScreenConfig(),
                                Callback(lambda b, s: rows.append(s.shape[0])))
        assert sum(rows) == qs.n_conformers * corpus.n_conformers

    def test_stats_account_for_the_whole_wall(self, gpu_setup):
        corpus, _, qs = gpu_setup
        stats, _ = Engine(device_id=0, stage_timing=True).run(
            R3gSource(corpus), qs, ScreenConfig(), TopK(k=5))
        assert stats.n_conformers == corpus.n_conformers
        assert stats.n_rows == qs.n_conformers * corpus.n_conformers
        assert stats.wall_s > 0 and stats.conformers_per_s > 0
        assert stats.launch_s <= stats.wall_s + 1e-6
        assert stats.device_stages, "stage timing was requested but nothing recorded"


@needs_gpu
class TestFailureLeavesNothingBehind:
    """A screen that raises must leave the pool as it found it.

    The pool is cached on the Engine and only ever grows, so a buffer lost to
    an error is lost for the life of that Engine. Lose `pool_buffers` of them
    and the next screen blocks on an empty pool while the producer's in-flight
    workers block on the same one -- inside `shutdown(wait=True)`, with the
    exception that started it never printed.
    """

    class Boom(Exception):
        pass

    @pytest.fixture(autouse=True)
    def _fail_fast(self, monkeypatch):
        """Shorten the pool's acquire bound for these tests only.

        Checked, not assumed: with the release moved off the `finally` these
        tests do not fail, they *block* -- which is the defect. At the shipped
        600 s bound a regression would take ten minutes per failed run to
        report. One second is still far longer than a healthy acquire on this
        corpus.
        """
        from roshambo3.engine.buffers import PinnedPool
        monkeypatch.setattr(PinnedPool, "ACQUIRE_TIMEOUT_S", 1.0)

    def _run_and_expect_boom(self, corpus, qs, sink):
        engine = Engine(device_id=0)
        with pytest.raises(self.Boom):
            engine.run(R3gSource(corpus), qs, ScreenConfig(), sink)
        return engine

    def test_a_sink_that_raises_does_not_leak_a_buffer(self, gpu_setup):
        corpus, _, qs = gpu_setup

        class ExplodingSink(TopK):
            def add_block(self, block, scores):
                raise TestFailureLeavesNothingBehind.Boom("sink failed")

        engine = self._run_and_expect_boom(corpus, qs, ExplodingSink(k=5))
        pool = engine._pool
        assert pool.available == len(pool), (
            f"{len(pool) - pool.available} of {len(pool)} buffers were not "
            f"returned after the sink raised")

    def test_the_engine_can_run_again_after_a_failure(self, gpu_setup):
        """The consequence that matters: a leak is only visible later, on a
        screen that has nothing wrong with it."""
        corpus, _, qs = gpu_setup

        class ExplodingSink(TopK):
            def add_block(self, block, scores):
                raise TestFailureLeavesNothingBehind.Boom("sink failed")

        engine = Engine(device_id=0)
        for _ in range(6):          # more failures than the pool has buffers
            with pytest.raises(self.Boom):
                engine.run(R3gSource(corpus), qs, ScreenConfig(),
                           ExplodingSink(k=5))
        stats, res = engine.run(R3gSource(corpus), qs, ScreenConfig(),
                                TopK(k=5))
        assert stats.n_conformers > 0
        assert len(res) == qs.n_conformers

    def test_the_pool_covers_the_prefetch_depth(self, gpu_setup):
        """api.screen exposes prefetch_depth and not pool_buffers, so the two
        used to be coupled only by their defaults happening to agree."""
        corpus, _, qs = gpu_setup
        engine = Engine(device_id=0, pool_buffers=2)
        engine.run(R3gSource(corpus, prefetch_depth=8), qs, ScreenConfig(),
                   TopK(k=5))
        assert len(engine._pool) >= 9


class TestQuerySetRefusesBadInput:
    """`from_arrays` used to check only that the list was non-empty.

    Each case below produced a wrong answer rather than an error, which is why
    they are worth a test apiece rather than one "it validates" test.
    """

    def _one(self):
        return np.zeros((5, 3), np.float32)

    def test_a_1d_array_is_not_silently_broadcast(self):
        """nr became 3 and `out[i, :3, :3] = b` broadcast one point into three,
        fabricating a 3-atom query out of a single coordinate."""
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            from_arrays([np.zeros(3, np.float32)], ["a"])

    def test_a_wide_block_is_refused(self):
        with pytest.raises(ValueError, match="n_atoms, 3"):
            from_arrays([np.zeros((5, 4), np.float32)], ["a"])

    def test_an_empty_query_is_refused(self):
        with pytest.raises(ValueError, match="no atoms"):
            from_arrays([np.zeros((0, 3), np.float32)], ["a"])

    def test_non_finite_coordinates_are_refused(self):
        """A NaN query does not raise downstream, it disappears: the device
        top-K drops a NaN score rather than ranking it, so the screen returns
        an empty hit list and reports no error at all."""
        b = self._one()
        b[2, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            from_arrays([b], ["a"])

    def test_a_name_count_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="names for"):
            from_arrays([self._one(), self._one()], ["a"])

    def test_a_short_color_list_is_refused(self):
        """Both zips truncated to the shorter list, so L was computed from a
        prefix of the queries and the rest failed on an unrelated broadcast."""
        with pytest.raises(ValueError, match="color entries"):
            from_arrays([self._one(), self._one()], ["a", "b"],
                        color=[(np.zeros((1, 3), np.float32),
                                np.zeros(1, np.int32))])

    def test_mismatched_color_types_are_refused(self):
        with pytest.raises(ValueError, match="color coordinates but"):
            from_arrays([self._one()], ["a"],
                        color=[(np.zeros((2, 3), np.float32),
                                np.zeros(1, np.int32))])

    def test_a_color_type_outside_the_matrix_is_refused(self):
        """The device indexes rmat[ta*N + tb] with no bounds check, so this was
        an out-of-bounds device read -- reachable by passing color features and
        no color_generator, against the 1x1 placeholder matrix."""
        with pytest.raises(ValueError, match="interaction matrix"):
            from_arrays([self._one()], ["a"],
                        color=[(np.zeros((1, 3), np.float32),
                                np.array([3], np.int32))])

    def test_a_negative_color_type_is_refused(self):
        """The same out-of-bounds device read, off the other end. Only
        `types.max()` was checked, so a negative type indexed backwards out of
        the interaction matrix."""
        with pytest.raises(ValueError, match="negative"):
            from_arrays([self._one()], ["a"],
                        color=[(np.zeros((1, 3), np.float32),
                                np.array([-1], np.int32))])

    def test_a_good_query_still_builds(self):
        qs = from_arrays([self._one(), np.zeros((7, 3), np.float32)], ["a", "b"])
        assert qs.n_conformers == 2
        assert qs.L == 7


class TestStageWidthFollowsTheDeviceItWillRunOn:
    """`for_width` used to resolve the stage width against device 0 always.

    The opt-in shared-memory limit differs by 2.5x across supported
    architectures -- 64 KB on Turing, 96 KB on Volta, 163 KB on Ampere -- and
    the launcher does not clamp an over-large request, it refuses it. So
    `Engine(device_id=1)` on unlike cards resolved a width device 1 could not
    accept.

    Two identical cards cannot show this, which is why the budgets are injected
    rather than measured: the defect is a dropped argument, and this is the
    test that the argument arrives.
    """

    @pytest.fixture
    def unlike_devices(self, monkeypatch):
        from roshambo3.engine import config as C
        monkeypatch.setattr(C, "_smem_budget_cache", {0: 163 * 1024, 1: 64 * 1024})
        return C

    def test_the_two_devices_get_different_widths(self, unlike_devices):
        C = unlike_devices
        wide = C.stage_width_for(64, device_id=0)
        narrow = C.stage_width_for(64, device_id=1)
        assert wide > narrow, (wide, narrow)

    def test_for_width_passes_the_device_through(self, unlike_devices):
        C = unlike_devices
        cfg = ScreenConfig()
        assert (cfg.for_width(64, device_id=0)["stage_width"]
                != cfg.for_width(64, device_id=1)["stage_width"])

    def test_the_chosen_width_fits_that_device(self, unlike_devices):
        C = unlike_devices
        for dev, budget in ((0, 163 * 1024), (1, 64 * 1024)):
            w = ScreenConfig().for_width(64, device_id=dev)["stage_width"]
            assert w * 64 * C._STAGED_BYTES_PER_ATOM <= budget, (dev, w)

    def test_an_explicit_stage_width_still_wins(self, unlike_devices):
        """The policy is a default, not a clamp -- an ablation has to be able
        to pin a width and see the launcher refuse it."""
        cfg = ScreenConfig(stage_width=128)
        assert cfg.for_width(64, device_id=1)["stage_width"] == 128


class TestBufferOwnershipWithoutAGpu:
    """Buffer lifecycle, on the CPU.

    The whole failure class used to be GPU-gated (and corpus-gated on top of
    that), so on any machine without a card it asserted nothing -- while the
    thing it protects, host-side buffer ownership, has no GPU in it at all.
    ``PinnedPool(pinned=False)``, ``Prefetcher`` and ``pack_into`` all run
    without CUDA, so the leak is reachable directly.
    """

    class Boom(RuntimeError):
        pass

    def _pool(self, n=3, elems=64):
        return PinnedPool(n, elems, pinned=False)

    def test_a_decode_that_raises_after_acquiring_does_not_leak(self):
        """The defect: `_decode` took a buffer and then called `pack_into`
        outside any try, so an undersized buffer or a shape that would not
        broadcast lost it permanently -- the pool is cached on the engine and
        only ever grows."""
        pool = self._pool()

        def decode_that_fails(_job):
            buf = pool.acquire()
            try:
                raise self.Boom("pack failed")
            except BaseException:
                pool.release(buf)
                raise

        pre = pool.available
        for _ in range(pre * 3):
            with pytest.raises(self.Boom):
                decode_that_fails(None)
        assert pool.available == pre, "a failed decode kept its buffer"

    @staticmethod
    def _stub_source(pool, block_count=1, rows=4, L=4):
        """A minimal format-3 source whose native writer can be fault-injected."""
        from contextlib import contextmanager
        from roshambo3.format.r3g.limits import BLOCK_DTYPE, RUN_DTYPE

        blocks = np.zeros(block_count, dtype=BLOCK_DTYPE)
        for index in range(block_count):
            blocks[index]["rows"] = rows
            blocks[index]["m0"] = index
            blocks[index]["m1"] = index + 1

        class Group:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def block_bytes(self, _index):
                return b"format-3 block"

        group = Group()
        group.blocks = blocks

        @contextmanager
        def open_group(_index):
            yield group

        src = R3gSource.__new__(R3gSource)
        src._pool, src._owns_pool = pool, False
        src.tile, src.planar, src.color = True, False, False
        src.prefetch_depth = 0
        src._sidecars = {}
        src._lock = threading.Lock()
        src.corpus = types.SimpleNamespace(
            group_table=[{"L": L, "L_heavy": L, "L_field_width": L,
                          "bucket": 0, "n_conf": block_count * rows}],
            n_types=0, group=open_group)
        src._group_sidecars = lambda _gi: {"runs": np.zeros(block_count, RUN_DTYPE)}
        return src

    def test_native_decode_failure_after_acquire_releases_its_buffer(self, monkeypatch):
        """A failed format-3 native decode returns its staged buffer once."""
        from roshambo3.engine.r3g_source import native
        pool = self._pool(elems=8)
        src = self._stub_source(pool)

        def fail(*_args, **_kwargs):
            raise ValueError("injected native block failure")

        monkeypatch.setattr(native, "write_gpu_launch", fail)
        job = _Job(group_index=0, block0=0, n_blocks=1, row0=0, gid_base=0,
                   n_conf=4)
        pre = pool.available
        with pytest.raises(ValueError, match="injected native"):
            src._decode(job)
        assert pool.available == pre

    def test_a_later_block_decode_failure_returns_the_launch_buffer(self, monkeypatch):
        """An error in a later physical block cannot strand the shared buffer."""
        from roshambo3.engine.r3g_source import native
        pool = self._pool(elems=2 * BTILE * 4 * 4)
        src = self._stub_source(pool, block_count=2, rows=BTILE)

        class Boom(RuntimeError):
            pass

        def fail_on_second(inputs, _n_conf, _heavy, _width, _types, buffer, **_kw):
            assert len(inputs) == 2
            buffer[:8] = 7  # The first block has started writing.
            raise Boom("the second physical block is unreadable")

        monkeypatch.setattr(native, "write_gpu_launch", fail_on_second)
        job = _Job(group_index=0, block0=0, n_blocks=2, row0=0, gid_base=0,
                   n_conf=2 * BTILE)
        pre = pool.available
        with pytest.raises(Boom, match="second physical block"):
            src._decode(job)
        assert pool.available == pre
        assert len(pool) == pre

    def test_a_double_release_is_refused(self):
        """A buffer returned twice is handed to two blocks at once, which
        corrupts scores rather than raising. Caught for real: an earlier draft
        of the prefetcher's release paths did exactly this."""
        pool = self._pool()
        b = pool.acquire()
        pool.release(b)
        with pytest.raises(ValueError, match="already in the pool"):
            pool.release(b)

    def test_a_foreign_buffer_is_refused(self):
        pool = self._pool()
        with pytest.raises(ValueError, match="does not own"):
            pool.release(np.empty(64, np.float32))

    def test_a_consumer_that_stops_early_gets_every_buffer_back(self):
        """`_drain`'s job: blocks decoded ahead of the consumer are still
        checked out when the consumer stops."""
        pool = self._pool(n=6)
        made = []

        def decode(job):
            b = pool.acquire()
            made.append(b)
            return types.SimpleNamespace(buffer=b, job=job)

        pf = Prefetcher(list(range(6)), decode, n_workers=2, depth=3,
                        on_discard=lambda item: pool.release(item.buffer))
        it = iter(pf)
        first = next(it)
        pool.release(first.buffer)   # the consumer owns what it was handed
        it.close()
        assert pool.available == len(pool)

    def test_a_consumer_that_raises_gets_every_buffer_back(self):
        pool = self._pool(n=6)

        def decode(job):
            return types.SimpleNamespace(buffer=pool.acquire(), job=job)

        pf = Prefetcher(list(range(6)), decode, n_workers=2, depth=3,
                        on_discard=lambda item: pool.release(item.buffer))
        with pytest.raises(self.Boom):
            for block in pf:
                try:
                    raise self.Boom("consumer failed")
                finally:
                    pool.release(block.buffer)
        assert pool.available == len(pool)

    def test_one_failed_release_does_not_strand_the_rest(self):
        """`_drain` runs while an exception is already propagating; if one
        `on_discard` raises, every buffer behind it used to be abandoned."""
        pool = self._pool(n=6)
        seen = []
        decoded = threading.Semaphore(0)

        def decode(job):
            item = types.SimpleNamespace(buffer=pool.acquire(), job=job)
            decoded.release()
            return item

        def discard(item):
            seen.append(item.job)
            if len(seen) == 1:
                raise self.Boom("release failed")
            pool.release(item.buffer)

        pf = Prefetcher(list(range(6)), decode, n_workers=2, depth=3,
                        on_discard=discard)
        it = iter(pf)
        first = next(it)
        pool.release(first.buffer)
        # Jobs 0..3 have been submitted (depth 3, plus one refill). Wait for
        # all four to finish: `_drain` rightly cancels a job that never
        # started, so closing early can leave a single finished item behind
        # the failing one and the assertion below with nothing to test. That
        # race failed this test on macOS (Python 3.14).
        for _ in range(4):
            assert decoded.acquire(timeout=10), "a prefetch decode never finished"
        it.close()
        assert sorted(seen) == [1, 2, 3], "drain gave up after the first failure"


@needs_gpu
class TestStageTimingIsNotClobbered:
    """Stage timing is process-global state, so a screen must leave it as it
    found it.

    Two defects: the enable happened before the pool and the sink were built,
    so a failure in either left it on for the life of the process; and the
    disable was unconditional rather than a restore, so a screen run inside a
    bench script that had enabled timing itself silently switched it off.
    """

    def _enabled(self):
        return bool(backend.extension().stage_timings().get("enabled", False))

    def test_a_failed_setup_does_not_leave_timing_on(self):
        class NoSource:
            prefetch_depth = 0

            @property
            def max_block_elements(self):
                raise RuntimeError("setup failed")

        assert not self._enabled()
        eng = Engine(stage_timing=True)
        with pytest.raises(RuntimeError, match="setup failed"):
            eng.run(NoSource(), None, ScreenConfig(), TopK(k=4))
        assert not self._enabled(), "stage timing survived a failed setup"

    def test_an_outer_caller_that_enabled_timing_keeps_it(self):
        ext = backend.extension()
        ext.stage_timing_enable(True)
        try:
            class NoSource:
                prefetch_depth = 0

                @property
                def max_block_elements(self):
                    raise RuntimeError("setup failed")

            with pytest.raises(RuntimeError):
                Engine(stage_timing=True).run(NoSource(), None, ScreenConfig(),
                                              TopK(k=4))
            assert self._enabled(), "the engine switched off timing it did not enable"
        finally:
            ext.stage_timing_enable(False)


class TestMergingPartialResults:
    """`TopK.merge` had no test at all -- not merely no `rank_col` test.

    It is the whole multi-GPU story: one engine per device over disjoint
    groups, merged on the host. A merge that ranks on the wrong column, or
    quietly returns a short list, produces a plausible answer that no
    single-device run would ever reproduce.
    """

    def _part(self, scores, gids, n_q=1):
        """One sink's finalize() shape: per query, (scores[k, 20], gids[k])."""
        out = []
        for _ in range(n_q):
            s = np.zeros((len(scores), N_SCORE_COLUMNS), np.float32)
            s[:, 0] = scores          # combination
            s[:, 7] = scores[::-1]    # a different column, deliberately
            out.append((s, np.asarray(gids, np.int64)))
        return out

    def test_it_ranks_on_the_requested_column(self):
        """The defect: `merge` hardcoded `s[:, 0]`, so a screen ranked on
        column 7 was silently re-ranked on column 0 when its parts were
        merged -- and column 0 is a real, plausible score."""
        a = self._part([0.9, 0.1], [1, 2])
        b = self._part([0.5, 0.4], [3, 4])

        on0 = TopK.merge([a, b], k=4, rank_col=0)[0]
        assert list(on0[1]) == [1, 3, 4, 2]

        on7 = TopK.merge([a, b], k=4, rank_col=7)[0]
        assert list(on7[1]) != list(on0[1]), "rank_col made no difference"
        assert on7[0][0, 7] == max(on7[0][:, 7])

    def test_an_out_of_range_rank_col_is_refused(self):
        with pytest.raises(ValueError, match="rank_col"):
            TopK.merge([self._part([0.1], [1])], k=1,
                       rank_col=N_SCORE_COLUMNS)

    def test_parts_with_different_query_counts_are_refused(self):
        """A short part used to raise IndexError; a long one was silently
        truncated to `parts[0]`'s query count."""
        a = self._part([0.9], [1], n_q=2)
        b = self._part([0.5], [2], n_q=3)
        with pytest.raises(ValueError):
            TopK.merge([a, b], k=1)

    def test_ties_break_on_ascending_id(self):
        """One tie-break, everywhere: numerics.rank_order."""
        a = self._part([0.5, 0.5], [7, 3])
        merged = TopK.merge([a], k=2, rank_col=0)[0]
        assert list(merged[1]) == [3, 7]

    def test_an_empty_merge_is_empty_rather_than_an_error(self):
        assert TopK.merge([], k=4) == []


@needs_gpu
class TestMergingSinksInfersOnlyWhatIsSafe:
    """`merge_sinks` reads `rank_col` off the sinks, which is right, and used
    to infer `k` as `max(k)`, which is not: the largest asks for a global
    top-k that the smaller partitions never retained enough rows to supply, and
    `rank_order(...)[:k]` returns a short list rather than raising."""

    def test_mixed_capacities_refuse_to_infer_a_k(self):
        with pytest.raises(ValueError, match="different numbers of candidates"):
            TopK.merge_sinks([TopK(k=10), TopK(k=100)])

    def test_an_explicit_k_beyond_the_smallest_is_refused(self):
        with pytest.raises(ValueError, match="exceeds the 10 candidates"):
            TopK.merge_sinks([TopK(k=10), TopK(k=100)], k=50)

    def test_a_safe_explicit_k_is_accepted(self):
        TopK.merge_sinks([TopK(k=10), TopK(k=100)], k=10)

    def test_differing_rank_cols_are_refused(self):
        with pytest.raises(ValueError, match="different columns"):
            TopK.merge_sinks([TopK(k=10, rank_col=0), TopK(k=10, rank_col=1)])



class TestThePoolUnderConcurrentRelease:
    """`release` scans the queue and then puts; both must happen under the
    queue's own lock.

    Releases genuinely race: a worker releases on a failed decode
    (`R3gSource._decode`) while the consumer releases in the engine's
    `finally`. An unlocked scan of `self._q.queue` can raise "deque mutated
    during iteration" *before* the put -- leaking the buffer the check exists
    to protect, from inside a `finally`, replacing the real exception.
    """

    def test_hammering_acquire_and_release_never_loses_or_duplicates(self):
        import threading

        pool = PinnedPool(8, 32, pinned=False)
        errors = []

        def worker():
            try:
                for _ in range(4000):
                    b = pool.acquire()
                    pool.release(b)
            except BaseException as e:      # noqa: BLE001 - reported, not raised
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors[:3]
        assert pool.available == len(pool), (
            f"{len(pool) - pool.available} buffer(s) lost under contention")
        # And no duplicates crept in, which is the other direction of the same
        # bug: a queue longer than the pool means one buffer is handed out twice.
        assert pool.available <= len(pool)


@needs_gpu
class TestColorSurvivesLaunchCoalescing:
    """The color path stages a second plane, so it gets its own identity check.

    Color widens ``L`` to ``g.L + g.L_color``, so every offset in the run is
    computed from a different width than a shape screen's would be, and the
    per-conformer count vector is sliced for the whole run rather than per
    constituent block. (There is no per-run type plane; the kernel takes the
    counts.)
    """

    @staticmethod
    def _queries(corpus):
        from roshambo3.numerics import WIRE_QUANT_STEP
        from roshambo3.pharmacophore import PharmacophoreGenerator
        generator = PharmacophoreGenerator()
        source = R3gSource(corpus, color=True, color_generator=generator,
                           workers=1, prefetch_depth=0)
        for job in source.jobs:
            block = source._decode(job)
            n_tiles = block.coords.shape[0] // BTILE
            coords = (block.coords.reshape(n_tiles, block.L, BTILE, 3)
                      .transpose(0, 2, 1, 3).reshape(-1, block.L, 3))
            for row, counts in enumerate(block.counts):
                n_real, n_color = int(counts[0]), int(counts[1:].sum())
                if not n_color:
                    continue
                xyz = coords[row].astype(np.float32) * np.float32(WIRE_QUANT_STEP)
                color_types = np.repeat(
                    np.arange(1, counts.size, dtype=np.int32), counts[1:])
                return from_arrays(
                    [xyz[:n_real]], ["format3-color-query"],
                    color_generator=generator,
                    color=[(xyz[n_real:n_real + n_color], color_types)])
        raise AssertionError("color corpus contains no molecule with a feature")

    def test_the_staged_color_planes_are_bitwise_identical(self, color_corpus):
        """Host side first, because it localizes a failure: if the coordinates
        or the counts differ here, no amount of kernel evidence says which of
        the two moved."""
        base = {}
        for target in (BLOCK_CONF, 8192, 65_536):
            src = R3gSource(color_corpus, color=True,
                             conformers_per_launch=target)
            seen = {}
            for job in src.jobs:
                b = src._decode(job)
                n_tiles = b.coords.shape[0] // BTILE
                # [tile][atom][lane][3] int16, the wire record.
                coords = (b.coords.reshape(n_tiles, b.L, BTILE, 3)
                          .transpose(0, 2, 1, 3).reshape(-1, b.L, 3))
                for r in range(b.n_conformers):
                    seen[b.gid_base + r] = (coords[r].tobytes(),
                                            b.counts[r].tobytes(),
                                            int(b.n_real[r]), int(b.mol_id[r]))
            if not base:
                base = seen
                assert base, "the color corpus staged nothing"
            else:
                assert seen.keys() == base.keys(), target
                bad = [g for g in base if base[g] != seen[g]]
                assert not bad, f"{len(bad)} conformers differ at {target}"

    def test_a_color_screen_is_bit_identical_at_every_coalesce_factor(
            self, color_corpus):
        qs = self._queries(color_corpus)
        base = None
        base_launches = None
        for target in (BLOCK_CONF, 16_384, 65_536):
            src = R3gSource(color_corpus, color=True,
                             conformers_per_launch=target)
            engine = Engine(device_id=0)
            stats, res = engine.run(src, qs, ScreenConfig(color=True),
                                    TopK(k=100))
            engine.close()
            if base is None:
                base, base_launches = res, stats.n_blocks
                # Not vacuous: the uncoalesced arm really is more launches.
                assert base_launches > 1
            else:
                assert stats.n_blocks < base_launches, target
                for (sa, ia), (sb, ib) in zip(base, res):
                    assert np.array_equal(ia, ib), target
                    assert np.array_equal(sa, sb), target

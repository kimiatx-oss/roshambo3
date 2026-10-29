"""The CPU backend must be reachable through `Engine`, not only by hand.

CPU numbers were once all taken by calling `backend.cpu.score_pairs`
directly, and nobody had tried the backend seam. A plain `Engine` run on the CPU backend did not fall back to something slow --
it **threw**, three times over:

1. `engine.py` passes `library_tile_mode` on every launch, because the party
   that packed the buffer is the only one that knows its layout -- and the CPU
   backend refused the knob outright rather than honoring the one layout it
   can read. An untiled block also reported `tile_mode=1`, misdescribing its
   own buffer, which was harmless only while every consumer tiled.
2. The default sink is a device-resident `TopK`.
3. `ScreenConfig.for_width` emits the argument list a CUDA launch takes, so the
   screen died on `smem_carveout` before it scored a pair.

Each of those is a seam that was built and never driven. This file drives it.
"""
from __future__ import annotations

import numpy as np
import pytest

@pytest.fixture(scope="module")
def screened(smoke_corpus):
    from roshambo3.engine import Engine, HostTopK
    from roshambo3.engine.config import ScreenConfig
    from roshambo3.engine.query import from_arrays
    from roshambo3.engine import R3gSource
    from roshambo3.format import Catalog
    import roshambo3.backend.cpu as cpu

    c = smoke_corpus
    cat = Catalog(c)
    qcs = cat.coordinates(np.array([0, 1], dtype=np.int64))
    queries = from_arrays([a[:, :3] for a in qcs], ["q0", "q1"])
    engine = Engine(backend=cpu)
    _, res = engine.run(
        R3gSource(c, tile=False, workers=2, conformers_per_launch=1024),
        queries, config=ScreenConfig(), sink=HostTopK(k=50))
    return c, cat, queries, engine, res, cpu


def test_a_plain_screen_runs_at_all(screened):
    """The whole point: it used to raise before scoring a pair."""
    _, _, _, _, res, _ = screened
    assert len(res) == 2
    for scores, gids in res:
        assert scores.shape[0] == 50 and gids.shape[0] == 50


def test_cuda_only_knobs_are_dropped_at_their_default(screened):
    """Dropped because nobody chose them -- not because they were ignored."""
    _, _, _, engine, _, cpu = screened
    dropped = engine._dropped_launch_keys
    assert "smem_carveout" in dropped
    assert dropped <= cpu.inert_launch_knobs()


def test_a_cuda_only_knob_the_caller_CHOSE_is_still_refused(screened):
    """The other half of the contract, and the half that is easy to lose.

    Dropping an inert knob at its default must not decay into dropping it
    always: a knob someone deliberately moved has to reach the backend and be
    refused by name. The launch key is not the config field -- `lanes_per_group`
    is emitted from `lanes` -- so the movable keys are found by resolving
    a moved config rather than guessed, and the test fails rather
    than skips if nothing there is movable.
    """
    from roshambo3.engine import Engine, HostTopK
    from roshambo3.engine.config import ScreenConfig
    from roshambo3.engine import R3gSource
    c, _, queries, _, _, cpu = screened

    inert = cpu.inert_launch_knobs()
    ref = ScreenConfig().for_width(32, device_id=0, L_shape=32,
                                   query_atoms=32, n_features=1)
    movable = []
    for key in sorted(inert):
        cur = getattr(ScreenConfig(), key, None)
        if not isinstance(cur, int) or isinstance(cur, bool):
            continue
        cfg = ScreenConfig()
        object.__setattr__(cfg, key, int(cur) + 4)
        probe = cfg.for_width(32, device_id=0, L_shape=32,
                              query_atoms=32, n_features=1)
        if probe.get(key) != ref.get(key):
            movable.append((key, cfg))
    assert movable, (
        "no CUDA-only launch key could be moved through ScreenConfig, so this "
        "test would silently stop checking anything -- the defect shape "
        "vacuous checks take. Fix the test, do not skip it.")

    from_backend = 0
    for key, cfg in movable:
        with pytest.raises(Exception) as excinfo:
            Engine(backend=cpu).run(
                R3gSource(c, tile=False, workers=2,
                           conformers_per_launch=1024),
                queries, config=cfg, sink=HostTopK(k=50))
        msg = str(excinfo.value)
        assert key in msg, f"{key} was moved and the refusal never named it"
        if "cannot honor" in msg:
            from_backend += 1
    # KNOWN GAP, pinned as an observation rather than asserted away.
    #
    # `from_backend` is currently 0: every CUDA-only knob a caller can move
    # through `ScreenConfig` is refused by **the pre-launch cell check**
    # before it reaches the backend, and that check is CUDA-specific -- it
    # tells a CPU-backend caller a CUDA launch setting is not supported. So
    # the contract is upheld (the knob is
    # refused, and the message names it) but by the wrong guard, for a reason
    # about a device that is not in use. Same shape as the
    # `library_tile_mode` and `smem_carveout` gaps this file exists for, and
    # the next one to close.
    #
    # Asserted as == 0 deliberately: when that check becomes backend-aware
    # this test will fail, which is the reminder to update the comment rather
    # than discover the change by accident.
    assert from_backend == 0, (
        "a moved CUDA-only knob now reaches the backend and is refused there. "
        "That is an improvement -- the cell check has presumably become "
        "backend-aware. Update this test and the note above.")


def test_the_backend_itself_refuses_a_cuda_knob_by_name(screened):
    """The seam-level half, driven directly because the Engine cannot reach it.

    The test above shows the pre-launch cell check fires first for everything a caller
    can move through `ScreenConfig`. The backend's own refusal is still the
    contract and still has to work, so it is exercised here without the Engine
    in the way.
    """
    _, _, queries, _, _, cpu = screened
    n = 4
    a = dict(query_coords=np.zeros((1, n, 4), np.float32),
             query_n_atoms=np.array([n], np.int32),
             library_coords=np.zeros((2, n, 4), np.float32),
             library_counts=np.full((2, 1), n, np.uint8),
             pair_query_row=np.zeros(2, np.int32),
             pair_library_row=np.arange(2, dtype=np.int32))
    for key in sorted(cpu.inert_launch_knobs()):
        with pytest.raises(Exception, match="cannot honor"):
            cpu.score_pairs(**a, **{key: 1})


def test_the_engine_path_is_bit_identical_to_a_direct_call(screened):
    """Blocking, prefetching and host reduction must change nothing.

    The determinism argument is that the output is a pure function of the
    SET screened, not of block size or launch structure. This is that claim,
    driven through the real source and the real sink rather than a passthrough.
    """
    from roshambo3 import numerics
    c, cat, queries, _, res, cpu = screened

    n = int(sum(int(g["n_conf"]) for g in c.group_table))
    allc = cat.coordinates(np.arange(n, dtype=np.int64))
    L = max(a.shape[0] for a in allc)
    lc = np.zeros((n, L, 4), np.float32)
    cnt = np.zeros((n, 1), np.uint8)
    for i, a in enumerate(allc):
        k = a.shape[0]
        lc[i, :k, :3] = a[:k, :3]
        lc[i, :k, 3] = 1.0
        cnt[i, 0] = k
    # The direct call must use the SAME configuration the Engine selected,
    # or this compares blocking against a knob change. The Engine now supplies
    # a default configuration, so a bare direct call is a different kernel.
    _, _, _, engine, _, _ = screened
    same = cpu.default_screen_knobs(
        n, 0.8, int(np.median(np.asarray(queries.n_real))))
    direct = cpu.score_pairs(
        query_coords=queries.coords, query_n_atoms=queries.n_real,
        library_coords=lc, library_counts=cnt,
        pair_query_row=np.repeat(np.arange(2, dtype=np.int32), n),
        pair_library_row=np.tile(np.arange(n, dtype=np.int32), 2),
        **same)[:, 1].reshape(2, n)
    ids = np.arange(n, dtype=np.int64)
    for q, (scores, gids) in enumerate(res):
        g = gids.astype(np.int64)
        assert np.array_equal(scores[:, 1], direct[q, g]), "not bit-identical"
        assert np.array_equal(g, numerics.rank_order(direct[q], ids)[:len(g)]), (
            "the host reducer disagrees with numerics.rank_order -- the one "
            "tie-break this repository has")


def test_an_untiled_block_says_so(screened):
    """It used to report tile_mode=1 while holding an untiled buffer."""
    from roshambo3.engine.buffers import PinnedPool
    from roshambo3.engine import R3gSource
    c, _, _, _, _, _ = screened

    def source(tile):
        # A source left to build its own pool pins it, which needs the CUDA
        # extension; hand it an ordinary one, as the Engine's CPU path does.
        n = R3gSource(c, tile=tile, workers=1, conformers_per_launch=1024).max_block_elements
        return R3gSource(c, tile=tile, workers=1, conformers_per_launch=1024,
                          pool=PinnedPool(2, n, pinned=False))

    assert int(getattr(next(source(False).blocks()), "tile_mode", 1)) == 0
    assert int(getattr(next(source(True).blocks()), "tile_mode", 0)) == 1


def test_the_anchor_is_not_moved_by_the_default(screened):
    """The Engine's default configuration must not reach `score_pairs`' own defaults.

    The extension is the CPU reference anchor and `test_anchor.py`
    reaches it through this seam, so a caller that asks for nothing must still
    get the scalar exact path. That is why the default is published for the
    Engine to *ask* for rather than applied inside the backend.
    """
    import inspect
    _, _, _, _, _, cpu = screened
    sig = inspect.signature(cpu.score_pairs)
    for knob in ("field_h", "field_width", "field_batched", "field_gridxf"):
        assert knob not in sig.parameters, (
            f"{knob} became a named parameter of the seam; if it also gained a "
            "non-default value it would move the stamped anchor")


def test_the_default_is_off_below_the_crossover_and_on_above(screened):
    """The field-path scale threshold, as a floor rather than a fix.

    The field path is 0.26x the analytic grain at 128 library conformers
    per launch and 2.29x at 32,768, and 0.54x at one query atom against
    2.21x at 32 -- a blanket default would be a large regression on small
    screens, and on thin queries at any size.
    """
    _, _, _, _, _, cpu = screened
    QA = cpu.FIELD_PATH_MIN_QUERY_ATOMS
    # Below either threshold the answer is the analytic conformer grain --
    # NOT `{}`. Returning nothing let the extension fall through to its own
    # defaults, `grain=-1, width=1`, which is the reference
    # ANCHOR: 12.2k pairs/s against the grain's 82.5k on the same panel.
    small = cpu.default_screen_knobs(cpu.FIELD_PATH_MIN_LIBRARY - 1, 0.8, QA)
    assert small.get("grain") == 2 and "field_h" not in small
    # And the query-atom threshold, which the first version of this default
    # ignored: the field path's advantage is deleting the N_A factor, and at
    # one query atom there is none to delete. Measured 0.54x at 1 atom.
    thin = cpu.default_screen_knobs(1 << 20, 0.8, QA - 1)
    assert thin.get("grain") == 2 and "field_h" not in thin
    big = cpu.default_screen_knobs(cpu.FIELD_PATH_MIN_LIBRARY, 0.8, QA)
    assert big["field_h"] > 0 and big["field_width"] == 8
    assert big["field_unroll"] == 0 and big["field_pipeline"] is True
    assert big["field_gridxf"] == 1 and big["field_allin"] == 1
    assert big["field_batched"] is True
    # This is undecided; it must not arrive as a default.
    assert "field_winner_only" not in big


def test_a_caller_who_sets_a_knob_beats_the_default(screened):
    """A default that overrode the caller would not be a default."""
    from roshambo3.engine import Engine
    _, _, _, _, _, cpu = screened
    e = Engine(backend=cpu)
    chosen = {"field_h": 0.5, "field_batched": False}
    out = e._backend_screen_defaults(dict(chosen), 65536)
    assert out["field_h"] == 0.5
    assert out["field_batched"] is False
    assert out["field_gridxf"] == 1      # still filled in where nothing was set


def test_a_backend_without_the_hook_is_untouched(screened):
    """The CUDA path must see no change at all."""
    from roshambo3.engine import Engine
    _, _, _, _, _, cpu = screened
    e = Engine(backend=cpu)
    e.backend = type("Bare", (), {})()
    args = {"a": 1, "b": 2}
    assert e._backend_screen_defaults(dict(args), 65536) == args
    assert e._launch_args_for_backend(dict(args), (1, 1, 1, 1), dict) == args

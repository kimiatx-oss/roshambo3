"""The staging-width policy and the analytic launcher must agree on one budget.

``engine.config.stage_width_for`` chooses a library staging width; the launcher
then decides whether that width can be launched. Both compare a shared-memory
requirement against the *same* device limit, and for a long time they compared
**different requirements**: the policy sized the staged library alone, the
launcher sized the whole block. The difference is everything else the block
holds -- the query's xyzw copy, the per-block cache, the query type ids and the
interaction tables -- which is 488 B at a 22-atom shape query.

A few hundred bytes of slack is almost always there, so this was invisible
until the staged buffer was *exactly* the whole budget. On a Volta card the
opt-in limit is 98,304 B and ``256 x 64 x 6`` is 98,304 B exactly, so the
policy returned 256 and the launcher refused the launch it had just been
handed: a crash at the shipped defaults on every corpus with an L = 64 bucket,
found on a real screen.

**The failure mode is the two disagreeing, so these tests compare them.** A
test that the policy returns 128 at some cell would pass against a policy that
returns 128 for the wrong reason and would say nothing about the next
architecture. Two instruments here:

* on the CPU, every width the policy can choose is checked against the
  launcher's own sum (``backend.cuda.analytic_block_shared_bytes``), over the
  opt-in limits of every supported architecture;
* on the GPU, the policy's answer is **handed to the launcher**, at every
  shipped bucket width and at the width where this device's staged buffer
  exactly fills its budget -- which is L = 64 on Volta, L = 66 here, and is
  derived from the device rather than tabulated.

The GPU half is what makes this more than a restatement of the arithmetic:
a passing run on a 99 KiB card proves nothing about a
96 KiB one, so the boundary cell is computed from whatever card is present.
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.backend import cuda as backend
from roshambo3.backend.cuda import PoseSearch, analytic_block_shared_bytes
from roshambo3.engine import ScreenConfig
from roshambo3.engine import config as C
from tests.gpu_device import device_present

#: Opt-in shared memory per block, in KiB, on every architecture this ships
#: for: Turing, Volta, sm_86/89/120, sm_80, sm_90. 96 is the one that bit.
ARCHITECTURE_BUDGETS_KIB = (48, 64, 96, 99, 163, 227)

#: The padded widths a prepared corpus actually buckets into.
BUCKET_WIDTHS = (16, 24, 32, 40, 48, 56, 64, 80, 128)


#: Decided without a CUDA call: counting devices builds a context, and doing
#: that while pytest collects -- before any GPU lock, and where a foreign MPS
#: daemon hangs it -- is what this used to do.
needs_gpu = pytest.mark.skipif(not (backend.available() and device_present()),
                               reason="no CUDA device in this build")


@pytest.fixture(scope="module")
def device_ids() -> list[int]:
    """Every device this machine has, enumerated when a test first asks."""
    return list(range(int(backend.extension().device_count())))


def _library_only_rule(L: int, budget: int, **kw) -> int:
    """The selection rule as it stood before the accounting fix.

    Kept here, not to be shipped, but so that the tests below can be run
    against the defect: installed as ``STAGE_WIDTH_POLICY`` it makes this file
    fail, which is the difference between a test of the fix and a restatement
    of it.
    """
    for w in C.STAGE_WIDTHS:
        if w == 0 or w * L * 6 <= budget:
            return w
    return 0


# --------------------------------------------------------------------------
# The policy against the launcher's own sum -- no GPU needed
# --------------------------------------------------------------------------

class TestThePolicySizesTheWholeBlock:

    @pytest.mark.parametrize("budget_kib", ARCHITECTURE_BUDGETS_KIB)
    @pytest.mark.parametrize("query_atoms", (8, 22, 64, 128))
    def test_every_chosen_width_fits_the_launchers_own_sum(self, budget_kib,
                                                           query_atoms):
        """The invariant, over every width a bucket could have.

        Not "the policy returns 128 at L = 64": that is one cell of one card.
        This is the property the launcher checks, checked against the same
        expression the launcher checks it with.
        """
        budget = budget_kib * 1024
        for L in range(1, 257):
            w = C.stage_width_for(L, budget=budget, query_atoms=query_atoms)
            need = analytic_block_shared_bytes(query_atoms, L, w)
            assert need <= budget, (
                f"policy picked stage width {w} at L = {L} for a {budget} B "
                f"budget, and the launcher needs {need} B for that block "
                f"-- over by {need - budget} B")

    def test_the_cell_that_crashed_steps_down(self):
        """The Volta cell, named because it is the one that shipped broken.

        98,304 B of opt-in shared memory, and ``256 * 64 * 6`` is 98,304 B on
        the nose. The staged buffer fits and the block does not.
        """
        # The query record was trimmed: 488 B -> 208 B at 22 atoms and one
        # feature type. The cell still crashes at 256 -- 98,304 B of staged
        # buffer leaves nothing at all, so no query record fits, however small.
        assert analytic_block_shared_bytes(22, 64, 256) == 98304 + 208
        # It still steps down, and now it steps down by ONE RUNG rather
        # than by a factor of two. 253 fits here; 224 is the widest whole number
        # of 32-row tiles inside that, and the ladder delivers it.
        assert C.stage_width_for(64, budget=98304, query_atoms=22) == 224
        # The same equality recurs one width down: 128 * 128 * 6 is 98,304 too.
        # With the old ladder the next rung was 0 and this bucket stopped staging
        # entirely; 96 fits and is now what it gets.
        assert C.stage_width_for(128, budget=98304, query_atoms=22) == 96

    # The finer ladder widened these lists rather than changing what they
    # mean; the query trim then NARROWED them, and for a reason worth stating: the old
    # rule sizes the staged buffer alone against the whole budget, so it fails
    # at every L where some rung lands within the query record's size of the
    # limit -- and the record is now 208 B where it was 488. A smaller record
    # is a smaller target. The defect is unchanged; less of it is reachable.
    #
    # What SURVIVES the trim is exact equality: at 96 KiB, L = 64 and L = 128
    # put a rung on the limit to the byte, where no record fits however small.
    # Turing's list empties entirely -- 65,536 is not a multiple of 1,536, so no
    # rung lands on the nose there, and 208 B fits in every remainder that 488
    # did not.
    @pytest.mark.parametrize("budget,expected", [
        # Volta: 96 KiB exactly. 64 and 128 are exact landings (256*64*6 and
        # 128*128*6 are both 98,304); 73 and 256 leave less than 208 B.
        (96 * 1024, [64, 73, 128, 256]),
        (99 * 1024, [66, 88, 132, 176]),      # sm_86/89/120: 99 KiB exactly
        # Turing: no rung lands on the nose, and every remainder holds 208 B.
        (64 * 1024, []),
    ])
    def test_the_failing_set_is_enumerated_and_is_now_empty(
            self, monkeypatch, budget, expected):
        """The set of widths the old rule got wrong, and the same set now.

        Enumerated from the arithmetic rather than asserted: at every ``L`` the
        rule is asked for a width and the launcher's sum is asked whether that
        width fits. The Volta row is the crash that shipped; the sm_120 row is why
        a passing run on this machine proved nothing.
        """
        def refused(rule):
            monkeypatch.setattr(C, "STAGE_WIDTH_POLICY", rule)
            return [L for L in range(1, 257)
                    if analytic_block_shared_bytes(
                        22, L, C.stage_width_for(L, budget=budget,
                                                 query_atoms=22)) > budget]

        assert refused(_library_only_rule) == expected
        assert refused(C.widest_stage_that_fits) == []

    def test_the_query_panel_changes_the_answer(self, monkeypatch):
        """The query record is an input, not a constant.

        At L = 63 on a 96 KiB card the staged buffer leaves 1,536 B. An 8-atom
        query needs 112 B of that and fits; a 256-atom query needs 1,840 B and
        does not. A policy that took only ``L`` could not tell these apart,
        which is exactly why it could not tell the crashing cell apart either.

        **The panel at which the two answers separate is itself a measurement.**
        With the old ladder the second answer was 128 (a factor of two) and the
        panel that produced it was 128 atoms. The finer ladder made that step
        one rung, 224. The query-record trim then made a 128-atom panel FIT -- 944 B rather
        than 2,608 -- so the separating panel moves to 256 atoms and the answer
        is 224 again. The property under test is that the two answers differ;
        which panel separates them is what the record's size is worth, stated
        as a test rather than as a claim.
        """
        monkeypatch.setattr(C, "_smem_budget_cache", {0: 96 * 1024})
        assert C.stage_width_for(63, device_id=0, query_atoms=8) == 256
        # This used to be 224 and is now 256 -- the trim bought this
        # panel a whole rung at this bucket.
        assert C.stage_width_for(63, device_id=0, query_atoms=128) == 256
        assert C.stage_width_for(63, device_id=0, query_atoms=256) == 224

    def test_the_selection_rule_is_replaceable_without_touching_the_arithmetic(
            self, monkeypatch):
        """Fitting and choosing are separate functions.

        Whether "the widest width that fits" is the right *objective* is
        unmeasured -- shared memory and L1 are one SRAM, and a 5.9x cost was measured
        from SRAM partitioned as shared and never used.
        What this pins is that answering it is a one-function edit rather than
        a re-derivation of the shared-memory sum.
        """
        monkeypatch.setattr(C, "STAGE_WIDTH_POLICY",
                            lambda L, budget, **kw: 0)
        assert C.stage_width_for(16, budget=227 * 1024) == 0

    def test_the_sum_is_the_launchers_sum_term_for_term(self):
        """Each term named, so a change to one of them is visible here.

        The launcher expression is ``align16(8*4 + 2*RP*4 + 2*N_features*4
        + NmolA*6 + NmolA) + stage_bytes``, with ``RP = N_features`` where the
        color loop reads the diagonal alone and ``N_features^2`` where it does
        not. Before it, ``NmolA*4*4 + 8*4 + NmolA*4 + 2*N_features^2*4 +
        2*N_features*4 + stage_bytes``.
        """
        assert analytic_block_shared_bytes(0, 0, 0) == 8 * 4 + 2 * 4 + 2 * 4
        # SEVEN bytes per padded query atom -- three int16 planes and a u8 type
        # id, where it was 20 (four float32 and an int32). The step measured
        # here is 16 rather than 7 because the block is aligned to 16 B for the
        # staged library, so one more atom crosses one boundary; over 16 atoms
        # the difference is exactly 7 each.
        assert (analytic_block_shared_bytes(23, 0, 0)
                - analytic_block_shared_bytes(22, 0, 0)) == 16
        assert (analytic_block_shared_bytes(38, 0, 0)
                - analytic_block_shared_bytes(22, 0, 0)) == 16 * 7
        # Six bytes per staged atom per thread: three int16 planes.
        assert analytic_block_shared_bytes(0, 64, 256) - 48 == 256 * 64 * 6
        # The interaction tables are LINEAR in the feature count where the
        # color loop reads the diagonal alone (the shipped structure), and
        # quadratic where it does not.
        assert (analytic_block_shared_bytes(0, 0, 0, n_features=7)
                - analytic_block_shared_bytes(0, 0, 0, n_features=1)
                == 2 * (7 - 1) * 4 + 2 * (7 - 1) * 4)
        assert (analytic_block_shared_bytes(0, 0, 0, n_features=7,
                                            interaction_diagonal_only=False)
                - analytic_block_shared_bytes(0, 0, 0, n_features=1,
                                              interaction_diagonal_only=False)
                == 2 * (49 - 1) * 4 + 2 * (7 - 1) * 4)


# --------------------------------------------------------------------------
# The policy's answer, handed to the launcher
# --------------------------------------------------------------------------

def _shape_only_launch(device_id: int, L: int, stage_width: int,
                       query_atoms: int = 22, n_library: int = 64,
                       steps: int = 2):
    """One shape-only ANALYTIC launch at an explicit stage width.

    Deliberately at the launcher boundary rather than through ``Engine``: the
    thing under test is whether the launcher accepts what the policy chose, and
    a screen can only reach the bucket widths a corpus happens to have.
    """
    from roshambo3.layout import tile32

    rng = np.random.default_rng(20260827 + L)
    n_real_q = min(query_atoms, L) if L < query_atoms else query_atoms

    qc = np.zeros((1, query_atoms, 4), np.float32)
    qc[0, :n_real_q, :3] = rng.normal(0.0, 2.0, size=(n_real_q, 3))
    qc[0, :n_real_q, 3] = 1.0

    n_real_t = np.full(n_library, min(L, 24), np.int32)
    lib = np.zeros((n_library, L, 4), np.float32)
    for i in range(n_library):
        k = int(n_real_t[i])
        lib[i, :k, :3] = rng.normal(0.0, 2.0, size=(k, 3))
        lib[i, :k, 3] = 1.0

    return backend.score_pairs(
        query_coords=qc,
        query_types=np.zeros((1, query_atoms), np.int32),
        query_n_atoms=np.array([n_real_q], np.int32),
        library_coords=tile32(lib),
        library_counts=n_real_t.astype(np.uint8).reshape(-1, 1),
        pair_query_row=np.zeros(n_library, np.int32),
        pair_library_row=np.arange(n_library, dtype=np.int32),
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.zeros((1, 1), np.float32),
        pose_search=PoseSearch.ANALYTIC,
        color=False,
        stage_width=stage_width,
        steps=steps,
        device_id=device_id,
    )


def _exactly_full_widths(budget: int, query_atoms: int = 22) -> list[int]:
    """Widths where the staged buffer alone would fill this device's budget.

    Derived from the card, not tabulated: 64 and 128 on a 96 KiB Volta, 66 and
    132 on a 99 KiB sm_120. These are the cells where sizing the library
    instead of the block is the difference between a screen and a
    ``RuntimeError``, so they are where a policy/launcher disagreement shows.
    """
    out = []
    for w in (256, 128):
        L = budget // (w * 6)
        if L and analytic_block_shared_bytes(query_atoms, L, w) > budget:
            out.append(L)
    return out


@needs_gpu
class TestTheLauncherAcceptsWhatThePolicyChose:
    """Each check runs on every device present, in turn."""

    def test_every_bucket_width_launches(self, device_ids):
        for device_id in device_ids:
            self._every_bucket_width_launches(device_id)

    def test_the_width_that_exactly_fills_the_budget_launches(self, device_ids):
        for device_id in device_ids:
            self._the_width_that_exactly_fills_the_budget_launches(device_id)

    def test_the_rule_this_replaced_is_refused_at_the_same_cell(
            self, device_ids, monkeypatch):
        for device_id in device_ids:
            self._the_rule_this_replaced_is_refused_at_the_same_cell(
                device_id, monkeypatch)

    def _every_bucket_width_launches(self, device_id):
        """Every padded width a prepared corpus buckets into, on this card."""
        budget = C.shared_memory_budget(device_id)
        refused = []
        for L in BUCKET_WIDTHS:
            w = C.stage_width_for(L, device_id=device_id, query_atoms=22)
            try:
                _shape_only_launch(device_id, L, w)
            except Exception as exc:                          # noqa: BLE001
                refused.append((L, w, f"{type(exc).__name__}: {exc}"))
        assert not refused, (
            f"device {device_id} allows {budget} B per block and refused the "
            f"policy's own choice at {refused}")

    def _the_width_that_exactly_fills_the_budget_launches(self, device_id):
        """The cell the shipped policy got wrong, on whatever card is present.

        This is the test that fails without the accounting fix. It cannot be
        written as a constant: sm_120 has 3,072 B of slack at L = 64 and passes
        either way, so the width is computed from this device's own opt-in
        limit.
        """
        budget = C.shared_memory_budget(device_id)
        widths = _exactly_full_widths(budget)
        assert widths, (
            f"device {device_id}'s {budget} B budget has no width at which a "
            f"staged buffer exactly fills it; this test cannot say anything "
            f"here and should be given a card that has one")
        refused = []
        for L in widths:
            w = C.stage_width_for(L, device_id=device_id, query_atoms=22)
            try:
                _shape_only_launch(device_id, L, w)
            except Exception as exc:                          # noqa: BLE001
                refused.append((L, w, f"{type(exc).__name__}: {exc}"))
        assert not refused, (
            f"device {device_id} allows {budget} B per block and refused the "
            f"policy's own choice at {refused}")

    def _the_rule_this_replaced_is_refused_at_the_same_cell(
            self, device_id, monkeypatch):
        """The instrument itself, checked -- and the crash, on this card.

        Every assertion above is that nothing raised, which a launcher that
        accepted anything at all would also satisfy. So the pre-fix rule is
        installed as the policy and *its* answer is handed to the same launcher
        at the same cell. It must be refused, with the launcher's own message.
        """
        monkeypatch.setattr(C, "STAGE_WIDTH_POLICY", _library_only_rule)
        budget = C.shared_memory_budget(device_id)
        for L in _exactly_full_widths(budget):
            w = C.stage_width_for(L, device_id=device_id, query_atoms=22)
            with pytest.raises(RuntimeError, match="shared memory"):
                _shape_only_launch(device_id, L, w)


@needs_gpu
def test_the_engine_tells_the_policy_which_panel_it_is_screening(
        smoke_corpus, monkeypatch):
    """The panel has to reach the policy or the fix is inert in production.

    ``stage_width_for`` cannot charge for a query record it was never shown,
    and the only caller that knows the panel is ``Engine.run``.
    """
    from roshambo3.engine import Engine, R3gSource, TopK
    from roshambo3.engine.query import from_arrays

    rng = np.random.default_rng(20260827)
    qs = from_arrays([rng.normal(0.0, 2.0, size=(19, 3)).astype(np.float32)],
                     ["q0"])
    seen = []
    original = ScreenConfig.for_width

    def spy(self, L, device_id=0, L_shape=None, **kw):
        seen.append(dict(kw))
        return original(self, L, device_id=device_id, L_shape=L_shape, **kw)

    monkeypatch.setattr(ScreenConfig, "for_width", spy)
    Engine(device_id=0).run(R3gSource(smoke_corpus), qs,
                            ScreenConfig(pose_search=PoseSearch.ANALYTIC),
                            TopK(k=10))

    assert seen, "the engine resolved no per-bucket arguments at all"
    for kw in seen:
        assert kw["query_atoms"] == int(qs.coords.shape[1])
        assert kw["n_features"] == int(qs.interaction_r.shape[0])


class TestTheLadderIsAMenuAndTheDeviceChooses:
    """``STAGE_WIDTHS`` used to be ``(256, 128, 0)`` -- two factors of two
    apart -- so a fit boundary that fell between two rungs cost the whole
    factor. On Volta at L = 64 the boundary is 253 and the launch ran at 128;
    at L = 128 and 136 nothing fit and it stopped staging altogether.

    What is checked here is the ladder's *shape*, not any particular width:
    a rung must be launchable (a whole number of work-item tiles), the rungs
    must be a descending ladder ending in "do not stage", and every rung must
    have a compiled cell -- a ladder that offers a width the dispatch switch
    refuses is a refused launch at exactly the bucket that needed it.
    """

    def test_the_rungs_descend_and_end_in_no_staging(self):
        widths = list(C.STAGE_WIDTHS)
        assert widths[-1] == 0, "the ladder must be able to answer 'do not stage'"
        assert widths[:-1] == sorted(widths[:-1], reverse=True)
        assert len(set(widths)) == len(widths)

    def test_every_rung_is_a_whole_number_of_work_item_tiles(self):
        """``stage_width`` is the block dimension AND the work-item chunk
        (``interface.cpp``: ``chunk = stage``), so a width that is not a
        multiple of ``ROSHAMBO3_BTILE`` starts every work item off a tile
        boundary and every warp then reads across two tiles."""
        for w in C.STAGE_WIDTHS:
            if w:
                assert w % 32 == 0, w
                assert w <= 1024, "a stage width is also a block dimension"

    def test_every_rung_has_a_compiled_cell(self):
        """``engine.config.STAGE_WIDTHS`` and the launcher's dispatch table are
        the same ladder written twice. Read out of the source rather than
        launched, so this runs without a GPU and without a build."""
        import os
        import re
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(os.path.dirname(here), "src", "roshambo3",
                           "backend", "_cuda_src", "kernels.cu")
        assert os.path.exists(src), f"expected CUDA kernel sources at {src}"
        text = open(src).read()
        body = text[text.index("static inline int analytic_stage_dispatch_index"):]
        body = body[:body.index("}")]
        listed = {int(m) for m in re.findall(r"case\s+(\d+):", body)}
        for w in C.STAGE_WIDTHS:
            assert w in listed, (
                f"stage width {w} is on the policy's ladder and has no entry in "
                f"analytic_stage_dispatch_index, so the launcher would refuse "
                f"the width the policy just chose")
        shipped = {int(m) for m in re.findall(
            r"ROSHAMBO3_ANALYTIC_LAUNCH_SHIPPED\((\d+)\)", text)}
        for w in C.STAGE_WIDTHS:
            if w:
                assert w in shipped, (
                    f"stage width {w} has an index but no shipped dispatch case")

    def test_the_ladder_reaches_the_boundary_at_every_bucket(self):
        """The rungs exist to be *reachable*: at every bucket width and on every
        architecture, the ladder must deliver the widest multiple of 32 that
        fits. This is what the old ladder failed, silently, on 46.7% of
        one color corpus's conformers on sm_70."""
        for kib in ARCHITECTURE_BUDGETS_KIB:
            budget = kib * 1024
            for L in BUCKET_WIDTHS + (72, 88, 104, 112, 120, 136):
                room = budget - analytic_block_shared_bytes(22, 1, 0, n_features=7)
                best = min(max(room // (L * 6), 0), 256) // 32 * 32
                got = C.widest_stage_that_fits(L, budget, query_atoms=22,
                                               n_features=7)
                assert got == best, (kib, L, got, best)

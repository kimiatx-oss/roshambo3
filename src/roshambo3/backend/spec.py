"""The vocabulary a ``ScreenConfig`` is written in, and the arithmetic it needs.

Split out of ``backend.cuda`` for one reason: ``engine.config`` imports
every name below at module scope, so ``ScreenConfig`` could not be constructed
without importing the CUDA wrapper. That was harmless while CUDA was the only
backend and is not once there is a second one.

**Several of these are CUDA-shaped, and that is not hidden.** ``CoordResidency``
is about shared memory and registers; ``UpdateLanes`` is about a warp;
``carveout_for`` returns a value only ``cudaFuncSetAttribute`` consumes. They
live here because a configuration must be *expressible* without a GPU build, not
because they are device-independent. A backend that cannot honor one of them
**refuses it by name** rather than ignoring it -- that rule is older than this
module and is the reason it is worth stating here: an ignored knob is a silent
wrong answer, and this file is where a reader will come looking for what the
knobs mean.

Nothing here imports the extension, and nothing here may. The two helpers are
pure arithmetic over the same expressions the launcher uses, which is the point
of them: one expression decides both what the launcher asks the driver for and
what a caller may ask the launcher for.
"""

from __future__ import annotations

import enum

from ..layout import MAX_FEATURES  # noqa: F401  (re-exported for callers)


class PoseSearch(enum.IntEnum):
    """Which kernel finds the pose."""

    FIELD = 0
    """A precomputed query field on a lattice drives the search, and the
    converged pose is rescored with the exact analytic pair sum in the same
    launch. The production path. Shape only, for now."""

    ANALYTIC = 1
    """The exact analytic optimizer throughout. The Tier-3 specification, the
    path that carries color, and how ground truth is generated. Slower."""


class CoordResidency(enum.IntEnum):
    """Where a library's coordinates live *between optimizer steps*.

    This is emphatically **not** a level of the Tier-0/1/2/3 filter-and-score
    cascade. ROSHAMBO2 called it ``grid_tier``, which read as though it were,
    and that ambiguity is the single best argument for the renaming pass.

    SHARED, GLOBAL and GLOBAL_LDCG are bit-identical to each other by
    construction: the quantization expression is character-identical and the
    per-lane atom order and accumulation order are unchanged. REGISTER is
    set-identical rather than bit-identical.
    """

    SHARED = 0
    """Quantized once per conformer into a shared-memory record. **The default.**
    Still the fastest tier at L = 16 and L = 24 on the
    narrow half of each bucket -- the streaming tiers win by leaving L1 to the
    field, and at a narrow stride with few real atoms there is not enough field
    pressure for that to pay. Carveout 50 goes with it (:func:`carveout_for`)."""
    GLOBAL = 1
    """Re-read from global every step with an ordinary load. Requests no
    staged shared memory."""
    GLOBAL_LDCS = 4
    """Streamed through ``__ldcs``: allocates an L1 line but marks it
    **evict-first**, so it is the first candidate to go.

    A different bet from ``GLOBAL_LDCG``'s, and the one that fits the SHAPE
    channel: each library atom is read once per step, so the line is worth
    having for the handful of instructions that use it and worth nothing
    afterwards, whereas ``__ldcg`` declines to allocate it at all."""
    GLOBAL_LDCG = 2
    """The same, through ``__ldcg``: caches in L2 and never allocates an L1
    line, leaving L1 to the field. One word of source, no template surface, and
    bit-identical to the staged tier at every bucket checked (192,000,000
    words each). Its win grows monotonically with the padded stride and with the
    real atom count -- 0.895x at L = 16 up to 1.93x at L = 56 on the narrow
    half of each bucket, and 2.05x already at L = 24 on the wide half. **Not the
    default**: see ``ScreenConfig.coord_residency``.
    Carveout 0 goes with it (:func:`carveout_for`)."""
    REGISTER = 3
    """Held in registers for the whole conformer. Only instantiated for a fixed
    set of padded widths; anything else raises."""


class UpdateLanes(enum.IntEnum):
    """Who runs the seven-component Adagrad update.

    **Bit-identical under ``UpdateRule.ADAGRAD_QADD``, and not under
    ``ADAGRAD_EXP``.** This docstring used to say "all bit-identical" flatly, and
    that was true while the additive rule was the only one the field kernel had:
    its seven components are independent, so which lane runs which is a change to
    *where* a value lives and nothing else.

    The exponential map's update is a quaternion product that mixes all four
    components of ``q`` with all three of the tangent step, so it does not
    decompose over lanes at all. ``PER_LANE`` and ``REDUCE_SCATTER`` gather the
    step and run the product redundantly (``apply_update_lanes``); ``ALL`` runs
    the shared ``apply_update``. Same expressions, two register-pressure
    contexts, and ptxas contracts them into FFMAs differently -- confirmed by
    forcing ``quat_left_exp`` out of line, which makes all three agree exactly.

    ``PER_LANE`` and ``REDUCE_SCATTER`` remain bit-identical to each other under
    both rules; only ``ALL`` splits off, and only under the exp map. Measured
    envelope over the seeded panel at every start mode, 25 and 100 steps: mean
    absolute dT 2.6e-5 to 8.4e-5, p99 3.0e-4 to 1.5e-3, top-1 set and order unchanged
    in every cell. That is the measured
    amplification curve for a one-FFMA change, and ``doc/NUMERICS_POLICY.md``
    section 2 predicts it: splitting one pair's work across cooperating lanes and
    recombining it forfeits bit-identity necessarily.

    So this enum selects a *performance* arrangement, and under the exp map two
    arrangements are two measurements rather than one measurement twice. The
    shipped value is ``PER_LANE``.
    """

    ALL = 0
    """Every one of the four cooperating lanes runs all seven components."""
    PER_LANE = 2
    """Component ``c`` on lane ``c % 4``. The default."""
    REDUCE_SCATTER = 3
    """PER_LANE with the butterfly all-reduce replaced by a reduce-scatter."""


class UpdateRule(enum.IntEnum):
    """The optimizer's update. Only the first two are shipping.

    ADAM_EXP, QN2_EXP and BFGS_EXP are compiled but reachable only through
    ``roshambo3.experimental``: one settings dict must not mean different
    arithmetic depending on which kernel it reaches.
    """

    ADAGRAD_QADD = 0
    """The additive Adagrad step on ``q[1:4]`` plus a renormalization.

    The former default; kept as the control that ``bench/xcheck_roshambo2.py``
    and ``tests/data/golden_scores.npz`` pin, and reachable on both kernels."""
    ADAGRAD_EXP = 1
    """The same step through the exponential map. **The default**, on
    both the field and the analytic kernel."""
    ADAM_EXP = 2
    QN2_EXP = 3
    BFGS_EXP = 4


class ColorLoop(enum.IntEnum):
    """The structure of the color pair loop.

    ``SAME_TYPE_NOW`` is the default and was measured as the winner. The
    other three are kept reachable because they are the arms the choice rests
    on, and because a future non-canonical interaction matrix needs ``BOUNDED``.
    """

    MASKED_PADDED = 0
    """All pairs over the padded width, masked. The former default, and the
    last known instance of padding cost -- its cost grew as ``(L - n_real)^2``
    over pure padding.

    **Retired and refused, not instantiated.** It was the only structure
    that needed a loop reaching into the padding, and therefore the last thing
    on the library side that read the 0/1 mask in the 4th coordinate component.
    ``BOUNDED`` reproduced it BITWISE on the canonical
    matrix -- 245,760 of 245,760 score words, and every block of a
    200,000-conformer color corpus -- so deleting it loses no reachable number.

    The value is kept rather than renumbered so that a stale 0 in an archived
    script fails loudly instead of quietly selecting a different structure."""
    BOUNDED = 1
    """The same all-pairs form, bounded to the real color slots, and
    nothing else. The only structure that is correct for a
    NON-CANONICAL interaction matrix, because it is the only one that evaluates
    a cross-type pair. It is also the reference the other structures
    are checked against, in ``MASKED_PADDED``'s place."""
    SAME_TYPE = 2
    """Walk the library's color slots; for each, run only the query's segment of
    that same feature type. Requires the canonical matrix."""
    SAME_TYPE_NOW = 3
    """SAME_TYPE with the weight multiply dropped. **The default.**

    Every surviving structure is run-
    bounded off the count vector and none can reach one, so ``BOUNDED`` and
    ``SAME_TYPE`` drop the multiply too and this is the same code as
    ``SAME_TYPE``. Collapsing the two is a follow-up."""


class ColorPlacement(enum.IntEnum):
    """Where the color term is evaluated.

    A runtime argument on purpose: the placement
    choice does not foreclose combo-optimal optimization, because choosing
    ``AT_POSE`` is choosing an argument and not a structure.
    """

    IN_GRADIENT = 0
    """In every gradient call, so the pose is optimized against the mixed
    objective: mode 3, combo-optimal. The former default, at
    ``N_starts * (steps + 1)`` = 104 color evaluations per pair."""
    AT_POSE = 1
    """The pose is optimized on shape alone and the color term is evaluated once
    per start, at the converged pose: mode 2, and what
    ``doc/ARCHITECTURE.md`` describes. Four evaluations."""


#: Padded widths for which REGISTER residency is instantiated.
#:
#: This is the host-side mirror of the ladder's ``apl_`` list -- ``L = apl * WA``
#: at ``WA = 4`` -- and the two must move together, because the kernel's
#: register tile is ``3 * ceil(NmolB/WA)`` floats and there is one instantiation
#: per width. The first nine entries are the widths a SHAPE
#: record reaches.
#:
#: The rest are the widths a MERGED record reaches (the color plane is
#: inside the padded width, so the same molecules pad to 88...136 with color
#: where they padded to 40...88 without it). A build without
#: that block refuses them at the ladder's named throw rather than here, which
#: is the correct order -- this list is a courtesy check, not the authority.
REGISTER_WIDTHS = (16, 24, 32, 40, 48, 56, 64, 72, 88, 96,
                   104, 112, 120, 128, 136)

#: Residency tiers whose kernel requests NO staged shared memory at all.
#: The staged tier's entire shared commitment is the quantized coordinate
#: record -- 12.7 KB at L = 32, 22.2 KB at L = 56 -- and these three ask for
#: none of it.
UNSTAGED_RESIDENCY = frozenset({CoordResidency.GLOBAL,
                                CoordResidency.GLOBAL_LDCG,
                                CoordResidency.GLOBAL_LDCS,
                                CoordResidency.REGISTER})

#: The shared/L1 carveout the staged tier wants. The viable window for a
#: kernel that stages and also reads through L1 is 40-60, and the driver's
#: inferred default sits outside it on this card -- 4.54x for one unset knob.
STAGED_CARVEOUT = 25


def carveout_for(coord_residency: CoordResidency,
                 color_residency: CoordResidency = CoordResidency.GLOBAL,
                 color_stage: int = 0) -> int:
    """The shared/L1 carveout that belongs to a residency tier.

    **The carveout is an attribute of the ARM, not of the kernel family**,
    and ``cudaFuncSetAttribute`` is sticky,
    so "leave it alone" is not a control. Those two facts together make an
    uncoupled pair of defaults a trap rather than a nuisance: a kernel that
    requests **zero** shared memory still has an explicitly-set carveout 50
    honored on sm_120, so half the SRAM is partitioned away as shared and never
    used. That measured **5.9x** slower on the production kernel, in the direction opposite to the staged arm's preference.

    So the two knobs resolve together here, in one function both entry points
    call, rather than in a comment telling each caller to remember. An
    *explicit* ``smem_carveout`` is still honored -- the
    carveout must stay re-sweepable per arm -- but a caller who moves the residency tier and says nothing
    about the carveout gets the one that goes with it.
    """
    # The carveout is an attribute of the ARM, and a launch can request shared memory for its COLOR coordinates while
    # streaming its shape ones. A streaming shape tier beside a staged color
    # tier is still a launch that stages, so it wants the staged carveout --
    # reading the shape tier alone would partition zero shared to a kernel that
    # asks for 12 KB of it, the same trap in the other direction.
    stages = (CoordResidency(coord_residency) not in UNSTAGED_RESIDENCY
              or (CoordResidency(color_residency) is CoordResidency.SHARED
                  and int(color_stage) > 0))
    return STAGED_CARVEOUT if stages else 0

#: Block width the shipped analytic cell stages the library at. Not a
#: universal default: staging is shape-only, so a color launch has none.
DEFAULT_STAGE_WIDTH = 256

#: The shipped staged library record: three int16 planes (x, y, z), the weight
#: plane dropped. Named rather than folded into a bytes-per-atom constant
#: because the two halves move independently -- the float32 record was four
#: planes of four bytes, and narrowing the record again is live work -- and a
#: single "6" cannot say which of the two changed.
STAGED_PLANES = 3
STAGED_ELEMENT_BYTES = 2

#: The shipped shared QUERY record: the same three int16 planes.
#: The weight plane is not stored because every loop that could read it is
#: bounded by the real atom counts, where it is exactly 1.0f.
QUERY_PLANES = 3
QUERY_ELEMENT_BYTES = 2


def analytic_block_shared_bytes(
        query_atoms: int, library_atoms: int, stage_width: int, *,
        n_features: int = 1, planes: int = STAGED_PLANES,
        element_bytes: int = STAGED_ELEMENT_BYTES,
        interaction_diagonal_only: bool = True) -> int:
    """Shared memory one analytic block requests, in bytes.

    This is ``optimize_pairs_analytic``'s own ``sharedMemSize``, term for term
    and in its order (``_cuda_src/kernels.cu``, the sum just above the opt-in
    limit check). It exists so that **one** expression decides both what the
    launcher asks the driver for and what a caller may ask the launcher for.

    The launcher used to be the only place this sum was written, and
    ``engine.config.stage_width_for`` compared a *different, smaller* quantity
    -- the staged buffer alone -- against the same device limit. Everything the
    block puts beside the staged library (the query's xyzw copy, the per-block
    cache, the query type ids and the interaction tables -- 488 B at a 22-atom
    shape query then, 208 B after the trim) went uncounted, so a staged
    buffer that exactly filled the budget was accepted by the policy and
    refused by the launcher. On a Volta
    card the opt-in limit is 98,304 B and ``256 x 64 x 6`` is 98,304 B exactly,
    so a shape-only screen of any corpus with an L = 64 bucket died at the
    shipped defaults.

    Being byte-exact matters, not merely conservative: the launcher's check is
    ``>``, so a policy that over-estimated would step down a width that in fact
    fits, and one that under-estimates by a byte crashes.

    ``query_atoms`` is the launcher's ``NmolA`` -- the query panel's PADDED
    width, ``query_coords.shape[1]``, color plane included -- and
    ``n_features`` is ``interaction_r.shape[0]``. ``library_atoms`` is the
    bucket's padded width ``L``.

    **Three of the four query terms are trimmed.** The query's coordinates are
    three int16 planes rather than four float32 ones (the weight was always
    exactly 1.0f wherever a loop could read it); its type ids are ``u8`` rather
    than ``int32``, because a type id is 0..7; and the interaction matrices are
    staged **diagonal only** where the color loop reads only the diagonal, which
    is the shipped structure -- 2*F floats rather than 2*F*F, i.e. 56
    bytes rather than 392 at F = 7. 920 B -> 304 B at a 22-atom query.

    ``interaction_diagonal_only`` mirrors the kernel's
    ``CLOOP >= COLOR_LOOP_SAME_TYPE``, which the launcher derives from the same
    ``color_loop`` it dispatches on. It defaults to True because that is what
    ships; pass False for a below-SAME_TYPE arm.
    """
    q = int(query_atoms)
    f = int(n_features)
    rp = f if interaction_diagonal_only else f * f
    fixed = (8 * 4                                  # the per-block cache
             + 2 * rp * 4                           # r and p
             + 2 * f * 4                            # the color segment table
             + q * QUERY_PLANES * QUERY_ELEMENT_BYTES   # xyz, int16
             + q * 1)                               # the type ids, u8
    # The staged library starts 16-byte aligned; the kernel walks the same
    # offsets (`analytic_query_block_bytes` in kernels.cu).
    fixed = (fixed + 15) & ~15
    stage_bytes = (int(stage_width) * int(library_atoms)
                   * int(planes) * int(element_bytes))
    return fixed + stage_bytes

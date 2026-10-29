"""Screen configuration: small, validated, and all shipping.

Nothing here is a measurement arm. ROSHAMBO2's equivalent entry point reached 20
keyword arguments and 11 method codes because every experiment added one and
none were ever removed; arms live in ``roshambo3.experimental`` instead, where
they cannot accrete on the surface a user reads.

The one coupling that must not be broken: ``steps`` and ``lr_t`` are a
single knob, so ``lr_t`` is derived from ``steps`` unless both are given, and a
pair off the measured budget curve is refused. A step-count change that does not
carry its ``lr_t`` is not a speedup, it is a silent accuracy loss -- and it looks
exactly like a speedup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from ..backend.spec import (ColorLoop, ColorPlacement, CoordResidency, PoseSearch,
                            STAGED_ELEMENT_BYTES, STAGED_PLANES, UpdateLanes,
                            UpdateRule, analytic_block_shared_bytes, carveout_for)
from ..optimizer import resolve_optimizer_settings

#: Coarsest lattice spacing that passes the accuracy gate, by padded width.
#: Accuracy wants coarser and residency wants finer, and they land one notch
#: apart at every bucket, so there is no single h that satisfies both -- which is
#: why this is a per-bucket policy and not a constant. Largest key <= L wins.
FIELD_SPACING_BY_WIDTH: Mapping[int, float] = {
    16: 0.4, 24: 0.4, 32: 0.5, 40: 0.6, 56: 0.6,
}

#: Register budget by padded width, as `__launch_bounds__` min-blocks:
#: 4 -> 64 registers, 3 -> 85, 2 -> 128, 1 -> 255.
#:
#: **This table reaches the kernel only under ``CoordResidency.REGISTER``.** The
#: launch macro for tiers 0-2 is pinned at ``MINB = 4``
#: (``ROSHAMBO3_FIELD_LAUNCH``), so at the shipped residency it is inert.
#:
#: Measured on the shape-only field kernel, the L = 56 entry is beaten 1.66x by
#: ``MINB = 2`` (479.0 ms against 289.2).
#:
#: **The table is still inert at the shipped residency, and the L = 56 entry
#: stays at 1.** The register tier measured 0.678-0.737x whole-screen on a
#: color wave -- so
#: the tier this table exists for is not a candidate, and refitting an entry that
#: only it reads would be tuning a path nothing selects. ``MIN_BLOCKS_BY_WIDTH``
#: reaches the kernel only under ``CoordResidency.REGISTER``.
#:
#: The shape-only ``MINB`` ranking also **inverts** with color present:
#: ``MINB = 2`` beats ``MINB = 4`` by 8.6% there, because 128 registers cut the
#: spill that 64 cannot hold. So the shape-only table was not merely fitted to an
#: operating point that did not exist -- it was fitted to the opposite one.
MIN_BLOCKS_BY_WIDTH: Mapping[int, int] = {16: 4, 24: 4, 32: 4, 40: 4, 48: 3, 56: 1}

#: Fallback shared-memory budget, used only when no device can be asked.
#:
#: 48 KB is the amount every CUDA architecture gives a block without the opt-in
#: attribute, so it is the one value that is never wrong. The real limit varies
#: by a factor of three and a half -- 64 KB on Turing, 99 KB on sm_86/89/120,
#: 163 KB on sm_80, 227 KB on sm_90 -- which is why this used to be hardcoded to
#: 99 KB and was therefore correct on exactly one family. On Turing that policy
#: would ask for 96 KB at L = 64 and the launcher would refuse the launch: a
#: clear error rather than a wrong answer, but an avoidable one.
_SMEM_FALLBACK = 48 * 1024

#: Bytes per staged atom: three int16 planes, x/y/z only. Kept as one number
#: because callers describe the record that way; the launcher's own sum takes
#: the planes and the element size separately (``analytic_block_shared_bytes``).
_STAGED_BYTES_PER_ATOM = STAGED_PLANES * STAGED_ELEMENT_BYTES

#: Block widths the analytic kernel is instantiated for, widest first. 0 means
#: "do not stage" -- re-read the library from global each step instead.
#:
#: **The rungs are a menu; the selection is the device's.** ``stage_width`` is a
#: template parameter of ``optimize_pairs_analytic_kernel`` and also the block
#: dimension, so it cannot be an arbitrary runtime integer -- every width here
#: costs a compiled instantiation. What must not be a constant is the *choice*,
#: and it is not: ``stage_width_for`` asks the device for
#: ``cudaDevAttrMaxSharedMemoryPerBlockOptin`` and takes the widest rung whose
#: whole block fits inside it. So the ladder is per-device by construction; what
#: this tuple decides is only how finely the boundary can be approached.
#:
#: The rungs used to be ``(256, 128, 0)``, two factors of two apart, and a
#: boundary that falls between two rungs costs the whole factor. Measured
#: exactly, per bucket, on both architectures:
#:
#:   * **sm_70, L = 64: 253 fits and the old ladder delivered 128.** The staged
#:     buffer alone at width 256 is ``256 x 64 x 6 = 98,304 B``, which is the
#:     Volta opt-in limit *exactly*, so any other byte in the block -- and there
#:     were 920 of them at the time and are 304 since the query trim --
#:     forced the drop. Invisible, because nothing errors.
#:   * **L = 128 and 136 got no staging at all on sm_70** (136 on sm_120):
#:     128 does not fit, and the next rung was 0.
#:   * Over a 10M-conformer color corpus's buckets the old ladder lost width
#:     on **46.7% of conformers on sm_70 and 32.6% on sm_120**.
#:
#: Why multiples of 32 and not the exact boundary (224 rather than 253): the
#: launcher chunks work items to ``chunk = stage_width`` (``interface.cpp``), so
#: a width that is not a whole number of ``ROSHAMBO3_BTILE = 32`` tiles starts
#: every work item off a tile boundary and every warp then reads across two
#: tiles. That spends the coalescing the tile already buys on a few percent of
#: width, which is the wrong trade. With this granularity the ladder attains the
#: multiple-of-32 arithmetic bound at **every** bucket on both architectures.
#:
#: **Every** multiple of 32 up to 256 is a rung, not only the ones the corpora
#: in hand select. Each costs 5 compiled kernels (one per update rule), and the
#: first draft pruned 192, 64 and 32 on the grounds that no bucket in the
#: color corpus reaches them. That was wrong twice over: **L = 80 selects
#: 192 on both sm_70 and sm_120**, and it is a real bucket edge that corpus
#: simply has no conformers in; and 64 and 32 are what a 48 or 64 KiB budget --
#: Turing, or the no-device fallback -- needs at the wide buckets. A missing
#: rung is never wrong, only narrow -- an invisible loss.
STAGE_WIDTHS = (256, 224, 192, 160, 128, 96, 64, 32, 0)

#: Padded query width the staging policy assumes when a caller does not name
#: one.
#:
#: The launcher's shared-memory requirement is a function of the QUERY as well
#: as the library bucket (7 B per padded query atom, 20 B
#: before it), so a policy asked only
#: for ``L`` is answering a question it has not been given. ``Engine.run``
#: always names the panel's real width, so this default is reached only by
#: direct callers -- and it is a bound rather than a typical value: no prepared
#: corpus in ``$ROSHAMBO3_CORPORA`` has a bucket wider than this, and against
#: every even bucket width it selects the same stage width as the true 22-atom
#: query does on both a 96 KiB and a 99 KiB budget (the widths where the two
#: answers could differ are L = 63, 65 and 125-131, none of them a bucket).
DEFAULT_QUERY_ATOMS = 128


_smem_budget_cache: dict[int, int] = {}


def shared_memory_budget(device_id: int = 0) -> int:
    """Opt-in shared memory per block on ``device_id``, in bytes.

    Asked of the device, cached, and falling back to :data:`_SMEM_FALLBACK` when
    there is no device to ask -- which is the honest thing to do, since the
    consequence of guessing high is a refused launch on hardware we are not
    running on.
    """
    if device_id in _smem_budget_cache:
        return _smem_budget_cache[device_id]
    budget = _SMEM_FALLBACK
    try:
        from ..backend.cuda import extension
        budget = int(extension().device_info(device_id)["shared_memory_per_block_optin"])
    except Exception:
        pass
    _smem_budget_cache[device_id] = budget
    return budget


#: Opt-in shared memory per block, in bytes, on the two architectures this
#: project has measured on. Used to compute the query bound off-device --
#: exact arithmetic, no GPU -- so a limit can be quoted without one to hand.
#: Verified values, not nameplate: sm_70's is the number hit exactly
#: (``256 x 64 x 6 == 98,304``), and sm_120's is the measured
#: ``shared_per_sm_optin``.
MEASURED_SMEM_OPTIN = {"sm_70": 98_304, "sm_120": 102_400}


def query_atom_limit(budget: int, *, n_features: int = 1,
                     interaction_diagonal_only: bool = True) -> int:
    """The largest padded query width a block on ``budget`` bytes can hold.

    **The bound, computed rather than discovered.** A query has no
    ``uint8`` sidecar and none of ``prep.build._why_reject``'s library limits
    apply to it -- copying ``MAX_N_REAL = 255`` across would be a wrong bound
    that happens to look principled. What actually bounds a query is shared
    memory, and otherwise the only thing that says so is the launcher, at
    screen time, after a corpus has been opened.

    This inverts :func:`~roshambo3.backend.cuda.analytic_block_shared_bytes` at
    its **most favorable configuration** -- ``stage_width = 0``, the unstaged
    rung at the bottom of ``STAGE_WIDTHS``, where the staged library costs
    nothing. So it is the bound below which *no* configuration on this device
    can run the query, which is the only bound that can be checked before the
    library is known. A query inside it may still be refused later by a
    specific (bucket, stage width) pair; a query outside it cannot be screened
    at all.

    Deliberately a function of ``budget`` and not a constant. The stage-width
    ladder once had exactly that defect, where a constant fitted to one card
    cost the other a factor of two.
    """
    from ..backend.spec import (QUERY_ELEMENT_BYTES, QUERY_PLANES,
                                analytic_block_shared_bytes)
    # The fixed terms that do not scale with the query, taken from the same
    # expression the launcher uses so the two cannot drift.
    zero_q = analytic_block_shared_bytes(
        0, 0, 0, n_features=n_features,
        interaction_diagonal_only=interaction_diagonal_only)
    per_atom = QUERY_PLANES * QUERY_ELEMENT_BYTES + 1
    if int(budget) <= zero_q:
        return 0
    # `fixed` is rounded up to 16 bytes before the staged library is added, so
    # solving on the unrounded sum and flooring is correct and conservative by
    # at most one atom.
    return max(0, (int(budget) - zero_q) // per_atom)


#: The launch settings whose dispatch cells this build instantiates at one
#: value only, and that value. A setting at its shipped value is in the build
#: by construction; a setting away from it has no cell, and the launcher's
#: `!launched_` ladder would throw.
#:
#: Keyed on the names ``for_width()`` emits, so this table and a run record's
#: ``config`` block are talking about the same thing. A ``None`` value is a
#: setting resolved per path, which this table does not check.
#: ``coord_residency`` is checked separately: only SHARED is instantiated.
SHIPPED_LAUNCH_SETTINGS: dict = {
    "lanes_per_group": 4,
    "start_groups_per_row": 1,
    "update_lanes": 2,
    "stream_min_blocks_per_sm": None,
    "color_stage": 0,
    "color_residency": 1,
    "interleave": 0,
}


def unsupported_cells(settings: dict, *,
                      bucket: Optional[int] = None) -> list[str]:
    """Resolved settings this build carries no dispatch cell for.

    **The check at config resolution, not at launch.** The
    launcher's ``!launched_`` throw is exact and names every knob, and it is
    the right object for a developer changing kernel settings. It is also a
    wall of template parameters, and since the build compiles only the
    shipped configuration's cells the thing it can meet is *a bucket no cell
    covers, mid-screen*. A throw during a production run is worse than a
    refusal before it.

    So this runs first and says the same thing in the caller's vocabulary --
    which bucket, which setting, which value. **It does not replace the
    throw**: the ladder is still the backstop, and a request off an axis this
    table does not model still reaches it. Returns reasons rather than
    raising, so a caller can decide whether a resolved setting is worth
    refusing over.
    """
    where = "" if bucket is None else f"bucket {bucket}: "
    out = []

    # The shipped FIELD combo chain is compiled unconditionally, including
    # SHARED color / CSTAGE 48 (see kernels.cu, "COMBINATION cells"). Compare
    # the complete chain guard: exempting any isolated stage=48 setting would
    # incorrectly admit other schedules that have no cells.
    shipped_combo = (
        settings.get("pose_search") == PoseSearch.FIELD
        and bool(settings.get("color"))
        and settings.get("color_placement") == ColorPlacement.IN_GRADIENT
        and settings.get("coord_residency") == CoordResidency.SHARED
        and settings.get("lanes_per_group") == 4
        and settings.get("update_lanes") == UpdateLanes.PER_LANE
        and settings.get("update_rule") in (UpdateRule.ADAGRAD_QADD, UpdateRule.ADAGRAD_EXP)
        and settings.get("start_groups_per_row") in (None, 1, 4)
        and settings.get("stream_min_blocks_per_sm") in (None, 2)
        and settings.get("color_residency") == CoordResidency.SHARED
        and settings.get("color_stage") == 48
        and settings.get("interleave") == 0
        and settings.get("query_color_residency") == CoordResidency.SHARED
        and not settings.get("color_ballot", False)
        and settings.get("color_loop") in (None, ColorLoop.SAME_TYPE_NOW))

    residency = settings.get("coord_residency")
    if residency is not None:
        r = int(residency)
        if r in (1, 2, 3, 4):
            out.append(
                f"{where}coord_residency={r} is not supported by this build; "
                f"leave coord_residency at SHARED.")

    for name, shipped in SHIPPED_LAUNCH_SETTINGS.items():
        if shipped_combo and name in ("color_stage", "color_residency", "start_groups_per_row"):
            continue
        value = settings.get(name)
        if value is None or shipped is None or int(value) == int(shipped):
            continue
        out.append(
            f"{where}{name}={int(value)} is not supported by this build "
            f"outside the shipped combination schedule (supported here: "
            f"{shipped}); leave {name} at {shipped}.")
    return out


def stage_fits(L: int, stage_width: int, budget: int, *,
               query_atoms: int = DEFAULT_QUERY_ATOMS,
               n_features: int = 1) -> bool:
    """Will the analytic launcher accept ``stage_width`` for this launch?

    Asked of the launcher's own shared-memory sum rather than of the staged
    buffer alone. The distinction matters: the two differ by
    everything the block puts beside the staged library, and where the staged
    buffer alone exactly fills the budget that difference is the difference
    between a screen and a ``RuntimeError``.
    """
    return analytic_block_shared_bytes(
        query_atoms, L, stage_width, n_features=n_features) <= budget


def widest_stage_that_fits(L: int, budget: int, *,
                           query_atoms: int = DEFAULT_QUERY_ATOMS,
                           n_features: int = 1) -> int:
    """The selection rule: the widest instantiated width the launcher accepts.

    The staged library is a *per-thread* shared allocation, so its footprint is
    ``width * L * 6`` bytes and grows with the bucket. At L = 80 a 256-wide
    block needs 120 KB against the 99 KB a block may have, so a fixed 256 is not
    merely suboptimal for wide buckets -- the launch is refused. This is why the
    stage width is a per-bucket policy and not a constant, exactly as the
    lattice spacing and the register budget are.

    **This rule is a fit test, not an optimum**, and it is a separate function
    from :func:`stage_width_for` so that it can be replaced by one with a
    measurement behind it without touching the arithmetic or the budget
    plumbing. What it does not consider: shared memory and L1 are one physical
    SRAM split by the carveout, so staging at the widest width that fits
    partitions the whole budget away from the L1 the same launch still reads
    through. ``backend.cuda.carveout_for`` carries the measured precedent in
    the opposite direction -- **5.9x** measured from SRAM partitioned as shared
    and never used -- and ``STAGE_WIDTHS`` already ends in ``0``, so "stream
    from global" is an answer this rule can reach only when nothing fits, never
    because it is faster. Whether that is the right objective at large ``L`` is
    unmeasured, and is not this function's question.
    """
    for w in STAGE_WIDTHS:
        if w == 0 or stage_fits(L, w, budget, query_atoms=query_atoms,
                                n_features=n_features):
            return w
    return 0


#: The staging rule in force. Named and indirected so that a measured rule can replace the
#: objective in one place; every caller goes through ``stage_width_for``.
STAGE_WIDTH_POLICY = widest_stage_that_fits


def stage_width_for(L: int, budget: Optional[int] = None,
                    device_id: int = 0, *,
                    query_atoms: int = DEFAULT_QUERY_ATOMS,
                    n_features: int = 1) -> int:
    """Stage width for a bucket of padded width ``L``, per :data:`STAGE_WIDTH_POLICY`.

    Resolves the budget -- from ``budget`` if given, otherwise from the device --
    and delegates. ``query_atoms`` and ``n_features`` describe the QUERY panel
    the launch will carry (``query_coords.shape[1]`` and
    ``interaction_r.shape[0]``); they are part of what the launcher charges
    against the same budget, so a policy that does not take them is answering a
    smaller question than the one it is asked. ``Engine.run`` supplies both;
    :data:`DEFAULT_QUERY_ATOMS` documents what a caller who does not gets.
    """
    if budget is None:
        budget = shared_memory_budget(device_id)
    return STAGE_WIDTH_POLICY(L, budget, query_atoms=query_atoms,
                              n_features=n_features)


def _by_width(policy: Mapping[int, float], L: int, default):
    """Largest key <= L wins; ``default`` if none does."""
    best, val = None, default
    for k, v in policy.items():
        if k <= L and (best is None or k > best):
            best, val = k, v
    return val


def _default_cpu_workers() -> int:
    """Respect process affinity; SMT siblings are usable independent workers."""
    from .._cpu_platform import available_cpu_count
    return available_cpu_count()


@dataclass(frozen=True)
class CPUWorkerOptions:
    """Explicit worker-owned CPU shape, rescore or combo top-K contract.

    The source validates the numbering of color queries. AT_POSE rescores
    exactly once at the shape-best winning start; IN_GRADIENT retains mode-3
    combo search. ``self_overlap_policy="prepared"`` reuses verified stored
    shape VAA (converted to native units) and once-prepared current-R/P color
    self terms. ``"recompute"`` explicitly keeps exact per-block recomputation. Query self contexts belong to the Engine
    and are reused across runs while their complete binding remains valid.
    The scoring tuple is analytic color16 with pipelined FP32 field shape8.
    The optimizer-color controls are ``None`` by default and resolve once per
    worker scan: combo enables canonical torque factoring, the dedicated
    canonical grain16 helper, and the faster unclamped raw-bits optimizer
    exponential; shape/rescore keep these controls disabled and use the
    baseline exponential. Raw bits changes canonical optimizer trajectories
    only; exact finals and the native general-matrix fallback are retained.
    Set ``optimizer_exp_mode="baseline"`` to opt out of approximate exp while
    leaving unspecified exact canonical optimizations enabled.
    ``optimizer_exp_mode="rawbits"`` explicitly retains the input clamp.
    The default ``"rawbits_unclamped"`` omits that clamp with the same tail mask.
    ``final_start_policy="winner"`` chooses the actual converged SEARCH
    objective, then accurately scores only its winner (shape criterion for
    shape/rescore; mixed criterion for combo). ``"all_start"`` retains accurate
    all-start selection. This does not change optimizer exponential, self terms
    or cached query geometry. Unsafe/unsupported combo routes retain all-start
    selection. Accurate analytic finals remain default.
    Once at native call entry it falls back to guarded raw bits if FP invalid
    exceptions are unmasked, AVX2 is unavailable, or internal native workers/an
    executor are used. It never changes FP control words and may set the invalid
    status flag. Public workers use one native thread per block.
    Explicit ``False`` values disable either exact optimization. The canonical helper
    is independent of the older invariant-plan flag and does not affect
    general/field dispatch. None of these controls changes cached query
    geometry. Explicit nondefault optimizer-color requests are rejected
    outside combo. Low-level ``run_worker_owned_cpu`` defaults remain
    conservative and explicit. ``shape_field_interpolation="auto"`` currently
    selects FP32, including the measured AArch64 prepared-field route. Use
    ``"fp32"`` to pin that arithmetic explicitly. Low-level half requests
    remain available with native runtime-feature and finite-range checks.
    Memory budgets default to ``None`` (no configured cap). An explicit positive
    byte limit enables conservative preflight checks, not an RSS cap or an
    allocation request. Top-K storage remains bounded by workers, queries and K;
    native overflow checks and allocation failures still apply without budgets.
    Omitted ``workers`` uses the process's available logical CPUs; explicit
    counts remain authoritative. The screen facade's worker argument fills an
    omitted options count before calling Engine.
    ``cpu_qos="inherit"`` preserves ordinary scheduling. On macOS,
    ``"background"`` and ``"user_initiated"`` set both the submitting thread
    and compute workers; they are scheduler hints, not P/E-core pinning.
    Explicit policies are refused on other operating systems. Worker policy
    and read-back QoS are recorded in ``stats.cpu_scan["worker_placement"]``.
    """

    self_overlap_policy: str = "prepared"
    workers: Optional[int] = None
    memory_budget_bytes: Optional[int] = None
    topk_state_budget_bytes: Optional[int] = None
    factor_canonical_torque: Optional[bool] = None
    optimizer_exp_mode: Optional[str] = None
    canonical_color_kernel: Optional[bool] = None
    shape_field_interpolation: str = "auto"
    final_start_policy: str = "winner"
    cpu_qos: str = "inherit"


@dataclass
class ScreenConfig:
    """How to score. Defaults are the shipped configuration."""

    pose_search: PoseSearch = PoseSearch.FIELD
    color: bool = False
    mixing: float = 0.0
    """0 = shape, 1 = color, in between = a mixed objective."""

    steps: int = 25
    lr_t: Optional[float] = None
    """None derives it from ``steps``. Give both only with a reason."""
    lr_q: float = 0.1
    start_mode: int = 1
    """1 = identity plus 180 degrees about x, y, z: four starts."""
    update_rule: UpdateRule = UpdateRule.ADAGRAD_EXP
    """The exponential map, the default on both kernels.

    This is a SECOND default and it is the one engine callers get:
    ``resolved_optimizer`` always passes ``opt_mode`` explicitly, so
    ``optimizer.DEFAULT_OPT_MODE`` never fills it in from here. Flipping one and
    not the other would leave the default unchanged in practice. The argument
    and the measured recall tie that makes it safe are in
    ``optimizer.DEFAULT_OPT_MODE``."""
    budget_override: Optional[str] = None
    """A stated reason for an off-curve (steps, lr_t) pair, or refusal."""

    # --- the field path ---
    field_spacing_a: Optional[float] = None
    """None uses the per-bucket policy. A scalar pins one spacing everywhere."""
    field_pad_a: float = 8.0
    coord_residency: CoordResidency = CoordResidency.SHARED
    """Where a library's coordinates live between optimizer steps.

    **SHARED, and the register tier is now closed rather than deferred.** All
    four tiers were measured on a SHAPE-only wave and REGISTER won at 1.318x
    whole-screen, winning or tying every bucket; it was not shipped, on the
    argument that the register file is what analytic color contends for and a
    default fitted without color would be fitted to an operating point that did
    not yet exist. That operating point was then built and measured. The caution
    was right and the mechanism was not.

    REGISTER does not out-compete color for the register file. It is given LESS
    of it: ``__launch_bounds__(256, 4)`` pins the tier at 64 registers per thread
    where the shipped color cell holds 124-127, and its coordinate tile is
    ``3 * ceil(NmolB / lanes_per_group)`` floats over the MERGED record's padded
    width, which carries the color plane. On a 10M-conformer color corpus that tile
    is 2.16x larger conformer-weighted than the shape plane needs, and it does
    not fit: ALL 168 tier-3 color instantiations spill, 304-960 bytes of stack
    per thread against the shipped cell's flat 32. A register-resident tier that
    spills its tile is a global-memory tier with extra steps.

    Measured whole-screen on sm_70, 9,999,998 conformers x 32 queries, advisory:
    0.737x at MINB 2 and 0.678x at MINB 4 against what shipped, and 0.754x /
    0.694x against the staged-color default. Staging color beside it changes
    nothing worth reporting.

    What would reopen it is one change and it is a real one: size the register
    tile by the SHAPE width rather than the padded one. The tile is filled only
    for ``j < NB`` -- the conformer's real shape atom count -- so every slot past
    ``ceil(L_shape / lanes_per_group)`` is allocated, zero-filled, and skipped by
    the atom loop's own predicate. ``APL`` is a template parameter keyed on
    ``NmolB`` in the dispatch; keying it on the launch's shape width is a change
    to the dispatch and not to the arithmetic, and it would cut the tile 2.16x.
    Until that exists, this stays SHARED.

    What does ship is the carveout coupling below, which is a footgun fix
    and true regardless of color.

    What the measurement rules OUT is a flat ``GLOBAL_LDCG`` default: it is
    0.895x at L = 16 and 0.860x at L = 24 against what ships, which is 47% of
    the corpus's conformers, and 1.14x over a whole screen. Shipping
    ``__ldcg`` is per-bucket or it is nothing.

    **Measured jointly with color, the default still does not move,
    because the answer turned out to be per-MODE rather than per-bucket.** On
    the combination objective ``GLOBAL`` with ``start_groups_per_row=2`` is the
    winning cell -- 3,407 ms against the shipped cell's 7,597 at 25 steps on the
    10M corpus -- but the same cell makes a SHAPE-ONLY screen slower (2,097 ms
    against 1,810): warp supply is not the
    shape kernel's limit, so nothing that buys warps helps it. A caller running
    mode 3 should name ``CoordResidency.GLOBAL`` and ``start_groups_per_row=2``;
    a caller running mode 1 should not."""
    min_blocks_per_sm: Optional[int] = None
    """None uses the per-bucket policy.

    Reaches the kernel only under ``CoordResidency.REGISTER``: the launch macro
    for tiers 0-2 is pinned at ``MINB = 4``, so the policy below is dispatched
    for the register tier alone."""
    smem_carveout: Optional[int] = None
    """The shared/L1 split. ``None`` resolves from ``coord_residency``.

    ALWAYS set explicitly on the launch: ``cudaFuncSetAttribute`` is sticky, so
    "unset" is not a control -- a zero-shared-memory kernel that inherits
    carveout 50 was measured 5.9x slower. What ``None``
    means here is therefore *not* "leave it alone"; it means "the value that
    goes with the residency tier", resolved by
    ``roshambo3.backend.cuda.carveout_for`` and passed on every launch. Naming a
    value is still honored, so the carveout stays re-sweepable per arm."""
    update_lanes: UpdateLanes = UpdateLanes.PER_LANE
    #: `S`, start-per-lane. How many lane groups share one library row
    #: and split its start orientations between them. 4 is selected for the
    #: color mode-3 field path; this is separate from the historical shape-only
    #: operating point. Field path only; a power of two with 4*S <= 32, and no
    #: larger than the start mode's start count.
    #:
    #: Swept with color present, ``S = 2`` was worth **1.81-1.88x on the combination objective**
    #: having bought nothing at all on shape -- its SMEM saving is budget, and
    #: mode 3 is the first arm with something to spend it on. Round 3's
    #: cross-architecture selection finds the streamed color path's best
    #: complete operating point at ``S = 4``.
    start_groups_per_row: Optional[int] = None

    # --- the analytic path ---
    stage_width: Optional[int] = None
    """Block width for staging the library in shared memory. None picks the
    widest that fits the bucket; 0 disables staging."""

    color_loop: Optional[ColorLoop] = None
    """The color pair loop's structure.

    ``None`` resolves in ``score_pairs`` to the structure the interaction matrix
    admits: the winner for the canonical diagonal matrix, ``BOUNDED`` for any
    other, because the same-type structures do not evaluate a cross-type pair.
    Naming one explicitly is honored or refused, never silently substituted."""
    lanes_per_group: int = 4
    """`WA`, the lanes cooperating on one row's atom loop.

    `S` wins by halving the shared
    memory per unit of work, and `WA` is the other lever on that same quantity.
    A power of two with ``lanes_per_group * start_groups_per_row <= 32``, which
    is what keeps the cross-group best an in-warp reduction."""
    color_residency: CoordResidency = CoordResidency.SHARED
    """Where the **library color** coordinates live between
    optimizer steps.

    Its own axis, separate from shape residency. There are four sets of atoms in this kernel and
    ``coord_residency`` governs exactly one of them -- library SHAPE. Library
    color had no policy at all: a plain global dereference in the color loop
    that never went through ``COORD_FETCH``, so it bypassed every tier, at the
    global stride, for every atom pair, on every step.

    That is the set with the **strongest reuse argument**. Shape reads each
    library atom once per step and has nothing to protect; color makes
    O(#query atoms in this plane) passes over the same slot inside one call and
    then reads it again next step. On this axis ``__ldcs`` should LOSE and
    ``SHARED`` or ``REGISTER`` should WIN, which is the opposite of what the
    shape channel wants; if it does not, the reuse argument was wrong.

    ``SHARED`` and ``REGISTER`` are sized by ``color_stage``."""
    color_stage: int = 48
    """The fixed number of library color slots per row that
    ``color_residency`` SHARED and REGISTER hold, with a global fallback for the
    tail past it. The per-conformer color total is mean 18.88 and max
    63, so a fixed count with a spill path is the shape the distribution asks
    for rather than a compromise."""
    query_color_residency: CoordResidency = CoordResidency.SHARED
    """Where query-color coordinates are read in mode 3.

    ``SHARED`` is the existing full-query staging path. The global policies
    stage only the shape-width prefix needed by exact shape rescoring and read
    color features from the original query record with the selected cache hint.
    Query-register duplication is resource-priced separately and refused when
    it cannot launch."""
    color_ballot: bool = False
    """Warp-vote skip for absent query color segments.

    Off by default. This is a production-kernel causal arm, not a probe; it is
    measured only after the residency/accumulator sweep."""
    stream_min_blocks_per_sm: Optional[int] = None
    """``MINB`` on the streaming and staged residency tiers.

    ``2`` is the selected streamed-field minimum. Its OWN argument rather than a widening
    of ``min_blocks_per_sm``, which reaches the kernel only under
    ``CoordResidency.REGISTER``: widening it would have silently changed the
    shipped shape path at two of nine buckets, since ``MIN_BLOCKS_BY_WIDTH``
    hands 3 at L = 48 and 1 at L = 56 to every launch and those values are
    inert on the streaming tiers today.

    It is the knob that SETS the register count -- ``__launch_bounds__(256,
    MINB)`` divides a 64K file, so 4 is 64 registers per thread and 2 is 128.
    Every color instantiation compiling to exactly 64 is a statement about this
    argument, not about color."""

    interleave: int = 0
    """Experimental combo accumulator and scheduling strategy.

    ``0/1`` use separate shape and color gradient accumulators, ``2/3`` use an
    exact scalar-overlap prepass followed by one weighted gradient accumulator,
    and ``4/5`` use one accumulator with the preceding pose's Tanimoto
    prefactors. Odd values stagger shape/color order between warps. The stale
    arm is an optimizer change; the two-phase arm preserves the real-arithmetic
    objective but permits reassociation. Mode 3 only."""

    color_field_min_sources: Optional[int] = None
    """The query-side feature count a color plane must clear before a
    field is built for it, on ``PoseSearch.FIELD`` with
    ``ColorPlacement.IN_GRADIENT``.

    A field replaces a plane's O(n_query x n_library) pair sum with an
    O(n_library) pass of lookups, so the saving factor for a plane is its QUERY
    count. Over 8.2M molecules: Hydrophobe mean 11.47 and **88.2% of all
    same-type pair work**; the other five planes total ~7.4 atoms and save 2-3x,
    which does not pay for a grid build. So one plane is worth a field and five
    are not, and a threshold is how that is expressed without naming a feature
    type in the kernel.

    ``None`` inherits the backend policy (analytic on CPU/CUDA, cutoff 12 on
    Metal). ``0`` explicitly selects all-analytic color. A served plane is a
    declared numerical policy change, and its gate is not inherited from shape's 0.4 A -- that was fitted over
    ``n_real`` mean 24.57 against 11.47 sources here, at a different Gaussian
    width. The reported color overlap is unaffected either way: it is the exact
    analytic pair sum at the converged pose, and the field is
    used only inside the pose search."""
    color_field_spacing_a: Optional[float] = None
    """Color lattice spacing. None inherits Metal's context; CUDA uses shape spacing."""

    color_placement: ColorPlacement = ColorPlacement.AT_POSE
    """Where the color term is evaluated.

    ``AT_POSE`` is mode 2 and what ``doc/ARCHITECTURE.md`` describes: the
    pose is shape-optimal and the color term is scored once at it.
    ``IN_GRADIENT`` is mode 3, combo-optimal --
    at the shipped ``steps=25, start_mode=1`` that is 104 color evaluations per
    pair where mode 2 makes 4. Mode 3 is one argument away, not one rebuild
    away."""

    cpu_worker: Optional[CPUWorkerOptions] = None
    """Explicit worker-owned CPU top-K route; unset preserves ordinary dispatch.

    Requires FIELD, ADAGRAD_EXP and four starts. Shape-only needs no color
    numbering. Color AT_POSE preserves shape-best-start exact rescoring;
    IN_GRADIENT selects combo. Approximate options affect combo only.
    Engine uses exact TopK/MoleculeTopK/HostTopK objects as result requests,
    without preparing their sink state. Dense/callback sinks are unsupported.
    """

    def resolved_optimizer(self) -> dict:
        """Resolve steps/lr_t through the steps/lr_t coupling, or raise."""
        s: dict = {"steps": int(self.steps), "lr_q": float(self.lr_q),
                   "opt_mode": int(self.update_rule)}
        if self.lr_t is not None:
            s["lr_t"] = float(self.lr_t)
        if self.budget_override:
            s["budget_override"] = self.budget_override
        return resolve_optimizer_settings(s)

    def for_width(self, L: int, device_id: int = 0,
                  L_shape: Optional[int] = None, *,
                  query_atoms: int = DEFAULT_QUERY_ATOMS,
                  n_features: int = 1,
                  library_color_offset: Optional[int] = None) -> dict:
        """Kernel arguments for a launch over a bucket of padded width ``L``.

        Every launch is single-bucket, which is what makes a per-bucket lattice
        spacing and register budget free.

        ``device_id`` is not decoration: the stage width is chosen against that
        device's opt-in shared-memory limit, and the limits differ by a factor
        of two and a half across supported architectures (64 KB on Turing,
        163 KB on Ampere). Resolving a width against device 0 and launching it
        on device 1 does not merely under- or over-shoot -- the launcher
        refuses a request larger than the device allows.

        ``library_color_offset`` is where a library conformer's color run
        begins, in the kernel's convention: ``-1`` for the merged record
        (after each conformer's own real atoms) and an absolute slot for the
        split one. ``None`` means "derive it", which gives the split record's
        ``L_shape`` -- the value that is correct for every split-record corpus.
        It is passed rather than derived because it is a fact about **the block
        that was packed**, and two blocks of the same width can differ in it.

        ``query_atoms`` and ``n_features`` describe the query panel this launch
        will carry. They are not decoration either: the launcher charges the
        query's shared-memory copy and the interaction tables against the same
        budget as the staged library, so the widest stage width that fits is a
        function of both operands. See :func:`stage_width_for`.
        """
        opt = self.resolved_optimizer()
        # The two per-bucket policies that describe the QUERY FIELD resolve
        # from the SHAPE plane's width, not from the padded width of the record.
        # They were fitted against how big the molecules are, and `L` stopped
        # being that proxy the moment a color plane joined the padded width --
        # the same molecules resolved h = 0.5 A shape-built at L = 32 and
        # h = 0.6 A color-built at L = 72, so a color screen drew a coarser
        # lattice than a shape screen of the same molecules. Everything sized
        # against the record the kernel reads keeps using `L`.
        Ls = int(L_shape) if L_shape else int(L)
        spacing = (self.field_spacing_a if self.field_spacing_a is not None
                   else _by_width(FIELD_SPACING_BY_WIDTH, Ls, 0.8))
        minb = (self.min_blocks_per_sm if self.min_blocks_per_sm is not None
                else _by_width(MIN_BLOCKS_BY_WIDTH, Ls, 4))
        stage = (self.stage_width if self.stage_width is not None
                 else stage_width_for(L, device_id=device_id,
                                      query_atoms=query_atoms,
                                      n_features=n_features))
        # Named once, because the carveout is resolved FROM it. Splitting the
        # two would let an explicit residency reach the kernel beside a carveout
        # chosen for a different one -- a measured 2.4-6.8x loss this makes
        # unconstructible.
        residency = CoordResidency(self.coord_residency)
        # The color-PASS knobs, and where they are allowed to be anything but
        # their inert value. `color_residency`, `color_stage`, `interleave`,
        # `query_color_residency` and `color_ballot` describe the loop that runs
        # only at `color_placement = IN_GRADIENT` -- mode 3 -- and the
        # launcher REFUSES a non-inert value anywhere else by name, rather than
        # ignoring it. So a config that ships a non-inert default for them (as
        # `color_residency = SHARED, color_stage = 48` has since the operating
        # point was locked) makes every mode-1 and mode-2 launch a hard error
        # unless they are resolved here, exactly as `stage_width` is below.
        #
        # An explicit knob that does not apply is refused, but a DEFAULT that
        # does not apply resolves to the value that means "not used". Without
        # this, the shipped shape FIELD screen throws `color_residency,
        # color_stage and interleave describe the color pass` on its first block.
        in_gradient = (bool(self.color)
                       and ColorPlacement(self.color_placement)
                       is ColorPlacement.IN_GRADIENT)
        color_residency = (CoordResidency(self.color_residency) if in_gradient
                           else CoordResidency.GLOBAL)
        color_stage = int(self.color_stage) if in_gradient else 0
        query_color_residency = (CoordResidency(self.query_color_residency)
                                 if in_gradient else CoordResidency.SHARED)
        color_ballot = bool(self.color_ballot) and in_gradient
        interleave = int(self.interleave) if in_gradient else 0
        if self.color:
            # Staging is shape-only: the staged record holds coordinates, and
            # the color loop still reads the type plane at the global stride, so
            # the kernel refuses the combination rather than reading past its
            # buffer. Color therefore streams library molecules from global.
            stage = 0
        resolved = {
            "stage_width": int(stage),
            "pose_search": self.pose_search,
            "color": bool(self.color),
            "mixing": float(self.mixing),
            "steps": int(opt["steps"]),
            "lr_t": float(opt["lr_t"]),
            "lr_q": float(opt["lr_q"]),
            "start_mode": int(self.start_mode),
            "update_rule": UpdateRule(int(opt["opt_mode"])),
            "field_spacing_a": float(spacing),
            "field_pad_a": float(self.field_pad_a),
            "coord_residency": residency,
            "min_blocks_per_sm": int(minb),
            # From the EFFECTIVE color knobs above, not the declared ones: a
            # shape launch does not stage a color tier, so charging it the
            # staged carveout would partition SRAM away from the L1 it reads
            # through -- a cost measured in both directions.
            "smem_carveout": int(self.smem_carveout
                                 if self.smem_carveout is not None
                                 else carveout_for(
                                     residency, color_residency,
                                     color_stage)),
            "update_lanes": self.update_lanes,
            "start_groups_per_row": (None if self.start_groups_per_row is None
                                     else int(self.start_groups_per_row)),
            "color_loop": (None if self.color_loop is None
                           else ColorLoop(self.color_loop)),
            # A shape launch reaches a staged cell, which carries no color loop
            # arm; naming one there is a refusal in score_pairs. So a shape
            # config passes the default and says nothing.
            "color_placement": ColorPlacement(self.color_placement),
            "lanes_per_group": int(self.lanes_per_group),
            "color_residency": int(color_residency),
            "color_stage": int(color_stage),
            "query_color_residency": int(query_color_residency),
            # score_pairs derives this from the panel's maximum real-atom
            # count. A library bucket's L_shape is unrelated and may be
            # smaller than a query, so forwarding Ls here would make valid
            # non-shared query-color screens refuse on the narrow buckets.
            "query_shape_width": 0,
            # Where a library conformer's color run begins. NOT the same
            # quantity as `query_shape_width` above, and not derivable from the
            # launch's largest real atom count: `combine_shape_color` puts a
            # library conformer's features at the group's PADDED shape width,
            # so this is `Ls` and only `Ls`. Zero on a shape launch, which
            # score_pairs refuses to see a nonzero value for.
            #
            # On the merged record it is -1 instead -- "after each
            # conformer's own real atoms" -- which is what the caller passes,
            # because the width alone cannot say which layout the block is in.
            # NOT derivable from `Ls` any more, and no longer the same quantity
            # as the field policy's width proxy even on the split record.
            "library_color_offset": (
                0 if not self.color
                else (int(Ls) if library_color_offset is None
                      else int(library_color_offset))),
            "color_ballot": bool(color_ballot),
            "interleave": int(interleave),
            "stream_min_blocks_per_sm": (
                None if self.stream_min_blocks_per_sm is None
                else int(self.stream_min_blocks_per_sm)),
            "color_field_min_sources": int(self.color_field_min_sources or 0),
            "color_field_spacing_a": float(
                self.color_field_spacing_a
                if self.color_field_spacing_a is not None else 0.0),
        }
        # The supported-cells check is deliberately NOT here. `for_width` is a pure
        # resolver -- tests and analysis call it to ask what a configuration
        # would resolve to, without committing to launching it -- so refusing
        # here would conflate "what does this resolve to" with "may this run".
        # `Engine.run` checks every bucket the screen will actually touch,
        # before the first launch. See `unsupported_cells`.
        return resolved

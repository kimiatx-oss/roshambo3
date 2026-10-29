"""The methodology rules the harness enforces, not documents.

The split is between "hard refuse" (the run cannot be recorded as a timing
result at all) and "auto-tag advisory" (the run is recorded, but flagged as
unfit to be cited as the basis for a decision):

- Hard refuse: non-Release build, a build compiled with one of the forbidden
  nvcc numerics flags, GPU lock unobtainable for a requested decision-grade
  run, a launch too small to reach the measured throughput plateau when the
  corpus could have delivered a larger one, and claiming ``burst=False`` for
  a run under 60s of sustained kernel time. These invalidate the measurement
  itself; there is no honest number to salvage. A decision-grade
  number from a binary compiled out of a dirty tree is refused here too: it
  cites a commit that does not describe it.

- **There is no "batch below 1e6 conformers" rule.** Launches never cross a
  group boundary, so a 14-group 10M corpus caps its mean launch at 714,286
  and a fixed 1e6 floor refuses *every* such screen at *every* setting. See
  :func:`check_launch_saturation`.
- Auto-tag advisory: dirty git, non-Release build, GPU lock
  not held, GPU found busy by another process, clocks unlocked with >5%
  throttle observed, a failed ground-truth audit, or `max_results < 10*N`. These
  still produce a usable-for-exploration number, just not a citable one.

Note "GPU lock not held" appears in *both* lists: it is a hard refusal only
when the caller explicitly asked for a decision-grade run *and* the flock
itself could not be acquired at all (another decision-grade run already holds
it); it is merely an advisory tag when the lock was never attempted (e.g. an
advisory-only request) or was attempted and held, but ``nvidia-smi`` still
shows another process on the GPU. A non-Release build, by contrast, is
refused unconditionally regardless of decision-grade vs advisory intent --
"refuse to record a *timing* run" carries no advisory carve-out, because a
non-Release build's timing is not a real number to begin with. Every other advisory-list condition degrades the run to advisory
rather than refusing it outright.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_CONFORMERS_PER_LAUNCH = 1_000_000
"""**Not an enforced threshold. Kept bound for the archived scripts that
import it, whose numbers must keep reproducing.**

**A launch never crosses a group boundary**, so "the corpus was big" and
"each launch was big" are different claims, and 1e6 is a threshold no screen
of a 14-group 10M corpus can satisfy at any setting (its structural ceiling is
9,999,998/14 = 714,286).

It was never a measured number. What replaced it is:
:data:`SATURATING_CONFORMERS_PER_LAUNCH` and :data:`SATURATION_MARGIN` for the
launch size, and :data:`MAX_LAUNCH_OVERHEAD_FRACTION` for the per-launch
overhead the rule is actually about."""

SATURATING_CONFORMERS_PER_LAUNCH = 65_536
"""The launch size at which this engine reaches its throughput plateau --
**measured**, and the same figure ``engine.DEFAULT_CONFORMERS_PER_LAUNCH``
ships.

Measured on a 10M tuning corpus (9,994,493 conformers, 14 groups, 8 queries,
lock held, wall time, never ``launch_s``):

    4,096 -> 6.6502 s   1.00x      32,768  -> 2.6270 s   2.53x
    8,192 -> 5.5554 s   1.20x      65,536  -> 2.3229 s   2.86x
   16,384 -> 3.5975 s   1.85x     131,072  -> 2.2717 s   2.93x

65,536 is within 2.2% of the best size measured, and a 4,096 launch of the same
corpus costs 2.86x.

Conservative on both axes that matter: it was measured on the largest card here,
and a smaller device or a larger query panel saturates *sooner*, so a run that
clears this on any other configuration has more margin, not less."""

SATURATION_MARGIN = 0.5
"""How far below the saturating launch size a run may sit and still be
recorded as a decision.

Read off the same curve, not fitted to any corpus: **half** the saturating size
(32,768) measures 2.531x against 2.863x, so the launch structure is costing
11.6% of the plateau throughput at the boundary. A quarter of it (16,384) costs
35%, which is not a number anyone should quote. The boundary is therefore the
last point on the measured curve that is within roughly ten percent of the
plateau.

It admits 62,111 -- the shipped default's measured mean on a 10M color
corpus -- with 1.90x of room, and refuses a 4,086-conformer launch by
8x. There is no reading of the retired 1e6 that does both."""

MAX_LAUNCH_OVERHEAD_FRACTION = 0.05
"""The share of a run's measured time that may be paid **once per launch**
rather than once per conformer.

This is the quantity the launch-size rule is about, and unlike the launch
size it is measurable directly from the run's own stage counters -- see
:func:`launch_overhead_fraction`. Five percent because that is already this
module's materiality threshold for a run-level perturbation
(:data:`THROTTLE_ADVISORY_THRESHOLD`, :data:`SW_POWER_CAP_ADVISORY_THRESHOLD`):
a launch structure that costs more of the run than a throttling episode does is
the same size of claim about the same number.

Advisory rather than refusing, because it is a **lower bound**: h2d and the
scoring kernel also carry per-launch components that no single run can separate
(an ablation attributes 30% of the coalescing saving to the kernel itself
and 12% to h2d, on top of the 56% this fraction can see). The instrument that
refuses is :func:`check_launch_saturation`."""

MIN_MAX_RESULTS_MULTIPLE = 10
MIN_SUSTAINED_KERNEL_S = 60.0
MIN_WARMUP_REPEATS = 3
REQUIRED_TIMED_REPEATS = 11
THROTTLE_ADVISORY_THRESHOLD = 0.05  # >5% of samples show active throttling

# ---------------------------------------------------------------------------
# Clock and operating-point provenance
# ---------------------------------------------------------------------------

SW_POWER_CAP_ADVISORY_THRESHOLD = 0.05
"""Fraction of samples with ``sw_power_cap`` Active above which a
run is *power-limited*. Kept as its own named constant (even though it
shares THROTTLE_ADVISORY_THRESHOLD's value today) because "power-limited"
and "clock/thermal-throttled" are different claims about why the achieved
clock wasn't the requested one, and the record must say which -- a future change to one threshold should not silently move
the other."""

CLOCK_MATERIAL_RATIO = 1.01
"""Two runs' achieved median SM clocks are treated as "the same operating
point" only within this ratio. The smallest *resolvable* ratio measured
under the locked clock at ~1.005x (relative sd 0.049%); below that floor two
clocks cannot be said to differ at all, so flagging them as "different
operating points" would itself be an unjustified claim. 1.01 is a small
margin above that floor -- comfortably below every real cross-operating-
point gap actually observed on this project (1395 vs 1762 vs 2265 MHz), so
it will not swamp real comparisons in false positives while still catching
them."""

_PERCENT_OF_PEAK_RE = re.compile(r"(pct|percent|fraction)_of_(peak|ceiling)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Measurement regimes -- calibration vs. headline
# ---------------------------------------------------------------------------

MEASUREMENT_REGIMES = frozenset({"calibration", "headline"})

# ---------------------------------------------------------------------------
# Run kinds
# ---------------------------------------------------------------------------
#
# Not every number is a rate: a run that quotes recall and RBO and no
# throughput must not be refused for failing a bar about *timing repeats*.
#
# Two kinds, with explicit gate lists. Deliberately NOT a `skip_checks` set:
# an exemption mechanism is what this must not become, because the next
# person with an inconvenient refusal reaches for it.
RUN_KIND_RATE = "rate"
RUN_KIND_ACCURACY = "accuracy"
RUN_KINDS = frozenset({RUN_KIND_RATE, RUN_KIND_ACCURACY})

# Gates that exist to protect a THROUGHPUT figure, and therefore have nothing
# to say about a run that reports none. Named here rather than inline so a
# table of the gates can be generated from the source of truth instead of
# transcribed from it.
RATE_ONLY_GATES = (
    "check_repeats",            # >= 3 warm-ups, >= 11 timed repeats
    "check_launch_saturation",  # launch big enough to reach plateau
    "burst",                    # sustained kernel time floor
)

# A throughput figure by any name. The point of matching on the RECORD rather
# than trusting `kind` is that the label is a caller's assertion and the record
# is evidence: a run carrying `start_pairs_per_s` is a rate run whatever it
# declares, and `enforce` refuses the contradiction rather than exempting it.
# --- what counts as "this record carries a rate" ---------------------------
#
# Fourth design. Each of the first three was defeated, and the failures walked
# from one side to the other: v1 and v2 missed real rates, v3 refused real
# ACCURACY records -- including one whose `recall_per_query` was reported to
# the user as a "throughput figure".
#
# The lesson each round taught, kept here because the next person will be
# tempted by the same shortcut: **a key name alone decides nothing.** Widening
# the pattern trades misses for false positives one-for-one, and false
# positives are the worse currency because they push callers back to
# `json.dump`, which is the disease this contract is treating.
#
# So the decision uses three inputs, and only the first is the name:
#
#   1. WHICH FAMILY the key is in -- a throughput, a fraction, or a duration.
#      They are different things and the old single pattern conflated them.
#   2. THE VALUE's shape and magnitude -- a fraction in [0,1] is not a
#      throughput however it is spelled.
#   3. WHICH BLOCK it sits in -- a duration is throughput evidence in `timing`
#      and ordinary bookkeeping anywhere else.
#
# **This is not anti-tamper and must not be read as such.** Detection is by
# name, so a caller who records a rate under a name that is not a rate is
# invisible to it -- `{"x": 1.1e9}` passed every version of this design and
# always will. Mislabelling a FIELD is falsifying the record, a different and
# graver offence than mislabelling the RUN, and it is out of reach here by
# construction rather than by oversight.

#: An unambiguous throughput. These count wherever they appear.
_RATE_KEY_RE = re.compile(
    r"(_per_s(ec(ond)?)?(_|$)|per_s$|throughput"
    r"|pairs_per|conformers_per|conf_per|calls_per|_per_core"
    # `(^|_)qps(_|$)`, not `\bqps\b`: `_` IS a word character, so the boundary
    # never fires in snake_case and `mean_qps` walked through -- the same
    # trailing-separator lesson v1 was defeated by, on the very same key.
    r"|bandwidth|(^|_)qps(_|$)|gcups|pairs/s|/s$|_speed(_|$)|^speed(_|$)"
    r"|speedup"
    r"|ns_per_|us_per_|ms_per_|_per_pair|_per_conformer"
    r"|_per_(min|minute|hour|hr|h)(_|$)"
    r"|(pairs|conf(ormers)?|calls|ops|samples|mols?)_(s|min)(_|$)"
    # Unit-scoped. A bare `_per_query` is ACCURACY vocabulary in this repo --
    # METRICS.md requires a per-query table, so `recall_per_query`,
    # `P_per_query`, `trials_per_query` and `top1_per_query` are all committed
    # and all legitimate. Only a per-query TIME or RATE is rate evidence.
    # Word-boundary-anchored: without the leading `(^|_)`, the bare `s`
    # alternative matched the final letter of `trials_per_query`, which is a
    # COUNT and accuracy vocabulary.
    r"|(^|_)(ns|us|ms|s|sec|time|rate)_per_query)", re.I)

#: The only keys an accuracy run may carry in `timing`.
#:
#: **Fifth design, and the shape changed because four blacklists in a row were
#: defeated.** Rounds 1-4 all tried to enumerate what a rate LOOKS like, and
#: each was beaten by a spelling nobody had thought of -- `qps` across an
#: underscore, `throughput_hz`, a list of numeric strings, `kernel_s` and the
#: rest of this file's own `PER_LAUNCH_STAGES` vocabulary. Enumerating an open
#: set is not winnable, and every widening cost a false positive elsewhere.
#:
#: `timing` is where a rate run puts its numbers -- that is the documented API,
#: `run.timing = {...}` -- so for that block the question inverts: instead of
#: asking "is any of this a rate", require that none of it is anything but
#: bookkeeping. A closed set cannot be defeated by a spelling, because an
#: unrecognised key is a refusal rather than a pass.
#:
#: The other blocks keep the detector: they are not where rates conventionally
#: go, and a whitelist there would refuse half the repo.
_ACCURACY_TIMING_ALLOWED = frozenset({"n_repeats", "n_warmup", "notes", "kind"})

#: A percentage. A fraction spelled 0-100 defeats a magnitude test that reads
#: [0,1] as "fraction" -- and the committed record set is full of them:
#: `l1_sector_hit_rate_pct`, `l2_hit_rate_pct` and their siblings run to 99.99
#: across 558 occurrences. Found by auditing the magnitude rule against every
#: committed document rather than by a probe, which is the only way this class
#: shows up.
_PCT_KEY_RE = re.compile(r"(_|^)(pct|percent|percentage)(_|$)", re.I)

#: A frequency in hertz is a configured sampling rate, never a throughput of
#: work. Pattern rather than an enumerated key, because `sampling_rate_hz`,
#: `sample_rate_hz` and `poll_rate_hz` are the same fact three ways.
_HZ_KEY_RE = re.compile(r"_hz(_|$)", re.I)

#: A fraction, which reads like a rate and is not one. Decided by MAGNITUDE
#: rather than by enumeration: `basin_flip_rate`, `baseline_active_rate`,
#: `violation_rate_in_topN`, `win_rate`, `tie_rate` and their successors are
#: generated faster by METRICS.md than any frozen list can absorb.
_FRACTION_KEY_RE = re.compile(r"(^|_)rate(s)?(_|$)", re.I)

#: A duration. Throughput evidence in `timing` -- that is where a rate run puts
#: its numbers -- and ordinary bookkeeping elsewhere, where `warmup_seconds`,
#: `deadline_seconds` and an audit's own `total_ms` all legitimately live.
_DURATION_KEY_RE = re.compile(
    r"((^|_)(wall|elapsed|duration|runtime|total|time|mean|median|min|max"
    r"|p\d+)_(s|ms|us|ns)(_|$)"
    r"|(^|_)(seconds?|minutes?|millis(econds?)?)(_|$))", re.I)

#: Keys that match above and are still not measurements. Short by design now
#: that magnitude and block context carry most of the load.
_RATE_KEY_EXEMPT = frozenset({
    # The conformers-per-molecule corpus fact, whose only honest home in the record is
    # `dataset`. A contract that refuses it stops the ledger carrying the
    # evidence a sibling gate requires; this entry is why the list exists.
    "conformers_per_molecule", "conformers_per_mol",
    # MEASURED, not configured -- an earlier comment here filed it under
    # "configured durations", which was simply untrue. It is exempted for
    # compatibility: committed accuracy records carry it and
    # `ctx.sustained_kernel_s` consumes it.
    "sustained_s",
})

#: A string carries a number if it STARTS with one, after an optional approx or
#: inequality marker -- "~1.13e9 pairs/s" and "<0.5 s" are natural informal
#: recording. A digit must still follow, so "~/scratch/..." stays a path.
#: Known residual: a digit-leading PATH under a rate-named key ("10M-run/
#: rates.csv") still reads as a number, and size-named directories are real
#: vocabulary here.
_NUMERIC_STR = re.compile(r"^\s*[~<>≈]?\s*[-+]?(\d|\.\d)")


def _numeric_leaves(v, out=None):
    """Every number reachable in ``v``, for the magnitude test."""
    out = [] if out is None else out
    if isinstance(v, bool):
        return out
    if isinstance(v, (int, float)):
        out.append(float(v))
    elif isinstance(v, str):
        # A numeric string INSIDE a container. Handled at the top level since
        # v1 and missed inside lists and dicts until round four, so a list of
        # per-repeat rates written as text walked through the canonical key.
        if _NUMERIC_STR.match(v):
            head = re.split(r"[^0-9eE+.\-]", v.strip().lstrip("~<>\u2248 "))[0]
            try:
                out.append(float(head))
            except ValueError:
                out.append(1e30)   # numeric-looking, unparseable: not a fraction
    elif isinstance(v, (list, tuple)):
        for x in v:
            _numeric_leaves(x, out)
    elif isinstance(v, dict):
        for x in v.values():
            _numeric_leaves(x, out)
    return out


def _looks_numeric(v) -> bool:
    """Could this value BE a measured number?

    Handles the containers a rate actually arrives in: review defeated two
    versions here, first with a list of per-repeat rates and then with a
    ``{"mean": ..., "std": ...}`` summary dict, which is this repo's own house
    style -- so the canonical rate key carrying a summary walked through the
    contract built to catch it.
    """
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        return bool(_NUMERIC_STR.match(v))
    if isinstance(v, (list, tuple, dict)):
        return bool(_numeric_leaves(v))
    return False


#: Sub-keys holding a COUNT beside a fraction, in this repo's summary style --
#: `{"value": 0.0051, "n_ties": 50737}`. Including them in the magnitude test
#: made a summarized fraction read as a throughput, which is the v3 failure
#: class rebuilt one container deeper.
_COUNT_SUBKEY_RE = re.compile(
    r"^(n|n_.*|count|size|total|denom(inator)?|samples?|queries|pairs)$", re.I)


def _fraction_leaves(v):
    """Numeric leaves that are candidate fractions -- counts excluded."""
    if isinstance(v, dict):
        out = []
        for k, x in v.items():
            if not _COUNT_SUBKEY_RE.match(str(k)):
                out.extend(_fraction_leaves(x))
        return out
    return _numeric_leaves(v)


def _is_fraction(v, pct: bool = False) -> bool:
    """A fraction rather than a throughput.

    ``pct`` widens the window to [0, 100] for a key that says it is a
    percentage. Without it, every percent-spelled hit rate in the committed
    record set reads as a throughput -- the magnitude rule's one real blind
    spot, and it is a big family.
    """
    nums = _fraction_leaves(v)
    hi = 100.0 if pct else 1.0
    return bool(nums) and all(0.0 <= n <= hi for n in nums)


_RECALL_RE = re.compile(r"recall", re.I)
_CONFORMER_RE = re.compile(r"conformer", re.I)
_MOLECULE_RE = re.compile(r"molecule", re.I)
_PER_QUERY_RE = re.compile(r"per[_-]?quer(y|ies)", re.I)
"""The two measurement regimes. ``calibration``: clocks locked, one
operating point per arm, used to decide between implementations or attribute
a bottleneck -- no absolute-throughput claim is made from it.
``headline``: clocks unlocked, the card's natural operating point,
reported as an achieved distribution and asserted nothing about. Numbers
taken at different clocks are never combined, so a number from one regime may
never be combined with a number from the other."""


@dataclass(frozen=True)
class LaunchShape:
    """How a screen was cut into launches -- derived from the screen, never typed.

    It exists as a type rather than as three loose integers because of the
    specific way the old check failed: ``check_min_batch`` took a number, and
    a caller handed it 20,381,286 -- the corpus total -- so the check returned the
    answer the caller wanted for a reason unrelated to what it measures. A
    ``LaunchShape`` cannot be handed a corpus total, because
    :meth:`from_screen` reads the launch count off the screen that produced it
    and divides.

    ``n_groups`` is here because **launches never cross a group boundary**
    (the premise ``MoleculeTopK.merge`` rests on). The
    launch count can therefore never fall below the group count, so
    :attr:`ceiling` is the largest mean launch a corpus can deliver at any
    setting -- 714,286 on a 14-group 10M corpus. A threshold that ignores that
    is a threshold no screen of that corpus can satisfy.
    """

    conformers_per_launch: float
    n_launches: int
    n_conformers: int
    n_groups: int | None = None
    """The corpus's group count, or None when the run did not record it. None
    means the ceiling is unknown, and an unknown ceiling is treated strictly --
    a corpus-bound run gets the benefit of the doubt only when it can show the
    structure that bounds it."""

    @property
    def ceiling(self) -> float | None:
        """The largest mean launch this corpus can deliver, at any setting."""
        if not self.n_groups:
            return None
        return self.n_conformers / self.n_groups

    @classmethod
    def from_screen(cls, stats, source=None, n_groups: int | None = None):
        """From a :class:`ScreenStats`, and the source that produced it.

        ``stats.n_blocks`` is the number of *launches* -- coalesced
        runs, not on-disk blocks -- so the mean is ``n_conformers / n_blocks``.
        ``source`` supplies the group count when it is an ``R3gSource``; pass
        ``n_groups`` directly when it is not.
        """
        launches = int(getattr(stats, "n_blocks", 0) or 0)
        total = int(getattr(stats, "n_conformers", 0) or 0)
        if n_groups is None and source is not None:
            groups = getattr(source, "group_indices", None)
            n_groups = len(groups) if groups is not None else None
        return cls(
            conformers_per_launch=(total / launches) if launches else 0.0,
            n_launches=launches,
            n_conformers=total,
            n_groups=n_groups,
        )


PER_LAUNCH_STAGES = ("alloc_s", "precompute_s", "free_s")
"""The device stages whose cost is paid once per launch rather than once per
conformer, from a 60-launch-vs-1-launch ablation of the same 245,760
conformers: precompute 0.0247 ->
0.0014 s (94% of it is per launch), free 0.0077 -> 0.0001 s (99%), and alloc,
which is a cudaMalloc per launch by construction.

``kernel_s`` and ``h2d_s`` are deliberately **not** here even though the same
ablation attributes 30% and 12% of the coalescing saving to them: their
per-launch component cannot be separated from their per-conformer component
within a single run, and a fraction that guessed at it would not be a
measurement. That is why the fraction they do support is a lower bound and an
advisory rather than the refusing instrument."""


@dataclass
class RunRequest:
    """What the caller is asking the harness to do and record."""
    decision_grade: bool
    """True: caller wants this run eligible to be cited as the basis for a
    decision. False: caller explicitly only wants an advisory/exploratory
    number (e.g. a quick correctness-adjacent smoke run)."""
    n_conformers: int
    """Conformers in **one launch**, not in the corpus.

    The launch-size rule reads :attr:`launch` instead, because this field
    takes a number, and a caller once passed the corpus total. What is left for
    this field is the per-batch truncation rule (:func:`check_max_results`),
    which is genuinely about one call's population."""
    max_results: int
    claim_non_burst: bool
    """True iff the caller is asserting this run's kernel time is long
    enough to not be labeled "burst" (i.e. wants to use it for 1e9/1e11
    projections)."""
    n_repeats: int
    n_warmup: int
    padding_change_is_the_subject: bool = False
    """Whether this comparison is DELIBERATELY between two padding layouts.

    "Two configurations compared for throughput must have identical L
    histograms, or the comparison is between data layouts, not kernels" is a
    rule about accidents. A padding study is the exception: it changes `L` on
    purpose, beside a point that holds `L` fixed, so the two effects can be
    separated. The
    declaration is required, and it is required *from the caller*, for the same
    reason `check_device_comparable` will not infer a cross-device ratio: a
    deliberate comparison and an accidental one look identical in the numbers
    and differ only in whether someone decided.

    Setting it does not make the difference disappear -- the two tables are
    carried in the run record either way, so a reader can see what was compared
    -- it makes the refusal an advisory tag instead."""
    launch: "LaunchShape | None" = None
    """How this screen was cut into launches.

    ``None`` means the run did not record its launch structure, which is not
    the same as "the launches were fine": the check then falls back to
    :attr:`n_conformers`, the field a corpus total was passed to, and says so.
    Build it with ``LaunchShape.from_screen(stats, source)``."""

    kind: str = RUN_KIND_RATE
    """What sort of number this run produced -- :data:`RUN_KINDS`.

    ``"rate"`` (the default, so every existing caller keeps its gates) means a
    throughput figure; the timing gates in :data:`RATE_ONLY_GATES` apply and
    its grade certifies the rate evidence only. Accuracy-shaped values in
    ``metrics`` are supplemental and do not invoke the accuracy gates.
    ``"accuracy"`` means recall/ranking and no throughput; those three gates do
    not apply and the accuracy gates in :func:`enforce` do.

    **This is a declaration, not an exemption, and the difference is enforced.**
    :func:`find_rate_fields` reads the record, and a run that declares
    ``accuracy`` while carrying a throughput figure is refused as malformed
    rather than exempted -- so relabelling an inconvenient timing refusal does
    not make it go away, it changes which refusal you get. An accuracy run is
    also still GRADED: skipping a timing bar is not skipping grading."""

    chose_a_value: str | None = None
    """Name of the parameter this run TUNED, if it tuned one.

    ``None`` means nothing was chosen and the fit/report split does not apply.
    Naming something (``"h"``, ``"n_steps"``) asserts that a value was selected
    from this run, and then :attr:`RunContext.fit_corpus` and
    :attr:`RunContext.report_corpus` must both be named and must **differ**.

    Declared by the caller because the harness cannot infer it: a sweep that
    reports every cell and a sweep that picks one look identical in the
    numbers. A grid-spacing run chose ``h = 1.2`` on a tuning corpus at a 0.98
    recall floor and measured 0.9750 on the evaluation corpus -- below the floor
    it was selected by.
    The discipline caught that only because the script happened to report both;
    nothing made it."""

    max_results_applied_per_batch: bool = True
    """Whether ``max_results`` can truncate each batch before a global merge.

    The per-batch truncation rule was written for the legacy host reducer.  A device-side
    top-K observes the whole batch before retaining K and therefore has no
    per-batch truncation hazard; its K may be smaller than ``10*N``
    without misrepresenting which population the kernel processed.
    """


@dataclass
class RunContext:
    """What was actually true/measured about the run."""
    is_release_build: bool
    git_dirty: bool
    """Whether the checkout has uncommitted **source** changes.

    Deliberately source-scoped rather than whole-tree: a results directory is a
    run's OUTPUT, and a check that reads it as input makes recording a graded run
    downgrade the next one. `bench/gitinfo.py` splits the two and records both;
    this field is the half that bears on reproducibility."""
    gpu_lock_held: bool
    gpu_lock_acquisition_failed: bool
    """True iff the harness tried (because decision_grade=True) to acquire
    the flock and another run already held it."""
    other_process_on_gpu: bool
    clocks_locked: bool
    throttle_fraction: float
    sustained_kernel_s: float
    measurement_regime: str
    """``"calibration"`` (clocks locked, one operating
    point for every arm, used to decide between implementations or attribute
    a bottleneck) or ``"headline"`` (clocks unlocked, the card's natural
    operating point, reported as an achieved distribution rather than
    asserted). Deliberately **not defaulted**: a locked clock is not always
    the right one, and a field that silently defaulted to either value would let a
    calibration number and a headline number be combined without anyone
    deciding to. Every caller must state which regime this run is -- a
    ``RunContext`` built without it raises ``TypeError`` before ``enforce()``
    is ever called, which is the point.

    Validity (one of :data:`MEASUREMENT_REGIMES`) is checked in
    :func:`enforce`, not here, so a caller who passes a typo still gets a
    real :class:`RunContext` to inspect in a test failure message rather
    than an opaque construction-time exception.
    """
    gold_ref_measurement_regime: str | None = None
    """``measurement_regime`` of ``gold_ref_run_id``'s run, if known. ``None``
    means either there is no gold-ref comparison (``gold_ref_run_id is
    None``) or that run never recorded a regime -- the two
    cases are distinguished the same way :attr:`gold_ref_achieved_clock_mhz_median`
    is: by whether ``gold_ref_run_id`` itself is set."""
    padding_table: list | None = None
    """This run's ``(bucket, L, L_color, n_conf)`` rows -- the padded-width
    histogram of the library it screened, canonicalized by
    :func:`canonical_padding_table`.

    ``None`` means the run did not record one, which is not the same as "the
    padding matched": a comparison against a gold ref whose padding is unknown
    is exactly the case :func:`check_padding_comparable` refuses, on the model
    of :func:`check_clock_comparable` rather than
    :func:`check_arch_profile_comparable`. Older run records carry
    none, so the refusal is scoped to runs that name a
    ``gold_ref_run_id`` AND carry a table on at least one side."""
    gold_ref_padding_table: list | None = None
    """``padding_table`` of ``gold_ref_run_id``'s run, if known."""
    gpu_uuid: str | None = None
    """The GPU UUID (``bench.gpu_lock.DeviceIdentity.uuid``) this run executed
    on, if known. Used only for the cross-device comparison refusal below --
    a ratio between two runs is not the same claim as an absolute number, and
    ratios are permitted only when taken on one card in one session; this field is
    what lets ``enforce()`` tell whether that precondition holds."""
    gold_ref_gpu_uuid: str | None = None
    """``gpu_uuid`` of ``gold_ref_run_id``'s run, if known."""
    checked_a_gpu_build: bool = True
    """Whether there was a CUDA extension to inspect at all.

    ``numerics.check_build()`` reports ``ok`` when it found no forbidden flag,
    which is **vacuously true** on an install with no extension -- and this
    harness reads exactly that to decide whether a timing may be recorded. A
    timed GPU measurement from a build with no GPU code is not a measurement, so
    it is refused rather than tagged.

    **Defaulted permissively, which is why it must come from**
    :meth:`RunContext.from_build`. Left alone it asserts that a GPU build was
    inspected, and every harness script that assembled its context by hand
    took that default -- so this refusal could not fire on a real run. A
    decision-grade context that was not built from an inspected build is now
    refused outright, before this field is read.
    """
    checked_host_flags: bool = True
    """Whether the build stamp carried host compiler flags to inspect.

    The third instance of ``checked_a_gpu_build``'s shape:
    ``check_cxx_flags`` returns an empty list
    both when it inspected the host flags and found nothing and when the stamp
    recorded none, and the harness cannot tell those apart from the result.
    The host half of the extension prepares and reduces what the kernel sees.

    **Advisory, not a refusal.** Host flags are not a measured axis here
    (``numerics.SUSPECT_CXX_FLAGS``: "the policy does not claim a number it has
    not taken"), so an uninspected host build is unverified rather than known
    wrong -- the same standing as ``arch_probe_error``.

    Defaulted permissively for the same reason the others are, and safe for the
    same reason: :attr:`build_derived` is refused first, so a decision-grade
    context cannot reach this field without having come from
    :meth:`RunContext.from_build`.
    """

    # ---- the CPU half of the provenance ----
    #
    # Every field below defaults to the answer that keeps a GPU run's verdict
    # exactly what it was. The CPU rules fire only when `backend_name` says a
    # CPU backend produced the number, so adding them cannot silently change
    # how any existing GPU record grades.

    backend_name: str = "cuda"
    """Which backend produced this number (``roshambo3.backend.select``'s name,
    or the module's ``__name__``). Not cosmetic: it is what routes the CPU
    rules, and a record that cannot say which backend ran is a record whose
    numbers cannot be compared with anything."""

    cpu_identity: dict | None = None
    """``bench.cpu.cpu_identity().as_dict()`` -- model, physical cores, SMT,
    cache sizes, ISA flags, governor. Asked of the machine, because a
    hand-maintained hardware table has been wrong before."""

    cpu_lock_held: bool = False
    """Whether ``flock $ROSHAMBO3_CPU_LOCK`` was held for the run's duration.
    The CPU analogue of the GPU lock, and it catches a case the GPU check cannot
    see at all: a GPU screen holding the GPU lock is itself heavy CPU load."""

    cpu_lock_acquisition_failed: bool = False

    cpu_busy: bool = False
    """Another process was using a material share of the machine
    (``bench.cpu.query_cpu_state().busy``). As on the GPU, holding the lock does
    not mean the machine is yours -- correctness work runs unlocked by design."""

    cpu_foreign_processes: list = field(default_factory=list)

    cpu_affinity: list | None = None
    """The CPUs this run was pinned to, as actually set rather than as
    requested. ``None`` means unpinned, which is an advisory on a timed run:
    affinity is the ONE thing we can control here -- clocks cannot be locked and
    SMT cannot be changed on these machines -- so declining to control it is
    declining the only control there is."""

    cpu_isa_selected: str | None = None
    """Which dispatch target the process ACTUALLY selected at run time.

    The CPU analogue of a wrong-architecture GPU build, and the reason it is a
    refusal rather than a note: a
    build that silently ran a baseline SSE2 path looks like a slow kernel, not
    like a broken build, and with a dispatching SIMD library it is easy to
    reach by accident. Asking the binary what it contains is not the same
    question as asking the process what it chose -- both are recorded."""

    cpu_isa_available: list = field(default_factory=list)
    """The dispatch targets compiled into the loaded extension."""

    checked_a_cpu_build: bool = False
    """Whether a CPU extension was actually inspected for this run.

    The twin of :attr:`checked_a_gpu_build`, and it defaults to **False** where
    that one defaults to True. The asymmetry is deliberate: a permissive
    default on a field that
    answers "did we check?" makes every unpopulated context claim it checked.
    ``checked_a_gpu_build`` could not be flipped without changing what every
    existing GPU record means; this field is new, so it starts strict."""

    cpu_vector_width: int | None = None
    """The SIMD width the run's scores were computed at.

    A vectorized sum is a reassociation, so a CPU score
    is a function of its width, and **a record that does not name its width
    cannot be compared with anything**. Refused rather than advised for the
    same reason :attr:`cpu_isa_selected` is: the comparison it silently
    corrupts is between two runs that both look fine."""

    cpu_threads: int | None = None
    """How many threads the run used.

    Recorded, and deliberately NOT thresholded. The CPU analogue of launch-size
    saturation is thread-count and work-chunk saturation, which must be
    *measured, not assumed* -- and it has not been measured yet. So this
    field carries the fact and no rule reads it. A threshold invented here
    would be a GPU constant wearing CPU clothes."""

    achieved_cpu_mhz: dict | None = None
    """``bench.cpu.CpuFrequencySampler.summary()``'s achieved distribution.
    None means the run did not sample, which is not the same as "the clock was
    steady" -- a run that did not look may not claim a clock."""

    device_stage_seconds: dict | None = None
    """This run's ``ScreenStats.device_stages`` -- the extension's own per-stage
    accumulators, populated only under ``Engine(stage_timing=True)``.

    Carried so :func:`launch_overhead_fraction` can measure the launch
    structure's cost out of the run itself rather than from a constant taken on
    another card with another query panel. ``None`` means the run did
    not record them, which the rule reports as unmeasured rather than as zero."""

    measured_total_s: float | None = None
    """The wall time the run's throughput figure was computed over -- the
    denominator :func:`launch_overhead_fraction` divides by. Deliberately not
    ``launch_s``: with a device sink ``score_pairs`` returns without
    synchronizing, so that field times enqueue."""

    build_source_dirty: bool | None = None
    """Whether the **compiled subtree was dirty when this binary was built**
    (``BuildInfo.cuda_src_dirty``, via :meth:`from_build`).

    Not the same fact as :attr:`git_dirty`, which is this *checkout*'s state
    while the run happens. A binary built from a patched tree and then run from
    a clean checkout reports ``git_dirty=False`` and this ``True``: its stamp
    names a commit that does not contain the code it ran. A ``decision``-grade
    number from such a build is refused.

    ``None`` means the binary could not say -- it predates the stamp, or its
    stamp generator did not run. That is an advisory, not a pass: "nobody
    looked" and "the tree was clean" are different claims, the same distinction
    ``checked_a_gpu_build`` draws."""

    build_source_subtree: str | None = None
    """Which source tree the build stamp above describes.

    The dirty-build refusal used to hardcode ``_cuda_src`` in its message while
    the check itself fires for CPU builds too -- so a run refused for a dirty
    ``_cpu_src/interface.cpp`` was told to go look at the CUDA subtree. The
    refusal was right and its explanation misdirected the one investigation it
    exists to start."""

    build_source_sha256: str | None = None
    """``BuildInfo.cuda_src_sha256`` -- sha256 over the bytes of every file this
    binary was compiled from. The field a later reader uses to establish which
    source produced a number, when the commit cannot say."""

    build_source_stamp_phase: str | None = None
    """``"build"``, ``"configure"``, ``"none"`` or None -- when the two fields
    above were taken. ``"configure"`` is stale across a
    configure-then-edit-then-build cycle and is flagged as such."""

    ground_truth_audit_passed: bool | None = None
    """None if no ground-truth audit ran against this run (not
    automatically a problem); False is what triggers the advisory tag.
    This field is not serialized, so naming it
    for what it means costs no compatibility."""
    arch_profile: str = "unknown"
    """This run's build's :attr:`VerifiedBuild.arch_profile` --
    "native"/"fat"/"unknown"."""
    arch_probe_error: str | None = None
    """Why the loaded ``.so``'s architecture list could not be read, if it
    could not.

    The same shape of defect as ``checked_a_gpu_build``: ``cuobjdump`` is not
    on ``PATH`` under a default NVIDIA apt install, and an empty list would be
    indistinguishable from "probed, found nothing". A library compiled for the
    wrong architecture JIT-compiles at runtime, and this probe is what catches
    it. A run that could not run it is advisory: not because the build is
    wrong, but because nothing established that it is right.

    Comes from :meth:`from_build`; ``None`` here is the claim "the probe ran",
    not the absence of one."""
    gold_ref_arch_profile: str | None = None
    """The arch_profile of ``gold_ref_run_id``'s build, if this run is being
    compared against a prior gold-reference run. None if there is no
    gold-ref comparison being made."""

    numerics_violations: list[str] = field(default_factory=list)
    """Forbidden nvcc numerics flags found in the loaded build's stamp
    (``roshambo3.numerics.check_build``). Non-empty means this build's
    arithmetic is not the pinned arithmetic, and no score it produced is
    comparable with any other run's."""

    numerics_advisories: list[str] = field(default_factory=list)
    """Toolkit-version or architecture-scope notes. Recorded and tagged,
    not fatal -- the toolkit axis is a measured perturbation but roughly 300x
    weaker than any of the forbidden flags."""

    # ---- clock and operating-point provenance ----

    achieved_clock_mhz_median: float | None = None
    """This run's *achieved* median clocks.sm (from ClockPowerSampler.summary()
    -- never the requested value). None if this run did not sample clocks at
    all (e.g. a pre-flight refusal before any GPU time was spent)."""

    sw_power_cap_fraction: float = 0.0
    """Fraction of clock samples with clocks_event_reasons.sw_power_cap
    Active. Distinguished from the generic throttle_fraction above: this is
    what separates "power-limited" from "clock-locked" as a claim."""

    profiler_control_state: str = "none"
    """This run's :class:`bench.clocks.ProfilerControlState.state`
    -- "none", "profiler-active", or "undetermined". Defaults to "none" for
    callers (e.g. existing tests) that never checked; a caller that *did*
    check and could not determine the state must pass "undetermined"
    explicitly rather than rely on this default -- see
    clocks.detect_profiler_clock_control's own docstring on why a failed
    check is not the same as a clean one."""

    profiler_control_tool: str | None = None
    """"ncu"/"nsys"/None -- which tool was detected, if any."""

    gold_ref_run_id: str | None = None
    """The run_id this run is being compared against, if any. None means no
    gold-ref comparison is being made at all -- distinct from "a comparison
    is being made but the gold ref's operating point wasn't recorded",
    which is a known failure mode and must not be silently treated as
    "no problem"."""

    gold_ref_achieved_clock_mhz_median: float | None = None
    """gold_ref_run_id's achieved median clocks.sm, if known. None if there
    is no comparison (gold_ref_run_id is None) *or* the comparison exists
    but that run's clock was never recorded -- check_clock_comparable
    distinguishes these two cases from its two arguments together, not from
    this field alone."""

    gold_ref_profiler_control_state: str | None = None
    """gold_ref_run_id's profiler_control_state, if known."""

    percent_of_peak_violations: list[str] = field(default_factory=list)
    """Key paths in this run's ``metrics`` dict that look like a
    percent-of-peak figure but do not carry a normalizing-clock field
    alongside them -- see find_percent_of_peak_fields_missing_clock(). Non-
    empty is a hard refusal: this is the specific, named error behind the
    project's wrong "5.6% of the MUFU.EX2 ceiling" figure (true value at the
    measurement's own clock: 12.9%)."""

    timing: dict | None = None
    """This run's ``timing`` block, verbatim. :func:`enforce` reads it with
    :func:`find_rate_fields` to police the run-kind contract, which has to be
    done from the record rather than from :attr:`RunRequest.kind` -- see that
    field. ``None`` is treated as "no timing recorded"."""

    record_blocks: dict | None = None
    """The record's other **caller-writable** blocks -- ``config``,
    ``dataset``, ``audits``, ``artifacts``, ``query_panel``, ``notes`` -- so the
    run-kind contract sees what a caller can put a number into, rather than
    whichever blocks someone remembered.

    ``operating_point`` is deliberately absent and this docstring used to claim
    it was present. It is built by ``CpuRun`` from sampled machine state, not
    supplied by the caller, so it is not a surface anyone can hide a rate in --
    and listing it while not wiring it is the kind of false promise that let
    ``query_panel`` go unscanned through two review rounds.

    Review defeated the first version end to end by parking a throughput figure
    in ``config`` and in ``audits``: both reach the written record, neither was
    scanned, and the run graded as an accuracy run. Scanning only ``timing`` and
    ``metrics`` made the contract a statement about where a caller chose to put
    a number.

    **What is deliberately still not covered**: free prose. A rate written into
    ``notes`` as English is not flagged, because ``notes`` is not a rate-named
    key and a string is only read when it sits under one. That is a boundary,
    not an oversight -- a record that *describes* a measurement in prose is not
    a record that *carries* a rate -- and it is stated here so nobody reads the
    contract as wider than it is."""

    metrics: dict | None = None
    """This run's ``metrics`` block, verbatim. The accuracy-kind gates read
    it for both recall levels (conformer and molecule) and the per-query table
    (``doc/METRICS.md``). Passing it in rather than having the caller
    pre-compute a summary is deliberate: a check the caller must remember to
    run is a check that does not run."""

    conformers_per_molecule: float | None = None
    """The screened corpus's mean conformers per molecule.

    Accuracy runs report both recall levels; this is what says whether the
    corpus could tell them apart. A corpus with exactly one conformer per
    molecule (as at group 5 of the 10M tuning and evaluation corpora) has their molecule-level recall IS their
    conformer-level recall and "both levels present" passes on a corpus that
    cannot exercise the distinction. ``None`` means unrecorded, which is itself
    an advisory: a degenerate pass then cannot be told from a real one."""

    fit_corpus: str | None = None
    """Corpus a tuned value was fitted on. See
    :attr:`RunRequest.chose_a_value`."""

    report_corpus: str | None = None
    """Corpus results are reported on, and it must differ from
    :attr:`fit_corpus`. Fitting and reporting on the same corpus inflated
    recall by +0.0092 at K=30."""

    build_derived: bool = False
    """Whether the build-dependent fields above were filled in from a build
    that was actually inspected, rather than left at their defaults.

    Every one of those defaults is the *permissive* answer -- there was GPU
    code, no forbidden flag was found, the architecture probe ran. That is the
    right default for a hand-built context in a unit test and the wrong one for
    a run whose number will be cited, and for a long time every real harness
    script took it: all six forwarded ``arch_profile`` and none of the rest, so
    the no-GPU-build refusal and the pinned-arithmetic refusal could not fire outside the
    tests that set the fields by hand.

    :meth:`from_build` is the only thing that sets this. A decision-grade run
    without it is refused -- not because the build is wrong, but because
    nothing looked."""

    @classmethod
    def from_build(cls, build, arithmetic_check=None, **runtime):
        """The one place a build's verdict becomes part of a run's context.

        ``build`` is a :class:`bench.build_check.VerifiedBuild` and
        ``arithmetic_check`` a :class:`roshambo3.numerics.ArithmeticCheck`
        (``numerics.check_build()``). Everything else is what only the run
        itself knows -- the lock, the clocks, the other processes on the card
        -- and is passed through unchanged.

        Derived here rather than at each call site because a field-by-field
        copy is exactly how the three fields below came to be dropped by every
        caller at once. Adding a build-dependent field means editing this
        method, and nothing else.
        """
        # WHICH compiled artifact produced this number. A tree can
        # carry both extensions, and they are separate compilations with
        # separate flags -- `-O3 -DNDEBUG` for the CUDA host half against
        # `-march=x86-64-v3 -fno-fast-math -fno-associative-math` for the CPU
        # one. Reading the CUDA fields for a CPU run does not leave the record
        # incomplete, it fills it with another binary's answers, which is
        # worse than a gap because nothing looks wrong.
        cpu_run = is_cpu_backend(runtime.get("backend_name", "cuda"))
        derived = dict(
            is_release_build=(build.cpu_is_release if cpu_run
                              else build.is_release),
            arch_profile=build.arch_profile,
            arch_probe_error=build.arch_probe_error,
            # The build says which source it was compiled from; the run
            # says which checkout it was launched from. Both reach the record,
            # and only the first can refuse a decision-grade number.
            build_source_dirty=getattr(
                build, "cpu_src_dirty" if cpu_run else "cuda_src_dirty", None),
            build_source_sha256=getattr(
                build, "cpu_src_sha256" if cpu_run else "cuda_src_sha256",
                None),
            build_source_subtree=("src/roshambo3/backend/_cpu_src" if cpu_run
                                  else "src/roshambo3/backend/_cuda_src"),
            build_source_stamp_phase=getattr(
                build,
                "cpu_src_stamp_phase" if cpu_run else "cuda_src_stamp_phase",
                None),
        )
        if cpu_run:
            # Asked of the BUILD rather than accepted from the caller, for the
            # reason this constructor exists: a field a caller can supply is a
            # field a caller can get wrong, and `selected_target` is answered
            # by the process itself. Passing any of these separately raises
            # below, via the same overlap guard that protects the CUDA fields.
            # NOT `bool(build.with_cpu)`. Presence is not inspection: the
            # arithmetic check is the only thing that reads the CPU extension's
            # host flags, so with no check passed this must fail CLOSED rather
            # than fall back to "an extension exists". The check overwrites it
            # below when one is given. A hand-rolled harness that forgets is
            # exactly the population this guards -- and a permissive default
            # would make every unpopulated context claim it checked.
            derived.update(
                checked_a_cpu_build=False,
                cpu_isa_selected=build.cpu_selected_target,
                cpu_isa_available=list(build.cpu_compiled_targets),
                cpu_vector_width=build.cpu_vector_width,
            )
        if arithmetic_check is not None:
            derived["checked_a_gpu_build"] = bool(
                arithmetic_check.checked_a_gpu_build)
            # One name, one meaning. `build.with_cpu` above is PRESENCE; the
            # check's `checked_a_cpu_build` is presence AND flags-read, which
            # is what the refusal at the bottom of this file actually claims
            # ("the flags it was compiled with were never read"). Wiring the
            # weaker one under the stronger one's name left a CPU extension
            # that is present but unstamped merely advisory-capped, while both
            # the field name and its refusal text said refused. Found in
            # review, and it STRENGTHENS the gate: the only state that changes
            # is present-but-unstamped, which goes from advisory to refused.
            if getattr(arithmetic_check, "checked_a_cpu_build", None) is not None:
                derived["checked_a_cpu_build"] = bool(
                    arithmetic_check.checked_a_cpu_build)
            # By the backend that produced the number: `checked_host_flags`
            # is read off the CUDA stamp, and for a CPU run the flags that
            # matter are the CPU extension's, which `_cpu_flag_advisories`
            # inspects. Judging a CPU run on the CUDA stamp made a CPU-only
            # tree fail a check it had actually passed.
            derived["checked_host_flags"] = bool(
                getattr(arithmetic_check,
                        "checked_cpu_host_flags" if cpu_run
                        else "checked_host_flags", True))
            derived["numerics_violations"] = list(
                arithmetic_check.violations)
            adv = list(getattr(arithmetic_check, "advisories", []))
            if not cpu_run:
                # The mirror of the CPU branch below: a GPU run on a CUDA-only
                # tree must not draw "no CPU extension is present", which says
                # nothing about the binary that produced its number.
                host_adv = set(getattr(
                    arithmetic_check, "host_advisories", []))
                adv = [a for a in adv if a not in host_adv]
            if cpu_run:
                # Drop the advisories that are about the CUDA binary.
                # A CPU run was being told "build targets 'native' ... do not
                # quote a number from it", which is true of the GPU extension
                # and says nothing about the one that produced this number.
                # Violations are NOT filtered: a forbidden nvcc flag says the
                # build as a whole is not what the policy pins, and refusing
                # is the conservative reading until someone measures otherwise.
                device_adv = set(getattr(
                    arithmetic_check, "device_advisories", []))
                adv = [a for a in adv if a not in device_adv]
            derived["numerics_advisories"] = adv
        overridden = sorted(set(derived) & set(runtime))
        if overridden:
            raise TypeError(
                f"{', '.join(overridden)} come(s) from the inspected build and "
                f"must not be passed separately -- that is the drift this "
                f"constructor exists to prevent.")
        derived.update(runtime)
        return cls(build_derived=True, **derived)


@dataclass
class EnforcementResult:
    hard_refuse: bool
    refusal_reason: str | None
    advisory_reasons: list[str] = field(default_factory=list)
    burst: bool = False
    power_limited: bool = False
    """True iff sw_power_cap was Active for more than
    SW_POWER_CAP_ADVISORY_THRESHOLD of this run's clock samples. Always
    computed and exposed regardless of whether clocks_locked was also True --
    "power-limited" and "clock-locked" are different claims about the same
    run, not a refusal/advisory pair, and the record should say which
    happened rather than only the latter."""

    @property
    def status(self) -> str:
        if self.hard_refuse:
            return "refused"
        if self.advisory_reasons:
            return "advisory"
        return "decision"


def check_max_results(n_conformers: int, max_results: int) -> bool:
    """A default max_results=10,000 silently truncates per-batch
    before merging. Returns True iff compliant."""
    return max_results >= MIN_MAX_RESULTS_MULTIPLE * n_conformers


def check_arch_profile_comparable(arch_profile: str, gold_ref_arch_profile: str | None) -> bool:
    """A "native" single-arch build (e.g. ``ROSHAMBO_ARCH_NATIVE=ON``, the
    fast dev-loop path targeting only the build machine's own GPU) uses
    different register allocation/occupancy than a "fat" multi-arch build
    (the default wheel-distribution 80;86;89;90;120 library list).
    A throughput number from one is therefore not on the same footing as one
    from the other. Returns True iff there is no gold-ref comparison being
    made, or the two profiles match."""
    if gold_ref_arch_profile is None:
        return True
    return arch_profile == gold_ref_arch_profile


SATURATION_OK = "ok"
SATURATION_CORPUS_BOUND = "corpus-bound"
SATURATION_TOO_SMALL = "too-small"


def check_launch_saturation(shape: "LaunchShape") -> str:
    """Launch-size saturation, asked of the launch rather than of the corpus.

    Returns one of :data:`SATURATION_OK` (the launch reaches the measured
    throughput plateau, or comes within :data:`SATURATION_MARGIN` of it),
    :data:`SATURATION_CORPUS_BOUND` (it does not, and the corpus's group
    structure is *why* -- no setting could have done better, so the number is
    real but not a citable absolute), or :data:`SATURATION_TOO_SMALL` (it does
    not, and the corpus could have delivered more).

    The three-way answer is the whole point. The retired ``check_min_batch``
    was two-way, and a two-way answer has to choose between refusing every
    screen of a 10M corpus whose 14 groups cap the mean at 714,286 -- which is
    what it did -- and admitting 4,096-conformer launches. There is no
    reading of a single constant that does both.
    """
    target = SATURATION_MARGIN * SATURATING_CONFORMERS_PER_LAUNCH
    if shape.conformers_per_launch >= target:
        return SATURATION_OK
    ceiling = shape.ceiling
    if ceiling is not None and shape.conformers_per_launch >= (
            SATURATION_MARGIN * ceiling):
        return SATURATION_CORPUS_BOUND
    return SATURATION_TOO_SMALL


def launch_overhead_fraction(device_stage_seconds, total_s) -> float | None:
    """The share of a run's time paid once per launch, from its own counters.

    The quantity the launch-size rule is about, measured rather than proxied.
    ``device_stage_seconds`` is a :class:`ScreenStats` ``device_stages`` dict,
    which is populated only under ``Engine(stage_timing=True)``; None back means
    it was not recorded, which is not zero.

    A **lower bound** -- see :data:`PER_LAUNCH_STAGES` for which stages it can
    attribute and which it cannot. One 4,096-conformer-launch record puts it at 22.8%
    (precompute 4.687 s + free 2.184 s + alloc 0.034 s over 30.297 s wall,
    4,988 launches), which is 1.38 ms of fixed cost per launch.
    """
    if not device_stage_seconds or not total_s:
        return None
    if not device_stage_seconds.get("enabled"):
        return None
    per_launch = sum(float(device_stage_seconds.get(k, 0.0) or 0.0)
                     for k in PER_LAUNCH_STAGES)
    return per_launch / float(total_s)


def launch_conformers_from_stats(stats) -> float:
    """Conformers per ``score_pairs`` call, from a :class:`ScreenStats`.

    ``stats.n_conformers`` is the whole screen's total and
    ``stats.n_blocks`` is the number of launches -- which is the
    number of *coalesced runs*, not the number of on-disk blocks. Handing a
    check the corpus total instead answers "was the screen big" rather than
    "was each launch big", and those diverge by four orders of magnitude at the
    shipped default (a 1e7-conformer screen at 65,536 per launch).

    **Prefer** :meth:`LaunchShape.from_screen`, which carries this figure plus
    the group count that bounds it.
    """
    n = getattr(stats, "n_blocks", 0)
    return (stats.n_conformers / n) if n else 0.0


def check_repeats(n_repeats: int, n_warmup: int) -> bool:
    return n_warmup >= MIN_WARMUP_REPEATS and n_repeats >= REQUIRED_TIMED_REPEATS


def find_rate_fields(obj, _path: str = "", *, durations_count: bool = False
                     ) -> list[str]:
    """Key paths in a ``timing``/``metrics`` dict that look like a throughput.

    This is the structural half of the run-kind contract, and it is the
    half that makes the contract worth having. ``kind`` is a caller's
    assertion; this is evidence. A record carrying ``start_pairs_per_s`` is a
    rate run whatever it declares, so :func:`enforce` refuses the
    contradiction instead of letting the label decide which gates apply --
    otherwise the first person whose timing run is refused relabels it
    ``accuracy`` and the refusal disappears.

    Recurses into nested dicts and into lists of dicts, because accuracy
    records are free-form and a rate can be buried a level down (per-parameter
    rows are often a list of dicts).
    """
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{_path}.{k}" if _path else str(k)
            key = str(k)
            if _rate_bearing(key, v, durations_count):
                hits.append(path)
            hits.extend(find_rate_fields(v, path,
                                         durations_count=durations_count))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list, tuple)):
                hits.extend(find_rate_fields(v, f"{_path}[{i}]",
                                             durations_count=durations_count))
    return hits


def _rate_bearing(key: str, v, durations_count: bool) -> bool:
    """Is ``key: v`` a throughput figure? See the block comment above."""
    if key.lower() in _RATE_KEY_EXEMPT or not _looks_numeric(v):
        return False
    if _RATE_KEY_RE.search(key):
        # `_hz` is consulted HERE, after the rate pattern, not before it.
        # Checked first it neutralised every rate name it was appended to:
        # `throughput_hz`, `pairs_per_s_hz` and `qps_hz` all walked through,
        # and work-per-second in hertz is an honest spelling rather than
        # field-falsification. It only exempts a key that is a frequency and
        # nothing else once the `_hz` is removed.
        if _HZ_KEY_RE.search(key) and not _RATE_KEY_RE.search(
                _HZ_KEY_RE.sub("_", key)):
            return False
        # A `*_rate` spelling reaching here is decided by magnitude: a value in
        # [0,1] is a fraction. `hit_rate`, `tie_rate`, `win_rate`,
        # `baseline_active_rate` and the rest of that open-ended family are
        # retired by this line rather than by enumeration.
        pct = bool(_PCT_KEY_RE.search(key))
        if _FRACTION_KEY_RE.search(key) and _is_fraction(v, pct):
            return False
        return True
    if _HZ_KEY_RE.search(key):
        return False          # a bare frequency is a clock, not work per second
    if _FRACTION_KEY_RE.search(key):
        return not _is_fraction(v, bool(_PCT_KEY_RE.search(key)))
    return durations_count and bool(_DURATION_KEY_RE.search(key))


def find_recall_levels(metrics) -> dict:
    """Which recall levels a record carries, and whether they are distinct.

    Accuracy runs report conformer- **and** molecule-level recall. The trap
    is that the check can be satisfied trivially: a corpus with one conformer
    per molecule (as at group 5 of the 10M tuning and evaluation corpora) has
    molecule-level recall that *is* its conformer-level recall, and a
    "both levels present" gate passes on a corpus that cannot exercise the
    distinction. Making a degenerate pass *visible as one* is therefore a
    job for the corpus, not for these values:
    :attr:`RunContext.conformers_per_molecule` carries it and
    :func:`enforce` advises when it is ~1 or was never recorded. Inferring
    degeneracy by comparing the two levels' values directly is wrong on any
    record spanning corpora.

    Matching is on key patterns because accuracy records are free-form:
    ``recall_conformer_level`` (``metrics.evaluate_primary``), a ``conformer``
    sub-dict beside a ``molecule`` one, and
    ``conformer_recall`` all count.
    """
    found = {"conformer": False, "molecule": False}
    values = {"conformer": [], "molecule": []}

    def walk(o, ctx_conf=False, ctx_mol=False):
        if isinstance(o, dict):
            for k, v in o.items():
                ks = str(k)
                is_conf = ctx_conf or bool(_CONFORMER_RE.search(ks))
                is_mol = ctx_mol or bool(_MOLECULE_RE.search(ks))
                # A recall number under a conformer/molecule-flavoured key, or
                # a conformer/molecule-flavoured key under a recall-flavoured
                # one -- both orderings occur in committed records.
                if _RECALL_RE.search(ks) or ctx_conf or ctx_mol:
                    if isinstance(v, (int, float)):
                        if is_conf and not is_mol:
                            found["conformer"] = True
                            values["conformer"].append(float(v))
                        elif is_mol and not is_conf:
                            found["molecule"] = True
                            values["molecule"].append(float(v))
                walk(v, is_conf and not is_mol, is_mol and not is_conf)
        elif isinstance(o, list):
            for v in o:
                walk(v, ctx_conf, ctx_mol)

    walk(metrics)
    return {"conformer": found["conformer"], "molecule": found["molecule"],
            "n_conformer_values": len(values["conformer"]),
            "n_molecule_values": len(values["molecule"])}


def has_per_query_table(metrics) -> bool:
    """Whether a per-query table sits beside the aggregate (``METRICS.md``).

    "Aggregate as mean, min and p10 -- never a bare mean, and always publish
    the per-query table." A key matching ``per_query`` whose value is a list or
    dict with more than one entry counts; a scalar under that name does not,
    because that is an aggregate wearing the name of a table.
    """
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if _PER_QUERY_RE.search(str(k)) and isinstance(v, (list, dict)):
                if len(v) > 1:
                    return True
            if has_per_query_table(v):
                return True
    elif isinstance(metrics, list):
        return any(has_per_query_table(v) for v in metrics)
    return False


def find_percent_of_peak_fields_missing_clock(metrics: dict, _path: str = "") -> list[str]:
    """The project's own "5.6% of the MUFU.EX2 ceiling" figure was
    quoted without saying which clock the ceiling was computed at -- at the
    measurement's own clock the true figure was 12.9%, not 5.6%. Any metric
    whose name looks like a percent/fraction-of-peak-or-ceiling figure must
    carry an explicit normalizing-clock field alongside it in the same dict:
    either ``"<field>_clock_mhz"`` or a shared ``"normalizing_clock_mhz"``.
    An implicit assumption about which clock is exactly the error this
    function exists to catch.

    Best-effort key-pattern matching over a free-form, possibly-nested
    ``metrics`` dict (recurses into nested dicts, not into lists). Returns
    the list of offending key paths -- empty if none. Matches
    "pct_of_peak", "percent_of_ceiling", "fraction_of_peak", etc.,
    case-insensitively.
    """
    violations: list[str] = []
    if not isinstance(metrics, dict):
        return violations
    keys = set(metrics.keys())
    for k, v in metrics.items():
        path = f"{_path}.{k}" if _path else k
        if isinstance(v, dict):
            violations.extend(find_percent_of_peak_fields_missing_clock(v, path))
            continue
        # A clock-field key is never itself a percent-of-peak figure, even
        # if its name happens to embed one (e.g. "pct_of_peak_x_clock_mhz"
        # as the sibling of "pct_of_peak_x") -- without this, every valid
        # sibling clock field would be flagged as missing its own sibling.
        if k.endswith("_clock_mhz") or k == "normalizing_clock_mhz":
            continue
        if _PERCENT_OF_PEAK_RE.search(k):
            sibling = f"{k}_clock_mhz"
            if sibling not in keys and "normalizing_clock_mhz" not in keys:
                violations.append(
                    f"{path!r} looks like a percent-of-peak figure but neither "
                    f"{sibling!r} nor 'normalizing_clock_mhz' is present alongside "
                    "it in the same dict (an unrecorded normalizing clock is "
                    "exactly the error behind the project's wrong 5.6%-vs-12.9% "
                    "MUFU.EX2 figure)"
                )
    return violations


def check_clock_comparable(
    achieved_clock_mhz: float | None,
    gold_ref_clock_mhz: float | None,
    ratio_threshold: float = CLOCK_MATERIAL_RATIO,
) -> bool:
    """The model to follow is check_arch_profile_comparable(), extended to
    clocks: returns True iff there is no gold-ref clock to compare
    against, or the two achieved clocks are within ``ratio_threshold`` of
    each other.

    Unlike the arch-profile check, a gold-ref comparison whose *own* clock is
    unrecorded is not treated as "no problem" -- it returns False (not
    comparable), because "a comparison is being made but one side's operating
    point was never recorded" is precisely the failure to catch, not an
    absence of a claim to check.
    """
    if gold_ref_clock_mhz is None:
        return True
    if achieved_clock_mhz is None:
        return False
    lo, hi = sorted((achieved_clock_mhz, gold_ref_clock_mhz))
    if lo <= 0:
        return False
    return (hi / lo) <= ratio_threshold


def check_regime_comparable(regime: str, gold_ref_regime: str | None) -> bool:
    """Returns True iff there is no gold-ref regime to
    compare against, or the two regimes are the same.

    Unlike :func:`check_clock_comparable`'s clock field, a gold-ref
    comparison whose own regime is unrecorded is treated as "no problem" here
    (mirroring :func:`check_arch_profile_comparable`'s model instead) --
    the regime field postdates most of this project's run records, so ``None`` on the
    gold-ref side usually means "measured before the regime existed", not "a
    fact was dropped". ``enforce()`` still tags that case advisory, just not
    a hard refusal: only a *known* mismatch is refused outright, because
    the regime rule ("must never be combined") is stronger than the clock-
    ratio tolerance and admits no material-difference threshold.
    """
    if gold_ref_regime is None:
        return True
    return regime == gold_ref_regime


def canonical_padding_table(rows) -> list:
    """A padded-width histogram in one canonical form, so two can be compared.

    ``rows`` is any iterable of mappings or 4-tuples carrying ``bucket``, ``L``,
    ``L_color`` and ``n_conf``. Rows sharing a ``(bucket, L, L_color)`` key are
    summed and the result is sorted by that key, so the answer does not depend
    on how many GROUPS a bucket happened to be split into -- which is a
    property of the corpus builder's 2e6 cap, not of the data layout the kernel
    sees. ``L_color`` is 0 on a merged-color corpus and on a shape-only
    one; that a merged corpus reaches the same slot count with one number
    rather than two is exactly the difference this table has to make visible.

    Returned as a list of 4-lists rather than tuples so it round-trips through
    JSON unchanged and a stored table compares equal to a freshly built one.
    """
    agg: dict = {}
    for r in rows or ():
        if isinstance(r, dict):
            key = (int(r["bucket"]), int(r["L"]), int(r.get("L_color", 0) or 0))
            n = int(r["n_conf"])
        else:
            b, L, Lc, n = r
            key = (int(b), int(L), int(Lc or 0))
            n = int(n)
        agg[key] = agg.get(key, 0) + n
    return [[b, L, Lc, n] for (b, L, Lc), n in sorted(agg.items())]


def check_padding_comparable(padding_table, gold_ref_padding_table) -> bool:
    """The padding rule.

    "Two configurations compared for throughput must have identical L
    histograms, or the comparison is between data layouts, not kernels."
    Without it, the one rule that is *about* the library's shape could be
    violated by a run that passed every other check.

    Returns True iff there is no gold-ref comparison, or the two runs screened
    the same padded-width histogram. Follows :func:`check_clock_comparable`'s
    model, not :func:`check_arch_profile_comparable`'s: a comparison whose gold
    ref carries no table is **not** comparable, because "a comparison is being
    made and one side's padding was never recorded" is the same shape of
    failure as an unrecorded clock. The one asymmetry is that a gold ref
    recorded before this field existed has no table at all, so
    :func:`enforce` only consults this when at least one side has one.

    A deliberate padding comparison is licensed by
    ``RunRequest.padding_change_is_the_subject``, not by this function: the
    tables genuinely differ, and saying so is the point.
    """
    if gold_ref_padding_table is None:
        return True
    if padding_table is None:
        return False
    return (canonical_padding_table(padding_table)
            == canonical_padding_table(gold_ref_padding_table))


def check_device_comparable(gpu_uuid: str | None, gold_ref_gpu_uuid: str | None) -> bool:
    """Returns True iff there is no gold-ref device to compare against,
    or the two runs executed on the same physical GPU (by UUID, never by
    ``nvidia-smi`` index -- ``CUDA_VISIBLE_DEVICES`` renumbers indices).

    A cross-device *ratio* between two arms on one card in one session is
    permitted (the matched-architecture head-to-head
    against ROSHAMBO2); what this refuses is a ``gold_ref_run_id`` comparison
    silently spanning two different cards -- e.g. one arm timed on a host's
    GPU 0 and its ``gold_ref_run_id`` on GPU 1 -- which is a different claim
    from "the same device, twice" and must be stated, not assumed."""
    if gold_ref_gpu_uuid is None:
        return True
    return gpu_uuid == gold_ref_gpu_uuid


def is_cpu_backend(backend_name) -> bool:
    """Whether a record's ``backend_name`` names a CPU backend.

    One predicate, used by :func:`enforce` and by
    :meth:`RunContext.from_build`, because the alternative is two spellings of
    the same test that can drift apart -- and the thing that would drift is
    *which rules apply to a number*, which is the worst place for it.
    """
    return str(backend_name).lower() != "cuda"


def enforce(req: RunRequest, ctx: RunContext) -> EnforcementResult:
    advisory: list[str] = []

    # ---- hard refusals: no honest number can be salvaged ----
    if req.decision_grade and not ctx.build_derived:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                "this run's context was not derived from an inspected build, so "
                "every build-dependent fact below it is at its default -- and "
                "every one of those defaults is the permissive answer: a GPU "
                "build was checked, no forbidden flag was found, the "
                "architecture probe ran. None of that was established. Build "
                "the context with RunContext.from_build(verify_build(), "
                "numerics.check_build(), ...)."
            ),
        )

    if req.kind not in RUN_KINDS:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                f"kind={req.kind!r} is not one of {sorted(RUN_KINDS)}."
            ),
        )

    # The run-kind contract is structural, not declarative. A record carrying a
    # throughput figure is a rate run whatever it declares, so the contradiction
    # is refused rather than the gates being skipped -- otherwise `accuracy`
    # becomes the escape hatch for any inconvenient timing refusal.
    if req.kind == RUN_KIND_ACCURACY:
        # `timing` is a WHITELIST, not a detector. See
        # `_ACCURACY_TIMING_ALLOWED`: four blacklists in a row were defeated by
        # spellings nobody enumerated, so on the one block where rates
        # conventionally live the question is inverted -- an unrecognised key
        # is a refusal rather than a pass. That closes the whole spelling game
        # for `kernel_s`, `qps_mean`, `throughput_hz` and every successor.
        stray = sorted(k for k in (ctx.timing or {})
                       if str(k) not in _ACCURACY_TIMING_ALLOWED)
        # The other blocks keep the detector: they are not where rates go, and
        # a whitelist there would refuse half the repo.
        rate_fields = (find_rate_fields(ctx.metrics)
                       + find_rate_fields(ctx.record_blocks))
        if stray:
            return EnforcementResult(
                hard_refuse=True,
                refusal_reason=(
                    "this run declares kind='accuracy' but its `timing` block "
                    f"carries {', '.join(stray[:6])}"
                    + (" ..." if len(stray) > 6 else "")
                    + ". An accuracy run reports recall and ranking and takes "
                    "no timing; `timing` is where a rate run puts its numbers, "
                    "so for that block the contract allows only "
                    f"{sorted(_ACCURACY_TIMING_ALLOWED)} and refuses anything "
                    "else rather than trying to recognise every spelling of a "
                    "rate. Either drop the timing or declare kind='rate' and "
                    "meet the timing gates."
                ),
            )
        if rate_fields:
            return EnforcementResult(
                hard_refuse=True,
                refusal_reason=(
                    "this run declares kind='accuracy' but its record carries "
                    f"throughput figures: {', '.join(sorted(rate_fields)[:6])}"
                    + (" ..." if len(rate_fields) > 6 else "")
                    + ". An accuracy run reports recall and ranking and no "
                    "rate; a record with both is malformed, not exempt. Either "
                    "drop the rate or declare kind='rate' and meet the timing "
                    "gates."
                ),
            )

    if ctx.measurement_regime not in MEASUREMENT_REGIMES:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                f"measurement_regime={ctx.measurement_regime!r} is not one of "
                f"{sorted(MEASUREMENT_REGIMES)}. This field "
                "has no default precisely so a run cannot be recorded without "
                "someone deciding which regime it is in."
            ),
        )

    if ctx.gold_ref_run_id is not None and not check_regime_comparable(
            ctx.measurement_regime, ctx.gold_ref_measurement_regime):
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                f"this run's measurement_regime={ctx.measurement_regime!r} "
                f"differs from gold_ref_run_id={ctx.gold_ref_run_id!r}'s "
                f"measurement_regime={ctx.gold_ref_measurement_regime!r} "
                "(a calibration number and a headline number "
                "are numbers from different clocks by construction, and may "
                "never be combined into one ratio, table, or derived "
                "quantity -- this is a hard refusal, not an advisory tag, "
                "because the regime rule admits no material-difference "
                "threshold the way the clock-ratio check does)."
            ),
        )

    if ctx.gold_ref_run_id is not None and not check_device_comparable(
            ctx.gpu_uuid, ctx.gold_ref_gpu_uuid):
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                f"this run executed on gpu_uuid={ctx.gpu_uuid!r} but "
                f"gold_ref_run_id={ctx.gold_ref_run_id!r} executed on "
                f"gpu_uuid={ctx.gold_ref_gpu_uuid!r} -- a comparison against "
                "a prior run must say whether it is the same physical card "
                "twice or two different cards; this run's context claims "
                "neither explicitly (a cross-device ratio is only "
                "licensed when it is a stated, deliberate comparison, not an "
                "accidental one)."
            ),
        )

    if (ctx.gold_ref_run_id is not None
            and (ctx.padding_table is not None
                 or ctx.gold_ref_padding_table is not None)
            and not check_padding_comparable(ctx.padding_table,
                                             ctx.gold_ref_padding_table)
            and not req.padding_change_is_the_subject):
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                f"this run screened a different padded-width histogram from "
                f"gold_ref_run_id={ctx.gold_ref_run_id!r} "
                f"({canonical_padding_table(ctx.padding_table)} vs "
                f"{canonical_padding_table(ctx.gold_ref_padding_table)}) -- "
                "two configurations compared for throughput must "
                "have identical L histograms, or the comparison is between "
                "data layouts and not between kernels. If the padding IS the "
                "thing under test, say so with "
                "RunRequest(padding_change_is_the_subject=True); the tables "
                "are carried in the record either way."
            ),
        )

    if not ctx.is_release_build:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                "build is not Release (build stamp); a non-Release/debug "
                "build's timing cannot be recorded as a benchmark result."
            ),
        )

    if req.decision_grade and ctx.build_source_dirty is True:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                "this binary was compiled from a working tree that matched no "
                "commit -- `git status --porcelain` on the compiled source "
                f"tree ({ctx.build_source_subtree or 'src/roshambo3/backend'}) "
                "was non-empty at build time "
                f"(compiled source sha256={ctx.build_source_sha256}, dirty "
                f"paths in the record's build block). ROSHAMBO3_GIT_SHA "
                "therefore names a commit that is NOT the source this binary "
                "contains, and a decision-grade number may not cite a commit "
                "that does not describe it. An advisory run may proceed "
                "on a patched tree -- that is what a patched tree is for."
            ),
        )

    # The no-GPU-build refusal, scoped to the runs it is about. Absence of a CUDA
    # extension disqualifies a GPU number and is the NORMAL state for a CPU one
    # -- screening without a GPU is outcome 1 of the CPU workstream, and the CI
    # `cpu` job builds exactly this. Unscoped, this refused every number from
    # the configuration the work exists to serve.
    if not ctx.checked_a_gpu_build and not is_cpu_backend(ctx.backend_name):
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                "there is no CUDA extension in this build, so no device "
                "arithmetic was checked and no kernel ran. `check_build().ok` is "
                "vacuously true here -- it means 'found no forbidden flag', and "
                "there were no flags to inspect. A timed GPU number from such a "
                "build cannot exist."
            ),
        )

    # ...and the CPU twin, which is the same refusal about the other artifact.
    # Without it, scoping the rule above would have turned a refusal into a
    # hole: a CPU run whose CPU build was never inspected would sail through
    # having checked nothing.
    # Deliberately NOT gated on `decision_grade`, and that is a choice rather
    # than an omission -- recorded here because review flagged it as stricter
    # than the CUDA analogue, where unstamped host flags are only an advisory.
    #
    # The asymmetry is intended. The numerics policy DEFERS the CPU host flags
    # rather than refusing them, which makes noticing them the entire
    # safeguard; an advisory-grade number from a binary nobody inspected is
    # still a number whose `-ffast-math` nobody would have seen, and advisory
    # numbers get quoted. The CUDA half can afford the softer
    # treatment because its forbidden flags ARE refused outright.
    #
    # Revisit if the cost shows up: the symptom would be legitimate
    # exploratory CPU runs unable to record anything at all.
    if is_cpu_backend(ctx.backend_name) and not ctx.checked_a_cpu_build:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                "this run used a CPU backend, but no CPU extension was "
                "inspected -- so the flags it was compiled with were never "
                "read, and `-ffast-math` and friends would not have been "
                "noticed. The numerics policy defers those flags rather than "
                "refusing them, which makes noticing them the whole safeguard. "
                "Build the context with RunContext.from_build(verify_build(), "
                "numerics.check_build(), backend_name='cpu', ...)."
            ),
        )

    if ctx.numerics_violations:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                "the loaded build does not use the pinned arithmetic: "
                + "; ".join(ctx.numerics_violations)
                + ". Every perturbing build axis was measured to reach the "
                "same output distribution, so there is no small-perturbation "
                "regime in which this is harmless; a number from this build is "
                "not comparable with a number from any other."
            ),
        )

    if ctx.percent_of_peak_violations:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                "a percent-of-peak figure's normalizing clock is not recorded "
                ": " + "; ".join(ctx.percent_of_peak_violations)
            ),
        )

    # ---- launch-size saturation -------------------------------------------
    # A fixed 1e6 floor would refuse every screen of a 10M corpus at every
    # setting, because launches do not cross group boundaries. What this
    # protects is that the number reported is the machine's throughput rather than the launch structure's, which is a
    # question about the launch and about what the corpus could have delivered.
    shape = req.launch or LaunchShape(
        conformers_per_launch=float(req.n_conformers),
        n_launches=1,
        n_conformers=int(req.n_conformers),
    )
    # SATURATING_CONFORMERS_PER_LAUNCH is 65,536 because it was MEASURED as
    # this engine's plateau on a GPU. There is no launch on a CPU and no reason
    # its plateau would sit at the same number, so applying it there is a GPU
    # constant wearing CPU clothes -- and it refused every CPU run under 32,768
    # conformers, which is most of them. The CPU analogue (thread-count and
    # work-chunk saturation) must be MEASURED, not assumed, and has not been.
    # Until then ctx.cpu_threads carries the fact and no rule reads it.
    # The launch-size rule is about whether a launch was big enough to reach the
    # THROUGHPUT plateau. An accuracy run's launch size changes how long it
    # takes and not what it retrieves, so the gate has nothing to say about it.
    saturation = (SATURATION_OK
                  if (req.kind != RUN_KIND_RATE
                      or is_cpu_backend(ctx.backend_name))
                  else check_launch_saturation(shape))
    if saturation == SATURATION_TOO_SMALL:
        ceiling = shape.ceiling
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                f"{shape.conformers_per_launch:,.0f} conformers per launch is "
                f"below {SATURATION_MARGIN:g} x "
                f"{SATURATING_CONFORMERS_PER_LAUNCH:,} = "
                f"{SATURATION_MARGIN * SATURATING_CONFORMERS_PER_LAUNCH:,.0f}, "
                "the measured point at which this engine is within ~10% of its "
                "throughput plateau (a 4,096 launch of a 10M "
                "corpus costs 2.86x against 65,536)"
                + (f", and this corpus's {shape.n_groups} groups would have "
                   f"permitted a mean launch of {ceiling:,.0f} -- launches "
                   "never cross a group boundary, so that is its ceiling, not "
                   "its floor"
                   if ceiling is not None else
                   " (this run recorded no group structure, so no corpus "
                   "ceiling could be established in its defence -- pass "
                   "RunRequest(launch=LaunchShape.from_screen(stats, source)))")
                + "."
            ),
        )

    if req.kind == RUN_KIND_RATE and not check_repeats(req.n_repeats,
                                                       req.n_warmup):
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                f"requires >= {MIN_WARMUP_REPEATS} warm-up runs and "
                f"{REQUIRED_TIMED_REPEATS} timed repeats; got "
                f"n_warmup={req.n_warmup}, n_repeats={req.n_repeats}."
            ),
        )

    if req.decision_grade and ctx.gpu_lock_acquisition_failed:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                "GPU lock is held by "
                "another run; a second decision-grade GPU benchmark must not "
                "run concurrently with it (GPU arbitration protocol)."
            ),
        )

    burst = ctx.sustained_kernel_s < MIN_SUSTAINED_KERNEL_S
    if req.kind == RUN_KIND_RATE and burst and req.claim_non_burst:
        return EnforcementResult(
            hard_refuse=True,
            refusal_reason=(
                f"sustained kernel time {ctx.sustained_kernel_s:.1f}s is "
                f"below the {MIN_SUSTAINED_KERNEL_S:.0f}s threshold; this run "
                "cannot be recorded with burst=False (clock/"
                "thermal drift is real on short runs, and burst numbers must "
                "not be used for 1e9/1e11 projections)."
            ),
        )

    # ---- accuracy gates: skipping a timing bar is NOT skipping grading -----
    # These gates bind `kind="accuracy"`. The reciprocal design was measured
    # and the asymmetry kept: the single RunRecord status
    # grades the declared measurement kind, while a rate run may carry
    # supplemental recall/ranking diagnostics. Applying these refusals to all
    # such fields would re-grade 72 existing rate-ledger records and conflate a rate grade with a
    # separate accuracy claim. A rate grade therefore certifies rate evidence
    # only; callers needing a decision-grade accuracy result record an
    # accuracy-kind run, which must not carry timing/rate fields.
    if req.kind == RUN_KIND_ACCURACY:
        levels = find_recall_levels(ctx.metrics)
        if not (levels["conformer"] and levels["molecule"]):
            missing = [k for k in ("conformer", "molecule") if not levels[k]]
            return EnforcementResult(
                hard_refuse=True,
                refusal_reason=(
                    "an accuracy run must report recall at BOTH levels "
                    f"(two-level recall); this record has no {' or '.join(missing)}"
                    "-level recall. Molecule-level recall falls below "
                    "conformer-level whenever a molecule has several "
                    "conformers, so the conformer number alone overstates what "
                    "a user gets."
                ),
            )

        if not has_per_query_table(ctx.metrics):
            return EnforcementResult(
                hard_refuse=True,
                refusal_reason=(
                    "an accuracy run must publish the per-query table beside "
                    "the aggregate (doc/METRICS.md: 'never a bare mean'). No "
                    "key matching per_query with more than one entry was found "
                    "in this record's metrics. The 1e-4 quantile threshold "
                    "ranged widely across queries, and a mean hides it."
                ),
            )

        if req.chose_a_value is not None:
            fit, rep = ctx.fit_corpus, ctx.report_corpus
            if not fit or not rep or fit == rep:
                return EnforcementResult(
                    hard_refuse=True,
                    refusal_reason=(
                        f"this run declares it chose {req.chose_a_value!r}, so "
                        "the fit/report rule requires a fit corpus and a report corpus "
                        "that are named and DIFFERENT; got "
                        f"fit_corpus={fit!r}, report_corpus={rep!r}. Fitting "
                        "and reporting on the same corpus inflated recall by "
                        "+0.0092 at K=30."
                    ),
                )

        # The CPU ISA and vector width are NOT re-checked here, deliberately.
        # The CPU rules below already refuse every CPU run that fails to
        # record either, whatever its kind and whatever its intent. A
        # duplicate check here would be unreachable in all four combinations
        # of (kind, decision_grade), and a gate that cannot fire is a mistake.

        # The degenerate two-level-recall pass, made visible rather than
        # refused. A corpus with one conformer per molecule satisfies the gate
        # above structurally without exercising it, and that is worth
        # recording, not blocking -- such a corpus is legitimate to measure on.
        cpm = ctx.conformers_per_molecule
        if cpm is None:
            advisory.append(
                "conformers-per-molecule was not recorded, so a molecule-level "
                "recall that cannot differ from the conformer-level one is "
                "indistinguishable from one that could")
        elif cpm <= 1.0 + 1e-9:
            advisory.append(
                f"the screened corpus carries {cpm:g} conformers per molecule, "
                "so its molecule-level recall IS its conformer-level recall by "
                "construction -- the two-level rule is satisfied structurally "
                "and not exercised")

    # ---- auto-tag advisory: a real number, just not citable as a decision basis ----
    if ctx.git_dirty:
        advisory.append("git working tree is dirty")

    if req.decision_grade and ctx.build_source_dirty is None:
        advisory.append(
            "this binary does not say which source it was compiled from: its "
            "stamp carries a commit and no content hash, so nothing in this "
            "record could contradict the commit if it named the wrong source"
            ". Rebuild against a tree that carries scripts/source_stamp.py")
    elif ctx.build_source_stamp_phase == "configure":
        advisory.append(
            "this binary's source stamp was taken at CONFIGURE time, not build "
            "time, so a configure-then-edit-then-build cycle would have stamped "
            "the pre-edit source as clean")

    # A CPU run does not take the GPU lock, and reporting that it failed
    # to is noise that trains a reader to skim advisories. Its obligation is the
    # CPU lock, which is a hard refusal below -- stricter, not laxer.
    if (req.decision_grade and not ctx.gpu_lock_held
            and not is_cpu_backend(ctx.backend_name)):
        advisory.append("GPU lock was not held for a decision-grade run")

    if ctx.other_process_on_gpu:
        advisory.append(
            "another process was present on the GPU per "
            "--query-compute-apps (shared-machine contention)"
        )

    if (not ctx.clocks_locked) and ctx.throttle_fraction > THROTTLE_ADVISORY_THRESHOLD:
        advisory.append(
            f"clocks unlocked and throttling observed on "
            f"{ctx.throttle_fraction:.0%} of samples (> "
            f"{THROTTLE_ADVISORY_THRESHOLD:.0%} threshold)"
        )

    if (ctx.gold_ref_run_id is not None and req.padding_change_is_the_subject
            and not check_padding_comparable(ctx.padding_table,
                                             ctx.gold_ref_padding_table)):
        advisory.append(
            "the padded-width histogram differs from gold_ref_run_id's and the "
            "request declares that as the subject of the comparison: "
            "this ratio is between two data layouts, and any statement it "
            "supports about the KERNEL needs a third arm holding L fixed")

    if saturation == SATURATION_CORPUS_BOUND:
        advisory.append(
            f"{shape.conformers_per_launch:,.0f} conformers per launch is "
            f"below {SATURATION_MARGIN * SATURATING_CONFORMERS_PER_LAUNCH:,.0f}"
            f", but this corpus's {shape.n_groups} groups cap the mean launch "
            f"at {shape.ceiling:,.0f} and no setting could have done better "
            "(launches never cross a group boundary). The ratio this "
            "run supports is real; its absolute throughput is not the "
            "machine's")

    # Kind-scoped: a rate-shaped gate. Its advisory text describes a fallback
    # the accuracy path never takes, since saturation is short-circuited to OK
    # for accuracy runs, and unscoped it would keep a clean accuracy run on a
    # GPU from reaching decision.
    if (req.kind == RUN_KIND_RATE and req.launch is None
            and not is_cpu_backend(ctx.backend_name)):
        advisory.append(
            "this run did not record its launch structure, so the launch-size "
            "rule fell back to RunRequest.n_conformers -- the field a caller "
            "once handed a corpus total, which is how a 4,096-conformer launch "
            "passed a 1,000,000-conformer check. Pass "
            "RunRequest(launch=LaunchShape.from_screen(stats, source))")

    overhead = launch_overhead_fraction(ctx.device_stage_seconds,
                                        ctx.measured_total_s)
    if overhead is not None and overhead > MAX_LAUNCH_OVERHEAD_FRACTION:
        advisory.append(
            f"{overhead:.1%} of this run's time was spent on work paid once "
            f"per launch ({', '.join(PER_LAUNCH_STAGES)}), above the "
            f"{MAX_LAUNCH_OVERHEAD_FRACTION:.0%} threshold -- and that is a "
            "lower bound, since the per-launch components of the kernel and of "
            "h2d cannot be separated within one run")

    if ctx.ground_truth_audit_passed is False:
        advisory.append("the ground-truth correctness audit failed on this run")

    if (req.max_results_applied_per_batch and
            not check_max_results(req.n_conformers, req.max_results)):
        advisory.append(
            f"max_results={req.max_results} < 10 * n_conformers="
            f"{10 * req.n_conformers} (per-batch truncation)"
        )

    if ctx.arch_probe_error:
        advisory.append(
            "the loaded extension's architecture list could not be read, so "
            "the architecture check (a wrong-architecture build that JIT-compiles at "
            "runtime) was not checked: " + ctx.arch_probe_error)

    # ---- the CPU rules, gated on a CPU backend having produced this ----
    if is_cpu_backend(ctx.backend_name):
        if ctx.cpu_isa_selected is None:
            return EnforcementResult(
                hard_refuse=True,
                refusal_reason=(
                    "this run does not record which SIMD dispatch target the "
                    "process actually selected. That is the CPU analogue of a "
                    "wrong-architecture build and it fails the same way: a build "
                    "that silently ran a "
                    "baseline scalar path looks like a slow kernel rather than "
                    "a broken build, and with a dispatching SIMD library it is "
                    "reachable by accident. Asking the binary what targets it "
                    "contains is a different question and does not "
                    "substitute."))
        # This gate says so itself -- "a decision-grade CPU TIMING must
        # hold flock". Contention changes how long an accuracy run takes and
        # not what it retrieves, so the lock is a rate requirement; the machine being busy is still recorded in the
        # operating point either way, so nothing is hidden by not refusing.
        if (req.kind == RUN_KIND_RATE and req.decision_grade
                and not ctx.cpu_lock_held):
            return EnforcementResult(
                hard_refuse=True,
                refusal_reason=(
                    "a decision-grade CPU timing must hold "
                    "flock $ROSHAMBO3_CPU_LOCK for its duration. Both machines "
                    "are shared, and a GPU screen holding the GPU lock is heavy "
                    "CPU load that a GPU-only contention check cannot see "
                    "(GPU arbitration protocol, CPU twin)."))
        if ctx.cpu_vector_width is None:
            return EnforcementResult(
                hard_refuse=True,
                refusal_reason=(
                    "this run does not record the SIMD vector width its scores "
                    "were computed at (the dispatch target's lane count). "
                    "The numerics policy accepts that a CPU "
                    "score is a FUNCTION of that width -- an 8-lane reduction "
                    "and a 16-lane one do not produce the same float -- in "
                    "exchange for every record naming it, and a record "
                    "which does not name its width "
                    "cannot be compared with anything. Refused rather than "
                    "advised because the comparison it corrupts is one where "
                    "both runs look fine."))
        # Both of these say "timed"/"claim a clock" in their own text.
        # Affinity and the achieved-clock distribution govern how long a run
        # takes, not what it retrieves, so on an accuracy run they would tag
        # every record advisory forever and no accuracy number could ever grade
        # decision.
        if req.kind == RUN_KIND_RATE and ctx.cpu_affinity is None:
            advisory.append(
                "this timed CPU run was not pinned to a verified CPU set "
                "(binding was unsupported or not requested). The scheduler "
                "may migrate workers between cores or core classes. Record "
                "scheduler policy and observed placement when available; "
                "QoS hints do not establish CPU binding.")
        if req.kind == RUN_KIND_RATE and (
                ctx.achieved_cpu_mhz is None
                or not ctx.achieved_cpu_mhz.get("n_samples")):
            advisory.append(
                "this CPU run sampled no frequencies, so it may not claim a "
                "clock. 'did not look' and 'the clock was steady' are different "
                "claims, and the rule that numbers from different clocks "
                "must not be combined needs a distribution to apply to.")
        # Same reasoning as the lock, the affinity and the clock above, applied
        # consistently: contention changes how long a run takes and not what it
        # retrieves. Un-gated, on a permanently-shared machine it keeps
        # accuracy runs from reaching `decision`. The foreign-process
        # list stays in the operating point either way, so nothing is hidden.
        if req.kind == RUN_KIND_RATE and (ctx.cpu_busy
                                          or ctx.cpu_foreign_processes):
            advisory.append(
                f"another process was using this machine during the run "
                f"({len(ctx.cpu_foreign_processes)} over the busy threshold). "
                "As on the GPU, holding the lock does not mean the machine is "
                "yours: correctness work runs unlocked by design.")

    if not ctx.checked_host_flags:
        advisory.append(
            "this build's stamp records no host compiler flags, so the host "
            "half of the extension -- which prepares and reduces what the "
            "kernel sees -- was not inspected. An empty host-flag advisory "
            "list means 'nothing to check' here, not 'checked and clean'. "
            "Advisory rather than a refusal because the host-flag effect is "
            "not measured on this project, unlike the nvcc flags.")

    for a in ctx.numerics_advisories:
        advisory.append(a)

    if not check_arch_profile_comparable(ctx.arch_profile, ctx.gold_ref_arch_profile):
        advisory.append(
            f"build arch_profile={ctx.arch_profile!r} differs from "
            f"gold_ref_run_id's arch_profile={ctx.gold_ref_arch_profile!r} "
            "(native vs fat builds have different register allocation/"
            "occupancy; not directly comparable)"
        )

    # ---- power-limited vs clock-locked ----
    power_limited = ctx.sw_power_cap_fraction > SW_POWER_CAP_ADVISORY_THRESHOLD
    if ctx.clocks_locked and power_limited:
        advisory.append(
            f"sw_power_cap Active on {ctx.sw_power_cap_fraction:.0%} of samples "
            "despite a successful clock-lock request -- this run is "
            "power-limited, not clock-locked (the lock command can "
            "report success while the power ceiling still binds instead, e.g. "
            "1800 MHz requested sitting at 1762 MHz with the cap Active; "
            "'clocks locked' and 'power-limited' are different claims)"
        )

    # ---- cross-operating-point comparisons ----
    if ctx.gold_ref_run_id is not None:
        # A comparison is being made (gold_ref_run_id says so) -- checked
        # directly against None here rather than solely through
        # check_clock_comparable()'s return value, because that function's
        # own contract (see its docstring) treats gold_ref_clock_mhz=None as
        # "no comparison to make" and returns True -- correct for a
        # standalone caller with no other signal, but here gold_ref_run_id
        # already establishes a comparison exists, so an unrecorded clock on
        # either side must not be silently folded into "comparable".
        if ctx.achieved_clock_mhz_median is None or ctx.gold_ref_achieved_clock_mhz_median is None:
            advisory.append(
                f"comparing against gold_ref_run_id={ctx.gold_ref_run_id!r} but "
                "the achieved clock is not recorded on one or both sides -- an "
                "unrecorded normalizing clock is exactly the error "
                "behind the project's wrong 5.6%-vs-12.9% MUFU.EX2 figure"
            )
        elif not check_clock_comparable(ctx.achieved_clock_mhz_median,
                                         ctx.gold_ref_achieved_clock_mhz_median):
            advisory.append(
                f"achieved clock {ctx.achieved_clock_mhz_median:.0f} MHz differs "
                f"from gold_ref_run_id={ctx.gold_ref_run_id!r}'s achieved clock "
                f"{ctx.gold_ref_achieved_clock_mhz_median:.0f} MHz by more than "
                f"{CLOCK_MATERIAL_RATIO}x -- numbers taken at different "
                "clocks must never be combined into one ratio, table, or "
                "derived quantity"
            )

        # A *known* regime mismatch is a hard refusal above; an *unknown*
        # gold-ref regime (a run that predates the field) is advisory --
        # it cannot be ruled comparable, but it also cannot be asserted
        # incomparable, so it is flagged rather than refused.
        if ctx.gold_ref_measurement_regime is None:
            advisory.append(
                f"comparing against gold_ref_run_id={ctx.gold_ref_run_id!r} but "
                "its measurement_regime is not recorded (it may predate "
                "the field) -- cannot confirm this is a same-regime "
                "comparison"
            )

        gold_prof = ctx.gold_ref_profiler_control_state or "none"
        if ctx.profiler_control_state == "undetermined":
            advisory.append(
                f"this run's profiler clock-control state is undetermined "
                f"(comparing against gold_ref_run_id={ctx.gold_ref_run_id!r}) -- "
                "cannot rule out an ncu/nsys clock seizure affecting comparability"
            )
        if gold_prof == "undetermined":
            advisory.append(
                f"gold_ref_run_id={ctx.gold_ref_run_id!r}'s profiler clock-control "
                "state is undetermined -- cannot rule out an ncu/nsys clock seizure "
                "affecting comparability"
            )
        if ("undetermined" not in (ctx.profiler_control_state, gold_prof)
                and ctx.profiler_control_state != gold_prof):
            advisory.append(
                f"profiler clock-control state differs: this run="
                f"{ctx.profiler_control_state!r} vs gold_ref_run_id="
                f"{ctx.gold_ref_run_id!r}={gold_prof!r} -- ncu/nsys seize the GPU "
                "clock, so a profiled run and an unprofiled run are not the "
                "same operating point even if the raw clocks.sm reading looks "
                "similar"
            )

    return EnforcementResult(
        hard_refuse=False,
        refusal_reason=None,
        advisory_reasons=advisory,
        burst=burst,
        power_limited=power_limited,
    )

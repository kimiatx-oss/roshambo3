"""The public surface: one function for the common case.

    import roshambo3 as r3
    hits = r3.screen("queries.sdf", "$EXTERNAL_DATA", k=100_000)
    print(hits[0].gids[:10], hits[0].tanimoto_shape[:10])

Everything below that is the same objects, assembled by hand -- an
:class:`~roshambo3.engine.Engine`, a :class:`~roshambo3.engine.Source`, a
:class:`~roshambo3.engine.Sink` and a :class:`~roshambo3.engine.ScreenConfig`.
There is deliberately no options-only entry point that hides them: a screen on a
1e11-conformer corpus is a system, and a caller who needs to place its parts
should not have to fight a facade to reach them.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import wraps
from typing import Any, Optional, Sequence, Union

import numpy as np

from .engine import (CPUWorkerOptions, DEFAULT_PREFETCH_DEPTH, Engine, MoleculeTopK,
                     ScreenConfig, TopK, conformers_per_launch, make_source,
                     from_rdkit)
from .engine.query import QuerySet
from .engine.sink import HostTopK
from .format import Catalog, Corpus
from .results import QueryHits, ScreenResult

PathLike = Union[str, os.PathLike]


def load_queries(source: Union[PathLike, QuerySet, Sequence],
                 *, color_generator=None, backend=None) -> QuerySet:
    """Build a :class:`QuerySet` from an SDF path, RDKit molecules, or itself.

    Queries are prepared in process, by the same code the corpus builder uses --
    they have to be in the same coordinate frame as the library molecules or the score
    means nothing.
    """
    if isinstance(source, QuerySet):
        return source
    if isinstance(source, (str, os.PathLike)):
        from rdkit import Chem
        path = str(source)
        if path.lower().endswith((".sdf", ".sd", ".mol")):
            supplier = Chem.SDMolSupplier(path, removeHs=False)
        elif path.lower().endswith((".smi", ".smiles")):
            raise ValueError(
                "a query needs 3D coordinates, and a SMILES file has none. "
                "Generate conformers first, then pass the SDF.")
        else:
            raise ValueError(f"unrecognized query file type: {path}")
        mols = [m for m in supplier if m is not None]
        if not mols:
            raise ValueError(f"no readable molecules in {path}")
        return from_rdkit(mols, color_generator=color_generator, backend=backend)
    return from_rdkit(list(source), color_generator=color_generator, backend=backend)


def open_corpus(corpus: Union[PathLike, Corpus]) -> Corpus:
    from .format.r3g.reader import open_corpus as open_r3g
    if isinstance(corpus, Corpus):
        return corpus
    return open_r3g(str(corpus))


def _check_color(config, corpus, color_generator, *, trusted_definitions=()) -> None:
    """Check a color screen's preconditions, naming which one is unmet.

    The preconditions are the ordinary ones a color screen always had and
    always will:

    * the corpus must actually carry color (a screen against a shape-only
      corpus would silently report a color Tanimoto of exactly zero for
      every hit otherwise -- a wrong answer dressed as a right one);
    * the queries need features too, which needs a ``color_generator``.

    The third precondition this function used to carry -- that the pose search
    be the analytic one -- is gone: the field path carries color now.
    The field is still the query's *shape* field and there is still no color
    field; what changed is that the color gradient never needed one. It is an
    analytic pair sum over a small same-type set, evaluated in the
    same loop that samples the shape texture.
    """
    if not getattr(config, "color", False):
        return
    if not corpus.has_color:
        raise ValueError(
            f"color was requested against {getattr(corpus, 'path', 'this corpus')}, "
            f"which stores no color features -- it was not built with "
            f"color_generator=... (see roshambo3.prep.prepare). A screen "
            f"would otherwise succeed and report a color Tanimoto of exactly "
            f"zero for every hit rather than failing.")
    from . import definitions
    if color_generator is None:
        definitions.check_corpus_color(
            corpus, None, trusted_definitions=trusted_definitions)
        return
    # The fourth precondition, and the only one that is silent when unmet.
    # The query's type ids come from this generator's FEATURES_ENUM and the
    # library's from whichever generator built the corpus. Every array is the
    # same shape either way, the canonical diagonal matrix is its own
    # permutation, and the score that comes out is a plausible number computed
    # against the wrong features. This is the one place both objects are in
    # hand, so it is checked here.
    definitions.check_corpus_color(
        corpus, color_generator, trusted_definitions=trusted_definitions)


def _format3_color_generator(corpus, color_generator, trusted_definitions):
    """Resolve the descriptor-backed generator only after the trust check."""
    descriptor = getattr(corpus, "descriptor", None)
    if (color_generator is None and isinstance(descriptor, dict)
            and "format" in descriptor and descriptor.get("color") is not None):
        from .format.r3g.descriptor import isolated_generator_from_descriptor
        return isolated_generator_from_descriptor(
            descriptor["color"], trusted_definitions=trusted_definitions)
    return color_generator


def _resolve_color_generator(queries, corpus, color_generator, trusted_definitions):
    """Resolve an inferred generator only for queries featurized here.

    A prepared QuerySet carries type ids and matrices but no full semantic color
    descriptor. Its producer must therefore be supplied explicitly so the
    format-3 gate can compare the query descriptor with the corpus.
    """
    if (isinstance(queries, QuerySet) and color_generator is None
            and isinstance(getattr(corpus, "descriptor", None), dict)
            and "format" in corpus.descriptor):
        raise ValueError(
            "a format-3 color screen with a prepared QuerySet requires the "
            "color_generator that produced its features; pass it explicitly "
            "so its descriptor can be checked against the corpus")
    return _format3_color_generator(corpus, color_generator, trusted_definitions)


def _clear_session_on_failure(method):
    @wraps(method)
    def guarded(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except BaseException:
            engine = kwargs.get("engine")
            if isinstance(engine, Engine):
                engine._worker_query_cache = None
                engine._worker_self_cache = None
            raise
    return guarded


@_clear_session_on_failure
def screen(queries: Union[PathLike, QuerySet, Sequence],
           corpus: Union[PathLike, Corpus],
           *,
           k: int = 1000,
           config: Optional[ScreenConfig] = None,
           device_id: int = 0,
           groups: Optional[Sequence[int]] = None,
           workers: Optional[int] = None,
           prefetch_depth: int = DEFAULT_PREFETCH_DEPTH,
           launch_conformers: Optional[int] = None,
           color_generator=None,
           level: str = "conformer",
           stage_timing: bool = False,
           backend: Optional[str] = None,
           engine: Optional[Engine] = None,
           trust_definitions: Sequence[str] = ()) -> ScreenResult:
    """Screen ``queries`` against ``corpus`` and return the top ``k`` per query.

    ``k`` is a count, but think of it as a quantile: the top 1e5 of 1e9 is the
    1e-4 quantile and of 1e11 the 1e-6, and those are different physical regimes
    with different score thresholds. A ``k`` that is a large fraction of a small
    corpus is not measuring what the same ``k`` measures on a large one.

    ``level`` chooses **what one hit is**, and the two are different questions
    with different answers:

    ``"conformer"``
        the top ``k`` conformers. A molecule with several conformers among them
        occupies several of the ``k`` slots.
    ``"molecule"``
        the top ``k`` molecules, each scored by its best conformer and named
        once. The returned gid is the winning conformer's, so the pose is still
        recoverable. This is the level every enrichment statistic is defined at.

    ``trust_definitions`` are sha256 hashes (or prefixes of at least 16 hex
    characters) of custom color definitions to trust for this call, as
    ``roshambo3 screen --trust-definition`` (``roshambo3.definitions``). Accepted
    and checked before query featurization or screening.

    The default is ``"conformer"`` because that is what the device top-K
    retains and what every number recorded before this existed was measured on.
    It is not the recommended level for a user asking for "the best molecules".

    ``launch_conformers`` is how many conformers go into one ``score_pairs``
    call. ``None`` means
    :func:`roshambo3.engine.conformers_per_launch`, which takes the default
    and bounds it by what the device top-K's state will cost at *this* ``k``
    and query count -- the state is ``n_queries * (k + 2 * launch) * 20``
    floats, so it grows with the panel as well as with the launch size, and its
    capacity is fixed on the first launch. This is the only place that knows
    both numbers, which is why the bound is applied here and not in the source.

    For several GPUs, run one :class:`Engine` per device over disjoint
    ``groups`` and merge with :meth:`roshambo3.engine.TopK.merge`.

    ``backend='cpu'`` selects CPU scoring. Supported color FIELD/IN_GRADIENT
    shape, fixed-pose color and combo requests use the fast worker route, with
    ``workers`` unless explicit ``config.cpu_worker`` options supply worker
    count/budgets. Omitted ``workers`` uses CPU affinity (including SMT), or
    the system CPU count when affinity is unavailable; GPU decoding retains
    its eight-worker default. Public worker memory budgets default to no cap;
    explicit budgets are optional preflight safeguards, not allocation sizes.
    Worker options do
    not turn shape/rescore objectives into combo. They default to an automatic
    mode-aware policy: combo uses
    the exact-algebra canonical evaluator and torque factoring plus the faster
    unclamped raw-bits optimizer exponential when safe (guarded otherwise),
    while shape/rescore retain the
    baseline exponential and disable combo-only controls. Set
    ``CPUWorkerOptions(optimizer_exp_mode="baseline")``
    to opt out of approximate exp without disabling the exact canonical
    optimizations.
    Explicit option values remain authoritative; native eligibility and
    general-matrix fallback rules still apply. Unsupported worker configurations
    (including non-FIELD requests) retain their ordinary route unless worker
    options were explicitly requested.
    Ordinary CPU requests use host conformer top-K; ordinary molecule top-K
    is unsupported (the fast worker route supports both ranking levels).
    Pass a caller-owned ``engine=Engine(backend='cpu')`` to retain immutable
    query fields across supported worker screens (automatic combo or explicit
    shape/rescore); ordinary screens do not retain fields. Call
    ``engine.close()`` to release retained state.
    Prepared QuerySet color numbering must be declared by its generator.
    Configuration objects are never mutated.
    """
    if level not in ("conformer", "molecule"):
        raise ValueError(
            f"level must be 'conformer' or 'molecule', got {level!r}")
    from .backend import select
    selected_backend = engine.backend if engine is not None else select(backend)
    corp = open_corpus(corpus)
    cfg = config or ScreenConfig()
    if cfg.color:
        color_generator = _resolve_color_generator(
            queries, corp, color_generator, trust_definitions)
    # Check before featurizing query molecules. In particular, no custom
    # corpus definition reaches RDKit until the trust gate has run.
    color_checked = False
    if cfg.color and (color_generator is not None
                      or isinstance(getattr(corp, "descriptor", None), dict)):
        _check_color(cfg, corp, color_generator, trusted_definitions=trust_definitions)
        color_checked = True
    qs = load_queries(queries, color_generator=color_generator, backend=selected_backend)
    owned_engine = engine is None
    if engine is None:
        engine = Engine(device_id=device_id, stage_timing=stage_timing, backend=backend)
    else:
        if not isinstance(engine, Engine):
            raise TypeError("engine must be an Engine")
        if backend is not None:
            from .backend import select
            if select(backend) is not engine.backend:
                raise ValueError("backend conflicts with the supplied Engine")
        if device_id != 0 and int(device_id) != engine.device_id:
            raise ValueError("device_id conflicts with the supplied Engine")
        if stage_timing and not engine.stage_timing:
            raise ValueError("stage_timing conflicts with the supplied Engine")
    is_cpu = getattr(engine.backend, "__name__", "") == "roshambo3.backend.cpu"
    if workers is None:
        if is_cpu:
            from .engine.config import _default_cpu_workers
            workers = _default_cpu_workers()
        else:
            workers = 8
    if cfg.cpu_worker is not None and type(cfg.cpu_worker) is not CPUWorkerOptions:
        raise TypeError("cpu_worker must be a CPUWorkerOptions")
    if cfg.cpu_worker is not None and cfg.cpu_worker.workers is None:
        cfg = replace(cfg, cpu_worker=replace(cfg.cpu_worker, workers=workers))
    from .engine.engine import _cpu_worker_config_compatible
    if (is_cpu and cfg.cpu_worker is None and not engine.stage_timing
            and engine.device_id == 0
            and _cpu_worker_config_compatible(cfg)):
        cfg = replace(cfg, cpu_worker=CPUWorkerOptions(workers=workers))
    if not color_checked:
        _check_color(cfg, corp, color_generator,
                     trusted_definitions=trust_definitions)
    per_launch = (conformers_per_launch(qs.n_conformers, int(k))
                  if launch_conformers is None else int(launch_conformers))
    launch_policy = getattr(engine.backend, "launch_conformers", None)
    if launch_policy is not None and launch_conformers is None:
        widths = [int(g["L_field_width"]) for g in corp.group_table]
        per_launch = launch_policy(qs.n_conformers, int(k), max(widths),
                                   device_id=engine.device_id)
    source = make_source(corp, groups=groups, workers=workers,
                        prefetch_depth=prefetch_depth,
                        conformers_per_launch=per_launch,
                        color=bool(cfg.color), tile=not (is_cpu or cfg.cpu_worker is not None),
                        color_generator=color_generator,
                        trust_definitions=trust_definitions)
    if is_cpu and cfg.cpu_worker is None:
        if level == "molecule":
            raise NotImplementedError("ordinary CPU molecule screening is unsupported; CPU combo workers support molecule top-K")
        sink = HostTopK(k=int(k))
    else:
        sink = engine.make_topk(k=int(k), level=level)
    try:
        stats, per_query = engine.run(source, qs, cfg, sink)
    finally:
        if owned_engine:
            engine.close()

    # The source already read every group's run table; reusing them
    # saves reopening each group to resolve the hits' molecule ids.
    # Only the groups the hits touch: a ScreenResult keeps its Catalog, and
    # every group's table would be ~864 MB on a 540M-conformer library.
    run_tables = getattr(source, "run_tables", None)
    touched = None
    if callable(run_tables):
        offsets = np.cumsum([int(g["n_conf"]) for g in corp.group_table])
        hit_gids = [np.asarray(ids, dtype=np.int64) for _, ids in per_query]
        hit_gids = np.concatenate(hit_gids) if hit_gids else np.empty(0, np.int64)
        groups = set(np.searchsorted(offsets, hit_gids, side="right").tolist())
        touched = {gi: t for gi, t in run_tables().items() if gi in groups}
    catalog = Catalog(corp, run_tables=touched)
    hits = [QueryHits(query_name=qs.names[i], query_index=i,
                      gids=np.asarray(ids), scores=np.asarray(sc),
                      rank_col=sink.rank_col, level=level,
                      mol_id=catalog.mol_ids(ids), conf_idx=catalog.conf_indices(ids))
            for i, (sc, ids) in enumerate(per_query)]
    return ScreenResult(hits=hits, corpus=corp, stats=stats, _catalog=catalog)


def score_matrix(queries: Union[PathLike, QuerySet, Sequence],
                 corpus: Union[PathLike, Corpus],
                 *,
                 config: Optional[ScreenConfig] = None,
                 device_id: int = 0,
                 groups: Optional[Sequence[int]] = None,
                 max_bytes: int = 4 << 30,
                 color_generator=None, backend=None,
                 trust_definitions: Sequence[str] = ()):
    """Every pair's full score row: ``[n_queries, n_conformers, 20]``.

    The all-pairs mode. It copies the whole dense block back, which is exactly
    what the device top-K exists to avoid, so it is size-guarded and is for a
    corpus small enough that the matrix itself is the deliverable -- not for a
    screen. A row is 80 B, so 8 queries against 1e6 conformers is 6.4 GB.

    ``trust_definitions``: as in :func:`screen`.
    """
    from .engine import DenseScores

    corp = open_corpus(corpus)
    cfg = config or ScreenConfig()
    if cfg.color:
        color_generator = _resolve_color_generator(
            queries, corp, color_generator, trust_definitions)
    _check_color(cfg, corp, color_generator, trusted_definitions=trust_definitions)
    qs = load_queries(queries, color_generator=color_generator, backend=backend)
    source = make_source(corp, groups=groups, color=bool(cfg.color),
                         color_generator=color_generator, trust_definitions=trust_definitions)
    engine = Engine(device_id=device_id, backend=backend)
    try:
        _, dense = engine.run(source, qs, cfg, DenseScores(max_bytes=max_bytes))
        return dense
    finally:
        engine.close()

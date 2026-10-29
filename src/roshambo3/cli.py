"""The ``roshambo3`` command line.

A thin wrapper over the library, which is the primary product. Anything the CLI
can do is reachable in three lines of Python, on purpose -- a screen at
1e9-conformer scale is something people script, and a tool that can only be
driven from a shell is a tool that has to be re-implemented to be automated.

    roshambo3 info    <corpus>
    roshambo3 screen  <queries.sdf> <corpus> [-k N] [-o hits.csv]
    roshambo3 prepare <library.sdf> <corpus>
    roshambo3 verify  <corpus> [--tier open|scan|deep] [--all] [--workers N]
    roshambo3 definitions show|trust|untrust|list ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

def _cmd_info_format3(args) -> int:
    """Summarize a format-3 corpus from its descriptor and group headers
    (the ``open`` tier only: nothing beyond them is read)."""
    import os
    from .format.r3g.reader import open_corpus

    c = open_corpus(args.corpus, purpose="inspect")
    d = c.descriptor
    from .format.r3g.limits import group_file_name
    group_bytes = sum(os.path.getsize(os.path.join(args.corpus, "groups",
                                                   group_file_name(e["bucket"], e["group"])))
                      for e in c.group_table)
    total = (group_bytes + os.path.getsize(os.path.join(args.corpus, "molecules.parquet"))
             + os.path.getsize(os.path.join(args.corpus, "corpus.json")))
    print(f"{args.corpus}")
    print(f"  conformers      {c.n_conf:,}")
    print(f"  molecules       {c.n_mol:,}")
    print(f"  groups          {len(c.group_table)}")
    print(f"  bytes/conformer {total / max(1, c.n_conf):.1f}")
    print(f"  ids_sha256      {d['ids_sha256']}")
    print(f"  fingerprint     {c.fingerprint}")
    fmt = d["format"]
    color = d.get("color")
    print(f"  format          r3g v{fmt['major']} encoding={fmt['encoding']} "
          f"step=2^{fmt['step_log2']} A "
          f"color={','.join(color['types']) if color else 'none'}")
    print(f"  scanned         {'yes' if c.scanned else 'no (roshambo3 verify --tier scan)'}")
    if args.buckets:
        print("\n  bucket  L_field  L_min  L_max  conformers  molecules  groups")
        by_bucket: dict = {}
        for e in c.group_table:
            by_bucket.setdefault(e["bucket"], []).append(e)
        for b, es in sorted(by_bucket.items()):
            print(f"  {b:>6}  {es[0]['L_field_width']:>7}  {min(e['L'] for e in es):>5}  "
                  f"{max(e['L'] for e in es):>5}  {sum(e['n_conf'] for e in es):>10,}  "
                  f"{sum(e['n_mol'] for e in es):>9,}  {len(es):>6}")
    return 0


def _cmd_info(args) -> int:
    return _cmd_info_format3(args)


def _cmd_screen(args) -> int:
    from . import screen
    from .engine import ScreenConfig
    from .backend.spec import ColorPlacement, PoseSearch

    from .backend import select
    selected = select(args.backend)
    is_metal = selected.__name__ == "roshambo3.backend.metal"
    analytic = args.analytic
    if args.color and args.mode is None and not is_metal:
        # The field is the query's shape field; there is no color field. Route
        # the request to the path that can serve it rather than failing in the
        # kernel. Scoped to `screen`: `prepare` has a --color and no --analytic.
        analytic = True
    color = args.color if args.mode is None else args.mode != "shape"
    if args.mode is not None and (args.color or args.analytic):
        raise ValueError("--mode cannot be combined with legacy --color or --analytic")
    mixing = args.mixing if args.mixing is not None else (.5 if args.mode in ("rescore", "combo") else 0.)
    config = ScreenConfig(
        color_placement=ColorPlacement.IN_GRADIENT if args.mode == "combo" else ColorPlacement.AT_POSE,
        pose_search=PoseSearch.ANALYTIC if analytic else PoseSearch.FIELD,
        color=color,
        mixing=mixing,
        steps=args.steps,
    )
    # Format 3 owns the complete color descriptor; API.screen resolves its
    # shipped generator or trusted isolated adapter before query featurization.
    generator = None

    res = screen(args.queries, args.corpus, k=args.k, config=config,
                 device_id=args.device, workers=args.workers, backend=args.backend,
                 prefetch_depth=args.prefetch,
                 launch_conformers=args.launch_conformers,
                 stage_timing=args.timing,
                 color_generator=generator, level=args.level,
                 trust_definitions=args.trust_definition)

    s = res.stats
    print(f"screened {s.n_conformers:,} conformers x {len(res)} queries "
          f"in {s.wall_s:.2f}s ({s.conformers_per_s:,.0f} conformers/s)",
          file=sys.stderr)
    print(f"  {args.k:,} best {args.level}s per query", file=sys.stderr)
    if args.level == "conformer":
        # Said out loud, because "-k 100" reads as "100 molecules" and is not.
        n = min(len(res[0]), 1000)
        distinct = res.distinct_molecules_in_hits(n=n)["n_molecules"]
        print(f"  the first {n:,} of them cover {distinct:,} distinct "
              f"molecules; --level molecule for the top {args.k:,} molecules",
              file=sys.stderr)
    if args.timing:
            # launch_s times ENQUEUE, not execution: with a device sink
        # score_pairs returns without synchronizing. It is printed because it
        # is a real host cost, and labeled because a 1.6e9 pairs/s figure
        # was once quoted from a number measured this way.
        print(f"  producer_wait {s.producer_wait_s:.3f}s  "
              f"enqueue {s.launch_s:.3f}s (not kernel time)  "
              f"{s.n_blocks:,} launches of "
              f"{s.n_conformers / max(1, s.n_blocks):,.0f} conformers",
              file=sys.stderr)
        print("  device stages: " + json.dumps(
            {k: round(v, 4) for k, v in s.device_stages.items()
             if isinstance(v, float)}), file=sys.stderr)

    if args.out:
        _write_csv(res, args.out, with_names=args.names)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        from .results import spreadsheet_safe_text
        for h in res:
            print(f"# {h.query_name}")
            names = (res.catalog.molecules(h.gids, columns=["ext_id", "smiles"])
                     if args.names else None)
            for i in range(min(len(h), args.head)):
                line = (f"  gid={h.gids[i]}  mol_id={h.mol_id[i]}  "
                        f"conf_idx={h.conf_idx[i]}  shape={h.tanimoto_shape[i]:.6f}")
                if names is not None:
                    line += (f"  ext_id={spreadsheet_safe_text(names['ext_id'][i])}"
                             f"  smiles={spreadsheet_safe_text(names['smiles'][i])}")
                print(line)
    return 0


def _write_csv(res, path: str, with_names: bool = False) -> None:
    import csv
    from .results import spreadsheet_safe_text

    table = res.to_table()
    cols = list(table)
    extra: dict = {}
    if with_names:
        # One catalog lookup per query, not per hit: the join reads whole parquet
        # row groups, so asking for one molecule costs about what asking for all
        # of them does.
        ext, smi = [], []
        for h in res:
            m = res.catalog.molecules(h.gids, columns=["ext_id", "smiles"])
            ext.extend(m["ext_id"])
            smi.extend(m["smiles"])
        extra = {"ext_id": [spreadsheet_safe_text(v) for v in ext],
                 "smiles": [spreadsheet_safe_text(v) for v in smi]}
        cols = cols[:2] + list(extra) + cols[2:]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        n = len(table["gid"])
        for i in range(n):
            row = []
            for c in cols:
                value = extra[c][i] if c in extra else table[c][i]
                row.append(spreadsheet_safe_text(value))
            w.writerow(row)


def _cmd_prepare(args) -> int:
    from .prep import prepare

    generator = None
    if args.color:
        from .pharmacophore import PharmacophoreGenerator
        generator = PharmacophoreGenerator()

    reported = [0]

    def progress(n_mol, n_conf):
        # One line per completed work unit, to stderr: a build over a large
        # library is a job someone watches, and the rule is to wait on the
        # progress signal rather than the terminal artifact.
        if n_mol - reported[0] >= 10_000 or args.verbose:
            reported[0] = n_mol
            print(f"  {n_mol:,} molecules, {n_conf:,} conformers",
                  file=sys.stderr)

    sqlite_source = None
    if (args.input_format.startswith("sqlite") or args.sample_seed is not None
            or args.sample_p_lo != 0.0 or args.sample_p_hi != 1.0
            or args.hac_window or args.source_rows is not None
            or (args.input_format == "auto" and len(args.inputs) == 1
                and str(args.inputs[0]).lower().endswith((".sqlite", ".sqlite3", ".db")))):
        sqlite_source = dict(p_lo=args.sample_p_lo, p_hi=args.sample_p_hi,
                             hac_lo=args.hac_window[0] if args.hac_window else None,
                             hac_hi=args.hac_window[1] if args.hac_window else None,
                             n_rows=args.source_rows)
        if args.sample_seed is not None:
            sqlite_source["seed"] = args.sample_seed

    c = prepare(args.inputs, args.out, name=args.name, workers=args.workers,
                records_per_unit=args.records_per_unit,
                block_rows=args.block_rows,
                color_generator=generator, note=args.note,
                input_format=args.input_format, max_records=args.max_records,
                temp_dir=args.temp_dir,
                color_self=not args.no_color_self,
                resume=args.resume, dry_run=args.dry_run,
                expect_sha256_ids=args.expect_ids_sha256,
                expect_ids_from=args.expect_ids_from,
                sqlite_source=sqlite_source,
                verbose_rejects=args.verbose_rejects,
                progress=progress)
    if args.dry_run:
        print(json.dumps(c, indent=1))
        return 0
    with open(Path(args.out) / "build.json", "rb") as f:
        build = json.load(f)
    stats, timing = build["statistics"], build["timing"]
    print(f"format 3 corpus: {c.n_mol:,} molecules, {c.n_conf:,} conformers, "
          f"{len(c.group_table)} groups", file=sys.stderr)
    print(f"  fingerprint {c.fingerprint}", file=sys.stderr)
    print(f"  ids_sha256  {c.descriptor['ids_sha256']}", file=sys.stderr)
    print(f"  {stats['bytes']['per_conformer']:.1f} bytes/conformer; "
          f"block_rows {build['parameters']['block_rows']}", file=sys.stderr)
    print(f"  scatter {timing['scatter_s']:.1f}s, gather {timing['gather_s']:.1f}s, "
          f"scan {timing['validate_s']:.1f}s", file=sys.stderr)
    print(args.out)
    return 0


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {v}")
    return v


def _cmd_verify_format3(args) -> int:
    """Validate a format-3 corpus (doc/FORMAT.md section 16) at one tier; a
    passing scan or deep is cached, which is what lets the corpus screen."""
    from .format.r3g.errors import CorpusValidationError
    from .format.r3g.reader import verify

    tier = args.tier or "scan"
    try:
        r = verify(args.corpus, tier, all_rows=args.all, workers=args.workers)
    except CorpusValidationError as e:
        print(f"REFUSED  tier={e.tier}  code={e.code}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        return 1
    how = "cached scan; open re-run" if r.cached else f"decoder {r.backend}"
    print(f"format 3  tier={tier}  OK  ({r.seconds:.2f} s, {how})")
    print(f"  fingerprint {r.fingerprint}")
    print(f"  groups {len(r.groups)}  molecules {r.descriptor['n_mol']:,}  "
          f"conformers {r.descriptor['n_conf']:,}")
    # FORMAT.md 10.6: deep recomputes it from molecules.parquet (and refuses a
    # mismatch); scan checks the parquet's mol_id set against the run tables.
    print(f"  ids_sha256 {r.descriptor['ids_sha256']}"
          f"{'  (recomputed from molecules.parquet: match)' if tier == 'deep' else ''}")
    if tier == "deep":
        print(f"  re-perceived molecules {r.sampled_molecules:,}"
              f"{' (all)' if args.all else ' (sample)'}")
        if r.backend == "native":
            k = ("every block, --all" if r.reference_sample is None
                 else f"a sample of K = {r.reference_sample} per corpus")
            print(f"  reference cross-decode {r.reference_blocks:,} blocks ({k})")
        print(f"  sandbox workers {r.workers}")
        print("  phases " + ", ".join(f"{k} {v:.1f} s" for k, v in r.phase_seconds.items()))
    if r.extra_files:
        # section 1 item 3: reported, never read
        print(f"  ignored files: {', '.join(repr(n) for n in r.extra_files)}")
    return 0



def _cmd_verify(args) -> int:
    return _cmd_verify_format3(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roshambo3",
        description="Ultra-scale 3D molecular shape screening on GPUs.")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    sub = p.add_subparsers(dest="command")

    web = sub.add_parser("web", help="launch the optional screening web application")
    web.add_argument("--config", required=True, help="database/server TOML configuration")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.set_defaults(func=_cmd_web)

    i = sub.add_parser("info", help="summarize a prepared corpus")
    i.add_argument("corpus")
    i.add_argument("--buckets", action="store_true", help="per-bucket breakdown")
    i.set_defaults(func=_cmd_info)

    s = sub.add_parser("screen", help="screen queries against a corpus")
    s.add_argument("queries", help="SDF with 3D coordinates")
    s.add_argument("corpus")
    s.add_argument("--backend", choices=["cuda", "metal", "cpu"], default=None)
    s.add_argument("--mode", choices=["shape", "rescore", "combo"], default=None)
    s.add_argument("-k", type=int, default=1000, help="hits per query (default 1000)")
    s.add_argument("--level", choices=["conformer", "molecule"],
                   default="conformer",
                   help="what one hit is. 'conformer' (the default) "
                        "returns the k best conformers, so one molecule can "
                        "occupy several slots. 'molecule' returns the k best "
                        "molecules, each scored by its best conformer -- the "
                        "level every enrichment statistic is defined at.")
    s.add_argument("-o", "--out", help="write hits to this CSV")
    s.add_argument("--names", action="store_true",
                   help="include ext_id and SMILES (one catalog read per query)")
    s.add_argument("--head", type=int, default=10,
                   help="rows to print per query when not writing a file")
    s.add_argument("--analytic", action="store_true",
                   help="use the exact analytic optimizer instead of the field")
    s.add_argument("--color", action="store_true",
                   help="score color as well as shape (implies --analytic). "
                        "Needs a corpus built with --color (roshambo3 prepare "
                        "--color).")
    s.add_argument("--mixing", type=float, default=None,
                   help="combination weight: 0 (default) is pure shape, 1 is "
                        "pure color, in between mixes the two. At the default "
                        "the combination Tanimoto is exactly the shape "
                        "Tanimoto and --color is a reported column, nothing "
                        "more.")
    s.add_argument("--steps", type=int, default=25,
                   help="optimizer steps; lr_t is derived from it")
    s.add_argument("--device", type=int, default=0)
    s.add_argument("--workers", type=int, default=8)
    from .engine import DEFAULT_PREFETCH_DEPTH
    s.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH_DEPTH)
    s.add_argument("--launch-conformers", type=int, default=None,
                   help="conformers per score_pairs launch. Default: "
                        "65,536, bounded by what the device top-K state costs "
                        "at this -k and query count. 4096 is the older "
                        "behavior, one launch per on-disk block.")
    s.add_argument("--timing", action="store_true", help="report the stage breakdown")
    s.add_argument("--trust-definition", action="append", default=[], metavar="SHA256",
                   help="trust a custom color definition for this run only, by its sha256 "
                        "(a prefix of at least 16 hex characters is accepted); repeatable. "
                        "See `roshambo3 definitions`.")
    s.set_defaults(func=_cmd_screen)

    pr = sub.add_parser("prepare", help="build a format-3 corpus from 3D SDF, RDKit "
                                        "binary-zstd or SQLite input")
    pr.add_argument("inputs", nargs="+", help="SDF files or one .bin.zst RDKit stream")
    pr.add_argument("-o", "--out", required=True, help="corpus directory to create")
    pr.add_argument("--name", help="corpus name (default: the first input's stem)")
    pr.add_argument("--color", action="store_true",
                    help="perceive RDKit BaseFeatures pharmacophore features and "
                         "store each molecule's feature table (FORMAT.md section "
                         "8); every conformer's pseudo-atoms are derived from its "
                         "stored heavy atoms. Every conformer gets color, never "
                         "only survivors.")
    pr.add_argument("--records-per-unit", type=int, default=20_000)
    pr.add_argument("--block-rows", type=int, default=4096, metavar="N",
                    help="cut coordinate blocks at the last molecule boundary at "
                         "or before N rows, 1 <= N <= 4096 (FORMAT.md section 6); "
                         "a larger molecule gets a block of its own. Changes no "
                         "score; recorded in build.json (default 4096)")
    pr.add_argument("--input-format", default="auto",
                    choices=["auto", "sdf", "rdkit-binary-zstd", "sqlite",
                             "sqlite:pubchem", "sqlite:text-id"],
                    help="input reader; auto recognizes a single .bin.zst stream "
                         "or a .sqlite/.sqlite3/.db database")
    pr.add_argument("--sample-seed", type=int, default=None,
                    help="SQLite input: hash seed (default: the D-tier seed)")
    pr.add_argument("--sample-p-lo", type=float, default=0.0,
                    help="SQLite input: keep molecules whose hash is >= this")
    pr.add_argument("--sample-p-hi", type=float, default=1.0,
                    help="SQLite input: keep molecules whose hash is < this")
    pr.add_argument("--hac-window", type=int, nargs=2, metavar=("LO", "HI"),
                    help="SQLite input: also require LO <= heavy atoms <= HI")
    pr.add_argument("--source-rows", type=int, default=None,
                    help="SQLite input: the table's row count, to skip counting "
                         "it (a full walk of a large file on spinning disk)")
    pr.add_argument("--max-records", type=int,
                    help="build only this deterministic prefix of a binary stream")
    pr.add_argument("--temp-dir",
                    help="scratch root for raw prepare shards; defaults beside output")
    pr.add_argument("--workers", type=int, default=0,
                    help="0 = min(8, cpus), or all CPUs for binary-zstd input")
    pr.add_argument("--no-color-self", action="store_true",
                    help="with --color, do not write the color_self section "
                         "(written by default, for canonical interaction "
                         "matrices only; FORMAT.md section 18.2)")
    pr.add_argument("--resume", action="store_true",
                    help="continue the interrupted build left in OUT.partial; "
                         "inputs and options must be identical")
    pr.add_argument("--dry-run", action="store_true",
                    help="print the record count, estimated CPU time, bytes "
                         "to read/write and free disk space as JSON; write "
                         "nothing")
    pr.add_argument("--expect-ids-sha256", default=None, metavar="HEX",
                    help="fail, without publishing, unless the built "
                         "corpus's ids_sha256 (the sorted mol_id list as "
                         "little-endian int64; equal to the corpus descriptor hash) "
                         "equals this")
    pr.add_argument("--expect-ids-from", default=None, metavar="CORPUS_DIR",
                    help="check every work unit, before it is committed, "
                         "against the molecule IDs in this format-3 corpus's "
                         "molecules.parquet (mol_id == stable_id(ext_id), "
                         "every ID present), so a wrong "
                         "source stops at its first unit; the finished "
                         "build's ids_sha256 must also equal the reference's "
                         "(skipped for a --max-records prefix)")
    pr.add_argument("--verbose-rejects", action="store_true",
                    help="also print the first few rejected molecules' ids and "
                         "reasons (SMILES included) and dropped conformers' ids; "
                         "by default only counts by reason code are printed and "
                         "the ids stay in build.json, so the log of a build that "
                         "completes names no molecule. A refused build names the "
                         "molecule (and, for a variant conflict, its SMILES): "
                         "inspect its log before committing it. For public corpora")
    pr.add_argument("--note", default="", help="free text recorded in build.json")
    pr.add_argument("-v", "--verbose", action="store_true")
    pr.set_defaults(func=_cmd_prepare)

    v = sub.add_parser("verify", help="validate a format-3 corpus")
    v.add_argument("corpus")
    v.add_argument("--tier", choices=("open", "scan", "deep"), default=None,
                   help="format 3: validator tier (default scan; deep decodes every block and "
                        "re-perceives a sample of molecules)")
    v.add_argument("--all", action="store_true",
                   help="format 3, --tier deep: re-perceive every molecule, not a sample")
    v.add_argument("--workers", type=_positive_int, default=None, metavar="N",
                   help="--tier deep: RDKit sandbox children run at once (default "
                        "min(16, CPU count)); the checks and any refusal do not depend on it")
    v.set_defaults(func=_cmd_verify)

    d = sub.add_parser("definitions", help="inspect and trust the custom color definitions "
                       "stored in corpora (never imports RDKit)")
    dsub = d.add_subparsers(dest="definitions_command", required=True)
    ds = dsub.add_parser("show", help="print a corpus's stored color definition")
    ds.add_argument("corpus")
    ds.add_argument("--diff", action="store_true",
                    help="unified diff against the nearest shipped RDKit BaseFeatures.fdef")
    dt = dsub.add_parser("trust", help="trust one exact definition from now on")
    dt.add_argument("hash", help="the full 64-hex sha256, or a prefix of at least 16 hex "
                    "characters together with --corpus")
    dt.add_argument("--corpus", help="resolve a prefix against this corpus's definition hash")
    dt.add_argument("--note", default="", help="free text stored with the entry")
    du = dsub.add_parser("untrust", help="stop trusting a definition")
    du.add_argument("hash", help="the full hash or an unambiguous prefix (>= 16 hex) of an entry")
    dsub.add_parser("list", help="list the trusted definitions")
    d.set_defaults(func=_cmd_definitions)

    return p


def _cmd_definitions(args) -> int:
    from . import definitions as defs

    cmd = args.definitions_command
    if cmd == "show":
        sys.stdout.write(defs.show(args.corpus, diff=args.diff))
        if not args.diff:
            sys.stdout.write("\n")
        return 0
    if cmd == "trust":
        color = None
        if args.corpus:
            from .format.r3g.validate import read_descriptor
            color = read_descriptor(args.corpus).data.get("color")
            if color is None:
                raise ValueError(f"{args.corpus} is shape-only: it has no definition to trust")
        h = defs.resolve_prefix(args.hash, color)
        if defs.is_shipped(h):
            print(f"{h} is a shipped RDKit default; it needs no trust entry")
            return 0
        added = defs.trust(h, note=args.note)
        print(f"{'trusted' if added else 'already trusted'} {h}  ({defs.trust_file_path()})")
        return 0
    if cmd == "untrust":
        print(f"untrusted {defs.untrust(args.hash)}")
        return 0
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entries = defs.list_trusted()
    for w in caught:
        print(f"roshambo3: {w.message}", file=sys.stderr)
    for h, (date, note) in sorted(entries.items(), key=lambda kv: kv[1][0]):
        print(f"{h}\t{date}\t{note}")
    return 0


def _cmd_web(args):
    try:
        from .web.app import serve
    except ImportError as exc:
        raise RuntimeError("Install web dependencies with pip install 'roshambo3[web]'") from exc
    return serve(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__
        print(__version__)
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    from .definitions import UntrustedDefinitionError
    try:
        return args.func(args)
    except UntrustedDefinitionError as e:
        print(str(e), file=sys.stderr)        # already the documented "error: ..." text
        return 1
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # These are the library saying "this cannot mean what you asked for" --
        # a missing corpus, color against a shape-only corpus, an off-budget
        # (steps, lr_t) pair. A traceback here tells a user nothing they can act
        # on; the message does, and the library's messages are written to.
        print(f"roshambo3: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fresh Python-process measurements, with API and process times kept separate.

Process wall includes imports, provenance collection, query loading, screening,
and result serialization. API wall comes from the child's screen() call.
Filesystem caches are not flushed. These cold-process diagnostics are separate
from the warmed performance matrix and retain the child's grading receipts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from bench.apple_silicon import PANEL_KEYS, key_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--qos", default="inherit")
    parser.add_argument("--modes", default="combo,shape,rescore")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.repeats < 1:
        parser.error("workers/repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    for mode in args.modes.split(","):
        for repeat in range(args.repeats):
            for mol_id, conf_idx in PANEL_KEYS:
                identity = (mol_id, conf_idx)
                child = args.out / f"{mode}-r{repeat}-{key_label(identity)}"
                if (child / "process-timing.json").exists():
                    continue
                command = [sys.executable, "-m", "bench.apple_silicon", "--corpus", args.corpus,
                    "--queries", args.queries, "--out", str(child), "--workers", str(args.workers),
                    "--qos", args.qos, "--modes", mode,
                    "--panel-keys", f"{mol_id}:{conf_idx}", "--fresh",
                    "--warmup", "0", "--repeats", "1", "--provisional"]
                started = time.perf_counter()
                result = subprocess.run(command, capture_output=True, text=True)
                seconds = time.perf_counter() - started
                child.mkdir(parents=True, exist_ok=True)
                (child / "launch.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"fresh process failed; see {child / 'launch.log'}")
                api = json.loads((child / "timings.jsonl").read_text().splitlines()[0])
                record = {"mode": mode, "repeat": repeat,
                          "mol_id": mol_id, "conf_idx": conf_idx,
                          "workers": args.workers, "qos": args.qos,
                          "process_wall_s": seconds, "api_wall_s": api["wall_s"],
                          "provisional": True, "child_grade": api["grade"]}
                (child / "process-timing.json").write_text(json.dumps(record, indent=2))
                with (args.out / "process-timings.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

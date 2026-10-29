#!/usr/bin/env python3
"""Explicit developer audit of historical CPU evidence; no fabricated waivers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.cpu_run import ungraded_timed_artifacts


def audit(results):
    return {d.name: missing for d in sorted(results.glob("T*"))
            if (missing := ungraded_timed_artifacts(d))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    args = p.parse_args()
    offenders = audit(args.results)
    print(json.dumps({"status": "fail" if offenders else "pass", "offenders": offenders}, indent=2))
    return bool(offenders)


if __name__ == "__main__":
    sys.exit(main())

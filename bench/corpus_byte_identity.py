"""Compare two format-3 corpus directories byte for byte.

Every file must be identical except ``build.json`` (host, paths and timings)
and ``corpus.json``, which is compared as parsed JSON with ``prep.commit``
removed (the builder's git commit, provenance only: FORMAT.md section 13.3).
Prints one JSON line and exits 0 when the two are identical on that basis.

    python bench/corpus_byte_identity.py <corpus_a> <corpus_b>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys


def _files(root: str) -> dict:
    out = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            path = os.path.join(dirpath, n)
            rel = os.path.relpath(path, root)
            if rel in ("build.json", "corpus.json"):
                continue
            with open(path, "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def _descriptor(root: str) -> dict:
    with open(os.path.join(root, "corpus.json")) as f:
        d = json.load(f)
    d.get("prep", {}).pop("commit", None)
    return d


def main(a: str, b: str) -> int:
    fa, fb = _files(a), _files(b)
    differ = sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k))
    same_descriptor = _descriptor(a) == _descriptor(b)
    record = {"a": os.path.abspath(a), "b": os.path.abspath(b),
              "files_compared": len(fa.keys() | fb.keys()),
              "files_differing": differ,
              "corpus_json_equal_without_prep_commit": same_descriptor,
              "identical": not differ and same_descriptor}
    print(json.dumps(record))
    return 0 if record["identical"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))

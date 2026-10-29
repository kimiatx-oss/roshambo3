#!/usr/bin/env python
"""Print the allow-list entry (and the text to ship) for the installed RDKit's
``BaseFeatures.fdef`` (``roshambo3.definitions``).

Reads ``RDConfig.RDDataDir/BaseFeatures.fdef``, hashes it exactly as the corpus
format does (``reference.definition_hash``: SHA-256 of the LF-normalized text),
and prints

* the ``allowlist.tsv`` line to add (``<sha256> TAB <rdkit version> TAB <date>``),
* the ``PROVENANCE`` text for ``src/roshambo3/data/definitions/<version>/``.

With ``--write`` it also writes ``BaseFeatures.fdef`` and ``PROVENANCE`` into that
directory and appends the line to ``allowlist.tsv`` when its hash is not listed
yet. Nothing is overwritten.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src",
                    "roshambo3", "data", "definitions")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="add the files to the package data")
    args = ap.parse_args(argv)
    import rdkit
    from rdkit import RDConfig

    from roshambo3.format.r3g.reference import definition_hash

    src = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    with open(src, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    h = definition_hash(text)
    today = datetime.date.today().isoformat()
    entry = f"{h}\t{rdkit.__version__}\t{today}"
    prov = (f"RDKit {rdkit.__version__}, BSD-3-Clause (https://github.com/rdkit/rdkit)\n"
            f"source  RDConfig.RDDataDir/BaseFeatures.fdef of the RDKit release above\n"
            f"bytes   {len(raw)}\n"
            f"sha256  {h}\n"
            f"The text is shipped verbatim. Its hash is the corpus-format definition hash\n"
            f"(doc/FORMAT.md section 10.3); allowlist.tsv lists it. Regenerate with\n"
            f"scripts/definition_allowlist.py --write.\n")
    print("allowlist.tsv entry:\n" + entry + "\n\nPROVENANCE:\n" + prov)
    if not args.write:
        return 0
    d = os.path.join(DATA, rdkit.__version__)
    os.makedirs(d, exist_ok=True)
    for name, content in (("BaseFeatures.fdef", raw), ("PROVENANCE", prov.encode())):
        p = os.path.join(d, name)
        if os.path.exists(p):
            print(f"exists, not overwritten: {p}", file=sys.stderr)
        else:
            with open(p, "wb") as f:
                f.write(content)
    al = os.path.join(DATA, "allowlist.tsv")
    have = open(al).read() if os.path.exists(al) else ""
    if h not in have:
        with open(al, "a") as f:
            if not have:
                f.write("# sha256\trdkit_version\tdate_added\n")
            f.write(entry + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

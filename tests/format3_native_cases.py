"""Vector cases for the native format-3 codec's standalone driver.

Turns the format-3 test vectors -- every block of the conformance corpus
(``tests/data/format3/{shape,color}``), the per-width oracle blocks
(``widths/``) and every block of every adversarial seed (committed, plus the
generated ones materialized by ``make_adversarial.build``) -- into ``.case``
files that ``tests/r3g_codec.cpp --cases DIR`` runs through every compiled ISA
and every writer, then checks the files it writes back:

* every ISA path produced identical bytes for every output;
* the conformance blocks reassemble to ``EXPECTED.json``'s ``heavy``,
  ``merged``, ``pseudo`` and ``counts`` digests; the width blocks to
  ``widths/EXPECTED.json``'s;
* the GPU and CPU outputs are the merged record in tile order / as float32;
* every adversarial block either decoded or was refused -- the driver
  returning at all (under ASan/UBSan, or under qemu for NEON) is the test;
  the refusal codes are reported.

Adversarial group files are parsed TOLERANTLY here (no validator in front, as
the fuzzer does): a header that does not parse yields no cases, every
section and block slice is clamped to the file, and whatever the block table
says is handed to the decoder.

Used by ``tests/test_format3_native_cpp.py`` and as a CLI
(``write DIR`` / ``check DIR``).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "format3"
BTILE = 32
STEP = np.float32(0.00390625)
MAX_FILE = 64 << 20
U64 = 1 << 64


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < U64 else None


def write_case(path: Path, *, block: bytes, runs: bytes, rows: int, L_heavy: int, L: int,
               n_types: int, features=None) -> None:
    fo = ft = fao = fa = b""
    if features is not None:
        fo, ft, fao, fa = (np.ascontiguousarray(x).tobytes() for x in features)
    head = struct.pack("<11Q", rows % U64, L_heavy % U64, L % U64, n_types % U64,
                       1 if features is not None else 0, len(block), len(runs),
                       len(fo) // 4, len(ft), len(fao) // 4, len(fa))
    path.write_bytes(b"R3GCASE1" + head + block + runs + fo + ft + fao + fa)


def _read_regular(path: Path):
    """The file's bytes, or None for anything not a regular file (FIFO,
    symlink, ...) or too large: never blocks, never follows a link."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_FILE:
            return None
        with os.fdopen(os.dup(fd), "rb") as f:
            return f.read()
    finally:
        os.close(fd)


def raw_group_blocks(path: Path):
    """Yield ``(index, case_kwargs, first_row)`` for every block record of a
    group file, parsed without validation and clamped to the file."""
    data = _read_regular(path)
    if data is None or len(data) < 64 or data[:4] != b"R3G\0":
        return
    hoff = struct.unpack_from("<Q", data, 8)[0]
    if not 64 <= hoff <= len(data):
        return
    try:
        header = json.loads(data[hoff:])
    except (ValueError, RecursionError):
        return
    if not isinstance(header, dict):
        return
    secs = {}
    for e in header.get("sections", []) if isinstance(header.get("sections"), list) else []:
        if isinstance(e, dict) and isinstance(e.get("name"), str):
            off, ln = _int(e.get("offset")), _int(e.get("length"))
            if off is not None and ln is not None:
                secs[e["name"]] = data[off:off + ln]
    L_heavy, L, n_types = (_int(header.get(k)) for k in ("L_heavy", "L", "n_types"))
    if None in (L_heavy, L, n_types) or "blocks" not in secs or "coords" not in secs:
        return
    coords, blocks = secs["coords"], secs["blocks"]
    runs = secs.get("runs", b"")
    n_mol = len(runs) // 16
    feats = None
    if n_types:
        def arr(name, dt):
            b = secs.get(name, b"")
            return np.frombuffer(b[:len(b) - len(b) % np.dtype(dt).itemsize], dtype=dt)
        feats = (arr("feature_offsets", "<u4"), arr("feature_types", "u1"),
                 arr("feature_atom_offsets", "<u4"), arr("feature_atoms", "u1"))
    for i in range(len(blocks) // 64):
        rec = blocks[64 * i:64 * i + 64]
        first_row, rows, off, ln, m0, m1 = struct.unpack_from("<IIQQII", rec, 0)
        m0c, m1c = min(m0, n_mol), min(max(m1, m0), n_mol)
        f = None
        if feats is not None:
            f = (feats[0][m0c:m1c + 1], feats[1], feats[2], feats[3])
        yield i, dict(block=coords[off:off + ln], runs=runs[16 * m0c:16 * m1c], rows=rows,
                      L_heavy=L_heavy, L=L, n_types=n_types, features=f), first_row


def write_all(out_dir: Path) -> dict:
    """Write every case; return the manifest the checker needs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"conformance": {}, "widths": {}, "adversarial": {}}
    # Conformance corpus.
    for variant in ("shape", "color"):
        for gpath in sorted((DATA / variant / "groups").glob("*.r3g")):
            key = f"{variant}/groups/{gpath.name}"
            entries = []
            for i, kw, first_row in raw_group_blocks(gpath):
                name = f"conf_{variant}_{gpath.stem}_{i:04d}.case"
                write_case(out_dir / name, **kw)
                entries.append({"case": name, "first_row": first_row, "rows": kw["rows"],
                                "L_heavy": kw["L_heavy"], "L": kw["L"], "n_types": kw["n_types"]})
            manifest["conformance"][key] = entries
    # Per-width oracle blocks.
    wexp = json.loads((DATA / "widths" / "EXPECTED.json").read_text())
    for fname, e in sorted(wexp.items()):
        nr = e["n_real"]
        assert len(set(nr)) == 1, "a width vector with mixed n_real needs a multi-molecule run slice"
        runs = struct.pack("<QHB5x", 0, len(nr), nr[0])
        name = f"width_{fname}.case"
        write_case(out_dir / name, block=(DATA / "widths" / fname).read_bytes(), runs=runs,
                   rows=len(nr), L_heavy=e["L"], L=e["L"], n_types=0)
        manifest["widths"][fname] = {"case": name, "rows": len(nr), "L": e["L"]}
    # Adversarial seeds: committed, then generated into a scratch directory.
    aexp = json.loads((DATA / "adversarial" / "EXPECTED.json").read_text())
    gen_root = Path(tempfile.mkdtemp(prefix="r3g_adv_", dir=out_dir))
    sys.path.insert(0, str(DATA / "adversarial"))
    try:
        import make_adversarial  # noqa: E402  (test-data generator)
    finally:
        sys.path.pop(0)
    for seed, e in sorted(aexp.items()):
        root = DATA / "adversarial" / seed
        if e.get("generated"):
            root = gen_root / seed
            try:
                make_adversarial.build(seed, str(root))
            except Exception as exc:  # a generator that cannot run here is reported, not hidden
                manifest["adversarial"][seed] = {"error": repr(exc), "cases": []}
                continue
        cases = []
        gdir = root / "groups"
        paths = sorted(gdir.iterdir()) if gdir.is_dir() and not gdir.is_symlink() else []
        for gpath in paths:
            for i, kw, _ in raw_group_blocks(gpath):
                name = f"adv_{seed}_{gpath.stem}_{i:04d}.case"
                write_case(out_dir / name, **kw)
                cases.append(name)
        manifest["adversarial"][seed] = {"cases": cases, "code": e["code"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def _statuses(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        isa, what, code = line.split()
        out[(isa, what)] = code
    return out


def _tile(merged: np.ndarray, mode: int) -> np.ndarray:
    rows, _, L = merged.shape
    n_rows = -(-rows // BTILE) * BTILE
    m = np.zeros((n_rows, 3, L), np.int16)
    m[:rows] = merged
    t = m.reshape(n_rows // BTILE, BTILE, 3, L)
    return (t.transpose(0, 3, 1, 2) if mode == 1 else t.transpose(0, 3, 2, 1)).reshape(-1)


def check_all(out_dir: Path, isas=None) -> dict:
    """Check the driver's outputs; return a report. Raises AssertionError on
    any mismatch."""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    report = {"isas": None, "conformance": {}, "widths": {}, "adversarial": {}}
    exp = json.loads((DATA / "EXPECTED.json").read_text())
    wexp = json.loads((DATA / "widths" / "EXPECTED.json").read_text())

    def outputs(case):
        st = _statuses(out_dir / (case + ".status"))
        found = sorted({k[0] for k in st})
        if report["isas"] is None:
            report["isas"] = found
        assert found == report["isas"], (case, found)
        if isas is not None:
            assert found == sorted(isas), (case, found, isas)
        return st, found

    def same_across(case, found, suffix):
        blobs = [(out_dir / f"{case}.{i}.{suffix}").read_bytes() for i in found]
        assert all(b == blobs[0] for b in blobs), f"{case}: ISA paths differ in {suffix}"
        return blobs[0]

    for key, entries in manifest["conformance"].items():
        variant, _, gname = key.partition("/")
        e = exp[variant]["groups"][gname]
        heavy, merged, counts = [], [], []
        for ent in entries:
            st, found = outputs(ent["case"])
            assert all(v == "ok" for v in st.values()), (ent["case"], st)
            rows, Lh, L, nt = ent["rows"], ent["L_heavy"], ent["L"], ent["n_types"]
            h = np.frombuffer(same_across(ent["case"], found, "heavy"), np.int16).reshape(rows, 3, Lh)
            m = np.frombuffer(same_across(ent["case"], found, "merged"), np.int16).reshape(rows, 3, L)
            c = np.frombuffer(same_across(ent["case"], found, "counts"), np.uint8).reshape(rows, nt + 1)
            for mode, suf in ((1, "gpu1"), (-1, "gpu-1")):
                g = np.frombuffer(same_across(ent["case"], found, suf), np.int16)
                assert np.array_equal(g, _tile(m, mode)), (ent["case"], suf)
            cpu = np.frombuffer(same_across(ent["case"], found, "cpu"), np.float32).reshape(rows, L, 3)
            assert np.array_equal(cpu, m.transpose(0, 2, 1).astype(np.float32) * STEP), ent["case"]
            words = np.frombuffer(same_across(ent["case"], found, "words"), np.uint64)
            want = np.zeros(rows, np.uint64)
            for t in range(nt + 1):
                want |= c[:, t].astype(np.uint64) << np.uint64(8 * t)
            want |= c[:, 1:].sum(axis=1).astype(np.uint64) << np.uint64(56)
            assert np.array_equal(words, want), ent["case"]
            heavy.append(h); merged.append(m); counts.append(c)
        H, M, C = np.concatenate(heavy), np.concatenate(merged), np.concatenate(counts)
        pseudo = [M[r, :, int(C[r, 0]):int(C[r, 0]) + int(C[r, 1:].sum())].T for r in range(M.shape[0])]
        P = np.concatenate(pseudo) if pseudo else np.zeros((0, 3), np.int16)
        got = {"heavy_sha256": _sha(H), "merged_sha256": _sha(M), "counts_sha256": _sha(C),
               "pseudo_sha256": _sha(np.ascontiguousarray(P, dtype=np.int16))}
        for k, v in got.items():
            assert v == e[k], f"{key}: {k} {v} != {e[k]}"
        report["conformance"][key] = {"blocks": len(entries), "digests": "match"}

    for fname, ent in manifest["widths"].items():
        st, found = outputs(ent["case"])
        assert st[(found[0], "heavy")] == "ok", (fname, st)
        h = same_across(ent["case"], found, "heavy")
        assert hashlib.sha256(h).hexdigest() == wexp[fname]["decoded_sha256"], fname
        report["widths"][fname] = "match"

    for seed, ent in manifest["adversarial"].items():
        codes = set()
        for case in ent.get("cases", []):
            st, found = outputs(case)
            for what in ("heavy", "merged", "gpu1", "gpu-1", "cpu"):
                vals = {st[(i, what)] for i in found}
                assert len(vals) == 1, (case, what, vals)  # every ISA refuses alike
            codes |= set(st.values())
            for suffix in ("heavy", "merged", "counts", "gpu1", "gpu-1", "cpu", "words"):
                if (out_dir / f"{case}.{found[0]}.{suffix}").exists():
                    same_across(case, found, suffix)
        report["adversarial"][seed] = {"cases": len(ent.get("cases", [])),
                                       "native_codes": sorted(codes),
                                       "validator_code": ent.get("code"),
                                       **({"error": ent["error"]} if "error" in ent else {})}
    return report


if __name__ == "__main__":
    cmd, d = sys.argv[1], Path(sys.argv[2])
    if cmd == "write":
        m = write_all(d)
        print(json.dumps({k: len(v) for k, v in m.items()}))
    elif cmd == "check":
        r = check_all(d, isas=sys.argv[3].split(",") if len(sys.argv) > 3 else None)
        print(json.dumps(r, indent=1, sort_keys=True))
    else:
        raise SystemExit("usage: format3_native_cases.py write|check DIR [isa,isa]")

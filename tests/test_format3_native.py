"""The native format-3 codec through its Python binding.

* Every block of the conformance corpus decodes, on every compiled ISA path,
  to the reference decoder's arrays and to ``EXPECTED.json``'s digests
  (heavy, merged, pseudo, counts); the per-width vectors to theirs.
* The GPU writer equals ``engine/buffers.py::pack_into`` of the merged record
  (both tile modes, padding zeroed); the CPU writer equals
  ``prepare_cpu_block_coords`` of it; the count word is the CUDA binding's.
* The encoder reproduces the stored blocks and ``reference.encode_block``.
* The self-overlap binding slices ``vaa`` / ``color_self`` and reports an
  absent ``color_self`` as absent.
* Refusals are ``CodecError`` (a ``ValueError``) naming the failed check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from roshambo3.format.r3g import native as N
from roshambo3.format.r3g import reference as R

DATA = Path(__file__).resolve().parent / "data" / "format3"
EXPECTED = json.loads((DATA / "EXPECTED.json").read_text())
ISAS = N.compiled_isas()


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.fixture(autouse=True)
def _restore_isa():
    yield
    N.force_isa("auto")


def _groups():
    for variant in ("shape", "color"):
        for gname in sorted(EXPECTED[variant]["groups"]):
            yield variant, gname


def _blocks(gf):
    h = gf.header
    runs = gf.section_array("runs")
    blocks = gf.section_array("blocks")
    fa = None
    if h["n_types"]:
        fa = {k: gf.section_array(k) for k in
              ("feature_offsets", "feature_types", "feature_atom_offsets", "feature_atoms")}
    for i, b in enumerate(blocks):
        m0, m1 = int(b["m0"]), int(b["m1"])
        fs = None
        if fa is not None:
            fs = N.FeatureSlice(fa["feature_offsets"][m0:m1 + 1], fa["feature_types"],
                                fa["feature_atom_offsets"], fa["feature_atoms"])
        yield (gf.block_bytes(i), runs[m0:m1], int(b["rows"]), int(b["first_row"]), fs, runs,
               m0, m1)


def test_isas_and_forcing():
    assert "scalar" in ISAS and len(ISAS) == 2
    assert N.isa() in ISAS
    for isa in ISAS:
        N.force_isa(isa)
        assert N.isa() == isa
    missing = {"avx2", "neon"} - set(ISAS)
    for isa in missing:
        with pytest.raises(N.CodecError) as e:
            N.force_isa(isa)
        assert e.value.code == "decode.isa_unavailable"
    with pytest.raises(N.CodecError):
        N.force_isa("sse9")


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("variant,gname", list(_groups()))
def test_conformance_digests(isa, variant, gname):
    N.force_isa(isa)
    e = EXPECTED[variant]["groups"][gname]
    gf = R.open_group(str(DATA / variant / gname))
    try:
        h = gf.header
        ref = R.decode_group(gf)
        heavy = np.zeros((h["n_conf"], 3, h["L_heavy"]), np.int16)
        merged = np.zeros((h["n_conf"], 3, h["L"]), np.int16)
        counts = np.zeros((h["n_conf"], h["n_types"] + 1), np.uint8)
        for data, runs, rows, r0, fs, all_runs, m0, m1 in _blocks(gf):
            out = N.decode_block(data, runs, rows, h["L_heavy"])
            nr = np.repeat(all_runs["n_real"][m0:m1], all_runs["n_conf"][m0:m1])
            assert np.array_equal(out, R.decode_block(data, nr, h["L_heavy"]))
            heavy[r0:r0 + rows] = out
            mg, ct, t = N.decode_merged(data, runs, rows, h["L_heavy"], h["L"], h["n_types"], fs)
            merged[r0:r0 + rows] = mg
            counts[r0:r0 + rows] = ct
            assert all(x >= 0 for x in t) and t.self_overlap == 0.0
        assert np.array_equal(heavy, ref["heavy"])
        assert np.array_equal(merged, ref["merged"])
        assert np.array_equal(counts, ref["counts"])
        n_color = counts[:, 1:].sum(axis=1).astype(int)
        pseudo = [merged[r, :, int(counts[r, 0]):int(counts[r, 0]) + n_color[r]].T
                  for r in range(merged.shape[0])]
        pseudo = np.ascontiguousarray(np.concatenate(pseudo), dtype=np.int16)
        assert _sha(heavy) == e["heavy_sha256"]
        assert _sha(merged) == e["merged_sha256"]
        assert _sha(counts) == e["counts_sha256"]
        assert _sha(pseudo) == e["pseudo_sha256"]
    finally:
        gf.close()


@pytest.mark.parametrize("isa", ISAS)
def test_width_vectors(isa):
    N.force_isa(isa)
    for fname, e in json.loads((DATA / "widths" / "EXPECTED.json").read_text()).items():
        nr = e["n_real"]
        runs = R.pack_runs([0], [len(nr)], [nr[0]])
        out = N.decode_block((DATA / "widths" / fname).read_bytes(), runs, len(nr), e["L"])
        assert _sha(out) == e["decoded_sha256"], fname


def _pack_into(merged, L, planar):
    from roshambo3.engine.buffers import pack_into
    from roshambo3.numerics import WIRE_QUANT_STEP
    n_rows, n = N.gpu_layout_elements(merged.shape[0], L)
    dst = np.full(n, 12345, np.int16)
    view, nb = pack_into(merged, L, dst, float(WIRE_QUANT_STEP), tile=True, planar=planar)
    return dst


@pytest.mark.parametrize("variant,gname", list(_groups()))
def test_writers_match_consumers(variant, gname):
    from roshambo3._roshambo3_cpu import prepare_cpu_block_coords
    gf = R.open_group(str(DATA / variant / gname))
    try:
        h = gf.header
        vaa = gf.section_array("vaa")
        cs = gf.section_array("color_self") if "color_self" in gf.sections else None
        for data, runs, rows, r0, fs, _, _, _ in _blocks(gf):
            merged, counts, _ = N.decode_merged(data, runs, rows, h["L_heavy"], h["L"],
                                                h["n_types"], fs)
            so = N.SelfOverlaps(vaa, cs, r0)
            for isa in ISAS:
                N.force_isa(isa)
                for mode, planar in ((1, False), (-1, True)):
                    n_rows, n = N.gpu_layout_elements(rows, h["L"])
                    out = np.full(n + 7, -7, np.int16)
                    res = N.write_gpu_block(data, runs, rows, h["L_heavy"], h["L"], h["n_types"],
                                            fs, out, tile_mode=mode, self_overlaps=so)
                    assert np.array_equal(out[:n], _pack_into(merged, h["L"], planar))
                    assert np.all(out[n:] == -7)
                    assert res.n_rows == n_rows and res.rows == rows
                    assert res.coords.shape == (n_rows, h["L"], 3)
                    assert np.array_equal(res.counts, counts)
                    want = np.zeros(rows, np.uint64)
                    for t in range(counts.shape[1]):
                        want |= counts[:, t].astype(np.uint64) << np.uint64(8 * t)
                    want |= counts[:, 1:].sum(axis=1).astype(np.uint64) << np.uint64(56)
                    assert np.array_equal(res.count_words, want)
                    assert np.array_equal(res.vaa, vaa[r0:r0 + rows])
                    if cs is None:
                        assert res.color_self is None
                    else:
                        assert np.array_equal(res.color_self, cs[r0:r0 + rows])
                    assert res.timings.self_overlap >= 0.0
                scale = np.float32(2.0 ** -8)
                res = N.write_cpu_block(data, runs, rows, h["L_heavy"], h["L"], h["n_types"],
                                        fs, float(scale), self_overlaps=so)
                want = prepare_cpu_block_coords(merged, counts, h["L"], float(scale),
                                                merged_color=h["n_types"] > 0)
                assert res.coords.dtype == np.float32
                assert res.coords.tobytes() == np.asarray(want).tobytes()
    finally:
        gf.close()


def test_encoder_reproduces_stored_blocks():
    for variant, gname in _groups():
        gf = R.open_group(str(DATA / variant / gname))
        try:
            for data, runs, rows, _, _, all_runs, m0, m1 in _blocks(gf):
                heavy = N.decode_block(data, runs, rows, gf.header["L_heavy"])
                assert N.encode_block(heavy, runs) == bytes(data)
        finally:
            gf.close()


def test_encoder_matches_reference_random():
    rng = np.random.default_rng(20261018)
    for trial in range(60):
        n_mol = int(rng.integers(1, 6))
        n_conf = rng.integers(1, 40, n_mol)
        n_real = rng.integers(1, 80, n_mol)
        L = R.pad8(int(n_real.max()))
        rows = int(n_conf.sum())
        w = int(rng.integers(0, 17))
        lo = -32768 if w == 16 else int(rng.integers(-32768, 32768 - (1 << w) + 1))
        xyz = rng.integers(lo, lo + (1 << w), (rows, 3, L)).astype(np.int16)
        nr = np.repeat(n_real, n_conf)
        xyz[np.broadcast_to(np.arange(L)[None, None, :] >= nr[:, None, None], xyz.shape)] = 0
        runs = R.pack_runs(np.arange(n_mol), n_conf, n_real)
        enc = N.encode_block(xyz, runs)
        assert enc == R.encode_block(xyz, nr)
        for isa in ISAS:
            N.force_isa(isa)
            assert np.array_equal(N.decode_block(enc, runs, rows, L), xyz)


def _one_block():
    xyz = np.zeros((2, 3, 8), np.int16)
    xyz[:, :, :5] = np.arange(30, dtype=np.int16).reshape(2, 3, 5) * 37
    runs = R.pack_runs([7], [2], [5])
    return N.encode_block(xyz, runs), runs, xyz


def test_refusals_are_codec_errors():
    data, runs, _ = _one_block()

    def code(fn):
        with pytest.raises(N.CodecError) as e:
            fn()
        assert isinstance(e.value, ValueError)
        assert str(e.value).startswith(e.value.code + ":")
        return e.value.code

    assert code(lambda: N.decode_block(data, runs, 0, 8)) == "layout.blocks"
    assert code(lambda: N.decode_block(data, runs, 3, 8)) == "layout.runs"
    assert code(lambda: N.decode_block(data[:-1], runs, 2, 8)) == "limit.block_length"
    assert code(lambda: N.decode_block(data[:10], runs, 2, 8)) == "limit.block_length"
    bad = bytearray(data); bad[7] = 17
    assert code(lambda: N.decode_block(bytes(bad), runs, 2, 8)) == "limit.bit_width"
    bad = bytearray(data); bad[20] = 1
    assert code(lambda: N.decode_block(bytes(bad), runs, 2, 8)) == "layout.block_header"
    assert code(lambda: N.decode_block(data, runs, 2, 4)) == "layout.widths"
    assert code(lambda: N.decode_block(data, runs, 2, 8, out=np.zeros(47, np.int16))) == \
        "decode.output_capacity"
    assert code(lambda: N.decode_block(data, runs, 2, 8, out=np.zeros(48, np.int32))) == \
        "decode.argument"
    assert code(lambda: N.decode_block(data, runs.tobytes()[:15], 2, 8)) == "decode.argument"
    out = np.zeros(32 * 8 * 3, np.int16)
    assert code(lambda: N.write_gpu_block(data, runs, 2, 8, 8, 0, None, out, tile_mode=0)) == \
        "decode.argument"
    assert code(lambda: N.write_gpu_block(data, runs, 2, 8, 8, 0, None, out[:-1])) == \
        "decode.output_capacity"
    assert code(lambda: N.write_cpu_block(data, runs, 2, 8, 8, 0, None, 0.0)) == "decode.argument"
    assert code(lambda: N.write_cpu_block(data, runs, 2, 8, 8, 7, None, 1.0)) == "limit.n_types"
    so = N.SelfOverlaps(np.ones(1, np.float32), None, 0)
    assert code(lambda: N.write_cpu_block(data, runs, 2, 8, 8, 0, None, 1.0,
                                          self_overlaps=so)) == "decode.self_overlap_range"
    fs = N.FeatureSlice(np.array([0, 1], np.uint32), np.array([1], np.uint8),
                        np.array([0, 1], np.uint32), np.array([5], np.uint8))
    assert code(lambda: N.decode_merged(data, runs, 2, 8, 8, 1, fs)) == "layout.features"
    assert code(lambda: N.decode_merged(data, runs, 2, 8, 8, 1, None)) == \
        "decode.features_missing"
    assert code(lambda: N.block_length(10, 17, 0, 0)) == "limit.bit_width"
    assert N.block_length(3, 2, 0, 9) == 384  # FORMAT.md Appendix A


def test_heavy_decode_into_caller_buffer_and_any_buffer_type():
    data, runs, xyz = _one_block()
    out = np.full(100, 99, np.int16)
    got = N.decode_block(memoryview(data), runs, 2, 8, out=out)
    assert np.array_equal(got, xyz) and np.all(out[48:] == 99)
    arr = np.frombuffer(b"\0" + data, np.uint8)[1:]  # an unaligned uint8 view
    assert np.array_equal(N.decode_block(arr, runs.view(np.uint8).reshape(-1, 16), 2, 8), xyz)


def test_read_into_is_bounded_and_refuses_a_short_read(tmp_path):
    """FORMAT.md 17 item 12 / S7: bounded pread; a file truncated after it was
    opened is io.short_read, not a SIGBUS."""
    import os
    data, runs, xyz = _one_block()
    path = tmp_path / "blocks.bin"
    path.write_bytes(b"\xAA" * 100 + data)
    fd = os.open(path, os.O_RDONLY)
    try:
        got = N.read_into(fd, 100, len(data))
        assert got.tobytes() == data
        assert np.array_equal(N.decode_block(got, runs, 2, 8), xyz)
        out = np.zeros(len(data) + 5, np.uint8)
        N.read_into(fd, 100, len(data), out=out)
        assert out[:len(data)].tobytes() == data and not out[len(data):].any()
        os.truncate(path, 100 + len(data) - 1)  # truncated after open
        with pytest.raises(N.CodecError) as e:
            N.read_into(fd, 100, len(data))
        assert e.value.code == "io.short_read"
        with pytest.raises(N.CodecError) as e:
            N.read_into(fd, 2 ** 63 - 2, 10)
        assert e.value.code == "decode.overflow"
    finally:
        os.close(fd)
    with pytest.raises(N.CodecError) as e:
        N.read_into(fd, 0, 4)  # closed descriptor
    assert e.value.code == "io.read_error"

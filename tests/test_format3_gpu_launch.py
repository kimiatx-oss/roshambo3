"""Native launch bounds and placement; no CUDA context or timed harness."""
import numpy as np
import pytest

from roshambo3.format.r3g import native as N, reference as R


def inputs(sizes=(7, 31, 3), ids=None):
    ids = list(range(len(sizes))) if ids is None else ids
    blocks = []
    expected = []
    for i, (rows, mid) in enumerate(zip(sizes, ids, strict=True)):
        runs = R.pack_runs([mid], [rows], [1])
        heavy = np.zeros((rows, 3, 8), dtype=np.int16)
        heavy[:, :, 0] = np.arange(rows * 3, dtype=np.int16).reshape(rows, 3) + i * 311
        blocks.append((N.encode_block(heavy, runs), runs, rows, None))
        expected.append(heavy)
    return blocks, np.concatenate(expected)


def call(blocks, rows=None, **kwargs):
    total = sum(b[2] for b in blocks) if rows is None else rows
    out = np.full(((total + 31) // 32) * 32 * 8 * 3, 24736, dtype=np.int16)
    return N.write_gpu_launch(blocks, total, 8, 8, 0, out, **kwargs)


@pytest.mark.parametrize("tile", [1, -1])
@pytest.mark.parametrize("sizes", [(7, 31, 3), (32, 33, 31), (4096, 4096, 1)])
@pytest.mark.parametrize("isa", N.compiled_isas())
def test_native_tiles_and_ids_across_physical_boundaries(tile, sizes, isa):
    N.force_isa(isa)
    try:
        blocks, heavy = inputs(sizes, ids=[0, 2**63 - 1, 103])
        coords, counts, ids = call(blocks, tile_mode=tile)
        nt = coords.shape[0] // 32
        xyz = (coords.reshape(nt, 8, 32, 3).transpose(0, 2, 3, 1)
               if tile == 1 else coords.reshape(nt, 8, 3, 32).transpose(0, 3, 2, 1))
        xyz = xyz.reshape(-1, 3, 8)
        np.testing.assert_array_equal(xyz[:len(heavy)], heavy)
        assert not xyz[len(heavy):].any()
        np.testing.assert_array_equal(ids, np.repeat([0, 2**63 - 1, 103], sizes))
        assert counts.shape == (sum(sizes), 1) and np.all(counts == 1)
    finally:
        N.force_isa("auto")


def test_launch_refuses_late_corruption_before_writes():
    blocks, _ = inputs()
    bad = bytearray(blocks[-1][0])
    bad[6] = 17
    blocks[-1] = (bad, *blocks[-1][1:])
    out = np.full(64 * 8 * 3, 123, np.int16)
    with pytest.raises(N.CodecError, match="limit.bit_width"):
        N.write_gpu_launch(blocks, 41, 8, 8, 0, out)
    assert np.all(out == 123)


@pytest.mark.parametrize("which", ["out", "counts", "ids", "rows", "id_range", "alias"])
def test_launch_refusals(which):
    blocks, _ = inputs()
    out = np.empty(64 * 8 * 3, np.int16)
    counts = np.empty(41, np.uint8)
    ids = np.empty(41, np.int64)
    rows = 41
    if which == "out":
        out = out[:2]
    elif which == "counts":
        counts = counts[:2]
    elif which == "ids":
        ids = ids[:2]
    elif which == "rows":
        rows = 40
    elif which == "id_range":
        blocks[1][1].view(np.uint8)[7] = 128
    elif which == "alias":
        counts = out.view(np.uint8)[:41]
    with pytest.raises(N.CodecError):
        N.write_gpu_launch(blocks, rows, 8, 8, 0, out,
                           counts_out=counts, mol_ids_out=ids)


def test_raw_native_capacity_refusal():
    blocks, _ = inputs()
    raw = [(memoryview(b), r.view(np.uint8), rows, f) for b, r, rows, f in blocks]
    with pytest.raises(N.CodecError, match="decode.output_capacity"):
        N._ext().r3g_write_gpu_launch(raw, 41, 8, 8, 0, 1,
                                    np.empty(3, np.int16), np.empty(41, np.uint8),
                                    np.empty(41, np.int64))


@pytest.mark.parametrize("tile", [1, -1])
@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("sizes", [(1,) * 33, (31, 1, 32, 33, 7, 31, 2, 17, 23),
                                   (4050, 4050, 61)])
@pytest.mark.parametrize("isa", N.compiled_isas())
@pytest.mark.parametrize("builder", ["baseline", "fused"])
@pytest.mark.parametrize("heavy_width,nreal_pair,device_width",
                         [(8, (3, 7), 11), (16, (9, 15), 33),
                          (256, (255, 255), 512)])
def test_launch_partial_tiles_match_merged_oracle(tile, color, sizes, isa, builder,
                                                 heavy_width, nreal_pair, device_width):
    """Every start residue, full/partial boundaries and real builder row sizes.

    The independent merged-record decoder supplies coordinates and metadata;
    only the launch writer performs tile placement. Output sentinels prove that
    final padding and arbitrary physical boundaries stay inside the capacity.
    """
    N.force_isa(isa)
    N._set_builder_mode(builder)
    try:
        rng = np.random.default_rng(29420261021)
        L, nt = device_width, 2 if color else 0
        blocks, expected, expected_counts, expected_ids = [], [], [], []
        for bi, rows in enumerate(sizes):
            confs = [rows] if rows == 1 else [rows // 2, rows - rows // 2]
            nreal = list(nreal_pair[:len(confs)])
            mids = [2**40 + bi * 2 + j for j in range(len(confs))]
            runs = R.pack_runs(mids, confs, nreal)
            heavy = rng.integers(-32000, 32001, size=(rows, 3, heavy_width), dtype=np.int16)
            real_per_row = np.repeat(nreal, confs)
            heavy[np.broadcast_to(np.arange(heavy_width)[None, None, :] >=
                                  real_per_row[:, None, None], heavy.shape)] = 0
            features = None
            if color:
                # Nonzero feature offsets exercise group-wide sidecar slices.
                atom_lists = [[0], [0]] + [[0], [0, 1], [0, 1, 2]] * len(confs)
                features = N.FeatureSlice(
                    np.arange(2, 3 * len(confs) + 3, 3, dtype=np.uint32),
                    np.array([1, 1] + [1, 2, 2] * len(confs), np.uint8),
                    np.array([0, *np.cumsum([len(a) for a in atom_lists])], np.uint32),
                    np.array([a for atoms in atom_lists for a in atoms], np.uint8))
            encoded = N.encode_block(heavy, runs)
            merged, counts, _ = N.decode_merged(encoded, runs, rows, heavy_width, L, nt, features)
            blocks.append((encoded, runs, rows, features))
            expected.append(merged)
            expected_counts.append(counts)
            expected_ids.extend(np.repeat(mids, confs))
        total = sum(sizes)
        padded = ((total + 31) // 32) * 32
        n = padded * L * 3
        out = np.full(n + 29, 24736, np.int16)
        counts_out = np.full(total * (nt + 1) + 19, 199, np.uint8)
        ids_out = np.full(total + 11, -891, np.int64)
        coords, counts, ids = N.write_gpu_launch(
            blocks, total, heavy_width, L, nt, out[:n], tile_mode=tile,
            counts_out=counts_out[:total * (nt + 1)], mol_ids_out=ids_out[:total])
        xyz = (coords.reshape(-1, L, 32, 3).transpose(0, 2, 3, 1)
               if tile == 1 else coords.reshape(-1, L, 3, 32).transpose(0, 3, 2, 1))
        xyz = xyz.reshape(padded, 3, L)
        np.testing.assert_array_equal(xyz[:total], np.concatenate(expected))
        assert not xyz[total:].any()
        np.testing.assert_array_equal(counts, np.concatenate(expected_counts))
        np.testing.assert_array_equal(ids, expected_ids)
        assert np.all(out[n:] == 24736)
        assert np.all(counts_out[total * (nt + 1):] == 199)
        assert np.all(ids_out[total:] == -891)
    finally:
        N._set_builder_mode("auto")
        N.force_isa("auto")

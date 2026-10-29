"""Queries live on the library wire record's wire-step grid.

The CUDA analytic kernel reads the query as an int16 record on that
grid; when its self-overlap V_AA was computed from the unrounded float32 query,
so the Tanimoto's numerator and denominator described two coordinate sets
(+3.16e-5 mean bias on the golden queries).
`engine.query.from_arrays` now snaps every query coordinate to the grid once,
with the format's own operations, so every backend scores one coordinate set.

The general per-backend oracle / frame-consistency test family is not built
here.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from roshambo3.numerics import WireQuantizationError

from roshambo3.engine.query import from_arrays, snap_to_wire_grid
from roshambo3.format import constants as wire
from roshambo3.numerics import WIRE_QUANT_STEP
from roshambo3.prep.conformer import stored_self_volume

from gpu_device import device_present


def _on_grid(values_i16):
    return values_i16.astype(np.float32) * np.float32(WIRE_QUANT_STEP)


# --------------------------------------------------------------------------
# the snap itself
# --------------------------------------------------------------------------

def test_the_snap_is_the_identity_on_every_int16_value():
    """Exhaustive: a query taken from a corpus must come back bit for bit."""
    v = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    x = _on_grid(v)
    got = snap_to_wire_grid(x.reshape(-1, 1))
    assert got.dtype == np.float32
    assert np.array_equal(got.reshape(-1).view(np.uint32), x.view(np.uint32))


def test_from_arrays_is_a_no_op_on_grid_aligned_queries():
    """Every int16 value, through the public funnel, shape and color alike."""
    v = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    xyz = _on_grid(v).reshape(256, 256)            # 256 queries
    blocks = [np.ascontiguousarray(xyz[i, :255].reshape(85, 3)) for i in range(256)]
    tail = [np.ascontiguousarray(np.repeat(xyz[i, 255], 3).reshape(1, 3))
            for i in range(256)]
    color = [(t, np.array([1], np.int32)) for t in tail]
    rmat = np.ones((2, 2), np.float32)
    pmat = np.ones((2, 2), np.float32)

    class _Gen:
        interaction_matrix_r = rmat
        interaction_matrix_p = pmat

    qs = from_arrays(blocks, [f"q{i}" for i in range(256)], backend="cpu",
                     color=color, color_generator=_Gen())
    for i in range(256):
        assert np.array_equal(qs.coords[i, :85, :3].view(np.uint32),
                              blocks[i].view(np.uint32))
        assert np.array_equal(qs.coords[i, 85, :3].view(np.uint32),
                              tail[i][0].view(np.uint32))
        assert qs.vaa[i] == np.float32(stored_self_volume(blocks[i]))


def test_an_off_grid_query_is_snapped_and_its_vaa_follows():
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 2.0, size=(17, 3))           # float64, off grid
    c = rng.normal(0.0, 2.0, size=(4, 3))
    qs = from_arrays([a], ["q"], backend="cpu",
                     color=[(c, np.array([1, 1, 2, 2], np.int32))],
                     color_generator=type("G", (), {
                         "interaction_matrix_r": np.ones((3, 3), np.float32),
                         "interaction_matrix_p": np.ones((3, 3), np.float32)}))
    want = _on_grid(wire.quantize(a))
    assert np.array_equal(qs.coords[0, :17, :3], want)
    assert np.array_equal(qs.coords[0, 17:21, :3], _on_grid(wire.quantize(c)))
    assert not np.array_equal(qs.coords[0, :17, :3], a.astype(np.float32))
    # V_AA describes the snapped coordinates, not the ones passed in.
    assert qs.vaa[0] == np.float32(stored_self_volume(want))


def test_a_float32_and_a_float64_query_snap_from_their_own_precision():
    """from_rdkit passes float64; a float32 caller gets its own value rounded."""
    a64 = np.array([[0.0004999999, 1.0, -2.0], [3.0, 0.0, 1.0]])
    q = from_arrays([a64], ["q"], backend="cpu").coords[0, :2, :3]
    assert np.array_equal(q, _on_grid(wire.quantize(a64)))


@pytest.mark.parametrize("bad", [128.0, -128.00390625, 140.0])
def test_a_coordinate_off_the_int16_range_raises_rather_than_clips(bad):
    a = np.zeros((3, 3))
    a[1, 2] = bad
    with pytest.raises(ValueError, match="query 0 has a coordinate outside"):
        from_arrays([a], ["q"], backend="cpu")
    c = np.zeros((1, 3))
    c[0, 0] = bad
    with pytest.raises(ValueError, match="query 0 color has a coordinate"):
        from_arrays([np.zeros((3, 3))], ["q"], backend="cpu",
                    color=[(c, np.array([1], np.int32))])


def test_the_edge_of_the_range_is_representable():
    a = np.array([[127.99609375, -128.0, 0.0], [0.0, 0.0, 0.0]])
    q = from_arrays([a], ["q"], backend="cpu").coords[0, :2, :3]
    assert np.array_equal(q, _on_grid(np.array([[32767, -32768, 0], [0, 0, 0]],
                                               np.int16)))


def test_from_rdkit_matches_the_corpus_builders_record():
    """A query prepared from a conformer equals that conformer's library
    record: the builder quantizes the float64 prepared coordinates
    (prep/sources.py `_prep_one`, then prep/build.py `fmt.quantize`)."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    from roshambo3.engine.query import from_rdkit
    from roshambo3.prep.sources import _prep_one

    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1C(F)(F)F"))
    assert AllChem.EmbedMolecule(m, randomSeed=11) == 0
    qs = from_rdkit([m], names=["q"], backend="cpu")
    prepared = _prep_one(m)[0]
    want = _on_grid(wire.quantize(prepared))
    n = want.shape[0]
    assert np.array_equal(qs.coords[0, :n, :3], want)
    assert qs.vaa[0] == np.float32(stored_self_volume(want))


# --------------------------------------------------------------------------
# one coordinate set on the device
# --------------------------------------------------------------------------

def _cuda():
    from roshambo3.backend import cuda
    if not cuda.available() or not device_present():
        pytest.skip("no CUDA extension in this build, or no device")
    return cuda


def _golden_panel(int16_library):
    import bench.make_golden as mk
    from roshambo3.layout import tile32

    coords, n_real = mk.make_input()
    nq, nl = mk.N_QUERIES, mk.N_LIBRARY
    lib = np.ascontiguousarray(coords[nq:])
    tn = np.ascontiguousarray(n_real[nq:])
    if int16_library:
        q16 = wire.quantize(lib[..., :3])
        q16[lib[..., 3] == 0] = 0
        tc = tile32(np.ascontiguousarray(q16))
        lib_xyz = [_on_grid(q16[j, :tn[j]]).astype(np.float64) for j in range(nl)]
    else:
        tc = tile32(lib)
        lib_xyz = [lib[j, :tn[j], :3].astype(np.float64) for j in range(nl)]
    raw_q = [coords[i, :n_real[i], :3] for i in range(nq)]
    return mk, coords, n_real, raw_q, tc, tn, lib_xyz


def _kernel_minus_fp64(cuda, qc, qn, tc, tn, lib_xyz, q_xyz, mk, pose_search):
    """Kernel shape Tanimoto minus fp64 at the kernel's own pose, with BOTH
    V_AA and V_AB taken from `q_xyz` (the kernel's composition: R(q), t, then the
    start rotation). Also returns the kernel's V_AA column."""
    from roshambo3.backend import UpdateRule
    from roshambo3.reference import overlap as ref

    nq, nl = qc.shape[0], tn.shape[0]
    pq = np.repeat(np.arange(nq, dtype=np.int32), nl)
    pt = np.tile(np.arange(nl, dtype=np.int32), nq)
    v = cuda.score_pairs(
        query_coords=qc, query_types=np.zeros(qc.shape[:2], np.int32),
        query_n_atoms=qn, library_coords=tc,
        library_counts=tn.astype(np.uint8).reshape(-1, 1),
        pair_query_row=pq, pair_library_row=pt,
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.zeros((1, 1), np.float32),
        pose_search=pose_search, update_rule=UpdateRule.ADAGRAD_QADD, **mk.OPT)
    vaa = [ref.self_overlap(a) for a in q_xyz]
    vbb = [ref.self_overlap(b) for b in lib_xyz]
    err = np.empty(v.shape[0])
    for r in range(v.shape[0]):
        s = v[r].astype(np.float64)
        bt = ref.apply_pose(ref.apply_pose(lib_xyz[pt[r]], s[9:13], s[13:16]),
                            s[16:20], np.zeros(3))
        err[r] = s[1] - ref.tanimoto(ref.pair_overlap(q_xyz[pq[r]], bt),
                                     vaa[pq[r]], vbb[pt[r]])
    vaa_rel = v[:, 5].astype(np.float64) / np.array([vaa[i] for i in pq]) - 1.0
    return err, vaa_rel


# Bounds. The defect this file exists for is a numerator/denominator split
# of one wire-step query rounding: on these inputs it is a +2.6e-5 to +3.2e-5
# MEAN bias in the shape Tanimoto and a 4.4e-5 relative V_AA offset
# (+3.16e-5 on the golden queries).
# float32 evaluation with ex2.approx sits about 1e-6 per row and 2e-7 relative
# in V_AA. Each bound below is set at least 5x under the defect's size and
# several-fold over the arithmetic floor, so it can fail on the split and pass
# on rounding. Per-row bounds are only asserted where the library is on the
# grid too: a float32 library is re-quantized by int16 staging (its declared
# budget, p99 1.3e-4 per row) and only the mean is meaningful there.
MEAN_BIAS_BOUND = 5e-6
VAA_REL_BOUND = 2e-6
PER_ROW_BOUND_I16 = 1e-5


@pytest.mark.parametrize("int16_library", [True, False],
                         ids=["int16-library", "golden-float32-library"])
def test_the_analytic_kernel_scores_one_query_coordinate_set(int16_library):
    cuda = _cuda()
    from roshambo3.backend import PoseSearch

    mk, coords, n_real, raw_q, tc, tn, lib_xyz = _golden_panel(int16_library)
    qs = from_arrays(raw_q, [f"q{i}" for i in range(len(raw_q))], backend="cpu")
    qn = qs.n_real
    snapped = [qs.coords[i, :qn[i], :3].astype(np.float64) for i in range(len(raw_q))]
    err, vaa_rel = _kernel_minus_fp64(cuda, qs.coords, qn, tc, tn, lib_xyz,
                                      snapped, mk, PoseSearch.ANALYTIC)
    assert abs(err.mean()) < MEAN_BIAS_BOUND, err.mean()
    assert np.abs(vaa_rel).max() < VAA_REL_BOUND, np.abs(vaa_rel).max()
    if int16_library:
        assert np.abs(err).max() < PER_ROW_BOUND_I16, np.abs(err).max()


def test_direct_off_grid_query_self_uses_the_scored_wire_frame():
    """Raw direct calls reconstruct query self terms on the same int16
    wire grid as their analytic cross terms, including off-grid inputs."""
    cuda = _cuda()
    from roshambo3.backend import PoseSearch

    mk, coords, n_real, raw_q, tc, tn, lib_xyz = _golden_panel(True)
    nq = len(raw_q)
    qc = np.ascontiguousarray(coords[:nq])
    qn = np.ascontiguousarray(n_real[:nq])
    snapped = [_on_grid(wire.quantize(a)).astype(np.float64) for a in raw_q]
    err, vaa_rel = _kernel_minus_fp64(cuda, qc, qn, tc, tn, lib_xyz, snapped,
                                      mk, PoseSearch.ANALYTIC)
    assert abs(err.mean()) <= MEAN_BIAS_BOUND
    assert np.abs(vaa_rel).max() <= VAA_REL_BOUND


def test_cpu_and_cuda_receive_the_same_query_array(monkeypatch, smoke_corpus):
    """The engine hands both backends `QuerySet.coords`, the snapped record."""
    if not os.environ.get("ROSHAMBO3_CORPORA"):
        pytest.skip("needs ROSHAMBO3_CORPORA for a real corpus")
    pytest.importorskip("roshambo3._roshambo3_cpu")
    cuda = _cuda()
    import roshambo3.backend.cpu as cpu
    from roshambo3.engine import Engine, HostTopK, ScreenConfig
    from roshambo3.engine.r3g_source import R3gSource
    from roshambo3.format import Corpus
    from roshambo3.paths import corpus_path

    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.5, size=(14, 3))
    qs = from_arrays([a], ["q"])
    corpus = smoke_corpus

    class _Seen(Exception):
        pass

    seen = {}
    for name, mod, kw in (("cpu", cpu, {"tile": False}), ("cuda", cuda, {})):
        def spy(*args, _name=name, **kwargs):
            seen[_name] = np.array(kwargs["query_coords"], copy=True)
            raise _Seen
        monkeypatch.setattr(mod, "score_pairs", spy)
        with pytest.raises(_Seen):
            Engine(backend=mod).run(
                R3gSource(corpus, workers=1, conformers_per_launch=1024, **kw),
                qs, config=ScreenConfig(), sink=HostTopK(k=5))
    assert np.array_equal(seen["cpu"][..., :3], seen["cuda"][..., :3])
    assert np.array_equal(seen["cuda"], qs.coords)
    assert np.array_equal(seen["cpu"][0, :14, :3], _on_grid(wire.quantize(a)))


# --------------------------------------------------------------------------
# the funnel: a QuerySet built by hand is held to the same grid
# --------------------------------------------------------------------------

def _fields(qs, **over):
    kw = dict(coords=qs.coords, types=qs.types, n_real=qs.n_real,
              names=list(qs.names), interaction_r=qs.interaction_r,
              interaction_p=qs.interaction_p, vaa=qs.vaa)
    kw.update(over)
    return kw


def test_a_hand_built_query_set_on_the_grid_is_accepted_unchanged():
    from dataclasses import replace
    from roshambo3.engine.query import QuerySet
    rng = np.random.default_rng(3)
    qs = from_arrays([rng.normal(0, 3, (7, 3)), rng.normal(0, 3, (4, 3))],
                     ["a", "b"], backend="cpu")
    again = QuerySet(**_fields(qs))
    assert again.coords is qs.coords
    # A float64 copy of on-grid float32 values is the same float32 record.
    QuerySet(**_fields(qs, coords=qs.coords.astype(np.float64)))
    replace(qs, names=["c", "d"])
    # Negative zero is zero on the grid.
    neg = qs.coords.copy()
    neg[0, 0, 0] = np.float32(-0.0)
    QuerySet(**_fields(qs, coords=neg))


def test_a_hand_built_query_set_off_the_grid_is_refused():
    """``api.load_queries`` returns a QuerySet as given and ``Engine.run``
    reads ``coords`` directly; constructing one is where the grid is held."""
    from roshambo3.api import load_queries
    from roshambo3.engine.query import QuerySet
    rng = np.random.default_rng(4)
    qs = from_arrays([rng.normal(0, 3, (7, 3))], ["a"], backend="cpu")
    off = qs.coords.copy()
    off[0, 2, 1] += np.float32(1e-4)
    with pytest.raises(ValueError, match="off the library wire record's wire-step grid"):
        QuerySet(**_fields(qs, coords=off))
    with pytest.raises(ValueError, match="int16 range"):
        big = qs.coords.copy()
        big[0, 0, 0] = 140.0
        QuerySet(**_fields(qs, coords=big))
    assert load_queries(qs) is qs

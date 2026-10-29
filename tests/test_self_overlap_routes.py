"""Builder-produced format-3 self sections, truthful counters, and the self-overlap margin policy.

Format-3 sections are authenticated before use and bounded slices are checked
at each claim. Stored and recomputed values both describe the scored frame.
"""
import hashlib
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from roshambo3 import scores as scores_module
from roshambo3.backend import cpu
from cpu_test_capabilities import RETAINED_COLOR_GRAIN
from roshambo3.reference.color import MODE_COMBO_OPTIMAL
from roshambo3.engine import Engine, R3gSource, ScreenConfig
from roshambo3.engine import cpu_modes
from roshambo3.engine.config import ColorPlacement
from roshambo3.engine.query import from_rdkit
from roshambo3.format.r3g.errors import CorpusValidationError
from roshambo3.format.r3g.reader import open_corpus, verify
from roshambo3.pharmacophore import PharmacophoreGenerator
from roshambo3.prep import prepare
from tests.test_prepare_self_sidecars import _jittered_library, _matrices

MODES = ("shape", "rescore", "combo")
SHAPE_CONVERSION = "float32(VAA)*float32(1/SHAPE_PREFACTOR)"
COMBO_CONVERSION = "float32(float64(VAA)/float64(SHAPE_PREFACTOR))"


def build_inputs(path, *, color_self=True, noncanonical=False):
    cpu.extension()
    base, mols = _jittered_library(n_mols=5, n_confs=2)
    generator = PharmacophoreGenerator()
    if noncanonical:
        generator.interaction_matrix_p[1, 2] = .4
    corpus = prepare(mols, str(path), workers=1, color_generator=generator,
                     color_self=color_self, block_rows=4)
    return corpus, base, generator


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_inputs(tmp_path_factory.mktemp("routes") / "corpus")


def _inputs(corpus, base, generator, mode):
    color = mode != "shape"
    source = R3gSource(corpus, color=color, color_generator=generator if color else None,
                       tile=False, workers=1, prefetch_depth=0)
    queries = from_rdkit([Chem.Mol(base)], color_generator=generator if color else None)
    config = ScreenConfig(color=color, mixing=.5 if color else 0.,
        color_placement=ColorPlacement.IN_GRADIENT if mode == "combo" else ColorPlacement.AT_POSE,
        field_spacing_a=.5, steps=25)
    return source, queries, config


def _run(built, mode, policy="prepared", *, engine=None, inputs=None):
    source, queries, config = inputs or _inputs(*built, mode)
    owned = engine is None
    engine = engine or Engine(backend=cpu)
    try:
        stats, rows = engine.run_worker_owned_cpu(source, queries, config, workers=2,
            k=5, self_overlap_policy=policy, shape_backend="pipelined_field",
            shape_grain="library_atom", shape_width=8)
    finally:
        if owned:
            engine.close()
    assert stats.cpu_scan["mode"] == mode
    return stats, rows


def _block_inputs(built):
    source, _, _ = _inputs(*built, "combo")
    d = source.cpu_block_descriptors()[0]
    decoded = source.decode_cpu_block(d)
    coords, counts, vaa = decoded.result.coords, decoded.result.counts, decoded.result.vaa
    # This is the standalone CPU API's explicit in-memory metadata contract,
    # not an invented on-disk format or an assertion about corpus provenance.
    metadata = dict(schema=cpu._VAA_SCHEMA, frame="stored", units="angstrom^3",
        coordinate_units="angstrom", wire_quantization_step_angstrom=2**-8,
        algorithm="self_overlap_from_stored_quantized_xyz_v1", row_order="group_coordinate_row",
        n_rows=d.n_conf, shape_coordinates_sha256=cpu._shape_library_fingerprint(coords, counts),
        values_sha256=hashlib.sha256(vaa.tobytes()).hexdigest(), verified=True)
    return source, d, decoded, metadata


@pytest.mark.parametrize("mode", MODES)
def test_verified_stored_used_without_optional_verification_flag(built, mode):
    stats, _ = _run(built, mode)
    scan = stats.cpu_scan
    assert scan["stored_vaa_route"] == "verified_stored"
    assert scan["stored_vaa_verified_blocks"] == stats.n_blocks
    assert scan["library_shape_self_evaluations"] == 0
    assert scan["stored_vaa_frame_verified"]
    assert scan["stored_vaa_precision_qualified"] is False
    assert scan["library_color_self_evaluations"] == 0
    assert scan["color_self_sidecar_blocks"] == (stats.n_blocks if mode != "shape" else 0)


@pytest.mark.parametrize("mode", ["rescore", "combo"])
def test_absent_optional_color_self_counts_every_binding_evaluation(tmp_path, mode):
    available = build_inputs(tmp_path / "stored")
    absent = build_inputs(tmp_path / "absent", color_self=False)
    _, consumed = _run(available, mode)
    source, queries, config = _inputs(*absent, mode)
    engine = Engine(backend=cpu)
    try:
        for repeat in range(2):
            stats, recomputed = _run(absent, mode, engine=engine, inputs=(source, queries, config))
            scan = stats.cpu_scan
            assert scan["stored_vaa_route"] == "verified_stored"
            assert scan["color_self_sidecar_blocks"] == 0
            assert scan["color_self_binding_recompute_blocks"] == stats.n_blocks
            assert scan["library_color_self_evaluations"] == stats.n_conformers
            assert scan["library_color_self_binding_evaluations"] == stats.n_conformers
            assert scan["prepared_self_retained_bytes"] == 0
            assert scan["self_cache_hits"] == 0  # Block handles release after each claim.
            assert scan["library_self_evaluation_counter_source"] == (
                "native_and_python_counted_color_recomputed_at_binding" if mode == "combo"
                else "python_counted_color_recomputed_at_binding")
            for a, b in zip(consumed[0], recomputed[0]):
                np.testing.assert_array_equal(a, b)
    finally:
        engine.close()
        source.close()


@pytest.mark.parametrize("mode", ["rescore", "combo"])
def test_other_matrix_corpus_recomputes_color_self(tmp_path, mode):
    inputs = build_inputs(tmp_path / "other", noncanonical=True)
    corpus, _, _ = inputs
    with corpus.group(0) as group:
        assert "color_self" not in group._og.sections
    stats, _ = _run(inputs, mode)
    assert stats.cpu_scan["color_self_sidecar_blocks"] == 0
    assert stats.cpu_scan["library_color_self_binding_evaluations"] == stats.n_conformers
    assert stats.cpu_scan["library_color_self_evaluations"] == stats.n_conformers


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("route", ["verified_stored", "recompute"])
def test_stored_vaa_conversion_is_route_true(built, mode, route):
    stats, _ = _run(built, mode, policy="prepared" if route == "verified_stored" else "recompute")
    scan = stats.cpu_scan
    if route == "recompute":
        assert scan["stored_vaa_used"] is False
        assert scan["stored_vaa_conversion"] is None
    else:
        conversion = COMBO_CONVERSION if mode == "combo" else SHAPE_CONVERSION
        assert scan["stored_vaa_conversion"] == f"{conversion} on {route}"


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("target", ["vaa", "color_self"])
def test_tampered_section_is_refused_either_way(tmp_path, monkeypatch, active, target):
    import os
    monkeypatch.setattr(scores_module, "SELF_OVERLAP_MARGIN_TIGHT_ACTIVE", active)
    built = build_inputs(tmp_path / "corpus")
    corpus, _, _ = built
    with corpus.group(0) as group:
        path = group.path
        offset = group._og.sections[target]["offset"]
    stat = os.stat(path)
    with open(path, "r+b") as file:
        file.seek(offset)
        value = file.read(1)
        file.seek(offset)
        file.write(bytes([value[0] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(CorpusValidationError, match="group 0 block.*" + target):
        _run(built, "combo")


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_worker_sink_receives_verified_and_recompute_margin(built, monkeypatch, block_margins, active, mode):
    monkeypatch.setattr(scores_module, "SELF_OVERLAP_MARGIN_TIGHT_ACTIVE", active)
    for policy, route in (("prepared", "verified_stored"), ("recompute", "recompute")):
        block_margins.clear()
        source, query, config = _inputs(*built, mode)
        stats, _ = _run(built, mode, policy, inputs=(source, query, config))
        want = 1e-5 if active else 1e-3
        assert sorted(block_margins) == sorted((d.gid_base, want) for d in source.cpu_block_descriptors())
        assert stats.cpu_scan["self_overlap_route"] == route
        assert stats.cpu_scan["self_overlap_bound_margins"] == {route: want}
        source.close()


@pytest.fixture
def tight(monkeypatch):
    """The tight margin tier switched on for one test. The shipped
    default stays False (test_the_tight_tier_ships_inactive)."""
    monkeypatch.setattr(scores_module, "SELF_OVERLAP_MARGIN_TIGHT_ACTIVE", True)


@pytest.fixture
def block_margins(monkeypatch):
    """Spy on the margin every native block sink is built with."""
    seen = []
    ext = cpu.extension()
    original = ext.make_worker_local_topk_block_row_sink

    class Spy:
        def __getattr__(self, name):
            return getattr(ext, name)

        def make_worker_local_topk_block_row_sink(self, *args, **kwargs):
            sink = original(*args, **kwargs)
            seen.append((int(args[2]), sink.self_overlap_bound_margin))
            return sink
    monkeypatch.setattr(cpu, "extension", lambda: Spy())
    return seen


def test_the_tight_tier_ships_inactive():
    assert scores_module.SELF_OVERLAP_MARGIN_TIGHT_ACTIVE is False
    assert scores_module.SELF_OVERLAP_BOUND_MARGIN == 1e-3
    assert scores_module.SELF_OVERLAP_BOUND_MARGIN_TIGHT == 1e-5
    assert scores_module.SELF_OVERLAP_MARGIN_POLICY == {
        "verified_stored": (1e-3, 1e-5),
        "recompute": (1e-3, 1e-5),
    }
    for route in scores_module.SELF_OVERLAP_MARGIN_POLICY:
        assert scores_module.self_overlap_bound_margin(route) == 1e-3


def test_the_policy_table_once_active(tight):
    margin = scores_module.self_overlap_bound_margin
    assert margin("verified_stored") == 1e-5
    assert margin("recompute") == 1e-5
    for route in ("mixed", "verified", None, "retired"):
        with pytest.raises(ValueError, match="self-overlap route"):
            margin(route)


@pytest.mark.parametrize("active", [False, True])
def test_scan_classification(monkeypatch, active):
    monkeypatch.setattr(scores_module, "SELF_OVERLAP_MARGIN_TIGHT_ACTIVE", active)
    check = scores_module.check_scan_self_overlap_routes
    assert check({}) is None
    assert check({"verified_stored": 3}) == "verified_stored"
    assert check({"recompute": 1, "verified_stored": 0}) == "recompute"
    assert check({"verified_stored": 2, "recompute": 1}) == (
        "verified_stored+recompute")
    both_valid_routes = {"verified_stored": 2, "recompute": 1}
    assert scores_module.scan_self_overlap_route(both_valid_routes) == (
        "verified_stored+recompute")
    assert check(both_valid_routes) == "verified_stored+recompute"
    with pytest.raises(ValueError, match="unknown self-overlap route"):
        check({"stored": 1})


def test_the_python_bound_check_follows_the_route(tight):
    def row(overlap):
        r = np.zeros((1, scores_module.N_SCORE_COLUMNS), np.float32)
        r[0, scores_module.OVERLAP_VOLUME] = overlap
        r[0, scores_module.SELF_OVERLAP_QUERY] = 100.0
        r[0, scores_module.SELF_OVERLAP_LIBRARY] = 100.0
        return r
    between = row(100.0 * (1 + 1e-4))       # past 1e-5, inside 1e-3
    for route in ("verified_stored", "recompute"):
        with pytest.raises(scores_module.InconsistentSelfOverlapError):
            scores_module.check_self_overlap_bound(between, route=route)


def test_the_python_bound_check_names_its_route():
    """No default tier: a caller that omits the route is an error, not the
    loosest margin (which, once the tight tier is active, would silently
    admit rows a verified or recompute route refuses)."""
    row = np.zeros((1, scores_module.N_SCORE_COLUMNS), np.float32)
    with pytest.raises(TypeError, match="route"):
        scores_module.check_self_overlap_bound(row)
    with pytest.raises(TypeError, match="route"):
        scores_module.check_self_overlap_bound(row, None, "q")
    with pytest.raises(ValueError, match="unknown self-overlap route"):
        scores_module.check_self_overlap_bound(row, route="stored")


@pytest.mark.parametrize("margin,refused", [(0.5, False), (0.0, True)])
def test_the_native_sink_checks_rows_at_the_margin_it_is_given(margin, refused):
    """The margin reaches the native predicate, not just the Python side."""
    ext = cpu.extension()
    collector = ext.make_worker_local_topk_direct_row_sink(
        1, 1, 2, 0, "conformer", 2, False, 1 << 20)
    sink = ext.make_worker_local_topk_block_row_sink(
        collector, 0, 10, 2, self_overlap_bound_margin=margin)
    assert sink.self_overlap_bound_margin == margin
    values = np.zeros((2, 20), np.float32)
    values[:, scores_module.SELF_OVERLAP_QUERY] = 100.0
    values[:, scores_module.SELF_OVERLAP_LIBRARY] = 100.0
    values[1, scores_module.OVERLAP_VOLUME] = 120.0   # 20% past the bound
    rows = (values, np.zeros(2, np.int32), np.arange(2, dtype=np.int32))
    if refused:
        with pytest.raises(ValueError, match="conformer 11"):
            sink.ingest_scores(*rows)
    else:
        sink.ingest_scores(*rows)
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite and > -1"):
            ext.make_worker_local_topk_block_row_sink(
                collector, 0, 10, 2, self_overlap_bound_margin=bad)


def test_a_native_caller_that_names_no_margin_reads_it_from_scores(monkeypatch):
    ext = cpu.extension()
    collector = ext.make_worker_local_topk_direct_row_sink(
        1, 1, 2, 0, "conformer", 2, False, 1 << 20)
    assert ext.make_worker_local_topk_block_row_sink(
        collector, 0, 10, 2).self_overlap_bound_margin == 1e-3
    monkeypatch.setitem(scores_module.SELF_OVERLAP_MARGIN_POLICY,
                        scores_module.ROUTE_RECOMPUTE, (0.25, 0.25))
    assert ext.make_worker_local_topk_block_row_sink(
        collector, 0, 10, 2).self_overlap_bound_margin == 0.25




def test_the_margin_literal_has_one_definition():
    """Fails if a second copy of the self-overlap margin appears in the
    source, or a backend stops reading the one definition."""
    import re
    from pathlib import Path
    import roshambo3
    src = Path(roshambo3.__file__).parent
    texts = {path.relative_to(src).as_posix(): path.read_text()
             for path in src.rglob("*")
             if path.suffix in (".py", ".hpp", ".cpp", ".cu", ".cuh", ".metal", ".h")}
    literal = re.compile(r"\b\w*margin\w*\"?\)?\s*=\s*(?:1e-0*3|0\.001)\b",
                         re.IGNORECASE)
    copies = [(name, m.group(0)) for name, text in texts.items()
              for m in literal.finditer(text)]
    assert copies == [("scores.py", "SELF_OVERLAP_BOUND_MARGIN = 1e-3")], copies
    sink = texts["backend/_cpu_src/worker_local_topk_sink.hpp"]
    assert "kMargin" not in sink
    assert re.search(r"check_self_overlap_bound\([^)]*double margin\)", sink)
    assert "self_overlap_bound_margin(" in texts["backend/cuda.py"]
    assert "self_overlap_bound_margin(" in texts["backend/metal.py"]
    assert "SELF_OVERLAP_BOUND_MARGIN)" not in texts["backend/cuda.py"]
    assert "SELF_OVERLAP_BOUND_MARGIN)" not in texts["backend/metal.py"]
    assert 'py::arg("self_overlap_bound_margin") = 1e-3' not in texts[
        "backend/_cuda_src/interface.cpp"]


@pytest.mark.parametrize("active", [False, True])
def test_score_pairs_mapped_sink_margin_follows_its_self_overlaps(
        built, monkeypatch, active):
    """``cpu.score_pairs(worker_local_topk=...)`` sends each supported
    stored or recomputed self-overlap route's margin to the native sink."""
    monkeypatch.setattr(scores_module, "SELF_OVERLAP_MARGIN_TIGHT_ACTIVE", active)
    corpus, base, generator = built
    r, p = _matrices(generator)
    source, d, decoded, vaa_meta = _block_inputs(built)
    coords, counts, vaa = decoded.result.coords, decoded.result.counts, decoded.result.vaa
    n = int(d.n_conf)
    queries = from_rdkit([Chem.Mol(base)], color_generator=generator)
    qc = np.ascontiguousarray(queries.coords, np.float32)
    qt = np.ascontiguousarray(queries.types, np.int32)
    qn = np.ascontiguousarray(queries.n_real, np.int32)
    seen = []
    ext = cpu.extension()
    original = ext.make_worker_local_topk_row_sink

    class Spy:
        def __getattr__(self, name):
            return getattr(ext, name)

        def make_worker_local_topk_row_sink(self, *args, **kwargs):
            seen.append(kwargs["self_overlap_bound_margin"])
            return original(*args, **kwargs)
    monkeypatch.setattr(cpu, "extension", lambda: Spy())
    common = dict(query_coords=qc, query_types=qt, query_n_atoms=qn,
                  library_coords=coords, library_counts=counts,
                  pair_query_row=np.zeros(n, np.int32),
                  pair_library_row=np.arange(n, dtype=np.int32),
                  interaction_r=r, interaction_p=p, color=True,
                  color_mode=MODE_COMBO_OPTIMAL,
                  color_grain=RETAINED_COLOR_GRAIN, mixing=0.45, steps=2,
                  start_mode=1,
                  worker_local_topk=cpu.CPUWorkerLocalTopKConfig(k=3),
                  library_conformer_ids=np.arange(n, dtype=np.int64))
    tight_margin = 1e-5 if active else 1e-3
    cpu.score_pairs(**common)
    assert seen[-1] == tight_margin                      # recompute
    prepared = cpu.prepare_combo_self_overlaps(
        query_coords=qc, query_types=qt, query_n_atoms=qn,
        library_coords=coords, library_counts=counts,
        interaction_r=r, interaction_p=p,
        library_vaa=vaa, library_vaa_metadata=vaa_meta)
    assert prepared.library.library_shape_from_stored_volume
    cpu.score_pairs(prepared_self_overlaps=prepared, **common)
    assert seen[-1] == tight_margin                      # verified stored
    assert len(seen) == 2
    source.close()

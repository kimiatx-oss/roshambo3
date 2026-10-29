from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from bench import screening_regression as sr


def _fixture_panel(tmp_path):
    target = "ADA"
    npz, mp = sr.panel_paths(tmp_path / "panels", target)
    n = sr.PANEL_SIZE
    coords = np.zeros((n, 8, 3), np.int16)
    counts = np.ones((n, 1), np.uint8)
    gid = np.arange(n, dtype=np.int64)
    mol = np.arange(1000, 1000+n, dtype=np.int64)
    conf = np.zeros(n, np.int64)
    hac = np.tile(np.arange(1, 9, dtype=np.int16), 16)
    cat = np.full(n, "top16", dtype="U16")
    selection = np.linspace(1, 0, n, dtype=np.float32)
    qcoords = np.zeros((1, 1, 4), np.float32); qcoords[0, 0, 3] = 1
    qtypes = np.zeros((1, 1), np.int32)
    qn = np.ones(1, np.int32)
    ir = np.ones((1, 1), np.float32); ip = np.zeros((1, 1), np.float32)
    geometry = sr.array_sha(coords, counts, gid, mol, conf,
                            qcoords, qtypes, qn, ir, ip)
    sr._immutable_npz(npz, coords_i16=coords, counts=counts, gid=gid,
        mol_id=mol, conf_idx=conf, hac=hac, category=cat,
        selection_score=selection, query_coords=qcoords, query_types=qtypes,
        query_n_atoms=qn, interaction_r=ir, interaction_p=ip)
    meta = {"protocol": sr.PROTOCOL, "target": target, "query_hac": 20,
            "geometry_sha256": geometry, "npz_sha256": sr.file_sha(npz)}
    mp.write_bytes(sr._json_bytes(meta))
    return tmp_path / "panels", target, meta, mol, conf


def _make_candidate(directory, target, panel_meta, ids, scores, variant="clean"):
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{target}.shape.cpu"
    npz, mp = directory / f"{stem}.npz", directory / f"{stem}.json"
    sr._immutable_npz(npz, ids=np.asarray(ids, np.int64),
                      scores=np.asarray(scores, np.float32))
    meta = {"target": target, "lane": "shape", "backend": "cpu",
            "panel_sha256": panel_meta["npz_sha256"],
            "geometry_sha256": panel_meta["geometry_sha256"],
            "config": sr.config_for("shape", "cpu"), "variant": variant,
            "candidate_source_sha256": "candidate-source-is-not-a-cache-key",
            "npz_sha256": sr.file_sha(npz)}
    mp.write_bytes(sr._json_bytes(meta))
    return directory


def _import_refs(tmp_path, panel_dir, target, panel_meta, mol, conf, scores):
    from roshambo3.metrics import canonical_ids
    ids = canonical_ids(mol, conf)
    source_npz = tmp_path / "source.npz"
    sr._immutable_npz(source_npz, ids=ids, scores=np.asarray(scores, np.float32))
    source_json = tmp_path / "source.json"
    source_json.write_bytes(sr._json_bytes({
        "target": target, "lane": "shape", "backend": "cpu",
        "panel_sha256": panel_meta["npz_sha256"],
        "geometry_sha256": panel_meta["geometry_sha256"],
        "config": sr.config_for("shape", "cpu"),
        "candidate_source_sha256": "historical-reference-source-hash"}))
    fixed, current = tmp_path / "fixed", tmp_path / "current"
    sr.import_cache(target, lane="shape", backend="cpu", role="fixed",
        source_npz=source_npz, source_json=source_json,
        panel_dir=panel_dir, output_dir=fixed)
    sr.import_cache(target, lane="shape", backend="cpu", role="current",
        source_npz=source_npz, source_json=source_json,
        panel_dir=panel_dir, output_dir=current)
    return ids, fixed, current, source_npz, source_json



def test_frozen_config_names_complete_optimizer_and_combo_residency():
    cpu = sr.config_for("shape", "cpu")
    combo = sr.config_for("combo", "cuda")
    assert cpu["steps"] == 100 and cpu["start_mode"] == 1
    assert cpu["lr_q"] == 0.1 and cpu["lr_t"] == pytest.approx(0.2)
    assert cpu["shape_backend"] == "scalar_analytic"
    assert combo["color_residency"] == "GLOBAL"
    assert "coord_residency" not in combo or combo["coord_residency"] == "SHARED"
    assert combo["color_placement"] == "IN_GRADIENT"
    assert combo["library_color_offset"] == -1

def test_selector_is_deterministic_and_obeys_categories():
    mol = np.repeat(np.arange(100, dtype=np.int64), 2)
    conf = np.tile(np.array([0, 1], np.int64), 100)
    gid = np.arange(200, dtype=np.int64)
    scores = np.repeat(np.linspace(1.0, 0.01, 100), 2) + np.tile([0.001, 0.0], 100)
    a = sr.select_panel_rows("PUR2", gid, mol, conf, scores)
    b = sr.select_panel_rows("PUR2", gid, mol, conf, scores)
    assert a == b
    assert len(a) == 128
    assert [sum(x["category"] == c for x in a)
            for c in ("top16", "cutoff17_48", "hash32", "extra48")] == [16, 32, 32, 48]
    assert len({x["gid"] for x in a}) == 128
    assert len({x["mol_id"] for x in a[:80]}) == 80


def test_selector_refuses_duplicate_pair_and_insufficient_extra_rows():
    mol = np.repeat(np.arange(100, dtype=np.int64), 2)
    conf = np.tile(np.array([0, 1], np.int64), 100)
    gid = np.arange(200, dtype=np.int64)
    scores = np.linspace(1, 0, 200)
    conf[1] = conf[0]
    with pytest.raises(ValueError, match="duplicate"):
        sr.select_panel_rows("PUR2", gid, mol, conf, scores)
    conf = np.zeros(100, np.int64)
    with pytest.raises(ValueError, match="extra conformers"):
        sr.select_panel_rows("PUR2", np.arange(100), np.arange(100), conf,
                             np.linspace(1, 0, 100))


def test_reference_import_is_immutable_and_source_hash_is_provenance(tmp_path):
    panel_dir, target, pm, mol, conf = _fixture_panel(tmp_path)
    values = np.r_[np.linspace(0.999, 0.990, 8), np.linspace(0.5, 0.0, 120)]
    ids, fixed, current, source_npz, source_json = _import_refs(
        tmp_path, panel_dir, target, pm, mol, conf, values)
    fmeta, fids, fscores = sr._load_scores(fixed, target, "shape", "cpu")
    cmeta, cids, cscores = sr._load_scores(current, target, "shape", "cpu")
    assert np.array_equal(fids, cids)
    assert np.array_equal(fscores, cscores)
    assert fmeta["source_sha256_provenance"] == "historical-reference-source-hash"
    # A changed candidate source is valid; immutable panel/config binding is the key.
    candidate = _make_candidate(tmp_path / "candidate", target, pm, ids, values)
    result = sr.evaluate(lane="shape", backend="cpu", panel_dir=panel_dir,
        fixed_dir=fixed, current_dir=current, candidate_dir=candidate,
        targets=[target])
    assert result["detector_pass"]
    assert result["per_query"][0]["fixed_equals_current"]
    with pytest.raises(FileExistsError, match="overwrite"):
        sr.import_cache(target, lane="shape", backend="cpu", role="current",
            source_npz=source_npz, source_json=source_json,
            panel_dir=panel_dir, output_dir=current)


def test_stable_row_reorder_passes_but_cyclic_identity_assignment_is_caught(tmp_path):
    panel_dir, target, pm, mol, conf = _fixture_panel(tmp_path)
    values = np.r_[np.linspace(0.999, 0.990, 8), np.linspace(0.5, 0.0, 120)]
    ids, fixed, current, _, _ = _import_refs(tmp_path, panel_dir, target, pm, mol, conf, values)
    perm = np.roll(np.arange(128), 37)
    stable = _make_candidate(tmp_path / "stable", target, pm, ids[perm], values[perm])
    clean = sr.evaluate(lane="shape", backend="cpu", panel_dir=panel_dir,
        fixed_dir=fixed, current_dir=current, candidate_dir=stable, targets=[target])
    assert clean["detector_pass"]
    # The identity labels rotate while score rows stay put: this is not a stable reorder.
    cyclic = _make_candidate(tmp_path / "cyclic", target, pm, np.roll(ids, -1), values)
    caught = sr.evaluate(lane="shape", backend="cpu", panel_dir=panel_dir,
        fixed_dir=fixed, current_dir=current, candidate_dir=cyclic, targets=[target])
    assert not caught["detector_pass"]
    assert caught["comparisons"]["fixed"]["levels"]["conformer"]["mean"] < 0.95


def test_score_offset_and_missing_rows_fail_frozen_checks(tmp_path):
    panel_dir, target, pm, mol, conf = _fixture_panel(tmp_path)
    values = np.r_[np.linspace(0.999, 0.990, 8), np.linspace(0.5, 0.0, 120)]
    ids, fixed, current, _, _ = _import_refs(tmp_path, panel_dir, target, pm, mol, conf, values)
    offset = _make_candidate(tmp_path / "offset", target, pm, ids, values + 0.04)
    report = sr.evaluate(lane="shape", backend="cpu", panel_dir=panel_dir,
        fixed_dir=fixed, current_dir=current, candidate_dir=offset, targets=[target])
    assert not report["detector_pass"]
    assert not report["comparisons"]["fixed"]["signed_score_movement"]["global"]["abs_mean_bias_gate"]
    keep = np.ones(128, bool); keep[6::7] = False
    missing = _make_candidate(tmp_path / "missing", target, pm, ids[keep], values[keep])
    report = sr.evaluate(lane="shape", backend="cpu", panel_dir=panel_dir,
        fixed_dir=fixed, current_dir=current, candidate_dir=missing, targets=[target])
    assert not report["detector_pass"]
    assert not report["identity_coverage_all_queries"]
    assert report["per_query"][0]["candidate_n"] < 128
    assert report["per_query"][0]["selection_metrics_status"] == "unavailable_identity_coverage"
    assert report["per_query"][0]["fixed"]["levels"]["conformer"]["status"] == "unavailable"


def test_empty_candidate_rows_are_unavailable_not_low_scores(tmp_path):
    panel_dir, target, pm, mol, conf = _fixture_panel(tmp_path)
    values = np.r_[np.linspace(0.999, 0.990, 8), np.linspace(0.5, 0.0, 120)]
    ids, fixed, current, _, _ = _import_refs(tmp_path, panel_dir, target, pm, mol, conf, values)
    empty = _make_candidate(tmp_path / "empty", target, pm,
                            np.empty(0, np.int64), np.empty(0, np.float32))
    report = sr.evaluate(lane="shape", backend="cpu", panel_dir=panel_dir,
        fixed_dir=fixed, current_dir=current, candidate_dir=empty, targets=[target])
    row = report["per_query"][0]
    assert row["candidate_n"] == 0
    assert row["missing_candidate_ids"] == ids.tolist()
    assert row["selection_metrics_status"] == "unavailable_identity_coverage"
    assert report["status"] == "complete"
    assert not report["detector_pass"]
    assert report["comparisons"]["fixed"]["levels"]["conformer"]["status"] == "unavailable"


def test_candidate_fault_config_cannot_bypass_frozen_config_validation(tmp_path):
    panel_dir, target, pm, mol, conf = _fixture_panel(tmp_path)
    values = np.r_[np.linspace(0.999, 0.990, 8), np.linspace(0.5, 0.0, 120)]
    ids, fixed, current, _, _ = _import_refs(tmp_path, panel_dir, target, pm, mol, conf, values)
    candidate = _make_candidate(tmp_path / "zero_steps", target, pm, ids, values,
                               variant="zero_steps")
    meta_path = candidate / f"{target}.shape.cpu.json"
    meta = json.loads(meta_path.read_text())
    meta["config"] = dict(meta["config"], steps=0)
    meta_path.write_bytes(sr._json_bytes(meta))
    with pytest.raises(ValueError, match="routine config mismatch"):
        sr.evaluate(lane="shape", backend="cpu", panel_dir=panel_dir,
            fixed_dir=fixed, current_dir=current, candidate_dir=candidate,
            targets=[target])


def test_candidate_must_pass_against_both_independent_references(tmp_path):
    panel_dir, target, pm, mol, conf = _fixture_panel(tmp_path)
    values = np.r_[np.linspace(0.90, 0.89, 8), np.linspace(0.5, 0.0, 120)]
    ids, fixed, _, _, source_json = _import_refs(
        tmp_path, panel_dir, target, pm, mol, conf, values)
    shifted_npz = tmp_path / "shifted-current-source.npz"
    sr._immutable_npz(shifted_npz, ids=ids, scores=np.asarray(values + 0.015, np.float32))
    current = tmp_path / "shifted-current"
    sr.import_cache(target, lane="shape", backend="cpu", role="current",
        source_npz=shifted_npz, source_json=source_json,
        panel_dir=panel_dir, output_dir=current)
    candidate = _make_candidate(tmp_path / "candidate-plus-030", target, pm, ids,
                                values + 0.03)
    report = sr.evaluate(lane="shape", backend="cpu", panel_dir=panel_dir,
        fixed_dir=fixed, current_dir=current, candidate_dir=candidate,
        targets=[target])
    fixed_bias = report["comparisons"]["fixed"]["signed_score_movement"]["global"]
    current_bias = report["comparisons"]["current"]["signed_score_movement"]["global"]
    assert not fixed_bias["abs_mean_bias_gate"]
    assert current_bias["abs_mean_bias_gate"]
    assert not report["detector_pass"]


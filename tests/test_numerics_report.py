"""Coverage must never infer execution or certification from absent evidence."""
from dataclasses import asdict
import json

import pytest

from bench.numerics_report import (Cell, CoverageReport, legacy_availability, main,
                                    required_matrix, required_plants)


def receipt(**kwargs):
    return CoverageReport(source_stamp="source-a", panel_stamp="seeded-panel-v1", **kwargs)


def record(report, cell, status="passed", **kwargs):
    report.record(**asdict(cell), status=status, reason="live assertion", build_stamp={"compiler": "test"},
                  arithmetic_stamp={"exp": "declared"}, evidence={"qualification": {
                      "source_stamp": report.source_stamp, "panel_stamp": report.panel_stamp,
                      "protocol_stamp": report.protocol_stamp, "clean_pass": True,
                      "plants": {plant: {"detected": True, "detector": cell.check,
                          "mutant_source_sha256": "a"*64, "control_source_sha256": "b"*64,
                          "artifact": "synthetic-test-receipt",
                          "rows_pass": True, "mean_pass": False,
                          "mean_model_derivation": "synthetic aggregate model"} for plant in required_plants(cell)}}}, **kwargs)


def test_missing_rows_are_not_run_and_matrix_is_hardware_specific():
    report = receipt()
    assert {row["status"] for row in report.rows()} == {"not_run"}
    assert not report.certified
    assert {cell.hardware for cell in report.required if cell.backend == "cpu"} == {"x86_64", "aarch64"}
    record(report, report.required[0])
    assert not report.certified
    assert len(report.rows()) == len(report.required)


def test_every_required_row_must_pass_and_stamps_must_be_identified():
    report = receipt()
    for cell in report.required:
        record(report, cell)
    assert report.certified
    report.panel_stamp = "unidentified"
    assert not report.certified
    report.panel_stamp = "panel"
    record(report, report.required[0], "not_run")
    assert not report.certified
    record(report, report.required[0], "failed")
    assert next(row for row in report.rows() if row["check"] == report.required[0].check
                and row["path"] == report.required[0].path and row["channel"] == report.required[0].channel
                and row["hardware"] == report.required[0].hardware)["status"] == "failed"


def test_missing_path_arithmetic_stamp_cannot_certify():
    report = receipt()
    report.record("cpu", "scalar", "fixed_pose", "passed", "assertions passed")
    observed = next(row for row in report.rows() if row["observations"])
    assert observed["status"] == "not_run"
    assert "incomplete" in observed["reasons"][0]


def test_future_profile_requires_metal_texture32():
    cells = required_matrix("cpu-cuda-metal")
    assert set(required_matrix("cpu-cuda")) < set(cells)
    assert Cell("metal", "texture32", "optimizer", "color", "apple") in cells


def test_merge_disjoint_hardware_receipts_and_round_trip():
    left, right = receipt(), receipt()
    for cell in left.required:
        record(left if cell.backend == "cpu" else right, cell)
    left.merge(CoverageReport.from_dict(right.as_dict()))
    assert left.certified
    assert CoverageReport.from_dict(json.loads(json.dumps(left.as_dict()))).certified


@pytest.mark.parametrize("field,value", [("source_stamp", "other"), ("panel_stamp", "other"),
    ("protocol_stamp", "other"), ("source_stamp", "unidentified"), ("source_stamp", "source-a-dirty")])
def test_incompatible_or_unidentified_receipts_refused(field, value):
    left, right = receipt(), receipt()
    setattr(right, field, value)
    if value.endswith("-dirty"):
        setattr(left, field, value)
    with pytest.raises(ValueError):
        left.merge(right)


def test_deleted_matrix_row_is_rejected():
    payload = receipt().as_dict()
    payload["required_matrix"].pop()
    with pytest.raises(ValueError, match="matrix"):
        CoverageReport.from_dict(payload)


def test_prerequisite_presence_never_implies_execution(tmp_path):
    (tmp_path / "results/engine-baseline").mkdir(parents=True)
    report = receipt()
    legacy_availability(report, root=tmp_path, environ={"ROSHAMBO2_CHECKOUT": str(tmp_path)})
    legacy = [row for row in report.rows() if row["backend"] == "legacy"]
    assert len(legacy) == 3
    assert all(row["status"] == "not_run" for row in legacy)
    assert "configured" in next(row for row in legacy if row["path"] == "roshambo2_xcheck")["reasons"][0]


def test_cli_emits_report_but_certification_fails_for_missing_hardware(tmp_path, capsys):
    assert main(["--output", str(tmp_path), "--certify"]) == 1
    assert "NOT CERTIFIED" in capsys.readouterr().out
    assert not json.loads((tmp_path / "coverage.json").read_text())["certified"]
    assert "not_run" in (tmp_path / "coverage.md").read_text()


def test_cli_rejects_stale_receipt(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(receipt().as_dict()))
    with pytest.raises(SystemExit):
        main([str(path), "--output", str(tmp_path), "--source-stamp", "newer-source"])


def test_passing_execution_without_fault_evidence_never_certifies():
    report = receipt()
    for cell in report.required:
        report.record(**asdict(cell), status="passed", reason="assertions passed",
                      build_stamp={"x": 1}, arithmetic_stamp={"x": 1})
    assert not report.certified
    assert report.as_dict()["qualification_missing"]


def test_same_cell_with_changed_arithmetic_is_not_mergeable():
    left, right = receipt(), receipt()
    cell = left.required[0]
    record(left, cell)
    right.record(**asdict(cell), status="passed", reason="other arithmetic",
                 build_stamp={"x": 1}, arithmetic_stamp={"x": 999})
    with pytest.raises(ValueError, match="incompatible arithmetic"):
        left.merge(right)


def test_schedule_mutation_uses_truthful_settings_identity():
    receipt = {"mutant_settings_sha256": "a"*64, "control_settings_sha256": "b"*64}
    assert CoverageReport._distinct_mutation(receipt, "step_schedule")
    assert not CoverageReport._distinct_mutation(receipt, "constant")
    receipt["mutant_settings_sha256"] = receipt["control_settings_sha256"]
    assert not CoverageReport._distinct_mutation(receipt, "step_schedule")


def test_metal_analytic_unreachable_checks_are_explicit_not_applicable():
    from bench.numerics_report import not_applicable_matrix
    report = receipt(profile="cpu-cuda-metal")
    unreachable = not_applicable_matrix(report.profile)
    assert len(unreachable) == 4
    assert {r["check"] for r in unreachable} == {"optimizer", "end_to_end"}
    assert all(r["status"] == "not_applicable" and r["path"] == "analytic" for r in unreachable)
    assert "not_applicable" in report.markdown()
    assert report.as_dict()["not_applicable"] == unreachable
    assert len([c for c in report.required if c.backend == "metal"]) == 36
    assert all(Cell("metal", "analytic", c, ch, "apple") in report.required
               for c in ("fixed_pose", "frame", "determinism") for ch in ("shape", "color"))
    assert all(Cell("metal", p, c, ch, "apple") in report.required
               for p in ("manual", "texture16", "texture32")
               for c in ("fixed_pose", "frame", "determinism", "optimizer", "end_to_end")
               for ch in ("shape", "color"))
    payload = report.as_dict()
    payload["not_applicable"] = []
    with pytest.raises(ValueError, match="not-applicable inventory"):
        CoverageReport.from_dict(payload)


def test_row_implied_mean_cannot_qualify_systematic_bias():
    report = receipt()
    cell = next(c for c in report.required if c.check == "fixed_pose")
    record(report, cell)
    row = next(r for r in report.rows() if r["observations"])
    plant = row["observations"][0]["evidence"]["qualification"]["plants"]["systematic_bias"]
    assert not report.qualification_missing(row)
    del plant["mean_model_derivation"]
    assert "systematic_bias" in report.qualification_missing(row)
    plant["mean_model_derivation"] = "test model"
    plant["rows_pass"] = False
    assert "systematic_bias" in report.qualification_missing(row)

"""Portable numerical coverage receipts; execution and certification are separate.

A receipt reports observations, never infers a pass from backend availability.
Use ``python -m bench.numerics_report --help`` to combine compatible receipts.
The declared matrix is intentionally independent of pytest collection: a deleted,
newly skipped, deselected, or unimplemented test leaves a visible not-run row.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import subprocess

PROTOCOL = "oracle-numerics-v1"
CHECKS = ("fixed_pose", "frame", "determinism", "optimizer", "end_to_end")
CHANNELS = ("shape", "color")
PATHS = {"cpu": ("scalar", "vector", "field"),
         "cuda": ("analytic", "field"),
         "metal": ("analytic", "manual", "texture16", "texture32")}
PROFILES = ("cpu-cuda", "cpu-cuda-metal")
SEVERITY = {"passed": 0, "not_run": 1, "failed": 2}

# Execution is not qualification: every detector needs a clean control and a
# real planted-defect failure before its cell can certify a release gate.
REQUIRED_PLANTS = {
    "fixed_pose": ("constant", "systematic_bias"),
    "frame": ("frame_split",),
    "determinism": ("reduction_order",),
    "optimizer": ("step_schedule",),
    "end_to_end": ("pose_inverse", "translation_scale", "row_association"),
}


def required_plants(cell):
    plants = list(REQUIRED_PLANTS.get(cell.check, ()))
    if cell.check == "fixed_pose" and cell.path in ("field", "manual", "texture16", "texture32"):
        plants.append("sampler_precision")
    if cell.channel == "color" and cell.check == "end_to_end":
        plants.extend(("color_score", "color_gradient"))
    return plants


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True, order=True)
class Cell:
    backend: str
    path: str
    check: str
    channel: str = "shape"
    hardware: str = ""


def hardware_for(backend: str) -> str:
    machine = platform.machine().lower()
    if backend == "cpu":
        return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    return {"cuda": "cuda", "metal": "apple"}.get(backend, "external")


def required_matrix(profile: str) -> tuple[Cell, ...]:
    if profile not in PROFILES:
        raise ValueError(f"unknown certification profile: {profile}")
    backends = ("cpu", "cuda", "metal") if profile == "cpu-cuda-metal" else ("cpu", "cuda")
    return tuple(Cell(backend, path, check, channel, hardware)
                 for backend in backends
                 for hardware in (("x86_64", "aarch64") if backend == "cpu"
                                  else (hardware_for(backend),))
                 for path in PATHS[backend] for check in CHECKS for channel in CHANNELS
                 if not (backend == "metal" and path == "analytic"
                         and check in ("optimizer", "end_to_end")))


def not_applicable_matrix(profile):
    """Unsupported API combinations, independent of this host's hardware.

    Metal supports PoseSearch.FIELD only. Its analytic path is poses= final
    scoring, so it has no independent optimizer or public screen endpoint.
    All three field samplers remain required, including unsupported-on-host
    texture32: that is missing hardware coverage, never N/A.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown certification profile: {profile}")
    if profile != "cpu-cuda-metal":
        return []
    return [dict(**asdict(Cell("metal", "analytic", check, channel, "apple")),
                 status="not_applicable", reason="Metal analytic is poses= final scoring; API supports FIELD optimization only")
            for check in ("optimizer", "end_to_end") for channel in CHANNELS]


def source_stamp(root: Path | None = None) -> str:
    """Include dirty state so a dirty tree cannot masquerade as its clean HEAD."""
    root = root or Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unidentified"
    return head + ("-dirty" if dirty else "")


class CoverageReport:
    def __init__(self, *, profile="cpu-cuda", source_stamp="unidentified",
                 panel_stamp="unidentified", protocol_stamp=PROTOCOL):
        self.profile = profile
        self.required = required_matrix(profile)
        self.source_stamp = source_stamp
        self.panel_stamp = panel_stamp
        self.protocol_stamp = protocol_stamp
        self.observations: dict[Cell, list[dict]] = {}

    def record(self, backend, path, check, status, reason, *, channel="shape",
               hardware=None, build_stamp=None, arithmetic_stamp=None, evidence=None):
        if status not in SEVERITY:
            raise ValueError(f"invalid status: {status}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("every observation needs a reason")
        if backend in PATHS:
            if path not in PATHS[backend] or check not in CHECKS or channel not in CHANNELS:
                raise ValueError(f"undeclared numerical cell: {backend}/{path}/{check}/{channel}")
        elif backend != "legacy":
            raise ValueError(f"unknown backend: {backend}")
        cell = Cell(backend, path, check, channel, hardware or hardware_for(backend))
        # A successful assertion without provenance is useful execution evidence,
        # but cannot certify the declared arithmetic on a named build.
        if status == "passed" and (not build_stamp or not arithmetic_stamp):
            status = "not_run"
            reason = "assertions passed, but build/arithmetic stamps are incomplete: " + reason
        observation = dict(status=status, reason=reason, build_stamp=dict(build_stamp or {}),
                           arithmetic_stamp=dict(arithmetic_stamp or {}), evidence=dict(evidence or {}))
        # Fail early on nonportable payloads rather than corrupting the CI artifact.
        json.dumps(observation, allow_nan=False)
        for previous in self.observations.get(cell, []):
            if (previous["status"] == status == "passed"
                    and previous["arithmetic_stamp"] != observation["arithmetic_stamp"]):
                raise ValueError("incompatible arithmetic declarations for the same coverage cell")
        self.observations.setdefault(cell, []).append(observation)

    def rows(self):
        result = []
        for cell in sorted(set(self.required) | set(self.observations)):
            observations = self.observations.get(cell, [])
            status = max((o["status"] for o in observations), key=SEVERITY.get, default="not_run")
            result.append(dict(**asdict(cell), required=cell in self.required, status=status,
                               reasons=[o["reason"] for o in observations] or
                               ["no execution receipt for this required cell"],
                               observations=observations))
        return result

    @staticmethod
    def _distinct_mutation(receipt, plant):
        # A learning-rate plant changes declared settings, not native source.
        # Keep that distinction rather than inventing a changed source hash.
        kinds = ("source", "settings") if plant == "step_schedule" else ("source",)
        return any(_sha256(receipt.get("mutant_" + kind + "_sha256"))
                   and _sha256(receipt.get("control_" + kind + "_sha256"))
                   and receipt["mutant_" + kind + "_sha256"] != receipt["control_" + kind + "_sha256"]
                   for kind in kinds)

    def qualification_missing(self, row):
        cell = Cell(**{key: row[key] for key in asdict(Cell("", "", ""))})
        covered = set()
        for observation in row["observations"]:
            if observation["status"] != "passed":
                continue
            evidence = observation["evidence"].get("qualification", {})
            if not isinstance(evidence, dict):
                continue
            if any(evidence.get(k) != getattr(self, k) for k in
                   ("source_stamp", "panel_stamp", "protocol_stamp")):
                continue
            if evidence.get("clean_pass") is not True:
                continue
            for plant, receipt in evidence.get("plants", {}).items():
                if (isinstance(receipt, dict) and receipt.get("detected") is True
                        and receipt.get("detector") == cell.check
                        and self._distinct_mutation(receipt, plant)
                        and receipt.get("artifact")):
                    if plant == "systematic_bias" and not (
                            receipt.get("rows_pass") is True
                            and receipt.get("mean_pass") is False
                            and receipt.get("mean_model_derivation")):
                        continue
                    covered.add(plant)
        missing = set(required_plants(cell)) - covered
        # Execution can pass while a conservative source-geometry uniqueness
        # bound does not transfer. That is visible unqualified coverage, never
        # certification of the prepared fixture or observed optimizer failure.
        if cell.check == "end_to_end" and any(
                observation["evidence"].get("fixture_qualification", {}).get(
                    "all_production_qualified") is False
                for observation in row["observations"]):
            missing.add("prepared_geometry_uniqueness")
        return sorted(missing)

    @property
    def certified(self):
        return (self.source_stamp != "unidentified" and
                not self.source_stamp.endswith("-dirty") and
                self.panel_stamp != "unidentified" and self.protocol_stamp == PROTOCOL and
                all(row["status"] == "passed" and not self.qualification_missing(row)
                    for row in self.rows() if row["required"]))

    def as_dict(self):
        return dict(schema_version=1, protocol_stamp=self.protocol_stamp,
                    source_stamp=self.source_stamp, panel_stamp=self.panel_stamp,
                    profile=self.profile, certified=self.certified,
                    required_matrix=[asdict(cell) for cell in self.required], rows=self.rows(),
                    not_applicable=not_applicable_matrix(self.profile),
                    qualification_missing=[dict(**{k: row[k] for k in asdict(Cell("", "", ""))},
                        plants=self.qualification_missing(row)) for row in self.rows()
                        if row["required"] and self.qualification_missing(row)])

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported coverage schema")
        report = cls(**{key: payload[key] for key in
                        ("profile", "source_stamp", "panel_stamp", "protocol_stamp")})
        if payload.get("required_matrix") != [asdict(cell) for cell in report.required]:
            raise ValueError("receipt required matrix differs from declared protocol")
        if payload.get("not_applicable", []) != not_applicable_matrix(report.profile):
            raise ValueError("receipt not-applicable inventory differs from declared protocol")
        for row in payload["rows"]:
            for obs in row["observations"]:
                report.record(**{key: row[key] for key in asdict(Cell("", "", ""))}, **obs)
        return report

    def merge(self, other):
        for key in ("source_stamp", "panel_stamp", "protocol_stamp", "profile"):
            value = getattr(self, key)
            if value != getattr(other, key) or value == "unidentified":
                raise ValueError(f"incompatible or unidentified receipt {key}")
        if self.source_stamp.endswith("-dirty"):
            raise ValueError("dirty source receipts cannot be merged for certification")
        for cell, observations in other.observations.items():
            for observation in observations:
                self.record(**asdict(cell), **observation)
        return self

    def markdown(self):
        rows = self.rows()
        counts = {status: sum(row["status"] == status for row in rows) for status in SEVERITY}
        def escape(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        lines = [f"Numerics coverage ({self.profile}): " +
                 ("CERTIFIED" if self.certified else "INCOMPLETE / NOT CERTIFIED"),
                 f"Source: `{self.source_stamp}`; protocol: `{self.protocol_stamp}`; "
                 f"panel: `{self.panel_stamp}`.",
                 "; ".join(f"{status}: {count}" for status, count in counts.items()), "",
                 "| Backend | Hardware | Path | Channel | Check | Required | Status | Reason |",
                 "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append("| " + " | ".join(escape(v) for v in
                         (row["backend"], row["hardware"], row["path"], row["channel"],
                          row["check"], "yes" if row["required"] else "no", row["status"],
                          "; ".join(row["reasons"]))) + " |")
        for row in not_applicable_matrix(self.profile):
            lines.append("| " + " | ".join(escape(v) for v in
                         (row["backend"], row["hardware"], row["path"], row["channel"],
                          row["check"], "no", row["status"], row["reason"])) + " |")
        missing = [(row, self.qualification_missing(row)) for row in rows if row["required"]]
        missing = [(row, plants) for row, plants in missing if plants]
        if missing:
            lines.extend(["", "Fault qualification still required:", ""])
            for row, plants in missing:
                lines.append("- " + "/".join(str(row[k]) for k in
                    ("backend", "hardware", "path", "channel", "check")) + ": " + ", ".join(plants))
        return "\n".join(lines) + "\n"

    def write(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "coverage.json").write_text(json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n")
        (directory / "coverage.md").write_text(self.markdown())


def legacy_availability(report, *, root=None, environ=None):
    """Prerequisite visibility is never execution evidence."""
    import os
    root = root or Path(__file__).resolve().parents[1]
    environ = os.environ if environ is None else environ
    checkout = environ.get("ROSHAMBO2_CHECKOUT")
    corpora = environ.get("ROSHAMBO3_CORPORA")
    details = {
        "golden_field": "not selected or not executed; golden fixture " +
                        ("present" if (root / "tests/data/golden_scores.npz").exists() else "absent"),
        "roshambo2_xcheck": "not executed; ROSHAMBO2 checkout " +
                          (f"configured at {checkout}" if checkout else "not configured (ROSHAMBO2_CHECKOUT)"),
        "engine_baseline": "not executed; corpora " +
                           (f"configured at {corpora}" if corpora else "not configured (ROSHAMBO3_CORPORA)") +
                           "; baseline directory " +
                           ("present" if (root / "results/engine-baseline").is_dir() else "absent"),
    }
    for path, reason in details.items():
        report.record("legacy", path, "availability", "not_run", reason)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="*", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="cpu-cuda")
    parser.add_argument("--source-stamp", default=None)
    parser.add_argument("--panel-stamp", default="unidentified")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--certify", action="store_true", help="fail if ANY required cell did not pass")
    args = parser.parse_args(argv)
    if args.receipts:
        report = CoverageReport.from_dict(json.loads(args.receipts[0].read_text()))
        expected_source = args.source_stamp or source_stamp()
        if report.source_stamp != expected_source:
            parser.error("stale receipt: source stamp does not match requested/current source")
        if args.panel_stamp != "unidentified" and report.panel_stamp != args.panel_stamp:
            parser.error("receipt panel does not match requested panel")
        if report.profile != args.profile:
            parser.error("receipt does not match requested profile")
        for path in args.receipts[1:]:
            report.merge(CoverageReport.from_dict(json.loads(path.read_text())))
    else:
        report = CoverageReport(profile=args.profile, source_stamp=args.source_stamp or source_stamp(),
                                panel_stamp=args.panel_stamp)
        legacy_availability(report)
    report.write(args.output)
    print(report.markdown())
    return int(args.certify and not report.certified)


if __name__ == "__main__":
    raise SystemExit(main())

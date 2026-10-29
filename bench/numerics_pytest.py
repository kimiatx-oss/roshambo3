"""Pytest bridge for the explicit numerical certification matrix.

Mark each executing arm with ``numerics_coverage(backend=..., path=...,
check=..., channel=...)``. Attach the actual path stamps through the
``numerics_coverage`` fixture's ``stamp(build=..., arithmetic=...)`` method.
Unmarked tests cannot accidentally certify a numerical arm.
"""
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from bench.numerics_report import CoverageReport, PROFILES, legacy_availability, source_stamp


def pytest_addoption(parser):
    group = parser.getgroup("numerics coverage")
    group.addoption("--numerics-report-dir", type=Path, default=None,
                    help="write coverage.json and coverage.md here")
    group.addoption("--numerics-outcomes", type=Path, default=None,
                    help="portable per-test observations for combining suite lanes")
    group.addoption("--numerics-profile", choices=PROFILES, default="cpu-cuda")
    group.addoption("--optimizer-model-qualification", action="store_true",
                    help="run the unchanged finite-perturbation optimizer quality gates")
    group.addoption("--numerics-panel-stamp", default="unidentified",
                    help="versioned seeded panel identifier shared by hardware receipts")


def pytest_configure(config):
    config.addinivalue_line("markers", "numerics_coverage(backend, path, check, channel='shape', hardware=None): numerical coverage cell")
    config.addinivalue_line("markers", "optimizer_model_qualification: finite perturbation-model quality qualification; explicit lane")
    identity = getattr(config, "workerinput", {}).get("numerics_identity")
    config._numerics_coverage = CoverageReport(**(identity or dict(
        profile=config.getoption("--numerics-profile"), source_stamp=source_stamp(),
        panel_stamp=config.getoption("--numerics-panel-stamp"))))
    legacy_availability(config._numerics_coverage)
    config._numerics_outcomes = {}
    config._numerics_deferred = {}
    config._numerics_worker_errors = []


class Observation:
    def __init__(self, item):
        self.item = item

    def stamp(self, *, build, arithmetic, evidence=None):
        self.item._numerics_stamps = dict(build_stamp=build, arithmetic_stamp=arithmetic,
                                         evidence=evidence or {})


@pytest.fixture
def numerics_coverage(request):
    return Observation(request.node)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--optimizer-model-qualification"):
        deferred = [item for item in items if item.get_closest_marker("optimizer_model_qualification")]
        for item in deferred:
            config._numerics_deferred[item.nodeid] = dict(
                markers=[dict(m.kwargs) for m in item.iter_markers("numerics_coverage")],
                status="not_run",
                reason="finite optimizer perturbation model remains unqualified; explicit --optimizer-model-qualification lane not requested",
                stamps={"evidence": {"qualification_lane": "optimizer_model_qualification"}})
        if deferred:
            config.hook.pytest_deselected(items=deferred)
            items[:] = [item for item in items if item not in deferred]
    for item in items:
        # Preserve the FIELD arm's existing arithmetic and architecture gate.
        # Merely collecting the test never counts as executing it.
        if (item.path.name == "test_golden.py" and
                item.originalname == "test_scores_are_bit_identical"):
            item.add_marker(pytest.mark.numerics_coverage(
                backend="legacy", path="golden_field", check="availability"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()
    markers = list(item.iter_markers("numerics_coverage"))
    if not markers:
        return
    state = item.config._numerics_outcomes.setdefault(item.nodeid,
        dict(markers=[dict(marker.kwargs) for marker in markers], status="not_run",
             reason="test call did not execute", stamps={}))
    if result.failed:
        state.update(status="failed", reason=f"{result.when} failed: {result.longrepr}")
    elif result.skipped and state["status"] != "failed":
        detail = result.longrepr
        reason = detail[2] if isinstance(detail, tuple) else str(detail)
        state.update(status="not_run", reason=f"{result.when}: {reason}")
    elif result.when == "call" and result.passed and state["status"] != "failed":
        if hasattr(result, "wasxfail"):
            state.update(status="not_run", reason="unexpected pass under xfail; not certification evidence")
        else:
            state.update(status="passed", reason=f"assertions passed: {item.nodeid}")
    state["stamps"] = getattr(item, "_numerics_stamps", {})


def _identity(report):
    return {key: getattr(report, key) for key in
            ("profile", "source_stamp", "panel_stamp", "protocol_stamp")}


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    # Workers use the controller's immutable provenance, including dirty state.
    node.workerinput["numerics_identity"] = _identity(node.config._numerics_coverage)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    config = node.config
    payload = node.workeroutput.get("numerics_outcomes")
    if error or payload is None:
        config._numerics_worker_errors.append("worker lost numerical outcomes")
        return
    data = json.loads(payload)
    if data["identity"] != _identity(config._numerics_coverage):
        config._numerics_worker_errors.append("worker numerical provenance differs from controller")
        return
    for name in ("outcomes", "deferred"):
        target = getattr(config, "_numerics_" + name)
        for nodeid, state in data[name].items():
            if nodeid in target and target[nodeid] != state:
                config._numerics_worker_errors.append("conflicting numerical outcomes: " + nodeid)
            else:
                target[nodeid] = state


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    report = config._numerics_coverage
    if hasattr(config, "workerinput"):
        # JSON avoids execnet restrictions on otherwise JSON-portable scalars.
        config.workeroutput["numerics_outcomes"] = json.dumps(dict(
            identity=_identity(report), outcomes=config._numerics_outcomes,
            deferred=config._numerics_deferred), allow_nan=False)
        return  # Only the controller writes reports or the CI step summary.
    _populate_report(report, config._numerics_outcomes, config._numerics_deferred,
                     config._numerics_worker_errors)
    if config._numerics_worker_errors and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    outcomes = config.getoption("--numerics-outcomes")
    if outcomes:
        outcomes.parent.mkdir(parents=True, exist_ok=True)
        outcomes.write_text(json.dumps(dict(identity=_identity(report),
            outcomes=config._numerics_outcomes, deferred=config._numerics_deferred,
            errors=config._numerics_worker_errors), allow_nan=False))
    directory = config.getoption("--numerics-report-dir")
    if directory:
        report.write(directory)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as stream:
                stream.write(report.markdown())


def _populate_report(report, outcomes, deferred, errors):
    states = {**deferred, **outcomes}
    for nodeid in sorted(states):
        state = states[nodeid]
        for cell in state["markers"]:
            if cell["backend"] == "legacy":
                from bench.numerics_report import Cell, hardware_for
                report.observations.pop(Cell(**cell, hardware=hardware_for("legacy")), None)
            report.record(**cell, status=state["status"], reason=state["reason"], **state["stamps"])
    if errors:
        for cell in report.required:
            report.record(**asdict(cell), status="not_run", reason="; ".join(sorted(set(errors))))


def report_from_payloads(payloads):
    """Combine CPU/GPU lanes by test identity, retaining every observation once."""
    identity = payloads[0]["identity"]
    merged = {"outcomes": {}, "deferred": {}}
    errors = []
    for payload in payloads:
        if payload["identity"] != identity:
            raise ValueError("suite lanes have incompatible numerical provenance")
        errors.extend(payload.get("errors", []))
        for name in merged:
            for nodeid, state in payload[name].items():
                if nodeid in merged[name] and merged[name][nodeid] != state:
                    raise ValueError("conflicting suite numerical outcome: " + nodeid)
                merged[name][nodeid] = state
    report = CoverageReport(**identity)
    legacy_availability(report)
    _populate_report(report, merged["outcomes"], merged["deferred"], errors)
    return report


def pytest_terminal_summary(terminalreporter):
    report = terminalreporter.config._numerics_coverage
    terminalreporter.section("numerics coverage: " + ("CERTIFIED" if report.certified else "NOT CERTIFIED"))
    counts = Counter(row["status"] for row in report.rows() if row["required"])
    terminalreporter.write_line(f"{report.profile} required cells: " + ", ".join(
        f"{status}={counts[status]}" for status in ("passed", "failed", "not_run")))
    groups = Counter((row["backend"], row["hardware"], row["status"])
                     for row in report.rows() if row["required"])
    for (backend, hardware, status), count in sorted(groups.items()):
        terminalreporter.write_line(f"  {backend}/{hardware}: {status}={count}")
    for row in report.rows():
        if row["backend"] == "legacy":
            terminalreporter.write_line(f"  {row['path']}: {row['status']}: " + "; ".join(row["reasons"]))
    directory = terminalreporter.config.getoption("--numerics-report-dir")
    terminalreporter.write_line(f"Full per-path/check reasons: {directory}/coverage.md" if directory else
        "Use --numerics-report-dir DIR for per-path/check JSON and Markdown receipts.")

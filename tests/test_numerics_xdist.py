"""Distributed numerical receipts retain serial evidence, including non-passes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bench import numerics_pytest as bridge
from bench.numerics_report import CoverageReport

ROOT = Path(__file__).resolve().parents[1]


def test_serial_and_two_worker_reports_are_identical(tmp_path):
    pytest.importorskip('xdist')
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    (tmp_path / 'test_receipts.py').write_text('''
import pytest
pytestmark = pytest.mark.numerics_coverage(backend='cpu', path='scalar', check='fixed_pose')
@pytest.fixture
def teardown_failure():
    yield
    assert False, 'planted teardown failure'
@pytest.fixture
def setup_failure():
    pytest.fail('planted setup failure')
@pytest.mark.parametrize('number', range(4))
def test_pass(number, numerics_coverage):
    numerics_coverage.stamp(build={'source': 'synthetic'}, arithmetic={'exp': 'test'})
def test_skip():
    pytest.skip('deliberately unrun')
@pytest.mark.xfail(reason='planted expected failure')
def test_xfail():
    assert False
@pytest.mark.xfail(reason='planted unexpected pass')
def test_xpass():
    pass
def test_setup(setup_failure):
    pass
def test_teardown(teardown_failure, numerics_coverage):
    numerics_coverage.stamp(build={'source': 'synthetic'}, arithmetic={'exp': 'test'})
@pytest.mark.optimizer_model_qualification
def test_deferred():
    raise AssertionError('must be deselected')
''')
    reports = []
    for name, parallel in [('serial', []), ('parallel', ['-n', '2'])]:
        directory = tmp_path / name
        result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'bench.numerics_pytest',
            *parallel, '--numerics-report-dir', str(directory),
            '--numerics-panel-stamp', 'synthetic-v1', str(tmp_path / 'test_receipts.py')],
            cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(ROOT),
                'GITHUB_STEP_SUMMARY': str(tmp_path / (name + '-summary.md'))},
            capture_output=True, text=True, timeout=90)
        assert result.returncode == 1, result.stdout + result.stderr
        reports.append(json.loads((directory / 'coverage.json').read_text()))
        assert (tmp_path / (name + '-summary.md')).read_text().count('Numerics coverage (') == 1
    assert reports[0] == reports[1]
    assert (tmp_path/'serial/coverage.md').read_bytes() == (tmp_path/'parallel/coverage.md').read_bytes()
    cell = next(row for row in reports[0]['rows'] if row['backend']=='cpu'
                and row['path']=='scalar' and row['check']=='fixed_pose' and row['observations'])
    assert len(cell['observations']) == 10  # nine executed items + one deferred, once
    assert cell['status'] == 'failed'
    assert not reports[1]['certified']


def test_lost_worker_cannot_certify_or_exit_successfully(tmp_path):
    report = CoverageReport(source_stamp='source', panel_stamp='panel')
    config = SimpleNamespace(_numerics_coverage=report, _numerics_outcomes={},
        _numerics_deferred={}, _numerics_worker_errors=[], getoption=lambda key: None)
    bridge.pytest_testnodedown(SimpleNamespace(config=config, workeroutput={}), 'crash')
    session = SimpleNamespace(config=config, exitstatus=0)
    bridge.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert not report.certified
    assert all('worker lost' in row['reasons'][0] for row in report.rows())

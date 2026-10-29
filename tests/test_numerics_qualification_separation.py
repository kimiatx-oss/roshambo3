"""Execution and prepared-geometry uniqueness are different obligations."""
import pytest
from bench.numerics_end_to_end import require_fixture_qualification
from bench.numerics_report import CoverageReport


def test_fixture_requirement_fails_closed_on_missing_or_unqualified_case():
    for report in ({}, {'fixture_qualification': {'cases': []}},
                   {'fixture_qualification': {'cases': [{'production_qualified': False}]}}):
        with pytest.raises(AssertionError, match='qualification unavailable'):
            require_fixture_qualification(report)
    require_fixture_qualification({'fixture_qualification': {
        'cases': [{'production_qualified': True}]}})


def test_passed_execution_does_not_certify_unqualified_fixture():
    report = CoverageReport(profile='cpu-cuda', source_stamp='test-source',
                            panel_stamp='test-panel')
    report.record('cpu', 'vector', 'end_to_end', 'passed', 'strict execution checks passed',
                  hardware="x86_64", build_stamp={"compiler": "synthetic"},
                  arithmetic_stamp={"model": "synthetic"}, evidence={
                      'fixture_qualification': {'all_production_qualified': False}})
    row = next(row for row in report.rows() if row['backend'] == 'cpu'
               and row['path'] == 'vector' and row['check'] == 'end_to_end'
               and row['channel'] == 'shape' and row['hardware'] == 'x86_64')
    assert row['status'] == 'passed'
    assert 'prepared_geometry_uniqueness' in report.qualification_missing(row)
    assert not report.certified

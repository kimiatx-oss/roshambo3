"""The explicit model lane stays red when invoked; default coverage stays unknown."""
import json
from pathlib import Path
import subprocess
import sys


def test_model_lane_is_visible_by_default_and_preserves_failures(tmp_path):
    root = Path(__file__).resolve().parents[1]
    probe = tmp_path / 'test_probe.py'
    probe.write_text('''import pytest

def test_default_integrity():
    assert True

@pytest.mark.optimizer_model_qualification
@pytest.mark.numerics_coverage(backend="cpu",path="vector",check="optimizer",channel="shape")
def test_finite_model_quality():
    assert False, "unchanged model control failure"
''')
    for explicit in (False, True):
        output = tmp_path / str(explicit)
        arguments = ['-p', 'bench.numerics_pytest', '-q', str(probe),
                     '--numerics-report-dir', str(output)]
        if explicit:
            arguments.append('--optimizer-model-qualification')
        code = ('import sys; '
                'sys.meta_path[:]=[f for f in sys.meta_path if "ScikitBuild" not in type(f).__name__]; '
                f'sys.path[:0]={[str(root / "src"), str(root)]!r}; '
                'import pytest; '
                f'raise SystemExit(pytest.main({arguments!r}))')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == int(explicit), result.stdout + result.stderr
        report = json.loads((output / 'coverage.json').read_text())
        row = next(r for r in report['rows'] if r['backend'] == 'cpu' and r['path'] == 'vector'
                   and r['check'] == 'optimizer' and r['channel'] == 'shape'
                   and r['observations'])
        assert row['status'] == ('failed' if explicit else 'not_run')
        assert not report['certified']
        if not explicit:
            assert '--optimizer-model-qualification' in row['reasons'][0]

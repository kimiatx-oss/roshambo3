"""The parallel entry point partitions the full inventory without dropping tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.test_suite import combine_outcomes, plan
from bench.test_pytest import gpu_file, gpu_item, pytest_sessionstart
from bench.numerics_pytest import report_from_payloads
from bench.numerics_report import CoverageReport


def test_plan_is_disjoint_complete_and_conservative(tmp_path):
    tests=tmp_path/'tests';tests.mkdir()
    (tests/'test_compute.py').write_text('def test_sum(): pass')
    (tests/'test_accelerator.py').write_text('def test_device(): pass')
    (tests/'suite_resources.json').write_text(json.dumps({'test_compute.py':[], 'test_accelerator.py':['*']}))
    groups=plan(tmp_path)
    assert groups=={'cpu':['tests/test_compute.py'],'gpu':['tests/test_accelerator.py']}


def test_raw_parallel_gpu_selection_is_refused(tmp_path):
    path=tmp_path/'test_accelerator.py';path.write_text('import cuda')
    (tmp_path/'suite_resources.json').write_text(json.dumps({path.name:['*']}))
    config=SimpleNamespace(option=SimpleNamespace(numprocesses=2),args=[str(path)], getoption=lambda name: None)
    with pytest.raises(pytest.UsageError,match='GPU-bearing'):
        pytest_sessionstart(SimpleNamespace(config=config))


def test_lane_overlap_is_refused():
    item=dict(tests={'same':[]},deselected=[],durations_by_file={})
    with pytest.raises(ValueError,match='overlap'):
        combine_outcomes([item,item])


def test_numerical_lane_merge_orders_test_identity_and_deduplicates_deferred():
    identity={key:getattr(CoverageReport(source_stamp='source',panel_stamp='panel'),key)
              for key in ['profile','source_stamp','panel_stamp','protocol_stamp']}
    state=dict(markers=[dict(backend='cpu',path='scalar',check='fixed_pose')],
               status='passed',reason='assertions passed',stamps=dict(build_stamp={'source':'test'},arithmetic_stamp={'exp':'test'}))
    deferred=dict(state,status='not_run',reason='deferred')
    first=dict(identity=identity,outcomes={'z':state},deferred={'deferred':deferred},errors=[])
    second=dict(identity=identity,outcomes={'a':state},deferred={'deferred':deferred},errors=[])
    report=report_from_payloads([first,second]).as_dict()
    combined=dict(identity=identity,outcomes={'z':state,'a':state},deferred={'deferred':deferred},errors=[])
    assert report==report_from_payloads([combined]).as_dict()
    row=next(r for r in report['rows'] if r['backend']=='cpu' and r['observations'])
    assert len(row['observations'])==3
    with pytest.raises(ValueError,match='provenance'):
        report_from_payloads([first,dict(second,identity=dict(identity,source_stamp='other'))])


def test_gpu_lock_spans_the_entire_test_protocol(tmp_path, monkeypatch):
    from bench import gpu_lock, test_pytest
    path=tmp_path/'test_device.py';path.write_text('import cuda')
    (tmp_path/'suite_resources.json').write_text(json.dumps({path.name:['*']}))
    events=[]
    class Lock:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):events.append('acquire')
        def __exit__(self,*args):events.append('release')
    monkeypatch.setattr(gpu_lock,'GpuLock',Lock)
    monkeypatch.setattr(test_pytest,'_devices',lambda:(object(),))
    config=SimpleNamespace(getoption=lambda name:name=='--suite-gpu-lock')
    monkeypatch.setattr(test_pytest,'_event',lambda *args,**kwargs:None)
    protocol=test_pytest.pytest_runtest_protocol(SimpleNamespace(path=path,config=config,nodeid='case'),None)
    next(protocol)
    assert events==['acquire']
    # The yield wraps setup (including module fixtures), call and teardown.
    with pytest.raises(StopIteration):next(protocol)
    assert events==['acquire','release']


def test_collection_skips_and_errors_are_retained():
    from bench.test_pytest import Outcomes
    observer=Outcomes(SimpleNamespace())
    observer.pytest_collectreport(SimpleNamespace(passed=False, nodeid='test_missing.py',
        outcome='skipped',longrepr=('test_missing.py',1,'dependency missing')))
    observer.pytest_collectreport(SimpleNamespace(passed=False, nodeid='test_broken.py',
        outcome='failed',longrepr='planted import error'))
    assert observer.collection=={'test_missing.py':dict(outcome='skipped',reason='dependency missing'),
                                'test_broken.py':dict(outcome='failed',reason='planted import error')}


def test_all_visible_gpus_are_resolved_in_uuid_order(monkeypatch):
    from bench import gpu_lock, test_pytest
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(gpu_lock,'_list_physical_gpus',lambda:[(0,'B'),(1,'A')])
    monkeypatch.setattr(gpu_lock,'resolve_device',lambda i:SimpleNamespace(uuid=['B','A'][i]))
    test_pytest._devices.cache_clear()
    try:
        assert [d.uuid for d in test_pytest._devices()]==['A','B']
    finally:
        test_pytest._devices.cache_clear()


def test_gpu_resolution_failure_never_means_unlocked(monkeypatch):
    from bench import gpu_lock, test_pytest
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    def fail():raise gpu_lock.DeviceResolutionError('planted mapping failure')
    monkeypatch.setattr(gpu_lock,'_list_physical_gpus',fail)
    test_pytest._devices.cache_clear()
    with pytest.raises(gpu_lock.DeviceResolutionError,match='mapping'):
        test_pytest._devices()
    test_pytest._devices.cache_clear()



def test_inventory_requires_new_files_to_declare_resources(tmp_path):
    (tmp_path/'suite_resources.json').write_text('{}')
    path=tmp_path/'test_unknown.py';path.write_text('def test_new(): pass')
    with pytest.raises(pytest.UsageError,match='declare GPU selectors'):
        gpu_file(path)


def test_repository_inventory_has_no_stale_or_missing_files_or_selectors():
    import ast
    from bench.test_pytest import RESOURCE_FILE, resource_manifest
    directory=RESOURCE_FILE.parent
    inventory=resource_manifest()
    assert set(inventory)=={p.name for p in directory.rglob('test_*.py')}
    for name, selectors in inventory.items():
        tree=ast.parse((directory/name).read_text())
        nodes=set()
        def visit(parent,prefix=''):
            for node in parent.body:
                if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
                    nodes.add(prefix+node.name)
                    if isinstance(node,ast.ClassDef):visit(node,prefix+node.name+'::')
        visit(tree)
        assert set(selectors)-{'*','@backend'}<=nodes, name
        if '@backend' in selectors:
            assert any(isinstance(n,ast.arg) and n.arg=='backend' for n in ast.walk(tree)),name


def test_gpu_words_in_cpu_and_mock_tests_do_not_serialize_them():
    from bench.test_suite import ROOT
    groups=plan()
    for name in ['test_bench_gpu_run.py','test_bench_clocks.py','test_format3_gpu_launch.py',
                 'test_numerics_fixtures.py','test_definitions.py','test_numerics_report.py']:
        assert 'tests/'+name in groups['cpu']
        assert 'tests/'+name not in groups['gpu']
    # The preparation and public numerical pipelines contain real GPU tests,
    # but their CPU tests are no longer swept into the GPU lane.
    for name in ['test_prepare.py','test_numerics_end_to_end.py','test_web.py']:
        assert 'tests/'+name in groups['cpu'] and 'tests/'+name in groups['gpu']
    cpu=SimpleNamespace(path=ROOT/'tests/test_numerics_end_to_end.py',
        nodeid='tests/test_numerics_end_to_end.py::test_real_shape_public_pipeline[identity-cpu-field]',
        callspec=SimpleNamespace(params={'backend':'cpu'}))
    device=SimpleNamespace(**{**vars(cpu),'callspec':SimpleNamespace(params={'backend':'cuda'})})
    assert not gpu_item(cpu) and gpu_item(device)


def test_mixed_file_cpu_gpu_routing_preserves_serial_outcomes_and_numerics(tmp_path):
    import os,subprocess,sys
    from bench.test_suite import ROOT
    from bench.numerics_pytest import report_from_payloads
    tests=tmp_path/'tests';tests.mkdir()
    (tests/'conftest.py').write_text("pytest_plugins=['bench.numerics_pytest','bench.test_pytest']\n")
    (tests/'suite_resources.json').write_text(json.dumps({'test_mixed.py':['TestDevice','test_device','@backend']}))
    (tests/'test_mixed.py').write_text("""
import pytest
@pytest.mark.numerics_coverage(backend='cpu',path='scalar',check='fixed_pose')
def test_gpu_mock(numerics_coverage):
    numerics_coverage.stamp(build={'source':'synthetic'},arithmetic={'exp':'test'})
class TestDevice:
    @pytest.mark.parametrize('number',range(2))
    def test_context(self,number):pass
@pytest.mark.parametrize('number',range(2),ids=['device::zero','gpu mock::one'])
def test_device(number):pass
@pytest.mark.parametrize('backend',['cpu','cuda','metal'])
def test_backend(backend):pass
@pytest.mark.optimizer_model_qualification
def test_deferred():pass
""")
    payloads=[];numbers=[]
    for name,flags in [('serial',[]),('cpu',['-n','2','--dist','loadfile','--suite-lane','cpu']),
                       ('gpu',['--suite-lane','gpu'])]:
        result=subprocess.run([sys.executable,'-m','pytest',str(tests),*flags,
            '--suite-outcomes',str(tmp_path/(name+'.json')),
            '--numerics-outcomes',str(tmp_path/(name+'-numerics.json'))],
            cwd=tmp_path,env={**os.environ,'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':''},
            capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
        payloads.append(json.loads((tmp_path/(name+'.json')).read_text()))
        numbers.append(json.loads((tmp_path/(name+'-numerics.json')).read_text()))
    merged=combine_outcomes(payloads[1:])
    assert merged['tests']==payloads[0]['tests']
    assert len(payloads[1]['tests'])==2 and len(payloads[2]['tests'])==6
    assert merged['deselected']==payloads[0]['deselected']==['tests/test_mixed.py::test_deferred']
    assert report_from_payloads(numbers[1:]).as_dict()==report_from_payloads(numbers[:1]).as_dict()



def test_raw_cpu_lane_requires_hidden_devices_before_collection(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    config=SimpleNamespace(option=SimpleNamespace(numprocesses=4),args=['tests'],
        getoption=lambda name:'cpu' if name=='--suite-lane' else None)
    with pytest.raises(pytest.UsageError,match='before collection'):
        pytest_sessionstart(SimpleNamespace(config=config))

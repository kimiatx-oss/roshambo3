"""Suite scheduling support: isolated caches, short GPU holds and outcome receipts."""
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import time

import pytest

RESOURCE_FILE = Path(__file__).resolve().parents[1]/'tests/suite_resources.json'
# Scheduling tests compile synthetic children independently; they must not
# hold the same outer compiler lock that those children exercise.
COMPILER_LOCK_EXEMPT = {'test_numerics_xdist.py', 'test_suite_runner.py',
                        'test_shared_cache.py', 'test_cmake_arch_discovery.py'}


@lru_cache(None)
def resource_manifest(path=RESOURCE_FILE):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or any(not isinstance(v, list) or
            any(not isinstance(s, str) for s in v) for v in data.values()):
        raise pytest.UsageError('suite resource inventory must map test files to GPU selector lists')
    return data


def gpu_selectors(path):
    path = Path(path)
    manifest_path = path.parent/'suite_resources.json'
    # The production inventory lives next to tests; standalone synthetic
    # scheduling tests can supply their own inventory the same way.
    if not manifest_path.is_file():
        raise pytest.UsageError('missing suite resource inventory: '+str(manifest_path))
    manifest = resource_manifest(manifest_path)
    if path.name not in manifest:
        raise pytest.UsageError('declare GPU selectors (or [] for CPU) in suite_resources.json: '+path.name)
    return manifest[path.name]


def gpu_file(path):
    return bool(gpu_selectors(path))


def gpu_item(item):
    selectors = gpu_selectors(item.path)
    # Strip the entire parameter label before interpreting class/function
    # separators: pytest IDs may themselves contain '::' or GPU words.
    local = item.nodeid.split('::', 1)[-1].split('[', 1)[0]
    for selector in selectors:
        if selector == '@backend':
            backend = getattr(item, 'callspec', None)
            backend = backend.params.get('backend') if backend is not None else None
            if backend in ('cuda', 'metal'):
                return True
        elif selector == '*' or local == selector or local.startswith(selector+'::'):
            return True
    return False


@lru_cache(None)
def compiling_file(path):
    if Path(path).name in COMPILER_LOCK_EXEMPT:
        return False
    return bool(re.search(r'''["'](?:g\+\+|c\+\+|clang\+\+|cmake|nvcc|ninja)["']''', Path(path).read_text()))


def pytest_addoption(parser):
    group = parser.getgroup('suite resources')
    group.addoption('--suite-outcomes', type=Path)
    group.addoption('--suite-events', type=Path)
    group.addoption('--suite-cache-root', type=Path)
    group.addoption('--suite-gpu-lock', action='store_true')
    group.addoption('--suite-lane', choices=('cpu', 'gpu'))


def pytest_configure(config):
    config.pluginmanager.register(Outcomes(config), 'suite-outcomes')


class Outcomes:
    def __init__(self, config):
        self.config = config
        self.tests = defaultdict(list)
        self.durations = defaultdict(float)
        self.deselected = set()
        self.collection = {}

    def pytest_deselected(self, items):
        routed = getattr(self.config, '_suite_routing', set())
        self.deselected.update(item.nodeid for item in items if item.nodeid not in routed)

    def pytest_runtest_logreport(self, report):
        if hasattr(self.config, 'workerinput'):
            return
        status, reason = report.outcome, None
        if hasattr(report, 'wasxfail'):
            status = 'xfail' if report.skipped else 'xpass'
            reason = report.wasxfail
        elif report.skipped:
            reason = report.longrepr[2]
        elif report.failed:
            reason = str(report.longrepr)
        self.tests[report.nodeid].append(dict(phase=report.when, outcome=status, reason=reason))
        self.durations[report.nodeid.split('::')[0]] += report.duration

    def pytest_collectreport(self, report):
        if report.passed:
            return
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        state = dict(outcome=report.outcome, reason=reason)
        previous = self.collection.setdefault(report.nodeid, state)
        if previous != state:
            raise pytest.UsageError('workers disagree on collection outcome: '+report.nodeid)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):
        self.deselected.update(node.workeroutput.get('suite_deselected', []))
        for nodeid, state in node.workeroutput.get('suite_collection', {}).items():
            if self.collection.setdefault(nodeid, state) != state:
                raise pytest.UsageError('workers disagree on collection outcome: '+nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        if hasattr(self.config, 'workerinput'):
            self.config.workeroutput['suite_deselected'] = sorted(self.deselected)
            self.config.workeroutput['suite_collection'] = self.collection
            return
        path = self.config.getoption('--suite-outcomes')
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(tests=dict(sorted(self.tests.items())),
                deselected=sorted(self.deselected), collection=dict(sorted(self.collection.items())),
                durations_by_file=dict(self.durations)), indent=2)+'\n')


@pytest.fixture(scope='session', autouse=True)
def _shared_suite_cache(tmp_path_factory, request):
    previous = os.environ.get('ROSHAMBO3_TEST_CACHE')
    path = request.config.getoption('--suite-cache-root')
    if path is None:
        base = tmp_path_factory.getbasetemp()
        path = (base.parent if hasattr(request.config, 'workerinput') else base)/'shared-artifacts'
    if previous is None:
        os.environ['ROSHAMBO3_TEST_CACHE'] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('ROSHAMBO3_TEST_CACHE', None)


@lru_cache(None)
def _devices():
    from bench.gpu_lock import _list_physical_gpus, resolve_device
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.strip() in ('', '-1'):
        return ()  # Explicitly disabled, no CUDA context is possible.
    try:
        physical = _list_physical_gpus()
    except FileNotFoundError:
        import importlib.util
        if importlib.util.find_spec('roshambo3._roshambo3_cuda') is not None:
            raise pytest.UsageError('CUDA extension present but nvidia-smi cannot resolve lock devices')
        return ()  # A CPU-only install with no NVIDIA query utility.
    if not physical:
        return ()  # Successful query positively identified no NVIDIA GPUs.
    count = len(visible.split(',')) if visible is not None else len(physical)
    devices = {device.uuid: device for device in (resolve_device(i) for i in range(count))}
    # Query/mapping errors propagate. Never convert an uncertain device to an
    # unlocked CUDA execution, and acquire every visible device in one order.
    return tuple(devices[key] for key in sorted(devices))


def _event(config, **entry):
    path = config.getoption('--suite-events')
    if path:
        from filelock import FileLock
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path)+'.lock'):
            with path.open('a') as stream:
                stream.write(json.dumps(dict(pid=os.getpid(), worker=os.environ.get('PYTEST_XDIST_WORKER', 'serial'), **entry))+'\n')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    from tests.shared_cache import compile_lock
    with ExitStack() as resources:
        if compiling_file(str(item.path)):
            resources.enter_context(compile_lock())
        locked = False
        if item.config.getoption('--suite-gpu-lock') and gpu_item(item):
            devices = _devices()
            for device in devices:
                from bench.gpu_lock import GpuLock
                # Fail rather than wait with an already-created context. The
                # entry point can rerun this lane once the other owner finishes.
                resources.enter_context(GpuLock('test-suite', device, blocking=False))
                locked = True
        _event(item.config, event='start', nodeid=item.nodeid, gpu_lock=locked, monotonic=time.monotonic())
        try:
            yield
        finally:
            _event(item.config, event='finish', nodeid=item.nodeid, gpu_lock=locked, monotonic=time.monotonic())


@pytest.hookimpl(hookwrapper=True)
def pytest_collection(session):
    # Protect CUDA availability probes/imports too, but release before tests.
    with ExitStack() as resources:
        if session.config.getoption('--suite-gpu-lock'):
            devices = _devices()
            for device in devices:
                from bench.gpu_lock import GpuLock
                resources.enter_context(GpuLock('test-suite-collection', device, blocking=False))
        yield


def pytest_sessionstart(session):
    config = session.config
    if config.getoption('--suite-lane') == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise pytest.UsageError('the CPU suite lane requires CUDA_VISIBLE_DEVICES="" before collection')
    if hasattr(config, 'workerinput') or not getattr(config.option, 'numprocesses', 0):
        return
    for argument in config.args:
        path = Path(str(argument).split('::')[0])
        paths = path.rglob('test_*.py') if path.is_dir() else [path]
        if (config.getoption('--suite-lane') != 'cpu' and
                any(p.is_file() and gpu_file(str(p.resolve())) for p in paths)):
            raise pytest.UsageError('GPU-bearing files cannot run on multiple workers; use python -m bench.test_suite')


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    lane = config.getoption('--suite-lane')
    if not lane:
        return
    if lane == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise pytest.UsageError('the CPU suite lane requires CUDA_VISIBLE_DEVICES=""')
    selected, routed = [], []
    for item in items:
        (selected if gpu_item(item) == (lane == 'gpu') else routed).append(item)
    # Routing is reported to pytest's terminal, but it is not scientific
    # deselection: those exact tests execute in the complementary suite lane.
    config._suite_routing = {item.nodeid for item in routed}
    try:
        if routed:
            config.hook.pytest_deselected(items=routed)
    finally:
        config._suite_routing = set()
    items[:] = selected

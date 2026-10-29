import numpy as np
import pytest
from bench.numerics_optimizer_real import freeze, exceeds, seed_intervals, pack, scored


def case():
    return dict(a=np.array([[0.,0.,0.],[1.,.5,0.]]),b=np.array([[0.,0.,0.],[1.,.5,0.]]),
        ac=np.array([[.25,.1,0.],[-.25,1.,0.]]),bc=np.array([[.25,.1,0.],[-.25,1.,0.]]),
        at=np.array([1,2]),bt=np.array([1,2]),r=np.diag([0.,1.,1.]),p=np.diag([0.,1.,1.]))


def test_combo_rescore_counts_typed_color_at_own_pose():
    rows=np.zeros((1,20));rows[:,9]=rows[:,16]=1
    values,_=scored([case()],rows,'combo')
    assert values[0]==pytest.approx(1.)
    rows[:,13]=3.
    moved,_=scored([case()],rows,'combo')
    assert moved[0]<values[0]-.5


def test_permutation_preserves_color_segmentation_and_count():
    c=case();c['r']=c['p']=np.eye(7)
    inputs=pack([c],'combo',270300)
    assert inputs['library_counts'].tolist()==[[2,1,1,0,0,0,0]]
    assert sorted(inputs['query_types'][0].tolist())==[0,0,1,2]


def test_control_seed_confidence_does_not_count_dependent_pair_rows():
    controls=[dict(overall=dict(n=128,mean_loss=-.01,winning_start_disagreement=0.,basin_change=.1)) for _ in range(32)]
    limits=freeze(controls,.001)
    assert limits['mean_loss']==pytest.approx(.011)
    assert seed_intervals(controls)['mean_loss']==pytest.approx([-.01,-.01])
    assert exceeds(dict(overall=dict(mean_loss=-.02,winning_start_disagreement=0.,basin_change=.1)),limits)==[]


@pytest.fixture(scope='module')
def live_control_module(tmp_path_factory):
    """Compile only the tiny test instrument; never rebuild production here."""
    import importlib.util
    import os
    from pathlib import Path
    import shlex
    import subprocess
    import sysconfig
    pybind11 = pytest.importorskip('pybind11')
    from roshambo3.backend import cpu
    if not cpu.available():
        pytest.skip('CPU extension unavailable')
    path=tmp_path_factory.mktemp('optimizer-control') / ('_numerics_optimizer_control'+sysconfig.get_config_var('EXT_SUFFIX'))
    source=Path(__file__).resolve().parents[1] / 'bench/optimizer_control.cpp'
    flags=shlex.split(cpu.build_stamp()['cxx_flags'].split('[')[0])
    # Python supplies Darwin's bundle/dynamic-lookup flags as well as Linux's
    # shared-library flag; raw -shared leaves Python symbols unresolved on macOS.
    link_flags = shlex.split(sysconfig.get_config_var('LDSHARED'))[1:]
    command=[os.environ.get('CXX','c++'),*flags,'-std=c++17',*link_flags,'-fPIC',
        '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),str(source),'-o',str(path)]
    subprocess.run(command,check=True)
    spec=importlib.util.spec_from_file_location('_numerics_optimizer_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.__compiler_command__=command
    module.__compiler_flags__=flags
    return module


@pytest.fixture(scope='module')
def live_optimizer_evidence(live_control_module, request):
    """Reuse controls; execute CUDA quality checks when selected and available."""
    import sys
    from bench import numerics_optimizer_real as runner
    wanted = any(item.name == 'test_live_real_cuda_optimizer' for item in request.session.items)
    cuda_reason = 'CUDA optimizer check not selected'
    if wanted:
        try:
            from roshambo3 import _roshambo3_cuda as extension
            cuda_reason = None if extension.device_count() > 0 else 'CUDA device unavailable'
        except (ImportError, RuntimeError) as error:
            cuda_reason = 'CUDA backend unavailable: ' + str(error)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, '_numerics_optimizer_control', live_control_module)
        evidence = runner.run(cuda=cuda_reason is None)
    return evidence, cuda_reason


@pytest.mark.optimizer_model_qualification
@pytest.mark.numerics_coverage(backend='cpu',path='vector',check='optimizer',channel='shape')
@pytest.mark.numerics_coverage(backend='cpu',path='vector',check='optimizer',channel='color')
def test_live_real_optimizer_smoke(live_optimizer_evidence,numerics_coverage):
    """Explicit model qualification preserves the full frozen panel and limits."""
    from bench.numerics_backends import verified_build
    evidence, _ = live_optimizer_evidence
    numerics_coverage.stamp(build=verified_build('cpu'),
        arithmetic=dict(model='v2 finite perturbation model; full preregistered panel'),evidence=evidence)
    for objective,item in evidence['objectives'].items():
        assert item['heldout_valid'], (objective,item['limits'])
        assert not item['cpu']['exceeding'], (objective,item['cpu']['production'])
        assert item['cpu']['plant_exceeding'], (objective,'step-schedule plant survived')
    assert evidence['objectives']['shape']['zero_instrument']['exact_per_pair_match']


@pytest.mark.optimizer_model_qualification
@pytest.mark.numerics_coverage(backend='cuda',path='analytic',check='optimizer',channel='shape')
@pytest.mark.numerics_coverage(backend='cuda',path='analytic',check='optimizer',channel='color')
def test_live_real_cuda_optimizer(live_optimizer_evidence,numerics_coverage):
    from bench.numerics_backends import verified_build
    evidence, reason = live_optimizer_evidence
    if reason is not None:
        pytest.skip(reason)
    numerics_coverage.stamp(build=verified_build('cuda'),
        arithmetic=dict(model='v2 finite perturbation model; full preregistered panel'),
        evidence=evidence)
    for objective, item in evidence['objectives'].items():
        assert item['heldout_valid'], (objective, item['limits'])
        arm = item['cuda']
        assert arm['ran'], (objective, arm.get('why'))
        assert not arm['exceeding'], (objective, arm['production'])
        assert arm['plant_exceeding'], (objective, 'step-schedule plant survived')


def test_zero_instrument_rejects_cancelling_per_pair_errors():
    from bench.numerics_optimizer_real import assert_zero_instrument
    anchor = (np.array([.4, .6]), [np.zeros((2, 3)), np.ones((2, 3))])
    assert_zero_instrument(anchor, anchor)
    changed = (anchor[0] + [.01, -.01], anchor[1])
    assert np.mean(anchor[0] - changed[0]) == 0
    with pytest.raises(AssertionError, match="per-pair scores"):
        assert_zero_instrument(anchor, changed)
    changed_pose = (anchor[0], [anchor[1][0] + .001, anchor[1][1]])
    with pytest.raises(AssertionError, match="physical pose"):
        assert_zero_instrument(anchor, changed_pose)


def test_live_zero_amplitude_instrument_matches_every_scalar_pose(live_control_module):
    """Default exact instrument check, independent of finite-model quality gates.

    This checks an unchanged computation promise. It does not calibrate an
    allowance, infer optimizer equivalence, or certify screening accuracy.
    """
    from bench.numerics_optimizer_real import (SETTINGS, assert_zero_instrument,
                                               panel, scored)
    from bench.numerics_backends import verified_build
    from roshambo3.backend import cpu
    verified_build('cpu')
    cases, _ = panel()
    anchor = cpu.score_pairs(**pack(cases, 'shape'), **SETTINGS, grain=-1, poly=False)
    zero = np.asarray([live_control_module.score(c['a'], c['b'], 0, 0.,
                       SETTINGS['steps'], SETTINGS['lr_q'], SETTINGS['lr_t'])[0]
                       for c in cases])
    assert_zero_instrument(scored(cases, anchor, 'shape'), scored(cases, zero, 'shape'))

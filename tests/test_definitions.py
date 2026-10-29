"""Custom color definitions -- the shipped allow-list, the trust store, the
refusal text, the pre-pass, and the limited subprocess (``roshambo3.definitions``)."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
import warnings

import pytest

from roshambo3 import definitions as defs
from roshambo3.format.r3g import _sandbox, reference
from roshambo3.format.r3g.errors import CorpusValidationError

from tests import definition_cases as cases

BASE_HASH = "766f9790513f8f94c67ec6d1c8b6d5bede59b81f6da8c2081f57e49fb28d9784"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def xdg(tmp_path, monkeypatch):
    """Never touch the real user config directory."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "xdg"


def base_text() -> str:
    return defs.shipped_definitions()[-1].text


def custom_color(extra: str = "DefineFeature Foo [F]\n  Family Halogen\n  Weights 1.0\nEndFeature\n"):
    return cases.with_definition(base_text() + "\n" + extra,
                                 types=cases.conformance_color()["types"])


def write_corpus(tmp_path, color, name="c"):
    d = tmp_path / name
    d.mkdir()
    desc = json.load(open(os.path.join(cases.CONFORMANCE_COLOR, "corpus.json")))
    n = len(color["types"])
    eye = {"dtype": "f32", "shape": [n + 1, n + 1],
           "values": [[1.0 if i == j and i else 0.0 for j in range(n + 1)] for i in range(n + 1)]}
    desc["color"] = dict(color, r=eye, p=eye)
    (d / "corpus.json").write_text(json.dumps(desc))
    return str(d)


def run_cli(*argv, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([sys.executable, "-m", "roshambo3.cli", *argv], capture_output=True,
                          text=True, env=e, cwd=ROOT, timeout=120)


def _secure_trust_parent(path):
    """Make a trust-store parent with secure permissions under any umask."""
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)


def _write_trust_fixture(path, text):
    _secure_trust_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o600)


# ----------------------------------------------------------------- allow-list

def test_shipped_package_data():
    ships = defs.shipped_definitions()
    assert ships and ships[-1].hash == BASE_HASH and ships[-1].version == "2026.09.1"
    root = os.path.join(os.path.dirname(defs.__file__), "data", "definitions", "2026.09.1")
    assert os.path.exists(os.path.join(root, "PROVENANCE"))
    assert BASE_HASH in open(os.path.join(root, "PROVENANCE")).read()


def test_shipped_matches_installed_rdkit():
    rdkit = pytest.importorskip("rdkit")
    from rdkit import RDConfig
    text = open(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"), encoding="utf-8").read()
    if rdkit.__version__ == "2026.09.1":
        assert reference.definition_hash(text) == BASE_HASH


def test_shipped_definition_passes_without_trust(xdg):
    color = cases.conformance_color()
    v = defs.check_definition(color)
    assert v.status == "shipped" and v.in_process and v.hash == BASE_HASH
    assert not xdg.exists()                     # the trust file was never created


def test_shape_only_needs_no_definition(xdg):
    assert defs.check_definition(None) is None
    assert not xdg.exists()


# ------------------------------------------------------------- refusal / trust

GOLDEN = """error: {corpus} uses a custom color definition that is not one of RDKit's defaults.
  definition  sha256 {h12}…{h4}
  families    {fams}
  differs     adds family "Halogen"
  size        {kb} KB, 28 patterns  (within limits)

Custom definitions are compiled and run by RDKit on your queries. Vet it before
trusting it:
  roshambo3 definitions show {corpus}            # full text
  roshambo3 definitions trust {h12} --corpus {corpus}   # trust it from now on
or for this run only:
  roshambo3 screen ... --trust-definition {h12}"""


def test_untrusted_refusal_golden_text():
    color = custom_color()
    h = reference.definition_hash(color["definition"])
    with pytest.raises(defs.UntrustedDefinitionError) as ei:
        defs.check_definition(color, corpus_path="lib/c3")
    e = ei.value
    kb = f"{len(color['definition'].encode()) / 1024:.1f}"
    want = GOLDEN.format(corpus="lib/c3", h12=h[:16], h4=h[-4:], kb=kb,
                         fams=", ".join(sorted(set(f for f in e.families))))
    assert str(e) == want
    assert e.hash == h and e.n_patterns == 28 and e.within_limits and "Halogen" in e.families
    assert isinstance(e, ValueError)


def test_differs_line_reports_changed_patterns():
    text = base_text().replace("DefineFeature SingleAtomAcceptor", "DefineFeature SingleAtomAcceptor")
    lines = text.split("\n")
    # change one Acceptor pattern and add nothing
    idx = next(i for i, l in enumerate(lines) if l.startswith("DefineFeature SingleAtomAcceptor"))
    parts = lines[idx].split()
    parts[2] = parts[2].replace("O", "[O,S]", 1) if "O" in parts[2] else parts[2] + ";!a"
    lines[idx] = " ".join(parts)
    color = cases.with_definition("\n".join(lines), types=cases.conformance_color()["types"])
    with pytest.raises(defs.UntrustedDefinitionError) as ei:
        defs.check_definition(color)
    assert ei.value.differs == "changes Acceptor (1 pattern)"


def test_smarts_list_is_compared_with_the_shipped_families():
    color = cases.with_definition(None, generator="smarts_list", types=["Acceptor", "Donor"])
    color["definition"] = {"Acceptor": ["[O;H0]"], "Donor": ["[N;!H0]"]}
    with pytest.raises(defs.UntrustedDefinitionError) as ei:
        defs.check_definition(color)
    assert ei.value.differs.startswith("removes ")
    assert "changes Acceptor (1 pattern)" in ei.value.differs


def test_trusted_flag_lets_it_pass_prefix_or_full():
    color = custom_color()
    h = reference.definition_hash(color["definition"])
    for t in (h, h[:16], h[:20]):
        v = defs.check_definition(color, trusted=[t])
        assert v.status == "trusted" and not v.in_process
    with pytest.raises(defs.UntrustedDefinitionError):
        defs.check_definition(color, trusted=["0" * 64])     # another text: refused again


@pytest.mark.parametrize("bad", ["abcdef", "766f9790513", "766f97905135f8f", "zz" * 8, ""])
def test_short_or_malformed_prefix_is_refused(bad):
    color = custom_color()
    with pytest.raises(ValueError, match="hex|at least"):
        defs.check_definition(color, trusted=[bad])


def test_prefix_must_name_this_corpus():
    with pytest.raises(ValueError, match="not a prefix"):
        defs.resolve_prefix("0" * 16, custom_color())
    with pytest.raises(ValueError, match="--corpus"):
        defs.resolve_prefix("a" * 16, None)


def test_trust_store_roundtrip_and_format(xdg):
    h = "ab" * 32
    assert defs.trust(h, note="my\tnote") is True
    assert defs.trust(h) is False
    path = defs.trust_file_path()
    assert path == str(xdg / "roshambo3" / "trusted-definitions")
    h2, date, note = open(path).read().rstrip("\n").split("\t")
    assert (h2, note) == (h, "my note") and len(date) == 10
    assert os.listdir(os.path.dirname(path)) == ["trusted-definitions"]   # no temp left
    assert defs.list_trusted() == {h: (date, "my note")}
    with pytest.raises(ValueError, match="full 64-hex"):
        defs.trust(h[:20])


@pytest.mark.filterwarnings("ignore::roshambo3.definitions.TrustFileWarning")
def test_trust_file_malformed_lines_are_reported_and_skipped(xdg):
    p = defs.trust_file_path()
    good = "cd" * 32
    _write_trust_fixture(p, f"# comment\n\nnot-a-hash\t2026-01-01\tx\n{good}\t2026-01-02\tok\n"
                            f"{'ef' * 32}\tnot-a-date\n{'12' * 32}\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = defs.list_trusted()
    assert list(got) == [good]
    msgs = [str(x.message) for x in w]
    assert any(":3:" in m for m in msgs) and any(":5:" in m for m in msgs) \
        and any(":6:" in m for m in msgs) and len(msgs) == 3
    color = custom_color()
    with pytest.raises(defs.UntrustedDefinitionError):         # a bad line never trusts anything
        defs.check_definition(color)


def test_trust_file_entry_passes_the_gate_and_untrust_removes_it():
    color = custom_color()
    h = reference.definition_hash(color["definition"])
    defs.trust(h, note="vetted")
    assert defs.check_definition(color).status == "trusted"
    assert defs.untrust(h[:16]) == h
    with pytest.raises(defs.UntrustedDefinitionError):
        defs.check_definition(color)
    with pytest.raises(ValueError, match="no trusted"):
        defs.untrust(h)


def test_untrust_ambiguous_or_short_prefix_refused():
    defs.trust("ab" * 8 + "00" * 24)
    defs.trust("ab" * 8 + "11" * 24)
    with pytest.raises(ValueError, match="ambiguous"):
        defs.untrust("ab" * 8)
    with pytest.raises(ValueError, match="at least"):
        defs.untrust("abab")
    assert len(defs.list_trusted()) == 2


def test_comments_survive_untrust(xdg):
    p = defs.trust_file_path()
    _write_trust_fixture(p, "# mine\n")
    h = "ab" * 32
    defs.trust(h)
    defs.untrust(h)
    assert open(p).read() == "# mine\n"


# ------------------------------------------------------------------------ CLI

def test_cli_trust_persists_across_processes(tmp_path, xdg):
    color = custom_color()
    corpus = write_corpus(tmp_path, color)
    h = reference.definition_hash(color["definition"])
    env = {"XDG_CONFIG_HOME": str(xdg)}
    r = run_cli("definitions", "trust", h[:16], env=env)                # prefix, no --corpus
    assert r.returncode == 1 and "--corpus" in r.stderr
    assert not xdg.exists()
    r = run_cli("definitions", "trust", h[:16], "--corpus", corpus, "--note", "lab", env=env)
    assert r.returncode == 0, r.stderr
    r = run_cli("definitions", "list", env=env)
    assert r.stdout.split("\t")[0] == h and r.stdout.rstrip().endswith("lab")
    assert defs.check_definition(color).status == "trusted"             # this process reads the file
    r = run_cli("definitions", "untrust", h[:16], env=env)
    assert r.returncode == 0 and h in r.stdout
    assert run_cli("definitions", "list", env=env).stdout == ""


def test_cli_trust_full_hash_without_corpus(xdg):
    h = "ab" * 32
    r = run_cli("definitions", "trust", h, env={"XDG_CONFIG_HOME": str(xdg)})
    assert r.returncode == 0 and defs.list_trusted() == {h: (defs.list_trusted()[h][0], "")}


def test_cli_trust_of_shipped_default_is_a_no_op(xdg):
    r = run_cli("definitions", "trust", BASE_HASH, env={"XDG_CONFIG_HOME": str(xdg)})
    assert r.returncode == 0 and "shipped" in r.stdout and not xdg.exists()


def test_show_never_imports_rdkit(tmp_path):
    color = custom_color("DefineFeature Foo [F]\n  Family Halogen\n  Weights 1.0\nEndFeature\n"
                         "# \x1b[31mred\n")
    corpus = write_corpus(tmp_path, color)
    code = ("import sys\nfrom roshambo3.cli import main\n"
            "import io, contextlib\nbuf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf):\n"
            f"    rc = main(['definitions','show',{corpus!r}])\n"
            f"    rc2 = main(['definitions','show',{corpus!r},'--diff'])\n"
            f"    rc3 = main(['definitions','list'])\n"
            "assert rc == rc2 == rc3 == 0\n"
            "out = buf.getvalue()\n"
            "assert 'Halogen' in out and '\\x1b' not in out and '\\\\x1b[31mred' in out, out[-200:]\n"
            "import roshambo3, roshambo3.definitions\n"
            "assert 'rdkit' not in sys.modules, 'rdkit imported'\n"
            "assert 'roshambo3.pharmacophore' not in sys.modules\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path),
                       env={**os.environ, "XDG_CONFIG_HOME": str(tmp_path / "x")}, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_show_smarts_list_prints_canonical_json(tmp_path):
    color = cases.with_definition(None, generator="smarts_list", types=["Acceptor"])
    color["definition"] = {"Acceptor": ["[O;H0]"]}
    corpus = write_corpus(tmp_path, color)
    assert defs.show(corpus) == '{"Acceptor":["[O;H0]"]}'
    assert defs.show(tmp_path.__class__(cases.CONFORMANCE_COLOR) .as_posix(), diff=True).startswith("identical")


def test_screen_flag_and_api_keyword_exist():
    from roshambo3 import api, cli
    ns = cli.build_parser().parse_args(["screen", "q.sdf", "c", "--trust-definition", "a" * 16,
                                        "--trust-definition", "b" * 64])
    assert ns.trust_definition == ["a" * 16, "b" * 64]
    for fn in (api.screen, api.score_matrix):
        assert "trust_definitions" in inspect.signature(fn).parameters


def test_cli_prints_the_refusal_verbatim():
    from roshambo3 import cli
    import io, contextlib
    err = io.StringIO()

    def boom(args):
        defs.check_definition(custom_color(), corpus_path="c3")

    ns = cli.build_parser().parse_args(["definitions", "list"])
    cli.build_parser  # noqa
    old = cli._cmd_definitions
    try:
        cli._cmd_definitions = boom
        ns2 = cli.build_parser().parse_args(["definitions", "list"])
        with contextlib.redirect_stderr(err):
            rc = cli.main(["definitions", "list"])
    finally:
        cli._cmd_definitions = old
    assert rc == 1 and err.getvalue().startswith("error: c3 uses a custom color definition")


# ---------------------------------------------------------------- the pre-pass

@pytest.mark.parametrize("name", sorted(cases.CASES))
def test_pre_pass_refuses_even_when_trusted(name):
    make, code = cases.CASES[name]
    color = cases.with_definition(make())
    h = reference.definition_hash(color["definition"])
    t0 = time.monotonic()
    for kw in ({}, {"trusted": [h]}):                      # untrusted, then trusted
        with pytest.raises(CorpusValidationError) as ei:
            defs.check_definition(color, **kw)
        assert ei.value.code == code
    with pytest.raises(CorpusValidationError) as ei:
        defs.perceive(color, ["CCO"], trusted=[h])         # perceive gates too; no subprocess
    assert ei.value.code == code
    assert time.monotonic() - t0 < 10


def test_file_path_reaches_limit_24_directly():
    text = "# " + "x" * 70_000 + "\n" + base_text()
    with pytest.raises(CorpusValidationError) as ei:
        defs.check_definition(cases.with_definition(text))
    assert ei.value.code == "limit.definition_size"


# ----------------------------------------------------------------- the sandbox

def _multiprocessing_helpers() -> set[int]:
    """Pids of multiprocessing's own long-lived helpers, if this process started any.

    A forkserver or resource tracker started by an earlier test (the sqlite-source
    forkserver test) is a child of the pytest process that outlives that test. It
    is not the sandbox's, and it is not a leak.
    """
    from multiprocessing import forkserver, resource_tracker
    pids = {getattr(forkserver._forkserver, "_forkserver_pid", None),
            getattr(resource_tracker._resource_tracker, "_pid", None)}
    return {p for p in pids if p}


def _children_of_this_process():
    """[(pid, status)] of our direct children, or None if they cannot be listed."""
    try:
        import psutil
    except ImportError:                      # dev dependency only
        psutil = None
    if psutil is not None:
        kids = []
        for c in psutil.Process().children(recursive=False):
            try:
                kids.append((c.pid, c.status()))
            except psutil.NoSuchProcess:     # exited and reaped while we looked
                pass
        return kids
    if os.path.isdir("/proc"):
        kids = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat") as f:
                    stat = f.read()
            except OSError:
                continue
            # comm may contain spaces and parentheses; fields follow the last ")".
            state, ppid = stat.rsplit(")", 1)[1].split()[:2]
            if int(ppid) == os.getpid():
                kids.append((int(name), state))
        return kids
    return None


def no_children():
    """Assert the sandbox left no child process behind.

    Children that multiprocessing itself keeps alive are excluded by pid, so this
    stays a check on the sandbox and not on whatever ran before it. A leaked sandbox
    child (live or zombie) is still reported. Without psutil or /proc, fall back to
    waitpid, which cannot tell the helpers apart.
    """
    kids = _children_of_this_process()
    if kids is None:
        with pytest.raises(ChildProcessError):
            os.waitpid(-1, os.WNOHANG)
        return
    ignore = _multiprocessing_helpers()
    kids = [k for k in kids if k[0] not in ignore]
    assert not kids, f"sandbox left child processes (pid, state): {kids}"


def test_perceive_matches_in_process_rdkit():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import ChemicalFeatures
    color = cases.conformance_color()
    smi = ["CCO", cases.IMATINIB, "c1ccccc1C(=O)[O-]", "not a smiles"]
    got = defs.perceive(color, smi)
    fac = ChemicalFeatures.BuildFeatureFactoryFromString(color["definition"])
    tid = {t: i + 1 for i, t in enumerate(color["types"])}
    for s, p in zip(smi, got):
        m = Chem.MolFromSmiles(s)
        if m is None:
            assert p == defs.Perception(False, 0, 0, None)
            continue
        mh = Chem.AddHs(m)
        want = [(tid[f.GetFamily()], tuple(f.GetAtomIds())) for f in fac.GetFeaturesForMol(mh)
                if f.GetFamily() in tid]
        assert p.parsed and p.n_heavy == m.GetNumAtoms() and p.n_atoms == mh.GetNumAtoms()
        assert p.features == want
    no_children()


@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS is not enforced on macOS")
def test_memory_limit_is_named():
    with pytest.raises(defs.DefinitionPerceptionError) as ei:
        defs.perceive(cases.conformance_color(), ["C" * 300_000], in_process=False,
                      limits=_sandbox.Limits(memory_bytes=400 << 20))
    assert ei.value.limit == "memory" and "memory limit" in str(ei.value)
    no_children()


def test_slow_match_is_killed_in_the_subprocess_when_trusted():
    """The pre-pass admits a 16-atom `*~*...` chain; only the subprocess stops it."""
    color = cases.with_definition(cases.any_bond_chain(16))
    h = reference.definition_hash(color["definition"])
    summary = defs.check_definition(color, trusted=[h]).summary          # passes the pre-pass
    assert summary["definefeature_lines"] == 1
    t0 = time.monotonic()
    with pytest.raises(defs.DefinitionPerceptionError) as ei:
        defs.perceive(color, [cases.IMATINIB] * 50, trusted=[h],
                      limits=_sandbox.Limits(cpu_seconds=2, wall_seconds=5, cpu_per_molecule=0, wall_per_molecule=0))
    dt = time.monotonic() - t0
    assert ei.value.limit in ("cpu", "wall") and f"{ei.value.limit} limit" in str(ei.value)
    assert dt < 8
    no_children()


@pytest.mark.skipif(not os.environ.get("ROSHAMBO3_SLOW_TESTS"),
                    reason="runs the 30 s default CPU limit; set ROSHAMBO3_SLOW_TESTS=1")
def test_slow_match_at_the_default_limits():
    color = cases.with_definition(cases.any_bond_chain(16))
    t0 = time.monotonic()
    with pytest.raises(defs.DefinitionPerceptionError) as ei:
        defs.perceive(color, [cases.IMATINIB] * 400,
                      trusted=[reference.definition_hash(color['definition'])])
    # 400 molecules: 50 s CPU / 100 s wall budget. This chain matches >255 times per molecule, so
    # the feature limit may stop it first; any of the three is a clean, named stop.
    assert ei.value.limit in ("cpu", "wall", "features")
    assert time.monotonic() - t0 < 110
    no_children()


def test_wall_clock_kill_reaps_the_group():
    t0 = time.monotonic()
    with pytest.raises(_sandbox.SandboxFailure) as ei:
        _sandbox.run_child({"perceive_any": True, "generator": "rdkit_fdef",
                            "definition": cases.any_bond_chain(16), "types": ["Hydrophobe"],
                            "molecules": [[cases.IMATINIB, True]] * 50},
                           limits=_sandbox.Limits(cpu_seconds=30, wall_seconds=1.5, cpu_per_molecule=0, wall_per_molecule=0))
    assert ei.value.limit == "wall" and time.monotonic() - t0 < 6
    no_children()


def test_reply_is_validated_like_a_corpus_table():
    req = {"types": ["A", "B"], "molecules": [["C", True]]}
    ok = [[True, 1, [[1, [0, 1]]], 5]]
    assert _sandbox._validate_reply(ok, req, 0.0) == ok
    for bad in ([[True, 1, [[3, [0]]], 5]],          # type above n_types
                [[True, 1, [[0, [0]]], 5]],          # type 0
                [[True, 1, [[1, [5]]], 5]],          # index >= n_atoms
                [[True, 1, [[1, []]], 5]],           # empty feature
                [[True, 1, [[1.5, [0]]], 5]],        # not an integer
                [[True, 9, None, 5]],                # n_heavy > n_atoms
                [[True, 1, None]],                   # wrong shape
                []):
        with pytest.raises(_sandbox.SandboxFailure):
            _sandbox._validate_reply(bad, req, 0.0)


def test_deep_tier_accepts_the_conformance_corpora_through_the_sandbox():
    from roshambo3.format.r3g.reader import verify
    r = verify(cases.CONFORMANCE_COLOR, "deep", all_rows=True)
    assert r.sampled_molecules > 0


# ------------------------------------------------------------ review fixes

DRUGS = [cases.IMATINIB, "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCCCN1CCOCC1",
         "CN1CCN(CC1)c1ccc(cc1)Nc1ncc2cc(c(=O)n(C)c2n1)-c1ccccc1"]


def full_batch(n=_sandbox.BATCH):
    return [DRUGS[i % len(DRUGS)] for i in range(n)]


def trusted_copy():
    """The shipped text plus a comment: a different hash, so a custom definition."""
    return cases.with_definition(base_text() + "# trusted copy\n",
                                 types=cases.conformance_color()["types"])


def test_full_batch_of_drug_sized_molecules_shipped_runs_in_process(monkeypatch):
    monkeypatch.setattr(_sandbox.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("a shipped definition must not spawn"))
    res = defs.perceive(cases.conformance_color(), full_batch())
    assert len(res) == _sandbox.BATCH and all(p.parsed and p.features for p in res)


def test_full_batch_under_a_trusted_copy_goes_through_the_sandbox_and_agrees():
    color = trusted_copy()
    h = reference.definition_hash(color["definition"])
    assert not defs.is_shipped(h)
    batch = full_batch()
    t0 = time.monotonic()
    sandboxed = defs.perceive(color, batch, trusted=[h])
    dt = time.monotonic() - t0
    assert len(sandboxed) == len(batch) and dt < 30            # nowhere near the batch budget
    ship = defs.perceive(cases.conformance_color(), batch)      # in-process, packaged text
    assert sandboxed == ship                                    # same text, same answers
    assert _sandbox.last_peak_rss_bytes and _sandbox.last_peak_rss_bytes > 0


def test_budget_scales_with_the_batch():
    lim = _sandbox.Limits()
    assert lim.cpu_seconds + lim.cpu_per_molecule * _sandbox.BATCH > lim.cpu_seconds
    assert _sandbox.BATCH <= 250


def test_in_process_is_for_shipped_only():
    with pytest.raises(ValueError, match="shipped"):
        defs.perceive(trusted_copy(), ["CCO"], in_process=True,
                      trusted=[reference.definition_hash(trusted_copy()["definition"])])


def test_run_child_runs_the_pre_pass_itself(monkeypatch):
    monkeypatch.setattr(_sandbox.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("pre-pass must refuse before any spawn"))
    req = {"perceive_any": True, "generator": "rdkit_fdef", "definition": cases.macro_bomb(),
           "types": ["Hydrophobe"], "molecules": [["CCO", True]]}
    with pytest.raises(CorpusValidationError) as ei:
        _sandbox.run_child(req)
    assert ei.value.code == "limit.definition_depth"


def smarts_chains(L=10, k=6):
    fams = [f"F{i}" for i in range(k)]
    color = cases.with_definition(None, generator="smarts_list", types=fams)
    color["definition"] = {f: ["~".join(["*"] * (L - i))] for i, f in enumerate(fams)}
    return color


def test_feature_explosion_is_stopped_in_the_child():
    """Six 10-atom `*~*` chains pass the pre-pass and match millions of times."""
    color = smarts_chains()
    h = reference.definition_hash(color["definition"])
    t0 = time.monotonic()
    with pytest.raises(defs.DefinitionPerceptionError) as ei:
        defs.perceive(color, [cases.IMATINIB] * 20, trusted=[h])
    assert ei.value.limit == "features" and "features limit" in str(ei.value)
    assert time.monotonic() - t0 < 20
    assert ei.value.peak_rss_bytes < (600 << 20)
    no_children()


def test_reply_is_read_under_a_cap_and_the_child_is_killed(monkeypatch):
    assert _sandbox.max_reply_bytes(250) == 250 * (255 * (16 * 8 + 16) + 48) + 4096
    monkeypatch.setattr(_sandbox, "max_reply_bytes", lambda n: 200)
    with pytest.raises(defs.DefinitionPerceptionError) as ei:
        defs.perceive(cases.conformance_color(), full_batch(20), in_process=False)
    assert ei.value.limit == "reply"
    no_children()


def test_reply_validation_enforces_the_feature_limits():
    req = {"types": ["A"], "molecules": [["C", True]]}
    for bad in ([[True, 1, [[1, [0]]] * 256, 5]],
                [[True, 1, [[1, list(range(17))]], 20]]):
        with pytest.raises(_sandbox.SandboxFailure):
            _sandbox._validate_reply(bad, req, 0.0)
    _sandbox._validate_reply([[True, 1, [[1, list(range(16))]] * 255, 20]], req, 0.0)


def test_molblock_input_preserves_atom_order_and_explicit_hydrogens():
    Chem = pytest.importorskip("rdkit.Chem")
    import random
    from rdkit.Chem import AllChem, ChemicalFeatures
    color = cases.conformance_color()
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    perm = list(range(mol.GetNumAtoms()))
    random.Random(3).shuffle(perm)                       # hydrogens interleaved with heavy atoms
    mol = Chem.RenumberAtoms(mol, perm)
    mb = Chem.MolToMolBlock(mol)
    fac = ChemicalFeatures.BuildFeatureFactoryFromString(color["definition"])
    tid = {t: i + 1 for i, t in enumerate(color["types"])}
    ref = Chem.MolFromMolBlock(mb, removeHs=False)
    want = [(tid[f.GetFamily()], tuple(f.GetAtomIds())) for f in fac.GetFeaturesForMol(ref)
            if f.GetFamily() in tid]
    for in_process in (True, False):
        (p,) = defs.perceive(color, molblocks=[mb], in_process=in_process)
        assert p.parsed and p.n_atoms == mol.GetNumAtoms() and p.features == want
        assert p.n_heavy == sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
    (bad,) = defs.perceive(color, molblocks=["not a molblock"])
    assert bad.parsed is False
    with pytest.raises(ValueError, match="exactly one"):
        defs.perceive(color, ["CCO"], molblocks=[mb])


def test_generator_from_descriptor_refuses_custom_text():
    from roshambo3.format.r3g import descriptor
    with pytest.raises(defs.UntrustedDefinitionError):
        descriptor.generator_from_descriptor(custom_color())
    color = custom_color()
    defs.trust(reference.definition_hash(color["definition"]))
    with pytest.raises(ValueError, match="subprocess"):
        descriptor.generator_from_descriptor(color)
    pytest.importorskip("rdkit")
    assert descriptor.generator_from_descriptor(cases.conformance_color()) is not None


def test_trust_file_that_is_a_directory_or_unreadable_is_a_clean_error(xdg):
    p = defs.trust_file_path()
    _secure_trust_parent(p)
    os.mkdir(p, mode=0o700)                            # the file path is a directory
    with pytest.raises(defs.TrustFileError, match="regular"):
        defs.list_trusted()
    with pytest.raises(defs.TrustFileError):
        defs.trust("ab" * 32)
    os.rmdir(p)
    with open(p, "w", encoding="utf-8"):
        pass
    os.chmod(p, 0o600)
    os.chmod(p, 0)
    try:
        if os.geteuid() != 0:
            with pytest.raises(defs.TrustFileError, match="cannot read"):
                defs.list_trusted()
    finally:
        os.chmod(p, 0o600)


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX permissions")
def test_group_or_world_writable_trust_store_is_refused(xdg):
    h = "ab" * 32
    defs.trust(h)
    p = defs.trust_file_path()
    os.chmod(p, 0o666)
    with pytest.raises(defs.TrustFileError, match="writable"):
        defs.check_definition(custom_color())
    os.chmod(p, 0o600)
    os.chmod(os.path.dirname(p), 0o777)
    with pytest.raises(defs.TrustFileError, match="writable"):
        defs.list_trusted()
    os.chmod(os.path.dirname(p), 0o700)
    assert list(defs.list_trusted()) == [h]


def test_sandbox_environment_cwd_and_rdkit_path(monkeypatch):
    children = []
    real = _sandbox.subprocess.Popen

    def spy(*a, **k):
        # macOS also starts `ps` through subprocess.run while the sandbox
        # parent polls the child's RSS. Capture the isolated RDKit child, not
        # that unrelated helper process.
        argv = a[0]
        if len(argv) >= 4 and argv[1:3] == ["-I", "-c"] and argv[3] == _sandbox._CHILD:
            children.append((argv, k.get("cwd"), k.get("env")))
        return real(*a, **k)

    monkeypatch.setenv("PYTHONPATH", "/nonexistent-evil")
    monkeypatch.setattr(_sandbox.subprocess, "Popen", spy)
    defs.perceive(cases.conformance_color(), ["CCO"], in_process=False)
    assert len(children) == 1
    args, cwd, env = children[0]
    assert cwd == os.path.abspath(os.sep) and "PYTHONPATH" not in env
    assert args[1] == "-I" and os.path.isdir(os.path.join(args[-1], "rdkit"))
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_peak_rss_is_this_childs_own():
    with pytest.raises(_sandbox.SandboxFailure):
        _sandbox.run_child({"perceive_any": True, "generator": "rdkit_fdef",
                            "definition": cases.any_bond_chain(16), "types": ["Hydrophobe"],
                            "molecules": [[cases.IMATINIB, True]] * 5},
                           limits=_sandbox.Limits(cpu_seconds=30, wall_seconds=1,
                                                  cpu_per_molecule=0, wall_per_molecule=0))
    big = _sandbox.last_peak_rss_bytes
    _sandbox.run_child({"perceive_any": False, "generator": None, "definition": None,
                        "types": [], "molecules": [["C", False]]})
    small = _sandbox.last_peak_rss_bytes
    assert big and small and 10 << 20 < small < 1 << 30


@pytest.mark.skipif(not os.environ.get("ROSHAMBO3_SLOW_TESTS"),
                    reason="a full 2,000-molecule shipped batch in the sandbox (~35 s)")
def test_2000_molecule_shipped_batch_in_the_sandbox_is_within_the_default_budget():
    req = {"perceive_any": True, "generator": "rdkit_fdef",
           "definition": cases.conformance_color()["definition"],
           "types": cases.conformance_color()["types"],
           "molecules": [[s, True] for s in full_batch(2000)]}
    assert len(_sandbox.run_child(req)) == 2000

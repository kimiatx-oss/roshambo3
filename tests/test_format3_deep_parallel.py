"""Format 3, ``deep``: the RDKit items run in parallel sandbox children.

The batches are shrunk (``BATCH`` 2, ``PARSE_BATCH`` 3) so a 12-molecule
corpus is many batches. Faults are injected -- an unparsable SMILES, a SMILES
with the wrong heavy-atom count, a wrong feature table, a wrong self-overlap,
and several at once -- and the refusal with ``workers > 1`` must be the serial
one: same code, same path, same message (which names the row and mol_id of
the first failure). Every run passes a fixed seed (``seed=0``).
"""

from __future__ import annotations

import numpy as np
import pytest

from roshambo3.format.r3g import _sandbox, validate as V
from roshambo3.format.r3g import writer as W
from roshambo3.format.r3g.errors import CorpusValidationError

pytest.importorskip("rdkit")

from tests.test_format3_validator import build_rdkit_corpus  # noqa: E402

SMILES12 = ["Nc1ccc(cc1)-c1ccc(N)cc1", "c1cnc[nH]1", "OC(=O)c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCO", "C", "CCN", "c1ccccc1O", "CC(=O)N",
            "OCCO", "c1ccncc1", "CCCCO"]


@pytest.fixture
def small_batches(monkeypatch):
    monkeypatch.setattr(_sandbox, "BATCH", 2)
    monkeypatch.setattr(_sandbox, "PARSE_BATCH", 3)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The correct color corpus's molecules and descriptor, built once."""
    root = tmp_path_factory.mktemp("deep-parallel") / "ok"
    mols, cd = build_rdkit_corpus(root, SMILES12)
    return root, mols, cd


def _write(tmp_path, mols, cd, name):
    root = tmp_path / name
    W.write_corpus(str(root), mols, color=cd, prep_version="t", prep_commit=None, block_rows=3)
    return root


def _outcome(root, workers, all_rows=True):
    try:
        r = V.validate_corpus(str(root), "deep", all_rows=all_rows, seed=0, workers=workers)
        return ("accept", r.sampled_molecules)
    except CorpusValidationError as e:
        return (e.tier, e.code, e.path, str(e))


def _copy(mols):
    out = []
    for m in mols:
        c = W.Molecule(mol_id=m.mol_id, coords=m.coords.copy(), vaa=m.vaa.copy(),
                       features=list(m.features), color_self=None if m.color_self is None
                       else m.color_self.copy(), ext_id=m.ext_id, smiles=m.smiles, mw=m.mw)
        out.append(c)
    return out


def test_the_correct_corpus_passes_with_any_worker_count(built, small_batches):
    root = built[0]
    for workers in (1, 2, 4, 7):
        assert _outcome(root, workers) == ("accept", len(SMILES12))
        assert _outcome(root, workers, all_rows=False) == ("accept", len(SMILES12))


FAULTS = {
    "bad_smiles": [("smiles", 9, "C1CC")],
    "wrong_heavy_atoms": [("smiles", 7, "CCCCN")],
    "wrong_features": [("features", 8, None)],
    "wrong_vaa": [("vaa", 10, None)],
    "wrong_color_self": [("color_self", 9, None)],
    "color_self_then_vaa": [("color_self", 2, None), ("vaa", 7, None)],
    # several faults: the lowest row wins, whatever finishes first
    "two_smiles": [("smiles", 10, "C1CC"), ("smiles", 3, "CCCCCCCO")],
    "features_then_smiles": [("features", 2, None), ("smiles", 11, "C1CC")],
    "smiles_then_features": [("smiles", 4, "C1CC"), ("features", 6, None)],
    # a self-overlap mismatch is deferred behind items 2-3 (FORMAT.md 16.4)
    "vaa_and_smiles": [("vaa", 1, None), ("smiles", 11, "C1CC")],
}


@pytest.mark.parametrize("name", sorted(FAULTS))
def test_a_fault_is_refused_as_the_serial_walk_refuses_it(name, built, small_batches, tmp_path):
    _, mols, cd = built
    mols = _copy(mols)
    for kind, i, val in FAULTS[name]:
        if kind == "smiles":
            mols[i].smiles = val
        elif kind == "features":
            mols[i].features = mols[i].features[1:]
        elif kind == "vaa":
            mols[i].vaa = mols[i].vaa * np.float32(1.001)
        else:
            mols[i].color_self = mols[i].color_self * 1.001
    root = _write(tmp_path, mols, cd, name)
    V.validate_corpus(str(root), "scan")
    serial = _outcome(root, 1)
    assert serial[0] == "deep", serial
    for workers in (2, 3, 8):
        assert _outcome(root, workers) == serial, workers
    want = {"bad_smiles": "limit.smiles", "wrong_heavy_atoms": "limit.smiles",
            "wrong_features": "layout.features", "wrong_vaa": "value.self_overlap",
            "wrong_color_self": "value.self_overlap", "color_self_then_vaa": "value.self_overlap",
            "two_smiles": "limit.smiles", "features_then_smiles": "layout.features",
            "smiles_then_features": "limit.smiles", "vaa_and_smiles": "limit.smiles"}[name]
    assert serial[1] == want, serial
    if name == "two_smiles":
        assert "(mol_id 1003)" in serial[3]
    if name in ("wrong_color_self", "color_self_then_vaa"):
        assert "color_self" in serial[3]


def test_a_failed_large_batch_is_replayed_in_small_batches(monkeypatch, tmp_path):
    """A child that fails as a whole on a large batch (here: forced) is re-run
    in ``BATCH`` pieces, so the row-level refusal of the piece is reported,
    exactly as the 250-molecule serial walk would report it. A shape corpus,
    so no molecule is re-perceived and the batches are ``PARSE_BATCH`` long."""
    mols, _ = build_rdkit_corpus(tmp_path / "shape-ok", SMILES12, color=False, n_conf=1)
    mols[5].smiles = "C1CC"
    root = _write(tmp_path, mols, None, "replay")
    monkeypatch.setattr(_sandbox, "BATCH", 2)
    monkeypatch.setattr(_sandbox, "PARSE_BATCH", 6)
    real = _sandbox.run_child
    sizes = []

    def flaky(req, **kw):
        sizes.append(len(req["molecules"]))
        if len(req["molecules"]) > 2:
            raise _sandbox.SandboxFailure("killed after the 1 s wall-clock limit", limit="wall")
        return real(req, **kw)

    monkeypatch.setattr(_sandbox, "run_child", flaky)
    for workers in (1, 4):
        out = _outcome(root, workers, all_rows=False)
        assert out[1] == "limit.smiles" and "row 5 " in out[3] and "does not parse" in out[3]
    assert max(sizes) == 6 and 2 in sizes


def test_a_whole_batch_failure_without_replay_keeps_the_serial_code(built, monkeypatch, tmp_path):
    """A failure in a batch of at most ``BATCH`` is reported as before:
    ``value.definition`` when the batch re-perceives, else ``limit.smiles``."""
    root = built[0]
    monkeypatch.setattr(_sandbox, "BATCH", 2)
    monkeypatch.setattr(_sandbox, "PARSE_BATCH", 2)

    def dead(req, **kw):
        raise _sandbox.SandboxFailure("killed after the 1 s wall-clock limit", limit="wall")

    monkeypatch.setattr(_sandbox, "run_child", dead)
    for workers in (1, 4):
        assert _outcome(root, workers)[1] == "value.definition"     # --all perceives
        e = _outcome(root, workers, all_rows=False)
        # the first batch (rows 0-1) holds a sampled molecule only by chance;
        # with seed 0 every molecule is sampled (12 < 256), so it perceives
        assert e[1] == "value.definition"


def test_workers_is_validated(built):
    with pytest.raises(ValueError):
        V.validate_corpus(str(built[0]), "deep", workers=0)


# ---------------------------------------------------------------------------
# Section 16.4 item 1: the reference cross-decode sample is per corpus
# ---------------------------------------------------------------------------

DATA = __import__("pathlib").Path(__file__).resolve().parent / "data" / "format3"


def _n_blocks(root) -> int:
    import json
    d = json.loads((root / "corpus.json").read_bytes())
    n = 0
    for g in d["groups"]:
        og = V.open_group(str(root), g)
        n += og.blocks.size
        og.close()
    return n


@pytest.mark.parametrize("variant", ["shape", "color"])
def test_the_reference_sample_is_k_blocks_per_corpus(variant, monkeypatch):
    from roshambo3.format.r3g import codec
    if codec.backend() != "native":
        pytest.skip("the cross-decode runs with the native decoder only")
    root = DATA / variant
    total = _n_blocks(root)
    for seed in (0, 1, 2):
        r = V.validate_corpus(str(root), "deep", seed=seed)
        assert r.reference_sample == V.DEEP_REFERENCE_SAMPLE_BLOCKS
        # a corpus of at most K blocks has every block cross-decoded
        assert total < V.DEEP_REFERENCE_SAMPLE_BLOCKS
        assert r.reference_blocks == total, (seed, r.reference_blocks, total)
    for k in (1, 4, total - 1, total, total + 3):
        monkeypatch.setattr(V, "DEEP_REFERENCE_SAMPLE_BLOCKS", k)
        for seed in range(6):
            r = V.validate_corpus(str(root), "deep", seed=seed)
            assert r.reference_sample == k and r.reference_blocks == min(k, total), (k, seed)
    r = V.validate_corpus(str(root), "deep", all_rows=True, seed=0)
    assert r.reference_sample is None and r.reference_blocks == total


def test_a_native_reference_disagreement_on_a_sampled_block_is_refused(monkeypatch, tmp_path):
    from roshambo3.format.r3g import codec, reference as R
    if codec.backend() != "native":
        pytest.skip("the cross-decode runs with the native decoder only")
    real = R.decode_block
    calls = []

    def off_by_one(data, n_real, L=None, **kw):
        out = real(data, n_real, L, **kw)
        calls.append(1)
        out[0, 0, 0] += 1
        return out

    monkeypatch.setattr(R, "decode_block", off_by_one)
    for all_rows in (False, True):
        with pytest.raises(CorpusValidationError) as e:
            V.validate_corpus(str(DATA / "shape"), "deep", all_rows=all_rows, seed=0)
        assert e.value.code == "internal.error" and "disagree" in str(e.value)
    assert calls


# ---------------------------------------------------------------------------
# Section 16.4 item 4: color_self by the native path, cross-checked
# ---------------------------------------------------------------------------

def test_native_color_self_matches_the_reference_on_every_row(built):
    """On the correct corpus, with every molecule cross-checked (``--all``),
    the native and reference values agree and match the stored ones."""
    r = V.validate_corpus(str(built[0]), "deep", all_rows=True, seed=0)
    assert r.sampled_molecules == len(SMILES12)


@pytest.mark.parametrize("all_rows", [False, True])
def test_a_native_reference_color_self_disagreement_is_refused(built, monkeypatch, all_rows):
    real = V._native_color_self

    def skewed(*a, **kw):
        out = real(*a, **kw)
        return None if out is None else {k: v * 1.01 for k, v in out.items()}

    monkeypatch.setattr(V, "_native_color_self", skewed)
    with pytest.raises(CorpusValidationError) as e:
        V.validate_corpus(str(built[0]), "deep", all_rows=all_rows, seed=0)
    assert e.value.code == "internal.error" and "disagree" in str(e.value)


def test_without_the_cpu_extension_the_reference_computes_every_row(built, monkeypatch, tmp_path):
    from roshambo3.backend import cpu

    def missing():
        raise ImportError("not built")

    monkeypatch.setattr(cpu, "extension", missing)
    assert V.validate_corpus(str(built[0]), "deep", seed=0).sampled_molecules == len(SMILES12)
    _, mols, cd = built
    mols = _copy(mols)
    mols[4].color_self = mols[4].color_self * 1.001
    root = _write(tmp_path, mols, cd, "cs")
    with pytest.raises(CorpusValidationError) as e:
        V.validate_corpus(str(root), "deep", seed=0)
    assert e.value.code == "value.self_overlap" and "color_self" in str(e.value)


def _serial_expected_codes(root, n_sample, batch):
    """For each parquet row: the code the serial walk gives when the child of
    that row's ``batch``-aligned piece fails as a whole -- ``value.definition``
    if the piece holds a re-perceived molecule, else ``limit.smiles``. The
    re-perceived set is derived here independently of the validator: the
    seed-0 draw over molecules in group order, mapped to parquet (mol_id) rows."""
    import json
    d = json.loads((root / "corpus.json").read_bytes())
    order = []
    for g in d["groups"]:
        og = V.open_group(str(root), g)
        order += og.section_array("runs")["mol_id"].astype(np.int64).tolist()
        og.close()
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(order), size=n_sample, replace=False))
    sampled_ids = {order[i] for i in sample}
    rows = sorted(order)                       # parquet row order
    codes = []
    for row in range(len(rows)):
        lo = row - row % batch
        piece = rows[lo:lo + batch]
        codes.append("value.definition" if sampled_ids & set(piece) else "limit.smiles")
    return codes, rows


def test_a_whole_child_failure_in_a_misaligned_sample_gives_the_serial_code(
        built, monkeypatch, tmp_path):
    """Only some of 12 molecules re-perceived (mixed samples of 2-9), batches joined
    from 2-row pieces up to 6 rows or 2 re-perceived molecules: a child that
    fails on any one row's piece is refused with the serial walk's code, for
    every row and worker count (fix: batches stay on the 2-row grid)."""
    root = built[0]
    monkeypatch.setattr(_sandbox, "BATCH", 2)
    monkeypatch.setattr(_sandbox, "PARSE_BATCH", 6)
    import pyarrow.parquet as pq
    smiles = pq.read_table(root / "molecules.parquet", columns=["smiles"]).column(0).to_pylist()
    real = _sandbox.run_child
    starts = []

    def failing_on(target):
        def run_child(req, **kw):
            texts = [t for t, _ in req["molecules"]]
            starts.append(len(texts))
            if smiles[target] in texts:
                raise _sandbox.SandboxFailure("killed after the 1 s wall-clock limit",
                                              limit="wall")
            return real(req, **kw)
        return run_child

    seen = set()
    for n_sample in (2, 3, 5, 7, 9):
        monkeypatch.setattr(V, "DEEP_SAMPLE_MOLECULES", n_sample)
        want, rows = _serial_expected_codes(root, n_sample, 2)
        seen.update(want)
        for target in range(len(rows)):
            monkeypatch.setattr(_sandbox, "run_child", failing_on(target))
            for workers in (1, 4):
                out = _outcome(root, workers, all_rows=False)
                assert out[1] == want[target], (n_sample, target, workers, out)
    assert seen == {"value.definition", "limit.smiles"}
    assert max(starts) > 2          # some batches really were joined

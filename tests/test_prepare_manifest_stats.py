"""The build record's statistics without n-sized copies
are bitwise the ones the copying expressions give. Format 3: they are
in ``build.json``, computed from the group files' run tables and ``vaa``
sections."""
import struct
from pathlib import Path

import numpy as np
import pytest

from roshambo3.prep import build, prepare
from tests.format3_rows import read_corpus
from tests.prep_fixtures import fixtures


def bits(x):
    return struct.pack("<d", float(x))


@pytest.mark.parametrize("n", [1, 2, 9, 128, 129, 8193, 1_000_003, 3_000_001, 5_000_000])
def test_hac_statistics_are_bitwise_numpys(n):
    rng = np.random.default_rng(n)
    nr = rng.integers(1, 90, n).astype(np.uint8)
    got = build._hac_statistics(nr)
    assert got["hac_hist"] == np.bincount(nr.astype(np.int64)).tolist()
    assert bits(got["hac_mean"]) == bits(nr.mean())
    assert bits(got["hac_sd"]) == bits(nr.std())


@pytest.mark.parametrize("n", [100, 1000, 8193, 100_003])
@pytest.mark.parametrize("leaf", [128, 1000, 4096])
def test_pairwise_sum_reproduces_numpy(n, leaf):
    x = np.random.default_rng(n + leaf).standard_normal(n) * 1e3
    assert bits(build._pairwise_sum(lambda a, m: x[a:a + m], 0, n, leaf)) == \
        bits(np.add.reduce(x))
    # The emulation is what makes it exact: plain chunked sums are not.
    differs = 0
    for trial in range(20):
        y = np.random.default_rng(trial).standard_normal(n) * 1e3
        chunked = sum(float(np.add.reduce(y[a:a + leaf])) for a in range(0, n, leaf))
        differs += chunked != float(np.add.reduce(y))
        assert bits(build._pairwise_sum(lambda a, m: y[a:a + m], 0, n, leaf)) == \
            bits(np.add.reduce(y))
    if n > 8 * leaf:
        assert differs > 0


def test_in_place_selection_matches_copying_calls():
    rng = np.random.default_rng(3)
    vaa = np.round(rng.gamma(4.0, 60.0, 1_000_001), 1).astype(np.float32)   # ties
    expected = (float(np.median(vaa)), float(np.quantile(vaa, 0.01)),
                float(np.quantile(vaa, 0.99)))
    work = vaa.copy()
    got = (float(np.median(work, overwrite_input=True)),
           float(np.quantile(work, 0.01, overwrite_input=True)),
           float(np.quantile(work, 0.99, overwrite_input=True)))
    assert [bits(v) for v in got] == [bits(v) for v in expected]


def test_build_statistics_unchanged(tmp_path):
    prepare(fixtures(), str(tmp_path / "corpus"), workers=1,
            records_per_unit=2, max_conf_per_group=2)
    rows = read_corpus(tmp_path / "corpus", merged=False)
    assert len(rows.groups) > 1
    nr, vaa = rows.n_real, rows.vaa
    assert nr.dtype == np.uint8 and vaa.dtype == np.float32
    m = rows.build["statistics"]
    assert m["hac_hist"] == np.bincount(nr.astype(np.int64)).tolist()
    for key, value in (("hac_mean", nr.mean()), ("hac_sd", nr.std()),
                       ("vaa_median", np.median(vaa)),
                       ("vaa_q01", np.quantile(vaa, 0.01)),
                       ("vaa_q99", np.quantile(vaa, 0.99))):
        assert bits(m[key]) == bits(value), key

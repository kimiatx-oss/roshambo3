"""Benchmark input identity and decoded-query preservation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench import apple_silicon
from bench.compare_cpu_port import compare
from tests.codec_fixtures import FEATURES_ENUM, make_codec_fixture
from tests.test_cpu_worker_owned_scan import _query_set


def test_corpus_fingerprint_detects_changed_payload(tmp_path):
    data = tmp_path / "payload.rsb3"
    data.write_bytes(b"first")
    first = apple_silicon.corpus_fingerprint(tmp_path)
    data.write_bytes(b"other")
    assert apple_silicon.corpus_fingerprint(tmp_path)["sha256"] != first["sha256"]


def test_comparison_refuses_unmatched_inputs_before_reading_scores(tmp_path):
    roots = [tmp_path / name for name in ("a", "b")]
    for root, digest in zip(roots, ("one", "two")):
        root.mkdir()
        (root / "provenance.json").write_text(json.dumps(
            {"queries_sha256": digest, "corpus_sha256": "same"}))
    with pytest.raises(ValueError, match="queries_sha256"):
        compare(*roots, 16, 14)


def test_frozen_queries_preserve_decoded_shape_color_and_type_bytes(tmp_path, monkeypatch):
    fixture = make_codec_fixture(tmp_path / "corpus", color_layout="merged")
    expected = _query_set(fixture, (0, 4))
    panel_keys = [(41, 0), (17, 0)]
    monkeypatch.setattr(apple_silicon, "RDKitPharmacophoreGenerator", lambda:
        SimpleNamespace(FEATURES_ENUM=FEATURES_ENUM,
                        interaction_matrix_r=expected.interaction_r,
                        interaction_matrix_p=expected.interaction_p))
    destination = tmp_path / "queries.npz"
    apple_silicon.freeze(fixture.corpus, destination, panel_keys)
    for index in range(2):
        query = apple_silicon.load_query(destination, index)
        for key in ("coords", "types", "n_real", "vaa"):
            np.testing.assert_array_equal(getattr(query, key),
                                          getattr(expected, key)[index:index+1])

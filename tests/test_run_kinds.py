"""A run has a kind, and the kind is checked against the record.

The defect: `enforce()` applied `check_repeats()` unconditionally, so a run
that quoted recall and RBO and no throughput at all was hard-refused for
failing a bar about *timing repeats*. The accuracy run that chose the
shipped grid spacing graded `refused` for exactly that.

The risk in fixing it, which these tests exist to pin: `accuracy` must not
become the escape hatch for any inconvenient timing refusal. So the kind is
checked against the record's contents, not trusted, and an accuracy run is
still graded rather than merely exempted.
"""
from __future__ import annotations

import pytest

from bench import rules


def _ctx(**kw):
    """A context that passes everything not under test."""
    base = dict(
        is_release_build=True, git_dirty=False, gpu_lock_held=True,
        gpu_lock_acquisition_failed=False, other_process_on_gpu=False,
        clocks_locked=True, throttle_fraction=0.0, sustained_kernel_s=60.0,
        measurement_regime="calibration", build_derived=True,
        backend_name="cpu", cpu_isa_selected="AVX2", cpu_vector_width=8,
        conformers_per_molecule=4.0,
        # The CPU extension was inspected and its host flags read -- a
        # gate that fires before the run-kind checks and is not what these
        # tests are about.
        checked_a_cpu_build=True,
        # A stamped build with a content hash -- provenance is NOT a
        # rate-shaped concern and still applies to an accuracy run, so the
        # fixture has to satisfy it rather than the gate being scoped away.
        build_source_sha256="0" * 64, build_source_dirty=False,
    )
    base.update(kw)
    return rules.RunContext(**base)


def _req(**kw):
    base = dict(decision_grade=True, n_conformers=131072, max_results=1000,
                claim_non_burst=False, n_repeats=11, n_warmup=3,
                max_results_applied_per_batch=False)
    base.update(kw)
    return rules.RunRequest(**base)


GOOD_METRICS = {
    "recall_conformer_level": {"mean": 0.99, "per_query": [0.98, 1.0, 0.99]},
    "recall_molecule_level": {"mean": 0.97, "per_query": [0.96, 0.98, 0.97]},
}


# --- the defect itself -----------------------------------------------------

def test_accuracy_run_with_no_timing_is_not_refused_for_a_timing_bar():
    """The accuracy-run case: recall and RBO, no throughput, n_repeats=1."""
    r = rules.enforce(
        _req(kind="accuracy", n_repeats=1, n_warmup=0),
        _ctx(metrics=GOOD_METRICS, sustained_kernel_s=0.0))
    assert r.status != "refused", r.refusal_reason
    assert r.status == "decision", r.advisory_reasons


def test_the_same_run_as_a_rate_is_still_refused():
    """The gate is not weakened -- it is scoped."""
    r = rules.enforce(_req(kind="rate", n_repeats=1, n_warmup=0),
                      _ctx(metrics=GOOD_METRICS))
    assert r.status == "refused"
    assert "timed repeats" in r.refusal_reason


# --- the escape-hatch risk, which is the whole design constraint -----------

@pytest.mark.parametrize("where,block", [
    ("timing", {"start_pairs_per_s": 1.1e6}),
    ("timing", {"n_repeats": 1, "conformers_per_s": 4.2e5}),
    ("metrics", {"arm": {"throughput": 9.9}}),
    ("metrics", {"rows": [{"h": 0.8}, {"h": 1.0, "pairs_per_sec": 3.0}]}),
])
def test_declaring_accuracy_while_carrying_a_rate_is_refused(where, block):
    """Relabelling a refused timing run must change which refusal you get,
    not make it go away."""
    kw = {"metrics": GOOD_METRICS}
    kw[where] = block if where == "timing" else {**GOOD_METRICS, **block}
    r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0), _ctx(**kw))
    assert r.status == "refused"
    assert "declares kind='accuracy'" in r.refusal_reason


def test_an_unknown_kind_is_refused_rather_than_defaulted():
    r = rules.enforce(_req(kind="acuracy"), _ctx(metrics=GOOD_METRICS))
    assert r.status == "refused" and "not one of" in r.refusal_reason


def test_rate_is_the_default_so_existing_callers_keep_their_gates():
    assert _req().kind == rules.RUN_KIND_RATE
    r = rules.enforce(_req(n_repeats=1, n_warmup=0), _ctx())
    assert r.status == "refused" and "timed repeats" in r.refusal_reason


# --- accuracy runs are GRADED, not exempted ---------------------------------

def test_accuracy_run_missing_molecule_level_recall_is_refused():
    r = rules.enforce(
        _req(kind="accuracy"),
        _ctx(metrics={"recall_conformer_level":
                      {"mean": 0.99, "per_query": [1.0, 0.98]}}))
    assert r.status == "refused"
    assert "two-level recall" in r.refusal_reason and "molecule" in r.refusal_reason


def test_accuracy_run_without_a_per_query_table_is_refused():
    r = rules.enforce(
        _req(kind="accuracy"),
        _ctx(metrics={"recall_conformer_level": {"mean": 0.99},
                      "recall_molecule_level": {"mean": 0.97}}))
    assert r.status == "refused" and "per-query table" in r.refusal_reason


def test_a_scalar_named_per_query_does_not_satisfy_the_table_gate():
    """An aggregate wearing the name of a table."""
    r = rules.enforce(
        _req(kind="accuracy"),
        _ctx(metrics={"recall_conformer_level": {"mean": .99, "per_query": .99},
                      "recall_molecule_level": {"mean": .97, "per_query": .97}}))
    assert r.status == "refused" and "per-query table" in r.refusal_reason


@pytest.mark.parametrize("fit,rep", [
    (None, None), ("D2-TUNE-10M", None), (None, "D3-EVAL-10M"),
    ("D3-EVAL-10M", "D3-EVAL-10M"),
])
def test_choosing_a_value_requires_a_real_fit_report_split(fit, rep):
    r = rules.enforce(
        _req(kind="accuracy", chose_a_value="h"),
        _ctx(metrics=GOOD_METRICS, fit_corpus=fit, report_corpus=rep))
    assert r.status == "refused" and "fit/report rule" in r.refusal_reason


def test_a_real_fit_report_split_passes():
    r = rules.enforce(
        _req(kind="accuracy", chose_a_value="h"),
        _ctx(metrics=GOOD_METRICS, fit_corpus="D2-TUNE-10M",
             report_corpus="D3-EVAL-10M"))
    assert r.status == "decision", (r.refusal_reason, r.advisory_reasons)


def test_a_run_that_chose_nothing_needs_no_split():
    r = rules.enforce(_req(kind="accuracy"), _ctx(metrics=GOOD_METRICS))
    assert r.status == "decision"


@pytest.mark.parametrize("isa,width", [(None, 8), ("AVX2", None), (None, None)])
@pytest.mark.parametrize("decision_grade", [True, False])
def test_cpu_accuracy_run_is_covered_by_the_pre_existing_width_gate(
        isa, width, decision_grade):
    """The CPU dispatch-target rule needs no accuracy-specific gate; this pins why.

    The rules already refuse any CPU run that fails to record its ISA or its
    vector width, for every kind and every intent. The kind check deliberately
    does NOT re-check it: the duplicate would be unreachable, and an
    unreachable check is a defect in itself. If someone ever
    narrows that width gate, this test fails and says where to look.
    """
    r = rules.enforce(
        _req(kind="accuracy", decision_grade=decision_grade),
        _ctx(metrics=GOOD_METRICS, cpu_isa_selected=isa,
             cpu_vector_width=width))
    assert r.status == "refused"
    assert "D029" in r.refusal_reason or "dispatch target" in r.refusal_reason


def test_gpu_accuracy_run_is_not_asked_for_a_cpu_width():
    r = rules.enforce(
        _req(kind="accuracy"),
        _ctx(metrics=GOOD_METRICS, backend_name="cuda",
             cpu_isa_selected=None, cpu_vector_width=None))
    assert r.status != "refused", r.refusal_reason


# --- a one-conformer-per-molecule pass is made VISIBLE, not refused --------

def test_one_conformer_per_molecule_corpus_is_advised_not_refused():
    r = rules.enforce(
        _req(kind="accuracy"),
        _ctx(metrics=GOOD_METRICS, conformers_per_molecule=1.0))
    assert r.status == "advisory"
    assert any("satisfied structurally and not exercised" in a
               for a in r.advisory_reasons), r.advisory_reasons


def test_unrecorded_conformers_per_molecule_is_advised():
    r = rules.enforce(
        _req(kind="accuracy"),
        _ctx(metrics=GOOD_METRICS, conformers_per_molecule=None))
    assert r.status == "advisory"
    assert any("indistinguishable" in a for a in r.advisory_reasons)


# --- the detectors, directly ------------------------------------------------

def test_find_rate_fields_reaches_into_lists_of_dicts():
    hits = rules.find_rate_fields(
        {"rows": [{"h": 0.8}, {"h": 1.0, "start_pairs_per_s": 2.0}]})
    assert hits == ["rows[1].start_pairs_per_s"]


def test_find_rate_fields_ignores_non_numeric_and_unrelated_keys():
    assert rules.find_rate_fields(
        {"note": "pairs_per_s is not recorded here", "recall_mean": 0.99}) == []


# --- the escape hatch, after review defeated the first version four ways -----
#
# Each case below is one a reviewer actually got through `enforce()` as an
# `accuracy` run. They are regression tests in the strict sense: every one of
# them graded `decision` before the widening.

@pytest.mark.parametrize("block,payload", [
    # a LIST of rates -- the natural shape for per-repeat data
    ("timing", {"pairs_per_s": [1.1e9]}),
    ("timing", {"per_repeat_pairs_per_s": [1.0, 2.0, 3.0]}),
    # a rate written as a string
    ("timing", {"throughput": "1.13e9 pairs/s"}),
    # key spellings the first regex missed
    ("timing", {"qps": 4.2e5}),
    ("timing", {"gcups": 3.3}),
    ("timing", {"conf_per_sec_": 9.9}),      # trailing _ defeated the $ anchor
    ("timing", {"ns_per_pair": 3.34}),       # an inverse rate is a rate
    ("timing", {"million_pairs_s": 1.1}),
    ("timing", {"speed": 1.0e6}),
    # blocks that were never scanned at all
    ("record_blocks", {"config": {"start_pairs_per_s": 1.1e6}}),
    ("record_blocks", {"audits": {"nested": {"qps": 5.0}}}),
    ("record_blocks", {"artifacts": {"rate_pairs": 7.0}}),
])
def test_a_rate_anywhere_in_the_record_defeats_the_accuracy_label(block, payload):
    kw = {"metrics": GOOD_METRICS, block: payload}
    r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0), _ctx(**kw))
    assert r.status == "refused", f"{block}={payload} escaped the contract"
    assert "declares kind='accuracy'" in r.refusal_reason


def test_prose_is_still_allowed_and_that_boundary_is_deliberate():
    """A record that DESCRIBES a measurement is not one that CARRIES a rate."""
    r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0),
                      _ctx(metrics=GOOD_METRICS,
                           record_blocks={"notes": "we measured pairs per second"}))
    assert r.status == "decision", r.refusal_reason


def test_ordinary_accuracy_fields_are_not_mistaken_for_rates():
    """The widened pattern must not start refusing legitimate records."""
    r = rules.enforce(
        _req(kind="accuracy", n_repeats=1, n_warmup=0),
        _ctx(metrics=GOOD_METRICS,
             record_blocks={"dataset": {"n_conformers": 131072},
                            "config": {"threads": 24, "field_h": 0.8},
                            "operating_point": {"achieved_clock_mhz":
                                                {"median": 4084.3}}}))
    assert r.status == "decision", r.refusal_reason


# --- recording a graded run must not downgrade the next one -----------------

class TestTheLedgerWriteDoesNotDowngradeTheNextRun:
    """`run.write()` appends to a TRACKED ledger file.

    Before this was scoped, that made grading a run dirty the tree and drop the
    next one to `advisory` -- a full run once graded
    advisory on nothing but the ledger entry its predecessor had written. The
    check is right in intent (the binary and the script match a commit) and was
    over-broad in scope: the ledger is a run's OUTPUT, not its input.
    """

    def test_output_paths_are_split_from_source_paths(self):
        from bench.gitinfo import _split_status
        src, out = _split_status(
            " M bench/rules.py\n"
            "?? results/T135/runs.jsonl\n"
            "M  src/roshambo3/numerics.py\n"
            "R  old.py -> results/moved.json")
        assert src == ["bench/rules.py", "src/roshambo3/numerics.py"]
        assert out == ["results/T135/runs.jsonl", "results/moved.json"]

    def test_a_results_only_diff_is_not_source_dirty(self):
        from bench.gitinfo import _split_status
        src, out = _split_status("?? results/T99/runs.jsonl")
        assert not src and out, "a ledger write alone must not read as source dirt"

    def test_the_path_parse_survives_a_stripped_leading_space(self):
        """`_git` strips its output, eating the first line's leading space.

        Fixed offsets shifted by one and silently produced "ench/rules.py",
        which would misclassify any path whose prefix match depends on its
        first character -- including the ledger directory.
        """
        from bench.gitinfo import _split_status
        src, _ = _split_status("M bench/rules.py")     # already stripped
        assert src == ["bench/rules.py"]

    def test_whole_tree_dirtiness_is_still_recorded(self):
        """Narrowing the GRADE must not hide the state from the record."""
        from bench.gitinfo import get_git_info
        d = get_git_info().as_dict()
        assert "dirty" in d and "source_dirty" in d
        assert "output_dirty_paths" in d


# --- round two of the escape hatch, and the false positives it introduced ----
#
# The widened contract was defeated twice more and simultaneously began
# refusing legitimate records. Both directions are pinned here, because a
# contract that cries wolf pushes callers back to hand-written JSON, which is
# a worse outcome than the hole it was closing.

@pytest.mark.parametrize("block,payload", [
    # a summary DICT under the canonical rate key -- this repo's house style
    ("timing", {"pairs_per_s": {"mean": 1.1e9, "std": 2.0e7}}),
    # query_panel was the one caller-writable block left unscanned
    ("record_blocks", {"query_panel": {"per_query_pairs_per_s": [1e9, 1.1e9]}}),
    # duration spellings: median_s counted while mean_s did not
    ("timing", {"mean_s": 0.51}),
    ("timing", {"wall_ms": 12.0}),
    ("timing", {"time_s": 1.0}),
    ("timing", {"runtime_s": 1.0}),
    ("timing", {"total_s": 2.0}),
    ("timing", {"speedup": 1.13}),          # a shipped headline shape
    ("timing", {"mols_per_min": 500}),
    ("timing", {"minutes": 3}),
])
def test_round_two_defeats_are_closed(block, payload):
    kw = {"metrics": GOOD_METRICS, block: payload}
    r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0), _ctx(**kw))
    assert r.status == "refused", f"{block}={payload} escaped"


@pytest.mark.parametrize("payload,why", [
    ({"dataset": {"conformers_per_molecule": 4.0}},
     "the two-level recall rule's own corpus fact; `dataset` is its only honest home, and a "
     "contract that refuses it stops the ledger carrying evidence a sibling "
     "gate requires"),
    ({"config": {"timeout_seconds": 600}}, "a configured duration is an input"),
    ({"config": {"budget_seconds": 30}}, "likewise"),
    ({"config": {"sampling_rate_hz": 10}}, "a sampling frequency is an input"),
    ({"audits": {"tie_rate": 0.0012}},
     "a fraction, and native accuracy vocabulary -- T44 counted 50,737 ties"),
    ({"artifacts": {"rates_csv": "results/T138/rates.csv"}},
     "a pointer, not a carried rate; nearly every path here contains a digit"),
    ({"dataset": {"n_conformers": 131072}}, "a count"),
    ({"notes": "we measured pairs per second"}, "prose describes, it does not carry"),
])
def test_legitimate_records_are_not_refused(payload, why):
    """False positives are worse than the hole: they push callers to json.dump."""
    r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0),
                      _ctx(metrics=GOOD_METRICS, record_blocks=payload))
    assert r.status == "decision", f"{why}\n  refused: {r.refusal_reason}"


# --- round three: the contract refused the repo's own committed ledger -------
#
# v3 flagged `recall_per_query` -- the per-query table METRICS.md REQUIRES --
# as a throughput, so a real accuracy record was refused with a message
# calling its recall table a rate. The strongest possible regression is the
# a representative synthetic record keeps this independent of archived ledgers.

def test_combined_accuracy_tables_are_not_flagged_as_carrying_rates():
    metrics = {"recall_per_query": [0.98, 1.0, 0.99], "recall_mean": 0.99,
               "rbo_ext_mean": 0.97, "trials_per_query": [10, 12, 11],
               "tie_rate": 0.0012}
    assert rules.find_rate_fields(metrics) == []





@pytest.mark.parametrize("key,value", [
    ("recall_per_query", [0.98, 1.0, 0.99]),   # METRICS.md requires this
    ("P_per_query", [0.9, 0.95]),
    ("trials_per_query", [10, 12]),            # a count; `s` of "trials" matched
    ("top1_per_query", [1, 1]),
    ("tie_rate", 0.0012),                      # fractions, decided by magnitude
    ("win_rate", 0.61),
    ("basin_flip_rate", 0.4),
    ("baseline_active_rate", 0.02),
    ("sampling_rate_hz", 10),                  # a configured frequency
    ("poll_rate_hz", 5),
])
def test_accuracy_vocabulary_is_not_a_rate(key, value):
    assert rules.find_rate_fields({key: value}) == [], f"{key} false-positived"


@pytest.mark.parametrize("key", ["time_per_query", "s_per_query", "ns_per_pair"])
def test_unit_scoped_per_query_still_catches_real_rates(key):
    assert rules.find_rate_fields({key: 3.3}), f"{key} escaped"


@pytest.mark.parametrize("key", ["wall_s", "total_ms", "warmup_seconds",
                                 "deadline_seconds", "wall_s_not_a_timing"])
def test_durations_count_in_timing_and_nowhere_else(key):
    """A duration is throughput evidence where a rate run puts its numbers,
    and ordinary bookkeeping everywhere else."""
    assert rules.find_rate_fields({key: 1.0}, durations_count=True)
    assert rules.find_rate_fields({key: 1.0}) == []


@pytest.mark.parametrize("s", ["1.13e9 pairs/s", "~1.13e9 pairs/s", "<0.5 s",
                               " .9e9 pairs/s"])
def test_informally_written_rates_are_caught(s):
    assert rules.find_rate_fields({"throughput": s}), f"{s!r} escaped"


def test_a_mislabelled_number_is_out_of_reach_and_that_is_stated():
    """Detection is by name. This is not anti-tamper, and the file says so."""
    assert rules.find_rate_fields({"x": 1.1e9}) == []


@pytest.mark.parametrize("key,value,flagged", [
    # Percent-spelled fractions: 558 of these are committed, up to 99.99.
    # A magnitude rule that reads [0,1] as "fraction" calls them throughputs.
    ("l1_sector_hit_rate_pct", 99.99, False),
    ("l2_hit_rate_pct", 84.2, False),
    ("l1tex_sector_hit_rate_pct", 12.0, False),
    ("hit_rate_percent", 50.0, False),
    # ...but a "percentage" above 100 is not one.
    ("hit_rate_pct", 150.0, True),
    # Real throughputs under a bare `rate` spelling stay caught.
    ("rate", 8923.0, True),
    ("rate_from_min", 1.653e6, True),
    ("survivor_set_kernel_rate", 3.665e6, True),
])
def test_percent_spelled_fractions_are_not_throughputs(key, value, flagged):
    got = bool(rules.find_rate_fields({key: value}))
    assert got is flagged, f"{key}={value}: expected flagged={flagged}"


# --- round four: four more defeats and two false-positive families ----------
#
# The fifth design changed shape rather than widening again: `timing` became a
# WHITELIST, because four blacklists in a row were each beaten by a spelling
# nobody had enumerated. An unrecognised key there is now a refusal.

@pytest.mark.parametrize("payload", [
    {"pairs_per_s": ["1.13e9 (r1)", "1.15e9 (r2)"]},   # strings in a list
    {"throughput": {"value": "1.13e9 pairs/s"}},       # string in a dict
    {"throughput_hz": 1.13e9},                         # _hz neutralised the name
    {"pairs_per_s_hz": 1.1e9},
    {"mean_qps": 43000},                               # \b cannot cross an _
    {"qps_mean": 43000},
    {"search_qps": 1.0e4},
])
def test_round_four_detector_defeats_are_closed(payload):
    r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0),
                      _ctx(metrics=GOOD_METRICS, record_blocks={"audits": payload}))
    assert r.status == "refused", f"{payload} escaped"


@pytest.mark.parametrize("payload,why", [
    ({"tie_rate": {"value": 0.0051, "n_ties": 50737}},
     "a fraction summarized beside its count -- 50,737 is this repo's own T44 "
     "tie count, and counting it in the magnitude test made the fraction a rate"),
    ({"win_rate": {"mean": 0.62, "n": 48}}, "same shape"),
    ({"win_rate": "0.62"}, "a fraction written as a string"),
    ({"sm_clock_hz": 1.4e9}, "a clock frequency, not work per second"),
])
def test_round_four_false_positives_are_closed(payload, why):
    r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0),
                      _ctx(metrics=GOOD_METRICS, record_blocks={"audits": payload}))
    assert r.status == "decision", f"{why}\n  refused: {r.refusal_reason}"


class TestTimingIsAWhitelistForAccuracyRuns:
    """The shape change. An unrecognised key in `timing` is a refusal.

    This is what closes the spelling game permanently: `kernel_s`, `h2d_s`,
    `launch_s` and the rest of this file's own PER_LAUNCH_STAGES vocabulary
    escaped every blacklist, and so would the next name nobody thought of.
    """

    @pytest.mark.parametrize("key", ["kernel_s", "h2d_s", "launch_s", "alloc_s",
                                     "precompute_s", "free_s", "latency_ms",
                                     "score_s", "start_pairs_per_s", "wall_s",
                                     "some_name_nobody_enumerated"])
    def test_any_unrecognised_timing_key_refuses(self, key):
        r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0),
                          _ctx(metrics=GOOD_METRICS, timing={key: 1.0}))
        assert r.status == "refused"
        assert "`timing` block" in r.refusal_reason

    def test_the_allowed_bookkeeping_keys_pass(self):
        r = rules.enforce(_req(kind="accuracy", n_repeats=1, n_warmup=0),
                          _ctx(metrics=GOOD_METRICS,
                               timing={"n_repeats": 1, "n_warmup": 0}))
        assert r.status == "decision", r.refusal_reason

    def test_a_rate_run_keeps_its_full_timing_block(self):
        """The whitelist binds accuracy runs only."""
        # cpu_lock_held: the lock gate is kind-scoped, so it never fires on the
        # accuracy fixtures above and the fixture omits it. A rate run needs it.
        r = rules.enforce(_req(kind="rate"),
                          _ctx(metrics=GOOD_METRICS, cpu_lock_held=True,
                               cpu_affinity=[0], achieved_cpu_mhz={"n_samples": 9},
                               timing={"kernel_s": 1.0, "start_pairs_per_s": 1e6}))
        assert r.status == "decision", r.refusal_reason


def test_rate_kind_with_recall_rbo_and_topk_fields_keeps_rate_only_grade():
    """Supplemental accuracy metrics do not bind the accuracy-kind gates."""
    metrics = {"recall_mean": 0.99, "rbo_ext_mean": 0.97,
               "topk_set_agreement": 0.96}
    rate_ctx = _ctx(metrics=metrics, cpu_lock_held=True,
                    cpu_affinity=[0], achieved_cpu_mhz={"n_samples": 9})
    r = rules.enforce(_req(kind="rate"), rate_ctx)
    assert r.status == "decision", r.refusal_reason

    # The same content is an accuracy-kind run if the caller says that is the
    # number being graded; it then reaches the accuracy preconditions.
    accuracy = rules.enforce(_req(kind="accuracy"),
                             _ctx(metrics=metrics, timing={}))
    assert accuracy.status == "refused"
    assert "two-level recall" in accuracy.refusal_reason


def test_combined_rate_record_does_not_imply_two_level_accuracy():
    """Supplemental metrics never silently promote a rate run to accuracy."""
    record = {"status": "decision", "metrics": {"recall_mean": 0.99,
              "rbo_ext_mean": 0.97, "topk_set_agreement": 0.96},
              "timing": {"start_pairs_per_s": 1e6}}
    metrics = record["metrics"]
    levels = rules.find_recall_levels(metrics)
    assert record["status"] == "decision"
    assert metrics["rbo_ext_mean"] > 0.0 and metrics["recall_mean"] > 0.0
    assert not (levels["conformer"] and levels["molecule"])

    rate = rules.enforce(_req(kind="rate"),
        _ctx(metrics=metrics, timing=record.get("timing"),
             conformers_per_molecule=4.0, cpu_lock_held=True,
             cpu_affinity=[0], achieved_cpu_mhz={"n_samples": 9}))
    assert rate.status == record["status"]

    # Pin the other direction: asking the harness to grade these same metrics
    # as accuracy still applies the existing accuracy gates.
    accuracy = rules.enforce(_req(kind="accuracy"),
                             _ctx(metrics=metrics, timing={}))
    assert accuracy.status == "refused"
    assert "two-level recall" in accuracy.refusal_reason

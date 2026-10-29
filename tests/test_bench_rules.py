"""The harness's enforced rules.

The failure mode being defended against is not a wrong number -- it is a
*plausible* number that nobody can reproduce or compare. So these tests feed the
rule engine non-compliant runs and assert it refuses them, in the same spirit as
`bench/demo_refuse.py` but without needing a GPU.
"""

from __future__ import annotations

import pathlib

import pytest

from bench import rules


def _ctx(**kw):
    d = dict(is_release_build=True, git_dirty=False, gpu_lock_held=True,
             gpu_lock_acquisition_failed=False, other_process_on_gpu=False,
             clocks_locked=False, throttle_fraction=0.0,
             sustained_kernel_s=120.0,
             # Required, not defaulted, on the real dataclass --
             # every test below that does not itself vary this field needs a
             # valid value here so it is testing what it says it is testing.
             measurement_regime="calibration",
             # Every test below varies one *other* fact, and a context that
             # never inspected its build is refused before any of them are
             # reached. TestAContextThatNeverInspectedItsBuild covers that rule.
             build_derived=True)
    d.update(kw)
    return rules.RunContext(**d)


def _req(**kw):
    d = dict(decision_grade=True, n_conformers=2_000_000,
             max_results=20_000_000, claim_non_burst=True,
             n_repeats=11, n_warmup=3)
    d.update(kw)
    return rules.RunRequest(**d)


class TestHardRefusals:
    def test_a_compliant_run_is_accepted(self):
        """The tests below are only meaningful if the baseline passes."""
        r = rules.enforce(_req(), _ctx())
        assert not r.hard_refuse, r.refusal_reason

    def test_a_non_release_build_is_refused(self):
        """A bare `cmake ..` leaves host code at -O0 with asserts on, which
        silently invalidates any timing."""
        r = rules.enforce(_req(), _ctx(is_release_build=False))
        assert r.hard_refuse and "Release" in r.refusal_reason

    def test_a_build_with_no_cuda_extension_is_refused(self):
        """`check_build().ok` means "found no forbidden flag", which is
        vacuously true when there is nothing to inspect -- and the harness reads
        exactly that. A timed GPU number from a build with no GPU code cannot
        exist."""
        r = rules.enforce(_req(), _ctx(checked_a_gpu_build=False))
        assert r.hard_refuse
        assert "no CUDA extension" in r.refusal_reason
        assert "vacuously" in r.refusal_reason

    def test_a_forbidden_numerics_flag_is_refused(self):
        """Every perturbing build axis reaches the same output distribution, so
        there is no small-perturbation regime to tolerate."""
        r = rules.enforce(_req(), _ctx(numerics_violations=["--use_fast_math"]))
        assert r.hard_refuse and "pinned arithmetic" in r.refusal_reason

    def test_the_gpu_build_rule_is_independent_of_the_release_rule(self):
        """Both must fire on their own, or one masks the other."""
        assert rules.enforce(_req(), _ctx(checked_a_gpu_build=False)).hard_refuse
        assert rules.enforce(_req(), _ctx(is_release_build=False)).hard_refuse
        both = rules.enforce(_req(), _ctx(is_release_build=False,
                                          checked_a_gpu_build=False))
        assert both.hard_refuse

    def test_uninspected_host_flags_are_an_advisory_not_a_refusal(self):
        """`check_cxx_flags` returns [] both when it inspected the host flags and
        found none of SUSPECT_CXX_FLAGS and when the stamp recorded none, so
        the harness is told separately which happened. Advisory rather than
        fatal because -- unlike the six nvcc flags -- the host-flag effect has
        never been measured here, and the policy does not claim a number it has
        not taken.
        """
        r = rules.enforce(_req(), _ctx(checked_host_flags=False))
        assert not r.hard_refuse
        assert r.status == "advisory"
        assert any("host compiler flags" in a for a in r.advisory_reasons)

    def test_a_build_whose_host_flags_were_inspected_says_nothing(self):
        """An advisory that fires on every compliant build is one people learn
        to ignore."""
        r = rules.enforce(_req(), _ctx())
        assert not any("host compiler flags" in a for a in r.advisory_reasons)


class TestMeasurementRegimeIsRequired:
    """Calibration and headline must not be mixed silently,
    so the field carries no default at all -- a RunContext cannot even be
    constructed without stating one."""

    def test_the_dataclass_has_no_default(self):
        """Constructing a RunContext with every OTHER field, but omitting
        measurement_regime, must fail before rules.enforce() is ever
        reached -- the whole point is that this cannot be
        forgotten and silently defaulted."""
        with pytest.raises(TypeError):
            rules.RunContext(
                is_release_build=True, git_dirty=False, gpu_lock_held=True,
                gpu_lock_acquisition_failed=False, other_process_on_gpu=False,
                clocks_locked=False, throttle_fraction=0.0,
                sustained_kernel_s=120.0, build_derived=True,
            )

    def test_an_invalid_regime_value_is_refused(self):
        """A typo (or an old caller passing a value never defined)
        must be caught by enforce(), not silently accepted as a third
        regime."""
        r = rules.enforce(_req(), _ctx(measurement_regime="unlocked"))
        assert r.hard_refuse
        assert "measurement_regime" in r.refusal_reason

    def test_calibration_and_headline_are_both_valid(self):
        for regime in ("calibration", "headline"):
            r = rules.enforce(_req(), _ctx(measurement_regime=regime))
            assert not r.hard_refuse, r.refusal_reason


class TestCrossRegimeComparisonIsRefused:
    """A comparison spanning both regimes is a hard refusal, not merely an
    advisory tag: the two must never be combined, with no
    material-difference threshold the way the clock-ratio check has."""

    def test_a_known_regime_mismatch_trips_the_refusal(self):
        """A known mismatch trips the refusal."""
        r = rules.enforce(_req(), _ctx(
            measurement_regime="headline",
            gold_ref_run_id="run_calibration_example",
            gold_ref_measurement_regime="calibration",
        ))
        assert r.hard_refuse
        assert "measurement_regime" in r.refusal_reason
        assert "headline" in r.refusal_reason and "calibration" in r.refusal_reason

    def test_the_same_regime_on_both_sides_is_not_refused(self):
        r = rules.enforce(_req(), _ctx(
            measurement_regime="headline",
            gold_ref_run_id="run_headline_example",
            gold_ref_measurement_regime="headline",
        ))
        assert not r.hard_refuse

    def test_an_unrecorded_gold_ref_regime_is_advisory_not_refused(self):
        """A gold-ref run that predates the field never recorded a regime at
        all -- that is a real gap, flagged, but not the same claim as a
        KNOWN mismatch, so it must not hard-refuse."""
        r = rules.enforce(_req(), _ctx(
            measurement_regime="calibration",
            gold_ref_run_id="run_pre_d023_example",
            gold_ref_measurement_regime=None,
        ))
        assert not r.hard_refuse
        assert any("measurement_regime is not recorded" in a for a in r.advisory_reasons)

    def test_check_regime_comparable_directly(self):
        assert rules.check_regime_comparable("calibration", None)
        assert rules.check_regime_comparable("calibration", "calibration")
        assert not rules.check_regime_comparable("calibration", "headline")


class TestCrossDeviceComparisonIsRefused:
    """Refuse a cross-device comparison alongside the
    existing check_arch_profile_comparable, generalizing the same pattern
    used for clocks and arch profiles."""

    def test_a_known_device_mismatch_trips_the_refusal(self):
        r = rules.enforce(_req(), _ctx(
            gpu_uuid="GPU-aaaa",
            gold_ref_run_id="run_other_card_example",
            gold_ref_gpu_uuid="GPU-bbbb",
        ))
        assert r.hard_refuse
        assert "GPU-aaaa" in r.refusal_reason and "GPU-bbbb" in r.refusal_reason

    def test_the_same_device_on_both_sides_is_not_refused(self):
        r = rules.enforce(_req(), _ctx(
            gpu_uuid="GPU-aaaa",
            gold_ref_run_id="run_same_card_example",
            gold_ref_gpu_uuid="GPU-aaaa",
        ))
        assert not r.hard_refuse

    def test_no_gold_ref_device_recorded_is_not_refused(self):
        """No comparison is being made about devices at all -- distinct
        from a comparison whose device is unknown, mirrored on
        check_arch_profile_comparable's own model."""
        r = rules.enforce(_req(), _ctx(gpu_uuid="GPU-aaaa"))
        assert not r.hard_refuse

    def test_check_device_comparable_directly(self):
        assert rules.check_device_comparable("GPU-aaaa", None)
        assert rules.check_device_comparable("GPU-aaaa", "GPU-aaaa")
        assert not rules.check_device_comparable("GPU-aaaa", "GPU-bbbb")


class TestAdvisories:
    def test_a_run_that_did_not_hold_the_lock_is_not_decision_grade(self):
        """A run that finds the card busy and proceeds is advisory, never a
        number anyone may cite."""
        r = rules.enforce(_req(), _ctx(gpu_lock_held=False))
        assert r.hard_refuse or r.advisory_reasons

    def test_another_process_on_the_gpu_is_flagged(self):
        r = rules.enforce(_req(), _ctx(other_process_on_gpu=True))
        assert r.hard_refuse or r.advisory_reasons

    def test_a_throttled_run_is_flagged(self):
        r = rules.enforce(_req(), _ctx(throttle_fraction=0.9))
        assert r.advisory_reasons or r.hard_refuse

    def test_a_failed_ground_truth_audit_is_flagged(self):
        r = rules.enforce(_req(), _ctx(ground_truth_audit_passed=False))
        assert r.advisory_reasons
        assert any("ground-truth" in a or "audit" in a for a in r.advisory_reasons)

    def test_a_ground_truth_audit_that_did_not_run_is_not_a_problem(self):
        """None means "no audit ran", which is not the same as a failed one."""
        base = rules.enforce(_req(), _ctx())
        r = rules.enforce(_req(), _ctx(ground_truth_audit_passed=None))
        assert len(r.advisory_reasons) == len(base.advisory_reasons)


class TestMeasurementShape:
    def test_a_batch_too_small_to_measure_is_rejected(self):
        """Below about a million conformers per launch the GPU is not being
        measured; N=1e4 under-reports by 5x."""
        r = rules.enforce(_req(n_conformers=10_000), _ctx())
        assert r.hard_refuse or r.advisory_reasons

    def test_max_results_that_truncates_the_measurement_is_rejected(self):
        """max_results silently truncating what is being measured, applied per
        batch before merging, invalidates it."""
        r = rules.enforce(_req(n_conformers=2_000_000, max_results=1_000), _ctx())
        assert r.hard_refuse or r.advisory_reasons

    def test_device_topk_does_not_inherit_legacy_per_batch_truncation(self):
        """Device top-K sees the whole batch before retaining K."""
        r = rules.enforce(_req(n_conformers=2_000_000, max_results=1_000,
                               max_results_applied_per_batch=False), _ctx())
        assert not r.hard_refuse
        assert not any("max_results" in reason for reason in r.advisory_reasons)

    def test_too_few_repeats_is_rejected(self):
        r = rules.enforce(_req(n_repeats=1, n_warmup=0), _ctx())
        assert r.hard_refuse or r.advisory_reasons


class TestConstantsAreStated:
    def test_the_thresholds_are_named_constants_not_literals(self):
        """A rule whose threshold is a literal buried in a branch is a rule
        nobody can find or cite."""
        for name in ("MIN_CONFORMERS_PER_LAUNCH", "MIN_SUSTAINED_KERNEL_S",
                     "REQUIRED_TIMED_REPEATS", "MIN_WARMUP_REPEATS",
                     "THROTTLE_ADVISORY_THRESHOLD"):
            assert hasattr(rules, name), name


class TestDemoRefuse:
    """The demo scenarios must actually run.

    They are the harness's own evidence that each rule refuses something, and
    they exercise code paths nothing else does -- `verify_build`, the run-record
    writer, the lock. Two of them were silently broken by a rename until this
    test existed: `threading.Thread(target=...)` and `mp.Process(target=...)`
    had their keyword renamed along with the project's `target` -> `library`
    pass, which no import check would catch because the failure is at call time.
    """

    def test_every_scenario_runs_and_lands_in_a_state(self, tmp_path):
        from bench.demo_refuse import run_all_demos

        results = run_all_demos(tmp_path)
        assert len(results) >= 9
        for r in results:
            rec = r["record"]
            assert rec["status"] in ("refused", "advisory", "decision"), rec["status"]
            if rec["status"] == "refused":
                assert rec["refusal_reason"], r["kind"]
            if rec["status"] == "advisory":
                assert rec["advisory_reasons"], r["kind"]

    def test_the_no_gpu_build_scenario_is_present_and_refuses(self, tmp_path):
        """The no-GPU-build rule has its own scenario."""
        from bench.demo_refuse import run_all_demos

        by_kind = {r["kind"]: r for r in run_all_demos(tmp_path)}
        assert "no_gpu_build" in by_kind
        assert by_kind["no_gpu_build"]["record"]["status"] == "refused"


class TestVerifyBuild:
    def test_it_inspects_the_loaded_extension_not_the_source_tree(self):
        """The highest-yield mistake here was timing a library
        compiled for the wrong architecture and JIT-ing at runtime, which a
        build stamp cannot detect. Under an editable install the .so and the
        Python package are in different directories, so globbing beside the
        package silently reports no architectures at all -- which is how this
        was found."""
        from bench.build_check import verify_build
        from roshambo3.backend import cuda as backend

        b = verify_build()
        if not backend.available():
            assert not b.with_cuda and b.arch_list == []
            return
        assert b.with_cuda
        if b.arch_probe_error:
            # The probe reports why it could not run, rather than an empty list
            # that reads as "this build has no SASS". cuobjdump is not on PATH
            # under a default NVIDIA apt install, which is how this was found.
            pytest.skip(f"arch probe unavailable: {b.arch_probe_error}")
        assert b.arch_list, "no architectures read from the loaded .so"
        assert b.so_sha256, "the loaded .so was not hashed"

    def test_it_does_not_raise_without_an_extension(self):
        """A refused record on a machine with no GPU build is exactly what the
        harness must be able to write, so this must never raise."""
        from bench.build_check import verify_build
        assert verify_build() is not None


class TestAnArchProbeThatCouldNotRun:
    """``cuobjdump`` is not on ``PATH`` under a default NVIDIA apt install, and the
    probe returned an empty architecture list -- indistinguishable from
    "probed, and this build has no SASS". A wrong-architecture build is the
    highest-yield mistake in this project (a library compiled for the wrong architecture that
    JIT-compiles at runtime, which a build stamp cannot see), and this probe is
    what catches it.
    """

    def test_a_run_that_could_not_check_pitfall_1_is_advisory(self):
        r = rules.enforce(_req(), _ctx(arch_probe_error="cuobjdump is not on PATH"))
        assert not r.hard_refuse
        assert r.status == "advisory"
        assert any("wrong-architecture" in a for a in r.advisory_reasons)

    def test_a_run_that_could_check_is_not_degraded_by_this(self):
        r = rules.enforce(_req(), _ctx(arch_probe_error=None))
        assert not any("wrong-architecture" in a for a in r.advisory_reasons)

    def test_the_probe_says_why_rather_than_returning_an_empty_list(self, monkeypatch):
        import shutil

        from bench import build_check
        monkeypatch.setattr(build_check.shutil, "which", lambda _n: None)
        archs, ptx, why = build_check._cuobjdump_arch_info(
            pathlib.Path("/nonexistent.so"))
        assert archs == [] and ptx is False
        assert why and "PATH" in why


class TestAContextThatNeverInspectedItsBuild:
    """The defect that made three rules unreachable on any real run.

    ``RunContext``'s build-dependent fields each default to the permissive
    answer -- a GPU build was checked, no forbidden flag was found, the
    architecture probe ran. Every ROSHAMBO2-era harness script forwarded
    ``arch_profile`` and none of the rest, so the no-GPU-build refusal, the
    pinned-arithmetic refusal and the arch-probe advisory could fire only from
    tests and demos that set the fields by hand. The tests passed; the pipeline
    was dead.

    The rule is about provenance rather than about any one field: a
    decision-grade context that did not come from ``from_build`` is refused,
    because nothing established the facts it is asserting.
    """

    def test_a_decision_grade_run_is_refused_without_a_build(self):
        r = rules.enforce(_req(decision_grade=True),
                          _ctx(build_derived=False))
        assert r.hard_refuse
        assert "not derived from an inspected build" in r.refusal_reason

    def test_an_advisory_run_is_not_refused_for_it(self):
        """Exploratory numbers are allowed to be assembled by hand; they are
        not citable in the first place."""
        r = rules.enforce(_req(decision_grade=False),
                          _ctx(build_derived=False))
        assert not r.hard_refuse

    def test_from_build_sets_every_build_dependent_field(self):
        from bench.build_check import VerifiedBuild
        from roshambo3.numerics import ArithmeticCheck

        b = VerifiedBuild(
            build_type="Release", cxx_flags="-O3", cuda_flags="-O3",
            with_cuda=True, cuda_architectures_requested="90",
            nvcc_version="12.9", git_sha="abc", arch_list=["sm_90"],
            ptx_present=True, so_sha256={"_roshambo3_cuda": "d"},
            arch_probe_error="cuobjdump is not on PATH")
        chk = ArithmeticCheck(violations=["--use_fast_math"],
                              advisories=["toolkit 12.9"], _with_cuda=True)
        ctx = rules.RunContext.from_build(
            b, chk, git_dirty=False, gpu_lock_held=True,
            gpu_lock_acquisition_failed=False, other_process_on_gpu=False,
            clocks_locked=False, throttle_fraction=0.0,
            sustained_kernel_s=120.0, measurement_regime="calibration")

        assert ctx.build_derived
        assert ctx.is_release_build
        assert ctx.arch_profile == "native"
        assert ctx.arch_probe_error == "cuobjdump is not on PATH"
        assert ctx.checked_a_gpu_build
        # `from_build` is the only place this is derived, for the
        # same reason as the three above -- a field-by-field copy at each call
        # site is exactly how all of them came to be dropped at once.
        assert not ctx.checked_host_flags, (
            "ArithmeticCheck() defaults _host_flags_seen False; from_build must "
            "carry that through rather than take RunContext's permissive default")
        assert ctx.numerics_violations == ["--use_fast_math"]
        # And the verdict actually follows from the build, rather than from
        # whatever the caller remembered to type.
        assert rules.enforce(_req(), ctx).hard_refuse

    def test_from_build_refuses_to_let_a_caller_override_the_build(self):
        """The whole point is that these come from one place."""
        from bench.build_check import VerifiedBuild

        b = VerifiedBuild("Release", "-O3", "-O3", True, "90", "12.9", "abc")
        with pytest.raises(TypeError, match="inspected build"):
            rules.RunContext.from_build(
                b, None, is_release_build=False, git_dirty=False,
                gpu_lock_held=True, gpu_lock_acquisition_failed=False,
                other_process_on_gpu=False, clocks_locked=False,
                throttle_fraction=0.0, sustained_kernel_s=120.0)


class TestTheBuildReachesTheRecord:
    """End to end: verify_build() -> from_build() -> enforce() -> the JSONL.

    Tested as a pipeline because each half worked in isolation and the join
    did not. ``VerifiedBuild.as_dict()`` was a hand-written literal that dropped
    ``arch_probe_error``, and no call site passed it into the context -- so the
    failure was swallowed twice: once on the way to the record, once on the way
    to the rules.
    """

    def test_every_field_is_serialized_or_deliberately_not(self):
        import dataclasses

        from bench.build_check import VerifiedBuild

        b = VerifiedBuild("Release", "-O3", "-O3", True, "90", "12.9", "abc")
        emitted = set(b.as_dict())
        for f in dataclasses.fields(b):
            assert (f.name in emitted
                    or f.name in VerifiedBuild._NOT_SERIALIZED), (
                f"{f.name} is neither serialized nor in _NOT_SERIALIZED; a "
                f"field that is silently dropped is how arch_probe_error was "
                f"lost")

    def test_an_unreadable_arch_probe_reaches_the_written_record(self, tmp_path):
        import json

        from bench.build_check import VerifiedBuild
        from bench.gitinfo import GitInfo
        from bench.run_record import append_jsonl, build_run_record
        from roshambo3.numerics import ArithmeticCheck

        build = VerifiedBuild(
            build_type="Release", cxx_flags="-O3", cuda_flags="-O3",
            with_cuda=True, cuda_architectures_requested="90",
            nvcc_version="12.9", git_sha="abc", arch_list=[],
            ptx_present=False, so_sha256={"_roshambo3_cuda": "d"},
            arch_probe_error="cuobjdump is not on PATH")
        ctx = rules.RunContext.from_build(
            build, ArithmeticCheck(_with_cuda=True),
            git_dirty=False, gpu_lock_held=True,
            gpu_lock_acquisition_failed=False, other_process_on_gpu=False,
            clocks_locked=False, throttle_fraction=0.0,
            sustained_kernel_s=120.0, measurement_regime="calibration")
        res = rules.enforce(_req(), ctx)

        assert not res.hard_refuse
        assert res.status == "advisory"
        assert any("wrong-architecture" in a for a in res.advisory_reasons)

        rec = build_run_record(
            ticket_id="T52",
            git=GitInfo(sha="abc", dirty=False, branch="b"),
            build=build, clocks_locked=False, gpu_lock_held=True,
            dataset={"n": 1}, query_panel={"name": "n/a"}, config={},
            timing={}, metrics={}, audits={}, artifacts={},
            status=res.status, refusal_reason=res.refusal_reason,
            advisory_reasons=res.advisory_reasons)
        path = tmp_path / "runs.jsonl"
        append_jsonl(rec, path)

        on_disk = json.loads(path.read_text().strip())
        assert on_disk["build"]["arch_probe_error"] == "cuobjdump is not on PATH"
        assert any("wrong-architecture" in a for a in on_disk["advisory_reasons"])


class TestPaddingComparability:
    """Two configurations compared for throughput must have identical L
    histograms, or the comparison is between data layouts, not kernels. Clock,
    architecture profile, regime and device are checked elsewhere; this is the
    one rule about the LIBRARY's shape.
    """

    SPLIT = [{"bucket": 0, "L": 16, "L_color": 16, "n_conf": 100},
             {"bucket": 1, "L": 24, "L_color": 32, "n_conf": 200}]
    MERGED = [{"bucket": 0, "L": 24, "L_color": 0, "n_conf": 100},
              {"bucket": 1, "L": 48, "L_color": 0, "n_conf": 200}]

    def test_no_gold_ref_is_not_a_problem(self):
        assert rules.check_padding_comparable(self.SPLIT, None)

    def test_an_unrecorded_gold_ref_padding_is_not_comparable(self):
        """Follows check_clock_comparable's model, not the arch profile's: "a
        comparison is being made and one side's padding was never recorded" is
        the same shape of failure as an unrecorded clock, not an absence of a
        claim to check."""
        assert not rules.check_padding_comparable(None, self.SPLIT)

    def test_the_same_histogram_compares_equal(self):
        assert rules.check_padding_comparable(self.SPLIT, list(self.SPLIT))

    def test_a_different_histogram_does_not(self):
        assert not rules.check_padding_comparable(self.MERGED, self.SPLIT)

    def test_how_a_bucket_was_split_into_groups_does_not_matter(self):
        """The 2e6-conformer group cap is a property of the builder, not of the
        layout the kernel sees, so two corpora that differ only in how many
        groups a bucket became must compare equal."""
        one = [{"bucket": 3, "L": 56, "L_color": 0, "n_conf": 2_000_000}]
        two = [{"bucket": 3, "L": 56, "L_color": 0, "n_conf": 1_000_000},
               {"bucket": 3, "L": 56, "L_color": 0, "n_conf": 1_000_000}]
        assert rules.check_padding_comparable(two, one)

    def test_the_canonical_form_round_trips_through_json(self):
        """A stored table is read back as lists, and it has to compare equal to
        a freshly built one or every comparison against a recorded run fails
        for the wrong reason."""
        import json
        t = rules.canonical_padding_table(self.SPLIT)
        assert json.loads(json.dumps(t)) == t

    def test_tuples_and_mappings_canonicalize_alike(self):
        assert (rules.canonical_padding_table(self.SPLIT)
                == rules.canonical_padding_table(
                    [(r["bucket"], r["L"], r["L_color"], r["n_conf"])
                     for r in self.SPLIT]))

    def test_enforce_refuses_an_undeclared_padding_change(self):
        res = rules.enforce(
            _req(),
            _ctx(gold_ref_run_id="run_before",
                 gold_ref_measurement_regime="calibration",
                 padding_table=self.MERGED, gold_ref_padding_table=self.SPLIT))
        assert res.hard_refuse
        assert "padded-width histogram" in res.refusal_reason

    def test_a_declared_padding_change_is_an_advisory_not_a_refusal(self):
        """An A/B arm may change L on purpose. Declaring it does not make
        the difference disappear -- the tables are in the record either way --
        it makes the refusal a tag."""
        res = rules.enforce(
            _req(padding_change_is_the_subject=True),
            _ctx(gold_ref_run_id="run_before",
                 gold_ref_measurement_regime="calibration",
                 padding_table=self.MERGED, gold_ref_padding_table=self.SPLIT))
        assert not res.hard_refuse
        assert any("padded-width histogram" in r for r in res.advisory_reasons)

    def test_matching_padding_raises_nothing_even_when_declared(self):
        res = rules.enforce(
            _req(padding_change_is_the_subject=True),
            _ctx(gold_ref_run_id="run_before",
                 gold_ref_measurement_regime="calibration",
                 padding_table=self.SPLIT,
                 gold_ref_padding_table=list(self.SPLIT)))
        assert not res.hard_refuse
        assert not any("padded-width" in r for r in res.advisory_reasons)

    def test_a_run_that_records_no_table_on_either_side_is_left_alone(self):
        """Older run records carry nothing here, and those
        comparisons must not start failing retroactively."""
        res = rules.enforce(
            _req(),
            _ctx(gold_ref_run_id="run_before",
                 gold_ref_measurement_regime="calibration"))
        assert not res.hard_refuse

    def test_a_gold_ref_with_a_table_and_a_run_without_one_is_refused(self):
        res = rules.enforce(
            _req(),
            _ctx(gold_ref_run_id="run_before",
                 gold_ref_measurement_regime="calibration",
                 gold_ref_padding_table=self.SPLIT))
        assert res.hard_refuse
        assert "padded-width histogram" in res.refusal_reason


class TestABuildThatCannotNameItsSource:
    """`ROSHAMBO3_GIT_SHA` names a commit, not the compiled source.

    A control arm built by restoring an older source file into a newer
    worktree stamps the newer commit while containing the older code, and
    nothing in the record could contradict it. The build stamp therefore carries a content hash of what was
    compiled and a dirty marker; these are the rules that read them.
    """

    def test_a_clean_build_is_accepted(self):
        r = rules.enforce(_req(), _ctx(build_source_dirty=False,
                                       build_source_sha256="a" * 64,
                                       build_source_stamp_phase="build"))
        assert not r.hard_refuse, r.refusal_reason
        assert not any("which source" in a for a in r.advisory_reasons)

    def test_a_decision_grade_number_from_a_patched_tree_is_refused(self):
        r = rules.enforce(_req(), _ctx(build_source_dirty=True,
                                       build_source_sha256="b" * 64,
                                       build_source_stamp_phase="build"))
        assert r.hard_refuse
        assert "matched no commit" in r.refusal_reason
        assert "b" * 64 in r.refusal_reason

    def test_an_advisory_run_from_a_patched_tree_may_proceed(self):
        """A patched tree is how every A/B control arm is built; refusing it
        outright would refuse the measurement this project mostly does."""
        r = rules.enforce(_req(decision_grade=False),
                          _ctx(build_source_dirty=True,
                               build_source_sha256="b" * 64))
        assert not r.hard_refuse, r.refusal_reason

    def test_a_binary_that_cannot_say_is_advisory_not_clean(self):
        """None is "nobody looked", which is not False. Same distinction as
        `checked_a_gpu_build`."""
        r = rules.enforce(_req(), _ctx(build_source_dirty=None))
        assert not r.hard_refuse
        assert any("does not say which source" in a for a in r.advisory_reasons)

    def test_a_configure_time_stamp_is_flagged(self):
        """`cmake --build` after an edit does not reconfigure, so a stamp taken
        only at configure time calls a configure-then-edit-then-build cycle
        clean -- the same defect one step smaller."""
        r = rules.enforce(_req(), _ctx(build_source_dirty=False,
                                       build_source_stamp_phase="configure"))
        assert not r.hard_refuse
        assert any("CONFIGURE time" in a for a in r.advisory_reasons)

    def test_the_source_fields_come_from_the_build_not_the_caller(self):
        from bench.build_check import VerifiedBuild

        b = VerifiedBuild(
            build_type="Release", cxx_flags="-O3", cuda_flags="-O3",
            with_cuda=True, cuda_architectures_requested="120",
            nvcc_version="13.2", git_sha="37a5d44",
            cuda_src_sha256="c" * 64, cuda_src_tree="d" * 40,
            cuda_src_dirty=True,
            cuda_src_dirty_files=["src/roshambo3/backend/_cuda_src/kernels.cu"],
            cuda_src_stamp_phase="build",
            arch_list=["sm_120"], so_sha256={"_roshambo3_cuda": "e"})
        ctx = rules.RunContext.from_build(
            b, None, git_dirty=False, gpu_lock_held=True,
            gpu_lock_acquisition_failed=False, other_process_on_gpu=False,
            clocks_locked=False, throttle_fraction=0.0,
            sustained_kernel_s=120.0, measurement_regime="calibration")

        assert ctx.build_source_dirty is True
        assert ctx.build_source_sha256 == "c" * 64
        assert ctx.build_source_stamp_phase == "build"
        assert rules.enforce(_req(), ctx).hard_refuse

        # And it reaches the written record, dirty file names included.
        d = b.as_dict()
        assert d["cuda_src_sha256"] == "c" * 64
        assert d["cuda_src_dirty"] is True
        assert d["cuda_src_dirty_files"] == [
            "src/roshambo3/backend/_cuda_src/kernels.cu"]

    def test_the_run_time_checkout_and_the_build_time_source_are_two_facts(self):
        """A binary built from a patched tree and then run from a clean
        checkout reports `git_dirty=False`. That is not the same claim."""
        clean_checkout_dirty_build = _ctx(git_dirty=False,
                                          build_source_dirty=True)
        assert rules.enforce(_req(), clean_checkout_dirty_build).hard_refuse

        dirty_checkout_clean_build = _ctx(git_dirty=True,
                                          build_source_dirty=False)
        r = rules.enforce(_req(), dirty_checkout_clean_build)
        assert not r.hard_refuse
        assert any("working tree is dirty" in a for a in r.advisory_reasons)


class TestTheLaunchSizeRule:
    """Launch-size saturation, asked of the launch rather than of the corpus.

    The retired rule compared one number against 1e6. Because launches never
    cross a group boundary, a 14-group 10M corpus caps its mean launch at
    714,286 -- so that rule refused *every* screen of it at *every* setting.
    And it could return the answer the caller wanted for the wrong reason
    when handed a corpus total.

    Numbers below are from measured runs.
    """

    # LIT-PCBA-20M, 20,381,286 conformers, 18 groups, 4,988 launches.
    HEAD_TO_HEAD = dict(conformers_per_launch=20_381_286 / 4_988,
                        n_launches=4_988, n_conformers=20_381_286,
                        n_groups=18)
    # A 10M color corpus at the shipped 65,536 default: 161 launches.
    TEN_MILLION_COLOR = dict(conformers_per_launch=9_999_998 / 161,
                             n_launches=161, n_conformers=9_999_998,
                             n_groups=14)

    def _shape(self, **kw):
        return rules.LaunchShape(**kw)

    def test_the_head_to_heads_shape_is_refused(self):
        """20.4M total with 4,086 per launch -- the case this rule exists for
        and the case the old check passed."""
        r = rules.enforce(_req(launch=self._shape(**self.HEAD_TO_HEAD)), _ctx())
        assert r.hard_refuse
        assert "4,086" in r.refusal_reason
        # And the refusal says what the corpus could have done instead, which
        # is the half the old rule had no way to express.
        assert "1,132,294" in r.refusal_reason
        assert "18 groups" in r.refusal_reason

    def test_a_ten_million_conformer_screen_is_admitted(self):
        """The shipped default's measured mean on a 10M color corpus is
        62,111. Under the old rule there was no setting that passed."""
        r = rules.enforce(_req(launch=self._shape(**self.TEN_MILLION_COLOR)),
                          _ctx())
        assert not r.hard_refuse, r.refusal_reason
        assert not any("per launch" in a for a in r.advisory_reasons)

    def test_the_same_corpus_cut_into_4096_launches_is_refused(self):
        """Admitting the corpus must not amount to admitting any screen of it:
        at a 4,096 block size the same corpus is refused."""
        r = rules.enforce(
            _req(launch=self._shape(conformers_per_launch=9_999_998 / 2_449,
                                    n_launches=2_449, n_conformers=9_999_998,
                                    n_groups=14)),
            _ctx())
        assert r.hard_refuse
        assert "714,286" in r.refusal_reason

    def test_a_corpus_that_cannot_saturate_is_advisory_not_refused(self):
        """48,660 conformers in 6 groups cap the mean launch at 8,110. No
        setting reaches the plateau, so the number is real and its absolute
        throughput is not the machine's -- which is what advisory means."""
        r = rules.enforce(
            _req(launch=self._shape(conformers_per_launch=8_110.0,
                                    n_launches=6, n_conformers=48_660,
                                    n_groups=6)),
            _ctx())
        assert not r.hard_refuse, r.refusal_reason
        assert any("cap the mean launch" in a for a in r.advisory_reasons)

    def test_leaving_launch_size_on_the_table_is_refused_even_on_a_small_corpus(self):
        """The corpus-bound relaxation is for launches the corpus bounds, not
        for every launch on a small corpus. The same 6-group, 48,660-conformer
        corpus screened at 1,014 per launch used an eighth of what its own
        structure permitted."""
        r = rules.enforce(
            _req(launch=self._shape(conformers_per_launch=48_660 / 48,
                                    n_launches=48, n_conformers=48_660,
                                    n_groups=6)),
            _ctx())
        assert r.hard_refuse
        assert "8,110" in r.refusal_reason

    def test_pitfall_14s_own_case_is_still_refused(self):
        """N=1e4, the two-point ROSHAMBO2 measurement the rule was written
        from. With no group structure recorded there is no ceiling to plead."""
        r = rules.enforce(_req(n_conformers=10_000), _ctx())
        assert r.hard_refuse
        assert "no group structure" in r.refusal_reason

    def test_a_run_that_does_not_say_how_it_was_launched_is_flagged(self):
        r = rules.enforce(_req(n_conformers=2_000_000), _ctx())
        assert not r.hard_refuse
        assert any("did not record its launch structure" in a
                   for a in r.advisory_reasons)

    def test_check_launch_saturation_is_three_way(self):
        big = rules.LaunchShape(65_536.0, 100, 6_553_600, 10)
        bound = rules.LaunchShape(8_110.0, 6, 48_660, 6)
        small = rules.LaunchShape(4_086.0, 4_988, 20_381_286, 18)
        assert rules.check_launch_saturation(big) == rules.SATURATION_OK
        assert (rules.check_launch_saturation(bound)
                == rules.SATURATION_CORPUS_BOUND)
        assert (rules.check_launch_saturation(small)
                == rules.SATURATION_TOO_SMALL)

    def test_the_boundary_is_where_the_measured_curve_says_it_is(self):
        """32,768 is `SATURATION_MARGIN * SATURATING_CONFORMERS_PER_LAUNCH`,
        and both come off the measured launch-size curve rather than from a corpus that
        needed to pass."""
        boundary = (rules.SATURATION_MARGIN
                    * rules.SATURATING_CONFORMERS_PER_LAUNCH)
        assert boundary == 32_768
        one_below = rules.LaunchShape(boundary - 1, 1, int(boundary), None)
        one_above = rules.LaunchShape(boundary, 1, int(boundary), None)
        assert (rules.check_launch_saturation(one_below)
                == rules.SATURATION_TOO_SMALL)
        assert rules.check_launch_saturation(one_above) == rules.SATURATION_OK

    def test_the_retired_constant_is_bound_but_no_longer_enforced(self):
        """Twelve archived scripts import `MIN_CONFORMERS_PER_LAUNCH` and their
        numbers have to keep reproducing, so the name stays. What must not
        survive is `enforce` consulting it."""
        assert rules.MIN_CONFORMERS_PER_LAUNCH == 1_000_000
        assert not hasattr(rules, "check_min_batch")
        # 62,111 is three orders of magnitude below the retired constant.
        r = rules.enforce(
            _req(launch=rules.LaunchShape(**self.TEN_MILLION_COLOR)), _ctx())
        assert not r.hard_refuse


class TestLaunchShapeCannotBeHandedACorpusTotal:
    """The specific mechanism of a corpus-total false pass.

    `check_min_batch` took an integer, so nothing distinguished "one launch's
    conformers" from "the whole corpus's". `LaunchShape.from_screen` reads the
    launch count off the screen and divides, so the mistake is not expressible.
    """

    class _Stats:
        n_conformers = 20_381_286
        n_blocks = 4_988

    class _Source:
        group_indices = tuple(range(18))

    def test_from_screen_divides_rather_than_trusting_a_caller(self):
        shape = rules.LaunchShape.from_screen(self._Stats(), self._Source())
        assert shape.n_conformers == 20_381_286
        assert shape.n_launches == 4_988
        assert shape.n_groups == 18
        assert round(shape.conformers_per_launch) == 4_086
        assert round(shape.ceiling) == 1_132_294
        assert (rules.check_launch_saturation(shape)
                == rules.SATURATION_TOO_SMALL)

    def test_an_unknown_group_count_leaves_the_ceiling_unknown(self):
        shape = rules.LaunchShape.from_screen(self._Stats())
        assert shape.n_groups is None
        assert shape.ceiling is None

    def test_a_screen_with_no_launches_does_not_divide_by_zero(self):
        class Empty:
            n_conformers = 0
            n_blocks = 0
        assert rules.LaunchShape.from_screen(Empty()).conformers_per_launch == 0


class TestLaunchOverheadIsMeasuredNotProxied:
    """The launch-overhead fraction itself, out of the run.

    A measured record carries the stage counters, so the fraction can be
    computed from an archived run rather than asserted about a hypothetical
    one.
    """

    HEAD_TO_HEAD_STAGES = {
        "alloc_s": 0.03406132897362113,
        "h2d_s": 2.1122373435646296,
        "kernel_s": 19.334592449478805,
        "precompute_s": 4.686625889036804,
        "d2h_s": 0.0,
        "free_s": 2.1836557392962277,
        "index_s": 0.1304568494670093,
        "topk_s": 0.9963426599279046,
        "n_calls": 4988.0,
        "enabled": 1.0,
    }
    HEAD_TO_HEAD_WALL_S = 30.29731158539653

    def test_an_archived_records_own_counters_put_it_at_23_percent(self):
        f = rules.launch_overhead_fraction(self.HEAD_TO_HEAD_STAGES,
                                           self.HEAD_TO_HEAD_WALL_S)
        assert 0.227 < f < 0.229, f
        assert f > rules.MAX_LAUNCH_OVERHEAD_FRACTION

    def test_it_reaches_the_verdict_as_an_advisory(self):
        r = rules.enforce(
            _req(launch=rules.LaunchShape(65_536.0, 100, 6_553_600, 10)),
            _ctx(device_stage_seconds=self.HEAD_TO_HEAD_STAGES,
                 measured_total_s=self.HEAD_TO_HEAD_WALL_S))
        assert not r.hard_refuse, r.refusal_reason
        assert any("paid once per launch" in a for a in r.advisory_reasons)

    def test_unrecorded_stage_timing_is_not_zero(self):
        assert rules.launch_overhead_fraction(None, 10.0) is None
        assert rules.launch_overhead_fraction({"enabled": 0.0}, 10.0) is None
        r = rules.enforce(
            _req(launch=rules.LaunchShape(65_536.0, 100, 6_553_600, 10)),
            _ctx())
        assert not any("paid once per launch" in a for a in r.advisory_reasons)

    def test_a_coalesced_screen_is_below_the_threshold(self):
        """The same stages, at the launch count the shipped default gives on a
        10M corpus: 161 launches instead of 4,988."""
        scaled = dict(self.HEAD_TO_HEAD_STAGES)
        for k in rules.PER_LAUNCH_STAGES:
            scaled[k] = self.HEAD_TO_HEAD_STAGES[k] * 161 / 4988
        f = rules.launch_overhead_fraction(scaled, 12.503938275855035)
        assert f < rules.MAX_LAUNCH_OVERHEAD_FRACTION, f

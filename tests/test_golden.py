"""The arithmetic, pinned to committed scores that need no other checkout.

**The ANALYTIC arm's bit comparison is retired.** Accepted
changes move that arm's bits through code generation, and the int16
query record moves them by design, so after every legitimate kernel change it
failed exactly as it would for a forbidden flag or a bug. It could not tell
them apart, and the architecture gate meant it ran only on full Release builds.
The drift it reported was declared. Other
checks replace it that can tell those cases apart. The committed
`analytic` array stays in the fixture, and the result-invariant check below
still reads it. Only the FIELD arm is still compared bit for bit.

**WHICH UPDATE RULE THE COMMITTED SCORES ARE: the ADDITIVE one, mode 0
(``UpdateRule.ADAGRAD_QADD``) -- which is NOT the default.** The
fixture predates that flip, so its bits belong to the Adagrad step added to
``q[1:4]`` plus a renormalization, on both kernels. ``bench/make_golden.py``
names the rule in the call rather than inheriting it, and records the rule
inside the ``.npz`` as ``update_rules``, so this cannot be got wrong by reading
either file alone. If those two ever disagree, the assertion below says so
instead of reporting a numerics change.

**Pinned rather than regenerated, on purpose.** Regenerating stamps the
generating build's ``arch`` and ``nvcc_version`` into the fixture, so it may only
be done from the multi-arch Release build the committed file names; doing it from
a ``native`` dev build would narrow the gate to one machine rather than widen it.
See ``bench/make_golden.py``'s note on ``OPT``.

**What that leaves uncovered, and where it is covered instead.** These bits
fingerprint the machinery both rules share -- field sampling, the gradient, the
reductions, the Tier-3 cascade, every derived constant -- and they do not touch
``quat_left_exp`` or ``apply_update_lanes``, which only the default runs. Those
are pinned at bit level without a committed fixture by
``tests/test_exp_map_field_cuda.py``: under the exp map,
``UpdateLanes.PER_LANE`` and ``REDUCE_SCATTER`` must return bit-identical scores,
which no arithmetic change inside the distributed update can survive.
``UpdateLanes.ALL`` is deliberately not part of that gate -- it is not
bit-identical under this rule, and why, and recorded it on the enum. A second
committed arm here is the better answer and is an append to
``make_golden.ARMS`` for whoever next regenerates on a Release build.

`bench/xcheck_roshambo2.py` is the strongest numerics check in the project and
it is not portable: it needs a ROSHAMBO2 checkout and a built development corpus.
Neither exists on CI, on a fresh machine, or in a public release -- and once
ROSHAMBO2 is gone, nothing else can distinguish a *declared* numerics change
from a bug. A reassociated sum, a constant edited in the header, a compiler flag
that slipped into the build: all of them change scores, and the policy says
every one of those axes lands on the same output distribution, so there is no
"small perturbation" regime to notice them by eye.

So the input is generated from a fixed seed and the output is committed beside
it (`bench/make_golden.py`), and this compares them **bit for bit**. The numbers
are not a physical claim and nothing may be cited from them; they are a
fingerprint whose only job is to stop changing.

**This is bit-exact against the build that generated it, and only that build.**
Scores are not bitwise portable across architectures (measured,
`doc/NUMERICS_POLICY.md` §8) and the toolkit is a measured arithmetic axis, so
the fixture records its `arch` and `nvcc_version` and the comparison skips with
a reason on any other build. On this project's second platform (sm_70, CUDA
12.9) it skips rather than failing. It is portable in the sense that matters --
it needs no ROSHAMBO2 and no corpus -- not in the sense that its bits travel.

**When this fails on the build it was generated for**, one of two things is true
and the difference matters:

* a numerics change was made deliberately -- then it must be declared in
  `doc/NUMERICS_POLICY.md` §4, measured, and the fixture regenerated with
  `python bench/make_golden.py` under the GPU lock on a Release build.
  **Changing the default update rule is not that**, and that change is the
  worked example: the default moved, this fixture stayed, and what kept it meaningful
  was pinning the rule in the generator -- not regenerating against a different
  optimizer, which would have replaced a fingerprint of the arithmetic with a
  fingerprint of somebody's default;
* it was not -- then this is the bug the gate exists to find, and the fixture
  must not be regenerated to make it pass.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from roshambo3 import numerics
from roshambo3.backend import cuda as backend

GOLDEN = pathlib.Path(__file__).resolve().parent / "data" / "golden_scores.npz"

needs_gpu = pytest.mark.skipif(not backend.available(),
                               reason="no CUDA extension in this build")


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"{GOLDEN} not generated; run bench/make_golden.py")
    return np.load(GOLDEN)


def _build_matches(golden) -> "str | None":
    """Why this build cannot be compared against the fixture, or None.

    **Scores are not bitwise portable across architectures** -- measured,
    `doc/NUMERICS_POLICY.md` section 8 -- and the toolkit is a measured
    arithmetic axis too: 13.2 against 12.9 leaves 92.95% of pairs bit-identical.
    So a bit-exact fixture is only meaningful against the build that produced
    it, and on this project's own second platform (sm_70, CUDA 12.9) it
    would otherwise fail with a message offering only "declared numerics change,
    or a bug" -- neither of which is true, and whose natural remedy,
    regenerating, silently repoints the only portable bit-level check at
    different arithmetic.

    `bench/arch_probe.cu` makes the same distinction deliberately: it *prints*
    scores for two hosts to diff rather than asserting they are equal.
    """
    from roshambo3 import get_build_info

    bi = get_build_info()
    want_arch = str(golden["arch"])
    want_nvcc = str(golden["nvcc_version"])
    if str(bi.cuda_architectures) != want_arch:
        return (f"fixture was generated for arch {want_arch!r}, this build is "
                f"{str(bi.cuda_architectures)!r}")
    if str(bi.nvcc_version) != want_nvcc:
        return (f"fixture was generated with nvcc {want_nvcc}, this build has "
                f"{bi.nvcc_version}")
    return None


@needs_gpu
class TestTheGoldenScoresAreReproduced:

    def _recompute(self, golden):
        import bench.make_golden as mk
        why = _build_matches(golden)
        if why:
            pytest.skip(f"not comparable: {why}. Scores are not bitwise "
                        f"portable across architectures or toolkits "
                        f"(NUMERICS_POLICY section 8) -- this is not a failure, "
                        f"and the fixture must NOT be regenerated to make it "
                        f"run here.")
        return mk.compute(golden["coords"], golden["n_real"])

    # ANALYTIC retired from the bit comparison; see the module docstring.
    @pytest.mark.parametrize("kernel", ["field"])
    def test_scores_are_bit_identical(self, golden, kernel):
        got = self._recompute(golden)[kernel]
        want = golden[kernel]
        assert got.shape == want.shape
        if np.array_equal(got, want):
            return
        diff = got != want
        cols = np.unique(np.argwhere(diff)[:, 1])
        detail = []
        for c in cols[:6]:
            d = np.abs(got[:, c].astype(np.float64)
                       - want[:, c].astype(np.float64))
            d = d[d > 0]
            detail.append(f"col {c}: n={d.size} max={d.max():.3e}")
        pytest.fail(
            f"{kernel}: {int(diff.sum())}/{got.size} words differ from the "
            f"committed golden scores. Declared numerics change, or a bug -- "
            f"see this module's docstring. " + "; ".join(detail))

    def test_the_fixture_says_which_update_rule_its_bits_are(self, golden):
        """The generator and the fixture must name the same rule.

        Not decoration. A reader who assumes the committed scores are the
        *default* rule's would read a genuine numerics change as a bug, or the
        reverse. Older fixtures predate the field and are accepted as mode 0,
        which is what they are -- the key was added along with the flip
        that made it necessary.
        """
        import bench.make_golden as mk

        want = tuple(mk.ARMS)
        got = (tuple(str(x) for x in golden["update_rules"])
               if "update_rules" in golden else ("ADAGRAD_QADD",))
        assert got == want, (
            f"the fixture was generated at update rule(s) {got} but "
            f"bench/make_golden.py now generates {want}. One of the two moved; "
            f"a bit comparison across that is not a numerics check.")

    def test_the_input_still_follows_from_the_seed(self, golden):
        """The seed is the input; if the generator drifts, the fixture stops
        being reproducible from anything but itself."""
        import bench.make_golden as mk
        coords, n_real = mk.make_input()
        assert np.array_equal(coords, golden["coords"])
        assert np.array_equal(n_real, golden["n_real"])


@needs_gpu
class TestResultInvariantsHoldOnRealOutput:
    """The check ROSHAMBO2 ran on every call and the port dropped.

    It shares no code with the kernel, which is the whole point: the C++ and
    CUDA paths once agreed with each other by
    sharing a typo, and no cross-check between two implementations can see
    that. An invariant can.
    """

    @pytest.mark.parametrize("kernel", ["field", "analytic"])
    def test_the_committed_scores_obey_them(self, golden, kernel):
        """Deliberately not gated on the build matching: these are
        inequalities, not bit comparisons, so they must hold on every
        architecture and toolkit. A build where they do not is broken
        regardless of which one generated the fixture."""
        assert numerics.check_result_invariants(golden[kernel]) == []


class TestTheInvariantsThemselves:
    """The checker has to fail on a broken block, or it certifies nothing."""

    def _clean(self, n=4):
        s = np.zeros((n, 20), np.float32)
        s[:, 9] = 1.0        # unit quaternion
        s[:, 5] = 10.0       # V_AA
        s[:, 6] = 20.0       # V_BB
        s[:, 3] = 12.0       # V_AB <= sqrt(200) = 14.14
        return s

    def test_a_clean_block_passes(self):
        assert numerics.check_result_invariants(self._clean()) == []

    def test_a_pose_that_is_not_a_rotation_is_caught(self):
        s = self._clean()
        s[2, 9] = 1.7
        problems = numerics.check_result_invariants(s)
        assert any("not unit norm" in p for p in problems)

    def test_an_unwritten_row_is_named_as_such(self):
        """An all-zero quaternion is uninitialized memory, not a wrong pose,
        and reporting it as "slightly off" would send the reader hunting for
        an arithmetic bug."""
        s = self._clean()
        s[1, 9] = 0.0
        assert any("all-zero pose quaternion" in p
                   for p in numerics.check_result_invariants(s))

    def test_an_overlap_larger_than_cauchy_schwarz_allows_is_caught(self):
        s = self._clean()
        s[0, 3] = 15.0
        assert any("Cauchy-Schwarz" in p
                   for p in numerics.check_result_invariants(s))

    def test_a_non_finite_score_is_caught(self):
        s = self._clean()
        s[3, 1] = np.nan
        assert any("non-finite" in p
                   for p in numerics.check_result_invariants(s))

    def test_the_color_channel_is_not_flagged_while_it_is_all_zero(self):
        """Color is refused end to end, so its columns are zero -- vacuously
        consistent, and a checker that flagged it would be unusable."""
        assert numerics.check_result_invariants(self._clean()) == []

    def test_a_wrong_shaped_block_is_refused_rather_than_guessed_at(self):
        with pytest.raises(ValueError, match=r"\[n, 20\]"):
            numerics.check_result_invariants(np.zeros((4, 19), np.float32))

    def test_assert_form_raises_with_the_detail(self):
        s = self._clean()
        s[0, 3] = 15.0
        with pytest.raises(AssertionError, match="Cauchy-Schwarz"):
            numerics.assert_result_invariants(s)

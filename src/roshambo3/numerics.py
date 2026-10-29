# MIT License
#
# Copyright (c) 2026 Kimia Therapeutics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The numerics, precision and determinism policy, as code rather than prose.

``doc/NUMERICS_POLICY.md`` states the policy; this module is where the parts of
it that can be checked *are* checked, so that a violation is an exception and
not a paragraph somebody was supposed to have read.  Everything here is cheap
and import-safe: no CUDA, no GPU, no compiled extension required except where a
function explicitly asks for a build stamp.

Three things live here.

**1. The pinned arithmetic.**  A sweep enumerated, over 463,200 real pairs on
Blackwell sm_120, exactly which build knobs change the answer.  Six nvcc
numerics flags do (``--use_fast_math``, ``-fmad=false``, ``-ftz=true``,
``-prec-div=false``, ``-prec-sqrt=false``, and a reordered reduction), all of
them landing on the *same* output distribution (median |dT| ~ 6e-8, p99 ~ 2.7e-5,
max ~ 2e-2) despite input perturbations spanning orders of magnitude.  There is
therefore no "small perturbation" regime to tolerate an epsilon in, which is why
the policy pins the arithmetic instead of declaring a tolerance.
:data:`FORBIDDEN_CUDA_FLAGS` is that enumeration, and
:func:`assert_arithmetic_pinned` turns it into a checked invariant against the
build stamp.

**2. The reduced-overlap frame.**  The authoritative definition of the constants
the kernels are written in.  ``backend/_cuda_src/kernels.cuh``
carries the same numbers as float literals for the device code; the values here
are what they are checked against.

**3. The deterministic tie-break.**  float32 score ties are common at
M = 1e7, and an unstable sort breaks them differently between runs and backends,
so "rank changed" stops meaning anything.  :func:`rank_order` is the one ranking
rule, and it must be applied identically to whatever is called gold and whatever
is called candidate.

Scope: every measured number quoted here was taken on Blackwell sm_120.
Cross-architecture score reproducibility is **unvalidated** -- see
``doc/NUMERICS_POLICY.md`` section 8.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 1. Physical constants and the reduced-overlap frame
# ---------------------------------------------------------------------------

KAPPA = 2.41798793102
"""Gaussian-fit constant relating an atom's hard-sphere radius to its width."""

CARBON_RADIUS = 1.7
"""All shape Gaussians use the carbon radius; the code is uniform-radius."""

A = np.float32(np.float32(KAPPA) / np.float32(np.float32(CARBON_RADIUS) ** 2))
"""Per-atom Gaussian width, KAPPA / r_C^2 = 0.836674."""

ALPHA = np.float32(np.float32(A * A) / np.float32(A + A))
"""The pair exponent coefficient a1*a2/(a1+a2) = A/2 = 0.418337.

The pair term is exp(-ALPHA * d^2); the kernel evaluates it as exp2(-d'^2)
with d' = COORD_SCALE * d.
"""

SHAPE_PREFACTOR = np.float32(8.0 * (math.pi / (2.0 * float(A))) ** 1.5)
"""8 * (pi/2A)^1.5 = 20.5795.  Cancels identically out of the Tanimoto.

The Tanimoto T = V_AB/(V_AA + V_BB - V_AB) is homogeneous of degree zero, so a
prefactor shared by all three volumes has no effect on it whatsoever.  The
kernels therefore accumulate the reduced overlap S and apply this constant only
where a raw volume is reported.
"""

COLOR_PREFACTOR = np.float32((math.pi / 2.0) ** 1.5)
"""(pi/2)^1.5 = 1.9687, the color prefactor at the canonical width r = 1.

With the default diagonal interaction matrix every color feature pair carries
this same constant, so it cancels out of the color Tanimoto the same way.  A
non-diagonal matrix does not break the extraction -- the per-pair residual
p^2 * r^-1.5 stays inside the accumulated term -- but it does mean the color
channel is no longer a single well-conditioned sum, so
:func:`check_interaction_matrix` reports it at query prep instead of letting it
pass silently.
"""

COORD_SCALE = np.float32(math.sqrt(float(ALPHA) * math.log2(math.e)))
"""sqrt(alpha * log2 e) = 0.7768736.

Coordinates are carried pre-multiplied by this, so the pair term is exactly
exp2f(-|delta|^2) with no multiply in the innermost loop.  __expf(x) is
ex2.approx(x * log2 e) on this hardware anyway, so folding the coefficient into
the coordinate scale makes the operation the machine performs explicit rather
than hiding it behind a general-purpose exp() call.  It also gives the exponent
and any future int16 quantization a single shared frame instead of two that
have to be reconciled.
"""

INV_TWO_ALPHA = np.float32(1.0 / (2.0 * float(ALPHA)))
"""Converts a color interaction width r into its exp2 coefficient in the
shape frame: exp(-(r/2) d^2) == exp2(-r * INV_TWO_ALPHA * d'^2)."""

MINUS_2LN2 = np.float32(-2.0 * math.log(2.0))
"""d/dx exp2(-x) = -ln2 exp2(-x); every gradient term carries -2 ln 2."""

WIRE_QUANT_STEP = np.float32(2.0 ** -8)
"""The int16 wire grid, 2^-8 Angstrom (NUMERICS_POLICY 4e).

This differs from the reduced frame's COORD_SCALE above. It must agree with
format.constants.COORD_SCALE, both format-3 STEP_LOG2 constants, and the compiled
CUDA step. The device keeps the two operations
``cscale * (ROSHAMBO3_STAGE_INV_Q_XYZ * (float)v)``: dequantization then frame
scaling. Folding them changes the rounding contract (NUMERICS_POLICY 4c).
At this binary step, int16 dequantization is exact and
``(short)__float2int_rn(256.0f * (0.00390625f * (float)v)) == v`` for every
int16 value, verified over all 65,536 values.
"""

#: The float literals the device header must carry, keyed by macro name.  A
#: test asserts these still match kernels.cuh, so the two cannot drift.
DEVICE_CONSTANTS = {
    "ROSHAMBO3_SHAPE_PREFACTOR": SHAPE_PREFACTOR,
    "ROSHAMBO3_COLOR_PREFACTOR": COLOR_PREFACTOR,
    "ROSHAMBO3_COORD_SCALE": COORD_SCALE,
    "ROSHAMBO3_INV_TWO_ALPHA": INV_TWO_ALPHA,
    "ROSHAMBO3_MINUS_2LN2": MINUS_2LN2,
    # The wire record's step, pinned to the header the same way the
    # reduced-frame constants are.  It is load-bearing now that the host no
    # longer dequantizes: a header edit here would scale every library
    # coordinate by the wrong constant with no other symptom.
    "ROSHAMBO3_STAGE_INV_Q_XYZ": WIRE_QUANT_STEP,
    "ROSHAMBO3_STAGE_Q_XYZ": np.float32(1.0) / WIRE_QUANT_STEP,
}


class WireQuantizationError(ValueError):
    """A corpus's coordinate units differ from the backend wire record."""


# ---------------------------------------------------------------------------
# 2. The pinned arithmetic
# ---------------------------------------------------------------------------

PINNED_CUDA_TOOLKIT = "13.2"
"""The toolkit version the canonical Tier 3 numbers were measured with.

The PTX -> SASS stage was measured not to decide the arithmetic here: driver
PTX JIT vs offline SASS, ``ptxas -O1`` and ``ptxas -O2`` are all bit-identical.
That localises the remaining codegen exposure to the nvcc frontend, which is
what a toolkit change replaces -- so the toolkit version must be *recorded*
with every number, and a change of it must be a deliberate act.

Measured:
building the identical source with CUDA 12.9 instead of 13.2 leaves 92.95% of
pairs bit-identical on all 20 columns, and where it does differ |dT_shape| is
p99 = 8.9e-8, max = 3.0e-7, with 10 of 463,200 pairs landing in a different
basin.  The optimizer trajectory itself is bit-identical for 99.998% of pairs;
only the final V_AB reduction moves.  That is a real perturbation -- so pin and
record it -- but it is two to five orders of magnitude weaker than any of the
six numerics flags on every statistic (300x on p99 |dT|, 5e4x on max, 550x on
the basin-flip rate), which is why a toolkit mismatch is an advisory and a
forbidden flag is a hard error.
"""

VALIDATED_CUDA_ARCH = "sm_120"
"""Every number in this policy was measured on Blackwell sm_120.

What a change of architecture does to those numbers was measured by running
the identical source, flags, CUDA 12.9 toolkit and bit-identical float32 inputs
on Volta sm_70 and on Blackwell sm_120 over the same 463,200-pair panel
(``doc/NUMERICS_POLICY.md`` section 8):

* **at a fixed pose the scoring arithmetic is bit-identical** on all 20 columns
  for 100.000% of pairs -- and ``V_AA``/``V_BB`` stay bit-identical even at 100
  steps;
* **the optimizer is not.** One Adagrad step injects ~1 ULP (|dT| p99 8.9e-8);
  100 steps amplify it to p99 2.6e-5 with a 1.22% basin-flip rate, the same
  saturated distribution ``--use_fast_math`` reaches.  So a score is **not
  bitwise portable across architectures** and two scores may still only be
  compared bitwise when their stamps agree on architecture;
* **the ranked output survives anyway**: top-1 and top-5 sets *and* order are
  identical for 480/480 queries, top-10 for 479/480, and a top-N merged across
  a Volta-scored and a Blackwell-scored shard reproduces the homogeneous
  top-10 exactly for every query.

sm_70 is outside the sm_80+ deploy library and Volta is architecturally further
from Blackwell than Ampere is, so that is an upper bound on the
deploy-relevant effect rather than a prediction of one.  This constant still
names sm_120 because that is where the policy's own tables were taken.
"""

CROSS_ARCH_REPRODUCIBILITY = (
    "measured, sm_70 vs sm_120 (doc/NUMERICS_POLICY.md section 8): NOT bitwise portable "
    "-- 0.12% of pairs bit-identical after 100 Adagrad steps, |dT_shape| p99 2.6e-5 overall "
    "and 3.6e-7 on the T>=0.7 stratum -- but bit-identical at a fixed pose (steps=0), and "
    "the ranked output is unchanged: top-1 and top-5 set and order identical for 480/480 "
    "queries. sm_70 is outside the sm_80+ deploy library, so this bounds the effect rather "
    "than predicting Ampere-Blackwell."
)
"""What every run record must say about comparing scores across GPUs.

It is a single string rather than a dict because it is quoted verbatim in run
records and read by humans, and because ``test_numerics_policy.py`` asserts on
its content -- a structured field would let the claim drift from the tests that
check it.
"""

OVERRIDE_ENV = "ROSHAMBO3_NUMERICS_OVERRIDE"
"""Set to 1 to downgrade a forbidden-flag violation to a warning.

This exists for exactly one purpose: deliberately building perturbed arithmetic
in order to *measure* it.  A run that sets it is not a run whose
scores may be compared with any other, and :func:`numerics_stamp` records that
it was set so a run record cannot hide it.
"""


@dataclass(frozen=True)
class ForbiddenFlag:
    """A build flag that measurably changes the answer, and by how much."""

    flag: str
    median_dT: float
    p99_dT: float
    max_dT: float
    basin_flip_rate: float
    why: str

    @property
    def key(self) -> str:
        return _normalize_flag(self.flag)


def _normalize_flag(tok: str) -> str:
    """nvcc accepts -x, --x and mixes _ and -; compare on a canonical form."""
    return tok.lstrip("-").replace("_", "-").lower()


#: The enumerated no-go list, with the measured effect on |dT_shape| over
#: 463,200 pairs of the tuning panel at the production configuration
#: (start_mode=1, 100 steps, lr=0.1, shape-only), Blackwell sm_120.
FORBIDDEN_CUDA_FLAGS = (
    ForbiddenFlag(
        "--use_fast_math", 6.0e-8, 2.9e-5, 1.5e-2, 0.0130,
        "implies -ftz=true -prec-div=false -prec-sqrt=false and swaps the "
        "libdevice transcendentals for their approximate forms; four "
        "independent arithmetic changes in one switch",
    ),
    ForbiddenFlag(
        "-fmad=false", 6.0e-8, 2.7e-5, 2.1e-2, 0.0124,
        "turns off FMA contraction, so every a*b+c in the inner loop rounds "
        "twice instead of once",
    ),
    ForbiddenFlag(
        "-ftz=true", 3.0e-8, 2.5e-5, 1.6e-2, 0.0120,
        "flushes denormals to zero; the pair term exp2(-d^2) reaches denormal "
        "magnitudes for distant atom pairs, which are exactly the terms a "
        "large molecule has most of",
    ),
    ForbiddenFlag(
        "-prec-div=false", 6.0e-8, 2.9e-5, 1.2e-2, 0.0127,
        "approximate division; the Tanimoto and the Adagrad step are both "
        "divisions",
    ),
    ForbiddenFlag(
        "-prec-sqrt=false", 3.0e-8, 2.6e-5, 1.6e-2, 0.0122,
        "approximate square root; Adagrad takes a sqrt every step and the "
        "quaternion is renormalized every step",
    ),
)

#: Not a compiler flag, but the same class of change and measured the same way:
#: reversing the accumulation order of the inner loop moves |dT_shape| by
#: median 8.9e-8 / p99 2.8e-5 / max 1.7e-2, with a 1.2% basin-flip rate.  The
#: canonical Tier 3 reduction order is the source order; a kernel rewrite that
#: reassociates it is a declared numerics change, not a refactor.
REDUCTION_ORDER_EFFECT = {"median": 8.9e-8, "p99": 2.8e-5, "max": 1.7e-2,
                          "basin_flip_rate": 0.0124}

#: The format-2 (RSB3) int16 coordinate round trip, for comparison: median 1.1e-5 /
#: p99 1.0e-4 / max 5.6e-2.  Two orders above the compiler axes, i.e. storage
#: quantization is a bigger arithmetic decision than any build flag.
INT16_ROUNDTRIP_EFFECT = {"median": 1.1e-5, "p99": 1.0e-4, "max": 5.6e-2}


def check_cuda_flags(cuda_flags: str) -> list[str]:
    """Return a human-readable violation per forbidden flag present.

    Args:
        cuda_flags: the ``cuda_flags`` field of the build stamp, i.e. the
            nvcc flags the loaded ``_roshambo3_cuda`` was actually compiled
            with -- not what a README says the build ought to use.

    Returns:
        A list of violation strings, empty if the build is compliant.
    """
    if not cuda_flags or cuda_flags == "none":
        return []
    present = {_normalize_flag(tok) for tok in cuda_flags.split()}
    # a flag may also arrive glued into --ptxas-options=a,b or after -Xptxas
    blob = _normalize_flag(cuda_flags.replace(",", " "))
    out = []
    for f in FORBIDDEN_CUDA_FLAGS:
        if f.key in present or f.key in blob:
            out.append(
                f"{f.flag} is present in the build's nvcc flags. It is on the "
                f"enumerated no-go list: {f.why}. Measured effect on the shape "
                f"Tanimoto over 463,200 pairs (sm_120): median "
                f"{f.median_dT:.1e}, p99 {f.p99_dT:.1e}, max {f.max_dT:.1e}, "
                f"basin-flip rate {f.basin_flip_rate:.1%}."
            )
    return out


#: Host-compiler flags that relax floating point in the extension's host code.
#: Listed by analogy with the measured device-side flags above -- **not measured
#: on this project**, because host code is not the specification (see
#: doc/NUMERICS_POLICY.md section 2). Hence advisory, not fatal: the policy does
#: not claim a number it has not taken.
#:
#: ROSHAMBO3 dropped ROSHAMBO2's C++/OpenMP backend, so these no longer threaten
#: an oracle -- the oracle is now ``roshambo3.reference``, which is float64
#: NumPy and compiled by nobody. They still reach the host half of the CUDA
#: extension, which is why the check stays.
SUSPECT_CXX_FLAGS = ("-ffast-math", "-funsafe-math-optimizations", "-Ofast",
                     "-fassociative-math", "-ffinite-math-only")


def host_flags_were_recorded(cxx_flags: str) -> bool:
    """Whether the build stamp actually carried host compiler flags.

    ``check_cxx_flags`` returns an empty list both when it inspected the host
    flags and found nothing and when there was nothing to inspect, and those
    are different claims. Like ``checked_a_gpu_build`` and ``arch_probe_error``, it is answered by a separate predicate rather than by widening
    what the existing one means.
    """
    return bool(cxx_flags) and cxx_flags != "none"


def check_cxx_flags(cxx_flags: str) -> list[str]:
    """Advisories for host flags that relax the extension's host arithmetic.

    An empty result means "inspected, found none" only when
    :func:`host_flags_were_recorded` is true. Callers that need to tell the two
    apart must ask both, exactly as they must ask ``ok`` and
    ``checked_a_gpu_build`` for the device side.
    """
    if not host_flags_were_recorded(cxx_flags):
        return []
    present = {_normalize_flag(t) for t in cxx_flags.split()}
    return [
        f"{f} is present in the host compiler flags. It relaxes floating point "
        f"in the extension's host code, which prepares and reduces what the "
        f"kernel sees. Unlike the nvcc flags this effect is NOT measured on "
        f"this project, so it is reported rather than refused."
        for f in SUSPECT_CXX_FLAGS if _normalize_flag(f) in present
    ]


def check_toolkit(nvcc_version: str) -> str | None:
    """Return an advisory string if the toolkit is not the pinned one."""
    if not nvcc_version or nvcc_version == "none":
        return None
    got = ".".join(str(nvcc_version).split(".")[:2])
    if got == PINNED_CUDA_TOOLKIT:
        return None
    return (
        f"CUDA toolkit {got} is not the pinned {PINNED_CUDA_TOOLKIT}. The "
        f"toolkit version is a measured arithmetic axis: 13.2 vs 12.9 leaves "
        f"92.95% of pairs bit-identical, with |dT_shape| p99 = 8.9e-8 and "
        f"max = 3.0e-7 on the rest (sm_120). Scores from this build are "
        f"not bitwise comparable with scores from the pinned toolkit."
    )


@dataclass(frozen=True)
class ArithmeticCheck:
    """Result of checking a build stamp against the pinned arithmetic."""

    violations: list[str] = field(default_factory=list)
    """Hard failures: a forbidden flag was compiled in."""

    advisories: list[str] = field(default_factory=list)
    """Recorded, not fatal: toolkit version, architecture scope."""

    overridden: bool = False
    """True iff ROSHAMBO3_NUMERICS_OVERRIDE was set."""

    _with_cuda: bool = False
    """Whether a CUDA extension was present; read via ``checked_a_gpu_build``."""

    _host_flags_seen: bool = False
    """Whether the CUDA stamp carried host compiler flags; read via
    ``checked_host_flags``."""

    _cpu_host_flags_seen: bool = False
    """Whether the CPU extension's OWN stamp carried compiler flags; read via
    ``checked_cpu_host_flags``.

    The twin of :attr:`_host_flags_seen`, and the reason it exists is the same
    misfire :attr:`device_advisories` records. ``_host_flags_seen`` is read off
    ``get_build_info()``, which is the CUDA extension's stamp -- so on a tree
    with no CUDA extension it is False, and a CPU run was told its host flags
    had not been inspected when in fact ``_cpu_flag_advisories`` had just read
    them. The flags that matter to a CPU number are the CPU extension's."""

    host_advisories: list[str] = field(default_factory=list)
    """The subset of :attr:`advisories` that is about the CPU extension.

    The mirror of :attr:`device_advisories`: without it a **GPU** run on a
    CUDA-only tree would draw "no CPU extension is present", the same misfire
    pointed the other way. A scoping mechanism applied in one direction only
    is half a fix."""

    device_advisories: list[str] = field(default_factory=list)
    """The subset of :attr:`advisories` that is about the CUDA extension.

    Kept as a *subset marker* rather than by splitting the list, so every
    existing reader of ``advisories`` sees exactly what it saw before. It is
    needed because a CPU run would otherwise be told "build targets 'native' ... do
    not quote a number from it", which is true of the GPU binary and says
    nothing about the CPU one -- and an advisory that fires on a compliant run
    is one people learn to skim."""

    _with_cpu: bool = False
    """Whether a CPU extension was present and its flags were read; exposed via
    ``checked_a_cpu_build``. ``get_build_info()`` is the CUDA extension's
    stamp, so without this the CPU extension's flags would never be looked at
    and a future ``-ffast-math`` there would not be noticed. The policy
    DEFERS those flags rather than refusing them, which makes noticing them
    the entire safeguard."""

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def checked_cpu_host_flags(self) -> bool:
        """Whether the CPU extension's own compiler flags were read.

        What ``checked_host_flags`` means for a CPU-backend run.
        """
        return self._cpu_host_flags_seen

    @property
    def checked_a_cpu_build(self) -> bool:
        """Whether there was a CPU build whose flags were read.

        The twin of :attr:`checked_a_gpu_build` and it means the same thing:
        ``ok`` is vacuously true about an artifact nobody inspected.
        """
        return self._with_cpu

    @property
    def checked_a_gpu_build(self) -> bool:
        """Whether there was a CUDA build to check at all.

        ``ok`` means "found no violations", which is vacuously true when no
        extension is present. A benchmark harness must require **both**: an
        ``ok`` result on a build that does not exist is not evidence the
        arithmetic is pinned.
        """
        return self._with_cuda

    @property
    def checked_host_flags(self) -> bool:
        """Whether there were host compiler flags to check at all.

        ``advisories`` being free of host-flag entries is vacuously true when
        the stamp carried no host flags, in the same way ``ok`` is vacuously
        true with no extension. Host flags are **not** a measured axis on this
        project (see :data:`SUSPECT_CXX_FLAGS`), so a build whose host flags
        could not be read is an advisory rather than a refusal -- the policy
        does not claim a number it has not taken, in either direction.
        """
        return self._host_flags_seen


def _cpu_flag_advisories() -> "tuple[bool, list[str]]":
    """Inspect the CPU extension's OWN compiler flags.

    ``get_build_info()`` is the CUDA extension's stamp. Its ``cxx_flags`` are
    that extension's host half (` -O3 -DNDEBUG` on this tree); the CPU
    extension is a separate compilation carrying
    ``-march=x86-64-v3 -fno-fast-math -fno-associative-math``. This reads the
    second, so :data:`SUSPECT_CXX_FLAGS` applies to the artifact that does the
    CPU arithmetic.

    Never raises: a tree with no CPU extension is a legitimate state, and the
    caller distinguishes it through ``checked_a_cpu_build`` rather than through
    an exception.
    """
    try:
        from .backend import cpu as _cpu_backend
        stamp = _cpu_backend.build_stamp()
    except (ImportError, ModuleNotFoundError):
        # Genuinely absent. This is a legitimate state and the caller reports
        # it as "no CPU extension is present".
        return False, []
    except Exception as exc:                                # noqa: BLE001
        # Present but unreadable -- a partial build, a broken stamp, an import
        # that fails for a reason other than absence. The caller's message for
        # an empty advisory list is "no CPU extension is present", which is
        # FALSE here, so say what actually happened instead of returning the
        # same empty signal as absence. Still not an exception: the CUDA path
        # calls this too and must not fail because a CPU artifact is broken.
        return False, [
            "a CPU extension is present but could not be inspected "
            f"({type(exc).__name__}: {str(exc)[:120]}), so the flags it was "
            "compiled with were not read. This is NOT the same state as "
            "'no CPU extension is present' and must not be reported as one."]

    flags = str(stamp.get("cxx_flags", "none"))
    if not host_flags_were_recorded(flags):
        return False, [
            "a CPU extension is present but its stamp records no compiler "
            "flags, so nothing checked whether it was built with relaxed "
            "floating point. An empty advisory list here means 'nothing to "
            "check', not 'checked and clean'."]

    adv = [a.replace("in the host compiler flags",
                     "in the CPU EXTENSION's compiler flags")
           for a in check_cxx_flags(flags)]
    return True, adv


def check_build(build_info=None) -> ArithmeticCheck:
    """Check a build stamp against the pinned arithmetic.

    Args:
        build_info: a ``roshambo3.BuildInfo``.  Read from the loaded extension
            if omitted.
    """
    if build_info is None:
        from ._build_info import get_build_info

        build_info = get_build_info()

    violations = check_cuda_flags(getattr(build_info, "cuda_flags", "none"))
    cxx_flags = getattr(build_info, "cxx_flags", "none")
    advisories = check_cxx_flags(cxx_flags)
    host_flags_seen = host_flags_were_recorded(cxx_flags)
    with_cuda = bool(getattr(build_info, "with_cuda", False))
    device_advisories: list[str] = []
    if not with_cuda:
        # DEVICE-SCOPED, or a CPU-only machine would be ungradeable: every CPU
        # run there would draw this advisory, and an advisory downgrades
        # `decision` to `advisory`: a CPU number's grade came to depend on
        # whether an unrelated extension had been compiled beside it.
        #
        # This is the exact mirror of the defect `host_advisories` documents.
        no_cuda = (
            "no CUDA extension is present, so no device arithmetic was checked. "
            "`ok` is vacuously true here -- require `checked_a_gpu_build` too "
            "before treating this as evidence the arithmetic is pinned.")
        advisories.append(no_cuda)
        device_advisories.append(no_cuda)
    if with_cuda and not host_flags_seen:
        advisories.append(
            "this build has a CUDA extension but its stamp records no host "
            "compiler flags, so the host half of the extension -- which "
            "prepares and reduces what the kernel sees -- was not inspected. "
            "An empty host-flag advisory list here means 'nothing to check', "
            "not 'checked and clean'. Advisory rather than fatal because the "
            "host-flag effect is not measured on this project.")
    if with_cuda:
        adv = check_toolkit(getattr(build_info, "nvcc_version", "none"))
        if adv:
            advisories.append(adv)
            device_advisories.append(adv)
        archs = getattr(build_info, "cuda_architectures", "") or ""
        if archs.strip() == "native":
            # A development build. CMake resolved `native` to whatever card is in
            # this machine, so the SASS may well be sm_120 -- but the stamp records
            # the literal string, and an unresolved stamp is not evidence. Say that,
            # rather than the flatly untrue "does not include sm_120": an advisory
            # that is wrong in an obvious way is one people learn to ignore.
            device_advisories.append(
                f"build targets 'native', so the stamp does not record which "
                f"architecture was compiled. It is a development build and is "
                f"not portable -- a wheel built this way fails to load, or "
                f"silently JIT-compiles from mismatched PTX, on any other GPU. "
                f"Do not quote a number from it; rebuild with the explicit "
                f"architecture list first."
            )
            advisories.append(
                f"build targets 'native', so the stamp does not record which "
                f"architecture was compiled. It is a development build and is "
                f"not portable -- a wheel built this way fails to load, or "
                f"silently JIT-compiles from mismatched PTX, on any other GPU. "
                f"Do not quote a number from it; rebuild with the explicit "
                f"architecture list first."
            )
        elif "120" not in archs:
            device_advisories.append(
                f"build library molecules {archs!r}, which does not include the "
                f"architecture this policy's numbers were measured on "
                f"({VALIDATED_CUDA_ARCH}). Scores from it are not bitwise "
                f"comparable with scores taken on {VALIDATED_CUDA_ARCH}: "
                f"sm_70 vs sm_120 measured bit-identical at a fixed pose but "
                f"|dT_shape| p99 = 2.6e-5 after 100 Adagrad steps "
                f"(doc/NUMERICS_POLICY.md section 8)."
            )
            advisories.append(
                f"build library molecules {archs!r}, which does not include the "
                f"architecture this policy's numbers were measured on "
                f"({VALIDATED_CUDA_ARCH}). Scores from it are not bitwise "
                f"comparable with scores taken on {VALIDATED_CUDA_ARCH}: "
                f"sm_70 vs sm_120 measured bit-identical at a fixed pose but "
                f"|dT_shape| p99 = 2.6e-5 after 100 Adagrad steps "
                f"(doc/NUMERICS_POLICY.md section 8)."
            )
    # The CPU extension is a second compiled artifact. Checked unconditionally rather than only on the CPU path,
    # because "which backend will be used" is not known here and inspecting an
    # artifact that then goes unused costs an import.
    with_cpu, cpu_adv = _cpu_flag_advisories()
    advisories.extend(cpu_adv)
    host_advisories = list(cpu_adv)
    if not with_cpu and not cpu_adv:
        no_cpu = (
            "no CPU extension is present, so no CPU arithmetic was checked. As "
            "with the CUDA half, `ok` is vacuously true about an artifact "
            "nobody inspected -- require `checked_a_cpu_build` too before "
            "reading this as evidence about a CPU number.")
        advisories.append(no_cpu)
        host_advisories.append(no_cpu)

    return ArithmeticCheck(violations=violations, advisories=advisories,
                           overridden=bool(os.environ.get(OVERRIDE_ENV)),
                           _with_cpu=with_cpu,
                           device_advisories=device_advisories,
                           host_advisories=host_advisories,
                           _with_cuda=with_cuda,
                           _host_flags_seen=host_flags_seen,
                           _cpu_host_flags_seen=with_cpu)


def compiled_wire_constants():
    """Return compiled CUDA constants, or Python constants without CUDA.

    Importing this module and reading these bindings does not create a context.
    """
    try:
        from . import _roshambo3_cuda as ext
    except ImportError:
        return float(WIRE_QUANT_STEP), 1.0 / float(WIRE_QUANT_STEP)
    return float(ext.compiled_wire_step()), float(ext.compiled_wire_inverse_step())


def _assert_wire_constants():
    from .format import constants
    from .format.r3g import limits, reference
    step = np.float32(WIRE_QUANT_STEP)
    for name, value in (("format.constants.COORD_SCALE", constants.COORD_SCALE),
                        ("limits.STEP_LOG2", 2.0 ** limits.STEP_LOG2),
                        ("reference.STEP_LOG2", 2.0 ** reference.STEP_LOG2)):
        if np.float32(value) != step:
            raise RuntimeError(f"WIRE_QUANT_STEP={float(step)} disagrees with {name}={value}")
    compiled, inverse = compiled_wire_constants()
    if np.float32(compiled) != step or np.float32(compiled) * np.float32(inverse) != np.float32(1):
        raise RuntimeError(f"WIRE_QUANT_STEP={float(step)} disagrees with compiled wire step={compiled}, inverse={inverse}; product must equal 1")


_checked = False


def assert_arithmetic_pinned(build_info=None, force: bool = False) -> ArithmeticCheck:
    """Raise unless the loaded build uses the pinned arithmetic.

    Called on the first use of a compute backend, so the policy is enforced by
    the code that would otherwise produce the wrong number rather than by a
    convention in a README.  Cheap: the result is cached after the first call
    unless ``force``.

    Raises:
        RuntimeError: a forbidden nvcc flag is compiled into the loaded
            extension and :data:`OVERRIDE_ENV` is not set.
    """
    global _checked
    _assert_wire_constants()
    if _checked and not force:
        return ArithmeticCheck()
    res = check_build(build_info)
    _checked = True
    if res.violations:
        msg = ("roshambo3: the loaded build does not use the pinned "
               "arithmetic.\n  " + "\n  ".join(res.violations) +
               "\n\nT24 measured that all six perturbing build axes reach the "
               "same output distribution, so there is no small-perturbation "
               "regime in which this is harmless: the policy is to pin the "
               "arithmetic, not to tolerate an epsilon (doc/NUMERICS_POLICY.md)."
               f"\nSet {OVERRIDE_ENV}=1 only if the point of this build is to "
               "measure the perturbation.")
        if res.overridden:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            raise RuntimeError(msg)
    for a in res.advisories:
        warnings.warn("roshambo3: " + a, RuntimeWarning, stacklevel=2)
    return res


def numerics_stamp(build_info=None) -> dict:
    """The block every run record must carry.

    A number that cannot say which arithmetic produced it is not reproducible,
    and "which arithmetic" is a specific, short list: the toolkit
    version, the nvcc flags, the architecture list, the git SHA of the source,
    and whether anyone reached for the override.
    """
    if build_info is None:
        try:
            from ._build_info import get_build_info

            build_info = get_build_info()
        except Exception:                      # pragma: no cover - no build
            build_info = None
    res = check_build(build_info) if build_info is not None else ArithmeticCheck()
    return {
        "policy": "doc/NUMERICS_POLICY.md",
        "pinned_cuda_toolkit": PINNED_CUDA_TOOLKIT,
        "validated_cuda_arch": VALIDATED_CUDA_ARCH,
        "cross_architecture_reproducibility": CROSS_ARCH_REPRODUCIBILITY,
        "cuda_toolkit": getattr(build_info, "nvcc_version", None),
        "cuda_flags": getattr(build_info, "cuda_flags", None),
        "cxx_flags": getattr(build_info, "cxx_flags", None),
        "cuda_architectures": getattr(build_info, "cuda_architectures", None),
        "build_type": getattr(build_info, "build_type", None),
        "git_sha": getattr(build_info, "git_sha", None),
        # Beside the commit, never instead of it. `git_sha` names what was
        # checked out; these name what was COMPILED, and the two disagree
        # exactly in the case the stamp exists to disambiguate -- a control arm,
        # a bisect, a patch script. `cuda_src_sha256` is the durable identity:
        # it is a hash of the bytes, so an uncommitted edit moves it where the
        # commit and the tree hash do not. `cuda_src_dirty` is null when nobody
        # looked, which is not the same as false.
        "cuda_src_sha256": getattr(build_info, "cuda_src_sha256", None),
        "cuda_src_tree": getattr(build_info, "cuda_src_tree", None),
        "cuda_src_dirty": getattr(build_info, "cuda_src_dirty", None),
        "cuda_src_dirty_files": list(
            getattr(build_info, "cuda_src_dirty_files", ()) or ()),
        "source_stamp_phase": getattr(build_info, "source_stamp_phase", None),
        "nthreads": getattr(build_info, "nthreads", None),
        "reduced_frame": {k: float(v) for k, v in DEVICE_CONSTANTS.items()},
        "wire_quantization": {
            "python_step": float(WIRE_QUANT_STEP),
            "compiled_step": compiled_wire_constants()[0],
            "compiled_inverse_step": compiled_wire_constants()[1],
        },
        "arithmetic_ok": res.ok,
        # `arithmetic_ok` means "found no forbidden flag",
        # which is vacuously true when there was nothing to inspect -- so a
        # record carrying it alone can be read as verified when nothing was
        # verified. These two say whether anyone looked, on each side.
        "arithmetic_checked_a_gpu_build": res.checked_a_gpu_build,
        "arithmetic_checked_host_flags": res.checked_host_flags,
        "arithmetic_violations": res.violations,
        "arithmetic_advisories": res.advisories,
        "numerics_override": res.overridden,
    }


# ---------------------------------------------------------------------------
# 3. Color interaction matrix: is the prefactor extraction the canonical one?
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionMatrixReport:
    canonical: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.canonical


def check_interaction_matrix(rmat, pmat) -> InteractionMatrixReport:
    """Report whether a color interaction matrix is the canonical diagonal one.

    ``PharmacophoreGenerator.set_interactions`` accepts an arbitrary list of
    (type, type, r, p) tuples, so an arbitrary matrix can reach the kernel.  The
    canonical case -- r = 1 and p = 1 exactly on the diagonal, p = 0 off it,
    off-diagonal r ignored -- is the
    one in which every color term shares the single prefactor (pi/2)^1.5, so
    the color Tanimoto is exactly scale-free and the color channel is one
    well-conditioned sum.

    The kernel handles a non-canonical matrix *exactly* (the per-pair residual
    p^2 r^-1.5 is folded into the accumulated term, not approximated away), so
    this is not a correctness guard.  It is a reporting guard: a caller who has
    changed the matrix is scoring on a differently conditioned color sum, and
    that has to appear in the run record rather than be discovered later.

    Args:
        rmat: interaction widths, square, index 0 is the "no feature" slot.
        pmat: interaction heights, same shape.

    Returns:
        An :class:`InteractionMatrixReport`; truthy iff canonical.
    """
    # One definition of "canonical", shared with the format: exact float32 diag r = diag p = 1 and off-diagonal p = 0 on
    # the color types 1..n, row/column 0 and off-diagonal widths ignored, and
    # the matrices valid (finite; widths 0 or >= 1e-6; heights >= 0).
    # ``format.r3g.reference.canonical_matrix_reasons`` states it; this is
    # the same function. The CPU/CUDA canonical kernels keep their own
    # stricter internal test (off-diagonal r == 1), a performance choice.
    from .format.r3g.reference import canonical_matrix_reasons
    r = np.asarray(rmat, dtype=np.float64)
    p = np.asarray(pmat, dtype=np.float64)
    if r.shape != p.shape or r.ndim != 2 or r.shape[0] != r.shape[1]:
        return InteractionMatrixReport(False, ["rmat/pmat are not the same square matrix"])
    if r.shape[0] <= 1:
        # the 1x1 all-zero placeholder handed in when color scoring is off
        return InteractionMatrixReport(True, [])
    reasons = canonical_matrix_reasons(r, p)
    return InteractionMatrixReport(not reasons, reasons)


# ---------------------------------------------------------------------------
# 4. The deterministic tie-break
# ---------------------------------------------------------------------------

def rank_order(scores, ids=None) -> np.ndarray:
    """Indices that sort ``scores`` descending, ties broken by ascending id.

    float32 ties are common at M = 1e7, and ``argsort()`` is not stable by
    default, so two runs -- or a gold and a candidate -- that agree perfectly on
    every score can still disagree on rank purely through sort order.  That
    manufactures "rank changes" out of nothing, and, worse, hides real ones in
    the noise.  This is the one ranking rule; it must be applied identically to
    both sides of any comparison (``metrics.rank_g`` is defined the same way).

    Args:
        scores: 1-D array of scores, higher is better.
        ids: 1-D array of canonical ids, ascending order wins a tie.  Defaults
            to the position in ``scores``, which makes the sort stable.

    Returns:
        Integer index array.
    """
    scores = np.asarray(scores)
    if ids is None:
        ids = np.arange(len(scores))
    ids = np.asarray(ids)
    if ids.shape != scores.shape:
        raise ValueError(f"ids {ids.shape} does not match scores {scores.shape}")
    # np.lexsort orders by the LAST key first: primary -scores, secondary ids
    return np.lexsort((ids, -scores))


# ---------------------------------------------------------------------------
# 5. Result invariants
# ---------------------------------------------------------------------------

#: Relative slack allowed when checking the result invariants.
#:
#: These are exact mathematical statements about the quantities a correct
#: kernel produces, evaluated on float32 sums of up to L^2 terms, so the
#: check has to allow accumulated rounding. It is deliberately tight: the
#: defects it exists to catch -- a pose that is not a rotation, an overlap
#: larger than the self-overlaps allow -- are wrong by percent, not by ulps.
#: This is not a tolerance on a *comparison* of two score sets, which the
#: numerics policy forbids; it is a tolerance on an inequality.
RESULT_INVARIANT_RTOL = 1e-3


def check_result_invariants(scores, *, rtol: float = RESULT_INVARIANT_RTOL):
    """Check an ``[n, 20]`` score block against what a correct result must obey.

    Two invariants, chosen because they are cheap and because neither can be
    satisfied by a kernel that agrees with a reference implementation *for the
    wrong reason*:

    1. **The reported pose is a rotation.** The quaternion must be unit norm.
       A non-unit quaternion is not a rigid motion, so the "pose" it describes
       scales the molecule -- and the overlap computed at it is not an overlap
       of the two shapes.
    2. **Cauchy-Schwarz on the overlap.** ``V_AB <= sqrt(V_AA * V_BB)`` for the
       shape channel and likewise for color, since each is an inner product of
       two Gaussian densities. Equivalently the Tanimoto is in ``[0, 1]``. An
       overlap exceeding this bound means the sum is not the inner product it
       is documented to be.

    This is the check ROSHAMBO2 ran on every call and the port dropped. It
    matters specifically because it shares no code with the kernel: **the
    C++ and CUDA paths once agreed with each other by
    sharing a typo**, and a cross-check between two implementations cannot see
    that. An invariant can.

    Args:
        scores: ``[n, 20]`` float32 block, or anything broadcastable to it.
        rtol: relative slack; see :data:`RESULT_INVARIANT_RTOL`.

    Returns:
        A list of human-readable violations, empty if the block is clean.
    """
    from . import scores as cols

    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != cols.N_SCORE_COLUMNS:
        raise ValueError(
            f"expected an [n, {cols.N_SCORE_COLUMNS}] score block, got {s.shape}")
    if s.shape[0] == 0:
        return []

    problems: list[str] = []

    finite = np.isfinite(s)
    if not finite.all():
        bad = int(np.flatnonzero(~finite.all(axis=1))[0])
        problems.append(
            f"{int((~finite).any(axis=1).sum())} row(s) carry a non-finite "
            f"value (first: row {bad})")

    q = s[:, cols.POSE_QUAT]
    norm = np.sqrt((q * q).sum(axis=1))
    # An all-zero quaternion is what an unwritten row looks like; report it as
    # its own thing rather than as a rotation that is merely off.
    unwritten = norm == 0.0
    off = ~unwritten & (np.abs(norm - 1.0) > rtol)
    if unwritten.any():
        problems.append(
            f"{int(unwritten.sum())} row(s) carry an all-zero pose quaternion, "
            f"which is not a rotation at all -- an unwritten row rather than a "
            f"wrong one")
    if off.any():
        i = int(np.flatnonzero(off)[0])
        problems.append(
            f"{int(off.sum())} row(s) have a pose quaternion that is not unit "
            f"norm, so the pose is not a rigid motion (first: row {i}, "
            f"|q| = {norm[i]:.6g})")

    for name, ab, aa, bb in (
            ("shape", cols.OVERLAP_VOLUME, cols.SELF_OVERLAP_QUERY,
             cols.SELF_OVERLAP_LIBRARY),
            ("color", cols.OVERLAP_COLOR, cols.SELF_COLOR_QUERY,
             cols.SELF_COLOR_LIBRARY)):
        v_ab, v_aa, v_bb = s[:, ab], s[:, aa], s[:, bb]
        # Rows with no self-overlap at all are the color channel before color
        # is wired through; an all-zero channel is vacuously consistent.
        live = (v_aa > 0) & (v_bb > 0)
        if not live.any():
            continue
        bound = np.sqrt(v_aa[live] * v_bb[live])
        violated = v_ab[live] > bound * (1.0 + rtol)
        if violated.any():
            j = int(np.flatnonzero(violated)[0])
            problems.append(
                f"{int(violated.sum())} row(s) violate Cauchy-Schwarz on the "
                f"{name} channel: V_AB = {v_ab[live][j]:.6g} exceeds "
                f"sqrt(V_AA*V_BB) = {bound[j]:.6g}")
        if (v_ab[live] < -rtol * bound).any():
            problems.append(
                f"row(s) report a negative {name} overlap, which a sum of "
                f"positive Gaussians cannot produce")

    return problems


def assert_result_invariants(scores, *, rtol: float = RESULT_INVARIANT_RTOL) -> None:
    """:func:`check_result_invariants`, raising instead of reporting."""
    problems = check_result_invariants(scores, rtol=rtol)
    if problems:
        raise AssertionError(
            "the result block violates invariants a correct kernel cannot:\n  "
            + "\n  ".join(problems))

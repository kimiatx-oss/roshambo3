"""Build configuration stamp for the compiled backend.

CMake bakes the build type, CUDA architecture list, nvcc version, compiler flags
and git SHA into the ``_roshambo3_cuda`` extension at build time (see the
``target_compile_definitions`` block in ``CMakeLists.txt``). Reading the stamp
back out of the compiled extension -- rather than from a separately generated
file -- means it can never go stale relative to the ``.so`` that is actually
loaded, and it survives editable installs, wheel installs and a manual
``cmake && make install`` identically.

Unlike ROSHAMBO2, there is no always-built CPU extension to carry the stamp:
ROSHAMBO3 is CUDA-only, so a machine with no toolkit gets a pure-Python install
and no extension at all. :func:`get_build_info` therefore reports
``with_cuda=False`` instead of raising, which is what lets the CPU CI job assert
the CUDA-less path works.

The benchmark harness consumes this to refuse to record a timing from a
non-Release build. See ``bench/rules.py`` and ``doc/NUMERICS_POLICY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildInfo:
    """Snapshot of the compile-time configuration of the CUDA backend."""

    build_type: str
    """``CMAKE_BUILD_TYPE`` the extension was compiled with. A benchmark
    harness must require ``Release``: a bare ``cmake ..`` leaves host code at
    -O0 with asserts on, which silently invalidates any timing."""

    git_sha: str
    """Commit SHA of the source tree at configure time, or ``"unknown"`` when
    built outside a git checkout (an sdist, for instance).

    **This names a commit, not the source that was compiled.** A worktree with a
    restored file, a bisect, or a patch script all produce a binary whose HEAD
    does not describe its ``_cuda_src``, and that is exactly the situation the
    stamp exists to disambiguate. Read :attr:`cuda_src_sha256` for what was compiled; this field is kept unchanged because archived run
    records key on it."""

    with_cuda: bool
    """Whether a CUDA compiler was found and ``_roshambo3_cuda`` was built."""

    cuda_architectures: str
    """Comma-separated ``CMAKE_CUDA_ARCHITECTURES`` actually used, or
    ``"none"``. A build that targets only ``native`` is not portable and a
    wheel must never be published from one."""

    nvcc_version: str
    """nvcc version used, or ``"none"``. This is a measured arithmetic axis --
    nvcc 12.9 and 13.2 produce different sm_120 SASS -- so a run record
    that omits it cannot be compared against one that does."""

    cxx_flags: str
    """Host compiler flags."""

    cuda_flags: str
    """Device compiler flags, or ``"none"``. ``roshambo3.numerics`` checks these
    against the pinned set; ``--use_fast_math`` here invalidates every score."""

    cuda_src_sha256: str = "unknown"
    """sha256 over the **bytes of every file that was compiled** -- the stamp's
    answer to "which source is this".

    Dirty-aware by construction: an uncommitted edit moves it, where
    :attr:`git_sha` and :attr:`cuda_src_tree` both stay put. It is the same
    algorithm, in the same module (``scripts/source_stamp.py``), that the
    binary reuse registry keys on -- so a binary's hash and the key it was
    stored under are the same number, and ``tests/test_source_stamp.py``
    asserts it rather than trusting it.

    ``"unknown"`` means the generator did not run (a compile driven by
    something other than this project's CMake). That is not "clean"."""

    cuda_src_tree: str = "unknown"
    """``git rev-parse HEAD:src/roshambo3/backend/_cuda_src`` at build time.

    Git's own content hash of the compiled subtree -- the key
    the baseline and accuracy harnesses settled on, and the right one for "do I need to rebuild" against a *committed* tree, because a
    pure-Python change moves ``src`` while leaving the kernel bit-identical.
    It describes the index, so it does **not** move for an uncommitted
    edit; :attr:`cuda_src_dirty` is what covers that, and
    :attr:`cuda_src_sha256` is what covers it without anyone remembering."""

    cuda_src_dirty: bool | None = None
    """Whether ``git status --porcelain`` on the compiled subtree was non-empty
    when this binary was built. ``None`` means **nobody looked**, which is not
    the same claim as ``False`` -- the same distinction
    ``checked_a_gpu_build`` and the architecture probe draw.

    True means the working tree matched no commit at all, so :attr:`git_sha`
    and :attr:`cuda_src_tree` are both naming source this binary does not
    contain. ``bench/rules.py`` refuses to record a decision-grade number from
    such a build."""

    cuda_src_dirty_files: tuple = ()
    """The porcelain paths, capped at 20. Present so a mismatch says *which*
    file, which is the first question asked of a restored ``kernels.cu``."""

    source_stamp_phase: str = "none"
    """When the source stamp was taken: ``"build"``, ``"configure"``, or
    ``"none"``.

    ``"build"`` is the shipped path and the only fully trustworthy one:
    ``cmake --build`` after an edit does not reconfigure, so a stamp taken only
    at configure time would call a configure-then-edit-then-build cycle clean.
    ``"none"`` means there is no stamp at all and every field above is
    ``"unknown"``."""

    nthreads: int = 256
    """Combo optimizer block size. Measurement builds use 128/256/512."""

    compiler_launcher: str = ""
    """The compiler cache this build ran through, or empty for none.

    ``kernels.cu`` is one translation unit, so a cache is all-or-nothing per
    file: it does nothing for an edit-compile loop and everything for rebuilding
    identical source in a different tree, which is what every A/B control arm
    does. Recorded rather than left implicit because a cache is a thing that can
    hand back the wrong object, and a number whose binary cannot say whether one
    was in play cannot be argued with."""

    final_exact_shape: bool = True
    """Whether this build evaluates the final exact shape rescore."""

    final_exact_color: bool = True
    """Whether this build evaluates the final exact color rescore."""

    @property
    def source_identified(self) -> bool:
        """Whether this binary can say which source it was compiled from.

        False on a build whose stamp generator did not run. A run record from
        such a binary carries a commit and nothing that could contradict it,
        which is the state the source stamp exists to end."""
        return self.cuda_src_sha256 not in ("", "unknown")

    @property
    def is_release(self) -> bool:
        return self.build_type == "Release"


def _tristate(value) -> bool | None:
    """The stamp's -1/0/1 dirty marker as True/False/None.

    None is deliberately not False: "the dirty check did not run" and "the tree
    was clean" are different claims, and collapsing them is how a build that
    nobody inspected comes to look inspected."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return None if v < 0 else bool(v)


_NO_CUDA = BuildInfo(
    build_type="none",
    git_sha="unknown",
    with_cuda=False,
    cuda_architectures="none",
    nvcc_version="none",
    cxx_flags="none",
    cuda_flags="none",
)


def get_build_info() -> BuildInfo:
    """Return the :class:`BuildInfo` for the loaded extension.

    Never raises: when the CUDA extension is absent the returned stamp has
    ``with_cuda=False`` and ``build_type="none"``, which callers must treat as
    "no GPU backend here", not as a Release build.
    """
    try:
        from . import _roshambo3_cuda as _ext  # type: ignore[attr-defined]
    except ImportError:
        return _NO_CUDA

    return BuildInfo(
        build_type=_ext.__build_type__,
        git_sha=_ext.__git_sha__,
        with_cuda=True,
        cuda_architectures=_ext.__cuda_architectures__,
        nvcc_version=_ext.__nvcc_version__,
        cxx_flags=_ext.__cxx_flags__,
        cuda_flags=_ext.__cuda_flags__,
        cuda_src_sha256=str(getattr(_ext, "__cuda_src_sha256__", "unknown")),
        cuda_src_tree=str(getattr(_ext, "__cuda_src_tree__", "unknown")),
        # -1 is the stamp's "nobody looked"; None is this dataclass's. A binary
        # built before the stamp existed has no attribute at all, which is the same thing.
        cuda_src_dirty=_tristate(getattr(_ext, "__cuda_src_dirty__", -1)),
        cuda_src_dirty_files=tuple(
            f for f in str(
                getattr(_ext, "__cuda_src_dirty_files__", "")).split(",") if f),
        source_stamp_phase=str(
            getattr(_ext, "__source_stamp_phase__", "none")),
        nthreads=int(getattr(_ext, "__nthreads__", 256)),
        compiler_launcher=str(getattr(_ext, "__compiler_launcher__", "")),
        final_exact_shape=bool(getattr(_ext, "__final_exact_shape__", True)),
        final_exact_color=bool(getattr(_ext, "__final_exact_color__", True)),
    )

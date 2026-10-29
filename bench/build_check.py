"""Build-stamp verification: the enforcement point against timing a binary
built for the wrong architecture or at the wrong optimization level.

The build bakes a machine-readable build stamp (``roshambo3.get_build_info()``) into
the compiled extension so it can never go stale relative to the ``.so``
actually loaded. This module is where the harness turns that stamp into an
enforcement decision: a benchmark harness that only *prints* "warning:
non-Release build" is advisory-shaped tooling wearing a mandatory-shaped
costume. ``verify_build()`` is the thing every timed run must call before it
is allowed to record anything.

It also independently recovers the two build-stamp fields ``BuildInfo``
does not carry but the run-record schema requires: the actual SASS/PTX
architecture list embedded in the loaded ``.so`` (via ``cuobjdump``, i.e. not
merely trusting the CMake variable that requested it) and the ``.so``'s
sha256 (so a run record can prove exactly which binary produced a number).
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class VerifiedBuild:
    build_type: str
    cxx_flags: str
    cuda_flags: str
    with_cuda: bool
    cuda_architectures_requested: str
    nvcc_version: str
    git_sha: str
    """The commit that was checked out at configure time. It names an index,
    not the source that was compiled -- see the four fields below."""
    cuda_src_sha256: str = "unknown"
    """sha256 over the bytes of every compiled file, from the loaded
    extension's own stamp. The identity that moves for an uncommitted edit."""
    cuda_src_tree: str = "unknown"
    """`git rev-parse HEAD:src/roshambo3/backend/_cuda_src` at build time."""
    cuda_src_dirty: "bool | None" = None
    """Whether the compiled subtree was dirty when this binary was built.
    ``None`` means the check did not run -- not that it passed."""
    cuda_src_dirty_files: list = field(default_factory=list)
    cuda_src_stamp_phase: str = "none"
    """"build", "configure" or "none" -- when the three fields above were
    taken. Only "build" survives a configure-then-edit-then-build cycle."""
    nthreads: int = 256
    arch_list: list[str] = field(default_factory=list)
    """SASS architectures actually embedded in the loaded .so, from
    ``cuobjdump --list-elf`` -- independent of what CMake was *told* to
    build, in case of a stale/mismatched artifact."""
    ptx_present: bool = False
    """Whether any forward-compatible PTX is embedded (JIT fallback path for
    GPUs newer than any SASS target)."""
    so_sha256: dict[str, str] = field(default_factory=dict)
    """{"_roshambo3_cuda": sha256}"""
    arch_probe_error: "str | None" = None
    """Why :attr:`arch_list` is empty, when it is empty because the probe could
    not run rather than because the build has no CUDA. A run whose numbers will
    be quoted must not have this set: it means the architecture went unchecked."""

    # ---- The CPU extension is a second compiled artifact, and every
    # field above describes the CUDA one. Recording a CPU run through the
    # fields above does not merely leave gaps -- it fills them with ANOTHER
    # BINARY'S answers. `get_build_info().cxx_flags` is the CUDA extension's
    # ` -O3 -DNDEBUG`, while the CPU extension carries
    # `-march=x86-64-v3 -fno-fast-math -fno-associative-math`. A record that
    # named the first for a number produced by the second would be wrong rather
    # than incomplete, which is worse and much harder to notice.
    with_cpu: bool = False
    """Whether a CPU extension is loadable at all. The CPU analogue of
    :attr:`with_cuda`, and false is a legitimate state (a CUDA-only build)."""
    cpu_src_sha256: str = "unknown"
    """sha256 over the bytes of every file in ``_cpu_src/``, from the loaded
    extension's own stamp. The identity that moves for an uncommitted
    edit, and the thing that makes a CPU number attributable to a binary."""
    cpu_src_tree: str = "unknown"
    cpu_src_dirty: "bool | None" = None
    """``None`` means the check did not run -- not that it passed."""
    cpu_src_dirty_files: list = field(default_factory=list)
    cpu_src_stamp_phase: str = "none"
    cpu_build_type: str = "unknown"
    """The CPU extension's own Release/Debug, which is not required to agree
    with :attr:`build_type` -- they are separate compilations."""
    cpu_cxx_flags: str = "unknown"
    """The flags the CPU extension was actually compiled with. Distinct from
    :attr:`cxx_flags`, which is the CUDA extension's host half."""
    cpu_compiled_targets: list = field(default_factory=list)
    """SIMD targets the binary CONTAINS. Not what the process selected -- that
    is a different question and is asked separately."""
    cpu_selected_target: "str | None" = None
    """The target runtime dispatch actually chose, asked of the process."""
    cpu_vector_width: "int | None" = None
    """A CPU score is a function of the vector width, so a record that
    does not name its width cannot be compared with anything."""
    cpu_probe_error: "str | None" = None
    """Why the CPU fields are at their defaults, when they are at their
    defaults because the probe could not run rather than because there is no
    CPU build. The twin of :attr:`arch_probe_error`, and it exists for the same
    reason: "did not look" and "looked and found nothing" must not serialize
    identically."""

    @property
    def is_release(self) -> bool:
        return self.build_type == "Release"

    @property
    def cpu_is_release(self) -> bool:
        """The CPU extension's own Release check.

        Separate from :attr:`is_release` because they are separate
        compilations: a tree can carry a Release CUDA extension and a Debug CPU
        one, and a CPU timing taken against the latter is not a number.
        """
        return self.cpu_build_type == "Release"

    @property
    def arch_profile(self) -> str:
        """"native" (exactly one SASS arch embedded -- e.g. the
        ``ROSHAMBO_ARCH_NATIVE=ON`` dev path targeting only the build
        machine's own GPU) vs "fat" (more than one, e.g. the default
        80;86;89;90;120 wheel-distribution build) vs "unknown" (no CUDA
        build, or ``cuobjdump`` unavailable so the list couldn't be
        recovered -- :attr:`arch_probe_error` distinguishes the last case).

        This distinction matters because a fat multi-arch
        binary changes register allocation and can change occupancy relative
        to a native single-arch build -- so a throughput number measured on
        one is not directly comparable to a number measured on the other.
        See :func:`bench.rules.check_arch_profile_comparable`.
        """
        if not self.arch_list:
            return "unknown"
        return "native" if len(self.arch_list) == 1 else "fat"

    #: Fields deliberately absent from :meth:`as_dict`, and why. Anything not
    #: named here must be serialized -- ``tests/test_bench_rules.py`` asserts
    #: the two sets cover every field, because this dict used to be a
    #: hand-written literal and silently dropped ``arch_probe_error``: a run on
    #: a machine where ``cuobjdump`` is missing wrote a record indistinguishable
    #: from one where the architecture had actually been checked.
    _NOT_SERIALIZED = {
        # Emitted under different keys, immediately below.
        "build_type", "cxx_flags", "cuda_flags",
        # The run record carries git provenance separately, from GitInfo.
        "git_sha",
    }

    def as_dict(self) -> dict:
        d = {
            "type": self.build_type,
            "flags": {"cxx": self.cxx_flags, "cuda": self.cuda_flags},
            # Derived, not a field: kept because a reader of the record should
            # not have to re-implement the native/fat rule to compare two runs.
            "arch_profile": self.arch_profile,
        }
        for f in fields(self):
            if f.name not in self._NOT_SERIALIZED:
                d[f.name] = getattr(self, f.name)
        return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cuobjdump_arch_info(so_path: Path) -> tuple[list[str], bool, "str | None"]:
    """Return (sorted SASS arch list, ptx_present, why_not) for an extension.

    Read from the file on disk, not from the CMake variable that requested the
    build -- the failure is a library compiled for the wrong architecture that
    JIT-compiles at runtime, and a build stamp cannot see that.

    ``why_not`` is the third value because the first two cannot express "the
    probe did not run". It used to return ``([], False)`` when ``cuobjdump``
    was absent, which is indistinguishable from "probed, and this build has no
    SASS and no PTX" -- so on any machine where the toolkit is installed but
    not on `PATH` (which is every default NVIDIA apt install) the check that
    guards the highest-yield mistake in this project silently did not run.
    """
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return [], False, ("cuobjdump is not on PATH, so the loaded .so's "
                           "architectures could not be read. Add the CUDA "
                           "toolkit's bin directory to PATH.")
    archs: set[str] = set()
    ptx = False
    try:
        elf = subprocess.run(
            [cuobjdump, "--list-elf", str(so_path)],
            capture_output=True, text=True, check=True,
        ).stdout
        for line in elf.splitlines():
            # e.g. "ELF file    1: cuda_functions.sm_120.cubin"
            if ".sm_" in line:
                archs.add("sm_" + line.rsplit(".sm_", 1)[1].split(".")[0])
        ptxout = subprocess.run(
            [cuobjdump, "--list-ptx", str(so_path)],
            capture_output=True, text=True, check=True,
        ).stdout
        ptx = bool(ptxout.strip())
    except subprocess.CalledProcessError as e:
        return [], False, f"cuobjdump failed on {so_path}: {e}"
    return sorted(archs), ptx, None


def _cpu_build_info(so_hashes: dict) -> dict:
    """Ask the LOADED CPU extension what it is, or say why we could not.

    Same contract as the CUDA half of :func:`verify_build`: never raises. A
    machine with no CPU extension is a legitimate state and must produce a
    *recordable* verdict rather than an exception, and
    refusing is ``rules.enforce``'s job, not this function's.

    The distinction this function exists to preserve: ``{}`` from a missing
    extension and ``{}`` from a failed probe are different facts, so the second
    sets ``cpu_probe_error`` and the first does not.
    """
    try:
        from roshambo3.backend import cpu as _cpu_backend
    except ImportError as exc:
        return {"with_cpu": False, "cpu_probe_error": None,
                "_why": f"no CPU backend module ({exc})"}

    try:
        stamp = _cpu_backend.build_stamp()
    except Exception as exc:                       # noqa: BLE001
        # Deliberately broad. The stamp reaches into a compiled module's dunder
        # attributes; a build missing any one of them raises something specific
        # and uninteresting, and the useful outcome is always the same: record
        # that the probe failed, with the reason, and let enforce() refuse.
        return {"with_cpu": False,
                "cpu_probe_error": f"{type(exc).__name__}: {exc}"}

    try:
        from roshambo3 import _roshambo3_cpu as _ext
        so_path = Path(_ext.__file__).resolve()
        if so_path.is_file():
            so_hashes["_roshambo3_cpu"] = _sha256(so_path)
    except (ImportError, AttributeError, TypeError):
        pass

    return {
        "with_cpu": True,
        "cpu_src_sha256": str(stamp.get("cpu_src_sha256", "unknown")),
        "cpu_src_tree": str(stamp.get("cpu_src_tree", "unknown")),
        # The extension stamps this as an int; None must stay reachable and
        # must mean "the check did not run", so absence is not coerced to False.
        "cpu_src_dirty": (None if stamp.get("cpu_src_dirty") is None
                          else bool(stamp["cpu_src_dirty"])),
        "cpu_src_dirty_files": list(stamp.get("cpu_src_dirty_files", [])),
        "cpu_src_stamp_phase": str(stamp.get("source_stamp_phase", "none")),
        "cpu_build_type": str(stamp.get("build_type", "unknown")),
        "cpu_cxx_flags": str(stamp.get("cxx_flags", "unknown")),
        "cpu_compiled_targets": list(stamp.get("compiled_targets", [])),
        "cpu_selected_target": stamp.get("selected_target"),
        "cpu_vector_width": (None if stamp.get("vector_width") is None
                             else int(stamp["vector_width"])),
        "cpu_probe_error": None,
    }


def verify_build() -> VerifiedBuild:
    """Import the compiled backend and build a :class:`VerifiedBuild`.

    Never raises on a missing extension. ``get_build_info()`` reports
    ``with_cuda=False`` rather than failing, and a *refused* record on a machine
    with no GPU build is exactly what the harness must be able to write.
    The refusal is `rules.enforce`'s job, not this function's.

    The ``.so`` is hashed and its architectures read from the **loaded** object,
    not from the source tree or the build stamp. The
    highest-yield mistake in this project's history was timing a library
    compiled for the wrong architecture and silently JIT-ing at runtime, and a
    build stamp cannot detect it.
    """
    import roshambo3

    b = roshambo3.get_build_info()

    # Ask the LOADED extension module where it is, rather than globbing beside
    # the package. Under an editable install the two are different directories --
    # the Python package stays in src/ while the .so goes to site-packages -- so
    # a glob finds nothing and silently reports an empty architecture list. And
    # inspecting the object actually loaded is the point anyway.
    so_hashes: dict[str, str] = {}
    arch_list: list[str] = []
    ptx_present = False
    try:
        from roshambo3 import _roshambo3_cuda as _ext
        so_path = Path(_ext.__file__).resolve()
    except (ImportError, AttributeError, TypeError):
        so_path = None
    arch_err = None
    if so_path is not None and so_path.is_file():
        so_hashes["_roshambo3_cuda"] = _sha256(so_path)
        if b.with_cuda:
            arch_list, ptx_present, arch_err = _cuobjdump_arch_info(so_path)

    # Mutates so_hashes to add "_roshambo3_cpu" when there is one, so a
    # record proves which CPU binary produced its number the same way it
    # already proves which CUDA one did.
    cpu = _cpu_build_info(so_hashes)
    cpu.pop("_why", None)

    return VerifiedBuild(
        build_type=b.build_type,
        cxx_flags=b.cxx_flags,
        cuda_flags=b.cuda_flags,
        with_cuda=b.with_cuda,
        cuda_architectures_requested=b.cuda_architectures,
        nvcc_version=b.nvcc_version,
        git_sha=b.git_sha,
        cuda_src_sha256=b.cuda_src_sha256,
        cuda_src_tree=b.cuda_src_tree,
        cuda_src_dirty=b.cuda_src_dirty,
        cuda_src_dirty_files=list(b.cuda_src_dirty_files),
        cuda_src_stamp_phase=b.source_stamp_phase,
        nthreads=b.nthreads,
        arch_list=arch_list,
        ptx_present=ptx_present,
        so_sha256=so_hashes,
        arch_probe_error=arch_err,
        **cpu,
    )

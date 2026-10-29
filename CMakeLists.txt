cmake_minimum_required(VERSION 3.24)  # 3.24+ for native CUDA arch detection

# CUDA is deliberately NOT a project() language. Enabling a language locks in
# CMake's own default for CMAKE_CUDA_ARCHITECTURES *at that moment*; setting the
# variable afterwards is silently ignored because it is already DEFINED. CUDA is
# enabled explicitly below, after our default is set, and only if a working
# compiler is found -- so a CUDA-less machine still installs the Python package.
project(roshambo3 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Opt-in fast dev builds: target only the GPU(s) on this machine. OFF by default
# because such a build is not portable -- the wheel fails to load, or silently
# JITs from mismatched PTX, on any other GPU.
option(ROSHAMBO3_ARCH_NATIVE "Target only the detected CUDA architecture(s). Fast dev builds; NOT portable." OFF)
if(DEFINED ROSHAMBO3_NTHREADS AND NOT ROSHAMBO3_NTHREADS STREQUAL "256")
    message(FATAL_ERROR "The tool build uses 256 CUDA threads; other values are not part of this release")
endif()
set(ROSHAMBO3_NTHREADS 256)
foreach(_r3_final SHAPE COLOR)
    if(DEFINED ROSHAMBO3_FINAL_EXACT_${_r3_final} AND NOT ROSHAMBO3_FINAL_EXACT_${_r3_final})
        message(FATAL_ERROR "Exact final scoring is required in the tool release")
    endif()
endforeach()
set(ROSHAMBO3_FINAL_EXACT_SHAPE ON)
set(ROSHAMBO3_FINAL_EXACT_COLOR ON)

# A compiler cache, if one is installed. `kernels.cu` is ONE translation unit,
# so the cache is all-or-nothing per file and it does NOT help an edit-compile
# loop -- the file changed, so the hash changed. What it helps is exactly the
# pattern this project uses constantly: rebuilding *identical* source in a
# different tree, which is what every control arm and every A/B does.
#
# It is recorded in the build stamp rather than left implicit: a compiler cache
# is a thing that can hand back the wrong object, and a run record that cannot
# say whether one was in play cannot be argued with. Set to OFF, or to a full
# path, to control it explicitly.
option(ROSHAMBO3_CCACHE
    "Use ccache (or sccache) as the compiler launcher when one is installed." ON)
set(ROSHAMBO3_COMPILER_LAUNCHER "" CACHE STRING
    "Explicit compiler launcher; empty means auto-detect when ROSHAMBO3_CCACHE.")

set(_r3_launcher "")
if(ROSHAMBO3_COMPILER_LAUNCHER)
    set(_r3_launcher "${ROSHAMBO3_COMPILER_LAUNCHER}")
elseif(ROSHAMBO3_CCACHE)
    find_program(_r3_ccache NAMES ccache sccache)
    if(_r3_ccache)
        set(_r3_launcher "${_r3_ccache}")
    endif()
endif()
if(_r3_launcher)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${_r3_launcher}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${_r3_launcher}")
    message(STATUS "roshambo3: compiler launcher ${_r3_launcher}")
else()
    message(STATUS "roshambo3: no compiler launcher (no ccache found, or off)")
endif()

# The architectures we WANT, oldest first. Not what we get: the answer is the
# intersection of this with whatever the installed nvcc still supports, which is
# discovered rather than assumed (see below).
#
# sm_70 is here because nothing in this code needs Ampere -- there is not one
# `__CUDA_ARCH__` guard in either translation unit -- and Volta is measured
# working. CUDA 13 dropped it, CUDA 12 has it, and the
# build should simply do the right thing on each.
set(ROSHAMBO3_ARCH_WISHLIST "70;75;80;86;89;90;120" CACHE STRING
    "Architectures to build for, if the installed CUDA toolkit supports them.")

set(ROSHAMBO3_CUDA "AUTO" CACHE STRING
    "Build the CUDA extension: ON (require), OFF (skip), AUTO (build if nvcc is found).")
set_property(CACHE ROSHAMBO3_CUDA PROPERTY STRINGS AUTO ON OFF)
if(NOT ROSHAMBO3_CUDA MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "ROSHAMBO3_CUDA must be AUTO, ON, or OFF")
endif()

include(CheckLanguage)
if(NOT ROSHAMBO3_CUDA STREQUAL "OFF")
    check_language(CUDA)
endif()

# ---------------------------------------------------------------------------
# Ask nvcc which architectures it supports, rather than hardcoding a
# toolkit-version table.
#
# `nvcc --list-gpu-arch` prints one `compute_NN` per line and is the toolkit
# describing itself, so a future release dropping an architecture needs no edit
# here -- it just stops appearing. A version->arch table would need maintaining
# forever and would be wrong for exactly as long as nobody noticed.
#
# Ordering matters and is easy to get wrong: CMAKE_CUDA_ARCHITECTURES must be set
# BEFORE the CUDA language is enabled, because enabling it fixes the variable's
# default and a later `set()` is silently ignored. `check_language(CUDA)` gives
# us CMAKE_CUDA_COMPILER without enabling anything, which is the window this
# probe runs in.
# ---------------------------------------------------------------------------
function(roshambo3_supported_archs out_var)
    set(${out_var} "" PARENT_SCOPE)
    if(NOT CMAKE_CUDA_COMPILER)
        return()
    endif()
    execute_process(
        COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
        OUTPUT_VARIABLE _out ERROR_VARIABLE _err
        RESULT_VARIABLE _rc OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _rc EQUAL 0)
        message(STATUS "roshambo3: nvcc --list-gpu-arch failed (rc=${_rc}); "
                       "falling back to the wishlist unfiltered.")
        return()
    endif()
    string(REGEX MATCHALL "compute_([0-9]+)" _matches "${_out}")
    set(_archs "")
    foreach(_m IN LISTS _matches)
        string(REGEX REPLACE "compute_" "" _n "${_m}")
        list(APPEND _archs "${_n}")
    endforeach()
    list(REMOVE_DUPLICATES _archs)
    set(${out_var} "${_archs}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(ROSHAMBO3_ARCH_NATIVE)
        # A development build. Not portable, and numerics.check_build() reports it
        # as such so that no number from it gets quoted.
        set(CMAKE_CUDA_ARCHITECTURES native)
        message(STATUS "roshambo3: CUDA architectures = native (development build)")
    elseif(CMAKE_CUDA_COMPILER)
        roshambo3_supported_archs(_r3_supported)
        set(_r3_selected "")
        foreach(_a IN LISTS ROSHAMBO3_ARCH_WISHLIST)
            if(NOT _r3_supported OR "${_a}" IN_LIST _r3_supported)
                list(APPEND _r3_selected "${_a}")
            endif()
        endforeach()
        if(NOT _r3_selected)
            message(FATAL_ERROR
                "roshambo3: this CUDA toolkit supports none of "
                "ROSHAMBO3_ARCH_WISHLIST (${ROSHAMBO3_ARCH_WISHLIST}). "
                "Supported: ${_r3_supported}")
        endif()

        # SASS for every selected architecture, plus embedded PTX for the newest
        # one only, so the driver can JIT for hardware newer than anything we
        # targeted. PTX for every arch would bloat the binary for no benefit --
        # the driver only ever JITs from the newest it can use.
        set(_r3_arch_arg "")
        foreach(_a IN LISTS _r3_selected)
            list(APPEND _r3_arch_arg "${_a}-real")
        endforeach()
        list(GET _r3_selected -1 _r3_newest)
        list(APPEND _r3_arch_arg "${_r3_newest}-virtual")
        set(CMAKE_CUDA_ARCHITECTURES "${_r3_arch_arg}")

        set(_r3_dropped "")
        foreach(_a IN LISTS ROSHAMBO3_ARCH_WISHLIST)
            if(NOT "${_a}" IN_LIST _r3_selected)
                list(APPEND _r3_dropped "${_a}")
            endif()
        endforeach()
        message(STATUS "roshambo3: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")
        if(_r3_dropped)
            message(STATUS "roshambo3: this toolkit does not support "
                           "sm_${_r3_dropped} -- omitted.")
        endif()
    endif()
endif()

set(_r3_src ${CMAKE_CURRENT_SOURCE_DIR}/src/roshambo3/backend/_cuda_src)
# The native sources land in phase C of the port. Until then the tree is still a
# valid, installable pure-Python package -- phase B is pure Python and has to be
# testable before any kernel exists.
if(EXISTS ${_r3_src}/kernels.cu)
    set(_r3_have_sources ON)
else()
    set(_r3_have_sources OFF)
    message(STATUS "roshambo3: native sources not present yet -- installing the Python package only.")
endif()

# AUTO builds the extension when a CUDA compiler is present, which is the
# historical behavior and stays the default. OFF skips it outright.
#
# OFF exists because `kernels.cu` is one translation unit of ~10,000 lines and
# takes minutes even at the pruned default cell set, and a great deal of work in
# this repository does not touch it: the data lane (`prep/`, `format/`), the
# metrics, the harness and every task whose corpora are built rather than
# screened. Paying a kernel compile to run `roshambo3 prepare` is pure waste, and
# before this option the only way to avoid it was to hide `nvcc` from CMake,
# which nobody would guess.
#
#     CMAKE_ARGS='-DROSHAMBO3_CUDA=OFF' uv pip install -e .
#
# The package imports and the CPU paths work; `_build_info.with_cuda` is False,
# which is what the CPU CI job already asserts on, and any GPU entry point
# raises with that reason rather than failing obscurely.

# Instrument only the CPU Python extension when a sanitizer build is requested.
# This is intended for format-3 memory-safety tests and fuzzing, not release
# wheels or the CUDA extension. Keep the selected flags in the CPU build stamp.
set(ROSHAMBO3_SANITIZE "" CACHE STRING
    "Sanitizers for the CPU extension (supported: address,undefined).")
set_property(CACHE ROSHAMBO3_SANITIZE PROPERTY STRINGS "" address,undefined)
if(NOT ROSHAMBO3_SANITIZE STREQUAL "")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        message(FATAL_ERROR "roshambo3: ROSHAMBO3_SANITIZE requires GCC or Clang, got ${CMAKE_CXX_COMPILER_ID}.")
    endif()
    if(NOT ROSHAMBO3_SANITIZE MATCHES "^(address|undefined)(,(address|undefined))*$")
        message(FATAL_ERROR "roshambo3: ROSHAMBO3_SANITIZE must be a comma-separated combination of address and undefined.")
    endif()
    set(_r3_cpu_sanitize_flag "-fsanitize=${ROSHAMBO3_SANITIZE}")
endif()

if(ROSHAMBO3_CUDA STREQUAL "OFF")
    set(ROSHAMBO3_WITH_CUDA OFF)
    message(STATUS "roshambo3: ROSHAMBO3_CUDA=OFF -- skipping the GPU extension. "
                   "CPU paths only; nothing that needs a GPU will run.")
elseif(CMAKE_CUDA_COMPILER AND _r3_have_sources)
    enable_language(CUDA)
    set(ROSHAMBO3_WITH_CUDA ON)
elseif(ROSHAMBO3_CUDA STREQUAL "ON")
    message(FATAL_ERROR "roshambo3: ROSHAMBO3_CUDA=ON was requested but no working CUDA "
                        "compiler was found. Set it to AUTO or OFF, or fix the toolchain.")
elseif(NOT CMAKE_CUDA_COMPILER)
    set(ROSHAMBO3_WITH_CUDA OFF)
    message(WARNING "roshambo3: no working CUDA compiler found -- the GPU extension will not be built.")
else()
    set(ROSHAMBO3_WITH_CUDA OFF)
endif()

# ---------------------------------------------------------------------------
# Build stamp. Captured so the benchmark harness can refuse to record a
# non-Release run (see bench/rules.py and doc/NUMERICS_POLICY.md). Baked into
# the extension as compile definitions rather than a generated file, so it
# cannot go stale or get lost between editable and wheel installs.
# ---------------------------------------------------------------------------
# The commit is this tree's own HEAD when the source directory is the top of a
# git checkout; otherwise (an sdist or tarball, or a tree inside an unrelated
# repository) "unknown". The source stamp below identifies the compiled bytes
# either way.
set(ROSHAMBO3_GIT_SHA "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --show-toplevel
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _r3_git_root OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_r3_git_root STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE ROSHAMBO3_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY)
    endif()
endif()

# ---------------------------------------------------------------------------
# Which SOURCE was compiled -- as opposed to which commit was checked out.
#
# ROSHAMBO3_GIT_SHA above is `git rev-parse HEAD` at configure time. It names an
# index, and every A/B arm this project measures is built from a tree that
# departs from its index: a bisect, a patch script, a control arm. A file
# restored into a worktree yields a binary stamping a commit whose code it does
# not contain, and nothing in a run record from that binary can contradict it.
#
# So the extension also carries a sha256 over the BYTES of every file about to
# be compiled -- dirty-aware by construction rather than by anyone remembering
# to check -- plus the subtree tree-hash and a porcelain-dirty marker as
# provenance. ONE implementation, shared with the binary reuse registry:
# scripts/source_stamp.py, which that registry now calls for its key. Two
# implementations of "which source is this" is the drift to avoid.
#
# Generated at BUILD time, not only at configure time: `cmake --build` after an
# edit does not reconfigure, so a configure-then-edit-then-build cycle would
# otherwise stamp the pre-edit source and call it clean. The generator rewrites
# the header only when its content changes, so a build that did not touch the
# source does not recompile interface.cpp for a stamp that did not move.
#
# It cannot fail a build (the registry's rule too, same reason): any
# problem yields "unknown" and a -1 dirty marker, which `bench/rules.py` reads
# as "nobody looked" rather than as "clean".
# ---------------------------------------------------------------------------
set(_r3_stamp_script "${CMAKE_CURRENT_SOURCE_DIR}/scripts/source_stamp.py")
set(_r3_stamp_header "${CMAKE_CURRENT_BINARY_DIR}/roshambo3_source_stamp.h")
# The scalar CPU kernel is the reference anchor, so it needs the same
# content-addressed identity for the same reason -- and it needs it
# harder, because a fixture generated by the anchor is only interpretable if it
# can name the exact source that produced it. Same generator, same phases,
# different subtree and macro family (`--prefix cpu`); scripts/source_stamp.py
# is parameterized rather than copied, because two implementations of "which
# source is this" is precisely the drift to avoid.
set(_r3_cpu_stamp_header "${CMAKE_CURRENT_BINARY_DIR}/roshambo3_cpu_source_stamp.h")

# Deliberately NOT find_package(Python3): pybind11 and scikit-build-core do
# their own Python discovery below, and a second one here can select a
# different interpreter for the extension than the one the caller asked for.
# The generator is stdlib-only, so any python3 will do, and the interpreter the
# caller already named wins when there is one.
set(_r3_stamp_python "")
foreach(_c "${Python_EXECUTABLE}" "${PYTHON_EXECUTABLE}" "${Python3_EXECUTABLE}")
    if(_c AND EXISTS "${_c}" AND NOT _r3_stamp_python)
        set(_r3_stamp_python "${_c}")
    endif()
endforeach()
if(NOT _r3_stamp_python)
    find_program(_r3_stamp_python_found NAMES python3 python)
    if(_r3_stamp_python_found)
        set(_r3_stamp_python "${_r3_stamp_python_found}")
    endif()
endif()

set(_r3_stamp_ok OFF)
if(_r3_stamp_python AND EXISTS "${_r3_stamp_script}")
    execute_process(
        COMMAND "${_r3_stamp_python}" "${_r3_stamp_script}"
                --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                --phase configure --header "${_r3_stamp_header}"
        RESULT_VARIABLE _r3_stamp_rc
        OUTPUT_VARIABLE _r3_stamp_out OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_r3_stamp_rc EQUAL 0)
        set(_r3_stamp_ok ON)
        message(STATUS "roshambo3: ${_r3_stamp_out}")
        execute_process(
            COMMAND "${_r3_stamp_python}" "${_r3_stamp_script}"
                    --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                    --subtree "src/roshambo3/backend/_cpu_src" --prefix cpu
                    --phase configure --header "${_r3_cpu_stamp_header}"
            RESULT_VARIABLE _r3_cpu_stamp_rc
            OUTPUT_VARIABLE _r3_cpu_stamp_out OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_r3_cpu_stamp_rc EQUAL 0)
            message(STATUS "roshambo3: ${_r3_cpu_stamp_out}")
        else()
            message(WARNING "roshambo3: the CPU source stamp generator failed "
                            "(rc=${_r3_cpu_stamp_rc}); the anchor will report an "
                            "unknown source hash, which bench/rules.py reads as "
                            "'nobody looked' rather than as clean.")
        endif()
    else()
        message(WARNING "roshambo3: the source stamp generator failed "
                        "(rc=${_r3_stamp_rc}); this binary will report an "
                        "unknown source hash and an unchecked dirty marker.")
    endif()
else()
    message(WARNING "roshambo3: no python3 found for ${_r3_stamp_script}; this "
                    "binary will report an unknown source hash and an "
                    "unchecked dirty marker.")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _r3_bt)
set(_r3_cxx_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_r3_bt}}")

# -lineinfo keeps source correlation for ncu/nsys without disabling
# optimization; --ptxas-options=-v reports per-kernel register and shared-memory
# use on every build, so an occupancy regression is visible immediately.
#
# --use_fast_math is deliberately ABSENT and must stay absent: it was measured
# at up to 1.5e-2 |dT| with 1.30% optimizer basin flips. See
# doc/NUMERICS_POLICY.md; roshambo3/numerics.py enforces this at import.
set(_r3_extra_cuda_flags "-lineinfo --ptxas-options=-v")

# CUDA 12.x warns once per translation unit for every target below compute_75
# ("Support for offline compilation for architectures prior to ... will be
# removed"). We know: that is exactly why the architecture list is discovered
# rather than fixed. Silence it only when such a target is actually present, so
# the warning still appears if someone adds a pre-Turing target by hand to a
# toolkit that is about to drop it.
set(_r3_pre_turing OFF)
foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "^([0-9]+)" _n "${_a}")
    if(_n AND _n LESS 75)
        set(_r3_pre_turing ON)
    endif()
endforeach()
if(_r3_pre_turing)
    string(APPEND _r3_extra_cuda_flags " -Wno-deprecated-gpu-targets")
endif()

# ---------------------------------------------------------------------------
# The CPU shape kernel.
#
# Built UNCONDITIONALLY. CXX is already a project() language, so this costs a
# CUDA-less install nothing it was not already paying -- and building it always
# is what puts the CPU kernel under the CI `cpu` job, which asserts
# `not with_cuda` and runs the GPU-free suite. That job is now the CPU kernel's
# standing regression gate, and it is coverage the CUDA kernel does not have.
#
# Deliberately NOT gated behind a ROSHAMBO3_CPU option: an optional kernel is
# one that silently is not there, and `backend.select("cpu")` would then fail
# on a build that looked fine. If it must become optional later, the option has
# to be recorded in the build stamp, which says what the binary CONTAINS, not
# what the caller typed.
# ---------------------------------------------------------------------------
set(_r3_cpu_src "${CMAKE_CURRENT_SOURCE_DIR}/src/roshambo3/backend/_cpu_src")
# pybind11's own Python discovery, through CMake's FindPython rather than the
# FindPythonInterp/FindPythonLibs modules CMake has removed (policy CMP0148).
# Still one discovery, and it honors the Python_EXECUTABLE hint
# scikit-build-core passes, so it selects the interpreter the caller asked for
# (see the note on the source-stamp interpreter above). Applies to every find_package(pybind11) below.
if(NOT DEFINED PYBIND11_FINDPYTHON)
    set(PYBIND11_FINDPYTHON ON)
endif()
if(EXISTS "${_r3_cpu_src}/interface.cpp")
    find_package(pybind11 REQUIRED)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(_r3_cpu_aarch64 TRUE)
        set(_r3_cpu_isa_source "${_r3_cpu_src}/grain_unavailable.cpp")
        set(_r3_cpu_default_arch "-march=armv8-a+simd")
        set(_r3_color_canonical_source "${_r3_cpu_src}/color_canonical_neon.cpp")
        set(_r3_color_canonical_four_start_source
            "${_r3_cpu_src}/color_canonical_four_start_neon.cpp"
            "${_r3_cpu_src}/color_canonical_two_stream_neon.cpp")
    else()
        set(_r3_cpu_isa_source "${_r3_cpu_src}/grain_avx512.cpp")
        set(_r3_cpu_default_arch "-march=x86-64-v3")
        set(_r3_color_canonical_source "${_r3_cpu_src}/color_canonical_kernel.cpp")
        set(_r3_color_canonical_four_start_source
            "${_r3_cpu_src}/color_canonical_four_start_xmm.cpp"
            "${_r3_cpu_src}/color_canonical_pair_ymm.cpp")
    endif()
    # Several sources share one extension: interface.cpp and the CPU analytic
    # color/field units use -march=x86-64-v3, while the AVX-512 shape kernel is
    # in its own translation unit with its own flag (set below). Runtime guards
    # keep AVX-512 arms unavailable on hosts without it while preserving the
    # portable v3 target.
    #
    # interface.cpp FIRST IS LOAD-BEARING, not alphabetical or
    # incidental. Both TUs include a few of the same headers (kernel.hpp,
    # score.hpp, grain.hpp), and every one of THIS project's own
    # shared definitions is `static`/anonymous-namespace so the linker can no
    # longer choose between an x86-64-v3 and an -mavx512f compile of them --
    # but a definition this project does not own can still be genuinely
    # shared, e.g. libstdc++'s `std::vector<float>::_M_fill_assign`, ODR-used
    # by both TUs via `SoaConformer`/`LaneBlock`. For those, implicit LTO's
    # COMDAT resolution still falls back to LINK ORDER -- first object listed
    # wins (proved causally by relinking with the order reversed and watching a zmm/vpermt2ps body
    # replace a scalar one). Listing `interface.cpp` first is what makes that
    # fallback resolve to the x86-64-v3 compile rather than the -mavx512f
    # one. `tests/test_cpu_isa_definitions_are_not_shared.py` disassembles
    # the actual linked module and would catch this order ever stopping
    # being enough, but it cannot repair a reordering -- only this comment,
    # and whoever reads it before reordering these two lines, can.
    set(_r3_cpu_sources
        "${_r3_cpu_src}/interface.cpp"
        "${_r3_cpu_isa_source}"
        "${_r3_cpu_src}/combo_executor.cpp"
        "${_r3_cpu_src}/cpu_platform.cpp"
        "${_r3_cpu_src}/cpu_platform_bindings.cpp"
        "${_r3_cpu_src}/cpu_block_prepare.cpp"
        "${_r3_cpu_src}/color_scalar.cpp"
        "${_r3_cpu_src}/worker_local_topk.cpp"
        "${_r3_cpu_src}/worker_local_topk_bindings.cpp"
        "${_r3_cpu_src}/color_conformer_kernel.cpp"
        "${_r3_color_canonical_source}"
        "${_r3_cpu_src}/color_canonical_four_start.cpp"
        "${_r3_color_canonical_four_start_source}"
        "${_r3_cpu_src}/color_exact_final.cpp"
        "${_r3_cpu_src}/final_rescore.cpp"
        "${_r3_cpu_src}/final_rescore_bindings.cpp"
        "${_r3_cpu_src}/color_invariant_plan.cpp"
        "${_r3_cpu_src}/color_field.cpp"
        "${_r3_cpu_src}/shape_evaluator.cpp"
        "${_r3_cpu_src}/shape_analytic.cpp"
        "${_r3_cpu_src}/shape_field_evaluator.cpp"
        "${_r3_cpu_src}/shape_combo_api.cpp"
        "${_r3_cpu_src}/reduced_pose.cpp"
        "${_r3_cpu_src}/shared_pose_api.cpp"
        # Format-3 coordinate codec (doc/FORMAT.md). Its ISA unpack
        # kernels are added per architecture just below.
        "${_r3_cpu_src}/r3g_codec.cpp"
        "${_r3_cpu_src}/r3g_unpack_scalar.cpp"
        "${_r3_cpu_src}/r3g_build_scalar.cpp"
        "${_r3_cpu_src}/r3g_encode.cpp"
        "${_r3_cpu_src}/r3g_io.cpp"
        "${_r3_cpu_src}/r3g_codec_bindings.cpp")
    if(_r3_cpu_aarch64)
        list(APPEND _r3_cpu_sources "${_r3_cpu_src}/r3g_unpack_neon.cpp" "${_r3_cpu_src}/r3g_build_neon.cpp")
    else()
        list(APPEND _r3_cpu_sources "${_r3_cpu_src}/r3g_unpack_avx2.cpp" "${_r3_cpu_src}/r3g_build_avx2.cpp")
    endif()
    if(_r3_cpu_aarch64)
        list(APPEND _r3_cpu_sources "${_r3_cpu_src}/final_rescore_neon.cpp")
        list(APPEND _r3_cpu_sources "${_r3_cpu_src}/field_neon.cpp"
             "${_r3_cpu_src}/field_neon_fp16.cpp"
             "${_r3_cpu_src}/field_neon_fp16fml.cpp")
    endif()
    if(NOT _r3_cpu_aarch64)
        list(APPEND _r3_cpu_sources "${_r3_cpu_src}/final_rescore_avx2.cpp")
    endif()
    pybind11_add_module(_roshambo3_cpu ${_r3_cpu_sources})

    # ONLY this file. The anchor in score.hpp must keep the codegen its
    # fixture was generated against, so it stays at the target set below and
    # nothing here may widen it.
    if(NOT _r3_cpu_aarch64)
        set(_r3_avx512_opts "-mavx512f")
        set_source_files_properties("${_r3_cpu_src}/grain_avx512.cpp"
            PROPERTIES COMPILE_OPTIONS "${_r3_avx512_opts}")
    else()
        # FP16 instructions stay isolated in one dispatch-protected TU; the
        # rest of the extension retains the portable armv8-a+simd target.
        set(_r3_neon_fp16_opts "-march=armv8.2-a+fp16")
        set_source_files_properties("${_r3_cpu_src}/field_neon_fp16.cpp"
            PROPERTIES COMPILE_OPTIONS "${_r3_neon_fp16_opts}")
        set_source_files_properties("${_r3_cpu_src}/field_neon_fp16fml.cpp"
            PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16fml")
    endif()
    target_include_directories(_roshambo3_cpu PRIVATE
        "${_r3_cpu_src}" "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_features(_roshambo3_cpu PRIVATE cxx_std_17)
    # Named once, used twice: as the flags AND as the stamp. `_r3_cxx_flags`
    # above is CMAKE_CXX_FLAGS plus the build-type flags -- the GLOBAL set --
    # and a target's own `target_compile_options` are not in it. So a stamp
    # built from `_r3_cxx_flags` alone would report " -O3 -DNDEBUG" for a binary
    # compiled with -march=x86-64-v3, omitting the one flag that changed its
    # arithmetic. The stamp must say what the binary CONTAINS, and the record
    # must say which contraction happened, and contraction is determined by these
    # flags.
    # The -march/-mtune half is a cache variable so a build can be re-taken at a
    # different microarchitecture target WITHOUT editing this file -- the
    # experiment then leaves no diff to forget to revert. The DEFAULT is
    # unchanged and is the shipping target: a NAMED portable feature level, not
    # a host-detected one (see the paragraphs below).
    #
    # It does not loosen the arithmetic. -fno-fast-math and -fno-associative-math
    # are appended here, outside the variable, so no value of it can turn them
    # off. What a different -march CAN change is which multiply-add pairs the
    # compiler chooses to contract, and that is a declared numerics change --
    # so a build at a non-default value owes a score comparison, not just a
    # timing.
    set(ROSHAMBO3_CPU_ARCH_FLAGS "${_r3_cpu_default_arch}" CACHE STRING
        "-march/-mtune flags for the CPU kernel translation unit.")
    separate_arguments(_r3_cpu_arch NATIVE_COMMAND "${ROSHAMBO3_CPU_ARCH_FLAGS}")
    set(_r3_cpu_opts ${_r3_cpu_arch} -fno-fast-math -fno-associative-math)
    if(ROSHAMBO3_SANITIZE)
        list(APPEND _r3_cpu_opts "${_r3_cpu_sanitize_flag}" -fno-omit-frame-pointer)
    endif()
    string(REPLACE ";" " " _r3_cpu_opts_str "${_r3_cpu_opts}")
    # AFTER the REPLACE above, which assigns rather than appends and would
    # otherwise discard this. Placed before the target_compile_definitions that
    # bakes the string in -- the same ordering trap that made the stamp report
    # " -O3 -DNDEBUG" for a -march=x86-64-v3 binary earlier in this ladder.
    #
    # In the stamp as well as on the command line: contraction is settled
    # by RECORDING the compiler, its version and its flags rather than pinning
    # them, so a per-source flag appearing in no stamp field is a flag the
    # record cannot account for.
    if(NOT _r3_cpu_aarch64)
        string(APPEND _r3_cpu_opts_str " [grain_avx512.cpp: ${_r3_avx512_opts}]")
    else()
        string(APPEND _r3_cpu_opts_str
            " [field_neon_fp16.cpp: ${_r3_neon_fp16_opts}] [field_neon_fp16fml.cpp: -march=armv8.2-a+fp16fml]")
    endif()
    target_compile_options(_roshambo3_cpu PRIVATE ${_r3_cpu_opts})
    if(ROSHAMBO3_SANITIZE)
        target_link_options(_roshambo3_cpu PRIVATE "${_r3_cpu_sanitize_flag}")
    endif()

    # The generated CPU source stamp lives in the build tree; interface.cpp
    # picks it up with __has_include, so a hand-rolled compile without CMake
    # still builds and simply reports "unknown".
    target_include_directories(_roshambo3_cpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    if(_r3_stamp_ok)
        # Regenerated at BUILD time for the same reason the CUDA one is:
        # `cmake --build` after an edit does not reconfigure, so a
        # configure-then-edit-then-build cycle would stamp the pre-edit source
        # and call it clean. For an ANCHOR that is not a bookkeeping nicety --
        # it is the difference between a fixture that names its generator and
        # one that names a generator that no longer exists.
        add_custom_target(roshambo3_cpu_source_stamp ALL
            COMMAND "${_r3_stamp_python}" "${_r3_stamp_script}"
                    --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                    --subtree "src/roshambo3/backend/_cpu_src" --prefix cpu
                    --phase build --header "${_r3_cpu_stamp_header}"
            BYPRODUCTS "${_r3_cpu_stamp_header}"
            COMMENT "roshambo3: hashing the CPU anchor source about to be compiled"
            VERBATIM)
        add_dependencies(_roshambo3_cpu roshambo3_cpu_source_stamp)
    endif()

    target_compile_definitions(_roshambo3_cpu PRIVATE
        "ROSHAMBO3_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
        "ROSHAMBO3_GIT_SHA=\"${ROSHAMBO3_GIT_SHA}\""
        "ROSHAMBO3_CXX_FLAGS=\"${_r3_cxx_flags} ${_r3_cpu_opts_str}\"")
    # -march=x86-64-v3: AVX2 + FMA + BMI1/2 + F16C, a NAMED portable
    # microarchitecture level rather than `native`, so the binary is the same
    # program on Zen 3, Cascade Lake and any modern x86 cloud instance.
    # `native` would bake one machine's specifics into the kernel frozen as
    # the reference anchor.
    #
    # WHY IT IS WORTH A DECLARED NUMERICS CHANGE, measured: without it this kernel is built for
    # BASELINE x86-64 -- SSE2, no FMA, no SSE4.1 -- and objdump found zero
    # `vfmadd` in the shipped binary. Enabling it is 1.16x. A reference kernel
    # that is needlessly slow inflates every later arm's speedup.
    #
    # (Time it away from compiler runs: a timing loop started right after
    # back-to-back g++ invocations depresses whichever arm runs first.)
    #
    # AND THE WIN IS VECTORIZATION, NOT CONTRACTION. This is the part worth
    # carrying forward: `-march=x86-64-v3 -fno-tree-vectorize` carries 128
    # scalar `vfmadd` and zero packed ops, and runs at 1.003x. So FMA
    # contraction by itself buys 0.3%; the 1.16x is packed arithmetic.
    #
    # WHAT IT COSTS: max |dT| 1.85e-6 against the baseline build -- inside the
    # ~2e-3 budget and an order below the cross-GPU p99 of 2.6e-5. The change
    # is FMA CONTRACTION, which the numerics policy places at the same magnitude
    # as vector width, accepts, and judges on recall.
    #
    # AND WHY `-fno-associative-math` IS WHAT MAKES IT SAFE: it forbids the
    # vectorizer from reassociating the accumulation. Measured -- max |dT| is
    # 1.85e-6 for the vectorized and the non-vectorized build ALIKE, so the
    # vectorizer found data-parallel 3-vector arithmetic (delta, cross product)
    # and not a reassociated reduction, and only the contraction moves anything.
    # That is also what keeps this binary usable as the frozen anchor, which
    # wants source-order accumulation. `-march=x86-64-v3` alone would not be
    # safe for a reference; these two flags together are.
    #
    # Floor: x86-64-v3 requires Haswell (2013) or Excavator. Deliberate.
    install(TARGETS _roshambo3_cpu LIBRARY DESTINATION roshambo3)


else()
    message(STATUS "roshambo3: no CPU kernel sources -- skipping _roshambo3_cpu.")
endif()

if(ROSHAMBO3_WITH_CUDA)
    find_package(pybind11 REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    set(_r3_nvcc_version "${CMAKE_CUDA_COMPILER_VERSION}")
    string(REPLACE ";" "," _r3_arch_str "${CMAKE_CUDA_ARCHITECTURES}")
    set(_r3_cuda_flags "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${_r3_bt}} ${_r3_extra_cuda_flags}")

    # topk.cu and molecule_topk.cu are separate translation units on purpose:
    # they are the only ones that include CUB, whose headers are heavy enough to
    # tax every build if folded into kernels.cu.
    pybind11_add_module(_roshambo3_cuda
        ${_r3_src}/kernels.cu
        ${_r3_src}/molecule_topk.cu
        ${_r3_src}/topk.cu
        ${_r3_src}/dense_rows.cu
        ${_r3_src}/interface.cpp)

    target_link_libraries(_roshambo3_cuda PRIVATE CUDA::cudart)

    # The generated source stamp lives in the build tree; interface.cpp picks it
    # up with __has_include, so a hand-rolled compile without CMake still builds
    # and simply reports "unknown".
    target_include_directories(_roshambo3_cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    if(_r3_stamp_ok)
        add_custom_target(roshambo3_source_stamp ALL
            COMMAND "${_r3_stamp_python}" "${_r3_stamp_script}"
                    --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                    --phase build --header "${_r3_stamp_header}"
            BYPRODUCTS "${_r3_stamp_header}"
            COMMENT "roshambo3: hashing the source about to be compiled"
            VERBATIM)
        add_dependencies(_roshambo3_cuda roshambo3_source_stamp)
    endif()
    separate_arguments(_r3_cuda_opts NATIVE_COMMAND "${_r3_extra_cuda_flags}")
    target_compile_options(_roshambo3_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${_r3_cuda_opts}>)

    target_compile_definitions(_roshambo3_cuda PRIVATE
        "ROSHAMBO3_NTHREADS=${ROSHAMBO3_NTHREADS}"
        "ROSHAMBO3_FINAL_EXACT_SHAPE=$<IF:$<BOOL:${ROSHAMBO3_FINAL_EXACT_SHAPE}>,1,0>"
        "ROSHAMBO3_FINAL_EXACT_COLOR=$<IF:$<BOOL:${ROSHAMBO3_FINAL_EXACT_COLOR}>,1,0>"
        "ROSHAMBO3_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
        "ROSHAMBO3_GIT_SHA=\"${ROSHAMBO3_GIT_SHA}\""
        "ROSHAMBO3_CUDA_ARCHITECTURES=\"${_r3_arch_str}\""
        "ROSHAMBO3_NVCC_VERSION=\"${_r3_nvcc_version}\""
        "ROSHAMBO3_CXX_FLAGS=\"${_r3_cxx_flags}\""
        "ROSHAMBO3_CUDA_FLAGS=\"${_r3_cuda_flags}\""
        "ROSHAMBO3_COMPILER_LAUNCHER_USED=\"${_r3_launcher}\"")

    install(TARGETS _roshambo3_cuda LIBRARY DESTINATION roshambo3)
endif()

# Metal uses runtime shader compilation, so Command Line Tools are sufficient.
set(ROSHAMBO3_METAL "AUTO" CACHE STRING "Build Metal: AUTO, ON, or OFF")
set_property(CACHE ROSHAMBO3_METAL PROPERTY STRINGS AUTO ON OFF)
if(NOT ROSHAMBO3_METAL MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "ROSHAMBO3_METAL must be AUTO, ON, or OFF")
endif()
set(_r3_metal_platform OFF)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    execute_process(COMMAND xcrun --sdk macosx --show-sdk-version
        OUTPUT_VARIABLE _r3_metal_sdk OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_r3_metal_sdk VERSION_GREATER_EQUAL 15.0)
        set(_r3_metal_platform ON)
    endif()
endif()
if(ROSHAMBO3_METAL STREQUAL "ON" AND NOT _r3_metal_platform)
    message(FATAL_ERROR "Metal requires native Apple Silicon macOS with SDK 15 or newer")
endif()
if(_r3_metal_platform AND NOT ROSHAMBO3_METAL STREQUAL "OFF")
    enable_language(OBJCXX)
    find_package(pybind11 REQUIRED)
    find_library(_r3_metal_framework Metal REQUIRED)
    find_library(_r3_foundation_framework Foundation REQUIRED)
    pybind11_add_module(_roshambo3_metal src/roshambo3/backend/_metal_src/runtime.mm)
    target_compile_features(_roshambo3_metal PRIVATE cxx_std_17)
    target_compile_options(_roshambo3_metal PRIVATE -fobjc-arc -fno-fast-math)
    target_compile_definitions(_roshambo3_metal PRIVATE ROSHAMBO3_METAL_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
    target_link_libraries(_roshambo3_metal PRIVATE ${_r3_metal_framework} ${_r3_foundation_framework})
    target_include_directories(_roshambo3_metal PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    if(_r3_stamp_ok)
        add_custom_target(roshambo3_metal_source_stamp ALL
            COMMAND "${_r3_stamp_python}" "${_r3_stamp_script}"
                    --repo "${CMAKE_CURRENT_SOURCE_DIR}"
                    --subtree "src/roshambo3/backend/_metal_src" --prefix metal
                    --phase build --header "${CMAKE_CURRENT_BINARY_DIR}/roshambo3_metal_stamp.h"
            BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/roshambo3_metal_stamp.h"
            VERBATIM)
        add_dependencies(_roshambo3_metal roshambo3_metal_source_stamp)
    endif()
    install(TARGETS _roshambo3_metal LIBRARY DESTINATION roshambo3)
endif()

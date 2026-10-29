# Installation

Use Python 3.11 or later, CMake 3.24 or later, and a C++ compiler. Python
dependencies are declared in `pyproject.toml`. Native CPU implementations include
x86 and AArch64 paths. The first release distributes source archives and an sdist; see `doc/RELEASE_VALIDATION.md`.

On x86-64, the native extension requires `x86-64-v3` support, including AVX2
and FMA. In CPU-family terms, this means Intel Haswell or newer Core/Xeon
CPUs, or AMD Excavator or any Zen generation (including Ryzen and EPYC).
The operating system or virtual machine must expose these instruction-set
features. This requirement also applies to CUDA builds, which include the
native CPU extension.

```bash
python -m pip install .
```

CUDA is optional. Require it explicitly to avoid accidentally testing a CPU build:

```bash
python -m pip install . --config-settings=cmake.define.ROSHAMBO3_CUDA=ON
```

For a CPU build:

```bash
python -m pip install . --config-settings=cmake.define.ROSHAMBO3_CUDA=OFF
```

CUDA architecture selection is discovered from the toolkit. Use
`ROSHAMBO3_ARCH_WISHLIST` or `CMAKE_CUDA_ARCHITECTURES` to override it.
`ROSHAMBO3_ARCH_NATIVE=ON` is for local development, not portable distribution.
Record the compiler, toolkit, architecture, source hash, and flags with measured
results. Read `doc/NUMERICS_POLICY.md` before changing arithmetic or compiler flags.

```bash
python -c 'import roshambo3; print(roshambo3.get_build_info())'
python -m pip install '.[web]'
roshambo3 web --config web/example.toml
```

Minimal Linux hosts also need RDKit drawing shared libraries for the webapp.
On Debian/Ubuntu, install `libxrender1` and `libxext6` (and their dependencies)
if importing `rdkit.Chem.Draw` reports a missing shared library.

Browser assets are included. To rebuild them from source, use Node 22 or later,
then run `npm ci` and `npm run build` in `web/`.

Production build controls are `ROSHAMBO3_CUDA`, `ROSHAMBO3_ARCH_WISHLIST`,
`ROSHAMBO3_ARCH_NATIVE`, `CMAKE_CUDA_ARCHITECTURES`, `ROSHAMBO3_CCACHE`, and
`ROSHAMBO3_COMPILER_LAUNCHER`. The CUDA thread count and exact final scoring
are fixed; a build that sets a nondefault value for either is rejected.

## Apple Silicon

Native Apple Silicon CPU builds use arm64 Python 3.11+ and Apple clang with
`ROSHAMBO3_CUDA=OFF`. See [Apple Silicon support](doc/APPLE_SILICON.md) for
build commands, QoS controls, numerical validation, and benchmark recipes.

Apple Silicon macOS 15+ also supports the experimental Metal GPU backend.
Build with `ROSHAMBO3_CUDA=OFF` and `ROSHAMBO3_METAL=ON` through CMake config
settings; select `--backend metal` explicitly. `ROSHAMBO3_METAL=AUTO` builds
on a compatible native Mac, and `OFF` omits it for a CPU-only package.
See `doc/METAL.md` for requirements, numerical policies and validation limits.

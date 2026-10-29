#!/usr/bin/env python3
"""Generate the self-contained golden scores that pin the arithmetic.

    python bench/make_golden.py            # writes tests/data/golden_scores.npz
    python bench/make_golden.py --check    # regenerate and diff, changing nothing

**Which update rule these bits are.** The committed fixture is the ADDITIVE
Adagrad rule, not the current default. See the note on ``OPT`` below: the rule is
pinned in the call rather than inherited, and the reason for pinning rather than
regenerating is recorded there along with the coverage gap it leaves.

**Why this exists.** The bit-identity gate (``bench/xcheck_roshambo2.py``) is the
strongest numerics check in the project and it is not portable: it needs a
ROSHAMBO2 checkout and a built 200K-conformer development corpus, neither of which exists on CI, on a
new machine, or in a public release. Once ROSHAMBO2 is gone there is nothing
left that can detect an *undeclared* numerics change -- a reassociated sum, a
constant edited in the header, a compiler flag that slipped into the build. The
policy calls that "declared numerics change #N or a bug", and until now nothing
could tell which.

So: input generated from a fixed seed rather than read from a corpus, expected
output committed beside it, and a test that compares **bit for bit**. That is
the same trick ``bench/arch_probe.cu`` uses to let two hosts agree on an input
without transferring a file.

The scores here are not a physical claim and nothing may be cited from them.
They are a fingerprint of this arithmetic on this build, and their only job is
to stop changing.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "tests" / "data" / "golden_scores.npz"

#: Fixed, and never to be changed without regenerating: the seed *is* the input.
SEED = 20260821
N_QUERIES = 4
N_LIBRARY = 512
L = 24
N_REAL_LO, N_REAL_HI = 12, 24

#: The settings the committed fixture was generated at. steps and lr_t are ONE
#: knob, so those two may not be varied independently.
#:
#: ``update_rule`` IS PINNED HERE AND IT IS NO LONGER THE DEFAULT. The committed
#: ``tests/data/golden_scores.npz`` was generated before the exponential
#: map the default, so **its bits are the ADDITIVE rule's** -- the Adagrad step
#: added to ``q[1:4]`` plus a renormalization. Naming it here is what keeps the
#: fixture and the generator describing the same arithmetic; leaving it to the
#: argument default would silently repoint the generator at a different optimizer
#: and turn "the fingerprint changed" into "someone flipped a default", which is
#: the one distinction this file exists to preserve.
#:
#: The choice was to pin rather than regenerate, and the reason is recorded
#: because it is not the obvious one: regenerating writes a new ``arch`` and
#: ``nvcc_version`` into the fixture, so it may only be done on the multi-arch
#: **Release** build the committed fixture names
#: (``75,80,86,89,90,120-real,120-virtual``). Doing it from a ``native`` dev
#: build would not add coverage, it would narrow the gate to one machine.
#:
#: **The gap this leaves, stated rather than left to be discovered.** The
#: committed fingerprint covers the shared machinery -- the field sampling, the
#: gradient, the reductions, the Tier-3 cascade, every derived constant -- under
#: the additive rule, and it does NOT cover ``quat_left_exp`` or
#: ``apply_update_lanes``. Those are covered at bit level without a committed
#: fixture instead, by ``tests/test_exp_map_field_cuda.py``, which asserts that
#: ``UpdateLanes.PER_LANE`` and ``REDUCE_SCATTER`` return bit-identical scores
#: under the exp map -- they share the distributed update and differ only in a
#: reduction proved order-preserving, so an arithmetic change inside either
#: has to break it. (``UpdateLanes.ALL`` is NOT bit-identical to those two under
#: the exp map, for a reason recorded in the enum's own
#: docstring; it is not available as a gate.) Adding a second committed arm here
#: is the better answer and is a one-line change (``ARMS`` below) for whoever next
#: regenerates on a Release build.
OPT = {"steps": 25, "lr_t": 0.4, "lr_q": 0.1, "start_mode": 1}

#: The update rule the committed arms were taken at. A list, so a second arm is
#: an append rather than an edit -- see the note on ``OPT``.
ARMS = ("ADAGRAD_QADD",)


def make_input():
    """Coordinates and atom counts, from the seed alone.

    Geometry deliberately molecule-shaped rather than uniform noise: atoms
    within a few Angstrom of the origin, so the Gaussians actually overlap and
    the optimizer has a basin to find. A cloud of unrelated points would score
    near zero everywhere and pin nothing.
    """
    rng = np.random.default_rng(SEED)
    n_rows = N_QUERIES + N_LIBRARY
    n_real = rng.integers(N_REAL_LO, N_REAL_HI + 1, size=n_rows).astype(np.int32)

    coords = np.zeros((n_rows, L, 4), np.float32)
    for i in range(n_rows):
        k = int(n_real[i])
        # A short random walk with ~1.5 A steps: connected, compact, and not
        # symmetric, which is what makes the pose search do real work.
        step = rng.normal(0.0, 1.0, size=(k, 3))
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        xyz = np.cumsum(step * 1.5, axis=0)
        xyz -= xyz.mean(axis=0)
        coords[i, :k, :3] = xyz.astype(np.float32)
        coords[i, :k, 3] = 1.0          # weight: real atom
    return coords, n_real


def compute(coords, n_real):
    from roshambo3.backend import PoseSearch, UpdateRule, score_pairs

    qc = np.ascontiguousarray(coords[:N_QUERIES])
    qn = np.ascontiguousarray(n_real[:N_QUERIES])
    tc_raw = np.ascontiguousarray(coords[N_QUERIES:])
    tn = np.ascontiguousarray(n_real[N_QUERIES:])

    from roshambo3.layout import tile32
    tc = tile32(tc_raw)

    qt = np.zeros((N_QUERIES, L), np.int32)
    # The library's per-conformer count vector. A shape-only library is
    # one column -- the real atom count -- and there is no type plane at all.
    tcnt = tn.astype(np.uint8).reshape(-1, 1)
    rmat = np.ones((1, 1), np.float32)
    pmat = np.zeros((1, 1), np.float32)
    pq = np.repeat(np.arange(N_QUERIES, dtype=np.int32), N_LIBRARY)
    pt = np.tile(np.arange(N_LIBRARY, dtype=np.int32), N_QUERIES)

    out = {}
    for name, ps in (("field", PoseSearch.FIELD),
                     ("analytic", PoseSearch.ANALYTIC)):
        for arm in ARMS:
            # The fixture key is unsuffixed for the first arm, so the committed
            # file keeps working unchanged; a second arm would be
            # f"{name}_{arm.lower()}".
            key = name if arm == ARMS[0] else f"{name}_{arm.lower()}"
            out[key] = score_pairs(
                query_coords=qc, query_types=qt, query_n_atoms=qn,
                library_coords=tc, library_counts=tcnt,
                pair_query_row=pq, pair_library_row=pt,
                interaction_r=rmat, interaction_p=pmat,
                pose_search=ps, update_rule=UpdateRule[arm], **OPT)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare, writing nothing")
    args = ap.parse_args()

    from roshambo3 import get_build_info, numerics

    bi = get_build_info()
    print(f"build: {bi.build_type} nvcc {bi.nvcc_version} "
          f"arch {bi.cuda_architectures}")
    if not bi.is_release:
        print("REFUSED: a golden fixture must come from a Release build.",
              file=sys.stderr)
        return 2
    chk = numerics.check_build()
    if not chk.ok:
        print(f"REFUSED: build carries forbidden flags: {chk.violations}",
              file=sys.stderr)
        return 2

    coords, n_real = make_input()
    got = compute(coords, n_real)

    # The invariants, on the way past: a fixture that violates them would pin
    # a wrong answer forever.
    for name, v in got.items():
        problems = numerics.check_result_invariants(v)
        if problems:
            print(f"REFUSED: {name} violates result invariants: {problems}",
                  file=sys.stderr)
            return 2
        print(f"{name}: {v.shape} invariants clean")

    if args.check:
        ref = np.load(GOLDEN)
        rc = 0
        for name, v in got.items():
            same = np.array_equal(ref[name], v)
            print(f"{name:<9} -> {'BIT-IDENTICAL' if same else 'DIFFERS'}")
            rc |= 0 if same else 1
        return rc

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GOLDEN, coords=coords, n_real=n_real,
        seed=np.int64(SEED), nvcc_version=str(bi.nvcc_version),
        wire_quant_step=numerics.WIRE_QUANT_STEP,
        arch=str(bi.cuda_architectures),
        # The rule the score arrays were taken at, recorded IN the fixture so a
        # reader cannot mistake which optimizer the committed bits belong to.
        update_rules=np.array(ARMS), **got)
    print(f"wrote {GOLDEN} ({GOLDEN.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

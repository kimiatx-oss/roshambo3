#!/usr/bin/env python3
"""Bit-identity gate: ROSHAMBO3's ported kernels against ROSHAMBO2's own.

The port dropped nine of eleven kernels, deleted two OpenMP multi-GPU splitters,
rewrote the 1,809-line Python boundary down to 912, and renamed roughly a
hundred identifiers. None of that may change a single bit of a score.

This is the check that establishes it, and the reason renaming happened *during*
the port rather than after: a rename applied afterwards would mean re-running
this, and a rename that changes a result is a transcription error that only this
gate can catch.

    # 1. fixture, read once so the two runs cannot differ on their input
    python bench/xcheck_roshambo2.py fixture  <outdir>

    # 2. the reference, from the ROSHAMBO2 checkout's own interpreter
    cd $ROSHAMBO2_CHECKOUT
    .venv/bin/python /path/to/bench/xcheck_roshambo2.py roshambo2 <outdir>

    # 3. the candidate, and the comparison
    python bench/xcheck_roshambo2.py roshambo3 <outdir>
    python bench/xcheck_roshambo2.py compare   <outdir>

Both builds must be Release, the same nvcc, and the same architecture list. A
`native` dev build is NOT comparable -- the arithmetic may be identical, but the
policy has not measured that, and `numerics.check_build()` will say so.

THE COLOR ARM POINTS SOMEWHERE ELSE, ON PURPOSE
-----------------------------------------------
The four steps above compare against **R3f** -- our own fork, through
`optimize_overlap_sparse`, which exists at no upstream
commit. That makes them a *port-fidelity* gate: they prove the rewrite moved no
bits. Valuable, and not evidence about the published tool.

The color arm compares against **U+N** (`33ef8e1`, branch `fix/numerics`) --
ROSHAMBO2 with its quaternion, reflection and combo-gradient defects fixed,
which is the correctness reference. Gating color against plain
upstream would confound the comparison with `420557f`, a defect that is itself
in the color path. It uses upstream's **dense** `optimize_overlap()`, because
upstream has no sparse entry point and `_pairs()` builds the full Cartesian
product anyway, so a dense all-pairs evaluation is the same computation reached
by a different call.

    python bench/xcheck_roshambo2.py color_fixture <outdir>
    cd ~/packages/upstream/roshambo2-worktree-at-33ef8e1
    .venv/bin/python /path/to/bench/xcheck_roshambo2.py upstream_color <outdir>
    python bench/xcheck_roshambo2.py roshambo3_color <outdir>
    python bench/xcheck_roshambo2.py compare_color   <outdir>

It needs **no corpus and no builder**: the fixture is real
RDKit conformers of real drug molecules with real RDKit pharmacophore features,
built from SMILES in this file, so the color kernel is validated before a day is
spent rebuilding corpora for it.

Two configurations, because they answer different questions:

  ``identity``  ``steps=0, start_mode=0``. No optimizer trajectory and no
                start-mode rotation, so the *only* thing being compared is the
                arithmetic of the four volume sums. This is the arm that can
                attribute a difference to a named mechanism instead of to
                accumulated divergence.
  ``shipped``   ``steps=25, start_mode=1``, the production settings.

Two interaction matrices, because the canonical one hides errors: it is
diagonal with one shared prefactor, exactly the configuration in which a
spurious width factor cancels -- which is how the float64 oracle's spurious
ALPHA survived as long as it did. ``off_diagonal`` has off-diagonal widths and
non-unit heights and does not cancel.

Trap: at ``mixing = 0`` the combination Tanimoto is exactly the shape
Tanimoto, so a color arm run at the default measures nothing about color and
passes. ``COLOR_MIXING`` is 0.5.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

#: Resolved lazily, inside the fixture step. The roshambo2 arm of this gate
#: runs under the ROSHAMBO2 checkout's own interpreter, which has no roshambo3
#: on its path -- a module-level import here makes the reference arm
#: unrunnable, which is the one arm that cannot be worked around.
CORPUS_NAME = "D1-DEV-200K"
GROUP = 6
N_TARGETS = 8192
N_QUERIES = 8
#: The settings both arms of this gate run at. steps and lr_t are ONE knob.
#:
#: ``update_rule`` IS PINNED, DELIBERATELY, AND IT IS NO LONGER THE DEFAULT.
#: This gate's whole claim is that the ROSHAMBO3 port moved no bits relative to
#: **R3f**, and R3f computes the ADDITIVE Adagrad step -- torque added to
#: ``q[1:4]``, then a renormalization. The exponential map is now the default
#: (``optimizer.DEFAULT_OPT_MODE``), which is a different optimizer and would
#: return different bits by design. Inheriting the default here would silently
#: turn a bit-identity gate into a comparison of two different algorithms, which
#: fails and tells the reader nothing.
#:
#: So the rule is named in the call rather than left to an argument default. If
#: someone later wants this gate to cover the exp map, that needs a reference
#: that computes the exp map -- which R3f does not -- not a change to this line.
#:
#: The previous value of this dict carried ``"opt_mode": 0`` and **never reached
#: the kernel**: it was a key nothing read, because ``score_pairs`` takes
#: ``update_rule`` and the call below did not pass it. It read as a pin and was
#: not one. That is precisely the failure this comment exists to prevent, so the
#: key is spelled the way the API spells it and is passed explicitly.
OPT = {"steps": 25, "lr_t": 0.4, "lr_q": 0.1}


def make_fixture(out: pathlib.Path) -> None:
    from roshambo3.format import Corpus
    from roshambo3.layout import tile32
    from roshambo3.paths import corpus_path

    c = Corpus(str(corpus_path("format3") / CORPUS_NAME))
    with c.group(GROUP) as g:
        n = min(N_TARGETS, g.n_conf)
        n_real = np.asarray(g.n_real[:n], dtype=np.int32)
        arr = np.zeros((n, g.L, 4), dtype=np.float32)
        cursor = 0
        for bi in range(g.n_blocks):
            if cursor == n:
                break
            xyz = g.decode_block(bi)
            rows = min(xyz.shape[0], n - cursor)
            xyz = xyz[:rows].transpose(0, 2, 1).astype(np.float32)
            arr[cursor:cursor + rows, :g.L_heavy, :3] = xyz * np.float32(c.step)
            for local, atoms in enumerate(n_real[cursor:cursor + rows]):
                arr[cursor + local, :int(atoms), 3] = 1.0
            cursor += rows
        if cursor != n:
            raise ValueError(f"group {GROUP}: decoded {cursor} of {n} rows")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "fixture.npz",
             query_coords=np.ascontiguousarray(arr[:N_QUERIES]),
             query_n_atoms=n_real[:N_QUERIES],
             library_coords=tile32(arr),
             library_n_atoms=n_real,
             L=np.int32(arr.shape[1]), n=np.int32(arr.shape[0]),
             nq=np.int32(N_QUERIES), bucket=np.int32(g.bucket))
    print(f"fixture: {N_QUERIES} queries x {arr.shape[0]} library molecules, "
          f"L={arr.shape[1]}, bucket={g.bucket}")


def _load(out: pathlib.Path):
    f = np.load(out / "fixture.npz")
    return (f["query_coords"], f["query_n_atoms"],
            f["library_coords"], f["library_n_atoms"],
            int(f["L"]), int(f["n"]), int(f["nq"]))


def _pairs(nq: int, n: int):
    return (np.repeat(np.arange(nq, dtype=np.int32), n),
            np.tile(np.arange(n, dtype=np.int32), nq))


def run_roshambo2(out: pathlib.Path) -> None:
    from roshambo2.backends.cuda_backend import optimize_overlap_sparse

    qc, qn, tc, tn, L, n, nq = _load(out)

    class Data:
        """Stand-in for Roshambo2Data; only f_x/f_types/f_n_real are read."""
        def __init__(self, x, nreal):
            self.f_x = np.ascontiguousarray(x, np.float32)
            self.f_types = np.zeros((x.shape[0], x.shape[1]), np.int32)
            self.f_n_real = np.ascontiguousarray(nreal, np.int32)
            self.f_n = self.f_n_real
        def tofloat32(self):
            pass

    pq, pt = _pairs(nq, n)
    arms = (
        ("field", "blockwise_grid_t36",
         dict(grid_h=0.8, grid_pad=8.0, grid_upd=2, grid_carveout=50,
              grid_tier=0, grid_minb=4)),
        # These are ROSHAMBO2's own keyword names, passed into ROSHAMBO2. They
        # cannot be renamed here without breaking the call, which is the one
        # legitimate reason to silence the naming check.
        ("analytic", "blockwise_tiled_t27",  # lint-conventions: allow
         dict(t27_qvec=True, t27_stage_w=256, t27_stage_i16=True,  # lint-conventions: allow
              t27_xhoist=True, t27_now=True, t27_stage_p3=True)),  # lint-conventions: allow
    )
    for name, method, kw in arms:
        v = optimize_overlap_sparse(Data(qc, qn), Data(tc, tn), pq, pt,
                                    start_mode=1, mixing=0.0,
                                    optimizer_settings=dict(OPT),
                                    method=method, n_gpus=1, **kw)
        np.save(out / f"r2_{name}.npy", np.asarray(v, np.float32))
        print(f"roshambo2 {name}: {v.shape} T_shape mean {v[:, 1].mean():.6f}")


def run_roshambo3(out: pathlib.Path) -> None:
    from roshambo3 import get_build_info, scores as S
    from roshambo3.backend import (ColorLoop, ColorPlacement, PoseSearch,
                                   UpdateRule, score_pairs)

    qc, qn, tc, tn, L, n, nq = _load(out)
    bi = get_build_info()
    print("roshambo3 build:", bi.build_type, bi.nvcc_version, bi.cuda_architectures)

    qt = np.zeros((nq, L), np.int32)
    # The library's per-conformer count vector. A shape-only library is
    # one column, the real atom count, and there is no type plane.
    tcnt = np.ascontiguousarray(tn, np.uint8).reshape(-1, 1)
    rmat = np.ones((1, 1), np.float32)
    pmat = np.zeros((1, 1), np.float32)
    pq, pt = _pairs(nq, n)
    for name, ps in (("field", PoseSearch.FIELD), ("analytic", PoseSearch.ANALYTIC)):
        v = score_pairs(query_coords=qc, query_types=qt, query_n_atoms=qn,
                        library_coords=tc, library_counts=tcnt,
                        pair_query_row=pq, pair_library_row=pt,
                        interaction_r=rmat, interaction_p=pmat,
                        pose_search=ps, steps=OPT["steps"], lr_t=OPT["lr_t"],
                        lr_q=OPT["lr_q"], start_mode=1,
                        # The additive rule, explicitly: see OPT above. R3f has
                        # no exponential map, so this gate can only be run
                        # against mode 0.
                        update_rule=UpdateRule.ADAGRAD_QADD)
        np.save(out / f"r3_{name}.npy", v)
        print(f"roshambo3 {name}: {v.shape} "
              f"T_shape mean {v[:, S.TANIMOTO_SHAPE].mean():.6f}")


def compare(out: pathlib.Path) -> int:
    ok = True
    for name in ("field", "analytic"):
        a = np.load(out / f"r2_{name}.npy")
        b = np.load(out / f"r3_{name}.npy")
        same = np.array_equal(a, b)
        diff = a != b
        nd = int(diff.sum())
        print(f"{name:<9} {a.shape}  ->  "
              f"{'BIT-IDENTICAL' if same else f'{nd}/{a.size} words differ'}")
        if not same:
            ok = False
            for c in np.unique(np.argwhere(diff)[:, 1])[:8]:
                d = np.abs(a[:, c] - b[:, c])
                d = d[d > 0]
                print(f"          col {c:2d}: n={d.size} "
                      f"median {np.median(d):.3e} max {d.max():.3e}")
    print("\nGATE:", "PASS -- the port changed no bits" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The color arm, against U+N (33ef8e1). See the module docstring.
# ---------------------------------------------------------------------------

#: Real drug molecules, so the geometry and the feature counts are real rather
#: than sampled (accuracy claims on synthetic random molecules are
#: worthless, and a synthetic "molecule" has no pharmacophores at all). Twenty
#: four of them, embedded once by RDKit with a pinned seed.
COLOR_SMILES = (
    "CC(=O)Oc1ccccc1C(=O)O",                              # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                       # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                          # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                                  # paracetamol
    "CN1CCC[C@H]1c1cccnc1",                                # nicotine
    "OC(=O)c1ccccc1O",                                     # salicylic acid
    "CC(C)NCC(O)COc1ccccc1",                               # propranolol core
    "Clc1ccccc1C1=NCC(=O)Nc2ccc(Cl)cc21",                  # a benzodiazepine
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",                          # procainamide
    "NC(=O)c1ccccc1O",                                     # salicylamide
    "CC1=C(C(=O)Nc2ccccc2)S(=O)(=O)c2ccccc21",             # a sultam amide
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",                        # naproxen
    "Nc1ccc(cc1)S(=O)(=O)N",                               # sulfanilamide
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",         # glucose
    "CC(N)Cc1ccccc1",                                      # amphetamine
    "c1ccc2c(c1)[nH]c1ccccc12",                            # carbazole
    "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",                    # chlorpromazine
    "OC(=O)CCc1ccccc1",                                    # hydrocinnamic acid
    "CC(=O)Nc1ccc(cc1)S(=O)(=O)N",                         # sulfacetamide-like
    "COc1cc2c(cc1OC)C(=O)c1ccccc1C2=O",                    # a dimethoxyquinone
    "NCCc1ccc(O)c(O)c1",                                   # dopamine
    "OC(=O)c1cc(O)c(O)c(O)c1",                             # gallic acid
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",                       # salbutamol
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",                          # theobromine
)
COLOR_N_QUERIES = 6
COLOR_MIXING = 0.5
"""Never 0: at mixing = 0 the combination Tanimoto is exactly the shape
Tanimoto and the arm measures nothing about color."""

#: (name, kernel settings). A ladder, so a difference can be attributed to the
#: step that introduced it rather than to "the two are not the same":
#:
#:   identity  the volume arithmetic alone.
#:   rotated   + the start-mode rotation and the choice among four starts, and
#:             with it the one *structural* difference between the two trees on
#:             this path: roshambo3 precomputes the query's self-overlap once on the
#:             unrotated molecule, where upstream recomputes it per start mode
#:             on the rotated one (doc/NUMERICS_POLICY.md section 4c). Same
#:             quantity mathematically (V_AA is rotation-invariant), not the
#:             same float32.
#:   shipped   + 25 Adagrad steps, the production settings.
COLOR_ARMS = (
    ("identity", {"steps": 0, "start_mode": 0}),
    ("rotated", {"steps": 0, "start_mode": 1}),
    ("shipped", {"steps": OPT["steps"], "start_mode": 1}),
)
COLOR_MATRICES = ("canonical", "off_diagonal")

#: Two more things are pinned in this gate's color arm, for the same reason
#: ``update_rule`` is pinned above: the reference decides, not the default.
#:
#: ``color_placement`` is pinned to IN_GRADIENT because U+N has no other
#: placement. Its ``optimize_overlap`` evaluates the color term inside every
#: gradient call, so its pose is combo-optimal; roshambo3's default is now
#: AT_POSE (mode 2), which converges somewhere else on purpose. Comparing
#: those two would measure the objective, not the arithmetic, and would fail
#: while telling the reader nothing. If someone later wants this gate to cover
#: mode 2, that needs a reference that implements mode 2 -- which U+N does not.
#:
#: ``color_loop`` is pinned PER MATRIX, and the reason is the point of the pin.
#: The same-type structures evaluate only ``ta == tb``, which is exact for the
#: canonical diagonal matrix and WRONG for the off-diagonal one, where a
#: cross-type pair carries real height. So the canonical arm runs the shipped
#: structure -- the gate should exercise what ships -- and the off-diagonal arm
#: runs BOUNDED, the loop bound with every pair still evaluated.
#: Both are bit-identical to the earlier masked loop, so this
#: pins the gate's numbers rather than changing them.
COLOR_LOOP_BY_MATRIX = {"canonical": "SAME_TYPE_NOW", "off_diagonal": "BOUNDED"}


def _color_generators():
    """The canonical generator and a deliberately non-canonical one.

    The class is character-identical in both checkouts (only roshambo3's
    `check_interaction_matrix` report was added), so the *same* code assigns
    features whichever interpreter builds the fixture -- but the fixture is
    built once and saved either way, so neither arm depends on that.
    """
    from roshambo3.pharmacophore import RDKitPharmacophoreGenerator

    canonical = RDKitPharmacophoreGenerator()
    families = list(canonical.FEATURES.keys())
    # Off-diagonal widths and non-unit heights, and a *varying* diagonal: a
    # uniform diagonal would still let one shared prefactor be factored out,
    # which is the cancellation the canonical matrix hides errors behind.
    # Symmetric in (i, j), because the kernel reads r[ta,tb] for one ordering
    # of a feature pair and r[tb,ta] for the other and an asymmetric matrix
    # would make the color overlap itself asymmetric. Every entry is nonzero,
    # so no pair is skipped and the whole matrix is exercised.
    interactions = []
    for i, a in enumerate(families):
        for j, b in enumerate(families):
            if j < i:
                continue
            total, product = i + j, i * j
            r = 0.7 + 0.13 * ((total * 3 + product) % 6)
            p = 0.55 + 0.19 * ((total * 5 + product * 2) % 5)
            interactions.append((a, b, r, p))
            if j != i:
                interactions.append((b, a, r, p))
    off_diagonal = RDKitPharmacophoreGenerator(interactions=interactions)
    return {"canonical": canonical, "off_diagonal": off_diagonal}


def make_color_fixture(out: pathlib.Path) -> None:
    """Real conformers, real features, saved once so both arms read one input.

    The layout is the one both kernels expect of a molecule: ``[0, n_real)`` is
    heavy atoms at weight 1, then the pharmacophore pseudo-atoms at weight 1,
    then padding at weight 0. Everything past ``n_real`` is what the color loop
    ranges over, in both implementations.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    gens = _color_generators()
    canonical = gens["canonical"]

    shapes, feats, ftypes = [], [], []
    kept = []
    for smiles in COLOR_SMILES:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"unparseable SMILES in COLOR_SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"no conformer for {smiles}")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        heavy = Chem.RemoveHs(mol)
        conf = heavy.GetConformer()
        xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(heavy.GetNumAtoms())], np.float64)
        # Centered, as preparation does: the pose search starts from the
        # centroid-aligned molecule and an off-origin fixture would only be
        # testing the translation gradient's reach.
        xyz = xyz - xyz.mean(0)
        cxyz, ctypes = canonical.generate_color_atoms(heavy)
        if cxyz.shape[0] == 0:
            raise ValueError(f"no pharmacophore features on {smiles}")
        cxyz = np.asarray(cxyz, np.float64) - xyz.mean(0)
        shapes.append(xyz)
        feats.append(cxyz)
        ftypes.append(np.asarray(ctypes, np.int32))
        kept.append(smiles)

    n = len(kept)
    widths = [s.shape[0] + f.shape[0] for s, f in zip(shapes, feats)]
    # Padded to a multiple of 8, which is what the producer's bucket widths are.
    L = int(((max(widths) + 7) // 8) * 8)

    coords = np.zeros((n, L, 4), np.float32)
    types = np.zeros((n, L), np.int32)
    n_real = np.zeros(n, np.int32)
    for i, (s, f, t) in enumerate(zip(shapes, feats, ftypes)):
        ns, nf = s.shape[0], f.shape[0]
        coords[i, :ns, :3] = s
        coords[i, :ns, 3] = 1.0
        coords[i, ns:ns + nf, :3] = f
        coords[i, ns:ns + nf, 3] = 1.0
        types[i, ns:ns + nf] = t
        n_real[i] = ns

    # The generator emits color atoms as one contiguous run per type, in
    # **fdef order, which is not FEATURES_ENUM order** (Neg before Pos). This
    # fixture is built by calling the generator, so it carries whatever layout
    # the generator produces by construction -- but that is exactly the kind of
    # thing that is true until it is not, so it is checked and recorded rather
    # than assumed. A hand-built type plane would be gating a layout the
    # generator never produces.
    runs = []
    for t in ftypes:
        seen, prev, ok = [], None, True
        for v in t:
            if v != prev:
                if v in seen:
                    ok = False
                seen.append(v)
                prev = v
        runs.append((ok, tuple(seen)))
    contiguous = all(ok for ok, _ in runs)
    orders = sorted({order for _, order in runs})
    print(f"type runs contiguous per type: {contiguous} "
          f"({len(orders)} distinct type orders observed)")
    if not contiguous:
        raise ValueError(
            "the generator emitted a type twice in one molecule's feature run. "
            "Features were measured to come out as one contiguous run per "
            "type; this fixture would be gating a layout it does not produce")

    out.mkdir(parents=True, exist_ok=True)
    saved = {
        "coords": coords, "types": types, "n_real": n_real,
        "L": np.int32(L), "n": np.int32(n), "nq": np.int32(COLOR_N_QUERIES),
        "n_features": np.int32(len(canonical.FEATURES) + 1),
        "smiles": np.array(kept),
    }
    for name, g in gens.items():
        saved[f"rmat_{name}"] = np.ascontiguousarray(g.interaction_matrix_r,
                                                     np.float32)
        saved[f"pmat_{name}"] = np.ascontiguousarray(g.interaction_matrix_p,
                                                     np.float32)
    saved["type_runs_contiguous"] = np.int32(contiguous)
    saved["observed_type_orders"] = np.array(
        [",".join(str(int(x)) for x in o) for o in orders])
    np.savez(out / "color_fixture.npz", **saved)
    print(f"color fixture: {n} molecules, {COLOR_N_QUERIES} of them queries, "
          f"L={L}, heavy atoms {min(s.shape[0] for s in shapes)}-"
          f"{max(s.shape[0] for s in shapes)}, features "
          f"{min(f.shape[0] for f in feats)}-{max(f.shape[0] for f in feats)}")
    same_type_apart = 0
    for f, t in zip(feats, ftypes):
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                if t[i] == t[j] and np.linalg.norm(f[i] - f[j]) > 1e-3:
                    same_type_apart += 1
    # Same-type features at nonzero separation are the configuration
    # the canonical matrix never produces from a *diagonal* pair, and the one a
    # spurious width factor shows up in.
    print(f"same-type feature pairs at nonzero separation: {same_type_apart}")
    if same_type_apart == 0:
        raise ValueError(
            "this fixture has no same-type feature pair at nonzero separation, "
            "so a spurious width factor in the color exponent would cancel and "
            "the arm would pass without measuring it")


def _load_color(out: pathlib.Path):
    f = np.load(out / "color_fixture.npz", allow_pickle=False)
    return f


def _pose_invariants(v: np.ndarray, tag: str) -> dict:
    """Upstream's own two checks (`33ef8e1`), applied to whichever arm produced
    ``v``: a reported quaternion must be a unit quaternion, and V_AB must obey
    Cauchy-Schwarz against V_AA V_BB. Reported rather than asserted, because a
    violation is a finding about the arm, not a reason to produce no record."""
    q = v[:, 9:13].astype(np.float64)
    qn = np.abs(np.linalg.norm(q, axis=1) - 1.0)
    vab, vaa, vbb = (v[:, 3].astype(np.float64), v[:, 5].astype(np.float64),
                     v[:, 6].astype(np.float64))
    den = np.sqrt(np.maximum(vaa * vbb, 0.0))
    ratio = np.divide(vab, den, out=np.zeros_like(vab), where=den > 0)
    rec = {"max_quat_norm_error": float(qn.max()),
           "max_cauchy_schwarz_ratio": float(ratio.max())}
    print(f"  {tag}: max ||q||-1 = {rec['max_quat_norm_error']:.3e}, "
          f"max V_AB/sqrt(V_AA V_BB) = {rec['max_cauchy_schwarz_ratio']:.6f}")
    return rec


def run_upstream_color(out: pathlib.Path) -> None:
    """The U+N arm: ROSHAMBO2 at 33ef8e1, through the DENSE entry point.

    Run from the upstream checkout, under its own interpreter. Nothing here
    imports roshambo3 -- that is the whole reason the fixture is a saved npz.
    """
    import json
    import subprocess

    from roshambo2.backends.cuda_backend import CudaShapeOverlay

    f = _load_color(out)
    coords, types, n_real = f["coords"], f["types"], f["n_real"]
    nq, n = int(f["nq"]), int(f["n"])

    class Data:
        """Stand-in for Roshambo2Data. Only these five members are read."""

        def __init__(self, x, t, nreal):
            self.f_x = np.ascontiguousarray(x, np.float32)
            self.f_types = np.ascontiguousarray(t, np.int32)
            self.f_n_real = np.ascontiguousarray(nreal, np.int32)
            self.f_n = self.f_n_real
            self.f_names = [f"m{i}" for i in range(x.shape[0])]

        def tofloat32(self):
            pass

    class Gen:
        """Stand-in for a PharmacophoreGenerator: only the two matrices."""

        def __init__(self, r, p):
            self.interaction_matrix_r = np.ascontiguousarray(r, np.float32)
            self.interaction_matrix_p = np.ascontiguousarray(p, np.float32)

    rec = {"arm": "U+N", "entry_point": "CudaShapeOverlay.optimize_overlap "
                                        "(dense)"}
    # Resolved from the imported roshambo2 package's own directory, NOT from the
    # cwd. The runner invokes this from the roshambo3 worktree, so `git
    # rev-parse HEAD` with no `-C` recorded *roshambo3's* HEAD as the reference
    # arm's commit -- a provenance defect against the rule "the commit goes in
    # the run record, not in a comment", and
    # one that reads as plausible until you look it up.
    import roshambo2
    checkout = str(pathlib.Path(roshambo2.__file__).resolve().parents[1])
    rec["checkout"] = checkout
    try:
        rec["commit"] = subprocess.check_output(
            ["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
        rec["tree"] = subprocess.check_output(
            ["git", "-C", checkout, "rev-parse", "HEAD^{tree}"],
            text=True).strip()
        rec["describes"] = subprocess.check_output(
            ["git", "-C", checkout, "log", "-1", "--format=%s"],
            text=True).strip()
        rec["source_delta_vs_base"] = subprocess.check_output(
            ["git", "-C", checkout, "diff", "--name-only", "33ef8e1", "HEAD",
             "--", "roshambo2"], text=True).split()
    except Exception as e:                       # noqa: BLE001
        rec["commit_error"] = str(e)

    query = Data(coords[:nq], types[:nq], n_real[:nq])
    library = Data(coords, types, n_real)
    for matrix in COLOR_MATRICES:
        gen = Gen(f[f"rmat_{matrix}"], f[f"pmat_{matrix}"])
        for arm, kw in COLOR_ARMS:
            settings = {"lr_q": OPT["lr_q"], "lr_t": OPT["lr_t"],
                        "steps": kw["steps"], "legacy_combo_gradient": False}
            overlay = CudaShapeOverlay(query, library,
                                       start_mode=kw["start_mode"],
                                       color_generator=gen,
                                       mixing=COLOR_MIXING, verbosity=0,
                                       n_gpus=1, optimizer_settings=settings)
            v = np.asarray(overlay.optimize_overlap(), np.float32)
            # [nq, n, 20] -> the query-major pair order _pairs() builds.
            v = np.ascontiguousarray(v.reshape(nq * n, 20))
            np.save(out / f"un_color_{matrix}_{arm}.npy", v)
            print(f"U+N {matrix}/{arm}: {v.shape} T_combo mean "
                  f"{v[:, 0].mean():.6f} T_color mean {v[:, 2].mean():.6f} "
                  f"V_color mean {v[:, 4].mean():.6f}")
            rec[f"{matrix}/{arm}"] = _pose_invariants(v, f"{matrix}/{arm}")
            if not np.any(v[:, 4] != 0):
                raise ValueError(
                    f"U+N {matrix}/{arm} produced an identically zero color "
                    f"overlap column, so this arm measures nothing about "
                    f"color -- fix the fixture before comparing anything")
    (out / "un_color_run.json").write_text(json.dumps(rec, indent=2) + "\n")


def run_roshambo3_color(out: pathlib.Path) -> None:
    """The candidate arm, through the public `score_pairs` at its defaults.

    `stage_width` is deliberately not named: resolving it is the library's job,
    and a gate that named it would not be exercising the path a caller takes.
    """
    import json

    from roshambo3 import get_build_info, scores as S
    from roshambo3.backend import (ColorLoop, ColorPlacement, PoseSearch,
                                   UpdateRule, score_pairs)
    from roshambo3.layout import tile32

    f = _load_color(out)
    coords, types, n_real = f["coords"], f["types"], f["n_real"]
    nq, n = int(f["nq"]), int(f["n"])
    bi = get_build_info()
    print("roshambo3 build:", bi.build_type, bi.nvcc_version,
          bi.cuda_architectures)

    # The coordinates go through tile32; the count vector does NOT, and cannot
    # -- it is one 8-byte record per conformer, already coalesced. A type plane
    # left in per-conformer order while the coordinates were tiled (so every
    # color term read back as exactly zero) is not expressible: there is no
    # plane to mis-stride.
    #
    # This fixture puts each molecule's features immediately after its OWN real
    # atoms, and `n_real` varies per molecule, so the color offset is -1 rather
    # than an absolute slot. A corpus block is the other case.
    lib_coords = tile32(np.ascontiguousarray(coords, np.float32))
    from roshambo3.layout import counts_from_types
    lib_counts = counts_from_types(types[:n], n_real[:n],
                                   int(f["rmat_canonical"].shape[0]))
    pq, pt = _pairs(nq, n)

    rec = {"arm": "roshambo3", "build_type": bi.build_type,
           "nvcc": bi.nvcc_version, "architectures": bi.cuda_architectures,
           "git_sha": bi.git_sha, "mixing": COLOR_MIXING}
    for matrix in COLOR_MATRICES:
        for arm, kw in COLOR_ARMS:
            v = score_pairs(
                query_coords=np.ascontiguousarray(coords[:nq], np.float32),
                query_types=np.ascontiguousarray(types[:nq], np.int32),
                query_n_atoms=np.ascontiguousarray(n_real[:nq], np.int32),
                library_coords=lib_coords, library_counts=lib_counts,
                library_color_offset=-1,
                pair_query_row=pq, pair_library_row=pt,
                interaction_r=f[f"rmat_{matrix}"],
                interaction_p=f[f"pmat_{matrix}"],
                pose_search=PoseSearch.ANALYTIC, color=True,
                mixing=COLOR_MIXING, lr_q=OPT["lr_q"], lr_t=OPT["lr_t"],
                steps=kw["steps"], start_mode=kw["start_mode"],
                # The additive rule, for the same reason as the shape arm: U+N
                # is ROSHAMBO2 with its defects fixed and has no exponential
                # map, so the reference this is compared against computes mode 0.
                update_rule=UpdateRule.ADAGRAD_QADD,
                # See COLOR_LOOP_BY_MATRIX above.
                color_loop=ColorLoop[COLOR_LOOP_BY_MATRIX[matrix]],
                color_placement=ColorPlacement.IN_GRADIENT)
            np.save(out / f"r3_color_{matrix}_{arm}.npy", v)
            print(f"roshambo3 {matrix}/{arm}: {v.shape} T_combo mean "
                  f"{v[:, S.TANIMOTO_COMBINATION].mean():.6f} T_color mean "
                  f"{v[:, S.TANIMOTO_COLOR].mean():.6f} V_color mean "
                  f"{v[:, S.OVERLAP_COLOR].mean():.6f}")
            rec[f"{matrix}/{arm}"] = _pose_invariants(v, f"{matrix}/{arm}")
            if not np.any(v[:, S.OVERLAP_COLOR] != 0):
                raise ValueError(
                    f"roshambo3 {matrix}/{arm} produced an identically zero "
                    f"color overlap column -- a mis-strided color layout")
    (out / "r3_color_run.json").write_text(json.dumps(rec, indent=2) + "\n")


def run_color_oracle(out: pathlib.Path) -> None:
    """The float64 oracle at the identity pose, for the ``identity`` cells.

    This is what turns "not bit-identical" into an attributed statement. Both
    arms evaluate the *same* expression in float32, by different routes -- one
    in Angstrom with `exp` and a per-pair prefactor, one in the reduced frame
    with `exp2f` and the prefactor hoisted out of the sum -- so neither is the
    other's reference and their difference is not evidence about either. The
    float64 oracle in `roshambo3.reference.overlap` *is* a reference, and if
    both arms sit the same distance from it, the difference between them is
    float32 rounding of a reassociated sum and nothing else.

    Only the ``identity`` cell is oracled: at ``steps=0, start_mode=0`` the
    pose is the identity, so the oracle needs no optimizer of its own. The
    ``shipped`` cell converges to two different poses and comparing scores at
    different poses would measure the trajectory, not the arithmetic.
    """
    import json

    from roshambo3.reference import overlap as R

    f = _load_color(out)
    coords, types, n_real = f["coords"], f["types"], f["n_real"]
    nq, n, L = int(f["nq"]), int(f["n"]), int(f["L"])

    shape = [coords[i, :n_real[i], :3].astype(np.float64) for i in range(n)]
    feat, ftype = [], []
    for i in range(n):
        sel = np.flatnonzero(types[i] != 0)
        feat.append(coords[i, sel, :3].astype(np.float64))
        ftype.append(types[i][sel].astype(np.int64))

    rec = {}
    for matrix in COLOR_MATRICES:
        rmat, pmat = f[f"rmat_{matrix}"], f[f"pmat_{matrix}"]
        v_self = [R.self_overlap(s) for s in shape]
        c_self = [R.color_overlap(feat[i], ftype[i], feat[i], ftype[i],
                                  rmat, pmat) for i in range(n)]
        rows = np.zeros((nq * n, 20), np.float64)
        k = 0
        for i in range(nq):
            for j in range(n):
                v_ab = R.pair_overlap(shape[i], shape[j])
                c_ab = R.color_overlap(feat[i], ftype[i], feat[j], ftype[j],
                                       rmat, pmat)
                ts = R.tanimoto(v_ab, v_self[i], v_self[j])
                tc = R.tanimoto(c_ab, c_self[i], c_self[j])
                rows[k, 0] = ts * (1 - COLOR_MIXING) + tc * COLOR_MIXING
                rows[k, 1] = ts
                rows[k, 2] = tc
                rows[k, 3] = v_ab
                rows[k, 4] = c_ab
                rows[k, 5] = v_self[i]
                rows[k, 6] = v_self[j]
                rows[k, 7] = c_self[i]
                rows[k, 8] = c_self[j]
                rows[k, 9] = 1.0
                k += 1
        np.save(out / f"oracle_color_{matrix}_identity.npy", rows)
        rec[matrix] = {"n_rows": int(rows.shape[0]),
                       "mean_tanimoto_color": float(rows[:, 2].mean()),
                       "mean_overlap_color": float(rows[:, 4].mean())}
        print(f"oracle {matrix}/identity: {rows.shape} T_color mean "
              f"{rows[:, 2].mean():.6f} V_color mean {rows[:, 4].mean():.6f}")
    (out / "oracle_color_run.json").write_text(json.dumps(rec, indent=2) + "\n")


def _rel(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.divide(d, scale, out=np.zeros_like(d), where=scale > 0)
    return {"median_rel": float(np.median(rel)),
            "p99_rel": float(np.quantile(rel, 0.99)),
            "max_rel": float(rel.max())}


def attribute_color(out: pathlib.Path) -> int:
    """Both float32 arms against the float64 oracle, on the identity cells.

    The claim this establishes, or refutes: the two arms differ from each other
    by about as much as each differs from the oracle, i.e. by float32 rounding
    of the same expression evaluated two ways -- not by an algebraic difference,
    which would put one of them far from the oracle and the other close.
    """
    import json

    rec = {}
    for matrix in COLOR_MATRICES:
        o = np.load(out / f"oracle_color_{matrix}_identity.npy")
        a = np.load(out / f"un_color_{matrix}_identity.npy").astype(np.float64)
        b = np.load(out / f"r3_color_{matrix}_identity.npy").astype(np.float64)
        cell = {}
        print(f"\n{matrix}/identity -- distance from the float64 oracle")
        print(f"  {'column':<24}{'U+N max rel':>14}{'r3 max rel':>14}"
              f"{'arm-to-arm':>14}")
        for col, name in ((0, "tanimoto_combination"), (1, "tanimoto_shape"),
                          (2, "tanimoto_color"), (3, "overlap_volume"),
                          (4, "overlap_color"), (5, "self_overlap_query"),
                          (6, "self_overlap_library"), (7, "self_color_query"),
                          (8, "self_color_library")):
            un = _rel(o[:, col], a[:, col])
            r3 = _rel(o[:, col], b[:, col])
            both = _rel(a[:, col], b[:, col])
            cell[name] = {"upstream_vs_oracle": un, "roshambo3_vs_oracle": r3,
                          "arm_to_arm": both}
            print(f"  {name:<24}{un['max_rel']:>14.3e}{r3['max_rel']:>14.3e}"
                  f"{both['max_rel']:>14.3e}")
        rec[matrix] = cell

    # Does the difference reach the ranked output? Recall, not bit identity, is
    # the acceptance criterion, and doc/NUMERICS_POLICY.md section 8
    # states the cross-architecture result in exactly this shape: scores are not
    # bitwise portable, the ranked output does not change. Same question here.
    #
    # This is an EXACT rank comparison over 24 conformers per query, applying
    # numerics.rank_order -- the project's one tie-break -- identically to both
    # arms. It is deliberately NOT called a recall measurement: N is orders of
    # magnitude below any floor in doc/METRICS.md, and a recall number from 24
    # conformers would be a number-shaped decoration.
    from roshambo3.numerics import rank_order

    f = _load_color(out)
    nq, n = int(f["nq"]), int(f["n"])
    ids = np.arange(n, dtype=np.int64)
    ranks: dict = {}
    for matrix in COLOR_MATRICES:
        for arm, _ in COLOR_ARMS:
            a = np.load(out / f"un_color_{matrix}_{arm}.npy")
            b = np.load(out / f"r3_color_{matrix}_{arm}.npy")
            same_order, same_top1, n_ties = 0, 0, 0
            for q in range(nq):
                sl = slice(q * n, (q + 1) * n)
                ra = rank_order(a[sl, 0], ids)
                rb = rank_order(b[sl, 0], ids)
                same_order += int(np.array_equal(ra, rb))
                same_top1 += int(ra[0] == rb[0])
                n_ties += int(n - np.unique(a[sl, 0]).size)
            ranks[f"{matrix}/{arm}"] = {
                "n_queries": nq, "library_size": n,
                "queries_with_identical_full_order": same_order,
                "queries_with_identical_top1": same_top1,
                "float32_ties_in_reference": n_ties,
            }
            print(f"  rank order {matrix}/{arm:<9} identical for "
                  f"{same_order}/{nq} queries, top-1 for {same_top1}/{nq}")
    rec["rank_agreement"] = ranks

    (out / "attribute_color.json").write_text(json.dumps(rec, indent=2) + "\n")
    return 0


#: Columns the deciding measurement is stated over: the combination
#: Tanimoto and both channel columns.
COLOR_DECIDING_COLUMNS = ((0, "tanimoto_combination"), (1, "tanimoto_shape"),
                          (2, "tanimoto_color"))


def compare_color(out: pathlib.Path) -> int:
    import json

    rec = {"deciding_columns": [c[1] for c in COLOR_DECIDING_COLUMNS],
           "cells": {}}
    all_identical = True
    for matrix in COLOR_MATRICES:
        for arm, _ in COLOR_ARMS:
            a = np.load(out / f"un_color_{matrix}_{arm}.npy")
            b = np.load(out / f"r3_color_{matrix}_{arm}.npy")
            same = np.array_equal(a, b)
            all_identical = all_identical and same
            cell: dict = {"bit_identical": bool(same),
                          "n_rows": int(a.shape[0]),
                          "words_differing": int((a != b).sum())}
            print(f"\n{matrix}/{arm}  {a.shape}  -> "
                  f"{'BIT-IDENTICAL' if same else 'not bit-identical'}")
            cols = {}
            for c in range(a.shape[1]):
                x, y = a[:, c].astype(np.float64), b[:, c].astype(np.float64)
                d = np.abs(x - y)
                scale = np.maximum(np.abs(x), np.abs(y))
                rel = np.divide(d, scale, out=np.zeros_like(d), where=scale > 0)
                cols[c] = {"n_differing": int((d > 0).sum()),
                           "max_abs": float(d.max()),
                           "median_rel": float(np.median(rel)),
                           "p99_rel": float(np.quantile(rel, 0.99)),
                           "max_rel": float(rel.max())}
            cell["columns"] = cols
            for c, name in COLOR_DECIDING_COLUMNS:
                s = cols[c]
                print(f"  col {c:2d} {name:<22} "
                      f"n_diff {s['n_differing']:>6}/{a.shape[0]}  "
                      f"rel median {s['median_rel']:.3e} p99 "
                      f"{s['p99_rel']:.3e} max {s['max_rel']:.3e}")
            worst = max(cols[c]["max_rel"] for c, _ in COLOR_DECIDING_COLUMNS)
            cell["worst_deciding_max_rel"] = worst
            rec["cells"][f"{matrix}/{arm}"] = cell

    rec["bit_identical"] = bool(all_identical)
    (out / "compare_color.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("\nCOLOR GATE vs U+N (33ef8e1):",
          "BIT-IDENTICAL" if all_identical else
          "NOT bit-identical -- see compare_color.json")
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    what, out = sys.argv[1], pathlib.Path(sys.argv[2])
    if what == "fixture":
        make_fixture(out); return 0
    if what == "roshambo2":
        run_roshambo2(out); return 0
    if what == "roshambo3":
        run_roshambo3(out); return 0
    if what == "compare":
        return compare(out)
    if what == "color_fixture":
        make_color_fixture(out); return 0
    if what == "upstream_color":
        run_upstream_color(out); return 0
    if what == "roshambo3_color":
        run_roshambo3_color(out); return 0
    if what == "compare_color":
        return compare_color(out)
    if what == "color_oracle":
        run_color_oracle(out); return 0
    if what == "attribute_color":
        return attribute_color(out)
    print(f"unknown step {what!r}"); return 2


if __name__ == "__main__":
    raise SystemExit(main())

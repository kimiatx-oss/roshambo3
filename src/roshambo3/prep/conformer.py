"""Conformer preparation: the coordinate frame, self-overlap, and stable ids.

Every conformer -- query and corpus alike -- is stored centered on its
heavy-atom centroid and rotated into its heavy-atom principal-axis frame. That
is what makes the int16 coordinate quantization safe (the extent stays small:
measured max |coord| is 15.9 A over PubChem HAC>=45 against the [-128.0, +127.99609375] A that
int16 at the wire-step covers) and it is what the optimizer's start modes assume.

**There is one implementation, here, and both paths use it.** ROSHAMBO2 had two:
``roshambo2/classes.py`` for queries and ``tools/rsb3/prep.py`` for corpus
building, the latter documented as reproducing the former "exactly". It did not
-- see :func:`principal_axis_rotation` -- and two implementations of the frame
that queries and library molecules are compared in is a defect waiting to happen whether
or not it has fired yet.
"""

from __future__ import annotations

import hashlib

import numpy as np

#: Gaussian shape constants. These are the numerics policy's, not free
#: parameters -- see ``roshambo3.numerics`` and ``doc/NUMERICS_POLICY.md``.
KAPPA = 2.41798793102
CARBON_RADIUS = 1.7
GAUSS_ALPHA = KAPPA / (CARBON_RADIUS * CARBON_RADIUS)

#: Note ``np.pi``, in full precision. ROSHAMBO2's corpus builder inherited a
#: truncated ``PI = 3.14159265358`` from its PyTorch reference backend, while
#: ``numerics.py`` -- the policy authority, and the source of the kernel's
#: ``SHAPE_PREFACTOR`` -- used ``math.pi``. So the ``vaa`` sidecar and the
#: kernel's own self-overlap were computed from different constants.
#:
#: Adopting the canonical value here costs nothing and is not a declared
#: numerics change: measured over 4,000 conformers the relative difference is
#: at most 4.7e-12, against a float32 epsilon of 1.2e-7, and **zero** stored
#: float32 sidecar values change. Every built corpus stays valid.
GAUSS_CONST = (np.pi / (2 * GAUSS_ALPHA)) ** 1.5

_EYE3 = np.eye(3, dtype=np.float64)


def make_right_handed(v: np.ndarray) -> np.ndarray:
    """Force ``det(v) == +1``, flipping the least significant axis if needed.

    Neither ``np.linalg.svd`` nor ``np.linalg.eigh`` promises anything about the
    handedness of the basis it returns: any singular or eigen vector may come
    back with either sign, so ``det(v)`` can be -1. Applied as ``r @ v.T`` that
    is a **reflection**, and the molecule is silently replaced by its mirror
    image. The optimizer only ever searches proper rotations, so a mirrored
    conformer can never be superposed back onto a chiral partner: its score is
    the wrong enantiomer's.

    Flipping the least significant principal axis restores a right-handed frame
    while perturbing it as little as possible.
    """
    if np.linalg.det(v) < 0:
        v = v.copy()
        v[-1] = -v[-1]
    # The permanent invariant: no conformer may ever be written
    # under a left-handed basis. This is the single choke point both prep
    # paths go through (module docstring), so asserting the postcondition
    # here -- rather than by re-deriving a rotation from already-prepared
    # coordinates in the corpus builder, which is exactly the near-degenerate,
    # SVD-sign-unstable computation a read-back proxy trips over --
    # is what makes this a real guarantee rather than another proxy.
    # See tests/test_prep.py::TestPrincipalAxisRotation::test_is_always_a_proper_rotation.
    assert np.linalg.det(v) >= -1e-9, (
        f"make_right_handed produced a left-handed basis (det={np.linalg.det(v)!r}); "
        f"this must never happen and indicates a bug in this function itself")
    return v


def principal_axis_rotation(coords: np.ndarray) -> np.ndarray:
    """Rotation whose rows are the principal axes, descending by variance.

    Applied as ``coords @ v.T``. Robust to degenerate and non-finite geometry:
    the normal path is the SVD of the coordinate covariance, falling back to a
    symmetric eigendecomposition and finally to the identity (centering only),
    so one bad molecule can never abort a batch.

    **The right-handedness fix is the part that must not be dropped.**
    ROSHAMBO2's corpus builder (``tools/rsb3/prep.py``) omitted it while its
    frontend (``roshambo2/classes.py``) applied it, so the two could in
    principle place a chiral molecule in mirror-image frames. Measured on 3,000
    raw molecules from ``test/confs.sdf`` the two agree (0 left-handed frames),
    so there is no evidence any built corpus is affected -- but the guarantee
    was absent, and near-degenerate geometry is exactly where an SVD's sign
    convention stops being predictable.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0] if coords.ndim == 2 else 0
    if n < 2:
        # A single atom has no principal axes; identity is correct, not an error.
        return _EYE3
    if not np.all(np.isfinite(coords)):
        return _EYE3

    cvm = np.cov(coords, rowvar=False)
    if np.all(np.isfinite(cvm)):
        try:
            v = np.linalg.svd(cvm)[2]
            if np.all(np.isfinite(v)):
                return make_right_handed(v)
        except np.linalg.LinAlgError:
            pass
        # The covariance is symmetric PSD, so eigh is a more robust equivalent.
        try:
            evals, evecs = np.linalg.eigh(cvm)
            v = evecs[:, np.argsort(evals)[::-1]].T  # descending, as SVD gives
            if np.all(np.isfinite(v)):
                return make_right_handed(v)
        except np.linalg.LinAlgError:
            pass
    return _EYE3


def self_volume(r: np.ndarray) -> float:
    """Gaussian self-overlap ``V_AA`` in cubic Angstrom.

    Precomputed per conformer into the ``vaa`` sidecar, 4 B each, because it is
    the one filter quantity whose whole-corpus sidecar fits in device memory.
    """
    g = r @ r.T
    d = np.diag(g)
    d2 = d[:, None] + d[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)
    return float(8.0 * GAUSS_CONST * np.exp(-GAUSS_ALPHA * d2 * 0.5).sum())


def stored_self_volume(r: np.ndarray) -> float:
    """Compute physical Å³ self-volume from recovered stored coordinates.

    ``r`` is already dequantized from the int16 wire record using the same
    float32 ``the wire-step`` multiplication the reader uses. Direct coordinate
    differences avoid the
    cancellation in the norm/dot-product identity used by the historical
    prepared-frame producer.
    """
    x = np.asarray(r, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("stored coordinates must have shape [n_atoms, 3]")
    delta = x[:, None, :] - x[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    return float(8.0 * GAUSS_CONST * np.exp(-GAUSS_ALPHA * d2 * 0.5).sum())


def prep_conformer(coords: np.ndarray,
                   align_ref: np.ndarray | None = None) -> np.ndarray:
    """Center and principal-axis align a heavy-atom coordinate block.

    ``align_ref`` supplies the frame when it must come from something other than
    ``coords`` itself -- color pseudo-atoms are placed in the frame of the real
    atoms, never their own, or the two would not be in the same space.
    """
    r = np.asarray(coords, dtype=np.float64)
    ref = r if align_ref is None else np.asarray(align_ref, dtype=np.float64)
    if ref.shape[0] == 0:
        # A frame cannot be derived from no atoms: the centroid is a mean over
        # an empty slice and the inertia tensor is singular, so NumPy would
        # return NaN behind two RuntimeWarnings rather than fail. There is
        # nothing to center or align, and the caller is `_why_reject`'s "no
        # heavy atoms" path, which discards this (reachable because
        # RemoveAllHs strips even H2 to no atoms).
        return r.reshape(0, 3)
    return (r - ref.mean(axis=0)) @ principal_axis_rotation(ref).T


def strip_hydrogens(mol):
    """Remove every hydrogen, isotopes included. **The one definition of what
    "heavy atom" means in this repository.**

    `Chem.RemoveHs` by design retains isotopically labeled hydrogens, so a
    deuterated compound kept its deuteriums -- and because the shape code is
    uniform-radius (the fourth coordinate is a padding mask, not a
    radius), each retained deuterium was then scored as a **carbon**-sized
    Gaussian. Methane with one deuterium leaves 2 atoms under `RemoveHs` and 1
    under `RemoveAllHs`.

    That was never an approximation to be toleranced. Deuterium has protium's
    van der Waals radius, so if protium is stripped for shape purposes -- the
    only purpose this corpus serves -- then so is deuterium.

    **It lives here, alone, because it was duplicated.** The library path and
    the query path each carried their own `RemoveHs` call, so the defect
    existed twice and had to be found twice; worse, fixing one alone would have
    been worse than fixing neither, since a stripped library scored against a
    deuterated query is an asymmetry where both-wrong is at least symmetric.
    Anything that needs heavy atoms calls this. What legitimately differs
    between the two callers -- multiple conformers against one, float64 against
    float32, self-volumes, names, color -- stays with the caller.
    """
    from rdkit import Chem

    return Chem.RemoveAllHs(mol)


def prep_rdkit_mol(mol, remove_hs: bool = True):
    """Yield ``(coords[n, 3] float64, n_real, vaa)`` per conformer of ``mol``.

    Every conformer of a molecule shares a heavy-atom count, which is what lets
    bucketing on ``n_real`` keep a molecule's conformers together -- and that in
    turn is what downstream de-duplication relies on.
    """
    from rdkit import Chem

    if remove_hs:
        mol = strip_hydrogens(mol)
    out = []
    for conf in mol.GetConformers():
        r = prep_conformer(np.asarray(conf.GetPositions(), dtype=np.float64))
        out.append((r, r.shape[0], self_volume(r)))
    return out


def stable_id(text: str) -> int:
    """Deterministic non-negative int64 id for a string-keyed molecule.

    External ids (``compound-example``) and vendor ids
    (``vendor-example``) are strings, but the format's
    ``mol_id`` sidecar is int64. blake2b-64 gives a stable mapping and the
    original string is preserved as ``ext_id`` in ``meta.parquet``, so identity
    stays recoverable. Collision probability at 1e6 ids over 2**63 is ~5e-8.

    Note what this is *not*: a molecule id must never be recovered by splitting
    a conformer name on its last underscore. That convention silently merges
    Enamine ids, which legitimately end in ``_<digits>``.
    """
    return int.from_bytes(
        hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"
    ) & 0x7FFFFFFFFFFFFFFF

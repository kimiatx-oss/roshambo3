"""The staged library-COLOR record: three int16 planes, with the type derived.

The staged color tier holds a fixed number of a conformer's color slots in
shared memory (``color_stage``) and spills the tail to global. Its record used
to be four int16 planes -- x, y, z and the feature TYPE. A change replaced the
library's per-slot type plane with a per-conformer count vector, at which point
the fourth plane stored a quantity the loop can derive: a slot's type is
``r3_slot_type(cntB, u, N_features)``, indexed by the slot's offset ``u``
within the conformer's color run, and every reader of the staged record already
holds ``u``. The record is three planes now.

The shipped combination schedule stages ``color_stage = 48`` slots with SHARED
color residency; the streamed comparison arm is the GLOBAL-residency,
unstaged cell ``score_pairs`` reaches at its own defaults. Both are compiled
in the shipped build at ``stream_min_blocks_per_sm = 2``.

Two properties, and they fail differently:

1. **Staged equals streamed, bit for bit.** The staged tier re-quantizes each
   coordinate through ``r3_quantize`` and the streaming tiers read the library
   record directly, so the two agree exactly only when the library is ON the
   corpus's wire-step grid -- which is what a corpus block is, and what this
   fixture is built to be. A wrong plane stride, a wrong pad, or a type
   read from the wrong slot all move bits here.
2. **The count vector is load-bearing on the staged path too.** Deleting the
   type plane is only safe if the staged loop actually derives the type; a loop
   that ignored it would pass property 1 by reading the same coordinates and
   scoring some other type assignment. So a deliberately shifted count vector
   must CHANGE the staged score, exactly as it must on the unstaged one.
"""

from __future__ import annotations

from roshambo3.numerics import WIRE_QUANT_STEP

import numpy as np
import pytest

from roshambo3 import backend, scores as S
from roshambo3.backend import ColorPlacement, CoordResidency, PoseSearch

pytestmark = pytest.mark.skipif(not backend.available(),
                                reason="no CUDA extension in this build")

#: The instantiated staged width: the shipped combination schedule's.
STAGE_WIDTHS = (48,)

L_SHAPE = 24
L_COLOR = 40
N_ROWS = 64
N_REAL = 14
N_COLOR = 34
N_FEATURES = 7
MIXING = 0.5

CANONICAL_R = np.ones((N_FEATURES, N_FEATURES), np.float32)
CANONICAL_P = np.diag([0.0] + [1.0] * (N_FEATURES - 1)).astype(np.float32)

#: The shipped combination schedule's `stream_min_blocks_per_sm`, named
#: rather than left to `score_pairs` to resolve.
MINB_S = 2


def _on_the_grid(rng, shape):
    """Coordinates on the corpus's wire-step grid, in float32.

    Built from int16 and multiplied out exactly as the corpus's float32 record
    was, so ``r3_quantize`` recovers the int16 and the staged and streamed tiers
    read the same number rather than two roundings of it.
    """
    q = rng.integers(-3000, 3001, size=shape).astype(np.int16)
    return (q.astype(np.float32) * WIRE_QUANT_STEP).astype(np.float32)


def _fixture(seed: int = 20260827):
    """One query and a library of molecules with long color runs.

    Molecule-shaped rather than uniform noise, for the reason
    ``bench/make_golden.py`` gives: a cloud of unrelated points scores near zero
    everywhere and a difference in the scoring path has nothing to show up in.
    """
    from roshambo3.layout import counts_from_types, tile32

    rng = np.random.default_rng(seed)
    width = L_SHAPE + L_COLOR
    lib_c = np.zeros((N_ROWS, width, 4), np.float32)
    lib_t = np.zeros((N_ROWS, width), np.int32)
    q_c = np.zeros((1, N_REAL + N_COLOR, 4), np.float32)
    q_t = np.zeros((1, N_REAL + N_COLOR), np.int32)

    # Ascending in type and contiguous per type, which is the postcondition the
    # count vector's run bounds encode and the same-type loop relies on. Six
    # planes, dominated by one, which is the real distribution's shape.
    feature_types = np.array(sorted([1] * 20 + [2] * 5 + [3] * 4 + [4] * 3
                                    + [5] * 1 + [6] * 1), np.int32)
    assert feature_types.size == N_COLOR

    for r in range(N_ROWS):
        drift = np.float32(0.11) * r
        shape_xyz = _on_the_grid(rng, (N_REAL, 3)) + drift
        color_xyz = _on_the_grid(rng, (N_COLOR, 3)) + drift
        lib_c[r, :N_REAL, :3] = shape_xyz
        lib_c[r, :N_REAL, 3] = 1.0
        lib_c[r, L_SHAPE:L_SHAPE + N_COLOR, :3] = color_xyz
        lib_c[r, L_SHAPE:L_SHAPE + N_COLOR, 3] = 1.0
        lib_t[r, L_SHAPE:L_SHAPE + N_COLOR] = feature_types
        if r == 0:
            q_c[0, :N_REAL] = lib_c[r, :N_REAL]
            q_c[0, N_REAL:N_REAL + N_COLOR, :3] = color_xyz
            q_c[0, N_REAL:N_REAL + N_COLOR, 3] = 1.0
            q_t[0, N_REAL:N_REAL + N_COLOR] = feature_types

    # The drift above is a float32 add, which can leave a coordinate off the
    # grid; re-quantize the whole block so the fixture's premise holds exactly.
    lib_c[:, :, :3] = (np.rint(lib_c[:, :, :3]  / WIRE_QUANT_STEP).astype(np.int16)
                       .astype(np.float32) * WIRE_QUANT_STEP)
    q_c[:, :, :3] = (np.rint(q_c[:, :, :3]  / WIRE_QUANT_STEP).astype(np.int16)
                     .astype(np.float32) * WIRE_QUANT_STEP)

    n_real = np.full(N_ROWS, N_REAL, np.int32)
    counts = counts_from_types(lib_t, n_real, N_FEATURES)
    return dict(query_coords=q_c, query_types=q_t,
                query_n_atoms=np.array([N_REAL], np.int32),
                library_coords=tile32(lib_c), library_counts=counts,
                library_color_offset=L_SHAPE,
                pair_query_row=np.zeros(N_ROWS, np.int32),
                pair_library_row=np.arange(N_ROWS, dtype=np.int32),
                interaction_r=CANONICAL_R, interaction_p=CANONICAL_P)


def _score(fixture, *, color_stage: int, steps: int, start_mode: int = 1):
    residency = (CoordResidency.SHARED if color_stage
                 else CoordResidency.GLOBAL)
    return backend.score_pairs(
        pose_search=PoseSearch.FIELD, color=True, mixing=MIXING,
        steps=steps, start_mode=start_mode,
        color_placement=ColorPlacement.IN_GRADIENT,
        color_residency=residency, color_stage=int(color_stage),
        stream_min_blocks_per_sm=MINB_S, **fixture)


@pytest.fixture(scope="module")
def fixture():
    return _fixture()


class TestTheStagedColorRecord:

    def test_the_color_channel_is_not_zero(self, fixture):
        """The failure mode that reads as success: a color sum that is
        computed, written, and exactly zero. Asserted before anything is
        compared to anything, or every comparison below is between zeros."""
        v = _score(fixture, color_stage=STAGE_WIDTHS[-1], steps=25)
        assert np.count_nonzero(v[:, S.OVERLAP_COLOR]) >= N_ROWS // 2
        assert v[:, S.TANIMOTO_COLOR].max() > 0.0

    @pytest.mark.parametrize("color_stage", STAGE_WIDTHS)
    @pytest.mark.parametrize("steps", (0, 25))
    def test_staged_is_bit_identical_to_streamed(self, fixture, color_stage,
                                                 steps):
        """The record's stride and its type derivation, in one statement.

        ``steps = 0`` evaluates at the identity pose, so a moved bit is the
        color arithmetic itself; ``steps = 25`` runs the shipped trajectory,
        where one wrong slot compounds.
        """
        streamed = _score(fixture, color_stage=0, steps=steps)
        staged = _score(fixture, color_stage=color_stage, steps=steps)
        a = streamed.view(np.uint32)
        b = staged.view(np.uint32)
        moved = int(np.count_nonzero(a != b))
        assert moved == 0, (
            f"color_stage={color_stage}, steps={steps}: {moved} of {a.size} "
            f"score words differ between the staged and the streamed tier")

    @pytest.mark.parametrize("color_stage", STAGE_WIDTHS)
    def test_a_shifted_count_vector_changes_the_staged_score(self, fixture,
                                                             color_stage):
        """The staged loop derives the type rather than ignoring it.

        A staged reader that had stopped consulting the count vector would still
        read the right coordinates and would still pass the comparison above; it
        would score the wrong type assignment, silently. One fewer feature of
        the first type this library carries moves the run boundary by one slot
        and nothing raises -- exactly the unstaged ``shifted`` arm, now on the tier that
        used to carry its own copy of the type.
        """
        good = _score(fixture, color_stage=color_stage, steps=0)
        shifted = dict(fixture)
        counts = np.array(fixture["library_counts"], copy=True)
        t = int(np.flatnonzero(counts[0, 1:] > 0)[0]) + 1
        counts[:, t] -= 1
        shifted["library_counts"] = counts
        bad = _score(shifted, color_stage=color_stage, steps=0)
        assert not np.array_equal(good.view(np.uint32), bad.view(np.uint32)), (
            f"color_stage={color_stage}: a count vector one feature short "
            f"produced the identical score block, so the staged loop is not "
            f"reading it")

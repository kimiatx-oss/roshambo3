"""The consuming path validates relationships, not only shapes.

Earlier hardening covered the paths that *build* things -- the corpus writer, the sidecars,
``QuerySet.from_arrays``. It did not harden the path that consumes them, and
that path had two ways around every check added there: a directly-constructed
``QuerySet`` (a plain dataclass, so ``from_arrays``' checks never ran) and a
direct ``backend.score_pairs`` call.

Every check exercised here guards a read the device performs with no bounds
check of its own, so what these tests prevent is a wrong number or a segfault,
not an exception.
"""
from __future__ import annotations

import numpy as np
import pytest

from roshambo3.backend import cuda as backend

pytestmark = pytest.mark.skipif(not backend.available(),
                                reason="needs the CUDA extension")

L = 8
N_Q = 2
N_LIB = 32


def _arrays(**over):
    n_lib = over.pop("n_lib", N_LIB)
    n_q = over.pop("n_q", N_Q)
    kw = dict(
        query_coords=np.zeros((n_q, L, 4), np.float32),
        query_types=np.zeros((n_q, L), np.int32),
        query_n_atoms=np.full(n_q, L, np.int32),
        library_coords=np.zeros((n_lib, L, 4), np.float32),
        library_counts=np.full((n_lib, 1), L, np.uint8),
        pair_query_row=np.repeat(np.arange(n_q, dtype=np.int32), n_lib),
        pair_library_row=np.tile(np.arange(n_lib, dtype=np.int32), n_q),
        interaction_r=np.ones((1, 1), np.float32),
        interaction_p=np.zeros((1, 1), np.float32),
    )
    kw["query_coords"][..., 3] = 1.0
    kw["library_coords"][..., 3] = 1.0
    kw.update(over)
    return kw


class TestRelationshipsBetweenArguments:
    """Each argument relationship the device relies on, beyond shape."""

    def test_a_short_query_n_atoms_is_refused(self):
        kw = _arrays(query_n_atoms=np.full(N_Q - 1, L, np.int32))
        with pytest.raises(ValueError, match="one count per query row"):
            backend.score_pairs(**kw)

    def test_a_type_plane_that_does_not_match_the_coordinates_is_refused(self):
        kw = _arrays(query_types=np.zeros((N_Q, L + 3), np.int32))
        with pytest.raises(ValueError, match=r"query_types must be \[n_q, L\]"):
            backend.score_pairs(**kw)

    def test_more_count_rows_than_coordinate_rows_is_refused(self):
        kw = _arrays(library_counts=np.full((N_LIB + 8, 1), L, np.uint8))
        with pytest.raises(ValueError, match="read out of bounds"):
            backend.score_pairs(**kw)

    def test_a_non_square_interaction_matrix_is_refused(self):
        kw = _arrays(interaction_r=np.ones((1, 2), np.float32))
        with pytest.raises(ValueError, match="square and the same shape"):
            backend.score_pairs(**kw)

    def test_a_type_id_past_the_matrix_dimension_is_refused(self):
        """The device indexes rmat[ta*N+tb] with no bounds check, so this is an
        out-of-bounds device read rather than a wrong answer."""
        kw = _arrays()
        kw["query_types"] = kw["query_types"].copy()
        kw["query_types"][0, 0] = 5
        with pytest.raises(ValueError, match="outside a 1x1 interaction matrix"):
            backend.score_pairs(**kw)

    def test_a_pair_row_outside_the_query_range_is_refused(self):
        kw = _arrays()
        kw["pair_query_row"] = kw["pair_query_row"].copy()
        kw["pair_query_row"][0] = N_Q + 4
        with pytest.raises(ValueError, match="pair_query_row ranges over"):
            backend.score_pairs(**kw)

    def test_a_pair_row_outside_the_library_range_is_refused(self):
        kw = _arrays()
        kw["pair_library_row"] = kw["pair_library_row"].copy()
        kw["pair_library_row"][3] = N_LIB + 1
        with pytest.raises(ValueError, match="pair_library_row ranges over"):
            backend.score_pairs(**kw)


class TestTheFrozenPairIndexContract:
    """The O(n) range check is skipped only for an array nobody can write to.

    Measured at 1.13% of a saturating launch, which is above
    METRICS section 8's 0.5% kernel-only resolvable delta. It was hoisted to
    `Engine._pair_index`, which validated once per (n_q, n) and froze; now
    the engine launches dense and passes no pair list, so this contract
    now serves direct callers with an explicit (sparse) pair list only.
    """

    def test_a_writeable_pair_index_is_checked(self):
        kw = _arrays()
        bad = kw["pair_library_row"].copy()
        bad[0] = N_LIB + 1
        assert bad.flags.writeable
        kw["pair_library_row"] = bad
        with pytest.raises(ValueError, match="pair_library_row ranges over"):
            backend.score_pairs(**kw)

    def test_freezing_is_the_only_way_to_skip_it(self):
        """Deliberately demonstrated with a BAD array: the point of the
        contract is that freezing asserts something, so a frozen lie gets
        through. If this ever stops being true the contract has changed and
        the comment in score_pairs is stale.

        The launch runs in a child process. Its out-of-range read can
        fault asynchronously and leave the CUDA context poisoned, which in
        this process failed about 200 later tests. The range check runs on
        the host before any launch, so the child reports it either way: it
        prints `ready` before the call, and `check-fired` only if the check
        raised. A child that faults afterwards still printed `ready`.
        """
        import subprocess
        import sys
        child = (
            "import numpy as np\n"
            "from roshambo3.backend import cuda as backend\n"
            "from tests.test_backend_boundary import _arrays\n"
            "kw = _arrays()\n"
            "bad_q = kw['pair_query_row'].copy()\n"
            "bad_t = kw['pair_library_row'].copy()\n"
            f"bad_t[0] = {N_LIB + 1}\n"
            "bad_q.flags.writeable = False\n"
            "bad_t.flags.writeable = False\n"
            "kw['pair_query_row'] = bad_q\n"
            "kw['pair_library_row'] = bad_t\n"
            "print('ready', flush=True)\n"
            "try:\n"
            "    backend.score_pairs(**kw)\n"
            "except ValueError as e:\n"
            "    if 'ranges over' in str(e):\n"
            "        print('check-fired', flush=True)\n"
            "except Exception:\n"
            "    pass\n")
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        out = subprocess.run([sys.executable, "-c", child], cwd=root,
                             capture_output=True, text=True, timeout=300)
        lines = out.stdout.split()
        assert "ready" in lines, (
            f"the child never reached the call: rc={out.returncode}\n"
            f"{out.stderr[-2000:]}")
        assert "check-fired" not in lines, (
            "a frozen pair index must skip the O(n) range check")

    def test_a_converted_array_is_checked_even_if_the_input_was_frozen(self):
        """`ascontiguousarray` returns the same object only when no conversion
        is needed. A frozen int64 array becomes a writeable int32 copy, and the
        copy is checked -- so the default direction is to check."""
        kw = _arrays()
        bad = kw["pair_library_row"].astype(np.int64)
        bad[0] = N_LIB + 1
        bad.flags.writeable = False
        kw["pair_library_row"] = bad
        with pytest.raises(ValueError, match="pair_library_row ranges over"):
            backend.score_pairs(**kw)


class TestTheEngineLaunchesDense:
    def test_the_engine_builds_no_pair_index(self):
        """A dense launch restates the launch shape, so the engine no
        longer builds, caches or passes a pair index. The fakes in
        test_web.py and test_screen_mode_contracts.py check that the run loop
        passes None; this checks the cache is gone rather than merely unused."""
        from roshambo3.engine.engine import Engine
        assert not hasattr(Engine, "_pair_index")
        assert not hasattr(Engine, "_PAIR_IDX_MAX")

    def test_half_a_pair_list_is_refused(self):
        kw = _arrays()
        kw["pair_query_row"] = None
        with pytest.raises(ValueError, match="or neither"):
            backend.score_pairs(**kw)


class TestTheQueryBound:
    """The bound computed, and checked where the query is built.

    A query has no uint8 sidecar, so none of ``prep.build._why_reject``'s
    library limits apply to it -- ``MAX_N_REAL = 255`` would be a wrong bound
    that happens to look principled. What bounds a query is shared memory.
    """

    def test_the_bound_is_a_function_of_the_device_not_a_constant(self):
        """This defect once appeared in the stage-width ladder, where a
        constant fitted to one card cost the other a factor of two."""
        from roshambo3.engine.config import (MEASURED_SMEM_OPTIN,
                                             query_atom_limit)
        v70 = query_atom_limit(MEASURED_SMEM_OPTIN["sm_70"])
        v120 = query_atom_limit(MEASURED_SMEM_OPTIN["sm_120"])
        assert v70 == 14_036 and v120 == 14_621
        assert v120 > v70, "the larger budget must admit the larger query"

    def test_more_color_features_lower_the_bound_slightly(self):
        from roshambo3.engine.config import query_atom_limit
        b = 102_400
        assert query_atom_limit(b, n_features=7) < query_atom_limit(b, n_features=1)

    def test_a_budget_below_the_fixed_terms_admits_no_query(self):
        from roshambo3.engine.config import query_atom_limit
        assert query_atom_limit(8) == 0

    def test_the_bound_matches_the_launcher_expression_it_inverts(self):
        """Byte-exact, not merely conservative. The launcher's check is `>`, so
        a bound that over-estimates refuses a query that fits and one that
        under-estimates crashes -- the same argument
        `analytic_block_shared_bytes` makes for itself."""
        from roshambo3.backend.cuda import analytic_block_shared_bytes
        from roshambo3.engine.config import query_atom_limit
        budget = 102_400
        n = query_atom_limit(budget)
        assert analytic_block_shared_bytes(n, 0, 0) <= budget
        assert analytic_block_shared_bytes(n + 1, 0, 0) > budget

    def test_an_oversized_query_is_refused_where_it_is_built(self):
        """Not at screen time, after a corpus has been opened and a panel
        built."""
        import numpy as np
        from roshambo3.engine.query import from_arrays
        from roshambo3.engine.config import (query_atom_limit,
                                             shared_memory_budget)
        n = query_atom_limit(shared_memory_budget(0)) + 1
        with pytest.raises(ValueError, match="past the .* this device"):
            from_arrays([np.zeros((n, 3), np.float32)], ["huge"])

    def test_a_normal_query_is_nowhere_near_the_bound(self):
        """The finding, pinned so it is not quietly lost: the unstaged bound is
        thousands of atoms and no molecule reaches it. What a large query
        actually costs is stage width -- throughput -- not a refusal."""
        import numpy as np
        from roshambo3.engine.query import from_arrays
        from roshambo3.engine.config import (query_atom_limit,
                                             shared_memory_budget)
        qs = from_arrays([np.zeros((300, 3), np.float32)], ["a-large-molecule"])
        assert qs.L == 300
        assert query_atom_limit(shared_memory_budget(0)) > 20 * qs.L


class TestUnsupportedCellsIsNamedBeforeTheLadderThrows:
    """The `!launched_` throw stays exactly as it is -- it is the
    backstop, and its detail is the point for a developer changing kernel
    settings. This says the same thing in the caller's vocabulary first."""

    def test_a_setting_off_the_shipped_value_is_named(self):
        from roshambo3.engine.config import unsupported_cells
        got = unsupported_cells({"coord_residency": 3}, bucket=32)
        assert len(got) == 1
        assert "bucket 32" in got[0]
        assert "coord_residency=3" in got[0]
        assert "not supported" in got[0]

    def test_every_non_shared_residency_tier_is_named(self):
        from roshambo3.engine.config import unsupported_cells
        for r in (1, 2, 3, 4):
            got = unsupported_cells({"coord_residency": r})
            assert got and f"coord_residency={r}" in got[0], r

    def test_a_shipped_value_is_not_refused(self):
        """An error that fires on the shipped configuration is one people learn
        to route around."""
        from roshambo3.engine.config import unsupported_cells
        shipped = {"coord_residency": 0, "lanes_per_group": 4,
                   "start_groups_per_row": 1, "update_lanes": 2,
                   "color_stage": 0, "color_residency": 1, "interleave": 0}
        assert unsupported_cells(shipped) == []

    def test_a_launch_setting_off_its_shipped_value_is_named(self):
        from roshambo3.engine.config import unsupported_cells
        got = unsupported_cells({"lanes_per_group": 2})
        assert len(got) == 1 and "lanes_per_group=2" in got[0]


class TestHeavyAtomCountIsNotAtomsPlaced:
    """`meta.parquet`'s `hac` and the `n_real` sidecar measure different
    things, and the difference is exactly the hydrogens `RemoveHs` keeps.

    Pinned here because the identity is what made the disagreement benign as
    metadata and load-bearing as geometry: every retained hydrogen became a
    real atom the kernel scored, at the carbon radius, because the shape code is
    uniform-radius.

    **The preparation side is fixed** -- both strip sites now
    use `RemoveAllHs`, so nothing isotopic reaches the coordinate array. The
    RDKit behavior below is unchanged and is still the mechanism; what changed
    is that the repository no longer relies on it. **Corpora already on disk
    were built with the old strip and still carry the phantom atoms** until
    they are rebuilt.
    """

    def test_removehs_keeps_isotopic_hydrogens(self):
        """The mechanism, on a molecule small enough to read. Deuterium is a
        hydrogen to a heavy-atom count and an atom to the preparation."""
        Chem = pytest.importorskip("rdkit.Chem")
        light = Chem.MolFromSmiles("CCO")
        heavy = Chem.MolFromSmiles("[2H]C([2H])([2H])CO")
        assert light.GetNumHeavyAtoms() == heavy.GetNumHeavyAtoms() == 3

        kept_light = Chem.RemoveHs(Chem.AddHs(light))
        kept_heavy = Chem.RemoveHs(Chem.AddHs(heavy))
        n_h = lambda m: sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 1)
        assert n_h(kept_light) == 0
        assert n_h(kept_heavy) == 3, "RemoveHs keeps isotopic hydrogens"
        # And that is exactly the gap: same hac, different atoms placed.
        assert (kept_heavy.GetNumAtoms() - kept_light.GetNumAtoms()) == 3

    def test_the_prepared_conformer_no_longer_places_them(self):
        """It used to, and that was the defect: they reached the coordinate
        array, and the shape code is uniform-radius
        (`numerics.CARBON_RADIUS` for every real atom), so each deuterium was
        scored as a **carbon**-sized Gaussian.

        This assertion is inverted from what it once pinned. Deuterium
        has protium's van der Waals radius, so stripping one and keeping the
        other was the preparation step failing at its only job -- not an
        approximation with a cost to be measured. See
        `tests/test_isotopic_hydrogens_are_stripped.py`."""
        Chem = pytest.importorskip("rdkit.Chem")
        pytest.importorskip("rdkit.Chem.AllChem")
        from rdkit.Chem import AllChem
        from roshambo3.prep.conformer import prep_rdkit_mol

        def n_placed(smiles):
            m = Chem.AddHs(Chem.MolFromSmiles(smiles))
            assert AllChem.EmbedMolecule(m, randomSeed=20260902) == 0
            (_, n_real, _), = prep_rdkit_mol(m)
            return n_real

        light = n_placed("CCO")
        heavy = n_placed("[2H]C([2H])([2H])CO")
        assert light == 3, f"three heavy atoms placed, got {light}"
        assert heavy == 3, (
            "the three deuteriums must not reach the coordinate array -- under "
            f"the uniform carbon radius each is scored as a carbon; got {heavy}")
        assert light == heavy, (
            "and with the isotope-blind strip these two agree, which is the "
            "hac-versus-n_real disagreement resolved at the source for corpora "
            "built from here on")

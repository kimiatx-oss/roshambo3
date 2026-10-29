

class TestColorAtomsComeOutInEnumOrder:
    """The corpus layout indexes per-type runs positionally, so emission order
    has to equal `FEATURES_ENUM` order -- and RDKit's fdef order is not that.

    `BaseFeatures.fdef` declares NegIonizable *before* PosIonizable, while
    `FEATURES_ENUM` has Pos = 3 and Neg = 4. The type labels were always correct
    (assigned by name), but a consumer mapping the k-th contiguous run of slots to
    type k+1 was wrong for every molecule carrying both -- 2.49% of one benchmark corpus.
    """

    def _gen(self):
        from roshambo3.pharmacophore import RDKitPharmacophoreGenerator
        return RDKitPharmacophoreGenerator()

    def test_the_fdef_order_really_does_disagree_with_the_enum(self):
        # Pins the premise. If RDKit ever reorders its fdef this test says so,
        # rather than the sort silently becoming a no-op nobody re-justifies.
        import os
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
        g = self._gen()
        fac = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
        fdef = [f for f in fac.GetFeatureFamilies() if f in g.FEATURES_ENUM]
        enum = sorted(g.FEATURES_ENUM, key=g.FEATURES_ENUM.get)
        assert fdef != enum, "fdef and enum agree; the sort is now untested"
        assert fdef.index("NegIonizable") < fdef.index("PosIonizable")
        assert enum.index("PosIonizable") < enum.index("NegIonizable")

    def test_a_molecule_with_both_ionizable_types_is_emitted_in_enum_order(self):
        import numpy as np
        from rdkit import Chem
        g = self._gen()
        # An amino acid: carboxylate (Neg) and amine (Pos) both present.
        mol = Chem.AddHs(Chem.MolFromSmiles("NCCCCC(N)C(=O)O"))
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol, randomSeed=20260821)
        _, types = g.generate_color_atoms(mol)
        assert (g.FEATURES_ENUM["PosIonizable"] in types
                and g.FEATURES_ENUM["NegIonizable"] in types), \
            "fixture must carry both ionizable types or it tests nothing"
        assert np.all(np.diff(types) >= 0), f"not in enum order: {types.tolist()}"

    def test_every_type_forms_exactly_one_contiguous_run(self):
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
        g = self._gen()
        for smi in ("NCCCCC(N)C(=O)O", "c1ccccc1CCN", "OC(=O)c1ccc(O)cc1",
                    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "NC(=O)c1ccccc1"):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(mol, randomSeed=20260821)
            _, types = g.generate_color_atoms(mol)
            assert np.all(np.diff(types) >= 0), f"{smi}: {types.tolist()}"
            # one run per type == number of runs equals number of distinct types
            runs = 1 + int((np.diff(types) != 0).sum()) if len(types) else 0
            assert runs == len(set(types.tolist())), f"{smi} interleaved"

    def test_the_counts_are_recoverable_positionally(self):
        # What the corpus actually stores: six per-type counts. With emission in
        # enum order, count[k] and run k are the same thing by construction.
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
        g = self._gen()
        mol = Chem.AddHs(Chem.MolFromSmiles("NCCCCC(N)C(=O)O"))
        AllChem.EmbedMolecule(mol, randomSeed=20260821)
        _, types = g.generate_color_atoms(mol)
        counts = np.array([int((types == v).sum())
                           for v in sorted(g.FEATURES_ENUM.values())])
        assert counts.sum() == len(types)
        # prefix sum gives each run's start; the slice must be homogeneous
        start = 0
        for k, n in enumerate(counts, start=1):
            assert set(types[start:start + n].tolist()) <= {k}
            start += n

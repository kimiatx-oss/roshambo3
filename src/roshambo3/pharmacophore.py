# MIT License
# 
# Copyright (c) 2026 Kimia Therapeutics
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import logging
import os
from dataclasses import dataclass

import numpy as np

from rdkit.Chem import AllChem, MolFromSmarts
from rdkit.Chem.rdchem import Mol as rdkitMol
from rdkit import RDConfig

from roshambo3.numerics import check_interaction_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RDKitFeaturePlan:
    """Coordinate-independent feature assignments of one molecule graph.

    ``types[i]`` is feature ``i``'s type id (``FEATURES_ENUM``, 1-based) and
    ``atom_ids[i]`` the atom indices it was matched on, in the molecule the
    plan was made from (at build, the molecule's stored SMILES parsed and
    with hydrogens added). Features are grouped by ascending type and, within a type, in
    the generator's own enumeration order.

    Format 3 records one positioning rule for every generator: a feature's
    position is the **equal-weight** mean of its atoms (``doc/FORMAT.md``
    sections 8.3 and 10.2, ``positioning = "centroid"``), derived per conformer
    from the stored heavy-atom coordinates. A plan therefore carries no
    positions and no weights; a definition that declares unequal weights for a
    selected family is refused at build (``definitions.unequal_weights``).
    """
    atom_ids: tuple[tuple[int, ...], ...]
    types: np.ndarray


#: The shipped color feature order, and therefore the shipped type numbering:
#: ``FEATURES_ENUM[name]`` is this tuple's index plus one (0 is a normal atom).
#:
#: The historical order remains named for research comparisons. New corpora
#: carry their actual type names in the definition fingerprint; the shipped
#: generator order and descriptor agree in tests/test_color_plane_order.py.
HISTORICAL_FEATURE_ORDER = (
    ('Donor', 'rdkit'),
    ('Acceptor', 'rdkit'),
    ('PosIonizable', 'rdkit'),
    ('NegIonizable', 'rdkit'),
    ('Aromatic', 'rdkit'),
    ('Hydrophobe', 'rdkit'),
)

#: **THE SHIPPED ORDER**. The two largest color planes first, so a
#: `CSTAGE`-wide staging prefix covers more of the same-type pair work.
#:
#: Measured over 30.4 M conformers on three corpora: Hydrophobe is the largest
#: plane and Acceptor the second in *all three*, and together they carry 89-94%
#: of the color pair work. Under the historical numbering Hydrophobe sorted
#: LAST, so a fixed staging prefix spilled the highest-work plane first.
#:
#: Worth **1.0226x on sm_70 and 1.0315x on sm_120** at the tie point, above
#: `doc/METRICS.md` section 8's floor on both, and free in retrieval: measured
#: |dT| max 4.77e-7, recall@1000 = 1.0 at both
#: levels, RBO@0.98 min 0.99999992.
#:
#: The tail below the first two is NOT universal -- DUDE-Z prefers Donor over
#: Aromatic -- and was settled as not worth litigating, because it is an
#: argument about the last ~6% of the work.
WORK_DESCENDING_FEATURE_ORDER = (
    ('Hydrophobe', 'rdkit'),
    ('Acceptor', 'rdkit'),
    ('Aromatic', 'rdkit'),
    ('Donor', 'rdkit'),
    ('PosIonizable', 'rdkit'),
    ('NegIonizable', 'rdkit'),
)


def _descriptor_type_order(generator):
    """Return feature family names in their validated descriptor id order."""
    enum = generator.FEATURES_ENUM
    values = list(enum.values())
    n_types = len(enum)
    if (not n_types or any(type(value) is not int for value in values)
            or sorted(values) != list(range(1, n_types + 1))):
        raise ValueError("feature type ids must be exactly 1..n for a descriptor")
    return [family for family, _ in sorted(enum.items(), key=lambda item: item[1])]


class BasePharmacophoreGenerator():
    """Base Class for Roshambo2 PharmacophoreGenerators

       Functional PharmacophoreGenerators classes are subclass of this.

       If you want to make your own please see the user/developer guide.

    """
    def __init__(self):
        self.FEATURES = {}

        # enumerate for better compatibility with backend codes
        # Note count from 1, type 0 are normal atoms
        self.FEATURES_ENUM = { key: i+1 for i,key in enumerate(self.FEATURES.keys())}



    def set_interactions(self, FEATURES_INTERACTIONs):
        """set the interaction matrix.

            Args: 
                FEATURES_INTERACTIONs (List[Tuple]): map of the interactions between pairs of features, 
                    e.g. [ ('feature1', 'feature2', 1.0, 1.0), ... ]. Where the last two values are the gaussian width and height. 
                    Any pair interactions not included are set to zero. 
        
        """

        self.FEATURES_INTERACTIONs = FEATURES_INTERACTIONs
        
        # turn it into a matrix, note zeros for non interactions
        n_types = len(self.FEATURES_ENUM)
        self.interaction_matrix_r = np.ones((n_types + 1, n_types + 1))
        self.interaction_matrix_p = np.zeros((n_types + 1, n_types + 1))

        for interaction in self.FEATURES_INTERACTIONs:
            key1 = interaction[0]
            key2 = interaction[1]
            r = interaction[2]
            p = interaction[3]

            i = self.FEATURES_ENUM[key1]
            j = self.FEATURES_ENUM[key2]

            self.interaction_matrix_r[i,j] = r
            self.interaction_matrix_p[i,j] = p

        # Is this the canonical diagonal matrix (r = p = 1 on the diagonal,
        # p = 0 off it)? In that case every color term shares the single
        # prefactor (pi/2)^1.5, which cancels identically out of the color
        # Tanimoto, and the color channel is one well-conditioned sum. The
        # kernels handle a non-canonical matrix exactly -- the per-pair residual
        # p^2 r^-1.5 is folded into the accumulated term, not approximated away
        # -- so this is not a correctness guard. It is here so a caller who has
        # changed the matrix is *told*, at query prep, that the color channel
        # is no longer the one every measured number in doc/NUMERICS_POLICY.md
        # was taken on, rather than discovering it from a score that moved.
        self.interaction_matrix_report = check_interaction_matrix(
            self.interaction_matrix_r, self.interaction_matrix_p)
        if not self.interaction_matrix_report.canonical:
            logger.warning(
                "roshambo3: non-canonical color interaction matrix: %s. The "
                "color overlap is still computed exactly, but it no longer "
                "shares the single (pi/2)^1.5 prefactor the default matrix "
                "gives; record this in the run record.",
                "; ".join(self.interaction_matrix_report.reasons))

    def get_feature_indexes(self):
        """Get the mapping from feature name to index.
            
            Returns:
                Dict: Mapping from feature name (str) to index (int) counting starts at 1.
        """

        return self.FEATURES_ENUM
    
    def get_index_to_feature(self):

        return {i:F for F,i in self.FEATURES_ENUM.items()}

    def generate_color_atoms(self, rdkitMol):
        """ takes an RDKit molecule and returns the color atoms

            you will need to have created an assign_features method for you subclass.
        
            return type should be a list of coordinates and list of the color type
            the color type is an integer corresponding to the self.FEATURES_ENUM of the class.

            Args:
                rdkitMol (rdkitMol): An RDKit molecule.

            Returns:
                np.array, dtype=float32: Numpy array of shape [l,3] containing the coordinates of l color dummy atoms.
                np.array, dtype=int: Numpy array of shape [l] containing the color type of each color dummy atom. The indexes must correspond to self.FEATURES_ENUM.
        """
    
        features = self.assign_features(rdkitMol)

        # Emit in FEATURES_ENUM order, grouped, so the k-th contiguous run of
        # slots always carries type k+1. Two reasons, and neither is cosmetic:
        #
        # 1. RDKit's feature factory returns families in the order its .fdef
        #    declares them, which is NOT this class's enum order -- BaseFeatures.fdef
        #    puts NegIonizable before PosIonizable while FEATURES_ENUM has Pos=5,
        #    Neg=6, and declares Donor first while the enum puts it fourth
        #    (the literal was reordered; see the note there). The type *labels*
        #    were always right (assigned by name, below),
        #    but the run order was not, so any consumer mapping the k-th run to
        #    type k+1 positionally was wrong for every molecule carrying both --
        #    2.49% of one corpus, and silently right everywhere else.
        # 2. The color corpus layout requires the runs to be contiguous per type.
        #    Before this sort that held only because the current fdef happens to
        #    group by family; a future RDKit, a caller-supplied `fdefName`, or
        #    CustomSMARTSPharmacophoreGenerator's own dict order could interleave
        #    them. Sorting makes contiguity a postcondition rather than a
        #    property of a data file we do not own.
        #
        # Stable, so within a type the factory's own order is preserved.
        features = sorted(features, key=lambda f: self.FEATURES_ENUM[f[0]])

        l=len(features)
        color_coords = np.zeros((l,3),dtype=np.float32)
        color_type = np.zeros(l, dtype=int)

        for i in range(l):
            feature = features[i]
            color_coords[i,:] = np.array(feature[2], dtype=np.float32)
            color_type[i] = self.FEATURES_ENUM[feature[0]]

        # The postcondition the layout depends on: non-decreasing type, hence one
        # contiguous run per type. Cheap at this size and it converts a silent
        # mislabelling into a loud failure.
        if l > 1 and not np.all(color_type[1:] >= color_type[:-1]):
            raise AssertionError(
                "color types are not grouped in FEATURES_ENUM order after "
                f"sorting: {color_type.tolist()}. The corpus layout indexes "
                "per-type runs positionally and cannot represent interleaving.")

        return color_coords, color_type

    def make_feature_plan(self, mol):
        """The molecule's features as an :class:`RDKitFeaturePlan`, or ``None``
        when this generator cannot state its features as atom sets (a corpus
        cannot then be built with it: format 3 stores only ``(type, atoms)``).
        """
        return None

    def feature_patterns_stereo_free(self) -> bool:
        """True only if this generator POSITIVELY knows its features cannot
        depend on stereochemistry. Opt-in: the default, "not known", is False.

        A caller that wants to treat two copies of one molecule as the same
        graph for feature perception (``SdfSource`` merging a molecule's
        conformer records) must have the generator vouch for it. A generator
        that has not looked, cannot enumerate its patterns, or matches in code
        of its own that might read chirality or bond direction must answer
        False, which keeps every such caller on its strict, exact path.
        """
        return False





class RDKitPharmacophoreGenerator(BasePharmacophoreGenerator):
    """Assigns pharmacophore features to an RDKit molecule using RDKit default fdef files.

    This class is modified from Roshambo provided under MIT license:
    https://github.com/molecularinformatics/roshambo_biogen/tree/Paperless

    """
    def __init__(self, features = None, interactions = None, fdefName = None, *,
                 factory=None, definition=None):
        """
            Args:
                features (Dict, Optional): Feature dictionary, if None the default is used.
                interactions (List, Optional): Interaction map, if None the default is used.
                fdefName (str, Optional): RDKit fdef file to use, if None the default RDKit BaseFeatures.fdef is used.
                factory (Optional): Prebuilt RDKit feature factory. Used when a
                    trusted format-3 descriptor supplies its definition text.
                definition (str, Optional): Definition text associated with
                    ``factory``; avoids reopening a path when describing it.

        """


        self._default_feature_plan = features is None and fdefName is None
        if fdefName is None:
            self.fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
        else:
            self.fdefName = fdefName

        # self.factory = AllChem.BuildFeatureFactory(self.fdefName)
        # wait untill we need it to build it because it cannot be pickled by python
        self.factory = factory
        self._definition_text = definition

        if features is None:
            # THE ORDER OF THIS LITERAL IS THE TYPE NUMBERING, AND IT IS A
            # SHARED-MEMORY STAGING DECISION, NOT A CHEMICAL TAXONOMY.
            #
            # IT IS THE HISTORICAL ORDER, and the reorder the numbers below
            # argue for is NOT shipped here. Every corpus on disk carries this
            # numbering: 44 of the 48 color corpora in the tree are the
            # split layout, which records no numbering at all, and the four
            # that do record one record this. Shipping the reorder refuses all
            # of them, including every corpus earlier measurements rest on.
            # The reorder is measured on a sweep branch against permuted
            # copies of two corpora; whether it lands is that measurement's to
            # decide, and it lands with a corpus migration or not at all.
            #
            # `FEATURES_ENUM` is built by enumerating this dict's insertion
            # order, and everything downstream follows from it: the type ids,
            # the interaction matrices, the per-conformer count vector, the
            # order of the contiguous color-slot runs in the corpus record,
            # and `build_query_color_segments`. So this literal is the only
            # place the order is stated -- there is no permutation table
            # anywhere, deliberately (see below).
            #
            # WHY THIS ORDER. The color kernel stages a fixed prefix of each
            # conformer's color slots into shared memory (`CSTAGE`) and reads
            # the tail from global. Slots are staged in type order, so putting
            # the biggest planes first is what decides how much of the pair
            # work the prefix covers. Measured over 30.4 M conformers on three
            # corpora -- mean color atoms per conformer, and each type's share
            # of same-type pair work (sum of n^2):
            #
            #   corpus                  Don   Acc  PosI  NegI   Aro  Hydrophobe
            #   analog-10M-color       1.65  3.28  0.39  0.10  1.99  11.49 (86.7% of work)
            #   DUDE-Z                 2.22  5.64  0.20  1.49  1.56   8.22 (62.7%)
            #   LIT-PCBA               1.52  3.61  0.27  0.05  2.26  11.07 (84.2%)
            #
            # Hydrophobe is the largest plane and Acceptor the second in all
            # three, and together they carry 94.4 / 89.2 / 93.9 % of the color
            # pair work. Work coverage at a given CSTAGE on
            # the analog-10M color corpus, type-id order against this one:
            #
            #   CSTAGE     12     16     20     24     32     48
            #   old      26.9%  47.1%  67.5%  82.9%  96.3%  99.9%
            #   this     59.1%  79.2%  90.4%  95.7%  99.1% 100.0%
            #
            # ("old" is this literal's order; "this" is the work-descending
            # one.) CSTAGE 24 in the work-descending order covers more pair
            # work than CSTAGE 32 in this one at three quarters of the bytes,
            # and the bytes are the point: shared memory and L1 are one
            # physical SRAM, and the query shape field streams through the L1
            # half. The measured cost of getting that split wrong is
            # 27.85 s against 16.43 s on one 10 M color screen.
            #
            # WHAT THE WORK COVERAGE DOES NOT SAY is whether it turns into
            # throughput, and that is why the reorder is not here. A sweep of
            # CSTAGE at a fixed carveout found 24 and 32 slower than the
            # shipped 48 in both orderings; a follow-up established that it
            # was confounded -- the carveout decides blocks per SM, and its
            # test arms had a 32 KB partition against the tie point's 64 KB --
            # and re-ran the grid with each CSTAGE at its own best carveout.
            #
            # Below Hydrophobe and Acceptor the ordering does drift between
            # corpora -- DUDE-Z prefers Donor over Aromatic -- and that tail is
            # an argument about the last ~6% of the work. It was settled as
            # not worth litigating, and it is fixed
            # here rather than made configurable.
            #
            # AND DO NOT ADD A PERMUTATION LAYER, a reorder pass, or a
            # "logical order for humans, physical order for the kernel" split
            # -- in either direction. A shuffling step that exists in one place
            # gets forgotten in another, and the failure `layout.py` names is
            # silent: donors scored against acceptors. Moving this literal, and
            # re-preparing every corpus with `roshambo3 prepare`, keeps
            # `layout.py`'s ascending-run invariant true by construction, and
            # is the only way the order may ever change.
            #
            # Nothing between RDKit and the record is positional: families are
            # matched by name (`feature.GetFamily()` against `FEATURES.keys()`),
            # ids come from `FEATURES_ENUM[name]`, emission is sorted by that
            # id, and `set_interactions` resolves both ends of every tuple
            # through it. The existing note at `generate_color_atoms` records
            # why that sort has to be there: RDKit's `.fdef` declares families
            # in an order that is not this one, and the positional bug was
            # found the hard way (2.49% of one corpus, silently right
            # everywhere else).
            #
            # A corpus numbered differently from this literal is refused
            # rather than scored -- `Corpus.require_color_features_enum`, from
            # `api._check_color` and from `R3gSource.__init__`. A corpus that
            # records NO numbering is assumed to carry this one, with a warning,
            # because that is what it will be; see that method for why the
            # assumption is safe and what it would take to make it unsafe.
            self.FEATURES = dict(WORK_DESCENDING_FEATURE_ORDER)
        else:
            self.FEATURES = features


        # enumerate for better compatability with backend codes
        # Note count from 1, type 0 are normal atoms
        self.FEATURES_ENUM = { key: i+1 for i,key in enumerate(self.FEATURES.keys())}

        # define the interactions
        # format (type, type, radius, height)
        if interactions == None:
            # By name, not by position: `set_interactions` resolves both ends
            # of every tuple through `FEATURES_ENUM`, so this list's order is
            # cosmetic and the matrix it builds is the same one whatever order
            # it is written in. Kept in the enum's order so a reader comparing
            # the two does not have to check that.
            FEATURES_INTERACTIONs = [
                ('Hydrophobe', 'Hydrophobe', 1.0, 1.0),
                ('Acceptor', 'Acceptor', 1.0, 1.0),
                ('Aromatic', 'Aromatic', 1.0, 1.0),
                ('Donor', 'Donor', 1.0, 1.0),
                ('PosIonizable', 'PosIonizable', 1.0, 1.0),
                ('NegIonizable', 'NegIonizable', 1.0, 1.0),
            ]
        else:
            FEATURES_INTERACTIONs = interactions

        self.set_interactions(FEATURES_INTERACTIONs)


    def descriptor_color(self):
        """Return this generator's color object from FORMAT.md section 10.2."""
        from roshambo3.format.r3g.reference import color_descriptor
        import rdkit

        return color_descriptor(
            generator="rdkit_fdef",
            definition=self.definition_text(),
            types=_descriptor_type_order(self),
            r=self.interaction_matrix_r,
            p=self.interaction_matrix_p,
            rdkit_version=rdkit.__version__,
        )


    def assign_features(self, mol: rdkitMol):
        """Assigns features to an RDKit molecule.
        
            Args:
                mol (rdkitMol)

            Returns:
                List: A list of the features. Will be used by the generate_color_atoms method.
        
        """
        # first time:
        if self.factory is None:
            # print("creating RDKit FeatureFactory")
            self.factory = self._build_factory()
        
        features = self.factory.GetFeaturesForMol(mol)

        pharmacophores = []
        for feature in features:
            fam = feature.GetFamily()
            if fam in self.FEATURES.keys():
                pos = feature.GetPos()
                atom_indices = feature.GetAtomIds()
                p = [
                    fam,
                    atom_indices,
                    [pos[0], pos[1], pos[2]],
                ]
                pharmacophores.append(p)
        return pharmacophores

    #: Characters that can express stereochemistry in a SMARTS string: atom
    #: chirality (``@``, ``@@``, ``@TH1``...) and directional bonds (``/``,
    #: backslash). ``@`` is ALSO SMARTS' ring-bond primitive, so a pattern that
    #: merely uses a ring bond is treated as possibly stereo-dependent too:
    #: that costs speed in a caller that wanted the answer, never correctness.
    #: A text scan of the expanded strings, not of the parsed query, because
    #: stereo inside a recursive ``$(...)`` is not visible on the parsed
    #: top-level atoms.
    _STEREO_SMARTS_CHARACTERS = ("@", "/", "\\")

    def feature_patterns_stereo_free(self) -> bool:
        """True iff the default BaseFeatures plan is in use and no feature
        definition's SMARTS (recursion expanded, as ``GetFeatureDefs`` returns
        them) contains any stereo syntax. Anything else -- a caller-supplied
        fdef or feature set, a factory that will not build or enumerate its
        definitions -- is False (see the base class). The answer is computed
        once per generator, from the installed RDKit's fdef, so an RDKit
        release that adds a chirality or bond-direction pattern flips it.

        RDKit's feature factory also happens to match without chirality today
        (a ``[C@H]`` pattern matches both enantiomers); that is a default this
        project does not control, which is why the patterns themselves are
        checked and not that behavior.
        """
        cached = getattr(self, "_stereo_free_cache", None)
        if cached is not None:
            return cached
        try:
            if not self._default_feature_plan:
                free = False
            else:
                if self.factory is None:
                    self.factory = self._build_factory()
                definitions = self.factory.GetFeatureDefs()
                free = (len(definitions) > 0 and not any(
                    c in str(smarts)
                    for smarts in definitions.values()
                    for c in self._STEREO_SMARTS_CHARACTERS))
        except Exception:
            free = False
        self._stereo_free_cache = free
        return free

    def make_feature_plan(self, mol):
        """The selected families' features of ``mol`` as atom sets, for any
        fdef (the default ``BaseFeatures.fdef`` or a caller's).

        One factory pass over the graph; no coordinates are read, so the plan
        serves every conformer. Types are ``FEATURES_ENUM`` ids, grouped by
        ascending type (stable, so within a type the factory's enumeration
        order is kept, exactly as ``generate_color_atoms`` orders them).
        Whether the definition's weights are representable is the builder's
        check (``definitions.unequal_weights``), not this method's.
        """
        if self.factory is None:
            self.factory = self._build_factory()
        features = []
        for feature in self.factory.GetFeaturesForMol(mol):
            family = feature.GetFamily()
            if family in self.FEATURES_ENUM:
                features.append((self.FEATURES_ENUM[family],
                                 tuple(int(i) for i in feature.GetAtomIds())))
        features.sort(key=lambda item: item[0])
        return RDKitFeaturePlan(tuple(item[1] for item in features),
                                np.asarray([item[0] for item in features],
                                           dtype=np.int64))

    def _build_factory(self):
        """The RDKit factory, built from :meth:`definition_text` -- the text
        a build records -- never by reopening ``fdefName``: a file edited
        while a build runs must not be recorded as one text and perceived
        with another (each worker builds its own factory)."""
        from rdkit.Chem import ChemicalFeatures
        return ChemicalFeatures.BuildFeatureFactoryFromString(self.definition_text())

    def definition_text(self) -> str:
        """The fdef text this generator perceives with (the file's contents,
        or the text a descriptor supplied)."""
        if self._definition_text is None:
            with open(self.fdefName, encoding="utf-8") as definition_file:
                self._definition_text = definition_file.read()
        return self._definition_text




class CustomSMARTSPharmacophoreGenerator(BasePharmacophoreGenerator):
    """Assigns pharmacophore features to an RDKit molecule using custom SMARTS via RDKit.

    This class is modified from Roshambo provided under MIT license:
    https://github.com/molecularinformatics/roshambo_biogen/tree/Paperless

    """
    def __init__(self, features = None, interactions = None, *, types=None):
        """
            Example:
                see /example/customs_features.py
                
            Args:
                features (Dict): Dictionary where the key is the feature name and the value is a SMARTS pattern that defines the feature.
                interactions (List): An interaction map for feature pair interactions. Any pair not specified will have their interactions set to zero.
        """

        # no defaults here, user must supply both features and interactions
        assert(features is not None and interactions is not None)
        self.FEATURES = features
        selected_types = list(self.FEATURES) if types is None else list(types)
        missing_types = [family for family in selected_types if family not in self.FEATURES]
        if missing_types or len(set(selected_types)) != len(selected_types):
            raise ValueError(f"invalid selected feature families: {missing_types or selected_types}")
        self.FEATURES_ENUM = {key: i+1 for i, key in enumerate(selected_types)}

        self.compiled_smarts = {
            k: list(map(MolFromSmarts, self.FEATURES[k])) for k in selected_types
        }

        self.set_interactions(interactions)


    def descriptor_color(self):
        """Return this generator's color object from FORMAT.md section 10.2."""
        from roshambo3.format.r3g.reference import color_descriptor
        import rdkit

        return color_descriptor(
            generator="smarts_list",
            definition=self.FEATURES,
            types=_descriptor_type_order(self),
            r=self.interaction_matrix_r,
            p=self.interaction_matrix_p,
            rdkit_version=rdkit.__version__,
        )



    def _family_matches(self, mol, family):
        """Every match of ``family``'s patterns, each atom tuple once, in
        pattern order then RDKit's match order. Identical tuples are kept once
        (a site matched by two of a family's patterns is one feature); the
        same atoms matched in another order are two. That is the rule
        ``assign_features`` has always applied, and the one the ``deep``
        validator's sandboxed re-perception applies (``format/r3g/_sandbox``),
        so a stored table is reproducible from the stored definition."""
        seen = []
        for pattern in self.compiled_smarts[family]:
            for match in mol.GetSubstructMatches(pattern):
                if match not in seen:
                    seen.append(match)
        return seen

    def make_feature_plan(self, mol):
        """The selected families' matches of ``mol`` as atom sets (see
        :class:`RDKitFeaturePlan`): one ``GetSubstructMatches`` pass per
        pattern, no coordinates read, grouped by ascending type."""
        atom_ids, types = [], []
        for family, type_id in sorted(self.FEATURES_ENUM.items(), key=lambda kv: kv[1]):
            for match in self._family_matches(mol, family):
                atom_ids.append(tuple(int(a) for a in match))
                types.append(type_id)
        return RDKitFeaturePlan(tuple(atom_ids), np.asarray(types, dtype=np.int64))

    def compute_match_centroid(self, mol, matched_pattern):
        conf = mol.GetConformer()
        positions = [conf.GetAtomPosition(i) for i in matched_pattern]
        center = np.mean(positions, axis=0)
        return tuple(center)


    def find_matches(self, mol, patterns):
        matches = []
        for pattern in patterns:
            # Get all matches for that pattern
            matched = mol.GetSubstructMatches(pattern)
            for m in matched:
                # Get the centroid of each matched group
                # centroid = average_match(mol, m)
                centroid = self.compute_match_centroid(mol, m)
                # Add the atom indices and (x, y, z) coordinates to the list of matches
                matches.append([m, centroid])
        return matches


    def assign_features(self, rdkit_mol):
        matches = {}
        for key, value in self.compiled_smarts.items():
            matches[key] = self.find_matches(rdkit_mol, value)

        # Sometimes, a site can match multiple SMARTS representing the same pharmacophore,
        # so we need to keep it only once
        cleaned_matches = {}
        for key, value in matches.items():
            unique_lists = []
            for lst in value:
                if lst not in unique_lists:
                    unique_lists.append(lst)
            cleaned_matches[key] = unique_lists

        pharmacophore = []
        for key, value in cleaned_matches.items():
            for match in value:
                p = [key, match[0], match[1]]
                pharmacophore.append(p)

        return pharmacophore
        


# Default
PharmacophoreGenerator=RDKitPharmacophoreGenerator

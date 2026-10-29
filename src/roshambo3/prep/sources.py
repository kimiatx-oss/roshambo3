"""Where molecules come from when building a corpus.

A source yields ``(ext_id, meta, conformers)`` per molecule, where
``conformers`` is a list of ``(coords[n_real, 3] float64, n_real, vaa)`` already
in the prepared frame. All of a molecule's conformers must be yielded together,
and they necessarily share ``n_real`` -- that is what lets bucketing on
``n_real`` keep a molecule's conformers in one bucket, which de-duplication
downstream relies on.

The SDF convention, stated because it is a real constraint rather than a
detail: **records sharing a ``_Name`` are conformers of one molecule, wherever
they appear in the file.** Grouping is by name across the whole input, not by
adjacency, and that is not a nicety -- real SDFs interleave. In ROSHAMBO2's own
test file 16 of 480 molecules have their conformers split across two or three
separate runs with other molecules in between, and an adjacency rule would have
read each run as its own single-conformer molecule. Nothing detects that; it
just quietly changes what a "molecule-level" result means.

The consequence is that **names must be unique per molecule**. Two genuinely
different molecules sharing a name are now merged rather than kept apart by
happening to be non-adjacent, and the builder raises rather than merging them.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import os
import struct
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

from .conformer import prep_conformer, self_volume, stable_id, strip_hydrogens
from ..format.r3g.limits import LIMITS as _R3G_LIMITS

#: Bytes of a record delimiter line, used when indexing an SDF for parallel reads.
SDF_DELIM = b"$$$$"

#: Most records ``SdfSource`` merges into one multi-conformer Mol.
#: The format-3 run table's per-molecule limit (FORMAT.md
#: section 7), from the format's one statement of it, as
#: ``build.MAX_CONFORMERS_PER_MOLECULE`` is: a molecule with more conformers
#: is rejected at layout whichever path prepared it, so beyond this the merge
#: buys nothing and the records stream through the per-record path instead.
MAX_MERGED_CONFORMERS = _R3G_LIMITS["n_conf_per_molecule"]


class BuildRefused(ValueError):
    """A stop that resuming cannot fix: the same inputs, options and scratch
    would stop the same way again (a resume identity mismatch, a failed id
    check or count reconciliation, an input that violates a layout invariant).
    ``<out>.progress.json`` records it with ``retryable: false``, so a
    supervisor stops rather than looping. Defined here, not in ``build``, so
    the sources can raise it; ``build`` re-exports it. Raised inside scatter
    workers too, so it takes the message as its only argument (an exception
    crosses the process boundary pickled, and pickling replays ``args``)."""
    retryable = False


@dataclass
class MoleculeRecord:
    ext_id: str
    mol_id: int
    meta: dict
    conformers: list
    atom_order: Optional[list] = None
    """The canonical heavy-atom permutation, one ``int64 [n_real]`` array per
    conformer, parallel to ``conformers`` (not inside the conformer tuples,
    whose length the builder reads as "has color"). ``order[slot]`` is the
    index, in the hydrogen-free source record the conformer came from
    (``strip_hydrogens(record)``, source atom order), of the heavy atom stored
    at ``slot``. Per conformer, because records of one id that differ only in
    atom order have different source orders. A consumer holding atom indices
    of a source record (color feature atoms) maps them to slots through
    its inverse. See ``canonical_heavy`` (FORMAT.md section 4.2)."""
    features: Optional[list] = None
    """The molecule's color feature table (``None`` on a shape-only build):
    ``[(type, slots), ...]`` with ``type`` the generator's ``FEATURES_ENUM`` id
    (1-based) and ``slots`` the feature's heavy atoms as **slot** indices
    (stored atom order, FORMAT.md section 8.2), ascending within a feature.
    Features are grouped by ascending type and, within a type, kept in the
    generator's enumeration order; the builder sorts them by ``(type, slots)``
    (FORMAT.md section 8.1 rule 4) unless asked to keep that order for a
    measurement. One table per molecule, perceived once, on the molecule's
    stored SMILES (``_group_by_name``, ``feature_table``), so every
    conformer shares it by construction."""


def _add_timing(timing: Optional[dict], key: str, started: float) -> None:
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - started


def _descriptors(mol, timing: Optional[dict] = None) -> dict:
    """``rotor_count`` of the molecule as given. ``mw`` is
    ``_molecular_weight``'s, set at emission. The ``smiles``
    column is not computed here: it is ``canonical_heavy``'s, the very string
    the stored atom order was taken from, so its atom count always agrees
    with ``n_real`` (``RemoveHs`` would keep deuterium)."""
    from rdkit.Chem import rdMolDescriptors

    mw = None      # from the canonical graph: ``_molecular_weight``
    try:
        rot = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:
        rot = None
    return {"smiles": None, "mw": mw, "rotor_count": rot}


def _molecular_weight(heavy) -> Optional[float]:
    """Average molecular weight, hydrogens included, of the canonical
    heavy-atom molecule (its hydrogens counted as implicit/total H).

    A function of content: computed on the canonical atom order, so it does
    not depend on the source's atom order at float rounding. Isotopic hydrogens count as protium,
    as ``RemoveAllHs`` treats them; molecular weight still means the whole
    molecule, unlike ``hac``."""
    from rdkit.Chem import Descriptors
    try:
        return float(Descriptors.MolWt(heavy))
    except Exception:
        return None


def _smiles_output_order(mol) -> np.ndarray:
    """``_smilesAtomOutputOrder`` of the ``MolToSmiles`` call just made on
    ``mol``, as an ``int64`` permutation of its atoms (checked)."""
    if mol.GetNumAtoms() == 0:
        return np.zeros(0, dtype=np.int64)    # no heavy atoms: rejected later
    raw = mol.GetProp("_smilesAtomOutputOrder")
    order = np.array([int(t) for t in raw.strip().strip("[]").split(",")
                      if t.strip()], dtype=np.int64)
    if (order.size != mol.GetNumAtoms()
            or not np.array_equal(np.sort(order), np.arange(order.size))):
        raise ValueError(f"_smilesAtomOutputOrder {raw!r} is not a permutation "
                         f"of the {mol.GetNumAtoms()} atoms")
    return order


def canonical_heavy(mol):
    """``(heavy, smiles, order)``: the hydrogen-free molecule in canonical
    SMILES atom order, its SMILES, and the permutation (FORMAT.md
    sections 4.2 and 11.1).

    * ``smiles`` is ``Chem.MolToSmiles(strip_hydrogens(mol),
      isomericSmiles=False)`` -- every hydrogen removed, isotopic ones too
      (``RemoveAllHs``, never ``RemoveHs``, which keeps deuterium), and no stereochemistry,
      which lives in the coordinates.
    * ``order`` is ``_smilesAtomOutputOrder`` **of that same call** (the
      isomeric and non-isomeric canonical orders can differ):
      ``order[slot]`` is the index of the slot's atom in the stripped source
      molecule.
    * ``heavy`` is the stripped molecule renumbered by ``order``
      (``Chem.RenumberAtoms``, which carries every conformer and keeps the
      chiral tags consistent), so atom ``i`` of ``heavy`` -- and slot ``i`` of
      every stored conformer -- is atom ``i`` of ``Chem.MolFromSmiles(smiles)``.

    The coordinates are permuted **before** the frame is computed
    (``_prep_raw``), so the prepared frame is a function of content, not of
    the source's atom order; the centroid and covariance are therefore summed
    in a new order (a declared numerics change). Canonical order is a property of the RDKit release (FORMAT.md
    section 13.3). Feature perception runs on the molecule ``smiles`` parses
    to, hydrogens added (``feature_table``; FORMAT.md 8.2), whose atom ``i``
    is slot ``i`` too.
    """
    from rdkit import Chem

    stripped = strip_hydrogens(mol)
    smiles = Chem.MolToSmiles(stripped, isomericSmiles=False)
    order = _smiles_output_order(stripped)
    heavy = Chem.RenumberAtoms(stripped, order.tolist()) if order.size else stripped
    return heavy, smiles, order


#: The per-slot properties ``smiles_slot_mismatch`` compares, in
#: ``_slot_signature``'s tuple order, by the name its reason text uses.
_SLOT_FIELDS = ("element", "formal charge", "total H", "radical electrons")


def _slot_signature(mol):
    """Per slot ``(atomic number, formal charge, total H, radical
    electrons)`` and the bond set ``{(i, j, bond type)}`` with ``i < j`` --
    what ``smiles_slot_mismatch`` requires the parsed SMILES to reproduce,
    slot for slot (``_SLOT_FIELDS`` names the per-slot entries).

    Radical electrons are compared because the parsed molecule is the one
    color features are perceived on (``parse_stored_smiles``): element,
    charge, hydrogens and bonds do not always fix them -- a pickled record
    can carry a radical count that its SMILES, read back, does not."""
    atoms = [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetTotalNumHs(),
              a.GetNumRadicalElectrons())
             for a in mol.GetAtoms()]
    bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()))
                   for b in mol.GetBonds())
    return atoms, bonds


def parse_stored_smiles(smiles: Optional[str], heavy):
    """``(parsed, why)``: the stored SMILES parsed as every reader parses it
    (``MolFromSmiles``, CXSMILES disabled; FORMAT.md 11.1) and ``None`` when
    it reproduces the stored slots, else ``(parsed or None, reason)``.

    ``parsed`` is the molecule the builder perceives color features on
    (``feature_table``; FORMAT.md 8.2): the very molecule the
    ``deep`` validator and query featurization from the corpus definition
    re-perceive, so build and verify agree by construction. Atom ``i`` of
    ``parsed`` is slot ``i``.

    The check: atom ``i`` of the parsed molecule must have slot ``i``'s
    element, formal charge, total hydrogen count and radical electron count,
    and every bond must join the same two slots with the same bond type
    (aromatic, single, double, ...). ``heavy`` is the canonical heavy-atom
    molecule the slots were taken from (``canonical_heavy``).

    RDKit can write a canonical SMILES it cannot read back: PubChem 12157944
    is written ``Cn1ccn(C)[si+]1C``, an aromatic charged silicon that does
    not kekulize on parsing. The builder rejects such a molecule (its reason
    in ``build.json``'s rejected list) rather than refusing the build."""
    from rdkit import Chem

    want_atoms, want_bonds = _slot_signature(heavy)
    if not want_atoms and not smiles:
        return None, None   # no heavy atoms: ``_why_reject`` rejects the molecule
    parsed = None
    if smiles:
        from rdkit import rdBase
        params = Chem.SmilesParserParams()
        params.allowCXSMILES = False
        # RDKit logs a SMILES it cannot parse verbatim to stderr ("Failed
        # parsing SMILES '...'"), which would put the structure into a build
        # log; the failure is recorded in the reject reason instead.
        with rdBase.BlockLogs():
            parsed = Chem.MolFromSmiles(smiles, params)
    if parsed is None:
        return None, (f"SMILES round-trip check: its canonical SMILES {smiles!r} "
                      f"does not parse; slot i must be atom i of MolFromSmiles(smiles) "
                      f"(FORMAT.md 11.1)")
    got_atoms, got_bonds = _slot_signature(parsed)
    if got_atoms != want_atoms or got_bonds != want_bonds:
        if len(got_atoms) != len(want_atoms):
            where = "atom count"
        else:
            where = next(
                (f"slot {i}: " + ", ".join(
                    f"{field} {gv} vs stored {wv}"
                    for field, gv, wv in zip(_SLOT_FIELDS, g, w) if gv != wv)
                 for i, (g, w) in enumerate(zip(got_atoms, want_atoms)) if g != w),
                "the bonds")
        return parsed, (f"SMILES round-trip check: its canonical SMILES {smiles!r} "
                        f"does not reproduce the stored slots ({where}); slot i must "
                        f"be atom i of MolFromSmiles(smiles), element, charge, "
                        f"hydrogens, radical electrons and bonds (FORMAT.md 11.1)")
    return parsed, None


def smiles_slot_mismatch(smiles: Optional[str], heavy) -> Optional[str]:
    """Why the stored SMILES does not parse to the stored slots, or None if
    it does: ``parse_stored_smiles``'s reason alone."""
    return parse_stored_smiles(smiles, heavy)[1]


def check_smiles_slots(name: str, smiles: Optional[str], heavy) -> None:
    """``smiles_slot_mismatch`` as a ``BuildRefused`` naming the molecule,
    for callers that want a hard stop (the census and spot-check scripts).
    The builder itself rejects such a molecule per molecule instead
    (``_group_by_name``)."""
    why = smiles_slot_mismatch(smiles, heavy)
    if why is not None:
        raise BuildRefused(f"molecule {name!r}: {why}")


def _hash_framed(h, data: bytes) -> None:
    """One length-framed item of a work unit's content digest."""
    h.update(struct.pack("<Q", len(data)))
    h.update(data)


def payload_digest(blobs) -> str:
    """Content digest of a binary-stream batch: its raw payloads, framed."""
    h = hashlib.sha256(b"roshambo3.unit.binary.v1")
    for blob in blobs:
        _hash_framed(h, blob)
    return h.hexdigest()


def _prep_raw(raw, timing: Optional[dict] = None):
    """Prepare real-atom coordinates: ``(coords, n_real, vaa)``.

    Shape only. Color pseudo-atoms are no longer prepared here: a molecule's
    features are a table of heavy-atom slots (``feature_table``), and their
    positions are derived per conformer from the **stored** int16 heavy-atom
    coordinates at gather (FORMAT.md section 8.3)."""
    t0 = time.perf_counter()
    prepared = prep_conformer(raw)
    _add_timing(timing, "canonical_frame_s", t0)
    t0 = time.perf_counter()
    vaa = self_volume(prepared)
    _add_timing(timing, "self_volume_s", t0)
    return prepared, prepared.shape[0], vaa


def _prep_one(mol, timing: Optional[dict] = None, canonical=None):
    """Prepare one RDKit molecule into ``(coords, n_real, vaa)``.

    Hydrogens are removed for the shape calculation, and the heavy atoms are
    put in canonical SMILES order before the frame is computed
    (``canonical_heavy``; ``canonical`` passes its result when the caller
    already has it). Color features are perceived separately, once per
    molecule, on its stored SMILES with hydrogens added (``feature_table``;
    FORMAT.md 8.2).
    """
    t0 = time.perf_counter()
    # strip_hydrogens (inside canonical_heavy), not Chem.RemoveHs: RemoveHs
    # retains isotopically labeled hydrogens, which the uniform-radius
    # kernel then scores as carbon.
    heavy = (canonical if canonical is not None else canonical_heavy(mol))[0]
    _add_timing(timing, "remove_hs_s", t0)
    t0 = time.perf_counter()
    raw = np.asarray(heavy.GetConformer().GetPositions(), dtype=np.float64)
    _add_timing(timing, "coordinates_s", t0)
    return _prep_raw(raw, timing)


def feature_table(color_generator, parsed, name: str,
                  timing: Optional[dict] = None) -> list:
    """The color feature table of one molecule in slot indices
    (FORMAT.md sections 8.1-8.2).

    ``parsed`` is the molecule's stored canonical SMILES parsed as readers
    parse it (``parse_stored_smiles``, after its round-trip check passed), so
    atom ``i`` of ``parsed`` is slot ``i``. Features are perceived on
    ``perception_molecule(parsed)`` -- ``Chem.AddHs`` of it -- which is
    exactly the molecule the ``deep`` validator and query featurization from
    the corpus definition perceive on: build and verify agree by
    construction, and library and query perceive under one aromaticity model
    whatever model the input carried (a pickled input is not re-sanitized
    and keeps its writer's model). No coordinate is read.

    ``AddHs`` appends the hydrogens after the heavy atoms, so the remap to
    slots is the identity on heavy atoms. A feature that references a
    hydrogen cannot be stored (its position must come from heavy atoms
    alone) and is refused, naming the molecule and the family:
    ``BuildRefused``. None does under BaseFeatures; a custom definition could.

    Returns ``[(type, slots), ...]``: slots ascending within a feature;
    features grouped by ascending type, within a type in the plan's
    (generator's enumeration) order. Duplicates are kept.
    """
    t0 = time.perf_counter()
    n_heavy = parsed.GetNumAtoms()
    if any(a.GetAtomicNum() == 1 for a in parsed.GetAtoms()):
        raise RuntimeError(
            f"molecule {name!r}: its parsed SMILES holds a hydrogen atom; the "
            f"stored SMILES is of the hydrogen-free molecule (FORMAT.md 11.1)")
    mol = perception_molecule(parsed)
    plan = color_generator.make_feature_plan(mol)
    if plan is None:
        raise BuildRefused(
            f"color generator {type(color_generator).__name__} gives no feature "
            f"plan (type, atoms) for molecule {name!r}; a corpus stores each "
            f"feature as its atoms (FORMAT.md section 8), so it cannot be "
            f"built with this generator")
    names = None
    out = []
    for atoms, t in zip(plan.atom_ids, np.asarray(plan.types).tolist()):
        atoms = [int(x) for x in atoms]
        if any(x >= n_heavy for x in atoms):
            if names is None:
                names = {v: k for k, v in color_generator.FEATURES_ENUM.items()}
            raise BuildRefused(
                f"molecule {name!r}: its {names.get(int(t), t)!r} feature on "
                f"atoms {atoms} references a hydrogen. A feature's "
                f"position is derived from the stored heavy atoms alone "
                f"(FORMAT.md section 8.2), so a definition "
                f"that matches hydrogens cannot be stored")
        out.append((int(t), tuple(sorted(atoms))))
    out.sort(key=lambda f: f[0])          # stable: within a type, plan order
    _add_timing(timing, "color_features_s", t0)
    return out


def perception_molecule(mol):
    """``mol`` with every hydrogen explicit (``Chem.AddHs``, which appends
    new hydrogens after the existing atoms): the molecule features are
    perceived on. At build ``mol`` is the parsed stored SMILES
    (``feature_table``), as in ``deep`` and query featurization."""
    from rdkit import Chem
    return Chem.AddHs(mol)


def _name_of(mol, index: int) -> str:
    from rdkit import Chem
    if mol.HasProp("_Name"):
        n = mol.GetProp("_Name").strip()
        if n:
            return n
    # No name: fall back to the canonical SMILES so the id is at least stable
    # across runs. A positional id would not be, and a corpus whose molecule ids
    # change when it is rebuilt is a corpus no result can cite.
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return f"_unnamed_{index}"


def index_sdf(path: str) -> tuple[list[int], list[bytes]]:
    """``(byte offset, title line)`` per record, for parallel reading.

    One sequential pass. SDF records are delimited by a ``$$$$`` line, so the
    offsets are exact rather than heuristic -- a split landing inside a record
    silently drops a molecule.

    The title line comes back too, and it is not a convenience. A work unit must
    never split a *molecule*, only a molecule boundary: consecutive records
    sharing a name are one multi-conformer molecule, so a unit boundary between
    two of them makes both workers emit that molecule, each numbering its
    conformers from zero. The result is a corpus containing every straddling
    conformer twice under the same ``(mol_id, conf_idx)``. The duplicate-id check
    in the builder catches it, but only after the work is done -- so the split
    is made correctly instead.
    """
    offsets: list[int] = [0]
    titles: list[bytes] = []
    with open(path, "rb") as f:
        pos = 0
        want_title = True
        for line in f:
            if want_title:
                titles.append(line.strip())
                want_title = False
            pos += len(line)
            if line.strip() == SDF_DELIM:
                offsets.append(pos)
                want_title = True
    size = os.path.getsize(path)
    while offsets and offsets[-1] >= size:
        offsets.pop()
    titles = titles[:len(offsets)]
    return offsets, titles


class SdfSource:
    """Molecules from one or more SDF files with 3D coordinates.

    ``merge_conformers`` (default True) lets a molecule stored as one record
    per conformer -- LIT-PCBA's and PUR2's layout, ~59 records sharing a name --
    take ``_group_by_name``'s multi-conformer branch, which canonicalizes
    the graph once per molecule instead of once per conformer. It applies only when the records are provably one graph
    (see ``_merge_records``) and a color generator is set; in every other
    case, and whenever the merged path would raise RDKit's sanitize family,
    the molecule takes exactly the per-record path. The
    ``MoleculeRecord`` and ``failures`` produced are identical either way;
    ``False`` exists so tests can run both paths.

    Stereochemistry is respected conservatively. Records that differ in it are
    merged ONLY when the generator positively certifies that its feature
    patterns cannot depend on stereo (``feature_patterns_stereo_free``: the
    default RDKit generator checks every expanded SMARTS for stereo syntax; a
    generator that does not answer, or answers anything but ``True``, is
    treated as stereo-dependent). Otherwise any difference -- one record's
    chirality or bond stereo assigned differently from another's, or one
    assigned and the other unspecified, or a signature RDKit cannot compute --
    keeps the whole molecule on the per-record path.
    """

    def __init__(self, paths: Sequence[str], color_generator=None,
                 merge_conformers: bool = True):
        self.paths = [str(p) for p in paths]
        self.color_generator = color_generator
        self.merge_conformers = bool(merge_conformers)
        self._stereo_free = None        # the generator's certificate, once asked
        # per-unit chemistry rejections; reset at the top of iter_unit
        self.failures: list[tuple[str, str]] = []
        # per-unit {molecule name: conformers dropped} and content digest;
        # reset by iter_unit
        self.dropped_conformers: dict = {}
        self.unit_sha256: Optional[str] = None
        for p in self.paths:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

    def units(self, target_records: int = 20_000) -> list[tuple]:
        """Split into work units, each a list of byte ranges to read.

        Records are grouped **by name across the whole file**, not by adjacency.
        Real SDFs interleave: in ROSHAMBO2's own test file, 16 of 480 molecules
        have their conformers split across two or three separate runs, with other
        molecules in between. Grouping by adjacency reads those as several
        single-conformer molecules that then collide on one ``mol_id`` -- so the
        builder's duplicate check fires and the build fails, which is better than
        the alternative but is not a build.

        The cost is holding one offset and one title per record while planning.
        That is fine for the libraries anyone feeds to an SDF reader; a prepared ``.r3g`` corpus
        is the format for a corpus large enough for it not to be.
        """
        out = []
        for p in self.paths:
            offs, titles = index_sdf(p)
            if not offs:
                continue
            size = os.path.getsize(p)
            ends = offs[1:] + [size]

            by_name: dict = {}
            for i, name in enumerate(titles):
                by_name.setdefault(name, []).append(i)

            unit: list = []
            n_in_unit = 0
            for name, records in by_name.items():
                unit.append([(offs[i], ends[i]) for i in records])
                n_in_unit += len(records)
                if n_in_unit >= target_records:
                    out.append((p, unit))
                    unit, n_in_unit = [], 0
            if unit:
                out.append((p, unit))
        return out

    def _read_molecule(self, f, ranges) -> list[bytes]:
        parts = []
        for start, end in ranges:
            f.seek(start)
            parts.append(f.read(end - start))
        return parts

    def unit_digest(self, unit) -> str:
        """The content digest ``iter_unit`` computes for ``unit``, re-read
        without any RDKit work: a resume checks its committed units with it."""
        path, molecules = unit
        h = hashlib.sha256(b"roshambo3.unit.sdf.v1")
        with open(path, "rb") as f:
            for ranges in molecules:
                for part in self._read_molecule(f, ranges):
                    _hash_framed(h, part)
        return h.hexdigest()

    def iter_unit(self, unit) -> Iterator[MoleculeRecord]:
        from rdkit import Chem

        path, molecules = unit
        self.failures = []
        self.dropped_conformers = {}
        self.unit_sha256 = None
        h = hashlib.sha256(b"roshambo3.unit.sdf.v1")
        with open(path, "rb") as f:
            for ranges in molecules:
                # One molecule's records, concatenated: a valid mini-SDF whose
                # every record shares a name, so adjacency grouping inside it is
                # correct by construction.
                parts = self._read_molecule(f, ranges)
                for part in parts:
                    _hash_framed(h, part)
                supplier = Chem.ForwardSDMolSupplier(io.BytesIO(b"".join(parts)),
                                                     removeHs=False, sanitize=True)
                # A record RDKit cannot parse is skipped, which renumbers the
                # molecule's later conformers (conf_idx counts accepted ones):
                # counted against the record's title, which is the name every
                # record of this molecule shares.
                title = parts[0].split(b"\n", 1)[0].strip().decode("utf-8", "replace")
                supplier = _count_unparsed(
                    supplier, title or f"(untitled record at byte {ranges[0][0]} of {path})",
                    self.dropped_conformers)
                if self.merge_conformers and self.color_generator is not None:
                    yield from self._iter_merged(supplier)
                else:
                    yield from _group_by_name(supplier, 0, self.color_generator,
                                              self.failures, dropped=self.dropped_conformers)
        self.unit_sha256 = h.hexdigest()

    def _iter_merged(self, supplier) -> Iterator[MoleculeRecord]:
        """One molecule's records through the merged path when that is exact.

        Falls back to the per-record path -- today's behavior, on the very
        same parsed records -- when the records are not one graph, when the
        molecule has more records than a corpus can store (bounding the merged
        Mol at ``MAX_MERGED_CONFORMERS``), when the merged Mol does not yield
        one record with every conformer, and when preparing the merged Mol
        raises RDKit's sanitize family (so a reject stays a reject, with the
        per-record ``failures`` entries, rather than escaping from the merged
        branch, which has no handler).

        Records that are one graph up to stereochemistry are merged too ONLY
        when the generator certifies its patterns stereo-free (see the class
        docstring); then each record retains its own source atom permutation. See ``_merge_records``.
        """
        from rdkit.Chem.rdchem import MolSanitizeException

        # Parse at most one more record than the bound, so a pathological
        # molecule is never fully materialized: past the bound it streams
        # through the per-record path exactly as before.
        mols = []
        for mol in supplier:
            mols.append(mol)
            if len(mols) > MAX_MERGED_CONFORMERS:
                break
        if len(mols) > MAX_MERGED_CONFORMERS:
            yield from _group_by_name(itertools.chain(mols, supplier), 0,
                                      self.color_generator, self.failures,
                                      dropped=self.dropped_conformers)
            return
        n_failures = len(self.failures)
        dropped_before = dict(self.dropped_conformers)
        if self._stereo_free is None:
            self._stereo_free = _generator_is_stereo_free(self.color_generator)
        try:
            merged, same_stereo = _merge_records(
                mols, stereo_independent=self._stereo_free)
            ok = merged is not None
            if ok:
                recs = list(_group_by_name([merged], 0, self.color_generator,
                                           self.failures, dropped=self.dropped_conformers))
                n_present = merged.GetNumConformers()
                if len(recs) != 1 or len(recs[0].conformers) != n_present:
                    # A conformer dropped (or the molecule rejected) inside
                    # the merged branch: let the per-record path decide,
                    # rather than re-deriving it.
                    ok = False
                else:
                    # Per record, exactly as the per-record path takes them:
                    # in the first record's atom order (``_align_to``), with
                    # the permutation composed back to the record's own
                    # source order.
                    first = next(m for m in mols if m is not None)
                    ref = _RefKeys(first)
                    merged_order = recs[0].atom_order[0]
                    orders = []
                    for m in mols:
                        if m is None:
                            continue
                        aligned, heavy_map = _align_to(ref, m)
                        canonical = (None, None, merged_order)
                        order = _source_order(canonical, heavy_map)
                        if order is None:
                            ok = False
                            break
                        orders.append(order)
                    if ok:
                        recs[0].atom_order = orders
        except MolSanitizeException:
            ok = False
        if not ok:
            del self.failures[n_failures:]
            self.dropped_conformers.clear()
            self.dropped_conformers.update(dropped_before)
            yield from _group_by_name(mols, 0, self.color_generator,
                                      self.failures, dropped=self.dropped_conformers)
            return
        yield from recs


def _count_unparsed(supplier, name: str, dropped: dict):
    """Pass ``supplier``'s records through, counting the ``None``s (records
    RDKit could not parse) against ``name`` in ``dropped``."""
    for mol in supplier:
        if mol is None:
            dropped[name] = dropped.get(name, 0) + 1
        yield mol


class RDKitSource:
    """Molecules from an in-memory sequence of RDKit molecules."""

    def __init__(self, mols: Sequence, color_generator=None):
        self.mols = list(mols)
        self.color_generator = color_generator
        # per-unit chemistry rejections; reset at the top of iter_unit
        self.failures: list[tuple[str, str]] = []
        self.dropped_conformers: dict = {}
        self.unit_sha256: Optional[str] = None

    def __getstate__(self):
        # macOS multiprocessing spawns rather than forks. RDKit's default
        # pickle omits _Name and other properties, losing the grouping identity
        # computed by units() in the parent. Serialize explicitly without
        # changing RDKit's process-global default pickle policy.
        from rdkit import Chem
        state = dict(self.__dict__)
        state["mols"] = [None if mol is None else
                         mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
                         for mol in self.mols]
        return state

    def __setstate__(self, state):
        from rdkit import Chem
        self.__dict__.update(state)
        self.mols = [None if blob is None else Chem.Mol(blob)
                     for blob in state["mols"]]

    def units(self, target_records: int = 20_000) -> list[tuple]:
        """As for SDF: grouped by name across the whole sequence, not by position."""
        by_name: dict = {}
        for i, m in enumerate(self.mols):
            if m is None:
                continue
            name = (m.GetProp("_Name").strip() if m.HasProp("_Name") else "")
            by_name.setdefault(name or f"_unnamed_{i}", []).append(i)

        out, unit, n = [], [], 0
        for name, idxs in by_name.items():
            unit.append(idxs)
            n += len(idxs)
            if n >= target_records:
                out.append(tuple(unit))
                unit, n = [], 0
        if unit:
            out.append(tuple(unit))
        return out

    def unit_digest(self, unit) -> str:
        """Content digest of a unit: its molecules pickled with every
        property (an in-memory source has no input file to identify)."""
        from rdkit import Chem
        h = hashlib.sha256(b"roshambo3.unit.rdkit.v1")
        for idxs in unit:
            for i in idxs:
                mol = self.mols[i]
                _hash_framed(h, b"" if mol is None else
                             mol.ToBinary(Chem.PropertyPickleOptions.AllProps))
        return h.hexdigest()

    def iter_unit(self, unit) -> Iterator[MoleculeRecord]:
        self.failures = []
        self.dropped_conformers = {}
        self.unit_sha256 = self.unit_digest(unit)
        for idxs in unit:
            yield from _group_by_name([self.mols[i] for i in idxs], idxs[0],
                                      self.color_generator, self.failures,
                                      dropped=self.dropped_conformers)


def _read_exact(reader, size: int, *, record: int, what: str) -> bytes:
    """Read exactly ``size`` bytes from a decompressor stream.

    ``ZstdDecompressionReader.read`` is allowed to return less than requested.
    Treating one short read as a whole pickle corrupts the framing of every
    record after it, so this intentionally loops rather than relying on the
    happy-path behavior of a particular zstandard release.
    """
    out = bytearray()
    while len(out) < size:
        block = reader.read(size - len(out))
        if not block:
            raise ValueError(
                f"truncated RDKit binary-zstd {what} at record {record}: "
                f"wanted {size} bytes, got {len(out)}")
        out.extend(block)
    return bytes(out)


class RdkitBinaryZstdSource:
    """One solid zstd stream of length-prefixed ``Mol.ToBinary`` records.

    A record is already one molecule with all of its conformers. Unlike SDF,
    there is no global title regrouping pass and no safe byte range at which a
    second decompressor can start, hence one serial work unit. The builder's
    scatter files, not this source, are the external-memory partition.
    """

    # The compressed input is necessarily read serially, but its *payloads*
    # are independent molecules and can be prepared by separate workers.
    parallel_payloads = True

    def __init__(self, path: str, color_generator=None, max_records: Optional[int] = None,
                 profile: bool = False):
        self.path = str(path)
        self.color_generator = color_generator
        self.max_records = max_records
        self.profile = profile
        self.failures: list[tuple[str, str]] = []
        self.dropped_conformers: dict = {}
        self.unit_sha256: Optional[str] = None
        self.stats: dict[str, float | int] = {}
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        if max_records is not None and max_records < 1:
            raise ValueError("max_records must be positive")

    def units(self, target_records: int = 20_000) -> list[tuple]:
        del target_records
        return [(self.path, self.max_records)]

    def iter_payload_batches(self, records_per_batch: int):
        """Yield bounded raw-payload batches from the single decompressor.

        This deliberately does no RDKit work: the parent owns exactly one zstd
        stream and workers deserialize their own byte blobs.  At most one batch
        per worker is submitted by the builder, bounding parent-side RAM.
        """
        try:
            import zstandard
        except ImportError as e:
            raise ImportError("rdkit-binary-zstd input requires zstandard") from e
        if records_per_batch < 1:
            raise ValueError("records_per_batch must be positive")
        framed_s = 0.0
        n_records = 0
        compressed_total = os.path.getsize(self.path)
        compressed_read = 0
        slot = 0
        with open(self.path, "rb") as fh:
            with zstandard.ZstdDecompressor().stream_reader(fh) as reader:
                while self.max_records is None or n_records < self.max_records:
                    first = n_records
                    blobs = []
                    while (len(blobs) < records_per_batch and
                           (self.max_records is None or n_records < self.max_records)):
                        t0 = time.perf_counter()
                        header = reader.read(4)
                        framed_s += time.perf_counter() - t0
                        if not header:
                            break
                        if len(header) < 4:
                            header += _read_exact(reader, 4 - len(header),
                                                  record=n_records, what="header")
                        size = struct.unpack("<I", header)[0]
                        if size > (1 << 30):
                            raise ValueError(f"RDKit binary-zstd record {n_records} declares an implausible {size:,}-byte payload")
                        t0 = time.perf_counter()
                        blobs.append(_read_exact(reader, size, record=n_records,
                                                 what="payload"))
                        framed_s += time.perf_counter() - t0
                        n_records += 1
                    if not blobs:
                        break
                    yield slot, first, tuple(blobs)
                    slot += 1
                compressed_read = fh.tell()
        self.stats = {"source_records": n_records,
                      "source_compressed_bytes_read": compressed_read,
                      "source_compressed_bytes_total": compressed_total,
                      "frame_s": framed_s}

    def iter_unit(self, unit) -> Iterator[MoleculeRecord]:
        try:
            import zstandard
            from rdkit import Chem
        except ImportError as e:
            raise ImportError(
                "rdkit-binary-zstd input requires the 'zstandard' and 'rdkit' "
                "runtime dependencies") from e

        path, limit = unit
        self.failures = []
        self.dropped_conformers = {}
        self.unit_sha256 = None
        digest = hashlib.sha256(b"roshambo3.unit.binary.v1")
        framed_s = deserialize_s = 0.0
        n_records = n_conformers = 0
        compressed_total = os.path.getsize(path)
        compressed_read = 0
        with open(path, "rb") as fh:
            with zstandard.ZstdDecompressor().stream_reader(fh) as reader:
                while limit is None or n_records < limit:
                    t0 = time.perf_counter()
                    header = reader.read(4)
                    framed_s += time.perf_counter() - t0
                    if not header:
                        break
                    if len(header) < 4:
                        header += _read_exact(reader, 4 - len(header),
                                              record=n_records, what="header")
                    size = struct.unpack("<I", header)[0]
                    # RDKit pickles are small relative to this ceiling. It is
                    # an allocation guard, not a format claim: a corrupt length
                    # must not turn into a multi-gigabyte bytearray request.
                    if size > (1 << 30):
                        raise ValueError(
                            f"RDKit binary-zstd record {n_records} declares "
                            f"an implausible {size:,}-byte payload")
                    t0 = time.perf_counter()
                    blob = _read_exact(reader, size, record=n_records,
                                       what="payload")
                    framed_s += time.perf_counter() - t0
                    _hash_framed(digest, blob)
                    t0 = time.perf_counter()
                    try:
                        mol = Chem.Mol(blob)
                    except Exception as e:
                        self.failures.append((f"_record_{n_records}",
                                              f"{type(e).__name__}: {e}"))
                        n_records += 1
                        deserialize_s += time.perf_counter() - t0
                        continue
                    deserialize_s += time.perf_counter() - t0
                    if mol is None:
                        self.failures.append((f"_record_{n_records}",
                                              "RDKit returned None for binary payload"))
                        n_records += 1
                        continue
                    n_conformers += mol.GetNumConformers()
                    # A binary record is exactly one molecule, so this does
                    # not inherit SDF's across-file regrouping assumption.
                    yield from _group_by_name(
                        [mol], n_records, self.color_generator, self.failures,
                        timing=self.stats if self.profile else None,
                        contiguous_conformer_ids=True,
                        dropped=self.dropped_conformers)
                    n_records += 1
                # zstandard's stream reader closes its underlying file on
                # exit, so query the compressed position before leaving this
                # context rather than from the outer file context.
                compressed_read = fh.tell()
        self.unit_sha256 = digest.hexdigest()
        self.stats = {
            "source_records": n_records,
            "source_conformers": n_conformers,
            "source_compressed_bytes_read": compressed_read,
            "source_compressed_bytes_total": compressed_total,
            "frame_s": framed_s,
            "deserialize_s": deserialize_s,
            **({f"prep_{k}": v for k, v in self.stats.items()} if self.profile else {}),
        }


def _graph_key(mol, stereo: bool = True) -> bytes:
    """A molecule's graph, coordinates and properties excluded, as bytes.

    The pickle carries atom order, atomic numbers, formal charges, isotopes,
    explicit/implicit H counts, radicals, aromatic flags, the bond list
    (begin, end, type, aromatic flag), chiral tags, bond stereo, stereo groups
    and substance groups. Pickled with the process default property options,
    so records with equal keys have equal graph properties.
    ``stereo=False`` first removes the stereochemistry
    (``Chem.RemoveStereochemistry``: chiral tags, bond stereo and directions,
    stereo groups), leaving what the default feature plan reads -- the
    BaseFeatures.fdef SMARTS carry no chirality or bond direction --, holds
    every atom's hydrogens as a total explicit count and clears chiral tags
    and hybridization (the explicit/implicit split and a non-tetrahedral tag
    with its SP3D/SP3D2 follow 3D stereo perception, not the graph) and then
    lists the bonds in a canonical order (ascending ``(lower, upper)`` atom
    index, begin atom the lower), so that two molecules with the same atoms in
    the same order and the same bonds compare equal however their bond lists
    happen to be ordered. ``RenumberAtoms`` keeps the source's bond order, so
    without this two records that differ only in atom order would still differ
    after both were put in canonical atom order. Bond stereo cannot
    survive a bond rebuild, which is why only the stereo-free key does it.
    """
    from rdkit import Chem
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()
    if not stereo:
        Chem.RemoveStereochemistry(graph)
        # Hydrogen counts by total, not by how they are held: removing a
        # hydrogen from an atom that 3D perception marked chiral leaves an
        # *explicit* H count, from one it did not an *implicit* one, so two
        # conformer records of one molecule would differ only in stereo
        # perception (35 DUD-Z ids, e.g. the
        # pseudo-asymmetric C4 of 1,4-dihydropyridines).
        # Likewise stereo perception can leave a non-tetrahedral chiral tag
        # and the hybridization it implies (a distorted sulfonyl S read as
        # trigonal-bipyramidal, SP3D: one LIT-PCBA record of 20.4M), which
        # the graph itself does not determine: cleared too.
        graph.UpdatePropertyCache(strict=False)
        for atom in graph.GetAtoms():
            atom.SetNumExplicitHs(atom.GetTotalNumHs())
            atom.SetNoImplicit(True)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            atom.SetHybridization(Chem.HybridizationType.UNSPECIFIED)
        rw = Chem.RWMol(graph)
        bonds = sorted(
            (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
             max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
             b.GetBondType(), b.GetIsAromatic())
            for b in rw.GetBonds())
        for i, j, _, _ in bonds:
            rw.RemoveBond(i, j)
        for i, j, kind, aromatic in bonds:
            rw.AddBond(i, j, kind)
            rw.GetBondBetweenAtoms(i, j).SetIsAromatic(aromatic)
        graph = rw.GetMol()
    return graph.ToBinary()


class _RefKeys:
    """The first record of a molecule, as its later records are compared
    with it (keys computed once, the stereo-free one only when needed)."""

    def __init__(self, mol):
        self.mol = mol
        self.full = _graph_key(mol)
        self._free = None

    @property
    def free(self) -> bytes:
        if self._free is None:
            self._free = _graph_key(self.mol, stereo=False)
        return self._free


def _align_to(ref: "_RefKeys", mol):
    """``(aligned, heavy_map)``: ``mol`` renumbered into ``ref.mol``'s atom
    order, when they are one graph up to stereo in different atom orders;
    otherwise ``(mol, None)`` unchanged.

    Every record of a molecule is put in its first record's atom order before
    it is prepared, by both SDF paths, so that the merged path
    can take records that differ only in atom order, the two paths produce the
    same records, and every conformer of a molecule takes its symmetric atoms
    into the same slots. The correspondence comes from RDKit's canonical
    ranks (``breakTies``, chirality excluded) and is then **verified** --
    the renumbered record's stereo-free graph key must equal the first
    record's -- so a wrong correspondence is never used. Records it cannot
    align (a deuterated and a protium record, say: the hydrogen-bearing
    graphs differ) are prepared in their own order; whether they belong under
    one id is decided on the hydrogen-free canonical graph, not here.

    ``heavy_map[k]`` is the index in ``strip_hydrogens(mol)`` of heavy atom
    ``k`` of ``strip_hydrogens(aligned)``, so a permutation of the aligned
    record's heavy atoms is composed back to ``mol``'s own source order.
    """
    from rdkit import Chem

    if mol is ref.mol:
        return mol, None
    if _graph_key(mol) == ref.full or _graph_key(mol, stereo=False) == ref.free:
        return mol, None                       # already in the same atom order
    n = mol.GetNumAtoms()
    if n != ref.mol.GetNumAtoms():
        return mol, None
    try:
        mine = list(Chem.CanonicalRankAtoms(mol, breakTies=True,
                                            includeChirality=False))
        theirs = list(Chem.CanonicalRankAtoms(ref.mol, breakTies=True,
                                              includeChirality=False))
    except Exception:
        return mol, None
    by_rank = {r: i for i, r in enumerate(mine)}
    if len(by_rank) != n or set(theirs) != set(by_rank):
        return mol, None
    new_order = [by_rank[r] for r in theirs]
    aligned = Chem.RenumberAtoms(mol, new_order)
    if _graph_key(aligned, stereo=False) != ref.free:
        return mol, None
    # RenumberAtoms leaves ring info unset, and feature perception reads it:
    # recompute it exactly as sanitization did (symmetrized SSSR).
    Chem.SanitizeMol(aligned, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
    heavy_new = [k for k in range(n) if aligned.GetAtomWithIdx(k).GetAtomicNum() != 1]
    heavy_old = {i: r for r, i in enumerate(
        i for i in range(n) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1)}
    heavy_map = np.array([heavy_old[new_order[k]] for k in heavy_new], dtype=np.int64)
    return aligned, heavy_map


def _source_order(canonical, heavy_map) -> Optional[np.ndarray]:
    """``canonical``'s permutation in the source record's own heavy-atom
    order, or None if ``heavy_map`` does not describe its heavy atoms."""
    order = canonical[2]
    if heavy_map is None:
        return order
    if heavy_map.size != order.size:
        return None
    return heavy_map[order]


def _generator_is_stereo_free(generator) -> bool:
    """Whether ``generator`` has positively certified its feature patterns
    stereo-independent. Anything short of an explicit ``True`` -- no such
    method, an exception, a truthy non-bool -- is False: "not known" is never
    "independent"."""
    check = getattr(generator, "feature_patterns_stereo_free", None)
    if check is None:
        return False
    try:
        return check() is True
    except Exception:
        return False


def _stereo_signature(mol):
    """Everything stereochemical RDKit holds about ``mol`` as a hashable tuple
    (per-atom chiral tags, per-bond stereo, direction and stereo atoms, the
    potential stereo elements with their specified/unspecified/unknown status,
    and stereo groups), or ``None`` when it cannot be computed. ``None`` is
    "unknown": the caller must not treat it as equal to anything, itself
    included."""
    from rdkit import Chem
    try:
        atoms = tuple(int(a.GetChiralTag()) for a in mol.GetAtoms())
        bonds = tuple((int(b.GetStereo()), int(b.GetBondDir()),
                       tuple(int(i) for i in b.GetStereoAtoms()))
                      for b in mol.GetBonds())
        potential = tuple(sorted(
            (int(i.type), int(i.specified), int(i.centeredOn))
            for i in Chem.FindPotentialStereo(mol)))
        groups = tuple((int(g.GetGroupType()), tuple(a.GetIdx() for a in g.GetAtoms()))
                       for g in mol.GetStereoGroups())
    except Exception:
        return None
    return atoms, bonds, potential, groups


def _merge_records(mols, *, stereo_independent):
    """``(merged, same_stereo)``: one Mol carrying every record's conformer, or
    ``(None, None)`` if merging them would not be exact.

    ``stereo_independent`` (required, no default) is the generator's
    certificate that its feature patterns cannot depend on stereochemistry.
    When it is False the merge is STRICT: every record must carry the same
    stereochemistry as the first, by both the pickled graph key and
    ``_stereo_signature``, so known-different stereo never merges, assigned
    stereo never merges with unspecified, and stereo RDKit cannot report
    (``None``) never merges at all, not even with itself. In strict mode a
    merged Mol therefore always has ``same_stereo`` True.

    ``mols`` are one molecule's parsed SDF records in file order (``None`` for
    a record RDKit could not parse, skipped by both paths and counted as a
    dropped conformer by the source). They are merged only if every parsed
    record has exactly one conformer, the same ``_name_of`` as the first, and,
    once put in the first record's atom order (``_align_to``, exactly as the
    per-record path puts it), a byte-equal stereo-free ``_graph_key`` -- which
    means the same atom count, atomic numbers, formal charges, isotopes, bond
    list (type, aromatic flag) and explicit-H counts. Records that differ only
    in atom order therefore merge; records that differ in more return
    ``(None, None)`` and take the per-record path, which is where a charge or
    protonation variant is refused (``_group_by_name``: the one place the rule
    is decided, on the hydrogen-free graph, so a deuterated and a protium
    record -- different here, equal there -- are accepted). Checked before any
    ``AddConformer``, which itself raises on an atom-count mismatch. Fewer
    than two parsed records: nothing to merge.

    ``same_stereo`` says whether the full keys are equal too. SDF parsing
    assigns chiral tags and bond stereo per record from its 3D coordinates, and
    real conformer sets disagree on them (an unspecified center embedded both
    ways: 57 of PUR2's 811 molecules, 56 by a chiral
    tag and 1 by bond stereo). The feature plan does not read them;
    the boolean records whether the source stereochemistry agrees.

    Conformer order is record order, with ids set explicitly to 0..n-1: the
    multi-conformer branch looks conformers up by id and by position alike.
    Each conformer's coordinates are its aligned record's, so they follow the
    first record's atom order.
    """
    from rdkit import Chem
    # Positions count the unparsed records, as _group_by_name's index does, so
    # the name a nameless, un-SMILES-able record would get agrees too.
    present = [(i, m) for i, m in enumerate(mols) if m is not None]
    if len(present) < 2:
        return None, None
    first = present[0][1]
    name = _name_of(first, present[0][0])
    ref = _RefKeys(first)
    same_stereo = True
    first_signature = None
    if not stereo_independent:
        first_signature = _stereo_signature(first)
        if first_signature is None:
            return None, None
    records = []
    for i, m in present:
        if m.GetNumConformers() != 1:
            return None, None
        if m is first:
            records.append(m)
            continue
        if _name_of(m, i) != name:
            return None, None
        # In the first record's atom order, as the per-record path takes it.
        m, _ = _align_to(ref, m)
        if _graph_key(m) == ref.full:
            if (not stereo_independent
                    and _stereo_signature(m) != first_signature):
                return None, None
        elif not stereo_independent or _graph_key(m, stereo=False) != ref.free:
            return None, None
        else:
            same_stereo = False
        records.append(m)
    merged = Chem.Mol(first)
    merged.RemoveAllConformers()
    for i, m in enumerate(records):
        conf = Chem.Conformer(m.GetConformer())
        conf.SetId(i)
        merged.AddConformer(conf, assignId=False)
    if [c.GetId() for c in merged.GetConformers()] != list(range(len(records))):
        raise AssertionError("merged conformer ids are not 0..n-1 in record order")
    return merged, same_stereo


def _try_prep(mol, name: str, failures,
              timing: Optional[dict] = None, ref=None):
    """Prepare one single-conformer record, or record why it could not be.

    Returns ``(prepared, canonical, order)`` -- the conformer tuple, the
    ``canonical_heavy`` triple of the record (in ``ref``'s atom order when
    it can be aligned to it, see ``_align_to``) and the permutation in the
    record's own source order -- or None for a rejected record. Color
    features are not perceived here but once per molecule, on its stored
    SMILES (``_group_by_name``).

    Only RDKit's sanitize family is caught, and the width of that clause is
    load-bearing: `MolSanitizeException` **subclasses ValueError**, so a guard
    written as `except ValueError: raise` would re-raise the very exception it
    is meant to absorb. Catching the RDKit base class instead keeps our own
    layout invariants in `_prep_one` -- the color-order and type-range checks,
    both plain `ValueError` -- fatal, because those mean the corpus format is
    being violated and that is a bug rather than a bad molecule.
    """
    from rdkit.Chem.rdchem import MolSanitizeException

    try:
        aligned, heavy_map = (mol, None) if ref is None else _align_to(ref, mol)
        canonical = canonical_heavy(aligned)
        order = _source_order(canonical, heavy_map)
        if order is None:                      # not describable: own order
            aligned, canonical = mol, canonical_heavy(mol)
            order = canonical[2]
        prepared = _prep_one(aligned, timing=timing, canonical=canonical)
    except MolSanitizeException as e:           # bad chemistry, this molecule only
        if failures is not None:
            failures.append((name, f"{type(e).__name__}: {e}"))
        return None
    return prepared, canonical, order


def _check_conformer_ids(name, ids: list, contiguous: bool) -> None:
    """A multi-conformer molecule's conformer ids must be unique, and, for
    sources whose record order is the conformer order (the binary stream and
    SQLite), exactly ``0..n-1`` in that order. Loud, never repaired: a
    conformer paired with another conformer's features is a confident wrong
    score. Refused, not retried: the input is what it is, so a resume would
    stop on the same molecule again."""
    if len(set(ids)) != len(ids):
        raise BuildRefused(
            f"molecule {name!r} carries duplicate conformer ids {ids[:12]}; "
            f"each conformer must have its own id")
    if contiguous and ids != list(range(len(ids))):
        raise BuildRefused(
            f"molecule {name!r} carries conformer ids {ids[:12]}"
            f"{'...' if len(ids) > 12 else ''}, not 0..{len(ids) - 1} in "
            f"record order, which this source requires")


def _formula(heavy) -> str:
    from rdkit.Chem import rdMolDescriptors
    return rdMolDescriptors.CalcMolFormula(heavy)    # with net charge


def _variant_refusal(name: str, kept: str, other: str,
                     same_formula: bool = False) -> BuildRefused:
    hint = (" The two have the same formula and net charge, so they may be "
            "resonance or Lewis forms of one state (an amidinium or guanidinium "
            "drawn with the charge on another nitrogen, a dative versus a "
            "charge-separated N-oxide or nitro group) or tautomers: if so, "
            "standardize the input's charges and bond orders so every record "
            "draws the molecule the same way." if same_formula else "")
    return BuildRefused(
        f"molecule {name!r}: its records differ in more than stereochemistry "
        f"or atom order -- a charge, protonation, tautomer, isotope, element, "
        f"atom-count or bond difference between hydrogen-free graphs (first "
        f"{kept!r}, then {other!r}; the non-isomeric SMILES can coincide when "
        f"only isotopes or hydrogen counts differ). A corpus stores one graph "
        f"per molecule id (FORMAT.md 11.1): give each state its "
        f"own id." + hint)


def _group_by_name(mols, first_index: int, color_generator, failures=None,
                   timing: Optional[dict] = None,
                   contiguous_conformer_ids: bool = False,
                   dropped: Optional[dict] = None):
    """Group consecutive molecules sharing a name into one multi-conformer record.

    A molecule may also arrive with several conformers already attached, in
    which case each is taken.

    **Identity (FORMAT.md 4.2).** Every record is prepared in canonical
    SMILES heavy-atom order (``canonical_heavy``), after being put in the
    molecule's first record's atom order (``_align_to``). Records of one name
    whose hydrogen-free canonical graphs differ in more than stereochemistry
    -- compared as ``_graph_key(heavy, stereo=False)``, so atom-order and
    stereo-only differences are accepted, and a deuterated and a protium
    record are equal -- are **refused** (``BuildRefused`` naming the id): a
    charge or protonation variant cannot be stored under one id. Every path
    meets here: the per-record SDF path, the merged path's fallback and its
    merged Mol, the over-``MAX_MERGED_CONFORMERS`` stream, ``RDKitSource``,
    and the binary and SQLite sources' multi-conformer records. The record's
    ``meta["smiles"]`` is the SMILES the stored order was taken from, checked
    at emission to parse back to the stored slots (``smiles_slot_mismatch``),
    and ``atom_order`` carries each conformer's permutation. A molecule whose
    SMILES fails that check is **rejected whole, not fatal**: its reason is
    appended to ``failures`` (one entry for the molecule) and no record is
    emitted, so none of its conformers reaches the corpus. It is a property
    of the molecule's graph and RDKit's writer, not of the input's
    consistency, unlike a variant, which still refuses the build.

    A conformer whose chemistry RDKit will not process -- most often an
    aromatic perception that cannot be kekulized, which `Chem.RemoveHs` raises
    on rather than returns None for -- is **rejected, not fatal**. Its reason
    is appended to ``failures`` so the caller can surface it in the build's
    reject report, it is counted per name in ``dropped`` (a dropped conformer
    renumbers the molecule's later ``conf_idx``, which counts accepted
    conformers in source order), and a molecule left with no usable conformer
    emits no record at all. Before this, one such molecule anywhere in the
    input aborted the whole `prepare` pool: a 10M-molecule build died ~25
    minutes in on a single unkekulizable ring, discarding every shard already
    written. The SDF path reaches this more readily than the in-memory one,
    because a molecule that round-trips through a mol block can parse cleanly
    and still carry aromatic flags that will not kekulize.

    **Color (FORMAT.md 8.2).** With a color generator, the
    feature table is perceived once per molecule, at emission, **after** the
    SMILES round-trip check passes (a molecule that fails it is rejected and
    never perceived), on ``Chem.AddHs`` of the molecule that check parsed
    (``parse_stored_smiles``, ``feature_table``) -- the molecule ``deep``
    and query featurization perceive on. No source record is perceived, so
    the input's aromaticity model (a pickle keeps its writer's; an SDF
    record is re-sanitized on reading) cannot reach the table, and every
    conformer shares the one table by construction. A sanitize-family error
    from perception rejects the molecule whole, as the round-trip check does.

    ``contiguous_conformer_ids``: a multi-conformer molecule must carry
    conformer ids ``0..n-1`` (unique ids are required always). The binary and
    SQLite sources set it.
    """
    pending_name: Optional[str] = None
    confs: list = []
    orders: list = []
    meta: dict = {}
    ident: dict = {}
    idx = first_index

    def drop(name, n=1):
        if dropped is not None and n:
            dropped[name] = dropped.get(name, 0) + n

    def accept_graph(name, canonical):
        """The variant rule, on the hydrogen-free canonical graph."""
        key = _graph_key(canonical[0], stereo=False)
        if "key" not in ident:
            ident.update(key=key, smiles=canonical[1], heavy=canonical[0])
        elif key != ident["key"]:
            raise _variant_refusal(
                name, ident["smiles"], canonical[1],
                _formula(ident["heavy"]) == _formula(canonical[0]))

    def emit():
        if pending_name is None or not confs:
            return None
        n_reals = {c[1] for c in confs}
        if len(n_reals) != 1:
            # Bucketing on n_real is what keeps a molecule's conformers together,
            # and it only works because they share it. If they do not, the input
            # is not what it claims to be (and will not change on a resume).
            raise BuildRefused(
                f"molecule {pending_name!r} has conformers with different heavy "
                f"atom counts {sorted(n_reals)} -- conformers of one molecule "
                f"must share a heavy-atom count, and bucketing depends on it")
        parsed, why = parse_stored_smiles(ident["smiles"], ident["heavy"])
        if why is not None:
            # Rejected whole, as ``build._why_reject`` rejects at layout: the
            # identifier is the molecule's name, so the id gate treats it as
            # it treats every other reject. See FORMAT.md 11.1.
            if failures is not None:
                failures.append((pending_name, why))
            return None
        features = None
        if color_generator is not None:
            # After the round-trip check, never before: a molecule that fails
            # it is rejected above and never perceived.
            features = []
            if parsed is not None:          # None only without heavy atoms
                from rdkit.Chem.rdchem import MolSanitizeException
                try:
                    features = feature_table(color_generator, parsed, pending_name,
                                             timing=timing)
                except MolSanitizeException as e:
                    if failures is not None:
                        failures.append((pending_name, f"{type(e).__name__}: {e}"))
                    return None
        return MoleculeRecord(ext_id=pending_name,
                              mol_id=stable_id(pending_name),
                              meta=dict(meta, smiles=ident["smiles"],
                                        mw=_molecular_weight(ident["heavy"]),
                                        hac=confs[0][1]),
                              conformers=confs.copy(),
                              atom_order=orders.copy(),
                              features=features)

    ref = None
    for mol in mols:
        if mol is None:
            idx += 1
            continue
        name = _name_of(mol, idx)
        if name != pending_name:
            rec = emit()
            if rec is not None:
                yield rec
            t0 = time.perf_counter()
            pending_name, confs, orders, meta = name, [], [], _descriptors(mol)
            ident = {}
            ref = None
            _add_timing(timing, "descriptors_s", t0)
        n_conf = mol.GetNumConformers()
        if n_conf == 0:
            idx += 1
            continue
        if ref is None:
            ref = _RefKeys(mol)
        if n_conf == 1:
            got = _try_prep(mol, name, failures, timing, ref)
            if got is None:
                drop(name)
            else:
                prepared, canonical, order = got
                accept_graph(name, canonical)
                confs.append(prepared)
                orders.append(order)
            idx += 1
            continue

        from rdkit.Chem.rdchem import MolSanitizeException
        conformers = list(mol.GetConformers())
        _check_conformer_ids(name, [c.GetId() for c in conformers],
                             contiguous_conformer_ids)
        try:
            # One graph for every conformer: strip and canonicalize once
            # (``RenumberAtoms`` carries every conformer, by id).
            aligned, heavy_map = _align_to(ref, mol)
            canonical = canonical_heavy(aligned)
            order = _source_order(canonical, heavy_map)
            if order is None:
                aligned, canonical = mol, canonical_heavy(mol)
                order = canonical[2]
        except MolSanitizeException as e:
            # Bad chemistry rejects the molecule, exactly as the
            # per-conformer branch rejects each of its conformers (one
            # failures entry per conformer), instead of killing the build
            # on every resume.
            if failures is not None:
                failures.extend((name, f"{type(e).__name__}: {e}")
                                for _ in conformers)
            drop(name, len(conformers))
            idx += 1
            continue
        accept_graph(name, canonical)
        heavy = canonical[0]
        for c in conformers:
            try:
                # By conformer ID: the position names another conformer
                # whenever the ids are not 0..n-1 in order.
                raw = np.asarray(heavy.GetConformer(c.GetId()).GetPositions(),
                                 dtype=np.float64)
                prepared = _prep_raw(raw, timing)
            except MolSanitizeException as e:
                if failures is not None:
                    failures.append((name, f"{type(e).__name__}: {e}"))
                drop(name)
                continue
            confs.append(prepared)
            orders.append(order)
        idx += 1

    rec = emit()
    if rec is not None:
        yield rec

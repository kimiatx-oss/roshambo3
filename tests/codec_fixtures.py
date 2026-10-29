"""Small, real-writer format-3 fixtures for decoder and consumer tests.

Written for format 3 (``doc/FORMAT.md`` Appendix B.8). The corpus is written through the format's
own writer (``format.r3g.writer.write_corpus``), not through the builder, so
its rows are exactly the arrays below; it passes ``roshambo3 verify --tier
scan`` (recorded in the scan cache, so it opens for a screen). These
fixtures exercise readers and metadata contracts; they do not produce score
or accuracy references.

The corpus (``block_conf`` = the writer's ``block_rows``, default 3), in
corpus order ``(n_real, mol_id, conf_idx)``, all in bucket 0:

* group 0: molecule 41 (``n_real`` 2, ``block_conf`` conformers) fills block
  0 exactly -- a molecule always ends at a block edge (FORMAT.md 6 rule 1;
  the format-2 fixture split molecule 900 across that edge, which format 3
  forbids) -- then molecule 900 (``n_real`` 3, one conformer) is a short
  final block;
* group 1: molecule 17 (``n_real`` 4, two conformers), so global conformer
  ids continue at ``block_conf + 1``.

Molecule ids are not gids and are not sorted in corpus-row order (41, 900,
17): rows follow the content order, which sorts by ``n_real`` before
``mol_id``. Physical block rows are ``[block_conf, 1, 2]``.

``color_layout``: ``"merged"`` writes a color corpus (the shipped RDKit
BaseFeatures definition and the default generator's canonical matrices, so a
query from ``PharmacophoreGenerator`` matches it) whose kernel layout puts
each row's derived pseudo-atoms after its real atoms (FORMAT.md 8.4);
``"none"`` a shape-only one. ``"split"`` does not exist in format 3 and is
refused, as are a format-2 ``codec`` (format 3 has the one encoding
``bp256-16``), ``shuffle=True`` and ``empty_final_group=True`` (an empty
group is invalid, FORMAT.md 9.3) -- refused, never mapped (Appendix B.8).

Each molecule has a real SMILES whose BaseFeatures perception (hydrogens
added, FORMAT.md 8.2) is exactly its stored feature table, so the corpus
passes ``verify --tier deep`` too (``tests/test_codec_fixtures.py`` checks
it): molecule 41 is methanol ``CO`` (an Acceptor and a Donor on the oxygen),
900 formic acid ``O=CO`` (two Acceptors, a Donor and the three-atom
NegIonizable, whose derived position is an integer mean), 17 fluoroform
``FC(F)F`` (no feature: row ``block_conf + 1`` is an empty-color row, as
row 4 was). Features are per molecule in format 3, so every conformer of a
molecule has the same counts. The coordinates stay synthetic (they are not
the SMILES' geometry), which no check relates; ``vaa`` and ``color_self``
describe the stored coordinates, as FORMAT.md 18 requires.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Iterator, Optional

import numpy as np

from roshambo3.format.r3g.limits import (BLOCK_DTYPE, FIRST_SECTION_OFFSET, LIMITS,
                                         group_file_name)
from roshambo3.reference.overlap import color_overlap, self_overlap

#: The one format-3 coordinate encoding (FORMAT.md section 5).
ENCODING = "bp256-16"
#: The format-3 coordinate step (FORMAT.md section 4: 2^-8 A).
COORD_SCALE = 2.0 ** -8

FEATURES_ENUM = {
    "Hydrophobe": 1,
    "Acceptor": 2,
    "Aromatic": 3,
    "Donor": 4,
    "PosIonizable": 5,
    "NegIonizable": 6,
}

#: The molecules, in corpus order: (mol_id, canonical non-isomeric SMILES,
#: feature table as BaseFeatures perceives it, FORMAT.md 8.1 order).
#: ``n_real`` is the SMILES' heavy-atom count; ``n_conf`` is ``block_conf``
#: for 41, 1 for 900 and 2 for 17.
MOLECULES = (
    (41, "CO", [(2, (1,)), (4, (1,))]),
    (900, "O=CO", [(2, (0,)), (2, (2,)), (4, (2,)), (6, (0, 1, 2))]),
    (17, "FC(F)F", []),
)
#: Whether ``verify --tier deep`` (re-perception included) holds for the
#: color layout. It does: the tables above are what RDKit 2026.09.1's
#: BaseFeatures perceives on the SMILES.
COLOR_DEEP_VALID = True


def _n_real(smiles: str) -> int:
    return sum(ch.isalpha() and ch.isupper() for ch in smiles)


@dataclass(frozen=True)
class CodecLayout:
    codec: str
    shuffle: bool
    color_layout: str


@dataclass(frozen=True)
class PhysicalBlock:
    """A format-3 block (FORMAT.md 6) and its expected row/GID interval.

    ``record`` is the block-table entry ``(first_row, rows, offset, length,
    m0, m1)`` (the format-2 fixture's was the RSB3 index record)."""

    stream: str
    group_index: int
    block_index: int
    row_start: int
    row_stop: int
    gid_start: int
    gid_stop: int
    path: Path
    record: tuple[int, ...]

    @property
    def row_slice(self) -> slice:
        return slice(self.row_start, self.row_stop)

    @property
    def gid_slice(self) -> slice:
        return slice(self.gid_start, self.gid_stop)


@dataclass(frozen=True)
class CodecFixture:
    """A tiny corpus plus exact stored-array and block-boundary expectations.

    ``corpus`` is the opened format-3 corpus
    (``format.r3g.reader.open_corpus``); every per-row array is in gid order.
    ``shape_xyz_i16`` is ``int16[rows, 3, 8]``: real atoms, then (merged) the
    derived pseudo-atoms, then zeros -- the kernel layout; ``heavy_xyz_i16``
    the stored heavy atoms alone. ``color_xyz_i16`` is always ``None`` (no
    split layout in format 3)."""

    root: Path
    corpus: object
    codec: str
    shuffle: bool
    color_layout: str
    block_conf: int
    global_ids: np.ndarray
    molecule_ids: np.ndarray
    conformer_indices: np.ndarray
    n_real: np.ndarray
    counts: np.ndarray
    color_counts: Optional[np.ndarray]
    shape_xyz_i16: np.ndarray
    color_xyz_i16: Optional[np.ndarray]
    merged_type_ids_i8: Optional[np.ndarray]
    physical_blocks: tuple[PhysicalBlock, ...]
    expected_decode_counts: dict[str, int]
    heavy_xyz_i16: Optional[np.ndarray] = None
    vaa: Optional[np.ndarray] = None
    color_self: Optional[np.ndarray] = None
    molecule_features: Optional[dict] = None

    def blocks_for(self, stream: str) -> tuple[PhysicalBlock, ...]:
        return tuple(b for b in self.physical_blocks if b.stream == stream)


#: The name the format-3 migration matrix uses for this surface.
FixtureCorpus = CodecFixture


def codec_support_reason(codec: Optional[str]) -> Optional[str]:
    """``None`` for the format-3 encoding, else why the codec is refused."""
    if codec in (None, ENCODING):
        return None
    return (f"codec {codec!r} does not exist in format 3: coordinates have one "
            f"encoding, {ENCODING!r} (FORMAT.md section 5, Appendix B.8)")


def iter_codec_layouts() -> Iterator[CodecLayout]:
    """The layouts format 3 has: one encoding, no shuffle, color merged or
    absent (FORMAT.md Appendix B.8)."""
    for color_layout in ("none", "merged"):
        yield CodecLayout(ENCODING, False, color_layout)


def _xyz_rows(gids: np.ndarray, n_real: np.ndarray, width: int) -> np.ndarray:
    xyz = np.zeros((gids.size, 3, width), dtype=np.int16)
    for row, (gid, count) in enumerate(zip(gids, n_real)):
        for axis in range(3):
            for atom in range(int(count)):
                xyz[row, axis, atom] = (int(gid) + 1) * 100 + axis * 10 + atom
    return xyz


def _color_descriptor() -> dict:
    from roshambo3.pharmacophore import PharmacophoreGenerator
    color = PharmacophoreGenerator().descriptor_color()
    if {t: i + 1 for i, t in enumerate(color["types"])} != FEATURES_ENUM:
        raise AssertionError("the default generator's numbering is not FEATURES_ENUM")
    return color


def make_codec_fixture(root: str | Path, *, codec: Optional[str] = None,
                       shuffle: Optional[bool] = None, color_layout: str = "none",
                       block_conf: int = 3,
                       empty_final_group: bool = False) -> CodecFixture:
    """Write the tiny two-group format-3 corpus described in the module
    docstring through the format-3 writer, verify it at ``scan`` and open it.

    ``codec`` accepts only ``"bp256-16"`` (or ``None``) and ``shuffle`` only
    ``False`` (or ``None``); ``color_layout`` is ``"merged"`` or ``"none"``;
    ``empty_final_group`` must be False. Anything else is refused with a
    ``ValueError`` naming why (FORMAT.md Appendix B.8): never mapped."""
    from roshambo3.format.r3g import reference as R
    from roshambo3.format.r3g.reader import open_corpus, verify
    from roshambo3.format.r3g.writer import Molecule, write_corpus

    if color_layout == "split":
        raise ValueError("color_layout='split' does not exist in format 3: a row's color "
                         "pseudo-atoms are derived and follow its real atoms (FORMAT.md "
                         "8.4, Appendix B.8); use 'merged' or 'none'")
    if color_layout not in {"none", "merged"}:
        raise ValueError("color_layout must be 'merged' or 'none'")
    reason = codec_support_reason(codec)
    if reason:
        raise ValueError(reason)
    if shuffle not in (None, False):
        raise ValueError("shuffle does not exist in format 3 (one encoding, FORMAT.md "
                         "section 5, Appendix B.8)")
    if empty_final_group:
        raise ValueError("an empty group is invalid in format 3 (FORMAT.md 9.3, "
                         "Appendix B.8)")
    block_conf = int(block_conf)
    if not 1 <= block_conf <= LIMITS["block_rows"]:
        raise ValueError(f"block_conf must be in [1, {LIMITS['block_rows']}]")

    color = color_layout == "merged"
    descriptor = _color_descriptor() if color else None
    n_confs = (block_conf, 1, 2)
    root = Path(root)
    n_real = np.concatenate([np.full(n, _n_real(smi), dtype=np.uint8)
                             for (_, smi, _), n in zip(MOLECULES, n_confs)])
    molecule_ids = np.concatenate([np.full(n, mid, dtype="<i8")
                                   for (mid, _, _), n in zip(MOLECULES, n_confs)])
    conformer_indices = np.concatenate([np.arange(n, dtype=np.uint16) for n in n_confs])
    n_total = int(sum(n_confs))
    global_ids = np.arange(n_total, dtype=np.int64)
    heavy = _xyz_rows(global_ids, n_real, 8)

    shape_xyz = heavy.copy()
    color_counts = np.zeros((n_total, 6), dtype=np.uint8) if color else None
    merged_types = np.zeros((n_total, 8), dtype=np.int8) if color else None
    vaa = np.empty(n_total, dtype=np.float32)
    color_self = np.zeros(n_total, dtype=np.float64) if color else None
    if color:
        r = np.asarray(descriptor["r"]["values"], dtype=np.float64)
        p = np.asarray(descriptor["p"]["values"], dtype=np.float64)
    mols = []
    features_of = {}
    row = 0
    for (mid, smiles, feats), n in zip(MOLECULES, n_confs):
        nr = _n_real(smiles)
        feats = list(feats) if color else []
        features_of[mid] = feats
        coords = heavy[row:row + n, :, :nr].copy()
        for k in range(n):
            # The row's own self-overlap from its stored (quantized) atoms
            # (FORMAT.md 18.1), and the derived pseudo-atoms' color
            # self-overlap under the corpus's matrices (18.2), float64.
            vaa[row + k] = np.float32(self_overlap(
                coords[k].T.astype(np.float64) * COORD_SCALE))
            if color:
                pos = R.derive_positions(coords[k], feats)
                shape_xyz[row + k, :, nr:nr + len(feats)] = pos.T
                types = np.asarray([t for t, _ in feats], dtype=np.int64)
                merged_types[row + k, nr:nr + len(feats)] = types
                color_counts[row + k] = np.bincount(types, minlength=7)[1:]
                b = pos.astype(np.float64) * COORD_SCALE
                color_self[row + k] = color_overlap(b, types, b, types, r, p)
        mols.append(Molecule(mid, coords, vaa[row:row + n].copy(), feats,
                             color_self[row:row + n].copy() if color else None,
                             ext_id=f"fixture-{mid}", smiles=smiles, mw=12.011 * nr,
                             rotors=0))
        row += n
    counts = (np.column_stack((n_real, color_counts)).astype(np.uint8)
              if color else n_real[:, None].copy())

    # Each caller still gets an independently writable corpus; the immutable
    # template is shared across workers and never exposed via hard links.
    import os
    if os.environ.get("ROSHAMBO3_TEST_CACHE"):
        import hashlib
        import shutil
        from tests.shared_cache import cached_directory
        def build(directory):
            write_corpus(str(directory / "corpus"), mols, color=descriptor,
                 prep_version="tests/codec_fixtures.py", prep_commit=None,
                 block_rows=block_conf, max_conf_per_group=block_conf + 1,
                 build={"source": "tests/codec_fixtures.py", "block_rows": block_conf,
                        "max_conf_per_group": block_conf + 1})
        template = cached_directory("codec-corpus", dict(block_conf=block_conf,
            color_layout=color_layout, helper=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), build)
        shutil.copytree(template / "corpus", root)
    else:
        write_corpus(str(root), mols, color=descriptor,
                 prep_version="tests/codec_fixtures.py", prep_commit=None,
                 block_rows=block_conf, max_conf_per_group=block_conf + 1,
                 build={"source": "tests/codec_fixtures.py", "block_rows": block_conf,
                        "max_conf_per_group": block_conf + 1})
    verify(str(root), "scan")
    corpus = open_corpus(str(root))

    physical_blocks = []
    group_bases = [0, n_confs[0] + n_confs[1]]
    for gi, entry in enumerate(corpus.group_table):
        with corpus.group(gi) as g:
            for bi, b in enumerate(g.blocks):
                r0, n = int(b["first_row"]), int(b["rows"])
                physical_blocks.append(PhysicalBlock(
                    stream="shape", group_index=gi, block_index=bi,
                    row_start=r0, row_stop=r0 + n,
                    gid_start=group_bases[gi] + r0, gid_stop=group_bases[gi] + r0 + n,
                    path=root / "groups" / group_file_name(entry["bucket"], entry["group"]),
                    record=tuple(int(b[k]) for k in ("first_row", "rows", "offset",
                                                     "length", "m0", "m1"))))
    if [b.row_stop - b.row_start for b in physical_blocks] != [block_conf, 1, 2]:
        raise AssertionError(f"unexpected block rows {physical_blocks}")
    return CodecFixture(
        root=root, corpus=corpus, codec=ENCODING, shuffle=False,
        color_layout=color_layout, block_conf=block_conf,
        global_ids=global_ids, molecule_ids=molecule_ids,
        conformer_indices=conformer_indices, n_real=n_real, counts=counts,
        color_counts=color_counts, shape_xyz_i16=shape_xyz, color_xyz_i16=None,
        merged_type_ids_i8=merged_types, physical_blocks=tuple(physical_blocks),
        expected_decode_counts={"shape": len(physical_blocks)},
        heavy_xyz_i16=heavy, vaa=vaa, color_self=color_self,
        molecule_features=features_of)


def truncate_block_record_copy(block: PhysicalBlock, destination: str | Path,
                               *, remove_bytes: int = 1) -> Path:
    """Write a private copy of ``block``'s group file whose block-table entry
    advertises ``remove_bytes`` fewer bytes, so a reader that trusts the
    table hands the decoder a short buffer.

    The copy is self-consistent inside the file (the ``blocks`` section hash
    in the header is updated) but no longer matches the corpus's group table,
    and its blocks no longer tile the ``coords`` section: ``open`` refuses it
    (``layout.blocks``), and a decoder given the short bytes must refuse them
    (``in_len != block_length``, FORMAT.md 17 item 3). The source file is
    never changed."""
    import hashlib
    from roshambo3.format.r3g import jsonstrict
    if remove_bytes < 1:
        raise ValueError("remove_bytes must be positive")
    source = Path(block.path)
    destination = Path(destination)
    if destination.resolve() == source.resolve():
        raise ValueError("truncated fixture destination must be a separate file")
    if destination.exists():
        raise FileExistsError(destination)
    data = bytearray(source.read_bytes())
    if len(data) < FIRST_SECTION_OFFSET:
        raise ValueError(f"{source} has no complete format-3 prologue")
    header_offset = struct.unpack_from("<Q", data, 8)[0]
    header = json.loads(bytes(data[header_offset:]).decode())
    entry = next(s for s in header["sections"] if s["name"] == "blocks")
    blocks = np.frombuffer(bytes(data[entry["offset"]:entry["offset"] + entry["length"]]),
                           dtype=BLOCK_DTYPE).copy()
    length = int(blocks[block.block_index]["length"])
    if length != block.record[3]:
        raise ValueError("block metadata no longer matches the fixture record")
    if length <= remove_bytes:
        raise ValueError("cannot truncate the block by that many bytes")
    blocks[block.block_index]["length"] = length - remove_bytes
    raw = blocks.tobytes()
    data[entry["offset"]:entry["offset"] + entry["length"]] = raw
    entry["sha256"] = hashlib.sha256(raw).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(data[:header_offset]) + jsonstrict.dumps(header))
    return destination

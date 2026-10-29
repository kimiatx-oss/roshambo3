"""Constants, limits, the section schema and the refusal bounds of format 3.

Everything here restates ``doc/FORMAT.md`` (sections 2, 3.3, 14) in one place
so a spec amendment lands in one edit:

* :data:`LIMITS` -- the limits table (section 14.1);
* :func:`expected_sections` -- which sections a group file carries, in order,
  with their dtypes (section 3.3), as a function of what the descriptor says;
* :class:`Bounds` -- the time and memory bounds of limits 33-34. They are
  configuration, not constants: the spec revises them, and a caller (a test,
  a slow CI host) may pass its own.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

MAGIC = b"R3G\0"
MAJOR = 3
STEP_LOG2 = -8
ENCODING = "bp256-16"
BUCKET_EDGES = (0, 12, 16, 20, 24, 28, 32, 38, 44, 50, 56, 64, 72, 256)
N_BUCKETS = len(BUCKET_EDGES) - 1
L_ALIGN = 8
PROLOGUE_SIZE = 16
FIRST_SECTION_OFFSET = 64
BLOCK_HEADER_SIZE = 32
BLOCK_RECORD_SIZE = 64
RUN_RECORD_SIZE = 16
INT64_MAX = (1 << 63) - 1
SIGNATURE_FILE = "corpus.json.sig"

LIMITS = {
    "json_bytes": 16 * 1024 * 1024,
    "json_depth": 32,
    "json_string": 64 * 1024,
    "groups": 1 << 16,
    "blocks": 1 << 20,
    "n_conf": 1 << 24,
    "n_real": 255,
    "n_conf_per_molecule": 4096,
    "L_heavy": 256,
    "L": 512,
    "L_field_width": 256,
    "n_types": 6,
    "features_per_type": 255,
    "features_per_molecule": 255,
    "atoms_per_feature": 16,
    "block_rows": 4096,
    "block_rows_single_molecule": 4096,
    "bit_width": 16,
    "n_feat_atoms": (1 << 32) - 1,
    "parquet_row_group": 1 << 14,          # rows per row group (S4; 2^16 before)
    "parquet_footer_base": 64 * 1024,      # footer bytes before parsing (S3)
    "parquet_footer_per_row_group": 1024,
    "parquet_row_bytes": 4608,
    "parquet_overhead": 1 << 20,
    "smiles": 4096,
    "ext_id": 256,
    "mol_id": 1 << 63,
}

#: dtype name -> numpy dtype (closed enumeration, section 3.3).
DTYPES = {
    "u8": np.dtype("u1"), "u16": np.dtype("<u2"), "u32": np.dtype("<u4"),
    "u64": np.dtype("<u8"), "i16": np.dtype("<i2"), "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
}

BLOCK_DTYPE = np.dtype([("first_row", "<u4"), ("rows", "<u4"), ("offset", "<u8"),
                        ("length", "<u8"), ("m0", "<u4"), ("m1", "<u4"),
                        ("sha256", "u1", 32)])
RUN_DTYPE = np.dtype([("mol_id", "<u8"), ("n_conf", "<u2"), ("n_real", "u1"),
                      ("reserved", "u1", 5)])
assert BLOCK_DTYPE.itemsize == BLOCK_RECORD_SIZE and RUN_DTYPE.itemsize == RUN_RECORD_SIZE

CONVENTIONS = {
    "hydrogens": "RemoveAllHs",
    "features_perceived": "with_hydrogens",
    "atom_order": "canonical_smiles",
    "protonation": "as_given",
    "standardization": "none",
    "conformers": "as_given",
}
FRAME = {"center": "heavy_atom_centroid", "axes": "principal_axes", "handedness": "right"}
FORMAT_OBJECT = {"major": MAJOR, "encoding": ENCODING, "step_log2": STEP_LOG2}
GENERATORS = ("rdkit_fdef", "smarts_list")
#: ``color.positioning`` values this implementation accepts (section 10.2).
POSITIONING = ("centroid",)

#: Every section name the format defines, with its dtype and whether it is a
#: color section (section 3.3). The order is file order.
SECTION_SCHEMA = (
    ("coords", "u8", False),
    ("blocks", "u8", False),
    ("runs", "u8", False),
    ("feature_offsets", "u32", True),
    ("feature_types", "u8", True),
    ("feature_atom_offsets", "u32", True),
    ("feature_atoms", "u8", True),
    ("vaa", "f32", False),
    ("color_self", "f64", True),
)
SECTION_DTYPE = {name: dt for name, dt, _ in SECTION_SCHEMA}
#: Sections that may be absent although their group is a color group.
OPTIONAL_SECTIONS = frozenset({"color_self"})
#: The fixed-width sections (2-9) that limit 34's memory bound scales with.
FIXED_SECTIONS = tuple(name for name, _, _ in SECTION_SCHEMA if name != "coords")


def expected_sections(n_types: int, *, positioning: Optional[str] = None) -> list[str]:
    """Section names a group with ``n_types`` color types carries, in file
    order, optional ones included (section 3.3). ``positioning`` is the
    descriptor's ``color.positioning``: the hook for a positioning rule that
    adds a section."""
    return [name for name, _, color in SECTION_SCHEMA if not color or n_types > 0]


@dataclasses.dataclass(frozen=True)
class Bounds:
    """Limits 33-34: refusal time and validator memory, per corpus file.

    ``open_seconds`` bounds the ``open`` tier on any input (and every
    adversarial seed at its tier); ``scan`` and ``deep`` get
    ``per_file_seconds + section_bytes / min_bytes_per_second``. Memory:
    ``open`` stays within ``rss_base_bytes`` above the interpreter, ``scan``
    within ``rss_base_bytes + rss_section_factor * fixed-section bytes``.
    """
    open_seconds: float = 20.0
    per_file_seconds: float = 20.0
    min_bytes_per_second: float = 1024 * 1024
    rss_base_bytes: int = 256 * 1024 * 1024
    rss_section_factor: int = 16

    def scan_seconds(self, section_bytes: int) -> float:
        return self.per_file_seconds + section_bytes / self.min_bytes_per_second

    def scan_rss(self, fixed_section_bytes: int) -> int:
        return self.rss_base_bytes + self.rss_section_factor * fixed_section_bytes


#: The bounds ``doc/FORMAT.md`` states (limits 33-34, as relaxed by Appendix B
#: S1: 20 s, 1 MiB/s + 20 s per file).
DEFAULT_BOUNDS = Bounds()


def pad8(n: int) -> int:
    """``max(8, ceil(n / 8) * 8)`` (section 2)."""
    return max(L_ALIGN, -(-int(n) // L_ALIGN) * L_ALIGN)


def roundup64(x: int) -> int:
    return -(-int(x) // 64) * 64


def bucket_of(n_real: int) -> int:
    """Bucket index of a heavy-atom count (section 9.1)."""
    n = int(n_real)
    if not 1 <= n <= 255:
        raise ValueError(f"n_real {n} outside [1, 255]")
    for b in range(N_BUCKETS):
        if BUCKET_EDGES[b] <= n < BUCKET_EDGES[b + 1]:
            return b
    raise AssertionError("unreachable")


def group_file_name(bucket: int, group: int) -> str:
    """``b{bucket:02d}_g{group:04d}.r3g`` (section 1), the name only."""
    return f"b{int(bucket):02d}_g{int(group):04d}.r3g"


def block_length(total: int, w_x: int, w_y: int, w_z: int) -> int:
    """``32 + 32 * G * (w_x + w_y + w_z)``, ``G = ceil(total / 256)`` (section 5.1)."""
    return BLOCK_HEADER_SIZE + 32 * (-(-int(total) // 256)) * (int(w_x) + int(w_y) + int(w_z))

# ROSHAMBO3 corpus format 3 (`.r3g`)

The paper repository (https://github.com/kimiatx-oss/roshambo3-paper) publishes an excerpt of the development record
(`tickets/`, `decisions/`) covering this format.

**Status:** normative, major version 3. The reference
implementation is `src/roshambo3/format/r3g/reference.py` (pure Python and
NumPy; the oracle, not the production path), and the conformance corpus is
`tests/data/format3/`. An implementation is correct when it reproduces
`tests/data/format3/EXPECTED.json` (its digests are over the arrays defined
in `tests/data/format3/README.md`: the padded `int16[rows, 3, L]` layouts of
section 4.3, the per-row pseudo-atoms of section 8.4 concatenated over rows,
the count vectors of section 8.5, and the raw bytes of the stored sections)
and refuses every file under `tests/data/format3/adversarial/` as
`adversarial/EXPECTED.json` says.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are to be read as in
RFC 2119. Every binary integer in this document is **little-endian**, and
every signed one (`int16`) is **two's complement**; this holds for every field
of every table below without being repeated there. JSON numbers are decimal.
Byte offsets and sizes are in bytes. `ceil`, `floor` and `//` are integer
operations (`ceil(n / 8)` is ceiling division). `A` is Angstrom. Two words
carry a second meaning that the context makes clear: a *group* is a group file
(sections 3, 9.3) and, inside section 5, a 256-value packing group; a *run* is
one molecule's consecutive rows (sections 6, 7).

Sections 1-13 define the files. Sections 14-18 define the limits, the
validator, the decoder contract and the versioning policy. Appendix A is a
worked byte-level example; Appendix B records the design choices this
document makes, with the reason for each, and the security hardening rules
S1-S9.

---

## 1. The corpus directory

A corpus is a directory:

```
<corpus>/
  corpus.json              the descriptor (section 10)      REQUIRED
  molecules.parquet        the molecule table (section 11)  REQUIRED
  build.json               provenance (section 12)          OPTIONAL
  corpus.json.sig          reserved (section 12.2)          IGNORED
  groups/
    b00_g0000.r3g          one group file per group (section 3)
    b00_g0001.r3g
    b01_g0000.r3g
    ...
```

1. The group file name is **derived by the reader** from the group table in
   `corpus.json` (section 10.4): the group with bucket `b` and group index `g`
   is at `groups/b{b:02d}_g{g:04d}.r3g`, where `{x:02d}` and `{x:04d}` are the
   decimal index zero-padded to at least 2 and at least 4 digits respectively
   (an index of 10,000 or more is written with as many digits as it needs).
   The descriptor carries no file name and no path; a descriptor that carries
   one is invalid (section 10.4, closed schema).
2. **Opening relative to directory descriptors.** A reader opens the corpus
   root -- the path the user named -- once, with `open(root, O_RDONLY |
   O_DIRECTORY)` (a symlink in or at the end of that path is followed: it is
   the user's choice, and corpora are shared by symlink). It opens `groups/`
   **relative to the root descriptor**, `openat(root_fd, "groups", O_RDONLY |
   O_DIRECTORY | O_NOFOLLOW)`, and every corpus file relative to its
   directory's descriptor: `openat(root_fd, "corpus.json", ...)`,
   `openat(root_fd, "molecules.parquet", ...)`, `openat(groups_fd,
   "b00_g0000.r3g", ...)`, each with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK`. Every
   name passed to `openat` is a single path component derived by the reader
   (item 1, or one of the fixed names above), never a path and never a string
   read from a file. So no component under the root is ever resolved through
   a symlink (a symlinked `groups/` or group file fails with `ELOOP` or
   `ENOTDIR`), and there is no containment check to race: a
   `realpath`-then-`open` check guards only the last component and is not
   sufficient. The reader then `fstat`s the descriptor and refuses anything
   that is not `S_ISREG` (a FIFO, a socket, a device, a directory), and reads
   from **that descriptor** only -- never reopening by path. `O_NONBLOCK` is
   what keeps a FIFO from blocking the `open()` call itself; on a regular
   file it is cleared again after `fstat`. A standalone group file (section
   3.3) is opened by the path the user named, with the same flags. Nothing
   in any file of a corpus is ever used as a file name, a module name, a
   plugin name or a command.
3. Any other file in the corpus directory is ignored by readers. `verify` MAY
   report their presence; it MUST NOT read them.
4. **Reads are bounded `pread`s, not memory maps.** A reader reads a corpus
   file with `pread` (or `read`) into buffers it owns, sized from values
   that have already passed the limits (section 14.2), and treats a short
   read -- the file shrank after `fstat` -- as a refusal (`io.short_read`).
   A reader MUST NOT `mmap` a corpus file unless it also converts the
   `SIGBUS` that an access past the end of a file truncated after mapping
   raises into an error, for the whole life of the mapping (a `SIGBUS`
   handler that `siglongjmp`s out of a guarded access, or an equivalent); a
   process killed by `SIGBUS` is a crash, which section 16 forbids. Reason
   and cost: page cache warm, single thread, `pread` in 64 KiB pieces into a
   reused buffer moves 10.9 GiB/s against 10.0 GiB/s for copying out of a
   fresh mapping, so the rule costs a producer nothing it cannot recover by reading in
   bounded pieces (single 1-4 MiB `pread`s ran at 3.4 GiB/s).

## 2. Fixed constants

| Constant | Value | Where it is stated |
|---|---|---|
| Magic bytes | `52 33 47 00` (`"R3G\0"`) | section 3.1 |
| Major version | `3` | section 3.1, section 13 |
| Coordinate step | 2^-8 A = 0.00390625 A (`format.step_log2 = -8`) | section 4 |
| Coordinate encoding | `"bp256-16"` (the only value) | section 5 |
| Bucket edges on `n_real` | `[0, 12, 16, 20, 24, 28, 32, 38, 44, 50, 56, 64, 72, 256]` | section 9.1 |
| Width alignment `L_ALIGN` | 8 | section 8.6 |
| Hash | SHA-256, written as 64 lowercase hexadecimal characters | section 3.5 |
| Block table record size | 64 bytes | section 6 |
| Run table record size | 16 bytes | section 7 |

`pad8(n)` means `max(8, ceil(n / 8) * 8)` throughout.

## 3. The group file

A group file holds every conformer of one group: the bit-packed heavy-atom
coordinates, the block table, the molecule run table, the color feature table
(color corpora), the shape self-overlaps and the color self-overlaps (color
corpora with a canonical interaction matrix). It is self-contained: it carries
the corpus fingerprint (section 10.5), so a group file separated from its
directory still names what it is.

### 3.1 Prologue

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | magic | MUST be `52 33 47 00` |
| 4 | 4 | major version, u32 | MUST be `3`; a reader MUST refuse any other value (section 13) |
| 8 | 8 | `header_offset`, u64 | absolute offset of the JSON header |
| 16 | 48 | reserved | MUST be zero |

The reserved bytes exist so that the first section starts at offset 64. A
reader MUST check `64 <= header_offset <= file_size` and
`file_size - header_offset <= 16 MiB` (section 14) **before** reading the
header, and MUST read the header with a bounded read of exactly
`file_size - header_offset` bytes, never "read to end of file".

### 3.2 Sections and their extents

The bytes `[64, header_offset)` hold the sections, in the order of section
3.3, each described by one entry of the header's section table (section 3.4).
For every section:

* `offset` is an absolute file offset and MUST be a multiple of 64.
* `length` MUST equal `prod(shape) * itemsize(dtype)`, as Python integers.
  (For the `coords` section the shape is `[length]` with dtype `u8`, so the
  rule holds trivially; its real size constraint is section 6.)
* The section's *extent* is `[offset, roundup64(offset + length))`, where
  `roundup64(x) = ceil(x / 64) * 64`.
* The first section's offset MUST be 64; every later section's offset MUST
  equal the previous section's extent end; `header_offset` MUST equal the last
  section's extent end. The extents therefore tile `[64, header_offset)` with
  no gap and no overlap, and the header starts on a 64-byte boundary.
* Every byte of an extent beyond `offset + length` (the alignment padding)
  MUST be zero. Writers MUST write zeros there; `deep` checks it.
* A section MAY have `length == 0` only when its shape has a zero dimension.
  Only `feature_types` and `feature_atoms` (a color corpus with no features in
  the group) can legally have that.
* The section's `sha256` is over its `length` data bytes, not its padding.

### 3.3 The sections

In file order. "Color" sections exist in every group of a color corpus and in
no group of a shape-only corpus; their presence is decided by `corpus.json`
alone (the `color` key, section 10.2), never by the group file.

| # | name | dtype | shape | present | content |
|---|---|---|---|---|---|
| 1 | `coords` | `u8` | `[n_bytes]` | always | the concatenated BP256-16 blocks (section 5) |
| 2 | `blocks` | `u8` | `[n_blocks, 64]` | always | the block table (section 6) |
| 3 | `runs` | `u8` | `[n_mol, 16]` | always | the molecule run table (section 7) |
| 4 | `feature_offsets` | `u32` | `[n_mol + 1]` | color | per-molecule feature index ranges (section 8.1) |
| 5 | `feature_types` | `u8` | `[n_feat]` | color | the type of each feature (section 8.1) |
| 6 | `feature_atom_offsets` | `u32` | `[n_feat + 1]` | color | per-feature atom index ranges (section 8.1) |
| 7 | `feature_atoms` | `u8` | `[n_feat_atoms]` | color | heavy-atom indices (section 8.1) |
| 8 | `vaa` | `f32` | `[n_conf]` | always | the shape self-overlap `V_AA` of each row (section 18.1) |
| 9 | `color_self` | `f64` | `[n_conf]` | color, **optional** | the color self-overlap of each row (section 18.2) |

Section 9 is optional. It MAY be present only in a color corpus whose
interaction matrices are canonical (section 18.2); the validator refuses it
otherwise, and a reader whose corpus lacks it recomputes. The section table
MUST list exactly the sections this table says are present, in this order,
each once. Section names are a closed enumeration: any other name is invalid.

A reader holding only a group file (no `corpus.json`) takes color presence
from the header's `n_types` (`> 0` means the color sections are present) and
`color_self` presence from the section table; the checks that need the
descriptor -- the fingerprint, `header_sha256`, the group-table fields and
the canonical-matrix condition on `color_self` -- are made when it is
available (section 16.2), and a reader that will *screen* always has it.

dtype is a closed enumeration: `u8`, `u16`, `u32`, `u64`, `i16`, `f32`, `f64`,
all little-endian; `itemsize` is 1, 2, 4, 8, 2, 4, 8 respectively. Each
section's dtype and shape MUST be exactly what this table says (`n_blocks`,
`n_mol`, `n_feat`, `n_feat_atoms` and `n_conf` are whatever the file holds,
subject to section 14).

### 3.4 The header

The header is the bytes `[header_offset, file_size)`. It MUST be a single JSON
object in UTF-8 obeying the strict JSON rules of section 16.1 (no duplicate
keys, no `NaN`/`Infinity`, depth and string limits); an empty header
(`header_offset == file_size`) is refused as `json.syntax`. Writers MUST emit
it in canonical JSON (section 10.5.1: sorted keys, no whitespace) and readers
MAY check that it is (comparing the bytes with the canonical re-serialization
is an `open`-tier check and is performed by the reference). The schema is
closed and **every key below is required**, including every key of every
section entry: a key not listed here, or a listed key absent, makes the file
invalid. The example is shown with whitespace and in reading order for
legibility only; the stored bytes are canonical.

```json
{
  "L": 24,
  "L_heavy": 16,
  "bucket": 1,
  "fingerprint": "7b1c...e9",
  "group": 0,
  "major": 3,
  "n_conf": 123,
  "n_mol": 17,
  "n_types": 6,
  "sections": [
    {"name": "coords", "offset": 64, "length": 5120, "dtype": "u8", "shape": [5120], "sha256": "..."},
    {"name": "blocks", "offset": 5184, "length": 64, "dtype": "u8", "shape": [1, 64], "sha256": "..."},
    "..."
  ]
}
```

| key | type | rule |
|---|---|---|
| `major` | int | MUST equal the prologue's major version (3) |
| `fingerprint` | 64 hex | MUST equal the corpus fingerprint (section 10.5) when the descriptor is available |
| `bucket` | int | the bucket index (section 9.1); MUST match the file name and the group table |
| `group` | int | the group index within the bucket; MUST match the file name and the group table |
| `n_conf` | int | number of rows; MUST equal the sum of `n_conf` over the run table, the sum of `rows` over the block table, and the group table's `n_conf` |
| `n_mol` | int | number of molecules; MUST equal the run table's row count and the group table's `n_mol` |
| `n_types` | int | number of color types: `len(color.types)` of the descriptor (1..6), or 0 for a shape-only corpus |
| `L_heavy` | int | `pad8(max n_real over the group)` (section 8.6) |
| `L` | int | the device width, `pad8(max (n_real + n_color) over the group)` (section 8.6) |
| `sections` | list | one object per section, in file order: `name`, `offset`, `length`, `dtype`, `shape` (list of ints), `sha256` (64 hex); closed |

Integers are JSON integers (no fraction, no exponent, not booleans).

### 3.5 Hashes and the integrity chain

Every hash in this format is SHA-256 (`hashlib.sha256`), hex-encoded in
lowercase. Reason: it is in every Python standard library and it is
hardware-accelerated on every deployment target; one algorithm everywhere
keeps the reference reader dependency-free (Appendix B.2).

The chain, from the top:

1. `corpus.json` records each group's `header_sha256`: SHA-256 of the header
   bytes `[header_offset, file_size)` exactly as stored.
2. The header records each section's `sha256` over its `length` data bytes.
3. The `blocks` section records each block's SHA-256 over its encoded bytes
   -- the whole block, its 32-byte header included (section 6) -- so a block
   fetched by byte range can be verified on its own once the header has been
   verified against `corpus.json`.

A detached signature over `corpus.json` is reserved and deferred (section
12.2). `open` verifies `header_sha256` **before it uses any header value**,
then the section table, then the `blocks` section's hash and the block-table
walk, then the fingerprint (all O(header) plus 64 bytes per block); `deep` verifies the rest
of the chain (section 16.4). `corpus.json` also records `molecules_sha256`
(section 10.7) so that a future signature over `corpus.json` covers
`molecules.parquet` as well.

## 4. Coordinates

### 4.1 What is stored

A row is one conformer. Its stored coordinates are its **heavy atoms only**
(every hydrogen removed, isotopes included: `prep/conformer.py::strip_hydrogens`,
`Chem.RemoveAllHs`), as `int16` in units of the step, in the conformer's own
frame:

* **Frame.** Centered on the heavy-atom centroid and rotated into the
  heavy-atom principal-axis frame: the rows of the rotation are the
  eigenvectors of the heavy-atom coordinate covariance in descending order
  of variance, with the determinant forced to +1 by negating the last axis
  when it is -1 (right-handed), as
  `src/roshambo3/prep/conformer.py::prep_conformer` computes it
  (`principal_axis_rotation`, with `make_right_handed`; the sign convention
  of the first two axes is the SVD's, which is why the frame is a builder
  property and a reader never recomputes it). The SVD's sign choice depends
  on the BLAS/LAPACK build and on the CPU kernel it dispatches, so the same
  conformer can be stored rotated by pi about one axis on two hosts (section
  13 item 3). Only heavy atoms define the frame.
* **Step.** `2^-8 A`. A stored value `v` is the coordinate `v * 2^-8 A`; the
  range is `[-128 A, +127.996 A]`. A prepared coordinate outside that range
  MUST be refused at build (it cannot be stored). A reader dequantizes as
  `float32(v) * float32(2^-8)`, which is exact.
* **Rounding.** `v = rint(x / 2^-8) = rint(x * 256)` with **round half to
  even** (`numpy.rint`) in float64, as `format/r3g/reference.py::quantize` does.
  `x * 256` is exact in float64, so there is no division rounding before the
  `rint`. Non-finite coordinates MUST be refused at build.
* **The int16 grid is the one geometry.** The self-overlaps
  (section 18), the derived color positions (section 8.3) and the snapped
  query are computed from the stored integers, never from the float
  coordinates the builder started from. No float coordinate is stored.

### 4.2 Atom order within a conformer

Atom `i` of a row is atom `i` of `Chem.MolFromSmiles(smiles)` for that
molecule's `smiles` in `molecules.parquet` (section 11): the builder reorders
heavy atoms into the **canonical SMILES output order** (RDKit's
`_smilesAtomOutputOrder` of the canonical SMILES of the hydrogen-stripped
molecule). Every conformer of a molecule has the same heavy-atom count
`n_real` and the same atom order (up to symmetry-equivalent atoms, a graph
automorphism; records that cannot be aligned to the first record, such as a
deuterated record in another atom order, are canonicalized on their own). Canonical SMILES order is a property of an
RDKit release: byte identity of a rebuild holds within one RDKit release, not
necessarily across releases (section 13.3).

### 4.3 Padding

Padding is not stored. A decoder writing a padded layout of width `L` fills
slots `[n_real, L)` (or `[n_real + n_color, L)` in the device layout, section
8.4) with zeros. The **padded per-row layout** this document and the
conformance digests use is `int16[rows, 3, L]`: per row the `L` x values,
then the `L` y values, then the `L` z values (the format-3 per-row order: x, then y, then z);
the device tile layouts and the CPU float32 layout are derived from it by
the consumers (the native codec) and are not defined here.

## 5. The coordinate encoding: BP256-16

The `coords` section is the concatenation of the group's **blocks** (section
6), each a self-contained encoding of a run of consecutive rows. There is no
general-purpose compressor and no other encoding: `format.encoding` is the
closed enumeration `{"bp256-16"}`, and a different encoding is a different
major version (section 13).

### 5.1 Block layout

```
offset  size  field
0       2     min_x   int16
2       2     min_y   int16
4       2     min_z   int16
6       1     w_x     u8, in [0, 16]
7       1     w_y     u8, in [0, 16]
8       1     w_z     u8, in [0, 16]
9       23    reserved, MUST be zero
32      Sx    axis stream x
32+Sx   Sy    axis stream y
32+Sx+Sy Sz   axis stream z
```

with, for `total` the number of real atoms in the block (section 5.2) and
`G = ceil(total / 256)` the number of 256-value groups,

```
Sa = G * w_a * 32           bytes, for a in {x, y, z}
block_length = 32 + 32 * G * (w_x + w_y + w_z)
```

`block_length` is a pure function of the block header and of `total`; `total`
is computed from the **run table** over the block's molecule range `[m0, m1)`
(section 5.2; the range comes from the block table, a stored total does not
exist and MUST NOT be invented from any other byte). A decoder MUST check that the
bytes it is handed are exactly `block_length` long (section 17), and the block
table's `length` MUST equal it (section 6). Because the header is 32 bytes and
every stream is a multiple of 32 bytes, every block and every axis stream
starts at a 32-byte-aligned offset within the `coords` section.

### 5.2 The axis stream

The block covers rows `[first_row, first_row + rows)` of the group, which are
the conformers of molecules `[m0, m1)` of the run table, in order. The stream
of axis `a` is the sequence of the stored int16 values of coordinate `a` of
**every real atom of every row of the block, in row order and atom order**:

```
values_a = [ coord[row][a][i]  for row in block rows (ascending)
                                for i in range(n_real(row)) ]
total    = len(values_a) = sum over m in [m0, m1) of n_conf[m] * n_real[m]
```

Padding slots are not in the stream. `total >= 1` always (a block has at least
one row and every row has `n_real >= 1`).

Every value is stored as the unsigned difference `u = v - min_a`, in `w_a`
bits; the encoder (section 5.5) guarantees `0 <= u < 2^w_a`.

### 5.3 Group packing (the "vertical" layout)

The stream is cut into groups of 256 values. Value index `j` (0-based within
the stream) belongs to group `g = j // 256` at in-group index `r = j % 256`,
and sits in **lane** `l = r % 16` at **position** `k = r // 16`:

```
j = g*256 + 16*k + l         g in [0, G),  k in [0, 16),  l in [0, 16)
```

Each packing group is stored as `w_a` consecutive 256-bit words; word `i`
(`0 <= i < w_a`) is 32 bytes holding one little-endian u16 per lane: lane `l`'s
u16 of word `i` is at byte offset

```
group_base = g * w_a * 32
byte(g, i, l) = group_base + 32*i + 2*l          (2 bytes, little-endian)
```

within the axis stream. The `w_a` u16 values of one lane (its u16 in word 0,
word 1, ..., word `w_a - 1`) form a `16 * w_a`-bit lane bitstream: bit `b` of
the lane bitstream is bit `b % 16` of the lane's u16 in word `b // 16`. The
value at position `k` occupies lane bits `[k*w_a, (k+1)*w_a)` with its least
significant bit at lane bit `k*w_a`: lane bit `k*w_a + bit` holds bit `bit`
of `u`. The three axes are three independent streams, never interleaved.

Equivalently, in scalar pseudo-code (this is what the reference does and what
section 17 calls the scalar reference decoder):

```
def unpack_axis(stream_bytes, w, mn, total):
    out = []
    G = ceil(total / 256)
    for g in range(G):
        for k in range(16):
            for l in range(16):
                u = 0
                for bit in range(w):
                    b = k*w + bit
                    word = u16_at(stream_bytes, g*w*32 + 32*(b // 16) + 2*l)
                    u |= ((word >> (b % 16)) & 1) << bit
                out.append(int16_wrap(mn + u))     # see 5.6
    return out[:total]
```

`u16_at(bytes, offset)` reads the little-endian u16 at `offset`;
`int16_wrap(x)` is `((x + 32768) mod 65536) - 32768` (section 5.6).

### 5.4 Tail group, `w = 0`, `w = 16`

* **Tail.** When `total` is not a multiple of 256, the last group has
  `256*G - total` unused positions (`j >= total`). Encoders MUST store `u = 0`
  there (so the bytes of a block are a function of its content), and decoders
  MUST ignore them (they produce no output; a decoder that unpacks whole
  groups into scratch discards the tail). `deep` checks that they are zero.
* **`w = 0`.** The stream has zero bytes; every value of the axis equals
  `min_a`. This happens whenever the axis is constant over the block (a
  single-atom molecule, a block of linear or planar molecules whose third
  axis is identically zero).
* **`w = 16`.** Each value occupies exactly one u16: lane `l`, position `k` is
  the u16 at byte `g*512 + 32*k + 2*l`. `u` may be any value in `[0, 65535]`.

### 5.5 Encoder requirements (byte identity)

For every axis of every block an encoder MUST use

```
min_a = min(values_a)
w_a   = bit_length(max(values_a) - min_a)      # 0 when all values are equal
```

(`bit_length` of a non-negative integer: 0 for 0, 1 for 1, 2 for 2-3, ...;
`max - min <= 65535` so `w_a <= 16`), MUST write `u = v - min_a` for every
value and `u = 0` for every tail position, and MUST write zeros in the 23
reserved header bytes. Two encoders given the same rows therefore produce the
same bytes, which is what makes the block hash and the rebuild byte-identity
(section 13 item 3) meaningful. A decoder MUST NOT rely on `min_a` or `w_a` being
minimal.

### 5.6 Decoded value and overflow

The decoded int16 value is `int16_wrap(min_a + u)`: the sum reduced modulo
2^16 into two's-complement int16 (what a 16-bit SIMD add does). In a valid
file `min_a + u` lies in `[-32768, 32767]` by construction (`u <= max - min`)
and the wrap is the identity; on a tampered block a decoder MUST still
produce the wrapped value -- not saturate, not error -- so that every decoder
path is bit-identical on every input (section 17 item 9), and it is not
required to detect the out-of-range case, because a coordinate is never used
as an index or a size: a tampered block yields wrong coordinates, never an
out-of-bounds access. Everything derived from a decoded coordinate (section
8.3, section 18) is computed from the wrapped int16.

### 5.7 Per-block widths are the encoding

The widths are per (block, axis). A per-conformer or per-molecule width scheme
is a different byte layout and therefore a different major version, not a
flag (section 13).

## 6. The block table

The `blocks` section is `u8[n_blocks, 64]`; record `i` describes block `i`:

| offset | size | field | rule |
|---|---|---|---|
| 0 | 4 | `first_row` u32 | row index of the block's first row within the group |
| 4 | 4 | `rows` u32 | number of rows; `>= 1` |
| 8 | 8 | `offset` u64 | byte offset of the block **relative to the start of the `coords` section's data** |
| 16 | 8 | `length` u64 | byte length of the block; MUST equal `block_length` of section 5.1 |
| 24 | 4 | `m0` u32 | index of the block's first molecule in the run table |
| 28 | 4 | `m1` u32 | one past its last molecule; `m1 > m0` |
| 32 | 32 | `sha256` | SHA-256 of the block's `length` bytes, raw (not hex) |

Rules:

1. **Blocks are molecule-aligned.** Rows `[first_row, first_row + rows)` are
   exactly the rows of molecules `[m0, m1)`: `first_row` equals the sum of
   `n_conf` over molecules `[0, m0)` and `rows` equals the sum over `[m0, m1)`.
   A molecule's run is never split across blocks; a molecule therefore always
   ends exactly at a block edge, and that is the only place a block may end.
2. **Blocks tile the group.** Block 0 has `first_row = 0` and `m0 = 0`; block
   `i + 1` has `first_row = first_row_i + rows_i` and `m0 = m1_i`; the last
   block has `first_row + rows == n_conf` and `m1 == n_mol`. `n_blocks >= 1`.
3. **Blocks tile the payload.** Block 0 has `offset = 0`; block `i + 1` has
   `offset = offset_i + length_i`; the last block ends at the `coords`
   section's `length`. Every `offset` is then a multiple of 32.
4. **Row cap.** Every block MUST have `1 <= rows <= 4096`, including a
   block holding one molecule. Every molecule has at most 4,096 conformers
   (section 7). Decoded heavy coordinates occupy at most
   `4096 x 256 x 6 = 6,291,456 B`; device coordinates occupy at most
   `4096 x 512 x 6 = 12,582,912 B` (12.6 MB). Device rows are padded to a
   multiple of 32, still at most 4,096; size the pinned pool with
   `native.gpu_layout_elements(rows, L)`. CPU float32 coordinates occupy at
   most `4096 x 512 x 12 = 25,165,824 B` (25.2 MB).
5. **Write rule** (what a writer MUST do, parameterized by `block_rows`,
   `1 <= block_rows <= 4096`, default 4096, recorded in `build.json`):
   walk the group's molecules in order; append a molecule to the current block
   if the block is empty or `rows + n_conf(molecule) <= block_rows`; otherwise
   close the block and start a new one with that molecule. A reader MUST NOT
   assume this rule; it takes the table as given, subject to rules 1-4.

## 7. The molecule run table

The `runs` section is `u8[n_mol, 16]`; record `m` describes molecule `m` of
the group, in corpus order (section 9):

| offset | size | field | rule |
|---|---|---|---|
| 0 | 8 | `mol_id` u64 | `0 <= mol_id < 2^63` (it is a non-negative int64 everywhere else; `prep/conformer.py::stable_id` masks to 63 bits) |
| 8 | 2 | `n_conf` u16 | `1 <= n_conf <= 4096` |
| 10 | 1 | `n_real` u8 | `1 <= n_real <= 255` |
| 11 | 5 | reserved | MUST be zero |

Rules:

1. The rows of molecule `m` are `[run_start[m], run_start[m] + n_conf[m])`
   with `run_start[m] = sum of n_conf over [0, m)`; the run table's `n_conf`
   values sum to the header's `n_conf`. Row `row` belongs to molecule
   `m = searchsorted(run_start, row, side="right") - 1` (the last `m` with
   `run_start[m] <= row`), and its `conf_idx` is `row - run_start[m]`.
2. The table is sorted by `(n_real, mol_id)` ascending (section 9.2) and no
   `mol_id` repeats, in the group or anywhere else in the corpus. (`mol_id`
   alone is therefore not sorted across the table; a reader MUST NOT expect
   it to be.)
3. Every `n_real` lies in the group's bucket: `edges[b] <= n_real < edges[b+1]`
   (section 9.1).
4. There is **no stored per-conformer count array**, no per-conformer
   `mol_id`, `conf_idx` or `n_real` array. A row's counts are derived
   (section 8.5).

## 8. Color: the feature table and derived pseudo-atoms

A color corpus stores each molecule's pharmacophore features once, as
`(type, heavy-atom indices)`. No pseudo-atom coordinate is stored; positions
are derived per conformer from the decoded heavy atoms (section 8.3). The
builder perceives and stores features once per molecule. Ordinary screening
readers use that table without SMARTS matching; `deep` validation separately
re-perceives sampled molecules (every molecule with `--all`) inside a bounded
subprocess to check the stored table (section 16.4).

### 8.1 Layout (CSR)

Four sections (section 3.3), all per group:

```
feature_offsets      u32[n_mol + 1]    features of molecule m are
                                       f in [feature_offsets[m], feature_offsets[m+1])
feature_types        u8[n_feat]        type of feature f, in [1, n_types]
feature_atom_offsets u32[n_feat + 1]   atoms of feature f are
                                       feature_atoms[feature_atom_offsets[f] : feature_atom_offsets[f+1]]
feature_atoms        u8[n_feat_atoms]  heavy-atom indices (section 4.2 order), each < n_real of the molecule
```

Feature indices `f` are group-wide (the four arrays are per group); atom
indices are **0-based** and per molecule (an index into the molecule's own
heavy atoms, section 4.2); types are **1-based**.

Rules:

1. `feature_offsets[0] == 0`, non-decreasing, `feature_offsets[n_mol] == n_feat`.
2. `feature_atom_offsets[0] == 0`, strictly increasing,
   `feature_atom_offsets[n_feat] == n_feat_atoms`; every feature has between 1
   and 16 atoms, and `n_feat_atoms <= 2^32 - 1` so the u32 offsets can hold
   it (section 14).
3. Within a feature, atom indices are strictly ascending (so a feature is a
   set), and every index is `< n_real` of the feature's molecule.
4. Within a molecule, features are sorted by `(type, atom-index tuple)`
   ascending: type ascending, and within a type lexicographically by the atom
   tuple. Two features with the same type and the same atoms MAY both be
   present (they are then adjacent); such duplicates are meaningful (two
   coincident identical features count four times one feature's self term,
   `doc/COLOR_CONTRACT.md` section 3) and MUST NOT be merged.
5. Per molecule, at most 255 features of any one type and at most 255 features
   in total (section 14).
6. Types are `1..n_types` in the corpus's numbering: type `t` is
   `color.types[t - 1]` of the descriptor (section 10.2). Type 0 is a real
   atom and never appears in the table.

Rule 4 makes the table a pure function of the molecule's graph and the
definition (up to RDKit's perception), checkable by equality in `deep`, and
makes the kernel layout order (section 8.4) independent of the order in which
a feature factory enumerates matches. (Appendix B.9.)

### 8.2 Stored meaning of the atom indices

The indices are **heavy-atom indices in the stored atom order** (section 4.2:
canonical SMILES output order of the hydrogen-stripped molecule), which is
the atom order of `MolFromSmiles(smiles)` (section 11.1). A feature that
references a hydrogen cannot be stored and MUST be refused at build.

Features are perceived on the molecule **with every hydrogen present**
(`prep.conventions.features_perceived = "with_hydrogens"`): on
`Chem.AddHs(Chem.MolFromSmiles(smiles))`, where `smiles` is the molecule's
stored SMILES parsed with CXSMILES disabled (section 11.1). That one molecule
serves every perception: the builder's (after the SMILES round-trip check of
section 11.1 has passed; a molecule that fails it is rejected and never
perceived), the `deep` tier's re-check, and featurizing a query from the
corpus's own definition. `AddHs` appends the hydrogens after the heavy atoms,
so atom `i < n_real` of the perceived molecule is slot `i` and a feature atom
`>= n_real` is a hydrogen. Perception MUST NOT use the input molecule as read:
its aromaticity model is whatever its reader left (an SDF record is
sanitized on reading and carries RDKit's default model; an RDKit binary
pickle is not, and keeps the model it was written with), and a table
perceived under another model is one `deep` refuses (`layout.features`).
No coordinate passes through the SMILES: coordinates come from the source
conformers in canonical slot order; only perception uses the parsed
molecule. What the non-isomeric SMILES drops -- stereochemistry and
isotopes -- the feature definitions do not read. Hydrogens MUST be added
before matching: perceived on the hydrogen-stripped graph, BaseFeatures gets
the Hydrophobe set wrong for 13,183 of 13,500 sampled conformers.

**Perception rule.** Only the families listed in `color.types` count; a
feature's type is `types.index(family) + 1`.
* `rdkit_fdef`: `ChemicalFeatures.BuildFeatureFactoryFromString(definition)`
  on the stored text, then `factory.GetFeaturesForMol(mol_with_hydrogens)`;
  a feature's atoms are its `GetAtomIds()`. Every block of a selected family
  MUST have a `Weights` line with all values `1.0` (equal weights);
  a builder refuses a definition that violates this, naming the family.
* `smarts_list`: for each selected family, for each pattern in the family's
  list order, `Chem.MolFromSmarts(pattern)` and
  `mol_with_hydrogens.GetSubstructMatches(query)` with RDKit's defaults
  (`uniquify=True`, `useChirality=False`, `useQueryQueryMatches=False`,
  `maxMatches=1000`); each match is an ordered atom-index tuple, and a tuple
  identical (same atoms, same order) to one already taken for that family is
  dropped -- the same atoms in another order, or in another family, are kept.
  `maxMatches=1000` silently truncates a pattern with more matches; the
  per-molecule feature limits (section 14) refuse such a molecule anyway. A
  pattern that does not parse MUST be refused, at build and by `deep`.
* Then, for both: a feature containing a hydrogen atom is refused at build;
  atoms are slots (identity on the heavy atoms) and the table is ordered by
  section 8.1 rules 3-4 (duplicates kept).

### 8.3 Derived pseudo-atom positions

For feature `f` with atoms `a_1 .. a_k` (`k >= 1`) of a row whose decoded
heavy-atom coordinates are `v[i]` (the wrapped int16 of section 5.6, step
units), the pseudo-atom position is computed **per axis independently** as
the **integer mean rounded half away from zero**:

```
s = v[a_1] + v[a_2] + ... + v[a_k]        exact integer (fits int32: |s| <= 16 * 32768)
q = sign(s) * ((2*|s| + k) // (2*k))      sign(0) = 0; // is floor division
```

For `k = 1` this is `q = v[a_1]`, the atom's own stored coordinate exactly.
Examples: `s = 5, k = 2 -> 3`; `s = -5, k = 2 -> -3`; `s = 7, k = 3 -> 2`;
`s = -7, k = 3 -> -2`; `s = 3, k = 6 -> 1` (0.5 rounds away from zero);
`s = -3, k = 6 -> -1`. The result is an int16 in `[-32768, 32767]` whenever the
inputs are, and it is the same on every platform because no floating-point
operation is involved. (This differs from the quantized float centroid by at
most one step for multi-atom features; that is a declared numerics property
of format 3, not a spec defect.)

### 8.4 Kernel layout order (the merged record)

In any padded per-row layout a reader produces (the device int16 layout, the
CPU float32 layout), row slots are:

```
[0, n_real)                     the real atoms, in stored order
[n_real, n_real + n_color)      the molecule's features, in feature-table order
                                (section 8.1 rule 4: one contiguous ascending
                                run per type)
[n_real + n_color, L)           zero padding
```

where `n_color = feature_offsets[m+1] - feature_offsets[m]` for the row's
molecule `m`, and "feature-table order" is the **stored** order `f =
feature_offsets[m], feature_offsets[m] + 1, ...`: a decoder never sorts (a
table that violates rule 4 is invalid and is refused by `scan`). A row's
pseudo-atoms on their own are the `int16[n_color, 3]` array of those
positions in the same order. This is the merged record the kernels read
with `library_color_offset < 0`, and `pharmacophore.generate_color_atoms`'s
contiguous-run invariant holds by construction of rule 4.

### 8.5 Derived per-row counts

A row's count vector is derived directly from the run and feature tables:

```
counts    = uint8[n_types + 1]
counts[0] = n_real[m]
counts[t] = number of features of molecule m with type t,  t in [1, n_types]
n_color   = sum of counts[1..n_types]       (not an element of the vector)
```

The vector has exactly `n_types + 1` entries (`[n_real]` alone for a
shape-only corpus). The device's per-conformer **count word** is a different
object -- 8 bytes: byte 0 `n_real`, bytes 1-6 the per-type counts (zero
beyond `n_types`), byte 7 `n_color` (`layout.MAX_FEATURES = 7` type bytes
including byte 0; that is why `n_types <= 6`) -- which the producer builds
from the vector; the format stores neither. Kernel consumers use
`layout.color_view_from_counts` for library count vectors;
`layout.counts_from_types` decodes explicit types on the query side.

### 8.6 Widths

Per group, recorded in the header (section 3.4) and in the group table
(section 10.4), and recomputed and checked by the validator:

```
L_heavy       = pad8(max over the group's molecules of n_real)
L             = pad8(max over the group's molecules of (n_real + n_color))     (the device width)
L_field_width = pad8(max over every molecule of the bucket of n_real)
              = max over the bucket's groups of L_heavy
```

`L_heavy <= 256`, `L <= 512` (section 14). `L_heavy` and `L` are in the group
header and the group table; `L_field_width`, a bucket property, is recorded
**once per group in the group table only** (section 10.4), not in the group
header. It sets the query-field lattice and changes numerics
(`prep/build.py` `L_field`), so it is carried exactly. For a shape-only corpus
`L == L_heavy`.

## 9. Corpus order, ids and the units of work

### 9.1 Buckets

Bucket `b` holds the molecules with `edges[b] <= n_real < edges[b+1]` for
`edges = [0, 12, 16, 20, 24, 28, 32, 38, 44, 50, 56, 64, 72, 256]` (13
buckets, `b` in `[0, 12]`; `n_real <= 255` so the last bucket is `[72, 255]`).
These are `format/r3g/limits.py::BUCKET_EDGES` with the open top edge written as
256. They are fixed by this document and are not recorded in the descriptor.
A bucket with no molecules has no groups.

### 9.2 Corpus order and the global conformer id

The rows of the corpus are ordered by the key

```
(bucket(n_real), n_real, mol_id, conf_idx)
```

ascending. The **global conformer id** (gid) of a row is its position in that
order, counting from 0 over the whole corpus. gid is corpus-local: results
that outlive a corpus key on `(mol_id, conf_idx)`; the ranking tie-break
(`numerics.rank_order`: descending score, ties by ascending gid) therefore
resolves ties by `(n_real, mol_id, conf_idx)` within a bucket, and by bucket
across buckets.

Within a group the rows are in that order, so the run table is sorted by
`(n_real, mol_id)` and each run's rows are `conf_idx = 0, 1, ...`. The groups
of a bucket are consecutive slices of the bucket's molecules, in order; the
group table (section 10.4) lists groups in `(bucket, group)` ascending order,
and

```
gid(group i, row r) = sum over j < i of n_conf_j + r
```

Two builds of the same records (each molecule's records, and the atoms within
each record, in the same order), in any molecule order, with any worker count,
resumed or not, produce the same rows in the same order; with the same
`block_rows` and `max_conf_per_group` they produce identical group files,
and an identical `corpus.json` from the same roshambo3
release (`prep.version`) recording the same `prep.commit`, within one RDKit
and numpy release and on one BLAS build and CPU (section 13.3; a development
version with a `null` `prep.commit` claims this only for the same source
tree).
Supplying a molecule's atoms in another order yields the same rows, SMILES and
coordinate sets; symmetry-equivalent atoms may then occupy each other's slots
(a graph automorphism).

### 9.3 Groups

A group is a consecutive, non-empty run of a bucket's molecules. **Write rule**
(parameterized by `max_conf_per_group`, default 2,000,000, recorded in
`build.json`): walk the bucket's molecules in order; append a molecule to the
current group if the group is empty or `n_conf + n_conf(molecule) <=
max_conf_per_group`; otherwise close the group and start the next. Group
indices count from 0 within each bucket. An empty group (zero molecules) is
invalid, so is a group with zero blocks. Readers take the group table as given
subject to section 14.

### 9.4 Blocks and the scheduling cost model

A block (section 6) is the unit of work for both the GPU producer and the CPU
workers. Its **cost** is

```
cost(block) = rows * L      with L the device width of the block's group
```

The **block directory** is the list of every block of every group, built in
memory by the reader from the group headers when the corpus is opened; it is
not a file. The GPU producer consumes blocks in file order (group table order,
then block index). CPU schedulers order the directory **expensive-first**:
descending `cost`, ties broken by ascending group-table index, then ascending
block index. This is the one definition.

## 10. `corpus.json`, the descriptor

### 10.1 Overview

`corpus.json` is a single JSON object, strict (section 16.1), canonical when
written (section 10.5.1), at most 16 MiB (section 14). The schema is
**closed**: an unknown key anywhere in it is an error. There are no flags: one
encoding, one layout, one step.

```json
{
  "format": {"major": 3, "encoding": "bp256-16", "step_log2": -8},
  "frame": {"center": "heavy_atom_centroid", "axes": "principal_axes", "handedness": "right"},
  "color": { "...": "section 10.2; absent for a shape-only corpus" },
  "prep": {
    "version": "0.1.0.dev0",
    "commit": "63c0c0a7...",
    "conventions": {
      "hydrogens": "RemoveAllHs",
      "features_perceived": "with_hydrogens",
      "atom_order": "canonical_smiles",
      "protonation": "as_given",
      "standardization": "none",
      "conformers": "as_given"
    }
  },
  "groups": [
    {"bucket": 0, "group": 0, "L_heavy": 8, "L": 8, "L_field_width": 8,
     "n_conf": 1, "n_mol": 1, "header_sha256": "..."}
  ],
  "n_conf": 1,
  "n_mol": 1,
  "ids_sha256": "...",
  "molecules_sha256": "..."
}
```

| key | fingerprinted | rule |
|---|---|---|
| `format.major` | yes | int, MUST be 3 |
| `format.encoding` | yes | string, MUST be `"bp256-16"` |
| `format.step_log2` | yes | int, MUST be -8 (the step is `2 ** step_log2` A) |
| `frame.center` | yes | MUST be `"heavy_atom_centroid"` |
| `frame.axes` | yes | MUST be `"principal_axes"` |
| `frame.handedness` | yes | MUST be `"right"` |
| `color` | yes (section 10.2) | object; present iff the corpus has color |
| `prep.version` | no | string: the roshambo3 version that built the corpus |
| `prep.commit` | no | string (git commit) or `null`: optional provenance, `null` unless the builder ran from a clean checkout (section 13.3) |
| `prep.conventions` | yes | object with exactly the six keys above, each MUST have the value shown (closed enumerations with one value each) |
| `groups` | no | the group table, section 10.4 |
| `n_conf` | no | int; MUST equal the sum of the group table's `n_conf` |
| `n_mol` | no | int; MUST equal the sum of the group table's `n_mol` and the parquet row count |
| `ids_sha256` | no | 64 hex, section 10.6 |
| `molecules_sha256` | no | 64 hex, section 10.7 |

Every value of a fingerprinted field is an integer, a string, a list or an
object; **no floating-point number appears in a fingerprinted field** except
inside the matrices, which are hashed as pinned bytes (section 10.5).

### 10.2 The `color` object

Present if and only if the corpus has color. Closed schema:

| key | fingerprinted | rule |
|---|---|---|
| `generator` | yes | `"rdkit_fdef"` or `"smarts_list"` |
| `definition` | yes | for `rdkit_fdef` a string, the `.fdef` text verbatim with line endings normalized to LF; for `smarts_list` a JSON object `{family: [smarts, ...]}` (section 10.3) |
| `types` | yes | list of strings, length `n_types`, unique, non-empty, each a family name of the definition; **type `t` is `types[t-1]`** (this list is the `{name: id}` numbering: `id = index + 1`) |
| `positioning` | yes | MUST be `"centroid"`: a feature's position is the equal-weight centroid of its atoms (the integer mean of section 8.3). No weights are stored; a definition whose blocks **for a family in `types`** declare unequal weights cannot be represented and MUST be refused at build (BaseFeatures' `ZnBinder` weights are in a family that is not selected). The pre-pass does not check weights |
| `r` | yes (as bytes) | the interaction width matrix, section 10.2.1 |
| `p` | yes (as bytes) | the interaction height matrix, section 10.2.1 |
| `rdkit_version` | **no** | string: `rdkit.__version__` of the build; a mismatch with the reader's RDKit **warns**, it never refuses |

`n_types` MUST satisfy `1 <= n_types <= 6` (section 14).

#### 10.2.1 Matrices

```json
"r": {"dtype": "f32", "shape": [7, 7], "values": [[1.0, 0.0, "..."], "..."]}
```

`dtype` MUST be `"f32"`; `shape` MUST be `[n_types + 1, n_types + 1]` (row and
column 0 are the real-atom type and are never scored); `values` is a list of
`shape[0]` lists of `shape[1]` JSON numbers. Every value MUST be finite and
MUST round-trip exactly through float32 (`float32(x) == x`); every `p` value
MUST be `>= 0`, and every `r` value MUST be either exactly `0` or `>= 1e-6` (a
zero width is skipped by every backend; a positive width below about 1e-30
makes `r^(-3/2)` overflow and the term NaN on CUDA and Metal). Indexing is `[query_type, library_type]`
(`doc/COLOR_CONTRACT.md` section 1). The matrices are compared and hashed as
their float32 little-endian bytes in row-major order (section 10.5), never as
JSON text or `repr`.

The matrices are **canonical** iff, for types `1..n` (row and column 0 are
ignored): `p[t][t] == 1` and `p[t][u] == 0` for `t != u`, and `r[t][t] == 1`,
all compared exactly in float32. Off-diagonal `r` entries may hold any valid
value and are ignored: their `p` is 0, so they contribute nothing to any score.
The shipped default (`r` all ones, `p` the identity) is canonical. One shared
function implements this test (`numerics.check_interaction_matrix`) for the
builder, writer and validator; section 18.2 depends on it. (Backends may use a
stricter internal condition for their own fast paths; that is not the format
rule.)

### 10.3 Definition text and its hash

For `generator = "rdkit_fdef"`, `definition` is the `.fdef` file text. For
`generator = "smarts_list"`, `definition` is the JSON object
`{family_name: [smarts, ...], ...}` of the custom SMARTS generator itself (not
JSON inside a string); its hash is over its canonical JSON (section 10.5.1).

**Normalization.** Before storing and before hashing, `\r\n` and lone `\r`
are replaced by `\n`; nothing else is changed (trailing whitespace, a missing
final newline and all other bytes are kept). A stored `definition` MUST NOT
contain a carriage return; readers refuse one. The text MUST be valid UTF-8
of at most 64 KiB (section 14) and MUST pass the pre-pass of section 15.

```
definition_hash = sha256(utf8(definition))             # rdkit_fdef: the normalized text
definition_hash = sha256(canonical_json(definition))   # smarts_list: the object
```

RDKit's `BaseFeatures.fdef` of release 2026.09.1 (6,794 bytes, no CR) hashes
to `766f9790513f8f94c67ec6d1c8b6d5bede59b81f6da8c2081f57e49fb28d9784`, the
SHA-256 of the raw file. The allow-list of shipped definitions (the trust
gate) and the trust flag name this hash.

### 10.4 The group table

`groups` is a list, in `(bucket, group)` ascending order, one closed object
per group:

| key | rule |
|---|---|
| `bucket` | int in `[0, 12]` |
| `group` | int `>= 0`; the groups of a bucket are `0, 1, 2, ...` with no gap |
| `L_heavy`, `L` | ints, section 8.6; MUST equal the group header's |
| `L_field_width` | int, section 8.6: the bucket-wide `pad8(max n_real)`, equal for every group of the bucket; recorded here only |
| `n_conf`, `n_mol` | ints `>= 1`; MUST equal the group header's |
| `header_sha256` | 64 hex: SHA-256 of the group file's header bytes (section 3.5) |

**No file name, no path, no URL.** The reader derives the path (section 1).
The table is not fingerprinted (the fingerprint is semantics; the table is
content).

### 10.5 The semantic fingerprint

The fingerprint names everything that changes what a score means. Two corpora
with the same fingerprint are interchangeable in meaning (their scores are
comparable); every group file carries it (section 3.4), and a reader MUST
refuse a group whose fingerprint differs from its descriptor's.

```
fingerprint = sha256(canonical_json(F))
F = {
  "format":           corpus.json["format"],
  "frame":            corpus.json["frame"],
  "prep_conventions": corpus.json["prep"]["conventions"],
  "color":            C      # key absent when corpus.json has no "color"
}
C = corpus.json["color"] with
  - "rdkit_version" removed,
  - "r" replaced by {"dtype": "f32", "shape": [n, n], "sha256": hex(sha256(f32_le_bytes(r)))},
  - "p" replaced likewise.
```

`f32_le_bytes(M)` is the `shape[0] * shape[1] * 4` bytes of the matrix as
float32 little-endian in row-major order (`np.asarray(M, "<f4").tobytes()`).
`F` contains only integers, strings, booleans, lists and objects, so its
canonical JSON is byte-stable across processes, machines and languages. The
reference computes it (`reference.fingerprint`), and
`tests/test_format3_reference.py` checks it in a separate interpreter.

#### 10.5.1 Canonical JSON

Canonical JSON, used for the fingerprint, for the group header and for
`corpus.json` as written:

* object keys sorted by Unicode code point (keys are ASCII in every schema of
  this document, so this equals byte order);
* no whitespace outside strings (`separators=(",", ":")`);
* UTF-8, non-ASCII characters written literally (`ensure_ascii=False`);
  strings escape only `"`, `\` and control characters below U+0020, with the
  short escapes `\n`, `\r`, `\t`, `\b`, `\f` and `\u00XX` otherwise;
* integers as decimal; booleans as `true`/`false`; `null` as `null`;
* floating-point numbers (they occur only in the matrices' `values`, which are
  never fingerprinted as JSON) are written in the shortest form that
  round-trips to the same float64 (Python's `repr`);
* no `NaN` or `Infinity`.

Python: `json.dumps(obj, sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False).encode("utf-8")` is the definition; the
fingerprinted fields hold no floats, so no float-formatting rule is needed
for byte stability.

### 10.6 `ids_sha256`

```
ids_sha256 = sha256( int64_le_bytes( sorted(unique(mol_id of molecules.parquet)) ) )
```

The sorted, unique `mol_id` column serialized as little-endian int64 (every
`mol_id` is in `[0, 2^63)`, so the u64 and int64 bytes coincide). It describes
the distributed content, so it lives in the distributed descriptor, and it is
**not** an input to the fingerprint. It is directly comparable to a format-2
manifest's `sha256_ids` (the SHA-256 of its `ids.i64`). `verify --tier deep`
recomputes it from `molecules.parquet`; `scan` checks that the parquet's
`mol_id` set equals the union of the run tables.

### 10.7 `molecules_sha256`

SHA-256 of the bytes of `molecules.parquet` as stored. Not fingerprinted (the
Parquet writer embeds its version string, so the bytes are a build property).
`scan` reads the whole file anyway and checks it; a future detached signature
over `corpus.json` (section 12.2) thereby covers the molecule table too.

## 11. `molecules.parquet`

One Parquet file, one row per molecule, exactly these columns (Arrow types),
in this order, **every column REQUIRED (non-nullable)**:

| column | Arrow type | rule |
|---|---|---|
| `mol_id` | `uint64` | `[0, 2^63)`; rows sorted by `mol_id` ascending; unique; the set equals the union of every group's run table |
| `ext_id` | `string` (UTF-8) | the source identifier; at most 256 bytes; no C0 or C1 control characters (U+0000-U+001F, U+007F-U+009F) and no bidi controls (U+200E, U+200F, U+202A-U+202E, U+2066-U+2069) |
| `smiles` | `string` | section 11.1; at most 4 KiB |
| `hac` | `uint8` | heavy-atom count; MUST equal the molecule's `n_real` |
| `mw` | `float64` | molecular weight, finite, `>= 0` (informational) |
| `rotors` | `uint16` | rotatable-bond count (informational) |

Writers use **ZSTD level 9**; readers accept ZSTD or
UNCOMPRESSED. Every data page is `PLAIN`-encoded (v1 or v2; the all-defined
levels may use RLE or BIT_PACKED). The declared uncompressed-size checks and
page walk remain best-effort resource bounds; decompression bombs are outside
the hardening scope (section 16). Metadata and page headers are checked in this order:

1. **before the footer is parsed**: the file size MUST be at most
   `n_mol x 4,608 + 1 MiB` (limit 21; the per-row sum of the column caps is
   4,371 bytes -- 8 + 260 + 4,100 + 1 + 8 + 2 with the two 4-byte string
   length prefixes -- rounded up to 4.5 KiB), and `molecules_sha256`
   (section 10.7) MUST match; then the reader reads the first 4 and the last
   8 bytes itself: the file MUST begin and end with `PAR1`, and the
   footer length (the little-endian u32 before the final `PAR1`) MUST be at
   most `64 KiB + 1 KiB x ceil(n_mol / 2^14)` and at most `file_size - 12`,
   **before any Parquet library parses the footer**. Basis: a legal footer
   for this schema is ~200 bytes plus ~280 bytes per row group (pyarrow 26,
   measured), there are at most `ceil(n_mol / 2^14)` row groups (step 2),
   and statistics, the only variable-length per-chunk content, are refused
   (step 2), so the bound is 3.6x a legal footer per row group plus a fixed
   64 KiB. A reader that hands the footer to a library SHOULD also pass the
   library's own Thrift limits (pyarrow: `thrift_string_size_limit` = the
   footer bound, `thrift_container_size_limit` = `max(1024, ceil(n_mol /
   2^14))`);
2. **before any pyarrow call**, parse the bounded footer against the closed
   schema in `parquet_footer.check_footer`: FileMetaData, SchemaElement,
   RowGroup, ColumnChunk, ColumnMetaData, logical STRING/INTEGER types,
   encoding statistics and column orders. Unknown, repeated or out-of-order
   fields, wrong wire types, out-of-range integers and oversized lists are
   refused. The schema has exactly six flat REQUIRED columns. SizeStatistics
   (ColumnMetaData field 16) is permitted only with repetition/definition
   histograms of at most one entry (empty or omitted is legal); the unencoded
   byte-array byte count is permitted only on BYTE_ARRAY columns. No external
   `file_path`, bloom filter, column/offset index, encryption, key-value
   metadata or embedded Arrow schema is allowed. Then the Parquet library
   checks the metadata: Arrow types are the Parquet logical types' mapping;
   the schema (names, order, types) exactly the table
   above and every column REQUIRED (`max_definition_level == 0`,
   `max_repetition_level == 0`); `num_rows == n_mol` and the row groups'
   `num_rows` summing to it; at most `ceil(n_mol / 2^14)` row groups, every
   row group at most 2^14 rows; every column chunk `ZSTD` or `UNCOMPRESSED`, its
   encodings a subset of `{PLAIN, RLE}`, no dictionary page, no statistics,
   and both its `total_compressed_size` and
   `total_uncompressed_size` at most `rows_in_group x cap_column + 4 KiB`
   with `cap_column` = 8 (`mol_id`), 260 (`ext_id`), 4,100 (`smiles`), 4
   (`hac`, stored `INT32`), 8 (`mw`), 4 (`rotors`, stored `INT32`); its byte
   range inside the data, `[4, file_size - 8 - footer_length)`;
3. **page headers**: for every column chunk the reader walks the pages from
   `data_page_offset` over `total_compressed_size` bytes, reading one page
   header at a time (never the chunk), and parses each header against a
   **closed schema**, not with a generic Thrift skipper
   (`reference.walk_parquet_pages` is the definition). In the parquet-format
   Thrift definition a `PageHeader` and the data-page sub-headers a legal
   file can carry contain no list, set, map, binary or double field, so:
   * a header is at most **256 bytes** (the largest legal one is 105: four
     i32 fields of `PageHeader` at <= 9 bytes each with a long-form field
     header, a 4-byte sub-struct field header, `DataPageHeaderV2`'s six i32
     fields, one bool, an empty `Statistics` and two stop bytes);
   * the allowed fields are, by id: `PageHeader` 1 `type`, 2
     `uncompressed_page_size`, 3 `compressed_page_size`, 4 `crc` (all i32),
     5 `data_page_header`, 8 `data_page_header_v2` (structs);
     `DataPageHeader` 1-4 (i32) and 5 `statistics`; `DataPageHeaderV2` 1-6
     (i32), 7 `is_compressed` (bool) and 8 `statistics`; `statistics` MUST be
     the **empty** struct (pyarrow writes `1c 00` even with statistics off);
   * any other field id, any field of another Thrift type -- in particular
     every container type (list 9, set 10, map 11: a compact-protocol list of
     bools consumes no bytes per element, so a 12-byte header declaring 2^62
     of them is an unbounded loop for a generic skipper) -- a field id that
     is not strictly greater than the previous one, an i32 outside the i32
     range or a varint longer than 5 bytes (3 for a field id), and a
     required field missing (`PageHeader` 1-3, `DataPageHeader` 1-4,
     `DataPageHeaderV2` 1-6) are refused;
   * every page MUST be `DATA_PAGE` (carrying exactly field 5) or
     `DATA_PAGE_V2` (exactly field 8) -- never a dictionary or index page --
     `PLAIN`, with non-negative compressed and uncompressed sizes. Sizes must
     be equal for UNCOMPRESSED chunks and for v2 pages with
     `is_compressed == false`. V2 has `num_nulls == 0` and non-negative level
     byte lengths summing to at most the compressed page size. The accumulated
     declared uncompressed sizes, including headers, obey the column cap,
     and MUST hold **at least one value**; the pages MUST tile the chunk
     exactly, and their `num_values` MUST sum to the row group's rows (the
     walk stops as soon as the running sum exceeds it, so it visits at most
     `rows` pages);
4. **then, one column chunk of one row group at a time**: each column MUST
   decode to exactly the row group's row count, with no nulls; string chunks
   MUST pass full UTF-8 validation (`limit.ext_id` or `limit.smiles`). Read and
   range-check `rotors` as uint16. For the remaining checks (the largest is a
   `smiles` chunk of at most `2^14 x 4,100 B + 4 KiB` = 67 MB): `mol_id` in `[0, 2^63)`
   (`limit.mol_id`), strictly ascending across the whole file (sorted and
   unique), and equal position by position to the sorted union of the run
   tables (section 16.3 item 7; the union is one `n_mol`-sized sort, within
   limit 34's allowance of 16 x the run-table bytes); `hac == n_real` by the
   same alignment (`limit.hac`); `smiles` non-empty, at most 4,096 bytes, and
   consisting only of printable ASCII `0x21-0x7E` other than `|` (which
   excludes whitespace and every CXSMILES suffix; `limit.smiles`; the parse
   itself is `deep`); `ext_id` non-empty, at most 256 bytes, with no C0, C1
   or bidi control character (`limit.ext_id`); `mw` finite and `>= 0`
   (`value.parquet`); `rotors` is in `[0, 65535]`. Every other violation in
   this list is `value.parquet`.
5. no extension types; readers MUST NOT enable any extension-type
   deserialization (`pyarrow >= 15`, the pickle-CVE floor, stays).

A reader opens the file once per section 1 item 2 and hands that descriptor
to the Parquet library (hashing and parsing the same descriptor; a file
modified between the two reads is a local race outside this document's
scope). **Writers** MUST write no statistics (an empty page-header
`Statistics` struct is tolerated, step 3), no embedded Arrow schema, no
dictionary pages, data pages `PLAIN` and ZSTD level 9, and row groups of
2^14 rows (the last shorter), so two writers of one release produce
identical bytes; the writer-version string the Parquet footer carries
(`created_by`) is the one field excluded from the byte-identity claim of
section 13.3. Readers refuse statistics (step 2). **Memory:** with row
groups of at most 2^14 rows read one column chunk at a time, the reference's
scan of `molecules.parquet` peaks at 202-211 MiB above the interpreter for
one and for four full row groups at the string caps (limit 34; with 2^16-row
groups read whole it was measured at 846 MiB, which is why the cap is 2^14 and reads are
per column chunk). The file's bytes are
hashed as `molecules_sha256` (section 10.7) and its `mol_id` column as
`ids_sha256` (section 10.6); neither is fingerprinted.

### 11.1 The SMILES rule

`smiles` is the **non-isomeric** canonical SMILES of the molecule **after
`RemoveAllHs`** (isotopic hydrogens removed too; `Chem.MolToSmiles(mol,
isomericSmiles=False)`), written so that `Chem.MolFromSmiles(smiles)` yields
the heavy atoms in **stored slot order**: atom `i` of the parsed molecule is
coordinate slot `i` (section 4.2), element for element. The builder
validates this at build (parse with CXSMILES disabled, compare with the
stored slots); `deep` re-checks it with CXSMILES parsing disabled. A
molecule whose SMILES does not parse, or parses to other slots (element,
formal charge, total hydrogens, radical electrons, bonds with their types),
is rejected by the builder, with its reason in `build.json`'s rejected list,
rather than refusing the build. (RDKit can write a canonical SMILES it
cannot parse: PubChem CID 12157944 is written `Cn1ccn(C)[si+]1C`, aromatic
charged silicon that does not kekulize.) A rejected molecule is absent from
`molecules.parquet` and so from `ids_sha256`. `deep` still refuses a corpus
whose SMILES does not round-trip. The parsed molecule is the one color
features are perceived on (section 8.2), so the check runs first; radical
electrons are compared because they are the one property perception could
read that the rest of the slot comparison does not cover.

**The SMILES carries no stereochemistry.** Stereo lives in the 3D coordinates
(display stereo is assigned from them, as the web overlay does), a molecule's
stereo-only conformer variants share one record under one id, so an isomeric SMILES would be ambiguous for them, and the
non-isomeric canonical order is what pins slot order. Being non-isomeric,
it carries no isotope labels either: a molecule with isotopically labeled
heavy atoms (e.g. 13C) is stored with its unlabeled graph, and the label is
not representable in a format-3 corpus. Records of one id that differ in a
heavy-atom isotope are still refused at build. The SMILES MUST NOT use
CXSMILES extensions (no `|...|` suffix); readers parse with
`allowCXSMILES=False`. A molecule whose conformers differ in anything but
stereo (a charge or protonation variant under one id) cannot be represented
and is refused at build.

## 12. Provenance and reserved files

### 12.1 `build.json`

Optional. A JSON object of build provenance: the builder's inputs, parameters
(`block_rows`, `max_conf_per_group`, sampling options), timings and host.
It is **never fingerprinted, never required, never read by a screen**. Readers
MUST NOT change any behavior on its content. `verify` MAY check that it
parses within the JSON limits of section 14 and MUST NOT fail a corpus for
anything in it except an unparsable or over-limit document at `deep`
(section 16.4 item 5). It is **not distributed**: a distributed corpus is
`corpus.json`, `molecules.parquet` and `groups/` only, and a
packager omits `build.json`. Because it records host, paths, timings and worker
counts, two builds of the same records produce identical files except
`build.json`. It also records what section 13.3 rests rebuild identity on:
the roshambo3 release (`roshambo3_version`) and, under `identity.code`, the
RDKit and numpy releases and a content digest of the code that produces the
stored bytes (the builder's `prep/` and `format/` packages,
`pharmacophore.py`, `numerics.py` and `layout.py`, and the CPU extension's
source hash). It does not record the Parquet writer's release; the Parquet
footer's `created_by` names it.

### 12.2 `corpus.json.sig` (reserved)

Reserved for a detached signature over the bytes of `corpus.json`. Major
version 3 defines no signature format; one would be defined by a later
revision of this document. A current reader MUST ignore
the file if present and MUST NOT treat its absence as an error.

## 13. Versioning policy

1. **The major version means the byte layout.** It is the u32 in the prologue
   and `format.major` in the descriptor. A reader MUST refuse any major
   version it does not implement, by number, before parsing anything else. A
   new coordinate encoding, a new section, a changed record layout or a
   changed canonical order is a new major version, never a header flag.
   There are no minor versions and no feature flags.
2. **The fingerprint means semantics.** Anything that changes what a score
   means -- the step, the frame, the color definition, numbering, positioning
   rule or matrices, the prep conventions -- is in the fingerprint (section
   10.5). A reader compares the corpus's `color` against the query's
   generator **by content** and refuses a mismatch naming the first
   difference; an RDKit version difference warns.
3. **Byte identity of rebuilds** holds for the same inputs, the same
   `block_rows` and `max_conf_per_group`, the same **roshambo3 release**
   (`prep.version`) and the same RDKit and numpy releases, which `build.json`
   records (section 12.1; canonical SMILES order and feature perception are
   RDKit-release properties), **and the same BLAS/LAPACK build and CPU
   kernel dispatch**, which `build.json` does not record (the conformer frame
   is the SVD's, whose axis signs depend on both; section 4): group files are
   identical; `corpus.json` is
   identical except that `prep.commit` is recorded as found, so it may
   differ there alone; `molecules.parquet` is identical, given also the
   same Parquet writer release, which `created_by` names, except for that
   `created_by` string in the Parquet footer (section 11). `prep.commit` is
   optional provenance, present only when the builder ran from a clean
   checkout; a builder installed from a wheel or a source archive records
   `null`, and identity does not depend on it. A development version (a
   `.dev` suffix) with a `null` `prep.commit` names no particular source --
   the builder can change between two development builds without the
   version moving -- so for it identity is claimed only for the same source
   tree (`build.json`'s `identity.code` digest identifies the code that
   produces the stored bytes; section 12.1). `prep.version` is
   `roshambo3.__version__`, read from the metadata of the installed
   distribution that owns the imported code (else the source tree's own
   version string), so it is `pyproject.toml`'s version. Across RDKit releases the
   fingerprint is unchanged (the RDKit version is excluded) while files may
   differ; that is a reproducibility note, not an incompatibility.
   Likewise, builds that differ only in the BLAS build or the CPU (for
   example, the same numpy wheel on an Intel AVX-512 host and an AMD host) may
   store a small fraction of conformers rotated by pi about one principal
   axis -- measured at about 1.3 per million rows on LIT-PCBA -- with every
   other section identical; scores under the default rotating starts
   (`start_mode=1`, whose start set is closed under those rotations) are
   expected to be unchanged.
4. No historical format is readable. A format-2 `.rsb3` file
   fails at the magic (`"RSB3"` is not `"R3G\0"`), so it cannot be mistaken for
   a format-3 file with a different version.

## 14. Limits

Exceeding any limit makes the file invalid; the validator refuses it with the
limit named (section 16). Every product below is computed in arbitrary
precision (Python integers), never in a fixed-width type.

### 14.1 The limits table

| # | Quantity | Limit | Tier | Basis |
|---|---|---|---|---|
| 1 | `corpus.json` size | <= 16 MiB | open | -- |
| 2 | group header size (`file_size - header_offset`) | <= 16 MiB, and the header MUST end exactly at EOF | open | -- |
| 3 | JSON nesting depth (any JSON in the corpus) | <= 32 | open | -- |
| 4 | JSON string length | <= 65,536 bytes: the raw bytes of the JSON string literal between its quotes, escapes included, measured on the document before parsing | open | the definition text is the largest string |
| 5 | groups in a corpus | <= 65,536 (2^16) | open | a 2^20-entry group table cannot fit limit 1; 2^16 x 2M conformers is 1.4 x 10^11 rows |
| 6 | blocks per group | <= 2^20 | open | -- |
| 7 | `n_conf` per group | <= 2^24 | open | builder cap 2M |
| 8 | `n_mol` per group | <= `n_conf` | open | each molecule has >= 1 row |
| 9 | `n_real` | 1..255 | scan (run table) | `MAX_N_REAL`; the run table stores u8 |
| 10 | `n_conf` per molecule | 1..4,096 | scan | stored as u16 |
| 11 | `L_heavy` | `pad8(max n_real)` <= 256 | open (declared) / scan (recomputed) | checked against the run table's `n_real` |
| 12 | device `L` | `pad8(max(n_real + n_color))` <= 512 | open / scan | `n_color` derived from the feature table |
| 13 | `L_field_width` | = `pad8(max n_real over the bucket)` <= 256 | scan | makes the bucket-wide width checkable |
| 14 | `n_types` | 0 for a shape-only corpus; 1..6 for a color corpus | open | the device count word is 8 bytes -- byte 0 the real-atom count, bytes 1-6 per-type counts, byte 7 the color total (`layout.MAX_FEATURES = 7` counting byte 0) -- so at most 6 types |
| 15 | features per type per molecule | <= 255 | scan (implied by 16) | u8 counts; implied by limit 16, which is checked, so a file exceeding it is refused as `limit.features_per_molecule` |
| 16 | features per molecule (`n_color`, the color total) | <= 255 | scan | the count word's color-total byte (byte 7) is u8 |
| 17 | atoms per feature | 1..16 | scan | BaseFeatures max 8, in `Arom8` |
| 18 | block `rows` | 1..4096 for every block | open | rows are capped, and the decoded size follows from the cap |
| 19 | block byte `length` | MUST equal `32 + 32*G*(w_x + w_y + w_z)` (section 5.1) with `total` from the run table | scan | the closed form of the bit-packed layout |
| 20 | `n_feat_atoms` (and `n_feat`) per group | <= 2^32 - 1 | open (shape) / scan | the CSR offsets are u32 |
| 21 | `molecules.parquet` | file size <= `n_mol x 4,608 + 1 MiB`; footer length <= `64 KiB + 1 KiB x ceil(n_mol / 2^14)`, checked before the footer is parsed; at most `ceil(n_mol / 2^14)` row groups of <= 2^14 rows; column chunk sizes <= `rows x cap_column + 4 KiB`; no statistics; every page a ZSTD or UNCOMPRESSED `PLAIN` data page holding >= 1 value, verified from page headers of <= 256 bytes parsed against a closed schema; all columns REQUIRED; no embedded Arrow schema | scan (before any data page) | best-effort declared-size bounds (section 11). (The decoded block sizes follow from rows 11, 12 and 18 and need no row of their own: 6.3 MB heavy, 12.6 MB device, 25.2 MB CPU float32, section 6 rule 4.) |
| 22 | bit width `w_a` | 0..16 | scan | -- |
| 23 | zstd window and every other codec parameter | -- | -- | **dropped**: there is no compressor |
| 24 | definition text size | <= 64 KiB | scan (pre-pass); for text stored in `corpus.json`, limit 4 refuses it at `open` first | -- |
| 25 | definition text lines | <= 4,096 | scan (pre-pass) | bounds the pre-pass itself |
| 26 | `AtomType` lines / `DefineFeature` lines, case-insensitive | <= 256 / <= 256 | scan (pre-pass) | BaseFeatures: 35 / 27 |
| 27 | expanded SMARTS per pattern / total over every `AtomType` and `DefineFeature` pattern, UTF-8 bytes | <= 8,192 / <= 131,072 | scan (pre-pass) | BaseFeatures: 590 B / 4,475 B under RDKit's expansion |
| 28 | `$(` recursion depth in any expanded pattern, macro wrapping counted | <= 8 | scan (pre-pass) | the nested-SMARTS probe and the macro bomb (section 15) |
| 29 | top-level atoms per expanded pattern (`[...]`, organic subset, aromatic, `*`, `A`, `a`) | <= 16 | scan (pre-pass) | follows from 17 (a feature's atoms are the pattern's matched atoms; BaseFeatures max 8); refuses the 18-atom chains |
| 30 | SMARTS-list definitions: patterns / each (UTF-8) / canonical JSON | <= 256 / <= 1 KiB / <= 64 KiB | scan (pre-pass) | -- |
| 31 | `smiles` | <= 4 KiB; no CXSMILES | scan / deep | -- |
| 32 | `ext_id` | <= 256 bytes of UTF-8; no C0/C1 or bidi controls | scan | -- |
| 33 | time to refuse | `open`: <= 20 s per corpus file (group file or `corpus.json`) on any input, and <= 20 s for every adversarial seed at its stated tier; `scan` and `deep`: linear in the bytes read, at least 1 MiB/s of section bytes plus 20 s per file for the reference (`deep` adds a term linear in the molecule count, divided across its workers, and none in the group count; section 16.4); all on CI-class hardware, measured end to end (Appendix B, S1; the measured worst cases sit far inside) | -- | best-effort engineering target (section 16, hardening scope); tested by `tests/test_format3_reference.py` and the adversarial manifest |
| 34 | validator peak memory | per corpus file: `open` <= 256 MiB above the interpreter; `scan` <= 256 MiB + 16 x the bytes of the file's fixed sections (sections 2-9), with no allocation proportional to `n_mol` or `n_feat` beyond that factor (checks are vectorized or chunked, never one Python object per molecule); `deep` adds the one block being decoded; `molecules.parquet` (`scan`): <= 256 MiB above the interpreter plus 16 x 8 bytes per molecule (the sorted `mol_id` and `n_real` union it is compared with), independent of the size of any one row group beyond the 2^14-row cap (section 11) | -- | best-effort engineering target (section 16, hardening scope). Measured: a 103 MiB group at limits 15-17 scans at 163 MiB with the feature checks chunked (2.8 GiB, over this limit, unchunked); full parquet row groups at the string caps at 202-211 MiB |
| 35 | `mol_id` | `0 <= mol_id < 2^63` | scan | `mol_id` is a non-negative int64 everywhere else |
| 36 | `hac` | == `n_real` | scan | -- |
| 37 | JSON containers per document | <= 2^18 objects and arrays, counted before parsing | open | best-effort allocation guard; `json.containers` |

**Nothing in the file is executed or used as a name.** `format.encoding`,
every section name, every dtype, every `generator`, every convention string
is a closed enumeration checked against this document. The only corpus text a
reader ever compiles is the color definition, and only after the pre-pass
(section 15) and the trust gate.

### 14.2 Allocation rule

A reader MUST bound every allocation by header values that have already
passed the limits above before allocating: the pinned pool from `rows` and
`L` using `native.gpu_layout_elements` (section 6 rule 4: at most
12,582,912 B device, including tile padding; 25,165,824 B CPU float32); scratch from
`total`; parquet row groups from limit 21; nothing from a value that has not
been checked. Every size, offset and count is computed in arbitrary
precision or in overflow-checked 64-bit arithmetic; the maximum of every
quantity at the legal limits, and the narrowest integer type it fits, is
tabulated in section 14.2.1 (section 17 item 11).

#### 14.2.1 Integer widths at the legal limits

Every product, sum and offset of sections 5-9, 11 and 17 at the legal limits
of section 14.1: `n_real <= 255`; `n_conf <= 4,096` per molecule; every
block `<= 4,096` rows; `n_conf <= 2^24` and blocks `<= 2^20` per group;
groups `<= 2^16`; `L_heavy <= 256`, `L <= 512`; `w <= 16`; features per
molecule `<= 255`, atoms per feature `<= 16`; parquet row groups `<= 2^14`
rows. The maxima are computed with Python integers.

| quantity | expression | maximum | fits |
|---|---|---:|---|
| block `total` (5.2), one or several molecules | 4,096 x 255 | 1,044,480 | u32 |
| packing groups `G` (5.1) | ceil(total / 256) | 4,080 | u32 |
| axis stream bytes `Sa` (5.1) | G x 16 x 32 | 2,088,960 | u32 |
| `block_length` (5.1) | 32 + 32 G (w_x+w_y+w_z) | 6,266,912 | u32 (stored u64) |
| stream byte offset `byte(g,i,l)` (5.3) | < Sa | 2,088,959 | u32 |
| lane bit `b = k w + bit` (5.3) | 15 x 16 + 15 | 255 | u8 |
| `min_a + u` before the wrap (5.6) | 32,767 + 65,535 | 98,302 | i32 |
| decoded heavy elements per block (4.3) | 4,096 x 3 x 256 | 3,145,728 | u32 |
| decoded device elements per block (8.4) | 4,096 x 3 x 512 | 6,291,456 | u32 |
| decoded device bytes per block | x 2 | 12,582,912 | u32 / size_t |
| `coords` section bytes per group (block `offset`) | 2^20 x 32 + 1,536 x (2^24 x 255 / 256 + 2^20) | 27,313,307,648 | u64 |
| `first_row`, `m0`, `m1`, run starts (6, 7) | <= n_conf per group | 16,777,216 | u32 |
| sum of `n_conf x n_real` over a group | 2^24 x 255 | 4,278,190,080 | u32 (2^24 spare; accumulate in u64) |
| `n_feat` per group (`feature_offsets` values) | 2^24 x 255 | 4,278,190,080 | u32 |
| `n_feat_atoms` per group, *without* limit 20 | n_feat x 16 | 68,451,041,280 | u64 -- exceeds u32 |
| `n_feat_atoms` per group, with limit 20 | 2^32 - 1 | 4,294,967,295 | u32 (`feature_atom_offsets`) |
| 8.3 sum `s` | 16 x 32,768 | 524,288 | i32 |
| 8.3 `2 abs(s) + k` | 2 x 524,288 + 16 | 1,048,592 | i32 |
| count bytes, `n_color` (8.5) | limits 15, 16 | 255 | u8 |
| `n_real + n_color` | 255 + 255 | 510 | u16 (<= L = 512) |
| block cost `rows x L` (9.4) | 4,096 x 512 | 2,097,152 | u32 |
| group file size (every section at its maximum) | 64 + sections + padding + 16 MiB header | 53,619,983,037 | u64 / off_t |
| section `length = prod(shape) x itemsize` (3.2), legal | largest: `coords` | 27,313,307,648 | u64, checked |
| global conformer id (9.2) | 2^16 groups x 2^24 | 1,099,511,627,776 | i64 -- exceeds u32 |
| corpus `n_conf`, `n_mol` (10.1) | 2^16 x 2^24 | 1,099,511,627,776 | i64 |
| blocks in the block directory (9.4) | 2^16 x 2^20 | 68,719,476,736 | u64 -- exceeds u32 |
| parquet file-size cap (11 step 1) | n_mol x 4,608 + 1 MiB | 5,066,549,581,840,384 | u64 |
| parquet row groups (11 step 2) | ceil(n_mol / 2^14) | 67,108,864 | u32 |
| parquet footer-length cap (11 step 1) | 64 KiB + 1 KiB x row groups | 68,719,542,272 | u64 (the stored length is u32: compare in 64-bit) |
| parquet column-chunk cap (11 step 2) | 2^14 x 4,100 + 4 KiB | 67,178,496 | i32 (parquet page sizes) |
| parquet pages per chunk (11 step 3) | <= rows, each >= 1 value | 16,384 | u32 |
| `ids_sha256` input bytes (10.6) | n_mol x 8 | 8,796,093,022,208 | u64 |

Consequences:

1. Four stored fields exceed u32 and are specified wider: the `coords` section
   of one group and therefore block-table `offset`s (u64, section 6);
   `n_feat_atoms`, which limit 20 caps at `2^32 - 1` so the u32
   `feature_atom_offsets` hold it, refused explicitly at `open` as
   `limit.n_feat_atoms`; the global conformer id and the corpus totals
   (64-bit, section 17 item 11); and the block-directory entry count (u64).
2. The sum of `n_conf x n_real` over a group fits u32 with only 2^24 to
   spare; accumulate it in u64.
3. File-controlled values not yet checked (section shapes, offsets and
   lengths in a JSON header; block-table `offset + length`) can be anything
   up to the JSON integer's range: they MUST fit int64 when parsed (else
   `schema.value`) and are combined only in overflow-checked arithmetic or
   in comparisons that cannot overflow.
4. The parquet footer-length cap exceeds u32 above about 2^36 molecules
   while the stored footer length is u32: compare in 64-bit.
5. A block's `coords` byte range, `coords_offset + offset + length`, is at
   most `header_offset` (<= 53.6 GB) in a legal file: u64.

## 15. The definition-text pre-pass

An RDKit-free pure function over the stored `definition`, run in the `scan`
tier before any RDKit code sees the text, and reused by the trust gate
(below; the same function; there is one implementation,
`reference.check_definition_text(definition, generator, types)`). It accepts
a **closed grammar** -- a subset of what RDKit's `.fdef` parser accepts,
chosen so that everything the pre-pass accepts is bounded to parse and to
build -- and refuses everything else, naming the limit or the rule. It MUST
stop at the first limit crossed (the expansion itself is bounded) and MUST
complete within limit 33 on any input.

**The trust gate** (`src/roshambo3/definitions.py`) decides whether a color
definition may be compiled at all. RDKit's own `BaseFeatures.fdef` is shipped
and recognized by its hash (section 10.3); any other definition needs an
explicit opt-in naming its exact SHA-256 (`roshambo3 definitions trust`,
`--trust-definition`). Every definition, trusted or not, passes this pre-pass
before RDKit sees it, and every RDKit feature factory built from corpus text
-- query featurization and the `deep` tier's re-perception alike -- is built
in the gate's sandbox (`src/roshambo3/format/r3g/_sandbox.py`): a separate
process with `RLIMIT_AS` of 2 GiB and a CPU budget of 30 s plus 0.05 s per
molecule of its batch, killed as a whole on a wall-clock deadline (section
16.4 item 3). Trust removes the refusal, never the bounds.

**What RDKit's parser does** (measured on RDKit 2026.09.1; the grammar
models exactly this):

* keywords are case-insensitive whole tokens (`AtomTypeX` is an error;
  BaseFeatures itself spells `Atomtype` twice);
* a line's **pattern is its third whitespace-separated token**; anything
  after it is ignored, so there is no inline-comment rule and `#` is only
  ever the SMARTS triple bond;
* a line ending in `\` is joined with the next;
* `AtomType Name pattern` adds to `Name`'s positive definitions, `AtomType
  !Name pattern` to its negations; names are case-sensitive strings of any
  characters except `{`, `}` and whitespace;
* a reference `{Name}` expands, at the point of use, over the definitions
  recorded **so far** (`{Name}` inside a later definition of `Name` sees the
  earlier ones) to

  ```
  $([ !$(n_k);...;!$(n_1); p_1' ,$(p_2),...,$(p_j) ])
  ```

  with the negations newest first, each in its own `$(...)`, AND-joined
  (`;`) before the positives; the first positive `p_1'` with one outer
  `[...]` removed when it has one; every later positive in its own `$(...)`,
  OR-joined (`,`); and a lone unbracketed positive giving just `$(p_1)`.
  Every macro reference is therefore at least one more level of SMARTS
  recursion (`reference.expand_macro` is the definition);
* the SMARTS of every `AtomType` and `DefineFeature` is parsed when the file
  is read, referenced or not;
* blank and comment lines may appear anywhere, inside a block included.

For `generator = "rdkit_fdef"`, the text MUST satisfy all of the following,
checked in this order (rule 1 over the whole text, then rules 2-6 in one
pass over the lines, then rule 7):

1. **Text.** Encodable as UTF-8 (a lone surrogate written as a JSON escape is
   refused, `value.definition`); no CR; at most 64 KiB of UTF-8 (limit 24);
   at most 4,096 lines, counting `text.split("\n")` so a final newline
   yields one empty last line (BaseFeatures: 218); **no line ends in a
   backslash** after trailing whitespace is removed, comment lines included
   (continuation lines are refused, not joined).
2. **Lines.** Each line is split on whitespace. An empty line or one whose
   first token starts with `#` is skipped. Otherwise the first token,
   lowercased, MUST be one of `atomtype`, `definefeature`, `family`,
   `weights`, `endfeature`; **any other line is refused** (`value.definition`;
   there is no include, no file directive, no unknown keyword). Outside a
   `DefineFeature` ... `EndFeature` block only `AtomType` lines may appear;
   inside one only `Family`, `Weights` and `EndFeature`; `AtomType` or
   `DefineFeature` inside a block, `Family`/`Weights`/`EndFeature` outside
   one, and a block left open at the end are refused. `AtomType` and
   `DefineFeature` lines need a name token and a pattern token; the pattern
   is the third token and later tokens are ignored. `Weights` values are not
   checked here (the builder refuses unequal weights for families in
   `color.types`, section 10.2); what else RDKit may reject (a missing
   `Family`, a bad `Weights` count, unbalanced brackets) is RDKit's to
   reject inside the subprocess.
3. **Counts.** At most 256 `AtomType` and 256 `DefineFeature` lines (limit
   26), counted case-insensitively (BaseFeatures: 35 and 27; limit 26's
   `definition_line_count`; limit 25's 4,096 lines is `definition_lines`).
4. **Expansion** (limit 27). Every `{Name}` in an `AtomType` or
   `DefineFeature` pattern is expanded as RDKit does (above). An unterminated
   `{`, or a name with no earlier definition (positive or negated) -- which
   includes the first self-reference -- is refused (`value.definition`). The
   expanded size is accumulated in UTF-8 bytes **as the pattern is
   produced**: a pattern exceeding 8,192 bytes, or the running total over
   every expanded `AtomType` and `DefineFeature` pattern exceeding 131,072
   bytes, is refused the moment it is exceeded (`limit.definition_expansion`;
   BaseFeatures: longest 590 B, total 4,475 B).
5. **Structure** (limits 28, 29), for **every** expanded pattern, `AtomType`
   patterns included (RDKit parses them all): the `$(` recursion depth is at
   most 8, computed with one parenthesis stack scanned left to right over the
   whole pattern, inside and outside brackets alike: `$(` pushes an entry
   marked *recursion*; **every** other `(` (a branch, or the `%(` of an
   extended ring closure) pushes an entry marked *not recursion*; **every**
   `)` pops one entry (a `)` with nothing to pop is ignored -- RDKit rejects
   the pattern); the depth at any point is the number of *recursion* entries
   on the stack, and the pattern's depth is its maximum. (A counter that
   skips a branch `(` but still decrements on its `)` -- for a branch inside
   a bracket atom inside a recursion -- reports `[$(*(*)[$(*(*)...)])]` as
   depth 1 at any real depth, so it is not conforming.) And the number of
   top-level atoms -- bracket atoms `[...]`, the organic-subset symbols
   `B C N O P S F Cl Br I`, the aromatic symbols `b c n o p s se as`, and the
   wildcards `*`, `A`, `a`, counted where the depth is 0 and outside every bracket
   atom -- is at most 16. Per pattern the order is expansion size, then depth, then atoms.
6. Macro references inside `Family` or `Weights` lines mean nothing and are
   not expanded.
7. **Types.** Every name in `color.types` MUST be the `Family` of at least
   one `DefineFeature` block (`value.definition`). Families not in `types`
   are allowed (BaseFeatures' `ZnBinder`, `LumpedHydrophobe`).

The 543-byte macro bomb of the threat model (18 doubling levels) reaches
recursion depth 9 at its ninth `AtomType` and is refused by limit 28 there,
one level before its expansion would pass 8 KiB; its lowercase and `!Name`
variants are refused the same way; an 18-atom chain of `*`, `A` or `a`, with
or without a leading `C#`, is refused by limit 29 whatever the keyword's
case; a continued line is refused by rule 1.

For `generator = "smarts_list"`: `definition` MUST be a JSON object (an
`open`-tier `schema.type` when it is not; the pre-pass itself says
`value.definition` when called directly) mapping **non-empty family names to
non-empty lists of strings**; its canonical JSON is at most 64 KiB, it holds
at most 256 patterns in total and each pattern is at most 1 KiB of UTF-8 (all
three `limit.definition_patterns`, limit 30); every name in `color.types`
MUST be a key (`value.definition`; extra families are allowed); and each
pattern MUST pass rule 5 (`{` is literal here: there are no macros).

The pre-pass bounds parsing and factory construction, not matching cost (a
16-atom `*~*...` chain, or an 18-atom one inside a `$(...)` recursion, is
within the limits and still costs seconds on a large query): the subprocess
time and memory limits of the trust gate's sandbox do, and they apply to every build of a factory
from corpus text, trusted or not, **including the `deep` tier's
re-perception** (section 16.4). Passing the pre-pass means "bounded to parse
and build", not "cheap to match". A definition stored in `corpus.json` can
never reach limit 24: limit 4 (the JSON string literal, escapes included)
refuses it at `open` first; limit 24 is reached only through a direct call
(the trust gate's file path).

## 16. The validator

The validator coordinates structural checks in Python and has three tiers.
Screening requires `open` and `scan` validation before native scoring; `deep`
additionally invokes checked native decoding and self-overlap routines when
available, with reference fallbacks as described below.
The CLI is `roshambo3 verify <corpus> [--tier
open|scan|deep] [--all]`, default `scan`. A reader always runs `open`
on what it opens; a corpus is not screened unless `scan` has passed (cached
locally). The local scan cache is not authenticated or cryptographically
bound to all section bytes. A user who can forge cache entries or alter files
and restore their stat information can bypass scan checks.

**Hardening scope.** The validator and decoders are hardened against hangs,
resource exhaustion and memory-safety faults caused by malformed files.
Cache forgery and value-only tampering (a wrong coordinate, a lowered
self-overlap) are known limitations outside that scope, as are
decompression bombs; corpus signatures are not part of major version 3
(section 12.2).

Every refusal is one exception type, `CorpusValidationError` (a `ValueError`),
carrying `code` (a closed enumeration, section 16.5), `tier`, `path` and a
message naming the file, the field and the limit. No other exception type may
escape the validator on any input: not `KeyError`, `IndexError`,
`UnicodeDecodeError`, `struct.error`, `RecursionError`, `MemoryError` or
`OverflowError`. Limits 33 and 34 are best-effort engineering targets.

### 16.1 Strict JSON

Every JSON document of a corpus is parsed with: size checked against limits 1
and 2 before parsing; **duplicate keys refused** (an `object_pairs_hook`);
`NaN`, `Infinity`, `-Infinity` and overflowing float literals such as `1e999`
refused (`json.nan`); lone surrogate escapes refused (`json.syntax`); nesting depth
<= 32 (limit 3, checked by the hook, so the recursion limit is never reached);
string length <= 64 KiB (limit 4); at most 2^18 JSON containers per
document (limit 37), counted before parsing; the top level MUST be an object; and a
closed schema: an unknown key, a missing required key, or a value of the
wrong JSON type is an error.

### 16.2 `open`: O(header), always

Per group file:

1. magic, major version; `64 <= header_offset <= file_size`; header size
   (limit 2); the header read with a bounded read; prologue reserved bytes
   zero; **then `header_sha256` against the group table, before any header
   value is used** (when the descriptor is at hand).
2. strict JSON (16.1); closed header schema (3.4); every integer non-negative
   and within its limit (3-8, 11, 12, 14).
3. the section table: names exactly the set of section 3.3 in order; dtypes
   and shapes exactly as section 3.3 (the shape's rank is checked before its
   dimensions are multiplied); `length == prod(shape) * itemsize`;
   offsets 64-aligned, tiling `[64, header_offset)` with no gap or overlap
   (3.2).
4. the `blocks` section's hash against the header, then the section is read
   (it is small: 64 B per block) and checked: rows ascend and tile
   `[0, n_conf)`; molecule ranges tile `[0, n_mol)`; byte ranges tile
   `[0, coords.length)`; the row cap (limit 18).
5. the fingerprint equals the descriptor's; `bucket`, `group`, `n_conf`,
   `n_mol`, `L_heavy`, `L`, `n_types` equal the group table's; a `color_self`
   section is present only if the descriptor's matrices are canonical
   (section 18.2).

For `corpus.json`: strict JSON; closed schema (section 10); limits 1, 3, 4, 5;
`format` values; `prep.conventions` values; `color` schema, `types` unique
and in range (limit 14), matrices finite, `>= 0` for `r`, exactly
float32-representable, correctly shaped; the group table in order with no
gaps; totals consistent. The fingerprint is computed here.

`open` reads no section but `blocks`.

### 16.3 `scan`: O(fixed sections), vectorized

Everything in `open`, then, per group, reading the `runs`, `feature_*`, `vaa`
and `color_self` sections (`pread`, section 1 item 4) and the 32-byte block
headers:

1. run table: every reserved byte zero; `n_real` and `n_conf` in range
   (limits 9, 10); `mol_id` in range (limit 35) and not repeated across the
   corpus; sorted by `(n_real, mol_id)`; every `n_real` in the bucket; the
   sum of `n_conf` equals the header's.
2. blocks: each block's `rows` equals the sum of `n_conf` over `[m0, m1)` and
   `first_row` the sum over `[0, m0)`; the 32-byte block header's widths are
   in `[0, 16]` (limit 22), its reserved bytes zero, and the block's `length`
   equals the closed form (limit 19).
3. widths: `L_heavy` and `L` recomputed from the run and feature tables equal
   the header's; the group table's `L_field_width` equals the bucket-wide
   maximum of `L_heavy` over the bucket's groups (limits 11-13).
4. feature table (color): offsets monotone and consistent (8.1 rules 1-2);
   atoms per feature (limit 17); atoms strictly ascending and `< n_real` of
   the molecule; types in `[1, n_types]`; sorted per 8.1 rule 4; the total
   count per molecule (limit 16, which implies limit 15).
5. self-overlaps: every `vaa` finite and `> 0`; every `color_self` finite and
   `>= 0`.
6. the definition pre-pass (section 15).
7. `molecules.parquet`, in section 11's order: file size,
   `molecules_sha256`, the leading and trailing `PAR1` and the footer-length
   bound, all before the footer is parsed; the footer rules; every page
   header against the closed schema; then, one column chunk of one row group
   at a time, the row rules (`mol_id`
   range and order, position-wise equality with the sorted union of the run
   tables -- a vectorized merge, never a Python set -- `hac == n_real`,
   the `smiles` and `ext_id` string rules, `mw`; the SMILES parse is `deep`).
8. across groups: no `mol_id` appears in two groups (a sort and a diff over
   the concatenated run tables).

Every per-molecule and per-feature check above is a vectorized or chunked
array operation (limit 34); `scan` decodes no coordinate block and imports
no RDKit.

### 16.4 `deep`: opt-in

Everything in `scan`, interleaved with it: per group, in group-table order,
the group's `scan` checks and then its item 1 checks, so a `deep` refusal in
one group is reported before a `scan` refusal in a later group. After the last
group, any reference cross-decodes still owed to item 1's sample are completed
from groups already read; then come the corpus-wide `scan` checks (the bucket-wide `L_field_width`,
duplicate `mol_id` across groups, the definition pre-pass, then
`molecules.parquet`), and then items 2-5:

1. every block hash and every section hash, each hashed in bounded pieces so
   no section and no block need be resident; alignment padding zero (3.2);
   every block decoded and its tail positions checked to be zero (5.4) --
   tail positions exist only in each axis stream's last packing group, so
   the tail check reads and unpacks just those `w_a x 32` bytes. The
   decode is the **native decoder's**, with a random sample of
   **K = 32 blocks per corpus** (seeded from `os.urandom`; every block with
   `--all`) additionally decoded by the pure-Python reference and compared
   bit for bit. The K blocks are allotted to the groups in proportion to
   their rows (the group table's `n_conf`) and drawn uniformly within a
   group; an allotment larger than a group's block count carries to the next
   group, and what is left after the last group is spent on the earlier
   groups' unchecked blocks (each group reopened, its blocks re-read and
   re-hashed), so exactly `min(K, total blocks)` are cross-decoded -- every
   block of a corpus of at most K. `verify` prints the count. The sample is
   per corpus, not per group: it exists to catch a native-decoder bug, and a
   per-group sample adds cost with the group count but no assurance. The
   reference unpack applies section 5.3's bit rule with NumPy (tested
   against the scalar loop) and is independent of the native codec. A full
   pure-Python decode of a large corpus is not practical and is not
   required.
2. `ids_sha256` recomputed from the parquet (10.6).
3. every `smiles` parsed with `allowCXSMILES=False`: the parse succeeds and
   the parsed molecule has exactly `n_real` heavy atoms. (No element list is
   stored, so slot order is checked indirectly: the re-perceived feature
   table of item 4 can only match on the right atom order.) Items 3 and 4
   run in up to `--workers` sandbox children at once (default
   `min(16, CPU count)`), each with the trust gate's sandbox limits and
   budgets that scale with its own batch: consecutive rows, at most 2,000 molecules of which
   at most 250 are re-perceived (a child costs about 72 ms to start and
   0.1 ms per parse). The parent's threads only feed and wait on
   the children; no RDKit and no corpus definition runs in the parent. The
   results are checked in row order, so the refusal is the one a serial
   walk reports first -- the lowest row, then the check order -- whatever
   the worker count. Every batch is whole 250-row pieces of the serial
   walk's grid (`[250k, 250k + 250)`), and a batch of more than 250
   molecules whose child fails as a whole is re-run piece by piece before
   it is reported, so the code is the serial walk's (`value.definition` when
   the failing piece re-perceives, else `limit.smiles`). The re-run runs in
   the parent alongside the pool, so up to `workers + 1` children are live
   at once, each with its own 2 GiB `RLIMIT_AS`: `(workers + 1) x 2 GiB` of
   address space at most. Adversarial cost: the children run concurrently
   and a refusal is reported in row order, so a hostile definition (or
   SMILES) can burn about `workers x` the per-batch CPU budget (`30 s +
   0.05 s` per molecule) before the refusal, plus one 250-molecule piece's
   budget for the re-run; the children still running are then killed.
4. on a sample of molecules seeded from `os.urandom` (every molecule with
   `--all`): hydrogens added (`Chem.AddHs`), features re-perceived with the
   stored definition through `BuildFeatureFactoryFromString` (never through a
   file) **inside the trust gate's time- and memory-limited subprocess**
   (the factory is built from corpus text, so the same bounds apply here as
   on the trust path; never in the validator's own process), remapped to
   heavy indices, sorted per 8.1 rule 4, and compared for equality with the
   stored table; `vaa` and `color_self` recomputed from the decoded
   coordinates and compared within the tolerance of section 18. `vaa` is
   computed as the builder computes it (`prep/conformer.py::stored_self_volume`);
   `color_self` by the native CPU path the builder writes it with
   (`backend/cpu.py::prepare_combo_library_self_overlaps`), and for 16 of
   the sampled molecules (every molecule with `--all`) also by the float64
   reference (`reference/overlap.py::color_overlap`), which must agree with
   the native value within the same tolerance or the corpus is refused
   `internal.error` (a validator defect, not a corpus one).
5. `build.json`, if present, parses within the JSON limits (nothing in it is
   checked further).

**Check order.** `deep` refuses in this order: for each group in
group-table order, `open`, that group's `scan` checks (section 16.3 items
1-5), then its item 1 checks (hashes, padding, tail positions, decode); then
the remainder of item 1's reference cross-decode sample; then the bucket-wide
`L_field_width` check (16.3 item 3), duplicate `mol_id`
across groups (16.3 item 8), the definition pre-pass (16.3 item 6) and the
`molecules.parquet` checks (16.3 item 7); then the `ids_sha256` of item 2;
then the SMILES and feature re-perception of items 3-4; then the
self-overlap comparison of item 4, a mismatch in which is recorded per group
but deferred behind everything before it; then the `build.json` parse of
item 5. A self-overlap mismatch is therefore
reported only when nothing earlier fails, and an adversarial seed that
carries placeholder self-overlaps is refused with its designated earlier
code. A sandbox failure during feature re-perception is
`value.definition`; a SMILES-only failure is `limit.smiles`.

`deep` is the only tier that touches RDKit (through the subprocess). The
reference module implements items 1 (with the reference decoder), 2 and 5
and stubs 3-4, which the validator (`format/r3g/validate.py`, with
`format/r3g/_sandbox.py`) implements. Beyond the hash chain, `deep` checks
sampled self-overlaps against decoded coordinates and sampled feature tables
against re-perception, as specified in item 4; `--all` covers every molecule.
Without the native CPU extension, sampled color self-overlaps use the float64
reference for every sampled row. These checks do not authenticate the corpus
or prove that coordinates match an external source. Cache forgery and
value-only tampering remain outside the hardening scope (section 16), despite
the particular inconsistencies these checks can detect.

### 16.5 Refusal codes

The closed enumeration of `CorpusValidationError.code`. `adversarial/EXPECTED.json`
names one per seed.

| code | meaning |
|---|---|
| `io.not_regular_file` | a corpus file is not a regular file or is a symlink, `groups/` is not a directory or is a symlink, or a name is not a single derived component (section 1 item 2) |
| `io.short_read` | a read returned fewer bytes than the validated extent: the file shrank after it was opened (section 1 item 4) |
| `io.missing_file` | `corpus.json`, `molecules.parquet` or a listed group file is absent |
| `prologue.magic` | wrong magic |
| `prologue.major_version` | unknown major version (prologue or descriptor) |
| `prologue.header_offset` | `header_offset` outside `[64, file_size]` |
| `prologue.reserved` | reserved prologue bytes not zero |
| `json.size` | limit 1 or 2 |
| `json.syntax` | not valid JSON / not UTF-8 / not an object |
| `json.duplicate_key` | a repeated key |
| `json.nan` | `NaN`, `Infinity`, or float-literal overflow |
| `json.depth` | limit 3 |
| `json.string_length` | limit 4 |
| `json.containers` | limit 37 |
| `schema.unknown_key` | a key this document does not define |
| `schema.missing_key` | a required key absent |
| `schema.type` | wrong JSON type, or an out-of-enumeration value (encoding, dtype, section name, convention, generator, positioning) |
| `schema.value` | a value violating a rule of this document other than a limit (e.g. `step_log2 != -8`, a path-like or non-integer group entry, a negative integer, groups out of order) |
| `limit.<name>` | a limit of section 14.1 exceeded; `<name>` is one of `groups`, `blocks`, `n_conf`, `n_mol`, `n_real`, `n_conf_per_molecule`, `L_heavy`, `L`, `L_field_width`, `n_types`, `features_per_type`, `features_per_molecule`, `atoms_per_feature`, `n_feat_atoms`, `block_rows`, `block_length`, `bit_width`, `definition_size` (24), `definition_lines` (25), `definition_line_count` (26), `definition_expansion` (27), `definition_depth` (28), `definition_atoms` (29), `definition_patterns` (30), `smiles` (31; also a deep parse failure or wrong heavy-atom count), `ext_id` (32), `mol_id` (35, run table or parquet), `hac` (36) |
| `layout.sections` | section table not as section 3.2/3.3 (set, order, dtype, shape, length, alignment, tiling, overlap) |
| `layout.blocks` | block table not as section 6 (tiling of rows, molecules or bytes; empty block; row cap) |
| `layout.runs` | run table not as section 7 (reserved bytes, order, bucket membership, duplicate `mol_id`, sums) |
| `layout.features` | feature table not as section 8.1 (offsets, order, duplicates of order, index `>= n_real`, type out of range), or deep re-perception differs |
| `layout.widths` | declared `L_heavy`, `L` or `L_field_width` differ from the recomputed values |
| `layout.block_header` | reserved block-header bytes not zero |
| `value.self_overlap` | a `vaa` not finite or `<= 0`, or a `color_self` not finite or `< 0`, or deep recomputation differs by more than `1e-5` relative |
| `value.matrix` | a matrix value non-finite, negative width, not float32-representable, or wrongly shaped |
| `value.definition` | the definition pre-pass refused for a grammar reason (a `limit.definition_*` is raised instead when a specific limit is the cause): a line outside the closed grammar, a continuation line, an undefined macro or unterminated `{`, a block misplaced or unclosed, CR or a lone surrogate in the text, a type that is no block's `Family`, a `smarts_list` whose values are not non-empty lists of strings or that lacks a type's key (a `smarts_list` that is not an object at all is `schema.type` at `open`; the pre-pass says `value.definition` only when called directly) |
| `value.parquet` | `molecules.parquet`: schema, REQUIRED-ness, an embedded Arrow schema, row count, file or chunk size, codec or encoding (footer or page header), a dictionary page, row-group size, sort or uniqueness, set mismatch with the run tables, `mw` |
| `integrity.fingerprint` | a group header's fingerprint differs from the descriptor's |
| `integrity.header_hash` | a group header's bytes do not hash to the group table's `header_sha256` |
| `integrity.section_hash` | a section's bytes do not hash to the header's `sha256` (deep) |
| `integrity.block_hash` | a block's bytes do not hash to the block table's (deep) |
| `integrity.padding` | alignment padding or tail positions not zero (deep) |
| `integrity.ids_hash` | `ids_sha256` mismatch (deep) |
| `integrity.molecules_hash` | `molecules.parquet` does not hash to `molecules_sha256` (scan) |
| `validation.scan_required` | screening requested without a cached passing scan |
| `internal.error` | the validator met a condition it did not anticipate and turned it into a refusal (a validator defect, including a decoder refusal after scan, still a refusal and never a crash) |

The native codec raises `CodecError`, rather than `CorpusValidationError`, for
`decode.{argument,output_capacity,overflow,allocation,self_overlap_range,
features_missing,isa_unavailable}`, `encode.range`, and `io.read_error`.
These are codec-only refusal codes. Section-table corruption can be refused
at `open` before a native call is possible; native-vector manifests may record
`native_code` when the tolerant test parser hands the codec a different view
of such a defect (for example, an incomplete molecule range becomes a run-sum
mismatch). This does not change the `layout.runs` rule for a run-sum mismatch.

## 17. Decoder requirements (native and otherwise)

A decoder is handed bytes that may not have been validated (range reads
fetch bytes after validation; validators have bugs). It MUST be safe on its
own. Function signatures and binding names are the native codec's
(`format/r3g/native.pyi`); this section is the contract.

1. **Inputs carry their lengths.** The block bytes as `(ptr, in_len)`; the
   output as `(ptr, out_capacity)` in elements; the run-table slice as
   `(ptr, n_molecules)`; the feature-table slices with their lengths. `rows`,
   `L_heavy`, `L` and `n_types` are passed from the (validated) group table
   -- or, for a standalone group file, from its validated header, which
   carries the same values.
2. **Nothing read from the block is used before it is checked.** Each width
   byte is checked `<= 16` *before* it selects anything; the unpack routine
   is selected by a `switch` with a default error arm, never by indexing a
   table with a file byte.
3. **`in_len` MUST equal `block_length`** (section 5.1) computed from the run
   table slice and the three widths, in overflow-checked arithmetic; a
   mismatch is an error before any stream byte is read.
4. **Every size is overflow-checked.** `total`, `G`, `rows * L`, every
   product, with `__builtin_mul_overflow` or equivalent; a size that does not
   fit `size_t` is an error.
5. **Every row fits its output width, and the output fits the buffer.**
   `out_capacity >= rows * 3 * L` (elements) is checked before any write;
   `1 <= n_conf <= 4096` and `n_real >= 1` for every molecule of the slice, and
   `n_real <= L_heavy` and `n_real + n_color <= L`, are checked per molecule
   before any write; a row is never written past the caller's width; padding
   is written as zeros within the width only.
6. **Reads stay inside `[in, in + in_len)`.** No decoder may load past the
   last byte of a stream (a vector loop that copies whole 8-value chunks
   reads past the last real value). Scratch buffers are the decoder's own and sized from
   checked values.
7. **Feature table slices are checked before they are indexed.** The
   feature-table slice a block call receives is `feature_offsets[m0 : m1 + 1]`
   and the feature arrays it addresses; whether the caller rebases the
   offsets to the slice or passes the group-wide arrays with the slice's
   start is the binding's choice, but either way every `feature_offsets`
   value is checked against the length of the feature arrays and every
   `feature_atom_offsets` value against the length of `feature_atoms` before
   either is used as an index, and both are checked monotone. Atom-offset pairs are rechecked at the
   gather use site, even after earlier validation. Inputs must remain unchanged
   throughout a call; outputs must not overlap inputs or one another (the
   Python bindings refuse overlaps as `decode.argument`). Then every atom
   index is checked `< n_real` of its molecule inside the gather loop, every
   type `in [1, n_types]`, the atom count per feature `in [1, 16]`, and the
   per-type and total counts of each molecule `<= 255` **before** the u8
   count vector or count word is built; the integer mean follows section 8.3.
8. **Errors are codes, never aborts.** A decoder returns an error code (or
   raises a typed exception at the Python boundary); it never asserts, never
   calls `abort`, never throws across a C boundary, and on error has written
   nothing outside `[out, out + out_capacity)`.
9. **Bit identity.** Every decoder path (scalar, AVX2, NEON, the Python
   reference) produces the same bytes for the same input, including the
   wrap of section 5.6; the conformance digests (`tests/data/format3/EXPECTED.json`)
   are the test.
10. **An empty block is refused** (`rows == 0`), as is a run-table slice that
    does not sum to `rows`.
    Run-slice sum mismatch uses `layout.runs`; empty blocks use `layout.blocks`.
11. **Integer widths.** Every quantity a native reader computes fits the
    type given for it in section 14.2.1 at the legal limits, and the reader
    MUST still compute every product, sum and
    offset that involves a **file value not yet checked against its limit**
    in overflow-checked arithmetic (items 3-4). In particular: a JSON integer
    in a group header or `corpus.json` that does not fit a signed 64-bit
    integer is refused (`schema.value`) when parsed, never wrapped or
    saturated; `prod(shape) * itemsize` (section 3.2), `offset + length`,
    `roundup64(offset + length)` and block-table `coords_offset + offset +
    length` are checked for overflow before they are compared, or written as
    comparisons that cannot overflow (`length <= header_offset - offset`
    after `offset <= header_offset`); the sums of `n_conf` over a group's run
    table (<= 2^24), of `n_conf x n_real` over a group (<= 4,278,190,080,
    which fits u32 with 16.7 M to spare, so accumulate it in u64), and the
    global conformer id (section 9.2; <= 2^40 at limits 5 and 7) are 64-bit.
    The global conformer id MUST be carried as a signed or unsigned 64-bit
    integer everywhere (the CPU and device top-k already use `int64`).
12. **Reads.** A native reader reads with bounded `pread`s and refuses a
    short read; it MAY `mmap` only under section 1 item 4's `SIGBUS` rule.

## 18. The self-overlap sections

Both are computed by the builder from the stored integers (a self-overlap
must describe the scored coordinates), dequantized at the step
(`float(v) * 2^-8`, exact), and stored per row in corpus order.

### 18.1 `vaa` (f32, every corpus)

`V_AA` of the row's heavy atoms in cubic Angstrom, the Gaussian shape
self-overlap over all ordered pairs (`i = j` included):

```
V_AA  = 8 * (pi / (2 * alpha))^(3/2) * sum_{i, j} exp(-(alpha / 2) * |r_i - r_j|^2)
alpha = kappa / R_C^2,   kappa = 2.41798793102,   R_C = 1.7 A
```

with `r_i` the dequantized stored coordinates in Angstrom (every heavy atom
has the carbon radius), computed in float64 as
`prep/conformer.py::stored_self_volume` does and stored as float32. MUST be
finite and `> 0`.

### 18.2 `color_self` (f64, color corpora with canonical matrices)

The color self-overlap of the row's **derived** pseudo-atoms (section 8.3,
dequantized at the step) under the corpus's matrices: the full ordered double
sum of `doc/COLOR_CONTRACT.md` sections 1 and 3, in physical units,

```
S = (pi / 2)^(3/2) * sum_{i, j : r[t_i, t_j] != 0}
        p[t_i, t_j]^2 * r[t_i, t_j]^(-3/2) * exp(-(r[t_i, t_j] / 2) * |b_i - b_j|^2)
```

over all ordered pairs of the row's pseudo-atoms (`i = j` included), where
`t_i` is pseudo-atom `i`'s type, `b_i` its position in Angstrom, and `r` and
`p` are the corpus's width and height matrices indexed `[t_i, t_j]`; a pair
with zero width is skipped. It is the quantity
`backend/cpu.py::prepare_combo_library_self_overlaps` prepares
(`reference/overlap.py::color_overlap` is the float64 form). Stored as
float64; MUST be finite and `>= 0` (a molecule with no feature has `0.0`).

**Canonical matrices** are defined once, in section 10.2.1: identity `p` and
unit diagonal `r` on types `1..n`, exact in float32, off-diagonal `r` and row
and column 0 ignored. `numerics.check_interaction_matrix` and
`format/r3g/reference.py::canonical_matrix_reasons` implement it.

**Presence** is optional: a builder MAY omit the section for any color
corpus (the builder's `--no-color-self`), and MUST omit it when the matrices are not
canonical. If it is present the matrices MUST be canonical; the validator
checks that. A reader whose corpus lacks the section recomputes on the CPU
path.

Tolerance: a builder MAY compute either value with the native float32 path or
an fp64 reference; the `deep` tier accepts a recomputed value within relative
`1e-5` (measured agreement between the two paths: p99 <= 1e-6, max <=
1e-5). The conformance corpus
states which path produced its values (`tests/data/format3/README.md`).

---

## Appendix A. A worked example

One block of one row with `n_real = 3` and stored coordinates

```
x = [-1, 0, 2]      y = [5, 5, 5]      z = [-300, 100, 7]
```

Per axis: `min_x = -1`, `max - min = 3`, `w_x = 2`; `min_y = 5`, `w_y = 0`;
`min_z = -300`, `max - min = 400`, `w_z = 9`. `total = 3`, `G = 1`.
`block_length = 32 + 32 * 1 * (2 + 0 + 9) = 384`.

Header (32 bytes): `ff ff` (-1) `05 00` (5) `d4 fe` (-300) `02 00 09`
then 23 zero bytes.

Axis x: `u = [0, 1, 3]` at `j = 0, 1, 2`, i.e. group 0, position `k = 0`,
lanes `l = 0, 1, 2`. With `w = 2` each lane's bits `[0, 2)` hold its value:
word 0 lane 0 = `0`, lane 1 = `1`, lane 2 = `3`, every other lane and word 1
zero. Stream x (64 bytes): `00 00 01 00 03 00` then 26 zero bytes (word 0),
then 32 zero bytes (word 1).

Axis y: `w = 0`, no bytes.

Axis z: `u = [0, 400, 307]`. `400 = 0b110010000`, `307 = 0b100110011`. Lane 1
word 0 = `400 & 0xFFFF = 400` (`90 01`), lane 2 word 0 = `307` (`33 01`);
words 1-8 are zero because every value fits in the first 16 lane bits
(`k = 0`, bits `[0, 9)`). Stream z (288 bytes): `00 00 90 01 33 01` then 26
zero bytes, then 8 x 32 zero bytes.

The file's `blocks` record: `first_row = 0`, `rows = 1`, `offset = 0`,
`length = 384`, `m0 = 0`, `m1 = 1`, and the SHA-256 of those 384 bytes. The
reference test `tests/test_format3_reference.py::TestBitPacking::test_worked_example`
pins these bytes.

A second example that crosses a word: with `w = 3` and lane 0 holding
positions `k = 5` and `k = 6` with values `u = 7` and `u = 5`: position 5
occupies lane bits `[15, 18)`, so bit 15 of word 0 is 1 (7 = 0b111, its LSB),
and bits 0-1 of word 1 are `11`; position 6 occupies lane bits `[18, 21)`,
bits 2-4 of word 1 = `101`. Lane 0 word 0 = `0x8000`, lane 0 word 1 =
`0b10111 = 0x17`.

## Appendix B. Design choices and their reasons

Each item records a choice this document makes, with its reason. The
numbering is stable (code and tests cite items as "Appendix B.n").

1. **Magic `"R3G\0"`, not `"RSB3"`.** With a new magic a format-2 reader
   fails at the magic ("not an rsb3 file") instead of reading major version 3
   as a format-2 file of a later version and reporting a misleading error
   about its color order.
2. **SHA-256 for every hash** (blocks, sections, headers, fingerprint,
   definition hash, `ids_sha256`). One algorithm keeps the pure-Python
   reference dependency-free; SHA-NI makes it ~1.5-2 GB/s per core where it
   matters (range reads).
3. **`mol_id` stored as u64 with range `[0, 2^63)`**; the int64 and u64 bytes
   coincide, so `ids_sha256` is comparable to format 2's.
4. **Sections start at 64 and extents, not lengths, tile the payload.**
   64-byte alignment and `length == prod(shape) * itemsize` cannot both hold
   for a payload starting at offset 16 (16 is not 64-aligned; `f32[n_conf]`
   is rarely a multiple of 64). Hence a 48-byte zero prologue pad, exact
   lengths, zero alignment padding inside each extent, and the tiling rule on
   extents (section 3.2).
5. **Record tables are `u8[n, 64]` and `u8[n, 16]` sections** (blocks, runs)
   rather than several SoA sections: one pointer per table is what the
   decoder's run-table slice `(ptr, n_molecules)` needs, and 16-byte run
   records keep u64 loads aligned (5 reserved zero bytes per molecule;
   6.5 MB instead of 4.1 MB on LIT-PCBA, immaterial).
6. **32-byte block header, 32-byte-aligned streams.** 23 reserved bytes per
   block (one part in 10^5 at 4,096 rows) make every 256-bit word load
   aligned; a 9-byte header would put every stream at an odd alignment. The
   header stores the three int16 minimums first and the three width bytes
   after them (not interleaved `(min, w)` per axis). An AVX2 codec of the
   stream layout decodes every width identically to the reference and
   produces the same stream bytes.
7. **Every molecule and every block is capped at 4,096 conformers/rows.** The
   decoded-size caps of section 6 rule 4 follow from that and the width
   limits, and agree across sections 6, 14 and 17.
8. **Smallest legal block is one row; an empty group is illegal; a molecule
   always ends at a block edge.** For `tests/codec_fixtures.py`: tiny blocks
   (`block_rows = 3`) are legal; `codec=`, `shuffle=` and a split color
   layout do not exist in format 3 and a fixture keyword naming them is
   refused, not mapped.
9. **Features within a molecule are sorted by `(type, atom tuple)`, atoms
   ascending within a feature, duplicates allowed.** Makes the table a pure
   function of the graph and the definition and checkable by equality in
   `deep`, and makes the kernel's within-type order independent of RDKit's
   match enumeration. The within-type order relative to RDKit's enumeration
   order changes the kernel's color accumulation order; that is part of
   format 3's declared color numerics. Within a feature the order is
   immaterial (the mean is symmetric).
10. **The `{name: id}` numbering is the ordered `color.types` list** (`id =
    index + 1`), not a second redundant dictionary.
11. **`positioning` is the one-value enumeration `"centroid"` with equal
    weights and no stored weights.** The derivation rule (integer mean)
    cannot express unequal weights, so the builder refuses them: all-ones
    weights only, for the families in `color.types`. Non-identity
    *interaction matrices* are a separate matter and are permitted (section
    10.2.1; `color_self` absent, section 18.2).
12. **`format.step_log2 = -8`** rather than a float `step`: no floating-point
    number enters the fingerprint. Matrices are pinned as float32 bytes (what
    every kernel consumes) and their JSON values must be float32-exact.
13. **Definition normalization is CRLF/CR to LF only.** No trailing-space
    stripping: BaseFeatures has five trailing-space lines, and keeping them
    makes `definition_hash` equal the raw file's SHA-256
    (`766f9790...d9784`).
14. **`color_self` is optional, and allowed only with canonical matrices**
    (section 18.2). Presence is read from the section table; the canonical
    condition is checked against the descriptor. This keeps the builder's
    `--no-color-self` legal and the "no `color_self` for a non-canonical
    matrix" rule in force without a flag in the descriptor.
15. **The `L_field_width` rule is bucket-wide `pad8(max n_real)`**, as
    `prep/build.py` computes `L_field`; stated as an equality the validator
    checks.
16. **Limits that differ from a naive reading of the data model:** at most
    2^16 groups (a 2^20-entry group table cannot fit a 16 MiB
    `corpus.json`); at most 255 features per molecule (the count word's total
    byte is u8); `n_types <= 6` (the count word); a row cap and a closed-form
    block length instead of a raw decoded-length bound; no codec limits (there
    is no compressor). The time and memory bounds (33, 34) are per file and,
    for `scan` and `deep`, linear in bytes read: a flat bound on any input
    cannot hold for a 100 MB group.
17. **Pre-pass rules for the chained-`*` seed**: limit 29 (16 top-level atoms
    per pattern) follows from the 16-atoms-per-feature limit and refuses the
    18-atom chain without a SMARTS subset; the pre-pass does not claim to
    bound matching cost (the trust gate's subprocess does).
18. **Refusal time bounds and memory bounds per file** (limits 33, 34): 20 s
    for `open` and for every seed; linear in bytes read for `scan`/`deep`
    with a stated floor of 1 MiB/s plus 20 s per file; RSS bounded by a
    multiple of the file's sections (S1 below).
19. **Per-block widths are major version 3**; a per-conformer scheme is a
    new major version.
20. **Group file name digits are minimums** (`b{:02d}_g{:04d}`), so the
    name rule imposes no cap below limit 5.
21. **`molecules.parquet` is ZSTD level 9, `PLAIN`, non-null and read one
    column chunk at a time** (section 11). Readers also accept UNCOMPRESSED;
    declared uncompressed-size bounds are best-effort. Types `mol_id uint64`,
    `hac uint8`, `rotors uint16`; hashed as `molecules_sha256` and byte-stable
    up to `created_by` within one writer release.
22. **Write rules for groups and blocks are normative for writers and not
    assumed by readers**, so the byte-identity claim of section 13 item 3 is
    a statement about two builds with the same `block_rows` and
    `max_conf_per_group`.
23. **The definition pre-pass accepts a closed grammar** (section 15), because
    RDKit's case-insensitive keywords, backslash continuations and `!Name`
    negations each bypass a looser check: a lowercase macro bomb costs RDKit
    10.5 s and 641 MB, a negation bomb 7.7 s and 475 MB, a continued 18-`*`
    chain 13 s per query. Everything the grammar accepts is modeled as RDKit
    expands it -- `{Name}` becomes `$([!$(n);...;p_1',$(p_2),...])`
    (measured through `GetFeatureDefs`), the pattern is the third token and
    trailing tokens are ignored, `A` and `a` are atoms -- so sizes and depths
    are counted the way RDKit sees them; everything else is refused rather
    than interpreted. The model must be exact: two readings of the expansion
    rule can give the same refusal codes on every seed and still differ.
24. **The SMILES carries no stereochemistry** (section 11.1): the
    non-isomeric canonical order pins the slots; stereo lives in the
    coordinates; stereo-only conformer variants share one record.
25. **`L_field_width` is stored once per group, in the group table** and not
    in the group header: a bucket property with one home, checked bucket-wide.
26. **`deep` decodes with the native decoder** and cross-checks a sample
    against the reference (section 16.4): a full pure-Python decode of a
    large corpus is not practical; the reference is the oracle, not the
    production path.
27. **No `deep` check for int16 overflow of `min + u`.** A tampered block
    that wraps gives wrong coordinates, which is outside the hardening scope
    (section 16); the wrap itself is memory-safe (section 5.6).

### Security hardening rules (S1-S9)

Scope: the hardening scope of section 16 (hangs, resource exhaustion and
memory safety from malformed files). Every implementation an item applies
to MUST follow it: the reference and the validator for every item, and the
native codec and producer for S6-S8.

* **S1. Time bounds are generous** (limit 33): `open` and every seed
  <= 20 s; `scan`/`deep` floor 1 MiB/s of section bytes plus 20 s per file.
  Reason: denial of service is a minor concern, and tight bounds refuse
  legal corpora on slow or loaded hosts -- a legal group of 2^20 tiny blocks
  takes 2.1 s to `open`. The memory bounds (limit 34) are separate.
* **S2. Parquet page headers are parsed against a closed schema** (section
  11 step 3): no container, binary or double type, no unknown or repeated
  field id, only an empty `Statistics`, at most 256 bytes per header, at
  least one value per page, the walk stops once `num_values` exceeds the
  rows. Reason: a 12-byte header declaring a list of 2^62 bools (a bool
  element consumes no bytes) runs for hours in a generic skipper. Code:
  `reference.parse_page_header`, `walk_parquet_pages`. Seeds:
  `parquet_page_bool_list`, `parquet_page_type_dictionary`,
  `parquet_page_size_mismatch`, `parquet_page_encoding_dictionary` (each
  with an honest footer).
* **S3. The parquet footer length is bounded before any library parses
  it** (section 11 step 1): `PAR1` at both ends, footer length <= `64 KiB +
  1 KiB x ceil(n_mol / 2^14)`, and the library's own Thrift limits set.
  Seed: `parquet_footer_oversize` (generated).
* **S4. Parquet row groups are at most 2^14 rows, at most
  `ceil(n_mol / 2^14)` of them, no statistics, and the reader reads one
  column chunk at a time** (section 11; limits 21, 34). Measured: one legal
  2^16-row group at the string caps, read whole, was measured at 846 MiB
  above the interpreter; at 2^14 rows, column by column, 202-211 MiB. Writers write
  row groups of 2^14 rows.
* **S5. Limit 28's depth is a real parenthesis stack** (section 15 rule 5):
  every `(` is pushed, marked recursion or not, every `)` pops. A counter
  that skips branch parentheses reports `[$(*(*)[$(*(*)...)])]` as depth 1
  at any depth. Code: `reference.smarts_structure`. Seed:
  `definition_branch_depth`.
* **S6. Corpus files are opened relative to directory descriptors**
  (section 1 item 2): root `O_DIRECTORY` (a symlinked root is followed),
  `groups/` `O_DIRECTORY | O_NOFOLLOW` relative to the root fd, each file
  `O_NOFOLLOW | O_NONBLOCK` relative to its directory fd, every name a single
  derived component; no `realpath` containment check. Code:
  `reference.CorpusDir`. Seed: `groups_dir_symlink` (generated).
* **S7. Reads are bounded `pread`s; a short read is `io.short_read`; `mmap`
  only with `SIGBUS` handled** (section 1 item 4, section 17 item 12). Code:
  `reference._pread_exact`, `GroupFile.read`. Test:
  `TestFileAccess::test_a_file_truncated_after_open_is_a_refusal_not_sigbus`.
* **S8. Integer widths** (section 17 item 11): every quantity's maximum at
  the limits fits its type; JSON integers that do not fit int64 are refused
  (`schema.value`); limit 20 (`n_feat`, `n_feat_atoms` <= 2^32 - 1) is
  checked explicitly at `open` (`limit.n_feat_atoms`) rather than only
  implied by the u32 offsets; the global conformer id reaches 2^40 and is
  64-bit everywhere.
* **S9. Bounded validator memory at the limits** (limit 34): the `scan`
  tier's feature-table checks run in chunks of 2^16 features (a legal
  103 MiB group at limits 15-17 peaks at 163 MiB chunked, 2.8 GiB
  unchunked); `deep` hashes sections and blocks in pieces and unpacks only
  each axis's last packing group for the tail check (a 100 MB block: 27 MiB
  instead of 518 MiB). Limit 15 is implied by limit 16 and not counted
  separately.

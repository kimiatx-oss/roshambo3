# Format 3 conformance corpus and adversarial seeds

The committed files here are the oracle for every implementation of
`doc/FORMAT.md`. `tests/test_format3_reference.py` checks the
pure-Python reference (`src/roshambo3/format/r3g/reference.py`) against them;
the native codec, the writer/reader/validator and the fuzzers test
against the same files.

| path | what |
|---|---|
| `shape/` | a shape-only corpus: 47 molecules, 4,210 conformers, 12 groups over 9 buckets |
| `color/` | the same molecules with RDKit `BaseFeatures` color (47 molecules, 4,206 conformers, 12 groups) |
| `EXPECTED.json` | per corpus: fingerprint, `ids_sha256`, `corpus.json` digest; per group: file digest, widths, block rows, and the SHA-256 of every decoded array (padded heavy coordinates, merged kernel layout, derived pseudo-atoms, counts, run table, feature tables, `vaa`, `color_self`) |
| `make_conformance.py` | regenerates `shape/`, `color/` and `EXPECTED.json` (provenance; needs RDKit) |
| `widths/` | per-width regression-oracle blocks (encoded by the reference encoder itself and cross-checked against an AVX2 prototype, not an independent truth) `w01..w07.bin`, `w14..w16.bin` (one row of 16 atoms; the corpora cover widths 0 and 8-13) and `wrap.bin` (a tampered `min_x` so `min + u` wraps); `widths/EXPECTED.json` gives each block's `n_real`, `L`, widths and decoded digest |
| `adversarial/<name>/` | one minimal corpus per threat-model row (and per review finding), each wrong in exactly one way: 93 seeds, 79 committed; the pure pre-pass seeds are bare `definition.json` files (`generator`, `definition`, `types`) handed straight to `check_definition_text`, as the trust path does |
| `repro/malformed_footer_{shape,color}.parquet` | two `molecules.parquet` files with a hostile Parquet footer (found in review) that once aborted the reader; `tests/test_format3_fix_round.py` puts each behind valid groups and requires a typed `value.parquet` refusal |
| `adversarial/EXPECTED.json` | per seed: the validator tier that must refuse it, the refusal code, the threat-model row, whether it is generated |
| `adversarial/make_adversarial.py` | regenerates the seeds; `build(name, dir)` materializes the 14 `generated` seeds (too large to commit, or a FIFO / symlink that cannot be committed); `definition_oversize_prepass` is a bare `definition.fdef` for the pre-pass, not a corpus |

**Digest definitions** (`EXPECTED.json["<variant>"]["groups"][<name>]`),
each the SHA-256 of the C-contiguous little-endian bytes of:

| key | array |
|---|---|
| `heavy_sha256` | `int16[n_conf, 3, L_heavy]`: every block decoded, rows in group order, per row x[L_heavy] y[L_heavy] z[L_heavy], zeros beyond `n_real` (FORMAT.md 4.3) |
| `merged_sha256` | `int16[n_conf, 3, L]`: the kernel layout of 8.4 (real atoms, derived pseudo-atoms in stored feature order, zeros) |
| `pseudo_sha256` | `int16[N, 3]`: each row's pseudo-atoms `[n_color, 3]` (x, y, z per pseudo-atom) concatenated over rows in row order; `N = sum over rows of n_color` |
| `counts_sha256` | `uint8[n_conf, n_types + 1]`: the count vector of 8.5 per row |
| `runs_sha256`, `vaa_sha256`, `color_self_sha256`, `feature_*_sha256` | the stored section's `length` data bytes, exactly as in the file |
| `file_sha256` | the whole group file |

**Provenance** (`EXPECTED.json["provenance"]`): RDKit 2026.09.1, NumPy 2.5.4,
pyarrow 26.0.0, Python 3.13, `block_rows = 16`, `max_conf_per_group = 40`,
seed 20261017. ETKDG embedding and canonical atom order are RDKit-release
properties, and the Parquet footer carries the pyarrow version, so a
regeneration is byte-identical only under those versions; under any other
the committed files still are the oracle.

**Self-overlaps** were computed with the fp64 references,
`prep.conformer.stored_self_volume` (`vaa`) and
`reference.overlap.color_overlap` (`color_self`), from the stored integers
dequantized at 2^-8 A -- not with the native CPU path. FORMAT.md section 18
allows either within its tolerance.

**SMILES** are non-isomeric (FORMAT.md 11.1); `molecules.parquet` is
`ZSTD` level 9, `PLAIN`, non-null, no statistics, no embedded Arrow schema
(section 11), so its bytes depend on pyarrow only through the footer's
`created_by`.

**Synthetic molecules**: `chain255` (255 heavy atoms, a seeded 3-D random
walk; `n_real = 255`, and in `color/` 253 Hydrophobe features, so the device
width is 512), `ethane4096` (color: two atoms on the x axis, exactly 4,096
conformers, one block of exactly 4,096 rows, `w = 0` on y and z) and `one4096`
(shape: one atom, 4,096 conformers, a single-molecule block at the 4,096-row
cap, `w = 0` on every axis). Every other molecule is ETKDG-embedded.

Size: the directory (committed files, no `__pycache__`) is under the 512 KiB
budget; `test_directory_is_within_the_byte_budget` enforces it.

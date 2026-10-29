// Inputs must remain unchanged for the duration of a call. Output byte ranges
// must not overlap inputs or one another; the Python bindings enforce this.
// Corpus format 3 (doc/FORMAT.md): the native BP256-16 coordinate codec.
//
// Pure C++17, no Python and no pybind11: the libFuzzer harness links this
// header's implementation (r3g_codec.cpp, r3g_unpack_*.cpp, r3g_encode.cpp)
// standalone. The pybind11 layer is r3g_codec_bindings.cpp.
//
// The decoder contract is FORMAT.md section 17. Every entry point:
//   * takes every buffer with its length (inputs in bytes or elements as
//     named, outputs in ELEMENTS of the output type);
//   * checks the explicit shape arguments (rows, L_heavy, L, n_types) against
//     the limits table (FORMAT.md section 14) before any arithmetic;
//   * computes every size with __builtin_{mul,add}_overflow, never trusting a
//     byte of the block for a size. Integer widths follow FORMAT.md section 17
//     item 11 / FORMAT.md section 14.2.1: the run slice's row and
//     atom sums are accumulated in u64 (a block's total is <= 1,044,480, a
//     group's <= 4,278,190,080), every size is a checked size_t; the codec never
//     handles a global conformer id (the callers keep it 64-bit);
//   * returns a Status; never throws, never aborts, never asserts, never
//     writes outside [out, out + out_capacity) nor reads outside
//     [in, in + in_len). On error the output contents are unspecified (they
//     may be partially written, always within the buffer).
//
// Layouts written (FORMAT.md 4.3 and 8.4; consumers in engine/buffers.py and
// cpu_block_prepare.cpp):
//   heavy   int16 [rows][3][L_heavy]   per row x[L] y[L] z[L], zeros past n_real
//   merged  int16 [rows][3][L]         real atoms, derived pseudo-atoms in stored
//                                      feature order, zeros
//   gpu     int16 tiled, n_rows = ceil(rows / 32) * 32, of the merged record:
//             tile_mode  1: [n_rows/32][L][32][3]   (interleaved, default)
//             tile_mode -1: [n_rows/32][L][3][32]   (planar)
//           padding lanes and slots zeroed; values are stored int16 grid units
//   cpu     float32 [rows][L][3], float(v) * coord_scale, zeros past
//           n_real + n_color (cpu_block_prepare.cpp::prepare_cpu_block_coords)
//
// Alignment: no INPUT pointer needs any alignment (block bytes, run records,
// feature arrays and self-overlap sections are all read with memcpy, so a
// slice of an arbitrary file buffer is accepted). OUTPUT pointers must be
// naturally aligned for their element type (the binding checks).
//
// Allocation: the reference/baseline decoders allocate unpack scratch, ceil(total/256)
// * 256 int16 per axis plus the block's derived pseudo-atoms, sized only from
// checked values (at most ~6.3 MB for a 4,096-row block of 255-atom rows,
// typically < 1 MB), kept per thread between calls up to 16 MiB per buffer
// (three buffers), freed after a call that needed more. The caller's
// output buffers are never reallocated. The explicitly selected fused CPU/GPU
// writers retain the same axis scratch, but use no full-block pseudo buffer:
// their only coordinate stage is int16 [L][3][32] (<= 98,304 bytes), or the
// caller's planar GPU output itself. One molecule's feature cache crosses tiles.
//
// Thread safety: every function is reentrant. The forced-ISA override is a
// process-wide setting meant for tests.
#pragma once

#include <cstddef>
#include <cstdint>

namespace r3 {
namespace r3g {

// Fixed constants of FORMAT.md sections 2, 5, 6, 7, 14.
constexpr std::size_t kBlockHeaderSize = 32;
constexpr std::size_t kRunRecordSize = 16;
constexpr std::size_t kGroupValues = 256;     // values per BP256-16 packing group
constexpr unsigned kMaxBitWidth = 16;
constexpr std::size_t kMaxNReal = 255;
constexpr std::size_t kMaxLHeavy = 256;
constexpr std::size_t kMaxL = 512;
constexpr std::size_t kMaxNTypes = 6;
constexpr std::size_t kMaxAtomsPerFeature = 16;
constexpr std::size_t kMaxFeatureCount = 255;      // per type and per molecule
constexpr std::size_t kMaxBlockRowsMulti = 4096;   // a block of >= 2 molecules
constexpr std::size_t kMaxBlockRowsSingle = 4096; // a block of one molecule
constexpr std::size_t kTileRows = 32;              // layout.BTILE / ROSHAMBO3_BTILE

// Error codes. status_code() maps each onto FORMAT.md section 16.5's refusal
// names where they overlap, so a fuzzing oracle can compare the Python and
// native refusals.
enum class Status : int {
    kOk = 0,
    kBadArgument = 1,          // null pointer with nonzero length, bad tile mode, bad scale
    kRowsZero = 2,             // a block with zero rows            layout.blocks
    kRowsMismatch = 3,         // run slice does not sum to rows    layout.runs
    kBlockRows = 4,            // rows over the block cap           limit.block_rows
    kMoleculesZero = 5,        // empty run slice                   layout.blocks
    kNConf = 6,                // a molecule with n_conf == 0       limit.n_conf_per_molecule
    kNReal = 7,                // a molecule with n_real == 0       limit.n_real
    kLHeavy = 8,               // L_heavy outside [1, 256]          limit.L_heavy
    kL = 9,                    // L outside [1, 512]                limit.L
    kNTypes = 10,              // n_types outside [0, 6]            limit.n_types
    kRowExceedsWidth = 11,     // n_real > L_heavy or n_real + n_color > L   layout.widths
    kTruncated = 12,           // in_len < 32 (no complete header)  limit.block_length
    kBlockLength = 13,         // in_len != block_length            limit.block_length
    kBitWidth = 14,            // a width byte > 16                 limit.bit_width
    kBlockHeaderReserved = 15, // reserved header bytes not zero    layout.block_header
    kOutputTooSmall = 16,      // an output capacity below the need decode.output_capacity
    kSizeOverflow = 17,        // a size product overflows size_t   decode.overflow
    kFeatureOffsets = 18,      // CSR offsets out of range / not monotone   layout.features
    kFeatureType = 19,         // type outside [1, n_types]         layout.features
    kFeatureAtomIndex = 20,    // atom index >= n_real              layout.features
    kAtomsPerFeature = 21,     // atoms per feature outside [1, 16] limit.atoms_per_feature
    kFeaturesPerType = 22,     // > 255 features of one type        limit.features_per_type
    kFeaturesPerMolecule = 23, // > 255 features in a molecule      limit.features_per_molecule
    kFeaturesMissing = 24,     // n_types > 0 without a feature slice, or the reverse
    kSelfOverlapRange = 25,    // first_row + rows past a self-overlap section
    kAllocation = 26,          // scratch allocation failed
    kIsaUnavailable = 27,      // a forced ISA that this build does not contain
    kEncodeRange = 28,         // encoder: an input value range the format cannot hold
    kShortRead = 29,           // read_exact: the file ended before `length` bytes  io.short_read
    kIoError = 30,             // read_exact: pread failed (errno)  io.read_error
};

// The FORMAT.md 16.5 refusal name (e.g. "limit.bit_width"), or a
// "decode.<name>" name for a condition 16.5 has no code for. Never null.
const char* status_code(Status s) noexcept;
// A one-line English description of the failed check. Never null.
const char* status_message(Status s) noexcept;

// --- ISA selection --------------------------------------------------------
enum class Isa : int { kAuto = 0, kScalar = 1, kAvx2 = 2, kNeon = 3 };
bool isa_compiled(Isa isa) noexcept;      // is this unpack path in the build
// The path in use. On first use it is read once from ROSHAMBO3_R3G_ISA
// (scalar|avx2|neon|auto; unset or auto = the default, AVX2 on x86-64, NEON
// on aarch64); force_isa overrides it afterwards.
Isa active_isa() noexcept;
Status force_isa(Isa isa) noexcept;       // kAuto restores the default
const char* isa_name(Isa isa) noexcept;

// Private test/benchmark override. Auto enables the measured NEON GPU writer;
// CPU and unqualified ISA/consumer combinations retain baseline.
// Do not mutate this process-wide setting during concurrent calls.
enum class BuilderMode : int { kAuto = 0, kBaseline = 1, kFused = 2 };
Status set_builder_mode(BuilderMode mode) noexcept;
BuilderMode builder_mode() noexcept;
const char* builder_mode_name(BuilderMode mode) noexcept;
// Normal nonalias dispatch; native hostile aliases always retain baseline.
bool use_fused_builder(bool gpu) noexcept;

// --- Inputs ---------------------------------------------------------------

// The block's run-table slice: runs[m0 : m1] of the group's `runs` section,
// n_molecules 16-byte records (mol_id u64, n_conf u16, n_real u8, 5 reserved).
// Fields are read little-endian byte by byte; no alignment is required.
struct RunSlice {
    const std::uint8_t* records;
    std::size_t n_molecules;
};

// The block's feature-table slice (FORMAT.md 8.1, section 17 item 7).
// `offsets` is feature_offsets[m0 : m1 + 1] -- n_molecules + 1 values, NOT
// rebased: they index the group-wide arrays below, which are passed whole.
struct FeatureSlice {
    const std::uint32_t* offsets;      std::size_t offsets_len;
    const std::uint8_t* types;         std::size_t types_len;         // feature_types
    const std::uint32_t* atom_offsets; std::size_t atom_offsets_len;  // feature_atom_offsets
    const std::uint8_t* atoms;         std::size_t atoms_len;         // feature_atoms
};

// One block call's inputs. rows, L_heavy, L and n_types come from the
// (validated) group table or group header, never from the block bytes.
struct BlockInput {
    const std::uint8_t* in;    // the block's bytes, header included
    std::size_t in_len;
    RunSlice runs;
    std::size_t rows;
    std::size_t L_heavy;
    std::size_t L;             // device width; ignored by decode_heavy
    std::size_t n_types;       // 0 for a shape-only corpus
    FeatureSlice features;     // all null/0 when n_types == 0
};

// The self-overlap sections, whole (FORMAT.md 18). color_self may be absent
// (null with length 0): the binding then reports it absent, not an error.
struct SelfOverlapInput {
    const float* vaa;          std::size_t vaa_len;
    const double* color_self;  std::size_t color_self_len;
    std::size_t first_row;     // the block table's first_row
};

// Wall-clock seconds of one block call.
struct BlockTimings {
    double decode = 0.0;        // baseline: checks, unpack, layout; fused: checks + unpack
    double color = 0.0;         // baseline: feature checks/derive; fused: derive + emit
    double self_overlap = 0.0;  // the vaa / color_self binding
};

// --- Sizes ----------------------------------------------------------------

// block_length of FORMAT.md 5.1 for `total` real atoms and the three widths.
// Widths are checked <= 16 first.
Status block_length(std::size_t total, unsigned w_x, unsigned w_y, unsigned w_z,
                    std::size_t* out_len) noexcept;
// The GPU layout's padded row count, ceil(rows / 32) * 32, and element count
// n_rows * L * 3.
Status gpu_layout_elements(std::size_t rows, std::size_t L,
                           std::size_t* n_rows, std::size_t* elements) noexcept;

// --- Decoders ---------------------------------------------------------------

// The padded heavy layout int16 [rows][3][L_heavy] (FORMAT.md 4.3), the
// array the conformance `heavy_sha256` digest is over.
Status decode_heavy(const BlockInput& block, std::int16_t* out,
                    std::size_t out_capacity) noexcept;

// The merged record int16 [rows][3][L] (FORMAT.md 8.4) and, when counts is
// non-null, the count vectors uint8 [rows][n_types + 1] (8.5).
Status decode_merged(const BlockInput& block, std::int16_t* out,
                     std::size_t out_capacity, std::uint8_t* counts,
                     std::size_t counts_capacity,
                     BlockTimings* timings) noexcept;

// GPU writer: the tiled int16 layout engine/buffers.py::pack_into writes
// (tile_mode 1 interleaved, -1 planar) of the merged record. Optional outputs
// (null to skip): counts uint8 [rows][n_types + 1]; count_words uint64 [rows]
// in the device's count word layout (byte 0 n_real, bytes 1..6 per-type
// counts, zero past n_types, byte 7 n_color; kernels.cuh r3_counts_t).
Status write_gpu_block(const BlockInput& block, int tile_mode,
                       std::int16_t* out, std::size_t out_capacity,
                       std::uint8_t* counts, std::size_t counts_capacity,
                       std::uint64_t* count_words, std::size_t count_words_capacity,
                       BlockTimings* timings) noexcept;

// GPU launch writer: consecutive whole blocks with common widths/types. Tiles
// are relative to the launch, including when an interior block has a short tile.
// No intermediate device-layout buffer; counts and signed molecule IDs are
// expanded directly into caller-owned per-launch arrays. All capacities and all
// constituent metadata are checked before the first output write.
Status write_gpu_launch(const BlockInput* blocks, std::size_t n_blocks,
                        std::size_t rows, int tile_mode,
                        std::int16_t* out, std::size_t out_capacity,
                        std::uint8_t* counts, std::size_t counts_capacity,
                        std::int64_t* mol_ids, std::size_t mol_ids_capacity) noexcept;

// CPU writer: float32 [rows][L][3], float(v) * coord_scale, zeros past
// n_real + n_color -- prepare_cpu_block_coords' layout. coord_scale must be
// finite and > 0. Optional counts / count_words as for the GPU writer.
Status write_cpu_block(const BlockInput& block, float coord_scale,
                       float* out, std::size_t out_capacity,
                       std::uint8_t* counts, std::size_t counts_capacity,
                       std::uint64_t* count_words, std::size_t count_words_capacity,
                       BlockTimings* timings) noexcept;

// Self-overlap binding: copies vaa[first_row : first_row + rows] into vaa_out
// and, when the section is present and color_self_out non-null,
// color_self[first_row : first_row + rows] into color_self_out. Each present
// section must hold first_row + rows values (overflow-checked).
// *color_self_present reports whether color_self was bound.
Status bind_self_overlaps(const SelfOverlapInput& sections, std::size_t rows,
                          float* vaa_out, std::size_t vaa_out_capacity,
                          double* color_self_out, std::size_t color_self_out_capacity,
                          bool* color_self_present) noexcept;

// --- Reads (FORMAT.md section 17 item 12, Appendix B S7) ----------------------

// The codec itself does no I/O: every decoder takes bytes. A caller that
// reads a block MUST use
// bounded preads and refuse a short read, or mmap only with SIGBUS converted
// into an error (FORMAT.md section 1 item 4). read_exact is that bounded
// pread, for native callers: it reads exactly `length` bytes at `offset` of
// `fd` into out[0, length), retrying on EINTR and on partial reads, and
// returns kShortRead if the file ends first (a file truncated after it was
// opened), kIoError on any other failure, kSizeOverflow if offset + length
// does not fit a signed 64-bit file offset. POSIX only (r3g_io.cpp; not part
// of the standalone fuzzing core).
Status read_exact(int fd, std::uint64_t offset, std::uint8_t* out, std::size_t length) noexcept;

// --- Encoder ----------------------------------------------------------------

// Encodes int16 [rows][3][L_heavy] (padding ignored) for the run slice into
// the exact bytes of FORMAT.md 5.5. *out_len receives the encoded length; if
// out_capacity is smaller the call returns kOutputTooSmall with *out_len set
// (out may be null to query the length).
Status encode_block(const std::int16_t* coords, std::size_t coords_len,
                    RunSlice runs, std::size_t rows, std::size_t L_heavy,
                    std::uint8_t* out, std::size_t out_capacity,
                    std::size_t* out_len) noexcept;

}  // namespace r3g
}  // namespace r3

// Format-3 codec core: checks, ISA dispatch, derived color and the writers.
// The contract is r3g_codec.hpp and doc/FORMAT.md section 17.
//
// Every call follows one order: (1) the explicit shape arguments against the
// limits table; (2) the run slice, molecule by molecule, with checked sums;
// (3) the block header (widths before anything else reads them, reserved
// bytes), then in_len against the closed-form block length; (4) the feature
// slice, offsets before they index anything; (5) output capacities. Only then
// is a stream byte read or an output element written.
#include "r3g_codec.hpp"

#include "r3g_unpack.hpp"
#include "r3g_build.hpp"

#include <algorithm>
#include <array>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace r3 {
namespace r3g {
namespace {

using Clock = std::chrono::steady_clock;

inline double seconds_since(Clock::time_point t0) noexcept {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

inline bool mul(std::size_t a, std::size_t b, std::size_t* r) noexcept {
    return !__builtin_mul_overflow(a, b, r);
}
inline bool add(std::size_t a, std::size_t b, std::size_t* r) noexcept {
    return !__builtin_add_overflow(a, b, r);
}
inline bool mul64(std::uint64_t a, std::uint64_t b, std::uint64_t* r) noexcept {
    return !__builtin_mul_overflow(a, b, r);
}
inline bool add64(std::uint64_t a, std::uint64_t b, std::uint64_t* r) noexcept {
    return !__builtin_add_overflow(a, b, r);
}
// A u64 quantity as a size_t, refusing what does not fit (32-bit hosts).
inline bool to_size(std::uint64_t v, std::size_t* r) noexcept {
    if (v > static_cast<std::uint64_t>(SIZE_MAX)) return false;
    *r = static_cast<std::size_t>(v);
    return true;
}

inline std::uint32_t rd_u16(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8);
}
inline std::int16_t rd_i16(const std::uint8_t* p) noexcept {
    return detail::wrap16(0, rd_u16(p));
}
template <class T>
inline T load(const T* base, std::size_t i) noexcept {  // unaligned-safe element read
    T v;
    std::memcpy(&v, reinterpret_cast<const unsigned char*>(base) + i * sizeof(T), sizeof(T));
    return v;
}

// ---- ISA selection ---------------------------------------------------------

constexpr Isa kDefaultIsa =
#if defined(__x86_64__) || defined(_M_X64)
    Isa::kAvx2;
#elif defined(__aarch64__) || defined(_M_ARM64)
    Isa::kNeon;
#else
    Isa::kScalar;
#endif

std::atomic<int> g_builder_mode{static_cast<int>(BuilderMode::kAuto)};
std::atomic<int> g_isa{-1};  // -1: not yet read from the environment

Isa isa_from_env() noexcept {
    const char* v = std::getenv("ROSHAMBO3_R3G_ISA");
    if (!v || !*v || std::strcmp(v, "auto") == 0) return kDefaultIsa;
    Isa want = kDefaultIsa;
    if (std::strcmp(v, "scalar") == 0) want = Isa::kScalar;
    else if (std::strcmp(v, "avx2") == 0) want = Isa::kAvx2;
    else if (std::strcmp(v, "neon") == 0) want = Isa::kNeon;
    // An unknown or uncompiled request falls back to the default rather than
    // failing every decode; force_isa() is the checked route.
    return isa_compiled(want) ? want : kDefaultIsa;
}

// The unpack dispatch: a switch on the (already resolved) ISA, then each
// kernel's own switch on the width. Neither indexes anything with a value
// from the file.
bool unpack_axis(Isa isa, const std::uint8_t* s, std::size_t groups, unsigned w,
                 std::int16_t mn, std::int16_t* dst) noexcept {
    if (w > kMaxBitWidth) return false;
    switch (isa) {
        case Isa::kScalar:
            return detail::unpack_axis_scalar(s, groups, w, mn, dst);
#if defined(__x86_64__) || defined(_M_X64)
        case Isa::kAvx2:
            return detail::unpack_axis_avx2(s, groups, w, mn, dst);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
        case Isa::kNeon:
            return detail::unpack_axis_neon(s, groups, w, mn, dst);
#endif
        default:
            return false;
    }
}

// ---- Scratch ---------------------------------------------------------------

// Bytes of each scratch buffer a thread keeps between calls. 16 MiB covers
// every block of >= 2 molecules (<= 4,096 rows x 255 atoms: ~6.3 MB of axis
// scratch, ~6.3 MB of pseudo-atoms at 255 features); the rare larger
// single-molecule block frees its scratch when it returns, so a pool of N
// worker threads retains at most ~3 x 16 MiB x N.
constexpr std::size_t kScratchRetain = std::size_t(16) << 20;
constexpr std::size_t kSlack = kMaxL;  // int16 past each axis, for row reads

struct Scratch {
    std::vector<std::int16_t> axes;    // 3 x (groups * 256 + kSlack)
    std::vector<std::int16_t> pseudo;  // [n_pseudo][3]
    std::vector<std::int16_t> stage;   // a tile or a row of the merged record
    std::vector<detail::RowRun> row_runs;
    void trim() noexcept {
        if (axes.capacity() * 2 > kScratchRetain) std::vector<std::int16_t>().swap(axes);
        if (pseudo.capacity() * 2 > kScratchRetain) std::vector<std::int16_t>().swap(pseudo);
        if (stage.capacity() * 2 > kScratchRetain) std::vector<std::int16_t>().swap(stage);
    }
};
thread_local Scratch t_scratch;

template <class T>
bool ensure(std::vector<T>& v, std::size_t n) noexcept {
    try {
        if (v.size() < n) v.resize(n);
        return true;
    } catch (...) {
        return false;
    }
}

// ---- The checked block -----------------------------------------------------

struct Molecule {
    std::size_t n_conf;
    std::size_t n_real;
    std::size_t n_color;     // 0 when shape-only
    std::size_t f0;          // first feature (group-wide index)
    std::uint8_t type_counts[kMaxNTypes + 1];  // [0] unused
};

struct Checked {
    std::size_t rows = 0, total = 0, groups = 0;
    unsigned w[3] = {0, 0, 0};
    std::int16_t mn[3] = {0, 0, 0};
    const std::uint8_t* stream[3] = {nullptr, nullptr, nullptr};
    std::vector<Molecule> mols;
    std::size_t n_pseudo = 0;   // sum over molecules of n_conf * n_color
};

Status check_shape(const BlockInput& b, bool need_device_width) noexcept {
    if ((b.in == nullptr && b.in_len != 0) ||
        (b.runs.records == nullptr && b.runs.n_molecules != 0))
        return Status::kBadArgument;
    if (b.rows == 0) return Status::kRowsZero;
    if (b.L_heavy == 0 || b.L_heavy > kMaxLHeavy) return Status::kLHeavy;
    if (need_device_width && (b.L == 0 || b.L > kMaxL)) return Status::kL;
    if (b.n_types > kMaxNTypes) return Status::kNTypes;
    if (b.runs.n_molecules == 0) return Status::kMoleculesZero;
    if (b.rows > kMaxBlockRowsSingle ||
        (b.runs.n_molecules >= 2 && b.rows > kMaxBlockRowsMulti))
        return Status::kBlockRows;
    return Status::kOk;
}

// Steps (2) and (3): the run slice and the block bytes.
Status check_runs_and_block(const BlockInput& b, Checked* c) noexcept {
    c->rows = b.rows;
    // A run slice longer than the rows can be is refused before it is walked
    // (every molecule has n_conf >= 1).
    if (b.runs.n_molecules > b.rows) return Status::kRowsMismatch;
    try {
        c->mols.resize(b.runs.n_molecules);
    } catch (...) {
        return Status::kAllocation;
    }
    // Row and atom sums in u64 (FORMAT.md 17 item 11), checked anyway.
    std::uint64_t rows = 0, total64 = 0;
    for (std::size_t m = 0; m < b.runs.n_molecules; ++m) {
        const std::uint8_t* rec = b.runs.records + m * kRunRecordSize;
        const std::size_t n_conf = rd_u16(rec + 8);
        const std::size_t n_real = rec[10];
        if (n_conf == 0 || n_conf > 4096) return Status::kNConf;
        if (n_real == 0) return Status::kNReal;
        if (n_real > b.L_heavy) return Status::kRowExceedsWidth;
        std::uint64_t atoms = 0;
        if (!add64(rows, n_conf, &rows) || !mul64(n_conf, n_real, &atoms) ||
            !add64(total64, atoms, &total64))
            return Status::kSizeOverflow;
        if (rows > b.rows) return Status::kRowsMismatch;
        Molecule& mol = c->mols[m];
        mol.n_conf = n_conf;
        mol.n_real = n_real;
        mol.n_color = 0;
        mol.f0 = 0;
        std::memset(mol.type_counts, 0, sizeof mol.type_counts);
    }
    if (rows != b.rows) return Status::kRowsMismatch;
    std::size_t total = 0;
    if (!to_size(total64, &total)) return Status::kSizeOverflow;
    c->total = total;
    c->groups = total / kGroupValues + (total % kGroupValues != 0 ? 1 : 0);

    if (b.in_len < kBlockHeaderSize) return Status::kTruncated;
    for (int a = 0; a < 3; ++a) {
        const unsigned w = b.in[6 + a];
        if (w > kMaxBitWidth) return Status::kBitWidth;
        c->w[a] = w;
        c->mn[a] = rd_i16(b.in + 2 * a);
    }
    for (std::size_t i = 9; i < kBlockHeaderSize; ++i)
        if (b.in[i] != 0) return Status::kBlockHeaderReserved;
    std::size_t need = 0;
    Status s = block_length(total, c->w[0], c->w[1], c->w[2], &need);
    if (s != Status::kOk) return s;
    if (b.in_len != need) return Status::kBlockLength;
    const std::uint8_t* p = b.in + kBlockHeaderSize;
    for (int a = 0; a < 3; ++a) {
        c->stream[a] = p;
        p += c->groups * c->w[a] * 32;  // within in_len: the sum is `need`
    }
    return Status::kOk;
}

// Step (4): the feature slice (FORMAT.md 17 item 7). Fills each molecule's
// n_color, first feature and per-type counts, and the block's pseudo count.
Status check_features(const BlockInput& b, std::size_t width, Checked* c) noexcept {
    const FeatureSlice& f = b.features;
    const std::size_t n_mol = b.runs.n_molecules;
    if (b.n_types == 0) {
        if (f.offsets != nullptr || f.offsets_len != 0) return Status::kFeaturesMissing;
        c->n_pseudo = 0;
        for (const Molecule& mol : c->mols)
            if (mol.n_real > width) return Status::kRowExceedsWidth;
        return Status::kOk;
    }
    if (f.offsets == nullptr) return Status::kFeaturesMissing;
    if ((f.types == nullptr && f.types_len) || (f.atom_offsets == nullptr && f.atom_offsets_len) ||
        (f.atoms == nullptr && f.atoms_len))
        return Status::kBadArgument;
    if (f.offsets_len != n_mol + 1) return Status::kFeatureOffsets;
    // Offsets first: in range of the array they index, and monotone.
    for (std::size_t i = 0; i <= n_mol; ++i) {
        const std::size_t o = load(f.offsets, i);
        if (o > f.types_len) return Status::kFeatureOffsets;
        if (i && o < load(f.offsets, i - 1)) return Status::kFeatureOffsets;
    }
    const std::size_t f_end = load(f.offsets, n_mol);
    // feature_atom_offsets[f + 1] must exist for every referenced f < f_end.
    if (f_end >= f.atom_offsets_len && f_end > load(f.offsets, 0)) return Status::kFeatureOffsets;
    std::size_t n_pseudo = 0;
    for (std::size_t m = 0; m < n_mol; ++m) {
        Molecule& mol = c->mols[m];
        const std::size_t f0 = load(f.offsets, m), f1 = load(f.offsets, m + 1);
        const std::size_t n_color = f1 - f0;
        if (n_color > kMaxFeatureCount) return Status::kFeaturesPerMolecule;
        unsigned counts[kMaxNTypes + 1] = {0, 0, 0, 0, 0, 0, 0};
        for (std::size_t ft = f0; ft < f1; ++ft) {
            const std::size_t t = f.types[ft];
            if (t < 1 || t > b.n_types) return Status::kFeatureType;
            const std::size_t a0 = load(f.atom_offsets, ft), a1 = load(f.atom_offsets, ft + 1);
            if (a1 > f.atoms_len || a1 <= a0) return Status::kFeatureOffsets;
            if (a1 - a0 > kMaxAtomsPerFeature) return Status::kAtomsPerFeature;
            for (std::size_t a = a0; a < a1; ++a)
                if (f.atoms[a] >= mol.n_real) return Status::kFeatureAtomIndex;
            // Subsumed by the total check above (each type's count is at most
            // the total); kept as the u8 vector's own guard.
            if (++counts[t] > kMaxFeatureCount) return Status::kFeaturesPerType;
        }
        if (mol.n_real + n_color > width) return Status::kRowExceedsWidth;
        mol.n_color = n_color;
        mol.f0 = f0;
        for (std::size_t t = 1; t <= kMaxNTypes; ++t)
            mol.type_counts[t] = static_cast<std::uint8_t>(counts[t]);
        std::size_t p = 0;
        if (!mul(mol.n_conf, n_color, &p) || !add(n_pseudo, p, &n_pseudo))
            return Status::kSizeOverflow;
    }
    c->n_pseudo = n_pseudo;
    return Status::kOk;
}

// Unpack the three axes into scratch: axis a at axes[a * stride], stride =
// groups * 256 + kSlack, the slack zeroed so a row read never leaves the
// scratch and never reads an indeterminate value.
Status unpack_all(const Checked& c, std::vector<std::int16_t>& axes,
                  std::size_t* stride_out) noexcept {
    std::size_t vals = 0, stride = 0, n = 0;
    if (!mul(c.groups, kGroupValues, &vals) || !add(vals, kSlack, &stride) ||
        !mul(stride, 3, &n))
        return Status::kSizeOverflow;
    if (!ensure(axes, n)) return Status::kAllocation;
    const Isa isa = active_isa();
    for (int a = 0; a < 3; ++a) {
        std::int16_t* dst = axes.data() + a * stride;
        if (!unpack_axis(isa, c.stream[a], c.groups, c.w[a], c.mn[a], dst))
            return Status::kBitWidth;  // unreachable: widths were checked
        std::memset(dst + vals, 0, kSlack * sizeof(std::int16_t));
    }
    *stride_out = stride;
    return Status::kOk;
}

// sign(s) * ((2|s| + k) // (2k)), FORMAT.md 8.3. |s| <= 16 * 32768.
inline std::int16_t mean_half_away(std::int32_t s, std::int32_t k) noexcept {
    if (s == 0) return 0;
    const std::int32_t mag = s < 0 ? -s : s;
    const std::int32_t q = (2 * mag + k) / (2 * k);
    return static_cast<std::int16_t>(s < 0 ? -q : q);
}

// Derive every pseudo-atom of the block into pseudo[n_pseudo][3], rows in
// order, features in stored order; fill counts / count words.
Status derive_color_guarded(const BlockInput& b, const Checked& c, const std::int16_t* axes,
                    std::size_t stride, std::vector<std::int16_t>& pseudo,
                    std::uint8_t* counts, std::uint64_t* words) noexcept {
    const FeatureSlice& f = b.features;
    std::size_t n = 0;
    if (!mul(c.n_pseudo, 3, &n)) return Status::kSizeOverflow;
    if (!ensure(pseudo, n == 0 ? 1 : n)) return Status::kAllocation;
    std::int16_t* out = pseudo.data();
    const std::size_t nc = b.n_types + 1;
    std::size_t pos = 0, row = 0;
    for (const Molecule& mol : c.mols) {
        std::uint64_t word = mol.n_real;
        for (std::size_t t = 1; t <= kMaxNTypes; ++t)
            word |= static_cast<std::uint64_t>(mol.type_counts[t]) << (8 * t);
        word |= static_cast<std::uint64_t>(mol.n_color) << 56;
        for (std::size_t j = 0; j < mol.n_conf; ++j, ++row, pos += mol.n_real) {
            if (counts) {
                counts[row * nc] = static_cast<std::uint8_t>(mol.n_real);
                for (std::size_t t = 1; t < nc; ++t) counts[row * nc + t] = mol.type_counts[t];
            }
            if (words) words[row] = word;
            for (std::size_t ft = mol.f0; ft < mol.f0 + mol.n_color; ++ft) {
                const std::size_t a0 = load(f.atom_offsets, ft), a1 = load(f.atom_offsets, ft + 1);
                if (a0 >= a1 || a1 > f.atoms_len || a1 - a0 > 16)
                    return Status::kFeatureOffsets;
                const std::int32_t k = static_cast<std::int32_t>(a1 - a0);
                for (int ax = 0; ax < 3; ++ax) {
                    const std::int16_t* v = axes + ax * stride + pos;
                    std::int32_t s = 0;
                    for (std::size_t a = a0; a < a1; ++a) {
                        const std::size_t idx = f.atoms[a];
                        if (idx >= mol.n_real) return Status::kFeatureAtomIndex;
                        s += v[idx];
                    }
                    *out++ = mean_half_away(s, k);
                }
            }
        }
    }
    return Status::kOk;
}

// Exact feature arity is dispatched once, not once per conformer/axis.
// Integer sums fit int32 even at k=16 and both int16 extrema; a constant
// divisor permits exact compiler strength reduction, never a float reciprocal.
template <std::size_t K>
void derive_feature(const std::int16_t* axes, std::size_t stride, std::size_t pos,
                    const Molecule& mol, const std::uint8_t* atoms,
                    std::size_t feature, std::int16_t* out) noexcept {
    std::uint8_t indices[K];
    std::memcpy(indices, atoms, K);
    for (std::size_t j = 0; j < mol.n_conf; ++j) {
        const std::size_t base = pos + j * mol.n_real;
        std::int16_t* dst = out + (j * mol.n_color + feature) * 3;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const std::int16_t* v = axes + axis * stride + base;
            if constexpr (K == 1) {
                dst[axis] = v[indices[0]];
            } else {
                std::int32_t sum = 0;
                for (std::size_t a = 0; a < K; ++a) sum += v[indices[a]];
                dst[axis] = mean_half_away(sum, static_cast<std::int32_t>(K));
            }
        }
    }
}

// C++ callers promise non-overlapping inputs and outputs. Keep the original
// checked use-site path for a violation of that promise (the native defensive
// alias regression deliberately overwrites feature offsets through words).
// Subtraction avoids overflowing pointer endpoints; oversized spans conservatively
// select the guarded path instead of forming an overflowing byte count.
bool overlaps(const void* a, std::size_t an, const void* b, std::size_t bn) noexcept {
    if (!a || !b || !an || !bn) return false;
    const auto ap = reinterpret_cast<std::uintptr_t>(a);
    const auto bp = reinterpret_cast<std::uintptr_t>(b);
    return ap <= bp ? bp - ap < an : ap - bp < bn;
}
bool metadata_aliases_features(const BlockInput& b, std::uint8_t* counts,
                              std::uint64_t* words) noexcept {
    std::size_t cn = 0, wn = 0;
    if (!mul(b.rows, b.n_types + 1, &cn) || !mul(b.rows, sizeof(*words), &wn)) return true;
    if (overlaps(counts, cn, words, wn)) return true;
    const FeatureSlice& f = b.features;
    const void* ptrs[] = {f.offsets, f.types, f.atom_offsets, f.atoms};
    const std::size_t sizes[] = {f.offsets_len, f.types_len, f.atom_offsets_len, f.atoms_len};
    const std::size_t widths[] = {sizeof(*f.offsets), sizeof(*f.types), sizeof(*f.atom_offsets), sizeof(*f.atoms)};
    for (std::size_t i = 0; i < 4; ++i) {
        std::size_t bytes = 0;
        if (!mul(sizes[i], widths[i], &bytes)) return true;
        if (overlaps(counts, cn, ptrs[i], bytes) || overlaps(words, wn, ptrs[i], bytes)) return true;
    }
    return false;
}

Status derive_color(const BlockInput& b, const Checked& c, const std::int16_t* axes,
                    std::size_t stride, std::vector<std::int16_t>& pseudo,
                    std::uint8_t* counts, std::uint64_t* words) noexcept {
    if (metadata_aliases_features(b, counts, words))
        return derive_color_guarded(b, c, axes, stride, pseudo, counts, words);
    std::size_t n = 0;
    if (!mul(c.n_pseudo, 3, &n)) return Status::kSizeOverflow;
    if (!ensure(pseudo, n == 0 ? 1 : n)) return Status::kAllocation;
    const FeatureSlice& f = b.features;
    const std::size_t nc = b.n_types + 1;
    std::size_t pos = 0, row = 0, pseudo_pos = 0;
    for (const Molecule& mol : c.mols) {
        std::uint64_t word = mol.n_real;
        for (std::size_t t = 1; t <= kMaxNTypes; ++t)
            word |= static_cast<std::uint64_t>(mol.type_counts[t]) << (8 * t);
        word |= static_cast<std::uint64_t>(mol.n_color) << 56;
        for (std::size_t j = 0; j < mol.n_conf; ++j) {
            if (counts) {
                counts[(row + j) * nc] = static_cast<std::uint8_t>(mol.n_real);
                for (std::size_t t = 1; t < nc; ++t)
                    counts[(row + j) * nc + t] = mol.type_counts[t];
            }
            if (words) words[row + j] = word;
        }
        for (std::size_t ft = 0; ft < mol.n_color; ++ft) {
            const std::size_t a0 = load(f.atom_offsets, mol.f0 + ft);
            const std::size_t a1 = load(f.atom_offsets, mol.f0 + ft + 1);
            // One defensive check at each feature's use site, before any
            // gather; the immutable validated metadata is then reused by all rows.
            if (a0 >= a1 || a1 > f.atoms_len || a1 - a0 > kMaxAtomsPerFeature)
                return Status::kFeatureOffsets;
            for (std::size_t a = a0; a < a1; ++a)
                if (f.atoms[a] >= mol.n_real) return Status::kFeatureAtomIndex;
            std::int16_t* out = pseudo.data() + pseudo_pos;
#define R3_FEATURE_ARITY(K) case K: derive_feature<K>(axes, stride, pos, mol, f.atoms + a0, ft, out); break
            switch (a1 - a0) {
                R3_FEATURE_ARITY(1); R3_FEATURE_ARITY(2); R3_FEATURE_ARITY(3); R3_FEATURE_ARITY(4);
                R3_FEATURE_ARITY(5); R3_FEATURE_ARITY(6); R3_FEATURE_ARITY(7); R3_FEATURE_ARITY(8);
                R3_FEATURE_ARITY(9); R3_FEATURE_ARITY(10); R3_FEATURE_ARITY(11); R3_FEATURE_ARITY(12);
                R3_FEATURE_ARITY(13); R3_FEATURE_ARITY(14); R3_FEATURE_ARITY(15); R3_FEATURE_ARITY(16);
            }
#undef R3_FEATURE_ARITY
        }
        pos += mol.n_conf * mol.n_real;
        row += mol.n_conf;
        pseudo_pos += mol.n_conf * mol.n_color * 3;
    }
    return Status::kOk;
}

// One row of the merged record into dst[3][L]: real atoms from the axis
// scratch at `pos`, then n_color pseudo-atoms, then zeros.
inline void merged_row(const std::int16_t* axes, std::size_t stride, std::size_t pos,
                       std::size_t n_real, const std::int16_t* pseudo, std::size_t n_color,
                       std::size_t L, std::int16_t* dst) noexcept {
    for (int a = 0; a < 3; ++a) {
        std::int16_t* d = dst + a * L;
        std::memcpy(d, axes + a * stride + pos, n_real * sizeof(std::int16_t));
        for (std::size_t i = 0; i < n_color; ++i) d[n_real + i] = pseudo[3 * i + a];
        std::memset(d + n_real + n_color, 0, (L - n_real - n_color) * sizeof(std::int16_t));
    }
}

// The heavy layout's row writer for the active ISA (r3g_unpack.hpp): the
// scalar path is memcpy + memset, the SIMD paths masked whole-chunk copies.
void heavy_rows(Isa isa, const std::int16_t* axes, std::size_t stride,
                const detail::RowRun* runs, std::size_t n_runs, std::size_t L,
                std::int16_t* out) noexcept {
    switch (isa) {
#if defined(__x86_64__) || defined(_M_X64)
        case Isa::kAvx2:
            detail::heavy_rows_avx2(axes, stride, runs, n_runs, L, out);
            return;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
        case Isa::kNeon:
            detail::heavy_rows_neon(axes, stride, runs, n_runs, L, out);
            return;
#endif
        default:
            detail::heavy_rows_scalar(axes, stride, runs, n_runs, L, out);
            return;
    }
}

struct Common {
    Checked c;
    std::size_t stride = 0;
};

// Steps 1-4 for a device-width call, then the unpack; times decode and color.
Status prepare(const BlockInput& b, Common* k, Clock::time_point t0,
               BlockTimings* tm) noexcept {
    Status s = check_shape(b, true);
    if (s != Status::kOk) return s;
    s = check_runs_and_block(b, &k->c);
    if (s != Status::kOk) return s;
    if (tm) tm->decode += seconds_since(t0);
    const Clock::time_point t1 = Clock::now();
    s = check_features(b, b.L, &k->c);
    if (tm) tm->color += seconds_since(t1);
    return s;
}

Status check_count_outputs(const BlockInput& b, std::uint8_t* counts, std::size_t counts_cap,
                           std::uint64_t* words, std::size_t words_cap) noexcept {
    std::size_t need = 0;
    if (counts && (!mul(b.rows, b.n_types + 1, &need) || counts_cap < need))
        return Status::kOutputTooSmall;
    if (words && words_cap < b.rows) return Status::kOutputTooSmall;
    return Status::kOk;
}

Status finish(Status s) noexcept {
    t_scratch.trim();
    return s;
}

}  // namespace

// ---- Public ----------------------------------------------------------------

Status set_builder_mode(BuilderMode mode) noexcept {
    if (mode != BuilderMode::kAuto && mode != BuilderMode::kBaseline && mode != BuilderMode::kFused)
        return Status::kBadArgument;
    g_builder_mode.store(static_cast<int>(mode), std::memory_order_relaxed);
    return Status::kOk;
}
BuilderMode builder_mode() noexcept {
    return static_cast<BuilderMode>(g_builder_mode.load(std::memory_order_relaxed));
}
bool use_fused_builder(bool gpu) noexcept {
    const BuilderMode mode = builder_mode();
    if (mode != BuilderMode::kAuto) return mode == BuilderMode::kFused;
    // Five eligible paired runs per shape/color, layout and 1/8-worker group.
    // Only NEON GPU passed both prep gates.
    // The CPU tile path failed; no CPU scoring promotion is warranted.
    return gpu && active_isa() == Isa::kNeon;
}
const char* builder_mode_name(BuilderMode mode) noexcept {
    switch (mode) {
        case BuilderMode::kAuto: return "auto";
        case BuilderMode::kBaseline: return "baseline";
        case BuilderMode::kFused: return "fused";
    }
    return "unknown";
}

const char* status_code(Status s) noexcept {
    switch (s) {
        case Status::kOk: return "ok";
        case Status::kBadArgument: return "decode.argument";
        case Status::kRowsZero: return "layout.blocks";
        case Status::kRowsMismatch: return "layout.runs";
        case Status::kBlockRows: return "limit.block_rows";
        case Status::kMoleculesZero: return "layout.blocks";
        case Status::kNConf: return "limit.n_conf_per_molecule";
        case Status::kNReal: return "limit.n_real";
        case Status::kLHeavy: return "limit.L_heavy";
        case Status::kL: return "limit.L";
        case Status::kNTypes: return "limit.n_types";
        case Status::kRowExceedsWidth: return "layout.widths";
        case Status::kTruncated: return "limit.block_length";
        case Status::kBlockLength: return "limit.block_length";
        case Status::kBitWidth: return "limit.bit_width";
        case Status::kBlockHeaderReserved: return "layout.block_header";
        case Status::kOutputTooSmall: return "decode.output_capacity";
        case Status::kSizeOverflow: return "decode.overflow";
        case Status::kFeatureOffsets: return "layout.features";
        case Status::kFeatureType: return "layout.features";
        case Status::kFeatureAtomIndex: return "layout.features";
        case Status::kAtomsPerFeature: return "limit.atoms_per_feature";
        case Status::kFeaturesPerType: return "limit.features_per_type";
        case Status::kFeaturesPerMolecule: return "limit.features_per_molecule";
        case Status::kFeaturesMissing: return "decode.features_missing";
        case Status::kSelfOverlapRange: return "decode.self_overlap_range";
        case Status::kAllocation: return "decode.allocation";
        case Status::kIsaUnavailable: return "decode.isa_unavailable";
        case Status::kEncodeRange: return "encode.range";
        case Status::kShortRead: return "io.short_read";
        case Status::kIoError: return "io.read_error";
    }
    return "decode.unknown";
}

const char* status_message(Status s) noexcept {
    switch (s) {
        case Status::kOk: return "ok";
        case Status::kBadArgument: return "a null buffer with a nonzero length, or an invalid argument";
        case Status::kRowsZero: return "a block has at least one row (rows == 0)";
        case Status::kRowsMismatch: return "the run-table slice does not sum to the block's rows";
        case Status::kBlockRows: return "block rows over the cap (4096)";
        case Status::kMoleculesZero: return "the run-table slice is empty";
        case Status::kNConf: return "a molecule has n_conf == 0";
        case Status::kNReal: return "a molecule has n_real == 0";
        case Status::kLHeavy: return "L_heavy outside [1, 256]";
        case Status::kL: return "device width L outside [1, 512]";
        case Status::kNTypes: return "n_types outside [0, 6]";
        case Status::kRowExceedsWidth: return "a row exceeds its output width (n_real > L_heavy or n_real + n_color > L)";
        case Status::kTruncated: return "the block is shorter than its 32-byte header";
        case Status::kBlockLength: return "the block's length is not 32 + 32 * G * (w_x + w_y + w_z)";
        case Status::kBitWidth: return "a bit width above 16";
        case Status::kBlockHeaderReserved: return "reserved block-header bytes are not zero";
        case Status::kOutputTooSmall: return "an output buffer is smaller than the block needs";
        case Status::kSizeOverflow: return "a size computation overflows";
        case Status::kFeatureOffsets: return "feature offsets out of range or not monotone, or a feature with no atoms";
        case Status::kFeatureType: return "a feature type outside [1, n_types]";
        case Status::kFeatureAtomIndex: return "a feature atom index >= the molecule's n_real";
        case Status::kAtomsPerFeature: return "a feature with more than 16 atoms";
        case Status::kFeaturesPerType: return "more than 255 features of one type in a molecule";
        case Status::kFeaturesPerMolecule: return "more than 255 features in a molecule";
        case Status::kFeaturesMissing: return "the feature slice is missing (n_types > 0) or present (n_types == 0)";
        case Status::kSelfOverlapRange: return "the block's rows run past a self-overlap section";
        case Status::kAllocation: return "scratch allocation failed";
        case Status::kIsaUnavailable: return "the requested ISA path is not in this build";
        case Status::kEncodeRange: return "an input the encoder cannot represent";
        case Status::kShortRead: return "the file ended before the requested bytes (truncated after open?)";
        case Status::kIoError: return "pread failed";
    }
    return "unknown status";
}

bool isa_compiled(Isa isa) noexcept {
    switch (isa) {
        case Isa::kAuto:
        case Isa::kScalar:
            return true;
        case Isa::kAvx2:
#if defined(__x86_64__) || defined(_M_X64)
            return true;
#else
            return false;
#endif
        case Isa::kNeon:
#if defined(__aarch64__) || defined(_M_ARM64)
            return true;
#else
            return false;
#endif
    }
    return false;
}

Isa active_isa() noexcept {
    int v = g_isa.load(std::memory_order_acquire);
    if (v < 0) {
        int want = static_cast<int>(isa_from_env());
        int expected = -1;
        g_isa.compare_exchange_strong(expected, want, std::memory_order_acq_rel);
        v = g_isa.load(std::memory_order_acquire);
    }
    return static_cast<Isa>(v);
}

Status force_isa(Isa isa) noexcept {
    if (!isa_compiled(isa)) return Status::kIsaUnavailable;
    const Isa resolved = isa == Isa::kAuto ? kDefaultIsa : isa;
    g_isa.store(static_cast<int>(resolved), std::memory_order_release);
    return Status::kOk;
}

const char* isa_name(Isa isa) noexcept {
    switch (isa) {
        case Isa::kAuto: return "auto";
        case Isa::kScalar: return "scalar";
        case Isa::kAvx2: return "avx2";
        case Isa::kNeon: return "neon";
    }
    return "unknown";
}

Status block_length(std::size_t total, unsigned w_x, unsigned w_y, unsigned w_z,
                    std::size_t* out_len) noexcept {
    if (!out_len) return Status::kBadArgument;
    if (w_x > kMaxBitWidth || w_y > kMaxBitWidth || w_z > kMaxBitWidth) return Status::kBitWidth;
    const std::size_t groups = total / kGroupValues + (total % kGroupValues != 0 ? 1 : 0);
    const std::size_t wsum = static_cast<std::size_t>(w_x) + w_y + w_z;  // <= 48
    std::size_t a = 0, b = 0, r = 0;
    if (!mul(groups, 32, &a) || !mul(a, wsum, &b) || !add(b, kBlockHeaderSize, &r))
        return Status::kSizeOverflow;
    *out_len = r;
    return Status::kOk;
}

Status gpu_layout_elements(std::size_t rows, std::size_t L, std::size_t* n_rows,
                           std::size_t* elements) noexcept {
    if (!n_rows || !elements) return Status::kBadArgument;
    const std::size_t tiles = rows / kTileRows + (rows % kTileRows != 0 ? 1 : 0);
    std::size_t nr = 0, a = 0, e = 0;
    if (!mul(tiles, kTileRows, &nr) || !mul(nr, L, &a) || !mul(a, 3, &e))
        return Status::kSizeOverflow;
    *n_rows = nr;
    *elements = e;
    return Status::kOk;
}

Status decode_heavy(const BlockInput& b, std::int16_t* out, std::size_t out_capacity) noexcept {
    Status s = check_shape(b, false);
    if (s != Status::kOk) return s;
    if (out == nullptr && out_capacity != 0) return Status::kBadArgument;
    Checked c;
    s = check_runs_and_block(b, &c);
    if (s != Status::kOk) return finish(s);
    const std::size_t L = b.L_heavy;
    std::size_t need = 0, rl = 0;
    if (!mul(b.rows, 3, &rl) || !mul(rl, L, &need)) return finish(Status::kSizeOverflow);
    if (out_capacity < need) return finish(Status::kOutputTooSmall);
    std::size_t stride = 0;
    s = unpack_all(c, t_scratch.axes, &stride);
    if (s != Status::kOk) return finish(s);
    // The run table as (n_conf, n_real) pairs; Molecule's first two fields.
    std::vector<detail::RowRun>& rr = t_scratch.row_runs;
    if (!ensure(rr, c.mols.size())) return finish(Status::kAllocation);
    for (std::size_t m = 0; m < c.mols.size(); ++m) rr[m] = {c.mols[m].n_conf, c.mols[m].n_real};
    heavy_rows(active_isa(), t_scratch.axes.data(), stride, rr.data(), c.mols.size(), L, out);
    return finish(Status::kOk);
}

namespace {

// Shared by the three device-width calls: checks, unpack, color.
Status run_common(const BlockInput& b, Common* k, std::uint8_t* counts,
                  std::size_t counts_cap, std::uint64_t* words, std::size_t words_cap,
                  BlockTimings* tm) noexcept {
    const Clock::time_point t0 = Clock::now();
    Status s = prepare(b, k, t0, tm);
    if (s != Status::kOk) return s;
    s = check_count_outputs(b, counts, counts_cap, words, words_cap);
    if (s != Status::kOk) return s;
    return Status::kOk;
}

// Dispatch once per writer, outside every molecule/feature/tile loop.
struct BuildKernels {
    decltype(&detail::build_heavy_tile_scalar) heavy;
    decltype(&detail::build_feature_tile_scalar) feature;
    decltype(&detail::emit_gpu_tile_scalar) gpu;
    detail::CpuTileEmitter cpu;
};
BuildKernels build_kernels(Isa isa, float step) noexcept {
    (void)step;
    switch (isa) {
#if defined(__x86_64__) || defined(_M_X64)
        case Isa::kAvx2: return {detail::build_heavy_tile_avx2, detail::build_feature_tile_avx2,
                                detail::emit_gpu_tile_avx2, detail::emit_cpu_tile_avx2};
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
        case Isa::kNeon: return {detail::build_heavy_tile_neon, detail::build_feature_tile_neon,
                                detail::emit_gpu_tile_neon, detail::select_cpu_tile_emitter_neon(step)};
#endif
        default: return {detail::build_heavy_tile_scalar, detail::build_feature_tile_scalar,
                         detail::emit_gpu_tile_scalar, detail::emit_cpu_tile_scalar};
    }
}

// Hostile C++ aliases keep the original defensive baseline, including its
// feature-offset double-fetch refusal order. Bindings already reject aliases.
bool writer_aliases_inputs(const BlockInput& b, void* out, std::size_t bytes,
                           std::uint8_t* counts, std::uint64_t* words) noexcept {
    if (metadata_aliases_features(b, counts, words)) return true;
    struct Span { const void* p; std::size_t n; };
    std::size_t cn = 0, wn = 0, rn = 0, fn = 0, an = 0;
    if (!mul(b.rows, b.n_types + 1, &cn) || !mul(b.rows, sizeof(*words), &wn) ||
        !mul(b.runs.n_molecules, kRunRecordSize, &rn) ||
        !mul(b.features.offsets_len, sizeof(*b.features.offsets), &fn) ||
        !mul(b.features.atom_offsets_len, sizeof(*b.features.atom_offsets), &an)) return true;
    const Span inputs[] = {{b.in, b.in_len}, {b.runs.records, rn},
        {b.features.offsets, fn}, {b.features.types, b.features.types_len},
        {b.features.atom_offsets, an}, {b.features.atoms, b.features.atoms_len}};
    const Span outputs[] = {{out, bytes}, {counts, cn}, {words, wn}};
    for (std::size_t i = 0; i < 3; ++i) {
        for (const Span& in : inputs)
            if (overlaps(outputs[i].p, outputs[i].n, in.p, in.n)) return true;
        for (std::size_t j = 0; j < i; ++j)
            if (overlaps(outputs[i].p, outputs[i].n, outputs[j].p, outputs[j].n)) return true;
    }
    return false;
}

struct CachedFeature {
    std::uint8_t arity;
    std::uint8_t atoms[kMaxAtomsPerFeature];
};
Status cache_features(const BlockInput& b, const Molecule& mol,
                      std::array<CachedFeature, kMaxFeatureCount>& cache) noexcept {
    const FeatureSlice& f = b.features;
    for (std::size_t ft = 0; ft < mol.n_color; ++ft) {
        const std::size_t a0 = load(f.atom_offsets, mol.f0 + ft);
        const std::size_t a1 = load(f.atom_offsets, mol.f0 + ft + 1);
        // Recheck at the gather use site before caching immutable metadata.
        if (a0 >= a1 || a1 > f.atoms_len || a1 - a0 > kMaxAtomsPerFeature)
            return Status::kFeatureOffsets;
        CachedFeature& cf = cache[ft];
        cf.arity = static_cast<std::uint8_t>(a1 - a0);
        for (std::size_t a = 0; a < cf.arity; ++a) {
            const std::uint8_t idx = f.atoms[a0 + a];
            if (idx >= mol.n_real) return Status::kFeatureAtomIndex;
            cf.atoms[a] = idx;
        }
    }
    return Status::kOk;
}

// All shape/header/features/capacities are checked by the writer before this
// call. Products below involve only checked values (rows<=4096, L<=512).
// The sole tile is [slot][axis][32]; each segment belongs to one molecule.
Status build_fused(const BlockInput& b, Common* k, int tile_mode, float step,
                   std::int16_t* gpu_out, float* cpu_out, std::uint8_t* counts,
                   std::uint64_t* words, BlockTimings* tm) noexcept {
    const auto t0 = Clock::now();
    std::size_t stage_n = 0;
    if (!mul(b.L, 3 * kTileRows, &stage_n)) return Status::kSizeOverflow;
    static_assert(kMaxL * 3 * kTileRows * sizeof(std::int16_t) == 98304,
                  "bounded tile scratch contract");
    const bool planar = gpu_out && tile_mode == -1;
    if (!planar && !ensure(t_scratch.stage, stage_n)) return Status::kAllocation;
    Status s = unpack_all(k->c, t_scratch.axes, &k->stride);
    // Fused attribution: decode includes validation (also feature checks) and
    // unpack. Color includes cached derivation, counts and final emission.
    if (tm) { tm->decode += tm->color + seconds_since(t0); tm->color = 0.0; }
    if (s != Status::kOk) return s;
    const auto t1 = Clock::now();
    const BuildKernels kernels = build_kernels(active_isa(), step);
    std::array<CachedFeature, kMaxFeatureCount> cache;
    std::size_t row = 0, pos = 0, mi = 0, jm = 0;
    const std::size_t nc = b.n_types + 1;
    while (row < b.rows) {
        const std::size_t tile_rows = std::min(kTileRows, b.rows - row);
        std::int16_t* tile = planar ? gpu_out + (row / kTileRows) * stage_n : t_scratch.stage.data();
        std::size_t lane = 0;
        while (lane < tile_rows) {
            const Molecule& mol = k->c.mols[mi];
            if (jm == 0) {
                s = cache_features(b, mol, cache);
                if (s != Status::kOk) return s;
            }
            const std::size_t lanes = std::min(tile_rows - lane, mol.n_conf - jm);
            kernels.heavy(t_scratch.axes.data(), k->stride, pos, mol.n_real, lane, lanes, tile);
            for (std::size_t ft = 0; ft < mol.n_color; ++ft)
                kernels.feature(tile, mol.n_real + ft, cache[ft].atoms, cache[ft].arity, lane, lanes);
            // Occupied values are overwritten by kernels. Zero only the
            // segment's padding, leaving other molecule segments untouched.
            for (std::size_t slot = mol.n_real + mol.n_color; slot < b.L; ++slot)
                for (std::size_t ax = 0; ax < 3; ++ax)
                    std::memset(tile + (slot * 3 + ax) * kTileRows + lane, 0, lanes * sizeof(*tile));
            std::uint64_t word = mol.n_real;
            for (std::size_t t = 1; t <= kMaxNTypes; ++t)
                word |= static_cast<std::uint64_t>(mol.type_counts[t]) << (8 * t);
            word |= static_cast<std::uint64_t>(mol.n_color) << 56;
            for (std::size_t j = 0; j < lanes; ++j) {
                if (counts) {
                    counts[(row + lane + j) * nc] = static_cast<std::uint8_t>(mol.n_real);
                    for (std::size_t t = 1; t < nc; ++t)
                        counts[(row + lane + j) * nc + t] = mol.type_counts[t];
                }
                if (words) words[row + lane + j] = word;
            }
            pos += lanes * mol.n_real;
            lane += lanes;
            jm += lanes;
            if (jm == mol.n_conf) { jm = 0; ++mi; }
        }
        if (gpu_out) {
            // Only the final GPU tile has inactive rows; CPU writes real rows.
            if (tile_rows < kTileRows)
                for (std::size_t slot = 0; slot < b.L; ++slot)
                    for (std::size_t ax = 0; ax < 3; ++ax)
                        std::memset(tile + (slot * 3 + ax) * kTileRows + tile_rows, 0,
                                    (kTileRows - tile_rows) * sizeof(*tile));
            if (!planar) kernels.gpu(tile, b.L, tile_mode, gpu_out + (row / kTileRows) * stage_n);
        } else {
            kernels.cpu(tile, b.L, tile_rows, step, cpu_out + row * b.L * 3);
        }
        row += tile_rows;
    }
    if (tm) tm->color += seconds_since(t1);
    return Status::kOk;
}

Status unpack_and_derive(const BlockInput& b, Common* k, std::uint8_t* counts,
                         std::uint64_t* words, BlockTimings* tm) noexcept {
    Clock::time_point t = Clock::now();
    Status s = unpack_all(k->c, t_scratch.axes, &k->stride);
    if (tm) tm->decode += seconds_since(t);
    if (s != Status::kOk) return s;
    t = Clock::now();
    s = derive_color(b, k->c, t_scratch.axes.data(), k->stride, t_scratch.pseudo, counts, words);
    if (tm) tm->color += seconds_since(t);
    return s;
}

// Append one validated block into launch-relative tiles. A physical block can
// end inside a tile: retain that tile until the next block fills it, and clear
// inactive lanes only at the end of the launch. Color derivation is unchanged;
// the ISA heavy gather and tile emitter replace per-row merged-record scatter.
Status append_gpu_tiles(const BlockInput& b, Common* k, std::size_t row0,
                        bool final_block, int tile_mode, std::int16_t* out,
                        std::uint8_t* counts) noexcept {
    Status s = unpack_and_derive(b, k, counts, nullptr, nullptr);
    if (s != Status::kOk) return s;
    const std::size_t stage_n = b.L * 3 * kTileRows;
    const bool planar = tile_mode == -1;
    if (!planar && !ensure(t_scratch.stage, stage_n)) return Status::kAllocation;
    const BuildKernels kernels = build_kernels(active_isa(), 1.0f);
    const std::int16_t* pseudo = t_scratch.pseudo.data();
    std::size_t row = 0, pos = 0;
    for (const Molecule& mol : k->c.mols) {
        std::size_t remaining = mol.n_conf;
        while (remaining) {
            const std::size_t launch_row = row0 + row;
            const std::size_t lane = launch_row % kTileRows;
            const std::size_t lanes = std::min(kTileRows - lane, remaining);
            std::int16_t* dst = out + (launch_row / kTileRows) * stage_n;
            std::int16_t* tile = planar ? dst : t_scratch.stage.data();
            kernels.heavy(t_scratch.axes.data(), k->stride, pos, mol.n_real,
                          lane, lanes, tile);
            for (std::size_t feature = 0; feature < mol.n_color; ++feature)
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    std::int16_t* segment = tile + ((mol.n_real + feature) * 3 + axis) * kTileRows + lane;
                    for (std::size_t j = 0; j < lanes; ++j)
                        segment[j] = pseudo[(j * mol.n_color + feature) * 3 + axis];
                }
            for (std::size_t slot = mol.n_real + mol.n_color; slot < b.L; ++slot)
                for (std::size_t axis = 0; axis < 3; ++axis)
                    std::memset(tile + (slot * 3 + axis) * kTileRows + lane, 0,
                                lanes * sizeof(*tile));
            row += lanes;
            pos += lanes * mol.n_real;
            if (mol.n_color) pseudo += lanes * mol.n_color * 3;
            remaining -= lanes;
            const std::size_t end_lane = lane + lanes;
            const bool launch_tail = final_block && row == b.rows;
            if (launch_tail && end_lane < kTileRows)
                for (std::size_t slot = 0; slot < b.L; ++slot)
                    for (std::size_t axis = 0; axis < 3; ++axis)
                        std::memset(tile + (slot * 3 + axis) * kTileRows + end_lane, 0,
                                    (kTileRows - end_lane) * sizeof(*tile));
            if (!planar && (end_lane == kTileRows || launch_tail))
                kernels.gpu(tile, b.L, tile_mode, dst);
        }
    }
    return Status::kOk;
}

}  // namespace

Status decode_merged(const BlockInput& b, std::int16_t* out, std::size_t out_capacity,
                     std::uint8_t* counts, std::size_t counts_capacity,
                     BlockTimings* tm) noexcept {
    if (tm) *tm = BlockTimings{};
    if (out == nullptr && out_capacity != 0) return Status::kBadArgument;
    Common k;
    Status s = run_common(b, &k, counts, counts_capacity, nullptr, 0, tm);
    if (s != Status::kOk) return finish(s);
    std::size_t need = 0, rl = 0;
    if (!mul(b.rows, 3, &rl) || !mul(rl, b.L, &need)) return finish(Status::kSizeOverflow);
    if (out_capacity < need) return finish(Status::kOutputTooSmall);
    s = unpack_and_derive(b, &k, counts, nullptr, tm);
    if (s != Status::kOk) return finish(s);
    const Clock::time_point t = Clock::now();
    const std::int16_t* axes = t_scratch.axes.data();
    const std::int16_t* ps = t_scratch.pseudo.data();
    std::size_t pos = 0;
    std::int16_t* o = out;
    for (const Molecule& mol : k.c.mols) {
        for (std::size_t j = 0; j < mol.n_conf; ++j) {
            merged_row(axes, k.stride, pos, mol.n_real, ps, mol.n_color, b.L, o);
            pos += mol.n_real;
            ps += 3 * mol.n_color;
            o += 3 * b.L;
        }
    }
    if (tm) tm->decode += seconds_since(t);
    return finish(Status::kOk);
}

Status write_gpu_block(const BlockInput& b, int tile_mode, std::int16_t* out,
                       std::size_t out_capacity, std::uint8_t* counts,
                       std::size_t counts_capacity, std::uint64_t* words,
                       std::size_t words_capacity, BlockTimings* tm) noexcept {
    if (tm) *tm = BlockTimings{};
    if (tile_mode != 1 && tile_mode != -1) return Status::kBadArgument;
    if (out == nullptr && out_capacity != 0) return Status::kBadArgument;
    Common k;
    Status s = run_common(b, &k, counts, counts_capacity, words, words_capacity, tm);
    if (s != Status::kOk) return finish(s);
    std::size_t n_rows = 0, need = 0;
    s = gpu_layout_elements(b.rows, b.L, &n_rows, &need);
    if (s != Status::kOk) return finish(s);
    if (out_capacity < need) return finish(Status::kOutputTooSmall);
    std::size_t output_bytes = 0;
    if (!mul(need, sizeof(*out), &output_bytes)) return finish(Status::kSizeOverflow);
    if (use_fused_builder(true) &&
        !writer_aliases_inputs(b, out, output_bytes, counts, words))
        return finish(build_fused(b, &k, tile_mode, 1.0f, out, nullptr, counts, words, tm));
    s = unpack_and_derive(b, &k, counts, words, tm);
    if (s != Status::kOk) return finish(s);

    const Clock::time_point t = Clock::now();
    const std::size_t L = b.L;
    // Stage one tile of the merged record, [32][3][L], then write it in tile
    // order. Lanes past the last row are zero in the stage.
    std::size_t stage_n = 0;
    if (!mul(kTileRows * 3, L, &stage_n)) return finish(Status::kSizeOverflow);
    if (!ensure(t_scratch.stage, stage_n)) return finish(Status::kAllocation);
    std::int16_t* stage = t_scratch.stage.data();
    const std::int16_t* axes = t_scratch.axes.data();
    const std::int16_t* ps = t_scratch.pseudo.data();
    std::size_t pos = 0, row = 0, mi = 0, jm = 0;
    const std::size_t tiles = n_rows / kTileRows;
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        std::size_t lanes = 0;
        for (; lanes < kTileRows && row < b.rows; ++lanes, ++row) {
            const Molecule& mol = k.c.mols[mi];
            merged_row(axes, k.stride, pos, mol.n_real, ps, mol.n_color, L,
                       stage + lanes * 3 * L);
            pos += mol.n_real;
            ps += 3 * mol.n_color;
            if (++jm == mol.n_conf) { jm = 0; ++mi; }
        }
        if (lanes < kTileRows)
            std::memset(stage + lanes * 3 * L, 0, (kTileRows - lanes) * 3 * L * sizeof(std::int16_t));
        std::int16_t* dst = out + tile * kTileRows * L * 3;
        if (tile_mode == 1) {  // [atom][lane][xyz]
            for (std::size_t a = 0; a < L; ++a)
                for (std::size_t l = 0; l < kTileRows; ++l) {
                    const std::int16_t* sr = stage + l * 3 * L + a;
                    std::int16_t* d = dst + (a * kTileRows + l) * 3;
                    d[0] = sr[0];
                    d[1] = sr[L];
                    d[2] = sr[2 * L];
                }
        } else {  // [atom][xyz][lane]
            for (std::size_t a = 0; a < L; ++a)
                for (std::size_t c3 = 0; c3 < 3; ++c3) {
                    std::int16_t* d = dst + (a * 3 + c3) * kTileRows;
                    for (std::size_t l = 0; l < kTileRows; ++l) d[l] = stage[l * 3 * L + c3 * L + a];
                }
        }
    }
    if (tm) tm->decode += seconds_since(t);
    return finish(Status::kOk);
}

Status write_gpu_launch(const BlockInput* blocks, std::size_t n_blocks,
                        std::size_t rows, int tile_mode,
                        std::int16_t* out, std::size_t out_capacity,
                        std::uint8_t* counts, std::size_t counts_capacity,
                        std::int64_t* mol_ids, std::size_t mol_ids_capacity) noexcept {
    if (!blocks || !n_blocks || !rows || n_blocks > rows || !out || !counts || !mol_ids ||
        (tile_mode != 1 && tile_mode != -1)) return Status::kBadArgument;
    Status first_shape = check_shape(blocks[0], true);
    if (first_shape != Status::kOk) return first_shape;
    const std::size_t L = blocks[0].L, nt = blocks[0].n_types;
    std::size_t nr = 0, need = 0, count_need = 0;
    Status s = gpu_layout_elements(rows, L, &nr, &need);
    if (s != Status::kOk) return s;
    if (!mul(rows, nt + 1, &count_need)) return Status::kSizeOverflow;
    if (out_capacity < need || counts_capacity < count_need || mol_ids_capacity < rows)
        return Status::kOutputTooSmall;
    std::vector<Common> checked;
    try { checked.resize(n_blocks); } catch (...) { return Status::kAllocation; }
    std::size_t total = 0;
    for (std::size_t bi = 0; bi < n_blocks; ++bi) {
        const BlockInput& b = blocks[bi];
        if (b.L != L || b.n_types != nt || b.L_heavy != blocks[0].L_heavy)
            return finish(Status::kBadArgument);
        s = prepare(b, &checked[bi], Clock::now(), nullptr);
        if (s != Status::kOk) return finish(s);
        if (!add(total, b.rows, &total)) return finish(Status::kSizeOverflow);
        if (total > rows) return finish(Status::kRowsMismatch);
        for (std::size_t mi = 0; mi < b.runs.n_molecules; ++mi)
            if (b.runs.records[mi * kRunRecordSize + 7] & 0x80)
                return finish(Status::kBadArgument);  // mol_id must fit int64
    }
    if (total != rows) return finish(Status::kRowsMismatch);
    // The general path retains at most one launch-relative tile across blocks.
    // Validation of every block above completes before any output is written.
    std::size_t row0 = 0;
    for (std::size_t bi = 0; bi < n_blocks; ++bi) {
        const BlockInput& b = blocks[bi];
        Common& k = checked[bi];
        std::uint8_t* cb = counts + row0 * (nt + 1);
        // Preserve the existing optimized writer whenever the entire block
        // occupies full launch-relative tiles (or ends the launch).
        if (row0 % kTileRows == 0 && (b.rows % kTileRows == 0 || bi + 1 == n_blocks)) {
            const std::size_t offset = row0 * L * 3;
            s = write_gpu_block(b, tile_mode, out + offset, out_capacity - offset,
                                cb, b.rows * (nt + 1), nullptr, 0, nullptr);
        } else {
            s = append_gpu_tiles(b, &k, row0, bi + 1 == n_blocks, tile_mode, out, cb);
        }
        if (s != Status::kOk) return finish(s);
        std::size_t row = row0;
        for (std::size_t mi = 0; mi < b.runs.n_molecules; ++mi) {
            std::uint64_t id = 0;
            for (unsigned byte = 0; byte < 8; ++byte)
                id |= std::uint64_t(b.runs.records[mi * kRunRecordSize + byte]) << (8 * byte);
            for (std::size_t j = 0; j < k.c.mols[mi].n_conf; ++j)
                mol_ids[row++] = static_cast<std::int64_t>(id);
        }
        row0 += b.rows;
    }
    // Clear just the final launch padding; never introduce interior padding.
    for (std::size_t row = rows; row < nr; ++row)
        for (std::size_t a = 0; a < L; ++a)
            for (std::size_t c = 0; c < 3; ++c) {
                const std::size_t tile = row / kTileRows, lane = row % kTileRows;
                const std::size_t dst = tile * kTileRows * L * 3 +
                    (tile_mode == 1 ? (a * kTileRows + lane) * 3 + c
                                    : (a * 3 + c) * kTileRows + lane);
                out[dst] = 0;
            }
    return finish(Status::kOk);
}

Status write_cpu_block(const BlockInput& b, float coord_scale, float* out,
                       std::size_t out_capacity, std::uint8_t* counts,
                       std::size_t counts_capacity, std::uint64_t* words,
                       std::size_t words_capacity, BlockTimings* tm) noexcept {
    if (tm) *tm = BlockTimings{};
    if (!std::isfinite(coord_scale) || !(coord_scale > 0.0f)) return Status::kBadArgument;
    if (out == nullptr && out_capacity != 0) return Status::kBadArgument;
    Common k;
    Status s = run_common(b, &k, counts, counts_capacity, words, words_capacity, tm);
    if (s != Status::kOk) return finish(s);
    std::size_t need = 0, rl = 0;
    if (!mul(b.rows, b.L, &rl) || !mul(rl, 3, &need)) return finish(Status::kSizeOverflow);
    if (out_capacity < need) return finish(Status::kOutputTooSmall);
    std::size_t output_bytes = 0;
    if (!mul(need, sizeof(*out), &output_bytes)) return finish(Status::kSizeOverflow);
    if (use_fused_builder(false) &&
        !writer_aliases_inputs(b, out, output_bytes, counts, words))
        return finish(build_fused(b, &k, 0, coord_scale, nullptr, out, counts, words, tm));
    s = unpack_and_derive(b, &k, counts, words, tm);
    if (s != Status::kOk) return finish(s);

    const Clock::time_point t = Clock::now();
    const std::size_t L = b.L;
    const std::int16_t* axes = t_scratch.axes.data();
    const std::int16_t* ps = t_scratch.pseudo.data();
    std::size_t pos = 0;
    float* o = out;
    for (const Molecule& mol : k.c.mols) {
        for (std::size_t j = 0; j < mol.n_conf; ++j) {
            // Write AoS directly: avoid an int16 merged-row scratch and its
            // axis transpose. Preserve the exact int16->float conversion and
            // single float multiply for every occupied coordinate.
            for (std::size_t a = 0; a < mol.n_real; ++a, o += 3) {
                o[0] = static_cast<float>(axes[pos + a]) * coord_scale;
                o[1] = static_cast<float>(axes[k.stride + pos + a]) * coord_scale;
                o[2] = static_cast<float>(axes[2 * k.stride + pos + a]) * coord_scale;
            }
            for (std::size_t a = 0; a < 3 * mol.n_color; ++a) o[a] = static_cast<float>(ps[a]) * coord_scale;
            o += 3 * mol.n_color;
            const std::size_t padding = 3 * (L - mol.n_real - mol.n_color);
            std::memset(o, 0, padding * sizeof(float));
            o += padding;
            pos += mol.n_real;
            ps += 3 * mol.n_color;
        }
    }
    if (tm) tm->decode += seconds_since(t);
    return finish(Status::kOk);
}

Status bind_self_overlaps(const SelfOverlapInput& so, std::size_t rows, float* vaa_out,
                          std::size_t vaa_out_capacity, double* cs_out,
                          std::size_t cs_out_capacity, bool* cs_present) noexcept {
    if (cs_present) *cs_present = false;
    if (rows == 0) return Status::kRowsZero;
    if ((so.vaa == nullptr && so.vaa_len) || (so.color_self == nullptr && so.color_self_len) ||
        (vaa_out == nullptr && vaa_out_capacity) || (cs_out == nullptr && cs_out_capacity))
        return Status::kBadArgument;
    std::size_t end = 0;
    if (!add(so.first_row, rows, &end)) return Status::kSizeOverflow;
    if (vaa_out) {
        if (so.vaa == nullptr || end > so.vaa_len) return Status::kSelfOverlapRange;
        if (vaa_out_capacity < rows) return Status::kOutputTooSmall;
    }
    const bool present = so.color_self != nullptr;
    if (present && cs_out) {
        if (end > so.color_self_len) return Status::kSelfOverlapRange;
        if (cs_out_capacity < rows) return Status::kOutputTooSmall;
    }
    if (vaa_out)
        std::memcpy(vaa_out, reinterpret_cast<const unsigned char*>(so.vaa) + so.first_row * sizeof(float),
                    rows * sizeof(float));
    if (present && cs_out) {
        std::memcpy(cs_out,
                    reinterpret_cast<const unsigned char*>(so.color_self) + so.first_row * sizeof(double),
                    rows * sizeof(double));
        if (cs_present) *cs_present = true;
    }
    return Status::kOk;
}

}  // namespace r3g
}  // namespace r3

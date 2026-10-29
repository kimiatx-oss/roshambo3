// Format-3 codec: the BP256-16 block encoder (FORMAT.md 5.5), for the corpus
// builder. Byte-identical with reference.encode_block: per axis
// min_a = min over the block's real atoms, w_a = bit_length(max - min_a),
// u = v - min_a packed into the vertical layout of 5.3, zeros in tail
// positions and in the 23 reserved header bytes.
#include "r3g_codec.hpp"

#include <cstring>

namespace r3 {
namespace r3g {
namespace {

inline bool mul(std::size_t a, std::size_t b, std::size_t* r) noexcept {
    return !__builtin_mul_overflow(a, b, r);
}
inline bool add(std::size_t a, std::size_t b, std::size_t* r) noexcept {
    return !__builtin_add_overflow(a, b, r);
}
inline unsigned bit_length(std::uint32_t x) noexcept {
    unsigned n = 0;
    while (x) { ++n; x >>= 1; }
    return n;
}
inline std::int16_t load_i16(const std::int16_t* base, std::size_t i) noexcept {
    std::int16_t v;
    std::memcpy(&v, reinterpret_cast<const unsigned char*>(base) + 2 * i, 2);
    return v;
}
inline void or_u16(std::uint8_t* p, std::uint32_t bits) noexcept {
    p[0] = static_cast<std::uint8_t>(p[0] | (bits & 0xFFu));
    p[1] = static_cast<std::uint8_t>(p[1] | ((bits >> 8) & 0xFFu));
}

}  // namespace

Status encode_block(const std::int16_t* coords, std::size_t coords_len, RunSlice runs,
                    std::size_t rows, std::size_t L_heavy, std::uint8_t* out,
                    std::size_t out_capacity, std::size_t* out_len) noexcept {
    if (!out_len) return Status::kBadArgument;
    *out_len = 0;
    if ((coords == nullptr && coords_len) || (runs.records == nullptr && runs.n_molecules) ||
        (out == nullptr && out_capacity))
        return Status::kBadArgument;
    if (rows == 0) return Status::kRowsZero;
    if (L_heavy == 0 || L_heavy > kMaxLHeavy) return Status::kLHeavy;
    if (runs.n_molecules == 0) return Status::kMoleculesZero;
    if (rows > kMaxBlockRowsSingle || (runs.n_molecules >= 2 && rows > kMaxBlockRowsMulti))
        return Status::kBlockRows;
    if (runs.n_molecules > rows) return Status::kRowsMismatch;
    std::size_t need_in = 0, rl = 0;
    if (!mul(rows, 3, &rl) || !mul(rl, L_heavy, &need_in)) return Status::kSizeOverflow;
    if (coords_len < need_in) return Status::kBadArgument;

    struct Run { std::size_t n_conf, n_real; };
    Run checked[kMaxBlockRowsSingle];
    std::size_t sum_rows = 0, total = 0;
    for (std::size_t m = 0; m < runs.n_molecules; ++m) {
        const std::uint8_t* rec = runs.records + m * kRunRecordSize;
        const std::size_t n_conf = static_cast<std::size_t>(rec[8]) | (static_cast<std::size_t>(rec[9]) << 8);
        const std::size_t n_real = rec[10];
        if (n_conf == 0 || n_conf > 4096) return Status::kNConf;
        if (n_real == 0) return Status::kNReal;
        if (n_real > L_heavy) return Status::kRowExceedsWidth;
        std::size_t atoms = 0;
        if (!add(sum_rows, n_conf, &sum_rows) || !mul(n_conf, n_real, &atoms) ||
            !add(total, atoms, &total))
            return Status::kSizeOverflow;
        if (sum_rows > rows) return Status::kRowsMismatch;
        checked[m] = Run{n_conf, n_real};
    }
    if (sum_rows != rows) return Status::kRowsMismatch;

    // Per-axis frame of reference over the real atoms.
    std::int32_t lo[3] = {32767, 32767, 32767}, hi[3] = {-32768, -32768, -32768};
    {
        std::size_t row = 0;
        for (std::size_t m = 0; m < runs.n_molecules; ++m) {
            const std::size_t n_conf = checked[m].n_conf;
            const std::size_t n_real = checked[m].n_real;
            for (std::size_t j = 0; j < n_conf; ++j, ++row)
                for (int a = 0; a < 3; ++a)
                    for (std::size_t i = 0; i < n_real; ++i) {
                        const std::int32_t v = load_i16(coords, (row * 3 + a) * L_heavy + i);
                        if (v < lo[a]) lo[a] = v;
                        if (v > hi[a]) hi[a] = v;
                    }
        }
    }
    unsigned w[3];
    for (int a = 0; a < 3; ++a) w[a] = bit_length(static_cast<std::uint32_t>(hi[a] - lo[a]));
    std::size_t len = 0;
    Status s = block_length(total, w[0], w[1], w[2], &len);
    if (s != Status::kOk) return s;
    *out_len = len;
    if (out_capacity < len) return Status::kOutputTooSmall;

    std::memset(out, 0, len);
    for (int a = 0; a < 3; ++a) {
        const std::uint16_t m16 = static_cast<std::uint16_t>(lo[a] & 0xFFFF);
        out[2 * a] = static_cast<std::uint8_t>(m16 & 0xFFu);
        out[2 * a + 1] = static_cast<std::uint8_t>(m16 >> 8);
        out[6 + a] = static_cast<std::uint8_t>(w[a]);
    }
    const std::size_t groups = total / kGroupValues + (total % kGroupValues != 0 ? 1 : 0);
    std::uint8_t* stream = out + kBlockHeaderSize;
    for (int a = 0; a < 3; ++a) {
        const unsigned wa = w[a];
        if (wa != 0) {
            std::size_t j = 0, row = 0;
            for (std::size_t m = 0; m < runs.n_molecules; ++m) {
                const std::size_t n_conf = checked[m].n_conf;
                const std::size_t n_real = checked[m].n_real;
                for (std::size_t c = 0; c < n_conf; ++c, ++row)
                    for (std::size_t i = 0; i < n_real; ++i, ++j) {
                        const std::uint32_t u = static_cast<std::uint32_t>(
                            load_i16(coords, (row * 3 + a) * L_heavy + i) - lo[a]);
                        const std::size_t g = j / kGroupValues, r = j % kGroupValues;
                        const unsigned k = static_cast<unsigned>(r / 16), l = static_cast<unsigned>(r % 16);
                        const unsigned bpos = k * wa, wi = bpos >> 4, sh = bpos & 15;
                        std::uint8_t* base = stream + g * wa * 32 + 2 * l;
                        or_u16(base + 32 * wi, (u << sh) & 0xFFFFu);
                        if (sh + wa > 16) or_u16(base + 32 * (wi + 1), u >> (16 - sh));
                    }
            }
        }
        stream += groups * wa * 32;
    }
    return Status::kOk;
}

}  // namespace r3g
}  // namespace r3

// Standalone fixture for the native format-3 codec (src/roshambo3/backend/
// _cpu_src/r3g_*.cpp), compiled and run by tests/test_format3_native_cpp.py
// -- plain, and with -fsanitize=address,undefined. No Python, no pybind11.
//
//   r3g_codec_test                 run every built-in test; exit 0 iff all pass
//   r3g_codec_test --cases DIR     additionally run every DIR/*.case file (written
//                                  by the Python test from the conformance corpus,
//                                  the per-width vectors and the adversarial seeds)
//                                  through every compiled ISA and every writer,
//                                  writing DIR/<case>.<isa>.<output> files (or a
//                                  .status file) for the Python side to digest.
//
// Built-in tests:
//   * every width 0..16 on every axis, packing-group tails of 1..255 values with
//     G = 1 and G = 2, encoded by the native encoder, compared byte for byte with
//     a literal bit-by-bit transcription of FORMAT.md 5.3-5.5, decoded by every
//     compiled ISA and by a bit-by-bit transcription of 5.3's pseudo-code;
//   * n_real 1..255 at every L that is a multiple of 8 from pad8(n_real) to 256
//     (n_real = 255 -> L = 256 included);
//   * random multi-molecule blocks with features through every writer, every ISA
//     identical, checked against an independent derivation;
//   * the 5.6 wrap (a tampered min);
//   * every refusal path, each with a guard region past the output capacity that
//     must stay untouched.
#include "r3g_codec.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <random>
#include <string>
#include <vector>

using namespace r3::r3g;

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            if (++g_failures <= 50) {                                      \
                std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
                std::fprintf(stderr, __VA_ARGS__);                         \
                std::fprintf(stderr, "\n");                                \
            }                                                              \
        }                                                                  \
    } while (0)

static std::vector<Isa> compiled() {
    std::vector<Isa> v;
    for (Isa i : {Isa::kScalar, Isa::kAvx2, Isa::kNeon})
        if (isa_compiled(i)) v.push_back(i);
    return v;
}

// ---- Independent transcriptions of the spec --------------------------------

static std::size_t pad8(std::size_t n) { return std::max<std::size_t>(8, (n + 7) / 8 * 8); }

static int16_t wrap(int32_t x) {
    int32_t r = ((x + 32768) % 65536 + 65536) % 65536 - 32768;
    return static_cast<int16_t>(r);
}

// FORMAT.md 5.3 pseudo-code, bit by bit.
static std::vector<int16_t> oracle_unpack(const uint8_t* s, unsigned w, int16_t mn, std::size_t total) {
    std::vector<int16_t> out;
    std::size_t G = (total + 255) / 256;
    for (std::size_t g = 0; g < G; ++g)
        for (unsigned k = 0; k < 16; ++k)
            for (unsigned l = 0; l < 16; ++l) {
                uint32_t u = 0;
                for (unsigned bit = 0; bit < w; ++bit) {
                    unsigned b = k * w + bit;
                    std::size_t off = g * w * 32 + 32 * (b / 16) + 2 * l;
                    uint32_t word = s[off] | (uint32_t(s[off + 1]) << 8);
                    u |= ((word >> (b % 16)) & 1u) << bit;
                }
                out.push_back(wrap(int32_t(mn) + int32_t(u)));
            }
    out.resize(total);
    return out;
}

struct Mol { std::size_t n_conf, n_real; };

static std::vector<uint8_t> runs_of(const std::vector<Mol>& mols) {
    std::vector<uint8_t> r(mols.size() * 16, 0);
    for (std::size_t m = 0; m < mols.size(); ++m) {
        uint64_t id = 1000 + m;
        std::memcpy(&r[m * 16], &id, 8);
        r[m * 16 + 8] = uint8_t(mols[m].n_conf & 0xFF);
        r[m * 16 + 9] = uint8_t(mols[m].n_conf >> 8);
        r[m * 16 + 10] = uint8_t(mols[m].n_real);
    }
    return r;
}

static std::size_t rows_of(const std::vector<Mol>& mols) {
    std::size_t r = 0;
    for (auto& m : mols) r += m.n_conf;
    return r;
}

// FORMAT.md 5.5, bit by bit, from padded [rows][3][L].
static std::vector<uint8_t> oracle_encode(const std::vector<int16_t>& xyz, const std::vector<Mol>& mols, std::size_t L) {
    std::vector<int32_t> vals[3];
    std::size_t row = 0;
    for (auto& m : mols)
        for (std::size_t c = 0; c < m.n_conf; ++c, ++row)
            for (int a = 0; a < 3; ++a)
                for (std::size_t i = 0; i < m.n_real; ++i) vals[a].push_back(xyz[(row * 3 + a) * L + i]);
    std::vector<uint8_t> out(32, 0);
    std::size_t total = vals[0].size(), G = (total + 255) / 256;
    std::vector<uint8_t> streams[3];
    for (int a = 0; a < 3; ++a) {
        int32_t mn = *std::min_element(vals[a].begin(), vals[a].end());
        int32_t mx = *std::max_element(vals[a].begin(), vals[a].end());
        unsigned w = 0;
        while ((uint32_t(mx - mn) >> w) != 0) ++w;
        uint16_t m16 = uint16_t(mn);
        out[2 * a] = m16 & 0xFF; out[2 * a + 1] = m16 >> 8; out[6 + a] = uint8_t(w);
        streams[a].assign(G * w * 32, 0);
        for (std::size_t j = 0; j < total; ++j) {
            uint32_t u = uint32_t(vals[a][j] - mn);
            std::size_t g = j / 256, r = j % 256, k = r / 16, l = r % 16;
            for (unsigned bit = 0; bit < w; ++bit)
                if ((u >> bit) & 1u) {
                    std::size_t b = k * w + bit;
                    std::size_t off = g * w * 32 + 32 * (b / 16) + 2 * l;
                    uint16_t word = streams[a][off] | (uint16_t(streams[a][off + 1]) << 8);
                    word |= uint16_t(1u << (b % 16));
                    streams[a][off] = word & 0xFF; streams[a][off + 1] = word >> 8;
                }
        }
    }
    for (int a = 0; a < 3; ++a) out.insert(out.end(), streams[a].begin(), streams[a].end());
    return out;
}

static std::mt19937_64 rng(20261018);

// Random padded coordinates hitting exactly width w per axis (when total >= 2).
static std::vector<int16_t> random_coords(const std::vector<Mol>& mols, std::size_t L, const unsigned w[3]) {
    std::size_t rows = rows_of(mols);
    std::vector<int16_t> xyz(rows * 3 * L, 0);
    for (int a = 0; a < 3; ++a) {
        int32_t span = w[a] == 0 ? 0 : int32_t((1u << w[a]) - 1);
        int32_t mn = int32_t(rng() % uint64_t(65536 - span)) - 32768;
        std::size_t row = 0, j = 0;
        for (auto& m : mols)
            for (std::size_t c = 0; c < m.n_conf; ++c, ++row)
                for (std::size_t i = 0; i < m.n_real; ++i, ++j) {
                    int32_t v = mn + (span ? int32_t(rng() % uint64_t(span + 1)) : 0);
                    if (j == 0) v = mn;
                    if (j == 1) v = mn + span;
                    xyz[(row * 3 + a) * L + i] = int16_t(v);
                }
    }
    return xyz;
}

static BlockInput make_input(const std::vector<uint8_t>& block, const std::vector<uint8_t>& runs,
                             std::size_t rows, std::size_t L_heavy, std::size_t L = 0, std::size_t n_types = 0) {
    BlockInput in{};
    in.in = block.data(); in.in_len = block.size();
    in.runs = RunSlice{runs.data(), runs.size() / 16};
    in.rows = rows; in.L_heavy = L_heavy; in.L = L ? L : L_heavy; in.n_types = n_types;
    return in;
}

static bool decode_all_isas(const BlockInput& in, std::size_t n, std::vector<int16_t>* first) {
    bool same = true;
    std::vector<int16_t> ref;
    for (Isa isa : compiled()) {
        force_isa(isa);
        std::vector<int16_t> out(n + 16, int16_t(0x5A5A));
        Status s = decode_heavy(in, out.data(), n);
        CHECK(s == Status::kOk, "decode %s status %s", isa_name(isa), status_code(s));
        for (std::size_t i = n; i < n + 16; ++i) CHECK(out[i] == int16_t(0x5A5A), "guard overwritten");
        out.resize(n);
        if (ref.empty()) ref = out; else same = same && (out == ref);
    }
    force_isa(Isa::kAuto);
    if (first) *first = ref;
    return same;
}

// ---- Built-in tests --------------------------------------------------------

static void test_widths_and_tails() {
    for (unsigned w = 0; w <= 16; ++w) {
        for (std::size_t G0 = 0; G0 < 2; ++G0) {
            for (std::size_t tail = 1; tail <= 255; ++tail) {
                std::size_t total = G0 * 256 + tail;
                // Split into single-row molecules of random n_real (<= 255).
                std::vector<Mol> mols;
                std::size_t rem = total;
                while (rem) {
                    std::size_t n = 1 + rng() % std::min<std::size_t>(rem, 255);
                    mols.push_back({1, n});
                    rem -= n;
                }
                unsigned ws[3] = {w, (w + 5) % 17, 16 - w};
                if (total < 2) ws[0] = ws[1] = ws[2] = 0;
                const std::size_t L = 256, rows = mols.size();
                auto xyz = random_coords(mols, L, ws);
                auto runs = runs_of(mols);
                std::size_t len = 0;
                CHECK(encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), mols.size()}, rows, L, nullptr, 0, &len) == Status::kOutputTooSmall, "size query");
                std::vector<uint8_t> enc(len);
                CHECK(encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), mols.size()}, rows, L, enc.data(), enc.size(), &len) == Status::kOk, "encode");
                auto oracle = oracle_encode(xyz, mols, L);
                CHECK(enc == oracle, "encoder bytes differ from the bit-by-bit oracle w=%u total=%zu", w, total);
                for (int a = 0; a < 3; ++a) CHECK(enc[6 + a] == ws[a], "width %u expected %u", enc[6 + a], ws[a]);
                std::vector<int16_t> dec;
                auto in = make_input(enc, runs, rows, L);
                CHECK(decode_all_isas(in, rows * 3 * L, &dec), "ISA paths differ w=%u total=%zu", w, total);
                CHECK(dec == xyz, "round trip differs w=%u total=%zu", w, total);
                // Against the 5.3 pseudo-code, axis by axis.
                std::size_t off = 32;
                std::size_t G = (total + 255) / 256;
                for (int a = 0; a < 3; ++a) {
                    int16_t mn = int16_t(enc[2 * a] | (enc[2 * a + 1] << 8));
                    auto vals = oracle_unpack(enc.data() + off, enc[6 + a], mn, total);
                    off += G * enc[6 + a] * 32;
                    std::size_t j = 0, row = 0;
                    bool ok = true;
                    for (auto& m : mols)
                        for (std::size_t c = 0; c < m.n_conf; ++c, ++row)
                            for (std::size_t i = 0; i < m.n_real; ++i, ++j)
                                ok = ok && dec[(row * 3 + a) * L + i] == vals[j];
                    CHECK(ok, "decoder differs from the 5.3 pseudo-code w=%u total=%zu axis %d", w, total, a);
                }
            }
        }
    }
}

static void test_n_real_and_widths() {
    for (std::size_t n = 1; n <= 255; ++n) {
        for (std::size_t L = pad8(n); L <= 256; L += 8) {
            std::vector<Mol> mols = {{3, n}};
            unsigned ws[3] = {unsigned(rng() % 17), unsigned(rng() % 17), unsigned(rng() % 17)};
            auto xyz = random_coords(mols, L, ws);
            auto runs = runs_of(mols);
            std::size_t len = 0;
            encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 1}, 3, L, nullptr, 0, &len);
            std::vector<uint8_t> enc(len);
            CHECK(encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 1}, 3, L, enc.data(), len, &len) == Status::kOk, "encode");
            std::vector<int16_t> dec;
            auto in = make_input(enc, runs, 3, L);
            CHECK(decode_all_isas(in, 3 * 3 * L, &dec), "ISA paths differ n=%zu L=%zu", n, L);
            CHECK(dec == xyz, "round trip n=%zu L=%zu", n, L);
            if (n == 255) CHECK(L == 256 || L > 256, "pad8(255) is 256");
        }
    }
    // Widths that are not a multiple of 8 (the SIMD row writers fall back to
    // the scalar one there): L = n_real, and n_real + 3.
    for (std::size_t n : {1, 2, 5, 7, 9, 13, 100, 250, 253, 255}) {
        for (std::size_t L : {n, std::min<std::size_t>(n + 3, 256)}) {
            if (L % 8 == 0) continue;
            std::vector<Mol> mols = {{4, n}, {2, std::max<std::size_t>(1, n / 2)}};
            unsigned ws[3] = {unsigned(rng() % 17), unsigned(rng() % 17), unsigned(rng() % 17)};
            auto xyz = random_coords(mols, L, ws);
            auto runs = runs_of(mols);
            std::size_t len = 0;
            encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 2}, 6, L, nullptr, 0, &len);
            std::vector<uint8_t> enc(len);
            CHECK(encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 2}, 6, L, enc.data(), len, &len) == Status::kOk, "encode L=%zu", L);
            std::vector<int16_t> dec;
            CHECK(decode_all_isas(make_input(enc, runs, 6, L), 6 * 3 * L, &dec), "ISA paths differ L=%zu", L);
            CHECK(dec == xyz, "round trip n=%zu L=%zu (not a multiple of 8)", n, L);
        }
    }
}

static void test_wrap() {
    // One row of one atom per value 0..3; then tamper min so min + u wraps.
    std::vector<Mol> mols = {{1, 4}};
    std::vector<int16_t> xyz(3 * 8, 0);
    for (int i = 0; i < 4; ++i) { xyz[i] = int16_t(i); xyz[8 + i] = int16_t(-i); xyz[16 + i] = 0; }
    auto runs = runs_of(mols);
    std::size_t len = 0;
    encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 1}, 1, 8, nullptr, 0, &len);
    std::vector<uint8_t> enc(len);
    encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 1}, 1, 8, enc.data(), len, &len);
    enc[0] = 0xFE; enc[1] = 0x7F;  // min_x = 32766
    std::vector<int16_t> dec;
    CHECK(decode_all_isas(make_input(enc, runs, 1, 8), 24, &dec), "wrap: ISA paths differ");
    CHECK(dec[0] == 32766 && dec[1] == 32767 && dec[2] == -32768 && dec[3] == -32767, "wrap values %d %d %d %d", dec[0], dec[1], dec[2], dec[3]);
}

// A random color block: molecules, features, and every writer on every ISA.
struct ColorCase {
    std::vector<Mol> mols;
    std::vector<uint8_t> runs, block;
    std::vector<uint32_t> fo, fao;
    std::vector<uint8_t> ft, fa;
    std::vector<int16_t> heavy;  // [rows][3][L_heavy]
    std::size_t rows, L_heavy, L, n_types;
    BlockInput input() const {
        BlockInput in = make_input(block, runs, rows, L_heavy, L, n_types);
        if (!fo.empty()) {
            in.features = FeatureSlice{fo.data(), fo.size(), ft.data(), ft.size(), fao.data(), fao.size(), fa.data(), fa.size()};
        }
        return in;
    }
};

static ColorCase random_color_case(std::size_t n_mol, std::size_t n_types, std::size_t prefix_features) {
    ColorCase c;
    c.n_types = n_types;
    std::size_t maxr = 1, maxw = 1;
    // A group-wide feature table with `prefix_features` features of earlier
    // molecules ahead of this block's, so the offsets are not rebased.
    c.fo.push_back(uint32_t(prefix_features));
    for (std::size_t i = 0; i < prefix_features; ++i) {
        c.ft.push_back(1);
        c.fao.push_back(uint32_t(c.fa.size()));
        c.fa.push_back(0);
    }
    std::vector<std::vector<std::pair<int, std::vector<int>>>> feats(n_mol);
    for (std::size_t m = 0; m < n_mol; ++m) {
        Mol mol{1 + rng() % 9, 1 + rng() % 60};
        c.mols.push_back(mol);
        std::size_t nf = n_types ? rng() % 12 : 0;
        for (std::size_t f = 0; f < nf; ++f) {
            int t = 1 + int(rng() % n_types);
            std::size_t k = 1 + rng() % std::min<std::size_t>(16, mol.n_real);
            std::vector<int> atoms;
            while (atoms.size() < k) {
                int a = int(rng() % mol.n_real);
                if (std::find(atoms.begin(), atoms.end(), a) == atoms.end()) atoms.push_back(a);
            }
            std::sort(atoms.begin(), atoms.end());
            feats[m].push_back({t, atoms});
        }
        std::sort(feats[m].begin(), feats[m].end());
        for (auto& f : feats[m]) {
            c.ft.push_back(uint8_t(f.first));
            c.fao.push_back(uint32_t(c.fa.size()));
            for (int a : f.second) c.fa.push_back(uint8_t(a));
        }
        c.fo.push_back(uint32_t(c.ft.size()));
        maxr = std::max(maxr, mol.n_real);
        maxw = std::max(maxw, mol.n_real + feats[m].size());
    }
    c.fao.push_back(uint32_t(c.fa.size()));
    if (!n_types) { c.fo.clear(); c.ft.clear(); c.fao.clear(); c.fa.clear(); }
    c.L_heavy = pad8(maxr);
    c.L = pad8(maxw);
    c.rows = rows_of(c.mols);
    unsigned ws[3] = {unsigned(rng() % 17), unsigned(rng() % 17), unsigned(rng() % 17)};
    c.heavy = random_coords(c.mols, c.L_heavy, ws);
    c.runs = runs_of(c.mols);
    std::size_t len = 0;
    encode_block(c.heavy.data(), c.heavy.size(), RunSlice{c.runs.data(), n_mol}, c.rows, c.L_heavy, nullptr, 0, &len);
    c.block.resize(len);
    encode_block(c.heavy.data(), c.heavy.size(), RunSlice{c.runs.data(), n_mol}, c.rows, c.L_heavy, c.block.data(), len, &len);
    return c;
}

// The merged record derived independently from the case's own tables.
static std::vector<int16_t> expected_merged(const ColorCase& c, std::vector<uint8_t>* counts) {
    std::vector<int16_t> out(c.rows * 3 * c.L, 0);
    counts->assign(c.rows * (c.n_types + 1), 0);
    std::size_t row = 0;
    for (std::size_t m = 0; m < c.mols.size(); ++m) {
        for (std::size_t j = 0; j < c.mols[m].n_conf; ++j, ++row) {
            std::size_t nr = c.mols[m].n_real;
            (*counts)[row * (c.n_types + 1)] = uint8_t(nr);
            for (int a = 0; a < 3; ++a)
                for (std::size_t i = 0; i < nr; ++i)
                    out[(row * 3 + a) * c.L + i] = c.heavy[(row * 3 + a) * c.L_heavy + i];
            if (!c.n_types) continue;
            std::size_t slot = nr;
            for (uint32_t f = c.fo[m]; f < c.fo[m + 1]; ++f, ++slot) {
                (*counts)[row * (c.n_types + 1) + c.ft[f]]++;
                long k = long(c.fao[f + 1] - c.fao[f]);
                for (int a = 0; a < 3; ++a) {
                    long s = 0;
                    for (uint32_t q = c.fao[f]; q < c.fao[f + 1]; ++q) s += c.heavy[(row * 3 + a) * c.L_heavy + c.fa[q]];
                    long mag = s < 0 ? -s : s;
                    long qv = (2 * mag + k) / (2 * k);
                    out[(row * 3 + a) * c.L + slot] = int16_t(s < 0 ? -qv : qv);
                }
            }
        }
    }
    return out;
}

static void check_color_writers(const ColorCase& c, int trial) {
    BlockInput in = c.input();
    std::vector<uint8_t> ecounts;
    auto emerged = expected_merged(c, &ecounts);
    std::size_t nc = c.n_types + 1;
    std::vector<int16_t> ref_gpu[2];
    std::vector<float> ref_cpu;
    for (Isa isa : compiled()) {
        force_isa(isa);
        std::vector<int16_t> mg(emerged.size() + 8, 7);
        std::vector<uint8_t> cnt(c.rows * nc + 8, 7);
        BlockTimings tm;
        Status s = decode_merged(in, mg.data(), emerged.size(), cnt.data(), c.rows * nc, &tm);
        CHECK(s == Status::kOk, "merged %s", status_code(s));
        CHECK(std::equal(emerged.begin(), emerged.end(), mg.begin()), "merged differs trial %d isa %s", trial, isa_name(isa));
        CHECK(std::equal(ecounts.begin(), ecounts.end(), cnt.begin()), "counts differ trial %d", trial);
        CHECK(mg[emerged.size()] == 7 && cnt[c.rows * nc] == 7, "guard");
        for (int tmi = 0; tmi < 2; ++tmi) {
            int mode = tmi ? -1 : 1;
            std::size_t n_rows = 0, n = 0;
            gpu_layout_elements(c.rows, c.L, &n_rows, &n);
            std::vector<int16_t> g(n + 8, 9);
            std::vector<uint64_t> words(c.rows + 1, 9);
            s = write_gpu_block(in, mode, g.data(), n, nullptr, 0, words.data(), c.rows, &tm);
            CHECK(s == Status::kOk, "gpu %s", status_code(s));
            CHECK(g[n] == 9 && words[c.rows] == 9, "gpu guard");
            // Expected tile layout from the expected merged record.
            bool ok = true;
            for (std::size_t r = 0; r < n_rows; ++r)
                for (std::size_t a = 0; a < c.L; ++a)
                    for (int x = 0; x < 3; ++x) {
                        int16_t want = r < c.rows ? emerged[(r * 3 + x) * c.L + a] : 0;
                        std::size_t t = r / 32, l = r % 32;
                        std::size_t idx = mode == 1 ? ((t * c.L + a) * 32 + l) * 3 + x
                                                    : ((t * c.L + a) * 3 + x) * 32 + l;
                        ok = ok && g[idx] == want;
                    }
            CHECK(ok, "gpu layout trial %d mode %d isa %s", trial, mode, isa_name(isa));
            for (std::size_t r = 0; r < c.rows; ++r) {
                uint64_t want = 0; unsigned ncol = 0;
                for (std::size_t t = 0; t < nc; ++t) want |= uint64_t(ecounts[r * nc + t]) << (8 * t);
                for (std::size_t t = 1; t < nc; ++t) ncol += ecounts[r * nc + t];
                want |= uint64_t(ncol) << 56;
                CHECK(words[r] == want, "count word row %zu", r);
            }
            g.resize(n);
            if (ref_gpu[tmi].empty()) ref_gpu[tmi] = g; else CHECK(ref_gpu[tmi] == g, "gpu ISA differ");
        }
        std::vector<float> f(c.rows * c.L * 3 + 4, -1.0f);
        const float scale = 0.00390625f;
        s = write_cpu_block(in, scale, f.data(), c.rows * c.L * 3, nullptr, 0, nullptr, 0, &tm);
        CHECK(s == Status::kOk, "cpu %s", status_code(s));
        CHECK(f[c.rows * c.L * 3] == -1.0f, "cpu guard");
        bool ok = true;
        for (std::size_t r = 0; r < c.rows; ++r)
            for (std::size_t a = 0; a < c.L; ++a)
                for (int x = 0; x < 3; ++x)
                    ok = ok && f[(r * c.L + a) * 3 + x] == float(emerged[(r * 3 + x) * c.L + a]) * scale;
        CHECK(ok, "cpu layout trial %d", trial);
        f.resize(c.rows * c.L * 3);
        if (ref_cpu.empty()) ref_cpu = f; else CHECK(std::memcmp(ref_cpu.data(), f.data(), f.size() * 4) == 0, "cpu ISA differ");
    }
    force_isa(Isa::kAuto);
}

static void test_writers() {
    for (int trial = 0; trial < 300; ++trial) {
        ColorCase c = random_color_case(1 + rng() % 40, trial % 7, trial % 3 ? 5 : 0);
        check_color_writers(c, trial);
    }
}

// Every arity, signed half ties, near-ties and full-range sums, through
// both tiled layouts and direct CPU output on every compiled ISA. Prefix
// features exercise non-rebased group-wide offsets; row sizes cross every
// relevant vector/tile boundary, including a maximal molecule.
static void test_feature_arities_and_signed_means() {
    const std::size_t row_counts[] = {1, 7, 8, 31, 32, 33, 255, 4096};
    for (int k = 1; k <= 16; ++k) {
        std::vector<int> sums = {-32768 * k, -32768 * k + 1, 32767 * k - 1, 32767 * k, 0};
        for (int q : {-32767, -2, -1, 0, 1, 2, 32766})
            for (int r = -1; r <= 1; ++r) {
                const int v = q * k + k / 2 + r;
                if (v >= -32768 * k && v <= 32767 * k) {
                    sums.push_back(v);
                    if (-v >= -32768 * k && -v <= 32767 * k) sums.push_back(-v);
                }
            }
        for (std::size_t rows : row_counts) {
            ColorCase c;
            c.rows = rows; c.L_heavy = 16; c.L = 24; c.n_types = 1;
            c.mols = {{rows, 16}}; c.runs = runs_of(c.mols);
            c.fo = {2, 3}; c.ft = {1, 1, 1}; c.fao = {0, 1, 2, uint32_t(2 + k)}; c.fa = {0, 0};
            for (int a = 0; a < k; ++a) c.fa.push_back(uint8_t(a));
            c.heavy.assign(rows * 3 * c.L_heavy, 0);
            for (std::size_t row = 0; row < rows; ++row)
                for (int axis = 0; axis < 3; ++axis) {
                    int remaining = sums[(row + std::size_t(axis) * 17) % sums.size()];
                    for (int atom = 0; atom < k; ++atom) {
                        const int value = std::max(-32768, std::min(32767, remaining));
                        c.heavy[(row * 3 + axis) * c.L_heavy + atom] = int16_t(value);
                        remaining -= value;
                    }
                    CHECK(remaining == 0, "directed sum fits k=%d", k);
                }
            std::size_t bytes = 0;
            CHECK(encode_block(c.heavy.data(), c.heavy.size(), RunSlice{c.runs.data(), 1}, rows, 16, nullptr, 0, &bytes) == Status::kOutputTooSmall, "arity encode size probe");
            c.block.resize(bytes);
            CHECK(encode_block(c.heavy.data(), c.heavy.size(), RunSlice{c.runs.data(), 1}, rows, 16, c.block.data(), bytes, &bytes) == Status::kOk, "arity encode");
            check_color_writers(c, 1000 + k);
            std::vector<uint8_t> counts;
            const auto expected = expected_merged(c, &counts);
            for (Isa isa : compiled()) {
                force_isa(isa);
                for (float scale : {0.001f, 1.25f}) {
                    std::vector<float> out(rows * 3 * c.L + 1, -999.f);
                    CHECK(write_cpu_block(c.input(), scale, out.data(), out.size() - 1, nullptr, 0, nullptr, 0, nullptr) == Status::kOk, "direct CPU scale");
                    bool equal = true;
                    for (std::size_t r = 0; r < rows; ++r)
                        for (std::size_t a = 0; a < c.L; ++a)
                            for (int axis = 0; axis < 3; ++axis) {
                                const float want = float(expected[(r * 3 + axis) * c.L + a]) * scale;
                                equal = equal && std::memcmp(&want, &out[(r * c.L + a) * 3 + axis], sizeof(float)) == 0;
                            }
                    CHECK(equal && out.back() == -999.f, "direct CPU scale bits/guard k=%d rows=%zu", k, rows);
                }
            }
        }
    }
    force_isa(Isa::kAuto);
}

static void test_self_overlaps() {
    std::vector<float> vaa(100);
    std::vector<double> cs(100);
    for (int i = 0; i < 100; ++i) { vaa[i] = float(i) + 0.5f; cs[i] = i * 2.0; }
    SelfOverlapInput so{vaa.data(), 100, cs.data(), 100, 40};
    std::vector<float> vo(11, -1); std::vector<double> co(11, -1);
    bool present = false;
    CHECK(bind_self_overlaps(so, 10, vo.data(), 10, co.data(), 10, &present) == Status::kOk && present, "bind");
    CHECK(vo[0] == 40.5f && vo[9] == 49.5f && vo[10] == -1 && co[9] == 98.0 && co[10] == -1, "bind values");
    SelfOverlapInput absent{vaa.data(), 100, nullptr, 0, 90};
    CHECK(bind_self_overlaps(absent, 10, vo.data(), 10, co.data(), 10, &present) == Status::kOk && !present, "absent");
    CHECK(bind_self_overlaps(absent, 11, vo.data(), 11, nullptr, 0, &present) == Status::kSelfOverlapRange, "range");
    CHECK(bind_self_overlaps(so, 10, vo.data(), 9, nullptr, 0, &present) == Status::kOutputTooSmall, "vaa cap");
    CHECK(bind_self_overlaps(so, 10, vo.data(), 10, co.data(), 9, &present) == Status::kOutputTooSmall, "cs cap");
    SelfOverlapInput short_cs{vaa.data(), 100, cs.data(), 45, 40};
    CHECK(bind_self_overlaps(short_cs, 10, vo.data(), 10, co.data(), 10, &present) == Status::kSelfOverlapRange, "cs range");
    SelfOverlapInput huge{vaa.data(), 100, nullptr, 0, SIZE_MAX - 3};
    CHECK(bind_self_overlaps(huge, 10, vo.data(), 10, nullptr, 0, &present) == Status::kSizeOverflow, "overflow");
    CHECK(bind_self_overlaps(so, 0, vo.data(), 10, nullptr, 0, &present) == Status::kRowsZero, "zero rows");
}

// ---- Refusals ----------------------------------------------------------------

static Status heavy_status(const BlockInput& in, std::size_t cap) {
    std::vector<int16_t> out(cap + 64, int16_t(0x3C3C));
    Status s = decode_heavy(in, out.data(), cap);
    for (std::size_t i = cap; i < out.size(); ++i)
        if (out[i] != int16_t(0x3C3C)) { CHECK(false, "write past capacity %zu", cap); break; }
    return s;
}

static Status all_writers_status(const BlockInput& in) {
    // Every writer must agree on the refusal (each runs the same checks).
    std::vector<int16_t> o16(1 << 20); std::vector<float> of(1 << 20);
    std::vector<uint8_t> cnt(1 << 16); std::vector<uint64_t> w(1 << 16);
    Status a = decode_merged(in, o16.data(), o16.size(), cnt.data(), cnt.size(), nullptr);
    Status b = write_gpu_block(in, 1, o16.data(), o16.size(), cnt.data(), cnt.size(), w.data(), w.size(), nullptr);
    Status c = write_cpu_block(in, 0.5f, of.data(), of.size(), cnt.data(), cnt.size(), w.data(), w.size(), nullptr);
    CHECK(a == b && b == c, "writers disagree: %s %s %s", status_code(a), status_code(b), status_code(c));
    return a;
}

static void test_refusals() {
    // A valid 2-molecule base block.
    std::vector<Mol> mols = {{3, 5}, {2, 7}};
    unsigned ws[3] = {9, 3, 12};
    auto xyz = random_coords(mols, 8, ws);
    auto runs = runs_of(mols);
    std::size_t len = 0;
    encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 2}, 5, 8, nullptr, 0, &len);
    std::vector<uint8_t> enc(len);
    encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 2}, 5, 8, enc.data(), len, &len);
    const std::size_t cap = 5 * 3 * 8;
    CHECK(heavy_status(make_input(enc, runs, 5, 8), cap) == Status::kOk, "base ok");

    // Truncated: every length short of valid, and one long.
    for (std::size_t n = 0; n < enc.size(); ++n) {
        std::vector<uint8_t> t(enc.begin(), enc.begin() + n);
        BlockInput in = make_input(t, runs, 5, 8);
        if (n == 0) in.in = nullptr;  // a null pointer with zero length is legal and short
        Status s = heavy_status(in, cap);
        CHECK(s == (n < 32 ? Status::kTruncated : Status::kBlockLength), "truncated %zu -> %s", n, status_code(s));
    }
    {
        std::vector<uint8_t> t = enc; t.push_back(0);
        CHECK(heavy_status(make_input(t, runs, 5, 8), cap) == Status::kBlockLength, "long");
    }
    // Widths 17..255 on each axis.
    for (int a = 0; a < 3; ++a)
        for (unsigned w = 17; w <= 255; ++w) {
            std::vector<uint8_t> t = enc; t[6 + a] = uint8_t(w);
            CHECK(heavy_status(make_input(t, runs, 5, 8), cap) == Status::kBitWidth, "width %u axis %d", w, a);
        }
    // Reserved header bytes.
    for (std::size_t i = 9; i < 32; ++i) {
        std::vector<uint8_t> t = enc; t[i] = 1;
        CHECK(heavy_status(make_input(t, runs, 5, 8), cap) == Status::kBlockHeaderReserved, "reserved %zu", i);
    }
    // Rows vs the run slice.
    CHECK(heavy_status(make_input(enc, runs, 4, 8), cap) == Status::kRowsMismatch, "rows < declared");
    CHECK(heavy_status(make_input(enc, runs, 6, 8), cap) == Status::kRowsMismatch, "rows > declared");
    CHECK(heavy_status(make_input(enc, runs, 0, 8), cap) == Status::kRowsZero, "zero rows");
    CHECK(all_writers_status(make_input(enc, runs, 0, 8)) == Status::kRowsZero, "zero rows writers");
    {
        std::vector<uint8_t> none;
        CHECK(heavy_status(make_input(enc, none, 5, 8), cap) == Status::kMoleculesZero, "empty run slice");
    }
    // Count > width: n_real > L_heavy.
    CHECK(heavy_status(make_input(enc, runs, 5, 6), cap) == Status::kRowExceedsWidth, "n_real > L_heavy");
    {
        std::vector<uint8_t> r = runs; r[8] = 0; r[9] = 0;  // n_conf 0
        CHECK(heavy_status(make_input(enc, r, 5, 8), cap) == Status::kNConf, "n_conf 0");
        r = runs; r[10] = 0;  // n_real 0
        CHECK(heavy_status(make_input(enc, r, 5, 8), cap) == Status::kNReal, "n_real 0");
    }
    // Shape-argument limits.
    CHECK(heavy_status(make_input(enc, runs, 5, 0), cap) == Status::kLHeavy, "L_heavy 0");
    CHECK(heavy_status(make_input(enc, runs, 5, 257), cap) == Status::kLHeavy, "L_heavy 257");
    CHECK(all_writers_status(make_input(enc, runs, 5, 8, 513)) == Status::kL, "L 513");
    {
        BlockInput in = make_input(enc, runs, 5, 8, 8); in.L = 0;
        CHECK(all_writers_status(in) == Status::kL, "L 0");
    }
    CHECK(all_writers_status(make_input(enc, runs, 5, 8, 8, 7)) == Status::kNTypes, "n_types 7");
    CHECK(heavy_status(make_input(enc, runs, 4097, 8), cap) == Status::kBlockRows, "rows 4097 with 2 molecules");
    {
        std::vector<Mol> one = {{1, 1}};
        auto r1 = runs_of(one);
        CHECK(heavy_status(make_input(enc, r1, 4097, 8), cap) == Status::kBlockRows, "rows 4097 single");
    }
    // Output capacity one element short (and zero).
    CHECK(heavy_status(make_input(enc, runs, 5, 8), cap - 1) == Status::kOutputTooSmall, "cap - 1");
    CHECK(heavy_status(make_input(enc, runs, 5, 8), 0) == Status::kOutputTooSmall, "cap 0");
    // Overflowing size products.
    {
        std::size_t n = 0, nr = 0;
        CHECK(block_length(SIZE_MAX, 16, 16, 16, &n) == Status::kSizeOverflow, "block_length overflow");
        CHECK(block_length(SIZE_MAX / 2, 16, 16, 16, &n) == Status::kSizeOverflow, "block_length overflow 2");
        CHECK(block_length(5, 17, 0, 0, &n) == Status::kBitWidth, "block_length width");
        CHECK(block_length(5, 1, 2, 3, nullptr) == Status::kBadArgument, "block_length null");
        CHECK(gpu_layout_elements(SIZE_MAX, 512, &nr, &n) == Status::kSizeOverflow, "gpu elements overflow");
        CHECK(gpu_layout_elements(SIZE_MAX / 600, 512, &nr, &n) == Status::kSizeOverflow, "gpu elements overflow 2");
        CHECK(gpu_layout_elements(33, 24, &nr, &n) == Status::kOk && nr == 64 && n == 64 * 24 * 3, "gpu elements");
    }
    // Null pointers with nonzero lengths.
    {
        BlockInput in = make_input(enc, runs, 5, 8); in.in = nullptr;
        CHECK(heavy_status(in, cap) == Status::kBadArgument, "null in");
        in = make_input(enc, runs, 5, 8); in.runs.records = nullptr;
        CHECK(heavy_status(in, cap) == Status::kBadArgument, "null runs");
        CHECK(decode_heavy(make_input(enc, runs, 5, 8), nullptr, cap) == Status::kBadArgument, "null out");
    }
    // Writers: tile mode, scale, capacities one short.
    {
        BlockInput in = make_input(enc, runs, 5, 8, 8);
        std::vector<int16_t> g(32 * 8 * 3 + 64, 0x11);
        std::vector<float> f(5 * 8 * 3 + 64, 1.0f);
        std::vector<uint8_t> cnt(5 + 64, 0x22); std::vector<uint64_t> w(5 + 64, 0x33);
        CHECK(write_gpu_block(in, 0, g.data(), 768, nullptr, 0, nullptr, 0, nullptr) == Status::kBadArgument, "tile 0");
        CHECK(write_gpu_block(in, 2, g.data(), 768, nullptr, 0, nullptr, 0, nullptr) == Status::kBadArgument, "tile 2");
        CHECK(write_gpu_block(in, 1, g.data(), 767, nullptr, 0, nullptr, 0, nullptr) == Status::kOutputTooSmall, "gpu cap - 1");
        CHECK(write_gpu_block(in, 1, g.data(), 768, cnt.data(), 4, nullptr, 0, nullptr) == Status::kOutputTooSmall, "counts cap - 1");
        CHECK(write_gpu_block(in, 1, g.data(), 768, nullptr, 0, w.data(), 4, nullptr) == Status::kOutputTooSmall, "words cap - 1");
        for (std::size_t i = 768; i < g.size(); ++i) CHECK(g[i] == 0x11, "gpu guard");
        for (std::size_t i = 4; i < cnt.size(); ++i) CHECK(cnt[i] == 0x22, "count guard");
        for (std::size_t i = 4; i < w.size(); ++i) CHECK(w[i] == 0x33, "word guard");
        CHECK(write_cpu_block(in, 0.5f, f.data(), 119, nullptr, 0, nullptr, 0, nullptr) == Status::kOutputTooSmall, "cpu cap - 1");
        for (std::size_t i = 119; i < f.size(); ++i) CHECK(f[i] == 1.0f, "cpu guard");
        std::vector<int16_t> m(120 + 8, 0x44);
        CHECK(decode_merged(in, m.data(), 119, nullptr, 0, nullptr) == Status::kOutputTooSmall, "merged cap - 1");
        for (std::size_t i = 119; i < m.size(); ++i) CHECK(m[i] == 0x44, "merged guard");
        CHECK(write_cpu_block(in, 0.0f, f.data(), 120, nullptr, 0, nullptr, 0, nullptr) == Status::kBadArgument, "scale 0");
        CHECK(write_cpu_block(in, -1.0f, f.data(), 120, nullptr, 0, nullptr, 0, nullptr) == Status::kBadArgument, "scale < 0");
        CHECK(write_cpu_block(in, std::numeric_limits<float>::quiet_NaN(), f.data(), 120, nullptr, 0, nullptr, 0, nullptr) == Status::kBadArgument, "scale nan");
        CHECK(write_cpu_block(in, std::numeric_limits<float>::infinity(), f.data(), 120, nullptr, 0, nullptr, 0, nullptr) == Status::kBadArgument, "scale inf");
    }
    // Feature-table refusals, on a color block.
    {
        ColorCase base;
        // Two molecules, n_real 5 and 7; features: m0 {(1,[0,4]), (2,[3])}, m1 {(1,[6])}.
        base.mols = mols; base.runs = runs; base.block = enc; base.heavy = xyz;
        base.rows = 5; base.L_heavy = 8; base.L = 16; base.n_types = 2;
        base.fo = {0, 2, 3};
        base.ft = {1, 2, 1};
        base.fao = {0, 2, 3, 4};
        base.fa = {0, 4, 3, 6};
        CHECK(all_writers_status(base.input()) == Status::kOk, "color base ok");
        auto bad = [&](auto mutate, Status want, const char* what) {
            ColorCase c = base; mutate(c);
            Status s = all_writers_status(c.input());
            CHECK(s == want, "%s -> %s (want %s)", what, status_code(s), status_code(want));
        };
        bad([](ColorCase& c) { c.fa[3] = 7; }, Status::kFeatureAtomIndex, "atom index == n_real");
        bad([](ColorCase& c) { c.fa[1] = 5; }, Status::kFeatureAtomIndex, "atom index == n_real m0");
        bad([](ColorCase& c) { c.fa[3] = 255; }, Status::kFeatureAtomIndex, "atom index 255");
        bad([](ColorCase& c) { c.ft[1] = 3; }, Status::kFeatureType, "type > n_types");
        bad([](ColorCase& c) { c.ft[1] = 0; }, Status::kFeatureType, "type 0");
        bad([](ColorCase& c) { c.fo = {0, 2}; }, Status::kFeatureOffsets, "offsets slice short");
        bad([](ColorCase& c) { c.fo = {0, 2, 4}; }, Status::kFeatureOffsets, "offset past types");
        bad([](ColorCase& c) { c.fo = {2, 1, 3}; }, Status::kFeatureOffsets, "offsets not monotone");
        bad([](ColorCase& c) { c.fao = {0, 2, 3}; }, Status::kFeatureOffsets, "atom offsets slice short");
        bad([](ColorCase& c) { c.fao = {0, 2, 3, 5}; }, Status::kFeatureOffsets, "atom offset past atoms");
        bad([](ColorCase& c) { c.fao = {0, 2, 2, 4}; }, Status::kFeatureOffsets, "feature with zero atoms");
        bad([](ColorCase& c) { c.fao = {0, 3, 2, 4}; }, Status::kFeatureOffsets, "atom offsets decreasing");
        bad([](ColorCase& c) { c.L = 7; }, Status::kRowExceedsWidth, "n_real + n_color > L");
        bad([](ColorCase& c) { c.n_types = 0; }, Status::kFeaturesMissing, "features with n_types 0");
        bad([](ColorCase& c) { c.fo.clear(); }, Status::kFeaturesMissing, "n_types without features");
        // 17 atoms in one feature (n_real raised so indices are legal).
        {
            std::vector<Mol> m2 = {{1, 20}};
            unsigned w2[3] = {4, 4, 4};
            auto x2 = random_coords(m2, 24, w2);
            auto r2 = runs_of(m2);
            std::size_t l2 = 0;
            encode_block(x2.data(), x2.size(), RunSlice{r2.data(), 1}, 1, 24, nullptr, 0, &l2);
            std::vector<uint8_t> e2(l2);
            encode_block(x2.data(), x2.size(), RunSlice{r2.data(), 1}, 1, 24, e2.data(), l2, &l2);
            ColorCase c; c.mols = m2; c.runs = r2; c.block = e2; c.rows = 1; c.L_heavy = 24; c.L = 32; c.n_types = 1;
            c.fo = {0, 1}; c.ft = {1}; c.fao = {0, 17};
            for (int i = 0; i < 17; ++i) c.fa.push_back(uint8_t(i));
            CHECK(all_writers_status(c.input()) == Status::kAtomsPerFeature, "17 atoms");
            c.fao = {0, 16}; c.fa.resize(16);
            CHECK(all_writers_status(c.input()) == Status::kOk, "16 atoms ok");
            // 256 features in one molecule.
            c.L = 512; c.fo = {0, 256}; c.ft.assign(256, 1); c.fao.clear(); c.fa.clear();
            for (int i = 0; i < 256; ++i) { c.fao.push_back(uint32_t(i)); c.fa.push_back(0); }
            c.fao.push_back(256);
            CHECK(all_writers_status(c.input()) == Status::kFeaturesPerMolecule, "256 features");
            c.fo = {0, 255}; c.ft.resize(255); c.fao.resize(256); c.fa.resize(255);
            CHECK(all_writers_status(c.input()) == Status::kOk, "255 features ok");
        }
    }
    // Forced ISA that is not compiled.
    for (Isa isa : {Isa::kAvx2, Isa::kNeon})
        if (!isa_compiled(isa)) CHECK(force_isa(isa) == Status::kIsaUnavailable, "force %s", isa_name(isa));
    CHECK(force_isa(Isa::kAuto) == Status::kOk, "auto");
    // Encoder refusals.
    {
        std::size_t n = 0;
        std::vector<uint8_t> out(len);
        CHECK(encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 2}, 5, 8, out.data(), len - 1, &n) == Status::kOutputTooSmall && n == len, "encode cap - 1");
        CHECK(encode_block(xyz.data(), xyz.size() - 1, RunSlice{runs.data(), 2}, 5, 8, out.data(), len, &n) == Status::kBadArgument, "encode coords short");
        CHECK(encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 2}, 0, 8, out.data(), len, &n) == Status::kRowsZero, "encode rows 0");
        CHECK(encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 2}, 4, 8, out.data(), len, &n) == Status::kRowsMismatch, "encode rows");
        CHECK(encode_block(xyz.data(), xyz.size(), RunSlice{runs.data(), 2}, 5, 6, out.data(), len, &n) == Status::kRowExceedsWidth, "encode n_real > L");
    }
    // status_code / status_message never null, for every enumerator.
    for (int i = 0; i <= 30; ++i) {
        CHECK(status_code(Status(i)) != nullptr && status_message(Status(i)) != nullptr, "status %d", i);
    }
}

// ---- Case files (driven from Python) --------------------------------------
// Layout, little-endian u64 fields: magic "R3GCASE1", rows, L_heavy, L,
// n_types, has_features, len(block), len(runs), len(offsets) [u32 count],
// len(types), len(atom_offsets) [u32 count], len(atoms), then the arrays.

static bool read_file(const std::string& p, std::vector<uint8_t>* d) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    d->clear();
    uint8_t buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) d->insert(d->end(), buf, buf + n);
    std::fclose(f);
    return true;
}
static void write_file(const std::string& p, const void* d, std::size_t n) {
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) { CHECK(false, "cannot write %s", p.c_str()); return; }
    if (n) std::fwrite(d, 1, n, f);
    std::fclose(f);
}

static int run_case(const std::string& dir, const std::string& name) {
    std::vector<uint8_t> d;
    if (!read_file(dir + "/" + name, &d) || d.size() < 8 + 11 * 8 || std::memcmp(d.data(), "R3GCASE1", 8) != 0) {
        CHECK(false, "bad case file %s", name.c_str());
        return 1;
    }
    uint64_t h[11];
    std::memcpy(h, d.data() + 8, sizeof h);
    std::size_t pos = 8 + sizeof h;
    auto take = [&](std::size_t bytes, std::vector<uint8_t>* out) {
        out->assign(d.begin() + pos, d.begin() + pos + bytes);
        pos += bytes;
    };
    std::vector<uint8_t> block, runs, fo8, ft, fao8, fa;
    take(h[5], &block); take(h[6], &runs); take(h[7] * 4, &fo8); take(h[8], &ft); take(h[9] * 4, &fao8); take(h[10], &fa);
    // Copy u32 arrays into aligned storage of their own (the core does not
    // need it; this keeps the fixture honest about what it hands over).
    std::vector<uint32_t> fo(h[7]), fao(h[9]);
    if (h[7]) std::memcpy(fo.data(), fo8.data(), h[7] * 4);
    if (h[9]) std::memcpy(fao.data(), fao8.data(), h[9] * 4);
    BlockInput in{};
    in.in = block.empty() ? nullptr : block.data(); in.in_len = block.size();
    in.runs = RunSlice{runs.empty() ? nullptr : runs.data(), runs.size() / 16};
    in.rows = h[0]; in.L_heavy = h[1]; in.L = h[2]; in.n_types = h[3];
    if (h[4]) in.features = FeatureSlice{fo.data(), fo.size(), ft.data(), ft.size(), fao.data(), fao.size(), fa.data(), fa.size()};
    const std::string base = dir + "/" + name;
    // Output sizes are bounded here so a hostile header cannot make the
    // fixture allocate gigabytes: the core refuses anything larger anyway.
    auto bounded = [](std::size_t a, std::size_t b, std::size_t c) -> std::size_t {
        std::size_t r;
        if (__builtin_mul_overflow(a, b, &r) || __builtin_mul_overflow(r, c, &r) || r > (std::size_t(1) << 28)) return 0;
        return r;
    };
    std::size_t n_heavy = bounded(in.rows, 3, in.L_heavy);
    std::size_t n_merged = bounded(in.rows, 3, in.L);
    std::size_t gr = 0, n_gpu = 0;
    if (gpu_layout_elements(in.rows, in.L, &gr, &n_gpu) != Status::kOk || n_gpu > (std::size_t(1) << 28)) n_gpu = 0;
    std::size_t n_counts = bounded(in.rows, in.n_types + 1, 1);
    std::string statuses;
    for (Isa isa : compiled()) {
        force_isa(isa);
        const std::string p = base + "." + isa_name(isa);
        std::vector<int16_t> o16(std::max<std::size_t>({n_heavy, n_merged, n_gpu, 1}));
        std::vector<float> of(std::max<std::size_t>(n_merged, 1));
        std::vector<uint8_t> cnt(std::max<std::size_t>(n_counts, 1));
        std::vector<uint64_t> w(std::max<std::size_t>(in.rows <= (1u << 24) ? in.rows : 0, 1));
        Status s = decode_heavy(in, o16.data(), n_heavy);
        statuses += std::string(isa_name(isa)) + " heavy " + status_code(s) + "\n";
        if (s == Status::kOk) write_file(p + ".heavy", o16.data(), n_heavy * 2);
        s = decode_merged(in, o16.data(), n_merged, cnt.data(), n_counts, nullptr);
        statuses += std::string(isa_name(isa)) + " merged " + status_code(s) + "\n";
        if (s == Status::kOk) { write_file(p + ".merged", o16.data(), n_merged * 2); write_file(p + ".counts", cnt.data(), n_counts); }
        for (int mode : {1, -1}) {
            s = write_gpu_block(in, mode, o16.data(), n_gpu, cnt.data(), n_counts, w.data(), in.rows, nullptr);
            statuses += std::string(isa_name(isa)) + (mode == 1 ? " gpu1 " : " gpu-1 ") + status_code(s) + "\n";
            if (s == Status::kOk) {
                write_file(p + (mode == 1 ? ".gpu1" : ".gpu-1"), o16.data(), n_gpu * 2);
                write_file(p + ".words", w.data(), in.rows * 8);
            }
        }
        s = write_cpu_block(in, 0.00390625f, of.data(), n_merged, nullptr, 0, nullptr, 0, nullptr);
        statuses += std::string(isa_name(isa)) + " cpu " + status_code(s) + "\n";
        if (s == Status::kOk) write_file(p + ".cpu", of.data(), n_merged * 4);
    }
    force_isa(Isa::kAuto);
    write_file(base + ".status", statuses.data(), statuses.size());
    return 0;
}

static void run_cases(const std::string& dir) {
    DIR* dp = opendir(dir.c_str());
    if (!dp) { CHECK(false, "no case dir %s", dir.c_str()); return; }
    std::vector<std::string> names;
    while (dirent* e = readdir(dp)) {
        std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".case") == 0) names.push_back(n);
    }
    closedir(dp);
    std::sort(names.begin(), names.end());
    for (auto& n : names) run_case(dir, n);
    std::printf("cases: %zu\n", names.size());
}

// Defence at the gather use site even if a C++ caller violates no-alias.
static void test_feature_offset_double_fetch() {
    std::vector<uint8_t> block(32, 0), runs(16, 0);
    runs[8] = 2; runs[10] = 4;
    uint32_t offsets[] = {0, 2};
    uint8_t types[] = {1, 1}, atoms[] = {0, 1};
    alignas(uint64_t) uint32_t atom_offsets[] = {0, 1, 2, 2};
    BlockInput b = make_input(block, runs, 2, 8, 8, 1);
    b.features = FeatureSlice{offsets, 2, types, 2, atom_offsets, 4, atoms, 2};
    std::vector<float> out(2 * 8 * 3);
    BlockTimings tm;
    const Status status = write_cpu_block(b, 1.0f, out.data(), out.size(), nullptr, 0,
        reinterpret_cast<uint64_t*>(atom_offsets), 2, &tm);
    CHECK(status == Status::kFeatureOffsets, "double fetch must refuse before an over-read");
}

int main(int argc, char** argv) {
    std::printf("compiled ISAs:");
    for (Isa i : compiled()) std::printf(" %s", isa_name(i));
    std::printf("\n");
    test_widths_and_tails();
    test_n_real_and_widths();
    test_wrap();
    CHECK(builder_mode() == BuilderMode::kAuto, "builder starts in auto");
    for (BuilderMode mode : {BuilderMode::kBaseline, BuilderMode::kFused, BuilderMode::kAuto}) {
        CHECK(set_builder_mode(mode) == Status::kOk, "select builder");
        CHECK(builder_mode() == mode, "builder override retained");
        for (Isa isa : compiled()) {
            CHECK(force_isa(isa) == Status::kOk, "builder selection ISA");
            CHECK(use_fused_builder(false) == (mode == BuilderMode::kFused), "CPU default gate");
            CHECK(use_fused_builder(true) == (mode == BuilderMode::kFused ||
                  (mode == BuilderMode::kAuto && isa == Isa::kNeon)), "GPU default gate");
        }
        CHECK(force_isa(Isa::kAuto) == Status::kOk, "restore builder selection ISA");
        // Identical independent oracle, output guards and refusal checks for
        // both writers, every compiled ISA and every feature arity. This also
        // exercises molecule segments crossing tiles under the sanitizers.
        test_writers();
        test_feature_arities_and_signed_means();
        test_refusals();
        test_feature_offset_double_fetch();
    }
    CHECK(set_builder_mode(static_cast<BuilderMode>(99)) == Status::kBadArgument,
          "unknown builder refused");
    CHECK(builder_mode() == BuilderMode::kAuto, "invalid override leaves mode unchanged");
    CHECK(set_builder_mode(BuilderMode::kAuto) == Status::kOk, "restore auto");
    test_self_overlaps();
    if (argc == 3 && std::string(argv[1]) == "--cases") {
        // File-backed conformance/adversarial outputs exercise the new path;
        // the independent built-in oracle above covers both overrides.
        CHECK(set_builder_mode(BuilderMode::kFused) == Status::kOk, "conformance fused");
        run_cases(argv[2]);
        CHECK(set_builder_mode(BuilderMode::kAuto) == Status::kOk, "restore auto");
    }
    std::printf("checks: %d failures: %d\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

// Direct libFuzzer entry point for the format-3 native codec.
//
// Inputs use the compact R3GCASE1 record emitted by
// tests/format3_native_cases.py when available. The record carries one block,
// its run slice, dimensions and optional feature arrays. Arbitrary bytes also
// take a small raw-input path. No corpus validator or Python wrapper runs
// before the codec. Every input is passed through scalar and the host SIMD
// implementation; statuses and successful bytes must agree. The existing
// conformance tests compare scalar results with reference.decode_block,
// including FORMAT.md 5.6's modulo-2^16 wrap.
#include "r3g_codec.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace r3::r3g;

namespace {
constexpr std::size_t kOutputRows = 64;  // keep each fuzz iteration bounded
constexpr std::size_t kMaxArrayElements = 1u << 16;

#if defined(__aarch64__) || defined(_M_ARM64)
constexpr Isa kVectorIsa = Isa::kNeon;
#else
constexpr Isa kVectorIsa = Isa::kAvx2;
#endif

struct Cursor {
    const std::uint8_t* p;
    std::size_t left;

    const std::uint8_t* take(std::size_t requested, std::size_t* actual) {
        const std::size_t n = std::min(requested, left);
        const std::uint8_t* out = p;
        p += n;
        left -= n;
        *actual = n;
        return out;
    }
};

std::uint64_t read_u64_le(const std::uint8_t* p) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(p[i]) << (8 * i);
    return value;
}

std::size_t bounded_size(std::uint64_t n) noexcept {
    return n > std::numeric_limits<std::size_t>::max()
        ? std::numeric_limits<std::size_t>::max() : static_cast<std::size_t>(n);
}

std::vector<std::uint32_t> take_u32(Cursor& c, std::uint64_t requested) {
    const std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(
        requested, kMaxArrayElements));
    const std::size_t n = std::min(want, c.left / sizeof(std::uint32_t));
    std::vector<std::uint32_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* p = c.p + 4 * i;
        out[i] = std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
                 (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    }
    c.p += 4 * n;
    c.left -= 4 * n;
    return out;
}

std::vector<std::uint8_t> take_bytes(Cursor& c, std::uint64_t requested) {
    std::size_t actual = 0;
    const std::uint8_t* p = c.take(bounded_size(requested), &actual);
    return std::vector<std::uint8_t>(p, p + actual);
}

struct Input {
    std::vector<std::uint8_t> block;
    std::vector<std::uint8_t> runs;
    std::vector<std::uint32_t> feature_offsets;
    std::vector<std::uint8_t> feature_types;
    std::vector<std::uint32_t> feature_atom_offsets;
    std::vector<std::uint8_t> feature_atoms;
    std::uint64_t rows = 0;
    std::uint64_t L_heavy = 0;
    std::uint64_t L = 0;
    std::uint64_t n_types = 0;
    bool has_features = false;
};

Input parse_case(const std::uint8_t* data, std::size_t size) {
    Input out;
    if (size >= 96 && std::memcmp(data, "R3GCASE1", 8) == 0) {
        std::uint64_t h[11];
        for (std::size_t i = 0; i < 11; ++i) h[i] = read_u64_le(data + 8 + i * 8);
        out.rows = h[0];
        out.L_heavy = h[1];
        out.L = h[2];
        out.n_types = h[3];
        out.has_features = h[4] != 0;
        Cursor c{data + 96, size - 96};
        out.block = take_bytes(c, h[5]);
        out.runs = take_bytes(c, h[6]);
        out.feature_offsets = take_u32(c, h[7]);
        out.feature_types = take_bytes(c, h[8]);
        out.feature_atom_offsets = take_u32(c, h[9]);
        out.feature_atoms = take_bytes(c, h[10]);
        return out;
    }

    // Generic byte inputs still reach the native decoder without needing a
    // well-formed wrapper. All dimensions are bounded before output allocation.
    if (size < 4) {
        if (size) out.block.assign(data, data + size);
        return out;
    }
    out.rows = data[0] % (kOutputRows + 1);
    out.L_heavy = data[1];
    out.L = data[2] | (std::uint64_t(data[3] & 1) << 8);
    out.n_types = 0;
    const std::size_t payload_start = std::min<std::size_t>(5, size);
    const std::size_t run_count = size > 4 ? data[4] % 5 : 0;
    const std::size_t run_bytes = std::min(run_count * kRunRecordSize, size - payload_start);
    out.runs.assign(data + payload_start, data + payload_start + run_bytes);
    out.block.assign(data + payload_start + run_bytes, data + size);
    return out;
}

struct Outcome {
    Status status;
    std::vector<std::int16_t> output;
};

void trap_if(bool failed) {
    if (failed) __builtin_trap();
}

Outcome decode_heavy_with(Isa isa, const Input& in) {
    const bool small = in.rows <= kOutputRows && in.L_heavy <= kMaxLHeavy;
    const std::size_t count = small
        ? static_cast<std::size_t>(in.rows) * 3 * static_cast<std::size_t>(in.L_heavy)
        : 16;
    Outcome result{Status::kBadArgument, std::vector<std::int16_t>(count, 0x5a5a)};
    const RunSlice runs{in.runs.empty() ? nullptr : in.runs.data(),
                        in.runs.size() / kRunRecordSize};
    BlockInput block{in.block.empty() ? nullptr : in.block.data(), in.block.size(), runs,
                     bounded_size(in.rows), bounded_size(in.L_heavy),
                     bounded_size(in.L), bounded_size(in.n_types), {nullptr, 0, nullptr, 0,
                                                                     nullptr, 0, nullptr, 0}};
    trap_if(force_isa(isa) != Status::kOk);
    result.status = decode_heavy(block, result.output.empty() ? nullptr : result.output.data(),
                                 result.output.size());
    return result;
}

Outcome decode_merged_with(Isa isa, const Input& in) {
    const bool small = in.rows <= kOutputRows && in.L <= kMaxL && in.n_types <= kMaxNTypes;
    const std::size_t n = small
        ? static_cast<std::size_t>(in.rows) * 3 * static_cast<std::size_t>(in.L) : 16;
    const std::size_t nc = small
        ? static_cast<std::size_t>(in.rows) * (static_cast<std::size_t>(in.n_types) + 1) : 16;
    Outcome result{Status::kBadArgument, std::vector<std::int16_t>(n, 0x5a5a)};
    std::vector<std::uint8_t> counts(nc, 0xa5);
    const RunSlice runs{in.runs.empty() ? nullptr : in.runs.data(),
                        in.runs.size() / kRunRecordSize};
    FeatureSlice features{in.feature_offsets.empty() ? nullptr : in.feature_offsets.data(),
                          in.feature_offsets.size(),
                          in.feature_types.empty() ? nullptr : in.feature_types.data(),
                          in.feature_types.size(),
                          in.feature_atom_offsets.empty() ? nullptr : in.feature_atom_offsets.data(),
                          in.feature_atom_offsets.size(),
                          in.feature_atoms.empty() ? nullptr : in.feature_atoms.data(),
                          in.feature_atoms.size()};
    if (!in.has_features) features = {nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0};
    BlockInput block{in.block.empty() ? nullptr : in.block.data(), in.block.size(), runs,
                     bounded_size(in.rows), bounded_size(in.L_heavy), bounded_size(in.L),
                     bounded_size(in.n_types), features};
    BlockTimings timings;
    trap_if(force_isa(isa) != Status::kOk);
    result.status = decode_merged(block, result.output.empty() ? nullptr : result.output.data(),
                                  result.output.size(), counts.empty() ? nullptr : counts.data(),
                                  counts.size(), &timings);
    if (result.status == Status::kOk) {
        // Keep the count output in the same byte stream as coordinates so both
        // native paths must agree on the complete successful result.
        result.output.reserve(result.output.size() + counts.size());
        for (std::uint8_t v : counts) result.output.push_back(static_cast<std::int16_t>(v));
    }
    return result;
}

void compare(const Outcome& scalar, const Outcome& vector) {
    trap_if(scalar.status != vector.status);
    if (scalar.status == Status::kOk) {
        trap_if(scalar.output.size() != vector.output.size());
        trap_if(std::memcmp(scalar.output.data(), vector.output.data(),
                            scalar.output.size() * sizeof(std::int16_t)) != 0);
    }
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    Input in = parse_case(data, size);
    const Outcome scalar_heavy = decode_heavy_with(Isa::kScalar, in);
    const Outcome scalar_merged = decode_merged_with(Isa::kScalar, in);
    if (isa_compiled(kVectorIsa)) {
        const Outcome vector_heavy = decode_heavy_with(kVectorIsa, in);
        const Outcome vector_merged = decode_merged_with(kVectorIsa, in);
        compare(scalar_heavy, vector_heavy);
        compare(scalar_merged, vector_merged);
    }
    force_isa(Isa::kAuto);
    return 0;
}

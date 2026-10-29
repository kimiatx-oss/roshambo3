// New launch writer exercised standalone under ASan/UBSan.
#include "r3g_codec.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
using namespace r3::r3g;

int run(bool color, int mode, std::array<std::size_t, 3> sizes) {
    constexpr std::size_t L = 16;
    std::array<std::array<std::uint8_t, 32>, 3> bytes{};
    std::array<std::array<std::uint8_t, 16>, 3> runs{};
    std::uint32_t fo[] = {0, 2}, ao[] = {0, 1, 3};
    std::uint8_t ft[] = {1, 2}, fa[] = {0, 0, 1};
    std::array<BlockInput, 3> blocks{};
    const FeatureSlice features{fo, 2, ft, 2, ao, 3, fa, 3};
    const std::size_t rows = sizes[0] + sizes[1] + sizes[2], nc = color ? 3 : 1;
    for (std::size_t i = 0; i < 3; ++i) {
        bytes[i][0] = 249; bytes[i][1] = 255;  // -7
        bytes[i][2] = 13;
        bytes[i][4] = 255; bytes[i][5] = 255;  // -1
        runs[i][0] = i + 1;
        runs[i][8] = sizes[i] & 255; runs[i][9] = sizes[i] >> 8;
        runs[i][10] = 8;
        blocks[i] = BlockInput{bytes[i].data(), 32, RunSlice{runs[i].data(), 1},
                               sizes[i], 8, L, color ? 2u : 0u,
                               color ? features : FeatureSlice{}};
    }
    std::size_t nr = 0, n = 0;
    if (gpu_layout_elements(rows, L, &nr, &n) != Status::kOk) return 1;
    std::vector<std::int16_t> out(n, 123);
    std::vector<std::uint8_t> counts(rows * nc);
    std::vector<std::int64_t> ids(rows);
    auto call = [&](std::size_t cap) {
        return write_gpu_launch(blocks.data(), 3, rows, mode, out.data(), cap,
                                counts.data(), counts.size(), ids.data(), ids.size());
    };
    if (call(n - 1) != Status::kOutputTooSmall) return 2;
    bytes[2][6] = 17;
    if (call(n) != Status::kBitWidth || !std::all_of(out.begin(), out.end(), [](auto v) { return v == 123; })) return 3;
    bytes[2][6] = 0;
    runs[2][7] = 128;
    if (call(n) != Status::kBadArgument) return 4;
    runs[2][7] = 0;
    if (call(n) != Status::kOk) return 5;
    for (std::size_t row = 0; row < nr; ++row) {
        if (row < rows) {
            const std::int64_t id = row < sizes[0] ? 1 : row < sizes[0] + sizes[1] ? 2 : 3;
            if (ids[row] != id || counts[row * nc] != 8) return 6;
            if (color && (counts[row * nc + 1] != 1 || counts[row * nc + 2] != 1)) return 7;
        }
        for (std::size_t a = 0; a < L; ++a)
            for (std::size_t c = 0; c < 3; ++c) {
                const std::size_t tile = row / 32, lane = row % 32;
                const std::size_t index = tile * 32 * L * 3 +
                    (mode == 1 ? (a * 32 + lane) * 3 + c : (a * 3 + c) * 32 + lane);
                const std::int16_t value[] = {-7, 13, -1};
                const auto expected = row < rows && a < (color ? 10u : 8u) ? value[c] : 0;
                if (out[index] != expected) return 8;
            }
    }
    return 0;
}
int main() {
    for (Isa isa : {Isa::kScalar, Isa::kAvx2, Isa::kNeon}) {
        if (!isa_compiled(isa)) continue;
        if (force_isa(isa) != Status::kOk) return 9;
        for (bool color : {false, true})
            for (int mode : {1, -1})
                for (auto sizes : {std::array<std::size_t, 3>{7, 31, 3},
                                   std::array<std::size_t, 3>{32, 33, 31},
                                   std::array<std::size_t, 3>{4096, 4096, 1}})
                    if (int result = run(color, mode, sizes)) return result;
    }
    std::puts("GPU launch writer: all checks passed");
}

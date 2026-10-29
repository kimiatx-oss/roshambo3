// Scalar reference for the checked bounded-tile builder contract.
#include "r3g_build.hpp"
#include <cstring>
namespace r3 { namespace r3g { namespace detail {
void build_heavy_tile_scalar(const std::int16_t* axes, std::size_t stride,
    std::size_t pos, std::size_t n_real, std::size_t first, std::size_t lanes,
    std::int16_t* tile) noexcept {
    for (std::size_t slot = 0; slot < n_real; ++slot)
        for (std::size_t ax = 0; ax < 3; ++ax)
            for (std::size_t lane = 0; lane < lanes; ++lane)
                tile[(slot * 3 + ax) * kBuildLanes + first + lane] =
                    axes[ax * stride + pos + lane * n_real + slot];
}
namespace {
template<unsigned K>
void feature(std::int16_t* tile, std::size_t slot, const std::uint8_t* atoms,
             std::size_t first, std::size_t lanes) noexcept {
    for (std::size_t ax = 0; ax < 3; ++ax)
        for (std::size_t lane = first; lane < first + lanes; ++lane) {
            if constexpr (K == 1) {
                tile[(slot * 3 + ax) * kBuildLanes + lane] =
                    tile[(atoms[0] * 3 + ax) * kBuildLanes + lane];
            } else {
                std::int32_t sum = 0;
                for (unsigned a = 0; a < K; ++a)
                    sum += tile[(atoms[a] * 3 + ax) * kBuildLanes + lane];
                tile[(slot * 3 + ax) * kBuildLanes + lane] = builder_mean(sum, K);
            }
        }
}
} // namespace
void build_feature_tile_scalar(std::int16_t* tile, std::size_t slot,
    const std::uint8_t* atoms, unsigned arity, std::size_t first,
    std::size_t lanes) noexcept {
#define R3_ARITY(K) case K: feature<K>(tile, slot, atoms, first, lanes); return
    switch (arity) {
        R3_ARITY(1); R3_ARITY(2); R3_ARITY(3); R3_ARITY(4);
        R3_ARITY(5); R3_ARITY(6); R3_ARITY(7); R3_ARITY(8);
        R3_ARITY(9); R3_ARITY(10); R3_ARITY(11); R3_ARITY(12);
        R3_ARITY(13); R3_ARITY(14); R3_ARITY(15); R3_ARITY(16);
    }
#undef R3_ARITY
}
void emit_gpu_tile_scalar(const std::int16_t* tile, std::size_t L, int mode,
                          std::int16_t* out) noexcept {
    if (mode == -1) {
        if (out != tile) std::memcpy(out, tile, L * 3 * kBuildLanes * sizeof(*out));
        return;
    }
    for (std::size_t slot = 0; slot < L; ++slot)
        for (std::size_t lane = 0; lane < kBuildLanes; ++lane)
            for (std::size_t ax = 0; ax < 3; ++ax)
                out[(slot * kBuildLanes + lane) * 3 + ax] =
                    tile[(slot * 3 + ax) * kBuildLanes + lane];
}
void emit_cpu_tile_scalar(const std::int16_t* tile, std::size_t L,
                         std::size_t lanes, float step, float* out) noexcept {
    for (std::size_t lane = 0; lane < lanes; ++lane)
        for (std::size_t slot = 0; slot < L; ++slot)
            for (std::size_t ax = 0; ax < 3; ++ax)
                out[(lane * L + slot) * 3 + ax] =
                    static_cast<float>(tile[(slot * 3 + ax) * kBuildLanes + lane]) * step;
}
}}} // namespace r3::r3g::detail

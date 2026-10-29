// Internal coordinate-builder microkernels. Inputs have already been validated.
// Tile layout: int16 [slot][axis=3][lane=32], index (slot*3+axis)*32+lane.
// A segment has 0 < lanes <= 32-lane_begin and belongs to one molecule.
// Heavy axes are flat molecule/conformer atom runs, separated by axis_stride;
// the segment's first conformer starts at pos, subsequent rows stride n_real.
// Heavy and feature kernels write only their segment, never padding lanes.
// Caller initializes padding before an emitter reads it. Feature atoms index
// heavy slots, 1<=arity<=16, feature_slot>=n_real. Means are exact signed
// half-away-from-zero int32 sums. No input/output alignment is required.
#pragma once
#include <cstddef>
#include <cstdint>
namespace r3 { namespace r3g { namespace detail {
constexpr std::size_t kBuildLanes = 32;
using CpuTileEmitter = void (*)(const std::int16_t* tile, std::size_t L,
                               std::size_t lanes, float step, float* out) noexcept;
#define R3_BUILD_DECL(ISA) \
void build_heavy_tile_##ISA(const std::int16_t* axes, std::size_t axis_stride, \
    std::size_t pos, std::size_t n_real, std::size_t lane_begin, std::size_t lanes, \
    std::int16_t* tile) noexcept; \
void build_feature_tile_##ISA(std::int16_t* tile, std::size_t feature_slot, \
    const std::uint8_t* atoms, unsigned arity, std::size_t lane_begin, std::size_t lanes) noexcept; \
void emit_gpu_tile_##ISA(const std::int16_t* tile, std::size_t L, int tile_mode, \
    std::int16_t* out) noexcept; \
void emit_cpu_tile_##ISA(const std::int16_t* tile, std::size_t L, std::size_t lanes, \
    float step, float* out) noexcept
// GPU output is [slot][lane][axis] for mode +1; mode -1 equals tile layout
// and permits out==tile. CPU output is [lane][slot][axis], no padded rows.
R3_BUILD_DECL(scalar);
#if defined(__x86_64__) || defined(_M_X64)
R3_BUILD_DECL(avx2);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
R3_BUILD_DECL(neon);
// Choose once per writer call. Binary 2^-8/2^-10 variants preserve the scalar
// float(integer)*step bits while using NEON fixed-point conversion.
CpuTileEmitter select_cpu_tile_emitter_neon(float step) noexcept;
#endif
#undef R3_BUILD_DECL
inline std::int16_t builder_mean(std::int32_t sum, unsigned k) noexcept {
    const std::int32_t mag = sum < 0 ? -sum : sum;
    const auto q = (2 * mag + static_cast<std::int32_t>(k)) / (2 * static_cast<std::int32_t>(k));
    return static_cast<std::int16_t>(sum < 0 ? -q : q);
}
}}} // namespace r3::r3g::detail

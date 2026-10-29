// Internal to the format-3 codec: the per-ISA BP256-16 axis unpack kernels.
//
// Each unpacks `groups` whole 256-value packing groups of one axis stream
// (FORMAT.md 5.3) into dst[0, groups * 256), writing int16_wrap(mn + u)
// (5.6). The caller has already checked w <= 16 and that the stream holds
// exactly groups * w * 32 bytes; nonetheless each kernel selects its
// width-specialized body with a `switch` whose default arm returns false
// without reading or writing anything -- a width is never used as an index.
// Reads stay within [s, s + groups * w * 32); writes within
// [dst, dst + groups * 256). No alignment is required of `s`.
#pragma once

#include <cstddef>
#include <cstdint>

namespace r3 {
namespace r3g {
namespace detail {

// The padded heavy layout's row writer: for each run (n_conf rows of n_real
// atoms), each row's three axes are copied from the axis scratch
// (axes + a * stride + pos) into out[row][a][0, n_real) and zero-filled to L.
// The SIMD variants load whole 8- or 16-value chunks of the scratch and mask
// past n_real, so they read up to L - 1 values beyond a row; the caller
// guarantees `stride` leaves at least kMaxL readable values after the last
// one (its zeroed slack). Writes stay inside [out, out + rows * 3 * L).
struct RowRun {
    std::size_t n_conf;
    std::size_t n_real;
};
void heavy_rows_scalar(const std::int16_t* axes, std::size_t stride, const RowRun* runs,
                       std::size_t n_runs, std::size_t L, std::int16_t* out) noexcept;
#if defined(__x86_64__) || defined(_M_X64)
void heavy_rows_avx2(const std::int16_t* axes, std::size_t stride, const RowRun* runs,
                     std::size_t n_runs, std::size_t L, std::int16_t* out) noexcept;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
void heavy_rows_neon(const std::int16_t* axes, std::size_t stride, const RowRun* runs,
                     std::size_t n_runs, std::size_t L, std::int16_t* out) noexcept;
#endif

bool unpack_axis_scalar(const std::uint8_t* s, std::size_t groups, unsigned w,
                        std::int16_t mn, std::int16_t* dst) noexcept;
#if defined(__x86_64__) || defined(_M_X64)
bool unpack_axis_avx2(const std::uint8_t* s, std::size_t groups, unsigned w,
                      std::int16_t mn, std::int16_t* dst) noexcept;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
bool unpack_axis_neon(const std::uint8_t* s, std::size_t groups, unsigned w,
                      std::int16_t mn, std::int16_t* dst) noexcept;
#endif

// int16_wrap(mn + u) of FORMAT.md 5.6: the sum modulo 2^16 reinterpreted as
// two's complement, written without relying on implementation-defined
// narrowing.
inline std::int16_t wrap16(std::int16_t mn, std::uint32_t u) noexcept {
    const std::uint16_t r = static_cast<std::uint16_t>(
        (static_cast<std::uint32_t>(static_cast<std::uint16_t>(mn)) + u) & 0xFFFFu);
    return r < 0x8000u ? static_cast<std::int16_t>(r)
                       : static_cast<std::int16_t>(static_cast<std::int32_t>(r) - 65536);
}

}  // namespace detail
}  // namespace r3g
}  // namespace r3

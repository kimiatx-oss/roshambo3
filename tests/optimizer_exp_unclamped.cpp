// Correctness only (not a microbenchmark): intermediate NaNs/overflow must be
// bitwise cleared before FP consumption. Never changes FP control words.
#include "optimizer_exp.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
    using r3::cpu::OptimizerExpMode;
    using r3::cpu::resolve_optimizer_exp_mode;
    assert(resolve_optimizer_exp_mode(OptimizerExpMode::RawBitsUnclamped, false)
        == OptimizerExpMode::RawBits);
    assert(resolve_optimizer_exp_mode(OptimizerExpMode::RawBitsUnclamped, true)
        == OptimizerExpMode::RawBitsUnclamped);
    for (auto mode : {OptimizerExpMode::Baseline, OptimizerExpMode::RawBits}) {
        assert(resolve_optimizer_exp_mode(mode, false) == mode);
        assert(resolve_optimizer_exp_mode(mode, true) == mode);
    }
    const unsigned before = _mm_getcsr();
    assert(before & _MM_MASK_INVALID);
    const std::array<float, 16> inputs = {0.f, -0.f, -.1f, -1.f,
        -20.f, -79.999f, -80.f, -80.001f, -87.f, -88.f, -89.f,
        -100.f, -177.f, -178.f, -1000.f, -1.e20f};
    for (size_t i = 0; i < inputs.size(); i += 8) {
        const __m256 x = _mm256_loadu_ps(inputs.data() + i);
        alignas(32) uint32_t a[8], b[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(a),
            _mm256_castps_si256(r3::cpu::RawBitsOptimizerExp{}(x)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(b),
            _mm256_castps_si256(r3::cpu::RawBitsUnclampedOptimizerExp{}(x)));
        for (int lane = 0; lane < 8; ++lane) assert(a[lane] == b[lane]);
    }
    const unsigned after = _mm_getcsr();
    assert((before & ~0x3fu) == (after & ~0x3fu));
    std::cout << "16 boundary/tail inputs bitwise equal; MXCSR controls unchanged\n";
}

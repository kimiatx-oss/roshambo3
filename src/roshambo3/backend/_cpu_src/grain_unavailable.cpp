// Explicit unavailable x86 entry points for native non-x86 builds.
#include "grain.hpp"
#include "grain_conformer.hpp"
#include <stdexcept>

namespace r3::cpu {
PairResult run_avx512_grain(bool, Grain, const Conformer&, const Conformer&,
                            float, float, int, int, float, float,
                            const SoaConformer*, OpCensus*) {
    throw std::invalid_argument(
        "AVX-512 grain unavailable on this architecture");
}

void run_avx512_block(bool, const Conformer&, const LaneBlock&, float, int, int,
                      float, float, float, float, int, PairResult*, OpCensus*) {
    throw std::invalid_argument(
        "AVX-512 block unavailable on this architecture");
}
}

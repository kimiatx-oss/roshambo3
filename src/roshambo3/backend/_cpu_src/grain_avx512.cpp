// The width-16 kernel, in its OWN translation unit compiled with `-mavx512f`.
//
// WHY A SEPARATE TU RATHER THAN A TARGET ATTRIBUTE OR PRAGMA
// ----------------------------------------------------------
// Both of the in-file mechanisms were tried and both produced silently wrong
// code. `__attribute__((target("avx512f")))` on the entry point does not reach
// the grain templates it instantiates -- an implicit instantiation is compiled
// at the translation unit's default target, which is `-march=x86-64-v3`, so
// `__m512` values crossed a boundary that target has no ABI for. Moving the
// definition inside a `#pragma GCC target("avx512f")` region did not fix it
// either.
//
// Neither errored. The kernel returned CONSTANTS: 77.1732 for a 5-atom query
// and for a 16-atom query alike, 49.1064 for a 32-atom one -- values with no
// dependence on the input at all, while every AVX-512 primitive passed a
// standalone unit probe (ReduceSum exact; the polynomial at 3.709e-6 max
// relative error against the scalar polynomial's 3.79e-6).
//
// A real translation unit with a real `-mavx512f` on the compiler command line
// has none of this ambiguity: everything in it, template instantiations
// included, is compiled for AVX-512.
//
// **The anchor is untouched by this.** Only this file gets the flag
// (`set_source_files_properties` in CMakeLists), so `score.hpp`'s scalar path
// stays at `-march=x86-64-v3` and keeps producing the numbers its fixture
// names. That is the property the whole arrangement exists to protect, and it
// is checked on every run by `width_gate.py`'s `anchor_reproduces_fixture`.
//
// THAT SENTENCE IS ABOUT THE COMMAND LINE, AND IT LEAVES A GAP.
// `score.hpp`'s scalar ENTRY POINT (`optimize_pair`) is indeed never called
// from this file, so it is untouched in the sense above. But this file
// reaches `grain.hpp` and `grain_conformer.hpp`, which reach `score.hpp` and
// `kernel.hpp` too -- and several plain HELPER functions there
// (`adagrad_exp_step`, the quaternion helpers, `SoaConformer::build`) WERE
// genuinely compiled twice, once per ISA, with the linker keeping exactly one
// COMDAT copy of each under implicit LTO. Being untouched on the command
// line said nothing about which copy that was: it was link-order-determined
// (interface.cpp listed first, CMakeLists.txt), not enforced, and reversing
// that order was enough to make `SoaConformer::build` compile to a genuine
// `vpermt2ps`/zmm body. Fixed by making
// every one of those helpers `static`/anonymous-namespace, so each TU keeps
// its own copy and there is nothing left for the linker to choose between.
//
// Dispatch stays a RUNTIME question -- `__builtin_cpu_supports("avx512f")` in
// interface.cpp -- so one binary serves both machines and width 16 is REFUSED
// rather than silently narrowed where the CPU cannot run it.
#include "grain.hpp"
#include "grain_conformer.hpp"

namespace r3 {
namespace cpu {

PairResult run_avx512_grain(bool poly, Grain gr, const Conformer& a,
                            const Conformer& b, float s_aa, float s_bb,
                            int start_mode, int steps, float lr_q,
                            float lr_t_scaled, const SoaConformer* a_soa,
                            OpCensus* census) {
    return poly ? optimize_pair_grain<simd::Avx512Tag, true>(
                      simd::Avx512Tag{}, gr, a, b, s_aa, s_bb, start_mode,
                      steps, lr_q, lr_t_scaled, a_soa, census)
                : optimize_pair_grain<simd::Avx512Tag, false>(
                      simd::Avx512Tag{}, gr, a, b, s_aa, s_bb, start_mode,
                      steps, lr_q, lr_t_scaled, a_soa, census);
}

void run_avx512_block(bool poly, const Conformer& a, const LaneBlock& blk,
                      float s_aa, int start_mode, int steps, float lr_q,
                      float lr_t_scaled, float cutoff_d2, float converge_tol,
                      int only_start, PairResult* out, OpCensus* census) {
    if (poly)
        optimize_block_conformer<simd::Avx512Tag, true>(
            simd::Avx512Tag{}, a, blk, s_aa, start_mode, steps, lr_q,
            lr_t_scaled, cutoff_d2, converge_tol, only_start, out, census);
    else
        optimize_block_conformer<simd::Avx512Tag, false>(
            simd::Avx512Tag{}, a, blk, s_aa, start_mode, steps, lr_q,
            lr_t_scaled, cutoff_d2, converge_tol, only_start, out, census);
}

}  // namespace cpu
}  // namespace r3

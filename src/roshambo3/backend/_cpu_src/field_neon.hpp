// Native four-channel FP32 field interpolation; public width-eight scheduling
// is two four-atom groups, not an eight-lane NEON vector.
//
// The AArch64 NEON counterpart of field_avx2.hpp: the query shape field
// sampled at each transformed library atom, returning the overlap value and
// the seven-component pose gradient the optimizer consumes.
// The field itself is shared with x86 -- node-major [z][y][x][f,gx,gy,gz]
// FieldNaive32/FieldNaive16 with a zeroed tail (field.hpp) -- and so are the
// conventions this file must not change: the stored gradient already carries
// -2 ln2, so the consumer applies `-gnorm` (field.hpp header and
// `field_gradient`); the lever arm is the world-space `2 M b`, formed before
// translation; samples outside the half-open cell range are zero.
//
// Lane geometry differs from AVX2. A 128-bit NEON register holds exactly one
// node, i.e. the four channels of one texel, so one lerp advances value and
// all three gradient channels at once and there is no cross-half pairing or
// shuffle to port. Independence comes from different
// ATOMS: preparation puts four atoms in the lanes, and stencils are
// scheduled as independent per-atom trees.
//
// The shipped ARM path is the prepared FP32 sampler with proven uint32 base
// offsets, vector torque and paired stencils (field_neon_dataflow.hpp). The
// FP16 entry points are explicit low-level policies only: prepared FP32
// measured ahead of every half policy tested, and public auto selects FP32.
//
// The macros below are private build-time ablation switches, retained so the
// recorded experiments stay reproducible; their defaults are the promoted
// configuration. Scores may be a function of these choices: the native tail
// contraction was accepted as numerical drift gated on recall; the prep,
// address, torque and paired-stencil switches each matched baseline output
// bytes on the frozen all-eight panel.
#pragma once
#include "score.hpp"
#include "field.hpp"
// Matched native screening selects the single-tree full prepared-state loop.
// Explicit private defines retain original/two-tree/pointer-pipeline controls.
// 0: original per-atom loop; 1: two interleaved stencil trees; 2: prepare one
// block ahead into a second buffer; 3: single-tree prepared state (default).
// 3 won; 1 and 2 lost to it.
#ifndef R3_FIELD_NEON_SCHEDULE
#define R3_FIELD_NEON_SCHEDULE 3
#endif
// NEON4 preparation: four atoms' transform, bounds, cell and fraction work in
// lanes; 0 is the scalar loop. +5.89% all-eight geomean.
#ifndef R3_FIELD_NEON_PREP
#define R3_FIELD_NEON_PREP 1
#endif
// One proven uint32 base offset per atom instead of four size_t row offsets,
// when FieldNaive32::prepare_neon_offsets32 succeeds. +11.99% all-eight.
#ifndef R3_FIELD_NEON_ADDRESS
#define R3_FIELD_NEON_ADDRESS 1
#endif
// Cross products for four atoms at once, folded in source order. +1.50%.
#ifndef R3_FIELD_NEON_TORQUE
#define R3_FIELD_NEON_TORQUE 1
#endif
// Two atoms' stencil trees explicitly interleaved. +0.75% all-eight,
// arithmetic byte-identical to the sequential order.
#ifndef R3_NEON_PAIRED_STENCILS
#define R3_NEON_PAIRED_STENCILS 1
#endif
namespace r3::cpu {
// The library conformer transposed to SoA once per pair and reused by every
// start and step -- the NEON copy of field_avx2.hpp's `ConformerSoA`, which
// explains why (gathers from the AoS record lost). Capacity is
// kConformerSoACapacity atoms (score.hpp): callers refuse larger conformers
// before building (shape_field_evaluator.cpp), and build() clamps at it
// exactly as the x86 copy does, so a caller that skipped the check cannot
// read or write past the arrays. Value-initialization
// zeroes the slots past n, though no NEON loop reads them: vector preparation
// loads only whole groups of four real atoms and finishes the tail scalar.
struct ConformerSoA {
    alignas(16) float x[kConformerSoACapacity]{}, y[kConformerSoACapacity]{},
        z[kConformerSoACapacity]{};
    int n = 0, padded = 0;

    void build(const Conformer& b) {
        // Callers refuse larger conformers (interface.cpp,
        // shape_field_evaluator.cpp); clamping `n` as well keeps every reader,
        // which loops to `n`, inside the arrays even if one did not.
        n = b.n_atoms < kConformerSoACapacity ? b.n_atoms
                                              : kConformerSoACapacity;
        padded = (n + 7) & ~7;  // no NEON path reads `padded`
        for (int j = 0; j < n; ++j) {
            x[j] = b.xyz[3 * j];
            y[j] = b.xyz[3 * j + 1];
            z[j] = b.xyz[3 * j + 2];
        }
    }
};

// All gradient entry points share one contract, field.hpp's
// `field_gradient`: (m, t) is the shared FP32 reduced pose, `gnorm` the
// gradient normalization, g[1..3] the torque, g[4..6] the translation
// gradient, and *score (if non-null) the summed field value. g[0] is zero.
// Reductions run in atom source order; each is FP32 throughout.
void field_gradient_neon(const FieldNaive32&, const Conformer&, const float[9],
                         const float[3], float, float[7], float*);
void field_gradient_neon_pipelined(const FieldNaive32&, const ConformerSoA&,
                                   const float[9], const float[3], float,
                                   float[7], float*);
// Value only, for the fresh final SEARCH evaluation that picks the winning
// start: same samples and summation order as the gradient
// path, with the gradient and torque work omitted.
void field_value_neon_pipelined(const FieldNaive32&, const ConformerSoA&,
                                const float[9], const float[3], float*);
// Runtime capability queries for the separately compiled FP16 / FP16FML
// translation units; they forward to cpu_platform (HWCAP on Linux, sysctl on
// macOS). Dispatch happens once at query setup, never per sample.
bool field_neon_fp16_supported() noexcept;
bool field_neon_fp16_arithmetic_supported() noexcept;
bool field_neon_fp16_fml_supported() noexcept;
// H1: FP16 nodes, X lerp by widening FMLSL/FMLAL2, FP32 Y/Z. Passed
// recall, lost to H2 and F0 on speed; closed at the primary stop.
void field_gradient_neon_fp16_fml_pipelined(const FieldNaive16&,
                                            const ConformerSoA&, const float[9],
                                            const float[3], float, float[7],
                                            float*);
// FP32 -> FP16 node quantization (the NEON copy of field.hpp
// `quantize_field`), and the reverse widening used by the F0_Q control:
// half-rounded node values held in FP32 storage, isolating value change from
// runtime conversion cost.
void quantize_field_neon_fp16(const FieldNaive32&, FieldNaive16&);
void widen_field_neon_fp16(const FieldNaive16&, FieldNaive32&);
// H0: FP16 storage, eight widenings per sample, FP32 XYZ arithmetic.
void field_gradient_neon_fp16_pipelined(const FieldNaive16&,
                                        const ConformerSoA&, const float[9],
                                        const float[3], float, float[7],
                                        float*);
// H2: FP16 storage, native-half Y and Z lerps, FP32 X last. A distinct
// numerical contract (half fractions and intermediates), guarded by the
// one-time +-32720 node bound in shape_field_evaluator.cpp.
void field_gradient_neon_fp16_mixed_yz_pipelined(const FieldNaive16&,
                                                 const ConformerSoA&,
                                                 const float[9], const float[3],
                                                 float, float[7], float*);
}

// The single prepared FP32 pose for CPU shape and analytic color.
#pragma once

#include "color_invariant_plan.hpp"

namespace r3 {
namespace cpu {

// A reduced color query is a distinct view type so physical-Angstrom query
// coordinates cannot be passed accidentally to an evaluator expecting A'=sA.
struct ReducedColorConformer {
    ColorConformer features{};
};

// Shared immutable-per-evaluation transform. `matrix` is row-major M=sR and
// `translation` is t'=s*t, both FP32 bytes consumed directly by shape/color.
// `unit_quaternion` is retained only for reporting; no evaluator may rebuild a
// matrix from it. Internal consumers require a complete, finite matrix and
// translation, constructed by prepare_reduced_pose_fp32 (or checked once at
// a raw-pose API boundary). Do not mutate these bytes during evaluation.
// This is a caller precondition, not a runtime marker or per-consumer scan.
struct PreparedReducedPose {
    float matrix[9]{};
    float translation[3]{};
    float unit_quaternion[4]{};
};

// Compose starts before calling this function. It validates and normalizes
// the composed quaternion in FP32 once, constructs M=sR in FP32, and copies
// the optimizer's already reduced translation bytes unchanged.
PreparedReducedPose
prepare_reduced_pose_fp32(const float composed_quaternion[4],
                          const float reduced_translation[3]);

}  // namespace cpu
}  // namespace r3

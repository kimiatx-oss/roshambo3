// Shared-pose construction lives in one TU so every consumer receives the same
// FP32 matrix and translation bytes.
//
// This is the single checked constructor of the shared FP32 reduced-frame
// pose (doc/CPU_SHARED_POSE_CONTRACT.md). Shape, analytic color, the exact
// scalar fallbacks and per-start / final-winner exact scoring all consume
// the `PreparedReducedPose` built here; none may renormalize a quaternion or
// rebuild a matrix of its own. One construction is what makes "the same
// pose" a byte-level statement rather than an approximate one.
#include "reduced_pose.hpp"

#include <cmath>
#include <stdexcept>

#include "kernel.hpp"

namespace r3 {
namespace cpu {

// Build `M = COORD_SCALE * R(q / |q|)` and `t' = reduced_translation` from a
// composed (current state x selected start) quaternion `(w, x, y, z)` and a
// translation that is ALREADY in the reduced frame -- the optimizer's state
// holds `t'`; physical translation is recovered only for the output record.
//
// This is the checked boundary: null inputs, a nonfinite component,
// or an FP32 norm that is zero/nonfinite (underflow or overflow) throw here,
// so evaluators may rely on a finite matrix and translation without
// rescanning them. A positive finite norm bounds every normalized component,
// so the matrix is finite even for a subnormal squared component; this is a
// finiteness guarantee, not exact unit length. See the contract doc, "Trusted
// internal construction and consumers".
PreparedReducedPose
prepare_reduced_pose_fp32(const float composed_quaternion[4],
                          const float reduced_translation[3]) {
    if (composed_quaternion == nullptr || reduced_translation == nullptr)
        throw std::invalid_argument(
            "shared reduced pose inputs are incomplete");
    for (int k = 0; k < 4; ++k)
        if (!std::isfinite(composed_quaternion[k]))
            throw std::invalid_argument(
                "shared reduced pose quaternion must be finite and nonzero");
    for (int k = 0; k < 3; ++k)
        if (!std::isfinite(reduced_translation[k]))
            throw std::invalid_argument(
                "shared reduced pose translation must be finite");

    // Left-to-right sum, parenthesized so the FP32 rounding order is pinned
    // (NUMERICS_POLICY section 2), as is the normalization order.
    const float norm_squared =
        ((composed_quaternion[0] * composed_quaternion[0] +
          composed_quaternion[1] * composed_quaternion[1]) +
         composed_quaternion[2] * composed_quaternion[2]) +
        composed_quaternion[3] * composed_quaternion[3];
    const float norm = std::sqrt(norm_squared);
    if (!(norm > 0.0f) || !std::isfinite(norm))
        throw std::invalid_argument(
            "shared reduced pose quaternion must be finite and nonzero");

    PreparedReducedPose pose;
    for (int k = 0; k < 4; ++k)
        pose.unit_quaternion[k] = composed_quaternion[k] / norm;
    for (int k = 0; k < 3; ++k)
        pose.translation[k] = reduced_translation[k];

    // Standard rotation matrix of a unit quaternion, 3x3 row-major, applied
    // to the LIBRARY molecule (`M*B + t'`); the query is never moved.
    const float w = pose.unit_quaternion[0];
    const float x = pose.unit_quaternion[1];
    const float y = pose.unit_quaternion[2];
    const float z = pose.unit_quaternion[3];
    const float rotation[9] = {
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - w * z),
        2.0f * (x * z + w * y),        2.0f * (x * y + w * z),
        1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - w * x),
        2.0f * (x * z - w * y),        2.0f * (y * z + w * x),
        1.0f - 2.0f * (x * x + y * y)};
    // Fold the reduced-frame scale into the matrix once: library atoms stay
    // physical, query atoms are prepared as `s*A`, so `M*B + t'` lands in the
    // reduced frame with no per-atom multiply (NUMERICS_POLICY section 4).
    for (int k = 0; k < 9; ++k)
        pose.matrix[k] = COORD_SCALE * rotation[k];
    return pose;
}

}  // namespace cpu
}  // namespace r3

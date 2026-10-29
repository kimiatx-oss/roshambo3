// Same physical default model; independently measured arithmetic policy.
#pragma once
#include "kernel.hpp"
#ifndef R3_COLOR_PARAM_FIXED
#define R3_COLOR_PARAM_FIXED 0
#endif
#ifndef R3_COLOR_PARAM_NORMALIZE
#define R3_COLOR_PARAM_NORMALIZE 1
#endif
#ifndef R3_COLOR_PARAM_FUSED
#define R3_COLOR_PARAM_FUSED 1
#endif
namespace r3::cpu {
struct DefaultColorParameters {
    // Match Matrix and ColorInvariantPlan's existing narrowing order exactly.
    static constexpr double physical_coefficient = 1.9687012432153024;
    static constexpr float coefficient = float(physical_coefficient);
    static constexpr double scale_squared =
        double(COORD_SCALE) * double(COORD_SCALE);
    static constexpr float width = float(1.0 / scale_squared);
    static constexpr float exponent_scale = -0.5f * width;
    static constexpr float torque_scale = -2.0f * width;
    static constexpr float translation_scale = -width;
    static constexpr float rawbits_scale = 12102203.0f;
    static constexpr float fused_rawbits_scale = exponent_scale * rawbits_scale;
};
}  // namespace r3::cpu

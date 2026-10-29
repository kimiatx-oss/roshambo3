#pragma once
#include "field_neon.hpp"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>

namespace r3::cpu::detail {
// The original per-atom gradient loop over a half-storage field,
// parameterized by the sampler (H0, H2 or H1). Used only by the
// R3_FIELD_NEON_SCHEDULE=0 control; the default schedule uses gradient_prepared.
// Same contract, conventions and atom-order reduction as field_neon.cpp's
// `gradient`; the sampler is a template parameter, resolved at compile time,
// so there is no per-sample indirect call.
template <class Sampler>
void gradient_fp16(const FieldNaive16& field, const ConformerSoA& library,
                   const float matrix[9], const float translation[3],
                   float gnorm, float gradient[7], float* score,
                   Sampler sampler) {
    std::fill(gradient, gradient + 7, 0.f);
    float32x4_t sum = vdupq_n_f32(0.f);
    float torque[3]{};
    for (int first = 0; first < library.n; first += 8) {
        const int count = std::min(8, library.n - first);
        float mx[8], my[8], mz[8];
        for (int j = 0; j < count; ++j) {
            const int atom = first + j;
            const float bx = library.x[atom], by = library.y[atom],
                        bz = library.z[atom];
            mx[j] = matrix[0] * bx + matrix[1] * by + matrix[2] * bz;
            my[j] = matrix[3] * bx + matrix[4] * by + matrix[5] * bz;
            mz[j] = matrix[6] * bx + matrix[7] * by + matrix[8] * bz;
        }
        for (int j = 0; j < count; ++j) {
            const float32x4_t value =
                sampler(field, mx[j] + translation[0], my[j] + translation[1],
                        mz[j] + translation[2]);
            sum = vaddq_f32(sum, value);
            // No torque from a zero sample or a nonfinite lever arm.
            if (vgetq_lane_f32(value, 1) == 0.f &&
                vgetq_lane_f32(value, 2) == 0.f &&
                vgetq_lane_f32(value, 3) == 0.f)
                continue;
            if (!(std::isfinite(mx[j]) && std::isfinite(my[j]) &&
                  std::isfinite(mz[j])))
                continue;
            const float gx = -gnorm * vgetq_lane_f32(value, 1);
            const float gy = -gnorm * vgetq_lane_f32(value, 2);
            const float gz = -gnorm * vgetq_lane_f32(value, 3);
            torque[0] += 2.f * my[j] * gz - 2.f * mz[j] * gy;
            torque[1] += 2.f * mz[j] * gx - 2.f * mx[j] * gz;
            torque[2] += 2.f * mx[j] * gy - 2.f * my[j] * gx;
        }
    }
    gradient[1] = torque[0];
    gradient[2] = torque[1];
    gradient[3] = torque[2];
    gradient[4] = -gnorm * vgetq_lane_f32(sum, 1);
    gradient[5] = -gnorm * vgetq_lane_f32(sum, 2);
    gradient[6] = -gnorm * vgetq_lane_f32(sum, 3);
    if (score != nullptr) *score = vgetq_lane_f32(sum, 0);
}

}

// Standalone analytic-kernel executables link the shared conformer
// evaluator but do not exercise its optional field path. These minimal
// definitions satisfy that seam; extension integration tests link the real
// color_field.cpp implementation and cover field behavior.
#include "color_field_channels.hpp"

namespace r3 { namespace cpu {

ColorFieldSample ColorFieldBundle::sample_target_type(
        int, const float*) const {
    return {};
}

bool ColorFieldBundle::can_sample_target_type(int, const float*) const {
    return false;
}

int ColorFieldBundle::backend_code(int) const {
    return COLOR_FIELD_ANALYTIC;
}

bool ColorFieldBundle::field_active(int) const {
    return false;
}

}}  // namespace r3::cpu

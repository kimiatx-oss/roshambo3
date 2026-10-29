// Public winner policy; retained compile control for historical comparisons.
#pragma once
#include <cmath>
#include <string>
#include <stdexcept>
#ifndef R3_FINAL_WINNER_ONLY
#define R3_FINAL_WINNER_ONLY 1
#endif
namespace r3 {
namespace cpu {
constexpr bool kWinnerOnlyFinalSelection = R3_FINAL_WINNER_ONLY != 0;

inline bool resolve_final_start_policy(const std::string& policy) {
    if (policy == "default") return kWinnerOnlyFinalSelection;
    if (policy == "winner") return true;
    if (policy == "all_start") return false;
    throw std::invalid_argument(
        "final_start_policy must be winner or all_start");
}

// Strict greater-than preserves the earliest canonical start on ties.
inline bool converged_start_is_better(float objective, float best) {
    if (!std::isfinite(objective))
        throw std::runtime_error("nonfinite final start objective");
    return objective > best;
}
}
}

#pragma once

#include "rolodex/types.hpp"

#include <cmath>
#include <cstddef>

inline float squared_l2(const TVector &a, const TVector &b) {
    float s = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

inline float l2_distance(const TVector &a, const TVector &b) {
    return std::sqrt(squared_l2(a, b));
}

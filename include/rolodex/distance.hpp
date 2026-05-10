#pragma once

#include "rolodex/types.hpp"

#include <cmath>
#include <cstddef>

// Overload for serial: contiguous float pointers — compiler can auto-vectorize.
inline float squared_l2(const float *__restrict a, const float *__restrict b, std::size_t n) {
    float s = 0.0f;
#pragma omp simd reduction(+ : s)
    for (std::size_t i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

inline float l2_distance(const float *a, const float *b, std::size_t n) {
    return std::sqrt(squared_l2(a, b, n));
}

// Overload for OpenMP / MPI: TVector arguments — existing callers unchanged.
inline float squared_l2(const TVector &a, const TVector &b) {
    return squared_l2(a.data(), b.data(), a.size());
}

inline float l2_distance(const TVector &a, const TVector &b) {
    return std::sqrt(squared_l2(a, b));
}

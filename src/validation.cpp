#include "rolodex/validation.hpp"

#include "rolodex/distance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>

namespace rolodex {
namespace validation {
namespace {

bool vectors_close(const TVector &a, const TVector &b, float eps) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

bool vector_in_list(const TVector &cand, const std::vector<TVector> &list, float eps) {
    for (const TVector &v : list) {
        if (vectors_close(cand, v, eps)) {
            return true;
        }
    }
    return false;
}

} // namespace

float recall_at_k(const ValidationPoint &vp, const QueryResult &predicted, int k, float eps) {
    if (k <= 0) {
        return 0.0f;
    }
    const std::size_t gt_k = std::min(static_cast<std::size_t>(k), vp.neighbors.size());
    if (gt_k == 0) {
        return 0.0f;
    }
    int hits = 0;
    for (std::size_t i = 0; i < gt_k; ++i) {
        if (vector_in_list(vp.neighbors[i], predicted.neighbors, eps)) {
            hits++;
        }
    }
    return static_cast<float>(hits) / static_cast<float>(gt_k);
}

void print_miss_diagnostics(const ValidationPoint &vp, const QueryResult &predicted, int k,
                            float eps, std::ostream &err) {
    const std::size_t gt_k = std::min(static_cast<std::size_t>(k), vp.neighbors.size());
    err << "  missed ground-truth (index in GT prefix, expected distance from file):\n";
    for (std::size_t i = 0; i < gt_k; ++i) {
        if (!vector_in_list(vp.neighbors[i], predicted.neighbors, eps)) {
            const float expected_d = (i < vp.distances.size())
                                         ? vp.distances[i]
                                         : std::numeric_limits<float>::quiet_NaN();
            err << "    GT[" << i << "] expected_distance=" << expected_d << '\n';
        }
    }
    err << "  predicted neighbors not in GT prefix (received distance from query):\n";
    for (std::size_t pi = 0; pi < predicted.neighbors.size(); ++pi) {
        const TVector &p = predicted.neighbors[pi];
        bool in_gt_prefix = false;
        for (std::size_t i = 0; i < gt_k; ++i) {
            if (vectors_close(p, vp.neighbors[i], eps)) {
                in_gt_prefix = true;
                break;
            }
        }
        if (!in_gt_prefix) {
            const float recv_d = (pi < predicted.distances.size()) ? predicted.distances[pi]
                                                                   : squared_l2(vp.query, p);
            err << "    received_distance=" << recv_d << '\n';
        }
    }
}

} // namespace validation
} // namespace rolodex

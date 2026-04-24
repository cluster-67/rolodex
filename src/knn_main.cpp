#include "rolodex/dataset.hpp"
#include "rolodex/distance.hpp"
#include "rolodex/kmeans.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

// Tune here; CLI will be added later.
constexpr const char *kDatasetFile = "/pscratch/sd/a/ac3354/data/fashion-mnist-784-euclidean.hdf5";
constexpr int kNumClusters = 10;
constexpr int kTopK = 5;
constexpr int kNprobe = 1;
/** -1 loads all validation queries (clamped in Dataset to file size). */
constexpr int kValidationCount = 10;

constexpr float kVectorMatchEps = 1e-4f;

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

/** Recall@K: first `k` ground-truth neighbors vs predicted list (same length up to k). */
float recall_at_k(const ValidationPoint &vp, const std::vector<TVector> &predicted, int k,
                  float eps) {
    if (k <= 0) {
        return 0.0f;
    }
    const std::size_t gt_k = std::min(static_cast<std::size_t>(k), vp.neighbors.size());
    if (gt_k == 0) {
        return 0.0f;
    }
    int hits = 0;
    for (std::size_t i = 0; i < gt_k; ++i) {
        if (vector_in_list(vp.neighbors[i], predicted, eps)) {
            hits++;
        }
    }
    return static_cast<float>(hits) / static_cast<float>(gt_k);
}

void print_miss_diagnostics(const ValidationPoint &vp, const std::vector<TVector> &predicted, int k,
                            const TVector &query, float eps) {
    const std::size_t gt_k = std::min(static_cast<std::size_t>(k), vp.neighbors.size());
    std::cerr << "  missed ground-truth (index in GT prefix, expected distance from file):\n";
    for (std::size_t i = 0; i < gt_k; ++i) {
        if (!vector_in_list(vp.neighbors[i], predicted, eps)) {
            const float expected_d = (i < vp.distances.size())
                                         ? vp.distances[i]
                                         : std::numeric_limits<float>::quiet_NaN();
            std::cerr << "    GT[" << i << "] expected_distance=" << expected_d << '\n';
        }
    }
    std::cerr << "  predicted neighbors not in GT prefix (squared L2 to query):\n";
    for (const TVector &p : predicted) {
        bool in_gt_prefix = false;
        for (std::size_t i = 0; i < gt_k; ++i) {
            if (vectors_close(p, vp.neighbors[i], eps)) {
                in_gt_prefix = true;
                break;
            }
        }
        if (!in_gt_prefix) {
            std::cerr << "    sq_dist=" << squared_l2(query, p) << '\n';
        }
    }
}

} // namespace

int main() {
    if (kNumClusters <= 0 || kTopK <= 0 || kNprobe <= 0) {
        std::cerr << "kNumClusters, kTopK, and kNprobe must be positive\n";
        return 1;
    }

    Dataset dataset(kDatasetFile);
    dataset.load_dataset();

    SerialKNNAlgorithm knn_algorithm(&dataset, kNumClusters);

    using clock = std::chrono::steady_clock;
    const auto cluster_build_start = clock::now();
    knn_algorithm.create_clusters();
    const auto cluster_build_end = clock::now();
    const double cluster_build_ms =
        std::chrono::duration<double, std::milli>(cluster_build_end - cluster_build_start).count();
    std::cout << "cluster_build_time_ms=" << cluster_build_ms << '\n';

    try {
        dataset.load_validation_dataset(kValidationCount);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load validation dataset: " << e.what() << '\n';
        return 1;
    }

    const std::vector<ValidationPoint> &validation_points = dataset.get_validation_points();
    if (validation_points.empty()) {
        std::cerr << "No validation points loaded\n";
        return 1;
    }

    double query_total_ms = 0.0;
    double query_min_ms = std::numeric_limits<double>::infinity();
    double query_max_ms = 0.0;
    float recall_sum = 0.0f;

    for (std::size_t qi = 0; qi < validation_points.size(); ++qi) {
        const ValidationPoint &vp = validation_points[qi];

        const auto q_start = clock::now();
        const std::vector<TVector> predicted =
            knn_algorithm.query_clusters(vp.query, kTopK, kNprobe);
        const auto q_end = clock::now();
        const double q_ms = std::chrono::duration<double, std::milli>(q_end - q_start).count();

        query_total_ms += q_ms;
        query_min_ms = std::min(query_min_ms, q_ms);
        query_max_ms = std::max(query_max_ms, q_ms);

        const float recall = recall_at_k(vp, predicted, kTopK, kVectorMatchEps);
        recall_sum += recall;

        std::cout << "query=" << qi << " recall@" << kTopK << "=" << recall
                  << " query_time_ms=" << q_ms << '\n';

        if (recall < 1.0f - 1e-6f) {
            std::cerr << "query=" << qi << " recall below 1.0; diagnostics:\n";
            print_miss_diagnostics(vp, predicted, kTopK, vp.query, kVectorMatchEps);
        }
    }

    const double n = static_cast<double>(validation_points.size());
    std::cout << "aggregate: mean_recall@" << kTopK << "=" << (recall_sum / static_cast<float>(n))
              << '\n';
    std::cout << "aggregate: query_time_total_ms=" << query_total_ms
              << " mean_ms=" << (query_total_ms / n) << " min_ms=" << query_min_ms
              << " max_ms=" << query_max_ms << '\n';

    return 0;
}

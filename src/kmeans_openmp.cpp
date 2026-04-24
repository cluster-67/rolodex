#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"

#include <omp.h>
#include <cstdlib>
#include <iostream>

OpenMPKNNAlgorithm::OpenMPKNNAlgorithm(Dataset *dataset, int num_clusters)
    : KNNAlgorithm(dataset, num_clusters) {
    centroids_.resize(static_cast<std::size_t>(num_clusters_));
    membership_.resize(dataset_->get_points().size(), -1);
}

void OpenMPKNNAlgorithm::create_clusters() {
    std::vector<TVector> &points = dataset_->get_points();

    // Step 1: Main thread randomly picks K initial centroids.
    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const int point_idx = static_cast<int>(rand() % points.size());
        centroids_[static_cast<std::size_t>(c_idx)] = points[static_cast<std::size_t>(point_idx)];
        membership_[static_cast<std::size_t>(point_idx)] = c_idx;
    }

    const int max_iters = 10000;
    // Stop when fewer than 0.1% of points change membership in a round.
    const float convergence_threshold = 0.001f;
    int iters = 0;

    while (true) {
        iters++;
        int membership_change_count = 0;

        // Step 2: Each thread computes nearest centroid for its slice of points.
        // No write conflict: each point_idx maps to exactly one membership_ slot,
        // and the loop is partitioned so each slot is touched by exactly one thread.
        #pragma omp parallel for schedule(static) reduction(+:membership_change_count)
        for (std::size_t point_idx = 0; point_idx < points.size(); point_idx++) {
            const int nearest = find_nearest_centroid(points[point_idx]);
            if (membership_[point_idx] != nearest) {
                membership_[point_idx] = nearest;
                membership_change_count++;
            }
        }

        std::cout << "Iteration " << iters << " with " << membership_change_count
                  << " membership changes" << '\n';

        const float change_ratio =
            static_cast<float>(membership_change_count) / static_cast<float>(points.size());

        if (iters >= max_iters || change_ratio < convergence_threshold) {
            break;
        }

        // Steps 3-5: Parallel local accumulation then global reduce.
        update_centroids();
    }
}

void OpenMPKNNAlgorithm::update_centroids() {
    std::vector<TVector> &points = dataset_->get_points();
    if (points.empty()) {
        return;
    }

    const std::size_t dim = points[0].size();
    // Size to max possible threads so the parallel region below can freely use
    // any thread id without bounds issues.
    const int max_threads = omp_get_max_threads();

    // Step 3: Per-thread local accumulators — no shared writes, no locking needed.
    std::vector<std::vector<TVector>> local_sums(
        static_cast<std::size_t>(max_threads),
        std::vector<TVector>(static_cast<std::size_t>(num_clusters_), TVector(dim, 0.0f)));
    std::vector<std::vector<float>> local_counts(
        static_cast<std::size_t>(max_threads),
        std::vector<float>(static_cast<std::size_t>(num_clusters_), 0.0f));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (std::size_t point_idx = 0; point_idx < points.size(); point_idx++) {
            const std::size_t c = static_cast<std::size_t>(membership_[point_idx]);
            for (std::size_t i = 0; i < dim; i++) {
                local_sums[static_cast<std::size_t>(tid)][c][i] += points[point_idx][i];
            }
            local_counts[static_cast<std::size_t>(tid)][c] += 1.0f;
        }
    }

    // Steps 4-5: Reduce all thread-local sums/counts into the global centroid.
    for (std::size_t c = 0; c < static_cast<std::size_t>(num_clusters_); c++) {
        TVector global_sum(dim, 0.0f);
        float global_count = 0.0f;

        for (int t = 0; t < max_threads; t++) {
            const std::size_t ti = static_cast<std::size_t>(t);
            for (std::size_t i = 0; i < dim; i++) {
                global_sum[i] += local_sums[ti][c][i];
            }
            global_count += local_counts[ti][c];
        }

        if (global_count > 0.0f) {
            for (std::size_t i = 0; i < dim; i++) {
                centroids_[c][i] = global_sum[i] / global_count;
            }
        }
    }
}

int OpenMPKNNAlgorithm::find_nearest_centroid(TVector &point) {
    int nearest_centroid_idx = 0;
    float nearest_sq = squared_l2(point, centroids_[0]);
    for (int c_idx = 1; c_idx < num_clusters_; c_idx++) {
        const float d = squared_l2(point, centroids_[static_cast<std::size_t>(c_idx)]);
        if (d < nearest_sq) {
            nearest_sq = d;
            nearest_centroid_idx = c_idx;
        }
    }
    return nearest_centroid_idx;
}

std::vector<TVector> OpenMPKNNAlgorithm::query_clusters(TVector &query, int top_k) {
    (void)top_k;
    const int nearest_centroid_idx = find_nearest_centroid(query);
    std::vector<int> nearest_points_indices = find_nearest_points(nearest_centroid_idx, top_k);
    std::vector<TVector> nearest_points;
    nearest_points.reserve(nearest_points_indices.size());
    std::vector<TVector> &pts = dataset_->get_points();
    for (int idx : nearest_points_indices) {
        nearest_points.push_back(pts[static_cast<std::size_t>(idx)]);
    }
    return nearest_points;
}

std::vector<int> OpenMPKNNAlgorithm::find_nearest_points(int centroid_idx, int top_k) {
    (void)top_k;
    std::vector<int> nearest_points_indices;
    const std::vector<TVector> &pts = dataset_->get_points();
    for (std::size_t i = 0; i < pts.size(); i++) {
        if (membership_[i] == centroid_idx) {
            nearest_points_indices.push_back(static_cast<int>(i));
        }
    }
    return nearest_points_indices;
}

#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <omp.h>

OpenMPKNNAlgorithm::OpenMPKNNAlgorithm(Dataset *dataset, int num_clusters, bool cache_enabled)
    : KNNAlgorithm(dataset, num_clusters, cache_enabled) {
    centroids_.resize(static_cast<std::size_t>(num_clusters_));
    membership_.resize(dataset_->get_points().size(), -1);
}

void OpenMPKNNAlgorithm::create_clusters(int update_frequency) {
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

#pragma omp parallel for schedule(static) reduction(+ : membership_change_count)
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

        if (iters % update_frequency == 0) {
            update_centroids();
        }
    }
}

void OpenMPKNNAlgorithm::update_centroids() {
    std::vector<TVector> &points = dataset_->get_points();
    if (points.empty()) {
        return;
    }

    const std::size_t dim = points[0].size();
    const int max_threads = omp_get_max_threads();

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

int OpenMPKNNAlgorithm::find_nearest_centroid(const TVector &point) const {
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

QueryResult OpenMPKNNAlgorithm::query_clusters(const TVector &query, int top_k, int nprobe) const {
    const std::vector<TVector> &pts = dataset_->get_points();
    if (pts.empty() || top_k <= 0 || num_clusters_ <= 0) {
        return QueryResult{};
    }

    const int nprobe_clamped = std::max(1, std::min(nprobe, num_clusters_));

    std::vector<std::pair<float, int>> centroid_dists;
    centroid_dists.reserve(static_cast<std::size_t>(num_clusters_));
    for (int c = 0; c < num_clusters_; ++c) {
        centroid_dists.emplace_back(squared_l2(query, centroids_[static_cast<std::size_t>(c)]), c);
    }
    std::sort(centroid_dists.begin(), centroid_dists.end(),
              [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                  if (a.first != b.first) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });

    std::vector<std::size_t> candidate_indices;
    for (int p = 0; p < nprobe_clamped; ++p) {
        const int centroid_idx = centroid_dists[static_cast<std::size_t>(p)].second;
        const std::vector<int> from_cluster = find_nearest_points(centroid_idx, top_k);
        for (int idx : from_cluster) {
            candidate_indices.push_back(static_cast<std::size_t>(idx));
        }
    }

    if (candidate_indices.empty()) {
        return QueryResult{};
    }

    std::vector<std::pair<float, std::size_t>> scored;
    scored.reserve(candidate_indices.size());
    for (std::size_t idx : candidate_indices) {
        scored.emplace_back(squared_l2(query, pts[idx]), idx);
    }

    const std::size_t k_out = std::min(static_cast<std::size_t>(top_k), scored.size());
    std::partial_sort(
        scored.begin(), scored.begin() + static_cast<long>(k_out), scored.end(),
        [](const std::pair<float, std::size_t> &a, const std::pair<float, std::size_t> &b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });

    QueryResult result;
    result.neighbors.reserve(k_out);
    result.distances.reserve(k_out);
    for (std::size_t i = 0; i < k_out; ++i) {
        result.neighbors.push_back(pts[scored[i].second]);
        result.distances.push_back(scored[i].first);
    }
    return result;
}

std::vector<int> OpenMPKNNAlgorithm::find_nearest_points(int centroid_idx, int top_k) const {
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

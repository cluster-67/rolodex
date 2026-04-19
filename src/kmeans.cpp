#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"

#include <cstdlib>
#include <iostream>

KNNAlgorithm::KNNAlgorithm(Dataset *dataset, int num_clusters)
    : dataset_(dataset), num_clusters_(num_clusters) {}

SerialKNNAlgorithm::SerialKNNAlgorithm(Dataset *dataset, int num_clusters)
    : KNNAlgorithm(dataset, num_clusters) {
    centroids_.resize(static_cast<std::size_t>(num_clusters_));
    membership_.resize(dataset_->get_points().size(), -1);
}

void SerialKNNAlgorithm::create_clusters() {
    std::vector<TVector> &points = dataset_->get_points();

    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const int point_idx = static_cast<int>(rand() % points.size());
        centroids_[static_cast<std::size_t>(c_idx)] = points[static_cast<std::size_t>(point_idx)];
        membership_[static_cast<std::size_t>(point_idx)] = c_idx;
    }

    int iters = 0;
    while (true) {
        iters++;
        int membership_change_count = 0;
        for (std::size_t point_idx = 0; point_idx < points.size(); point_idx++) {
            int nearest_centroid_idx = find_nearest_centroid(points[point_idx]);
            int prev_membership = membership_[point_idx];
            if (prev_membership != nearest_centroid_idx) {
                membership_[point_idx] = nearest_centroid_idx;
                membership_change_count++;
            }
        }

        std::cout << "Iteration " << iters << " with " << membership_change_count
                  << " membership changes" << '\n';
        if (membership_change_count == 0) {
            break;
        }

        update_centroids();
    }
}

void SerialKNNAlgorithm::update_centroids() {
    std::vector<TVector> &points = dataset_->get_points();
    if (points.empty()) {
        return;
    }

    const std::size_t dim = points[0].size();
    std::vector<TVector> centroid_sums(static_cast<std::size_t>(num_clusters_), TVector(dim, 0.0f));
    std::vector<float> centroid_counts(static_cast<std::size_t>(num_clusters_), 0.0f);

    for (std::size_t point_idx = 0; point_idx < points.size(); point_idx++) {
        const int centroid_idx = membership_[point_idx];
        for (std::size_t i = 0; i < points[point_idx].size(); i++) {
            centroid_sums[static_cast<std::size_t>(centroid_idx)][i] += points[point_idx][i];
        }
        centroid_counts[static_cast<std::size_t>(centroid_idx)] += 1.0f;
    }

    for (int c_idx = 0; c_idx < num_clusters_; c_idx++) {
        const float count = centroid_counts[static_cast<std::size_t>(c_idx)];
        if (count <= 0.0f) {
            continue;
        }
        for (std::size_t i = 0; i < centroid_sums[static_cast<std::size_t>(c_idx)].size(); i++) {
            centroid_sums[static_cast<std::size_t>(c_idx)][i] /= count;
        }
        centroids_[static_cast<std::size_t>(c_idx)] =
            centroid_sums[static_cast<std::size_t>(c_idx)];
    }
}

int SerialKNNAlgorithm::find_nearest_centroid(TVector &point) {
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

std::vector<TVector> SerialKNNAlgorithm::query_clusters(TVector &query, int top_k) {
    (void)top_k;
    int nearest_centroid_idx = find_nearest_centroid(query);
    std::vector<int> nearest_points_indices = find_nearest_points(nearest_centroid_idx, top_k);
    std::vector<TVector> nearest_points;
    nearest_points.reserve(nearest_points_indices.size());
    std::vector<TVector> &pts = dataset_->get_points();
    for (int idx : nearest_points_indices) {
        nearest_points.push_back(pts[static_cast<std::size_t>(idx)]);
    }
    return nearest_points;
}

std::vector<int> SerialKNNAlgorithm::find_nearest_points(int centroid_idx, int top_k) {
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

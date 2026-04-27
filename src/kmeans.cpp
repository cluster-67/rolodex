#include "rolodex/kmeans.hpp"
#include "rolodex/distance.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>

KNNAlgorithm::KNNAlgorithm(Dataset *dataset, int num_clusters)
    : dataset_(dataset), num_clusters_(num_clusters) {}

// ── SerialKNNAlgorithm ────────────────────────────────────────────────────────

SerialKNNAlgorithm::SerialKNNAlgorithm(Dataset *dataset, int num_clusters)
    : KNNAlgorithm(dataset, num_clusters) {
    const std::size_t k   = static_cast<std::size_t>(num_clusters_);
    const std::size_t dim = dataset_->dim();
    centroids_flat_.resize(k * dim, 0.0f);
    centroid_sums_.resize(k * dim, 0.0f);
    centroid_counts_.resize(k, 0.0f);
    membership_.resize(dataset_->n_points(), -1);
}

void SerialKNNAlgorithm::create_clusters() {
    const float*       data = dataset_->get_flat();
    const std::size_t  n    = dataset_->n_points();
    const std::size_t  dim  = dataset_->dim();
    const std::size_t  k    = static_cast<std::size_t>(num_clusters_);

    // Seed centroids from random points
    for (std::size_t c = 0; c < k; c++) {
        const std::size_t idx = static_cast<std::size_t>(rand()) % n;
        membership_[idx] = static_cast<int>(c);
        std::copy(data + idx * dim, data + idx * dim + dim,
                  centroids_flat_.data() + c * dim);
    }

    int iters = 0;
    while (true) {
        iters++;
        int changes = 0;
        for (std::size_t i = 0; i < n; i++) {
            const int nearest = find_nearest_centroid(data + i * dim, dim);
            if (membership_[i] != nearest) {
                membership_[i] = nearest;
                changes++;
            }
        }
        std::cout << "Iteration " << iters << " with " << changes
                  << " membership changes\n";
        if (changes == 0) break;
        update_centroids();
    }
}

void SerialKNNAlgorithm::update_centroids() {
    const float*       data = dataset_->get_flat();
    const std::size_t  n    = dataset_->n_points();
    const std::size_t  dim  = dataset_->dim();
    const std::size_t  k    = static_cast<std::size_t>(num_clusters_);

    // Reuse buffers — zero them in place
    std::fill(centroid_sums_.begin(),   centroid_sums_.end(),   0.0f);
    std::fill(centroid_counts_.begin(), centroid_counts_.end(), 0.0f);

    for (std::size_t i = 0; i < n; i++) {
        const std::size_t c     = static_cast<std::size_t>(membership_[i]);
        const float*      point = data + i * dim;
        float*            csum  = centroid_sums_.data() + c * dim;
        for (std::size_t j = 0; j < dim; j++) {
            csum[j] += point[j];
        }
        centroid_counts_[c] += 1.0f;
    }

    for (std::size_t c = 0; c < k; c++) {
        const float count = centroid_counts_[c];
        if (count <= 0.0f) continue;
        const float* csum = centroid_sums_.data()   + c * dim;
        float*       cen  = centroids_flat_.data()  + c * dim;
        for (std::size_t j = 0; j < dim; j++) {
            cen[j] = csum[j] / count;
        }
    }
}

int SerialKNNAlgorithm::find_nearest_centroid(const float* point, std::size_t dim) {
    int   best_c  = 0;
    float best_sq = squared_l2(point, centroids_flat_.data(), dim);
    for (int c = 1; c < num_clusters_; c++) {
        const float d = squared_l2(point, centroids_flat_.data() + c * dim, dim);
        if (d < best_sq) { best_sq = d; best_c = c; }
    }
    return best_c;
}

std::vector<int> SerialKNNAlgorithm::find_nearest_points(int centroid_idx, int /*top_k*/) {
    std::vector<int> result;
    for (std::size_t i = 0; i < membership_.size(); i++) {
        if (membership_[i] == centroid_idx)
            result.push_back(static_cast<int>(i));
    }
    return result;
}

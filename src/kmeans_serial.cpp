#include "rolodex/kmeans.hpp"

#include "rolodex/distance.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <utility>

SerialKNNAlgorithm::SerialKNNAlgorithm(Dataset *dataset, int num_clusters)
    : KNNAlgorithm(dataset, num_clusters) {
    centroids_.resize(static_cast<std::size_t>(num_clusters_));
    membership_.resize(dataset_->get_points().size(), -1);
}

void SerialKNNAlgorithm::create_clusters(int update_frequency) {
    (void)update_frequency;
    std::vector<TVector> &points = dataset_->get_points();
    if (points.empty() || num_clusters_ <= 0) {
        return;
    }

    if (load_clusters_from_cache()) {
        std::cout << "Loaded cluster cache from " << get_cache_path() << '\n';
        return;
    }

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

    save_clusters_to_cache();
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
        for (float &i : centroid_sums[static_cast<std::size_t>(c_idx)]) {
            i /= count;
        }
        centroids_[static_cast<std::size_t>(c_idx)] =
            centroid_sums[static_cast<std::size_t>(c_idx)];
    }
}

int SerialKNNAlgorithm::find_nearest_centroid(const TVector &point) const {
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

QueryResult SerialKNNAlgorithm::query_clusters(const TVector &query, int top_k, int nprobe) const {
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

std::vector<int> SerialKNNAlgorithm::find_nearest_points(int centroid_idx, int top_k) const {
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

std::string SerialKNNAlgorithm::get_cache_path() const {
    return build_cache_path("serial");
}

bool SerialKNNAlgorithm::load_clusters_from_cache() {
    const std::vector<TVector> &points = dataset_->get_points();
    if (points.empty() || num_clusters_ <= 0) {
        return false;
    }

    const std::size_t num_points = points.size();
    const std::size_t dim = points[0].size();
    const uint64_t ds_sig_expected = dataset_signature();

    std::ifstream in(get_cache_path().c_str(), std::ios::binary);
    if (!in) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    int32_t cached_num_clusters = 0;
    uint64_t cached_num_points = 0;
    uint64_t cached_dim = 0;
    uint64_t cached_dataset_sig = 0;

    in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    in.read(reinterpret_cast<char *>(&cached_num_clusters), sizeof(cached_num_clusters));
    in.read(reinterpret_cast<char *>(&cached_num_points), sizeof(cached_num_points));
    in.read(reinterpret_cast<char *>(&cached_dim), sizeof(cached_dim));
    in.read(reinterpret_cast<char *>(&cached_dataset_sig), sizeof(cached_dataset_sig));
    if (!in.good()) {
        return false;
    }

    const uint32_t kMagic = 0x4B4D4348; // "KMCH"
    const uint32_t kVersion = 2;
    if (magic != kMagic || version != kVersion ||
        cached_num_clusters != static_cast<int32_t>(num_clusters_) ||
        cached_num_points != static_cast<uint64_t>(num_points) ||
        cached_dim != static_cast<uint64_t>(dim) ||
        cached_dataset_sig != ds_sig_expected) {
        return false;
    }

    centroids_.assign(static_cast<std::size_t>(num_clusters_), TVector(dim, 0.0f));
    membership_.assign(num_points, -1);

    for (int c = 0; c < num_clusters_; ++c) {
        in.read(reinterpret_cast<char *>(centroids_[static_cast<std::size_t>(c)].data()),
                static_cast<std::streamsize>(dim * sizeof(float)));
        if (!in.good()) {
            return false;
        }
    }

    std::vector<int32_t> serialized_membership(num_points);
    in.read(reinterpret_cast<char *>(serialized_membership.data()),
            static_cast<std::streamsize>(num_points * sizeof(int32_t)));
    if (!in.good()) {
        return false;
    }

    for (std::size_t i = 0; i < num_points; ++i) {
        membership_[i] = static_cast<int>(serialized_membership[i]);
    }

    for (int m : membership_) {
        if (m < 0 || m >= num_clusters_) {
            return false;
        }
    }

    return true;
}

void SerialKNNAlgorithm::save_clusters_to_cache() const {
    const std::vector<TVector> &points = dataset_->get_points();
    if (points.empty() || num_clusters_ <= 0) {
        return;
    }
    const std::size_t num_points = points.size();
    const std::size_t dim = points[0].size();
    const uint64_t ds_sig = dataset_signature();

    if (!ensure_cache_root_dir()) {
        std::cerr << "Warning: failed to create cluster cache dir '" << cache_root_dir()
                  << "': errno=" << errno << '\n';
        return;
    }

    std::ofstream out(get_cache_path().c_str(), std::ios::binary);
    if (!out) {
        std::cerr << "Warning: failed to open cluster cache file for write: " << get_cache_path()
                  << '\n';
        return;
    }

    const uint32_t kMagic = 0x4B4D4348; // "KMCH"
    const uint32_t kVersion = 2;
    const int32_t saved_num_clusters = static_cast<int32_t>(num_clusters_);
    const uint64_t saved_num_points = static_cast<uint64_t>(num_points);
    const uint64_t saved_dim = static_cast<uint64_t>(dim);

    out.write(reinterpret_cast<const char *>(&kMagic), sizeof(kMagic));
    out.write(reinterpret_cast<const char *>(&kVersion), sizeof(kVersion));
    out.write(reinterpret_cast<const char *>(&saved_num_clusters), sizeof(saved_num_clusters));
    out.write(reinterpret_cast<const char *>(&saved_num_points), sizeof(saved_num_points));
    out.write(reinterpret_cast<const char *>(&saved_dim), sizeof(saved_dim));
    out.write(reinterpret_cast<const char *>(&ds_sig), sizeof(ds_sig));

    for (int c = 0; c < num_clusters_; ++c) {
        out.write(reinterpret_cast<const char *>(centroids_[static_cast<std::size_t>(c)].data()),
                  static_cast<std::streamsize>(dim * sizeof(float)));
    }

    std::vector<int32_t> serialized_membership(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        serialized_membership[i] = static_cast<int32_t>(membership_[i]);
    }
    out.write(reinterpret_cast<const char *>(serialized_membership.data()),
              static_cast<std::streamsize>(num_points * sizeof(int32_t)));

    if (!out.good()) {
        std::cerr << "Warning: failed while writing cluster cache file: " << get_cache_path()
                  << '\n';
        return;
    }
    std::cout << "Saved cluster cache to " << get_cache_path() << '\n';
}
